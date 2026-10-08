"""PPO training for 7鬼523 — a faithful port of ``ppo_multidiscrete_mask.py``.

The reference script in ``ppo-implementation-details/`` targets gym 0.21 +
Python <3.10 (and a MicroRTS CNN), so it cannot run here unmodified.  This
module keeps its algorithm and implementation details, swapping in gymnasium,
a flat-observation MLP, and this project's environment.  See
``docs/adr/0003-training-stack.md``.

Typical runs::

    uv sync --group train
    uv run --group train 7g523-train --total-timesteps 1000000          # vs RandomBot
    uv run --group train 7g523-train --opponent self --load-checkpoint runs/<new-run>/agent.pt
    uv run --group train 7g523-eval --checkpoint runs/<new-run>/agent.pt --episodes 500
    uv run --group train tensorboard --logdir runs --port 6006          # 看曲线
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim

from .actions import nvec_for
from .env import Seven523Env, observation_dim, validate_reward_cap
from .eval import evaluate
from .history import (
    EVENT_DIM,
    EVENT_ORDERINGS,
    ORDERINGS,
    PASS_MODES,
    WENT_OUT_MODES,
    EventHistoryWrapper,
    HistorySequenceWrapper,
)
from .league import LeagueConfig, PfspController, build_league, parse_pool_member
from .metrics import MetricsLogger, TensorboardLogger
from .networks import (
    Agent,
    NeuralPolicy,
    WarmStartLayoutError,
    history_layout,
    save_agent,
    warm_start_from,
)
from .policies import Policy, WeightedPolicy
from .ppo import PPOConfig, RolloutBatch, compute_gae, ppo_update
from .rules import Rules

__all__ = [
    "MetricsLogger",
    "TensorboardLogger",
    "annealed_lr",
    "lr_scale",
    "main",
    "make_env",
    "parse_args",
    "parse_pool_member",
    "train",
]


def lr_scale(schedule: str, update: int, num_updates: int) -> float:
    """Learning-rate fraction at 1-based ``update`` (ADR-0003 schedule).

    ``linear`` reproduces the historical ``1 - (update - 1) / num_updates``
    exactly; ``cosine`` is the standard half-cosine from 1 to 0, so both
    schedules start at 1 and end at 0 and only the shape differs.
    """
    if schedule == "cosine":
        return 0.5 * (1.0 + math.cos(math.pi * (update - 1.0) / num_updates))
    if schedule == "linear":
        return 1.0 - (update - 1.0) / num_updates
    raise ValueError(f"unknown lr schedule {schedule!r}")


#: Default absolute LR floor under ``--anneal-lr`` (operator decision
#: 2026-10-07): the annealed schedule never decays below 1e-5.  Pass
#: ``--lr-floor 0`` (or call :func:`annealed_lr` with ``floor=0``) to restore
#: the historical anneal-to-zero behaviour.
_DEFAULT_LR_FLOOR = 1e-5


def annealed_lr(
    schedule: str,
    update: int,
    num_updates: int,
    learning_rate: float,
    floor: float = _DEFAULT_LR_FLOOR,
) -> float:
    """Optimizer LR at 1-based ``update``: the schedule clamped to ``floor``.

    The floor never raises the rate above ``learning_rate`` (so ``floor``
    larger than the peak is capped, and ``learning_rate=0`` stays exactly
    zero for frozen-LR smoke tests).
    """
    scaled = lr_scale(schedule, update, num_updates) * learning_rate
    return max(scaled, min(float(floor), float(learning_rate)))


def _bool(value: str) -> bool:
    return value.lower() in {"true", "1", "yes", "y", "t"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PPO for 7鬼523")
    parser.add_argument("--exp-name", type=str, default="seven523-ppo")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument(
        "--torch-deterministic",
        type=_bool,
        default=True,
        nargs="?",
        const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`",
    )
    parser.add_argument(
        "--cuda",
        type=_bool,
        default=True,
        nargs="?",
        const=True,
        help="if toggled, cuda will be enabled by default",
    )

    # Algorithm specific arguments (names and defaults follow the reference).
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument(
        "--anneal-lr", type=_bool, default=True, nargs="?", const=True
    )
    parser.add_argument(
        "--lr-schedule",
        choices=["linear", "cosine"],
        default="linear",
        help="LR decay shape under --anneal-lr (default linear)",
    )
    parser.add_argument(
        "--lr-floor",
        type=float,
        default=_DEFAULT_LR_FLOOR,
        help=(
            "absolute lower bound of the annealed learning rate (default "
            "1e-5, operator decision 2026-10-07); pass 0 to restore the "
            "historical anneal-to-zero behaviour"
        ),
    )
    parser.add_argument(
        "--optimizer",
        choices=["adam", "adamw"],
        default="adam",
        help="optimizer; adamw enables decoupled --weight-decay",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="decoupled weight decay; only valid with --optimizer adamw",
    )
    parser.add_argument("--gae", type=_bool, default=True, nargs="?", const=True)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument(
        "--norm-adv", type=_bool, default=True, nargs="?", const=True
    )
    parser.add_argument("--clip-coef", type=float, default=0.1)
    parser.add_argument(
        "--clip-vloss", type=_bool, default=True, nargs="?", const=True
    )
    parser.add_argument(
        "--vf-nll",
        type=_bool,
        default=False,
        nargs="?",
        const=True,
        help=(
            "heteroscedastic critic (Amendment A2): the value head predicts "
            "(mean, logvar) and is trained with the Gaussian NLL instead of "
            "the clipped MSE; --clip-vloss does not apply"
        ),
    )
    parser.add_argument(
        "--vf-sample",
        type=_bool,
        default=False,
        nargs="?",
        const=True,
        help=(
            "Amendment A3: draw the rollout value from N(mean, var) instead "
            "of using the mean (requires --vf-nll); evaluation/search keep "
            "using the mean"
        ),
    )
    parser.add_argument(
        "--vf-decoupled",
        type=_bool,
        default=False,
        nargs="?",
        const=True,
        help=(
            "Amendment A4: keep the base clipped-MSE objective on the mean "
            "and train the variance head with the mean detached (requires "
            "--vf-nll)"
        ),
    )
    parser.add_argument(
        "--vf-outcome",
        type=_bool,
        default=False,
        nargs="?",
        const=True,
        help=(
            "outcome-decomposed value: the critic is trained on the jump-free "
            "return and a separate head predicts the discounted win/loss term "
            "E[gamma**(T-t)*seat_outcome]; requires a reward mode with a "
            "discrete outcome jump (arcsin/terminal_win/trick_diff_win/win) "
            "and is incompatible with --vf-nll"
        ),
    )
    parser.add_argument(
        "--outcome-coef",
        type=float,
        default=1.0,
        help=(
            "--vf-outcome only: weight of the outcome head's masked MSE "
            "loss in the total PPO objective"
        ),
    )
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=None)

    # 7鬼523 / run management arguments.
    parser.add_argument("--num-players", type=int, default=2)
    parser.add_argument(
        "--reward-shaping",
        choices=[
            "terminal",
            "trick_diff",
            "win",
            "trick_diff_win",
            "terminal_win",
            "saturate",
            "arcsin",
        ],
        default="arcsin",
        help=(
            "arcsin (default): terminal margin shaped by "
            "(1-α)*margin + α*(2/pi)*arcsin(margin) (steep near 0/100 "
            "points) + win_jump * sign(own - best other) at the 50-point "
            "boundary; terminal: legacy sparse terminal return; trick_diff: "
            "per-step potential difference (telescopes to the terminal "
            "return); win: terminal sign(own - best other); trick_diff_win: "
            "both; terminal_win: terminal return + win_jump * sign(own - best "
            "other); saturate: terminal return, but wins are capped at "
            "--reward-cap tau"
        ),
    )
    parser.add_argument(
        "--win-jump",
        type=float,
        default=1.0,
        help=(
            "terminal_win/arcsin only: the win/loss boundary jump λ in "
            "r = terminal margin + λ * seat_outcome (finite, 0 <= λ <= 10); "
            "ignored by every other mode"
        ),
    )
    parser.add_argument(
        "--arcsin-mix",
        type=float,
        default=0.5,
        help=(
            "arcsin only: blend weight α in r = (1-α)*margin + "
            "α*(2/pi)*arcsin(margin) (finite, 0 <= α <= 1; α=0 is "
            "terminal_win, α=1 is the full arcsin curve); ignored by every "
            "other mode"
        ),
    )
    parser.add_argument(
        "--reward-cap",
        type=float,
        default=None,
        help=(
            "saturate only: the return-unit cap τ on wins (required, finite, "
            "0 <= τ <= 1; τ=0.2 is a 20-point margin); giving it to any other "
            "mode is an error"
        ),
    )
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument(
        "--activation",
        choices=["relu", "tanh", "gelu", "silu"],
        default="relu",
        help="hidden-layer activation (reference ppo.py uses tanh)",
    )
    parser.add_argument(
        "--arch",
        choices=[
            "shared",
            "towers",
            "ln",
            "deep",
            "deep_ln",
            "deep_res",
            "deep_lnres",
            "head21",
            "head31",
            "head12",
            "head22",
            "head32",
            "head13",
            "head23",
            "head33",
            "head21ln",
            "head31ln",
            "head12ln",
            "head22ln",
            "head32ln",
            "head13ln",
            "head23ln",
            "head33ln",
            "lnres21",
            "lnres31",
            "lnres12",
            "lnres22",
            "lnres32",
            "lnres13",
            "lnres23",
            "lnres33",
            "gnres11",
            "gnres21",
            "gnres31",
            "gnres12",
            "gnres22",
            "gnres32",
            "gnres13",
            "gnres23",
            "gnres33",
            "bnres11",
            "bnres21",
            "bnres31",
            "bnres12",
            "bnres22",
            "bnres32",
            "bnres13",
            "bnres23",
            "bnres33",
            "res411",
            "res421",
            "res431",
            "res412",
            "res422",
            "res432",
            "res413",
            "res423",
            "res433",
            "bres411",
            "bres421",
            "bres431",
            "bres412",
            "bres422",
            "bres432",
            "bres413",
            "bres423",
            "bres433",
            "bres211",
            "bres221",
            "bres231",
            "bres212",
            "bres222",
            "bres232",
            "bres213",
            "bres223",
            "bres233",
        ],
        default="shared",
        help=(
            "trunk topology: shared = one trunk feeding both heads "
            "(default); towers = independent actor/critic MLPs "
            "(reference ppo.py layout); ln = shared + LayerNorm before each "
            "hidden activation; deep = shared + a third hidden layer; "
            "deep_ln = both; deep_res/deep_lnres = deep/deep_ln + residual "
            "skips (same parameter counts); head<a><c> = shared trunk + "
            "a-layer actor head / c-layer critic head (a,c in {1,2,3}; "
            "multi-layer heads are residual by default); head<a><c>ln = same "
            "grid with pre-norm LayerNorm inside every residual head block; "
            "lnres<a><c>/gnres<a><c>/bnres<a><c> = residual 3-block body "
            "normalized with LayerNorm / GroupNorm (--gn-groups) / BatchNorm "
            "and *non-residual* a-layer actor / c-layer critic plain heads; "
            "res<d><a><c> = the same plain heads with a no-norm residual body "
            "of d blocks (res4... = depth 4); bres<d><a><c> = the same body "
            "with every residual block widened to a hidden -> mid -> hidden "
            "bottleneck (mid = --res-expansion * --hidden-size)"
        ),
    )
    parser.add_argument(
        "--actor-out-std",
        type=float,
        default=0.01,
        help=(
            "orthogonal-init std of the actor output layer (default 0.01 = "
            "historical); deep actor heads may need a larger gain (e.g. 0.1) "
            "so their hidden layers are not starved by the small output scale"
        ),
    )
    parser.add_argument(
        "--gn-groups",
        type=int,
        default=8,
        help=(
            "gnres<a><c> only: number of groups of the body's GroupNorm "
            "(default 8; must divide --hidden-size)"
        ),
    )
    parser.add_argument(
        "--res-expansion",
        type=float,
        default=3.0,
        help=(
            "bres<d><a><c> only: multiplier of the residual block's hidden "
            "width (hidden -> mid -> hidden, mid = factor * --hidden-size); "
            "default 3.  Values < 1 make a compression bottleneck: 0.25 at "
            "--hidden-size 64 is the (64, 16, 64) block.  hidden * factor "
            "must be a positive integer"
        ),
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=0,
        metavar="L",
        help=(
            "D1-lite sequence memory: public card-history token length fed to "
            "a GRU encoder (0 = off, historical MLP; 54 covers the whole deck)"
        ),
    )
    parser.add_argument(
        "--seq-emb", type=int, default=16, help="card embedding width (--seq-len > 0)"
    )
    parser.add_argument(
        "--seq-hidden",
        type=int,
        default=32,
        help="GRU hidden width concatenated to the observation (--seq-len > 0)",
    )
    parser.add_argument(
        "--seq-blind",
        type=_bool,
        default=False,
        nargs="?",
        const=True,
        help=(
            "capacity control: keep the encoder/trunk but feed an all-pad "
            "history, separating encoder capacity from sequence information"
        ),
    )
    parser.add_argument(
        "--seq-order",
        choices=list(ORDERINGS),
        default="chrono",
        help=(
            "history token layout: chrono = public play order (default); "
            "sorted = same card multiset sorted by card id (order ablation, "
            "isolates the GRU/order contribution)"
        ),
    )
    parser.add_argument(
        "--event-len",
        type=int,
        default=0,
        metavar="L",
        help=(
            "EVH event-level history: max public events fed to the event "
            "encoder (0 = off; 108 covers the 2-player upper bound)"
        ),
    )
    parser.add_argument(
        "--event-emb", type=int, default=32, help="event MLP width (--event-len > 0)"
    )
    parser.add_argument(
        "--event-hidden",
        type=int,
        default=32,
        help="event GRU hidden width concatenated to the observation",
    )
    parser.add_argument(
        "--seat-emb",
        type=int,
        default=16,
        help="relative-seat embedding width (--event-len > 0)",
    )
    parser.add_argument(
        "--event-blind",
        type=_bool,
        default=False,
        nargs="?",
        const=True,
        help=(
            "zero the event vectors and the seat segment (channel alive, "
            "information zero -- seed-matched MLP-like control)"
        ),
    )
    parser.add_argument(
        "--event-noisy",
        type=_bool,
        default=False,
        nargs="?",
        const=True,
        help=(
            "replace the event vectors with fixed-seed i.i.d. noise and pad "
            "the seats: information-zero but non-degenerate (Q-A control)"
        ),
    )
    parser.add_argument(
        "--event-seat",
        choices=["concat", "sum", "none"],
        default="concat",
        help=(
            "relative-seat usage: concat (default), sum (needs seat-emb == "
            "event-emb), none = zeroed seat segment (event_seatblind arm)"
        ),
    )
    parser.add_argument(
        "--event-order",
        choices=list(EVENT_ORDERINGS),
        default="chrono",
        help=(
            "event order: chrono (default) or shuffled = fixed permutation "
            "inside each trick block (order/recency ablation)"
        ),
    )
    parser.add_argument(
        "--event-pass",
        choices=list(PASS_MODES),
        default="keep",
        help="keep (default) or drop pass events (rhythm ablation)",
    )
    parser.add_argument(
        "--event-boundary-blind",
        type=_bool,
        default=False,
        nargs="?",
        const=True,
        help="zero the opens_trick flag (trick-boundary ablation)",
    )
    parser.add_argument(
        "--event-wentout",
        choices=list(WENT_OUT_MODES),
        default="keep",
        help="keep (default) or drop the went_out flag (ablation)",
    )
    parser.add_argument(
        "--opponent",
        choices=["random", "self", "mix", "pool"],
        default="random",
        help=(
            "stage 1: scripted random bot; stage 2: frozen self-play snapshot; "
            "mix: frozen self + random pool (see --mix-random-prob); "
            "pool: N-member league (see --pool-member)"
        ),
    )
    parser.add_argument(
        "--pool-member",
        action="append",
        default=None,
        metavar="[WEIGHT@]SPEC",
        help=(
            "with --opponent pool: repeat per league member; SPEC is "
            "random / self (refreshable) / ckpt:<agent.pt>; "
            "weight defaults to 1"
        ),
    )
    parser.add_argument(
        "--pool-episode",
        type=_bool,
        default=False,
        nargs="?",
        const=True,
        help=(
            "with --opponent pool/mix: freeze one league member per episode "
            "(EpisodeMixturePolicy) instead of re-drawing per decision; "
            "default off keeps the legacy per-decision MixturePolicy"
        ),
    )
    parser.add_argument(
        "--pfsp",
        type=_bool,
        default=False,
        nargs="?",
        const=True,
        help=(
            "with --opponent pool --pool-episode: re-weight members every "
            "--pfsp-every episodes by (1 - shrunk learner win rate)^2 + "
            "--pfsp-epsilon, mixed with uniform; default off (static weights)"
        ),
    )
    parser.add_argument(
        "--pfsp-every",
        type=int,
        default=100,
        metavar="K",
        help="episodes between PFSP weight updates (default 100)",
    )
    parser.add_argument(
        "--pfsp-uniform-mix",
        type=float,
        default=0.5,
        metavar="M",
        help=(
            "weight on the uniform mixture component: 0 = pure PFSP, "
            "1 = uniform (default 0.5)"
        ),
    )
    parser.add_argument(
        "--pfsp-epsilon",
        type=float,
        default=0.02,
        help="additive smoothing in (1 - wr)^2 + epsilon (default 0.02)",
    )
    parser.add_argument(
        "--pfsp-prior",
        type=float,
        default=10.0,
        help=(
            "Beta prior pseudo-count (alpha = beta = prior/2) shrinking each "
            "member's observed win rate toward 0.5; 0 disables shrinkage "
            "(default 10)"
        ),
    )
    parser.add_argument(
        "--mix-random-prob",
        type=float,
        default=0.5,
        help="with --opponent mix, per-decision probability of facing RandomBot",
    )
    parser.add_argument(
        "--self-play-refresh",
        type=int,
        default=50,
        help="updates between frozen self-play snapshots (0 = never)",
    )
    parser.add_argument(
        "--self-play-sample",
        type=_bool,
        default=False,
        nargs="?",
        const=True,
        help=(
            "--opponent self/mix: sample the frozen self-play opponent's "
            "actions instead of acting greedily"
        ),
    )
    parser.add_argument(
        "--pool-sample",
        type=_bool,
        default=True,
        nargs="?",
        const=True,
        help=(
            "--opponent pool: pool members (ckpt: and self) sample actions "
            "instead of acting greedily (default true, operator decision "
            "ADR-0017); pass false to restore the historical greedy pool"
        ),
    )
    parser.add_argument(
        "--load-checkpoint", type=str, default=None, help="warm-start agent.pt"
    )
    parser.add_argument("--run-dir", type=str, default="runs")
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=100,
        help="updates between checkpoint.pt saves (0 = final agent.pt only)",
    )
    parser.add_argument(
        "--snapshot-interval",
        type=int,
        default=0,
        help=(
            "updates between snapshot checkpoints under <run_dir>/snapshots "
            "(0 = off)"
        ),
    )
    parser.add_argument(
        "--snapshot-steps",
        type=int,
        default=0,
        metavar="N",
        help=(
            "save a snapshot after the first update whose step count reaches "
            "each multiple of N environment steps (0 = off; independent of "
            "--snapshot-interval, which is measured in updates)"
        ),
    )
    parser.add_argument(
        "--eval-interval", type=int, default=0, help="updates between evals (0 = off)"
    )
    parser.add_argument("--eval-opponent", choices=["random"], default="random")
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument(
        "--tensorboard",
        type=_bool,
        default=True,
        nargs="?",
        const=True,
        help="write TensorBoard events to <run_dir>/tb (CSV is always written)",
    )

    args = parser.parse_args(argv)
    args.batch_size = int(args.num_envs * args.num_steps)
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0.0:
        raise SystemExit(
            f"--weight-decay must be finite and >= 0, got {args.weight_decay!r}"
        )
    if args.weight_decay > 0.0 and args.optimizer != "adamw":
        raise SystemExit(
            "--weight-decay requires --optimizer adamw; the default adam path "
            "has no decoupled weight decay"
        )
    return args


def make_env(
    rules: Rules,
    learner: int,
    opponents: list[Policy],
    seed: int,
    idx: int,
    reward_shaping: str = "arcsin",
    win_jump: float = 1.0,
    arcsin_mix: float = 0.5,
    reward_cap: float | None = None,
    seq_len: int = 0,
    seq_order: str = "chrono",
    event_len: int = 0,
    event_order: str = "chrono",
    event_pass: str = "keep",
    event_boundary_blind: bool = False,
    event_went_out: str = "keep",
    event_blind: bool = False,
    event_noisy: bool = False,
):
    """CleanRL-style thunk: one sub-env + episode statistics for the vector env."""

    def thunk() -> gym.Env:
        env = Seven523Env(
            rules=rules,
            opponents=opponents,
            seed=seed + idx,
            learner=learner,
            reward_shaping=reward_shaping,
            win_jump=win_jump,
            arcsin_mix=arcsin_mix,
            reward_cap=reward_cap,
        )
        if seq_len > 0:
            env = HistorySequenceWrapper(env, seq_len, seq_order)
        if event_len > 0:
            env = EventHistoryWrapper(
                env,
                rules,
                length=event_len,
                order=event_order,
                pass_mode=event_pass,
                boundary_blind=event_boundary_blind,
                went_out=event_went_out,
                blind=event_blind,
                noisy=event_noisy,
            )
        return gym.wrappers.RecordEpisodeStatistics(env)

    return thunk


def _episode_info(infos: dict) -> dict | None:
    """Finished-episode stats from a vector-env info dict.

    ``RecordEpisodeStatistics`` reports the terminal stats either directly
    (``AutoresetMode.NEXT_STEP``: ``infos["episode"]``) or nested under the
    auto-reset payload (``SAME_STEP``: ``infos["final_info"]["episode"]``).
    Both are dict-of-arrays indexed by sub-env.
    """
    episode = infos.get("episode")
    if episode is not None:
        return episode
    final_info = infos.get("final_info")
    if final_info is not None:
        return final_info.get("episode")
    return None


def _final_outcome(infos: dict, index: int) -> int | None:
    """The env-published win/tie/loss for one finished sub-env, if present."""
    final_info = infos.get("final_info")
    if not isinstance(final_info, dict):
        return None
    outcomes = final_info.get("outcome")
    if outcomes is None:
        return None
    return int(outcomes[index])


def _final_reward_jump(infos: dict, index: int) -> float | None:
    """The env-published terminal win/loss addend for one finished sub-env."""
    final_info = infos.get("final_info")
    if not isinstance(final_info, dict):
        return None
    jumps = final_info.get("reward_jump")
    if jumps is None:
        return None
    return float(jumps[index])


def _episode_outcome_labels(
    outcome: int,
    step: int,
    start: int,
    gamma: float,
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Discounted outcome target for rollout rows ``[start, step]``.

    The terminal jump is received at ``step``, so a state ``t`` in the
    episode carries ``gamma**(step - t) * outcome``; the same discount that
    the GAE jump stream applies to the terminal reward.
    """
    steps = torch.arange(start, step + 1, device=device)
    return float(outcome) * gamma ** (step - steps)


#: Reward modes whose episode return always has the sign of the seat outcome,
#: so the PFSP fallback may infer the outcome from the reward.  ``saturate``
#: (and especially K0, where a win scores 0) is deliberately not here: reading
#: the outcome off the reward sign would score wins as ties.
_SIGN_LIKE_REWARDS = frozenset({"win"})

#: Reward modes the ``--vf-outcome`` decomposition is defined for: their
#: terminal reward is exactly ``base + scale * seat_outcome`` (the env
#: publishes ``info["reward_jump"]`` for all of them).
_OUTCOME_JUMP_REWARDS = frozenset(
    {"arcsin", "terminal_win", "trick_diff_win", "win"}
)


def train(args: argparse.Namespace) -> Path:
    """Run PPO and return the run directory (checkpoints + ``metrics.csv``)."""
    if args.pool_episode and args.opponent not in {"pool", "mix"}:
        raise SystemExit("--pool-episode needs --opponent pool or mix")
    validate_reward_cap(args.reward_shaping, args.reward_cap)
    if not 0.0 <= args.arcsin_mix <= 1.0:
        raise SystemExit("--arcsin-mix must be in [0, 1]")
    if args.lr_floor < 0.0:
        raise SystemExit("--lr-floor must be >= 0")
    if not math.isfinite(args.outcome_coef) or args.outcome_coef < 0.0:
        raise SystemExit("--outcome-coef must be finite and >= 0")
    if args.gn_groups < 1:
        raise SystemExit("--gn-groups must be >= 1")
    if args.arch.startswith("gnres") and args.hidden_size % args.gn_groups != 0:
        raise SystemExit(
            f"--gn-groups {args.gn_groups} must divide --hidden-size "
            f"{args.hidden_size} for --arch {args.arch}"
        )
    if args.vf_outcome and args.reward_shaping not in _OUTCOME_JUMP_REWARDS:
        raise SystemExit(
            "--vf-outcome needs a reward mode with a discrete outcome jump "
            f"({', '.join(sorted(_OUTCOME_JUMP_REWARDS))}), not "
            f"--reward-shaping {args.reward_shaping}"
        )
    if args.vf_outcome and args.vf_nll:
        raise SystemExit(
            "--vf-outcome is incompatible with --vf-nll: the decomposition "
            "is defined for the mean-only critic"
        )
    if args.pfsp and not (args.opponent == "pool" and args.pool_episode):
        raise SystemExit("--pfsp needs --opponent pool --pool-episode")
    if args.pfsp:
        if args.pfsp_every < 1:
            raise ValueError("--pfsp-every must be >= 1")
        if not 0.0 <= args.pfsp_uniform_mix <= 1.0:
            raise ValueError("--pfsp-uniform-mix must be in [0, 1]")
        if args.pfsp_epsilon < 0:
            raise ValueError("--pfsp-epsilon must be non-negative")
        if args.pfsp_prior < 0:
            raise ValueError("--pfsp-prior must be non-negative")
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    run_dir = Path(args.run_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True))
    logger = MetricsLogger(run_dir / "metrics.csv")
    tensorboard = TensorboardLogger(run_dir / "tb", enabled=args.tensorboard)
    tensorboard.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n"
        + "\n".join(f"|{key}|{value}|" for key, value in sorted(vars(args).items())),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    if args.vf_decoupled and not args.vf_nll:
        raise SystemExit("--vf-decoupled requires --vf-nll true")
    rules = Rules(num_players=args.num_players)
    learner = 0
    obs_dim = observation_dim(rules.num_players)
    nvec = np.asarray(nvec_for(rules), dtype=np.int64)

    agent = Agent(
        obs_dim,
        nvec,
        hidden=args.hidden_size,
        activation=args.activation,
        arch=args.arch,
        actor_out_std=args.actor_out_std,
        gn_groups=args.gn_groups,
        res_expansion=args.res_expansion,
        vf_nll=args.vf_nll,
        vf_sample=args.vf_sample,
        vf_outcome=args.vf_outcome,
        seq_len=args.seq_len,
        seq_emb=args.seq_emb,
        seq_hidden=args.seq_hidden,
        seq_blind=args.seq_blind,
        seq_order=args.seq_order,
        event_len=args.event_len,
        event_dim=EVENT_DIM,
        event_emb=args.event_emb,
        event_hidden=args.event_hidden,
        seat_emb=args.seat_emb,
        event_blind=args.event_blind,
        event_seat=args.event_seat,
        event_order=args.event_order,
        event_pass=args.event_pass,
        event_boundary_blind=args.event_boundary_blind,
        event_noisy=args.event_noisy,
        event_went_out=args.event_wentout,
        num_players=rules.num_players,
    ).to(device)
    if args.load_checkpoint:
        try:
            warm = warm_start_from(args.load_checkpoint, agent, device=str(device))
        except WarmStartLayoutError as error:
            if error.reason is not None:
                detail = error.reason
                guidance = "train the value head from scratch or match vf_nll"
            elif error.loaded_arch is not None:
                detail = f"arch {error.loaded_arch!r} into {error.agent_arch!r}"
                guidance = (
                    "only the shared<->towers bridge is supported for "
                    "cross-architecture warm starts"
                )
            else:
                detail = f"layout {error.loaded_layout!r} into {error.agent_layout!r}"
                guidance = (
                    "the event arm needs a from-scratch run or a "
                    "same-layout checkpoint"
                )
            raise SystemExit(
                f"--load-checkpoint {args.load_checkpoint}: cannot warm-start "
                f"{detail} (obs_dim {error.loaded_obs_dim}->{error.agent_obs_dim}); "
                f"{guidance}"
            ) from error
        except RuntimeError as error:
            raise SystemExit(
                f"--load-checkpoint {args.load_checkpoint}: architecture does "
                f"not match the requested agent ({error})"
            ) from error
        if not warm.exact:
            print(
                f"warm start: copied {len(warm.copied)} tensors from "
                f"{args.load_checkpoint} (arch {warm.arch}->{args.arch}, "
                f"nvec {warm.nvec} -> {nvec.tolist()})"
            )
    optimizer_cls = optim.AdamW if args.optimizer == "adamw" else optim.Adam
    optimizer = optimizer_cls(
        agent.parameters(),
        lr=args.learning_rate,
        eps=1e-5,
        weight_decay=args.weight_decay,
    )
    config = PPOConfig.from_args(args)

    league = build_league(
        LeagueConfig(
            opponent=args.opponent,
            num_envs=args.num_envs,
            num_players=args.num_players,
            seed=args.seed,
            pool_member=tuple(args.pool_member or ()),
            mix_random_prob=args.mix_random_prob,
            pool_episode=args.pool_episode,
            self_play_sample=args.self_play_sample,
            pool_sample=args.pool_sample,
        ),
        rules=rules,
        agent=agent,
        device=device,
    )
    frozen = league.frozen
    opponents = league.opponents
    member_ids = league.member_ids
    episode_mixtures = league.episode_mixtures

    # PFSP (direction C): cumulative per-member win/draw/loss record from the
    # learner's seat; every --pfsp-every finished episodes the pool weights are
    # recomputed and pushed into every live EpisodeMixturePolicy.
    pfsp: PfspController | None = None
    pfsp_writer = None
    if args.pfsp:
        pfsp = PfspController(
            member_ids,
            prior=args.pfsp_prior,
            epsilon=args.pfsp_epsilon,
            uniform_mix=args.pfsp_uniform_mix,
            every=args.pfsp_every,
        )
        pfsp_writer = (run_dir / "pfsp_weights.csv").open("w")
        pfsp_writer.write(
            "global_step,episodes,member_id,weight,win_rate,wins,draws,losses\n"
        )

    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                rules,
                learner,
                opponents[idx],
                args.seed,
                idx,
                args.reward_shaping,
                win_jump=args.win_jump,
                arcsin_mix=args.arcsin_mix,
                reward_cap=args.reward_cap,
                seq_len=args.seq_len,
                seq_order=args.seq_order,
                event_len=args.event_len,
                event_order=args.event_order,
                event_pass=args.event_pass,
                event_boundary_blind=args.event_boundary_blind,
                event_went_out=args.event_wentout,
                event_blind=args.event_blind,
                event_noisy=args.event_noisy,
            )
            for idx in range(args.num_envs)
        ],
        # Match the gym 0.21 reference stack (ADR-0003): a finished sub-env is
        # reset on the same step.  gymnasium's default NEXT_STEP would swallow
        # the next action, store it with reward=0/done=0, and chain the row
        # across episodes (~4% chimera transitions on lvlbase).
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    assert isinstance(envs.single_action_space, gym.spaces.MultiDiscrete), (
        "only MultiDiscrete action spaces are supported"
    )
    assert envs.single_action_space.nvec.tolist() == nvec.tolist()
    assert tuple(envs.single_observation_space.shape) == (obs_dim,)

    # ALGO Logic: storage setup
    obs = torch.zeros((args.num_steps, args.num_envs, obs_dim), device=device)
    actions = torch.zeros(
        (args.num_steps, args.num_envs, len(nvec)), device=device
    )
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)
    action_masks = torch.zeros(
        (args.num_steps, args.num_envs, int(nvec.sum())), device=device
    )
    seqs = (
        torch.zeros(
            (args.num_steps, args.num_envs, int(args.seq_len)), device=device
        )
        if args.seq_len > 0
        else None
    )
    events = (
        torch.zeros(
            (args.num_steps, args.num_envs, int(args.event_len), EVENT_DIM),
            device=device,
        )
        if args.event_len > 0
        else None
    )
    event_seats = (
        torch.zeros(
            (args.num_steps, args.num_envs, int(args.event_len)),
            dtype=torch.long,
            device=device,
        )
        if args.event_len > 0
        else None
    )
    event_masks = (
        torch.zeros(
            (args.num_steps, args.num_envs, int(args.event_len)),
            dtype=torch.bool,
            device=device,
        )
        if args.event_len > 0
        else None
    )
    # --vf-outcome: the env publishes the terminal win/loss addend; the critic
    # trains on the jump-free stream while the outcome head gets the discounted
    # outcomes of the episodes that finish inside each rollout.  The buffers
    # are fresh per rollout (inside the update loop) because every cell must be
    # zero at step 0: a stale jump or label from the previous rollout would
    # silently contaminate both streams.
    jump_rewards: torch.Tensor | None = None
    values_jump: torch.Tensor | None = None
    outcome_targets: torch.Tensor | None = None
    outcome_mask: torch.Tensor | None = None
    segment_start: np.ndarray | None = None
    jump_scale = 0.0
    if args.vf_outcome:
        scale = float(envs.envs[0].get_wrapper_attr("reward_jump_scale"))
        if scale <= 0.0:
            raise ValueError(
                f"reward_shaping={args.reward_shaping!r} publishes no outcome "
                f"jump for --vf-outcome"
            )
        jump_scale = scale

    num_updates = args.total_timesteps // args.batch_size
    if num_updates < 1:
        raise ValueError(
            f"total_timesteps={args.total_timesteps} is smaller than one batch "
            f"({args.batch_size}); lower --num-envs/--num-steps or raise the budget"
        )

    global_step = 0
    episodes_done = 0
    #: Step-based snapshot boundaries (see ``--snapshot-steps``).  An update
    #: saves a snapshot at every boundary it reaches, so batch sizes that do
    #: not divide N land on the first update at/after each multiple of N.
    snapshot_steps = int(getattr(args, "snapshot_steps", 0) or 0)
    next_snapshot_step = snapshot_steps
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
    next_done = torch.zeros(args.num_envs, device=device)

    try:
        for update in range(1, num_updates + 1):
            if (
                frozen is not None
                and args.self_play_refresh
                and update % args.self_play_refresh == 0
            ):
                assert history_layout(frozen.agent) == history_layout(agent), (
                    "self-play frozen opponent history layout diverged from the "
                    f"learner: {history_layout(frozen.agent)!r} vs "
                    f"{history_layout(agent)!r}"
                )
                frozen.agent.load_state_dict(agent.state_dict())

            if args.anneal_lr:
                optimizer.param_groups[0]["lr"] = annealed_lr(
                    args.lr_schedule,
                    update,
                    num_updates,
                    args.learning_rate,
                    args.lr_floor,
                )

            if args.vf_outcome:
                jump_rewards = torch.zeros(
                    (args.num_steps, args.num_envs), device=device
                )
                values_jump = torch.zeros(
                    (args.num_steps, args.num_envs), device=device
                )
                outcome_targets = torch.zeros(
                    (args.num_steps, args.num_envs), device=device
                )
                outcome_mask = torch.zeros(
                    (args.num_steps, args.num_envs),
                    dtype=torch.bool,
                    device=device,
                )
                segment_start = np.zeros(args.num_envs, dtype=np.int64)
            ep_returns: list[float] = []
            ep_lengths: list[int] = []
            for step in range(args.num_steps):
                global_step += args.num_envs
                obs[step] = next_obs
                dones[step] = next_done
                action_masks[step] = torch.as_tensor(
                    np.asarray(
                        [
                            env.get_wrapper_attr("action_mask")
                            for env in envs.envs
                        ],
                        dtype=np.float32,
                    ),
                    device=device,
                )

                # ALGO LOGIC: action logic
                seq_tokens: torch.Tensor | None = None
                if seqs is not None:
                    seq_tokens = torch.as_tensor(
                        np.asarray(
                            [
                                env.get_wrapper_attr("last_seq")
                                for env in envs.envs
                            ],
                            dtype=np.int64,
                        ),
                        dtype=torch.int64,
                        device=device,
                    )
                    if args.seq_blind:
                        seq_tokens = torch.zeros_like(seq_tokens)
                    seqs[step] = seq_tokens
                event_tensors: torch.Tensor | None = None
                seat_tensors: torch.Tensor | None = None
                mask_tensors: torch.Tensor | None = None
                if events is not None:
                    event_tensors = torch.as_tensor(
                        np.asarray(
                            [
                                env.get_wrapper_attr("last_events")
                                for env in envs.envs
                            ]
                        ),
                        dtype=torch.float32,
                        device=device,
                    )
                    seat_tensors = torch.as_tensor(
                        np.asarray(
                            [
                                env.get_wrapper_attr("last_event_seats")
                                for env in envs.envs
                            ]
                        ),
                        dtype=torch.int64,
                        device=device,
                    )
                    mask_tensors = torch.as_tensor(
                        np.asarray(
                            [
                                env.get_wrapper_attr("last_event_mask")
                                for env in envs.envs
                            ]
                        ),
                        dtype=torch.bool,
                        device=device,
                    )
                    events[step] = event_tensors
                    assert event_seats is not None and event_masks is not None
                    event_seats[step] = seat_tensors
                    event_masks[step] = mask_tensors
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(
                        next_obs,
                        action_masks[step],
                        seqs=seq_tokens,
                        events=event_tensors,
                        event_seats=seat_tensors,
                        event_mask=mask_tensors,
                    )
                    if args.vf_outcome:
                        # The critic head is the jump-free value; the outcome
                        # head supplies the jump baseline for the advantages.
                        assert values_jump is not None
                        value, outcome_pred = agent.get_value_and_outcome(
                            next_obs,
                            seqs=seq_tokens,
                            events=event_tensors,
                            event_seats=seat_tensors,
                            event_mask=mask_tensors,
                        )
                        values_jump[step] = (jump_scale * outcome_pred).flatten()
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, termination, truncation, infos = envs.step(
                    action.cpu().numpy()
                )
                rewards[step] = torch.as_tensor(
                    reward, dtype=torch.float32, device=device
                ).view(-1)
                next_obs = torch.as_tensor(
                    next_obs, dtype=torch.float32, device=device
                )
                finished = np.logical_or(termination, truncation)
                next_done = torch.as_tensor(
                    finished.astype(np.float32), device=device
                )
                if (episode_info := _episode_info(infos)) is not None:
                    for idx in np.flatnonzero(finished):
                        episode_return = float(episode_info["r"][idx])
                        episode_length = int(episode_info["l"][idx])
                        ep_returns.append(episode_return)
                        ep_lengths.append(episode_length)
                        tensorboard.log_episode(
                            global_step, episode_return, episode_length
                        )
                        if args.vf_outcome:
                            # Backfill the episode's states still inside this
                            # rollout with the discounted outcome label.  The
                            # pre-rollout head of an unfinished episode is
                            # already gone, which is why the label carries a
                            # mask instead of being assumed zero.
                            assert (
                                jump_rewards is not None
                                and outcome_targets is not None
                                and outcome_mask is not None
                                and segment_start is not None
                            )
                            jump = _final_reward_jump(infos, int(idx))
                            outcome = _final_outcome(infos, int(idx))
                            if jump is None or outcome is None:
                                raise RuntimeError(
                                    "finished episode without "
                                    "info['reward_jump']/info['outcome'] under "
                                    "--vf-outcome; the env must publish the "
                                    "reward decomposition"
                                )
                            start = int(segment_start[idx])
                            jump_rewards[step, idx] = jump
                            outcome_targets[start : step + 1, idx] = (
                                _episode_outcome_labels(
                                    outcome,
                                    step,
                                    start,
                                    args.gamma,
                                    device=device,
                                )
                            )
                            outcome_mask[start : step + 1, idx] = True
                            segment_start[idx] = step + 1
                        if pfsp is not None:
                            # The env publishes the terminal win/tie/loss; the
                            # reward sign is only a fallback for exotic wrappers.
                            outcome = _final_outcome(infos, int(idx))
                            if outcome is None:
                                if args.reward_shaping not in _SIGN_LIKE_REWARDS:
                                    raise RuntimeError(
                                        "finished episode without "
                                        "info['outcome'] under "
                                        f"--reward-shaping "
                                        f"{args.reward_shaping}; refusing to "
                                        "infer the outcome from the reward "
                                        "sign"
                                    )
                                outcome = int(episode_return > 0.0) - int(
                                    episode_return < 0.0
                                )
                            counted = False
                            for policy in episode_mixtures[int(idx)]:
                                if not isinstance(policy, WeightedPolicy):
                                    continue
                                member_id = policy.finished_id
                                if member_id is None:
                                    continue
                                pfsp.record(
                                    member_id, outcome, episode=not counted
                                )
                                counted = True
                            weights = pfsp.maybe_update()
                            if weights is not None:
                                ordered = [
                                    weights[member_id] for member_id in member_ids
                                ]
                                for env_mixtures in episode_mixtures:
                                    for policy in env_mixtures:
                                        if isinstance(policy, WeightedPolicy):
                                            policy.set_weights(ordered)
                                for member_id in member_ids:
                                    wins, draws, losses = pfsp.records[member_id]
                                    assert pfsp_writer is not None
                                    pfsp_writer.write(
                                        f"{global_step},{pfsp.episodes},{member_id},"
                                        f"{weights[member_id]:.6g},"
                                        f"{pfsp.win_rate(member_id):.6g},"
                                        f"{wins},{draws},{losses}\n"
                                    )
                                    tensorboard.add_scalar(
                                        f"pfsp/weight/{member_id}",
                                        weights[member_id],
                                        global_step,
                                    )
                                pfsp_writer.flush()
                                print(
                                    f"pfsp update {pfsp.updates} @ step "
                                    f"{global_step} (episodes {pfsp.episodes}): "
                                    + ", ".join(
                                        f"{member_id}={weights[member_id]:.3f}"
                                        for member_id in member_ids
                                    )
                                )
            episodes_done += len(ep_returns)

            # bootstrap value if not done
            next_seqs: torch.Tensor | None = None
            if seqs is not None:
                next_seqs = torch.as_tensor(
                    np.asarray(
                        [env.get_wrapper_attr("last_seq") for env in envs.envs],
                        dtype=np.int64,
                    ),
                    dtype=torch.int64,
                    device=device,
                )
                if args.seq_blind:
                    next_seqs = torch.zeros_like(next_seqs)
            next_events: torch.Tensor | None = None
            next_event_seats: torch.Tensor | None = None
            next_event_mask: torch.Tensor | None = None
            if events is not None:
                next_events = torch.as_tensor(
                    np.asarray(
                        [env.get_wrapper_attr("last_events") for env in envs.envs]
                    ),
                    dtype=torch.float32,
                    device=device,
                )
                next_event_seats = torch.as_tensor(
                    np.asarray(
                        [
                            env.get_wrapper_attr("last_event_seats")
                            for env in envs.envs
                        ]
                    ),
                    dtype=torch.int64,
                    device=device,
                )
                next_event_mask = torch.as_tensor(
                    np.asarray(
                        [
                            env.get_wrapper_attr("last_event_mask")
                            for env in envs.envs
                        ]
                    ),
                    dtype=torch.bool,
                    device=device,
                )
            with torch.no_grad():
                if args.vf_outcome:
                    assert jump_rewards is not None and values_jump is not None
                    # GAE is linear in (reward, value), so the total advantage
                    # splits exactly into the jump-free stream plus the jump
                    # stream baselined by the outcome head.  The critic target
                    # (returns) stays jump-free.
                    next_value, next_outcome = agent.get_value_and_outcome(
                        next_obs,
                        next_seqs,
                        next_events,
                        next_event_seats,
                        next_event_mask,
                    )
                    next_value = next_value.reshape(1, -1)
                    next_value_jump = (jump_scale * next_outcome).reshape(1, -1)
                    adv_base, returns = compute_gae(
                        rewards - jump_rewards,
                        values,
                        dones,
                        next_value,
                        next_done,
                        gamma=args.gamma,
                        gae_lambda=args.gae_lambda,
                        use_gae=args.gae,
                    )
                    adv_jump, _ = compute_gae(
                        jump_rewards,
                        values_jump,
                        dones,
                        next_value_jump,
                        next_done,
                        gamma=args.gamma,
                        gae_lambda=args.gae_lambda,
                        use_gae=args.gae,
                    )
                    advantages = adv_base + adv_jump
                else:
                    next_value = agent.get_value(
                        next_obs,
                        next_seqs,
                        next_events,
                        next_event_seats,
                        next_event_mask,
                    ).reshape(1, -1)
                    advantages, returns = compute_gae(
                        rewards,
                        values,
                        dones,
                        next_value,
                        next_done,
                        gamma=args.gamma,
                        gae_lambda=args.gae_lambda,
                        use_gae=args.gae,
                    )

            batch = RolloutBatch.flatten(
                obs,
                actions,
                logprobs,
                advantages,
                returns,
                values,
                action_masks,
                seqs,
                events,
                event_seats,
                event_masks,
                outcome_targets,
                outcome_mask,
            )
            losses = ppo_update(agent, optimizer, batch, config)

            sps = int(global_step / (time.time() - start_time))
            metrics: dict[str, float | None] = {
                "episodic_return": float(np.mean(ep_returns)) if ep_returns else None,
                "episodic_length": float(np.mean(ep_lengths)) if ep_lengths else None,
                "episodes": float(episodes_done),
                "learning_rate": optimizer.param_groups[0]["lr"],
                **losses,
                "sps": float(sps),
            }

            if args.eval_interval and update % args.eval_interval == 0:
                policy = NeuralPolicy(agent, rules, device=device)
                evaluation = evaluate(
                    policy,
                    rules=rules,
                    opponent=args.eval_opponent,
                    episodes=args.eval_episodes,
                    seed=args.seed + update,
                )
                # ``NeuralPolicy`` puts the *live* learner into eval mode; the
                # learner must keep collecting/updating in train mode (a
                # ``bnres`` body would otherwise freeze its running statistics
                # from the first periodic eval onward).  Frozen opponents are
                # separate deep copies, so only the learner is restored.
                agent.train()
                metrics.update(
                    eval_return=evaluation["mean_return"],
                    eval_score=evaluation["learner_score"],
                    eval_score_diff=evaluation["score_diff"],
                    eval_win_rate=evaluation["win_rate"],
                )

            if args.log_interval and update % args.log_interval == 0:
                print(
                    f"update {update}/{num_updates}  step {global_step}  "
                    f"ep_return {metrics['episodic_return']}  "
                    f"v_loss {metrics['value_loss']:.3f}  "
                    f"pg_loss {metrics['policy_loss']:.3f}  "
                    f"entropy {metrics['entropy']:.3f}  "
                    f"kl {metrics['approx_kl']:.4f}  sps {sps}"
                )

            logger.log(global_step, **metrics)
            tensorboard.log_update(global_step, metrics)

            if args.checkpoint_interval and update % args.checkpoint_interval == 0:
                save_agent(
                    run_dir / "checkpoint.pt",
                    agent,
                    extra={"global_step": global_step, "args": vars(args)},
                )

            if args.snapshot_interval and update % args.snapshot_interval == 0:
                snapshot_dir = run_dir / "snapshots"
                snapshot_dir.mkdir(parents=True, exist_ok=True)
                save_agent(
                    snapshot_dir / f"checkpoint_step{global_step:07d}.pt",
                    agent,
                    extra={"global_step": global_step, "args": vars(args)},
                )

            if snapshot_steps and global_step >= next_snapshot_step:
                snapshot_dir = run_dir / "snapshots"
                snapshot_dir.mkdir(parents=True, exist_ok=True)
                save_agent(
                    snapshot_dir / f"checkpoint_step{global_step:07d}.pt",
                    agent,
                    extra={"global_step": global_step, "args": vars(args)},
                )
                while global_step >= next_snapshot_step:
                    next_snapshot_step += snapshot_steps

        save_agent(
            run_dir / "agent.pt",
            agent,
            extra={"global_step": global_step, "args": vars(args)},
        )
    finally:
        envs.close()
        tensorboard.close()
        if pfsp_writer is not None:
            pfsp_writer.close()

    return run_dir


def main() -> None:
    args = parse_args()
    run_dir = train(args)
    print(f"run written to {run_dir}")


if __name__ == "__main__":
    main()

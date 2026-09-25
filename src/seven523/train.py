"""PPO training for 7鬼523 — a faithful port of ``ppo_multidiscrete_mask.py``.

The reference script in ``ppo-implementation-details/`` targets gym 0.21 +
Python <3.10 (and a MicroRTS CNN), so it cannot run here unmodified.  This
module keeps its algorithm and implementation details, swapping in gymnasium,
a flat-observation MLP, and this project's environment.  See
``docs/adr/0003-training-stack.md``.

Typical runs::

    uv sync --group train
    uv run --group train 7g523-train --total-timesteps 1000000          # vs GreedyBot
    uv run --group train 7g523-train --opponent self --load-checkpoint runs/<new-run>/agent.pt
    uv run --group train 7g523-eval --checkpoint runs/<new-run>/agent.pt --episodes 500
    uv run --group train tensorboard --logdir runs --port 6006          # 看曲线
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim

from .actions import nvec_for
from .env import Seven523Env, observation_dim
from .eval import evaluate
from .networks import Agent, NeuralPolicy, load_agent, save_agent, warm_start_into
from .policies import (
    EpisodeMixturePolicy,
    MixturePolicy,
    Policy,
    make_scripted_policies,
    pfsp_weights,
    policy_from_spec,
)
from .ppo import PPOConfig, RolloutBatch, compute_gae, ppo_update
from .rules import Rules

__all__ = ["MetricsLogger", "TensorboardLogger", "main", "make_env", "parse_args", "train"]

_LOG_FIELDS = (
    "episodic_return",
    "episodic_length",
    "episodes",
    "learning_rate",
    "value_loss",
    "policy_loss",
    "entropy",
    "old_approx_kl",
    "approx_kl",
    "clipfrac",
    "explained_variance",
    "sps",
    "eval_return",
    "eval_score",
    "eval_score_diff",
    "eval_win_rate",
)


def _bool(value: str) -> bool:
    return value.lower() in {"true", "1", "yes", "y", "t"}


def parse_pool_member(raw: str) -> tuple[float, str]:
    """Split a ``[WEIGHT@]SPEC`` league member (``1.5@ckpt:runs/<new-run>/agent.pt``)."""
    if "@" in raw:
        weight_raw, spec = raw.split("@", 1)
        weight = float(weight_raw)
        if weight < 0:
            raise ValueError(f"pool member weight must be non-negative: {raw!r}")
        return weight, spec.strip()
    return 1.0, raw.strip()


def _pool_member_ids(specs: list[str]) -> list[str]:
    """Stable, readable per-member ids: the spec, disambiguated if repeated."""
    counts = Counter(specs)
    seen: dict[str, int] = {}
    ids: list[str] = []
    for spec in specs:
        if counts[spec] == 1:
            ids.append(spec)
            continue
        seen[spec] = seen.get(spec, 0) + 1
        ids.append(f"{spec}#{seen[spec]}")
    return ids


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
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=None)

    # 7鬼523 / run management arguments.
    parser.add_argument("--num-players", type=int, default=2)
    parser.add_argument(
        "--reward-shaping",
        choices=["terminal", "trick_diff", "win", "trick_diff_win"],
        default="terminal",
        help=(
            "terminal: legacy sparse terminal return; trick_diff: per-step "
            "potential difference (telescopes to the terminal return); win: "
            "terminal sign(own - best other); trick_diff_win: both"
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
        choices=["shared", "towers"],
        default="shared",
        help=(
            "trunk topology: shared = one trunk feeding both heads "
            "(default); towers = independent actor/critic MLPs "
            "(reference ppo.py layout)"
        ),
    )
    parser.add_argument(
        "--opponent",
        choices=["greedy", "random", "self", "mix", "pool"],
        default="greedy",
        help=(
            "stage 1: scripted bot; stage 2: frozen self-play snapshot; "
            "mix: frozen self + greedy pool (see --mix-greedy-prob); "
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
            "greedy / random / self (refreshable) / ckpt:<agent.pt>; "
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
        "--mix-greedy-prob",
        type=float,
        default=0.5,
        help="with --opponent mix, per-decision probability of facing GreedyBot",
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
        help="sample self-play opponent actions instead of acting greedily",
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
        "--eval-interval", type=int, default=0, help="updates between evals (0 = off)"
    )
    parser.add_argument("--eval-opponent", choices=["greedy", "random"], default="greedy")
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
    return args


def make_env(
    rules: Rules,
    learner: int,
    opponents: list[Policy],
    seed: int,
    idx: int,
    reward_shaping: str = "terminal",
):
    """CleanRL-style thunk: one sub-env + episode statistics for the vector env."""

    def thunk() -> gym.Env:
        env = Seven523Env(
            rules=rules,
            opponents=opponents,
            seed=seed + idx,
            learner=learner,
            reward_shaping=reward_shaping,
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


class MetricsLogger:
    """Append one CSV row per update; no tensorboard/wandb dependency."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.write_text("global_step," + ",".join(_LOG_FIELDS) + "\n")

    def log(self, global_step: int, **values: float | None) -> None:
        cells = []
        for field in _LOG_FIELDS:
            value = values.get(field)
            cells.append("" if value is None else f"{value:.6g}")
        with self.path.open("a") as handle:
            handle.write(",".join([str(global_step), *cells]) + "\n")


_TB_TAGS: dict[str, str] = {
    "episodes": "charts/episodes",
    "learning_rate": "charts/learning_rate",
    "value_loss": "losses/value_loss",
    "policy_loss": "losses/policy_loss",
    "entropy": "losses/entropy",
    "old_approx_kl": "losses/old_approx_kl",
    "approx_kl": "losses/approx_kl",
    "clipfrac": "losses/clipfrac",
    "explained_variance": "losses/explained_variance",
    "sps": "charts/SPS",
    "eval_return": "eval/mean_return",
    "eval_score": "eval/learner_score",
    "eval_score_diff": "eval/score_diff",
    "eval_win_rate": "eval/win_rate",
}


class TensorboardLogger:
    """Thin tensorboardX wrapper; no-op when disabled or not installed."""

    def __init__(self, log_dir: str | Path, enabled: bool = True) -> None:
        self.writer = None
        if enabled:
            try:
                from tensorboardX import SummaryWriter
            except ImportError:
                print(
                    "tensorboardX not installed; skipping TensorBoard "
                    "(metrics.csv is still written; run `uv sync --group train`)"
                )
            else:
                self.writer = SummaryWriter(str(log_dir))

    @property
    def enabled(self) -> bool:
        return self.writer is not None

    def add_scalar(self, tag: str, value: float | None, step: int) -> None:
        if self.writer is not None and value is not None:
            self.writer.add_scalar(tag, value, step)

    def add_text(self, tag: str, text: str, step: int = 0) -> None:
        if self.writer is not None:
            self.writer.add_text(tag, text, step)

    def log_update(self, global_step: int, metrics: dict[str, float | None]) -> None:
        for key, tag in _TB_TAGS.items():
            self.add_scalar(tag, metrics.get(key), global_step)

    def log_episode(self, global_step: int, episode_return: float, length: int) -> None:
        self.add_scalar("charts/episodic_return", episode_return, global_step)
        self.add_scalar("charts/episodic_length", float(length), global_step)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None


def _self_play_opponents(
    args: argparse.Namespace,
    rules: Rules,
    obs_dim: int,
    nvec: np.ndarray,
    device: torch.device,
    agent: Agent,
) -> tuple[list[list[Policy]], NeuralPolicy]:
    frozen = NeuralPolicy(
        Agent(
            obs_dim,
            nvec,
            hidden=args.hidden_size,
            activation=args.activation,
            arch=args.arch,
        ),
        rules,
        device=device,
        sample=args.self_play_sample,
        seed=args.seed,
    )
    frozen.agent.load_state_dict(agent.state_dict())
    return [[frozen] * rules.num_players for _ in range(args.num_envs)], frozen


def train(args: argparse.Namespace) -> Path:
    """Run PPO and return the run directory (checkpoints + ``metrics.csv``)."""
    if args.pool_episode and args.opponent not in {"pool", "mix"}:
        raise SystemExit("--pool-episode needs --opponent pool or mix")
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
    ).to(device)
    if args.load_checkpoint:
        loaded, _ = load_agent(args.load_checkpoint, device=device)
        if loaded.obs_dim != obs_dim:
            raise SystemExit(
                f"--load-checkpoint {args.load_checkpoint}: cannot "
                f"warm-start a {loaded.obs_dim}-wide checkpoint into the v5 "
                f"{obs_dim}-wide layout for {rules.num_players} players"
            )
        if loaded.nvec.tolist() == nvec.tolist() and loaded.arch == args.arch:
            agent.load_state_dict(loaded.state_dict())
        else:
            copied = warm_start_into(agent, loaded)
            print(
                f"warm start: copied {len(copied)} tensors from "
                f"{args.load_checkpoint} (arch {loaded.arch}->{args.arch}, "
                f"nvec {loaded.nvec.tolist()} -> {nvec.tolist()})"
            )
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    config = PPOConfig.from_args(args)

    frozen: NeuralPolicy | None = None
    member_ids: list[str] = []
    episode_mixtures: list[list[EpisodeMixturePolicy]] = [
        [] for _ in range(args.num_envs)
    ]
    if args.opponent in {"self", "mix", "pool"}:
        opponents, frozen = _self_play_opponents(
            args, rules, obs_dim, nvec, device, agent
        )
        if args.opponent == "mix":
            greedy = make_scripted_policies("greedy", rules, seed=args.seed)[0]
            mix_members: list[tuple[float, Policy, str]] = [
                (1.0 - args.mix_greedy_prob, frozen, "self"),
                (args.mix_greedy_prob, greedy, "greedy"),
            ]
            opponents = []
            for idx in range(args.num_envs):
                seat_policies: list[Policy] = []
                for seat in range(rules.num_players):
                    rng = random.Random(args.seed + idx * 1000 + seat)
                    if args.pool_episode:
                        policy: Policy = EpisodeMixturePolicy(mix_members, rng=rng)
                        if seat != learner:
                            episode_mixtures[idx].append(policy)
                    else:
                        policy = MixturePolicy(
                            [(weight, member) for weight, member, _ in mix_members],
                            rng=rng,
                        )
                    seat_policies.append(policy)
                opponents.append(seat_policies)
        elif args.opponent == "pool":
            if not args.pool_member:
                raise SystemExit("--opponent pool needs at least one --pool-member")
            specs = [parse_pool_member(raw)[1] for raw in args.pool_member]
            member_ids = _pool_member_ids(specs)
            members: list[tuple[float, Policy, str]] = []
            for index, raw in enumerate(args.pool_member):
                weight, spec = parse_pool_member(raw)
                if spec == "self":
                    member: Policy = frozen
                else:
                    member = policy_from_spec(
                        spec, rules, seed=args.seed + index, device=str(device)
                    )
                members.append((weight, member, member_ids[index]))
            opponents = []
            for idx in range(args.num_envs):
                seat_policies = []
                for seat in range(rules.num_players):
                    rng = random.Random(args.seed + idx * 1000 + seat)
                    if args.pool_episode:
                        policy = EpisodeMixturePolicy(members, rng=rng)
                        if seat != learner:
                            episode_mixtures[idx].append(policy)
                    else:
                        policy = MixturePolicy(
                            [(weight, member) for weight, member, _ in members],
                            rng=rng,
                        )
                    seat_policies.append(policy)
                opponents.append(seat_policies)
    else:
        opponents = [
            make_scripted_policies(args.opponent, rules, seed=args.seed + idx)
            for idx in range(args.num_envs)
        ]

    # PFSP (direction C): cumulative per-member win/draw/loss record from the
    # learner's seat; every --pfsp-every finished episodes the pool weights are
    # recomputed and pushed into every live EpisodeMixturePolicy.
    pfsp_records: dict[str, list[int]] = {}
    pfsp_episodes = 0
    pfsp_last_update = 0
    pfsp_updates = 0
    pfsp_writer = None

    def pfsp_update(global_step: int) -> None:
        nonlocal pfsp_last_update, pfsp_updates
        weights = pfsp_weights(
            {
                member_id: (
                    float(record[0]),
                    float(record[1]),
                    float(record[2]),
                )
                for member_id, record in pfsp_records.items()
            },
            prior=args.pfsp_prior,
            epsilon=args.pfsp_epsilon,
            uniform_mix=args.pfsp_uniform_mix,
        )
        ordered = [weights[member_id] for member_id in member_ids]
        for env_mixtures in episode_mixtures:
            for policy in env_mixtures:
                policy.set_weights(ordered)
        for member_id in member_ids:
            wins, draws, losses = pfsp_records[member_id]
            games = wins + draws + losses
            denominator = args.pfsp_prior + games
            win_rate = (
                (0.5 * args.pfsp_prior + wins + 0.5 * draws) / denominator
                if denominator
                else 0.5
            )
            assert pfsp_writer is not None
            pfsp_writer.write(
                f"{global_step},{pfsp_episodes},{member_id},{weights[member_id]:.6g},"
                f"{win_rate:.6g},{wins},{draws},{losses}\n"
            )
            tensorboard.add_scalar(
                f"pfsp/weight/{member_id}", weights[member_id], global_step
            )
        pfsp_writer.flush()
        pfsp_updates += 1
        pfsp_last_update = pfsp_episodes
        print(
            f"pfsp update {pfsp_updates} @ step {global_step} "
            f"(episodes {pfsp_episodes}): "
            + ", ".join(
                f"{member_id}={weights[member_id]:.3f}" for member_id in member_ids
            )
        )

    if args.pfsp:
        pfsp_records = {member_id: [0, 0, 0] for member_id in member_ids}
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

    num_updates = args.total_timesteps // args.batch_size
    if num_updates < 1:
        raise ValueError(
            f"total_timesteps={args.total_timesteps} is smaller than one batch "
            f"({args.batch_size}); lower --num-envs/--num-steps or raise the budget"
        )

    global_step = 0
    episodes_done = 0
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
                frozen.agent.load_state_dict(agent.state_dict())

            if args.anneal_lr:
                frac = 1.0 - (update - 1.0) / num_updates
                optimizer.param_groups[0]["lr"] = frac * args.learning_rate

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
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(
                        next_obs, action_masks[step]
                    )
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
                        if pfsp_records:
                            outcome = 0
                            if episode_return > 0.0:
                                outcome = 1
                            elif episode_return < 0.0:
                                outcome = -1
                            for policy in episode_mixtures[int(idx)]:
                                member_id = policy.finished_id
                                if member_id is None:
                                    continue
                                record = pfsp_records.setdefault(
                                    member_id, [0, 0, 0]
                                )
                                if outcome > 0:
                                    record[0] += 1
                                elif outcome < 0:
                                    record[2] += 1
                                else:
                                    record[1] += 1
                            pfsp_episodes += 1
                            if pfsp_episodes - pfsp_last_update >= args.pfsp_every:
                                pfsp_update(global_step)
            episodes_done += len(ep_returns)

            # bootstrap value if not done
            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
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
                obs, actions, logprobs, advantages, returns, values, action_masks
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

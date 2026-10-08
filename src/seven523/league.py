"""Opponent-league construction and PFSP re-weighting for the PPO trainer.

The trainer owns the rollout / update loop; this module owns the league: it
parses pool specs, builds the per-env / per-seat rosters (scripted bots, frozen
self-play, mix, or pool), and keeps the cumulative PFSP win/draw/loss record
that drives pool re-weighting.  ``train()`` only assembles: it builds a
:class:`LeagueConfig`, calls :func:`build_league`, and feeds finished-episode
outcomes to a :class:`PfspController`.

Like :mod:`seven523.train` this is a training-side module and imports torch
through :mod:`seven523.networks`.
"""
from __future__ import annotations

import copy
import random
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from .networks import Agent, NeuralPolicy, history_layout
from .policies import (
    EpisodeMixturePolicy,
    MixturePolicy,
    Policy,
    make_scripted_policies,
    pfsp_weights,
    policy_from_spec,
)
from .rules import Rules

__all__ = [
    "League",
    "LeagueConfig",
    "PfspController",
    "build_league",
    "parse_pool_member",
    "pool_member_ids",
]

#: The trainer always learns in seat 0, so ``episode_mixtures`` covers the
#: other seats only.
_LEARNER = 0


@dataclass(frozen=True, slots=True)
class LeagueConfig:
    """Everything :func:`build_league` needs beyond rules / learner / device.

    ``opponent`` is the CLI mode (``random`` / ``self`` / ``mix`` /
    ``pool``); the pool-only fields are ignored by the other modes.
    """

    opponent: str
    num_envs: int
    num_players: int
    seed: int
    pool_member: tuple[str, ...] = ()
    mix_random_prob: float = 0.5
    pool_episode: bool = False
    self_play_sample: bool = False
    #: Pool mode only: every policy member (``ckpt:`` and ``self``) samples
    #: actions instead of acting greedily.  Default true is the operator
    #: decision of ADR-0017; ``self``/``mix`` keep ``self_play_sample``.
    pool_sample: bool = True


@dataclass(slots=True)
class League:
    """The assembled league: rosters, per-episode mixtures, ids, frozen snapshot."""

    opponents: list[list[Policy]]
    episode_mixtures: list[list[EpisodeMixturePolicy]]
    member_ids: list[str]
    frozen: NeuralPolicy | None


def parse_pool_member(raw: str) -> tuple[float, str]:
    """Split a ``[WEIGHT@]SPEC`` league member (``1.5@ckpt:runs/<new-run>/agent.pt``)."""
    if "@" in raw:
        weight_raw, spec = raw.split("@", 1)
        weight = float(weight_raw)
        if weight < 0:
            raise ValueError(f"pool member weight must be non-negative: {raw!r}")
        return weight, spec.strip()
    return 1.0, raw.strip()


def pool_member_ids(specs: Sequence[str]) -> list[str]:
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


def _mixture_opponents(
    config: LeagueConfig,
    rules: Rules,
    members: Sequence[tuple[float, Policy, str]],
    episode_mixtures: list[list[EpisodeMixturePolicy]],
    seed: int,
) -> list[list[Policy]]:
    """One roster per env: each seat draws from ``members`` (episode-scoped or not)."""
    opponents: list[list[Policy]] = []
    for idx in range(config.num_envs):
        seat_policies: list[Policy] = []
        for seat in range(rules.num_players):
            rng = random.Random(seed + idx * 1000 + seat)
            if config.pool_episode:
                policy: Policy = EpisodeMixturePolicy(members, rng=rng)
                if seat != _LEARNER:
                    episode_mixtures[idx].append(policy)
            else:
                policy = MixturePolicy(
                    [(weight, member) for weight, member, _ in members],
                    rng=rng,
                )
            seat_policies.append(policy)
        opponents.append(seat_policies)
    return opponents


def build_league(
    config: LeagueConfig,
    *,
    rules: Rules,
    agent: Agent,
    device: torch.device | str,
    seed_rng: random.Random | None = None,
) -> League:
    """Assemble the opponent rosters and episode mixtures described by ``config``.

    ``agent`` is the live learner; for ``self`` / ``mix`` / ``pool`` a frozen
    copy is snapshotted from it immediately.  ``seed_rng`` optionally overrides
    ``config.seed`` as the base seed; without it the derived seeds are exactly
    the trainer's historical ``seed + idx * 1000 + seat``.
    """
    seed = config.seed if seed_rng is None else seed_rng.randrange(1 << 32)
    frozen: NeuralPolicy | None = None
    member_ids: list[str] = []
    episode_mixtures: list[list[EpisodeMixturePolicy]] = [
        [] for _ in range(config.num_envs)
    ]
    opponents: list[list[Policy]]

    if config.opponent in {"self", "mix", "pool"}:
        # Snapshot the learner *with its full second-input config* (seq/event):
        # an explicitly rebuilt MLP loses the encoders and the strict
        # ``load_state_dict`` refresh below would raise.  ``deepcopy`` also
        # guarantees the frozen layout can never diverge from the learner's.
        frozen = NeuralPolicy(
            copy.deepcopy(agent),
            rules,
            device=device,
            sample=(
                config.pool_sample
                if config.opponent == "pool"
                else config.self_play_sample
            ),
            seed=seed,
        )
        assert history_layout(frozen.agent) == history_layout(agent), (
            "frozen self-play opponent history layout diverged from the learner: "
            f"{history_layout(frozen.agent)!r} vs {history_layout(agent)!r}"
        )
        frozen.agent.load_state_dict(agent.state_dict())
        if config.opponent == "mix":
            scripted = make_scripted_policies("random", rules, seed=seed)[0]
            mix_members: list[tuple[float, Policy, str]] = [
                (1.0 - config.mix_random_prob, frozen, "self"),
                (config.mix_random_prob, scripted, "random"),
            ]
            opponents = _mixture_opponents(
                config, rules, mix_members, episode_mixtures, seed
            )
        elif config.opponent == "pool":
            if not config.pool_member:
                raise SystemExit("--opponent pool needs at least one --pool-member")
            specs = [parse_pool_member(raw)[1] for raw in config.pool_member]
            member_ids = pool_member_ids(specs)
            members: list[tuple[float, Policy, str]] = []
            for index, raw in enumerate(config.pool_member):
                weight, spec = parse_pool_member(raw)
                if spec == "self":
                    member: Policy = frozen
                else:
                    member = policy_from_spec(
                        spec,
                        rules,
                        seed=seed + index,
                        device=str(device),
                        sample=config.pool_sample,
                    )
                members.append((weight, member, member_ids[index]))
            opponents = _mixture_opponents(
                config, rules, members, episode_mixtures, seed
            )
        else:
            # One frozen instance shared by every seat and env: refreshing its
            # agent in place publishes the new snapshot without rebuilding envs.
            opponents = [
                [frozen] * rules.num_players for _ in range(config.num_envs)
            ]
    else:
        opponents = [
            make_scripted_policies(config.opponent, rules, seed=seed + idx)
            for idx in range(config.num_envs)
        ]

    return League(
        opponents=opponents,
        episode_mixtures=episode_mixtures,
        member_ids=member_ids,
        frozen=frozen,
    )


class PfspController:
    """Cumulative per-member win/draw/loss from the learner's seat + weight updates.

    ``record`` feeds one observed game against a member.  ``episodes`` counts
    the finished episodes observed (the old ``pfsp_episodes``): each call counts
    one episode by default, and callers recording several seat games from one
    finished episode pass ``episode=False`` for the extra seats.  ``maybe_update``
    returns the PFSP weights once ``every`` further episodes have been recorded,
    and ``None`` otherwise.  The weight math stays in
    :func:`seven523.policies.pfsp_weights`.
    """

    def __init__(
        self,
        member_ids: Sequence[str],
        *,
        prior: float,
        epsilon: float,
        uniform_mix: float,
        every: int,
    ) -> None:
        self.member_ids = list(member_ids)
        self.prior = float(prior)
        self.epsilon = float(epsilon)
        self.uniform_mix = float(uniform_mix)
        self.every = int(every)
        self.records: dict[str, list[int]] = {
            member_id: [0, 0, 0] for member_id in self.member_ids
        }
        self.episodes = 0
        self.updates = 0
        self._last_update = 0

    def record(self, member_id: str, outcome: int, *, episode: bool = True) -> None:
        """Add one result (``+1`` win / ``0`` draw / ``-1`` loss) for ``member_id``."""
        record = self.records.setdefault(member_id, [0, 0, 0])
        if outcome > 0:
            record[0] += 1
        elif outcome < 0:
            record[2] += 1
        else:
            record[1] += 1
        if episode:
            self.episodes += 1

    def win_rate(self, member_id: str) -> float:
        """Beta-shrunk learner win rate for ``member_id`` (draws count half)."""
        wins, draws, losses = self.records[member_id]
        denominator = self.prior + wins + draws + losses
        if not denominator:
            return 0.5
        return (0.5 * self.prior + wins + 0.5 * draws) / denominator

    def maybe_update(self) -> dict[str, float] | None:
        """Return ``{member_id: weight}`` when due, else ``None``.

        An update is due once ``every`` further episodes have been recorded
        since the last one; when it fires it advances the update counter and
        the last-count baseline.
        """
        if self.episodes - self._last_update < self.every:
            return None
        self._last_update = self.episodes
        self.updates += 1
        return pfsp_weights(
            {
                member_id: (float(wins), float(draws), float(losses))
                for member_id, (wins, draws, losses) in self.records.items()
            },
            prior=self.prior,
            epsilon=self.epsilon,
            uniform_mix=self.uniform_mix,
        )

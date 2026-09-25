"""Opponent policies — the real seam (scripted bots today, neural policy later).

Every policy returns a joint action ``(template_id, suit | None)``.  Scripted
bots always pass ``suit=None`` (the engine then uses the strongest suits); only
:class:`~seven523.networks.NeuralPolicy` searches the suit head (ADR-0004).
"""
from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Protocol

from .actions import catalog_for, index_hand, legal_ids, resolve_indexed
from .game import View
from .rules import DEFAULT_RULES, Rules

#: ``(template_id, suit | None)``.
JointAction = tuple[int, int | None]


class Policy(Protocol):
    def act(self, view: View) -> JointAction: ...


class RandomBot:
    """Uniform over the legal templates; strongest-suit realisation."""

    def __init__(self, rng: random.Random | None = None) -> None:
        self.rng = rng or random.Random()

    def act(self, view: View) -> JointAction:
        return self.rng.choice(legal_ids(view.mask)), None


class GreedyBot:
    """Weakest legal beat, non-bombs first; PASS only when nothing else is legal."""

    def __init__(self, rules: Rules = DEFAULT_RULES) -> None:
        self.rules = rules
        self.catalog = catalog_for(rules)
        self.pass_id = len(self.catalog) - 1

    def act(self, view: View) -> JointAction:
        best_id = self.pass_id
        best_key: tuple | None = None
        by_rank = index_hand(view.hand)
        for action_id in legal_ids(view.mask):
            if action_id == self.pass_id:
                continue
            combo = resolve_indexed(self.catalog[action_id], by_rank, self.rules)
            if combo is None:
                continue
            key = (combo.is_bomb, combo.strength)
            if best_key is None or key < best_key:
                best_key = key
                best_id = action_id
        return best_id, None


class MixturePolicy:
    """Weighted mixture of policies, re-drawn per decision (a tiny league).

    The prior ladder report (``docs/experiments/ladder-report.md`` §4.1) names
    an opponent pool as the untested route out of the vs-Greedy plateau: a
    single deterministic opponent gives a saturating gradient.  This policy
    keeps two (or more) opponents in play without changing the env or the
    trainer: every call independently draws one member by weight, so the
    learner sees both a scripted anchor and a frozen self snapshot.

    The draw is per *decision*, not per episode, because the environment never
    tells a policy when an episode starts.  Each instance owns its RNG, so
    runs stay reproducible from ``seed``.
    """

    def __init__(
        self, members: list[tuple[float, Policy]], rng: random.Random | None = None
    ) -> None:
        if not members:
            raise ValueError("MixturePolicy needs at least one member")
        if any(weight < 0 for weight, _ in members):
            raise ValueError("MixturePolicy weights must be non-negative")
        if sum(weight for weight, _ in members) <= 0:
            raise ValueError("MixturePolicy needs a positive total weight")
        self.members = list(members)
        self.rng = rng or random.Random()

    def act(self, view: View) -> JointAction:
        total = sum(weight for weight, _ in self.members)
        roll = self.rng.random() * total
        acc = 0.0
        for weight, policy in self.members:
            acc += weight
            if roll <= acc:
                return policy.act(view)
        return self.members[-1][1].act(view)


class EpisodeMixturePolicy:
    """Weighted league mixture drawn once per episode (T5, direction C).

    Unlike :class:`MixturePolicy`, which re-draws a member on every decision,
    this policy freezes one member for the whole episode: ``start_episode``
    samples once by the current weights and ``act`` forwards to that member
    until the next ``start_episode``.  A stable opponent identity is what makes
    per-member win rates (and therefore PFSP re-weighting) meaningful.

    Members are ``(weight, policy, member_id)`` triples; two-tuples are also
    accepted and get ``"0"``, ``"1"``, ... as ids.  ``current_id`` is the
    member of the episode in play; ``finished_id`` is the member of the episode
    that just ended, set when the *next* ``start_episode`` runs.  Under the
    trainer's SAME_STEP auto-reset the finished member is therefore readable at
    the terminal step, before the next member is drawn.

    :meth:`set_weights` replaces the weights used by future draws (PFSP); it
    never changes the member already frozen for the episode in progress.  With
    weights held at their initial values the draw is per-episode uniformly, or
    by the static league weights, exactly as ``MixturePolicy`` is per-decision.
    """

    def __init__(
        self,
        members: Sequence[tuple],
        rng: random.Random | None = None,
    ) -> None:
        if not members:
            raise ValueError("EpisodeMixturePolicy needs at least one member")
        parsed: list[tuple[float, Policy, str]] = []
        for index, member in enumerate(members):
            if len(member) == 3:
                weight, policy, member_id = member
            elif len(member) == 2:
                weight, policy = member
                member_id = str(index)
            else:
                raise ValueError(
                    "EpisodeMixturePolicy members must be "
                    "(weight, policy, id) or (weight, policy)"
                )
            weight = float(weight)
            if weight < 0:
                raise ValueError("EpisodeMixturePolicy weights must be non-negative")
            parsed.append((weight, policy, str(member_id)))
        if sum(weight for weight, _, _ in parsed) <= 0:
            raise ValueError("EpisodeMixturePolicy needs a positive total weight")
        self.members = parsed
        self.rng = rng or random.Random()
        self._weights = [weight for weight, _, _ in parsed]
        self._current: Policy | None = None
        self.current_id: str | None = None
        self.finished_id: str | None = None

    @property
    def weights(self) -> list[float]:
        """The weights future ``start_episode`` draws will use."""
        return list(self._weights)

    def set_weights(self, weights: Sequence[float]) -> None:
        """Replace the draw distribution (length must match ``members``)."""
        if len(weights) != len(self.members):
            raise ValueError(
                f"expected {len(self.members)} weights, got {len(weights)}"
            )
        weights = [float(weight) for weight in weights]
        if any(weight < 0 for weight in weights):
            raise ValueError("EpisodeMixturePolicy weights must be non-negative")
        if sum(weights) <= 0:
            raise ValueError("EpisodeMixturePolicy needs a positive total weight")
        self._weights = weights

    def start_episode(self) -> str:
        """Freeze one member for the episode and return its id."""
        self.finished_id = self.current_id
        total = sum(self._weights)
        roll = self.rng.random() * total
        acc = 0.0
        chosen = len(self.members) - 1
        for index, weight in enumerate(self._weights):
            acc += weight
            if roll <= acc:
                chosen = index
                break
        weight, policy, member_id = self.members[chosen]
        self._current = policy
        self.current_id = member_id
        return member_id

    def act(self, view: View) -> JointAction:
        if self._current is None:
            raise RuntimeError(
                "EpisodeMixturePolicy.act() needs start_episode() first"
            )
        return self._current.act(view)


def pfsp_weights(
    records: Mapping[str, tuple[float, float, float]],
    *,
    prior: float = 10.0,
    epsilon: float = 0.02,
    uniform_mix: float = 0.5,
) -> dict[str, float]:
    """PFSP sampling distribution from per-member learner records.

    ``records`` maps a member id to ``(wins, draws, losses)`` *from the
    learner's seat*; draws count as half a win.  The observed win rate is first
    shrunk toward 0.5 by a ``Beta(prior/2, prior/2)`` prior, then weighted as
    ``(1 - wr)**2 + epsilon`` so that members the learner currently loses to are
    sampled more often (prioritized fictitious self-play).  Finally the
    distribution is mixed with uniform by ``uniform_mix`` (1.0 = pure uniform,
    0.0 = pure PFSP) to limit forgetting.  The return value sums to 1.
    """
    if not records:
        raise ValueError("pfsp_weights needs at least one member")
    if prior < 0:
        raise ValueError("pfsp_weights prior must be non-negative")
    if epsilon < 0:
        raise ValueError("pfsp_weights epsilon must be non-negative")
    if not 0.0 <= uniform_mix <= 1.0:
        raise ValueError("pfsp_weights uniform_mix must be in [0, 1]")

    scores: dict[str, float] = {}
    total = 0.0
    for member_id, (wins, draws, losses) in records.items():
        games = wins + draws + losses
        credits = wins + 0.5 * draws
        denominator = prior + games
        win_rate = (0.5 * prior + credits) / denominator if denominator else 0.5
        score = (1.0 - win_rate) ** 2 + epsilon
        scores[member_id] = score
        total += score
    uniform = 1.0 / len(records)
    return {
        member_id: (1.0 - uniform_mix) * (score / total)
        + uniform_mix * uniform
        for member_id, score in scores.items()
    }


def make_scripted_policies(
    mode: str, rules: Rules = DEFAULT_RULES, seed: int | None = None
) -> list[Policy]:
    """One opponent policy per seat for the built-in modes ('random' / 'greedy')."""
    if mode == "random":
        rng = random.Random(seed)
        return [
            RandomBot(random.Random(rng.randrange(1 << 32)))
            for _ in range(rules.num_players)
        ]
    if mode == "greedy":
        return [GreedyBot(rules) for _ in range(rules.num_players)]
    raise ValueError(f"unknown scripted opponent mode: {mode!r}")


_AGENT_CACHE: dict[tuple[str, str], object] = {}


def policy_from_spec(
    spec: str,
    rules: Rules = DEFAULT_RULES,
    seed: int | None = None,
    device: str = "cpu",
) -> Policy:
    """Build a policy from the shared ``random`` / ``greedy`` / ``ckpt:<path>`` grammar.

    The single spec parser for the tracing/rating tools (ADR-0006).  Torch is
    imported lazily inside the ``ckpt:`` branch, so the default import stays
    torch-free; agent weights are cached per ``(path, device)``.
    """
    if spec == "random":
        return RandomBot(random.Random(seed))
    if spec == "greedy":
        return GreedyBot(rules)
    if spec.startswith("ckpt:"):
        from .networks import NeuralPolicy, load_agent  # lazy: torch is train-only

        path = spec[len("ckpt:") :]
        if not path:
            raise ValueError("ckpt: spec needs a checkpoint path")
        key = (path, device)
        if key not in _AGENT_CACHE:
            agent, _ = load_agent(path, device=device)
            _AGENT_CACHE[key] = agent
        return NeuralPolicy(  # type: ignore[arg-type]
            _AGENT_CACHE[key], rules, device=device, seed=seed
        )
    raise ValueError(
        f"unknown policy spec {spec!r} (want random / greedy / ckpt:<path>)"
    )

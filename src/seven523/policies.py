"""Opponent policies — the real seam (scripted bots today, neural policy later).

Every policy returns a joint action ``(template_id, suit | None)``.  Scripted
bots always pass ``suit=None`` (the engine then uses the strongest suits); only
:class:`~seven523.networks.NeuralPolicy` searches the suit head (ADR-0004).
"""
from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

from .actions import legal_ids
from .game import View
from .rules import DEFAULT_RULES, Rules

__all__ = [
    "EpisodeMixturePolicy",
    "EpisodePolicy",
    "JointAction",
    "MixturePolicy",
    "Policy",
    "RandomBot",
    "WeightedPolicy",
    "buildable_by_grammar",
    "default_id_for_spec",
    "make_scripted_policies",
    "missing_ckpt_path",
    "pfsp_weights",
    "policy_from_spec",
    "split_entrant",
    "validate_spec",
]

#: ``(template_id, suit | None)``.
JointAction = tuple[int, int | None]


class Policy(Protocol):
    def act(self, view: View) -> JointAction: ...


@runtime_checkable
class EpisodePolicy(Protocol):
    def start_episode(self) -> str: ...
    """Draw the opponent for the episode; called once per env reset."""


@runtime_checkable
class WeightedPolicy(Protocol):
    def set_weights(self, weights: Sequence[float]) -> None: ...
    current_id: str | None
    finished_id: str | None


class RandomBot:
    """Uniform over the legal templates; strongest-suit realisation."""

    def __init__(self, rng: random.Random | None = None) -> None:
        self.rng = rng or random.Random()

    def act(self, view: View) -> JointAction:
        return self.rng.choice(legal_ids(view.mask)), None


class MixturePolicy:
    """Weighted mixture of policies, re-drawn per decision (a tiny league).

    The prior ladder report (``docs/experiments/ladder-report.md`` §4.1) names
    an opponent pool as the untested route out of the single-scripted-opponent
    plateau: a
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
    """One opponent policy per seat for the built-in scripted mode ('random')."""
    if mode == "random":
        rng = random.Random(seed)
        return [
            RandomBot(random.Random(rng.randrange(1 << 32)))
            for _ in range(rules.num_players)
        ]
    raise ValueError(f"unknown scripted opponent mode: {mode!r}")


_AGENT_CACHE: dict[tuple[str, str], object] = {}


def policy_from_spec(
    spec: str,
    rules: Rules = DEFAULT_RULES,
    seed: int | None = None,
    device: str = "cpu",
    *,
    sample: bool = False,
) -> Policy:
    """Build a policy from the shared ``random`` / ``ckpt:<path>`` grammar.

    The single spec parser for the tracing/rating tools (ADR-0006/0012).  Torch
    is imported lazily inside the ``ckpt:`` branch, so the default import stays
    torch-free; agent weights are cached per ``(path, device)``.  ``sample``
    only affects ``ckpt:`` policies (``NeuralPolicy`` argmax by default); the
    training league passes the pool's sampling default through it, while every
    evaluation/rating caller keeps the deterministic argmax.
    """
    if spec == "random":
        return RandomBot(random.Random(seed))
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
            _AGENT_CACHE[key], rules, device=device, seed=seed, sample=sample
        )
    raise ValueError(
        f"unknown policy spec {spec!r} (want random / ckpt:<path>)"
    )


def default_id_for_spec(spec: str) -> str:
    """Auto-id for a bare spec: ``ckpt:<path>`` -> parent dir name (fallback
    stem, then ``'ckpt'``); any other spec is its own id."""
    if spec.startswith("ckpt:"):
        path = Path(spec[len("ckpt:") :])
        return path.parent.name or path.stem or "ckpt"
    return spec


def split_entrant(raw: str) -> tuple[str, str]:
    """``'ID=SPEC'`` -> ``(ID, SPEC)`` stripped; a bare spec gets an auto-id."""
    if "=" in raw:
        id_, spec = raw.split("=", 1)
        return id_.strip(), spec.strip()
    spec = raw.strip()
    return default_id_for_spec(spec), spec


def missing_ckpt_path(spec: str | None) -> str | None:
    """The path of a ``ckpt:`` spec when the file is absent (or the path is
    empty), else ``None``.  ``random`` / other / ``None`` -> ``None``."""
    if spec is None or not spec.startswith("ckpt:"):
        return None
    path = spec[len("ckpt:") :]
    if path and Path(path).is_file():
        return None
    return path


def buildable_by_grammar(spec: str | None) -> bool:
    """Whether the stock ``random`` / ``ckpt:<path>`` grammar can build ``spec``.

    The caller-owned capability counterpart of :func:`validate_spec` for the
    placement/web admission gates: ``None`` (a spec-less subject) is not
    buildable, and anything outside the stock grammar (e.g. a run-local
    ``rolloutt:`` search spec) needs an injected factory.
    """
    if spec is None:
        return False
    return validate_spec(str(spec)) is None


def validate_spec(spec: str) -> str | None:
    """Return an error message when the spec is not usable, else ``None``.

    The spec grammar is ``random`` / ``ckpt:<path>``; the messages are kept
    byte-identical to those the tools raised before this lived here.
    """
    if spec == "random":
        return None
    if spec.startswith("ckpt:"):
        path = spec[len("ckpt:") :]
        if not path:
            return "ckpt: spec needs a checkpoint path"
        if not Path(path).is_file():
            return f"checkpoint not found: {path}"
        return None
    return f"unknown policy spec {spec!r} (want random / ckpt:<path>)"

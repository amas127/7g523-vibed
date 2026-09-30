"""Result-based rating: the pure OpenSkill rating core (ADR-0011).

One :class:`PlayedGame` is one 牌局 (发牌 → 撬底) and that is the only unit this
module observes: 一墩 results and per-decision statistics never enter, so
statistics cluster by game by construction.  The module is pure in-process
math — no torch, no numpy, no paths, no RNG, no clock — which keeps it testable
without the training stack and lets every caller cross the same seam: the M2
ladder builder, the arena, and the D3 human-placement estimator.

The estimator is the standard OpenSkill Weng–Lin **Plackett–Luce** Gaussian
model (:class:`openskill.models.PlackettLuce`): every entrant carries a Gaussian
rating ``(mu, sigma)`` and :func:`fit_ratings` replays the games in
chronological order, applying the model's online update once per 牌局.  A
pinned anchor is a fixed ``mu`` with ``sigma = 0``; an id's :class:`Prior` is
its initial Gaussian; ``tau`` (openskill's dynamics term) keeps ``sigma`` from
collapsing, which replaces the old per-id result window.

Ratings are reported on the project's 0-based scale — openskill's default
``sigma = 25/3`` scaled by 24, ``beta = sigma / 2``, ``tau = sigma / 100``, with
the pinned RandomBot at ``mu = 0`` (ADR-0012) — so anchors, rung spacing and the
placement thresholds keep their numbers.  The probability model is the Gaussian
one from openskill, not the old 400-point logistic: absolute values and CIs are
not comparable to the retired Bradley–Terry MAP and must be re-measured
(ADR-0011, T15).
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from openskill.models import PlackettLuce, PlackettLuceRating

__all__ = [
    "DEFAULT_BETA",
    "DEFAULT_ELO_SCALE",
    "DEFAULT_MU",
    "DEFAULT_PRIOR",
    "DEFAULT_SIGMA",
    "DEFAULT_TAU",
    "Fit",
    "FitConfig",
    "Gaussian",
    "PlayedGame",
    "Prior",
    "Rating",
    "Rung",
    "RungSelection",
    "expected_score",
    "fit_ratings",
    "select_rungs",
]

#: The project's rating scale: openskill's default model on the historical
#: 1500/200 axis, shifted so the pinned RandomBot gauge is 0 (ADR-0012:
#: the pre-2026-09-25 scale pinned random at 1000, hence the -1000 translation).
DEFAULT_MU = 500.0
DEFAULT_SIGMA = 200.0

#: OpenSkill defaults, expressed on that scale (``beta = sigma / 2``,
#: ``tau = sigma / 100``).
DEFAULT_BETA = DEFAULT_SIGMA / 2.0
DEFAULT_TAU = DEFAULT_SIGMA / 100.0

#: The retired 400-point logistic scale, kept only for the reporting helpers
#: that express a win probability as an Elo difference (``duel``) or as one
#: Bernoulli observation's SE (``prior``); the estimator does not use it.
DEFAULT_ELO_SCALE = 400.0


@dataclass(frozen=True, slots=True)
class Prior:
    """Gaussian prior on a free entrant's rating: ``(mu, sigma)``."""

    mu: float = DEFAULT_MU
    sigma: float = DEFAULT_SIGMA

    def __post_init__(self) -> None:
        if not (math.isfinite(self.mu) and math.isfinite(self.sigma)):
            raise ValueError("prior mu/sigma must be finite")
        if self.sigma <= 0.0:
            raise ValueError(f"prior sigma must be positive, got {self.sigma}")


DEFAULT_PRIOR = Prior()


@runtime_checkable
class Gaussian(Protocol):
    """A Gaussian rating shape: anything carrying ``mu``/``sigma``.

    :class:`Prior`, :class:`Rating` and the placement package's ``Opponent``
    all satisfy it structurally, which is what :func:`expected_score` needs.
    """

    mu: float
    sigma: float


@dataclass(frozen=True, slots=True)
class PlayedGame:
    """One 牌局 as the estimator sees it: who sat where and the final 分数.

    ``seats`` and ``scores`` are parallel; ids must be distinct.  A game may
    hold any number of seats; :func:`fit_ratings` rates all of them with the
    multiplayer Plackett–Luce update, ties (equal 分数) included.
    """

    seed: int
    seats: tuple[str, ...]
    scores: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.seats) != len(self.scores):
            raise ValueError("seats and scores must be parallel")
        if len(self.seats) < 2:
            raise ValueError("a 对局 needs at least two seats")
        if len(set(self.seats)) != len(self.seats):
            raise ValueError(f"seat ids must be distinct, got {self.seats!r}")
        if any(not isinstance(score, int) for score in self.scores):
            raise ValueError("scores must be ints")
        if any(score < 0 for score in self.scores):
            raise ValueError("scores must be non-negative")


@dataclass(frozen=True, slots=True)
class FitConfig:
    """The OpenSkill model constants used to replay a game record.

    ``prior`` is the initial rating for ids with no explicit :class:`Prior`;
    ``beta`` is the performance-noise sd and ``tau`` the additive dynamics
    term that keeps ``sigma`` from collapsing (both default to openskill's
    ``sigma / 2`` and ``sigma / 100``).  A positive ``tau`` is required: a
    zero dynamics term makes the standard update degenerate after enough games.
    """

    prior: Prior = DEFAULT_PRIOR
    beta: float = DEFAULT_BETA
    tau: float = DEFAULT_TAU

    def __post_init__(self) -> None:
        if not isinstance(self.prior, Prior):
            raise ValueError(f"prior must be a Prior, got {self.prior!r}")
        for name, value in (("beta", self.beta), ("tau", self.tau)):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite, got {value!r}")


@dataclass(frozen=True, slots=True)
class Rating:
    """A fitted Gaussian rating: ``mu ± sigma`` over ``n`` 牌局.

    Anchors are exact: they come back with ``mu`` as pinned and ``sigma == 0``.
    ``sigma`` is the OpenSkill uncertainty (not the old Bradley–Terry Hessian
    SE); lower means better determined.
    """

    mu: float
    sigma: float
    n: int


@dataclass(frozen=True, slots=True)
class Fit:
    """The outcome of :func:`fit_ratings` (``ratings`` covers anchors too)."""

    ratings: dict[str, Rating]
    games: int


def _model(config: FitConfig) -> PlackettLuce:
    """The openskill model for one fit (``mu``/``sigma`` seed the defaults)."""
    return PlackettLuce(
        mu=config.prior.mu,
        sigma=config.prior.sigma,
        beta=config.beta,
        tau=config.tau,
    )


def expected_score(
    rating_a: Gaussian,
    rating_b: Gaussian,
    *,
    config: FitConfig = FitConfig(),
) -> float:
    """``P(a beats b)`` under the Plackett–Luce Gaussian model.

    Uses each side's ``(mu, sigma)``, so an opponent with a wide prior is
    genuinely less predictable; a pinned anchor's ``sigma == 0`` makes it a
    known quantity.  This is the model's own probability, which the D3
    information-optimal pairing maximizes ``p * (1 - p)`` of.
    """
    model = _model(config)
    return float(
        model.predict_win(
            [
                [model.rating(mu=rating_a.mu, sigma=rating_a.sigma)],
                [model.rating(mu=rating_b.mu, sigma=rating_b.sigma)],
            ]
        )[0]
    )


def fit_ratings(
    games: Iterable[PlayedGame],
    *,
    anchors: Mapping[str, float],
    priors: Mapping[str, Prior] | None = None,
    config: FitConfig = FitConfig(),
) -> Fit:
    """Replay ``games`` (chronological) through the Plackett–Luce update.

    Every id seen in ``games`` is rated; ids in ``anchors`` are pinned (their
    ``mu`` is returned exactly as given with ``sigma == 0`` and they never
    update).  Ids in ``priors`` with no games still get their prior back
    (``n == 0``), which is how D3's trace prior enters; any other free id
    starts from ``config.prior``.  The update is order-dependent: ``games`` is
    the history, and later 牌局 carry more weight through the dynamics term.
    Errors: ``ValueError`` for a non-finite anchor, a prior for a pinned
    anchor, or an invalid ``config``/``prior``.
    """
    games = tuple(games)
    anchors = dict(anchors)
    for id_, value in anchors.items():
        if not math.isfinite(value):
            raise ValueError(f"anchor {id_!r} rating must be finite, got {value!r}")
    priors = dict(priors or {})
    for id_, prior in priors.items():
        if not isinstance(prior, Prior):
            raise ValueError(f"prior for {id_!r} must be a Prior, got {prior!r}")
        if id_ in anchors:
            # Anchors win; a prior there is a caller mistake worth surfacing early.
            raise ValueError("priors must not target pinned anchors")
    if not isinstance(config, FitConfig):
        raise ValueError(f"config must be a FitConfig, got {config!r}")

    model = _model(config)
    counts: dict[str, int] = {}
    state: dict[str, PlackettLuceRating] = {}

    def initial(id_: str) -> PlackettLuceRating:
        if id_ in anchors:
            return model.rating(mu=anchors[id_], sigma=0.0, name=id_)
        prior = priors.get(id_, config.prior)
        return model.rating(mu=prior.mu, sigma=prior.sigma, name=id_)

    for id_ in set(anchors) | set(priors):
        state[id_] = initial(id_)

    for game in games:
        for id_ in game.seats:
            if id_ not in state:
                state[id_] = initial(id_)
        teams = [[state[id_]] for id_ in game.seats]
        updated = model.rate(teams, scores=[float(score) for score in game.scores])
        for seat, id_ in enumerate(game.seats):
            counts[id_] = counts.get(id_, 0) + 1
            if id_ not in anchors:
                state[id_] = updated[seat][0]

    ratings = {
        id_: Rating(
            mu=rating.mu,
            sigma=0.0 if id_ in anchors else rating.sigma,
            n=counts.get(id_, 0),
        )
        for id_, rating in state.items()
    }
    return Fit({id_: ratings[id_] for id_ in sorted(ratings)}, len(games))


@dataclass(frozen=True, slots=True)
class Rung:
    """One chosen ladder level."""

    id: str
    mu: float
    sigma: float


@dataclass(frozen=True, slots=True)
class RungSelection:
    """A best-effort rung pick; ``ok`` states whether the spacing request held.

    ``wide_gaps`` are adjacent selected rungs farther apart than
    ``max_spacing``; ``tail_gap`` is the rating distance from the last rung to
    the highest candidate.  Nothing is silently relaxed: callers report both.
    """

    rungs: tuple[Rung, ...]
    requested: int
    min_spacing: float
    max_spacing: float
    wide_gaps: tuple[tuple[str, str, float], ...]
    tail_gap: float
    ok: bool


def select_rungs(
    ratings: Mapping[str, Rating],
    *,
    count: int = 5,
    min_spacing: float = 100.0,
    max_spacing: float = 150.0,
) -> RungSelection:
    """Pick up to ``count`` rungs from a pool of fitted ratings, lowest first.

    Deterministic: sort by rating mean then id; start at the lowest; repeatedly
    take the lowest id at least ``min_spacing`` above the current rung.  A
    shortfall (fewer candidates than ``count``) and any spacing hole are
    reported, never hidden.
    """
    if count < 1:
        raise ValueError("count must be at least 1")
    if not (0.0 < min_spacing <= max_spacing):
        raise ValueError("require 0 < min_spacing <= max_spacing")
    ordered = sorted(
        (Rung(id_, float(rating.mu), float(rating.sigma)) for id_, rating in ratings.items()),
        key=lambda rung: (rung.mu, rung.id),
    )
    if not ordered:
        return RungSelection((), count, min_spacing, max_spacing, (), 0.0, False)
    selected = [ordered[0]]
    for rung in ordered[1:]:
        if len(selected) >= count:
            break
        if rung.mu - selected[-1].mu >= min_spacing:
            selected.append(rung)
    wide_gaps = tuple(
        (selected[i].id, selected[i + 1].id, selected[i + 1].mu - selected[i].mu)
        for i in range(len(selected) - 1)
        if selected[i + 1].mu - selected[i].mu > max_spacing
    )
    tail_gap = ordered[-1].mu - selected[-1].mu
    ok = (
        len(selected) >= 2
        and len(selected) == count
        and not wide_gaps
        and tail_gap <= max_spacing
    )
    return RungSelection(
        tuple(selected), count, min_spacing, max_spacing, wide_gaps, tail_gap, ok
    )

"""Result-based Elo estimation: the pure rating core (ADR-0006).

One :class:`PlayedGame` is one 牌局 (发牌 → 撬底) and that is the only unit this
module observes: 一墩 results and per-decision statistics never enter, so
statistics cluster by game by construction.  The module is pure in-process
math — no torch, no numpy, no paths, no RNG, no clock — which keeps it testable
without the training stack and lets every caller cross the same seam: the M2
ladder builder today, the D3 hybrid estimator (window + trace prior) later.

The estimator is a Bradley-Terry MAP.  A free entrant's rating solves

``p = 1 / (1 + 10 ** ((R_opponent - R) / 400))``

with a Gaussian prior on ``R`` (``Prior``), damped-Newton coordinate sweeps (step
cap ``FitConfig.step_cap``, rating clamped to ``[rating_min, rating_max]``), and
an optional Gaussian 分差 term ``N(c * (R - R_opponent), sigma ** 2)``.  Anchors
are pinned: their rating is a constant and their ``se`` is exactly 0.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping

__all__ = [
    "DEFAULT_ELO_SCALE",
    "DEFAULT_PRIOR",
    "Fit",
    "FitConfig",
    "PlayedGame",
    "Prior",
    "Rating",
    "Rung",
    "RungSelection",
    "expected_score",
    "fit_ratings",
    "select_rungs",
]

DEFAULT_ELO_SCALE = 400.0
BETA_SCALE = math.log(10.0)


@dataclass(frozen=True, slots=True)
class Prior:
    """Gaussian prior on a free entrant's rating."""

    mean: float = 1500.0
    sd: float = 200.0

    def __post_init__(self) -> None:
        if not (math.isfinite(self.mean) and math.isfinite(self.sd)):
            raise ValueError("prior mean/sd must be finite")
        if self.sd <= 0.0:
            raise ValueError(f"prior sd must be positive, got {self.sd}")


DEFAULT_PRIOR = Prior()


@dataclass(frozen=True, slots=True)
class PlayedGame:
    """One 牌局 as the estimator sees it: who sat where and the final 分数.

    ``seats`` and ``scores`` are parallel; ids must be distinct.  A game may
    hold any number of seats, but :func:`fit_ratings` currently supports exactly
    two (pairwise Bradley-Terry); the shape is N-agnostic so multi-seat support
    does not force an interface change later.
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
    """Estimator constants and the optional 分差 (margin) channel."""

    elo_scale: float = DEFAULT_ELO_SCALE
    prior: Prior = DEFAULT_PRIOR
    step_cap: float = 300.0
    rating_min: float = 400.0
    rating_max: float = 2600.0
    max_iter: int = 200
    tol: float = 1e-7
    margin: tuple[float, float] | None = None  # (c, sigma)

    def __post_init__(self) -> None:
        if not math.isfinite(self.elo_scale) or self.elo_scale <= 0.0:
            raise ValueError("elo_scale must be positive and finite")
        if not math.isfinite(self.step_cap) or self.step_cap <= 0.0:
            raise ValueError("step_cap must be positive and finite")
        if not self.rating_min < self.rating_max:
            raise ValueError("rating_min must be below rating_max")
        if self.max_iter < 1:
            raise ValueError("max_iter must be at least 1")
        if self.tol <= 0.0:
            raise ValueError("tol must be positive")
        if self.margin is not None:
            c, sigma = self.margin
            if not (math.isfinite(c) and c > 0.0):
                raise ValueError("margin c must be positive and finite")
            if not (math.isfinite(sigma) and sigma > 0.0):
                raise ValueError("margin sigma must be positive and finite")


@dataclass(frozen=True, slots=True)
class Rating:
    """A fitted rating: ``elo ± se`` over ``n`` 牌局 (anchors have ``se == 0``)."""

    elo: float
    se: float
    n: int


@dataclass(frozen=True, slots=True)
class Fit:
    """The outcome of :func:`fit_ratings` (``ratings`` covers anchors too)."""

    ratings: dict[str, Rating]
    iterations: int
    converged: bool


def expected_score(
    rating_a: float, rating_b: float, *, scale: float = DEFAULT_ELO_SCALE
) -> float:
    """``P(a beats b)`` under the 400-point logistic scale."""
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be positive and finite")
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / scale))


def _terms_for(
    game: PlayedGame,
) -> dict[str, tuple[str, float, float]]:
    """Per-seat ``(opponent, score, own-minus-opponent 分差)`` of one 对局."""
    own_scores = game.scores
    terms: dict[str, tuple[str, float, float]] = {}
    for seat, me in enumerate(game.seats):
        other = game.seats[1 - seat]
        mine, theirs = own_scores[seat], own_scores[1 - seat]
        score = 1.0 if mine > theirs else (0.0 if mine < theirs else 0.5)
        terms[me] = (other, score, float(mine - theirs))
    return terms


def _gradient_info(
    rating: float,
    terms: Iterable[tuple[str, float, float]],
    current: Mapping[str, float],
    anchors: Mapping[str, float],
    prior: Prior,
    beta: float,
    margin: tuple[float, float] | None,
    scale: float,
) -> tuple[float, float]:
    """Log-posterior gradient and Fisher information for one free rating."""
    gradient = 0.0
    info = 1.0 / (prior.sd * prior.sd)
    for opponent, score, own_margin in terms:
        opponent_rating = current[opponent] if opponent in current else anchors[opponent]
        p = expected_score(rating, opponent_rating, scale=scale)
        gradient += beta * (score - p)
        info += beta * beta * p * (1.0 - p)
        if margin is not None:
            c, sigma = margin
            residual = own_margin - c * (rating - opponent_rating)
            gradient += c * residual / (sigma * sigma)
            info += c * c / (sigma * sigma)
    gradient -= (rating - prior.mean) / (prior.sd * prior.sd)
    return gradient, info


def _inverse_diagonal(matrix: list[list[float]]) -> list[float]:
    """Diagonal of the inverse of a symmetric positive-definite matrix.

    Pure-Python Cholesky + triangular inversion keeps the rating core
    numpy-free.  The prior precision (``1 / sd**2``) adds to every diagonal
    entry, so every pivot is strictly positive by construction.
    """
    n = len(matrix)
    lower = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            total = matrix[i][j] - sum(
                lower[i][k] * lower[j][k] for k in range(j)
            )
            if i == j:
                if total <= 0.0:
                    raise ValueError(
                        "information matrix is not positive definite; "
                        "check priors and ratings"
                    )
                lower[i][j] = math.sqrt(total)
            else:
                lower[i][j] = total / lower[j][j]
    inverse_lower = [[0.0] * n for _ in range(n)]
    for i in range(n):
        inverse_lower[i][i] = 1.0 / lower[i][i]
        for j in range(i):
            inverse_lower[i][j] = -sum(
                lower[i][k] * inverse_lower[k][j] for k in range(j, i)
            ) / lower[i][i]
    return [
        sum(inverse_lower[k][i] * inverse_lower[k][i] for k in range(n))
        for i in range(n)
    ]


def fit_ratings(
    games: Iterable[PlayedGame],
    *,
    anchors: Mapping[str, float],
    priors: Mapping[str, Prior] | None = None,
    window: int | None = None,
    config: FitConfig = FitConfig(),
) -> Fit:
    """Joint Bradley-Terry MAP over every unpinned id seen in ``games``.

    ``games`` is chronological.  ``window=W`` keeps each free id's last ``W``
    牌局 (``None`` = all history, ``W >= 1``); it is a precision floor, not a
    convergence aid.  Ids in ``anchors`` are pinned: their rating is returned
    exactly as given with ``se == 0`` and any prior for them is ignored.  Ids in
    ``priors`` with no games still get their prior back (``n == 0``), which is
    how D3's trace prior enters.  Errors: ``ValueError`` for seat counts other
    than two, non-finite anchors, or invalid ``window``/``config``.
    """
    if window is not None and window < 1:
        raise ValueError("window must be at least 1 (or None for all history)")
    games = tuple(games)
    for game in games:
        if len(game.seats) != 2:
            raise ValueError(
                "fit_ratings supports two-seat 对局; got "
                f"{len(game.seats)} seats in {game!r}"
            )
    anchors = dict(anchors)
    for id_, value in anchors.items():
        if not math.isfinite(value):
            raise ValueError(f"anchor {id_!r} rating must be finite, got {value!r}")
    priors = dict(priors or {})
    for id_, prior in priors.items():
        if not isinstance(prior, Prior):
            raise ValueError(f"prior for {id_!r} must be a Prior, got {prior!r}")
    if any(id_ in anchors for id_ in priors):
        # Anchors win; a prior there is a caller mistake worth surfacing early.
        raise ValueError("priors must not target pinned anchors")

    occurrences: dict[str, list[int]] = {}
    for index, game in enumerate(games):
        for id_ in game.seats:
            occurrences.setdefault(id_, []).append(index)
    free_ids = sorted((set(occurrences) | set(priors)) - set(anchors))

    retained: dict[str, set[int]] = {}
    for id_ in free_ids:
        indices = occurrences.get(id_, [])
        retained[id_] = set(indices[-window:]) if window is not None else set(indices)

    terms: dict[str, list[tuple[str, float, float]]] = {}
    for id_ in free_ids:
        rows: list[tuple[str, float, float]] = []
        for index in sorted(retained[id_]):
            rows.append(_terms_for(games[index])[id_])
        terms[id_] = rows

    beta = BETA_SCALE / config.elo_scale
    rating = {id_: priors.get(id_, config.prior).mean for id_ in free_ids}
    iterations = 0
    converged = False
    for iteration in range(1, config.max_iter + 1):
        max_step = 0.0
        for id_ in free_ids:
            prior = priors.get(id_, config.prior)
            gradient, info = _gradient_info(
                rating[id_], terms[id_], rating, anchors, prior, beta,
                config.margin, config.elo_scale,
            )
            step = max(-config.step_cap, min(config.step_cap, gradient / info))
            updated = min(config.rating_max, max(config.rating_min, rating[id_] + step))
            max_step = max(max_step, abs(updated - rating[id_]))
            rating[id_] = updated
        iterations = iteration
        if max_step < config.tol:
            converged = True
            break

    ratings: dict[str, Rating] = {}
    index = {id_: position for position, id_ in enumerate(free_ids)}
    information = [[0.0] * len(free_ids) for _ in free_ids]
    for id_ in free_ids:
        prior = priors.get(id_, config.prior)
        information[index[id_]][index[id_]] += 1.0 / (prior.sd * prior.sd)
    for id_ in free_ids:
        i = index[id_]
        for opponent, _score, _margin in terms[id_]:
            opponent_rating = (
                rating[opponent] if opponent in rating else anchors[opponent]
            )
            p = expected_score(rating[id_], opponent_rating, scale=config.elo_scale)
            info = beta * beta * p * (1.0 - p)
            if config.margin is not None:
                c, sigma = config.margin
                info += c * c / (sigma * sigma)
            information[i][i] += info
            if opponent in index:
                # Free-vs-free games couple both ratings; the joint Hessian
                # (not the diagonal Fisher) is what gives the honest SE.  Each
                # game already contributes ``info`` once to each player's own
                # diagonal through that player's term, so only the pair term is
                # added here -- and only once per unordered pair (the double
                # count made every free-vs-free game contribute 2*w and read
                # the SE up to ~19% too small in cross-heavy designs).
                j = index[opponent]
                if i < j:
                    information[i][j] -= info
                    information[j][i] -= info
    inverse_diagonal = _inverse_diagonal(information)
    for id_ in free_ids:
        ratings[id_] = Rating(
            rating[id_], math.sqrt(inverse_diagonal[index[id_]]), len(retained[id_])
        )
    for id_, value in anchors.items():
        ratings[id_] = Rating(value, 0.0, len(occurrences.get(id_, [])))
    return Fit({id_: ratings[id_] for id_ in sorted(ratings)}, iterations, converged)


@dataclass(frozen=True, slots=True)
class Rung:
    """One chosen ladder level."""

    id: str
    elo: float
    se: float


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

    Deterministic: sort by rating then id; start at the lowest; repeatedly take
    the lowest id at least ``min_spacing`` above the current rung.  A shortfall
    (fewer candidates than ``count``) and any spacing hole are reported, never
    hidden.
    """
    if count < 1:
        raise ValueError("count must be at least 1")
    if not (0.0 < min_spacing <= max_spacing):
        raise ValueError("require 0 < min_spacing <= max_spacing")
    ordered = sorted(
        (Rung(id_, float(rating.elo), float(rating.se)) for id_, rating in ratings.items()),
        key=lambda rung: (rung.elo, rung.id),
    )
    if not ordered:
        return RungSelection((), count, min_spacing, max_spacing, (), 0.0, False)
    selected = [ordered[0]]
    for rung in ordered[1:]:
        if len(selected) >= count:
            break
        if rung.elo - selected[-1].elo >= min_spacing:
            selected.append(rung)
    wide_gaps = tuple(
        (selected[i].id, selected[i + 1].id, selected[i + 1].elo - selected[i].elo)
        for i in range(len(selected) - 1)
        if selected[i + 1].elo - selected[i].elo > max_spacing
    )
    tail_gap = ordered[-1].elo - selected[-1].elo
    ok = (
        len(selected) >= 2
        and len(selected) == count
        and not wide_gaps
        and tail_gap <= max_spacing
    )
    return RungSelection(
        tuple(selected), count, min_spacing, max_spacing, wide_gaps, tail_gap, ok
    )

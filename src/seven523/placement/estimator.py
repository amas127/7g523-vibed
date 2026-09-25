"""The D3 placement estimator: shared joint fit, weights and level bands.

Every non-anchor opponent enters the joint fit with a free prior so its label
uncertainty propagates through the Hessian (HR §5.2); the trace and result
channels are weighted by Gaussian precision (HR §5.5).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from ..elo import Fit, FitConfig, PlayedGame, Prior, fit_ratings

if TYPE_CHECKING:
    # Annotation only: ``fit_session`` reads Opponent fields but the estimator
    # must not create a runtime estimator -> opponents dependency.
    from .opponents import Opponent

#: Cold start before any trace exists (HR §5.4: the first-game prior).
COLD_START_PRIOR = Prior(1500.0, 300.0)

#: 分差 likelihood constants, calibrated on the frozen study (HR §5.3):
#: ``margin ~ N(c·(R−R_opp), σ²)``.  With the trace prior present the σ is
#: doubled (HR §7.4: the channels are correlated, σ×2 restores coverage).
MARGIN_C = 0.143
MARGIN_SIGMA = 44.5
MARGIN_TRACE_FACTOR = 2.0

#: HR §5.2: free ladder levels enter the joint fit with a wide source prior
#: (their own label uncertainty); pinned anchors are exact instead.
RUNG_PRIOR_SD = 30.0

#: The result-channel decomposition prior (``FitConfig``'s own default).
RESULT_PRIOR = Prior(1500.0, 200.0)


# -- estimation --------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SessionConfig:
    """Session constants; defaults are the HR §5.3/§5.4 production choices."""

    games: int = 10
    stop_ci: float = 50.0
    min_games_before_stop: int = 1
    explore_games: int = 2
    z: float = 1.96
    margin_c: float = MARGIN_C
    margin_sigma: float = MARGIN_SIGMA
    trace_margin_factor: float = MARGIN_TRACE_FACTOR
    rung_prior_sd: float = RUNG_PRIOR_SD
    seat_start: int = 0
    human_id: str = "human"

    def __post_init__(self) -> None:
        if self.games < 2 or self.games % 2 != 0:
            raise ValueError("games must be an even number >= 2 (5/5 seats)")
        if self.stop_ci <= 0.0:
            raise ValueError("stop_ci must be positive")
        if not 1 <= self.min_games_before_stop <= self.games:
            raise ValueError("min_games_before_stop must be in 1..games")
        if not 0 <= self.explore_games <= self.games:
            raise ValueError("explore_games must be in 0..games")
        if self.z <= 0.0:
            raise ValueError("z must be positive")
        if self.margin_c <= 0.0 or self.margin_sigma <= 0.0:
            raise ValueError("margin constants must be positive")
        if self.trace_margin_factor <= 0.0:
            raise ValueError("trace_margin_factor must be positive")
        if self.rung_prior_sd <= 0.0:
            raise ValueError("rung_prior_sd must be positive")
        if self.seat_start not in (0, 1):
            raise ValueError("seat_start must be 0 or 1")
        if not self.human_id or "@" in self.human_id:
            raise ValueError("human_id must be non-empty and '@'-free")


def session_margin(
    config: SessionConfig, *, with_trace: bool
) -> tuple[float, float]:
    """``(c, σ)``: σ is doubled while the trace prior is in play (HR §7.4)."""
    sigma = config.margin_sigma * (config.trace_margin_factor if with_trace else 1.0)
    return (config.margin_c, sigma)


def fit_session(
    games: Sequence[PlayedGame],
    *,
    human_id: str,
    human_prior: Prior,
    opponents: Sequence[Opponent],
    anchors: Mapping[str, float],
    margin: tuple[float, float],
    rung_prior_sd: float = RUNG_PRIOR_SD,
    rung_centers: Mapping[str, float] | None = None,
) -> Fit:
    """The D3 fit: free human prior + free rung priors + pinned anchors.

    Every non-anchor opponent enters with ``Prior(center, rung_prior_sd)`` so
    its label uncertainty propagates through the joint Hessian (HR §5.2);
    ``window=None`` because the first 10 games are the whole point (HR §5.3).
    """
    priors: dict[str, Prior] = {human_id: human_prior}
    for opponent in opponents:
        if opponent.anchor:
            continue
        center = float((rung_centers or {}).get(opponent.id, opponent.elo))
        priors[opponent.id] = Prior(center, rung_prior_sd)
    return fit_ratings(
        tuple(games),
        anchors=dict(anchors),
        priors=priors,
        window=None,
        config=FitConfig(margin=margin),
    )


def channel_weights(
    trace_sd: float | None, result_se: float
) -> dict[str, float]:
    """Gaussian-precision split between the trace prior and result likelihood.

    The trace channel contributes ``1/σ_traj²``; the result channel contributes
    ``1/se_result²`` from a same-design fit with only the weak default prior.
    The two weights sum to 1 (HR §5.5's "两通道的权重").
    """
    result_precision = 0.0 if result_se <= 0.0 else 1.0 / (result_se * result_se)
    if trace_sd is None:
        return {"trace": 0.0, "result": 1.0}
    prior_precision = 1.0 / (trace_sd * trace_sd)
    total = prior_precision + result_precision
    if total <= 0.0:
        return {"trace": 0.0, "result": 0.0}
    return {"trace": prior_precision / total, "result": result_precision / total}


def nearest_level(
    elo: float, levels: Mapping[str, float]
) -> dict[str, Any] | None:
    """The manifest level closest to ``elo`` (ties by elo then id)."""
    if not levels:
        return None
    id_, value = min(
        levels.items(), key=lambda item: (abs(float(item[1]) - elo), item[1], item[0])
    )
    return {"id": str(id_), "elo": float(value), "distance": abs(float(value) - elo)}


def band_for(ci_half_width: float, *, stop_ci: float = 50.0) -> str:
    """HR §5.5 bands: ``placed`` / ``provisional`` (±1 档) / ``coarse``."""
    if ci_half_width <= stop_ci:
        return "placed"
    if ci_half_width <= 2.0 * stop_ci:
        return "provisional"
    return "coarse"

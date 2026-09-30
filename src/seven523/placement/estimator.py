"""The D3 placement estimator: shared OpenSkill fit, weights and level bands.

Every non-anchor opponent enters the fit with a free prior so its label
uncertainty propagates into the human's posterior; the trace and result
channels are combined by Gaussian precision (HR §5.5).  The estimator itself is
the standard OpenSkill Plackett–Luce update in :mod:`seven523.elo` (ADR-0011).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..elo import Fit, FitConfig, PlayedGame, Prior, fit_ratings

if TYPE_CHECKING:
    # Annotation only: ``fit_session`` reads Opponent fields but the estimator
    # must not create a runtime estimator -> opponents dependency.
    from .opponents import Opponent

#: Cold start before any trace exists (HR §5.4: the first-game prior,
#: translated by -1000 to the RandomBot-0 gauge; ADR-0012 re-derives it).
#: T17 applies the deferred scale factor ``c = 0.4656209850248892`` to the
#: RandomBot-0 constants so they match the published MLE label scale (same
#: least-squares-through-the-origin rule as ``prior.ANCHOR_CENTER``):
#: (500.0, 300.0) -> (232.8104925124446, 139.68629550746675).
COLD_START_PRIOR = Prior(232.8104925124446, 139.68629550746675)

#: HR §5.2: free ladder levels enter the joint fit with a wide source prior
#: (their own label uncertainty); pinned anchors are exact instead.  Rescaled
#: by the same ``c``: 30.0 -> 13.968629550746675.
RUNG_PRIOR_SD = 13.968629550746675

#: The result-channel decomposition prior (``FitConfig``'s own default,
#: translated by -1000 to the RandomBot-0 gauge and rescaled by ``c``):
#: (500.0, 200.0) -> (232.8104925124446, 93.12419700497784).
RESULT_PRIOR = Prior(232.8104925124446, 93.12419700497784)


# -- estimation --------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SessionConfig:
    """Session constants; defaults are the HR §5.3/§5.4 production choices."""

    games: int = 10
    stop_ci: float = 50.0
    min_games_before_stop: int = 1
    explore_games: int = 2
    z: float = 1.96
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
        if self.rung_prior_sd <= 0.0:
            raise ValueError("rung_prior_sd must be positive")
        if self.seat_start not in (0, 1):
            raise ValueError("seat_start must be 0 or 1")
        if not self.human_id or "@" in self.human_id:
            raise ValueError("human_id must be non-empty and '@'-free")


def fit_session(
    games: Sequence[PlayedGame],
    *,
    human_id: str,
    human_prior: Prior,
    opponents: Sequence[Opponent],
    anchors: Mapping[str, float],
    rung_prior_sd: float = RUNG_PRIOR_SD,
    rung_centers: Mapping[str, float] | None = None,
) -> Fit:
    """The D3 fit: free human prior + free rung priors + pinned anchors.

    Every non-anchor opponent starts from ``Prior(center, rung_prior_sd)`` so a
    rung's label uncertainty is a wide starting Gaussian, not a pin; the
    Plackett–Luce replay then moves both sides with every 牌局.
    """
    priors: dict[str, Prior] = {human_id: human_prior}
    for opponent in opponents:
        if opponent.anchor:
            continue
        center = float((rung_centers or {}).get(opponent.id, opponent.mu))
        priors[opponent.id] = Prior(center, rung_prior_sd)
    return fit_ratings(
        tuple(games),
        anchors=dict(anchors),
        priors=priors,
        config=FitConfig(),
    )


def channel_weights(
    trace_sd: float | None, result_sigma: float
) -> dict[str, float]:
    """Gaussian-precision split between the trace prior and result channel.

    The trace channel contributes ``1/σ_traj²``; the result channel contributes
    ``1/σ_result²`` from a same-design fit with only the weak default prior.
    Both are Gaussian standard deviations (the OpenSkill posterior uncertainty
    for the result channel), and the two weights sum to 1 (HR §5.5's
    "两通道的权重").
    """
    result_precision = 0.0 if result_sigma <= 0.0 else 1.0 / (result_sigma * result_sigma)
    if trace_sd is None:
        return {"trace": 0.0, "result": 1.0}
    prior_precision = 1.0 / (trace_sd * trace_sd)
    total = prior_precision + result_precision
    if total <= 0.0:
        return {"trace": 0.0, "result": 0.0}
    return {"trace": prior_precision / total, "result": result_precision / total}


def nearest_level(
    mu: float, levels: Mapping[str, float]
) -> dict[str, Any] | None:
    """The manifest level closest to ``mu`` (ties by rating then id)."""
    if not levels:
        return None
    id_, value = min(
        levels.items(), key=lambda item: (abs(float(item[1]) - mu), item[1], item[0])
    )
    return {"id": str(id_), "mu": float(value), "distance": abs(float(value) - mu)}


def band_for(ci_half_width: float, *, stop_ci: float = 50.0) -> str:
    """HR §5.5 bands: ``placed`` / ``provisional`` (±1 档) / ``coarse``."""
    if ci_half_width <= stop_ci:
        return "placed"
    if ci_half_width <= 2.0 * stop_ci:
        return "provisional"
    return "coarse"

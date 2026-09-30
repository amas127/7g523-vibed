"""Pure deal-budget arithmetic for the resolution design (standard library only).

The design's 80%-power rule (``docs/experiments/evaluation-resolution-design.md``
§4.2) is

    N = ceil( 2 * tau**2 * ((z_power + z_alpha) / delta_elo) ** 2 / m )

where ``tau = 20 * sigma_400`` is the per-pair contrast sd on the project's
Elo-probit axis, ``m`` is the round-robin band size and
``z_alpha = Phi^-1(1 - alpha / 2)``.  ``power_from_deals`` is the inverse:
``Phi(delta / SE - z_alpha)`` with ``SE = tau * sqrt(2 / (m * N))``.

The default ``sigma_400`` is the revision-3 measurement, 11.6 (review H1:
the retired 10.14 constant under-counts N by ~30% and is only reachable by
passing it explicitly).  Nothing here touches a fit, a training run or the
filesystem; the functions are deterministic and total for valid inputs.
"""
from __future__ import annotations

import math
from statistics import NormalDist

__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_BAND_SIZE",
    "DEFAULT_POWER",
    "DEFAULT_SIGMA_400",
    "TAU_PER_SIGMA",
    "deals_per_pair",
    "power_from_deals",
    "tau_from_sigma",
]

#: Revision-3 (T17) per-pair contrast sd on the 400-Elo-probit scale.
DEFAULT_SIGMA_400 = 11.6

#: Design §3.2: ``tau = 20 * sigma_400``.
TAU_PER_SIGMA = 20.0

DEFAULT_POWER = 0.8
DEFAULT_ALPHA = 0.05
DEFAULT_BAND_SIZE = 4


def _positive_finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be positive and finite, got {value!r}")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {value!r}")
    return number


def _probability(value: float, name: str) -> float:
    number = _positive_finite(value, name)
    if not 0.0 < number < 1.0:
        raise ValueError(f"{name} must lie in (0, 1), got {value!r}")
    return number


def tau_from_sigma(
    sigma_400: float = DEFAULT_SIGMA_400, *, multiple: float = TAU_PER_SIGMA
) -> float:
    """``tau = multiple * sigma_400`` (design §3.2 uses ``multiple = 20``)."""
    sigma = _positive_finite(sigma_400, "sigma_400")
    factor = _positive_finite(multiple, "multiple")
    return factor * sigma


def deals_per_pair(
    delta_elo: float,
    *,
    sigma_400: float = DEFAULT_SIGMA_400,
    power: float = DEFAULT_POWER,
    alpha: float = DEFAULT_ALPHA,
    m: int = DEFAULT_BAND_SIZE,
    tau_multiple: float = TAU_PER_SIGMA,
) -> int:
    """Deals per round-robin pair for the given power at ``delta_elo``.

    ``m`` is the band size (the design uses m = 4, so every pair carries
    ``2 / (m * N)`` of the contrast variance).  Returns ``ceil`` of the
    continuous budget, so the achieved power is at least ``power``.
    """
    delta = _positive_finite(delta_elo, "delta_elo")
    target = _probability(power, "power")
    tail = _probability(alpha, "alpha")
    if isinstance(m, bool) or not isinstance(m, int) or m < 2:
        raise ValueError(f"m must be an int >= 2, got {m!r}")
    tau = tau_from_sigma(sigma_400, multiple=tau_multiple)
    z_alpha = NormalDist().inv_cdf(1.0 - tail / 2.0)
    z_power = NormalDist().inv_cdf(target)
    return math.ceil(2.0 * tau**2 * ((z_power + z_alpha) / delta) ** 2 / m)


def power_from_deals(
    delta_elo: float,
    deals: int,
    *,
    sigma_400: float = DEFAULT_SIGMA_400,
    alpha: float = DEFAULT_ALPHA,
    m: int = DEFAULT_BAND_SIZE,
    tau_multiple: float = TAU_PER_SIGMA,
) -> float:
    """``P(two-sided CI excludes 0 | true delta = delta_elo)`` for ``deals``.

    The design's normal approximation: ``SE = tau * sqrt(2 / (m * N))`` and
    ``power = Phi(delta / SE - z_alpha)``.  This is the CI-only leg of the
    action rule, not the compound ``+10`` rule (which is capped at 0.5 for a
    true ``+10`` effect; see ``tools/refit_mle.action_power``).
    """
    delta = _positive_finite(delta_elo, "delta_elo")
    if isinstance(deals, bool) or not isinstance(deals, int) or deals < 1:
        raise ValueError(f"deals must be an int >= 1, got {deals!r}")
    tail = _probability(alpha, "alpha")
    if isinstance(m, bool) or not isinstance(m, int) or m < 2:
        raise ValueError(f"m must be an int >= 2, got {m!r}")
    tau = tau_from_sigma(sigma_400, multiple=tau_multiple)
    se = tau * math.sqrt(2.0 / (m * deals))
    z_alpha = NormalDist().inv_cdf(1.0 - tail / 2.0)
    return NormalDist().cdf(delta / se - z_alpha)

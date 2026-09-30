"""Order-free anchored probit MLE for the published absolute table (ADR-0013).

``elo.fit_ratings`` is the online Plackett-Luce replay (order-dependent,
positive ``tau``).  This module is the explicitly separate second estimator:
a pure-Python MAP fit used only to publish an absolute table, and never
imported by ``elo.py`` (ADR-0013 decision 3).

Model
-----
Every id carries a prior Gaussian ``(mu, sigma)`` and the link is the static
Thurstone/probit one, the same Gaussian link as
:func:`seven523.elo.expected_score`.  One game with ``N`` seats contributes one
term per unordered seat pair ``(i, j)``:

* ``d = mu_i - mu_j``
* ``s = sqrt(2) * beta`` for every pair (homoscedastic link): ``beta`` is the
  per-seat performance sd, so the performance difference has sd
  ``sqrt(2) * beta``.  A prior ``sigma`` never enters the likelihood; it only
  sets the width of the Gaussian MAP penalty below, which keeps the published
  scale invariant to the prior convention used for the centres
* ``score_i > score_j`` -> ``log Phi((d - eps) / s)``
* ``score_i < score_j`` -> ``log Phi((-d - eps) / s)``
* equal scores -> ``log[Phi((eps - d) / s) - Phi((-eps - d) / s)]``

with the ordered-probit draw margin ``eps >= 0``: the latent performance
difference must exceed ``+eps`` for a win and fall below ``-eps`` for a loss.
Every outcome uses the same ``+/- eps`` thresholds (the standard ordered
probit), which is what identifies ``eps``.  The old hybrid, where the
decisive terms carried no threshold, made the tie likelihood strictly
increasing in ``eps`` and so had no finite estimate.

The MAP objective is the sum of those terms minus
``0.5 * sum over free ids (mu_i - prior.mu)**2 / prior.sigma**2``.  Anchors
stay pinned at their given value and still contribute to every pair; only the
free ids' ``mu`` (and, when ``config.draw_margin is None``, ``log(eps)``) are
optimized.  The Gaussian prior is what keeps a perfect record against the
gauge finite.

``config.draw_margin is None`` estimates ``eps`` jointly from all outcomes.
Two corners carry no information about ``eps`` and are handled explicitly
rather than by a solver clamp: with no tied pair the maximum is at the
``eps = 0`` boundary, so the fit reports ``draw_margin = 0``; with every pair
tied the margin is unbounded, so the fit reports the tie-rate-implied value
``s_mean * Phi_inverse((1 + 0.999) / 2)``.  Both corners set
``MleFit.margin_identified`` to ``False`` so a consumer never mistakes the
boundary/tie-rate value for a data-identified estimate; only a mixed
win/loss/tie record sets it ``True``.

Publication
-----------
``MleConfig.prior`` defaults to :data:`seven523.elo.DEFAULT_PRIOR`
(``Prior(500, 200)``, the ADR-0012 axis).  A consumer publishing an absolute
table should normally pass the frozen study manifest through
``tools/refit_mle.py --prior-manifest``, so every free id is anchored to its
measured centre instead of the generic default.  The manifest's ``sigma``
only sets each id's MAP penalty width; it never enters the link scale, so
publications from the same games under different prior widths share one
``s`` and differ only by shrinkage.

Solver
------
Damped Newton on the free parameters: the ridge-damped Hessian must be
positive definite (pure-Python Cholesky) before its step is used, otherwise
the ridge grows and steepest descent is the fallback; every step is
trust-region clipped and compared by Armijo backtracking with strict
decrease, so an ascent or zero-length step is never accepted.  The iteration
stops at ``max |grad| <= tol`` or when the Newton decrement falls below
``_DECREMENT_TOL * (1 + |objective|)`` (the float-precision optimum);
``MleFit.converged is True`` in both cases.  When ``max_iter`` or the line
search is exhausted first, ``MleFit.converged is False`` and callers should
treat the last iterate as provisional rather than publish it silently.  When
``draw_margin`` is ``None``
the margin is estimated jointly and initialized from the observed tie rate
(``eps0 = s_mean * Phi_inverse((1 + p_tie) / 2)``, floored positive).  The
returned ``Rating.sigma`` is the Laplace posterior standard deviation
``sqrt(diag(inv(H)))`` at the optimum, where ``H`` is the Hessian of the
negative MAP objective (prior terms included), computed with pure-Python
Gaussian elimination.  A negative or non-finite inverse-Hessian diagonal is
never floored to a fabricated precision: that id's ``sigma`` is published as
``math.inf`` and ``MleFit.converged`` becomes ``False``.

Purity
------
Standard library only (``math``/``dataclasses``/``typing`` plus the value
types from :mod:`seven523.elo`): no numpy/scipy/torch, no IO, no RNG.
Deal-clustered bootstrap belongs to the calling tool.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from .elo import DEFAULT_BETA, DEFAULT_PRIOR, PlayedGame, Prior, Rating

__all__ = ["MleConfig", "MleFit", "fit_mle"]

_SQRT2 = math.sqrt(2.0)
_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)

#: Positive floor for the draw margin; needed only to keep ``log`` finite at
#: the numerical boundary.  A solved sigma is never floored to this value.
_EPS_FLOOR = 1e-12

#: Saturation point for the Gaussian tail helpers.  Beyond ``|z| = 1e150``
#: the exact ``z**2`` exponents (``>= 1e300``) leave the float range; every
#: helper clamps here so ``_log_gaussian_cdf`` / ``_mills_ratio`` and the
#: decisive/tie derivatives stay finite for any finite argument.  The clamp
#: is far beyond the tail where a published rating can carry information.
_TAIL_CUTOFF = 1e150

#: ``log(sys.float_info.max)`` rounded up: ``exp`` above this overflows.
_LOG_FLOAT_MAX = 709.782712893384

#: Finite ceiling for the prior precision ``(1 / sigma) ** 2``.  A sigma
#: whose square underflows (or whose reciprocal overflows) saturates here so
#: ``fit_mle`` is total for every finite ``Prior`` instead of dividing by a
#: zero/overflowing square.
_PRIOR_PRECISION_CEIL = 1e300

#: The estimated ``log(eps)`` is clamped to a huge but finite window so an
#: all-ties record cannot overflow ``exp``.
_T_FLOOR = math.log(_EPS_FLOOR)
_T_CEIL = math.log(1e13)

#: ``beta`` below this makes ``beta**2`` underflow to zero, so the decisive
#: curvature ``1 / s**2`` could divide by zero.
_BETA_FLOOR = 1e-150

#: A width ``w`` is "narrow" when ``w * (|middle| + 1)`` is at most this, so
#: the density expansion in the interval helpers converges to float error.
_NARROW_WIDTH = 0.01

#: Per-step trust-region caps: a Newton iterate may move a ``mu`` at most this
#: far and ``log(eps)`` by at most these (``exp(4)`` on the margin).
_MAX_MU_STEP = 1e4
_MAX_T_STEP = 4.0

#: Backtracking stops once the step is this small a fraction of the descent.
_ALPHA_FLOOR = 2.0**-50

#: A Newton decrement at or below this times ``1 + |objective|`` means the
#: model cannot predict a decrease beyond round-off: the float-precision
#: optimum (the standard Newton-decrement stopping rule).
_DECREMENT_TOL = 1e-12

#: Anything below this pivot magnitude is treated as a singular Hessian.
_PIVOT_FLOOR = 1e-300


def _clip_tail(value: float) -> float:
    """Clamp ``value`` to the representable Gaussian-tail window."""
    if value > _TAIL_CUTOFF:
        return _TAIL_CUTOFF
    if value < -_TAIL_CUTOFF:
        return -_TAIL_CUTOFF
    return value


def _exp_or_inf(exponent: float) -> float:
    """``exp(exponent)`` with overflow returning ``math.inf``, never raising."""
    if exponent > _LOG_FLOAT_MAX:
        return math.inf
    return math.exp(exponent)


def _accurate_sum(values: Iterable[float]) -> float:
    """``math.fsum`` with an overflow fallback (plain ``sum`` never raises)."""
    materialized = list(values)
    try:
        return math.fsum(materialized)
    except (OverflowError, ValueError):
        return sum(materialized)


def _positive_finite(value: object, name: str) -> float:
    """``value`` as a positive finite float, or ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be positive and finite, got {value!r}")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {value!r}")
    return number


def _gaussian_cdf(value: float) -> float:
    return 0.5 * math.erfc(-value / _SQRT2)


def _log_gaussian_cdf(value: float) -> float:
    """``log(Phi(value))`` without underflowing in the left tail.

    For ``value < -10`` the asymptotic series ``1 - 1/x**2 + 3/x**4 - ...``
    (``x = -value``) is summed until the terms fall below float resolution,
    which keeps ``log Phi`` accurate to better than 1e-9 relative even in the
    ``(-12, -10)`` window where the old fixed-order series stopped early.
    ``value`` is clamped to ``_TAIL_CUTOFF`` so an extreme finite argument can
    never overflow ``value**2`` into ``inf`` / ``nan``.
    """
    value = _clip_tail(value)
    if value >= -10.0:
        return math.log(_gaussian_cdf(value))
    inverse_sq = 1.0 / (value * value)  # |value| <= 1e150, so value**2 is finite
    term = 1.0
    series = 1.0
    for order in range(1, 101):
        term *= -(2 * order - 1) * inverse_sq
        series += term
        if abs(term) <= 1e-18 * series:
            break
    return -0.5 * value * value - math.log(-value) - _LOG_SQRT_2PI + math.log(series)


def _log_gaussian_density(value: float) -> float:
    value = _clip_tail(value)
    return -0.5 * value * value - _LOG_SQRT_2PI


def _mills_ratio(value: float) -> float:
    """``phi(value) / Phi(value)``, stable in both tails.

    In the left tail the direct form ``x / series`` (``x = -value``, the same
    asymptotic series as :func:`_log_gaussian_cdf`) avoids the subtraction of
    two ``~x**2 / 2`` logarithms, which would lose the ``log x`` term once
    ``x**2`` dwarfs it (e.g. ``x = 1e150``).
    """
    value = _clip_tail(value)
    if value < -10.0:
        inverse_sq = 1.0 / (value * value)
        series = 1.0
        term = 1.0
        for order in range(1, 101):
            term *= -(2 * order - 1) * inverse_sq
            series += term
            if abs(term) <= 1e-18 * series:
                break
        return -value / series
    return _exp_or_inf(_log_gaussian_density(value) - _log_gaussian_cdf(value))


def _u_plus_mills(value: float) -> float:
    """``value + phi(value)/Phi(value)`` without cancellation in the tails.

    For a decisive term ``log Phi(u)`` the second derivative is
    ``-m * (u + m) / s**2`` with ``m = mills(u)``.  For ``u < -10`` the
    subtraction ``u + m`` cancels catastrophically in float; the hazard
    series ``m = x + 1/x - 2/x**3 + ...`` (``x = -u``) gives the small
    difference directly from the same asymptotic terms as
    :func:`_log_gaussian_cdf`.
    """
    value = _clip_tail(value)
    if value >= _TAIL_CUTOFF:
        # phi/Phi underflows: the curvature contribution is exactly zero.
        return 0.0
    if value <= -10.0:
        inverse_sq = 1.0 / (value * value)
        series = 1.0
        correction = 1.0
        term = 1.0
        corr_term = 1.0
        for order in range(1, 101):
            term *= -(2 * order - 1) * inverse_sq
            corr_term *= -(2 * order + 1) * inverse_sq
            series += term
            correction += corr_term
            if abs(term) + abs(corr_term) <= 1e-18 * max(series, correction):
                break
        return correction / (-value * series)
    return value + _mills_ratio(value)


def _decisive_terms(u: float, s: float) -> tuple[float, float, float]:
    """``(log Phi(u), dL/dn, d2L/dn2)`` for ``L = log Phi(n / s)``.

    The second derivative uses the stable ``-m * (u + m) / s**2`` form with
    ``u + m`` from :func:`_u_plus_mills`, so a large ``|u|`` (e.g. anchors at
    ``1e7``) cannot flip the sign of the curvature through the catastrophic
    cancellation in the naive ``-u*m - m*m``.
    """
    u = _clip_tail(u)
    mills = _mills_ratio(u)
    log_p = _log_gaussian_cdf(u)
    first = mills / s
    second = -mills * _u_plus_mills(u) / (s * s)
    return log_p, first, second


def _gaussian_ppf(probability: float) -> float:
    """Inverse Gaussian CDF (Acklam's rational approximation + Halley step)."""
    a = (
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    )
    d = (
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    )
    low = 0.02425
    if not 0.0 < probability < 1.0:
        raise ValueError(f"probability must be in (0, 1), got {probability!r}")
    if probability < low:
        q = math.sqrt(-2.0 * math.log(probability))
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    elif probability > 1.0 - low:
        q = math.sqrt(-2.0 * math.log(1.0 - probability))
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    else:
        q = probability - 0.5
        r = q * q
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
            ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0
        )
    error = _gaussian_cdf(x) - probability
    u = error * math.sqrt(2.0 * math.pi) * math.exp(0.5 * x * x)
    return x - u / (1.0 + 0.5 * x * u)


def _narrow_interval_scale(middle: float, width: float) -> float:
    """``P / (phi(middle) * width)`` for a narrow interval by local expansion.

    ``P = Phi(middle + width/2) - Phi(middle - width/2)``; the expansion
    ``P = phi(middle) * width * (1 + (middle**2 - 1) * width**2 / 24 + ...)``
    avoids subtracting two nearly equal CDFs.  Used only when
    ``width * (|middle| + 1) <= _NARROW_WIDTH``, where the next term is below
    float resolution.  The fourth-order numerator is grouped around
    ``(middle * width)**2`` so a huge ``middle`` cannot overflow ``middle**4``
    and raise ``OverflowError``.
    """
    t = middle * width
    t_sq = t * t
    w_sq = width * width
    return (
        1.0
        + (t_sq - w_sq) / 24.0
        + (t_sq * t_sq / 24.0 - t_sq * w_sq / 4.0 + w_sq * w_sq / 8.0) / 80.0
    )


def _density_expansion_log_p(middle: float, width: float) -> float:
    """``log(Phi(middle + width/2) - Phi(middle - width/2))`` via the local
    density expansion, or ``-inf`` when the expansion itself leaves the float
    range (a genuinely unrepresentable interval probability)."""
    scale = _narrow_interval_scale(middle, width)
    if not math.isfinite(scale) or scale <= 0.0:
        return -math.inf
    return _log_gaussian_density(middle) + math.log(width) + math.log(scale)


def _log_cdf_diff(hi: float, lo: float) -> float:
    """``log(Phi(hi) - Phi(lo))`` for ``hi >= lo``, evaluated stably.

    Narrow intervals (relative to the local density scale) use the density
    expansion in every branch, including the tails; only genuinely wide
    intervals may be formed as a difference of CDFs.
    """
    if hi <= lo:
        return -math.inf
    width = hi - lo
    middle = 0.5 * (hi + lo)
    if width * (abs(middle) + 1.0) <= _NARROW_WIDTH:
        return (
            _log_gaussian_density(middle)
            + math.log(width)
            + math.log(_narrow_interval_scale(middle, width))
        )
    if lo >= 0.0:
        # Both probabilities are near the upper half: use the survival form.
        l_hi = _log_gaussian_cdf(-lo)
        l_lo = _log_gaussian_cdf(-hi)
        gap = l_lo - l_hi
        if gap >= 0.0:  # numerically degenerate interval
            return _density_expansion_log_p(middle, width)
        return l_hi + math.log1p(-math.exp(gap))
    if hi <= 0.0:
        l_hi = _log_gaussian_cdf(hi)
        l_lo = _log_gaussian_cdf(lo)
        gap = l_lo - l_hi
        if gap >= 0.0:
            return _density_expansion_log_p(middle, width)
        return l_hi + math.log1p(-math.exp(gap))
    probability = _gaussian_cdf(hi) - _gaussian_cdf(lo)
    if probability <= 0.0:  # interval narrower than float resolution
        return _density_expansion_log_p(middle, width)
    return math.log(probability)


@dataclass(frozen=True, slots=True)
class MleConfig:
    """The anchored probit MLE's fixed constants and solver budget.

    ``beta`` is the per-seat performance sd and pins the homoscedastic pair
    scale ``s = sqrt(2) * beta``; ``prior`` supplies the Gaussian MAP
    penalty centres and precisions and never touches ``s``.  ``draw_margin``
    is ``None`` to estimate the ordered-probit margin jointly from tied
    pairs, or a fixed non-negative value.  ``ridge`` is added to the Newton
    Hessian diagonal; ``tol``/``max_iter`` end the iteration.
    """

    beta: float = DEFAULT_BETA
    prior: Prior = DEFAULT_PRIOR
    draw_margin: float | None = None
    max_iter: int = 100
    tol: float = 1e-9
    ridge: float = 1e-9

    def __post_init__(self) -> None:
        _positive_finite(self.beta, "beta")
        if self.beta < _BETA_FLOOR:
            raise ValueError(
                f"beta must be at least {_BETA_FLOOR:g} so beta**2 does not "
                f"underflow, got {self.beta!r}"
            )
        _positive_finite(self.ridge, "ridge")
        if not isinstance(self.prior, Prior):
            raise ValueError(f"prior must be a Prior, got {self.prior!r}")
        if self.draw_margin is not None:
            margin = self.draw_margin
            if isinstance(margin, bool) or not isinstance(margin, (int, float)):
                raise ValueError(
                    f"draw_margin must be None or a finite value >= 0, got {margin!r}"
                )
            if not math.isfinite(margin) or margin < 0.0:
                raise ValueError(
                    f"draw_margin must be None or a finite value >= 0, got {margin!r}"
                )


@dataclass(frozen=True, slots=True)
class MleFit:
    """The MAP optimum plus its uncertainty and separation diagnostics.

    ``ratings`` values are :class:`seven523.elo.Rating` with ``mu`` the MAP
    estimate, ``sigma`` the Laplace posterior sd (``0`` for anchors) and ``n``
    the games seen.  ``separation`` names free ids whose published value is not
    trustworthy: a perfect record against pinned anchors, a posterior mean
    more than five prior sigmas from its prior centre, or a non-identified
    (infinite) Laplace ``sigma``.  ``converged`` is ``False`` when Newton
    stopped on ``max_iter`` or a failed line search with a gradient above
    ``tol``, and also when any free id's inverse-Hessian diagonal was
    negative or non-finite (its ``sigma`` is then ``math.inf``); the ratings
    are then the last iterate and should be treated as provisional rather
    than published as a converged estimate.  ``margin_identified`` is ``False``
    when ``draw_margin`` is a boundary or tie-rate-implied value that the data
    did not identify (no ties -> ``0``, all pairs tied -> the tie-rate value)
    and ``True`` when the margin was estimated from mixed outcomes or fixed by
    the caller.
    """

    ratings: dict[str, Rating]
    games: int
    draw_margin: float
    iterations: int
    separation: frozenset[str]
    converged: bool = True
    margin_identified: bool = True


@dataclass(frozen=True, slots=True)
class _Pair:
    """One seat-pair likelihood term with the fixed homoscedastic scale ``s``."""

    first: int  # free-parameter index, or -1 for an anchor
    second: int
    first_mu: float  # pinned anchor mu (unused when ``first >= 0``)
    second_mu: float
    s: float
    kind: int  # +1 first wins, -1 second wins, 0 tie


def _clean_tie_terms(
    log_p: float,
    ld: float,
    ldd: float,
    le: float,
    lee: float,
    lde: float,
) -> tuple[float, float, float, float, float, float]:
    """Drop NaN derivative artifacts (an undefined ``inf - inf``) to zero.

    ``log_p`` is the log-likelihood and is already finite or ``-inf``; only
    the derivative slots are sanitized.  An interval whose density ratios
    overflowed carries no resolvable derivative information, so a zero
    contribution is the saturated-model limit and keeps the Laplace Hessian
    total instead of poisoning it with ``nan``.
    """
    return (
        log_p,
        0.0 if math.isnan(ld) else ld,
        0.0 if math.isnan(ldd) else ldd,
        0.0 if math.isnan(le) else le,
        0.0 if math.isnan(lee) else lee,
        0.0 if math.isnan(lde) else lde,
    )


def _tie_terms(
    a: float, b: float, s: float, width: float | None = None
) -> tuple[float, float, float, float, float, float]:
    """Tie log-likelihood and its derivatives.

    Returns ``(log_p, dL/dd, d2L/dd2, dL/deps, d2L/deps2, d2L/dd deps)`` for
    ``p = Phi(a) - Phi(b)`` with ``a = (eps - d)/s`` and ``b = (-eps - d)/s``.
    ``width`` is the exact interval width ``2*eps/s`` when the caller knows
    it; the rounded ``a - b`` is only the fallback (subtracting the two
    rounded endpoints cancels for narrow intervals with large ``|middle|``).
    Narrow intervals are evaluated through the density expansion; when the
    density ratio itself would overflow the float range (a subnormal
    ``width``), the same quantities are re-formed through their logarithms so
    an overflow can never escape as ``ArithmeticError``.
    """
    if width is None:
        width = a - b
    a = _clip_tail(a)
    b = _clip_tail(b)
    if width > 2.0 * _TAIL_CUTOFF:
        # The interval already covers the whole representable tail; anything
        # wider is saturated and must not overflow ``half`` into ``inf``.
        width = 2.0 * _TAIL_CUTOFF
    if width <= 0.0:
        return -math.inf, 0.0, 0.0, 0.0, 0.0, 0.0
    middle = 0.5 * (a + b)
    half = 0.5 * width
    if width * (abs(middle) + 1.0) <= _NARROW_WIDTH:
        scale = _narrow_interval_scale(middle, width)
        if not math.isfinite(scale) or scale <= 0.0:
            return -math.inf, 0.0, 0.0, 0.0, 0.0, 0.0
        log_p = _log_gaussian_density(middle) + math.log(width) + math.log(scale)
        base = _exp_or_inf(-0.125 * width * width) / (width * scale)
        if math.isfinite(base):
            gap = middle * half
            total_ratio = 2.0 * base * math.cosh(gap)
            delta_ratio = 2.0 * base * math.sinh(gap)
            d_prime_ratio = middle * delta_ratio - half * total_ratio
            e_prime_ratio = half * delta_ratio - middle * total_ratio
            s_sq = s * s
            ld = delta_ratio / s
            le = total_ratio / s
            ldd = d_prime_ratio / s_sq - ld * ld
            lee = d_prime_ratio / s_sq - le * le
            lde = -e_prime_ratio / s_sq - delta_ratio * total_ratio / s_sq
            return _clean_tie_terms(log_p, ld, ldd, le, lee, lde)
        # r_a = phi(a)/p exceeds the float range: re-form every published
        # quantity through its logarithm instead of the density ratio.
        delta = middle * width
        e_delta = _exp_or_inf(delta)  # |delta| <= _NARROW_WIDTH < 0.01
        e1 = math.expm1(delta)
        log_r_a = (
            -0.5 * delta
            - 0.125 * width * width
            - math.log(width)
            - math.log(scale)
        )
        log_s = math.log(s)
        if e1 == 0.0:
            ld = 0.0
        else:
            ld = math.copysign(
                _exp_or_inf(log_r_a + math.log(abs(e1)) - log_s), e1
            )
        le = _exp_or_inf(log_r_a + math.log1p(e_delta) - log_s)
        d_gap = middle * e1 - half * (1.0 + e_delta)
        e_gap = half * e1 - middle * (1.0 + e_delta)
        if d_gap == 0.0:
            d_prime_over_s2 = 0.0
        else:
            d_prime_over_s2 = math.copysign(
                _exp_or_inf(log_r_a + math.log(abs(d_gap)) - 2.0 * log_s), d_gap
            )
        if e_gap == 0.0:
            e_prime_over_s2 = 0.0
        else:
            e_prime_over_s2 = math.copysign(
                _exp_or_inf(log_r_a + math.log(abs(e_gap)) - 2.0 * log_s), e_gap
            )
        ldd = d_prime_over_s2 - ld * ld
        lee = d_prime_over_s2 - le * le
        lde = -e_prime_over_s2 - ld * le
        return _clean_tie_terms(log_p, ld, ldd, le, lee, lde)
    # Wide interval: the ratio form stays finite even when ``p`` itself
    # underflows during a line-search overshoot.
    log_p = _log_cdf_diff(a, b)
    if not math.isfinite(log_p):
        # The rounded endpoints collapsed in float although the exact width is
        # positive: the interval probability is below float resolution.
        return -math.inf, 0.0, 0.0, 0.0, 0.0, 0.0
    log_phi_a = _log_gaussian_density(a)
    log_phi_b = _log_gaussian_density(b)
    ratio_a = _exp_or_inf(log_phi_a - log_p)
    ratio_b = _exp_or_inf(log_phi_b - log_p)
    if not (math.isfinite(ratio_a) and math.isfinite(ratio_b)):
        # Saturated interval: the two ratios overflowed together, so their
        # difference is not resolvable; the derivative contribution is zero.
        return log_p, 0.0, 0.0, 0.0, 0.0, 0.0
    total_ratio = ratio_a + ratio_b
    delta_ratio = ratio_b - ratio_a
    # ``-a * ratio_a + b * ratio_b`` and ``-a * ratio_a - b * ratio_b``,
    # rewritten symmetrically so a near-symmetric interval stays accurate.
    d_prime_ratio = middle * delta_ratio - half * total_ratio
    e_prime_ratio = half * delta_ratio - middle * total_ratio
    s_sq = s * s
    ld = delta_ratio / s
    le = total_ratio / s
    ldd = d_prime_ratio / s_sq - ld * ld
    lee = d_prime_ratio / s_sq - le * le
    lde = -e_prime_ratio / s_sq - delta_ratio * total_ratio / s_sq
    return _clean_tie_terms(log_p, ld, ldd, le, lee, lde)


def _evaluate(
    x: Sequence[float],
    t: float,
    pairs: Sequence[_Pair],
    prior_mean: Sequence[float],
    prior_precision: Sequence[float],
    *,
    estimate_eps: bool,
    fixed_eps: float,
) -> tuple[float, list[float], list[list[float]]]:
    """Objective, gradient and Hessian of the negative MAP objective."""
    n_free = len(x)
    n_params = n_free + (1 if estimate_eps else 0)
    t_index = n_free
    grad = [0.0] * n_params
    hess = [[0.0] * n_params for _ in range(n_params)]
    log_likelihood = 0.0
    eps = math.exp(t) if estimate_eps else fixed_eps
    for pair in pairs:
        first, second = pair.first, pair.second
        mu_first = x[first] if first >= 0 else pair.first_mu
        mu_second = x[second] if second >= 0 else pair.second_mu
        d = mu_first - mu_second
        if not math.isfinite(d):
            # Finite anchors whose difference overflows float: clamp to the
            # documented tail window so the tail helpers stay total.
            d = math.copysign(_TAIL_CUTOFF, d) if math.isinf(d) else 0.0
        le = lde = 0.0
        if pair.kind > 0:
            log_p, first_derivative, second_derivative = _decisive_terms(
                (d - eps) / pair.s, pair.s
            )
            ld = first_derivative
            le = -first_derivative
            ldd = second_derivative
            lee = second_derivative
            lde = -second_derivative
            log_likelihood += log_p
        elif pair.kind < 0:
            log_p, first_derivative, second_derivative = _decisive_terms(
                (-d - eps) / pair.s, pair.s
            )
            ld = -first_derivative
            le = -first_derivative
            ldd = second_derivative
            lee = second_derivative
            lde = second_derivative
            log_likelihood += log_p
        else:
            log_p, ld, ldd, le, lee, lde = _tie_terms(
                (eps - d) / pair.s,
                (-eps - d) / pair.s,
                pair.s,
                2.0 * eps / pair.s,
            )
            log_likelihood += log_p
        if first >= 0:
            grad[first] -= ld
            hess[first][first] -= ldd
        if second >= 0:
            grad[second] += ld
            hess[second][second] -= ldd
        if first >= 0 and second >= 0:
            hess[first][second] += ldd
            hess[second][first] += ldd
        if estimate_eps:
            grad[t_index] -= eps * le
            hess[t_index][t_index] -= eps * eps * lee + eps * le
            if first >= 0:
                hess[first][t_index] -= eps * lde
                hess[t_index][first] -= eps * lde
            if second >= 0:
                hess[second][t_index] += eps * lde
                hess[t_index][second] += eps * lde
    value = -log_likelihood
    for index in range(n_free):
        offset = x[index] - prior_mean[index]
        value += 0.5 * prior_precision[index] * offset * offset
        grad[index] += prior_precision[index] * offset
        hess[index][index] += prior_precision[index]
    return value, grad, hess


def _solve(matrix: Sequence[Sequence[float]], rhs: Sequence[float]) -> list[float]:
    """Solve ``matrix @ x = rhs`` by Gaussian elimination with partial pivoting."""
    n = len(rhs)
    a = [[*row, rhs[index]] for index, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(a[row][col]))
        if abs(a[pivot][col]) < _PIVOT_FLOOR:
            raise ValueError("singular Hessian")
        a[col], a[pivot] = a[pivot], a[col]
        inverse = 1.0 / a[col][col]
        for row in range(col + 1, n):
            factor = a[row][col] * inverse
            if factor == 0.0:
                continue
            for column in range(col, n + 1):
                a[row][column] -= factor * a[col][column]
    x = [0.0] * n
    for row in range(n - 1, -1, -1):
        total = a[row][n] - _accurate_sum(
            a[row][col] * x[col] for col in range(row + 1, n)
        )
        x[row] = total / a[row][row]
    return x


def _inverse_diagonal(matrix: Sequence[Sequence[float]]) -> list[float]:
    """Diagonal of ``matrix ** -1`` by Gauss-Jordan elimination."""
    n = len(matrix)
    if n == 0:
        return []
    a = [
        list(row) + [1.0 if row_index == col else 0.0 for col in range(n)]
        for row_index, row in enumerate(matrix)
    ]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(a[row][col]))
        if abs(a[pivot][col]) < _PIVOT_FLOOR:
            raise ValueError("singular Hessian")
        a[col], a[pivot] = a[pivot], a[col]
        inverse = 1.0 / a[col][col]
        for column in range(2 * n):
            a[col][column] *= inverse
        for row in range(n):
            if row == col:
                continue
            factor = a[row][col]
            if factor == 0.0:
                continue
            for column in range(2 * n):
                a[row][column] -= factor * a[col][column]
    return [a[row][n + row] for row in range(n)]


def _cholesky(matrix: Sequence[Sequence[float]]) -> list[list[float]] | None:
    """Lower Cholesky factor, or ``None`` when ``matrix`` is not positive definite."""
    n = len(matrix)
    lower = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            total = matrix[i][j] - _accurate_sum(
                lower[i][k] * lower[j][k] for k in range(j)
            )
            if i == j:
                if not total > 0.0:
                    return None
                lower[i][j] = math.sqrt(total)
            else:
                lower[i][j] = total / lower[j][j]
    return lower


def _newton_step(
    hess: Sequence[Sequence[float]], grad: Sequence[float], ridge: float
) -> list[float] | None:
    """A descent Newton step from a positive-definite ridge-damped Hessian.

    The ridge grows by factors of ten until the Cholesky factor exists and
    ``grad . step < 0``; ``None`` means no usable Newton direction was found.
    """
    n = len(grad)
    for attempt in range(40):
        augment = ridge * 10.0**attempt
        if not math.isfinite(augment):
            return None
        damped = [list(row) for row in hess]
        for index in range(n):
            damped[index][index] += augment
        if _cholesky(damped) is None:
            continue
        try:
            step = _solve(damped, [-entry for entry in grad])
        except ValueError:
            continue
        slope = _accurate_sum(entry * move for entry, move in zip(grad, step))
        if slope < 0.0 and all(math.isfinite(move) for move in step):
            return step
    return None


def _steepest_descent(grad: Sequence[float]) -> list[float]:
    """``-grad`` scaled so its largest component has magnitude one."""
    scale = max((abs(entry) for entry in grad), default=0.0)
    if scale <= 0.0:
        return [0.0] * len(grad)
    return [-entry / scale for entry in grad]


def _projected_grad_max(grad: Sequence[float], t: float, estimate_eps: bool) -> float:
    """``max |grad|`` with the ``t`` component projected out at a bound."""
    if not estimate_eps:
        return max((abs(entry) for entry in grad), default=0.0)
    top = max((abs(entry) for entry in grad[:-1]), default=0.0)
    if grad:
        last = grad[-1]
        at_floor = t <= _T_FLOOR + 1e-9
        at_ceiling = t >= _T_CEIL - 1e-9
        if not ((at_floor and last > 0.0) or (at_ceiling and last < 0.0)):
            top = max(top, abs(last))
    return top


def _initial_draw_margin(pairs: Sequence[_Pair]) -> float:
    """Observed-tie-rate starting point for the jointly estimated margin."""
    if not pairs:
        return _EPS_FLOOR
    ties = sum(1 for pair in pairs if pair.kind == 0)
    tie_rate = min(ties / len(pairs), 0.999)
    # Scale before summing: ``math.fsum`` of large finite ``s`` values would
    # otherwise raise ``OverflowError`` ('intermediate overflow in fsum').
    count = len(pairs)
    s_mean = _accurate_sum(pair.s / count for pair in pairs)
    quantile = _gaussian_ppf(0.5 * (1.0 + tie_rate))
    return max(s_mean * quantile, _EPS_FLOOR)


def _pair_scale(beta: float) -> float:
    """The homoscedastic pair scale ``s = sqrt(2) * beta``.

    ``beta`` is the performance standard deviation of one seat, so the
    performance difference of a pair has sd ``sqrt(2) * beta``.  The scale is
    the same for every pair, anchors included; a prior ``sigma`` is the width
    of the MAP penalty and deliberately never appears here, which is what
    makes the published levels invariant to the prior convention.
    """
    return _SQRT2 * beta


def _prior_precision(sigma: float) -> float:
    """``(1 / sigma) ** 2`` with the finite ``_PRIOR_PRECISION_CEIL`` cap.

    A finite ``Prior`` may have a sigma whose square overflows (``1e308``) or
    underflows to zero (``1e-300``); neither may reach the fit as an exception
    or a fake infinite precision.
    """
    inverse = 1.0 / sigma
    precision = inverse * inverse
    if not math.isfinite(precision):
        return _PRIOR_PRECISION_CEIL
    return precision


def fit_mle(
    games: Iterable[PlayedGame],
    *,
    anchors: Mapping[str, float],
    priors: Mapping[str, Prior] | None = None,
    config: MleConfig = MleConfig(),
) -> MleFit:
    """Fit the anchored probit MAP model to ``games`` (order-free).

    Anchors are pinned exactly and still enter every pair; ids in ``priors``
    with no games come back with their prior ``mu``/``sigma`` and ``n == 0``;
    an id seen in a game but absent from ``priors`` uses ``config.prior``.
    ``config.draw_margin is None`` estimates the tie margin jointly, else the
    given value is fixed.  Input order never reaches the arithmetic: the games
    are canonicalised by ``(seed, seats, scores)`` before the pair sum, so the
    fit is bit-for-bit independent of the order the caller enumerates them in.
    A negative or non-finite inverse-Hessian diagonal is returned as an
    infinite ``Rating.sigma`` with ``converged=False``, never as a fabricated
    floor; a boundary/tie-rate margin is flagged by ``margin_identified``.
    Errors: ``ValueError`` for a non-finite anchor, a prior for a pinned
    anchor, an invalid prior or an invalid ``config``.
    """
    games = tuple(sorted(games, key=lambda game: (game.seed, game.seats, game.scores)))
    anchors = dict(anchors)
    for id_, value in anchors.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"anchor {id_!r} rating must be finite, got {value!r}")
        if not math.isfinite(value):
            raise ValueError(f"anchor {id_!r} rating must be finite, got {value!r}")
    priors = dict(priors or {})
    for id_, prior in priors.items():
        if not isinstance(prior, Prior):
            raise ValueError(f"prior for {id_!r} must be a Prior, got {prior!r}")
        if id_ in anchors:
            raise ValueError("priors must not target pinned anchors")
    if not isinstance(config, MleConfig):
        raise ValueError(f"config must be a MleConfig, got {config!r}")

    counts: dict[str, int] = {}
    for game in games:
        for id_ in game.seats:
            counts[id_] = counts.get(id_, 0) + 1
    free_ids = sorted((set(counts) | set(priors)) - set(anchors))
    index = {id_: position for position, id_ in enumerate(free_ids)}

    def prior_for(id_: str) -> Prior:
        return priors.get(id_, config.prior)

    # The link is homoscedastic: ``beta`` alone pins the scale for every
    # pair, and a prior sigma enters only the MAP penalty below.
    pair_s = _pair_scale(config.beta)

    pairs: list[_Pair] = []
    for game in games:
        seats = game.seats
        for first in range(len(seats)):
            for second in range(first + 1, len(seats)):
                left, right = seats[first], seats[second]
                if game.scores[first] > game.scores[second]:
                    kind = 1
                elif game.scores[first] < game.scores[second]:
                    kind = -1
                else:
                    kind = 0
                pairs.append(
                    _Pair(
                        first=index.get(left, -1),
                        second=index.get(right, -1),
                        first_mu=anchors.get(left, 0.0),
                        second_mu=anchors.get(right, 0.0),
                        s=pair_s,
                        kind=kind,
                    )
                )

    prior_mean = [prior_for(id_).mu for id_ in free_ids]
    prior_precision = [_prior_precision(prior_for(id_).sigma) for id_ in free_ids]
    n_pairs = len(pairs)
    n_ties = sum(1 for pair in pairs if pair.kind == 0)
    if config.draw_margin is not None:
        estimate_eps = False
        fixed_eps = max(float(config.draw_margin), _EPS_FLOOR)
        reported_margin = float(config.draw_margin)
        margin_identified = True
    elif n_ties == 0:
        # No tie can inform eps and every decisive term decreases in eps, so
        # the maximum lies at the eps = 0 boundary.
        estimate_eps = False
        fixed_eps = 0.0
        reported_margin = 0.0
        margin_identified = False
    elif n_ties == n_pairs:
        # Every pair is tied: the ordered-probit likelihood is monotone in
        # eps and no finite maximum exists.  Report the tie-rate-implied
        # margin instead of an optimizer clamp; ``margin_identified`` is
        # ``False`` so no consumer mistakes it for a fitted value.
        estimate_eps = False
        fixed_eps = _initial_draw_margin(pairs)
        reported_margin = fixed_eps
        margin_identified = False
    else:
        estimate_eps = True
        fixed_eps = 0.0
        reported_margin = 0.0
        margin_identified = True
    x = list(prior_mean)
    t = math.log(_initial_draw_margin(pairs)) if estimate_eps else 0.0

    iterations = 0
    converged = False
    while iterations < config.max_iter:
        value, grad, hess = _evaluate(
            x,
            t,
            pairs,
            prior_mean,
            prior_precision,
            estimate_eps=estimate_eps,
            fixed_eps=fixed_eps,
        )
        if _projected_grad_max(grad, t, estimate_eps) <= config.tol:
            converged = True
            break
        step = _newton_step(hess, grad, config.ridge)
        if step is None:
            step = _steepest_descent(grad)
        decrement = -_accurate_sum(entry * move for entry, move in zip(grad, step))
        if decrement <= _DECREMENT_TOL * (1.0 + abs(value)):
            # The Newton model predicts no decrease beyond round-off.
            converged = True
            break
        for position in range(len(x)):
            step[position] = min(max(step[position], -_MAX_MU_STEP), _MAX_MU_STEP)
        if estimate_eps:
            step[-1] = min(max(step[-1], -_MAX_T_STEP), _MAX_T_STEP)
        slope = _accurate_sum(entry * move for entry, move in zip(grad, step))
        if not slope < 0.0:
            # A clipped Newton step can lose its descent property; fall back
            # to the (also clipped) steepest-descent direction.
            step = _steepest_descent(grad)
            for position in range(len(x)):
                step[position] = min(max(step[position], -_MAX_MU_STEP), _MAX_MU_STEP)
            if estimate_eps:
                step[-1] = min(max(step[-1], -_MAX_T_STEP), _MAX_T_STEP)
            slope = _accurate_sum(entry * move for entry, move in zip(grad, step))
            if not slope < 0.0:
                break
        alpha = 1.0
        accepted = False
        while alpha >= _ALPHA_FLOOR:
            trial_x = [
                x[position] + alpha * step[position] for position in range(len(x))
            ]
            if estimate_eps:
                trial_t = min(max(t + alpha * step[-1], _T_FLOOR), _T_CEIL)
            else:
                trial_t = t
            if trial_x == x and trial_t == t:
                break  # zero-length step: no further progress is possible
            trial_value = _evaluate(
                trial_x,
                trial_t,
                pairs,
                prior_mean,
                prior_precision,
                estimate_eps=estimate_eps,
                fixed_eps=fixed_eps,
            )[0]
            if math.isfinite(trial_value) and trial_value < value + 1e-4 * alpha * slope:
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            break
        x = trial_x
        t = trial_t
        iterations += 1
    if estimate_eps and t >= _T_CEIL - 1e-6:
        # The constrained optimum is the numerical ceiling: eps is not
        # identified, so the last iterate must not be treated as converged
        # and the ceiling value is never published as the estimate.
        converged = False
        margin_identified = False
        reported_margin = _initial_draw_margin(pairs)
    elif estimate_eps:
        reported_margin = math.exp(t)

    # Laplace posterior sd: diag(inv(H)) at the optimum; H already includes the
    # prior terms.  A numerically singular Hessian becomes an infinite sd with
    # ``converged=False`` below (never a ridge-floored fabricated precision).
    # The final evaluation also settles the convergence flag when the loop
    # stopped on ``max_iter``: a negligible Newton decrement is the
    # float-precision optimum, not a stall far from it.
    value, grad, hess = _evaluate(
        x,
        t,
        pairs,
        prior_mean,
        prior_precision,
        estimate_eps=estimate_eps,
        fixed_eps=fixed_eps,
    )
    if not converged:
        if _projected_grad_max(grad, t, estimate_eps) <= config.tol:
            converged = True
        else:
            final_step = _newton_step(hess, grad, config.ridge)
            if final_step is None:
                final_step = _steepest_descent(grad)
            decrement = -_accurate_sum(
                entry * move for entry, move in zip(grad, final_step)
            )
            if decrement <= _DECREMENT_TOL * (1.0 + abs(value)):
                converged = True
    try:
        diagonal = _inverse_diagonal(hess)
    except ValueError:
        # A numerically singular Hessian means the Laplace variance is not
        # resolvable at all; never substitute a ridge-fabricated precision.
        # Every free id is checked below and published as an infinite sd with
        # ``converged=False``.
        diagonal = [math.nan] * len(hess)
    sigma_by_id: dict[str, float] = {}
    unidentified: set[str] = set()
    for position, id_ in enumerate(free_ids):
        entry = diagonal[position]
        if not math.isfinite(entry) or entry <= 0.0:
            # Non-identified Laplace variance: publish an infinite sd and
            # withdraw ``converged`` instead of flooring to the old 1e-6.
            converged = False
            if counts.get(id_, 0) > 0:
                sigma_by_id[id_] = math.inf
                unidentified.add(id_)
            else:
                sigma_by_id[id_] = prior_for(id_).sigma
        else:
            sigma_by_id[id_] = math.sqrt(entry)

    ratings: dict[str, Rating] = {}
    for id_ in sorted(set(counts) | set(priors) | set(anchors)):
        if id_ in anchors:
            ratings[id_] = Rating(anchors[id_], 0.0, counts.get(id_, 0))
        elif counts.get(id_, 0) == 0:
            prior = prior_for(id_)
            ratings[id_] = Rating(prior.mu, prior.sigma, 0)
        else:
            position = index[id_]
            ratings[id_] = Rating(x[position], sigma_by_id[id_], counts[id_])

    separated: set[str] = set(unidentified)
    for id_ in free_ids:
        if counts.get(id_, 0) == 0:
            continue
        prior = prior_for(id_)
        if abs(x[index[id_]] - prior.mu) > 5.0 * prior.sigma:
            separated.add(id_)
        wins = losses = ties = 0
        for game in games:
            if id_ not in game.seats:
                continue
            seat = game.seats.index(id_)
            mine = game.scores[seat]
            for other_seat, other_id in enumerate(game.seats):
                if other_id == id_ or other_id not in anchors:
                    continue
                other = game.scores[other_seat]
                if mine > other:
                    wins += 1
                elif mine < other:
                    losses += 1
                else:
                    ties += 1
        if wins > 0 and losses == 0 and ties == 0:
            separated.add(id_)

    return MleFit(
        ratings=ratings,
        games=len(games),
        draw_margin=reported_margin,
        iterations=iterations,
        separation=frozenset(separated),
        converged=converged,
        margin_identified=margin_identified,
    )

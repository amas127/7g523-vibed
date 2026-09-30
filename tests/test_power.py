"""Pure deal-budget arithmetic (``seven523.power``), review H1/H2 corrections.

The formula is the design's §4.2 80%-power rule with the revision-3
``sigma_400 = 11.6`` default (review H1): the old 10.14 constant under-counts
N by ~30%; it is only reachable by passing it explicitly.
"""
from __future__ import annotations

import math

import pytest

from seven523.power import (
    DEFAULT_ALPHA,
    DEFAULT_POWER,
    DEFAULT_SIGMA_400,
    deals_per_pair,
    power_from_deals,
    tau_from_sigma,
)


def test_revision_three_sigma_is_the_default_and_tau_is_twenty_sigma():
    assert pytest.approx(11.6) == DEFAULT_SIGMA_400
    assert pytest.approx(0.8) == DEFAULT_POWER
    assert pytest.approx(0.05) == DEFAULT_ALPHA
    assert tau_from_sigma(DEFAULT_SIGMA_400) == pytest.approx(20.0 * 11.6)


def test_deals_matches_the_close_formula_and_ceils():
    # N = ceil(2 * tau**2 * (z_power + z_alpha)**2 / (delta**2 * m)).
    from statistics import NormalDist

    z = NormalDist().inv_cdf(0.8) + NormalDist().inv_cdf(0.975)
    expected = math.ceil(2.0 * (20.0 * 11.6) ** 2 * (z / 10.0) ** 2 / 4.0)
    assert deals_per_pair(10.0) == expected
    # The retirement of 10.14 must be visible: the old sigma under-counts.
    old = deals_per_pair(10.0, sigma_400=10.14)
    assert deals_per_pair(10.0) > old
    assert old == math.ceil(2.0 * (20.0 * 10.14) ** 2 * (z / 10.0) ** 2 / 4.0)


def test_power_from_deals_inverts_the_budget():
    n = deals_per_pair(10.0)
    assert power_from_deals(10.0, n) >= DEFAULT_POWER
    assert power_from_deals(10.0, n - 1) < DEFAULT_POWER
    assert power_from_deals(20.0, n) > power_from_deals(10.0, n)


def test_budget_scales_with_sigma_and_band_size():
    assert deals_per_pair(10.0, sigma_400=11.6, m=4) < deals_per_pair(
        10.0, sigma_400=11.6, m=2
    )
    assert deals_per_pair(20.0) < deals_per_pair(10.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"delta_elo": 0.0},
        {"delta_elo": -10.0},
        {"delta_elo": math.inf},
        {"sigma_400": 0.0},
        {"power": 0.0},
        {"power": 1.0},
        {"alpha": 0.0},
        {"m": 1},
    ],
)
def test_invalid_budgets_raise(kwargs):
    with pytest.raises(ValueError):
        deals_per_pair(kwargs.pop("delta_elo", 10.0), **kwargs)


def test_invalid_power_inputs_raise():
    with pytest.raises(ValueError):
        power_from_deals(10.0, 0)
    with pytest.raises(ValueError):
        power_from_deals(-1.0, 10)
    with pytest.raises(ValueError):
        tau_from_sigma(0.0)

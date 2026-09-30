"""The pure rating core: OpenSkill Plackett–Luce, priors, anchors and rungs."""
from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import pytest

from seven523.elo import (
    DEFAULT_BETA,
    DEFAULT_MU,
    DEFAULT_SIGMA,
    DEFAULT_TAU,
    FitConfig,
    PlayedGame,
    Prior,
    Rating,
    expected_score,
    fit_ratings,
    select_rungs,
)


def played(seed: int, a: str, b: str, sa: int, sb: int) -> PlayedGame:
    return PlayedGame(seed, (a, b), (sa, sb))


def gaussian_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def test_expected_score_is_centered_and_sigma_aware():
    even = Rating(DEFAULT_MU, DEFAULT_SIGMA, 0)
    assert expected_score(even, even) == pytest.approx(0.5)
    stronger = Rating(1900.0, DEFAULT_SIGMA, 0)
    assert expected_score(stronger, even) > 0.5
    assert expected_score(stronger, even) + expected_score(even, stronger) == pytest.approx(1.0)
    # A pinned (sigma == 0) opponent is a known quantity; a wide prior is not.
    certain = Rating(DEFAULT_MU, 0.0, 0)
    uncertain = Rating(DEFAULT_MU, 2 * DEFAULT_SIGMA, 0)
    assert abs(expected_score(stronger, certain) - 0.5) > abs(
        expected_score(stronger, uncertain) - 0.5
    )


def test_expected_score_matches_the_plackett_luce_gaussian():
    a, b = Rating(1900.0, 100.0, 10), Rating(1500.0, 200.0, 10)
    variance = 2 * DEFAULT_BETA**2 + a.sigma**2 + b.sigma**2
    assert expected_score(a, b) == pytest.approx(
        gaussian_cdf((a.mu - b.mu) / math.sqrt(variance))
    )


def test_prior_only_entrant_comes_back_untouched():
    fit = fit_ratings([], anchors={}, priors={"you": Prior(1420.0, 80.0)})
    rating = fit.ratings["you"]
    assert (rating.mu, rating.sigma, rating.n) == (1420.0, 80.0, 0)
    assert fit.games == 0


def test_anchors_are_pinned_with_zero_sigma():
    fit = fit_ratings(
        [played(1, "you", "a", 100, 0), played(2, "a", "you", 100, 0)],
        anchors={"a": 1234.5},
        priors={"you": Prior(1500.0, 200.0)},
    )
    anchor = fit.ratings["a"]
    assert (anchor.mu, anchor.sigma, anchor.n) == (1234.5, 0.0, 2)
    assert fit.games == 2


def test_fit_matches_the_plackett_luce_update():
    # The wrapper's job is to call the standard model correctly; this pins the
    # configured model (1500/200 scale, beta=sigma/2, tau=sigma/100) against
    # the library's own single-game update.
    from openskill.models import PlackettLuce

    model = PlackettLuce(
        mu=DEFAULT_MU, sigma=DEFAULT_SIGMA, beta=DEFAULT_BETA, tau=DEFAULT_TAU
    )
    expected = model.rate(
        [
            [model.rating(mu=DEFAULT_MU, sigma=DEFAULT_SIGMA)],
            [model.rating(mu=DEFAULT_MU, sigma=0.0)],
        ],
        scores=[100.0, 0.0],
    )[0][0]
    fit = fit_ratings(
        [played(1, "you", "a", 100, 0)],
        anchors={"a": DEFAULT_MU},
        priors={"you": Prior()},
    )
    assert fit.ratings["you"].mu == pytest.approx(expected.mu)
    assert fit.ratings["you"].sigma == pytest.approx(expected.sigma)
    assert fit.ratings["you"].n == 1


def test_draws_leave_equal_ratings_unchanged():
    games = [played(i, "you", "a", 50, 50) for i in range(4)]
    fit = fit_ratings(
        games, anchors={"a": 1500.0}, priors={"you": Prior(1500.0, 200.0)}
    )
    assert fit.ratings["you"].mu == pytest.approx(1500.0)
    assert fit.ratings["you"].sigma < 200.0
    assert fit.ratings["you"].n == 4


def test_multiplayer_games_rank_by_score_and_handle_ties():
    game = PlayedGame(1, ("x", "y", "z"), (50, 50, 30))
    fit = fit_ratings([game], anchors={})
    assert fit.ratings["x"].mu == pytest.approx(fit.ratings["y"].mu)
    assert fit.ratings["x"].mu > fit.ratings["z"].mu
    assert all(rating.n == 1 for rating in fit.ratings.values())


def test_recent_games_dominate_old_form():
    # tau replaces the retired per-id window: after 5 losses then 5 wins the
    # rating sits above the anchor, i.e. recent 牌局 carry more weight.
    games = [played(i, "you", "a", 0, 100) for i in range(5)]
    games += [played(100 + i, "you", "a", 100, 0) for i in range(5)]
    fit = fit_ratings(
        games, anchors={"a": 1500.0}, priors={"you": Prior(1500.0, 100.0)}
    )
    assert fit.ratings["you"].mu > 1505.0
    assert fit.ratings["you"].n == 10


def test_default_prior_bounds_a_perfect_record():
    games = [played(i, "you", "even", 100, 0) for i in range(40)]
    fit = fit_ratings(games, anchors={"even": DEFAULT_MU})
    assert DEFAULT_MU + 400.0 < fit.ratings["you"].mu < DEFAULT_MU + 600.0
    assert math.isfinite(fit.ratings["you"].sigma)


def test_a_far_anchor_carries_little_information():
    # 40 wins against an anchor far below the prior do not move the estimate:
    # p→1 and the Plackett–Luce update nearly vanishes (the plan's p(1-p) point).
    games = [played(i, "you", "far", 100, 0) for i in range(40)]
    fit = fit_ratings(games, anchors={"far": DEFAULT_MU - 3000.0})
    assert fit.ratings["you"].mu == pytest.approx(DEFAULT_MU, abs=5.0)


def test_fit_validation_errors():
    with pytest.raises(ValueError):
        PlayedGame(1, ("a", "a"), (100, 0))
    with pytest.raises(ValueError):
        PlayedGame(1, ("a",), (100,))
    with pytest.raises(ValueError):
        played(1, "a", "b", -1, 0)
    with pytest.raises(ValueError):
        fit_ratings([], anchors={"a": math.inf})
    with pytest.raises(ValueError):
        fit_ratings(
            [], anchors={"a": 1500.0}, priors={"a": Prior()}
        )  # priors must not target anchors
    with pytest.raises(ValueError):
        Prior(1500.0, 0.0)
    with pytest.raises(ValueError):
        Prior(math.nan, 200.0)
    with pytest.raises(ValueError):
        FitConfig(tau=0.0)
    with pytest.raises(ValueError):
        FitConfig(beta=0.0)
    with pytest.raises(ValueError):
        FitConfig(prior=object())  # type: ignore[arg-type]


def test_select_rungs_picks_spaced_levels():
    ratings = {
        f"lvl{i}": Rating(1000.0 + 120.0 * i, 20.0, 100) for i in range(5)
    }
    selection = select_rungs(ratings, count=5)
    assert [rung.id for rung in selection.rungs] == ["lvl0", "lvl1", "lvl2", "lvl3", "lvl4"]
    assert selection.ok
    assert not selection.wide_gaps
    assert selection.tail_gap == 0.0


def test_select_rungs_reports_gaps_and_shortfalls():
    wide = {"a": Rating(1000.0, 20.0, 100), "b": Rating(1400.0, 20.0, 100)}
    selection = select_rungs(wide, count=2)
    assert selection.wide_gaps == (("a", "b", 400.0),)
    assert not selection.ok

    tail = {
        "a": Rating(1000.0, 20.0, 100),
        "b": Rating(1120.0, 20.0, 100),
        "c": Rating(1900.0, 20.0, 100),
    }
    selection = select_rungs(tail, count=2)
    assert [rung.id for rung in selection.rungs] == ["a", "b"]
    assert selection.tail_gap == pytest.approx(780.0)
    assert not selection.ok

    short = {"a": Rating(1000.0, 20.0, 100)}
    selection = select_rungs(short, count=5)
    assert len(selection.rungs) == 1
    assert not selection.ok

    empty = select_rungs({}, count=3)
    assert empty.rungs == () and not empty.ok

    with pytest.raises(ValueError):
        select_rungs(short, count=0)
    with pytest.raises(ValueError):
        select_rungs(short, min_spacing=200.0, max_spacing=100.0)


def test_importing_the_core_does_not_load_torch():
    code = (
        "import sys; import seven523.elo, seven523.ladder, seven523.study; "
        "assert 'torch' not in sys.modules, 'torch leaked into default imports'"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    assert result.returncode == 0, result.stderr

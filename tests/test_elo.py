"""The pure rating core: BT-MAP, damping, priors, window, margin and rungs."""
from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import pytest

from seven523.elo import (
    DEFAULT_ELO_SCALE,
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


def reference_rating(
    *,
    wins: float = 0.0,
    losses: float = 0.0,
    draws: float = 0.0,
    opponent: float = 1500.0,
    prior: Prior = Prior(),
    margin: float | None = None,
    c: float = 0.15,
    sigma: float = 45.0,
) -> float:
    """Bisection on the same MAP objective, written independently of the module."""
    beta = math.log(10.0) / DEFAULT_ELO_SCALE
    precision = 1.0 / (prior.sd * prior.sd)
    total = wins + losses + draws

    def gradient(rating: float) -> float:
        p = 1.0 / (1.0 + 10.0 ** ((opponent - rating) / DEFAULT_ELO_SCALE))
        grad = beta * (wins + 0.5 * draws - total * p) - precision * (rating - prior.mean)
        if margin is not None:
            grad += c * (margin * total - c * total * (rating - opponent)) / (sigma * sigma)
        return grad

    lo, hi = 400.0, 2600.0
    if gradient(lo) <= 0.0:
        return lo
    if gradient(hi) >= 0.0:
        return hi
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if gradient(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def test_expected_score_is_centered_and_symmetric():
    assert expected_score(1500.0, 1500.0) == pytest.approx(0.5)
    assert expected_score(1900.0, 1500.0) == pytest.approx(10 / 11)
    assert expected_score(1500.0, 1900.0) == pytest.approx(1 / 11)
    with pytest.raises(ValueError):
        expected_score(1500.0, 1500.0, scale=0.0)


def test_prior_only_entrant_comes_back_untouched():
    fit = fit_ratings(
        [], anchors={}, priors={"you": Prior(1420.0, 80.0)}
    )
    rating = fit.ratings["you"]
    assert (rating.elo, rating.se, rating.n) == (1420.0, 80.0, 0)
    assert fit.converged


def test_single_win_matches_the_reference_map():
    prior = Prior(1500.0, 200.0)
    fit = fit_ratings(
        [played(1, "you", "a", 100, 0)],
        anchors={"a": 1500.0},
        priors={"you": prior},
    )
    expected = reference_rating(wins=1.0, prior=prior)
    assert fit.ratings["you"].elo == pytest.approx(expected, abs=0.5)
    assert fit.ratings["you"].n == 1
    assert fit.converged


def test_anchors_are_pinned_with_zero_se():
    fit = fit_ratings(
        [played(1, "you", "a", 100, 0), played(2, "a", "you", 100, 0)],
        anchors={"a": 1234.5},
        priors={"you": Prior(1500.0, 200.0)},
    )
    anchor = fit.ratings["a"]
    assert (anchor.elo, anchor.se, anchor.n) == (1234.5, 0.0, 2)


def test_se_uses_joint_hessian_when_free_ids_play_each_other():
    # Audit §2.3 minimal repro: 100 balanced x-vs-y games, no anchors.  The
    # diagonal Fisher reports ~34; the joint Hessian must report ~142.
    games = [
        played(i, "x", "y", 100, 0) if i % 2 == 0 else played(i, "y", "x", 100, 0)
        for i in range(100)
    ]
    fit = fit_ratings(games, anchors={})
    assert fit.ratings["x"].se == pytest.approx(142.47, rel=0.01)
    assert fit.ratings["y"].se == pytest.approx(fit.ratings["x"].se, rel=1e-9)


def test_se_joint_hessian_does_not_double_count_free_games():
    # A free-vs-free game contributes w once to each player's own diagonal and
    # -w to the pair off-diagonal.  The original assembly also added w to the
    # opponent's diagonal, i.e. 2*w per game, which understates the SE by up to
    # ~6% here (and ~19% in cross-heavy many-candidate designs).
    games = []
    for i in range(20):
        games.append(played(i, "x", "anchor", 100 if i % 2 else 0, 0 if i % 2 else 100))
        games.append(played(100 + i, "y", "anchor", 100 if i % 2 else 0, 0 if i % 2 else 100))
        games.append(played(200 + i, "x", "y", 100 if i % 2 else 0, 0 if i % 2 else 100))
    fit = fit_ratings(games, anchors={"anchor": 1500.0})

    beta = math.log(10.0) / DEFAULT_ELO_SCALE
    r = {i: fit.ratings[i].elo for i in ("x", "y", "anchor")}

    def w(a: str, b: str) -> float:
        p = expected_score(r[a], r[b])
        return beta * beta * p * (1.0 - p)

    tau = 1.0 / 200.0**2
    cross = 20 * w("x", "y")
    info = [
        [tau + 20 * w("x", "anchor") + cross, -cross],
        [-cross, tau + 20 * w("y", "anchor") + cross],
    ]
    det = info[0][0] * info[1][1] - cross * cross
    assert fit.ratings["x"].se == pytest.approx(math.sqrt(info[1][1] / det), rel=1e-9)
    assert fit.ratings["y"].se == pytest.approx(math.sqrt(info[0][0] / det), rel=1e-9)


def test_se_stays_diagonal_for_anchor_only_games():
    games = [played(i, "you", "a", 100, 0) for i in range(4)]
    fit = fit_ratings(
        games, anchors={"a": 1500.0}, priors={"you": Prior(1500.0, 200.0)}
    )
    beta = math.log(10.0) / DEFAULT_ELO_SCALE
    p = expected_score(fit.ratings["you"].elo, 1500.0)
    info = 1.0 / 200.0**2 + 4 * beta * beta * p * (1.0 - p)
    assert fit.ratings["you"].se == pytest.approx(1.0 / math.sqrt(info), rel=1e-9)


def test_window_keeps_only_the_recent_games():
    games = [played(i, "you", "a", 0, 100) for i in range(5)]
    games += [played(100 + i, "you", "a", 100, 0) for i in range(5)]
    prior = Prior(1500.0, 100.0)
    full = fit_ratings(games, anchors={"a": 1500.0}, priors={"you": prior})
    recent = fit_ratings(
        games, anchors={"a": 1500.0}, priors={"you": prior}, window=5
    )
    assert full.ratings["you"].elo == pytest.approx(1500.0, abs=0.5)
    assert recent.ratings["you"].elo > full.ratings["you"].elo + 100
    assert recent.ratings["you"].n == 5


def test_draws_count_as_half_points():
    games = [played(i, "you", "a", 50, 50) for i in range(4)]
    fit = fit_ratings(
        games, anchors={"a": 1500.0}, priors={"you": Prior(1500.0, 200.0)}
    )
    assert fit.ratings["you"].elo == pytest.approx(1500.0)
    assert fit.ratings["you"].n == 4


def test_margin_channel_pulls_toward_the_point_difference():
    # 6 wins by 40 and 6 losses by 20: the score channel alone says 1500, the
    # average +10 分差 says slightly stronger.
    games = [PlayedGame(i, ("you", "a"), (70, 30)) for i in range(6)]
    games += [PlayedGame(100 + i, ("you", "a"), (40, 60)) for i in range(6)]
    prior = Prior(1500.0, 200.0)
    plain = fit_ratings(games, anchors={"a": 1500.0}, priors={"you": prior})
    margin = fit_ratings(
        games,
        anchors={"a": 1500.0},
        priors={"you": prior},
        config=FitConfig(margin=(0.15, 45.0)),
    )
    assert plain.ratings["you"].elo == pytest.approx(1500.0)
    assert margin.ratings["you"].elo > plain.ratings["you"].elo + 20
    assert margin.ratings["you"].elo < 1600.0
    assert margin.ratings["you"].elo == pytest.approx(
        reference_rating(wins=6.0, losses=6.0, prior=prior, margin=10.0), abs=1.0
    )
    assert margin.ratings["you"].se < plain.ratings["you"].se


def test_adversarial_records_clamp_without_overshooting():
    # A perfect record against a wide prior has no finite MAP; damping + the
    # [400, 2600] clamp must stop it exactly at the bound, never overshoot.
    wins = [played(i, "you", "even", 100, 0) for i in range(40)]
    fit = fit_ratings(
        wins, anchors={"even": 1500.0}, priors={"you": Prior(1500.0, 1e6)}
    )
    assert fit.ratings["you"].elo == 2600.0
    assert fit.converged
    losses = [played(1000 + i, "you", "even", 0, 100) for i in range(40)]
    fit = fit_ratings(
        losses, anchors={"even": 1500.0}, priors={"you": Prior(1500.0, 1e6)}
    )
    assert fit.ratings["you"].elo == 400.0
    assert fit.converged


def test_perfect_record_with_the_default_prior_stays_bounded():
    # With a bounded prior the MAP is finite and well inside the clamp.
    games = [played(i, "you", "even", 100, 0) for i in range(40)]
    fit = fit_ratings(games, anchors={"even": 1500.0})
    assert 1900.0 < fit.ratings["you"].elo < 2100.0


def test_a_far_anchor_carries_little_information():
    # p→1: 40 wins against a much weaker anchor do not justify a high rating;
    # the prior keeps the estimate near its mean (the plan's p(1-p) point).
    games = [played(i, "you", "far", 100, 0) for i in range(40)]
    fit = fit_ratings(games, anchors={"far": 400.0})
    assert fit.ratings["you"].elo == pytest.approx(1515.0, abs=5.0)


def test_fit_validation_errors():
    with pytest.raises(ValueError):
        PlayedGame(1, ("a", "a"), (100, 0))
    with pytest.raises(ValueError):
        PlayedGame(1, ("a",), (100,))
    with pytest.raises(ValueError):
        played(1, "a", "b", -1, 0)
    three_seat = PlayedGame(1, ("a", "b", "c"), (40, 30, 30))
    with pytest.raises(ValueError):
        fit_ratings([three_seat], anchors={"a": 1500.0})
    with pytest.raises(ValueError):
        fit_ratings([], anchors={"a": math.inf})
    with pytest.raises(ValueError):
        fit_ratings([], anchors={}, priors={"you": Prior()}, window=0)
    with pytest.raises(ValueError):
        fit_ratings(
            [], anchors={"a": 1500.0}, priors={"a": Prior()}
        )  # priors must not target anchors
    with pytest.raises(ValueError):
        Prior(1500.0, 0.0)
    with pytest.raises(ValueError):
        FitConfig(margin=(0.15, 0.0))


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

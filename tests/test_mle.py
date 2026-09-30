"""The order-free anchored probit MLE: objective, solver, sigma and separation."""
from __future__ import annotations

import math
import random

import pytest

from seven523.elo import (
    DEFAULT_BETA,
    DEFAULT_MU,
    DEFAULT_SIGMA,
    PlayedGame,
    Prior,
    Rating,
)
from seven523.mle import (
    MleConfig,
    MleFit,
    _decisive_terms,
    _evaluate,
    _gaussian_ppf,
    _log_cdf_diff,
    _log_gaussian_cdf,
    _log_gaussian_density,
    _mills_ratio,
    _narrow_interval_scale,
    _newton_step,
    _Pair,
    _pair_scale,
    _tie_terms,
    fit_mle,
)

_SQRT_2PI = math.sqrt(2.0 * math.pi)


def gaussian_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def two_player(seed: int, a: str, b: str, score_a: int, score_b: int) -> PlayedGame:
    return PlayedGame(seed, (a, b), (score_a, score_b))


def sample_game(
    rng: random.Random,
    seed: int,
    a: str,
    b: str,
    mu_a: float,
    mu_b: float,
    s: float,
    margin: float = 0.0,
) -> PlayedGame:
    """One probit draw: performance difference ~ N(mu_a - mu_b, s^2)."""
    difference = (mu_a - mu_b) + s * rng.gauss(0.0, 1.0)
    if difference > margin:
        return two_player(seed, a, b, 100, 0)
    if difference < -margin:
        return two_player(seed, a, b, 0, 100)
    return two_player(seed, a, b, 50, 50)


def pair_scale(beta: float = DEFAULT_BETA) -> float:
    """The published homoscedastic link scale ``s = sqrt(2) * beta``.

    A prior sigma never enters ``s`` (ADR-0013): it is a MAP penalty width,
    so the same data fit under any prior convention shares one link scale.
    """
    return math.sqrt(2.0) * beta


def independent_nll(
    games: list[PlayedGame],
    mus: dict[str, float],
    priors: dict[str, Prior],
    margin: float,
    anchors: dict[str, float] | None = None,
) -> float:
    """A test-side re-implementation of the documented MAP objective.

    The decisive terms carry the same ``+/- margin`` thresholds as the tie
    interval, the standard ordered-probit likelihood the module contracts.
    """
    anchors = dict(anchors or {})

    def mu(id_: str) -> float:
        return anchors[id_] if id_ in anchors else mus[id_]

    total = 0.0
    for game in games:
        for p in range(len(game.seats)):
            for q in range(p + 1, len(game.seats)):
                a, b = game.seats[p], game.seats[q]
                d = mu(a) - mu(b)
                s = pair_scale()
                if game.scores[p] > game.scores[q]:
                    total += math.log(gaussian_cdf((d - margin) / s))
                elif game.scores[p] < game.scores[q]:
                    total += math.log(gaussian_cdf((-d - margin) / s))
                else:
                    upper = gaussian_cdf((margin - d) / s)
                    lower = gaussian_cdf((-margin - d) / s)
                    total += math.log(upper - lower)
    regularizer = 0.5 * sum(
        (mus[id_] - prior.mu) ** 2 / prior.sigma**2 for id_, prior in priors.items()
    )
    return -total + regularizer


def test_anchor_is_pinned_exactly_with_zero_sigma():
    games = [two_player(i, "you", "random", 100, 0) for i in range(6)]
    fit = fit_mle(games, anchors={"random": 12.5}, priors={"you": Prior(0.0, 150.0)})
    anchor = fit.ratings["random"]
    assert (anchor.mu, anchor.sigma, anchor.n) == (12.5, 0.0, 6)
    assert fit.games == 6


def test_fit_is_order_free_for_shuffled_games():
    rng = random.Random(7)
    s = pair_scale()
    games = [
        sample_game(
            rng,
            i,
            "b" if i % 4 == 0 else "a",
            "a" if i % 4 == 0 else "b",
            40.0,
            -20.0,
            s,
            30.0,
        )
        for i in range(240)
    ]
    priors = {"a": Prior(100.0, 180.0), "b": Prior(-30.0, 180.0)}
    first = fit_mle(games, anchors={"random": 0.0}, priors=priors)
    shuffled = list(games)
    random.Random(11).shuffle(shuffled)
    second = fit_mle(shuffled, anchors={"random": 0.0}, priors=priors)
    biggest = max(
        abs(first.ratings[id_].mu - second.ratings[id_].mu) for id_ in ("a", "b")
    )
    assert biggest < 1e-6


def test_recovers_known_probit_levels_on_a_probit_dgp():
    rng = random.Random(20260926)
    truths = {"strong": 220.0, "mid": 110.0, "weak": 20.0}
    s = pair_scale()
    games: list[PlayedGame] = []
    for id_, mu in truths.items():
        for _ in range(1800):
            games.append(sample_game(rng, len(games), id_, "random", mu, 0.0, s))
    pairs = [("strong", "mid"), ("strong", "weak"), ("mid", "weak")]
    for a, b in pairs:
        for _ in range(800):
            games.append(sample_game(rng, len(games), a, b, truths[a], truths[b], s))
    priors = {id_: Prior(DEFAULT_MU, DEFAULT_SIGMA) for id_ in truths}
    fit = fit_mle(games, anchors={"random": 0.0}, priors=priors)
    for id_, mu in truths.items():
        assert fit.ratings[id_].mu == pytest.approx(mu, abs=30.0)
    difference = fit.ratings["strong"].mu - fit.ratings["weak"].mu
    assert difference == pytest.approx(200.0, abs=30.0)


def test_link_scale_is_homoscedastic_and_prior_free():
    # ADR-0013: s = sqrt(2) * beta for every pair; a prior sigma is a MAP
    # penalty width, never part of the published link scale.
    assert _pair_scale(DEFAULT_BETA) == pytest.approx(math.sqrt(2.0) * DEFAULT_BETA)
    # The old plug-in scale at the default prior (beta=100, sigma=200) was
    # sqrt(2*beta**2 + 2*sigma**2), about 2.2x larger.
    old_plugin = math.sqrt(2.0 * DEFAULT_BETA**2 + 2.0 * DEFAULT_SIGMA**2)
    assert _pair_scale(DEFAULT_BETA) < 0.5 * old_plugin


def test_prior_width_alone_does_not_move_the_levels_or_scale():
    # Scale-stability regression: the old plug-in scale coupled the prior
    # sigma into s, so the same 4000-game study published lvl2 = 209 under
    # sigma = 200 but 130.6 under the manifest's sigma ~ 38.  With a fixed
    # homoscedastic s the two fits must agree on the pairwise mu difference
    # to within the prior's shrinkage (a few percent), not the old ~2x.
    rng = random.Random(20261013)
    s = _pair_scale(DEFAULT_BETA)
    games = [
        sample_game(rng, seed, "a", "b", 150.0, 0.0, s) for seed in range(2400)
    ]
    narrow = fit_mle(
        games, anchors={}, priors={"a": Prior(0.0, 38.0), "b": Prior(0.0, 38.0)}
    )
    wide = fit_mle(
        games, anchors={}, priors={"a": Prior(0.0, 200.0), "b": Prior(0.0, 200.0)}
    )
    d_narrow = narrow.ratings["a"].mu - narrow.ratings["b"].mu
    d_wide = wide.ratings["a"].mu - wide.ratings["b"].mu
    # Both fits recover the DGP whose performance difference is 150 at s.
    assert d_wide == pytest.approx(150.0, abs=15.0)
    assert d_narrow == pytest.approx(d_wide, rel=0.03)
    # The prior still regularizes: the narrower width shrinks the Laplace sd.
    assert narrow.ratings["a"].sigma < wide.ratings["a"].sigma
    # Reproduced win probabilities share the single homoscedastic link.
    assert gaussian_cdf(d_narrow / s) == pytest.approx(
        gaussian_cdf(d_wide / s), rel=0.01
    )


def test_draw_margin_is_estimated_from_ties_and_zero_without_them():
    rng = random.Random(5)
    s = pair_scale()
    priors = {"a": Prior(DEFAULT_MU, DEFAULT_SIGMA), "b": Prior(DEFAULT_MU, DEFAULT_SIGMA)}
    tied = [sample_game(rng, i, "a", "b", 120.0, 20.0, s, 60.0) for i in range(1600)]
    tie_rate = sum(g.scores[0] == g.scores[1] for g in tied) / len(tied)
    assert 0.0 < tie_rate < 1.0
    fit_ties = fit_mle(tied, anchors={}, priors=priors)
    # The ordered-probit margin is identified: the estimate is finite and sits
    # near the tie-rate-implied value, not at the 1e13 solver clamp.
    implied = s * _gaussian_ppf(0.5 * (1.0 + tie_rate))
    assert fit_ties.converged
    assert 0.0 < fit_ties.draw_margin < 1e6
    assert fit_ties.draw_margin == pytest.approx(implied, rel=0.5)
    # Ties must materially move the levels: treating the same rounds as
    # impossible ties (margin 0) changes the fit.
    fixed_zero = fit_mle(
        tied, anchors={}, priors=priors, config=MleConfig(draw_margin=0.0)
    )
    moved = max(
        abs(fit_ties.ratings[id_].mu - fixed_zero.ratings[id_].mu) for id_ in priors
    )
    assert moved > 1.0
    decisive = [sample_game(rng, 10_000 + i, "a", "b", 0.0, 0.0, s) for i in range(200)]
    assert all(game.scores[0] != game.scores[1] for game in decisive)
    fit_none = fit_mle(decisive, anchors={}, priors=priors)
    assert fit_none.draw_margin == 0.0
    fixed = fit_mle(tied, anchors={}, priors=priors, config=MleConfig(draw_margin=25.0))
    assert fixed.draw_margin == 25.0


def test_perfect_record_against_anchor_stays_finite_and_is_flagged():
    games = [two_player(i, "you", "random", 100, 0) for i in range(25)]
    fit = fit_mle(games, anchors={"random": 0.0}, priors={"you": Prior(0.0, 100.0)})
    rating = fit.ratings["you"]
    assert math.isfinite(rating.mu) and math.isfinite(rating.sigma)
    assert rating.sigma > 0.0
    assert abs(rating.mu) < 5_000.0
    assert "you" in fit.separation
    mixed = [two_player(i, "you", "random", 100, 0) for i in range(5)]
    mixed += [two_player(100 + i, "you", "random", 0, 100) for i in range(5)]
    fit_mixed = fit_mle(mixed, anchors={"random": 0.0}, priors={"you": Prior(0.0, 100.0)})
    assert "you" not in fit_mixed.separation


def test_prior_distance_alone_flags_separation():
    # Losing to the gauge pulls the estimate far below a high prior centre,
    # without any perfect record.
    games = [two_player(i, "you", "random", 0, 100) for i in range(40)]
    fit = fit_mle(games, anchors={"random": 0.0}, priors={"you": Prior(1000.0, 20.0)})
    assert "you" in fit.separation
    assert math.isfinite(fit.ratings["you"].mu)


def test_laplace_sigma_is_positive_and_shrinks_with_more_games():
    rng = random.Random(3)
    s = pair_scale()
    priors = {"a": Prior(DEFAULT_MU, DEFAULT_SIGMA), "b": Prior(DEFAULT_MU, DEFAULT_SIGMA)}
    few = [sample_game(rng, i, "a", "b", 0.0, 0.0, s) for i in range(8)]
    many = [sample_game(rng, i, "a", "b", 0.0, 0.0, s) for i in range(300)]
    fit_few = fit_mle(few, anchors={}, priors=priors)
    fit_many = fit_mle(many, anchors={}, priors=priors)
    for id_ in ("a", "b"):
        assert fit_few.ratings[id_].sigma > 0.0
        assert math.isfinite(fit_few.ratings[id_].sigma)
        assert fit_many.ratings[id_].sigma < fit_few.ratings[id_].sigma


def test_three_seat_game_rates_every_pair():
    game = PlayedGame(1, ("a", "b", "c"), (100, 50, 0))
    priors = {id_: Prior(DEFAULT_MU, DEFAULT_SIGMA) for id_ in ("a", "b", "c")}
    fit = fit_mle([game], anchors={}, priors=priors)
    assert fit.ratings["a"].mu > fit.ratings["b"].mu > fit.ratings["c"].mu
    assert all(fit.ratings[id_].n == 1 for id_ in ("a", "b", "c"))
    assert all(fit.ratings[id_].sigma > 0.0 for id_ in ("a", "b", "c"))


def test_prior_only_ids_are_returned_with_zero_games():
    fit = fit_mle(
        [two_player(1, "a", "b", 100, 0)],
        anchors={},
        priors={
            "a": Prior(DEFAULT_MU, DEFAULT_SIGMA),
            "b": Prior(DEFAULT_MU, DEFAULT_SIGMA),
            "you": Prior(1420.0, 80.0),
        },
    )
    assert fit.ratings["you"] == Rating(1420.0, 80.0, 0)
    assert fit.ratings["a"].n == 1 and fit.ratings["b"].n == 1
    empty = fit_mle([], anchors={}, priors={"you": Prior(1420.0, 80.0)})
    assert empty.ratings["you"] == Rating(1420.0, 80.0, 0)
    assert empty.games == 0
    assert empty.separation == frozenset()


def test_anchors_with_no_games_are_returned_pinned():
    fit = fit_mle([], anchors={"random": 7.0})
    assert fit.ratings["random"] == Rating(7.0, 0.0, 0)


def test_free_ids_absent_from_priors_use_the_config_prior():
    games = [two_player(i, "you", "random", 100, 0) for i in range(5)]
    games += [two_player(100 + i, "you", "random", 0, 100) for i in range(5)]
    centre = Prior(-40.0, 60.0)
    fit = fit_mle(games, anchors={"random": centre.mu}, config=MleConfig(prior=centre))
    assert isinstance(fit, MleFit)
    assert fit.ratings["you"].mu == pytest.approx(centre.mu, abs=1e-6)


def test_fit_mle_validates_anchors_priors_and_config():
    games = [two_player(1, "you", "random", 100, 0)]
    with pytest.raises(ValueError):
        fit_mle(games, anchors={"random": float("nan")})
    with pytest.raises(ValueError):
        fit_mle(games, anchors={"random": 0.0}, priors={"random": Prior()})
    with pytest.raises(ValueError):
        fit_mle(games, anchors={}, priors={"you": 3.0})
    with pytest.raises(ValueError):
        fit_mle(games, anchors={}, config=object())


def test_fit_matches_an_independently_coded_objective():
    # Two free players and a gauge, with ties, so mu and the margin are both
    # estimated; perturbing the fitted point must not lower the objective.
    rng = random.Random(17)
    priors = {"a": Prior(100.0, 150.0), "b": Prior(0.0, 150.0)}
    s = pair_scale()
    games = [sample_game(rng, i, "a", "b", 100.0, 0.0, s, 60.0) for i in range(120)]
    fit = fit_mle(games, anchors={"random": 0.0}, priors=priors)
    mus = {id_: fit.ratings[id_].mu for id_ in priors}
    base = independent_nll(games, mus, priors, fit.draw_margin)
    for id_ in mus:
        for delta in (-0.5, 0.5):
            perturbed = dict(mus)
            perturbed[id_] += delta
            assert independent_nll(games, perturbed, priors, fit.draw_margin) >= base - 1e-9
    for factor in (0.5, 1.5):
        assert (
            independent_nll(games, mus, priors, fit.draw_margin * factor) >= base - 1e-9
        )


def test_two_player_optimum_matches_the_stationarity_condition():
    # One decisive game between two equally prior-centred players: by symmetry
    # mu_a + mu_b = 2 * prior.mu, and mu_a solves q(d/s)/s = (mu_a - prior.mu)/sigma^2.
    prior = Prior(DEFAULT_MU, DEFAULT_SIGMA)
    s = pair_scale()
    fit = fit_mle(
        [two_player(1, "a", "b", 100, 0)],
        anchors={},
        priors={"a": prior, "b": prior},
    )
    mu_a = fit.ratings["a"].mu
    mu_b = fit.ratings["b"].mu
    assert mu_a + mu_b == pytest.approx(2.0 * prior.mu, abs=1e-6)
    u = (mu_a - mu_b) / s
    mills = math.exp(-0.5 * u * u) / (_SQRT_2PI * gaussian_cdf(u))
    residual = mills / s - (mu_a - prior.mu) / prior.sigma**2
    assert residual == pytest.approx(0.0, abs=1e-7)


def test_mle_config_defaults_and_validation():
    config = MleConfig()
    assert config.beta == DEFAULT_BETA
    assert config.prior == Prior()
    assert config.draw_margin is None
    for kwargs in (
        {"beta": 0.0},
        {"beta": -1.0},
        {"beta": float("nan")},
        {"ridge": 0.0},
        {"ridge": float("inf")},
        {"prior": 3.0},
        {"draw_margin": -0.1},
        {"draw_margin": float("nan")},
        {"draw_margin": float("inf")},
    ):
        with pytest.raises(ValueError):
            MleConfig(**kwargs)


def test_mle_config_rejects_a_beta_whose_square_underflows():
    # F5 regression: beta = 5e-324 is positive and finite, but beta**2 == 0.0,
    # so an all-pinned pair used to divide by a zero scale.
    for beta in (5e-324, 1e-200, 1e-160):
        with pytest.raises(ValueError):
            MleConfig(beta=beta)


def test_all_ties_report_the_tie_rate_implied_margin():
    # Every pair tied: the ordered-probit likelihood is monotone in eps and no
    # finite maximum exists, so the contract is the tie-rate-implied value,
    # never the 1e13 solver clamp.
    games = [two_player(i, "a", "b", 50, 50) for i in range(8)]
    priors = {"a": Prior(), "b": Prior()}
    fit = fit_mle(games, anchors={}, priors=priors)
    s = pair_scale()
    implied = s * _gaussian_ppf(0.5 * (1.0 + 0.999))
    assert fit.converged
    assert fit.draw_margin == pytest.approx(implied, rel=1e-6)
    assert fit.draw_margin < 1e6
    assert math.isfinite(fit.ratings["a"].mu)


def test_log_cdf_diff_and_derivatives_for_narrow_intervals():
    # F3 regression: at draw_margin=0 the tie interval is ~1e-15 wide, where
    # the old difference of log-CDFs returned log Phi(hi) instead of
    # log(width * phi(mid)).  References are mpmath (dps=80) values at the
    # exact float endpoints.
    s = 245.0
    for ratio, log_p, ld, ldd, le, lee, lde in (
        (
            0.01,
            -33.35807861695414,
            -4.0816326530612245e-05,
            -1.665972511453561e-05,
            9.99957938720736e11,
            -9.999158792106231e23,
            4.53344921724107e-22,
        ),
        (
            0.1,
            -33.36515582473351,
            -4.0816326530612244e-04,
            -1.665972511453561e-05,
            1.002087321043395e12,
            -1.0041789989959285e24,
            4.5238158784879e-21,
        ),
    ):
        d = ratio * s
        a = (1e-12 - d) / s
        b = (-1e-12 - d) / s
        assert _log_cdf_diff(a, b) == pytest.approx(log_p, abs=1e-12)
        terms = _tie_terms(a, b, s)
        assert terms[0] == pytest.approx(log_p, abs=1e-12)
        assert terms[1] == pytest.approx(ld, rel=1e-12)
        assert terms[2] == pytest.approx(ldd, rel=1e-12)
        assert terms[3] == pytest.approx(le, rel=1e-12)
        assert terms[4] == pytest.approx(lee, rel=1e-12)
        # The mixed derivative is a ~1e-22 cancellation of ~1e7 terms, i.e.
        # zero at double precision; only its magnitude matters.
        assert terms[5] == pytest.approx(lde, abs=1e-15)


def test_newton_step_never_returns_an_ascent_direction():
    # F2 regression: the raw ridge-damped Hessian can be indefinite, making
    # the naive Newton step an ascent direction; the step builder must grow
    # the ridge until the step descends.
    indefinite = [[1.0, 0.0, 0.0], [0.0, -2.0, 0.0], [0.0, 0.0, 1.0]]
    grad = [1.0, 1.0, 0.0]
    step = _newton_step(indefinite, grad, 1e-9)
    assert step is not None
    assert math.fsum(g * p for g, p in zip(grad, step)) < 0.0


def test_evaluate_gradient_and_hessian_match_finite_differences():
    # Tie second derivatives (notably the d/eps cross term) are used for the
    # Laplace sigma and the Newton step; check them against the objective.
    pairs = [
        _Pair(0, 1, 0.0, 0.0, 283.0, 0),
        _Pair(0, 1, 0.0, 0.0, 250.0, 1),
    ]
    prior_mean = [0.0, 0.0]
    precision = [1.0 / 200.0**2, 1.0 / 200.0**2]
    for eps in (30.0, 200.0):
        params = [50.0, -40.0, math.log(eps)]
        value, grad, hess = _evaluate(
            params[:2],
            params[2],
            pairs,
            prior_mean,
            precision,
            estimate_eps=True,
            fixed_eps=0.0,
        )
        assert math.isfinite(value)

        def objective(point: list[float]) -> float:
            return _evaluate(
                point[:2],
                point[2],
                pairs,
                prior_mean,
                precision,
                estimate_eps=True,
                fixed_eps=0.0,
            )[0]

        steps = [1e-4 * max(1.0, abs(entry)) for entry in params]
        for i in range(3):
            up, down = list(params), list(params)
            up[i] += steps[i]
            down[i] -= steps[i]
            fd = (objective(up) - objective(down)) / (2.0 * steps[i])
            assert fd == pytest.approx(grad[i], abs=1e-6)
        for i in range(3):
            for j in range(3):
                pp, pm, mp_, mm = (list(params) for _ in range(4))
                pp[i] += steps[i]
                pp[j] += steps[j]
                pm[i] += steps[i]
                pm[j] -= steps[j]
                mp_[i] -= steps[i]
                mp_[j] += steps[j]
                mm[i] -= steps[i]
                mm[j] -= steps[j]
                fd = (
                    objective(pp)
                    - objective(pm)
                    - objective(mp_)
                    + objective(mm)
                ) / (4.0 * steps[i] * steps[j])
                assert fd == pytest.approx(hess[i][j], abs=1e-6)


def test_solver_reaches_stationarity_on_a_hard_tie_dataset():
    # F2 regression: 1800 games with 174 ties used to leave ridge-only Newton
    # at a point with max|FD gradient| ~ 1e2 and an objective far above the
    # optimum.  The returned fit must be a stationarity point of the
    # documented ordered-probit MAP objective.
    rng = random.Random(4242)
    truths = {"a": 300.0, "b": 50.0, "c": -100.0}
    s = pair_scale()
    games: list[PlayedGame] = []
    for _ in range(1800):
        seats = tuple(rng.sample([*truths, "random"], 2))
        performance = {
            id_: truths.get(id_, 0.0) + s * rng.gauss(0.0, 1.0) for id_ in seats
        }
        if max(performance.values()) - min(performance.values()) <= 60.0:
            scores = dict.fromkeys(seats, 50)
        else:
            best = max(performance, key=performance.get)
            scores = {id_: (100 if id_ == best else 0) for id_ in seats}
        games.append(
            PlayedGame(len(games), seats, tuple(scores[id_] for id_ in seats))
        )
    assert sum(len(set(game.scores)) == 1 for game in games) > 100
    priors = {id_: Prior(500.0, DEFAULT_SIGMA) for id_ in truths}
    fit = fit_mle(games, anchors={"random": 0.0}, priors=priors)
    assert fit.converged
    assert fit.iterations < 100
    mus = {id_: fit.ratings[id_].mu for id_ in truths}
    for id_ in truths:
        step = 1e-3 * max(1.0, abs(mus[id_]))
        up, down = dict(mus), dict(mus)
        up[id_] += step
        down[id_] -= step
        slope = (
            independent_nll(games, up, priors, fit.draw_margin, {"random": 0.0})
            - independent_nll(games, down, priors, fit.draw_margin, {"random": 0.0})
        ) / (2.0 * step)
        assert abs(slope) < 1e-3
    step = 1e-4 * fit.draw_margin
    slope = (
        independent_nll(games, mus, priors, fit.draw_margin + step, {"random": 0.0})
        - independent_nll(games, mus, priors, fit.draw_margin - step, {"random": 0.0})
    ) / (2.0 * step)
    assert abs(fit.draw_margin * slope) < 1e-3


def _reference_log_cdf(u: float) -> float:
    """High-precision ``log Phi(u)``: mpmath when importable, else the tail series."""
    try:
        import mpmath
    except ImportError:
        x = -u
        inverse_sq = 1.0 / (x * x)
        series = 1.0
        term = 1.0
        for order in range(1, 10):  # through x**-18
            term *= -(2 * order - 1) * inverse_sq
            series += term
        return (
            -0.5 * u * u
            - math.log(x)
            - 0.5 * math.log(2.0 * math.pi)
            + math.log(series)
        )
    mpmath.mp.dps = 60
    return float(mpmath.log(mpmath.erfc(-mpmath.mpf(u) / mpmath.sqrt(2)) / 2))


def _reference_mills(u: float) -> float:
    """High-precision ``phi(u) / Phi(u)``: mpmath when importable, else the series."""
    try:
        import mpmath
    except ImportError:
        x = -u
        inverse_sq = 1.0 / (x * x)
        series = 1.0
        term = 1.0
        for order in range(1, 10):
            term *= -(2 * order - 1) * inverse_sq
            series += term
        return x / series
    mpmath.mp.dps = 60
    value = mpmath.mpf(u)
    return float(
        mpmath.exp(-value * value / 2)
        / (mpmath.sqrt(2 * mpmath.pi) * mpmath.erfc(-value / mpmath.sqrt(2)) / 2)
    )


def test_left_tail_helpers_are_accurate_below_the_old_series_cutoff():
    # Minor F1 regression: the fixed series stopped at -15/x**6 and was
    # ~1e-6 relatively inaccurate just below u = -10, three orders above the
    # solver's gradient tolerance.  Both log Phi and the Mills ratio must now
    # track the high-precision reference to well below that.
    for u in (-10.000001, -10.5, -11.0, -12.0, -15.0, -20.0, -1e6, -4e7):
        assert _log_gaussian_cdf(u) == pytest.approx(
            _reference_log_cdf(u), rel=1e-9
        )
        assert _mills_ratio(u) == pytest.approx(_reference_mills(u), rel=1e-9)


def test_decisive_second_derivative_is_stable_at_large_u():
    # T16-4-F3 regression: the naive ``-u*m - m*m`` cancels at |u| ~ 1e7
    # (here it even flips sign), which forced the published Laplace sigma to
    # the fabricated 1e-6 floor.  The stable form must recover -1/s**2.
    s = 245.0
    for u in (-4e7, -1e8, -1e150):
        _, _, second = _decisive_terms(u, s)
        assert second == pytest.approx(-1.0 / (s * s), rel=1e-9)
    u = (500.0 - 1e7) / 0.245
    old_mills = math.exp(_log_gaussian_density(u) - _log_gaussian_cdf(u))
    naive = (-u * old_mills - old_mills * old_mills) / (s * s)
    assert naive > 0.0  # wrong sign: log Phi is concave in u
    assert _decisive_terms(u, s)[2] < 0.0


def test_large_anchor_difference_does_not_collapse_sigma_to_the_floor():
    # T16-4-F3: a 1e7 anchor gap used to publish sigma = 1e-6 with an empty
    # separation list and no warning.
    games = [
        two_player(0, "c", "big", 100, 0),
        two_player(1, "c", "small", 0, 100),
    ]
    fit = fit_mle(
        games,
        anchors={"big": 1e7, "small": -1e7},
        priors={"c": Prior(500.0, 200.0)},
    )
    sigma = fit.ratings["c"].sigma
    assert math.isfinite(fit.ratings["c"].mu)
    assert math.isfinite(sigma)
    assert 1.0 < sigma < 300.0  # not the old fabricated 1e-6 floor


def test_extreme_finite_anchors_never_produce_nan():
    # T16-4-F1 regression: 1e300 anchors overflowed sigma**2 into NaN sigmas
    # and an invalid (non-strict) JSON artifact.
    games = [
        two_player(0, "c", "big", 100, 0),
        two_player(1, "c", "small", 0, 100),
    ]
    fit = fit_mle(
        games,
        anchors={"big": 1e300, "small": -1e300},
        priors={"c": Prior(500.0, 200.0)},
    )
    for rating in fit.ratings.values():
        assert not math.isnan(rating.mu)
        assert not math.isnan(rating.sigma)
        assert rating.sigma >= 0.0
    assert math.isfinite(fit.draw_margin)


def test_prior_sigmas_whose_square_is_unrepresentable_do_not_raise():
    # T16-4-F2 regression: sigma**2 raised OverflowError at 1e308 and
    # ZeroDivisionError at 1e-300 inside fit_mle.
    decisive = [two_player(0, "a", "b", 100, 0)]
    for sigma in (1e-300, 1e308):
        fit = fit_mle(
            decisive,
            anchors={},
            priors={"a": Prior(500.0, sigma), "b": Prior(400.0, sigma)},
        )
        for id_ in ("a", "b"):
            rating = fit.ratings[id_]
            assert math.isfinite(rating.mu)
            assert not math.isnan(rating.sigma)
            assert math.isfinite(rating.sigma) or rating.sigma == math.inf
    # A tie with a subnormal width exercises the overflow-safe tie path.
    fit = fit_mle(
        [two_player(0, "a", "b", 50, 50), two_player(1, "a", "b", 100, 0)],
        anchors={},
        priors={"a": Prior(500.0, 1e308), "b": Prior(400.0, 1e308)},
    )
    for rating in fit.ratings.values():
        assert not math.isnan(rating.mu)
        assert not math.isnan(rating.sigma)


def test_margin_identified_tracks_whether_the_data_fix_the_margin():
    # T16-4-F4: the boundary (no ties) and tie-rate (all ties) margins are not
    # data-identified and must be flagged; a mixed record identifies eps.
    priors = {"a": Prior(), "b": Prior()}
    decisive = [
        two_player(i, "a", "b", 100, 0) if i % 2 else two_player(i, "a", "b", 0, 100)
        for i in range(8)
    ]
    all_ties = [two_player(i, "a", "b", 50, 50) for i in range(8)]
    mixed = [
        two_player(0, "a", "b", 50, 50),
        two_player(1, "a", "b", 100, 0),
        two_player(2, "a", "b", 0, 100),
    ]
    assert fit_mle(decisive, anchors={}, priors=priors).margin_identified is False
    assert fit_mle(all_ties, anchors={}, priors=priors).margin_identified is False
    assert fit_mle(mixed, anchors={}, priors=priors).margin_identified is True
    fixed = fit_mle(
        decisive, anchors={}, priors=priors, config=MleConfig(draw_margin=25.0)
    )
    assert fixed.margin_identified is True


def test_non_identified_laplace_sigma_is_infinite_and_not_converged():
    # T16-4-F1/F3: a Hessian diagonal that is non-finite or <= 0 must be
    # published as math.inf with converged=False and a separation flag, never
    # floored to the old 1e-6.
    priors = {"a": Prior(500.0, 1e308), "b": Prior(400.0, 1e308)}
    games = [
        two_player(0, "a", "b", 50, 50),
        two_player(1, "a", "b", 100, 0),
    ]
    fit = fit_mle(games, anchors={}, priors=priors)
    assert fit.converged is False
    for id_ in ("a", "b"):
        assert math.isinf(fit.ratings[id_].sigma)
    assert {"a", "b"} <= set(fit.separation)


def test_tie_terms_use_the_exact_interval_width():
    # Minor F2 regression: subtracting the rounded endpoints gave width
    # 1.0000889e-12 instead of the exact 2*eps/s = 1e-12, i.e. a ~1e-4
    # relative error in the tie likelihood and its eps derivative.
    s, eps, d = 400.0, 2e-10, -1600.0
    a = (eps - d) / s
    b = (-eps - d) / s
    exact = 2.0 * eps / s
    middle = 0.5 * (a + b)
    reference_log_p = (
        _log_gaussian_density(middle)
        + math.log(exact)
        + math.log(_narrow_interval_scale(middle, exact))
    )
    ratio_a = math.exp(_log_gaussian_density(a) - reference_log_p)
    ratio_b = math.exp(_log_gaussian_density(b) - reference_log_p)
    reference_le = (ratio_a + ratio_b) / s
    fixed = _tie_terms(a, b, s, exact)
    rounded = _tie_terms(a, b, s)
    assert fixed[0] == pytest.approx(reference_log_p, rel=1e-12)
    assert fixed[3] == pytest.approx(reference_le, rel=1e-9)
    assert abs(rounded[0] - reference_log_p) > 1e-8
    assert abs((rounded[3] - reference_le) / reference_le) > 1e-5


def test_evaluate_tie_derivative_uses_the_exact_width():
    # The fit must forward the exact width into _tie_terms, not the rounded
    # endpoint difference.
    s, eps, d = 400.0, 2e-10, -1600.0
    pair = _Pair(0, 1, 0.0, 0.0, s, 0)
    value, grad, _ = _evaluate(
        [-d / 2.0, d / 2.0],
        math.log(eps),
        [pair],
        [0.0, 0.0],
        [0.0, 0.0],
        estimate_eps=True,
        fixed_eps=0.0,
    )
    assert math.isfinite(value)
    a = (eps - d) / s
    b = (-eps - d) / s
    exact = 2.0 * eps / s
    middle = 0.5 * (a + b)
    reference_log_p = (
        _log_gaussian_density(middle)
        + math.log(exact)
        + math.log(_narrow_interval_scale(middle, exact))
    )
    ratio_a = math.exp(_log_gaussian_density(a) - reference_log_p)
    ratio_b = math.exp(_log_gaussian_density(b) - reference_log_p)
    reference_le = (ratio_a + ratio_b) / s
    assert grad[2] == pytest.approx(-eps * reference_le, rel=1e-9)

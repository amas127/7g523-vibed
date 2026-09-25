"""Candidate-vs-candidate 换座对局: the audit's §6.5 head-to-head estimator.

Absolute anchor Elo is a noisy currency for comparing two candidates: the
anchors are far away, the logistic mapping saturates, and (per
``docs/experiments/elo-reliability-audit.md`` §3.5) the parent/child offset
even flips sign from one deal set to the next.  This module implements the
audit's recommendation (§6.5): seat the two candidates **against each other on
the same deal, once per seat**, and cluster the statistics by 牌 (the twin
pair).  The comparison then only depends on the head-to-head outcome, not on
the anchors' distance or on the candidate's seat phase.

The module is pure in-process math — no torch, no numpy, no clock, no global
RNG.  The bootstrap takes an explicit :class:`random.Random`, so the same seed
reproduces the same interval bit for bit.  ``ladder.play_games`` is the only
orchestration seam: :func:`plan_duel_schedule` produces the same
``ScheduledGame`` schedule type, so the CLI is a thin argparse shell.
"""
from __future__ import annotations

import math
import random
import statistics
from typing import Mapping, Sequence

from .elo import DEFAULT_ELO_SCALE, PlayedGame
from .ladder import Entrant, ScheduledGame

__all__ = ["combine_duel_seeds", "paired_duel_stats", "plan_duel_schedule"]

#: The 400-point logistic scale, shared with :mod:`seven523.elo`.


def plan_duel_schedule(
    left: Entrant, right: Entrant, *, pairs: int, seed: int
) -> tuple[ScheduledGame, ...]:
    """Plan a seat-balanced head-to-head: ``pairs`` deals, two games per deal.

    Every deal seed is reused for a twin pair — ``left`` on seat 0 then seat 1 —
    so hand and seat asymmetry cancels inside each deal and the residual
    bootstrap only resamples 牌.  ``subject`` is always ``left.id``, which
    keeps every trace of the duel under one directory.  ``pairs`` is a deal
    count (never a game count); odd values are fine.  Pure and deterministic.
    """
    if left.id == right.id:
        raise ValueError(
            f"a duel needs two distinct ids, got {left.id!r} on both sides"
        )
    if pairs < 1:
        raise ValueError(f"pairs must be at least 1, got {pairs}")
    master = random.Random(seed)
    schedule: list[ScheduledGame] = []
    for _ in range(pairs):
        deal_seed = master.randrange(1 << 32)
        schedule.append(ScheduledGame(deal_seed, (left.id, right.id), left.id))
        schedule.append(ScheduledGame(deal_seed, (right.id, left.id), left.id))
    return tuple(schedule)


def _clamp_probability(p: float, deals: int) -> float:
    """Keep ``p`` inside the range a finite sample can actually produce.

    With ``deals`` twin pairs there are ``2 * deals`` games, so the smallest
    non-zero relative frequency is ``1 / (2 * deals)``; clamping there keeps
    ``log10(p / (1 - p))`` finite for all-win / all-loss bootstrap resamples
    (the Elo difference simply saturates).
    """
    floor = 1.0 / (2 * deals)
    return min(max(p, floor), 1.0 - floor)


def _elo_diff(p: float, deals: int) -> float:
    """``400 * log10(p / (1 - p))`` — the left-minus-right Elo difference."""
    p = _clamp_probability(p, deals)
    return DEFAULT_ELO_SCALE * math.log10(p / (1.0 - p))


def _quantile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated quantile of an already-sorted sequence."""
    if not values:
        raise ValueError("quantile of an empty sample")
    if len(values) == 1:
        return values[0]
    position = q * (len(values) - 1)
    lower = int(math.floor(position))
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def _binomial_two_sided_p(successes: int, trials: int) -> float:
    """Exact two-sided binomial p-value for ``H0: p = 0.5`` (pure ``math``).

    Sums every outcome whose probability is no larger than the observed one,
    the standard Clopper-like two-sided definition; ties contribute nothing
    and yield ``p = 1``.
    """
    if trials <= 0:
        return 1.0
    pmf = [math.comb(trials, k) for k in range(trials + 1)]
    observed = pmf[min(max(successes, 0), trials)]
    return min(1.0, sum(value for value in pmf if value <= observed) / (1 << trials))


def paired_duel_stats(
    games: Sequence[PlayedGame],
    *,
    left_id: str,
    right_id: str,
    bootstrap: int = 4000,
    rng: random.Random,
    confidence: float = 0.95,
) -> dict:
    """Summarise a seat-balanced duel from the left candidate's viewpoint.

    ``games`` must contain exactly two games per 牌 seed — one with each
    candidate on seat 0 — as produced by :func:`plan_duel_schedule` and
    ``ladder.play_games``.  The clustering unit is that twin pair: the
    bootstrap resamples **deals** (with replacement), so the paired seat
    structure and the within-deal correlation are preserved.

    Reported (all JSON-serialisable):

    * ``wins``/``draws``/``losses`` over games and ``winrate`` =
      ``(wins + 0.5 * draws) / games`` (the expected-score convention Elo
      uses, so an all-win duel maps to the finite clamp);
    * ``mean_score_diff`` (0–100 point scale) and ``mean_return`` (÷100);
    * percentile CIs at ``confidence`` for the winrate, the mean margin and
      ``elo_diff = 400 * log10(p / (1 - p))`` with ``p`` the per-deal mean
      winrate, clamped to ``[1 / (2 * deals), 1 - 1 / (2 * deals)]``;
    * the per-deal sign test: how many deals the left candidate won outright
      on the summed twin margin, lost, or tied, with an exact binomial p-value.

    ``rng`` must be an explicit :class:`random.Random`; the result is a
    deterministic function of it, and the input order is irrelevant because
    deals are processed in seed order.
    """
    if bootstrap < 1:
        raise ValueError(f"bootstrap must be at least 1, got {bootstrap}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must lie in (0, 1), got {confidence}")
    if left_id == right_id:
        raise ValueError(f"left_id and right_id must differ, got {left_id!r}")
    if not games:
        raise ValueError("no games to summarise")

    by_deal: dict[int, list[PlayedGame]] = {}
    for game in games:
        if len(game.seats) != 2:
            raise ValueError(f"a duel game needs exactly 2 seats, got {game.seats!r}")
        missing = {left_id, right_id} - set(game.seats)
        if missing:
            raise ValueError(f"game on seed {game.seed} is missing {sorted(missing)}")
        by_deal.setdefault(game.seed, []).append(game)

    per_deal: list[tuple[float, float]] = []
    wins = draws = losses = 0
    margin_total = 0.0
    net_wins = net_losses = net_ties = 0
    for seed in sorted(by_deal):
        twins = by_deal[seed]
        if len(twins) != 2:
            raise ValueError(
                f"deal {seed} has {len(twins)} games; "
                "a duel needs one twin per seat"
            )
        seats = sorted(game.seats.index(left_id) for game in twins)
        if seats != [0, 1]:
            raise ValueError(
                f"deal {seed} does not seat {left_id!r} on both seats (got {seats})"
            )
        deal_wins = deal_draws = deal_losses = 0
        deal_margin = 0.0
        for game in twins:
            seat = game.seats.index(left_id)
            diff = game.scores[seat] - game.scores[1 - seat]
            deal_margin += diff
            if diff > 0:
                deal_wins += 1
            elif diff < 0:
                deal_losses += 1
            else:
                deal_draws += 1
        wins += deal_wins
        draws += deal_draws
        losses += deal_losses
        margin_total += deal_margin
        net = deal_wins - deal_losses
        if net > 0:
            net_wins += 1
        elif net < 0:
            net_losses += 1
        else:
            net_ties += 1
        # Each deal contributes its own average over the two twin games, so
        # resampling deals is exactly a 牌-cluster bootstrap.
        per_deal.append(((deal_wins + 0.5 * deal_draws) / 2.0, deal_margin / 2.0))

    deals = len(per_deal)
    total_games = 2 * deals
    winrate = (wins + 0.5 * draws) / total_games
    mean_score_diff = margin_total / total_games
    elo_diff = _elo_diff(winrate, deals)

    winrates: list[float] = []
    margins: list[float] = []
    elo_diffs: list[float] = []
    for _ in range(bootstrap):
        winrate_sum = 0.0
        margin_sum = 0.0
        for _ in range(deals):
            deal_winrate, deal_margin = per_deal[rng.randrange(deals)]
            winrate_sum += deal_winrate
            margin_sum += deal_margin
        p = winrate_sum / deals
        winrates.append(p)
        margins.append(margin_sum / deals)
        elo_diffs.append(_elo_diff(p, deals))
    winrates.sort()
    margins.sort()
    elo_diffs.sort()

    alpha = (1.0 - confidence) / 2.0
    sign_p = _binomial_two_sided_p(net_wins, net_wins + net_losses)
    return {
        "left_id": left_id,
        "right_id": right_id,
        "deals": deals,
        "games": total_games,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "winrate": winrate,
        "winrate_ci": [_quantile(winrates, alpha), _quantile(winrates, 1.0 - alpha)],
        "mean_score_diff": mean_score_diff,
        "mean_score_diff_ci": [
            _quantile(margins, alpha),
            _quantile(margins, 1.0 - alpha),
        ],
        "mean_return": mean_score_diff / 100.0,
        "elo_diff": elo_diff,
        "elo_diff_ci": [
            _quantile(elo_diffs, alpha),
            _quantile(elo_diffs, 1.0 - alpha),
        ],
        "deal_sign": {
            "left_net_wins": net_wins,
            "left_net_losses": net_losses,
            "ties": net_ties,
            "p_value": sign_p,
        },
        "bootstrap": bootstrap,
        "confidence": confidence,
    }


def _combine_metric(
    values: Sequence[float],
    lows: Sequence[float],
    highs: Sequence[float],
    z: float,
) -> dict:
    """One metric's multi-seed readout under the wave5 §4.2 pooling rule.

    Point estimate = arithmetic mean of the per-seed values.  The bootstrap
    term is the root-mean-square of the per-seed implied SEs
    (``half_width / z``); the between-seed term is the sample sd (ddof=1) of
    the point estimates.  The combined SE is ``max(both) / sqrt(k)`` and the CI
    is ``mean ± z * SE`` — conservative in the sense that it never reads finer
    than either the within-seed bootstrap or the seed-to-seed spread allows.
    """
    count = len(values)
    implied = [(high - low) / (2.0 * z) for low, high in zip(lows, highs)]
    bootstrap_se = math.sqrt(sum(se * se for se in implied) / count)
    between_sd = statistics.stdev(values)
    mean = sum(values) / count
    se = max(bootstrap_se, between_sd) / math.sqrt(count)
    half_width = z * se
    return {
        "mean": mean,
        "ci": [mean - half_width, mean + half_width],
        "se": se,
        "half_width": half_width,
        "bootstrap_se": bootstrap_se,
        "between_seed_sd": between_sd,
    }


def combine_duel_seeds(
    per_seed: Sequence[Mapping[str, object]],
    *,
    seeds: Sequence[int] | None = None,
    z: float = 1.96,
) -> dict:
    """Merge several :func:`paired_duel_stats` runs of the same pair.

    The audit/pilot rule (``docs/experiments/wave5-500k-report.md`` §4.2, used
    by ``head-to-head-pilot.md`` §3.2): point estimate = mean of the per-seed
    values; CI = ``max(bootstrap-implied SE, between-seed sd) / sqrt(k) * z``
    with ``z = 1.96`` for the 95% interval.  Each of ``elo_diff``, ``winrate``
    and ``mean_score_diff`` is merged that way, the deal sign test is pooled,
    and the per-seed dicts are echoed under ``per_seed`` (tagged with ``seed``
    when ``seeds`` is given).  All runs must share the pair, bootstrap count
    and confidence; at least two seeds are required (a single run is already a
    complete answer and is returned unchanged by the CLI).
    """
    count = len(per_seed)
    if count < 2:
        raise ValueError(
            f"combining needs at least 2 seeds, got {count}; "
            "a single run is already a complete reading"
        )
    if z <= 0.0:
        raise ValueError(f"z must be positive, got {z}")
    if seeds is not None and len(seeds) != count:
        raise ValueError(f"got {len(seeds)} seeds for {count} runs")

    left_id = per_seed[0]["left_id"]
    right_id = per_seed[0]["right_id"]
    bootstrap = per_seed[0]["bootstrap"]
    confidence = per_seed[0]["confidence"]
    for index, stats in enumerate(per_seed):
        if stats["left_id"] != left_id or stats["right_id"] != right_id:
            raise ValueError(
                f"run {index} compares {stats['left_id']!r} vs {stats['right_id']!r}, "
                f"expected {left_id!r} vs {right_id!r}"
            )
        if stats["bootstrap"] != bootstrap or stats["confidence"] != confidence:
            raise ValueError(
                f"run {index} used bootstrap={stats['bootstrap']!r} "
                f"confidence={stats['confidence']!r}, expected {bootstrap!r} / "
                f"{confidence!r}"
            )

    metrics: dict[str, dict] = {}
    for name in ("elo_diff", "winrate", "mean_score_diff"):
        try:
            values = [float(stats[name]) for stats in per_seed]  # type: ignore[index,arg-type]
            intervals = [stats[f"{name}_ci"] for stats in per_seed]  # type: ignore[index]
            lows = [float(pair[0]) for pair in intervals]  # type: ignore[index]
            highs = [float(pair[1]) for pair in intervals]  # type: ignore[index]
        except (KeyError, TypeError, IndexError) as exc:
            raise ValueError(f"run is missing a usable {name!r} entry") from exc
        metrics[name] = _combine_metric(values, lows, highs, z)

    net_wins = net_losses = ties = 0
    for stats in per_seed:
        sign = stats["deal_sign"]  # type: ignore[index]
        net_wins += int(sign["left_net_wins"])  # type: ignore[index]
        net_losses += int(sign["left_net_losses"])  # type: ignore[index]
        ties += int(sign["ties"])  # type: ignore[index]

    tagged: list[dict] = [dict(stats) for stats in per_seed]
    if seeds is not None:
        for seed, stats in zip(seeds, tagged):
            stats["seed"] = seed

    return {
        "left_id": left_id,
        "right_id": right_id,
        "k": count,
        "seeds": list(seeds) if seeds is not None else None,
        "deals": sum(int(stats["deals"]) for stats in per_seed),  # type: ignore[arg-type]
        "games": sum(int(stats["games"]) for stats in per_seed),  # type: ignore[arg-type]
        "bootstrap": bootstrap,
        "confidence": confidence,
        "z": z,
        "per_seed": tagged,
        "combined": {
            **metrics,
            "deal_sign": {
                "left_net_wins": net_wins,
                "left_net_losses": net_losses,
                "ties": ties,
                "p_value": _binomial_two_sided_p(net_wins, net_wins + net_losses),
            },
        },
        "method": (
            "point = per-seed mean; CI = max(bootstrap-implied SE, "
            "between-seed sd) / sqrt(k) * z (wave5-500k-report.md §4.2)"
        ),
    }

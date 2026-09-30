"""Head-to-head duel: twin schedule, paired bootstrap, and the CLI smoke."""
from __future__ import annotations

import importlib.util
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import pytest

from seven523.duel import (
    _binomial_two_sided_p,
    combine_duel_seeds,
    paired_duel_stats,
    plan_duel_schedule,
)
from seven523.elo import PlayedGame
from seven523.ladder import Entrant

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"


def _load_tool(name: str):
    spec = importlib.util.spec_from_file_location(name, TOOLS_DIR / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _game(
    seed: int, left_seat: int, left_score: int, right_score: int
) -> PlayedGame:
    """One twin game from the left candidate's seat."""
    if left_seat == 0:
        return PlayedGame(seed, ("a", "b"), (left_score, right_score))
    return PlayedGame(seed, ("b", "a"), (right_score, left_score))


def test_plan_duel_schedule_pairs_deals_and_swaps_seats():
    left, right = Entrant("a", "random"), Entrant("b", "random")
    schedule = plan_duel_schedule(left, right, pairs=3, seed=11)
    assert len(schedule) == 6  # pairs is a deal count, not a game count

    twins: dict[int, list] = defaultdict(list)
    for game in schedule:
        assert game.subject == "a"
        assert game.seats in {("a", "b"), ("b", "a")}
        twins[game.seed].append(game)
    assert len(twins) == 3
    for games in twins.values():
        assert len(games) == 2
        assert sorted(game.seats.index("a") for game in games) == [0, 1]

    assert plan_duel_schedule(left, right, pairs=3, seed=11) == schedule
    assert len(plan_duel_schedule(left, right, pairs=1, seed=0)) == 2  # odd pairs ok
    with pytest.raises(ValueError):
        plan_duel_schedule(left, right, pairs=0, seed=0)
    with pytest.raises(ValueError):
        plan_duel_schedule(left, left, pairs=1, seed=0)


def test_paired_duel_stats_all_wins_saturates_the_clamp():
    games = [
        _game(seed, seat, 80, 20) for seed in range(4) for seat in (0, 1)
    ]
    stats = paired_duel_stats(
        games, left_id="a", right_id="b", bootstrap=200, rng=random.Random(0)
    )
    assert stats["deals"] == 4
    assert stats["games"] == 8
    assert (stats["wins"], stats["draws"], stats["losses"]) == (8, 0, 0)
    assert stats["winrate"] == 1.0
    assert stats["mean_score_diff"] == pytest.approx(60.0)
    assert stats["mean_return"] == pytest.approx(0.6)
    saturated = 400.0 * math.log10(0.875 / 0.125)  # p clamped to 1 - 1/(2*4)
    assert stats["elo_diff"] == pytest.approx(saturated)
    assert stats["winrate_ci"] == [1.0, 1.0]
    assert stats["elo_diff_ci"] == pytest.approx([saturated, saturated])
    assert stats["deal_sign"]["left_net_wins"] == 4
    assert stats["deal_sign"]["left_net_losses"] == 0
    assert stats["deal_sign"]["ties"] == 0
    assert stats["deal_sign"]["p_value"] == pytest.approx(0.125)


def test_paired_duel_stats_symmetric_zero_net_has_degenerate_zero_ci():
    games = []
    for seed in range(6):
        games.append(_game(seed, 0, 70, 30))
        games.append(_game(seed, 1, 30, 70))  # the twin loses the same margin
    stats = paired_duel_stats(
        games, left_id="a", right_id="b", bootstrap=300, rng=random.Random(1)
    )
    assert stats["winrate"] == 0.5
    assert stats["mean_score_diff"] == 0.0
    assert stats["elo_diff"] == 0.0
    assert stats["winrate_ci"] == [0.5, 0.5]
    assert stats["mean_score_diff_ci"] == [0.0, 0.0]
    assert stats["elo_diff_ci"] == [0.0, 0.0]
    assert stats["deal_sign"] == {
        "left_net_wins": 0,
        "left_net_losses": 0,
        "ties": 6,
        "p_value": 1.0,
    }


def test_paired_duel_stats_balanced_mix_ci_covers_zero():
    games = []
    for seed in range(10):
        if seed % 2 == 0:
            games.append(_game(seed, 0, 75, 25))
            games.append(_game(seed, 1, 75, 25))
        else:
            games.append(_game(seed, 0, 25, 75))
            games.append(_game(seed, 1, 25, 75))
    stats = paired_duel_stats(
        games, left_id="a", right_id="b", bootstrap=2000, rng=random.Random(2)
    )
    assert stats["winrate"] == 0.5
    assert stats["elo_diff"] == 0.0
    assert stats["winrate_ci"][0] < 0.5 < stats["winrate_ci"][1]
    assert stats["elo_diff_ci"][0] < 0.0 < stats["elo_diff_ci"][1]
    assert stats["deal_sign"]["left_net_wins"] == 5
    assert stats["deal_sign"]["left_net_losses"] == 5
    assert stats["deal_sign"]["p_value"] == pytest.approx(1.0)


def test_paired_duel_stats_bootstrap_is_deterministic_and_order_free():
    games = []
    for seed in range(40):
        if seed % 4 == 0:  # left loses both twins
            games.append(_game(seed, 0, 30, 70))
            games.append(_game(seed, 1, 35, 65))
        elif seed % 4 == 1:  # split twins
            games.append(_game(seed, 0, 80, 20))
            games.append(_game(seed, 1, 45, 55))
        else:  # left wins both twins
            games.append(_game(seed, 0, 60 + seed % 5, 40 - seed % 5))
            games.append(_game(seed, 1, 55 + seed % 3, 45 - seed % 3))
    first = paired_duel_stats(
        games, left_id="a", right_id="b", bootstrap=500, rng=random.Random(7)
    )
    second = paired_duel_stats(
        games, left_id="a", right_id="b", bootstrap=500, rng=random.Random(7)
    )
    assert first == second
    shuffled = list(reversed(games))
    assert (
        paired_duel_stats(
            shuffled, left_id="a", right_id="b", bootstrap=500, rng=random.Random(7)
        )
        == first
    )
    different = paired_duel_stats(
        games, left_id="a", right_id="b", bootstrap=500, rng=random.Random(8)
    )
    assert different["elo_diff_ci"] != first["elo_diff_ci"]


def test_paired_duel_stats_swapped_ids_mirror_every_statistic():
    games = []
    for seed in range(9):
        games.append(_game(seed, 0, 64, 36))
        games.append(_game(seed, 1, 22, 78) if seed % 4 else _game(seed, 1, 58, 42))
    forward = paired_duel_stats(
        games,
        left_id="a",
        right_id="b",
        bootstrap=500,
        rng=random.Random(3),
    )
    mirrored_games = [
        PlayedGame(game.seed, (game.seats[1], game.seats[0]), (game.scores[1], game.scores[0]))
        for game in games
    ]
    backward = paired_duel_stats(
        mirrored_games,
        left_id="b",
        right_id="a",
        bootstrap=500,
        rng=random.Random(3),
    )
    assert backward["winrate"] == pytest.approx(1.0 - forward["winrate"])
    assert backward["mean_score_diff"] == pytest.approx(-forward["mean_score_diff"])
    assert backward["elo_diff"] == pytest.approx(-forward["elo_diff"])
    assert backward["winrate_ci"][0] == pytest.approx(1.0 - forward["winrate_ci"][1])
    assert backward["winrate_ci"][1] == pytest.approx(1.0 - forward["winrate_ci"][0])
    assert backward["elo_diff_ci"][0] == pytest.approx(-forward["elo_diff_ci"][1])
    assert backward["elo_diff_ci"][1] == pytest.approx(-forward["elo_diff_ci"][0])


def test_paired_duel_stats_validation():
    good = [_game(0, 0, 60, 40), _game(0, 1, 60, 40)]
    with pytest.raises(ValueError):
        paired_duel_stats([], left_id="a", right_id="b", rng=random.Random(0))
    with pytest.raises(ValueError):  # missing twin
        paired_duel_stats(good[:1], left_id="a", right_id="b", rng=random.Random(0))
    with pytest.raises(ValueError):  # left repeated on one seat
        paired_duel_stats(
            [good[0], _game(0, 0, 60, 40)],
            left_id="a",
            right_id="b",
            rng=random.Random(0),
        )
    with pytest.raises(ValueError):  # right id not seated
        paired_duel_stats(
            [PlayedGame(0, ("a", "c"), (60, 40)), PlayedGame(0, ("c", "a"), (60, 40))],
            left_id="a",
            right_id="b",
            rng=random.Random(0),
        )
    with pytest.raises(ValueError):
        paired_duel_stats(good, left_id="a", right_id="a", rng=random.Random(0))
    with pytest.raises(ValueError):
        paired_duel_stats(
            good, left_id="a", right_id="b", bootstrap=0, rng=random.Random(0)
        )
    with pytest.raises(ValueError):
        paired_duel_stats(
            good,
            left_id="a",
            right_id="b",
            rng=random.Random(0),
            confidence=1.0,
        )


def _legacy_binomial_two_sided_p(successes: int, trials: int) -> float:
    """The pre-fix formula, inlined so the regression pins old == new at n=800."""
    if trials <= 0:
        return 1.0
    pmf = [math.comb(trials, k) for k in range(trials + 1)]
    observed = pmf[min(max(successes, 0), trials)]
    return min(
        1.0,
        sum(value for value in pmf if value <= observed) / float(1 << trials),
    )


def test_binomial_two_sided_p_matches_legacy_formula_at_800_trials():
    trials = 800
    rng = random.Random(20240517)
    successes = [0, 1, 200, 399, 400, 401, 600, 799, 800]
    successes += [rng.randrange(trials + 1) for _ in range(8)]
    for k in successes:
        assert _binomial_two_sided_p(k, trials) == _legacy_binomial_two_sided_p(k, trials)


def test_binomial_two_sided_p_does_not_overflow_trial_counts():
    # ``float(1 << 1024)`` already overflows, so 1600 trials exercise the guard
    # that the old 3200-trial case reached twice as slowly.
    trials = 1600
    for k in (0, 1, trials // 3, trials // 2, trials):
        p = _binomial_two_sided_p(k, trials)
        assert 0.0 <= p <= 1.0
    assert _binomial_two_sided_p(800, trials) == 1.0
    for k in (0, 1, trials // 2, trials):
        assert _binomial_two_sided_p(k, trials) == _binomial_two_sided_p(
            trials - k, trials
        )


def test_binomial_two_sided_p_nonpositive_trials_is_one():
    for trials in (-3, 0):
        assert _binomial_two_sided_p(0, trials) == 1.0
        assert _binomial_two_sided_p(5, trials) == 1.0


def test_cli_smoke_json_fields(capsys):
    tool = _load_tool("head_to_head")
    code = tool.main(
        [
            "--left", "a=random",
            "--right", "b=random",
            "--pairs", "4",
            "--seed", "0",
            "--bootstrap", "200",
            "--json",
        ]
    )
    assert code == 0
    stats = json.loads(capsys.readouterr().out)
    assert set(stats) == {
        "left_id", "right_id", "deals", "games", "wins", "draws", "losses",
        "winrate", "winrate_ci", "mean_score_diff", "mean_score_diff_ci",
        "mean_return", "elo_diff", "elo_diff_ci", "deal_sign", "bootstrap",
        "confidence",
    }
    assert stats["left_id"] == "a"
    assert stats["right_id"] == "b"
    assert stats["deals"] == 4
    assert stats["games"] == 8
    assert stats["wins"] + stats["draws"] + stats["losses"] == 8
    assert len(stats["winrate_ci"]) == len(stats["elo_diff_ci"]) == 2
    assert set(stats["deal_sign"]) == {
        "left_net_wins", "left_net_losses", "ties", "p_value",
    }


def test_cli_swapped_sides_mirror_exactly(capsys):
    tool = _load_tool("head_to_head")
    common = ["--pairs", "6", "--seed", "5", "--bootstrap", "300", "--json"]
    assert tool.main(["--left", "a=random", "--right", "b=random", *common]) == 0
    forward = json.loads(capsys.readouterr().out)
    assert tool.main(["--left", "b=random", "--right", "a=random", *common]) == 0
    backward = json.loads(capsys.readouterr().out)
    # The swapped schedule is the same physical deal set with the ids renamed,
    # so every left-view statistic mirrors exactly.
    assert forward["winrate"] + backward["winrate"] == pytest.approx(1.0)
    assert forward["elo_diff"] == pytest.approx(-backward["elo_diff"])
    assert forward["winrate_ci"][0] == pytest.approx(1.0 - backward["winrate_ci"][1])
    assert forward["winrate_ci"][1] == pytest.approx(1.0 - backward["winrate_ci"][0])
    assert forward["mean_score_diff"] == pytest.approx(-backward["mean_score_diff"])


def test_cli_rejects_bad_sides(capsys):
    tool = _load_tool("head_to_head")
    with pytest.raises(SystemExit):
        tool.main(["--left", "a=random", "--right", "a=random", "--pairs", "1"])
    with pytest.raises(SystemExit):
        tool.main(["--left", "nope", "--right", "b=random"])
    with pytest.raises(SystemExit):
        tool.main(["--left", "a=random", "--right", "b=random", "--pairs", "0"])


def _fake_stats(
    elo_diff: float,
    half_width: float,
    *,
    left: str = "a",
    right: str = "b",
    bootstrap: int = 1000,
    confidence: float = 0.95,
    net: tuple[int, int, int] = (1, 1, 2),
) -> dict:
    """A minimal dict with the keys :func:`combine_duel_seeds` reads."""
    index = elo_diff / 10000.0  # keeps winrate in range without affecting elo
    net_wins, net_losses, ties = net
    return {
        "left_id": left,
        "right_id": right,
        "deals": 4,
        "games": 8,
        "bootstrap": bootstrap,
        "confidence": confidence,
        "elo_diff": elo_diff,
        "elo_diff_ci": [elo_diff - half_width, elo_diff + half_width],
        "winrate": 0.5 + index,
        "winrate_ci": [0.5 + index - half_width / 10000.0, 0.5 + index + half_width / 10000.0],
        "mean_score_diff": elo_diff / 4.0,
        "mean_score_diff_ci": [
            elo_diff / 4.0 - half_width / 4.0,
            elo_diff / 4.0 + half_width / 4.0,
        ],
        "deal_sign": {
            "left_net_wins": net_wins,
            "left_net_losses": net_losses,
            "ties": ties,
            "p_value": 1.0,
        },
    }


def test_combine_duel_seeds_wave5_pooling_math():
    fake = [_fake_stats(elo, 10.0) for elo in (10.0, 20.0, 30.0)]
    combined = combine_duel_seeds(fake, seeds=[0, 1, 2])
    assert combined["k"] == 3
    assert combined["seeds"] == [0, 1, 2]
    assert [stats["seed"] for stats in combined["per_seed"]] == [0, 1, 2]
    assert combined["deals"] == 12 and combined["games"] == 24
    elo = combined["combined"]["elo_diff"]
    between_sd = statistics.stdev([10.0, 20.0, 30.0])
    assert elo["mean"] == pytest.approx(20.0)
    assert elo["bootstrap_se"] == pytest.approx(10.0 / 1.96)
    assert elo["between_seed_sd"] == pytest.approx(between_sd)
    assert elo["se"] == pytest.approx(between_sd / math.sqrt(3))
    assert elo["half_width"] == pytest.approx(elo["se"] * 1.96)
    assert elo["ci"] == pytest.approx([20.0 - elo["half_width"], 20.0 + elo["half_width"]])
    assert combined["combined"]["deal_sign"] == {
        "left_net_wins": 3,
        "left_net_losses": 3,
        "ties": 6,
        "p_value": 1.0,
    }


def test_combine_duel_seeds_bootstrap_dominates_when_wider():
    fake = [_fake_stats(0.0, 50.0) for _ in range(3)]
    elo = combine_duel_seeds(fake)["combined"]["elo_diff"]
    assert elo["between_seed_sd"] == 0.0
    assert elo["bootstrap_se"] == pytest.approx(50.0 / 1.96)
    assert elo["se"] == pytest.approx(50.0 / 1.96 / math.sqrt(3))
    assert elo["half_width"] == pytest.approx(50.0 / math.sqrt(3))


def test_combine_duel_seeds_accepts_real_paired_stats():
    per_seed = []
    for seed in (0, 1, 2):
        games = []
        for deal in range(3):
            games.append(_game(seed * 10 + deal, 0, 60 + deal, 40))
            games.append(_game(seed * 10 + deal, 1, 60 + deal, 40))
        per_seed.append(
            paired_duel_stats(
                games, left_id="a", right_id="b", bootstrap=100, rng=random.Random(seed)
            )
        )
    combined = combine_duel_seeds(per_seed, seeds=[0, 1, 2])
    assert combined["deals"] == 9 and combined["games"] == 18
    elo = combined["combined"]["elo_diff"]
    assert elo["mean"] == pytest.approx(
        sum(stats["elo_diff"] for stats in per_seed) / 3.0
    )
    assert combined["combined"]["deal_sign"]["left_net_wins"] == 9
    assert combined["combined"]["deal_sign"]["p_value"] == pytest.approx(2.0 / 2**9)


def test_combine_duel_seeds_validation():
    good = [_fake_stats(0.0, 10.0), _fake_stats(0.0, 10.0)]
    with pytest.raises(ValueError):  # a single run is already complete
        combine_duel_seeds(good[:1])
    with pytest.raises(ValueError):  # different pair
        combine_duel_seeds([good[0], _fake_stats(0.0, 10.0, right="c")])
    with pytest.raises(ValueError):  # bootstrap must match
        combine_duel_seeds([good[0], {**good[1], "bootstrap": 999}])
    with pytest.raises(ValueError):  # confidence must match
        combine_duel_seeds([good[0], {**good[1], "confidence": 0.9}])
    with pytest.raises(ValueError):  # seed labels must line up
        combine_duel_seeds(good, seeds=[0])
    with pytest.raises(ValueError):
        combine_duel_seeds(good, z=0.0)


def test_cli_multi_seed_json_merges(capsys):
    tool = _load_tool("head_to_head")
    code = tool.main(
        [
            "--left", "a=random",
            "--right", "b=random",
            "--pairs", "2",
            "--seeds", "0,1,2",
            "--bootstrap", "100",
            "--json",
        ]
    )
    assert code == 0
    document = json.loads(capsys.readouterr().out)
    assert document["k"] == 3
    assert document["seeds"] == [0, 1, 2]
    assert document["deals"] == 6 and document["games"] == 12
    assert [stats["seed"] for stats in document["per_seed"]] == [0, 1, 2]
    assert set(document["combined"]) == {
        "elo_diff", "winrate", "mean_score_diff", "deal_sign",
    }
    for name in ("elo_diff", "winrate", "mean_score_diff"):
        metric = document["combined"][name]
        assert set(metric) == {
            "mean", "ci", "se", "half_width", "bootstrap_se", "between_seed_sd",
        }
        values = [stats[name] for stats in document["per_seed"]]
        assert metric["mean"] == pytest.approx(sum(values) / 3.0)
        assert metric["between_seed_sd"] == pytest.approx(statistics.stdev(values))
        assert metric["se"] == pytest.approx(
            max(metric["bootstrap_se"], metric["between_seed_sd"]) / math.sqrt(3)
        )
        assert metric["half_width"] == pytest.approx(1.96 * metric["se"])
        assert metric["ci"] == pytest.approx(
            [metric["mean"] - metric["half_width"], metric["mean"] + metric["half_width"]]
        )
        for stats in document["per_seed"]:
            assert len(stats[f"{name}_ci"]) == 2
    sign = document["combined"]["deal_sign"]
    assert sign["left_net_wins"] + sign["left_net_losses"] + sign["ties"] == 6


def test_cli_single_seed_via_seeds_matches_seed_flag(capsys):
    tool = _load_tool("head_to_head")
    common = [
        "--left", "a=random", "--right", "b=random",
        "--pairs", "3", "--bootstrap", "200", "--json",
    ]
    assert tool.main([*common, "--seed", "4"]) == 0
    single = capsys.readouterr().out
    assert tool.main([*common, "--seeds", "4"]) == 0
    assert capsys.readouterr().out == single
    # The single-seed contract is still exactly the old stats dict.
    stats = json.loads(single)
    assert stats["deals"] == 3
    assert "seed" not in stats


def test_cli_seed_flag_validation(capsys):
    tool = _load_tool("head_to_head")
    with pytest.raises(SystemExit):  # mutually exclusive
        tool.main(
            ["--left", "a=random", "--right", "b=random", "--seed", "0", "--seeds", "0,1"]
        )
    with pytest.raises(SystemExit):
        tool.main(["--left", "a=random", "--right", "b=random", "--seeds", "x"])
    with pytest.raises(SystemExit):
        tool.main(["--left", "a=random", "--right", "b=random", "--seeds", ""])


def test_cli_multi_seed_games_out_splits_by_seed(tmp_path, capsys):
    tool = _load_tool("head_to_head")
    path = tmp_path / "games.jsonl"
    code = tool.main(
        [
            "--left", "a=random", "--right", "b=random",
            "--pairs", "2", "--seeds", "0,1", "--bootstrap", "100",
            "--json", "--games-out", str(path),
        ]
    )
    assert code == 0
    capsys.readouterr()
    for seed in (0, 1):
        lines = path.with_name(f"games_seed{seed}.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        assert len(lines) == 4  # 2 deals x 2 seats
        for line in lines:
            assert set(json.loads(line)) == {
                "seed", "seats", "scores", "subject", "opponent",
                "subject_seat", "kind", "rules_id",
            }
    assert not path.exists()  # multi-seed never writes the bare path


def test_cli_single_seed_games_out_writes_exact_path(tmp_path, capsys):
    tool = _load_tool("head_to_head")
    path = tmp_path / "games.jsonl"
    assert tool.main(
        [
            "--left", "a=random", "--right", "b=random",
            "--pairs", "2", "--seed", "0", "--bootstrap", "100",
            "--json", "--games-out", str(path),
        ]
    ) == 0
    capsys.readouterr()
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4
    assert all(json.loads(line)["kind"] == "cross" for line in lines)


def test_h2h_screen_smoke(tmp_path, capsys):
    tool = _load_tool("h2h_screen")
    summary_path = tmp_path / "summary.json"
    games_dir = tmp_path / "games"
    code = tool.main(
        [
            "--pair", "a=random", "b=random",
            "--pair", "b=random", "a=random",
            "--seeds", "0,1",
            "--pairs", "2",
            "--bootstrap", "100",
            "--json",
            "--out", str(summary_path),
            "--games-out", str(games_dir),
        ]
    )
    assert code == 0
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["seeds"] == [0, 1]
    assert summary["pairs_per_seed"] == 2
    assert len(summary["results"]) == 2
    for entry in summary["results"]:
        result = entry["result"]
        assert result["k"] == 2
        assert result["deals"] == 4 and result["games"] == 8
        metric = result["combined"]["elo_diff"]
        assert metric["ci"][0] <= metric["mean"] <= metric["ci"][1]
    assert len(list(games_dir.glob("*.jsonl"))) == 4  # 2 pairs x 2 seeds
    assert json.loads(capsys.readouterr().out) == summary

"""P4b deal-twin tests: schedule, paired estimator and the session pipeline.

Everything here is torch-free: the raw arm's ``policy_from_spec`` is
monkeypatched to a scripted bot and the search arm uses a stub factory that
records its calls, so the only torch-loading path (``ckpt:``) is never taken.
The session traces are replayed through the engine's own replay path.
"""
from __future__ import annotations

import json
import math
import random
from collections import Counter
from pathlib import Path

import pytest

import seven523.twin as twin_module
from seven523.play import QuitGame, replay_trace
from seven523.policies import RandomBot
from seven523.rules import DEFAULT_RULES
from seven523.trace import load_trace
from seven523.twin import (
    ARM_RAW,
    ARM_SEARCH,
    TWIN_REPORT_SCHEMA,
    TWIN_SESSION_SCHEMA,
    TwinGameRecord,
    TwinSession,
    paired_twin_stats,
    plan_twin_schedule,
)

SILENT = lambda *args, **kwargs: None  # noqa: E731
BASE_SPEC = "ckpt:runs/ei2_value_t5/t_leafq/critic.pt"
SEARCH_PARAMS = {
    "trunc_ply": 5,
    "rollout_k": 32,
    "max_candidates": 6,
    "max_rollout_ply": 400,
    "value_ckpt": "runs/ei2_value_t5/t_leafq/critic.pt",
}


def _record(
    pair: int,
    arm: str,
    *,
    seed: int = 1000 + 7,
    seat: int = 0,
    order: int = 0,
    human: int,
    opponent: int,
) -> TwinGameRecord:
    scores = [0, 0]
    scores[seat] = human
    scores[1 - seat] = opponent
    result = "win" if human > opponent else ("loss" if human < opponent else "draw")
    return TwinGameRecord(
        index=2 * pair + order,
        pair=pair,
        seed=seed,
        seat=seat,
        arm=arm,
        order=order,
        scores=(scores[0], scores[1]),
        human_score=human,
        opponent_score=opponent,
        margin=human - opponent,
        result=result,
        trace=f"g{2 * pair + order:04d}__s{seed}__seat{seat}__vs{arm}.json",
    )


def _patch_raw(monkeypatch, calls) -> None:
    """Replace the raw arm's ``policy_from_spec`` with a torch-free bot."""

    def fake(spec, rules=DEFAULT_RULES, seed=None, device="cpu"):
        calls.append((spec, seed))
        return RandomBot(random.Random(seed))

    monkeypatch.setattr(twin_module, "policy_from_spec", fake)


def _session(
    tmp_path: Path,
    *,
    pairs: int = 2,
    search_factory=None,
    search_calls=None,
    bootstrap: int = 200,
) -> TwinSession:
    def factory(spec, rules, seed, search):
        if search_calls is not None:
            search_calls.append((spec, dict(search), seed))
        if search_factory is not None:
            return search_factory(spec, rules, seed, search)
        return RandomBot(random.Random(seed))

    return TwinSession(
        base_spec=BASE_SPEC,
        raw_id="critic_raw",
        search_id="o4lite_t5k32",
        search_params=SEARCH_PARAMS,
        directory=tmp_path / "twins" / "twin-1",
        session_id="twin-1",
        search_factory=factory,
        pairs=pairs,
        rng=random.Random(11),
        rng_seed=11,
        bootstrap=bootstrap,
        bootstrap_seed=5,
        created_at="2026-09-28T00:00:00",
        raw_label="raw（critic.pt 原始 argmax）",
        search_label="搜索 t5·K32·C6",
    )


# -- schedule ----------------------------------------------------------------


def test_plan_twin_schedule_pairs_and_balances():
    schedule = plan_twin_schedule(random.Random(11), 4)
    assert len(schedule) == 8
    assert [game.index for game in schedule] == list(range(8))
    for pair in range(4):
        first, second = schedule[2 * pair], schedule[2 * pair + 1]
        assert first.index == 2 * pair and second.index == 2 * pair + 1
        assert first.pair == second.pair == pair
        assert first.seed == second.seed
        assert first.seat == second.seat == pair % 2
        assert {first.arm, second.arm} == {ARM_RAW, ARM_SEARCH}
        assert (first.order, second.order) == (0, 1)
        expected_first = ARM_RAW if pair % 2 == 0 else ARM_SEARCH
        assert first.arm == expected_first
    assert len({game.seed for game in schedule[::2]}) == 4
    counts = Counter((game.arm, game.seat) for game in schedule)
    for key in ((ARM_RAW, 0), (ARM_RAW, 1), (ARM_SEARCH, 0), (ARM_SEARCH, 1)):
        assert counts[key] == 2
    assert plan_twin_schedule(random.Random(11), 4) == schedule
    other = plan_twin_schedule(
        random.Random(11), 4, seat_start=1, order_start=ARM_SEARCH
    )
    assert other[0].seat == 1 and other[0].arm == ARM_SEARCH
    assert other[2].seat == 0 and other[2].arm == ARM_RAW
    with pytest.raises(ValueError):
        plan_twin_schedule(random.Random(0), 3)
    with pytest.raises(ValueError):
        plan_twin_schedule(random.Random(0), 1)
    with pytest.raises(ValueError):
        plan_twin_schedule(random.Random(0), 2, order_start="coin")
    with pytest.raises(ValueError):
        plan_twin_schedule(random.Random(0), 2, seat_start=2)


# -- paired estimator ---------------------------------------------------------


def test_paired_twin_stats_known_values():
    records = [
        # pair 0: raw win +10, search win +20 -> d_margin 10, d_win 0
        _record(0, ARM_RAW, human=60, opponent=50),
        _record(0, ARM_SEARCH, seat=0, order=1, human=70, opponent=50),
        # pair 1: raw loss -5, search win +15 -> d_margin 20, d_win +1
        _record(1, ARM_RAW, seat=1, order=1, human=40, opponent=45),
        _record(1, ARM_SEARCH, seat=1, order=0, human=55, opponent=40),
        # pair 2: raw draw 0, search win +10 -> d_margin 10, d_win +0.5
        _record(2, ARM_RAW, human=50, opponent=50),
        _record(2, ARM_SEARCH, order=1, human=55, opponent=45),
        # pair 3: raw win +5, search win +5 -> d_margin 0, d_win 0
        _record(3, ARM_RAW, human=50, opponent=45),
        _record(3, ARM_SEARCH, order=1, human=50, opponent=45),
    ]
    stats = paired_twin_stats(
        records, bootstrap=500, rng=random.Random(3), min_pairs=30
    )
    assert stats["counts"] == {
        "games_played": 8,
        "pairs_complete": 4,
        "pairs_incomplete": 0,
    }
    margin = stats["delta"]["margin_points"]
    assert margin["value"] == pytest.approx(10.0)
    assert margin["sd"] == pytest.approx(math.sqrt(200.0 / 3.0))
    winrate = stats["delta"]["winrate"]
    assert winrate["value"] == pytest.approx(0.375)
    assert winrate["raw_winrate"] == pytest.approx(0.625)
    assert winrate["search_winrate"] == pytest.approx(1.0)
    expected_elo = 400.0 * math.log10(0.75 / 0.25) - 400.0 * math.log10(
        0.625 / 0.375
    )
    assert stats["delta"]["elo"]["value"] == pytest.approx(expected_elo)
    sign = stats["delta"]["sign"]
    assert (sign["improved"], sign["same"], sign["worsened"]) == (3, 1, 0)
    assert sign["p_value"] == pytest.approx(0.25)
    swing = stats["delta"]["outcome_swing"]
    assert (swing["better"], swing["same"], swing["worse"]) == (2, 2, 0)
    per_pair = stats["delta"]["per_pair"]
    assert [entry["pair"] for entry in per_pair] == [0, 1, 2, 3]
    assert per_pair[1]["d_margin"] == 20
    assert per_pair[2]["d_win"] == 0.5
    assert per_pair[3]["order"] == [ARM_RAW, ARM_SEARCH]
    assert stats["resolution"]["ok"] is False
    assert stats["resolution"]["mde80_points"] == pytest.approx(
        (1.96 + 0.8416) * margin["sd"] / math.sqrt(4)
    )
    # Arm stats see every committed game, including incomplete pairs.
    assert stats["arm_stats"][ARM_RAW]["games"] == 4
    assert stats["arm_stats"][ARM_SEARCH]["mean_margin"] == pytest.approx(
        (20 + 15 + 10 + 5) / 4
    )


def test_paired_twin_stats_all_win_clamps_elo_finite():
    records = []
    for pair in range(3):
        records.append(_record(pair, ARM_RAW, human=40, opponent=60))
        records.append(
            _record(pair, ARM_SEARCH, order=1, human=60, opponent=40)
        )
    stats = paired_twin_stats(records, bootstrap=50, rng=random.Random(0))
    # floor = 1 / games-per-arm = 1/3 -> p in [1/3, 2/3]; all-win vs all-loss
    # saturates at 400*log10(4).
    assert stats["delta"]["elo"]["value"] == pytest.approx(400.0 * math.log10(4.0))
    assert math.isfinite(stats["delta"]["elo"]["value"])
    assert stats["delta"]["sign"]["p_value"] == pytest.approx(0.25)


def test_paired_twin_stats_bootstrap_deterministic_and_order_free():
    records = []
    raw_margins = [10, -5, 20, 0, 7, -3]
    search_margins = [12, -2, 25, 3, 6, 1]
    for pair, (raw, search) in enumerate(
        zip(raw_margins, search_margins, strict=True)
    ):
        records.append(_record(pair, ARM_RAW, human=50 + raw // 2, opponent=50 - raw // 2))
        records.append(
            _record(
                pair,
                ARM_SEARCH,
                order=1,
                human=50 + search // 2,
                opponent=50 - search // 2,
            )
        )

    def run(rng, data):
        return paired_twin_stats(data, bootstrap=300, rng=rng)

    first = run(random.Random(1), records)
    second = run(random.Random(1), list(reversed(records)))
    assert first == second
    third = run(random.Random(2), records)
    assert third["delta"]["margin_points"]["value"] == first["delta"]["margin_points"]["value"]
    assert (
        third["delta"]["margin_points"]["ci95"]
        != first["delta"]["margin_points"]["ci95"]
    )
    assert first["delta"]["margin_points"]["ci95"][0] < first["delta"][
        "margin_points"
    ]["ci95"][1]


def test_paired_twin_stats_drops_incomplete_pairs():
    records = [
        _record(0, ARM_RAW, human=60, opponent=50),
        _record(0, ARM_SEARCH, order=1, human=65, opponent=50),
        _record(1, ARM_RAW, human=40, opponent=50),
    ]
    stats = paired_twin_stats(records, bootstrap=100, rng=random.Random(0))
    assert stats["counts"]["pairs_complete"] == 1
    assert stats["counts"]["pairs_incomplete"] == 1
    assert stats["delta"]["margin_points"]["value"] == pytest.approx(5.0)
    assert stats["delta"]["winrate"]["value"] == pytest.approx(0.0)
    assert stats["resolution"]["ok"] is False
    assert stats["resolution"]["mde80_points"] is None
    assert stats["arm_stats"][ARM_RAW]["games"] == 2


def test_paired_twin_stats_empty_records_yield_nulls():
    stats = paired_twin_stats([], bootstrap=10, rng=random.Random(0))
    assert stats["counts"]["pairs_complete"] == 0
    assert stats["delta"]["margin_points"]["value"] is None
    assert stats["delta"]["winrate"]["value"] is None
    assert stats["delta"]["elo"]["value"] is None
    assert stats["delta"]["sign"]["p_value"] == 1.0
    assert stats["resolution"]["ok"] is False


# -- session construction -----------------------------------------------------


def test_twin_session_construction_validates(tmp_path):
    base = {
        "base_spec": BASE_SPEC,
        "raw_id": "raw",
        "search_id": "search",
        "search_params": SEARCH_PARAMS,
        "directory": tmp_path / "t",
        "session_id": "t",
        "search_factory": lambda spec, rules, seed, search: RandomBot(random.Random(seed)),
        "pairs": 2,
        "rng": random.Random(0),
    }
    with pytest.raises(ValueError, match="search_factory"):
        TwinSession(**{**base, "search_factory": None})
    with pytest.raises(ValueError, match="ckpt"):
        TwinSession(**{**base, "base_spec": "random"})
    with pytest.raises(ValueError, match="trunc_ply"):
        TwinSession(**{**base, "search_params": {"rollout_k": 8}})
    with pytest.raises(ValueError, match="even"):
        TwinSession(**{**base, "pairs": 3})
    with pytest.raises(ValueError, match="seat_start"):
        TwinSession(**{**base, "seat_start": 3})


# -- session simulate ---------------------------------------------------------


def test_twin_session_simulate_writes_session_and_report(tmp_path, monkeypatch):
    raw_calls: list[tuple[str, int]] = []
    search_calls: list[tuple[str, dict, int]] = []
    _patch_raw(monkeypatch, raw_calls)
    session = _session(tmp_path, pairs=2, search_calls=search_calls)
    report = session.run(
        human_policy_factory=lambda seat: RandomBot(random.Random(100 + seat)),
        print_fn=SILENT,
    )

    assert report["schema"] == TWIN_REPORT_SCHEMA
    assert report["session"]["bank"] == 1
    assert len(session.records) == 4
    assert report["counts"] == {
        "games_played": 4,
        "pairs_complete": 2,
        "pairs_incomplete": 0,
        "pairs_total": 2,
        "abandoned_games": 0,
    }
    assert report["config"]["rng_seed"] == 11
    assert report["delta"]["margin_points"]["value"] is not None
    # The factory is called for the search arm only; the raw arm goes through
    # policy_from_spec (both arms: 2 games each).
    assert len(raw_calls) == 2
    assert len(search_calls) == 2
    assert all(spec == BASE_SPEC for spec, _seed in raw_calls)
    assert all(search == SEARCH_PARAMS for _spec, search, _seed in search_calls)

    directory = tmp_path / "twins" / "twin-1"
    assert (directory / "session.json").is_file()
    assert (directory / "report.json").is_file()
    document = json.loads((directory / "session.json").read_text())
    assert document["schema"] == TWIN_SESSION_SCHEMA
    assert len(document["games"]) == 4
    assert len(document["plan"]) == 4
    assert document["incomplete_pairs"] == []

    initials: dict[int, list[dict]] = {}
    for record in document["games"]:
        trace = load_trace(directory / record["trace"])
        annotation = trace["deal_twin"]
        assert annotation["schema"] == "seven523.deal-twin"
        assert annotation["pair"] == record["pair"]
        assert annotation["arm"] == record["arm"]
        assert annotation["seat"] == record["seat"]
        assert annotation["session"] == "twin-1"
        if record["arm"] == ARM_SEARCH:
            assert annotation["search"] == SEARCH_PARAMS
            assert trace["opponent_search"] == SEARCH_PARAMS
        else:
            assert annotation["search"] is None
            assert "opponent_search" not in trace
        assert replay_trace(trace, print_fn=SILENT) is True
        initials.setdefault(record["pair"], []).append(trace["initial"])
    for snapshots in initials.values():
        assert len(snapshots) == 2
        assert snapshots[0] == snapshots[1]

    # The report's delta is exactly the estimator on the persisted records.
    recomputed = paired_twin_stats(
        session.records, bootstrap=200, rng=random.Random(5), min_pairs=30
    )
    assert report["delta"] == recomputed["delta"]
    assert report["resolution"] == recomputed["resolution"]


def test_twin_trace_labels_and_filenames(tmp_path, monkeypatch):
    _patch_raw(monkeypatch, [])
    session = _session(tmp_path, pairs=2)
    session.run(
        human_policy_factory=lambda seat: RandomBot(random.Random(seat)),
        print_fn=SILENT,
    )
    directory = tmp_path / "twins" / "twin-1"
    for record in session.records:
        trace = load_trace(directory / record.trace)
        other = 1 - record.seat
        role = "twin_raw" if record.arm == ARM_RAW else "twin_search"
        arm_id = "critic_raw" if record.arm == ARM_RAW else "o4lite_t5k32"
        assert trace["players"][record.seat] == f"human@seat{record.seat}"
        assert trace["players"][other] == f"{role}:{arm_id}@seat{other}"
        assert record.trace == (
            f"g{record.index:04d}__s{record.seed}__seat{record.seat}"
            f"__vs{record.arm}.json"
        )


def test_twin_session_run_with_chooser_quit_marks_pair_incomplete(
    tmp_path, monkeypatch
):
    _patch_raw(monkeypatch, [])

    def chooser(*_args, **_kwargs):
        raise QuitGame()

    session = _session(tmp_path, pairs=2)
    report = session.run(chooser_factory=lambda seat: chooser, print_fn=SILENT)
    assert report["counts"]["games_played"] == 0
    assert report["counts"]["abandoned_games"] == 1
    assert report["counts"]["pairs_incomplete"] == 1
    assert report["delta"]["margin_points"]["value"] is None
    assert session.pairs_incomplete() == [0]
    document = json.loads((session.directory / "session.json").read_text())
    assert document["stopped_reason"] == "quit"
    assert document["incomplete_pairs"] == [0]


def test_twin_session_records_share_seed_and_seat_inside_a_pair(tmp_path, monkeypatch):
    _patch_raw(monkeypatch, [])
    session = _session(tmp_path, pairs=4)
    session.run(
        human_policy_factory=lambda seat: RandomBot(random.Random(seat)),
        print_fn=SILENT,
    )
    by_pair: dict[int, list] = {}
    for record in session.records:
        by_pair.setdefault(record.pair, []).append(record)
    assert set(by_pair) == {0, 1, 2, 3}
    for group in by_pair.values():
        assert len(group) == 2
        assert group[0].seed == group[1].seed
        assert group[0].seat == group[1].seat
        assert {group[0].arm, group[1].arm} == {ARM_RAW, ARM_SEARCH}

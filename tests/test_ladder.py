"""M2 ladder: schedule invariants, trace round-trip, and the CLI -> D1 path."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from seven523.elo import Prior
from seven523.ladder import Entrant, build_ladder, plan_games, play_games
from seven523.play import replay_trace
from seven523.study import load_manifest
from seven523.trace import load_trace

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
SILENT = lambda *args, **kwargs: None  # noqa: E731


def _load_tool(name: str):
    spec = importlib.util.spec_from_file_location(name, TOOLS_DIR / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def anchors() -> list[Entrant]:
    return [
        Entrant("random", "random", pinned=1000.0),
        Entrant("greedy", "greedy", pinned=1315.0),
    ]


def test_plan_games_meets_each_anchor_and_pairs_deals():
    entrants = [*anchors(), Entrant("c1", "greedy"), Entrant("c2", "greedy")]
    schedule = plan_games(entrants, games_per_anchor=4, seed=7)
    assert len(schedule) == 4 * 2 * 2

    pairs = Counter(
        (game.subject, other)
        for game in schedule
        for other in game.seats
        if other != game.subject
    )
    for candidate in ("c1", "c2"):
        for anchor in ("random", "greedy"):
            assert pairs[(candidate, anchor)] == 4

    seeds: dict[tuple[str, str], list[int]] = defaultdict(list)
    for game in schedule:
        opponent = next(seat for seat in game.seats if seat != game.subject)
        seeds[(game.subject, opponent)].append(game.seed)
    assert sorted(seeds[("c1", "random")]) == sorted(seeds[("c2", "random")])
    assert sorted(seeds[("c1", "greedy")]) == sorted(seeds[("c2", "greedy")])

    seat0 = sum(1 for game in schedule if game.subject == "c1" and game.seats[0] == "c1")
    seat1 = sum(1 for game in schedule if game.subject == "c1" and game.seats[1] == "c1")
    assert seat0 == seat1 == 4

    twin_seats: dict[tuple[str, str, int], list[int]] = defaultdict(list)
    for game in schedule:
        opponent = next(seat for seat in game.seats if seat != game.subject)
        twin_seats[(game.subject, opponent, game.seed)].append(
            game.seats.index(game.subject)
        )
    for seats in twin_seats.values():
        assert sorted(seats) == [0, 1]


def test_plan_games_is_order_invariant_and_seat_balanced():
    forward = [
        *anchors(),
        Entrant("c1", "greedy"),
        Entrant("c2", "random"),
        Entrant("c3", "random"),
    ]
    backward = [*anchors(), forward[4], forward[3], forward[2]]
    left = plan_games(forward, games_per_anchor=4, seed=3)
    right = plan_games(backward, games_per_anchor=4, seed=3)
    key = lambda game: (game.seed, game.seats, game.subject)  # noqa: E731
    assert sorted(left, key=key) == sorted(right, key=key)
    for schedule in (left, right):
        for subject in ("c1", "c2", "c3"):
            twins: dict[tuple[str, int], list[int]] = defaultdict(list)
            for game in schedule:
                if game.subject != subject:
                    continue
                opponent = next(seat for seat in game.seats if seat != subject)
                twins[(opponent, game.seed)].append(game.seats.index(subject))
            assert twins
            for seats in twins.values():
                assert sorted(seats) == [0, 1]


def test_plan_games_cross_games_are_optional():
    entrants = [*anchors(), Entrant("c1", "greedy"), Entrant("c2", "greedy")]
    schedule = plan_games(entrants, games_per_anchor=2, cross=4, seed=0)
    cross_games = [
        game
        for game in schedule
        if game.subject == "c1"
        and not ({"random", "greedy"} & set(game.seats))
    ]
    assert len(cross_games) == 4
    twins = defaultdict(list)
    for game in cross_games:
        twins[game.seed].append(game.seats.index("c1"))
    for seats in twins.values():
        assert sorted(seats) == [0, 1]


def test_plan_games_validation():
    with pytest.raises(ValueError):
        plan_games([Entrant("c1", "greedy")])
    with pytest.raises(ValueError):
        plan_games([*anchors()])
    with pytest.raises(ValueError):
        plan_games([*anchors(), Entrant("c1", "greedy"), Entrant("c1", "random")])
    with pytest.raises(ValueError):
        plan_games([*anchors(), Entrant("c1", "greedy")], games_per_anchor=0)
    with pytest.raises(ValueError):
        plan_games([*anchors(), Entrant("c1", "greedy")], games_per_anchor=3)
    with pytest.raises(ValueError):
        plan_games([*anchors(), Entrant("c1", "greedy")], cross=-1)
    with pytest.raises(ValueError):
        plan_games([*anchors(), Entrant("c1", "greedy")], cross=1)


def test_entrant_validation():
    with pytest.raises(ValueError):
        Entrant("x", "greedy", pinned=1000.0, prior=Prior())
    with pytest.raises(ValueError):
        Entrant("a@b", "greedy")
    with pytest.raises(ValueError):
        Entrant("x", "")


def test_play_games_writes_replayable_traces(tmp_path):
    entrants = [*anchors(), Entrant("g1", "greedy")]
    schedule = plan_games(entrants, games_per_anchor=2, seed=5)
    results = play_games(
        schedule, entrants, out=tmp_path, created_at="2025-01-01T00:00:00"
    )
    assert len(results) == 4
    for result in results:
        assert sum(result.scores) == 100
        assert len(result.seats) == 2

    traces = sorted((tmp_path / "g1").glob("*.json"))
    assert len(traces) == 4
    for path in traces:
        trace = load_trace(path)
        assert trace["created_at"] == "2025-01-01T00:00:00"
        assert trace["players"][trace["human_seat"]].startswith("subject:g1@")
        assert replay_trace(trace, print_fn=SILENT)


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_play_games_writes_per_game_jsonl(tmp_path):
    entrants = [*anchors(), Entrant("g1", "greedy")]
    schedule = plan_games(entrants, games_per_anchor=2, seed=5)
    results_path = tmp_path / "nested" / "results.jsonl"
    results = play_games(
        schedule,
        entrants,
        results_out=results_path,
        created_at="2025-01-01T00:00:00",
    )
    lines = _jsonl(results_path)
    assert len(lines) == len(results) == len(schedule) == 4
    for game, result, line in zip(schedule, results, lines):
        seat = game.seats.index(game.subject)
        assert line == {
            "seed": game.seed,
            "seats": list(game.seats),
            "scores": list(result.scores),
            "subject": game.subject,
            "opponent": game.seats[1 - seat],
            "subject_seat": seat,
            "kind": "anchor",  # g1 only ever meets the two pinned anchors
        }
        assert sum(line["scores"]) == 100


def test_play_games_jsonl_marks_cross_games_and_appends(tmp_path):
    entrants = [*anchors(), Entrant("c1", "greedy"), Entrant("c2", "greedy")]
    schedule = plan_games(entrants, games_per_anchor=2, cross=2, seed=0)
    path = tmp_path / "games.jsonl"
    play_games(schedule, entrants, results_out=path)
    first = _jsonl(path)
    assert len(first) == len(schedule)
    assert {line["kind"] for line in first} == {"anchor", "cross"}
    cross = [line for line in first if line["kind"] == "cross"]
    assert cross and all(
        "random" not in line["seats"] and "greedy" not in line["seats"]
        for line in cross
    )
    # A second run appends, never truncates (resume-friendly).
    play_games(schedule, entrants, results_out=path)
    assert len(_jsonl(path)) == 2 * len(schedule)


def test_play_games_jsonl_is_deterministic_per_seed(tmp_path):
    entrants = [*anchors(), Entrant("g1", "greedy")]
    schedule = plan_games(entrants, games_per_anchor=2, seed=5)
    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"
    play_games(schedule, entrants, results_out=first)
    play_games(schedule, entrants, results_out=second)
    assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8")


def _parallel_schedule():
    entrants = [*anchors(), Entrant("g1", "greedy"), Entrant("r1", "random")]
    return entrants, plan_games(entrants, games_per_anchor=6, seed=11)


def test_play_games_parallel_matches_serial_bit_for_bit(tmp_path):
    entrants, schedule = _parallel_schedule()
    serial_dir = tmp_path / "serial"
    parallel_dir = tmp_path / "parallel"
    serial = play_games(
        schedule,
        entrants,
        out=serial_dir,
        results_out=tmp_path / "serial.jsonl",
        created_at="2025-01-01T00:00:00",
        workers=1,
    )
    parallel = play_games(
        schedule,
        entrants,
        out=parallel_dir,
        results_out=tmp_path / "parallel.jsonl",
        created_at="2025-01-01T00:00:00",
        workers=4,
    )
    assert parallel == serial
    assert len(parallel) == len(schedule)

    # results_out: one file, schedule order, byte-for-byte the serial output.
    assert (
        tmp_path / "parallel.jsonl"
    ).read_bytes() == (tmp_path / "serial.jsonl").read_bytes()
    assert not list(tmp_path.glob(".*shard*"))  # shard temps are merged and removed

    # traces: global schedule index (a local index would collide across shards).
    serial_files = sorted(
        path.relative_to(serial_dir) for path in serial_dir.rglob("*.json")
    )
    parallel_files = sorted(
        path.relative_to(parallel_dir) for path in parallel_dir.rglob("*.json")
    )
    assert parallel_files == serial_files
    assert len(serial_files) == len(schedule)
    assert {path.name.split("__")[0][1:] for path in serial_files} == {
        f"{index:04d}" for index in range(len(schedule))
    }
    for rel in serial_files:
        assert (parallel_dir / rel).read_bytes() == (serial_dir / rel).read_bytes()


def test_play_games_parallel_results_out_appends(tmp_path):
    entrants, schedule = _parallel_schedule()
    path = tmp_path / "games.jsonl"
    path.write_bytes(b'{"prior":1}\n')
    play_games(schedule, entrants, results_out=path, workers=2)
    lines = _jsonl(path)
    assert len(lines) == 1 + len(schedule)
    assert lines[0] == {"prior": 1}
    assert [line["seed"] for line in lines[1:]] == [game.seed for game in schedule]
    assert not list(tmp_path.glob(".*shard*"))


def test_play_games_parallel_empty_schedule(tmp_path):
    entrants = [*anchors(), Entrant("g1", "greedy")]
    assert play_games((), entrants, workers=4) == []
    path = tmp_path / "empty.jsonl"
    assert play_games((), entrants, results_out=path, workers=4) == []
    assert path.exists()
    assert path.read_bytes() == b""


def test_play_games_workers_validation():
    entrants, schedule = _parallel_schedule()
    with pytest.raises(ValueError):
        play_games(schedule, entrants, workers=0)


def test_play_games_parallel_warns_on_cuda(capsys):
    entrants, schedule = _parallel_schedule()
    # random/greedy never touch torch, so this exercises the warning alone.
    play_games(schedule, entrants, workers=2, device="cuda")
    err = capsys.readouterr().err
    assert "CUDA context" in err
    assert "device='cuda'" in err


def test_play_games_parallel_rejects_unpicklable_factory():
    entrants, schedule = _parallel_schedule()
    with pytest.raises(ValueError, match="picklable"):
        play_games(
            schedule,
            entrants,
            workers=2,
            factory=lambda spec, rules, seed: None,
        )


def test_build_ladder_workers_match_serial():
    entrants = [*anchors(), Entrant("g1", "greedy"), Entrant("r1", "random")]
    serial = build_ladder(
        entrants, games_per_anchor=4, seed=2, count=2, max_spacing=300.0, workers=1
    )
    parallel = build_ladder(
        entrants, games_per_anchor=4, seed=2, count=2, max_spacing=300.0, workers=2
    )
    assert parallel.schedule == serial.schedule
    assert parallel.fit.ratings == serial.fit.ratings
    assert parallel.selection == serial.selection


def test_cli_build_ladder_workers_parallel(tmp_path):
    builder = _load_tool("build_ladder")
    games = tmp_path / "games.jsonl"
    code = builder.main(
        [
            "--anchor", "random=random",
            "--anchor", "greedy=greedy",
            "--candidate", "g1=greedy",
            "--games-per-anchor", "4",
            "--seed", "1",
            "--no-traces",
            "--games-out", str(games),
            "--workers", "2",
        ]
    )
    assert code == 0
    lines = _jsonl(games)
    assert len(lines) == 8
    assert all(line["kind"] == "anchor" for line in lines)
    with pytest.raises(SystemExit):
        builder.main(["--candidate", "g1=greedy", "--workers", "0"])


def test_cli_head_to_head_workers_match_single(capsys):
    tool = _load_tool("head_to_head")

    def run(workers: int) -> dict:
        code = tool.main(
            [
                "--left", "greedy",
                "--right", "random",
                "--pairs", "2",
                "--seed", "5",
                "--bootstrap", "100",
                "--workers", str(workers),
                "--json",
            ]
        )
        assert code == 0
        return json.loads(capsys.readouterr().out)

    assert run(2) == run(1)
    with pytest.raises(SystemExit):
        tool.main(["--left", "greedy", "--right", "random", "--workers", "0"])


def test_build_ladder_forwards_games_out(tmp_path):
    entrants = [*anchors(), Entrant("g1", "greedy"), Entrant("r1", "random")]
    path = tmp_path / "ladder_games.jsonl"
    ladder = build_ladder(
        entrants,
        games_per_anchor=4,
        seed=2,
        count=2,
        max_spacing=300.0,
        games_out=path,
    )
    lines = _jsonl(path)
    assert len(lines) == len(ladder.schedule) == 16  # 2 anchors x 2 candidates x 4
    for game, line in zip(ladder.schedule, lines):
        assert line["seed"] == game.seed
        assert line["seats"] == list(game.seats)
        assert line["subject"] == game.subject


def test_cli_games_out_works_with_no_traces(tmp_path):
    builder = _load_tool("build_ladder")
    study = tmp_path / "study"
    games = tmp_path / "games.jsonl"
    code = builder.main(
        [
            "--anchor", "random=random",
            "--anchor", "greedy=greedy",
            "--candidate", "g1=greedy",
            "--games-per-anchor", "4",
            "--seed", "1",
            "--study", str(study),
            "--rungs", "2",
            "--spacing", "50:200",
            "--no-traces",
            "--games-out", str(games),
        ]
    )
    assert code == 0
    lines = _jsonl(games)
    assert len(lines) == 8  # 4 games per anchor x 2 anchors
    assert {line["kind"] for line in lines} == {"anchor"}
    assert not study.exists()  # --no-traces still skips the manifest


def test_build_ladder_fits_anchors_and_candidates():
    entrants = [*anchors(), Entrant("g1", "greedy"), Entrant("r1", "random")]
    ladder = build_ladder(entrants, games_per_anchor=4, seed=2, count=2, max_spacing=300.0)
    assert set(ladder.fit.ratings) == {"random", "greedy", "g1", "r1"}
    assert ladder.fit.converged
    assert ladder.fit.ratings["random"].se == 0.0
    assert all(math.isfinite(rating.elo) for rating in ladder.fit.ratings.values())
    assert len(ladder.selection.rungs) == 2
    assert ladder.selection.rungs[0].id == "r1"


def test_build_ladder_requires_two_anchors():
    entrants = [Entrant("random", "random", pinned=1000.0), Entrant("g1", "greedy")]
    with pytest.raises(ValueError):
        build_ladder(entrants, games_per_anchor=1)


def test_cli_manifest_feeds_the_d1_tool(tmp_path):
    builder = _load_tool("build_ladder")
    measure = _load_tool("measure_trace_signal")
    study = tmp_path / "study"
    code = builder.main(
        [
            "--anchor", "random=random",
            "--anchor", "greedy=greedy",
            "--candidate", "g1=greedy",
            "--games-per-anchor", "4",
            "--seed", "1",
            "--study", str(study),
            "--rungs", "2",
            "--spacing", "50:200",
        ]
    )
    assert code == 0
    manifest = load_manifest(study / "manifest.json")
    assert {"random", "greedy", "g1"} <= set(manifest["levels"])
    assert manifest["estimator"]["kind"] == "bt-map"

    csv_path = tmp_path / "features.csv"
    assert measure.cmd_features(
        argparse.Namespace(study=str(study), out=str(csv_path), level=[], verify=True)
    ) == 0
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 8  # 4 games per anchor x 2 anchors (seat-paired twins)
    assert {row["level_id"] for row in rows} == {"g1"}
    for row in rows:
        assert float(row["level_elo_ref"]) == pytest.approx(manifest["levels"]["g1"])

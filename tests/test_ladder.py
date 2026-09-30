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
from support import FirstLegalBot

from seven523.elo import DEFAULT_SIGMA, Prior
from seven523.ladder import (
    Entrant,
    build_ladder,
    manifest_priors,
    plan_games,
    play_games,
)
from seven523.play import replay_trace
from seven523.policies import policy_from_spec
from seven523.rules import DEFAULT_RULES, Rules, rules_id, rules_identity
from seven523.study import load_manifest, save_manifest
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
    # The pinned gauge: the RandomBot defines 0 (ADR-0012).
    return [Entrant("random", "random", pinned=0.0)]


def _factory(spec, rules, seed):
    """Test spec ``fixed`` -> FirstLegalBot; production specs pass through."""
    if spec == "fixed":
        return FirstLegalBot(rules)
    return policy_from_spec(spec, rules, seed)


def test_plan_games_meets_each_anchor_and_pairs_deals():
    entrants = [*anchors(), Entrant("c1", "fixed"), Entrant("c2", "fixed")]
    schedule = plan_games(entrants, games_per_anchor=4, seed=7)
    assert len(schedule) == 4 * 1 * 2

    pairs = Counter(
        (game.subject, other)
        for game in schedule
        for other in game.seats
        if other != game.subject
    )
    for candidate in ("c1", "c2"):
        for anchor in ("random",):
            assert pairs[(candidate, anchor)] == 4

    seeds: dict[tuple[str, str], list[int]] = defaultdict(list)
    for game in schedule:
        opponent = next(seat for seat in game.seats if seat != game.subject)
        seeds[(game.subject, opponent)].append(game.seed)
    assert sorted(seeds[("c1", "random")]) == sorted(seeds[("c2", "random")])

    seat0 = sum(1 for game in schedule if game.subject == "c1" and game.seats[0] == "c1")
    seat1 = sum(1 for game in schedule if game.subject == "c1" and game.seats[1] == "c1")
    assert seat0 == seat1 == 2

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
        Entrant("c1", "fixed"),
        Entrant("c2", "random"),
        Entrant("c3", "random"),
    ]
    backward = [*anchors(), forward[3], forward[2], forward[1]]
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
    entrants = [*anchors(), Entrant("c1", "fixed"), Entrant("c2", "fixed")]
    schedule = plan_games(entrants, games_per_anchor=2, cross=4, seed=0)
    cross_games = [
        game
        for game in schedule
        if game.subject == "c1"
        and not ({"random"} & set(game.seats))
    ]
    assert len(cross_games) == 4
    twins = defaultdict(list)
    for game in cross_games:
        twins[game.seed].append(game.seats.index("c1"))
    for seats in twins.values():
        assert sorted(seats) == [0, 1]


def test_plan_games_validation():
    with pytest.raises(ValueError):
        plan_games([Entrant("c1", "fixed")])
    with pytest.raises(ValueError):
        plan_games([*anchors()])
    with pytest.raises(ValueError):
        plan_games([*anchors(), Entrant("c1", "fixed"), Entrant("c1", "random")])
    with pytest.raises(ValueError):
        plan_games([*anchors(), Entrant("c1", "fixed")], games_per_anchor=0)
    with pytest.raises(ValueError):
        plan_games([*anchors(), Entrant("c1", "fixed")], games_per_anchor=3)
    with pytest.raises(ValueError):
        plan_games([*anchors(), Entrant("c1", "fixed")], cross=-1)
    with pytest.raises(ValueError):
        plan_games([*anchors(), Entrant("c1", "fixed")], cross=1)


def test_entrant_validation():
    with pytest.raises(ValueError):
        Entrant("x", "fixed", pinned=0.0, prior=Prior())
    with pytest.raises(ValueError):
        Entrant("a@b", "fixed")
    with pytest.raises(ValueError):
        Entrant("x", "")


def test_manifest_priors_subjects_win_over_levels():
    document = {
        "levels": {"c1": 300.0, "c2": 350.0},
        "subjects": [
            {"id": "c1", "mu": 320.0, "sigma": 40.0},
            {"id": "c3", "mu": 400.0, "sigma": 30.0},
        ],
    }
    priors = manifest_priors(document)
    assert priors == {
        "c1": Prior(320.0, 40.0),
        "c2": Prior(350.0, DEFAULT_SIGMA),
        "c3": Prior(400.0, 30.0),
    }


def test_manifest_priors_levels_use_default_sigma():
    assert manifest_priors({"levels": {"c1": 123.0}}, default_sigma=77.0) == {
        "c1": Prior(123.0, 77.0)
    }
    assert manifest_priors({}) == {}


def test_manifest_priors_skips_invalid_entries_and_repairs_bad_sigma():
    document = {
        "levels": {
            "ok": 100.0,
            "nan": float("nan"),
            "inf": float("inf"),
            "text": "100",
            "bool": True,
            "none": None,
        },
        "subjects": [
            {"id": "good", "mu": 200.0, "sigma": 25.0},
            {"id": "no-sigma", "mu": 210.0},
            {"id": "bad-sigma", "mu": 220.0, "sigma": float("nan")},
            {"id": "zero-sigma", "mu": 230.0, "sigma": 0.0},
            {"id": "negative-sigma", "mu": 240.0, "sigma": -5.0},
            {"id": "bad-mu", "mu": "300"},
            {"id": "missing-mu"},
            {"id": "", "mu": 1.0},
            {"mu": 1.0},
            "not-a-subject",
        ],
    }
    priors = manifest_priors(document)
    assert priors == {
        "ok": Prior(100.0, DEFAULT_SIGMA),
        "good": Prior(200.0, 25.0),
        "no-sigma": Prior(210.0, DEFAULT_SIGMA),
        "bad-sigma": Prior(220.0, DEFAULT_SIGMA),
        "zero-sigma": Prior(230.0, DEFAULT_SIGMA),
        "negative-sigma": Prior(240.0, DEFAULT_SIGMA),
    }
    assert all(
        math.isfinite(prior.mu) and math.isfinite(prior.sigma)
        for prior in priors.values()
    )


def test_manifest_priors_does_not_mutate_the_document():
    document = {
        "levels": {"c1": 300.0, "c2": float("nan")},
        "subjects": [{"id": "c1", "mu": 320.0, "sigma": 40.0}],
    }
    snapshot = json.dumps(document, sort_keys=True)
    manifest_priors(document)
    assert json.dumps(document, sort_keys=True) == snapshot
    assert set(document) == {"levels", "subjects"}


def test_play_games_writes_replayable_traces(tmp_path):
    entrants = [*anchors(), Entrant("g1", "fixed")]
    schedule = plan_games(entrants, games_per_anchor=2, seed=5)
    results = play_games(
        schedule,
        entrants,
        factory=_factory,
        out=tmp_path,
        created_at="2025-01-01T00:00:00",
    )
    assert len(results) == 2
    for result in results:
        assert sum(result.scores) == 100
        assert len(result.seats) == 2

    traces = sorted((tmp_path / "g1").glob("*.json"))
    assert len(traces) == 2
    for path in traces:
        trace = load_trace(path)
        assert trace["created_at"] == "2025-01-01T00:00:00"
        assert trace["players"][trace["human_seat"]].startswith("subject:g1@")
        assert replay_trace(trace, print_fn=SILENT)


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_play_games_writes_per_game_jsonl(tmp_path):
    entrants = [*anchors(), Entrant("g1", "fixed")]
    schedule = plan_games(entrants, games_per_anchor=2, seed=5)
    results_path = tmp_path / "nested" / "results.jsonl"
    results = play_games(
        schedule,
        entrants,
        factory=_factory,
        results_out=results_path,
        created_at="2025-01-01T00:00:00",
    )
    lines = _jsonl(results_path)
    assert len(lines) == len(results) == len(schedule) == 2
    for game, result, line in zip(schedule, results, lines):
        seat = game.seats.index(game.subject)
        assert line == {
            "seed": game.seed,
            "seats": list(game.seats),
            "scores": list(result.scores),
            "subject": game.subject,
            "opponent": game.seats[1 - seat],
            "subject_seat": seat,
            "kind": "anchor",  # g1 only ever meets the pinned gauge
            "rules_id": rules_id(DEFAULT_RULES),
        }
        assert sum(line["scores"]) == 100


def test_play_games_jsonl_records_rules_id(tmp_path):
    rules = Rules(straight_max=6)
    entrants = [*anchors(), Entrant("g1", "fixed")]
    schedule = plan_games(entrants, games_per_anchor=2, seed=5)
    path = tmp_path / "games.jsonl"
    play_games(schedule, entrants, rules=rules, factory=_factory, results_out=path)
    lines = _jsonl(path)
    assert lines
    assert {line["rules_id"] for line in lines} == {rules_id(rules)}
    assert rules_id(rules) != rules_id(DEFAULT_RULES)


def test_play_games_jsonl_marks_cross_games_and_appends(tmp_path):
    entrants = [*anchors(), Entrant("c1", "fixed"), Entrant("c2", "fixed")]
    schedule = plan_games(entrants, games_per_anchor=2, cross=2, seed=0)
    path = tmp_path / "games.jsonl"
    play_games(schedule, entrants, factory=_factory, results_out=path)
    first = _jsonl(path)
    assert len(first) == len(schedule)
    assert {line["kind"] for line in first} == {"anchor", "cross"}
    cross = [line for line in first if line["kind"] == "cross"]
    assert cross and all(
        "random" not in line["seats"] for line in cross
    )
    # A second run appends, never truncates (resume-friendly).
    play_games(schedule, entrants, factory=_factory, results_out=path)
    assert len(_jsonl(path)) == 2 * len(schedule)


def test_play_games_jsonl_is_deterministic_per_seed(tmp_path):
    entrants = [*anchors(), Entrant("g1", "fixed")]
    schedule = plan_games(entrants, games_per_anchor=2, seed=5)
    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"
    play_games(schedule, entrants, factory=_factory, results_out=first)
    play_games(schedule, entrants, factory=_factory, results_out=second)
    assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8")


def _parallel_schedule():
    entrants = [*anchors(), Entrant("g1", "fixed"), Entrant("r1", "random")]
    return entrants, plan_games(entrants, games_per_anchor=2, seed=11)


def test_play_games_parallel_matches_serial_bit_for_bit(tmp_path):
    entrants, schedule = _parallel_schedule()
    serial_dir = tmp_path / "serial"
    parallel_dir = tmp_path / "parallel"
    serial = play_games(
        schedule,
        entrants,
        factory=_factory,
        out=serial_dir,
        results_out=tmp_path / "serial.jsonl",
        created_at="2025-01-01T00:00:00",
        workers=1,
    )
    parallel = play_games(
        schedule,
        entrants,
        factory=_factory,
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
    play_games(schedule, entrants, factory=_factory, results_out=path, workers=2)
    lines = _jsonl(path)
    assert len(lines) == 1 + len(schedule)
    assert lines[0] == {"prior": 1}
    assert [line["seed"] for line in lines[1:]] == [game.seed for game in schedule]
    assert not list(tmp_path.glob(".*shard*"))


def test_play_games_results_dir_serial_rerun_replaces_stale_shards(tmp_path):
    entrants = [*anchors(), Entrant("g1", "fixed")]
    schedule = plan_games(entrants, games_per_anchor=2, seed=5)
    out_dir = tmp_path / "shards"
    results = play_games(schedule, entrants, factory=_factory, results_dir=out_dir, workers=1)
    assert len(results) == len(schedule)
    assert [path.name for path in sorted(out_dir.glob("shard_*.jsonl"))] == [
        "shard_00000.jsonl"
    ]
    first = _jsonl(out_dir / "shard_00000.jsonl")
    assert len(first) == len(schedule)
    assert [line["seed"] for line in first] == [game.seed for game in schedule]

    stale = out_dir / "shard_00009.jsonl"
    stale.write_text('{"stale":1}\n', encoding="utf-8")
    play_games(schedule, entrants, factory=_factory, results_dir=out_dir, workers=1)
    assert not stale.exists()  # a wider previous run's shard must not linger
    assert [path.name for path in sorted(out_dir.glob("shard_*.jsonl"))] == [
        "shard_00000.jsonl"
    ]
    assert _jsonl(out_dir / "shard_00000.jsonl") == first  # replaced, not appended


def test_play_games_results_dir_parallel_matches_serial(tmp_path):
    entrants, schedule = _parallel_schedule()
    serial_dir = tmp_path / "serial"
    parallel_dir = tmp_path / "parallel"
    serial = play_games(schedule, entrants, factory=_factory, results_dir=serial_dir, workers=1)
    parallel = play_games(schedule, entrants, factory=_factory, results_dir=parallel_dir, workers=4)
    assert parallel == serial

    shard_files = sorted(parallel_dir.glob("shard_*.jsonl"))
    assert [path.name for path in shard_files] == [
        f"shard_{index:05d}.jsonl" for index in range(4)
    ]
    serial_text = (serial_dir / "shard_00000.jsonl").read_text(encoding="utf-8")
    parallel_text = "".join(
        path.read_text(encoding="utf-8") for path in shard_files
    )
    # Shards are contiguous, so concatenating them restores schedule order and
    # matches the single-process JSONL byte for byte.
    assert parallel_text == serial_text


def test_play_games_results_out_and_dir_are_exclusive(tmp_path):
    entrants, schedule = _parallel_schedule()
    with pytest.raises(ValueError, match="mutually exclusive"):
        play_games(
            schedule,
            entrants,
            results_out=tmp_path / "games.jsonl",
            results_dir=tmp_path / "shards",
        )
    assert not (tmp_path / "games.jsonl").exists()
    assert not (tmp_path / "shards").exists()


def test_play_games_parallel_empty_schedule(tmp_path):
    entrants = [*anchors(), Entrant("g1", "fixed")]
    assert play_games((), entrants, workers=4) == []
    path = tmp_path / "empty.jsonl"
    assert play_games((), entrants, results_out=path, workers=4) == []
    assert path.exists()
    assert path.read_bytes() == b""
    # results_dir is the fresh-layout mode: an empty schedule must not mkdir.
    assert play_games((), entrants, results_dir=tmp_path / "shards", workers=4) == []
    assert not (tmp_path / "shards").exists()


def test_play_games_workers_validation():
    entrants, schedule = _parallel_schedule()
    with pytest.raises(ValueError):
        play_games(schedule, entrants, workers=0)


def test_play_games_parallel_warns_on_cuda(capsys):
    entrants, schedule = _parallel_schedule()
    # random/fixed never touch torch, so this exercises the warning alone.
    play_games(schedule, entrants, factory=_factory, workers=2, device="cuda")
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
    entrants = [*anchors(), Entrant("g1", "fixed"), Entrant("r1", "random")]
    serial = build_ladder(
        entrants,
        games_per_anchor=2,
        seed=2,
        count=2,
        max_spacing=300.0,
        factory=_factory,
        workers=1,
    )
    parallel = build_ladder(
        entrants,
        games_per_anchor=2,
        seed=2,
        count=2,
        max_spacing=300.0,
        factory=_factory,
        workers=2,
    )
    assert parallel.schedule == serial.schedule
    assert parallel.fit.ratings == serial.fit.ratings
    assert parallel.selection == serial.selection


def test_cli_build_ladder_workers_parallel(tmp_path):
    builder = _load_tool("build_ladder")
    games = tmp_path / "games.jsonl"
    study = tmp_path / "study"  # never the repo's legacy traces/study
    code = builder.main(
        [
            "--anchor", "random=random",
            "--candidate", "c1=random",
            "--candidate", "c2=random",
            "--games-per-anchor", "2",
            "--cross", "2",
            "--seed", "1",
            "--study", str(study),
            "--no-traces",
            "--games-out", str(games),
            "--workers", "2",
        ]
    )
    assert code == 0
    lines = _jsonl(games)
    assert len(lines) == 6
    assert {line["kind"] for line in lines} == {"anchor", "cross"}
    with pytest.raises(SystemExit):
        builder.main(
            [
                "--candidate", "c1=random",
                "--study", str(study),
                "--workers", "0",
            ]
        )


def test_cli_head_to_head_workers_match_single(capsys):
    tool = _load_tool("head_to_head")

    def run(workers: int) -> dict:
        code = tool.main(
            [
                "--left", "a=random",
                "--right", "b=random",
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
        tool.main(["--left", "a=random", "--right", "b=random", "--workers", "0"])


def test_build_ladder_forwards_games_out(tmp_path):
    entrants = [*anchors(), Entrant("g1", "fixed"), Entrant("r1", "random")]
    path = tmp_path / "ladder_games.jsonl"
    ladder = build_ladder(
        entrants,
        games_per_anchor=2,
        seed=2,
        count=2,
        max_spacing=300.0,
        factory=_factory,
        games_out=path,
    )
    lines = _jsonl(path)
    assert len(lines) == len(ladder.schedule) == 6  # 4 gauge + 2 cross games
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
            "--candidate", "c1=random",
            "--candidate", "c2=random",
            "--games-per-anchor", "2",
            "--cross", "0",
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
    assert len(lines) == 4  # 2 games vs the gauge x 2 candidates
    assert {line["kind"] for line in lines} == {"anchor"}
    assert {line["rules_id"] for line in lines} == {rules_id(DEFAULT_RULES)}
    assert not study.exists()  # --no-traces still skips the manifest


def test_cli_search_spec_points_at_the_run_local_factory(tmp_path):
    """``rolloutt:`` is outside build_ladder's grammar: fail loud with the path
    to the driver that can build it, not a bare unknown-spec crash."""
    builder = _load_tool("build_ladder")
    with pytest.raises(SystemExit, match="run-local factory"):
        builder.main(
            [
                "--anchor", "random=random",
                "--candidate",
                "search_leafq=rolloutt:runs/ei2_value_t5/t_leafq/critic.pt",
                "--study", str(tmp_path / "study"),
                "--no-traces",
            ]
        )


def test_cli_warns_when_cross_needs_two_candidates(tmp_path, capsys):
    """``--cross > 0`` with one candidate silently planned zero games."""
    builder = _load_tool("build_ladder")
    study = tmp_path / "study"
    code = builder.main(
        [
            "--anchor", "random=random",
            "--candidate", "c1=random",
            "--games-per-anchor", "2",
            "--cross", "2",
            "--seed", "1",
            "--study", str(study),
            "--no-traces",
        ]
    )
    assert code == 0
    err = capsys.readouterr().err
    assert "cross=2" in err
    assert "fewer than two --candidate" in err


def test_cli_cross_default_also_warns_for_one_candidate(tmp_path, capsys):
    """The implicit ``cross=games_per_anchor`` is warned about too."""
    builder = _load_tool("build_ladder")
    code = builder.main(
        [
            "--anchor", "random=random",
            "--candidate", "c1=random",
            "--games-per-anchor", "2",
            "--seed", "1",
            "--study", str(tmp_path / "study"),
            "--no-traces",
        ]
    )
    assert code == 0
    assert "fewer than two --candidate" in capsys.readouterr().err


def test_cli_two_candidates_do_not_warn_about_cross(tmp_path, capsys):
    builder = _load_tool("build_ladder")
    code = builder.main(
        [
            "--anchor", "random=random",
            "--candidate", "c1=random",
            "--candidate", "c2=random",
            "--games-per-anchor", "2",
            "--cross", "2",
            "--seed", "1",
            "--study", str(tmp_path / "study"),
            "--no-traces",
        ]
    )
    assert code == 0
    assert "fewer than two --candidate" not in capsys.readouterr().err


class _HaltGameplay(Exception):
    """Raised by the fake build_ladder to stop the CLI right after wiring."""


def test_cli_resolves_manifest_priors_and_explicit_prior_wins(tmp_path, monkeypatch):
    builder = _load_tool("build_ladder")
    study = tmp_path / "study"
    save_manifest(
        study / "manifest.json",
        {
            "version": 1,
            "rules_id": rules_id(DEFAULT_RULES),
            "levels": {"c1": 300.0, "random": 0.0},
            "subjects": [{"id": "c1", "mu": 320.0, "sigma": 40.0}],
        },
    )
    captured: dict = {}

    def fake_build_ladder(entrants, **kwargs):
        captured["entrants"] = tuple(entrants)
        raise _HaltGameplay

    monkeypatch.setattr(builder, "build_ladder", fake_build_ladder)
    base = [
        "--anchor", "random=random",
        "--candidate", "c1=random",
        "--study", str(study),
        "--no-traces",
    ]
    with pytest.raises(_HaltGameplay):
        builder.main(base)
    assert {entrant.id: entrant.prior for entrant in captured["entrants"]} == {
        "random": None,
        "c1": Prior(320.0, 40.0),  # manifest subject, even under --no-traces
    }

    with pytest.raises(_HaltGameplay):
        builder.main([*base, "--prior", "c1=500:25"])
    assert {entrant.id: entrant.prior for entrant in captured["entrants"]} == {
        "random": None,
        "c1": Prior(500.0, 25.0),  # explicit --prior wins over the manifest
    }


def test_cli_refuses_manifest_priors_without_a_rules_identity(tmp_path, monkeypatch):
    """A legacy manifest must not warm-start the fit (ADR-0013)."""
    builder = _load_tool("build_ladder")
    study = tmp_path / "study"
    save_manifest(
        study / "manifest.json",
        {
            "version": 1,
            "levels": {"c1": 300.0, "random": 0.0},
            "subjects": [{"id": "c1", "mu": 320.0, "sigma": 40.0}],
        },
    )

    def fake_build_ladder(*args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("build_ladder ran on an unidentifiable manifest")

    monkeypatch.setattr(builder, "build_ladder", fake_build_ladder)
    with pytest.raises(SystemExit, match="rules_id"):
        builder.main(
            [
                "--anchor", "random=random",
                "--candidate", "c1=random",
                "--study", str(study),
                "--no-traces",
            ]
        )


def test_cli_refuses_manifest_priors_from_another_rules_version(tmp_path, monkeypatch):
    """A cross-version manifest must not leak warm-start levels (ADR-0013)."""
    builder = _load_tool("build_ladder")
    study = tmp_path / "study"
    save_manifest(
        study / "manifest.json",
        {
            "version": 1,
            "rules_id": rules_id(Rules(straight_max=6)),
            "levels": {"c1": 300.0, "random": 0.0},
            "subjects": [{"id": "c1", "mu": 320.0, "sigma": 40.0}],
        },
    )

    def fake_build_ladder(*args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("build_ladder ran on a foreign-rules manifest")

    monkeypatch.setattr(builder, "build_ladder", fake_build_ladder)
    with pytest.raises(SystemExit, match="rules_id"):
        builder.main(
            [
                "--anchor", "random=random",
                "--candidate", "c1=random",
                "--study", str(study),
            ]
        )


def test_build_ladder_fits_anchors_and_candidates():
    entrants = [*anchors(), Entrant("g1", "fixed"), Entrant("r1", "random")]
    ladder = build_ladder(
        entrants,
        games_per_anchor=4,
        seed=2,
        count=2,
        max_spacing=300.0,
        factory=_factory,
    )
    assert set(ladder.fit.ratings) == {"random", "g1", "r1"}
    assert ladder.fit.games == 12  # 8 gauge + 4 cross games
    assert ladder.fit.ratings["random"].sigma == 0.0
    assert all(math.isfinite(rating.mu) for rating in ladder.fit.ratings.values())
    assert len(ladder.selection.rungs) == 2
    assert ladder.selection.rungs[0].id == "r1"


def test_build_ladder_requires_a_gauge():
    entrants = [Entrant("random", "random"), Entrant("g1", "fixed")]
    with pytest.raises(ValueError):
        build_ladder(entrants, games_per_anchor=1)


def test_cli_manifest_feeds_the_d1_tool(tmp_path):
    builder = _load_tool("build_ladder")
    measure = _load_tool("measure_trace_signal")
    study = tmp_path / "study"
    code = builder.main(
        [
            "--anchor", "random=random",
            "--candidate", "c1=random",
            "--games-per-anchor", "2",
            "--seed", "1",
            "--study", str(study),
            "--rungs", "2",
            "--spacing", "50:200",
        ]
    )
    assert code == 0
    manifest = load_manifest(study / "manifest.json")
    assert {"random", "c1"} <= set(manifest["levels"])
    assert manifest["estimator"]["kind"] == "openskill-plackett-luce"
    assert manifest["rules_id"] == rules_id(DEFAULT_RULES)
    assert manifest["rules"] == rules_identity(DEFAULT_RULES)

    csv_path = tmp_path / "features.csv"
    assert measure.cmd_features(
        argparse.Namespace(study=str(study), out=str(csv_path), level=[], verify=True)
    ) == 0
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2  # 2 games vs the gauge (seat-paired twins)
    assert {row["level_id"] for row in rows} == {"c1"}
    for row in rows:
        assert float(row["level_elo_ref"]) == pytest.approx(manifest["levels"]["c1"])

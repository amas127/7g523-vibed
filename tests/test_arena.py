"""Arena: shard coverage/merge order, parallel determinism, CLI smoke."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from seven523.arena import (
    arena_document,
    auto_id_from_path,
    infer_step,
    pair_diagnostics,
    pair_stats,
    play_parallel,
    ranking_rows,
    run_arena,
    write_tensorboard,
)
from seven523.ladder import (
    Entrant,
    make_factory,
    plan_games,
    split_schedule,
)
from seven523.policies import policy_from_spec
from seven523.rules import DEFAULT_RULES

try:
    import torch
except ImportError:  # pragma: no cover - torch is a train-group extra
    torch = None  # type: ignore[assignment]

requires_torch = pytest.mark.skipif(torch is None, reason="torch (train group) not installed")

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"


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


def _tiny_agent(path: Path, *, hidden: int = 8) -> None:
    from seven523.networks import Agent, save_agent

    torch.manual_seed(0)
    save_agent(path, Agent(161, [134, 4], hidden=hidden))


def _snapshot(games) -> list[tuple[int, tuple[str, ...], tuple[int, ...]]]:
    return [(game.seed, game.seats, game.scores) for game in games]


def test_split_schedule_covers_in_order_and_balances():
    entrants = [*anchors(), Entrant("c1", "greedy"), Entrant("c2", "random")]
    schedule = plan_games(entrants, games_per_anchor=4, cross=4, seed=3)
    shards = split_schedule(schedule, 5)
    assert sum(len(shard) for shard in shards) == len(schedule)
    assert [game for shard in shards for game in shard] == list(schedule)
    assert max(len(shard) for shard in shards) - min(
        len(shard) for shard in shards
    ) <= 1


def test_split_schedule_more_shards_than_games():
    entrants = [*anchors(), Entrant("c1", "greedy")]
    schedule = plan_games(entrants, games_per_anchor=2, seed=0)
    shards = split_schedule(schedule, 100)
    assert len(shards) == 100
    assert sum(len(shard) for shard in shards) == len(schedule)
    assert all(not shard for shard in shards[len(schedule) :])
    with pytest.raises(ValueError):
        split_schedule(schedule, 0)


def test_play_parallel_matches_serial_and_merges_in_order(tmp_path):
    entrants = [*anchors(), Entrant("c1", "greedy"), Entrant("c2", "random")]
    schedule = plan_games(entrants, games_per_anchor=4, cross=2, seed=3)
    serial = play_parallel(
        schedule, entrants, workers=1, games_out=tmp_path / "serial"
    )
    parallel = play_parallel(
        schedule, entrants, workers=3, games_out=tmp_path / "parallel"
    )
    assert _snapshot(parallel) == _snapshot(serial)
    assert [game.seed for game in parallel] == [game.seed for game in schedule]

    shard_files = sorted((tmp_path / "parallel").glob("shard_*.jsonl"))
    assert len(shard_files) == 3
    merged = [
        json.loads(line)
        for path in shard_files
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(merged) == len(schedule)
    # Shards are contiguous, so concatenating them restores schedule order and
    # matches the single-process JSONL byte for byte.
    serial_text = (tmp_path / "serial" / "shard_00000.jsonl").read_text(encoding="utf-8")
    parallel_text = "".join(
        path.read_text(encoding="utf-8") for path in shard_files
    )
    assert parallel_text == serial_text


def test_play_parallel_workers_validation():
    entrants = [*anchors(), Entrant("c1", "greedy")]
    schedule = plan_games(entrants, games_per_anchor=2, seed=0)
    with pytest.raises(ValueError):
        play_parallel(schedule, entrants, workers=0)
    assert play_parallel((), entrants, workers=2) == []


def test_run_arena_ranks_and_documents():
    entrants = [*anchors(), Entrant("g1", "greedy"), Entrant("r1", "random")]
    result = run_arena(entrants, games_per_anchor=20, cross=20, seed=1, workers=1)
    assert len(result.games) == 20 * 2 * 2 + 20
    assert result.fit.converged
    rows = ranking_rows(result)
    # Seed 1 under the RULES.md §3 family comparison: g1 (the same GreedyBot
    # policy as the greedy anchor) edges past the anchor, whose pin still holds.
    assert rows[0]["id"] == "g1"
    assert next(row["pinned"] for row in rows if row["id"] == "greedy") == 1315.0
    assert [row["elo"] for row in rows] == sorted(
        (row["elo"] for row in rows), reverse=True
    )
    assert all(row["se"] > 0 for row in rows if row["role"] == "candidate")
    assert result.fit.ratings["random"].se == 0.0

    document = arena_document(
        result,
        seed=1,
        games_per_anchor=20,
        cross=20,
        workers=1,
        device="cpu",
    )
    assert document["games"] == len(result.games)
    assert document["ranking"] == [row["id"] for row in rows]
    assert document["anchors"] == {"random": 1000.0, "greedy": 1315.0}
    assert sum(entry["games"] for entry in document["pair_stats"].values()) == len(
        result.games
    )
    for entry in document["transitivity"]:
        assert entry["games"] >= 20
        assert 0.0 < entry["expected"] < 1.0


def test_pair_stats_are_symmetric_and_ordered():
    entrants = [*anchors(), Entrant("g1", "greedy")]
    schedule = plan_games(entrants, games_per_anchor=4, seed=2)
    games = play_parallel(schedule, entrants, workers=1)
    stats = pair_stats(games)
    assert set(stats) == {"g1|greedy", "g1|random"}
    for entry in stats.values():
        assert entry["a_wins"] + entry["b_wins"] + entry["draws"] == entry["games"]
        assert 0.0 <= entry["a_rate"] <= 1.0


def test_pair_diagnostics_sorted_by_abs_z():
    entrants = [*anchors(), Entrant("g1", "greedy"), Entrant("r1", "random")]
    result = run_arena(entrants, games_per_anchor=20, cross=20, seed=5, workers=1)
    rows = pair_diagnostics(result, min_games=20)
    assert rows
    assert [abs(row["z"]) for row in rows] == sorted(
        (abs(row["z"]) for row in rows), reverse=True
    )


def test_step_inference_and_auto_ids():
    assert infer_step("base_step00696320.pt") == 696320
    assert infer_step("sp_step00983040") == 983040
    assert infer_step("w5_ctrl__1__1790319698") is None
    assert auto_id_from_path("runs/probe/base_step00020480.pt") == "base20480"
    assert auto_id_from_path("runs/probe_sp/sp_step00983040.pt") == "sp983040"
    assert auto_id_from_path("runs/w5_ctrl__1__1790319698/agent.pt") == "w5_ctrl"
    assert auto_id_from_path("runs/lvlbase__1__1790313091/agent.pt") == "lvlbase"
    assert auto_id_from_path("runs/lvl1/agent.pt") == "lvl1"


def test_make_factory_builds_scripted_policies():
    factory = make_factory(device="cpu")
    assert factory("greedy", DEFAULT_RULES, 0).__class__.__name__ == "GreedyBot"
    bot = factory("random", DEFAULT_RULES, 0)
    assert bot.__class__.__name__ == "RandomBot"


@requires_torch
def test_policy_from_spec_caches_ckpt_agent_loads(tmp_path, monkeypatch):
    from seven523 import networks, policies

    path = tmp_path / "agent.pt"
    _tiny_agent(path)
    calls: list[tuple] = []
    real_load = networks.load_agent

    def counting_load(*args, **kwargs):
        calls.append(args)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(networks, "load_agent", counting_load)
    policies._AGENT_CACHE.clear()
    try:
        first = policy_from_spec(f"ckpt:{path}", DEFAULT_RULES, 1)
        second = policy_from_spec(f"ckpt:{path}", DEFAULT_RULES, 2)
    finally:
        policies._AGENT_CACHE.clear()
    assert len(calls) == 1
    assert first.agent is second.agent  # same cached weights, fresh policy wrapper


@requires_torch
def test_write_tensorboard_logs_step_bearing_entrants(tmp_path):
    entrants = [*anchors(), Entrant("base_step00020480", "greedy")]
    result = run_arena(entrants, games_per_anchor=2, seed=0, workers=1)
    written = write_tensorboard(result, tmp_path / "tb", steps={"g1": 999})
    assert written == 1  # the anchor is skipped, the step-named candidate is kept
    assert list((tmp_path / "tb").glob("events.out.tfevents.*"))


def test_cli_smoke_writes_table_and_shards(tmp_path):
    tool = _load_tool("arena")
    out = tmp_path / "arena.json"
    games_out = tmp_path / "games"
    code = tool.main(
        [
            "--anchor", "random=random",
            "--anchor", "greedy=greedy",
            "--entrant", "g1=greedy",
            "--entrant", "r1=random",
            "--games-per-anchor", "4",
            "--cross", "4",
            "--seed", "1",
            "--workers", "2",
            "--out", str(out),
            "--games-out", str(games_out),
            "--step-map", "g1=1234",
        ]
    )
    assert code == 0
    document = json.loads(out.read_text(encoding="utf-8"))
    assert document["games"] == 4 * 2 * 2 + 4
    assert document["ranking"][0] == "g1"
    assert document["entrants"][0]["step"] in (None, 1234)
    shard_files = sorted(games_out.glob("shard_*.jsonl"))
    assert len(shard_files) == 2
    assert sum(len(path.read_text(encoding="utf-8").splitlines()) for path in shard_files) == document["games"]


def test_cli_rejects_duplicate_ids(tmp_path):
    tool = _load_tool("arena")
    with pytest.raises(SystemExit):
        tool.main(
            [
                "--entrant", "g1=greedy",
                "--entrant", "g1=random",
                "--games-per-anchor", "2",
                "--cross", "0",
                "--out", str(tmp_path / "x.json"),
            ]
        )


@requires_torch
def test_cli_glob_discovery_with_every_and_last(tmp_path):
    tool = _load_tool("arena")
    for step in (20480, 40960, 61440):
        _tiny_agent(tmp_path / f"base_step{step:08d}.pt")
    out = tmp_path / "out.json"
    code = tool.main(
        [
            "--glob", str(tmp_path / "base_step*.pt"),
            "--every", "2",
            "--last",
            "--games-per-anchor", "2",
            "--cross", "0",
            "--workers", "2",
            "--seed", "0",
            "--out", str(out),
        ]
    )
    assert code == 0
    document = json.loads(out.read_text(encoding="utf-8"))
    ids = {row["id"] for row in document["entrants"]}
    assert {"base20480", "base61440"} <= ids
    steps = {row["id"]: row["step"] for row in document["entrants"]}
    assert steps["base20480"] == 20480
    assert steps["base61440"] == 61440


def test_cli_glob_missing_pattern_errors():
    tool = _load_tool("arena")
    with pytest.raises(SystemExit):
        tool.main(
            [
                "--glob", "definitely/not/here/*.pt",
                "--games-per-anchor", "2",
                "--cross", "0",
            ]
        )

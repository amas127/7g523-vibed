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
    # The pinned gauge: the RandomBot defines 0 (ADR-0012).
    return [Entrant("random", "random", pinned=0.0)]


def _tiny_agent(path: Path, *, hidden: int = 8) -> None:
    from seven523.networks import Agent, save_agent

    torch.manual_seed(0)
    save_agent(path, Agent(161, [134, 4], hidden=hidden))


def test_split_schedule_covers_in_order_and_balances():
    entrants = [*anchors(), Entrant("c1", "random"), Entrant("c2", "random")]
    schedule = plan_games(entrants, games_per_anchor=4, cross=4, seed=3)
    shards = split_schedule(schedule, 5)
    assert sum(len(shard) for shard in shards) == len(schedule)
    assert [game for shard in shards for game in shard] == list(schedule)
    assert max(len(shard) for shard in shards) - min(
        len(shard) for shard in shards
    ) <= 1


def test_split_schedule_more_shards_than_games():
    entrants = [*anchors(), Entrant("c1", "random")]
    schedule = plan_games(entrants, games_per_anchor=2, seed=0)
    shards = split_schedule(schedule, 100)
    assert len(shards) == 100
    assert sum(len(shard) for shard in shards) == len(schedule)
    assert all(not shard for shard in shards[len(schedule) :])
    with pytest.raises(ValueError):
        split_schedule(schedule, 0)


def test_run_arena_ranks_and_documents():
    entrants = [
        *anchors(),
        Entrant("c1", "random"),
        Entrant("c2", "random"),
    ]
    result = run_arena(entrants, games_per_anchor=20, cross=2, seed=1, workers=1)
    assert len(result.games) == 20 // 2 * 1 * 2 * 2 + 2
    assert result.fit.games == len(result.games)
    rows = ranking_rows(result)
    assert {row["id"] for row in rows} == {"random", "c1", "c2"}
    assert [row["mu"] for row in rows] == sorted(
        (row["mu"] for row in rows), reverse=True
    )
    assert all(row["sigma"] > 0 for row in rows if row["role"] == "candidate")
    assert result.fit.ratings["random"].sigma == 0.0

    document = arena_document(
        result,
        seed=1,
        games_per_anchor=20,
        cross=2,
        workers=1,
        device="cpu",
    )
    assert document["games"] == len(result.games)
    assert document["ranking"] == [row["id"] for row in rows]
    assert document["anchors"] == {"random": 0.0}
    assert sum(entry["games"] for entry in document["pair_stats"].values()) == len(
        result.games
    )
    for entry in document["transitivity"]:
        assert entry["games"] >= 20
        assert 0.0 < entry["expected"] < 1.0

    # Largest observed-vs-rating disagreements, sorted by |z| descending.
    diagnostics = pair_diagnostics(result, min_games=20)
    assert diagnostics
    assert [abs(row["z"]) for row in diagnostics] == sorted(
        (abs(row["z"]) for row in diagnostics), reverse=True
    )


def test_pair_stats_are_symmetric_and_ordered():
    entrants = [*anchors(), Entrant("c1", "random")]
    schedule = plan_games(entrants, games_per_anchor=4, seed=2)
    games = play_parallel(schedule, entrants, workers=1)
    stats = pair_stats(games)
    assert set(stats) == {"c1|random"}
    for entry in stats.values():
        assert entry["a_wins"] + entry["b_wins"] + entry["draws"] == entry["games"]
        assert 0.0 <= entry["a_rate"] <= 1.0


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
    bot = factory("random", DEFAULT_RULES, 0)
    assert bot.__class__.__name__ == "RandomBot"
    with pytest.raises(ValueError):
        factory("greedy", DEFAULT_RULES, 0)


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
    entrants = [*anchors(), Entrant("base_step00020480", "random")]
    result = run_arena(entrants, games_per_anchor=2, seed=0, workers=1)
    written = write_tensorboard(result, tmp_path / "tb", steps={})
    assert written == 1  # the anchor is skipped, the step-named candidate is kept
    assert list((tmp_path / "tb").glob("events.out.tfevents.*"))


def test_cli_smoke_writes_table_and_shards(tmp_path):
    tool = _load_tool("arena")
    out = tmp_path / "arena.json"
    games_out = tmp_path / "games"
    code = tool.main(
        [
            "--anchor", "random=random",
            "--entrant", "c1=random",
            "--entrant", "c2=random",
            "--games-per-anchor", "2",
            "--cross", "2",
            "--seed", "1",
            "--workers", "2",
            "--out", str(out),
            "--games-out", str(games_out),
            "--step-map", "c1=1234",
        ]
    )
    assert code == 0
    document = json.loads(out.read_text(encoding="utf-8"))
    assert document["games"] == 2 // 2 * 1 * 2 * 2 + 2
    assert "c1" in document["ranking"]
    assert document["entrants"][0]["step"] in (None, 1234)
    shard_files = sorted(games_out.glob("shard_*.jsonl"))
    assert len(shard_files) == 2
    assert sum(len(path.read_text(encoding="utf-8").splitlines()) for path in shard_files) == document["games"]


def test_cli_rejects_duplicate_ids_and_missing_globs(tmp_path):
    tool = _load_tool("arena")
    with pytest.raises(SystemExit):
        tool.main(
            [
                "--entrant", "g1=random",
                "--entrant", "g1=random",
                "--games-per-anchor", "2",
                "--cross", "0",
                "--out", str(tmp_path / "x.json"),
            ]
        )
    with pytest.raises(SystemExit):
        tool.main(
            [
                "--glob", "definitely/not/here/*.pt",
                "--games-per-anchor", "2",
                "--cross", "0",
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
            "--workers", "1",
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

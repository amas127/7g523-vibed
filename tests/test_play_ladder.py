"""``tools/play_ladder.py``: registry integrity and 7g523-play argv plumbing.

After the old-rule/old-obs checkpoint rungs were retired, the registry only
advertises the two scripted anchors.  The checkpoint checks (file existence)
are kept so they cover re-added rungs without edits; ``play`` is checked at
the argv boundary (the real game loop is covered by ``tests/test_play.py``).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
ROOT = TOOLS_DIR.parent


def _load_tool(name: str):
    spec = importlib.util.spec_from_file_location(name, TOOLS_DIR / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


play_ladder = _load_tool("play_ladder")

#: The advertised batch: the two scripted anchors, no checkpoint rungs yet.
EXPECTED_IDS = [
    "random",
    "greedy",
]


def _forwarded(monkeypatch, argv):
    import seven523.play as play_module

    seen: dict[str, list[str]] = {}
    monkeypatch.setattr(
        play_module, "main", lambda forwarded: seen.setdefault("argv", forwarded)
    )
    assert play_ladder.main(argv) == 0
    return seen["argv"]


def _synthetic_rung(**overrides) -> play_ladder.Opponent:
    """A checkpoint entry using the OPPONENTS seam placeholder path."""
    fields = {
        "id": "new-run",
        "kind": "rung",
        "strength": "占位（新口径）",
        "arena": None,
        "ckpt": "runs/<new-run>/agent.pt",
        "note": "接缝占位档",
    }
    fields.update(overrides)
    return play_ladder.Opponent(**fields)


def test_registry_covers_the_advertised_set():
    assert [opponent.id for opponent in play_ladder.OPPONENTS] == EXPECTED_IDS
    assert len(play_ladder.BY_ID) == len(EXPECTED_IDS)
    for opponent in play_ladder.OPPONENTS:
        assert opponent.kind in play_ladder.KIND_LABELS
        assert opponent.strength and opponent.note
        if opponent.kind == "script":
            assert opponent.ckpt is None
        else:
            assert opponent.ckpt


def test_every_checkpoint_exists():
    # Vacuously true while no checkpoint rungs are registered; guards re-added
    # entries against typo'd paths.
    for opponent in play_ladder.OPPONENTS:
        path = play_ladder.ckpt_path(opponent)
        if path is not None:
            assert path.is_file(), f"{opponent.id}: {path} is missing"


def test_build_play_argv_for_scripts():
    assert play_ladder.build_play_argv(play_ladder.BY_ID["random"]) == [
        "--opponent",
        "random",
    ]
    assert play_ladder.build_play_argv(
        play_ladder.BY_ID["greedy"], ["--seat", "1"]
    ) == ["--opponent", "greedy", "--seat", "1"]


def test_build_play_argv_for_checkpoint():
    rung = _synthetic_rung()
    assert play_ladder.build_play_argv(rung, ["--rounds", "3"]) == [
        "--checkpoint",
        str(ROOT / "runs/<new-run>/agent.pt"),
        "--rounds",
        "3",
    ]
    assert play_ladder.launch_command(rung) == (
        "uv run --group train 7g523-play "
        "--checkpoint runs/<new-run>/agent.pt"
    )


def test_list_output_mentions_every_id_path_and_command(capsys):
    assert play_ladder.main(["list"]) == 0
    out = capsys.readouterr().out
    for opponent in play_ladder.OPPONENTS:
        assert opponent.id in out
        if opponent.ckpt:
            assert opponent.ckpt in out
        assert play_ladder.launch_command(opponent) in out
    assert "没有 checkpoint 档位" in out


def test_format_list_flags_missing_checkpoint():
    out = play_ladder.format_list([_synthetic_rung()])
    assert "runs/<new-run>/agent.pt（缺失）" in out
    assert "警告" in out
    assert "缺失" in out


def test_list_json_document(capsys):
    assert play_ladder.main(["list", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [entry["id"] for entry in payload] == EXPECTED_IDS
    greedy = next(entry for entry in payload if entry["id"] == "greedy")
    assert greedy["kind"] == "script"
    assert greedy["ckpt"] is None
    assert greedy["path"] is None
    assert greedy["play_argv"] == ["--opponent", "greedy"]
    assert greedy["command"] == "uv run 7g523-play --opponent greedy"


def test_play_forwards_extra_args(monkeypatch):
    assert _forwarded(
        monkeypatch, ["play", "greedy", "--", "--seat", "1", "--rounds", "3"]
    ) == [
        "--opponent",
        "greedy",
        "--seat",
        "1",
        "--rounds",
        "3",
    ]


def test_play_script_id(monkeypatch):
    assert _forwarded(monkeypatch, ["play", "random", "--", "--rounds", "2"]) == [
        "--opponent",
        "random",
        "--rounds",
        "2",
    ]


def test_play_help_after_id_is_forwarded(monkeypatch):
    assert _forwarded(monkeypatch, ["play", "greedy", "--help"])[-1] == "--help"


def test_play_unknown_id_fails(capsys):
    assert play_ladder.main(["play", "nope"]) == 2
    assert "未知对手" in capsys.readouterr().err


def test_play_missing_checkpoint_fails(monkeypatch, capsys):
    ghost = _synthetic_rung(id="ghost", ckpt="runs/ghost/agent.pt")
    monkeypatch.setitem(play_ladder.BY_ID, "ghost", ghost)
    assert play_ladder.main(["play", "ghost"]) == 2
    err = capsys.readouterr().err
    assert "检查点不存在" in err
    assert "runs/ghost/agent.pt" in err


def test_help_is_available(capsys):
    with pytest.raises(SystemExit) as excinfo:
        play_ladder.main(["--help"])
    assert excinfo.value.code == 0
    assert "list" in capsys.readouterr().out
    with pytest.raises(SystemExit) as excinfo:
        play_ladder.main(["play", "--help"])
    assert excinfo.value.code == 0

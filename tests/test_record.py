"""``record.py``: the single play → trace → save seam.

The three batch harnesses (ladder, placement, the D1 tool) used to re-type this
pipeline; these tests pin the seam's own contract: scores equal the trace's
final scores, the file name is the study convention, the JSON round-trips, and
a game without a ``trace_dir`` writes nothing.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from seven523.policies import make_scripted_policies
from seven523.record import RecordedGame, play_recorded, policy_seed
from seven523.rules import DEFAULT_RULES
from seven523.trace import load_trace, trace_filename

SILENT = lambda *args, **kwargs: None  # noqa: E731

PLAYERS = ("subject:hero@seat0", "anchor:greedy@seat1")


def _recorded(tmp_path: Path, **overrides) -> RecordedGame:
    settings: dict = {
        "rules": DEFAULT_RULES,
        "seed": 42,
        "human_seat": 0,
        "players": PLAYERS,
        "created_at": "2025-01-01T00:00:00",
        "trace_dir": tmp_path,
        "trace_index": 7,
        "opponent": "greedy",
        "print_fn": SILENT,
    }
    settings.update(overrides)
    return play_recorded(make_scripted_policies("random", DEFAULT_RULES, 3), **settings)


def test_play_recorded_scores_name_and_round_trip(tmp_path):
    recorded = _recorded(tmp_path)
    assert recorded.seed == 42
    assert recorded.human_seat == 0
    assert recorded.scores == tuple(recorded.trace["final_scores"])
    assert sum(recorded.scores) == DEFAULT_RULES.total_points
    assert recorded.trace_path is not None
    assert recorded.trace_path == tmp_path / trace_filename(7, 42, 0, "greedy")
    assert recorded.trace_path.name == trace_filename(7, 42, 0, "greedy")

    loaded = load_trace(recorded.trace_path)
    assert loaded == recorded.trace
    assert loaded["players"] == list(PLAYERS)
    assert loaded["seed"] == 42
    assert loaded["human_seat"] == 0
    assert loaded["created_at"] == "2025-01-01T00:00:00"


def test_play_recorded_without_trace_dir_saves_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    recorded = _recorded(tmp_path, trace_dir=None)
    assert recorded.trace_path is None
    assert recorded.scores == tuple(recorded.trace["final_scores"])
    assert not list(tmp_path.rglob("*.json"))


def test_play_recorded_requires_players_and_opponent_when_saving(tmp_path):
    with pytest.raises(ValueError):
        _recorded(tmp_path, players=None)
    with pytest.raises(ValueError):
        _recorded(tmp_path, players=[])
    with pytest.raises(ValueError):
        _recorded(tmp_path, opponent="")


def test_policy_seed_is_stable_and_unique_per_seat():
    assert policy_seed(42, 0) == (42 + 101) & 0xFFFF_FFFF
    assert policy_seed(42, 0) == policy_seed(42, 0)
    assert policy_seed(42, 0) != policy_seed(42, 1)
    assert len({policy_seed(42, seat) for seat in range(4)}) == 4
    assert policy_seed(0xFFFF_FFFF, 3) == (0xFFFF_FFFF + 4 * 101) & 0xFFFF_FFFF


def test_two_recorded_runs_with_the_same_inputs_are_identical(tmp_path):
    first = _recorded(tmp_path / "first")
    second = _recorded(tmp_path / "second")
    assert first.scores == second.scores
    assert first.trace == second.trace
    assert first.trace_path is not None and second.trace_path is not None
    assert first.trace_path.read_bytes() == second.trace_path.read_bytes()

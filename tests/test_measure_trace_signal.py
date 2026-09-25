"""D1 smoke tests: the trace-signal tool generates, extracts and calibrates.

The tool lives outside the package (``tools/measure_trace_signal.py``) and is
imported here by path so the pipeline stays exercised like the rest of the repo.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

TOOL_PATH = Path(__file__).resolve().parents[1] / "tools" / "measure_trace_signal.py"
_spec = importlib.util.spec_from_file_location("measure_trace_signal", TOOL_PATH)
assert _spec and _spec.loader
measure = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = measure
_spec.loader.exec_module(measure)


def _generate(study: Path, subjects: list[str], games: int) -> None:
    args = argparse.Namespace(
        out=str(study),
        games=games,
        seed=0,
        num_players=2,
        subject=subjects,
        anchor=None,
        level=[],
    )
    assert measure.cmd_generate(args) == 0


def _features(study: Path, csv_path: Path) -> list[dict[str, str]]:
    args = argparse.Namespace(
        study=str(study), out=str(csv_path), level=[], verify=True
    )
    assert measure.cmd_features(args) == 0
    with csv_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_generate_and_extract_s1_features(tmp_path):
    study, csv_path = tmp_path / "study", tmp_path / "features.csv"
    _generate(study, ["random"], games=4)
    rows = _features(study, csv_path)

    assert len(rows) == 4
    assert list(rows[0]) == list(measure.FEATURE_COLUMNS)
    assert {row["level_id"] for row in rows} == {"random"}
    assert {float(row["level_elo_ref"]) for row in rows} == {1000.0}
    assert {int(row["seat"]) for row in rows} == {0, 1}  # seats rotate
    for row in rows:
        assert row["result"] in {"win", "draw", "loss"}
        assert float(row["tricks_won"]) <= float(row["tricks_total"])
        assert float(row["decisions"]) >= float(row["tricks_total"])
        assert -100.0 <= float(row["score_diff"]) <= 100.0
        for rate in ("trick_win_rate", "pass_rate", "bomb_rate", "lead_rate"):
            assert 0.0 <= float(row[rate]) <= 1.0
    # the paired design uses the same deal seed across subjects
    _generate(study, ["greedy"], games=4)
    all_rows = _features(study, tmp_path / "features_all.csv")
    random_seeds = sorted(row["seed"] for row in all_rows if row["level_id"] == "random")
    greedy_seeds = sorted(row["seed"] for row in all_rows if row["level_id"] == "greedy")
    assert random_seeds == greedy_seeds


def test_generate_manifest_keys_and_frozen_levels(tmp_path):
    study = tmp_path / "study"
    _generate(study, ["random"], games=2)
    manifest = json.loads((study / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest) == {
        "version",
        "created_at",
        "seed",
        "games",
        "num_players",
        "levels",
        "subjects",
        "anchors",
    }
    assert manifest["levels"] == {"random": 1000.0, "greedy": 1315.0}
    assert manifest["subjects"] == [
        {"id": "random", "spec": "random", "elo": 1000.0}
    ]
    assert manifest["anchors"] == [
        {"id": "random", "elo": 1000.0},
        {"id": "greedy", "elo": 1315.0},
    ]
    # A re-run with an explicit new level does not thaw the frozen label;
    # the D1 metadata (seed) still refreshes.
    args = argparse.Namespace(
        out=str(study),
        games=2,
        seed=1,
        num_players=2,
        subject=["random"],
        anchor=None,
        level=["random=1100"],
    )
    assert measure.cmd_generate(args) == 0
    frozen = json.loads((study / "manifest.json").read_text(encoding="utf-8"))
    assert frozen["levels"]["random"] == 1000.0
    assert frozen["seed"] == 1


def test_calibrate_reports_m_eff_but_flags_two_levels(tmp_path, capsys):
    study, artifacts = tmp_path / "study", tmp_path / "artifacts"
    _generate(study, ["random", "greedy"], games=6)
    rows = _features(study, artifacts / "features.csv")
    assert len(rows) == 12

    args = argparse.Namespace(
        features=str(artifacts / "features.csv"),
        out=str(artifacts),
        folds=2,
        bootstrap=50,
        seed=0,
        min_levels=3,
    )
    assert measure.cmd_calibrate(args) == 0
    summary = json.loads((artifacts / "summary.json").read_text(encoding="utf-8"))
    assert summary["levels"] == {"random": 1000.0, "greedy": 1315.0}
    assert isinstance(summary["m_eff"], float) and summary["m_eff"] > 0
    assert summary["conclusive"] is False
    assert "pipeline check only" in capsys.readouterr().out


def test_mann_whitney_auc_handles_separation_and_ties():
    high = np.array([3.0, 4.0, 5.0])
    low = np.array([0.0, 1.0, 2.0])
    assert measure._mann_whitney_auc(high, low) == 1.0
    assert measure._mann_whitney_auc(low, high) == 0.0
    same = np.array([1.0, 2.0, 3.0])
    assert measure._mann_whitney_auc(same, same) == 0.5


def test_spearman_ranks_monotone_pairs():
    assert measure._spearman([(1000, 0.1), (1200, 0.5), (1400, 0.9)]) == 1.0
    assert measure._spearman([(1000, 0.9), (1200, 0.5), (1400, 0.1)]) == -1.0

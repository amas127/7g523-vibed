"""Study manifest schema: round-trips, freeze semantics and D1 compatibility."""
from __future__ import annotations

import json

import pytest

from seven523.study import load_manifest, merge_manifest, save_manifest


def test_missing_manifest_is_an_empty_study(tmp_path):
    assert load_manifest(tmp_path / "manifest.json") == {}


def test_manifest_round_trip(tmp_path):
    path = tmp_path / "study" / "manifest.json"
    document = {"version": 1, "seed": 7, "levels": {"lvl1": 1432.0}}
    assert save_manifest(path, document) == path
    assert load_manifest(path) == document


def test_malformed_manifest_raises(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError):
        load_manifest(path)


def test_merge_freezes_levels_and_subject_elo_by_default():
    document = {
        "seed": 3,
        "levels": {"random": 1000.0, "lvl1": 1400.0},
        "subjects": [{"id": "lvl1", "spec": "ckpt:a.pt", "elo": 1400.0}],
    }
    merged = merge_manifest(
        document,
        levels={"random": 1000.0, "lvl1": 1450.0, "lvl2": 1600.0},
        subjects=[
            {"id": "lvl1", "spec": "ckpt:a.pt", "elo": 1450.0, "games": 200},
            {"id": "lvl2", "spec": "ckpt:b.pt", "elo": 1600.0, "games": 200},
        ],
    )
    assert merged["levels"] == {"random": 1000.0, "lvl1": 1400.0, "lvl2": 1600.0}
    subjects = {entry["id"]: entry for entry in merged["subjects"]}
    assert subjects["lvl1"]["elo"] == 1400.0  # frozen
    assert subjects["lvl1"]["games"] == 200  # other metadata refreshes
    assert subjects["lvl2"]["elo"] == 1600.0
    assert merged["seed"] == 3  # D1 metadata preserved


def test_merge_refit_overwrites_and_accepts_extra_fields():
    document = {"levels": {"lvl1": 1400.0}}
    merged = merge_manifest(
        document,
        levels={"lvl1": 1450.0},
        subjects=[{"id": "lvl1", "elo": 1450.0}],
        anchors=[{"id": "random", "elo": 1000.0}],
        rungs=[{"id": "lvl1", "elo": 1450.0, "se": 30.0}],
        estimator={"kind": "bt-map", "folds": 5},
        frozen_at="2025-01-01T00:00:00",
        refit=True,
    )
    assert merged["levels"] == {"lvl1": 1450.0}
    assert merged["subjects"][0]["elo"] == 1450.0
    assert merged["anchors"] == [{"id": "random", "elo": 1000.0}]
    assert merged["rungs"][0]["id"] == "lvl1"
    assert merged["estimator"]["kind"] == "bt-map"
    assert merged["frozen_at"] == "2025-01-01T00:00:00"
    assert merged["version"] == 1


def test_merge_rejects_subjects_without_ids():
    with pytest.raises(ValueError):
        merge_manifest({}, levels={}, subjects=[{"elo": 1500.0}])
    with pytest.raises(ValueError):
        merge_manifest(
            {"subjects": [{"elo": 1500.0}]}, levels={}, subjects=[]
        )


def test_saved_manifest_is_json_text(tmp_path):
    path = save_manifest(tmp_path / "manifest.json", {"levels": {"a": 1.5}})
    assert json.loads(path.read_text(encoding="utf-8")) == {"levels": {"a": 1.5}}

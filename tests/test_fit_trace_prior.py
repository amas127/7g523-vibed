"""M1 tests for ``tools/fit_trace_prior.py`` (offline 轨迹 S1 prior calibration).

The tool lives outside the package and is imported by path so the offline
pipeline is exercised exactly like the CLI.  The regression tests use a fixed
subset of the checked-out ``traces/study`` ladder (first 25 games of each of the
6 levels) under **both** label maps that matter:

* ``t2`` (default, C-6/T13): HR §1 full-data targets;
* ``manifest`` (comparison only): the 2026-09-25 refit study levels.

If the S1 extractor, the ridge design, the de-shrink or the ``σ_traj`` table
drifts, the locked numbers fail.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from seven523.elo import Prior

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

_spec = importlib.util.spec_from_file_location(
    "fit_trace_prior", TOOLS / "fit_trace_prior.py"
)
assert _spec and _spec.loader
fit = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = fit
_spec.loader.exec_module(fit)

STUDY = ROOT / "traces" / "study"
PER_LEVEL = 25

# ``traces/`` is gitignored (data artifact); a fresh checkout without the ladder
# skips instead of erroring.  The evaluation environment ships the 1200-trace
# study, so the locked regressions below are exercised normally.
pytestmark = pytest.mark.skipif(
    not STUDY.is_dir(), reason="traces/study ladder is not present"
)

#: Locked on the fixed subset (α=30, reps=400, seed=0, LOLO + de-shrink).
T2_EXPECTED = {"a": -683.3, "b": 1.5462, "sigma": {"5": 73.8, "10": 66.7, "20": 63.2}}
MANIFEST_EXPECTED = {
    "a": -673.5,
    "b": 1.5407,
    "sigma": {"5": 63.2, "10": 54.4, "20": 50.2},
}


@pytest.fixture(scope="module")
def ladder():
    manifest = {
        str(name): float(value)
        for name, value in json.loads(
            (STUDY / "manifest.json").read_text(encoding="utf-8")
        )["levels"].items()
    }
    loaded = {}
    for source, labels in (("t2", fit.T2_LABELS), ("manifest", manifest)):
        rows, stats = fit.load_rows(
            STUDY, labels, verify=True, max_per_level=PER_LEVEL
        )
        loaded[source] = (rows, labels, stats)
    return loaded


@pytest.mark.parametrize(
    "source, expected",
    [("t2", T2_EXPECTED), ("manifest", MANIFEST_EXPECTED)],
)
def test_lolo_deshrink_regression_locks_ab_and_sigma(ladder, source, expected):
    rows, labels, stats = ladder[source]
    assert len(rows) == 6 * PER_LEVEL
    assert stats["failed"] == 0

    doc = fit.build_prior(
        rows,
        labels,
        alpha=30.0,
        scheme="lolo",
        deshrink=True,
        reps=400,
        seed=0,
        label_source="t2_full_data" if source == "t2" else "manifest_levels",
    )
    assert doc["scheme"] == "lolo"
    assert doc["labels"] == ("t2_full_data" if source == "t2" else "manifest_levels")
    assert doc["deshrink"]["applied"] is True
    assert doc["deshrink"]["a"] == pytest.approx(expected["a"], abs=0.1)
    assert doc["deshrink"]["b"] == pytest.approx(expected["b"], abs=1e-3)
    assert set(doc["sigma_traj"]) == {str(n) for n in range(1, 21)}
    for n, value in expected["sigma"].items():
        assert doc["sigma_traj"][n] == pytest.approx(value, abs=0.05)


def test_label_map_drives_both_subject_and_opponent_features(ladder):
    t2_rows, _t2_labels, _stats = ladder["t2"]
    manifest_rows, manifest_labels, _stats = ladder["manifest"]

    # T2 mode: both the subject target and the anchor feature use T2 numbers.
    assert {row["level_elo_ref"] for row in t2_rows} == set(fit.T2_LABELS.values())
    assert {row["opponent_elo_ref"] for row in t2_rows} == {
        fit.T2_LABELS["random"],
        fit.T2_LABELS["greedy"],
    }
    # Manifest mode: the same rows carry the manifest anchor ratings instead.
    assert {row["opponent_elo_ref"] for row in manifest_rows} == {
        manifest_labels["random"],
        manifest_labels["greedy"],
    }
    assert fit.T2_LABELS["random"] not in {
        row["opponent_elo_ref"] for row in manifest_rows
    }


def test_prior_schema_and_round_trip(ladder, tmp_path):
    rows, labels, _stats = ladder["t2"]
    doc = fit.build_prior(rows, labels, reps=100)
    path = fit.save_prior(doc, tmp_path / "prior.json", created_at="2026-09-25T00:00:00")
    loaded = fit.load_prior(path)

    assert loaded["schema"] == fit.SCHEMA
    assert loaded["version"] == fit.VERSION
    assert loaded["created_at"] == "2026-09-25T00:00:00"
    assert loaded["labels"] == "t2_full_data"
    assert loaded["hyperparams"]["label_source"] == "t2_full_data"
    assert set(loaded["prior"]) == {"mean", "sd"}
    assert Prior(**loaded["prior"]) == Prior(1500.0, 300.0)
    assert "NaN" not in path.read_text(encoding="utf-8")

    model = loaded["model"]
    assert len(model["features"]) == len(fit.MODEL_FEATURES) == 15
    assert len(model["mean"]) == len(model["scale"]) == len(model["coefficients"]) == 17
    assert model["anchor"] == {
        "name": fit.ANCHOR_FEATURE,
        "center": fit.ANCHOR_CENTER,
        "scale": fit.ANCHOR_SCALE,
    }

    # the serialized model reproduces the in-memory predictions
    for row in rows[:5]:
        assert fit.predict_elo(loaded, row) == pytest.approx(
            fit.predict_elo(doc, row), abs=1e-9
        )

    mu = fit.session_mean(loaded, rows[:3])
    session = fit.prior_for_session(loaded, mu, 10)
    assert session.mean == pytest.approx(mu)
    assert session.sd == pytest.approx(loaded["sigma_traj"]["10"])
    cold = fit.prior_for_session(loaded, 1234.0, 0)
    assert (cold.mean, cold.sd) == (1500.0, 300.0)


def test_build_prior_is_deterministic(ladder):
    rows, labels, _stats = ladder["t2"]
    first = fit.build_prior(rows, labels, reps=100)
    second = fit.build_prior(rows, labels, reps=100)
    assert json.dumps(first, ensure_ascii=False) == json.dumps(second, ensure_ascii=False)


def test_cli_smoke_writes_prior(tmp_path, capsys):
    out = tmp_path / "prior.json"
    code = fit.main(
        [
            "fit",
            "--study",
            str(STUDY),
            "--out",
            str(out),
            "--max-per-level",
            "3",
            "--no-verify",
            "--reps",
            "20",
            "--created-at",
            "2026-09-25T00:00:00",
        ]
    )
    assert code == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["schema"] == fit.SCHEMA
    assert doc["labels"] == "t2_full_data"  # default target per C-6/T13
    assert doc["data"]["label_source"] == "t2_full_data"
    assert doc["data"]["labels"] == fit.T2_LABELS
    assert doc["data"]["n_traces"] == 18
    assert doc["data"]["verify"] is False
    assert set(doc["sigma_traj"]) == {"1", "2", "3"}
    assert doc["calibration"]["m_eff"] is not None
    assert "sha256" in capsys.readouterr().out

    manifest_out = tmp_path / "prior_manifest.json"
    assert (
        fit.main(
            [
                "fit",
                "--study",
                str(STUDY),
                "--out",
                str(manifest_out),
                "--labels",
                "manifest",
                "--max-per-level",
                "3",
                "--no-verify",
                "--reps",
                "20",
            ]
        )
        == 0
    )
    comparison = json.loads(manifest_out.read_text(encoding="utf-8"))
    assert comparison["labels"] == "manifest_levels"
    assert comparison["data"]["labels"]["lvl1"] == pytest.approx(1146.2736, abs=1e-3)

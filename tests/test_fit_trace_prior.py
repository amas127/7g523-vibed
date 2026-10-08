"""M1 tests for the trace S1 prior calibration (``seven523.prior``).

The core maths and the artifact model now live in :mod:`seven523.prior` (the
owner shared with the D1 tool), so the regression tests import it directly.
The CLI contract is exercised by loading ``tools/fit_trace_prior.py`` by path,
exactly like the command line.

The regression tests use a fixed subset of the checked-out ``traces/study``
ladder (first 10 games of each of the 4 T2 rung levels) under **both** label
maps that matter:

* ``t2`` (default, C-6/T13): the 2026-09-27 w2m/T23 rerating (rules
  revision 3) — one shared homoscedastic probit-MLE table over the 10-level
  pool, published by ``tools/refit_mle.py`` and adopted into the study
  manifest (ADR-0013 decision 3);
* ``manifest`` (comparison only): the same study's manifest levels.  The
  2026-09-29 ``search_leafq`` publication re-fit the raw ladder in one joint
  probit-MLE (ADR-0013), so the raw levels moved (≤8.2) and the manifest now
  carries an 11th level; T2 stays the shipped prior's calibration table.

If the S1 extractor, the ridge design, the de-shrink or the ``σ_traj`` table
drifts, the locked numbers fail.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from seven523 import prior
from seven523.elo import Prior
from seven523.rules import DEFAULT_RULES, rules_id

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"

_spec = importlib.util.spec_from_file_location(
    "fit_trace_prior", TOOLS / "fit_trace_prior.py"
)
assert _spec and _spec.loader
fit = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = fit
_spec.loader.exec_module(fit)

STUDY = ROOT / "traces" / "study"
PER_LEVEL = 10

#: The 2026-10-08 published 13-level joint probit-MLE table (P5a: 18,000 raw
#: + 16,000 bres221 7M/9M beta=1 search games; ADR-0013 single fit,
#: ``--keep-rungs`` keeps the rungs at ``lvl1``/``lvl4``).  The raw levels are
#: a fresh fit, not the 2026-09-29 table; ``search_leafq`` keeps its published
#: 2026-09-29 value (carried over, not re-measured in P5a), and the web-local
#: ``head32ln5m``/``bres221_7M`` raw entries stay outside this measured
#: contract.  Search rungs are prior-off at placement time (μ≳260 lies beyond
#: the raw trace-prior domain).
REFIT_MLE_LEVELS: dict[str, float] = {
    "random": 0.0,
    "lvl1": 87.37339532345482,
    "lvl2": 109.37677605377907,
    "lvl3": 125.81194449020524,
    "lvl4": 175.5121569659044,
    "ws_s2": 183.3932271777491,
    "pself_s2": 182.81256800253172,
    "w2m_low": 185.5821533622947,
    "w2m_plain": 185.8243222782719,
    "w2m_ctl": 187.0353134733631,
    "search_leafq": 260.0309697237381,
    "bres221_7M_b1": 262.136097329836,
    "bres221_9M_b1": 265.69792886764594,
}

# ``traces/`` is gitignored (data artifact); a fresh checkout without the ladder
# skips instead of erroring.  The T17 re-record (2026-09-26, rules revision 3,
# 4000 traces) is the shipped study; ``legacy_study`` remains so a ladder that
# still stamps a foreign rules identity skips through the revision-3 replay
# gate instead of failing inside the fixture.  The 2026-09-29 asset cleanup
# keeps only ``traces/study/manifest.json``, so the corpus regressions skip
# unless the ladder files are checked out again.
LADDER_PRESENT = any(
    path.is_file() and path.name != "manifest.json" for path in STUDY.rglob("*")
)
study_corpus = pytest.mark.skipif(
    not (STUDY.is_dir() and LADDER_PRESENT),
    reason="traces/study ladder corpus is not present (manifest-only checkout)",
)
manifest_present = pytest.mark.skipif(
    not (STUDY / "manifest.json").is_file(),
    reason="traces/study manifest is not present",
)


def _manifest_is_legacy(manifest_path: Path) -> bool:
    """True when a study manifest stamps a rules identity other than the engine's.

    The revision-3 撬底 change (ADR-0014) refuses every v2 trace, so the
    fixture-based regressions below cannot run against the shipped T15 ladder.
    ``plans.md`` T17 re-records the study and republishes the manifest; the
    refusal itself stays covered by ``tests/test_trace.py`` and
    ``tests/test_placement.py``.  A malformed manifest is *not* legacy: the
    requested fixtures should fail loudly instead of skipping over corruption.
    """
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except ValueError:
        return False
    if "rules_id" not in manifest:
        return True  # unidentified data cannot be proven current
    return manifest["rules_id"] != rules_id(DEFAULT_RULES)


#: The shipped ``traces/study`` is the T17 revision-3 re-record; the guard
#: keeps the three fixture regressions skipped if a legacy ladder is shipped.
LEGACY_STUDY = _manifest_is_legacy(STUDY / "manifest.json")
legacy_study = pytest.mark.skipif(
    LEGACY_STUDY,
    reason=(
        "traces/study does not stamp the current rules identity; the T17 "
        "re-record (plans.md) has not landed"
    ),
)


def test_manifest_legacy_detection_follows_the_rules_id(tmp_path):
    # The revision-3 replay gate makes every v2 study trace unusable, so the
    # skip guard must key on the manifest identity, not on the traces merely
    # being present, or the three regressions below silently run on refused
    # data.  Missing manifests stay non-legacy so fixture errors surface.
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"rules_id": "fbd43015d526ee72"}), encoding="utf-8")
    assert _manifest_is_legacy(manifest)
    manifest.write_text(json.dumps({}), encoding="utf-8")
    assert _manifest_is_legacy(manifest)
    manifest.write_text(
        json.dumps({"rules_id": rules_id(DEFAULT_RULES)}), encoding="utf-8"
    )
    assert not _manifest_is_legacy(manifest)
    assert not _manifest_is_legacy(tmp_path / "missing.json")


@pytest.fixture(scope="module")
def ladder():
    rows, stats = prior.load_rows(
        STUDY, prior.T2_LABELS, verify=True, max_per_level=PER_LEVEL
    )
    return rows, prior.T2_LABELS, stats


@pytest.fixture(scope="module")
def manifest_ladder():
    # The manifest label map re-reads the same traces the ``ladder`` fixture
    # already replayed, and only feeds the label-map test below, so skip the
    # second verification pass.
    manifest = {
        str(name): float(value)
        for name, value in json.loads(
            (STUDY / "manifest.json").read_text(encoding="utf-8")
        )["levels"].items()
    }
    rows, stats = prior.load_rows(
        STUDY, manifest, verify=False, max_per_level=PER_LEVEL
    )
    return rows, manifest, stats


@study_corpus
@legacy_study
def test_label_map_drives_both_subject_and_opponent_features(
    ladder, manifest_ladder
):
    t2_rows, _t2_labels, _stats = ladder
    manifest_rows, manifest_labels, _stats = manifest_ladder

    # T2 mode: both the subject target and the opponent feature use T2 numbers.
    # The T17 study has no random-subject traces (the gauge enters as an
    # opponent only) and now includes cross-rung games, so the opponent
    # feature covers the gauge and the rung labels.
    rung_ids = ("lvl1", "lvl2", "lvl3", "lvl4")
    assert {row["level_elo_ref"] for row in t2_rows} == {
        prior.T2_LABELS[id_] for id_ in rung_ids
    }
    opponents = {row["opponent_elo_ref"] for row in t2_rows}
    assert prior.T2_LABELS["random"] in opponents
    assert opponents <= set(prior.T2_LABELS.values())
    # Manifest mode: the same rows carry the current manifest levels.  The
    # 2026-09-29 search-rung joint refit moved the raw labels and added
    # ``search_leafq`` (no trace corpus yet), so compare the raw subset.
    assert set(manifest_labels) == set(prior.T2_LABELS) | {"search_leafq"}
    assert manifest_labels == REFIT_MLE_LEVELS
    # Every T17 study trace is subject-vs-random, so the opponent feature is
    # the gauge in both maps; the 2026-09-29 refit left random=0.0.
    manifest_opponents = {row["opponent_elo_ref"] for row in manifest_rows}
    assert manifest_opponents == opponents == {0.0}


@manifest_present
def test_t2_labels_are_the_prior_calibration_table_and_the_manifest_adds_search():
    # T2 is the 2026-09-27 w2m/T23 rerating the shipped prior artifact was
    # calibrated against.  The 2026-09-29 search-rung publication re-fit the
    # raw ladder in the same joint MLE (ADR-0013), so the manifest carries the
    # new raw values plus ``search_leafq``; T2 stays frozen until the prior is
    # re-fitted against the current table (search-config-plan.md §4.3/§5).
    # Never the superseded T17 table (82.75/106.90/147.71/185.34), the T15
    # revision-2, online-Plackett-Luce or BT-MAP numbers.
    assert prior.T2_LABELS == {
        "random": 0.0,
        "lvl1": 84.68039955139497,
        "lvl2": 113.59840294812228,
        "lvl3": 134.39575022531008,
        "lvl4": 187.7188102655843,
        "pself_s2": 187.27092280701837,
        "w2m_ctl": 191.49039136761593,
        "w2m_low": 190.02147844096504,
        "w2m_plain": 190.2826214288363,
        "ws_s2": 185.96380656765228,
    }
    manifest = json.loads((STUDY / "manifest.json").read_text(encoding="utf-8"))
    # Web-local, unmeasured additions (head32ln5m …) are deliberately outside
    # the published MLE contract; the measured table must still match it.
    web_local = {
        str(entry["id"])
        for entry in manifest["subjects"]
        if entry.get("web_local")
    }
    measured_levels = {
        id_: mu
        for id_, mu in manifest["levels"].items()
        if id_ not in web_local
    }
    assert measured_levels == REFIT_MLE_LEVELS
    assert manifest["estimator"]["kind"] == "probit-mle"
    raw = {id_: manifest["levels"][id_] for id_ in prior.T2_LABELS}
    assert raw != prior.T2_LABELS  # a cross-fit refit, not the old table
    # Both refits moved the raw ladder while staying in the same neighborhood;
    # the 2026-10-08 joint fit moved lvl4 the most (-12.2).
    assert max(abs(raw[id_] - prior.T2_LABELS[id_]) for id_ in raw) <= 15.0
    specs = {entry["id"]: entry["spec"] for entry in manifest["subjects"]}
    assert specs["lvl1"].startswith("ckpt:")
    assert specs["random"] == "random"
    assert specs["search_leafq"] == "rolloutt:runs/ei2_value_t5/t_leafq/critic.pt"


PRIOR_ARTIFACT = ROOT / "artifacts" / "human-elo" / "prior.json"


@manifest_present
@pytest.mark.skipif(
    not PRIOR_ARTIFACT.exists(), reason="M1 prior artifact is not present"
)
def test_shipped_prior_is_calibrated_to_the_published_mle_table():
    # The shipped prior must be derived from the same adopted table as the
    # manifest and T2: a re-fit onto a retired gauge (or a stale artifact) must
    # not slip through unnoticed by the placement-owned shipped-prior test.
    artifact = prior.TracePrior.load(PRIOR_ARTIFACT)
    manifest = json.loads((STUDY / "manifest.json").read_text(encoding="utf-8"))
    assert artifact.labels == prior.T2_LABELS
    # The manifest refit (search rung joint MLE) moved the raw levels; the
    # shipped prior still carries the T2 calibration it was fit on.
    assert {id_: manifest["levels"][id_] for id_ in prior.T2_LABELS} == {
        id_: REFIT_MLE_LEVELS[id_] for id_ in prior.T2_LABELS
    }
    assert artifact.doc["labels"] == "t2_full_data"
    # The 2026-09-27 v2/R1 publication (independent bank2 confirmation): the
    # shipped artifact must stay the quadratic-pairwise + cell-penalty model,
    # not silently fall back to the v1 linear design.
    assert artifact.doc["kind"] == "trace-s1-poly2-ridge-deshrunk"
    assert artifact.doc["model"]["expansion"] == prior.EXPANSION_QUADRATIC_PAIRWISE
    assert artifact.doc["hyperparams"]["cell_penalty"] == 3.0
    # The adapter must read sigma from the artifact's own sigma_traj table.
    for n in (1, 5, 10, 20):
        assert artifact.prior_for_session(prior.T2_LABELS["lvl1"], n).sigma == (
            pytest.approx(float(artifact.doc["sigma_traj"][str(n)]))
        )
    # Gauge invariant: humans start above the strongest published bot level.
    assert artifact.cold_start.mu > manifest["levels"]["lvl4"]
    assert artifact.cold_start.sigma > 0.0


@study_corpus
@legacy_study
def test_prior_schema_and_round_trip(ladder, tmp_path):
    rows, labels, _stats = ladder
    doc = prior.build_prior(rows, labels, reps=100)
    path = prior.save_prior(doc, tmp_path / "prior.json", created_at="2026-09-25T00:00:00")
    loaded = prior.load_prior(path)

    assert loaded["schema"] == prior.SCHEMA
    assert loaded["version"] == prior.VERSION
    assert loaded["created_at"] == "2026-09-25T00:00:00"
    assert loaded["labels"] == "t2_full_data"
    assert loaded["hyperparams"]["label_source"] == "t2_full_data"
    assert set(loaded["prior"]) == {"mean", "sd"}
    assert Prior(
        float(loaded["prior"]["mean"]), float(loaded["prior"]["sd"])
    ) == Prior(prior.COLD_START_PRIOR["mean"], prior.COLD_START_PRIOR["sd"])
    assert "NaN" not in path.read_text(encoding="utf-8")

    model = loaded["model"]
    assert len(model["features"]) == len(prior.MODEL_FEATURES) == 15
    assert len(model["mean"]) == len(model["scale"]) == len(model["coefficients"]) == 17
    assert model["anchor"] == {
        "name": prior.ANCHOR_FEATURE,
        "center": prior.ANCHOR_CENTER,
        "scale": prior.ANCHOR_SCALE,
    }

    # the serialized model reproduces the in-memory predictions
    for row in rows[:5]:
        assert prior.predict_elo(loaded, row) == pytest.approx(
            prior.predict_elo(doc, row), abs=1e-9
        )

    mu = prior.session_mean(loaded, rows[:3])
    session = prior.prior_for_session(loaded, mu, 10)
    assert session.mu == pytest.approx(mu)
    assert session.sigma == pytest.approx(loaded["sigma_traj"]["10"])
    cold = prior.prior_for_session(loaded, 1234.0, 0)
    assert (cold.mu, cold.sigma) == (
        prior.COLD_START_PRIOR["mean"],
        prior.COLD_START_PRIOR["sd"],
    )


@study_corpus
@legacy_study
def test_build_prior_is_deterministic(ladder):
    rows, labels, _stats = ladder
    first = prior.build_prior(rows, labels, reps=100)
    second = prior.build_prior(rows, labels, reps=100)
    assert json.dumps(first, ensure_ascii=False) == json.dumps(second, ensure_ascii=False)


@study_corpus
@legacy_study
def test_build_prior_default_flags_are_the_v1_model(ladder):
    # The v2 parameters are opt-in: the defaults must stay the shipped linear
    # document so every existing artifact and training run is untouched.
    rows, labels, _stats = ladder
    default = prior.build_prior(rows, labels, reps=20)
    explicit = prior.build_prior(
        rows, labels, reps=20, expansion="linear", cell_penalty=0.0
    )
    assert json.dumps(default, ensure_ascii=False) == json.dumps(
        explicit, ensure_ascii=False
    )
    assert default["version"] == prior.VERSION
    assert default["model"]["expansion"] == "linear"
    assert default["hyperparams"]["cell_penalty"] == 0.0


@study_corpus
@legacy_study
def test_v2_prior_schema_and_round_trip(ladder, tmp_path):
    rows, labels, _stats = ladder
    doc = prior.build_prior(
        rows,
        labels,
        reps=20,
        expansion=prior.EXPANSION_QUADRATIC_PAIRWISE,
        cell_penalty=3.0,
    )
    assert doc["version"] == prior.VERSION_QUADRATIC
    assert doc["kind"] == "trace-s1-poly2-ridge-deshrunk"
    model = doc["model"]
    assert model["expansion"] == prior.EXPANSION_QUADRATIC_PAIRWISE
    assert len(model["features"]) == len(prior.MODEL_FEATURES) == 15
    assert (
        len(model["mean"])
        == len(model["scale"])
        == len(model["coefficients"])
        == 152
    )
    assert doc["hyperparams"]["cell_penalty"] == 3.0
    assert doc["calibration"]["cell_drift_var"] >= 0.0
    assert doc["calibration"]["drift_sd"] >= 0.0

    path = prior.save_prior(
        doc, tmp_path / "prior_v2.json", created_at="2026-09-27T00:00:00"
    )
    loaded = prior.load_prior(path)
    assert loaded["model"] == model
    for row in rows[:5]:
        assert prior.predict_elo(loaded, row) == pytest.approx(
            prior.predict_elo(doc, row), abs=1e-9
        )


@study_corpus
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
    assert doc["schema"] == prior.SCHEMA
    assert doc["labels"] == "t2_full_data"  # default target per C-6/T13
    assert doc["data"]["label_source"] == "t2_full_data"
    assert doc["data"]["labels"] == prior.T2_LABELS
    assert doc["data"]["n_traces"] == 12
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
    # The manifest label map is the current joint-fit table (search rung
    # publication), not the frozen T2 prior-calibration table.
    assert comparison["data"]["labels"] == REFIT_MLE_LEVELS
    assert comparison["data"]["labels"]["lvl1"] == pytest.approx(
        REFIT_MLE_LEVELS["lvl1"], abs=1e-9
    )

    v2_out = tmp_path / "prior_v2.json"
    assert (
        fit.main(
            [
                "fit",
                "--study",
                str(STUDY),
                "--out",
                str(v2_out),
                "--max-per-level",
                "3",
                "--no-verify",
                "--reps",
                "20",
                "--expansion",
                "quadratic_pairwise",
                "--cell-penalty",
                "3",
                "--created-at",
                "2026-09-27T00:00:00",
            ]
        )
        == 0
    )
    v2 = json.loads(v2_out.read_text(encoding="utf-8"))
    assert v2["version"] == prior.VERSION_QUADRATIC
    assert v2["kind"] == "trace-s1-poly2-ridge-deshrunk"
    assert v2["model"]["expansion"] == "quadratic_pairwise"
    assert len(v2["model"]["coefficients"]) == 152
    assert v2["hyperparams"]["cell_penalty"] == 3.0
    assert v2["data"]["expansion"] == "quadratic_pairwise"
    assert v2["data"]["cell_penalty"] == 3.0

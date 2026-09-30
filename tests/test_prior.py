"""Core tests for :mod:`seven523.prior`, the shared trace S1 prior owner.

These cover the maths that moved out of the tools: the ridge helpers, the
session mean with explicit opponent ratings (the live placement shape) and the
online artifact wrapper.  The study-scale calibration regressions stay in
``tests/test_fit_trace_prior.py`` (they need the checked-out ladder).
"""
from __future__ import annotations

import numpy as np
import pytest

from seven523 import prior
from seven523.elo import Prior


def _synthetic_rows(levels=("a", "b", "c"), per_level=40, seed=0):
    """Study-shaped rows: a level signal, two opponent cells, noise features."""
    rng = np.random.default_rng(seed)
    truth = {"a": 0.0, "b": 100.0, "c": 200.0, "d": 300.0}
    rows = []
    for level in levels:
        for index in range(per_level):
            row = {feature: float(rng.normal()) for feature in prior.MODEL_FEATURES}
            row[prior.ANCHOR_FEATURE] = float(rng.normal(scale=50.0))
            row["level_id"] = level
            row["level_elo_ref"] = truth[level]
            row["opponent_id"] = "opp_a" if index % 2 else "opp_b"
            rows.append(row)
    return rows


def _legacy_linear_design(rows, fill):
    """The v1 17-column construction, rebuilt independently of the owner."""
    index = prior.MODEL_FEATURES.index(prior.INTERACTION_FEATURE)
    matrix = np.empty((len(rows), 17))
    for position, row in enumerate(rows):
        values = np.array(
            [
                float(row[feature])
                if prior._finite(row.get(feature))
                else float(fill[feature])
                for feature in prior.MODEL_FEATURES
            ]
        )
        anchor = (
            float(row[prior.ANCHOR_FEATURE]) - prior.ANCHOR_CENTER
        ) / prior.ANCHOR_SCALE
        matrix[position, :15] = values
        matrix[position, 15] = anchor
        matrix[position, 16] = values[index] * anchor
    return matrix


def test_design_matrix_linear_is_bitwise_the_v1_construction():
    rows = _synthetic_rows(levels=("a", "b"), per_level=5)
    rows[0][prior.MODEL_FEATURES[0]] = float("nan")  # the fill must kick in
    fill = prior.fill_values(rows)
    assert np.array_equal(
        prior.design_matrix(rows, fill), _legacy_linear_design(rows, fill)
    )


def test_design_matrix_quadratic_pairwise_column_order():
    rows = _synthetic_rows(levels=("a",), per_level=4)
    fill = prior.fill_values(rows)
    matrix = prior.design_matrix(
        rows, fill, expansion=prior.EXPANSION_QUADRATIC_PAIRWISE
    )
    features = len(prior.MODEL_FEATURES)
    assert matrix.shape == (len(rows), 152)
    for position, row in enumerate(rows):
        values = np.array([row[feature] for feature in prior.MODEL_FEATURES])
        anchor = (
            float(row[prior.ANCHOR_FEATURE]) - prior.ANCHOR_CENTER
        ) / prior.ANCHOR_SCALE
        assert np.array_equal(matrix[position, :features], values)
        assert matrix[position, features] == anchor
        assert matrix[position, features + 1] == anchor**2
        start = features + 2
        assert np.array_equal(
            matrix[position, start : start + features], values * anchor
        )
        start += features
        assert np.array_equal(matrix[position, start : start + features], values**2)
        start += features
        pairs = np.array(
            [
                values[left] * values[right]
                for left in range(features)
                for right in range(left + 1, features)
            ]
        )
        assert np.array_equal(matrix[position, start:], pairs)


def test_design_matrix_rejects_an_unknown_expansion():
    rows = _synthetic_rows(levels=("a",), per_level=2)
    with pytest.raises(ValueError, match="expansion"):
        prior.design_matrix(rows, prior.fill_values(rows), expansion="cubic")


def test_ridge_fit_cell_penalty_zero_is_bitwise_the_v1_solve():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(30, 4))
    y = x @ np.array([1.0, -2.0, 0.5, 3.0]) + rng.normal(scale=0.1, size=30)
    cells = np.array(["a|a", "a|b", "b|b"] * 10)
    plain = prior.ridge_fit(x, y, 30.0)
    zero = prior.ridge_fit(x, y, 30.0, cells=cells, cell_penalty=0.0)
    assert all(
        np.array_equal(np.asarray(left), np.asarray(right))
        for left, right in zip(plain, zero)
    )


def test_cell_penalty_pulls_the_training_cell_mean_residuals_toward_zero():
    rng = np.random.default_rng(1)
    cells = np.array(["a|a"] * 20 + ["a|b"] * 20)
    x = rng.normal(size=(40, 2))
    y = 0.5 * x[:, 0] + np.where(cells == "a|a", 5.0, -5.0)

    def cell_means(model):
        residual = y - prior.ridge_predict(model, x)
        return np.array([residual[cells == cell].mean() for cell in ("a|a", "a|b")])

    plain = cell_means(prior.ridge_fit(x, y, 1.0))
    penalized = cell_means(
        prior.ridge_fit(x, y, 1.0, cells=cells, cell_penalty=10.0)
    )
    assert np.abs(penalized).max() < np.abs(plain).max()


def test_ridge_fit_rejects_bad_cell_penalty_arguments():
    x = np.zeros((4, 2))
    y = np.zeros(4)
    with pytest.raises(ValueError, match="non-negative"):
        prior.ridge_fit(x, y, 1.0, cells=np.array(["a"] * 4), cell_penalty=-1.0)
    with pytest.raises(ValueError, match="requires"):
        prior.ridge_fit(x, y, 1.0, cell_penalty=1.0)


def test_build_prior_v2_artifact_round_trips_and_keeps_v1_readable(tmp_path):
    rows = _synthetic_rows()
    levels = {"a": 0.0, "b": 100.0, "c": 200.0}
    v1 = prior.build_prior(rows, levels, reps=20)
    v2 = prior.build_prior(
        rows,
        levels,
        reps=20,
        expansion=prior.EXPANSION_QUADRATIC_PAIRWISE,
        cell_penalty=3.0,
    )

    assert v1["version"] == prior.VERSION
    assert v1["kind"] == "trace-s1-ridge-deshrunk"
    assert v1["model"]["expansion"] == prior.EXPANSION_LINEAR
    assert len(v1["model"]["coefficients"]) == 17
    assert v1["hyperparams"]["cell_penalty"] == 0.0

    assert v2["version"] == prior.VERSION_QUADRATIC
    assert v2["kind"] == "trace-s1-poly2-ridge-deshrunk"
    assert v2["model"]["expansion"] == prior.EXPANSION_QUADRATIC_PAIRWISE
    assert len(v2["model"]["coefficients"]) == 152
    assert len(v2["model"]["mean"]) == len(v2["model"]["scale"]) == 152
    assert v2["hyperparams"]["cell_penalty"] == 3.0
    assert v2["calibration"]["cell_drift_var"] >= 0.0
    assert v2["calibration"]["drift_sd"] >= 0.0

    for doc in (v1, v2):
        path = prior.save_prior(
            doc,
            tmp_path / f"prior_{doc['model']['expansion']}.json",
            created_at="2026-09-27T00:00:00",
        )
        loaded = prior.load_prior(path)
        assert loaded["model"]["expansion"] == doc["model"]["expansion"]
        for row in rows[:5]:
            assert prior.predict_elo(loaded, row) == pytest.approx(
                prior.predict_elo(doc, row), abs=1e-12
            )

    # A legacy v1 artifact without ``model.expansion`` keeps the linear path.
    legacy = {
        **v1,
        "model": {k: val for k, val in v1["model"].items() if k != "expansion"},
    }
    for row in rows[:5]:
        assert prior.predict_elo(legacy, row) == prior.predict_elo(v1, row)


def test_predict_elo_v2_rebuilds_the_documented_design_and_deshrink():
    rows = _synthetic_rows(levels=("a", "b"), per_level=30)
    levels = {"a": 0.0, "b": 100.0}
    doc = prior.build_prior(
        rows,
        levels,
        reps=5,
        expansion=prior.EXPANSION_QUADRATIC_PAIRWISE,
        cell_penalty=1.0,
    )
    model = doc["model"]
    features = len(model["features"])
    for row in rows[:3]:
        values = np.array([row[feature] for feature in model["features"]])
        anchor = (
            float(row[prior.ANCHOR_FEATURE]) - float(model["anchor"]["center"])
        ) / float(model["anchor"]["scale"])
        pairs = np.array(
            [
                values[left] * values[right]
                for left in range(features)
                for right in range(left + 1, features)
            ]
        )
        design = np.concatenate(
            [values, [anchor, anchor**2], values * anchor, values**2, pairs]
        )
        raw = float(
            ((design - model["mean"]) / model["scale"]) @ model["coefficients"]
            + model["intercept"]
        )
        expected = doc["deshrink"]["a"] + doc["deshrink"]["b"] * raw
        assert prior.predict_elo(doc, row) == pytest.approx(expected, abs=1e-9)


def test_build_prior_rejects_unknown_expansion_and_negative_cell_penalty():
    rows = _synthetic_rows(levels=("a", "b"), per_level=5)
    levels = {"a": 0.0, "b": 100.0}
    with pytest.raises(ValueError, match="expansion"):
        prior.build_prior(rows, levels, reps=5, expansion="cubic")
    with pytest.raises(ValueError, match="cell_penalty"):
        prior.build_prior(rows, levels, reps=5, cell_penalty=-1.0)


def test_ridge_fit_and_predict_recover_a_linear_signal():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 3))
    coefficients = np.array([2.0, -1.0, 0.5])
    y = 5.0 + x @ coefficients
    model = prior.ridge_fit(x, y, 1e-6)
    predictions = prior.ridge_predict(model, x)
    assert np.allclose(predictions, y, atol=1e-3)


def test_ridge_fit_keeps_constant_columns_finite():
    x = np.column_stack([np.ones(20), np.linspace(0.0, 1.0, 20)])
    y = 3.0 + 2.0 * x[:, 1]
    model = prior.ridge_fit(x, y, 1.0)
    assert np.all(np.isfinite(model[1]))
    assert np.all(np.isfinite(prior.ridge_predict(model, x)))


def test_session_mean_averages_predictions_and_reads_explicit_opponents(monkeypatch):
    rows = [{"x": 1.0}, {"x": 3.0}]
    seen: list[float | None] = []

    def fake_predict(doc, row, opponent_elo=None):
        seen.append(opponent_elo)
        if opponent_elo is None:
            return 1000.0 + float(row["x"])
        return float(row["x"]) + float(opponent_elo)

    monkeypatch.setattr(prior, "predict_elo", fake_predict)

    assert prior.session_mean({}, rows) == pytest.approx(1002.0)
    assert prior.session_mean(
        {}, rows, opponent_elos=[1100.0, 1200.0]
    ) == pytest.approx(1152.0)
    assert seen == [None, None, 1100.0, 1200.0]
    with pytest.raises(ValueError):
        prior.session_mean({}, [])


def test_trace_prior_reads_labels_meta_and_cold_start():
    doc = {
        "schema": prior.SCHEMA,
        "version": prior.VERSION,
        "kind": "trace-s1-ridge-deshrunk",
        "prior": {"mean": 1500.0, "sd": 300.0},
        "labels": "t2_full_data",
        "created_at": "2026-09-25T00:00:00",
        "scheme": "lolo",
        "deshrink": {"a": -890.92, "b": 1.7122, "applied": True},
        "data": {"labels": {"random": 1026.9, "lvl1": 1128.6}},
    }
    model = prior.TracePrior(doc, path="prior.json")
    assert model.cold_start == Prior(1500.0, 300.0)
    assert model.labels == {"random": 1026.9, "lvl1": 1128.6}
    assert model.meta() == {
        "path": "prior.json",
        "schema": prior.SCHEMA,
        "version": prior.VERSION,
        "kind": "trace-s1-ridge-deshrunk",
        "labels": "t2_full_data",
        "created_at": "2026-09-25T00:00:00",
        "scheme": "lolo",
        "deshrink": {"a": -890.92, "b": 1.7122, "applied": True},
    }


def test_trace_prior_without_a_prior_object_falls_back_to_cold_start():
    model = prior.TracePrior({"schema": prior.SCHEMA})
    assert model.cold_start == Prior(
        prior.COLD_START_PRIOR["mean"], prior.COLD_START_PRIOR["sd"]
    )


def test_load_prior_rejects_a_foreign_schema(tmp_path):
    path = tmp_path / "prior.json"
    path.write_text('{"schema": "something-else"}', encoding="utf-8")
    with pytest.raises(ValueError, match="trace-prior"):
        prior.load_prior(path)

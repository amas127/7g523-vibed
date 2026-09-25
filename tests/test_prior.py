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
        "data": {"labels": {"random": 1026.9, "greedy": 1314.4}},
    }
    model = prior.TracePrior(doc, path="prior.json")
    assert model.cold_start == Prior(1500.0, 300.0)
    assert model.labels == {"random": 1026.9, "greedy": 1314.4}
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
    assert model.cold_start == Prior(1500.0, 300.0)


def test_load_prior_rejects_a_foreign_schema(tmp_path):
    path = tmp_path / "prior.json"
    path.write_text('{"schema": "something-else"}', encoding="utf-8")
    with pytest.raises(ValueError, match="trace-prior"):
        prior.load_prior(path)

"""Trace S1 prior: feature extraction, ridge calibration and the artifact model.

This module is the single owner of the shared 轨迹 S1 core that used to be
split between the D1 signal-measurement tool and the M1 prior-calibration
tool:

* :func:`extract_features` — the per-game S1 feature row for a trace's subject
  seat, shared by signal measurement and prior calibration;
* :func:`ridge_fit` / :func:`ridge_predict` — the standardized ridge helpers;
* the study scanning helpers (:func:`trace_paths`, :func:`extract_row`,
  :func:`load_rows`) and the label map (:func:`resolve_levels`);
* :data:`SCHEMA` and :func:`build_prior` / :func:`load_prior` /
  :func:`save_prior` — the ``prior.json`` artifact schema and training;
* :func:`predict_elo` / :func:`session_mean` / :func:`prior_for_session` — the
  online model, with :class:`TracePrior` the artifact as used by placement.

The tools import from here and stay thin CLIs; :mod:`seven523.placement`
imports the same owner instead of loading a tool by path.  The artifact schema
and every numeric convention are unchanged by the extraction.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .actions import catalog_for
from .combos import ComboKind
from .elo import DEFAULT_ELO_SCALE, Prior
from .play import replay_trace
from .trace import load_trace, parse_player_label, rules_from_json

__all__ = [
    "ANCHOR_CENTER",
    "ANCHOR_FEATURE",
    "ANCHOR_SCALE",
    "BETA",
    "CLEAN_FEATURES",
    "COLD_START_PRIOR",
    "DEFAULT_ALPHA",
    "DEFAULT_REPS",
    "DEFAULT_SCHEME",
    "DEFAULT_SEED",
    "INTERACTION_FEATURE",
    "MAX_SESSION",
    "MODEL_FEATURES",
    "PACE_FEATURES",
    "SCHEMA",
    "SINGLE_GAME_SE",
    "T2_LABELS",
    "VERSION",
    "TracePrior",
    "build_prior",
    "data_fingerprint",
    "design_matrix",
    "expand",
    "extract_features",
    "extract_row",
    "fill_values",
    "load_prior",
    "load_rows",
    "opponent_of",
    "predict_elo",
    "prior_for_session",
    "resolve_levels",
    "ridge_fit",
    "ridge_predict",
    "save_prior",
    "session_mean",
    "session_prior_sd",
    "silent",
    "trace_paths",
]

#: 400-point Elo scale and the SE of one single-game Bernoulli observation at
#: p=0.5 (the plan's ``347``); ``m_eff = (SINGLE_GAME_SE / s) ** 2``.
BETA = math.log(10.0) / DEFAULT_ELO_SCALE
SINGLE_GAME_SE = 1.0 / (BETA * math.sqrt(0.25))

BOMB_KINDS: frozenset[ComboKind] = frozenset(
    {ComboKind.SMALL_BOMB, ComboKind.BIG_BOMB}
)

#: Artifact schema tag/version; consumers reject anything else.
SCHEMA = "seven523.trace-prior"
VERSION = 1

#: The study's ``CLEAN + PACE`` S1 feature set, matching the research pipeline
#: (``runs/human_elo_10g/calibrate.py``; HR §5.2 "12–17 维").
CLEAN_FEATURES: tuple[str, ...] = (
    "trick_win_rate",
    "trick_point_share",
    "lead_rate",
    "pass_rate",
    "early_trick_win_rate",
    "mid_trick_win_rate",
    "late_trick_win_rate",
    "mean_points_per_won_trick",
    "bomb_rate",
    "first_trick_won",
    "tricks_won",
    "trick_points_won",
    "dug_won",
)
PACE_FEATURES: tuple[str, ...] = ("decisions_per_trick", "tricks_total")
MODEL_FEATURES: tuple[str, ...] = CLEAN_FEATURES + PACE_FEATURES

#: HR §5.2 requires the opponent strength plus the ``trick_win_rate``
#: interaction so a rung change is not read as a skill change.  The fixed
#: center/scale are the research design constants (``calibrate.py``); only the
#: interaction's centering is not absorbed by the ridge standardization.
ANCHOR_FEATURE = "opponent_elo_ref"
INTERACTION_FEATURE = "trick_win_rate"
ANCHOR_CENTER = 1230.0
ANCHOR_SCALE = 300.0

#: HR §5.2 recommends ``α ≈ 30``; the session table is built for n=1..20 so
#: ``placement`` never has to interpolate.  ``reps=400`` is the research
#: simulation's default (the one behind the published 99/87/82 table).
DEFAULT_ALPHA = 30.0
DEFAULT_SCHEME = "lolo"
DEFAULT_REPS = 400
DEFAULT_SEED = 0
MAX_SESSION = 20

#: Cold start (n=0): the research's first-game prior (HR §5.4).
COLD_START_PRIOR: dict[str, float] = {"mean": 1500.0, "sd": 300.0}

#: HR §1 T2 targets: full-data estimates of each study level under the shipped
#: estimator (anchors pinned random=1000/greedy=1315, margin=(0.143, 44.5),
#: neutral prior).  The default calibration target per C-6/T13; re-derived from
#: ``traces/study`` and matching HR §1 to 0.1 Elo.  ``--level NAME=ELO``
#: refreshes them for a different study.
T2_LABELS: dict[str, float] = {
    "random": 1026.9,
    "greedy": 1314.4,
    "lvl1": 1128.6,
    "lvl2": 1232.1,
    "lvl3": 1356.1,
    "lvl4": 1447.8,
}


def silent(*_args: object, **_kwargs: object) -> None:
    """print_fn for the reuse of ``play_game`` / ``replay_trace`` without noise."""


def expand(values: Sequence[str] | None) -> list[str]:
    """Flatten repeated flags and comma-separated lists into one list."""
    out: list[str] = []
    for value in values or ():
        out.extend(part.strip() for part in value.split(",") if part.strip())
    return out


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


# -- S1 features -------------------------------------------------------------


def extract_features(trace: dict[str, Any], *, verify: bool = True) -> dict[str, Any]:
    """The S1 per-game feature row for the trace's subject (``human_seat``) seat.

    Everything is read from the recorded steps; ``verify`` first re-runs the
    trace through :func:`seven523.play.replay_trace`, the same strict path as
    ``7g523-play --replay``, so a corrupted trace fails instead of feeding the
    calibration silently.
    """
    if verify:
        replay_trace(trace, print_fn=silent)
    rules = rules_from_json(trace["rules"])
    catalog = catalog_for(rules)
    pass_id = len(catalog) - 1
    subject = int(trace["human_seat"])
    steps = trace["steps"]
    final = [int(value) for value in trace["final_scores"]]

    tricks: list[dict[str, Any]] = []
    subject_steps: list[dict[str, Any]] = []
    leads = 0
    leader = int(trace["initial"]["starter"])
    for step in steps:
        seat = int(step["seat"])
        if seat == subject:
            subject_steps.append(step)
            if seat == leader:
                leads += 1
        if step["trick_over"]:
            tricks.append(
                {
                    "winner": int(step["winner"]),
                    "points": int(step["points"]),
                    "dug": bool(step["dug"]),
                }
            )
            leader = int(step["winner"])

    total_tricks = len(tricks)
    tricks_won = sum(1 for trick in tricks if trick["winner"] == subject)
    won_no_dug = [
        trick
        for trick in tricks
        if trick["winner"] == subject and not trick["dug"]
    ]
    trick_points_won = sum(trick["points"] for trick in won_no_dug)
    pot_total = sum(trick["points"] for trick in tricks if not trick["dug"])
    dug_trick = next((trick for trick in tricks if trick["dug"]), None)

    def win_rate(part: Sequence[dict[str, Any]]) -> float:
        if not part:
            return math.nan
        return sum(1 for trick in part if trick["winner"] == subject) / len(part)

    cut1, cut2 = total_tricks // 3, (2 * total_tricks) // 3

    decisions = len(subject_steps)
    passes = sum(int(step["action"]) == pass_id for step in subject_steps)
    bombs = sum(catalog[int(step["action"])].kind in BOMB_KINDS for step in subject_steps)

    own = final[subject]
    others = [score for seat, score in enumerate(final) if seat != subject]
    best_other = max(others)
    result = "win" if own > best_other else ("draw" if own == best_other else "loss")

    def rate(numerator: float, denominator: float) -> float:
        return numerator / denominator if denominator else math.nan

    return {
        "result": result,
        "final_score_own": own,
        "score_diff": own - sum(others) / len(others),
        "tricks_total": total_tricks,
        "tricks_won": tricks_won,
        "trick_win_rate": rate(tricks_won, total_tricks),
        "decisions": decisions,
        "pass_rate": rate(passes, decisions),
        "bomb_rate": rate(bombs, decisions),
        "lead_rate": rate(leads, decisions),
        "trick_points_won": trick_points_won,
        "dug": int(dug_trick is not None),
        "dug_won": int(dug_trick is not None and dug_trick["winner"] == subject),
        "dug_points": int(dug_trick["points"]) if dug_trick is not None else 0,
        "mean_points_per_won_trick": rate(trick_points_won, len(won_no_dug)),
        "trick_point_share": rate(trick_points_won, pot_total),
        "early_trick_win_rate": win_rate(tricks[:cut1]),
        "mid_trick_win_rate": win_rate(tricks[cut1:cut2]),
        "late_trick_win_rate": win_rate(tricks[cut2:]),
        "first_trick_won": int(bool(tricks) and tricks[0]["winner"] == subject),
        "decisions_per_trick": rate(decisions, total_tricks),
    }


# -- study scanning -----------------------------------------------------------


def resolve_levels(
    manifest: Mapping[str, Any],
    overrides: Sequence[str] | None = None,
    *,
    labels: str = "t2",
) -> dict[str, float]:
    """The selected label map plus ``NAME=ELO`` CLI overrides (D1 convention).

    ``labels="t2"`` (default) starts from :data:`T2_LABELS` (HR §1/C-6);
    ``labels="manifest"`` starts from the study manifest's ``levels`` and is a
    comparison mode only.  Overrides win in both modes, and the resulting map
    feeds both the subject labels and the opponent-strength feature.
    """
    if labels not in ("t2", "manifest"):
        raise ValueError(f"labels must be 't2' or 'manifest', got {labels!r}")
    if labels == "t2":
        levels = dict(T2_LABELS)
    else:
        levels = {
            str(name): float(value)
            for name, value in (manifest.get("levels") or {}).items()
        }
    for item in expand(overrides):
        if "=" not in item:
            raise SystemExit(f"--level expects NAME=ELO, got {item!r}")
        name, value = item.split("=", 1)
        levels[name.strip()] = float(value)
    return levels


def trace_paths(study: str | Path) -> list[Path]:
    """Study trace files: ``<study>/<level>/<game>.json`` (flat fallback)."""
    study = Path(study)
    paths = sorted(
        path for path in study.glob("*/*.json") if path.name != "manifest.json"
    )
    if not paths:
        paths = sorted(
            path for path in study.glob("*.json") if path.name != "manifest.json"
        )
    return paths


def opponent_of(
    trace: Mapping[str, Any], subject: int, levels: Mapping[str, float]
) -> tuple[str, float]:
    """Opponent id/Elo of the first non-subject seat known to the manifest."""
    players = trace.get("players", [])
    for seat, label in enumerate(players):
        if seat == subject:
            continue
        _role, opponent_id, _seat = parse_player_label(str(label))
        if opponent_id in levels:
            return opponent_id, float(levels[opponent_id])
    return "unknown", math.nan


def extract_row(
    path: str | Path, levels: Mapping[str, float], *, verify: bool = True
) -> dict[str, Any]:
    """One per-game S1 feature row for ``path``'s subject seat.

    ``level_id`` is the trace's directory name; the level label is looked up in
    ``levels`` (NaN when unknown, so the caller can drop unlabeled rows).
    """
    path = Path(path)
    trace = load_trace(path)
    subject = int(trace["human_seat"])
    opponent_id, opponent_elo = opponent_of(trace, subject, levels)
    level_id = path.parent.name
    return {
        "trace_id": path.name,
        "level_id": level_id,
        "level_elo_ref": float(levels[level_id]) if level_id in levels else math.nan,
        "opponent_id": opponent_id,
        "opponent_elo_ref": opponent_elo,
        "seat": subject,
        "seed": trace.get("seed"),
        **extract_features(trace, verify=verify),
    }


def load_rows(
    study: str | Path,
    levels: Mapping[str, float],
    *,
    verify: bool = True,
    max_per_level: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read every labeled trace under ``study``; returns ``(rows, stats)``.

    Unlabeled rows (unknown level or opponent Elo) and malformed traces are
    counted in ``stats`` instead of aborting the whole calibration.
    """
    rows: list[dict[str, Any]] = []
    games_per_level: dict[str, int] = {}
    failed = 0
    skipped_cap = 0
    for path in trace_paths(study):
        level_id = path.parent.name
        if max_per_level and games_per_level.get(level_id, 0) >= max_per_level:
            skipped_cap += 1
            continue
        try:
            row = extract_row(path, levels, verify=verify)
        except (KeyError, ValueError, TypeError) as exc:
            failed += 1
            print(f"  skip {path}: {exc}", file=sys.stderr)
            continue
        if not _finite(row["level_elo_ref"]) or not _finite(row["opponent_elo_ref"]):
            failed += 1
            continue
        rows.append(row)
        games_per_level[level_id] = games_per_level.get(level_id, 0) + 1
    stats = {
        "failed": failed,
        "skipped_cap": skipped_cap,
        "games_per_level": games_per_level,
    }
    return rows, stats


def data_fingerprint(
    study: str | Path, paths: Sequence[str | Path], manifest: str | Path | None = None
) -> str:
    """SHA-256 over every trace (relative path + bytes) and the manifest bytes."""
    study = Path(study)
    digest = hashlib.sha256()
    for path in sorted(Path(path) for path in paths):
        digest.update(path.relative_to(study).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    manifest_path = Path(manifest) if manifest is not None else study / "manifest.json"
    if manifest_path.exists():
        digest.update(b"manifest.json\0")
        digest.update(hashlib.sha256(manifest_path.read_bytes()).digest())
    return digest.hexdigest()


# -- ridge calibration --------------------------------------------------------


def ridge_fit(
    x: np.ndarray, y: np.ndarray, alpha: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Standardize ``x`` and solve the ridge normal equations for ``y``."""
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    z = (x - mean) / scale
    center = float(y.mean())
    weights = np.linalg.solve(
        z.T @ z + alpha * np.eye(x.shape[1]), z.T @ (y - center)
    )
    return mean, scale, weights, center


def ridge_predict(
    model: tuple[np.ndarray, np.ndarray, np.ndarray, float], x: np.ndarray
) -> np.ndarray:
    mean, scale, weights, center = model
    return ((x - mean) / scale) @ weights + center


def fill_values(
    rows: Sequence[Mapping[str, Any]], features: Sequence[str] = MODEL_FEATURES
) -> dict[str, float]:
    """One deterministic fill per design feature: the pooled median (0 if none).

    A handful of study games never won a non-dug trick, so
    ``mean_points_per_won_trick`` (and rare phase slices) can be NaN.  A single
    pooled median — rather than the research's per-level medians — keeps the
    fit-time and inference-time imputation identical, which matters because a
    human trace has no study level.
    """
    fill: dict[str, float] = {}
    for feature in features:
        values = [float(row[feature]) for row in rows if _finite(row.get(feature))]
        fill[feature] = float(np.median(values)) if values else 0.0
    return fill


def design_matrix(
    rows: Sequence[Mapping[str, Any]], fill: Mapping[str, float]
) -> np.ndarray:
    """``[CLEAN+PACE features, anchor, trick_win_rate×anchor]`` (17 columns)."""
    index = {name: position for position, name in enumerate(MODEL_FEATURES)}
    matrix = np.empty((len(rows), len(MODEL_FEATURES) + 2), dtype=float)
    for position, row in enumerate(rows):
        values = np.array(
            [
                float(row[feature]) if _finite(row.get(feature)) else float(fill[feature])
                for feature in MODEL_FEATURES
            ]
        )
        anchor = (float(row[ANCHOR_FEATURE]) - ANCHOR_CENTER) / ANCHOR_SCALE
        matrix[position, : len(MODEL_FEATURES)] = values
        matrix[position, len(MODEL_FEATURES)] = anchor
        matrix[position, len(MODEL_FEATURES) + 1] = (
            values[index[INTERACTION_FEATURE]] * anchor
        )
    return matrix


def _affine_on_training_levels(
    model: tuple[np.ndarray, np.ndarray, np.ndarray, float],
    x: np.ndarray,
    y: np.ndarray,
    level_values: np.ndarray,
    train: np.ndarray,
) -> tuple[float, float]:
    """Fit ``y_level = a + b · mean(g)`` on the training levels (HR §5.2)."""
    levels = sorted(set(map(str, level_values[train])))
    means = np.array(
        [
            ridge_predict(model, x[train & (level_values == level)]).mean()
            for level in levels
        ]
    )
    labels = np.array(
        [y[train & (level_values == level)].mean() for level in levels]
    )
    # Plain degree-1 least squares; equivalent to np.polyfit without its
    # conditioning rank warning (the raw scale is ~10^3).
    column = np.vstack([means, np.ones_like(means)]).T
    slope, intercept = np.linalg.lstsq(column, labels, rcond=None)[0]
    return float(slope), float(intercept)


def _cv_predictions(
    x: np.ndarray,
    y: np.ndarray,
    level_values: np.ndarray,
    groups: np.ndarray,
    alpha: float,
    *,
    deshrink: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Out-of-fold raw and de-shrunk predictions.

    ``grouped`` holds out whole deals (seeds, all levels in every fold);
    ``lolo`` holds out whole levels (style transfer, the honest scheme).
    De-shrink is re-fit inside every fold so the held-out level never feeds
    its own affine calibration.
    """
    raw = np.empty_like(y)
    calibrated = np.empty_like(y)
    slopes: dict[str, float] = {}
    for group in sorted(set(map(str, groups))):
        held_out = groups == group
        model = ridge_fit(x[~held_out], y[~held_out], alpha)
        raw[held_out] = ridge_predict(model, x[held_out])
        if deshrink:
            slope, intercept = _affine_on_training_levels(
                model, x, y, level_values, ~held_out
            )
            calibrated[held_out] = intercept + slope * raw[held_out]
            slopes[group] = slope
        else:
            calibrated[held_out] = raw[held_out]
    return raw, calibrated, slopes


def session_prior_sd(
    y: np.ndarray,
    predictions: np.ndarray,
    level_values: np.ndarray,
    n: int,
    *,
    reps: int = DEFAULT_REPS,
    seed: int = DEFAULT_SEED,
) -> float | None:
    """Empirical SD of ``mean(pred over n games) − true level`` (HR §4.1).

    Draws ``reps`` sessions per level without replacement; ``None`` when any
    level has fewer than ``n`` games.
    """
    rng = np.random.default_rng(seed)
    errors: list[float] = []
    for level in sorted(set(map(str, level_values))):
        indices = np.flatnonzero(level_values == level)
        if len(indices) < n:
            return None
        truth = float(y[indices].mean())
        for _ in range(reps):
            sample = rng.choice(indices, size=n, replace=False)
            errors.append(float(predictions[sample].mean()) - truth)
    return float(np.std(errors, ddof=1))


def _level_order(level_values: np.ndarray, levels: Mapping[str, float]) -> list[str]:
    return sorted(
        set(map(str, level_values)), key=lambda level: (levels[level], level)
    )


def build_prior(
    rows: Sequence[Mapping[str, Any]],
    levels: Mapping[str, float],
    *,
    alpha: float = DEFAULT_ALPHA,
    scheme: str = DEFAULT_SCHEME,
    deshrink: bool = True,
    reps: int = DEFAULT_REPS,
    seed: int = DEFAULT_SEED,
    label_source: str = "t2_full_data",
) -> dict[str, Any]:
    """Calibrate the artifact document (no ``data``/``created_at`` fields)."""
    if scheme not in ("lolo", "grouped"):
        raise ValueError(f"scheme must be 'lolo' or 'grouped', got {scheme!r}")
    if not rows:
        raise ValueError("no usable rows to calibrate")
    level_values = np.array([str(row["level_id"]) for row in rows])
    if len(set(map(str, level_values))) < 2:
        raise ValueError("at least two labelled levels are required to calibrate")
    unknown = sorted(set(map(str, level_values)) - set(levels))
    if unknown:
        raise ValueError(f"rows carry levels missing from the label map: {unknown}")
    y = np.array([float(row["level_elo_ref"]) for row in rows], dtype=float)
    fill = fill_values(rows)
    x = design_matrix(rows, fill)
    if scheme == "lolo":
        groups = level_values
    else:
        groups = np.array(
            [
                str(row["seed"]) if row.get("seed") is not None else str(row["trace_id"])
                for row in rows
            ]
        )

    raw_oof, calibrated_oof, fold_slopes = _cv_predictions(
        x, y, level_values, groups, alpha, deshrink=deshrink
    )
    full_model = ridge_fit(x, y, alpha)
    slope, intercept = _affine_on_training_levels(
        full_model, x, y, level_values, np.ones(len(rows), dtype=bool)
    )
    order = _level_order(level_values, levels)

    residual = y - calibrated_oof
    residual_sd = float(residual.std(ddof=1)) if len(residual) > 1 else 0.0
    rmse = float(np.sqrt(np.mean(residual**2)))
    m_eff = (
        float((SINGLE_GAME_SE / residual_sd) ** 2) if residual_sd > 0.0 else None
    )
    biases = np.array(
        [float(residual[level_values == level].mean()) for level in order]
    )
    within = [
        residual[level_values == level] - bias
        for level, bias in zip(order, biases)
    ]
    within = np.concatenate(within) if within else np.array([])
    tau_between = float(biases.std(ddof=1)) if len(biases) > 1 else 0.0
    sigma_within = float(within.std(ddof=1)) if len(within) > 1 else 0.0

    sigma_traj: dict[str, float] = {}
    for n in range(1, MAX_SESSION + 1):
        value = session_prior_sd(
            y, calibrated_oof, level_values, n, reps=reps, seed=seed
        )
        if value is None:
            break
        sigma_traj[str(n)] = value

    raw_means, calibrated_means, labels = {}, {}, {}
    for level in order:
        mask = level_values == level
        raw = float(ridge_predict(full_model, x[mask]).mean())
        raw_means[level] = raw
        calibrated_means[level] = intercept + slope * raw
        labels[level] = float(y[mask].mean())

    mean, scale, coefficients, center = full_model
    calibration = {
        "scheme": scheme,
        "deshrink": bool(deshrink),
        "rmse_oof": rmse,
        "residual_sd": residual_sd,
        "m_eff": m_eff,
        "tau_between": tau_between,
        "sigma_within": sigma_within,
        "single_game_se": float(SINGLE_GAME_SE),
        "fold_slopes": {key: float(value) for key, value in fold_slopes.items()},
        "level_means": {
            level: {
                "label": labels[level],
                "raw": raw_means[level],
                "calibrated": calibrated_means[level],
                "n": int((level_values == level).sum()),
            }
            for level in order
        },
    }
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "kind": f"trace-s1-ridge-{'deshrunk' if deshrink else 'raw'}",
        "labels": label_source,
        "prior": dict(COLD_START_PRIOR),
        "scheme": scheme,
        "deshrink": {"a": intercept, "b": slope, "applied": bool(deshrink)},
        "sigma_traj": sigma_traj,
        "model": {
            "features": list(MODEL_FEATURES),
            "anchor": {
                "name": ANCHOR_FEATURE,
                "center": ANCHOR_CENTER,
                "scale": ANCHOR_SCALE,
            },
            "interaction": INTERACTION_FEATURE,
            "fill_values": {name: float(value) for name, value in fill.items()},
            "mean": [float(value) for value in mean],
            "scale": [float(value) for value in scale],
            "coefficients": [float(value) for value in coefficients],
            "intercept": float(center),
        },
        "calibration": calibration,
        "hyperparams": {
            "alpha": float(alpha),
            "scheme": scheme,
            "deshrink": bool(deshrink),
            "reps": int(reps),
            "seed": int(seed),
            "max_session": MAX_SESSION,
            "label_source": label_source,
        },
    }


# -- prior model --------------------------------------------------------------


def predict_elo(
    doc: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    opponent_elo: float | None = None,
) -> float:
    """De-shrunk Elo prediction ``f'(φ)`` for one feature row."""
    model = doc["model"]
    names = model["features"]
    values = np.array(
        [
            float(row[feature])
            if _finite(row.get(feature))
            else float(model["fill_values"].get(feature, 0.0))
            for feature in names
        ]
    )
    anchor_elo = (
        float(row[ANCHOR_FEATURE]) if opponent_elo is None else float(opponent_elo)
    )
    anchor = (anchor_elo - float(model["anchor"]["center"])) / float(
        model["anchor"]["scale"]
    )
    interaction = values[names.index(model["interaction"])] * anchor
    design = np.concatenate([values, [anchor, interaction]])
    raw = float(
        ((design - np.asarray(model["mean"])) / np.asarray(model["scale"]))
        @ np.asarray(model["coefficients"])
        + float(model["intercept"])
    )
    deshrink = doc["deshrink"]
    if not deshrink["applied"]:
        return raw
    return float(deshrink["a"]) + float(deshrink["b"]) * raw


def session_mean(
    doc: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    opponent_elos: Sequence[float] | None = None,
) -> float:
    """``μ_traj``: the mean de-shrunk prediction over a session's trace rows.

    ``rows`` normally carry ``opponent_elo_ref`` (the study-row shape).  A live
    placement session knows each opponent rating next to the extracted feature
    row and passes it through ``opponent_elos`` instead.
    """
    if not rows:
        raise ValueError("a session needs at least one trace row")
    if opponent_elos is None:
        return float(np.mean([predict_elo(doc, row) for row in rows]))
    return float(
        np.mean(
            [
                predict_elo(doc, row, opponent_elo=float(elo))
                for row, elo in zip(rows, opponent_elos, strict=True)
            ]
        )
    )


def prior_for_session(
    doc: Mapping[str, Any], mu_traj: float, n: int
) -> Prior:
    """``Prior(μ_traj, σ_traj(n))``; ``n=0`` returns the cold-start prior."""
    if n < 0:
        raise ValueError("session length must be non-negative")
    if n == 0:
        return Prior(**doc["prior"])
    table = doc.get("sigma_traj") or {}
    key = str(int(n))
    if key not in table:
        available = sorted(int(value) for value in table)
        floor = [value for value in available if value <= n]
        if floor:
            key = str(floor[-1])
        elif available:
            key = str(available[0])
        else:
            return Prior(float(mu_traj), float(doc["prior"]["sd"]))
    return Prior(float(mu_traj), float(table[key]))


def load_prior(path: str | Path) -> dict[str, Any]:
    """Read a prior artifact and check its schema tag."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if doc.get("schema") != SCHEMA:
        raise ValueError(f"{path} is not a {SCHEMA} artifact")
    return doc


def save_prior(
    doc: Mapping[str, Any],
    path: str | Path,
    *,
    created_at: str | None = None,
) -> Path:
    """Write the artifact (stable ordering) with a ``created_at`` stamp."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(doc)
    if created_at is not None:
        payload["created_at"] = created_at
    elif "created_at" not in payload:
        payload["created_at"] = datetime.now().isoformat(timespec="seconds")
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path


class TracePrior:
    """The M1 ``prior.json`` artifact as used online.

    The constructor validates the schema; :meth:`load` reads and validates a
    file.  All predictions delegate to the module functions so the offline
    calibration and the online channel share one implementation.
    """

    def __init__(
        self,
        doc: Mapping[str, Any],
        *,
        path: str | Path | None = None,
    ) -> None:
        if doc.get("schema") != SCHEMA:
            raise ValueError(
                f"not a {SCHEMA} artifact, got {doc.get('schema')!r}"
            )
        self.doc = doc
        self.path = Path(path) if path is not None else None

    @classmethod
    def load(cls, path: str | Path) -> TracePrior:
        return cls(load_prior(path), path=path)

    def predict_elo(self, row: Mapping[str, Any], *, opponent_elo: float) -> float:
        return float(predict_elo(self.doc, row, opponent_elo=float(opponent_elo)))

    def prior_for_session(self, mu_traj: float, n: int) -> Prior:
        return prior_for_session(self.doc, float(mu_traj), int(n))

    def features(
        self, trace: Mapping[str, Any], *, verify: bool = False
    ) -> dict[str, Any]:
        return dict(extract_features(trace, verify=verify))

    @property
    def cold_start(self) -> Prior:
        prior = self.doc.get("prior") or COLD_START_PRIOR
        return Prior(float(prior["mean"]), float(prior["sd"]))

    @property
    def labels(self) -> dict[str, float]:
        """Elos the prior was calibrated against (opponent-strength feature)."""
        data = self.doc.get("data") or {}
        return {
            str(name): float(value)
            for name, value in (data.get("labels") or {}).items()
        }

    def meta(self) -> dict[str, Any]:
        return {
            "path": str(self.path) if self.path is not None else None,
            "schema": self.doc.get("schema"),
            "version": self.doc.get("version"),
            "kind": self.doc.get("kind"),
            "labels": self.doc.get("labels"),
            "created_at": self.doc.get("created_at"),
            "scheme": self.doc.get("scheme"),
            "deshrink": dict(self.doc.get("deshrink") or {}),
        }

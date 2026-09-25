#!/usr/bin/env python3
"""M1 of ``docs/human-elo-plan.md``: offline 轨迹 S1 prior calibration.

Pipeline (HR = ``docs/experiments/human-elo-10-games-research.md`` §5):

1. read a trace study (``traces/study``); the **default label target is T2**
   (HR §1: the full-200-game estimates random=1026.9 / greedy=1314.4 /
   lvl1=1128.6 / lvl2=1232.1 / lvl3=1356.1 / lvl4=1447.8, re-verified against
   the shipped study with the shipped estimator).  The manifest labels (refit
   or not) are T1 and are kept as a comparison mode only, per C-6/T13: "新标定
   与 D3 先验不得以 T1 为目标",
2. extract one per-game 轨迹 S1 feature row per trace through the D1 extractor
   (``tools.measure_trace_signal.extract_features`` — features are never
   re-implemented here).  The opponent-strength feature and the subject label
   always come from the same selected label map, so each mode is internally
   self-consistent,
3. fit a standardized ridge ``g(φ) -> Elo`` over the ``CLEAN + PACE`` features
   plus the opponent strength and its ``trick_win_rate`` interaction (HR §5.2),
4. affine-de-shrink the level means: ``f'(φ) = a + b · g(φ)`` (HR §5.2),
5. tabulate the empirical session prior error ``σ_traj(n)`` — by default under
   leave-one-level-out cross-validation with the within-fold de-shrink applied
   (HR §5.2: "表里应存 LOLO+去收缩版本作为默认").

The artifacts are ``artifacts/human-elo/prior.json`` (default T2 labels) and
``artifacts/human-elo/prior_manifest_labels.json`` (``--labels manifest``,
comparison).  Their ``prior`` object is the cold-start
:class:`seven523.elo.Prior` (``Prior(**doc["prior"])``), and
:func:`prior_for_session` builds the ``Prior(μ_traj, σ_traj(n))`` that D3 feeds
to ``fit_ratings(priors=...)`` without touching ``elo.py`` math.

Reproduce the shipped artifacts::

    .venv/bin/python tools/fit_trace_prior.py fit \
        --study traces/study --out artifacts/human-elo/prior.json
    .venv/bin/python tools/fit_trace_prior.py fit --labels manifest \
        --study traces/study --out artifacts/human-elo/prior_manifest_labels.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

_TOOLS = Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

# The design/statistics helpers are shared with D1 on purpose: one
# standardization convention and one ``S1`` extractor.
from measure_trace_signal import (  # noqa: E402
    SINGLE_GAME_SE,
    _ridge_fit,
    _ridge_predict,
    extract_features,
)
from seven523.elo import Prior  # noqa: E402
from seven523.study import load_manifest  # noqa: E402
from seven523.trace import load_trace, parse_player_label  # noqa: E402

__all__ = [
    "ANCHOR_FEATURE",
    "CLEAN_FEATURES",
    "COLD_START_PRIOR",
    "DEFAULT_ALPHA",
    "DEFAULT_REPS",
    "DEFAULT_SCHEME",
    "DEFAULT_SEED",
    "MAX_SESSION",
    "MODEL_FEATURES",
    "PACE_FEATURES",
    "SCHEMA",
    "T2_LABELS",
    "VERSION",
    "build_prior",
    "cmd_fit",
    "data_fingerprint",
    "design_matrix",
    "extract_row",
    "fill_values",
    "load_prior",
    "load_rows",
    "main",
    "predict_elo",
    "prior_for_session",
    "resolve_levels",
    "save_prior",
    "session_mean",
    "trace_paths",
]

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


def _expand(values: Sequence[str] | None) -> list[str]:
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
    for item in _expand(overrides):
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
            _ridge_predict(model, x[train & (level_values == level)]).mean()
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
        model = _ridge_fit(x[~held_out], y[~held_out], alpha)
        raw[held_out] = _ridge_predict(model, x[held_out])
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
    full_model = _ridge_fit(x, y, alpha)
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
        raw = float(_ridge_predict(full_model, x[mask]).mean())
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


def session_mean(doc: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> float:
    """``μ_traj``: the mean de-shrunk prediction over a session's trace rows."""
    if not rows:
        raise ValueError("a session needs at least one trace row")
    return float(np.mean([predict_elo(doc, row) for row in rows]))


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


def cmd_fit(args: argparse.Namespace) -> int:
    study = Path(args.study)
    manifest = load_manifest(study / "manifest.json")
    levels = resolve_levels(manifest, args.level, labels=args.labels)
    label_source = "t2_full_data" if args.labels == "t2" else "manifest_levels"
    rows, stats = load_rows(
        study,
        levels,
        verify=args.verify,
        max_per_level=args.max_per_level,
    )
    if len(rows) < 2:
        print(
            f"not enough usable traces under {study}: {len(rows)} "
            f"(failed={stats['failed']}); need at least 2 rows",
            file=sys.stderr,
        )
        return 1
    try:
        doc = build_prior(
            rows,
            levels,
            alpha=args.alpha,
            scheme=args.scheme,
            deshrink=not args.no_deshrink,
            reps=args.reps,
            seed=args.seed,
            label_source=label_source,
        )
    except ValueError as exc:
        print(f"cannot calibrate: {exc}", file=sys.stderr)
        return 1
    paths = trace_paths(study)
    doc["data"] = {
        "study": str(args.study),
        "manifest": str(study / "manifest.json"),
        "n_traces": len(rows),
        "n_levels": len(doc["calibration"]["level_means"]),
        "games_per_level": dict(stats["games_per_level"]),
        "failed": stats["failed"],
        "skipped_cap": stats["skipped_cap"],
        "verify": bool(args.verify),
        "fingerprint": data_fingerprint(study, paths),
        "label_source": label_source,
        "labels": {name: float(value) for name, value in sorted(levels.items())},
    }
    out = save_prior(doc, args.out, created_at=args.created_at)
    digest = hashlib.sha256(out.read_bytes()).hexdigest()

    calibration = doc["calibration"]
    levels_text = ", ".join(
        f"{name}={value:.1f}" for name, value in sorted(levels.items())
    )
    table_text = ", ".join(
        f"n={n}->{doc['sigma_traj'][n]:.1f}"
        for n in ("5", "10", "20")
        if n in doc["sigma_traj"]
    )
    print(
        f"fit_trace_prior: {len(rows)} traces / "
        f"{doc['data']['n_levels']} levels (verify={args.verify}), "
        f"alpha={args.alpha:g} scheme={args.scheme} "
        f"deshrink={doc['deshrink']['applied']}"
    )
    print(f"labels: {levels_text}")
    m_eff = calibration["m_eff"]
    m_eff_text = "n/a" if m_eff is None else f"{m_eff:.2f}"
    print(
        f"{args.scheme} OOF: RMSE={calibration['rmse_oof']:.1f} "
        f"s={calibration['residual_sd']:.1f} m_eff={m_eff_text} "
        f"tau={calibration['tau_between']:.1f} "
        f"sigma_w={calibration['sigma_within']:.1f}"
    )
    print(
        f"deshrink: a={doc['deshrink']['a']:.1f} b={doc['deshrink']['b']:.3f}"
    )
    print(f"sigma_traj: {table_text}")
    print(f"prior -> {out} (sha256 {digest})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate the trace S1 prior (M1, docs/human-elo-plan.md)"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("fit", help="fit the prior artifact from a trace study")
    fit.add_argument("--study", default="traces/study", help="trace study directory")
    fit.add_argument(
        "--out",
        default="artifacts/human-elo/prior.json",
        help="output prior.json path",
    )
    fit.add_argument(
        "--level",
        action="append",
        default=[],
        help="NAME=ELO override (repeat or comma-separate)",
    )
    fit.add_argument(
        "--labels",
        choices=("t2", "manifest"),
        default="t2",
        help=(
            "label target: t2 = HR §1 full-data values (default, C-6); "
            "manifest = study levels (comparison only)"
        ),
    )
    fit.add_argument(
        "--alpha", type=float, default=DEFAULT_ALPHA, help="ridge strength"
    )
    fit.add_argument(
        "--scheme",
        choices=("lolo", "grouped"),
        default=DEFAULT_SCHEME,
        help="session CV scheme (default lolo; HR §5.2)",
    )
    fit.add_argument(
        "--reps",
        type=int,
        default=DEFAULT_REPS,
        help="session resamples per level for the sigma_traj table (HR §5.2: 400)",
    )
    fit.add_argument("--seed", type=int, default=DEFAULT_SEED)
    fit.add_argument(
        "--max-per-level",
        type=int,
        default=0,
        help="cap traces per level (0 = all; smoke/debug only)",
    )
    fit.add_argument(
        "--no-verify",
        dest="verify",
        action="store_false",
        default=True,
        help="skip replay validation (fast; D1 default is verify)",
    )
    fit.add_argument(
        "--no-deshrink",
        dest="no_deshrink",
        action="store_true",
        default=False,
        help="disable the affine level-mean de-shrink (diagnostics only)",
    )
    fit.add_argument(
        "--created-at",
        default=None,
        help="override the artifact timestamp (reproducible builds)",
    )
    fit.set_defaults(func=cmd_fit)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

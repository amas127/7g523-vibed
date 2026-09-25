#!/usr/bin/env python3
"""M1 of ``docs/human-elo-plan.md``: offline 轨迹 S1 prior calibration.

Pipeline (HR = ``docs/experiments/human-elo-10-games-research.md`` §5):

1. read a trace study (``traces/study``); the **default label target is T2**
   (HR §1: the full-200-game estimates random=1026.9 / greedy=1314.4 /
   lvl1=1128.6 / lvl2=1232.1 / lvl3=1356.1 / lvl4=1447.8, re-verified against
   the shipped study with the shipped estimator).  The manifest labels (refit
   or not) are T1 and are kept as a comparison mode only, per C-6/T13: "新标定
   与 D3 先验不得以 T1 为目标",
2. extract one per-game 轨迹 S1 feature row per trace through the shared
   extractor (:func:`seven523.prior.extract_features` — features are never
   re-implemented here).  The opponent-strength feature and the subject label
   always come from the same selected label map, so each mode is internally
   self-consistent,
3. fit a standardized ridge ``g(φ) -> Elo`` over the ``CLEAN + PACE`` features
   plus the opponent strength and its ``trick_win_rate`` interaction (HR §5.2),
4. affine-de-shrink the level means: ``f'(φ) = a + b · g(φ)`` (HR §5.2),
5. tabulate the empirical session prior error ``σ_traj(n)`` — by default under
   leave-one-level-out cross-validation with the within-fold de-shrink applied
   (HR §5.2: "表里应存 LOLO+去收缩版本作为默认").

The core statistics and the artifact model live in :mod:`seven523.prior`; this
CLI only wires the study and the files.  The artifacts are
``artifacts/human-elo/prior.json`` (default T2 labels) and
``artifacts/human-elo/prior_manifest_labels.json`` (``--labels manifest``,
comparison).  Their ``prior`` object is the cold-start
:class:`seven523.elo.Prior` (``Prior(**doc["prior"])``), and
:func:`seven523.prior.prior_for_session` builds the ``Prior(μ_traj, σ_traj(n))``
that D3 feeds to ``fit_ratings(priors=...)`` without touching ``elo.py`` math.

Reproduce the shipped artifacts::

    .venv/bin/python tools/fit_trace_prior.py fit \
        --study traces/study --out artifacts/human-elo/prior.json
    .venv/bin/python tools/fit_trace_prior.py fit --labels manifest \
        --study traces/study --out artifacts/human-elo/prior_manifest_labels.json
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Iterable

from seven523.prior import (
    ANCHOR_CENTER,
    ANCHOR_FEATURE,
    ANCHOR_SCALE,
    CLEAN_FEATURES,
    COLD_START_PRIOR,
    DEFAULT_ALPHA,
    DEFAULT_REPS,
    DEFAULT_SCHEME,
    DEFAULT_SEED,
    MAX_SESSION,
    MODEL_FEATURES,
    PACE_FEATURES,
    SCHEMA,
    T2_LABELS,
    VERSION,
    build_prior,
    data_fingerprint,
    design_matrix,
    extract_row,
    fill_values,
    load_prior,
    load_rows,
    predict_elo,
    prior_for_session,
    resolve_levels,
    save_prior,
    session_mean,
    trace_paths,
)
from seven523.study import load_manifest

__all__ = [
    "ANCHOR_CENTER",
    "ANCHOR_FEATURE",
    "ANCHOR_SCALE",
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

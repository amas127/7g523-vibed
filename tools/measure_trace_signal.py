#!/usr/bin/env python3
"""D1 of ``docs/human-elo-plan.md``: measure the Elo signal carried by traces.

The plan's M1 milestone: an offline lab bench that

1. **generates** reproducible scripted traces (a scripted strategy stands in
   for the human seat; M2 swaps in trained checkpoints),
2. **extracts** the S1 per-trick feature row for the subject seat, and
3. **calibrates** ``features -> Elo`` and reports the per-game residual SD
   ``s`` with ``m_eff = (347 / s) ** 2`` and a game-cluster bootstrap CI.

It deliberately touches neither the engine nor training.  Traces are produced
through :func:`seven523.play.play_game` (the same recorder as
``7g523-play --save-trace``) and read back through
:func:`seven523.play.replay_trace` (the same integrity path as ``--replay``),
so a study trace and a human trace are the same artifact.

M1 smoke test (random subject, the pinned gauge):

    uv run python tools/measure_trace_signal.py run \\
        --subject random --games 200 \\
        --study traces/study --artifacts artifacts/trace-signal

The three stages can also be run separately with ``generate`` / ``features`` /
``calibrate``.  ``calibrate`` prints the per-level feature table, the
out-of-fold residual SD and the ``m_eff -> CI`` conversion; ``summary.json``
next to the CSV keeps the same numbers for the report.

Caveat owned by the tool: with fewer than ``--min-levels`` distinct levels (or
fewer than 100 games per level) the ``m_eff`` number is a *pipeline check*, not
a go/no-go measurement — M2's 4-6 checkpoint ladder is what makes it decisive.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections.abc import Iterable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from seven523.policies import Policy, policy_from_spec, split_entrant
from seven523.prior import (
    SINGLE_GAME_SE,
    expand,
    extract_features,
    ridge_fit,
    ridge_predict,
    silent,
    trace_paths,
)
from seven523.record import play_recorded, policy_seed
from seven523.rules import Rules, rules_id, rules_identity
from seven523.study import load_manifest, merge_manifest, save_manifest
from seven523.trace import load_trace, parse_player_label, player_label

__all__ = [
    "DEFAULT_LEVEL_ELO",
    "FEATURE_COLUMNS",
    "S1_MODEL_FEATURES",
    "SINGLE_GAME_SE",
    "build_policy",
    "extract_features",
    "main",
]

#: Anchor rating (the RandomBot gauge defines 0; ADR-0012).  The old
#: ``human-elo-plan`` §3.1 window-MAP numbers (random=1000 / greedy=1315) are
#: retired with the GreedyBot; re-derive labels once the M2 ladder is re-measured.
DEFAULT_LEVEL_ELO: dict[str, float] = {"random": 0.0}

Z95 = 1.959963984540054

#: S1 features used by the ridge calibration (all per game, subject view).
S1_MODEL_FEATURES: tuple[str, ...] = (
    "trick_win_rate",
    "trick_point_share",
    "mean_points_per_won_trick",
    "pass_rate",
    "bomb_rate",
    "lead_rate",
    "early_trick_win_rate",
    "mid_trick_win_rate",
    "late_trick_win_rate",
)

#: Every column of the features CSV, in file order (plan §3.3 schema, S1 part).
FEATURE_COLUMNS: tuple[str, ...] = (
    "trace_id",
    "level_id",
    "level_elo_ref",
    "opponent_id",
    "opponent_elo_ref",
    "seat",
    "seed",
    "result",
    "final_score_own",
    "score_diff",
    "tricks_total",
    "tricks_won",
    "trick_win_rate",
    "decisions",
    "pass_rate",
    "bomb_rate",
    "lead_rate",
    "trick_points_won",
    "dug",
    "dug_won",
    "dug_points",
    "mean_points_per_won_trick",
    "trick_point_share",
    "early_trick_win_rate",
    "mid_trick_win_rate",
    "late_trick_win_rate",
    "first_trick_won",
    "decisions_per_trick",
)

#: Columns shown in the calibration's per-level table.
DISPLAY_FEATURES: tuple[str, ...] = (
    "tricks_total",
    "tricks_won",
    "trick_win_rate",
    "trick_point_share",
    "mean_points_per_won_trick",
    "pass_rate",
    "bomb_rate",
    "lead_rate",
    "early_trick_win_rate",
    "mid_trick_win_rate",
    "late_trick_win_rate",
    "first_trick_won",
    "decisions_per_trick",
)


def _resolve_levels(raw: Sequence[str]) -> dict[str, float]:
    levels = dict(DEFAULT_LEVEL_ELO)
    for item in expand(raw):
        if "=" not in item:
            raise SystemExit(f"--level expects NAME=ELO, got {item!r}")
        name, value = item.split("=", 1)
        levels[name.strip()] = float(value)
    return levels


def _parse_subject(raw: str, levels: dict[str, float]) -> tuple[str, str]:
    """``random`` / ``lvl3=ckpt:runs/.../agent.pt`` -> ``(level_id, spec)``."""
    name, spec = split_entrant(raw)
    if name not in levels:
        if spec in DEFAULT_LEVEL_ELO:
            levels[name] = DEFAULT_LEVEL_ELO[spec]
        else:
            raise SystemExit(
                f"level {name!r} has no Elo; pass --level {name}=<elo>"
            )
    return name, spec


def build_policy(spec: str, rules: Rules, seed: int) -> Policy:
    """A fresh policy for one game; delegates to the shared spec parser (ADR-0006)."""
    try:
        return policy_from_spec(spec, rules, seed)
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from exc


# -- generation --------------------------------------------------------------


def cmd_generate(args: argparse.Namespace) -> int:
    rules = Rules(num_players=args.num_players)
    levels = _resolve_levels(args.level)
    subjects = [
        _parse_subject(item, levels) for item in (expand(args.subject) or ["random"])
    ]
    anchors = expand(args.anchor) or ["random"]
    for anchor in anchors:
        if anchor not in levels:
            raise SystemExit(
                f"anchor {anchor!r} has no Elo; pass --level {anchor}=<elo>"
            )
    if not subjects:
        raise SystemExit("nothing to generate; pass --subject")

    out = Path(args.out)
    master = random.Random(args.seed)
    # Pair the deal across subjects (same seed -> same deal), so grouped CV by
    # seed holds out whole deals instead of leaking them across levels.
    game_seeds = [master.randrange(1 << 32) for _ in range(args.games)]
    created_at = datetime.now().isoformat(timespec="seconds")

    for subject_id, subject_spec in subjects:
        subject_dir = out / subject_id
        n = rules.num_players
        for index, game_seed in enumerate(game_seeds):
            seat = index % n
            anchor = anchors[(index // n) % len(anchors)]
            policy = build_policy(subject_spec, rules, policy_seed(game_seed, seat))
            policies: list[Policy] = [policy] * n
            for other in range(n):
                if other != seat:
                    policies[other] = build_policy(
                        anchor, rules, policy_seed(game_seed, other)
                    )
            players = [
                player_label(
                    "subject" if s == seat else "anchor",
                    subject_id if s == seat else anchor,
                    s,
                )
                for s in range(n)
            ]
            play_recorded(
                policies,
                rules=rules,
                seed=game_seed,
                human_seat=seat,
                players=players,
                created_at=created_at,
                trace_dir=subject_dir,
                trace_index=index,
                opponent=anchor,
                print_fn=silent,
            )
        print(f"  {subject_id}: {args.games} games -> {subject_dir}")

    document = load_manifest(out / "manifest.json")
    document = merge_manifest(
        document,
        levels=levels,
        subjects=[
            {
                "id": subject_id,
                "spec": subject_spec,
                "mu": levels[subject_id],
            }
            for subject_id, subject_spec in subjects
        ],
        anchors=[{"id": anchor, "mu": levels[anchor]} for anchor in anchors],
        rules=rules_identity(rules),
        rules_id=rules_id(rules),
    )
    document.update(
        {
            "version": 1,
            "created_at": created_at,
            "seed": args.seed,
            "games": args.games,
            "num_players": rules.num_players,
        }
    )
    save_manifest(out / "manifest.json", document)
    print(
        f"generated {len(subjects)} subject(s) x {args.games} games "
        f"(anchors: {', '.join(anchors)}) -> {out}"
    )
    return 0


# -- S1 features -------------------------------------------------------------


def _opponent_info(
    trace: dict[str, Any], subject: int, levels: dict[str, float]
) -> tuple[str, float]:
    players = trace.get("players", [])
    n = len(trace["final_scores"])
    for seat in range(n):
        if seat == subject:
            continue
        label = str(players[seat]) if seat < len(players) else ""
        role, opponent, _ = parse_player_label(label)
        if role == "anchor":
            return opponent, levels.get(opponent, math.nan)
    return "unknown", math.nan


def _feature_rows(
    study: Path, levels: dict[str, float], verify: bool
) -> tuple[list[dict[str, Any]], int]:
    study = Path(study)
    paths = trace_paths(study)
    rows: list[dict[str, Any]] = []
    failed = 0
    for path in paths:
        try:
            trace = load_trace(path)
            subject = int(trace["human_seat"])
            features = extract_features(trace, verify=verify)
            level_id = path.parent.name if path.parent != study else "human"
            opponent_id, opponent_elo = _opponent_info(trace, subject, levels)
            rows.append(
                {
                    "trace_id": str(path.relative_to(study)),
                    "level_id": level_id,
                    "level_elo_ref": levels.get(level_id, math.nan),
                    "opponent_id": opponent_id,
                    "opponent_elo_ref": opponent_elo,
                    "seat": subject,
                    "seed": trace.get("seed"),
                    **features,
                }
            )
        except (KeyError, ValueError, TypeError) as exc:
            failed += 1
            print(f"  skip {path}: {exc}", file=sys.stderr)
    return rows, failed


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.10g}"
    return str(value)


def cmd_features(args: argparse.Namespace) -> int:
    study = Path(args.study)
    manifest = load_manifest(study / "manifest.json")
    levels = dict(manifest.get("levels", {}))
    levels.update(_resolve_levels(args.level))
    rows, failed = _feature_rows(study, levels, verify=args.verify)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FEATURE_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _cell(row.get(column)) for column in FEATURE_COLUMNS})
    print(f"features: {len(rows)} rows -> {out} ({failed} failed)")
    return 0 if rows else 1


# -- calibration -------------------------------------------------------------


def _as_float(value: Any) -> float:
    if value is None or value == "":
        return math.nan
    return float(value)


def _load_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _fold_ids(groups: Sequence[Any], folds: int, seed: int) -> np.ndarray:
    unique = sorted(set(groups), key=str)
    random.Random(seed).shuffle(unique)
    fold_of = {group: index % folds for index, group in enumerate(unique)}
    return np.array([fold_of[group] for group in groups], dtype=int)


def _select_alpha(
    x: np.ndarray,
    y: np.ndarray,
    groups: Sequence[Any],
    alphas: Sequence[float],
    folds: int,
    seed: int,
) -> float:
    fold = _fold_ids(groups, folds, seed)
    best_alpha, best_mse = float(alphas[0]), math.inf
    for alpha in alphas:
        predictions = np.empty_like(y)
        for index in range(folds):
            train = fold != index
            model = ridge_fit(x[train], y[train], alpha)
            predictions[~train] = ridge_predict(model, x[~train])
        mse = float(np.mean((y - predictions) ** 2))
        if mse < best_mse:
            best_alpha, best_mse = float(alpha), mse
    return best_alpha


def _out_of_fold(
    x: np.ndarray,
    y: np.ndarray,
    groups: Sequence[Any],
    folds: int,
    seed: int,
    alphas: Sequence[float],
) -> tuple[np.ndarray, list[float]]:
    fold = _fold_ids(groups, folds, seed)
    predictions = np.empty_like(y)
    alphas_used: list[float] = []
    for index in range(folds):
        train = fold != index
        inner = _select_alpha(
            x[train], y[train], [g for g, keep in zip(groups, train) if keep],
            alphas, folds, seed + index + 1,
        )
        alphas_used.append(inner)
        model = ridge_fit(x[train], y[train], inner)
        predictions[~train] = ridge_predict(model, x[~train])
    return predictions, alphas_used


def _mann_whitney_auc(high: np.ndarray, low: np.ndarray) -> float:
    """P(high > low) + 0.5 P(tie) via average ranks."""
    values = np.concatenate([high, low])
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start
        while end + 1 < len(values) and sorted_values[end + 1] == sorted_values[start]:
            end += 1
        ranks[order[start : end + 1]] = (start + end) / 2.0 + 1.0
        start = end + 1
    rank_sum = ranks[: len(high)].sum()
    return (rank_sum - len(high) * (len(high) + 1) / 2.0) / (len(high) * len(low))


def _spearman(pairs: Sequence[tuple[float, float]]) -> float:
    x = np.array([pair[0] for pair in pairs], dtype=float)
    y = np.array([pair[1] for pair in pairs], dtype=float)
    rx = np.argsort(np.argsort(x, kind="mergesort"), kind="mergesort").astype(float)
    ry = np.argsort(np.argsort(y, kind="mergesort"), kind="mergesort").astype(float)
    if rx.std() == 0 or ry.std() == 0:
        return math.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def _ci_half_width(games: int, m_eff: float) -> float:
    return Z95 * SINGLE_GAME_SE / math.sqrt(games * m_eff)


def _print_level_table(
    usable: list[tuple[dict[str, str], float, list[float]]], level_order: list[str]
) -> dict[str, Any]:
    by_level: dict[str, list[int]] = {level: [] for level in level_order}
    for index, (row, _y, _values) in enumerate(usable):
        by_level.setdefault(row["level_id"], []).append(index)

    def column(feature: str, indices: Sequence[int]) -> np.ndarray:
        values = np.array([_as_float(usable[i][0].get(feature)) for i in indices])
        return values[~np.isnan(values)]

    table: dict[str, Any] = {}
    header = f"{'feature':<28}" + "".join(
        f"{level:>18}" for level in level_order
    ) + f"{'AUC':>8}"
    print(header)
    print("-" * len(header))
    for feature in DISPLAY_FEATURES:
        cells = []
        means: dict[str, float | None] = {}
        for level in level_order:
            values = column(feature, by_level[level])
            means[level] = float(values.mean()) if len(values) else None
            if len(values):
                se = values.std(ddof=1) / math.sqrt(len(values)) if len(values) > 1 else 0.0
                cells.append(f"{values.mean():.3f} ±{se:.3f}")
            else:
                cells.append("—")
        auc = math.nan
        if len(level_order) >= 2:
            high = column(feature, by_level[level_order[-1]])
            low = column(feature, by_level[level_order[0]])
            if len(high) and len(low):
                auc = _mann_whitney_auc(high, low)
        table[feature] = {
            "mean_by_level": means,
            "auc_low_vs_high": None if math.isnan(auc) else auc,
        }
        auc_text = "—" if math.isnan(auc) else f"{auc:.3f}"
        print(f"{feature:<28}" + "".join(f"{cell:>18}" for cell in cells) + f"{auc_text:>8}")
    return table


def cmd_calibrate(args: argparse.Namespace) -> int:
    raw_rows = _load_rows(args.features)
    usable: list[tuple[dict[str, str], float, list[float]]] = []
    dropped = 0
    for row in raw_rows:
        y = _as_float(row.get("level_elo_ref"))
        values = [_as_float(row.get(feature)) for feature in S1_MODEL_FEATURES]
        if math.isnan(y) or any(math.isnan(value) for value in values):
            dropped += 1
            continue
        usable.append((row, y, values))
    if not usable:
        print("no usable rows: features need level_elo_ref and a complete S1 vector")
        return 1

    x = np.array([values for _row, _y, values in usable], dtype=float)
    y = np.array([_y for _row, _y, _values in usable], dtype=float)
    groups = [row.get("seed") or row["trace_id"] for row, _y, _values in usable]
    level_order = sorted(
        {row["level_id"] for row, _y, _values in usable},
        key=lambda level: float(
            next(_y for row, _y, _values in usable if row["level_id"] == level)
        ),
    )
    counts = {
        level: sum(1 for row, _y, _values in usable if row["level_id"] == level)
        for level in level_order
    }
    n_levels = len(level_order)
    conclusive = n_levels >= args.min_levels and min(counts.values()) >= 100

    print(
        f"S1 calibration: rows={len(raw_rows)} used={len(usable)} "
        f"dropped={dropped} levels={n_levels}"
    )
    print(
        "levels: "
        + ", ".join(
            f"{level}={next(_y for row, _y, _v in usable if row['level_id'] == level):.0f}"
            f" (n={counts[level]})"
            for level in level_order
        )
    )
    level_table = _print_level_table(usable, level_order)
    if n_levels >= 3:
        elo_of = {
            level: next(_y for row, _y, _v in usable if row["level_id"] == level)
            for level in level_order
        }
        for stats in level_table.values():
            pairs = [
                (elo_of[level], mean)
                for level in level_order
                if (mean := stats["mean_by_level"][level]) is not None
            ]
            stats["spearman_elo_vs_feature"] = _spearman(pairs) if len(pairs) >= 3 else None

    summary: dict[str, Any] = {
        "features_csv": str(args.features),
        "rows_total": len(raw_rows),
        "rows_used": len(usable),
        "rows_dropped": dropped,
        "levels": {
            level: next(
                _y for row, _y, _v in usable if row["level_id"] == level
            )
            for level in level_order
        },
        "games_per_level": counts,
        "model_features": list(S1_MODEL_FEATURES),
        "single_game_se_elo": SINGLE_GAME_SE,
        "per_level": level_table,
        "conclusive": conclusive,
        "min_levels_required": args.min_levels,
    }

    if n_levels >= 2 and float(y.std()) > 1e-9:
        alphas = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
        out_of_fold, alphas_used = _out_of_fold(
            x, y, groups, args.folds, args.seed, alphas
        )
        residual = y - out_of_fold
        s = float(residual.std(ddof=1))
        m_eff = (SINGLE_GAME_SE / s) ** 2

        rng = np.random.default_rng(args.seed)
        samples = rng.integers(0, len(residual), size=(args.bootstrap, len(residual)))
        s_samples = residual[samples].std(axis=1, ddof=1)
        s_lo, s_hi = (float(v) for v in np.percentile(s_samples, [2.5, 97.5]))
        m_lo, m_hi = (SINGLE_GAME_SE / s_hi) ** 2, (SINGLE_GAME_SE / s_lo) ** 2

        full_alpha = _select_alpha(x, y, groups, alphas, args.folds, args.seed)
        full_model = ridge_fit(x, y, full_alpha)
        fitted = ridge_predict(full_model, x)
        r2 = 1.0 - float(np.sum((y - fitted) ** 2)) / float(np.sum((y - y.mean()) ** 2))
        rmse = float(np.sqrt(np.mean(residual**2)))

        ci_table = {
            str(games): _ci_half_width(games, m_eff)
            for games in (10, 20, 50, 100)
        }
        summary.update(
            {
                "model": {
                    "kind": "ridge",
                    "alpha_full": full_alpha,
                    "alphas_used": alphas_used,
                    "cv": {"scheme": "grouped-by-seed", "folds": args.folds},
                    "r2_in_sample": r2,
                },
                "residual_sd": s,
                "residual_sd_ci95": [s_lo, s_hi],
                "rmse_oof": rmse,
                "m_eff": m_eff,
                "m_eff_ci95": [m_lo, m_hi],
                "ci_half_width": ci_table,
            }
        )

        print(
            f"\nout-of-fold residual SD s = {s:.2f} Elo "
            f"(95% CI {s_lo:.2f}–{s_hi:.2f}, {args.folds}-fold grouped by seed)"
        )
        print(
            f"m_eff = (347 / s)^2 = {m_eff:.2f} "
            f"(95% CI {m_lo:.2f}–{m_hi:.2f}, game-cluster bootstrap "
            f"n={args.bootstrap})"
        )
        print(f"in-sample R2 = {r2:.3f}, out-of-fold RMSE = {rmse:.1f} Elo")
        print(
            "expected 95% CI half-width: "
            + ", ".join(
                f"{games} games ±{half:.0f}" for games, half in ci_table.items()
            )
        )
        if not conclusive:
            print(
                "\nNOTE: pipeline check only — m_eff needs >= "
                f"{args.min_levels} levels with >= 100 games each (M2 ladder); "
                "do not quote as a go/no-go number."
            )
    else:
        summary["m_eff"] = None
        print("\nnot enough levels/variance for a residual SD and m_eff")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"summary -> {summary_path}")
    return 0


# -- CLI ---------------------------------------------------------------------


def _add_study_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--study", default="traces/study", help="trace study directory")
    parser.add_argument(
        "--artifacts", default="artifacts/trace-signal", help="output directory"
    )
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-players", type=int, default=2)
    parser.add_argument(
        "--subject",
        action="append",
        default=None,
        help="random | id=ckpt:<agent.pt> (repeat or comma-separate)",
    )
    parser.add_argument(
        "--anchor",
        action="append",
        default=None,
        help="opponent levels, default random (repeat or comma-separate)",
    )
    parser.add_argument("--level", action="append", default=[], help="NAME=ELO override")
    parser.add_argument("--no-verify", dest="verify", action="store_false", default=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--min-levels", type=int, default=3)


def cmd_run(args: argparse.Namespace) -> int:
    generate_args = argparse.Namespace(
        out=args.study,
        games=args.games,
        seed=args.seed,
        num_players=args.num_players,
        subject=args.subject,
        anchor=args.anchor,
        level=args.level,
    )
    if cmd_generate(generate_args):
        return 1
    features_path = Path(args.artifacts) / "features.csv"
    features_args = argparse.Namespace(
        study=args.study,
        out=str(features_path),
        level=args.level,
        verify=args.verify,
    )
    if cmd_features(features_args):
        return 1
    calibrate_args = argparse.Namespace(
        features=str(features_path),
        out=args.artifacts,
        folds=args.folds,
        bootstrap=args.bootstrap,
        seed=args.seed,
        min_levels=args.min_levels,
    )
    return cmd_calibrate(calibrate_args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure the Elo signal in 7鬼523 traces (D1, docs/human-elo-plan.md)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="generate + features + calibrate in one shot")
    _add_study_flags(run)
    run.set_defaults(func=cmd_run)

    generate = sub.add_parser("generate", help="simulate scripted games and save traces")
    generate.add_argument("--out", default="traces/study")
    generate.add_argument("--games", type=int, default=200)
    generate.add_argument("--seed", type=int, default=0)
    generate.add_argument("--num-players", type=int, default=2)
    generate.add_argument("--subject", action="append", default=None)
    generate.add_argument("--anchor", action="append", default=None)
    generate.add_argument("--level", action="append", default=[])
    generate.set_defaults(func=cmd_generate)

    features = sub.add_parser("features", help="extract the S1 feature CSV")
    features.add_argument("--study", default="traces/study")
    features.add_argument("--out", default="artifacts/trace-signal/features.csv")
    features.add_argument("--level", action="append", default=[])
    features.add_argument("--no-verify", dest="verify", action="store_false", default=True)
    features.set_defaults(func=cmd_features)

    calibrate = sub.add_parser("calibrate", help="fit features -> Elo and report m_eff")
    calibrate.add_argument("--features", default="artifacts/trace-signal/features.csv")
    calibrate.add_argument("--out", default="artifacts/trace-signal")
    calibrate.add_argument("--folds", type=int, default=5)
    calibrate.add_argument("--bootstrap", type=int, default=1000)
    calibrate.add_argument("--seed", type=int, default=0)
    calibrate.add_argument("--min-levels", type=int, default=3)
    calibrate.set_defaults(func=cmd_calibrate)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

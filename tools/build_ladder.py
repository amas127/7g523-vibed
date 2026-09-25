#!/usr/bin/env python3
"""M2 CLI: rate a set of candidate checkpoints against pinned anchors.

    uv run --group train python tools/build_ladder.py \
        --candidate run_a=ckpt:runs/<new-run-a>/agent.pt \
        --candidate run_b=ckpt:runs/<new-run-b>/agent.pt

The one-command path: candidates come as ``[id=]spec`` (``random`` / ``greedy``
/ ``ckpt:<agent.pt>``; a bare ``ckpt:`` id defaults to the checkpoint's parent
directory), anchors default to Random=1000 / Greedy=1315, and the measured
ratings are frozen into ``<study>/manifest.json`` exactly where
``tools/measure_trace_signal.py`` reads them.  Spacing problems are printed,
never silently relaxed; the exit status stays 0 so a shortfall can be fixed and
re-fitted with ``--refit``.  ``--games-out PATH`` keeps the raw per-game results
as JSONL even under ``--no-traces`` (audit §6.2 #2).
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from seven523.elo import FitConfig, Prior
from seven523.ladder import DEFAULT_ANCHOR_ELO, Entrant, build_ladder
from seven523.policies import split_entrant, validate_spec
from seven523.study import load_manifest, merge_manifest, save_manifest

__all__ = ["main", "parse_args"]


def _require_spec(spec: str) -> None:
    """Exit with the shared grammar's message when ``spec`` is unusable."""
    error = validate_spec(spec)
    if error:
        raise SystemExit(error)


def _parse_spacing(raw: str) -> tuple[float, float]:
    parts = raw.split(":")
    if len(parts) != 2:
        raise SystemExit(f"--spacing expects MIN:MAX, got {raw!r}")
    try:
        low, high = float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise SystemExit(f"--spacing expects MIN:MAX, got {raw!r}") from exc
    if not 0.0 < low <= high:
        raise SystemExit(f"require 0 < MIN <= MAX, got {raw!r}")
    return low, high


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure an Elo ladder from 对局 results (M2, docs/human-elo-plan.md)"
    )
    parser.add_argument(
        "--candidate",
        action="append",
        default=None,
        metavar="[ID=]SPEC",
        help="candidate ckpt/bot spec (repeat; required)",
    )
    parser.add_argument(
        "--anchor",
        action="append",
        default=None,
        metavar="ID=SPEC",
        help="pinned anchor, default random=random and greedy=greedy",
    )
    parser.add_argument(
        "--anchor-elo",
        action="append",
        default=[],
        metavar="ID=ELO",
        help="override an anchor rating (random/greedy have defaults)",
    )
    parser.add_argument(
        "--prior",
        action="append",
        default=[],
        metavar="ID=MEAN:SD",
        help="warm-start prior for a candidate (e.g. its parent's rating)",
    )
    parser.add_argument(
        "--games-per-anchor",
        type=int,
        default=100,
        help="games per candidate per anchor (even; each deal is played in both seats)",
    )
    parser.add_argument(
        "--cross",
        type=int,
        default=0,
        help="extra games per candidate pair (even; each deal is played in both seats)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--window", type=int, default=None, help="trailing games per id (default all)")
    parser.add_argument("--rungs", type=int, default=5)
    parser.add_argument("--spacing", default="100:150", help="MIN:MAX Elo between rungs")
    parser.add_argument("--study", default="traces/study")
    parser.add_argument("--refit", action="store_true", help="overwrite frozen levels")
    parser.add_argument("--no-traces", action="store_true", help="fit without writing traces")
    parser.add_argument(
        "--games-out",
        default=None,
        metavar="PATH",
        help=(
            "append one JSON line per game to PATH (independent of "
            "--no-traces; audit §6.2 #2)"
        ),
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="torch device for ckpt: policies (cpu / cuda / cuda:0)",
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="spawn worker processes (default 1)"
    )
    return parser.parse_args(argv)


def _parse_priors(raw: list[str]) -> dict[str, Prior]:
    priors: dict[str, Prior] = {}
    for item in raw:
        if "=" not in item or ":" not in item:
            raise SystemExit(f"--prior expects ID=MEAN:SD, got {item!r}")
        id_, value = item.split("=", 1)
        mean, sd = value.split(":", 1)
        priors[id_.strip()] = Prior(float(mean), float(sd))
    return priors


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    anchor_elo = dict(DEFAULT_ANCHOR_ELO)
    for item in args.anchor_elo:
        if "=" not in item:
            raise SystemExit(f"--anchor-elo expects ID=ELO, got {item!r}")
        id_, value = item.split("=", 1)
        anchor_elo[id_.strip()] = float(value)

    entrants: list[Entrant] = []
    for raw in args.anchor or ["random=random", "greedy=greedy"]:
        id_, spec = split_entrant(raw)
        _require_spec(spec)
        if id_ not in anchor_elo:
            raise SystemExit(f"anchor {id_!r} has no rating; pass --anchor-elo {id_}=<elo>")
        entrants.append(Entrant(id_, spec, pinned=anchor_elo[id_]))
    if not args.candidate:
        raise SystemExit("at least one --candidate is required")
    priors = _parse_priors(args.prior)
    for raw in args.candidate:
        id_, spec = split_entrant(raw)
        _require_spec(spec)
        entrants.append(Entrant(id_, spec, prior=priors.get(id_)))

    min_spacing, max_spacing = _parse_spacing(args.spacing)
    if args.workers < 1:
        raise SystemExit(f"--workers must be at least 1, got {args.workers}")
    if args.games_per_anchor < 100:
        print(
            f"warning: --games-per-anchor {args.games_per_anchor} is below the "
            "plan's 100 games per anchor",
            file=sys.stderr,
        )
    created_at = datetime.now().isoformat(timespec="seconds")
    study = Path(args.study)
    ladder = build_ladder(
        entrants,
        games_per_anchor=args.games_per_anchor,
        cross=args.cross,
        seed=args.seed,
        out=None if args.no_traces else study,
        games_out=args.games_out,
        created_at=created_at,
        count=args.rungs,
        min_spacing=min_spacing,
        max_spacing=max_spacing,
        window=args.window,
        config=FitConfig(),
        device=args.device,
        workers=args.workers,
    )

    print(f"{'level':<20}{'elo ± se':>18}{'games':>8}  role")
    print("-" * 60)
    for entrant in ladder.entrants:
        rating = ladder.fit.ratings[entrant.id]
        role = "anchor" if entrant.is_anchor else "candidate"
        print(
            f"{entrant.id:<20}{rating.elo:>12.1f} ±{rating.se:>5.1f}"
            f"{rating.n:>8}  {role}"
        )
    print(f"\nfit: {ladder.fit.iterations} sweeps, "
          f"{'converged' if ladder.fit.converged else 'NOT converged'}")
    if args.games_out is not None:
        print(f"games: {Path(args.games_out)}")
    rungs = ", ".join(f"{rung.id}({rung.elo:.0f})" for rung in ladder.selection.rungs)
    print(f"rungs ({len(ladder.selection.rungs)}/{ladder.selection.requested}): {rungs}")
    for lower, upper, gap in ladder.selection.wide_gaps:
        print(f"  wide gap: {lower} -> {upper}: {gap:.0f} Elo")
    if ladder.selection.tail_gap > ladder.selection.max_spacing:
        print(f"  uncovered top: {ladder.selection.tail_gap:.0f} Elo above the last rung")
    if not ladder.selection.ok:
        print("  spacing not satisfied — train/fill more levels, then rerun with --refit")

    if not args.no_traces:
        levels = {entrant.id: ladder.fit.ratings[entrant.id].elo for entrant in ladder.entrants}
        subjects = [
            {
                "id": entrant.id,
                "spec": entrant.spec,
                "elo": ladder.fit.ratings[entrant.id].elo,
                "se": ladder.fit.ratings[entrant.id].se,
                "games": ladder.fit.ratings[entrant.id].n,
            }
            for entrant in ladder.entrants
        ]
        anchors = [
            {"id": entrant.id, "elo": entrant.pinned}
            for entrant in ladder.entrants
            if entrant.is_anchor
        ]
        rungs = [
            {"id": rung.id, "elo": rung.elo, "se": rung.se}
            for rung in ladder.selection.rungs
        ]
        estimator = {
            "kind": "bt-map",
            "window": args.window,
            "seed": args.seed,
            "games_per_anchor": args.games_per_anchor,
            "cross": args.cross,
            "converged": ladder.fit.converged,
        }
        document = load_manifest(study / "manifest.json")
        previous_levels = dict(document.get("levels") or {})
        if not document:
            document = {
                "version": 1,
                "created_at": created_at,
                "seed": args.seed,
                "games": args.games_per_anchor,
                "num_players": 2,
            }
        document = merge_manifest(
            document,
            levels=levels,
            subjects=subjects,
            anchors=anchors,
            rungs=rungs,
            estimator=estimator,
            frozen_at=created_at,
            refit=args.refit,
        )
        path = save_manifest(study / "manifest.json", document)
        print(f"manifest: {path}")
        if not args.refit and any(
            id_ in previous_levels and abs(previous_levels[id_] - elo) > 1e-9
            for id_, elo in levels.items()
        ):
            print("note: existing frozen levels kept; pass --refit to update them")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

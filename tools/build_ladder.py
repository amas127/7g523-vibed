#!/usr/bin/env python3
"""M2 CLI: rate a set of candidate checkpoints against pinned anchors.

    uv run --group train python tools/build_ladder.py \
        --candidate run_a=ckpt:runs/<new-run-a>/agent.pt \
        --candidate run_b=ckpt:runs/<new-run-b>/agent.pt

The one-command path: candidates come as ``[id=]spec`` (``random`` /
``ckpt:<agent.pt>``; a bare ``ckpt:`` id defaults to the checkpoint's parent
directory), the pinned anchor defaults to the RandomBot gauge (``mu = 0``), and
the measured ratings are frozen into ``<study>/manifest.json`` exactly where
``tools/measure_trace_signal.py`` reads them.  Spacing problems are printed,
never silently relaxed; the exit status stays 0 so a shortfall can be fixed and
re-fitted with ``--refit``.  ``--games-out PATH`` keeps the raw per-game results
as JSONL even under ``--no-traces`` (audit §6.2 #2).
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from seven523.elo import FitConfig, Prior
from seven523.ladder import (
    DEFAULT_ANCHOR_RATING,
    Entrant,
    build_ladder,
    manifest_priors,
)
from seven523.policies import split_entrant, validate_spec
from seven523.rules import DEFAULT_RULES, rules_id, rules_identity
from seven523.study import load_manifest, merge_manifest, save_manifest

__all__ = ["main", "parse_args"]


def _require_spec(spec: str) -> None:
    """Exit with the shared grammar's message when ``spec`` is unusable.

    Run-local ``rolloutt:`` search specs are outside this tool's grammar: the
    wrapper lives in ``runs/o4lite-search`` and needs that driver's factory,
    so the row must be measured there and published via
    ``tools/refit_mle.py --manifest-out --refit --spec ID=SPEC`` (ADR-0010
    keeps this tool consuming, not extending, the spec dialect).
    """
    if spec.startswith("rolloutt:"):
        raise SystemExit(
            f"build_ladder cannot build the search spec {spec!r}: search "
            "wrappers need the run-local factory (runs/o4lite-search). "
            "Measure it there and publish with tools/refit_mle.py "
            "--manifest-out --refit --spec ID=SPEC."
        )
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
        description="Measure a rating ladder from 对局 results (M2, docs/human-elo-plan.md)"
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
        help="pinned anchor (gauge), default random=random at mu=0",
    )
    parser.add_argument(
        "--anchor-elo",
        action="append",
        default=[],
        metavar="ID=MU",
        help="override an anchor rating (random defaults to 0)",
    )
    parser.add_argument(
        "--prior",
        action="append",
        default=[],
        metavar="ID=MEAN:SIGMA",
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
        default=None,
        help=(
            "candidate-vs-candidate games per pair (even; each deal is played "
            "in both seats); default = --games-per-anchor (ADR-0012)"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rungs", type=int, default=5)
    parser.add_argument("--spacing", default="100:150", help="MIN:MAX rating between rungs")
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
            raise SystemExit(f"--prior expects ID=MEAN:SIGMA, got {item!r}")
        id_, value = item.split("=", 1)
        mean, sd = value.split(":", 1)
        priors[id_.strip()] = Prior(float(mean), float(sd))
    return priors


def _require_prior_rules_identity(document: Mapping[str, Any]) -> None:
    """Refuse warm-starting from a manifest whose rule identity is not ours.

    ``merge_manifest`` applies the same gate when writing, but priors are read
    before any game is played (and under ``--no-traces`` too), so a legacy or
    cross-version manifest must fail here instead of silently leaking stale
    ratings into the fit (ADR-0013).
    """
    expected = rules_id(DEFAULT_RULES)
    existing = document.get("rules_id")
    if existing is not None:
        if str(existing) != expected:
            raise SystemExit(
                f"study manifest rules_id {str(existing)!r} does not match "
                f"{expected!r}; cross-version ratings must not warm-start a fit "
                "(ADR-0013). Re-measure it under the current rules."
            )
        return
    if document.get("levels") or document.get("subjects"):
        raise SystemExit(
            "study manifest has measured levels/subjects but no rules_id; "
            "cross-version ratings must not warm-start a fit (ADR-0013). "
            "Re-measure it under the current rules."
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    anchor_elo = dict(DEFAULT_ANCHOR_RATING)
    for item in args.anchor_elo:
        if "=" not in item:
            raise SystemExit(f"--anchor-elo expects ID=ELO, got {item!r}")
        id_, value = item.split("=", 1)
        anchor_elo[id_.strip()] = float(value)

    # Warm-start candidates from the frozen manifest levels/subjects, even
    # under --no-traces; an explicit --prior always wins (ADR-0013 §7).  The
    # identity gate runs first: a legacy or cross-version manifest must not
    # leak its stale levels into the fit (T16-F1).
    study = Path(args.study)
    prior_document = load_manifest(study / "manifest.json")
    _require_prior_rules_identity(prior_document)
    priors = {**manifest_priors(prior_document), **_parse_priors(args.prior)}
    entrants: list[Entrant] = []
    for raw in args.anchor or ["random=random"]:
        id_, spec = split_entrant(raw)
        _require_spec(spec)
        if id_ not in anchor_elo:
            raise SystemExit(f"anchor {id_!r} has no rating; pass --anchor-elo {id_}=<elo>")
        entrants.append(Entrant(id_, spec, pinned=anchor_elo[id_]))
    if not args.candidate:
        raise SystemExit("at least one --candidate is required")
    effective_cross = args.cross if args.cross is not None else args.games_per_anchor
    if effective_cross > 0 and len(args.candidate) < 2:
        print(
            f"warning: cross={effective_cross} but fewer than two --candidate "
            "entries; no candidate-vs-candidate games will be planned",
            file=sys.stderr,
        )
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
        config=FitConfig(),
        device=args.device,
        workers=args.workers,
    )

    print(f"{'level':<20}{'mu ± sigma':>18}{'games':>8}  role")
    print("-" * 60)
    for entrant in ladder.entrants:
        rating = ladder.fit.ratings[entrant.id]
        role = "anchor" if entrant.is_anchor else "candidate"
        print(
            f"{entrant.id:<20}{rating.mu:>12.1f} ±{rating.sigma:>5.1f}"
            f"{rating.n:>8}  {role}"
        )
    print(f"\nfit: {ladder.fit.games} games, openskill-plackett-luce")
    if args.games_out is not None:
        print(f"games: {Path(args.games_out)}")
    rungs = ", ".join(f"{rung.id}({rung.mu:.0f})" for rung in ladder.selection.rungs)
    print(f"rungs ({len(ladder.selection.rungs)}/{ladder.selection.requested}): {rungs}")
    for lower, upper, gap in ladder.selection.wide_gaps:
        print(f"  wide gap: {lower} -> {upper}: {gap:.0f} rating")
    if ladder.selection.tail_gap > ladder.selection.max_spacing:
        print(f"  uncovered top: {ladder.selection.tail_gap:.0f} rating above the last rung")
    if not ladder.selection.ok:
        print("  spacing not satisfied — train/fill more levels, then rerun with --refit")

    if not args.no_traces:
        levels = {entrant.id: ladder.fit.ratings[entrant.id].mu for entrant in ladder.entrants}
        subjects = [
            {
                "id": entrant.id,
                "spec": entrant.spec,
                "mu": ladder.fit.ratings[entrant.id].mu,
                "sigma": ladder.fit.ratings[entrant.id].sigma,
                "games": ladder.fit.ratings[entrant.id].n,
            }
            for entrant in ladder.entrants
        ]
        anchors = [
            {"id": entrant.id, "mu": entrant.pinned}
            for entrant in ladder.entrants
            if entrant.is_anchor
        ]
        rungs = [
            {"id": rung.id, "mu": rung.mu, "sigma": rung.sigma}
            for rung in ladder.selection.rungs
        ]
        estimator = {
            "kind": "openskill-plackett-luce",
            "seed": args.seed,
            "games_per_anchor": args.games_per_anchor,
            "cross": args.cross,
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
            rules=rules_identity(DEFAULT_RULES),
            rules_id=rules_id(DEFAULT_RULES),
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

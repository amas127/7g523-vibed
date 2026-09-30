#!/usr/bin/env python3
"""Checkpoint tournament arena CLI: rate a whole league against each other.

    uv run --group train python tools/arena.py \
        --entrant candidate=ckpt:runs/<new-run>/agent.pt \
        --glob 'runs/<new-run>/model_step*.pt' --every 6 --last \
        --cross 60 --games-per-anchor 60 --seed 0 \
        --workers 6 --device cpu \
        --out runs/arena/pilot.json --games-out runs/arena/games \
        --tb runs/arena/tb

Entrants come as ``[ID=]SPEC`` (``random`` / ``ckpt:<path>``) and
may also be discovered with ``--glob`` + ``--every`` sampling (auto ids come
from the file name, ``model_step00020480.pt`` -> ``model20480``).  The pinned
anchor defaults to the RandomBot gauge (``mu = 0``).  The schedule and the parallel
sharding live in ``src/seven523/arena.py``; this file is
the argparse shell plus the table printer.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from glob import glob
from pathlib import Path

# Run as ``python tools/arena.py``: reuse the shared spec grammar.
from seven523.arena import (
    arena_document,
    auto_id_from_path,
    run_arena,
    write_tensorboard,
)
from seven523.elo import FitConfig
from seven523.ladder import DEFAULT_ANCHOR_RATING, Entrant
from seven523.policies import split_entrant, validate_spec

__all__ = ["discover_paths", "main", "parse_args"]


def _require_spec(spec: str) -> None:
    """Exit with the shared grammar's message when ``spec`` is unusable."""
    error = validate_spec(spec)
    if error:
        raise SystemExit(error)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rate a league of checkpoints in one parallel Bradley-Terry "
            "tournament (docs/experiments/tournament-arena.md)"
        )
    )
    parser.add_argument(
        "--entrant",
        action="append",
        default=[],
        metavar="[ID=]SPEC",
        help="candidate ckpt/bot spec (repeat; ID defaults to the file name)",
    )
    parser.add_argument(
        "--glob",
        action="append",
        default=[],
        metavar="PATTERN",
        help="discover every matching checkpoint as an entrant (repeat; sampled by --every)",
    )
    parser.add_argument(
        "--every",
        type=int,
        default=1,
        help="keep every k-th sorted --glob match (default 1 = all)",
    )
    parser.add_argument(
        "--last",
        action="store_true",
        help="also include the last --glob match (useful with --every > 1)",
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
        "--games-per-anchor",
        type=int,
        default=60,
        help="games per entrant per anchor (even; each deal is played in both seats)",
    )
    parser.add_argument(
        "--cross",
        type=int,
        default=60,
        help="extra games per entrant pair (even; each deal is played in both seats)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--workers", type=int, default=1, help="spawn worker processes (default 1)"
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="torch device for ckpt: policies (cpu / cuda / cuda:0)",
    )
    parser.add_argument("--out", default=None, metavar="JSON", help="write the league table")
    parser.add_argument(
        "--games-out",
        default=None,
        metavar="DIR",
        help="write one JSONL per shard: DIR/shard_NNNNN.jsonl",
    )
    parser.add_argument(
        "--tb",
        default=None,
        metavar="DIR",
        help="write each step-bearing entrant's rating as arena/rating/<id> scalars",
    )
    parser.add_argument(
        "--step-map",
        action="append",
        default=[],
        metavar="ID=STEP",
        help="explicit training step for an entrant (repeat)",
    )
    return parser.parse_args(argv)


def discover_paths(pattern: str, every: int, include_last: bool) -> list[str]:
    """Sorted ``glob`` matches, sampled every ``every``; ``include_last`` adds the tail."""
    if every < 1:
        raise SystemExit(f"--every must be at least 1, got {every}")
    matches = sorted(glob(pattern))
    if not matches:
        raise SystemExit(f"--glob matched nothing: {pattern}")
    picked = matches[::every]
    if include_last and matches[-1] not in picked:
        picked.append(matches[-1])
    return picked


def _parse_steps(raw: list[str]) -> dict[str, int]:
    steps: dict[str, int] = {}
    for item in raw:
        if "=" not in item:
            raise SystemExit(f"--step-map expects ID=STEP, got {item!r}")
        id_, value = item.split("=", 1)
        try:
            steps[id_.strip()] = int(value)
        except ValueError as exc:
            raise SystemExit(f"--step-map expects ID=STEP, got {item!r}") from exc
    return steps


def _parse_anchor_elo(raw: list[str]) -> dict[str, float]:
    values = dict(DEFAULT_ANCHOR_RATING)
    for item in raw:
        if "=" not in item:
            raise SystemExit(f"--anchor-elo expects ID=MU, got {item!r}")
        id_, value = item.split("=", 1)
        values[id_.strip()] = float(value)
    return values


def _add(entrants: list[Entrant], seen: dict[str, str], id_: str, spec: str, **kwargs) -> None:
    if id_ in seen:
        raise SystemExit(
            f"duplicate entrant id {id_!r} (from {seen[id_]!r} and {spec!r}); "
            "pass an explicit ID=SPEC to rename one"
        )
    seen[id_] = spec
    entrants.append(Entrant(id_, spec, **kwargs))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    anchor_elo = _parse_anchor_elo(args.anchor_elo)
    entrants: list[Entrant] = []
    seen: dict[str, str] = {}

    for raw in args.anchor or ["random=random"]:
        id_, spec = split_entrant(raw)
        _require_spec(spec)
        if id_ not in anchor_elo:
            raise SystemExit(
                f"anchor {id_!r} has no rating; pass --anchor-elo {id_}=<elo>"
            )
        _add(entrants, seen, id_, spec, pinned=anchor_elo[id_])

    for raw in args.entrant:
        id_, spec = split_entrant(raw)
        _require_spec(spec)
        _add(entrants, seen, id_, spec)

    for pattern in args.glob:
        for path in discover_paths(pattern, args.every, args.last):
            id_ = auto_id_from_path(path)
            _require_spec(f"ckpt:{path}")
            _add(entrants, seen, id_, f"ckpt:{path}")

    candidates = [entrant for entrant in entrants if not entrant.is_anchor]
    if not candidates:
        raise SystemExit("at least one --entrant or --glob candidate is required")
    steps = _parse_steps(args.step_map)

    print(
        f"arena: {len(candidates)} candidates + "
        f"{sum(1 for e in entrants if e.is_anchor)} anchors, "
        f"gpa={args.games_per_anchor} cross={args.cross} seed={args.seed}, "
        f"workers={args.workers} device={args.device}",
        file=sys.stderr,
    )
    started = time.perf_counter()
    result = run_arena(
        entrants,
        games_per_anchor=args.games_per_anchor,
        cross=args.cross,
        seed=args.seed,
        workers=args.workers,
        device=args.device,
        games_out=args.games_out,
        config=FitConfig(),
    )
    elapsed = time.perf_counter() - started
    games = len(result.games)
    rate = games / elapsed if elapsed > 0 else float("inf")
    print(
        f"played {games} games in {elapsed:.1f}s ({rate:.1f} games/s)",
        file=sys.stderr,
    )

    document = arena_document(
        result,
        seed=args.seed,
        games_per_anchor=args.games_per_anchor,
        cross=args.cross,
        workers=args.workers,
        device=args.device,
        steps=steps,
        created_at=datetime.now().isoformat(timespec="seconds"),
    )

    print(f"{'entrant':<26}{'mu':>9}{'sigma':>7}{'games':>8}{'step':>10}  role")
    print("-" * 72)
    for row in document["entrants"]:
        step = "" if row["step"] is None else str(row["step"])
        print(
            f"{row['id']:<26}{row['mu']:>9.1f}{row['sigma']:>7.1f}"
            f"{row['games']:>8}{step:>10}  {row['role']}"
        )
    fit = document["fit"]
    print(
        f"\nfit: {fit['games']} games, {fit['estimator']}"
    )
    if document["transitivity"]:
        worst = document["transitivity"][0]
        print(
            "largest rating/observed disagreement: "
            f"{worst['a']} vs {worst['b']} "
            f"(obs {worst['observed']:.3f} vs exp {worst['expected']:.3f}, "
            f"{worst['games']} games, z={worst['z']:+.1f})"
        )
    if args.out is not None:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"table: {out_path}")
    if args.games_out is not None:
        print(f"games: {Path(args.games_out)}/shard_*.jsonl")
    if args.tb is not None:
        written = write_tensorboard(result, args.tb, steps=steps)
        print(f"tensorboard: {args.tb} ({written} scalars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Batch screening table: several candidate pairs x several seeds x ``--pairs``.

    .venv/bin/python tools/h2h_screen.py \
        --pair candidate_a=ckpt:runs/<new-run-a>/agent.pt \
               candidate_b=ckpt:runs/<new-run-b>/agent.pt \
        --pair candidate_a=ckpt:runs/<new-run-a>/agent.pt \
               candidate_c=ckpt:runs/<new-run-c>/agent.pt \
        --seeds 0,1,2 --pairs 200 --bootstrap 1000

Every ``--pair`` is a ``[ID=]SPEC`` duo (``random`` / ``greedy`` / ``ckpt:``),
played with the seat-balanced duel of ``tools/head_to_head.py`` on each seed;
the readings are merged under the wave5 §4.2 rule
(``seven523.duel.combine_duel_seeds``).  The default screening is 3 seeds x 200
deals — the pilot's recommended first pass before committing to 1500+ deals for
a single promising pair.  ``--games-out DIR`` keeps one JSONL per pair x seed
(audit §6.2 #2); ``--out`` writes the full JSON summary.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import head_to_head as h2h  # noqa: E402
from seven523.duel import ELO_SCALE  # noqa: E402
from seven523.ladder import Entrant  # noqa: E402

__all__ = ["main", "parse_args"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Screen several candidate pairs with multi-seed head-to-head "
            "duels and one merged table (head-to-head-pilot.md §5.4)"
        )
    )
    parser.add_argument(
        "--pair",
        nargs=2,
        action="append",
        default=None,
        metavar=("LEFT", "RIGHT"),
        help="[ID=]SPEC duo, repeatable",
    )
    parser.add_argument(
        "--seeds",
        default="0,1,2",
        help="comma-separated deal-set seeds (default 0,1,2)",
    )
    parser.add_argument(
        "--pairs",
        type=int,
        default=200,
        help="deals per seed and pair (default 200)",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
        help="deal-cluster bootstrap resamples per seed (default 1000)",
    )
    parser.add_argument("--device", default="cpu", help="torch device for ckpt: policies")
    parser.add_argument(
        "--workers", type=int, default=1, help="spawn worker processes (default 1)"
    )
    parser.add_argument(
        "--games-out",
        default=None,
        metavar="DIR",
        help="write one per-game JSONL per pair and seed under DIR",
    )
    parser.add_argument(
        "--out",
        default=None,
        metavar="PATH",
        help="write the JSON summary (per-seed + combined) to PATH",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the JSON summary instead of the table",
    )
    return parser.parse_args(argv)


def _view(result: dict) -> dict:
    """Normalise a single-seed stats dict and a multi-seed envelope for display."""
    if "combined" in result:
        return {
            "k": result["k"],
            "seeds": result["seeds"],
            "deals": result["deals"],
            "games": result["games"],
            "elo_diff": result["combined"]["elo_diff"],
            "winrate": result["combined"]["winrate"],
        }
    half = (result["elo_diff_ci"][1] - result["elo_diff_ci"][0]) / 2.0
    return {
        "k": 1,
        "seeds": [result["seed"]],
        "deals": result["deals"],
        "games": result["games"],
        "elo_diff": {
            "mean": result["elo_diff"],
            "ci": result["elo_diff_ci"],
            "bootstrap_se": half / 1.96,
            "between_seed_sd": 0.0,
        },
        "winrate": {
            "mean": result["winrate"],
            "ci": result["winrate_ci"],
            "bootstrap_se": (result["winrate_ci"][1] - result["winrate_ci"][0]) / 2.0 / 1.96,
            "between_seed_sd": 0.0,
        },
    }


def _print_table(rows: list[tuple[str, str, dict]]) -> None:
    print(
        f"{'pair':<28}{'k':>2} {'seeds':<12}{'deals':>6}  "
        f"{'elo_diff':>8} [{'95% CI':^17}]  "
        f"{'winrate':>8}  {'boot SE':>7} {'seed sd':>7}"
    )
    print("-" * 118)
    for left, right, result in rows:
        view = _view(result)
        elo = view["elo_diff"]
        winrate = view["winrate"]
        seeds = ",".join(str(seed) for seed in view["seeds"])
        label = f"{left} - {right}"
        print(
            f"{label:<28}"
            f"{view['k']:>2} {seeds:<12}{view['deals']:>6}  "
            f"{elo['mean']:>+8.1f} "
            f"[{elo['ci'][0]:+.1f}, {elo['ci'][1]:+.1f}]  "
            f"{winrate['mean']:>8.4f}  "
            f"{elo['bootstrap_se']:>7.1f} {elo['between_seed_sd']:>7.1f}"
        )
    print(
        f"\n(positive elo_diff = left stronger; {ELO_SCALE:.0f}-point logistic scale; "
        "merged CI = mean ± max(bootstrap SE, seed sd)/sqrt(k) × 1.96)"
    )


def _run_pair(
    left: Entrant,
    right: Entrant,
    *,
    seeds: list[int],
    args: argparse.Namespace,
) -> dict:
    games_out = None
    if args.games_out is not None:
        games_out = Path(args.games_out) / f"{left.id}__vs__{right.id}.jsonl"
    result = h2h.run_duel_seeds(
        left,
        right,
        seeds=seeds,
        pairs=args.pairs,
        bootstrap=args.bootstrap,
        device=args.device,
        games_out=games_out,
        workers=args.workers,
    )
    if len(seeds) == 1:
        result = {**result, "seed": seeds[0]}
    return result


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.pair:
        raise SystemExit("at least one --pair LEFT RIGHT is required")
    if args.pairs < 1:
        raise SystemExit(f"--pairs must be at least 1, got {args.pairs}")
    if args.bootstrap < 1:
        raise SystemExit(f"--bootstrap must be at least 1, got {args.bootstrap}")
    if args.workers < 1:
        raise SystemExit(f"--workers must be at least 1, got {args.workers}")
    seeds = h2h._resolve_seeds(argparse.Namespace(seed=None, seeds=args.seeds))

    rows: list[tuple[str, str, dict]] = []
    for left_raw, right_raw in args.pair:
        left = h2h._entrant(left_raw, "pair-left")
        right = h2h._entrant(right_raw, "pair-right")
        if left.id == right.id:
            raise SystemExit(f"pair {left_raw!r} / {right_raw!r} reuses id {left.id!r}")
        print(
            f"=== {left.id} vs {right.id} on seeds {seeds}",
            file=sys.stderr,
        )
        rows.append((left.id, right.id, _run_pair(left, right, seeds=seeds, args=args)))

    summary = {
        "seeds": seeds,
        "pairs_per_seed": args.pairs,
        "bootstrap": args.bootstrap,
        "results": [
            {
                "left_id": left,
                "right_id": right,
                "result": result,
            }
            for left, right, result in rows
        ],
    }
    if args.out is not None:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"summary: {path}", file=sys.stderr)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        _print_table(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

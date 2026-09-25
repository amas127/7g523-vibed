#!/usr/bin/env python3
"""Candidate-vs-candidate 换座对局 CLI (audit §6.5).

    uv run --group train python tools/head_to_head.py \
        --left candidate_a=ckpt:runs/<new-run-a>/agent.pt \
        --right candidate_b=ckpt:runs/<new-run-b>/agent.pt \
        --pairs 400 --seed 0 --device cuda --bootstrap 4000

Both candidates play the *same* ``--pairs`` deals, once from each seat, and the
statistics cluster by deal (twin pair) with a paired bootstrap.  Unlike two
absolute Elo fits, the answer does not depend on the anchor distance or on the
candidate's seat phase (``docs/experiments/elo-reliability-audit.md`` §3.5/§6.5).
``--json`` prints the machine-readable result dict on stdout; ``--out`` writes
replayable traces through the normal ladder path; ``--games-out`` writes the
per-game JSONL (audit §6.2 #2) even without traces.

Pass ``--seeds 0,1,2`` to repeat the duel on independent deal sets and merge the
readings under the wave5 §4.2 rule (point = per-seed mean; CI half-width =
``max(bootstrap-implied SE, between-seed sd) / sqrt(k) * 1.96``).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections.abc import Sequence
from pathlib import Path

# Run as ``python tools/head_to_head.py``: reuse the shared spec grammar.

from seven523.duel import combine_duel_seeds, paired_duel_stats, plan_duel_schedule  # noqa: E402
from seven523.ladder import Entrant, play_games  # noqa: E402
from seven523.policies import split_entrant, validate_spec  # noqa: E402

__all__ = ["main", "parse_args", "run_duel_seeds"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Play two candidates head-to-head on the same deals, seats "
            "swapped, and bootstrap the Elo difference (docs/experiments/"
            "elo-reliability-audit.md §6.5)"
        )
    )
    parser.add_argument(
        "--left",
        required=True,
        metavar="[ID=]SPEC",
        help="left candidate (random / greedy / ckpt:<path>)",
    )
    parser.add_argument(
        "--right",
        required=True,
        metavar="[ID=]SPEC",
        help="right candidate (random / greedy / ckpt:<path>)",
    )
    parser.add_argument(
        "--pairs",
        type=int,
        default=400,
        help="number of deals; each deal is played twice, seats swapped (default 400)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="deal-set and bootstrap seed (default 0; exclusive with --seeds)",
    )
    parser.add_argument(
        "--seeds",
        default=None,
        metavar="S0,S1,...",
        help=(
            "comma-separated deal-set seeds; each runs --pairs deals and the "
            "readings are merged (wave5 §4.2). Exclusive with --seed"
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
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=4000,
        help="deal-cluster bootstrap resamples (default 4000)",
    )
    parser.add_argument(
        "--out",
        default=None,
        metavar="TRACE_DIR",
        help="optional directory for replayable traces (subject = left id)",
    )
    parser.add_argument(
        "--games-out",
        default=None,
        metavar="PATH",
        help=(
            "per-game JSONL (audit §6.2 #2); single seed writes PATH, "
            "multiple seeds write PATH_seed<k> (or PATH/duel_seed<k>.jsonl "
            "when PATH has no suffix)"
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print only the machine-readable stats dict on stdout",
    )
    return parser.parse_args(argv)


def _entrant(raw: str, side: str) -> Entrant:
    id_, spec = split_entrant(raw)
    error = validate_spec(spec)
    if error:
        raise SystemExit(error)
    if not id_:
        raise SystemExit(f"--{side} needs a non-empty id, got {raw!r}")
    return Entrant(id_, spec)


def _resolve_seeds(args: argparse.Namespace) -> list[int]:
    """``--seeds 0,1,2`` wins; otherwise the single ``--seed`` (default 0)."""
    if args.seeds is not None:
        if args.seed is not None:
            raise SystemExit("pass either --seed or --seeds, not both")
        try:
            seeds = [int(part.strip()) for part in args.seeds.split(",") if part.strip()]
        except ValueError as exc:
            raise SystemExit(
                f"--seeds expects comma-separated ints, got {args.seeds!r}"
            ) from exc
        if not seeds:
            raise SystemExit("--seeds needs at least one seed")
        return seeds
    return [0 if args.seed is None else args.seed]


def _games_out_path(
    raw: str | Path | None, seed: int, seeds: Sequence[int]
) -> Path | None:
    """Resolve ``--games-out`` to one JSONL per deal set.

    A single seed writes exactly ``raw``.  Several seeds must not share one
    file (their deal seeds are drawn independently, so a cross-run seed
    collision would break twin pairing), hence the ``_seed<k>`` suffix; a bare
    path with no suffix is treated as a directory.
    """
    if raw is None:
        return None
    path = Path(raw)
    if len(seeds) == 1:
        return path
    if path.suffix:
        return path.with_name(f"{path.stem}_seed{seed}{path.suffix}")
    return path / f"duel_seed{seed}.jsonl"


def run_duel_seeds(
    left: Entrant,
    right: Entrant,
    *,
    seeds: Sequence[int],
    pairs: int,
    bootstrap: int,
    device: str = "cpu",
    out: str | Path | None = None,
    games_out: str | Path | None = None,
    workers: int = 1,
) -> dict:
    """Play one candidate pair on every seed and return its reading.

    A single seed returns the raw :func:`paired_duel_stats` dict (unchanged CLI
    contract); several seeds return the :func:`combine_duel_seeds` envelope.
    Shared by ``head_to_head.py`` and the ``h2h_screen.py`` batch table.
    """
    per_seed = []
    for seed in seeds:
        schedule = plan_duel_schedule(left, right, pairs=pairs, seed=seed)
        print(
            f"seed {seed}: playing {len(schedule)} games ({pairs} deals, seats "
            f"swapped) on {device}",
            file=sys.stderr,
        )
        played = play_games(
            schedule,
            (left, right),
            out=out,
            results_out=_games_out_path(games_out, seed, seeds),
            device=device,
            workers=workers,
        )
        per_seed.append(
            paired_duel_stats(
                played,
                left_id=left.id,
                right_id=right.id,
                bootstrap=bootstrap,
                rng=random.Random(seed),
            )
        )
    if len(seeds) == 1:
        return per_seed[0]
    return combine_duel_seeds(per_seed, seeds=seeds)


def _print_report(stats: dict) -> None:
    left, right = stats["left_id"], stats["right_id"]
    low, high = stats["winrate_ci"]
    diff_low, diff_high = stats["mean_score_diff_ci"]
    elo_low, elo_high = stats["elo_diff_ci"]
    sign = stats["deal_sign"]
    confidence = 100.0 * stats["confidence"]
    print(f"head-to-head: {left} vs {right}")
    print(
        f"deals {stats['deals']}  games {stats['games']}"
        f"  bootstrap {stats['bootstrap']}"
    )
    print(
        f"W/D/L            {stats['wins']:>6} /{stats['draws']:>6} /{stats['losses']:>6}"
    )
    print(
        f"winrate          {stats['winrate']:>6.4f}  "
        f"[{low:.4f}, {high:.4f}]  ({confidence:.0f}% CI)"
    )
    print(
        f"mean score diff  {stats['mean_score_diff']:>+6.2f}  "
        f"[{diff_low:+.2f}, {diff_high:+.2f}]  "
        f"(return {stats['mean_return']:+.4f})"
    )
    print(
        f"elo_diff {left}-{right}  {stats['elo_diff']:>+6.1f}  "
        f"[{elo_low:+.1f}, {elo_high:+.1f}]"
    )
    print(
        f"deal sign        {sign['left_net_wins']} win / "
        f"{sign['left_net_losses']} loss / {sign['ties']} tie  "
        f"exact p={sign['p_value']:.4f}"
    )


def _print_combined_report(combined: dict) -> None:
    """Per-seed rows plus the merged wave5 §4.2 readout."""
    left, right = combined["left_id"], combined["right_id"]
    seed_list = ", ".join(str(seed) for seed in combined["seeds"])
    print(f"head-to-head: {left} vs {right}  ({combined['k']} seeds: {seed_list})")
    print(
        f"deals {combined['deals']}  games {combined['games']}"
        f"  bootstrap {combined['bootstrap']}"
    )
    print("per seed:")
    for stats in combined["per_seed"]:
        elo_low, elo_high = stats["elo_diff_ci"]
        wr_low, wr_high = stats["winrate_ci"]
        print(
            f"  seed {stats['seed']:<3} elo_diff {stats['elo_diff']:>+6.1f} "
            f"[{elo_low:+.1f}, {elo_high:+.1f}]  "
            f"winrate {stats['winrate']:.4f} [{wr_low:.4f}, {wr_high:.4f}]  "
            f"W/D/L {stats['wins']}/{stats['draws']}/{stats['losses']}"
        )
    merged = combined["combined"]
    elo = merged["elo_diff"]
    winrate = merged["winrate"]
    score = merged["mean_score_diff"]
    sign = merged["deal_sign"]
    print(
        f"combined (point = mean; CI = max(bootstrap SE, seed sd)/sqrt(k) x "
        f"{combined['z']:.2f}):"
    )
    print(
        f"  elo_diff {left}-{right}  {elo['mean']:>+6.1f}  "
        f"[{elo['ci'][0]:+.1f}, {elo['ci'][1]:+.1f}]  "
        f"(bootstrap SE {elo['bootstrap_se']:.1f}, "
        f"seed sd {elo['between_seed_sd']:.1f})"
    )
    print(
        f"  winrate          {winrate['mean']:>6.4f}  "
        f"[{winrate['ci'][0]:.4f}, {winrate['ci'][1]:.4f}]"
    )
    print(
        f"  mean score diff  {score['mean']:>+6.2f}  "
        f"[{score['ci'][0]:+.2f}, {score['ci'][1]:+.2f}]"
    )
    print(
        f"  deal sign        {sign['left_net_wins']} win / "
        f"{sign['left_net_losses']} loss / {sign['ties']} tie  "
        f"exact p={sign['p_value']:.4f}"
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.pairs < 1:
        raise SystemExit(f"--pairs must be at least 1, got {args.pairs}")
    if args.bootstrap < 1:
        raise SystemExit(f"--bootstrap must be at least 1, got {args.bootstrap}")
    if args.workers < 1:
        raise SystemExit(f"--workers must be at least 1, got {args.workers}")

    left = _entrant(args.left, "left")
    right = _entrant(args.right, "right")
    if left.id == right.id:
        raise SystemExit(
            "--left and --right must have distinct ids; pass ID=SPEC on one side"
        )
    seeds = _resolve_seeds(args)

    result = run_duel_seeds(
        left,
        right,
        seeds=seeds,
        pairs=args.pairs,
        bootstrap=args.bootstrap,
        device=args.device,
        out=args.out,
        games_out=args.games_out,
        workers=args.workers,
    )
    if args.out is not None:
        print(f"traces: {Path(args.out)}", file=sys.stderr)
    if args.games_out is not None:
        for seed in seeds:
            print(
                f"games: {_games_out_path(args.games_out, seed, seeds)}",
                file=sys.stderr,
            )
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif len(seeds) == 1:
        _print_report(result)
    else:
        _print_combined_report(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

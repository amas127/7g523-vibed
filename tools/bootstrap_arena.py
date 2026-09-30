#!/usr/bin/env python3
"""Bootstrap arena CLI: grow and re-rate the 基线池 of checkpoints.

    uv run --group train python tools/bootstrap_arena.py init --arena runs/bootstrap/a1
    uv run --group train python tools/bootstrap_arena.py round --arena runs/bootstrap/a1
    uv run --group train python tools/bootstrap_arena.py run --arena runs/bootstrap/a1 --rounds 8
    uv run --group train python tools/bootstrap_arena.py rate --arena runs/bootstrap/a1
    uv run python tools/bootstrap_arena.py status --arena runs/bootstrap/a1

``init`` trains the seed models (vs the RandomBot) and rates them into a fresh
pool; ``round`` runs fork → train → qualify → trim → rate rounds on an existing
pool; ``run`` does both; ``rate`` reruns only the mass random-matching pass
(``--refit`` replays the whole recorded history instead); ``status`` prints the
band table without loading torch.  Everything except the CLI lives in
``src/seven523/bootstrap.py``; TensorBoard events stay on for the training
subprocesses (``<arena>/train/<run>/tb``) and the pool summary
(``<arena>/tb``).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from seven523.bootstrap import (
    BootstrapConfig,
    SubprocessTrainer,
    init_pool,
    make_play_fn,
    rate_pool,
    refit_pool,
    run_bootstrap,
    run_round,
    status_rows,
)
from seven523.pool import load_pool

__all__ = ["build_config", "main", "parse_args", "print_pool"]


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--arena", default="runs/bootstrap/arena", help="arena directory (pool.json lives here)"
    )
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--seed-models", type=int, default=5)
    parser.add_argument("--seed-steps", type=int, default=100_000)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--forks", type=int, default=2, help="forks per parent")
    parser.add_argument("--max-children", type=int, default=10)
    parser.add_argument("--train-steps", type=int, default=200_000)
    parser.add_argument("--train-workers", type=int, default=5)
    parser.add_argument("--train-threads", type=int, default=4)
    parser.add_argument(
        "--train-extra",
        action="append",
        default=None,
        metavar="ARG",
        help="extra 7g523-train argument (repeat; appended verbatim)",
    )
    parser.add_argument("--cuda", action="store_true", help="train on CUDA (default CPU)")
    parser.add_argument(
        "--no-tensorboard", action="store_true", help="disable TensorBoard writers"
    )
    parser.add_argument("--pfsp", action="store_true", help="PFSP-weight training pool members")
    parser.add_argument(
        "--include-random",
        type=float,
        default=0.0,
        metavar="W",
        help="extra training-pool weight on the RandomBot (0 = off)",
    )
    parser.add_argument("--qualify-games", type=int, default=500)
    parser.add_argument("--mass-games", type=int, default=200)
    parser.add_argument("--anchor-games", type=int, default=20)
    parser.add_argument("--workers", type=int, default=6, help="play_games shards")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--band-width", type=float, default=200.0)
    parser.add_argument("--band-cap", type=int, default=5)
    parser.add_argument("--margin", type=float, default=10.0)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--progress-margin", type=float, default=10.0)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap arena: fork/train/qualify/trim/rate rounds over a rated pool"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("init", "train + rate the seed models into a new pool"),
        ("round", "run fork/train/qualify/trim/rate rounds on an existing pool"),
        ("run", "init (or resume) then run rounds"),
        ("rate", "rerun the mass random-matching rating pass"),
        ("status", "print the pool's band table"),
    ):
        sub = subparsers.add_parser(name, help=help_text)
        _add_common_args(sub)
        if name == "rate":
            sub.add_argument(
                "--refit",
                action="store_true",
                help="replay the full game history instead of playing new games",
            )
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> BootstrapConfig:
    return BootstrapConfig(
        arena_dir=Path(args.arena),
        base_seed=args.base_seed,
        seed_models=args.seed_models,
        seed_steps=args.seed_steps,
        rounds=args.rounds,
        forks_per_parent=args.forks,
        max_children=args.max_children,
        train_steps=args.train_steps,
        train_workers=args.train_workers,
        train_threads=args.train_threads,
        train_extra=tuple(args.train_extra or ()),
        cuda=args.cuda,
        tensorboard=not args.no_tensorboard,
        pfsp=args.pfsp,
        include_random=args.include_random,
        qualify_games=args.qualify_games,
        mass_games=args.mass_games,
        anchor_games=args.anchor_games,
        workers=args.workers,
        device=args.device,
        band_width=args.band_width,
        band_cap=args.band_cap,
        margin=args.margin,
        patience=args.patience,
        progress_margin=args.progress_margin,
    )


def print_pool(pool, *, title: str = "pool") -> None:
    """Print the band table: lowest band first, strongest first within."""
    bands = pool.bands()
    summary = " ".join(f"{band}:[{len(members)}]" for band, members in bands.items())
    print(f"{title}: {len(pool)} members, round {pool.round}, bands {summary or '-'}")
    print(f"{'band':>4} {'id':<28} {'mu':>8} {'sigma':>7} {'games':>6} {'rnd':>4}  parent")
    for row in status_rows(pool):
        print(
            f"{row['band']:>4} {row['id']:<28} {row['mu']:>8.1f} {row['sigma']:>7.1f} "
            f"{row['games']:>6} {row['round']:>4}  {row['parent'] or '-'}"
        )


def _report(result) -> None:
    for report in result.reports:
        print(
            f"round {report.round}: trained {len(report.trained)}, "
            f"added {len(report.added)}, evicted {len(report.evicted)}, "
            f"games {report.games}, productive={report.productive}"
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = build_config(args)
    play = make_play_fn(config)
    trainer = SubprocessTrainer(config)

    if args.command == "status":
        pool_path = Path(args.arena) / "pool.json"
        if not pool_path.is_file():
            raise SystemExit(f"no pool at {pool_path}; run init/run first")
        print_pool(load_pool(pool_path))
        return 0

    if args.command == "init":
        pool = init_pool(config, trainer=trainer, play=play)
        print_pool(pool)
        return 0

    if args.command == "run":
        result = run_bootstrap(config, trainer=trainer, play=play)
        _report(result)
        print_pool(result.pool)
        return 0

    pool_path = Path(args.arena) / "pool.json"
    if not pool_path.is_file():
        raise SystemExit(f"no pool at {pool_path}; run init/run first")
    pool = load_pool(pool_path)

    if args.command == "round":
        for _ in range(config.rounds):
            run_round(pool, config, trainer=trainer, play=play)
        print_pool(pool)
        return 0

    if args.command == "rate":
        if args.refit:
            fit = refit_pool(pool, config)
            print(f"refit: {fit.games} games replayed from {args.arena}/rounds")
            print(f"{'id':<28} {'pool mu':>9} {'refit mu':>9} {'delta':>8}")
            for member in pool.sorted_members():
                rating = fit.ratings.get(member.id)
                if rating is None:
                    continue
                print(
                    f"{member.id:<28} {member.mu:>9.1f} {rating.mu:>9.1f} "
                    f"{rating.mu - member.mu:>+8.1f}"
                )
            out = Path(args.arena) / "refit.json"
            out.write_text(
                json.dumps(
                    {
                        "games": fit.games,
                        "ratings": {
                            member_id: {"mu": rating.mu, "sigma": rating.sigma, "n": rating.n}
                            for member_id, rating in fit.ratings.items()
                        },
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            print(f"refit: {out}")
            return 0
        rate_pool(pool, config, play=play)
        print_pool(pool, title="rated")
        return 0

    raise SystemExit(f"unknown command {args.command!r}")  # pragma: no cover


if __name__ == "__main__":
    raise SystemExit(main())

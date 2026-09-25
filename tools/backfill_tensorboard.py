#!/usr/bin/env python3
"""Backfill TensorBoard events from ``runs/<run>/metrics.csv``.

Some runs are launched with ``--tensorboard False`` (e.g. ten parallel
screening runs, where CSV is enough and event-file I/O is one more thing to
contend on).  The CSV already carries every logged scalar, so the event file
can be reconstructed after the fact:

    uv run --group train python tools/backfill_tensorboard.py runs/ebo_*

Skips a run that already has events unless ``--force``.  Writes into the same
``<run>/tb`` location the trainer uses, so ``tensorboard --logdir runs`` picks
both live and backfilled runs up without extra configuration.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from seven523.metrics import LOG_FIELDS, TensorboardLogger

__all__ = ["backfill", "main", "parse_args"]


def _optional_float(raw: str) -> float | None:
    if raw == "" or raw.lower() in {"nan", "none"}:
        return None
    return float(raw)


def backfill(run_dir: Path, *, force: bool = False) -> Path | None:
    """Write one scalar series per ``metrics.csv`` column; return the tb path."""
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        raise SystemExit(f"no metrics.csv under {run_dir}")
    tb_path = run_dir / "tb"
    if any(tb_path.glob("events.out.tfevents.*")) and not force:
        return None

    logger = TensorboardLogger(tb_path, enabled=True)
    if logger.writer is None:
        raise SystemExit("tensorboardX is not installed; run `uv sync --group train`")
    args_path = run_dir / "args.json"
    if args_path.exists():
        args = json.loads(args_path.read_text())
        logger.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n"
            + "\n".join(f"|{key}|{value}|" for key, value in sorted(args.items())),
        )

    with metrics_path.open() as handle:
        for row in csv.DictReader(handle):
            step = int(row["global_step"])
            metrics = {field: _optional_float(row.get(field, "")) for field in LOG_FIELDS}
            logger.log_update(step, metrics)
            if metrics["episodic_return"] is not None and metrics["episodic_length"] is not None:
                logger.log_episode(
                    step, metrics["episodic_return"], int(metrics["episodic_length"])
                )
    logger.close()
    return tb_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="run directories (or their parent)")
    parser.add_argument("--force", action="store_true", help="overwrite existing tb events")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_dirs: list[Path] = []
    for path in args.runs:
        run_dirs.extend(sorted(p for p in path.glob("*") if p.is_dir()) if not (path / "metrics.csv").exists() else [path])
    for run_dir in run_dirs:
        written = backfill(run_dir, force=args.force)
        state = "wrote" if written else "skipped (tb exists)"
        print(f"{run_dir}: {state}")


if __name__ == "__main__":
    main()

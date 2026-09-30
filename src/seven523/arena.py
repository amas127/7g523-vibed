"""Checkpoint arena: one OpenSkill league over many snapshots at once.

Where ``ladder.build_ladder`` rates a handful of candidates against pinned
anchors and ``duel.plan_duel_schedule`` resolves a single pair, the arena puts
*the whole league* into one tournament: every entrant plays both pinned anchors
and every other entrant (seat-paired twins from :func:`ladder.plan_games`), and
one joint :func:`elo.fit_ratings` replay turns all of it into a common rating
scale.

Two things this module adds over ``build_ladder``:

* **Parallelism** — :func:`play_parallel` is a thin compatibility wrapper
  over :func:`seven523.ladder.play_games`, which owns the shard machinery: the
  schedule is split into contiguous shards played by a spawn-based
  :class:`ProcessPoolExecutor` (spawn avoids forking a CUDA context).  Each
  worker keeps its own ``policies._AGENT_CACHE``, so a checkpoint is loaded from
  disk once per worker, not once per game.  Every policy is seeded from its
  game, so ``workers=1`` and ``workers>1`` return bit-identical ``PlayedGame``
  lists — the regression test pins this.
* **Joining** — one joint :func:`elo.fit_ratings` over the whole league.

The module is pure Python except for the lazy torch import behind a ``ckpt:``
spec; schedule construction and fitting stay in ``ladder``/``elo``.
"""
from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .elo import Fit, FitConfig, PlayedGame, expected_score, fit_ratings
from .ladder import Entrant, ScheduledGame, plan_games, play_games
from .rules import DEFAULT_RULES, Rules

__all__ = [
    "ArenaResult",
    "arena_document",
    "auto_id_from_path",
    "infer_step",
    "pair_diagnostics",
    "pair_stats",
    "play_parallel",
    "ranking_rows",
    "run_arena",
    "write_tensorboard",
]

#: File stems that carry no information (the id should come from the directory).
_PLACEHOLDER_STEMS = frozenset({"agent", "checkpoint", "model"})

#: ``my_run__1__1790319698`` -> ``my_run`` (run-suffix cleanup).
_RUN_SUFFIX = re.compile(r"(?:__\d+)+$")

#: ``model_step00020480`` -> model20480 (step compression into the id).
_STEP_NAME = re.compile(r"^(?P<base>.+?)[_-]?step0*(?P<num>\d+)$")

#: Extracts a training step from an id or spec.
_STEP_TEXT = re.compile(r"step0*(\d+)")


def play_parallel(
    schedule: Sequence[ScheduledGame],
    entrants: Sequence[Entrant],
    *,
    workers: int = 1,
    device: str = "cpu",
    rules: Rules = DEFAULT_RULES,
    games_out: str | Path | None = None,
) -> list[PlayedGame]:
    """Play ``schedule`` with ``workers`` spawn processes, merged in schedule order.

    Thin compatibility wrapper over :func:`seven523.ladder.play_games`, which
    now owns the shard machinery; the arena CLI and its regression test speak
    this name.  Per-game policy seeds only depend on the game, never on which
    shard runs it, so the returned list is identical for every ``workers``.
    With ``games_out`` (a directory) each shard writes its own
    ``shard_NNNNN.jsonl`` directly; existing ``shard_*.jsonl`` files are
    removed first so a rerun replaces rather than duplicates rows.
    """
    return play_games(
        schedule,
        entrants,
        rules=rules,
        workers=workers,
        device=device,
        results_dir=games_out,
    )


@dataclass(frozen=True, slots=True)
class ArenaResult:
    """The fitted league plus the raw material that produced it."""

    entrants: tuple[Entrant, ...]
    fit: Fit
    schedule: tuple[ScheduledGame, ...]
    games: tuple[PlayedGame, ...]


def run_arena(
    entrants: Sequence[Entrant],
    *,
    games_per_anchor: int = 60,
    cross: int = 60,
    seed: int = 0,
    workers: int = 1,
    device: str = "cpu",
    rules: Rules = DEFAULT_RULES,
    games_out: str | Path | None = None,
    config: FitConfig = FitConfig(),
) -> ArenaResult:
    """Plan, play (possibly in parallel) and fit one league tournament.

    Anchors are pinned exactly as in ``build_ladder``; candidate ``prior`` s
    warm-start the fit.  Every entrant plays every anchor and every other
    entrant, so the whole league shares one Elo scale after the single joint
    fit.
    """
    entrants = tuple(entrants)
    schedule = plan_games(
        entrants, games_per_anchor=games_per_anchor, cross=cross, seed=seed
    )
    anchors = {
        entrant.id: entrant.pinned
        for entrant in entrants
        if entrant.is_anchor and entrant.pinned is not None
    }
    priors = {
        entrant.id: entrant.prior
        for entrant in entrants
        if not entrant.is_anchor and entrant.prior is not None
    }
    games = play_parallel(
        schedule,
        entrants,
        workers=workers,
        device=device,
        rules=rules,
        games_out=games_out,
    )
    fit = fit_ratings(games, anchors=anchors, priors=priors, config=config)
    return ArenaResult(entrants, fit, schedule, tuple(games))


def infer_step(text: str) -> int | None:
    """Extract a training step from ``model_step00696320.pt`` / ``sp983040``."""
    match = _STEP_TEXT.search(text)
    return int(match.group(1)) if match else None


def auto_id_from_path(path: str | Path) -> str:
    """Derive a short, step-bearing entrant id from a checkpoint path.

    ``runs/<new-run>/model_step00020480.pt`` -> ``model20480``;
    ``runs/my_run__1__1790319698/agent.pt`` -> ``my_run`` (the ``agent`` stem
    falls back to the run directory, whose run suffix is stripped).
    """
    path = Path(path)
    name = path.stem
    if name in _PLACEHOLDER_STEMS:
        name = path.parent.name
    name = _RUN_SUFFIX.sub("", name)
    match = _STEP_NAME.match(name)
    if match:
        name = f"{match.group('base')}{int(match.group('num'))}"
    return name or path.stem


def ranking_rows(
    result: ArenaResult, *, steps: Mapping[str, int] | None = None
) -> list[dict]:
    """One row per entrant, highest rating first (ties broken by id).

    ``step`` comes from the explicit ``steps`` map first, then from the id or
    spec (``step00020480``); ``None`` means the entrant has no training step.
    """
    explicit = dict(steps or {})
    rows: list[dict] = []
    for entrant in result.entrants:
        rating = result.fit.ratings[entrant.id]
        step = explicit.get(entrant.id)
        if step is None:
            step = infer_step(entrant.id)
        if step is None:
            step = infer_step(entrant.spec)
        rows.append(
            {
                "id": entrant.id,
                "spec": entrant.spec,
                "role": "anchor" if entrant.is_anchor else "candidate",
                "pinned": entrant.pinned,
                "mu": rating.mu,
                "sigma": rating.sigma,
                "games": rating.n,
                "step": step,
            }
        )
    rows.sort(key=lambda row: (-row["mu"], row["id"]))
    return rows


def pair_stats(games: Iterable[PlayedGame]) -> dict[str, dict]:
    """Per unordered pair ``"a|b"``: W/D/L and ``a``'s expected-score rate.

    Pure and order-independent; the keys use sorted ids so a game reads the
    same regardless of who sat left.
    """
    records: dict[tuple[str, str], dict[str, float]] = {}
    for game in games:
        left, right = game.seats
        a, b = (left, right) if left < right else (right, left)
        sa = game.scores[game.seats.index(a)]
        sb = game.scores[game.seats.index(b)]
        record = records.setdefault(
            (a, b), {"games": 0, "a_wins": 0, "b_wins": 0, "draws": 0, "a_score": 0.0}
        )
        record["games"] += 1
        if sa > sb:
            record["a_wins"] += 1
            record["a_score"] += 1.0
        elif sa < sb:
            record["b_wins"] += 1
        else:
            record["draws"] += 1
            record["a_score"] += 0.5
    stats: dict[str, dict] = {}
    for (a, b), record in sorted(records.items()):
        games_played = int(record["games"])
        stats[f"{a}|{b}"] = {
            "games": games_played,
            "a_wins": int(record["a_wins"]),
            "b_wins": int(record["b_wins"]),
            "draws": int(record["draws"]),
            "a_rate": record["a_score"] / games_played,
        }
    return stats


def pair_diagnostics(
    result: ArenaResult, *, min_games: int = 20
) -> list[dict]:
    """Largest observed-vs-model disagreements, a transitivity smoke test.

    The Plackett–Luce fit assumes the pair probabilities are consistent with
    one rating per entrant; a pair whose observed rate is far from the model's
    expectation is the first place a cycle would show.  ``z`` uses a
    deal-clustered binomial approximation (``games // 2`` effective deals, the
    twin count) and is a diagnostic, not a calibrated test.  Sorted by
    ``|z|`` descending.
    """
    rows: list[dict] = []
    for key, record in pair_stats(result.games).items():
        a, b = key.split("|")
        games_played = record["games"]
        if games_played < min_games:
            continue
        expected = expected_score(
            result.fit.ratings[a], result.fit.ratings[b]
        )
        deals = max(1, games_played // 2)
        se = math.sqrt(max(expected * (1.0 - expected), 1e-12) / deals)
        rows.append(
            {
                "a": a,
                "b": b,
                "games": games_played,
                "expected": expected,
                "observed": record["a_rate"],
                "z": (record["a_rate"] - expected) / se,
            }
        )
    rows.sort(key=lambda row: -abs(row["z"]))
    return rows


def arena_document(
    result: ArenaResult,
    *,
    seed: int,
    games_per_anchor: int,
    cross: int,
    workers: int,
    device: str,
    steps: Mapping[str, int] | None = None,
    created_at: str | None = None,
) -> dict:
    """The machine-readable league table written by ``--out``."""
    rows = ranking_rows(result, steps=steps)
    return {
        "created_at": created_at,
        "seed": seed,
        "games_per_anchor": games_per_anchor,
        "cross": cross,
        "workers": workers,
        "device": device,
        "games": len(result.games),
        "anchors": {
            entrant.id: entrant.pinned
            for entrant in result.entrants
            if entrant.is_anchor
        },
        "fit": {
            "estimator": "openskill-plackett-luce",
            "games": result.fit.games,
        },
        "entrants": rows,
        "ranking": [row["id"] for row in rows],
        "pair_stats": pair_stats(result.games),
        "transitivity": pair_diagnostics(result),
    }


def write_tensorboard(
    result: ArenaResult,
    out_dir: str | Path,
    *,
    steps: Mapping[str, int] | None = None,
) -> int:
    """Log each measured rating at its training step as ``arena/rating/<id>``.

    Ids sharing the ``arena/elo`` prefix group into one TensorBoard chart, so
    the scalar dashboard draws the progress curves directly.  Entrants without
    a step (e.g. the pinned anchors, final-only runs) are skipped.
    Returns the number of scalars written.
    """
    from torch.utils.tensorboard import SummaryWriter

    explicit = dict(steps or {})
    written = 0
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with SummaryWriter(log_dir=str(out)) as writer:
        for entrant in result.entrants:
            step = explicit.get(entrant.id)
            if step is None:
                step = infer_step(entrant.id)
            if step is None:
                step = infer_step(entrant.spec)
            if step is None:
                continue
            writer.add_scalar(
                f"arena/rating/{entrant.id}", result.fit.ratings[entrant.id].mu, int(step)
            )
            written += 1
    return written

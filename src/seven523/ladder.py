"""M2 ladder construction: schedule paired 对局, play them, fit and select rungs.

The plan (``docs/human-elo-plan.md`` §3.1) needs a set of bot levels whose Elo
is *measured*, not trained: every candidate plays each pinned anchor enough
times for a usable rating, and the result is frozen into the study manifest so
``tools/measure_trace_signal.py`` reads the same ratings as labels.

This module is the orchestration around the pure core in :mod:`seven523.elo`:
:func:`plan_games` is a pure schedule (paired deals, seat-balanced twins), and
:func:`play_games` drives it through :func:`seven523.play.play_game` so every
ladder 牌局 produces the same replay-verifiable trace as a human game.  The only
injected behaviour is the policy factory (:func:`seven523.policies.policy_from_spec`
by default) — the single lazy-torch boundary (ADR-0006).

``play_games(workers=N)`` shards the schedule into contiguous chunks played by
spawn processes (the same mechanism ``arena`` used to own).  Per-game policy
seeds depend only on the game, and trace names use the *global* schedule index,
so every ``workers`` value returns bit-identical ``PlayedGame`` s and trace
files; ``workers=1`` is the original single-process loop.
"""
from __future__ import annotations

import json
import multiprocessing
import pickle
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Callable, Sequence

from .elo import Fit, FitConfig, PlayedGame, Prior, RungSelection, fit_ratings, select_rungs
from .policies import Policy, policy_from_spec
from .record import play_recorded, policy_seed
from .rules import DEFAULT_RULES, Rules
from .trace import player_label

__all__ = [
    "Entrant",
    "Ladder",
    "ScheduledGame",
    "build_ladder",
    "make_factory",
    "plan_games",
    "play_games",
    "split_schedule",
]

#: The plan's anchor ratings (§3.1); the CLI defaults to these two 秤砣.
DEFAULT_ANCHOR_ELO: dict[str, float] = {"random": 1000.0, "greedy": 1315.0}


def _silent(*_args: object, **_kwargs: object) -> None:
    """print_fn for the reuse of ``play_game`` without terminal noise."""


def make_factory(*, device: str = "cpu") -> Callable[[str, Rules, int], Policy]:
    """The policy factory injected into :func:`play_games`.

    ``policy_from_spec`` caches loaded agents per ``(path, device)`` at module
    scope, so a long-lived worker process pays each checkpoint's disk read
    exactly once; ``partial`` keeps the factory picklable for spawn workers.
    """
    return partial(policy_from_spec, device=device)


@dataclass(frozen=True, slots=True)
class Entrant:
    """A policy entering the ladder: a pinned anchor or a free candidate."""

    id: str
    spec: str
    pinned: float | None = None
    prior: Prior | None = None

    def __post_init__(self) -> None:
        if not self.id or "@" in self.id:
            raise ValueError(f"entrant id must be non-empty and free of '@': {self.id!r}")
        if not self.spec:
            raise ValueError(f"entrant {self.id!r} needs a policy spec")
        if self.pinned is not None and self.prior is not None:
            raise ValueError(f"pinned anchor {self.id!r} cannot take a prior")

    @property
    def is_anchor(self) -> bool:
        return self.pinned is not None


@dataclass(frozen=True, slots=True)
class ScheduledGame:
    """One planned 牌局: which entrant sits where, and who is the study subject."""

    seed: int
    seats: tuple[str, ...]
    subject: str

    def __post_init__(self) -> None:
        if self.subject not in self.seats:
            raise ValueError(f"subject {self.subject!r} is not seated in {self.seats!r}")


@dataclass(frozen=True, slots=True)
class Ladder:
    """The measured ladder: entrants, the fit, the selection, and the schedule."""

    entrants: tuple[Entrant, ...]
    fit: Fit
    selection: RungSelection
    schedule: tuple[ScheduledGame, ...]


def plan_games(
    entrants: Sequence[Entrant],
    *,
    games_per_anchor: int = 100,
    cross: int = 0,
    seed: int = 0,
) -> tuple[ScheduledGame, ...]:
    """Plan counterbalanced 对局: every candidate meets every anchor ``games_per_anchor`` times.

    ``games_per_anchor`` must be even.  One deal seed is drawn per
    (round, anchor) and reused by every candidate in that slot, so grouped
    comparisons see the same deals; every deal is then scheduled **twice with
    the candidate on each seat**, so the per-deal hand/seat asymmetry cancels
    inside the twin pair and the candidate's list position no longer changes
    which games it plays (the old index-parity seat rotation made Elo drift
    with candidate order — see docs/experiments/elo-reliability-audit.md).
    ``cross`` adds candidate-vs-candidate games per pair (even; 0 by default:
    two anchors already make every rating identifiable).  Pure and
    deterministic — no policies are built here.
    """
    entrants = tuple(entrants)
    if len({entrant.id for entrant in entrants}) != len(entrants):
        raise ValueError("entrant ids must be unique")
    anchors = [entrant for entrant in entrants if entrant.is_anchor]
    candidates = [entrant for entrant in entrants if not entrant.is_anchor]
    if not candidates:
        raise ValueError("a ladder needs at least one candidate (unpinned entrant)")
    if not anchors:
        raise ValueError("a ladder needs at least one pinned anchor for identifiability")
    if games_per_anchor < 1:
        raise ValueError("games_per_anchor must be at least 1")
    if games_per_anchor % 2 != 0:
        raise ValueError(
            "games_per_anchor must be even: each deal is played in both seats"
        )
    if cross < 0:
        raise ValueError("cross cannot be negative")
    if cross % 2 != 0:
        raise ValueError("cross must be even: each deal is played in both seats")

    master = random.Random(seed)
    schedule: list[ScheduledGame] = []
    for index in range(games_per_anchor // 2):
        for anchor in anchors:
            deal_seed = master.randrange(1 << 32)
            for candidate in candidates:
                for seat in (0, 1):
                    seats = [anchor.id, anchor.id]
                    seats[seat] = candidate.id
                    schedule.append(
                        ScheduledGame(deal_seed, tuple(seats), candidate.id)
                    )
    for index in range(cross // 2):
        deal_seed = master.randrange(1 << 32)
        for left in range(len(candidates)):
            for right in range(left + 1, len(candidates)):
                first, second = candidates[left].id, candidates[right].id
                schedule.append(ScheduledGame(deal_seed, (first, second), first))
                schedule.append(ScheduledGame(deal_seed, (second, first), first))
    return tuple(schedule)


def split_schedule(
    schedule: Sequence[ScheduledGame], shards: int
) -> tuple[tuple[ScheduledGame, ...], ...]:
    """Split ``schedule`` into ``shards`` contiguous, size-balanced chunks.

    Contiguous chunks keep a worker's JSONL in schedule order and make the
    merge a plain ``extend``.  Pure and deterministic; when there are more
    shards than games the tail shards are empty.
    """
    schedule = tuple(schedule)
    if shards < 1:
        raise ValueError(f"shards must be at least 1, got {shards}")
    base, extra = divmod(len(schedule), shards)
    chunks: list[tuple[ScheduledGame, ...]] = []
    start = 0
    for index in range(shards):
        size = base + (1 if index < extra else 0)
        chunks.append(schedule[start : start + size])
        start += size
    return tuple(chunks)


@dataclass(frozen=True, slots=True)
class _ShardJob:
    """One worker task: a contiguous slice plus the global index it starts at.

    ``offset`` keeps :func:`trace_filename` on the global schedule index, and
    ``factory`` is normally the picklable :func:`make_factory` partial (spawn
    workers re-import it by reference, so a closure cannot cross the boundary).
    """

    index: int
    offset: int
    schedule: tuple[ScheduledGame, ...]
    entrants: tuple[Entrant, ...]
    rules: Rules
    factory: Callable[[str, Rules, int], Policy] | None
    out: str | None
    results_out: str | None
    created_at: str | None
    device: str


def _play_shard(job: _ShardJob) -> tuple[int, list[PlayedGame]]:
    """Worker entry point; torch threads are capped to one per process."""
    if any(entrant.spec.startswith("ckpt:") for entrant in job.entrants):
        try:
            import torch

            torch.set_num_threads(1)
        except ImportError:  # pragma: no cover - torch is a train-group extra
            pass
    factory = job.factory
    if factory is None:
        factory = make_factory(device=job.device)
    return job.index, _play_games_range(
        job.schedule,
        job.entrants,
        rules=job.rules,
        factory=factory,
        out_path=Path(job.out) if job.out is not None else None,
        results_path=Path(job.results_out) if job.results_out is not None else None,
        stamp=job.created_at,
        index_offset=job.offset,
    )


def _run_shards(jobs: tuple[_ShardJob, ...]) -> list[PlayedGame]:
    """Run jobs in spawn workers and flatten back into schedule order.

    A single shard runs in-process: ``arena``'s ``workers=1`` path must not pay
    a process-pool round trip, and :func:`play_games` never builds one shard
    for its own parallel path.
    """
    if len(jobs) == 1:
        return _play_shard(jobs[0])[1]
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(jobs), mp_context=context) as pool:
        by_index = dict(pool.map(_play_shard, jobs))
    return [game for index in range(len(jobs)) for game in by_index[index]]


def _merge_shard_results(temp_paths: Sequence[Path], results_path: Path) -> None:
    """Append each shard's JSONL to the final file in schedule order, then unlink.

    Called from a ``finally`` so a worker crash still keeps the finished games
    of the surviving shards; appending (never truncating) mirrors the serial
    writer's resume-friendly contract.
    """
    with results_path.open("ab") as final:
        for temp in temp_paths:
            if temp.exists():
                final.write(temp.read_bytes())
                temp.unlink()


def _play_games_range(
    schedule: Sequence[ScheduledGame],
    entrants: Sequence[Entrant],
    *,
    rules: Rules,
    factory: Callable[[str, Rules, int], Policy],
    out_path: Path | None,
    results_path: Path | None,
    stamp: str | None,
    index_offset: int = 0,
) -> list[PlayedGame]:
    """The original single-process loop, shared by the serial and shard paths.

    ``index_offset`` shifts only the trace index, so a shard writes the same
    ``g0012__...`` name the serial run would have written for that slot.
    """
    by_id = {entrant.id: entrant for entrant in entrants}
    results_handle = None
    if results_path is not None:
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_handle = results_path.open("a", encoding="utf-8")

    results: list[PlayedGame] = []
    try:
        for index, game in enumerate(schedule):
            policies = [
                factory(by_id[id_].spec, rules, policy_seed(game.seed, seat))
                for seat, id_ in enumerate(game.seats)
            ]
            human = game.seats.index(game.subject)
            opponent = game.seats[1 - human]
            labels = [
                player_label(
                    "subject"
                    if seat == human
                    else ("anchor" if by_id[id_].is_anchor else "candidate"),
                    id_,
                    seat,
                )
                for seat, id_ in enumerate(game.seats)
            ]
            recorded = play_recorded(
                policies,
                rules=rules,
                seed=game.seed,
                human_seat=human,
                players=labels,
                created_at=stamp or "",
                trace_dir=out_path / game.subject if out_path is not None else None,
                trace_index=index_offset + index,
                opponent=opponent,
                print_fn=_silent,
            )
            played = PlayedGame(
                seed=recorded.seed, seats=game.seats, scores=recorded.scores
            )
            results.append(played)
            if results_handle is not None:
                line = {
                    "seed": recorded.seed,
                    "seats": list(game.seats),
                    "scores": list(recorded.scores),
                    "subject": game.subject,
                    "opponent": opponent,
                    "subject_seat": human,
                    "kind": "anchor" if by_id[opponent].is_anchor else "cross",
                }
                results_handle.write(
                    json.dumps(line, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
                results_handle.flush()  # a crash keeps every finished game
    finally:
        if results_handle is not None:
            results_handle.close()
    return results


def _warn_cuda_workers(workers: int, device: str) -> None:
    """Warn that spawn workers each open a CUDA context (GPU-memory risk)."""
    if workers > 1 and device.startswith("cuda"):
        print(
            f"warning: workers={workers} with device={device!r}: every spawn "
            "worker opens its own CUDA context, multiplying GPU memory use; "
            "lower --workers or use --device cpu if you hit OOM",
            file=sys.stderr,
        )


def _play_games_parallel(
    schedule: tuple[ScheduledGame, ...],
    entrants: tuple[Entrant, ...],
    *,
    workers: int,
    rules: Rules,
    factory: Callable[[str, Rules, int], Policy],
    out_path: Path | None,
    results_path: Path | None,
    stamp: str | None,
    device: str,
) -> list[PlayedGame]:
    """Shard, spawn and merge; see :func:`play_games` for the contract."""
    n_shards = min(workers, len(schedule))
    chunks = split_schedule(schedule, n_shards)
    _warn_cuda_workers(n_shards, device)
    try:
        pickle.dumps(factory)
    except Exception as exc:  # defensive: spawn re-imports the factory by reference
        raise ValueError(
            "workers>1 needs a picklable factory (spawn workers import it by "
            "reference); pass workers=1 or a module-level function"
        ) from exc
    offsets: list[int] = []
    total = 0
    for chunk in chunks:
        offsets.append(total)
        total += len(chunk)
    temp_paths: tuple[Path, ...] = ()
    if results_path is not None:
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_path.touch()  # serial opens the file before the first game too
        temp_paths = tuple(
            results_path.with_name(f".{results_path.name}.shard_{index:05d}.tmp")
            for index in range(n_shards)
        )
        for temp in temp_paths:
            temp.unlink(missing_ok=True)  # a crashed run must not leak old rows
    jobs = tuple(
        _ShardJob(
            index=index,
            offset=offsets[index],
            schedule=chunks[index],
            entrants=entrants,
            rules=rules,
            factory=factory,
            out=str(out_path) if out_path is not None else None,
            results_out=str(temp_paths[index]) if temp_paths else None,
            created_at=stamp,
            device=device,
        )
        for index in range(n_shards)
    )
    try:
        return _run_shards(jobs)
    finally:
        if results_path is not None:
            _merge_shard_results(temp_paths, results_path)


def play_games(
    schedule: Sequence[ScheduledGame],
    entrants: Sequence[Entrant],
    *,
    rules: Rules = DEFAULT_RULES,
    factory: Callable[[str, Rules, int], Policy] | None = None,
    out: str | Path | None = None,
    results_out: str | Path | None = None,
    created_at: str | None = None,
    device: str = "cpu",
    workers: int = 1,
) -> list[PlayedGame]:
    """Play every scheduled 牌局, optionally saving one trace per game.

    Traces follow ``trace.py`` exactly (same recorder as ``7g523-play``): the
    subject seat becomes ``human_seat``, so the D1 tool's S1 features read a
    ladder trace the same way they read a human one.  ``created_at`` is injected
    so deterministic tests can pin it; when omitted and ``out`` is given, the
    wall clock supplies it.

    ``results_out`` names a JSONL file that receives one line per game, appended
    as soon as the game finishes (audit §6.2 #2).  Each line is
    ``{"seed", "seats", "scores", "subject", "opponent", "subject_seat",
    "kind"}`` with ``kind`` ``"anchor"`` when the subject's opponent is pinned
    and ``"cross"`` otherwise — enough to re-pair twins, bootstrap by deal, or
    join against the fitting schedule later.  ``None`` (the default) keeps the
    old behaviour: nothing is written.

    ``workers > 1`` shards the schedule into contiguous chunks played by spawn
    processes and merged back into schedule order.  Per-game policy seeds depend
    only on the game, so the returned list is identical for every ``workers``.
    Trace names keep the *global* schedule index (each job carries its offset);
    ``results_out`` is written per shard and merged into the one append-mode
    JSONL at the end, byte-for-byte the serial output.  ``workers=1`` (the
    default) runs the original single-process loop and never starts a pool.
    """
    if workers < 1:
        raise ValueError(f"workers must be at least 1, got {workers}")
    schedule = tuple(schedule)
    entrants = tuple(entrants)
    out_path = Path(out) if out is not None else None
    results_path = Path(results_out) if results_out is not None else None
    stamp = created_at
    if out_path is not None and stamp is None:
        stamp = datetime.now().isoformat(timespec="seconds")
    if factory is None:
        factory = make_factory(device=device)
    if workers == 1 or len(schedule) <= 1:
        return _play_games_range(
            schedule,
            entrants,
            rules=rules,
            factory=factory,
            out_path=out_path,
            results_path=results_path,
            stamp=stamp,
        )
    return _play_games_parallel(
        schedule,
        entrants,
        workers=workers,
        rules=rules,
        factory=factory,
        out_path=out_path,
        results_path=results_path,
        stamp=stamp,
        device=device,
    )


def build_ladder(
    entrants: Sequence[Entrant],
    *,
    games_per_anchor: int = 100,
    cross: int = 0,
    seed: int = 0,
    rules: Rules = DEFAULT_RULES,
    out: str | Path | None = None,
    games_out: str | Path | None = None,
    created_at: str | None = None,
    count: int = 5,
    min_spacing: float = 100.0,
    max_spacing: float = 150.0,
    window: int | None = None,
    config: FitConfig = FitConfig(),
    factory: Callable[[str, Rules, int], Policy] | None = None,
    device: str = "cpu",
    workers: int = 1,
) -> Ladder:
    """Plan, play, fit and select in one call — the default M2 path.

    Requires at least two pinned anchors (the plan's 秤砣 requirement); a
    candidate's ``prior`` (e.g. its parent model's rating) warm-starts the fit
    as data.  Pass ``out`` to write the study traces, ``None`` to fit silently;
    pass ``games_out`` for the per-game JSONL (audit §6.2 #2) independently of
    ``out`` — e.g. a ``--no-traces`` screening still keeps every result.
    ``workers`` is forwarded to :func:`play_games` (default 1, serial).
    """
    entrants = tuple(entrants)
    anchors = [entrant for entrant in entrants if entrant.is_anchor]
    if len(anchors) < 2:
        raise ValueError(
            "the plan requires at least two anchors (Random/Greedy or rated "
            "checkpoints); pass --anchor for a second one"
        )
    schedule = plan_games(
        entrants, games_per_anchor=games_per_anchor, cross=cross, seed=seed
    )
    played = play_games(
        schedule,
        entrants,
        rules=rules,
        factory=factory,
        out=out,
        results_out=games_out,
        created_at=created_at,
        device=device,
        workers=workers,
    )
    fixed = {entrant.id: entrant.pinned for entrant in anchors if entrant.pinned is not None}
    priors: dict[str, Prior] = {
        entrant.id: entrant.prior
        for entrant in entrants
        if not entrant.is_anchor and entrant.prior is not None
    }
    fit = fit_ratings(played, anchors=fixed, priors=priors, window=window, config=config)
    candidates = {
        entrant.id: fit.ratings[entrant.id] for entrant in entrants if not entrant.is_anchor
    }
    selection = select_rungs(
        candidates, count=count, min_spacing=min_spacing, max_spacing=max_spacing
    )
    return Ladder(entrants, fit, selection, schedule)

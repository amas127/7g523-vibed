"""One 牌局 → trace + scores: the single batch-recording seam.

``ladder``, ``placement`` and the trace-prior feature extractor all need the same
three steps — deal and play a game through :func:`seven523.play.play_game`,
assemble the trace with :func:`seven523.trace.build_trace`, and save it with
:func:`seven523.trace.save_trace`.  Before this module each harness wrote that
pipeline by hand and drifted: the per-seat policy seed was derived from the
schedule *index* in one place and from the *seat* in another, and the trace
destination and filename conventions were re-typed per caller.

:func:`play_recorded` is now the only batch-recording path.  It owns the seed
helper (:func:`policy_seed`), the record → trace assembly and the on-disk
naming, while callers keep what is genuinely theirs: the policy roster, the
player labels, the schedule index and the JSONL/result bookkeeping.  The trace
format itself still lives in :mod:`seven523.trace`; ``7g523-play``'s
interactive ``--save-trace`` writes a user-named file and stays out of scope.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from .policies import Policy
from .play import play_game
from .rules import DEFAULT_RULES, Rules
from .trace import build_trace, save_trace, trace_filename

__all__ = [
    "RecordedGame",
    "play_recorded",
    "policy_seed",
]


@dataclass(frozen=True, slots=True)
class RecordedGame:
    """The outcome of one recorded 牌局: scores, trace document and its path."""

    seed: int
    human_seat: int
    scores: tuple[int, ...]
    trace: dict[str, Any]
    trace_path: Path | None


def policy_seed(seed: int, seat: int) -> int:
    """The one per-seat policy seed: ``(seed + 101 * (seat + 1)) & 0xFFFF_FFFF``.

    Deriving from the *seat* (never the schedule index) keeps a deal's policies
    independent of where the game sits in a schedule, so sharded and serial
    runs — and twin games with the seats swapped — build the same policy roster.
    """
    return (seed + 101 * (seat + 1)) & 0xFFFF_FFFF


def play_recorded(
    policies: Sequence[Policy],
    *,
    rules: Rules = DEFAULT_RULES,
    seed: int | None = None,
    human_seat: int = 0,
    players: Sequence[str] | None = None,
    created_at: str | None = None,
    trace_dir: Path | None = None,
    trace_index: int = 0,
    opponent: str = "",
    chooser: Callable[..., Any] | None = None,
    print_fn: Callable[..., None] = print,
) -> RecordedGame:
    """Drive one game through :func:`play_game`, then build (+ optionally save) its trace.

    ``play_recorded`` is the only batch-recording path: it fills the record,
    builds the trace with :func:`~seven523.trace.build_trace`, and saves it as
    ``trace_dir / trace_filename(trace_index, seed, human_seat, opponent)``.
    ``trace_dir`` with a missing ``players`` or ``opponent`` raises
    :class:`ValueError`.  ``trace_path`` is :data:`None` when ``trace_dir`` is
    :data:`None` (the game is still played and the record still built).
    """
    if trace_dir is not None and (not players or not opponent):
        raise ValueError("saving a trace needs players and an opponent")
    record: dict[str, Any] = {}
    play_game(
        policies,
        chooser=chooser,
        rules=rules,
        human_seat=human_seat,
        seed=seed,
        print_fn=print_fn,
        record=record,
    )
    scores = tuple(int(value) for value in record["final_scores"])
    trace = build_trace(
        rules,
        seed=seed,
        human_seat=human_seat,
        players=list(players) if players is not None else [],
        created_at=created_at or "",
        **record,
    )
    trace_path: Path | None = None
    if trace_dir is not None:
        trace_path = save_trace(
            trace_dir / trace_filename(trace_index, seed, human_seat, opponent),
            trace,
        )
    return RecordedGame(
        seed=seed,  # type: ignore[arg-type]
        human_seat=human_seat,
        scores=scores,
        trace=trace,
        trace_path=trace_path,
    )

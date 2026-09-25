"""Game traces: the on-disk format for a saved 对局.

A trace is the opening :class:`~seven523.game.Deal` plus one entry per move and
the resulting public scores.  This module owns the *format*: serialising cards,
rules and the deal; the per-step record (:func:`step_record`); the player
identity and study-file naming conventions; and reading/writing JSON.  Terminal
replay lives in :mod:`seven523.play`, which drives the recorded actions back
through :class:`~seven523.match.Match` and raises on any divergence, so a trace
doubles as an integrity check of the engine.

Keeping the codec here means the format version, every field name and every
naming convention live in one module, and it can be tested without a terminal or
torch.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .cards import CARD_ORDER, card_id, sorted_cards
from .game import Deal, Game, GameState, StepResult
from .rules import Rules

__all__ = [
    "TRACE_VERSION",
    "build_trace",
    "card_from_json",
    "card_json",
    "deal_from_json",
    "deal_json",
    "initial_snapshot",
    "load_trace",
    "parse_player_label",
    "player_label",
    "rules_from_json",
    "rules_json",
    "save_trace",
    "state_from_snapshot",
    "step_record",
    "trace_filename",
]

TRACE_VERSION = 2


# -- cards -------------------------------------------------------------------


def card_json(card) -> dict[str, Any]:
    return {"id": card_id(card), "label": str(card)}


def card_from_json(data: dict[str, Any]):
    return CARD_ORDER[int(data["id"])]


# -- rules -------------------------------------------------------------------


def rules_json(rules: Rules) -> dict[str, Any]:
    return asdict(rules)


def rules_from_json(data: dict[str, Any]) -> Rules:
    # Traces recorded while ``Rules.comparison`` existed carry a stale key;
    # family comparison is now the only semantics, so drop it instead of
    # rejecting the record (and never mutate the caller's dict).
    return Rules(**{key: value for key, value in data.items() if key != "comparison"})


# -- the opening deal --------------------------------------------------------


def deal_json(deal: Deal) -> dict[str, Any]:
    """Serialise a deal: hands, draw pile, reveals and the starter."""
    return {
        "starter": deal.starter,
        "hands": [
            [card_json(card) for card in sorted_cards(hand)] for hand in deal.hands
        ],
        "draw_pile": [card_json(card) for card in deal.draw_pile],
        "revealed": [card_json(card) for card in deal.revealed],
    }


def deal_from_json(data: dict[str, Any]) -> Deal:
    return Deal(
        hands=tuple(
            frozenset(card_from_json(card) for card in hand) for hand in data["hands"]
        ),
        draw_pile=tuple(card_from_json(card) for card in data["draw_pile"]),
        revealed=tuple(card_from_json(card) for card in data["revealed"]),
        starter=int(data["starter"]),
    )


def initial_snapshot(state: GameState) -> dict[str, Any]:
    """The opening state as a serialisable deal (valid before the first move)."""
    return deal_json(Deal.from_state(state))


def state_from_snapshot(snapshot: dict[str, Any], rules: Rules) -> GameState:
    """Rebuild the opening :class:`GameState` recorded by :func:`initial_snapshot`."""
    return Game(rules).restore(deal_from_json(snapshot))


# -- players, steps and file names -------------------------------------------


def player_label(role: str, player_id: str, seat: int) -> str:
    """The trace's player identity: ``role:id@seatN`` (e.g. ``anchor:greedy@seat1``)."""
    return f"{role}:{player_id}@seat{seat}"


def parse_player_label(label: str) -> tuple[str | None, str, int | None]:
    """Best-effort inverse of :func:`player_label` → ``(role, id, seat)``.

    Labels written by hand (``human@seat0``, ``贪心 bot``) parse with ``role``
    and/or ``seat`` set to ``None``, so a new convention never crashes a reader.
    """
    role: str | None = None
    rest = label
    if ":" in label:
        role, rest = label.split(":", 1)
    player_id, _, seat_text = rest.partition("@seat")
    return role, player_id, int(seat_text) if seat_text.isdigit() else None


def step_record(
    seat: int,
    action_id: int,
    suit: int | None,
    *,
    text: str,
    state: GameState,
    result: StepResult,
) -> dict[str, Any]:
    """One trace step, recorded *after* the move; the field names live here."""
    return {
        "seat": seat,
        "action": action_id,
        "suit": suit,
        "text": text,
        "scores": list(state.scores),
        "hand_sizes": [len(hand) for hand in state.hands],
        "draw_count": len(state.draw_pile),
        "trick_over": result.trick_over,
        "winner": result.winner,
        "points": result.points_taken,
        "dug": result.dug,
    }


def trace_filename(index: int, seed: int, human_seat: int, opponent: str) -> str:
    """The study-file convention: ``g0007__s42__seat1__vsgreedy.json``."""
    return f"g{index:04d}__s{seed}__seat{human_seat}__vs{opponent}.json"


# -- trace documents ---------------------------------------------------------


def build_trace(
    rules: Rules,
    *,
    seed: int | None,
    human_seat: int,
    players: list[str],
    created_at: str,
    **record: Any,
) -> dict[str, Any]:
    """Assemble a complete trace document around ``record`` (initial/steps/...)."""
    return {
        "version": TRACE_VERSION,
        "created_at": created_at,
        "rules": rules_json(rules),
        "seed": seed,
        "human_seat": human_seat,
        "players": list(players),
        **record,
    }


def save_trace(path: str | Path, trace: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(trace, ensure_ascii=False, indent=2))
    return path


def load_trace(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())

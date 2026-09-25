"""Game traces: the on-disk format for a saved 对局.

A trace is the opening :class:`~seven523.game.Deal` plus one entry per move and
the resulting public scores.  This module owns only the *format* — serialising
cards, rules and the deal, and reading/writing JSON.  Terminal replay lives in
:mod:`seven523.play`, which drives the recorded actions back through
:class:`~seven523.match.Match` and raises on any divergence, so a trace doubles
as an integrity check of the engine.

Keeping the codec here means the format version and every field name live in one
module, and it can be tested without a terminal or torch.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .cards import CARD_ORDER, card_id, sorted_cards
from .game import Deal, Game, GameState
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
    "rules_from_json",
    "rules_json",
    "save_trace",
    "state_from_snapshot",
]

TRACE_VERSION = 1


# -- cards -------------------------------------------------------------------


def card_json(card) -> dict[str, Any]:
    return {"id": card_id(card), "label": str(card)}


def card_from_json(data: dict[str, Any]):
    return CARD_ORDER[int(data["id"])]


# -- rules -------------------------------------------------------------------


def rules_json(rules: Rules) -> dict[str, int]:
    return asdict(rules)


def rules_from_json(data: dict[str, Any]) -> Rules:
    return Rules(**data)


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

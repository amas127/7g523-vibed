"""The browser's JSON projection of one game — the only place cards become JSON.

The engine has one projection seam for agents (:meth:`Game.view`, ADR-0002);
the browser needs a second, richer one: concrete cards resolved from action
templates, human-readable combo labels, and the public-information card counter.
This module owns every field name the page reads, and nothing else — pure
functions over engine objects, no HTTP and no session state.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

from ..actions import legal_ids, resolve, suit_options
from ..cards import (
    CARD_ORDER,
    NATURAL_INDEX,
    NATURAL_ORDER,
    RANK_LABELS,
    SUIT_LABELS,
    Rank,
    Suit,
    card_key,
    point_value,
)
from ..combos import Combo, ComboKind, classify
from ..game import Game, Play, View

__all__ = [
    "DECK_TOTALS",
    "KIND_NAMES",
    "POINTS_BY_RANK",
    "action_json",
    "card_json",
    "combo_json",
    "combo_label",
    "counter_json",
    "hand_key",
    "legal_actions",
    "ordered_cards",
    "play_json",
    "seat_names",
]

#: Full-deck facts derived from the engine's own card order, not re-typed.
DECK_TOTALS: Counter = Counter(card.rank for card in CARD_ORDER)
POINTS_BY_RANK: dict[Rank, int] = {
    card.rank: point_value(card) for card in CARD_ORDER if point_value(card)
}

KIND_NAMES: dict[ComboKind, str] = {
    ComboKind.SINGLE: "单牌",
    ComboKind.PAIR: "对子",
    ComboKind.STRAIGHT: "顺子",
    ComboKind.CONSECUTIVE_PAIRS: "连对",
    ComboKind.SMALL_BOMB: "小炸弹",
    ComboKind.BIG_BOMB: "大炸弹",
}
KIND_ORDER: dict[ComboKind, int] = {
    ComboKind.SINGLE: 0,
    ComboKind.PAIR: 1,
    ComboKind.STRAIGHT: 2,
    ComboKind.CONSECUTIVE_PAIRS: 3,
    ComboKind.SMALL_BOMB: 4,
    ComboKind.BIG_BOMB: 5,
}
KIND_BY_NAME = {kind.name.lower(): kind for kind in ComboKind}


def card_json(card: Any) -> dict[str, Any]:
    """One card for the browser: rank label, suit glyph, point value."""
    return {
        "rank": RANK_LABELS[card.rank],
        "suit": None if card.suit is None else SUIT_LABELS[card.suit],
        "label": str(card),
        "order": int(card.rank),
        "natural": NATURAL_INDEX.get(card.rank, -1),
        "points": point_value(card),
        "red": card.suit in (Suit.HEART, Suit.DIAMOND),
        "joker": card.suit is None,
    }


def _sequence_order(cards: tuple[Any, ...]) -> tuple[Any, ...]:
    """Read a straight/连对 along the 13-rank cycle, starting where the run
    actually starts (``Q-K-A-2-3``, not ``3-Q-K-A-2``)."""
    ranks = {card.rank for card in cards}
    predecessors = {
        rank: NATURAL_ORDER[(NATURAL_INDEX[rank] - 1) % len(NATURAL_ORDER)]
        for rank in ranks
    }
    starts = [rank for rank in ranks if predecessors[rank] not in ranks]
    start = min(starts, key=lambda rank: NATURAL_INDEX[rank]) if starts else min(
        ranks, key=lambda rank: NATURAL_INDEX[rank]
    )
    ordered: list[Any] = []
    cursor = NATURAL_INDEX[start]
    while len(ordered) < len(cards):
        rank = NATURAL_ORDER[cursor % len(NATURAL_ORDER)]
        if rank in ranks:
            ordered.extend(sorted((c for c in cards if c.rank == rank), key=card_key))
        cursor += 1
    return tuple(ordered)


def ordered_cards(combo: Combo) -> tuple[Any, ...]:
    """Display order: natural sequence for 顺子/连对, point order otherwise."""
    if combo.kind in (ComboKind.STRAIGHT, ComboKind.CONSECUTIVE_PAIRS):
        return _sequence_order(combo.cards)
    return tuple(sorted(combo.cards, key=card_key))


def combo_label(combo: Combo) -> str:
    """One human-readable line for a combo, e.g. ``顺子 5-6-7`` / ``王炸``."""
    if (
        combo.kind is ComboKind.SMALL_BOMB
        and len(combo.cards) == 2
        and all(card.suit is None for card in combo.cards)
    ):
        name = "王炸"
    else:
        name = KIND_NAMES[combo.kind]
    if combo.kind in (ComboKind.STRAIGHT, ComboKind.CONSECUTIVE_PAIRS):
        ranks = "-".join(RANK_LABELS[card.rank] for card in ordered_cards(combo))
        return f"{name} {ranks}"
    return f"{name} " + " ".join(str(card) for card in ordered_cards(combo))


def combo_json(combo: Combo) -> dict[str, Any]:
    """The combo fields shared by plays, the incumbent and legal actions."""
    top = combo.top_card
    return {
        "kind": combo_label(combo).split(" ", 1)[0],
        "kind_key": combo.kind.name.lower(),
        "label": combo_label(combo),
        "cards": [card_json(card) for card in ordered_cards(combo)],
        "count": len(combo.cards),
        "bomb": combo.kind in (ComboKind.SMALL_BOMB, ComboKind.BIG_BOMB),
        "top_order": int(top.rank),
        "top_suit": -1 if top.suit is None else int(top.suit),
    }


def play_json(play: Play, rules) -> dict[str, Any]:
    """One public action-log entry; a pass has no cards."""
    cards = tuple(play.cards)
    result: dict[str, Any] = {
        "seat": play.seat,
        "pass": not cards,
        "opens_trick": bool(play.opens_trick),
        "went_out": bool(play.went_out),
        "cards": [],
        "label": "过",
        "kind_key": "pass",
    }
    if cards:
        combo = classify(cards, rules)
        if combo is not None:
            result.update(combo_json(combo))
    return result


def action_json(game: Game, view: View, action_id: int) -> dict[str, Any]:
    """One legal template resolved to concrete cards, plus its suit options."""
    action = game.catalog[action_id]
    if action.is_pass:
        return {
            "action_id": action_id,
            "kind": "过",
            "kind_key": "pass",
            "label": "过",
            "cards": [],
            "count": 0,
            "bomb": False,
            "suits": [],
            "top_order": -1,
            "top_suit": -1,
        }
    combo = resolve(action, view.hand, game.rules, suit=None)
    assert combo is not None, "catalog action must resolve against a legal hand"
    data = combo_json(combo)
    data["action_id"] = action_id
    data["suits"] = [
        {"value": int(suit), "label": SUIT_LABELS[suit]}
        for suit in suit_options(action, view.hand, game.rules)
    ]
    return data


def _action_sort_key(item: dict[str, Any]) -> tuple[int, int, int, int]:
    if item["kind_key"] == "pass":
        return (9, 0, -1, -1)
    kind = KIND_BY_NAME.get(item["kind_key"], ComboKind.SINGLE)
    return (KIND_ORDER[kind], item["count"], item["top_order"], item["top_suit"])


def legal_actions(game: Game, view: View) -> list[dict[str, Any]]:
    """Every legal action, weakest first, PASS last (the UI's display order)."""
    items = [action_json(game, view, action_id) for action_id in legal_ids(view.mask)]
    items.sort(key=_action_sort_key)
    return items


def counter_json(view: View) -> dict[str, Any]:
    """Public-information card counter: mine / played / unseen per rank.

    ``unseen`` is how many cards of that rank are still in the draw pile or an
    opponent's hand.  It is derived only from public facts — own hand, the
    public play log, and every seat's opening reveal — so it is exactly what a
    human could count at the table (set union by physical card, no double
    counting a revealed card that was later played).
    """
    played_cards = [card for play in view.plays for card in play.cards]
    seen = set(view.hand) | set(played_cards) | set(view.revealed)
    mine_counts = Counter(card.rank for card in view.hand)
    played_counts = Counter(card.rank for card in played_cards)
    seen_counts = Counter(card.rank for card in seen)
    rows: list[dict[str, Any]] = []
    unseen_cards = 0
    unseen_points = 0
    for rank in Rank:
        total = DECK_TOTALS[rank]
        unseen = total - seen_counts.get(rank, 0)
        points = POINTS_BY_RANK.get(rank, 0)
        unseen_cards += unseen
        unseen_points += unseen * points
        rows.append(
            {
                "rank": RANK_LABELS[rank],
                "total": total,
                "mine": mine_counts.get(rank, 0),
                "played": played_counts.get(rank, 0),
                "unseen": unseen,
                "points": points,
            }
        )
    return {
        "rows": rows,
        "unseen_cards": unseen_cards,
        "unseen_points": unseen_points,
    }


def hand_key(card: Any) -> tuple[int, int]:
    """Display order for the human hand: 自然序（3…2）ascending, jokers last.

    Natural order (RULES.md §1.5) is the straight cycle, not the point order —
    the hand reads 3 4 5 6 7 8 9 10 J Q K A 2 小王 大王, with suits ascending
    (♦ ♣ ♥ ♠) when a rank repeats.
    """
    natural = NATURAL_INDEX.get(card.rank)
    order = natural if natural is not None else (13 if card.rank is Rank.SMALL_JOKER else 14)
    return (order, -1 if card.suit is None else int(card.suit))


def seat_names(
    human_seat: int, num_players: int, opponent_id: str
) -> list[str]:
    """Display names per seat: ``你`` for the human, the rung id otherwise."""
    names: list[str] = []
    for seat in range(num_players):
        if seat == human_seat:
            names.append("你")
        elif num_players == 2:
            names.append(opponent_id)
        else:
            names.append(f"{opponent_id}·{seat}")
    return names

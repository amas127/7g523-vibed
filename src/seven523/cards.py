"""Cards, ranks, suits and the standard 54-card deck.

Rank strength is the game's point order (RULES.md §1.2), ascending::

    4 < 6 < 8 < 9 < 10 < J < Q < K < A < 3 < 2 < 5 < 小王 < 大王 < 7

Straight adjacency uses a *different* order, the 13-rank natural cycle
``3-4-5-6-7-8-9-10-J-Q-K-A-2`` (RULES.md §1.5).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable


class Suit(IntEnum):
    """Suit strength, ascending: 方片 < 梅花 < 红心 < 黑桃."""

    DIAMOND = 0
    CLUB = 1
    HEART = 2
    SPADE = 3


class Rank(IntEnum):
    """Point strength, ascending (RULES.md §1.2)."""

    R4 = 0
    R6 = 1
    R8 = 2
    R9 = 3
    R10 = 4
    RJ = 5
    RQ = 6
    RK = 7
    RA = 8
    R3 = 9
    R2 = 10
    R5 = 11
    SMALL_JOKER = 12
    BIG_JOKER = 13
    R7 = 14


JOKER_RANKS: frozenset[Rank] = frozenset({Rank.SMALL_JOKER, Rank.BIG_JOKER})
STANDARD_RANKS: tuple[Rank, ...] = tuple(r for r in Rank if r not in JOKER_RANKS)

#: Natural order for straights / consecutive pairs — a 13-rank cycle (RULES.md §1.5).
NATURAL_ORDER: tuple[Rank, ...] = (
    Rank.R3,
    Rank.R4,
    Rank.R5,
    Rank.R6,
    Rank.R7,
    Rank.R8,
    Rank.R9,
    Rank.R10,
    Rank.RJ,
    Rank.RQ,
    Rank.RK,
    Rank.RA,
    Rank.R2,
)
NATURAL_INDEX: dict[Rank, int] = {rank: i for i, rank in enumerate(NATURAL_ORDER)}
RANK_INDEX: dict[Rank, int] = {rank: i for i, rank in enumerate(Rank)}

RANK_LABELS: dict[Rank, str] = {
    Rank.R4: "4",
    Rank.R6: "6",
    Rank.R8: "8",
    Rank.R9: "9",
    Rank.R10: "10",
    Rank.RJ: "J",
    Rank.RQ: "Q",
    Rank.RK: "K",
    Rank.RA: "A",
    Rank.R3: "3",
    Rank.R2: "2",
    Rank.R5: "5",
    Rank.SMALL_JOKER: "小王",
    Rank.BIG_JOKER: "大王",
    Rank.R7: "7",
}
SUIT_LABELS: dict[Suit, str] = {
    Suit.DIAMOND: "♦",
    Suit.CLUB: "♣",
    Suit.HEART: "♥",
    Suit.SPADE: "♠",
}


@dataclass(frozen=True, slots=True)
class Card:
    """One physical card; jokers have ``suit is None``."""

    rank: Rank
    suit: Suit | None = None

    def __post_init__(self) -> None:
        if (self.suit is None) != (self.rank in JOKER_RANKS):
            raise ValueError(f"jokers have no suit and non-jokers need one: {self!r}")

    @property
    def is_joker(self) -> bool:
        return self.rank in JOKER_RANKS

    def __str__(self) -> str:
        if self.is_joker:
            return RANK_LABELS[self.rank]
        return f"{SUIT_LABELS[self.suit]}{RANK_LABELS[self.rank]}"


def card_key(card: Card) -> tuple[int, int]:
    """Total order over cards: ``(point rank, suit)``; jokers use suit ``-1``."""
    return (int(card.rank), -1 if card.suit is None else int(card.suit))


def sorted_cards(cards: Iterable[Card]) -> tuple[Card, ...]:
    return tuple(sorted(cards, key=card_key))


def make_deck() -> tuple[Card, ...]:
    """The standard 54-card deck in a deterministic order (13×4 + 大王 + 小王)."""
    return tuple(Card(rank, suit) for rank in STANDARD_RANKS for suit in Suit) + (
        Card(Rank.SMALL_JOKER),
        Card(Rank.BIG_JOKER),
    )


CARD_ORDER: tuple[Card, ...] = make_deck()
CARD_INDEX: dict[Card, int] = {card: i for i, card in enumerate(CARD_ORDER)}
NUM_CARDS: int = len(CARD_ORDER)

POINT_VALUE: dict[Rank, int] = {Rank.R5: 5, Rank.R10: 10, Rank.RK: 10}


def point_value(card: Card) -> int:
    """Scoring value: 5 → 5, 10 → 10, K → 10, everything else → 0."""
    return POINT_VALUE.get(card.rank, 0)


def is_point_card(card: Card) -> bool:
    return card.rank in POINT_VALUE


def card_id(card: Card) -> int:
    """Stable ``0..53`` index used by the observation encoder."""
    return CARD_INDEX[card]

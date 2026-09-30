"""Combo recognition and comparison — the deepest rules module.

Only two entry points: :func:`classify` and :func:`beats`.  Every comparison
rule from RULES.md §3 lives here, so callers never re-implement "what beats
what".
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

from .cards import JOKER_RANKS, NATURAL_INDEX, Card, Rank, card_key, sorted_cards
from .rules import DEFAULT_RULES, Rules


class ComboKind(Enum):
    SINGLE = "single"
    PAIR = "pair"
    STRAIGHT = "straight"
    CONSECUTIVE_PAIRS = "consecutive_pairs"
    SMALL_BOMB = "small_bomb"
    BIG_BOMB = "big_bomb"


#: Comparison families (RULES.md §3.1): a non-bomb combo only compares against
#: combos of its own family — 单张族 ``SINGLE/STRAIGHT``, 对子族
#: ``PAIR/CONSECUTIVE_PAIRS``.  Bombs ignore families and beat everything.
FAMILY: dict[ComboKind, int] = {
    ComboKind.SINGLE: 0,
    ComboKind.STRAIGHT: 0,
    ComboKind.PAIR: 1,
    ComboKind.CONSECUTIVE_PAIRS: 1,
}

#: Flat-tier ordering (pre-2026-09-25 RULES.md).  :attr:`Combo.strength`
#: uses it to *sort* candidate plays (bombs first by tier, then rank);
#: :func:`beats` uses only the bomb hierarchy.
TIER: dict[ComboKind, int] = {
    ComboKind.SINGLE: 1,
    ComboKind.PAIR: 1,
    ComboKind.STRAIGHT: 2,
    ComboKind.CONSECUTIVE_PAIRS: 2,
    ComboKind.SMALL_BOMB: 3,
    ComboKind.BIG_BOMB: 4,
}
BOMB_KINDS = frozenset({ComboKind.SMALL_BOMB, ComboKind.BIG_BOMB})
_CYCLE = len(NATURAL_INDEX)


@dataclass(frozen=True, slots=True)
class Combo:
    """A validated play: its kind plus the concrete cards (canonically sorted)."""

    kind: ComboKind
    cards: tuple[Card, ...]

    @property
    def tier(self) -> int:
        return TIER[self.kind]

    @property
    def family(self) -> int | None:
        """Comparison family (RULES.md §3.1); ``None`` for bombs, which are exempt."""
        return FAMILY.get(self.kind)

    @property
    def size(self) -> int:
        return len(self.cards)

    @property
    def top_card(self) -> Card:
        return max(self.cards, key=card_key)

    @property
    def top_key(self) -> tuple[int, int]:
        return card_key(self.top_card)

    @property
    def top_rank(self) -> int:
        return int(self.top_card.rank)

    @property
    def is_bomb(self) -> bool:
        return self.kind in BOMB_KINDS

    @property
    def strength(self) -> tuple:
        """Ordering key for callers that sort legal plays (bots, demos)."""
        if self.is_bomb:
            return (self.tier, self.top_rank)
        return (self.tier, self.size, self.top_key)

    def __str__(self) -> str:
        return " ".join(str(card) for card in self.cards)


def _contiguous(ranks: Iterable[Rank]) -> bool:
    """True if ``ranks`` form a contiguous arc on the 13-rank natural cycle."""
    positions = {NATURAL_INDEX[rank] for rank in ranks}
    length = len(positions)
    return any(
        all((start + step) % _CYCLE in positions for step in range(length))
        for start in positions
    )


def classify(cards: Iterable[Card], rules: Rules = DEFAULT_RULES) -> Combo | None:
    """Recognise a legal combo, or return ``None`` for anything illegal.

    Total and pure: any multiset is accepted, nothing is raised.  Length caps
    (hand size) are *not* applied here — the action catalog applies them.
    """
    combo_cards = tuple(cards)
    if not combo_cards or len(set(combo_cards)) != len(combo_cards):
        return None

    counts = Counter(card.rank for card in combo_cards)
    ranks = tuple(counts)
    has_joker = any(rank in JOKER_RANKS for rank in ranks)
    n = len(combo_cards)

    if n == 1:
        return Combo(ComboKind.SINGLE, sorted_cards(combo_cards))
    if n == 2:
        if len(ranks) == 2 and all(rank in JOKER_RANKS for rank in ranks):
            return Combo(ComboKind.SMALL_BOMB, sorted_cards(combo_cards))
        if len(counts) == 1 and not has_joker:
            return Combo(ComboKind.PAIR, sorted_cards(combo_cards))
        return None
    if n == 3 and len(counts) == 1 and not has_joker:
        return Combo(ComboKind.SMALL_BOMB, sorted_cards(combo_cards))
    if n == 4 and len(counts) == 1 and not has_joker:
        return Combo(ComboKind.BIG_BOMB, sorted_cards(combo_cards))
    if has_joker:
        return None
    if n >= rules.straight_min and len(counts) == n and _contiguous(ranks):
        return Combo(ComboKind.STRAIGHT, sorted_cards(combo_cards))
    if (
        n % 2 == 0
        and n // 2 >= rules.consecutive_pairs_min
        and all(value == 2 for value in counts.values())
        and _contiguous(ranks)
    ):
        return Combo(ComboKind.CONSECUTIVE_PAIRS, sorted_cards(combo_cards))
    return None


def beats(candidate: Combo, incumbent: Combo | None, rules: Rules = DEFAULT_RULES) -> bool:
    """True iff ``candidate`` strictly beats ``incumbent`` (RULES.md §3).

    ``beats(c, None)`` is True: any legal combo may lead.  Bombs beat every
    non-bomb and compare among themselves by tier then rank.  Non-bombs only
    compare inside their family (单张: single/straight; 对子: pair/consecutive
    pairs); a cross-family play never beats.  Within a family, size comes
    first, then the top card's point order and suit.
    """
    if incumbent is None:
        return True
    if candidate.is_bomb or incumbent.is_bomb:
        if candidate.tier != incumbent.tier:
            return candidate.tier > incumbent.tier
        return candidate.top_rank > incumbent.top_rank
    if candidate.family != incumbent.family:
        return False
    return (candidate.size, candidate.top_key) > (incumbent.size, incumbent.top_key)

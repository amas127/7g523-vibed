"""The fixed, game-wide action catalog (ADR-0001).

Actions are abstract rank templates; :func:`resolve` turns one into concrete
cards.  Since ADR-0004 a second *suit head* lets the policy pick the top
card's suit: :func:`resolve` takes an optional ``suit`` and :func:`top_rank` /
:func:`suit_options` describe where that choice applies.  :func:`action_mask`
is the single legality authority shared by the environment and every bot.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

from .cards import NATURAL_ORDER, RANK_LABELS, STANDARD_RANKS, Card, Rank, Suit, card_key
from .combos import Combo, ComboKind, beats, classify
from .rules import DEFAULT_RULES, Rules

#: Size of the suit head: ``♦ ♣ ♥ ♠`` (ADR-0004).
SUIT_N: int = len(Suit)


@dataclass(frozen=True, slots=True)
class Action:
    kind: ComboKind | None
    ranks: tuple[Rank, ...] = ()
    label: str = ""

    @property
    def is_pass(self) -> bool:
        return self.kind is None

    @property
    def need(self) -> tuple[tuple[Rank, int], ...]:
        """How many cards of each rank the template requires."""
        if self.kind is ComboKind.PAIR:
            return ((self.ranks[0], 2),)
        if self.kind is ComboKind.SMALL_BOMB:
            if len(self.ranks) == 1:
                return ((self.ranks[0], 3),)
            return tuple((rank, 1) for rank in self.ranks)
        if self.kind is ComboKind.BIG_BOMB:
            return ((self.ranks[0], 4),)
        if self.kind is ComboKind.CONSECUTIVE_PAIRS:
            return tuple((rank, 2) for rank in self.ranks)
        return tuple((rank, 1) for rank in self.ranks)  # SINGLE / STRAIGHT

    def __str__(self) -> str:
        if self.is_pass:
            return "PASS"
        return self.label or "".join(RANK_LABELS[rank] for rank in self.ranks)


def build_catalog(rules: Rules = DEFAULT_RULES) -> tuple[Action, ...]:
    """Enumerate every template the game can ever offer, ending with PASS."""
    catalog: list[Action] = []

    for rank in Rank:
        catalog.append(Action(ComboKind.SINGLE, (rank,)))
    for rank in STANDARD_RANKS:
        catalog.append(Action(ComboKind.PAIR, (rank,)))

    cycle = len(NATURAL_ORDER)
    max_straight = min(rules.straight_max, cycle)
    for length in range(rules.straight_min, max_straight + 1):
        for start in range(cycle):
            ranks = tuple(NATURAL_ORDER[(start + k) % cycle] for k in range(length))
            catalog.append(Action(ComboKind.STRAIGHT, ranks))

    max_pairs = min(rules.consecutive_pairs_max, cycle // 2)
    for pairs in range(rules.consecutive_pairs_min, max_pairs + 1):
        for start in range(cycle):
            ranks = tuple(NATURAL_ORDER[(start + k) % cycle] for k in range(pairs))
            catalog.append(Action(ComboKind.CONSECUTIVE_PAIRS, ranks))

    for rank in STANDARD_RANKS:
        catalog.append(Action(ComboKind.SMALL_BOMB, (rank,)))
    catalog.append(
        Action(ComboKind.SMALL_BOMB, (Rank.SMALL_JOKER, Rank.BIG_JOKER), "王炸")
    )
    for rank in STANDARD_RANKS:
        catalog.append(Action(ComboKind.BIG_BOMB, (rank,)))

    catalog.append(Action(None, ()))
    return tuple(catalog)


@lru_cache(maxsize=None)
def catalog_for(rules: Rules) -> tuple[Action, ...]:
    return build_catalog(rules)


CATALOG: tuple[Action, ...] = catalog_for(DEFAULT_RULES)
PASS_ID: int = len(CATALOG) - 1


def index_hand(hand: Iterable[Card]) -> dict[Rank, list[Card]]:
    """Group a hand by rank, strongest suits first, ready for repeated lookups."""
    by_rank: dict[Rank, list[Card]] = {}
    for card in hand:
        by_rank.setdefault(card.rank, []).append(card)
    for pool in by_rank.values():
        pool.sort(key=card_key, reverse=True)
    return by_rank


def _resolve_indexed(
    action: Action,
    by_rank: dict[Rank, list[Card]],
    rules: Rules,
    target: Rank | None = None,
    suit: Suit | None = None,
) -> Combo | None:
    chosen: list[Card] = []
    for rank, count in action.need:
        pool = by_rank.get(rank)
        if pool is None or len(pool) < count:
            return None
        if rank == target:
            # The top rank is the only rank whose suit can decide the trick:
            # play the requested card (or the strongest) plus the weakest extras.
            preferred = pool[0]
            if suit is not None:
                held = next((card for card in pool if card.suit is suit), None)
                if held is not None:
                    preferred = held
            rest = [card for card in pool if card is not preferred]
            chosen.append(preferred)
            chosen.extend(rest[::-1][: count - 1])
            continue
        # Anywhere else suits are comparison-free, and keeping stronger cards of
        # the same rank weakly dominates: spend the weakest ones first.
        chosen.extend(pool[-count:])
    combo = classify(chosen, rules)
    if combo is None or combo.kind != action.kind:
        return None
    return combo


def top_rank(action: Action) -> Rank | None:
    """The rank whose suit decides the combo's tie-break.

    ``None`` for PASS and bombs: their comparison ignores suits entirely, so
    the suit head is inert there.
    """
    if action.kind in (None, ComboKind.SMALL_BOMB, ComboKind.BIG_BOMB):
        return None
    if action.kind in (ComboKind.SINGLE, ComboKind.PAIR):
        return action.ranks[0]
    # STRAIGHT / CONSECUTIVE_PAIRS: the top card is the max rank in point order.
    return max(action.ranks, key=int)


def suit_options(
    action: Action, hand: Iterable[Card], rules: Rules = DEFAULT_RULES
) -> tuple[Suit, ...]:
    """Top-card suits available for ``action``, strongest first (empty if inert)."""
    target = top_rank(action)
    if target is None:
        return ()
    pool = index_hand(hand).get(target, [])
    held = {card.suit for card in pool}
    return tuple(suit for suit in reversed(Suit) if suit in held)


def resolve(
    action: Action,
    hand: Iterable[Card],
    rules: Rules = DEFAULT_RULES,
    suit: Suit | int | None = None,
) -> Combo | None:
    """Select concrete cards for ``action`` from ``hand``.

    The rank structure comes from the template; suits fill it by dominance:

    * the *top rank* (the only rank that can decide the trick) plays the
      requested suit when held, else the strongest, plus the **weakest** extras;
    * every other card plays the **weakest** suit available.

    Spending weak cards first keeps stronger cards of the same rank for future
    same-rank ties, which weakly dominates every other realisation: changing a
    non-top suit never changes legality or the trick outcome, and a stronger
    card of a rank can mimic any move a weaker one could make.

    An unavailable suit falls back to the strongest realisation.  Returns
    ``None`` when the hand cannot realise the template (PASS included).
    """
    if action.kind is None:
        return None
    by_rank = index_hand(hand)
    target = top_rank(action)
    if suit is None:
        return _resolve_indexed(action, by_rank, rules, target, None)
    if target is None:  # bombs: suit is irrelevant, keep the strongest cards
        return _resolve_indexed(action, by_rank, rules)
    return _resolve_indexed(action, by_rank, rules, target, Suit(int(suit)))


def action_mask(
    hand: Iterable[Card],
    incumbent: Combo | None,
    rules: Rules = DEFAULT_RULES,
) -> int:
    """134-bit mask of every legal action for this hand against ``incumbent``.

    Legality is checked on the strongest-suit realisation: ``beats`` orders by
    ``(rank, suit)`` within a kind, so if any suit arrangement beats the
    incumbent, the strongest one does too.
    """
    catalog = catalog_for(rules)
    by_rank = index_hand(hand)
    mask = 0
    for index, action in enumerate(catalog):
        if action.is_pass:
            if incumbent is not None:
                mask |= 1 << index
            continue
        combo = _resolve_indexed(action, by_rank, rules, top_rank(action), None)
        if combo is not None and beats(combo, incumbent, rules):
            mask |= 1 << index
    return mask


def legal_ids(mask: int) -> list[int]:
    """Ascending action ids set in ``mask``."""
    ids: list[int] = []
    while mask:
        low = mask & -mask
        ids.append(low.bit_length() - 1)
        mask ^= low
    return ids


def split_action(action: object) -> tuple[int, int | None]:
    """Normalise an action from any boundary to ``(template_id, suit | None)``.

    Accepts an ``int`` (template only, strongest suits), a 1-element sequence
    (old single-head checkpoints) or a 2-element sequence / array produced by
    the ``MultiDiscrete([134, 4])`` action space.
    """
    if hasattr(action, "__len__"):
        values = list(action)  # type: ignore[arg-type]
        if len(values) == 1:
            return int(values[0]), None
        if len(values) >= 2:
            suit = values[1]
            return int(values[0]), None if suit is None else int(suit)
        raise ValueError(f"empty action: {action!r}")
    return int(action), None  # type: ignore[arg-type]

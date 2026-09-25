import random

import numpy as np
import pytest

from seven523.actions import (
    CATALOG,
    PASS_ID,
    _action_mask_cached,
    action_mask,
    build_catalog,
    catalog_for,
    index_hand,
    joint_mask_bits,
    legal_ids,
    nvec_for,
    resolve,
    resolve_indexed,
)
from seven523.cards import Card, Rank, Suit, make_deck
from seven523.combos import ComboKind, beats, classify
from seven523.rules import DEFAULT_RULES, Rules


def action_for(kind, ranks):
    for action in CATALOG:
        if action.kind is kind and action.ranks == ranks:
            return action
    raise KeyError((kind, ranks))


def test_catalog_size_and_pass():
    assert len(CATALOG) == 134
    assert PASS_ID == 133
    assert CATALOG[PASS_ID].is_pass


def test_catalog_templates_are_unique():
    seen = set()
    for action in CATALOG:
        key = (action.kind, action.ranks)
        assert key not in seen
        seen.add(key)


def test_pair_resolves_two_cards():
    hand = [
        Card(Rank.R7, Suit.SPADE),
        Card(Rank.R7, Suit.HEART),
        Card(Rank.R7, Suit.CLUB),
    ]
    combo = resolve(action_for(ComboKind.PAIR, (Rank.R7,)), hand)
    assert combo.kind is ComboKind.PAIR and combo.size == 2


def test_bombs_resolve_three_or_four_cards():
    hand = [Card(Rank.R4, suit) for suit in Suit]
    assert resolve(action_for(ComboKind.SMALL_BOMB, (Rank.R4,)), hand).size == 3
    assert resolve(action_for(ComboKind.BIG_BOMB, (Rank.R4,)), hand).size == 4


def test_consecutive_pairs_resolve_two_of_each():
    hand = [
        Card(rank, suit)
        for rank in (Rank.R3, Rank.R4, Rank.R5)
        for suit in (Suit.SPADE, Suit.HEART)
    ]
    combo = resolve(
        action_for(ComboKind.CONSECUTIVE_PAIRS, (Rank.R3, Rank.R4, Rank.R5)), hand
    )
    assert combo.kind is ComboKind.CONSECUTIVE_PAIRS and combo.size == 6


def test_resolve_prefers_strongest_suits():
    hand = [Card(Rank.R7, Suit.HEART), Card(Rank.R7, Suit.SPADE)]
    combo = resolve(action_for(ComboKind.SINGLE, (Rank.R7,)), hand)
    assert combo.top_card.suit is Suit.SPADE


def test_joker_bomb_resolves():
    hand = [Card(Rank.SMALL_JOKER), Card(Rank.BIG_JOKER)]
    combo = resolve(
        action_for(ComboKind.SMALL_BOMB, (Rank.SMALL_JOKER, Rank.BIG_JOKER)), hand
    )
    assert combo.kind is ComboKind.SMALL_BOMB


def test_leading_mask_has_no_pass():
    hand = [Card(Rank.R7, Suit.SPADE)]
    mask = action_mask(hand, None)
    assert not (mask >> PASS_ID) & 1
    assert mask != 0


def test_following_mask_always_allows_pass():
    hand = [Card(Rank.R4, Suit.SPADE)]
    incumbent = classify([Card(Rank.R7, Suit.SPADE)])
    mask = action_mask(hand, incumbent)
    assert (mask >> PASS_ID) & 1


def test_mask_matches_resolve_and_beats():
    rng = random.Random(0)
    deck = list(make_deck())
    incumbent = classify([Card(Rank.R4, Suit.DIAMOND)])
    for _ in range(100):
        rng.shuffle(deck)
        hand = deck[:7]
        mask = action_mask(hand, incumbent)
        for index, action in enumerate(CATALOG):
            if action.is_pass:
                continue
            combo = resolve(action, hand)
            expected = combo is not None and beats(combo, incumbent)
            assert bool((mask >> index) & 1) is expected


def test_legal_ids_round_trip():
    mask = (1 << 2) | (1 << 5) | (1 << 133)
    assert legal_ids(mask) == [2, 5, 133]


# -- suit head (ADR-0004) -----------------------------------------------------


def test_top_rank_of_templates():
    from seven523.actions import top_rank

    assert top_rank(action_for(ComboKind.SINGLE, (Rank.R7,))) is Rank.R7
    assert top_rank(action_for(ComboKind.PAIR, (Rank.R5,))) is Rank.R5
    straight = action_for(ComboKind.STRAIGHT, (Rank.R4, Rank.R5, Rank.R6))
    assert top_rank(straight) is Rank.R5  # point order makes 5 the top card
    assert top_rank(action_for(ComboKind.SMALL_BOMB, (Rank.R7,))) is None
    assert top_rank(CATALOG[PASS_ID]) is None


def test_suit_options_are_strongest_first_and_empty_for_bombs():
    from seven523.actions import suit_options

    hand = [Card(Rank.R7, Suit.HEART), Card(Rank.R7, Suit.CLUB)]
    assert suit_options(action_for(ComboKind.SINGLE, (Rank.R7,)), hand) == (
        Suit.HEART,
        Suit.CLUB,
    )
    assert suit_options(action_for(ComboKind.SMALL_BOMB, (Rank.R7,)), hand) == ()


def test_resolve_honours_the_requested_suit():
    hand = [
        Card(Rank.R7, Suit.SPADE),
        Card(Rank.R7, Suit.HEART),
        Card(Rank.R7, Suit.CLUB),
    ]
    action = action_for(ComboKind.SINGLE, (Rank.R7,))
    assert resolve(action, hand, suit=Suit.HEART).top_card.suit is Suit.HEART
    # an unavailable suit falls back to the strongest realisation
    assert resolve(action, hand, suit=Suit.DIAMOND).top_card.suit is Suit.SPADE


def test_pair_suit_choice_keeps_the_requested_top_card():
    hand = [
        Card(Rank.R7, Suit.SPADE),
        Card(Rank.R7, Suit.HEART),
        Card(Rank.R7, Suit.CLUB),
    ]
    combo = resolve(action_for(ComboKind.PAIR, (Rank.R7,)), hand, suit=Suit.HEART)
    assert combo.top_card.suit is Suit.HEART
    assert Card(Rank.R7, Suit.SPADE) not in combo.cards  # saved for later
    assert combo.size == 2


def test_straight_suit_choice_applies_to_the_top_rank():
    hand = [
        Card(Rank.R4, Suit.SPADE),
        Card(Rank.R5, Suit.HEART),
        Card(Rank.R5, Suit.CLUB),
        Card(Rank.R6, Suit.SPADE),
    ]
    action = action_for(ComboKind.STRAIGHT, (Rank.R4, Rank.R5, Rank.R6))
    combo = resolve(action, hand, suit=Suit.CLUB)
    assert combo.top_card.rank is Rank.R5 and combo.top_card.suit is Suit.CLUB


def test_bomb_ignores_the_suit_request():
    hand = [Card(Rank.R7, suit) for suit in Suit]
    action = action_for(ComboKind.SMALL_BOMB, (Rank.R7,))
    assert resolve(action, hand, suit=Suit.DIAMOND).cards == resolve(action, hand).cards


def test_non_top_suits_spend_weakest_first():
    hand = [
        Card(Rank.R4, Suit.SPADE),
        Card(Rank.R4, Suit.DIAMOND),
        Card(Rank.R5, Suit.HEART),
        Card(Rank.R5, Suit.DIAMOND),
        Card(Rank.R6, Suit.SPADE),
        Card(Rank.R6, Suit.DIAMOND),
    ]
    action = action_for(ComboKind.STRAIGHT, (Rank.R4, Rank.R5, Rank.R6))
    combo = resolve(action, hand)
    # top rank is 5 (point order); the non-deciding ranks spend ♦ and keep ♠
    assert combo.top_card == Card(Rank.R5, Suit.HEART)
    assert Card(Rank.R4, Suit.DIAMOND) in combo.cards
    assert Card(Rank.R6, Suit.DIAMOND) in combo.cards
    assert Card(Rank.R4, Suit.SPADE) not in combo.cards
    assert Card(Rank.R6, Suit.SPADE) not in combo.cards


def test_pair_default_keeps_the_middle_suit():
    hand = [
        Card(Rank.R7, Suit.SPADE),
        Card(Rank.R7, Suit.HEART),
        Card(Rank.R7, Suit.CLUB),
    ]
    combo = resolve(action_for(ComboKind.PAIR, (Rank.R7,)), hand)
    assert combo.top_card.suit is Suit.SPADE
    assert Card(Rank.R7, Suit.CLUB) in combo.cards  # weakest spent
    assert Card(Rank.R7, Suit.HEART) not in combo.cards  # stronger card kept


def test_small_bomb_keeps_the_strongest_card():
    hand = [Card(Rank.R7, suit) for suit in Suit]
    combo = resolve(action_for(ComboKind.SMALL_BOMB, (Rank.R7,)), hand)
    assert combo.size == 3
    assert Card(Rank.R7, Suit.SPADE) not in combo.cards


def test_split_action_accepts_every_boundary_shape():
    from seven523.actions import split_action

    assert split_action(12) == (12, None)
    assert split_action((12, 3)) == (12, 3)
    assert split_action([12]) == (12, None)
    assert split_action([12, None]) == (12, None)
    assert split_action(np.array([12, 1])) == (12, 1)


# -- catalog -----------------------------------------------------------------


def test_catalog_for_default_rules_is_the_exported_catalog():
    assert catalog_for(DEFAULT_RULES) is CATALOG
    assert len(CATALOG) == 134


def counts_by_kind(catalog):
    counts = {}
    for action in catalog:
        counts[action.kind] = counts.get(action.kind, 0) + 1
    return counts


def test_default_catalog_composition():
    counts = counts_by_kind(CATALOG)
    assert counts[ComboKind.SINGLE] == 15
    assert counts[ComboKind.PAIR] == 13
    assert counts[ComboKind.STRAIGHT] == 65  # lengths 3..7 x 13 starts
    assert counts[ComboKind.CONSECUTIVE_PAIRS] == 13
    assert counts[ComboKind.SMALL_BOMB] == 14  # 13 triples + 王炸
    assert counts[ComboKind.BIG_BOMB] == 13
    assert counts[None] == 1  # PASS


@pytest.mark.parametrize(
    "rules, expected",
    [
        (Rules(straight_max=13), 212),  # 15+13+143+13+14+13+1
        (Rules(straight_min=4), 121),  # straights 4..7
        (Rules(consecutive_pairs_max=6), 173),  # 3..6 pairs
        (Rules(straight_max=13, consecutive_pairs_max=6), 251),
    ],
)
def test_custom_rules_resize_the_catalog(rules, expected):
    catalog = build_catalog(rules)
    assert len(catalog) == expected
    assert catalog[-1].is_pass
    assert catalog_for(rules) is catalog_for(rules)
    # templates must stay unique under every configuration
    seen = {(a.kind, a.ranks) for a in catalog}
    assert len(seen) == len(catalog)


def test_action_need_templates():
    assert action_for(ComboKind.SINGLE, (Rank.R7,)).need == ((Rank.R7, 1),)
    assert action_for(ComboKind.PAIR, (Rank.R7,)).need == ((Rank.R7, 2),)
    assert action_for(ComboKind.SMALL_BOMB, (Rank.R4,)).need == ((Rank.R4, 3),)
    assert action_for(ComboKind.BIG_BOMB, (Rank.R4,)).need == ((Rank.R4, 4),)
    assert action_for(
        ComboKind.SMALL_BOMB, (Rank.SMALL_JOKER, Rank.BIG_JOKER)
    ).need == ((Rank.SMALL_JOKER, 1), (Rank.BIG_JOKER, 1))


# -- resolve corner cases ----------------------------------------------------


def test_resolve_pass_and_unrealisable_templates_return_none():
    hand = [Card(Rank.R4, Suit.SPADE)]
    assert resolve(CATALOG[PASS_ID], hand) is None
    assert resolve(action_for(ComboKind.PAIR, (Rank.R7,)), hand) is None
    assert resolve(action_for(ComboKind.SMALL_BOMB, (Rank.R4,)), hand) is None
    assert resolve(
        action_for(ComboKind.STRAIGHT, (Rank.R3, Rank.R4, Rank.R5)), hand
    ) is None


def test_resolve_on_empty_hand_never_resolves_non_pass():
    for action in CATALOG:
        if not action.is_pass:
            assert resolve(action, []) is None


def test_resolve_keeps_the_strongest_bomb_card():
    hand = [Card(Rank.R4, suit) for suit in Suit]
    small = resolve(action_for(ComboKind.SMALL_BOMB, (Rank.R4,)), hand)
    # bombs ignore suits when they compare, so spend the weakest three and keep ♠
    assert {card.suit for card in small.cards} == {Suit.DIAMOND, Suit.CLUB, Suit.HEART}
    big = resolve(action_for(ComboKind.BIG_BOMB, (Rank.R4,)), hand)
    assert set(big.cards) == set(hand)


# -- action_mask corner cases ------------------------------------------------


def test_mask_for_empty_hand_leading_is_zero():
    assert action_mask([], None) == 0


def test_mask_for_empty_hand_following_is_pass_only():
    incumbent = classify([Card(Rank.R7, Suit.SPADE)])
    mask = action_mask([], incumbent)
    assert mask == (1 << PASS_ID)


def test_mask_following_sets_pass_and_only_legal_beats():
    incumbent = classify([Card(Rank.R7, Suit.SPADE)])  # highest single
    mask = action_mask([Card(Rank.R4, Suit.SPADE)], incumbent)
    assert mask == (1 << PASS_ID)


def test_legal_ids_full_mask_is_every_action():
    assert legal_ids((1 << len(CATALOG)) - 1) == list(range(len(CATALOG)))


def test_mask_includes_joker_bomb_against_a_small_bomb():
    joker_bomb = action_for(ComboKind.SMALL_BOMB, (Rank.SMALL_JOKER, Rank.BIG_JOKER))
    hand = [Card(Rank.SMALL_JOKER), Card(Rank.BIG_JOKER)]
    incumbent = classify(
        [Card(Rank.R5, suit) for suit in (Suit.SPADE, Suit.HEART, Suit.CLUB)]
    )
    mask = action_mask(hand, incumbent)
    index = CATALOG.index(joker_bomb)
    assert (mask >> index) & 1


# -- head layout / memoisation (ADR-0004) -----------------------------------


def test_nvec_matches_catalog_and_suit_head():
    assert nvec_for(DEFAULT_RULES) == (len(CATALOG), 4)
    custom = Rules(num_players=3, straight_max=13)
    assert nvec_for(custom) == (len(catalog_for(custom)), 4)


def test_joint_mask_bits_lay_out_template_then_preference_head():
    template = (1 << 5) | (1 << PASS_ID)
    bits = joint_mask_bits(template, (len(CATALOG), 4))
    assert len(bits) == len(CATALOG) + 4
    assert bits[5] and bits[PASS_ID]
    assert not bits[0]
    assert bits[len(CATALOG):] == [True] * 4


def test_joint_mask_bits_handles_old_single_head_checkpoints():
    assert joint_mask_bits(1, (3,)) == [True, False, False]


def test_resolve_indexed_reuses_the_index_and_matches_resolve():
    hand = [Card(Rank.R7, suit) for suit in Suit]
    by_rank = index_hand(hand)
    for action in CATALOG:
        assert resolve_indexed(action, by_rank) == resolve(action, hand)


def test_action_mask_reuses_the_immutable_inputs():
    hand = frozenset([Card(Rank.R7, Suit.SPADE), Card(Rank.R7, Suit.HEART)])
    _action_mask_cached.cache_clear()
    first = action_mask(hand, None)
    second = action_mask(hand, None)
    info = _action_mask_cached.cache_info()
    assert first == second
    assert info.misses == 1 and info.hits == 1

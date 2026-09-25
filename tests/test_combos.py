import pytest

from seven523.cards import NATURAL_ORDER, Card, Rank, Suit
from seven523.combos import ComboKind, beats, classify
from seven523.rules import Rules


def cards(*specs):
    return [Card(*spec) for spec in specs]


def run(*specs):
    return classify(cards(*specs))


# -- shapes ---------------------------------------------------------------


def test_single_and_pair():
    assert run((Rank.R7, Suit.SPADE)).kind is ComboKind.SINGLE
    pair = run((Rank.R7, Suit.SPADE), (Rank.R7, Suit.HEART))
    assert pair.kind is ComboKind.PAIR
    assert pair.size == 2


def test_jokers_do_not_form_a_pair():
    assert run((Rank.SMALL_JOKER,), (Rank.BIG_JOKER,)).kind is ComboKind.SMALL_BOMB
    assert run((Rank.SMALL_JOKER,), (Rank.R7, Suit.SPADE)) is None


def test_straight_uses_the_natural_cycle():
    for ranks in (
        (Rank.RA, Rank.R2, Rank.R3),
        (Rank.RQ, Rank.RK, Rank.RA),
        (Rank.RQ, Rank.RK, Rank.RA, Rank.R2, Rank.R3),
        (Rank.R2, Rank.R3, Rank.R4),
    ):
        combo = run(*[(rank, Suit.SPADE) for rank in ranks])
        assert combo is not None and combo.kind is ComboKind.STRAIGHT


def test_non_contiguous_or_joker_straights_are_invalid():
    assert run((Rank.R3, Suit.SPADE), (Rank.R4, Suit.SPADE), (Rank.R6, Suit.SPADE)) is None
    assert run((Rank.RA, Suit.SPADE), (Rank.R2, Suit.SPADE), (Rank.SMALL_JOKER,)) is None


def test_short_straight_is_invalid():
    assert run((Rank.R3, Suit.SPADE), (Rank.R4, Suit.SPADE)) is None


def test_consecutive_pairs():
    combo = run(
        *[(rank, suit) for rank in (Rank.R3, Rank.R4, Rank.R5) for suit in (Suit.SPADE, Suit.HEART)]
    )
    assert combo.kind is ComboKind.CONSECUTIVE_PAIRS
    assert combo.size == 6


def test_duplicate_cards_are_invalid():
    assert classify([Card(Rank.R7, Suit.SPADE), Card(Rank.R7, Suit.SPADE)]) is None


# -- comparison -----------------------------------------------------------


def test_suit_tie_break():
    spade = run((Rank.R7, Suit.SPADE))
    heart = run((Rank.R7, Suit.HEART))
    assert beats(spade, heart)
    assert not beats(heart, spade)


def test_point_order_of_singles():
    seven = run((Rank.R7, Suit.DIAMOND))
    big = run((Rank.BIG_JOKER,))
    small = run((Rank.SMALL_JOKER,))
    five = run((Rank.R5, Suit.DIAMOND))
    assert beats(seven, big) and beats(big, small) and beats(small, five)


def test_beats_is_irreflexive_and_none_means_leading():
    combo = run((Rank.R7, Suit.SPADE))
    assert not beats(combo, combo)
    assert beats(combo, None)


def test_straight_beats_single_but_pair_does_not():
    single = run((Rank.R7, Suit.SPADE))
    pair = run((Rank.R7, Suit.SPADE), (Rank.R7, Suit.HEART))
    straight = run(*[(rank, Suit.SPADE) for rank in (Rank.R8, Rank.R9, Rank.R10)])
    assert beats(straight, single)
    assert not beats(pair, single)
    assert not beats(single, pair)


def test_consecutive_pairs_beat_pair():
    pair = run((Rank.R7, Suit.SPADE), (Rank.R7, Suit.HEART))
    run_pairs = run(
        *[(rank, suit) for rank in (Rank.R3, Rank.R4, Rank.R5) for suit in (Suit.SPADE, Suit.HEART)]
    )
    assert beats(run_pairs, pair)


def test_size_then_top_card():
    short = run(*[(rank, Suit.SPADE) for rank in (Rank.R3, Rank.R4, Rank.R5)])
    long = run(*[(rank, Suit.SPADE) for rank in (Rank.R3, Rank.R4, Rank.R5, Rank.R6)])
    assert beats(long, short)
    assert not beats(short, long)


def test_top_card_uses_point_order_not_sequence_top():
    straight_456 = run(*[(rank, Suit.SPADE) for rank in (Rank.R4, Rank.R5, Rank.R6)])
    straight_234 = run(*[(rank, Suit.SPADE) for rank in (Rank.R2, Rank.R3, Rank.R4)])
    assert straight_456.top_card.rank is Rank.R5
    assert straight_234.top_card.rank is Rank.R2
    assert beats(straight_456, straight_234)


# -- bombs ----------------------------------------------------------------


def test_bomb_tiers():
    triple_four = run(*[(Rank.R4, suit) for suit in (Suit.SPADE, Suit.HEART, Suit.CLUB)])
    quad_four = run(*[(Rank.R4, suit) for suit in Suit])
    seven_bomb = run(*[(Rank.R7, suit) for suit in (Suit.SPADE, Suit.HEART, Suit.CLUB)])
    straight = run(*[(rank, Suit.SPADE) for rank in (Rank.R8, Rank.R9, Rank.R10)])

    assert triple_four.kind is ComboKind.SMALL_BOMB
    assert quad_four.kind is ComboKind.BIG_BOMB
    assert beats(triple_four, straight)
    assert beats(quad_four, seven_bomb)  # tier 4 beats every small bomb


def test_small_bombs_compare_by_rank_and_joker_bomb_sits_below_seven():
    four = run(*[(Rank.R4, suit) for suit in (Suit.SPADE, Suit.HEART, Suit.CLUB)])
    five = run(*[(Rank.R5, suit) for suit in (Suit.SPADE, Suit.HEART, Suit.CLUB)])
    joker = run((Rank.SMALL_JOKER,), (Rank.BIG_JOKER,))
    seven = run(*[(Rank.R7, suit) for suit in (Suit.SPADE, Suit.HEART, Suit.CLUB)])

    assert beats(five, four)
    assert beats(joker, five)  # 王炸 > 555
    assert beats(seven, joker)  # 777 > 王炸
    assert not beats(joker, seven)


def test_big_bombs_compare_by_rank():
    quad_four = run(*[(Rank.R4, suit) for suit in Suit])
    quad_five = run(*[(Rank.R5, suit) for suit in Suit])
    assert beats(quad_five, quad_four)


@pytest.mark.parametrize("bad", [[Card(Rank.R7, Suit.SPADE)] * 2])
def test_classify_total(bad):
    assert classify(bad) is None


# -- classify corner cases ------------------------------------------------


def test_empty_input_is_not_a_combo():
    assert classify([]) is None


def test_single_jokers_are_singles():
    assert run((Rank.SMALL_JOKER,)).kind is ComboKind.SINGLE
    assert run((Rank.BIG_JOKER,)).kind is ComboKind.SINGLE


def test_every_natural_triple_is_a_straight():
    cycle = len(NATURAL_ORDER)
    for start in range(cycle):
        ranks = [NATURAL_ORDER[(start + step) % cycle] for step in range(3)]
        combo = run(*[(rank, Suit.SPADE) for rank in ranks])
        assert combo is not None and combo.kind is ComboKind.STRAIGHT, ranks


def test_straight_and_consecutive_pairs_are_not_capped_by_hand_size():
    # classify is a pure function: length caps live in the action catalog.
    straight = run(
        *[(rank, Suit.SPADE) for rank in NATURAL_ORDER[:8]]  # 3..10, 8 cards
    )
    assert straight.kind is ComboKind.STRAIGHT and straight.size == 8

    four_pairs = run(
        *[
            (rank, suit)
            for rank in (Rank.R3, Rank.R4, Rank.R5, Rank.R6)
            for suit in (Suit.SPADE, Suit.HEART)
        ]
    )
    assert four_pairs.kind is ComboKind.CONSECUTIVE_PAIRS and four_pairs.size == 8


def test_two_adjacent_pairs_are_not_a_combo():
    assert (
        run(
            (Rank.R3, Suit.SPADE),
            (Rank.R3, Suit.HEART),
            (Rank.R4, Suit.SPADE),
            (Rank.R4, Suit.HEART),
        )
        is None
    )


def test_joker_bomb_plus_a_card_is_not_a_combo():
    assert run((Rank.SMALL_JOKER,), (Rank.BIG_JOKER,), (Rank.R7, Suit.SPADE)) is None


def test_classified_cards_are_canonically_sorted():
    pair = run((Rank.R7, Suit.HEART), (Rank.R7, Suit.SPADE))
    assert pair.cards == (Card(Rank.R7, Suit.HEART), Card(Rank.R7, Suit.SPADE))
    assert pair.top_card.suit is Suit.SPADE


def test_exact_bomb_sizes():
    two = run((Rank.R7, Suit.SPADE), (Rank.R7, Suit.HEART))
    three = run(
        (Rank.R7, Suit.SPADE), (Rank.R7, Suit.HEART), (Rank.R7, Suit.CLUB)
    )
    four = run(*[(Rank.R7, suit) for suit in Suit])
    assert two.kind is ComboKind.PAIR
    assert three.kind is ComboKind.SMALL_BOMB
    assert four.kind is ComboKind.BIG_BOMB


# -- beats corner cases ---------------------------------------------------


def test_small_bombs_compare_by_rank_not_size():
    joker_bomb = run((Rank.SMALL_JOKER,), (Rank.BIG_JOKER,))  # 2 cards
    three_3 = run(*[(Rank.R3, suit) for suit in (Suit.SPADE, Suit.HEART, Suit.CLUB)])
    three_4 = run(*[(Rank.R4, suit) for suit in (Suit.SPADE, Suit.HEART, Suit.CLUB)])
    three_5 = run(*[(Rank.R5, suit) for suit in (Suit.SPADE, Suit.HEART, Suit.CLUB)])

    assert joker_bomb.size == 2 and three_3.size == 3
    assert beats(joker_bomb, three_3)  # 王炸 > 333 despite being smaller
    assert beats(joker_bomb, three_4)
    assert beats(joker_bomb, three_5)
    assert not beats(three_3, joker_bomb)


def test_big_bomb_beats_every_small_bomb():
    quad_4 = run(*[(Rank.R4, suit) for suit in Suit])
    three_7 = run(*[(Rank.R7, suit) for suit in (Suit.SPADE, Suit.HEART, Suit.CLUB)])
    joker_bomb = run((Rank.SMALL_JOKER,), (Rank.BIG_JOKER,))
    assert beats(quad_4, three_7)
    assert beats(quad_4, joker_bomb)


def test_pair_and_straight_are_incomparable():
    pair = run((Rank.R7, Suit.SPADE), (Rank.R7, Suit.HEART))
    straight = run(*[(rank, Suit.SPADE) for rank in (Rank.R8, Rank.R9, Rank.R10)])
    assert not beats(straight, pair)  # cross-family: a straight never beats a pair
    assert not beats(pair, straight)


def test_single_and_consecutive_pairs_are_incomparable():
    single = run((Rank.R7, Suit.SPADE))
    pairs = run(
        *[
            (rank, suit)
            for rank in (Rank.R3, Rank.R4, Rank.R5)
            for suit in (Suit.SPADE, Suit.HEART)
        ]
    )
    assert not beats(pairs, single)  # cross-family: pairs never beat a single
    assert not beats(single, pairs)


def test_straight_and_consecutive_pairs_are_incomparable():
    straight = run(*[(rank, Suit.SPADE) for rank in (Rank.R3, Rank.R4, Rank.R5)])
    pairs = run(
        *[
            (rank, suit)
            for rank in (Rank.R3, Rank.R4, Rank.R5)
            for suit in (Suit.SPADE, Suit.HEART)
        ]
    )
    assert not beats(straight, pairs)
    assert not beats(pairs, straight)


def test_suit_tie_breaks_pairs_and_straights():
    high_pair = run((Rank.R7, Suit.SPADE), (Rank.R7, Suit.HEART))
    low_pair = run((Rank.R7, Suit.HEART), (Rank.R7, Suit.CLUB))
    assert beats(high_pair, low_pair)
    assert not beats(low_pair, high_pair)

    high_straight = run(
        (Rank.R3, Suit.SPADE), (Rank.R4, Suit.SPADE), (Rank.R5, Suit.SPADE)
    )
    low_straight = run(
        (Rank.R3, Suit.SPADE), (Rank.R4, Suit.SPADE), (Rank.R5, Suit.HEART)
    )
    assert high_straight.top_card.rank is Rank.R5
    assert beats(high_straight, low_straight)
    assert not beats(low_straight, high_straight)


def test_straight_size_beats_a_higher_top_card():
    short = run(*[(rank, Suit.SPADE) for rank in (Rank.R3, Rank.R4, Rank.R5)])
    longer = run(
        *[(rank, Suit.SPADE) for rank in (Rank.R8, Rank.R9, Rank.R10, Rank.RJ)]
    )
    assert short.top_rank > longer.top_rank  # 5 is the highest point rank here
    assert beats(longer, short)
    assert not beats(short, longer)


def test_family_chains_are_monotone_over_representatives():
    single = run((Rank.R7, Suit.SPADE))
    straight = run(*[(rank, Suit.SPADE) for rank in (Rank.R3, Rank.R4, Rank.R5)])
    pair = run((Rank.R7, Suit.SPADE), (Rank.R7, Suit.HEART))
    pairs = run(
        *[
            (rank, suit)
            for rank in (Rank.R3, Rank.R4, Rank.R5)
            for suit in (Suit.SPADE, Suit.HEART)
        ]
    )
    small = run(*[(Rank.R3, suit) for suit in (Suit.SPADE, Suit.HEART, Suit.CLUB)])
    big = run(*[(Rank.R4, suit) for suit in Suit])
    for chain in ([single, straight, small, big], [pair, pairs, small, big]):
        for lower, higher in zip(chain, chain[1:]):
            assert beats(higher, lower)
            assert not beats(lower, higher)


# -- legacy tier comparison (traces recorded before 2026-09-25) --------------

LEGACY = Rules(comparison="tier")


def test_legacy_tier_straight_beats_pair():
    pair = run((Rank.R7, Suit.SPADE), (Rank.R7, Suit.HEART))
    straight = run(*[(rank, Suit.SPADE) for rank in (Rank.R3, Rank.R4, Rank.R5)])
    assert beats(straight, pair, LEGACY)
    assert not beats(pair, straight, LEGACY)


def test_legacy_tier_consecutive_pairs_beat_single():
    single = run((Rank.R7, Suit.SPADE))
    pairs = run(
        *[
            (rank, suit)
            for rank in (Rank.R3, Rank.R4, Rank.R5)
            for suit in (Suit.SPADE, Suit.HEART)
        ]
    )
    assert beats(pairs, single, LEGACY)
    assert not beats(single, pairs, LEGACY)


def test_legacy_tier_is_monotone_across_tiers():
    tier1 = [
        run((Rank.R7, Suit.SPADE)),
        run((Rank.R7, Suit.SPADE), (Rank.R7, Suit.HEART)),
    ]
    tier2 = [
        run(*[(rank, Suit.SPADE) for rank in (Rank.R3, Rank.R4, Rank.R5)]),
        run(
            *[
                (rank, suit)
                for rank in (Rank.R3, Rank.R4, Rank.R5)
                for suit in (Suit.SPADE, Suit.HEART)
            ]
        ),
    ]
    for lower in tier1:
        for higher in tier2:
            assert beats(higher, lower, LEGACY)
            assert not beats(lower, higher, LEGACY)
    # Same tier, different kind never compares (the old bug and the old rule).
    assert not beats(tier1[1], tier1[0], LEGACY)
    assert not beats(tier2[0], tier2[1], LEGACY)
    assert not beats(tier2[1], tier2[0], LEGACY)


def test_legacy_tier_bombs_beat_everything_and_compare_by_tier_then_rank():
    pair = run((Rank.R7, Suit.SPADE), (Rank.R7, Suit.HEART))
    small = run(*[(Rank.R3, suit) for suit in (Suit.SPADE, Suit.HEART, Suit.CLUB)])
    big = run(*[(Rank.R4, suit) for suit in Suit])
    assert beats(small, pair, LEGACY)
    assert beats(big, small, LEGACY)
    assert not beats(small, big, LEGACY)


def test_legacy_tier_still_compares_same_kind_by_size_then_top_card():
    short = run(*[(rank, Suit.SPADE) for rank in (Rank.R3, Rank.R4, Rank.R5)])
    long = run(*[(rank, Suit.SPADE) for rank in (Rank.R3, Rank.R4, Rank.R5, Rank.R6)])
    assert beats(long, short, LEGACY)
    assert not beats(short, long, LEGACY)
    spade = run((Rank.R7, Suit.SPADE))
    heart = run((Rank.R7, Suit.HEART))
    assert beats(spade, heart, LEGACY)
    assert not beats(heart, spade, LEGACY)

import pytest

from seven523.cards import (
    CARD_ORDER,
    NUM_CARDS,
    Card,
    Rank,
    Suit,
    card_id,
    card_key,
    is_point_card,
    make_deck,
    point_value,
    sorted_cards,
)


def test_deck_is_54_unique_cards():
    deck = make_deck()
    assert len(deck) == 54
    assert len(set(deck)) == 54
    assert deck == make_deck()
    assert deck == CARD_ORDER
    assert NUM_CARDS == 54


def test_deck_has_100_points():
    assert sum(point_value(card) for card in make_deck()) == 100
    assert sum(is_point_card(card) for card in make_deck()) == 12


def test_point_order_is_the_game_order():
    high_to_low = [
        Rank.R7,
        Rank.BIG_JOKER,
        Rank.SMALL_JOKER,
        Rank.R5,
        Rank.R2,
        Rank.R3,
        Rank.RA,
        Rank.RK,
        Rank.RQ,
        Rank.RJ,
        Rank.R10,
        Rank.R9,
        Rank.R8,
        Rank.R6,
        Rank.R4,
    ]
    assert high_to_low == sorted(high_to_low, key=int, reverse=True)


def test_suit_order():
    assert Suit.DIAMOND < Suit.CLUB < Suit.HEART < Suit.SPADE


def test_card_key_orders_rank_then_suit():
    seven_spade = Card(Rank.R7, Suit.SPADE)
    seven_heart = Card(Rank.R7, Suit.HEART)
    assert card_key(seven_spade) > card_key(seven_heart)
    assert card_key(Card(Rank.BIG_JOKER)) > card_key(Card(Rank.SMALL_JOKER))


def test_point_values():
    assert point_value(Card(Rank.R5, Suit.DIAMOND)) == 5
    assert point_value(Card(Rank.R10, Suit.DIAMOND)) == 10
    assert point_value(Card(Rank.RK, Suit.DIAMOND)) == 10
    assert point_value(Card(Rank.RA, Suit.DIAMOND)) == 0


def test_card_rejects_suit_mismatch():
    with pytest.raises(ValueError):
        Card(Rank.R7)  # non-joker needs a suit
    with pytest.raises(ValueError):
        Card(Rank.SMALL_JOKER, Suit.SPADE)  # jokers have no suit
    with pytest.raises(ValueError):
        Card(Rank.BIG_JOKER, Suit.HEART)


def test_card_equality_and_hash():
    assert Card(Rank.R7, Suit.SPADE) == Card(Rank.R7, Suit.SPADE)
    assert Card(Rank.SMALL_JOKER) == Card(Rank.SMALL_JOKER)
    assert len({Card(Rank.R7, Suit.SPADE), Card(Rank.R7, Suit.SPADE)}) == 1


def test_joker_suit_is_none_and_card_ids_are_a_bijection():
    ids = [card_id(card) for card in make_deck()]
    assert sorted(ids) == list(range(54))
    for card in make_deck():
        assert CARD_ORDER[card_id(card)] == card
    assert Card(Rank.SMALL_JOKER).suit is None
    assert Card(Rank.BIG_JOKER).suit is None


def test_joker_card_key_uses_suit_minus_one():
    assert card_key(Card(Rank.SMALL_JOKER)) == (int(Rank.SMALL_JOKER), -1)
    assert card_key(Card(Rank.BIG_JOKER)) == (int(Rank.BIG_JOKER), -1)


def test_sorted_cards_is_ascending_card_key():
    unsorted = [
        Card(Rank.R7, Suit.DIAMOND),
        Card(Rank.R4, Suit.SPADE),
        Card(Rank.R7, Suit.SPADE),
        Card(Rank.BIG_JOKER),
    ]
    result = sorted_cards(unsorted)
    assert list(result) == sorted(unsorted, key=card_key)
    assert result[-1].rank is Rank.R7 and result[-1].suit is Suit.SPADE


def test_is_point_card_and_jokers():
    assert all(
        is_point_card(Card(rank, Suit.DIAMOND)) for rank in (Rank.R5, Rank.R10, Rank.RK)
    )
    assert all(
        not is_point_card(Card(rank, Suit.DIAMOND))
        for rank in (Rank.R4, Rank.R6, Rank.R7, Rank.RA, Rank.R2)
    )
    assert not is_point_card(Card(Rank.SMALL_JOKER))
    assert not is_point_card(Card(Rank.BIG_JOKER))


def test_card_str_labels():
    assert str(Card(Rank.R7, Suit.SPADE)) == "♠7"
    assert str(Card(Rank.BIG_JOKER)) == "大王"
    assert str(Card(Rank.SMALL_JOKER)) == "小王"

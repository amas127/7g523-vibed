import pytest

from seven523.actions import catalog_for
from seven523.rules import DEFAULT_RULES, Rules


def test_default_rules_match_the_documented_variant():
    assert DEFAULT_RULES.num_players == 2
    assert DEFAULT_RULES.hand_size == 7
    assert DEFAULT_RULES.straight_min == 3
    assert DEFAULT_RULES.straight_max == 7
    assert DEFAULT_RULES.consecutive_pairs_min == 3
    assert DEFAULT_RULES.consecutive_pairs_max == 3
    assert DEFAULT_RULES.total_points == 100


@pytest.mark.parametrize("num_players", [1, 0, -1])
def test_rejects_fewer_than_two_players(num_players):
    with pytest.raises(ValueError):
        Rules(num_players=num_players)


def test_rejects_deal_that_overflows_the_deck():
    with pytest.raises(ValueError):
        Rules(num_players=8, hand_size=7)  # 56 > 54
    assert Rules(num_players=7, hand_size=7).num_players == 7


@pytest.mark.parametrize(
    "straight_min, straight_max",
    [(2, 7), (3, 2), (3, 1)],
)
def test_rejects_bad_straight_bounds(straight_min, straight_max):
    with pytest.raises(ValueError):
        Rules(straight_min=straight_min, straight_max=straight_max)


@pytest.mark.parametrize(
    "pairs_min, pairs_max",
    [(2, 3), (3, 2)],
)
def test_rejects_bad_consecutive_pair_bounds(pairs_min, pairs_max):
    with pytest.raises(ValueError):
        Rules(consecutive_pairs_min=pairs_min, consecutive_pairs_max=pairs_max)


def test_rules_are_hashable_and_catalogs_are_cached():
    assert hash(Rules()) == hash(DEFAULT_RULES)
    assert catalog_for(DEFAULT_RULES) is catalog_for(Rules())
    # A distinct, equal-valued Rules instance must hit the same cache entry.
    assert catalog_for(Rules(straight_max=13)) is catalog_for(Rules(straight_max=13))

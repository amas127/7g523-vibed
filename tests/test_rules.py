import pytest

import seven523.rules as rules_module
from seven523.actions import catalog_for
from seven523.rules import (
    DEFAULT_RULES,
    RULES_REVISION,
    Rules,
    rules_id,
    rules_identity,
)


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


# The default id is a worked example pinned by hand: it was computed once from
# the plain JSON payload (revision + fields, sorted keys, compact separators),
# not re-derived here, so any serialisation drift fails the literal.
def test_default_rules_identity_is_a_pinned_worked_example():
    assert RULES_REVISION == 3  # 撬底 on going out (ADR-0014)
    assert rules_identity() == {
        "revision": 3,
        "fields": {
            "num_players": 2,
            "hand_size": 7,
            "straight_min": 3,
            "straight_max": 7,
            "consecutive_pairs_min": 3,
            "consecutive_pairs_max": 3,
            "total_points": 100,
        },
    }
    assert rules_id() == "2e36dbea44893696"
    assert rules_id(DEFAULT_RULES) == "2e36dbea44893696"


def test_equal_fields_give_the_same_id():
    assert rules_id(Rules()) == rules_id(DEFAULT_RULES)
    assert rules_id(Rules(straight_max=13)) == rules_id(Rules(straight_max=13))


def test_a_changed_field_changes_the_id():
    assert rules_id(Rules(total_points=101)) != rules_id(DEFAULT_RULES)


def test_a_bumped_revision_changes_the_id(monkeypatch):
    baseline = rules_id(DEFAULT_RULES)
    monkeypatch.setattr(rules_module, "RULES_REVISION", 4)
    assert rules_id(DEFAULT_RULES) != baseline

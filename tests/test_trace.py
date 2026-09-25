"""The trace codec in isolation: format round-trips, no terminal or torch."""
import json
import random

import pytest

from seven523.cards import CARD_ORDER
from seven523.game import Deal, Game
from seven523.rules import DEFAULT_RULES, Rules
from seven523.trace import (
    TRACE_VERSION,
    build_trace,
    card_from_json,
    card_json,
    deal_from_json,
    deal_json,
    initial_snapshot,
    load_trace,
    rules_from_json,
    rules_json,
    save_trace,
    state_from_snapshot,
)


def test_card_json_round_trip():
    for card in (CARD_ORDER[0], CARD_ORDER[13], CARD_ORDER[-1]):
        data = card_json(card)
        assert set(data) == {"id", "label"}
        assert card_from_json(data) == card


def test_rules_json_round_trips_every_field():
    rules = Rules(num_players=3, straight_max=13)
    assert rules_from_json(rules_json(rules)) == rules


def test_deal_json_round_trips():
    game = Game(DEFAULT_RULES)
    state = game.new(random.Random(3))
    restored = deal_from_json(deal_json(Deal.from_state(state)))
    assert restored.hands == state.hands
    assert restored.draw_pile == state.draw_pile
    assert restored.revealed == state.revealed
    assert restored.starter == state.current


def test_initial_snapshot_rebuilds_the_opening_state():
    game = Game(DEFAULT_RULES)
    state = game.new(random.Random(9))
    rebuilt = state_from_snapshot(initial_snapshot(state), DEFAULT_RULES)
    assert rebuilt.hands == state.hands
    assert rebuilt.draw_pile == state.draw_pile
    assert rebuilt.revealed == state.revealed
    assert rebuilt.current == state.current
    assert rebuilt.scores == (0,) * DEFAULT_RULES.num_players


def test_save_and_load_trace_round_trip(tmp_path):
    trace = build_trace(
        DEFAULT_RULES,
        seed=7,
        human_seat=1,
        players=["a", "b"],
        created_at="2026-01-01T00:00:00",
        initial={"starter": 0},
        steps=[{"seat": 0, "action": 1}],
        final_scores=[100, 0],
        winner=0,
    )
    path = save_trace(tmp_path / "deep" / "game.json", trace)
    loaded = load_trace(path)
    assert loaded == trace
    assert json.loads(path.read_text())["version"] == TRACE_VERSION


def test_build_trace_stamps_version_and_metadata():
    trace = build_trace(
        DEFAULT_RULES,
        seed=None,
        human_seat=0,
        players=["a", "b"],
        created_at="t",
        initial={},
        steps=[],
    )
    assert trace["version"] == TRACE_VERSION
    assert trace["human_seat"] == 0
    assert trace["rules"] == rules_json(DEFAULT_RULES)
    assert trace["players"] == ["a", "b"]

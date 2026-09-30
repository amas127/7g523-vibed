"""The trace codec in isolation: format round-trips, no terminal or torch."""
import json
import random

import pytest

from seven523.actions import legal_ids
from seven523.cards import CARD_ORDER
from seven523.game import Deal, Game
from seven523.rules import DEFAULT_RULES, RULES_REVISION, Rules
from seven523.trace import (
    TRACE_VERSION,
    build_trace,
    card_from_json,
    card_json,
    deal_from_json,
    deal_json,
    initial_snapshot,
    load_trace,
    parse_player_label,
    player_label,
    rules_from_json,
    rules_json,
    save_trace,
    state_from_snapshot,
    step_record,
    trace_filename,
)


def test_card_json_round_trip():
    for card in (CARD_ORDER[0], CARD_ORDER[13], CARD_ORDER[-1]):
        data = card_json(card)
        assert set(data) == {"id", "label"}
        assert card_from_json(data) == card


def test_rules_json_round_trips_every_field():
    rules = Rules(num_players=3, straight_max=13)
    assert set(rules_json(rules)) == {
        "num_players",
        "hand_size",
        "straight_min",
        "straight_max",
        "consecutive_pairs_min",
        "consecutive_pairs_max",
        "total_points",
    }
    assert rules_from_json(rules_json(rules)) == rules


def test_stale_comparison_key_reads_as_family_without_mutating_the_record():
    legacy = rules_json(DEFAULT_RULES)
    legacy["comparison"] = "tier"
    legacy["revision"] = 2
    assert rules_from_json(legacy) == DEFAULT_RULES
    # Reading a legacy record must not rewrite the caller's copy in place, and
    # the trace-only revision stamp is not a ``Rules`` field.
    assert legacy["comparison"] == "tier"
    assert legacy["revision"] == 2


def test_v3_trace_records_the_rules_revision(tmp_path):
    trace = build_trace(
        DEFAULT_RULES,
        seed=11,
        human_seat=0,
        players=["human@seat0", "anchor:random@seat1"],
        created_at="2026-09-24T00:00:00",
        initial={"starter": 0},
        steps=[],
        final_scores=[100, 0],
    )
    assert trace["version"] == TRACE_VERSION == 3
    assert trace["rules"]["revision"] == RULES_REVISION == 3
    # The stamp survives a save/load round trip.
    loaded = load_trace(save_trace(tmp_path / "game.json", trace))
    assert loaded["rules"]["revision"] == RULES_REVISION


def test_legacy_v1_trace_is_refused_by_replay():
    # A v1 record predates the rules-revision stamp; the 撬底 terminal rule
    # changed in revision 3, so replay must refuse it instead of diverging.
    from seven523.play import replay_trace

    trace = build_trace(
        DEFAULT_RULES,
        seed=11,
        human_seat=0,
        players=["human@seat0", "anchor:random@seat1"],
        created_at="2026-09-24T00:00:00",
        initial={"starter": 0},
        steps=[],
        final_scores=[100, 0],
    )
    trace["version"] = 1
    trace["rules"].pop("revision", None)
    with pytest.raises(ValueError, match="version 1"):
        replay_trace(trace, print_fn=lambda *args, **kwargs: None)


def test_legacy_v2_trace_is_refused_by_replay():
    # v2 (family comparison, old winner-only 撬底) is the corpus that ships in
    # traces/study and traces/pool10; replay refuses it loudly (ADR-0014).
    from seven523.play import replay_trace

    trace = build_trace(
        DEFAULT_RULES,
        seed=11,
        human_seat=0,
        players=["human@seat0", "anchor:random@seat1"],
        created_at="2026-09-24T00:00:00",
        initial={"starter": 0},
        steps=[],
        final_scores=[100, 0],
    )
    trace["version"] = 2
    trace["rules"].pop("revision", None)
    with pytest.raises(ValueError, match="version 2"):
        replay_trace(trace, print_fn=lambda *args, **kwargs: None)


def test_trace_from_another_rules_revision_is_refused():
    from seven523.play import replay_trace

    trace = build_trace(
        DEFAULT_RULES,
        seed=11,
        human_seat=0,
        players=["human@seat0", "anchor:random@seat1"],
        created_at="2026-09-24T00:00:00",
        initial={"starter": 0},
        steps=[],
        final_scores=[100, 0],
    )
    trace["rules"]["revision"] = RULES_REVISION + 1
    with pytest.raises(ValueError, match="rules revision"):
        replay_trace(trace, print_fn=lambda *args, **kwargs: None)


@pytest.mark.parametrize("stamp", [float(RULES_REVISION), "3", True])
def test_non_int_rules_revision_stamps_are_refused(stamp):
    # Revision identity is the engine integer: JSON ``3.0`` (float), ``"3"``
    # and booleans must not slip past a value-lax ``!=`` comparison.
    from seven523.play import replay_trace

    trace = build_trace(
        DEFAULT_RULES,
        seed=11,
        human_seat=0,
        players=["human@seat0", "anchor:random@seat1"],
        created_at="2026-09-24T00:00:00",
        initial={"starter": 0},
        steps=[],
        final_scores=[100, 0],
    )
    trace["rules"]["revision"] = stamp
    with pytest.raises(ValueError, match="rules revision"):
        replay_trace(trace, print_fn=lambda *args, **kwargs: None)


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
    assert rebuilt.played == ()


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
    assert "comparison" not in loaded["rules"]
    assert loaded["rules"]["revision"] == RULES_REVISION


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
    assert trace["rules"] == {**rules_json(DEFAULT_RULES), "revision": RULES_REVISION}
    assert trace["players"] == ["a", "b"]


def test_player_label_round_trips_and_tolerates_hand_written_labels():
    assert player_label("anchor", "random", 1) == "anchor:random@seat1"
    assert parse_player_label("anchor:random@seat1") == ("anchor", "random", 1)
    assert parse_player_label("subject:lvl3@seat0") == ("subject", "lvl3", 0)
    # Labels the terminal writes by hand parse without crashing a reader.
    assert parse_player_label("human@seat0") == (None, "human", 0)
    assert parse_player_label("随机 bot") == (None, "随机 bot", None)


def test_trace_filename_is_the_study_convention():
    assert trace_filename(7, 42, 1, "random") == "g0007__s42__seat1__vsrandom.json"


def test_step_record_owns_the_step_schema():
    game = Game(DEFAULT_RULES)
    state = game.new(random.Random(11))
    seat = state.current
    action_id = legal_ids(game.view(state, seat).mask)[0]
    after, result = game.step(state, action_id)
    record = step_record(seat, action_id, None, text="单牌", state=after, result=result)
    assert set(record) == {
        "seat",
        "action",
        "suit",
        "text",
        "scores",
        "hand_sizes",
        "draw_count",
        "trick_over",
        "winner",
        "points",
        "dug",
    }
    assert record["seat"] == seat
    assert record["action"] == action_id
    assert record["scores"] == list(after.scores)
    assert record["hand_sizes"] == [len(hand) for hand in after.hands]
    assert record["draw_count"] == len(after.draw_pile)
    assert (
        record["trick_over"],
        record["winner"],
        record["points"],
        record["dug"],
    ) == (result.trick_over, result.winner, result.points_taken, result.dug)

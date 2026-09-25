"""The Match driver: rotation, policy dispatch, and the legality clamp.

These exercise the loop through its own interface — no gymnasium, no torch.
"""
import random

import pytest

from seven523.game import Game
from seven523.match import Match
from seven523.policies import GreedyBot, RandomBot, make_scripted_policies
from seven523.rules import DEFAULT_RULES, Rules


class _AlwaysIllegal:
    """A policy that never returns a legal template."""

    def __init__(self) -> None:
        self.calls = 0

    def act(self, view):
        self.calls += 1
        return 10_000, None


def test_run_to_end_with_all_policies_finishes():
    match = Match(DEFAULT_RULES, [GreedyBot(), GreedyBot()], rng=random.Random(0))
    state = match.run_to_end()
    assert state.done
    assert sum(state.scores) == DEFAULT_RULES.total_points
    assert match.steps == sum(match.turns)
    assert all(turns > 0 for turns in match.turns)


def test_advance_stops_at_an_external_seat():
    match = Match(DEFAULT_RULES, [None, GreedyBot()], rng=random.Random(3))
    match.advance()
    assert match.done or match.seat == 0
    assert match.policy_for(match.seat) is None


def test_run_to_end_requires_a_policy_for_every_seat():
    match = Match(DEFAULT_RULES, [None, GreedyBot()], rng=random.Random(3))
    with pytest.raises(RuntimeError, match="no policy"):
        match.run_to_end()


def test_external_illegal_action_is_rejected_strictly():
    match = Match(DEFAULT_RULES, [None, None], rng=random.Random(1))
    view = match.view()
    illegal = next(
        index for index in range(len(match.game.catalog)) if not (view.mask >> index) & 1
    )
    with pytest.raises(ValueError):
        match.step(illegal, None)


def test_policy_illegal_action_is_clamped_and_counted():
    match = Match(DEFAULT_RULES, [_AlwaysIllegal()] * 2, rng=random.Random(2))
    state = match.run_to_end()
    assert state.done
    assert match.illegal_actions == match.steps
    assert match.steps > 0


def test_on_turn_observes_every_turn():
    match = Match(DEFAULT_RULES, [RandomBot(random.Random(0)), GreedyBot()], rng=random.Random(4))
    seen = []
    match.on_turn = lambda seat, action_id, suit, view, result: seen.append(
        (seat, action_id, view.seat, result)
    )
    match.run_to_end()
    assert len(seen) == match.steps
    assert all(seat == view_seat for seat, _, view_seat, _ in seen)


def test_view_defaults_to_the_seat_to_move_and_accepts_an_explicit_seat():
    match = Match(DEFAULT_RULES, [None, None], rng=random.Random(5))
    assert match.view().seat == match.seat
    other = (match.seat + 1) % DEFAULT_RULES.num_players
    assert match.view(other).seat == other


def test_match_can_be_built_from_a_replay_state():
    game = Game(DEFAULT_RULES)
    state = game.new(random.Random(7))
    match = Match(DEFAULT_RULES, [None, None], state=state)
    assert match.state is state
    assert match.view().seat == state.current


def test_advance_and_explicit_step_interleave():
    # External seat 0 drives, GreedyBot drives seat 1: the two together finish.
    match = Match(DEFAULT_RULES, [None, GreedyBot()], rng=random.Random(9))
    while not match.done:
        if match.seat == 0:
            legal = [
                index
                for index in range(len(match.game.catalog))
                if (match.view().mask >> index) & 1
            ]
            match.step(random.Random(match.seat + match.steps).choice(legal), None)
        else:
            match.advance()
    assert match.done
    assert sum(match.state.scores) == DEFAULT_RULES.total_points


def test_policies_list_length_is_validated():
    with pytest.raises(ValueError, match="policies"):
        Match(Rules(num_players=3), [GreedyBot()], rng=random.Random(0))


def test_custom_rules_match_still_terminates():
    rules = Rules(num_players=3, straight_max=13)
    match = Match(rules, make_scripted_policies("greedy", rules, seed=0), rng=random.Random(0))
    match.run_to_end()
    assert sum(match.state.scores) == rules.total_points

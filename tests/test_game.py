import random
from dataclasses import replace

import pytest

from seven523.actions import CATALOG, catalog_for, split_action
from seven523.cards import Card, Rank, Suit, card_key, point_value
from seven523.combos import ComboKind, classify
from seven523.env import encode_observation
from seven523.game import Game, GameState, Phase
from seven523.policies import GreedyBot, RandomBot
from seven523.rules import DEFAULT_RULES, Rules


def points(cards):
    return sum(point_value(card) for card in cards)


def check_invariants(state, rules=DEFAULT_RULES):
    total_cards = (
        sum(len(hand) for hand in state.hands)
        + len(state.draw_pile)
        + len(state.trick_cards)
        + state.collected
    )
    assert total_cards == 54

    total_points = (
        sum(state.scores)
        + state.trick_points
        + sum(points(hand) for hand in state.hands)
        + points(state.draw_pile)
    )
    assert total_points == 100

    for hand in state.hands:
        assert len(hand) <= rules.hand_size
    if not state.done:
        assert state.hands[state.current]


def play_episode(rng, bots):
    game = Game()
    state = game.new(rng)
    steps = 0
    while not state.done:
        check_invariants(state)
        seat = state.current
        view = game.view(state, seat)
        assert view.mask != 0
        action_id, suit = bots[seat].act(view)
        assert (view.mask >> action_id) & 1
        state, result = game.step(state, action_id, suit)
        if result.trick_over and result.refilled:
            offsets = [(filled - result.winner - 1) % 2 for filled in result.refilled]
            assert offsets == sorted(offsets), "refill must follow the winner rotation"
        steps += 1
        assert steps < 20_000, "episode did not terminate"
    check_invariants(state)
    assert sum(state.scores) == 100
    assert state.draw_pile == ()
    return state


def test_random_games_terminate_with_invariants():
    for seed in range(50):
        play_episode(
            random.Random(seed),
            [RandomBot(random.Random(seed)), RandomBot(random.Random(seed + 1))],
        )


def test_greedy_games_terminate_with_invariants():
    for seed in range(50):
        play_episode(random.Random(1000 + seed), [GreedyBot(), GreedyBot()])


def test_reveal_determines_starter():
    state = Game().new(random.Random(3))
    smallest = min(state.revealed, key=card_key)
    assert state.revealed[state.current] == smallest


def test_returns_are_zero_sum_and_bounded():
    game = Game()
    state = play_episode(random.Random(4), [GreedyBot(), GreedyBot()])
    returns = game.returns(state)
    assert abs(sum(returns)) < 1e-9
    assert all(-1.0 <= value <= 1.0 for value in returns)


def test_illegal_action_is_rejected():
    game = Game()
    state = game.new(random.Random(5))
    view = game.view(state, state.current)
    illegal = next(index for index in range(len(CATALOG)) if not (view.mask >> index) & 1)
    with pytest.raises(ValueError):
        game.step(state, illegal)


# -- projection / leakage --------------------------------------------------


def make_state(hands, draw, *, revealed, incumbent=None, current=0, scores=None, played=()):
    n = len(hands)
    return GameState(
        hands=tuple(frozenset(hand) for hand in hands),
        draw_pile=tuple(draw),
        scores=tuple(scores or (0,) * n),
        trick_points=0,
        trick_cards=(),
        revealed=tuple(revealed),
        current=current,
        incumbent=incumbent,
        last_player=None,
        collected=0,
        phase=Phase.PLAY,
        played=tuple(played),
    )


def test_view_and_observation_do_not_leak_hidden_state():
    game = Game()
    own = [Card(Rank.R7, Suit.SPADE), Card(Rank.R3, Suit.HEART)]
    revealed = [Card(Rank.R3, Suit.HEART), Card(Rank.R2, Suit.SPADE)]
    hidden_a = [Card(Rank.R2, Suit.SPADE), Card(Rank.R5, Suit.CLUB)]
    hidden_b = [Card(Rank.R2, Suit.HEART), Card(Rank.R5, Suit.DIAMOND)]
    draw_a = [Card(Rank.R4, Suit.SPADE), Card(Rank.R6, Suit.CLUB)]
    draw_b = [Card(Rank.R6, Suit.CLUB), Card(Rank.R4, Suit.SPADE)]
    played = [Card(Rank.R9, Suit.CLUB)]

    state_a = make_state([own, hidden_a], draw_a, revealed=revealed, played=played)
    state_b = make_state([own, hidden_b], draw_b, revealed=revealed, played=played)
    view_a = game.view(state_a, 0)
    view_b = game.view(state_b, 0)

    for field in (
        "mask",
        "incumbent",
        "current",
        "scores",
        "counts",
        "draw_count",
        "revealed",
        "trick_cards",
        "played",
        "last_player",
        "done",
    ):
        assert getattr(view_a, field) == getattr(view_b, field)
    assert encode_observation(view_a) == encode_observation(view_b)


def test_played_history_accumulates_finished_tricks_only():
    game = Game()
    a = [Card(Rank.R7, Suit.SPADE), Card(Rank.R4, Suit.DIAMOND)]
    b = [Card(Rank.R3, Suit.HEART), Card(Rank.R6, Suit.CLUB)]
    state = make_state([a, b], [], revealed=[a[0], b[0]], current=0)
    assert state.played == ()
    assert game.view(state, 0).played == ()

    state, _ = game.step(state, action_id(ComboKind.SINGLE, (Rank.R7,)))
    # The trick is still open: its cards are public in ``trick_cards``, not in
    # the collected history yet.
    assert state.played == ()
    assert game.view(state, 1).played == ()

    state, result = game.step(state, PASS)
    assert result.trick_over and result.winner == 0 and not result.dug
    assert state.played == (Card(Rank.R7, Suit.SPADE),)
    assert game.view(state, 0).played == (Card(Rank.R7, Suit.SPADE),)
    assert game.view(state, 1).played == (Card(Rank.R7, Suit.SPADE),)


def test_played_history_accounts_for_every_card_in_a_full_game():
    game = Game()
    policy = GreedyBot()
    for seed in range(5):
        state = game.new(random.Random(seed))
        while not state.done:
            action, suit = policy.act(game.view(state, state.current))
            state, _ = game.step(state, action, suit)
            if state.done:
                break
            # Until 撬底 collects the losers' hands, every card that left the
            # draw pile or a hand sits either in an open trick or in ``played``.
            assert len(state.played) + len(state.trick_cards) == 54 - len(
                state.draw_pile
            ) - sum(len(hand) for hand in state.hands)
        assert state.done


# -- multi-player flow (crafted states) --------------------------------------


def action_id(kind, ranks):
    return next(
        index
        for index, action in enumerate(CATALOG)
        if action.kind is kind and action.ranks == ranks
    )


PASS = action_id(None, ())


def test_refill_order_and_partial_exhaustion_three_players():
    game = Game(Rules(num_players=3))
    a = [Card(Rank.R7, Suit.SPADE)]
    b = [Card(Rank.R4, Suit.DIAMOND)]
    c = [Card(Rank.R4, Suit.CLUB)]
    revealed = [a[0], b[0], c[0]]
    draw = tuple(
        Card(rank, suit)
        for rank in (Rank.R6, Rank.R8, Rank.R9, Rank.RJ, Rank.RQ, Rank.RA)
        for suit in Suit
    )[:21]

    state = make_state([a, b, c], draw, revealed=revealed, current=0)
    state, _ = game.step(state, action_id(ComboKind.SINGLE, (Rank.R7,)))
    state, _ = game.step(state, PASS)
    state, result = game.step(state, PASS)
    assert result.trick_over and result.winner == 0
    assert result.refilled == (1, 2, 0)  # winner's next seat first, winner last
    assert [len(hand) for hand in state.hands] == [7, 7, 7]
    assert len(state.draw_pile) == 2  # 21 - 6 - 6 - 7

    partial = tuple(draw[:7])
    state = make_state([a, b, c], partial, revealed=revealed, current=0)
    state, _ = game.step(state, action_id(ComboKind.SINGLE, (Rank.R7,)))
    state, _ = game.step(state, PASS)
    state, result = game.step(state, PASS)
    assert result.refilled == (1, 2)  # pile ran out before the winner
    assert [len(hand) for hand in state.hands] == [0, 7, 2]
    assert state.draw_pile == ()


def test_dug_collects_trick_points_and_other_hands_points():
    game = Game(DEFAULT_RULES)
    a = [Card(Rank.R5, Suit.DIAMOND)]
    b = [Card(Rank.RK, Suit.HEART), Card(Rank.R10, Suit.SPADE), Card(Rank.R4, Suit.DIAMOND)]
    state = make_state([a, b], [], revealed=[a[0], b[0]], current=0)

    state, _ = game.step(state, action_id(ComboKind.SINGLE, (Rank.R5,)))
    state, result = game.step(state, PASS)  # K and 10 cannot beat 5 in point order

    assert result.dug and result.done
    assert result.points_taken == 25  # trick 5 + B's K(10) + 10(10)
    assert state.scores == (25, 0)
    assert state.hands[1] == frozenset({Card(Rank.R4, Suit.DIAMOND)})  # non-point stays
    assert state.collected == 3  # 5 played + K + 10 collected


def test_non_winner_with_empty_hand_is_skipped_then_held():
    game = Game(DEFAULT_RULES)
    a = [Card(Rank.R7, Suit.SPADE), Card(Rank.R4, Suit.DIAMOND)]
    b = []
    state = make_state([a, b], [], revealed=[a[0]], current=0)

    state, result = game.step(state, action_id(ComboKind.SINGLE, (Rank.R7,)))
    assert result.trick_over and result.winner == 0 and not result.dug
    assert state.hands[1] == frozenset()
    assert state.current == 0
    assert not state.done

    state, result = game.step(state, action_id(ComboKind.SINGLE, (Rank.R4,)))
    assert result.dug and state.done


def test_step_after_terminal_raises():
    state = replace(
        make_state([[Card(Rank.R7, Suit.SPADE)]], [], revealed=[Card(Rank.R7, Suit.SPADE)]),
        phase=Phase.DONE,
    )
    with pytest.raises(RuntimeError):
        Game().step(state, 0)


def test_step_honours_a_legal_suit_request():
    game = Game()
    a = [Card(Rank.R7, Suit.SPADE), Card(Rank.R7, Suit.HEART), Card(Rank.R7, Suit.CLUB)]
    b = [Card(Rank.R4, Suit.DIAMOND)]
    state = make_state([a, b], [], revealed=[a[0], b[0]], current=0)
    state, _ = game.step(state, action_id(ComboKind.PAIR, (Rank.R7,)), Suit.HEART)
    assert set(state.trick_cards) == {
        Card(Rank.R7, Suit.HEART),
        Card(Rank.R7, Suit.CLUB),
    }
    assert Card(Rank.R7, Suit.SPADE) in state.hands[0]  # the spade is kept


def test_step_falls_back_when_the_suit_request_would_be_illegal():
    game = Game()
    hand = [Card(Rank.R7, Suit.SPADE), Card(Rank.R7, Suit.CLUB)]
    incumbent = classify([Card(Rank.R7, Suit.HEART)])
    b = [Card(Rank.R4, Suit.DIAMOND)]
    state = make_state(
        [hand, b],
        [],
        revealed=[hand[0], b[0]],
        incumbent=incumbent,
        current=0,
    )
    # ♠7 beats ♥7, ♣7 does not: the club request must fall back to the spade.
    state, result = game.step(state, action_id(ComboKind.SINGLE, (Rank.R7,)), Suit.CLUB)
    assert not result.trick_over
    assert Card(Rank.R7, Suit.SPADE) in state.trick_cards
    assert state.hands[0] == frozenset({Card(Rank.R7, Suit.CLUB)})


# -- multi-player fuzz -------------------------------------------------------


def play_episode_rules(rules, rng, bots):
    game = Game(rules)
    state = game.new(rng)
    steps = 0
    while not state.done:
        check_invariants(state, rules)
        seat = state.current
        view = game.view(state, seat)
        assert view.mask != 0
        chosen, suit = bots[seat].act(view)
        assert (view.mask >> chosen) & 1
        state, result = game.step(state, chosen, suit)
        if result.trick_over and result.refilled:
            offsets = [
                (filled - result.winner - 1) % rules.num_players
                for filled in result.refilled
            ]
            assert offsets == sorted(offsets)
        steps += 1
        assert steps < 50_000, "episode did not terminate"
    check_invariants(state, rules)
    assert sum(state.scores) == 100
    assert state.draw_pile == ()
    return state


@pytest.mark.parametrize("num_players", [2, 3, 4, 5])
def test_random_games_terminate_with_invariants_for_many_players(num_players):
    rules = Rules(num_players=num_players)
    for seed in range(10):
        play_episode_rules(
            rules,
            random.Random(seed),
            [RandomBot(random.Random(seed + seat)) for seat in range(num_players)],
        )


@pytest.mark.parametrize("num_players", [2, 3, 4])
def test_greedy_games_terminate_for_many_players(num_players):
    rules = Rules(num_players=num_players)
    for seed in range(10):
        play_episode_rules(
            rules,
            random.Random(100 + seed),
            [GreedyBot(rules) for _ in range(num_players)],
        )


def test_returns_are_zero_sum_for_many_players():
    for num_players in (3, 4):
        rules = Rules(num_players=num_players)
        game = Game(rules)
        state = play_episode_rules(
            rules,
            random.Random(7),
            [GreedyBot(rules) for _ in range(num_players)],
        )
        returns = game.returns(state)
        assert abs(sum(returns)) < 1e-9
        assert all(-1.0 <= value <= 1.0 for value in returns)


def test_greedy_bot_with_custom_rules_uses_the_matching_catalog():
    rules = Rules(straight_max=13)
    assert GreedyBot(rules).pass_id == len(catalog_for(rules)) - 1
    for seed in range(5):
        play_episode_rules(
            rules, random.Random(seed), [GreedyBot(rules), GreedyBot(rules)]
        )


def test_reveal_determines_starter_three_players():
    for seed in range(10):
        state = Game(Rules(num_players=3)).new(random.Random(seed))
        assert len(state.revealed) == 3
        smallest = min(state.revealed, key=card_key)
        assert state.revealed[state.current] == smallest

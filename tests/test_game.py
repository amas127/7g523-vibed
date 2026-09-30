import random
from dataclasses import replace

import pytest
from support import FirstLegalBot

from seven523.actions import CATALOG, PASS_ID
from seven523.cards import Card, Rank, Suit, card_key, point_value
from seven523.combos import ComboKind, classify
from seven523.env import encode_observation
from seven523.game import Game, GameState, Phase, seat_outcome
from seven523.policies import RandomBot
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
        # Revision 3 (ADR-0014): an empty hand with an empty draw pile must have
        # ended the round; the two can never coexist in a live state.
        assert state.draw_pile or all(state.hands)


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
        trick_winner = state.last_player
        state, result = game.step(state, action_id, suit)
        if result.trick_over and result.refilled:
            assert trick_winner is not None
            offsets = [
                (filled - trick_winner - 1) % 2 for filled in result.refilled
            ]
            assert offsets == sorted(offsets), "refill must follow the winner rotation"
        steps += 1
        assert steps < 20_000, "episode did not terminate"
    check_invariants(state)
    assert sum(state.scores) == 100
    assert state.draw_pile == ()
    return state


def test_random_games_terminate_with_invariants():
    for seed in range(25):
        play_episode(
            random.Random(seed),
            [RandomBot(random.Random(seed)), RandomBot(random.Random(seed + 1))],
        )


def test_scripted_games_terminate_with_invariants():
    for seed in range(25):
        play_episode(random.Random(1000 + seed), [FirstLegalBot(), FirstLegalBot()])


def test_reveal_determines_starter():
    state = Game().new(random.Random(3))
    smallest = min(state.revealed, key=card_key)
    assert state.revealed[state.current] == smallest


def test_returns_are_zero_sum_and_bounded():
    game = Game()
    state = play_episode(random.Random(4), [FirstLegalBot(), FirstLegalBot()])
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


def make_state(
    hands,
    draw,
    *,
    revealed,
    incumbent=None,
    current=0,
    scores=None,
    played=(),
    trick_points=0,
    trick_cards=(),
    last_player=None,
    empty_order=(),
    collected=0,
    phase=Phase.PLAY,
):
    n = len(hands)
    return GameState(
        hands=tuple(frozenset(hand) for hand in hands),
        draw_pile=tuple(draw),
        scores=tuple(scores or (0,) * n),
        trick_points=trick_points,
        trick_cards=tuple(trick_cards),
        revealed=tuple(revealed),
        current=current,
        incumbent=incumbent,
        last_player=last_player,
        collected=collected,
        phase=phase,
        played=tuple(played),
        empty_order=tuple(empty_order),
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
    policy = FirstLegalBot()
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


def test_immediate_dig_collects_trick_points_and_other_hands_points():
    # RULES.md 4.8 (revision 3): leading the last card with an empty draw pile
    # ends the round on the spot — the opponent never gets to respond.
    game = Game(DEFAULT_RULES)
    a = [Card(Rank.R5, Suit.DIAMOND)]
    b = [Card(Rank.RK, Suit.HEART), Card(Rank.R10, Suit.SPADE), Card(Rank.R4, Suit.DIAMOND)]
    state = make_state([a, b], [], revealed=[a[0], b[0]], current=0)

    state, result = game.step(state, action_id(ComboKind.SINGLE, (Rank.R5,)))

    assert result.trick_over and result.dug and result.done
    assert result.winner == 0  # the actor digs; the truncated trick is theirs
    assert result.refilled == ()
    assert result.points_taken == 25  # trick 5 + B's K(10) + 10(10)
    assert state.scores == (25, 0)
    assert state.hands[1] == frozenset({Card(Rank.R4, Suit.DIAMOND)})  # non-point stays
    assert state.collected == 3  # 5 played + K + 10 collected
    assert state.empty_order == (0,)
    assert state.current == 0
    with pytest.raises(RuntimeError, match="game is over"):
        game.step(state, PASS)


def test_immediate_dig_from_each_leading_seat():
    # Both seats lead their last card (a 5) with an empty draw pile; the actor
    # digs the other hand's points in either seat assignment.
    game = Game(DEFAULT_RULES)
    for seat in (0, 1):
        other = 1 - seat
        hands: list[list[Card]] = [[], []]
        hands[seat] = [Card(Rank.R5, Suit.DIAMOND)]
        hands[other] = [
            Card(Rank.RK, Suit.HEART),
            Card(Rank.R10, Suit.SPADE),
            Card(Rank.R4, Suit.DIAMOND),
        ]
        state = make_state(
            hands, [], revealed=[hands[0][0], hands[1][0]], current=seat
        )
        state, result = game.step(state, action_id(ComboKind.SINGLE, (Rank.R5,)))
        assert result.dug and result.done and result.winner == seat
        assert result.points_taken == 25
        assert state.scores[seat] == 25 and sum(state.scores) == 25
        assert state.hands[other] == frozenset({Card(Rank.R4, Suit.DIAMOND)})


def test_immediate_dig_when_the_last_card_is_a_response():
    # B beats A's lead with its only card and the draw pile is empty: B digs
    # immediately; A never gets another turn.
    game = Game(DEFAULT_RULES)
    a = [Card(Rank.R4, Suit.SPADE), Card(Rank.RK, Suit.CLUB)]
    b = [Card(Rank.R5, Suit.HEART)]
    state = make_state([a, b], [], revealed=[a[0], b[0]], current=0)

    state, first = game.step(state, action_id(ComboKind.SINGLE, (Rank.R4,)))
    assert not first.trick_over and state.current == 1  # A still has the K

    state, result = game.step(state, action_id(ComboKind.SINGLE, (Rank.R5,)))
    assert result.dug and result.done and result.winner == 1
    assert result.points_taken == 15  # the 5 plus A's swept K
    assert state.scores == (0, 15)
    assert state.hands[0] == frozenset()  # the K was swept
    with pytest.raises(RuntimeError, match="game is over"):
        game.step(state, PASS)


def test_banked_scores_are_untouched_by_the_dig_sweep():
    # Already-collected points stay where they are; the dig only adds the
    # still-uncollected trick and hand points.
    game = Game(DEFAULT_RULES)
    a = [Card(Rank.R5, Suit.DIAMOND)]
    b = [Card(Rank.RK, Suit.HEART), Card(Rank.R10, Suit.SPADE)]
    state = make_state([a, b], [], revealed=[a[0], b[0]], current=0, scores=(30, 20))

    state, result = game.step(state, action_id(ComboKind.SINGLE, (Rank.R5,)))

    assert result.dug and result.done
    assert result.points_taken == 25
    assert state.scores == (55, 20)  # A's bank 30 + 25; B's bank 20 untouched
    assert sum(state.scores) == 75


def test_non_empty_draw_pile_defers_the_dig_and_refills():
    # RULES.md 4.9: going out while the draw pile is not empty is not a dig;
    # the seat refills at trick end and play continues normally.
    game = Game(DEFAULT_RULES)
    a = [Card(Rank.R7, Suit.SPADE)]  # last card, led first
    b = [Card(Rank.R4, Suit.DIAMOND)]
    draw = tuple(
        Card(rank, suit)
        for rank in (Rank.R6, Rank.R8, Rank.R9, Rank.RJ, Rank.RQ, Rank.RK, Rank.RA)
        for suit in (Suit.SPADE, Suit.CLUB, Suit.DIAMOND)
    )[:10]
    state = make_state([a, b], draw, revealed=[a[0], b[0]], current=0)

    state, result = game.step(state, action_id(ComboKind.SINGLE, (Rank.R7,)))
    assert not result.trick_over  # B may still respond
    assert state.empty_order == (0,)
    assert not state.done

    state, result = game.step(state, PASS)
    assert result.trick_over and result.winner == 0
    assert not result.dug and not result.done
    assert result.refilled == (1, 0)  # winner's next seat first, winner last
    assert [len(hand) for hand in state.hands] == [4, 7]  # 10 cards: 6 + 4
    assert state.draw_pile == ()
    assert state.empty_order == ()  # A refilled, so it is no longer out
    assert state.current == 0  # the winner leads the next trick


def test_refill_corner_sends_the_earliest_empty_player():
    # A empties mid-trick while the draw pile still has cards; seat 1 wins the
    # trick; the refill drains the pile before A's turn, so A digs although it
    # did not win the trick.  Hand-computed: trick 4♦+7♠ = 0 points; the swept
    # K♥ is 10; seat 2's refill (4 diamonds) has no points and stays.
    game = Game(Rules(num_players=3))
    hands = [
        [Card(Rank.R4, Suit.DIAMOND)],
        [Card(Rank.R7, Suit.SPADE), Card(Rank.RK, Suit.HEART)],
        [Card(Rank.R6, Suit.CLUB), Card(Rank.R8, Suit.SPADE)],
    ]
    draw = [
        Card(Rank.R6, Suit.DIAMOND),
        Card(Rank.R8, Suit.DIAMOND),
        Card(Rank.R9, Suit.DIAMOND),
        Card(Rank.RJ, Suit.DIAMOND),
    ]
    state = make_state(
        hands,
        draw,
        revealed=[hands[0][0], hands[1][0], hands[2][0]],
        current=0,
    )

    state, _ = game.step(state, action_id(ComboKind.SINGLE, (Rank.R4,)))
    state, _ = game.step(state, action_id(ComboKind.SINGLE, (Rank.R7,)))
    state, result = game.step(state, PASS)

    assert result.trick_over and result.dug and result.done
    assert result.winner == 0  # the earliest (and only) empty seat digs
    assert result.refilled == (2,)  # seat 2 took the last four cards
    assert result.points_taken == 10  # K♥ swept from seat 1
    assert state.scores == (10, 0, 0)
    assert state.hands[1] == frozenset()  # the K was the whole hand
    assert state.hands[2] == frozenset(hands[2]) | frozenset(draw)  # no points
    assert state.empty_order == (0,)


@pytest.mark.parametrize("empty_order,digger", [((2, 0), 2), ((0, 2), 0)])
def test_first_empty_priority_picks_the_earliest_digger(empty_order, digger):
    # The trick ends with seats 0 and 2 both empty and the refill starving both;
    # seat 1 (not empty) drains the pile.  Whoever emptied first digs.
    game = Game(Rules(num_players=3))
    incumbent = classify([Card(Rank.R7, Suit.SPADE)])
    hands = [
        [],
        [Card(Rank.RK, Suit.HEART), Card(Rank.R4, Suit.SPADE)],
        [],
    ]
    draw = [Card(Rank.R6, Suit.DIAMOND), Card(Rank.R8, Suit.DIAMOND)]
    state = make_state(
        hands,
        draw,
        revealed=[
            Card(Rank.R4, Suit.DIAMOND),
            hands[1][0],
            Card(Rank.R6, Suit.CLUB),
        ],
        incumbent=incumbent,
        current=1,
        trick_cards=(Card(Rank.R7, Suit.SPADE),),
        last_player=0,
        empty_order=empty_order,
    )

    state, result = game.step(state, PASS)

    assert result.dug and result.done
    assert result.winner == digger
    assert result.refilled == (1,)  # only the non-empty seat can take cards
    assert result.points_taken == 10  # K♥ swept from seat 1
    assert state.scores[digger] == 10
    # seat 1 refilled with the two diamonds before the sweep took its K♥
    assert state.hands[1] == frozenset(
        {
            Card(Rank.R4, Suit.SPADE),
            Card(Rank.R6, Suit.DIAMOND),
            Card(Rank.R8, Suit.DIAMOND),
        }
    )


def test_first_empty_priority_follows_play_order_not_refill_rotation():
    # Seat 0 empties first, seat 3 second; seat 2 drains the refill pile before
    # either is reached, so both stay empty.  Seat 0 digs (earliest) even though
    # the refill rotation visits 3 before 0 — the order comes from play, not
    # from the refill cycle.
    game = Game(Rules(num_players=4))
    hands = [
        [Card(Rank.R4, Suit.DIAMOND)],
        [Card(Rank.R9, Suit.HEART), Card(Rank.R5, Suit.CLUB)],
        [
            Card(Rank.RJ, Suit.CLUB),
            Card(Rank.R4, Suit.SPADE),
            Card(Rank.R6, Suit.HEART),
        ],
        [Card(Rank.R6, Suit.CLUB)],
    ]
    draw = [
        Card(Rank.R10, Suit.DIAMOND),
        Card(Rank.RQ, Suit.DIAMOND),
        Card(Rank.RA, Suit.DIAMOND),
        Card(Rank.R3, Suit.DIAMOND),
    ]
    state = make_state(hands, draw, revealed=[hand[0] for hand in hands], current=0)

    state, _ = game.step(state, action_id(ComboKind.SINGLE, (Rank.R4,)))  # 0 out
    state, _ = game.step(state, PASS)  # 1
    state, _ = game.step(state, PASS)  # 2
    state, _ = game.step(state, action_id(ComboKind.SINGLE, (Rank.R6,)))  # 3 out
    state, _ = game.step(state, action_id(ComboKind.SINGLE, (Rank.R5,)))  # 1 wins
    state, result = game.step(state, PASS)  # 2 ends the trick

    assert result.trick_over and result.dug and result.done
    assert result.winner == 0
    assert result.refilled == (2,)  # seat 2 took the last four cards
    assert result.points_taken == 15  # the trick's 5♣ + the drawn 10♦
    assert state.scores == (15, 0, 0, 0)
    assert state.empty_order == (0, 3)


def test_immediate_dig_on_a_last_pair():
    # Emptiness is about the hand, not the card count: a final pair digs too.
    game = Game(DEFAULT_RULES)
    a = [Card(Rank.R5, Suit.DIAMOND), Card(Rank.R5, Suit.CLUB)]
    b = [Card(Rank.RK, Suit.HEART), Card(Rank.R10, Suit.SPADE)]
    state = make_state([a, b], [], revealed=[a[0], b[0]], current=0)

    state, result = game.step(state, action_id(ComboKind.PAIR, (Rank.R5,)))

    assert result.dug and result.done and result.winner == 0
    assert result.points_taken == 30  # 5+5 trick + K 10 + 10 10
    assert state.scores == (30, 0)
    assert state.hands[1] == frozenset()


@pytest.mark.parametrize("seat", [0, 1, 2])
def test_immediate_dig_three_players(seat):
    # Three-seat sanity: any seat can go out and sweep both opponents.
    game = Game(Rules(num_players=3))
    hands: list[list[Card]] = [[], [], []]
    hands[seat] = [Card(Rank.R5, Suit.DIAMOND)]
    hands[(seat + 1) % 3] = [Card(Rank.RK, Suit.HEART), Card(Rank.R4, Suit.SPADE)]
    hands[(seat + 2) % 3] = [Card(Rank.R10, Suit.CLUB), Card(Rank.R6, Suit.DIAMOND)]
    revealed = [hand[0] if hand else Card(Rank.R4, Suit.CLUB) for hand in hands]
    state = make_state(hands, [], revealed=revealed, current=seat)

    state, result = game.step(state, action_id(ComboKind.SINGLE, (Rank.R5,)))

    assert result.dug and result.done and result.winner == seat
    assert result.points_taken == 25  # trick 5 + K 10 + 10 10
    assert state.scores[seat] == 25
    assert state.hands[(seat + 1) % 3] == frozenset({Card(Rank.R4, Suit.SPADE)})
    assert state.hands[(seat + 2) % 3] == frozenset({Card(Rank.R6, Suit.DIAMOND)})


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
        trick_winner = state.last_player
        state, result = game.step(state, chosen, suit)
        if result.trick_over and result.refilled:
            assert trick_winner is not None
            offsets = [
                (filled - trick_winner - 1) % rules.num_players
                for filled in result.refilled
            ]
            assert offsets == sorted(offsets)
        steps += 1
        assert steps < 50_000, "episode did not terminate"
    check_invariants(state, rules)
    assert sum(state.scores) == 100
    assert state.draw_pile == ()
    return state


@pytest.mark.parametrize("num_players", [3, 4, 5])
def test_random_games_terminate_with_invariants_for_many_players(num_players):
    rules = Rules(num_players=num_players)
    for seed in range(10):
        play_episode_rules(
            rules,
            random.Random(seed),
            [RandomBot(random.Random(seed + seat)) for seat in range(num_players)],
        )


@pytest.mark.parametrize("num_players", [2, 3, 4])
def test_scripted_games_terminate_for_many_players(num_players):
    rules = Rules(num_players=num_players)
    for seed in range(10):
        play_episode_rules(
            rules,
            random.Random(100 + seed),
            [FirstLegalBot(rules) for _ in range(num_players)],
        )


def test_returns_are_zero_sum_for_many_players():
    for num_players in (3, 4):
        rules = Rules(num_players=num_players)
        game = Game(rules)
        state = play_episode_rules(
            rules,
            random.Random(7),
            [FirstLegalBot(rules) for _ in range(num_players)],
        )
        returns = game.returns(state)
        assert abs(sum(returns)) < 1e-9
        assert all(-1.0 <= value <= 1.0 for value in returns)


def test_fixed_bot_with_custom_rules_plays_full_games():
    rules = Rules(straight_max=13)
    for seed in range(2):
        play_episode_rules(
            rules, random.Random(seed), [FirstLegalBot(rules), FirstLegalBot(rules)]
        )


def test_seat_outcome_is_the_single_win_tie_loss_definition():
    assert seat_outcome((60, 40), 0) == 1
    assert seat_outcome((40, 60), 0) == -1
    assert seat_outcome((50, 50), 0) == 0
    assert seat_outcome((40, 30, 30), 0) == 1
    assert seat_outcome((40, 40, 20), 0) == 0
    assert seat_outcome((20, 40, 40), 0) == -1


def test_reveal_determines_starter_three_players():
    for seed in range(10):
        state = Game(Rules(num_players=3)).new(random.Random(seed))
        assert len(state.revealed) == 3
        smallest = min(state.revealed, key=card_key)
        assert state.revealed[state.current] == smallest


# -- public action log (Play / GameState.plays / View.plays) -----------------


def _trick_blocks(plays):
    """Split the action log into trick blocks (one opener each)."""
    blocks = []
    for play in plays:
        if play.opens_trick or not blocks:
            blocks.append([play])
        else:
            blocks[-1].append(play)
    return blocks


def test_plays_parity_with_played_and_trick_cards():
    """The card multiset of ``plays`` always equals ``played + trick_cards``."""
    for seed in range(6):
        game = Game()
        state = game.new(random.Random(seed))
        policy = FirstLegalBot()
        while not state.done:
            for seat in range(2):
                view = game.view(state, seat)
                cards = [card for play in view.plays for card in play.cards]
                assert cards == list(view.played) + list(view.trick_cards)
            action, suit = policy.act(game.view(state, state.current))
            state, _ = game.step(state, action, suit)


def test_opens_trick_counts_match_tricks():
    for seed in range(6):
        game = Game()
        state = game.new(random.Random(seed))
        policy = FirstLegalBot()
        started = 0
        while not state.done:
            plays = game.view(state, 0).plays
            # One opener per started trick, and the trick blocks partition the log.
            assert sum(play.opens_trick for play in plays) == started
            assert len(_trick_blocks(plays)) == started
            if game.view(state, state.current).incumbent is None:
                started += 1
            action, suit = policy.act(game.view(state, state.current))
            state, _ = game.step(state, action, suit)
        assert sum(play.opens_trick for play in state.plays) == started


def test_trick_winner_is_last_non_pass():
    """Normal tricks go to the last non-pass; a dig trick to the digger."""
    for seed in range(6):
        game = Game()
        state = game.new(random.Random(seed))
        policy = FirstLegalBot()
        while not state.done:
            action, suit = policy.act(game.view(state, state.current))
            state, result = game.step(state, action, suit)
            if not result.trick_over:
                continue
            block = _trick_blocks(state.plays)[-1]
            last_non_pass = [play.seat for play in block if play.cards][-1]
            if not result.dug:
                assert result.winner == last_non_pass
            else:
                # Immediate dig ends on the actor (the last non-pass); the
                # refill corner digs with ``empty_order[0]`` (ADR-0014).
                assert result.winner == last_non_pass or (
                    state.empty_order and result.winner == state.empty_order[0]
                )


def test_pass_only_when_incumbent():
    for seed in range(6):
        game = Game()
        state = game.new(random.Random(seed))
        policy = FirstLegalBot()
        while not state.done:
            view = game.view(state, state.current)
            passes = [play for play in view.plays if not play.cards]
            assert all(not play.opens_trick for play in passes)
            assert all(not play.went_out for play in passes)
            if view.incumbent is None:
                # Leading a fresh trick: pass is not in the catalog mask.
                assert not (view.mask >> PASS_ID) & 1
            action, suit = policy.act(view)
            state, _ = game.step(state, action, suit)


def test_plays_reconstructed_by_replay():
    from seven523.match import Match
    from seven523.trace import initial_snapshot, state_from_snapshot

    rules = DEFAULT_RULES
    match = Match(
        rules,
        [FirstLegalBot(rules) for _ in range(rules.num_players)],
        rng=random.Random(11),
    )
    snapshot = initial_snapshot(match.state)
    actions = []
    while not match.state.done:
        view = match.view()
        action, suit = match.policy_for(match.seat).act(view)
        actions.append((action, suit))
        match.step(action, suit)

    rebuilt = state_from_snapshot(snapshot, rules)
    game = Game(rules)
    assert rebuilt.plays == ()
    for action, suit in actions:
        rebuilt, _ = game.step(rebuilt, action, suit)
    assert rebuilt.plays == match.state.plays


def test_plays_view_does_not_leak_hidden_state():
    game = Game()
    own = [Card(Rank.R7, Suit.SPADE), Card(Rank.R3, Suit.HEART)]
    revealed = [Card(Rank.R3, Suit.HEART), Card(Rank.R2, Suit.SPADE)]
    hidden_a = [Card(Rank.R2, Suit.SPADE), Card(Rank.R5, Suit.CLUB)]
    hidden_b = [Card(Rank.R2, Suit.HEART), Card(Rank.R5, Suit.DIAMOND)]
    draw_a = [Card(Rank.R4, Suit.SPADE), Card(Rank.R6, Suit.CLUB)]
    draw_b = [Card(Rank.R6, Suit.CLUB), Card(Rank.R4, Suit.SPADE)]
    state_a = make_state([own, hidden_a], draw_a, revealed=revealed)
    state_b = make_state([own, hidden_b], draw_b, revealed=revealed)
    assert game.view(state_a, 0).plays == game.view(state_b, 0).plays == ()

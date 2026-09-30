"""Human-vs-bot terminal play: pure helpers + scripted-input game loop."""
import random
from typing import Any

import pytest
from support import FirstLegalBot

import seven523.play as play_module
from seven523.actions import PASS_ID, catalog_for, legal_ids, resolve, suit_options
from seven523.cards import Card, Rank, Suit, card_key
from seven523.combos import ComboKind
from seven523.game import Deal, Game, GameState, Phase, StepResult
from seven523.play import (
    ChooserPolicy,
    QuitGame,
    _trick_text,
    action_text,
    combo_text,
    interactive_chooser,
    main as play_main,
    opponent_identity,
    parse_choice,
    play_game,
    replay_trace,
    state_panel,
    suit_hint,
)
from seven523.policies import RandomBot
from seven523.rules import DEFAULT_RULES
from seven523.trace import (
    build_trace,
    load_trace,
    save_trace,
)


def make_state(seed=0):
    game = Game(DEFAULT_RULES)
    state = game.new(random.Random(seed))
    return game, state


def test_combo_and_action_text():
    game, state = make_state()
    view = game.view(state, state.current)
    legal = legal_ids(view.mask)
    assert legal
    for action_id in legal:
        text = action_text(game, action_id, view.hand)
        assert text and "[?]" not in text
    assert action_text(game, len(game.catalog) - 1, view.hand) == "过"

    first_play = next(
        action_id for action_id in legal if action_id != len(game.catalog) - 1
    )
    combo = resolve(game.catalog[first_play], view.hand)
    assert combo is not None
    assert combo_text(combo).split()[0] in {
        "单牌",
        "对子",
        "顺子",
        "连对",
        "小炸弹",
        "王炸",
        "大炸弹",
    }


def test_state_panel_shows_hand_and_scores():
    game, state = make_state(seed=3)
    panel = state_panel(game, state, human_seat=0)
    assert "你的手牌" in panel
    assert "分数" in panel
    assert "底牌堆" in panel


@pytest.mark.parametrize("human_seat", [0, 1])
def test_play_game_with_scripted_chooser_always_terminates(human_seat):
    rules = DEFAULT_RULES
    opponents = [FirstLegalBot(rules), FirstLegalBot(rules)]
    calls = []

    def chooser(game, state, view):
        legal = legal_ids(view.mask)
        assert legal
        calls.append(1)
        return legal[0]

    scores = play_game(
        opponents,
        chooser=chooser,
        rules=rules,
        human_seat=human_seat,
        seed=11,
        print_fn=lambda *args, **kwargs: None,
    )
    assert len(scores) == rules.num_players
    assert sum(scores) == rules.total_points
    assert calls, "the human should have had at least one turn"


def test_play_game_accepts_random_opponents():
    rules = DEFAULT_RULES
    opponents = [RandomBot(random.Random(0)), RandomBot(random.Random(1))]
    scores = play_game(
        opponents,
        chooser=lambda game, state, view: legal_ids(view.mask)[0],
        rules=rules,
        seed=5,
        print_fn=lambda *args, **kwargs: None,
    )
    assert sum(scores) == rules.total_points


def test_play_game_propagates_quit():
    rules = DEFAULT_RULES
    opponents = [RandomBot(), RandomBot()]

    def chooser(game, state, view):
        raise QuitGame

    with pytest.raises(QuitGame):
        play_game(
            opponents,
            chooser=chooser,
            rules=rules,
            seed=0,
            print_fn=lambda *args, **kwargs: None,
        )


# -- game traces -------------------------------------------------------------

SILENT = lambda *args, **kwargs: None  # noqa: E731


def _played_trace(chooser=None, *, rules=DEFAULT_RULES):
    opponents = [FirstLegalBot(rules), FirstLegalBot(rules)]
    record = {}
    scores = play_game(
        opponents,
        chooser=chooser
        or (lambda game, state, view: legal_ids(view.mask)[0]),
        rules=rules,
        seed=21,
        print_fn=SILENT,
        record=record,
    )
    trace = build_trace(
        rules,
        seed=21,
        human_seat=0,
        players=["human@seat0", "贪心 bot"],
        created_at="2026-01-01T00:00:00",
        **record,
    )
    return trace, scores


def test_trace_records_every_move_and_replays(tmp_path):
    trace, scores = _played_trace()
    assert trace["final_scores"] == list(scores)
    assert sum(trace["final_scores"]) == DEFAULT_RULES.total_points
    assert trace["steps"]
    assert all("scores" in step and "text" in step for step in trace["steps"])

    path = tmp_path / "game.json"
    save_trace(path, trace)
    loaded = load_trace(path)
    assert loaded["final_scores"] == trace["final_scores"]
    assert replay_trace(loaded, print_fn=SILENT) is True


def test_replay_detects_tampering(tmp_path):
    trace, _ = _played_trace()
    # PASS is never legal for the leader of a trick, so this must fail loudly.
    trace["steps"][0]["action"] = PASS_ID
    with pytest.raises(ValueError, match="illegal"):
        replay_trace(trace, print_fn=SILENT)


def test_replay_detects_score_tampering():
    trace, _ = _played_trace()
    trace["steps"][-1]["scores"] = [0, 0]
    with pytest.raises(ValueError, match="diverged"):
        replay_trace(trace, print_fn=SILENT)


def test_replay_can_be_paused_step_by_step():
    trace, _ = _played_trace()
    prompts = []

    def fake_input(prompt):
        prompts.append(prompt)
        return ""

    assert replay_trace(trace, print_fn=SILENT, pause=True, input_fn=fake_input)
    assert len(prompts) == len(trace["steps"])


# -- suit head in play / traces (ADR-0004) ------------------------------------


def _crafted_state(hands, *, current=0, incumbent=None):
    return GameState(
        hands=tuple(frozenset(hand) for hand in hands),
        draw_pile=(),
        scores=(0,) * len(hands),
        trick_points=0,
        trick_cards=(),
        revealed=tuple(hand[0] for hand in hands),
        current=current,
        incumbent=incumbent,
        last_player=None,
        collected=0,
        phase=Phase.PLAY,
    )


def test_suit_hint_and_parse_choice():
    game = Game(DEFAULT_RULES)
    hand = [
        Card(Rank.R7, Suit.SPADE),
        Card(Rank.R7, Suit.HEART),
        Card(Rank.R4, Suit.DIAMOND),
    ]
    state = _crafted_state([hand, [Card(Rank.R6, Suit.CLUB)]])
    view = game.view(state, 0)
    legal = legal_ids(view.mask)
    action_id = next(
        index
        for index in legal
        if game.catalog[index].kind is ComboKind.SINGLE
        and game.catalog[index].ranks == (Rank.R7,)
    )
    hint = suit_hint(game, action_id, view.hand)
    assert "♠" in hint and "♥" in hint
    assert parse_choice(str(action_id), legal, game, view) == (action_id, None)
    assert parse_choice(f"{action_id}♥", legal, game, view) == (action_id, 2)
    assert parse_choice(f"{action_id} ♥", legal, game, view) == (action_id, 2)
    with pytest.raises(ValueError, match="顶牌"):
        parse_choice(f"{action_id}♦", legal, game, view)  # no ♦7 in hand
    with pytest.raises(ValueError):
        parse_choice("999", legal, game, view)


def test_trace_records_and_replays_suit_choices():
    def weakest_top_suit_chooser(game, state, view):
        for action_id in legal_ids(view.mask):
            options = suit_options(game.catalog[action_id], view.hand, game.rules)
            if len(options) > 1:
                return action_id, int(options[-1])
        return legal_ids(view.mask)[0]

    trace, _ = _played_trace(weakest_top_suit_chooser)
    assert any(step["suit"] is not None for step in trace["steps"])
    assert replay_trace(trace, print_fn=SILENT) is True


def test_old_traces_without_suit_still_replay():
    trace, _ = _played_trace()
    for step in trace["steps"]:
        step.pop("suit", None)
    assert replay_trace(trace, print_fn=SILENT) is True


# -- rung-labelled opponent traces (M3) ---------------------------------------


def test_opponent_identity_labels_rungs_and_anchors():
    # The scripted bot is the study's pinned gauge.
    assert opponent_identity(None, "random") == ("anchor", "random")
    # A checkpoint carries its rung id, taken from the run directory.
    assert opponent_identity("runs/lvl3/agent.pt", "random") == ("opponent", "lvl3")
    assert opponent_identity("agent.pt", "random") == ("opponent", "agent")
    with pytest.raises(ValueError):
        opponent_identity(None, "mystery")
    with pytest.raises(ValueError):
        opponent_identity(None, "greedy")  # retired scripted bot


def test_play_main_writes_rung_labelled_trace(tmp_path, monkeypatch):
    def stub_chooser(human_seat, print_fn=print):
        def chooser(game, state, view):
            return legal_ids(view.mask)[0]

        return chooser

    monkeypatch.setattr("seven523.play.interactive_chooser", stub_chooser)
    path = tmp_path / "random.json"
    play_main(["--opponent", "random", "--seed", "5", "--save-trace", str(path)])
    trace = load_trace(path)
    assert trace["players"] == ["human@seat0", "anchor:random@seat1"]
    assert replay_trace(trace, print_fn=SILENT) is True


def test_interactive_chooser_is_exported_for_placement():
    # The placement CLI reuses this chooser; only its construction is checked
    # here (actually playing needs input()).
    assert callable(interactive_chooser(0, print_fn=SILENT))


# -- 撬底 message and bot-dig regression --------------------------------------


def _single_action_id(rank: Rank) -> int:
    return next(
        index
        for index, action in enumerate(catalog_for(DEFAULT_RULES))
        if action.kind is ComboKind.SINGLE and action.ranks == (rank,)
    )


class ScriptedSingles:
    """A ``Policy`` that leads a fixed sequence of single-card templates."""

    def __init__(self, ranks: list[Rank]) -> None:
        self._actions = [_single_action_id(rank) for rank in ranks]

    def act(self, view):
        return self._actions.pop(0), None


class AlwaysPass:
    def act(self, view):
        return PASS_ID


def _pass_chooser(game, state, view):
    return PASS_ID


def _bot_dig_deal(human_seat: int) -> Deal:
    """2-seat deal: the bot digs with its last card, sweeping the human hand.

    The human holds ♠K ♠10 ♠5 (25 points) and only ever passes; the bot holds
    the other three suits of 5/10/K (75 points), leads one card per trick and
    banks 65 before the last K triggers 撬底:
    65 (banked) + 10 (final trick) + 25 (swept hand) = 100.
    """
    human = frozenset(
        {
            Card(Rank.RK, Suit.SPADE),
            Card(Rank.R10, Suit.SPADE),
            Card(Rank.R5, Suit.SPADE),
        }
    )
    bot = frozenset(
        Card(rank, suit)
        for rank in (Rank.R5, Rank.R10, Rank.RK)
        for suit in (Suit.CLUB, Suit.DIAMOND, Suit.HEART)
    )
    hands = (human, bot) if human_seat == 0 else (bot, human)
    return Deal(
        hands=hands,
        draw_pile=(),
        revealed=tuple(min(hand, key=card_key) for hand in hands),
        starter=1 - human_seat,
    )


def _bot_dig_deal_with_human_hand(human_hand: frozenset[Card]):
    """Factory: the bot digs after leading every point card; human never plays.

    The bot holds all twelve 5/10/K cards (100 points) and empties its hand on
    the last one, so the human's hand is swept.  Vary ``human_hand`` to drive
    what the 撬底 message is allowed to claim about it.
    """

    def factory(human_seat: int) -> Deal:
        bot = frozenset(
            Card(rank, suit)
            for rank in (Rank.R5, Rank.R10, Rank.RK)
            for suit in (Suit.CLUB, Suit.DIAMOND, Suit.HEART, Suit.SPADE)
        )
        hands = (human_hand, bot) if human_seat == 0 else (bot, human_hand)
        return Deal(
            hands=hands,
            draw_pile=(),
            revealed=tuple(
                min(hand, key=card_key) if hand else Card(Rank.R3, Suit.CLUB)
                for hand in hands
            ),
            starter=1 - human_seat,
        )

    return factory


_BOT_DIG_ALL_POINTS = [Rank.R5] * 4 + [Rank.R10] * 4 + [Rank.RK] * 4


def _human_dig_deal(human_seat: int) -> Deal:
    """2-seat deal: the human plays its last card and digs the bot's hand."""
    human = frozenset({Card(Rank.R5, Suit.DIAMOND)})
    bot = frozenset(
        {
            Card(Rank.RK, Suit.HEART),
            Card(Rank.R10, Suit.HEART),
            Card(Rank.R5, Suit.HEART),
        }
    )
    hands = (human, bot) if human_seat == 0 else (bot, human)
    return Deal(
        hands=hands,
        draw_pile=(),
        revealed=tuple(min(hand, key=card_key) for hand in hands),
        starter=human_seat,
    )


def _play_on_crafted_deal(
    monkeypatch,
    deal_factory,
    human_seat: int,
    bot_policy,
    chooser,
    *,
    seed: int = 12345,
):
    """Drive ``play_game`` on a crafted deal instead of a random one.

    ``play_game`` builds its deal through ``Match``; the wrapper injects the
    crafted opening state for whichever seat is wrapped in the human's
    ``ChooserPolicy``, so the same deal works with the human at either seat.
    """
    real_match = play_module.Match

    class CraftedMatch(real_match):
        def __init__(self, rules, policies, *, state=None, rng=None):
            if state is None:
                seat = next(
                    index
                    for index, policy in enumerate(policies)
                    if isinstance(policy, ChooserPolicy)
                )
                state = Game(rules).restore(deal_factory(seat))
            super().__init__(rules, policies, state=state, rng=rng)

    monkeypatch.setattr(play_module, "Match", CraftedMatch)
    lines: list[str] = []
    record: dict[str, Any] = {}
    scores = play_game(
        [bot_policy, bot_policy],
        chooser=chooser,
        rules=DEFAULT_RULES,
        human_seat=human_seat,
        seed=seed,
        print_fn=lines.append,
        record=record,
    )
    return scores, record, lines


def test_trick_text_dug_message_names_the_swept_human_hand():
    dug_by_bot = StepResult(
        trick_over=True, winner=1, points_taken=35, dug=True, done=True
    )
    assert _trick_text(
        dug_by_bot, ["你", "对手"], human_seat=0, human_hand_points=True
    ) == ("墩结束：对手 收下 35 分，撬底！收走全场剩余分（含你手上的点牌）")
    dug_by_human = StepResult(
        trick_over=True, winner=0, points_taken=30, dug=True, done=True
    )
    assert _trick_text(
        dug_by_human, ["你", "对手"], human_seat=0, human_hand_points=False
    ) == ("墩结束：你 收下 30 分，撬底！收走全场剩余分")


def test_trick_text_dug_message_omits_sweep_clause_without_human_hand_points():
    # The bot triggers 撬底 but the human's pre-dig hand held no point cards:
    # the parenthetical must not claim a sweep of the human's hand (F1).
    dug_by_bot = StepResult(
        trick_over=True, winner=1, points_taken=10, dug=True, done=True
    )
    assert _trick_text(
        dug_by_bot, ["你", "对手"], human_seat=0, human_hand_points=False
    ) == ("墩结束：对手 收下 10 分，撬底！收走全场剩余分")


@pytest.mark.parametrize("human_seat", [0, 1])
def test_play_game_bot_dig_sweeps_human_hand_points_and_replays(
    monkeypatch, human_seat
):
    # Hand-computed predictions, independent of any engine-side sum.
    human_hand_points = 10 + 10 + 5  # ♠K + ♠10 + ♠5, still in hand at the dig
    bot_banked_before_dug = 3 * 5 + 3 * 10 + 2 * 10  # three 5s, three 10s, two Ks
    final_trick_points = 10  # the bot's last card is a K
    assert (human_hand_points, bot_banked_before_dug, final_trick_points) == (
        25,
        65,
        10,
    )

    ranks = [Rank.R5] * 3 + [Rank.R10] * 3 + [Rank.RK] * 3
    scores, record, lines = _play_on_crafted_deal(
        monkeypatch,
        _bot_dig_deal,
        human_seat,
        ScriptedSingles(ranks),
        _pass_chooser,
    )
    human, bot = human_seat, 1 - human_seat
    expected = (0, 100) if human_seat == 0 else (100, 0)
    assert scores == expected
    assert sum(scores) == 100  # all four suits of 5/10/K are in this deal

    dug_steps = [index for index, step in enumerate(record["steps"]) if step["dug"]]
    assert len(dug_steps) == 1
    dug_index = dug_steps[0]
    dug = record["steps"][dug_index]
    assert dug["points"] == final_trick_points + human_hand_points == 35
    assert dug["hand_sizes"] == [0, 0]  # the sweep empties the human's hand

    # The human only ever passed, so its bank must not move when the (empty)
    # hand points are swept; the bot's final score is bank + trick + hand.
    before = record["steps"][dug_index - 1]["scores"]
    expected_before = (
        [0, bot_banked_before_dug]
        if human_seat == 0
        else [bot_banked_before_dug, 0]
    )
    assert before == expected_before
    assert scores[human] == before[human] == 0
    assert scores[bot] == before[bot] + final_trick_points + human_hand_points
    assert scores[bot] == bot_banked_before_dug + final_trick_points + human_hand_points

    dug_lines = [line for line in lines if "撬底" in line]
    assert len(dug_lines) == 1
    assert "收走全场剩余分" in dug_lines[0]
    assert "含你手上的点牌" in dug_lines[0]

    trace = build_trace(
        DEFAULT_RULES,
        seed=12345,
        human_seat=human_seat,
        players=[
            f"human@seat{seat}" if seat == human_seat else f"anchor:random@seat{seat}"
            for seat in range(2)
        ],
        created_at="2026-01-01T00:00:00",
        **record,
    )
    replay_lines: list[str] = []
    assert replay_trace(trace, print_fn=replay_lines.append) is True
    replay_dug = [line for line in replay_lines if "撬底" in line]
    assert len(replay_dug) == 1 and "含你手上的点牌" in replay_dug[0]


@pytest.mark.parametrize("human_seat", [0, 1])
def test_play_game_bot_dig_omits_sweep_clause_for_point_free_human_hand(
    monkeypatch, human_seat
):
    # Human holds ♠9 only (0 points).  The bot still digs, but nothing in the
    # human's hand is swept, so the message must not claim there was (F1).
    human_hand = frozenset({Card(Rank.R9, Suit.SPADE)})
    scores, record, lines = _play_on_crafted_deal(
        monkeypatch,
        _bot_dig_deal_with_human_hand(human_hand),
        human_seat,
        ScriptedSingles(_BOT_DIG_ALL_POINTS),
        _pass_chooser,
    )
    expected = (0, 100) if human_seat == 0 else (100, 0)
    assert scores == expected

    dug = [step for step in record["steps"] if step["dug"]]
    assert len(dug) == 1
    assert dug[0]["points"] == 10  # the final trick only; the hand adds 0
    # The point-free card survives the sweep, so the hand was not emptied.
    assert dug[0]["hand_sizes"][human_seat] == len(human_hand) == 1

    dug_lines = [line for line in lines if "撬底" in line]
    assert len(dug_lines) == 1
    assert "收走全场剩余分" in dug_lines[0]
    assert "含你手上的点牌" not in dug_lines[0]

    trace = build_trace(
        DEFAULT_RULES,
        seed=12345,
        human_seat=human_seat,
        players=[
            f"human@seat{seat}" if seat == human_seat else f"anchor:random@seat{seat}"
            for seat in range(2)
        ],
        created_at="2026-01-01T00:00:00",
        **record,
    )
    replay_lines: list[str] = []
    assert replay_trace(trace, print_fn=replay_lines.append) is True
    replay_dug = [line for line in replay_lines if "撬底" in line]
    assert len(replay_dug) == 1
    assert "含你手上的点牌" not in replay_dug[0]


@pytest.mark.parametrize("human_seat", [0, 1])
def test_play_game_bot_dig_omits_sweep_clause_for_empty_human_hand(
    monkeypatch, human_seat
):
    # The human has no cards at all when the bot digs, so there is nothing to
    # claim about its hand (F1; mirrors the recorded hand_sizes [0, 0] trace).
    scores, record, lines = _play_on_crafted_deal(
        monkeypatch,
        _bot_dig_deal_with_human_hand(frozenset()),
        human_seat,
        ScriptedSingles(_BOT_DIG_ALL_POINTS),
        _pass_chooser,
    )
    expected = (0, 100) if human_seat == 0 else (100, 0)
    assert scores == expected

    dug = [step for step in record["steps"] if step["dug"]]
    assert len(dug) == 1
    assert dug[0]["points"] == 10  # the final trick only; the hand adds 0
    assert dug[0]["hand_sizes"] == [0, 0]

    dug_lines = [line for line in lines if "撬底" in line]
    assert len(dug_lines) == 1
    assert "收走全场剩余分" in dug_lines[0]
    assert "含你手上的点牌" not in dug_lines[0]

    trace = build_trace(
        DEFAULT_RULES,
        seed=12345,
        human_seat=human_seat,
        players=[
            f"human@seat{seat}" if seat == human_seat else f"anchor:random@seat{seat}"
            for seat in range(2)
        ],
        created_at="2026-01-01T00:00:00",
        **record,
    )
    replay_lines: list[str] = []
    assert replay_trace(trace, print_fn=replay_lines.append) is True
    replay_dug = [line for line in replay_lines if "撬底" in line]
    assert len(replay_dug) == 1
    assert "含你手上的点牌" not in replay_dug[0]


@pytest.mark.parametrize("human_seat", [0, 1])
def test_play_game_human_dig_message_omits_the_sweep_clause(monkeypatch, human_seat):
    # The human's 5 (5 points) plus the swept bot hand ♥K ♥10 ♥5 (25) = 30.
    chooser_calls: list[int] = []

    def play_the_last_card(game, state, view):
        chooser_calls.append(1)
        return _single_action_id(Rank.R5)

    scores, record, lines = _play_on_crafted_deal(
        monkeypatch,
        _human_dig_deal,
        human_seat,
        AlwaysPass(),
        play_the_last_card,
    )
    human, bot = human_seat, 1 - human_seat
    assert chooser_calls == [1]
    assert scores == ((30, 0) if human_seat == 0 else (0, 30))
    assert scores[human] == 5 + 25
    assert scores[bot] == 0

    dug = [step for step in record["steps"] if step["dug"]]
    assert len(dug) == 1 and dug[0]["points"] == 30
    dug_lines = [line for line in lines if "撬底" in line]
    assert len(dug_lines) == 1
    assert "收走全场剩余分" in dug_lines[0]
    assert "含你手上的点牌" not in dug_lines[0]  # the winner is the human

    trace = build_trace(
        DEFAULT_RULES,
        seed=12345,
        human_seat=human_seat,
        players=[
            f"human@seat{seat}" if seat == human_seat else f"anchor:random@seat{seat}"
            for seat in range(2)
        ],
        created_at="2026-01-01T00:00:00",
        **record,
    )
    replay_lines: list[str] = []
    assert replay_trace(trace, print_fn=replay_lines.append) is True
    replay_dug = [line for line in replay_lines if "撬底" in line]
    assert len(replay_dug) == 1 and "含你手上的点牌" not in replay_dug[0]


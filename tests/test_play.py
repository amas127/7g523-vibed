"""Human-vs-bot terminal play: pure helpers + scripted-input game loop."""
import random

import pytest

from seven523.actions import PASS_ID, legal_ids, resolve, suit_options
from seven523.cards import CARD_ORDER, Card, Rank, Suit
from seven523.combos import ComboKind
from seven523.game import Game, GameState, Phase
from seven523.policies import GreedyBot, RandomBot
from seven523.play import (
    QuitGame,
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
from seven523.trace import (
    build_trace,
    card_from_json,
    card_json,
    initial_snapshot,
    load_trace,
    save_trace,
    state_from_snapshot,
)
from seven523.rules import DEFAULT_RULES


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
    opponents = [GreedyBot(rules), GreedyBot(rules)]
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


def test_card_json_round_trip():
    for card in (CARD_ORDER[0], CARD_ORDER[13], CARD_ORDER[-1]):
        assert card_from_json(card_json(card)) == card


def test_state_from_snapshot_rebuilds_the_deal():
    game, state = make_state(seed=9)
    rebuilt = state_from_snapshot(initial_snapshot(state), DEFAULT_RULES)
    assert rebuilt.hands == state.hands
    assert rebuilt.draw_pile == state.draw_pile
    assert rebuilt.revealed == state.revealed
    assert rebuilt.current == state.current
    assert rebuilt.played == ()


def _played_trace(chooser=None, *, rules=DEFAULT_RULES):
    opponents = [GreedyBot(rules), GreedyBot(rules)]
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
    # Scripted bots are the study's pinned anchors.
    assert opponent_identity(None, "greedy") == ("anchor", "greedy")
    assert opponent_identity(None, "random") == ("anchor", "random")
    # A checkpoint carries its rung id, taken from the run directory.
    assert opponent_identity("runs/lvl3/agent.pt", "greedy") == ("opponent", "lvl3")
    assert opponent_identity("agent.pt", "greedy") == ("opponent", "agent")
    with pytest.raises(ValueError):
        opponent_identity(None, "mystery")


def test_play_main_writes_rung_labelled_trace(tmp_path, monkeypatch):
    def stub_chooser(human_seat, print_fn=print):
        def chooser(game, state, view):
            return legal_ids(view.mask)[0]

        return chooser

    monkeypatch.setattr("seven523.play.interactive_chooser", stub_chooser)
    path = tmp_path / "greedy.json"
    play_main(["--opponent", "greedy", "--seed", "5", "--save-trace", str(path)])
    trace = load_trace(path)
    assert trace["players"] == ["human@seat0", "anchor:greedy@seat1"]
    assert replay_trace(trace, print_fn=SILENT) is True


def test_interactive_chooser_is_exported_for_placement():
    # The placement CLI reuses this chooser; only its construction is checked
    # here (actually playing needs input()).
    assert callable(interactive_chooser(0, print_fn=SILENT))

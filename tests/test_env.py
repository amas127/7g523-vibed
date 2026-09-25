import importlib.util
import json
import random
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

from seven523.actions import CATALOG, SUIT_N, catalog_for, joint_mask_bits
from seven523.cards import (
    CARD_ORDER,
    RANK_INDEX,
    Card,
    Rank,
    Suit,
    card_id,
    make_deck,
)
from seven523.combos import Combo, ComboKind
from seven523.env import (
    OBS_VERSION,
    Seven523Env,
    encode_observation,
    observation_dim,
)
from seven523.game import Game, GameState, Phase, View, seat_outcome
from seven523.policies import GreedyBot, MixturePolicy, RandomBot
from seven523.rules import DEFAULT_RULES, Rules

#: Pilot layout-b encoder; gitignored, so the bit-for-bit comparison against it
#: is skipped when the file is absent.  It produced ``_OBS_V5_GOLDEN``.
_PILOT_ENCODER = (
    Path(__file__).resolve().parents[1] / "runs" / "obs-slim" / "obs_slim_encoder.py"
)

#: v5 snapshots: the pilot's layout b prefix plus independently computed B0/B1.
_OBS_V5_GOLDEN = Path(__file__).parent / "data" / "obs_v5_golden.json"


def _load_v5_golden():
    return json.loads(_OBS_V5_GOLDEN.read_text())


def _load_pilot_encoder():
    spec = importlib.util.spec_from_file_location("pilot_obs_slim_encoder", _PILOT_ENCODER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _card(spec):
    rank, suit = spec
    return Card(Rank(rank), None if suit is None else Suit(suit))


def _combo(spec):
    if spec is None:
        return None
    return Combo(ComboKind(spec["kind"]), tuple(_card(c) for c in spec["cards"]))


def _view(spec):
    """Rebuild a :class:`View` from the JSON fixture."""
    return View(
        seat=spec["seat"],
        hand=frozenset(_card(c) for c in spec["hand"]),
        mask=spec["mask"],
        incumbent=_combo(spec["incumbent"]),
        current=spec["current"],
        scores=tuple(spec["scores"]),
        counts=tuple(spec["counts"]),
        draw_count=spec["draw_count"],
        revealed=tuple(_card(c) for c in spec["revealed"]),
        trick_cards=tuple(_card(c) for c in spec["trick_cards"]),
        played=tuple(_card(c) for c in spec.get("played", ())),
        last_player=spec["last_player"],
        done=spec["done"],
    )


#: The v5 segment order and fixed widths; variable blocks expand per player.
_V5_SEGMENTS = (
    ("hand", 54),
    ("inc_rank", 15),
    ("inc_suit", 4),
    ("inc_kind", 6),
    ("inc_size", 1),
    ("draw", 1),
    ("scores", None),
    ("opp_count", None),
    ("opp_revealed", None),
    ("trick_points", 1),
    ("remaining_points", 1),
    ("point_hold", 1),
    ("unseen", 54),
    ("last_player", 1),
)


def _span_map(num_players):
    """``name -> (start, width)`` for the v5 layout at ``num_players``."""
    spans: dict[str, tuple[int, int]] = {}
    offset = 0
    for name, width in _V5_SEGMENTS:
        if name == "scores":
            width = num_players
        elif name == "opp_count":
            width = num_players - 1
        elif name == "opp_revealed":
            width = 19 * (num_players - 1)
        assert width is not None
        spans[name] = (offset, width)
        offset += width
    return spans


def test_observation_dim_is_the_only_v5_layout():
    assert OBS_VERSION == 5
    assert observation_dim(2) == 161
    assert observation_dim(3) == 182
    assert observation_dim(7) == 266


@pytest.mark.parametrize("num_players", [2, 3, 4, 5, 6, 7])
def test_observation_dim_formula(num_players):
    assert observation_dim(num_players) == 119 + 21 * num_players


@pytest.mark.parametrize("num_players", [2, 3, 4, 5, 6, 7])
def test_encode_observation_length_matches_dim(num_players):
    rules = Rules(num_players=num_players)
    game = Game(rules)
    state = game.new(random.Random(0))
    view = game.view(state, 0)
    obs = encode_observation(view, rules)
    assert len(obs) == observation_dim(num_players)


@pytest.mark.parametrize("num_players", [2, 3, 7])
def test_v5_layout_spans_are_contiguous_and_cover_the_dimension(num_players):
    spans = _span_map(num_players)
    offset = 0
    for name, _fixed_width in _V5_SEGMENTS:
        start, width = spans[name]
        assert start == offset
        assert width > 0
        offset += width
    assert offset == observation_dim(num_players)


def test_v5_layout_matches_the_documented_offsets():
    assert tuple(_span_map(2).items()) == (
        ("hand", (0, 54)),
        ("inc_rank", (54, 15)),
        ("inc_suit", (69, 4)),
        ("inc_kind", (73, 6)),
        ("inc_size", (79, 1)),
        ("draw", (80, 1)),
        ("scores", (81, 2)),
        ("opp_count", (83, 1)),
        ("opp_revealed", (84, 19)),
        ("trick_points", (103, 1)),
        ("remaining_points", (104, 1)),
        ("point_hold", (105, 1)),
        ("unseen", (106, 54)),
        ("last_player", (160, 1)),
    )


def test_b1_unseen_segment_is_the_public_set_complement():
    """``unseen`` = 54 − own hand − revealed − played − current trick."""
    rules = DEFAULT_RULES
    deck = make_deck()
    hand = frozenset(deck[0:5])
    revealed = (deck[5], deck[6])
    played = tuple(deck[7:20])
    trick = tuple(deck[20:23])
    view = View(
        seat=0,
        hand=hand,
        mask=0,
        incumbent=None,
        current=0,
        scores=(20, 30),
        counts=(5, 7),
        draw_count=24,
        revealed=revealed,
        trick_cards=trick,
        played=played,
        last_player=1,
        done=False,
    )
    obs = encode_observation(view, rules)
    base, width = _span_map(2)["unseen"]
    assert width == 54
    seen = set(hand) | set(revealed) | set(played) | set(trick)
    assert sum(obs[base : base + width]) == 54 - len(seen)
    for card in CARD_ORDER:
        assert obs[base + card_id(card)] == (0.0 if card in seen else 1.0)


@pytest.mark.parametrize("num_players", [2, 3])
def test_b1_last_player_segment_normalises_the_incumbent_owner(num_players):
    """None (no incumbent) is 0.0; seat s is (s + 1) / num_players."""
    rules = Rules(num_players=num_players)
    base, width = _span_map(num_players)["last_player"]
    assert width == 1
    for last_player in (None, *range(num_players)):
        view = View(
            seat=0,
            hand=frozenset(),
            mask=0,
            incumbent=None,
            current=0,
            scores=(0,) * num_players,
            counts=(0,) * num_players,
            draw_count=0,
            revealed=tuple(Card(Rank.R4, Suit.SPADE) for _ in range(num_players)),
            trick_cards=(),
            played=(),
            last_player=last_player,
            done=False,
        )
        obs = encode_observation(view, rules)
        expected = 0.0 if last_player is None else (last_player + 1) / num_players
        assert obs[base] == pytest.approx(expected)


def test_s1_segments_reencode_every_public_field():
    """The slim blocks are lossless decodes of the public fields."""
    fixture = _load_v5_golden()
    assert fixture["cases"]
    for case in fixture["cases"]:
        rules = Rules(**case["rules"])
        view = _view(case["view"])
        players = rules.num_players
        obs = encode_observation(view, rules)
        spans = _span_map(players)

        start, width = spans["hand"]
        assert width == 54
        for card in CARD_ORDER:
            assert obs[start + card_id(card)] == (
                1.0 if card in view.hand else 0.0
            )

        rank_start, _ = spans["inc_rank"]
        suit_start, _ = spans["inc_suit"]
        if view.incumbent is None:
            assert obs[rank_start : rank_start + 15] == [0.0] * 15
            assert obs[suit_start : suit_start + 4] == [0.0] * 4
        else:
            top = view.incumbent.top_card
            assert sum(obs[rank_start : rank_start + 15]) == 1.0
            assert obs[rank_start + RANK_INDEX[top.rank]] == 1.0
            if top.suit is None:
                assert obs[suit_start : suit_start + 4] == [0.0] * 4
            else:
                assert sum(obs[suit_start : suit_start + 4]) == 1.0
                assert obs[suit_start + int(top.suit)] == 1.0

        start, _ = spans["scores"]
        for k in range(players):
            assert obs[start + k] == pytest.approx(
                view.scores[(view.seat + k) % players] / rules.total_points
            )
        start, width = spans["opp_count"]
        assert width == players - 1
        for k in range(1, players):
            assert obs[start + k - 1] == pytest.approx(
                view.counts[(view.seat + k) % players] / rules.hand_size
            )
        start, width = spans["opp_revealed"]
        assert width == 19 * (players - 1)
        for k in range(1, players):
            chunk = start + 19 * (k - 1)
            card = view.revealed[(view.seat + k) % players]
            assert sum(obs[chunk : chunk + 15]) == 1.0
            assert obs[chunk + RANK_INDEX[card.rank]] == 1.0
            if card.suit is None:
                assert obs[chunk + 15 : chunk + 19] == [0.0] * 4
            else:
                assert sum(obs[chunk + 15 : chunk + 19]) == 1.0
                assert obs[chunk + 15 + int(card.suit)] == 1.0

        start, width = spans["last_player"]
        assert width == 1
        expected = (
            0.0
            if view.last_player is None
            else (view.last_player + 1) / players
        )
        assert obs[start] == pytest.approx(expected)


def test_s1_inc_suit_and_opp_revealed_leave_jokers_all_zero():
    rules = Rules(num_players=2)
    joker = Card(Rank.SMALL_JOKER, None)
    view = View(
        seat=1,
        hand=frozenset({Card(Rank.RA, Suit.SPADE)}),
        mask=0,
        incumbent=Combo(ComboKind.SINGLE, (joker,)),
        current=0,
        scores=(0, 0),
        counts=(1, 1),
        draw_count=52,
        revealed=(joker, Card(Rank.R4, Suit.CLUB)),
        trick_cards=(joker,),
        played=(),
        last_player=0,
        done=False,
    )
    obs = encode_observation(view, rules)
    spans = _span_map(2)
    rank_start, _ = spans["inc_rank"]
    suit_start, _ = spans["inc_suit"]
    opp_start, _ = spans["opp_revealed"]
    assert obs[rank_start + RANK_INDEX[Rank.SMALL_JOKER]] == 1.0
    assert obs[suit_start : suit_start + 4] == [0.0] * 4
    # opponent seat 0's revealed joker: rank slot set, suit chunk zero
    assert obs[opp_start + RANK_INDEX[Rank.SMALL_JOKER]] == 1.0
    assert obs[opp_start + 15 : opp_start + 19] == [0.0] * 4
    # the own revealed card (seat 1's R4 of clubs) is never encoded by S1
    assert obs[opp_start + RANK_INDEX[Rank.R4]] == 0.0
    assert obs[opp_start + 15 + int(Suit.CLUB)] == 0.0


def test_v5_golden_fixture_matches_the_encoder():
    """The committed v5 snapshots are pilot layout b + independent B0/B1."""
    fixture = _load_v5_golden()
    assert fixture["cases"]
    assert fixture["source"]
    for case in fixture["cases"]:
        rules = Rules(**case["rules"])
        view = _view(case["view"])
        obs = encode_observation(view, rules)
        assert len(obs) == len(case["obs_v5"])
        assert obs == case["obs_v5"], case["label"]
        assert obs[: len(case["obs_b"])] == case["obs_b"], case["label"]


@pytest.mark.skipif(
    not _PILOT_ENCODER.is_file(),
    reason="pilot layout-b encoder is gitignored and absent",
)
def test_v5_prefix_matches_the_pilot_layout_b_encoder():
    fixture = _load_v5_golden()
    pilot = _load_pilot_encoder()
    for case in fixture["cases"]:
        rules = Rules(**case["rules"])
        view = _view(case["view"])
        pilot_b = pilot.encode_b(view, rules)
        assert pilot_b == case["obs_b"], case["label"]
        obs = encode_observation(view, rules)
        assert obs[: len(pilot_b)] == pilot_b, case["label"]


@pytest.mark.skipif(
    not _PILOT_ENCODER.is_file(),
    reason="pilot layout-b encoder is gitignored and absent",
)
@pytest.mark.parametrize("num_players", [2, 3, 4, 5, 6, 7])
def test_v5_prefix_matches_the_pilot_for_every_table_size(num_players):
    """The S1 prefix is self-centrically identical to layout b, for any n."""
    pilot = _load_pilot_encoder()
    rules = Rules(num_players=num_players)
    game = Game(rules)
    state = game.new(random.Random(num_players))
    for seat in range(num_players):
        view = game.view(state, seat)
        pilot_b = pilot.encode_b(view, rules)
        obs = encode_observation(view, rules)
        assert obs[: len(pilot_b)] == pilot_b, seat


def test_encoding_does_not_leak_hidden_cards():
    """Two states differing only in hidden cards encode bit-for-bit equal.

    ADR-0002: the public projection may not carry opponent hands or the draw
    pile order.  The B0 block is derived from scores, the current trick and
    the own hand only; B1's ``unseen`` is a public set complement, so swapping
    which hidden cards sit in an opponent hand versus the pile is invisible.
    """
    rules = DEFAULT_RULES
    own = [Card(Rank.R5, Suit.SPADE), Card(Rank.R9, Suit.HEART)]
    revealed = [Card(Rank.R4, Suit.SPADE), Card(Rank.R2, Suit.HEART)]
    trick = (Card(Rank.R10, Suit.DIAMOND), Card(Rank.RK, Suit.CLUB))
    played = (Card(Rank.R3, Suit.SPADE), Card(Rank.R7, Suit.HEART))
    hidden_a = [Card(Rank.R5, Suit.CLUB), Card(Rank.R3, Suit.DIAMOND)]
    hidden_b = [Card(Rank.R6, Suit.CLUB), Card(Rank.R8, Suit.DIAMOND)]
    draw_a = [Card(Rank.R7, Suit.SPADE), Card(Rank.R5, Suit.DIAMOND)]
    draw_b = [Card(Rank.R6, Suit.SPADE), Card(Rank.R4, Suit.DIAMOND)]

    def state(hidden, draw):
        return GameState(
            hands=(frozenset(own), frozenset(hidden)),
            draw_pile=tuple(draw),
            scores=(25, 15),
            trick_points=20,
            trick_cards=trick,
            revealed=tuple(revealed),
            current=0,
            incumbent=None,
            last_player=1,
            collected=4,
            phase=Phase.PLAY,
            played=played,
        )

    game = Game(rules)
    view_a = game.view(state(hidden_a, draw_a), 0)
    view_b = game.view(state(hidden_b, draw_b), 0)
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
    assert encode_observation(
        view_a, rules
    ) == encode_observation(view_b, rules)


def test_env_publishes_the_v5_observation():
    env = Seven523Env(seed=0, opponents=[GreedyBot(), GreedyBot()])
    assert env.obs_dim == observation_dim(2) == 161
    assert env.observation_space.shape == (env.obs_dim,)
    obs, _ = env.reset()
    assert np.allclose(obs, encode_observation(env.view(), env.rules))


def run_env_episode(env, seed=0):
    obs, info = env.reset()
    assert info == {}
    assert obs.shape == env.observation_space.shape
    assert env.observation_space.contains(obs)
    assert len(env.action_mask) == int(env.action_space.nvec.sum())
    rng = random.Random(seed)
    done = False
    reward = 0.0
    steps = 0
    while not done:
        legal = [
            index
            for index in range(env.action_space_n)
            if env.action_mask[index]
        ]
        assert legal, "the learner must always have a legal action"
        obs, reward, done, truncated, info = env.step(
            (rng.choice(legal), rng.randrange(SUIT_N))
        )
        assert obs.shape == env.observation_space.shape
        assert not truncated
        if not done:
            assert info == {}
        steps += 1
        assert steps < 50_000
    assert -1.0 <= reward <= 1.0
    return reward


def test_env_spaces():
    env = Seven523Env(seed=0)
    assert isinstance(env.action_space, gym.spaces.MultiDiscrete)
    assert env.action_space.nvec.tolist() == [len(CATALOG), SUIT_N]
    assert env.observation_space.shape == (observation_dim(2),)
    assert env.observation_space.dtype == np.float32


def test_env_state_before_reset_raises():
    env = Seven523Env(seed=0)
    with pytest.raises(RuntimeError):
        _ = env.state


def test_env_action_space_matches_rules_catalog():
    for rules in (
        Rules(),
        Rules(num_players=3),
        Rules(straight_max=13),
        Rules(num_players=3, straight_max=13, consecutive_pairs_max=6),
    ):
        env = Seven523Env(rules=rules, seed=0)
        assert env.action_space_n == len(catalog_for(rules))
        assert env.action_space.nvec.tolist() == [len(catalog_for(rules)), SUIT_N]
        run_env_episode(env)


def test_env_custom_rules_episode_is_finite_and_legal():
    rules = Rules(num_players=3, straight_max=13)
    env = Seven523Env(rules=rules, seed=1)
    run_env_episode(env, seed=1)


def test_mixture_policy_draws_members_by_weight():
    class _Fixed:
        def __init__(self, action):
            self.action = action

        def act(self, view):
            return self.action, None

    low = _Fixed(0)
    high = _Fixed(1)
    mixture = MixturePolicy([(0.25, low), (0.75, high)], random.Random(0))
    picks = [mixture.act(object())[0] for _ in range(400)]
    share = picks.count(1) / len(picks)
    assert 0.65 < share < 0.85

    only_low = MixturePolicy([(1.0, low), (0.0, high)], random.Random(0))
    assert all(only_low.act(object())[0] == 0 for _ in range(20))


def test_mixture_policy_returns_legal_actions():
    env = Seven523Env(seed=0, opponents=[GreedyBot(), GreedyBot()])
    env.reset()
    view = env.game.view(env.state, env.learner)
    mixture = MixturePolicy(
        [(0.5, GreedyBot()), (0.5, RandomBot(random.Random(0)))], random.Random(0)
    )
    for _ in range(20):
        action_id, _suit = mixture.act(view)
        assert view.mask >> action_id & 1


def test_env_learner_not_zero():
    rules = Rules(num_players=3)
    env = Seven523Env(
        rules=rules,
        seed=2,
        learner=2,
        opponents=[GreedyBot(rules) for _ in range(3)],
    )
    run_env_episode(env, seed=2)


def test_env_action_mask_matches_the_learner_view():
    env = Seven523Env(seed=3)
    env.reset()
    view = env.game.view(env.state, env.learner)
    assert env.action_mask == joint_mask_bits(view.mask, env.nvec)
    # the published mask is never empty in a playable state
    assert any(env.action_mask) or view.done


def test_env_reset_with_the_same_seed_is_reproducible():
    first = Seven523Env(seed=0, opponents=[GreedyBot(), GreedyBot()])
    second = Seven523Env(seed=0, opponents=[GreedyBot(), GreedyBot()])
    obs_a, _ = first.reset(seed=42)
    obs_b, _ = second.reset(seed=42)
    assert np.array_equal(obs_a, obs_b)
    assert first.action_mask == second.action_mask


# -- reward shaping (T1) -----------------------------------------------------


def _play_episode(env, action_seed=0):
    """Drive one full episode with deterministic random legal actions.

    Returns the list of per-step rewards.  Opponents are whatever the env was
    built with; identical seeds + identical action stream give identical
    dynamics for the shaping modes under test.
    """
    env.reset()
    rng = random.Random(action_seed)
    rewards: list[float] = []
    done = False
    while not done:
        legal = [i for i in range(env.action_space_n) if env.action_mask[i]]
        assert legal
        _obs, reward, done, truncated, _info = env.step(
            (rng.choice(legal), rng.randrange(SUIT_N))
        )
        rewards.append(reward)
        assert not truncated
    return rewards


def _random_opponents(rules, seed):
    """Fresh, identically seeded random opponents (one per seat)."""
    return [
        RandomBot(random.Random(seed * 1000 + seat))
        for seat in range(rules.num_players)
    ]


def _win_bonus(state, learner):
    """The production win/tie/loss definition, under test."""
    return float(seat_outcome(state.scores, learner))


def test_reward_shaping_defaults_to_terminal_and_rejects_unknown_modes():
    assert Seven523Env(seed=0).reward_shaping == "terminal"
    for mode in ("terminal", "trick_diff", "win", "trick_diff_win"):
        assert Seven523Env(seed=0, reward_shaping=mode).reward_shaping == mode
    for bad in ("", "none", "TERMINAL", "trickdiff"):
        with pytest.raises(ValueError, match="reward_shaping"):
            Seven523Env(seed=0, reward_shaping=bad)


def test_terminal_mode_rewards_are_zero_until_the_terminal_return():
    rules = Rules()
    env = Seven523Env(
        rules=rules, seed=0, opponents=_random_opponents(rules, 0)
    )
    rewards = _play_episode(env, action_seed=0)
    assert len(rewards) > 1
    assert all(reward == 0.0 for reward in rewards[:-1])
    assert rewards[-1] == env.game.returns(env.state)[env.learner]


def test_terminal_step_publishes_scores_and_outcome():
    rules = Rules()
    env = Seven523Env(rules=rules, seed=0, opponents=_random_opponents(rules, 0))
    env.reset()
    rng = random.Random(0)
    done = False
    info: dict = {}
    while not done:
        legal = [i for i in range(env.action_space_n) if env.action_mask[i]]
        _obs, _reward, done, _truncated, info = env.step(
            (rng.choice(legal), rng.randrange(SUIT_N))
        )
    assert info["scores"] == list(env.state.scores)
    assert info["outcome"] == seat_outcome(env.state.scores, env.learner)
    assert sum(info["scores"]) == rules.total_points


@pytest.mark.parametrize("num_players", [2, 3])
def test_trick_diff_rewards_telescope_to_the_terminal_return(num_players):
    rules = Rules(num_players=num_players)
    env = Seven523Env(
        rules=rules,
        seed=1,
        opponents=_random_opponents(rules, 1),
        reward_shaping="trick_diff",
    )
    for episode in range(3):
        rewards = _play_episode(env, action_seed=episode)
        assert env.state.done  # every game ends with 撬底
        terminal_return = env.game.returns(env.state)[env.learner]
        assert sum(rewards) == pytest.approx(terminal_return, abs=1e-9)


def test_trick_diff_has_far_more_nonzero_reward_steps_than_terminal():
    rules = Rules()
    terminal = Seven523Env(
        rules=rules, seed=2, opponents=_random_opponents(rules, 2)
    )
    trick = Seven523Env(
        rules=rules,
        seed=2,
        opponents=_random_opponents(rules, 2),
        reward_shaping="trick_diff",
    )
    episodes = 3
    term_nonzero = sum(
        reward != 0.0
        for episode in range(episodes)
        for reward in _play_episode(terminal, action_seed=episode)
    )
    trick_nonzero = sum(
        reward != 0.0
        for episode in range(episodes)
        for reward in _play_episode(trick, action_seed=episode)
    )
    assert term_nonzero <= episodes  # at most the terminal step can be nonzero
    assert trick_nonzero >= 5 * max(1, term_nonzero)


@pytest.mark.parametrize("num_players", [2, 3])
def test_win_mode_reward_is_the_sign_of_the_score_gap(num_players):
    rules = Rules(num_players=num_players)
    env = Seven523Env(
        rules=rules,
        seed=3,
        opponents=_random_opponents(rules, 3),
        reward_shaping="win",
    )
    for episode in range(3):
        rewards = _play_episode(env, action_seed=episode)
        assert all(reward == 0.0 for reward in rewards[:-1])
        assert rewards[-1] in (-1.0, 0.0, 1.0)
        assert rewards[-1] == _win_bonus(env.state, env.learner)


def test_trick_diff_win_adds_the_win_bonus_on_the_terminal_step_only():
    rules = Rules()
    opponents_a = _random_opponents(rules, 4)
    opponents_b = _random_opponents(rules, 4)
    diff_env = Seven523Env(
        rules=rules, seed=4, opponents=opponents_a, reward_shaping="trick_diff"
    )
    both_env = Seven523Env(
        rules=rules, seed=4, opponents=opponents_b, reward_shaping="trick_diff_win"
    )
    for episode in range(3):
        diff_rewards = _play_episode(diff_env, action_seed=episode)
        both_rewards = _play_episode(both_env, action_seed=episode)
        assert len(both_rewards) == len(diff_rewards)
        win = _win_bonus(both_env.state, both_env.learner)
        terminal_return = both_env.game.returns(both_env.state)[both_env.learner]
        assert sum(both_rewards) == pytest.approx(terminal_return + win, abs=1e-9)
        assert both_rewards[:-1] == pytest.approx(diff_rewards[:-1], abs=1e-12)
        assert both_rewards[-1] == pytest.approx(diff_rewards[-1] + win, abs=1e-12)


def test_env_episode_is_finite_and_legal():
    env = Seven523Env(
        opponents=[RandomBot(random.Random(0)), GreedyBot()],
        seed=5,
    )
    obs, _ = env.reset()
    assert obs.shape == env.observation_space.shape
    assert len(env.action_mask) == len(CATALOG) + SUIT_N
    assert any(env.action_mask[: len(CATALOG)])

    rng = random.Random(6)
    done = False
    reward = 0.0
    steps = 0
    while not done:
        legal = [index for index in range(env.action_space_n) if env.action_mask[index]]
        assert legal
        obs, reward, done, truncated, info = env.step(
            (rng.choice(legal), rng.randrange(SUIT_N))
        )
        assert obs.shape == env.observation_space.shape
        assert not truncated
        if not done:
            assert info == {}
        steps += 1
        assert steps < 20_000, "env episode did not terminate"

    assert -1.0 <= reward <= 1.0

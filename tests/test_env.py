import importlib.util
import json
import math
import random
from dataclasses import replace
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
from support import FirstLegalBot

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
from seven523.game import Game, GameState, Phase, Play, View, seat_outcome
from seven523.policies import MixturePolicy, RandomBot
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
        plays=tuple(
            Play(
                seat=play["seat"],
                cards=tuple(_card(c) for c in play["cards"]),
                opens_trick=play["opens_trick"],
                went_out=play["went_out"],
            )
            for play in spec.get("plays", ())
        ),
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


def test_empty_order_stays_out_of_the_view_and_observation():
    """The revision-3 emptiness order is engine-internal (ADR-0002/ADR-0014)."""
    rules = DEFAULT_RULES
    game = Game(rules)
    base = game.new(random.Random(0))
    before = game.view(base, 0)
    state_a = replace(base, empty_order=())
    state_b = replace(base, empty_order=(0, 1))
    view_a = game.view(state_a, 0)
    view_b = game.view(state_b, 0)
    assert view_a == view_b == before
    assert not hasattr(view_a, "empty_order")
    assert encode_observation(view_a, rules) == encode_observation(view_b, rules)


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
    env = Seven523Env(seed=0, opponents=[FirstLegalBot(), FirstLegalBot()])
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
    assert -2.0 <= reward <= 2.0  # ADR-0016 default arcsin + λ=1 range
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
    env = Seven523Env(seed=0, opponents=[FirstLegalBot(), FirstLegalBot()])
    env.reset()
    view = env.game.view(env.state, env.learner)
    mixture = MixturePolicy(
        [(0.5, FirstLegalBot()), (0.5, RandomBot(random.Random(0)))], random.Random(0)
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
        opponents=[FirstLegalBot(rules) for _ in range(3)],
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
    first = Seven523Env(seed=0, opponents=[FirstLegalBot(), FirstLegalBot()])
    second = Seven523Env(seed=0, opponents=[FirstLegalBot(), FirstLegalBot()])
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


def test_reward_shaping_defaults_to_arcsin_and_rejects_unknown_modes():
    default = Seven523Env(seed=0)
    assert default.reward_shaping == "arcsin"  # ADR-0016 default recipe
    assert default.arcsin_mix == 0.5
    assert default.win_jump == 1.0
    assert default.reward_cap is None
    for mode in ("terminal", "trick_diff", "win", "trick_diff_win", "terminal_win", "arcsin"):
        assert Seven523Env(seed=0, reward_shaping=mode).reward_shaping == mode
    saturated = Seven523Env(seed=0, reward_shaping="saturate", reward_cap=0.2)
    assert saturated.reward_shaping == "saturate"
    assert saturated.reward_cap == 0.2
    for bad in ("", "none", "TERMINAL", "trickdiff"):
        with pytest.raises(ValueError, match="reward_shaping"):
            Seven523Env(seed=0, reward_shaping=bad)


def test_terminal_mode_rewards_are_zero_until_the_terminal_return():
    rules = Rules()
    env = Seven523Env(
        rules=rules,
        seed=0,
        opponents=_random_opponents(rules, 0),
        reward_shaping="terminal",
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


# -- terminal_win / win_jump (Tier 1 boundary jump) ---------------------------


def _terminal_state(rules, scores):
    """A minimal finished GameState carrying ``scores`` (points conserved)."""
    assert sum(scores) == rules.total_points
    return GameState(
        hands=tuple(frozenset() for _ in range(rules.num_players)),
        draw_pile=(),
        scores=tuple(scores),
        trick_points=0,
        trick_cards=(),
        revealed=(),
        current=0,
        incumbent=None,
        last_player=None,
        collected=rules.total_points,
        phase=Phase.DONE,
    )


@pytest.mark.parametrize(
    ("scores", "outcome"),
    [((60, 40), 1), ((50, 50), 0), ((40, 60), -1)],
)
def test_terminal_win_is_the_margin_plus_lambda_times_the_outcome(scores, outcome):
    rules = Rules()
    env = Seven523Env(
        rules=rules, seed=0, reward_shaping="terminal_win", win_jump=0.25
    )
    state = _terminal_state(rules, scores)
    margin = env.game.returns(state)[env.learner]
    assert seat_outcome(state.scores, env.learner) == outcome
    assert env._reward(state, 0.0, 0.0) == pytest.approx(
        margin + 0.25 * outcome, abs=1e-12
    )


def test_terminal_win_uses_seat_outcome_not_a_50_point_threshold():
    """3 players: own<50 can win, and a nonzero-margin tie stays a tie."""
    rules = Rules(num_players=3)
    env = Seven523Env(
        rules=rules, seed=0, reward_shaping="terminal_win", win_jump=1.0
    )
    low_win = _terminal_state(rules, (40, 30, 30))
    high_tie = _terminal_state(rules, (35, 35, 30))
    assert seat_outcome(low_win.scores, env.learner) == 1
    assert env._reward(low_win, 0.0, 0.0) == pytest.approx(0.1 + 1.0, abs=1e-12)
    assert seat_outcome(high_tie.scores, env.learner) == 0
    assert env._reward(high_tie, 0.0, 0.0) == pytest.approx(0.025, abs=1e-12)


def test_terminal_win_rewards_are_zero_until_the_terminal_step():
    rules = Rules()
    env = Seven523Env(
        rules=rules,
        seed=8,
        opponents=_random_opponents(rules, 8),
        reward_shaping="terminal_win",
        win_jump=1.0,
    )
    rewards = _play_episode(env, action_seed=0)
    assert len(rewards) > 1
    assert all(reward == 0.0 for reward in rewards[:-1])
    margin = env.game.returns(env.state)[env.learner]
    assert rewards[-1] == pytest.approx(
        margin + _win_bonus(env.state, env.learner), abs=1e-12
    )


@pytest.mark.parametrize("win_jump", [0.0, 0.25, 1.0])
def test_terminal_win_equals_terminal_plus_lambda_times_win(win_jump):
    """Step-by-step identity: r_jump = r_terminal + λ·r_win for every step."""
    rules = Rules()
    terminal = Seven523Env(
        rules=rules,
        seed=6,
        opponents=_random_opponents(rules, 6),
        reward_shaping="terminal",
    )
    win = Seven523Env(
        rules=rules,
        seed=6,
        opponents=_random_opponents(rules, 6),
        reward_shaping="win",
    )
    jumped = Seven523Env(
        rules=rules,
        seed=6,
        opponents=_random_opponents(rules, 6),
        reward_shaping="terminal_win",
        win_jump=win_jump,
    )
    for episode in range(3):
        term_rewards = _play_episode(terminal, action_seed=episode)
        win_rewards = _play_episode(win, action_seed=episode)
        jump_rewards = _play_episode(jumped, action_seed=episode)
        assert len(jump_rewards) == len(term_rewards) == len(win_rewards)
        if win_jump == 0.0:
            # λ=0 must be bit-for-bit terminal, not merely close.
            assert jump_rewards == term_rewards
        expected = [t + win_jump * w for t, w in zip(term_rewards, win_rewards)]
        assert jump_rewards == pytest.approx(expected, abs=1e-12)


def test_terminal_win_with_a_large_jump_orders_by_outcome_like_win():
    rules = Rules()
    loss = _terminal_state(rules, (0, 100))
    tie = _terminal_state(rules, (50, 50))
    win = _terminal_state(rules, (100, 0))
    for win_jump in (2.0, 10.0):
        env = Seven523Env(
            rules=rules,
            seed=0,
            reward_shaping="terminal_win",
            win_jump=win_jump,
        )
        rewards = [env._reward(state, 0.0, 0.0) for state in (loss, tie, win)]
        assert rewards[0] < rewards[1] < rewards[2]


def test_win_jump_validation():
    assert Seven523Env(seed=0).win_jump == 1.0
    assert Seven523Env(seed=0, win_jump=0.0).win_jump == 0.0
    for bad in (float("nan"), float("inf"), float("-inf"), -0.25, 10.5):
        with pytest.raises(ValueError, match="win_jump"):
            Seven523Env(seed=0, reward_shaping="terminal_win", win_jump=bad)
    # The value is validated even when the mode ignores it, so a typo cannot
    # silently reach a future terminal_win run.
    with pytest.raises(ValueError, match="win_jump"):
        Seven523Env(seed=0, win_jump=float("nan"))


# -- arcsin margin shaping + 50-point boundary jump ---------------------------


@pytest.mark.parametrize(
    ("scores", "outcome"),
    [((60, 40), 1), ((50, 50), 0), ((40, 60), -1)],
)
def test_arcsin_is_the_shaped_margin_plus_lambda_times_the_outcome(scores, outcome):
    rules = Rules()
    env = Seven523Env(
        rules=rules, seed=0, reward_shaping="arcsin", win_jump=0.25, arcsin_mix=1.0
    )
    state = _terminal_state(rules, scores)
    margin = env.game.returns(state)[env.learner]
    shaped = (2.0 / math.pi) * math.asin(margin)
    assert seat_outcome(state.scores, env.learner) == outcome
    assert env._reward(state, 0.0, 0.0) == pytest.approx(
        shaped + 0.25 * outcome, abs=1e-12
    )


def test_arcsin_amplifies_extreme_margins_and_is_flat_at_the_50_point_boundary():
    rules = Rules()
    env = Seven523Env(
        rules=rules, seed=0, reward_shaping="arcsin", win_jump=0.0, arcsin_mix=1.0
    )

    def shaped(scores):
        return env._reward(_terminal_state(rules, scores), 0.0, 0.0)

    assert shaped((0, 100)) == pytest.approx(-1.0, abs=1e-12)
    assert shaped((100, 0)) == pytest.approx(1.0, abs=1e-12)
    assert shaped((50, 50)) == pytest.approx(0.0, abs=1e-12)
    # Near 0/100 the transform is much steeper than at the 50-point boundary:
    # the 0 -> 1 point step is ~0.128 while 49 -> 50 is ~0.013.
    assert abs(shaped((1, 99)) - shaped((0, 100))) > 0.1
    assert abs(shaped((50, 50)) - shaped((49, 51))) < 0.02
    values = [shaped((own, 100 - own)) for own in range(101)]
    assert all(b >= a for a, b in zip(values, values[1:]))


def test_arcsin_mix_blends_the_curve_and_zero_reduces_to_terminal_win():
    rules = Rules()
    scores = (40, 60)
    margin = (40 - 60) / rules.total_points
    outcome = seat_outcome(scores, 0)

    def reward(mix):
        env = Seven523Env(
            rules=rules,
            seed=0,
            reward_shaping="arcsin",
            win_jump=0.25,
            arcsin_mix=mix,
        )
        return env._reward(_terminal_state(rules, scores), 0.0, 0.0)

    assert reward(0.0) == pytest.approx(margin + 0.25 * outcome, abs=1e-12)
    assert reward(0.5) == pytest.approx(
        0.5 * margin + 0.5 * (2.0 / math.pi) * math.asin(margin) + 0.25 * outcome,
        abs=1e-12,
    )
    assert reward(1.0) == pytest.approx(
        (2.0 / math.pi) * math.asin(margin) + 0.25 * outcome, abs=1e-12
    )
    # α=0 is exactly terminal_win (same linear margin, same boundary jump).
    jump = Seven523Env(
        rules=rules, seed=0, reward_shaping="terminal_win", win_jump=0.25
    )
    state = _terminal_state(rules, scores)
    assert reward(0.0) == pytest.approx(jump._reward(state, 0.0, 0.0), abs=1e-12)
    # The default is the half-strength curve.
    assert Seven523Env(seed=0, reward_shaping="arcsin").arcsin_mix == 0.5
    for bad in (-0.25, 1.25, float("nan")):
        with pytest.raises(ValueError, match="arcsin_mix"):
            Seven523Env(seed=0, reward_shaping="arcsin", arcsin_mix=bad)
    with pytest.raises(ValueError, match="arcsin_mix"):
        Seven523Env(seed=0, arcsin_mix=1.5)


def test_arcsin_jump_at_50_keeps_the_tie_neutral_and_separates_win_loss():
    rules = Rules()
    env = Seven523Env(
        rules=rules, seed=0, reward_shaping="arcsin", win_jump=0.5, arcsin_mix=1.0
    )

    def r(scores):
        return env._reward(_terminal_state(rules, scores), 0.0, 0.0)

    assert r((50, 50)) == pytest.approx(0.0, abs=1e-12)  # a draw stays neutral
    assert r((51, 49)) == pytest.approx(
        0.5 + (2.0 / math.pi) * math.asin(0.02), abs=1e-12
    )
    assert r((49, 51)) == pytest.approx(
        -0.5 + (2.0 / math.pi) * math.asin(-0.02), abs=1e-12
    )
    # The jump separates the limits just below/above 50 points.
    assert r((49, 51)) < r((50, 50)) < r((51, 49))


def test_arcsin_rewards_are_zero_until_the_terminal_step():
    rules = Rules()
    env = Seven523Env(
        rules=rules,
        seed=8,
        opponents=_random_opponents(rules, 8),
        reward_shaping="arcsin",
        win_jump=1.0,
        arcsin_mix=1.0,
    )
    rewards = _play_episode(env, action_seed=0)
    assert len(rewards) > 1
    assert all(reward == 0.0 for reward in rewards[:-1])
    margin = env.game.returns(env.state)[env.learner]
    expected = (2.0 / math.pi) * math.asin(margin) + env._win_bonus(env.state)
    assert rewards[-1] == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize(
    "mode", ["terminal", "trick_diff", "win", "trick_diff_win"]
)
def test_win_jump_is_ignored_by_legacy_modes(mode):
    rules = Rules()
    first = Seven523Env(
        rules=rules,
        seed=7,
        opponents=_random_opponents(rules, 7),
        reward_shaping=mode,
    )
    second = Seven523Env(
        rules=rules,
        seed=7,
        opponents=_random_opponents(rules, 7),
        reward_shaping=mode,
        win_jump=5.0,
    )
    for episode in range(2):
        assert _play_episode(first, action_seed=episode) == _play_episode(
            second, action_seed=episode
        )


# -- reward decomposition (`reward_jump_scale` / info["reward_jump"]) ---------


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("terminal", 0.0),
        ("trick_diff", 0.0),
        ("win", 1.0),
        ("trick_diff_win", 1.0),
        ("terminal_win", 0.25),
        ("arcsin", 0.25),
        ("saturate", 0.0),
    ],
)
def test_reward_jump_scale_matches_the_mode(mode, expected):
    kwargs = {"reward_cap": 0.2} if mode == "saturate" else {}
    env = Seven523Env(
        seed=0, reward_shaping=mode, win_jump=0.25, **kwargs
    )
    assert env.reward_jump_scale == pytest.approx(expected)


def _play_episode_with_info(env, action_seed=0):
    """Like :func:`_play_episode` but also return the terminal info dict."""
    env.reset()
    rng = random.Random(action_seed)
    rewards: list[float] = []
    done = False
    info: dict = {}
    while not done:
        legal = [i for i in range(env.action_space_n) if env.action_mask[i]]
        assert legal
        _obs, reward, done, truncated, info = env.step(
            (rng.choice(legal), rng.randrange(SUIT_N))
        )
        rewards.append(reward)
        assert not truncated
    return rewards, info


def test_reward_jump_separates_the_terminal_win_terms():
    rules = Rules()
    # arcsin: terminal reward = shaped margin + jump.
    env = Seven523Env(
        rules=rules,
        seed=3,
        opponents=_random_opponents(rules, 3),
        reward_shaping="arcsin",
        win_jump=0.25,
        arcsin_mix=0.5,
    )
    rewards, info = _play_episode_with_info(env, action_seed=0)
    margin = env.game.returns(env.state)[env.learner]
    shaped = 0.5 * margin + 0.5 * (2.0 / math.pi) * math.asin(margin)
    assert info["reward_jump"] == pytest.approx(0.25 * info["outcome"])
    assert rewards[-1] - info["reward_jump"] == pytest.approx(shaped, abs=1e-12)
    # win: the whole terminal reward is the outcome jump.
    win_env = Seven523Env(
        rules=rules,
        seed=3,
        opponents=_random_opponents(rules, 3),
        reward_shaping="win",
    )
    win_rewards, win_info = _play_episode_with_info(win_env, action_seed=0)
    assert win_rewards[-1] == pytest.approx(win_info["reward_jump"], abs=1e-12)
    assert win_info["reward_jump"] == pytest.approx(win_info["outcome"], abs=1e-12)


def test_modes_without_a_jump_publish_a_zero_reward_jump():
    rules = Rules()
    env = Seven523Env(
        rules=rules,
        seed=3,
        opponents=_random_opponents(rules, 3),
        reward_shaping="terminal",
    )
    _rewards, info = _play_episode_with_info(env, action_seed=0)
    assert info["reward_jump"] == 0.0


# -- saturate / reward_cap (Tier 1 margin clamp) -------------------------------


@pytest.mark.parametrize(
    ("scores", "cap", "expected"),
    [
        ((60, 40), 0.2, 0.2),  # win above the cap -> cap
        ((70, 30), 0.7, 0.4),  # win below the cap -> margin
        ((55, 45), 0.2, 0.1),  # win below the cap -> margin
        ((50, 50), 0.2, 0.0),  # tie -> zero
        ((40, 60), 0.2, -0.2),  # loss keeps the margin
        ((0, 100), 0.0, -1.0),  # K0: losses unchanged, wins zero
    ],
)
def test_saturate_clamps_wins_and_keeps_losses(scores, cap, expected):
    rules = Rules()
    env = Seven523Env(
        rules=rules, seed=0, reward_shaping="saturate", reward_cap=cap
    )
    state = _terminal_state(rules, scores)
    assert env._reward(state, 0.0, 0.0) == pytest.approx(expected, abs=1e-12)


def test_saturate_zero_cap_makes_wins_score_like_ties():
    rules = Rules()
    env = Seven523Env(
        rules=rules, seed=0, reward_shaping="saturate", reward_cap=0.0
    )
    win = _terminal_state(rules, (100, 0))
    tie = _terminal_state(rules, (50, 50))
    loss = _terminal_state(rules, (0, 100))
    assert env._reward(win, 0.0, 0.0) == 0.0
    assert env._reward(tie, 0.0, 0.0) == 0.0
    assert env._reward(loss, 0.0, 0.0) == -1.0


def test_saturate_uses_seat_outcome_not_a_50_point_threshold():
    """3 players: own<50 can win; a nonzero-margin tie stays a tie."""
    rules = Rules(num_players=3)
    env = Seven523Env(
        rules=rules, seed=0, reward_shaping="saturate", reward_cap=0.7
    )
    low_win = _terminal_state(rules, (40, 30, 30))
    high_tie = _terminal_state(rules, (35, 35, 30))
    assert seat_outcome(low_win.scores, env.learner) == 1
    assert env._reward(low_win, 0.0, 0.0) == pytest.approx(0.1, abs=1e-12)
    assert seat_outcome(high_tie.scores, env.learner) == 0
    assert env._reward(high_tie, 0.0, 0.0) == 0.0
    # 4 players: the branch is chosen by the outcome, so a loss keeps its
    # (possibly positive) margin exactly like the specification says.
    rules4 = Rules(num_players=4)
    env4 = Seven523Env(
        rules=rules4, seed=0, reward_shaping="saturate", reward_cap=0.1
    )
    positive_loss = _terminal_state(rules4, (26, 27, 22, 25))
    assert seat_outcome(positive_loss.scores, env4.learner) == -1
    margin = env4.game.returns(positive_loss)[env4.learner]
    assert margin > 0.0
    assert env4._reward(positive_loss, 0.0, 0.0) == pytest.approx(margin, abs=1e-12)


def test_saturate_rewards_are_zero_until_the_terminal_step():
    rules = Rules()
    env = Seven523Env(
        rules=rules,
        seed=8,
        opponents=_random_opponents(rules, 8),
        reward_shaping="saturate",
        reward_cap=0.2,
    )
    rewards = _play_episode(env, action_seed=0)
    assert len(rewards) > 1
    assert all(reward == 0.0 for reward in rewards[:-1])
    margin = env.game.returns(env.state)[env.learner]
    outcome = seat_outcome(env.state.scores, env.learner)
    expected = margin if outcome < 0 else min(margin, 0.2)
    assert rewards[-1] == pytest.approx(expected, abs=1e-12)


def test_saturate_episode_matches_the_formula_and_differs_from_terminal():
    """Same seed / same actions: saturate is not a renamed terminal arm."""
    rules = Rules()
    terminal = Seven523Env(
        rules=rules,
        seed=9,
        opponents=_random_opponents(rules, 9),
        reward_shaping="terminal",
    )
    k2 = Seven523Env(
        rules=rules,
        seed=9,
        opponents=_random_opponents(rules, 9),
        reward_shaping="saturate",
        reward_cap=0.2,
    )
    k0 = Seven523Env(
        rules=rules,
        seed=9,
        opponents=_random_opponents(rules, 9),
        reward_shaping="saturate",
        reward_cap=0.0,
    )
    for episode in range(3):
        term_rewards = _play_episode(terminal, action_seed=episode)
        k2_rewards = _play_episode(k2, action_seed=episode)
        k0_rewards = _play_episode(k0, action_seed=episode)
        assert len(k2_rewards) == len(k0_rewards) == len(term_rewards)
        assert k2_rewards[:-1] == [0.0] * (len(k2_rewards) - 1)
        assert k0_rewards[:-1] == [0.0] * (len(k0_rewards) - 1)
        margin = terminal.game.returns(terminal.state)[terminal.learner]
        outcome = seat_outcome(terminal.state.scores, terminal.learner)
        expected_k2 = margin if outcome < 0 else min(margin, 0.2)
        expected_k0 = margin if outcome < 0 else 0.0
        assert k2_rewards[-1] == pytest.approx(expected_k2, abs=1e-12)
        assert k0_rewards[-1] == pytest.approx(expected_k0, abs=1e-12)
        if outcome > 0 and margin > 0.2:
            assert k2_rewards[-1] != term_rewards[-1]
            assert k0_rewards[-1] != term_rewards[-1]
    # Deterministic arm guard on constructed states.
    win = _terminal_state(rules, (70, 30))
    assert terminal._reward(win, 0.0, 0.0) == pytest.approx(0.4, abs=1e-12)
    assert k2._reward(win, 0.0, 0.0) == pytest.approx(0.2, abs=1e-12)
    assert k0._reward(win, 0.0, 0.0) == 0.0


def test_saturate_ignores_win_jump():
    rules = Rules()
    first = Seven523Env(
        rules=rules,
        seed=7,
        opponents=_random_opponents(rules, 7),
        reward_shaping="saturate",
        reward_cap=0.2,
    )
    second = Seven523Env(
        rules=rules,
        seed=7,
        opponents=_random_opponents(rules, 7),
        reward_shaping="saturate",
        reward_cap=0.2,
        win_jump=5.0,
    )
    for episode in range(2):
        assert _play_episode(first, action_seed=episode) == _play_episode(
            second, action_seed=episode
        )
    # win_jump is still validated even though saturate ignores its value.
    with pytest.raises(ValueError, match="win_jump"):
        Seven523Env(
            seed=0,
            reward_shaping="saturate",
            reward_cap=0.2,
            win_jump=float("nan"),
        )


def test_saturate_requires_an_explicit_reward_cap():
    with pytest.raises(ValueError, match="reward_cap"):
        Seven523Env(seed=0, reward_shaping="saturate")
    with pytest.raises(ValueError, match="reward_cap"):
        Seven523Env(seed=0, reward_shaping="saturate", reward_cap=None)


@pytest.mark.parametrize(
    "mode", ["terminal", "trick_diff", "win", "trick_diff_win", "terminal_win"]
)
def test_reward_cap_is_rejected_by_every_other_mode(mode):
    with pytest.raises(ValueError, match="reward_cap"):
        Seven523Env(seed=0, reward_shaping=mode, reward_cap=0.2)
    with pytest.raises(ValueError, match="reward_cap"):
        Seven523Env(seed=0, reward_cap=0.2)  # default mode is terminal


@pytest.mark.parametrize(
    "cap", [-0.1, 1.1, float("nan"), float("inf"), float("-inf")]
)
def test_reward_cap_range_validation(cap):
    with pytest.raises(ValueError, match="reward_cap"):
        Seven523Env(seed=0, reward_shaping="saturate", reward_cap=cap)


@pytest.mark.parametrize("cap", [0.0, 0.2, 0.7, 1.0])
def test_reward_cap_accepts_the_closed_unit_interval(cap):
    env = Seven523Env(seed=0, reward_shaping="saturate", reward_cap=cap)
    assert env.reward_cap == cap


def test_env_episode_is_finite_and_legal():
    env = Seven523Env(
        opponents=[RandomBot(random.Random(0)), FirstLegalBot()],
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

    assert -2.0 <= reward <= 2.0  # ADR-0016 default arcsin + λ=1 range

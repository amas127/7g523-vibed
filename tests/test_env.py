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
    point_value,
)
from seven523.combos import Combo, ComboKind
from seven523.env import (
    OBS_VERSION,
    Seven523Env,
    encode_observation,
    observation_dim,
    observation_num_players,
    segment_spans,
)
from seven523.game import Game, GameState, Phase, View
from seven523.policies import GreedyBot, MixturePolicy, RandomBot
from seven523.rules import DEFAULT_RULES, Rules

#: Golden v1 encodings + legacy ``base680k`` actions generated from the T1
#: snapshot package (``/tmp/t7_disc_pkg``) while that layout was live.
_LEGACY_FIXTURE = Path(__file__).parent / "data" / "legacy_v1_views.json"

#: Pilot layout-b encoder; gitignored, so the bit-for-bit comparison against it
#: is skipped when the file is absent.  It produced ``_OBS_V5_GOLDEN``.
_PILOT_ENCODER = (
    Path(__file__).resolve().parents[1] / "runs" / "obs-slim" / "obs_slim_encoder.py"
)

#: v5 snapshots: the pilot's layout b prefix plus independently computed B0/B1.
_OBS_V5_GOLDEN = Path(__file__).parent / "data" / "obs_v5_golden.json"


def _load_legacy_fixture():
    return json.loads(_LEGACY_FIXTURE.read_text())


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


def _span_map(obs_version, num_players):
    """``name -> (start, width)`` for one layout."""
    return {
        name: (start, width)
        for name, start, width in segment_spans(obs_version, num_players)
    }


def test_observation_dim_by_version():
    assert observation_dim(2, obs_version=1) == 191
    assert observation_dim(2, obs_version=2) == 194
    assert observation_dim(2, obs_version=3) == 249
    assert observation_dim(2, obs_version=4) == 106
    assert observation_dim(2, obs_version=5) == 161
    # v1 with three players collides with v2 with two players: the version is
    # part of the checkpoint identity, never inferred from the dimension.
    assert observation_dim(3, obs_version=1) == 194
    assert observation_dim(3, obs_version=2) == 197
    assert observation_dim(3, obs_version=3) == 252
    assert observation_dim(3, obs_version=4) == 127
    assert observation_dim(3, obs_version=5) == 182
    assert observation_dim(2) == observation_dim(2, OBS_VERSION) == 161
    with pytest.raises(ValueError, match="obs_version"):
        observation_dim(2, obs_version=6)
    with pytest.raises(ValueError, match="obs_version"):
        observation_dim(2, obs_version=0)


def test_observation_num_players_tells_the_colliding_layouts_apart():
    assert observation_num_players(191, obs_version=1) == 2
    assert observation_num_players(194, obs_version=1) == 3
    assert observation_num_players(194, obs_version=2) == 2
    assert observation_num_players(197, obs_version=2) == 3
    assert observation_num_players(249, obs_version=3) == 2
    assert observation_num_players(252, obs_version=3) == 3
    assert observation_num_players(251, obs_version=3) is None
    assert observation_num_players(192, obs_version=1) is None
    assert observation_num_players(106, obs_version=4) == 2
    assert observation_num_players(127, obs_version=4) == 3
    assert observation_num_players(105, obs_version=4) is None
    assert observation_num_players(161, obs_version=5) == 2
    assert observation_num_players(182, obs_version=5) == 3
    assert observation_num_players(160, obs_version=5) is None
    with pytest.raises(ValueError, match="obs_version"):
        observation_num_players(191, obs_version=9)


@pytest.mark.parametrize("num_players", [2, 3, 4, 5, 6, 7])
def test_observation_dim_formula(num_players):
    assert observation_dim(num_players, obs_version=1) == 185 + 3 * num_players
    assert observation_dim(num_players, obs_version=2) == 188 + 3 * num_players
    assert observation_dim(num_players, obs_version=3) == 243 + 3 * num_players
    assert observation_dim(num_players, obs_version=4) == 64 + 21 * num_players
    assert observation_dim(num_players, obs_version=5) == 119 + 21 * num_players


@pytest.mark.parametrize("num_players", [2, 3, 4, 5, 6, 7])
@pytest.mark.parametrize("obs_version", [1, 2, 3, 4, 5])
def test_encode_observation_length_matches_dim(num_players, obs_version):
    rules = Rules(num_players=num_players)
    game = Game(rules)
    state = game.new(random.Random(0))
    view = game.view(state, 0)
    obs = encode_observation(view, rules, obs_version)
    assert len(obs) == observation_dim(num_players, obs_version)


@pytest.mark.parametrize("num_players", [2, 3])
def test_v1_is_an_exact_prefix_of_v2(num_players):
    """Appending the B0 block must not move or rescale any v1 slot."""
    rules = Rules(num_players=num_players)
    game = Game(rules)
    state = game.new(random.Random(11))
    view = game.view(state, 1)
    v1 = encode_observation(view, rules, obs_version=1)
    v2 = encode_observation(view, rules, obs_version=2)
    assert len(v1) == observation_dim(num_players, obs_version=1)
    assert len(v2) == observation_dim(num_players, obs_version=2)
    assert v2[: len(v1)] == v1


@pytest.mark.parametrize("num_players", [2, 3])
def test_v1_and_v2_are_exact_prefixes_of_v3(num_players):
    """B1 appends the unseen/last_player block without moving any old slot."""
    rules = Rules(num_players=num_players)
    game = Game(rules)
    state = game.new(random.Random(11))
    view = game.view(state, 1)
    v1 = encode_observation(view, rules, obs_version=1)
    v2 = encode_observation(view, rules, obs_version=2)
    v3 = encode_observation(view, rules, obs_version=3)
    assert len(v3) == observation_dim(num_players, obs_version=3)
    assert v3[: len(v1)] == v1
    assert v3[: len(v2)] == v2


@pytest.mark.parametrize("num_players", [2, 3])
def test_v4_is_an_exact_prefix_of_v5(num_players):
    """B1 appends to the slim layout without moving the B0 block."""
    rules = Rules(num_players=num_players)
    game = Game(rules)
    state = game.new(random.Random(11))
    view = game.view(state, 1)
    v4 = encode_observation(view, rules, obs_version=4)
    v5 = encode_observation(view, rules, obs_version=5)
    assert len(v4) == observation_dim(num_players, obs_version=4)
    assert len(v5) == observation_dim(num_players, obs_version=5)
    assert v5[: len(v4)] == v4


@pytest.mark.parametrize("obs_version", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("num_players", [2, 3, 7])
def test_segment_spans_are_contiguous_and_cover_the_dimension(
    obs_version, num_players
):
    spans = segment_spans(obs_version, num_players)
    assert spans
    offset = 0
    names = []
    for name, start, width in spans:
        assert start == offset
        assert width > 0
        names.append(name)
        offset += width
    assert names == list(dict.fromkeys(names))  # ordered and unique
    assert offset == observation_dim(num_players, obs_version)


def test_segment_spans_v5_is_the_documented_layout():
    assert segment_spans(5, 2) == (
        ("hand", 0, 54),
        ("inc_rank", 54, 15),
        ("inc_suit", 69, 4),
        ("inc_kind", 73, 6),
        ("inc_size", 79, 1),
        ("draw", 80, 1),
        ("scores", 81, 2),
        ("opp_count", 83, 1),
        ("opp_revealed", 84, 19),
        ("trick_points", 103, 1),
        ("remaining_points", 104, 1),
        ("point_hold", 105, 1),
        ("unseen", 106, 54),
        ("last_player", 160, 1),
    )


def test_segment_spans_v4_is_the_v5_prefix():
    v4 = segment_spans(4, 3)
    v5 = segment_spans(5, 3)
    v4_width = sum(width for _, _, width in v4)
    assert v5[: len(v4)] == v4
    assert v5[len(v4) :] == (
        ("unseen", v4_width, 54),
        ("last_player", v4_width + 54, 1),
    )


@pytest.mark.parametrize("obs_version", [3, 5])
def test_b1_unseen_segment_is_the_public_set_complement(obs_version):
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
    obs = encode_observation(view, rules, obs_version=obs_version)
    base, width = _span_map(obs_version, 2)["unseen"]
    assert width == 54
    seen = set(hand) | set(revealed) | set(played) | set(trick)
    assert sum(obs[base : base + width]) == 54 - len(seen)
    for card in CARD_ORDER:
        assert obs[base + card_id(card)] == (0.0 if card in seen else 1.0)


@pytest.mark.parametrize("obs_version", [3, 5])
@pytest.mark.parametrize("num_players", [2, 3])
def test_b1_last_player_segment_normalises_the_incumbent_owner(
    num_players, obs_version
):
    """None (no incumbent) is 0.0; seat s is (s + 1) / num_players."""
    rules = Rules(num_players=num_players)
    base, width = _span_map(obs_version, num_players)["last_player"]
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
        obs = encode_observation(view, rules, obs_version=obs_version)
        expected = 0.0 if last_player is None else (last_player + 1) / num_players
        assert obs[base] == pytest.approx(expected)


@pytest.mark.parametrize("obs_version", [4, 5])
def test_s1_segments_reencode_every_public_field(obs_version):
    """The slim blocks are lossless decodes of the v1 public fields."""
    fixture = _load_legacy_fixture()
    assert fixture["cases"]
    for case in fixture["cases"]:
        rules = Rules(**case["rules"])
        view = _view(case["view"])
        players = rules.num_players
        obs = encode_observation(view, rules, obs_version=obs_version)
        spans = _span_map(obs_version, players)

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

        if obs_version == 5:
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
    obs = encode_observation(view, rules, obs_version=5)
    spans = _span_map(5, 2)
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


def test_v2_point_segments_on_a_crafted_view():
    rules = DEFAULT_RULES
    view = View(
        seat=0,
        hand=frozenset(
            {
                Card(Rank.R5, Suit.SPADE),
                Card(Rank.RK, Suit.HEART),
                Card(Rank.R4, Suit.CLUB),
            }
        ),
        mask=(1 << len(CATALOG)) - 1,
        incumbent=None,
        current=0,
        scores=(20, 30),
        counts=(3, 4),
        draw_count=52,
        revealed=(Card(Rank.R3, Suit.DIAMOND),),
        trick_cards=(Card(Rank.R10, Suit.DIAMOND), Card(Rank.R5, Suit.CLUB)),
        played=(),
        last_player=1,
        done=False,
    )
    obs = encode_observation(view, rules, obs_version=2)
    base = observation_dim(2, obs_version=1)
    assert len(obs) == 194
    assert obs[base] == pytest.approx(15 / 100)  # trick_points: 10 + 5
    assert obs[base + 1] == pytest.approx(35 / 100)  # 100 - 50 score - 15 trick
    assert obs[base + 2] == pytest.approx(15 / 100)  # own hand: 5 + 10
    # v1 knows nothing about the block
    assert encode_observation(view, rules, obs_version=1) == obs[:base]


def test_v2_point_segments_on_real_midgame_views():
    cases = [
        case
        for case in _load_legacy_fixture()["cases"]
        if case["view"]["trick_cards"]
        and sum(point_value(_card(c)) for c in case["view"]["trick_cards"]) > 0
    ]
    assert cases, "fixture has no trick with points"
    for case in cases:
        rules = Rules(**case["rules"])
        view = _view(case["view"])
        obs = encode_observation(view, rules, obs_version=2)
        base = observation_dim(rules.num_players, obs_version=1)
        trick_points = sum(point_value(c) for c in view.trick_cards)
        remaining = rules.total_points - sum(view.scores) - trick_points
        hold = sum(point_value(c) for c in view.hand)
        assert obs[base] == pytest.approx(trick_points / rules.total_points)
        assert obs[base + 1] == pytest.approx(remaining / rules.total_points)
        assert obs[base + 2] == pytest.approx(hold / rules.total_points)


def test_v1_encoding_matches_legacy_snapshot_fixture():
    """Every v1 slot must equal the encoding the T1 snapshot produced."""
    fixture = _load_legacy_fixture()
    assert fixture["cases"]
    for case in fixture["cases"]:
        rules = Rules(**case["rules"])
        obs = encode_observation(_view(case["view"]), rules, obs_version=1)
        assert obs == case["obs_v1"], case["label"]


def test_v5_golden_fixture_matches_the_encoder():
    """The committed v5 snapshots are pilot layout b + independent B0/B1."""
    fixture = _load_v5_golden()
    assert fixture["cases"]
    assert fixture["source"]
    for case in fixture["cases"]:
        rules = Rules(**case["rules"])
        view = _view(case["view"])
        obs = encode_observation(view, rules, obs_version=5)
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
        obs = encode_observation(view, rules, obs_version=5)
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
        obs = encode_observation(view, rules, obs_version=5)
        assert obs[: len(pilot_b)] == pilot_b, seat


@pytest.mark.parametrize("obs_version", [1, 2, 3, 4, 5])
def test_encoding_does_not_leak_hidden_cards(obs_version):
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
        view_a, rules, obs_version=obs_version
    ) == encode_observation(view_b, rules, obs_version=obs_version)


@pytest.mark.parametrize("obs_version", [1, 2, 3, 4, 5])
def test_env_publishes_the_configured_observation_version(obs_version):
    env = Seven523Env(
        seed=0,
        opponents=[GreedyBot(), GreedyBot()],
        obs_version=obs_version,
    )
    assert env.obs_version == obs_version
    assert env.obs_dim == observation_dim(2, obs_version)
    assert env.observation_space.shape == (env.obs_dim,)
    obs, _ = env.reset()
    assert np.allclose(
        obs, encode_observation(env.view(), env.rules, obs_version)
    )


def test_env_defaults_to_v5_and_rejects_unknown_versions():
    assert Seven523Env(seed=0).obs_version == OBS_VERSION == 5
    assert Seven523Env(seed=0).obs_dim == 161
    with pytest.raises(ValueError, match="obs_version"):
        Seven523Env(seed=0, obs_version=6)


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
        assert info == {} and not truncated
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
    scores = state.scores
    best_other = max(score for seat, score in enumerate(scores) if seat != learner)
    own = scores[learner]
    if own > best_other:
        return 1.0
    if own < best_other:
        return -1.0
    return 0.0


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
        assert info == {} and not truncated
        steps += 1
        assert steps < 20_000, "env episode did not terminate"

    assert -1.0 <= reward <= 1.0

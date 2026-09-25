"""Observation-version compatibility for checkpoints and the neural policy.

T2 adds the B0 observation block (v2) on top of the legacy v1 layout.  The
version is explicit checkpoint state: legacy payloads (no ``obs_version``) load
as v1, and a policy always encodes with its own agent's version -- so a v1
checkpoint keeps acting on the legacy prefix even inside a v2 environment.
"""
from __future__ import annotations

import importlib
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from seven523.cards import CARD_ORDER, RANK_INDEX  # noqa: E402
from seven523.env import (  # noqa: E402
    Seven523Env,
    encode_observation,
    observation_dim,
    segment_spans,
)
from seven523.networks import (  # noqa: E402
    Agent,
    NeuralPolicy,
    _first_layer_remap,
    load_agent,
    save_agent,
    warm_start_into,
)
from seven523.policies import GreedyBot  # noqa: E402
from seven523.rules import DEFAULT_RULES, Rules  # noqa: E402

OBS_V1 = 191
OBS_V2 = 194
OBS_V3 = 249
OBS_V4 = 106
OBS_V5 = 161
NVEC = [134, 4]
_FIXTURE = Path(__file__).parent / "data" / "legacy_v1_views.json"
_SNAPSHOT = Path("/tmp/t7_disc_pkg")


# -- fixture helpers ---------------------------------------------------------


def _legacy_fixture():
    return json.loads(_FIXTURE.read_text())


def _view(spec):
    """Rebuild a :class:`View` from a JSON fixture spec."""
    from seven523.cards import Card, Rank, Suit
    from seven523.combos import Combo, ComboKind
    from seven523.game import View

    def card(pair):
        rank, suit = pair
        return Card(Rank(rank), None if suit is None else Suit(suit))

    incumbent = spec["incumbent"]
    return View(
        seat=spec["seat"],
        hand=frozenset(card(c) for c in spec["hand"]),
        mask=spec["mask"],
        incumbent=None
        if incumbent is None
        else Combo(
            ComboKind(incumbent["kind"]),
            tuple(card(c) for c in incumbent["cards"]),
        ),
        current=spec["current"],
        scores=tuple(spec["scores"]),
        counts=tuple(spec["counts"]),
        draw_count=spec["draw_count"],
        revealed=tuple(card(c) for c in spec["revealed"]),
        trick_cards=tuple(card(c) for c in spec["trick_cards"]),
        played=tuple(card(c) for c in spec.get("played", ())),
        last_player=spec["last_player"],
        done=spec["done"],
    )


def _to_legacy_view(view, snap):
    """Translate a current-code View into the snapshot package's View class."""

    def card(c):
        suit = None if c.suit is None else snap.cards.Suit(int(c.suit))
        return snap.cards.Card(snap.cards.Rank(int(c.rank)), suit)

    def cards(seq):
        return tuple(card(c) for c in seq)

    incumbent = None
    if view.incumbent is not None:
        incumbent = snap.combos.Combo(
            snap.combos.ComboKind(view.incumbent.kind.value),
            cards(view.incumbent.cards),
        )
    return snap.game.View(
        seat=view.seat,
        hand=frozenset(card(c) for c in view.hand),
        mask=view.mask,
        incumbent=incumbent,
        current=view.current,
        scores=view.scores,
        counts=view.counts,
        draw_count=view.draw_count,
        revealed=cards(view.revealed),
        trick_cards=cards(view.trick_cards),
        last_player=view.last_player,
        done=view.done,
    )


@pytest.fixture(scope="module")
def legacy_snapshot(tmp_path_factory):
    """Import the T1 snapshot package from a *copy*, never touching the original."""
    if not (_SNAPSHOT / "seven523" / "env.py").exists():
        pytest.skip("T1 snapshot package /tmp/t7_disc_pkg is not available")
    dest = tmp_path_factory.mktemp("legacy_pkg")
    # Copy so the read-only reference is never at risk of a bytecode write.
    shutil.copytree(
        _SNAPSHOT / "seven523",
        dest / "legacy523",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    sys.path.insert(0, str(dest))
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        snap = SimpleNamespace(
            env=importlib.import_module("legacy523.env"),
            game=importlib.import_module("legacy523.game"),
            cards=importlib.import_module("legacy523.cards"),
            combos=importlib.import_module("legacy523.combos"),
            rules=importlib.import_module("legacy523.rules"),
            networks=importlib.import_module("legacy523.networks"),
        )
    finally:
        sys.path.remove(str(dest))
        sys.dont_write_bytecode = previous
    return snap


@pytest.fixture(scope="module")
def v1_checkpoint(tmp_path_factory):
    """A synthesized v1 checkpoint replacing the retired trained artefacts.

    Compatibility assertions compare two code paths on the same file, so only
    a valid v1 payload matters -- not the weights.  This keeps the tests
    independent of ``runs/``.
    """
    path = tmp_path_factory.mktemp("v1_checkpoint") / "agent.pt"
    torch.manual_seed(0)
    save_agent(path, Agent(OBS_V1, NVEC, hidden=16, obs_version=1))
    return path


# -- checkpoint version field ------------------------------------------------


def test_agent_records_obs_version_in_the_checkpoint(tmp_path):
    torch.manual_seed(0)
    agent = Agent(OBS_V2, NVEC, hidden=16, obs_version=2)
    assert agent.obs_version == 2
    path = tmp_path / "v2.pt"
    save_agent(path, agent)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["obs_version"] == 2
    loaded, _ = load_agent(path)
    assert loaded.obs_version == 2
    assert loaded.obs_dim == OBS_V2


def test_legacy_checkpoint_without_obs_version_loads_as_v1(tmp_path):
    torch.manual_seed(0)
    payload = {
        "model": Agent(OBS_V1, NVEC, hidden=16).state_dict(),
        "obs_dim": OBS_V1,
        "nvec": NVEC,
        "hidden": 16,
    }
    path = tmp_path / "legacy.pt"
    torch.save(payload, path)
    agent, _ = load_agent(path)
    assert agent.obs_version == 1
    assert agent.obs_dim == OBS_V1


def test_agent_defaults_to_v1_for_legacy_call_sites():
    # Direct constructions predating the version field keep meaning v1.
    assert Agent(OBS_V1, NVEC, hidden=8).obs_version == 1


def test_three_player_v1_and_two_player_v2_are_distinguished_by_version(tmp_path):
    agent_three = Agent(observation_dim(3, 1), NVEC, hidden=8, obs_version=1)
    agent_two = Agent(observation_dim(2, 2), NVEC, hidden=8, obs_version=2)
    assert agent_three.obs_dim == agent_two.obs_dim == 194
    path_three = tmp_path / "np3.pt"
    path_two = tmp_path / "np2.pt"
    save_agent(path_three, agent_three)
    save_agent(path_two, agent_two)
    loaded_three, _ = load_agent(path_three)
    loaded_two, _ = load_agent(path_two)
    # Same dimension, different version: only the explicit field can tell them apart.
    assert (loaded_three.obs_dim, loaded_three.obs_version) == (194, 1)
    assert (loaded_two.obs_dim, loaded_two.obs_version) == (194, 2)


# -- warm start column padding -----------------------------------------------


def test_warm_start_pads_the_first_layer_and_keeps_the_v1_function():
    torch.manual_seed(0)
    old = Agent(OBS_V1, NVEC, hidden=16, obs_version=1)
    new = Agent(OBS_V2, NVEC, hidden=16, obs_version=2)
    copied = warm_start_into(new, old)
    assert "network.0.weight[:, :191]" in copied
    assert torch.equal(new.network[0].weight[:, :OBS_V1], old.network[0].weight)
    assert torch.count_nonzero(new.network[0].weight[:, OBS_V1:]) == 0
    assert torch.equal(new.network[0].bias, old.network[0].bias)
    assert torch.equal(new.actor.weight, old.actor.weight)
    assert torch.equal(new.actor.bias, old.actor.bias)
    assert torch.equal(new.critic.weight, old.critic.weight)
    assert torch.equal(new.critic.bias, old.critic.bias)
    # The padded model is the old function on the v1 prefix, bit-for-bit.
    obs = torch.rand(5, OBS_V2)
    assert torch.equal(new.get_value(obs), old.get_value(obs[:, :OBS_V1]))
    mask = torch.ones(5, sum(NVEC), dtype=torch.bool)
    torch.manual_seed(1234)
    action, logprob, _, _ = new.get_action_and_value(obs, mask)
    torch.manual_seed(1234)
    old_action, old_logprob, _, _ = old.get_action_and_value(obs[:, :OBS_V1], mask)
    assert torch.equal(action, old_action)
    assert torch.equal(logprob, old_logprob)


def test_warm_start_pads_the_v2_checkpoint_into_v3():
    """B1 appends two segments: the v2 function must survive bit-for-bit."""
    torch.manual_seed(0)
    old = Agent(OBS_V2, NVEC, hidden=16, obs_version=2)
    new = Agent(OBS_V3, NVEC, hidden=16, obs_version=3)
    copied = warm_start_into(new, old)
    assert "network.0.weight[:, :194]" in copied
    assert torch.equal(new.network[0].weight[:, :OBS_V2], old.network[0].weight)
    assert torch.count_nonzero(new.network[0].weight[:, OBS_V2:]) == 0
    assert torch.equal(new.network[0].bias, old.network[0].bias)
    assert torch.equal(new.actor.weight, old.actor.weight)
    assert torch.equal(new.actor.bias, old.actor.bias)
    assert torch.equal(new.critic.weight, old.critic.weight)
    assert torch.equal(new.critic.bias, old.critic.bias)
    # The padded model is the old function on the v2 prefix, bit-for-bit.
    obs = torch.rand(5, OBS_V3)
    assert torch.equal(new.get_value(obs), old.get_value(obs[:, :OBS_V2]))
    mask = torch.ones(5, sum(NVEC), dtype=torch.bool)
    torch.manual_seed(1234)
    action, logprob, _, _ = new.get_action_and_value(obs, mask)
    torch.manual_seed(1234)
    old_action, old_logprob, _, _ = old.get_action_and_value(obs[:, :OBS_V2], mask)
    assert torch.equal(action, old_action)
    assert torch.equal(logprob, old_logprob)


def test_v1_checkpoint_still_pads_through_to_v3():
    torch.manual_seed(0)
    old = Agent(OBS_V1, NVEC, hidden=16, obs_version=1)
    new = Agent(OBS_V3, NVEC, hidden=16, obs_version=3)
    copied = warm_start_into(new, old)
    assert "network.0.weight[:, :191]" in copied
    assert torch.equal(new.network[0].weight[:, :OBS_V1], old.network[0].weight)
    assert torch.count_nonzero(new.network[0].weight[:, OBS_V1:]) == 0
    obs = torch.rand(5, OBS_V3)
    assert torch.equal(new.get_value(obs), old.get_value(obs[:, :OBS_V1]))


def test_warm_start_refuses_layouts_that_only_look_compatible_by_width():
    torch.manual_seed(0)
    # v1 3-player and v2 2-player are both 194 wide, but the per-player
    # segment boundaries shift, so the first layer must not be copied.
    source = Agent(194, NVEC, hidden=16, obs_version=1)
    target = Agent(194, NVEC, hidden=16, obs_version=2)
    untouched = target.network[0].weight.clone()
    copied = warm_start_into(target, source)
    assert "network.0.weight" not in copied
    assert torch.equal(target.network[0].weight, untouched)
    assert "network.2.weight" in copied  # layout-independent layers still move
    assert torch.equal(target.critic.weight, source.critic.weight)


def test_warm_start_refuses_cross_player_count_padding():
    torch.manual_seed(0)
    source = Agent(191, NVEC, hidden=16, obs_version=1)  # v1 2-player
    target = Agent(197, NVEC, hidden=16, obs_version=2)  # v2 3-player
    untouched = target.network[0].weight.clone()
    copied = warm_start_into(target, source)
    assert not any(entry.startswith("network.0.weight") for entry in copied)
    assert torch.equal(target.network[0].weight, untouched)


def test_padding_keeps_the_policy_action_identical_on_real_views():
    torch.manual_seed(0)
    old = Agent(OBS_V1, NVEC, hidden=16, obs_version=1)
    new = Agent(OBS_V2, NVEC, hidden=16, obs_version=2)
    warm_start_into(new, old)
    old_policy = NeuralPolicy(old, DEFAULT_RULES)
    new_policy = NeuralPolicy(new, DEFAULT_RULES)
    env = Seven523Env(
        seed=0, opponents=[GreedyBot(), GreedyBot()], obs_version=2
    )
    env.reset()
    for _ in range(30):
        view = env.view()
        action = old_policy.act(view)
        assert new_policy.act(view) == action
        env.step(action)
        if env.state.done:
            env.reset()


def test_padding_keeps_the_policy_action_identical_on_real_views_v2_to_v3():
    torch.manual_seed(0)
    old = Agent(OBS_V2, NVEC, hidden=16, obs_version=2)
    new = Agent(OBS_V3, NVEC, hidden=16, obs_version=3)
    warm_start_into(new, old)
    old_policy = NeuralPolicy(old, DEFAULT_RULES)
    new_policy = NeuralPolicy(new, DEFAULT_RULES)
    env = Seven523Env(
        seed=0, opponents=[GreedyBot(), GreedyBot()], obs_version=3
    )
    env.reset()
    for _ in range(30):
        view = env.view()
        action = old_policy.act(view)
        assert new_policy.act(view) == action
        env.step(action)
        if env.state.done:
            env.reset()


# -- slim (v4/v5) first-layer remap ------------------------------------------


def _spans(obs_version, num_players=2):
    return {
        name: (start, width)
        for name, start, width in segment_spans(obs_version, num_players)
    }


def _block(spans, name):
    start, width = spans[name]
    return slice(start, start + width)


def test_first_layer_remap_refuses_prefix_rollback_and_foreign_pairs():
    for src, src_dim, dst, dst_dim in (
        (1, OBS_V1, 2, OBS_V2),  # prefix chains use the plain copy
        (2, OBS_V2, 3, OBS_V3),
        (1, OBS_V1, 3, OBS_V3),
        (4, OBS_V4, 5, OBS_V5),
        (4, OBS_V4, 1, OBS_V1),  # rollbacks are never guessed at
        (5, OBS_V5, 4, OBS_V4),
        (3, OBS_V3, 4, OBS_V4),  # v3 -> slim is rejected at the load point
        (3, OBS_V3, 5, OBS_V5),
        (5, OBS_V5, 1, OBS_V1),
        (1, OBS_V1, 4, OBS_V4 + 1),  # foreign width
        (1, 999, 4, OBS_V4),
    ):
        assert _first_layer_remap(src, src_dim, dst, dst_dim) is None


def test_first_layer_remap_maps_the_v2_segments_to_v4():
    remap = _first_layer_remap(2, OBS_V2, 4, OBS_V4)
    assert remap is not None
    assert remap.shape == (OBS_V4, OBS_V2)
    src = _spans(2)
    dst = _spans(4)

    # Same-meaning segments copy column for column.
    for dst_name, src_name in (
        ("hand", "hand"),
        ("inc_kind", "incumbent_kind"),
        ("inc_size", "incumbent_size"),
        ("draw", "draw_count"),
        ("trick_points", "trick_points"),
        ("remaining_points", "remaining_points"),
        ("point_hold", "point_hold"),
    ):
        assert torch.equal(
            remap[_block(dst, dst_name), _block(src, src_name)],
            torch.eye(dst[dst_name][1]),
        )

    # incumbent_top -> inc_rank: one mean per rank; suit half stays zero.
    inc_src = _block(src, "incumbent_top")
    inc_rank = _block(dst, "inc_rank")
    for index, card in enumerate(CARD_ORDER):
        divisor = 4 if card.suit is not None else 1
        rank = RANK_INDEX[card.rank]
        assert remap[inc_rank.start + rank, inc_src.start + index] == pytest.approx(
            1.0 / divisor
        )
    assert torch.allclose(remap[inc_rank].sum(dim=1), torch.ones(15))
    assert torch.count_nonzero(remap[_block(dst, "inc_suit")]) == 0

    # scores keep absolute order (the legacy learner was seat 0);
    # hand_counts drops the self slot and shifts the opponents down.
    assert torch.equal(
        remap[_block(dst, "scores"), _block(src, "scores")], torch.eye(2)
    )
    opp_count = _block(dst, "opp_count")
    hand_counts = _block(src, "hand_counts")
    assert remap[opp_count.start, hand_counts.start] == 0.0
    assert remap[opp_count.start, hand_counts.start + 1] == 1.0

    # revealed -> opp_revealed: the opponent block shares the incumbent rank
    # mean and zeroes its suit half (the owner is not identifiable in the
    # source).
    opp = _block(dst, "opp_revealed")
    revealed = _block(src, "revealed")
    assert opp.stop - opp.start == 19
    assert torch.equal(
        remap[opp.start : opp.start + 15, revealed], remap[inc_rank, inc_src]
    )
    assert torch.count_nonzero(remap[opp.start + 15 : opp.stop]) == 0

    # Dropped legacy segments leave all-zero columns.
    for dropped in ("rank_counts", "current"):
        assert torch.count_nonzero(remap[:, _block(src, dropped)]) == 0
    assert torch.count_nonzero(remap[:, hand_counts.start]) == 0


def test_first_layer_remap_repeats_the_reveal_map_for_every_opponent():
    src_dim = observation_dim(3, 2)
    dst_dim = observation_dim(3, 4)
    remap = _first_layer_remap(2, src_dim, 4, dst_dim)
    assert remap is not None
    dst = _spans(4, num_players=3)
    opp = _block(dst, "opp_revealed")
    assert opp.stop - opp.start == 38
    assert torch.equal(
        remap[opp.start : opp.start + 19], remap[opp.start + 19 : opp.stop]
    )


def test_first_layer_remap_zero_fills_the_missing_b0_b1_segments():
    for src_version, src_dim in ((1, OBS_V1), (2, OBS_V2)):
        for dst_version, dst_dim in ((4, OBS_V4), (5, OBS_V5)):
            remap = _first_layer_remap(src_version, src_dim, dst_version, dst_dim)
            assert remap is not None
            dst = _spans(dst_version)
            for name in ("trick_points", "remaining_points", "point_hold"):
                block = remap[_block(dst, name)]
                if src_version == 1:
                    # v1 has no B0: the new columns start at zero.
                    assert torch.count_nonzero(block) == 0
                else:
                    # v2 carries B0, so it must be remapped, not zeroed.
                    assert torch.count_nonzero(block) > 0
            if dst_version == 5:
                for name in ("unseen", "last_player"):
                    assert torch.count_nonzero(remap[_block(dst, name)]) == 0


def test_warm_start_v2_into_v4_projects_the_first_layer():
    torch.manual_seed(0)
    old = Agent(OBS_V2, NVEC, hidden=16, obs_version=2)
    new = Agent(OBS_V4, NVEC, hidden=16, obs_version=4)
    remap = _first_layer_remap(2, OBS_V2, 4, OBS_V4)
    assert remap is not None
    copied = warm_start_into(new, old)
    assert any("network.0.weight@remap:v2->v4" in entry for entry in copied)
    assert torch.equal(new.network[0].weight, old.network[0].weight @ remap.T)
    assert torch.equal(new.network[0].bias, old.network[0].bias)
    assert torch.equal(new.network[2].weight, old.network[2].weight)
    assert torch.equal(new.actor.weight, old.actor.weight)
    assert torch.equal(new.actor.bias, old.actor.bias)
    assert torch.equal(new.critic.weight, old.critic.weight)
    assert torch.equal(new.critic.bias, old.critic.bias)


def test_warm_start_v1_into_v5_remaps_both_towers():
    torch.manual_seed(0)
    old = Agent(OBS_V1, NVEC, hidden=16, obs_version=1)
    new = Agent(OBS_V5, NVEC, hidden=16, obs_version=5, arch="towers")
    remap = _first_layer_remap(1, OBS_V1, 5, OBS_V5)
    assert remap is not None
    copied = warm_start_into(new, old)
    for name, tower in (
        ("actor_network", new.actor_network),
        ("critic_network", new.critic_network),
    ):
        assert any(f"{name}.0.weight@remap:v1->v5" in entry for entry in copied)
        assert torch.equal(tower[0].weight, old.network[0].weight @ remap.T)
        assert torch.equal(tower[0].bias, old.network[0].bias)
    assert torch.equal(new.actor.weight, old.actor.weight)
    assert torch.equal(new.critic.weight, old.critic.weight)


def test_warm_start_v4_into_v5_is_a_bit_for_bit_prefix_copy():
    torch.manual_seed(0)
    old = Agent(OBS_V4, NVEC, hidden=16, obs_version=4)
    new = Agent(OBS_V5, NVEC, hidden=16, obs_version=5)
    copied = warm_start_into(new, old)
    assert f"network.0.weight[:, :{OBS_V4}]" in copied
    assert torch.equal(new.network[0].weight[:, :OBS_V4], old.network[0].weight)
    assert torch.count_nonzero(new.network[0].weight[:, OBS_V4:]) == 0
    assert torch.equal(new.network[0].bias, old.network[0].bias)
    obs = torch.rand(5, OBS_V5)
    assert torch.equal(new.get_value(obs), old.get_value(obs[:, :OBS_V4]))


def test_warm_start_skips_the_first_layer_for_an_unmappable_rollback():
    torch.manual_seed(0)
    source = Agent(OBS_V5, NVEC, hidden=16, obs_version=5)
    target = Agent(OBS_V4, NVEC, hidden=16, obs_version=4)
    untouched = target.network[0].weight.clone()
    copied = warm_start_into(target, source)
    assert not any("network.0.weight" in entry for entry in copied)
    assert torch.equal(target.network[0].weight, untouched)
    assert "network.2.weight" in copied


# -- v1 checkpoints across observation versions -------------------------------


def test_v1_checkpoint_actions_match_the_snapshot_on_the_legacy_fixture(
    v1_checkpoint, legacy_snapshot
):
    """A synthesized v1 checkpoint reads the recorded views like the snapshot.

    The fixture's ``base_action`` values came from a trained checkpoint that is
    no longer shipped, so the legacy package is the oracle: the same file must
    drive the rebuilt code path to the same actions.
    """
    agent, _ = load_agent(v1_checkpoint)
    assert (agent.obs_dim, agent.obs_version) == (OBS_V1, 1)
    policy = NeuralPolicy(agent, Rules(num_players=2))

    legacy_agent, _ = legacy_snapshot.networks.load_agent(v1_checkpoint)
    legacy_policy = legacy_snapshot.networks.NeuralPolicy(
        legacy_agent, legacy_snapshot.rules.Rules(num_players=2)
    )

    checked = 0
    for case in _legacy_fixture()["cases"]:
        if case["num_players"] != 2:
            continue
        view = _view(case["view"])
        action = policy.act(view)
        assert (view.mask >> action[0]) & 1, case["label"]
        assert action == legacy_policy.act(
            _to_legacy_view(view, legacy_snapshot)
        ), case["label"]
        checked += 1
    assert checked == 12


def test_v1_checkpoint_acts_in_a_v2_env_like_the_snapshot(
    v1_checkpoint, legacy_snapshot
):
    rules = DEFAULT_RULES
    agent, _ = load_agent(v1_checkpoint)
    assert (agent.obs_dim, agent.obs_version) == (191, 1)
    policy = NeuralPolicy(agent, rules)

    legacy_agent, _ = legacy_snapshot.networks.load_agent(v1_checkpoint)
    legacy_policy = legacy_snapshot.networks.NeuralPolicy(
        legacy_agent, legacy_snapshot.rules.DEFAULT_RULES
    )

    env = Seven523Env(seed=0, opponents=[GreedyBot(), GreedyBot()], obs_version=2)
    assert env.obs_dim == 194
    env.reset()
    for _ in range(40):
        view = env.view()
        action = policy.act(view)
        assert env.action_mask[action[0]]
        assert action == legacy_policy.act(_to_legacy_view(view, legacy_snapshot))
        env.step(action)
        if env.state.done:
            env.reset()


def test_v1_checkpoint_acts_in_a_v3_env_like_the_snapshot(
    v1_checkpoint, legacy_snapshot
):
    """A v1 checkpoint keeps reading its 191-wide prefix inside a v3 env."""
    rules = DEFAULT_RULES
    agent, _ = load_agent(v1_checkpoint)
    assert (agent.obs_dim, agent.obs_version) == (191, 1)
    policy = NeuralPolicy(agent, rules)

    legacy_agent, _ = legacy_snapshot.networks.load_agent(v1_checkpoint)
    legacy_policy = legacy_snapshot.networks.NeuralPolicy(
        legacy_agent, legacy_snapshot.rules.DEFAULT_RULES
    )

    env = Seven523Env(seed=0, opponents=[GreedyBot(), GreedyBot()], obs_version=3)
    assert env.obs_dim == 249
    env.reset()
    for _ in range(40):
        view = env.view()
        action = policy.act(view)
        assert env.action_mask[action[0]]
        assert action == legacy_policy.act(_to_legacy_view(view, legacy_snapshot))
        env.step(action)
        if env.state.done:
            env.reset()


# -- v1 encoder against the live snapshot ------------------------------------


def test_v1_encoder_matches_the_live_snapshot(legacy_snapshot):
    fixture = _legacy_fixture()
    assert fixture["cases"]
    for case in fixture["cases"]:
        rules = Rules(**case["rules"])
        view = _view(case["view"])
        legacy_rules = legacy_snapshot.rules.Rules(**case["rules"])
        legacy_view = _to_legacy_view(view, legacy_snapshot)
        assert (
            encode_observation(view, rules, obs_version=1)
            == legacy_snapshot.env.encode_observation(legacy_view, legacy_rules)
        ), case["label"]


def test_fixture_views_keep_the_v1_v2_prefix_under_v3():
    for case in _legacy_fixture()["cases"]:
        # The fixture views are 3-player in places; encoding works per rules.
        view = _view(case["view"])
        rules = Rules(**case["rules"])
        obs_v1 = encode_observation(view, rules, obs_version=1)
        obs_v2 = encode_observation(view, rules, obs_version=2)
        obs_v3 = encode_observation(view, rules, obs_version=3)
        assert len(obs_v1) + 3 == len(obs_v2)
        assert len(obs_v2) + 55 == len(obs_v3)
        assert obs_v2[: len(obs_v1)] == obs_v1
        assert obs_v3[: len(obs_v2)] == obs_v2


# -- independent actor/critic towers (T6 / direction F) ----------------------


def test_arch_checkpoint_round_trips_and_legacy_defaults_to_shared(tmp_path):
    torch.manual_seed(0)
    for arch in ("shared", "towers"):
        agent = Agent(OBS_V1, NVEC, hidden=16, obs_version=1, arch=arch)
        path = tmp_path / f"{arch}.pt"
        save_agent(path, agent)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        assert payload["arch"] == arch
        loaded, _ = load_agent(path)
        assert loaded.arch == arch
        obs = torch.rand(3, OBS_V1)
        assert torch.equal(loaded.get_value(obs), agent.get_value(obs))
        assert torch.equal(loaded.policy_logits(obs), agent.policy_logits(obs))
    legacy = tmp_path / "legacy.pt"
    torch.save(
        {
            "model": Agent(OBS_V1, NVEC, hidden=16).state_dict(),
            "obs_dim": OBS_V1,
            "nvec": NVEC,
            "hidden": 16,
        },
        legacy,
    )
    assert load_agent(legacy)[0].arch == "shared"
    with pytest.raises(ValueError):
        Agent(OBS_V1, NVEC, hidden=16, arch="twotowers")


def test_arch_parameter_counts_match_the_design():
    torch.manual_seed(0)
    shared = Agent(OBS_V1, NVEC, hidden=128, obs_version=1)
    towers = Agent(OBS_V1, NVEC, hidden=128, obs_version=1, arch="towers")
    count = lambda agent: sum(p.numel() for p in agent.parameters())
    assert count(shared) == 59_019
    assert count(towers) == 100_107
    one_trunk = sum(p.numel() for p in shared.network.parameters())
    assert one_trunk == 41_088
    assert count(towers) - count(shared) == one_trunk


def test_warm_start_shared_into_towers_is_bit_identical():
    torch.manual_seed(0)
    source = Agent(OBS_V1, NVEC, hidden=16, obs_version=1)
    target = Agent(OBS_V1, NVEC, hidden=16, obs_version=1, arch="towers")
    copied = warm_start_into(target, source)
    assert len(copied) == 12
    assert "actor_network.0.weight<-network.0.weight" in copied
    for tower in (target.actor_network, target.critic_network):
        assert torch.equal(tower[0].weight, source.network[0].weight)
        assert torch.equal(tower[0].bias, source.network[0].bias)
        assert torch.equal(tower[2].weight, source.network[2].weight)
        assert torch.equal(tower[2].bias, source.network[2].bias)
    assert torch.equal(target.actor.weight, source.actor.weight)
    assert torch.equal(target.critic.weight, source.critic.weight)
    obs = torch.rand(5, OBS_V1)
    assert torch.equal(target.policy_logits(obs), source.policy_logits(obs))
    assert torch.equal(target.get_value(obs), source.get_value(obs))
    mask = torch.ones(5, sum(NVEC), dtype=torch.bool)
    torch.manual_seed(1234)
    action, logprob, _, value = target.get_action_and_value(obs, mask)
    torch.manual_seed(1234)
    old_action, old_logprob, _, old_value = source.get_action_and_value(obs, mask)
    assert torch.equal(action, old_action)
    assert torch.equal(logprob, old_logprob)
    assert torch.equal(value, old_value)


def test_warm_start_towers_into_towers_is_bit_identical():
    torch.manual_seed(0)
    source = Agent(OBS_V1, NVEC, hidden=16, obs_version=1, arch="towers")
    target = Agent(OBS_V1, NVEC, hidden=16, obs_version=1, arch="towers")
    copied = warm_start_into(target, source)
    assert len(copied) == 12
    obs = torch.rand(5, OBS_V1)
    assert torch.equal(target.policy_logits(obs), source.policy_logits(obs))
    assert torch.equal(target.get_value(obs), source.get_value(obs))


def test_warm_start_towers_into_shared_takes_the_actor_tower():
    torch.manual_seed(0)
    source = Agent(OBS_V1, NVEC, hidden=16, obs_version=1, arch="towers")
    target = Agent(OBS_V1, NVEC, hidden=16, obs_version=1)
    copied = warm_start_into(target, source)
    assert "network.0.weight<-actor_network.0.weight" in copied
    assert torch.equal(target.network[0].weight, source.actor_network[0].weight)
    assert torch.equal(target.network[0].bias, source.actor_network[0].bias)
    assert torch.equal(target.network[2].weight, source.actor_network[2].weight)
    assert torch.equal(target.actor.weight, source.actor.weight)
    obs = torch.rand(5, OBS_V1)
    # Policy survives bit-for-bit; the value head now reads the actor tower.
    assert torch.equal(target.policy_logits(obs), source.policy_logits(obs))
    assert torch.equal(
        target.get_value(obs), source.critic(source.actor_network(obs))
    )
    assert not torch.equal(target.get_value(obs), source.get_value(obs))


def test_warm_start_shared_v1_into_towers_v3_pads_both_towers():
    torch.manual_seed(0)
    old = Agent(OBS_V1, NVEC, hidden=16, obs_version=1)
    new = Agent(OBS_V3, NVEC, hidden=16, obs_version=3, arch="towers")
    copied = warm_start_into(new, old)
    assert "actor_network.0.weight[:, :191]<-network.0.weight" in copied
    assert "critic_network.0.weight[:, :191]<-network.0.weight" in copied
    for tower in (new.actor_network, new.critic_network):
        assert torch.equal(tower[0].weight[:, :OBS_V1], old.network[0].weight)
        assert torch.count_nonzero(tower[0].weight[:, OBS_V1:]) == 0
        assert torch.equal(tower[0].bias, old.network[0].bias)
        assert torch.equal(tower[2].weight, old.network[2].weight)
    obs = torch.rand(5, OBS_V3)
    assert torch.equal(new.policy_logits(obs), old.policy_logits(obs[:, :OBS_V1]))
    assert torch.equal(new.get_value(obs), old.get_value(obs[:, :OBS_V1]))


def test_warm_start_towers_v1_into_towers_v3_pads_both_towers():
    torch.manual_seed(0)
    old = Agent(OBS_V1, NVEC, hidden=16, obs_version=1, arch="towers")
    new = Agent(OBS_V3, NVEC, hidden=16, obs_version=3, arch="towers")
    copied = warm_start_into(new, old)
    assert "actor_network.0.weight[:, :191]" in copied
    assert "critic_network.0.weight[:, :191]" in copied
    for old_tower, new_tower in (
        (old.actor_network, new.actor_network),
        (old.critic_network, new.critic_network),
    ):
        assert torch.equal(new_tower[0].weight[:, :OBS_V1], old_tower[0].weight)
        assert torch.count_nonzero(new_tower[0].weight[:, OBS_V1:]) == 0
    obs = torch.rand(5, OBS_V3)
    assert torch.equal(new.policy_logits(obs), old.policy_logits(obs[:, :OBS_V1]))
    assert torch.equal(new.get_value(obs), old.get_value(obs[:, :OBS_V1]))


def test_warm_start_refuses_a_shifted_equal_width_layout_into_towers():
    torch.manual_seed(0)
    # v1 3-player and v2 2-player are both 194 wide; per-player segments shift,
    # so neither tower's first layer may be copied.
    source = Agent(194, NVEC, hidden=16, obs_version=1)
    target = Agent(194, NVEC, hidden=16, obs_version=2, arch="towers")
    actor_untouched = target.actor_network[0].weight.clone()
    critic_untouched = target.critic_network[0].weight.clone()
    copied = warm_start_into(target, source)
    assert not any(entry.startswith("actor_network.0.weight") for entry in copied)
    assert not any(entry.startswith("critic_network.0.weight") for entry in copied)
    assert torch.equal(target.actor_network[0].weight, actor_untouched)
    assert torch.equal(target.critic_network[0].weight, critic_untouched)
    assert "actor_network.2.weight<-network.2.weight" in copied
    assert torch.equal(target.actor_network[2].weight, source.network[2].weight)
    assert torch.equal(target.critic.weight, source.critic.weight)


def test_neural_policy_towers_acts_legally_and_matches_masked_argmax():
    from seven523.actions import joint_mask_bits

    torch.manual_seed(0)
    agent = Agent(OBS_V1, NVEC, hidden=16, obs_version=1, arch="towers")
    policy = NeuralPolicy(agent, DEFAULT_RULES)
    env = Seven523Env(
        seed=0, opponents=[GreedyBot(), GreedyBot()], obs_version=1
    )
    env.reset()
    decisions = 0
    for _ in range(20):
        view = env.view()
        action = policy.act(view)
        assert env.action_mask[action[0]]
        if action[1] is not None:
            mask = torch.tensor([joint_mask_bits(view.mask, NVEC)], dtype=torch.bool)
            obs = torch.tensor(
                [encode_observation(view, DEFAULT_RULES, 1)], dtype=torch.float32
            )
            logits = agent.policy_logits(obs)
            picks = [
                int(
                    torch.where(
                        head_mask,
                        head_logits,
                        torch.tensor(-1e8),
                    )
                    .argmax(dim=1)
                    .item()
                )
                for head_logits, head_mask in zip(
                    torch.split(logits, NVEC, dim=1), torch.split(mask, NVEC, dim=1)
                )
            ]
            assert action == (picks[0], picks[1])
        env.step(action)
        decisions += 1
        if env.state.done:
            env.reset()
    assert decisions == 20

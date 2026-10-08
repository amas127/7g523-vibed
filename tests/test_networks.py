"""Checkpoints and warm starts under the single v5 observation layout.

The layout is v5 everywhere; ``save_agent`` stamps it and :func:`load_agent`
rejects a missing or foreign version.  Warm starts copy same-layout weights
and bridge the shared/towers trunk topologies; a different ``obs_dim`` is a
hard error rather than a guessed column mapping.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from support import FirstLegalBot  # noqa: E402

from seven523.actions import joint_mask_bits  # noqa: E402
from seven523.env import Seven523Env, encode_observation  # noqa: E402
from seven523.history import EVENT_DIM  # noqa: E402
from seven523.networks import (  # noqa: E402
    Agent,
    EventSequenceEncoder,
    NeuralPolicy,
    PlainHead,
    ResidualHead,
    ResidualTrunk,
    WarmStartLayoutError,
    history_layout,
    load_agent,
    save_agent,
    warm_start_from,
    warm_start_into,
)
from seven523.rules import DEFAULT_RULES  # noqa: E402

OBS_V5 = 161
NVEC = [134, 4]


# -- checkpoint version field ------------------------------------------------


def test_agent_records_the_v5_obs_version_in_the_checkpoint(tmp_path):
    torch.manual_seed(0)
    agent = Agent(OBS_V5, NVEC, hidden=16)
    path = tmp_path / "v5.pt"
    save_agent(path, agent)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["obs_version"] == 5
    loaded, _ = load_agent(path)
    assert loaded.obs_dim == OBS_V5


@pytest.mark.parametrize("version", [None, 1, 2, 3, 4, 6])
def test_load_agent_rejects_a_missing_or_foreign_obs_version(tmp_path, version):
    torch.manual_seed(0)
    payload = {
        "model": Agent(OBS_V5, NVEC, hidden=16).state_dict(),
        "obs_dim": OBS_V5,
        "nvec": NVEC,
        "hidden": 16,
    }
    if version is not None:
        payload["obs_version"] = version
    path = tmp_path / "foreign.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="obs_version"):
        load_agent(path)


# -- warm start --------------------------------------------------------------


def test_warm_start_copies_a_same_layout_checkpoint():
    torch.manual_seed(0)
    source = Agent(OBS_V5, NVEC, hidden=16)
    target = Agent(OBS_V5, NVEC, hidden=16)
    copied = warm_start_into(target, source)
    assert "network.0.weight" in copied
    for key, value in target.state_dict().items():
        assert torch.equal(value, source.state_dict()[key]), key


def test_warm_start_rejects_a_different_obs_dim():
    torch.manual_seed(0)
    source = Agent(119, NVEC, hidden=16)
    target = Agent(OBS_V5, NVEC, hidden=16)
    untouched = target.network[0].weight.clone()
    with pytest.raises(ValueError, match="observation layouts must match"):
        warm_start_into(target, source)
    assert torch.equal(target.network[0].weight, untouched)


def test_warm_start_copies_the_template_head_by_prefix():
    torch.manual_seed(0)
    old = Agent(OBS_V5, [134], hidden=16)
    new = Agent(OBS_V5, NVEC, hidden=16)
    copied = warm_start_into(new, old)
    assert "actor.weight[:134]" in copied
    assert torch.equal(new.network[0].weight, old.network[0].weight)
    assert torch.equal(new.actor.weight[:134], old.actor.weight)
    assert torch.equal(new.critic.weight, old.critic.weight)


def test_warm_start_shared_into_towers_is_bit_identical():
    torch.manual_seed(0)
    source = Agent(OBS_V5, NVEC, hidden=16)
    target = Agent(OBS_V5, NVEC, hidden=16, arch="towers")
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
    obs = torch.rand(5, OBS_V5)
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
    source = Agent(OBS_V5, NVEC, hidden=16, arch="towers")
    target = Agent(OBS_V5, NVEC, hidden=16, arch="towers")
    copied = warm_start_into(target, source)
    assert len(copied) == 12
    obs = torch.rand(5, OBS_V5)
    assert torch.equal(target.policy_logits(obs), source.policy_logits(obs))
    assert torch.equal(target.get_value(obs), source.get_value(obs))


def test_warm_start_towers_into_shared_takes_the_actor_tower():
    torch.manual_seed(0)
    source = Agent(OBS_V5, NVEC, hidden=16, arch="towers")
    target = Agent(OBS_V5, NVEC, hidden=16)
    copied = warm_start_into(target, source)
    assert "network.0.weight<-actor_network.0.weight" in copied
    assert torch.equal(target.network[0].weight, source.actor_network[0].weight)
    assert torch.equal(target.network[0].bias, source.actor_network[0].bias)
    assert torch.equal(target.network[2].weight, source.actor_network[2].weight)
    assert torch.equal(target.actor.weight, source.actor.weight)
    obs = torch.rand(5, OBS_V5)
    # Policy survives bit-for-bit; the value head now reads the actor tower.
    assert torch.equal(target.policy_logits(obs), source.policy_logits(obs))
    assert torch.equal(
        target.get_value(obs), source.critic(source.actor_network(obs))
    )
    assert not torch.equal(target.get_value(obs), source.get_value(obs))


def test_warm_start_from_exact_layout_loads_everything(tmp_path):
    torch.manual_seed(0)
    source = Agent(OBS_V5, NVEC, hidden=16)
    path = tmp_path / "exact.pt"
    save_agent(path, source)
    target = Agent(OBS_V5, NVEC, hidden=16)
    result = warm_start_from(path, target)
    assert result.exact is True
    assert result.copied == []
    assert result.arch == "shared"
    assert result.nvec == NVEC
    for key, value in target.state_dict().items():
        assert torch.equal(value, source.state_dict()[key]), key


def test_warm_start_from_reports_the_partial_copy(tmp_path):
    torch.manual_seed(0)
    source = Agent(OBS_V5, [134], hidden=16)
    path = tmp_path / "old-head.pt"
    save_agent(path, source)
    target = Agent(OBS_V5, NVEC, hidden=16)
    result = warm_start_from(path, target)
    assert result.exact is False
    assert result.nvec == [134]
    assert "actor.weight[:134]" in result.copied
    assert torch.equal(target.actor.weight[:134], source.actor.weight)


def test_warm_start_from_rejects_a_different_obs_dim(tmp_path):
    torch.manual_seed(0)
    source = Agent(119, NVEC, hidden=16)
    path = tmp_path / "foreign.pt"
    save_agent(path, source)
    target = Agent(OBS_V5, NVEC, hidden=16)
    untouched = target.network[0].weight.clone()
    with pytest.raises(ValueError, match="observation layouts must match"):
        warm_start_from(path, target)
    assert torch.equal(target.network[0].weight, untouched)


# -- architecture field ------------------------------------------------------


def test_arch_checkpoint_round_trips_and_missing_arch_defaults_to_shared(tmp_path):
    torch.manual_seed(0)
    for arch in (
        "shared",
        "towers",
        "ln",
        "deep",
        "deep_ln",
        "deep_res",
        "deep_lnres",
        "head21",
        "head31",
        "head12",
        "head22",
        "head32",
        "head13",
        "head23",
        "head33",
        "head21ln",
        "head31ln",
        "head12ln",
        "head22ln",
        "head32ln",
        "head13ln",
        "head23ln",
        "head33ln",
    ):
        agent = Agent(OBS_V5, NVEC, hidden=16, arch=arch)
        path = tmp_path / f"{arch}.pt"
        save_agent(path, agent)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        assert payload["arch"] == arch
        loaded, _ = load_agent(path)
        assert loaded.arch == arch
        obs = torch.rand(3, OBS_V5)
        assert torch.equal(loaded.get_value(obs), agent.get_value(obs))
        assert torch.equal(loaded.policy_logits(obs), agent.policy_logits(obs))
    # A v5 payload without ``arch`` was shared by construction.
    no_arch = tmp_path / "no_arch.pt"
    torch.save(
        {
            "model": Agent(OBS_V5, NVEC, hidden=16).state_dict(),
            "obs_dim": OBS_V5,
            "nvec": NVEC,
            "hidden": 16,
            "obs_version": 5,
        },
        no_arch,
    )
    assert load_agent(no_arch)[0].arch == "shared"
    with pytest.raises(ValueError):
        Agent(OBS_V5, NVEC, hidden=16, arch="twotowers")


def test_arch_parameter_counts_match_the_design():
    torch.manual_seed(0)
    shared = Agent(OBS_V5, NVEC, hidden=128)
    towers = Agent(OBS_V5, NVEC, hidden=128, arch="towers")

    def count(agent):
        return sum(p.numel() for p in agent.parameters())
    assert count(shared) == 55_179
    assert count(towers) == 92_427
    one_trunk = sum(p.numel() for p in shared.network.parameters())
    assert one_trunk == 37_248
    assert count(towers) - count(shared) == one_trunk


def test_neural_policy_towers_acts_legally_and_matches_masked_argmax():
    torch.manual_seed(0)
    agent = Agent(OBS_V5, NVEC, hidden=16, arch="towers")
    policy = NeuralPolicy(agent, DEFAULT_RULES)
    env = Seven523Env(seed=0, opponents=[FirstLegalBot(), FirstLegalBot()])
    env.reset()
    decisions = 0
    for _ in range(20):
        view = env.view()
        action = policy.act(view)
        assert env.action_mask[action[0]]
        if action[1] is not None:
            mask = torch.tensor([joint_mask_bits(view.mask, NVEC)], dtype=torch.bool)
            obs = torch.tensor(
                [encode_observation(view, DEFAULT_RULES)], dtype=torch.float32
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


def test_neural_policy_value_and_outcome_readout_matches_the_agent():
    """``value_and_outcome`` is the raw head pass a live estimate panel shows."""
    torch.manual_seed(0)
    agent = Agent(OBS_V5, NVEC, hidden=16, vf_outcome=True)
    policy = NeuralPolicy(agent, DEFAULT_RULES)
    env = Seven523Env(seed=0, opponents=[FirstLegalBot(), FirstLegalBot()])
    env.reset()
    view = env.view()
    value, outcome = policy.value_and_outcome(view)
    assert isinstance(value, float) and isinstance(outcome, float)
    assert -1.0 <= outcome <= 1.0
    obs = torch.tensor(
        [encode_observation(view, DEFAULT_RULES)], dtype=torch.float32
    )
    with torch.no_grad():
        direct_value, direct_outcome = agent.get_value_and_outcome(
            obs, None, None, None, None
        )
    assert value == pytest.approx(float(direct_value.reshape(-1)[0]))
    assert outcome == pytest.approx(float(direct_outcome.reshape(-1)[0]))
    # A checkpoint without the outcome head still yields the critic margin.
    plain = NeuralPolicy(Agent(OBS_V5, NVEC, hidden=16), DEFAULT_RULES)
    margin, missing = plain.value_and_outcome(view)
    assert isinstance(margin, float)
    assert missing is None


# -- event (EVH) layout identity ---------------------------------------------


def test_event_parameter_count_and_default_path_are_pinned():
    torch.manual_seed(0)
    events = Agent(OBS_V5, NVEC, hidden=128, event_len=108)
    default = Agent(OBS_V5, NVEC, hidden=128)

    def count(agent):
        return sum(p.numel() for p in agent.parameters())
    assert count(default) == 55_179
    # Encoder 11056 + widened first trunk layer 32*128.
    assert count(events) == 55_179 + 11_056 + 32 * 128
    assert events.event_encoder is not None
    assert default.event_encoder is None
    assert not any(name.startswith("event_encoder") for name in default.state_dict())
    obs = torch.rand(4, OBS_V5)
    assert torch.equal(default.policy_logits(obs), default.policy_logits(obs, None))
    with pytest.raises(ValueError, match="no event encoder"):
        default.policy_logits(obs, None, torch.zeros(4, 4, EVENT_DIM))
    with pytest.raises(ValueError, match="no events were passed"):
        events.policy_logits(obs)


def test_event_encoder_zeroes_the_seat_segment_in_none_mode():
    torch.manual_seed(0)
    concat = Agent(OBS_V5, NVEC, hidden=16, event_len=8, seat_emb=4)
    seated = Agent(
        OBS_V5, NVEC, hidden=16, event_len=8, seat_emb=4, event_seat="none"
    )
    torch.manual_seed(0)
    events = torch.rand(2, 8, EVENT_DIM)
    seats = torch.randint(0, 3, (2, 8))
    mask = torch.ones(2, 8, dtype=torch.bool)
    # A zeroed seat segment can never change the output, so two seat patterns
    # with the same events decode identically.
    seats_b = torch.full_like(seats, 2)
    with torch.no_grad():
        out_a = seated.event_encoder(events, seats, mask)
        out_b = seated.event_encoder(events, seats_b, mask)
        out_c = concat.event_encoder(events, seats, mask)
    assert torch.allclose(out_a, out_b)
    assert out_a.shape == out_c.shape == (2, 32)
    assert seated.event_encoder.rnn.input_size == 32 + 4  # seat width preserved


def test_event_noisy_and_blind_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        Agent(OBS_V5, NVEC, hidden=16, event_len=8, event_blind=True, event_noisy=True)


def test_history_layout_marks_mlp_seq_and_event():
    torch.manual_seed(0)
    assert history_layout(Agent(OBS_V5, NVEC, hidden=16)) == "mlp"
    assert history_layout(Agent(OBS_V5, NVEC, hidden=16, seq_len=12)).startswith("seq:")
    layout = history_layout(Agent(OBS_V5, NVEC, hidden=16, event_len=8))
    assert layout.startswith("event:8:")
    both = Agent(OBS_V5, NVEC, hidden=16, seq_len=12, event_len=8)
    assert "+" in history_layout(both)


def test_warm_start_rejects_a_different_history_layout(tmp_path):
    torch.manual_seed(0)
    mlp = Agent(OBS_V5, NVEC, hidden=16)
    event = Agent(OBS_V5, NVEC, hidden=16, event_len=8)
    path = tmp_path / "event.pt"
    save_agent(path, event)
    target = Agent(OBS_V5, NVEC, hidden=16)
    with pytest.raises(WarmStartLayoutError, match="history layout"):
        warm_start_from(path, target)
    with pytest.raises(WarmStartLayoutError, match="history layout"):
        warm_start_into(target, event)
    # Same event layout round-trips exactly.
    twin = Agent(OBS_V5, NVEC, hidden=16, event_len=8)
    outcome = warm_start_from(path, twin)
    assert outcome.exact is True
    assert all(
        torch.equal(value, event.state_dict()[key])
        for key, value in twin.state_dict().items()
    )
    assert mlp.seq_encoder is None


def test_event_checkpoint_records_layout_and_event_fields(tmp_path):
    torch.manual_seed(0)
    agent = Agent(
        OBS_V5,
        NVEC,
        hidden=16,
        event_len=8,
        event_emb=4,
        event_hidden=6,
        seat_emb=4,
        event_noisy=True,
    )
    path = tmp_path / "event.pt"
    save_agent(path, agent)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["obs_version"] == 5
    assert payload["history_layout"] == history_layout(agent)
    assert payload["event_len"] == 8
    assert payload["event_noisy"] is True
    loaded, _ = load_agent(path)
    assert history_layout(loaded) == history_layout(agent)


def test_event_encoder_ignores_trailing_pads_beyond_each_rows_last_event():
    torch.manual_seed(0)
    encoder = EventSequenceEncoder()
    events = torch.rand(2, 12, EVENT_DIM)
    seats = torch.randint(0, 3, (2, 12))
    mask = torch.zeros(2, 12, dtype=torch.bool)
    mask[0, :7] = True
    mask[1, :3] = True
    clean = encoder(events, seats, mask)
    tampered = events.clone()
    tampered[0, 7:] = 999.0
    tampered[1, 3:] = -999.0
    assert torch.allclose(clean, encoder(tampered, seats, mask))
    # A batch element with no real event receives the padded all-zero input and
    # reads the initial hidden (zeros).
    empty = torch.zeros(1, 12, dtype=torch.bool)
    padded_events = torch.zeros(1, 12, EVENT_DIM)
    padded_seats = torch.full((1, 12), 2, dtype=torch.long)
    assert encoder(padded_events, padded_seats, empty).abs().max() == 0.0


# -- O2 trunk depth and LayerNorm (docs/depth-normalization-plan.md) ---------


TRUNK_SIGNATURES = {
    "shared": ["Linear", "ReLU", "Linear", "ReLU"],
    "ln": ["Linear", "LayerNorm", "ReLU", "Linear", "LayerNorm", "ReLU"],
    "deep": ["Linear", "ReLU", "Linear", "ReLU", "Linear", "ReLU"],
    "deep_ln": [
        "Linear",
        "LayerNorm",
        "ReLU",
        "Linear",
        "LayerNorm",
        "ReLU",
        "Linear",
        "LayerNorm",
        "ReLU",
    ],
}


def _param_count(agent):
    return sum(p.numel() for p in agent.parameters())


@pytest.mark.parametrize("arch", ["ln", "deep", "deep_ln"])
def test_o2_trunk_signature_and_parameter_count(arch):
    torch.manual_seed(0)
    agent = Agent(OBS_V5, NVEC, hidden=128, arch=arch)
    assert [type(module).__name__ for module in agent.network] == TRUNK_SIGNATURES[arch]
    expected = {"ln": 55_691, "deep": 71_691, "deep_ln": 72_459}
    assert _param_count(agent) == expected[arch]
    obs = torch.rand(4, OBS_V5)
    logits = agent.policy_logits(obs)
    value = agent.get_value(obs)
    assert logits.shape == (4, sum(NVEC))
    assert value.shape == (4, 1)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(value).all()


def test_default_shared_trunk_keeps_the_historical_layout():
    torch.manual_seed(0)
    agent = Agent(OBS_V5, NVEC, hidden=128)
    assert [type(module).__name__ for module in agent.network] == TRUNK_SIGNATURES["shared"]
    assert _param_count(agent) == 55_179
    assert [key for key in agent.state_dict() if key.startswith("network.")] == [
        "network.0.weight",
        "network.0.bias",
        "network.2.weight",
        "network.2.bias",
    ]


def test_wide_capacity_control_parameter_count():
    # --arch shared --hidden-size 157: the O2 capacity control (~+-1% of deep
    # and deep_ln).  Zero code change; descriptive-only arm.
    torch.manual_seed(0)
    wide = Agent(OBS_V5, NVEC, hidden=157)
    assert _param_count(wide) == 72_202
    assert abs(_param_count(wide) / 71_691 - 1) < 0.01
    assert abs(_param_count(wide) / 72_459 - 1) < 0.01


def test_residual_archs_are_parameter_matched_to_their_plain_twins():
    # O2 Amendment A1: deep_res/deep_lnres differ from deep/deep_ln only by
    # the skip connections, so any reading difference is the skip effect.
    torch.manual_seed(0)
    twins = {("deep", "deep_res"), ("deep_ln", "deep_lnres")}
    for plain_arch, residual_arch in twins:
        plain = Agent(OBS_V5, NVEC, hidden=128, arch=plain_arch)
        residual = Agent(OBS_V5, NVEC, hidden=128, arch=residual_arch)
        assert _param_count(residual) == _param_count(plain)
        expected = {"deep_res": 71_691, "deep_lnres": 72_459}
        assert _param_count(residual) == expected[residual_arch]
        assert isinstance(residual.network, ResidualTrunk)
        assert len(residual.network.blocks) == 3
        block_types = [type(m).__name__ for m in residual.network.blocks[1]]
        assert block_types == (
            ["Linear", "ReLU"] if residual_arch == "deep_res"
            else ["Linear", "LayerNorm", "ReLU"]
        )
        obs = torch.rand(4, OBS_V5)
        assert torch.isfinite(residual.policy_logits(obs)).all()
        assert torch.isfinite(residual.get_value(obs)).all()


# -- head-depth grid, residual multi-layer heads (docs/head-depth-plan.md) ---

HEAD_GRID = {
    (1, 1): ("shared", 55_179),
    (2, 1): ("head21", 71_691),
    (3, 1): ("head31", 88_203),
    (1, 2): ("head12", 71_691),
    (2, 2): ("head22", 88_203),
    (3, 2): ("head32", 104_715),
    (1, 3): ("head13", 88_203),
    (2, 3): ("head23", 104_715),
    (3, 3): ("head33", 121_227),
}

#: Head design v2 (plan section 10): pre-norm LayerNorm inside every residual
#: block, +2*hidden params per block (gamma/beta).
HEAD_GRID_LN = {
    (2, 1): ("head21ln", 71_947),
    (3, 1): ("head31ln", 88_715),
    (1, 2): ("head12ln", 71_947),
    (2, 2): ("head22ln", 88_715),
    (3, 2): ("head32ln", 105_483),
    (1, 3): ("head13ln", 88_715),
    (2, 3): ("head23ln", 105_483),
    (3, 3): ("head33ln", 122_251),
}


def _assert_head_structure(agent, actor_layers, critic_layers, *, ln):
    block_sig = ["LayerNorm", "Linear", "ReLU"] if ln else ["Linear", "ReLU"]
    if actor_layers == 1:
        assert type(agent.actor) is torch.nn.Linear
        assert agent.actor.out_features == sum(NVEC)
    else:
        assert isinstance(agent.actor, ResidualHead)
        assert len(agent.actor.blocks) == actor_layers - 1
        for block in agent.actor.blocks:
            assert [type(module).__name__ for module in block] == block_sig
        assert agent.actor.out.out_features == sum(NVEC)
    if critic_layers == 1:
        assert type(agent.critic) is torch.nn.Linear
        assert agent.critic.out_features == 1
    else:
        assert isinstance(agent.critic, ResidualHead)
        assert len(agent.critic.blocks) == critic_layers - 1
        for block in agent.critic.blocks:
            assert [type(module).__name__ for module in block] == block_sig
        assert agent.critic.out.out_features == 1


def test_head_grid_parameter_counts_and_residual_structure():
    # Multi-layer heads are residual by decision: layers - 1 skip blocks
    # h = h + ReLU(Linear(h)), then the output projection.  One-layer heads
    # have no hidden layer and stay the historical plain Linear.
    torch.manual_seed(0)
    for (actor_layers, critic_layers), (arch, expected) in HEAD_GRID.items():
        agent = Agent(OBS_V5, NVEC, hidden=128, arch=arch)
        assert _param_count(agent) == expected
        _assert_head_structure(agent, actor_layers, critic_layers, ln=False)
        obs = torch.rand(4, OBS_V5)
        assert torch.isfinite(agent.policy_logits(obs)).all()
        assert torch.isfinite(agent.get_value(obs)).all()


def test_head_grid_ln_is_pre_norm_and_parameter_counts():
    # Head design v2 (plan section 10): pre-norm LayerNorm as the block input
    # (LayerNorm -> Linear -> activation), skip still h = h + block(h).
    torch.manual_seed(0)
    for (actor_layers, critic_layers), (arch, expected) in HEAD_GRID_LN.items():
        agent = Agent(OBS_V5, NVEC, hidden=128, arch=arch)
        assert _param_count(agent) == expected
        _assert_head_structure(agent, actor_layers, critic_layers, ln=True)
        obs = torch.rand(4, OBS_V5)
        assert torch.isfinite(agent.policy_logits(obs)).all()
        assert torch.isfinite(agent.get_value(obs)).all()


def test_head_grid_ln_adds_two_norm_tensors_per_block():
    torch.manual_seed(0)
    plain = Agent(OBS_V5, NVEC, hidden=128, arch="head22")
    normed = Agent(OBS_V5, NVEC, hidden=128, arch="head22ln")
    assert _param_count(normed) - _param_count(plain) == 2 * 2 * 128
    assert len(normed.actor.blocks) == len(plain.actor.blocks) == 1
    assert len(normed.critic.blocks) == len(plain.critic.blocks) == 1


def test_actor_out_std_scales_the_actor_init_without_touching_the_critic():
    torch.manual_seed(0)
    low = Agent(OBS_V5, NVEC, hidden=128)
    torch.manual_seed(0)
    high = Agent(OBS_V5, NVEC, hidden=128, actor_out_std=0.1)
    # Orthogonal init: the gain scales the whole matrix, so the element std
    # scales with actor_out_std while the critic draws stay bit-identical.
    assert high.actor.weight.std().item() == pytest.approx(
        10.0 * low.actor.weight.std().item(), rel=0.05
    )
    assert torch.equal(high.critic.weight, low.critic.weight)
    assert torch.equal(high.network[0].weight, low.network[0].weight)
    with pytest.raises(ValueError, match="actor_out_std"):
        Agent(OBS_V5, NVEC, hidden=16, actor_out_std=0.0)
    with pytest.raises(ValueError, match="actor_out_std"):
        Agent(OBS_V5, NVEC, hidden=16, actor_out_std=float("nan"))


def test_head_grid_checkpoint_round_trip_and_warm_start_guard(tmp_path):
    torch.manual_seed(0)
    agent = Agent(OBS_V5, NVEC, hidden=16, arch="head33")
    path = tmp_path / "head33.pt"
    save_agent(path, agent)
    loaded, _ = load_agent(path)
    assert loaded.arch == "head33"
    obs = torch.rand(3, OBS_V5)
    assert torch.equal(loaded.policy_logits(obs), agent.policy_logits(obs))
    assert torch.equal(loaded.get_value(obs), agent.get_value(obs))
    # Cross-arch warm start is fail-loud: the head layouts are incompatible.
    for source_arch, target_arch in (
        ("shared", "head33"),
        ("head21", "head31"),
        ("head33", "head23"),
        ("head12", "head12ln"),
        ("head12ln", "head12"),
    ):
        source = Agent(OBS_V5, NVEC, hidden=16, arch=source_arch)
        target = Agent(OBS_V5, NVEC, hidden=16, arch=target_arch)
        with pytest.raises(WarmStartLayoutError):
            warm_start_into(target, source)
    # Same LN arch still full-loads.
    source = Agent(OBS_V5, NVEC, hidden=16, arch="head12ln")
    target = Agent(OBS_V5, NVEC, hidden=16, arch="head12ln")
    warm_start_into(target, source)
    assert torch.equal(target.critic.out.weight, source.critic.out.weight)
    assert len(target.critic.blocks[0]) == 3  # LayerNorm -> Linear -> ReLU


# -- residual body + plain heads, LN vs GroupNorm (docs/gnres-plan.md) ------

#: (actor, critic) head depths -> parameter count for the lnres/gnres families:
#: the 3-block residual body (+GroupNorm/LayerNorm) plus `(a-1)+(c-1)` plain
#: head hidden layers.  ``gnres11`` is the new GroupNorm body with the
#: historical single-layer heads (same count as ``deep_lnres``'s LN twin).
BODY_HEAD_GRID = {
    (1, 1): 72_459,
    (2, 1): 88_971,
    (3, 1): 105_483,
    (1, 2): 88_971,
    (2, 2): 105_483,
    (3, 2): 121_995,
    (1, 3): 105_483,
    (2, 3): 121_995,
    (3, 3): 138_507,
}


def _assert_plain_head(head, layers, out_dim):
    if layers == 1:
        assert type(head) is torch.nn.Linear
        assert head.out_features == out_dim
        return
    assert isinstance(head, PlainHead)
    # Non-residual by construction: no skip blocks, just Linear -> act pairs.
    assert not hasattr(head, "blocks")
    assert [type(m).__name__ for m in head.body] == ["Linear", "ReLU"] * (
        layers - 1
    )
    assert head.out.out_features == out_dim


def test_body_head_grid_counts_and_structures():
    torch.manual_seed(0)
    for family, norm in (
        ("lnres", "LayerNorm"),
        ("gnres", "GroupNorm"),
        ("bnres", "BatchNorm1d"),
    ):
        for (actor_layers, critic_layers), expected in BODY_HEAD_GRID.items():
            if family == "lnres" and (actor_layers, critic_layers) == (1, 1):
                continue  # deep_lnres already carries this cell
            arch = f"{family}{actor_layers}{critic_layers}"
            agent = Agent(OBS_V5, NVEC, hidden=128, arch=arch)
            assert _param_count(agent) == expected, arch
            assert isinstance(agent.network, ResidualTrunk)
            assert len(agent.network.blocks) == 3
            for block in agent.network.blocks:
                assert [type(m).__name__ for m in block] == [
                    "Linear",
                    norm,
                    "ReLU",
                ]
            _assert_plain_head(agent.actor, actor_layers, sum(NVEC))
            _assert_plain_head(agent.critic, critic_layers, 1)
            obs = torch.rand(4, OBS_V5)
            assert torch.isfinite(agent.policy_logits(obs)).all()
            assert torch.isfinite(agent.get_value(obs)).all()


def test_body_head_grid_norm_parameter_parity():
    torch.manual_seed(0)
    # The LN body already exists as deep_lnres; the (1,1) GroupNorm body must
    # cost exactly the same (2*hidden params per norm, three blocks).
    deep = Agent(OBS_V5, NVEC, hidden=128, arch="deep_lnres")
    gn11 = Agent(OBS_V5, NVEC, hidden=128, arch="gnres11")
    assert _param_count(deep) == _param_count(gn11) == 72_459
    for (actor_layers, critic_layers), _ in BODY_HEAD_GRID.items():
        if (actor_layers, critic_layers) == (1, 1):
            continue
        counts = {
            family: _param_count(
                Agent(
                    OBS_V5,
                    NVEC,
                    hidden=128,
                    arch=f"{family}{actor_layers}{critic_layers}",
                )
            )
            for family in ("lnres", "gnres", "bnres")
        }
        assert len(set(counts.values())) == 1, counts


def test_gnres_group_count_validation_and_checkpoint_round_trip(tmp_path):
    torch.manual_seed(0)
    agent = Agent(OBS_V5, NVEC, hidden=16, arch="gnres21", gn_groups=4)
    assert agent.network.blocks[1][1].num_groups == 4
    with pytest.raises(ValueError, match="must divide"):
        Agent(OBS_V5, NVEC, hidden=16, arch="gnres21", gn_groups=3)
    with pytest.raises(ValueError, match="gn_groups"):
        Agent(OBS_V5, NVEC, hidden=16, arch="gnres21", gn_groups=0)
    path = tmp_path / "gnres.pt"
    save_agent(path, agent)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["gn_groups"] == 4
    loaded, _ = load_agent(path)
    assert loaded.gn_groups == 4 and loaded.arch == "gnres21"
    obs = torch.rand(3, OBS_V5)
    assert torch.equal(loaded.policy_logits(obs), agent.policy_logits(obs))
    assert torch.equal(loaded.get_value(obs), agent.get_value(obs))


#: (actor, critic) head depths -> parameter count for the ``res4`` family:
#: a 4-block no-norm residual body (161->128 plus three 128->128 residual
#: blocks) plus the same plain heads as the lnres/gnres families.
RES4_GRID = {
    (1, 1): 88_203,
    (2, 1): 104_715,
    (3, 1): 121_227,
    (1, 2): 104_715,
    (2, 2): 121_227,
    (3, 2): 137_739,
    (1, 3): 121_227,
    (2, 3): 137_739,
    (3, 3): 154_251,
}


def test_res4_body_depth_and_plain_heads():
    torch.manual_seed(0)
    for (actor_layers, critic_layers), expected in RES4_GRID.items():
        arch = f"res4{actor_layers}{critic_layers}"
        agent = Agent(OBS_V5, NVEC, hidden=128, arch=arch)
        assert _param_count(agent) == expected, arch
        assert isinstance(agent.network, ResidualTrunk)
        assert len(agent.network.blocks) == 4
        for block in agent.network.blocks:
            # No normalization modules at all in this family.
            assert [type(m).__name__ for m in block] == ["Linear", "ReLU"]
        _assert_plain_head(agent.actor, actor_layers, sum(NVEC))
        _assert_plain_head(agent.critic, critic_layers, 1)
        obs = torch.rand(4, OBS_V5)
        assert torch.isfinite(agent.policy_logits(obs)).all()
        assert torch.isfinite(agent.get_value(obs)).all()


def test_res421_run_config_with_hidden_64_and_outcome_head():
    torch.manual_seed(0)
    agent = Agent(
        OBS_V5, NVEC, hidden=64, arch="res421", vf_outcome=True, actor_out_std=0.1
    )
    assert _param_count(agent) == 36_108
    obs = torch.rand(8, OBS_V5)
    value, outcome = agent.get_value_and_outcome(obs)
    assert value.shape == (8, 1) and outcome.shape == (8, 1)
    logits = agent.policy_logits(obs)
    assert logits.shape == (8, sum(NVEC))


def test_bres421_bottleneck_block_shape_and_outcome_head():
    torch.manual_seed(0)
    agent = Agent(
        OBS_V5, NVEC, hidden=32, arch="bres421", vf_outcome=True, actor_out_std=0.1
    )
    assert _param_count(agent) == 29_676
    trunk = agent.network
    assert isinstance(trunk, ResidualTrunk)
    assert len(trunk.blocks) == 4
    assert [type(m).__name__ for m in trunk.blocks[0]] == ["Linear", "ReLU"]
    assert (trunk.blocks[0][0].in_features, trunk.blocks[0][0].out_features) == (
        OBS_V5,
        32,
    )
    for block in trunk.blocks[1:]:
        assert [type(m).__name__ for m in block] == [
            "Linear",
            "ReLU",
            "Linear",
            "ReLU",
        ]
        assert (block[0].in_features, block[0].out_features) == (32, 96)
        assert (block[2].in_features, block[2].out_features) == (96, 32)
    _assert_plain_head(agent.actor, 2, sum(NVEC))
    _assert_plain_head(agent.critic, 1, 1)
    obs = torch.rand(8, OBS_V5)
    value, outcome = agent.get_value_and_outcome(obs)
    assert value.shape == (8, 1) and outcome.shape == (8, 1)
    assert agent.policy_logits(obs).shape == (8, sum(NVEC))


def test_bres221_depth_two_body():
    torch.manual_seed(0)
    agent = Agent(OBS_V5, NVEC, hidden=32, arch="bres221", vf_outcome=True)
    assert _param_count(agent) == 17_132
    trunk = agent.network
    assert len(trunk.blocks) == 2
    assert [type(m).__name__ for m in trunk.blocks[0]] == ["Linear", "ReLU"]
    assert [type(m).__name__ for m in trunk.blocks[1]] == [
        "Linear",
        "ReLU",
        "Linear",
        "ReLU",
    ]
    assert (trunk.blocks[1][0].in_features, trunk.blocks[1][0].out_features) == (
        32,
        96,
    )
    assert (trunk.blocks[1][2].in_features, trunk.blocks[1][2].out_features) == (
        96,
        32,
    )
    obs = torch.rand(4, OBS_V5)
    assert torch.isfinite(agent.policy_logits(obs)).all()
    assert torch.isfinite(agent.get_value(obs)).all()


def test_bres222_critic_two_layer_head():
    torch.manual_seed(0)
    agent = Agent(
        OBS_V5, NVEC, hidden=48, arch="bres222", res_expansion=2,
        vf_outcome=True, actor_out_std=0.1,
    )
    assert _param_count(agent) == 28_700
    _assert_plain_head(agent.actor, 2, sum(NVEC))
    _assert_plain_head(agent.critic, 2, 1)
    obs = torch.rand(4, OBS_V5)
    assert torch.isfinite(agent.policy_logits(obs)).all()
    value, outcome = agent.get_value_and_outcome(obs)
    assert value.shape == (4, 1) and outcome.shape == (4, 1)


def test_bres_compression_expansion_64x16x64():
    torch.manual_seed(0)
    agent = Agent(
        OBS_V5, NVEC, hidden=64, arch="bres221", res_expansion=0.25,
        vf_outcome=True,
    )
    assert _param_count(agent) == 25_756
    first = agent.network.blocks[1][0]
    second = agent.network.blocks[1][2]
    assert (first.in_features, first.out_features) == (64, 16)
    assert (second.in_features, second.out_features) == (16, 64)
    obs = torch.rand(4, OBS_V5)
    assert torch.isfinite(agent.policy_logits(obs)).all()
    # hidden * factor must land on a positive integer.
    with pytest.raises(ValueError, match="positive integer"):
        Agent(OBS_V5, NVEC, hidden=64, arch="bres221", res_expansion=0.3)
    with pytest.raises(ValueError, match="res_expansion"):
        Agent(OBS_V5, NVEC, hidden=64, arch="bres221", res_expansion=-0.25)


def test_bres_expansion_validation_and_checkpoint_round_trip(tmp_path):
    torch.manual_seed(0)
    with pytest.raises(ValueError, match="res_expansion"):
        Agent(OBS_V5, NVEC, hidden=32, arch="bres421", res_expansion=0)
    agent = Agent(OBS_V5, NVEC, hidden=16, arch="bres421", res_expansion=2)
    first = agent.network.blocks[1][0]
    assert (first.in_features, first.out_features) == (16, 32)
    path = tmp_path / "bres.pt"
    save_agent(path, agent)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["res_expansion"] == 2
    loaded, _ = load_agent(path)
    assert loaded.arch == "bres421" and loaded.res_expansion == 2
    obs = torch.rand(3, OBS_V5)
    agent.eval()
    loaded.eval()
    assert torch.equal(loaded.policy_logits(obs), agent.policy_logits(obs))
    # A different expansion is a different residual-block layout.
    other = Agent(OBS_V5, NVEC, hidden=16, arch="bres421", res_expansion=3)
    with pytest.raises(ValueError, match="bottleneck expansions"):
        warm_start_from(path, other)


def test_res_expansion_is_ignored_by_the_plain_res_body():
    torch.manual_seed(0)
    agent = Agent(OBS_V5, NVEC, hidden=16, arch="res421", res_expansion=5)
    block = agent.network.blocks[1][0]
    assert (block.in_features, block.out_features) == (16, 16)


def test_bnres_tracks_running_stats_and_round_trips(tmp_path):
    torch.manual_seed(0)
    agent = Agent(OBS_V5, NVEC, hidden=16, arch="bnres21")
    assert [type(m).__name__ for m in agent.network.blocks[1]] == [
        "Linear",
        "BatchNorm1d",
        "ReLU",
    ]
    obs = torch.rand(8, OBS_V5)
    agent.train()
    agent.get_value(obs)
    norm = agent.network.blocks[1][1]
    assert not bool((norm.running_mean == 0).all())
    assert bool((norm.running_var > 0).all())
    agent.eval()
    expected = agent.get_value(obs)
    path = tmp_path / "bnres.pt"
    save_agent(path, agent)
    loaded, _ = load_agent(path)
    loaded.eval()
    assert torch.equal(loaded.get_value(obs), expected)
    # The running statistics travel with the checkpoint.
    assert torch.equal(
        loaded.network.blocks[1][1].running_mean, norm.running_mean
    )


def test_gnres_composes_with_the_heteroscedastic_critic():
    torch.manual_seed(0)
    nll = Agent(OBS_V5, NVEC, hidden=16, arch="gnres21", vf_nll=True)
    obs = torch.rand(4, OBS_V5)
    mean, logvar = nll.get_value_dist(obs)
    assert mean.shape == (4,) and logvar.shape == (4,)
    assert torch.allclose(logvar, torch.zeros(4), atol=1e-6)
    assert torch.isfinite(nll.policy_logits(obs)).all()


def test_gnres_warm_start_rejects_a_group_count_mismatch(tmp_path):
    torch.manual_seed(0)
    source = Agent(OBS_V5, NVEC, hidden=16, arch="gnres21", gn_groups=4)
    path = tmp_path / "gn4.pt"
    save_agent(path, source)
    target = Agent(OBS_V5, NVEC, hidden=16, arch="gnres21", gn_groups=8)
    with pytest.raises(WarmStartLayoutError, match="GroupNorm"):
        warm_start_into(target, source)
    # The exact-load fast path must not bypass the guard.
    with pytest.raises(WarmStartLayoutError, match="GroupNorm"):
        warm_start_from(path, target)
    # Same group count still full-loads.
    same = Agent(OBS_V5, NVEC, hidden=16, arch="gnres21", gn_groups=4)
    assert warm_start_from(path, same).exact is True
    obs = torch.rand(3, OBS_V5)
    assert torch.equal(same.get_value(obs), source.get_value(obs))


def test_head_grid_vf_nll_keeps_the_logvar_row_zero():
    # The grid composes with the heteroscedastic critic: only the output layer
    # gains a row (+129 params), and logvar still starts at 0.
    torch.manual_seed(0)
    nll = Agent(OBS_V5, NVEC, hidden=128, arch="head33", vf_nll=True)
    assert _param_count(nll) == 121_227 + 129
    obs = torch.rand(4, OBS_V5)
    mean, logvar = nll.get_value_dist(obs)
    assert mean.shape == (4,) and logvar.shape == (4,)
    assert torch.allclose(logvar, torch.zeros(4), atol=1e-6)


def test_head_grid_shared_cell_is_the_plain_historical_layout():
    torch.manual_seed(0)
    agent = Agent(OBS_V5, NVEC, hidden=128)
    assert type(agent.actor) is torch.nn.Linear
    assert type(agent.critic) is torch.nn.Linear
    assert _param_count(agent) == 55_179


def test_vf_nll_head_shapes_and_checkpoint_round_trip(tmp_path):
    # Amendment A2: heteroscedastic value head (mean, logvar), +129 params.
    torch.manual_seed(0)
    plain = Agent(OBS_V5, NVEC, hidden=128)
    nll = Agent(OBS_V5, NVEC, hidden=128, vf_nll=True)
    assert _param_count(nll) == _param_count(plain) + 129
    obs = torch.rand(4, OBS_V5)
    assert nll.get_value(obs).shape == (4, 1)
    mean, logvar = nll.get_value_dist(obs)
    assert mean.shape == (4,) and logvar.shape == (4,)
    assert torch.allclose(logvar, torch.zeros(4), atol=1e-6)
    with pytest.raises(ValueError, match="no variance head"):
        plain.get_value_dist(obs)
    path = tmp_path / "nll.pt"
    save_agent(path, nll)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["vf_nll"] is True
    loaded, _ = load_agent(path)
    assert loaded.vf_nll is True
    assert torch.equal(loaded.get_value(obs), nll.get_value(obs))
    assert torch.allclose(
        loaded.get_value_dist(obs)[1], nll.get_value_dist(obs)[1]
    )


def test_warm_start_rejects_a_vf_nll_mismatch():
    torch.manual_seed(0)
    plain = Agent(OBS_V5, NVEC, hidden=16)
    nll = Agent(OBS_V5, NVEC, hidden=16, vf_nll=True)
    with pytest.raises(WarmStartLayoutError, match="variance-head"):
        warm_start_into(nll, plain)
    with pytest.raises(WarmStartLayoutError, match="variance-head"):
        warm_start_into(plain, nll)


def test_vf_sample_draws_around_the_mean_and_requires_vf_nll():
    torch.manual_seed(0)
    agent = Agent(OBS_V5, NVEC, hidden=16, vf_nll=True, vf_sample=True)
    obs = torch.rand(64, OBS_V5)
    mask = torch.ones(64, sum(NVEC), dtype=torch.bool)
    with torch.no_grad():
        mean = agent.get_value(obs)
        draws = torch.stack(
            [agent.get_action_and_value(obs, mask)[3] for _ in range(200)]
        )
    assert draws.shape == (200, 64, 1)
    assert not torch.equal(draws[0], draws[1])
    assert torch.allclose(draws.mean(dim=0), mean, atol=0.5)
    # Evaluation keeps the deterministic mean.
    assert torch.equal(agent.get_value(obs), mean)
    with pytest.raises(ValueError, match="requires vf_nll"):
        Agent(OBS_V5, NVEC, hidden=16, vf_sample=True)


def test_vf_sample_round_trips_through_a_checkpoint(tmp_path):
    torch.manual_seed(0)
    agent = Agent(OBS_V5, NVEC, hidden=16, vf_nll=True, vf_sample=True)
    path = tmp_path / "samp.pt"
    save_agent(path, agent)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["vf_sample"] is True and payload["vf_nll"] is True
    loaded, _ = load_agent(path)
    assert loaded.vf_sample is True and loaded.vf_nll is True


def test_vf_outcome_head_shapes_and_the_value_split():
    torch.manual_seed(0)
    plain = Agent(OBS_V5, NVEC, hidden=128)
    outcome = Agent(OBS_V5, NVEC, hidden=128, vf_outcome=True)
    assert _param_count(outcome) == _param_count(plain) + 129
    obs = torch.rand(4, OBS_V5)
    prediction = outcome.get_outcome(obs)
    assert prediction.shape == (4, 1)
    assert bool((prediction.abs() < 1.0).all())
    value, paired = outcome.get_value_and_outcome(obs)
    assert torch.equal(value, outcome.get_value(obs))
    assert torch.equal(paired, prediction)
    with pytest.raises(ValueError, match="no outcome head"):
        plain.get_outcome(obs)
    with pytest.raises(ValueError, match="no outcome head"):
        plain.get_value_and_outcome(obs)
    with pytest.raises(ValueError, match="mutually exclusive"):
        Agent(OBS_V5, NVEC, hidden=16, vf_nll=True, vf_outcome=True)


def test_vf_outcome_round_trips_through_a_checkpoint(tmp_path):
    torch.manual_seed(0)
    agent = Agent(OBS_V5, NVEC, hidden=16, vf_outcome=True)
    path = tmp_path / "outcome.pt"
    save_agent(path, agent)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["vf_outcome"] is True
    loaded, _ = load_agent(path)
    assert loaded.vf_outcome is True
    obs = torch.rand(4, OBS_V5)
    assert torch.equal(loaded.get_outcome(obs), agent.get_outcome(obs))
    assert torch.equal(loaded.get_value(obs), agent.get_value(obs))


def test_warm_start_bridges_a_vf_outcome_mismatch(tmp_path):
    torch.manual_seed(0)
    source = Agent(OBS_V5, NVEC, hidden=16)
    path = tmp_path / "plain.pt"
    save_agent(path, source)
    target = Agent(OBS_V5, NVEC, hidden=16, vf_outcome=True)
    before_head = target.outcome_head.weight.detach().clone()
    obs = torch.rand(4, OBS_V5)
    result = warm_start_from(path, target)
    assert result.exact is False
    # The shared trunk/actor/critic transfer; the new head keeps its init.
    assert torch.equal(target.get_value(obs), source.get_value(obs))
    assert torch.equal(target.outcome_head.weight.detach(), before_head)
    # The reverse direction drops the head and keeps the critic.
    back_path = tmp_path / "outcome.pt"
    trained = Agent(OBS_V5, NVEC, hidden=16, vf_outcome=True)
    save_agent(back_path, trained)
    dropped = Agent(OBS_V5, NVEC, hidden=16)
    assert warm_start_from(back_path, dropped).exact is False
    assert torch.equal(dropped.get_value(obs), trained.get_value(obs))


@pytest.mark.parametrize(
    "source_arch,target_arch",
    [("deep_ln", "deep"), ("shared", "deep"), ("ln", "shared"), ("deep", "ln")],
)
def test_cross_arch_warm_start_rejects_the_new_trunk_layouts(source_arch, target_arch):
    torch.manual_seed(0)
    source = Agent(OBS_V5, NVEC, hidden=16, arch=source_arch)
    target = Agent(OBS_V5, NVEC, hidden=16, arch=target_arch)
    untouched = target.network[0].weight.clone()
    with pytest.raises(WarmStartLayoutError, match="cross-architecture"):
        warm_start_into(target, source)
    assert torch.equal(target.network[0].weight, untouched)


def test_warm_start_from_rejects_new_archs_and_the_hidden_width_mismatch(tmp_path):
    torch.manual_seed(0)
    source = Agent(OBS_V5, NVEC, hidden=16, arch="deep_ln")
    path = tmp_path / "deep_ln.pt"
    save_agent(path, source)
    target = Agent(OBS_V5, NVEC, hidden=16, arch="deep")
    with pytest.raises(WarmStartLayoutError, match="cross-architecture"):
        warm_start_from(path, target)
    # Same arch, different hidden width: the whole-state load is fail-loud
    # (train.py turns the RuntimeError into SystemExit).
    narrow = Agent(OBS_V5, NVEC, hidden=16)
    narrow_path = tmp_path / "shared16.pt"
    save_agent(narrow_path, narrow)
    wide = Agent(OBS_V5, NVEC, hidden=32)
    with pytest.raises(RuntimeError):
        warm_start_from(narrow_path, wide)

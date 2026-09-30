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
    for arch in ("shared", "towers"):
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

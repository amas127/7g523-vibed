"""Checkpoints and warm starts under the single v5 observation layout.

The layout is v5 everywhere; ``save_agent`` stamps it and :func:`load_agent`
rejects a missing or foreign version.  Warm starts copy same-layout weights
and bridge the shared/towers trunk topologies; a different ``obs_dim`` is a
hard error rather than a guessed column mapping.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from seven523.actions import joint_mask_bits  # noqa: E402
from seven523.env import Seven523Env, encode_observation  # noqa: E402
from seven523.networks import (  # noqa: E402
    Agent,
    NeuralPolicy,
    load_agent,
    save_agent,
    warm_start_from,
    warm_start_into,
)
from seven523.policies import GreedyBot  # noqa: E402
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
    count = lambda agent: sum(p.numel() for p in agent.parameters())
    assert count(shared) == 55_179
    assert count(towers) == 92_427
    one_trunk = sum(p.numel() for p in shared.network.parameters())
    assert one_trunk == 37_248
    assert count(towers) - count(shared) == one_trunk


def test_neural_policy_towers_acts_legally_and_matches_masked_argmax():
    torch.manual_seed(0)
    agent = Agent(OBS_V5, NVEC, hidden=16, arch="towers")
    policy = NeuralPolicy(agent, DEFAULT_RULES)
    env = Seven523Env(seed=0, opponents=[GreedyBot(), GreedyBot()])
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

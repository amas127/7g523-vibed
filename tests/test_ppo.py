"""PPO math tested on its own: GAE and the update, no vector env needed."""
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from seven523.networks import Agent  # noqa: E402
from seven523.ppo import PPOConfig, RolloutBatch, compute_gae, ppo_update  # noqa: E402

OBS_DIM = 8
NVEC = [4, 2]


def _config(**overrides):
    base = dict(
        num_minibatches=2,
        update_epochs=1,
        norm_adv=True,
        clip_coef=0.1,
        clip_vloss=True,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=None,
    )
    base.update(overrides)
    return PPOConfig(**base)


def _tensor(values):
    return torch.tensor([[[v]] for v in values], dtype=torch.float32)


def test_compute_gae_matches_hand_computed_value():
    rewards = _tensor([1.0, 1.0])
    values = _tensor([0.0, 0.0])
    dones = _tensor([0.0, 0.0])
    next_value = torch.zeros(1, 1)
    next_done = torch.zeros(1)
    advantages, returns = compute_gae(
        rewards, values, dones, next_value, next_done, gamma=1.0, gae_lambda=1.0
    )
    assert torch.allclose(advantages.flatten(), torch.tensor([2.0, 1.0]))
    assert torch.allclose(returns.flatten(), torch.tensor([2.0, 1.0]))


def test_compute_gae_bootstraps_from_a_terminal_next_value():
    rewards = _tensor([1.0])
    values = _tensor([0.5])
    dones = _tensor([0.0])
    next_value = torch.tensor([[2.0]])
    next_done = torch.zeros(1)
    advantages, returns = compute_gae(
        rewards, values, dones, next_value, next_done, gamma=1.0, gae_lambda=1.0
    )
    # delta = 1 + 2 - 0.5 = 2.5; returns = advantage + value
    assert torch.allclose(advantages.flatten(), torch.tensor([2.5]))
    assert torch.allclose(returns.flatten(), torch.tensor([3.0]))


def test_compute_gae_discounts_with_gamma():
    rewards = _tensor([0.0, 1.0])
    values = _tensor([0.0, 0.0])
    dones = _tensor([0.0, 0.0])
    next_value = torch.zeros(1, 1)
    next_done = torch.zeros(1)
    advantages, _ = compute_gae(
        rewards, values, dones, next_value, next_done, gamma=0.5, gae_lambda=1.0
    )
    assert torch.allclose(advantages.flatten(), torch.tensor([0.5, 1.0]))


def test_compute_gae_without_gae_returns_monte_carlo_returns():
    rewards = _tensor([1.0, 2.0])
    values = _tensor([0.0, 0.0])
    dones = _tensor([0.0, 0.0])
    next_value = torch.zeros(1, 1)
    next_done = torch.zeros(1)
    advantages, returns = compute_gae(
        rewards,
        values,
        dones,
        next_value,
        next_done,
        gamma=1.0,
        gae_lambda=1.0,
        use_gae=False,
    )
    assert torch.allclose(returns.flatten(), torch.tensor([3.0, 2.0]))
    assert torch.allclose(advantages.flatten(), torch.tensor([3.0, 2.0]))


def test_config_from_args_copies_hyperparameters():
    args = SimpleNamespace(
        num_minibatches=3,
        update_epochs=2,
        norm_adv=False,
        clip_coef=0.2,
        clip_vloss=False,
        ent_coef=0.03,
        vf_coef=0.25,
        max_grad_norm=0.7,
        target_kl=0.02,
    )
    config = PPOConfig.from_args(args)
    assert config == _config(
        num_minibatches=3,
        update_epochs=2,
        norm_adv=False,
        clip_coef=0.2,
        clip_vloss=False,
        ent_coef=0.03,
        vf_coef=0.25,
        max_grad_norm=0.7,
        target_kl=0.02,
    )


def test_rollout_batch_flatten_shapes():
    steps, envs = 3, 2
    batch = RolloutBatch.flatten(
        obs=torch.zeros(steps, envs, OBS_DIM),
        actions=torch.zeros(steps, envs, len(NVEC)),
        logprobs=torch.zeros(steps, envs),
        advantages=torch.zeros(steps, envs),
        returns=torch.zeros(steps, envs),
        values=torch.zeros(steps, envs),
        action_masks=torch.ones(steps, envs, sum(NVEC)),
    )
    assert batch.size == steps * envs
    assert batch.obs.shape == (steps * envs, OBS_DIM)
    assert batch.actions.shape == (steps * envs, len(NVEC))


def test_ppo_update_moves_the_parameters_and_reports_finite_losses():
    torch.manual_seed(0)
    agent = Agent(OBS_DIM, NVEC, hidden=16)
    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)
    batch = RolloutBatch.flatten(
        obs=torch.rand(4, 2, OBS_DIM),
        actions=torch.zeros(4, 2, len(NVEC), dtype=torch.long),
        logprobs=torch.zeros(4, 2),
        advantages=torch.randn(4, 2),
        returns=torch.randn(4, 2),
        values=torch.randn(4, 2),
        action_masks=torch.ones(4, 2, sum(NVEC)),
    )
    before = agent.actor.weight.detach().clone()
    losses = ppo_update(agent, optimizer, batch, _config())
    assert set(losses) == {
        "value_loss",
        "policy_loss",
        "entropy",
        "old_approx_kl",
        "approx_kl",
        "clipfrac",
        "explained_variance",
    }
    assert all(np.isfinite(value) for value in losses.values())
    assert not torch.equal(before, agent.actor.weight.detach())

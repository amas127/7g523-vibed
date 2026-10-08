"""PPO math tested on its own: GAE and the update, no vector env needed."""
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from seven523.history import EVENT_DIM  # noqa: E402
from seven523.networks import Agent  # noqa: E402
from seven523.ppo import (  # noqa: E402
    PPOConfig,
    RolloutBatch,
    compute_gae,
    decoupled_value_loss,
    gaussian_nll,
    ppo_update,
)

OBS_DIM = 8
NVEC = [4, 2]


def _config(**overrides):
    base = {
        "num_minibatches": 2,
        "update_epochs": 1,
        "norm_adv": True,
        "clip_coef": 0.1,
        "clip_vloss": True,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "target_kl": None,
    }
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


def test_rollout_batch_flatten_shapes_with_events():
    steps, envs, length = 3, 2, 5
    batch = RolloutBatch.flatten(
        obs=torch.zeros(steps, envs, OBS_DIM),
        actions=torch.zeros(steps, envs, len(NVEC)),
        logprobs=torch.zeros(steps, envs),
        advantages=torch.zeros(steps, envs),
        returns=torch.zeros(steps, envs),
        values=torch.zeros(steps, envs),
        action_masks=torch.ones(steps, envs, sum(NVEC)),
        seqs=torch.zeros(steps, envs, 7, dtype=torch.long),
        events=torch.rand(steps, envs, length, EVENT_DIM),
        event_seats=torch.zeros(steps, envs, length, dtype=torch.long),
        event_mask=torch.ones(steps, envs, length, dtype=torch.bool),
    )
    assert batch.seqs.shape == (steps * envs, 7)
    assert batch.events.shape == (steps * envs, length, EVENT_DIM)
    assert batch.event_seats.shape == (steps * envs, length)
    assert batch.event_mask.shape == (steps * envs, length)
    # Absence stays absent on the MLP path.
    empty = RolloutBatch.flatten(
        obs=torch.zeros(steps, envs, OBS_DIM),
        actions=torch.zeros(steps, envs, len(NVEC)),
        logprobs=torch.zeros(steps, envs),
        advantages=torch.zeros(steps, envs),
        returns=torch.zeros(steps, envs),
        values=torch.zeros(steps, envs),
        action_masks=torch.ones(steps, envs, sum(NVEC)),
    )
    assert empty.seqs is None and empty.events is None
    assert empty.event_seats is None and empty.event_mask is None
    assert empty.outcome_targets is None and empty.outcome_mask is None


def test_rollout_batch_flatten_carries_outcome_fields():
    steps, envs = 3, 2
    targets = torch.randn(steps, envs)
    mask = torch.zeros(steps, envs, dtype=torch.bool)
    mask[0, 1] = True
    batch = RolloutBatch.flatten(
        obs=torch.zeros(steps, envs, OBS_DIM),
        actions=torch.zeros(steps, envs, len(NVEC)),
        logprobs=torch.zeros(steps, envs),
        advantages=torch.zeros(steps, envs),
        returns=torch.zeros(steps, envs),
        values=torch.zeros(steps, envs),
        action_masks=torch.ones(steps, envs, sum(NVEC)),
        outcome_targets=targets,
        outcome_mask=mask,
    )
    assert batch.outcome_targets.shape == (steps * envs,)
    assert torch.equal(batch.outcome_targets, targets.reshape(-1))
    assert batch.outcome_mask.shape == (steps * envs,)
    assert torch.equal(batch.outcome_mask, mask.reshape(-1))


def test_ppo_update_trains_the_event_encoder():
    torch.manual_seed(0)
    agent = Agent(OBS_DIM, NVEC, hidden=16, event_len=6, event_hidden=4)
    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)
    length = 6
    batch = RolloutBatch.flatten(
        obs=torch.rand(4, 2, OBS_DIM),
        actions=torch.zeros(4, 2, len(NVEC), dtype=torch.long),
        logprobs=torch.zeros(4, 2),
        advantages=torch.randn(4, 2),
        returns=torch.randn(4, 2),
        values=torch.randn(4, 2),
        action_masks=torch.ones(4, 2, sum(NVEC)),
        events=torch.rand(4, 2, length, EVENT_DIM),
        event_seats=torch.randint(0, 3, (4, 2, length)),
        event_mask=torch.ones(4, 2, length, dtype=torch.bool),
    )
    before = agent.event_encoder.mlp[0].weight.detach().clone()
    losses = ppo_update(agent, optimizer, batch, _config())
    assert all(np.isfinite(value) for value in losses.values())
    assert not torch.equal(before, agent.event_encoder.mlp[0].weight.detach())


def test_gaussian_nll_matches_the_closed_form():
    mean = torch.tensor([0.0, 1.0])
    logvar = torch.tensor([0.0, 2.0])
    target = torch.tensor([1.0, 3.0])
    expected = 0.5 * (logvar + (target - mean) ** 2 * torch.exp(-logvar))
    assert torch.allclose(gaussian_nll(mean, logvar, target), expected)
    # The clamp keeps extreme logvar values finite.
    assert torch.isfinite(gaussian_nll(mean, torch.tensor([50.0, -50.0]), target)).all()


def test_ppo_update_trains_the_variance_head():
    torch.manual_seed(0)
    agent = Agent(OBS_DIM, NVEC, hidden=16, vf_nll=True)
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
    before = agent.critic.weight.detach().clone()
    losses = ppo_update(agent, optimizer, batch, _config(vf_nll=True))
    assert all(np.isfinite(value) for value in losses.values())
    after = agent.critic.weight.detach()
    # Both the mean row and the zero-initialised logvar row receive gradient.
    assert not torch.equal(before[0], after[0])
    assert not torch.equal(before[1], after[1])


def test_decoupled_value_loss_keeps_the_base_mean_gradient():
    mean = torch.tensor([0.5, -0.5], requires_grad=True)
    logvar = torch.tensor([0.0, 1.0], requires_grad=True)
    target = torch.tensor([1.0, 0.0])
    old = torch.zeros(2)
    mean_loss, var_loss = decoupled_value_loss(
        mean, logvar, target, clip_vloss=False, old_value=old, clip_coef=0.1
    )
    assert torch.allclose(
        mean_loss.detach(), 0.5 * ((mean.detach() - target) ** 2).mean()
    )
    expected_var = 0.5 * (
        logvar.detach() + (target - mean.detach()) ** 2 * torch.exp(-logvar.detach())
    ).mean()
    assert torch.allclose(var_loss.detach(), expected_var)
    (mean_loss + var_loss).backward()
    # The mean path sees exactly the MSE gradient: the variance objective is
    # detached from it (no inverse-variance reweighting).
    assert torch.allclose(mean.grad, (mean.detach() - target) / 2)
    assert logvar.grad is not None and logvar.grad.abs().sum() > 0


def test_ppo_update_trains_the_decoupled_variance_head():
    torch.manual_seed(0)
    agent = Agent(OBS_DIM, NVEC, hidden=16, vf_nll=True)
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
    losses = ppo_update(
        agent, optimizer, batch, _config(vf_nll=True, vf_decoupled=True)
    )
    assert all(np.isfinite(value) for value in losses.values())


def test_outcome_gae_splits_linearly_into_base_and_jump_streams():
    torch.manual_seed(0)
    steps = 5
    base_rewards = torch.randn(steps, 1)
    jumps = torch.zeros(steps, 1)
    jumps[-1] = 0.75
    values_base = torch.randn(steps, 1)
    values_jump = torch.randn(steps, 1)
    dones = torch.zeros(steps, 1)
    next_base = torch.randn(1, 1)
    next_jump = torch.randn(1, 1)
    next_done = torch.zeros(1)
    kwargs = {"gamma": 0.9, "gae_lambda": 0.8}
    adv_base, ret_base = compute_gae(
        base_rewards, values_base, dones, next_base, next_done, **kwargs
    )
    adv_jump, _ = compute_gae(
        jumps, values_jump, dones, next_jump, next_done, **kwargs
    )
    adv_total, _ = compute_gae(
        base_rewards + jumps,
        values_base + values_jump,
        dones,
        next_base + next_jump,
        next_done,
        **kwargs,
    )
    # The decomposed advantages are exactly the total-reward advantages, and
    # the critic target (returns) is the jump-free stream.
    assert torch.allclose(adv_base + adv_jump, adv_total)
    assert torch.allclose(ret_base, adv_base + values_base)


def _outcome_batch(mask):
    targets = torch.cat([torch.full((6,), 0.5), torch.full((2,), -0.25)])
    return RolloutBatch.flatten(
        obs=torch.rand(4, 2, OBS_DIM),
        actions=torch.zeros(4, 2, len(NVEC), dtype=torch.long),
        logprobs=torch.zeros(4, 2),
        advantages=torch.randn(4, 2),
        returns=torch.randn(4, 2),
        values=torch.randn(4, 2),
        action_masks=torch.ones(4, 2, sum(NVEC)),
        outcome_targets=targets,
        outcome_mask=mask,
    )


def test_ppo_update_trains_the_outcome_head_and_reports_accuracy():
    torch.manual_seed(0)
    agent = Agent(OBS_DIM, NVEC, hidden=16, vf_outcome=True)
    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)
    mask = torch.tensor([True] * 6 + [False] * 2)
    batch = _outcome_batch(mask)
    before = agent.outcome_head.weight.detach().clone()
    losses = ppo_update(agent, optimizer, batch, _config(vf_outcome=True))
    assert {"outcome_loss", "outcome_accuracy"} <= set(losses)
    assert all(np.isfinite(value) for value in losses.values())
    assert not torch.equal(before, agent.outcome_head.weight.detach())


def test_ppo_update_masks_unavailable_outcome_labels():
    torch.manual_seed(0)
    agent = Agent(OBS_DIM, NVEC, hidden=16, vf_outcome=True)
    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)
    batch = _outcome_batch(torch.zeros(8, dtype=torch.bool))
    losses = ppo_update(agent, optimizer, batch, _config(vf_outcome=True))
    assert losses["outcome_loss"] == 0.0
    assert np.isnan(losses["outcome_accuracy"])

"""PPO's math, separated from the training loop.

``train()`` keeps the CLI, the vector env and the orchestration; this module
holds the two pieces worth testing on their own: generalised advantage
estimation and the clipped update.  The algorithm is a faithful port of
``ppo_multidiscrete_mask.py`` (ADR-0003) — only the boundaries moved.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .networks import Agent

__all__ = ["PPOConfig", "RolloutBatch", "compute_gae", "ppo_update"]


@dataclass(frozen=True, slots=True)
class PPOConfig:
    """The update hyperparameters, lifted out of the argparse namespace."""

    num_minibatches: int
    update_epochs: int
    norm_adv: bool
    clip_coef: float
    clip_vloss: bool
    ent_coef: float
    vf_coef: float
    max_grad_norm: float
    target_kl: float | None

    @classmethod
    def from_args(cls, args: Any) -> "PPOConfig":
        return cls(
            num_minibatches=args.num_minibatches,
            update_epochs=args.update_epochs,
            norm_adv=args.norm_adv,
            clip_coef=args.clip_coef,
            clip_vloss=args.clip_vloss,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            target_kl=args.target_kl,
        )


@dataclass(frozen=True, slots=True)
class RolloutBatch:
    """One flattened rollout: ``(batch, ...)`` tensors ready for the update."""

    obs: torch.Tensor
    actions: torch.Tensor
    logprobs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    values: torch.Tensor
    action_masks: torch.Tensor

    @property
    def size(self) -> int:
        return int(self.obs.shape[0])

    @classmethod
    def flatten(
        cls,
        obs: torch.Tensor,
        actions: torch.Tensor,
        logprobs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        values: torch.Tensor,
        action_masks: torch.Tensor,
    ) -> "RolloutBatch":
        """Collapse the ``(steps, envs, ...)`` storage into flat batches."""
        return cls(
            obs=obs.reshape((-1, obs.shape[-1])),
            actions=actions.reshape((-1, actions.shape[-1])),
            logprobs=logprobs.reshape(-1),
            advantages=advantages.reshape(-1),
            returns=returns.reshape(-1),
            values=values.reshape(-1),
            action_masks=action_masks.reshape((-1, action_masks.shape[-1])),
        )


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: torch.Tensor,
    next_done: torch.Tensor,
    *,
    gamma: float,
    gae_lambda: float,
    use_gae: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(advantages, returns)``.

    With ``use_gae`` this is generalised advantage estimation; otherwise plain
    discounted returns with ``advantages = returns - values`` (the reference's
    non-GAE branch).
    """
    num_steps = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    if use_gae:
        lastgaelam = 0.0
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = (
                delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            )
        returns = advantages + values
        return advantages, returns

    returns = torch.zeros_like(rewards)
    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            nextnonterminal = 1.0 - next_done
            next_return = next_value
        else:
            nextnonterminal = 1.0 - dones[t + 1]
            next_return = returns[t + 1]
        returns[t] = rewards[t] + gamma * nextnonterminal * next_return
    return returns - values, returns


def ppo_update(
    agent: Agent,
    optimizer: torch.optim.Optimizer,
    batch: RolloutBatch,
    config: PPOConfig,
) -> dict[str, float]:
    """Run the clipped update over ``batch``; return this update's loss metrics."""
    minibatch_size = batch.size // config.num_minibatches
    b_inds = np.arange(batch.size)
    clipfracs: list[float] = []
    old_approx_kl = approx_kl = pg_loss = v_loss = entropy_loss = torch.tensor(0.0)

    for _epoch in range(config.update_epochs):
        np.random.shuffle(b_inds)
        for start in range(0, batch.size, minibatch_size):
            mb_inds = b_inds[start : start + minibatch_size]

            _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                batch.obs[mb_inds],
                batch.action_masks[mb_inds],
                batch.actions.long()[mb_inds].T,
            )
            logratio = newlogprob - batch.logprobs[mb_inds]
            ratio = logratio.exp()

            with torch.no_grad():
                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs.append(
                    ((ratio - 1.0).abs() > config.clip_coef).float().mean().item()
                )

            mb_advantages = batch.advantages[mb_inds]
            if config.norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                    mb_advantages.std() + 1e-8
                )

            # Policy loss
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(
                ratio, 1 - config.clip_coef, 1 + config.clip_coef
            )
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value loss
            newvalue = newvalue.view(-1)
            if config.clip_vloss:
                v_loss_unclipped = (newvalue - batch.returns[mb_inds]) ** 2
                v_clipped = batch.values[mb_inds] + torch.clamp(
                    newvalue - batch.values[mb_inds],
                    -config.clip_coef,
                    config.clip_coef,
                )
                v_loss_clipped = (v_clipped - batch.returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 * ((newvalue - batch.returns[mb_inds]) ** 2).mean()

            entropy_loss = entropy.mean()
            loss = pg_loss - config.ent_coef * entropy_loss + v_loss * config.vf_coef

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm)
            optimizer.step()

        if config.target_kl is not None and approx_kl > config.target_kl:
            break

    y_pred, y_true = batch.values.cpu().numpy(), batch.returns.cpu().numpy()
    var_y = np.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

    return {
        "value_loss": float(v_loss.item()),
        "policy_loss": float(pg_loss.item()),
        "entropy": float(entropy_loss.item()),
        "old_approx_kl": float(old_approx_kl.item()),
        "approx_kl": float(approx_kl.item()),
        "clipfrac": float(np.mean(clipfracs)) if clipfracs else float("nan"),
        "explained_variance": float(explained_var),
    }

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

__all__ = [
    "PPOConfig",
    "RolloutBatch",
    "compute_gae",
    "decoupled_value_loss",
    "gaussian_nll",
    "ppo_update",
]


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
    vf_nll: bool = False
    vf_decoupled: bool = False
    vf_outcome: bool = False
    outcome_coef: float = 1.0

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
            vf_nll=getattr(args, "vf_nll", False),
            vf_decoupled=getattr(args, "vf_decoupled", False),
            vf_outcome=getattr(args, "vf_outcome", False),
            outcome_coef=getattr(args, "outcome_coef", 1.0),
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
    #: D1-lite history tokens ``(batch, length)``; ``None`` on the MLP path.
    seqs: torch.Tensor | None = None
    #: EVH event tensors ``(batch, event_length, EVENT_DIM)`` float32, relative
    #: seats ``(batch, event_length)`` int64 and mask ``(batch, event_length)``
    #: bool; all ``None`` on the non-event paths and always set/absent together.
    events: torch.Tensor | None = None
    event_seats: torch.Tensor | None = None
    event_mask: torch.Tensor | None = None
    #: ``vf_outcome`` only: regression target ``gamma**(T-t) * outcome``
    #: and its per-step availability mask (episodes finishing inside the
    #: rollout).  Always set/absent together.
    outcome_targets: torch.Tensor | None = None
    outcome_mask: torch.Tensor | None = None

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
        seqs: torch.Tensor | None = None,
        events: torch.Tensor | None = None,
        event_seats: torch.Tensor | None = None,
        event_mask: torch.Tensor | None = None,
        outcome_targets: torch.Tensor | None = None,
        outcome_mask: torch.Tensor | None = None,
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
            seqs=(
                None if seqs is None else seqs.reshape((-1, seqs.shape[-1]))
            ),
            events=(
                None
                if events is None
                else events.reshape((-1, events.shape[-2], events.shape[-1]))
            ),
            event_seats=(
                None
                if event_seats is None
                else event_seats.reshape((-1, event_seats.shape[-1]))
            ),
            event_mask=(
                None
                if event_mask is None
                else event_mask.reshape((-1, event_mask.shape[-1]))
            ),
            outcome_targets=(
                None
                if outcome_targets is None
                else outcome_targets.reshape(-1)
            ),
            outcome_mask=(
                None if outcome_mask is None else outcome_mask.reshape(-1)
            ),
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


def gaussian_nll(
    mean: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Per-element heteroscedastic Gaussian NLL (``logvar`` clamped to ±10)."""
    logvar = logvar.clamp(-10.0, 10.0)
    return 0.5 * (logvar + (target - mean) ** 2 * torch.exp(-logvar))


def decoupled_value_loss(
    mean: torch.Tensor,
    logvar: torch.Tensor,
    target: torch.Tensor,
    *,
    clip_vloss: bool,
    old_value: torch.Tensor,
    clip_coef: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Amendment A4: mean trained exactly like the base critic, variance apart.

    Returns ``(mean_loss, var_loss)``: the first is the same clipped MSE the
    mean-only critic uses; the second is the Gaussian NLL of the variance
    head with the mean detached, so the inverse-variance weight never
    reweights the mean's gradient.
    """
    if clip_vloss:
        unclipped = (mean - target) ** 2
        clipped = old_value + torch.clamp(mean - old_value, -clip_coef, clip_coef)
        mean_loss = 0.5 * torch.max(unclipped, (clipped - target) ** 2).mean()
    else:
        mean_loss = 0.5 * ((mean - target) ** 2).mean()
    logvar = logvar.clamp(-10.0, 10.0)
    var_loss = 0.5 * (
        logvar + (target - mean.detach()) ** 2 * torch.exp(-logvar)
    ).mean()
    return mean_loss, var_loss


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
    outcome_accuracy = float("nan")

    for _epoch in range(config.update_epochs):
        np.random.shuffle(b_inds)
        for start in range(0, batch.size, minibatch_size):
            mb_inds = b_inds[start : start + minibatch_size]

            _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                batch.obs[mb_inds],
                batch.action_masks[mb_inds],
                batch.actions.long()[mb_inds].T,
                seqs=None if batch.seqs is None else batch.seqs[mb_inds],
                events=None if batch.events is None else batch.events[mb_inds],
                event_seats=(
                    None
                    if batch.event_seats is None
                    else batch.event_seats[mb_inds]
                ),
                event_mask=(
                    None if batch.event_mask is None else batch.event_mask[mb_inds]
                ),
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
            if config.vf_nll:
                # Amendment A2: heteroscedastic critic.  The head predicts
                # (mean, logvar) and the loss is the Gaussian NLL; the clipped
                # MSE (config.clip_vloss) is defined for the mean-only head
                # and is not applied here.
                mean, logvar = agent.get_value_dist(
                    batch.obs[mb_inds],
                    seqs=None if batch.seqs is None else batch.seqs[mb_inds],
                    events=None if batch.events is None else batch.events[mb_inds],
                    event_seats=(
                        None
                        if batch.event_seats is None
                        else batch.event_seats[mb_inds]
                    ),
                    event_mask=(
                        None
                        if batch.event_mask is None
                        else batch.event_mask[mb_inds]
                    ),
                )
                if config.vf_decoupled:
                    mean_loss, var_loss = decoupled_value_loss(
                        mean,
                        logvar,
                        batch.returns[mb_inds],
                        clip_vloss=config.clip_vloss,
                        old_value=batch.values[mb_inds],
                        clip_coef=config.clip_coef,
                    )
                    v_loss = mean_loss + var_loss
                else:
                    v_loss = gaussian_nll(
                        mean, logvar, batch.returns[mb_inds]
                    ).mean()
            elif config.clip_vloss:
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

            # Outcome-head loss (``vf_outcome``): the discounted win/loss
            # target is only available for states whose episode ended inside
            # the rollout, so the masked subset sees the auxiliary target.
            outcome_loss = torch.tensor(0.0)
            if config.vf_outcome:
                outcome_pred = agent.get_outcome(
                    batch.obs[mb_inds],
                    seqs=None if batch.seqs is None else batch.seqs[mb_inds],
                    events=None if batch.events is None else batch.events[mb_inds],
                    event_seats=(
                        None
                        if batch.event_seats is None
                        else batch.event_seats[mb_inds]
                    ),
                    event_mask=(
                        None
                        if batch.event_mask is None
                        else batch.event_mask[mb_inds]
                    ),
                ).squeeze(-1)
                assert batch.outcome_targets is not None
                assert batch.outcome_mask is not None
                masked = batch.outcome_mask[mb_inds]
                target = batch.outcome_targets[mb_inds]
                if bool(masked.any()):
                    outcome_loss = ((outcome_pred - target)[masked] ** 2).mean()
                    with torch.no_grad():
                        outcome_accuracy = float(
                            (
                                torch.sign(outcome_pred[masked])
                                == torch.sign(target[masked])
                            )
                            .float()
                            .mean()
                        )

            entropy_loss = entropy.mean()
            loss = (
                pg_loss
                - config.ent_coef * entropy_loss
                + v_loss * config.vf_coef
                + config.outcome_coef * outcome_loss
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm)
            optimizer.step()

        if config.target_kl is not None and approx_kl > config.target_kl:
            break

    y_pred, y_true = batch.values.cpu().numpy(), batch.returns.cpu().numpy()
    var_y = np.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

    losses = {
        "value_loss": float(v_loss.item()),
        "policy_loss": float(pg_loss.item()),
        "entropy": float(entropy_loss.item()),
        "old_approx_kl": float(old_approx_kl.item()),
        "approx_kl": float(approx_kl.item()),
        "clipfrac": float(np.mean(clipfracs)) if clipfracs else float("nan"),
        "explained_variance": float(explained_var),
    }
    if config.vf_outcome:
        losses["outcome_loss"] = float(outcome_loss.item())
        losses["outcome_accuracy"] = outcome_accuracy
    return losses

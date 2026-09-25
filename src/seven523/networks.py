"""Torch actor-critic and the learned adapter of the ``Policy`` seam.

The rules core and :mod:`seven523.env` never import torch; this module is the
only place that does.  It needs the optional ``train`` dependency group:

    uv sync --group train
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from .env import encode_observation
from .game import View
from .rules import DEFAULT_RULES, Rules

__all__ = [
    "Agent",
    "CategoricalMasked",
    "NeuralPolicy",
    "layer_init",
    "load_agent",
    "save_agent",
    "warm_start_into",
]


def layer_init(layer: nn.Linear, std: float = 2**0.5, bias_const: float = 0.0) -> nn.Linear:
    """Orthogonal init from the reference implementation (PPO detail #1)."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class CategoricalMasked(Categorical):
    """Categorical that assigns ``-1e8`` logits to illegal actions.

    Entropy ignores the illegal entries, so it is not inflated by the mask.
    Port of ``ppo_multidiscrete_mask.py::CategoricalMasked``.
    """

    def __init__(
        self,
        probs: torch.Tensor | None = None,
        logits: torch.Tensor | None = None,
        validate_args: bool | None = None,
        masks: torch.Tensor | None = None,
    ) -> None:
        self.masks = masks
        if masks is None:
            super().__init__(probs, logits, validate_args)
        else:
            masks = masks.to(torch.bool)
            logits = torch.where(
                masks, logits, torch.tensor(-1e8, device=logits.device)
            )
            super().__init__(probs, logits, validate_args)

    def entropy(self) -> torch.Tensor:
        if self.masks is None:
            return super().entropy()
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks.to(torch.bool), p_log_p, 0.0)
        return -p_log_p.sum(-1)


class Agent(nn.Module):
    """MLP actor-critic with one masked categorical head per action dimension.

    The environment's action space is ``MultiDiscrete([134, 4])``: head 0 is
    the rank template, head 1 the top-card suit (ADR-0004).  The ``split``
    machinery keeps this generic for any future ``nvec``.
    """

    def __init__(
        self, obs_dim: int, nvec: Sequence[int], hidden: int = 128
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.nvec = torch.as_tensor(list(nvec), dtype=torch.long)
        self.hidden = int(hidden)
        self.network = nn.Sequential(
            layer_init(nn.Linear(self.obs_dim, hidden)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(hidden, int(self.nvec.sum())), std=0.01)
        self.critic = layer_init(nn.Linear(hidden, 1), std=1.0)

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(self.network(x))

    def get_action_and_value(
        self,
        x: torch.Tensor,
        action_mask: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(action, logprob, entropy, value)``.

        ``action`` is ``(num_heads, batch)`` when given (the training path);
        the sampled action is returned as ``(batch, num_heads)``, matching the
        reference script.
        """
        hidden = self.network(x)
        logits = self.actor(hidden)
        split_logits = torch.split(logits, self.nvec.tolist(), dim=1)
        split_masks = torch.split(action_mask, self.nvec.tolist(), dim=1)
        multi_categoricals = [
            CategoricalMasked(logits=head_logits, masks=head_mask)
            for head_logits, head_mask in zip(split_logits, split_masks)
        ]
        if action is None:
            action = torch.stack(
                [categorical.sample() for categorical in multi_categoricals]
            )
        else:
            action = action.long()
        logprob = torch.stack(
            [
                categorical.log_prob(head_action)
                for head_action, categorical in zip(action, multi_categoricals)
            ]
        ).sum(0)
        entropy = torch.stack(
            [categorical.entropy() for categorical in multi_categoricals]
        ).sum(0)
        return action.T, logprob, entropy, self.critic(hidden)


def save_agent(path: str | Path, agent: Agent, extra: dict[str, Any] | None = None) -> None:
    """Checkpoint the network so :class:`NeuralPolicy` / eval can rebuild it."""
    payload: dict[str, Any] = {
        "model": agent.state_dict(),
        "obs_dim": agent.obs_dim,
        "nvec": agent.nvec.tolist(),
        "hidden": agent.hidden,
    }
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)


def load_agent(
    path: str | Path, device: str | torch.device = "cpu"
) -> tuple[Agent, dict[str, Any]]:
    """Rebuild an :class:`Agent` from :func:`save_agent`; returns ``(agent, extra)``."""
    payload = torch.load(path, map_location=device, weights_only=True)
    agent = Agent(payload["obs_dim"], payload["nvec"], payload["hidden"])
    agent.load_state_dict(payload["model"])
    return agent.to(device), payload.get("extra", {})


def warm_start_into(agent: Agent, loaded: Agent) -> list[str]:
    """Copy compatible weights from ``loaded`` into ``agent`` in place.

    Used when the action heads changed (e.g. a 134-template checkpoint into the
    new 134+4 agent): the trunk and critic are copied whole, and the actor's
    flattened head rows are copied by prefix (head order is stable).
    """
    copied: list[str] = []
    source = loaded.state_dict()
    for key, value in agent.state_dict().items():
        origin = source.get(key)
        if origin is None:
            continue
        if origin.shape == value.shape:
            value.copy_(origin)
            copied.append(key)
        elif key in {"actor.weight", "actor.bias"} and origin.shape[0] < value.shape[0]:
            value[: origin.shape[0]].copy_(origin)
            copied.append(f"{key}[:{origin.shape[0]}]")
    return copied


class NeuralPolicy:
    """Inference adapter implementing the ``Policy`` protocol (greedy by default).

    Self-play opponents share one instance: refreshing its ``agent`` weights
    in place publishes the new frozen snapshot without rebuilding the envs.
    """

    def __init__(
        self,
        agent: Agent,
        rules: Rules = DEFAULT_RULES,
        device: str | torch.device = "cpu",
        sample: bool = False,
        seed: int | None = None,
    ) -> None:
        self.agent = agent.to(device).eval()
        self.rules = rules
        self.device = torch.device(device)
        self.sample = sample
        self.rng = random.Random(seed)

    @torch.no_grad()
    def act(self, view: View) -> tuple[int, int | None]:
        """Pick ``(template_id, suit | None)``; ``suit`` is ``None`` on the old
        single-head checkpoints so they keep working."""
        nvec = self.agent.nvec.tolist()
        bits = [(view.mask >> index) & 1 for index in range(nvec[0])]
        for size in nvec[1:]:
            # Extra heads are preference-only: every value is executable because
            # the engine falls back to the strongest realisation (ADR-0004).
            bits.extend([1] * size)
        mask = torch.tensor([bits], dtype=torch.bool, device=self.device)
        if not bool(mask[:, : nvec[0]].any()):
            raise ValueError("NeuralPolicy was asked to act with no legal action")
        obs = torch.as_tensor(
            encode_observation(view, self.rules), dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        logits = self.agent.actor(self.agent.network(obs))
        choices: list[int] = []
        for head_logits, head_mask in zip(
            torch.split(logits, nvec, dim=1), torch.split(mask, nvec, dim=1)
        ):
            if self.sample:
                choice = CategoricalMasked(
                    logits=head_logits, masks=head_mask
                ).sample()
            else:
                choice = torch.where(
                    head_mask,
                    head_logits,
                    torch.tensor(-1e8, device=self.device),
                ).argmax(dim=1)
            choices.append(int(choice.item()))
        return choices[0], (choices[1] if len(choices) > 1 else None)

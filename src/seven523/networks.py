"""Torch actor-critic and the learned adapter of the ``Policy`` seam.

The rules core and :mod:`seven523.env` never import torch; this module is the
only place that does.  It needs the optional ``train`` dependency group:

    uv sync --group train
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from .actions import joint_mask_bits
from .env import OBS_VERSION, encode_observation
from .game import View
from .rules import DEFAULT_RULES, Rules

__all__ = [
    "Agent",
    "CategoricalMasked",
    "NeuralPolicy",
    "WarmStart",
    "WarmStartLayoutError",
    "layer_init",
    "load_agent",
    "save_agent",
    "warm_start_from",
    "warm_start_into",
]


def layer_init(layer: nn.Linear, std: float = 2**0.5, bias_const: float = 0.0) -> nn.Linear:
    """Orthogonal init from the reference implementation (PPO detail #1)."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
}

#: ``shared`` = one trunk feeding both heads (ADR-0003 source); ``towers`` =
#: independent actor/critic trunks (the reference ``ppo.py`` layout).
_ARCHITECTURES = ("shared", "towers")


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
        self,
        obs_dim: int,
        nvec: Sequence[int],
        hidden: int = 128,
        activation: str = "relu",
        arch: str = "shared",
    ) -> None:
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"unknown activation {activation!r}; "
                f"choose from {sorted(_ACTIVATIONS)}"
            )
        if arch not in _ARCHITECTURES:
            raise ValueError(
                f"unknown arch {arch!r}; choose from {sorted(_ARCHITECTURES)}"
            )
        self.obs_dim = int(obs_dim)
        self.nvec = torch.as_tensor(list(nvec), dtype=torch.long)
        self.hidden = int(hidden)
        self.activation = activation
        #: Trunk topology: the shared checkpoint keys (``network.*``,
        #: ``actor.*``, ``critic.*``) stay unchanged by design.
        self.arch = arch
        make_activation = _ACTIVATIONS[activation]
        if arch == "towers":
            # No cross-connections: the policy and value losses never share a
            # hidden parameter, so their gradients cannot interfere.
            self.actor_network = nn.Sequential(
                layer_init(nn.Linear(self.obs_dim, hidden)),
                make_activation(),
                layer_init(nn.Linear(hidden, hidden)),
                make_activation(),
            )
            self.critic_network = nn.Sequential(
                layer_init(nn.Linear(self.obs_dim, hidden)),
                make_activation(),
                layer_init(nn.Linear(hidden, hidden)),
                make_activation(),
            )
        else:
            self.network = nn.Sequential(
                layer_init(nn.Linear(self.obs_dim, hidden)),
                make_activation(),
                layer_init(nn.Linear(hidden, hidden)),
                make_activation(),
            )
        self.actor = layer_init(nn.Linear(hidden, int(self.nvec.sum())), std=0.01)
        self.critic = layer_init(nn.Linear(hidden, 1), std=1.0)

    def _actor_hidden(self, x: torch.Tensor) -> torch.Tensor:
        """The trunk that feeds the policy head (shared trunk or actor tower)."""
        if self.arch == "towers":
            return self.actor_network(x)
        return self.network(x)

    def _critic_hidden(self, x: torch.Tensor) -> torch.Tensor:
        """The trunk that feeds the value head (shared trunk or critic tower)."""
        if self.arch == "towers":
            return self.critic_network(x)
        return self.network(x)

    def policy_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Raw (unmasked) action logits from the policy trunk and head."""
        return self.actor(self._actor_hidden(x))

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(self._critic_hidden(x))

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
        hidden = self._actor_hidden(x)
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
        # The shared architecture reuses the trunk it already computed for
        # the policy; towers computes its independent critic trunk here.
        value = (
            self.critic(hidden)
            if self.arch == "shared"
            else self.critic(self._critic_hidden(x))
        )
        return action.T, logprob, entropy, value


def save_agent(path: str | Path, agent: Agent, extra: dict[str, Any] | None = None) -> None:
    """Checkpoint the network so :class:`NeuralPolicy` / eval can rebuild it."""
    payload: dict[str, Any] = {
        "model": agent.state_dict(),
        "obs_dim": agent.obs_dim,
        "nvec": agent.nvec.tolist(),
        "hidden": agent.hidden,
        "activation": agent.activation,
        "obs_version": OBS_VERSION,
        "arch": getattr(agent, "arch", "shared"),
    }
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)


def load_agent(
    path: str | Path, device: str | torch.device = "cpu"
) -> tuple[Agent, dict[str, Any]]:
    """Rebuild an :class:`Agent` from :func:`save_agent`; returns ``(agent, extra)``.

    Only v5 checkpoints load.  A missing or different ``obs_version`` is a hard
    error: the dimension alone can never identify a foreign layout, so old
    payloads must be re-trained or explicitly migrated (see the ADR retiring
    v1-v4).
    """
    payload = torch.load(path, map_location=device, weights_only=True)
    version = payload.get("obs_version")
    if version != OBS_VERSION:
        raise ValueError(
            f"unsupported checkpoint obs_version {version!r}; expected "
            f"{OBS_VERSION} (v5)"
        )
    agent = Agent(
        payload["obs_dim"],
        payload["nvec"],
        payload["hidden"],
        activation=payload.get("activation", "relu"),
        # Checkpoints saved before the arch field are shared by construction.
        arch=payload.get("arch", "shared"),
    )
    agent.load_state_dict(payload["model"])
    return agent.to(device), payload.get("extra", {})


@dataclass(frozen=True, slots=True)
class WarmStart:
    """Outcome of :func:`warm_start_from`: what was copied and from where.

    ``exact`` is true when the checkpoint layout matched and the whole state
    dict was loaded directly (``copied`` is then empty); otherwise ``copied``
    holds the tensor labels copied by :func:`warm_start_into`, and ``arch`` /
    ``nvec`` describe the loaded checkpoint for the caller's log message.
    """

    copied: list[str]
    exact: bool
    arch: str
    nvec: list[int]


class WarmStartLayoutError(ValueError):
    """A checkpoint whose observation width does not match the agent's."""

    def __init__(self, loaded_obs_dim: int, agent_obs_dim: int) -> None:
        super().__init__(
            f"cannot warm-start a {loaded_obs_dim}-wide checkpoint into a "
            f"{agent_obs_dim}-wide agent: observation layouts must match "
            f"(v5 is the only supported layout)"
        )
        self.loaded_obs_dim = loaded_obs_dim
        self.agent_obs_dim = agent_obs_dim


def _mapped_source_key(agent: Agent, loaded: Agent, key: str) -> str:
    """Target state_dict key -> the key holding its weights in ``loaded``.

    Same architectures map to themselves.  Across architectures the shared
    trunk is the bridge: ``shared→towers`` copies it into both towers, while
    ``towers→shared`` keeps only the actor tower (see :func:`warm_start_into`).
    """
    agent_arch = getattr(agent, "arch", "shared")
    loaded_arch = getattr(loaded, "arch", "shared")
    if agent_arch == loaded_arch:
        return key
    if agent_arch == "towers" and key.startswith(("actor_network.", "critic_network.")):
        return "network." + key.split(".", 1)[1]
    if agent_arch == "shared" and key.startswith("network."):
        return "actor_network." + key.split(".", 1)[1]
    return key


def warm_start_into(agent: Agent, loaded: Agent) -> list[str]:
    """Copy compatible weights from ``loaded`` into ``agent`` in place.

    Used when the action heads changed (e.g. a 134-template checkpoint into the
    new 134+4 agent) or when the architecture changed: the trunk and critic are
    copied whole, and the actor's flattened head rows are copied by prefix (head
    order is stable).

    Across architectures the shared trunk bridges the layouts:
    ``shared→towers`` copies it into both towers (same observation means both
    heads keep the source function bit-for-bit), while ``towers→shared`` keeps
    the actor tower and drops the critic tower -- the policy stays bit-identical
    and the value head is re-adapted to the actor trunk, so value is *not*
    preserved in that direction.

    Both checkpoints must use the one live v5 layout; a different ``obs_dim``
    raises :class:`ValueError` rather than guessing a column mapping.
    """
    if loaded.obs_dim != agent.obs_dim:
        raise WarmStartLayoutError(loaded.obs_dim, agent.obs_dim)
    copied: list[str] = []
    source = loaded.state_dict()
    for key, value in agent.state_dict().items():
        origin_key = _mapped_source_key(agent, loaded, key)
        origin = source.get(origin_key)
        if origin is None:
            continue
        label = key if origin_key == key else f"{key}<-{origin_key}"
        if origin.shape == value.shape:
            value.copy_(origin)
            copied.append(label)
        elif key in {"actor.weight", "actor.bias"} and origin.shape[0] < value.shape[0]:
            value[: origin.shape[0]].copy_(origin)
            copied.append(f"{key}[:{origin.shape[0]}]")
    return copied


def warm_start_from(
    path: str | Path, agent: Agent, *, device: str = "cpu"
) -> WarmStart:
    """Load a same-layout checkpoint and warm-start ``agent`` in place.

    The checkpoint is rebuilt with :func:`load_agent`.  An ``obs_dim``
    mismatch raises :class:`WarmStartLayoutError` (a :class:`ValueError`) so
    callers can raise their own CLI error; callers that only need to log can
    instead print the returned :class:`WarmStart`.  An identical ``nvec`` /
    ``arch`` layout loads the whole state dict directly; otherwise
    :func:`warm_start_into` copies the shared trunk and heads and the copied
    labels are returned.
    """
    loaded, _ = load_agent(path, device=device)
    if loaded.obs_dim != agent.obs_dim:
        raise WarmStartLayoutError(loaded.obs_dim, agent.obs_dim)
    if loaded.nvec.tolist() == agent.nvec.tolist() and loaded.arch == agent.arch:
        agent.load_state_dict(loaded.state_dict())
        copied: list[str] = []
        exact = True
    else:
        copied = warm_start_into(agent, loaded)
        exact = False
    return WarmStart(
        copied=copied,
        exact=exact,
        arch=loaded.arch,
        nvec=loaded.nvec.tolist(),
    )


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
        # Preference heads (the suit head, ADR-0004) stay open; the shared
        # layout keeps this identical to what the environment publishes.
        mask = torch.tensor(
            [joint_mask_bits(view.mask, nvec)],
            dtype=torch.bool,
            device=self.device,
        )
        if not bool(mask[:, : nvec[0]].any()):
            raise ValueError("NeuralPolicy was asked to act with no legal action")
        obs = torch.as_tensor(
            encode_observation(view, self.rules),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        logits = self.agent.policy_logits(obs)
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

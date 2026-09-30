"""Torch actor-critic and the learned adapter of the ``Policy`` seam.

The rules core and :mod:`seven523.env` never import torch; this module is the
only place that does.  It needs the optional ``train`` dependency group:

    uv sync --group train
"""
from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from .actions import joint_mask_bits
from .env import OBS_VERSION, encode_observation
from .game import View
from .history import (
    EVENT_DIM,
    EVENT_ORDERINGS,
    PASS_MODES,
    WENT_OUT_MODES,
    encode_events,
    encode_history,
)
from .rules import DEFAULT_RULES, Rules

__all__ = [
    "Agent",
    "CategoricalMasked",
    "EventSequenceEncoder",
    "NeuralPolicy",
    "SequenceEncoder",
    "WarmStart",
    "WarmStartLayoutError",
    "history_layout",
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


class SequenceEncoder(nn.Module):
    """GRU encoder of the public play-history token sequence (D1-lite).

    ``0`` is the padding token; the feature is the hidden state at the last
    real card, so reading is order-aware and stateless (the whole sequence is
    re-read on every call).  Only built when ``Agent(seq_len=...) > 0``.
    """

    def __init__(self, emb: int = 16, hidden: int = 32, vocab: int = 55) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab, int(emb), padding_idx=0)
        self.rnn = nn.GRU(int(emb), int(hidden), batch_first=True)
        for name, param in self.rnn.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0.0)
            else:
                nn.init.orthogonal_(param, 1.0)
        self.hidden = int(hidden)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.dim() != 2:
            raise ValueError(f"expected (batch, length) tokens, got {tuple(tokens.shape)}")
        lengths = (tokens > 0).sum(dim=1).clamp(min=1)
        out, _ = self.rnn(self.embedding(tokens.long()))
        index = (lengths - 1).view(-1, 1, 1).expand(-1, 1, out.shape[-1])
        return out.gather(1, index).squeeze(1)


class EventSequenceEncoder(nn.Module):
    """Encoder of the event-level history (EVH): event MLP + seat + GRU.

    Each event is a 64-d vector (:func:`seven523.history.encode_events`); a
    shared two-layer MLP lifts it to ``emb``, a relative-seat embedding is
    concatenated (48-d GRU input) or added, and a GRU reads the sequence.  The
    feature is the hidden state at the last real event (``mask``-driven), so
    reading is stateless in exactly the D1-lite sense.

    ``seat_mode="none"`` keeps the GRU input width but zeroes the seat
    segment (the ``event_seatblind`` arm); ``seat_mode="sum"`` requires
    ``seat_emb == emb`` and adds instead of concatenating.
    """

    def __init__(
        self,
        dim: int = EVENT_DIM,
        emb: int = 32,
        hidden: int = 32,
        seat_emb: int = 16,
        num_players: int = 2,
        seat_mode: str = "concat",
    ) -> None:
        super().__init__()
        if seat_mode not in ("concat", "sum", "none"):
            raise ValueError(
                f"unknown event seat mode {seat_mode!r}; "
                "choose from ('concat', 'sum', 'none')"
            )
        self.dim = int(dim)
        self.emb = int(emb)
        self.hidden = int(hidden)
        self.seat_emb = int(seat_emb)
        self.num_players = int(num_players)
        self.seat_mode = str(seat_mode)
        self.mlp = nn.Sequential(
            layer_init(nn.Linear(self.dim, self.emb)),
            nn.ReLU(),
            layer_init(nn.Linear(self.emb, self.emb)),
        )
        if self.seat_mode == "sum":
            if self.seat_emb != self.emb:
                raise ValueError("event_seat='sum' requires seat_emb == event_emb")
            rnn_input = self.emb
        else:
            rnn_input = self.emb + self.seat_emb
        self.seat = nn.Embedding(
            self.num_players + 1, self.emb if self.seat_mode == "sum" else self.seat_emb,
            padding_idx=self.num_players,
        )
        self.rnn = nn.GRU(rnn_input, self.hidden, batch_first=True)
        for name, param in self.rnn.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0.0)
            else:
                nn.init.orthogonal_(param, 1.0)

    def forward(
        self, events: torch.Tensor, seats: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        x = self.mlp(events)
        seat_vec = self.seat(seats.long())
        if self.seat_mode == "none":
            seat_vec = torch.zeros_like(seat_vec)
            x = torch.cat([x, seat_vec], dim=-1)
        elif self.seat_mode == "sum":
            x = x + seat_vec
        else:
            x = torch.cat([x, seat_vec], dim=-1)
        lengths = mask.sum(dim=1)
        # Trailing pad events are all-zero and the readout is gathered at each
        # row's last real event, so the GRU may stop at the batch-wide maximum
        # (no causal step after it can influence an earlier gather).
        keep = max(int(lengths.max().item()) if lengths.numel() else 0, 1)
        x = x[:, :keep]
        lengths = lengths.clamp(min=1)
        out, _ = self.rnn(x)
        index = (lengths - 1).view(-1, 1, 1).expand(-1, 1, out.shape[-1])
        return out.gather(1, index).squeeze(1)


class Agent(nn.Module):
    """MLP actor-critic with one masked categorical head per action dimension.

    The environment's action space is ``MultiDiscrete([134, 4])``: head 0 is
    the rank template, head 1 the top-card suit (ADR-0004).  The ``split``
    machinery keeps this generic for any future ``nvec``.

    ``seq_len > 0`` opts into the D1-lite sequence memory: the trunk receives
    ``concat(obs, SequenceEncoder(public history))`` instead of ``obs``.  The
    default ``seq_len=0`` builds exactly the historical MLP -- same modules,
    same parameter count, same forward numerics.
    """

    def __init__(
        self,
        obs_dim: int,
        nvec: Sequence[int],
        hidden: int = 128,
        activation: str = "relu",
        arch: str = "shared",
        seq_len: int = 0,
        seq_emb: int = 16,
        seq_hidden: int = 32,
        seq_blind: bool = False,
        seq_order: str = "chrono",
        event_len: int = 0,
        event_dim: int = EVENT_DIM,
        event_emb: int = 32,
        event_hidden: int = 32,
        seat_emb: int = 16,
        event_blind: bool = False,
        event_seat: str = "concat",
        event_order: str = "chrono",
        event_pass: str = "keep",
        event_boundary_blind: bool = False,
        event_noisy: bool = False,
        event_went_out: str = "keep",
        num_players: int = 2,
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
        self.seq_len = int(seq_len)
        self.seq_emb = int(seq_emb)
        self.seq_hidden = int(seq_hidden)
        #: Blind control: keep the encoder and the wider trunk but feed an
        #: all-pad sequence, separating encoder capacity from sequence info.
        self.seq_blind = bool(seq_blind)
        #: Token layout of the history input: ``chrono`` (play order) or
        #: ``sorted`` (same multiset, canonical order).  The eval path must
        #: rebuild the sequence exactly like the training rollout did.
        self.seq_order = str(seq_order)
        #: Event-level history (EVH) config; ``event_len=0`` keeps the
        #: historical MLP (or D1-lite seq) path exactly.
        self.event_len = int(event_len)
        self.event_dim = int(event_dim)
        self.event_emb = int(event_emb)
        self.event_hidden = int(event_hidden)
        self.seat_emb = int(seat_emb)
        self.event_blind = bool(event_blind)
        self.event_seat = str(event_seat)
        self.event_order = str(event_order)
        self.event_pass = str(event_pass)
        self.event_boundary_blind = bool(event_boundary_blind)
        self.event_noisy = bool(event_noisy)
        self.event_went_out = str(event_went_out)
        self.num_players = int(num_players)
        if self.event_order not in EVENT_ORDERINGS:
            raise ValueError(
                f"unknown event order {self.event_order!r}; "
                f"choose from {EVENT_ORDERINGS}"
            )
        if self.event_pass not in PASS_MODES:
            raise ValueError(
                f"unknown event pass mode {self.event_pass!r}; "
                f"choose from {PASS_MODES}"
            )
        if self.event_went_out not in WENT_OUT_MODES:
            raise ValueError(
                f"unknown event went-out mode {self.event_went_out!r}; "
                f"choose from {WENT_OUT_MODES}"
            )
        if self.event_blind and self.event_noisy:
            raise ValueError("event_blind and event_noisy are mutually exclusive")
        if self.seq_len > 0:
            self.seq_encoder: SequenceEncoder | None = SequenceEncoder(
                emb=self.seq_emb, hidden=self.seq_hidden
            )
        else:
            self.seq_encoder = None
        if self.event_len > 0:
            self.event_encoder: EventSequenceEncoder | None = EventSequenceEncoder(
                dim=self.event_dim,
                emb=self.event_emb,
                hidden=self.event_hidden,
                seat_emb=self.seat_emb,
                num_players=self.num_players,
                seat_mode=self.event_seat,
            )
        else:
            self.event_encoder = None
        trunk_input = self.obs_dim
        if self.seq_encoder is not None:
            trunk_input += self.seq_hidden
        if self.event_encoder is not None:
            trunk_input += self.event_hidden
        make_activation = _ACTIVATIONS[activation]
        if arch == "towers":
            # No cross-connections: the policy and value losses never share a
            # hidden parameter, so their gradients cannot interfere.
            self.actor_network = nn.Sequential(
                layer_init(nn.Linear(trunk_input, hidden)),
                make_activation(),
                layer_init(nn.Linear(hidden, hidden)),
                make_activation(),
            )
            self.critic_network = nn.Sequential(
                layer_init(nn.Linear(trunk_input, hidden)),
                make_activation(),
                layer_init(nn.Linear(hidden, hidden)),
                make_activation(),
            )
        else:
            self.network = nn.Sequential(
                layer_init(nn.Linear(trunk_input, hidden)),
                make_activation(),
                layer_init(nn.Linear(hidden, hidden)),
                make_activation(),
            )
        self.actor = layer_init(nn.Linear(hidden, int(self.nvec.sum())), std=0.01)
        self.critic = layer_init(nn.Linear(hidden, 1), std=1.0)

    def _trunk_input(
        self,
        x: torch.Tensor,
        seqs: torch.Tensor | None,
        events: torch.Tensor | None = None,
        event_seats: torch.Tensor | None = None,
        event_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``x`` plus the optional second inputs (seq tokens and/or events)."""
        parts = [x]
        if self.seq_encoder is None:
            if seqs is not None:
                raise ValueError("agent has no sequence encoder but seqs were passed")
        else:
            if seqs is None:
                raise ValueError("agent has a sequence encoder but no seqs were passed")
            parts.append(self.seq_encoder(seqs))
        if self.event_encoder is None:
            if events is not None:
                raise ValueError("agent has no event encoder but events were passed")
        else:
            if events is None:
                raise ValueError("agent has an event encoder but no events were passed")
            if event_seats is None or event_mask is None:
                raise ValueError("events require event_seats and event_mask")
            parts.append(self.event_encoder(events, event_seats, event_mask))
        if len(parts) == 1:
            return x
        return torch.cat(parts, dim=-1)

    def _actor_hidden(
        self,
        x: torch.Tensor,
        seqs: torch.Tensor | None = None,
        events: torch.Tensor | None = None,
        event_seats: torch.Tensor | None = None,
        event_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """The trunk that feeds the policy head (shared trunk or actor tower)."""
        trunk = self._trunk_input(x, seqs, events, event_seats, event_mask)
        if self.arch == "towers":
            return self.actor_network(trunk)
        return self.network(trunk)

    def _critic_hidden(
        self,
        x: torch.Tensor,
        seqs: torch.Tensor | None = None,
        events: torch.Tensor | None = None,
        event_seats: torch.Tensor | None = None,
        event_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """The trunk that feeds the value head (shared trunk or critic tower)."""
        trunk = self._trunk_input(x, seqs, events, event_seats, event_mask)
        if self.arch == "towers":
            return self.critic_network(trunk)
        return self.network(trunk)

    def policy_logits(
        self,
        x: torch.Tensor,
        seqs: torch.Tensor | None = None,
        events: torch.Tensor | None = None,
        event_seats: torch.Tensor | None = None,
        event_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Raw (unmasked) action logits from the policy trunk and head."""
        return self.actor(
            self._actor_hidden(x, seqs, events, event_seats, event_mask)
        )

    def get_value(
        self,
        x: torch.Tensor,
        seqs: torch.Tensor | None = None,
        events: torch.Tensor | None = None,
        event_seats: torch.Tensor | None = None,
        event_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.critic(
            self._critic_hidden(x, seqs, events, event_seats, event_mask)
        )

    def get_action_and_value(
        self,
        x: torch.Tensor,
        action_mask: torch.Tensor,
        action: torch.Tensor | None = None,
        seqs: torch.Tensor | None = None,
        events: torch.Tensor | None = None,
        event_seats: torch.Tensor | None = None,
        event_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(action, logprob, entropy, value)``.

        ``action`` is ``(num_heads, batch)`` when given (the training path);
        the sampled action is returned as ``(batch, num_heads)``, matching the
        reference script.
        """
        hidden = self._actor_hidden(x, seqs, events, event_seats, event_mask)
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
            else self.critic(
                self._critic_hidden(x, seqs, events, event_seats, event_mask)
            )
        )
        return action.T, logprob, entropy, value


def history_layout(agent: Agent) -> str:
    """Identity of the agent's second-input layout (checkpoint guard).

    ``obs_dim`` alone cannot tell an event arm (trunk input 193) from an MLP
    arm (161), so warm starts and refreshes compare this string instead.
    """
    parts: list[str] = []
    seq_len = int(getattr(agent, "seq_len", 0))
    if seq_len > 0:
        parts.append(
            f"seq:{seq_len}:{agent.seq_emb}:{agent.seq_hidden}:"
            f"{int(bool(agent.seq_blind))}:{agent.seq_order}"
        )
    event_len = int(getattr(agent, "event_len", 0))
    if event_len > 0:
        parts.append(
            f"event:{event_len}:{agent.event_dim}:{agent.event_emb}:"
            f"{agent.event_hidden}:{agent.seat_emb}:{agent.event_seat}:"
            f"{int(bool(agent.event_blind))}:{agent.event_order}:"
            f"{agent.event_pass}:{int(bool(agent.event_boundary_blind))}:"
            f"{int(bool(agent.event_noisy))}:{agent.event_went_out}:"
            f"{agent.num_players}"
        )
    return "+".join(parts) if parts else "mlp"


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
        "seq_len": getattr(agent, "seq_len", 0),
        "seq_emb": getattr(agent, "seq_emb", 16),
        "seq_hidden": getattr(agent, "seq_hidden", 32),
        "seq_blind": getattr(agent, "seq_blind", False),
        "seq_order": getattr(agent, "seq_order", "chrono"),
        "event_len": getattr(agent, "event_len", 0),
        "event_dim": getattr(agent, "event_dim", EVENT_DIM),
        "event_emb": getattr(agent, "event_emb", 32),
        "event_hidden": getattr(agent, "event_hidden", 32),
        "seat_emb": getattr(agent, "seat_emb", 16),
        "event_blind": getattr(agent, "event_blind", False),
        "event_seat": getattr(agent, "event_seat", "concat"),
        "event_order": getattr(agent, "event_order", "chrono"),
        "event_pass": getattr(agent, "event_pass", "keep"),
        "event_boundary_blind": getattr(agent, "event_boundary_blind", False),
        "event_noisy": getattr(agent, "event_noisy", False),
        "event_went_out": getattr(agent, "event_went_out", "keep"),
        "num_players": getattr(agent, "num_players", 2),
        "history_layout": history_layout(agent),
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
        # Checkpoints saved before the sequence fields have no encoder.
        seq_len=payload.get("seq_len", 0),
        seq_emb=payload.get("seq_emb", 16),
        seq_hidden=payload.get("seq_hidden", 32),
        seq_blind=payload.get("seq_blind", False),
        seq_order=payload.get("seq_order", "chrono"),
        # Checkpoints saved before the event fields have no event encoder.
        event_len=payload.get("event_len", 0),
        event_dim=payload.get("event_dim", EVENT_DIM),
        event_emb=payload.get("event_emb", 32),
        event_hidden=payload.get("event_hidden", 32),
        seat_emb=payload.get("seat_emb", 16),
        event_blind=payload.get("event_blind", False),
        event_seat=payload.get("event_seat", "concat"),
        event_order=payload.get("event_order", "chrono"),
        event_pass=payload.get("event_pass", "keep"),
        event_boundary_blind=payload.get("event_boundary_blind", False),
        event_noisy=payload.get("event_noisy", False),
        event_went_out=payload.get("event_went_out", "keep"),
        num_players=payload.get("num_players", 2),
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
    """A checkpoint whose observation or second-input layout does not match."""

    def __init__(
        self,
        loaded_obs_dim: int,
        agent_obs_dim: int,
        *,
        loaded_layout: str | None = None,
        agent_layout: str | None = None,
    ) -> None:
        self.loaded_obs_dim = loaded_obs_dim
        self.agent_obs_dim = agent_obs_dim
        self.loaded_layout = loaded_layout
        self.agent_layout = agent_layout
        if loaded_layout is not None and loaded_layout != agent_layout:
            super().__init__(
                f"cannot warm-start a checkpoint with history layout "
                f"{loaded_layout!r} into an agent with layout {agent_layout!r}: "
                f"second-input layouts must match (the event arm needs its own "
                f"from-scratch run)"
            )
        else:
            super().__init__(
                f"cannot warm-start a {loaded_obs_dim}-wide checkpoint into a "
                f"{agent_obs_dim}-wide agent: observation layouts must match "
                f"(v5 is the only supported layout)"
            )


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
    if history_layout(loaded) != history_layout(agent):
        raise WarmStartLayoutError(
            loaded.obs_dim,
            agent.obs_dim,
            loaded_layout=history_layout(loaded),
            agent_layout=history_layout(agent),
        )
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
    if history_layout(loaded) != history_layout(agent):
        raise WarmStartLayoutError(
            loaded.obs_dim,
            agent.obs_dim,
            loaded_layout=history_layout(loaded),
            agent_layout=history_layout(agent),
        )
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
    """Inference adapter implementing the ``Policy`` protocol (argmax by default).

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
        seqs = None
        if getattr(self.agent, "seq_len", 0) > 0:
            length = int(self.agent.seq_len)
            if getattr(self.agent, "seq_blind", False):
                tokens = np.zeros(length, dtype=np.int64)
            else:
                tokens = encode_history(
                    view, length, getattr(self.agent, "seq_order", "chrono")
                )
            seqs = torch.as_tensor(tokens, dtype=torch.int64, device=self.device).unsqueeze(0)
        events = event_seats = event_mask = None
        if getattr(self.agent, "event_len", 0) > 0:
            event_values, seat_values, mask_values = encode_events(
                view,
                self.rules,
                int(self.agent.event_len),
                order=getattr(self.agent, "event_order", "chrono"),
                pass_mode=getattr(self.agent, "event_pass", "keep"),
                boundary_blind=getattr(self.agent, "event_boundary_blind", False),
                went_out=getattr(self.agent, "event_went_out", "keep"),
                blind=getattr(self.agent, "event_blind", False),
                noisy=getattr(self.agent, "event_noisy", False),
            )
            events = torch.as_tensor(
                event_values, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            event_seats = torch.as_tensor(
                seat_values, dtype=torch.int64, device=self.device
            ).unsqueeze(0)
            event_mask = torch.as_tensor(
                mask_values, dtype=torch.bool, device=self.device
            ).unsqueeze(0)
        logits = self.agent.policy_logits(
            obs, seqs, events, event_seats, event_mask
        )
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

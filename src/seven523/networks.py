"""Torch actor-critic and the learned adapter of the ``Policy`` seam.

The rules core and :mod:`seven523.env` never import torch; this module is the
only place that does.  It needs the optional ``train`` dependency group:

    uv sync --group train
"""
from __future__ import annotations

import math
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

#: Trunk topologies.  ``shared`` = one trunk feeding both heads (ADR-0003
#: source); ``towers`` = independent actor/critic trunks (the reference
#: ``ppo.py`` layout); ``ln`` = shared + LayerNorm before each hidden
#: activation; ``deep`` = shared with a third hidden layer; ``deep_ln`` =
#: both (the O2 depth/normalization factorial, docs/depth-normalization-plan.md);
#: ``deep_res``/``deep_lnres`` = ``deep``/``deep_ln`` plus residual skips on
#: blocks 2-3 (O2 Amendment A1, same parameter counts as their plain twins).
#: ``head<a><c>`` = the shared trunk plus a-head/critic-head depth grid
#: (docs/head-depth-plan.md): a = actor-head Linear layers, c = critic-head
#: Linear layers, both in {1,2,3}; ``shared`` is the (1,1) cell.  The ``ln``
#: suffix (``head<a><c>ln``) inserts LayerNorm inside every residual head
#: block (head design v2, plan section 10).  ``lnres<a><c>``/``gnres<a><c>``/
#: ``bnres<a><c>`` move the residual into a 3-block body normalized with
#: LayerNorm / GroupNorm / BatchNorm and give the actor/critic non-residual
#: (plain) ``<a>``/``<c>`` layer heads (docs/gnres-plan.md); ``gnres11`` is
#: the new GroupNorm body with the historical single-layer heads.
#: ``res<d><a><c>`` = no-norm residual body with ``d`` blocks plus the same
#: plain heads (docs/experiments/res4-plan.md); ``res4`` = 4 blocks.
#: ``bres<d><a><c>`` = the same body and heads but every residual block is a
#: bottleneck ``hidden -> mid -> hidden`` (``mid`` = ``--res-expansion``
#: times ``hidden``, default 3): ``bres421`` at hidden 32 is the
#: ``(32, 96, 32)`` block, ``bres221`` is the same block at depth 2.
_ARCHITECTURES = (
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
    "lnres21",
    "lnres31",
    "lnres12",
    "lnres22",
    "lnres32",
    "lnres13",
    "lnres23",
    "lnres33",
    "gnres11",
    "gnres21",
    "gnres31",
    "gnres12",
    "gnres22",
    "gnres32",
    "gnres13",
    "gnres23",
    "gnres33",
    "bnres11",
    "bnres21",
    "bnres31",
    "bnres12",
    "bnres22",
    "bnres32",
    "bnres13",
    "bnres23",
    "bnres33",
    "res411",
    "res421",
    "res431",
    "res412",
    "res422",
    "res432",
    "res413",
    "res423",
    "res433",
    "bres411",
    "bres421",
    "bres431",
    "bres412",
    "bres422",
    "bres432",
    "bres413",
    "bres423",
    "bres433",
    "bres211",
    "bres221",
    "bres231",
    "bres212",
    "bres222",
    "bres232",
    "bres213",
    "bres223",
    "bres233",
)


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


#: Normalization options of :class:`ResidualTrunk` (`bnres` adds ``batch``).
_TRUNK_NORMS = ("none", "layer", "group", "batch")


class ResidualTrunk(nn.Module):
    """O2 Amendment A1: ``blocks``-block trunk with residual skips.

    Identical module sequence and parameter count to ``deep`` (``norm='none'``)
    or ``deep_ln`` (``norm='layer'``) when ``blocks=3``; the only change is
    that every block after the first adds its output to the running hidden
    state (``h = h + block(h)``) instead of replacing it, so the comparison
    isolates the skip connection.  ``res<d><a><c>`` uses ``blocks=d`` with no
    normalization (docs/experiments/res4-plan.md).  ``mid`` turns every
    residual block into a bottleneck ``hidden -> mid -> hidden`` (the block
    input and output stay ``hidden``, so the skip is still exact); ``None``
    keeps the historical single hidden projection.

    ``norm`` picks the per-block normalization (``none``/``layer``/
    ``group``/``batch``; ``group`` needs ``gn_groups``); the block topology
    (norm between Linear and activation, on every block) is unchanged, so the
    normalization type is a single-factor comparison.  ``batch`` registers a
    :class:`nn.BatchNorm1d` whose running statistics are part of the
    checkpoint and must be used in eval mode (``NeuralPolicy`` does).
    """

    def __init__(
        self,
        in_dim: int,
        hidden: int,
        activation: str,
        *,
        norm: str = "none",
        gn_groups: int | None = None,
        blocks: int = 3,
        mid: int | None = None,
    ) -> None:
        super().__init__()
        if norm not in _TRUNK_NORMS:
            raise ValueError(
                f"unknown trunk norm {norm!r}; choose from {_TRUNK_NORMS}"
            )
        if gn_groups is not None and norm != "group":
            raise ValueError("gn_groups is only valid with norm='group'")
        if norm == "group" and gn_groups is None:
            raise ValueError("norm='group' requires gn_groups")
        if int(blocks) < 1:
            raise ValueError(f"blocks must be >= 1, got {blocks!r}")
        if mid is not None and int(mid) < 1:
            raise ValueError(f"mid must be >= 1 when set, got {mid!r}")
        make_activation = _ACTIVATIONS[activation]
        self.blocks = nn.ModuleList()
        for index in range(int(blocks)):
            # The bottleneck (``mid`` set) widens every residual block to
            # ``hidden -> mid -> hidden``; the input projection (block 0) and
            # the skip widths are unchanged.
            widths = (
                [(in_dim if index == 0 else hidden, hidden)]
                if mid is None or index == 0
                else [(hidden, int(mid)), (int(mid), hidden)]
            )
            modules: list[nn.Module] = []
            for fan_in, fan_out in widths:
                modules.append(layer_init(nn.Linear(fan_in, fan_out)))
                if norm == "layer":
                    modules.append(nn.LayerNorm(fan_out))
                elif norm == "group":
                    modules.append(nn.GroupNorm(gn_groups, fan_out))
                elif norm == "batch":
                    modules.append(nn.BatchNorm1d(fan_out))
                modules.append(make_activation())
            self.blocks.append(nn.Sequential(*modules))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.blocks[0](x)
        for block in self.blocks[1:]:
            h = h + block(h)
        return h


def _head_layers(arch: str) -> tuple[int, int]:
    """(actor, critic) head Linear layers for the grid archs."""
    if arch.startswith("head"):
        return int(arch[4]), int(arch[5])
    if arch.startswith(("lnres", "gnres", "bnres", "bres")):
        return int(arch[5]), int(arch[6])
    if arch.startswith("res"):  # res<depth><actor><critic>, e.g. res421
        return int(arch[4]), int(arch[5])
    return 1, 1


def _head_ln(arch: str) -> bool:
    """Whether a grid arch uses LayerNorm inside its residual head blocks."""
    return arch.startswith("head") and arch.endswith("ln")


class PlainHead(nn.Module):
    """Non-residual multi-layer actor/critic head (docs/gnres-plan.md).

    The ``lnres``/``gnres`` families put the skip connections and the
    normalization in the shared body; the heads stay plain: ``layers - 1``
    blocks of ``x = act(Linear(x))`` followed by the output projection, no
    skip and no norm.  ``layers == 1`` is built as the historical plain
    ``Linear`` by :class:`Agent` (same as :class:`ResidualHead`), so the
    family's parameter counts match ``head<a><c>`` plus the body difference.
    """

    def __init__(
        self,
        hidden: int,
        out_dim: int,
        out_std: float,
        activation: str,
        layers: int,
    ) -> None:
        super().__init__()
        make_activation = _ACTIVATIONS[activation]
        blocks: list[nn.Module] = []
        for _ in range(layers - 1):
            blocks.append(layer_init(nn.Linear(hidden, hidden)))
            blocks.append(make_activation())
        self.body = nn.Sequential(*blocks)
        self.out = layer_init(nn.Linear(hidden, out_dim), std=out_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.body(x))


class ResidualHead(nn.Module):
    """Residual multi-layer actor/critic head (docs/head-depth-plan.md).

    ``layers`` Linear layers: ``layers - 1`` residual hidden blocks
    (``h = h + act(Linear(h))``, or ``h = h + act(Linear(LayerNorm(h)))``
    when ``ln`` is set) followed by the output projection.  The ``ln`` variant
    applies the norm to the **block input** (pre-norm) so the residual stream
    variance stays stable with depth.  Residual is the default head design; a
    one-layer head is the historical plain ``Linear`` (built directly by
    :class:`Agent`, no skip is possible).
    """

    def __init__(
        self,
        hidden: int,
        out_dim: int,
        out_std: float,
        activation: str,
        layers: int,
        *,
        ln: bool = False,
    ) -> None:
        super().__init__()
        make_activation = _ACTIVATIONS[activation]
        blocks: list[nn.Module] = []
        for _ in range(layers - 1):
            modules: list[nn.Module] = []
            if ln:
                modules.append(nn.LayerNorm(hidden))
            modules.append(layer_init(nn.Linear(hidden, hidden)))
            modules.append(make_activation())
            blocks.append(nn.Sequential(*modules))
        self.blocks = nn.ModuleList(blocks)
        self.out = layer_init(nn.Linear(hidden, out_dim), std=out_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = x + block(x)
        return self.out(x)


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
        actor_out_std: float = 0.01,
        gn_groups: int = 8,
        res_expansion: float = 3.0,
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
        vf_nll: bool = False,
        vf_sample: bool = False,
        vf_outcome: bool = False,
    ) -> None:
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"unknown activation {activation!r}; "
                f"choose from {sorted(_ACTIVATIONS)}"
            )
        if not math.isfinite(actor_out_std) or actor_out_std <= 0.0:
            raise ValueError(
                f"actor_out_std must be finite and > 0, got {actor_out_std!r}"
            )
        if arch not in _ARCHITECTURES:
            raise ValueError(
                f"unknown arch {arch!r}; choose from {sorted(_ARCHITECTURES)}"
            )
        if int(gn_groups) < 1:
            raise ValueError(f"gn_groups must be >= 1, got {gn_groups!r}")
        if not math.isfinite(float(res_expansion)) or float(res_expansion) <= 0.0:
            raise ValueError(
                f"res_expansion must be finite and > 0, got {res_expansion!r}"
            )
        if arch.startswith("bres"):
            mid_width = hidden * float(res_expansion)
            if round(mid_width) < 1 or abs(mid_width - round(mid_width)) > 1e-9:
                raise ValueError(
                    "hidden * res_expansion must be a positive integer, got "
                    f"{hidden} * {res_expansion} = {mid_width}"
                )
        if arch.startswith("gnres") and hidden % int(gn_groups) != 0:
            raise ValueError(
                f"gn_groups={gn_groups} must divide hidden={hidden} for arch "
                f"{arch!r}"
            )
        self.obs_dim = int(obs_dim)
        self.nvec = torch.as_tensor(list(nvec), dtype=torch.long)
        self.hidden = int(hidden)
        self.activation = activation
        #: Trunk topology: the shared checkpoint keys (``network.*``,
        #: ``actor.*``, ``critic.*``) stay unchanged by design.
        self.arch = arch
        #: Number of groups of the ``gnres`` body's GroupNorm (unused by every
        #: other arch, but stored so a warm start cannot silently switch it).
        self.gn_groups = int(gn_groups)
        #: Bottleneck multiplier of the ``bres`` body's residual blocks (unused
        #: by every other arch, but stored so a warm start cannot silently
        #: switch it).  ``mid = hidden * res_expansion``; values < 1 are a
        #: compression bottleneck (e.g. 0.25 at hidden 64 = (64, 16, 64)).
        self.res_expansion = float(res_expansion)
        #: Amendment A2: heteroscedastic critic -> (mean, logvar) head trained
        #: with the Gaussian NLL (see docs/depth-normalization-plan.md §11).
        self.vf_nll = bool(vf_nll)
        #: Amendment A3: draw the rollout value from N(mean, var) instead of
        #: using the mean (Thompson-style value sampling).
        if vf_sample and not vf_nll:
            raise ValueError(
                "vf_sample requires vf_nll: there is no variance head to sample from"
            )
        self.vf_sample = bool(vf_sample)
        #: Outcome-decomposed value head (docs/experiments/vf-outcome.md): a
        #: second scalar head predicts the discounted win/loss term and the
        #: critic is trained on the jump-free return.  Kept separate from
        #: ``vf_nll`` until the two value layouts are composed deliberately.
        if vf_outcome and vf_nll:
            raise ValueError(
                "vf_outcome and vf_nll are mutually exclusive: the outcome "
                "head decomposition is only defined for the mean-only critic"
            )
        self.vf_outcome = bool(vf_outcome)
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
        elif arch in {"deep_res", "deep_lnres"}:
            # O2 Amendment A1: deep/deep_ln plus residual skips on blocks 2-3.
            self.network = ResidualTrunk(
                trunk_input,
                hidden,
                activation,
                norm="layer" if arch == "deep_lnres" else "none",
            )
        elif arch.startswith(("lnres", "gnres", "bnres")):
            # docs/gnres-plan.md: the deep_res/deep_lnres body topology with
            # the normalization switched to LayerNorm/GroupNorm/BatchNorm and
            # the heads moved to plain multi-layer MLPs (PlainHead below).
            family = arch[:5]
            norm = {"lnres": "layer", "gnres": "group", "bnres": "batch"}[
                family
            ]
            self.network = ResidualTrunk(
                trunk_input,
                hidden,
                activation,
                norm=norm,
                gn_groups=self.gn_groups if norm == "group" else None,
            )
        elif arch.startswith("bres"):
            # docs/experiments/bres-2m.md: the res4 body with every residual
            # block turned into a ``hidden -> mid -> hidden`` bottleneck; the
            # block count is arch[4] (bres<depth><actor><critic>).
            self.network = ResidualTrunk(
                trunk_input,
                hidden,
                activation,
                norm="none",
                blocks=int(arch[4]),
                mid=round(hidden * self.res_expansion),
            )
        elif arch.startswith("res"):
            # docs/experiments/res4-plan.md: no-norm residual body with the
            # block count encoded in the arch (res<depth><actor><critic>),
            # plus plain (non-residual) heads.  ``res4...`` = depth 4.
            self.network = ResidualTrunk(
                trunk_input,
                hidden,
                activation,
                norm="none",
                blocks=int(arch[3]),
            )
        else:
            # O2 factorial (docs/depth-normalization-plan.md): ``deep`` adds a
            # third hidden Linear; ``ln``/``deep_ln`` insert LayerNorm between
            # each hidden Linear and its activation, including the last one
            # (i.e. on the heads' input side).  ``shared`` builds exactly the
            # historical module sequence and consumes the same ``layer_init``
            # draws, so the default path stays bit-identical.
            layers: list[nn.Module] = []
            width = trunk_input
            for _ in range(3 if arch in {"deep", "deep_ln"} else 2):
                layers.append(layer_init(nn.Linear(width, hidden)))
                if arch in {"ln", "deep_ln"}:
                    layers.append(nn.LayerNorm(hidden))
                layers.append(make_activation())
                width = hidden
            self.network = nn.Sequential(*layers)
        # Head-depth grid (docs/head-depth-plan.md): ``head<a><c>`` gives the
        # actor/critic heads ``a``/``c`` Linear layers, and the ``ln`` suffix
        # adds pre-norm LayerNorm inside every residual head block.  Every
        # multi-layer ``head`` head is residual: ``layers - 1`` blocks of
        # ``h = h + act(Linear(h))`` followed by the output projection.  The
        # ``lnres``/``gnres`` families (docs/gnres-plan.md) move the residual
        # into a normalized body and build ``a``/``c``-layer *plain* heads
        # instead.  A one-layer head stays the historical plain Linear in
        # every family, so ``shared`` and every pre-grid arch keep the same
        # module types and init draws.  ``actor_out_std`` scales the actor
        # output init (default 0.01 keeps history; deep actor heads need a
        # larger gain to avoid starving their hidden layers).
        actor_layers, critic_layers = _head_layers(arch)
        head_ln = _head_ln(arch)
        plain_heads = arch.startswith(
            ("lnres", "gnres", "bnres", "res", "bres")
        )

        def make_head(out_dim: int, out_std: float, layers: int) -> nn.Module:
            if layers == 1:
                return layer_init(nn.Linear(hidden, out_dim), std=out_std)
            if plain_heads:
                return PlainHead(hidden, out_dim, out_std, activation, layers)
            return ResidualHead(
                hidden, out_dim, out_std, activation, layers, ln=head_ln
            )

        self.actor = make_head(int(self.nvec.sum()), actor_out_std, actor_layers)
        if self.vf_nll:
            # (mean, logvar); logvar starts at 0 so the initial variance is 1.
            self.critic = make_head(2, 1.0, critic_layers)
            final = (
                self.critic.out
                if isinstance(self.critic, (ResidualHead, PlainHead))
                else self.critic
            )
            nn.init.zeros_(final.weight[1])
            nn.init.zeros_(final.bias[1])
        else:
            self.critic = make_head(1, 1.0, critic_layers)
        if self.vf_outcome:
            # Scalar regression of the discounted win/loss outcome
            # (E[gamma**(T-t) * seat_outcome]); small init keeps the initial
            # prediction near 0 so the tanh does not saturate.
            self.outcome_head = layer_init(nn.Linear(hidden, 1), std=0.01)

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
        out = self.critic(
            self._critic_hidden(x, seqs, events, event_seats, event_mask)
        )
        return out[:, :1] if self.vf_nll else out

    def get_value_dist(
        self,
        x: torch.Tensor,
        seqs: torch.Tensor | None = None,
        events: torch.Tensor | None = None,
        event_seats: torch.Tensor | None = None,
        event_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(mean, logvar)`` from the heteroscedastic critic."""
        if not self.vf_nll:
            raise ValueError(
                "agent was built without vf_nll; the critic has no variance head"
            )
        out = self.critic(
            self._critic_hidden(x, seqs, events, event_seats, event_mask)
        )
        return out[:, 0], out[:, 1]

    def get_outcome(
        self,
        x: torch.Tensor,
        seqs: torch.Tensor | None = None,
        events: torch.Tensor | None = None,
        event_seats: torch.Tensor | None = None,
        event_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``tanh`` outcome prediction ``E[gamma**(T-t) * seat_outcome]``.

        The discount is part of the target (the jump is received at the
        terminal step), so the head estimates the jump's contribution to the
        value at the current state; its sign is the predicted win/loss.
        """
        if not self.vf_outcome:
            raise ValueError(
                "agent was built without vf_outcome; the critic has no outcome head"
            )
        hidden = self._critic_hidden(x, seqs, events, event_seats, event_mask)
        return torch.tanh(self.outcome_head(hidden))

    def get_value_and_outcome(
        self,
        x: torch.Tensor,
        seqs: torch.Tensor | None = None,
        events: torch.Tensor | None = None,
        event_seats: torch.Tensor | None = None,
        event_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One trunk pass returning ``(value, outcome)`` for the trainer."""
        if not self.vf_outcome:
            raise ValueError(
                "agent was built without vf_outcome; the critic has no outcome head"
            )
        hidden = self._critic_hidden(x, seqs, events, event_seats, event_mask)
        return self.critic(hidden), torch.tanh(self.outcome_head(hidden))

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
        # The shared trunk (also the new O2 layouts) reuses the hidden state
        # it already computed for the policy; towers computes its independent
        # critic trunk here.
        value = (
            self.critic(hidden)
            if self.arch != "towers"
            else self.critic(
                self._critic_hidden(x, seqs, events, event_seats, event_mask)
            )
        )
        if self.vf_nll:
            mean = value[:, 0]
            if self.vf_sample:
                logvar = value[:, 1].clamp(-10.0, 10.0)
                value = (
                    mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)
                ).unsqueeze(-1)
            else:
                value = value[:, :1]
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
        "vf_nll": getattr(agent, "vf_nll", False),
        "vf_sample": getattr(agent, "vf_sample", False),
        "vf_outcome": getattr(agent, "vf_outcome", False),
        "gn_groups": getattr(agent, "gn_groups", 8),
        "res_expansion": getattr(agent, "res_expansion", 3),
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
        vf_nll=payload.get("vf_nll", False),
        vf_sample=payload.get("vf_sample", False),
        vf_outcome=payload.get("vf_outcome", False),
        gn_groups=payload.get("gn_groups", 8),
        res_expansion=payload.get("res_expansion", 3),
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
    """A checkpoint whose arch, observation, or second-input layout mismatches."""

    def __init__(
        self,
        loaded_obs_dim: int,
        agent_obs_dim: int,
        *,
        loaded_layout: str | None = None,
        agent_layout: str | None = None,
        loaded_arch: str | None = None,
        agent_arch: str | None = None,
        reason: str | None = None,
    ) -> None:
        self.loaded_obs_dim = loaded_obs_dim
        self.agent_obs_dim = agent_obs_dim
        self.loaded_layout = loaded_layout
        self.agent_layout = agent_layout
        self.loaded_arch = loaded_arch
        self.agent_arch = agent_arch
        self.reason = reason
        if reason is not None:
            super().__init__(reason)
        elif loaded_arch is not None and loaded_arch != agent_arch:
            super().__init__(
                f"cannot warm-start a checkpoint with arch {loaded_arch!r} into "
                f"an agent with arch {agent_arch!r}: cross-architecture warm-start "
                f"is only bridged between 'shared' and 'towers'; the new trunk "
                f"layouts need their own from-scratch run (the layouts are not "
                f"partially compatible)"
            )
        elif loaded_layout is not None and loaded_layout != agent_layout:
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


def _gn_groups_mismatch(agent: Agent, loaded: Agent) -> bool:
    """Whether two ``gnres`` agents disagree on the GroupNorm group count."""
    agent_arch = getattr(agent, "arch", "shared")
    loaded_arch = getattr(loaded, "arch", "shared")
    if not (agent_arch.startswith("gnres") and loaded_arch.startswith("gnres")):
        return False
    return int(getattr(agent, "gn_groups", 8)) != int(
        getattr(loaded, "gn_groups", 8)
    )


def _res_expansion_mismatch(agent: Agent, loaded: Agent) -> bool:
    """Whether two ``bres`` agents disagree on the bottleneck multiplier."""
    agent_arch = getattr(agent, "arch", "shared")
    loaded_arch = getattr(loaded, "arch", "shared")
    if not (agent_arch.startswith("bres") and loaded_arch.startswith("bres")):
        return False
    return float(getattr(agent, "res_expansion", 3)) != float(
        getattr(loaded, "res_expansion", 3)
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

    A ``vf_outcome`` mismatch bridges as a partial copy: the shared
    trunk/actor/critic keys transfer and the added (or dropped) outcome head
    starts from its fresh init.  The two variants' critic targets differ (base
    vs total return), so the copied critic is a warm start, not an exact one.

    Both checkpoints must use the one live v5 layout; a different ``obs_dim``
    raises :class:`ValueError` rather than guessing a column mapping.  Across
    architectures only the ``shared``/``towers`` pair may bridge; any other
    arch mismatch raises :class:`WarmStartLayoutError` because the new trunk
    layouts have no partial-copy semantics (silently half-copying them would
    look like a successful warm start while leaving random weights behind).
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
    loaded_arch = getattr(loaded, "arch", "shared")
    agent_arch = getattr(agent, "arch", "shared")
    if agent_arch != loaded_arch and not (
        {agent_arch, loaded_arch} <= {"shared", "towers"}
    ):
        raise WarmStartLayoutError(
            loaded.obs_dim,
            agent.obs_dim,
            loaded_arch=loaded_arch,
            agent_arch=agent_arch,
        )
    if (
        bool(getattr(agent, "vf_nll", False)),
        bool(getattr(agent, "vf_sample", False)),
    ) != (
        bool(getattr(loaded, "vf_nll", False)),
        bool(getattr(loaded, "vf_sample", False)),
    ):
        raise WarmStartLayoutError(
            loaded.obs_dim,
            agent.obs_dim,
            reason=(
                "cannot warm-start between different value-head layouts "
                "(mean-only, variance-head, or sampled variance-head): "
                "the critic layouts differ"
            ),
        )
    if _gn_groups_mismatch(agent, loaded):
        raise WarmStartLayoutError(
            loaded.obs_dim,
            agent.obs_dim,
            reason=(
                "cannot warm-start between different GroupNorm group counts "
                f"({getattr(loaded, 'gn_groups', 8)} -> "
                f"{getattr(agent, 'gn_groups', 8)}): the normalization "
                "layouts differ"
            ),
        )
    if _res_expansion_mismatch(agent, loaded):
        raise WarmStartLayoutError(
            loaded.obs_dim,
            agent.obs_dim,
            reason=(
                "cannot warm-start between different bottleneck expansions "
                f"({getattr(loaded, 'res_expansion', 3)} -> "
                f"{getattr(agent, 'res_expansion', 3)}): the residual-block "
                "widths differ"
            ),
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
    if (
        loaded.nvec.tolist() == agent.nvec.tolist()
        and loaded.arch == agent.arch
        and bool(getattr(loaded, "vf_outcome", False))
        == bool(getattr(agent, "vf_outcome", False))
        and not _gn_groups_mismatch(agent, loaded)
        and not _res_expansion_mismatch(agent, loaded)
    ):
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
        obs, seqs, events, event_seats, event_mask = self._observation_inputs(view)
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

    def _observation_inputs(
        self, view: View
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """The single-row model inputs for ``view`` (obs + optional history).

        Shared by :meth:`act` and :meth:`value_and_outcome` so the readout is
        always evaluated on exactly the tensor layout the policy acts on.
        """
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
            seqs = torch.as_tensor(
                tokens, dtype=torch.int64, device=self.device
            ).unsqueeze(0)
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
        return obs, seqs, events, event_seats, event_mask

    @torch.no_grad()
    def value_and_outcome(self, view: View) -> tuple[float, float | None]:
        """Raw head readout for ``view``'s seat: ``(critic, outcome | None)``.

        ``critic`` is the value head's :meth:`Game.returns`-units margin
        (``own/total - mean(others)/total``); ``outcome`` is the
        ``vf_outcome`` head's discounted win/tie/loss estimate when the
        checkpoint carries one, else ``None``.  This is the one-forward-pass
        readout (no search, no rollouts) the web table shows live.
        """
        obs, seqs, events, event_seats, event_mask = self._observation_inputs(view)
        if getattr(self.agent, "vf_outcome", False):
            value, outcome = self.agent.get_value_and_outcome(
                obs, seqs, events, event_seats, event_mask
            )
            return float(value.reshape(-1)[0]), float(outcome.reshape(-1)[0])
        value = self.agent.get_value(obs, seqs, events, event_seats, event_mask)
        return float(value.reshape(-1)[0]), None

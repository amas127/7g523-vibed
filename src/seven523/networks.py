"""Torch actor-critic and the learned adapter of the ``Policy`` seam.

The rules core and :mod:`seven523.env` never import torch; this module is the
only place that does.  It needs the optional ``train`` dependency group:

    uv sync --group train
"""
from __future__ import annotations

import random
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from .actions import joint_mask_bits
from .cards import CARD_ORDER, RANK_INDEX
from .env import (
    encode_observation,
    observation_dim,
    observation_num_players,
    segment_spans,
)
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
        obs_version: int = 1,
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
        #: The encoder layout this network consumes.  Defaults to v1 so direct
        #: constructions predating observation versioning keep their meaning;
        #: :mod:`seven523.train` passes the run's version explicitly.
        self.obs_version = int(obs_version)
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
        "obs_version": int(getattr(agent, "obs_version", 1)),
        "arch": getattr(agent, "arch", "shared"),
    }
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)


def load_agent(
    path: str | Path, device: str | torch.device = "cpu"
) -> tuple[Agent, dict[str, Any]]:
    """Rebuild an :class:`Agent` from :func:`save_agent`; returns ``(agent, extra)``."""
    payload = torch.load(path, map_location=device, weights_only=True)
    agent = Agent(
        payload["obs_dim"],
        payload["nvec"],
        payload["hidden"],
        activation=payload.get("activation", "relu"),
        # Checkpoints saved before observation versioning are v1 by definition;
        # the dimension alone cannot tell v1 3-player from v2 2-player.
        obs_version=payload.get("obs_version", 1),
        # Checkpoints saved before the arch field are shared by construction.
        arch=payload.get("arch", "shared"),
    )
    agent.load_state_dict(payload["model"])
    return agent.to(device), payload.get("extra", {})


#: First-layer weight keys whose input columns are obs-layout specific and
#: therefore need the pad/guard treatment (one per possible trunk).
_INPUT_LAYER_KEYS = {
    "network.0.weight",
    "actor_network.0.weight",
    "critic_network.0.weight",
}


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


#: Cross-family observation-version pairs whose first layer needs a
#: segment-level remap: v4/v5 reorder and compress the legacy v1/v2 segments
#: (the ``hand`` block and the appended B0/B1 blocks still line up).  Prefix
#: pairs (v1 -> v2 -> v3 and v4 -> v5) are absent on purpose -- the plain
#: prefix copy in :func:`warm_start_into` already handles them, and every
#: other direction must not be guessed at (it is rejected at the train load
#: point).
_REMAP_VERSION_PAIRS: frozenset[tuple[int, int]] = frozenset(
    {(1, 4), (2, 4), (1, 5), (2, 5)}
)

#: Destination slim segment -> same-meaning source segment, copied column for
#: column.  Segments the source layout lacks stay zero (v1 has no B0/B1, v2
#: has no B1), which is exactly the appended-feature start.
_IDENTITY_SEGMENTS: dict[str, str] = {
    "hand": "hand",
    "inc_kind": "incumbent_kind",
    "inc_size": "incumbent_size",
    "draw": "draw_count",
    "trick_points": "trick_points",
    "remaining_points": "remaining_points",
    "point_hold": "point_hold",
    "unseen": "unseen",
    "last_player": "last_player",
}

_NUM_CARDS = len(CARD_ORDER)
_RANKS = len(RANK_INDEX)


def _rank_mean_matrix() -> torch.Tensor:
    """``(15, 54)`` map from a card one-hot to its rank one-hot.

    Each rank row averages the source card columns that share the rank (four
    for a standard rank, one per joker), so a rank one-hot activates the mean
    of that rank's four legacy card columns.  There is deliberately no suit
    half: the conservative compression drops suit weights to zero.
    """
    counts: dict[Any, int] = {}
    for card in CARD_ORDER:
        counts[card.rank] = counts.get(card.rank, 0) + 1
    matrix = torch.zeros(_RANKS, _NUM_CARDS)
    for index, card in enumerate(CARD_ORDER):
        matrix[RANK_INDEX[card.rank], index] = 1.0 / counts[card.rank]
    return matrix


_RANK_MEAN = _rank_mean_matrix()


@lru_cache(maxsize=None)
def _first_layer_remap(
    src_version: int, src_dim: int, dst_version: int, dst_dim: int
) -> torch.Tensor | None:
    """Column map from a source first layer to a destination first layer.

    Returns a ``(dst_dim, src_dim)`` matrix ``m`` such that the destination
    first layer can be initialised as ``w_dst = w_src @ m.T``.  The matrix is
    built segment by segment from :func:`seven523.env.segment_spans`, so no
    observation offset is hardcoded here and the map cannot drift from the
    encoder tables.

    Mapping rules for the supported forward pairs (v1/v2 -> v4/v5):

    * same-meaning segments (``hand``, ``inc_kind``, ``inc_size``, ``draw``,
      the B0 block and, when present, the B1 block) copy column for column;
    * ``incumbent_top`` -> ``inc_rank`` averages the source columns of each
      rank; its ``inc_suit`` half starts at zero (the conservative version of
      the design);
    * ``revealed`` -> each ``opp_revealed`` block gets the same rank average,
      with the suit half zeroed (the layout hides which revealed card belongs
      to which opponent, so this is the documented approximation);
    * ``hand_counts`` -> ``opp_count`` keeps the opponent slots and drops the
      self slot; ``scores`` stays identity because the legacy learner was
      always seat 0, so the self-centred rotation is a no-op at load time;
    * ``rank_counts`` and ``current`` are dropped (all-zero columns).

    ``None`` means "not remappable here": prefix pairs, unsupported
    directions, foreign dimensions and mixed player counts all fall back to
    :func:`warm_start_into`'s old prefix/skip behaviour.
    """
    src_version = int(src_version)
    dst_version = int(dst_version)
    if (src_version, dst_version) not in _REMAP_VERSION_PAIRS:
        return None
    src_players = observation_num_players(src_dim, src_version)
    dst_players = observation_num_players(dst_dim, dst_version)
    if (
        src_players is None
        or dst_players is None
        or src_players != dst_players
        or observation_dim(src_players, src_version) != src_dim
        or observation_dim(dst_players, dst_version) != dst_dim
    ):
        return None
    players = src_players
    src_spans = {
        name: (start, width)
        for name, start, width in segment_spans(src_version, players)
    }
    remap = torch.zeros(dst_dim, src_dim)
    for name, dst_start, width in segment_spans(dst_version, players):
        if name in _IDENTITY_SEGMENTS:
            source = src_spans.get(_IDENTITY_SEGMENTS[name])
            if source is None:
                continue  # the source layout lacks this segment: keep zeros
            src_start, src_width = source
            if src_width != width:
                return None  # layout drift: refuse to guess
            remap[dst_start : dst_start + width, src_start : src_start + width] = (
                torch.eye(width)
            )
        elif name == "inc_rank":
            source = src_spans.get("incumbent_top")
            if source is None or source[1] != _NUM_CARDS or width != _RANKS:
                return None
            src_start, _ = source
            remap[
                dst_start : dst_start + _RANKS, src_start : src_start + _NUM_CARDS
            ] = _RANK_MEAN
        elif name == "inc_suit":
            # Only the rank half of incumbent_top is approximated; the suit
            # half is the zero-start conservative variant (design §2).
            if "incumbent_top" not in src_spans or width != 4:
                return None
        elif name == "scores":
            source = src_spans.get("scores")
            if source is None or source[1] != width:
                return None
            src_start, _ = source
            # Learner seat 0, so self-centred order == absolute order here.
            remap[dst_start : dst_start + width, src_start : src_start + width] = (
                torch.eye(width)
            )
        elif name == "opp_count":
            source = src_spans.get("hand_counts")
            if source is None or source[1] != width + 1:
                return None
            src_start, _ = source
            for slot in range(1, width + 1):
                remap[dst_start + slot - 1, src_start + slot] = 1.0
        elif name == "opp_revealed":
            source = src_spans.get("revealed")
            block = _RANKS + 4
            if source is None or source[1] != _NUM_CARDS or width % block:
                return None
            src_start, _ = source
            for opponent in range(width // block):
                dst_block = dst_start + opponent * block
                remap[
                    dst_block : dst_block + _RANKS,
                    src_start : src_start + _NUM_CARDS,
                ] = _RANK_MEAN
        else:
            return None  # unknown destination segment: never guess
    return remap


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

    When the target observation is wider (an older-version checkpoint into a
    newer agent, e.g. v1/v2 → v3), each trunk's first layer copies the old
    input columns and zeroes the new ones, so the warm-started function is
    bit-identical on the old prefix and the appended features start with zero
    contribution.

    Across layout families (v1/v2 → v4/v5) the first layer cannot be copied by
    prefix -- S1 reorders and compresses segments -- so it is projected through
    :func:`_first_layer_remap` instead (rank means for the two compressed
    blocks, identity for the rest, zeros for dropped and missing segments).
    Any direction without a remap keeps the old skip-first-layer semantics;
    :mod:`seven523.train` rejects those pairs before calling this function.
    """
    copied: list[str] = []
    source = loaded.state_dict()
    for key, value in agent.state_dict().items():
        origin_key = _mapped_source_key(agent, loaded, key)
        origin = source.get(origin_key)
        if origin is None:
            continue
        label = key if origin_key == key else f"{key}<-{origin_key}"
        if origin_key in _INPUT_LAYER_KEYS and origin.shape[0] == value.shape[0]:
            src_version = int(getattr(loaded, "obs_version", 1))
            dst_version = int(getattr(agent, "obs_version", 1))
            remap = _first_layer_remap(
                src_version, loaded.obs_dim, dst_version, agent.obs_dim
            )
            if remap is not None:
                value.copy_(origin @ remap.to(origin.device).T)
                suffix = "" if origin_key == key else f"<-{origin_key}"
                copied.append(
                    f"{key}@remap:v{src_version}->v{dst_version}{suffix}"
                )
                continue
            # Input columns are layout-specific: only pad/truncate when both
            # checkpoints were built for the same player count (so the prefix
            # segments line up), the target version is newer, and the source
            # fits inside the target.  The 194-wide v1 3-player and v2
            # 2-player layouts are *not* prefix-compatible despite the width.
            source_players = observation_num_players(
                loaded.obs_dim, getattr(loaded, "obs_version", 1)
            )
            target_players = observation_num_players(
                agent.obs_dim, getattr(agent, "obs_version", 1)
            )
            compatible = (
                source_players is not None
                and source_players == target_players
                and getattr(loaded, "obs_version", 1)
                <= getattr(agent, "obs_version", 1)
                and origin.shape[1] <= value.shape[1]
            )
            if compatible and origin.shape[1] == value.shape[1]:
                value.copy_(origin)
                copied.append(label)
            elif compatible:
                old_width = origin.shape[1]
                value[:, :old_width].copy_(origin)
                value[:, old_width:].zero_()
                suffix = "" if origin_key == key else f"<-{origin_key}"
                copied.append(f"{key}[:, :{old_width}]{suffix}")
            continue
        if origin.shape == value.shape:
            value.copy_(origin)
            copied.append(label)
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
        # Encode with the checkpoint's own layout: a v1 agent inside a v2
        # environment keeps reading the legacy 191-wide prefix.
        self.obs_version = int(getattr(agent, "obs_version", 1))

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
            encode_observation(view, self.rules, self.obs_version),
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

"""Gymnasium environment adapter (ADR-0001/0002).

This module is the only place that knows the observation shape; the rules core
never sees a tensor.  It is a plain ``gymnasium.Env``, so it plugs into
``gymnasium.vector.SyncVectorEnv`` and the usual wrappers.

Contract: ``action_mask`` is a flat ``list[bool]`` of length
``sum(action_space.nvec)`` (134 templates + 4 suits, ADR-0004) that always
matches the observation just returned by ``reset``/``step``.
"""
from __future__ import annotations

import random
from collections import Counter

import gymnasium as gym
import numpy as np

from .actions import joint_mask_bits, nvec_for, split_action
from .cards import RANK_INDEX, card_id
from .combos import ComboKind
from .game import Game, GameState, View
from .match import Match
from .policies import Policy, RandomBot
from .rules import DEFAULT_RULES, Rules

_KIND_INDEX = {kind: i for i, kind in enumerate(ComboKind)}
_HAND = 54
_RANKS = 15
_INC_TOP = 54
_KINDS = len(ComboKind)
_NUM_CARDS = 54


def observation_dim(num_players: int) -> int:
    """Fixed observation length: hand + counts + incumbent + table."""
    return _HAND + _RANKS + _INC_TOP + _KINDS + 1 + 1 + 3 * num_players + _HAND


def encode_observation(view: View, rules: Rules = DEFAULT_RULES) -> list[float]:
    """Encode a :class:`View` into the fixed float vector an agent sees."""
    obs = [0.0] * observation_dim(rules.num_players)
    offset = 0

    for card in view.hand:  # own hand (private)
        obs[offset + card_id(card)] = 1.0
    offset += _HAND

    counts = Counter(card.rank for card in view.hand)
    for rank, index in RANK_INDEX.items():
        obs[offset + index] = counts.get(rank, 0) / 4.0
    offset += _RANKS

    if view.incumbent is not None:  # current trick (public)
        obs[offset + card_id(view.incumbent.top_card)] = 1.0
    offset += _INC_TOP

    if view.incumbent is not None:
        obs[offset + _KIND_INDEX[view.incumbent.kind]] = 1.0
    offset += _KINDS

    if view.incumbent is not None:
        obs[offset] = view.incumbent.size / rules.hand_size
    offset += 1

    obs[offset] = view.draw_count / _NUM_CARDS
    offset += 1

    for seat in range(rules.num_players):
        obs[offset + seat] = view.scores[seat] / rules.total_points
    offset += rules.num_players

    for seat in range(rules.num_players):
        obs[offset + seat] = view.counts[seat] / rules.hand_size
    offset += rules.num_players

    obs[offset + view.current] = 1.0
    offset += rules.num_players

    for card in view.revealed:  # public reveal (one known card each)
        obs[offset + card_id(card)] = 1.0
    offset += _HAND

    assert offset == len(obs)
    return obs


class Seven523Env(gym.Env):
    """Single-agent env: the learner plays ``learner``; opponents auto-advance.

    ``step`` applies the learner's action, then plays scripted opponents until
    the learner is to move again or the episode ends.

    After ``terminated=True`` the env must be reset before stepping again,
    exactly like any other Gymnasium env.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        rules: Rules = DEFAULT_RULES,
        opponents: list[Policy] | None = None,
        seed: int | None = None,
        learner: int = 0,
    ) -> None:
        self.rules = rules
        self.game = Game(rules)
        self.num_players = rules.num_players
        self.learner = learner
        self.obs_dim = observation_dim(self.num_players)
        self.nvec = nvec_for(rules)
        self.action_space_n = self.nvec[0]
        self._rng = random.Random(seed)
        self._opponents = opponents
        self._match: Match | None = None
        self.action_mask: list[bool] = [False] * sum(self.nvec)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.MultiDiscrete(list(self.nvec))

    # -- gymnasium API -------------------------------------------------------
    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng.seed(seed)
        if self._opponents is None:
            self._opponents = [
                RandomBot(random.Random(self._rng.random()))
                for _ in range(self.num_players)
            ]
        policies: list[Policy | None] = list(self._opponents)
        policies[self.learner] = None
        self._match = Match(self.rules, policies, rng=self._rng)
        self._match.advance()
        return self._publish(), {}

    def step(
        self, action: int | np.ndarray | tuple[int, int | None]
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        match = self._require_match()
        action_id, suit = split_action(action)
        match.step(action_id, suit)
        reward = 0.0
        if match.done:
            reward = self.game.returns(match.state)[self.learner]
        else:
            match.advance()
            if match.done:
                reward = self.game.returns(match.state)[self.learner]
        return self._publish(), reward, match.done, False, {}

    @property
    def state(self) -> GameState:
        """The omniscient engine state; for tests, replays and evaluation."""
        return self._require_match().state

    def view(self, seat: int | None = None) -> View:
        """The :class:`View` the learner (or ``seat``) would be given."""
        return self._require_match().view(self.learner if seat is None else seat)

    # -- internals -----------------------------------------------------------
    def _require_match(self) -> Match:
        if self._match is None:
            raise RuntimeError("call reset() first")
        return self._match

    def _publish(self) -> np.ndarray:
        view = self._require_match().view(self.learner)
        # Template head from the legality mask; preference heads stay open
        # (every value is executable: unavailable suits fall back).  ADR-0004.
        self.action_mask = joint_mask_bits(view.mask, self.nvec)
        return np.asarray(encode_observation(view, self.rules), dtype=np.float32)


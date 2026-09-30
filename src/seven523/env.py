"""Gymnasium environment adapter (ADR-0001/0002).

This module is the only place that knows the observation shape; the rules core
never sees a tensor.  It is a plain ``gymnasium.Env``, so it plugs into
``gymnasium.vector.SyncVectorEnv`` and the usual wrappers.

Contract: ``action_mask`` is a flat ``list[bool]`` of length
``sum(action_space.nvec)`` (134 templates + 4 suits, ADR-0004) that always
matches the observation just returned by ``reset``/``step``.
"""
from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from .actions import joint_mask_bits, nvec_for, split_action
from .cards import RANK_INDEX, card_id, make_deck, point_value
from .combos import ComboKind
from .game import Game, GameState, View, seat_outcome
from .match import Match
from .policies import EpisodePolicy, Policy, RandomBot
from .rules import DEFAULT_RULES, Rules

#: The observation layout is v5 (``119 + 21n``): the slim S1 family plus the
#: B0 point-context block and the B1 unseen/last-player block.  It is the only
#: layout, and it is written into every checkpoint payload so a stale file
#: fails loudly instead of being misread.
OBS_VERSION = 5

_KIND_INDEX = {kind: i for i, kind in enumerate(ComboKind)}
_HAND = 54
_RANKS = 15
_KINDS = len(ComboKind)
_NUM_CARDS = 54

#: Legal :attr:`Seven523Env.reward_shaping` modes; see the constructor docs.
_REWARD_SHAPING_MODES = (
    "terminal",
    "trick_diff",
    "win",
    "trick_diff_win",
    "terminal_win",
    "saturate",
)

#: Largest accepted ``win_jump``.  Terminal margins live in ``[−1, 1]``, so any
#: ``λ > 2`` already makes the outcome jump dominate every margin difference;
#: the cap only keeps the reward scale bounded (λ→∞ is the ``win`` limit).
_MAX_WIN_JUMP = 10.0

#: ``obs``/``offset`` writer for one segment of the observation vector.
_Writer = Callable[[View, Rules, list[float], int], None]


def validate_reward_cap(reward_shaping: str, reward_cap: float | None) -> None:
    """Validate the reward-cap contract shared by the env and the trainer.

    ``saturate`` requires an explicit cap in ``[0, 1]``; every other mode must
    leave it unset.  There is no silent fallback: ``reward_cap=None`` under
    ``saturate`` is an error, never ``terminal``.
    """
    if reward_shaping == "saturate":
        if reward_cap is None:
            raise ValueError(
                "reward_shaping='saturate' requires an explicit reward_cap "
                "(tau in [0, 1]); None would silently fall back to terminal"
            )
    elif reward_cap is not None:
        raise ValueError(
            f"reward_cap is only valid with reward_shaping='saturate', not "
            f"{reward_shaping!r}"
        )
    if reward_cap is not None and (
        not math.isfinite(reward_cap) or not 0.0 <= reward_cap <= 1.0
    ):
        raise ValueError(f"reward_cap must be in [0, 1], got {reward_cap!r}")


@dataclass(frozen=True, slots=True)
class _Segment:
    """One contiguous block of the observation: its width and how to fill it.

    ``width is None`` means a variable block: ``per_player`` slots for each of
    the ``rules.num_players - players_offset`` players, in seat order starting
    at the acting seat (writers may rotate).  A set ``width`` is a fixed
    block.  The dimensions and the encoder are both derived from the segment
    tables, so a segment cannot drift between the two.
    """

    name: str
    width: int | None
    write: _Writer
    per_player: int = 1
    players_offset: int = 0


def _write_hand(view: View, rules: Rules, obs: list[float], offset: int) -> None:
    for card in view.hand:  # own hand (private)
        obs[offset + card_id(card)] = 1.0


def _write_incumbent_kind(view: View, rules: Rules, obs: list[float], offset: int) -> None:
    if view.incumbent is not None:
        obs[offset + _KIND_INDEX[view.incumbent.kind]] = 1.0


def _write_incumbent_size(view: View, rules: Rules, obs: list[float], offset: int) -> None:
    if view.incumbent is not None:
        obs[offset] = view.incumbent.size / rules.hand_size


def _write_draw_count(view: View, rules: Rules, obs: list[float], offset: int) -> None:
    obs[offset] = view.draw_count / _NUM_CARDS


def _write_trick_points(view: View, rules: Rules, obs: list[float], offset: int) -> None:
    """B0: points already lying on the table in the current trick."""
    obs[offset] = (
        sum(point_value(card) for card in view.trick_cards) / rules.total_points
    )


def _write_remaining_points(
    view: View, rules: Rules, obs: list[float], offset: int
) -> None:
    """B0: points that nobody has won yet (draw pile + hands + table)."""
    trick_points = sum(point_value(card) for card in view.trick_cards)
    obs[offset] = (
        rules.total_points - sum(view.scores) - trick_points
    ) / rules.total_points


def _write_point_hold(view: View, rules: Rules, obs: list[float], offset: int) -> None:
    """B0: point value of the own hand."""
    obs[offset] = sum(point_value(card) for card in view.hand) / rules.total_points


def _write_unseen(view: View, rules: Rules, obs: list[float], offset: int) -> None:
    """B1: multi-hot of the cards nobody at this table can locate yet.

    ``unseen = 54 − own hand − revealed − played − current trick``.  It is the
    union of the opponents' hands and the order-unknown draw pile, so it never
    says *where* a hidden card is -- only that it has not surfaced (ADR-0002).
    """
    seen = (
        set(view.hand)
        | set(view.revealed)
        | set(view.played)
        | set(view.trick_cards)
    )
    for card in make_deck():
        if card not in seen:
            obs[offset + card_id(card)] = 1.0


def _write_last_player(view: View, rules: Rules, obs: list[float], offset: int) -> None:
    """B1: who owns the current incumbent (``View.last_player``), normalised.

    ``0.0`` when the current trick has no incumbent (``last_player is None``);
    otherwise ``(last_player + 1) / num_players``, so seats ``0..n-1`` map to
    ``1/n..1``.  The ``+1`` keeps the no-incumbent sentinel out of the seat
    range for every player count (2 and 3 players both see four distinct
    values).
    """
    if view.last_player is not None:
        obs[offset] = (view.last_player + 1) / rules.num_players


def _write_inc_rank(view: View, rules: Rules, obs: list[float], offset: int) -> None:
    """S1: rank of the incumbent's top card (jokers are the last two ranks)."""
    if view.incumbent is not None:
        obs[offset + RANK_INDEX[view.incumbent.top_card.rank]] = 1.0


def _write_inc_suit(view: View, rules: Rules, obs: list[float], offset: int) -> None:
    """S1: suit of the incumbent's top card; a joker leaves the block zero."""
    if view.incumbent is not None:
        suit = view.incumbent.top_card.suit
        if suit is not None:
            obs[offset + int(suit)] = 1.0


def _write_scores_rotated(
    view: View, rules: Rules, obs: list[float], offset: int
) -> None:
    """S1: scores rotated to the acting seat: ``scores[(seat + k) % n]``."""
    players = rules.num_players
    for k in range(players):
        obs[offset + k] = (
            view.scores[(view.seat + k) % players] / rules.total_points
        )


def _write_opp_counts(view: View, rules: Rules, obs: list[float], offset: int) -> None:
    """S1: opponent hand sizes rotated to the acting seat; self is dropped."""
    players = rules.num_players
    for k in range(1, players):
        obs[offset + k - 1] = (
            view.counts[(view.seat + k) % players] / rules.hand_size
        )


def _write_opp_revealed(
    view: View, rules: Rules, obs: list[float], offset: int
) -> None:
    """S1: each opponent's revealed card as ``rank[15] + suit[4]`` slots."""
    players = rules.num_players
    for k in range(1, players):
        card = view.revealed[(view.seat + k) % players]
        obs[offset + RANK_INDEX[card.rank]] = 1.0
        if card.suit is not None:
            obs[offset + _RANKS + int(card.suit)] = 1.0
        offset += _RANKS + 4


#: The v5 observation layout, in order.  This is the single source of truth
#: for both :func:`observation_dim` and :func:`encode_observation`.
_SEGMENTS: tuple[_Segment, ...] = (
    _Segment("hand", _HAND, _write_hand),
    _Segment("inc_rank", _RANKS, _write_inc_rank),
    _Segment("inc_suit", 4, _write_inc_suit),
    _Segment("inc_kind", _KINDS, _write_incumbent_kind),
    _Segment("inc_size", 1, _write_incumbent_size),
    _Segment("draw", 1, _write_draw_count),
    _Segment("scores", None, _write_scores_rotated),
    _Segment("opp_count", None, _write_opp_counts, per_player=1, players_offset=1),
    _Segment(
        "opp_revealed",
        None,
        _write_opp_revealed,
        per_player=_RANKS + 4,
        players_offset=1,
    ),
    _Segment("trick_points", 1, _write_trick_points),
    _Segment("remaining_points", 1, _write_remaining_points),
    _Segment("point_hold", 1, _write_point_hold),
    _Segment("unseen", _HAND, _write_unseen),
    _Segment("last_player", 1, _write_last_player),
)


def _segment_width(segment: _Segment, num_players: int) -> int:
    if segment.width is not None:
        return segment.width
    return segment.per_player * (num_players - segment.players_offset)


def observation_dim(num_players: int) -> int:
    """Fixed v5 observation length: ``119 + 21n`` (2 players: 161)."""
    return sum(_segment_width(segment, num_players) for segment in _SEGMENTS)


def encode_observation(view: View, rules: Rules = DEFAULT_RULES) -> list[float]:
    """Encode a :class:`View` into the fixed v5 float vector an agent sees."""
    obs = [0.0] * observation_dim(rules.num_players)
    offset = 0
    for segment in _SEGMENTS:
        segment.write(view, rules, obs, offset)
        offset += _segment_width(segment, rules.num_players)
    assert offset == len(obs)
    return obs


class Seven523Env(gym.Env):
    """Single-agent env: the learner plays ``learner``; opponents auto-advance.

    ``step`` applies the learner's action, then plays scripted opponents until
    the learner is to move again or the episode ends.

    The observation is the fixed v5 layout (see :func:`encode_observation`).

    After ``terminated=True`` the env must be reset before stepping again,
    exactly like any other Gymnasium env.

    ``reward_shaping`` selects how :meth:`step` decomposes the terminal return
    (the observation, action space and checkpoints are unaffected):

    ``terminal``
        Legacy behavior, bit-for-bit: ``0.0`` on every non-terminal step and
        ``Game.returns()[learner]`` on the terminal step.
    ``trick_diff``
        Per-step potential difference ``Φ(s') − Φ(s)`` with
        ``Φ = own/total − mean(others)/total`` (see :meth:`_potential`).  The
        episode's rewards telescope to the terminal return, so the objective
        is unchanged but the credit is spread over the trick boundaries.
    ``win``
        ``0.0`` on every non-terminal step; on the terminal step
        ``sign(own − max(others)) ∈ {−1, 0, +1}`` (a tie is ``0``).  The
        objective is the match outcome rather than the score gap.
    ``trick_diff_win``
        ``trick_diff`` on every step, plus the ``win`` bonus added on the
        terminal step; an episode sums to the terminal return plus the win
        term.
    ``terminal_win``
        ``0.0`` on every non-terminal step; on the terminal step the legacy
        margin plus a discrete jump at the win/loss boundary:
        ``Game.returns()[learner] + win_jump * seat_outcome(scores, learner)``.
        Two players reduce to ``(own−other)/100 + λ·sign(own−other)``; other
        player counts use ``seat_outcome`` uniformly (never a 50-point
        threshold).  ``win_jump = 0`` is exactly ``terminal``, and ``win_jump
        → ∞`` tends to ``win``.  Only this mode reads :attr:`win_jump`.
    ``saturate``
        ``0.0`` on every non-terminal step; on the terminal step the margin is
        clamped from above on a win.  The branch is outcome-conditioned (never
        a 50-point threshold), so it generalises to ``n`` players: losses keep
        ``Game.returns()[learner]``, ties score ``0.0``, and wins score
        ``min(margin, reward_cap)``.  Two players reduce to ``own < 50:
        margin``, ``own == 50: 0``, ``own > 50: min(margin, τ)``; ``τ = 0.2``
        is ``d = 20``.  Only this mode reads :attr:`reward_cap`, and it must be
        given explicitly (``None`` is an error, never ``terminal``); ``τ`` is
        a return-unit cap in ``[0, 1]``.

    An unknown mode raises :class:`ValueError`; ``win_jump`` must be finite
    and in ``[0, 10]``; ``reward_cap`` must be finite and in ``[0, 1]``, is
    required by ``saturate``, and is rejected by every other mode.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        rules: Rules = DEFAULT_RULES,
        opponents: list[Policy] | None = None,
        seed: int | None = None,
        learner: int = 0,
        reward_shaping: str = "terminal",
        win_jump: float = 1.0,
        reward_cap: float | None = None,
    ) -> None:
        if reward_shaping not in _REWARD_SHAPING_MODES:
            raise ValueError(
                f"unknown reward_shaping {reward_shaping!r}; expected one of "
                f"{', '.join(_REWARD_SHAPING_MODES)}"
            )
        if not math.isfinite(win_jump):
            raise ValueError(f"win_jump must be finite, got {win_jump!r}")
        if not 0.0 <= win_jump <= _MAX_WIN_JUMP:
            raise ValueError(
                f"win_jump must be in [0, {_MAX_WIN_JUMP}], got {win_jump!r}"
            )
        validate_reward_cap(reward_shaping, reward_cap)
        self.rules = rules
        self.game = Game(rules)
        self.num_players = rules.num_players
        self.learner = learner
        self.reward_shaping = reward_shaping
        self.win_jump = win_jump
        self.reward_cap = reward_cap
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
        # Episode-scoped opponents (EpisodePolicy) freeze their member here,
        # once per reset, instead of re-drawing every decision.  Plain policies
        # do not implement the hook and are untouched.
        for opponent in self._opponents:
            if isinstance(opponent, EpisodePolicy):
                opponent.start_episode()
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
        before = self._potential()
        match.step(action_id, suit)
        if not match.done:
            match.advance()
        after = self._potential()
        reward = self._reward(match.state, before, after)
        info: dict = {}
        if match.done:
            info["scores"] = list(match.state.scores)
            info["outcome"] = seat_outcome(match.state.scores, self.learner)
        return self._publish(), reward, match.done, False, info

    def _potential(self) -> float:
        """Φ(state) for the learner: ``own/total − mean(others)/total``.

        The same formula as :meth:`Game.returns`, but valid at any point in the
        episode: the opening state has all-zero scores (Φ = 0) and a finished
        state evaluates to ``returns()[learner]``.  Points still sitting in
        ``trick_points`` are not attributed to anyone yet, exactly like a
        mid-episode snapshot.
        """
        scores = self._require_match().state.scores
        own = scores[self.learner]
        others = (sum(scores) - own) / (self.num_players - 1)
        return (own - others) / self.rules.total_points

    def _win_bonus(self, state: GameState) -> float:
        """Outcome against the best other seat: +1 win / 0 tie / −1 loss."""
        return float(seat_outcome(state.scores, self.learner))

    def _reward(self, state: GameState, before: float, after: float) -> float:
        """Shape the transition reward for the configured mode."""
        if self.reward_shaping == "terminal":
            if not state.done:
                return 0.0
            return self.game.returns(state)[self.learner]
        if self.reward_shaping == "trick_diff":
            return after - before
        if self.reward_shaping == "win":
            return self._win_bonus(state) if state.done else 0.0
        if self.reward_shaping == "terminal_win":
            if not state.done:
                return 0.0
            return self.game.returns(state)[self.learner] + self.win_jump * (
                self._win_bonus(state)
            )
        if self.reward_shaping == "saturate":
            if not state.done:
                return 0.0
            margin = self.game.returns(state)[self.learner]
            outcome = seat_outcome(state.scores, self.learner)
            if outcome > 0:
                cap = self.reward_cap
                assert cap is not None  # __init__ rejects saturate without a cap
                return min(margin, cap)
            return 0.0 if outcome == 0 else margin
        # trick_diff_win: local credit plus the terminal outcome term.
        reward = after - before
        if state.done:
            reward += self._win_bonus(state)
        return reward

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
        return np.asarray(
            encode_observation(view, self.rules), dtype=np.float32
        )


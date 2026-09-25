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
from collections.abc import Callable
from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from .actions import joint_mask_bits, nvec_for, split_action
from .cards import RANK_INDEX, card_id, make_deck, point_value
from .combos import ComboKind
from .game import Game, GameState, View
from .match import Match
from .policies import Policy, RandomBot
from .rules import DEFAULT_RULES, Rules

#: Observation layout versions.  v1 is the original ten-segment layout
#: (``185 + 3n``); v2 appends the three B0 point-context slots (``188 + 3n``);
#: v3 appends the B1 block to v2: 54 unseen-card slots plus the normalised
#: last-player slot (``243 + 3n``).  v4/v5 are the slim S1 family: v4 is S1 +
#: B0 (``64 + 21n``) and v5 adds B1 (``119 + 21n``), so v4 is a bit-for-bit
#: prefix of v5.  The v1-v3 family and the v4-v5 family are not prefixes of
#: each other -- S1 reorders and compresses segments -- and the version is
#: checkpoint state, never inferred from the dimension: v1 with three players
#: and v2 with two players are both 194 wide.
OBS_VERSION = 5
OBS_VERSIONS = (1, 2, 3, 4, 5)

_KIND_INDEX = {kind: i for i, kind in enumerate(ComboKind)}
_HAND = 54
_RANKS = 15
_INC_TOP = 54
_KINDS = len(ComboKind)
_NUM_CARDS = 54

#: Legal :attr:`Seven523Env.reward_shaping` modes; see the constructor docs.
_REWARD_SHAPING_MODES = ("terminal", "trick_diff", "win", "trick_diff_win")

#: ``obs``/``offset`` writer for one segment of the observation vector.
_Writer = Callable[[View, Rules, list[float], int], None]


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


def _write_rank_counts(view: View, rules: Rules, obs: list[float], offset: int) -> None:
    counts = Counter(card.rank for card in view.hand)
    for rank, index in RANK_INDEX.items():
        obs[offset + index] = counts.get(rank, 0) / 4.0


def _write_incumbent_top(view: View, rules: Rules, obs: list[float], offset: int) -> None:
    if view.incumbent is not None:  # current trick (public)
        obs[offset + card_id(view.incumbent.top_card)] = 1.0


def _write_incumbent_kind(view: View, rules: Rules, obs: list[float], offset: int) -> None:
    if view.incumbent is not None:
        obs[offset + _KIND_INDEX[view.incumbent.kind]] = 1.0


def _write_incumbent_size(view: View, rules: Rules, obs: list[float], offset: int) -> None:
    if view.incumbent is not None:
        obs[offset] = view.incumbent.size / rules.hand_size


def _write_draw_count(view: View, rules: Rules, obs: list[float], offset: int) -> None:
    obs[offset] = view.draw_count / _NUM_CARDS


def _write_scores(view: View, rules: Rules, obs: list[float], offset: int) -> None:
    for seat in range(rules.num_players):
        obs[offset + seat] = view.scores[seat] / rules.total_points


def _write_hand_counts(view: View, rules: Rules, obs: list[float], offset: int) -> None:
    for seat in range(rules.num_players):
        obs[offset + seat] = view.counts[seat] / rules.hand_size


def _write_current(view: View, rules: Rules, obs: list[float], offset: int) -> None:
    obs[offset + view.current] = 1.0


def _write_revealed(view: View, rules: Rules, obs: list[float], offset: int) -> None:
    for card in view.revealed:  # public reveal (one known card each)
        obs[offset + card_id(card)] = 1.0


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


#: The v1 observation layout, in order.  This is the single source of truth for
#: both :func:`observation_dim` and :func:`encode_observation`.
_SEGMENTS_V1: tuple[_Segment, ...] = (
    _Segment("hand", _HAND, _write_hand),
    _Segment("rank_counts", _RANKS, _write_rank_counts),
    _Segment("incumbent_top", _INC_TOP, _write_incumbent_top),
    _Segment("incumbent_kind", _KINDS, _write_incumbent_kind),
    _Segment("incumbent_size", 1, _write_incumbent_size),
    _Segment("draw_count", 1, _write_draw_count),
    _Segment("scores", None, _write_scores),
    _Segment("hand_counts", None, _write_hand_counts),
    _Segment("current", None, _write_current),
    _Segment("revealed", _HAND, _write_revealed),
)

#: v2 appends B0 to v1, so every v1 slot keeps its exact position and value.
_SEGMENTS_V2: tuple[_Segment, ...] = _SEGMENTS_V1 + (
    _Segment("trick_points", 1, _write_trick_points),
    _Segment("remaining_points", 1, _write_remaining_points),
    _Segment("point_hold", 1, _write_point_hold),
)

#: v3 appends B1 to v2, so the warm-start prefix fill is just a slice.
_SEGMENTS_V3: tuple[_Segment, ...] = _SEGMENTS_V2 + (
    _Segment("unseen", _HAND, _write_unseen),
    _Segment("last_player", 1, _write_last_player),
)

#: v4 is the slim S1 layout plus B0.  S1 reorders and compresses the v1
#: blocks: the incumbent becomes ``inc_rank`` + ``inc_suit`` and the
#: per-player blocks are rotated to the acting seat, with the self slot
#: dropped from hand counts and reveals.
_SEGMENTS_V4: tuple[_Segment, ...] = (
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
)

#: v5 appends B1 to v4, so every v4 slot keeps its exact position and value.
_SEGMENTS_V5: tuple[_Segment, ...] = _SEGMENTS_V4 + (
    _Segment("unseen", _HAND, _write_unseen),
    _Segment("last_player", 1, _write_last_player),
)

_SEGMENTS_BY_VERSION: dict[int, tuple[_Segment, ...]] = {
    1: _SEGMENTS_V1,
    2: _SEGMENTS_V2,
    3: _SEGMENTS_V3,
    4: _SEGMENTS_V4,
    5: _SEGMENTS_V5,
}


def _segments_for(obs_version: int) -> tuple[_Segment, ...]:
    try:
        return _SEGMENTS_BY_VERSION[obs_version]
    except KeyError:
        raise ValueError(
            f"unknown obs_version {obs_version!r}; expected one of "
            f"{', '.join(str(version) for version in OBS_VERSIONS)}"
        ) from None


def _segment_width(segment: _Segment, num_players: int) -> int:
    if segment.width is not None:
        return segment.width
    return segment.per_player * (num_players - segment.players_offset)


def segment_spans(
    obs_version: int, num_players: int
) -> tuple[tuple[str, int, int], ...]:
    """``(name, start, width)`` for every segment of one observation layout.

    Derived from the same segment tables that drive :func:`observation_dim`
    and :func:`encode_observation`, so callers (e.g. checkpoint remapping)
    never duplicate offsets.
    """
    spans: list[tuple[str, int, int]] = []
    offset = 0
    for segment in _segments_for(obs_version):
        width = _segment_width(segment, num_players)
        spans.append((segment.name, offset, width))
        offset += width
    return tuple(spans)


def observation_num_players(
    obs_dim: int, obs_version: int = OBS_VERSION
) -> int | None:
    """Infer the player count a width was built for, or ``None`` if foreign.

    Used by warm-start compatibility: two checkpoints can share the same
    ``obs_dim`` with different versions (v1 3-player == v2 2-player == 194),
    and only the version plus the per-player segment widths can tell them
    apart.  Scans the supported table sizes (2..7 players) and returns the
    first exact match.
    """
    _segments_for(obs_version)  # reject unknown versions before scanning
    for players in range(2, 8):
        if observation_dim(players, obs_version) == obs_dim:
            return players
    return None


def observation_dim(num_players: int, obs_version: int = OBS_VERSION) -> int:
    """Fixed observation length, summed from the version's segment layout.

    ``obs_version=1`` is ``185 + 3n``; ``obs_version=2`` is ``188 + 3n``;
    ``obs_version=3`` is ``243 + 3n``; the slim S1 layouts are
    ``obs_version=4`` = ``64 + 21n`` and ``obs_version=5`` = ``119 + 21n``
    (v5 is the default; 2 players: 191 / 194 / 249 / 106 / 161).
    """
    segments = _segments_for(obs_version)
    return sum(_segment_width(segment, num_players) for segment in segments)


def encode_observation(
    view: View, rules: Rules = DEFAULT_RULES, obs_version: int = OBS_VERSION
) -> list[float]:
    """Encode a :class:`View` into the fixed float vector an agent sees.

    ``v1 < v2 < v3`` and ``v4 < v5`` are each chains of exact prefixes, so a
    consumer of an older version reads the newer vector's prefix unchanged.
    The two families are not prefixes of each other: v4/v5 (S1) reorder and
    compress the legacy segments, so a v1-v3 model cannot read a v4/v5 vector
    column-for-column and vice versa.
    """
    segments = _segments_for(obs_version)
    obs = [0.0] * observation_dim(rules.num_players, obs_version)
    offset = 0
    for segment in segments:
        segment.write(view, rules, obs, offset)
        offset += _segment_width(segment, rules.num_players)
    assert offset == len(obs)
    return obs


class Seven523Env(gym.Env):
    """Single-agent env: the learner plays ``learner``; opponents auto-advance.

    ``step`` applies the learner's action, then plays scripted opponents until
    the learner is to move again or the episode ends.

    ``obs_version`` picks the observation layout: ``1`` is the original
    ten-segment vector, ``2`` appends the B0 point-context block, ``3``
    appends B1 to v2, ``4`` is the slim S1 layout plus B0 and ``5`` (default)
    appends B1 to v4.  v4 is a bit-for-bit prefix of v5; the legacy family
    (1-3) and the slim family (4-5) are separate layouts and not prefixes of
    each other.  The version is part of the checkpoint identity, not just the
    width.

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

    An unknown mode raises :class:`ValueError`.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        rules: Rules = DEFAULT_RULES,
        opponents: list[Policy] | None = None,
        seed: int | None = None,
        learner: int = 0,
        reward_shaping: str = "terminal",
        obs_version: int = OBS_VERSION,
    ) -> None:
        if reward_shaping not in _REWARD_SHAPING_MODES:
            raise ValueError(
                f"unknown reward_shaping {reward_shaping!r}; expected one of "
                f"{', '.join(_REWARD_SHAPING_MODES)}"
            )
        self.rules = rules
        self.game = Game(rules)
        self.num_players = rules.num_players
        self.learner = learner
        self.reward_shaping = reward_shaping
        self.obs_version = obs_version
        self.obs_dim = observation_dim(self.num_players, obs_version)
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
        # Episode-scoped opponents (EpisodeMixturePolicy) freeze their member
        # here, once per reset, instead of re-drawing every decision.  Plain
        # policies have no such hook and are untouched (duck typing).
        for opponent in self._opponents:
            start_episode = getattr(opponent, "start_episode", None)
            if callable(start_episode):
                start_episode()
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
        return self._publish(), reward, match.done, False, {}

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
        own = state.scores[self.learner]
        best_other = max(
            score for seat, score in enumerate(state.scores) if seat != self.learner
        )
        if own > best_other:
            return 1.0
        if own < best_other:
            return -1.0
        return 0.0

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
            encode_observation(view, self.rules, self.obs_version), dtype=np.float32
        )


"""Public play-history sequence for the opt-in sequence-memory prototype.

D1-lite (see ``docs/experiments/sequence-memory-pilot.md``): the v5
observation stays untouched; a second network input carries the ordered card
ids of the public history (``View.played`` + ``View.trick_cards``).  The
sequence is rebuilt from the :class:`~seven523.game.View` at every decision,
so both training rollout and inference use exactly the same function and the
policy stays stateless.

Token convention: ``0`` is padding, ``card_id(card) + 1`` is a real card
(``1..54``).  The sequence is oldest-first and right-padded; the encoder reads
it with a GRU and takes the hidden state of the last real card.
"""
from __future__ import annotations

import functools

import gymnasium as gym
import numpy as np

from .cards import NUM_CARDS, card_id
from .combos import ComboKind, classify
from .game import Play, View
from .rules import Rules

__all__ = [
    "EVENT_DIM",
    "EVENT_ORDERINGS",
    "HISTORY_LENGTH",
    "ORDERINGS",
    "PASS_MODES",
    "WENT_OUT_MODES",
    "EventHistoryWrapper",
    "HistorySequenceWrapper",
    "encode_events",
    "encode_history",
    "event_length",
]

#: One card can be played at most once, so 54 tokens cover the whole deck.
HISTORY_LENGTH = 54

#: Token layouts for the history input.  ``chrono`` keeps the public play
#: order (oldest first); ``sorted`` keeps the *same multiset* of cards but
#: canonicalises it by ascending card id, destroying the chronological order.
#: The sorted arm isolates the GRU's order/recursion contribution: it has the
#: same architecture, capacity, token vocabulary and bag information as the
#: chronological arm, and differs only in token order (see
#: ``docs/experiments/sequence-gru-ablation.md``).
ORDERINGS = ("chrono", "sorted")

#: Event feature vector width (see :func:`encode_events`): 54 card multihot +
#: 6 ``ComboKind`` one-hot + size + is_pass + opens_trick + went_out.
EVENT_DIM = 64

#: Event orderings.  ``shuffled`` applies a fixed, RNG-free permutation
#: *within each trick block* (the ``opens_trick`` event keeps its position), so
#: it ablate the order/recency axis without touching the trick structure.
EVENT_ORDERINGS = ("chrono", "shuffled")

#: Pass handling: ``keep`` records pass events (rhythm/turn count); ``drop``
#: removes them (pass ablation).
PASS_MODES = ("keep", "drop")

#: ``went_out`` flag handling: ``keep`` records the public "hand hit zero"
#: marker; ``drop`` zeroes it (ablation).
WENT_OUT_MODES = ("keep", "drop")

#: Feature offsets of the per-event vector.
_CARDS = slice(0, NUM_CARDS)
_KIND = slice(NUM_CARDS, NUM_CARDS + len(ComboKind))
_SIZE = NUM_CARDS + len(ComboKind)
_IS_PASS = _SIZE + 1
_OPENS = _IS_PASS + 1
_WENT_OUT = _OPENS + 1

_KIND_INDEX = {kind: index for index, kind in enumerate(ComboKind)}

#: Fixed seed of the ``event_noise`` control (information-zero, non-degenerate).
EVENT_NOISE_SEED = 0


def event_length(rules: Rules) -> int:
    """Strict upper bound on the public event count: ``num_players * 54``.

    Every non-pass play removes at least one card (so plays <= 54), and每墩
    at most ``n - 1`` passes end it, hence passes <= ``(n - 1) * plays``;
    total <= ``n * plays <= n * 54``.  ``_advance`` skipping empty seats only
    produces fewer events.
    """
    return rules.num_players * NUM_CARDS


@functools.lru_cache(maxsize=8)
def _event_noise(length: int, dim: int = EVENT_DIM) -> np.ndarray:
    """Deterministic i.i.d. ``N(0, 1)`` noise, fixed seed ``EVENT_NOISE_SEED``."""
    rng = np.random.RandomState(EVENT_NOISE_SEED)
    return rng.standard_normal((int(length), int(dim))).astype(np.float32)


def _write_event(
    row: np.ndarray,
    play: Play,
    rules: Rules,
    *,
    boundary_blind: bool,
    went_out: str,
) -> None:
    if play.cards:
        for card in play.cards:
            row[_CARDS.start + card_id(card)] = 1.0
        combo = classify(play.cards, rules)
        if combo is not None:
            row[_KIND.start + _KIND_INDEX[combo.kind]] = 1.0
        row[_SIZE] = len(play.cards) / rules.hand_size
    else:
        row[_IS_PASS] = 1.0
    row[_OPENS] = 1.0 if play.opens_trick and not boundary_blind else 0.0
    row[_WENT_OUT] = 1.0 if play.went_out and went_out == "keep" else 0.0


def _shuffle_within_tricks(plays: list[Play]) -> list[Play]:
    """Fixed, RNG-free permutation inside each trick block (ablation arm).

    The block opener (``opens_trick``) keeps its position; the remaining
    events are re-ordered by a hash-like key, so the same ``View`` always
    yields the same sequence (training/inference parity).
    """
    out: list[Play] = []
    index = 0
    while index < len(plays):
        block = [plays[index]]
        cursor = index + 1
        while cursor < len(plays) and not plays[cursor].opens_trick:
            block.append(plays[cursor])
            cursor += 1
        tail = block[1:]
        tail_size = len(tail)
        if tail_size > 1:
            order = sorted(
                range(tail_size), key=lambda k: (k * 2654435761) % tail_size
            )
            if order == list(range(tail_size)):
                # Small blocks can make the hash key monotone; rotate so the
                # ablation is never a no-op for a multi-event block.
                order = [*list(range(1, tail_size)), 0]
            block = [block[0]] + [tail[k] for k in order]
        out.extend(block)
        index = cursor
    return out


def encode_events(
    view: View,
    rules: Rules,
    length: int | None = None,
    order: str = "chrono",
    pass_mode: str = "keep",
    boundary_blind: bool = False,
    went_out: str = "keep",
    blind: bool = False,
    noisy: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode ``View.plays`` as ``(events, seats, mask)``.

    ``events`` is ``(length, EVENT_DIM)`` float32, oldest-first and right
    padded with zero vectors; ``seats`` is the **relative** seat
    ``(play.seat - view.seat) % n`` (pad index ``n``); ``mask`` is ``(length,)``
    bool with ``True`` on real events.  ``length=None`` uses
    :func:`event_length`; longer histories keep the most recent events.

    ``blind`` zeroes the whole event vector *and* moves the seat to the pad
    index (the ``event_blind`` plateau: the channel exists but carries no
    information).  ``noisy`` replaces each event vector with a fixed-seed,
    position-indexed ``N(0, 1)`` vector and pads the seats, keeping only the
    real event count: information-zero but non-degenerate (the main control).
    """
    if length is None:
        length = event_length(rules)
    length = int(length)
    if length < 0:
        raise ValueError("length must be non-negative")
    if order not in EVENT_ORDERINGS:
        raise ValueError(
            f"unknown event order {order!r}; choose from {EVENT_ORDERINGS}"
        )
    if pass_mode not in PASS_MODES:
        raise ValueError(
            f"unknown pass mode {pass_mode!r}; choose from {PASS_MODES}"
        )
    if went_out not in WENT_OUT_MODES:
        raise ValueError(
            f"unknown went-out mode {went_out!r}; choose from {WENT_OUT_MODES}"
        )
    if blind and noisy:
        raise ValueError("event_blind and event_noisy are mutually exclusive")

    plays = list(view.plays)
    if pass_mode == "drop":
        plays = [play for play in plays if play.cards]
    if order == "shuffled":
        plays = _shuffle_within_tricks(plays)
    if len(plays) > length:
        plays = plays[len(plays) - length :]

    events = np.zeros((length, EVENT_DIM), dtype=np.float32)
    seats = np.full((length,), rules.num_players, dtype=np.int64)
    mask = np.zeros((length,), dtype=bool)
    noise = _event_noise(length) if noisy else None
    for position, play in enumerate(plays):
        mask[position] = True
        if blind:
            continue
        if noise is not None:
            events[position] = noise[position]
            continue
        _write_event(
            events[position],
            play,
            rules,
            boundary_blind=boundary_blind,
            went_out=went_out,
        )
        seats[position] = (play.seat - view.seat) % rules.num_players
    return events, seats, mask


def encode_history(
    view: View, length: int = HISTORY_LENGTH, order: str = "chrono"
) -> np.ndarray:
    """Token ids (int64) of the public play history, padded with 0 at the end.

    ``View.played`` (finished tricks, in play order) followed by
    ``View.trick_cards`` (the current trick, in play order) is the complete
    public card history; both are public facts (ADR-0002).  Longer histories
    keep the most recent ``length`` cards.

    ``order="chrono"`` (default) keeps the play order, oldest first.
    ``order="sorted"`` sorts the kept cards by ascending card id (pads stay at
    the end), so only the *multiset* survives; it is the order-ablation arm.

    Both training rollout and inference call this function, so a checkpoint's
    ``seq_order`` must be passed to both sides (it is recorded by
    :func:`seven523.networks.save_agent`).
    """
    if length < 0:
        raise ValueError("length must be non-negative")
    if order not in ORDERINGS:
        raise ValueError(f"unknown history order {order!r}; choose from {ORDERINGS}")
    played = view.played + view.trick_cards
    if len(played) > length:
        played = played[len(played) - length :]
    ids = [card_id(card) + 1 for card in played]
    if order == "sorted":
        ids.sort()
    tokens = np.zeros(length, dtype=np.int64)
    tokens[: len(ids)] = ids
    return tokens


class EventHistoryWrapper(gym.Wrapper):
    """Expose the current ``View``'s event tensors on every step.

    The training loop reads ``last_events`` / ``last_event_seats`` /
    ``last_event_mask`` with ``env.get_wrapper_attr(...)`` right where it reads
    ``action_mask``, so the tensors paired with ``next_obs`` are always encoded
    from the same :class:`View`.  All encoding flags live here, so the training
    rollout and :class:`~seven523.networks.NeuralPolicy` (which calls
    :func:`encode_events` with the checkpoint's flags) share one code path.
    """

    def __init__(
        self,
        env: gym.Env,
        rules: Rules,
        length: int | None = None,
        order: str = "chrono",
        pass_mode: str = "keep",
        boundary_blind: bool = False,
        went_out: str = "keep",
        blind: bool = False,
        noisy: bool = False,
    ) -> None:
        super().__init__(env)
        self.rules = rules
        self.length = event_length(rules) if length is None else int(length)
        if self.length < 0:
            raise ValueError("length must be non-negative")
        if order not in EVENT_ORDERINGS:
            raise ValueError(
                f"unknown event order {order!r}; choose from {EVENT_ORDERINGS}"
            )
        if pass_mode not in PASS_MODES:
            raise ValueError(
                f"unknown pass mode {pass_mode!r}; choose from {PASS_MODES}"
            )
        if went_out not in WENT_OUT_MODES:
            raise ValueError(
                f"unknown went-out mode {went_out!r}; choose from {WENT_OUT_MODES}"
            )
        if blind and noisy:
            raise ValueError("event_blind and event_noisy are mutually exclusive")
        self.order = str(order)
        self.pass_mode = str(pass_mode)
        self.boundary_blind = bool(boundary_blind)
        self.went_out = str(went_out)
        self.blind = bool(blind)
        self.noisy = bool(noisy)
        self.last_events = np.zeros((self.length, EVENT_DIM), dtype=np.float32)
        self.last_event_seats = np.full(
            (self.length,), rules.num_players, dtype=np.int64
        )
        self.last_event_mask = np.zeros((self.length,), dtype=bool)

    def _refresh(self) -> None:
        (
            self.last_events,
            self.last_event_seats,
            self.last_event_mask,
        ) = encode_events(
            self.env.unwrapped.view(),
            self.rules,
            self.length,
            order=self.order,
            pass_mode=self.pass_mode,
            boundary_blind=self.boundary_blind,
            went_out=self.went_out,
            blind=self.blind,
            noisy=self.noisy,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._refresh()
        return obs, info

    def step(self, action):
        out = self.env.step(action)
        self._refresh()
        return out


class HistorySequenceWrapper(gym.Wrapper):
    """Expose ``last_seq`` (the current view's token sequence) on every step.

    The training loop reads it with ``env.get_wrapper_attr("last_seq")`` right
    where it reads ``action_mask``, so the sequence paired with ``next_obs`` is
    always the one encoded from the same :class:`View`.
    """

    def __init__(
        self, env: gym.Env, length: int = HISTORY_LENGTH, order: str = "chrono"
    ) -> None:
        super().__init__(env)
        if order not in ORDERINGS:
            raise ValueError(
                f"unknown history order {order!r}; choose from {ORDERINGS}"
            )
        self.length = int(length)
        self.order = str(order)
        self.last_seq = np.zeros(self.length, dtype=np.int64)

    def _refresh(self) -> None:
        self.last_seq = encode_history(
            self.env.unwrapped.view(), self.length, self.order
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._refresh()
        return obs, info

    def step(self, action):
        out = self.env.step(action)
        self._refresh()
        return out

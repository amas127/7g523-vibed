"""Game flow: deal → reveal → tricks → refill → 撬底.

A pure state machine: :meth:`Game.step` returns a new :class:`GameState`.
Reveal, scoring, refill and 撬底 are automatic transitions inside ``step`` —
they are never agent actions.  :meth:`Game.view` is the single projection an
agent or bot may observe (ADR-0002).
"""
from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import Enum

from .actions import action_mask, catalog_for, resolve
from .cards import Card, card_key, is_point_card, make_deck, point_value
from .combos import Combo, beats
from .rules import DEFAULT_RULES, Rules


class Phase(Enum):
    PLAY = "play"
    DONE = "done"


@dataclass(frozen=True, slots=True)
class Play:
    """One public action record: a seat played a combo (pass = empty cards).

    ``seat`` is the acting seat, ``cards`` the concrete cards in the combo
    (canonically sorted); a pass has ``cards == ()``.  ``opens_trick`` records
    that the actor faced no incumbent (this action started a trick) and
    ``went_out`` that the actor's hand became empty (a public count fact).
    """

    seat: int
    cards: tuple[Card, ...]
    opens_trick: bool
    went_out: bool


@dataclass(frozen=True, slots=True)
class GameState:
    """The omniscient engine state.  Agents must go through :class:`View`."""

    hands: tuple[frozenset[Card], ...]
    draw_pile: tuple[Card, ...]
    scores: tuple[int, ...]
    trick_points: int
    trick_cards: tuple[Card, ...]
    revealed: tuple[Card, ...]
    current: int
    incumbent: Combo | None
    last_player: int | None
    collected: int
    phase: Phase
    #: Every card already collected into a finished trick, in play order
    #: (oldest first).  A public fact: anyone at the table saw them played.
    #: Restored traces start empty because the trace format records actions,
    #: not this derived history.
    played: tuple[Card, ...] = ()
    #: Seats whose hand is empty, in the order they went out (RULES.md 4.8/4.9,
    #: ADR-0014).  A seat is appended when its last card is played and removed
    #: when a refill gives it cards back; the first entry digs at a refill that
    #: leaves the draw pile empty.  Engine-internal: never projected into
    #: :class:`View` (ADR-0002).
    empty_order: tuple[int, ...] = ()
    #: Every public action (pass included), oldest first.  Unlike ``played``
    #: (finished-trick cards, no seats) each entry carries the acting seat and
    #: the trick boundary; it is appended by :meth:`Game.step` and rebuilt
    #: naturally by trace replay (``restore`` starts empty).
    plays: tuple[Play, ...] = ()

    @property
    def done(self) -> bool:
        return self.phase is Phase.DONE

    def hand(self, seat: int) -> frozenset[Card]:
        return self.hands[seat]


@dataclass(frozen=True, slots=True)
class Deal:
    """The opening deal: enough to rebuild a game without the RNG.

    ``Game.new`` deals one; :meth:`Game.restore` turns it back into the opening
    :class:`GameState`.  Traces persist a :class:`Deal` instead of poking at
    ``GameState`` fields, so a new field cannot silently break replay.
    """

    hands: tuple[frozenset[Card], ...]
    draw_pile: tuple[Card, ...]
    revealed: tuple[Card, ...]
    starter: int

    @classmethod
    def from_state(cls, state: "GameState") -> "Deal":
        """The deal behind ``state``; only valid before the first move."""
        return cls(state.hands, state.draw_pile, state.revealed, state.current)


@dataclass(frozen=True, slots=True)
class StepResult:
    trick_over: bool = False
    winner: int | None = None
    points_taken: int = 0
    refilled: tuple[int, ...] = ()
    dug: bool = False
    done: bool = False


@dataclass(frozen=True, slots=True)
class View:
    """What one seat is allowed to know (ADR-0002): public facts + own hand."""

    seat: int
    hand: frozenset[Card]
    mask: int
    incumbent: Combo | None
    current: int
    scores: tuple[int, ...]
    counts: tuple[int, ...]
    draw_count: int
    revealed: tuple[Card, ...]
    trick_cards: tuple[Card, ...]
    played: tuple[Card, ...]
    last_player: int | None
    done: bool
    #: Public action log, oldest-first; see :attr:`GameState.plays`.  Has a
    #: default so pre-existing keyword constructions stay valid.
    plays: tuple[Play, ...] = ()


class Game:
    def __init__(self, rules: Rules = DEFAULT_RULES) -> None:
        self.rules = rules
        self.catalog = catalog_for(rules)

    # -- lifecycle ----------------------------------------------------------
    def new(self, rng: random.Random) -> GameState:
        """Deal and return the opening state (reveal already resolved)."""
        return self.restore(self._deal(rng))

    def _deal(self, rng: random.Random) -> Deal:
        rules = self.rules
        deck = list(make_deck())
        rng.shuffle(deck)
        hands = tuple(
            frozenset(deck[seat * rules.hand_size : (seat + 1) * rules.hand_size])
            for seat in range(rules.num_players)
        )
        draw_pile = tuple(deck[rules.num_players * rules.hand_size :])
        revealed = tuple(min(hand, key=card_key) for hand in hands)
        starter = min(range(rules.num_players), key=lambda s: card_key(revealed[s]))
        return Deal(hands=hands, draw_pile=draw_pile, revealed=revealed, starter=starter)

    def restore(self, deal: Deal) -> GameState:
        """Rebuild the opening state of a dealt game (used by trace replay)."""
        return GameState(
            hands=deal.hands,
            draw_pile=deal.draw_pile,
            scores=(0,) * self.rules.num_players,
            trick_points=0,
            trick_cards=(),
            revealed=deal.revealed,
            current=deal.starter,
            incumbent=None,
            last_player=None,
            collected=0,
            phase=Phase.PLAY,
        )

    # -- information ---------------------------------------------------------
    def view(self, state: GameState, seat: int) -> View:
        """Project the state for one seat; ``mask`` covers templates only (ADR-0004)."""
        hand = state.hands[seat]
        return View(
            seat=seat,
            hand=hand,
            mask=action_mask(hand, state.incumbent, self.rules),
            incumbent=state.incumbent,
            current=state.current,
            scores=state.scores,
            counts=tuple(len(h) for h in state.hands),
            draw_count=len(state.draw_pile),
            revealed=state.revealed,
            trick_cards=state.trick_cards,
            played=state.played,
            last_player=state.last_player,
            done=state.done,
            plays=state.plays,
        )

    # -- transitions ---------------------------------------------------------
    def step(
        self, state: GameState, action_id: int, suit: int | None = None
    ) -> tuple[GameState, StepResult]:
        if state.done:
            raise RuntimeError("the game is over")
        seat = state.current
        mask = action_mask(state.hands[seat], state.incumbent, self.rules)
        if not (mask >> action_id) & 1:
            raise ValueError(f"illegal action {action_id} for seat {seat}")

        action = self.catalog[action_id]
        if action.is_pass:
            noted = replace(
                state, plays=(*state.plays, Play(seat, (), False, False))
            )
            return self._advance(noted, StepResult())

        combo = resolve(action, state.hands[seat], self.rules, suit=suit)
        if combo is None or (
            state.incumbent is not None
            and not beats(combo, state.incumbent, self.rules)
        ):
            # A suit request can realise a weaker top card than the mask assumed;
            # fall back to the strongest realisation so every joint action stays
            # legal (the template was masked legal, so this cannot be None).
            combo = resolve(action, state.hands[seat], self.rules)
        if combo is None:  # the mask guarantees this cannot happen
            raise RuntimeError("legal action failed to resolve")
        hand = state.hands[seat].difference(combo.cards)
        hands = (*state.hands[:seat], hand, *state.hands[seat + 1 :])
        empty_order = state.empty_order
        if not hand and seat not in empty_order:
            empty_order = (*empty_order, seat)
        points = sum(point_value(card) for card in combo.cards)
        after = replace(
            state,
            hands=hands,
            incumbent=combo,
            last_player=seat,
            trick_cards=state.trick_cards + combo.cards,
            trick_points=state.trick_points + points,
            empty_order=empty_order,
            plays=(
                *state.plays,
                Play(seat, combo.cards, state.incumbent is None, not hand),
            ),
        )
        # 撬底, revision 3 (RULES.md 4.8, ADR-0014): going out with an empty
        # draw pile ends the round on the spot — the actor digs before any
        # opponent may respond, so the trick's winner is always the actor.
        if not hand and not after.draw_pile:
            return self._end_trick(after, StepResult(), immediate_digger=seat)
        return self._advance(after, StepResult())

    def _advance(self, state: GameState, result: StepResult) -> tuple[GameState, StepResult]:
        n = self.rules.num_players
        nxt = (state.current + 1) % n
        while nxt != state.last_player and not state.hands[nxt]:
            nxt = (nxt + 1) % n
        if nxt == state.last_player:
            return self._end_trick(state, result)
        return replace(state, current=nxt), result

    def _end_trick(
        self,
        state: GameState,
        result: StepResult,
        *,
        immediate_digger: int | None = None,
    ) -> tuple[GameState, StepResult]:
        rules = self.rules
        n = rules.num_players
        trick_winner = state.last_player
        if trick_winner is None:
            raise RuntimeError("trick ended without a player")

        scores = list(state.scores)
        hands = list(state.hands)
        draw_pile = state.draw_pile
        collected = state.collected + len(state.trick_cards)
        played = state.played + state.trick_cards

        # Refill every seat from the winner's next seat, winner last.
        refilled: list[int] = []
        if draw_pile:
            pile = list(draw_pile)
            for offset in range(1, n + 1):
                seat = (trick_winner + offset) % n
                need = rules.hand_size - len(hands[seat])
                take = min(need, len(pile))
                if take:
                    hands[seat] = hands[seat] | frozenset(pile[:take])
                    del pile[:take]
                    refilled.append(seat)
            draw_pile = tuple(pile)

        # A seat that got cards is no longer "out"; drop it from the order.
        refilled_seats = set(refilled)
        empty_order = tuple(
            seat for seat in state.empty_order if seat not in refilled_seats
        )

        # 撬底, revision 3 (RULES.md 4.8/4.9, ADR-0014): the first player to go
        # out digs.  The immediate path names the digger; otherwise a trick
        # whose refill emptied the draw pile sends the earliest empty seat.
        digger = immediate_digger
        if digger is None and not draw_pile and empty_order:
            digger = empty_order[0]

        gained = 0
        if digger is not None:
            for seat in range(n):
                if seat == digger:
                    continue
                kept = [card for card in hands[seat] if not is_point_card(card)]
                gained += sum(point_value(c) for c in hands[seat]) - sum(
                    point_value(c) for c in kept
                )
                collected += len(hands[seat]) - len(kept)
                hands[seat] = frozenset(kept)
            scores[digger] += state.trick_points + gained
        else:
            scores[trick_winner] += state.trick_points

        # ``winner`` names the seat that collected ``points_taken``: the trick
        # winner normally, the digger when the trick ends in 撬底.
        winner = digger if digger is not None else trick_winner
        after = replace(
            state,
            hands=tuple(hands),
            draw_pile=draw_pile,
            scores=tuple(scores),
            trick_points=0,
            trick_cards=(),
            current=winner,
            incumbent=None,
            last_player=None,
            collected=collected,
            played=played,
            empty_order=empty_order,
            phase=Phase.DONE if digger is not None else Phase.PLAY,
        )
        return after, replace(
            result,
            trick_over=True,
            winner=winner,
            points_taken=state.trick_points + gained,
            refilled=tuple(refilled),
            dug=digger is not None,
            done=digger is not None,
        )

    # -- rewards -------------------------------------------------------------
    def returns(self, state: GameState) -> tuple[float, ...]:
        """Terminal score reward: ``own/100 - mean(others)/100`` (RL-2)."""
        n = self.rules.num_players
        if not state.done:
            return (0.0,) * n
        total = sum(state.scores)
        if total != self.rules.total_points:
            raise AssertionError(f"points not conserved: {state.scores}")
        return tuple(
            score / self.rules.total_points
            - (total - score) / ((n - 1) * self.rules.total_points)
            for score in state.scores
        )


def seat_outcome(scores: Sequence[int], seat: int) -> int:
    """The one win/tie/loss definition: +1 / 0 / -1 against the best other seat."""
    own = scores[seat]
    best_other = max(score for other, score in enumerate(scores) if other != seat)
    return int(own > best_other) - int(own < best_other)

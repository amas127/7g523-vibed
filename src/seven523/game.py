"""Game flow: deal → reveal → tricks → refill → 撬底.

A pure state machine: :meth:`Game.step` returns a new :class:`GameState`.
Reveal, scoring, refill and 撬底 are automatic transitions inside ``step`` —
they are never agent actions.  :meth:`Game.view` is the single projection an
agent or bot may observe (ADR-0002).
"""
from __future__ import annotations

import random
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

    @property
    def done(self) -> bool:
        return self.phase is Phase.DONE

    def hand(self, seat: int) -> frozenset[Card]:
        return self.hands[seat]


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
    last_player: int | None
    done: bool


class Game:
    def __init__(self, rules: Rules = DEFAULT_RULES) -> None:
        self.rules = rules
        self.catalog = catalog_for(rules)

    # -- lifecycle ----------------------------------------------------------
    def new(self, rng: random.Random) -> GameState:
        rules = self.rules
        deck = list(make_deck())
        rng.shuffle(deck)
        hands: list[frozenset[Card]] = []
        for seat in range(rules.num_players):
            start = seat * rules.hand_size
            hands.append(frozenset(deck[start : start + rules.hand_size]))
        draw_pile = tuple(deck[rules.num_players * rules.hand_size :])
        revealed = tuple(min(hand, key=card_key) for hand in hands)
        starter = min(range(rules.num_players), key=lambda s: card_key(revealed[s]))
        return GameState(
            hands=tuple(hands),
            draw_pile=draw_pile,
            scores=(0,) * rules.num_players,
            trick_points=0,
            trick_cards=(),
            revealed=revealed,
            current=starter,
            incumbent=None,
            last_player=None,
            collected=0,
            phase=Phase.PLAY,
        )

    def is_terminal(self, state: GameState) -> bool:
        return state.done

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
            last_player=state.last_player,
            done=state.done,
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
            return self._advance(state, StepResult())

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
        hands = state.hands[:seat] + (hand,) + state.hands[seat + 1 :]
        points = sum(point_value(card) for card in combo.cards)
        after = replace(
            state,
            hands=hands,
            incumbent=combo,
            last_player=seat,
            trick_cards=state.trick_cards + combo.cards,
            trick_points=state.trick_points + points,
        )
        return self._advance(after, StepResult())

    def _advance(self, state: GameState, result: StepResult) -> tuple[GameState, StepResult]:
        n = self.rules.num_players
        nxt = (state.current + 1) % n
        while nxt != state.last_player and not state.hands[nxt]:
            nxt = (nxt + 1) % n
        if nxt == state.last_player:
            return self._end_trick(state, result)
        return replace(state, current=nxt), result

    def _end_trick(self, state: GameState, result: StepResult) -> tuple[GameState, StepResult]:
        rules = self.rules
        n = rules.num_players
        winner = state.last_player
        if winner is None:
            raise RuntimeError("trick ended without a player")

        scores = list(state.scores)
        scores[winner] += state.trick_points
        hands = list(state.hands)
        draw_pile = state.draw_pile
        collected = state.collected + len(state.trick_cards)

        # Refill every seat from the winner's next seat, winner last.
        refilled: list[int] = []
        if draw_pile:
            pile = list(draw_pile)
            for offset in range(1, n + 1):
                seat = (winner + offset) % n
                need = rules.hand_size - len(hands[seat])
                take = min(need, len(pile))
                if take:
                    hands[seat] = hands[seat] | frozenset(pile[:take])
                    del pile[:take]
                    refilled.append(seat)
            draw_pile = tuple(pile)

        # 撬底: the winner is empty and no refill was possible.
        dug = not hands[winner] and not draw_pile
        gained = 0
        if dug:
            for seat in range(n):
                if seat == winner:
                    continue
                kept = [card for card in hands[seat] if not is_point_card(card)]
                gained += sum(point_value(c) for c in hands[seat]) - sum(
                    point_value(c) for c in kept
                )
                collected += len(hands[seat]) - len(kept)
                hands[seat] = frozenset(kept)
            scores[winner] += gained

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
            phase=Phase.DONE if dug else Phase.PLAY,
        )
        return after, replace(
            result,
            trick_over=True,
            winner=winner,
            points_taken=state.trick_points + gained,
            refilled=tuple(refilled),
            dug=dug,
            done=dug,
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

    def score_winner(self, state: GameState) -> int | None:
        """Highest-score seat, or ``None`` for a draw (human play / evaluation)."""
        if not state.done:
            return None
        best = max(state.scores)
        winners = [seat for seat, score in enumerate(state.scores) if score == best]
        return winners[0] if len(winners) == 1 else None

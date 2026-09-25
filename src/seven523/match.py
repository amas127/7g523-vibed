"""The multi-seat driver: turn rotation and policy dispatch, nothing else.

``Game`` is a pure state machine; every caller that wants to *play* a game
(env, evaluation, terminal play, replay, tests) otherwise re-implements the same
loop: ask the current seat's policy for an action, project a :class:`View`, fall
back when an action is illegal, advance to the next seat, and repeat.  This
module owns that loop once.

Seats are either *policy-driven* (a :class:`~seven523.policies.Policy` acts) or
*external* (``None``: the caller drives them with :meth:`Match.step`).  The
environment pauses the learner seat; evaluation and terminal play give every
seat a policy; replay drives the recorded actions through :meth:`Match.step`.

An externally supplied action is trusted and validated strictly by ``Game`` (an
illegal one raises).  An action a *policy* returns is clamped to the first legal
action and counted in :attr:`Match.illegal_actions`, so a misbehaving agent can
never strand a game.
"""
from __future__ import annotations

import random
from typing import Callable, Sequence

from .actions import legal_ids, split_action
from .game import Game, GameState, StepResult, View
from .policies import Policy
from .rules import DEFAULT_RULES, Rules

#: ``(seat, action_id, suit, view, result)`` after every turn.
TurnHook = Callable[[int, int, "int | None", View, StepResult], None]


class Match:
    """Drives one 对局 on top of ``Game``; see the module docstring."""

    def __init__(
        self,
        rules: Rules = DEFAULT_RULES,
        policies: Sequence[Policy | None] | None = None,
        *,
        state: GameState | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self.game = Game(rules)
        self.rules = rules
        n = rules.num_players
        if policies is None:
            policies = [None] * n
        if len(policies) != n:
            raise ValueError(
                f"expected {n} policies, got {len(policies)}"
            )
        self._policies: list[Policy | None] = list(policies)
        self._state = state if state is not None else self.game.new(rng or random.Random())
        self.steps = 0
        self.turns: list[int] = [0] * n
        self.illegal_actions = 0
        #: Optional observer fired after every turn, whoever took it.
        self.on_turn: TurnHook | None = None

    # -- state access --------------------------------------------------------
    @property
    def state(self) -> GameState:
        return self._state

    @property
    def done(self) -> bool:
        return self._state.done

    @property
    def seat(self) -> int:
        """The seat to move."""
        return self._state.current

    def view(self, seat: int | None = None) -> View:
        """The projection for ``seat`` (default: the seat to move)."""
        return self.game.view(self._state, self.seat if seat is None else seat)

    def policy_for(self, seat: int) -> Policy | None:
        return self._policies[seat]

    def returns(self) -> tuple[float, ...]:
        return self.game.returns(self._state)

    # -- driving -------------------------------------------------------------
    def step(self, action_id: int | None = None, suit: int | None = None) -> StepResult:
        """Take one turn.

        With an ``action_id`` the action is used as given (validated strictly).
        Without one, the current seat's policy acts; an illegal result is
        clamped to the first legal action and counted.
        """
        if self._state.done:
            raise RuntimeError("the game is over")
        seat = self._state.current
        view = self.game.view(self._state, seat)
        if action_id is None:
            policy = self._policies[seat]
            if policy is None:
                raise RuntimeError(
                    f"seat {seat} has no policy; drive it with step(action_id)"
                )
            action_id, suit = self._clamp(view, *split_action(policy.act(view)))
        self._state, result = self.game.step(self._state, action_id, suit)
        self.steps += 1
        self.turns[seat] += 1
        if self.on_turn is not None:
            self.on_turn(seat, action_id, suit, view, result)
        return result

    def advance(
        self, *, stop: int | None = None
    ) -> StepResult | None:
        """Play policy-driven seats until a stop condition.

        Stops when the game ends, when the seat to move has no policy, or when
        it is ``stop``'s turn.  Returns the last :class:`StepResult`, or ``None``
        if no turn was taken.
        """
        last: StepResult | None = None
        while not self.done:
            if stop is not None and self.seat == stop:
                break
            if self._policies[self.seat] is None:
                break
            last = self.step()
        return last

    def run_to_end(self) -> GameState:
        """Play to the end with policies; every seat must have one."""
        self.advance()
        if not self.done:
            raise RuntimeError(
                f"seat {self.seat} has no policy; use advance()/step() to drive it"
            )
        return self._state

    # -- internals -----------------------------------------------------------
    def _clamp(self, view: View, action_id: int, suit: int | None) -> tuple[int, int | None]:
        if action_id >= 0 and (view.mask >> action_id) & 1:
            return action_id, suit
        legal = legal_ids(view.mask)
        if not legal:  # pragma: no cover - the engine always offers PASS when following
            raise RuntimeError(f"seat {view.seat} has no legal action")
        self.illegal_actions += 1
        return legal[0], None

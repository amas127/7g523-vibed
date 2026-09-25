"""7鬼523 — a pure-rules card game core with a Gymnasium training adapter.

Public API::

    from seven523 import Game, DEFAULT_RULES, CATALOG, action_mask

    game = Game(DEFAULT_RULES)
    state = game.new(random.Random(0))
    view = game.view(state, state.current)
    state, result = game.step(state, action_id)
"""
from __future__ import annotations

import random

from .actions import (
    CATALOG,
    PASS_ID,
    SUIT_N,
    Action,
    action_mask,
    build_catalog,
    catalog_for,
    legal_ids,
    resolve,
    split_action,
    suit_options,
    top_rank,
)
from .cards import (
    NUM_CARDS,
    Card,
    Rank,
    Suit,
    card_id,
    card_key,
    make_deck,
    point_value,
)
from .combos import Combo, ComboKind, beats, classify
from .env import Seven523Env, observation_dim
from .game import Game, GameState, Phase, StepResult, View
from .policies import GreedyBot, Policy, RandomBot, make_scripted_policies
from .rules import DEFAULT_RULES, Rules

__all__ = [
    "Action",
    "CATALOG",
    "PASS_ID",
    "SUIT_N",
    "action_mask",
    "build_catalog",
    "catalog_for",
    "legal_ids",
    "resolve",
    "split_action",
    "suit_options",
    "top_rank",
    "NUM_CARDS",
    "Card",
    "Rank",
    "Suit",
    "card_id",
    "card_key",
    "make_deck",
    "point_value",
    "Combo",
    "ComboKind",
    "beats",
    "classify",
    "Game",
    "GameState",
    "Phase",
    "StepResult",
    "View",
    "Seven523Env",
    "observation_dim",
    "GreedyBot",
    "Policy",
    "RandomBot",
    "make_scripted_policies",
    "DEFAULT_RULES",
    "Rules",
]


def main() -> None:
    """Play one demo game between random bots and print the trick log."""
    rng = random.Random(7)
    game = Game(DEFAULT_RULES)
    state = game.new(rng)
    bots: list[RandomBot] = [RandomBot(rng) for _ in range(DEFAULT_RULES.num_players)]
    while not state.done:
        seat = state.current
        view = game.view(state, seat)
        action_id, _suit = split_action(bots[seat].act(view))
        print(f"seat {seat}: {CATALOG[action_id]}")
        state, result = game.step(state, action_id)
        if result.trick_over:
            print(
                f"  trick -> seat {result.winner} (+{result.points_taken}) "
                f"dug={result.dug}"
            )
    print("scores:", state.scores)

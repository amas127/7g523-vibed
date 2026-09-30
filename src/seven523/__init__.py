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
    index_hand,
    joint_mask_bits,
    legal_ids,
    nvec_for,
    resolve,
    resolve_indexed,
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
from .game import Deal, Game, GameState, Phase, Play, StepResult, View, seat_outcome
from .match import Match
from .policies import Policy, RandomBot, make_scripted_policies
from .rules import DEFAULT_RULES, Rules

__all__ = [
    "CATALOG",
    "DEFAULT_RULES",
    "NUM_CARDS",
    "PASS_ID",
    "SUIT_N",
    "Action",
    "Card",
    "Combo",
    "ComboKind",
    "Deal",
    "Game",
    "GameState",
    "Match",
    "Phase",
    "Play",
    "Policy",
    "RandomBot",
    "Rank",
    "Rules",
    "Seven523Env",
    "StepResult",
    "Suit",
    "View",
    "action_mask",
    "beats",
    "build_catalog",
    "card_id",
    "card_key",
    "catalog_for",
    "classify",
    "index_hand",
    "joint_mask_bits",
    "legal_ids",
    "make_deck",
    "make_scripted_policies",
    "nvec_for",
    "observation_dim",
    "point_value",
    "resolve",
    "resolve_indexed",
    "seat_outcome",
    "split_action",
    "suit_options",
    "top_rank",
]


def main() -> None:
    """Play one demo game between random bots and print the trick log."""
    rng = random.Random(7)
    bots: list[RandomBot] = [RandomBot(rng) for _ in range(DEFAULT_RULES.num_players)]
    match = Match(DEFAULT_RULES, bots, rng=rng)

    def on_turn(seat, action_id, suit, view, result):
        print(f"seat {seat}: {CATALOG[action_id]}")
        if result.trick_over:
            print(
                f"  trick -> seat {result.winner} (+{result.points_taken}) "
                f"dug={result.dug}"
            )

    match.on_turn = on_turn
    match.run_to_end()
    print("scores:", match.state.scores)

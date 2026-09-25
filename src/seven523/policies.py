"""Opponent policies — the real seam (scripted bots today, neural policy later).

Every policy returns a joint action ``(template_id, suit | None)``.  Scripted
bots always pass ``suit=None`` (the engine then uses the strongest suits); only
:class:`~seven523.networks.NeuralPolicy` searches the suit head (ADR-0004).
"""
from __future__ import annotations

import random
from typing import Protocol

from .actions import catalog_for, legal_ids, resolve
from .game import View
from .rules import DEFAULT_RULES, Rules

#: ``(template_id, suit | None)``.
JointAction = tuple[int, int | None]


class Policy(Protocol):
    def act(self, view: View) -> JointAction: ...


class RandomBot:
    """Uniform over the legal templates; strongest-suit realisation."""

    def __init__(self, rng: random.Random | None = None) -> None:
        self.rng = rng or random.Random()

    def act(self, view: View) -> JointAction:
        return self.rng.choice(legal_ids(view.mask)), None


class GreedyBot:
    """Weakest legal beat, non-bombs first; PASS only when nothing else is legal."""

    def __init__(self, rules: Rules = DEFAULT_RULES) -> None:
        self.rules = rules
        self.catalog = catalog_for(rules)
        self.pass_id = len(self.catalog) - 1

    def act(self, view: View) -> JointAction:
        best_id = self.pass_id
        best_key: tuple | None = None
        for action_id in legal_ids(view.mask):
            if action_id == self.pass_id:
                continue
            combo = resolve(self.catalog[action_id], view.hand, self.rules)
            if combo is None:
                continue
            key = (combo.is_bomb, combo.strength)
            if best_key is None or key < best_key:
                best_key = key
                best_id = action_id
        return best_id, None


def make_scripted_policies(
    mode: str, rules: Rules = DEFAULT_RULES, seed: int | None = None
) -> list[Policy]:
    """One opponent policy per seat for the built-in modes ('random' / 'greedy')."""
    if mode == "random":
        rng = random.Random(seed)
        return [
            RandomBot(random.Random(rng.randrange(1 << 32)))
            for _ in range(rules.num_players)
        ]
    if mode == "greedy":
        return [GreedyBot(rules) for _ in range(rules.num_players)]
    raise ValueError(f"unknown scripted opponent mode: {mode!r}")

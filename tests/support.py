"""Test-only scripted policy: the smallest deterministic bot.

The production spec grammar now ships only ``random`` and ``ckpt:<path>``
(ADR-0012 removed the GreedyBot), but the suite still needs a deterministic,
torch-free policy to drive games and to stand in for the opponent seats.  This
is deliberately *not* a rated entrant and not part of the library: it plays the
first legal action id, which coincides with the retired bot's weakest-legal
choice in essentially every state (so existing trajectories and expected
numbers survive the removal).
"""
from __future__ import annotations

from seven523.actions import legal_ids
from seven523.game import View
from seven523.rules import DEFAULT_RULES, Rules

__all__ = ["FirstLegalBot"]


class FirstLegalBot:
    def __init__(self, rules: Rules = DEFAULT_RULES) -> None:
        self.rules = rules

    def act(self, view: View) -> tuple[int, int | None]:
        return legal_ids(view.mask)[0], None

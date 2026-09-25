"""Rule configuration — the "rules as data" seam.

The standard variant from RULES.md is :data:`DEFAULT_RULES`.  Fields live here
so callers never edit the engine to try a variant.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Rules:
    num_players: int = 2
    hand_size: int = 7
    straight_min: int = 3
    straight_max: int = 7
    consecutive_pairs_min: int = 3
    consecutive_pairs_max: int = 3
    total_points: int = 100

    def __post_init__(self) -> None:
        if self.num_players < 2:
            raise ValueError("need at least two players")
        if self.num_players * self.hand_size > 54:
            raise ValueError("not enough cards for the requested deal")
        if self.straight_min < 3 or self.straight_max < self.straight_min:
            raise ValueError("bad straight bounds")
        if self.consecutive_pairs_min < 3:
            raise ValueError("consecutive pairs need at least three pairs")
        if self.consecutive_pairs_max < self.consecutive_pairs_min:
            raise ValueError("bad consecutive-pair bounds")


DEFAULT_RULES = Rules()

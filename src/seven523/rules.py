"""Rule configuration — the "rules as data" seam.

The standard variant from RULES.md is :data:`DEFAULT_RULES`.  Fields live here
so callers never edit the engine to try a variant.  ``comparison`` selects the
combo-comparison semantics; ``"tier"`` is the pre-2026-09-25 variant kept only
to replay legacy traces.
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
    #: Combo comparison semantics consumed by :func:`~seven523.combos.beats`.
    #: ``"family"`` is the RULES.md §3.1 variant; ``"tier"`` is the flat-tier
    #: comparison used before 2026-09-25, kept only to replay old traces.
    comparison: str = "family"

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
        if self.comparison not in {"family", "tier"}:
            raise ValueError(f"unknown comparison {self.comparison!r}")


DEFAULT_RULES = Rules()

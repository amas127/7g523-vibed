"""Rule configuration — the "rules as data" seam.

The standard variant from RULES.md is :data:`DEFAULT_RULES`.  Fields live here
so callers never edit the engine to try a variant.  :func:`rules_id` freezes a
rule version into a short identity string: ratings measured under two different
ids are never pooled (ADR-0013).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any


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


# Behaviour-semantic revision of the rules.  Bump this whenever a rule or
# comparison-semantics change that affects measured ratings is not captured by
# a ``Rules`` field — for example the ADR-0007 tier-to-family change in
# ``combos.py``.  Forgetting a bump silently pools ratings across versions,
# which is costlier than a false mismatch, so RULES.md changes must carry it.
#
# History:
#   2 -> 3 (2026-09-26): 撬底 triggers on going out, not on winning the trick
#   (RULES.md 4.8/4.9, §7 R-Q13; ADR-0014).  A player who plays their last card
#   with an empty draw pile ends the round immediately and digs; a refill that
#   empties the draw pile sends the earliest-to-empty player.
RULES_REVISION: int = 3


def rules_identity(rules: Rules = DEFAULT_RULES) -> dict[str, Any]:
    """Canonical identity payload: the semantic revision plus all rule fields."""
    return {"revision": RULES_REVISION, "fields": asdict(rules)}


def rules_id(rules: Rules = DEFAULT_RULES) -> str:
    """Short stable id for ``rules``; never pool ratings across differing ids."""
    payload = json.dumps(
        rules_identity(rules), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

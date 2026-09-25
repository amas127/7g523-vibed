"""Opponent pool, scheduling and per-game selection for placement sessions.

The selectable pool comes from a study manifest (pinned anchors plus free
ladder rungs); seats alternate for an exact 5/5 split and the chooser is
Fisher information with Thompson exploration for the first games (HR §6.2).
"""
from __future__ import annotations

import math
import random
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from ..elo import Prior, expected_score
from ..policies import missing_ckpt_path
from ..trace import player_label

if TYPE_CHECKING:
    # Annotation only: ``stop_reason`` reads the config but must not create a
    # runtime opponents -> session dependency.
    from .session import SessionConfig


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


# -- opponents ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Opponent:
    """One selectable opponent: a pinned anchor or a free ladder rung."""

    id: str
    elo: float
    se: float = 0.0
    spec: str | None = None
    anchor: bool = False

    def __post_init__(self) -> None:
        if not self.id or "@" in self.id:
            raise ValueError(f"opponent id must be non-empty and '@'-free: {self.id!r}")
        if not _finite(self.elo):
            raise ValueError(f"opponent {self.id!r} needs a finite elo, got {self.elo!r}")
        if not _finite(self.se) or self.se < 0.0:
            raise ValueError(f"opponent {self.id!r} needs a non-negative se, got {self.se!r}")


class MissingCheckpointWarning(UserWarning):
    """A non-anchor opponent was skipped because its ``ckpt:`` file is gone."""


def load_opponents(
    manifest: Mapping[str, Any],
) -> tuple[tuple[Opponent, ...], dict[str, float]]:
    """Read the selectable pool and the pinned anchors from a study manifest.

    The pool is the manifest's ``levels`` map — the anchors (D-6's 秤砣) plus
    the trained rungs — sorted by rating then id so selection is deterministic.
    Each opponent's policy spec comes from the matching ``subjects`` entry.

    A rung whose ``ckpt:<path>`` spec names a file that no longer exists is
    skipped with a :class:`MissingCheckpointWarning` (the frozen study manifest
    may outlive the checkpoints it lists).  Anchors are never skipped: a missing
    anchor checkpoint raises ``ValueError`` because the pins are load-bearing.
    """
    levels = manifest.get("levels")
    if not isinstance(levels, Mapping) or not levels:
        raise ValueError("manifest needs a non-empty 'levels' map")
    anchor_entries = manifest.get("anchors")
    if not anchor_entries:
        raise ValueError("manifest needs 'anchors' for a placement session")
    subjects = {
        str(subject["id"]): subject for subject in manifest.get("subjects") or []
    }

    def spec_for(id_: str) -> str | None:
        spec = subjects.get(id_, {}).get("spec")
        return str(spec) if spec is not None else None

    anchors: dict[str, float] = {}
    for entry in anchor_entries:
        id_ = str(entry["id"])
        if id_ not in levels:
            raise ValueError(f"anchor {id_!r} is missing from manifest levels")
        spec = spec_for(id_)
        missing = missing_ckpt_path(spec)
        if missing is not None:
            raise ValueError(
                f"anchor {id_!r} spec {spec!r} points at a missing checkpoint "
                f"{missing!r}"
            )
        anchors[id_] = float(entry["elo"])
    opponents: list[Opponent] = []
    skipped: list[tuple[str, str, str]] = []
    for id_, elo in levels.items():
        id_ = str(id_)
        spec = spec_for(id_)
        missing = missing_ckpt_path(spec)
        if missing is not None:
            skipped.append((id_, str(spec), missing))
            continue
        subject = subjects.get(id_, {})
        opponents.append(
            Opponent(
                id=id_,
                elo=float(elo),
                se=float(subject.get("se") or 0.0),
                spec=spec,
                anchor=id_ in anchors,
            )
        )
    if skipped:
        details = "; ".join(
            f"{id_!r} spec={spec!r} path={path!r}" for id_, spec, path in skipped
        )
        warnings.warn(
            f"skipping {len(skipped)} opponent(s) with missing checkpoints: {details}",
            MissingCheckpointWarning,
            stacklevel=2,
        )
    opponents.sort(key=lambda opponent: (opponent.elo, opponent.id))
    return tuple(opponents), anchors


# -- scheduling and selection ------------------------------------------------


def plan_seats(games: int, *, start: int = 0) -> tuple[int, ...]:
    """Alternating seat plan; even ``games`` ⇒ an exact 5/5 split (D-6=(b))."""
    if games < 2:
        raise ValueError("a session needs at least 2 games")
    if games % 2 != 0:
        raise ValueError("games must be even so the seats split 50/50")
    if start not in (0, 1):
        raise ValueError(f"seat start must be 0 or 1, got {start!r}")
    return tuple((start + index) % 2 for index in range(games))


def plan_deals(rng: random.Random, games: int) -> tuple[int, ...]:
    """``games`` distinct deal seeds (one fresh deck per game, D-6=(b))."""
    if games < 1:
        raise ValueError("games must be at least 1")
    seeds: list[int] = []
    seen: set[int] = set()
    while len(seeds) < games:
        seed = rng.randrange(1 << 32)
        if seed in seen:
            continue
        seen.add(seed)
        seeds.append(seed)
    return tuple(seeds)


def select_opponent(
    posterior: Prior,
    candidates: Sequence[Opponent],
    *,
    n_games: int = 0,
    explore_games: int = 2,
    mode: str = "auto",
    rng: random.Random | None = None,
) -> Opponent:
    """Pick the next rung: Fisher information, with Thompson early exploration.

    ``info`` maximises the Bernoulli Fisher information
    ``β²·p(1−p)`` at the posterior mean (HR §6.1/§6.2); ``thompson`` samples the
    human rating from ``posterior`` and each candidate from its own
    ``N(elo, se)`` before taking the same argmax — the first ``explore_games``
    games use it so a biased trace prior cannot lock the session onto the wrong
    rung (HR §6.2: "前 1–2 局用 Thompson").  ``mode="auto"`` is Thompson while
    ``n_games < explore_games`` and info afterwards; ties break by (elo, id).
    """
    if mode not in ("auto", "info", "thompson"):
        raise ValueError(f"mode must be auto/info/thompson, got {mode!r}")
    if explore_games < 0:
        raise ValueError("explore_games must be non-negative")
    candidates = tuple(candidates)
    if not candidates:
        raise ValueError("no opponents to choose from")
    ordered = sorted(candidates, key=lambda opponent: (opponent.elo, opponent.id))
    chosen = mode
    if mode == "auto":
        chosen = "thompson" if n_games < explore_games else "info"

    best: Opponent | None = None
    best_score = -1.0
    if chosen == "info":
        for opponent in ordered:
            p = expected_score(posterior.mean, opponent.elo)
            score = p * (1.0 - p)
            if score > best_score:
                best, best_score = opponent, score
        assert best is not None
        return best

    rng = rng or random.Random()
    human_rating = rng.gauss(posterior.mean, posterior.sd)
    for opponent in ordered:
        rating = rng.gauss(opponent.elo, opponent.se) if opponent.se > 0.0 else opponent.elo
        p = expected_score(human_rating, rating)
        score = p * (1.0 - p)
        if score > best_score:
            best, best_score = opponent, score
    assert best is not None
    return best


def stop_reason(
    n_games: int, ci_half_width: float, config: SessionConfig
) -> str | None:
    """``"ci"`` when the honest CI has closed, ``"max_games"`` at the cap."""
    if n_games >= config.games:
        return "max_games"
    if n_games >= config.min_games_before_stop and ci_half_width <= config.stop_ci:
        return "ci"
    return None


def session_player_labels(
    *,
    human_id: str,
    human_seat: int,
    num_players: int,
    opponent: Opponent,
) -> list[str]:
    """Trace ``players`` labels for one session game.

    The human seat keeps the ``human@seatN`` convention used by
    ``7g523-play``; the opponent carries its rung identity — anchors keep the
    study's ``anchor:<id>@seatN`` role, every other rung is
    ``opponent:<id>@seatN`` — so the M1 calibration can read the opponent
    strength off the trace (HR §6.3/§8).
    """
    if not 0 <= human_seat < num_players:
        raise ValueError(f"human seat {human_seat} outside 0..{num_players - 1}")
    role = "anchor" if opponent.anchor else "opponent"
    return [
        f"{human_id}@seat{seat}"
        if seat == human_seat
        else player_label(role, opponent.id, seat)
        for seat in range(num_players)
    ]

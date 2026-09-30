"""M2/M3 of ``docs/human-elo-plan.md``: 10-game human placement sessions.

One placement session is a fixed 10-game 牌局 schedule (10 **different** deals,
5/5 seat rotation — owner D-6=(b)), an adaptive opponent chooser (info, with
Thompson exploration for the first games — HR §6.2), and the D3 estimator from
``docs/experiments/human-elo-10-games-research.md`` §5:

* **trace channel** — every game's trace is fed through the M1 artifact
  (``artifacts/human-elo/prior.json``): the S1 features are extracted with the
  D1 extractor, ``f'(φ)`` is predicted with the M1 model, and the running mean
  becomes ``μ_traj``; ``Prior(μ_traj, σ_traj(n))`` is the human's prior,
* **result channel** — the 牌局 results replay through the OpenSkill
  Plackett–Luce update (ADR-0011); the trace and result Gaussians are combined
  by precision (HR §5.5) so the two channels do not double-count,
* **honest posterior** — the OpenSkill sigma, a 95% CI, the nearest manifest
  level, and a ``provisional`` flag; 10 games are never dressed up as ±50.

The session persists ``traces/sessions/<id>/`` without any ``--save-trace``:
one trace per game, ``session.json`` after every game, ``report.json`` at the
end, and a small ``rungs.json`` sidecar with the opponent pool metadata.

The package is pure orchestration around :func:`seven523.play.play_game`
(interactive UX is reused unchanged) and :mod:`seven523.elo`; it never touches
the rating math.  ``run()`` takes any chooser factory, so tests and the
``--simulate`` CLI mode exercise the whole pipeline without a TTY.

``__init__`` re-exports the historical module's public API; the code lives in
:mod:`.opponents`, :mod:`.estimator`, :mod:`.session` and :mod:`.cli`.
"""
from __future__ import annotations

from ..prior import SCHEMA as TRACE_PRIOR_SCHEMA, TracePrior
from .cli import build_parser as build_parser, main as main
from .estimator import (
    COLD_START_PRIOR,
    RESULT_PRIOR as RESULT_PRIOR,
    RUNG_PRIOR_SD,
    SessionConfig,
    band_for as band_for,
    channel_weights,
    fit_session,
    nearest_level,
)
from .opponents import (
    MissingCheckpointWarning,
    Opponent,
    load_opponents,
    plan_deals,
    plan_seats,
    select_opponent,
    session_player_labels,
    stop_reason,
)
from .session import (
    REPORT_SCHEMA,
    SESSION_SCHEMA,
    VERSION as VERSION,
    GameRecord,
    PlacementSession,
    ScheduledGame,
    new_session_id,
)

__all__ = [
    "COLD_START_PRIOR",
    "REPORT_SCHEMA",
    "RUNG_PRIOR_SD",
    "SESSION_SCHEMA",
    "TRACE_PRIOR_SCHEMA",
    "GameRecord",
    "MissingCheckpointWarning",
    "Opponent",
    "PlacementSession",
    "ScheduledGame",
    "SessionConfig",
    "TracePrior",
    "build_parser",
    "channel_weights",
    "fit_session",
    "load_opponents",
    "main",
    "nearest_level",
    "new_session_id",
    "plan_deals",
    "plan_seats",
    "select_opponent",
    "session_player_labels",
    "stop_reason",
]

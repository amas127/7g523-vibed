"""M2/M3 of ``docs/human-elo-plan.md``: 10-game human placement sessions.

One placement session is a fixed 10-game 牌局 schedule (10 **different** deals,
5/5 seat rotation — owner D-6=(b)), an adaptive opponent chooser (info, with
Thompson exploration for the first games — HR §6.2), and the D3 estimator from
``docs/experiments/human-elo-10-games-research.md`` §5:

* **trace channel** — every game's trace is fed through the M1 artifact
  (``artifacts/human-elo/prior.json``): the S1 features are extracted with the
  D1 extractor, ``f'(φ)`` is predicted with the M1 model, and the running mean
  becomes ``μ_traj``; ``Prior(μ_traj, σ_traj(n))`` is the human's prior,
* **result channel** — the win/loss Bernoulli plus the 分差 Gaussian
  (``FitConfig.margin``) enter the same :func:`seven523.elo.fit_ratings`, with
  ``margin=(c, 2σ)`` while the trace prior is present (HR §5.3/§7.4) so the two
  channels do not double-count,
* **honest posterior** — the joint Hessian SE, a 95% CI, the nearest manifest
  level, and a ``provisional`` flag; 10 games are never dressed up as ±50.

The session persists ``traces/sessions/<id>/`` without any ``--save-trace``:
one trace per game, ``session.json`` after every game, ``report.json`` at the
end, and a small ``rungs.json`` sidecar with the opponent pool metadata.

The module is pure orchestration around :func:`seven523.play.play_game`
(interactive UX is reused unchanged) and :mod:`seven523.elo`; it never touches
the rating math.  ``run()`` takes any chooser factory, so tests and the
``--simulate`` CLI mode exercise the whole pipeline without a TTY.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
import warnings
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .elo import Fit, FitConfig, PlayedGame, Prior, expected_score, fit_ratings
from .policies import Policy, missing_ckpt_path, policy_from_spec
from .play import QuitGame, interactive_chooser
from .record import play_recorded, policy_seed
from .rules import DEFAULT_RULES, Rules
from .study import load_manifest
from .trace import player_label

__all__ = [
    "COLD_START_PRIOR",
    "MARGIN_C",
    "MARGIN_SIGMA",
    "MARGIN_TRACE_FACTOR",
    "REPORT_SCHEMA",
    "SESSION_SCHEMA",
    "RUNG_PRIOR_SD",
    "TRACE_PRIOR_SCHEMA",
    "GameRecord",
    "MissingCheckpointWarning",
    "Opponent",
    "PlacementSession",
    "SessionConfig",
    "TracePrior",
    "build_parser",
    "channel_weights",
    "fit_session",
    "load_opponents",
    "load_prior_tools",
    "main",
    "nearest_level",
    "new_session_id",
    "plan_deals",
    "plan_seats",
    "select_opponent",
    "session_player_labels",
    "session_margin",
    "stop_reason",
]

#: Document tags; readers reject anything else.
SESSION_SCHEMA = "seven523.placement-session"
REPORT_SCHEMA = "seven523.placement-report"
TRACE_PRIOR_SCHEMA = "seven523.trace-prior"
VERSION = 1


#: Cold start before any trace exists (HR §5.4: the first-game prior).
COLD_START_PRIOR = Prior(1500.0, 300.0)

#: 分差 likelihood constants, calibrated on the frozen study (HR §5.3):
#: ``margin ~ N(c·(R−R_opp), σ²)``.  With the trace prior present the σ is
#: doubled (HR §7.4: the channels are correlated, σ×2 restores coverage).
MARGIN_C = 0.143
MARGIN_SIGMA = 44.5
MARGIN_TRACE_FACTOR = 2.0

#: HR §5.2: free ladder levels enter the joint fit with a wide source prior
#: (their own label uncertainty); pinned anchors are exact instead.
RUNG_PRIOR_SD = 30.0

#: The result-channel decomposition prior (``FitConfig``'s own default).
RESULT_PRIOR = Prior(1500.0, 200.0)


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


# -- the M1 trace prior as an online channel ---------------------------------


_TOOLS_CACHE: Any = None


def load_prior_tools() -> Any:
    """Import ``tools/fit_trace_prior.py`` lazily (it owns the prior schema).

    ``placement`` deliberately does not re-implement ``predict_elo`` /
    ``prior_for_session`` / the S1 extractor: the M1 tool is the single owner
    of the artifact semantics.  The import is lazy so ``--help`` and
    ``--no-trace-prior`` sessions never need numpy.
    """
    global _TOOLS_CACHE
    if _TOOLS_CACHE is None:
        path = Path(__file__).resolve().parents[2] / "tools" / "fit_trace_prior.py"
        if not path.exists():
            raise FileNotFoundError(
                f"the M1 prior tool is missing: {path} "
                "(run from the repository, or pass --no-trace-prior)"
            )
        spec = importlib.util.spec_from_file_location(
            "seven523_fit_trace_prior", path
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _TOOLS_CACHE = module
    return _TOOLS_CACHE


class TracePrior:
    """The M1 ``prior.json`` artifact as used online.

    ``_tools`` is injectable so tests can drive a session with a synthetic
    predictor instead of the numpy-backed calibration tool.
    """

    def __init__(
        self,
        doc: Mapping[str, Any],
        *,
        path: str | Path | None = None,
        _tools: Any = None,
    ) -> None:
        if doc.get("schema") != TRACE_PRIOR_SCHEMA:
            raise ValueError(
                f"not a {TRACE_PRIOR_SCHEMA} artifact, got {doc.get('schema')!r}"
            )
        self.doc = doc
        self.path = Path(path) if path is not None else None
        self._tools = _tools

    @classmethod
    def load(cls, path: str | Path) -> TracePrior:
        tools = load_prior_tools()
        return cls(tools.load_prior(path), path=path, _tools=tools)

    def _impl(self) -> Any:
        if self._tools is None:
            self._tools = load_prior_tools()
        return self._tools

    def predict_elo(self, row: Mapping[str, Any], *, opponent_elo: float) -> float:
        return float(
            self._impl().predict_elo(self.doc, row, opponent_elo=float(opponent_elo))
        )

    def prior_for_session(self, mu_traj: float, n: int) -> Prior:
        return self._impl().prior_for_session(self.doc, float(mu_traj), int(n))

    def features(self, trace: Mapping[str, Any], *, verify: bool = False) -> dict[str, Any]:
        return dict(self._impl().extract_features(trace, verify=verify))

    @property
    def cold_start(self) -> Prior:
        prior = self.doc.get("prior") or {"mean": 1500.0, "sd": 300.0}
        return Prior(float(prior["mean"]), float(prior["sd"]))

    @property
    def labels(self) -> dict[str, float]:
        """Elos the prior was calibrated against (opponent-strength feature)."""
        data = self.doc.get("data") or {}
        return {
            str(name): float(value)
            for name, value in (data.get("labels") or {}).items()
        }

    def meta(self) -> dict[str, Any]:
        return {
            "path": str(self.path) if self.path is not None else None,
            "schema": self.doc.get("schema"),
            "version": self.doc.get("version"),
            "kind": self.doc.get("kind"),
            "labels": self.doc.get("labels"),
            "created_at": self.doc.get("created_at"),
            "scheme": self.doc.get("scheme"),
            "deshrink": dict(self.doc.get("deshrink") or {}),
        }


# -- estimation --------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SessionConfig:
    """Session constants; defaults are the HR §5.3/§5.4 production choices."""

    games: int = 10
    stop_ci: float = 50.0
    min_games_before_stop: int = 1
    explore_games: int = 2
    z: float = 1.96
    margin_c: float = MARGIN_C
    margin_sigma: float = MARGIN_SIGMA
    trace_margin_factor: float = MARGIN_TRACE_FACTOR
    rung_prior_sd: float = RUNG_PRIOR_SD
    seat_start: int = 0
    human_id: str = "human"

    def __post_init__(self) -> None:
        if self.games < 2 or self.games % 2 != 0:
            raise ValueError("games must be an even number >= 2 (5/5 seats)")
        if self.stop_ci <= 0.0:
            raise ValueError("stop_ci must be positive")
        if not 1 <= self.min_games_before_stop <= self.games:
            raise ValueError("min_games_before_stop must be in 1..games")
        if not 0 <= self.explore_games <= self.games:
            raise ValueError("explore_games must be in 0..games")
        if self.z <= 0.0:
            raise ValueError("z must be positive")
        if self.margin_c <= 0.0 or self.margin_sigma <= 0.0:
            raise ValueError("margin constants must be positive")
        if self.trace_margin_factor <= 0.0:
            raise ValueError("trace_margin_factor must be positive")
        if self.rung_prior_sd <= 0.0:
            raise ValueError("rung_prior_sd must be positive")
        if self.seat_start not in (0, 1):
            raise ValueError("seat_start must be 0 or 1")
        if not self.human_id or "@" in self.human_id:
            raise ValueError("human_id must be non-empty and '@'-free")


def session_margin(
    config: SessionConfig, *, with_trace: bool
) -> tuple[float, float]:
    """``(c, σ)``: σ is doubled while the trace prior is in play (HR §7.4)."""
    sigma = config.margin_sigma * (config.trace_margin_factor if with_trace else 1.0)
    return (config.margin_c, sigma)


def fit_session(
    games: Sequence[PlayedGame],
    *,
    human_id: str,
    human_prior: Prior,
    opponents: Sequence[Opponent],
    anchors: Mapping[str, float],
    margin: tuple[float, float],
    rung_prior_sd: float = RUNG_PRIOR_SD,
    rung_centers: Mapping[str, float] | None = None,
) -> Fit:
    """The D3 fit: free human prior + free rung priors + pinned anchors.

    Every non-anchor opponent enters with ``Prior(center, rung_prior_sd)`` so
    its label uncertainty propagates through the joint Hessian (HR §5.2);
    ``window=None`` because the first 10 games are the whole point (HR §5.3).
    """
    priors: dict[str, Prior] = {human_id: human_prior}
    for opponent in opponents:
        if opponent.anchor:
            continue
        center = float((rung_centers or {}).get(opponent.id, opponent.elo))
        priors[opponent.id] = Prior(center, rung_prior_sd)
    return fit_ratings(
        tuple(games),
        anchors=dict(anchors),
        priors=priors,
        window=None,
        config=FitConfig(margin=margin),
    )


def channel_weights(
    trace_sd: float | None, result_se: float
) -> dict[str, float]:
    """Gaussian-precision split between the trace prior and result likelihood.

    The trace channel contributes ``1/σ_traj²``; the result channel contributes
    ``1/se_result²`` from a same-design fit with only the weak default prior.
    The two weights sum to 1 (HR §5.5's "两通道的权重").
    """
    result_precision = 0.0 if result_se <= 0.0 else 1.0 / (result_se * result_se)
    if trace_sd is None:
        return {"trace": 0.0, "result": 1.0}
    prior_precision = 1.0 / (trace_sd * trace_sd)
    total = prior_precision + result_precision
    if total <= 0.0:
        return {"trace": 0.0, "result": 0.0}
    return {"trace": prior_precision / total, "result": result_precision / total}


def nearest_level(
    elo: float, levels: Mapping[str, float]
) -> dict[str, Any] | None:
    """The manifest level closest to ``elo`` (ties by elo then id)."""
    if not levels:
        return None
    id_, value = min(
        levels.items(), key=lambda item: (abs(float(item[1]) - elo), item[1], item[0])
    )
    return {"id": str(id_), "elo": float(value), "distance": abs(float(value) - elo)}


def band_for(ci_half_width: float, *, stop_ci: float = 50.0) -> str:
    """HR §5.5 bands: ``placed`` / ``provisional`` (±1 档) / ``coarse``."""
    if ci_half_width <= stop_ci:
        return "placed"
    if ci_half_width <= 2.0 * stop_ci:
        return "provisional"
    return "coarse"


# -- session -----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GameRecord:
    """One completed placement game, as written to ``session.json``."""

    index: int
    seed: int
    seat: int
    opponent_id: str
    opponent_elo: float
    scores: tuple[int, ...]
    result: str
    trace: str
    mu_traj: float | None = None
    sigma_traj: float | None = None
    posterior_elo: float | None = None
    posterior_se: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "seed": self.seed,
            "seat": self.seat,
            "opponent_id": self.opponent_id,
            "opponent_elo": self.opponent_elo,
            "scores": list(self.scores),
            "result": self.result,
            "trace": self.trace,
            "mu_traj": self.mu_traj,
            "sigma_traj": self.sigma_traj,
            "posterior_elo": self.posterior_elo,
            "posterior_se": self.posterior_se,
        }


def new_session_id(base: str | Path, *, now: datetime | None = None) -> str:
    """A timestamped session id that does not collide inside ``base``."""
    base = Path(base)
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    candidate = f"session-{stamp}"
    suffix = 2
    while (base / candidate).exists():
        candidate = f"session-{stamp}-{suffix}"
        suffix += 1
    return candidate


class PlacementSession:
    """One 10-game placement session: schedule → play → fit → report.

    Pure orchestration: :meth:`run` drives ``play_game`` with an injected
    chooser factory (interactive, scripted or test double), persists every
    artifact under ``directory``, and returns the report document.
    """

    def __init__(
        self,
        *,
        opponents: Sequence[Opponent],
        anchors: Mapping[str, float],
        directory: str | Path,
        session_id: str,
        config: SessionConfig | None = None,
        rules: Rules = DEFAULT_RULES,
        rng: random.Random | None = None,
        trace_prior: TracePrior | None = None,
        policy_factory: Callable[[Opponent, int], Policy] | None = None,
        feature_verify: bool = False,
        device: str = "cpu",
        created_at: str | None = None,
    ) -> None:
        self.opponents = tuple(opponents)
        if not self.opponents:
            raise ValueError("a session needs at least one opponent")
        self.anchors = {str(id_): float(elo) for id_, elo in anchors.items()}
        self.config = config or SessionConfig()
        self.rules = rules
        self.directory = Path(directory)
        self.session_id = session_id
        self.rng = rng or random.Random()
        self.trace_prior = trace_prior
        self.policy_factory = policy_factory
        self.feature_verify = feature_verify
        self.device = device
        self.created_at = created_at or datetime.now().isoformat(timespec="seconds")

        self._seats = plan_seats(self.config.games, start=self.config.seat_start)
        self._seeds = plan_deals(self.rng, self.config.games)
        self._games: list[PlayedGame] = []
        self._records: list[GameRecord] = []
        self._rows: list[tuple[dict[str, Any], float]] = []
        self._mu_traj: float | None = None
        self._human_prior: Prior = self._cold_prior()
        self._fit: Fit | None = None
        self._result_fit: Fit | None = None
        self._stopped_reason: str | None = None
        self._last_trace: dict[str, Any] | None = None

    # -- state ------------------------------------------------------------

    @property
    def levels(self) -> dict[str, float]:
        return {opponent.id: opponent.elo for opponent in self.opponents}

    @property
    def games(self) -> tuple[PlayedGame, ...]:
        return tuple(self._games)

    @property
    def records(self) -> tuple[GameRecord, ...]:
        return tuple(self._records)

    @property
    def fit(self) -> Fit | None:
        return self._fit

    def _cold_prior(self) -> Prior:
        if self.trace_prior is not None:
            return self.trace_prior.cold_start
        return COLD_START_PRIOR

    def posterior_prior(self) -> Prior:
        """The human posterior used to select the next opponent."""
        if self._fit is None:
            return self._cold_prior()
        rating = self._fit.ratings[self.config.human_id]
        return Prior(rating.elo, rating.se)

    def ci_half_width(self) -> float:
        if self._fit is None:
            return float("inf")
        return self.config.z * self._fit.ratings[self.config.human_id].se

    def select_opponent(self) -> Opponent:
        """Next rung under HR §6.2 (Thompson first, then info)."""
        return select_opponent(
            self.posterior_prior(),
            self.opponents,
            n_games=len(self._games),
            explore_games=self.config.explore_games,
            rng=self.rng,
        )

    # -- one game ---------------------------------------------------------

    def _opponent_policy(self, opponent: Opponent, seed: int) -> Policy:
        if self.policy_factory is not None:
            return self.policy_factory(opponent, seed)
        if opponent.spec is None:
            raise ValueError(
                f"opponent {opponent.id!r} has no policy spec in the manifest"
            )
        return policy_from_spec(opponent.spec, self.rules, seed=seed, device=self.device)

    def _play_one(
        self,
        opponent: Opponent,
        *,
        index: int,
        seat: int,
        seed: int,
        human_policy: Policy | None = None,
        chooser: Callable[..., Any] | None = None,
        print_fn: Callable[..., None],
    ) -> tuple[dict[str, Any], Path]:
        policy = self._opponent_policy(opponent, policy_seed(seed, seat))
        policies: list[Policy] = [policy] * self.rules.num_players
        if human_policy is not None:
            policies[seat] = human_policy
        recorded = play_recorded(
            policies,
            chooser=chooser,
            rules=self.rules,
            seed=seed,
            human_seat=seat,
            players=session_player_labels(
                human_id=self.config.human_id,
                human_seat=seat,
                num_players=self.rules.num_players,
                opponent=opponent,
            ),
            created_at=self.created_at,
            trace_dir=self.directory,
            trace_index=index,
            opponent=opponent.id,
            print_fn=print_fn,
        )
        assert recorded.trace_path is not None
        return recorded.trace, recorded.trace_path

    def _record_game(
        self, opponent: Opponent, *, seat: int, seed: int, scores: Sequence[int], trace_path: Path
    ) -> None:
        index = len(self._games)
        if seat == 0:
            seats = (self.config.human_id, opponent.id)
        else:
            seats = (opponent.id, self.config.human_id)
        self._games.append(PlayedGame(seed, seats, tuple(int(score) for score in scores)))
        mine, theirs = int(scores[seat]), int(scores[1 - seat])
        result = "win" if mine > theirs else ("loss" if mine < theirs else "draw")
        self._records.append(
            GameRecord(
                index=index,
                seed=seed,
                seat=seat,
                opponent_id=opponent.id,
                opponent_elo=opponent.elo,
                scores=tuple(int(score) for score in scores),
                result=result,
                trace=trace_path.name,
            )
        )

    def _refit(self) -> None:
        if self.trace_prior is not None:
            assert self._last_trace is not None
            opponent_elo = self._records[-1].opponent_elo
            row = self.trace_prior.features(self._last_trace, verify=self.feature_verify)
            self._rows.append((row, opponent_elo))
            self._mu_traj = sum(
                self.trace_prior.predict_elo(row, opponent_elo=elo)
                for row, elo in self._rows
            ) / len(self._rows)
            self._human_prior = self.trace_prior.prior_for_session(
                self._mu_traj, len(self._games)
            )
        else:
            self._human_prior = self._cold_prior()
        margin = session_margin(self.config, with_trace=self.trace_prior is not None)
        centers = self.trace_prior.labels if self.trace_prior is not None else None
        self._fit = fit_session(
            self._games,
            human_id=self.config.human_id,
            human_prior=self._human_prior,
            opponents=self.opponents,
            anchors=self.anchors,
            margin=margin,
            rung_prior_sd=self.config.rung_prior_sd,
            rung_centers=centers,
        )
        self._result_fit = fit_session(
            self._games,
            human_id=self.config.human_id,
            human_prior=RESULT_PRIOR,
            opponents=self.opponents,
            anchors=self.anchors,
            margin=margin,
            rung_prior_sd=self.config.rung_prior_sd,
            rung_centers=centers,
        )
        posterior = self._fit.ratings[self.config.human_id]
        self._records[-1] = replace(
            self._records[-1],
            mu_traj=self._mu_traj,
            sigma_traj=self._human_prior.sd if self.trace_prior is not None else None,
            posterior_elo=posterior.elo,
            posterior_se=posterior.se,
        )

    # -- persistence ------------------------------------------------------

    def _session_document(self) -> dict[str, Any]:
        return {
            "schema": SESSION_SCHEMA,
            "version": VERSION,
            "id": self.session_id,
            "created_at": self.created_at,
            "directory": str(self.directory),
            "config": asdict(self.config),
            "rules": asdict(self.rules),
            "anchors": dict(self.anchors),
            "opponents": [
                {
                    "id": opponent.id,
                    "elo": opponent.elo,
                    "se": opponent.se,
                    "spec": opponent.spec,
                    "anchor": opponent.anchor,
                }
                for opponent in self.opponents
            ],
            "prior": self.trace_prior.meta() if self.trace_prior is not None else None,
            "seats": list(self._seats),
            "seeds": list(self._seeds),
            "stopped_reason": self._stopped_reason,
            "games": [record.to_dict() for record in self._records],
        }

    def _write_json(self, name: str, document: Mapping[str, Any]) -> Path:
        path = self.directory / name
        path.write_text(
            json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return path

    def _persist(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self._write_json("session.json", self._session_document())

    def _write_rungs(self) -> None:
        self._write_json(
            "rungs.json",
            {
                "schema": "seven523.placement-rungs",
                "version": VERSION,
                "session": self.session_id,
                "anchors": dict(self.anchors),
                "opponents": self._session_document()["opponents"],
                "prior_labels": self.trace_prior.labels if self.trace_prior is not None else {},
            },
        )

    # -- report -----------------------------------------------------------

    def report(self) -> dict[str, Any]:
        """The final document: point, CI, band, provisional, two channels."""
        if self._fit is None or self._result_fit is None:
            raise ValueError("no games have been played yet")
        human = self._fit.ratings[self.config.human_id]
        result = self._result_fit.ratings[self.config.human_id]
        ci_half_width = self.config.z * human.se
        margin = session_margin(
            self.config, with_trace=self.trace_prior is not None
        )
        weights = channel_weights(
            self._human_prior.sd if self.trace_prior is not None else None,
            result.se,
        )
        nearest = nearest_level(human.elo, self.levels)
        rungs: list[dict[str, Any]] = []
        for opponent in self.opponents:
            fitted = self._fit.ratings.get(opponent.id)
            rungs.append(
                {
                    "id": opponent.id,
                    "anchor": opponent.anchor,
                    "spec": opponent.spec,
                    "elo": opponent.elo,
                    "se": opponent.se,
                    "prior_center": (
                        None
                        if opponent.anchor
                        else float(
                            (self.trace_prior.labels if self.trace_prior else {}).get(
                                opponent.id, opponent.elo
                            )
                        )
                    ),
                    "prior_sd": None if opponent.anchor else self.config.rung_prior_sd,
                    "fitted_elo": None if fitted is None else fitted.elo,
                    "fitted_se": None if fitted is None else fitted.se,
                    "games": None if fitted is None else fitted.n,
                }
            )
        return {
            "schema": REPORT_SCHEMA,
            "version": VERSION,
            "session": {
                "id": self.session_id,
                "directory": str(self.directory),
                "created_at": self.created_at,
                "session_json": str(self.directory / "session.json"),
                "report": str(self.directory / "report.json"),
            },
            "games_played": len(self._games),
            "stop": {
                "reason": self._stopped_reason,
                "games": len(self._games),
                "stop_ci": self.config.stop_ci,
                "min_games_before_stop": self.config.min_games_before_stop,
            },
            "human": {
                "id": self.config.human_id,
                "elo": human.elo,
                "se": human.se,
                "ci95": [human.elo - ci_half_width, human.elo + ci_half_width],
                "ci_half_width": ci_half_width,
                "n": human.n,
                "z": self.config.z,
            },
            "provisional": ci_half_width > self.config.stop_ci,
            "band": band_for(ci_half_width, stop_ci=self.config.stop_ci),
            "nearest_level": nearest,
            "levels": dict(self.levels),
            "channels": {
                "trace": {
                    "enabled": self.trace_prior is not None,
                    "mu_traj": self._mu_traj,
                    "sigma_traj": (
                        self._human_prior.sd if self.trace_prior is not None else None
                    ),
                    "n_traces": len(self._rows),
                    "artifact": (
                        self.trace_prior.meta() if self.trace_prior is not None else None
                    ),
                    "weight": weights["trace"],
                },
                "result": {
                    "enabled": True,
                    "se": result.se,
                    "games": len(self._games),
                    "prior": {"mean": RESULT_PRIOR.mean, "sd": RESULT_PRIOR.sd},
                    "weight": weights["result"],
                },
                "weights": weights,
                "margin": [margin[0], margin[1]],
            },
            "rungs": rungs,
            "plan": {
                "seats": list(self._seats),
                "seeds": list(self._seeds),
                "played": [record.to_dict() for record in self._records],
            },
            "fit": {
                "converged": self._fit.converged,
                "iterations": self._fit.iterations,
            },
        }

    # -- the loop ---------------------------------------------------------

    def run(
        self,
        chooser_factory: Callable[[int], Callable[..., Any]] | None = None,
        *,
        human_policy_factory: Callable[[int], Policy] | None = None,
        print_fn: Callable[..., None] = print,
    ) -> dict[str, Any] | None:
        """Play the scheduled games, persist everything, return the report.

        Exactly one human-seat driver must be given.  ``chooser_factory(seat)``
        returns the ``(game, state, view)`` chooser passed to ``play_game``
        (interactive terminal or a test double); ``human_policy_factory(seat)``
        returns the :class:`Policy` placed in the seat's slot in the policy
        sequence (a scripted human).  Returns ``None`` when the player quit
        before finishing a single game.
        """
        if (chooser_factory is None) == (human_policy_factory is None):
            raise ValueError(
                "run() needs exactly one of chooser_factory / human_policy_factory"
            )
        self.directory.mkdir(parents=True, exist_ok=True)
        self._write_rungs()
        self._persist()
        reason: str | None = None
        while len(self._games) < self.config.games:
            reason = stop_reason(len(self._games), self.ci_half_width(), self.config)
            if reason is not None:
                break
            index = len(self._games)
            opponent = self.select_opponent()
            seat = self._seats[index]
            seed = self._seeds[index]
            print_fn(
                f"第 {index + 1}/{self.config.games} 局：",
                f"对手 {opponent.id}（{opponent.elo:.0f}），你坐 {seat} 号位",
            )
            try:
                trace, path = self._play_one(
                    opponent,
                    index=index,
                    seat=seat,
                    seed=seed,
                    human_policy=(
                        human_policy_factory(seat)
                        if human_policy_factory is not None
                        else None
                    ),
                    chooser=(
                        chooser_factory(seat) if chooser_factory is not None else None
                    ),
                    print_fn=print_fn,
                )
            except (QuitGame, KeyboardInterrupt):
                reason = "quit"
                break
            self._last_trace = trace
            self._record_game(
                opponent,
                seat=seat,
                seed=seed,
                scores=trace["final_scores"],
                trace_path=path,
            )
            self._refit()
            self._stopped_reason = stop_reason(
                len(self._games), self.ci_half_width(), self.config
            )
            self._persist()
            human = self._fit.ratings[self.config.human_id]  # type: ignore[union-attr]
            print_fn(
                f"  打完 {len(self._games)} 局：估计 {human.elo:.0f} ± "
                f"{human.se:.0f}（95% ± {self.ci_half_width():.0f}）"
            )
        else:
            reason = "max_games"
        self._stopped_reason = reason or "max_games"
        if not self._games:
            self._persist()
            return None
        report = self.report()
        self._write_json("report.json", report)
        self._persist()
        return report


# -- CLI ---------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="7g523-elo",
        description=(
            "10 局定级会话：轨迹先验 + 胜负/分差的 BT-MAP，输出点估计、诚实 CI、"
            "最近档与 provisional 标记（docs/human-elo-plan.md M2/M3）"
        ),
    )
    parser.add_argument(
        "--manifest",
        default="traces/pool10/manifest.json",
        help=(
            "研究 manifest（levels/anchors/subjects；默认 traces/pool10/manifest.json，"
            "2026-09-25 牌型族规则后新训 500k 池；旧 traces/study/manifest.json "
            "仅存历史标签）"
        ),
    )
    parser.add_argument(
        "--prior",
        default="artifacts/human-elo/prior.json",
        help="M1 轨迹先验产物（默认 artifacts/human-elo/prior.json）",
    )
    parser.add_argument(
        "--no-trace-prior",
        action="store_true",
        help="只跑结果似然（冷启动；不需要 numpy/M1 产物）",
    )
    parser.add_argument(
        "--sessions-dir",
        default="traces/sessions",
        help="会话目录的父目录（默认 traces/sessions）",
    )
    parser.add_argument("--session-id", default=None, help="会话 id（默认时间戳）")
    parser.add_argument("--games", type=int, default=10, help="局数（偶数，默认 10）")
    parser.add_argument("--seed", type=int, default=None, help="发牌/调度随机种子")
    parser.add_argument(
        "--seat-start", type=int, choices=(0, 1), default=0, help="首局座位（默认 0）"
    )
    parser.add_argument("--stop-ci", type=float, default=50.0, help="CI 半宽停止阈值")
    parser.add_argument(
        "--min-games",
        type=int,
        default=1,
        help="早停前至少完成的局数（HR §5.4 可选用 8）",
    )
    parser.add_argument(
        "--explore-games",
        type=int,
        default=2,
        help="前几局用 Thompson 探索（默认 2，HR §6.2）",
    )
    parser.add_argument(
        "--simulate",
        default=None,
        metavar="SPEC",
        help="非交互：用 random / greedy / ckpt:<path> 当“真人”跑完整会话",
    )
    parser.add_argument("--device", default="cpu", help="ckpt 对手的设备（默认 cpu）")
    parser.add_argument(
        "--verify-traces",
        action="store_true",
        help="每局特征提取前重放校验（默认关；在线路径刚打完不必重放）",
    )
    parser.add_argument("--quiet", action="store_true", help="静音对局过程输出")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    print_fn: Callable[..., None] = (lambda *a, **k: None) if args.quiet else print
    try:
        opponents, anchors = load_opponents(load_manifest(args.manifest))
        trace_prior = None if args.no_trace_prior else TracePrior.load(args.prior)
        session_id = args.session_id or new_session_id(args.sessions_dir)
        directory = Path(args.sessions_dir) / session_id
        session = PlacementSession(
            opponents=opponents,
            anchors=anchors,
            directory=directory,
            session_id=session_id,
            config=SessionConfig(
                games=args.games,
                stop_ci=args.stop_ci,
                min_games_before_stop=args.min_games,
                explore_games=args.explore_games,
                seat_start=args.seat_start,
            ),
            rng=random.Random(args.seed),
            trace_prior=trace_prior,
            feature_verify=args.verify_traces,
            device=args.device,
        )
        if args.simulate:
            human = policy_from_spec(
                args.simulate, session.rules, seed=args.seed, device=args.device
            )
            report = session.run(
                human_policy_factory=lambda _seat, _policy=human: _policy,
                print_fn=print_fn,
            )
        else:
            chooser_factory = lambda seat: interactive_chooser(seat, print_fn=print_fn)  # noqa: E731
            report = session.run(chooser_factory, print_fn=print_fn)
    except (FileNotFoundError, ValueError, KeyError, OSError) as exc:
        print(f"7g523-elo: {exc}", file=sys.stderr)
        return 1

    if report is None:
        print("没有完成任何一局，未生成报告。")
        return 1
    human = report["human"]
    nearest = report["nearest_level"] or {}
    flag = "临时 provisional" if report["provisional"] else "已定级"
    print(
        f"定级：{human['elo']:.0f} ± {human['ci_half_width']:.0f}（95% CI），"
        f"最近档 {nearest.get('id', '?')}（{nearest.get('elo', float('nan')):.0f}），"
        f"{flag}"
    )
    print(f"报告：{report['session']['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

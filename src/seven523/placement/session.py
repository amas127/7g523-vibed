"""The placement session state machine: schedule → play → fit → report.

Persists ``traces/sessions/<id>/`` without any ``--save-trace``: one trace per
game, ``session.json`` after every game, ``report.json`` at the end, and a small
``rungs.json`` sidecar with the opponent pool metadata.  Pure orchestration
around :func:`seven523.play.play_game`; ``run()`` takes any chooser factory so
tests and the ``--simulate`` CLI mode work without a TTY.

The loop is also exposed step by step (:meth:`PlacementSession.next_game` →
:meth:`~PlacementSession.opponent_policy` → :meth:`~PlacementSession.commit_game`
→ :meth:`~PlacementSession.finish`) for a driver that plays a game itself —
the browser table (``7g523-web``) drives one game at a time and cannot block a
chooser.  ``run()`` is written on top of the same methods so both paths share
one owner for scheduling, refitting and persistence.
"""
from __future__ import annotations

import json
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from ..elo import Fit, PlayedGame, Prior
from ..play import QuitGame
from ..policies import Policy, buildable_by_grammar, policy_from_spec
from ..prior import TracePrior, session_mean
from ..record import play_recorded, policy_seed
from ..rules import DEFAULT_RULES, Rules
from .estimator import (
    COLD_START_PRIOR,
    RESULT_PRIOR,
    SessionConfig,
    band_for,
    channel_weights,
    fit_session,
    nearest_level,
)
from .opponents import (
    Opponent,
    plan_deals,
    plan_seats,
    select_opponent,
    session_player_labels,
    stop_reason,
)

#: Document tags; readers reject anything else.
SESSION_SCHEMA = "seven523.placement-session"
REPORT_SCHEMA = "seven523.placement-report"
VERSION = 1


# -- session -----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GameRecord:
    """One completed placement game, as written to ``session.json``."""

    index: int
    seed: int
    seat: int
    opponent_id: str
    opponent_mu: float
    scores: tuple[int, ...]
    result: str
    trace: str
    mu_traj: float | None = None
    sigma_traj: float | None = None
    posterior_mu: float | None = None
    posterior_sigma: float | None = None
    #: Why this game's row was kept out of the trace channel (a search rung's
    #: μ is outside the raw prior's fitting domain); ``None`` = the row was
    #: eligible.  Per-game provenance so a mixed pool can never hide a
    #: silently extrapolated row (ADR-0013).
    prior_off_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "seed": self.seed,
            "seat": self.seat,
            "opponent_id": self.opponent_id,
            "opponent_mu": self.opponent_mu,
            "scores": list(self.scores),
            "result": self.result,
            "trace": self.trace,
            "mu_traj": self.mu_traj,
            "sigma_traj": self.sigma_traj,
            "posterior_mu": self.posterior_mu,
            "posterior_sigma": self.posterior_sigma,
            "prior_off_reason": self.prior_off_reason,
        }


@dataclass(frozen=True, slots=True)
class ScheduledGame:
    """One game of the session schedule: which rung, which seat, which deal."""

    index: int
    opponent: Opponent
    seat: int
    seed: int


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
        #: Why at least one played game kept its row out of the trace channel.
        #: A search rung's μ comes from its own joint fit and lies outside the
        #: raw trace-prior domain, so the channel is scoped per played row
        #: instead of silently extrapolating (ADR-0013).  ``None`` while every
        #: played row was trace-eligible (or no prior was configured).
        self.prior_off_reason: str | None = None
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
        return {opponent.id: opponent.mu for opponent in self.opponents}

    @property
    def games(self) -> tuple[PlayedGame, ...]:
        return tuple(self._games)

    @property
    def records(self) -> tuple[GameRecord, ...]:
        return tuple(self._records)

    @property
    def fit(self) -> Fit | None:
        return self._fit

    @property
    def stopped_reason(self) -> str | None:
        """Why the session stopped (``ci`` / ``max_games`` / ``quit``), if yet."""
        return self._stopped_reason

    def _cold_prior(self) -> Prior:
        if self.trace_prior is not None:
            return self.trace_prior.cold_start
        return COLD_START_PRIOR

    def posterior_prior(self) -> Prior:
        """The human posterior used to select the next opponent."""
        if self._fit is None:
            return self._cold_prior()
        rating = self._fit.ratings[self.config.human_id]
        return Prior(rating.mu, rating.sigma)

    def ci_half_width(self) -> float:
        if self._fit is None:
            return float("inf")
        return self.config.z * self._fit.ratings[self.config.human_id].sigma

    def select_opponent(self) -> Opponent:
        """Next rung under HR §6.2 (Thompson first, then info)."""
        return select_opponent(
            self.posterior_prior(),
            self.opponents,
            n_games=len(self._games),
            explore_games=self.config.explore_games,
            rng=self.rng,
        )

    # -- stepwise API (a driver that plays the game itself) -----------------

    def begin(self) -> None:
        """Prepare the session directory: write ``rungs.json`` and ``session.json``.

        ``run()`` calls this before the first game; a stepwise driver calls it
        once after construction.  Idempotent.
        """
        self.directory.mkdir(parents=True, exist_ok=True)
        self._persist()

    def next_game(self) -> ScheduledGame | None:
        """The next scheduled game, or ``None`` when a stop rule fires.

        Sets :attr:`stopped_reason` (``"ci"`` or ``"max_games"``) when it
        stops.  The returned schedule is passed back to
        :meth:`opponent_policy` / :meth:`player_labels` / :meth:`commit_game`.
        """
        index = len(self._games)
        reason = stop_reason(index, self.ci_half_width(), self.config)
        if reason is not None:
            self._stopped_reason = reason
            return None
        return ScheduledGame(
            index=index,
            opponent=self.select_opponent(),
            seat=self._seats[index],
            seed=self._seeds[index],
        )

    def opponent_policy(self, scheduled: ScheduledGame) -> Policy:
        """The opponent policy for ``scheduled`` (per-seat policy seed)."""
        return self._opponent_policy(
            scheduled.opponent, policy_seed(scheduled.seed, scheduled.seat)
        )

    def player_labels(self, scheduled: ScheduledGame) -> list[str]:
        """Trace ``players`` labels for ``scheduled`` (human + rung identity)."""
        return session_player_labels(
            human_id=self.config.human_id,
            human_seat=scheduled.seat,
            num_players=self.rules.num_players,
            opponent=scheduled.opponent,
        )

    def commit_game(
        self,
        scheduled: ScheduledGame,
        *,
        scores: Sequence[int],
        trace: Mapping[str, Any],
        trace_path: Path,
    ) -> None:
        """Record an externally played game, then refit and persist.

        The caller owns the game itself (the browser drives it step by step);
        this method owns everything after it ends: the record, the trace-prior
        row, the two-channel fit and ``session.json``.  ``scheduled`` must be
        the current :meth:`next_game` result.
        """
        if scheduled.index != len(self._games):
            raise ValueError(
                f"scheduled game {scheduled.index} is not next "
                f"({len(self._games)} played)"
            )
        self._record_game(
            scheduled.opponent,
            seat=scheduled.seat,
            seed=scheduled.seed,
            scores=scores,
            trace_path=trace_path,
        )
        self._last_trace = dict(trace)
        self._refit()
        self._stopped_reason = stop_reason(
            len(self._games), self.ci_half_width(), self.config
        )
        self._persist()

    def finish(self, reason: str | None = None) -> dict[str, Any] | None:
        """Close the session: write the report (if any game ran) and return it.

        ``reason`` defaults to what :meth:`next_game` / :meth:`commit_game`
        already recorded, then ``"max_games"``.  Returns ``None`` when no game
        was played; ``report.json`` is written only when there is a report.
        """
        self._stopped_reason = reason or self._stopped_reason or "max_games"
        if not self._games:
            self._persist()
            return None
        report = self.report()
        self._write_json("report.json", report)
        self._persist()
        return report

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

    @staticmethod
    def _is_search_rung(opponent: Opponent) -> bool:
        """Whether ``opponent`` is a search rung the stock grammar cannot build."""
        return opponent.spec is not None and not buildable_by_grammar(opponent.spec)

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
        # A search rung's μ is outside the raw trace-prior fitting domain: its
        # row is excluded from the trace channel with per-game provenance
        # while raw rows keep the prior (ADR-0013).  With no prior configured
        # there is nothing to protect and no reason is recorded.
        prior_off_reason: str | None = None
        if self.trace_prior is not None and self._is_search_rung(opponent):
            prior_off_reason = (
                f"第 {index + 1} 局对手 {opponent.id!r} 是搜索 rung"
                f"（spec={opponent.spec!r}）：其 μ 来自搜索联合测量，"
                "超出轨迹先验的 raw 拟合域；该局行不进入轨迹通道"
                "（ADR-0013，禁止静默外推），其余 raw 局仍使用轨迹先验"
            )
            if self.prior_off_reason is None:
                self.prior_off_reason = prior_off_reason
        self._records.append(
            GameRecord(
                index=index,
                seed=seed,
                seat=seat,
                opponent_id=opponent.id,
                opponent_mu=opponent.mu,
                scores=tuple(int(score) for score in scores),
                result=result,
                trace=trace_path.name,
                prior_off_reason=prior_off_reason,
            )
        )

    def _refit(self) -> None:
        record = self._records[-1]
        if self.trace_prior is not None and record.prior_off_reason is None:
            assert self._last_trace is not None
            row = self.trace_prior.features(self._last_trace, verify=self.feature_verify)
            self._rows.append((row, record.opponent_mu))
            self._mu_traj = session_mean(
                self.trace_prior.doc,
                [row for row, _mu in self._rows],
                opponent_elos=[mu for _row, mu in self._rows],
            )
            self._human_prior = self.trace_prior.prior_for_session(
                self._mu_traj, len(self._rows)
            )
        elif self.trace_prior is None:
            self._human_prior = self._cold_prior()
        centers = self.trace_prior.labels if self.trace_prior is not None else None
        self._fit = fit_session(
            self._games,
            human_id=self.config.human_id,
            human_prior=self._human_prior,
            opponents=self.opponents,
            anchors=self.anchors,
            rung_prior_sd=self.config.rung_prior_sd,
            rung_centers=centers,
        )
        self._result_fit = fit_session(
            self._games,
            human_id=self.config.human_id,
            human_prior=RESULT_PRIOR,
            opponents=self.opponents,
            anchors=self.anchors,
            rung_prior_sd=self.config.rung_prior_sd,
            rung_centers=centers,
        )
        posterior = self._fit.ratings[self.config.human_id]
        self._records[-1] = replace(
            self._records[-1],
            mu_traj=self._mu_traj,
            sigma_traj=self._human_prior.sigma if self.trace_prior is not None else None,
            posterior_mu=posterior.mu,
            posterior_sigma=posterior.sigma,
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
            "prior_off_reason": self.prior_off_reason,
            "opponents": [
                {
                    "id": opponent.id,
                    "mu": opponent.mu,
                    "sigma": opponent.sigma,
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
        self._write_rungs()

    def _write_rungs(self) -> None:
        self._write_json(
            "rungs.json",
            {
                "schema": "seven523.placement-rungs",
                "version": VERSION,
                "session": self.session_id,
                "anchors": dict(self.anchors),
                "prior_off_reason": self.prior_off_reason,
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
        ci_half_width = self.config.z * human.sigma
        # Only trace-eligible (raw) rows inform the trace channel: with zero
        # rows the channel contributes no precision even when the prior
        # artifact is loaded, because every played game was a search rung.
        trace_used = self.trace_prior is not None and bool(self._rows)
        weights = channel_weights(
            self._human_prior.sigma if trace_used else None,
            result.sigma,
        )
        nearest = nearest_level(human.mu, self.levels)
        rungs: list[dict[str, Any]] = []
        for opponent in self.opponents:
            fitted = self._fit.ratings.get(opponent.id)
            rungs.append(
                {
                    "id": opponent.id,
                    "anchor": opponent.anchor,
                    "spec": opponent.spec,
                    "mu": opponent.mu,
                    "sigma": opponent.sigma,
                    "prior_center": (
                        None
                        if opponent.anchor
                        else float(
                            (self.trace_prior.labels if self.trace_prior else {}).get(
                                opponent.id, opponent.mu
                            )
                        )
                    ),
                    "prior_sd": None if opponent.anchor else self.config.rung_prior_sd,
                    "fitted_mu": None if fitted is None else fitted.mu,
                    "fitted_sigma": None if fitted is None else fitted.sigma,
                    "games": None if fitted is None else fitted.n,
                }
            )
        return {
            "schema": REPORT_SCHEMA,
            "version": VERSION,
            "prior_off_reason": self.prior_off_reason,
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
                "mu": human.mu,
                "sigma": human.sigma,
                "ci95": [human.mu - ci_half_width, human.mu + ci_half_width],
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
                        self._human_prior.sigma if trace_used else None
                    ),
                    "n_traces": len(self._rows),
                    "excluded_games": [
                        record.index
                        for record in self._records
                        if record.prior_off_reason is not None
                    ],
                    "prior_off_reason": self.prior_off_reason,
                    "artifact": (
                        self.trace_prior.meta() if self.trace_prior is not None else None
                    ),
                    "weight": weights["trace"],
                },
                "result": {
                    "enabled": True,
                    "sigma": result.sigma,
                    "games": len(self._games),
                    "prior": {"mu": RESULT_PRIOR.mu, "sigma": RESULT_PRIOR.sigma},
                    "weight": weights["result"],
                },
                "weights": weights,
            },
            "rungs": rungs,
            "plan": {
                "seats": list(self._seats),
                "seeds": list(self._seeds),
                "played": [record.to_dict() for record in self._records],
            },
            "fit": {
                "estimator": "openskill-plackett-luce",
                "games": self._fit.games,
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

        Implemented on the stepwise API (:meth:`next_game` → :meth:`commit_game`
        → :meth:`finish`), which the browser driver uses directly.
        """
        if (chooser_factory is None) == (human_policy_factory is None):
            raise ValueError(
                "run() needs exactly one of chooser_factory / human_policy_factory"
            )
        self.begin()
        reason: str | None = None
        while True:
            scheduled = self.next_game()
            if scheduled is None:
                break
            print_fn(
                f"第 {scheduled.index + 1}/{self.config.games} 局：",
                f"对手 {scheduled.opponent.id}（{scheduled.opponent.mu:.0f}），"
                f"你坐 {scheduled.seat} 号位",
            )
            try:
                trace, path = self._play_one(
                    scheduled.opponent,
                    index=scheduled.index,
                    seat=scheduled.seat,
                    seed=scheduled.seed,
                    human_policy=(
                        human_policy_factory(scheduled.seat)
                        if human_policy_factory is not None
                        else None
                    ),
                    chooser=(
                        chooser_factory(scheduled.seat)
                        if chooser_factory is not None
                        else None
                    ),
                    print_fn=print_fn,
                )
            except (QuitGame, KeyboardInterrupt):
                reason = "quit"
                break
            self.commit_game(
                scheduled,
                scores=trace["final_scores"],
                trace=trace,
                trace_path=path,
            )
            human = self._fit.ratings[self.config.human_id]  # type: ignore[union-attr]
            print_fn(
                f"  打完 {len(self._games)} 局：估计 {human.mu:.0f} ± "
                f"{human.sigma:.0f}（95% ± {self.ci_half_width():.0f}）"
            )
        return self.finish(reason)

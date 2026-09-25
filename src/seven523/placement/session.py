"""The placement session state machine: schedule → play → fit → report.

Persists ``traces/sessions/<id>/`` without any ``--save-trace``: one trace per
game, ``session.json`` after every game, ``report.json`` at the end, and a small
``rungs.json`` sidecar with the opponent pool metadata.  Pure orchestration
around :func:`seven523.play.play_game`; ``run()`` takes any chooser factory so
tests and the ``--simulate`` CLI mode work without a TTY.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..elo import Fit, PlayedGame, Prior
from .estimator import (
    COLD_START_PRIOR,
    RESULT_PRIOR,
    SessionConfig,
    band_for,
    channel_weights,
    fit_session,
    nearest_level,
    session_margin,
)
from .opponents import (
    Opponent,
    plan_deals,
    plan_seats,
    select_opponent,
    session_player_labels,
    stop_reason,
)
from ..policies import Policy, policy_from_spec
from ..play import QuitGame
from ..prior import TracePrior, session_mean
from ..record import play_recorded, policy_seed
from ..rules import DEFAULT_RULES, Rules

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
            self._mu_traj = session_mean(
                self.trace_prior.doc,
                [row for row, _elo in self._rows],
                opponent_elos=[elo for _row, elo in self._rows],
            )
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

"""The web table session: one human, free play or a placement session.

:class:`TableSession` is the stateful core behind ``7g523-web``.  It wraps the
*real* engine and the *real* placement pipeline:

* **free play** — one game against a chosen pool rung; the human seat is driven
  externally through :meth:`TableSession.apply_action`, the opponent seats by
  their :class:`~seven523.policies.Policy`.  The base model is the chosen
  opponent and the optional plugin's search wrapper still takes per-game t/K.
* **placement** — a full :class:`~seven523.placement.PlacementSession` driven
  one game at a time through its stepwise API (``next_game`` → ``commit_game``
  → ``finish``), so the browser gets the same scheduling, trace prior,
  OpenSkill fit and on-disk artifacts as the ``7g523-elo`` CLI.  The pool is
  the manifest's ``levels`` — raw rungs and search rungs together in the same
  joint-fit scale — capability-gated by the injected factory: stock web skips
  a search rung with a warning, plugin web builds it from the manifest's
  ``search_config``.  Thompson/Fisher may schedule any buildable level.

A played search rung's μ lies outside the raw trace-prior fitting domain, so
``PlacementSession`` scopes the trace channel per played row: raw games keep
the prior, a search-rung game's row is excluded with recorded provenance
(``prior_off_reason``), never silently extrapolated (ADR-0013).  The session
artifact itself is the ordinary ``traces/sessions/<id>/`` document.

Free play builds its opponent through :meth:`TableSession.free_play_policy`,
whose optional :attr:`WebConfig.policy_factory` seam lets a run-local plugin
wrap ``ckpt:`` specs (e.g. the O4-lite search deployment) without this module
importing it.

The browser never sees the engine: every read goes through :meth:`snapshot`,
which projects the human's :class:`~seven523.game.View` (ADR-0002) into the
JSON shape :mod:`seven523.web.view` owns.  All mutations are serialised by an
internal lock because the HTTP layer is threaded; games are short and the
human is the only driver.
"""
from __future__ import annotations

import math
import random
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ..actions import resolve
from ..cards import card_key
from ..game import Game, GameState, StepResult, View
from ..match import Match, TurnHook
from ..placement import (
    Opponent,
    PlacementSession,
    ScheduledGame,
    SessionConfig,
    load_opponents,
    new_session_id,
    session_player_labels,
)
from ..policies import Policy, buildable_by_grammar, policy_from_spec
from ..prior import TracePrior
from ..record import policy_seed
from ..rules import DEFAULT_RULES, RULES_REVISION, Rules
from ..study import load_manifest
from ..trace import (
    build_trace,
    initial_snapshot,
    save_trace,
    step_record,
    trace_filename,
)
from ..twin import (
    DEFAULT_BOOTSTRAP,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_CONFIDENCE,
    DEFAULT_PAIRS,
    MIN_PAIRS,
    TwinSession,
    new_twin_id,
)
from .view import (
    card_json,
    combo_json,
    combo_label,
    counter_json,
    hand_key,
    legal_actions,
    play_json,
    seat_names,
)

__all__ = [
    "PROTO_VERSION",
    "TableSession",
    "WebConfig",
    "WebError",
    "torch_available",
]

#: Bumped whenever the JSON API gains a field the page depends on.  The page
#: warns when it talks to an older server process (static files hot-reload,
#: the Python process does not), so a stale server cannot silently change
#: behaviour.  v3: ``/api/config`` gained ``search``.  v4: ``/api/config``
#: gained ``twin`` and ``/api/continue`` became ``continue_round``.  v5:
#: ``/api/config`` gained ``plugin`` / ``rated_rungs`` / ``unrated`` and the
#: snapshot gained ``series``.  v6: the rated/unrated paths were deleted and
#: placement uses the one merged manifest pool.  v7: the free-play search
#: wrapper is single-model and its ``options`` may declare float knobs
#: (``outcome_blend`` / beta).  v8: free play gained the opt-in live
#: ``estimate`` readout (snapshot field + start flag).
PROTO_VERSION = 8


class WebError(Exception):
    """A client-visible HTTP error; the server maps ``status`` onto the reply."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


def torch_available() -> bool:
    """Whether ``ckpt:`` opponents can be loaded in this interpreter."""
    import importlib.util

    return importlib.util.find_spec("torch") is not None


def require_torch(spec: str | None, label: str) -> None:
    """Refuse a torch-backed opponent early when torch is missing.

    ``ckpt:`` rungs need torch for the neural policy; a spec outside the
    stock grammar (an admitted ``rolloutt:`` search rung) needs it for the
    injected factory's value core.
    """
    if not spec or torch_available():
        return
    if spec.startswith("ckpt:") or not buildable_by_grammar(spec):
        raise WebError(
            f"对手 {label} 需要 torch（本解释器找不到 torch；"
            f"用 uv run --group train … 启动）"
        )


@dataclass(frozen=True, slots=True)
class WebConfig:
    """Where the web table reads its pool/prior from and writes artifacts to."""

    manifest: Path
    prior: Path | None = None
    device: str = "cpu"
    #: Placement sessions land here, one ``session-<stamp>/`` directory each.
    sessions_dir: Path = Path("traces/sessions")
    #: Twin sessions land here, one ``twin-<stamp>/`` directory each.
    twins_dir: Path = Path("traces/twins")
    #: Free-play traces land under ``free_traces_dir/<run-id>/``.
    free_traces_dir: Path = Path("traces/web")
    #: Server RNG seed for dealing; ``None`` is random.
    seed: int | None = None
    #: Free-play policy seam: ``(spec, rules, policy_seed, search) -> Policy``.
    #: ``None`` means the shared :func:`~seven523.policies.policy_from_spec`
    #: grammar on ``device``; a run-local search launcher injects its wrapper
    #: factory here and receives the validated per-game search parameters.
    policy_factory: Callable[[str, Rules, int, Mapping[str, Any]], Policy] | None = None
    #: Metadata for the injected search wrapper (``None`` = raw pool).  The page
    #: renders ``label`` / ``note`` / ``presets`` and validates against
    #: ``options`` (``key -> {min, max, default, type?}``; ``type="float"``
    #: admits fractional values such as the outcome blend); the server treats
    #: only ``options`` structurally, the rest is opaque JSON.
    search: Mapping[str, Any] | None = None
    #: Metadata for the deal-twin mode (``None`` = twin unavailable).  The
    #: run-local launcher pins both arms; the server reads only the arm ids,
    #: specs and search parameters and refuses any request that tries to
    #: override them.
    twin: Mapping[str, Any] | None = None
    #: Optional capability predicate for manifest specs (``None`` = no gate);
    #: passed through to :func:`~seven523.placement.load_opponents`.
    can_build_spec: Callable[[str | None], bool] | None = None
    #: Opaque plugin provenance block for ``/api/config`` (source/error/…).
    plugin: Mapping[str, Any] | None = None


@dataclass
class _CurrentGame:
    """One step-driven placement game in flight."""

    scheduled: ScheduledGame
    match: Match
    record: dict[str, Any]


class TableSession:
    """One live human table: at most one free game or placement session."""

    def __init__(
        self,
        web_config: WebConfig,
        *,
        opponents: Sequence[Opponent],
        anchors: Mapping[str, float],
        prior: TracePrior | None,
        search_configs: Mapping[str, Mapping[str, Any]] | None = None,
        rules: Rules = DEFAULT_RULES,
        rng: random.Random | None = None,
        run_id: str | None = None,
    ) -> None:
        if not opponents:
            raise ValueError("the web table needs at least one opponent")
        self.web_config = web_config
        self.rules = rules
        self.opponents = tuple(opponents)
        #: Measured search identities per opponent id (free-play defaults).
        self.search_configs = {
            str(id_): dict(config)
            for id_, config in (search_configs or {}).items()
        }
        self.anchors = {str(key): float(value) for key, value in anchors.items()}
        self.prior = prior
        self.rng = rng or random.Random(web_config.seed)
        self.run_id = run_id or datetime.now().strftime("run-%Y%m%d-%H%M%S")
        self.opponent_by_id = {opponent.id: opponent for opponent in self.opponents}
        self.lock = threading.RLock()
        self.manifest_warnings: list[str] = []
        self.reset()

    @classmethod
    def from_config(cls, web_config: WebConfig) -> "TableSession":
        """Load the opponent manifest (and prior, when configured) from disk."""
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            # A server without an injected capability gate can build exactly
            # the stock ``random``/``ckpt:`` grammar: anything else (a
            # manifest ``rolloutt:`` search rung) is skipped with a warning by
            # ``load_opponents`` instead of crashing mid-session.
            manifest = load_manifest(web_config.manifest)
            opponents, anchors = load_opponents(
                manifest,
                rules=DEFAULT_RULES,
                can_build=web_config.can_build_spec or buildable_by_grammar,
            )
        search_configs = {
            str(entry["id"]): entry["search_config"]
            for entry in manifest.get("subjects") or []
            if isinstance(entry, Mapping)
            and entry.get("id") is not None
            and "search_config" in entry
        }
        prior = None
        if web_config.prior is not None:
            prior = TracePrior.load(web_config.prior)
        session = cls(
            web_config,
            opponents=opponents,
            anchors=anchors,
            prior=prior,
            search_configs=search_configs,
        )
        session.manifest_warnings = [str(warning.message) for warning in caught]
        return session

    # -- lifecycle ---------------------------------------------------------

    def reset(self) -> None:
        """Drop the live game; keep the loaded pool/prior."""
        self.mode: str | None = None
        self.phase = "idle"
        self.error: str | None = None
        self.match: Match | None = None
        self.game: Game | None = None
        self.view: View | None = None
        self.state: GameState | None = None
        self.human_seat = 0
        self.opponent: Opponent | None = None
        self.seed: int | None = None
        self.names: list[str] = ["你", "对手"]
        self.log: list[dict[str, Any]] = []
        self.record: dict[str, Any] | None = None
        self.trace_path: str | None = None
        self.result: dict[str, Any] | None = None
        self.turn = 0
        self.tricks_completed = 0
        self.game_index = 0
        self.games_total = 0
        self.driver: _PlacementDriver | None = None
        self.session: PlacementSession | None = None
        self.twin_driver: _TwinDriver | None = None
        self.twin_session: TwinSession | None = None
        self.report: dict[str, Any] | None = None
        self.stopped_reason: str | None = None
        #: Effective free-play search parameters for the live game.
        self.search: dict[str, float] = {}
        #: The built opponent policy of the live free game (the live estimate
        #: readout asks it for the raw critic/outcome heads).
        self.free_policy: Policy | None = None
        #: Whether the live free game opted into the model estimate panel.
        self.show_estimate = False

    # -- config ------------------------------------------------------------

    def config_document(self) -> dict[str, Any]:
        """The ``GET /api/config`` payload: the pool, prior and capabilities."""
        with self.lock:
            opponents = [
                {
                    "id": opponent.id,
                    "mu": opponent.mu,
                    "sigma": opponent.sigma,
                    "anchor": bool(opponent.anchor),
                    "spec": opponent.spec,
                    **(
                        {"search_config": dict(self.search_configs[opponent.id])}
                        if opponent.id in self.search_configs
                        else {}
                    ),
                }
                for opponent in self.opponents
            ]
            return {
                "proto_version": PROTO_VERSION,
                "manifest": str(self.web_config.manifest),
                "rules_revision": RULES_REVISION,
                "num_players": self.rules.num_players,
                "opponents": opponents,
                "anchors": dict(self.anchors),
                "games_options": [2, 4, 6, 10],
                "prior": (
                    None
                    if self.prior is None
                    else {"path": str(self.web_config.prior), **self.prior.meta()}
                ),
                "torch": torch_available(),
                "search": (
                    None
                    if self.web_config.search is None
                    else dict(self.web_config.search)
                ),
                "twin": (
                    None
                    if self.web_config.twin is None
                    else dict(self.web_config.twin)
                ),
                "plugin": (
                    None
                    if self.web_config.plugin is None
                    else dict(self.web_config.plugin)
                ),
                "warnings": list(self.manifest_warnings),
            }

    # -- starting sessions ---------------------------------------------------

    def start(self, request: Mapping[str, Any]) -> None:
        """Start a free game or a placement session from a request body."""
        with self.lock:
            mode = str(request.get("mode") or "")
            if mode == "free":
                self._start_free(request)
            elif mode == "placement":
                self._start_placement(request)
            elif mode == "twin":
                self._start_twin(request)
            else:
                raise WebError(f"unknown mode {mode!r}")

    def _seed_from(self, request: Mapping[str, Any]) -> int:
        raw = request.get("seed")
        if raw in (None, ""):
            return self.rng.randrange(1 << 32)
        try:
            return int(raw)
        except (TypeError, ValueError) as exc:
            raise WebError(f"seed 需要是整数：{raw!r}") from exc

    def normalize_search(
        self,
        raw: Any,
        *,
        pinned: Mapping[str, Any] | None = None,
    ) -> dict[str, float]:
        """Validate a free-play ``search`` request against the launcher options.

        ``WebConfig.search["options"]`` declares ``key -> {min, max, default}``
        (plus ``type: "float"`` for fractional knobs); every declared key is
        returned with the request value or a default, so the factory always
        sees one complete parameter document.  ``pinned`` (the selected
        subject's published ``search_config``) supplies the default for a
        measured search rung, so omitting a knob plays the rung's measured
        identity while an explicit request value still overrides it.  A
        request with an unknown key or an out-of-range / wrongly-typed value
        is refused before any game state changes.
        """
        meta = self.web_config.search
        if meta is None:
            if raw:
                raise WebError(
                    "服务端未启用搜索包装：用 runs/o4lite-search/web_search.py 启动"
                )
            return {}
        options = meta.get("options") or {}
        if raw is None:
            raw = {}
        if not isinstance(raw, Mapping):
            raise WebError(f"search 需要是对象：{raw!r}")
        unknown = set(raw) - set(options)
        if unknown:
            raise WebError(f"未知搜索参数：{sorted(unknown)}")
        pinned = pinned or {}
        effective: dict[str, float] = {}
        for key, spec in options.items():
            value = raw.get(key, pinned.get(key, spec.get("default")))
            if isinstance(value, bool):
                raise WebError(f"搜索参数 {key} 需要是数字：{value!r}")
            if str(spec.get("type") or "int") == "float":
                try:
                    number = float(value)
                except (TypeError, ValueError) as exc:
                    raise WebError(
                        f"搜索参数 {key} 需要是数字：{value!r}"
                    ) from exc
                low, high = float(spec["min"]), float(spec["max"])
                if not math.isfinite(number) or not low <= number <= high:
                    raise WebError(
                        f"搜索参数 {key} 越界：{number:g}（允许 {low:g}..{high:g}）"
                    )
                effective[key] = number
                continue
            try:
                number = int(value)
            except (TypeError, ValueError) as exc:
                raise WebError(f"搜索参数 {key} 需要是整数：{value!r}") from exc
            low, high = int(spec["min"]), int(spec["max"])
            if not low <= number <= high:
                raise WebError(f"搜索参数 {key} 越界：{number}（允许 {low}..{high}）")
            effective[key] = number
        return effective

    def free_play_policy(
        self,
        spec: str,
        seed: int,
        search: Mapping[str, Any] | None = None,
    ) -> Policy:
        """Build one free-play opponent policy from its spec.

        The single seam the search launcher injects through: the default is
        the shared ``random`` / ``ckpt:<path>`` grammar, an injected factory
        may wrap ``ckpt:`` specs with the per-game search parameters (see
        :attr:`WebConfig.policy_factory`).
        """
        factory = self.web_config.policy_factory
        if factory is not None:
            return factory(spec, self.rules, seed, dict(search or {}))
        return policy_from_spec(
            spec, self.rules, seed=seed, device=self.web_config.device
        )

    def _manifest_policy_factory(
        self,
    ) -> Callable[[Opponent, int], Policy] | None:
        """Per-opponent factory for the shared placement pool.

        Only specs outside the stock grammar (search rungs admitted through
        :attr:`WebConfig.can_build_spec`) go through the injected factory with
        its pinned deployment defaults; raw rungs and the anchor keep
        :func:`~seven523.policies.policy_from_spec` so a rated raw opponent is
        never silently wrapped in search.  ``None`` when no factory is
        injected (the pool then contains only grammar-buildable specs).
        """
        factory = self.web_config.policy_factory
        if factory is None:
            return None
        rules = self.rules
        device = self.web_config.device

        def build(opponent: Opponent, seed: int) -> Policy:
            spec = opponent.spec
            if spec is None:
                raise ValueError(
                    f"opponent {opponent.id!r} has no policy spec in the manifest"
                )
            if buildable_by_grammar(spec):
                return policy_from_spec(spec, rules, seed=seed, device=device)
            # Search rungs carry no per-game overrides here: the injected
            # factory owns the manifest's pinned ``search_config`` and builds
            # the exact measured identity (the ``search`` mapping is empty).
            return factory(spec, rules, seed, {})

        return build

    def _start_free(self, request: Mapping[str, Any]) -> None:
        self.reset()
        opponent_id = str(request.get("opponent_id") or "random")
        opponent = self.opponent_by_id.get(opponent_id)
        if opponent is None:
            raise WebError(f"未知对手 {opponent_id!r}")
        require_torch(opponent.spec, opponent.id)
        try:
            seat = int(request.get("seat", 0))
        except (TypeError, ValueError) as exc:
            raise WebError("seat 需要是 0 或 1") from exc
        if not 0 <= seat < self.rules.num_players:
            raise WebError(f"seat 越界：{seat}")
        seed = self._seed_from(request)
        search = self.normalize_search(
            request.get("search"), pinned=self.search_configs.get(opponent.id)
        )
        show_estimate = request.get("estimate", False)
        if not isinstance(show_estimate, bool):
            raise WebError(f"estimate 需要是布尔值：{show_estimate!r}")
        policy = self.free_play_policy(
            opponent.spec or "random", policy_seed(seed, seat), search
        )
        policies: list[Policy | None] = [
            None if index == seat else policy for index in range(self.rules.num_players)
        ]
        self.mode = "free"
        self.human_seat = seat
        self.opponent = opponent
        self.seed = seed
        self.search = search
        self.free_policy = policy
        self.show_estimate = show_estimate
        self.names = seat_names(seat, self.rules.num_players, opponent.id)
        self.game_index = 1
        self.games_total = 1
        self.begin_match(Match(self.rules, policies, rng=random.Random(seed)))
        self.advance()

    def _start_placement(self, request: Mapping[str, Any]) -> None:
        """Start a placement session over the one merged manifest pool.

        ``self.opponents`` is the gate-filtered manifest ``levels``: anchors,
        raw rungs and buildable search rungs in the same joint-fit scale.
        Thompson/Fisher may pick any of them (owner-accepted C3); the trace
        prior stays configured and ``PlacementSession`` excludes only the
        played search-rung rows with recorded provenance (ADR-0013).
        """
        self.reset()
        if request.get("search_rung") not in (None, ""):
            raise WebError(
                "search_rung 路径已删除：定级池统一使用 manifest levels，"
                "搜索 rung 与其他档位一起被 Thompson/Fisher 调度"
            )
        try:
            games = int(request.get("games", 10))
        except (TypeError, ValueError) as exc:
            raise WebError("games 需要是偶数") from exc
        use_prior = bool(request.get("use_prior", self.prior is not None))
        if use_prior and self.prior is None:
            raise WebError("服务端未加载轨迹先验（--no-trace-prior）")
        if not torch_available():
            for opponent in self.opponents:
                if opponent.spec and (
                    opponent.spec.startswith("ckpt:")
                    or not buildable_by_grammar(opponent.spec)
                ):
                    raise WebError(
                        "定级池含 ckpt/搜索对手，需要 torch；"
                        "用 uv run --group train … 启动"
                    )
        seed = self._seed_from(request)
        try:
            config = SessionConfig(games=games)
        except ValueError as exc:
            raise WebError(str(exc)) from exc
        self.mode = "placement"
        self.games_total = games
        self.driver = _PlacementDriver(
            table=self,
            config=config,
            use_prior=use_prior,
            seed=seed,
            policy_factory=self._manifest_policy_factory(),
        )
        self.session = self.driver.session
        self.driver.start_next()
        if self.phase == "idle":  # a stop fired before the first game
            self.phase = "session_over"

    def _start_twin(self, request: Mapping[str, Any]) -> None:
        """Start a deal-twin session from the launcher-pinned metadata."""
        self.reset()
        meta = self.web_config.twin
        if meta is None:
            raise WebError(
                "twin 模式需要 runs/o4lite-search/web_twin.py 启动器"
            )
        if not torch_available():
            raise WebError(
                "twin 模式需要 torch；用 runs/o4lite-search/web_twin.py 启动器"
            )
        if "opponent_id" in request or "search" in request:
            raise WebError(
                "twin 的两臂由启动器固定：请求里不能带 opponent_id / search"
            )
        raw_pairs = request.get("pairs", meta.get("default_pairs", DEFAULT_PAIRS))
        try:
            pairs = int(raw_pairs)
        except (TypeError, ValueError) as exc:
            raise WebError("pairs 需要是偶数") from exc
        if pairs % 2 != 0 or not 2 <= pairs <= 100:
            raise WebError(f"pairs 需要是 2..100 之间的偶数：{pairs}")
        seed = self._seed_from(request)
        search_meta = dict(meta.get("search") or {})
        raw_meta = dict(meta.get("raw") or {})
        base_spec = str(search_meta.get("base_spec") or raw_meta.get("spec") or "")
        directory = self.web_config.twins_dir / new_twin_id(
            self.web_config.twins_dir
        )
        try:
            session = TwinSession(
                base_spec=base_spec,
                raw_id=str(raw_meta.get("id") or "raw"),
                search_id=str(search_meta.get("id") or "search"),
                search_params=dict(search_meta.get("params") or {}),
                directory=directory,
                session_id=directory.name,
                search_factory=self.web_config.policy_factory,
                pairs=pairs,
                seat_start=int(meta.get("seat_start", 0)),
                order_start=str(meta.get("order_start", "raw")),
                rules=self.rules,
                rng=random.Random(seed),
                rng_seed=seed,
                bootstrap=int(meta.get("bootstrap", DEFAULT_BOOTSTRAP)),
                bootstrap_seed=int(
                    meta.get("bootstrap_seed", DEFAULT_BOOTSTRAP_SEED)
                ),
                confidence=float(meta.get("confidence", DEFAULT_CONFIDENCE)),
                min_pairs=int(meta.get("min_pairs", MIN_PAIRS)),
                device=self.web_config.device,
                raw_label=raw_meta.get("label") or None,
                search_label=search_meta.get("label") or None,
                search_identity=search_meta.get("identity") or None,
            )
        except ValueError as exc:
            raise WebError(str(exc)) from exc
        self.twin_driver = _TwinDriver(table=self, session=session)
        self.mode = "twin"
        self.games_total = 2 * pairs
        self.twin_session = session
        self.twin_driver.start_next()
        if self.phase == "idle":  # an immediately exhausted plan
            self.phase = "session_over"

    # -- per-game plumbing ---------------------------------------------------

    def begin_match(self, match: Match, record: dict[str, Any] | None = None) -> None:
        """Attach a new match and start recording it as a trace."""
        self.match = match
        self.game = match.game
        if record is None:
            record = {
                "initial": initial_snapshot(match.state),
                "steps": [],
            }
        self.record = record
        match.on_turn = self._make_on_turn(match, record)
        self.tricks_completed = 0
        self.capture(match)
        self.turn += 1

    def _make_on_turn(self, match: Match, record: dict[str, Any]) -> TurnHook:
        def on_turn(
            seat: int, action_id: int, suit: int | None, view: View, result: StepResult
        ) -> None:
            action = match.game.catalog[action_id]
            if action.is_pass:
                text = "过"
            else:
                combo = resolve(action, view.hand, match.game.rules, suit=suit)
                text = combo_label(combo) if combo is not None else action.label
            record["steps"].append(
                step_record(
                    seat,
                    action_id,
                    suit,
                    text=text,
                    state=match.state,
                    result=result,
                )
            )
            with self.lock:
                kind_key = (
                    "pass" if action.is_pass else action.kind.name.lower()  # type: ignore[union-attr]
                )
                self.log.append(
                    {
                        "type": "play",
                        "seat": seat,
                        "text": text,
                        "pass": action.is_pass,
                        "kind_key": kind_key,
                        "step": len(record["steps"]),
                    }
                )
                if result.trick_over:
                    self.tricks_completed += 1
                    self.log.append(
                        {
                            "type": "trick",
                            "winner": result.winner,
                            "points": result.points_taken,
                            "refilled": list(result.refilled),
                            "dug": bool(result.dug),
                        }
                    )
                self.log = self.log[-80:]

        return on_turn

    def capture(self, match: Match) -> None:
        """Remember the human's projection of the live state."""
        self.game = match.game
        self.state = match.state
        self.view = match.game.view(match.state, self.human_seat)

    def advance(self) -> None:
        """Run policy seats until the human is to move (or the game is over)."""
        if self.match is None:
            raise WebError("没有进行中的对局", status=409)
        self.match.advance(stop=self.human_seat)
        self.capture(self.match)
        self.turn += 1
        if self.match.done:
            self._finish_game()
        else:
            self.phase = "human"

    # -- human actions -------------------------------------------------------

    def apply_action(self, request: Mapping[str, Any]) -> None:
        """Apply the human's action, then run the opponents to the next turn."""
        with self.lock:
            if self.phase != "human" or self.match is None:
                raise WebError("现在不是你的回合", status=409)
            raw = request.get("action_id")
            try:
                action_id = int(raw)  # type: ignore[arg-type]
            except (TypeError, ValueError) as exc:
                raise WebError(f"action_id 需要是整数：{raw!r}") from exc
            suit = request.get("suit")
            suit_value: int | None
            if suit is None:
                suit_value = None
            else:
                try:
                    suit_value = int(suit)
                except (TypeError, ValueError) as exc:
                    raise WebError(f"suit 需要是整数：{suit!r}") from exc
                if not 0 <= suit_value <= 3:
                    raise WebError(f"suit 越界：{suit_value}")
            try:
                self.match.step(action_id, suit_value)
            except (ValueError, RuntimeError) as exc:
                raise WebError(f"非法出牌：{exc}") from exc
            self.advance()

    def continue_round(self) -> None:
        """Start the next game of a placement **or** twin session."""
        with self.lock:
            if self.mode == "placement":
                self.continue_placement()
            elif self.mode == "twin":
                self.continue_twin()
            else:
                raise WebError("当前没有待继续的会话")

    def continue_placement(self) -> None:
        """Start the next placement game after a round-over screen."""
        with self.lock:
            if self.mode != "placement" or self.driver is None:
                raise WebError("没有进行中的定级会话")
            if self.phase != "round_over":
                raise WebError("当前没有待继续的牌局")
            self.driver.start_next()
            if self.phase == "idle":
                self.phase = "session_over"

    def continue_twin(self) -> None:
        """Start the next twin game after a round-over screen."""
        with self.lock:
            if self.mode != "twin" or self.twin_driver is None:
                raise WebError("没有进行中的 twin 会话")
            if self.phase != "round_over":
                raise WebError("当前没有待继续的牌局")
            self.twin_driver.start_next()
            if self.phase == "idle":
                self.phase = "session_over"

    def quit_session(self) -> None:
        """Abandon the live game (and close a placement/twin session, if any)."""
        with self.lock:
            if self.mode == "placement" and self.driver is not None:
                self.driver.abandon_current()
                if not self.driver.finished:
                    self.driver.finish("quit")
                elif self.phase != "session_over":
                    self.phase = "session_over"
            elif self.mode == "twin" and self.twin_driver is not None:
                self.twin_driver.abandon_current()
                if not self.twin_driver.finished:
                    self.twin_driver.finish("quit")
                elif self.phase != "session_over":
                    self.phase = "session_over"
            else:
                self.phase = "aborted"
                self.result = {
                    "aborted": True,
                    "scores": list(self.state.scores) if self.state else [],
                }

    # -- finishing -----------------------------------------------------------

    def _finish_game(self) -> None:
        if self.mode == "placement":
            if self.driver is None:
                raise WebError("定级会话状态不一致", status=500)
            self.driver.finish_game()
        elif self.mode == "twin":
            if self.twin_driver is None:
                raise WebError("twin 会话状态不一致", status=500)
            self.twin_driver.finish_game()
        else:
            self._finish_free()

    def _finish_free(self) -> None:
        if self.match is None or self.record is None or self.opponent is None:
            raise WebError("对局状态不一致", status=500)
        match = self.match
        record = self.record
        record["final_scores"] = list(match.state.scores)
        if self.search:
            # Free-play annotation: which (t, K, beta) this game faced.  Batch
            # traces never carry it; replays ignore unknown top-level keys.
            record["opponent_search"] = dict(self.search)
        trace = build_trace(
            self.rules,
            seed=self.seed,
            human_seat=self.human_seat,
            players=session_player_labels(
                human_id="human",
                human_seat=self.human_seat,
                num_players=self.rules.num_players,
                opponent=self.opponent,
            ),
            created_at=datetime.now().isoformat(timespec="seconds"),
            **record,
        )
        directory = self.web_config.free_traces_dir / self.run_id
        path = save_trace(
            directory
            / trace_filename(0, self.seed or 0, self.human_seat, self.opponent.id),
            trace,
        )
        self.trace_path = str(path)
        scores = list(match.state.scores)
        best = max(scores)
        winners = [seat for seat, score in enumerate(scores) if score == best]
        if len(winners) > 1:
            outcome = "平局"
        elif winners[0] == self.human_seat:
            outcome = "你赢了！"
        else:
            outcome = f"{self.names[winners[0]]}赢了"
        self.result = {
            "scores": scores,
            "winner": winners[0] if len(winners) == 1 else None,
            "outcome": outcome,
            "trace_path": self.trace_path,
            "steps": len(record["steps"]),
        }
        self.phase = "game_over"
        self.view = match.game.view(match.state, self.human_seat)

    # -- snapshot ------------------------------------------------------------

    def free_play_estimate(self) -> dict[str, Any] | None:
        """The opponent model's live one-pass readout for the human's seat.

        ``None`` unless the free-play request opted in and the built policy
        carries a value/outcome readout (a neural ``ckpt:`` opponent; the
        search wrapper delegates to its base policy).  The critic margin is
        converted from ``Game.returns`` units to final points; ``win_prob``
        maps the discounted outcome head with ``(1 + outcome) / 2`` and is
        ``None`` when the checkpoint carries no outcome head.  The readout is
        a raw head pass -- search, rollouts and beta do not enter it.
        """
        if not self.show_estimate or self.mode != "free" or self.view is None:
            return None
        reader = getattr(self.free_policy, "value_and_outcome", None)
        if reader is None:
            return None
        try:
            value, outcome = reader(self.view)
        except Exception:  # noqa: BLE001 - a readout never breaks the table
            return None
        margin = float(value) * float(self.rules.total_points)
        win_prob = (
            None
            if outcome is None
            else min(max((1.0 + float(outcome)) / 2.0, 0.0), 1.0)
        )
        return {"margin": margin, "win_prob": win_prob}

    def placement_progress(self) -> dict[str, Any] | None:
        """Interim placement numbers for the in-game sidebar."""
        session = self.session
        if self.mode != "placement" or session is None:
            return None
        progress: dict[str, Any] = {
            "session_id": session.session_id,
            "session_dir": str(session.directory),
            "games_played": len(session.records),
            "games_total": session.config.games,
            "use_prior": session.trace_prior is not None,
            "prior_off_reason": session.prior_off_reason,
            "stopped_reason": session.stopped_reason,
            "records": [record.to_dict() for record in session.records],
        }
        if session.fit is not None:
            report = session.report()
            progress["estimate"] = {
                "mu": report["human"]["mu"],
                "sigma": report["human"]["sigma"],
                "ci_half_width": report["human"]["ci_half_width"],
                "nearest_level": report["nearest_level"],
                "provisional": report["provisional"],
                "band": report["band"],
                "weights": report["channels"]["weights"],
            }
        if self.report is not None:
            progress["report"] = self.report
        return progress

    def twin_progress(self) -> dict[str, Any] | None:
        """Interim twin numbers for the in-game sidebar and the report screen."""
        session = self.twin_session
        if self.mode != "twin" or session is None:
            return None
        current = None
        if self.twin_driver is not None and self.twin_driver.current is not None:
            scheduled = self.twin_driver.current.scheduled
            current = {
                "pair": scheduled.pair,
                "arm": scheduled.arm,
                "order": scheduled.order,
                "seat": scheduled.seat,
            }
        return {
            "session_id": session.session_id,
            "session_dir": str(session.directory),
            "pairs_total": session.pairs,
            "pairs_complete": session.pairs_complete(),
            "pairs_incomplete": len(session.pairs_incomplete()),
            "games_played": len(session.records),
            "games_total": len(session.plan),
            "current": current,
            "last": (
                session.records[-1].to_dict() if session.records else None
            ),
            "interim": session.interim(),
            "arms": session.arms_dict(),
            "report": self.report,
        }

    def snapshot(self) -> dict[str, Any]:
        """Everything the page needs to render the current state."""
        with self.lock:
            out: dict[str, Any] = {
                "mode": self.mode,
                "phase": self.phase,
                "error": self.error,
                "human_seat": self.human_seat,
                "num_players": self.rules.num_players,
                "names": list(self.names),
                "opponent": (
                    None
                    if self.opponent is None
                    else {
                        "id": self.opponent.id,
                        "mu": self.opponent.mu,
                        "sigma": self.opponent.sigma,
                        "anchor": bool(self.opponent.anchor),
                        "spec": self.opponent.spec,
                    }
                ),
                "seed": self.seed,
                "search": (
                    dict(self.search)
                    if self.mode == "free" and self.web_config.search is not None
                    else None
                ),
                "model_estimate": self.free_play_estimate(),
                "game_index": self.game_index,
                "games_total": self.games_total,
                "turn": self.turn,
                "tricks_completed": self.tricks_completed,
                "log": list(self.log),
                "placement": self.placement_progress(),
                "twin": self.twin_progress(),
                "result": self.result,
                "trace_path": self.trace_path,
                "rules_revision": RULES_REVISION,
            }
            if self.view is not None and self.game is not None:
                view = self.view
                out.update(
                    {
                        "scores": list(view.scores),
                        "hand_counts": list(view.counts),
                        "draw_count": view.draw_count,
                        "trick_points": self.state.trick_points if self.state else 0,
                        "trick_cards": [
                            card_json(card)
                            for card in sorted(view.trick_cards, key=card_key)
                        ],
                        "incumbent": (
                            None
                            if view.incumbent is None
                            else {**combo_json(view.incumbent), "seat": view.last_player}
                        ),
                        "last_player": view.last_player,
                        "current": view.current,
                        "revealed": [
                            {"seat": seat, "card": card_json(card)}
                            for seat, card in enumerate(view.revealed)
                        ],
                        "your_hand": [
                            card_json(card)
                            for card in sorted(view.hand, key=hand_key)
                        ],
                        "plays": [
                            play_json(play, self.rules) for play in view.plays
                        ],
                        "counter": counter_json(view),
                        "legal": (
                            legal_actions(self.game, view)
                            if self.phase == "human"
                            else []
                        ),
                    }
                )
            return out


class _PlacementDriver:
    """Step-driven front end to a real :class:`PlacementSession`.

    The CLI's ``run()`` plays every game through a blocking chooser; the
    browser needs one game at a time, so this driver uses the session's
    stepwise API (:meth:`begin`, :meth:`next_game`, :meth:`commit_game`,
    :meth:`finish`) — the same owner the CLI loop uses.
    """

    def __init__(
        self,
        *,
        table: TableSession,
        config: SessionConfig,
        use_prior: bool,
        seed: int,
        opponents: Sequence[Opponent] | None = None,
        policy_factory: Callable[[Opponent, int], Policy] | None = None,
    ) -> None:
        sessions_dir = table.web_config.sessions_dir
        directory = sessions_dir / new_session_id(sessions_dir)
        self.table = table
        self.session = PlacementSession(
            opponents=(
                tuple(opponents) if opponents is not None else table.opponents
            ),
            anchors=table.anchors,
            directory=directory,
            session_id=directory.name,
            config=config,
            rules=table.rules,
            rng=random.Random(seed),
            trace_prior=table.prior if use_prior else None,
            device=table.web_config.device,
            policy_factory=policy_factory,
        )
        self.session.begin()
        self.current: _CurrentGame | None = None
        self.finished = False

    def start_next(self) -> None:
        with self.table.lock:
            if self.finished or self.current is not None:
                return
            session = self.session
            scheduled = session.next_game()
            if scheduled is None:
                self.finish(session.stopped_reason or "max_games")
                return
            policy = session.opponent_policy(scheduled)
            policies: list[Policy | None] = [
                None if seat == scheduled.seat else policy
                for seat in range(session.rules.num_players)
            ]
            match = Match(session.rules, policies, rng=random.Random(scheduled.seed))
            self.current = _CurrentGame(
                scheduled=scheduled,
                match=match,
                record={"initial": initial_snapshot(match.state), "steps": []},
            )
            table = self.table
            table.game_index = scheduled.index + 1
            table.opponent = scheduled.opponent
            table.seed = scheduled.seed
            table.human_seat = scheduled.seat
            table.names = seat_names(
                scheduled.seat, session.rules.num_players, scheduled.opponent.id
            )
            table.begin_match(match, self.current.record)
            table.advance()

    def finish_game(self) -> None:
        """Record the just-finished game through the session's own pipeline."""
        current = self.current
        if current is None:
            raise WebError("没有进行中的定级牌局", status=409)
        session = self.session
        table = self.table
        match = current.match
        scheduled = current.scheduled
        current.record["final_scores"] = list(match.state.scores)
        trace = build_trace(
            session.rules,
            seed=scheduled.seed,
            human_seat=scheduled.seat,
            players=session.player_labels(scheduled),
            created_at=session.created_at,
            **current.record,
        )
        path = save_trace(
            session.directory
            / trace_filename(
                scheduled.index, scheduled.seed, scheduled.seat, scheduled.opponent.id
            ),
            trace,
        )
        session.commit_game(
            scheduled,
            scores=trace["final_scores"],
            trace=trace,
            trace_path=path,
        )
        self.current = None
        table.trace_path = str(path)
        if session.stopped_reason is not None:
            self.finish(session.stopped_reason)
        else:
            table.phase = "round_over"

    def abandon_current(self) -> None:
        with self.table.lock:
            self.current = None

    def finish(self, reason: str) -> None:
        with self.table.lock:
            report = self.session.finish(reason)
            self.table.report = report
            self.table.stopped_reason = reason
            self.finished = True
            self.table.phase = "session_over"


class _TwinDriver:
    """Step-driven front end to a real :class:`TwinSession`.

    Same shape as :class:`_PlacementDriver`: the browser plays one game at a
    time through the session's stepwise API.  The schedule, the arm policies,
    the labels and the report all stay owned by :mod:`seven523.twin`; this
    driver only turns a scheduled game into a :class:`Match` and hands the
    finished record back for persistence.
    """

    def __init__(self, *, table: TableSession, session: TwinSession) -> None:
        self.table = table
        self.session = session
        self.session.begin()
        self.current: _CurrentGame | None = None
        self.finished = False

    def start_next(self) -> None:
        with self.table.lock:
            if self.finished or self.current is not None:
                return
            session = self.session
            scheduled = session.next_game()
            if scheduled is None:
                self.finish(session.stopped_reason or "max_pairs")
                return
            policy = session.opponent_policy(scheduled)
            policies: list[Policy | None] = [
                None if seat == scheduled.seat else policy
                for seat in range(session.rules.num_players)
            ]
            match = Match(session.rules, policies, rng=random.Random(scheduled.seed))
            self.current = _CurrentGame(
                scheduled=scheduled,  # type: ignore[arg-type]
                match=match,
                record={"initial": initial_snapshot(match.state), "steps": []},
            )
            table = self.table
            table.game_index = scheduled.index + 1
            table.games_total = len(session.plan)
            table.seed = scheduled.seed
            table.human_seat = scheduled.seat
            table.names = seat_names(
                scheduled.seat, session.rules.num_players, scheduled.arm
            )
            table.begin_match(match, self.current.record)
            table.advance()

    def finish_game(self) -> None:
        """Record the just-finished twin game through the session pipeline."""
        current = self.current
        if current is None:
            raise WebError("没有进行中的 twin 牌局", status=409)
        session = self.session
        table = self.table
        scheduled = current.scheduled
        current.record["final_scores"] = list(current.match.state.scores)
        trace = session.build_trace(scheduled, current.record)
        path = save_trace(session.trace_path_for(scheduled), trace)
        session.commit_game(
            scheduled,
            scores=trace["final_scores"],
            trace=trace,
            trace_path=path,
        )
        self.current = None
        table.trace_path = str(path)
        if len(session.records) >= len(session.plan):
            self.finish(session.stopped_reason or "max_pairs")
        else:
            table.phase = "round_over"

    def abandon_current(self) -> None:
        with self.table.lock:
            if self.current is not None:
                self.session.abandon_current()
            self.current = None

    def finish(self, reason: str) -> None:
        with self.table.lock:
            report = self.session.finish(reason)
            self.table.report = report
            self.table.stopped_reason = reason
            self.finished = True
            self.table.phase = "session_over"

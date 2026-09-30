"""Deal-twin human sessions: one deal, two arms, one human.

A *twin* session answers a paired question the placement session cannot: on the
same deals, same seat, how does the human's margin change when the opponent
switches from the raw ``critic.pt`` argmax (the **raw** arm) to the same base
policy wrapped in the deployed O4-lite search (the **search** arm)?

The schedule pairs two games per deal: both reuse the deal seed and the human
seat, so the opening snapshot is byte-identical inside a pair and the residual
``Δ_i = margin(search) − margin(raw)`` cancels hand, seat phase and deal
difficulty.  Seats alternate *across* pairs and so does the within-pair game
order, keeping both exactly balanced without polluting each Δ with a seat main
effect.  Statistics are clustered by pair with a percentile bootstrap over
pairs — the same unit of analysis as :mod:`seven523.duel`.

The module is pure orchestration and math: no torch, no HTTP, no clock, and the
bootstrap takes an explicit :class:`random.Random`, so a report is reproducible
bit for bit from ``session.json`` plus ``bootstrap_seed``.  The browser table
(``_TwinDriver`` in :mod:`seven523.web.table`) and ``web_twin.py --simulate``
are two front ends over the same stepwise API, exactly like
``PlacementSession``.
"""
from __future__ import annotations

import json
import math
import random
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .play import QuitGame
from .policies import Policy, policy_from_spec
from .record import play_recorded, policy_seed
from .rules import DEFAULT_RULES, RULES_REVISION, Rules
from .trace import (
    build_trace as _trace_document,
    rules_json,
    save_trace,
    trace_filename,
)

__all__ = [
    "ARM_RAW",
    "ARM_SEARCH",
    "CAVEATS",
    "DEFAULT_BOOTSTRAP",
    "DEFAULT_BOOTSTRAP_SEED",
    "DEFAULT_CONFIDENCE",
    "DEFAULT_PAIRS",
    "MIN_PAIRS",
    "TWIN_ARMS",
    "TWIN_REPORT_SCHEMA",
    "TWIN_SESSION_SCHEMA",
    "TWIN_TRACE_SCHEMA",
    "TWIN_VERSION",
    "ScheduledTwinGame",
    "TwinArm",
    "TwinGameRecord",
    "TwinSession",
    "new_twin_id",
    "paired_twin_stats",
    "plan_twin_schedule",
]

#: Document tags; readers reject anything else.
TWIN_SESSION_SCHEMA = "seven523.deal-twin-session"
TWIN_REPORT_SCHEMA = "seven523.deal-twin-report"
TWIN_TRACE_SCHEMA = "seven523.deal-twin"
TWIN_VERSION = 1

#: The two arms.  ``raw`` never goes through a factory; ``search`` always does.
ARM_RAW = "raw"
ARM_SEARCH = "search"
TWIN_ARMS = (ARM_RAW, ARM_SEARCH)

#: Resolution defaults (design: a session below ``MIN_PAIRS`` is a smoke test).
DEFAULT_PAIRS = 30
MIN_PAIRS = 30
DEFAULT_BOOTSTRAP = 4000
DEFAULT_BOOTSTRAP_SEED = 523
DEFAULT_CONFIDENCE = 0.95

#: Mandatory reading caveats; echoed verbatim into every report.
CAVEATS = [
    "open-label：打的时候页面显示当前臂（raw/search）；期望效应不能排除。",
    "CRN 记忆：同 pair 第二局是同一副牌、同一座位，真人可能记得手牌/出法；"
    "顺序交替只在总体期望上抵消。",
    "单 bank：一个真人 session 是一条时间序列，疲劳/学习漂移只由 half_split 诊断，"
    "不构成独立 bank。",
    "描述性：本会话不进 manifest/prior/OpenSkill 拟合，不产出评分或档位。",
    "机器预期：search 臂比 raw 强约 +135.7 Elo（t_leafq・单位修复的 confirm 口径，"
    "bank 501-505 [128.2,143.1]；旧 +124.7 是 critic_B 值核，不对应本次 pin），"
    "人对 search 的 Δ 预期为负；Δ 不显著不等于两臂等价。",
]


# -- schedule ----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ScheduledTwinGame:
    """One game of the twin schedule: which arm, which pair, seat and order."""

    index: int
    pair: int
    seed: int
    seat: int
    arm: str
    order: int


@dataclass(frozen=True, slots=True)
class TwinArm:
    """One arm identity: a shared base spec plus optional search parameters.

    The raw arm carries ``search=None``; the search arm carries the pinned
    parameter document (``trunc_ply`` / ``rollout_k`` / ...) and the
    ``identity`` string the machine evaluation uses (``rolloutt:<base>``).
    """

    id: str
    spec: str
    label: str
    search: Mapping[str, Any] | None = None
    identity: str | None = None

    def to_dict(self) -> dict[str, Any]:
        if self.search is None:
            return {"id": self.id, "spec": self.spec, "label": self.label}
        return {
            "id": self.id,
            "base_spec": self.spec,
            "identity": self.identity,
            "label": self.label,
            "params": dict(self.search),
            "value_ckpt": self.search.get("value_ckpt"),
        }


def plan_twin_schedule(
    rng: random.Random,
    pairs: int,
    *,
    seat_start: int = 0,
    order_start: str = ARM_RAW,
) -> tuple[ScheduledTwinGame, ...]:
    """Plan ``pairs`` twin deals: two games per deal, same seed and seat.

    Each pair draws one fresh seed and fixes the human seat for both games;
    ``seat = (seat_start + pair) % 2`` alternates across pairs and the first
    arm alternates too (``order_start`` names the arm that goes first on even
    pairs).  Both arms see every deal exactly once, and with ``pairs`` even
    every (arm, seat) combination occurs ``pairs / 2`` times.  Pure,
    deterministic and index-adjacent inside a pair.
    """
    if pairs < 2:
        raise ValueError(f"pairs must be at least 2, got {pairs}")
    if pairs % 2 != 0:
        raise ValueError(f"pairs must be even, got {pairs}")
    if seat_start not in (0, 1):
        raise ValueError(f"seat_start must be 0 or 1, got {seat_start!r}")
    if order_start not in TWIN_ARMS:
        raise ValueError(
            f"order_start must be {TWIN_ARMS}, got {order_start!r}"
        )
    master = rng or random.Random()
    seeds: list[int] = []
    seen: set[int] = set()
    while len(seeds) < pairs:
        seed = master.randrange(1 << 32)
        if seed in seen:
            continue
        seen.add(seed)
        seeds.append(seed)

    schedule: list[ScheduledTwinGame] = []
    for pair, seed in enumerate(seeds):
        seat = (seat_start + pair) % 2
        first = order_start if pair % 2 == 0 else _other_arm(order_start)
        second = _other_arm(first)
        for order, arm in enumerate((first, second)):
            schedule.append(
                ScheduledTwinGame(
                    index=2 * pair + order,
                    pair=pair,
                    seed=seed,
                    seat=seat,
                    arm=arm,
                    order=order,
                )
            )
    return tuple(schedule)


def _other_arm(arm: str) -> str:
    return ARM_SEARCH if arm == ARM_RAW else ARM_RAW


def new_twin_id(base: str | Path, *, now: datetime | None = None) -> str:
    """A timestamped twin id that does not collide inside ``base``."""
    base = Path(base)
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    candidate = f"twin-{stamp}"
    suffix = 2
    while (base / candidate).exists():
        candidate = f"twin-{stamp}-{suffix}"
        suffix += 1
    return candidate


# -- records -----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TwinGameRecord:
    """One completed twin game, as written to ``session.json``."""

    index: int
    pair: int
    seed: int
    seat: int
    arm: str
    order: int
    scores: tuple[int, ...]
    human_score: int
    opponent_score: int
    margin: int
    result: str
    trace: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "pair": self.pair,
            "seed": self.seed,
            "seat": self.seat,
            "arm": self.arm,
            "order": self.order,
            "scores": list(self.scores),
            "human_score": self.human_score,
            "opponent_score": self.opponent_score,
            "margin": self.margin,
            "result": self.result,
            "trace": self.trace,
        }


def _play_value(record: TwinGameRecord) -> float:
    """Expected-score credit for a game: win = 1, draw = 0.5, loss = 0."""
    if record.result == "win":
        return 1.0
    if record.result == "draw":
        return 0.5
    if record.result == "loss":
        return 0.0
    raise ValueError(f"unknown result {record.result!r} on pair {record.pair}")


# -- the paired estimator ----------------------------------------------------


def _clamp_probability(p: float, games: int) -> float:
    """Keep ``p`` inside the range a finite sample can actually produce.

    With ``games`` games per arm the smallest non-zero relative frequency is
    ``1 / games``; at that floor ``log10(p / (1 - p))`` stays finite (the Elo
    difference saturates).  A one-game arm has no resolution at all, so the
    floor is capped at 0.5 and the readout collapses to 0.
    """
    floor = min(1.0 / games, 0.5)
    return min(max(p, floor), 1.0 - floor)


def _elo_diff(p: float, games: int) -> float:
    """``400 * log10(p / (1 - p))`` — human-minus-opponent Elo readout."""
    p = _clamp_probability(p, games)
    return 400.0 * math.log10(p / (1.0 - p))


def _quantile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated quantile of an already-sorted sequence."""
    if not values:
        raise ValueError("quantile of an empty sample")
    if len(values) == 1:
        return values[0]
    position = q * (len(values) - 1)
    lower = math.floor(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def _binomial_two_sided_p(successes: int, trials: int) -> float:
    """Exact two-sided binomial p-value for ``H0: p = 0.5`` (pure ``math``)."""
    if trials <= 0:
        return 1.0
    pmf = [math.comb(trials, k) for k in range(trials + 1)]
    observed = pmf[min(max(successes, 0), trials)]
    return min(1.0, sum(value for value in pmf if value <= observed) / (1 << trials))


def _mean_or_none(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _subset_summary(groups: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Point estimates for one diagnostic half sample (no test, display only)."""
    return {
        "pairs": len(groups),
        "d_margin": _mean_or_none([float(g["d_margin"]) for g in groups]),
        "d_win": _mean_or_none([float(g["d_win"]) for g in groups]),
    }


def _arm_stats(records: Sequence[TwinGameRecord], arm: str) -> dict[str, Any]:
    games = [record for record in records if record.arm == arm]
    wins = sum(record.result == "win" for record in games)
    draws = sum(record.result == "draw" for record in games)
    losses = sum(record.result == "loss" for record in games)
    count = len(games)
    return {
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "games": count,
        "winrate": (wins + 0.5 * draws) / count if count else None,
        "mean_margin": (
            sum(record.margin for record in games) / count if count else None
        ),
    }


def paired_twin_stats(
    records: Sequence[TwinGameRecord],
    *,
    bootstrap: int = DEFAULT_BOOTSTRAP,
    rng: random.Random,
    confidence: float = DEFAULT_CONFIDENCE,
    min_pairs: int = MIN_PAIRS,
) -> dict[str, Any]:
    """Summarise a twin session from the search-minus-raw viewpoint.

    Only **complete pairs** enter the estimate — a pair is complete when both
    arms have one committed game with the same seed and seat.  The clustering
    unit is the pair, so the bootstrap resamples pairs (with replacement) and
    keeps each pair's raw/search games together; ``Δ_i`` is the margin
    difference and ``d_win`` the expected-score difference.  The raw arm is the
    reference, so ``delta.margin_points.value`` is the *human-vs-search minus
    human-vs-raw* expected margin, and the Elo delta is
    ``elo(search winrate) − elo(raw winrate)`` with each arm clamped at
    ``1 / games-per-arm``.

    ``rng`` must be an explicit :class:`random.Random`; the result is a
    deterministic function of it, and the input order is irrelevant because
    pairs are processed in pair order.  Incomplete pairs are counted and
    excluded; an empty/incomplete-only input yields ``None`` point estimates
    (a session that quit before completing a pair still gets a report).
    """
    if bootstrap < 1:
        raise ValueError(f"bootstrap must be at least 1, got {bootstrap}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must lie in (0, 1), got {confidence}")
    if min_pairs < 1:
        raise ValueError(f"min_pairs must be at least 1, got {min_pairs}")
    if rng is None:
        raise ValueError("paired_twin_stats needs an explicit random.Random rng")

    by_pair: dict[int, list[TwinGameRecord]] = {}
    for record in records:
        if record.arm not in TWIN_ARMS:
            raise ValueError(f"unknown arm {record.arm!r} on pair {record.pair}")
        by_pair.setdefault(record.pair, []).append(record)

    pairs: list[dict[str, Any]] = []
    incomplete = 0
    for pair in sorted(by_pair):
        group = sorted(by_pair[pair], key=lambda record: record.order)
        if len(group) == 1:
            incomplete += 1
            continue
        if len(group) != 2:
            raise ValueError(
                f"pair {pair} has {len(group)} games; a twin pair needs exactly 2"
            )
        arms = {record.arm for record in group}
        if arms != set(TWIN_ARMS):
            raise ValueError(f"pair {pair} does not hold one game per arm (got {arms})")
        raw = next(record for record in group if record.arm == ARM_RAW)
        search = next(record for record in group if record.arm == ARM_SEARCH)
        if raw.seed != search.seed or raw.seat != search.seat:
            raise ValueError(
                f"pair {pair} games must share the deal seed and seat "
                f"(raw seed={raw.seed} seat={raw.seat}, "
                f"search seed={search.seed} seat={search.seat})"
            )
        pairs.append(
            {
                "pair": pair,
                "seed": raw.seed,
                "seat": raw.seat,
                "order": [group[0].arm, group[1].arm],
                "margin_raw": raw.margin,
                "margin_search": search.margin,
                "d_margin": search.margin - raw.margin,
                "win_raw": _play_value(raw),
                "win_search": _play_value(search),
                "d_win": _play_value(search) - _play_value(raw),
                "result_raw": raw.result,
                "result_search": search.result,
            }
        )

    pairs_left = len(pairs)
    d_margins = [float(pair["d_margin"]) for pair in pairs]
    d_wins = [float(pair["d_win"]) for pair in pairs]
    raw_wins = [float(pair["win_raw"]) for pair in pairs]
    search_wins = [float(pair["win_search"]) for pair in pairs]

    improved = sum(1 for value in d_margins if value > 0)
    worsened = sum(1 for value in d_margins if value < 0)
    same = pairs_left - improved - worsened
    better = sum(
        1
        for pair in pairs
        if _swing(pair["result_search"]) > _swing(pair["result_raw"])
    )
    worse = sum(
        1
        for pair in pairs
        if _swing(pair["result_search"]) < _swing(pair["result_raw"])
    )
    swing_same = pairs_left - better - worse

    if pairs_left:
        margin_value: float | None = _mean_or_none(d_margins)
        winrate_value: float | None = _mean_or_none(d_wins)
        raw_winrate: float | None = _mean_or_none(raw_wins)
        search_winrate: float | None = _mean_or_none(search_wins)
        elo_value: float | None = _elo_diff(
            search_winrate, pairs_left
        ) - _elo_diff(raw_winrate, pairs_left)
        margins: list[float] = []
        winrates: list[float] = []
        elos: list[float] = []
        for _ in range(bootstrap):
            margin_sum = 0.0
            raw_sum = 0.0
            search_sum = 0.0
            for _ in range(pairs_left):
                index = rng.randrange(pairs_left)
                margin_sum += d_margins[index]
                raw_sum += raw_wins[index]
                search_sum += search_wins[index]
            margins.append(margin_sum / pairs_left)
            winrates.append((search_sum - raw_sum) / pairs_left)
            elos.append(
                _elo_diff(search_sum / pairs_left, pairs_left)
                - _elo_diff(raw_sum / pairs_left, pairs_left)
            )
        margins.sort()
        winrates.sort()
        elos.sort()
        alpha = (1.0 - confidence) / 2.0
        margin_ci: list[float] | None = [
            _quantile(margins, alpha),
            _quantile(margins, 1.0 - alpha),
        ]
        winrate_ci: list[float] | None = [
            _quantile(winrates, alpha),
            _quantile(winrates, 1.0 - alpha),
        ]
        elo_ci: list[float] | None = [
            _quantile(elos, alpha),
            _quantile(elos, 1.0 - alpha),
        ]
        sd = statistics.stdev(d_margins) if pairs_left >= 2 else 0.0
        se: float | None = sd / math.sqrt(pairs_left)
    else:
        margin_value = winrate_value = raw_winrate = search_winrate = None
        elo_value = margin_ci = winrate_ci = elo_ci = se = None
        sd = None

    resolution_ok = pairs_left >= min_pairs
    if pairs_left >= 2:
        mde = (1.96 + 0.8416) * float(sd) / math.sqrt(pairs_left)
    else:
        mde = None
    note = (
        "至少 30 对（60 局）才谈得上读出配对 Δ"
        if resolution_ok
        else f"只有 {pairs_left} 对（<{min_pairs}）：点估计/CI 只作流程验证，不能作为行动级结论"
    )

    first_arm_raw = [pair for pair in pairs if pair["order"][0] == ARM_RAW]
    first_arm_search = [pair for pair in pairs if pair["order"][0] == ARM_SEARCH]
    seat0 = [pair for pair in pairs if pair["seat"] == 0]
    seat1 = [pair for pair in pairs if pair["seat"] == 1]
    middle = pairs_left // 2
    return {
        "counts": {
            "games_played": len(records),
            "pairs_complete": pairs_left,
            "pairs_incomplete": incomplete,
        },
        "arm_stats": {
            ARM_RAW: _arm_stats(records, ARM_RAW),
            ARM_SEARCH: _arm_stats(records, ARM_SEARCH),
        },
        "delta": {
            "margin_points": {
                "value": margin_value,
                "ci95": margin_ci,
                "se": se,
                "sd": sd,
            },
            "winrate": {
                "value": winrate_value,
                "ci95": winrate_ci,
                "raw_winrate": raw_winrate,
                "search_winrate": search_winrate,
            },
            "elo": {"value": elo_value, "ci95": elo_ci},
            "sign": {
                "improved": improved,
                "same": same,
                "worsened": worsened,
                "p_value": _binomial_two_sided_p(improved, improved + worsened),
            },
            "outcome_swing": {"better": better, "same": swing_same, "worse": worse},
            "per_pair": pairs,
            "order_split": {
                ARM_RAW: _subset_summary(first_arm_raw),
                ARM_SEARCH: _subset_summary(first_arm_search),
            },
            "seat_split": {
                "seat0": _subset_summary(seat0),
                "seat1": _subset_summary(seat1),
            },
            "half_split": {
                "first": _subset_summary(pairs[:middle]),
                "second": _subset_summary(pairs[middle:]),
            },
        },
        "resolution": {
            "min_pairs": min_pairs,
            "pairs_complete": pairs_left,
            "ok": resolution_ok,
            "mde80_points": mde,
            "note": note,
        },
        "bootstrap": bootstrap,
        "confidence": confidence,
    }


def _swing(result: str) -> int:
    return {"win": 2, "draw": 1, "loss": 0}[result]


# -- the session state machine -----------------------------------------------


class TwinSession:
    """One deal-twin session: schedule → play → paired report.

    Stepwise API mirrors :class:`~seven523.placement.PlacementSession`:
    :meth:`begin` → :meth:`next_game` → :meth:`opponent_policy` /
    :meth:`player_labels` → :meth:`commit_game` (or :meth:`abandon_current`) →
    :meth:`finish`.  ``run()`` drives the same methods with an injected human
    policy or chooser and is the ``--simulate`` front end.

    The raw arm is always built by :func:`~seven523.policies.policy_from_spec`
    (never by ``search_factory``); the search arm is only ever built by
    ``search_factory``, with the pinned parameters.  Construction refuses a
    missing factory, a non-``ckpt:`` base spec and a parameter document without
    ``trunc_ply`` / ``rollout_k``.
    """

    def __init__(
        self,
        *,
        base_spec: str,
        raw_id: str,
        search_id: str,
        search_params: Mapping[str, Any],
        directory: str | Path,
        session_id: str,
        search_factory: Callable[[str, Rules, int, Mapping[str, Any]], Policy] | None,
        pairs: int = DEFAULT_PAIRS,
        seat_start: int = 0,
        order_start: str = ARM_RAW,
        rules: Rules = DEFAULT_RULES,
        rng: random.Random | None = None,
        bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
        bootstrap: int = DEFAULT_BOOTSTRAP,
        confidence: float = DEFAULT_CONFIDENCE,
        min_pairs: int = MIN_PAIRS,
        device: str = "cpu",
        created_at: str | None = None,
        raw_label: str | None = None,
        search_label: str | None = None,
        search_identity: str | None = None,
        rng_seed: int | None = None,
    ) -> None:
        if search_factory is None:
            raise ValueError(
                "TwinSession needs a search_factory: without one both arms would "
                "silently run raw and the comparison would be meaningless"
            )
        if not str(base_spec).startswith("ckpt:"):
            raise ValueError(
                f"the twin base spec must be a ckpt: spec, got {base_spec!r}"
            )
        if not isinstance(search_params, Mapping) or not {
            "trunc_ply",
            "rollout_k",
        } <= set(search_params):
            raise ValueError(
                "search_params must contain trunc_ply and rollout_k, "
                f"got {search_params!r}"
            )
        if rules.num_players != 2:
            raise ValueError(
                f"a twin session is heads-up, got {rules.num_players} players"
            )
        if pairs < 2:
            raise ValueError(f"pairs must be at least 2, got {pairs}")
        if pairs % 2 != 0:
            raise ValueError(f"pairs must be even, got {pairs}")
        if seat_start not in (0, 1):
            raise ValueError(f"seat_start must be 0 or 1, got {seat_start!r}")
        if order_start not in TWIN_ARMS:
            raise ValueError(
                f"order_start must be {TWIN_ARMS}, got {order_start!r}"
            )
        if bootstrap < 1:
            raise ValueError(f"bootstrap must be at least 1, got {bootstrap}")
        if not 0.0 < confidence < 1.0:
            raise ValueError(f"confidence must lie in (0, 1), got {confidence}")

        self.base_spec = str(base_spec)
        self.raw_id = str(raw_id)
        self.search_id = str(search_id)
        self.search_params = dict(search_params)
        self.directory = Path(directory)
        self.session_id = str(session_id)
        self.search_factory = search_factory
        self.pairs = int(pairs)
        self.seat_start = int(seat_start)
        self.order_start = order_start
        self.rules = rules
        self.rng = rng or random.Random()
        self.rng_seed = rng_seed
        self.bootstrap_seed = int(bootstrap_seed)
        self.bootstrap = int(bootstrap)
        self.confidence = float(confidence)
        self.min_pairs = int(min_pairs)
        self.device = device
        self.created_at = created_at or datetime.now().isoformat(timespec="seconds")
        self.raw_arm = TwinArm(
            id=self.raw_id,
            spec=self.base_spec,
            label=raw_label or self.raw_id,
        )
        self.search_arm = TwinArm(
            id=self.search_id,
            spec=self.base_spec,
            label=search_label or self.search_id,
            search=self.search_params,
            identity=search_identity
            or (
                "rolloutt:" + self.base_spec[len("ckpt:") :]
                if self.base_spec.startswith("ckpt:")
                else None
            ),
        )
        self.plan: tuple[ScheduledTwinGame, ...] = ()
        self._records: list[TwinGameRecord] = []
        self._current: ScheduledTwinGame | None = None
        self._abandoned_pairs: set[int] = set()
        self.abandoned_games = 0
        self.stopped_reason: str | None = None
        self._begun = False

    # -- state ------------------------------------------------------------

    @property
    def records(self) -> tuple[TwinGameRecord, ...]:
        return tuple(self._records)

    def config_dict(self) -> dict[str, Any]:
        return {
            "pairs": self.pairs,
            "seat_start": self.seat_start,
            "order_start": self.order_start,
            "bootstrap": self.bootstrap,
            "bootstrap_seed": self.bootstrap_seed,
            "confidence": self.confidence,
            "min_pairs": self.min_pairs,
            "rng_seed": self.rng_seed,
        }

    def arms_dict(self) -> dict[str, Any]:
        return {
            ARM_RAW: self.raw_arm.to_dict(),
            ARM_SEARCH: self.search_arm.to_dict(),
        }

    def _counts(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for record in self._records:
            counts[record.pair] = counts.get(record.pair, 0) + 1
        return counts

    def pairs_complete(self) -> int:
        return sum(count == 2 for count in self._counts().values())

    def pairs_incomplete(self) -> list[int]:
        counts = self._counts()
        incomplete = set(self._abandoned_pairs)
        for pair, count in counts.items():
            if count == 2:
                incomplete.discard(pair)
            elif count == 1:
                incomplete.add(pair)
        return sorted(pair for pair in incomplete if counts.get(pair, 0) < 2)

    # -- stepwise API ------------------------------------------------------

    def begin(self) -> None:
        """Draw the schedule, create the directory and write ``session.json``."""
        if self._begun:
            return
        self.plan = plan_twin_schedule(
            self.rng,
            self.pairs,
            seat_start=self.seat_start,
            order_start=self.order_start,
        )
        self._begun = True
        self._persist()

    def next_game(self) -> ScheduledTwinGame | None:
        """The next scheduled twin game, or ``None`` when the plan is done."""
        if len(self._records) >= len(self.plan):
            if self.stopped_reason is None:
                self.stopped_reason = "max_pairs"
            return None
        scheduled = self.plan[len(self._records)]
        self._current = scheduled
        return scheduled

    def opponent_policy(self, scheduled: ScheduledTwinGame) -> Policy:
        """The arm policy for ``scheduled`` (per-seat policy seed)."""
        seed = policy_seed(scheduled.seed, scheduled.seat)
        if scheduled.arm == ARM_SEARCH:
            return self.search_factory(
                self.base_spec, self.rules, seed, dict(self.search_params)
            )
        return policy_from_spec(
            self.base_spec, self.rules, seed=seed, device=self.device
        )

    def player_labels(self, scheduled: ScheduledTwinGame) -> list[str]:
        """Trace ``players`` labels: human seat plus a ``twin_<arm>:`` opponent."""
        arm = self.raw_arm if scheduled.arm == ARM_RAW else self.search_arm
        role = "twin_raw" if scheduled.arm == ARM_RAW else "twin_search"
        labels: list[str] = []
        for seat in range(self.rules.num_players):
            if seat == scheduled.seat:
                labels.append(f"human@seat{seat}")
            else:
                labels.append(f"{role}:{arm.id}@seat{seat}")
        return labels

    def trace_annotation(self, scheduled: ScheduledTwinGame) -> dict[str, Any]:
        """The top-level ``deal_twin`` block written into every twin trace."""
        arm = self.raw_arm if scheduled.arm == ARM_RAW else self.search_arm
        return {
            "schema": TWIN_TRACE_SCHEMA,
            "version": TWIN_VERSION,
            "session": self.session_id,
            "pairs_total": self.pairs,
            "pair": scheduled.pair,
            "arm": scheduled.arm,
            "arm_id": arm.id,
            "order": scheduled.order,
            "seat": scheduled.seat,
            "base_spec": self.base_spec,
            "search": (
                dict(self.search_params) if scheduled.arm == ARM_SEARCH else None
            ),
        }

    def trace_path_for(self, scheduled: ScheduledTwinGame) -> Path:
        return self.directory / trace_filename(
            scheduled.index, scheduled.seed, scheduled.seat, scheduled.arm
        )

    def build_trace(
        self, scheduled: ScheduledTwinGame, record: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Wrap a play record into a twin trace (annotation + search pin)."""
        trace = _trace_document(
            self.rules,
            seed=scheduled.seed,
            human_seat=scheduled.seat,
            players=self.player_labels(scheduled),
            created_at=self.created_at,
            **record,
        )
        trace["deal_twin"] = self.trace_annotation(scheduled)
        if scheduled.arm == ARM_SEARCH:
            trace["opponent_search"] = dict(self.search_params)
        return trace

    def commit_game(
        self,
        scheduled: ScheduledTwinGame,
        *,
        scores: Sequence[int],
        trace: Mapping[str, Any],
        trace_path: Path,
    ) -> TwinGameRecord:
        """Record an externally played twin game and persist the session."""
        if scheduled.index != len(self._records):
            raise ValueError(
                f"scheduled game {scheduled.index} is not next "
                f"({len(self._records)} played)"
            )
        values = [int(score) for score in scores]
        if len(values) != self.rules.num_players:
            raise ValueError(
                f"scores need {self.rules.num_players} entries, got {values!r}"
            )
        mine = values[scheduled.seat]
        theirs = values[1 - scheduled.seat]
        result = "win" if mine > theirs else ("loss" if mine < theirs else "draw")
        record = TwinGameRecord(
            index=scheduled.index,
            pair=scheduled.pair,
            seed=scheduled.seed,
            seat=scheduled.seat,
            arm=scheduled.arm,
            order=scheduled.order,
            scores=tuple(values),
            human_score=mine,
            opponent_score=theirs,
            margin=mine - theirs,
            result=result,
            trace=Path(trace_path).name,
        )
        self._records.append(record)
        self._current = None
        self._persist()
        return record

    def abandon_current(self) -> None:
        """Drop the in-flight game; its pair is counted incomplete."""
        if self._current is None:
            return
        self._abandoned_pairs.add(self._current.pair)
        self.abandoned_games += 1
        self._current = None

    def finish(self, reason: str | None = None) -> dict[str, Any]:
        """Close the session: persist ``session.json`` and write ``report.json``."""
        self.stopped_reason = reason or self.stopped_reason or "max_pairs"
        self._persist()
        report = self.report()
        self._write_json("report.json", report)
        return report

    # -- the loop ----------------------------------------------------------

    def run(
        self,
        human_policy_factory: Callable[[int], Policy] | None = None,
        *,
        chooser_factory: Callable[[int], Callable[..., Any]] | None = None,
        print_fn: Callable[..., None] = print,
    ) -> dict[str, Any]:
        """Play the scheduled games, persist everything, return the report.

        Exactly one driver must be given.  ``human_policy_factory(seat)``
        returns the scripted human's policy; ``chooser_factory(seat)`` returns
        the interactive ``(game, state, view)`` chooser.  A :class:`QuitGame`
        or Ctrl-C abandons the current game and finishes with ``quit``; the
        half-played pair stays counted as incomplete and out of the estimate.
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
            arm = self.raw_arm if scheduled.arm == ARM_RAW else self.search_arm
            print_fn(
                f"pair {scheduled.pair + 1}/{self.pairs} "
                f"局 {scheduled.index + 1}/{len(self.plan)}：",
                f"对手 {arm.label}，你坐 {scheduled.seat} 号位，"
                f"seed {scheduled.seed}",
            )
            try:
                self._play_one(
                    scheduled,
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
                self.abandon_current()
                break
        return self.finish(reason)

    def _play_one(
        self,
        scheduled: ScheduledTwinGame,
        *,
        human_policy: Policy | None,
        chooser: Callable[..., Any] | None,
        print_fn: Callable[..., None],
    ) -> TwinGameRecord:
        if human_policy is None and chooser is None:
            raise ValueError("_play_one needs a human policy or a chooser")
        policy = self.opponent_policy(scheduled)
        policies: list[Policy | None] = [policy] * self.rules.num_players
        if human_policy is not None:
            policies[scheduled.seat] = human_policy
        recorded = play_recorded(
            policies,  # type: ignore[arg-type]
            chooser=chooser,
            rules=self.rules,
            seed=scheduled.seed,
            human_seat=scheduled.seat,
            players=self.player_labels(scheduled),
            created_at=self.created_at,
            trace_dir=None,
            trace_index=scheduled.index,
            opponent=scheduled.arm,
            print_fn=print_fn,
        )
        trace = recorded.trace
        trace["deal_twin"] = self.trace_annotation(scheduled)
        if scheduled.arm == ARM_SEARCH:
            trace["opponent_search"] = dict(self.search_params)
        path = save_trace(self.trace_path_for(scheduled), trace)
        return self.commit_game(
            scheduled,
            scores=recorded.scores,
            trace=trace,
            trace_path=path,
        )

    # -- reporting ---------------------------------------------------------

    def interim(self) -> dict[str, Any]:
        """Cheap live readout for the in-game sidebar (no CI recompute)."""
        stats = paired_twin_stats(
            self._records,
            bootstrap=1,
            rng=random.Random(self.bootstrap_seed),
            confidence=self.confidence,
            min_pairs=self.min_pairs,
        )
        delta = stats["delta"]
        return {
            "pairs_complete": stats["counts"]["pairs_complete"],
            "margin_points_delta": delta["margin_points"]["value"],
            "winrate_delta": delta["winrate"]["value"],
            "sign": delta["sign"],
            "last_pair": delta["per_pair"][-1] if delta["per_pair"] else None,
        }

    def report(self) -> dict[str, Any]:
        """The final twin document: Δ (margin/winrate/Elo), sign, diagnostics."""
        stats = paired_twin_stats(
            self._records,
            bootstrap=self.bootstrap,
            rng=random.Random(self.bootstrap_seed),
            confidence=self.confidence,
            min_pairs=self.min_pairs,
        )
        counts = {
            "games_played": len(self._records),
            "pairs_complete": self.pairs_complete(),
            "pairs_incomplete": len(self.pairs_incomplete()),
            "pairs_total": self.pairs,
            "abandoned_games": self.abandoned_games,
        }
        return {
            "schema": TWIN_REPORT_SCHEMA,
            "version": TWIN_VERSION,
            "session": {
                "id": self.session_id,
                "directory": str(self.directory),
                "created_at": self.created_at,
                "session_json": str(self.directory / "session.json"),
                "report": str(self.directory / "report.json"),
                "bank": 1,
            },
            "config": self.config_dict(),
            "arms": self.arms_dict(),
            "counts": counts,
            "arm_stats": stats["arm_stats"],
            "delta": stats["delta"],
            "resolution": stats["resolution"],
            "bootstrap": self.bootstrap,
            "confidence": self.confidence,
            "bootstrap_seed": self.bootstrap_seed,
            "rules": {**rules_json(self.rules), "revision": RULES_REVISION},
            "caveats": list(CAVEATS),
        }

    # -- persistence -------------------------------------------------------

    def _session_document(self) -> dict[str, Any]:
        return {
            "schema": TWIN_SESSION_SCHEMA,
            "version": TWIN_VERSION,
            "id": self.session_id,
            "created_at": self.created_at,
            "directory": str(self.directory),
            "rules": {**rules_json(self.rules), "revision": RULES_REVISION},
            "config": self.config_dict(),
            "arms": self.arms_dict(),
            "plan": [asdict(scheduled) for scheduled in self.plan],
            "games": [record.to_dict() for record in self._records],
            "incomplete_pairs": self.pairs_incomplete(),
            "stopped_reason": self.stopped_reason,
        }

    def _write_json(self, name: str, document: Mapping[str, Any]) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / name
        path.write_text(
            json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return path

    def _persist(self) -> None:
        self._write_json("session.json", self._session_document())

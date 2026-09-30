#!/usr/bin/env python3
"""Publish the order-free anchored probit-MLE table (ADR-0013 decision 3).

    uv run python tools/refit_mle.py \
        --games traces/study/games_0.jsonl --games traces/study/games_1.jsonl \
        --anchors random=0 --bootstrap 200 --seed 0 \
        --out artifacts/mle/table.json

Reads one or more per-game JSONL files written by ``ladder.play_games``
(``{"seed", "seats", "scores", ..., "rules_id"}``), refuses any row whose
``rules_id`` is missing or differs from the current ``rules_id(DEFAULT_RULES)``
(ADR-0013: ratings are never pooled across rule versions), and fits the
anchored probit MAP model with :func:`seven523.mle.fit_mle`.  Free ids can be
warm-started from a study manifest through ``--prior-manifest``; that manifest
passes the same rules-identity gate as ``placement.load_opponents``.

The tool prints a mu-descending table (mu, Laplace sigma, games, 95% CI and
the separation flag) unless ``--json`` asks for the machine-readable artifact
only.  The artifact carries a per-id ``n`` map next to ``levels``/``sigmas``
(games seen, ``0`` for prior-only ids) so a consumer can tell a measured id
from one that only inherited its prior centre.  The 95% CI is a deal-clustered bootstrap that resamples the rows sharing
a ``seed`` with an explicit :class:`random.Random` ``(--seed)``; ``--bootstrap
0`` disables it and falls back to ``mu +/- 1.96 * sigma`` (the Laplace sd) and
the artifact's ``ci_source`` names which source produced the interval.
``--bootstrap-workers INT`` (default 4) fans the resampling over forked
processes while replaying the serial RNG stream exactly, so the artifact is
bit-for-bit independent of the worker count; 1 runs the serial loop.  A fit
that does not reach the solver's numerical optimum, or publishes a non-finite
sigma/CI, is reported on stderr instead of silently; non-finite numbers are
written as strict-JSON ``null`` (``allow_nan=False`` never emits NaN).  The
artifact also carries ``converged`` and ``margin_identified`` so a consumer can
never treat a provisional or boundary margin as a fitted value.  Nothing is
written without ``--out``, and the artifact carries the rule identity plus the
``separation`` id list so a consumer can never confuse two rule versions.

The published study manifest is a separate, explicit decision:
``--manifest-out PATH`` together with ``--refit`` merges the fit into that
manifest through :func:`seven523.study.merge_manifest` (same ``rules_id``
gate) and saves it with :func:`seven523.study.save_manifest`.  The merge
replaces each subject's ``mu``/``sigma`` and ``games`` while preserving its
``spec`` and every unrelated key, keeps the manifest anchors, records the fit
(``--spec ID=SPEC`` supplies the spec for a new measured id, so a fresh
subject is never published spec-less; ``--search-config ID=@FILE.json``
seeds/refreshes a subject's published ``search_config`` block and a refit
without it preserves the block), records the fit
provenance plus the games ``source`` in ``estimator``, and re-selects the
published ``rungs`` from the non-anchor levels.  Without both flags the
manifest is never touched, and ``--manifest-out`` without ``--refit`` is
refused instead of silently freezing the old levels.

Review fixes folded into that path:

* ``--prior-sigma FLOAT`` keeps each ``--prior-manifest`` prior centre and
  replaces its width with the explicit value (fail-loud on non-finite or
  non-positive); the effective value is recorded as ``estimator.prior_sigma``.
* ``--keep-rungs`` publishes the manifest's existing rung selection instead of
  re-running ``elo.select_rungs``; the rung ``mu``/``sigma`` are refreshed
  from the refit levels, missing/incoherent rungs are refused, and the
  selection status is recorded in ``estimator.rungs_selection_ok`` so the
  shortfall warning never lives only on stderr.
* prior-only ids (``n == 0``, no games) are excluded from the published
  manifest ``levels``/``subjects``/rungs; the table artifact still carries
  them with ``n == 0`` so a consumer can tell them from measured ids.

When ``--bootstrap >= 1`` the artifact also carries a ``resolution`` block
(the design §4.4 draft, H2/H4-corrected): per-band-pair ``win_prob`` (ties
credited as half a win, matching ``mle.fit_mle``'s ordered-probit terms),
percentile ``ci95_win_prob`` and bootstrap ``se_elo`` from the same deal
clustered replicates, ``achieved.power_at_delta`` restricted to the CI-only
notion, a separate ``action_power_at_delta`` for the compound
"CI excludes 0 AND point >= +10" rule (whose power at a true +10 is capped
at 0.5), and a ``transitivity`` diagnosis computed leave-one-bank-out across
the ``--games`` files.  A single ``--games`` bank reports
``transitivity.status = "not-computed"`` rather than an in-sample z.
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import random
import sys
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from itertools import pairwise
from pathlib import Path
from statistics import NormalDist
from typing import Any

from seven523.elo import (
    DEFAULT_BETA,
    DEFAULT_ELO_SCALE,
    PlayedGame,
    Prior,
    Rung,
    RungSelection,
    select_rungs,
)
from seven523.ladder import manifest_priors
from seven523.mle import MleConfig, MleFit, fit_mle
from seven523.rules import DEFAULT_RULES, rules_id, rules_identity
from seven523.study import load_manifest, merge_manifest, save_manifest

__all__ = [
    "action_power",
    "bootstrap_cis",
    "bootstrap_levels",
    "build_artifact",
    "build_resolution",
    "build_transitivity",
    "ci_only_power",
    "format_table",
    "load_games",
    "load_games_by_bank",
    "main",
    "pair_deal_counts",
    "parse_args",
    "percentile_cis",
    "publish_manifest",
]

#: Two-sided normal quantile for a 95% interval, the non-bootstrap fallback.
_Z95 = 1.96

#: Largest absolute rating the CLI accepts for an anchor or a prior centre.
#: The published scale is the 0/500/200 ADR-0012 axis; anything past a million
#: is a unit/serialisation error, and letting it through only produces
#: unrepresentable tail arithmetic.  Guards the tool's public boundary:
#: ``fit_mle`` itself stays total for any finite rating.
_SCALE_BOUND = 1e6

#: Legal ``--bootstrap-workers`` range.  1 is the serial reference path; the
#: upper bound keeps a stray value from oversubscribing the machine.
_BOOTSTRAP_WORKERS_MIN = 1
_BOOTSTRAP_WORKERS_MAX = 64

#: Fork-shared inputs for :func:`_bootstrap_replicates`.  :func:`bootstrap_levels`
#: sets it immediately before creating the pool so the forked children inherit
#: the deal table copy-on-write; the worker itself only receives its captured
#: RNG state and replicate count as task arguments.  ``None`` outside a run.
_BOOTSTRAP_CONTEXT: (
    tuple[
        Sequence[Sequence[PlayedGame]],
        Mapping[str, float],
        Mapping[str, Prior],
        MleConfig,
        tuple[str, ...],
    ]
    | None
) = None

#: Published rung spacing contract, matching ``ladder.build_ladder`` defaults:
#: the homoscedastic MLE levels are expected to yield only 2-3 rungs, and the
#: shortfall/wide gaps are reported rather than silently relaxed.
_RUNG_COUNT = 5
_RUNG_MIN_SPACING = 100.0
_RUNG_MAX_SPACING = 150.0

#: Resolution defaults from design §4.2/§4.4.  Review H2 keeps the two power
#: notions separate: ``power_at_delta`` is CI-only, ``action_power_at_delta``
#: is the compound "CI excludes 0 AND point >= +10" rule.
_TARGET_DELTA_ELO = 10.0
_TARGET_POWER = 0.8
_TARGET_ALPHA = 0.05

#: Review H4: the observed-vs-model |z| threshold.  It is only reported when a
#: genuine holdout (leave-one-bank-out) z exists; a single bank never produces
#: one.
_TRANSITIVITY_Z = 2.5

_SQRT2 = math.sqrt(2.0)


def _json_number(value: float) -> float | None:
    """``value`` when finite, else ``None`` (strict-JSON safe)."""
    return value if math.isfinite(value) else None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit and publish the order-free anchored probit-MLE table "
            "(ADR-0013 decision 3)"
        )
    )
    parser.add_argument(
        "--games",
        action="append",
        required=True,
        metavar="PATH",
        help="per-game JSONL written by play_games (repeatable, required)",
    )
    parser.add_argument(
        "--anchors",
        action="append",
        default=None,
        metavar="ID=MU",
        help="pinned anchor rating (repeatable; default random=0)",
    )
    parser.add_argument(
        "--prior-manifest",
        default=None,
        metavar="PATH",
        help=(
            "study manifest to warm-start free ids through manifest_priors; "
            "passes the same rules_id gate as load_opponents"
        ),
    )
    parser.add_argument(
        "--prior-sigma",
        type=float,
        default=None,
        metavar="FLOAT",
        help=(
            "replace the width (sigma) of every --prior-manifest prior while "
            "keeping its centre; requires --prior-manifest"
        ),
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=DEFAULT_BETA,
        help=f"probit performance-noise sd (default {DEFAULT_BETA})",
    )
    parser.add_argument(
        "--draw-margin",
        type=float,
        default=None,
        help="fixed ordered-probit draw margin; default estimates it from ties",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=200,
        help="deal-clustered bootstrap resamples; 0 disables (default 200)",
    )
    parser.add_argument(
        "--bootstrap-workers",
        type=int,
        default=4,
        metavar="INT",
        help=(
            "forked worker processes for the deal-clustered bootstrap; 1 runs "
            "serially and the result is bit-identical for every value "
            f"({_BOOTSTRAP_WORKERS_MIN}-{_BOOTSTRAP_WORKERS_MAX}, default 4)"
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="bootstrap RNG seed (default 0)"
    )
    parser.add_argument(
        "--out",
        default=None,
        metavar="PATH",
        help="write the JSON artifact here (nothing is written without it)",
    )
    parser.add_argument(
        "--manifest-out",
        default=None,
        metavar="PATH",
        help=(
            "publish the fit into this study manifest (requires --refit; "
            "the rules_id gate is enforced and the file must exist)"
        ),
    )
    parser.add_argument(
        "--refit",
        action="store_true",
        help=(
            "replace the published levels/subjects instead of freezing them "
            "(only meaningful with --manifest-out)"
        ),
    )
    parser.add_argument(
        "--keep-rungs",
        action="store_true",
        help=(
            "publish the target manifest's existing rungs (mu/sigma refreshed "
            "from the refit levels) instead of re-selecting them; requires "
            "--manifest-out --refit and a non-empty rungs list"
        ),
    )
    parser.add_argument(
        "--spec",
        action="append",
        default=None,
        metavar="ID=SPEC",
        help=(
            "policy spec for a measured subject the target manifest does not "
            "carry yet (repeatable; requires --manifest-out --refit); an "
            "existing published spec is never silently repointed"
        ),
    )
    parser.add_argument(
        "--search-config",
        action="append",
        default=None,
        metavar="ID=@PATH",
        help=(
            "seed or refresh a measured subject's published search_config "
            "from a JSON object file (repeatable; requires --manifest-out "
            "--refit); the block is the pinned search identity consumers "
            "rebuild from, so a refit preserves it when omitted"
        ),
    )
    parser.add_argument(
        "--band",
        default=None,
        metavar="ID,ID,...",
        help=(
            "comma-separated resolution band; default: every measured "
            "non-anchor id (requires --bootstrap >= 1)"
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print only the machine-readable artifact on stdout",
    )
    return parser.parse_args(argv)


def _parse_band(raw: str) -> list[str]:
    """``ID,ID,...`` as an ordered, duplicate-free band id list."""
    ids = [item.strip() for item in raw.split(",")]
    if len(ids) < 2 or any(not item for item in ids):
        raise SystemExit(
            f"--band expects at least two comma-separated ids, got {raw!r}"
        )
    if len(set(ids)) != len(ids):
        raise SystemExit(f"--band has duplicate ids: {raw!r}")
    return ids


def _parse_specs(raw: Sequence[str] | None) -> dict[str, str]:
    """``ID=SPEC`` items as a mapping; fail loud on a malformed pair or dup id.

    The spec itself is opaque here: ``rolloutt:`` is a run-local dialect the
    injected search factory owns (ADR-0010 makes ``policies`` the stock
    ``random``/``ckpt:`` owner, not the only possible spec dialect).
    """
    specs: dict[str, str] = {}
    for item in raw or ():
        if "=" not in item:
            raise SystemExit(f"--spec expects ID=SPEC, got {item!r}")
        id_, spec = item.split("=", 1)
        id_, spec = id_.strip(), spec.strip()
        if not id_:
            raise SystemExit(f"--spec expects a non-empty ID, got {item!r}")
        if not spec:
            raise SystemExit(f"--spec expects a non-empty SPEC for {id_!r}")
        if id_ in specs:
            raise SystemExit(f"--spec has duplicate id {id_!r}")
        specs[id_] = spec
    return specs


def _parse_search_configs(raw: Sequence[str] | None) -> dict[str, dict[str, Any]]:
    """``ID=@PATH`` items as ``id -> JSON object``; fail loud on malformed input.

    The block itself is opaque here (the run-local search consumers validate
    its keys); this tool owns only the publication gesture and the file
    parsing, so a typo cannot publish a half-written identity document.
    """
    configs: dict[str, dict[str, Any]] = {}
    for item in raw or ():
        if "=" not in item:
            raise SystemExit(f"--search-config expects ID=@PATH, got {item!r}")
        id_, path_text = item.split("=", 1)
        id_, path_text = id_.strip(), path_text.strip()
        if not id_:
            raise SystemExit(f"--search-config expects a non-empty ID, got {item!r}")
        if not path_text.startswith("@") or not path_text[1:].strip():
            raise SystemExit(
                f"--search-config expects ID=@PATH (a JSON object file), got {item!r}"
            )
        path = Path(path_text[1:]).expanduser()
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(
                f"--search-config {id_}: cannot read {path}: {exc}"
            ) from exc
        if not isinstance(document, Mapping):
            raise SystemExit(
                f"--search-config {id_}: {path} must be a JSON object"
            )
        if id_ in configs:
            raise SystemExit(f"--search-config has duplicate id {id_!r}")
        configs[id_] = dict(document)
    return configs


def _parse_anchors(raw: Sequence[str] | None) -> dict[str, float]:
    """``ID=MU`` items as a mapping; no ``--anchors`` means ``random=0``."""
    if raw is None:
        return {"random": 0.0}
    anchors: dict[str, float] = {}
    for item in raw:
        if "=" not in item:
            raise SystemExit(f"--anchors expects ID=MU, got {item!r}")
        id_, value = item.split("=", 1)
        id_ = id_.strip()
        if not id_:
            raise SystemExit(f"--anchors expects a non-empty ID, got {item!r}")
        try:
            mu = float(value)
        except ValueError as exc:
            raise SystemExit(f"--anchors expects ID=MU, got {item!r}") from exc
        if not math.isfinite(mu) or abs(mu) > _SCALE_BOUND:
            raise SystemExit(
                "--anchors rating must be finite and within the project scale "
                f"(|mu| <= {_SCALE_BOUND:g}), got {item!r}"
            )
        anchors[id_] = mu
    if not anchors:
        raise SystemExit("--anchors needs at least one ID=MU")
    return anchors


def _game_from_row(row: Mapping[str, Any], path: Path, number: int) -> PlayedGame:
    """One JSONL row as a ``PlayedGame``; malformed rows exit with context."""
    missing = [key for key in ("seed", "seats", "scores") if key not in row]
    if missing:
        raise SystemExit(f"{path}:{number}: missing field(s) {', '.join(missing)}")
    seed, seats, scores = row["seed"], row["seats"], row["scores"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise SystemExit(f"{path}:{number}: seed must be an int, got {seed!r}")
    if not isinstance(seats, (list, tuple)) or not all(
        isinstance(seat, str) for seat in seats
    ):
        raise SystemExit(f"{path}:{number}: seats must be a list of ids, got {seats!r}")
    if not isinstance(scores, (list, tuple)) or any(
        isinstance(score, bool) or not isinstance(score, int) for score in scores
    ):
        raise SystemExit(
            f"{path}:{number}: scores must be a list of ints, got {scores!r}"
        )
    try:
        return PlayedGame(int(seed), tuple(seats), tuple(scores))
    except ValueError as exc:
        raise SystemExit(f"{path}:{number}: {exc}") from exc


def _load_games_file(raw_path: str | Path) -> list[PlayedGame]:
    """Read one JSONL file, enforcing the current ``rules_id`` on every row."""
    expected = rules_id(DEFAULT_RULES)
    path = Path(raw_path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SystemExit(f"cannot read --games {path}: {exc}") from exc
    games: list[PlayedGame] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{number}: invalid JSON: {exc}") from exc
        if not isinstance(row, Mapping):
            raise SystemExit(f"{path}:{number}: row must be a JSON object")
        identity = row.get("rules_id")
        if identity is None:
            raise SystemExit(
                f"{path}:{number}: row has no rules_id; legacy results have "
                "no rule identity and cannot be pooled (ADR-0013). "
                "Re-measure them under the current rules."
            )
        if str(identity) != expected:
            raise SystemExit(
                f"{path}:{number}: rules_id {str(identity)!r} does not match "
                f"{expected!r}; ratings measured under different rule "
                "versions are never pooled (ADR-0013)."
            )
        games.append(_game_from_row(row, path, number))
    return games


def load_games_by_bank(
    paths: Sequence[str | Path],
) -> list[tuple[str, list[PlayedGame]]]:
    """Read each ``--games`` path as one independent deal bank.

    Returns ``(label, games)`` pairs in CLI order; a repeated path gets a
    ``#2`` suffix so two banks can never be silently conflated.  This is the
    input the leave-one-bank-out transitivity diagnostic needs (review H4):
    one file is one bank, and a single file cannot validate itself in-sample.
    """
    banks: list[tuple[str, list[PlayedGame]]] = []
    seen: dict[str, int] = {}
    for raw_path in paths:
        label = str(raw_path)
        seen[label] = seen.get(label, 0) + 1
        if seen[label] > 1:
            label = f"{label}#{seen[label]}"
        banks.append((label, _load_games_file(raw_path)))
    return banks


def load_games(paths: Sequence[str | Path]) -> list[PlayedGame]:
    """Read every JSONL row, enforcing the current ``rules_id`` on each one.

    Blank lines are skipped; a missing/mismatched identity, an unreadable file
    or a malformed row exits with ``SystemExit`` before any fit is attempted
    (ADR-0013: fail loud rather than pool across rule versions).  The bank
    boundaries are dropped here; :func:`load_games_by_bank` keeps them.
    """
    return [game for _, games in load_games_by_bank(paths) for game in games]


def _require_manifest_identity(
    document: Mapping[str, Any], *, context: str = "prior manifest"
) -> None:
    """The ``placement.load_opponents`` rules gate, as a CLI ``SystemExit``."""
    expected = rules_id(DEFAULT_RULES)
    existing = document.get("rules_id")
    if existing is not None:
        if str(existing) != expected:
            raise SystemExit(
                f"{context} rules_id {str(existing)!r} does not match "
                f"{expected!r}; cross-version ratings must not warm-start a fit "
                "(ADR-0013). Re-measure it under the current rules."
            )
        return
    if document.get("levels") or document.get("subjects"):
        raise SystemExit(
            f"{context} has measured levels/subjects but no rules_id; its "
            "ratings have no rule identity and cannot be pooled (ADR-0013). "
            "Re-measure it under the current rules."
        )


def _load_publish_manifest(path: Path) -> dict[str, Any]:
    """Read and gate the ``--manifest-out`` target before any fitting."""
    if not path.is_file():
        raise SystemExit(
            f"cannot read --manifest-out {path}: not a regular file; the "
            "publication target must already exist"
        )
    try:
        document = load_manifest(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read --manifest-out {path}: {exc}") from exc
    _require_manifest_identity(document, context="--manifest-out")
    return document


def _quantile(values: list[float], q: float) -> float:
    """Linear-interpolated quantile of an unsorted sample."""
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    lower = math.floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def percentile_cis(
    samples: Mapping[str, Sequence[float]], *, confidence: float = 0.95
) -> dict[str, tuple[float, float]]:
    """Per-id percentile CI from a bootstrap sample mapping."""
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must lie in (0, 1), got {confidence}")
    alpha = (1.0 - confidence) / 2.0
    return {
        id_: (_quantile(list(values), alpha), _quantile(list(values), 1.0 - alpha))
        for id_, values in samples.items()
    }


def _replicate_levels(
    fit: MleFit,
    *,
    ids: Sequence[str],
    anchors: Mapping[str, float],
    priors: Mapping[str, Prior],
    config: MleConfig,
) -> dict[str, float]:
    """One replicate's per-id ``mu`` with the zero-data fallbacks.

    An id absent from the replicate — possible only when all its deals were
    dropped and it has no prior — falls back to its zero-data value: the
    anchor, its prior mean, or ``config.prior.mu``.  The serial loop and the
    pool worker share this helper so both append exactly the same floats.
    """
    row: dict[str, float] = {}
    for id_ in ids:
        if id_ in fit.ratings:
            row[id_] = fit.ratings[id_].mu
        elif id_ in anchors:
            row[id_] = anchors[id_]
        elif id_ in priors:
            row[id_] = priors[id_].mu
        else:
            row[id_] = config.prior.mu
    return row


def _bootstrap_replicates(
    state: tuple[int, tuple[int, ...], float | None], count: int
) -> list[dict[str, float]]:
    """Refit ``count`` consecutive replicates from a captured MT state.

    Module-level and top-level so :mod:`pickle` can reference it across the
    fork.  The deal table, anchors, priors and config travel through the
    fork-inherited module global :data:`_BOOTSTRAP_CONTEXT` (copy-on-write)
    instead of being re-pickled per worker; only the captured RNG state and
    the replicate count are task arguments.  The worker owns a fresh
    :class:`random.Random`, so no two processes ever share mutable RNG state.
    """
    context = _BOOTSTRAP_CONTEXT
    if context is None:
        raise RuntimeError(
            "bootstrap worker has no shared context; use bootstrap_levels("
            "..., workers>=2) instead of calling this worker directly"
        )
    deals, anchors, priors, config, ids = context
    rng = random.Random()
    rng.setstate(state)
    draws = len(deals)
    rows: list[dict[str, float]] = []
    for _ in range(count):
        sample: list[PlayedGame] = []
        for _ in range(draws):
            sample.extend(deals[rng.randrange(draws)])
        rows.append(
            _replicate_levels(
                fit_mle(sample, anchors=anchors, priors=priors, config=config),
                ids=ids,
                anchors=anchors,
                priors=priors,
                config=config,
            )
        )
    return rows


def bootstrap_levels(
    games: Sequence[PlayedGame],
    *,
    anchors: Mapping[str, float],
    priors: Mapping[str, Prior],
    config: MleConfig,
    bootstrap: int,
    rng: random.Random,
    workers: int = 1,
) -> tuple[MleFit, dict[str, list[float]]]:
    """The deal-clustered refit bootstrap, keeping every replicate's levels.

    The resampling contract is exactly :func:`bootstrap_cis`'s: a deal is
    every row sharing one ``seed``, each replicate draws deals with
    replacement in sorted-seed order from the explicit ``rng`` and refits
    :func:`seven523.mle.fit_mle`, and an id absent from a replicate falls back
    to its zero-data value (anchor, prior mean, or ``config.prior.mu``).  The
    return value additionally exposes the per-id replicate ``mu`` list so the
    ``resolution`` contrasts are built from the same replicates instead of a
    second bootstrap loop.  Memory is ``ids * bootstrap`` floats; a caller
    that only wants CIs should use :func:`bootstrap_cis`.

    ``workers`` (default 1, the serial loop) fans the replicates out over a
    forked process pool.  The parent replays the serial draw stream, captures
    the exact Mersenne-Twister state at each worker's contiguous range start
    and reassembles the chunk results in replicate order, so every ``workers``
    setting reproduces the serial ``levels_by_replicate`` bit for bit; the
    effective process count is ``min(workers, bootstrap)``.  A worker failure
    propagates from ``future.result()`` instead of silently truncating.
    """
    if bootstrap < 1:
        raise ValueError(f"bootstrap must be at least 1, got {bootstrap}")
    if workers < 1:
        raise ValueError(f"workers must be at least 1, got {workers}")
    global _BOOTSTRAP_CONTEXT
    baseline = fit_mle(games, anchors=anchors, priors=priors, config=config)
    by_deal: dict[int, list[PlayedGame]] = {}
    for game in games:
        by_deal.setdefault(game.seed, []).append(game)
    deals = [by_deal[seed] for seed in sorted(by_deal)]
    samples: dict[str, list[float]] = {id_: [] for id_ in baseline.ratings}
    if not deals:
        return baseline, {
            id_: [baseline.ratings[id_].mu] * bootstrap for id_ in samples
        }
    active = min(workers, bootstrap)
    if active == 1:
        for _ in range(bootstrap):
            sample: list[PlayedGame] = []
            for _ in range(len(deals)):
                sample.extend(deals[rng.randrange(len(deals))])
            row = _replicate_levels(
                fit_mle(sample, anchors=anchors, priors=priors, config=config),
                ids=tuple(samples),
                anchors=anchors,
                priors=priors,
                config=config,
            )
            for id_ in samples:
                samples[id_].append(row[id_])
        return baseline, samples
    # Balanced contiguous ranges: replicate ``i`` stays in one chunk and the
    # chunk order is the serial replicate order.
    chunks = [
        (index * bootstrap // active, (index + 1) * bootstrap // active)
        for index in range(active)
    ]
    previous = _BOOTSTRAP_CONTEXT
    _BOOTSTRAP_CONTEXT = (deals, anchors, priors, config, tuple(samples))
    try:
        executor = ProcessPoolExecutor(
            max_workers=active,
            mp_context=multiprocessing.get_context("fork"),
        )
        try:
            draws = len(deals)
            futures = []
            for start, end in chunks:
                state = rng.getstate()
                futures.append(
                    executor.submit(_bootstrap_replicates, state, end - start)
                )
                # Advance the master stream exactly like the serial loop would,
                # so the next range starts from the state serial would see.
                for _ in range((end - start) * draws):
                    rng.randrange(draws)
            rows_by_chunk = [future.result() for future in futures]
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
    finally:
        _BOOTSTRAP_CONTEXT = previous
    for rows in rows_by_chunk:
        for row in rows:
            for id_ in samples:
                samples[id_].append(row[id_])
    return baseline, samples


def bootstrap_cis(
    games: Sequence[PlayedGame],
    *,
    anchors: Mapping[str, float],
    priors: Mapping[str, Prior],
    config: MleConfig,
    bootstrap: int,
    rng: random.Random,
    confidence: float = 0.95,
    workers: int = 1,
) -> dict[str, tuple[float, float]]:
    """Per-id percentile CI from a deal-clustered refit bootstrap.

    A deal is every row sharing one ``seed``; each replicate draws deals with
    replacement (in sorted-seed order, so the result is order-free), refits
    :func:`seven523.mle.fit_mle` and keeps every id's ``mu``.  An id absent
    from a replicate — possible only when all its deals were dropped and it has
    no prior — falls back to its zero-data value: the anchor, its prior mean,
    or ``config.prior.mu``.  ``workers`` is forwarded to
    :func:`bootstrap_levels`; it never changes the replicate stream.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must lie in (0, 1), got {confidence}")
    _, samples = bootstrap_levels(
        games,
        anchors=anchors,
        priors=priors,
        config=config,
        bootstrap=bootstrap,
        rng=rng,
        workers=workers,
    )
    return percentile_cis(samples, confidence=confidence)


def _gaussian_cdf(value: float) -> float:
    return 0.5 * math.erfc(-value / _SQRT2)


def _z_two_sided(alpha: float) -> float:
    return NormalDist().inv_cdf(1.0 - alpha / 2.0)


def _pair_win_prob(
    mu_a: float, mu_b: float, *, s: float, draw_margin: float
) -> float:
    """Tie-aware ``P(a finishes ahead of b)``; a tie counts as half a win.

    Mirrors ``mle.fit_mle``'s ordered-probit pair terms exactly:
    ``P(win) = Phi((d - eps) / s)`` and ``P(tie) = Phi((eps - d) / s) -
    Phi((-eps - d) / s)``, so ``win_prob = P(win) + 0.5 * P(tie)``.
    """
    d = mu_a - mu_b
    win = _gaussian_cdf((d - draw_margin) / s)
    tie = _gaussian_cdf((draw_margin - d) / s) - _gaussian_cdf(
        (-draw_margin - d) / s
    )
    return min(1.0, max(0.0, win + 0.5 * tie))


def _logistic_elo_diff(win_prob: float, deals: int) -> float:
    """``duel``'s ``400 * log10(p / (1 - p))`` with the same finite clamp.

    ``duel._clamp_probability`` floors ``p`` at ``1 / (2 * deals)`` (one twin
    deal contributes two games); this re-implements that semantics locally so
    ``duel.py`` stays untouched.
    """
    floor = 1.0 / (2.0 * max(int(deals), 1))
    p = min(max(win_prob, floor), 1.0 - floor)
    return DEFAULT_ELO_SCALE * math.log10(p / (1.0 - p))


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return number


def ci_only_power(
    delta_elo: float, se_elo: float, *, alpha: float = _TARGET_ALPHA
) -> float:
    """``P(two-sided CI excludes 0 | true delta)`` — the CI-only notion.

    ``Phi(delta / SE - z_{1-alpha/2})``, the design's §2.2 80%-power criterion
    (``SE <= delta / 2.8016`` at ``delta = 10``).  This is *not* the compound
    action rule's power; see :func:`action_power` for review H2's second,
    separate figure.
    """
    delta = _finite_number(delta_elo, "delta_elo")
    se = _finite_number(se_elo, "se_elo")
    if se < 0.0:
        raise ValueError(f"se_elo must be non-negative, got {se_elo!r}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha!r}")
    if se == 0.0:
        return 1.0 if delta > 0.0 else 0.0
    return _gaussian_cdf(delta / se - _z_two_sided(alpha))


def action_power(
    delta_elo: float,
    se_elo: float,
    *,
    threshold_elo: float = _TARGET_DELTA_ELO,
    alpha: float = _TARGET_ALPHA,
) -> float:
    """Power of the frozen compound rule (review H2): CI excludes 0 AND
    point estimate >= ``threshold_elo``.

    Under the normal approximation the point estimate is
    ``N(delta, se_elo**2)`` and the rule needs it above
    ``max(threshold_elo, z_{1-alpha/2} * se_elo)``.  At a true ``+10`` effect
    this is ``Phi(0) = 0.5`` for every sample size, so the 80% figure is
    unreachable for ``delta = +10``; reporting both fields keeps the CI-only
    criterion and the action rule from being conflated.
    """
    delta = _finite_number(delta_elo, "delta_elo")
    se = _finite_number(se_elo, "se_elo")
    threshold = _finite_number(threshold_elo, "threshold_elo")
    if se < 0.0:
        raise ValueError(f"se_elo must be non-negative, got {se_elo!r}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha!r}")
    if se == 0.0:
        return 1.0 if delta >= threshold else 0.0
    cut = max(threshold, _z_two_sided(alpha) * se)
    return _gaussian_cdf((delta - cut) / se)


def _sample_sd(values: Sequence[float]) -> float:
    """Sample sd (ddof = 1); a single value has sd 0.0."""
    count = len(values)
    if count < 2:
        return 0.0
    mean = math.fsum(values) / count
    variance = math.fsum((value - mean) ** 2 for value in values) / (count - 1)
    return math.sqrt(max(variance, 0.0))


def pair_deal_counts(
    games: Sequence[PlayedGame], band: Sequence[str]
) -> dict[tuple[str, str], int]:
    """Distinct deal seeds in which each band pair shares one game."""
    pairs = [(a, b) for i, a in enumerate(band) for b in band[i + 1 :]]
    seeds: dict[tuple[str, str], set[int]] = {pair: set() for pair in pairs}
    for game in games:
        seats = set(game.seats)
        for pair in pairs:
            if pair[0] in seats and pair[1] in seats:
                seeds[pair].add(game.seed)
    return {pair: len(found) for pair, found in seeds.items()}


def build_resolution(
    fit: MleFit,
    levels_by_replicate: Mapping[str, Sequence[float]],
    *,
    band: Sequence[str],
    beta: float,
    deals: Mapping[tuple[str, str], int],
    bootstrap: int,
    seed: int,
    delta_elo: float = _TARGET_DELTA_ELO,
    transitivity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The ``resolution`` block (design §4.4, H2/H4-corrected).

    Every contrast carries the point estimate (``delta_mu_probit``,
    ``win_prob`` with ties credited as half a win, ``delta_elo_logistic``
    through ``duel``'s clamped ``400 * log10`` formula and the pair's deal
    count) plus its bootstrap distribution from the same replicate levels:
    ``ci95_win_prob`` and ``se_elo``.  ``achieved.<pair>`` reports the CI-only
    ``power_at_delta`` and the separate compound-rule ``action_power_at_delta``
    so the two notions can never be conflated (review H2); ``transitivity`` is
    attached as computed by :func:`build_transitivity` (or a not-computed
    marker for a single bank).
    """
    margin = fit.draw_margin
    s = _SQRT2 * beta
    alpha = _TARGET_ALPHA
    contrasts: list[dict[str, Any]] = []
    achieved: dict[str, dict[str, Any]] = {}
    for position, a in enumerate(band):
        for b in band[position + 1 :]:
            key = f"{a}-{b}"
            mu_a = fit.ratings[a].mu
            mu_b = fit.ratings[b].mu
            deal_count = int(deals.get((a, b), 0))
            win_prob = _pair_win_prob(mu_a, mu_b, s=s, draw_margin=margin)
            win_replicates = [
                _pair_win_prob(rep_a, rep_b, s=s, draw_margin=margin)
                for rep_a, rep_b in zip(
                    levels_by_replicate[a], levels_by_replicate[b]
                )
            ]
            elo_replicates = [
                _logistic_elo_diff(value, deal_count) for value in win_replicates
            ]
            se_elo = _sample_sd(elo_replicates)
            contrasts.append(
                {
                    "a": a,
                    "b": b,
                    "delta_mu_probit": _json_number(mu_a - mu_b),
                    "win_prob": _json_number(win_prob),
                    "delta_elo_logistic": _json_number(
                        _logistic_elo_diff(win_prob, deal_count)
                    ),
                    "ci95_win_prob": [
                        _json_number(_quantile(win_replicates, alpha / 2.0)),
                        _json_number(
                            _quantile(win_replicates, 1.0 - alpha / 2.0)
                        ),
                    ],
                    "deals": deal_count,
                }
            )
            achieved[key] = {
                "se_elo": _json_number(se_elo),
                "deals": deal_count,
                "power_at_delta": _json_number(
                    ci_only_power(delta_elo, se_elo, alpha=alpha)
                ),
                "action_power_at_delta": _json_number(
                    action_power(delta_elo, se_elo, alpha=alpha)
                ),
            }
    return {
        "rules_id": rules_id(DEFAULT_RULES),
        "design": "seat-twin CRN complete round-robin",
        "band": list(band),
        "tie_credit": 0.5,
        "target": {
            "delta_elo_logistic": _json_number(delta_elo),
            "power": _TARGET_POWER,
            "alpha": alpha,
        },
        "achieved": achieved,
        "contrasts": contrasts,
        "transitivity": transitivity,
        "ci_source": f"deal-bootstrap(seed={seed}, n={bootstrap})",
        "action_rule": "CI95 excludes 0 and the point estimate is >= +10 Elo",
        "action_power_note": (
            "action_power_at_delta is the compound rule's power; at a true "
            "+10 Elo effect it is capped at 0.5 because P(point >= +10) = 0.5"
        ),
    }


def _bank_observed_z(
    games: Sequence[PlayedGame],
    prediction: float,
    key: tuple[str, str],
) -> tuple[float, int] | None:
    """One bank's deal-clustered observed-minus-model z for a band pair.

    ``prediction`` is the model's tie-aware win probability for ``key[0]``
    fitted *without* this bank (leave-one-bank-out).  Deal means are taken
    over each seed first; when the deal-clustered variance degenerates (one
    deal, or every deal identical) the binomial variance ``p(1-p)/n`` is the
    fallback, so a sign is still available for the diagnostic.
    """
    a, b = key
    outcomes_by_deal: dict[int, list[float]] = {}
    for game in games:
        if a not in game.seats or b not in game.seats:
            continue
        seat_a = game.seats.index(a)
        seat_b = game.seats.index(b)
        if game.scores[seat_a] > game.scores[seat_b]:
            outcome = 1.0
        elif game.scores[seat_a] < game.scores[seat_b]:
            outcome = 0.0
        else:
            outcome = 0.5
        outcomes_by_deal.setdefault(game.seed, []).append(outcome)
    if not outcomes_by_deal:
        return None
    deal_means = [
        math.fsum(values) / len(values) for values in outcomes_by_deal.values()
    ]
    n_deals = len(deal_means)
    n_obs = sum(len(values) for values in outcomes_by_deal.values())
    observed = math.fsum(deal_means) / n_deals
    if n_deals >= 2:
        variance = (
            math.fsum((value - observed) ** 2 for value in deal_means)
            / (n_deals - 1)
            / n_deals
        )
    else:
        variance = 0.0
    if not (variance > 0.0) or not math.isfinite(variance):
        variance = max(prediction * (1.0 - prediction), 1e-12) / max(n_obs, 1)
    se = math.sqrt(variance)
    if se <= 0.0 or not math.isfinite(se):
        return None
    return (observed - prediction) / se, n_obs


def build_transitivity(
    banks: Sequence[tuple[str, Sequence[PlayedGame]]],
    *,
    band: Sequence[str],
    anchors: Mapping[str, float],
    priors: Mapping[str, Prior],
    config: MleConfig,
) -> dict[str, Any]:
    """Leave-one-bank-out observed-vs-model residual diagnosis (review H4).

    Two or more ``--games`` files are the banks.  For every band pair each
    bank is scored against a model fitted on the *other* banks only, so the
    residual is genuinely out-of-sample; the per-bank zs are aggregated
    Stouffer-style (equal weights, ``sum(z) / sqrt(k)``).  With fewer than two
    banks the block reports ``status = "not-computed"`` with null metrics:
    an in-sample z from one bank is not a calibrated gate and is never
    published as one.
    """
    names = [name for name, _ in banks]
    base: dict[str, Any] = {
        "method": "leave-one-bank-out observed-vs-model winrate, deal clustered",
        "banks": names,
        "threshold": _TRANSITIVITY_Z,
    }
    if len(banks) < 2:
        return {
            **base,
            "status": "not-computed",
            "z": None,
            "max_abs_z": None,
            "flagged": [],
            "pairs": {},
            "reason": (
                "leave-one-bank-out needs at least two --games banks; a "
                "single bank's in-sample z is not a calibrated gate"
            ),
        }
    pairs = [(a, b) for i, a in enumerate(band) for b in band[i + 1 :]]
    stats: dict[str, dict[str, Any]] = {}
    for key in pairs:
        a, b = key
        bank_z: dict[str, float] = {}
        comparisons: dict[str, int] = {}
        for index, (name, games) in enumerate(banks):
            others = [
                game
                for other_index, (_, other) in enumerate(banks)
                if other_index != index
                for game in other
            ]
            fitted = fit_mle(
                others, anchors=anchors, priors=priors, config=config
            )
            if a not in fitted.ratings or b not in fitted.ratings:
                continue
            if fitted.ratings[a].n == 0 or fitted.ratings[b].n == 0:
                continue
            prediction = _pair_win_prob(
                fitted.ratings[a].mu,
                fitted.ratings[b].mu,
                s=_SQRT2 * config.beta,
                draw_margin=fitted.draw_margin,
            )
            result = _bank_observed_z(games, prediction, key)
            if result is None:
                continue
            z_value, n_obs = result
            bank_z[name] = z_value
            comparisons[name] = n_obs
        if len(bank_z) < 2:
            continue
        z_values = list(bank_z.values())
        aggregated = math.fsum(z_values) / math.sqrt(len(z_values))
        stats[f"{a}-{b}"] = {
            "z": _json_number(aggregated),
            "bank_z": {name: _json_number(value) for name, value in bank_z.items()},
            "comparisons": comparisons,
        }
    finite_z = [
        entry["z"] for entry in stats.values() if entry["z"] is not None
    ]
    max_abs = max((abs(value) for value in finite_z), default=None)
    flagged = sorted(
        key
        for key, entry in stats.items()
        if entry["z"] is not None and abs(entry["z"]) > _TRANSITIVITY_Z
    )
    return {
        **base,
        "status": "computed",
        "pairs": stats,
        "max_abs_z": None if max_abs is None else _json_number(max_abs),
        "flagged": flagged,
    }


def build_artifact(
    fit: MleFit,
    ci: Mapping[str, tuple[float, float]],
    *,
    anchors: Mapping[str, float],
    beta: float,
    bootstrap: int,
    seed: int,
    ci_source: str | None = None,
    prior_sigma: float | None = None,
    resolution: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The published JSON artifact with ids in sorted order (deterministic).

    Every number is passed through :func:`_json_number` so a non-finite value
    (an infinite Laplace sigma, for example) is published as ``null`` instead
    of NaN; the caller warns on stderr.  ``converged``, ``margin_identified``
    and ``ci_source`` expose the provenance ADR-0013 requires for a strict
    artifact.  ``prior_sigma`` records the explicit prior-width override (or
    ``null`` when the manifest widths were used) and ``resolution`` carries
    the per-band resolution block (or ``null`` when ``--bootstrap 0`` left no
    replicates to contrast).
    """
    if ci_source is None:
        ci_source = (
            f"deal-bootstrap(seed={seed}, n={bootstrap})"
            if bootstrap
            else "laplace(bootstrap=0)"
        )
    return {
        "version": 1,
        "rules_id": rules_id(DEFAULT_RULES),
        "rules": rules_identity(DEFAULT_RULES),
        "anchors": {id_: _json_number(anchors[id_]) for id_ in sorted(anchors)},
        "levels": {
            id_: _json_number(fit.ratings[id_].mu) for id_ in sorted(fit.ratings)
        },
        "sigmas": {
            id_: _json_number(fit.ratings[id_].sigma) for id_ in sorted(fit.ratings)
        },
        "n": {id_: fit.ratings[id_].n for id_ in sorted(fit.ratings)},
        "ci": {
            id_: [
                _json_number(ci[id_][0]),
                _json_number(ci[id_][1]),
            ]
            for id_ in sorted(fit.ratings)
        },
        "converged": fit.converged,
        "margin_identified": fit.margin_identified,
        "ci_source": ci_source,
        "separation": sorted(fit.separation),
        "estimator": {
            "kind": "probit-mle",
            "beta": _json_number(beta),
            "draw_margin": _json_number(fit.draw_margin),
            "bootstrap": bootstrap,
            "seed": seed,
            "prior_sigma": (
                None if prior_sigma is None else _json_number(prior_sigma)
            ),
        },
        "games": fit.games,
        "resolution": resolution,
    }


def format_table(fit: MleFit, ci: Mapping[str, tuple[float, float]]) -> str:
    """A mu-descending table: id, mu, sigma, n, 95% CI and the separation flag."""
    header = f"{'id':<22}{'mu':>10}{'sigma':>9}{'n':>6}{'95% CI':>27}  sep"
    lines = [header, "-" * len(header)]
    ordered = sorted(fit.ratings.items(), key=lambda item: (-item[1].mu, item[0]))
    for id_, rating in ordered:
        low, high = ci[id_]
        flag = "yes" if id_ in fit.separation else ""
        interval = f"[{low:.2f}, {high:.2f}]"
        lines.append(
            f"{id_:<22}{rating.mu:>10.2f}{rating.sigma:>9.2f}{rating.n:>6}"
            f"{interval:>27}  {flag}"
        )
    return "\n".join(lines)


def _evaluate_rungs(
    rungs: Sequence[Mapping[str, Any]],
    candidates: Sequence[float],
    *,
    count: int = _RUNG_COUNT,
    min_spacing: float = _RUNG_MIN_SPACING,
    max_spacing: float = _RUNG_MAX_SPACING,
) -> RungSelection:
    """Re-derive the ``select_rungs`` status of an already-chosen rung list.

    ``--keep-rungs`` never re-selects, but the same ``ok``/``wide_gaps``/
    ``tail_gap`` diagnostics must still be reported: the caller records them
    in the published ``estimator`` metadata so the shortfall warning never
    lives only on stderr.  Manifest coherence invariants are enforced here
    (non-empty, unique ids, strictly increasing ``mu``); a violation is a
    refused publication, not a silently inconsistent table.
    """
    selected: list[Rung] = []
    for rung in rungs:
        id_ = str(rung.get("id"))
        selected.append(
            Rung(
                id_,
                _finite_number(rung.get("mu"), f"rung {id_!r} mu"),
                _finite_number(rung.get("sigma", 0.0), f"rung {id_!r} sigma"),
            )
        )
    if not selected:
        raise ValueError("kept rungs must be non-empty")
    if len({rung.id for rung in selected}) != len(selected):
        raise ValueError("kept rungs must have unique ids")
    for previous, current in pairwise(selected):
        if current.mu <= previous.mu:
            raise ValueError("kept rungs must be strictly increasing in mu")
    wide_gaps = tuple(
        (selected[i].id, selected[i + 1].id, selected[i + 1].mu - selected[i].mu)
        for i in range(len(selected) - 1)
        if selected[i + 1].mu - selected[i].mu > max_spacing
    )
    candidate_values = list(candidates)
    tail_gap = (
        max(0.0, max(candidate_values) - selected[-1].mu)
        if candidate_values
        else 0.0
    )
    ok = (
        len(selected) >= 2
        and len(selected) == count
        and not wide_gaps
        and tail_gap <= max_spacing
    )
    return RungSelection(
        tuple(selected), count, min_spacing, max_spacing, wide_gaps, tail_gap, ok
    )


def _prepare_publication(
    document: Mapping[str, Any],
    fit: MleFit,
    *,
    anchors: Mapping[str, float],
    beta: float,
    bootstrap: int,
    seed: int,
    source: str,
    prior_sigma: float | None = None,
    keep_rungs: bool = False,
    resolution: Mapping[str, Any] | None = None,
    specs: Mapping[str, str] | None = None,
    search_configs: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any], RungSelection]:
    """Build the merged study manifest and rung selection without saving.

    Split out of :func:`publish_manifest` so the caller validates the
    publication before any file is written: a refused publication must not
    leave a half-updated manifest (and must not write the artifact first).
    Prior-only ids (``n == 0`` and not pinned) are excluded from the published
    ``levels``/``subjects``/rungs: the table artifact still carries them with
    ``n == 0``, but they are not measurements (review H3).

    ``specs`` (``--spec ID=SPEC``) supplies the policy spec for a measured id
    the target manifest does not carry yet, so a new subject is never
    published spec-less.  An id not in the fit, or a spec that would repoint
    an already published id, is refused.  ``search_configs``
    (``--search-config ID=@PATH``) seeds or refreshes a measured subject's
    published ``search_config`` block; an id not in the fit is refused, and a
    block already published is preserved through the refit when omitted.
    """
    previous = {
        str(entry["id"]): dict(entry)
        for entry in document.get("subjects") or []
        if isinstance(entry, Mapping) and "id" in entry
    }
    measured = {
        id_: rating
        for id_, rating in fit.ratings.items()
        if rating.n > 0 or id_ in anchors
    }
    # ``merge_manifest(refit=True)`` overlays the incoming levels on the old
    # mapping instead of clearing it, so the base document is pruned to the
    # measured ids here: a prior-only ghost (or any stale id) must not survive
    # as a published level (review H3).
    base_document = dict(document)
    base_document["levels"] = {
        str(id_): value
        for id_, value in (document.get("levels") or {}).items()
        if str(id_) in measured
    }
    base_document["subjects"] = [
        entry
        for entry in document.get("subjects") or []
        if isinstance(entry, Mapping) and str(entry.get("id")) in measured
    ]
    base_document["rungs"] = [
        entry
        for entry in document.get("rungs") or []
        if isinstance(entry, Mapping) and str(entry.get("id")) in measured
    ]
    specs = dict(specs or {})
    unknown = sorted(set(specs) - set(measured))
    if unknown:
        raise ValueError(
            "--spec id(s) not measured by this fit: " + ", ".join(unknown)
        )
    search_configs = dict(search_configs or {})
    unknown_configs = sorted(set(search_configs) - set(measured))
    if unknown_configs:
        raise ValueError(
            "search_config id(s) not measured by this fit: "
            + ", ".join(unknown_configs)
        )
    subjects: list[dict[str, Any]] = []
    for id_ in sorted(measured):
        rating = measured[id_]
        entry = previous.get(id_, {"id": id_})
        entry["id"] = id_
        entry["mu"] = _json_number(rating.mu)
        entry["sigma"] = _json_number(rating.sigma)
        entry["games"] = rating.n
        explicit = specs.get(id_)
        if explicit is not None:
            published = entry.get("spec")
            if published is not None and str(published) != explicit:
                raise ValueError(
                    f"--spec {id_!r} would repoint the published spec "
                    f"{published!r} -> {explicit!r}; edit the manifest "
                    "explicitly if that is intended"
                )
            entry["spec"] = explicit
        subjects.append(entry)

    anchor_entries: list[dict[str, Any]] = []
    for entry in document.get("anchors") or []:
        anchor = dict(entry)
        id_ = str(anchor.get("id"))
        if id_ in anchors:
            anchor["mu"] = _json_number(anchors[id_])
        anchor_entries.append(anchor)
    known_anchor_ids = {str(entry.get("id")) for entry in anchor_entries}
    for id_, mu in sorted(anchors.items()):
        if id_ not in known_anchor_ids:
            anchor_entries.append({"id": id_, "mu": _json_number(mu)})

    candidates = {
        id_: rating for id_, rating in measured.items() if id_ not in anchors
    }
    if keep_rungs:
        kept = list(document.get("rungs") or [])
        if not kept:
            raise ValueError(
                "--keep-rungs: the target manifest has no published rungs "
                "to keep"
            )
        rung_entries: list[dict[str, Any]] = []
        for rung in kept:
            id_ = str(rung.get("id"))
            if id_ not in measured:
                raise ValueError(
                    f"--keep-rungs: rung {id_!r} is not in the refit levels; "
                    "refusing to publish a rung without a level"
                )
            rating = measured[id_]
            entry = dict(rung)
            entry["id"] = id_
            entry["mu"] = _json_number(rating.mu)
            entry["sigma"] = _json_number(rating.sigma)
            rung_entries.append(entry)
        selection = _evaluate_rungs(
            rung_entries, [rating.mu for rating in candidates.values()]
        )
        rung_source = "kept"
    else:
        selection = select_rungs(
            candidates,
            count=_RUNG_COUNT,
            min_spacing=_RUNG_MIN_SPACING,
            max_spacing=_RUNG_MAX_SPACING,
        )
        rung_entries = [
            {
                "id": rung.id,
                "mu": _json_number(rung.mu),
                "sigma": _json_number(rung.sigma),
            }
            for rung in selection.rungs
        ]
        rung_source = "select_rungs"
    estimator = {
        "kind": "probit-mle",
        "beta": _json_number(beta),
        "draw_margin": _json_number(fit.draw_margin),
        "bootstrap": bootstrap,
        "seed": seed,
        "games": fit.games,
        "source": str(source),
        "prior_sigma": None if prior_sigma is None else _json_number(prior_sigma),
        "rungs_source": rung_source,
        "rungs": [str(entry["id"]) for entry in rung_entries],
        "rungs_selection_ok": selection.ok,
        "rungs_wide_gaps": [
            [left, right, _json_number(gap)]
            for left, right, gap in selection.wide_gaps
        ],
        "rungs_tail_gap": _json_number(selection.tail_gap),
    }
    merged = merge_manifest(
        base_document,
        levels={id_: _json_number(rating.mu) for id_, rating in measured.items()},
        subjects=subjects,
        anchors=anchor_entries,
        rungs=rung_entries,
        estimator=estimator,
        resolution=resolution,
        search_configs=search_configs,
        refit=True,
        rules_id=rules_id(DEFAULT_RULES),
        rules=rules_identity(DEFAULT_RULES),
    )
    # A refit replaces the level contract, so a subject that is no longer a
    # published level (a prior-only ghost carried by an earlier manifest) must
    # not survive the merge.
    published_ids = set(merged.get("levels") or {})
    merged["subjects"] = [
        entry
        for entry in merged.get("subjects") or []
        if str(entry.get("id")) in published_ids
    ]
    return merged, selection


def publish_manifest(
    path: str | Path,
    document: Mapping[str, Any],
    fit: MleFit,
    *,
    anchors: Mapping[str, float],
    beta: float,
    bootstrap: int,
    seed: int,
    source: str,
    prior_sigma: float | None = None,
    keep_rungs: bool = False,
    resolution: Mapping[str, Any] | None = None,
    specs: Mapping[str, str] | None = None,
    search_configs: Mapping[str, Mapping[str, Any]] | None = None,
) -> RungSelection:
    """Merge a fit into ``document`` and publish it as the study manifest.

    ``levels`` and ``subjects`` are rebuilt from the fit ids that were actually
    measured (``n > 0``, plus pinned anchors); each subject keeps its previous
    fields — notably ``spec``, ``search_config`` and any future metadata — and
    receives the fitted ``mu``/``sigma`` plus the MLE game count.  ``specs``
    optionally supplies the published spec for a new measured id (and only
    ever confirms an existing one); ``search_configs`` seeds or refreshes a
    subject's published search identity block.  The manifest's anchor
    entries are preserved (and their ``mu`` refreshed).  ``estimator`` records
    the full provenance: games ``source``, ``prior_sigma``, and the rung
    selection status (``rungs_source``/``rungs_selection_ok``/...).  By
    default ``rungs`` are re-selected from the non-anchor levels under the
    published spacing contract; with ``keep_rungs=True`` the target manifest's
    existing selection is preserved (ids and order), its ``mu``/``sigma``
    refreshed from the refit levels, and incoherent rungs are refused.  The
    merge goes through :func:`seven523.study.merge_manifest` with
    ``refit=True`` and the current rule identity; the document is saved with
    :func:`seven523.study.save_manifest`.  Returns the rung selection so the
    caller can report shortfalls and wide gaps as well as record them.
    """
    merged, selection = _prepare_publication(
        document,
        fit,
        anchors=anchors,
        beta=beta,
        bootstrap=bootstrap,
        seed=seed,
        source=source,
        prior_sigma=prior_sigma,
        keep_rungs=keep_rungs,
        resolution=resolution,
        specs=specs,
        search_configs=search_configs,
    )
    save_manifest(Path(path), merged)
    return selection


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.bootstrap < 0:
        raise SystemExit(f"--bootstrap must be non-negative, got {args.bootstrap}")
    if not (
        _BOOTSTRAP_WORKERS_MIN <= args.bootstrap_workers <= _BOOTSTRAP_WORKERS_MAX
    ):
        raise SystemExit(
            f"--bootstrap-workers must be between {_BOOTSTRAP_WORKERS_MIN} and "
            f"{_BOOTSTRAP_WORKERS_MAX}, got {args.bootstrap_workers}"
        )
    if args.prior_sigma is not None:
        if args.prior_manifest is None:
            raise SystemExit("--prior-sigma requires --prior-manifest")
        if not math.isfinite(args.prior_sigma) or args.prior_sigma <= 0.0:
            raise SystemExit(
                f"--prior-sigma must be positive and finite, got {args.prior_sigma!r}"
            )
    if args.keep_rungs and args.manifest_out is None:
        raise SystemExit("--keep-rungs requires --manifest-out --refit")
    if args.spec and args.manifest_out is None:
        raise SystemExit("--spec requires --manifest-out --refit")
    if args.search_config and args.manifest_out is None:
        raise SystemExit("--search-config requires --manifest-out --refit")
    if args.manifest_out is not None and not args.refit:
        raise SystemExit(
            "--manifest-out requires an explicit --refit; refusing to publish "
            "a frozen manifest by accident"
        )
    specs = _parse_specs(args.spec)
    search_configs = _parse_search_configs(args.search_config)
    band = _parse_band(args.band) if args.band is not None else None
    if band is not None and args.bootstrap == 0:
        raise SystemExit(
            "--band requires --bootstrap >= 1; there are no replicate levels "
            "to contrast"
        )
    publish_document: dict[str, Any] | None = None
    if args.manifest_out is not None:
        publish_document = _load_publish_manifest(Path(args.manifest_out))
        if args.keep_rungs and not (publish_document.get("rungs") or []):
            raise SystemExit(
                f"--keep-rungs: {args.manifest_out} has no published rungs to keep"
            )
    anchors = _parse_anchors(args.anchors)
    banks = load_games_by_bank(args.games)
    games = [game for _, bank_games in banks for game in bank_games]
    priors: dict[str, Prior] = {}
    if args.prior_manifest is not None:
        manifest_path = Path(args.prior_manifest)
        if not manifest_path.is_file():
            raise SystemExit(
                f"cannot read --prior-manifest {manifest_path}: not a regular file"
            )
        try:
            document = load_manifest(manifest_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise SystemExit(
                f"cannot read --prior-manifest {manifest_path}: {exc}"
            ) from exc
        _require_manifest_identity(document)
        priors = manifest_priors(document)
        if args.prior_sigma is not None:
            # Keep every published centre, replace only the MAP penalty width
            # so the fit is shrunk by an explicit, recorded amount (M1).
            priors = {
                id_: Prior(mu=prior.mu, sigma=args.prior_sigma)
                for id_, prior in priors.items()
            }
        for id_, prior in priors.items():
            if not math.isfinite(prior.mu) or abs(prior.mu) > _SCALE_BOUND:
                raise SystemExit(
                    f"--prior-manifest {manifest_path}: prior centre "
                    f"{prior.mu!r} for {id_!r} is outside the project scale "
                    f"(|mu| <= {_SCALE_BOUND:g})"
                )
    # A pinned anchor must never appear in priors (fit_mle rejects it), and a
    # real study manifest always carries its gauge level.
    priors = {id_: prior for id_, prior in priors.items() if id_ not in anchors}
    try:
        config = MleConfig(beta=args.beta, draw_margin=args.draw_margin)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    try:
        fit = fit_mle(games, anchors=anchors, priors=priors, config=config)
        if args.bootstrap:
            _, samples = bootstrap_levels(
                games,
                anchors=anchors,
                priors=priors,
                config=config,
                bootstrap=args.bootstrap,
                rng=random.Random(args.seed),
                workers=args.bootstrap_workers,
            )
            ci = percentile_cis(samples)
        else:
            ci = {
                id_: (
                    rating.mu - _Z95 * rating.sigma,
                    rating.mu + _Z95 * rating.sigma,
                )
                for id_, rating in fit.ratings.items()
            }
    except ArithmeticError as exc:
        raise SystemExit(f"probit-MLE fit failed: {exc}") from exc
    if not fit.converged:
        print(
            "warning: probit-MLE stopped without reaching the numerical "
            "optimum; treat the table as provisional",
            file=sys.stderr,
        )
    non_finite = sorted(
        id_
        for id_ in fit.ratings
        if not (
            math.isfinite(fit.ratings[id_].mu)
            and math.isfinite(fit.ratings[id_].sigma)
            and math.isfinite(ci[id_][0])
            and math.isfinite(ci[id_][1])
        )
    )
    if non_finite:
        print(
            "warning: non-finite estimate(s) for "
            + ", ".join(non_finite)
            + "; publishing null (see converged / margin_identified)",
            file=sys.stderr,
        )
    ci_source = (
        f"deal-bootstrap(seed={args.seed}, n={args.bootstrap})"
        if args.bootstrap
        else "laplace(bootstrap=0)"
    )
    resolution: dict[str, Any] | None = None
    if args.bootstrap:
        default_band = sorted(
            (
                id_
                for id_, rating in fit.ratings.items()
                if id_ not in anchors and rating.n > 0
            ),
            key=lambda id_: (-fit.ratings[id_].mu, id_),
        )
        band = default_band if band is None else band
        for id_ in band:
            if id_ not in fit.ratings:
                raise SystemExit(f"--band id {id_!r} is not in the fit")
            if fit.ratings[id_].n == 0 and id_ not in anchors:
                raise SystemExit(
                    f"--band id {id_!r} is prior-only (n = 0); the "
                    "resolution band needs measured ids"
                )
        if len(band) >= 2:
            resolution = build_resolution(
                fit,
                samples,
                band=band,
                beta=config.beta,
                deals=pair_deal_counts(games, band),
                bootstrap=args.bootstrap,
                seed=args.seed,
                transitivity=build_transitivity(
                    banks,
                    band=band,
                    anchors=anchors,
                    priors=priors,
                    config=config,
                ),
            )
    # Validate the publication before any file is written: a refused merge
    # must not leave a written artifact or a half-updated manifest.
    merged_publication: dict[str, Any] | None = None
    selection: RungSelection | None = None
    if publish_document is not None and args.manifest_out is not None:
        try:
            merged_publication, selection = _prepare_publication(
                publish_document,
                fit,
                anchors=anchors,
                beta=config.beta,
                bootstrap=args.bootstrap,
                seed=args.seed,
                source=",".join(str(path) for path in args.games),
                prior_sigma=args.prior_sigma,
                keep_rungs=args.keep_rungs,
                resolution=resolution,
                specs=specs,
                search_configs=search_configs,
            )
        except ValueError as exc:
            raise SystemExit(f"--manifest-out refused: {exc}") from exc
    artifact = build_artifact(
        fit,
        ci,
        anchors=anchors,
        beta=config.beta,
        bootstrap=args.bootstrap,
        seed=args.seed,
        ci_source=ci_source,
        prior_sigma=args.prior_sigma,
        resolution=resolution,
    )
    if merged_publication is not None and selection is not None:
        # Mirror the rung status into the machine-readable artifact as well as
        # the manifest estimator: the select_rungs shortfall must not live
        # only on stderr.
        artifact["estimator"].update(
            {
                "rungs_source": "kept" if args.keep_rungs else "select_rungs",
                "rungs": [rung.id for rung in selection.rungs],
                "rungs_selection_ok": selection.ok,
            }
        )
    # One serialization for both the artifact file and the --json stdout, so
    # the two channels are byte-identical (a verifier regression check).
    artifact_text = json.dumps(
        artifact, ensure_ascii=False, indent=2, allow_nan=False
    )
    if args.out is not None:
        out = Path(args.out)
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(artifact_text + "\n", encoding="utf-8")
        except OSError as exc:
            raise SystemExit(f"cannot write --out {out}: {exc}") from exc
        # stdout stays parseable JSON under --json.
        print(f"wrote: {out}", file=sys.stderr if args.json else sys.stdout)
    if merged_publication is not None and selection is not None:
        save_manifest(args.manifest_out, merged_publication)
        rung_text = ", ".join(
            f"{rung.id}({rung.mu:.2f})" for rung in selection.rungs
        )
        wide_text = ", ".join(
            f"{left}->{right} {gap:.2f}" for left, right, gap in selection.wide_gaps
        )
        print(
            f"published: {args.manifest_out} rungs=[{rung_text}] "
            f"wide_gaps=[{wide_text}] tail_gap={selection.tail_gap:.2f} "
            f"requested={selection.requested} ok={selection.ok}",
            file=sys.stderr,
        )
    if args.json:
        print(artifact_text)
    else:
        print(format_table(fit, ci))
        print(
            f"\nestimator: probit-mle, {fit.games} games, "
            f"draw_margin={fit.draw_margin:.6g}, ci={ci_source}, "
            f"converged={fit.converged}, "
            f"margin_identified={fit.margin_identified}, "
            f"separation={sorted(fit.separation)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

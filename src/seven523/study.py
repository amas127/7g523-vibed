"""Study manifests: the one owner of the ``traces/study`` document schema.

Both ends of the M2 ladder workflow cross this seam: ``ladder.build_ladder``
writes measured ratings into it, and the D1 calibration reads
``levels`` (``id -> measured rating mu``) to label the S1 feature rows.  Keeping
the schema — and the freeze rule — here means neither side re-implements it.

``merge_manifest`` freezes ratings by default: an id that already has a level
keeps its whole published record (level and subject entry) even if a new fit
produced different values, so traces already on disk stay consistent with the
labels published for them and a normal (non-refit) merge cannot clobber the
published estimator provenance.  Re-fitting is an explicit ``refit=True``
decision.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

__all__ = [
    "MANIFEST_VERSION",
    "RESOLUTION_REQUIRED_KEYS",
    "load_manifest",
    "merge_manifest",
    "save_manifest",
]

MANIFEST_VERSION = 1

#: Keys a published ``resolution`` block must carry.  Unknown keys are
#: preserved (forward compatibility), but the required ones pin the block's
#: rule identity, band, target, achieved resolutions, contrasts, holdout
#: transitivity diagnosis and CI provenance.
RESOLUTION_REQUIRED_KEYS = (
    "rules_id",
    "band",
    "target",
    "achieved",
    "contrasts",
    "transitivity",
    "ci_source",
)


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Read a manifest; a missing file is an empty study, malformed JSON is an error."""
    path = Path(path)
    if not path.exists():
        return {}
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"manifest {path} must be a JSON object")
    return document


def save_manifest(path: str | Path, document: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(document), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path


def _merge_frozen_rungs(
    published: Sequence[Mapping[str, Any]],
    incoming: Sequence[Mapping[str, Any]],
    frozen_levels: Mapping[str, float],
) -> list[dict[str, Any]]:
    """Keep the published rung list and append only rung ids it lacks.

    A normal (non-refit) merge must not replace the published rung selection
    with the fresh fit's selection: the fresh fit may have measured a
    different value for an id the levels contract froze.  A new rung id is
    appended with ``mu`` re-based on the merged frozen levels when they have
    it, so a rung can never disagree with its frozen level.
    """
    merged_rungs = [dict(rung) for rung in published]
    known = {str(rung.get("id")) for rung in merged_rungs}
    for rung in incoming:
        rung = dict(rung)
        id_ = str(rung.get("id"))
        if id_ in known:
            continue
        if id_ in frozen_levels:
            rung["mu"] = frozen_levels[id_]
        merged_rungs.append(rung)
        known.add(id_)
    return merged_rungs


def _check_finite_numbers(value: Any, *, path: str) -> None:
    """Reject NaN/Infinity (and non-JSON values) anywhere in a block."""
    if value is None or isinstance(value, (str, bool, int)):
        # Python ints are finite by construction (and can exceed float range,
        # where ``math.isfinite`` would raise OverflowError).
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _check_finite_numbers(item, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _check_finite_numbers(item, path=f"{path}[{index}]")
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must be finite, got {value!r}")
        return
    raise ValueError(f"{path} must be a JSON value, got {value!r}")


def _validate_search_config(config: Any, *, path: str) -> dict[str, Any]:
    """The publication gate for a subject ``search_config`` block.

    The block is opaque to this module (consumers own its keys), but it must
    be a mapping of JSON values with no NaN/Infinity: the manifest is the
    published identity source, so a malformed block must be refused at merge
    time rather than discovered mid-session.
    """
    if not isinstance(config, Mapping):
        raise ValueError(f"{path} must be a mapping, got {config!r}")
    _check_finite_numbers(config, path=path)
    return dict(config)


def _validate_resolution(
    resolution: Mapping[str, Any], *, rules_id: str | None
) -> None:
    """The publication gate for a ``resolution`` block (schema owner).

    Required keys must be present, ``rules_id`` must be a non-empty string
    matching the merge (when the merge carries one), and every number in the
    block must be finite — a NaN/Infinity never reaches a saved manifest.
    Unknown keys are deliberately preserved: the schema grows by adding keys,
    so a forward-compatible writer is not rejected.  That behaviour is pinned
    by ``tests/test_study.py::test_merge_resolution_preserves_unknown_keys``.
    """
    if not isinstance(resolution, Mapping):
        raise ValueError(f"resolution must be a mapping, got {resolution!r}")
    missing = [key for key in RESOLUTION_REQUIRED_KEYS if key not in resolution]
    if missing:
        raise ValueError(
            "resolution is missing required key(s): " + ", ".join(missing)
        )
    block_rules = resolution.get("rules_id")
    if not isinstance(block_rules, str) or not block_rules:
        raise ValueError("resolution.rules_id must be a non-empty string")
    if rules_id is not None and block_rules != rules_id:
        raise ValueError(
            f"resolution rules_id {block_rules!r} does not match {rules_id!r}; "
            "ratings from different rule versions must not be pooled"
        )
    _check_finite_numbers(resolution, path="resolution")


def merge_manifest(
    document: Mapping[str, Any],
    *,
    levels: Mapping[str, float],
    subjects: Sequence[Mapping[str, Any]],
    anchors: Sequence[Mapping[str, Any]] | None = None,
    rungs: Sequence[Mapping[str, Any]] | None = None,
    estimator: Mapping[str, Any] | None = None,
    resolution: Mapping[str, Any] | None = None,
    search_configs: Mapping[str, Mapping[str, Any]] | None = None,
    frozen_at: str | None = None,
    refit: bool = False,
    rules: Mapping[str, Any] | None = None,
    rules_id: str | None = None,
) -> dict[str, Any]:
    """Merge a fit result into an existing manifest and return the new document.

    ``levels`` is the contract the D1 calibration reads; it is frozen per id
    unless ``refit`` is set.  ``subjects`` merge by id under the same freeze:
    a subject whose level is already published keeps its whole published entry
    (``mu``, ``spec``, ``sigma`` and ``games``), so a normal merge cannot
    refresh the measurement metadata of a frozen rung or silently repoint a
    published id at a different checkpoint.  Only ids new to the levels
    contract take the incoming entry.

    ``rungs`` follow the frozen levels too: without ``refit`` the published
    rung list is preserved and only ids it does not contain yet are appended
    (with ``mu`` re-based on the merged frozen level).  ``refit=True`` replaces
    both the levels and the rung selection.  All unrelated keys (D1 metadata
    such as ``seed``/``games``/``num_players``) are preserved.

    ``estimator`` records the fit provenance.  Without ``refit`` an existing
    published estimator block is kept — a normal ``build_ladder`` merge must
    not replace the published probit-MLE provenance with its own OpenSkill
    metadata — and the incoming block is written only when the document has
    none yet.  ``refit=True`` replaces it.

    ``resolution`` follows the same freeze: without ``refit`` a published
    block is kept and the incoming one is only stamped when the document has
    none; ``refit=True`` replaces it.  Whenever a block is written it is
    validated by :func:`_validate_resolution` (required keys, matching
    non-empty ``rules_id``, finite numbers); unknown keys survive.

    ``search_configs`` maps a subject id to its published ``search_config``
    block (the pinned search identity consumers rebuild from).  A refit never
    drops a previously published block just because the incoming subject
    entry omits it; the explicit mapping seeds or refreshes it and is refused
    for an id without a subject.  Every block that survives is validated as a
    JSON mapping by :func:`_validate_search_config`.

    ``rules_id`` gates pooling across rule versions (ADR-0013).  When given,
    a document carrying a different id is refused, as is a legacy document
    that already holds measured ``levels``/``subjects`` but no id; a document
    with no measured payload is stamped.  ``rules`` is the optional identity
    payload, refreshed on a matching merge and stamped with a fresh document.
    With ``rules_id=None`` the merge behaves exactly as before the gate.
    """
    if rules_id is not None:
        existing_id = document.get("rules_id")
        if existing_id is not None and existing_id != rules_id:
            raise ValueError(
                f"manifest rules_id {existing_id!r} does not match {rules_id!r}; "
                "ratings from different rule versions must not be pooled"
            )
        if existing_id is None and (
            document.get("levels") or document.get("subjects")
        ):
            raise ValueError(
                "manifest has measured levels/subjects but no rules_id; "
                "re-measure under the current rules before merging (ADR-0013)"
            )

    merged = dict(document)
    previous_levels = set(merged.get("levels") or {})
    known_levels = dict(merged.get("levels") or {})
    if refit:
        known_levels.update(levels)
    else:
        for id_, elo in levels.items():
            known_levels.setdefault(id_, elo)
    merged["levels"] = known_levels

    known_subjects: dict[str, dict[str, Any]] = {}
    for entry in merged.get("subjects") or []:
        if "id" not in entry:
            raise ValueError(f"manifest subject without an id: {entry!r}")
        known_subjects[str(entry["id"])] = dict(entry)
    previous_subjects = {
        id_: dict(entry) for id_, entry in known_subjects.items()
    }
    for entry in subjects:
        entry = dict(entry)
        if "id" not in entry:
            raise ValueError(f"subject without an id: {entry!r}")
        id_ = str(entry["id"])
        previous = known_subjects.get(id_)
        if previous is not None and not refit and id_ in previous_levels:
            # Frozen level: keep the published record whole so sigma/games (the
            # measurement provenance) cannot be refreshed by a normal merge.
            entry = dict(previous)
        elif refit and "search_config" not in entry:
            # A refit replaces the subject entry; the published search
            # identity block is not part of the incoming fit and must survive.
            published_config = (previous or previous_subjects.get(id_) or {}).get(
                "search_config"
            )
            if published_config is not None:
                entry["search_config"] = published_config
        known_subjects[id_] = entry
    if search_configs is not None:
        for raw_id, config in search_configs.items():
            id_ = str(raw_id)
            entry = known_subjects.get(id_)
            if entry is None:
                raise ValueError(
                    f"search_config for unknown subject {id_!r}; refusing to "
                    "create a subject without a spec"
                )
            entry["search_config"] = _validate_search_config(
                config, path=f"subjects[{id_}].search_config"
            )
    for id_, entry in known_subjects.items():
        if "search_config" in entry:
            entry["search_config"] = _validate_search_config(
                entry["search_config"], path=f"subjects[{id_}].search_config"
            )
    merged["subjects"] = list(known_subjects.values())

    if anchors is not None:
        merged["anchors"] = [dict(entry) for entry in anchors]
    if rungs is not None:
        if refit:
            merged["rungs"] = [dict(rung) for rung in rungs]
        else:
            merged["rungs"] = _merge_frozen_rungs(
                merged.get("rungs") or [], rungs, known_levels
            )
    if estimator is not None and (refit or "estimator" not in merged):
        merged["estimator"] = dict(estimator)
    if resolution is not None:
        if not isinstance(resolution, Mapping):
            raise ValueError(f"resolution must be a mapping, got {resolution!r}")
        if refit or "resolution" not in merged:
            candidate = dict(resolution)
            _validate_resolution(candidate, rules_id=rules_id)
            merged["resolution"] = candidate
    if frozen_at is not None:
        merged["frozen_at"] = frozen_at
    if rules_id is not None:
        merged["rules_id"] = rules_id
        if rules is not None:
            merged["rules"] = dict(rules)
    merged.setdefault("version", MANIFEST_VERSION)
    return merged

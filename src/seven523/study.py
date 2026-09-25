"""Study manifests: the one owner of the ``traces/study`` document schema.

Both ends of the M2 ladder workflow cross this seam: ``ladder.build_ladder``
writes measured ratings into it, and ``tools/measure_trace_signal.py`` reads
``levels`` (``id -> measured Elo``) to label the S1 feature rows.  Keeping the
schema — and the freeze rule — here means neither side re-implements it.

``merge_manifest`` freezes ratings by default: an id that already has a level
keeps its old value even if a new fit produced a different one, so traces
already on disk stay consistent with the labels published for them.  Re-fitting
is an explicit ``refit=True`` decision.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

__all__ = [
    "MANIFEST_NAME",
    "MANIFEST_VERSION",
    "load_manifest",
    "merge_manifest",
    "save_manifest",
]

MANIFEST_VERSION = 1
MANIFEST_NAME = "manifest.json"


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


def merge_manifest(
    document: Mapping[str, Any],
    *,
    levels: Mapping[str, float],
    subjects: Sequence[Mapping[str, Any]],
    anchors: Sequence[Mapping[str, Any]] | None = None,
    rungs: Sequence[Mapping[str, Any]] | None = None,
    estimator: Mapping[str, Any] | None = None,
    frozen_at: str | None = None,
    refit: bool = False,
) -> dict[str, Any]:
    """Merge a fit result into an existing manifest and return the new document.

    ``levels`` is the contract ``tools/measure_trace_signal.py`` reads; it is
    frozen per id unless ``refit`` is set.  ``subjects`` merge by id with the
    same rule for their ``elo``, while their other metadata refreshes (games,
    spec, ``se``).  All unrelated keys (D1 metadata such as ``seed``/``games``/
    ``num_players``) are preserved.
    """
    merged = dict(document)
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
    for entry in subjects:
        entry = dict(entry)
        if "id" not in entry:
            raise ValueError(f"subject without an id: {entry!r}")
        id_ = str(entry["id"])
        previous = known_subjects.get(id_)
        if previous is not None and not refit and "elo" in previous:
            entry["elo"] = previous["elo"]
        known_subjects[id_] = entry
    merged["subjects"] = list(known_subjects.values())

    if anchors is not None:
        merged["anchors"] = [dict(entry) for entry in anchors]
    if rungs is not None:
        merged["rungs"] = [dict(rung) for rung in rungs]
    if estimator is not None:
        merged["estimator"] = dict(estimator)
    if frozen_at is not None:
        merged["frozen_at"] = frozen_at
    merged.setdefault("version", MANIFEST_VERSION)
    return merged

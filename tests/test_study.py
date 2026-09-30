"""Study manifest schema: round-trips, freeze semantics and D1 compatibility."""
from __future__ import annotations

import json
import math

import pytest

from seven523.rules import DEFAULT_RULES, rules_id, rules_identity
from seven523.study import load_manifest, merge_manifest, save_manifest


def test_missing_manifest_is_an_empty_study(tmp_path):
    assert load_manifest(tmp_path / "manifest.json") == {}


def test_manifest_round_trip(tmp_path):
    path = tmp_path / "study" / "manifest.json"
    document = {"version": 1, "seed": 7, "levels": {"lvl1": 1432.0}}
    assert save_manifest(path, document) == path
    assert load_manifest(path) == document


def test_malformed_manifest_raises(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError):
        load_manifest(path)


def test_merge_freezes_levels_and_subject_mu_by_default():
    document = {
        "seed": 3,
        "levels": {"random": 1000.0, "lvl1": 1400.0},
        "subjects": [{"id": "lvl1", "spec": "ckpt:a.pt", "mu": 1400.0}],
    }
    merged = merge_manifest(
        document,
        levels={"random": 1000.0, "lvl1": 1450.0, "lvl2": 1600.0},
        subjects=[
            {"id": "lvl1", "spec": "ckpt:a.pt", "mu": 1450.0, "games": 200},
            {"id": "lvl2", "spec": "ckpt:b.pt", "mu": 1600.0, "games": 200},
        ],
    )
    assert merged["levels"] == {"random": 1000.0, "lvl1": 1400.0, "lvl2": 1600.0}
    subjects = {entry["id"]: entry for entry in merged["subjects"]}
    # Published level: the whole subject record is frozen, including the
    # sigma/games measurement provenance a normal merge must not refresh.
    assert subjects["lvl1"] == {"id": "lvl1", "spec": "ckpt:a.pt", "mu": 1400.0}
    # New level: the incoming entry is taken whole.
    assert subjects["lvl2"] == {
        "id": "lvl2",
        "spec": "ckpt:b.pt",
        "mu": 1600.0,
        "games": 200,
    }
    assert merged["seed"] == 3  # D1 metadata preserved


def test_merge_refit_overwrites_and_accepts_extra_fields():
    document = {"levels": {"lvl1": 1400.0}}
    merged = merge_manifest(
        document,
        levels={"lvl1": 1450.0},
        subjects=[{"id": "lvl1", "mu": 1450.0}],
        anchors=[{"id": "random", "mu": 1000.0}],
        rungs=[{"id": "lvl1", "mu": 1450.0, "sigma": 30.0}],
        estimator={"kind": "openskill-plackett-luce", "games": 5},
        frozen_at="2025-01-01T00:00:00",
        refit=True,
    )
    assert merged["levels"] == {"lvl1": 1450.0}
    assert merged["subjects"][0]["mu"] == 1450.0
    assert merged["anchors"] == [{"id": "random", "mu": 1000.0}]
    assert merged["rungs"][0]["id"] == "lvl1"
    assert merged["estimator"]["kind"] == "openskill-plackett-luce"
    assert merged["frozen_at"] == "2025-01-01T00:00:00"
    assert merged["version"] == 1


def test_merge_without_refit_keeps_published_rungs_and_subject_spec():
    document = {
        "levels": {"random": 0.0, "lvl1": 54.85, "lvl3": 191.74},
        "subjects": [
            {"id": "lvl1", "spec": "ckpt:lvl1.pt", "mu": 54.85, "games": 100},
            {"id": "lvl3", "spec": "ckpt:lvl3.pt", "mu": 191.74, "games": 50},
        ],
        "rungs": [
            {"id": "lvl1", "mu": 54.85, "sigma": 6.04},
            {"id": "lvl3", "mu": 191.74, "sigma": 6.48},
        ],
    }
    merged = merge_manifest(
        document,
        levels={"random": 0.0, "lvl1": 7.67, "lvl3": 99.0},
        subjects=[
            {"id": "lvl1", "spec": "ckpt:lvl1.pt", "mu": 7.67, "games": 300},
            {"id": "lvl3", "spec": "random", "mu": 99.0, "games": 300},
        ],
        rungs=[{"id": "lvl1", "mu": 7.67, "sigma": 3.0}],
    )
    assert merged["levels"]["lvl1"] == 54.85  # frozen
    subjects = {entry["id"]: entry for entry in merged["subjects"]}
    # Published level: the whole published record is kept, including the
    # sigma/games measurement provenance (T17: a normal merge must not refresh
    # it, or the published rung uncertainty silently changes under frozen
    # levels).
    assert subjects["lvl1"] == {
        "id": "lvl1",
        "spec": "ckpt:lvl1.pt",
        "mu": 54.85,
        "games": 100,
    }
    # Different spec behind a published id: the published record is kept whole.
    assert subjects["lvl3"] == {
        "id": "lvl3",
        "spec": "ckpt:lvl3.pt",
        "mu": 191.74,
        "games": 50,
    }
    # The fresh fit cannot shrink or repoint the published rung selection.
    assert merged["rungs"] == [
        {"id": "lvl1", "mu": 54.85, "sigma": 6.04},
        {"id": "lvl3", "mu": 191.74, "sigma": 6.48},
    ]


def test_merge_without_refit_appends_new_rungs_from_frozen_levels():
    document = {
        "levels": {"lvl1": 54.85},
        "subjects": [{"id": "lvl1", "spec": "ckpt:lvl1.pt", "mu": 54.85}],
        "rungs": [{"id": "lvl1", "mu": 54.85, "sigma": 6.04}],
    }
    merged = merge_manifest(
        document,
        levels={"lvl1": 7.0, "lvl2": 120.22},
        subjects=[
            {"id": "lvl1", "spec": "ckpt:lvl1.pt", "mu": 7.0},
            {"id": "lvl2", "spec": "ckpt:lvl2.pt", "mu": 120.22},
        ],
        rungs=[
            {"id": "lvl1", "mu": 7.0, "sigma": 3.0},
            {"id": "lvl2", "mu": 120.22, "sigma": 5.0},
        ],
    )
    assert [rung["id"] for rung in merged["rungs"]] == ["lvl1", "lvl2"]
    # The published rung entry is untouched; the new one tracks its level.
    assert merged["rungs"][0] == {"id": "lvl1", "mu": 54.85, "sigma": 6.04}
    assert merged["rungs"][1] == {"id": "lvl2", "mu": 120.22, "sigma": 5.0}


def test_merge_without_refit_keeps_published_estimator_provenance():
    # A normal build_ladder merge (no --refit) must not clobber the published
    # probit-MLE estimator block with its own OpenSkill metadata (T17-F1).
    document = {
        "levels": {"lvl1": 82.75},
        "subjects": [
            {
                "id": "lvl1",
                "spec": "ckpt:lvl1.pt",
                "mu": 82.75,
                "sigma": 5.88,
                "games": 1600,
            }
        ],
        "estimator": {
            "kind": "probit-mle",
            "games": 4000,
            "source": "runs/t17_mle/games.jsonl",
        },
    }
    merged = merge_manifest(
        document,
        levels={"lvl1": 999.0},
        subjects=[
            {
                "id": "lvl1",
                "spec": "ckpt:lvl1.pt",
                "mu": 999.0,
                "sigma": 3.0,
                "games": 50,
            }
        ],
        estimator={"kind": "openskill-plackett-luce", "seed": 0},
    )
    assert merged["estimator"] == document["estimator"]
    assert merged["subjects"] == document["subjects"]
    assert merged["levels"] == {"lvl1": 82.75}


def test_merge_without_refit_stamps_estimator_when_absent():
    # A fresh study has no published provenance yet, so the incoming fit is
    # still stamped; only a published block is protected.
    merged = merge_manifest(
        {},
        levels={"lvl1": 1.0},
        subjects=[{"id": "lvl1", "mu": 1.0}],
        estimator={"kind": "openskill-plackett-luce", "seed": 0},
    )
    assert merged["estimator"] == {"kind": "openskill-plackett-luce", "seed": 0}


def test_merge_without_refit_takes_a_new_subject_whole():
    # Only ids already in the levels contract are frozen; a new id receives the
    # incoming sigma/games so a fresh measurement is not silently dropped.
    document = {
        "levels": {"lvl1": 82.75},
        "subjects": [
            {"id": "lvl1", "spec": "ckpt:a.pt", "mu": 82.75, "sigma": 5.9, "games": 100}
        ],
    }
    merged = merge_manifest(
        document,
        levels={"lvl2": 107.0},
        subjects=[
            {"id": "lvl2", "spec": "ckpt:b.pt", "mu": 107.0, "sigma": 6.0, "games": 200}
        ],
    )
    subjects = {entry["id"]: entry for entry in merged["subjects"]}
    assert subjects["lvl2"] == {
        "id": "lvl2",
        "spec": "ckpt:b.pt",
        "mu": 107.0,
        "sigma": 6.0,
        "games": 200,
    }
    assert subjects["lvl1"]["games"] == 100


def test_merge_refit_repoints_published_subject_and_replaces_rungs():
    document = {
        "levels": {"lvl1": 54.85},
        "subjects": [{"id": "lvl1", "spec": "ckpt:old.pt", "mu": 54.85}],
        "rungs": [{"id": "lvl1", "mu": 54.85, "sigma": 6.04}],
    }
    merged = merge_manifest(
        document,
        levels={"lvl1": 7.67},
        subjects=[{"id": "lvl1", "spec": "ckpt:new.pt", "mu": 7.67}],
        rungs=[{"id": "lvl1", "mu": 7.67, "sigma": 3.0}],
        refit=True,
    )
    assert merged["levels"] == {"lvl1": 7.67}
    assert merged["subjects"][0]["spec"] == "ckpt:new.pt"
    assert merged["subjects"][0]["mu"] == 7.67
    assert merged["rungs"] == [{"id": "lvl1", "mu": 7.67, "sigma": 3.0}]


def test_merge_refit_preserves_a_published_search_config():
    """A refit never drops the pinned search identity block by omission."""
    config = {
        "trunc_ply": 5,
        "rollout_k": 32,
        "max_candidates": 6,
        "max_rollout_ply": 400,
        "rollout_opponent": "ckpt:runs/w2m_ctl/agent.pt",
        "aggregate": "mean",
    }
    document = {
        "levels": {"search_leafq": 260.03},
        "subjects": [
            {
                "id": "search_leafq",
                "spec": "rolloutt:runs/ei2_value_t5/t_leafq/critic.pt",
                "mu": 260.03,
                "sigma": 3.855,
                "search_config": config,
            }
        ],
    }
    merged = merge_manifest(
        document,
        levels={"search_leafq": 259.0},
        subjects=[
            {
                "id": "search_leafq",
                "spec": "rolloutt:runs/ei2_value_t5/t_leafq/critic.pt",
                "mu": 259.0,
                "sigma": 4.0,
            }
        ],
        refit=True,
    )
    subject = merged["subjects"][0]
    assert subject["mu"] == 259.0
    assert subject["search_config"] == config


def test_merge_search_configs_seeds_and_refreshes():
    config = {"trunc_ply": 5, "rollout_k": 32, "aggregate": "mean"}
    seeded = merge_manifest(
        {
            "levels": {"search_leafq": 260.03},
            "subjects": [{"id": "search_leafq", "mu": 260.03}],
        },
        levels={"search_leafq": 260.03},
        subjects=[{"id": "search_leafq", "mu": 260.03}],
        search_configs={"search_leafq": config},
        refit=True,
    )
    assert seeded["subjects"][0]["search_config"] == config

    refreshed = merge_manifest(
        seeded,
        levels={"search_leafq": 260.03},
        subjects=[{"id": "search_leafq", "mu": 260.03}],
        search_configs={"search_leafq": {**config, "rollout_k": 64}},
        refit=True,
    )
    assert refreshed["subjects"][0]["search_config"]["rollout_k"] == 64

    with pytest.raises(ValueError, match="unknown subject"):
        merge_manifest(
            seeded,
            levels={},
            subjects=[],
            search_configs={"ghost": config},
            refit=True,
        )
    with pytest.raises(ValueError, match="must be a mapping"):
        merge_manifest(
            seeded,
            levels={},
            subjects=[],
            search_configs={"search_leafq": [1, 2]},
            refit=True,
        )


def test_merge_rejects_non_finite_search_config_values():
    document = {
        "levels": {"search_leafq": 260.03},
        "subjects": [
            {
                "id": "search_leafq",
                "mu": 260.03,
                "search_config": {"trunc_ply": float("nan")},
            }
        ],
    }
    with pytest.raises(ValueError, match="finite"):
        merge_manifest(document, levels={}, subjects=[], refit=True)


def test_merge_rejects_subjects_without_ids():
    with pytest.raises(ValueError):
        merge_manifest({}, levels={}, subjects=[{"elo": 1500.0}])
    with pytest.raises(ValueError):
        merge_manifest(
            {"subjects": [{"elo": 1500.0}]}, levels={}, subjects=[]
        )


def test_saved_manifest_is_json_text(tmp_path):
    path = save_manifest(tmp_path / "manifest.json", {"levels": {"a": 1.5}})
    assert json.loads(path.read_text(encoding="utf-8")) == {"levels": {"a": 1.5}}


def test_merge_stamps_rules_identity_on_fresh_document():
    identity = rules_identity(DEFAULT_RULES)
    rid = rules_id(DEFAULT_RULES)
    merged = merge_manifest(
        {},
        levels={"random": 0.0},
        subjects=[{"id": "random", "mu": 0.0}],
        rules=identity,
        rules_id=rid,
    )
    assert merged["rules_id"] == rid
    assert merged["rules"] == identity
    assert merged["levels"] == {"random": 0.0}
    assert merged["subjects"] == [{"id": "random", "mu": 0.0}]


def test_merge_stamps_rules_id_without_a_rules_block_when_not_given():
    merged = merge_manifest(
        {"seed": 4, "levels": {}, "subjects": []},
        levels={},
        subjects=[],
        rules_id="abcd1234",
    )
    assert merged["rules_id"] == "abcd1234"
    assert "rules" not in merged
    assert merged["seed"] == 4


def test_merge_with_matching_rules_id_accepts_and_refreshes_rules_block():
    document = {
        "rules_id": "abcd1234",
        "rules": {"revision": 2, "fields": {"num_players": 2}},
        "levels": {"random": 0.0},
        "subjects": [{"id": "random", "mu": 0.0}],
    }
    refreshed = {"revision": 2, "fields": {"num_players": 4}, "note": "refreshed"}
    merged = merge_manifest(
        document,
        levels={"random": 5.0},
        subjects=[{"id": "random", "mu": 5.0}],
        rules=refreshed,
        rules_id="abcd1234",
    )
    assert merged["rules_id"] == "abcd1234"
    assert merged["rules"] == refreshed
    assert merged["levels"] == {"random": 0.0}  # still frozen


def test_merge_with_matching_rules_id_keeps_existing_rules_block():
    identity = {"revision": 2, "fields": {"num_players": 2}}
    document = {"rules_id": "abcd1234", "rules": identity}
    merged = merge_manifest(document, levels={}, subjects=[], rules_id="abcd1234")
    assert merged["rules"] == identity


def test_merge_refuses_mismatched_rules_id_and_leaves_document_unchanged():
    document = {
        "rules_id": "old00000",
        "rules": {"revision": 1},
        "levels": {"random": 0.0},
        "subjects": [{"id": "random", "mu": 0.0}],
    }
    snapshot = json.loads(json.dumps(document))
    with pytest.raises(ValueError, match="rules_id"):
        merge_manifest(
            document,
            levels={"random": 5.0},
            subjects=[{"id": "random", "mu": 5.0}],
            rules_id="new00000",
        )
    assert document == snapshot


def test_merge_refuses_legacy_document_with_measured_levels_or_subjects():
    with pytest.raises(ValueError, match="rules_id"):
        merge_manifest(
            {"levels": {"random": 0.0}}, levels={}, subjects=[], rules_id="abcd1234"
        )
    with pytest.raises(ValueError, match="rules_id"):
        merge_manifest(
            {"subjects": [{"id": "random", "mu": 0.0}]},
            levels={},
            subjects=[],
            rules_id="abcd1234",
        )


def test_merge_without_rules_id_keeps_current_behaviour():
    document = {
        "levels": {"lvl1": 1400.0},
        "subjects": [{"id": "lvl1", "mu": 1400.0}],
    }
    merged = merge_manifest(
        document,
        levels={"lvl1": 1450.0},
        subjects=[{"id": "lvl1", "mu": 1450.0}],
    )
    assert merged["levels"] == {"lvl1": 1400.0}  # frozen as before
    assert merged["subjects"][0]["mu"] == 1400.0
    assert "rules_id" not in merged
    assert "rules" not in merged


def test_merge_does_not_mutate_input_document():
    identity = rules_identity(DEFAULT_RULES)
    rid = rules_id(DEFAULT_RULES)
    document = {
        "rules_id": rid,
        "rules": identity,
        "levels": {"random": 0.0},
        "subjects": [{"id": "random", "mu": 0.0}],
    }
    snapshot = json.loads(json.dumps(document))
    merged = merge_manifest(
        document,
        levels={"random": 5.0},
        subjects=[{"id": "random", "mu": 5.0}],
        rules=identity,
        rules_id=rid,
    )
    assert document == snapshot
    assert merged is not document
    assert merged["levels"] == {"random": 0.0}


# -- resolution block schema (review §0 item 4 / §4.4) ------------------------

RESOLUTION_REQUIRED_KEYS = (
    "rules_id",
    "band",
    "target",
    "achieved",
    "contrasts",
    "transitivity",
    "ci_source",
)


def _resolution_block(**overrides) -> dict:
    block = {
        "rules_id": rules_id(DEFAULT_RULES),
        "design": "seat-twin CRN complete round-robin",
        "band": ["lvl1", "lvl2"],
        "target": {"delta_elo_logistic": 10.0, "power": 0.8, "alpha": 0.05},
        "achieved": {
            "lvl1-lvl2": {
                "se_elo": 3.6,
                "deals": 1614,
                "power_at_delta": 0.8,
                "action_power_at_delta": 0.5,
            }
        },
        "contrasts": [
            {
                "a": "lvl1",
                "b": "lvl2",
                "delta_mu_probit": 24.2,
                "win_prob": 0.567,
                "ci95_win_prob": [0.54, 0.59],
            }
        ],
        "transitivity": {"status": "not-computed", "max_abs_z": None},
        "ci_source": "deal-bootstrap(seed=0, n=20)",
    }
    block.update(overrides)
    return block


def test_merge_refit_saves_a_validated_resolution_block():
    identity = rules_identity(DEFAULT_RULES)
    rid = rules_id(DEFAULT_RULES)
    block = _resolution_block()
    merged = merge_manifest(
        {},
        levels={"lvl1": 1.0},
        subjects=[{"id": "lvl1", "mu": 1.0}],
        resolution=block,
        refit=True,
        rules=identity,
        rules_id=rid,
    )
    assert merged["resolution"] == block
    assert merged["resolution"] is not block  # defensive copy


def test_merge_resolution_requires_the_schema_keys():
    rid = rules_id(DEFAULT_RULES)
    for missing in RESOLUTION_REQUIRED_KEYS:
        block = _resolution_block()
        del block[missing]
        with pytest.raises(ValueError, match="resolution"):
            merge_manifest(
                {}, levels={}, subjects=[], resolution=block, refit=True, rules_id=rid
            )
    with pytest.raises(ValueError, match="resolution"):
        merge_manifest(
            {}, levels={}, subjects=[], resolution=["not", "a", "mapping"], refit=True
        )


def test_merge_resolution_rules_id_must_match_the_manifest():
    rid = rules_id(DEFAULT_RULES)
    with pytest.raises(ValueError, match="rules_id"):
        merge_manifest(
            {},
            levels={},
            subjects=[],
            resolution=_resolution_block(rules_id="deadbeef"),
            refit=True,
            rules_id=rid,
        )
    block = _resolution_block()
    block["rules_id"] = None
    with pytest.raises(ValueError, match="rules_id"):
        merge_manifest({}, levels={}, subjects=[], resolution=block, refit=True)


def test_merge_resolution_rejects_non_finite_values():
    rid = rules_id(DEFAULT_RULES)
    for path, mutate in (
        (("achieved", "lvl1-lvl2", "se_elo"), math.nan),
        (("target", "power"), math.inf),
        (("contrasts", 0, "win_prob"), float("nan")),
    ):
        block = _resolution_block()
        target = block
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = mutate
        with pytest.raises(ValueError, match="finite"):
            merge_manifest(
                {}, levels={}, subjects=[], resolution=block, refit=True, rules_id=rid
            )


def test_merge_resolution_is_frozen_without_refit_unless_absent():
    rid = rules_id(DEFAULT_RULES)
    published = _resolution_block(ci_source="published")
    incoming = _resolution_block(ci_source="incoming")
    document = {
        "rules_id": rid,
        "levels": {},
        "subjects": [],
        "resolution": published,
    }
    merged = merge_manifest(document, levels={}, subjects=[], resolution=incoming)
    assert merged["resolution"] == published

    # A fresh study has no published resolution yet, so the validated incoming
    # block is stamped (mirrors the estimator provenance rule).
    fresh = merge_manifest(
        {}, levels={}, subjects=[], resolution=incoming, rules_id=rid
    )
    assert fresh["resolution"] == incoming


def test_merge_refit_replaces_the_published_resolution():
    rid = rules_id(DEFAULT_RULES)
    published = _resolution_block(ci_source="published")
    incoming = _resolution_block(ci_source="incoming")
    document = {"rules_id": rid, "resolution": published}
    merged = merge_manifest(
        document, levels={}, subjects=[], resolution=incoming, refit=True, rules_id=rid
    )
    assert merged["resolution"] == incoming


def test_merge_resolution_preserves_unknown_keys_explicitly():
    # Unknown keys are forward-compatible: they survive validation, and this
    # test is the explicit contract (rather than an accidental passthrough).
    block = _resolution_block()
    block["future_field"] = {"note": "kept", "big": 10**400}
    merged = merge_manifest(
        {},
        levels={},
        subjects=[],
        resolution=block,
        rules_id=rules_id(DEFAULT_RULES),
    )
    assert merged["resolution"]["future_field"] == {"note": "kept", "big": 10**400}


def test_merge_without_resolution_keeps_the_published_block():
    rid = rules_id(DEFAULT_RULES)
    published = _resolution_block()
    document = {"rules_id": rid, "resolution": published}
    merged = merge_manifest(document, levels={}, subjects=[])
    assert merged["resolution"] == published

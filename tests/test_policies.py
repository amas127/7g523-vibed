"""The shared policy-spec grammar: one owner for id/spec parsing and checks.

``random`` / ``ckpt:<path>`` is parsed here, not in each tool; the
error strings are the CLI contracts the tools turn into ``SystemExit``.
"""
from __future__ import annotations

from seven523 import policies
from seven523.policies import (
    default_id_for_spec,
    missing_ckpt_path,
    split_entrant,
    validate_spec,
)

# -- default_id_for_spec -----------------------------------------------------


def test_default_id_for_scripted_specs_is_the_spec():
    assert default_id_for_spec("random") == "random"
    assert default_id_for_spec("other") == "other"


def test_default_id_for_ckpt_uses_the_parent_directory(tmp_path):
    path = tmp_path / "run_a" / "agent.pt"
    assert default_id_for_spec(f"ckpt:{path}") == "run_a"


def test_default_id_for_ckpt_falls_back_to_stem_then_literal():
    # No directory component: the parent name is empty, so the stem wins.
    assert default_id_for_spec("ckpt:agent.pt") == "agent"
    assert default_id_for_spec("ckpt:") == "ckpt"


# -- split_entrant -----------------------------------------------------------


def test_split_entrant_explicit_id_wins_and_strips():
    assert split_entrant("g1=ckpt:runs/agent.pt") == ("g1", "ckpt:runs/agent.pt")
    assert split_entrant("  g1 = random  ") == ("g1", "random")


def test_split_entrant_derives_an_id_for_bare_specs():
    assert split_entrant("random") == ("random", "random")
    assert split_entrant(" other ") == ("other", "other")
    assert split_entrant("ckpt:runs/run_b/agent.pt") == (
        "run_b",
        "ckpt:runs/run_b/agent.pt",
    )
    assert split_entrant("ckpt:") == ("ckpt", "ckpt:")


def test_split_entrant_only_splits_at_the_first_equals():
    assert split_entrant("id=a=b") == ("id", "a=b")
    assert split_entrant("=random") == ("", "random")


# -- missing_ckpt_path -------------------------------------------------------


def test_missing_ckpt_path_ignores_non_ckpt_specs():
    assert missing_ckpt_path(None) is None
    assert missing_ckpt_path("random") is None
    assert missing_ckpt_path("other") is None


def test_missing_ckpt_path_returns_none_for_existing_files(tmp_path):
    live = tmp_path / "agent.pt"
    live.write_bytes(b"live")
    assert missing_ckpt_path(f"ckpt:{live}") is None


def test_missing_ckpt_path_returns_the_path_when_absent(tmp_path):
    gone = tmp_path / "gone.pt"
    assert missing_ckpt_path(f"ckpt:{gone}") == str(gone)
    # An empty path is missing too, but the empty string is not None.
    assert missing_ckpt_path("ckpt:") == ""
    # A directory is not a checkpoint file.
    assert missing_ckpt_path(f"ckpt:{tmp_path}") == str(tmp_path)


# -- validate_spec -----------------------------------------------------------


def test_validate_spec_accepts_scripted_and_live_ckpt(tmp_path):
    live = tmp_path / "agent.pt"
    live.write_bytes(b"live")
    assert validate_spec("random") is None
    assert validate_spec(f"ckpt:{live}") is None


def test_validate_spec_rejects_an_empty_ckpt_path():
    assert validate_spec("ckpt:") == "ckpt: spec needs a checkpoint path"


def test_validate_spec_reports_a_missing_checkpoint(tmp_path):
    gone = tmp_path / "gone.pt"
    assert validate_spec(f"ckpt:{gone}") == f"checkpoint not found: {gone}"


def test_validate_spec_reports_unknown_specs():
    assert validate_spec("banana") == (
        "unknown policy spec 'banana' (want random / ckpt:<path>)"
    )
    # The retired greedy spec is rejected like any other unknown spec.
    assert validate_spec("greedy") == (
        "unknown policy spec 'greedy' (want random / ckpt:<path>)"
    )


def test_spec_grammar_helpers_are_public():
    assert {
        "default_id_for_spec",
        "missing_ckpt_path",
        "split_entrant",
        "validate_spec",
    } <= set(policies.__all__)

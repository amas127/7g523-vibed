"""``seven523.web.plugin`` tests: discovery, precedence, error containment.

These are pure stdlib: no torch, no ``runs/`` import, no server.  The plugin
fixtures are written into ``tmp_path`` so the cwd default can be exercised
without touching the repository checkout.  The real run-local plugin (torch
free at import time) is loaded once against a temporary manifest to pin the
manifest-driven capability gate.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from seven523.rules import DEFAULT_RULES, rules_id
from seven523.web.plugin import (
    DEFAULT_PLUGIN_PATH,
    PLUGIN_ENV,
    PluginDiscovery,
    discover_plugin,
    load_web_plugin,
    resolve_plugin_path,
)

SEARCH_CONFIG = {
    "trunc_ply": 5,
    "rollout_k": 32,
    "max_candidates": 6,
    "max_rollout_ply": 400,
    "rollout_opponent": "ckpt:runs/w2m_ctl__11__1790516900/agent.pt",
    "aggregate": "mean",
}

GOOD_PLUGIN = '''
"""A minimal plugin used by the discovery tests."""


def build_plugin(*, device="cpu", manifest=None):
    return {
        "search": {"label": "test-search", "options": {}},
        "twin": {"label": "test-twin"},
        "warnings": ("fixture",),
    }
'''

LEGACY_PLUGIN = '''
"""A pre-manifest plugin whose hook takes device only (still loadable)."""


def build_plugin(*, device="cpu"):
    return {"search": {"label": "legacy-search", "options": {}}}
'''

ECHO_MANIFEST_PLUGIN = '''

def build_plugin(*, device="cpu", manifest=None):
    return {"warnings": (str(manifest),)}
'''

BROKEN_IMPORT_PLUGIN = """
raise RuntimeError("boom at import")
"""

NO_HOOK_PLUGIN = """
VALUE = 1
"""

UNKNOWN_KEY_PLUGIN = """
def build_plugin(*, device="cpu"):
    return {"nope": 1}
"""


def _write(tmp_path: Path, name: str, source: str) -> Path:
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return path


def _write_default(tmp_path: Path, source: str) -> Path:
    path = tmp_path / DEFAULT_PLUGIN_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


def test_discover_plugin_is_off_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv(PLUGIN_ENV, raising=False)
    discovery = discover_plugin(cwd=tmp_path)
    assert discovery == PluginDiscovery()
    assert discovery.plugin is None and discovery.source is None
    assert discovery.error is None and discovery.explicit is False
    assert discovery.path is None


def test_discover_plugin_loads_env_path(tmp_path, monkeypatch):
    path = _write(tmp_path, "my_plugin.py", GOOD_PLUGIN)
    monkeypatch.setenv(PLUGIN_ENV, str(path))
    discovery = discover_plugin(cwd=tmp_path)
    assert discovery.error is None
    assert discovery.explicit is True
    assert discovery.source == str(path)
    assert discovery.path == str(path)
    plugin = discovery.plugin
    assert plugin is not None
    assert plugin.search["label"] == "test-search"
    assert plugin.twin["label"] == "test-twin"
    assert plugin.warnings == ("fixture",)


def test_discover_plugin_forwards_the_manifest_path(tmp_path):
    """The hook receives ``--manifest`` so it can read the published config."""
    path = _write(tmp_path, "echo_plugin.py", ECHO_MANIFEST_PLUGIN)
    discovery = discover_plugin(plugin=path, manifest="/data/m.json")
    assert discovery.error is None
    assert discovery.plugin is not None
    assert discovery.plugin.warnings == ("/data/m.json",)
    # No manifest passed -> the hook still sees the explicit ``None``.
    discovery = discover_plugin(plugin=path)
    assert discovery.plugin is not None
    assert discovery.plugin.warnings == ("None",)


def test_legacy_plugin_hook_without_manifest_still_loads(tmp_path):
    path = _write(tmp_path, "legacy_plugin.py", LEGACY_PLUGIN)
    plugin = load_web_plugin(path, manifest="/data/m.json")
    assert plugin.search == {"label": "legacy-search", "options": {}}


def test_discover_plugin_env_off_disables(tmp_path, monkeypatch):
    _write_default(tmp_path, GOOD_PLUGIN)
    for value in ("", "0", "off", "none", "false", "no", "OFF", "None"):
        monkeypatch.setenv(PLUGIN_ENV, value)
        discovery = discover_plugin(cwd=tmp_path)
        assert discovery.plugin is None, value
        assert discovery.source is None, value
        assert discovery.error is None, value


def test_discover_plugin_contains_errors(tmp_path):
    cases = (
        (BROKEN_IMPORT_PLUGIN, "boom at import"),
        (NO_HOOK_PLUGIN, "build_plugin"),
        (UNKNOWN_KEY_PLUGIN, "未知插件键"),
    )
    for source, needle in cases:
        path = _write(tmp_path, "broken_plugin.py", source)
        discovery = discover_plugin(plugin=path, env="")
        assert discovery.plugin is None
        assert discovery.explicit is True
        assert discovery.error is not None and needle in discovery.error


def test_cli_explicit_plugin_failure_exits_1(tmp_path, capsys):
    """A user-named plugin that fails to load is fatal (no silent fallback)."""
    from seven523.web.cli import main

    path = _write(tmp_path, "broken.py", BROKEN_IMPORT_PLUGIN)
    code = main(["--plugin", str(path)])
    assert code == 1
    err = capsys.readouterr().err
    assert "插件加载失败" in err and "boom at import" in err


def test_unknown_plugin_capability_keys_are_refused(tmp_path):
    path = _write(
        tmp_path,
        "old_keys.py",
        '''\ndef build_plugin(*, device="cpu"):\n    return {"rated_rungs": {}, "unrated": {}}\n''',
    )
    discovery = discover_plugin(plugin=path, env="")
    assert discovery.plugin is None
    assert discovery.error is not None and "未知插件键" in discovery.error


def test_real_plugin_builds_only_manifest_configured_search_rungs(tmp_path):
    """The run-local plugin reads ``search_config`` from the given manifest.

    A ``rolloutt:`` subject with a valid block is admitted and its twin
    metadata is derived from the same block; a ``rolloutt:`` subject without
    one is skipped with a warning.  No artifact is consulted.
    """
    plugin_path = (
        Path(__file__).resolve().parents[1]
        / "runs"
        / "o4lite-search"
        / "web_plugin.py"
    )
    if not plugin_path.is_file():
        pytest.skip("run-local web plugin is not present")
    configured_spec = "rolloutt:runs/ei2_value_t5/t_leafq/critic.pt"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "rules_id": rules_id(DEFAULT_RULES),
                "levels": {"random": 0.0, "search_leafq": 260.03},
                "anchors": [{"id": "random", "mu": 0.0}],
                "subjects": [
                    {"id": "random", "spec": "random", "mu": 0.0, "sigma": 0.0},
                    {
                        "id": "search_leafq",
                        "spec": configured_spec,
                        "mu": 260.03,
                        "sigma": 3.855,
                        "search_config": dict(SEARCH_CONFIG),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    plugin = load_web_plugin(plugin_path, manifest=str(manifest))
    assert plugin.warnings == ()
    assert plugin.can_build is not None
    assert plugin.can_build("random") is True
    assert plugin.can_build(configured_spec) is True
    assert plugin.can_build("rolloutt:runs/other/critic.pt") is False
    assert plugin.twin is not None
    assert plugin.twin["search"]["identity"] == configured_spec
    params = plugin.twin["search"]["params"]
    assert params["trunc_ply"] == SEARCH_CONFIG["trunc_ply"]
    assert params["rollout_opponent"] == SEARCH_CONFIG["rollout_opponent"]
    assert params["value_ckpt"] == "runs/ei2_value_t5/t_leafq/critic.pt"
    factory = plugin.policy_factory
    # ``validate_search_config`` normalises the optional outcome blend and
    # the optional exact-endgame flag in.
    assert factory.search_configs[configured_spec] == {
        **SEARCH_CONFIG,
        "outcome_blend": 0.0,
        "endgame": 0,
    }


def test_real_plugin_warns_on_a_malformed_search_config(tmp_path):
    plugin_path = (
        Path(__file__).resolve().parents[1]
        / "runs"
        / "o4lite-search"
        / "web_plugin.py"
    )
    if not plugin_path.is_file():
        pytest.skip("run-local web plugin is not present")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "rules_id": rules_id(DEFAULT_RULES),
                "levels": {"random": 0.0, "search_leafq": 260.03},
                "anchors": [{"id": "random", "mu": 0.0}],
                "subjects": [
                    {"id": "random", "spec": "random", "mu": 0.0, "sigma": 0.0},
                    {
                        "id": "search_leafq",
                        "spec": "rolloutt:runs/x/critic.pt",
                        "mu": 260.03,
                        "sigma": 3.855,
                        "search_config": {"trunc_ply": 5},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    plugin = load_web_plugin(plugin_path, manifest=str(manifest))
    assert plugin.can_build("rolloutt:runs/x/critic.pt") is False
    assert any("缺少键" in warning for warning in plugin.warnings)
    assert plugin.twin is None


def test_resolve_plugin_path_precedence(tmp_path, monkeypatch):
    monkeypatch.delenv(PLUGIN_ENV, raising=False)
    default = _write_default(tmp_path, GOOD_PLUGIN)
    env_path = tmp_path / "env_plugin.py"
    cli_path = tmp_path / "cli_plugin.py"

    resolved, explicit = resolve_plugin_path(cwd=tmp_path)
    assert resolved == default and explicit is False

    resolved, explicit = resolve_plugin_path(env=str(env_path), cwd=tmp_path)
    assert resolved == env_path and explicit is True

    resolved, explicit = resolve_plugin_path(
        plugin=str(cli_path), env=str(env_path), cwd=tmp_path
    )
    assert resolved == cli_path and explicit is True

    resolved, explicit = resolve_plugin_path(
        plugin=str(cli_path), disabled=True, env=str(env_path), cwd=tmp_path
    )
    assert resolved is None and explicit is False

    # A missing default candidate is silent raw-only.
    resolved, explicit = resolve_plugin_path(cwd=tmp_path / "empty")
    assert resolved is None and explicit is False

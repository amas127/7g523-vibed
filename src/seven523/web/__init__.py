"""``7g523-web``: the browser card table over the real engine and rating pipeline.

One local HTTP server, one page, two ways to play:

* **自由对战** — pick any pool rung (the ``random`` anchor, the ladder, a
  top-platform arm, or a search rung) and a seat, then play a real game; every
  game is a replayable trace under ``traces/web/<run-id>/``.  With an injected
  search plugin, ``ckpt:`` opponents take per-game t/K and the base model is
  the chosen opponent.
* **定级模式** — a full :class:`~seven523.placement.PlacementSession` driven one
  game at a time over the manifest's ``levels``: raw rungs and search rungs in
  one joint-fit pool, capability-gated by the injected factory, with the same
  ``traces/sessions/<id>/`` artifacts as the ``7g523-elo`` CLI.

The stock CLI discovers one optional run-local plugin
(:mod:`.plugin`, ``runs/o4lite-search/web_plugin.py`` by default) when all
three injection seams are ``None``; the run-local launchers still inject
``policy_factory`` / ``search`` / ``twin`` explicitly.

``__init__`` re-exports the public API; the code lives in :mod:`.view`
(JSON projection), :mod:`.table` (session state), :mod:`.server` (HTTP),
:mod:`.plugin` (optional discovery) and :mod:`.cli` (entry point).
Human-play docs: ``docs/human-play.md``.
"""
from __future__ import annotations

from .cli import build_parser as build_parser, main as main
from .plugin import (
    PluginDiscovery as PluginDiscovery,
    PluginError as PluginError,
    WebPlugin as WebPlugin,
    discover_plugin as discover_plugin,
    load_web_plugin as load_web_plugin,
)
from .server import make_handler as make_handler, serve as serve
from .table import (
    PROTO_VERSION as PROTO_VERSION,
    TableSession,
    WebConfig,
    WebError,
    torch_available,
)
from .view import (
    action_json as action_json,
    card_json as card_json,
    combo_json as combo_json,
    counter_json as counter_json,
    hand_key as hand_key,
    legal_actions as legal_actions,
)

__all__ = [
    "PROTO_VERSION",
    "PluginDiscovery",
    "PluginError",
    "TableSession",
    "WebConfig",
    "WebError",
    "WebPlugin",
    "action_json",
    "build_parser",
    "card_json",
    "combo_json",
    "counter_json",
    "discover_plugin",
    "hand_key",
    "legal_actions",
    "load_web_plugin",
    "main",
    "make_handler",
    "serve",
    "torch_available",
]

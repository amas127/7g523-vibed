"""Optional plugin discovery for the stock ``7g523-web`` entry point.

The package itself owns no search wrapper, no ``rolloutt:`` spec and no torch
import (ADR-0010 §2): a run-local plugin file exports one module-level hook and
the stock CLI loads it *by fixed path* when the three injection seams are
``None``.  No directory scan, no entry-point discovery, no ``import runs.*``.

Discovery order (:func:`resolve_plugin_path`):

1. ``--no-plugin`` / ``disabled=True`` → off.
2. ``--plugin PATH`` → explicit; a load failure is fatal to the caller.
3. ``SEVEN523_WEB_PLUGIN`` — ``""`` / ``0`` / ``off`` / ``none`` / ``false`` /
   ``no`` (case-insensitive) disables; any other value is an explicit path.
4. ``<cwd>/runs/o4lite-search/web_plugin.py`` when it exists; a missing file is
   silent raw-only, a broken file degrades to raw-only with a warning.

The plugin contract is one hook::

    def build_plugin(*, device: str = "cpu", manifest: str | None = None) \
            -> WebPlugin | Mapping[str, Any]

returning any of ``policy_factory`` / ``search`` / ``twin`` / ``can_build`` /
``warnings``.  ``manifest`` is the server's ``--manifest`` path so a plugin
can rebuild search policies from the manifest's published ``search_config``
instead of a second artifact; a plugin whose hook does not accept it is still
called with ``device`` only (forward compatibility).  Unknown keys are a load
failure so a typo cannot silently disable a capability.  Every import,
hook-call, coerce and missing-file error is folded into
:attr:`PluginDiscovery.error`; the stock table never raises because of a
plugin.
"""
from __future__ import annotations

import importlib.util
import inspect
import os
import sys
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "DEFAULT_PLUGIN_PATH",
    "DISABLED_VALUES",
    "PLUGIN_ENV",
    "PLUGIN_HOOK",
    "PluginDiscovery",
    "PluginError",
    "WebPlugin",
    "coerce_plugin",
    "discover_plugin",
    "load_web_plugin",
    "resolve_plugin_path",
]

#: Environment variable that points at a plugin file (or disables discovery).
PLUGIN_ENV = "SEVEN523_WEB_PLUGIN"
#: The cwd-relative candidate checked when nothing explicit is given.
DEFAULT_PLUGIN_PATH = Path("runs") / "o4lite-search" / "web_plugin.py"
#: The single module-level hook a plugin must export.
PLUGIN_HOOK = "build_plugin"
#: Environment values that mean "no plugin" (compared case-insensitively).
DISABLED_VALUES = frozenset({"", "0", "off", "none", "false", "no"})

#: Keys a plugin mapping may carry; anything else is a load failure.
_PLUGIN_KEYS = frozenset(
    {
        "policy_factory",
        "search",
        "twin",
        "can_build",
        "warnings",
    }
)


class PluginError(Exception):
    """A plugin could not be loaded, called or coerced (never leaks to stock)."""


@dataclass(frozen=True, slots=True)
class WebPlugin:
    """The capabilities one plugin contributes to a :class:`WebConfig`."""

    policy_factory: Callable[[str, Any, int, Mapping[str, Any]], Any] | None = None
    search: Mapping[str, Any] | None = None
    twin: Mapping[str, Any] | None = None
    can_build: Callable[[str | None], bool] | None = None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PluginDiscovery:
    """The outcome of one discovery attempt; ``error`` is always non-fatal."""

    plugin: WebPlugin | None = None
    source: str | None = None
    error: str | None = None
    explicit: bool = False
    path: str | None = None


def resolve_plugin_path(
    plugin: str | Path | None = None,
    *,
    disabled: bool = False,
    env: str | None = None,
    cwd: str | Path | None = None,
) -> tuple[Path | None, bool]:
    """Resolve the plugin candidate to ``(path, explicit)``.

    ``disabled`` wins over everything; then an explicit ``plugin`` path; then
    ``env`` (``None`` reads :data:`PLUGIN_ENV` from the process environment);
    then the cwd-relative :data:`DEFAULT_PLUGIN_PATH` when it exists.  A
    disabled or missing candidate returns ``(None, False)``.
    """
    if disabled:
        return None, False
    if plugin is not None and str(plugin).strip():
        return Path(str(plugin)).expanduser(), True
    if env is None:
        env = os.environ.get(PLUGIN_ENV)
    if env is not None:
        text = str(env).strip()
        if text.lower() in DISABLED_VALUES:
            return None, False
        return Path(text).expanduser(), True
    base = Path(cwd) if cwd is not None else Path.cwd()
    candidate = base / DEFAULT_PLUGIN_PATH
    if candidate.is_file():
        return candidate, False
    return None, False


def coerce_plugin(obj: Any, source: str) -> WebPlugin:
    """Normalise a plugin hook result into a :class:`WebPlugin`.

    Accepts a ready :class:`WebPlugin` or a mapping; unknown keys, non-callable
    factories and non-mapping capability documents raise :class:`PluginError`.
    """
    if isinstance(obj, WebPlugin):
        return obj
    if not isinstance(obj, Mapping):
        raise PluginError(
            f"{source}: {PLUGIN_HOOK} 必须返回 WebPlugin 或 mapping，"
            f"得到 {type(obj).__name__}"
        )
    unknown = set(obj) - _PLUGIN_KEYS
    if unknown:
        raise PluginError(f"{source}: 未知插件键 {sorted(unknown)}")

    def optional_callable(key: str) -> Any:
        value = obj.get(key)
        if value is not None and not callable(value):
            raise PluginError(f"{source}: 插件键 {key!r} 需要可调用对象")
        return value

    def optional_mapping(key: str) -> Mapping[str, Any] | None:
        value = obj.get(key)
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise PluginError(f"{source}: 插件键 {key!r} 需要 mapping")
        return value

    raw_warnings = obj.get("warnings") or ()
    if isinstance(raw_warnings, str):
        raw_warnings = (raw_warnings,)
    try:
        warnings = tuple(str(warning) for warning in raw_warnings)
    except TypeError as exc:
        raise PluginError(f"{source}: warnings 需要字符串序列") from exc

    return WebPlugin(
        policy_factory=optional_callable("policy_factory"),
        search=optional_mapping("search"),
        twin=optional_mapping("twin"),
        can_build=optional_callable("can_build"),
        warnings=warnings,
    )


def _hook_accepts_manifest(hook: Callable[..., Any]) -> bool:
    """Whether the hook can take the optional ``manifest`` keyword.

    A hook declared with ``**kwargs`` or an explicit ``manifest`` parameter
    gets it; an older ``build_plugin(*, device=...)`` is called unchanged.
    """
    try:
        parameters = inspect.signature(hook).parameters
    except (TypeError, ValueError):  # pragma: no cover - exotic callables
        return False
    if "manifest" in parameters:
        return True
    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


def load_web_plugin(
    path: str | Path,
    *,
    device: str = "cpu",
    manifest: str | Path | None = None,
) -> WebPlugin:
    """Import a plugin file by path and call its ``build_plugin`` hook.

    The module is registered under a unique temporary name (so dataclasses and
    relative imports inside the plugin work) and its directory is prepended to
    ``sys.path`` (deduplicated), which lets the plugin ``import web_search``
    from the same run-local folder.  Every failure — missing file, import
    error, missing/non-callable hook, hook exception, malformed return — is a
    :class:`PluginError`.  A hook that declares a ``manifest`` parameter (or
    ``**kwargs``) receives the server's manifest path so it can rebuild the
    published search identities; older hooks are called with ``device`` only.
    """
    path = Path(path).expanduser()
    if not path.is_file():
        raise PluginError(f"插件文件不存在：{path}")
    directory = str(path.parent.resolve())
    if directory not in sys.path:
        sys.path.insert(0, directory)
    module_name = f"_seven523_web_plugin_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise PluginError(f"无法按路径加载插件：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # any plugin import failure is a PluginError
        sys.modules.pop(module_name, None)
        raise PluginError(
            f"插件导入失败：{type(exc).__name__}: {exc}"
        ) from exc
    hook = getattr(module, PLUGIN_HOOK, None)
    if not callable(hook):
        raise PluginError(f"插件缺少模块级 {PLUGIN_HOOK}(*, device) 可调用 hook")
    try:
        if _hook_accepts_manifest(hook):
            obj = hook(device=device, manifest=str(manifest) if manifest else None)
        else:
            obj = hook(device=device)
    except Exception as exc:  # the hook's failures are ours to report
        raise PluginError(
            f"{PLUGIN_HOOK} 调用失败：{type(exc).__name__}: {exc}"
        ) from exc
    return coerce_plugin(obj, source=str(path))


def discover_plugin(
    plugin: str | Path | None = None,
    *,
    disabled: bool = False,
    device: str = "cpu",
    manifest: str | Path | None = None,
    env: str | None = None,
    cwd: str | Path | None = None,
) -> PluginDiscovery:
    """Resolve and load a plugin; never raises, always returns a discovery.

    A missing candidate yields ``PluginDiscovery(None, None, None, False)``;
    an explicit path that fails keeps ``explicit=True`` so the caller can exit
    instead of silently degrading.  ``manifest`` is forwarded to the hook so
    the plugin reads the same published ``search_config`` the server uses.
    """
    path, explicit = resolve_plugin_path(
        plugin, disabled=disabled, env=env, cwd=cwd
    )
    if path is None:
        return PluginDiscovery()
    try:
        loaded = load_web_plugin(path, device=device, manifest=manifest)
    except PluginError as exc:
        return PluginDiscovery(
            plugin=None,
            source=str(path),
            error=str(exc),
            explicit=explicit,
            path=str(path),
        )
    return PluginDiscovery(
        plugin=loaded,
        source=str(path),
        error=None,
        explicit=explicit,
        path=str(path),
    )

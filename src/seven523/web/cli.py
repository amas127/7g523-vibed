"""The ``7g523-web`` CLI: argument parsing and the server entry point.

The browser table runs the same pool, prior and artifact layout as the
``7g523-elo`` CLI, so a placement session started here is the same session the
terminal tool would write: ``traces/sessions/<id>/``.  Free-play traces go to
``traces/web/<run-id>/`` and replay with ``7g523-play --replay``.
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .plugin import discover_plugin
from .server import serve
from .table import TableSession, WebConfig, torch_available

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="7g523-web",
        description=(
            "浏览器牌桌：真引擎 + 真定级管线（自适应选档 / 轨迹先验 / OpenSkill），"
            "定级模式产物与 7g523-elo 一致（docs/human-play.md）"
        ),
    )
    parser.add_argument(
        "--manifest",
        default="traces/pool10/manifest.json",
        help="对手 manifest（默认 traces/pool10/manifest.json）",
    )
    parser.add_argument(
        "--prior",
        default="artifacts/human-elo/prior.json",
        help="轨迹先验产物（默认 artifacts/human-elo/prior.json）",
    )
    parser.add_argument(
        "--no-trace-prior",
        action="store_true",
        help="只跑结果似然（冷启动；不需要先验产物）",
    )
    parser.add_argument(
        "--sessions-dir",
        default="traces/sessions",
        help="定级会话目录的父目录（默认 traces/sessions）",
    )
    parser.add_argument(
        "--twins-dir",
        default="traces/twins",
        help="deal-twin 会话目录的父目录（默认 traces/twins）",
    )
    parser.add_argument(
        "--trace-dir",
        default="traces/web",
        help="自由对战 trace 的父目录（默认 traces/web）",
    )
    parser.add_argument("--host", default="127.0.0.1", help="监听地址（默认本机）")
    parser.add_argument("--port", type=int, default=8765, help="端口（默认 8765）")
    parser.add_argument("--device", default="cpu", help="ckpt 对手的设备（默认 cpu）")
    parser.add_argument("--seed", type=int, default=None, help="发牌随机种子")
    parser.add_argument(
        "--plugin",
        default=None,
        metavar="PATH",
        help=(
            "按路径加载 run-local web 插件（默认探测 "
            "runs/o4lite-search/web_plugin.py；可被 SEVEN523_WEB_PLUGIN 覆盖）"
        ),
    )
    parser.add_argument(
        "--no-plugin",
        action="store_true",
        help="关闭插件发现，只用 stock raw 池/twin（搜索 rung 会跳过并警告）",
    )
    parser.add_argument("--open", action="store_true", help="启动后自动打开浏览器")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    policy_factory: Callable[[str, Any, int, Mapping[str, Any]], Any] | None = None,
    search: Mapping[str, Any] | None = None,
    twin: Mapping[str, Any] | None = None,
) -> int:
    """Run ``7g523-web``; ``policy_factory`` / ``search`` / ``twin`` are seams.

    The run-local search launcher (:file:`runs/o4lite-search/web_search.py`)
    injects its wrapper factory and display metadata here; the twin launcher
    (:file:`runs/o4lite-search/web_twin.py`) additionally injects the pinned
    deal-twin metadata.  All three are inert for the stock CLI (``None``).

    When all three seams are ``None`` the stock CLI attempts one optional
    plugin discovery (``--plugin`` > ``SEVEN523_WEB_PLUGIN`` > the cwd default
    ``runs/o4lite-search/web_plugin.py``): an explicitly named plugin that
    fails exits 1, a broken default candidate degrades to raw-only with a
    warning, and explicit injection (both launchers) skips discovery entirely.
    """
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    plugin_block: Mapping[str, Any] | None = None
    can_build_spec: Callable[[str | None], bool] | None = None
    if policy_factory is None and search is None and twin is None:
        discovery = discover_plugin(
            plugin=args.plugin,
            disabled=args.no_plugin,
            device=args.device,
            manifest=args.manifest,
        )
        if discovery.error is not None:
            if discovery.explicit:
                print(
                    f"7g523-web: 插件加载失败（{discovery.path}）：{discovery.error}",
                    file=sys.stderr,
                )
                return 1
            print(
                f"7g523-web: warning: 插件 {discovery.path} 不可用，回退 raw-only："
                f"{discovery.error}",
                file=sys.stderr,
            )
            plugin_block = {
                "source": discovery.path,
                "error": discovery.error,
                "warnings": [],
            }
        elif discovery.plugin is not None:
            loaded = discovery.plugin
            policy_factory = loaded.policy_factory
            search = loaded.search
            twin = loaded.twin
            can_build_spec = loaded.can_build
            plugin_block = {
                "source": discovery.source,
                "error": None,
                "warnings": list(loaded.warnings),
            }
            print(f"[7g523-web] 插件: {discovery.source}")
            print(
                "[7g523-web] 插件能力:",
                ", ".join(
                    key
                    for key, value in (
                        ("free-search", search is not None),
                        ("twin", twin is not None),
                        ("search-rungs", can_build_spec is not None),
                    )
                    if value
                )
                or "none",
            )
            for warning in loaded.warnings:
                print(f"[7g523-web] plugin warning: {warning}", file=sys.stderr)
    web_config = WebConfig(
        manifest=Path(args.manifest),
        prior=None if args.no_trace_prior else Path(args.prior),
        device=args.device,
        sessions_dir=Path(args.sessions_dir),
        twins_dir=Path(args.twins_dir),
        free_traces_dir=Path(args.trace_dir),
        seed=args.seed,
        policy_factory=policy_factory,
        search=search,
        twin=twin,
        can_build_spec=can_build_spec,
        plugin=plugin_block,
    )
    try:
        table = TableSession.from_config(web_config)
    except (FileNotFoundError, ValueError, KeyError, OSError) as exc:
        print(f"7g523-web: {exc}", file=sys.stderr)
        if not torch_available():
            print(
                "7g523-web: hint: ckpt 对手需要 torch，用 "
                "uv run --group train 7g523-web …",
                file=sys.stderr,
            )
        return 1
    try:
        serve(table, host=args.host, port=args.port, open_browser=args.open)
    except OSError as exc:
        print(f"7g523-web: 无法监听 {args.host}:{args.port}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""The local HTTP layer: a stdlib server over one :class:`TableSession`.

Routes (all JSON unless noted)::

    GET  /              the single page (static/index.html)
    GET  /app.css       the stylesheet
    GET  /app.js        the page script
    GET  /api/config    pool, prior, capabilities
    GET  /api/state     the current snapshot
    POST /api/start     {mode: "free"|"placement", ...}
    POST /api/action    {action_id, suit}
    POST /api/continue  next placement/twin game after a round-over screen
    POST /api/quit      abandon the live game / close the session

``ThreadingHTTPServer`` serves each request on its own thread; the session
serialises mutations itself, so the handler stays thin.  A failed request never
kills the server: :class:`WebError` becomes its HTTP status, anything else a
500 with a stderr traceback.
"""
from __future__ import annotations

import json
import sys
import threading
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from ..policies import buildable_by_grammar
from .table import PROTO_VERSION, TableSession, WebError, torch_available

__all__ = ["make_handler", "serve"]

STATIC = Path(__file__).resolve().parent / "static"
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
}


class Handler(BaseHTTPRequestHandler):
    """One HTTP request; the session is class-level (bound by :func:`serve`)."""

    server_version = f"7g523-web/{PROTO_VERSION}"
    table: TableSession  # bound by serve()

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write(f"[7g523-web] {self.address_string()} {fmt % args}\n")

    # -- replies -------------------------------------------------------------

    def send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path: Path) -> None:
        if not path.is_file():
            self.send_error(404)
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header(
            "Content-Type", CONTENT_TYPES.get(path.suffix, "application/octet-stream")
        )
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def read_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise WebError(f"请求不是合法 JSON：{exc}") from exc
        if not isinstance(data, dict):
            raise WebError("请求体需要是 JSON 对象")
        return data

    # -- routes --------------------------------------------------------------

    def do_GET(self) -> None:
        route = self.path.split("?", 1)[0]
        try:
            if route in ("/", "/index.html"):
                self.send_file(STATIC / "index.html")
            elif route == "/app.css":
                self.send_file(STATIC / "app.css")
            elif route == "/app.js":
                self.send_file(STATIC / "app.js")
            elif route == "/api/config":
                self.send_json(self.table.config_document())
            elif route == "/api/state":
                self.send_json(self.table.snapshot())
            else:
                self.send_json({"error": f"not found: {route}"}, status=404)
        except WebError as exc:
            self.send_json({"error": str(exc)}, status=exc.status)
        except Exception as exc:  # noqa: BLE001 - a bad request never kills the server
            traceback.print_exc()
            self.send_json({"error": f"server error: {exc}"}, status=500)

    def do_POST(self) -> None:
        route = self.path.split("?", 1)[0]
        try:
            body = self.read_body()
            table = self.table
            if route == "/api/start":
                table.start(body)
            elif route == "/api/action":
                table.apply_action(body)
            elif route == "/api/continue":
                table.continue_round()
            elif route == "/api/quit":
                table.quit_session()
            else:
                raise WebError(f"not found: {route}", status=404)
            self.send_json(table.snapshot())
        except WebError as exc:
            self.send_json({"error": str(exc)}, status=exc.status)
        except Exception as exc:  # noqa: BLE001 - a bad request never kills the server
            traceback.print_exc()
            with self.table.lock:
                self.table.phase = "error"
                self.table.error = str(exc)
                try:
                    snapshot: Any = self.table.snapshot()
                except Exception:  # noqa: BLE001 - the snapshot itself broke
                    snapshot = None
            self.send_json(
                {"error": f"server error: {exc}", "snapshot": snapshot},
                status=500,
            )


def make_handler(table: TableSession) -> type[Handler]:
    """Bind a :class:`TableSession` to a fresh handler class."""
    return type("BoundHandler", (Handler,), {"table": table})


def serve(
    table: TableSession,
    *,
    host: str,
    port: int,
    open_browser: bool = False,
) -> None:
    """Serve ``table`` until Ctrl-C; prints the pool/prior/torch banner."""
    print(
        "[7g523-web] 对手池:", ", ".join(opponent.id for opponent in table.opponents)
    )
    if torch_available():
        import torch

        print(f"[7g523-web] torch {torch.__version__} 可用：ckpt 对手可加载")
    else:
        print("[7g523-web] torch 不可用：只能玩 random（uv run --group train …）")
    if table.prior is not None:
        print(f"[7g523-web] 轨迹先验: {table.web_config.prior}")
    else:
        print("[7g523-web] 轨迹先验已关闭（冷启动）")
    if table.web_config.search is not None:
        label = table.web_config.search.get("label", "on")
        print(f"[7g523-web] 自由对战搜索包装: {label}")
    else:
        print(
            "[7g523-web] 自由对战搜索未启用（无逐局 t/K）：需要 "
            "runs/o4lite-search/web_search.py 启动器或插件"
        )
    plugin = table.web_config.plugin
    if plugin:
        if plugin.get("error"):
            print(f"[7g523-web] 插件不可用（已回退 raw-only）: {plugin.get('source')}")
        else:
            print(f"[7g523-web] 插件来源: {plugin.get('source')}")
        for warning in plugin.get("warnings") or ():
            print(f"[7g523-web] plugin warning: {warning}")
    search_rungs = [
        opponent.id
        for opponent in table.opponents
        if opponent.spec is not None and not buildable_by_grammar(opponent.spec)
    ]
    if search_rungs:
        print(
            "[7g523-web] 定级池含搜索 rung（与 raw 同表，混合调度）:",
            ", ".join(search_rungs),
        )
    else:
        print("[7g523-web] 定级池搜索 rung: 无（raw 池）")
    for warning in table.manifest_warnings:
        print("[7g523-web] warning:", warning)
    httpd = ThreadingHTTPServer((host, port), make_handler(table))
    url = f"http://{host}:{port}/"
    print(f"[7g523-web] 牌桌: {url}  (Ctrl-C 停止, api v{PROTO_VERSION})")
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[7g523-web] bye")

#!/usr/bin/env python3
"""Discover the human-playable 7g523 opponents and launch ``7g523-play``.

The registry below is the single vetted list of models that can be played
right now.  After the 2026-09 family-comparison rule change and the obs v5
migration, the checkpoint rungs trained under the old rules/layout were
retired, so the registry currently holds only the two scripted anchors
(``random``, ``greedy``).  New checkpoint entries are re-added at the seam
comment above ``OPPONENTS`` once they are trained, measured, and playable.

    uv run python tools/play_ladder.py list
    uv run python tools/play_ladder.py list --json
    uv run python tools/play_ladder.py play greedy -- --seat 1 --rounds 3

``play <id>`` resolves the id to the argv ``7g523-play`` already accepts
(``--opponent random|greedy`` or ``--checkpoint <path>``) and forwards every
remaining argument unchanged.  Checkpoint play needs the ``train`` dependency
group (torch); the scripted anchors run with the base environment.  Registry
paths are relative to the repository root and are resolved to absolute paths
before launch, so ``play`` also works from another working directory.

The table columns are placeholders for the re-added rungs: ``strength`` is
the contract label of a fresh measurement, ``arena`` a new joint
Bradley-Terry fit (mean Elo ``±`` cross-seed sd).  Every checkpoint is v5 by
construction: :func:`seven523.networks.load_agent` rejects anything else.
Values measured under the retired rules must not be mixed in, so the columns
stay ``-`` until then.
"""
from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

__all__ = [
    "KIND_LABELS",
    "OPPONENTS",
    "Opponent",
    "build_play_argv",
    "ckpt_path",
    "format_list",
    "launch_command",
    "main",
    "parse_args",
]

REPO_ROOT = Path(__file__).resolve().parent.parent

KIND_LABELS = {
    # Kinds reserved for re-added checkpoint entries (see OPPONENTS).
    "script": "脚本",
    "rung": "梯级",
    "platform": "平台",
    "practice": "练习",
    "lab": "实验室",
}


@dataclass(frozen=True, slots=True)
class Opponent:
    """One playable opponent.

    ``id`` doubles as the ``--opponent`` value for scripted entries (whose
    ``ckpt`` is ``None``).  ``strength`` is the measured strength label of a
    rung (step count for practice/lab entries); ``arena`` is an optional joint
    Bradley-Terry measurement (mean ± cross-seed sd), a different fit from
    ``strength``.
    """

    id: str
    kind: str
    strength: str
    arena: str | None
    ckpt: str | None
    note: str


#: Display order: the scripted anchors first, then any checkpoint rungs.
#:
#: 接缝（新模型接回）：新规则 / v5 观测下训练并测完强度的模型，按下面的模式在锚点
#: 之后追加 ``Opponent`` 条目，并把 id 加进 tests/test_play_ladder.py 的
#: EXPECTED_IDS（该测试同时校验 ckpt 存在）：
#:
#:     Opponent(
#:         id="<new-run>",
#:         kind="rung",                      # KIND_LABELS 中的类型
#:         strength="<新测强度标签>",         # 新口径值，不得复用旧规则数字
#:         arena="<新 arena 拟合或 None>",
#:         ckpt="runs/<new-run>/agent.pt",   # 占位符：替换为实际 run 目录
#:         note="<一句话说明>",
#:     ),
#:
#: 旧规则 / 旧观测（obs v1-v3）的 ``runs/*.pt`` 已全部退役，不要接回。
OPPONENTS: tuple[Opponent, ...] = (
    Opponent(
        id="random",
        kind="script",
        strength="锚 1000（固定）",
        arena="1000（锚定）",
        ckpt=None,
        note="均匀随机出合法牌；熟悉规则和界面用。",
    ),
    Opponent(
        id="greedy",
        kind="script",
        strength="锚 1315（固定）",
        arena="1315（锚定）",
        ckpt=None,
        note="贪心一手；所有比较的固定参照，也是默认对手。",
    ),
)

BY_ID: dict[str, Opponent] = {opponent.id: opponent for opponent in OPPONENTS}


def ckpt_path(opponent: Opponent) -> Path | None:
    """Absolute checkpoint path, or ``None`` for a scripted opponent."""
    if opponent.ckpt is None:
        return None
    return REPO_ROOT / opponent.ckpt


def build_play_argv(
    opponent: Opponent, extra: Sequence[str] = ()
) -> list[str]:
    """The argv ``seven523.play.main`` expects for this opponent."""
    argv = ["--opponent", opponent.id] if opponent.ckpt is None else [
        "--checkpoint",
        str(ckpt_path(opponent)),
    ]
    return [*argv, *extra]


def launch_command(opponent: Opponent) -> str:
    """Copy-pasteable ``7g523-play`` invocation (paths relative to the root)."""
    if opponent.ckpt is None:
        return f"uv run 7g523-play --opponent {opponent.id}"
    return (
        "uv run --group train 7g523-play "
        f"--checkpoint {opponent.ckpt}"
    )


def _display_width(text: str) -> int:
    return sum(
        2 if unicodedata.east_asian_width(char) in "WF" else 1 for char in text
    )


def _pad(text: str, width: int) -> str:
    return text + " " * max(0, width - _display_width(text))


def format_list(opponents: Sequence[Opponent] = OPPONENTS) -> str:
    """The human-readable table plus one launch command per opponent."""
    headers = (
        "id", "类型", "强度 / 步数", "arena Elo（3seed）", "ckpt", "说明"
    )
    rows: list[tuple[str, ...]] = []
    missing: list[Opponent] = []
    for opponent in opponents:
        path = ckpt_path(opponent)
        ckpt_cell = opponent.ckpt or "-"
        if path is not None and not path.is_file():
            ckpt_cell += "（缺失）"
            missing.append(opponent)
        rows.append(
            (
                opponent.id,
                KIND_LABELS.get(opponent.kind, opponent.kind),
                opponent.strength,
                opponent.arena or "-",
                ckpt_cell,
                opponent.note,
            )
        )

    widths = [
        max(_display_width(cell) for cell in column)
        for column in zip(headers, *rows)
    ]
    lines = [
        "可玩对手（在仓库根目录执行；ckpt 需要 train 依赖组 torch）",
        "",
        "  ".join(
            _pad(header, width) for header, width in zip(headers, widths)
        ).rstrip(),
        "  ".join("-" * width for width in widths),
    ]
    for row in rows:
        lines.append(
            "  ".join(
                _pad(cell, width) for cell, width in zip(row, widths)
            ).rstrip()
        )

    lines += [
        "",
        "启动命令（在仓库根目录执行；`python tools/play_ladder.py play <id> "
        "[-- 额外参数...]` 会生成同样的 argv）：",
    ]
    id_width = max((_display_width(opponent.id) for opponent in opponents), default=0)
    for opponent in opponents:
        lines.append(
            f"  {_pad(opponent.id, id_width)}  {launch_command(opponent)}"
        )
    lines += [
        "",
        "`--group train` 只在 ckpt 条目需要（torch）；脚本锚点用 "
        "`uv run 7g523-play --opponent random|greedy` 即可。",
        "透传示例：`uv run python tools/play_ladder.py "
        "play greedy -- --seat 1 --rounds 3`",
    ]
    if not any(opponent.ckpt is not None for opponent in opponents):
        lines += [
            "",
            "当前没有 checkpoint 档位：旧规则 / 旧观测模型已全部退役；"
            "新模型接回方式见 OPPONENTS 上方的接缝注释。",
        ]
    if missing:
        lines += [
            "",
            "警告：以下 checkpoint 缺失，play 会报错："
            + "、".join(opponent.id for opponent in missing),
        ]
    return "\n".join(lines)


def _entry_document(opponent: Opponent) -> dict[str, object]:
    path = ckpt_path(opponent)
    return {
        "id": opponent.id,
        "kind": opponent.kind,
        "strength": opponent.strength,
        "arena": opponent.arena,
        "ckpt": opponent.ckpt,
        "path": None if path is None else str(path),
        "note": opponent.note,
        "play_argv": build_play_argv(opponent),
        "command": launch_command(opponent),
    }


def cmd_list(args: argparse.Namespace) -> int:
    if args.json:
        print(
            json.dumps(
                [_entry_document(opponent) for opponent in OPPONENTS],
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(format_list())
    return 0


def cmd_play(opponent_id: str, extra: Sequence[str]) -> int:
    opponent = BY_ID.get(opponent_id)
    if opponent is None:
        print(
            f"play_ladder: 未知对手 id {opponent_id!r}；可用："
            + ", ".join(BY_ID),
            file=sys.stderr,
        )
        return 2
    path = ckpt_path(opponent)
    if path is not None and not path.is_file():
        print(
            f"play_ladder: 检查点不存在：{opponent.ckpt}（{path}）；"
            "先运行 `list` 查看可用档位。",
            file=sys.stderr,
        )
        return 2
    forwarded = list(extra)
    if forwarded[:1] == ["--"]:  # accept `play <id> -- ...` called directly
        forwarded = forwarded[1:]
    from seven523.play import main as play_main  # lazy: keeps list/--help torch-free

    play_main(build_play_argv(opponent, forwarded))
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="play_ladder",
        description=(
            "列出人机对战可用模型并启动 7g523-play"
            "（docs/human-play.md）"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="{list,play}")
    list_parser = subparsers.add_parser(
        "list", help="列出全部可玩对手、强度标签与启动命令"
    )
    list_parser.add_argument(
        "--json", action="store_true", help="输出 JSON（供脚本消费）"
    )
    play_parser = subparsers.add_parser(
        "play", help="用 7g523-play 启动某个对手"
    )
    play_parser.add_argument("id", help="list 里的对手 id")
    play_parser.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="透传给 7g523-play 的参数（建议用 -- 分隔）",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "list":
        return cmd_list(args)
    return cmd_play(args.id, args.extra)


if __name__ == "__main__":
    raise SystemExit(main())

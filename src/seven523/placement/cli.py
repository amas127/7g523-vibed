"""The ``7g523-elo`` CLI: argument parsing and the interactive/simulated run."""
from __future__ import annotations

import argparse
import random
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from ..play import interactive_chooser
from ..policies import Policy, buildable_by_grammar, policy_from_spec
from ..prior import TracePrior
from ..study import load_manifest
from .estimator import SessionConfig
from .opponents import Opponent, load_opponents
from .session import PlacementSession, new_session_id

# -- CLI ---------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="7g523-elo",
        description=(
            "10 局定级会话：轨迹先验 + OpenSkill（Plackett–Luce）结果似然，输出点估计、"
            "诚实 CI、最近档与 provisional 标记（docs/human-elo-plan.md M2/M3）"
        ),
    )
    parser.add_argument(
        "--manifest",
        default="traces/pool10/manifest.json",
        help=(
            "研究 manifest（levels/anchors/subjects；默认 traces/pool10/manifest.json，"
            "w2m/T23 revision-3 重标定后的 10 级当前池，与 traces/study/manifest.json 同源）"
        ),
    )
    parser.add_argument(
        "--prior",
        default="artifacts/human-elo/prior.json",
        help="M1 轨迹先验产物（默认 artifacts/human-elo/prior.json）",
    )
    parser.add_argument(
        "--no-trace-prior",
        action="store_true",
        help="只跑结果似然（冷启动；不需要 numpy/M1 产物）",
    )
    parser.add_argument(
        "--sessions-dir",
        default="traces/sessions",
        help="会话目录的父目录（默认 traces/sessions）",
    )
    parser.add_argument("--session-id", default=None, help="会话 id（默认时间戳）")
    parser.add_argument("--games", type=int, default=10, help="局数（偶数，默认 10）")
    parser.add_argument("--seed", type=int, default=None, help="发牌/调度随机种子")
    parser.add_argument(
        "--seat-start", type=int, choices=(0, 1), default=0, help="首局座位（默认 0）"
    )
    parser.add_argument("--stop-ci", type=float, default=50.0, help="CI 半宽停止阈值")
    parser.add_argument(
        "--min-games",
        type=int,
        default=1,
        help="早停前至少完成的局数（HR §5.4 可选用 8）",
    )
    parser.add_argument(
        "--explore-games",
        type=int,
        default=2,
        help="前几局用 Thompson 探索（默认 2，HR §6.2）",
    )
    parser.add_argument(
        "--simulate",
        default=None,
        metavar="SPEC",
        help="非交互：用 random / ckpt:<path> 当“真人”跑完整会话",
    )
    parser.add_argument("--device", default="cpu", help="ckpt 对手的设备（默认 cpu）")
    parser.add_argument(
        "--verify-traces",
        action="store_true",
        help="每局特征提取前重放校验（默认关；在线路径刚打完不必重放）",
    )
    parser.add_argument("--quiet", action="store_true", help="静音对局过程输出")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    policy_factory: Callable[[Opponent, int], Policy] | None = None,
) -> int:
    """Run ``7g523-elo``; ``policy_factory`` is the optional injection seam.

    Without a factory the pool is gated by the stock ``random``/``ckpt:``
    grammar: a manifest ``rolloutt:`` search rung (or any other spec the
    grammar cannot build) is skipped by :func:`load_opponents` with a warning
    instead of crashing at the first game.  A caller that passes a factory
    (whose ``(Opponent, seed)`` contract matches
    :class:`~seven523.placement.PlacementSession`) opts the whole pool in —
    anchors, raw rungs and search rungs in one joint-fit pool.  The trace
    prior stays on for every raw game; a played search rung's row is excluded
    from the trace channel with per-game provenance because its μ lies beyond
    the raw prior's fitting domain (ADR-0013).
    """
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    print_fn: Callable[..., None] = (lambda *a, **k: None) if args.quiet else print

    def can_build(spec: str | None) -> bool:
        if policy_factory is not None and spec is not None:
            return True
        return buildable_by_grammar(spec)

    try:
        opponents, anchors = load_opponents(
            load_manifest(args.manifest), can_build=can_build
        )
        trace_prior = None if args.no_trace_prior else TracePrior.load(args.prior)
        session_id = args.session_id or new_session_id(args.sessions_dir)
        directory = Path(args.sessions_dir) / session_id
        session = PlacementSession(
            opponents=opponents,
            anchors=anchors,
            directory=directory,
            session_id=session_id,
            config=SessionConfig(
                games=args.games,
                stop_ci=args.stop_ci,
                min_games_before_stop=args.min_games,
                explore_games=args.explore_games,
                seat_start=args.seat_start,
            ),
            rng=random.Random(args.seed),
            trace_prior=trace_prior,
            policy_factory=policy_factory,
            feature_verify=args.verify_traces,
            device=args.device,
        )
        if args.simulate:
            human = policy_from_spec(
                args.simulate, session.rules, seed=args.seed, device=args.device
            )
            report = session.run(
                human_policy_factory=lambda _seat, _policy=human: _policy,
                print_fn=print_fn,
            )
        else:
            chooser_factory = lambda seat: interactive_chooser(seat, print_fn=print_fn)  # noqa: E731
            report = session.run(chooser_factory, print_fn=print_fn)
    except (FileNotFoundError, ValueError, KeyError, OSError) as exc:
        print(f"7g523-elo: {exc}", file=sys.stderr)
        return 1

    if report is None:
        print("没有完成任何一局，未生成报告。")
        return 1
    human = report["human"]
    nearest = report["nearest_level"] or {}
    flag = "临时 provisional" if report["provisional"] else "已定级"
    print(
        f"定级：{human['mu']:.0f} ± {human['ci_half_width']:.0f}（95% CI），"
        f"最近档 {nearest.get('id', '?')}（{nearest.get('mu', float('nan')):.0f}），"
        f"{flag}"
    )
    print(f"报告：{report['session']['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Play one game in the terminal against a scripted bot or a trained agent.

Human plays one seat (``--seat``, default 0); every other seat is an opponent
policy.  Without ``--checkpoint`` the opponent is ``GreedyBot`` (default) or
``RandomBot``.

    uv run 7g523-play
    uv run --group train 7g523-play --checkpoint runs/<run>/agent.pt
    uv run 7g523-play --opponent random --seat 1 --rounds 3
    uv run 7g523-play --save-trace traces/my_game.json    # 保存对局轨迹
    uv run 7g523-play --replay traces/my_game.json         # 回放
"""
from __future__ import annotations

import argparse
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

from .actions import legal_ids, resolve, suit_options
from .cards import SUIT_LABELS, Suit, card_key
from .combos import Combo, ComboKind
from .game import Game, GameState, StepResult, View
from .match import Match
from .policies import GreedyBot, Policy, RandomBot
from .rules import DEFAULT_RULES, Rules
from .trace import (
    TRACE_VERSION,
    build_trace,
    initial_snapshot,
    load_trace,
    player_label,
    rules_from_json,
    save_trace,
    state_from_snapshot,
    step_record,
)

__all__ = [
    "HELP",
    "ChooserPolicy",
    "QuitGame",
    "action_text",
    "combo_text",
    "interactive_chooser",
    "main",
    "opponent_identity",
    "parse_args",
    "parse_choice",
    "play_game",
    "replay_trace",
    "state_panel",
    "suit_hint",
]

HELP = """\
输入可选列表里的编号出牌；q 退出，h 显示帮助。
顶牌有多花色可选时，列表会提示（默认 ♠，可换 ♥）；输入 编号+花色 即可，
如 10♥ 或 10 2；直接输编号则用默认最强花色。炸弹/王炸与花色无关。
牌型大小：单张族（单牌 < 顺子）、对子族（对子 < 连对）各自比较，跨族不可压；
炸弹压所有：小炸弹（含王炸）< 大炸弹；同族先比张数再比最大牌；同点数比花色（♠ > ♥ > ♣ > ♦）。
顺子/连对按自然环 3-4-5-6-7-8-9-10-J-Q-K-A-2，A-2-3 合法。"""

KIND_LABELS: dict[ComboKind, str] = {
    ComboKind.SINGLE: "单牌",
    ComboKind.PAIR: "对子",
    ComboKind.STRAIGHT: "顺子",
    ComboKind.CONSECUTIVE_PAIRS: "连对",
    ComboKind.SMALL_BOMB: "小炸弹",
    ComboKind.BIG_BOMB: "大炸弹",
}


class QuitGame(Exception):
    """The human asked to stop mid-game."""


def combo_text(combo: Combo) -> str:
    """One-line label for a played combo, e.g. ``对子 ♥K ♣K``."""
    if (
        combo.kind is ComboKind.SMALL_BOMB
        and len(combo.cards) == 2
        and all(card.is_joker for card in combo.cards)
    ):
        kind = "王炸"
    else:
        kind = KIND_LABELS[combo.kind]
    return f"{kind} " + " ".join(str(card) for card in combo.cards)


def action_text(game: Game, action_id: int, hand, suit: int | None = None) -> str:
    """Menu line for one catalog action: template label + concrete cards."""
    action = game.catalog[action_id]
    if action.is_pass:
        return "过"
    combo = resolve(action, hand, game.rules, suit=suit)
    if combo is None:
        return f"{action.label} [?]"
    return combo_text(combo)


def suit_hint(game: Game, action_id: int, hand) -> str:
    """Inline hint listing the alternative top-card suits, or ``""``."""
    options = suit_options(game.catalog[action_id], hand, game.rules)
    if len(options) < 2:
        return ""
    others = " ".join(SUIT_LABELS[suit] for suit in options[1:])
    return f"（默认 {SUIT_LABELS[options[0]]}，可换 {others}）"


def parse_choice(raw: str, legal: list[int], game: Game, view: View) -> tuple[int, int | None]:
    """Parse ``10`` / ``10♥`` / ``10 ♥`` / ``10 2`` into a joint action.

    Raises ``ValueError`` with a user-facing message for anything invalid.
    """
    suit: int | None = None
    raw = raw.strip()
    for char, value in {"♠": 3, "♥": 2, "♣": 1, "♦": 0}.items():
        if raw.endswith(char):
            suit = value
            raw = raw[:-1].strip()
            break
    else:
        parts = raw.split()
        if len(parts) == 2:
            raw = parts[0]
            try:
                suit = int(parts[1])
            except ValueError as exc:
                raise ValueError("第二个数字要填花色编号（0-3）。") from exc
    try:
        action_id = int(raw)
    except ValueError as exc:
        raise ValueError("请输入列表里的编号，例如 10 或 10♥。") from exc
    if action_id not in legal:
        raise ValueError("这不是合法出牌，请从上面编号里选。")
    if suit is None:
        return action_id, None
    options = [int(value) for value in suit_options(game.catalog[action_id], view.hand, game.rules)]
    if not options:
        raise ValueError("这手牌与花色无关（炸弹或过），直接输入编号即可。")
    if suit not in options:
        labels = " ".join(SUIT_LABELS[Suit(value)] for value in options)
        raise ValueError(f"顶牌不能用这张花色，可选：{labels}")
    return action_id, suit


def _seat_name(seat: int, human_seat: int, num_players: int) -> str:
    if seat == human_seat:
        return "你"
    if num_players == 2:
        return "对手"
    return f"对手{seat}"


def state_panel(game: Game, state: GameState, human_seat: int) -> str:
    """The board as the human sees it (public facts + own hand)."""
    rules = game.rules
    names = [_seat_name(seat, human_seat, rules.num_players) for seat in range(rules.num_players)]
    lines = [
        "─" * 46,
        "分数    " + "  |  ".join(
            f"{names[seat]} {state.scores[seat]}" for seat in range(rules.num_players)
        ),
        "手牌数  " + "  |  ".join(
            f"{names[seat]} {len(state.hands[seat])}" for seat in range(rules.num_players)
        )
        + f"    底牌堆 {len(state.draw_pile)} 张",
        f"本墩    {state.trick_points} 分已打出",
    ]
    if state.trick_cards:
        lines.append("弃牌    " + " ".join(str(card) for card in state.trick_cards))
    if state.incumbent is not None:
        owner = state.last_player if state.last_player is not None else state.current
        lines.append(f"当前牌  {names[owner]} 的 {combo_text(state.incumbent)}")
    lines.append(f"轮到    {names[state.current]}")
    hand = " ".join(str(card) for card in sorted(state.hand(human_seat), key=card_key))
    lines.append(f"你的手牌  {hand}")
    return "\n".join(lines)


def _trick_text(result: StepResult, names: list[str]) -> str:
    winner = result.winner
    assert winner is not None
    text = f"墩结束：{names[winner]} 收下 {result.points_taken} 分"
    if result.refilled:
        who = "、".join(names[seat] for seat in result.refilled)
        text += f"（补牌：{who}）"
    if result.dug:
        text += "，撬底！"
    return text


# -- game replay -------------------------------------------------------------


def replay_trace(
    trace: dict[str, Any],
    *,
    print_fn: Callable[..., None] = print,
    pause: bool = False,
    input_fn: Callable[[str], str] = input,
) -> bool:
    """Re-run a saved game and check every step against the recording.

    Raises ``ValueError`` on any divergence (illegal action, wrong seat,
    different score), which makes a trace an integrity check of the engine.
    """
    version = int(trace.get("version", 1))
    if version > TRACE_VERSION:
        raise ValueError(f"trace version {version} is newer than {TRACE_VERSION}")
    rules = rules_from_json(trace["rules"])
    state = state_from_snapshot(trace["initial"], rules)
    match = Match(rules, [None] * rules.num_players, state=state)
    human_seat = int(trace.get("human_seat", 0))
    names = [
        _seat_name(seat, human_seat, rules.num_players)
        for seat in range(rules.num_players)
    ]
    players = trace.get("players", names)
    print_fn(
        f"回放：{trace.get('created_at', '?')}  seed={trace.get('seed')}  "
        + " / ".join(str(player) for player in players)
    )
    reveal = " / ".join(
        f"{names[seat]} {state.revealed[seat]}" for seat in range(rules.num_players)
    )
    print_fn(f"亮牌：{reveal}（{names[state.current]}先手）")

    for index, step in enumerate(trace["steps"]):
        if state.done:
            raise ValueError(f"step {index}: trace has moves after the game ended")
        if state.current != step["seat"]:
            raise ValueError(
                f"step {index}: seat {state.current} is to move, "
                f"but the trace says {step['seat']}"
            )
        if pause:
            if input_fn("回车继续（q 退出回放）> ").strip() in {"q", "quit", "退出"}:
                raise QuitGame
        view = match.view()
        action_id = int(step["action"])
        suit = step.get("suit")
        if not (view.mask >> action_id) & 1:
            raise ValueError(f"step {index}: recorded action {action_id} is illegal")
        print_fn(
            f"[{index:>3}] {names[step['seat']]}："
            + action_text(match.game, action_id, view.hand, suit)
        )
        result = match.step(action_id, suit)
        state = match.state
        if list(state.scores) != [int(score) for score in step["scores"]]:
            raise ValueError(
                f"step {index}: scores diverged "
                f"{state.scores} != {step['scores']}"
            )
        if result.trick_over:
            print_fn("      " + _trick_text(result, names))

    if not state.done:
        raise ValueError("trace ended before the game did")
    if list(state.scores) != [int(score) for score in trace["final_scores"]]:
        raise ValueError(
            f"final scores diverged {state.scores} != {trace['final_scores']}"
        )
    print_fn(
        "回放结束："
        + "  ".join(
            f"{names[seat]} {state.scores[seat]}" for seat in range(rules.num_players)
        )
    )
    return True


class ChooserPolicy:
    """Adapter from the legacy ``(game, state, view)`` chooser to ``Policy``.

    The interactive/scripted chooser needs the live :class:`Game` and
    :class:`GameState`, which a :class:`~seven523.policies.Policy` never sees;
    :meth:`bind` attaches the :class:`Match` that owns them before play starts.
    """

    def __init__(self, chooser: Callable[[Game, GameState, View], Any]) -> None:
        self.chooser = chooser
        self._match: Match | None = None

    def bind(self, match: Match) -> None:
        self._match = match

    def act(self, view: View) -> Any:
        if self._match is None:
            raise RuntimeError("ChooserPolicy.act() before bind(match)")
        return self.chooser(self._match.game, self._match.state, view)


def play_game(
    policies: Sequence[Policy],
    *,
    chooser: Callable[[Game, GameState, View], Any] | None = None,
    rules: Rules = DEFAULT_RULES,
    human_seat: int = 0,
    seed: int | None = None,
    print_fn: Callable[..., None] = print,
    record: dict[str, Any] | None = None,
) -> tuple[int, ...]:
    """Deal and play one game; every seat is a :class:`Policy`.

    With ``chooser`` given, seat ``human_seat`` is wrapped in a bound
    :class:`ChooserPolicy` (interactive or scripted); otherwise the seat's own
    entry in ``policies`` acts.  ``chooser`` may raise :class:`QuitGame` to
    abandon the game.  When ``record`` is given it is filled with ``initial``
    (the deal), ``steps`` (one entry per move) and ``final_scores``.
    Returns the final scores (one per seat).
    """
    seat_policies: list[Policy | None] = list(policies)
    chooser_policy: ChooserPolicy | None = None
    if chooser is not None:
        chooser_policy = ChooserPolicy(chooser)
        seat_policies[human_seat] = chooser_policy
    match = Match(rules, seat_policies, rng=random.Random(seed))
    if chooser_policy is not None:
        chooser_policy.bind(match)
    game = match.game
    names = [
        _seat_name(seat, human_seat, rules.num_players)
        for seat in range(rules.num_players)
    ]
    if record is not None:
        record["initial"] = initial_snapshot(match.state)
        record["steps"] = []

    def on_turn(seat, action_id, suit, view, result):
        label = "你" if seat == human_seat else names[seat]
        text = action_text(game, action_id, view.hand, suit)
        print_fn(f"{label}出了 " + text)
        if record is not None:
            record["steps"].append(
                step_record(
                    seat,
                    action_id,
                    suit,
                    text=text,
                    state=match.state,
                    result=result,
                )
            )
        if result.trick_over:
            print_fn(_trick_text(result, names))

    match.on_turn = on_turn

    state = match.state
    reveal = " / ".join(
        f"{names[seat]} {state.revealed[seat]}" for seat in range(rules.num_players)
    )
    print_fn(f"亮牌：{reveal}（{names[state.current]}先手）")

    match.run_to_end()

    state = match.state
    best = max(state.scores)
    winners = [seat for seat, score in enumerate(state.scores) if score == best]
    if len(winners) > 1:
        outcome = "平局"
    elif winners[0] == human_seat:
        outcome = "你赢了！"
    else:
        outcome = f"{names[winners[0]]}赢了"
    scores = "  ".join(
        f"{names[seat]} {state.scores[seat]}" for seat in range(rules.num_players)
    )
    print_fn(f"本局结束：{scores} —— {outcome}")
    if record is not None:
        record["final_scores"] = list(state.scores)
    return state.scores


def opponent_identity(checkpoint: str | None, opponent: str = "greedy") -> tuple[str, str]:
    """Trace identity ``(role, id)`` of a ``7g523-play`` opponent.

    A checkpoint labels by its run directory: ``runs/<new-run>/agent.pt``
    becomes ``opponent:<new-run>@seatN`` (the directory name carries the run
    id), so a human trace can be calibrated against the measured ladder
    (HR §6.3/§8).
    The scripted bots are the pinned anchors and label as
    ``anchor:greedy@seatN`` / ``anchor:random@seatN``, matching the study
    traces.  Display names (``贪心 bot`` …) are not identities and never reach
    the trace.
    """
    if checkpoint:
        path = Path(checkpoint)
        name = path.parent.name if path.parent != Path(".") else path.stem
        return "opponent", name or "agent"
    if opponent not in {"greedy", "random"}:
        raise ValueError(f"unknown scripted opponent {opponent!r}")
    return "anchor", opponent


def interactive_chooser(
    human_seat: int, print_fn: Callable[..., None] = print
) -> Callable[[Game, GameState, View], int]:
    def chooser(game: Game, state: GameState, view: View) -> int:
        print_fn(state_panel(game, state, human_seat))
        legal = legal_ids(view.mask)
        print_fn("可选出牌：")
        for action_id in legal:
            hint = suit_hint(game, action_id, view.hand)
            suffix = f"  {hint}" if hint else ""
            print_fn(
                f"  [{action_id:>3}] {action_text(game, action_id, view.hand)}{suffix}"
            )
        while True:
            try:
                raw = input("请选择编号（h 帮助，q 退出）> ").strip()
            except EOFError as exc:
                raise QuitGame from exc
            if raw in {"q", "quit", "exit", "退出"}:
                raise QuitGame
            if raw in {"h", "?", "help", "帮助"}:
                print_fn(HELP)
                continue
            try:
                return parse_choice(raw, legal, game, view)
            except ValueError as exc:
                print_fn(str(exc))
                continue

    return chooser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="和 bot / 训练好的 agent 玩一局 7鬼523")
    parser.add_argument("--checkpoint", type=str, default=None, help="agent.pt（不传则用脚本 bot）")
    parser.add_argument("--opponent", choices=["greedy", "random"], default="greedy")
    parser.add_argument("--seat", type=int, default=0, help="你坐哪一家（默认 0）")
    parser.add_argument("--num-players", type=int, default=2)
    parser.add_argument("--seed", type=int, default=None, help="缺省随机")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--sample", action="store_true", help="神经对手按 mask 采样而非 argmax")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument(
        "--save-trace",
        type=str,
        default=None,
        metavar="PATH",
        help="把每局轨迹写成 JSON；多局时文件名自动加 _局号",
    )
    parser.add_argument(
        "--replay", type=str, default=None, metavar="PATH", help="回放保存的轨迹，不再对局"
    )
    parser.add_argument("--pause", action="store_true", help="回放时每步按回车")
    return parser.parse_args(argv)


def _trace_path(path: Path, round_no: int, rounds: int) -> Path:
    if rounds == 1:
        return path
    return path.with_name(f"{path.stem}_{round_no}{path.suffix or '.json'}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.replay:
        try:
            replay_trace(load_trace(args.replay), pause=args.pause)
        except (QuitGame, KeyboardInterrupt):
            print("已退出回放。")
        return

    rules = Rules(num_players=args.num_players)
    if args.checkpoint:
        from .networks import NeuralPolicy, load_agent  # lazy: torch is train-only

        agent, _ = load_agent(args.checkpoint, device=args.device)
        policy: Policy = NeuralPolicy(
            agent, rules, device=args.device, sample=args.sample
        )
        opponent_name = f"神经网络（{args.checkpoint}）"
    elif args.opponent == "random":
        policy = RandomBot()
        opponent_name = "随机 bot"
    else:
        policy = GreedyBot(rules)
        opponent_name = "贪心 bot"

    policies = [policy] * rules.num_players
    chooser = interactive_chooser(args.seat)
    role, identity = opponent_identity(args.checkpoint, args.opponent)
    print(f"对手：{opponent_name}；你坐 {args.seat} 号位。输入 h 看帮助。")
    try:
        for round_no in range(1, args.rounds + 1):
            if args.rounds > 1:
                print(f"===== 第 {round_no}/{args.rounds} 局 =====")
            seed = (
                random.randrange(1 << 32)
                if args.seed is None
                else args.seed + round_no - 1
            )
            record: dict[str, Any] = {}
            play_game(
                policies,
                chooser=chooser,
                rules=rules,
                human_seat=args.seat,
                seed=seed,
                record=record,
            )
            if args.save_trace:
                trace = build_trace(
                    rules,
                    seed=seed,
                    human_seat=args.seat,
                    players=[
                        f"human@seat{args.seat}"
                        if seat == args.seat
                        else player_label(role, identity, seat)
                        for seat in range(rules.num_players)
                    ],
                    created_at=datetime.now().isoformat(timespec="seconds"),
                    **record,
                )
                path = _trace_path(Path(args.save_trace), round_no, args.rounds)
                save_trace(path, trace)
                print(f"轨迹已保存：{path}")
    except QuitGame:
        print("已退出。")
    except KeyboardInterrupt:
        print("\n已退出。")


if __name__ == "__main__":
    main()

"""Evaluation of a learned policy against scripted opponents (DESIGN §7).

Reports average score, average score diff, win rate, episode length and the
illegal-action rate (which must stay 0: the mask is the legality authority).
"""
from __future__ import annotations

import argparse
import json
import statistics
from typing import Sequence

from .actions import split_action
from .env import Seven523Env
from .policies import Policy, make_scripted_policies
from .rules import DEFAULT_RULES, Rules

__all__ = ["evaluate", "main", "parse_args"]


def evaluate(
    policy: Policy,
    *,
    rules: Rules = DEFAULT_RULES,
    opponent: str = "greedy",
    episodes: int = 200,
    seed: int = 0,
    learner: int = 0,
    opponents: Sequence[Policy] | None = None,
) -> dict[str, float]:
    """Play ``episodes`` games as ``learner`` and aggregate the results."""
    if opponents is None:
        opponents = make_scripted_policies(opponent, rules, seed)
    env = Seven523Env(rules=rules, opponents=list(opponents), seed=seed, learner=learner)

    returns: list[float] = []
    lengths: list[int] = []
    scores: list[float] = []
    opp_scores: list[float] = []
    diffs: list[float] = []
    wins = draws = 0
    illegal = 0
    steps = 0

    for _ in range(episodes):
        env.reset()
        done = False
        last_reward = 0.0
        length = 0
        while not done:
            view = env.game.view(env.state, learner)
            action_id, suit = split_action(policy.act(view))
            if (
                not 0 <= action_id < env.action_space_n
                or not env.action_mask[action_id]
            ):
                illegal += 1
                action_id = next(
                    index
                    for index in range(env.action_space_n)
                    if env.action_mask[index]
                )
                suit = None
            _, last_reward, done, _, _ = env.step((action_id, suit))
            length += 1
            steps += 1

        final = env.state.scores
        own = final[learner]
        others = [score for seat, score in enumerate(final) if seat != learner]
        best_other = max(others)
        wins += own > best_other
        draws += own == best_other
        returns.append(last_reward)
        lengths.append(length)
        scores.append(own)
        opp_scores.append(statistics.fmean(others))
        diffs.append(own - statistics.fmean(others))

    return {
        "episodes": float(episodes),
        "learner_score": statistics.fmean(scores),
        "opponent_score": statistics.fmean(opp_scores),
        "score_diff": statistics.fmean(diffs),
        "win_rate": wins / episodes,
        "draw_rate": draws / episodes,
        "loss_rate": 1.0 - (wins + draws) / episodes,
        "mean_return": statistics.fmean(returns),
        "mean_length": statistics.fmean(lengths),
        "illegal_rate": illegal / steps if steps else 0.0,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a PPO checkpoint for 7鬼523")
    parser.add_argument("--checkpoint", type=str, required=True, help="agent.pt")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--opponent", choices=["greedy", "random"], default="greedy")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-players", type=int, default=2)
    parser.add_argument(
        "--sample", action="store_true", help="sample instead of acting greedily"
    )
    parser.add_argument("--json", action="store_true", help="print raw JSON")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    from .networks import NeuralPolicy, load_agent  # lazy: torch is a train-only dep

    rules = Rules(num_players=args.num_players)
    agent, _ = load_agent(args.checkpoint, device=args.device)
    policy = NeuralPolicy(agent, rules, device=args.device, sample=args.sample)
    metrics = evaluate(
        policy,
        rules=rules,
        opponent=args.opponent,
        episodes=args.episodes,
        seed=args.seed,
    )
    if args.json:
        print(json.dumps(metrics, indent=2))
        return
    print(f"episodes       {int(metrics['episodes'])}  vs {args.opponent}")
    print(f"learner score  {metrics['learner_score']:.2f} / 100")
    print(f"opponent score {metrics['opponent_score']:.2f} / 100")
    print(f"score diff     {metrics['score_diff']:+.2f}")
    print(
        f"win/draw/loss  {metrics['win_rate']:.1%} / {metrics['draw_rate']:.1%} / "
        f"{metrics['loss_rate']:.1%}"
    )
    print(f"mean return    {metrics['mean_return']:+.3f}")
    print(f"mean length    {metrics['mean_length']:.1f} steps")
    print(f"illegal rate   {metrics['illegal_rate']:.4%}")


if __name__ == "__main__":
    main()

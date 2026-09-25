import random

import gymnasium as gym
import numpy as np
import pytest

from seven523.actions import CATALOG, SUIT_N, catalog_for
from seven523.env import Seven523Env, encode_observation, observation_dim
from seven523.game import Game
from seven523.policies import GreedyBot, RandomBot
from seven523.rules import Rules


def test_observation_dim():
    assert observation_dim(2) == 191
    assert observation_dim(3) == 194


@pytest.mark.parametrize("num_players", [2, 3, 4, 5, 6, 7])
def test_observation_dim_formula(num_players):
    assert observation_dim(num_players) == 185 + 3 * num_players


@pytest.mark.parametrize("num_players", [2, 3, 4, 5, 6, 7])
def test_encode_observation_length_matches_dim(num_players):
    rules = Rules(num_players=num_players)
    game = Game(rules)
    state = game.new(random.Random(0))
    view = game.view(state, 0)
    obs = encode_observation(view, rules)
    assert len(obs) == observation_dim(num_players)


def run_env_episode(env, seed=0):
    obs, info = env.reset()
    assert info == {}
    assert obs.shape == env.observation_space.shape
    assert env.observation_space.contains(obs)
    assert len(env.action_mask) == int(env.action_space.nvec.sum())
    rng = random.Random(seed)
    done = False
    reward = 0.0
    steps = 0
    while not done:
        legal = [
            index
            for index in range(env.action_space_n)
            if env.action_mask[index]
        ]
        assert legal, "the learner must always have a legal action"
        obs, reward, done, truncated, info = env.step(
            (rng.choice(legal), rng.randrange(SUIT_N))
        )
        assert obs.shape == env.observation_space.shape
        assert info == {} and not truncated
        steps += 1
        assert steps < 50_000
    assert -1.0 <= reward <= 1.0
    return reward


def test_env_spaces():
    env = Seven523Env(seed=0)
    assert isinstance(env.action_space, gym.spaces.MultiDiscrete)
    assert env.action_space.nvec.tolist() == [len(CATALOG), SUIT_N]
    assert env.observation_space.shape == (observation_dim(2),)
    assert env.observation_space.dtype == np.float32


def test_env_state_before_reset_raises():
    env = Seven523Env(seed=0)
    with pytest.raises(RuntimeError):
        _ = env.state


def test_env_action_space_matches_rules_catalog():
    for rules in (
        Rules(),
        Rules(num_players=3),
        Rules(straight_max=13),
        Rules(num_players=3, straight_max=13, consecutive_pairs_max=6),
    ):
        env = Seven523Env(rules=rules, seed=0)
        assert env.action_space_n == len(catalog_for(rules))
        assert env.action_space.nvec.tolist() == [len(catalog_for(rules)), SUIT_N]
        run_env_episode(env)


def test_env_custom_rules_episode_is_finite_and_legal():
    rules = Rules(num_players=3, straight_max=13)
    env = Seven523Env(rules=rules, seed=1)
    run_env_episode(env, seed=1)


def test_env_learner_not_zero():
    rules = Rules(num_players=3)
    env = Seven523Env(
        rules=rules,
        seed=2,
        learner=2,
        opponents=[GreedyBot(rules) for _ in range(3)],
    )
    run_env_episode(env, seed=2)


def test_env_action_mask_matches_the_learner_view():
    env = Seven523Env(seed=3)
    env.reset()
    view = env.game.view(env.state, env.learner)
    assert env.action_mask == [
        bool((view.mask >> index) & 1) for index in range(env.action_space_n)
    ] + [True] * SUIT_N
    # the published mask is never empty in a playable state
    assert any(env.action_mask) or view.done


def test_env_reset_with_the_same_seed_is_reproducible():
    first = Seven523Env(seed=0, opponents=[GreedyBot(), GreedyBot()])
    second = Seven523Env(seed=0, opponents=[GreedyBot(), GreedyBot()])
    obs_a, _ = first.reset(seed=42)
    obs_b, _ = second.reset(seed=42)
    assert np.array_equal(obs_a, obs_b)
    assert first.action_mask == second.action_mask


def test_env_episode_is_finite_and_legal():
    env = Seven523Env(
        opponents=[RandomBot(random.Random(0)), GreedyBot()],
        seed=5,
    )
    obs, _ = env.reset()
    assert obs.shape == env.observation_space.shape
    assert len(env.action_mask) == len(CATALOG) + SUIT_N
    assert any(env.action_mask[: len(CATALOG)])

    rng = random.Random(6)
    done = False
    reward = 0.0
    steps = 0
    while not done:
        legal = [index for index in range(env.action_space_n) if env.action_mask[index]]
        assert legal
        obs, reward, done, truncated, info = env.step(
            (rng.choice(legal), rng.randrange(SUIT_N))
        )
        assert obs.shape == env.observation_space.shape
        assert info == {} and not truncated
        steps += 1
        assert steps < 20_000, "env episode did not terminate"

    assert -1.0 <= reward <= 1.0

"""Per-episode league policy + PFSP weighting (T5, structural direction C).

The new seam is :class:`EpisodeMixturePolicy`: a weighted pool drawn *once per
episode* (not per decision) with a settleable weight vector, plus the pure
:func:`pfsp_weights` function the trainer uses to re-weight the pool from the
learner's observed record.  These tests are torch-free on purpose.
"""
import random

import pytest
from support import FirstLegalBot

from seven523.env import Seven523Env
from seven523.policies import (
    EpisodeMixturePolicy,
    EpisodePolicy,
    MixturePolicy,
    RandomBot,
    WeightedPolicy,
    pfsp_weights,
)
from seven523.rules import DEFAULT_RULES


class _Fixed:
    """Policy stub returning one fixed template id (suit ``None``)."""

    def __init__(self, action: int) -> None:
        self.action = action
        self.calls = 0

    def act(self, view):
        self.calls += 1
        return self.action, None


def _members():
    low, high = _Fixed(0), _Fixed(1)
    return low, high, [(0.25, low, "low"), (0.75, high, "high")]


# -- EpisodeMixturePolicy ----------------------------------------------------


def test_policy_protocols_separate_plain_bots_from_episode_mixtures():
    # The env/trainer seams are opt-in: a plain bot implements neither hook,
    # while the episode mixture is the reference implementation of both.
    assert not isinstance(RandomBot(random.Random(0)), EpisodePolicy)
    assert not isinstance(RandomBot(random.Random(0)), WeightedPolicy)

    mixture = EpisodeMixturePolicy(
        [(1.0, FirstLegalBot(), "fixed"), (0.0, RandomBot(random.Random(1)), "random")],
        random.Random(0),
    )
    assert isinstance(mixture, EpisodePolicy)
    assert isinstance(mixture, WeightedPolicy)


def test_episode_mixture_act_forwards_only_to_the_frozen_member():
    low, high, members = _members()
    policy = EpisodeMixturePolicy(members, random.Random(0))
    for _ in range(20):
        policy.start_episode()
        low_before, high_before = low.calls, high.calls
        for _ in range(10):
            policy.act(object())
        if policy.current_id == "low":
            assert (low.calls, high.calls) == (low_before + 10, high_before)
        else:
            assert (low.calls, high.calls) == (low_before, high_before + 10)


def test_episode_mixture_draws_members_by_weight_across_episodes():
    _low, _high, members = _members()
    policy = EpisodeMixturePolicy(members, random.Random(0))
    picks = [policy.start_episode() for _ in range(400)]
    share = picks.count("high") / len(picks)
    assert 0.65 < share < 0.85
    # a zero weight is never drawn
    policy.set_weights([1.0, 0.0])
    assert all(policy.start_episode() == "low" for _ in range(50))


def test_episode_mixture_tracks_current_and_finished_ids():
    low, high, _ = _members()
    policy = EpisodeMixturePolicy(
        [(1.0, low, "low"), (0.0, high, "high")], random.Random(0)
    )
    assert policy.current_id is None and policy.finished_id is None
    policy.start_episode()
    assert policy.current_id == "low" and policy.finished_id is None
    policy.act(object())
    policy.start_episode()
    assert policy.current_id == "low" and policy.finished_id == "low"


def test_episode_mixture_set_weights_applies_to_the_next_episode():
    low, high, _ = _members()
    policy = EpisodeMixturePolicy(
        [(1.0, low, "low"), (0.0, high, "high")], random.Random(0)
    )
    policy.start_episode()
    for _ in range(5):
        policy.act(object())
    policy.set_weights([0.0, 1.0])
    # the episode in progress keeps the member frozen at start_episode
    for _ in range(5):
        policy.act(object())
    assert policy.current_id == "low"
    assert policy.start_episode() == "high"
    assert policy.finished_id == "low"
    assert (low.calls, high.calls) == (10, 0)


def test_episode_mixture_two_tuples_get_index_ids_and_validates_input():
    low, high, _ = _members()
    indexed = EpisodeMixturePolicy([(1.0, low), (1.0, high)], random.Random(0))
    assert indexed.start_episode() in {"0", "1"}
    with pytest.raises(ValueError):
        EpisodeMixturePolicy([])
    with pytest.raises(ValueError):
        EpisodeMixturePolicy([(-1.0, low, "low")])
    with pytest.raises(ValueError):
        EpisodeMixturePolicy([(0.0, low, "low")])
    policy = EpisodeMixturePolicy([(1.0, low, "low")], random.Random(0))
    with pytest.raises(RuntimeError):
        policy.act(object())  # no episode started yet
    with pytest.raises(ValueError):
        policy.set_weights([1.0, 2.0])
    with pytest.raises(ValueError):
        policy.set_weights([0.0])


# -- pfsp_weights ------------------------------------------------------------


def test_pfsp_weights_decrease_monotonically_with_the_learner_win_rate():
    # records are (wins, draws, losses) *from the learner's seat*.
    records = {
        "always_lost_to": (0, 0, 100),
        "even": (50, 0, 50),
        "always_beat": (100, 0, 0),
    }
    weights = pfsp_weights(records, prior=0.0, epsilon=0.0, uniform_mix=0.0)
    assert weights["always_lost_to"] > weights["even"] > weights["always_beat"]
    assert weights["always_beat"] == pytest.approx(0.0)
    assert sum(weights.values()) == pytest.approx(1.0)


def test_pfsp_weights_credit_draws_as_half_a_win():
    records = {"draws": (0, 10, 0), "coin_flip": (5, 0, 5)}
    weights = pfsp_weights(records, prior=0.0, epsilon=0.0, uniform_mix=0.0)
    assert weights["draws"] == pytest.approx(weights["coin_flip"])


def test_pfsp_weights_beta_prior_shrinks_extremes_toward_uniform():
    records = {"a": (100, 0, 0), "b": (0, 0, 100)}
    raw = pfsp_weights(records, prior=0.0, epsilon=0.0, uniform_mix=0.0)
    shrunk = pfsp_weights(records, prior=200.0, epsilon=0.0, uniform_mix=0.0)
    assert shrunk["b"] - shrunk["a"] < raw["b"] - raw["a"]
    # no games at all: the prior alone is uniform
    empty = pfsp_weights(
        {"a": (0, 0, 0), "b": (0, 0, 0)},
        prior=10.0,
        epsilon=0.0,
        uniform_mix=0.0,
    )
    assert empty == pytest.approx({"a": 0.5, "b": 0.5})


def test_pfsp_weights_epsilon_and_uniform_mix_boundaries():
    records = {"lost_to": (0, 0, 10), "beat": (10, 0, 0)}
    pure = pfsp_weights(records, prior=0.0, epsilon=0.0, uniform_mix=0.0)
    assert pure["beat"] == pytest.approx(0.0)
    assert pure["lost_to"] == pytest.approx(1.0)
    smoothed = pfsp_weights(records, prior=0.0, epsilon=0.01, uniform_mix=0.0)
    assert 0.0 < smoothed["beat"] < 0.5
    uniform = pfsp_weights(records, prior=0.0, epsilon=0.0, uniform_mix=1.0)
    assert uniform == pytest.approx({"lost_to": 0.5, "beat": 0.5})
    with pytest.raises(ValueError):
        pfsp_weights({}, prior=0.0, epsilon=0.0, uniform_mix=0.0)
    with pytest.raises(ValueError):
        pfsp_weights(records, prior=-1.0)
    with pytest.raises(ValueError):
        pfsp_weights(records, epsilon=-0.1)
    with pytest.raises(ValueError):
        pfsp_weights(records, uniform_mix=1.5)


# -- env.reset duck typing ---------------------------------------------------


class _DuckOpponent:
    """FirstLegalBot wrapped with the optional ``start_episode`` hook."""

    def __init__(self, rules=DEFAULT_RULES) -> None:
        self.inner = FirstLegalBot(rules)
        self.starts = 0

    def start_episode(self):
        self.starts += 1

    def act(self, view):
        return self.inner.act(view)


def _finish_episode(env, action_seed=0):
    rng = random.Random(action_seed)
    done = False
    while not done:
        legal = [i for i in range(env.action_space_n) if env.action_mask[i]]
        assert legal
        _obs, _reward, done, truncated, _info = env.step(
            (rng.choice(legal), rng.randrange(4))
        )
        assert not truncated


def test_env_reset_calls_start_episode_on_duck_typed_opponents():
    duck = _DuckOpponent()
    env = Seven523Env(seed=0, opponents=[FirstLegalBot(), duck])
    assert duck.starts == 0
    env.reset()
    assert duck.starts == 1
    _finish_episode(env)
    env.reset()
    assert duck.starts == 2


def test_env_reset_ignores_policies_without_start_episode():
    # Legacy policies (and the default random opponents) have no hook; reset
    # must not require one.
    env = Seven523Env(seed=0, opponents=[FirstLegalBot(), FirstLegalBot()])
    obs, _info = env.reset()
    assert obs.shape == (env.obs_dim,)
    run = Seven523Env(seed=1)
    run.reset()
    assert not hasattr(run._opponents[0], "start_episode")


def test_env_accepts_an_episode_mixture_opponent_end_to_end():
    # The new policy must be a drop-in env opponent: one member per episode
    # and no error at the auto-advance boundary.
    mixture = EpisodeMixturePolicy(
        [(0.5, FirstLegalBot(), "fixed"), (0.5, RandomBot(random.Random(0)), "random")],
        random.Random(0),
    )
    env = Seven523Env(seed=0, opponents=[FirstLegalBot(), mixture])
    env.reset()
    assert mixture.current_id in {"fixed", "random"}
    _finish_episode(env, action_seed=3)


def test_mixture_policy_per_decision_behaviour_is_unchanged():
    # Guard rail for the legacy class: it still draws every call (no member
    # freeze) and accepts two-tuples.
    low, high, _ = _members()
    legacy = MixturePolicy([(0.5, low), (0.5, high)], random.Random(0))
    picks = {legacy.act(object())[0] for _ in range(50)}
    assert picks == {0, 1}

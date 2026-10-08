"""League assembly and PFSP controller tests (training-side; torch required).

``build_league`` owns the roster / mixture construction that used to be inlined
in ``train()``; ``PfspController`` owns the cumulative win/draw/loss record and
the re-weighting cadence.  Torch lives in the optional ``train`` group.
"""
from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch")

from support import FirstLegalBot  # noqa: E402

from seven523.actions import nvec_for  # noqa: E402
from seven523.env import observation_dim  # noqa: E402
from seven523.league import (  # noqa: E402
    LeagueConfig,
    PfspController,
    build_league,
    parse_pool_member,
    pool_member_ids,
)
from seven523.networks import Agent, save_agent  # noqa: E402
from seven523.policies import (  # noqa: E402
    EpisodeMixturePolicy,
    MixturePolicy,
    RandomBot,
)
from seven523.rules import DEFAULT_RULES  # noqa: E402


def _agent(rules=DEFAULT_RULES, hidden=16):
    torch.manual_seed(0)
    return Agent(observation_dim(rules.num_players), nvec_for(rules), hidden=hidden)


def _config(opponent, **overrides):
    fields = {
        "opponent": opponent,
        "num_envs": 2,
        "num_players": DEFAULT_RULES.num_players,
        "seed": 7,
    }
    fields.update(overrides)
    return LeagueConfig(**fields)


def _build(config, agent=None):
    return build_league(
        config,
        rules=DEFAULT_RULES,
        agent=agent if agent is not None else _agent(),
        device="cpu",
    )


# -- build_league: scripted and self modes -----------------------------------


@pytest.mark.parametrize(
    ("opponent", "bot"),
    [("random", RandomBot)],
)
def test_build_league_scripted_rosters(opponent, bot):
    league = _build(_config(opponent))
    assert league.frozen is None
    assert league.member_ids == []
    assert league.episode_mixtures == [[], []]
    assert len(league.opponents) == 2
    for roster in league.opponents:
        assert len(roster) == DEFAULT_RULES.num_players
        assert all(isinstance(policy, bot) for policy in roster)


def test_build_league_self_freezes_a_weight_copy_and_keeps_mixtures_empty():
    agent = _agent()
    league = _build(_config("self", self_play_sample=True), agent=agent)
    assert league.frozen is not None
    assert league.frozen is not agent
    assert league.frozen.sample is True
    for key, value in agent.state_dict().items():
        assert torch.equal(league.frozen.agent.state_dict()[key], value), key
    assert league.member_ids == []
    assert league.episode_mixtures == [[], []]
    for roster in league.opponents:
        assert roster == [league.frozen] * DEFAULT_RULES.num_players


# -- build_league: mix and pool ----------------------------------------------


def test_build_league_mix_defaults_to_per_decision_mixtures():
    league = _build(_config("mix"))
    assert league.frozen is not None
    assert league.member_ids == []
    assert league.episode_mixtures == [[], []]
    for roster in league.opponents:
        assert all(isinstance(policy, MixturePolicy) for policy in roster)
        for policy in roster:
            assert [weight for weight, _ in policy.members] == [0.5, 0.5]
            assert policy.members[0][1] is league.frozen
            assert isinstance(policy.members[1][1], RandomBot)


def test_build_league_mix_with_pool_episode_populates_seat_mixtures():
    league = _build(_config("mix", pool_episode=True, mix_random_prob=0.25))
    assert league.frozen is not None
    for env_mixtures in league.episode_mixtures:
        assert len(env_mixtures) == DEFAULT_RULES.num_players - 1
        assert all(isinstance(policy, EpisodeMixturePolicy) for policy in env_mixtures)
        policy = env_mixtures[0]
        assert [weight for weight, _, _ in policy.members] == [0.75, 0.25]
        assert [member_id for _, _, member_id in policy.members] == ["self", "random"]
    for roster in league.opponents:
        assert all(isinstance(policy, EpisodeMixturePolicy) for policy in roster)


def test_build_league_pool_member_ids_weights_and_frozen_self():
    league = _build(
        _config(
            "pool",
            pool_member=("2@random", "1@self"),
            pool_episode=True,
        )
    )
    assert league.member_ids == ["random", "self"]
    assert league.frozen is not None
    for env_mixtures in league.episode_mixtures:
        policy = env_mixtures[0]
        assert [weight for weight, _, _ in policy.members] == [2.0, 1.0]
        assert [member_id for _, _, member_id in policy.members] == league.member_ids
        assert isinstance(policy.members[0][1], RandomBot)
        assert policy.members[1][1] is league.frozen


def test_build_league_pool_members_sample_by_default(tmp_path):
    path = tmp_path / "member.pt"
    save_agent(path, _agent())
    spec = f"1@ckpt:{path}"
    default = _build(_config("pool", pool_member=(spec,), pool_episode=True))
    member = default.episode_mixtures[0][0].members[0][1]
    assert member.sample is True
    greedy = _build(
        _config(
            "pool",
            pool_member=(spec,),
            pool_episode=True,
            pool_sample=False,
        )
    )
    member = greedy.episode_mixtures[0][0].members[0][1]
    assert member.sample is False


def test_build_league_pool_self_member_follows_pool_sample():
    sampling = _build(_config("pool", pool_member=("1@self",), pool_episode=True))
    assert sampling.frozen is not None and sampling.frozen.sample is True
    greedy = _build(
        _config(
            "pool",
            pool_member=("1@self",),
            pool_episode=True,
            pool_sample=False,
        )
    )
    assert greedy.frozen is not None and greedy.frozen.sample is False
    # ``self`` mode keeps its own flag (default argmax).
    own = _build(_config("self"))
    assert own.frozen is not None and own.frozen.sample is False


def test_build_league_pool_without_pool_episode_stays_per_decision():
    league = _build(_config("pool", pool_member=("random",)))
    assert league.member_ids == ["random"]
    assert league.episode_mixtures == [[], []]
    for roster in league.opponents:
        assert all(isinstance(policy, MixturePolicy) for policy in roster)


def test_build_league_pool_requires_a_member():
    with pytest.raises(SystemExit, match="--pool-member"):
        _build(_config("pool"))


def test_build_league_seed_rng_overrides_the_base_seed():
    draw_seed = random.Random(0).randrange(1 << 32)
    explicit = _build(
        _config(
            "pool",
            pool_member=("random", "self"),
            pool_episode=True,
            seed=draw_seed,
        )
    )
    via_rng = build_league(
        _config(
            "pool", pool_member=("random", "self"), pool_episode=True, seed=0
        ),
        rules=DEFAULT_RULES,
        agent=_agent(),
        device="cpu",
        seed_rng=random.Random(0),
    )
    explicit_draws = [p.start_episode() for p in explicit.episode_mixtures[0]]
    rng_draws = [p.start_episode() for p in via_rng.episode_mixtures[0]]
    assert explicit_draws == rng_draws


def test_pool_member_ids_disambiguate_repeated_specs():
    assert pool_member_ids(["random", "random", "self", "random"]) == [
        "random#1",
        "random#2",
        "self",
        "random#3",
    ]
    assert parse_pool_member("2.5@ckpt:runs/x/agent.pt") == (
        2.5,
        "ckpt:runs/x/agent.pt",
    )


# -- PfspController ----------------------------------------------------------


def test_pfsp_controller_records_outcomes_and_win_rate():
    pfsp = PfspController(
        ["a", "b"], prior=0.0, epsilon=0.0, uniform_mix=0.0, every=2
    )
    assert pfsp.records == {"a": [0, 0, 0], "b": [0, 0, 0]}
    assert pfsp.episodes == 0 and pfsp.updates == 0

    pfsp.record("a", 1)
    assert pfsp.records["a"] == [1, 0, 0]
    assert pfsp.win_rate("a") == 1.0
    assert pfsp.episodes == 1
    assert pfsp.maybe_update() is None

    pfsp.record("b", 0)
    assert pfsp.records["b"] == [0, 1, 0]
    assert pfsp.win_rate("b") == 0.5
    pfsp.record("b", -1, episode=False)
    assert pfsp.records["b"] == [0, 1, 1]
    assert pfsp.episodes == 2

    weights = pfsp.maybe_update()
    assert weights is not None
    assert pfsp.updates == 1
    assert set(weights) == {"a", "b"}
    # a is the harder opponent (learner won every time -> PFSP down-weights it)
    assert weights["a"] < weights["b"]
    assert sum(weights.values()) == pytest.approx(1.0)
    # no further episodes recorded -> no further update
    assert pfsp.maybe_update() is None
    assert pfsp.updates == 1


def test_pfsp_controller_win_rate_shrinks_toward_half_with_the_prior():
    pfsp = PfspController(
        ["a"], prior=10.0, epsilon=0.0, uniform_mix=0.0, every=1
    )
    assert pfsp.win_rate("a") == 0.5
    pfsp.record("a", 1)
    assert pfsp.win_rate("a") == pytest.approx((0.5 * 10 + 1) / 11)
    assert pfsp.maybe_update() is not None


def test_build_league_self_freezes_the_full_event_config():
    """The frozen opponent must carry the learner's event layout (step 7).

    A naive rebuild would drop the event encoder and the strict
    ``load_state_dict`` refresh would raise; ``deepcopy`` also keeps the
    relative-seat reconstruction keyed to the opponent's own view.
    """
    from seven523.env import Seven523Env
    from seven523.networks import history_layout

    torch.manual_seed(0)
    agent = Agent(
        observation_dim(DEFAULT_RULES.num_players),
        nvec_for(DEFAULT_RULES),
        hidden=16,
        event_len=8,
        event_hidden=4,
        event_noisy=True,
        event_seat="none",
    )
    league = _build(_config("self"), agent=agent)
    assert league.frozen is not None
    assert history_layout(league.frozen.agent) == history_layout(agent)
    env = Seven523Env(seed=0, opponents=[FirstLegalBot(), FirstLegalBot()])
    env.reset()
    action, _suit = league.frozen.act(env.view())
    assert env.action_mask[action]
    # Refreshing in place publishes the learner's new weights.
    with torch.no_grad():
        agent.actor.weight[0, 0] += 1.0
    league.frozen.agent.load_state_dict(agent.state_dict())
    assert torch.equal(league.frozen.agent.actor.weight, agent.actor.weight)
    assert torch.equal(
        league.frozen.agent.event_encoder.mlp[0].weight,
        agent.event_encoder.mlp[0].weight,
    )

"""PPO training / evaluation smoke tests.

Torch lives in the optional ``train`` dependency group
(``uv sync --group train``); without it this whole module is skipped.
"""
import pytest

torch = pytest.importorskip("torch")

from seven523.eval import evaluate  # noqa: E402
from seven523.networks import (  # noqa: E402
    Agent,
    CategoricalMasked,
    NeuralPolicy,
    load_agent,
    save_agent,
    warm_start_into,
)
from seven523.rules import DEFAULT_RULES  # noqa: E402
from seven523.train import make_env, parse_args, parse_pool_member, train  # noqa: E402

OBS_DIM = 161
NVEC = [134, 4]


def test_masked_categorical_never_samples_illegal_actions():
    torch.manual_seed(0)
    logits = torch.randn(64, 10)
    mask = torch.zeros(64, 10, dtype=torch.bool)
    mask[:, :3] = True
    dist = CategoricalMasked(logits=logits, masks=mask)
    samples = dist.sample((200,))
    assert bool((samples < 3).all())
    assert bool(torch.isfinite(dist.entropy()).all())
    assert bool(torch.isfinite(dist.log_prob(samples)).all())


def test_masked_categorical_all_false_mask_is_finite():
    torch.manual_seed(0)
    dist = CategoricalMasked(
        logits=torch.randn(4, 5), masks=torch.zeros(4, 5, dtype=torch.bool)
    )
    assert bool(torch.isfinite(dist.sample()).all())
    assert bool(torch.isfinite(dist.entropy()).all())


def test_agent_forward_shapes_and_replay():
    torch.manual_seed(0)
    agent = Agent(OBS_DIM, NVEC, hidden=16)
    obs = torch.zeros(4, OBS_DIM)
    mask = torch.ones(4, sum(NVEC), dtype=torch.bool)
    mask[:, NVEC[0] - 1] = False  # PASS illegal here, exactly like a leading trick

    action, logprob, entropy, value = agent.get_action_and_value(obs, mask)
    assert action.shape == (4, 2)
    assert logprob.shape == (4,)
    assert entropy.shape == (4,)
    assert value.shape == (4, 1)
    assert not bool((action[:, 0] == NVEC[0] - 1).any())
    assert bool((action[:, 1] < NVEC[1]).all())

    replayed, newlogprob, _, _ = agent.get_action_and_value(obs, mask, action.T)
    assert torch.equal(replayed, action)
    assert torch.allclose(newlogprob, logprob)


def test_activation_variants_forward_and_checkpoint_round_trip(tmp_path):
    torch.manual_seed(0)
    obs = torch.zeros(2, OBS_DIM)
    mask = torch.ones(2, sum(NVEC), dtype=torch.bool)
    for activation in ("relu", "tanh", "gelu", "silu"):
        agent = Agent(OBS_DIM, NVEC, hidden=16, activation=activation)
        action, logprob, entropy, value = agent.get_action_and_value(obs, mask)
        assert action.shape == (2, 2)
        assert bool(torch.isfinite(logprob).all())
        assert bool(torch.isfinite(entropy).all())
        assert bool(torch.isfinite(value).all())
        path = tmp_path / f"{activation}.pt"
        save_agent(path, agent)
        loaded, _ = load_agent(path)
        assert loaded.activation == activation
        assert torch.allclose(loaded.get_value(obs), agent.get_value(obs))
    legacy = tmp_path / "legacy.pt"
    torch.save(
        {
            "model": Agent(OBS_DIM, NVEC, hidden=16).state_dict(),
            "obs_dim": OBS_DIM,
            "nvec": NVEC,
            "hidden": 16,
        },
        legacy,
    )
    with pytest.raises(ValueError, match="obs_version"):
        load_agent(legacy)
    with pytest.raises(ValueError):
        Agent(OBS_DIM, NVEC, hidden=16, activation="swish")


def test_warm_start_copies_trunk_and_template_head():
    torch.manual_seed(0)
    old = Agent(OBS_DIM, [134], hidden=16)
    new = Agent(OBS_DIM, NVEC, hidden=16)
    copied = warm_start_into(new, old)
    assert "network.0.weight" in copied
    assert "actor.weight[:134]" in copied
    assert torch.equal(new.network[0].weight, old.network[0].weight)
    assert torch.equal(new.actor.weight[:134], old.actor.weight)
    assert torch.equal(new.critic.weight, old.critic.weight)


def test_checkpoint_round_trip(tmp_path):
    torch.manual_seed(0)
    agent = Agent(OBS_DIM, NVEC, hidden=16)
    path = tmp_path / "agent.pt"
    save_agent(path, agent, extra={"global_step": 7})
    loaded, extra = load_agent(path)
    assert extra == {"global_step": 7}
    obs = torch.rand(2, OBS_DIM)
    assert torch.allclose(agent.get_value(obs), loaded.get_value(obs))


def test_neural_policy_acts_legally():
    torch.manual_seed(0)
    from seven523.env import Seven523Env
    from seven523.policies import make_scripted_policies

    policy = NeuralPolicy(Agent(OBS_DIM, NVEC, hidden=16), DEFAULT_RULES)
    env = Seven523Env(seed=0, opponents=make_scripted_policies("greedy", seed=0))
    env.reset()
    for _ in range(20):
        if env.state.done:
            env.reset()
        view = env.game.view(env.state, env.learner)
        action = policy.act(view)
        assert action[0] in range(env.action_space_n)
        assert env.action_mask[action[0]]
        env.step(action)


def test_neural_policy_accepts_old_single_head_checkpoints(tmp_path):
    from seven523.env import Seven523Env
    from seven523.policies import make_scripted_policies

    torch.manual_seed(0)
    path = tmp_path / "old.pt"
    save_agent(path, Agent(OBS_DIM, [134], hidden=16))
    loaded, _ = load_agent(path)
    policy = NeuralPolicy(loaded, DEFAULT_RULES)
    env = Seven523Env(seed=0, opponents=make_scripted_policies("greedy", seed=0))
    env.reset()
    view = env.game.view(env.state, env.learner)
    action = policy.act(view)
    assert action[1] is None
    assert env.action_mask[action[0]]
    env.step(action)


def test_train_smoke_with_eval(tmp_path):
    args = parse_args(
        [
            "--cuda", "False",
            "--seed", "0",
            "--exp-name", "test",
            "--run-dir", str(tmp_path),
            "--total-timesteps", "512",
            "--num-envs", "2",
            "--num-steps", "64",
            "--num-minibatches", "2",
            "--update-epochs", "1",
            "--checkpoint-interval", "2",
            "--eval-interval", "2",
            "--eval-episodes", "3",
            "--log-interval", "0",
        ]
    )
    run_dir = train(args)
    num_updates = args.total_timesteps // args.batch_size
    assert (run_dir / "agent.pt").exists()
    assert (run_dir / "checkpoint.pt").exists()
    assert (run_dir / "args.json").exists()
    rows = (run_dir / "metrics.csv").read_text().strip().splitlines()
    assert len(rows) == num_updates + 1
    assert rows[-1].split(",")[0] == str(num_updates * args.batch_size)
    header = rows[0].split(",")
    last = dict(zip(header, rows[-1].split(",")))
    assert last["eval_return"] != ""
    assert float(last["episodes"]) > 0


def test_train_self_play_smoke(tmp_path):
    args = parse_args(
        [
            "--cuda", "False",
            "--seed", "1",
            "--exp-name", "selfplay",
            "--run-dir", str(tmp_path),
            "--total-timesteps", "256",
            "--num-envs", "2",
            "--num-steps", "64",
            "--num-minibatches", "2",
            "--update-epochs", "1",
            "--opponent", "self",
            "--self-play-refresh", "1",
            "--checkpoint-interval", "0",
            "--log-interval", "0",
        ]
    )
    run_dir = train(args)
    assert (run_dir / "agent.pt").exists()
    assert not (run_dir / "checkpoint.pt").exists()


def test_train_uses_same_step_autoreset(tmp_path, monkeypatch):
    """Regression: NEXT_STEP injects discarded-action rows into the buffer.

    gymnasium's default ``AutoresetMode.NEXT_STEP`` swallows the action on the
    step after a terminal state while still occupying a rollout row (a ~4%
    chimera-transition rate on real runs, see docs/experiments/ppo-alignment-audit.md).
    With ``SAME_STEP`` every buffer row is backed by exactly one real
    ``env.step`` call, restoring the gym 0.21 reference semantics.
    """
    from seven523.env import Seven523Env

    real_step = Seven523Env.step
    calls = {"n": 0}

    def counting_step(self, action):
        calls["n"] += 1
        return real_step(self, action)

    monkeypatch.setattr(Seven523Env, "step", counting_step)
    args = parse_args(
        [
            "--cuda", "False",
            "--seed", "0",
            "--exp-name", "autoreset",
            "--run-dir", str(tmp_path),
            "--total-timesteps", "256",
            "--num-envs", "2",
            "--num-steps", "64",
            "--num-minibatches", "2",
            "--update-epochs", "1",
            "--checkpoint-interval", "0",
            "--log-interval", "0",
        ]
    )
    train(args)
    num_updates = args.total_timesteps // args.batch_size
    assert calls["n"] == args.num_steps * args.num_envs * num_updates


def test_parse_reward_shaping_choices_and_default():
    assert parse_args([]).reward_shaping == "terminal"
    for mode in ("terminal", "trick_diff", "win", "trick_diff_win"):
        assert parse_args(["--reward-shaping", mode]).reward_shaping == mode
    with pytest.raises(SystemExit):
        parse_args(["--reward-shaping", "shaped"])


def test_make_env_passes_reward_shaping_through():
    from seven523.policies import make_scripted_policies

    opponents = make_scripted_policies("random", DEFAULT_RULES, seed=0)
    default_env = make_env(DEFAULT_RULES, 0, opponents, 0, 0)()
    assert default_env.unwrapped.reward_shaping == "terminal"
    shaped_env = make_env(
        DEFAULT_RULES, 0, opponents, 0, 0, reward_shaping="trick_diff_win"
    )()
    assert shaped_env.unwrapped.reward_shaping == "trick_diff_win"


def test_parse_pool_member():
    assert parse_pool_member("greedy") == (1.0, "greedy")
    assert parse_pool_member("2.5@ckpt:runs/x/agent.pt") == (2.5, "ckpt:runs/x/agent.pt")
    with pytest.raises(ValueError):
        parse_pool_member("-1@greedy")


def test_train_mix_opponent_smoke(tmp_path):
    args = parse_args(
        [
            "--cuda", "False",
            "--seed", "1",
            "--exp-name", "mix",
            "--run-dir", str(tmp_path),
            "--total-timesteps", "256",
            "--num-envs", "2",
            "--num-steps", "64",
            "--num-minibatches", "2",
            "--update-epochs", "1",
            "--opponent", "mix",
            "--mix-greedy-prob", "0.5",
            "--self-play-refresh", "1",
            "--checkpoint-interval", "0",
            "--log-interval", "0",
        ]
    )
    run_dir = train(args)
    assert (run_dir / "agent.pt").exists()


def test_train_pool_opponent_smoke(tmp_path):
    args = parse_args(
        [
            "--cuda", "False",
            "--seed", "1",
            "--exp-name", "pool",
            "--run-dir", str(tmp_path),
            "--total-timesteps", "256",
            "--num-envs", "2",
            "--num-steps", "64",
            "--num-minibatches", "2",
            "--update-epochs", "1",
            "--opponent", "pool",
            "--pool-member", "2@greedy",
            "--pool-member", "1@random",
            "--pool-member", "1@self",
            "--self-play-refresh", "1",
            "--checkpoint-interval", "0",
            "--log-interval", "0",
        ]
    )
    run_dir = train(args)
    assert (run_dir / "agent.pt").exists()


def test_pool_episode_and_pfsp_cli_defaults_are_off():
    args = parse_args([])
    assert args.pool_episode is False
    assert args.pfsp is False
    assert args.pfsp_every == 100
    assert args.pfsp_uniform_mix == 0.5
    assert args.pfsp_epsilon == 0.02
    assert args.pfsp_prior == 10.0


def _pool_run_args(tmp_path, *extra):
    return parse_args(
        [
            "--cuda", "False",
            "--seed", "1",
            "--exp-name", "pool-episode",
            "--run-dir", str(tmp_path),
            "--total-timesteps", "512",
            "--num-envs", "2",
            "--num-steps", "64",
            "--num-minibatches", "2",
            "--update-epochs", "1",
            "--opponent", "pool",
            "--pool-member", "1@greedy",
            "--pool-member", "1@random",
            "--self-play-refresh", "1",
            "--checkpoint-interval", "0",
            "--log-interval", "0",
            *extra,
        ]
    )


def test_pool_episode_off_never_touches_the_new_policy(tmp_path, monkeypatch):
    """The default pool path must stay bit-for-bit the legacy MixturePolicy."""
    from seven523.policies import EpisodeMixturePolicy

    calls = {"n": 0}
    real_start = EpisodeMixturePolicy.start_episode

    def counting_start(self):
        calls["n"] += 1
        return real_start(self)

    monkeypatch.setattr(EpisodeMixturePolicy, "start_episode", counting_start)
    args = _pool_run_args(tmp_path)
    assert args.pool_episode is False and args.pfsp is False
    run_dir = train(args)
    assert calls["n"] == 0
    assert not (run_dir / "pfsp_weights.csv").exists()


def test_train_pool_episode_and_pfsp_smoke(tmp_path):
    run_dir = train(
        _pool_run_args(
            tmp_path,
            "--pool-member", "1@self",
            "--pool-episode",
            "--pfsp",
            "--pfsp-every", "3",
        )
    )
    assert (run_dir / "agent.pt").exists()
    rows = (run_dir / "pfsp_weights.csv").read_text().strip().splitlines()
    header, data = rows[0], rows[1:]
    assert header == (
        "global_step,episodes,member_id,weight,win_rate,wins,draws,losses"
    )
    assert data, "PFSP never logged an update"
    # every update logs one row per member and the weights form a distribution
    by_episode: dict[int, list[tuple[str, float]]] = {}
    for row in data:
        step, episodes, member_id, weight, _wr, _w, _d, _l = row.split(",")
        by_episode.setdefault(int(episodes), []).append((member_id, float(weight)))
        assert 0.0 <= float(weight) <= 1.0
    for entries in by_episode.values():
        assert len({member for member, _ in entries}) == 3
        assert sum(weight for _, weight in entries) == pytest.approx(1.0)


def test_pfsp_needs_pool_episode(tmp_path):
    with pytest.raises(SystemExit):
        train(parse_args(["--pfsp"]))
    with pytest.raises(SystemExit):
        train(_pool_run_args(tmp_path, "--pfsp"))
    with pytest.raises(SystemExit):
        train(parse_args(["--pool-episode"]))


def test_evaluate_smoke(tmp_path):
    torch.manual_seed(0)
    agent = Agent(OBS_DIM, NVEC, hidden=16)
    policy = NeuralPolicy(agent, DEFAULT_RULES)
    metrics = evaluate(policy, episodes=5, opponent="random", seed=0)
    assert metrics["episodes"] == 5.0
    assert 0.0 <= metrics["learner_score"] <= 100.0
    assert 0.0 <= metrics["win_rate"] <= 1.0
    assert abs(
        metrics["win_rate"] + metrics["draw_rate"] + metrics["loss_rate"] - 1.0
    ) < 1e-9
    assert metrics["illegal_rate"] == 0.0
    assert metrics["mean_return"] == pytest.approx(
        metrics["score_diff"] / 100.0, abs=1e-6
    )


def _tiny_args(tmp_path, *extra):
    return parse_args(
        [
            "--cuda", "False",
            "--seed", "0",
            "--exp-name", "test",
            "--run-dir", str(tmp_path),
            "--total-timesteps", "512",
            "--num-envs", "2",
            "--num-steps", "64",
            "--num-minibatches", "2",
            "--update-epochs", "1",
            "--log-interval", "0",
            *extra,
        ]
    )


def test_train_writes_tensorboard_scalars(tmp_path):
    run_dir = train(_tiny_args(tmp_path, "--tensorboard", "True"))
    assert list((run_dir / "tb").glob("events.out.tfevents*")), (
        "no TensorBoard event file was written"
    )

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    accumulator = EventAccumulator(str(run_dir / "tb"))
    accumulator.Reload()
    tags = set(accumulator.Tags()["scalars"])
    assert {"charts/SPS", "losses/value_loss", "losses/entropy"} <= tags
    assert {"charts/episodic_return", "charts/episodic_length"} <= tags


def test_tensorboard_can_be_disabled(tmp_path):
    run_dir = train(_tiny_args(tmp_path, "--tensorboard", "False"))
    assert not list((run_dir / "tb").glob("events.out.tfevents*"))


# -- observation layout plumbing ---------------------------------------------


def test_make_env_uses_the_v5_layout():
    from seven523.policies import make_scripted_policies

    opponents = make_scripted_policies("random", DEFAULT_RULES, seed=0)
    env = make_env(DEFAULT_RULES, 0, opponents, 0, 0)()
    assert env.observation_space.shape == (161,)


def test_train_agent_uses_the_v5_layout(tmp_path):
    run_dir = train(_tiny_args(tmp_path, "--checkpoint-interval", "0"))
    agent, extra = load_agent(run_dir / "agent.pt")
    assert agent.obs_dim == 161
    assert "obs_version" not in extra["args"]
    payload = torch.load(run_dir / "agent.pt", map_location="cpu", weights_only=True)
    assert payload["obs_version"] == 5


# -- independent actor/critic towers (T6 / direction F) ----------------------


def test_parse_arch_default_and_choices():
    assert parse_args([]).arch == "shared"
    assert parse_args(["--arch", "towers"]).arch == "towers"
    with pytest.raises(SystemExit):
        parse_args(["--arch", "twotowers"])


def test_train_towers_smoke(tmp_path):
    run_dir = train(
        _tiny_args(tmp_path, "--arch", "towers", "--checkpoint-interval", "0")
    )
    agent, extra = load_agent(run_dir / "agent.pt")
    assert agent.arch == "towers"
    assert agent.obs_dim == 161
    assert extra["args"]["arch"] == "towers"


def test_train_towers_self_play_smoke(tmp_path):
    run_dir = train(
        _tiny_args(
            tmp_path,
            "--arch",
            "towers",
            "--opponent",
            "self",
            "--self-play-refresh",
            "1",
            "--checkpoint-interval",
            "0",
        )
    )
    assert load_agent(run_dir / "agent.pt")[0].arch == "towers"


def test_train_warm_starts_a_same_layout_checkpoint(tmp_path):
    torch.manual_seed(0)
    old = Agent(OBS_DIM, NVEC, hidden=16)
    path = tmp_path / "old.pt"
    save_agent(path, old)
    run_dir = train(
        _tiny_args(
            tmp_path,
            "--hidden-size",
            "16",
            "--learning-rate",
            "0",
            "--load-checkpoint",
            str(path),
            "--checkpoint-interval",
            "0",
        )
    )
    agent, _ = load_agent(run_dir / "agent.pt")
    assert agent.obs_dim == OBS_DIM
    assert torch.equal(agent.network[0].weight, old.network[0].weight)
    assert torch.equal(agent.actor.weight, old.actor.weight)
    assert torch.equal(agent.critic.weight, old.critic.weight)
    obs = torch.rand(4, OBS_DIM)
    assert torch.equal(agent.get_value(obs), old.get_value(obs))


def test_train_warm_starts_a_same_layout_checkpoint_into_towers(tmp_path):
    torch.manual_seed(0)
    old = Agent(OBS_DIM, NVEC, hidden=16)
    path = tmp_path / "old.pt"
    save_agent(path, old)
    run_dir = train(
        _tiny_args(
            tmp_path,
            "--arch",
            "towers",
            "--hidden-size",
            "16",
            "--learning-rate",
            "0",
            "--load-checkpoint",
            str(path),
            "--checkpoint-interval",
            "0",
        )
    )
    agent, _ = load_agent(run_dir / "agent.pt")
    assert agent.arch == "towers"
    assert agent.obs_dim == OBS_DIM
    for tower in (agent.actor_network, agent.critic_network):
        assert torch.equal(tower[0].weight, old.network[0].weight)
        assert torch.equal(tower[0].bias, old.network[0].bias)
    obs = torch.rand(4, OBS_DIM)
    assert torch.equal(agent.get_value(obs), old.get_value(obs))
    assert torch.equal(agent.policy_logits(obs), old.policy_logits(obs))


def test_train_rejects_a_different_layout_checkpoint(tmp_path):
    torch.manual_seed(0)
    old = Agent(191, NVEC, hidden=16)
    path = tmp_path / "foreign.pt"
    save_agent(path, old)
    with pytest.raises(SystemExit) as excinfo:
        train(
            _tiny_args(
                tmp_path,
                "--hidden-size",
                "16",
                "--learning-rate",
                "0",
                "--load-checkpoint",
                str(path),
                "--checkpoint-interval",
                "0",
            )
        )
    message = str(excinfo.value)
    assert "cannot warm-start" in message
    assert "191" in message and "161" in message


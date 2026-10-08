"""PPO training / evaluation smoke tests.

Torch lives in the optional ``train`` dependency group
(``uv sync --group train``); without it this whole module is skipped.
"""
import itertools
import json
import math

import pytest

torch = pytest.importorskip("torch")

from seven523.eval import evaluate  # noqa: E402
from seven523.networks import (  # noqa: E402
    Agent,
    CategoricalMasked,
    NeuralPolicy,
    load_agent,
    save_agent,
)
from seven523.rules import DEFAULT_RULES  # noqa: E402
from seven523.train import (  # noqa: E402
    _episode_outcome_labels,
    annealed_lr,
    lr_scale,
    make_env,
    parse_args,
    parse_pool_member,
    train,
)

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
    env = Seven523Env(seed=0, opponents=make_scripted_policies("random", seed=0))
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
    env = Seven523Env(seed=0, opponents=make_scripted_policies("random", seed=0))
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
            "--total-timesteps", "256",
            "--num-envs", "2",
            "--num-steps", "64",
            "--num-minibatches", "2",
            "--update-epochs", "1",
            "--checkpoint-interval", "2",
            "--eval-interval", "2",
            "--eval-episodes", "1",
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
            "--total-timesteps", "128",
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
    # Operator decision 2026-10-07 (ADR-0016): arcsin + 50-point jump is the
    # CLI default; historical runs reproduce with --reward-shaping terminal.
    assert parse_args([]).reward_shaping == "arcsin"
    assert parse_args([]).win_jump == 1.0
    assert parse_args([]).reward_cap is None
    for mode in (
        "terminal",
        "trick_diff",
        "win",
        "trick_diff_win",
        "terminal_win",
        "saturate",
        "arcsin",
    ):
        assert parse_args(["--reward-shaping", mode]).reward_shaping == mode
    assert parse_args(["--win-jump", "0.25"]).win_jump == 0.25
    assert parse_args([]).arcsin_mix == 0.5
    assert parse_args(["--arcsin-mix", "0.25"]).arcsin_mix == 0.25
    assert parse_args([]).actor_out_std == 0.01
    assert parse_args(["--actor-out-std", "0.1"]).actor_out_std == 0.1
    assert parse_args(["--reward-cap", "0.2"]).reward_cap == 0.2
    assert parse_args([]).lr_floor == pytest.approx(1e-5)
    with pytest.raises(SystemExit):
        parse_args(["--reward-shaping", "shaped"])


def test_lr_scale_linear_matches_the_historical_formula():
    for update, num_updates in ((1, 100), (2, 100), (50, 100), (100, 100)):
        assert lr_scale("linear", update, num_updates) == 1.0 - (
            update - 1.0
        ) / num_updates


def test_lr_scale_cosine_endpoints_and_monotone_decrease():
    assert lr_scale("cosine", 1, 100) == pytest.approx(1.0)
    # The phase matches the historical linear schedule ((update - 1) / N), so
    # the last update lands one phase step short of zero, not exactly at it.
    assert lr_scale("cosine", 100, 100) == pytest.approx(
        0.5 * (1.0 + math.cos(math.pi * 99 / 100))
    )
    assert 0.0 < lr_scale("cosine", 100, 100) < 5e-4
    values = [lr_scale("cosine", update, 100) for update in range(1, 101)]
    assert all(b <= a + 1e-12 for a, b in itertools.pairwise(values))
    with pytest.raises(ValueError):
        lr_scale("unknown", 1, 10)


def test_annealed_lr_floor_clamps_the_tail_and_zero_restores_history():
    lr = 2.5e-4
    assert annealed_lr("linear", 1, 100, lr) == pytest.approx(lr)
    assert annealed_lr("linear", 100, 100, lr) == pytest.approx(1e-5)
    historical = lr * (1.0 - 99 / 100)
    assert annealed_lr("linear", 100, 100, lr, floor=0.0) == pytest.approx(historical)
    # The floor never raises the LR above the requested peak.
    assert annealed_lr("linear", 100, 100, 0.0) == 0.0
    assert annealed_lr("linear", 1, 100, 1e-6) == pytest.approx(1e-6)
    assert annealed_lr("cosine", 100, 100, lr, floor=0.0) > 0.0
    values = [annealed_lr("linear", update, 100, lr) for update in range(1, 101)]
    assert all(b <= a + 1e-12 for a, b in itertools.pairwise(values))
    assert min(values) >= 1e-5


def test_train_snapshot_steps_writes_step_boundaries(tmp_path):
    run_dir = train(
        _tiny_args(
            tmp_path,
            "--total-timesteps", "512",
            "--num-envs", "2",
            "--num-steps", "64",
            "--snapshot-steps", "256",
            "--checkpoint-interval", "0",
        )
    )
    snaps = sorted(
        p.name for p in (run_dir / "snapshots").glob("checkpoint_step*.pt")
    )
    assert snaps == [
        "checkpoint_step0000256.pt",
        "checkpoint_step0000512.pt",
    ]


def test_optimizer_schedule_and_snapshot_flags():
    defaults = parse_args([])
    assert defaults.optimizer == "adam"
    assert defaults.weight_decay == 0.0
    assert defaults.lr_schedule == "linear"
    assert defaults.snapshot_interval == 0
    assert defaults.snapshot_steps == 0
    args = parse_args([
        "--optimizer", "adamw",
        "--weight-decay", "0.01",
        "--lr-schedule", "cosine",
        "--snapshot-interval", "50",
        "--snapshot-steps", "1000000",
    ])
    assert args.optimizer == "adamw"
    assert args.weight_decay == 0.01
    assert args.lr_schedule == "cosine"
    assert args.snapshot_interval == 50
    assert args.snapshot_steps == 1000000
    with pytest.raises(SystemExit):
        parse_args(["--weight-decay", "0.01"])
    with pytest.raises(SystemExit):
        parse_args(["--weight-decay", "-0.01"])
    with pytest.raises(SystemExit):
        parse_args(["--lr-schedule", "warmup"])


def test_make_env_passes_reward_shaping_through():
    from seven523.policies import make_scripted_policies

    opponents = make_scripted_policies("random", DEFAULT_RULES, seed=0)
    default_env = make_env(DEFAULT_RULES, 0, opponents, 0, 0)()
    assert default_env.unwrapped.reward_shaping == "arcsin"
    assert default_env.unwrapped.arcsin_mix == 0.5
    shaped_env = make_env(
        DEFAULT_RULES, 0, opponents, 0, 0, reward_shaping="trick_diff_win"
    )()
    assert shaped_env.unwrapped.reward_shaping == "trick_diff_win"
    jumped_env = make_env(
        DEFAULT_RULES,
        0,
        opponents,
        0,
        0,
        reward_shaping="terminal_win",
        win_jump=0.25,
    )()
    assert jumped_env.unwrapped.reward_shaping == "terminal_win"
    assert jumped_env.unwrapped.win_jump == 0.25
    arcsin_env = make_env(
        DEFAULT_RULES,
        0,
        opponents,
        0,
        0,
        reward_shaping="arcsin",
        win_jump=0.5,
        arcsin_mix=0.25,
    )()
    assert arcsin_env.unwrapped.reward_shaping == "arcsin"
    assert arcsin_env.unwrapped.arcsin_mix == 0.25
    saturated_env = make_env(
        DEFAULT_RULES,
        0,
        opponents,
        0,
        0,
        reward_shaping="saturate",
        reward_cap=0.2,
    )()
    assert saturated_env.unwrapped.reward_shaping == "saturate"
    assert saturated_env.unwrapped.reward_cap == 0.2


def test_train_terminal_win_smoke(tmp_path):
    args = parse_args(
        [
            "--cuda", "False",
            "--seed", "0",
            "--exp-name", "jump",
            "--run-dir", str(tmp_path),
            "--total-timesteps", "128",
            "--num-envs", "2",
            "--num-steps", "64",
            "--num-minibatches", "2",
            "--update-epochs", "1",
            "--reward-shaping", "terminal_win",
            "--win-jump", "0.25",
            "--checkpoint-interval", "0",
            "--log-interval", "0",
        ]
    )
    run_dir = train(args)
    assert (run_dir / "agent.pt").exists()
    saved = json.loads((run_dir / "args.json").read_text())
    assert saved["reward_shaping"] == "terminal_win"
    assert saved["win_jump"] == 0.25


def test_parse_vf_outcome_flags():
    defaults = parse_args([])
    assert defaults.vf_outcome is False
    assert defaults.outcome_coef == 1.0
    args = parse_args(["--vf-outcome", "true", "--outcome-coef", "0.25"])
    assert args.vf_outcome is True
    assert args.outcome_coef == 0.25


def test_episode_outcome_labels_discount_back_to_the_episode_start():
    labels = _episode_outcome_labels(1, step=4, start=2, gamma=0.5)
    assert torch.equal(labels, torch.tensor([0.25, 0.5, 1.0]))
    losses = _episode_outcome_labels(-1, step=2, start=0, gamma=0.9)
    assert torch.allclose(
        losses, -torch.tensor([0.9**2, 0.9, 1.0])
    )
    # A tie is a zero target everywhere.
    tie = _episode_outcome_labels(0, step=3, start=0, gamma=0.99)
    assert torch.equal(tie, torch.zeros(4))


def test_train_vf_outcome_smoke(tmp_path):
    args = parse_args(
        [
            "--cuda", "False",
            "--seed", "0",
            "--exp-name", "vf_outcome",
            "--run-dir", str(tmp_path),
            "--total-timesteps", "256",
            "--num-envs", "2",
            "--num-steps", "64",
            "--num-minibatches", "2",
            "--update-epochs", "1",
            "--vf-outcome", "true",
            "--checkpoint-interval", "0",
            "--log-interval", "0",
        ]
    )
    run_dir = train(args)
    saved = json.loads((run_dir / "args.json").read_text())
    assert saved["vf_outcome"] is True and saved["outcome_coef"] == 1.0
    loaded, _ = load_agent(run_dir / "agent.pt")
    assert loaded.vf_outcome is True
    rows = (run_dir / "metrics.csv").read_text().strip().splitlines()
    header = rows[0].split(",")
    assert "outcome_loss" in header and "outcome_accuracy" in header
    assert all(
        dict(zip(header, row.split(",")))["outcome_loss"] != ""
        for row in rows[1:]
    )
    accuracies = [
        float(dict(zip(header, row.split(",")))["outcome_accuracy"])
        for row in rows[1:]
    ]
    finite = [value for value in accuracies if math.isfinite(value)]
    assert finite and all(0.0 <= value <= 1.0 for value in finite)


def test_train_vf_outcome_buffers_are_fresh_per_rollout(tmp_path, monkeypatch):
    """Regression: a reused jump/target buffer leaked stale values.

    Under ``arcsin`` both decomposed reward streams are terminal-only, so a
    nonzero reward at a non-terminal step can only come from a cell that was
    left over from a previous rollout.
    """
    from seven523 import train as train_module

    calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    real_gae = train_module.compute_gae

    def recording_gae(rewards, values, dones, next_value, next_done, **kwargs):
        calls.append(
            (
                rewards.detach().clone(),
                dones.detach().clone(),
                next_done.detach().clone(),
            )
        )
        return real_gae(
            rewards, values, dones, next_value, next_done, **kwargs
        )

    monkeypatch.setattr(train_module, "compute_gae", recording_gae)
    args = parse_args(
        [
            "--cuda", "False",
            "--seed", "0",
            "--exp-name", "vf_fresh",
            "--run-dir", str(tmp_path),
            "--total-timesteps", "2048",
            "--num-envs", "2",
            "--num-steps", "128",
            "--num-minibatches", "2",
            "--update-epochs", "1",
            "--vf-outcome", "true",
            "--checkpoint-interval", "0",
            "--log-interval", "0",
        ]
    )
    train(args)
    # base + jump stream per update
    assert len(calls) == 2 * (args.total_timesteps // args.batch_size)
    for rewards, dones, next_done in calls:
        # ``dones[t]`` marks the state entering step ``t``, so transition ``t``
        # is terminal iff ``dones[t + 1]`` (or ``next_done`` on the last step).
        terminal = torch.empty_like(dones)
        terminal[:-1] = dones[1:]
        terminal[-1] = next_done
        leaked = (rewards != 0) & (terminal == 0)
        assert not bool(leaked.any()), (
            "a nonzero decomposed reward outside a terminal transition is a "
            "stale buffer cell"
        )


def test_train_rejects_vf_outcome_misuse_before_creating_the_run(tmp_path):
    base = [
        "--cuda", "False",
        "--seed", "0",
        "--exp-name", "vf_bad",
        "--run-dir", str(tmp_path),
        "--total-timesteps", "128",
        "--num-envs", "2",
        "--num-steps", "64",
        "--num-minibatches", "2",
        "--update-epochs", "1",
        "--checkpoint-interval", "0",
        "--log-interval", "0",
    ]
    with pytest.raises(SystemExit, match="discrete outcome jump"):
        train(
            parse_args(
                [*base, "--reward-shaping", "terminal", "--vf-outcome", "true"]
            )
        )
    with pytest.raises(SystemExit, match="incompatible"):
        train(parse_args([*base, "--vf-outcome", "true", "--vf-nll", "true"]))
    with pytest.raises(SystemExit, match="outcome-coef"):
        train(parse_args([*base, "--outcome-coef", "-1"]))
    assert not any(tmp_path.iterdir())


def test_train_saturate_smoke(tmp_path):
    args = parse_args(
        [
            "--cuda", "False",
            "--seed", "0",
            "--exp-name", "saturate",
            "--run-dir", str(tmp_path),
            "--total-timesteps", "128",
            "--num-envs", "2",
            "--num-steps", "64",
            "--num-minibatches", "2",
            "--update-epochs", "1",
            "--reward-shaping", "saturate",
            "--reward-cap", "0.2",
            "--checkpoint-interval", "0",
            "--log-interval", "0",
        ]
    )
    run_dir = train(args)
    assert (run_dir / "agent.pt").exists()
    saved = json.loads((run_dir / "args.json").read_text())
    assert saved["reward_shaping"] == "saturate"
    assert saved["reward_cap"] == 0.2


def test_train_rejects_reward_cap_misuse_before_creating_the_run(tmp_path):
    base = [
        "--cuda", "False",
        "--seed", "0",
        "--exp-name", "cap",
        "--run-dir", str(tmp_path),
        "--total-timesteps", "128",
        "--num-envs", "2",
        "--num-steps", "64",
        "--num-minibatches", "2",
        "--update-epochs", "1",
        "--checkpoint-interval", "0",
        "--log-interval", "0",
    ]
    with pytest.raises(ValueError, match="reward_cap"):
        train(parse_args([*base, "--reward-shaping", "saturate"]))
    with pytest.raises(ValueError, match="reward_cap"):
        train(parse_args([*base, "--reward-cap", "0.2"]))
    with pytest.raises(ValueError, match="reward_cap"):
        train(parse_args(
            [*base, "--reward-shaping", "saturate", "--reward-cap", "1.5"]
        ))
    assert not any(tmp_path.iterdir())  # validation precedes run-dir creation


def test_parse_pool_member():
    assert parse_pool_member("random") == (1.0, "random")
    assert parse_pool_member("2.5@ckpt:runs/x/agent.pt") == (2.5, "ckpt:runs/x/agent.pt")
    with pytest.raises(ValueError):
        parse_pool_member("-1@random")


def test_train_mix_opponent_smoke(tmp_path):
    args = parse_args(
        [
            "--cuda", "False",
            "--seed", "1",
            "--exp-name", "mix",
            "--run-dir", str(tmp_path),
            "--total-timesteps", "128",
            "--num-envs", "2",
            "--num-steps", "64",
            "--num-minibatches", "2",
            "--update-epochs", "1",
            "--opponent", "mix",
            "--mix-random-prob", "0.5",
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
            "--total-timesteps", "128",
            "--num-envs", "2",
            "--num-steps", "64",
            "--num-minibatches", "2",
            "--update-epochs", "1",
            "--opponent", "pool",
            "--pool-member", "2@random",
            "--pool-member", "1@self",
            "--self-play-refresh", "1",
            "--checkpoint-interval", "0",
            "--log-interval", "0",
        ]
    )
    run_dir = train(args)
    assert (run_dir / "agent.pt").exists()
    saved = json.loads((run_dir / "args.json").read_text())
    assert saved["pool_sample"] is True


def test_pool_sample_cli_default_is_on():
    # Operator decision 2026-10-07 (ADR-0017): pool opponents sample actions
    # by default; the historical greedy pool needs an explicit opt-out.
    assert parse_args([]).pool_sample is True
    assert parse_args(["--pool-sample", "False"]).pool_sample is False


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
            "--pool-member", "1@random",
            "--pool-member", "1@self",
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
    args = _pool_run_args(tmp_path, "--total-timesteps", "128")
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
        _step, episodes, member_id, weight, _wr, _w, _d, _l = row.split(",")
        by_episode.setdefault(int(episodes), []).append((member_id, float(weight)))
        assert 0.0 <= float(weight) <= 1.0
    for entries in by_episode.values():
        assert len({member for member, _ in entries}) == 3
        # The CSV writes each weight with ``%.6g`` (6 significant digits), so
        # the rounded triples may sum to 1.0 ± a few 1e-7 even though the
        # in-memory distribution sums exactly; compare within the format's
        # precision rather than the 1e-6 default (ADR-0014 changed the sampled
        # records, which tripped this pre-existing brittleness).
        assert sum(weight for _, weight in entries) == pytest.approx(1.0, abs=1e-5)


def test_pfsp_refuses_to_infer_outcomes_from_a_non_sign_like_reward(
    tmp_path, monkeypatch
):
    """K0/saturate wins score 0, so a reward-sign fallback would log ties."""
    from seven523.env import Seven523Env

    real_step = Seven523Env.step

    def stripped_step(self, action):
        obs, reward, terminated, truncated, info = real_step(self, action)
        info.pop("outcome", None)
        return obs, reward, terminated, truncated, info

    monkeypatch.setattr(Seven523Env, "step", stripped_step)
    args = _pool_run_args(
        tmp_path,
        "--pool-episode",
        "--pfsp",
        "--pfsp-every",
        "1",
        "--reward-shaping",
        "saturate",
        "--reward-cap",
        "0.2",
    )
    with pytest.raises(RuntimeError, match="outcome"):
        train(args)


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
            "--total-timesteps", "128",
            "--num-envs", "2",
            "--num-steps", "64",
            "--num-minibatches", "2",
            "--update-epochs", "1",
            "--log-interval", "0",
            *extra,
        ]
    )


def test_train_writes_tensorboard_scalars(tmp_path):
    run_dir = train(
        _tiny_args(tmp_path, "--total-timesteps", "256", "--tensorboard", "True")
    )
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
    assert "obs_version" not in extra["args"]
    payload = torch.load(run_dir / "agent.pt", map_location="cpu", weights_only=True)
    assert payload["obs_version"] == 5


# -- residual body + plain heads, LN/GroupNorm (docs/gnres-plan.md) ----------


def test_parse_gnres_arch_and_gn_groups():
    assert parse_args([]).gn_groups == 8
    assert parse_args(["--arch", "gnres21"]).arch == "gnres21"
    assert parse_args(["--arch", "lnres21"]).arch == "lnres21"
    assert parse_args(["--arch", "bnres21"]).arch == "bnres21"
    args = parse_args(["--arch", "gnres22", "--gn-groups", "16"])
    assert args.gn_groups == 16
    with pytest.raises(SystemExit):
        parse_args(["--arch", "gnres11x"])


def test_train_gnres_smoke(tmp_path):
    run_dir = train(
        _tiny_args(
            tmp_path,
            "--arch", "gnres21",
            "--actor-out-std", "0.1",
            "--gn-groups", "8",
            "--checkpoint-interval", "0",
        )
    )
    agent, extra = load_agent(run_dir / "agent.pt")
    assert agent.arch == "gnres21"
    assert agent.gn_groups == 8
    assert extra["args"]["gn_groups"] == 8
    payload = torch.load(run_dir / "agent.pt", map_location="cpu", weights_only=True)
    assert payload["gn_groups"] == 8


def test_parse_res_arch_family():
    assert parse_args(["--arch", "res421"]).arch == "res421"
    assert parse_args(["--arch", "res433"]).arch == "res433"
    with pytest.raises(SystemExit):
        parse_args(["--arch", "res421x"])


def test_parse_bres_arch_family():
    assert parse_args(["--arch", "bres421"]).arch == "bres421"
    assert parse_args(["--arch", "bres221"]).arch == "bres221"
    assert parse_args([]).res_expansion == 3
    assert (
        parse_args(["--arch", "bres421", "--res-expansion", "2"]).res_expansion
        == 2
    )
    assert (
        parse_args(["--arch", "bres221", "--res-expansion", "0.25"]).res_expansion
        == 0.25
    )
    with pytest.raises(SystemExit):
        parse_args(["--arch", "bres421x"])


def test_train_res4_vf_outcome_smoke(tmp_path):
    run_dir = train(
        _tiny_args(
            tmp_path,
            "--arch", "res421",
            "--hidden-size", "64",
            "--actor-out-std", "0.1",
            "--vf-outcome", "true",
            "--checkpoint-interval", "0",
        )
    )
    agent, extra = load_agent(run_dir / "agent.pt")
    assert agent.arch == "res421" and agent.hidden == 64 and agent.vf_outcome
    assert extra["args"]["hidden_size"] == 64


def test_train_bres_vf_outcome_smoke(tmp_path):
    run_dir = train(
        _tiny_args(
            tmp_path,
            "--arch", "bres421",
            "--hidden-size", "32",
            "--res-expansion", "3",
            "--actor-out-std", "0.1",
            "--vf-outcome", "true",
            "--checkpoint-interval", "0",
        )
    )
    agent, extra = load_agent(run_dir / "agent.pt")
    assert agent.arch == "bres421" and agent.hidden == 32 and agent.vf_outcome
    assert agent.res_expansion == 3
    block = agent.network.blocks[1][0]
    assert (block.in_features, block.out_features) == (32, 96)
    assert extra["args"]["res_expansion"] == 3


def test_train_bnres_vf_outcome_smoke(tmp_path):
    run_dir = train(
        _tiny_args(
            tmp_path,
            "--arch", "bnres21",
            "--actor-out-std", "0.1",
            "--vf-outcome", "true",
            "--checkpoint-interval", "0",
        )
    )
    agent, extra = load_agent(run_dir / "agent.pt")
    assert agent.arch == "bnres21" and agent.vf_outcome
    assert extra["args"]["vf_outcome"] is True


def test_train_restores_train_mode_after_the_periodic_eval(tmp_path, monkeypatch):
    """Regression: ``NeuralPolicy(agent)`` evals the live learner in place.

    A ``bnres`` body must keep updating its running statistics during the
    rollout/update, so the periodic evaluation (which needs eval mode for
    deterministic inference) has to restore train mode afterwards.
    """
    from seven523 import train as train_module

    modes: list[bool] = []
    real_ppo = train_module.ppo_update

    def recording_ppo(agent, optimizer, batch, config):
        modes.append(agent.training)
        return real_ppo(agent, optimizer, batch, config)

    monkeypatch.setattr(train_module, "ppo_update", recording_ppo)
    train(
        _tiny_args(
            tmp_path,
            "--total-timesteps", "256",
            "--arch", "bnres21",
            "--eval-interval", "1",
            "--eval-episodes", "1",
            "--checkpoint-interval", "0",
        )
    )
    assert modes == [True, True]


def test_train_rejects_gnres_group_misuse_before_creating_the_run(tmp_path):
    with pytest.raises(SystemExit, match="must divide"):
        train(
            _tiny_args(
                tmp_path,
                "--arch", "gnres21",
                "--hidden-size", "16",
                "--gn-groups", "3",
                "--checkpoint-interval", "0",
            )
        )
    assert not any(tmp_path.iterdir())
    with pytest.raises(SystemExit, match="gn-groups"):
        train(
            _tiny_args(
                tmp_path,
                "--arch", "gnres21",
                "--gn-groups", "0",
                "--checkpoint-interval", "0",
            )
        )
    assert not any(tmp_path.iterdir())


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



# -- event (EVH) training plumbing -------------------------------------------


def test_train_event_smoke_and_checkpoint_layout(tmp_path):
    args = parse_args(
        [
            "--cuda", "False",
            "--seed", "0",
            "--exp-name", "event",
            "--run-dir", str(tmp_path),
            "--total-timesteps", "128",
            "--num-envs", "2",
            "--num-steps", "32",
            "--num-minibatches", "2",
            "--update-epochs", "1",
            "--event-len", "12",
            "--event-emb", "4",
            "--event-hidden", "4",
            "--seat-emb", "2",
            "--checkpoint-interval", "0",
            "--log-interval", "0",
        ]
    )
    run_dir = train(args)
    payload = torch.load(run_dir / "agent.pt", map_location="cpu", weights_only=True)
    assert payload["event_len"] == 12
    assert payload["history_layout"].startswith("event:12:")
    loaded, _ = load_agent(run_dir / "agent.pt")
    assert loaded.event_len == 12


@pytest.mark.parametrize(
    "extra",
    [
        ["--event-noisy", "True"],
        ["--event-seat", "none"],
        ["--event-order", "shuffled"],
        ["--event-pass", "drop"],
        ["--event-boundary-blind", "True"],
        ["--event-blind", "True"],
    ],
)
def test_train_event_ablation_switches_smoke(tmp_path, extra):
    args = parse_args(
        [
            "--cuda", "False",
            "--seed", "0",
            "--exp-name", "eventabo",
            "--run-dir", str(tmp_path),
            "--total-timesteps", "128",
            "--num-envs", "2",
            "--num-steps", "32",
            "--num-minibatches", "2",
            "--update-epochs", "1",
            "--event-len", "8",
            "--event-emb", "4",
            "--event-hidden", "4",
            "--seat-emb", "2",
            "--checkpoint-interval", "0",
            "--log-interval", "0",
            *extra,
        ]
    )
    run_dir = train(args)
    loaded, _ = load_agent(run_dir / "agent.pt")
    assert loaded.event_len == 8


def test_train_event_warm_start_requires_same_layout(tmp_path):
    base = parse_args(
        [
            "--cuda", "False",
            "--seed", "0",
            "--exp-name", "mlp",
            "--run-dir", str(tmp_path),
            "--total-timesteps", "128",
            "--num-envs", "2",
            "--num-steps", "32",
            "--num-minibatches", "2",
            "--update-epochs", "1",
            "--checkpoint-interval", "0",
            "--log-interval", "0",
        ]
    )
    mlp_dir = train(base)
    event = parse_args(
        [
            "--cuda", "False",
            "--seed", "0",
            "--exp-name", "eventwarm",
            "--run-dir", str(tmp_path),
            "--total-timesteps", "128",
            "--num-envs", "2",
            "--num-steps", "32",
            "--num-minibatches", "2",
            "--update-epochs", "1",
            "--event-len", "12",
            "--checkpoint-interval", "0",
            "--log-interval", "0",
        ]
    )
    event.load_checkpoint = str(mlp_dir / "agent.pt")
    with pytest.raises(SystemExit, match="cannot warm-start layout"):
        train(event)

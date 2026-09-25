"""PPO training / evaluation smoke tests.

Torch lives in the optional ``train`` dependency group
(``uv sync --group train``); without it this whole module is skipped.
"""
import numpy as np
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
from seven523.train import parse_args, train  # noqa: E402

OBS_DIM = 191
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

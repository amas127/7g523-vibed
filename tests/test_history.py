"""D1-lite sequence memory: encoder, wrapper, policy reuse, PPO plumbing.

The invariants under test:
* the rollout wrapper and ``NeuralPolicy`` encode the *same* sequence for the
  same ``View`` (training/inference parity);
* ``seq_len=0`` keeps the historical MLP path exactly;
* a trained checkpoint round-trips the sequence architecture and ``head_to_head``
  can rebuild it through ``load_agent``.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch
from support import FirstLegalBot

import seven523.networks as networks_mod
from seven523.actions import joint_mask_bits, nvec_for
from seven523.cards import card_id, make_deck
from seven523.env import Seven523Env, encode_observation, observation_dim
from seven523.history import (
    HISTORY_LENGTH,
    ORDERINGS,
    HistorySequenceWrapper,
    encode_history,
)
from seven523.networks import Agent, NeuralPolicy, load_agent, save_agent
from seven523.ppo import PPOConfig, RolloutBatch, ppo_update
from seven523.rules import DEFAULT_RULES
from seven523.train import parse_args, train

OBS_DIM = observation_dim(2)
NVEC = nvec_for(DEFAULT_RULES)


def _config(**overrides):
    base = {
        "num_minibatches": 2,
        "update_epochs": 1,
        "norm_adv": True,
        "clip_coef": 0.2,
        "clip_vloss": True,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "target_kl": None,
    }
    base.update(overrides)
    return PPOConfig(**base)


def _first_legal(env: Seven523Env) -> int:
    return next(
        index
        for index, legal in enumerate(env.action_mask[: env.action_space_n])
        if legal
    )


# -- sequence encoding and rollout wrapper -----------------------------------


def test_encode_history_matches_view_order_and_masks_padding():
    env = Seven523Env(seed=3, opponents=[FirstLegalBot(), FirstLegalBot()])
    wrapper = HistorySequenceWrapper(env, HISTORY_LENGTH)
    wrapper.reset(seed=3)
    decisions = 0
    while not env.state.done and decisions < 80:
        wrapper.step(_first_legal(env))
        decisions += 1
        view = env.view()
        expected = encode_history(view, HISTORY_LENGTH)
        assert np.array_equal(wrapper.last_seq, expected)
        cards = view.played + view.trick_cards
        assert int((wrapper.last_seq > 0).sum()) == len(cards)
        for token, card in zip(wrapper.last_seq, cards):
            assert token == card_id(card) + 1
    assert decisions > 0


def test_encode_history_default_order_is_chronological():
    env = Seven523Env(seed=5, opponents=[FirstLegalBot(), FirstLegalBot()])
    env.reset(seed=5)
    deck = make_deck()
    view = replace(env.view(), played=tuple(deck[:4]), trick_cards=tuple(deck[4:6]))
    assert ORDERINGS == ("chrono", "sorted")
    assert np.array_equal(encode_history(view), encode_history(view, order="chrono"))
    with pytest.raises(ValueError, match="unknown history order"):
        encode_history(view, order="shuffled")


def test_encode_history_sorted_is_the_same_multiset_without_order():
    env = Seven523Env(seed=7, opponents=[FirstLegalBot(), FirstLegalBot()])
    env.reset(seed=7)
    # Descending card ids in play order guarantees chrono != sorted.
    ordered = sorted(make_deck(), key=card_id)
    played = (ordered[8], ordered[3], ordered[9], ordered[0])
    view = replace(env.view(), played=played, trick_cards=(ordered[5],))
    chrono = encode_history(view, HISTORY_LENGTH, "chrono")
    sorted_tokens = encode_history(view, HISTORY_LENGTH, "sorted")
    chrono_real = chrono[chrono > 0].tolist()
    sorted_real = sorted_tokens[sorted_tokens > 0].tolist()
    assert sorted(chrono_real) == sorted_real  # same multiset
    assert chrono_real != sorted_real  # order actually removed
    assert sorted_real == sorted(sorted_real)
    # Padding stays at the tail; real tokens start at index 0.
    count = len(sorted_real)
    assert int((sorted_tokens > 0).sum()) == count
    assert np.array_equal(sorted_tokens[count:], np.zeros(HISTORY_LENGTH - count))
    # Truncation keeps the same most-recent multiset as the chrono arm.
    sorted_truncated = encode_history(view, 3, "sorted")
    chrono_truncated = encode_history(view, 3, "chrono")
    assert sorted(sorted_truncated[sorted_truncated > 0].tolist()) == sorted(
        chrono_truncated[chrono_truncated > 0].tolist()
    )


def test_encode_history_keeps_the_most_recent_when_truncated():
    env = Seven523Env(seed=5, opponents=[FirstLegalBot(), FirstLegalBot()])
    env.reset(seed=5)
    deck = make_deck()
    view = replace(env.view(), played=tuple(deck[:4]), trick_cards=tuple(deck[4:6]))
    full = encode_history(view, HISTORY_LENGTH)
    assert np.array_equal(full[:6], [card_id(card) + 1 for card in deck[:6]])
    short = encode_history(view, 3)
    assert np.array_equal(short, [card_id(card) + 1 for card in deck[3:6]])


# -- policy / training parity ------------------------------------------------


def _masked_picks(
    agent: Agent,
    view,
    seqs: torch.Tensor,
    events: torch.Tensor | None = None,
    event_seats: torch.Tensor | None = None,
    event_mask: torch.Tensor | None = None,
):
    obs = torch.tensor(
        [encode_observation(view, DEFAULT_RULES)], dtype=torch.float32
    )
    logits = agent.policy_logits(obs, seqs, events, event_seats, event_mask)
    mask = torch.tensor([joint_mask_bits(view.mask, NVEC)], dtype=torch.bool)
    picks = []
    for head_logits, head_mask in zip(
        torch.split(logits, NVEC, dim=1), torch.split(mask, NVEC, dim=1)
    ):
        picks.append(
            int(
                torch.where(
                    head_mask, head_logits, torch.tensor(-1e8)
                ).argmax(dim=1)
            )
        )
    return picks[0], (picks[1] if len(picks) > 1 else None)


def test_neural_policy_rebuilds_the_same_sequence_as_the_rollout_wrapper():
    torch.manual_seed(0)
    agent = Agent(OBS_DIM, NVEC, hidden=16, seq_len=HISTORY_LENGTH)
    policy = NeuralPolicy(agent, DEFAULT_RULES)
    env = Seven523Env(seed=1, opponents=[FirstLegalBot(), FirstLegalBot()])
    wrapper = HistorySequenceWrapper(env, HISTORY_LENGTH)
    wrapper.reset(seed=1)
    decisions = 0
    while not env.state.done and decisions < 40:
        view = env.view()
        assert np.array_equal(wrapper.last_seq, encode_history(view))
        action = policy.act(view)
        seqs = torch.as_tensor(
            np.asarray([encode_history(view, HISTORY_LENGTH)]), dtype=torch.int64
        )
        assert action == _masked_picks(agent, view, seqs)
        decisions += 1
        wrapper.step(_first_legal(env))
    assert decisions >= 10


# -- architecture and checkpointing -----------------------------------------


def test_neural_policy_sorted_order_matches_the_rollout_wrapper():
    torch.manual_seed(0)
    agent = Agent(
        OBS_DIM, NVEC, hidden=16, seq_len=HISTORY_LENGTH, seq_order="sorted"
    )
    policy = NeuralPolicy(agent, DEFAULT_RULES)
    env = Seven523Env(seed=4, opponents=[FirstLegalBot(), FirstLegalBot()])
    wrapper = HistorySequenceWrapper(env, HISTORY_LENGTH, "sorted")
    wrapper.reset(seed=4)
    decisions = 0
    while not env.state.done and decisions < 25:
        view = env.view()
        assert np.array_equal(
            wrapper.last_seq, encode_history(view, HISTORY_LENGTH, "sorted")
        )
        action = policy.act(view)
        seqs = torch.as_tensor(
            np.asarray([encode_history(view, HISTORY_LENGTH, "sorted")]),
            dtype=torch.int64,
        )
        assert action == _masked_picks(agent, view, seqs)
        decisions += 1
        wrapper.step(_first_legal(env))
    assert decisions >= 10


def test_sequence_agent_round_trips_the_architecture(tmp_path):
    torch.manual_seed(0)
    agent = Agent(OBS_DIM, NVEC, hidden=16, seq_len=12, seq_emb=8, seq_hidden=6)
    path = tmp_path / "seq.pt"
    save_agent(path, agent, extra={"global_step": 3})
    loaded, extra = load_agent(path)
    assert extra == {"global_step": 3}
    assert (loaded.seq_len, loaded.seq_emb, loaded.seq_hidden) == (12, 8, 6)
    assert loaded.seq_order == "chrono"  # pre-order checkpoints default to chrono
    obs = torch.rand(3, OBS_DIM)
    seqs = torch.randint(0, 55, (3, 12))
    assert torch.allclose(agent.policy_logits(obs, seqs), loaded.policy_logits(obs, seqs))
    assert hasattr(loaded, "seq_encoder") and loaded.seq_encoder is not None


def test_sorted_sequence_agent_round_trips_the_order(tmp_path):
    torch.manual_seed(0)
    agent = Agent(
        OBS_DIM, NVEC, hidden=16, seq_len=12, seq_emb=8, seq_hidden=6,
        seq_order="sorted",
    )
    path = tmp_path / "sorted.pt"
    save_agent(path, agent)
    loaded, _ = load_agent(path)
    assert loaded.seq_order == "sorted"
    obs = torch.rand(3, OBS_DIM)
    seqs = torch.randint(1, 55, (3, 12))
    assert torch.allclose(agent.policy_logits(obs, seqs), loaded.policy_logits(obs, seqs))


def test_mlp_agent_path_is_unchanged():
    torch.manual_seed(0)
    agent = Agent(OBS_DIM, NVEC, hidden=16)
    assert agent.seq_encoder is None
    assert not any(name.startswith("seq_encoder") for name in agent.state_dict())
    obs = torch.rand(4, OBS_DIM)
    assert torch.equal(agent.policy_logits(obs), agent.policy_logits(obs, None))
    with pytest.raises(ValueError, match="no sequence encoder"):
        agent.policy_logits(obs, torch.zeros(4, 4, dtype=torch.long))


def test_seq_blind_agent_never_reads_the_history(monkeypatch):
    torch.manual_seed(0)
    agent = Agent(OBS_DIM, NVEC, hidden=16, seq_len=8, seq_blind=True)
    policy = NeuralPolicy(agent, DEFAULT_RULES)
    env = Seven523Env(seed=2, opponents=[FirstLegalBot(), FirstLegalBot()])
    env.reset(seed=2)
    view = env.view()

    def _boom(*args, **kwargs):  # pragma: no cover - must not be called
        raise AssertionError("history read although seq_blind=True")

    monkeypatch.setattr(networks_mod, "encode_history", _boom)
    action = policy.act(view)
    assert action[0] in range(env.action_space_n)
    zeros = torch.zeros(1, 8, dtype=torch.long)
    assert action == _masked_picks(agent, view, zeros)


def test_ppo_update_trains_the_sequence_encoder():
    torch.manual_seed(0)
    agent = Agent(OBS_DIM, NVEC, hidden=16, seq_len=10, seq_emb=8, seq_hidden=6)
    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)
    batch = RolloutBatch.flatten(
        obs=torch.rand(4, 2, OBS_DIM),
        actions=torch.zeros(4, 2, len(NVEC), dtype=torch.long),
        logprobs=torch.zeros(4, 2),
        advantages=torch.randn(4, 2),
        returns=torch.randn(4, 2),
        values=torch.randn(4, 2),
        action_masks=torch.ones(4, 2, sum(NVEC)),
        seqs=torch.randint(0, 55, (4, 2, 10)),
    )
    assert batch.seqs is not None and batch.seqs.shape == (8, 10)
    before = agent.seq_encoder.embedding.weight.detach().clone()
    losses = ppo_update(agent, optimizer, batch, _config())
    assert all(np.isfinite(value) for value in losses.values())
    assert not torch.equal(before, agent.seq_encoder.embedding.weight.detach())


def test_rollout_batch_without_seqs_keeps_none():
    batch = RolloutBatch.flatten(
        obs=torch.zeros(1, 1, OBS_DIM),
        actions=torch.zeros(1, 1, len(NVEC), dtype=torch.long),
        logprobs=torch.zeros(1, 1),
        advantages=torch.zeros(1, 1),
        returns=torch.zeros(1, 1),
        values=torch.zeros(1, 1),
        action_masks=torch.ones(1, 1, sum(NVEC)),
    )
    assert batch.seqs is None


# -- end-to-end training smoke ----------------------------------------------


def test_train_sequence_smoke(tmp_path):
    args = parse_args(
        [
            "--cuda", "False",
            "--seed", "0",
            "--exp-name", "seq",
            "--run-dir", str(tmp_path),
            "--total-timesteps", "256",
            "--num-envs", "2",
            "--num-steps", "64",
            "--num-minibatches", "2",
            "--update-epochs", "1",
            "--seq-len", "12",
            "--checkpoint-interval", "0",
            "--log-interval", "0",
        ]
    )
    run_dir = train(args)
    loaded, extra = load_agent(run_dir / "agent.pt")
    assert loaded.seq_len == 12
    assert extra["args"]["seq_len"] == 12
    assert (run_dir / "metrics.csv").exists()


def test_train_sequence_sorted_smoke(tmp_path):
    args = parse_args(
        [
            "--cuda", "False",
            "--seed", "0",
            "--exp-name", "seqsorted",
            "--run-dir", str(tmp_path),
            "--total-timesteps", "256",
            "--num-envs", "2",
            "--num-steps", "64",
            "--num-minibatches", "2",
            "--update-epochs", "1",
            "--seq-len", "12",
            "--seq-order", "sorted",
            "--checkpoint-interval", "0",
            "--log-interval", "0",
        ]
    )
    run_dir = train(args)
    loaded, extra = load_agent(run_dir / "agent.pt")
    assert loaded.seq_order == "sorted"
    assert extra["args"]["seq_order"] == "sorted"


def test_train_sequence_blind_smoke(tmp_path):
    args = parse_args(
        [
            "--cuda", "False",
            "--seed", "0",
            "--exp-name", "seqblind",
            "--run-dir", str(tmp_path),
            "--total-timesteps", "256",
            "--num-envs", "2",
            "--num-steps", "64",
            "--num-minibatches", "2",
            "--update-epochs", "1",
            "--seq-len", "12",
            "--seq-blind",
            "--checkpoint-interval", "0",
            "--log-interval", "0",
        ]
    )
    run_dir = train(args)
    loaded, extra = load_agent(run_dir / "agent.pt")
    assert loaded.seq_blind is True
    assert extra["args"]["seq_blind"] is True


# -- event history (EVH) ------------------------------------------------------
import random  # noqa: E402

#: Two cards of the same rank, for a guaranteed PAIR combo.
from seven523.combos import classify  # noqa: E402
from seven523.game import Game, Play  # noqa: E402
from seven523.history import (  # noqa: E402
    EVENT_DIM,
    EVENT_ORDERINGS,
    PASS_MODES,
    WENT_OUT_MODES,
    EventHistoryWrapper,
    encode_events,
    event_length,
)
from seven523.policies import RandomBot  # noqa: E402
from seven523.rules import Rules  # noqa: E402

TWO = Rules(num_players=2)
THREE = Rules(num_players=3)


def _same_rank_pair():
    deck = make_deck()
    by_rank = {}
    for card in deck:
        by_rank.setdefault(card.rank, []).append(card)
    return next(tuple(cards[:2]) for cards in by_rank.values() if len(cards) >= 2)


def _base_view():
    env = Seven523Env(seed=1, opponents=[FirstLegalBot(), FirstLegalBot()])
    env.reset(seed=1)
    return env.view()


def _row(events, index):
    return events[index]


def test_event_length_is_players_times_deck():
    assert event_length(TWO) == 108
    assert event_length(THREE) == 162
    assert EVENT_DIM == 64
    assert EVENT_ORDERINGS == ("chrono", "shuffled")
    assert PASS_MODES == ("keep", "drop")
    assert WENT_OUT_MODES == ("keep", "drop")


def test_encode_events_schema_offsets_and_flags():
    single = make_deck()[0]
    pair = _same_rank_pair()
    plays = (
        Play(seat=1, cards=pair, opens_trick=True, went_out=False),
        Play(seat=0, cards=(), opens_trick=False, went_out=False),
        Play(seat=1, cards=(single,), opens_trick=False, went_out=True),
    )
    view = replace(_base_view(), seat=0, plays=plays)
    events, seats, mask = encode_events(view, TWO, length=4)

    assert events.shape == (4, EVENT_DIM)
    assert events.dtype == np.float32
    assert seats.dtype == np.int64
    assert mask.dtype == np.bool_
    assert mask.tolist() == [True, True, True, False]

    # Event 0: pair multihot + kind one-hot + size + opener.
    row = _row(events, 0)
    assert row[:54].sum() == 2
    for card in pair:
        assert row[card_id(card)] == 1.0
    kind = classify(pair, TWO).kind
    assert row[54 : 60][list(type(kind)).index(kind)] == 1.0
    assert row[60] == pytest.approx(2 / TWO.hand_size)
    assert row[61] == 0.0  # is_pass
    assert row[62] == 1.0  # opens_trick
    assert row[63] == 0.0  # went_out

    # Event 1: pass.
    row = _row(events, 1)
    assert row[:54].sum() == 0 and row[54:60].sum() == 0 and row[60] == 0.0
    assert row[61] == 1.0 and row[62] == 0.0 and row[63] == 0.0

    # Event 2: single, closes the hand.
    row = _row(events, 2)
    assert row[:54].sum() == 1 and row[card_id(single)] == 1.0
    assert row[60] == pytest.approx(1 / TWO.hand_size)
    assert row[61] == 0.0 and row[62] == 0.0 and row[63] == 1.0

    # Relative seats: (1-0)%2=1, (0-0)%2=0, (1-0)%2=1, pad=2.
    assert seats.tolist() == [1, 0, 1, 2]
    # Padding event is all-zero and unmasked.
    assert _row(events, 3).sum() == 0.0


def test_encode_events_relative_seat_rotates_with_the_acting_seat():
    play = Play(seat=1, cards=(make_deck()[0],), opens_trick=True, went_out=False)
    played = (play,)
    view0 = replace(_base_view(), seat=0, plays=played)
    view1 = replace(_base_view(), seat=1, plays=played)
    assert encode_events(view0, TWO)[1][0] == 1  # opponent
    assert encode_events(view1, TWO)[1][0] == 0  # self


def test_encode_events_three_players_pads_with_n():
    play = Play(seat=2, cards=(make_deck()[0],), opens_trick=True, went_out=False)
    view = replace(_base_view(), seat=0, plays=(play,))
    _events, seats, _ = encode_events(view, THREE, length=3)
    assert seats.tolist()[0] == 2
    assert seats.tolist()[1:] == [3, 3]


def test_encode_events_keeps_the_most_recent_when_truncated():
    deck = make_deck()
    plays = tuple(
        Play(seat=i % 2, cards=(card,), opens_trick=i % 2 == 0, went_out=False)
        for i, card in enumerate(deck[:5])
    )
    view = replace(_base_view(), seat=0, plays=plays)
    events, seats, mask = encode_events(view, TWO, length=2)
    assert mask.tolist() == [True, True]
    for index, expected in enumerate(plays[3:], start=0):
        assert events[index][card_id(expected.cards[0])] == 1.0
        assert seats[index] == (expected.seat) % 2


def test_encode_events_blind_zeroes_events_and_seats():
    pair = _same_rank_pair()
    plays = (
        Play(1, pair, True, False),
        Play(0, (), False, False),
    )
    view = replace(_base_view(), seat=0, plays=plays)
    events, seats, mask = encode_events(view, TWO, length=4, blind=True)
    assert events.sum() == 0.0
    assert seats.tolist() == [2, 2, 2, 2]
    assert mask.tolist() == [True, True, False, False]


def test_encode_events_noisy_is_information_zero_but_count_preserving():
    pair = _same_rank_pair()
    view_a = replace(
        _base_view(),
        seat=0,
        plays=(Play(1, pair, True, False), Play(0, (), False, False)),
    )
    view_b = replace(
        _base_view(),
        seat=0,
        plays=(
            Play(1, (make_deck()[7],), True, False),
            Play(0, (make_deck()[9],), False, True),
        ),
    )
    events_a, seats_a, mask_a = encode_events(view_a, TWO, length=6, noisy=True)
    events_b, seats_b, mask_b = encode_events(view_b, TWO, length=6, noisy=True)
    # Two events in both views: same count, same noise, no content dependence.
    assert np.array_equal(events_a, events_b)
    assert np.array_equal(seats_a, seats_b)
    assert np.array_equal(mask_a, mask_b)
    assert mask_a.tolist() == [True, True, False, False, False, False]
    assert events_a[:2].std() > 0.1  # non-degenerate
    assert seats_a.tolist() == [2] * 6
    # Deterministic across calls (training/inference parity).
    assert np.array_equal(events_a, encode_events(view_a, TWO, length=6, noisy=True)[0])
    # A shorter view uses the same fixed prefix but a shorter real mask.
    _events_c, _, mask_c = encode_events(
        view_b, TWO, length=6, noisy=False
    )
    assert mask_c.sum() == 2


def test_encode_events_pass_drop_wentout_drop_and_boundary_blind():
    single = make_deck()[0]
    plays = (
        Play(1, (single,), True, True),
        Play(0, (), False, False),
    )
    view = replace(_base_view(), seat=0, plays=plays)
    events, _seats, mask = encode_events(view, TWO, length=2, pass_mode="drop")
    assert mask.tolist() == [True, False]
    assert events[0][62] == 1.0 and events[0][63] == 1.0
    events, _, _ = encode_events(
        view, TWO, length=2, boundary_blind=True, went_out="drop"
    )
    assert events[0][62] == 0.0 and events[0][63] == 0.0


def test_encode_events_shuffled_keeps_openers_and_is_deterministic():
    deck = make_deck()

    def play(i, opens):
        return Play(i % 2, (deck[i],), opens, False)

    plays = (
        play(0, True),
        play(1, False),
        play(2, False),
        play(3, False),
        play(4, True),
        play(5, False),
    )
    view = replace(_base_view(), seat=0, plays=plays)
    chrono, _, _ = encode_events(view, TWO)
    shuffled_a, _, mask_a = encode_events(view, TWO, order="shuffled")
    shuffled_b, _, _ = encode_events(view, TWO, order="shuffled")
    assert np.array_equal(shuffled_a, shuffled_b)  # deterministic
    assert mask_a.sum() == 6
    # Openers keep their positions (0 and 4); the trick blocks stay intact.
    assert shuffled_a[0][62] == 1.0 and shuffled_a[4][62] == 1.0
    assert all(shuffled_a[i][62] == 0.0 for i in (1, 2, 3, 5))
    block1_chrono = {card_id(deck[i]) for i in range(4)}
    block1_shuffled = {
        idx for idx in range(54) if shuffled_a[:4, idx].sum() == 1 and idx not in
        {card_id(deck[0])}
    }
    assert block1_shuffled == {card_id(deck[i]) for i in (1, 2, 3)}
    assert block1_chrono == {card_id(deck[i]) for i in range(4)}
    assert not np.array_equal(chrono, shuffled_a)  # order actually changed


def test_encode_events_rejects_unknown_knobs_and_blind_noisy_conflict():
    view = _base_view()
    with pytest.raises(ValueError, match="unknown event order"):
        encode_events(view, TWO, order="globalshuffle")
    with pytest.raises(ValueError, match="unknown pass mode"):
        encode_events(view, TWO, pass_mode="ghost")
    with pytest.raises(ValueError, match="unknown went-out mode"):
        encode_events(view, TWO, went_out="ghost")
    with pytest.raises(ValueError, match="mutually exclusive"):
        encode_events(view, TWO, blind=True, noisy=True)


def test_event_count_never_exceeds_the_analytic_bound():
    """Random-policy full games pin the ``n * 54`` upper bound."""
    observed = {2: 0, 3: 0}
    for num_players in (2, 3):
        rules = Rules(num_players=num_players)
        game = Game(rules)
        for seed in range(120):
            state = game.new(random.Random(seed))
            bots = [RandomBot(random.Random(seed * 10 + seat)) for seat in range(num_players)]
            while not state.done:
                state, _ = game.step(state, *bots[state.current].act(game.view(state, state.current)))
            observed[num_players] = max(observed[num_players], len(state.plays))
            assert len(state.plays) <= event_length(rules)
    # Sanity: the bound is not vacuous and matches the red-team measurements
    # (n=2 max 72, n=3 max 93 across 600 games) approximately.
    assert 40 <= observed[2] <= 108
    assert 50 <= observed[3] <= 162


def test_event_history_wrapper_matches_direct_encoding():
    env = Seven523Env(seed=3, opponents=[FirstLegalBot(), FirstLegalBot()])
    wrapper = EventHistoryWrapper(env, TWO, length=108)
    wrapper.reset(seed=3)
    decisions = 0
    while not env.state.done and decisions < 60:
        view = env.view()
        events, seats, mask = encode_events(view, TWO, 108)
        assert np.array_equal(wrapper.last_events, events)
        assert np.array_equal(wrapper.last_event_seats, seats)
        assert np.array_equal(wrapper.last_event_mask, mask)
        wrapper.step(_first_legal(env))
        decisions += 1
    assert decisions > 0


def test_event_history_wrapper_shuffled_and_blind_are_deterministic():
    env = Seven523Env(seed=6, opponents=[FirstLegalBot(), FirstLegalBot()])
    wrapper = EventHistoryWrapper(env, TWO, length=12, order="shuffled", blind=True)
    wrapper.reset(seed=6)
    for _ in range(5):
        view = env.view()
        expected = encode_events(view, TWO, 12, order="shuffled", blind=True)
        assert np.array_equal(wrapper.last_events, expected[0])
        assert np.array_equal(wrapper.last_event_seats, expected[1])
        assert np.array_equal(wrapper.last_event_mask, expected[2])
        wrapper.step(_first_legal(env))


# -- event policy / checkpoint parity ----------------------------------------


def _event_tensors(agent, view):
    events, seats, mask = encode_events(
        view,
        DEFAULT_RULES,
        int(agent.event_len),
        order=agent.event_order,
        pass_mode=agent.event_pass,
        boundary_blind=agent.event_boundary_blind,
        went_out=agent.event_went_out,
        blind=agent.event_blind,
        noisy=agent.event_noisy,
    )
    return (
        torch.as_tensor(events, dtype=torch.float32).unsqueeze(0),
        torch.as_tensor(seats, dtype=torch.int64).unsqueeze(0),
        torch.as_tensor(mask, dtype=torch.bool).unsqueeze(0),
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"event_seat": "none"},
        {"event_seat": "sum", "seat_emb": 32},
        {"event_noisy": True},
        {"event_blind": True, "event_seat": "none"},
        {"event_order": "shuffled", "event_pass": "drop"},
    ],
)
def test_neural_policy_rebuilds_the_same_events_as_the_rollout_wrapper(kwargs):
    torch.manual_seed(0)
    agent = Agent(OBS_DIM, NVEC, hidden=16, event_len=24, **kwargs)
    policy = NeuralPolicy(agent, DEFAULT_RULES)
    env = Seven523Env(seed=1, opponents=[FirstLegalBot(), FirstLegalBot()])
    wrapper = EventHistoryWrapper(
        env,
        DEFAULT_RULES,
        length=24,
        order=agent.event_order,
        pass_mode=agent.event_pass,
        boundary_blind=agent.event_boundary_blind,
        went_out=agent.event_went_out,
        blind=agent.event_blind,
        noisy=agent.event_noisy,
    )
    wrapper.reset(seed=1)
    decisions = 0
    while not env.state.done and decisions < 30:
        view = env.view()
        assert np.array_equal(wrapper.last_events, _event_tensors(agent, view)[0].numpy()[0])
        action = policy.act(view)
        assert action == _masked_picks(agent, view, None, *_event_tensors(agent, view))
        decisions += 1
        wrapper.step(_first_legal(env))
    assert decisions >= 10


def test_event_agent_round_trips_the_architecture(tmp_path):
    torch.manual_seed(0)
    agent = Agent(
        OBS_DIM,
        NVEC,
        hidden=16,
        event_len=12,
        event_emb=8,
        event_hidden=6,
        seat_emb=8,
        event_seat="sum",
        event_order="shuffled",
        event_pass="drop",
        event_boundary_blind=True,
        event_went_out="drop",
    )
    path = tmp_path / "event.pt"
    save_agent(path, agent, extra={"global_step": 3})
    loaded, extra = load_agent(path)
    assert extra == {"global_step": 3}
    assert (loaded.event_len, loaded.event_emb, loaded.event_hidden) == (12, 8, 6)
    assert loaded.event_seat == "sum"
    assert loaded.event_order == "shuffled"
    assert loaded.event_pass == "drop"
    assert loaded.event_boundary_blind is True
    assert loaded.event_went_out == "drop"
    obs = torch.rand(3, OBS_DIM)
    events = torch.rand(3, 12, EVENT_DIM)
    seats = torch.randint(0, 3, (3, 12))
    mask = torch.ones(3, 12, dtype=torch.bool)
    assert torch.allclose(
        agent.policy_logits(obs, None, events, seats, mask),
        loaded.policy_logits(obs, None, events, seats, mask),
    )
    assert hasattr(loaded, "event_encoder") and loaded.event_encoder is not None

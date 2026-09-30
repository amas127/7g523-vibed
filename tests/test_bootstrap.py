"""Bootstrap arena: pure planners, parent selection, and a fake end-to-end run."""
from __future__ import annotations

import importlib.util
import json
import sys
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from seven523.bootstrap import (
    ANCHOR_ID,
    BootstrapConfig,
    derive_seed,
    init_pool,
    load_round_games,
    make_play_fn,
    plan_anchor_games,
    plan_pool_games,
    plan_qualify_games,
    plan_rating_games,
    qualify_priors,
    rate_pool,
    refit_pool,
    run_bootstrap,
    select_parents,
    status_rows,
    train_argv,
    write_pool_tensorboard,
)
from seven523.elo import PlayedGame
from seven523.ladder import Entrant, ScheduledGame
from seven523.pool import Pool, PoolMember

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"


def member(
    id_: str,
    mu: float,
    *,
    sigma: float = 50.0,
    round_: int = 0,
    parent: str | None = None,
) -> PoolMember:
    return PoolMember(
        id=id_,
        spec=f"ckpt:runs/{id_}/agent.pt",
        mu=mu,
        sigma=sigma,
        round=round_,
        parent=parent,
    )


class FakeTrainer:
    """Writes a placeholder checkpoint per call; records every fork request."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.calls: list[dict] = []

    def train(
        self,
        *,
        parent: PoolMember | None,
        roster,
        seed: int,
        exp_name: str,
        steps: int,
    ) -> str:
        self.calls.append(
            {
                "parent": parent.id if parent is not None else None,
                "roster": [m.id for m in roster],
                "seed": seed,
                "exp_name": exp_name,
                "steps": steps,
            }
        )
        path = self.root / "train" / f"{exp_name}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
        return str(path)


class ConcurrentTrainer(FakeTrainer):
    """FakeTrainer that records how many ``train`` calls run at once."""

    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def train(self, *, parent, roster, seed, exp_name, steps) -> str:
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(0.05)
            return super().train(
                parent=parent, roster=roster, seed=seed, exp_name=exp_name, steps=steps
            )
        finally:
            with self._lock:
                self.active -= 1


def fake_play(schedule, entrants, out_dir: Path):
    """Subject-wins results; writes the shard JSONL ``load_round_games`` reads."""
    out_dir.mkdir(parents=True, exist_ok=True)
    games: list[PlayedGame] = []
    rows: list[dict] = []
    for game in schedule:
        scores = (100, 0) if game.seats[0] == game.subject else (0, 100)
        games.append(PlayedGame(seed=game.seed, seats=game.seats, scores=scores))
        rows.append(
            {"seed": game.seed, "seats": list(game.seats), "scores": list(scores)}
        )
    (out_dir / "shard_00000.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    return games


def draw_play(schedule, entrants, out_dir: Path):
    """All draws: every rating stays at its prior, so a band never separates."""
    out_dir.mkdir(parents=True, exist_ok=True)
    games = [
        PlayedGame(seed=game.seed, seats=game.seats, scores=(50, 50))
        for game in schedule
    ]
    rows = [
        {"seed": game.seed, "seats": list(game.seats), "scores": list(game.scores)}
        for game in games
    ]
    (out_dir / "shard_00000.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    return games


def smoke_config(tmp_path: Path, **overrides) -> BootstrapConfig:
    values: dict = {
        "arena_dir": tmp_path / "arena",
        "seed_models": 2,
        "seed_steps": 8,
        "rounds": 1,
        "forks_per_parent": 2,
        "max_children": 4,
        "train_steps": 8,
        "train_workers": 2,
        "train_threads": 1,
        "qualify_games": 4,
        "mass_games": 4,
        "anchor_games": 2,
        "workers": 1,
        "tensorboard": False,
    }
    values.update(overrides)
    return BootstrapConfig(**values)


def test_derive_seed_is_stable_and_distinct():
    assert derive_seed(0, "fork", 1, "a", 0) == derive_seed(0, "fork", 1, "a", 0)
    assert derive_seed(0, "fork", 1, "a", 0) != derive_seed(0, "fork", 1, "a", 1)
    assert 0 <= derive_seed(3, "mass", 2) < 1 << 32


def test_plan_pool_games_quotas_twins_and_no_self_pair():
    ids = ["a", "b", "c"]
    schedule = plan_pool_games(ids, ids, games=4, seed=7)
    assert len(schedule) == 12
    assert Counter(game.subject for game in schedule) == {"a": 4, "b": 4, "c": 4}
    for subject in ids:
        twins: dict[int, list] = defaultdict(list)
        for game in (g for g in schedule if g.subject == subject):
            assert len(set(game.seats)) == 2
            assert subject in game.seats
            twins[game.seed].append(game)
        assert all(len(pair) == 2 for pair in twins.values())
        for pair in twins.values():
            assert pair[0].seats == pair[1].seats[::-1]


def test_plan_pool_games_is_deterministic():
    first = plan_pool_games(["a", "b"], ["a", "b"], games=2, seed=1)
    second = plan_pool_games(["b", "a"], ["b", "a"], games=2, seed=1)
    assert first == second


def test_plan_anchor_games_quota_and_opponent():
    schedule = plan_anchor_games(["a", "b"], games=4, seed=0)
    assert len(schedule) == 8
    assert Counter(game.subject for game in schedule) == {"a": 4, "b": 4}
    assert all(ANCHOR_ID in game.seats for game in schedule)


def test_plan_rating_games_pool_plus_anchor():
    schedule = plan_rating_games(["a", "b"], games=4, anchor_games=2, seed=3)
    assert len(schedule) == 2 * 4 + 2 * 2
    assert len([g for g in schedule if ANCHOR_ID in g.seats]) == 4


def test_plan_rating_games_needs_a_pair():
    with pytest.raises(ValueError, match="two pool members"):
        plan_rating_games(["a"], games=2, anchor_games=0, seed=0)


def test_planners_reject_odd_quotas():
    with pytest.raises(ValueError, match="even"):
        plan_rating_games(["a", "b"], games=3, anchor_games=0, seed=0)


def test_plan_qualify_games_empty_pool_plays_the_anchor():
    schedule = plan_qualify_games(["n0"], [], games=4, anchor_games=2, seed=0)
    assert len(schedule) == 4
    assert all(ANCHOR_ID in game.seats for game in schedule)


def test_plan_qualify_games_pool_plus_anchor():
    schedule = plan_qualify_games(["n0", "n1"], ["p0"], games=4, anchor_games=2, seed=1)
    assert Counter(game.subject for game in schedule) == {"n0": 6, "n1": 6}
    opponents = {g.seats[1 - g.seats.index(g.subject)] for g in schedule}
    assert opponents == {"p0", ANCHOR_ID}


def test_qualify_priors_uses_live_incumbent_ratings():
    incumbent = PoolMember(id="p0", spec="ckpt:p0.pt", mu=900.0, sigma=20.0)
    newcomer = PoolMember(
        id="c0",
        spec="ckpt:c0.pt",
        mu=880.0,
        sigma=200.0,
        parent="p0",
        prior_mu=900.0,
        prior_sigma=200.0,
    )
    priors = qualify_priors([incumbent], [newcomer])
    assert (priors["p0"].mu, priors["p0"].sigma) == (900.0, 20.0)
    assert (priors["c0"].mu, priors["c0"].sigma) == (900.0, 200.0)


def test_train_argv_seed_and_fork(tmp_path):
    config = smoke_config(tmp_path, tensorboard=True, cuda=False)
    roster = [member("a", 100.0), member("b", 200.0)]

    seed_argv = train_argv(
        config, parent=None, roster=(), seed=5, exp_name="boot_seed", steps=16
    )
    assert seed_argv[seed_argv.index("--opponent") + 1] == "random"
    assert "--load-checkpoint" not in seed_argv
    assert seed_argv[seed_argv.index("--tensorboard") + 1] == "true"
    assert seed_argv[seed_argv.index("--total-timesteps") + 1] == "16"

    fork_argv = train_argv(
        config, parent=roster[0], roster=roster, seed=6, exp_name="boot_fork", steps=16
    )
    assert fork_argv[fork_argv.index("--load-checkpoint") + 1] == roster[0].ckpt
    assert fork_argv[fork_argv.index("--opponent") + 1] == "pool"
    assert "--pool-episode" in fork_argv
    members = [
        fork_argv[i + 1]
        for i, token in enumerate(fork_argv)
        if token == "--pool-member"
    ]
    assert members == [roster[0].spec, roster[1].spec]
    assert "--pfsp" not in fork_argv

    weighted = smoke_config(tmp_path, pfsp=True, include_random=0.25)
    argv = train_argv(
        weighted, parent=roster[0], roster=roster, seed=6, exp_name="x", steps=16
    )
    assert "--pfsp" in argv
    assert "0.25@random" in [
        argv[i + 1] for i, token in enumerate(argv) if token == "--pool-member"
    ]


def test_write_pool_tensorboard_smoke(tmp_path):
    pytest.importorskip("torch.utils.tensorboard")
    pool = Pool([member("a", 100.0), member("b", 250.0)])
    written = write_pool_tensorboard(pool, tmp_path, round_index=1, log=lambda *a: None)
    assert written == 7  # 2 members x (mu, sigma) + 2 bands + pool size
    assert list((tmp_path / "tb").glob("events.out.tfevents.*"))


def test_subprocess_trainer_runs_and_discovers_checkpoint(tmp_path, monkeypatch):
    from seven523 import bootstrap as module

    config = smoke_config(tmp_path)
    trainer = module.SubprocessTrainer(config)
    captured: dict = {}
    run_name = "bootstrap_seed0__7__123"

    def fake_run(argv, **kwargs):
        captured["argv"] = list(argv)
        captured["kwargs"] = kwargs
        run_dir = config.arena_dir / "train" / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "agent.pt").write_bytes(b"x")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    ckpt = trainer.train(parent=None, roster=(), seed=7, exp_name="bootstrap_seed0", steps=8)
    assert ckpt == str(config.arena_dir / "train" / run_name / "agent.pt")
    assert captured["argv"] == module.train_argv(
        config, parent=None, roster=(), seed=7, exp_name="bootstrap_seed0", steps=8
    )
    assert captured["kwargs"]["env"]["OMP_NUM_THREADS"] == str(config.train_threads)
    assert (config.arena_dir / "logs" / "bootstrap_seed0.log").is_file()


def test_make_play_fn_writes_shard_jsonl(tmp_path):
    config = smoke_config(tmp_path, workers=1)
    schedule = (
        ScheduledGame(11, ("a", "b"), "a"),
        ScheduledGame(11, ("b", "a"), "a"),
        ScheduledGame(12, ("a", "b"), "b"),
        ScheduledGame(12, ("b", "a"), "b"),
    )
    entrants = [Entrant("a", "random"), Entrant("b", "random")]
    out_dir = tmp_path / "mass"
    games = make_play_fn(config)(schedule, entrants, out_dir)
    assert len(games) == 4
    shard = out_dir / "shard_00000.jsonl"
    assert shard.is_file()
    assert len(shard.read_text(encoding="utf-8").splitlines()) == 4


def test_select_parents_strongest_first_then_caps():
    pool = Pool(
        [
            member("a", 10.0),
            member("b", 150.0),
            member("c", 250.0),
            member("d", 450.0),
            member("e", 610.0),
        ]
    )
    all_parents = select_parents(pool, BootstrapConfig(max_children=10, forks_per_parent=2))
    assert [parent.id for parent in all_parents] == ["e", "d", "c", "b", "a"]
    capped = select_parents(pool, BootstrapConfig(max_children=4, forks_per_parent=2))
    assert [parent.id for parent in capped] == ["e", "d"]
    one = select_parents(pool, BootstrapConfig(max_children=2, forks_per_parent=2))
    assert [parent.id for parent in one] == ["e"]


def test_run_bootstrap_fake_end_to_end(tmp_path):
    config = smoke_config(tmp_path)
    trainer = FakeTrainer(tmp_path)
    result = run_bootstrap(config, trainer=trainer, play=fake_play, log=lambda *a: None)

    assert result.pool.round == 1
    assert len(trainer.calls) == 6  # 2 seeds + 4 forks (both baselines fork twice)
    assert [call["parent"] for call in trainer.calls[:2]] == [None, None]
    assert sorted(call["parent"] for call in trainer.calls[2:]) == [
        "seed0",
        "seed0",
        "seed1",
        "seed1",
    ]
    assert all(call["steps"] == 8 for call in trainer.calls)
    assert len(result.reports) == 1
    report = result.reports[0]
    assert len(report.trained) == 4
    assert report.added == report.trained  # pool under cap: every child stays
    assert report.evicted == ()

    pool_path = config.arena_dir / "pool.json"
    assert pool_path.is_file()
    for members in result.pool.bands().values():
        assert 1 <= len(members) <= config.band_cap
    assert len(result.pool) == 6
    assert (config.arena_dir / "rounds" / "r000" / "qualify" / "shard_00000.jsonl").is_file()
    assert (config.arena_dir / "rounds" / "r001" / "mass" / "shard_00000.jsonl").is_file()
    assert (config.arena_dir / "events.jsonl").is_file()

    games = load_round_games(config.arena_dir)
    assert games
    fit = refit_pool(result.pool, config)
    assert fit.games == len(games)
    assert set(result.pool.members) <= set(fit.ratings)

    rows = status_rows(result.pool)
    assert len(rows) == 6
    assert all(set(row) == {"band", "id", "mu", "sigma", "games", "parent", "round", "spec"} for row in rows)


def test_train_all_runs_forks_concurrently(tmp_path):
    config = smoke_config(tmp_path, train_workers=2)
    trainer = ConcurrentTrainer(tmp_path)
    run_bootstrap(config, trainer=trainer, play=fake_play, log=lambda *a: None)
    assert trainer.max_active == 2


def test_rate_pool_single_member_uses_anchor(tmp_path):
    pool = Pool([member("solo", 100.0)])
    config = smoke_config(tmp_path, mass_games=4, anchor_games=2)
    fit = rate_pool(pool, config, play=fake_play, log=lambda *a: None)
    assert fit.games == 2  # one subject, one seat-paired anchor twin
    assert len(pool) == 1


def test_run_bootstrap_trims_over_cap(tmp_path):
    # Two tied seeds land in one band; cap 1 forces the first update to evict.
    config = smoke_config(
        tmp_path, seed_models=2, band_cap=1, rounds=2, patience=99
    )
    result = run_bootstrap(
        config, trainer=FakeTrainer(tmp_path), play=fake_play, log=lambda *a: None
    )
    assert result.pool.round == 2
    assert result.reports[0].evicted
    assert any(report.evicted for report in result.reports)
    for members in result.pool.bands().values():
        assert len(members) <= config.band_cap


def test_run_bootstrap_cap_holds_after_rerate(tmp_path):
    # All draws keep every model near the same rating, and band_cap=1 still has
    # to hold at rest: the trim now runs after the rerating, not before it.
    config = smoke_config(
        tmp_path, seed_models=2, band_cap=1, rounds=1, anchor_games=2
    )
    result = run_bootstrap(
        config, trainer=FakeTrainer(tmp_path), play=draw_play, log=lambda *a: None
    )
    assert result.pool.round == 1
    assert result.reports[0].evicted
    for members in result.pool.bands().values():
        assert len(members) <= config.band_cap


def test_init_resumes_without_retraining(tmp_path):
    config = smoke_config(tmp_path)
    pool = init_pool(
        config, trainer=FakeTrainer(tmp_path), play=fake_play, log=lambda *a: None
    )
    assert len(pool) == 2
    second = FakeTrainer(tmp_path)
    resumed = init_pool(config, trainer=second, play=fake_play, log=lambda *a: None)
    assert second.calls == []
    assert {m.id for m in resumed} == {m.id for m in pool}


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "bootstrap_arena_tool", TOOLS_DIR / "bootstrap_arena.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cli_build_config_defaults():
    module = _load_tool()
    args = module.parse_args(["status", "--arena", "somewhere", "--band-cap", "3"])
    config = module.build_config(args)
    assert config.arena_dir == Path("somewhere")
    assert config.band_cap == 3
    assert config.tensorboard is True
    args = module.parse_args(["rate", "--refit", "--no-tensorboard"])
    assert args.refit is True
    assert module.build_config(args).tensorboard is False


def test_cli_run_end_to_end(tmp_path, monkeypatch):
    module = _load_tool()
    monkeypatch.setattr(module, "SubprocessTrainer", lambda config: FakeTrainer(tmp_path))
    monkeypatch.setattr(module, "make_play_fn", lambda config: fake_play)
    arena = tmp_path / "cli-arena"
    code = module.main(
        [
            "run",
            "--arena",
            str(arena),
            "--seed-models",
            "2",
            "--seed-steps",
            "4",
            "--rounds",
            "1",
            "--forks",
            "2",
            "--max-children",
            "4",
            "--train-steps",
            "4",
            "--train-workers",
            "2",
            "--qualify-games",
            "4",
            "--mass-games",
            "4",
            "--anchor-games",
            "2",
            "--workers",
            "1",
            "--no-tensorboard",
        ]
    )
    assert code == 0
    assert (arena / "pool.json").is_file()
    assert module.main(["status", "--arena", str(arena)]) == 0
    assert module.main(["rate", "--refit", "--arena", str(arena)]) == 0
    assert (arena / "refit.json").is_file()

"""Bootstrap arena: fork → train → qualify → trim → rate rounds over a pool.

The arena starts from ``seed_models`` from-scratch PPO runs (vs the pinned
RandomBot), rates them (定级赛), and then repeats one round per batch:

1. fork: every pool member (strongest first, capped by ``max_children``) is
   trained with derived seeds, ``forks_per_parent`` children each; every child
   warm-starts from its parent and trains against a snapshot of the current
   pool (``--opponent pool --pool-episode``), one process per child,
   ``train_workers`` in parallel;
2. qualify: every child plays a small slate against the pool plus the RandomBot
   anchor, so it enters the merge with a measured rating instead of an
   inherited one;
3. rate: the whole pool (provisional newcomers included) plays ``mass_games``
   random opponents plus ``anchor_games`` vs the pinned RandomBot, and one
   OpenSkill replay refreshes every rating (the accuracy pass);
4. trim: :meth:`seven523.pool.Pool.trim` enforces the band slot rules on that
   fresh rating — the only eviction point of the round, so the saved pool
   always satisfies 1..``band_cap`` per band.

Every dependency that touches the outside world is injected: the ``Trainer``
(default: ``uv run … 7g523-train`` subprocesses) and the play function (default:
:func:`seven523.ladder.play_games`).  The schedule planners are pure, so the
round logic is testable without torch or a terminal.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Protocol, runtime_checkable

from .elo import (
    DEFAULT_MU,
    DEFAULT_SIGMA,
    Fit,
    FitConfig,
    PlayedGame,
    Prior,
    fit_ratings,
)
from .ladder import Entrant, ScheduledGame, play_games
from .pool import Pool, PoolConfig, PoolMember, load_pool, save_pool
from .rules import DEFAULT_RULES, Rules

__all__ = [
    "ANCHOR_ID",
    "ANCHOR_RATING",
    "BootstrapConfig",
    "BootstrapResult",
    "PlayFn",
    "RoundReport",
    "SubprocessTrainer",
    "Trainer",
    "append_event",
    "derive_seed",
    "init_pool",
    "load_round_games",
    "make_play_fn",
    "plan_anchor_games",
    "plan_pool_games",
    "plan_qualify_games",
    "plan_rating_games",
    "pool_entrants",
    "qualify_priors",
    "rate_pool",
    "refit_pool",
    "run_bootstrap",
    "run_round",
    "select_parents",
    "status_rows",
    "train_argv",
    "write_pool_tensorboard",
]

#: The pinned gauge every round is anchored to (ADR-0012).
ANCHOR_ID = "random"
ANCHOR_RATING = 0.0

PlayFn = Callable[
    [Sequence[ScheduledGame], Sequence[Entrant], Path], Sequence[PlayedGame]
]


@dataclass(frozen=True, slots=True)
class BootstrapConfig:
    """Everything one bootstrap arena run needs (CLI flags mutate these)."""

    arena_dir: Path = Path("runs/bootstrap/arena")
    base_seed: int = 0
    seed_models: int = 5
    seed_steps: int = 100_000
    rounds: int = 1
    forks_per_parent: int = 2
    max_children: int = 10
    train_steps: int = 200_000
    train_workers: int = 5
    train_threads: int = 4
    train_cmd: tuple[str, ...] = ("uv", "run", "--group", "train", "7g523-train")
    train_extra: tuple[str, ...] = ()
    cuda: bool = False
    tensorboard: bool = True
    pfsp: bool = False
    include_random: float = 0.0
    qualify_games: int = 500
    mass_games: int = 200
    anchor_games: int = 20
    workers: int = 6
    device: str = "cpu"
    band_width: float = 200.0
    band_cap: int = 5
    margin: float = 10.0
    patience: int = 2
    progress_margin: float = 10.0
    resume: bool = True
    rules: Rules = DEFAULT_RULES
    fit: FitConfig = field(default_factory=FitConfig)

    def __post_init__(self) -> None:
        if self.seed_models < 2:
            raise ValueError("seed_models must be at least 2 (a rating needs a pair)")
        for name in ("seed_steps", "train_steps", "rounds", "train_workers", "train_threads"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1")
        if self.forks_per_parent < 1:
            raise ValueError("forks_per_parent must be at least 1")
        if self.max_children < self.forks_per_parent:
            raise ValueError("max_children must cover at least one parent")
        if self.include_random < 0.0:
            raise ValueError("include_random must be non-negative")
        if self.patience < 1:
            raise ValueError("patience must be at least 1")
        _check_even(self.qualify_games, "qualify_games")
        _check_even(self.mass_games, "mass_games")
        _check_even(self.anchor_games, "anchor_games")
        if self.qualify_games < 2:
            raise ValueError("qualify_games must be at least 2")


def _check_even(value: int, name: str) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")
    if value % 2 != 0:
        raise ValueError(
            f"{name} must be even: every deal is played in both seats, got {value!r}"
        )


@dataclass(frozen=True, slots=True)
class RoundReport:
    """The audit line of one bootstrap round."""

    round: int
    parents: tuple[str, ...]
    trained: tuple[str, ...]
    added: tuple[str, ...]
    evicted: tuple[str, ...]
    productive: bool
    games: int


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    """The final pool plus every round report (stop rule already applied)."""

    pool: Pool
    reports: tuple[RoundReport, ...]


@runtime_checkable
class Trainer(Protocol):
    """The injected training seam: parent snapshot → child checkpoint path."""

    def train(
        self,
        *,
        parent: PoolMember | None,
        roster: Sequence[PoolMember],
        seed: int,
        exp_name: str,
        steps: int,
    ) -> str: ...


def train_argv(
    config: BootstrapConfig,
    *,
    parent: PoolMember | None,
    roster: Sequence[PoolMember],
    seed: int,
    exp_name: str,
    steps: int,
) -> list[str]:
    """The exact ``7g523-train`` argv for one seed/fork run (pure, no IO)."""
    argv = [*config.train_cmd]
    argv += [
        "--exp-name",
        exp_name,
        "--seed",
        str(seed),
        "--total-timesteps",
        str(steps),
        "--run-dir",
        str(Path(config.arena_dir) / "train"),
        "--cuda",
        "true" if config.cuda else "false",
        "--tensorboard",
        "true" if config.tensorboard else "false",
        "--checkpoint-interval",
        "0",
    ]
    if parent is None:
        argv += ["--opponent", "random"]
    else:
        argv += [
            "--load-checkpoint",
            parent.ckpt,
            "--opponent",
            "pool",
            "--pool-episode",
        ]
        for member in sorted(roster, key=lambda m: m.id):
            argv += ["--pool-member", member.spec]
        if config.include_random > 0.0:
            argv += ["--pool-member", f"{config.include_random}@random"]
        if config.pfsp:
            argv += ["--pfsp"]
    return [*argv, *config.train_extra]


@dataclass(slots=True)
class SubprocessTrainer:
    """The default trainer: one ``7g523-train`` subprocess per fork.

    ``roster`` becomes the training pool (``--pool-member ckpt:...``) and
    ``parent`` the warm start; seeds and threads are pinned so parallel forks
    do not oversubscribe the machine.  TensorBoard stays on
    (:attr:`BootstrapConfig.tensorboard`), writing under each run's ``tb/``.
    """

    config: BootstrapConfig

    def train(
        self,
        *,
        parent: PoolMember | None,
        roster: Sequence[PoolMember],
        seed: int,
        exp_name: str,
        steps: int,
    ) -> str:
        config = self.config
        arena = Path(config.arena_dir)
        argv = train_argv(
            config,
            parent=parent,
            roster=roster,
            seed=seed,
            exp_name=exp_name,
            steps=steps,
        )

        env = os.environ.copy()
        for variable in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            env[variable] = str(config.train_threads)
        log_path = arena / "logs" / f"{exp_name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log:
            subprocess.run(argv, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)

        train_root = arena / "train"
        matches = sorted(train_root.glob(f"{exp_name}__{seed}__*"))
        if not matches:
            raise RuntimeError(f"trainer produced no run directory for {exp_name!r}")
        run_dir = max(matches, key=lambda path: path.stat().st_mtime)
        ckpt = run_dir / "agent.pt"
        if not ckpt.is_file():
            raise RuntimeError(f"run {run_dir} has no agent.pt")
        return str(ckpt)


def derive_seed(base_seed: int, *parts: object) -> int:
    """A deterministic 32-bit seed from the base seed and any child/round key."""
    payload = "\x1f".join(str(part) for part in (base_seed, *parts)).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=4).digest(), "big")


def make_play_fn(config: BootstrapConfig) -> PlayFn:
    """The default play seam: sharded, seat-paired ``ladder.play_games``."""

    def play(
        schedule: Sequence[ScheduledGame],
        entrants: Sequence[Entrant],
        out_dir: Path,
    ) -> Sequence[PlayedGame]:
        return play_games(
            schedule,
            entrants,
            rules=config.rules,
            workers=config.workers,
            device=config.device,
            results_dir=out_dir,
        )

    return play


def pool_entrants(members: Sequence[PoolMember]) -> list[Entrant]:
    """The entrant list for one schedule: pool members plus the pinned anchor."""
    entrants = [Entrant(id=member.id, spec=member.spec) for member in members]
    entrants.append(Entrant(ANCHOR_ID, "random", pinned=ANCHOR_RATING))
    return entrants


def qualify_priors(
    incumbents: Sequence[PoolMember], newcomers: Sequence[PoolMember]
) -> dict[str, Prior]:
    """Fit priors for a qualifying slate.

    Incumbents enter from their **live** rating (they are already measured), so
    the slate's evidence lands on the newcomers instead of re-deriving the pool;
    a newcomer enters from its entry prior (the parent's rating at fork time).
    """
    priors = {
        member.id: Prior(mu=member.mu, sigma=member.sigma) for member in incumbents
    }
    priors.update({member.id: member.prior() for member in newcomers})
    return priors


def _twin_games(
    master: random.Random, subject: str, opponent: str, deals: int
) -> list[ScheduledGame]:
    """``deals`` seat-paired twins: same deal, subject on each seat once."""
    games: list[ScheduledGame] = []
    for _ in range(deals):
        deal = master.randrange(1 << 32)
        games.append(ScheduledGame(deal, (subject, opponent), subject))
        games.append(ScheduledGame(deal, (opponent, subject), subject))
    return games


def plan_anchor_games(
    subject_ids: Sequence[str], *, games: int, seed: int
) -> tuple[ScheduledGame, ...]:
    """``games`` (even) twin games per subject against the pinned RandomBot."""
    _check_even(games, "anchor games")
    master = random.Random(seed)
    schedule: list[ScheduledGame] = []
    for subject in sorted(set(subject_ids)):
        schedule.extend(_twin_games(master, subject, ANCHOR_ID, games // 2))
    return tuple(schedule)


def plan_pool_games(
    subject_ids: Sequence[str],
    opponent_ids: Sequence[str],
    *,
    games: int,
    seed: int,
) -> tuple[ScheduledGame, ...]:
    """``games`` (even) twin games per subject against uniform random opponents.

    Opponents are drawn from ``opponent_ids`` minus the subject (never a
    mirror match, which carries no information about either side).
    """
    _check_even(games, "pool games")
    opponents = sorted(set(opponent_ids))
    master = random.Random(seed)
    schedule: list[ScheduledGame] = []
    for subject in sorted(set(subject_ids)):
        pool = [opponent for opponent in opponents if opponent != subject]
        if not pool:
            raise ValueError(f"pool member {subject!r} has no opponent to play")
        for _ in range(games // 2):
            opponent = master.choice(pool)
            schedule.extend(_twin_games(master, subject, opponent, 1))
    return tuple(schedule)


def plan_rating_games(
    member_ids: Sequence[str], *, games: int, anchor_games: int, seed: int
) -> tuple[ScheduledGame, ...]:
    """The per-round mass-rating schedule: pool games plus anchor games."""
    ids = sorted(set(member_ids))
    if games and len(ids) < 2:
        raise ValueError("a rating round needs at least two pool members")
    schedule: list[ScheduledGame] = []
    if games:
        schedule.extend(plan_pool_games(ids, ids, games=games, seed=seed))
    if anchor_games:
        schedule.extend(
            plan_anchor_games(ids, games=anchor_games, seed=derive_seed(seed, "anchor"))
        )
    return tuple(schedule)


def plan_qualify_games(
    newcomer_ids: Sequence[str],
    pool_ids: Sequence[str],
    *,
    games: int,
    anchor_games: int,
    seed: int,
) -> tuple[ScheduledGame, ...]:
    """The newcomer slate: pool opponents, or anchors alone while the pool is empty."""
    subjects = sorted(set(newcomer_ids))
    if not subjects:
        raise ValueError("a qualifying slate needs at least one newcomer")
    incumbents = sorted(set(pool_ids) - set(subjects))
    schedule: list[ScheduledGame] = []
    if incumbents:
        schedule.extend(plan_pool_games(subjects, incumbents, games=games, seed=seed))
        if anchor_games:
            schedule.extend(
                plan_anchor_games(
                    subjects, games=anchor_games, seed=derive_seed(seed, "anchor")
                )
            )
    else:
        # Empty pool (init): the RandomBot carries the whole slate.
        schedule.extend(plan_anchor_games(subjects, games=games, seed=seed))
    return tuple(schedule)


def select_parents(pool: Pool, config: BootstrapConfig) -> list[PoolMember]:
    """The strongest pool members as fork parents, capped by ``max_children``.

    Every occupied member is a candidate (a pool of five baselines forks five
    times, ``forks_per_parent`` each); the strongest go first once the child
    budget cannot cover everyone.  ``select_parents`` never returns an empty
    list for a non-empty pool.
    """
    max_parents = max(1, config.max_children // config.forks_per_parent)
    return pool.sorted_members()[:max_parents]


@dataclass(frozen=True, slots=True)
class _TrainJob:
    parent: PoolMember | None
    roster: tuple[PoolMember, ...]
    seed: int
    exp_name: str
    steps: int


def _train_all(
    config: BootstrapConfig, trainer: Trainer, jobs: Sequence[_TrainJob], *, log
) -> list[str]:
    """Train every fork concurrently (subprocess-bound), preserving job order."""
    results = [""] * len(jobs)
    if not jobs:
        return results

    def run(index: int, job: _TrainJob) -> tuple[int, str]:
        return (
            index,
            trainer.train(
                parent=job.parent,
                roster=job.roster,
                seed=job.seed,
                exp_name=job.exp_name,
                steps=job.steps,
            ),
        )

    with ThreadPoolExecutor(max_workers=max(1, config.train_workers)) as pool:
        futures = [
            pool.submit(run, index, job) for index, job in enumerate(jobs)
        ]
        for future in futures:
            index, ckpt = future.result()
            results[index] = ckpt
            log(f"trained {jobs[index].exp_name} -> {ckpt}")
    return results


def _pool_config(config: BootstrapConfig) -> PoolConfig:
    return PoolConfig(
        band_width=config.band_width,
        band_cap=config.band_cap,
        margin=config.margin,
    )


def _qualify(
    config: BootstrapConfig,
    newcomers: Sequence[PoolMember],
    incumbents: Sequence[PoolMember],
    play: PlayFn,
    out_dir: Path,
    *,
    seed: int,
    log,
) -> list[PoolMember]:
    """Play the newcomer slate and return rated copies of ``newcomers``."""
    schedule = plan_qualify_games(
        [member.id for member in newcomers],
        [member.id for member in incumbents],
        games=config.qualify_games,
        anchor_games=config.anchor_games,
        seed=seed,
    )
    entrants = pool_entrants([*incumbents, *newcomers])
    games = play(schedule, entrants, out_dir)
    priors = qualify_priors(incumbents, newcomers)
    fit = fit_ratings(
        games,
        anchors={ANCHOR_ID: ANCHOR_RATING},
        priors=priors,
        config=config.fit,
    )
    rated: list[PoolMember] = []
    for member in newcomers:
        rating = fit.ratings.get(member.id)
        if rating is None:
            raise RuntimeError(f"qualifying slate did not rate {member.id!r}")
        rated.append(
            replace(
                member,
                mu=rating.mu,
                sigma=rating.sigma,
                games=member.games + rating.n,
            )
        )
    log(
        f"qualified {len(rated)} newcomer(s) on {len(games)} games: "
        + ", ".join(f"{m.id} {m.mu:.0f}±{m.sigma:.0f}" for m in rated)
    )
    return rated


def _mass_rate(
    pool: Pool,
    config: BootstrapConfig,
    play: PlayFn,
    *,
    round_index: int,
    log,
) -> tuple[Fit, int]:
    """The large random-matching pass: refresh every rating (caller trims/saves).

    A degenerate one-member pool (a standalone ``rate`` on a tiny pool, or a
    pool trimmed below a pair) falls back to anchor games only.
    """
    members = pool.sorted_members()
    if len(members) < 2:
        log("mass rating: pool has one member; anchor games only")
    schedule = plan_rating_games(
        [member.id for member in members],
        games=config.mass_games if len(members) >= 2 else 0,
        anchor_games=config.anchor_games,
        seed=derive_seed(config.base_seed, "mass", round_index),
    )
    out_dir = Path(config.arena_dir) / "rounds" / f"r{round_index:03d}" / "mass"
    games = play(schedule, pool_entrants(members), out_dir)
    fit = fit_ratings(
        games,
        anchors={ANCHOR_ID: ANCHOR_RATING},
        priors=pool.priors(),
        config=config.fit,
    )
    pool.apply_fit(fit)
    log(f"rated {len(members)} member(s) on {len(games)} games")
    return fit, len(games)


def append_event(arena: str | Path, event: Mapping) -> Path:
    """Append one audit row to ``events.jsonl`` (append-only, one JSON per line)."""
    path = Path(arena) / "events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(event), ensure_ascii=False, sort_keys=True) + "\n")
    return path


def write_pool_tensorboard(
    pool: Pool, arena: str | Path, *, round_index: int, log=print
) -> int:
    """Log ``pool/rating/<id>``, ``pool/sigma/<id>`` and band sizes per round.

    Best-effort: without the train group (no torch) this returns 0 silently.
    Returns the number of scalars written.
    """
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:  # pragma: no cover - train-group extra
        log("tensorboard: torch unavailable, skipping pool scalars")
        return 0
    out = Path(arena) / "tb"
    out.mkdir(parents=True, exist_ok=True)
    written = 0
    with SummaryWriter(log_dir=str(out)) as writer:
        for member in pool.sorted_members():
            writer.add_scalar(f"pool/rating/{member.id}", member.mu, round_index)
            writer.add_scalar(f"pool/sigma/{member.id}", member.sigma, round_index)
            written += 2
        for band, members in pool.bands().items():
            writer.add_scalar(f"pool/band/{band}", float(len(members)), round_index)
            written += 1
        writer.add_scalar("pool/size", float(len(pool)), round_index)
        written += 1
    return written


def init_pool(
    config: BootstrapConfig,
    *,
    trainer: Trainer,
    play: PlayFn,
    log=print,
) -> Pool:
    """Train the seed models and rate them, or resume an existing ``pool.json``."""
    arena = Path(config.arena_dir)
    pool_path = arena / "pool.json"
    if config.resume and pool_path.is_file():
        pool = load_pool(pool_path)
        log(f"init: resuming {pool_path} ({len(pool)} members, round {pool.round})")
        return pool

    jobs = [
        _TrainJob(
            parent=None,
            roster=(),
            seed=derive_seed(config.base_seed, "seed", index),
            exp_name=f"bootstrap_seed{index}",
            steps=config.seed_steps,
        )
        for index in range(config.seed_models)
    ]
    log(f"init: training {config.seed_models} seed models x {config.seed_steps} steps")
    ckpts = _train_all(config, trainer, jobs, log=log)
    newcomers = [
        PoolMember(
            id=f"seed{index}",
            spec=f"ckpt:{ckpt}",
            mu=DEFAULT_MU,
            sigma=DEFAULT_SIGMA,
            round=0,
            seed=jobs[index].seed,
        )
        for index, ckpt in enumerate(ckpts)
    ]
    rated = _qualify(
        config,
        newcomers,
        (),
        play,
        arena / "rounds" / "r000" / "qualify",
        seed=derive_seed(config.base_seed, "qualify", 0),
        log=log,
    )
    pool = Pool(rated, config=_pool_config(config), round_=0)
    over_cap = {
        band: len(members)
        for band, members in pool.bands().items()
        if len(members) > pool.config.band_cap
    }
    if over_cap:
        log(f"init: note: bands over cap until the first round trims them: {over_cap}")
    save_pool(pool, pool_path)
    append_event(
        arena,
        {
            "event": "init",
            "round": 0,
            "members": [member.id for member in pool.sorted_members()],
            "bands": {
                str(band): [member.id for member in members]
                for band, members in pool.bands().items()
            },
        },
    )
    if config.tensorboard:
        write_pool_tensorboard(pool, arena, round_index=0, log=log)
    log(f"init: pool has {len(pool)} members ({_bands_summary(pool)})")
    return pool


def _bands_summary(pool: Pool) -> str:
    bands = pool.bands()
    return " ".join(f"{band}:[{len(members)}]" for band, members in bands.items())


def run_round(
    pool: Pool,
    config: BootstrapConfig,
    *,
    trainer: Trainer,
    play: PlayFn,
    log=print,
) -> RoundReport:
    """One fork → train → qualify → trim → rate round (mutates and saves pool)."""
    round_index = pool.round + 1
    arena = Path(config.arena_dir)
    parents = select_parents(pool, config)
    if not parents:
        raise ValueError("cannot run a round on an empty pool")
    roster = tuple(pool.sorted_members())

    child_ids: list[str] = []
    jobs: list[_TrainJob] = []
    for parent in parents:
        for fork in range(config.forks_per_parent):
            child_id = f"{parent.id}_r{round_index}f{fork}"
            child_ids.append(child_id)
            jobs.append(
                _TrainJob(
                    parent=parent,
                    roster=roster,
                    seed=derive_seed(
                        config.base_seed, "fork", round_index, parent.id, fork
                    ),
                    exp_name=f"bootstrap_r{round_index:03d}_{child_id}",
                    steps=config.train_steps,
                )
            )
    log(
        f"round {round_index}: {len(jobs)} fork(s) from {len(parents)} parent(s) "
        f"x {config.train_steps} steps"
    )
    ckpts = _train_all(config, trainer, jobs, log=log)
    newcomers = [
        PoolMember(
            id=child_id,
            spec=f"ckpt:{ckpt}",
            mu=job.parent.mu,
            sigma=DEFAULT_SIGMA,
            parent=job.parent.id,
            seed=job.seed,
            round=round_index,
            prior_mu=job.parent.mu,
            prior_sigma=DEFAULT_SIGMA,
        )
        for child_id, ckpt, job in zip(child_ids, ckpts, jobs, strict=True)
    ]

    qualified = _qualify(
        config,
        newcomers,
        roster,
        play,
        arena / "rounds" / f"r{round_index:03d}" / "qualify",
        seed=derive_seed(config.base_seed, "qualify", round_index),
        log=log,
    )
    newcomer_ids = [member.id for member in qualified]
    pool.add(qualified)
    log(f"provisional: +{len(newcomer_ids)} -> {len(pool)} members")

    _, games_played = _mass_rate(
        pool, config, play, round_index=round_index, log=log
    )
    update = pool.trim(newcomer_ids)
    kept = set(pool.members)
    added = tuple(sorted(m for m in newcomer_ids if m in kept))
    parent_mu = {member.id: member.mu for member in roster}
    gains = {
        member.id: member.mu - parent_mu[member.parent]
        for member in pool
        if member.parent in parent_mu and member.round == round_index
    }
    productive = any(gain >= config.progress_margin for gain in gains.values())
    log(
        f"trim: +{len(added)} -{len(update.evicted)} -> "
        f"{len(pool)} members ({_bands_summary(pool)})"
    )
    pool.round = round_index
    save_pool(pool, arena / "pool.json")
    append_event(
        arena,
        {
            "event": "round",
            "round": round_index,
            "parents": [parent.id for parent in parents],
            "trained": child_ids,
            "added": list(added),
            "evicted": [member.id for member in update.evicted],
            "bands": {
                str(band): [member.id for member in members]
                for band, members in pool.bands().items()
            },
            "games": games_played,
            "productive": productive,
        },
    )
    if config.tensorboard:
        write_pool_tensorboard(pool, arena, round_index=round_index, log=log)
    return RoundReport(
        round=round_index,
        parents=tuple(parent.id for parent in parents),
        trained=tuple(child_ids),
        added=added,
        evicted=tuple(member.id for member in update.evicted),
        productive=productive,
        games=games_played,
    )


def run_bootstrap(
    config: BootstrapConfig,
    *,
    trainer: Trainer | None = None,
    play: PlayFn | None = None,
    log=print,
) -> BootstrapResult:
    """Init (or resume) a pool and run ``config.rounds`` rounds with the stop rule."""
    trainer = SubprocessTrainer(config) if trainer is None else trainer
    play = make_play_fn(config) if play is None else play
    pool = init_pool(config, trainer=trainer, play=play, log=log)
    reports: list[RoundReport] = []
    idle = 0
    for _ in range(config.rounds):
        report = run_round(pool, config, trainer=trainer, play=play, log=log)
        reports.append(report)
        idle = 0 if report.productive else idle + 1
        if idle >= config.patience:
            log(f"stop: {idle} consecutive unproductive round(s)")
            break
    return BootstrapResult(pool, tuple(reports))


def rate_pool(
    pool: Pool,
    config: BootstrapConfig,
    *,
    play: PlayFn,
    log=print,
) -> Fit:
    """Rerun the mass-rating pass for the current pool, trim, and save."""
    fit, games_played = _mass_rate(
        pool, config, play, round_index=pool.round + 1, log=log
    )
    update = pool.trim()
    save_pool(pool, Path(config.arena_dir) / "pool.json")
    append_event(
        Path(config.arena_dir),
        {
            "event": "rate",
            "round": pool.round,
            "members": len(pool),
            "games": games_played,
            "evicted": [member.id for member in update.evicted],
        },
    )
    if config.tensorboard:
        write_pool_tensorboard(
            pool, Path(config.arena_dir), round_index=pool.round, log=log
        )
    return fit


def load_round_games(arena: str | Path) -> list[PlayedGame]:
    """Every recorded rated game, in schedule order (qualify then mass per round)."""
    games: list[PlayedGame] = []
    rounds_dir = Path(arena) / "rounds"
    for round_dir in sorted(rounds_dir.glob("r*")):
        for phase in ("qualify", "mass"):
            for shard in sorted((round_dir / phase).glob("shard_*.jsonl")):
                for line in shard.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    games.append(
                        PlayedGame(
                            seed=int(row["seed"]),
                            seats=tuple(row["seats"]),
                            scores=tuple(row["scores"]),
                        )
                    )
    return games


def refit_pool(pool: Pool, config: BootstrapConfig) -> Fit:
    """Audit replay: re-fit the whole game history from each member's prior."""
    games = load_round_games(config.arena_dir)
    priors = {member.id: member.prior() for member in pool}
    return fit_ratings(
        games,
        anchors={ANCHOR_ID: ANCHOR_RATING},
        priors=priors,
        config=config.fit,
    )


def status_rows(pool: Pool) -> list[dict]:
    """One printable row per member, lowest band first, strongest first within."""
    rows: list[dict] = []
    for band, members in pool.bands().items():
        for member in members:
            rows.append(
                {
                    "band": band,
                    "id": member.id,
                    "mu": round(member.mu, 1),
                    "sigma": round(member.sigma, 1),
                    "games": member.games,
                    "parent": member.parent,
                    "round": member.round,
                    "spec": member.spec,
                }
            )
    return rows

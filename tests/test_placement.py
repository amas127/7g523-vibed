"""M2/M3 tests: placement scheduling, D3 estimator, report and CLI.

The sessions here never load torch: the tiny manifest's rung specs are the
scripted bots, and the trace channel is either disabled or driven by a
monkeypatched synthetic predictor.  One test exercises the shipped M1
``prior.json`` end to end (skipped when the artifact is absent).
"""
from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

from seven523.elo import PlayedGame, Prior
from seven523.placement import (
    COLD_START_PRIOR,
    REPORT_SCHEMA,
    SESSION_SCHEMA,
    TRACE_PRIOR_SCHEMA,
    MissingCheckpointWarning,
    Opponent,
    PlacementSession,
    SessionConfig,
    TracePrior,
    band_for,
    channel_weights,
    fit_session,
    load_opponents,
    main,
    nearest_level,
    new_session_id,
    plan_deals,
    plan_seats,
    select_opponent,
    session_margin,
    session_player_labels,
    stop_reason,
)
from seven523.policies import GreedyBot
from seven523.trace import load_trace, parse_player_label

ROOT = Path(__file__).resolve().parents[1]
PRIOR_PATH = ROOT / "artifacts" / "human-elo" / "prior.json"
SILENT = lambda *args, **kwargs: None  # noqa: E731

CANDIDATES = (
    Opponent("random", 1000.0, anchor=True),
    Opponent("lvl1", 1146.3),
    Opponent("lvl2", 1219.2),
    Opponent("greedy", 1315.0, anchor=True),
    Opponent("lvl3", 1363.6),
    Opponent("lvl4", 1429.7),
)


def _manifest(levels: dict[str, float] | None = None) -> dict:
    """A torch-free manifest: every rung spec is a scripted bot."""
    levels = levels or {"random": 1000.0, "greedy": 1315.0, "lvl1": 1200.0}
    anchors = [
        {"id": "random", "elo": levels["random"]},
        {"id": "greedy", "elo": levels["greedy"]},
    ]
    subjects = []
    for id_, elo in levels.items():
        spec = "random" if id_ == "random" else "greedy"
        subjects.append({"id": id_, "spec": spec, "elo": elo, "se": 0.0 if id_ in ("random", "greedy") else 6.0})
    return {"version": 1, "levels": levels, "anchors": anchors, "subjects": subjects}


def _manifest_file(tmp_path: Path, levels: dict[str, float] | None = None) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(_manifest(levels), ensure_ascii=False), encoding="utf-8")
    return path


def _first_legal(game, state, view):
    from seven523.actions import legal_ids

    return legal_ids(view.mask)[0]


def _run_session(
    tmp_path: Path,
    *,
    games: int = 10,
    trace_prior: TracePrior | None = None,
    seed: int = 11,
    levels: dict[str, float] | None = None,
) -> tuple[PlacementSession, dict, Path]:
    opponents, anchors = load_opponents(_manifest(levels))
    directory = tmp_path / "sessions" / "s1"
    session = PlacementSession(
        opponents=opponents,
        anchors=anchors,
        directory=directory,
        session_id="s1",
        config=SessionConfig(games=games),
        rng=random.Random(seed),
        trace_prior=trace_prior,
        policy_factory=lambda opponent, game_seed: GreedyBot(),
    )
    report = session.run(lambda seat: _first_legal, print_fn=SILENT)
    assert report is not None
    return session, report, directory


# -- scheduling --------------------------------------------------------------


def test_plan_seats_is_five_five():
    seats = plan_seats(10)
    assert seats == (0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
    assert seats.count(0) == seats.count(1) == 5
    assert plan_seats(4, start=1) == (1, 0, 1, 0)
    with pytest.raises(ValueError):
        plan_seats(9)
    with pytest.raises(ValueError):
        plan_seats(10, start=2)


def test_plan_deals_are_distinct_and_deterministic():
    first = plan_deals(random.Random(0), 10)
    second = plan_deals(random.Random(0), 10)
    assert first == second
    assert len(set(first)) == 10
    assert first != plan_deals(random.Random(1), 10)


def test_session_player_labels_anchor_and_rung():
    anchor = Opponent("greedy", 1315.0, anchor=True)
    rung = Opponent("lvl3", 1363.6)
    assert session_player_labels(
        human_id="human", human_seat=0, num_players=2, opponent=anchor
    ) == ["human@seat0", "anchor:greedy@seat1"]
    assert session_player_labels(
        human_id="human", human_seat=1, num_players=2, opponent=rung
    ) == ["opponent:lvl3@seat0", "human@seat1"]
    assert parse_player_label("opponent:lvl3@seat1") == ("opponent", "lvl3", 1)


# -- opponent selection ------------------------------------------------------


def p_score(opponent: Opponent, mean: float) -> float:
    """Bernoulli ``p(1-p)`` — Fisher information up to the constant beta^2."""
    p = 1.0 / (1.0 + 10.0 ** ((opponent.elo - mean) / 400.0))
    return p * (1.0 - p)


def test_select_opponent_info_is_the_fisher_argmax():
    posterior = Prior(1230.0, 60.0)
    chosen = select_opponent(posterior, CANDIDATES, mode="info")
    assert chosen.id in {opponent.id for opponent in CANDIDATES}
    # The chosen rung maximises information; Fisher = beta^2 * p(1-p), so the
    # p(1-p) comparison is exact.
    for opponent in CANDIDATES:
        assert p_score(chosen, posterior.mean) >= p_score(opponent, posterior.mean)


def test_select_opponent_is_monotone_in_the_posterior_mean():
    picked = [
        select_opponent(Prior(mean, 40.0), CANDIDATES, mode="info").elo
        for mean in range(800, 1801, 25)
    ]
    assert picked == sorted(picked)
    assert picked[0] == 1000.0 and picked[-1] == 1429.7


def test_select_opponent_auto_uses_thompson_then_info():
    train = lambda n, mode, seed: select_opponent(  # noqa: E731
        Prior(1230.0, 60.0), CANDIDATES, n_games=n, mode=mode, rng=random.Random(seed)
    ).id
    for seed in (0, 1, 2):
        assert train(0, "auto", seed) == train(0, "thompson", seed)
        assert train(1, "auto", seed) == train(1, "thompson", seed)
        assert train(2, "auto", seed) == train(2, "info", seed)
    assert train(9, "auto", 0) == train(9, "info", 0)


def test_thompson_is_deterministic_under_a_fixed_rng():
    first = select_opponent(Prior(1230.0, 60.0), CANDIDATES, mode="thompson", rng=random.Random(7))
    second = select_opponent(Prior(1230.0, 60.0), CANDIDATES, mode="thompson", rng=random.Random(7))
    assert first == second


# -- missing checkpoints -----------------------------------------------------


def _ckpt_manifest(
    tmp_path: Path, *, missing_anchor: bool = False, all_live: bool = False
) -> dict:
    """A manifest whose rungs are ``ckpt:`` specs (``load_opponents`` only stats them)."""
    live = tmp_path / "live.pt"
    live.write_bytes(b"live")
    gone = tmp_path / "gone.pt"
    if all_live:
        gone.write_bytes(b"live")
    levels = {
        "random": 1000.0,
        "greedy": 1315.0,
        "lvl_live": 1200.0,
        "lvl_gone": 1400.0,
    }
    anchors = [
        {"id": "random", "elo": 1000.0},
        {"id": "greedy", "elo": 1315.0},
    ]
    if missing_anchor:
        anchors.append({"id": "lvl_gone", "elo": 1400.0})
    subjects = [
        {"id": "random", "spec": "random", "elo": 1000.0, "se": 0.0},
        {"id": "greedy", "spec": "greedy", "elo": 1315.0, "se": 0.0},
        {"id": "lvl_live", "spec": f"ckpt:{live}", "elo": 1200.0, "se": 5.0},
        {"id": "lvl_gone", "spec": f"ckpt:{gone}", "elo": 1400.0, "se": 5.0},
    ]
    return {"version": 1, "levels": levels, "anchors": anchors, "subjects": subjects}


def test_load_opponents_skips_missing_ckpt_and_keeps_anchors(tmp_path):
    with pytest.warns(MissingCheckpointWarning) as caught:
        opponents, anchors = load_opponents(_ckpt_manifest(tmp_path))
    pool = {opponent.id for opponent in opponents}
    assert pool == {"random", "greedy", "lvl_live"}
    assert anchors == {"random": 1000.0, "greedy": 1315.0}
    message = str(caught[0].message)
    assert "lvl_gone" in message
    assert str(tmp_path / "gone.pt") in message


def test_load_opponents_rejects_a_missing_anchor_ckpt(tmp_path):
    with pytest.raises(ValueError, match="lvl_gone"):
        load_opponents(_ckpt_manifest(tmp_path, missing_anchor=True))


def test_load_opponents_keeps_existing_ckpts(tmp_path, recwarn):
    opponents, anchors = load_opponents(_ckpt_manifest(tmp_path, all_live=True))
    assert {opponent.id for opponent in opponents} == {
        "random",
        "greedy",
        "lvl_live",
        "lvl_gone",
    }
    assert anchors == {"random": 1000.0, "greedy": 1315.0}
    missing_warnings = [
        warning
        for warning in recwarn.list
        if issubclass(warning.category, MissingCheckpointWarning)
    ]
    assert missing_warnings == []


def test_session_artifacts_omit_the_dropped_rung(tmp_path):
    with pytest.warns(MissingCheckpointWarning):
        opponents, anchors = load_opponents(_ckpt_manifest(tmp_path))
    directory = tmp_path / "sessions" / "s-ckpt"
    session = PlacementSession(
        opponents=opponents,
        anchors=anchors,
        directory=directory,
        session_id="s-ckpt",
        config=SessionConfig(games=2),
        rng=random.Random(3),
        policy_factory=lambda opponent, seed: GreedyBot(),
    )
    report = session.run(lambda seat: _first_legal, print_fn=SILENT)
    assert report is not None
    expected = {"random", "greedy", "lvl_live"}
    assert set(report["levels"]) == expected
    assert {entry["id"] for entry in report["rungs"]} == expected
    rungs = json.loads((directory / "rungs.json").read_text())
    assert {entry["id"] for entry in rungs["opponents"]} == expected
    assert "lvl_gone" not in json.dumps(rungs)


# -- stop rule, margin, weights ----------------------------------------------


def test_stop_reason_ci_or_max_games():
    config = SessionConfig(games=10, stop_ci=50.0, min_games_before_stop=1)
    assert stop_reason(0, 10.0, config) is None
    assert stop_reason(3, 50.0, config) == "ci"
    assert stop_reason(3, 50.1, config) is None
    assert stop_reason(10, 500.0, config) == "max_games"
    guarded = SessionConfig(games=10, min_games_before_stop=8)
    assert stop_reason(3, 10.0, guarded) is None
    assert stop_reason(8, 10.0, guarded) == "ci"


def test_margin_doubles_with_the_trace_prior():
    config = SessionConfig()
    assert session_margin(config, with_trace=False) == (0.143, 44.5)
    assert session_margin(config, with_trace=True) == (0.143, 89.0)


def test_channel_weights_sum_to_one_and_fade_with_games():
    assert channel_weights(None, 120.0) == {"trace": 0.0, "result": 1.0}
    weights = channel_weights(80.0, 120.0)
    assert weights["trace"] + weights["result"] == pytest.approx(1.0)
    # A sharper result channel takes more of the weight.
    sharper = channel_weights(80.0, 40.0)
    assert sharper["result"] > weights["result"]


def test_nearest_level_and_band():
    levels = {"random": 1000.0, "greedy": 1315.0, "lvl3": 1363.6}
    assert nearest_level(1330.0, levels)["id"] == "greedy"
    assert nearest_level(1390.0, levels)["id"] == "lvl3"
    assert nearest_level(1500.0, {}) is None
    assert band_for(49.9) == "placed"
    assert band_for(80.0) == "provisional"
    assert band_for(120.0) == "coarse"


# -- estimator integration ---------------------------------------------------


def test_fit_session_uses_priors_and_margin():
    games = [
        PlayedGame(seed, ("human", "greedy"), (100, 0))
        for seed in range(6)
    ]
    opponents = (Opponent("greedy", 1315.0, anchor=True),)
    fit = fit_session(
        games,
        human_id="human",
        human_prior=Prior(1350.0, 60.0),
        opponents=opponents,
        anchors={"greedy": 1315.0},
        margin=(0.143, 89.0),
    )
    human = fit.ratings["human"]
    # Six wins pull above the prior mean; the prior keeps it below a free fit.
    assert 1350.0 < human.elo < 1900.0
    # The joint posterior is sharper than the prior alone (likelihood adds info).
    assert human.se < 60.0
    # Pinned anchors keep their exact rating and zero SE.
    assert fit.ratings["greedy"].elo == 1315.0
    assert fit.ratings["greedy"].se == 0.0


def test_fit_session_rung_prior_centre_enters_the_joint_hessian():
    games = [PlayedGame(seed, ("human", "lvl3"), (100, 0)) for seed in range(4)]
    fit = fit_session(
        games,
        human_id="human",
        human_prior=Prior(1300.0, 80.0),
        opponents=(Opponent("lvl3", 1363.6),),
        anchors={},
        margin=(0.143, 44.5),
        rung_centers={"lvl3": 1400.0},
    )
    rung = fit.ratings["lvl3"]
    assert rung.n == 4
    assert rung.se < 30.0  # the games inform the free rung, too
    # The human won all four, so the rung is pulled below its prior centre.
    assert 1200.0 < rung.elo < 1400.0


# -- full sessions -----------------------------------------------------------


def test_ten_game_session_schedule_report_and_artifacts(tmp_path):
    session, report, directory = _run_session(tmp_path, games=10)

    # 10 distinct deals and a 5/5 seat split (owner D-6=(b)).
    records = session.records
    assert len(records) == 10
    assert len({record.seed for record in records}) == 10
    assert [record.seat for record in records].count(0) == 5
    assert [record.seat for record in records].count(1) == 5
    assert all(record.opponent_id in session.levels for record in records)

    # Every game persisted a trace with rung-labelled players, no --save-trace.
    for record in records:
        trace = load_trace(directory / record.trace)
        assert trace["seed"] == record.seed
        assert trace["human_seat"] == record.seat
        human_label = trace["players"][record.seat]
        opponent_label = trace["players"][1 - record.seat]
        assert human_label == f"human@seat{record.seat}"
        role, opponent_id, seat = parse_player_label(opponent_label)
        assert (role, opponent_id) in {("anchor", record.opponent_id), ("opponent", record.opponent_id)}
        assert seat == 1 - record.seat

    # Report schema: point estimate, honest CI, nearest level, provisional.
    assert report["schema"] == REPORT_SCHEMA
    assert report["games_played"] == 10
    assert report["stop"]["reason"] in {"max_games", "ci", "quit"}
    human = report["human"]
    assert human["n"] == 10
    assert human["ci95"][0] < human["elo"] < human["ci95"][1]
    assert human["ci_half_width"] == pytest.approx(session.config.z * human["se"])
    assert report["provisional"] is True  # 10 games are never ±50 (HR §10)
    assert report["band"] == "coarse"
    assert report["nearest_level"]["id"] in session.levels
    assert set(report["levels"]) == set(session.levels)
    # Two channels with parameters.
    channels = report["channels"]
    assert channels["trace"]["enabled"] is False
    assert channels["trace"]["weight"] == 0.0
    assert channels["result"]["weight"] == 1.0
    assert channels["margin"] == [0.143, 44.5]
    assert channels["weights"]["trace"] + channels["weights"]["result"] == pytest.approx(1.0)
    assert report["fit"]["converged"] is True

    session_doc = json.loads((directory / "session.json").read_text())
    assert session_doc["schema"] == SESSION_SCHEMA
    assert len(session_doc["games"]) == 10
    assert session_doc["seats"] == list(plan_seats(10))
    rungs = json.loads((directory / "rungs.json").read_text())
    assert {entry["id"] for entry in rungs["opponents"]} == set(session.levels)
    assert (directory / "report.json").exists()


def test_empty_session_returns_no_report(tmp_path):
    from seven523.play import QuitGame

    opponents, anchors = load_opponents(_manifest())
    session = PlacementSession(
        opponents=opponents,
        anchors=anchors,
        directory=tmp_path / "s1",
        session_id="s1",
        config=SessionConfig(games=2),
        rng=random.Random(0),
        policy_factory=lambda opponent, seed: GreedyBot(),
    )

    def quit_now(game, state, view):
        raise QuitGame

    assert session.run(lambda seat: quit_now, print_fn=SILENT) is None


# -- trace prior integration -------------------------------------------------


def _fake_prior() -> TracePrior:
    """A synthetic prior document; the tests patch its predictor."""
    doc = {
        "schema": TRACE_PRIOR_SCHEMA,
        "version": 1,
        "kind": "trace-s1-ridge-deshrunk",
        "prior": {"mean": 1500.0, "sd": 300.0},
        "sigma_traj": {"1": 90.0, "2": 70.0, "3": 55.0, "4": 55.0},
        "data": {"labels": {}},
    }
    return TracePrior(doc)


def test_trace_prior_channel_updates_mu_and_sigma(tmp_path, monkeypatch):
    from seven523 import prior as prior_core

    monkeypatch.setattr(
        prior_core,
        "extract_features",
        lambda trace, verify=True: {"shift": float(trace["seed"] % 100)},
    )
    monkeypatch.setattr(
        prior_core,
        "predict_elo",
        lambda doc, row, opponent_elo=None: 1000.0 + float(row["shift"]),
    )
    session, report, _directory = _run_session(
        tmp_path, games=4, trace_prior=_fake_prior()
    )
    records = session.records
    assert [record.sigma_traj for record in records] == [90.0, 70.0, 55.0, 55.0]
    expected_mu = 1000.0 + sum(record.seed % 100 for record in records) / 4.0
    assert report["channels"]["trace"]["enabled"] is True
    assert report["channels"]["trace"]["n_traces"] == 4
    assert report["channels"]["trace"]["mu_traj"] == pytest.approx(expected_mu)
    assert report["channels"]["trace"]["sigma_traj"] == 55.0
    weights = report["channels"]["weights"]
    assert weights["trace"] > 0.0
    assert weights["trace"] + weights["result"] == pytest.approx(1.0)
    assert report["channels"]["margin"] == [0.143, 89.0]


def test_trace_prior_rejects_foreign_schema():
    with pytest.raises(ValueError, match="trace-prior"):
        TracePrior({"schema": "something-else"})


@pytest.mark.skipif(not PRIOR_PATH.exists(), reason="M1 prior artifact is not present")
def test_shipped_prior_artifact_loads_and_predicts():
    prior = TracePrior.load(PRIOR_PATH)
    assert prior.cold_start == COLD_START_PRIOR
    assert prior.labels["lvl1"] > 0.0
    assert prior.prior_for_session(1200.0, 5).sd == pytest.approx(80.5646, abs=1e-3)
    assert prior.prior_for_session(1200.0, 0) == COLD_START_PRIOR
    assert prior.meta()["schema"] == TRACE_PRIOR_SCHEMA


@pytest.mark.skipif(not PRIOR_PATH.exists(), reason="M1 prior artifact is not present")
def test_session_with_shipped_prior_artifact(tmp_path):
    _session, report, _directory = _run_session(
        tmp_path, games=2, trace_prior=TracePrior.load(PRIOR_PATH)
    )
    channel = report["channels"]["trace"]
    assert channel["enabled"] is True
    assert channel["n_traces"] == 2
    assert channel["mu_traj"] == pytest.approx(channel["mu_traj"])  # finite
    assert channel["artifact"]["schema"] == TRACE_PRIOR_SCHEMA
    assert report["channels"]["margin"] == [0.143, 89.0]


# -- session ids and CLI -----------------------------------------------------


def test_new_session_id_avoids_collisions(tmp_path):
    from datetime import datetime

    moment = datetime(2026, 9, 25, 12, 0, 0)
    first = new_session_id(tmp_path, now=moment)
    assert first == "session-20260925-120000"
    (tmp_path / first).mkdir()
    assert new_session_id(tmp_path, now=moment) == "session-20260925-120000-2"


def test_cli_help_smoke(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "--simulate" in out and "--no-trace-prior" in out


def test_cli_missing_prior_reports_on_stderr(tmp_path, capsys):
    manifest = _manifest_file(tmp_path)
    code = main(
        [
            "--manifest",
            str(manifest),
            "--prior",
            str(tmp_path / "missing.json"),
            "--games",
            "2",
            "--quiet",
        ]
    )
    assert code == 1
    assert "7g523-elo:" in capsys.readouterr().err


def test_cli_simulate_smoke(tmp_path, capsys):
    manifest = _manifest_file(tmp_path)
    sessions = tmp_path / "sessions"
    code = main(
        [
            "--manifest",
            str(manifest),
            "--no-trace-prior",
            "--simulate",
            "greedy",
            "--games",
            "2",
            "--seed",
            "5",
            "--sessions-dir",
            str(sessions),
            "--quiet",
        ]
    )
    assert code == 0
    report_paths = list(sessions.rglob("report.json"))
    assert len(report_paths) == 1
    report = json.loads(report_paths[0].read_text())
    assert report["schema"] == REPORT_SCHEMA
    assert report["games_played"] == 2
    assert report["human"]["n"] == 2
    assert len(list(sessions.rglob("g*.json"))) == 2
    assert "定级：" in capsys.readouterr().out


def test_pyproject_declares_the_entry_point():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '7g523-elo = "seven523.placement:main"' in text


def test_console_script_help_works():
    script = Path(sys.executable).parent / "7g523-elo"
    if not script.exists():
        pytest.skip("7g523-elo is not installed in this environment")
    result = subprocess.run(
        [str(script), "--help"], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0
    assert "--simulate" in result.stdout

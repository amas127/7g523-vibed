"""M2/M3 tests: placement scheduling, D3 estimator, report and CLI.

The sessions here never load torch: the tiny manifest's rung specs are the
scripted bots, and the trace channel is either disabled or driven by a
monkeypatched synthetic predictor.  One test exercises the shipped M1
``prior.json`` end to end (skipped when the artifact is absent).
"""
from __future__ import annotations

import json
import math
import random
import subprocess
import sys
from pathlib import Path

import pytest
from support import FirstLegalBot

import seven523.play as play_module
from seven523.actions import PASS_ID, catalog_for
from seven523.cards import Card, Rank, Suit, card_key
from seven523.combos import ComboKind
from seven523.elo import PlayedGame, Prior
from seven523.game import Deal, Game
from seven523.placement import (
    REPORT_SCHEMA,
    SESSION_SCHEMA,
    TRACE_PRIOR_SCHEMA,
    MissingCheckpointWarning,
    Opponent,
    PlacementSession,
    ScheduledGame,
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
    session_player_labels,
    stop_reason,
)
from seven523.play import ChooserPolicy
from seven523.rules import DEFAULT_RULES, Rules, rules_id
from seven523.trace import load_trace, parse_player_label

ROOT = Path(__file__).resolve().parents[1]
PRIOR_PATH = ROOT / "artifacts" / "human-elo" / "prior.json"
SHIPPED_POOL10_MANIFEST = ROOT / "traces" / "pool10" / "manifest.json"
SILENT = lambda *args, **kwargs: None  # noqa: E731

CANDIDATES = (
    Opponent("random", 0.0, anchor=True),     # the pinned gauge (ADR-0012)
    Opponent("lvl1", 146.3),
    Opponent("lvl2", 219.2),
    Opponent("lvl3", 363.6),
    Opponent("lvl4", 429.7),
)


def _manifest(
    levels: dict[str, float] | None = None,
    *,
    rules: Rules = DEFAULT_RULES,
    include_rules_id: bool = True,
) -> dict:
    """A torch-free manifest: every rung spec is a scripted bot."""
    levels = levels or {"random": 0.0, "lvl1": 200.0}
    anchors = [{"id": "random", "mu": levels["random"]}]
    subjects = []
    for id_, mu in levels.items():
        subjects.append(
            {
                "id": id_,
                "spec": "random",
                "mu": mu,
                "sigma": 0.0 if id_ == "random" else 6.0,
            }
        )
    document = {
        "version": 1,
        "levels": levels,
        "anchors": anchors,
        "subjects": subjects,
    }
    if include_rules_id:
        document["rules_id"] = rules_id(rules)
    return document


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
        policy_factory=lambda opponent, game_seed: FirstLegalBot(),
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
    anchor = Opponent("random", 0.0, anchor=True)
    rung = Opponent("lvl3", 363.6)
    assert session_player_labels(
        human_id="human", human_seat=0, num_players=2, opponent=anchor
    ) == ["human@seat0", "anchor:random@seat1"]
    assert session_player_labels(
        human_id="human", human_seat=1, num_players=2, opponent=rung
    ) == ["opponent:lvl3@seat0", "human@seat1"]
    assert parse_player_label("opponent:lvl3@seat1") == ("opponent", "lvl3", 1)


# -- opponent selection ------------------------------------------------------


def p_score(
    opponent: Opponent, mean: float, *, sigma: float = 60.0, beta: float = 100.0
) -> float:
    """Bernoulli ``p(1-p)`` under the Plackett–Luce Gaussian, computed here
    independently of the module (Fisher information up to the constant beta^2)."""
    sd = math.sqrt(2.0 * beta**2 + sigma**2 + opponent.sigma**2)
    p = 0.5 * (1.0 + math.erf((mean - opponent.mu) / (sd * math.sqrt(2.0))))
    return p * (1.0 - p)


def test_select_opponent_info_is_the_fisher_argmax():
    posterior = Prior(1230.0, 60.0)
    chosen = select_opponent(posterior, CANDIDATES, mode="info")
    assert chosen.id in {opponent.id for opponent in CANDIDATES}
    # The chosen rung maximises information; Fisher = beta^2 * p(1-p), so the
    # p(1-p) comparison is exact.
    for opponent in CANDIDATES:
        assert p_score(chosen, posterior.mu, sigma=posterior.sigma) >= p_score(
            opponent, posterior.mu, sigma=posterior.sigma
        )


def test_select_opponent_is_monotone_in_the_posterior_mean():
    picked = [
        select_opponent(Prior(mean, 40.0), CANDIDATES, mode="info").mu
        for mean in range(-300, 801, 25)
    ]
    assert picked == sorted(picked)
    assert picked[0] == 0.0 and picked[-1] == 429.7


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
        "random": 0.0,
        "lvl_live": 200.0,
        "lvl_gone": 420.0,
    }
    anchors = [{"id": "random", "mu": 0.0}]
    if missing_anchor:
        anchors.append({"id": "lvl_gone", "mu": 420.0})
    subjects = [
        {"id": "random", "spec": "random", "mu": 0.0, "sigma": 0.0},
        {"id": "lvl_live", "spec": f"ckpt:{live}", "mu": 200.0, "sigma": 5.0},
        {"id": "lvl_gone", "spec": f"ckpt:{gone}", "mu": 420.0, "sigma": 5.0},
    ]
    return {
        "version": 1,
        "levels": levels,
        "anchors": anchors,
        "subjects": subjects,
        "rules_id": rules_id(),
    }


def test_load_opponents_skips_missing_ckpt_and_keeps_anchors(tmp_path):
    with pytest.warns(MissingCheckpointWarning) as caught:
        opponents, anchors = load_opponents(_ckpt_manifest(tmp_path))
    pool = {opponent.id for opponent in opponents}
    assert pool == {"random", "lvl_live"}
    assert anchors == {"random": 0.0}
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
        "lvl_live",
        "lvl_gone",
    }
    assert anchors == {"random": 0.0}
    missing_warnings = [
        warning
        for warning in recwarn.list
        if issubclass(warning.category, MissingCheckpointWarning)
    ]
    assert missing_warnings == []


def test_load_opponents_rejects_a_retired_bt_map_manifest():
    legacy = {
        "levels": {"random": 1000.0, "greedy": 1315.0},
        "anchors": [{"id": "random", "elo": 1000.0}],
        "subjects": [{"id": "random", "spec": "random", "elo": 1000.0}],
    }
    with pytest.raises(ValueError, match="retired BT-MAP"):
        load_opponents(legacy)


def test_load_opponents_requires_a_rules_identity():
    """A measured manifest without ``rules_id`` must not be pooled (ADR-0013)."""
    with pytest.raises(ValueError, match="rules_id"):
        load_opponents(_manifest(include_rules_id=False))


def test_load_opponents_rejects_a_different_rules_identity():
    foreign = _manifest(rules=Rules(straight_max=13))
    with pytest.raises(ValueError, match="rules_id"):
        load_opponents(foreign)
    # The same document is loadable under the rules it was measured with.
    opponents, anchors = load_opponents(foreign, rules=Rules(straight_max=13))
    assert anchors == {"random": 0.0}
    assert {opponent.id for opponent in opponents} == {"random", "lvl1"}


def test_load_opponents_accepts_the_matching_rules_identity():
    opponents, anchors = load_opponents(_manifest(), rules=DEFAULT_RULES)
    assert anchors == {"random": 0.0}
    assert {opponent.id for opponent in opponents} == {"random", "lvl1"}


def test_load_opponents_can_build_admits_injected_specs(tmp_path):
    """A gate that accepts every spec keeps the pool byte-identical."""
    manifest = _ckpt_manifest(tmp_path, all_live=True)
    specs = {subject["spec"] for subject in manifest["subjects"]}
    opponents, anchors = load_opponents(manifest, can_build=lambda spec: spec in specs)
    assert anchors == {"random": 0.0}
    assert {opponent.id for opponent in opponents} == {
        "random",
        "lvl_live",
        "lvl_gone",
    }


def test_load_opponents_can_build_skips_unbuildable_rungs(tmp_path):
    """Non-anchor rungs the gate rejects are skipped with a warning.

    ``spec=None`` is included: with a gate configured, an unbuildable spec-less
    rung is dropped too.  Without the gate the historical behaviour is
    unchanged (``spec=None`` is kept until a game tries to build it).
    """
    manifest = _ckpt_manifest(tmp_path, all_live=True)
    with pytest.warns(MissingCheckpointWarning) as caught:
        opponents, anchors = load_opponents(
            manifest, can_build=lambda spec: spec == "random"
        )
    assert anchors == {"random": 0.0}
    assert {opponent.id for opponent in opponents} == {"random"}
    message = str(caught[0].message)
    assert "lvl_live" in message and "lvl_gone" in message
    assert "cannot be built" in message

    spec_less = _manifest()
    for subject in spec_less["subjects"]:
        if subject["id"] == "lvl1":
            subject.pop("spec")
    with pytest.warns(MissingCheckpointWarning):
        gated, _anchors = load_opponents(
            spec_less, can_build=lambda spec: spec is not None
        )
    assert {opponent.id for opponent in gated} == {"random"}
    # can_build=None: behaviour unchanged, spec-less rungs stay in the pool.
    ungated, _anchors = load_opponents(spec_less)
    assert {opponent.id for opponent in ungated} == {"random", "lvl1"}


def test_load_opponents_can_build_refuses_an_unbuildable_anchor(tmp_path):
    """Anchors are load-bearing: a rejected anchor is a hard error."""
    manifest = _ckpt_manifest(tmp_path, all_live=True)
    with pytest.raises(ValueError, match="not buildable"):
        load_opponents(manifest, can_build=lambda spec: False)


def test_load_opponents_search_spec_needs_the_injected_gate():
    """A published ``rolloutt:`` rung: ungated it is admitted (historical);
    the stock grammar gate drops it with the shared skip warning."""
    from seven523.policies import buildable_by_grammar

    manifest = _manifest(levels={"random": 0.0, "search_leafq": 260.03})
    for subject in manifest["subjects"]:
        if subject["id"] == "search_leafq":
            subject["spec"] = "rolloutt:runs/ei2_value_t5/t_leafq/critic.pt"
    ungated, _anchors = load_opponents(manifest)
    assert {opponent.id for opponent in ungated} == {"random", "search_leafq"}
    with pytest.warns(MissingCheckpointWarning) as caught:
        gated, _anchors = load_opponents(manifest, can_build=buildable_by_grammar)
    assert {opponent.id for opponent in gated} == {"random"}
    assert "search_leafq" in str(caught[0].message)
    assert "cannot be built" in str(caught[0].message)


def test_session_scopes_the_trace_prior_per_played_row(tmp_path, monkeypatch):
    """Merged pool: raw rows keep the trace prior; a played search rung's row
    is excluded with per-game provenance instead of being extrapolated
    (ADR-0013)."""
    from seven523 import prior as prior_core

    monkeypatch.setattr(
        prior_core,
        "extract_features",
        lambda trace, verify=True: {"shift": 1.0},
    )
    monkeypatch.setattr(
        prior_core,
        "predict_elo",
        lambda doc, row, opponent_elo=None: 400.0,
    )
    session = PlacementSession(
        opponents=(
            Opponent("random", 0.0, anchor=True),
            Opponent(
                "search_leafq",
                260.03,
                3.855,
                "rolloutt:runs/ei2_value_t5/t_leafq/critic.pt",
            ),
        ),
        anchors={"random": 0.0},
        directory=tmp_path / "s",
        session_id="s",
        config=SessionConfig(games=2),
        trace_prior=_fake_prior(),
        policy_factory=lambda opponent, seed: FirstLegalBot(),
    )
    session.begin()
    raw = ScheduledGame(index=0, opponent=session.opponents[0], seat=0, seed=101)
    session.commit_game(
        raw, scores=(100, 0), trace={"seed": 101}, trace_path=Path("g0.json")
    )
    assert session.records[0].prior_off_reason is None
    assert session.prior_off_reason is None
    assert len(session._rows) == 1
    assert session.trace_prior is not None
    search = ScheduledGame(index=1, opponent=session.opponents[1], seat=1, seed=102)
    session.commit_game(
        search, scores=(0, 100), trace={"seed": 102}, trace_path=Path("g1.json")
    )
    # The search row was excluded and the reason recorded per game and on the
    # session; the raw row still informs the trace channel.
    assert len(session._rows) == 1
    assert session.records[1].prior_off_reason is not None
    assert "search_leafq" in session.records[1].prior_off_reason
    assert "search_leafq" in (session.prior_off_reason or "")
    report = session.report()
    trace = report["channels"]["trace"]
    assert trace["enabled"] is True
    assert trace["n_traces"] == 1
    assert trace["excluded_games"] == [1]
    assert trace["prior_off_reason"] == session.prior_off_reason
    assert trace["mu_traj"] == pytest.approx(400.0)
    document = session._session_document()
    assert document["prior_off_reason"] == session.prior_off_reason
    assert document["games"][1]["prior_off_reason"] == session.records[1].prior_off_reason


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
        policy_factory=lambda opponent, seed: FirstLegalBot(),
    )
    report = session.run(lambda seat: _first_legal, print_fn=SILENT)
    assert report is not None
    expected = {"random", "lvl_live"}
    assert set(report["levels"]) == expected
    assert {entry["id"] for entry in report["rungs"]} == expected
    rungs = json.loads((directory / "rungs.json").read_text())
    assert {entry["id"] for entry in rungs["opponents"]} == expected
    assert "lvl_gone" not in json.dumps(rungs)


# -- stop rule, weights -------------------------------------------------------


def test_stop_reason_ci_or_max_games():
    config = SessionConfig(games=10, stop_ci=50.0, min_games_before_stop=1)
    assert stop_reason(0, 10.0, config) is None
    assert stop_reason(3, 50.0, config) == "ci"
    assert stop_reason(3, 50.1, config) is None
    assert stop_reason(10, 500.0, config) == "max_games"
    guarded = SessionConfig(games=10, min_games_before_stop=8)
    assert stop_reason(3, 10.0, guarded) is None
    assert stop_reason(8, 10.0, guarded) == "ci"


def test_channel_weights_sum_to_one_and_fade_with_games():
    assert channel_weights(None, 120.0) == {"trace": 0.0, "result": 1.0}
    weights = channel_weights(80.0, 120.0)
    assert weights["trace"] + weights["result"] == pytest.approx(1.0)
    # A sharper result channel takes more of the weight.
    sharper = channel_weights(80.0, 40.0)
    assert sharper["result"] > weights["result"]


def test_nearest_level_and_band():
    levels = {"random": 0.0, "lvl1": 200.0, "lvl3": 363.6}
    assert nearest_level(150.0, levels)["id"] == "lvl1"
    assert nearest_level(360.0, levels)["id"] == "lvl3"
    assert nearest_level(1500.0, {}) is None
    assert band_for(49.9) == "placed"
    assert band_for(80.0) == "provisional"
    assert band_for(120.0) == "coarse"


# -- estimator integration ---------------------------------------------------


def test_fit_session_uses_priors_and_pins_anchors():
    games = [
        PlayedGame(seed, ("human", "random"), (100, 0))
        for seed in range(6)
    ]
    opponents = (Opponent("random", 0.0, anchor=True),)
    fit = fit_session(
        games,
        human_id="human",
        human_prior=Prior(350.0, 60.0),
        opponents=opponents,
        anchors={"random": 0.0},
    )
    human = fit.ratings["human"]
    # Six wins pull above the prior mean; the prior keeps it below a free fit.
    assert 350.0 < human.mu < 900.0
    # The replay is sharper than the prior alone.
    assert human.sigma < 60.0
    # Pinned anchors keep their exact rating and zero sigma.
    assert fit.ratings["random"].mu == 0.0
    assert fit.ratings["random"].sigma == 0.0


def test_fit_session_rung_prior_centre_informs_the_rung():
    games = [PlayedGame(seed, ("human", "lvl3"), (100, 0)) for seed in range(4)]

    def rung_mu(center: float) -> float:
        fit = fit_session(
            games,
            human_id="human",
            human_prior=Prior(300.0, 80.0),
            opponents=(Opponent("lvl3", 363.6),),
            anchors={},
            rung_centers={"lvl3": center},
        )
        rung = fit.ratings["lvl3"]
        assert rung.n == 4
        assert rung.sigma < 31.0  # the prior width survives; the games move mu
        return rung.mu

    # The human won all four; the rung's free prior centre is load-bearing.
    assert rung_mu(100.0) < rung_mu(500.0)
    assert 100.0 < rung_mu(300.0) < 500.0


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
    assert human["ci95"][0] < human["mu"] < human["ci95"][1]
    assert human["ci_half_width"] == pytest.approx(session.config.z * human["sigma"])
    assert report["provisional"] is True  # 10 games are never ±50 (HR §10)
    assert report["band"] == "coarse"
    assert report["nearest_level"]["id"] in session.levels
    assert set(report["levels"]) == set(session.levels)
    # Two channels with parameters.
    channels = report["channels"]
    assert channels["trace"]["enabled"] is False
    assert channels["trace"]["weight"] == 0.0
    assert channels["result"]["weight"] == 1.0
    assert channels["weights"]["trace"] + channels["weights"]["result"] == pytest.approx(1.0)
    assert report["fit"]["estimator"] == "openskill-plackett-luce"
    assert report["fit"]["games"] == 10

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
        policy_factory=lambda opponent, seed: FirstLegalBot(),
    )

    def quit_now(game, state, view):
        raise QuitGame

    assert session.run(lambda seat: quit_now, print_fn=SILENT) is None


# -- bot-dig regression through a session -------------------------------------

#: The crafted bot hand: three suits of 5/10/K (the fourth suit stays with the
#: human).  The bot leads one card per trick, banks 65 over eight tricks, then
#: its last K digs the 25 points still in the human's hand:
#: 65 (banked) + 10 (trick) + 25 (swept) = 100.
_BOT_DIG_RANKS = [Rank.R5] * 3 + [Rank.R10] * 3 + [Rank.RK] * 3


def _single_id(rank: Rank) -> int:
    return next(
        index
        for index, action in enumerate(catalog_for(DEFAULT_RULES))
        if action.kind is ComboKind.SINGLE and action.ranks == (rank,)
    )


class _ScriptedLeads:
    """Leads a fixed sequence of single-card templates (the crafted bot)."""

    def __init__(self, ranks):
        self._ids = [_single_id(rank) for rank in ranks]

    def act(self, view):
        return self._ids.pop(0), None


def _bot_dig_deal(human_seat: int) -> Deal:
    """2-seat deal where the opponent digs with its last card."""
    human = frozenset(
        {
            Card(Rank.RK, Suit.SPADE),
            Card(Rank.R10, Suit.SPADE),
            Card(Rank.R5, Suit.SPADE),
        }
    )
    bot = frozenset(
        Card(rank, suit)
        for rank in (Rank.R5, Rank.R10, Rank.RK)
        for suit in (Suit.CLUB, Suit.DIAMOND, Suit.HEART)
    )
    hands = (human, bot) if human_seat == 0 else (bot, human)
    return Deal(
        hands=hands,
        draw_pile=(),
        revealed=tuple(min(hand, key=card_key) for hand in hands),
        starter=1 - human_seat,
    )


def _pass_chooser(game, state, view):
    return PASS_ID


def test_session_records_the_bot_dig_arithmetic(tmp_path, monkeypatch):
    """The session's ``PlayedGame.scores`` carry the dug-trick arithmetic.

    The passing human banks 0; the opponent banks 65 + 10 (final trick) + 25
    (the points swept from the human's hand) = 100, in both seat assignments.
    """
    real_match = play_module.Match

    class CraftedMatch(real_match):
        def __init__(self, rules, policies, *, state=None, rng=None):
            if state is None:
                human_seat = next(
                    seat
                    for seat, policy in enumerate(policies)
                    if isinstance(policy, ChooserPolicy)
                )
                state = Game(rules).restore(_bot_dig_deal(human_seat))
            super().__init__(rules, policies, state=state, rng=rng)

    monkeypatch.setattr(play_module, "Match", CraftedMatch)

    opponents, anchors = load_opponents(_manifest())
    directory = tmp_path / "sessions" / "dig"
    session = PlacementSession(
        opponents=opponents,
        anchors=anchors,
        directory=directory,
        session_id="dig",
        config=SessionConfig(games=2),
        rng=random.Random(0),
        policy_factory=lambda opponent, seed: _ScriptedLeads(_BOT_DIG_RANKS),
    )
    report = session.run(lambda seat: _pass_chooser, print_fn=SILENT)
    assert report is not None
    assert len(session.games) == 2

    for game, record, seat in zip(session.games, session.records, plan_seats(2)):
        expected = [0, 100] if seat == 0 else [100, 0]
        assert list(game.scores) == expected
        assert record.scores == tuple(expected)
        assert game.scores[seat] == 0  # the passing human banked nothing
        assert game.scores[1 - seat] == 100  # 65 banked + 10 trick + 25 sweep
        trace = load_trace(directory / record.trace)
        assert trace["final_scores"] == expected
        assert any(step["dug"] for step in trace["steps"])


# -- trace prior integration -------------------------------------------------


def _fake_prior() -> TracePrior:
    """A synthetic prior document; the tests patch its predictor."""
    doc = {
        "schema": TRACE_PRIOR_SCHEMA,
        "version": 1,
        "kind": "trace-s1-ridge-deshrunk",
        "prior": {"mean": 500.0, "sd": 300.0},
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
        lambda doc, row, opponent_elo=None: 500.0 + float(row["shift"]),
    )
    session, report, _directory = _run_session(
        tmp_path, games=4, trace_prior=_fake_prior()
    )
    records = session.records
    assert [record.sigma_traj for record in records] == [90.0, 70.0, 55.0, 55.0]
    expected_mu = 500.0 + sum(record.seed % 100 for record in records) / 4.0
    assert report["channels"]["trace"]["enabled"] is True
    assert report["channels"]["trace"]["n_traces"] == 4
    assert report["channels"]["trace"]["mu_traj"] == pytest.approx(expected_mu)
    assert report["channels"]["trace"]["sigma_traj"] == 55.0
    weights = report["channels"]["weights"]
    assert weights["trace"] > 0.0
    assert weights["trace"] + weights["result"] == pytest.approx(1.0)


def test_trace_prior_rejects_foreign_schema():
    with pytest.raises(ValueError, match="trace-prior"):
        TracePrior({"schema": "something-else"})


@pytest.mark.skipif(not PRIOR_PATH.exists(), reason="M1 prior artifact is not present")
def test_shipped_prior_artifact_loads_and_predicts():
    prior = TracePrior.load(PRIOR_PATH)
    # The shipped artifact is the 2026-09-27 w2m/T23 rerating on the
    # RandomBot=0 OpenSkill scale (ADR-0012) with the deferred placement
    # rescale applied, so its cold start is the module constant rather than a
    # stale literal.
    from seven523 import prior as prior_core

    cold = Prior(
        prior_core.COLD_START_PRIOR["mean"], prior_core.COLD_START_PRIOR["sd"]
    )
    assert prior.cold_start == cold
    assert prior.labels["lvl1"] > 0.0
    # Read the trajectory table from the artifact itself instead of pinning a
    # number that changes at every recalibration (the w2m/T23 rerating moved
    # sigma(5) to the current table value).
    sigma_5 = float(prior.doc["sigma_traj"]["5"])
    assert prior.prior_for_session(1200.0, 5).sigma == pytest.approx(sigma_5)
    assert 0.0 < sigma_5 < cold.sigma
    assert prior.prior_for_session(1200.0, 0) == cold
    assert prior.meta()["schema"] == TRACE_PRIOR_SCHEMA


@pytest.mark.skipif(
    not SHIPPED_POOL10_MANIFEST.exists(),
    reason="the shipped pool10 manifest is not present",
)
def test_shipped_pool10_mirror_is_current_and_accepted():
    # The w2m/T23 publication mirrors the study manifest into traces/pool10;
    # both copies must stay byte-identical and carry the revision-3 identity
    # so placement accepts them.  A legacy revision-2 manifest is refused by
    # the synthetic-manifest tests above; this one guards the shipped mirror.
    study_text = (ROOT / "traces" / "study" / "manifest.json").read_text(
        encoding="utf-8"
    )
    pool10_text = SHIPPED_POOL10_MANIFEST.read_text(encoding="utf-8")
    assert pool10_text == study_text
    manifest = json.loads(pool10_text)
    assert manifest["rules_id"] == rules_id(DEFAULT_RULES)
    assert manifest["estimator"]["kind"] == "probit-mle"
    opponents, anchors = load_opponents(manifest)
    assert anchors == {"random": 0.0}
    assert {opponent.id for opponent in opponents} == set(manifest["levels"])


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
            "random",
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


def _search_manifest_file(tmp_path: Path) -> Path:
    """A manifest whose only rung is the published ``rolloutt:`` search spec."""
    manifest = _manifest(levels={"random": 0.0, "search_leafq": 260.03})
    for subject in manifest["subjects"]:
        if subject["id"] == "search_leafq":
            subject["spec"] = "rolloutt:runs/ei2_value_t5/t_leafq/critic.pt"
    path = tmp_path / "search-manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_cli_skips_a_search_rung_without_a_factory(tmp_path):
    """No injected factory: the rung is dropped with a warning, session runs."""
    manifest = _search_manifest_file(tmp_path)
    sessions = tmp_path / "sessions"
    with pytest.warns(MissingCheckpointWarning, match="search_leafq"):
        code = main(
            [
                "--manifest",
                str(manifest),
                "--no-trace-prior",
                "--simulate",
                "random",
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
    rungs = json.loads(next(sessions.rglob("rungs.json")).read_text())
    assert {entry["id"] for entry in rungs["opponents"]} == {"random"}
    assert rungs["prior_off_reason"] is None


@pytest.mark.skipif(not PRIOR_PATH.exists(), reason="M1 prior artifact is not present")
def test_cli_injected_factory_merges_the_pool_and_scopes_the_prior(tmp_path):
    """A programmatically injected factory opts the search rung into the one
    pool; the trace prior stays on and only a played search row is excluded."""
    manifest = _search_manifest_file(tmp_path)
    sessions = tmp_path / "sessions"
    seen: list[tuple[str, int]] = []

    def factory(opponent: Opponent, seed: int):
        seen.append((opponent.id, seed))
        return FirstLegalBot()

    code = main(
        [
            "--manifest",
            str(manifest),
            "--prior",
            str(PRIOR_PATH),
            "--simulate",
            "random",
            "--games",
            "4",
            "--seed",
            "5",
            "--sessions-dir",
            str(sessions),
            "--quiet",
        ],
        policy_factory=factory,
    )
    assert code == 0
    rungs = json.loads(next(sessions.rglob("rungs.json")).read_text())
    assert {entry["id"] for entry in rungs["opponents"]} == {"random", "search_leafq"}
    session = json.loads(next(sessions.rglob("session.json")).read_text())
    report = json.loads(next(sessions.rglob("report.json")).read_text())
    played = [game["opponent_id"] for game in session["games"]]
    excluded = [
        game["index"]
        for game in session["games"]
        if game["prior_off_reason"] is not None
    ]
    # Raw rows keep the channel; only a played search rung is excluded.
    assert report["channels"]["trace"]["enabled"] is True
    assert len(excluded) == played.count("search_leafq")
    assert report["channels"]["trace"]["excluded_games"] == excluded
    assert report["channels"]["trace"]["n_traces"] == len(played) - len(excluded)
    if excluded:
        assert session["prior_off_reason"] == report["prior_off_reason"]
        assert rungs["prior_off_reason"] == session["prior_off_reason"]
        assert all(
            game["prior_off_reason"]
            for game in session["games"]
            if game["index"] in excluded
        )
    else:
        assert session["prior_off_reason"] is None
    assert all(opponent_id in {"random", "search_leafq"} for opponent_id, _ in seen)


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

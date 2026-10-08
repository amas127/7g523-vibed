"""``7g523-web`` tests: the JSON projection, the table session, HTTP and CLI.

Everything here is torch-free and offline: the pool is injected (a ``random``
anchor plus one scripted rung), the trace channel is disabled, and artifacts go
to ``tmp_path``.  The free-play traces are replayed through the engine's own
replay path so the step-driven recording is verified, not just parsed.
"""
from __future__ import annotations

import json
import random
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from seven523.cards import Card, Rank, Suit
from seven523.combos import classify
from seven523.placement import TRACE_PRIOR_SCHEMA, Opponent, TracePrior
from seven523.play import replay_trace
from seven523.policies import RandomBot
from seven523.rules import DEFAULT_RULES, rules_id
from seven523.trace import load_trace
from seven523.web import (
    PROTO_VERSION,
    TableSession,
    WebConfig,
    build_parser,
    hand_key,
)
from seven523.web.server import make_handler
from seven523.web.view import combo_label

SILENT = lambda *args, **kwargs: None  # noqa: E731
ROOT = Path(__file__).resolve().parents[1]

#: Deal-twin metadata mirroring the run-local launcher's pin.
TWIN_BASE_SPEC = "ckpt:runs/ei2_value_t5/t_leafq/critic.pt"
TWIN_PARAMS = {
    "trunc_ply": 5,
    "rollout_k": 32,
    "max_candidates": 6,
    "max_rollout_ply": 400,
    "value_ckpt": "runs/ei2_value_t5/t_leafq/critic.pt",
}
TWIN_META = {
    "label": "deal-twin 测试",
    "note": "测试用 twin 元数据",
    "raw": {"id": "critic_raw", "spec": TWIN_BASE_SPEC, "label": "raw"},
    "search": {
        "id": "o4lite_t5k32",
        "base_spec": TWIN_BASE_SPEC,
        "identity": "rolloutt:runs/ei2_value_t5/t_leafq/critic.pt",
        "params": dict(TWIN_PARAMS),
        "value_ckpt": TWIN_PARAMS["value_ckpt"],
        "label": "搜索 t5·K32·C6",
    },
    "pairs_options": [10, 20, 30, 50],
    "default_pairs": 30,
    "min_pairs": 30,
    "bootstrap": 4000,
    "bootstrap_seed": 523,
    "confidence": 0.95,
}


def _twin_factory(seen=None):
    def factory(spec, rules, seed, search):
        if seen is not None:
            seen.append((spec, dict(search), seed))
        return RandomBot(random.Random(seed))

    return factory


RATED_SPEC = "rolloutt:runs/ei2_value_t5/t_leafq/critic.pt"
SEARCH_OPTIONS = {
    "trunc_ply": {"min": 0, "max": 40, "default": 5},
    "rollout_k": {"min": 1, "max": 64, "default": 32},
}


def _minimal_prior() -> TracePrior:
    """A schema-valid prior whose predictor is never reached by these tests."""
    return TracePrior(
        {
            "schema": TRACE_PRIOR_SCHEMA,
            "version": 1,
            "prior": {"mean": 500.0, "sd": 300.0},
            "sigma_traj": {"1": 90.0, "2": 70.0},
            "data": {"labels": {}},
        }
    )


def _patch_no_torch_policy(monkeypatch, calls=None) -> None:
    """Route raw ``policy_from_spec`` calls to a scripted bot, pretend torch."""
    import seven523.web.table as table_module

    def fake(spec, rules=DEFAULT_RULES, seed=None, device="cpu"):
        if calls is not None:
            calls.append(spec)
        return RandomBot(random.Random(seed))

    monkeypatch.setattr(table_module, "torch_available", lambda: True)
    monkeypatch.setattr(table_module, "policy_from_spec", fake)


def _twin_table(tmp_path: Path, *, factory=None, twin=TWIN_META) -> TableSession:
    return _table(
        tmp_path,
        policy_factory=factory or _twin_factory(),
        twin=twin,
    )


def _patch_twin(monkeypatch, raw_calls=None) -> None:
    """Route the raw arm to a scripted bot and pretend torch is installed."""
    import seven523.twin as twin_module
    import seven523.web.table as table_module

    def fake(spec, rules=DEFAULT_RULES, seed=None, device="cpu"):
        if raw_calls is not None:
            raw_calls.append(spec)
        return RandomBot(random.Random(seed))

    monkeypatch.setattr(twin_module, "policy_from_spec", fake)
    monkeypatch.setattr(table_module, "torch_available", lambda: True)


def _table(
    tmp_path: Path,
    *,
    opponents=None,
    prior=None,
    policy_factory=None,
    search=None,
    twin=None,
    can_build_spec=None,
    plugin=None,
) -> TableSession:
    """A torch-free table at ``tmp_path`` with an injected pool."""
    opponents = opponents or (
        Opponent("random", 0.0, anchor=True),
        Opponent("lvl1", 100.0, spec="random"),
    )
    anchors = {"random": 0.0}
    web_config = WebConfig(
        manifest=tmp_path / "manifest.json",
        prior=None,
        sessions_dir=tmp_path / "sessions",
        twins_dir=tmp_path / "twins",
        free_traces_dir=tmp_path / "web",
        seed=0,
        policy_factory=policy_factory,
        search=search,
        twin=twin,
        can_build_spec=can_build_spec,
        plugin=plugin,
    )
    return TableSession(
        web_config,
        opponents=opponents,
        anchors=anchors,
        prior=prior,
        rng=random.Random(0),
        run_id="test",
    )


def _first_legal(table: TableSession) -> None:
    legal = table.snapshot()["legal"]
    table.apply_action({"action_id": legal[0]["action_id"]})


def _play_free_to_end(table: TableSession, *, opponent_id="random", seat=0) -> None:
    table.start({"mode": "free", "opponent_id": opponent_id, "seat": seat, "seed": 7})
    guard = 0
    while table.phase == "human":
        _first_legal(table)
        guard += 1
        assert guard < 200, "game did not finish"
    assert table.phase == "game_over"


# -- projection --------------------------------------------------------------


def test_hand_key_is_natural_order():
    hand = [
        Card(Rank.R2, Suit.DIAMOND),
        Card(Rank.RK, Suit.CLUB),
        Card(Rank.R3, Suit.SPADE),
        Card(Rank.R5, Suit.HEART),
        Card(Rank.BIG_JOKER),
        Card(Rank.SMALL_JOKER),
        Card(Rank.RA, Suit.CLUB),
    ]
    assert [str(card) for card in sorted(hand, key=hand_key)] == [
        "♠3",
        "♥5",
        "♣K",
        "♣A",
        "♦2",
        "小王",
        "大王",
    ]


def test_combo_label_reads_runs_along_the_cycle():
    combo = classify(
        (
            Card(Rank.RQ, Suit.SPADE),
            Card(Rank.RK, Suit.HEART),
            Card(Rank.RA, Suit.CLUB),
            Card(Rank.R2, Suit.DIAMOND),
            Card(Rank.R3, Suit.SPADE),
        ),
        DEFAULT_RULES,
    )
    assert combo is not None
    assert combo_label(combo) == "顺子 Q-K-A-2-3"


def test_counter_is_public_information(tmp_path):
    table = _table(tmp_path)
    table.start({"mode": "free", "opponent_id": "random", "seat": 0, "seed": 7})
    snap = table.snapshot()
    counter = snap["counter"]
    total = sum(row["total"] for row in counter["rows"])
    unseen = sum(row["unseen"] for row in counter["rows"])
    assert total == 54
    assert unseen == counter["unseen_cards"]
    # Seen = own hand ∪ played cards ∪ every reveal (the view's public set).
    seen = {card["label"] for card in snap["your_hand"]}
    seen |= {card["label"] for play in snap["plays"] for card in play["cards"]}
    seen |= {entry["card"]["label"] for entry in snap["revealed"]}
    assert total - unseen == len(seen)
    # Mine counts reconcile with the rendered hand.
    mine = {row["rank"]: row["mine"] for row in counter["rows"]}
    for card in snap["your_hand"]:
        assert mine[card["rank"]] >= 1


def test_snapshot_shapes_are_stable(tmp_path):
    table = _table(tmp_path)
    table.start({"mode": "free", "opponent_id": "random", "seat": 0, "seed": 7})
    snap = table.snapshot()
    assert snap["phase"] == "human"
    assert snap["tricks_completed"] == 0
    assert [seat["seat"] for seat in snap["revealed"]] == [0, 1]
    # PASS is offered only while following an incumbent, and sorts last.
    if any(action["kind_key"] == "pass" for action in snap["legal"]):
        assert snap["legal"][-1]["kind_key"] == "pass"
    # The rendered hand is exactly the engine hand in natural order.
    assert [card["label"] for card in snap["your_hand"]] == [
        str(card) for card in sorted(table.view.hand, key=hand_key)
    ]
    assert table.config_document()["proto_version"] == PROTO_VERSION


# -- free play ---------------------------------------------------------------


def test_free_game_records_a_replayable_trace(tmp_path):
    table = _table(tmp_path)
    _play_free_to_end(table)
    snap = table.snapshot()
    trace = load_trace(snap["result"]["trace_path"])
    assert trace["human_seat"] == 0
    assert trace["players"][0] == "human@seat0"
    assert trace["players"][1].startswith("anchor:random@seat1")
    assert trace["final_scores"] == snap["result"]["scores"]
    assert snap["result"]["steps"] == len(trace["steps"]) > 0
    assert replay_trace(trace, print_fn=SILENT) is True


def test_seed_reproduces_the_deal(tmp_path):
    first = _table(tmp_path)
    first.start({"mode": "free", "opponent_id": "random", "seat": 0, "seed": 7})
    second = _table(tmp_path)
    second.start({"mode": "free", "opponent_id": "random", "seat": 0, "seed": 7})
    hand = lambda table: [card["label"] for card in table.snapshot()["your_hand"]]  # noqa: E731
    assert hand(first) == hand(second)
    assert first.seed == second.seed == 7


def test_free_game_seat_one_uses_the_same_shape(tmp_path):
    table = _table(tmp_path)
    _play_free_to_end(table, seat=1)
    snap = table.snapshot()
    assert snap["human_seat"] == 1
    assert snap["names"] == ["random", "你"]
    trace = load_trace(snap["result"]["trace_path"])
    assert trace["human_seat"] == 1
    assert replay_trace(trace, print_fn=SILENT) is True


def test_trick_counter_counts_completed_tricks(tmp_path):
    table = _table(tmp_path)
    table.start({"mode": "free", "opponent_id": "random", "seat": 0, "seed": 7})
    guard = 0
    while table.phase == "human" and table.snapshot()["tricks_completed"] < 1:
        _first_legal(table)
        guard += 1
        assert guard < 200
    assert table.phase in ("human", "game_over")
    assert table.snapshot()["tricks_completed"] >= 1


def test_illegal_action_is_refused(tmp_path):
    table = _table(tmp_path)
    table.start({"mode": "free", "opponent_id": "random", "seat": 0, "seed": 7})
    legal = {action["action_id"] for action in table.snapshot()["legal"]}
    illegal = next(
        action_id
        for action_id in range(len(table.game.catalog))
        if action_id not in legal
    )
    with pytest.raises(Exception) as excinfo:
        table.apply_action({"action_id": illegal})
    assert "非法出牌" in str(excinfo.value)
    # The game survives the refused action.
    assert table.phase == "human"


# -- placement ---------------------------------------------------------------


def test_placement_session_runs_to_a_report(tmp_path):
    table = _table(tmp_path)
    table.start({"mode": "placement", "games": 2, "use_prior": False, "seed": 3})
    guard = 0
    while table.phase != "session_over":
        assert table.phase in ("human", "round_over")
        if table.phase == "round_over":
            table.continue_placement()
        else:
            _first_legal(table)
        guard += 1
        assert guard < 400, "placement session did not finish"
    progress = table.snapshot()["placement"]
    assert progress["games_played"] == 2
    report = progress["report"]
    assert report["games_played"] == 2
    assert report["human"]["n"] == 2
    assert report["channels"]["trace"]["enabled"] is False
    directory = Path(progress["session_dir"])
    assert (directory / "session.json").is_file()
    assert (directory / "rungs.json").is_file()
    assert (directory / "report.json").is_file()
    session_doc = json.loads((directory / "session.json").read_text())
    assert len(session_doc["games"]) == 2
    # Every placement trace replays through the engine.
    for record in session_doc["games"]:
        replay_trace(load_trace(directory / record["trace"]), print_fn=SILENT)


def test_quit_mid_free_game_marks_it_aborted(tmp_path):
    table = _table(tmp_path)
    table.start({"mode": "free", "opponent_id": "random", "seat": 0, "seed": 7})
    table.quit_session()
    snap = table.snapshot()
    assert snap["phase"] == "aborted"
    assert snap["result"]["aborted"] is True


def test_ckpt_opponent_needs_torch(tmp_path, monkeypatch):
    import seven523.web.table as table_module

    monkeypatch.setattr(table_module, "torch_available", lambda: False)
    table = _table(
        tmp_path,
        opponents=(
            Opponent("random", 0.0, anchor=True),
            Opponent("ckpt", 100.0, spec="ckpt:runs/nope/agent.pt"),
        ),
    )
    with pytest.raises(Exception) as excinfo:
        table.start({"mode": "free", "opponent_id": "ckpt", "seat": 0})
    assert "需要 torch" in str(excinfo.value)


def test_free_play_uses_the_policy_factory_seam(tmp_path, monkeypatch):
    import seven523.web.table as table_module

    monkeypatch.setattr(table_module, "torch_available", lambda: True)
    calls: list[tuple[str, dict]] = []

    def factory(spec, rules, seed, search):
        calls.append((spec, dict(search)))
        return RandomBot(random.Random(seed))

    search_meta = {
        "label": "t/K 可选",
        "options": {
            "trunc_ply": {"min": 0, "max": 40, "default": 5},
            "rollout_k": {"min": 1, "max": 64, "default": 16},
        },
    }
    table = _table(
        tmp_path,
        opponents=(
            Opponent("random", 0.0, anchor=True),
            Opponent("wrapped", 100.0, spec="ckpt:runs/w2m/agent.pt"),
        ),
        policy_factory=factory,
        search=search_meta,
    )
    table.start({"mode": "free", "opponent_id": "wrapped", "seat": 0, "seed": 7})
    assert [spec for spec, _ in calls] == ["ckpt:runs/w2m/agent.pt"]
    # No request parameters -> the declared defaults fill in.
    assert calls[0][1] == {"trunc_ply": 5, "rollout_k": 16}
    assert table.snapshot()["search"] == {"trunc_ply": 5, "rollout_k": 16}
    assert table.config_document()["search"] == search_meta
    # A stock table reports no search wrapper.
    assert _table(tmp_path).config_document()["search"] is None


def test_free_play_search_request_reaches_the_factory(tmp_path, monkeypatch):
    import seven523.web.table as table_module

    monkeypatch.setattr(table_module, "torch_available", lambda: True)
    seen: list[dict] = []

    def factory(spec, rules, seed, search):
        seen.append(dict(search))
        return RandomBot(random.Random(seed))

    table = _table(
        tmp_path,
        opponents=(
            Opponent("random", 0.0, anchor=True),
            Opponent("wrapped", 100.0, spec="ckpt:runs/w2m/agent.pt"),
        ),
        policy_factory=factory,
        search={
            "label": "t/K 可选",
            "options": {
                "trunc_ply": {"min": 0, "max": 40, "default": 5},
                "rollout_k": {"min": 1, "max": 64, "default": 16},
            },
        },
    )
    table.start(
        {
            "mode": "free",
            "opponent_id": "wrapped",
            "seat": 0,
            "seed": 7,
            "search": {"trunc_ply": 0, "rollout_k": 32},
        }
    )
    assert seen == [{"trunc_ply": 0, "rollout_k": 32}]
    assert table.snapshot()["search"] == {"trunc_ply": 0, "rollout_k": 32}


def test_free_play_search_request_is_validated(tmp_path, monkeypatch):
    import seven523.web.table as table_module

    monkeypatch.setattr(table_module, "torch_available", lambda: True)
    table = _table(
        tmp_path,
        opponents=(
            Opponent("random", 0.0, anchor=True),
            Opponent("wrapped", 100.0, spec="ckpt:runs/w2m/agent.pt"),
        ),
        policy_factory=lambda spec, rules, seed, search: RandomBot(random.Random(seed)),
        search={
            "label": "t/K 可选",
            "options": {
                "trunc_ply": {"min": 0, "max": 40, "default": 5},
                "rollout_k": {"min": 1, "max": 64, "default": 16},
            },
        },
    )
    base = {"mode": "free", "opponent_id": "wrapped", "seat": 0, "seed": 7}
    with pytest.raises(Exception) as unknown:
        table.start({**base, "search": {"nope": 1}})
    assert "未知搜索参数" in str(unknown.value)
    with pytest.raises(Exception) as range_:
        table.start({**base, "search": {"trunc_ply": 99}})
    assert "越界" in str(range_.value)
    with pytest.raises(Exception) as integer:
        table.start({**base, "search": {"rollout_k": "abc"}})
    assert "整数" in str(integer.value)
    # A refused request leaves no live game behind.
    assert table.phase == "idle"


def test_free_play_beta_is_a_validated_float(tmp_path, monkeypatch):
    """``outcome_blend`` is a float knob: no int truncation, finite, bounded."""
    import seven523.web.table as table_module

    monkeypatch.setattr(table_module, "torch_available", lambda: True)
    seen: list[dict] = []

    def factory(spec, rules, seed, search):
        seen.append(dict(search))
        return RandomBot(random.Random(seed))

    table = _table(
        tmp_path,
        opponents=(
            Opponent("random", 0.0, anchor=True),
            Opponent("wrapped", 100.0, spec="ckpt:runs/w2m/agent.pt"),
        ),
        policy_factory=factory,
        search={
            "label": "t/K/β 可选",
            "options": {
                "trunc_ply": {"min": 0, "max": 40, "default": 5},
                "rollout_k": {"min": 1, "max": 512, "default": 32},
                "outcome_blend": {
                    "min": 0.0,
                    "max": 8.0,
                    "default": 0.0,
                    "type": "float",
                },
            },
        },
    )
    base = {"mode": "free", "opponent_id": "wrapped", "seat": 0, "seed": 7}
    table.start({**base, "search": {"outcome_blend": 1.5}})
    assert seen == [{"trunc_ply": 5, "rollout_k": 32, "outcome_blend": 1.5}]
    assert table.snapshot()["search"] == {
        "trunc_ply": 5,
        "rollout_k": 32,
        "outcome_blend": 1.5,
    }
    with pytest.raises(Exception) as boolean:
        table.start({**base, "search": {"outcome_blend": True}})
    assert "需要是数字" in str(boolean.value)
    with pytest.raises(Exception) as out_of_range:
        table.start({**base, "search": {"outcome_blend": 9.0}})
    assert "越界" in str(out_of_range.value)
    with pytest.raises(Exception) as non_finite:
        table.start({**base, "search": {"outcome_blend": "inf"}})
    assert "越界" in str(non_finite.value)


class _ReadoutBot(RandomBot):
    """Random play plus the head readout the live estimate panel asks for."""

    def value_and_outcome(self, view):
        return 0.125, 0.5


def test_free_play_estimate_panel_is_opt_in(tmp_path, monkeypatch):
    import seven523.web.table as table_module

    monkeypatch.setattr(table_module, "torch_available", lambda: True)
    table = _table(
        tmp_path,
        opponents=(
            Opponent("random", 0.0, anchor=True),
            Opponent("wrapped", 100.0, spec="ckpt:runs/w2m/agent.pt"),
        ),
        policy_factory=lambda spec, rules, seed, search: (
            _ReadoutBot(random.Random(seed))
            if spec.startswith("ckpt:")
            else RandomBot(random.Random(seed))
        ),
    )
    base = {"mode": "free", "opponent_id": "wrapped", "seat": 0, "seed": 7}
    # Default off: the snapshot carries no estimate.
    table.start(base)
    assert table.snapshot()["model_estimate"] is None
    # Opt in: the critic margin converts to points and the outcome head maps
    # to a win probability; the flag must be a strict boolean.
    table.start({**base, "estimate": True})
    assert table.snapshot()["model_estimate"] == {"margin": 12.5, "win_prob": 0.75}
    with pytest.raises(Exception) as bad:
        table.start({**base, "estimate": "yes"})
    assert "布尔" in str(bad.value)
    # The random anchor has no model readout; the panel is simply absent.
    table.start({**base, "opponent_id": "random", "estimate": True})
    assert table.snapshot()["model_estimate"] is None


def test_search_request_needs_a_search_launcher(tmp_path):
    table = _table(tmp_path)
    with pytest.raises(Exception) as excinfo:
        table.start(
            {
                "mode": "free",
                "opponent_id": "random",
                "seat": 0,
                "search": {"trunc_ply": 5},
            }
        )
    assert "未启用搜索包装" in str(excinfo.value)


def test_free_play_trace_records_the_search_config(tmp_path, monkeypatch):
    import seven523.web.table as table_module

    monkeypatch.setattr(table_module, "torch_available", lambda: True)
    table = _table(
        tmp_path,
        opponents=(
            Opponent("random", 0.0, anchor=True),
            Opponent("wrapped", 100.0, spec="ckpt:runs/w2m/agent.pt"),
        ),
        policy_factory=lambda spec, rules, seed, search: RandomBot(random.Random(seed)),
        search={
            "label": "t/K 可选",
            "options": {
                "trunc_ply": {"min": 0, "max": 40, "default": 5},
                "rollout_k": {"min": 1, "max": 64, "default": 16},
            },
        },
    )
    table.start(
        {
            "mode": "free",
            "opponent_id": "wrapped",
            "seat": 0,
            "seed": 7,
            "search": {"trunc_ply": 3, "rollout_k": 8},
        }
    )
    while table.phase == "human":
        _first_legal(table)
    trace = load_trace(table.snapshot()["result"]["trace_path"])
    assert trace["opponent_search"] == {"trunc_ply": 3, "rollout_k": 8}
    assert replay_trace(trace, print_fn=SILENT) is True


def test_placement_rejects_the_deleted_search_rung_path(tmp_path):
    """The rated sub-pool path was deleted; ``search_rung`` is refused.

    Raw placement still never touches the free-play factory: the merged pool
    routes only non-grammar specs through it.
    """

    def factory(spec, rules, seed, search):
        raise AssertionError("raw placement must not use the free-play factory")

    table = _table(tmp_path, policy_factory=factory)
    with pytest.raises(Exception) as excinfo:
        table.start({"mode": "placement", "games": 2, "search_rung": "search_leafq"})
    assert "search_rung 路径已删除" in str(excinfo.value)
    assert table.phase == "idle"
    table.start({"mode": "placement", "games": 2, "use_prior": False, "seed": 3})
    assert table.phase in ("human", "game_over")
    assert table.snapshot()["search"] is None
    progress = table.snapshot()["placement"]
    assert "rated" not in progress and "search_rung" not in progress


def _search_manifest(tmp_path: Path, *, search_mu: float = 260.0309697237381) -> Path:
    """The published-shaped manifest: the ``random`` anchor + one search rung."""
    path = tmp_path / "search-manifest.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "rules_id": rules_id(DEFAULT_RULES),
                "levels": {"random": 0.0, "search_leafq": search_mu},
                "anchors": [{"id": "random", "mu": 0.0}],
                "subjects": [
                    {"id": "random", "spec": "random", "mu": 0.0, "sigma": 0.0},
                    {
                        "id": "search_leafq",
                        "spec": RATED_SPEC,
                        "mu": search_mu,
                        "sigma": 3.855022421173465,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _web_config(tmp_path: Path, manifest: Path, **overrides) -> WebConfig:
    fields: dict = {
        "manifest": manifest,
        "prior": None,
        "sessions_dir": tmp_path / "sessions",
        "twins_dir": tmp_path / "twins",
        "free_traces_dir": tmp_path / "web",
        "seed": 0,
    }
    fields.update(overrides)
    return WebConfig(**fields)


def test_stock_web_skips_a_manifest_search_rung(tmp_path):
    """No plugin/factory: the rung is dropped with a warning, never crash."""
    table = TableSession.from_config(_web_config(tmp_path, _search_manifest(tmp_path)))
    assert {opponent.id for opponent in table.opponents} == {"random"}
    assert table.anchors == {"random": 0.0}
    assert any(
        "search_leafq" in warning and "cannot be built" in warning
        for warning in table.manifest_warnings
    )
    table.start({"mode": "placement", "games": 2, "use_prior": False, "seed": 3})
    assert {opponent.id for opponent in table.session.opponents} == {"random"}


def test_config_exposes_each_opponents_pinned_search_identity(tmp_path):
    """The page defaults t/K/β from the rung's published ``search_config``."""
    manifest = _search_manifest(tmp_path)
    document = json.loads(manifest.read_text(encoding="utf-8"))
    document["subjects"][1]["search_config"] = {
        "trunc_ply": 5,
        "rollout_k": 32,
        "outcome_blend": 1,
    }
    manifest.write_text(json.dumps(document), encoding="utf-8")
    table = TableSession.from_config(
        _web_config(
            tmp_path,
            manifest,
            can_build_spec=lambda spec: spec in ("random", RATED_SPEC),
        )
    )
    entries = {entry["id"]: entry for entry in table.config_document()["opponents"]}
    assert entries["search_leafq"]["search_config"] == {
        "trunc_ply": 5,
        "rollout_k": 32,
        "outcome_blend": 1,
    }
    assert "search_config" not in entries["random"]


def test_free_play_uses_the_rungs_pinned_search_defaults(tmp_path, monkeypatch):
    """Omitting a knob plays the rung's published identity; explicit wins."""
    import seven523.web.table as table_module

    monkeypatch.setattr(table_module, "torch_available", lambda: True)
    manifest = _search_manifest(tmp_path)
    document = json.loads(manifest.read_text(encoding="utf-8"))
    document["subjects"][1]["search_config"] = {
        "trunc_ply": 5,
        "rollout_k": 32,
        "outcome_blend": 1,
    }
    manifest.write_text(json.dumps(document), encoding="utf-8")
    seen: list[dict] = []

    def factory(spec, rules, seed, search):
        seen.append(dict(search))
        return RandomBot(random.Random(seed))

    table = TableSession.from_config(
        _web_config(
            tmp_path,
            manifest,
            policy_factory=factory,
            can_build_spec=lambda spec: spec in ("random", RATED_SPEC),
            search={
                "label": "t/K/β",
                "options": {
                    "trunc_ply": {"min": 0, "max": 40, "default": 5},
                    "rollout_k": {"min": 1, "max": 512, "default": 16},
                    "outcome_blend": {
                        "min": 0.0,
                        "max": 8.0,
                        "default": 0.0,
                        "type": "float",
                    },
                },
            },
        )
    )
    base = {"mode": "free", "opponent_id": "search_leafq", "seat": 0, "seed": 3}
    table.start(base)
    assert seen[-1] == {"trunc_ply": 5, "rollout_k": 32, "outcome_blend": 1.0}
    table.start({**base, "search": {"rollout_k": 8, "outcome_blend": 0.0}})
    assert seen[-1] == {"trunc_ply": 5, "rollout_k": 8, "outcome_blend": 0.0}
    # The anchor has no pinned block: the global option defaults apply.
    table.start({**base, "opponent_id": "random"})
    assert seen[-1] == {"trunc_ply": 5, "rollout_k": 16, "outcome_blend": 0.0}


def test_manifest_search_rung_is_admitted_with_the_plugin_gate(
    tmp_path, monkeypatch
):
    """The manifest level is the pool entry: gate + factory admit it once.

    After publication the manifest is the single μ/σ source and the plugin
    factory owns the pinned ``search_config``; the id appears once in the
    merged pool (anchors + raw rungs + search rungs) with no artifact overlay.
    """
    raw_calls: list[str] = []
    _patch_no_torch_policy(monkeypatch, raw_calls)
    seen: list[tuple[str, dict, int]] = []

    def factory(spec, rules, seed, search):
        seen.append((spec, dict(search), seed))
        return RandomBot(random.Random(seed))

    config = _web_config(
        tmp_path,
        _search_manifest(tmp_path),
        policy_factory=factory,
        can_build_spec=lambda spec: spec in ("random", RATED_SPEC),
    )
    table = TableSession.from_config(config)
    assert {opponent.id for opponent in table.opponents} == {
        "random",
        "search_leafq",
    }
    table.start({"mode": "placement", "games": 2, "use_prior": False, "seed": 3})
    pool = [opponent.id for opponent in table.session.opponents]
    assert sorted(pool) == ["random", "search_leafq"]
    # Deterministically check both branches of the merged-pool adapter: the
    # anchor through the shared grammar, the rung through the factory with an
    # empty request (the plugin's manifest-configured identity applies).
    anchor = next(opp for opp in table.session.opponents if opp.anchor)
    search = next(opp for opp in table.session.opponents if not opp.anchor)
    assert isinstance(table.session.policy_factory(anchor, 99), RandomBot)
    assert raw_calls[-1] == "random"
    seen.clear()
    assert isinstance(table.session.policy_factory(search, 7), RandomBot)
    assert seen == [(RATED_SPEC, {}, 7)]


def test_placement_keeps_the_prior_when_the_pool_contains_a_search_rung(
    tmp_path, monkeypatch
):
    """C3: a search rung in the pool alone must not force the prior off.

    The trace prior stays configured; ``PlacementSession`` scopes exclusion to
    a search row that is actually played and records provenance there.
    """
    seen: list[tuple[str, dict, int]] = []

    def factory(spec, rules, seed, search):
        seen.append((spec, dict(search), seed))
        return RandomBot(random.Random(seed))

    _patch_no_torch_policy(monkeypatch)
    prior = _minimal_prior()
    table = _table(
        tmp_path,
        opponents=(
            Opponent("random", 0.0, anchor=True),
            Opponent("search_leafq", 260.0309697237381, 3.855022421173465, RATED_SPEC),
        ),
        prior=prior,
        policy_factory=factory,
    )
    table.start({"mode": "placement", "games": 2, "seed": 3})
    session = table.session
    assert session.trace_prior is prior
    assert session.prior_off_reason is None
    progress = table.snapshot()["placement"]
    assert progress["use_prior"] is True
    assert progress["prior_off_reason"] is None
    # The adapter routes only the non-grammar spec through the injected
    # factory, with empty per-game overrides (the factory's pinned defaults).
    search = table.opponent_by_id["search_leafq"]
    seen.clear()
    assert isinstance(session.policy_factory(search, 7), RandomBot)
    assert seen == [(RATED_SPEC, {}, 7)]


def test_config_document_exposes_plugin_capabilities(tmp_path):
    table = _table(
        tmp_path,
        policy_factory=_twin_factory(),
        search={"label": "t/K", "options": SEARCH_OPTIONS},
        plugin={
            "source": "runs/o4lite-search/web_plugin.py",
            "error": None,
            "warnings": ["fixture"],
        },
        can_build_spec=lambda spec: True,
    )
    config = table.config_document()
    assert config["proto_version"] == PROTO_VERSION == 8
    assert config["plugin"] == {
        "source": "runs/o4lite-search/web_plugin.py",
        "error": None,
        "warnings": ["fixture"],
    }
    # The deleted rated/unrated capabilities are absent from the payload.
    assert "rated_rungs" not in config
    assert "unrated" not in config
    assert config["search"] == {"label": "t/K", "options": SEARCH_OPTIONS}
    # The stock table reports no plugin capabilities at all.
    stock = _table(tmp_path).config_document()
    assert stock["plugin"] is None
    assert stock["search"] is None


def _run_placement_to_end(table: TableSession) -> None:
    guard = 0
    while table.phase != "session_over":
        assert table.phase in ("human", "round_over")
        if table.phase == "round_over":
            table.continue_placement()
        else:
            _first_legal(table)
        guard += 1
        assert guard < 400, "placement session did not finish"


def test_placement_merged_pool_runs_to_a_report(tmp_path, monkeypatch):
    """One pool: raw rungs and a search rung in the same session schedule."""
    raw_calls: list[str] = []
    _patch_no_torch_policy(monkeypatch, raw_calls)
    seen: list[tuple[str, dict, int]] = []

    def factory(spec, rules, seed, search):
        seen.append((spec, dict(search), seed))
        return RandomBot(random.Random(seed))

    _patch_no_torch_policy(monkeypatch)
    table = _table(
        tmp_path,
        opponents=(
            Opponent("random", 0.0, anchor=True),
            Opponent("lvl1", 100.0, spec="random"),
            Opponent("search_leafq", 260.0309697237381, 3.855022421173465, RATED_SPEC),
        ),
        policy_factory=factory,
    )
    table.start({"mode": "placement", "games": 4, "use_prior": False, "seed": 3})
    _run_placement_to_end(table)
    progress = table.snapshot()["placement"]
    report = progress["report"]
    assert report["games_played"] == 4
    assert {entry["id"] for entry in report["rungs"]} == {
        "random",
        "lvl1",
        "search_leafq",
    }
    assert set(report["levels"]) == {"random", "lvl1", "search_leafq"}
    assert report["channels"]["trace"]["enabled"] is False
    # Whenever the search rung was scheduled it went through the factory with
    # an empty request: the plugin's manifest-configured identity applies.
    assert all(params == {} for _spec, params, _seed in seen)
    assert all(spec == RATED_SPEC for spec, _params, _seed in seen)
    # Every played game is in the session artifact and replays.
    directory = Path(progress["session_dir"])
    session_doc = json.loads((directory / "session.json").read_text())
    rungs_doc = json.loads((directory / "rungs.json").read_text())
    assert len(session_doc["games"]) == 4
    assert {entry["id"] for entry in rungs_doc["opponents"]} == {
        "random",
        "lvl1",
        "search_leafq",
    }
    assert "search_rung" not in session_doc
    assert "rated" not in report
    for record in session_doc["games"]:
        replay_trace(load_trace(directory / record["trace"]), print_fn=SILENT)






# -- twin --------------------------------------------------------------------


def test_twin_mode_requires_launcher_metadata(tmp_path):
    table = _table(tmp_path, policy_factory=_twin_factory())
    with pytest.raises(Exception) as excinfo:
        table.start({"mode": "twin", "pairs": 2})
    assert "web_twin.py" in str(excinfo.value)
    assert table.phase == "idle"


def test_twin_start_validates_pairs(tmp_path, monkeypatch):
    _patch_twin(monkeypatch)
    table = _twin_table(tmp_path)
    for bad in ({"pairs": 3}, {"pairs": 200}, {"pairs": 2, "search": {}}):
        with pytest.raises(Exception):
            table.start({"mode": "twin", **bad})
        assert table.phase == "idle"
    with pytest.raises(Exception) as excinfo:
        table.start({"mode": "twin", "pairs": 2, "opponent_id": "x"})
    assert "opponent_id" in str(excinfo.value)
    table.start({"mode": "twin", "pairs": 2, "seed": 3})
    assert table.phase == "human"
    assert table.snapshot()["twin"]["pairs_total"] == 2


def test_twin_session_runs_to_report(tmp_path, monkeypatch):
    _patch_twin(monkeypatch)
    table = _twin_table(tmp_path)
    table.start({"mode": "twin", "pairs": 2, "seed": 3})
    guard = 0
    while table.phase != "session_over":
        assert table.phase in ("human", "round_over")
        if table.phase == "round_over":
            table.continue_round()
        else:
            _first_legal(table)
        guard += 1
        assert guard < 400, "twin session did not finish"
    twin = table.snapshot()["twin"]
    assert twin["pairs_total"] == 2
    assert twin["pairs_complete"] == 2
    assert twin["pairs_incomplete"] == 0
    assert twin["games_played"] == twin["games_total"] == 4
    assert twin["current"] is None
    assert twin["interim"]["margin_points_delta"] is not None
    report = twin["report"]
    assert report["schema"] == "seven523.deal-twin-report"
    assert report["counts"]["pairs_complete"] == 2
    assert report["bootstrap_seed"] == 523
    assert report["resolution"]["min_pairs"] == 30
    assert report["delta"]["per_pair"][0]["order"] == ["raw", "search"]
    directory = Path(twin["session_dir"])
    assert (directory / "session.json").is_file()
    assert (directory / "report.json").is_file()
    document = json.loads((directory / "session.json").read_text())
    assert len(document["games"]) == 4
    for record in document["games"]:
        replay_trace(load_trace(directory / record["trace"]), print_fn=SILENT)


def test_twin_uses_factory_only_for_search_arm(tmp_path, monkeypatch):
    raw_calls: list[str] = []
    seen: list[tuple] = []
    _patch_twin(monkeypatch, raw_calls)
    table = _twin_table(tmp_path, factory=_twin_factory(seen))
    table.start({"mode": "twin", "pairs": 2, "seed": 3})
    guard = 0
    while table.phase != "session_over":
        if table.phase == "round_over":
            table.continue_round()
        else:
            _first_legal(table)
        guard += 1
        assert guard < 400
    assert len(seen) == 2  # the search arm's two games
    assert len(raw_calls) == 2  # the raw arm's two games
    assert all(spec == TWIN_BASE_SPEC for spec in raw_calls)
    assert all(search == TWIN_PARAMS for _spec, search, _seed in seen)
    directory = Path(table.snapshot()["twin"]["session_dir"])
    for record in json.loads((directory / "session.json").read_text())["games"]:
        trace = load_trace(directory / record["trace"])
        if record["arm"] == "search":
            assert trace["deal_twin"]["search"] == TWIN_PARAMS
            assert trace["opponent_search"] == TWIN_PARAMS
        else:
            assert trace["deal_twin"]["search"] is None
            assert "opponent_search" not in trace


def test_placement_under_twin_launcher_stays_raw_only(tmp_path):
    def factory(spec, rules, seed, search):
        raise AssertionError("placement must stay raw-only under the twin launcher")

    table = _table(tmp_path, policy_factory=factory, twin=TWIN_META)
    table.start({"mode": "placement", "games": 2, "use_prior": False, "seed": 3})
    assert table.phase in ("human", "game_over")
    assert table.snapshot()["twin"] is None


def test_twin_quit_drops_incomplete_pair(tmp_path, monkeypatch):
    _patch_twin(monkeypatch)
    table = _twin_table(tmp_path)
    table.start({"mode": "twin", "pairs": 2, "seed": 3})
    while table.phase == "human":
        _first_legal(table)
    assert table.phase == "round_over"
    table.continue_round()
    assert table.phase == "human"
    assert table.snapshot()["twin"]["current"]["arm"] == "search"
    table.quit_session()
    snap = table.snapshot()
    assert snap["phase"] == "session_over"
    twin = snap["twin"]
    assert twin["pairs_complete"] == 0
    assert twin["pairs_incomplete"] == 1
    report = twin["report"]
    assert report["counts"]["pairs_complete"] == 0
    assert report["counts"]["pairs_incomplete"] == 1
    assert report["counts"]["abandoned_games"] == 1
    assert report["delta"]["margin_points"]["value"] is None
    assert report["arm_stats"]["raw"]["games"] == 1
    assert report["arm_stats"]["search"]["games"] == 0
    assert (Path(twin["session_dir"]) / "report.json").is_file()


# -- HTTP --------------------------------------------------------------------


def _serve(table: TableSession):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(table))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


def _post(base: str, route: str, payload: dict) -> tuple[int, dict]:
    request = urllib.request.Request(
        base + route,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.load(error)


def _get(base: str, route: str) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(base + route) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.load(error)


def test_http_routes_drive_a_free_game(tmp_path):
    table = _table(tmp_path)
    httpd, base = _serve(table)
    try:
        status, config = _get(base, "/api/config")
        assert status == 200
        assert config["proto_version"] == PROTO_VERSION
        assert [entry["id"] for entry in config["opponents"]] == ["random", "lvl1"]

        status, snap = _post(
            base,
            "/api/start",
            {"mode": "free", "opponent_id": "random", "seat": 0, "seed": 7},
        )
        assert status == 200
        assert snap["phase"] == "human"
        action_id = snap["legal"][0]["action_id"]

        status, snap = _post(base, "/api/action", {"action_id": action_id})
        assert status == 200
        assert snap["phase"] in ("human", "game_over")

        status, error = _post(base, "/api/action", {"action_id": 10_000})
        assert status == 400
        assert "非法出牌" in error["error"]

        status, state = _get(base, "/api/state")
        assert status == 200
        assert state["phase"] in ("human", "game_over")

        status, error = _get(base, "/api/nope")
        assert status == 404
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_http_twin_start_continue_quit(tmp_path, monkeypatch):
    _patch_twin(monkeypatch)
    table = _twin_table(tmp_path)
    httpd, base = _serve(table)
    try:
        status, config = _get(base, "/api/config")
        assert status == 200
        assert config["proto_version"] == PROTO_VERSION
        assert config["twin"]["search"]["identity"].startswith("rolloutt:")
        status, snap = _post(
            base, "/api/start", {"mode": "twin", "pairs": 2, "seed": 3}
        )
        assert status == 200
        assert snap["twin"]["current"]["arm"] == "raw"
        while snap["phase"] == "human":
            status, snap = _post(
                base, "/api/action", {"action_id": snap["legal"][0]["action_id"]}
            )
            assert status == 200
        assert snap["phase"] == "round_over"
        status, snap = _post(base, "/api/continue", {})
        assert status == 200
        assert snap["phase"] == "human"
        status, snap = _post(base, "/api/quit", {})
        assert status == 200
        assert snap["phase"] == "session_over"
        assert snap["twin"]["pairs_incomplete"] == 1
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_http_placement_merged_pool_smoke(tmp_path, monkeypatch):
    """Headless HTTP: one merged pool, no rated/unrated payload fields."""
    import seven523.web.table as table_module

    monkeypatch.setattr(table_module, "torch_available", lambda: True)
    monkeypatch.setattr(
        table_module,
        "policy_from_spec",
        lambda spec, rules=DEFAULT_RULES, seed=None, device="cpu": RandomBot(
            random.Random(seed)
        ),
    )
    seen: list[tuple[str, dict]] = []

    def factory(spec, rules, seed, search):
        seen.append((spec, dict(search)))
        return RandomBot(random.Random(seed))

    table = _table(
        tmp_path,
        opponents=(
            Opponent("random", 0.0, anchor=True),
            Opponent("search_leafq", 260.0309697237381, 3.855022421173465, RATED_SPEC),
        ),
        policy_factory=factory,
        search={"label": "t/K", "options": SEARCH_OPTIONS},
    )
    httpd, base = _serve(table)

    def finish_game(snap: dict) -> dict:
        while snap["phase"] == "human":
            status, snap = _post(
                base, "/api/action", {"action_id": snap["legal"][0]["action_id"]}
            )
            assert status == 200
        return snap

    try:
        status, config = _get(base, "/api/config")
        assert status == 200
        assert config["proto_version"] == PROTO_VERSION == 8
        assert "rated_rungs" not in config and "unrated" not in config
        assert [entry["id"] for entry in config["opponents"]] == [
            "random",
            "search_leafq",
        ]

        # Placement over the merged pool: two games, ordinary report.
        status, snap = _post(
            base,
            "/api/start",
            {"mode": "placement", "games": 2, "use_prior": False, "seed": 3},
        )
        assert status == 200
        snap = finish_game(snap)
        assert snap["phase"] == "round_over"
        status, snap = _post(base, "/api/continue", {})
        assert status == 200
        snap = finish_game(snap)
        assert snap["phase"] == "session_over"
        assert snap["placement"]["report"]["games_played"] == 2
        assert "rated" not in snap["placement"]
        assert "series" not in snap

        # The deleted request paths are explicit 400s, never silent replays.
        for payload, needle in (
            ({"mode": "unrated", "games": 2}, "unknown mode"),
            ({"mode": "placement", "games": 2, "search_rung": "x"}, "search_rung"),
        ):
            status, error = _post(base, "/api/start", payload)
            assert status == 400
            assert needle in error["error"]
        # Whenever the rung was scheduled the factory saw the pinned request.
        assert all(params == {} for _spec, params in seen)
    finally:
        httpd.shutdown()
        httpd.server_close()


# -- CLI ---------------------------------------------------------------------


def test_cli_defaults():
    args = build_parser().parse_args([])
    assert args.manifest == "traces/pool10/manifest.json"
    assert args.prior == "artifacts/human-elo/prior.json"
    assert args.sessions_dir == "traces/sessions"
    assert args.trace_dir == "traces/web"
    assert args.host == "127.0.0.1" and args.port == 8765
    assert args.plugin is None
    assert args.no_plugin is False
    assert build_parser().parse_args(["--no-trace-prior"]).no_trace_prior is True
    assert build_parser().parse_args(["--plugin", "x.py"]).plugin == "x.py"
    assert build_parser().parse_args(["--no-plugin"]).no_plugin is True


def test_pyproject_declares_the_web_entry_point():
    pyproject = (ROOT / "pyproject.toml").read_text()
    assert '7g523-web = "seven523.web:main"' in pyproject

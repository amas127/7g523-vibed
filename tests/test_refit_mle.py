"""``tools/refit_mle.py``: the anchored probit-MLE publication CLI.

The tool is exercised end to end on tmp JSONL files in the shape
``ladder.play_games`` writes plus a few pure-helper checks (bootstrap,
table).  Every row must carry the current ``rules_id`` and a legacy prior
manifest is refused exactly like ``placement.load_opponents`` does
(ADR-0013: ratings are never pooled across rule versions).  The
``--manifest-out --refit`` publication path is covered against a frozen
manifest with specs: levels/subjects/rungs/estimator refresh while ``spec``
and unrelated keys survive, and a legacy target or ``--manifest-out``
without ``--refit`` is refused without writing.
"""
from __future__ import annotations

import importlib.util
import json
import math
import random
import sys
from pathlib import Path

import pytest

from seven523.elo import (
    DEFAULT_BETA,
    PlayedGame,
    Prior,
    Rating,
    select_rungs,
)
from seven523.mle import MleConfig, MleFit, fit_mle
from seven523.rules import DEFAULT_RULES, rules_id, rules_identity

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"

_spec = importlib.util.spec_from_file_location("refit_mle", TOOLS / "refit_mle.py")
assert _spec and _spec.loader
refit = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = refit
_spec.loader.exec_module(refit)

IDENTITY = rules_id(DEFAULT_RULES)


def _row(
    seed: int,
    seats: tuple[str, ...],
    scores: tuple[int, ...],
    *,
    rules: str = IDENTITY,
    kind: str = "anchor",
) -> dict:
    return {
        "seed": seed,
        "seats": list(seats),
        "scores": list(scores),
        "subject": seats[0],
        "opponent": seats[1],
        "subject_seat": 0,
        "kind": kind,
        "rules_id": rules,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _ladder_rows() -> list[dict]:
    """Sixty games: ``strong`` > ``weak`` > ``random`` (anchor), no perfect records."""
    rows: list[dict] = []
    for seed in range(24):
        rows.append(_row(seed, ("strong", "random"), (100, 0)))
    for seed in range(24, 30):
        rows.append(_row(seed, ("random", "strong"), (100, 0)))
    for seed in range(30, 36):
        rows.append(_row(seed, ("weak", "random"), (100, 0)))
    for seed in range(36, 60):
        rows.append(_row(seed, ("random", "weak"), (100, 0)))
    return rows


def test_happy_path_prints_a_mu_descending_table(tmp_path, capsys):
    first = _write_jsonl(tmp_path / "a.jsonl", _ladder_rows()[:30])
    second = _write_jsonl(tmp_path / "b.jsonl", _ladder_rows()[30:])

    code = refit.main(
        ["--games", str(first), "--games", str(second), "--bootstrap", "0"]
    )
    output = capsys.readouterr().out

    assert code == 0
    # Pooling both files: strong wins more than it loses, weak the reverse.
    assert output.index("strong") < output.index("random") < output.index("weak")
    strong_line = next(
        line for line in output.splitlines() if line.startswith("strong")
    )
    assert "30" in strong_line.split()
    assert "random" in next(
        line for line in output.splitlines() if line.startswith("random")
    )


def test_json_and_out_artifact_schema(tmp_path, capsys):
    games = _write_jsonl(tmp_path / "games.jsonl", _ladder_rows())
    out = tmp_path / "table.json"

    assert (
        refit.main(
            ["--games", str(games), "--bootstrap", "0", "--json", "--out", str(out)]
        )
        == 0
    )
    captured = capsys.readouterr()
    stdout_doc = json.loads(captured.out)
    file_text = out.read_text(encoding="utf-8")
    file_doc = json.loads(file_text)
    assert stdout_doc == file_doc
    # T16-4 minor: the --out artifact and the --json stdout are byte-identical
    # (the same serialization, both newline-terminated).
    assert file_text == captured.out

    assert set(stdout_doc) == {
        "version",
        "rules_id",
        "rules",
        "anchors",
        "levels",
        "sigmas",
        "n",
        "ci",
        "converged",
        "margin_identified",
        "ci_source",
        "separation",
        "estimator",
        "games",
        "resolution",
    }
    # No bootstrap replicates -> no resolution block, explicitly null.
    assert stdout_doc["resolution"] is None
    assert stdout_doc["converged"] is True
    assert stdout_doc["margin_identified"] is False  # this slate has no ties
    assert stdout_doc["ci_source"] == "laplace(bootstrap=0)"
    assert stdout_doc["version"] == 1
    assert stdout_doc["rules_id"] == IDENTITY
    assert stdout_doc["rules"] == rules_identity(DEFAULT_RULES)
    assert stdout_doc["anchors"] == {"random": 0.0}
    assert stdout_doc["games"] == 60
    assert stdout_doc["separation"] == []  # no perfect records in this slate
    assert set(stdout_doc["levels"]) == {"strong", "weak", "random"}
    assert set(stdout_doc["sigmas"]) == {"strong", "weak", "random"}
    assert set(stdout_doc["ci"]) == {"strong", "weak", "random"}
    # Every id is measured in this slate, so the per-id games map is non-zero.
    assert stdout_doc["n"] == {"random": 60, "strong": 30, "weak": 30}

    assert stdout_doc["levels"]["strong"] > 0.0 > stdout_doc["levels"]["weak"]
    assert stdout_doc["levels"]["random"] == 0.0
    assert stdout_doc["sigmas"]["random"] == 0.0
    assert stdout_doc["ci"]["random"] == [0.0, 0.0]

    estimator = stdout_doc["estimator"]
    assert set(estimator) == {
        "kind",
        "beta",
        "draw_margin",
        "bootstrap",
        "seed",
        "prior_sigma",
    }
    assert estimator["prior_sigma"] is None  # no --prior-sigma override
    assert estimator["kind"] == "probit-mle"
    assert estimator["beta"] == pytest.approx(DEFAULT_BETA)
    assert estimator["bootstrap"] == 0
    assert estimator["seed"] == 0
    assert estimator["draw_margin"] >= 0.0

    # Without a bootstrap the CI is the Laplace one: mu +/- 1.96 * sigma.
    for id_ in ("strong", "weak"):
        mu = stdout_doc["levels"][id_]
        sigma = stdout_doc["sigmas"][id_]
        assert sigma > 0.0
        assert stdout_doc["ci"][id_][0] == pytest.approx(mu - 1.96 * sigma)
        assert stdout_doc["ci"][id_][1] == pytest.approx(mu + 1.96 * sigma)


def test_no_artifact_is_written_without_out(tmp_path, capsys):
    games = _write_jsonl(tmp_path / "games.jsonl", _ladder_rows()[:8])
    before = sorted(path.name for path in tmp_path.iterdir())

    assert refit.main(["--games", str(games), "--bootstrap", "0"]) == 0
    capsys.readouterr()

    assert sorted(path.name for path in tmp_path.iterdir()) == before


def test_cross_file_rules_id_mismatch_is_refused(tmp_path):
    first = _write_jsonl(tmp_path / "a.jsonl", _ladder_rows()[:3])
    second = _write_jsonl(
        tmp_path / "b.jsonl",
        [_row(100, ("strong", "random"), (100, 0), rules="deadbeef")],
    )

    with pytest.raises(SystemExit, match="rules_id"):
        refit.main(["--games", str(first), "--games", str(second), "--bootstrap", "0"])


def test_missing_rules_id_is_refused(tmp_path):
    row = _row(0, ("strong", "random"), (100, 0))
    del row["rules_id"]
    games = _write_jsonl(tmp_path / "legacy.jsonl", [row])

    with pytest.raises(SystemExit, match="rules_id"):
        refit.main(["--games", str(games), "--bootstrap", "0"])


def test_bootstrap_ci_is_reproducible_for_a_fixed_seed(tmp_path, capsys):
    games = _write_jsonl(tmp_path / "games.jsonl", _ladder_rows())

    def run(seed: int) -> dict:
        assert (
            refit.main(
                [
                    "--games",
                    str(games),
                    "--bootstrap",
                    "25",
                    "--seed",
                    str(seed),
                    "--json",
                ]
            )
            == 0
        )
        return json.loads(capsys.readouterr().out)

    first = run(7)
    second = run(7)
    assert first["ci"] == second["ci"]
    # A different stream moves the resampling draws.
    assert first["ci"] != run(8)["ci"]


def test_bootstrap_ci_brackets_the_point_estimate(tmp_path, capsys):
    games = _write_jsonl(tmp_path / "games.jsonl", _ladder_rows())

    assert (
        refit.main(
            ["--games", str(games), "--bootstrap", "25", "--seed", "3", "--json"]
        )
        == 0
    )
    doc = json.loads(capsys.readouterr().out)
    for id_, mu in doc["levels"].items():
        low, high = doc["ci"][id_]
        assert low <= high
        # The percentile interval must contain the point estimate for a
        # well-behaved sample; the anchor is exactly constant.
        assert low - 1e-9 <= mu <= high + 1e-9


def test_legacy_prior_manifest_is_refused(tmp_path):
    games = _write_jsonl(tmp_path / "games.jsonl", _ladder_rows()[:6])
    legacy = tmp_path / "legacy-manifest.json"
    legacy.write_text(
        json.dumps({"version": 1, "levels": {"ghost": 1234.0}}), encoding="utf-8"
    )

    with pytest.raises(SystemExit, match="rules_id"):
        refit.main(
            [
                "--games",
                str(games),
                "--prior-manifest",
                str(legacy),
                "--bootstrap",
                "0",
            ]
        )


def test_cross_version_prior_manifest_is_refused(tmp_path):
    games = _write_jsonl(tmp_path / "games.jsonl", _ladder_rows()[:6])
    other = tmp_path / "old-manifest.json"
    other.write_text(
        json.dumps({"version": 1, "rules_id": "deadbeef", "levels": {"ghost": 1.0}}),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="rules_id"):
        refit.main(
            ["--games", str(games), "--prior-manifest", str(other), "--bootstrap", "0"]
        )


def test_empty_legacy_manifest_is_accepted(tmp_path, capsys):
    games = _write_jsonl(tmp_path / "games.jsonl", _ladder_rows()[:6])
    empty = tmp_path / "empty.json"
    empty.write_text("{}", encoding="utf-8")

    assert (
        refit.main(
            [
                "--games",
                str(games),
                "--prior-manifest",
                str(empty),
                "--bootstrap",
                "0",
                "--json",
            ]
        )
        == 0
    )
    doc = json.loads(capsys.readouterr().out)
    assert doc["games"] == 6


def test_prior_manifest_warm_starts_and_ignores_the_anchor(tmp_path, capsys):
    games = _write_jsonl(tmp_path / "games.jsonl", _ladder_rows()[:6])
    manifest = {
        "version": 1,
        "rules_id": IDENTITY,
        "levels": {"ghost": 1234.0, "random": 0.0},
        "subjects": [{"id": "ghost", "mu": 1234.0, "sigma": 150.0}],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    assert (
        refit.main(
            [
                "--games",
                str(games),
                "--prior-manifest",
                str(path),
                "--bootstrap",
                "0",
                "--json",
            ]
        )
        == 0
    )
    doc = json.loads(capsys.readouterr().out)

    # ``ghost`` has no games: it is returned with its prior and n == 0.
    assert doc["levels"]["ghost"] == pytest.approx(1234.0)
    assert doc["sigmas"]["ghost"] == pytest.approx(150.0)
    assert doc["ci"]["ghost"][0] == pytest.approx(1234.0 - 1.96 * 150.0)
    # The pinned anchor is never passed as a prior (fit_mle would reject it).
    assert doc["levels"]["random"] == 0.0


def test_explicit_anchors_replace_the_default_gauge(tmp_path, capsys):
    games = _write_jsonl(tmp_path / "games.jsonl", _ladder_rows()[:6])

    assert (
        refit.main(
            [
                "--games",
                str(games),
                "--anchors",
                "random=10",
                "--bootstrap",
                "0",
                "--json",
            ]
        )
        == 0
    )
    doc = json.loads(capsys.readouterr().out)
    assert doc["anchors"] == {"random": 10.0}
    assert doc["levels"]["random"] == 10.0


def test_draw_margin_is_estimated_with_ties_and_fixable(tmp_path, capsys):
    rows = []
    for seed in range(12):
        scores = (50, 50) if seed % 3 == 0 else (100, 0)
        rows.append(_row(seed, ("a", "random"), scores))
    games = _write_jsonl(tmp_path / "ties.jsonl", rows)

    assert refit.main(["--games", str(games), "--json", "--bootstrap", "0"]) == 0
    estimated = json.loads(capsys.readouterr().out)["estimator"]["draw_margin"]
    assert 0.0 < estimated < 1e6  # not the 1e13 solver clamp

    assert (
        refit.main(
            [
                "--games",
                str(games),
                "--json",
                "--bootstrap",
                "0",
                "--draw-margin",
                "25",
            ]
        )
        == 0
    )
    fixed = json.loads(capsys.readouterr().out)["estimator"]["draw_margin"]
    assert fixed == pytest.approx(25.0)


def test_table_flags_a_perfect_record_against_the_anchor():
    games = [PlayedGame(seed, ("hero", "random"), (100, 0)) for seed in range(8)]
    fit = fit_mle(
        games, anchors={"random": 0.0}, priors={"hero": Prior(0.0, 200.0)}
    )

    # The Gaussian prior keeps the estimate finite, and the win record is
    # still reported as a separation risk rather than published silently.
    assert "hero" in fit.separation
    assert math.isfinite(fit.ratings["hero"].mu)

    table = refit.format_table(
        fit, {id_: (rating.mu, rating.mu) for id_, rating in fit.ratings.items()}
    )
    hero_line = next(line for line in table.splitlines() if line.startswith("hero"))
    assert hero_line.rstrip().endswith("yes")


def test_json_artifact_carries_the_separation_set(tmp_path, capsys):
    # F2 regression: machine consumers must see which published ids are
    # perfect-record / out-of-range values, not only the human table.
    games = _write_jsonl(
        tmp_path / "perfect.jsonl",
        [_row(seed, ("hero", "random"), (100, 0)) for seed in range(8)],
    )

    assert refit.main(["--games", str(games), "--json", "--bootstrap", "0"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["separation"] == ["hero"]
    assert "hero" in doc["levels"]


def test_missing_prior_manifest_is_refused(tmp_path, capsys):
    games = _write_jsonl(tmp_path / "games.jsonl", _ladder_rows()[:6])

    with pytest.raises(SystemExit, match="prior-manifest"):
        refit.main(
            [
                "--games",
                str(games),
                "--prior-manifest",
                str(tmp_path / "nope.json"),
                "--bootstrap",
                "0",
            ]
        )
    assert capsys.readouterr().err == ""


def test_directory_prior_manifest_is_refused(tmp_path):
    games = _write_jsonl(tmp_path / "games.jsonl", _ladder_rows()[:6])

    with pytest.raises(SystemExit, match="prior-manifest"):
        refit.main(
            [
                "--games",
                str(games),
                "--prior-manifest",
                str(tmp_path),
                "--bootstrap",
                "0",
            ]
        )


def test_malformed_prior_manifest_is_refused(tmp_path):
    games = _write_jsonl(tmp_path / "games.jsonl", _ladder_rows()[:6])
    for name, text in (
        ("bad.json", "{not json"),
        ("list.json", "[1, 2]"),
    ):
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        with pytest.raises(SystemExit, match="prior-manifest"):
            refit.main(
                [
                    "--games",
                    str(games),
                    "--prior-manifest",
                    str(path),
                    "--bootstrap",
                    "0",
                ]
            )


def test_out_path_errors_exit_with_context(tmp_path):
    games = _write_jsonl(tmp_path / "games.jsonl", _ladder_rows()[:6])

    with pytest.raises(SystemExit, match="--out"):
        refit.main(
            ["--games", str(games), "--out", str(tmp_path), "--bootstrap", "0"]
        )
    with pytest.raises(SystemExit, match="--out"):
        refit.main(
            [
                "--games",
                str(games),
                "--out",
                "/proc/nope/table.json",
                "--bootstrap",
                "0",
            ]
        )


def test_tiny_beta_exits_cleanly_without_a_traceback(tmp_path, capsys):
    pinned = _write_jsonl(
        tmp_path / "pinned.jsonl", [_row(0, ("a", "b"), (50, 50))]
    )

    with pytest.raises(SystemExit, match="beta"):
        refit.main(
            [
                "--games",
                str(pinned),
                "--anchors",
                "a=0",
                "--anchors",
                "b=0",
                "--beta",
                "5e-324",
                "--bootstrap",
                "0",
            ]
        )


def test_table_footer_names_the_ci_source(tmp_path, capsys):
    games = _write_jsonl(tmp_path / "games.jsonl", _ladder_rows())

    assert refit.main(["--games", str(games), "--bootstrap", "0"]) == 0
    laplace = capsys.readouterr().out
    assert "ci=laplace(bootstrap=0)" in laplace

    assert (
        refit.main(
            ["--games", str(games), "--bootstrap", "3", "--seed", "9"]
        )
        == 0
    )
    bootstrap = capsys.readouterr().out
    assert "ci=deal-bootstrap(seed=9, n=3)" in bootstrap


def _strict_load(text: str) -> dict:
    """``json.loads`` with NaN/Infinity rejected, matching allow_nan=False."""

    def reject(value: str) -> None:
        raise AssertionError(f"non-JSON constant in artifact: {value}")

    return json.loads(text, parse_constant=reject)


def test_bootstrap_ci_source_is_exposed(tmp_path, capsys):
    games = _write_jsonl(tmp_path / "games.jsonl", _ladder_rows())

    assert (
        refit.main(
            ["--games", str(games), "--bootstrap", "3", "--seed", "9", "--json"]
        )
        == 0
    )
    doc = _strict_load(capsys.readouterr().out)
    assert doc["ci_source"] == "deal-bootstrap(seed=9, n=3)"


def test_all_ties_artifact_flags_an_unidentified_margin(tmp_path, capsys):
    # T16-4-F4 regression: the tie-rate-implied margin must be flagged, not
    # published as a converged fitted value.
    rows = [_row(seed, ("a", "random"), (50, 50)) for seed in range(8)]
    games = _write_jsonl(tmp_path / "ties.jsonl", rows)

    assert (
        refit.main(["--games", str(games), "--bootstrap", "0", "--json"]) == 0
    )
    doc = _strict_load(capsys.readouterr().out)
    assert doc["margin_identified"] is False
    assert doc["estimator"]["draw_margin"] > 0.0


def test_anchor_beyond_the_project_scale_is_refused(tmp_path):
    games = _write_jsonl(tmp_path / "games.jsonl", _ladder_rows()[:6])

    with pytest.raises(SystemExit, match="scale"):
        refit.main(
            ["--games", str(games), "--anchors", "big=1e7", "--bootstrap", "0"]
        )


def test_prior_centre_beyond_the_project_scale_is_refused(tmp_path):
    games = _write_jsonl(tmp_path / "games.jsonl", _ladder_rows()[:6])
    manifest = {
        "version": 1,
        "rules_id": IDENTITY,
        "levels": {"ghost": 1e7},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(SystemExit, match="scale"):
        refit.main(
            [
                "--games",
                str(games),
                "--prior-manifest",
                str(path),
                "--bootstrap",
                "0",
            ]
        )


def test_extreme_prior_sigmas_still_produce_strict_json(tmp_path, capsys):
    # T16-4-F2 regression: sigma**2 used to raise OverflowError (1e308) or
    # ZeroDivisionError (1e-300) as an uncaught traceback.
    games = _write_jsonl(tmp_path / "games.jsonl", [_row(0, ("a", "b"), (100, 0))])
    for name, sigma in (("wide.json", 1e308), ("narrow.json", 1e-300)):
        manifest = {
            "version": 1,
            "rules_id": IDENTITY,
            "levels": {"a": 500.0, "b": 400.0},
            "subjects": [
                {"id": "a", "mu": 500.0, "sigma": sigma},
                {"id": "b", "mu": 400.0, "sigma": sigma},
            ],
        }
        path = tmp_path / name
        path.write_text(json.dumps(manifest), encoding="utf-8")

        assert (
            refit.main(
                [
                    "--games",
                    str(games),
                    "--prior-manifest",
                    str(path),
                    "--bootstrap",
                    "0",
                    "--json",
                ]
            )
            == 0
        )
        doc = _strict_load(capsys.readouterr().out)
        for id_ in ("a", "b"):
            sigma_out = doc["sigmas"][id_]
            assert sigma_out is None or math.isfinite(sigma_out)
            low, high = doc["ci"][id_]
            assert low is None or math.isfinite(low)
            assert high is None or math.isfinite(high)


def test_non_finite_estimates_are_published_as_null_with_a_warning(
    tmp_path, capsys, monkeypatch
):
    # ADR-0013 publication integrity: a non-finite sigma/CI must become null
    # in a strict artifact, with a stderr warning instead of a silent NaN.
    games = _write_jsonl(tmp_path / "games.jsonl", _ladder_rows()[:4])

    def fake_fit(games, *, anchors, priors=None, config=None):
        return MleFit(
            ratings={
                "random": Rating(0.0, 0.0, 1),
                "you": Rating(500.0, math.inf, 1),
            },
            games=1,
            draw_margin=0.0,
            iterations=2,
            separation=frozenset(),
            converged=False,
            margin_identified=False,
        )

    monkeypatch.setattr(refit, "fit_mle", fake_fit)
    assert (
        refit.main(["--games", str(games), "--bootstrap", "0", "--json"]) == 0
    )
    captured = capsys.readouterr()
    doc = _strict_load(captured.out)
    assert doc["sigmas"]["you"] is None
    assert doc["ci"]["you"] == [None, None]
    assert doc["converged"] is False
    assert doc["margin_identified"] is False
    assert "non-finite" in captured.err
    assert "provisional" in captured.err


# -- manifest publication path -----------------------------------------------


def _manifest_document(*, rules: bool = True) -> dict:
    """A frozen study manifest with specs, anchors and unrelated metadata."""
    document: dict = {
        "version": 1,
        "levels": {"strong": 1.0, "weak": -1.0, "random": 0.0},
        "subjects": [
            {
                "id": "strong",
                "spec": "ckpt:strong.pt",
                "mu": 1.0,
                "sigma": 9.0,
                "games": 3,
            },
            {
                "id": "weak",
                "spec": "ckpt:weak.pt",
                "mu": -1.0,
                "sigma": 9.0,
                "games": 3,
            },
            {
                "id": "random",
                "spec": "random",
                "mu": 0.0,
                "sigma": 0.0,
                "games": 3,
            },
        ],
        "anchors": [{"id": "random", "mu": 0.0}],
        "estimator": {"kind": "openskill-plackett-luce"},
        "note": "unrelated metadata must survive a merge",
    }
    if rules:
        document["rules_id"] = IDENTITY
        document["rules"] = rules_identity(DEFAULT_RULES)
    return document


def test_manifest_out_without_refit_is_refused(tmp_path):
    games = _write_jsonl(tmp_path / "games.jsonl", _ladder_rows()[:6])
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(_manifest_document()), encoding="utf-8")
    before = manifest.read_bytes()

    with pytest.raises(SystemExit, match="refit"):
        refit.main(
            [
                "--games",
                str(games),
                "--bootstrap",
                "0",
                "--manifest-out",
                str(manifest),
            ]
        )
    assert manifest.read_bytes() == before


def test_refit_without_manifest_out_never_writes_the_manifest(tmp_path, capsys):
    games = _write_jsonl(tmp_path / "games.jsonl", _ladder_rows()[:6])
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(_manifest_document()), encoding="utf-8")
    before = manifest.read_bytes()

    assert (
        refit.main(
            ["--games", str(games), "--bootstrap", "0", "--json", "--refit"]
        )
        == 0
    )
    capsys.readouterr()
    assert manifest.read_bytes() == before


def test_manifest_publication_updates_levels_and_preserves_specs(
    tmp_path, capsys
):
    games = _write_jsonl(tmp_path / "games.jsonl", _ladder_rows())
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(_manifest_document()), encoding="utf-8")

    assert (
        refit.main(
            [
                "--games",
                str(games),
                "--bootstrap",
                "0",
                "--json",
                "--refit",
                "--manifest-out",
                str(manifest),
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    doc = json.loads(manifest.read_text(encoding="utf-8"))

    # The published table is the fit, not the frozen placeholder.
    assert set(doc["levels"]) == {"strong", "weak", "random"}
    assert doc["levels"]["random"] == 0.0
    assert doc["levels"]["strong"] > 0.0 > doc["levels"]["weak"]
    assert doc["levels"]["strong"] != 1.0
    assert doc["rules_id"] == IDENTITY
    assert doc["rules"] == rules_identity(DEFAULT_RULES)

    # Subjects carry the fitted values and the MLE game count, and every
    # previous field (notably ``spec``) survives the merge.
    subjects = {entry["id"]: entry for entry in doc["subjects"]}
    assert set(subjects) == {"strong", "weak", "random"}
    assert subjects["strong"]["spec"] == "ckpt:strong.pt"
    assert subjects["weak"]["spec"] == "ckpt:weak.pt"
    assert subjects["random"]["spec"] == "random"
    for id_, count in (("strong", 30), ("weak", 30), ("random", 60)):
        assert subjects[id_]["mu"] == pytest.approx(doc["levels"][id_])
        assert subjects[id_]["games"] == count
    assert subjects["strong"]["sigma"] > 0.0
    assert subjects["weak"]["sigma"] > 0.0
    assert subjects["random"]["sigma"] == 0.0
    assert doc["anchors"] == [{"id": "random", "mu": 0.0}]
    assert doc["note"] == "unrelated metadata must survive a merge"

    estimator = doc["estimator"]
    assert estimator["kind"] == "probit-mle"
    assert estimator["games"] == 60
    assert estimator["source"] == str(games)
    assert estimator["bootstrap"] == 0
    assert estimator["seed"] == 0
    assert estimator["draw_margin"] >= 0.0

    # Rungs are re-selected from the non-anchor levels under the published
    # spacing contract; the stale frozen rung list must not survive.
    candidates = {
        id_: Rating(subjects[id_]["mu"], subjects[id_]["sigma"], 0)
        for id_ in ("strong", "weak")
    }
    selection = select_rungs(
        candidates, count=5, min_spacing=100.0, max_spacing=150.0
    )
    assert [entry["id"] for entry in doc["rungs"]] == [
        rung.id for rung in selection.rungs
    ]
    assert "random" not in {entry["id"] for entry in doc["rungs"]}
    assert "published:" in captured.err
    assert "wide_gaps=" in captured.err


def test_manifest_publication_refuses_a_legacy_target(tmp_path):
    games = _write_jsonl(tmp_path / "games.jsonl", _ladder_rows()[:6])
    manifest = tmp_path / "legacy.json"
    legacy = {"version": 1, "levels": {"strong": 1.0}}
    manifest.write_text(json.dumps(legacy), encoding="utf-8")
    before = manifest.read_bytes()

    with pytest.raises(SystemExit, match="rules_id"):
        refit.main(
            [
                "--games",
                str(games),
                "--bootstrap",
                "0",
                "--refit",
                "--manifest-out",
                str(manifest),
            ]
        )
    assert manifest.read_bytes() == before


def test_manifest_publication_refuses_a_missing_target(tmp_path):
    games = _write_jsonl(tmp_path / "games.jsonl", _ladder_rows()[:6])

    with pytest.raises(SystemExit, match="manifest-out"):
        refit.main(
            [
                "--games",
                str(games),
                "--bootstrap",
                "0",
                "--refit",
                "--manifest-out",
                str(tmp_path / "missing.json"),
            ]
        )
    assert not (tmp_path / "missing.json").exists()


def _spec_rows() -> list[dict]:
    """The ladder plus a new ``searchy`` id (16 games vs ``strong``)."""
    rows = _ladder_rows()
    rows += [_row(100 + seed, ("searchy", "strong"), (100, 0)) for seed in range(12)]
    rows += [_row(120 + seed, ("strong", "searchy"), (100, 0)) for seed in range(6)]
    return rows


def test_spec_supplies_a_new_subject_spec_and_survives_a_refit(tmp_path, capsys):
    """``--spec ID=SPEC`` publishes the spec a new measured id lacks.

    The historical blocker: a fit id absent from the target manifest's
    ``subjects`` was published minimal (no spec), so the rung was unusable by
    every consumer.  A second refit without ``--spec`` must keep the spec.
    """
    games = _write_jsonl(tmp_path / "games.jsonl", _spec_rows())
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(_manifest_document()), encoding="utf-8")
    spec = "rolloutt:runs/ei2_value_t5/t_leafq/critic.pt"

    base = [
        "--games",
        str(games),
        "--manifest-out",
        str(manifest),
        "--refit",
        "--bootstrap",
        "0",
        "--json",
    ]
    assert refit.main([*base, "--spec", f"searchy={spec}"]) == 0
    capsys.readouterr()
    published = json.loads(manifest.read_text(encoding="utf-8"))
    subjects = {entry["id"]: entry for entry in published["subjects"]}
    assert published["levels"]["searchy"] == pytest.approx(subjects["searchy"]["mu"])
    assert subjects["searchy"]["spec"] == spec
    assert subjects["searchy"]["games"] == 18

    # A later refit reads the published subject, so the spec is preserved
    # without repeating --spec.
    assert refit.main(base) == 0
    capsys.readouterr()
    again = {entry["id"]: entry for entry in json.loads(
        manifest.read_text(encoding="utf-8")
    )["subjects"]}
    assert again["searchy"]["spec"] == spec


def test_spec_refuses_unknown_and_repointed_ids(tmp_path):
    games = _write_jsonl(tmp_path / "games.jsonl", _ladder_rows())

    # An id the fit never measured would be silently unmeasurable: refuse.
    document = _manifest_document()
    manifest = tmp_path / "unknown.json"
    manifest.write_text(json.dumps(document), encoding="utf-8")
    before = manifest.read_bytes()
    with pytest.raises(SystemExit, match="refused"):
        refit.main(
            [
                "--games",
                str(games),
                "--manifest-out",
                str(manifest),
                "--refit",
                "--bootstrap",
                "0",
                "--spec",
                "ghost=ckpt:ghost.pt",
            ]
        )
    assert manifest.read_bytes() == before

    # An id already in the fit but with a different published spec would be
    # repointed by the refit: refuse instead of silently changing identity.
    manifest = tmp_path / "repoint.json"
    manifest.write_text(json.dumps(document), encoding="utf-8")
    before = manifest.read_bytes()
    with pytest.raises(SystemExit, match="repoint"):
        refit.main(
            [
                "--games",
                str(games),
                "--manifest-out",
                str(manifest),
                "--refit",
                "--bootstrap",
                "0",
                "--spec",
                "strong=ckpt:other.pt",
            ]
        )
    assert manifest.read_bytes() == before


def test_spec_requires_manifest_out_and_rejects_malformed_pairs(tmp_path):
    games = _write_jsonl(tmp_path / "games.jsonl", _ladder_rows()[:6])
    with pytest.raises(SystemExit, match="--spec"):
        refit.main(["--games", str(games), "--bootstrap", "0", "--spec", "a=b"])

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(_manifest_document()), encoding="utf-8")
    base = [
        "--games",
        str(games),
        "--manifest-out",
        str(manifest),
        "--refit",
        "--bootstrap",
        "0",
    ]
    for raw, needle in (("noequals", "ID=SPEC"), ("=spec", "non-empty ID"), ("id=", "non-empty SPEC")):
        with pytest.raises(SystemExit, match=needle):
            refit.main([*base, "--spec", raw])
    with pytest.raises(SystemExit, match="duplicate"):
        refit.main([*base, "--spec", "a=b", "--spec", "a=c"])


def test_search_config_seeds_a_subject_and_survives_a_refit(tmp_path, capsys):
    """``--search-config ID=@FILE.json`` publishes the pinned identity block.

    The block is seeded next to a new measured subject and a later refit that
    omits the flag must preserve it (the identity is not part of the fit).
    """
    games = _write_jsonl(tmp_path / "games.jsonl", _spec_rows())
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(_manifest_document()), encoding="utf-8")
    config = {
        "trunc_ply": 5,
        "rollout_k": 32,
        "max_candidates": 6,
        "max_rollout_ply": 400,
        "rollout_opponent": "ckpt:runs/w2m_ctl__11__1790516900/agent.pt",
        "aggregate": "mean",
    }
    config_path = tmp_path / "search_leafq.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    spec = "rolloutt:runs/ei2_value_t5/t_leafq/critic.pt"

    base = [
        "--games",
        str(games),
        "--manifest-out",
        str(manifest),
        "--refit",
        "--bootstrap",
        "0",
        "--json",
    ]
    assert (
        refit.main(
            [
                *base,
                "--spec",
                f"searchy={spec}",
                "--search-config",
                f"searchy=@{config_path}",
            ]
        )
        == 0
    )
    capsys.readouterr()
    published = {entry["id"]: entry for entry in json.loads(
        manifest.read_text(encoding="utf-8")
    )["subjects"]}
    assert published["searchy"]["spec"] == spec
    assert published["searchy"]["search_config"] == config

    # A later refit reads the published subject, so both the spec and the
    # search config survive without repeating either flag.
    assert refit.main(base) == 0
    capsys.readouterr()
    again = {entry["id"]: entry for entry in json.loads(
        manifest.read_text(encoding="utf-8")
    )["subjects"]}
    assert again["searchy"]["spec"] == spec
    assert again["searchy"]["search_config"] == config


def test_search_config_refresh_and_refusals(tmp_path):
    games = _write_jsonl(tmp_path / "games.jsonl", _spec_rows())
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(_manifest_document()), encoding="utf-8")
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"trunc_ply": 5, "rollout_k": 32}), encoding="utf-8"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    base = [
        "--games",
        str(games),
        "--manifest-out",
        str(manifest),
        "--refit",
        "--bootstrap",
        "0",
    ]
    assert (
        refit.main([*base, "--search-config", f"searchy=@{config_path}"])
        == 0
    )
    # Refresh: a second explicit flag replaces the published block.
    refreshed = {**config, "rollout_k": 64}
    config_path.write_text(json.dumps(refreshed), encoding="utf-8")
    assert (
        refit.main([*base, "--search-config", f"searchy=@{config_path}"])
        == 0
    )
    published = {entry["id"]: entry for entry in json.loads(
        manifest.read_text(encoding="utf-8")
    )["subjects"]}
    assert published["searchy"]["search_config"] == refreshed

    # Unknown id / malformed pairs / unreadable files are refused.
    for raw, needle in (
        ("ghost=@" + str(config_path), "refused"),
        ("noequals", "ID=@PATH"),
        ("=@" + str(config_path), "non-empty ID"),
        ("searchy=nope", "ID=@PATH"),
    ):
        with pytest.raises(SystemExit, match=needle):
            refit.main([*base, "--search-config", raw])
    with pytest.raises(SystemExit, match="cannot read"):
        refit.main([*base, "--search-config", "searchy=@/missing/file.json"])
    with pytest.raises(SystemExit, match="requires"):
        refit.main(["--games", str(games), "--bootstrap", "0", "--search-config", f"searchy=@{config_path}"])
    with pytest.raises(SystemExit, match="duplicate"):
        refit.main(
            [
                *base,
                "--search-config",
                f"searchy=@{config_path}",
                "--search-config",
                f"searchy=@{config_path}",
            ]
        )


def test_search_config_requires_a_json_object(tmp_path):
    games = _write_jsonl(tmp_path / "games.jsonl", _spec_rows())
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(_manifest_document()), encoding="utf-8")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps([1, 2]), encoding="utf-8")
    with pytest.raises(SystemExit, match="JSON object"):
        refit.main(
            [
                "--games",
                str(games),
                "--manifest-out",
                str(manifest),
                "--refit",
                "--bootstrap",
                "0",
                "--search-config",
                f"searchy=@{config_path}",
            ]
        )


# -- adopted T15 calibration artifact (runs/ is gitignored) -------------------

CALIBRATION = ROOT / "runs" / "t15_mle" / "calibration.json"
CALIBRATION_BASELINE = ROOT / "runs" / "t15_mle" / "pl_manifest_baseline.json"


@pytest.mark.skipif(
    not CALIBRATION.exists() or not CALIBRATION_BASELINE.exists(),
    reason="T15 calibration artifacts are not present",
)
def test_t15_calibration_covers_all_pairs_and_flags_strict_gaps():
    doc = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    levels = json.loads(CALIBRATION_BASELINE.read_text(encoding="utf-8"))["levels"]
    order = sorted(levels, key=levels.get)  # ascending mu: random, lvl1..lvl4
    expected = {
        f"{order[i]}>{order[j]}" for i in range(1, len(order)) for j in range(i)
    }
    assert doc["pairs_covered"] == len(expected) == 10
    assert {row["pair"] for row in doc["calibration"]} == expected
    # The tie-correct half-credit meter clears the 0.05 check on every pair ...
    assert doc["over_threshold"]["half_vs_tie_aware"] == []
    # ... while the strict meter explicitly flags the two documented gaps.
    assert {
        entry.split(" ")[0] for entry in doc["over_threshold"]["strict_vs_static"]
    } == {"lvl2>lvl1", "lvl4>random"}


# -- resolution / prior-sigma / rung preservation (review H2/H3/H4/M1/M4) ----


def _resolution_rows() -> list[dict]:
    """Cross games where ``strong`` and ``weak`` share deals (plus a gauge)."""
    rows: list[dict] = []
    for seed in range(20):
        rows.append(_row(seed, ("strong", "weak"), (100, 0)))
    for seed in range(20, 30):
        rows.append(_row(seed, ("weak", "strong"), (100, 0)))
    for seed in range(30, 40):
        rows.append(_row(seed, ("strong", "random"), (100, 0)))
    return rows


def _playable(rows: list[dict]) -> list[PlayedGame]:
    return [
        PlayedGame(row["seed"], tuple(row["seats"]), tuple(row["scores"]))
        for row in rows
    ]


def test_bootstrap_levels_records_each_replicate_and_keeps_the_cis_contract():
    # Review M4: the per-replicate levels feed the ``resolution`` contrasts
    # without changing ``bootstrap_cis``'s return contract.
    games = _playable(_ladder_rows())
    config = MleConfig()
    baseline, samples = refit.bootstrap_levels(
        games,
        anchors={"random": 0.0},
        priors={},
        config=config,
        bootstrap=5,
        rng=random.Random(9),
    )

    assert set(samples) == set(baseline.ratings)
    assert all(len(values) == 5 for values in samples.values())

    ci = refit.bootstrap_cis(
        games,
        anchors={"random": 0.0},
        priors={},
        config=config,
        bootstrap=5,
        rng=random.Random(9),
    )
    assert ci == refit.percentile_cis(samples)


def test_parallel_bootstrap_replays_the_serial_stream_bit_for_bit():
    # The pool path must reproduce the serial deal draws exactly: the parent
    # replays the stream and hands each worker its contiguous range's captured
    # MT state, so the replicate mu values (not just the CI endpoints) agree
    # bit for bit between workers=1 and workers=4 on the same seed.
    games = _playable(_ladder_rows())
    config = MleConfig()
    common = {
        "anchors": {"random": 0.0},
        "priors": {},
        "config": config,
        "bootstrap": 7,
    }
    baseline_serial, serial = refit.bootstrap_levels(
        games, rng=random.Random(9), workers=1, **common
    )
    baseline_parallel, parallel = refit.bootstrap_levels(
        games, rng=random.Random(9), workers=4, **common
    )

    assert set(baseline_parallel.ratings) == set(baseline_serial.ratings)
    assert set(parallel) == set(serial)
    for id_, values in serial.items():
        assert [repr(value) for value in parallel[id_]] == [
            repr(value) for value in values
        ]

    serial_ci = refit.bootstrap_cis(games, rng=random.Random(9), workers=1, **common)
    parallel_ci = refit.bootstrap_cis(games, rng=random.Random(9), workers=4, **common)
    assert parallel_ci == serial_ci


def test_parallel_bootstrap_is_deterministic_across_pools():
    games = _playable(_resolution_rows())
    config = MleConfig()
    runs = [
        refit.bootstrap_levels(
            games,
            anchors={"random": 0.0},
            priors={},
            config=config,
            bootstrap=6,
            rng=random.Random(11),
            workers=4,
        )[1]
        for _ in range(2)
    ]
    assert runs[0] == runs[1]
    assert all(len(values) == 6 for values in runs[0].values())


def test_bootstrap_workers_above_bootstrap_replicates_matches_serial():
    games = _playable(_ladder_rows())
    config = MleConfig()
    common = {
        "anchors": {"random": 0.0},
        "priors": {},
        "config": config,
        "bootstrap": 3,
    }
    _, serial = refit.bootstrap_levels(games, rng=random.Random(2), workers=1, **common)
    _, capped = refit.bootstrap_levels(games, rng=random.Random(2), workers=8, **common)
    assert capped == serial
    assert all(len(values) == 3 for values in capped.values())


def test_bootstrap_levels_rejects_nonpositive_workers():
    games = _playable(_ladder_rows())
    with pytest.raises(ValueError, match="workers"):
        refit.bootstrap_levels(
            games,
            anchors={"random": 0.0},
            priors={},
            config=MleConfig(),
            bootstrap=4,
            rng=random.Random(0),
            workers=0,
        )


def test_bootstrap_worker_without_a_shared_context_fails_loud():
    with pytest.raises(RuntimeError, match="shared context"):
        refit._bootstrap_replicates(random.Random(0).getstate(), 1)


def _exploding_worker(state, count):
    """A stand-in bootstrap worker that always fails (fail-loud test)."""
    raise RuntimeError("worker exploded")


def test_worker_failure_propagates_instead_of_truncating(monkeypatch):
    games = _playable(_ladder_rows())
    monkeypatch.setattr(refit, "_bootstrap_replicates", _exploding_worker)

    with pytest.raises(RuntimeError, match="worker exploded"):
        refit.bootstrap_levels(
            games,
            anchors={"random": 0.0},
            priors={},
            config=MleConfig(),
            bootstrap=4,
            rng=random.Random(1),
            workers=2,
        )


def test_bootstrap_workers_cli_rejects_out_of_range_and_non_integer(tmp_path):
    games = _write_jsonl(tmp_path / "games.jsonl", _ladder_rows()[:6])

    for value in ("0", "65"):
        with pytest.raises(SystemExit, match="bootstrap-workers"):
            refit.main(["--games", str(games), "--bootstrap-workers", value])
    # argparse's int conversion rejects a non-integer before main validates it.
    with pytest.raises(SystemExit):
        refit.main(["--games", str(games), "--bootstrap-workers", "many"])


def test_bootstrap_workers_cli_one_and_over_bootstrap_match_the_default(
    tmp_path, capsys
):
    games = _write_jsonl(tmp_path / "games.jsonl", _ladder_rows())
    args = ["--games", str(games), "--bootstrap", "3", "--seed", "4", "--json"]

    assert refit.main([*args, "--bootstrap-workers", "1"]) == 0
    serial = capsys.readouterr().out
    assert refit.main([*args, "--bootstrap-workers", "4"]) == 0
    parallel = capsys.readouterr().out
    assert parallel == serial
    # workers > bootstrap truncates to one range per replicate, not an error.
    assert refit.main([*args, "--bootstrap-workers", "8"]) == 0
    capped = capsys.readouterr().out
    assert capped == serial


def test_bootstrap_workers_default_is_four():
    assert refit.parse_args(["--games", "games.jsonl"]).bootstrap_workers == 4


def test_ci_only_power_matches_the_design_formula():
    # Design §2.2: power = Phi(delta / SE - 1.96); SE = delta / 2.8016 at 80%.
    assert refit.ci_only_power(10.0, 10.0 / 2.8016) == pytest.approx(0.8, abs=5e-4)
    assert refit.ci_only_power(0.0, 1.0) == pytest.approx(0.025, abs=1e-9)
    assert refit.ci_only_power(10.0, 0.0) == 1.0
    assert refit.ci_only_power(-10.0, 1.0) < 0.025


def test_action_power_is_capped_at_half_for_a_true_plus_ten_effect():
    # Review H2: the action rule is "CI excludes 0 AND point >= +10"; at a
    # true +10 effect P(point >= +10) = 0.5, so the compound rule can never
    # reach the CI-only 80%.
    assert refit.action_power(10.0, 3.6) == pytest.approx(0.5, abs=1e-12)
    assert refit.action_power(10.0, 1.0) == pytest.approx(0.5, abs=1e-12)
    # 1.96 * SE = 7.06 < 10, so the point-estimate leg still binds.
    assert refit.action_power(13.0, 3.6) == pytest.approx(0.7976, abs=1e-3)
    assert refit.action_power(10.0, 0.0) == 1.0
    assert refit.action_power(-5.0, 1.0) == pytest.approx(0.0, abs=1e-12)


def test_prior_sigma_replaces_the_manifest_width_and_is_recorded(tmp_path, capsys):
    games = _write_jsonl(tmp_path / "games.jsonl", _ladder_rows()[:6])
    manifest = {
        "version": 1,
        "rules_id": IDENTITY,
        "levels": {"ghost": 1234.0, "random": 0.0},
        "subjects": [{"id": "ghost", "mu": 1234.0, "sigma": 150.0}],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    assert (
        refit.main(
            [
                "--games",
                str(games),
                "--prior-manifest",
                str(path),
                "--prior-sigma",
                "30",
                "--bootstrap",
                "0",
                "--json",
            ]
        )
        == 0
    )
    doc = json.loads(capsys.readouterr().out)
    # The centre is kept, only the MAP penalty width is replaced.
    assert doc["levels"]["ghost"] == pytest.approx(1234.0)
    assert doc["sigmas"]["ghost"] == pytest.approx(30.0)
    assert doc["estimator"]["prior_sigma"] == pytest.approx(30.0)

    # Without the flag nothing changes and the metadata says so.
    assert (
        refit.main(
            [
                "--games",
                str(games),
                "--prior-manifest",
                str(path),
                "--bootstrap",
                "0",
                "--json",
            ]
        )
        == 0
    )
    default = json.loads(capsys.readouterr().out)
    assert default["sigmas"]["ghost"] == pytest.approx(150.0)
    assert default["estimator"]["prior_sigma"] is None


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_prior_sigma_rejects_non_positive_or_non_finite(tmp_path, value):
    games = _write_jsonl(tmp_path / "games.jsonl", _ladder_rows()[:6])
    manifest = {
        "version": 1,
        "rules_id": IDENTITY,
        "levels": {"ghost": 1.0},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(SystemExit, match="prior-sigma"):
        refit.main(
            [
                "--games",
                str(games),
                "--prior-manifest",
                str(path),
                "--prior-sigma",
                value,
                "--bootstrap",
                "0",
            ]
        )


def test_prior_sigma_requires_a_prior_manifest(tmp_path):
    games = _write_jsonl(tmp_path / "games.jsonl", _ladder_rows()[:6])

    with pytest.raises(SystemExit, match="prior-sigma"):
        refit.main(
            ["--games", str(games), "--prior-sigma", "30", "--bootstrap", "0"]
        )


def test_prior_only_ids_never_reach_the_published_manifest(tmp_path, capsys):
    # Review H3: an id with only a prior (n == 0) is a valid table row but
    # must not be published as a level or rung candidate.
    games = _write_jsonl(tmp_path / "games.jsonl", _ladder_rows())
    document = _manifest_document()
    document["levels"]["ghost"] = 1234.0
    document["subjects"].append(
        {"id": "ghost", "spec": "ckpt:ghost.pt", "mu": 1234.0, "sigma": 150.0}
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(document), encoding="utf-8")

    assert (
        refit.main(
            [
                "--games",
                str(games),
                "--prior-manifest",
                str(manifest),
                "--manifest-out",
                str(manifest),
                "--refit",
                "--bootstrap",
                "0",
                "--json",
            ]
        )
        == 0
    )
    artifact = json.loads(capsys.readouterr().out)
    published = json.loads(manifest.read_text(encoding="utf-8"))

    # The table artifact still marks ghost as prior-only via n == 0 ...
    assert artifact["levels"]["ghost"] == pytest.approx(1234.0)
    assert artifact["n"]["ghost"] == 0
    # ... but the published study manifest excludes it everywhere.
    assert "ghost" not in published["levels"]
    assert "ghost" not in {entry["id"] for entry in published["subjects"]}
    assert "ghost" not in {rung["id"] for rung in published["rungs"]}


def test_keep_rungs_preserves_the_published_selection_and_refreshes_levels(
    tmp_path, capsys
):
    games = _write_jsonl(tmp_path / "games.jsonl", _ladder_rows())
    document = _manifest_document()
    # select_rungs would collapse this slate to a single rung (the gap is
    # below min_spacing); --keep-rungs must publish both published ids.
    document["rungs"] = [
        {"id": "weak", "mu": -1.0, "sigma": 9.0, "note": "keep me"},
        {"id": "strong", "mu": 1.0, "sigma": 9.0},
    ]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(document), encoding="utf-8")

    assert (
        refit.main(
            [
                "--games",
                str(games),
                "--manifest-out",
                str(manifest),
                "--refit",
                "--keep-rungs",
                "--bootstrap",
                "0",
                "--json",
            ]
        )
        == 0
    )
    published = json.loads(manifest.read_text(encoding="utf-8"))
    artifact = json.loads(capsys.readouterr().out)
    assert artifact["estimator"]["rungs_source"] == "kept"
    assert artifact["estimator"]["rungs"] == ["weak", "strong"]
    assert artifact["estimator"]["rungs_selection_ok"] is False
    rungs = published["rungs"]
    assert [rung["id"] for rung in rungs] == ["weak", "strong"]
    assert rungs[0]["note"] == "keep me"  # unrelated rung metadata survives
    for rung in rungs:
        # Kept rungs track the refit levels so the manifest stays coherent.
        assert rung["mu"] == pytest.approx(published["levels"][rung["id"]])
    estimator = published["estimator"]
    assert estimator["rungs_source"] == "kept"
    assert estimator["rungs_selection_ok"] is False  # 2 rungs < requested 5
    assert estimator["rungs"] == ["weak", "strong"]


def test_keep_rungs_requires_manifest_out_and_published_rungs(tmp_path):
    games = _write_jsonl(tmp_path / "games.jsonl", _ladder_rows()[:6])

    with pytest.raises(SystemExit, match="keep-rungs"):
        refit.main(["--games", str(games), "--keep-rungs", "--bootstrap", "0"])

    document = _manifest_document()  # no rungs yet
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(document), encoding="utf-8")
    before = manifest.read_bytes()
    with pytest.raises(SystemExit, match="rungs"):
        refit.main(
            [
                "--games",
                str(games),
                "--manifest-out",
                str(manifest),
                "--refit",
                "--keep-rungs",
                "--bootstrap",
                "0",
            ]
        )
    assert manifest.read_bytes() == before


def test_keep_rungs_refuses_a_rung_without_a_refit_level(tmp_path):
    # The kept selection is an invariant, not a best effort: a rung that is
    # not a published level would leave the manifest incoherent.
    games = _write_jsonl(tmp_path / "games.jsonl", _ladder_rows()[:6])
    document = _manifest_document()
    document["rungs"] = [
        {"id": "weak", "mu": -1.0, "sigma": 9.0},
        {"id": "ghost", "mu": 100.0, "sigma": 9.0},
    ]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(document), encoding="utf-8")
    before = manifest.read_bytes()

    with pytest.raises(SystemExit, match="refused"):
        refit.main(
            [
                "--games",
                str(games),
                "--manifest-out",
                str(manifest),
                "--refit",
                "--keep-rungs",
                "--bootstrap",
                "0",
            ]
        )
    assert manifest.read_bytes() == before


def test_resolution_contrasts_share_the_bootstrap_and_split_the_power(tmp_path, capsys):
    games = _write_jsonl(tmp_path / "games.jsonl", _resolution_rows())

    assert (
        refit.main(
            [
                "--games",
                str(games),
                "--bootstrap",
                "20",
                "--seed",
                "4",
                "--band",
                "strong,weak",
                "--json",
            ]
        )
        == 0
    )
    resolution = json.loads(capsys.readouterr().out)["resolution"]

    assert resolution["band"] == ["strong", "weak"]
    assert resolution["tie_credit"] == 0.5  # win_prob credits ties as half
    assert resolution["ci_source"] == "deal-bootstrap(seed=4, n=20)"
    assert resolution["target"]["delta_elo_logistic"] == pytest.approx(10.0)
    assert len(resolution["contrasts"]) == 1

    contrast = resolution["contrasts"][0]
    assert contrast["a"] == "strong"
    assert contrast["b"] == "weak"
    assert contrast["deals"] == 30
    assert 0.0 < contrast["win_prob"] < 1.0
    low, high = contrast["ci95_win_prob"]
    assert 0.0 <= low <= high <= 1.0
    assert math.isfinite(contrast["delta_elo_logistic"])

    achieved = resolution["achieved"]["strong-weak"]
    assert set(achieved) == {
        "se_elo",
        "deals",
        "power_at_delta",
        "action_power_at_delta",
    }
    assert achieved["deals"] == 30
    assert achieved["se_elo"] >= 0.0
    # Both power fields are computed from the same bootstrap SE, with the
    # documented CI-only vs compound-rule (H2) semantics.
    assert achieved["power_at_delta"] == pytest.approx(
        refit.ci_only_power(10.0, achieved["se_elo"]), abs=1e-12
    )
    assert achieved["action_power_at_delta"] == pytest.approx(
        refit.action_power(10.0, achieved["se_elo"]), abs=1e-12
    )
    assert achieved["action_power_at_delta"] <= 0.5 + 1e-12


def test_resolution_band_defaults_to_measured_non_anchor_ids(tmp_path, capsys):
    games = _write_jsonl(tmp_path / "games.jsonl", _resolution_rows())

    assert (
        refit.main(
            ["--games", str(games), "--bootstrap", "5", "--seed", "1", "--json"]
        )
        == 0
    )
    resolution = json.loads(capsys.readouterr().out)["resolution"]
    # ``random`` is the anchor, so the default band is the measured ids in
    # mu-descending order; every unordered pair is contrasted.
    assert resolution["band"] == ["strong", "weak"]
    assert len(resolution["contrasts"]) == 1


def test_band_rejects_unknown_prior_only_and_missing_bootstrap(tmp_path):
    games = _write_jsonl(tmp_path / "games.jsonl", _resolution_rows()[:6])

    with pytest.raises(SystemExit, match="bootstrap"):
        refit.main(
            [
                "--games",
                str(games),
                "--band",
                "strong,weak",
                "--bootstrap",
                "0",
            ]
        )

    with pytest.raises(SystemExit, match="band"):
        refit.main(
            [
                "--games",
                str(games),
                "--band",
                "nope,strong",
                "--bootstrap",
                "1",
            ]
        )

    manifest = {
        "version": 1,
        "rules_id": IDENTITY,
        "levels": {"ghost": 1.0},
        "subjects": [{"id": "ghost", "mu": 1.0, "sigma": 150.0}],
    }
    path = tmp_path / "prior.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SystemExit, match="prior-only"):
        refit.main(
            [
                "--games",
                str(games),
                "--prior-manifest",
                str(path),
                "--band",
                "ghost,strong",
                "--bootstrap",
                "1",
            ]
        )


def test_transitivity_is_not_computed_for_a_single_bank(tmp_path, capsys):
    # Review H4: a single bank cannot validate itself in-sample.
    games = _write_jsonl(tmp_path / "games.jsonl", _resolution_rows())

    assert (
        refit.main(
            [
                "--games",
                str(games),
                "--bootstrap",
                "5",
                "--seed",
                "1",
                "--band",
                "strong,weak",
                "--json",
            ]
        )
        == 0
    )
    transitivity = json.loads(capsys.readouterr().out)["resolution"][
        "transitivity"
    ]
    assert transitivity["status"] == "not-computed"
    assert transitivity["z"] is None
    assert transitivity["max_abs_z"] is None
    assert transitivity["flagged"] == []


def test_transitivity_leave_one_bank_out_sees_conflicting_banks(tmp_path, capsys):
    bank_a = _write_jsonl(
        tmp_path / "a.jsonl",
        [_row(seed, ("strong", "weak"), (100, 0)) for seed in range(16)],
    )
    bank_b = _write_jsonl(
        tmp_path / "b.jsonl",
        [_row(1000 + seed, ("weak", "strong"), (100, 0)) for seed in range(16)],
    )

    assert (
        refit.main(
            [
                "--games",
                str(bank_a),
                "--games",
                str(bank_b),
                "--bootstrap",
                "5",
                "--seed",
                "2",
                "--band",
                "strong,weak",
                "--json",
            ]
        )
        == 0
    )
    resolution = json.loads(capsys.readouterr().out)["resolution"]
    transitivity = resolution["transitivity"]
    assert transitivity["status"] == "computed"
    assert len(transitivity["banks"]) == 2

    pair = transitivity["pairs"]["strong-weak"]
    z_by_bank = pair["bank_z"]
    assert len(z_by_bank) == 2
    zs = list(z_by_bank.values())
    # Each bank is scored by a model fitted without it: the two opposed banks
    # produce opposite-sign z, where an in-sample gate would see ~0.
    assert zs[0] * zs[1] < 0.0
    assert math.isfinite(pair["z"])
    assert transitivity["max_abs_z"] == pytest.approx(abs(pair["z"]))

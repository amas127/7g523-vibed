# Adversarial review context

> **历史决策过程（2026-09-25/26 裁决）**：本文是「无漂移 OpenSkill 两方案」对抗审查的
> 一次性上下文，裁决见 [`report.md`](./report.md) 与 [`verified-findings.md`](./verified-findings.md)；
> 审查结论已固化为 [`ADR-0013`](../docs/adr/0013-drift-free-rating-channel.md)（保留
> `elo.fit_ratings` 在线 seam，发布绝对表改走独立 `mle.py`），本文所述仓库事实只作当时快照。

## What is under review

Two AI-generated proposals for a "drift-free OpenSkill rating system" for iteratively improved AI checkpoints:

- `design-gemini.md` — Design A: chess/AlphaZero-engine flavoured (anchors, batch "refit", SPRT gate, cross-win matrix).
- `design-grok.md` — Design B: LLM-snapshot flavoured (two-layer online OpenSkill + full-history anchored MLE, model lifecycle states, bootstrap CIs).

This is an **adversarial review**. The goal is to find what is wrong, what is missing, and what would break in practice — not to praise. Assume both designs are flawed until a claim survives independent verification.

## The actual project these designs would be applied to

Repo root: `/home/amas/.local/src/7g523` — "7鬼523", a 54-card Chinese trick/shedding card game (N seats, base 2), trained via PPO self-play; ratings are used for model selection, ladder placement, progress measurement, and human placement. Key facts:

- Ratings are **relative**; RandomBot is the single pinned gauge (`mu = 0`, ADR-0012).
- Existing rating stack (READ THESE before judging):
  - `src/seven523/elo.py` — OpenSkill Weng–Lin **Plackett–Luce** core. `fit_ratings(games, *, anchors, priors, config)` replays a chronological game record with the model's online update once per game; anchors are pinned `(mu, sigma=0)` and never update; `FitConfig` requires `tau > 0` (defaults: `mu=500, sigma=200, beta=100, tau=2` on the project scale); N-seat and ties supported; order-dependent by design.
  - `src/seven523/ladder.py` — `plan_games` / `build_ladder`: **paired deals** (same seed, seat rotation), pinned gauge + optional anchors, `cross` candidate-vs-candidate games.
  - `src/seven523/duel.py` — same-deal twin comparison + **clustered bootstrap** by deal.
  - `src/seven523/bootstrap.py` — PPO bootstrap-arena trainer (fork/train/qualify/trim/rate), **not** a statistical bootstrap; the deal-clustered bootstrap lives in `duel.py`（`report.md` §7 已纠正本文此处的误标）。
  - `src/seven523/study.py` — manifest merge; **ratings frozen by default**, explicit `refit` override.
  - `src/seven523/arena.py`, `src/seven523/league.py` (`PfspController` — PFSP league training), `src/seven523/placement/` (human placement, 10-game session, priors from trace model).
  - Decisions: `docs/adr/0006-elo-rating-seam.md` (retired BT-MAP), `docs/adr/0011-openskill-rating-core.md`, `docs/adr/0012-single-gauge-and-greedy-removal.md`; interface summary in `DESIGN.md` §2.7; human plan in `docs/human-elo-plan.md`; tests in `tests/test_elo.py`, `tests/test_ladder.py`, `tests/test_duel.py`, `tests/test_study.py`.

## Environment / ground rules for reviewers

- **READ-ONLY on the repo.** Do not modify any file. Throwaway scripts and outputs go in `/tmp` only.
- Python with `openskill 6.2.0` preinstalled: `/home/amas/.local/src/7g523/.venv/bin/python`. The openskill source is in `.venv/lib/python3.*/site-packages/openskill/` — read it to check claims rather than guessing API semantics.
- The two design texts are pasted verbatim from other AI systems; code blocks may have lost line breaks in the paste. Judge semantics, not paste formatting.
- Every finding needs evidence: a quote from a design file, a repo file/line, or a reproducible command + observed output.
- Domain ambiguity is itself a finding: Design A assumes a chess/AlphaZero engine pipeline (SPRT vs best, opening books, Stockfish anchors); Design B assumes LLM chatbot snapshots (GPT/Claude versions, task versions). The concrete target is this repo (PPO card game, human placement). Flag where the mismatch changes the answer.

## What "drift-free" is supposed to mean (to be critiqued itself)

The user's concern: with OpenSkill used naively for self-play / incremental evaluation, (i) the absolute scale moves as new checkpoints arrive, (ii) online updates are path/order-dependent, (iii) cyclic intransitivity makes a 1-D rating oscillate, (iv) ratings inflate/deflate. The system must let you tell "the model really improved" apart from "the ruler moved".

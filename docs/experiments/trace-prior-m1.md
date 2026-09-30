# T13-M1：轨迹 S1 先验离线标定（7鬼523）

> 资产状态（2026-09-25）：本报告引用的模型 checkpoint 已删除，旧路径不再可用；数值为牌型族规则变更前口径。

> **现行口径（2026-09-29）**：本文件记载 M1 的 v1（4k 语料、17 列线性）产物；发布的 `artifacts/human-elo/prior.json` 已由 **v2/R1** 取代（study10、152 列二次展开 + cell 惩罚 λ=3，sha256 `15fddff1…`），见 [`prior-opponent-correction.md`](./prior-opponent-correction.md) 与 [`prior-v2-confirmation.md`](./prior-v2-confirmation.md)；本文 sha256 与数字为**当时**口径。

> **状态：M1 已完成（2026-09-25）。一句话结论：`tools/fit_trace_prior.py` 已交付，
> 默认产物 `artifacts/human-elo/prior.json` 以 **T2** 为标签（C-6：默认基准 = T2/去收缩标定，
> 不是 manifest T1）给出岭回归 + 仿射去收缩 + `σ_traj(n)` 表；对照产物
> `artifacts/human-elo/prior_manifest_labels.json` 仅作比较。6 项测试
> `tests/test_fit_trace_prior.py` 锁定 `(a,b)`、`σ_traj` 与 schema。M2/M3 离线可做，
> M4 真人试点等 D-3（真人招募）。**
>
> 口径与计划：[`../human-elo-plan.md`](../human-elo-plan.md) 修订节「改动计划」第 1 条与里程碑 M1；
> 统计与信号依据：[`human-elo-10-games-research.md`](./human-elo-10-games-research.md) §1/§5.2（HR）；
> 裁定：C-6（默认 T2，不用 manifest T1）；ADR-0006（结果似然为最终权威、轨迹先验只加速）。
> 产物：`tools/fit_trace_prior.py`、`artifacts/human-elo/prior*.json`、`traces/study`（gitignore
> 训练数据本体）、测试；本文数字可溯源到两个 prior JSON 的 `data`/`deshrink`/`sigma_traj`/
> `calibration` 字段。

## 1. 交付物

| 交付 | 路径 | sha256 | 说明 |
|---|---|---|---|
| M1 工具 | `tools/fit_trace_prior.py` | — | 读 `traces/study`、复用 D1 特征提取器、标准化岭回归 + 去收缩 + `σ_traj` 表；CLI `fit --labels {t2,manifest}`（默认 `t2`）`--study ... --out ...` |
| **默认先验（T2）** | `artifacts/human-elo/prior.json` | `1895af98f6febecd1b601e62d9b80ff11ca5f2eb7e9bf7baf4bcd13e2e5e4ab7` | `labels: "t2_full_data"`，T2 六值标签；D3 默认入口 |
| 对照先验（manifest T1） | `artifacts/human-elo/prior_manifest_labels.json` | `78e3535c7c99b81d52d94b31257e7f2c7bab493947e10a468541e08c28f59c55` | `labels: "manifest_levels"`，refit 后 manifest 标签；**仅对照** |
| 测试 | `tests/test_fit_trace_prior.py` | — | 6 项：LOLO 去收缩回归锁定两套 `(a,b)`/`σ_traj`、标签图同时驱动 subject 与 opponent 特征、schema 往返、确定性、CLI 冒烟 |

两个 JSON 同 schema：`Prior(**doc["prior"])` 是冷启动先验（`mean=1500, sd=300`）；
`prior_for_session(doc, μ_traj, n)` 返回 `Prior(μ_traj, σ_traj(n))`，直接喂
`fit_ratings(priors=...)`，不改 `elo.py` 数学核（ADR-0006）。

## 2. 数据

- `traces/study`：**1200 局 × 6 级**（random / greedy / lvl1–lvl4，每级 200 局），
  每个 trace 经严格 replay 校验（`data.failed=0`、`verify=true`）。
- 数据指纹：`48c32672c7e96d82750d16568901a7c24f964be57d879af206ec1eaeb25b8c25`
  （两套先验的 `data.fingerprint` 相同，保证 T2/manifest 只差标签映射）。
- **T2 六值**（用当前估计器从这 1200 局复算，与 HR §1 一致到 0.1；标签自身 SE≈17–18）：

| random | greedy | lvl1 | lvl2 | lvl3 | lvl4 |
|---|---|---|---|---|---|
| 1026.9 | 1314.4 | 1128.6 | 1232.1 | 1356.1 | 1447.8 |

## 3. 管线

按 HR §5.2：`traces/study` → 每局 1 行**轨迹 S1** 特征（当时复用
`tools/measure_trace_signal.extract_features`；实现现已收进 `src/seven523/prior.py`，`measure_trace_signal` 仅再导出）→ 标准化岭回归
`g(φ) → Elo` → 等级均值上的仿射**去收缩** `f'(φ) = a + b·g(φ)` → 会话级 `σ_traj(n)` 表。

- 特征：**CLEAN + PACE 15 维**（`trick_win_rate`、`trick_point_share`、`lead_rate`、
  `pass_rate`、early/mid/late 三段 `trick_win_rate`、`mean_points_per_won_trick`、`bomb_rate`、
  `first_trick_won`、`tricks_won`、`trick_points_won`、`dug_won`、`decisions_per_trick`、
  `tricks_total`）+ 对手 Elo 锚点（中心 **1230** / 尺度 **300**）+ 与 `trick_win_rate`
  的交互项（`model.anchor`、`model.interaction`）。
- 超参：`alpha=30`、`scheme="lolo"`、`deshrink=true`、`reps=400`、`seed=0`、
  `max_session=20`（`hyperparams`）。
- 标签自洽：subject 标签与对手强度特征来自**同一标签图**（T2 模式两者都用 T2 数，
  manifest 模式两者都用 manifest 数），避免混用两套刻度。
- `σ_traj(n)` 默认 = **LOLO + 域内去收缩**（HR §5.2「表里应存 LOLO+去收缩版本作为默认」）；
  n=5/10/20 见下表。

## 4. 标定结果：T2 默认 vs manifest 对照

| 量 | 默认（T2 标签） | 对照（manifest T1 标签） |
|---|---|---|
| `labels` | `t2_full_data` | `manifest_levels` |
| 仿射去收缩 `a` | **−890.92** | −879.40 |
| 仿射去收缩 `b` | **1.7122** | 1.7060 |
| `σ_traj(5)` | **80.6** | 70.1 |
| `σ_traj(10)` | **69.6** | 57.8 |
| `σ_traj(20)` | **64.6** | 51.4 |
| LOLO RMSE | **133.9** | 128.6 |
| `m_eff`（会话等效观测） | **6.7** | 7.3 |
| 数据 / 指纹 | 1200 局、`48c32672…` | 同数据、同指纹 |

数字来源：`prior.json` / `prior_manifest_labels.json` 的 `deshrink`、`sigma_traj`、
`calibration`（`rmse_oof`、`m_eff`）。`m_eff` 按 HR 的 `(347 / s)²` 口径随 σ 反推。

## 5. C-6 口径与「T2 比 manifest 更差」的观察

- **C-6（2026-09-25 裁定）**：D3 默认基准 = **T2/去收缩标定**（HR §1/§5.2），
  **不是** manifest T1；manifest 标签是契约/历史引用（带相位偏移，已 2026-09-25 refit，
  见 LRP §4.4），新标定与 D3 先验**不得以 T1 为目标**。本工具据此把 T2 设为默认，
  manifest 只保留为 `--labels manifest` 对照模式。
- **观察（留给 M4 真人数据判定）**：对照组的 RMSE/σ 更小（128.6 vs 133.9、σ(10) 57.8 vs
  69.6），但这不是「manifest 更准」：T2 标签本身来自同一批 200 局/级、**SE≈17–18**
  （HR §1）；而 refit manifest 是 **5 seed × 400 局/锚点**的换座配对合并值，**SE≈6–7**
  （LRP §2.1：lvl1±6.1、lvl2±6.1、lvl3±6.6、lvl4±7.1）。标签噪声更大 ⇒ 岭回归的不可约
  误差更大 ⇒ RMSE/σ_traj 更大，方向符合预期。该差异**不代表 manifest 可作为 D3 目标**；
  两组数字的取舍留给 M4 用真人数据（D-3）判定。
- 两套 JSON 的 `σ_traj` 表都覆盖 n=1–20；两者 `(a,b)` 差约 11.5 Elo 的截距/0.006 的斜率，
  等级均值预测差远小于 10 局定级的 RMSE——**默认用 T2 不会改变定级量级**，但必须保持口径统一。

## 6. 测试与复现

```bash
# 测试（6 项；含 LOLO+去收缩回归锁定两套 (a,b)/σ_traj）
.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_fit_trace_prior.py

# 复现两个产物（命令来自工具 docstring）
.venv/bin/python tools/fit_trace_prior.py fit \
  --study traces/study --out artifacts/human-elo/prior.json
.venv/bin/python tools/fit_trace_prior.py fit --labels manifest \
  --study traces/study --out artifacts/human-elo/prior_manifest_labels.json

sha256sum artifacts/human-elo/prior.json artifacts/human-elo/prior_manifest_labels.json
```

## 7. 后续

- **M2** `src/seven523/placement.py`（当时写法；现为 `src/seven523/placement/` 包）：会话状态 + `select_opponent`（info/Thompson）+ 10 副
  不同牌 + 5/5 座位轮换（owner D-6=(b)）+ 停止规则 + 报告；消费
  `prior_for_session(doc, μ_traj, n)`。
- **M3** `7g523-elo` CLI + `play.py` 写对手 rung 标签（离线可做）。
- **M4** 真人试点（≥8 人、10 局定级 + 参考局）——**等 D-3 真人招募**；届时用真人数据
  校验 T2/manifest 与 bot-LOLO 的 OOD 偏差，并最终裁定 §5 的观察。

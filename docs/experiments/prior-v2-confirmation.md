# bank2 独立发牌语料复核：先验 v2（R1）相对 v1 的 RMSE/σ 优势

> **状态：独立复核完成 + 已发布（2026-09-27）**：复核期间未改 `src/`/`tests`/`artifacts`；
> 复核 CONFIRMED 后已把 `artifacts/human-elo/prior.json` 切到 study10 训练的 R1（sha256
> `15fddff1…`，对照 `prior_manifest_labels.json` sha256 `0f331414…`），v1 4k 版归档
> `runs/archive/prior-v1-4k-20260927/`。复核对象是 `docs/experiments/prior-opponent-correction.md` §4.1 推荐的
> R1（v2 = 152 列二次展开 + cell 校准惩罚 λ=3）与 v1（17 列线性）在**新发牌池**
> `traces/study-bank2/`（40k、独立 deal seed、相邻档为主）上的 transfer 表现。
> 主口径是「用 study10 训好的模型直接预测 bank2，不 refit」。
> **结论：CONFIRMED（判据 6/6 全部满足，无凑数项）。**
>
> **2026-09-29 P6 更新（搜索 rung 运行时口径）**：定级会话中只对**真正打到搜索 rung 的对局**按局排除轨迹先验（该局行不进轨迹通道并记 `prior_off_reason`），raw 局照常使用本先验；先验本身未重拟（扩语料 follow-up 见 [`../search-config-plan.md`](../search-config-plan.md) §4.3/§5）。

## 0. 结论摘要

| 指标（transfer，n=39,960） | v1（study10 训） | v2/R1（study10 训） | 变化 | 4k 发布 v1（对照） |
|---|---|---|---|---|
| RMSE | 58.30 | **53.20** | **−5.11** | 67.71 |
| residual SD | 58.11 | 53.12 | −4.99 | 43.78 |
| `m_eff=(347.44/s)²` | 35.75 | 42.78 | +7.03 | 62.97 |
| σ(5) | 27.33 | **24.61** | −2.72 | 32.09 |
| σ(10) | 19.60 | **17.69** | −1.91 | 29.92 |
| `drift_sd` | 9.05 | **6.29** | −2.76 | 26.65 |
| `cell_drift_var` | 58.94 | **30.48** | −28.47 | 503.87 |
| 逐级 \|bias\| 最大 | 15.82（lvl2） | 10.89（lvl4） | — | 75.21（w2m_plain） |

- **v2 的优势在独立发牌池上复现且方向一致**：study10 LOLO 上 v2 −5.5 RMSE、drift 7.79
  vs 9.21；bank2 transfer 上 −5.11 RMSE、drift 6.29 vs 9.05。
- 逐级校准也整体更好：v2 的 9 级 \|bias\| 全部落在 v1 的 1.5× 以内，且 lvl1 \|bias\|
  从 7.51 降到 5.71（判据要求「劣化 ≤3 Elo」，实际是改善 1.79）。
- 4k 发布 v1 的直接 transfer 再次暴露报告 §3.5 的老问题（顶部 5 级 bias −65…−75、
  `cell_drift_var` 503.9），只能作负面对照，不参与 v2/v1 比较。
- **次口径（bank2 内部重训 LOLO）**：v1 RMSE 44.53、v2 44.31（两族在分布内基本打平），
  但 v2 的 drift 15.35 vs v1 25.77、`cell_drift_var` 171.2 vs 473.5，仍明显占优。
  bank2 的相邻档设计与 anchor↔level 相关（0.942，见 §6）使「分布内 refit」数字整体偏低，
  不能当作公平的跨设计比较，故不作为判据。
- 单点诚实提醒：v2 在 lvl1 的**逐级 RMSE** 反而更高（63.17 vs 56.82），lvl2 也略高
  （59.03 vs 58.32）；其 lvl1 的 level bias 却是更好的。判据按任务定义（RMSE 总体、
  σ(5)/σ(10)、drift、逐级 bias）逐条成立，这一点写入 §7 局限。

---

## 1. bank2 设计与生成

### 1.1 设计

- **subject**（9 个，按 T2 Elo 升序）：`lvl1, lvl2, lvl3, ws_s2, pself_s2, lvl4,
  w2m_low, w2m_plain, w2m_ctl`；`random` 是唯一 pinned anchor（rating 0）。
  spec 全部取自发布表 `traces/pool10/manifest.json`（只读）。
- **规模**：每人 2220 个 twin deal × 2 局换座 = **4440 局/人**，总计 **39,960 局
  （39,960 trace + 39,960 行 `games.jsonl`）**。
- **CRN（跨 subject 共享发牌）**：全新 master RNG `random.Random(700)` 每 round
  `r`（0..2219）抽一个 deal seed（`randrange(1<<32)`），同 round 9 个 subject 共用
  该 deal；每个 deal 对 subject 排两局 `(subject, opp)` 与 `(opp, subject)`，
  `subject` 字段恒为 subject。
- **对手抽样（每人独立 RNG `random.Random(700_000 + idx)`）**，按 subject 在 9 档
  Elo 中的位置距离 `d`（self 排除）：
  `d=1 → 3.0，d=2 → 1.5，d=3 → 0.5，d=4 → 0.5，d≥5 → 0`，`random → 1.0`。
  期望（中间档）≈ 相邻 50%、d=2 25%、远档 d=3/4 ≈17%、random ≈8%。
- 实际每 subject 对手构成（2220 次抽样的比例）：

| subject | 相邻 d=1 | d=2 | 远档 d≥3 | random |
|---|---|---|---|---|
| lvl1 | 0.444 | 0.235 | 0.165 | 0.156 |
| lvl2 | 0.638 | 0.151 | 0.115 | 0.095 |
| lvl3 | 0.533 | 0.271 | 0.103 | 0.093 |
| ws_s2 | 0.530 | 0.270 | 0.122 | 0.077 |
| pself_s2 | 0.499 | 0.239 | 0.182 | 0.081 |
| lvl4 | 0.536 | 0.246 | 0.132 | 0.086 |
| w2m_low | 0.547 | 0.275 | 0.081 | 0.096 |
| w2m_plain | 0.637 | 0.165 | 0.097 | 0.100 |
| w2m_ctl | 0.470 | 0.225 | 0.158 | 0.147 |

  汇总成 4 个子集：相邻 **21,464**（53.7%）、d=2 **9,226**（23.1%）、远档 **5,134**
  （12.8%）、random **4,136**（10.4%）。

### 1.2 生成命令（可复现）

```bash
cd /home/amas/.local/src/7g523
# plan-only（先看对手构成，写 runs/bank2/plan.json；不落 trace）
nice -n 5 .venv/bin/python runs/bank2/generate.py --plan-only
# 生成（39,960 局，8 workers、cpu；命令本身用 nice -n 5）
nice -n 5 .venv/bin/python runs/bank2/generate.py --workers 8 --device cpu \
    2>&1 | tee runs/bank2/generate.log
# 标签源 = 发布表（只读），复制进 bank
cp traces/pool10/manifest.json traces/study-bank2/manifest.json
```

- `play_games(schedule, entrants, out="traces/study-bank2",
  results_out="runs/bank2/games.jsonl", workers=8, device="cpu")`；`Entrant(id,
  spec)` 9 个 + `Entrant("random","random",pinned=0.0)`。
- 生成耗时：由 trace 文件 mtime 的跨度测约 **107 s**（40k 局；对比 study10 约 105 s）。
- `runs/bank2/plan.json` 在开打前落盘（2220 个 deal seed + 每人 2220 个对手 id），
  seed 审计只依赖它即可重放。

---

## 2. 核验计数与 seed 审计

`nice -n 5 .venv/bin/python runs/bank2/verify.py > runs/bank2/verify.json`（脚本重建
plan 并逐行核对 `games.jsonl`；`runs/bank2/seed_audit.json` 是脚本写出的完整核验+审计报告）：

| 核验项 | 结果 |
|---|---|
| 每级 trace 数 | 9 级全部 **4440/4440** |
| trace 总数 / 去重路径 | **39,960 / 39,960**（无文件名覆盖） |
| `games.jsonl` 行数 | **39,960** |
| `rules_id` | 全部 **`2e36dbea44893696`**（39,960/39,960） |
| unique deal seed | **2,220** |
| 行序 = plan（seed/subject/opponent/seat） | 39,960/39,960 匹配 |
| 每行对应 trace 文件存在 | 39,960/39,960，缺失 0 |
| seed 700 重放 2220 deal | 与 plan 逐位相等，且与 `games.jsonl` seed 集合相等 |
| 语料指纹（trace+manifest sha256） | `e556bc1b…891960`（`runs/bank2/fingerprint.json`） |

**轻量 seed 审计**（`runs/bank2/verify.py`，沿用 `runs/w2m/calibration/audit_seeds.py`
的采集口径；排除 `runs/bank2` 自身）：

| 项 | 数值 |
|---|---|
| 扫描 `runs/**/*.json`（含 1 个逐行恢复的拼接 stdout） | 6,356 个文件，22,010 个 seed 值 |
| 扫描 `runs/**/*.jsonl`（每行 `"seed": N`） | 510 个文件 |
| 扫描 study10/study trace 文件名内嵌 seed | 43,960 个文件，1,288 个 seed 值 |
| **bank2 的 2220 个 deal seed 与历史 seed 的交集** | **0（CONFIRMED 无重叠）** |

审计范围覆盖了 `runs/study10/games_pass{1,2}.jsonl`（即 study10 全部 39,960 局记录的
seed 字段 = 19,980 个 deal）与 `traces/study10` 的 39,960 个 trace 文件名，因此 bank2 是
**独立发牌池**，
不是 study10 的重抽或子集。完整的 2220 个 seed 集合 sha256（排序 csv）：
`edf75433f5b6428f…`（见 `runs/bank2/seed_audit.json`）。

---

## 3. 评估口径

### 3.1 主口径：transfer（不 refit）

- `rows, stats = prior.load_rows("traces/study-bank2", prior.T2_LABELS,
  verify=False)` → **39,960 行，failed 0，每级 4440，耗时 6.8 s**；缓存
  `runs/bank2/rows.pkl`。
- 对每一行 `prior.predict_elo(doc, row, opponent_elo=row["opponent_elo_ref"])`。
- 三个模型：

| key | 路径 | sha256 | kind / expansion / cell_penalty |
|---|---|---|---|
| `v1_study10` | `runs/study10/prior_fullrr_t2.json` | `ec46fd5a…60fab4` | v1 / linear / 0 |
| `v2_study10` | `runs/study10/prior_v2_t2.json` | `474b6561…80ad234` | v2 / quadratic_pairwise / 3.0 |
| `shipped4k_v1` | `artifacts/human-elo/prior.json`（只读） | `7e1c41ab…9f2af` | v1 / linear / 0 |

- 指标：RMSE；residual SD（ddof=1）；`m_eff=(SINGLE_GAME_SE/s)²`，其中
  `SINGLE_GAME_SE=347.4355855…`（即任务写法的 347.44）；逐级偏差
  **bias = 校准均值 − label**（正 = 预测偏高）；`drift_sd`/`cell_drift_var` 用
  `prior._drift_stats`（报告 §2.1 定义，cell 均值相对同 level 均值，`drift_sd` 只统计
  n≥10 的 cell）；`sigma_traj(n)` 用
  `prior.session_prior_sd(y, preds, lv, n, reps=400, seed=0)`，报 1/2/3/5/10/20；
  子集按 `|位置距离|`：相邻 d≤1、d=2、远档 d≥3、random。
- 运行：`nice -n 5 .venv/bin/python runs/bank2/evaluate.py 2>&1 | tee runs/bank2/evaluate.log`；
  预测耗时 v1 0.5 s、v2 4.5 s、4k 0.5 s。完整结果 `runs/bank2/metrics.json`，
  表格转写 `runs/bank2/metrics.md`。

### 3.2 次口径：bank2 内部重训（可选对照）

按 `prior._cv_predictions(rows, fill, y, lv, lv, alpha=30, expansion=…,
cell_penalty=…, deshrink=True)` 做 LOLO OOF：`bank2_v1`（linear/0）与
`bank2_v2_R1`（quadratic_pairwise/3），`fill=prior.fill_values(rows)`。

---

## 4. 主结果：transfer（n=39,960，银行独立发牌）

### 4.1 总量指标

| 模型 | RMSE | residual SD | m_eff | drift_sd | cell_drift_var | 预测耗时 |
|---|---|---|---|---|---|---|
| v1（study10） | 58.3025 | 58.1089 | 35.75 | 9.0484 | 58.94 | 0.5 s |
| **v2/R1（study10）** | **53.1969** | **53.1214** | **42.78** | **6.2927** | **30.48** | 4.5 s |
| 4k 发布 v1（对照） | 67.7148 | 43.7843 | 62.97 | 26.6489 | 503.87 | 0.5 s |

### 4.2 逐级偏差（校准均值 − label，Elo）

| 模型 | lvl1 | lvl2 | lvl3 | ws_s2 | pself_s2 | lvl4 | w2m_low | w2m_plain | w2m_ctl | max\|bias\| |
|---|---|---|---|---|---|---|---|---|---|---|
| v1 | +7.51 | −15.82 | −7.55 | +4.32 | −3.34 | −12.63 | −3.76 | −8.85 | −2.70 | 15.82 |
| **v2/R1** | **+5.71** | **−9.54** | **−0.07** | +5.58 | −2.30 | −10.89 | −3.16 | −7.68 | −3.27 | **10.89** |
| 4k v1 | +3.25 | −16.58 | −24.82 | −64.82 | −68.29 | −71.62 | −74.76 | −75.21 | −72.06 | 75.21 |

### 4.3 σ_traj（bank2 行抽样，reps=400，seed=0）

| 模型 | σ(1) | σ(2) | σ(3) | **σ(5)** | **σ(10)** | σ(20) |
|---|---|---|---|---|---|---|
| v1 | 58.13 | 41.97 | 33.56 | **27.33** | **19.60** | 14.50 |
| v2/R1 | 53.18 | 38.04 | 30.84 | **24.61** | **17.69** | 12.98 |
| 4k v1 | 43.48 | 36.93 | 34.42 | 32.09 | 29.92 | 29.62 |

### 4.4 子集 RMSE / n

| 模型 | 相邻 d≤1 | d=2 | 远档 d≥3 | random |
|---|---|---|---|---|
| v1 | 59.657 / 21,464 | 58.124 / 9,226 | 57.800 / 5,134 | 51.855 / 4,136 |
| **v2/R1** | **55.164** / 21,464 | **52.789** / 9,226 | **52.105** / 5,134 | **44.303** / 4,136 |
| 4k v1 | 71.190 / 21,464 | 69.356 / 9,226 | 67.195 / 5,134 | 40.811 / 4,136 |

v2 在 4 个子集上全部优于 v1（相邻 −4.49、d=2 −5.34、远档 −5.69、random −7.55）；
4k v1 只在 random 子集上看起来「好」（40.81），但那是它的整体压缩/校准失败的副产物
（逐级 bias 达 −65…−75）。

---

## 5. 确认判据逐条

判据（任务给定，transfer 口径，v2 vs v1 同为 study10 训）：

| # | 判据 | 门槛 | 实测 | 通过 |
|---|---|---|---|---|
| 1 | RMSE 至少低 3 | v2 ≤ v1 − 3 | 53.1969 vs 58.3025，**gain 5.1056** | ✅ |
| 2 | σ(5) 不劣于 v1+0.3 | v2 ≤ v1 + 0.3 | 24.6146 vs 27.3301（≤27.6301） | ✅ |
| 3 | σ(10) 不劣于 v1+0.3 | v2 ≤ v1 + 0.3 | 17.6899 vs 19.5959（≤19.8959） | ✅ |
| 4 | `drift_sd` 更低 | v2 < v1 | 6.2927 < 9.0484 | ✅ |
| 5 | 逐级 \|bias\| 不劣于 v1 的 1.5× | 9/9 满足 | 见下表，最紧的是 w2m_ctl：3.27 vs 限 4.05 | ✅ |
| 6 | lvl1 劣化 ≤3 Elo | Δ\|bias\| ≤ 3 | **−1.79**（7.51 → 5.71，改善） | ✅ |

判据 5 明细（|bias|，限 = 1.5×v1）：

| level | v1 | v2 | 1.5×v1 | 通过 |
|---|---|---|---|---|
| lvl1 | 7.510 | 5.715 | 11.264 | ✅ |
| lvl2 | 15.816 | 9.536 | 23.724 | ✅ |
| lvl3 | 7.549 | 0.067 | 11.324 | ✅ |
| ws_s2 | 4.319 | 5.576 | 6.479 | ✅ |
| pself_s2 | 3.342 | 2.299 | 5.013 | ✅ |
| lvl4 | 12.625 | 10.887 | 18.938 | ✅ |
| w2m_low | 3.758 | 3.163 | 5.636 | ✅ |
| w2m_plain | 8.848 | 7.681 | 13.272 | ✅ |
| w2m_ctl | 2.698 | 3.271 | 4.047 | ✅ |

**总判定：CONFIRMED（6/6）。** 机器可读判定在 `runs/bank2/metrics.json`
（`verdict.overall="CONFIRMED"`，`runs/bank2/metrics.md` 同步转写）。

---

## 6. 次口径：bank2 内部重训（LOLO OOF 对照）

| 模型 | RMSE | residual SD | m_eff | drift_sd | cell_drift_var | σ(5) | σ(10) |
|---|---|---|---|---|---|---|---|
| `refit_bank2_v1`（17 列） | 44.5268 | 44.5098 | 60.93 | 25.7677 | 473.48 | 22.275 | 17.580 |
| `refit_bank2_v2_R1`（152 列 + λ=3） | 44.3105 | 44.2775 | 61.57 | 15.3513 | 171.22 | 22.467 | 17.845 |

- 分布内重训时两族 RMSE 基本打平（差 −0.22），v2 仍以 ~40% 更低的 drift 占优；
  σ(5/10) 差 ±0.3 以内。
- 为什么 in-bank refit RMSE（44.5）远低于 transfer（58.3）：bank2 是相邻档为主的设计，
  逐级平均对手 Elo 与 subject label 的相关高达 **0.942**（`lvl1 112.9 → w2m_ctl 161.8`
  逐级单调），模型在分布内可以用 anchor 当 level 代理——这正是报告 §2.6 里 `adj1`
  refit（corr 0.951，anchor-only RMSE 32.75）的同类构造性相关。因此 in-bank refit
  只能说明「新 bank 自身分布上的上限」，不能代替 transfer 优劣；本报告的判据全部
  落在 transfer 口径。
- 对应地，in-bank refit 的 v1 drift 25.77 远高于 transfer v1 的 9.05：放弃 level 身份、
  仅靠 anchor+特征拟合时，相邻档设计把「对手=等级代理」的权重放得更大，跨折的
  cell 预测摆动也随之变大。

---

## 7. 局限

1. **仍是单个 bank**：bank2 只是「一个新发牌池 + 一种新对手分布」，CRN 与 twin deal
   结构与 study10 同类；结论覆盖「study10 训 → bank2 transfer」这一条迁移轴，不等于
   对任意 deal bank 都成立。好的一面是发牌 seed 与 study10 零重叠，且对手设计从
   全 RR 换成相邻档为主，transfer 优势在这条轴上复现。
2. **placement 是近似**：本文 σ_traj 用 `prior.session_prior_sd` 在 bank2 行上等概率
   抽样（研究口径），bank2 的对手构成只是 placement 选择器的代理；未接
   `src/seven523/placement/session.py` 的完整回路（选择器 posterior、session 内更新、
   self-cell 处理），因此不能声称真实 placement σ_traj。4k v1 的 σ(20)≈29.6 明显高于
   研究口径的 19–20，主要是它的 level 偏差而非抽样口径。
3. **人类域偏移未评估**：标签（T2 probit-MLE，σ≈6 Elo）与特征都来自 bot 对局；
   真人 trace 的风格/节奏偏移、`trick_win_rate` 等特征的可迁移性不在本复核范围。
4. **逐级 RMSE 的两处反向**：v2 在 lvl1（63.17 vs 56.82）与 lvl2（59.03 vs 58.32）的
   逐级 RMSE 略差于 v1（尽管 lvl1/lvl2 的 level bias 更好）。总体 RMSE 与漂移优势
   主要来自顶部 5 级与 random 子集（v2 逐级 RMSE：ws_s2 46.98、pself_s2 50.83、
   lvl4 51.10、w2m_low 50.67、w2m_plain 52.83、w2m_ctl 50.71 vs v1 的 57.2–60.5）。
   若下游更关心 lvl1 单级精度而非整体，需要另做判断。
5. **顶部 5 级的标签间距**（1.7–5.5 Elo）小于标签 SE，bank2 沿用同一发布表；这不影响
   本次「同表、同对手 Elo」下的 v2/v1 相对比较，但限制了对这 5 级绝对分辨率的解读。
6. **未做的事**：未重跑 placement 选择器模拟、未在 bank2 上做 deal 留出 grouped CV
   （LOLO 已足够覆盖 level-transfer 轴）、未评估 GBM/其他 headroom 模型。

---

## 8. 产物与复现清单

- 数据（gitignore）：`traces/study-bank2/`（9×4440 trace + `manifest.json` =
  `traces/pool10/manifest.json` 副本）；bank 指纹
  `e556bc1bd2c1627e862db4d2ad9c220424f4746b9fc71032fb3955e350891960`。
- 脚本/日志/结果（gitignore）：`runs/bank2/` —
  `generate.py`（plan+play）、`plan.json`/`plan.log`、`generate.log`、
  `verify.py`/`verify.json`/`verify.err`、`seed_audit.json`、`fingerprint.json`、
  `evaluate.py`/`evaluate.log`、`rows.pkl`/`rows_stats.json`、
  `metrics.json`/`metrics.md`、`games.jsonl`。
- 模型产物（只读）：v1 `runs/study10/prior_fullrr_t2.json`（`ec46fd5a…`）、
  v2 `runs/study10/prior_v2_t2.json`（`474b6561…`）、发布 4k
  `artifacts/human-elo/prior.json`（`7e1c41ab…`）。
- 本报告是本次任务唯一新增的非 gitignore 文件（未 commit，git 中仍是 untracked）；
  未改 `src/`/`tests/`/`artifacts/` 现有文件，未做任何发布动作。

```bash
# 完整复现（约 2 分钟生成 + 数秒评估；评估复用 rows.pkl 时无需重新解析 trace）
cd /home/amas/.local/src/7g523
nice -n 5 .venv/bin/python runs/bank2/generate.py --workers 8 --device cpu 2>&1 | tee runs/bank2/generate.log
cp traces/pool10/manifest.json traces/study-bank2/manifest.json
nice -n 5 .venv/bin/python runs/bank2/verify.py > runs/bank2/verify.json
nice -n 5 .venv/bin/python runs/bank2/evaluate.py 2>&1 | tee runs/bank2/evaluate.log
```

# 真人 Elo 快速定级实验计划（轨迹信号）

> 探索原型（一次性，未提交）：
> [`elo_prototype.html`](../src/seven523/elo_prototype.html)（评分规则）、
> [`elo_calibration_prototype.html`](../src/seven523/elo_calibration_prototype.html)（真人定级流程）、
> [`elo_placement_prototype.html`](../src/seven523/elo_placement_prototype.html)（窗口 + 密集梯级 + 20 局结算）。
> 本文件只写计划；实验报告另见 `docs/experiments/`（已建）。
>
> **规则口径（2026-09-25）**：非炸弹比较已改为牌型族（[ADR-0007](./adr/0007-family-comparison.md)）。本文引用的 `traces/study` T2 标签、`prior.json` 与全部 RMSE/Elo 数字均为旧 `tier` 口径产物；在 [plans.md](./plans.md) T15 重标定完成前，不得与新规则结果混比，也不得直接用于新规则下的定级。

## 修订（2026-09-25）：10 局定级结论与改动计划（先读）

> 依据：[`experiments/human-elo-10-games-research.md`](./experiments/human-elo-10-games-research.md)
> （全部数字、仿真与复现命令在该报告 §7/§10/§11）；阶梯标签复测见
> [`experiments/ladder-rerating-paired.md`](./experiments/ladder-rerating-paired.md)。
> 本节**替代**下文 §2–§4、§6–§8 的旧假设（20 局 ±50）：§8 的 4 项待定决策已由本节/HR 关闭或转移（分档粒度→LRP 实测间距与 §4 D-2；窗口默认→HR §5.3；S2 口径→不做；分差通道→σ×2 规则）；§5 的工程约束（阻尼牛顿+钳制、秤砣、按局整群重采样）仍然有效，但其中按旧 20 局纯结果口径举的窗口示例（如 ±50→W≈185）以本节与 HR §5.3 为准；§0–§1 仍作背景。**默认标签基准 = T2/去收缩标定（研究 §1/§5.2），不是 manifest T1**——T1 仅作历史/契约引用（refit 前带相位偏移；已 2026-09-25 refit，见 LRP §4.4），新标定与 D3 先验不得以 T1 为目标。

### 结论（量化）

- **10 局做不到「自身水平 ±50」。** 推荐组合（S1 去收缩轨迹先验 + 胜负 + σ×2 分差，
  BT-MAP）10 局 RMSE ≈ **54**（grouped 乐观）/ **72**（LOLO 诚实）；±1 档（±100）命中
  **83–93%**、同档（±50）**51–65%**；诚实 95% CI **±100–133**。
- RMSE≤50 需 **15–27 局**；95% CI 收进 ±50 需 **63–100 局**（纯结果口径约 91 局）。
- 10 局的正确产品语义：**点估计 + 诚实 CI + 最近档 + `provisional` 标记**；后台继续
  对局直到 `CI ≤ 50` 再落最终档。

### 对原计划的修订

| 原计划 | 修订 |
|---|---|
| §2 S2 为主力、S1 先跑通 | **S1 为主力**（逐墩特征 + 去收缩标定）；S2 无增量、弃用（除非先做价值头对真人的标定）|
| §2 S3 辅助 | **不用于评分**：grouped m_eff 47 但 LOLO 仅 8（风格泄漏、可被模仿）；最多用于选对手/风格 |
| §3.5 混合估计器（条件） | **落地为 D3**：`fit_ratings` + `Prior(μ_traj, σ_traj(n))` + `FitConfig(margin=(c, 2σ))`；10 局窗口=None，SE 用联合 Hessian |
| §4 验收：20 局 ±1 档 ≥95% / ±50 ≥80% | 10 局改为：P100 ≥ 83%、P50 随局数增长（15–27 局到 RMSE≤50）；**必须加真人 OOD 验证**（bot LOLO 只是代理）|
| §6 对手：系统推荐 | 自适应 `info`（argmax 后验 p(1−p)）；前 1–2 局 `Thompson` 防先验偏；>1600 档位信息塌缩，需 handicap 或更强 bot |
| 座位/牌堆 | **10 副不同牌 + 5/5 座位轮换**（owner D-6=(b)，2026-09-25）：体验自然，但**保留座位相位残差**（审计跨序 sd≈√2×SE）；定级局逐局记录座位与对手 rung id，报告标 `provisional`；研究曾倾向冻结牌堆 twin（HR §6.3），此处以 owner 的产品取舍为准 |

### 改动计划（文件级）

1. 新增 `tools/fit_trace_prior.py` + `artifacts/human-elo/prior.json`：离线标定——S1 特征
   （复用 `tools/measure_trace_signal.extract_features`）、岭回归、等级均值上的仿射
   「去收缩」(a,b)、`σ_traj(n)` 表（默认 LOLO + 去收缩）。**✅ 已交付（2026-09-25）**：
   默认 `prior.json` 用 **T2 标签**（sha256 `1895af98…`）a=−890.92、b=1.7122、
   σ(5/10/20)=80.6/69.6/64.6、LOLO RMSE 133.9（m_eff 6.7）；对照
   `prior_manifest_labels.json`（sha256 `78e3535c…`）a=−879.40、b=1.7060、
   σ=70.1/57.8/51.4、RMSE 128.6；6 项测试 `tests/test_fit_trace_prior.py`。
   细节与 C-6 口径见 [`experiments/trace-prior-m1.md`](./experiments/trace-prior-m1.md)。
2. 新增 `src/seven523/placement.py`：会话状态 + `select_opponent`（info/Thompson）+ 10 副
   不同牌 + 5/5 轮换调度（owner D-6=(b)）+ 停止规则（`CI≤50` 或 10 局）+ 报告（点估计/CI/最近档/provisional/
   两通道权重）。**✅ 已落地（2026-09-25）**：+ 24 项测试；冒烟产物 `traces/sessions/smoke_m23/`；
   细节见 [`experiments/placement-m2m3.md`](./experiments/placement-m2m3.md)。
3. 新增 `7g523-elo` CLI（`pyproject.toml [project.scripts]` + 薄壳）。**✅ 已落地（2026-09-25）**：
   `7g523-elo = "seven523.placement:main"`；`--help` 与 `--simulate` 冒烟通过。
4. `src/seven523/play.py`：把对手等级 id 写进 `players` 标签（如 `opponent:lvl3@seat1`），
   让真人 trace 能按对手强度标定（现仅写 `"贪心 bot"`）。**✅ 已落地（2026-09-25）**：
   `opponent_identity` 写 `anchor:greedy@seatN` / `opponent:lvlN@seatN`（+3 测试）。
5. （可选）真人 trace 的 rung 元数据 sidecar/manifest 记录。
6. 测试：`placement.select_opponent` 的 Fisher/单调性；`fit_trace_prior` 的留一等级标定
   回归（锁定 `(a,b)`、`σ_traj` 表）；CLI 冒烟；`elo.py` 现有测试零改动。
7. **不做**：`elo.py` 数学、S2 逐决策 regret 模块、S3 评分。

### 前置与依赖

- **真人标定是硬前置**：bot-LOLO 的 OOD 偏差未知；先做 ≥8 人、10 局定级 + 60–100 局
  参考局的验证（研究 §9.1），再决定是否上线。
- **阶梯标签 refit 已执行**（2026-09-25，LRP §4.4）：manifest levels 现为 lvl1 1146.3±6.1、lvl2 1219.2±6.1、lvl3 1363.6±6.6、lvl4 1429.7±7.1；D3 先验仍默认 T2/去收缩标定（研究 §5.2），不以 manifest T1 为目标。
- **D-6=(b) 的代价需量化**：10 副不同牌 + 5/5 轮换会引入座位相位残差，研究里 10 局 RMSE≈54–72/CI±100–133 是在 bot trace 上估的；建议在 M1/M2 完成后用轮换调度跑一次同样的离线仿真，若精度明显变差则报告上调推荐局数或标 provisional。
- 真人非平稳、梯级上限（>1600 信息塌缩）仍是开放问题（研究 §9.2）。

### 里程碑（替代 §7）

| 里程碑 | 内容 | 预估 |
|---|---|---|
| M1 | `tools/fit_trace_prior.py` + prior.json（离线标定 + LOLO 自测）**✅ 已交付（2026-09-25）**：T2 默认 `prior.json` + manifest 对照 + 6 项测试；σ(5/10/20)=80.6/69.6/64.6、LOLO RMSE 133.9（见 [`experiments/trace-prior-m1.md`](./experiments/trace-prior-m1.md)）| 1 天 |
| M2 | `src/seven523/placement.py` + 单测（含 10 副不同牌 + 5/5 轮换调度、逐局座位记录）**✅ 已交付（2026-09-25）**：24 项测试；info/Thompson 选档、两通道 BT-MAP（margin=(0.143,89)）、逐局 trace + `session.json`；产物 `traces/sessions/smoke_m23/`（见 [`experiments/placement-m2m3.md`](./experiments/placement-m2m3.md)）| 1 天 |
| M3 | `7g523-elo` CLI + `play.py` rung 标签 + 冒烟 **✅ 已交付（2026-09-25）**：`7g523-elo` 入口 + `opponent_identity`（`anchor:greedy@seatN`/`opponent:lvlN@seatN`）+3 测试；冒烟 `定级：1287 ± 182，最近档 greedy（1315），provisional`；全套 418 passed；`elo.py` 零改动（sha256 `80641f13…`）| 0.5 天 |
| M4 | 真人试点 ≥8 人（10 局定级 + 参考局），标定/评估拆分，出 RMSE/覆盖/档位命中；**仍等 D-3 招募** | 2–3 天 |

## 0. 摘要

- 只吃胜负的定级：每局 1 个伯努利，20 局的 95% CI 就是 **±150**（窗口 W=20 时这是精度地板，不是过渡状态）。
- 换成「胜负 + 分差」只值约 20% 的 RMSE 降幅（σ≈45，由 DESIGN.md 真实数据标定）：20 局 → ±131。
- 要让 20 局给出 ±50，唯一还没用上的信息源是**轨迹本身**：一局有几十个决策。
- 本计划：先离线量化三种轨迹信号的信息量（`m_eff`），再决定是否实现「轨迹先验 + 结果似然」的混合估计器。
- 第一交付物：`tools/measure_trace_signal.py`，不改训练、不改环境、不动现有引擎。

## 1. 问题与现状

**目标**：真人在人机对局中 20 局左右拿到可用评级（档位 + CI）。

**已实测**（原型内批量实验，浏览器中实时计算）：

| 方案 | 20 局 RMSE | \|误差\|≤100（±1 档） | 同档命中 |
|---|---|---|---|
| 胜负 | ~80 | ~77% | ~41% |
| 胜负 + 分差 | ~65 | ~86% | ~47% |

窗口地板（打满 400 局后只取最后 W 局）：

| W | 胜负 RMSE / 理论 CI | 分差 RMSE / 理论 CI |
|---|---|---|
| 20 | ~76 / ±152 | ~68 / ±131 |
| 50 | ~49 / ±96 | ~43 / ±83 |
| 200 | ~25 / ±48 | ~21 / ±42 |
| 全部历史 | ~18（随局数继续降） | ~15 |

**信息换算**（胜负尺度，全历史窗口）：

```
95% CI 半宽 ≈ 680 / √n,     n = 等效独立伯努利观测数
m_eff = 每局等效观测数
n = 局数 × m_eff
```

| m_eff | 20 局 CI | 达到 ±50 需要 |
|---|---|---|
| 1 | ±152 | ~185 局 |
| 5 | ±68 | ~37 局 |
| 10 | ±48 | ~19 局 |
| 20 | ±34 | ~9 局 |

**结论**：20 局 ±50 可行 ⇔ `m_eff ≥ ~9`（`680/√(20·m_eff) ≤ 50`）；`m_eff ≥ 5` 时 20 局约 ±68。
`m_eff` 是经验量，必须实测；局内相关 ρ 决定
`m_eff = m / (1 + (m−1)ρ)`，不能把一局内的 m 个观测当独立。

## 2. 候选信号（定义、依赖、风险）

### S1 逐墩胜负（per-trick；下称「轨迹 S1」）

- 定义：每墩的赢家（收分者）；每局特征 = 墩胜率、墩分差、（可选）按阶段拆分的墩胜率。
- 依赖：无神经网络；`trace.py` 已记录每步与墩结果。
- 标定：在已知等级的 bot 轨迹上回归 `特征 → Elo`。
- 风险：局内强相关；「赢墩」不等于「拿分」；对局节奏（撬底）会让后期墩权重不同。
- 预期：`m_eff` 2–7。

### S2 决策 regret（value-based，主力候选）

- 定义：回放每个决策点状态 `s`，用价值网络算
  `regret_t = V(最优动作 | s) − V(人类动作 | s)`；每局聚合（均值、P95、累计）。
  对应棋类的 centipawn loss。
- 依赖：可信的价值模型（最强 ckpt 的 value head，`networks.py` 的 `Agent`）。
- 标定：bot 轨迹上 `regret → Elo` 回归；用留出等级/种子验证。
- 风险：价值网络对人类离策略状态的泛化；若对 bot 自身都校准不好，直接放弃。
- 预期：`m_eff` 5–20（潜在最大，也最重）。

### S3 策略一致度（policy agreement，仅作辅助）

- 定义：`log P_k(a_实际 | s)` 在各级策略下的均值向量；分类/回归到 Elo。
- 依赖：多级 ckpt 策略（`networks.py`）。
- 风险：测的是**风格相似度**而非强度；可被模仿刷分；离策略偏差大。
- 预期：`m_eff` 未知，容易虚高。不单独使用。

**顺序**：先 S1 跑通全流程（最便宜），再 S2 做主力量，S3 只加进特征向量看增量。

## 3. 实验设计

### 3.1 机器人阶梯（前置）

- 需要 4–6 个等级、相邻约 100–150 Elo。现有：RandomBot(1000)、GreedyBot(1315)；
  其余用 `7g523-train` 训练（300k 步约 3 分钟/个，见 `docs/training.md`）。
- 命名与评分记录：`runs/lvl_<tag>/agent.pt`；评分先由窗口 MAP 互打得到，Random/Greedy 固定为秤砣。
- 要求：每个等级至少与两个不同锚点各打 100 局以上，避免阶梯局部失真。

### 3.2 数据采集

- 用脚本化对局（`Match` + `policies`），**不用人机**，保证给定 seed 可复现。
- 每个等级 L：对锚点与其他等级共 N ≥ 200 局，逐局 `build_trace` 落盘：
  `traces/study/<L>/<seed>.json`。
- trace 需含：`rules`、`deal`、每步座位/动作/花色/墩结果、`final_scores`、双方身份。
  （现有 `trace.py` 字段已满足；如需 `level` 元数据，用 `players`/文件名承载。）

### 3.3 特征计算

- 回放：`load_trace` → `state_from_snapshot` → 逐步重放（与 `7g523-play --replay` 相同的校验路径）。
- 在每个**受评座位**的决策点取 `Game.view(state, seat)`，前向 `Agent`（`torch.no_grad()`，CPU 足够）。
- 价值：`V(s)` 与各动作价值（若只有状态价值，则用 `V(s) − V(s')` 或策略 log-prob 近似替代，
  实现时在报告中注明）。
- 每局一行特征（CSV schema）：

```
trace_id, level_id, level_elo_ref, opponent_id, opponent_elo_ref, seat,
result(win/loss), final_score_own, score_diff,
tricks_total, tricks_won, trick_win_rate,
decisions, mean_regret, p95_regret, top_action_match_rate,
agree_l1..agree_lN (各级策略平均 log-prob),
phase_open_trick_rate, phase_mid_trick_rate, phase_end_trick_rate
```

### 3.4 标定与信息量评估

- 单调性/可分性：特征均值 ± SE 随等级作图；Spearman ρ、两两 AUC。
- `特征 → Elo`：线性或序数回归（分组交叉验证：按对局种子/等级留出）。
  报告单局残差 SD `s`（Elo 尺度）。
- 等效观测数：

```
m_eff = (347 / s)²        # 347 = 胜负单局的 Elo 尺度 SE；s 为特征单局残差 SD
```

- Bootstrap（按局整群重采样，避免把局内步当独立）给出 `m_eff` 的区间。
- 输出：`m_eff(S1/S2/S3)` 点估计 + 区间，换算成下表：

```
局数          10     20     50     100
m_eff = 2   ±152   ±108   ±68    ±48
m_eff = 5   ±96    ±68    ±43    ±30
m_eff = 10  ±68    ±48    ±30    ±21
```

### 3.5 混合估计器（若 `m_eff ≥ 3` 才做）

- 先验：轨迹指标 → Elo，带标定噪声 `σ_traj`（来自 3.4 的残差）。
- 似然：窗口内结果似然（胜负 logistic + 分差高斯），窗口 W 按目标精度选
  （W ≈ (680/目标)²，全历史窗口另说）。
- 后验：MAP + CI；轨迹先验在局数增加后自然被结果似然取代。
- 在线显示：窗口估计；结论值 = 后验 mean ± CI，且必须带 `m_eff`/校准版本的元数据。

## 4. 交付物与验收标准

| 编号 | 交付物 | 内容 |
|---|---|---|
| D1 | `tools/measure_trace_signal.py` | 输入 ckpt 列表与对手配置，输出特征 CSV 与 m_eff 表 |
| D2 | `docs/experiments/trace-signal-report.md` | 单调性、标定、m_eff、结论与建议 |
| D3 | （条件）`src/seven523/elo.py` + `7g523-elo` | 混合估计器 + CLI（读 traces，输出 `R ± CI`、档位、下一步建议） |

**验收（go / no-go）**（已由顶部修订版替代，以下保留原文）：

- S1/S2 至少一个满足：随等级单调（Spearman ≥ 0.8）、`m_eff ≥ 3`；
- 若做 D3：20 局后 ±1 档命中 ≥ 95%，或 ±50 命中 ≥ 80%；
- 结果似然始终为最终权威；轨迹通道只做加速，不覆盖结果。

## 5. 工程约束（实现 `elo.py` 时）

- 拟合必须用**阻尼牛顿 + 区间钳制**：朴素牛顿在 p→0/1（全胜/全负 vs 远档对手）时会过冲发散，
  原型中实测把真值 1100 的玩家算到 +39000。阻尼步长上限 300，评分钳制 [400, 2600]。
- 窗口语义：只取最近 W 局；W=∞ 表示全历史。窗口是精度地板，不是收敛加速器。
- 秤砣：Random/Greedy（或任选锚点）评分固定；新 checkpoint 从父模型评分起步。
- 数据纪律：所有对局强制入库、对手由系统推荐；否则轨迹特征与结果似然都有偏。
- 局内相关：任何按步聚合的统计量都要按局整群抽样/重采样。

## 6. 风险与开放问题

1. **离策略**：人类打法与 bot 分布不同，S2 的 value 可能对人类状态系统性偏差。
   缓解：先看 regret 在「低等级 bot 对高等级 bot」上的校准，再做人类。
2. **相关性**：局内相关、同一牌型跨局相关；用分层模型或按局聚合，避免假 CI。
3. **刷分**：S3 可被模仿；对外天梯只用结果通道计分，轨迹只用于个人定级/匹配。
4. **座位与先手**：亮牌定先可能带来座位效应；采集时强制轮换座位并把它作为协变量。
5. **非平稳**：真人会进步/手生；窗口/遗忘因子配合轨迹先验，重估间隔可配置。
6. **计算量**：多级训练与逐招前向的成本；先 S1 无网络跑通，再评估 S2 的预算。
7. **可解释性**：轨迹指标必须给出来源（哪几步、什么类型失误），否则玩家不信任分数。

## 7. 里程碑（已由顶部修订版替代）

| 里程碑 | 内容 | 预估 |
|---|---|---|
| M1 | S1 特征 + 报告模板；用 Random/Greedy traces 跑通全流程 | 0.5 天 |
| M2 | 训练 4–6 级 ckpt；采集 5 级 × 200 局；算 S1/S2 的 m_eff | 1–2 天 |
| M3 | 依据 m_eff 决定是否做混合估计器；若是，写 `elo.py` + CLI | 1–2 天 |
| M4 | 真人 trace 接入（`7g523-play --save-trace`）；A/B「纯结果 vs 混合」在 10/20 局的表现 | 1 天 |

## 8. 待定决策（已由顶部修订节关闭/转移：分档粒度→LRP 实测间距与 §4 D-2；窗口默认→HR §5.3；S2 口径→不做；分差通道→σ×2 规则；以下保留原文作历史）

- 阶梯分档粒度：100 还是 150 Elo？（影响 20 局可交付的档位分辨率）
- 窗口默认值：按目标精度反推（胜负：±50 → W≈185，±25 → W≈740；分差：±50 → W≈130，±25 → W≈550）。
- S2 的价值口径：状态价值差、动作价值、还是策略 log-prob 加 entropy 修正？
- 是否保留「分差」通道，还是并入 S2。

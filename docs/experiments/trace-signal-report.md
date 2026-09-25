# 轨迹信号实验报告（D2）

> 资产状态（2026-09-25）：本报告引用的模型 checkpoint 已删除，旧路径不再可用；数值为牌型族规则变更前口径。

> **状态：M1 管线自检（Random / Greedy 脚本轨迹）。**
> 本报告的工具、数据与数字都能复现，但 `m_eff` 只有两个等级，**不是** go/no-go 结论；
> M2 的 4–6 级 checkpoint 阶梯 + 5 级 × 200 局采集才产生可用于决策的 `m_eff(S1/S2/S3)`。
> 计划与验收标准见 [`../human-elo-plan.md`](../human-elo-plan.md)。

> **勘误（2026-09-25，manifest refit 执行后）**：`traces/study/manifest.json` levels 已按
> `ladder-rerating-paired.md` §4.2/§4.4 refit（lvl1 +26.4、lvl2 −13.7、lvl3 +4.4、
> lvl4 −59.7；当前契约值见 `README.md` §0.1）。本报告 §1 参考 Elo、§2 标定、§3.2
> 「相邻级距 100–150」与 §4 的「间距 113–130」等数字属旧相位口径，**标定数值作废**；待
> D-3 真人试点恢复时按新契约重跑。

## 1. 方法

交付物 D1：[`tools/measure_trace_signal.py`](../../tools/measure_trace_signal.py)。

- **生成**：`play_game`（与 `7g523-play --save-trace` 同一录制路径）跑脚本化对局；
  受评座位即 `human_seat`，对手为锚点 bot。每个等级 200 局，座位轮换；
  同一 `seed` 在不同等级生成**同一副牌**（成对设计，分组 CV 按种子留出整副牌）。
- **特征（S1）**：从轨迹逐步读回受评座位的逐墩胜负、墩分、过牌/炸弹/先手率、
  分阶段墩胜率等；读取前先走 `replay_trace` 的严格校验（与 `7g523-play --replay` 同一路径），
  损坏的轨迹直接失败而不是静默进入标定。
- **标定**：`features -> Elo` 岭回归（特征标准化，alpha 由嵌套分组 CV 选），
  按 seed 分 5 折取袋外预测，`s = SD(y − ŷ_oof)`，`m_eff = (347 / s)²`；
  CI 用按局整群 bootstrap（1000 次）。

### 数据

| 等级 | 策略 | 参考 Elo | 局数（进入标定） |
|---|---|---|---|
| random | `RandomBot` | 1000 | 196 / 200 |
| greedy | `GreedyBot` | 1315 | 197 / 200 |

参考 Elo 暂用计划 §3.1 的锚点值（Random 1000、Greedy 1315）；M2 有了阶梯后应改由窗口 MAP 互打重标定，
并写进 `manifest.json`。7 行因 S1 向量含空值（如整局未赢墩导致 `mean_points_per_won_trick` 无定义）被完整-case 剔除。

`artifacts/trace-signal/features.csv` 共 400 行，列 schema 见工具 `FEATURE_COLUMNS`（计划 §3.3 的 S1 部分）。

## 2. 结果

### 2.1 可分性（两等级）

AUC 方向：greedy（高等级）特征值 > random（低等级）的概率；`< 0.5` 表示该特征反向。

| 特征 | random | greedy | AUC |
|---|---|---|---|
| `trick_win_rate` | 0.395 ± 0.014 | 0.636 ± 0.016 | 0.793 |
| `trick_point_share` | 0.416 ± 0.015 | 0.621 ± 0.017 | 0.733 |
| `mean_points_per_won_trick` | 6.24 ± 0.25 | 8.73 ± 0.32 | 0.673 |
| `pass_rate` | 0.364 ± 0.007 | 0.133 ± 0.006 | 0.042 |
| `bomb_rate` | 0.017 ± 0.002 | 0.011 ± 0.002 | 0.454 |
| `lead_rate` | 0.458 ± 0.014 | 0.667 ± 0.014 | 0.775 |
| `early_trick_win_rate` | 0.432 ± 0.019 | 0.609 ± 0.022 | 0.677 |
| `mid_trick_win_rate` | 0.407 ± 0.018 | 0.641 ± 0.020 | 0.729 |
| `late_trick_win_rate` | 0.349 ± 0.020 | 0.656 ± 0.028 | 0.741 |
| `decisions_per_trick` | 1.610 ± 0.020 | 2.408 ± 0.055 | 0.831 |
| `tricks_total` | 18.58 ± 0.27 | 13.20 ± 0.31 | 0.182 |

观察：

- S1 确实有信号，但最强的两个分离量（`decisions_per_trick`、`tricks_total`）是**对局节奏**，
  不是直接强度：Greedy 会更快撬底结束（局更短、每墩更多轮），换对手/换规则时未必稳定。
- 干净的强度量是 `trick_win_rate` / `trick_point_share` / `lead_rate`；`pass_rate` 反向等效，
  分离度最高（1 − AUC = 0.958）。
- 单调性：只有 2 个等级，Spearman 平凡（ρ=1）；跨等级的单调性要等 M2。

### 2.2 标定

| 量 | 值 |
|---|---|
| 袋外残差 SD `s` | **88.7 Elo**（95% CI 82.0–95.2） |
| `m_eff = (347 / s)²` | **15.4**（95% CI 13.3–18.0） |
| 袋外 RMSE | 88.5 Elo |
| 样本内 R² | 0.696（9 特征岭回归，alpha 由嵌套分组 CV 选） |

### 2.3 `m_eff` 到局数 → 精度

`95% CI 半宽 ≈ 680 / √(n · m_eff)`，按点估计 `m_eff = 15.4`：

| 局数 | 10 | 20 | 50 | 100 |
|---|---|---|---|---|
| ±CI | 55 | 39 | 25 | 17 |

**这不是可用结论。** 目标「20 局 ±50」要求 `m_eff ≥ ~9`；本数字形式上达标，但见 §3 的偏差说明。

## 3. 局限（为什么 M1 的 `m_eff` 不能当作 go/no-go）

1. **等级内 y 是常数。** 标定目标 `level_elo_ref` 在等级内不变（同一策略），
   残差只反映「特征噪声漏进预测」，而不是「特征对连续真人能力的定位误差」。
   完美的等级分类器会让 `s = 0`、`m_eff = ∞`，这显然不代表 20 局 ±0。
   已实测该口径对 y 的人为设定敏感：同一次 3 级试跑中把 Greedy 复制成两个不同参考
   Elo（1315 与 1450），`s` 从 88.7 变到 152.4、`m_eff` 从 15.4 变到 5.2——特征完全没变，
   变的只是标签。M2 需要 4–6 个真等级（且真人定级时是真连续能力），这个口径才有意义。
2. **等级间距大（315 Elo）。** 两个一眼可分等级的 `m_eff` 必然乐观；M2 的相邻级距 100–150 才是工作区间。
3. **离策略 / 节奏混淆。** 特征在脚本 bot 对局上标定，真人打法不同（计划风险 1）；
   节奏类特征（`decisions_per_trick`、`tricks_total`）还会随对手/规则漂移，不宜进最终模型。
4. **相关未完全消除。** 按局聚合已避免把局内步当独立；成对设计 + 按种子分折避免了跨等级同一副牌的泄漏。

## 4. 结论与建议

- ✅ D1 管线跑通：生成 → 轨迹校验 → S1 特征 CSV → 标定 → `m_eff` + bootstrap CI，400 局约 8 秒。
  每一步都可单独运行（`generate` / `features` / `calibrate`）；随机策略单等级先跑通，再成对加入 Greedy。
- ⏭ **下一步（M2）**：训练 4–6 级 ckpt（相邻 100–150 Elo），每级 vs 两个锚点各 ≥100 局，
  用本工具重跑；届时 `summary.json` 的 `conclusive=false` 才会翻真。锚点评分用窗口 MAP 重标定。
  **已执行**：4 级 ckpt 已训练、评级并冻结进 manifest（5 级梯级 random + lvl1–4，间距 113–130）；
  同时发现 vs Greedy 的强度平台（~60k 步饱和，self-play 未破 ~1490 Elo），
  见 [`ladder-report.md`](./ladder-report.md)。
- ⏭ S2（价值 regret）在特征计算处预留：轨迹可经 `replay_trace` 重建状态，接 `Agent` 价值头即可；
  先验证 regret 在「低等级 vs 高等级」bot 上的校准，再谈真人。
- 不建议现在做 D3：先用 M2 的 5 级 `m_eff` 决定，避免把两等级的乐观数字带进混合估计器。

## 5. 复现

```bash
# M1 成对阶梯（Random + Greedy）
uv run python tools/measure_trace_signal.py run \
    --subject random --subject greedy --games 200 --seed 0 \
    --study traces/study --artifacts artifacts/trace-signal

# 或分步
uv run python tools/measure_trace_signal.py generate --subject random --subject greedy --out traces/study
uv run python tools/measure_trace_signal.py features --study traces/study --out artifacts/trace-signal/features.csv
uv run python tools/measure_trace_signal.py calibrate --features artifacts/trace-signal/features.csv

# M2 接 checkpoint 等级（从 manifest/参数取参考 Elo）
uv run python tools/measure_trace_signal.py run \
    --subject random --subject greedy --subject lvl3=ckpt:runs/lvl3__0__.../agent.pt \
    --level lvl3=1450 --games 200 \
    --study traces/study --artifacts artifacts/trace-signal
```

产物（均已 gitignore）：`traces/study/<level>/*.json`、`traces/study/manifest.json`、
`artifacts/trace-signal/{features.csv,summary.json}`。真实人机轨迹用
`7g523-play --save-trace` 保存后，可直接 `features --study <目录>` 复用同一条特征流水线。

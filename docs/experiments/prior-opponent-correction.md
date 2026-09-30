# S1 轨迹先验的对手强度修正（选项 D）：诊断、候选实测与推荐

> **状态：离线研究完成（2026-09-27；未改 `src/`/`tests/` 现有文件、未 commit、未触碰发布
> 产物）。** 研究在 `traces/study10/`（39,960 条 revision-3 trace）与 `traces/study/`
> （4,000 条已发布 study）上完成；分析脚本在 `runs/study10/research/`（gitignore），
> 全量数字与复现命令见 §8。**结论：全 RR LOLO RMSE 58.41 的主要来源不是对手漂移，
> 而是「衰减系数 × 去收缩放大」；对手漂移是真实但次要的项（cell_drift² 95 / MSE
> 3412）。在此口径下没有「只修对手项、其余不变」的免费午餐：所有减漂移的改法都会
> 以 level 偏差或 RMSE 为代价。推荐把修正做成一个**可选的 v2 先验**（二次特征展开
> + cell 校准惩罚 λ=3），它以新 `kind` 落地、默认行为不变，并同步修订验收指标。**

> **后续（2026-09-27，已实现 + 已发布）**：R1 已实现并在独立发牌 bank2 上复核
> **CONFIRMED**（[`prior-v2-confirmation.md`](./prior-v2-confirmation.md)：transfer v2/v1
> RMSE 53.20/58.30、σ(5/10)=24.61/17.69 vs 27.33/19.60、drift 6.29/9.05），发布为
> `artifacts/human-elo/prior.json`（sha256 `15fddff1…`，对照 `prior_manifest_labels.json`
> sha256 `0f331414…`，语料 `traces/study10`）；v1 4k 版归档
> `runs/archive/prior-v1-4k-20260927/`。

## 0. 结论摘要

1. **漂移的量化口径**：全 RR、LOLO、40k 口径下，逐 (subject level, opponent) cell 的
   OOF 偏差被分解为 `level_bias²=231.7`（占总 MSE 6.8%）、`cell_drift²=95.2`（2.8%）、
   `within_cell²=3084.6`（90.4%）。对手相关的系统性漂移（cell drift）SD ≈ **9.8 Elo**、
   max ≈ 18 Elo（§2.1–2.2）。
2. **漂移来自特征而非 anchor 列**：把 anchor 与交互列从 17 列降到 15 列后漂移不降反
   升（drift SD 9.2→10.5）；精确 OOF 分解给出 feature 8.73 / anchor 8.63 /
   interaction 6.37（部分相消，net 8.23）。贡献最大的特征是 `decisions_per_trick`、
   `tricks_total`、`lead_rate`、`pass_rate`、`trick_win_rate`（§2.3–2.4）。
3. **漂移修不掉的根因是衰减**：全 RR 语料上 raw 等级均值跨度只有 19.6 Elo，而
   within-cell raw 预测噪声 SD 17.2，去收缩 b=3.28 把噪声放大到 56.0；特征本身的
   单局噪声（例：`trick_win_rate` 等级间 SD 0.079 vs within-cell SD 0.208）决定了
   单局估计上限。**相邻档 refit 的 RMSE 26.93 / σ(5)=15.8 不可作为目标**：该子集
   诱导了 level↔对手 anchor 的相关（corr 0.951），只用 anchor 的 LOLO 模型就有
   RMSE 32.75（§2.5–2.6）。
4. **候选实测（全 RR LOLO）**：特征残差化/正交化、anchor 目标改造（offset）、Huber、
   rank 归一化、placement 权重拟合、两级 opponent 校准全部无净收益或破坏 level
   校准；cell 校准惩罚能把漂移 SD 9.2→7.7、`cell_drift²` 95→60（RMSE +0.5）；
   二次特征展开把 RMSE 降到 51.8–52.9；GBM 上限探针 42.6（漂移不变）。没有变体同时
   满足「RMSE≤35 且相邻子集不退化」（§3）。
5. **推荐**：新增可选先验 v2 = **15 维原特征 + anchor + a² + φ×a + φ² + 全部两两乘积
   （152 列）**，拟合时加 **cell 校准惩罚 λ=3**。全 RR LOLO：RMSE **52.87**、m_eff
   **43.3**、σ(5/10/20)=**26.8/20.7/18.0**、drift SD **7.8**、`cell_drift²` **67.0**；
   deal 留出 CV（grouped）50.83（baseline 56.97）。默认 v1 行为与 `predict_elo`
   签名不变，旧 artifact 继续可读（§4）。
6. **验收指标修订**：`全 RR RMSE ≤35` 不应作为验收（该语料 9 级中有 5 级标签只在
   5.5 Elo 内，压缩预测即可刷低 RMSE：b≈0.75 时 RMSE 36.7 但 lvl1 诊断偏差约 −75）。
   应以「per-level OOF 偏差 + 对手不变量（cell drift）+ placement 分布的 σ_traj」
   为准（§5）。

---

## 1. 口径、数据与复现

### 1.1 数据与标签

| 语料 | n | subject levels | 对手设计 | 标签 | 来源 |
|---|---|---|---|---|---|
| `traces/study10/` | 39,960 | 9（lvl1–4 + 顶部簇 5 级） | 全 RR：每级 vs random 888 + 其余 8 档各 444 | 发布 manifest = T2 | `runs/study10/README.md` |
| `traces/study/` | 4,000 | 4（lvl1–4） | 单边上行：每个 subject 只遇更高档 rung + random（除 random 外无更低档对手；本报告 §1.1 统计） | T2 = manifest | `tests/test_fit_trace_prior.py:46`；标签 `prior.py:162-173` |

标签 `prior.T2_LABELS`（`src/seven523/prior.py:162-173`），random=0，lvl1 84.68 /
lvl2 113.60 / lvl3 134.40 / lvl4 187.72，顶部簇 185.96–191.49。40k 语料的逐行特征
缓存为 `/tmp/study10_rows.pkl`（`prior.load_rows` 输出，39,960 行，15 特征 + level/
opponent 字段；生成命令见 §8）。

### 1.2 模型与指标（与发布口径一致）

- 模型：标准化岭回归 `ridge_fit`（α=30，`prior.py:435-455`），设计矩阵
  `[15 features, anchor, twr×anchor]`（17 列，`prior.py:475-494`），去收缩
  `a+b·g(φ)` 在每折的训练等级均值上拟合（`prior.py:497-519`）。
- OOF：`scheme="lolo"`，留一整个 subject level（`prior.py:522-554`）；`predict_elo`
  在线把真实对手 Elo 代入 anchor（`prior.py:727-760`）。
- `σ_traj(n)`：每级抽 n 条 session、`mean(pred)−label` 的经验 SD（`prior.py:556-586`），
  `m_eff=(347/s)²`（`prior.py:82-84`、`prior.py:634-638`）。
- **本报告子集口径**：对同一 full-RR fit 的 OOF 预测取子集残差（不是 refit）；
  `adj1` = 按 10 级 Elo 排序 |Δpos|≤1；`adj1_random` = adj1 ∪ {random}；
  `adj2_random` = |Δpos|≤2 ∪ {random}。README 里的 `26.93/38.54/42.01` 是
  **在子集上重新拟合**的口径，两者不可混比（§2.6）。

### 1.3 基线复现

`runs/study10/research/00_repro.py` 复现 `runs/study10/prior_fullrr_t2.json`
（sha `ec46fd5a…`）：

| 指标 | 本报告复现 | 发布实验产物 |
|---|---|---|
| full RR RMSE | 58.4080 | 58.4082 |
| residual_sd (m_eff) | 58.3735 (35.43) | 58.3737 (35.43) |
| σ(1/5/10/20) | 58.411/29.664/23.007/19.687 | 58.411/29.664/23.007/19.687 |

差异 0.0002 来自 fill 值：发布实现用全量 pooled median，本报告每折只用训练行的
median（更严格、无折内泄露）。下文所有数字均为 LOLO + 折内去收缩，n=39,960，
reps=400：17 列族为每折 fill（v1 复现 58.4080 vs 发布 58.4082）；二次展开族
（`12_cellpen_designs.py`/`09_nonlinear.py`）的展开矩阵一次用全量 fill 构造
（NaN 行 452/39,960，对 RMSE 影响 <0.01）。

---

## 2. 机制：漂移是什么、有多大、来自哪里

### 2.1 误差的方差分解（全 RR LOLO，n=39,960）

`runs/study10/research/mechanism.txt`、`oof_decomp.txt`：

| 分量 | 口径 | 数值 | 占 MSE |
|---|---|---|---|
| `level_bias²` | 每级残差均值² 的行均值 | 231.7 | 6.8% |
| `cell_drift²` | (cell 均值 − 同级 level 均值)² 的行均值 | 95.2 | 2.8% |
| `within_cell²` | cell 内残差方差的行均值 | 3084.6 | 90.4% |
| 合计 MSE | = RMSE² | 3411.5 | 100% |

即：**对手漂移是真实的（SD ≈9.8 Elo），但它只占 RMSE² 的约 3%**；把 cell drift 完全
抹掉、其余不动，RMSE 只会从 58.41 降到约 57.6。

### 2.2 逐 (level, opponent) OOF cell 偏差矩阵（y−ŷ，全 RR fit）

`mechanism.txt` M2，节选（完整 9×10；`--` = 自学对局不存在）：

| subject\opp | random | lvl1 | lvl2 | lvl3 | lvl4 | w2m_ctl |
|---|---|---|---|---|---|---|
| lvl1 | −28.6 | -- | −22.1 | −31.1 | −43.9 | −49.6 |
| lvl2 | +27.0 | +27.6 | -- | +30.0 | +10.3 | +11.8 |
| lvl3 | −9.3 | −3.3 | +11.4 | -- | +11.1 | +13.8 |
| lvl4 | −0.8 | −0.1 | +12.1 | +19.0 | -- | +16.9 |

逐 cell 校准偏差 `y−ŷ`（含 level 偏差）；去掉每级均值后的「cell drift」最大约
±16 Elo（如 lvl1 vs random −11.2、lvl3 vs random +14.9、w2m_plain vs w2m_ctl −16.3）。
任务描述里的「偏置随对手摆动 20–50」主要是 level 偏差 + 漂移的叠加。

### 2.3 特征随对手强度的漂移（M1）

等级内对 anchor 去均值后的 OLS 斜率（`mechanism.txt` M1；anchor 单位 = 300 Elo）：

| 特征 | slope | corr | slopeSD(级间) | 对手引起的水平均值跨度 |
|---|---|---|---|---|
| trick_win_rate | −0.497 | −0.511 | 0.039 | 0.322 |
| trick_point_share | −0.423 | −0.396 | 0.028 | 0.273 |
| lead_rate | −0.373 | −0.428 | 0.020 | 0.248 |
| pass_rate | +0.303 | +0.474 | 0.016 | 0.201 |
| tricks_won | −10.245 | −0.594 | 1.003 | 6.658 |
| trick_points_won | −39.929 | −0.412 | 2.234 | 25.461 |
| decisions_per_trick | +0.082 | +0.101 | 0.046 | 0.247 |
| tricks_total | −4.134 | −0.427 | 1.219 | 3.238 |

级间 slopeSD 相对小，说明对手效应大致是**等级无关的平移**；但 15 个特征同时平移，
线性修正（现有 1 个交互项）无法逐特征抵消，尤其 `decisions_per_trick` /
`tricks_total` / `lead_rate` 的系数大。

### 2.4 漂移的精确 OOF 分解（`oof_decomp.txt`）

对每折模型在 held-out level 的 cell 均值特征上做精确线性分解
（`ŷ_cell−ŷ_level = b·[Σ_j w_j·Δφ̄_j + u·Δa + v·Δ(twr·a)]`）：

| 分量 | SD (Elo) | mean\|·\| | max\|·\| |
|---|---|---|---|
| feature 漂移 | 8.73 | 7.81 | 25.42 |
| anchor 列 | 8.63 | 10.16 | 18.13 |
| 交互列 | 6.37 | 8.59 | 13.61 |
| 三项之和 | 8.23 | 7.69 | 17.81 |
| 实际 OOF 漂移 | 8.23 | 7.69 | 17.81 |

分解误差 ≤5e−12。**anchor 列贡献 8.6，方向与 feature 漂移部分相消**——去掉 anchor
+ 交互的 counterfactual 漂移为 8.73，比带 anchor 的 8.23 还略差。逐特征贡献最大的
是 `decisions_per_trick` 20.3、`tricks_total` 13.8、`lead_rate` 13.2、`pass_rate` 9.8、
`trick_win_rate` 9.1。

### 2.5 为什么线性修正不够：衰减 + 去收缩放大

同一 40k 语料上的对比（`runs/study10/research/04_attenuation.py`）：

| 口径 | raw 等级均值跨度 | raw within-cell SD | b̄ | 校准后 within-cell SD |
|---|---|---|---|---|
| 全 RR fit | 19.6 | 17.2 | 3.28 | 56.0 |
| adj1 refit（有污染，见 §2.6） | 93.4 | 8.4 | 1.26 | 10.6 |

- 全 RR 语料里，特征的主要方差是「对手造成的」，与 y 无关，岭回归因此**衰减**：
  raw 等级跨度只有 19.6，噪声却有 17.2；去收缩 b=3.28 把 17.2 放大到 56.0，
  这就是 `within_cell²=3085` 的来源。
- 逐对手 cell 单独拟合也差（E1）：random cell LOLO RMSE 39.1，lvl1 cell 33.7，
  顶部 5 级所在 cell 64–70。**不是「共享系数」的问题，是单局特征信噪比的问题**：
  例如 `trick_win_rate` 的等级间 SD 0.079 vs within-cell SD 0.208（E3）。
- 因此「输出与对手无关的实力」在本特征集上有一个信息天花板：线性模型下 OLS
  已接近该天花板；GBM 上限探针也只到 RMSE 42.6（§3.3）。

### 2.6 相邻档 refit 指标不可作为目标（重要口径纠正）

`runs/study10/README.md` 的 `adj1 refit RMSE 26.93 / σ(5)=15.8` 有严重的构造性相关：

| 语料 | corr(级均 anchor, label) | 只用 anchor 的 LOLO RMSE |
|---|---|---|
| 全 RR | −1.000（自排除伪相关，跨度仅 13 Elo） | 736.1 |
| adj1 | **+0.951** | **32.75** |
| adj1_random | +0.842 | 143.5 |
| adj2_random | +0.877 | 143.5 |

在 `adj1` 子集里，每个 subject level 只遇到与之相邻的对手，对手 Elo 本身几乎就是
subject level 的代理。**一个不含任何特征的 anchor-only 模型在该子集就能到
RMSE 32.75**。所以 26.93 不是「模型修好后能达到的水平」，它在无诱导相关的新对局
上不成立（§3.5 的跨语料 transfer 也没有确认二次模型的增益）。
---

## 3. 候选修法实测（全 RR LOLO，n=39,960）

### 3.1 主表

口径：full/adj1/adj1_random/adj2_random = 同一 fit 的 OOF 残差子集；`drift SD` =
逐 cell 校准均值相对同 level 校准均值的 SD；`cell/lvl/within var` = §2.1 的
ANOVA 三项；`b̄` = 折内去收缩斜率均值；`raw spread` = raw OOF 的等级均值跨度
（OOF 混合了各折尺度，仅作数量级参考）。完整数据：
`runs/study10/research/main_table.md`。

| 变体 | full RMSE | m_eff | σ1 | σ2 | σ3 | σ5 | σ10 | σ20 | adj1 RMSE | adj1+r RMSE | adj2+r RMSE | drift SD | cell var | lvl var | within var | b̄ | raw spread |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline（17 列） | 58.41 | 35.4 | 58.4 | 43.7 | 35.7 | 29.7 | 23.0 | 19.7 | 60.03 | 56.28 | 57.22 | 9.21 | 95.2 | 231.7 | 3084.6 | 3.28 | 19.6 |
| no_anchor（15 列） | 58.72 | 35.1 | 58.7 | 43.8 | 35.7 | 29.6 | 22.8 | 19.4 | 59.20 | 57.57 | 57.83 | 10.50 | 129.5 | 216.2 | 3101.9 | 3.31 | 19.4 |
| all_interactions（32 列） | 56.56 | 37.8 | 56.7 | 42.2 | 34.6 | 29.0 | 22.7 | 19.6 | 56.08 | 57.04 | 56.60 | 15.33 | 246.4 | 237.6 | 2714.6 | 3.12 | 21.2 |
| resid_linear | 58.97 | 34.8 | 59.0 | 44.0 | 35.8 | 29.7 | 22.9 | 19.6 | 59.18 | 57.32 | 57.71 | 9.43 | 98.7 | 220.3 | 3158.1 | 3.33 | 19.2 |
| resid_cat（逐对手中心） | 59.33 | 34.3 | 59.5 | 44.5 | 36.2 | 30.3 | 23.7 | 20.5 | 59.92 | 57.78 | 58.13 | 10.27 | 115.0 | 255.5 | 3149.4 | 3.35 | 19.4 |
| resid_cat_z（逐对手 z） | 60.01 | 33.5 | 60.1 | 44.9 | 36.7 | 30.7 | 24.2 | 20.9 | 59.56 | 60.45 | 59.79 | 11.29 | 141.4 | 268.8 | 3191.4 | 3.39 | 19.1 |
| resid_linear+anchor | 57.68 | 36.3 | 57.7 | 43.2 | 35.2 | 29.4 | 22.7 | 19.6 | 57.40 | 55.18 | 56.15 | 14.06 | 193.4 | 229.5 | 2903.9 | 3.22 | 20.2 |
| offset target（b） | 108.58 | 10.2 | 108.3 | 77.5 | 65.6 | 49.4 | 38.0 | 28.9 | 116.60 | 112.98 | 111.42 | 13.56 | 211.9 | 289.5 | 11288.2 | 2.56 | 23.0 |
| two-stage（嵌套 OOF，逐对手仿射） | 60.05 | 41.3 | 53.8 | 50.5 | 49.7 | 48.4 | 47.6 | 47.1 | 62.51 | 60.00 | 60.43 | 7.87 | 61.7 | 2873.6 | 671.1 | 0.11 | 25.5 |
| rank 归一化 | 63.89 | 29.6 | 63.5 | 47.8 | 38.8 | 32.8 | 25.9 | 22.2 | 61.81 | 66.65 | 64.82 | 11.87 | 177.9 | 308.6 | 3595.1 | 3.71 | 16.4 |
| Huber δ=40 | 58.62 | 35.2 | 58.8 | 43.9 | 35.8 | 29.8 | 23.0 | 19.7 | 60.61 | 55.90 | 57.11 | 8.90 | 86.5 | 231.3 | 3118.0 | 3.19 | 19.1 |
| cellpen λ=3（17 列） | 58.93 | 34.8 | 58.9 | 44.0 | 35.9 | 29.7 | 22.9 | 19.4 | 61.32 | 55.69 | 57.34 | 7.66 | 59.8 | 219.1 | 3194.2 | 1.61 | 52.1 |
| cellpen λ=10（17 列） | 60.10 | 33.5 | 60.1 | 44.7 | 36.4 | 29.9 | 22.8 | 19.2 | 63.15 | 56.02 | 58.27 | 7.29 | 52.9 | 200.8 | 3358.6 | 1.25 | 71.7 |
| 二次+pairwise | 51.83 | 45.0 | 52.5 | 39.2 | 31.7 | 26.6 | 20.6 | 18.0 | 53.42 | 53.62 | 52.63 | 13.45 | 195.9 | 203.9 | 2286.6 | 2.69 | 26.4 |
| **二次+pairwise+cellpen λ=3** | **52.87** | **43.3** | **53.4** | **40.2** | **32.1** | **26.8** | **20.7** | **18.0** | **56.40** | **51.20** | **52.02** | **7.79** | **67.0** | **198.5** | **2530.0** | **1.47** | **57.9** |
| 二次（无 pairwise）+cellpen λ=3 | 55.02 | 39.9 | 55.4 | 41.4 | 33.4 | 27.6 | 21.3 | 18.2 | 57.20 | 52.41 | 53.80 | 7.61 | 63.8 | 195.9 | 2767.1 | 1.53 | 54.4 |
| all_interactions+cellpen λ=3 | 58.16 | 35.7 | 58.2 | 43.4 | 35.4 | 29.4 | 22.8 | 19.5 | 60.78 | 54.87 | 56.55 | 8.55 | 77.9 | 224.4 | 3080.5 | 1.59 | 52.7 |
| GBM（φ+anchor，31 叶） | 45.49 | 58.7 | 45.7 | 35.0 | 29.5 | 25.3 | 21.2 | 19.4 | 47.17 | 46.34 | 45.91 | 11.04 | 140.0 | 301.3 | 1628.3 | 2.21 | 29.0 |
| GBM（φ+anchor，63 叶，headroom） | 42.59 | 67.1 | 42.8 | 33.5 | 28.6 | 25.0 | 21.6 | 20.1 | 44.80 | 42.52 | 42.51 | 9.67 | 107.2 | 343.5 | 1363.0 | 1.97 | 30.7 |

### 3.2 分类小结

**(a) 特征残差化/正交化（resid_linear/cat/z、rank）——无效。** 在平衡设计下，
逐对手减均值对 OLS 斜率是惰性的：Cov(φ−m(opp), y)=Cov(φ,y)、Var 变小，但等级间
φ̄ 差不变，去收缩后的预测几乎不变（resid_cat 与 baseline 的 raw 跨度 19.4 vs 19.6、
RMSE 59.33 vs 58.41）。逐对手 z 化/rank 化反而破坏等级间形状信息（60.01/63.89）。
这条不推荐。

**(b) 目标改造——不成立。** `y−opp_elo` 作目标（anchor 作 offset，`offset target`）
给出 RMSE 108.58：模型把「相对表现」的系数学到接近 0，去收缩后全局失准。
两级校准（先 15 特征打分、再逐对手仿射校正）能把 drift SD 降到 7.87、mean|drift|
降到 4.0，但把 level 校准打坏（`level_bias²` 232→2874，b̄=0.11）：修正吸收了等级
信号，预测被压到各对手的中心。**任何逐对手的无约束仿射校正都会重演这个失败。**

**(c) 更多交互/非线性——能降 RMSE，但漂移一般变差。** `all_interactions`
（15 φ + a + a² + 15 φa）RMSE 56.56（−1.85），drift SD 15.33（+6.1）；二次
feature 展开 RMSE 51.83（−6.6），drift 13.45。GBM 上限探针 42.59。
它们主要减少 `within_cell²`（3085→2287 / 1363），对 `level_bias²` 与 cell drift
无改善（GBM 的 cell drift 107、level bias² 344）。**非线性不是对手修正，是信噪比修正。**

**(d) 分层/随机效应——level 身份不能在线用。** 直接把 level 当特征会泄露；
可行的替身是「cell 校准惩罚」（在训练等级上约束每个 (level, opponent) 的残差均值，
预测时仍只用行内特征 + 对手 Elo）。17 列 + λ=3 把 drift SD 9.21→7.66、
`cell_drift²` 95→60、placement 带 RMSE 56.28→55.69，代价是 full RMSE +0.5、
σ_traj 基本不变。λ 的完整扫描见 §3.1 与 `06_cellpen.py`；λ≈3 是漂移/精度的拐点。

**(e) 稳健回归——无收益。** Huber δ=20/40（IRLS）RMSE 59.36/58.62，drift 9.4/8.9；
误差不是重尾驱动的，是衰减放大驱动的。

**(f) 语料侧——placement 权重拟合无收益，placement-mix σ 也几乎不变。**
- 用选择器模拟出的 (level, opponent) 频率做 WLS（`placeweight_f0.5`）等价于
  baseline（58.55 vs 58.41）；把 far cell 完全剔除（floor=0）会让设计退化（RMSE 537）。
- 用 10 局 placement session（Fisher + 2 局 Thompson；脚本 `08_placement_sigma.py`）
  重算 σ：
  baseline RR vs placement = 58.4/58.9（n=1）、43.7/43.7（n=2）、35.7/35.7（n=3）、
  29.7/28.6（n=5）、23.0/22.2（n=10）。**σ(5) 的“虚高”不是语料分布造成的**，
  换到 placement 分布几乎不动；它来自单局特征噪声与 level 偏差。

### 3.3 非线性 headroom（对照上限）

`runs/study10/research/09_nonlinear.py`、`07_extra.py`；sklearn 1.9.1（临时装到
`/tmp/research_pkgs`，未改项目依赖）：

| 模型 | full RMSE | σ(1/5/10/20) | drift SD | within var | 说明 |
|---|---|---|---|---|---|
| GBM φ only（31 叶） | 46.32 | 46.5/24.9/20.6/18.6 | 9.7 | 1770.5 | 不用 anchor 也远好于线性 |
| GBM φ+anchor（31 叶） | 45.49 | 45.7/25.3/21.2/19.4 | 11.0 | 1628.3 | anchor 无额外收益 |
| GBM φ+anchor（63 叶） | 42.59 | 42.8/25.0/21.6/20.1 | 9.7 | 1363.0 | headroom 上限 |
| 二次+pairwise（ridge） | 51.83 | 52.5/26.6/20.6/18.0 | 13.5 | 2286.6 | 可在线、可序列化 |
| 二次+pairwise+cellpen3 | 52.87 | 53.4/26.8/20.7/18.0 | 7.8 | 2530.0 | 推荐候选 |

结论：**即使高容量非线性模型也只到 RMSE ≈42.6，达不到 35**；GBM 也不修漂移
（drift SD 9.7 ≈ baseline 9.2）。headroom 主要来自把单局预测方差降下来，而不是
把对手项修好。

### 3.4 去收缩斜率扫描（解释「RMSE≤35」为什么是度规假象）

固定 OOF raw（baseline），只扫全 `b`（截距按 `mean(y)−b·mean(raw)` 重拟合），
n=39,960（`deshrink_sweep.txt`）：

| b | 0.00 | 0.25 | 0.50 | 0.75 | 1.00 | 1.50 | 2.00 | 2.50 | 3.00 | 3.23（现值） |
|---|---|---|---|---|---|---|---|---|---|---|
| RMSE | 38.59 | 37.40 | 36.78 | **36.74** | 37.30 | 40.09 | 44.70 | 50.65 | 57.53 | 60.95 |
| σ(5) | 38.6 | 37.2 | 36.0 | 34.9 | 34.0 | 32.7 | 32.3 | 32.7 | 34.0 | 34.8 |
| σ(20) | 38.6 | 37.2 | 35.8 | 34.5 | 33.3 | 31.0 | 29.1 | 27.6 | 26.7 | 26.4 |

RMSE 在 b≈0.75 触底到 36.7，但此时 lvl1 的偏差约 −75（诊断口径）、lvl2 −39，顶部
簇 +19…+25——因为 9 级里 5 级标签挤在 186–191。「全 RR RMSE≤35」可以用压预测
刷出来，不是可用的模型。此扫描用同一 OOF raw + 全局截距，是度规探针，不是发布
管线（管线为折内去收缩，baseline RMSE 58.41）。反方向看，session 最优的 b 随 n
变化（n=1→0.6、n=5→2.3、n=10→2.8、n=20→3.5，`deshrink_sweep.txt`）：现在
b=3.23 对 n≥10 接近最优，对 n=1 明显过大（σ1 61 vs 最优 36），这是「一个全局 b
服务所有 session 长度」的固有折中，不是语料分布问题。

### 3.5 语料、deal 留出与跨语料复核

**(i) 语料对照（baseline 17 列，在各语料内部做 LOLO）**：

| 语料 | n | RMSE | σ(5/10/20) | drift SD | level² | cell² | within² | b̄ |
|---|---|---|---|---|---|---|---|---|
| 全 RR（9 subject × 9 opponent） | 39,960 | 58.41 | 29.7/23.0/19.7 | 9.2 | 231.7 | 95.2 | 3084.6 | 3.28 |
| 9 subject × (random+lvl1–4) | 22,200 | 49.32 | 25.1/20.4/17.3 | 7.0 | 167.8 | 44.7 | 2220.4 | 2.72 |
| 4 subject（lvl1–4）× 全 9 对手 | 17,760 | 66.10 | 41.0/35.9/34.5 | 13.3 | 1079.1 | 205.4 | 3084.9 | 4.32 |
| 4 subject × (random+lvl1–4) | 8,880 | 56.23 | 35.2/32.0/29.6 | 11.1 | 807.6 | 118.6 | 2236.0 | 3.36 |
| **shipped 4k prior 直接在 study10 全 RR 上（不 refit）** | 39,960 | 62.02 | 30.4/28.1/26.5 | -- | 2307.0 | 814.7 | 724.3 | 2.38 |

现发布先验（`artifacts/human-elo/prior.json`，4k 单边上行语料训练）在全 RR 上
`level_bias²` 达 2307（占 MSE 60%）、`cell_drift²` 815：它在顶部簇 cell 上系统性
失准（逐级偏差 lvl1 −9.1 / lvl2 +11.5 / lvl3 +19.3 / 顶部 5 级 +55…+61；逐级 RMSE
30.8–32.9 vs 69.7–74.5）——4k 语料里最强 subject（lvl4）只遇过 random，没学过
「强 subject 对强对手」的特征分布。40k 全 RR refit 把
`level_bias²`/`cell_drift²` 压到 232/95，是「换语料」带来的主要收益；把 subject
限制到 4 级（`rr_rungs4`）反而更差（level² 1079），因为对手里出现了 5 个强档、而
subject 标签只有 4 个，去收缩 b 被推到 4.32。

**(ii) deal 留出（10 折 by seed，同 level 集）**：

| 模型 | grouped RMSE | σ(1/3/5/10/20) |
|---|---|---|
| baseline 17 列 | 56.97 | 57.0/33.4/26.7/19.2/14.9 |
| 17 列 + cellpen λ=3 | 57.74 | 57.7/33.8/26.9/19.2/14.8 |
| 二次+pairwise（ridge30） | **49.20** | 49.6/28.7/23.1/16.5/13.0 |
| 二次+pairwise+cellpen λ=3 | 50.83 | 51.2/29.4/23.4/16.7/12.8 |

二次展开在「新牌、旧等级」轴上也稳定领先（−6.1～−7.8 RMSE）；cellpen 以约
+1.6 RMSE 换取漂移。

**(iii) 跨语料 transfer（study10 全量拟合 → 直接预测 `traces/study` 4k，不 refit）**：

| 模型 | RMSE | 均值偏差 | lvl1/lvl2/lvl3/lvl4 偏差 |
|---|---|---|---|
| baseline 17 列 | 55.7 | +0.8 | −9.7 / +15.3 / +2.3 / −3.3 |
| 二次+pairwise+cellpen λ=3 | 56.6 | +1.5 | −6.3 / +10.5 / +1.3 / +5.7 |
| 二次+pairwise（ridge30） | 57.1 | +1.9 | −9.1 / +13.6 / +6.0 / +2.4 |

**未确认 R1 的增益**：4k 是单边上行设计（subject 只遇更高档 rung 与 random），与
study10 的全 RR 不同分布，三个模型都在 55.7–57.1；这提醒 40k LOLO 的优势必须在
新的全 RR deal bank 上复核（§6.2）。

---

## 4. 推荐方案与实施计划

### 4.1 推荐（R1）：v2 先验 = 二次展开 + cell 校准惩罚

**设计**：`[φ(15), a, a², φ×a(15), φ²(15), φ_iφ_j(105)]`，共 **152 列**；拟合目标

```
minimize  Σ_rows (y − Xw)² + α‖w_std‖² + λ·Σ_cells n_c·(mean residual in cell)²
```

cell = (subject level, opponent id)，只在**拟合时**用（训练标签本来就是 level），
预测时 `predict_elo` 仍只用该行的特征 + 对手 Elo，不含 level 身份。推荐
**λ=3**（漂移优先）/ λ=0（精度优先，见下）；α 仍为 30。

**为什么是它**：在所有实测变体里，它是唯一同时改善以下全部口径的：

| 口径 | baseline | R1（二次+cellpen3） | 变化 |
|---|---|---|---|
| full RR RMSE | 58.41 | 52.87 | −5.5 |
| m_eff | 35.4 | 43.3 | +7.9 |
| σ(5/10/20) | 29.7/23.0/19.7 | 26.8/20.7/18.0 | −2.9/−2.3/−1.7 |
| adj1 / adj1_random / adj2_random RMSE | 60.03 / 56.28 / 57.22 | 56.40 / 51.20 / 52.02 | −3.6 / −5.1 / −5.2 |
| drift SD | 9.21 | 7.79 | −1.4 |
| `cell_drift²` | 95.2 | 67.0 | −28 |
| `level_bias²` | 231.7 | 198.5 | −33 |
| deal 留出 CV（grouped）RMSE | 56.97 | 50.83 | −6.1 |
| placement-mix σ(5/10)（placement 口径） | 28.6/22.2 | 25.7/19.6 | −2.9/−2.6 |

（λ=0 的二次模型 full RMSE 51.83、grouped 49.20，更好，但 drift SD 13.45。若验收
只认精度，用 λ=0；若认「对手不变量」，用 λ=3。cellpen 的代价在 grouped CV 上约
+1.6 RMSE。）

**逐级检查**（`13_verify.py`）：R1 在顶部 5 级把 RMSE 从 56–59 降到 47–49，
lvl3 55.1、lvl1 59.6（略差于 baseline 58.8）、lvl2 67.6（差于 63.6）。lvl1/lvl2
仍是弱项（bias −37.6/+13.6）——这是 40k 全 RR 语料的最弱两级面对的对手普遍更强，
属于信息限制，不是对手项没修好。

**备选**：
- **R0（零 schema 变更）**：保持 17 列，只把 λ=3 的 cell 惩罚加进 `ridge_fit` 的
  调用。drift SD 9.21→7.66、`cell_drift²` 95→60、adj1_random 56.28→55.69，
  full RMSE +0.5、σ 不变。适合「不想动 `predict_elo`/schema」的场景，但收益很小。
- **R2（精度优先、新依赖）**：GBM/MLP 先验（新 `kind`，需要 sklearn/导出模型，
  或 torch MLP 离线训练后固化）。full RMSE 42.59、σ(5/10/20)=25.0/21.6/20.1，
  但 drift SD 9.67 与 baseline 持平，且不适合现在的「纯 numpy artifact」形态。

### 4.2 schema / 测试影响

- `tests/test_fit_trace_prior.py:227-228` pin 了
  `len(model["features"])==15` 与 `len(model["mean"])==len(model["scale"])==len(model["coefficients"])==17`，
  以及 `:229` 的 `model["anchor"]` 块。**默认仍走 v1（线性）路径**，这些断言不变；
  v2 加一个同构的 round-trip 测试（系数 152、`model["expansion"]`、`cell_penalty`）。
- `src/seven523/prior.py`：
  - `MODEL_FEATURES`（:112）不变；新增 `EXPANSION_LINEAR="linear"` /
    `EXPANSION_QUADRATIC_PAIRWISE="quadratic_pairwise"`。
  - `design_matrix(rows, fill, *, expansion="linear")`（:475）按 expansion 生成列；
    `predict_elo`（:727）读 `model.get("expansion", "linear")`，v1 artifact 无该字段时
    保持现行为。
  - 新增 `cellpen_ridge_fit(x, y, alpha, cell_ids, lam)`（或给 `ridge_fit` 加可选
    `cells/cell_lam`）；`_cv_predictions`（:522）与 `build_prior`（:589）透传
    cell 信息；`build_prior` 新增 `expansion=..., cell_penalty=...`；
    artifact 增加 `kind="trace-s1-poly2-ridge-deshrunk"`、`version=2`、
    `model.expansion`、`hyperparams.cell_penalty`。
  - `session_mean` / `prior_for_session` / `TracePrior` 签名不变。
- `tools/fit_trace_prior.py`：`fit` 增加 `--expansion {linear,quadratic_pairwise}`、
  `--cell-penalty FLOAT`；默认值保持现状，发布命令需显式开启。
- 兼容：`load_prior` 只校验 `schema`（:817-823），旧 artifact 继续可读；
  `TracePrior.predict_elo` 走同一个 `predict_elo`，无感。迁移只需重新生成
  `artifacts/human-elo/prior.json`（并保留旧 sha 记录）。

### 4.3 分步实施

1. 在 `prior.py` 抽出 `design_matrix(..., expansion=)` 与 `cellpen_ridge_fit`
   （纯 numpy，不引依赖）；v1 路径逐位不变（用 `00_repro.py` 回归 58.4080 验证）。
2. `build_prior` 接 `expansion`/`cell_penalty`；`_cv_predictions` 把 fold 训练行的
   cell id 传给 fit。补单测：v1 数值不变、v2 形状/缩放/去收缩可往返、cell 惩罚
   在 λ→0 时退化为 ridge_fit。
3. `tools/fit_trace_prior.py` 加参数与打印（把 `drift_sd`/`cell_drift²` 打进
   `calibration`，便于验收）。
4. 在 `traces/study10` 上跑 `--expansion quadratic_pairwise --cell-penalty 3`，
   核对 §4.1 的表；对 `traces/study` 与 `traces/study10` 双跑，记录 sha。
5. 用 placement 选择器模拟（§3.2f）复算 `sigma_traj` 与逐级偏差，作为发布前的
   第二道验收；确认无 `NaN`、`sigma_traj` 单调不降。
6. 发布决策：默认仍发布 v1，除非验收通过；v2 作为新 `kind` 并行存在，placement
   按 `kind` 自动兼容。

### 4.4 明确不做的事

- 不把 `adj1`/`adj1_random` refit 数字当验收（§2.6）。
- 不逐对手做无约束仿射/两级校正（§3.2b）。
- 不在线使用 subject level 身份；不把 `level_id` 写进任何特征或公式。
- 不为降 RMSE 压缩去收缩斜率（§3.4）。

---

## 5. 验收指标（修订建议）

现有「示例目标」是 全 RR RMSE≤35、相邻子集不退化、σ(5) 不虚高。实测结果：

- **RMSE≤35 不可达**：GBM headroom 42.59；且该指标可被压预测刷低（§3.4）。
- **相邻子集「不退化」**：R1 在 adj1/adj1_random/adj2_random 全部优于 baseline
  （56.40/51.20/52.02 vs 60.03/56.28/57.22），这条可以保留，但要在**同一 fit 的
  子集残差口径**下报（不是子集 refit）。
- **σ(5) 不虚高**：placement-mix σ(5) 与 RR σ(5) 基本相同（28.6 vs 29.7），
  「虚高」的 15.8 来自受污染的 refit；换成 placement 分布并不会自动降低 σ。
  真正能降的是 R1：placement σ(5) 25.7、σ(10) 19.6。

建议验收口径（全在 40k、LOLO、reps=400、每折 fill）：

| 指标 | 门槛 | R1 实测 |
|---|---|---|
| full RR RMSE | ≤ 55 | 52.87 |
| σ_traj(5 / 10 / 20) | ≤ 27 / 21 / 18.5 | 26.8 / 20.7 / 18.0 |
| `cell_drift²`（对手不变量） | ≤ 70 | 67.0 |
| 逐级 \|bias\| | 不劣于 baseline 的 1.5×，且 lvl1 不劣化 >3 Elo | lvl1 −37.6（baseline −39.9） |
| adj1 / adj1_random RMSE（子集残差） | 不劣于 baseline | 56.40 / 51.20 |
| deal 留出 grouped RMSE | 不劣于 baseline | 50.83（baseline 56.97） |
| placement-mix σ(5) | ≤ 27 | 25.7 |

---

## 6. 局限性与需独立复核的项

1. **单一 deal bank**：study10 的 888 副牌是 pass1/pass2 的 deal-twin（`runs/study10/
   README.md`），所有 LOLO 数字都在同一发牌池上；cell 均值的有效独立样本约 222 副/
   cell（40k 结构：每 cell 444 行 = 222 副 × 2），cell drift 的统计噪声约 3–4 Elo，
   9.8 的信号是真实的，但绝对值应在独立发牌池上复核。
2. **跨语料 transfer 未确认 R1 的增益**：把 study10 训好的模型直接用于
   `traces/study`（4k、单边设计）时，baseline 55.7、R1（二次+cellpen3）56.6、
   纯二次 57.1——没有复现 40k LOLO 的优势。差异来自两语料的对局设计（单边 vs 全
   RR）与等级集。**发布前需要一个新的全 RR deal bank 复核（建议新 seed、九级 ×
   相邻为主、含少量远档）**；在那之前 R1 只是「study10 内部口径的最优候选」。
3. **in-sample vs OOF**：§2.4 的 M3 分解是 full-data 模型的 in-sample 精确分解；
   §2.4 的 OOF 分解用每折模型；两者都只说明结构，不构成独立验证。
4. **placement 模拟是近似**：selector 的 posterior 用真值居中、self-cell 被排除
   （语料没有 L vs L 的自学对局），游戏结果/似然更新被跳过；真实 σ_traj 需要接
   `placement/session.py` 的完整回路复核（`src/seven523/placement/session.py:260-300`）。
5. **人类 vs bot**：标签与特征都来自 bot 对局；真人 trace 的域偏移（风格、节奏）
   未评估，`trick_win_rate` 等特征的可迁移性不明。
6. **标签噪声**：T2 是 probit-MLE 发布表（σ≈6 Elo），对 RMSE 的贡献约 6/58，
   可忽略；但顶部 5 级间距 1.7–5.5 Elo < 标签 SE 的量级，5 级是否应合并成 1–2 个
   校准点值得在下一轮语料设计中讨论。
7. **GBM headroom 未做 grouped/跨语料**，仅作量级参考。

---

## 7. 与任务假设的差异（诚实边界）

| 任务描述 | 实测 |
|---|---|
| 「预测把对手难度泄漏进预测，偏置随对手摆动 ~20–50」 | cell drift SD 9.8、max ~18；给定的 20–50 摆动含 level 偏差（§2.1–2.2） |
| 「加 anchor²/anchor³ 几乎无改善 ⇒ 不是光滑尺度问题，而是跨档外推」 | 确认：加全部 φ×a、φ²、pairwise 后漂移仍大（13.5）；但主因是特征噪声衰减，不只是外推 |
| 「相邻档子集 RMSE 26.93 / σ(5)=15.8」 | 该口径诱导 anchor↔level 相关（anchor-only RMSE 32.75），不能作目标（§2.6） |
| 「全 RR RMSE 降到 ~35」 | 不可达；且是度规假象（压预测 b≈0.75 即 36.7，lvl1 偏差约 −75） |

---

## 8. 复现命令与产物

```bash
cd /home/amas/.local/src/7g523
# 0) 逐行特征缓存（若 /tmp/study10_rows.pkl 丢失）
.venv/bin/python - <<'PY'
from seven523 import prior
rows, stats = prior.load_rows("traces/study10", prior.T2_LABELS, verify=False)
import pickle; pickle.dump(rows, open("/tmp/study10_rows.pkl", "wb"))
PY

# 1) 基线复现（对 runs/study10/prior_fullrr_t2.json）
.venv/bin/python runs/study10/research/00_repro.py

# 2) 机制：特征漂移 + cell 偏差矩阵 + in-sample 分解
.venv/bin/python runs/study10/research/01_mechanism.py > runs/study10/research/mechanism.txt

# 3) 精确 OOF 漂移分解（feature / anchor / interaction）
.venv/bin/python runs/study10/research/02_oof_decomp.py > runs/study10/research/oof_decomp.txt

# 4) 候选主表（逐对手 cell 拟合、语料效应）
.venv/bin/python runs/study10/research/04_attenuation.py

# 5) 残差化 / 两级 / 二次 / GBM / cellpen / rank / 加权
.venv/bin/python runs/study10/research/03_variants.py
.venv/bin/python runs/study10/research/05_twostage.py
.venv/bin/python runs/study10/research/06_cellpen.py
.venv/bin/python runs/study10/research/07_extra.py        # 需先 sys.path.append('/tmp/research_pkgs')
.venv/bin/python runs/study10/research/08_placement_sigma.py
.venv/bin/python runs/study10/research/09_nonlinear.py    # sklearn 同 07
.venv/bin/python runs/study10/research/10_rank.py
.venv/bin/python runs/study10/research/11_weighted.py
.venv/bin/python runs/study10/research/12_cellpen_designs.py
.venv/bin/python runs/study10/research/13_verify.py

# 6) 主表汇总
.venv/bin/python runs/study10/research/99_summary.py | tee runs/study10/research/main_table.md
```

- 数据：`traces/study10/`（不可变）、`traces/study/`；产物：`runs/study10/research/*.json`、
  `*.pkl`、`mechanism.txt`、`oof_decomp.txt`、`deshrink_sweep.txt`、`main_table.md`。
- 本报告未写 `src/`/`tests/`/`artifacts/`；sklearn 装在 `/tmp/research_pkgs`
  （`uv pip install --target`），未改 `pyproject.toml`/`uv.lock`。

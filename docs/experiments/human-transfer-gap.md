# 真人转移缺口调查：搜索的 +110 Elo 为什么没有在 web 28 局里体现

> **状态：调查完成（2026-09-28，子代理研究，未改 `src/`、未 commit、未跑训练）。**
> **对抗审查（2026-09-28）**：C1–C12 各由 1 个独立子代理复核（≥12 进程 / ≤4 GPU）；
> C10 方向被推翻、C12 应删除、C2/C5/C7/C8/C9/C11 需按修订口径收窄，见
> [`human-transfer-gap-adversarial.md`](./human-transfer-gap-adversarial.md)。
> **口径**：revision-3（`rules_id=2e36dbea44893696`）、obs v5、2 家、champion `w2m_ctl`
> （`runs/w2m_ctl__11__1790516900/agent.pt`，manifest μ=191.49）。搜索线口径见
> `runs/o4lite-search/README.md`、[`joint-search-training-wave1.md`](./joint-search-training-wave1.md)（下称 **JS**）、
> [`search-config-plan.md`](../search-config-plan.md)。
> **本文所有数字均由本文作者从原始文件重算**，脚本放 `/tmp/humangap/`（未入库）；复现算法见 §4。
> 关键断言一律给 `file:line` 或 trace 文件名+字段。跨 fit/跨尺度不可比（ADR-0013）。

---

## TL;DR

1. **搜索在 28 局里确实生效，而且部署与实验配置逐 bit 可复现（高置信）。**
   23 局 `opponent_search={"trunc_ply":5,"rollout_k":32}` 全部是同一配置；离线用
   `critic_B.pt` + K=32 + C=6 + rollout 对手=champion 重放 22 个 vs `w2m_ctl` 的搜索局，
   bot 的 **479/479** 个动作与轨迹完全一致，0 fallback（§0.2）。
   所以"搜索没开/开了别的配置"不是原因。

2. **"真人没打出相应水平"这个前提本身没有被数据确立：唯一同期定级是轨迹先验主导的 μ=169，
   同一批 7 局的纯结果拟合约 μ=294±83（高置信）。**
   `traces/sessions/session-20260928-092912/report.json`（09:29 定级，7 局）production 合并
   μ=169.0 [121.3,216.8]，但 trace 通道权重 0.92、μ_traj=160；结果通道 σ=83.2。
   我按同一估计器/同一先验结构重拟：结果-only 先验下 μ=**294.0** σ=83.2，冷启动先验下 μ=337.6。
   294 ≈ 191（raw）+103，与搜索线自称的 +111 量级接近。**"搜索 μ≈300、真人却赢"的矛盾只对
   production 合并值成立**；换成纯结果口径，人机约在同一个量级。

3. **+110 与 manifest μ 不可相加，search 从未进过 MLE 表（高置信）。**
   +111.2/+124.7 是 `400*log10(p/(1-p))` 的 h2h 相对分（`src/seven523/duel.py:30,71-74`），
   manifest 是 probit-MLE 绝对表；JS §9 与 `search-config-plan.md:20-22,85` 明确禁止相加，
   `search-config-plan.md:29` 的"μ≈300"是规划外推而非测量。搜索 rung 至今未测。

4. **测量的对手 = rollout 对手模型 = 同一个 frozen champion；对真人 rollout 模型有系统性误差（中高置信）。**
   评测右对手与 rollout 对手都是 raw `w2m_ctl`（`runs/o4lite-search/report.md:74-78`、
   `trunc_preregistration.md:6-7`）；真人动作与该模型 top-1 一致率只有 **44.5%**（top-6 覆盖 92.5%，
   22 个 w2m_ctl 搜索局共 481 个真人决策）。
   把 rollout 对手换成 `t18poolself`，同一批真人局面的搜索选择有 **13.2%** 改变
   （searched 决策的 17.0%）——说明结果对"对手是不是这个模型"敏感，而现有迁移臂只测过同族神经网络。

5. **同一配置在不同 deal bank 上差几十 Elo，而截断线没有 fresh-bank 复核（中高置信）。**
   全量 K=8 vs raw 在三个 bank 分别是 **+52.68 [39.61,65.75]（seed 0-2）、+28.52 [21.61,35.43]
   （10-14）、+42.64 [30.10,55.17]（30-32）**；screen 与 confirm 的 CI 不重叠（+42.64 与 confirm 有少量重叠）。
   截断 t=5/K16/K32 的全部结果只在 seed 30-49 上测过，没有独立 bank 的 confirm。

6. **非传递性无定论：截断 t=5−全量 = +67.9（直接）vs +34.8（经 raw 相减）（中置信）。**
   JS §12:167-170 自己把这列为开放问题；K32−K16 只有 +13.59 [7.22,20.20]。
   在这个误差量级下，"+124.7 相对 raw"不能直接外推成"对第三方的绝对强度"。

7. **t=5 的 value 截断读出噪声很大，并且有"覆盖确定撬底"的具体案例（中置信）。**
   479 个真人局面决策上，value 读出与单世界 champion 续行（全量 rollout）的
   **MAE≈33.7 分、EV=0.081**；同口径的自对局对照是 MAE≈35.4、EV=0.092。
   具体案例 `g0000__s3548246500__...` step 37：raw 选对子 6-6 = 立即撬底（终局 margin=0，32/32 世界一致），
   搜索改成单牌 ♣6（value=+0.0816），牺牲了保证不输的终局。该局最终仍 50-50，
   但它证明截断读出会在终局压过确定动作。

8. **搜索在真人局面比在基准里改判更多、且改判重尾（中置信）。**
   真人搜索局：searched 370/479=77.2%，改判 229/479=47.8%（searched 中 61.9%），
   基准 K32 是 searched 70.6%、searched 中 49.0%（`trunc_k32/combined.json` mechanism）。
   在 champion-vs-champion 单世界口径下，chosen 比 raw 好 24.0%、差 13.8%（最差 −110 分）、
   62.2% 平；净均值 +7.2 分。改判是净正但重尾。

9. **determinization 先验完全不利用对手历史（低-中置信）。**
   `sample_hidden` 对 unseen 池做**均匀**划分，只强制最新 `revealed` 归对手
   （`runs/o4lite-search/rollout_policy.py:214-260`、README.md:16）。人类历次 pass/领出
   携带的信息没有进入采样。这个缺陷在基准里同样存在，但没有在"人类信息更丰富"的分布上验证过。

10. **两个被数据削弱的假设（低置信，倾向否定）：value 在真人状态更 OOD 退化；top-6 限制是主因。**
    真人局面 EV=0.081 vs 同口径 in-distribution EV=0.092（无差异）；
    人类动作落在 champion top-6 内 92.5%；in-distribution 的 C=6→C=10 配对只有
    +5.14 [−6.46,+17.02]（`runs/o4lite-search/scale/paired_c10k8_vs_k8.json`）。

**总结论**：现有 28 局**不能**证明搜索对真人没有 +110 量级的提升，也**不能**证明有；
它唯一确定的是搜索开着、配置和评测一致、真人以 15W2D6L（均分 +31.3）赢多输少。
把"人机缺口"归因时，测量层（真人评分不可识别、跨尺度、对手=模型、bank 敏感、
未配对小样本）比"搜索算子失效"有更强的一手证据支持。

---

## 0. 事实基础

### 0.1 数据清单

- 28 局真人 web 轨迹：`traces/web/run-20260927-153209/`（3 局，09-27 15:34–15:38）、
  `run-20260928-090736/`（2 局，09-28 09:14–09:16）、`run-20260928-091716/`（23 局，09-28 09:19–10:05）。
- 全部 `version=3`、`rules.revision=3`、2 家、`hand_size=7`（与实验 rules_id `2e36dbea44893696` 的同口径规则场）；`final_scores` 为 `[seat0,seat1]`。
- 23 局顶层 `opponent_search={"trunc_ply":5,"rollout_k":32}`，5 局无该字段（null）。
- 无搜索 5 局：2 局 vs `random`、2 局 vs `w2m_plain`、1 局 vs `w2m_ctl`（原始 raw）。
  搜索 23 局：22 局 vs `w2m_ctl`、1 局 vs `random`（`g0000__s2964174766...`）。
- 全部 23 个搜索局在同一天 46 分钟内连续完成，中途没有改过 (t,K)（唯一配置）。

### 0.2 部署核实：搜索确实以实验配置运行

把每局 trace 用 `state_from_snapshot` 重放，在 bot 决策点用与 `web_search.py` 相同的构造：
`base=policy_from_spec(manifest spec)`、`ValueChampion(critic_B.pt, trunc_ply=5)`、
`O4LiteWrapper(K=32,C=6,rollout_opponent=champion,seed=policy_seed(game.seed, human_seat))`
（seed 约定见 `src/seven523/web/table.py:354-356`：policy seed 由**人类座位**派生；
一开始用 bot 座位派生会错 101，重放匹配率从 ~70% 变 100%，这也反证了该约定）。

结果（22 个 vs `w2m_ctl` 的搜索局，479 个 bot 决策）：

| 指标 | 值 |
|---|---|
| recorded action == 重放 wrapped action | **479/479 = 100%** |
| fallback | 0 |
| recorded action == raw argmax | 250/479 = 52.2% |
| search 改判（wrapped != raw） | 229/479 = 47.8% |
| 可搜索决策（>1 合法模板） | 370/479 = 77.2% |
| 改判 / 可搜索 | 229/370 = 61.9% |

对照：同一重放对无搜索局的 raw 策略是 22/22 完全一致（`g0000__s3407565805...`），
所以 100% 不是巧合。**部署 = 实验配置，且搜索真的在改判。**

### 0.3 28 局事实表（按 `created_at` 排序）

| # | 时间 | 对手 | search | human_seat | starter | 比分 [s0,s1] | 人类分 | bot分 | 人类净胜 | W/D/L | 撬底者 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 09-27T15:34 | random | 无 | 0 | 0 | [60, 40] | 60 | 40 | +20 | 胜 | 人 |
| 1 | 09-27T15:36 | w2m_plain | 无 | 0 | 0 | [55, 45] | 55 | 45 | +10 | 胜 | bot |
| 2 | 09-27T15:38 | w2m_plain | 无 | 0 | 0 | [100, 0] | 100 | 0 | +100 | 胜 | 人 |
| 3 | 09-28T09:14 | random | 无 | 0 | 1 | [100, 0] | 100 | 0 | +100 | 胜 | 人 |
| 4 | 09-28T09:16 | w2m_ctl | 无 | 0 | 1 | [55, 45] | 55 | 45 | +10 | 胜 | 人 |
| 5 | 09-28T09:19 | w2m_ctl | t5·K32 | 0 | 0 | [85, 15] | 85 | 15 | +70 | 胜 | bot |
| 6 | 09-28T09:22 | w2m_ctl | t5·K32 | 0 | 0 | [85, 15] | 85 | 15 | +70 | 胜 | 人 |
| 7 | 09-28T09:24 | w2m_ctl | t5·K32 | 1 | 1 | [65, 35] | 35 | 65 | -30 | 负 | bot |
| 8 | 09-28T09:26 | w2m_ctl | t5·K32 | 1 | 1 | [0, 100] | 100 | 0 | +100 | 胜 | 人 |
| 9 | 09-28T09:29 | random | t5·K32 | 1 | 1 | [10, 90] | 90 | 10 | +80 | 胜 | 人 |
| 10 | 09-28T09:41 | w2m_ctl | t5·K32 | 1 | 0 | [45, 55] | 55 | 45 | +10 | 胜 | bot |
| 11 | 09-28T09:43 | w2m_ctl | t5·K32 | 1 | 1 | [55, 45] | 45 | 55 | -10 | 负 | bot |
| 12 | 09-28T09:44 | w2m_ctl | t5·K32 | 1 | 0 | [65, 35] | 35 | 65 | -30 | 负 | bot |
| 13 | 09-28T09:46 | w2m_ctl | t5·K32 | 1 | 1 | [35, 65] | 65 | 35 | +30 | 胜 | bot |
| 14 | 09-28T09:47 | w2m_ctl | t5·K32 | 1 | 0 | [70, 30] | 30 | 70 | -40 | 负 | bot |
| 15 | 09-28T09:48 | w2m_ctl | t5·K32 | 1 | 0 | [50, 50] | 50 | 50 | +0 | 平 | 人 |
| 16 | 09-28T09:49 | w2m_ctl | t5·K32 | 1 | 1 | [30, 70] | 70 | 30 | +40 | 胜 | 人 |
| 17 | 09-28T09:50 | w2m_ctl | t5·K32 | 1 | 1 | [60, 40] | 40 | 60 | -20 | 负 | bot |
| 18 | 09-28T09:51 | w2m_ctl | t5·K32 | 1 | 1 | [40, 60] | 60 | 40 | +20 | 胜 | 人 |
| 19 | 09-28T09:53 | w2m_ctl | t5·K32 | 1 | 0 | [15, 85] | 85 | 15 | +70 | 胜 | bot |
| 20 | 09-28T09:54 | w2m_ctl | t5·K32 | 1 | 1 | [5, 95] | 95 | 5 | +90 | 胜 | 人 |
| 21 | 09-28T09:56 | w2m_ctl | t5·K32 | 1 | 0 | [35, 65] | 65 | 35 | +30 | 胜 | 人 |
| 22 | 09-28T09:58 | w2m_ctl | t5·K32 | 1 | 0 | [55, 45] | 45 | 55 | -10 | 负 | bot |
| 23 | 09-28T10:00 | w2m_ctl | t5·K32 | 1 | 1 | [5, 95] | 95 | 5 | +90 | 胜 | 人 |
| 24 | 09-28T10:01 | w2m_ctl | t5·K32 | 1 | 1 | [35, 65] | 65 | 35 | +30 | 胜 | 人 |
| 25 | 09-28T10:03 | w2m_ctl | t5·K32 | 1 | 0 | [25, 75] | 75 | 25 | +50 | 胜 | 人 |
| 26 | 09-28T10:04 | w2m_ctl | t5·K32 | 1 | 0 | [10, 90] | 90 | 10 | +80 | 胜 | 人 |
| 27 | 09-28T10:05 | w2m_ctl | t5·K32 | 1 | 1 | [50, 50] | 50 | 50 | +0 | 平 | bot |

**统计**（胜负口径按 `CONTEXT.md:79-81`：比分高者胜，平为真平局）：

- 搜索 23 局：**15胜 2平 6负**；score-convention winrate=(15+1)/23=**0.6957**；
  等价 h2h ΔElo(human − search)=`400*log10(0.6957/0.3043)`=**+143.6**。
- 人类净胜分：mean **+31.3**，sd 43.9，se 9.2（n=23）。
- Wilson 95% CI（严格胜率 15/23=0.652）：**[0.449, 0.812]**。
- 若真人≈raw 水平（搜索对 raw 胜率 0.672 ⇒ 真人胜率应≈0.328），
  观察到 ≥15 胜的二项概率 **p=0.00145**。
- 人类在搜索局收集到 1510/2300 分（65.7%），bot 790 分（34.3%）。
- 人类撬底 12 局（11胜1平0负，撬底时均分 78.3 vs 21.7）；bot 撬底 11 局
  （其中 bot 只赢 6、输 4、平 1，撬底时均分 48.2 vs 51.8）。
- bot pass 率 172/498=34.5%，人类 113/501=22.6%；bot 搜索局 5 个炸弹，人类 14 个（13 小 1 大）。

### 0.4 实验侧口径与数字（重算自原始 JSON）

| 端点 | 值 | 来源 |
|---|---|---|
| confirm 全量 K=8 vs raw | +28.52 [21.61,35.43]（z），winrate 0.5409，均分 +4.511，5 seeds×845 deal | `runs/o4lite-search/confirm/combined.json`；`report.md:42-54` |
| t=5 K=16 vs raw | **+111.16** [105.13,117.20]，winrate 0.6545，均分 +17.78，20 seeds×400 deal | `runs/o4lite-search/trunc_k16/combined.json` |
| t=5 K=32 vs raw | **+124.69** [119.39,130.00]，winrate 0.6720，均分 +20.36，20 seeds×400 deal | `runs/o4lite-search/trunc_k32/combined.json` |
| K16−K8（paired） | +25.01 [18.40,31.42] | `runs/trunc_k_axis/paired_k16_vs_k8.json` |
| K32−K16（paired） | +13.59 [7.22,20.20] | `runs/trunc_k_axis/paired_k32_vs_k16_20seed.json` |
| 同 bank t5 vs 全量（直接） | +67.94 [52.88,83.01] | `runs/o4lite-search/trunc_preregistration.md` 结果节 |
| 同 bank t5 减全量（经 raw 相减） | +34.8 | JS:129 |
| t 前沿（K=8，bank 30-32） | t0 −183.5 / t1 −40.8 / t2 +7.0 / t3 +65.4 / t5 +77.5 / t7 +81.7 / t10 +56.8 / 全量 +42.6 | JS:39-44 |
| full K=8 跨 bank 方差 | +52.68 [39.61,65.75]（0-2）/+42.64 [30.10,55.17]（30-32）/+28.52 [21.61,35.43]（10-14） | `screen/combined.json`、`abs_full/combined.json`、`confirm/combined.json` |
| manifest μ（选摘） | w2m_ctl 191.49±4.51 / w2m_plain 190.28 / lvl4 187.72 / ws_s2 185.96 / random 0 | `traces/pool10/manifest.json` |

**搜索线资产全部使用 bank 30-49**：`trunc_k8/k16/k32/combined.json` 的 `seeds` 均为 `[30..49]`，
`abs_trunc_t5`、`tfront_t*` 为 `[30,31,32]`；`trunc_vs_full_k16` 被中止（JS:169）。截断线没有独立 bank。

### 0.5 同期真人定级（关键）

`traces/sessions/session-20260928-092912/`（**09:29 开始，7 局，就发生在搜索自由对战 09:41 之前**）：

- `report.json` `human.mu=169.02`，95% CI [121.28,216.77]，`n=7`；`provisional=false`，`band="placed"`。
- trace 通道 `mu_traj=160.04`、`sigma_traj=23.88`、**权重 0.9239**；结果通道 σ=83.19。
- 该会话 7 局：human 6胜1负（+20 vs w2m_ctl、+40 vs lvl1、+30 vs lvl2、−10 vs lvl3、
  +60 vs lvl3、+90 vs lvl3、+60 vs ws_s2）。
- **重拟**（同一 `fit_ratings`/`FitConfig`、对手先验 σ=13.97、anchor random=0）：
  - 用 trace 先验 `Prior(160.04,23.88)` → μ=169.0 σ=24.4（复现 report）；
  - 用结果通道先验 `Prior(232.81,93.12)` → **μ=294.0 σ=83.2**（95% CI≈[131,457]）；
  - 冷启动先验 `Prior(232.81,139.69)` → μ=337.6 σ=111.5。
- 更早的 09-27 定级：μ=142.3±57.7（n=3）、92.2±63.1（n=2）、113.1±49.6（n=6）、215.2±50.1（n=6）。

**读法**：真人评级在 169 与 294 之间剧烈依赖先验；155 左右的差距主要来自 trace 先验（权重 0.92）。
搜索自称的 μ≈300（search-config-plan.md:29）与结果-only 的真人 μ≈294 恰好同量级，
与 production 合并值的 169 相差 ~125。

---

## 1. 候选原因清单

### C1. 28 局真人自由对战不是强度测量：样本极小、未配对、未定级、单人（置信：高）

- **命题**：用"真人赢多"来推断"搜索没体现 +110"在统计上不成立；这份数据没有能力分辨
  搜索 0 / +50 / +110 的转移。
- **支持证据**：
  - 23 局、单一真人、单一工作日；严格胜率 15/23，Wilson 95% CI **[0.449,0.812]**（本文计算）。
    均值 +31.3 分、se 9.2 分。
  - 自由对战不进评分链（`docs/search-config-plan.md:17-18,128`），搜索对手没有 manifest μ；
    5 局"无搜索"对照的对手是 random×2、w2m_plain×2、w2m_ctl×1（§0.3），
    不是同对手对照，且集中在最早时段。
  - 搜索局里人类 21/23 在 seat1，seat0 只有 2 局；starter 由亮牌决定，未做平衡设计。
- **竞争解释**：真人确实赢得多，说明搜索对真人无效。该解释不能排除小样本/对手混杂。
- **可证伪检验**：按 `run_h2h.py` 协议做 **成对 deal** 的人机对决（同一副牌×2 座位，
  人 vs raw、人 vs t5K32 同 bank），按 deal 聚簇 bootstrap；或把搜索作为 rung 进 MLE
  （search-config-plan §4.2），再看人类后验与 rung CI 是否分开。所需文件：
  `runs/o4lite-search/run_h2h.py`（`plan_duel_schedule`/`paired_duel_stats` 可复用），
  web 表驱动 `src/seven523/web/table.py:320-366`。

### C2. 真人评分不可识别：production μ=169 是 trace 先验主导；纯结果拟合约 μ=294（置信：高）

- **命题**："真人只有 ~169，却被搜索打败/赢搜索"的矛盾只对 production 合并值成立；
  同一批局的纯结果证据显示真人可能在搜索同一量级，缺口可能是评分口径造成的假象。
- **支持证据**：§0.5。trace 权重 0.9239；结果-only μ=294.0 σ=83.2；
  真人 placement 局还赢了 w2m_ctl(+20)、ws_s2(+60)。
- **竞争解释**：结果-only 的 7 局样本同样小（±163），μ=294 也可能高估；
  trace 先验是 HR 研究推荐的通道，不能随意丢弃。
- **可证伪检验**：
  1. 用同一天会话重拟，只在 `fit_ratings(priors=...)` 里把 human 先验依次换成
     `RESULT_PRIOR`、`COLD_START_PRIOR`、无先验，报告三个后验（本文算法 §4.2）；
  2. 增加定级局数到 HR 建议的 15–27 局（`human-elo-10-games-research.md` §0），
     看纯结果 μ 是否收敛到 <220 或 >280；
  3. 让同一个真人在搜索对手上打 10 局定级（需先按 C3 测 search rung）。
- **所需文件**：`traces/sessions/session-20260928-092912/{session.json,report.json,g*.json}`、
  `src/seven523/placement/estimator.py:26,36,74-100`。

### C3. 尺度/身份错配：+110 是 h2h 相对分，μ≈300 从未进 MLE；不能相加（置信：高）

- **命题**：把 `μ(w2m_ctl)=191.5` 加上 +111/+125 得到"搜索≈300"是未知偏置的外推；
  真人定级池只含 raw（搜索不进池），因此真人从未真的和"μ=300 的搜索"比过。
- **支持证据**：
  - `src/seven523/duel.py:30,71-74`：Elo = `400*log10(p/(1-p))`，400-点 logistic；
    manifest 为 `estimator.kind=probit-mle` 绝对表（`search-config-plan.md:20-22`）。
  - JS:129/167-170 与 `search-config-plan.md:26,85` 明言跨尺度不可加、P2 offset 方案被否决。
  - `search-config-plan.md:29` 的"估 μ≈300"是规划文本；`tools/refit_mle.py` 的搜索 rung
    测量（§4.2）尚未执行。
- **竞争解释**：即便尺度不同，实际差距仍可能很大；probit 与 logistic 在小差距下近似成比例，
  偏置不至于把 +110 变成 0。
- **可证伪检验**：按 `search-config-plan.md` §4.2 附录 A 命令，把 `search_t5k16` 加进
  w2m round-robin（10 对×100 deal×2 座）跑 4,000 局，与现有 18,000 局同一次 MLE；
  比较拟合 μ(search)−μ(w2m_ctl) 与 +111±6 的 h2h 复核闸门。

### C4. 对手模型自对弈假设：评测对手 = rollout 对手 = raw champion；真人 OOD（置信：中高）

- **命题**：搜索的收益在"被评估对手的每一步都等于 rollout 模型"时测得；真人不是这个模型，
  因此 +110 里的对手模型误差项从 0 变成了未知量。
- **支持证据**：
  - `report.md:74-78`："右对手与 rollout 对手模型是同一 frozen champion（模型误差≈0）……
    不直接外推到人类强度"；`trunc_preregistration.md:6-7`（rollout 对手=P0）。
  - 本文实测：22 个 w2m_ctl 搜索局的 481 个真人决策上，w2m_ctl 的 top-1 模板只命中 **44.5%**
    （full action 34.7%），top-6 覆盖 **92.5%**。即 rollout 里对手约 55% 的手动选择与真人不同。
  - 对手模型敏感性：把 rollout 对手从 w2m_ctl 换成 `t18poolself`，同一批真人局面
    **13.2%** 的搜索选择改变（searched 中 17.0%）。
  - 基准侧的迁移臂只覆盖同族网络：`external_wrapped_ws` +41.9、`champion2_ws_s2` +28.5、
    `oppmodel_pself` +24.5（`report.md:110-112`），且 `oppmodel_pself` between-seed sd 20.84
    是 confirm 的 4.6 倍（`report.md:60-64`）。
- **竞争解释**：基准迁移臂已经证明增益不依赖模型完全匹配（换 pself 仍 +24.5），
  人类也落在 top-6 内 92%，对手模型误差未必大到翻转结论。
- **可证伪检验**：
  1. 用人类轨迹训一个对手模型（或直接用 `evaluation` 池），把 rollout 对手换成它，
     在固定 deal bank 上跑 wrapped vs raw 的 h2h；或
  2. 在 28 局真人局面上做反事实：`rollout_opponent` 依次取 w2m_ctl / pself / human-model，
     统计选择翻转率（本文已有 13.2% 的 pself 读数，算法 §4.3）。
- **所需文件**：`runs/o4lite-search/rollout_policy.py:636-668`（rollout 对手注入）、
  `runs/o4lite-search/run_h2h.py:106,138`、`traces/web/run-20260928-091716/*.json`。

### C5. bank 敏感 + 截断线无 fresh-bank 复核（置信：中高）

- **命题**：搜索强度估计对 deal bank 的依赖可达 ~24 Elo；而 +111/+125 只在 seed 30-49 测过，
  没有独立 bank 的 confirm，所以部署档的绝对数字可能带 bank 偏置。
- **支持证据**：
  - 同一 full K=8 配置：+52.68（0-2）、+42.64（30-32）、+28.52（10-14）；
    screen 的 CI [39.61,65.75] 与 confirm 的 [21.61,35.43] 不重叠，abs_full 的 [30.10,55.17]
    与 confirm 只有末端重叠（§0.4）。
  - 截断线所有 `combined.json` 的 seeds 都是 30-49（§0.4）；
    JS 的 confirm（10-14）只测了全量 K=8。
  - 选型网格（t 前沿、K 轴）也都用 30-32/30-49（`tfront_*`、`trunc_k*`），
    没有留下未碰过的 bank 做复核。
- **竞争解释**：三个 full K=8 臂的权重/引擎版本可能略有差异，不能完全归因于 bank；
  20 seeds×400 deal 的 between-seed sd 只有 10-14，bank 均值不太可能偏太多；
  我把 seeds 30-32 剔除后重算 K16/K32 均值只变化 ≤1.5 Elo（K16: 112.6 vs 111.2；
  K32: 125.5 vs 124.7），所以"选型用的 3 个 seed 被复用于报数"这一具体选择效应很小，
  主要问题是**整个截断线只有一个 bank，没有独立复核**。
- **可证伪检验**：在 seed 50-69（全新 bank）上重跑 `trunc_k32` vs raw 与 `trunc_k16` vs raw，
  以及 `abs_full`；预注册判定沿用 z+t 双 CI。命令模板见 `runs/o4lite-search/README.md:52-57`
  （把 `--seeds` 换成 50,51,52）。

### C6. h2h 相对分非传递：截断 − 全量 直接 vs 相减矛盾（置信：中高）

- **命题**：+67.9（直接）与 +34.8（经 raw 相减）差 ~2.75σ，说明"相对 raw 的 Δ"不是可传递的
  序数；"+124.7 vs raw"不能直接换算成对第三方的期望。
- **支持证据**：JS:129/167-170；本文复算 §0.4（同 bank 三个量）。
- **竞争解释**：截断与全量在 raw 面前的共同相位不同，pair 设计只消除发牌/座位，
  不消除策略路径差异；2.75σ 也可能是一次统计波动。
- **可证伪检验**：跑被中止的 `trunc_vs_full_k16`（K=16 下截断 vs 全量直接对决），
  再跑两者各自 vs raw，三条 pairwise 一起做三角一致性检验；
  或换成多策略 Bradley-Terry 全表拟合并报残差。文件：
  `runs/o4lite-search/scale/paired_depth2_vs_k8.json` 同款 `paired_compare.py`。

### C7. 搜索目标（期望分差 / t=5 value 读出）与终局胜负错位，且有确定撬底被覆盖案例（置信：中）

- **命题**：搜索最大化的是隐藏世界平均的期望分差（截断后由 value 头读出），
  不是胜/负；t=5 的读出噪声很大，在有立即撬底（保证不输）的终局会做出投机选择。
- **支持证据**：
  - 机制：`rollout_policy.py:536-568` 对每个候选取 K 个决定化的 mean margin，
    `_argmax_first` 取最大；`rollout_trunc.py:140-152` 在第 5 ply 用 critic 读出，
    按行动方翻转符号。目标函数是分数差，不是胜负（`RULES.md:55,60,63`）。
  - 实测（479 个真人局面决策，champion-vs-champion 全量 rollout 作为单世界真值）：
    value 读出 vs 单世界续行 **EV=0.081、corr=0.33、MAE=33.7 分**；
    同口径 100 局自对局对照 **EV=0.092、corr=0.39、MAE=35.4 分**。
  - **确定撬底案例**：`traces/web/run-20260928-091716/g0000__s3548246500__seat1__vsw2m_ctl.json`
    step 37（0-based）：比分 50-50，draw=0，bot 手牌 ♣6♥6，bot 领出。
    raw 动作 = 模板 16（对子 6-6）→ 立即撬底，终局 margin=0.0（32/32 世界一致）。
    搜索动作 = 模板 1（单牌 ♣6）→ 每世界 value=+0.0816。搜索覆盖了 raw 的确定终局。
    该局最终 50-50（人类手上 0 分牌，没被惩罚），但选择本身是"保证平局 vs 投机 +8 分"。
  - 真人搜索局 bot 撬底 11 次只有 6 次赢，撬底时均分 48.2 vs 51.8；人类撬底 12 次 11 胜 1 平。
- **竞争解释**：基准也用同一目标，所以这是"继承的训练/搜索错位"，不是真人特有的原因；
  单世界真值本身方差极大（±100），MAE 34 分未必是"错"，而是读出与一次抽样续行的自然差异；
  该案例实际没有改变结果。
- **可证伪检验**：
  1. 在同样 479 个决策上跑 `t=full`（全量 rollout）与 `t=5` 的选择翻转率，
     并统计全量是否也覆盖撬底；
  2. 训练/换一个胜率 value 头（或对终局加"立即结束"硬约束），在真人局面上重算选择翻转率，
     再看 endgame 决策的变化；
  3. 用 `mechanism_probe.py` 在构造的终局集合上做单元测试（撬底可得时不得被非终局候选覆盖）。
- **所需文件**：`traces/web/run-20260928-091716/g0000__s3548246500__seat1__vsw2m_ctl.json`、
  `runs/o4lite-search/rollout_policy.py:508-574`、`runs/o4lite-search/rollout_trunc.py:140-162`。

### C8. 搜索在真人局面的改判更多且重尾：改判 61.9%、13.8% 单世界回归（置信：中）

- **命题**：搜索对真人做出的额外改判不是"免费的强度"，其中相当比例在单世界口径下劣于 raw；
  净效应为正（+7.2 分）但方差大，容易被 23 局的小样本放大或掩盖。
- **支持证据**：
  - 改判率：真人搜索局 229/479=47.8%（searched 中 61.9%），基准 K32 为 34.6%（searched 中 49.0%）。
  - 单世界 champion-vs-champion 口径：chosen 优于 raw 24.0%、劣于 raw 13.8%、平 62.2%；
    最差回归 −110/−110/−110/−100/−90 分（`s1851150312` step2、`s2320854290` step0、
    `s1778816045` step15/step23、`s3627354127` step11 等）。
  - 对照自对局 +16.6%/−12.8%/平 70.6%，均值 +1.7 分：真人局面的改判净收益更大、方差也更大。
- **竞争解释**：单世界真值不等于真人对局结果；回归可能只是 champion 模型的方差；
  13.8% 回归伴随 24.0% 提升，净正。
- **可证伪检验**：把 479 个决策按"搜索改判 vs 未改判"分组，与真人对局的实际得分增量做回归
  （需要成对 deal 才干净）；或在同 bank 上做 wrapped vs raw 的 confirm 并分阶段
  （早/中/终局）统计 margin。

### C9. determinization 采样先验不利用对手历史（置信：低-中）

- **命题**：搜索对隐藏世界的采样是"unseen 池均匀 + 最新 revealed 强制归位"，
  不使用人类的出牌序列（pass 了什么、领出什么）作为信息；真人行为可读出信息时，
  搜索的后验世界与真实世界偏离，可能选错。
- **支持证据**：`rollout_policy.py:214-260`（`sample_hidden` 用 `rng.permutation` 均匀划分；
  `public_replay` 只用于 `empty_order`），README.md:16；`self.rng` 每局独立（`:611`）。
- **竞争解释**：在基准里也这么做，且换对手的迁移臂仍为正；人类是否"更可读"没有测量；
  top-6 里已含人类实际动作 92%，采样误差未必改变模板选择。
- **可证伪检验**：从全部 28 局 + study 轨迹训一个"隐藏手牌后验"（例如按 pass/lead 历史的条件分布），
  替换 `sample_hidden` 为后验采样/重要性加权，在固定局面上比较选择翻转率与 h2h Δ。

### C10. value 截断在真人状态更 OOD——数据不支持（置信：低，倾向否定）

- **命题**（待检验假设）：critic 只在 P0-vs-P0 分布校准（`trunc_preregistration.md:21`），
  真人状态上读出偏差更大，使 t=5 搜索变差。
- **支持/反对证据**：真人局面 EV=0.081、MAE=33.7；同口径自对局 EV=0.092、MAE=35.4——
  **没有差异**（本文计算，n=479 vs n=2026）。因此"真人状态让 value 更差"这一简单版本不成立。
- **注意**：这不是说 value 很准（对单世界续行 MAE 34 分），而是说它同样不准于自对局；
  不能把 value 误差单独列为真人转移的差异原因。
- **可证伪检验**：在同一目标上比较 critic_A/critic_B/重训 critic（`runs/ei2_value_t5/`）
  在真人局面的 EV；或按对局阶段分层看 EV 是否随终局接近而恶化。

### C11. top-6 根候选限制——现有证据不支持它是主因（置信：低，倾向否定）

- **命题**：搜索只在 raw 网的 top-6 合法模板里选，好棋若不在其中就永远搜不到。
- **反对证据**：人类实际动作落在 champion top-6 内 **92.5%**；
  in-distribution 的 C=6→C=10 配对只有 +5.14 [−6.46,+17.02]（`runs/o4lite-search/scale/paired_c10k8_vs_k8.json`）。
- **竞争解释**：C=10 检验是全量 rollout/K=8，不是 t=5/K=32；对手/分布也不同；
  好棋在 top-6 外的频率可能在真人局面上升。该解释无法完全排除但缺乏正证据。
- **可证伪检验**：在真人局面上抽样（例如所有终局决策 + 随机 100 个），把
  `max_candidates` 扩到全部合法模板，比较选择翻转率；以及用全行动搜索的
  mean margin 与 top-6 搜索的差距做上界估计。

### C12. 真人跨局适应/可预测（置信：低）

- **命题**：真人可能在 46 分钟内学会固定 bot 的模式；raw 基策略完全确定，搜索只改一半。
- **支持证据**：raw 重放 100% 确定（§0.2）；真人后段（最后 9 局）7胜1负1平，前 10 局 6胜4负；
  搜索改判只覆盖 47.8% 决策，剩余 52.2% 仍是 raw 的确定动作。
- **竞争解释**：趋势不显著：Spearman(outcome~order) rho=0.091 p=0.679；
  前 10 vs 后 9 的 Fisher p=0.370；粗粒度相同状态的重复动作一致率只有 13.7%，
  没有"固定复读机"的证据。
- **可证伪检验**：更多同人同模型对局下按局序做 logistic 回归（胜负~局序，控制首发/座位）；
  或对同一粗状态集合做动作分布检验（需要逐决策 logits，trace 里没有，需重放前向）。

---

## 2. 数据缺口（无法从仓库回答的判断）

1. **真人的真实强度**：唯一同期定级 7 局，且 production 值被 trace 先验主导（169）；
   纯结果 294±83。没有搜索对手上的真人定级，也没有 15–27 局的口径。
2. **搜索的绝对 rung**：`search_t5k16` 从未进入 MLE（`search-config-plan.md` §4.2 仍是规划）；
   "μ≈300" 是外推。
3. **trace 不记录部署细节**：`opponent_search` 只有 `{trunc_ply, rollout_k}`；
   `value_ckpt`、`max_candidates`、`rollout_opponent`、每决策的 searched/fallback/value
   都不在 trace 顶层（`src/seven523/trace.py`、`table.py:562-565`）。
   本文只能靠离线重放证明是 critic_B；若当时服务端用了别的 value ckpt 而恰好同选，无法从 trace 区分。
4. **没有成对/随机化的人机数据**：不分 deal、不分座位、非盲；首发由亮牌随机，未分层。
5. **没有人类行为模型**：无法形式化检验 rollout 假设；本文只能给 top-1/top-6 一致率
   （44.5%/92.5%）。
6. **没有真人动作的逐决策 logits/value**，故"人类最优动作"与搜索候选集的关系只能做模板级比较。
7. **旧定级会话（09-25）不可用**：`traces/sessions/session-20260925-*` 是 revision-3 之前的规则，
   `report.json` 且用旧锚点（random=1000/greedy=1315），与当前 probit 表不可比。
8. **非传递性的来源无法从现有数据分解**：截断/全量/raw/人类四条边只有 3 条有数据，
   第四条（截断 vs 全量 @K16）被中止（JS:169），无法闭合三角。
9. **bot 在真人局的"真值"不可得**：人类是自适应的，counterfactual 只能拿 champion 当代理，
   因此 §C7/C8 的单世界回归幅度是模型内量，不是真人对局反事实。

---

## 3. 对抗审查指引（后续子代理按此攻击）

1. **C1/C2 的样本量攻击**：任何"真人很强/很弱"的断言先看 n=7 与 n=23 的 CI；
   用 Wilson/二项与结果-only 重拟复核。
2. **C4 的模型误差攻击**：检查 top-1 一致率是否被 suit/template 口径影响；
   重跑换对手模型实验时必须固定同一 rng stream（本文 §4.3 用同一 seed 的两个 wrapper，
   determinization 相同，差异只来自对手模型）。
3. **C5 的 bank 攻击**：所有截断类 `combined.json` 的 seeds 字段必须逐一核对；
   重跑新 bank 的命令必须带 `--pairs 400 --seeds <fresh>`，并与旧 bank 逐 deal 不可配对
   （只能独立复核，不能合并）。
4. **C6 的非传递攻击**：直接复算 `trunc_k32/combined.json` 与 `abs_full/combined.json`；
   任何"截断更强"的结论必须同时报直接对比与经 raw 相减两个口径。
5. **C7 的案例攻击**：重新执行 §4.4 的决策勘验；注意 32/32 世界一致是关键，
   若某个世界 pair 不是 0.0，则该案例降级。
6. **C10/C11 的否定攻击**：任何"value OOD/top-6 是主因"的新主张必须给出
   真人局面 vs 对照局面同口径的 EV/C=10 数据，否则视为无证据。

---

## 4. 复现算法（脚本在 `/tmp/humangap/`，本文不新增仓库文件）

### 4.1 轨迹重放与事实表

`/tmp/humangap/summarize.py`、`replay_analysis.py`/`replay2.py`：

```
for trace in sorted(traces/web/run-*/*.json, key=created_at):
    rules = rules_from_json(trace.rules); state = state_from_snapshot(trace.initial, rules)
    match = Match(rules, [None,None], state=state)
    for i, step in enumerate(trace.steps):
        assert match.state.current == step.seat
        view = match.view(); assert view.mask >> step.action & 1
        记录 seat/lead/hand/draw/trick_pts/action/kind/dug/...
        match.step(step.action, step.suit)
```

胜负按 `CONTEXT.md:79-81`（比分高者胜）；撬底者取 `step.dug==true` 的 seat。

### 4.2 真人定级重拟

```
games = [PlayedGame(seed, (human,opp) 或 (opp,human), final_scores) for g in session g*.json]
priors = {human: Prior(μ_h, σ_h)} | {opp: Prior(manifest.μ_opp, 13.9686)}
fit_ratings(games, anchors={"random":0}, priors=priors, config=FitConfig())
```

μ_h/σ_h 依次取 trace 通道 `(160.04,23.88)`、结果通道 `(232.81,93.12)`、冷启动 `(232.81,139.69)`。
本文得到 169.0 / 294.0 / 337.6，与 report.json 的 169.02 对齐。

### 4.3 部署重放（搜索是否生效）

`/tmp/humangap/verify_search2.py`：

```
sys.path = [src, runs/o4lite-search, runs/speed]; fast_engine.install()
agent = load_agent("runs/ei/value/critic_B.pt")
polseed = policy_seed(trace.seed, trace.human_seat)          # 注意：human seat
base = policy_from_spec(manifest.spec[opponent_id], rules, seed=polseed)
champ = ValueChampion(agent, rules, "cpu", trunc_ply=5)      # 全量臂用 BatchChampion
pol = O4LiteWrapper(base, champ, rules, determinizations=32, max_candidates=6,
                    max_rollout_ply=400, rollout_opponent=champ, seed=polseed)
逐 step：bot 决策点比较 pol.act(view) 与 trace action；同时比较 base.act(view)
```

对手模型敏感性：同一 seed 构造两个 wrapper，`rollout_opponent` 分别为 champ 与
`BatchChampion(load_agent("runs/t18poolself__2__1790499154/agent.pt"))`，逐决策比较模板。

### 4.4 终局案例与单世界 value 探针

- 案例：重放到 `g0000__s3548246500__seat1__vsw2m_ctl.json` step 37，调用
  `root_candidates`/`rebuild(rng=pol.rng)`/`batched_decide`，打印每个候选的 `means`
  与逐世界 `values`（对子 16 = 0.0；单牌 1 = 0.0816）。
- 探针 `/tmp/humangap/value_probe_human.py` / `value_probe_control.py`：
  对每个 bot 决策，先 `game.step(state, chosen)` 与 `game.step(state, raw)`，
  再用 `BatchChampion.rollout_values(..., opponent=None)` 得全量 champion-vs-champion 终局 margin，
  用 `ValueChampion(trunc_ply=5).rollout_values(..., opponent=vc5)` 得 t=5 读出；
  比较 EV/MAE 与 chosen-vs-raw 的真世界差。注意 `rollout_values` 会就地改写传入的 state 列表，
  每次调用必须传 `list(...)` 副本。

---

## 5. 附：本文新增/复算的关键数字一览

| 数字 | 值 | 算法 |
|---|---|---|
| 搜索局真人战绩 | 15W 2D 6L，均分 +31.3，Wilson [0.449,0.812] | §4.1 |
| 人类胜率 vs 搜索的 h2h Δ | +143.6 | 400*log10(0.6957/0.3043) |
| P(≥15 胜 \| 真人=raw 水平) | 0.00145 | 二项 |
| 部署重放匹配 | 479/479，fallback 0 | §4.3 |
| 搜索改判 | 229/479=47.8%（searched 61.9%） | §4.3 |
| 真人动作 top-1/top-6 命中 champion | 44.5% / 92.5%（n=481） | 重放 + masked logits |
| 换 rollout 对手 flipping | 63/479=13.2%（searched 17.0%） | §4.3 |
| t=5 value vs 单世界全量：真人/对照 | EV 0.081 / 0.092；MAE 33.7 / 35.4 | §4.4 |
| chosen vs raw 真世界 margin：真人/对照 | +7.2 / +1.7 分 | §4.4 |
| 真人 digs | 12 次，11W 1D 0L，均值 78.3 vs 21.7 | §4.1 |
| bot digs | 11 次，6W 4L 1D，均值 48.2 vs 51.8 | §4.1 |
| 同日真人定级（合并/trace 先验） | μ=169.0 [121.3,216.8] n=7 | `session-20260928-092912/report.json` |
| 同日真人定级（结果-only 重拟） | μ=294.0 σ=83.2 | §4.2 |
| full K8 跨 bank | +52.68 / +42.64 / +28.52 | §0.4 |

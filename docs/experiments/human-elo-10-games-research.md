# 真人 10 局快速定级研究：轨迹先验 + 结果似然的定量结论

> 资产状态（2026-09-25）：本报告引用的模型 checkpoint 已删除，旧路径不再可用；数值为牌型族规则变更前口径。

> **状态：离线研究完成（未改 `src/` 核心、未 commit、未触碰 head-to-head 相关文件）。**
> 数据 = `traces/study/` 的 6 级 × 200 局冻结轨迹（200 副完全共享的牌 × 6 个 bot 等级，
> 完美成对设计）；估计器 = 仓库自带的 `elo.fit_ratings`（ADR-0006 接缝）。
> **结论：10 局内做不到 ±50（推荐组合的点估计 RMSE≈54–72，诚实口径 95% CI≈±100–133）；
> 10 局的实际交付是「RMSE 54–72、±1 档命中 83–94%、带诚实 CI 的临时档位」。要达到
> RMSE≤50 需约 15–27 局，95% CI 收进 ±50 需约 60–100 局。最强信号是 S1 逐局特征 +
> 去收缩标定；S2（价值）/S3（策略一致度）经不起留一等级检验，不建议进生产。**

复现脚本与产物：`runs/human_elo_10g/`（gitignore 下），全部命令见 §11。

## 0. 关键数字（先看这里）

| 问题 | 答案 |
|---|---|
| 纯胜负，10 局 | RMSE(T2) ≈ **127**；名义 95% CI ±222（覆盖 0.94） |
| 胜负 + 分差，10 局 | RMSE ≈ **96**，P100 0.70，名义 CI ±144（覆盖 0.86） |
| **推荐组合**：S1 去收缩先验 + 胜负 + 分差(σ×2)，10 局（乐观口径 grouped CV） | RMSE ≈ **54**，P50 0.65，P100 0.93，CI ±100（覆盖 0.93） |
| 同上（诚实口径，留一等级 LOLO） | RMSE ≈ **72**，P50 0.51，P100 0.83，CI ±133（覆盖 0.94） |
| 10 局 ±50（任务定义需 m_eff≈18.5） | **做不到**：最好情形 P50=0.65、95% CI 仍 ±100 |
| 达到 RMSE≤50 所需局数 | 约 **15 局**（grouped 外推）/ **27 局**（LOLO 外推） |
| 95% CI 收进 ±50 所需局数 | 约 **63 局**（grouped 外推）/ **100 局**（LOLO 外推）；纯结果 win+margin 约 **91 局** |
| 最强信号 | **S1 逐局特征**（trick_win_rate / trick_point_share / lead_rate / pass_rate + 阶段拆解 + 对手强度）经「去收缩」标定；session m_eff 6.1（LOLO）–13.9（grouped） |
| 最被高估的信号 | **S3 策略一致度**：grouped CV m_eff=46.6，LOLO 只剩 8.0（风格泄漏，可被模仿） |
| 失败信号 | **S2 价值 regret**：价值头离策略且未标定，m_eff 3.4–4.0，叠加在 S1 上无增量 |
| 对手选择 | 自适应挑「当前后验下 p(1−p) 最大」的档；比固定 greedy 好约 2–4% RMSE，接近 oracle 的 94–98%；首批 1–2 局用 Thompson 防先验偏 |

## 1. 目标定义与理论下限（先核对公式）

**评分口径**：400 分 logistic Bradley–Terry，锚点 Random=1000、Greedy=1315 固定
（`docs/human-elo-plan.md:36-53`；`src/seven523/elo.py:224-275`）。

**「自身水平」的可测化**（本文全程两个目标）：

- **T1 = manifest 冻结标签**（`traces/study/manifest.json`：random 1000 / lvl1 1119.9 /
  lvl2 1232.9 / greedy 1315 / lvl3 1359.3 / lvl4 1489.4）。这是本文写作时的生产刻度，
  但带已知的座位相位偏移（`elo-reliability-audit.md` §5.4：lvl4 可能高估 ~40）。
  **勘误（2026-09-25 refit 执行后）**：manifest 已于当日 refit（lvl1 1146.3 / lvl2 1219.2 /
  lvl3 1363.6 / lvl4 1429.7，`ladder-rerating-paired.md` §4.4）；下文所有旧 T1 数字与
  基于旧 lvl4 的比较均为历史口径，当前契约见 `README.md` §0.1。
- **T2 = 用同一批 200 局、同一估计器（胜负+分差）拟合出的「满数据」值**：random 1026.9 /
  greedy 1314.4 / lvl1 1128.6 / lvl2 1232.1 / lvl3 1356.1 / lvl4 1447.8（SE 17–18）。
  T2 是「无限局数后该调度能收敛到的值」，用来把**会话采样/先验误差**和**标签相位偏差**分开。
  T2 与 T1 的差：random +26.9、lvl4 −41.6，其余 ≤9——这正是审计说的相位量级。

**验收口径**（沿用计划 §4 的 100–150 档距）：|误差|≤50 = 「同档内」；
|误差|≤100 = 「±1 档」；另报 95% CI 覆盖率。

**公式核对**（单局 Bernoulli，p=0.5）：

```
β = ln10/400 = 0.005756
SE₁ = 1/(β·√(p(1−p))) = 1/(0.005756×0.5) = 347.4 Elo
95% CI 半宽(n 局) = 1.96·347.4/√n = 680/√n      # 计划 §0 的公式正确
n=10, m_eff=1 → ±215.0
±50  ⇔ m_eff = (680/50)²/10 = 18.5
±100 ⇔ m_eff = (680/100)²/10 = 4.62
```

**但 ±215 是「无先验、无分差、每局 p=0.5」的悲观下界。** 本仓库实测（§7）：把分差
似然和 1500±200 弱先验算进去，10 局纯结果口径的名义 CI 是 ±144（RMSE 96），
不是 ±215。要在 10 局到 ±50，仍然必须把 `m_eff` 从 ≈1 提到 ≈18——本文任务就是
把轨迹能提供的 `m_eff` 榨出来并诚实测量。

## 2. 一手资料梳理（读了什么、采信什么）

| 资料 | 关键事实 | 对本题的意义 |
|---|---|---|
| `docs/human-elo-plan.md` | 20 局 ±150 地板；`m_eff` 框架（:40-53）；S1/S2/S3 定义与预期（:56-81）；标定与 `m_eff=(347/s)²`（:121-137）；D3 混合估计器（:140-146）；验收（:152-159） | 研究问题的母本；D1 工具、manifest、elo 接缝都按它落地 |
| `docs/experiments/trace-signal-report.md` | S1 岭回归 OOF `s=88.7`、`m_eff=15.4`；但只有 2 级、级距 315、级别内 y 恒定（§2.2、§3.1） | 它的 15.4 是**乐观上界**；本文用 6 级重测并把口径分成 grouped/LOLO |
| `tools/measure_trace_signal.py` | S1 特征提取（`FEATURE_COLUMNS` :88、`extract_features` :306）、严格 replay 校验（:306-320）、grouped-by-seed CV（:522-620）、`m_eff=(347/s)²`（:724） | 本文直接复用 `extract_features`；CV 口径扩成 LOLO |
| `src/seven523/trace.py` | 轨迹格式唯一 owner：`step_record` :128、`build_trace` :161、`TRACE_VERSION=1` | 字段清单见 §3；轨迹**不含** value/logits |
| `src/seven523/play.py` | 人机录制路径 `play_game(record=...)` :272-350；`--save-trace` :430-470；`replay_trace` 严格校验 | 人类 trace 与 study trace 同一 artifact；玩家标签目前不含等级 id（§8 需小改） |
| `src/seven523/env.py` | `encode_observation` 段布局 :60-110；`joint_mask_bits` | 离线算 S2 时必须从 View 重编码 obs |
| `src/seven523/elo.py` | `Prior` :45、`PlayedGame` :62、`FitConfig.margin` :99、`_gradient_info` :161、联合 Hessian SE :188/:292-310、`fit_ratings` :224、窗口 :218、`select_rungs` :364 | D3 只用到「先验 + 分差 + 窗口」三个已有接缝 |
| `docs/adr/0006-elo-rating-seam.md` | 轨迹先验通过 `priors={id: Prior(μ_traj, σ_traj)}` 进入同一 `fit_ratings`；结果似然是最终权威；决策级证据需新模块 | D3 不新增数学接缝；编排（选对手/停）放新 `placement.py` |
| 三个浏览器原型 | `elo_prototype`：K 退火（K=16/40）、400 分 logistic、阻尼牛顿 cap 300 + [400,2600]、`fitRating` SE；`elo_calibration_prototype`：4 锚 × 5 局预热 + 每 10 局批量 MAP + CI 达标即停；`elo_placement_prototype`：10 级 100 分密集梯、20 局结算、`pickOpponent`=argmin\|p−0.5\|（:262-269）、窗口=精度地板 | 数值方法全部被 `elo.py` 继承；缺的是**先验来自轨迹**和**成对座位/冻结牌堆**（审计 §6.1） |
| `docs/experiments/elo-reliability-audit.md` | 单 fit 的 Fisher SE 只覆盖「换牌」，不覆盖「换序/座位相位」（§0、§3.5）；换座配对把 sd 从 ~31 降到 ~15（§5.2）；`--cross` SE 曾低估 4.16×（§2.3，已修）；§5.4 manifest 标签带相位 | 定级调度必须**每副牌双座位**；本文所有仿真用共享牌堆已经是成对设计 |
| `docs/experiments/wave5-500k-report.md` §4 | 同 fit 内相对比较可靠；父/子偏移随牌集整体翻符号；跨 seed sd 可达 37 | 单个 10 局会话的绝对值不能跨牌集比；报告必须给 CI 与「跨牌集再练」的意思 |

## 3. 轨迹里实际可用的字段（`trace.py` 是唯一 owner）

一局 trace 的完整 schema（`build_trace` 组装，`step_record` 逐步写入）：

```jsonc
{
  "version": 1, "created_at": "...",
  "rules": {"num_players":2,"hand_size":7,"straight_min":3,"straight_max":7,
            "consecutive_pairs_min":3,"consecutive_pairs_max":3,"total_points":100},
  "seed": 3626764237,            // 同 seed ⇒ 同一副牌（发牌确定）
  "human_seat": 0,               // 受评座位
  "players": ["subject:lvl1@seat0", "anchor:random@seat1"],
  "initial": {"starter":0,"hands":[...],"draw_pile":[...],"revealed":[...]},
  "steps": [ {                   // 每个动作一条，含对手动作
     "seat":0,"action":9,"suit":2,"text":"单牌 ♣3",
     "scores":[0,0],"hand_sizes":[6,7],"draw_count":40,
     "trick_over":false,"winner":null,"points":0,"dug":false } , ...],
  "final_scores": [80,20]
}
```

**每局可直接读取/推导**（`tools/measure_trace_signal.py:300-405` 已实现）：

- 逐墩：赢家 `winner`、墩分 `points`、`dug`（撬底）、收分方、手牌数、底牌数、分数；
- 受评座位：墩胜率、墩分份额、过牌率、炸弹率、先手率、早/中/晚阶段墩胜率、
  首墩、每次赢墩均分、决策数/墩、是否赢下撬底；
- 结果：`final_scores` → 胜负、分差。

**轨迹里没有**（必须离线重算，且这正是 S2/S3 的成本与风险来源）：

- observation 向量与 action mask：可从 `initial` 重建 state、逐动作 replay，再用
  `env.encode_observation` + `Game.view` 得到（`env.py:60-110`）；
- 策略 logits / 概率：需要 checkpoint 前向（`Agent.actor`，`networks.py:110`）；
- 状态价值 `V(s)`：需要 checkpoint 价值头（`Agent.get_value`，`networks.py:104`）；
- 动作价值 `Q(s,a)`：**没有**。只能做 1 步 TD（`V(s_t)−V(s_{t+1})`）或短 rollout 近似，
  本文实测 1 步 TD 信号很弱（§4）；
- 对手手牌：规则不可见（`View` 只给公共信息 + 自己手牌，ADR-0002），价值头已按此训练。

## 4. 轨迹信号的信息量再评估（定量）

### 4.1 数据与方法

`traces/study/` 恰好是一个**完美成对阶梯**：200 副牌被 6 个等级全部打过（§11 的
`paired_probe.py` 验证 `all levels present on all deals: True`），每级 100 局 vs random、
100 局 vs greedy，lvl1/lvl3 与 lvl2/lvl4 是同（牌,座位,锚点）双胞胎。这比 D1 的
2 级 × 200 局强得多。

- S1 特征：`CLEAN+PACE`（trick_win_rate、point_share、lead_rate、pass_rate、阶段拆解、
  mean_points_per_won_trick、bomb_rate、first_trick_won、tricks_won、points_won、dug_won、
  decisions_per_trick、tricks_total）+ 对手强度 + trick_win_rate×对手强度。
- S2 特征（本文新采）：用 lvl1–lvl4 四个 checkpoint 在受评座位的每个决策上算
  `V_e(s)`、1 步 TD `V_e(s_t)−V_e(s_{t+1})`（末决策对终局 return）。
- S3 特征（本文新采）：同一批前向里算 模板头 `log P_e(a实际|s)`、argmax 命中、
  实际动作排名、策略熵、花色头熵。
- 标定：岭回归（标准化，α 由 CV 选），两种 CV：
  **grouped** = 按 seed 留出整副牌（同牌跨等级不泄漏，但每个等级都出现在训练里）；
  **LOLO** = 留出一整个等级（风格迁移，逼近「模型没见过这个人」的真人场景）。
- 口径：残差 `s = SD(y−ŷ_oof)`，`m_eff=(347/s)²`；并做方差分解
  `s² ≈ τ²(等级间偏差) + σ_w²(等级内噪声)`，再实测 n 局会话先验误差 `psd(n)`。

### 4.2 结果：m_eff 与「会话有效 m_eff」

`m_eff` **不是常数**：等级间偏差 τ 不随局数缩小，所以 10 局的会话有效信息量
`m_eff_session(10) = (680/(psd(10)·√10))²` 比逐局公式小得多。这是计划 §0 框架里
没有显式区分的点，也是所有乐观数字的来源。

| 信号 | CV | 逐局 s | 逐局 m_eff | τ（等级间） | σ_w（等级内） | psd(10) Elo | 会话 m_eff(10) |
|---|---|---|---|---|---|---|---|
| **S1**（clean+pace） | grouped | 107.3 | 10.5 | 83.3 | 75.8 | 80.2 | 7.2 |
| **S1** | **LOLO** | **134.8** | **6.6** | **120.5** | 78.0 | **113.4** | **3.6** |
| S1 + 去收缩标定 | grouped | 134.3 | 6.7 | 47.4 | 127.2 | 57.5 | 13.9 |
| S1 + 去收缩标定 | LOLO | 156.3 | 4.9 | 84.6 | 135.9 | 87.4 | 6.1 |
| S2 价值 `V_mean/V_last/V_min` | grouped | 153.3 | 5.1 | 160.7 | 44.8 | 146.9 | 2.1 |
| S2 价值 | LOLO | 188.8 | 3.4 | 200.9 | 44.7 | 183.6 | 1.4 |
| S2 1 步 TD | grouped | 142.6 | 5.9 | 139.1 | 64.9 | 128.2 | 2.8 |
| S2 1 步 TD | LOLO | 173.8 | 4.0 | 176.8 | 64.4 | 162.4 | 1.8 |
| S3 策略一致度 | grouped | 50.8 | **46.6** | 24.6 | 45.6 | 26.7 | **64.8** |
| S3 策略一致度 | **LOLO** | 122.3 | **8.0** | 126.2 | 41.0 | 116.1 | 3.4 |
| S1+S3 | LOLO | 122.6 | 8.0 | 126.3 | 41.7 | 116.1 | 3.4 |
| S1+S2 | LOLO | 135.0 | 6.6 | 120.0 | — | 112.1 | 3.7 |

三条可直接落地的读法：

1. **S1 是唯一稳的信号。** LOLO 逐局 `m_eff=6.6`（超过计划 go 阈值 3），但会话
   有效值只有 3.6——因为模型的等级间偏差 τ≈120 不会随局数缩小。grouped 口径给
   7.2/13.9，但那是「训练时见过这个人所属的等级」，对真人偏乐观。
2. **S3 是本数据集里最大的陷阱。** grouped CV 看起来能到 m_eff=46.6（`s≈51`，
   psd10≈27），非常诱人；LOLO 直接塌到 8.0，且 S3 的 τ=126 比 S1 还大。分组 CV 下
   「用同一等级的策略算自己动作的 log-prob」是在测**同一策略的重识别**，不是强度；
   真人不在任何等级的分布里，S3 只会更差。**不要单独用，也不要因为有 grouped 数字
   就把它写进模型。**
3. **S2 价值信号目前不值得做。** 价值头在离策略状态上系统性失准（τ=201），
   1 步 TD 稍好也远不如 S1；把 S2 加到 S1 上 LOLO 残差 135.0 vs 134.8（无增量）。
   「价值头置信度」本身也只是价值头尺度，不携带独立信息。若要做决策级 regret
   （Q 值/反事实 rollout），需要新模块且要先解决价值头对人类的标定（§9）。

### 4.3 两个被否掉的「更聪明」想法（负面结果）

- **同牌参考面板（冻结牌堆配对）**：审计 §6.5 建议用同牌/换座配对降方差。实测：
  同 (牌,座,锚点) 双胞胎的 margin 残差 SD = 54.6–59.3；去掉等级/锚点效应后的原始
  单局 SD = 42.4；用另外 5 级的同牌均值做参考，残差 SD = **54.5**（更大）。
  结论：这里的噪声是**等级×牌局交互**（不同打法对同一副牌的利用方式不同），
  不是共享的「牌运」，**配对不能降噪**。审计的配对收益针对的是「两个候选之间的
  共同相位」，不是「单人对锚点的绝对定级」。
- **去收缩标定**是有效的：LOLO 等级均值预测被压到 146 Elo 跨度（真值 489），
  用 5 个训练等级做 affine `y ≈ a+b·ŷ`（斜率 1.6–2.6）再外推，τ 从 120 降到 85，
  psd10 从 113 降到 87；代价是逐局噪声被同倍放大（σ_w 78→136），净收益约 25%。
  grouped 口径下 τ 85→47、psd10 81→58（净收益 ~28%）。生产建议固定 b≈1.6–1.8
  （而不是每折 2.6），保守。

## 5. 估计器规格（D3：轨迹先验 + 结果似然）

### 5.1 数据流

```
人类对局 (7g523-play --save-trace)         结果 (score_own, score_opp, 对手 rung id)
        │ 每局
        ├─ replay_trace 校验 ──> View ──> S1 特征 φ
        │                                     │
        │                            离线标定模型 g(φ) + 去收缩 a+b·g(φ)
        │                                     │  会话均值 μ_traj, 校准表 σ_traj(n)
        ▼                                     ▼
   窗口内 PlayedGame ───────────────► fit_ratings(anchors=秤砣, priors={"human": Prior(μ_traj,σ_traj)}, margin=(c,σ))
                                              │
                          后验 mean ± 1.96·se（联合 Hessian）、最近梯级、下一步对手
```

### 5.2 先验

- **特征→Elo**：岭回归（12–17 维，α≈30，标准化），训练集 = 冻结阶梯的 6 级 × 200 局；
  必须带**对手强度**（`anchor_elo_ref`）及其与 trick_win_rate 的交互，否则对手档位变化
  会被读成实力。
- **去收缩**：对等级均值做 affine 标定 `f'(φ)=a+b·f(φ)`，`b≈1.6–1.8`、`a≈−900`
  （在 6 级阶梯上拟合；5 折留一等级斜率范围 1.62–2.55，故取保守值）。
- **会话先验**：`μ_traj = mean_i f'(φ_i)`；`σ_traj(n)` 来自离线标定表（LOLO 去收缩：
  n=5→99、10→87、20→82；grouped 去收缩：n=5→71、10→58、20→50；表里应存 LOLO+去收缩
  版本作为默认）。`Prior(μ_traj, σ_traj)` 直接进 `fit_ratings(priors=...)`
  ——这就是 ADR-0006 约定的 D3 入口，**不需要改数学核**。
- **锚点/梯级**：Random、Greedy 保持 `anchors` 钉死（刻度）；lvl1–lvl4 作为自由 id
  给 `Prior(manifest_elo, 30)`，让它们自身的 label 不确定性进入联合 Hessian（旧口径 ±27–34，refit 后为 ±6–7，见 `README.md` §0.1；σ=30 需在 M2 实现时复核）；
  不要为了图省事把梯级全部钉死。
- **先验强度如何收缩**：`σ_traj(n)` 随 n 降，但因为 τ 不缩，10 局后就接近平台
  （LOLO 去收缩 ~80）；继续加局数时结果似然以 1/√n 压过先验，最终收敛到 T2。
  如果消息量更大，可以在离线表里按「轨迹信号质量」再分档（如 S1 特征完整/缺失），
  本文未测。

### 5.3 似然

- **胜负**：Bernoulli logistic，权重 β=ln10/400（`elo.py:_gradient_info:161`）。
- **分差**：`margin ~ N(c·(R−R_opp), σ²)`，用同一份 6 级数据标定：
  全样本 `c=0.123、σ=42.4`（受标签误差衰减 + 两端饱和压低斜率）；
  限制 \|ΔR\|≤260 的线性段 `c=0.143、σ=44.5`。**纯结果口径推荐 (0.143, 44.5)**。
  Fisher 比 `(c/σ)²/(β²·0.25) ≈ 1.0–1.25`——分差只值 ~0–25% 的信息，不是原型里
  假设的 1.35 倍；在弱先验/两端档位时它的稳定作用更大（§7.4）。
- **轨迹先验在场时，分差必须降权（§7.4 实测）**：特征与对局结果相关，
  「特征先验 + 分差似然」会重复计数。保持 σ、只加轨迹先验时 10 局覆盖只有 0.88；
  把分差 σ 乘 2（⇒ `(0.143, 89)`）后覆盖回到 0.93–0.94，RMSE 还略好于去掉分差。
  生产默认：**有轨迹先验时 `margin=(0.143, 89)`；无轨迹先验（冷启动/长期跟踪）时
  `margin=(0.143, 44.5)`**。
- **窗口/在线**：定级的前 10 局 `window=None`（全取），因为目标就是把这 10 局用完；
  定级后切 `window=50`（或遗忘因子）跟踪真人进步/手生，先验用最近 k 局轨迹重算。
- **逐局重算**：每次 `fit_ratings` 在 10 局规模下 ~30µs（实测 1000 次 0.03s），
  可以每局刷新后验，不需要「每 10 局批量」的原型妥协。

### 5.4 伪代码

```python
# 离线一次（新增 tools/fit_trace_prior.py）
model = ridge_fit(study_features, level_elo, alpha=30)          # g(φ)
a, b = affine_on_level_means(model, study_levels)               # 去收缩
prior_sd = {n: empirical_session_sd(n, scheme="lolo_shrunk")}   # 标定表

# 在线：人机定级会话（新增 src/seven523/placement.py）
posterior = Prior(trace_mean(f(model, first_trace)), 300)       # 首局前宽先验
for n in 1..10:
    opp = argmax_rung E_{R~posterior}[p(R, rung)(1 - p(R, rung))]  # §6
    game = play_one(opp, deal=schedule[n])                      # 冻结牌堆，双座位成对
    history.append(PlayedGame(seed=deal, seats=("human", opp.id), scores=game.scores))
    mu_traj = mean(f(model, φ_i) for i in history)              # 轨迹通道
    fit = fit_ratings(history, anchors={"random":1000,"greedy":1315},
                      priors={"human": Prior(mu_traj, prior_sd[n]), **rung_priors},
                      config=FitConfig(margin=(0.143, 89.0)))   # 轨迹先验在场时分差降权（§7.4）
    posterior = fit.ratings["human"]
    if 1.96 * posterior.se <= 50 and n >= 8: break              # 可选提前停
report(posterior.elo, 1.96*posterior.se, nearest_rung(posterior.elo),
       provisional=(1.96*posterior.se > 50))
```

### 5.5 停止规则与输出

- **硬上限 10 局**（产品约束）。10 局时：`R̂ = posterior.elo`，报 `±1.96·se`，
  最近梯级；**显式标记 provisional**（本研究表明 10 局的 CI 必然 > 50）。
- 分级：`CI ≤ 50` → 已定级；`50 < CI ≤ 100` → 临时（±1 档）；`> 100` → 粗档。
- 继续对局（不必是定级局）时把新 trace 并入窗口，CI 自然收窄；到 `CI ≤ 50` 再落最终档。
- 输出还应带上「两通道的权重」：轨迹先验贡献了多少精度、结果似然贡献了多少，
  以及最影响先验的 2–3 个特征/决策——这是计划 §6.7 的可解释性要求。

## 6. 对手选择（placement / active matchmaking）

### 6.1 信息量原则

一局对锚点 `a` 的 Bernoulli Fisher 信息 `I = β²·p(1−p)`，在 p=0.5 最大。
固定打 Greedy（原型 `pickOpponent` 用 argmin\|p−0.5\| 是对的）：
对 6 级阶梯的平均 `p(1−p)` 只有 0.206，自适应能到 0.231，oracle 0.250。

用 manifest 标签算「当前可选梯级里最好的 p(1−p)」（人类 R 在梯级外时）：

| 人类 R | 最近梯级 | p | p(1−p) | 相对最大信息 |
|---|---|---|---|---|
| 900 | random | 0.36 | 0.230 | 92% |
| 1000–1500（梯级内） | 总有相邻档 | 0.45–0.56 | 0.247–0.250 | 98.6–100% |
| 1600 | lvl4 | 0.65 | 0.226 | 90% |
| 1800 | lvl4 | 0.86 | 0.123 | 49% |
| 2000 | lvl4 | 0.95 | 0.048 | 19% |
| 2200 | lvl4 | 0.98 | 0.016 | 6.5% |

结论：**在 1000–1500 区间，6 级梯子足够密**（任何人的相邻档都能给出 ≥98% 的信息）；
**超过 ~1600 后信息量随饱和塌缩**，这时 10 局不可能准，必须扩梯级或加 handicap。

### 6.2 策略与设计实验

用 BT 生成胜负（保守：不用分差）、把 §4 的 LOLO/grouped 会话先验误差注入先验，
比较 6 种策略（每级 200 局会话 × 6 级，n=5/10/20）：

| 策略 | p(1−p) 平均 | LOLO n=10 RMSE | group n=10 RMSE | LOLO n=20 RMSE |
|---|---|---|---|---|
| 固定 greedy | 0.206 | 81.4 | 65.2 | 68.8 |
| 均匀随机档 | 0.186 | 84.0 | 66.9 | 69.7 |
| **info**（后验均值 argmax p(1−p)） | **0.234** | **80.8** | **64.8** | 66.7 |
| einfo（对后验期望 p(1−p)） | 0.234 | 80.8 | 64.8 | 66.7 |
| **thompson**（后验采样） | 0.220–0.226 | 82.5 | 65.1 | **64.9** |
| oracle（真值 argmax） | 0.250 | 79.4 | 63.3 | 67.2 |

读法：
- 自适应选择**严格优于**固定/均匀，但 10 局下只值 2–4% RMSE——**不是瓶颈；瓶颈是先验偏差**。
  覆盖率都在 0.94–0.96，说明「先验 sd + 结果似然」在这个理想世界里标定良好。
- **info 达到 oracle 的 94–98% 信息量**，但**前 1–2 局可能选错档**（先验 μ 有 ±80–115 偏差）。
  推荐：**前 2 局用 Thompson**（按后验采样选档，天然探索 ±1 档），之后切 info；
  或直接全程 Thompson（n=20 时它最好）。
- 避免「一直打错档」的机制：新人类第一局用轨迹先验把初始定位从 1500±200 收到
  μ_traj±(σ_traj(1)≈250–300)；紧随其后用 Thompson 探索相邻档；第 3 局起 info 稳定。

### 6.3 牌/座位调度

- 用**冻结牌堆 + 每副牌双座位**（审计 §6.1 的根因修复）：10 局 = 5 副牌 × 2 座位。
  每副牌人类在 0/1 号位各打一次，座位相位在牌内抵消；`ladder.plan_games` /
  `duel.plan_duel_schedule` 已实现这个 twin 结构（`ladder.py:92-148`、`duel.py:28-50`）。
- 取舍：10 局只见到 5 副牌，人类可能察觉重复；若更看重体验，用 10 副不同牌 + 座位
  5/5 轮换，代价是座位相位残差（审计实测跨序 sd≈√2×SE，10 局下不可忽略）。
- 定级局必须**冻结牌堆**并把对手 rung id 写进 trace（现在 `play.py` 的 `players`
  只写 `"贪心 bot"`/checkpoint 路径，不含 rung id；§8 需小改）。

## 7. 离线仿真结果（用真实 bot 轨迹当伪人类）

设置：每个等级 200 局真牌真结果（不是模型抽样）；每会话不放回抽 n 局；
估计器 = `fit_ratings`（胜负+分差 + 先验）；真值 = T2（200 局满数据估计）；
每格 150–300 个会话 × 6 个等级。`win` = 只胜负 + 1500±200；`traj` = 轨迹先验。

### 7.1 主表（vs T2）

表的前半段是**原始先验**（未去收缩），后半段是**去收缩先验 + 分差 σ×2** 的推荐组合。

| 方案 | n | RMSE | P(\|e\|≤50) | P(\|e\|≤100) | 名义 95% CI | 覆盖 |
|---|---|---|---|---|---|---|
| win（只有胜负+弱先验） | 5 | 158.6 | 0.22 | 0.39 | ±276 | 0.95 |
| win | 10 | 126.9 | 0.29 | 0.54 | ±222 | 0.94 |
| win+margin | 5 | 124.0 | 0.30 | 0.56 | ±192 | 0.88 |
| win+margin | 10 | 95.7 | 0.38 | 0.70 | ±144 | 0.86 |
| 原始 traj_group+win+margin（乐观） | 10 | 63.1 | 0.57 | 0.89 | ±110 | 0.92 |
| 原始 traj_lolo+win+margin（诚实） | 10 | 76.4 | 0.48 | 0.81 | ±127 | 0.90 |
| 原始 traj_lolo | 20 | 55.1 | 0.64 | 0.94 | ±97 | 0.93 |
| **推荐**：去收缩 traj_group + win + margin σ×2 | 10 | **54.3** | **0.65** | **0.93** | **±100** | 0.93 |
| **推荐**：去收缩 traj_lolo + win + margin σ×2 | 10 | **71.8** | **0.51** | **0.83** | **±133** | 0.94 |
| 推荐（grouped） | 20 | 43.4 | 0.75 | 0.98 | ±82 | 0.94 |
| 推荐（LOLO） | 20 | 57.3 | 0.62 | 0.92 | ±110 | 0.95 |
| win-only + 去收缩 traj_group（对照，n=50） | 50 | 29.5 | 0.90 | 1.00 | ±71 | 0.99 |
| win+margin（无轨迹先验） | 50 | 37.4 | 0.83 | 0.99 | ±68 | 0.93 |
| win（无轨迹先验） | 60 | 46.1 | 0.74 | 0.96 | ±102 | 0.98 |

**10 局结论**（vs T1 manifest 标签会再差 ~5–10 Elo）：
- 想要「点估计 RMSE ≤ 50」：grouped 去收缩 ~**14–15 局**，LOLO 去收缩 ~**26–28 局**。
- 想要「P(\|e\|≤50) ≥ 80%」：约 **40–50 局**。
- 想要「名义 95% CI 半宽 ≤ 50」：约 **63 局**（grouped 外推）/ **100 局**（LOLO 外推）；
  纯结果 win+margin 约 **91 局**——轨迹先验在早期把 CI 收窄，但到 50 分精度时结果
  似然已经主导，先验只省约 10–35% 局数。（由 `se²` 对 n 的线性拟合外推，
  锚点取推荐组合 n=10/20 的模拟值。）
- 10 局的 95% CI（推荐组合，LOLO）≈ **±133**，覆盖 0.94——是「诚实的宽区间 +
  点估计」；把它标成 ±50 是不诚实的。

### 7.2 逐等级偏差（LOLO 去收缩 prior 的等级均值 − T2）

| 等级 | T2 | manifest T1 | LOLO 去收缩后预测均值（范围） | 残余偏差 |
|---|---|---|---|---|
| random | 1026.9 | 1000.0 | 940 | −87 |
| lvl1 | 1128.6 | 1119.9 | 1172 | +43 |
| lvl2 | 1232.1 | 1232.9 | 1215 | −17 |
| greedy | 1314.4 | 1315.0 | 1426（原 1350） | +112 |
| lvl3 | 1356.1 | 1359.3 | 1378 | +22 |
| lvl4 | 1447.8 | 1489.4 | 1361（原 1293） | −87 |

去收缩把随机/强档的偏差砍半，但 greedy 反而被高估（它的风格特征像更强的模型）——
说明**风格与强度的残余混淆仍在**。这也是为什么真人 OOD 偏差必须用真人数据标定（§9）。

### 7.3 先验膨胀敏感性

把 `σ_traj` 乘 1.2/1.4/1.6：RMSE 只会变差（LOLO n=10：75.1 → 84.5），覆盖率几乎不动
（0.90 → 0.91）。**误差主要是先验均值的偏差，不是区间太窄**——加宽 CI 治不了；
要治就得改进去收缩/加特征/真人标定。

### 7.4 结果似然的两通道核对

- 分差通道在全样本的 Fisher 比只有 ~1.0–1.25（§5.3）。但在「neutral 先验、10 局」下，
  `win` RMSE 127 → `win+margin` 96：因为弱档/强档的 p(1−p) 很小（如 lvl4 vs random
  p=0.95），分差是异方差的、携带信息；这是分差的主要价值场景。
- **有轨迹先验后，分差与先验重复计数**（特征已编码结果信息）：保持 σ=44.5 时
  10/20 局覆盖掉到 0.88–0.90；σ×2 后覆盖回到 0.93–0.94，RMSE 还要好 ~1–4：
  LOLO n=10：σ×1 的 73.3/0.88 → σ×2 的 71.8/0.94；group n=10：57.5/0.88 → 54.3/0.93。
  只留胜负（去掉分差）则 LOLO 74.1/0.95、group 54.7/0.95。
- 结论：**保留分差，但在轨迹先验在场时按 σ×2 降权**；它值不了「把 m_eff 从 1 抬到 5」
  （约 1.25），但能在两端档位与长期 CI 上稳定出力。

## 8. 实现映射（D3 接缝）

**已经支持、不需要改**：

| 需求 | 现有接缝 |
|---|---|
| 轨迹先验 `(μ, σ)` | `elo.Prior`（`elo.py:45`）+ `fit_ratings(priors={...})`（`elo.py:224-232`），ADR-0006 明确此入口 |
| 胜负 + 分差似然 | `FitConfig.margin=(c, σ)`（`elo.py:99`）+ `_gradient_info`（`elo.py:161-186`） |
| 每局/窗口在线更新 | `fit_ratings(window=W)`（`elo.py:218-260`）；10 局内用 `None` |
| 诚实 SE（联合 Hessian） | `_inverse_diagonal`（`elo.py:188`）+ `fit_ratings:292-310`，自由-自由对局不再低估 |
| 梯级定义/最近档 | `select_rungs`（`elo.py:364`）；冻结标签在 `study.py` 的 manifest |
| 成对牌局/座位轮换 | `ladder.plan_games`（审计已修为候选内换座配对，`ladder.py:92-148`）、`duel.plan_duel_schedule`（`duel.py:28-50`） |
| 轨迹录制/回放校验 | `play.play_game(record=...)`（`play.py:272-350`）、`--save-trace`（`play.py:430-470`）、`trace.py` schema |
| S1 特征 | `tools/measure_trace_signal.extract_features`（`:300-405`），一行复用 |

**需要新增（不碰 `elo.py` 数学）**：

1. `tools/fit_trace_prior.py` + `artifacts/human-elo/prior.json`：把本文
   `calibrate.py`/`meta_calibration.py` 收敛成一个离线标定命令，产物 =
   特征列表、标准化参数、岭系数、去收缩 `(a,b)`、`σ_traj(n)` 表（默认 LOLO+去收缩）。
2. `src/seven523/placement.py`：会话状态 + `select_opponent`（info/Thompson）+
   冻结牌堆 twin 调度 + 停止规则 + 报告；纯编排，风格同 `ladder.py`。
3. `7g523-elo` CLI（`pyproject.toml [project.scripts]` + 薄壳）。
4. `play.py` 把对手等级 id 写进 `players` 标签（如 `opponent:lvl3@seat1`），
   让 `measure_trace_signal._opponent_info` 能读到 rung 评分；否则真人 trace
   无法按对手强度标定。
5. （可选）真人 trace 的 rung 元数据写入 manifest 或 sidecar，便于复算。

**测试**：`placement.select_opponent` 的单调性/Fisher 单测；`fit_trace_prior` 的
留一等级标定回归测试（锁定 tau/psd 表）；`7g523-elo` 的 CLI 冒烟；对
`elo.py` 现有测试零改动。

## 9. 验证计划与开放问题

### 9.1 真人验证（关键，必须先做）

1. 招募 N ≥ 8 人，覆盖 1000–1600；每人先走 **10 局定级会话**（冻结牌堆 + twin 座位 +
   自适应选档），再打 **60–100 局参考局**（同一牌堆、同一批梯级）作为 T2_human。
2. 一半真人标定 `(a,b)`/`σ_traj`，另一半评估：RMSE、P50/P100、CI 覆盖率、最近档命中；
   与本文 bot-LOLO 数字对比。**若真人 τ 远大于 bot τ，说明现有信号对人类的
   OOD 偏差不可省，必须真人标定或换信号。**
3. 可解释性回访：让玩家看 10 局报告（档位 ± CI + 关键决策），确认可接受。

### 9.2 开放问题

- **人类 OOD**：本文的 LOLO 只是「留出一个 bot 风格」的代理；真人风格偏差未知。
  缓解：真人标定集；或让真人先打 2 局「校准局」并对特征做域自适应。
- **非平稳**：真人边玩边学（前 10 局尤其明显）。缓解：先验用最近 3 局、结果窗口
  短；或显式建模趋势。本文的仿真假设能力恒定，这是乐观项。
- **梯级上限 ~1440–1489**（1489 为 manifest 旧相位值；复测合并 lvl4 = 1429.7 ± 7.1 已于 2026-09-25 refit 为 manifest 契约值，见 `ladder-rerating-paired.md` §4.4；勿混口径）：>1600 的真人信息量塌缩（§6.1）；需要训练更强的 bot
  或改用让子/限制 bot 的 handicap 匹配。
- **S2 还没真正被"做掉"**：本文只测了价值头 1 步 TD；真正的逐决策 regret
  （Q 值/短 rollout）需要新模块，且价值头必须先对真人标定。若要继续，建议先做
  「低等级 vs 高等级 bot」的 regret 校准实验（计划 §6.1）。
- **S3 只在 grouped 口径好看**：如果未来有真人数据，可重新评估「策略一致度」
  作为*风格*而非*强度*信号，用于选对手（同风格更稳）而不是评分。
- **模仿刷分**：任何把策略一致度/行为相似度直接计分的方案都可被模仿；结果似然
  始终是最终权威（ADR-0006 后果条款）。
- **标签相位**：manifest 的绝对 Elo 有 ±40 量级相位（审计 §5.4）；生产定级若要与
  天梯对接，先按审计 §6.1 的换座配对重测梯级，再冻结新 deck。

## 10. 结论（诚实版）

1. **≤10 局做不到「自身水平 ±50」。** 即使最乐观的 grouped 口径，推荐组合 10 局
   RMSE=54.3、P50=0.65、95% CI=±100；诚实的 LOLO 口径 RMSE=71.8、P50=0.51、CI=±133。
2. **10 局的实际可达精度**：点估计 RMSE ≈ **54–72**，±1 档（±100）命中 **83–93%**，
   同档（±50）命中 **51–65%**，带诚实的宽 CI 与最近档位。对「排个档、匹配对手」够用；
   对「精确到 50 分」不够。
3. **要 ±50**：点估计 RMSE≤50 需 **15–27 局**；95% CI 收进 ±50 需 **约 63–100 局**；
   纯结果口径需要更多（win+margin：RMSE≤50 约 33 局，CI±50 约 91 局）。
4. **最强信号是 S1 逐局特征 + 去收缩标定**（会话 m_eff 6.1 LOLO / 13.9 grouped）；
   S2 价值信号无增量且需要昂贵前向；S3 的 grouped 数字是风格泄漏，不可用于评分。
5. **推荐方案一句话**：冻结牌堆 + 双座位成对、S1 轨迹先验（岭回归 + 去收缩）+
   胜负与降权分差（σ×2）的 BT-MAP、按后验 Fisher 信息用 info/Thompson 自适应选档；
   10 局输出「点估计 + 诚实 CI + 最近档 + provisional 标记」，后台继续对局直到 CI ≤ 50。

## 11. 复现

```bash
cd /home/amas/.local/src/7g523
# 全部脚本在 runs/human_elo_10g/（gitignore），产物 JSON/CSV 同目录
.venv/bin/python runs/human_elo_10g/extract_features.py      # 6 级 × 200 局 S1 特征 + 5% 全 replay 校验
.venv/bin/python runs/human_elo_10g/calibrate.py             # 每级胜率/分差、grouped vs LOLO、m_eff、margin 标定
.venv/bin/python runs/human_elo_10g/s2_features.py           # S2/S3: 4 个 ckpt 价值头 TD + 策略一致度（~70s）
.venv/bin/python runs/human_elo_10g/calibrate_s2.py          # S2/S3 的 m_eff 与 S1+S2/S3 增量
.venv/bin/python runs/human_elo_10g/paired_probe.py          # 同牌参考面板/座位配对降噪检验（负面结果）
.venv/bin/python runs/human_elo_10g/simulate.py              # 1–10 局会话仿真（策略 × 方案，主表）
.venv/bin/python runs/human_elo_10g/extend_and_design.py     # n 扩到 60、求「达到 ±50 所需局数」
.venv/bin/python runs/human_elo_10g/design_study.py          # 对手选择 6 策略设计实验
.venv/bin/python runs/human_elo_10g/meta_calibration.py      # 去收缩标定（tau/psd 改善）
.venv/bin/python runs/human_elo_10g/prior_cal_check.py       # 去收缩 prior + 结果似然的最终数字
.venv/bin/python runs/human_elo_10g/inflation_check.py       # 先验膨胀敏感性
# σ×2 分差降权与 win-only 对照见 runs/human_elo_10g/winonly_curve.txt / winonly_results.json
```

产物：`features_all.csv`、`features_s2.csv`、`calibration.json`、`calibration_s2.json`、
`sim_results.json`、`extend_results.json`、`design_results.json`、`paired_probe.json`、
`prior_cal_results.json`、`inflation_results.json`、`winonly_results.json`。

**未改**：`src/`、`docs/experiments/` 已有报告、`traces/study/manifest.json`、
head-to-head 相关文件。

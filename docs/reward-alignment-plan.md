# 奖励-评估对齐（分差饱和）研究与预注册计划（revision-4，修复后）

> **角色**：修复/规划 agent。本阶段**零训练、零 h2h**；只读代码/只读数据（`numpy` 级 CPU 复算），
> 只写本文件与 `docs/reward-alignment-review.md`。未改 `src/`、`tests/`、`tools/`、`docs/adr/`、
> `docs/plans.md`、`docs/experiments/README.md`、`CONTEXT.md`、`DESIGN.md`、`RULES.md`；
> 未 commit/checkout/stash；未用 worktree。
>
> **修订说明**：本文件是 revision-3 的**修复版**（红队 41 条 findings，处置逐条见
> `docs/reward-alignment-review.md`）。关键修复：(1) 删除 M3 的「Elo 上界」门与跨候选斜率外推，
> 决策不再依赖该混杂量；(2) M2 降为**描述性**（200 局/anchor 无法归因，`stats-eval-01/03/07`）；
> (3) 路由收敛为**单张显式决策表**，且「收束」明确定义为**资源决策、不是「干预无效」的证据**；
> (4) 主端点重排为 warm-start 直测，from-random 臂降级（`rl-reward-07`）；
> (5) `saturate`/`--reward-cap` 语义、训练命令、跨 seed 合并、基线快照全部写死。
>
> **口径标签（硬性）**：全部设计与引用数字属 **revision-3（出空即撬底，`rules_id=2e36dbea44893696`，
> `src/seven523/rules.py:53`）**、obs v5（2 家 161 维）、纯 MLP（`arch=shared`、`hidden=128`、
> 单层 trunk）。跨 fit 绝对 Elo 不可比；跨规则版本（revision-2）数字只作历史。
>
> **上游状态**：`docs/selfplay-pool-plan.md`（另一 workflow，可能正在跑）占用 GPU。
> **本 workflow 的 Diagnose 必须纯离线 CPU**；训练实验只做设计、**不在本 workflow 执行**。
>
> **判定口径**：`docs/experiments/README.md` §3 / `evaluation-protocol-validation.md` §9：
> 任何「A 优于 B」须 ≥3 seed × 400 副（总 ≥1200）换座 h2h，报
> `mean ± 1.96·max(bootstrap SE, seed sd)/√k`；**CI 完全排除 0 且点估计 ≥ +10 才谈「值得行动」**，
> **< +5 止损**，<10 一律不追（80% power 需 ≥3200 副）。

---

## 0. 结论先行

### 0.1 用户提出的假设

> 「和 RandomBot 对战训练时，后期 episode return 仍在升高，但评估 Elo 不变。训练奖励（分差）
> 与评估（胜负）在‘已经确保胜利’之后错位——是否应把奖励在超过 50 分（或等价地‘胜负已定’）
> 后饱和/截断？」

### 0.2 事实核实（只读，已逐条核对；`file:line`）

| # | 事实 | 证据 |
|---|---|---|
| F1 | 训练奖励默认 `terminal`：非终局步 0，终局步 = `Game.returns()[learner]` | `env.py:41`、`:356`、`:358-361`；`train.py:120-122`、`:405`、`:424`、`:606` |
| F2 | `returns() = own/100 − mean(others)/100`（2 家即 `(own−other)/100`，**分差**） | `game.py:346-358` |
| F3 | 评估只用**胜负/平**：`seat_outcome = sign(own − max(others)) ∈ {−1,0,+1}` | `game.py:361-365`；`env.py:334-335` |
| F4 | 评分通道是 OpenSkill PL（ADR-0011）与 homoscedastic ordered-probit MLE（ADR-0013），似然只吃 outcome，不吃分差 | `docs/adr/0011`、`0013`；`runs/t17_screen/mle.json` |
| F5 | 2 家、`total_points=100`、终局分数守恒（不守恒直接 `AssertionError`）⇒ `own>50 ⇔ 必胜`、`own=50 ⇔ 平`、`own<50 ⇔ 必负` | `rules.py:24`；`game.py:352-353` |
| F6 | 现有奖励模式只有 `terminal/trick_diff/win/trick_diff_win` | `env.py:41`；`train.py:120-121` |
| F7 | 旧奖励实验（revision-2、**greedy 对手、热启动 `base_step00696320.pt`、A2 只 1 seed**）：A1 `trick_diff` seed1 四点 +9.99…+12.75，4-seed 未复现（平均 +4.47/+6.77）；A2 `win` null；γ=1 不优；D0/D2 critic 判别不干净 | `docs/experiments/reward-shaping-500k.md` §0/§1.2(`:42`/`:48`)/§2.2(`:98-101`)/§2.4/§3.4/§7 |
| F8 | 顶部平台不可分：512k–2M 五点 μ=187.60…191.61、σ≈5.26、CI 宽约 ±13 | `docs/experiments/t17-recalibration.md` §1；`runs/t17_screen/mle.json` |
| F9 | 训练 `eval_interval=0`，训练中**没有**同 run 在线 Elo；对手 `random`、冷启动、seed 1/2/3 | `runs/t17pool__*/args.json`、`runs/t17long__1__1790439615/args.json` |

### 0.3 本轮只读复算（预核实；命令见附录 A，未落盘）

**所有「候选」统计的样本量固定为 n=200 局/anchor 候选（100 副共享牌 × 2 座位，`games.jsonl`
`kind=anchor & opponent=random`，13 个候选 seed 集完全相同）。单候选 R 的 SE≈0.025；
配对的相邻平台 ΔR 的 95% CI 半宽 0.027–0.053。以下任何 |Δ|<0.05 的量在 n=200 上都判不了。**

- **后期 return 确实仍在涨，但幅度很小，且只有 1 个训练 seed**：`runs/t17long__1__1790439615/metrics.csv`
  在 `[512k, 2M]` 上 OLS 斜率 **+0.0182 / 1M 步**，moving-block(block=25) 95% CI
  **[+0.0106, +0.0244]**；窗口均值 `512–768k = 0.5162` vs `1.5–2M = 0.5360`，差
  **+0.0198**，iid-bootstrap CI **[+0.0109, +0.0290]**。折算成分数为 **≈ +1.0 个 own 分**。
  自相关实测≈0（lag1=−0.027、lag25=+0.012），Newey-West(25) SE 0.00354 ≈ iid SE 0.00355
  ⇒ 噪声主要来自「逐点方差大（sd≈0.06）+ 信号小」，不是自相关（修复 R2）。
- **同期 Elo 平**：这四点 screen μ = 187.60/191.61/187.69/191.11，极差 4.0，位于 ±13 的 CI 内（F8）。
- **归因未判定（不是「倾向成立」）**：顶部平台四点 `R` = 0.5535/0.5280/0.5500/0.5625（**非单调**），
  win% = 90.0/86.5/90.0/89.5（**非单调**），`E[r_term|w]` = 0.6306/0.6376/0.6311/0.6447。
  按 seed 配对 bootstrap（B=4000, `default_rng(0)`）：
  - 相邻 ΔR：512k→1M **−0.0255 [−0.0840,+0.0315]**、1M→1.5M **+0.0220 [−0.0345,+0.0800]**、
    1.5M→2M **+0.0125 [−0.0310,+0.0560]**——全部含 0、符号交替。
  - 512k→2M：ΔR **+0.0090 [−0.0425,+0.0610]**；ΔR_sign **0.0000 [−0.0850,+0.0800]**；
    加法分解 Δ(P_w·E[r_term|w]) **+0.0095 [−0.0375,+0.0570]**、Δ(P_l·E[r_term|l]) **−0.0005 [−0.0110,+0.0085]**。
  - **所有分量 CI 含 0 ⇒ 归因不可判定**。revision-3 的 §0.3「涨幅几乎全部来自已锁定胜利后的
    比分扩大」是 200 局上的点估计，**已撤销**。
- **分数→Elo 的映射在顶部断裂，且不能当上界**：13 候选整体 OLS 斜率 **308 ± 15.9 Elo / 单位 R**
  （由早期 16k–262k 与 random 锚点主导的**训练进度混杂**）；顶部 5 点斜率 **−4.6 ± 43.5**，
  95% CI **[−89.9, +80.7]**。用 ΔR=0.0198 折算：
  - 窗口均值差 → **+6.0 Elo**；OLS 斜率积分（0.0182×1.488M）→ **+8.4**；
    centered rolling-25 端点（0.5112→0.5566，差 +0.0454）→ **+14.0**。
  - 这三个估计跨越 §4 行动门槛 +10，且它们都不是因果量、也不是上界。
  - **本文件不再用该外推做任何门**（修复 `stats-eval-02/03`、`rl-reward-01/02`）。
- **13 候选 training return 与 μ 的相关主要由训练步数混杂**：Pearson 0.966、控制 `log(step)`
  后偏相关降到 **≈0.50–0.62**；顶部 4 点内相关仅 **0.47**、晚期 6 点仅 **0.33**。
  且训练 return 与 screen `R` 口径相差最大 **0.07**（如 `p1_499k` 0.526 vs 0.597）——**不可混用**。
- **结果无关方差份额（描述量，非门）**：`within/total` = 0.325(e1_2k)/0.466(p1_262k)/
  0.557(l1_512k)/0.478(l1_1M)/0.527(l1_1.5M)/0.558(l1_2M)/0.524(p1_499k)。原阈值 0.30
  连 2k 步都通过，**无区分度**（修复 `stats-eval-05/12`）。
- **跨 seed 变异与跨 fit 一样大**：`runs/t17pool__{1,2,3}` 在 450–500k 的 return 均值
  0.5213/0.5425/0.5286（seed sd≈0.011），与整个后期 drift(+0.0195) 同量级。
- **`norm_adv` 下饱和的真实机制**（`ppo.py:210-214` 逐 minibatch 减均值除 std，minibatch=256）：
  用观测 outcome 混合模拟，terminal 的胜局归一化 advantage **均值 +0.27 / 标准差 0.60**；
  K0(τ=0) 的胜局 **均值 +0.25 / 标准差 0.00**。即饱和**去掉胜局内的分差散布**，
  但保留一个被重新中心的胜/负水平——**不是**「胜局 advantage≈0、梯度全去损失/平局」（修复 `rl-reward-04/05`）。

### 0.4 判定与建议（供 owner 决策）

- **「错位」作为拓扑事实成立**：终局奖励含一个对评估**完全无意义**的「已锁定胜利后的分差」
  分量（F2/F3/F5）；这是用户问题的精确来源，无论经验效应大小。
- **「materiality（是否值得动）」用现有数据判不了**：后期增量仅 ≈ +1 个 own 分（n=1 seed）；
  归因在 n=200 上 CI 全部含 0；分数→Elo 外推跨 +10 门槛、且是混杂量。
  **本计划不做「不 material」的判断**，只报告「症状存在（seed 1）、归因未判定、效应量在噪声内」。
- **默认路径（Tier 0）= 离线诊断 + 由 owner 作资源决策收束**：执行 §2 的离线诊断，产出
  `docs/experiments/reward-alignment-diagnostic.md`。**收束是「不再为这条线排 GPU 预算」的
  资源决定，而不是「实验会无效」的证据**（§2.6 明确措辞，修复 `rl-reward-01`）。
- **可选路径（Tier 1）= 由 owner 触发的直接反事实实验**：只有在 (a) owner 想拿直接证据，
  (b) §4.1 的 **μ-headroom 预检通过**（见下）时才按 §4 执行。
  - **headroom 预检（新，硬前置）**：主端点必须在**能分辨奖励变化**的对手上测。from-random
    训练对 random 的胜率已停在 0.865–0.900，评估对这个区间不敏感（顶部 5 点 μ 差 ≤4.0）。
    若不能证明奖励变化可能把 vs-random 胜率推离该区间，则**只执行 warm-start 块**（§4.2），
    并把它作为唯一主端点。
- **关键提醒（R1，保留并加强）**：诊断检验的是「**可观测症状**」（后期 return vs Elo）；
  §4 检验的是「**换奖励能否改变优化轨迹/评估**」。两者不是同一命题，症状小不等于干预无效。

---

## 1. 背景事实的完整核实

### 1.1 训练奖励 = 分差（不是胜负）

`Seven523Env._reward`（`env.py:356`）在 `terminal` 模式下，非终局步返回 `0.0`，终局步返回
`self.game.returns(state)[self.learner]`（`env.py:358-361`）。`Game.returns` 定义为

```
own / total_points − (total − own) / ((n−1) * total_points)
```

（`game.py:355-357`），2 家即 `(own − other)/100`。RULES.md 的奖励口径为 RL-2「期望分差」
（`RULES.md:72`）。`env.py:41` 声明四个模式；`train.py:120-122/405/424/606` 逐层透传；
默认 `terminal`。

### 1.2 评估 = 胜负/平（不吃分差）

`seat_outcome(scores, seat) = int(own > best_other) − int(own < best_other)`
（`game.py:361-365`），其中 `best_other = max(other seats)`。这是仓库唯一定义（ADR-0009 之后
单一实现）。`env.step` 把 `scores` 与 `outcome` 放进 `info`（`env.py:334-335`），训练期的
PFSP/统计只用 `outcome`（`train.py:820-833`）。评估链：`duel`/`ladder` 用换座配对与按牌聚簇
bootstrap（`tools/head_to_head.py` docstring、`src/seven523/ladder.py`），发布绝对表用
ordered-probit MLE（ADR-0013，`runs/t17_screen/mle.json` 的 `estimator.kind = "probit-mle"`）。
**似然只吃 outcome；margin 不进入评分。**

### 1.3 2 家几何：`own>50 ⇔ 必胜`

`Rules.total_points = 100`（`rules.py:24`）；终局 `sum(scores) == total_points`，否则
`AssertionError`（`game.py:352-353`）。2 家时 `other = 100 − own`，故
`own > 50 ⇔ own > other`、`own = 50 ⇔ 平`、`own < 50 ⇔ 负`。**「超过 50 分后继续涨的分」对
`seat_outcome` 与任何 outcome-only 评分完全无信息。**

> **N 家注意**：`own>50` 只是 2 家的充分条件。3 家时 `own` 可与 `max(others)` 相对大小
> 和 50 无固定关系（例：40/30/30 时 `own=40<50` 但胜）。N 家的「胜负已定」必须以
> `own vs max(others)` 定义（见 §3.3），不能用 50 分阈值。

### 1.4 旧实验的边界（revision-2 + greedy + 热启动，不能与当前口径混比）

`docs/experiments/reward-shaping-500k.md`：

- **域声明（新）**：A1/A2 的训练命令为 `--opponent greedy --load-checkpoint
  runs/probe/base_step00696320.pt`（`:42`/`:48`），是 **revision-2、固定 GreedyBot、热启动**；
  A2 只训 **1 个 seed**。§4 拟议的臂是 **revision-3、`--opponent random`、冷启动、500k**。
  两者在规则版本、对手分布、初始化三个维度都不同。
- **A1 `trick_diff`**（逐墩 telescoping 分差）：seed 1 的 4 个 h2h 点估计 +9.99…+12.75 且 CI
  排 0，但补训 seed 2/3/4 **未复现**（4-seed 平均 +4.47 / +6.77）。按 <10 不追口径，
  **A1 独立杠杆不成立**。
- **A2 `win`（纯胜负）**：3×400 对 `base680k` −7.53 [−18.93,+3.86]、对 `w5_ctrl`
  +8.54 [−3.27,+20.36]，**null**；且只训 1 个 seed。
- γ=1 不优；D0/D2 critic 判别 H1/H0 均不被干净支持。
- 该报告的 h2h 母版参照 checkpoint 已删除，路径不可用；新实验的参照必须用 T17 资产。

**结论**：A2 的 null 是**另一个域的历史 null**，**不能**为当前域（revision-3/random/冷启动）
的 `K_τ` 提供效应上界；本文件删除「两端点夹住 K_τ」的措辞（修复 `rl-reward-08`）。

### 1.5 顶部平台与训练配置

T17 重标定（`docs/experiments/t17-recalibration.md` §1）显示顶部 5 点（`l1_512k`、`p1_499k`、
`l1_1.5M`、`l1_1M`、`l1_2M`）μ=187.60–191.61、CI 两两重叠。所有 T17 训练 run：
`num_envs=8 × num_steps=128 = 1024` steps/update、`num_minibatches=4`、`lr=2.5e-4` 线性衰减、
`arch=shared`、`hidden=128`、`gamma=0.99`、`gae_lambda=0.95`、`norm_adv=True`、`vf_coef=0.5`、
`ent_coef=0.01`、`--opponent random`、`eval_interval=0`、`total_timesteps=500000`
（实际 488 update = 499,712 步，`args.json`；`num_updates=total//batch_size`，`train.py:677`）。
**没有同 run 在线 Elo**，因此任何「return vs Elo」对齐都只能是跨 run/跨 fit 的关联（F9）。

### 1.6 `episodic_return` 的语义与噪声（实测，不假设）

`train.py:709-710` 每个 update 清空 `ep_returns`；`:812` 对每个完成的 env 追加终局回报；
`:950` 取该 update 内完成 episode 的**算术平均**。每个 update 约完成
`num_envs*num_steps/episodic_length ≈ 1024/24 ≈ 43` 局。因此 `metrics.csv` 的
`episodic_return` 是**逐 update、约 40–80 局**的均值，逐点噪声 sd ≈ 0.06（实测
t17long 后期 sd=0.059–0.062）。

**实测序列近似白噪声**（t17long `[512k,2M]` 去趋势后）：acf = −0.027(lag1)、−0.029(5)、
−0.021(10)、+0.012(25)、−0.018(50)；Newey-West SE(25)=0.00354 与 iid SE=0.00355 几乎相同。
⇒ **自相关不是主要噪声来源；主要来源是「逐点方差大 + 信号本身小」**。block bootstrap 仍保留，
但理由改为「对未知自相关的稳健性」，且 block 仅在实测 acf(lag1..lag25)>0.2 时才需要
（修复 R2 与 `rl-reward-09`）。

---

## 2. 现象核实设计（离线，零训练）

> 目标：把「后期 return 涨而 Elo 平」拆成**可判定的事实**，并明确**哪些量在 n=200 上判不了**。
> 全部输入只读；不训练、不 h2h、不跑 GPU。

### 2.1 数据源与 P0（配对由构造保证）

| 数据 | 路径 | 用途 |
|---|---|---|
| 训练曲线 | `runs/t17pool__{1,2,3}__1790439615/metrics.csv`、`runs/t17long__1__1790439615/metrics.csv`、`runs/t17early{2k,4k,8k}__1__17904396*/metrics.csv` | D1：后期 return 趋势 |
| 逐局评估 | `runs/t17_screen/games.jsonl`（18200 行；13 候选 × 200 局 vs random + C(13,2)×200 cross） | D2/D3/D4 |
| 共享 fit | `runs/t17_screen/mle.json`（`levels`/`sigmas`/`ci`/`n`/`estimator`） | D3 |

**P0（配对 gate）——按构造成立，不再「先实测」**：
- 发牌只由 `(deal_seed, rules)` 决定：`play.py:364` 用 `rng=random.Random(seed)` 构造
  `Match`，`game.py:140-147` 在 `_deal` 里 `rng.shuffle(deck)`；策略消耗的是**独立**种子
  `policy_seed(seed, seat) = (seed + 101·(seat+1))`（`record.py:47-54`，在 `ladder.py:396`
  调用）。因此同一 `deal_seed` 在不同候选下一定是同一副牌。
- `ladder.plan_games` 对每个 anchor index 只抽一次 `deal_seed = master.randrange(1<<32)`
  并在全部候选 × 2 座上复用（`ladder.py:239-249`）。
- **实测确认**：13 个候选的 anchor seed 集完全相同（100 seed/候选，每个出现 2 次）。
- **结论**：anchor 局可做**按 seed 配对**（CRN），**不得**退化为非配对——配对实测降低
  ΔR 的 SE（512k→2M：配对 SE=0.0271 vs 非配对 SE=0.036，修复 `stats-eval-09` 的错误子结论）。
- cross 局使用另一套 deal_seed（`ladder.py:250-259`），不作配对。
- **不再要求读 trace**（`runs/t17_screen` 用 `--no-traces`，无 trace 可读），修复 `engineering-impl-04`。

**P0 是唯一 gate 且已通过；它不花时间。真正的限制是「200 局/候选判不了 |Δ|<0.05」。**

### 2.2 D1：后期 return 趋势（within-run，估计量写死）

对每个 run：

1. 取 `episodic_return` 对 `global_step` 的序列。
2. **主估计量**：在预注册窗口 `[512k, 2M]`（t17long）上 OLS 斜率，单位 `per 1M steps`。
3. **主 CI**：moving-block bootstrap，`block=25`、随机起点均匀取 `[0, n−block]`、
   `B=2000`、`rng=np.random.default_rng(0)`、取 2.5/97.5 百分位。预注册敏感性
   `block ∈ {10,25,50}`，并同时报 Newey-West SE（lag ∈ {1,5,25,50}）与实测 acf。
4. **次估计量（描述）**：非重叠窗口均值差 `[512k,768k]` vs `[1.5M,2M]`，iid-bootstrap CI。
5. 把 Δreturn 折算成分数 `Δown = 50·Δreturn`。
6. **D1 判定只用主估计量的 CI**：`SYMPTOM := CI 排除 0 且斜率 > 0`。

**输出**：`runs/reward_alignment/d1_training_return_trend.csv` 与图 `d1_trend.svg`（best-effort）。

### 2.3 D2：增量分解（**描述性**；加法分解，禁用比值）

单位统一（修复 `rl-reward-11`，与 §3.1 完全一致）：**`d = own − other`（原始分数差，整数）**；
本节以及 §2.5/§A.2/§A.3 的分解量一律写成**回报尺度** `r_term = d/100 = (own − other)/100 ∈ [−1,1]`。
对候选 c（anchor 局）：

- `P_w(c), P_t(c), P_l(c)`：胜/平/负频率；
- `E[r_term|w](c), E[r_term|l](c)`；
- 加法分解（**这是唯一允许的归因口径**）：
  ```
  R(c)       = P_w(c)·E[r_term|w](c) + P_l(c)·E[r_term|l](c)
  R_sign(c)  = P_w(c) − P_l(c)
  ```
  令 `W(c) = P_w·E[r_term|w] ≥ 0`（胜局贡献）、`L(c) = P_l·E[r_term|l] ≤ 0`（负局贡献），则 `R = W + L`。
  对顶部平台相邻 step `(a→b)` 用**按 seed 配对 bootstrap**（B=4000，`default_rng(0)`）报：
  `ΔR`、`ΔR_sign`、`ΔW`、`ΔL`，各自点估计 + 95% CI。
- **删除**原「结果无关份额 = 1 − ΔR_sign/ΔR」比值式：`ΔR≈0` 时发散，实测给出负值/爆炸值
  （`stats-eval-07`）。**不接受任何比值式归因。**
- **删除**「涨幅几乎全部来自已锁定胜利后的比分扩大」的点归因。

**M2 判据（重定义为纯描述性，修复 `stats-eval-01/03/07`、`rl-reward-03`）**：

| 布尔 | 定义 | 说明 |
|---|---|---|
| `SYMPTOM` | D1 主估计量 95% CI 排除 0 且 >0 | 后期 return 在涨 |
| `MARGIN_ATTR` | `ΔW` 的 95% CI 排除 0 且 `ΔW>0` | 胜局贡献显著上升 |
| `LOSS_ATTR` | `ΔL` 的 95% CI 排除 0 且 `ΔL>0`（即负局变轻） | 负局贡献显著上升 |
| `ATTR_RESOLVED` | `MARGIN_ATTR` 或 `LOSS_ATTR` 为真 | 归因至少有一侧被分辨 |

**在 n=200 上：`MARGIN_ATTR = MARGIN_ATTR_vs_LOSS` 无法分辨，`ATTR_RESOLVED = False`
（实测 ΔW CI 含 0、ΔL CI 含 0）。报告只能写「归因未判定」，不得写「由…主导」。**
若要真正判定归因，须把 anchor 局提到 **≥1200 局/候选**（新成本项，须 owner 批准）——
本 Diagnose 不生成新对局，故**默认记 `ATTR_RESOLVED=False/未判定`**。

**输出**：`runs/reward_alignment/d2_return_decomposition.csv` 与图 `d2_decomposition.svg`。

### 2.4 D3：跨 run/跨 fit 对齐（描述，非因果；必须报偏相关与口径偏移）

把每个候选的训练 `episodic_return`（在该 checkpoint step 处的 rolling-25 均值，**窗口定义写死为
以该 step 为中心的 25 个 update**）与 screen μ 配对：

- **报偏相关**：`corr(return, μ)`、`corr(return, μ | log step)`、以及**顶部 4 点内**与
  **晚期 6 点内**的相关。实测：Pearson 0.966→偏相关 ≈0.50–0.62；顶部 4 点内 0.47；
  晚期 6 点内 0.33 ⇒ **整体相关主要由训练步数混杂**（修复 `stats-eval-06`）。
- **报口径偏移表**：逐 ckpt `return_centered` vs `R`。实测 t17long：512k 0.511/0.553、
  1M 0.528/0.528、1.5M 0.530/0.550、2M 0.557/0.562；t17pool：262k 0.4975/0.562、
  499k 0.526/0.597（**最大偏移 0.07**）。
- **明确写出**：training return 与 screen `R` 口径不同（不同对手实例、不同局数、不同 seed），
  **不得跨口径乘除**（修复 `stats-eval-06`）。
- 局限（保留）：跨 run（early/pool/long 各自 `anneal_lr` 不同）、跨 fit（μ CI ±13，p2/p3 不在
  该 fit 内）、非因果。

**输出**：`runs/reward_alignment/d3_alignment.csv`。

### 2.5 D4：方差份额（**描述量，不再是门**）

对顶部候选的逐局**回报** `r_term = d/100`（`d = own − other`，定义见 §2.3/§3.1）按 outcome 分层：

```
Var(r_term) = Var(E[r_term|o]) + E[Var(r_term|o)],   o ∈ {胜,平,负}
```

报告 `within/total` 份额。**明确声明它不是梯度占比**。实测 0.325–0.558。
**删除原 M4 阈值 0.30**：它连 2k 步的 e1_2k(0.325) 都通过，无区分度（修复 `stats-eval-05/12`）。
D4 只在报告里作为「分差散布还有多少」的描述，**不参与任何路由**。

**输出**：`runs/reward_alignment/d4_variance.csv`。

### 2.6 预注册判据与路由（**单张显式决策表**）

**判据（全部可判定；`ATTR_RESOLVED` 默认为 False）：**

| 布尔 | 定义 | 现状（§0.3） |
|---|---|---|
| `SYMPTOM` | `Δreturn` 主估计量 95% CI 排除 0 且 >0 | **True**（seed 1） |
| `ATTR_RESOLVED` | `MARGIN_ATTR` 或 `LOSS_ATTR` | **False（未判定）** |
| `PLATEAU_FLAT` | 顶部平台 `Δμ` 的 CI 覆盖 0 | **True**（±13） |

**路由（唯一表；不存在第二处阈值）：**

| 条件 | 走向 | 措辞义务 |
|---|---|---|
| `¬SYMPTOM` | 直接收束 | 写「后期 return 仍升」前提不成立 |
| `SYMPTOM ∧ ¬ATTR_RESOLVED` | 默认 Tier 0（**资源收束**）；owner 可显式触发 Tier 1 | **必须写「归因未判定；收束是资源决定，不是干预无效的证据」** |
| `SYMPTOM ∧ ATTR_RESOLVED ∧ PLATEAU_FLAT` | owner 决定 Tier 0 或 Tier 1 | 必须写「已分辨归因方向：…；但仍不能由症状推干预价值」 |

**删除**：原 §2.6 的 `M3 materiality` 行与 `M1∧M2∧¬M3` 读法；原 §0.4 的
「Elo 上界 ≥ +10 的一半」触发条件（该量既非因果也非上界，且与 §2.6 冲突，修复
`stats-eval-02/03`、`engineering-impl-01`、`rl-reward-01/02`）。**Elo 外推只在 §0.3 作为
「量级参考（非因果、非上界）」报告，不进门、不路由。**

**Tier 1 的触发**：由 owner 决定，且必须先过 §4.1 的 μ-headroom 预检。

### 2.7 需要的图/表

| 产物 | 内容 | 备注 |
|---|---|---|
| `d1_trend.svg` | t17long（及 pool）`episodic_return` vs step，叠加 rolling-25、OLS 线、窗口均值 | **best-effort**：缺 matplotlib 则 CSV+ASCII |
| `d2_decomposition.svg` | 顶部 5 点的 `P_w/P_t/P_l`、`E[r_term|w]`、`R`、`R_sign`；相邻 step 的加法 Δ 分解 + CI | best-effort |
| `d3_alignment.svg` | 13 候选 `return_centered` vs μ 散点（按 early/plateau 分色，标注口径偏移） | best-effort |
| `d4_variance.csv` | between/within 方差份额 | |
| `verdict.json` | `SYMPTOM/ATTR_RESOLVED/PLATEAU_FLAT` 布尔 + 数值 + 所有 Δ 的 CI + 输入文件 sha256 + 命令 + `figures: done/skipped` | |

### 2.8 因果性声明（写进报告）

- D1–D4 全部是**关联/描述**。不得写「分差奖励导致 Elo 平台」「饱和奖励会 +X Elo」。
- 顶部 decoupling 可由评估分辨率（±13 CI）、容量、LR 衰减、样本效率等解释；诊断只能**排除**
  「后期由胜率提升驱动」这一竞争解释（且前提是 `ATTR_RESOLVED`），不能**确立**奖励口径是瓶颈。

---

## 3. 奖励变体设计（只设计，不实现）

### 3.1 统一记号与 2 家精确定义

**全文只有一个 `d`**：`d = own − other`（**原始分数差，整数**）；
**全文只有一个回报尺度**：`r_term = (own−50)/50 = (own−other)/100 = d/100 ∈ [−1,1]`。
统计量（§2.3/§2.5/§A.2/§A.3）一律写成 `r_term`，**不再用 `d` 表示 return 单位**。
**τ 一律以 return 单位声明**；`τ=0.2 ⇔ d=20`、`τ=0.7 ⇔ d=70`（修复 `rl-reward-11`）。

| 模式 | 定义（终局步；`own>50 ⇔ 胜`） | 非终局步 | 与评估关系 | 现状 |
|---|---|---|---|---|
| `terminal`（对照 C0） | `r_term` | 0 | 分差 | 已有 |
| `win`（W） | `sign(d)` | 0 | **完全对齐** | 已有（A2 null，另一域） |
| `saturate τ`（新，K_τ） | `own<50`: `r_term`；`own=50`: `0`；`own>50`: `min(r_term, τ)` | 0 | 胜局饱和；保留负分差与「胜>平」 | **无** |

**参数 τ 取值**：顶部候选胜局 `r_term` 的分位数（复算）：`q10=0.2, q25=0.4, q50=0.7,
q75=0.85, q90=1.0`。分位数由每候选 173–180 个胜局给出（n 小），报告须给分位数 CI。

- **`K2`（τ=0.2）**：饱和 ~90% 胜局（q10），保留「胜>平」与贴近 50 分的边际信号。**唯一主臂。**
- **`K0`（τ=0）**：`own>50` 一律 0 ⇒ 胜局与平局同回报。**不是 outcome-aligned**（`rl-reward-04`），
  **仅作 secondary 描述臂**（「用户原话的字面读法」），**不参与收束/判定**。
- **`K7`（τ=0.7）**：饱和 ~50% 胜局（中位）。**与 K2 互补的剂量臂**（K0 与 K2 只在最低 10%
  胜局上不同，近乎共线；修复 `stats-eval-11`）。

### 3.2 与现有 `win` / `trick_diff` 的关系

- `K_τ` **不是** `win`：保留了负侧分差（输 0:100 与输 45:55 不同回报）与「胜>平」。
- `K_τ` **不是** `trick_diff`：`trick_diff` 改变信用分配（摊到每墩），`K_τ` 是**纯终局**、
  只改终局回报的幅度映射。
- `trick_diff_win` = `trick_diff` + `win` bonus；**没有**「饱和的 `trick_diff`」模式。

### 3.3 N 家推广（出空即撬底、`num_players=n`）

用 outcome 条件式（**不要用 50 分阈值**）：

```
m = max(scores[other])                    # best other
r_term = own/total − mean(others)/total   # 与 game.returns 一致
r = r_term                                  if own <  m   (负)
  = 0                                       if own == m   (平)
  = min(r_term, τ)                          if own >  m   (胜)
```

N>2 时 `r_term > 0` 不等价于「胜」（例：35/35/30 时 `r_term>0` 但 `own==m` 是平），
必须按 `max(others)` 判定 outcome。2 家时退化为 §3.1。当前评估链只支持 2 家（`ladder`、`elo`、
`duel` 硬编码）；N 家实现属 T7，不阻塞本实验。

### 3.4 对 value / `norm_adv` / GAE / entropy 的影响（**机制重述**）

- **回报尺度**：2 家 `terminal` ∈ [−1,+1]；`K0` ∈ [−1,0]；`K2` ∈ [−1,0.2]；`K7` ∈ [−1,0.7]。
- **`norm_adv=True`（`ppo.py:210-214`）**：按 minibatch（256 样本）减均值除 std。
  **修正后的机制**（实测，`stats-eval-05`/`rl-reward-05`）：
  - 饱和**不去掉**胜/负的（重新中心后的）水平 advantage：归一化后胜负仍有显著正 advantage；
  - 饱和**去掉的是胜局内部的分差散布**（terminal 胜局归一化 advantage sd≈0.60 → K0/K2 ≈0.00）；
  - 结果是**相对**下调分差梯度、相对上调胜负/损失梯度（损失:胜的归一化幅度比 ~10→~12.5）。
  - 因此 saturation 的**真实杠杆小于**「原始回报差」给人的印象，因为归一化吸收了部分水平变化。
- **value 尺度与 `vf_coef`**：value loss 是未归一化 returns 的 MSE（`ppo.py:216-231`），
  pg_loss 吃归一化 advantage。缩小回报尺度 ⇒ 降低 critic 相对学习权重。
  **首波不改 `vf_coef`/`ent_coef`**；`explained_variance`（`ppo.py:251`，分母是**本臂自己的**
  `Var(y_true)`）**不得跨臂比较**；`clip_vloss`（`clip_coef=0.1`）与 `max_grad_norm=0.5` 都是
  绝对回报尺度常数，绑定比例会随尺度变——**报告必须同时给这两个绑定比例**（修复 `rl-reward-08`）。
- **GAE（`ppo.py:118-153`）**：公式不变；终局稀疏奖励下 `return_t = γ^{T−t} r_T`，饱和只改
  `r_T` 幅度。**机制端点必须在归一化后计算**，并同时报归一化前后（修复 `rl-reward-05`）。
- **entropy**：`ent_coef=0.01` 固定（`train.py:116`）。回报尺度缩小会**相对放大** entropy 项。
  必须把 entropy 轨迹作为机制端点。
- **V-collapse 竞争预测与止损（新，修复 `rl-reward-06`）**：饱和降低终局回报方差 ⇒ 可能把
  critic 推向 A1 的塌缩区。历史探针（`reward-shaping-500k.md` §2.2，revision-2/greedy/热启动）：
  终局回报 sd ≈0.43；`base680k` V sd=0.302、A1 V sd=0.113（V-vs-R EV≈0、corr 0.24）、
  A2 V sd=0.495。**预注册预测**：V sd 与 V-vs-R EV 随饱和**下降**；**止损规则**：若某臂
  训练后期 V sd 落入 **≤0.15**（A1 塌缩带）且 V-vs-R EV ≤0，则该臂记为 **critic collapse**，
  **不**记为「饱和的奖励结论」。该规则与 h2h 主端点并列报告。

### 3.5 实现触点与 CLI 语义（写死；本 workflow 不改代码）

- `env.py:41` `_REWARD_SHAPING_MODES` 增加 `"saturate"`；`Seven523Env.__init__`（`env.py:268`）
  增加 `reward_cap: float | None = None`；`_reward`（`env.py:356`）加分支（复用
  `_win_bonus`/`seat_outcome`）。
- **CLI spec（修复 `engineering-impl-05`）**：
  - `train.py` 的 `--reward-shaping` choices 增加 `"saturate"`；新增 `--reward-cap`。
  - **约束（实现时必须 raise）**：`reward_shaping == "saturate"` 时 `reward_cap` 必须显式给出，
    否则 `ValueError`；`reward_cap` 仅在 `saturate` 下合法，其他模式下给出 `ValueError`；
    `reward_cap ∈ [0,1]`，否则 `ValueError`。**禁止用 `None ⇒ 不饱和` 让 saturate 静默等价
    `terminal`。**
  - 透传：`train.py:405/424/606` 的 `make_env`。
- **向后兼容**：现有 4 个模式的逐步结果必须**逐位不变**；不加 `--reward-cap` 时不得改变任何
  旧行为。契约测试固定 4 个模式（`tests/test_env.py:629-635`、`tests/test_train.py:239-256`）。
- **测试（新增）**：(a) `saturate` 无 cap 抛错；(b) `terminal`+cap 抛错；(c) `cap∉[0,1]` 抛错；
  (d) 同一 seed 下 K0/K2 的终局回报序列 ≠ `terminal`（防臂重复，修复 `engineering-impl-05`）。
- **checkpoint 布局不变**：奖励不进 checkpoint ⇒ 旧 ckpt 可热启动。
- **产物落盘约束（修复 `engineering-impl-09`）**：K0 下胜局终局回报=0，`train.py:818-823` 的
  `outcome` 回退（`sign(return)`）会把胜局记成平局。实现时**必须**让 `_final_outcome` 从
  `info` 直取；若 reward 非 sign-like 且 `info` 缺失 outcome，**直接 raise**，不得从回报符号推。
- **文档触点（修复 `engineering-impl-10/13`）**：`RULES.md:72` RL-2（「训练只看分数，不直接
  优化胜负」）会被饱和改动**直接冲突**，任何落地都需 owner + 新 ADR 修订；
  `DESIGN.md:199` 的模式枚举需在 owner 批准后更新（本 workflow 禁止改 DESIGN）；
  `CONTEXT.md` **没有**奖励词条（§胜负在 `CONTEXT.md:79-80`），无需改。

---

## 4. 后续实验设计（预注册；不在本 workflow 执行）

> 触发条件：owner 决定触发 Tier 1（§2.6），且 §4.1 的 μ-headroom 预检通过、
> `selfplay-pool` workflow 已收尾、GPU 空闲。执行前须由 owner 批准本预注册；
> 执行者不得即兴加臂/改 seed/改端点。**本文件只设计，不执行。**

### 4.1 硬前置：μ-headroom 预检（修复 `rl-reward-07`）

主端点必须测在**对奖励变化敏感**的对手上。执行前先做：

1. 用现有 `t17_screen` 证明：顶部 5 点的 vs-random 胜率 0.865–0.900、μ 极差 4.0，
   **评估对这段区间不敏感**。
2. 找一个**非饱和**对手（如 `l1_1M` 或 self 冻结 ckpt），使得候选在其上的胜率不贴顶
   （例如 ≤0.75），从而奖励变化原则上能推动可分辨的胜率/μ。
3. **若找不到该对手**（当前 2 家评估工具只能测 random 或冻结 ckpt）：**只执行 §4.2 的
   warm-start 块**，并把它作为**唯一主端点**；from-random 冷启动臂一律降为 exploratory。
4. 把预检结论（含数字）写入 `runs/ra/headroom.md` 与报告的预注册段。

### 4.2 臂（revision-3、obs v5、MLP hidden=128）

**主块（warm-start，primary）**：同一 warm-start 父模型 `l1_1M`（`checkpoint_step1024000.pt`），
只改奖励，训练 500k：

| 臂 | 奖励 | 说明 |
|---|---|---|
| `C0w` | `--reward-shaping terminal` | warm-start 对照（同种子重训） |
| `K2w` | `--reward-shaping saturate --reward-cap 0.2` | **唯一 confirmatory 主臂** |

**副块（from-random 冷启动，secondary/exploratory）**：只在 headroom 预检通过时执行：

| 臂 | 奖励 | 说明 |
|---|---|---|
| `C0` | `terminal` | 冷启动对照 |
| `K2` | `saturate --reward-cap 0.2` | 主臂 |
| `K7` | `saturate --reward-cap 0.7` | 剂量臂（secondary） |
| `K0` | `saturate --reward-cap 0` | **仅描述**（用户字面读法；非 outcome-aligned） |
| `W`（可选） | `win` | 阳性对照（另一域的历史 null） |

### 4.3 完整训练命令（修复 `engineering-impl-06`）

入口：`.venv/bin/python -m seven523.train`（等价 `uv run --group train 7g523-train`）。
与 `runs/t17pool__1__1790439615/args.json` 对齐的显式 flag：

```bash
# 冷启动臂（每个 arm × seed）
.venv/bin/python -m seven523.train \
  --exp-name ra_<arm> --seed <s> --total-timesteps 500000 \
  --num-envs 8 --num-steps 128 --num-minibatches 4 --update-epochs 4 \
  --num-players 2 --hidden-size 128 --arch shared \
  --learning-rate 2.5e-4 --anneal-lr --gamma 0.99 --gae --gae-lambda 0.95 \
  --clip-coef 0.1 --clip-vloss --norm-adv --vf-coef 0.5 --ent-coef 0.01 \
  --max-grad-norm 0.5 --torch-deterministic --cuda True --tensorboard False \
  --checkpoint-interval 8 --eval-interval 0 --log-interval 1 \
  --opponent random --reward-shaping <mode> [--reward-cap <tau>]
# warm-start 臂追加： --load-checkpoint runs/t17long__1__1790439615/snapshots/checkpoint_step1024000.pt
```

- **`--total-timesteps 500000`**（不要写 499712）：`num_updates = total // batch_size = 488`，
  与历史 run 一致（`train.py:677`）。
- 执行时必须把命令写入 `runs/ra_<arm>__<seed>__<ts>/command.txt`（沿用
  `runs/t17_screen/command.txt` 惯例）。

### 4.4 训练 seed、预算、排队

- **k=5 training seed**（pre-registered）。seed 取**fresh 集合**：建议 `200–204`（若被占用顺延），
  **不得**与 selfplay-pool 的 `39/40/100–148`、Wave-1 的 `10–18`、历史 `0–8/29–37` 重叠
  （修复 `engineering-impl-14`）。启动前 `ls runs` + 台账核对。
- **预算**：主块 2 臂 × 5 seed = 10 × 500k ≈ 10 × ~7 min ≈ **1.2 h GPU**（串行/≤3 并发，
  `nice -n 5`）；副块 3–5 臂 × 5 seed 另计（owner 决定）。
- **排队**：`selfplay-pool` 的波次 + h2h 全部结束后才启动；启动前 `ps aux | grep seven523`、
  `nvidia-smi` 必须干净。
- h2h 全 `--device cpu --workers 4`，严格串行。

### 4.5 deal seed 台账

- deal seed **fresh 集合**：建议 `220–228`（9 个，供 3 个 deal seed/端点 + 余量），
  从中取 3 个用于主端点；**不得**与 selfplay 的 `0–8/10–18/29–40/100–148` 重叠。
- 建台账写入报告预注册段（修复 `engineering-impl-14`）。
- `plan_games`/`plan_duel_schedule` 从 base seed 抽 deal seed，同 base seed 会前缀重叠
  （`selfplay-pool-plan.md` §3.6 实测）⇒ base seed 也必须 fresh。

### 4.6 端点（修复 `engineering-impl-07/11/15`）

**主端点（confirmatory）**：对每个训练 seed `s`，`K_arm_s` vs `C0_s`（warm-start 块为
`K2w_s` vs `C0w_s`）在 fresh deal seed × 400 副上换座 h2h。

**跨 seed 合并必须两步（`combine_duel_seeds` 只能合相同 pair 的多个 deal seed，
`duel.py:317-325`；不同训练 seed 的 pair 不同会 `ValueError`）**：

1. 每个训练 seed 内：`tools/head_to_head.py --seeds <d1>,<d2>,<d3> --pairs 400`
   或 `tools/h2h_screen.py`（内部用 `combine_duel_seeds` 合 deal seed）。
2. 跨 5 个训练 seed：复用 `runs/t17w1/aggregate.py` 的口径
   `mean ± q · max(RMS(per-seed SE), training_seed_sd)/√k`，**同时报 z 与 t**（k=5 时 t 为判定口径）。

完整命令：

```bash
.venv/bin/python tools/head_to_head.py \
  --left  K2w_s=<ckpt> --right C0w_s=<ckpt> \
  --seeds <d1>,<d2>,<d3> --pairs 400 --json \
  --games-out runs/ra/h2h_K2w_s<seed> --device cpu --workers 4
# 对每个训练 seed 重复；再：
.venv/bin/python runs/t17w1/aggregate.py --pair K2wVSC0w --seeds s1,s2,s3,s4,s5 --family confirmatory
```

**绝对参照（descriptive，不作判定）**：每个臂的 `agent.pt` 与冻结 `l1_1M`、与 `random` 的
3×400 h2h + 共享 screen μ。

**机制端点（每条都报）**：

1. **机制（写死口径）**：固定重放的 GAE 分解，比较 `corr(A_GAE, A_outcome)`、
   outcome 内/间方差份额；**必须报归一化前后的两套**，并声明 `norm_adv` 会吸收部分水平变化。
   该端点**需要新 probe**（训练产物无逐局 outcome/margin，`train.py:947-950` 只写聚合）；
   probe 未做则记 **N/A**（修复 `engineering-impl-11`）。
2. **胜率/平局/分差**：各 ckpt 的 100+ 副 screen；确认 `K_*` 未牺牲 `P_w`。
3. **critic 健康**：`V` 对真实回报与对 outcome 的 EV/corr、`V` sd；`value_loss`；
   **V 塌缩止损**（§3.4）。
4. **尺度混淆指标**：`explained_variance`（写清分母是本臂自己的 Var），
   **value-clip 绑定比例**、**global grad-norm 绑定比例**（修复 `rl-reward-08`）。
5. **稳定性**：`entropy`、`approx_kl`、`clipfrac` 轨迹（`metrics.csv`）。
6. **分差分布形状**：`E[r_term|w]`、`P(|d|≤10)`（`d = own − other`，原始分差）、平局率。

### 4.7 判定门（预注册）

- **值得行动**：主端点合并 **CI 完全排除 0 且点估计 ≥ +10**；然后按 EPV §5 用 5 seed × 400
  （≈80% power）或 3 × 800 复算，并做 +20 单侧非劣检验才谈 ≥+20。
- **止损**：主臂主端点**点估计 ≤ +5** ⇒ 停该臂；若 `K2`（及 warm-start 的 `K2w`）≤+5 ⇒
  **关闭奖励饱和线**（与 A1/A2 同口径）。
- **V 塌缩止损**（§3.4）：若主臂 V sd ≤0.15 且 V-vs-R EV ≤0 ⇒ 记 critic collapse，
  **不采纳为奖励结论**。
- **剂量单调性（机制，非判定）**：`K2`/`K7`（及可选 `W`）随 τ 的方向；**`K0` 不参与**。
- **超参混淆声明**：因 §3.4 的 `pg:v_loss` 配比效应，报告必须同时给「未改 `vf_coef` 的读数」
  与「若 effect 完全由 critic 权重解释则如何证伪」；**不得**在首波加 `vf_coef` 臂。

### 4.8 多重比较与预注册纪律

- 主臂 = **仅 `K2`（warm-start 块的 `K2w`）**；`K7`、`K0`、`W`、冷启动块全部 secondary，
  报告时注明未做 FWER 校正，**判定只看主端点主臂**。
- 固定 seed 列表、固定 deal seed、固定端点；执行后不得改门、不得挑 seed。
- 预检结论、命令、台账、`verdict.json` 全部归档。

### 4.9 功效

Δ=10 的 50% power ≈ 1500–2000 副总牌（EPV §5）；k=5 × 3 deal seed × 400 副 = 6000 副，
对 +10 有较充分功效；同 seed 配对进一步降方差。若主端点 CI 跨 0 但点 > +5，按 EPV §5
说明是否需加牌数，不得事后声称「接近显著」。

---

### 4.10 Tier 1：边界跳变变体（`terminal_win`，用户要求；实现已合入）

> **触发**：owner 按用户要求行使 §2.6 的 Tier 1 触发权（升级项 D1）。
> **实现状态（实现轮增补）**：代码 + 测试已合入（`env.py`/`train.py` + `tests/test_env.py`、
> `tests/test_train.py`），**本阶段零训练、零 h2h**；实验部分仍按本节预注册排队。
> 与 §4.1 的关系：本变异的主端点是「同 seed 臂 vs 对照」的**直接配对 h2h**，不经过
> vs-random screen，因此 §4.1 的 μ-headroom 预检不阻塞该主端点；次端点（vs `l1_1M`）
> 承担强对手 headroom 检验。

**变体定义（实现口径，写死）**：

- 模式名 `--reward-shaping terminal_win`；λ 由 `--win-jump` 控制（float，默认 `1.0`）。
- 非终局步 `0.0`（与 `terminal` 一致，**不引入逐墩信号**）；终局步
  `r = Game.returns()[learner] + λ · seat_outcome(scores, learner)`。
  2 家即 `r_term = (own−other)/100 + λ·sign(own−other)`；N 家用 `seat_outcome`
  统一（`sign(own − max(others))`），**不用 50 分阈值硬编码**（§1.3）。
- λ 校验：有限且 `0 ≤ λ ≤ 10`；`λ=0` 与 `terminal` **逐位一致**；随 λ 增大，边界
  跳变连续变强，`λ→∞` 的极限才是 `win`（λ>2 已使 outcome 排序完全支配 margin 差异）。
  与 `win`/`trick_diff_win`/§3 的 `saturate` 都不同：纯终局、无逐墩项、不截断分差，
  只在 50 分边界处加离散跳变。
- **默认路径不变**：不启用 `terminal_win` 时既有 4 个模式数值逐位不变；其余模式忽略
  `win_jump`；checkpoint/网络/obs 不受影响（奖励不进 checkpoint）。

**λ 取值与预注册臂**：

| 臂 | 模式 | λ | 说明 |
|---|---|---|---|
| `terminal`（对照） | `terminal` | — | seed 1/2/3 复用 `runs/t17pool__{1,2,3}__1790439615` |
| `J025` | `terminal_win` | 0.25 | 小跳变（≈ margin 尺度的 12.5%），最接近对照 |
| `J10` | `terminal_win` | 1.0 | 分差与胜负各占一半；**主臂** |

`λ∈{0.25, 1.0}` 两臂 + `terminal` 对照为预注册集合，不得事后加臂/改 λ。`λ→∞` 即 `win`：
revision-2（greedy、热启动、A2 仅 1 seed）的 `win` null 是该极限的**历史极端上界**，
**不能**当作本域（revision-3/random/冷启动）的效应上界（§1.4），须自测。

**预注册实验（与 §4 共享协议；差异处写死如下）**：

- **k ≥ 5 training seed**：seed 1/2/3 的 `terminal` 对照直接复用
  `runs/t17pool__{1,2,3}__1790439615`（500k、`--opponent random`、冷启动、镜像 t17pool config）；
  seed 4/5 的对照与全部 λ 臂（每 seed × 每 λ 一条新 run）需要时补训练。λ 臂命令 =
  §4.3 模板 + `--reward-shaping terminal_win --win-jump {0.25,1.0}`、`--total-timesteps 500000`，
  写 `args.json`/`command.txt`。
- **主端点（confirmatory）**：λ 臂 vs **同 seed** `terminal` 对照，3 deal seed × 400 副起、
  换座 h2h（§4.5 fresh deal seed 台账；§4.6 两步合并：先合 deal seed，再用
  `runs/t17w1/aggregate.py` 合训练 seed，**z/t 双报**）。门沿用 §4.7：**CI 完全排除 0 且
  点估计 ≥ +10** 才谈值得行动；**< +5 止损**；同 seed 配对降 Δ Elo 方差。
- **次端点（secondary）**：各 λ 臂 vs 冻结 `l1_1M` 的迁移（同 3×400 换座）。理由：
  vs `random` 胜率已饱和在 **86.5%–90.0%**（诊断 §3.2 / §0.3），headroom 主要在强对手；
  次端点只作描述，不参与止损/行动判定。
- **机制与风险**：(1) `norm_adv=True` 逐 minibatch 减均值除 std —— 跳变改变跨状态相对权重
  与胜局内散布，对回报仿射缩放不敏感 ⇒ 预期效应量小于「原始跳变」给人的印象；机制端点须
  报归一化前后并按 §4.6 报 clip/grad-norm 绑定比例。(2) 纯 `win` 旧 null 只作历史参照
  （另一域、1 seed）。(3) **诊断 R1**：症状 ≠ 干预价值 —— 诊断裁定 `SYMPTOM=True`
  （仅 seed 1：+0.01816 return/1M ≈ +1 个 own 分）、`ATTR_RESOLVED=False`、
  `PLATEAU_FLAT=True`；本实验既不被「症状小」推翻，也不由「拓扑错位存在」自动成立。
- **预算与排队**：排在 **selfplay-pool workflow 之后**（当前 GPU 3 并发在跑）；启动前
  `nvidia-smi`/`ps aux` 干净；10+ 条 500k run 按 ≤3 并发、`nice -n 5` 串行波次；
  h2h 全 `--device cpu --workers 4` 严格串行。
- **执行状态（2026-09-27 执行轮增补，已完成）**：12 条新 run
  （`ra_J025__{1..5}`、`ra_J10__{1..5}`、`ra_C0__{4,5}`）全部 499,712 步、无 NaN；
  C0 seed 1/2/3 复用 `runs/t17pool__{1,2,3}__1790439615`（`terminal` 默认路径在
  `saturate` 合入前后各做一次逐位冒烟校验通过）。主端点（deal seed 220/221/222 ×
  400 副 × 2 座 × 5 训练 seed 换座 h2h，`runs/t17w1/aggregate.py` 两步合并，z/t 双报）：
  **J10 vs C0 +6.25** [z −6.31,+18.80；t −11.53,+24.02]（p=0.329；两臂 Holm p=0.659）
  → CI 含 0、<+10 **不达行动门**，点 >+5 **未触发止损**；**J025 vs C0 −4.59**
  [z −20.08,+10.90；t −26.53,+17.36] → ≤+5 **止损**。次端点 vs `l1_1M`（描述）：
  J10 −1.60、J025 −5.36、C0 −8.70，CI 均含 0。机制/健康度：vs random P_w 0.879–0.897
  未牺牲、无 V 塌缩（标准化 V sd≈0.62–0.67、EV≈0.50–0.55）。报告见
  `docs/experiments/reward-alignment-terminal-win.md`。执行纪律：训练 ≤3 并发 `nice -n 5`；
  h2h 串行 `--device cpu --workers 4`。`saturate` 合入后 run 的 `args.json` 多一个惰性键
  `reward_cap: null`（非研究变量，已在校验表标注）。

### 4.11 第二波：分差饱和 K_τ（clamp；用户要求，实现已合入）

> **触发**：owner 按用户要求（「赢了后拿多少分都无所谓」）行使 §2.6 的 Tier 1 触发权，
> 与 §4.10 的跳变波并列。**实现状态（实现轮增补）**：`saturate` 模式 + `--reward-cap`
> 已按 §3.5 落地（`env.py`/`train.py` + `tests/test_env.py`、`tests/test_train.py`），
> **本阶段零训练、零 h2h**。与 §4.1 的关系同 §4.10：主端点是「同 seed 臂 vs 对照」的直接
> 配对 h2h，不经过 vs-random screen，因此 μ-headroom 预检不阻塞主端点；次端点（vs `l1_1M`）
> 承担强对手 headroom 检验。

**变体定义（实现口径，写死）**：模式名 `--reward-shaping saturate`，由 `--reward-cap`（τ）
控制（return 单位，`τ∈[0,1]`；`τ=0.2 ⇔ d=20`、`τ=0.7 ⇔ d=70`）。非终局步 `0.0`；终局步按
outcome 条件式（§3.3，N 家不硬编码 50 分）：`own<m`（负）取 `r_term`、`own==m`（平）取 `0`、
`own>m`（胜）取 `min(r_term, τ)`；2 家退化为 `own<50: r_term`、`own=50: 0`、
`own>50: min(r_term, τ)`。约束（实现 raise，无静默退化）：`saturate` 必须显式给
`--reward-cap`；其他模式给 `--reward-cap` 报错；`τ` 非有限或 `τ∉[0,1]` 报错。默认路径逐位
不变；`λ`/`τ` 互不读取（`win_jump` 不影响 saturate，`reward_cap` 也不影响旧模式）。

**臂（预注册；k≥5 training seed、500k、冷启动 `--opponent random`、镜像 t17pool 的 §4.3
命令模板，只改奖励）**：

| 臂 | 模式 | τ | 说明 |
|---|---|---|---|
| `K2` | `saturate` | 0.2 | **唯一 confirmatory 主臂**（§3.1：q10，饱和 ~90% 胜局） |
| `K7` | `saturate` | 0.7 | 剂量臂（§3.1：q50，饱和 ~50% 胜局；secondary） |
| `K0`（可选） | `saturate` | 0.0 | 端点锚：胜局一律 0；**不是 outcome-aligned**（§3.1），仅描述 |
| `win`（可选） | `win` | — | 端点锚：`τ→0` 与旧 `win` null 的极端（另一域历史 null，§1.4），仅参照 |

**同 seed 对照**：seed 1/2/3 的 `terminal` 对照复用 `runs/t17pool__{1,2,3}__1790439615`；
seed 4/5 的 `terminal` 对照与 §4.10 的 J 波**共用**（J 波补训的 seed 4/5 对照即本波对照，
不重复训练）。

**端点与判定（预注册）**：

- 主端点（confirmatory）：每个训练 seed 上 `K2_s` vs 同 seed `terminal` 对照，3 deal seed ×
  400 副起、换座 h2h；跨 seed 两步合并（§4.6：先合 deal seed，再 `runs/t17w1/aggregate.py`），
  **z/t 双报**。门沿用 §4.7：**CI 完全排除 0 且点估计 ≥ +10** 才谈值得行动；**点估计 ≤ +5
  止损并关闭奖励饱和线**。
- 次端点（secondary）：`K2`/`K7` vs 冻结 `l1_1M`（同 3×400 换座）；只作描述，不参与
  止损/行动判定。
- 多重比较：`saturate` 家族 = {`K2`,`K7`} 的主端点，做**独立 Holm 校正**（与 §4.10 J 家族
  {`J025`,`J10`} 分开：家族间不合并、不互相校正），同时报 raw 与 Holm 调整后 p；判定仍只看
  `K2` 主端点。
- V-collapse 止损（§3.4 预注册预测与规则）：饱和降低终局回报方差 ⇒ 预期 V sd 与 V-vs-R EV
  **下降**；若某臂训练后期 **V sd ≤0.15 且 V-vs-R EV ≤0**，记 **critic collapse**，**不**采纳为
  奖励结论，与 h2h 主端点并列报告。
- 机制端点照 §4.6：归一化前后的 GAE 分解、`P_w`、critic 健康、clip/grad-norm 绑定比例、
  `entropy`/`approx_kl`/`clipfrac`、分差分布形状。

**排队与预算**：排在 **§4.10 J 波之后**（J 波又排在 selfplay-pool 之后）；启动前
`nvidia-smi`/`ps aux` 干净；每臂 × seed 一条 500k run（`runs/ra_K2__...` 等），≤3 并发、
`nice -n 5`；h2h 全 `--device cpu --workers 4` 严格串行。训练 seed 与 deal seed 台账照
§4.4/§4.5：fresh 集合，不复用 selfplay/Wave-1/历史集合；`args.json`/`command.txt` 照 §4.3。

**执行状态（2026-09-27 执行轮增补，已完成）**：20 条新 run（`ra_K0/K2/K7/win__{1..5}`）
全部 499,712 步、488 update、无 NaN；C0 seed 1/2/3 复用 `runs/t17pool__{1,2,3}__1790439615`，
seed 4/5 复用 J 波 `ra_C0__4/5`（不重复训练）。Deal seed **229/230/231**（fresh；
与 selfplay 0–18/29–40/100–148 及 J 波 220–228 均不重叠）、screen 232。训练 08:20–09:21
（≤3 并发 `nice -n 5`，与外部 `t23wswd` 共享 GPU 前 23 min、未干预）；评测 09:22–09:36
（45 个 h2h + 20 screen + 20 V 探针，严格串行 `--device cpu --workers 4`）。主端点
（同 seed 配对、3×400 换座、`runs/t17w1/aggregate.py` 两步合并、z/t 双报）：
**K2 vs C0 −18.31** [z −26.04,−10.58；t −29.27,−7.36]（saturate 家族 Holm p=6.9e-06）
→ **CI 全负、点 ≤+5，止损并关闭奖励饱和线**；K7 +7.27 [z −1.76,+16.31；t −5.53,+20.08]
（Holm p=0.115）未判定；端点锚 K0 −27.72、`win` −6.77。次端点 vs 冻结 `l1_1M`（描述）：
K2 −17.94、K7 −5.57、win −16.36、K0 −30.10、C0 −4.44。V 止损未触发（K2 raw V sd 0.073
≤0.15 但 EV +0.233 >0，AND 不成立）。vs random P_w 0.869–0.901 未牺牲。执行时的并发 T23
改动新增惰性键 `optimizer/lr_schedule/weight_decay/snapshot_interval`，默认路径与 t17pool 的
LR 轨迹逐位一致（488/488）。报告：`docs/experiments/reward-alignment-saturate.md`；条目已加
`docs/experiments/README.md` §1 首行。

---

## 5. 诊断脚本、产物规格、验收与回滚

### 5.1 脚本与产物（Diagnose 阶段）

- **脚本**：`runs/reward_alignment/analyze.py`（`runs/` 已 gitignore，属一次性工具，
  **不进 repo、不改 `tools/`**）。只读输入 + 只写 `runs/reward_alignment/`。
- **依赖**：`numpy`（必需）、`scipy`（可用则用于 Spearman/偏相关）、`matplotlib`（**best-effort**）。
  CPU-only；固定 RNG 种子（`np.random.default_rng(0)`），结果可复现。
- **产物**：
  - `runs/reward_alignment/d1_training_return_trend.csv`
  - `runs/reward_alignment/d2_return_decomposition.csv`
  - `runs/reward_alignment/d3_alignment.csv`
  - `runs/reward_alignment/d4_variance.csv`
  - `runs/reward_alignment/verdict.json`（`SYMPTOM/ATTR_RESOLVED/PLATEAU_FLAT` + 数值 + 输入 sha256 +
    命令 + `figures: done/skipped`）
  - `d1_trend.svg`、`d2_decomposition.svg`、`d3_alignment.svg`（**best-effort；缺 matplotlib 则
    CSV+ASCII 表，并在 `verdict.json` 记 `figures: skipped`**；修复 `engineering-impl-08`）。
- **报告**：`docs/experiments/reward-alignment-diagnostic.md`（结构对齐 §2，含限制与因果性声明）；
  在 `docs/experiments/README.md` §1 表**首行**加一行（列格式照现有行：报告/主题/关键结论），
  **仅此一处允许改 README，且仅 §1**。若 owner 决定归档/开放，另在 `docs/plans.md` 记录
  （由 owner 批准，本 workflow 不擅自改）。

### 5.2 验收标准（修复脏工作区问题）

1. `analyze.py` 全程 CPU、无网络、无训练/h2h 子进程。
2. **工作区不得新增改动（相对开工前快照）**：Diagnose 开工前记录
   `git status --porcelain` 与 `git diff | sha256sum` 到 `runs/reward_alignment/baseline.txt`；
   结束时对比，**除允许的文档外无新增 delta**。**明确声明**：HEAD 已含 T15–T17 的约 100 项
   未提交改动（开工前实测 `git status --short | wc -l` ≈100，**加上本 workflow 的两份文档**；
   该计数随并发 workflow 持续漂移，故只作量级声明、不设硬门），含 `src/`、`tests/`、`docs/plans.md` 等，
   属预期、不在本阶段范围，**不得为「干净」而 git add/checkout**（修复 `engineering-impl-03`）。
3. 每个数字都能由 `analyze.py` + 上述输入重算；报告给 `runs/reward_alignment/` 文件与命令。
4. D1 报 moving-block block∈{10,25,50} 敏感性 + 实测 acf/Newey-West；D2 报加法分解 CI 与
   「`ATTR_RESOLVED` 未判定」的显式结论。
5. 报告显式给出 `SYMPTOM/ATTR_RESOLVED/PLATEAU_FLAT` 与 §2.6 路由，并**区分相关与因果**。
6. 不含任何「奖励导致 Elo」的因果措辞；不发明未由脚本产生的数字。
7. **`analyze.py` 全文内联进 `docs/experiments/reward-alignment-diagnostic.md`**（或归档到
   owner 批准的持久位置），避免 `runs/` 被清理后数字失去唯一可复算来源（修复 `engineering-impl-16`）。

### 5.3 回滚（文件级清单）

- 诊断：
  - 删除 `runs/reward_alignment/`（gitignored）；
  - 回退 `docs/experiments/reward-alignment-diagnostic.md`（若已写）；
  - 回退 `docs/experiments/README.md` §1 的那一行；
  - 若 owner 曾归档，回退 `docs/plans.md` 的对应条目（owner 决定）。
  - **不触及 `src/`、`tests/`、`tools/`。**
- 实验（未来）：训练 run 在 `runs/ra_*`（gitignored）；若判定为 null，按
  `reward-shaping-500k.md` 的做法保留 `runs/` 与报告、不改代码。若实现新奖励后被否决，
  回退 `env.py`/`train.py` 的对应 diff 并删除对应测试。

---

## 6. 风险、未决、ADR 与长程

### 6.1 风险

| # | 风险 | 缓解 |
|---|---|---|
| R1 | **诊断测的是「症状」，实验测的是「从零换奖励」——两者不是同一命题** | §0.4/§2.6 显式声明；收束定义为资源决策；实验有独立止损门 |
| R2 | 训练 `episodic_return` 逐 update 只有约 43 局，**信号小** | 实测 acf≈0、NW≈iid ⇒ 主要噪声是「逐点方差大+信号小」；block bootstrap 作稳健性；非重叠窗口对比 |
| R3 | 跨 run/跨 fit 对齐非因果、CI 宽（±13）、`p2/p3` 不在 fit | D3 只作描述，报偏相关与口径偏移；**不做任何 Elo 乘除外推的门** |
| R4 | `eval_interval=0`，无同 run 在线 Elo | 报告显式列为口径限制；未来实验可选把 `eval_interval>0`（需 owner 批准） |
| R5 | 饱和改回报尺度 ⇒ `pg:v_loss` 配比、entropy 相对权重、clip 绑定比例改变 | 首波不改 `vf_coef`/`ent_coef`；报 clip 绑定比例；判定不依赖单一尺度读数 |
| R6 | 2 家的 `own>50` 不能推广到 N 家 | §3.3 用 outcome 条件式；N 家属 T7 |
| R7 | 多臂多重比较 | §4.8：只认 `K2`/`K2w` 主端点 |
| R8 | ~~screen 候选 deal 不共享~~ | **已消解**：P0 按构造成立（§2.1），配对做 ΔR/ΔW/ΔL |
| R9 | `runs/` gitignored，诊断产物易失 | 报告内嵌 `analyze.py` 全文 + 数字与命令；`verdict.json` 记 sha256 |
| R10 | 「分差是稳健性的代理」（赢得多 = 对更强对手也更强） | D2 报 `E[r_term|w]` 与 `P_w` 的关系；这是**反对**饱和的线索，须列出 |
| R11 | n=200/anchor 判不了归因 | §2.3 明确 `ATTR_RESOLVED=False/未判定`；要判定需 ≥1200 局/候选（新成本，owner 批） |
| R12 | 主端点 vs random 已饱和，假阴性风险 | §4.1 μ-headroom 硬前置；否则只执行 warm-start 块 |
| R13 | 饱和可能塌缩 critic（A1 式） | §3.4 预注册 V sd 预测与 ≤0.15 止损规则 |

### 6.2 未决问题

- Q1：饱和相对两端的**剂量响应是否单调**？`K7` 是否可能优于 `K2` 与 `W`（非单调）？
- Q2：错位是否随**对手强度**变化？warm-start 块与非饱和对手（§4.1）回答此问。
- Q3：纯 `win`（A2）只训 1 个 seed 就判 null，**是否值得在 revision-3 下补 k=5**？
- Q4：稠密 + 饱和（`trick_diff` + 饱和终局项）是否优于纯终局饱和？（A1 稠密本身未复现。）
- Q5：release/cap 的选择是否应由**分差分布分位数**动态决定（如 q10），而非固定 τ？
- Q6：若饱和有效，`RULES.md:72` RL-2 与 `DESIGN.md:199` 的定义是否要更新？需 owner 与 ADR。

### 6.3 需要的 ADR / 文档

- **实验不改变**任何既有 ADR（奖励语义属实现层）。
- **若首波命中并产品化**：需新增 ADR 记录「奖励目标从分差改为 outcome-consistent 饱和」，
  同时修订 **`RULES.md:72` RL-2**（与现口径直接冲突）、同步 **`DESIGN.md:199`** 的模式枚举。
  `CONTEXT.md` **无奖励词条，无需改**（修复 `engineering-impl-13`）。本 workflow 不写 ADR。
- **待办登记（修复 `engineering-impl-10`）**：`DESIGN.md:199` 的枚举更新须在 owner 批准后
  执行；本轮及实现轮均不得擅自改 DESIGN。
- 诊断报告与 `docs/experiments/README.md` §1 索引由 Diagnose 阶段写（§5.1）。

### 6.4 长程

- **N 家 / 联盟训练（T7）**：reward alignment 在 N 家必须用 outcome 条件式（§3.3）。
- **与 selfplay-pool 线的交互**：池化对手改变对手分布；reward saturation 与非饱和对手可能交互。
  注意 `train.py:818-823` 的 outcome 回退在 K0 下会把胜局记为平局（§3.5 已列为实现约束）。
- **评估侧**：报告同时给 `R_sign` 与 `R`，维持「训练目标 vs 评估目标」的显式口径。

---

## 附录 A：复算命令与原始输出（可精确复现）

> 全部在 `/home/amas/.local/src/7g523` 下、`.venv/bin/python`（scipy 可用）或系统 `python3`
> （numpy）执行；只读输入。数字标注为「预核实」；Diagnose 阶段须由
> `runs/reward_alignment/analyze.py` 重算并落盘。

**A.1 后期 return 趋势（t17long `[512k,2M]`）**

```python
# 读 runs/t17long__1__1790439615/metrics.csv 的 global_step/episodic_return
# 主 CI：moving-block bootstrap, block=25, 随机起点∈[0,n-block], B=2000,
#        rng=np.random.default_rng(0), 取 2.5/97.5 百分位
# 输出：
#   OLS slope/1M = +0.01816, block CI = [+0.0106, +0.0244]
#   512-768k mean = 0.5162 (sd 0.0601, n=250)
#   1.5-2M   mean = 0.5360 (sd 0.0592, n=489)
#   窗口差 = +0.0198, iid-bootstrap CI = [+0.0109, +0.0290]
#   centered-roll25 端点 512k=0.5112, 2M=0.5566, 差 = +0.0454
#   acf lag1/5/10/25/50 = -0.027/-0.029/-0.021/+0.012/-0.018
#   Newey-West SE(lag25) = 0.00354 ~ iid SE 0.00355
#   block 敏感性 {10,25,50} = [0.0101,0.0244]/[0.0106,0.0244]/[0.0099,0.0243]
```

**A.2 `games.jsonl` 平台候选的胜/平/负与分差（anchor vs random，n=200/候选）**

| id | μ | R | win% | tie% | loss% | E[r_term\|w] | E[r_term\|l] |
|---|---|---|---|---|---|---|---|
| l1_512k | 187.60 | +0.5535 | 90.0 | 3.5 | 6.5 | 0.6306 | −0.2154 |
| l1_1M | 191.61 | +0.5280 | 86.5 | 3.5 | 10.0 | 0.6376 | −0.2350 |
| l1_1.5M | 187.69 | +0.5500 | 90.0 | 3.0 | 7.0 | 0.6311 | −0.2571 |
| l1_2M | 191.11 | +0.5625 | 89.5 | 4.5 | 6.0 | 0.6447 | −0.2417 |
| p1_499k | 190.06 | +0.5965 | 90.0 | 3.5 | 6.5 | 0.6794 | −0.2308 |

**A.3 配对增量分解（seed-clustered, B=4000, `default_rng(0)`），单位=return**

| (a→b) | ΔR | ΔR_sign | Δ(P_w·E[r_term\|w]) | Δ(P_l·E[r_term\|l]) |
|---|---|---|---|---|
| 512k→1M | −0.0255 [−0.0840,+0.0315] | −0.0700 [−0.1550,+0.0150] | −0.0160 [−0.0695,+0.0350] | −0.0095 [−0.0230,+0.0020] |
| 1M→1.5M | +0.0220 [−0.0345,+0.0800] | +0.0650 [−0.0350,+0.1650] | +0.0165 [−0.0340,+0.0660] | +0.0055 [−0.0085,+0.0220] |
| 1.5M→2M | +0.0125 [−0.0310,+0.0560] | +0.0050 [−0.0600,+0.0700] | +0.0090 [−0.0300,+0.0460] | +0.0035 [−0.0070,+0.0140] |
| 512k→2M | +0.0090 [−0.0425,+0.0610] | 0.0000 [−0.0850,+0.0800] | +0.0095 [−0.0375,+0.0570] | −0.0005 [−0.0110,+0.0085] |

> **第二轮修复（`rl-reward-11` 连带）**：上表第 1–3 行 `ΔL` 的符号原为相反（使 `ΔR ≠ ΔW+ΔL`），
> 已按 `ΔL = ΔR − ΔW` 逐行校正并同步 CI 端点；现每行严格满足 `ΔR = ΔW + ΔL`。

**A.4 分数→Elo 斜率（仅作「量级参考」，不作门）**

```python
# 13 候选 OLS: mu ~ R  => slope = 308.0 +- 15.9 Elo / 单位 R（训练进度混杂）
# 顶部 5 点 OLS:        => slope =  -4.6 +- 43.5, 95% CI [-89.9, +80.7]
# 三个 Delta 估计 -> Elo: 窗口 +6.0 / OLS 积分 +8.4 / 端点 +14.0
```

**A.5 胜局分差分位数（顶部候选，n=173–180）**

| id | 胜局 n | mean r_term | q10/q25/q50/q75/q90 |
|---|---|---|---|
| p1_499k | 180 | 0.679 | 0.2 / 0.5 / 0.7 / 0.9 / 1.0 |
| l1_1M | 173 | 0.638 | 0.2 / 0.4 / 0.7 / 0.8 / 1.0 |
| l1_2M | 179 | 0.645 | 0.2 / 0.4 / 0.7 / 0.9 / 1.0 |

**A.6 within/total 方差份额（描述量）**

```python
# o=sign(r_term); Var(r_term)=Var(E[r_term|o])+E[Var(r_term|o)]
# e1_2k 0.325 | p1_262k 0.466 | l1_512k 0.557 | l1_1M 0.478
# l1_1.5M 0.527 | l1_2M 0.558 | p1_499k 0.524
```

**A.7 D3 相关与口径偏移**

```python
# 13 候选 centered-roll25 return vs mu: Pearson 0.966; 偏相关|log step ~0.50-0.62
# 顶部 4 点内相关 0.47; 晚期 6 点内相关 0.33
# 偏移 (return_centered vs R): t17long 512k 0.511/0.553, 1M 0.528/0.528,
#   1.5M 0.530/0.550, 2M 0.557/0.562; t17pool 262k 0.4975/0.562, 499k 0.526/0.597
```

**A.8 anchor seed 集（P0 证据）**

```python
# games.jsonl kind=='anchor' & opponent=='random'：13 候选的 seed 集完全相同
# （每 seed 出现 2 次 = 2 座位；100 个 seed/候选）=> 按 seed 配对
# 发牌只依赖 deal_seed（play.py:364；game.py:140-147），策略种子独立（record.py:47-54）
```

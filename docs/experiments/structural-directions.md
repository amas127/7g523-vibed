# 结构性改动：突破 ~1440 Elo 平台的研究与设计

> 资产状态（2026-09-25）：本报告引用的模型 checkpoint 已删除，旧路径不再可用；数值为牌型族规则变更前口径。

> **状态：设计完成（2026-09-25）。本文只做研究与设计：未改 `src/`、未 commit、未动任何已有报告与
> `runs/` 里的既有产物。** 现状量化由三个只读临时脚本产生，已复制到 `runs/structural/`（gitignore）：
> `structural_analysis.py`（终局奖励/墩结构/观测缺口）、`value_accuracy.py`（价值网逐段精度）、
> `last_trick.py`（撬底对胜负的影响），输出在 `runs/structural/out_*.txt`。
>
> **后续（2026-09-25/26）**：本文 A/B/C/F 四条优先线均已执行并收束——A1 未复现（[`reward-shaping-500k.md`](./reward-shaping-500k.md)）、B0 null/B1 未确认（[`observation-augmentation-b0.md`](./observation-augmentation-b0.md)、[`observation-augmentation-b1.md`](./observation-augmentation-b1.md)）、T5 三效应 null（[`opponent-distribution-500k.md`](./opponent-distribution-500k.md)）、T6 负/关闭（[`twin-towers-500k.md`](./twin-towers-500k.md)）；其后唯一过行动门的强度杠杆为推理期搜索（[`joint-search-training-wave1.md`](./joint-search-training-wave1.md)）。本文数字为当时口径。
>
> 一手背景：平台与已排除轴见 [`elo-breakthrough-report.md`](./elo-breakthrough-report.md)（§5 结论：平台是
> 训练分布问题）、统一 500k 预算的 6 方案全不可分见 [`wave5-500k-report.md`](./wave5-500k-report.md)（§5
> 未试方向）、PPO 移植无算法缺陷见 [`ppo-alignment-audit.md`](./ppo-alignment-audit.md)（§6.1）、
> 激活/损失/LR 全否见 [`activation-loss-sweep-report.md`](./activation-loss-sweep-report.md)。
> 评估口径一律用修好后的：候选内换座配对 [`elo-reliability-audit.md`](./elo-reliability-audit.md)，
> 候选对候选同牌 head-to-head [`head-to-head-pilot.md`](./head-to-head-pilot.md)。
>
> 术语约定：**平台**指 vs 冻结 GreedyBot 的对外强度在 ~60k 步后饱和（最终平衡口径
> `base680k` = 1443.3 ± 14.7，胜率 62–65%）；本报告的"方向"都必须回答两个问题：
> (i) 它改变的是**优化**（同一最优策略学得更好）还是**可达上界**（可用信息/策略类/博弈结构变了）；
> (ii) 预期效应大小能否被现有评估分辨（20 Elo ≈ 400–500 副牌，10 Elo ≈ 1500–2000 副牌，§7.2）。

## 0. 摘要与优先级

现状量化的关键事实（全部为本次一手测量或源码定位，详见 §1）：

1. **奖励**：终局 only——每局恰好 1 个非零奖励，占 learner 步的 **4.17%**（1000 局实测；
   `game.py:269-281`、`env.py:191-204`）。终局回报 sd **0.457**（=45.7 分），平局 7.3%，
   |分差|≥41 的局占 39.5%；一局平均 10.56 墩、69.5% 的墩带分，单墩 |分| 均值 9.47。
2. **信用分配**：`base680k` 的价值网对**终局实际回报**的整体 EV 只有 0.499；按决策进度分桶时，
   前 1/5 决策的 EV = **0.065**、RMSE 0.446（几乎等于回报自身 sd 0.461），到末 1/5 才升到 0.853。
3. **观测**：10 段共 191 维（2 家；`185+3n`）。**没有已出牌/弃牌历史**——平均一个决策点有
   **24.7 张**（最多 54）已经打掉的牌从观测里消失；也**没有编码当前墩的分值**（`View.trick_cards`
   里有全部牌，但编码器只写了 incumbent 顶牌）。
4. **对手**：`--opponent mix/pool` 都是**逐决策**独立抽成员（`policies.py:59-99`），不是逐局；
   mix50 每局平均换 **~11.6 次**对手，pool4g（权重 1/1/1/1/2）约 **18 次**；**没有任何胜率反馈**，
   所以 wave1–5 的池全部是"均匀/固定权重、每决策重抽"——不是 league，更不是 PFSP。
   而且池成员是 20k/57k/sp20k/sp40k 快照，**全部弱于同一个 warm start 的 `base680k`**。
5. **>2 家**：训练侧 `--num-players` 已存在（`train.py:122`）且 3 家 PPO 冒烟可跑；
   但 2 家 ckpt 热启动到 3 家会**直接抛 RuntimeError**（`obs_dim` 191→194，nvec 未变所以走了
   `load_state_dict` 分支，`train.py:359-362`）；评估链 `plan_games` 固定 2 座位（`ladder.py:135-141`）、
   `fit_ratings` 只支持 2 家（`elo.py:246-249`）、`duel` 只支持 2 家（`duel.py:149-150`）、
   `build_ladder` 没有 N 家开关且 manifest 硬编码 `num_players: 2`（`tools/build_ladder.py:256`）。

**推荐优先级（先做哪个、为什么）**：

| 序 | 方向 | 一句话理由 | 预期效应 | 成本 |
|---|---|---|---|---|
| **1** | **A. 奖励重构**（逐墩 telescoping 分差，± 胜负对齐） | 最便宜、零兼容风险、直接打击已量化的信用分配瓶颈（早期 EV 0.065），还顺带修"奖励=分差 vs 评估=胜负"的口径错位 | +0～30 Elo（主要改优化） | ~50 LOC，1 天 |
| **2** | **B. 观测增广**（已出牌/未见牌 + 当前墩分） | 唯一能改变**可达上界**的低成本改动；补齐平均 24.7 张不可见牌的信息缺口 | +0～50 Elo（未知） | ~120 LOC + 兼容层，1–2 天 |
| 3 | C. 对手分布（逐局/PFSP + **至少一个 ≥ 学习者的对手**） | 现有池是逐决策均匀重抽且全员更弱；正确联赛语义 + 难度梯度是未被证伪的分布杠杆 | +0～20 Elo | ~200 LOC，1 天 |
| 4 | **F. 架构：独立 actor/critic 双塔**（对齐 `ppo.py`；2026-09-25 追加） | 当前共享主干是审计 A5 的有意偏差；双塔消除策略/价值梯度干扰，参数量 +70%，但审计未见容量/优化瓶颈（EV 健康）→ 预期低、成本最低 | +0～20 Elo（未知） | ~40–60 LOC + 测试，半天 |
| 5 | D. >2 家训练 | 改变博弈结构，但断了所有 2 家评估/迁移路径，且是**另一个游戏**；作为产品改动风险最大 | 未知 | ~300–500 LOC |
| — | E. 容量/更长预算 | 既有证据为负（h256 从零最低、1M 不优于 680k），只作 A/B 命中后的组合项 | ~0 | ~30 min/臂 |

决策规则（2026-09-25 经 EPV §6/§9 与 README §3 统一）：**任何臂要谈"值得行动"，必须在
3 seed × 400–500 副牌的同牌换座 head-to-head 里相对 `base680k` 与 `w5_ctrl` 的合并 95% CI 排除 0，
且点估计 ≥ +10 Elo；<10 Elo 的效应不值得追（需 1500+ 副牌，见 §7.2）。** 要声称"越过 +20 行动
门槛"，不得要求点估计 ≥ +20——EPV §6 证明该规则在 Δ=20 时 power≈50%（0.47–0.48），且几乎不随
N/seed 数改善；正确表述是对 +20 做单侧等效/非劣检验，并按 §7.2 与 EPV §5 的功率表加牌数
（建议 5 seed × 400 或 3 seed × 800）。A 先行的另一层意义是证伪：如果 A 在 3×400 副牌上连 +10 都拿不到，
"信用分配/目标错位"假设被否，B 的优先级立刻升到第一。

---

## 1. 现状量化（一手）

### 1.1 奖励：稀疏、高方差、与评估口径错位

源码事实：`Game.returns()` 只在 `state.done` 时返回非零（`game.py:269-281`）；`Seven523Env.step`
在 learner 行动后、以及对手自动推进后检查 `match.done`，非终局 reward 恒为 0（`env.py:191-204`）。
一局在**撬底**发生时结束（`game.py:231-256`），所以每局正好一个非零奖励。

`base680k` vs GreedyBot、1000 局、seed 0 的实测（`runs/structural/out_terminal.txt`）：

| 量 | 数值 | 含义 |
|---|---|---|
| 非零奖励步 / learner 步 | **1000 / 23991 = 4.1682%** | 95.8% 的 PPO 样本奖励为 0（`w5_ctrl` 同口径 4.13%） |
| 终局回报 mean ± sd | **+0.2016 ± 0.4567** | 回报单位 = 分差/100 |
| 胜/平/负 | 61.8% / 7.3% / 30.9% | `eval_1500` 口径 63.0/7.3/29.7 |
| 平均 \|分差\| | 41.04（中位 40） | \|分差\|≥41 占 39.5%，≤10 占 22.6% |
| learner 步/局 | 23.99（16–32） | 总轮次 48.5，每 step 一次 learner 决策 |
| 墩/局、带分墩 | 10.56、7.34（69.5%） | 30.5% 的墩 0 分 |
| 单墩 \|分\| | mean 9.47、中位 10、max 60 | 逐墩塑形的自然尺度 ≈0.095 回报单位 |
| 前半局墩分差 vs 终局分差 | corr **0.766**，胜负一致 75.2% | 结果主要由中局决定 |
| 末墩翻转胜负 | **1.4%**；末墩均 6.83 分 | 撬底是每局必经的终局事件，但不是主要摆动点 |

价值网精度（`base680k`，600 局，逐决策 V(s) vs 最终实际回报，`runs/structural/out_value.txt`）：

| 决策进度 | n | EV | RMSE | 回报 sd |
|---|---|---|---|---|
| [0,0.2) | 2520 | **0.065** | 0.446 | 0.461 |
| [0.2,0.4) | 2880 | 0.279 | 0.390 | 0.460 |
| [0.4,0.6) | 2850 | 0.506 | 0.324 | 0.460 |
| [0.6,0.8) | 2880 | 0.697 | 0.253 | 0.460 |
| [0.8,1.0) | 3232 | 0.853 | 0.177 | 0.461 |
| 全部 | 14362 | 0.499 | 0.326 | — |

读法：**早期状态的 V 几乎不携带关于最终回报的信息**，而训练日志里的 `explained_variance`
0.64–0.69（`elo-breakthrough-report.md` §1.1）是在学习目标（GAE returns）上的数字，不代表对真实
终局回报的预测已经够好。GAE 的信用视野 `1/(1−γλ)=1/(1−0.99×0.95)≈16.8` 步，能覆盖一局的大部分，
所以瓶颈更可能是**价值目标噪声**而不是信用传不到。

奖励与评估口径的错位（RL-2 vs 天梯）：训练最大化期望分差（`RULES.md` RL-2），而 Elo 似然只用
胜负/平（`elo.py`）。平局 7.3%、±10 分内的局 22.6%，且单墩 10 分尺度上"保胜"与"搏分"的取舍不同。
这不是已证实的瓶颈，但 A 里可以用一个胜负奖励臂直接检验（§2.5）。

### 1.2 观测：逐字段清单与缺口

编码器单一事实源 `env.py:_SEGMENTS:105-115`，维度公式 `observation_dim = 185 + 3n`
（`env.py:123-125`），2 家 191 维：

| 段 | 维度 | 内容 | 公开性 |
|---|---|---|---|
| `hand` | 54 | 自己手牌 multi-hot | 私有（允许） |
| `rank_counts` | 15 | 自己每点数计数 /4 | 私有（允许） |
| `incumbent_top` | 54 | 当前需压过的牌顶牌 multi-hot | 公开 |
| `incumbent_kind` | 6 | 牌型 one-hot | 公开 |
| `incumbent_size` | 1 | 张数 /7 | 公开 |
| `draw_count` | 1 | 底牌堆张数 /54（顺序未知） | 公开 |
| `scores` | n | 各家已得分 /100 | 公开 |
| `hand_counts` | n | 各家手牌数 /7 | 公开 |
| `current` | n | 当前座位 one-hot | 公开 |
| `revealed` | 54 | 开局亮牌 multi-hot | 公开 |

缺口（均为公开信息、但没编码）：

| 缺口 | 现状 | 量级 |
|---|---|---|
| **已出牌/弃牌历史** | `GameState` 只有 `trick_cards`（当前墩）与 `collected` 计数；墩结束后牌从 `View` 消失（`game.py:80-93`、`game.py:203-256`），`GameState` 不保存历史 | 决策点平均已有 **24.7 张**牌打完且不可见（最多 54）；无法做记牌/剩余大牌推断 |
| **当前墩分值** | `View.trick_cards` 有全部牌，但编码器只写 incumbent 顶牌（`env.py:65-69`）；`View` 没有 `trick_points` | 模型不知道本墩是 0 还是 30 分在搏 |
| **未出现牌池** | 无 | 应由"54 − 自己手牌 − 亮牌 − 已出牌 − 当前墩"推出，MLP 学集合减法很吃力 |
| `last_player` | 在 `View` 里（`game.py:93`）但未编码 | 2 家可从上/下家轮转推出，>2 不可 |

差分泄漏约束仍在：只能编码公开可推的量（ADR-0002 的差分泄漏测试，`DESIGN.md` §6.6）。

### 1.3 对手分布：逐决策重抽、固定权重、无反馈

| 方案 | 抽法 | 实际分布 | 反馈/PFSP |
|---|---|---|---|
| `--opponent greedy` | 恒 GreedyBot | 100% | — |
| `--opponent mix` | `MixturePolicy.act` 每次调用独立掷骰（`policies.py:86-99`；`train.py:377-391`） | 默认 p=0.5；**一局内平均换 ~11.6 次对手**，整局同一对手概率 ≈0.5²⁴≈6e-8 | 无 |
| `--opponent pool` | 同上，成员权重固定（`train.py:392-413`） | `w5_pool4g` = 1/1/1/1/2 → greedy 33.3%、四个快照各 16.7%；**换 ~18 次/局** | 无：权重只在构造时给定，训练中不更新 |
| `--opponent self` | 共享一个冻结 `NeuralPolicy`，按 refresh 原地刷新（`train.py:312-328`、`train.py:471`） | 始终一个镜像 | — |
| PFSP | **不存在**：`train.py` 没有任何胜率/难度统计；`MixturePolicy` 没有 `reset/start_episode` | — | — |

两个额外事实：(i) `w5_pool4g` 的四个快照（20k、57k、sp20k、sp40k，`runs/w5_launch.sh`）**都弱于
warm start 的 `base680k`**，所以"联赛"里没有 ≥ 学习者的对手；(ii) 逐决策混合意味着对手身份是
隐变量，策略面对的是一个每步可能换风格的 POMDP——这是"per-episode 对手抽样"被 wave5 §5 列为
未试方向的原因。

### 1.4 训练/评估健康度（引用，不重复）

- 平台在 ~60k 步出现，之后 KL 2–3e-3、clipfrac ~8%、EV 0.64–0.69：不是梯度消失
  （`elo-breakthrough-report.md` §1.1）。
- `metrics.csv` 的 entropy 被花色头占 ~88%，模板头 680k 时 H≈0.24 nats（其合法上界 ~1.31 的 18%）：
  **模板头已接近确定**（同上 §1.2）；花色头强制最弱只 +1.1pp 不显著。
- 500k 统一预算、6 方案（bigbatch/pool4g/vf1/mix_samp/scratch）合并 3 seed 全在 ±11 Elo 内
  （`wave5-500k-report.md` §4.2）；激活/损失/LR 全否（`activation-loss-sweep-report.md` §5）。
- 评估：候选内换座配对后单 fit Fisher SE≈15.5–16.2（`elo.py` 联合 Hessian 已修）；
  head-to-head 400 副牌 95% CI 半宽 ±18.8–20.9、3 seed 合并 ±11.3–16.1
  （`head-to-head-pilot.md` §3、§4）。

### 1.5 >2 家的支持度核实（代码 + 实测）

| 检查 | 结论 | 证据 |
|---|---|---|
| 规则/环境支持 | 2–7 家（`num_players × hand_size ≤ 54`） | `rules.py:19-21`；`tests/test_env.py:19-31`、`tests/test_game.py:298-327` |
| CLI | `--num-players`（默认 2） | `train.py:122` |
| 训练冒烟 | 3 家、1 个 update 正常结束 | 本次实测 `np3_smoke`（写入 `/tmp`，未入库） |
| 2 家 ckpt 热启动到 3 家 | **RuntimeError**：`size mismatch network.0.weight [128,191] vs [128,194]`；nvec 相同所以走了 strict `load_state_dict` 分支 | `train.py:357-366`；本次实测复现 |
| 评估 | `plan_games` 固定 `(0,1)` 两座位；`fit_ratings` 只支持 2 家；`duel` 只支持 2 家；`build_ladder` 无 N 开关且 manifest 写死 2 | `ladder.py:135-141`、`elo.py:246-249`、`duel.py:149-150`、`tools/build_ladder.py:256` |
| 跨 N 迁移 | `obs_dim = 185+3n` 随 N 变，2 家模型不能直接 3 家评估（反之亦然） | `env.py:123-125` |

---

## 2. 方向 A：奖励重构（逐墩 telescoping 分差 ± 胜负对齐）

### 2.1 机制

- **A1（主力）逐墩分差回报**：把终局回报沿时间**重分配**：每个 learner 步奖励
  `r_t = Φ(s_{t+1}) − Φ(s_t)`，其中 `Φ(s) = own/total − mean(others)/total`（与 `returns()` 同式）。
  一局求和恰好等于原终局回报（telescoping），**总回报不变**，只改变价值目标的分解。
  这不是新的最优策略，而是把"1 个 sd=45.7 的终局标量"换成"~10.6 个 sd≈9.5 的局部增量"，
  正是针对 §1.1 早期 EV=0.065 的信用分配问题。
- **A2（次选）胜负对齐**：终局奖励改成 `+1/0/−1`（对 `max(others)`），直接优化天梯似然
  （`elo.py` 只用胜负/平）。这改变最优策略：预期分差下降、胜率上升。
- 为什么可能移动平台：平台在 60k 步就出现而 LR 仍有 2.35e-4，说明早期爬升被价值噪声限制；
  A1 让前 60k 的 critic 学到更干净的价值面（可用 EV 曲线验证机制），A2 则直接消除 RL-2 与
  评估口径的错位。两者都不需要改观测/对手/网络，可与任何后续方向叠加。

### 2.2 实现方案（file/function/CLI）

| 位置 | 改动 |
|---|---|
| `env.py:Seven523Env.__init__` | 新参 `reward_shaping: str = "terminal"`（`terminal` / `trick_diff` / `win` / `trick_diff_win`） |
| `env.py:Seven523Env.step:191-204` | 进入时 `before = self._potential()`；执行 `match.step/advance` 后 `after = self._potential()`；按模式返回 `after - before`（A1）或终局胜负（A2）。`_potential()` 用 `match.state.scores` 算 `returns()` 同式，`reset` 后 Φ=0 |
| `env.py:Seven523Env` / `train.py:make_env` | 透传模式（`make_env` 当前签名 `(rules, learner, opponents, seed, idx)`） |
| `train.py:parse_args` | `--reward-shaping {terminal,trick_diff,win,trick_diff_win}`，默认 `terminal` |
| 测试 | `tests/test_env.py`：A1 的 telescoping 恒等式（一局塑形奖励和 == 终局回报，含撬底的局）；默认模式逐位等于旧行为；非零步比例 ~10× |

向后兼容：默认 `terminal` 时行为逐位不变；**obs/动作/checkpoint 全部不动**，`base680k` 与所有
现成工具（`build_ladder`/`head_to_head`/`7g523-eval`）原样可用。

### 2.3 成本与风险

- 代码量 ~50 LOC + 3–4 个测试；训练/评测吞吐不变（~1370 SPS greedy；500k ≈ 6 min），
  head-to-head 400 副牌 ≈14 s。
- 风险：
  1. **不是上界改动**：若平台真是"信息/分布受限"，A 可能只有 0～10 Elo。这正是它作为第一步的
     证伪价值。
  2. **折扣下的目标漂移**：`γ=0.99` 时重分配会轻微改变折扣目标；建议 A1 附带一个 `--gamma 1.0`
     的消融（不必单列实验臂，作为 A1 内的二选一）。`env.done` 屏蔽 bootstrap 的语义已正确
     （`ppo.py:90-123`），γ=1 不会跨局。
  3. **A2 的 reward hacking**：胜率升、分差降，属预期而非作弊；但若胜率不升则撤销。平局按 0 处理
     与现行 Elo 似然一致。
  4. 逐墩增量里含有**对手行动造成**的分变（与终局回报一样无法完全归因到 learner），不新增偏差。

### 2.4 验收协议

统一协议见 §7.1。方向特有：

| 项 | 设置 |
|---|---|
| 训练 | warm start `base680k`，seed 1，500k，SAME_STEP，`--opponent greedy`，同批控制 = `w5_ctrl`（现成） |
| 臂 | `A1=trick_diff`、`A2=win`（单独跑，不要把两者混一起）；A1 内附 `--gamma 1.0` 消融 |
| 机制检查 | 训练日志 EV 与价值 loss；离线复测 V(s) 早期分桶 EV（目标：0.065 → >0.3） |
| 主端点 | `tools/head_to_head.py --left base680k --right <arm> --seeds 0,1,2 --pairs 400 [--games-out]`；再对 `w5_ctrl` 同口径 |
| 次端点 | `build_ladder --games-per-anchor 400 --seed {0,1,2}` 合并；`7g523-eval --episodes 1500 --opponent greedy` |
| 判定 | 按 §7.1.5 统一口径：3×400 合并 CI 半宽 ±11–16；CI 排除 0 且点估计 ≥ +10 才谈值得行动；+20 行动门槛用对 +20 的单侧等效/非劣检验（不用点估计≥20）；若只有 +10 量级，用 `--pairs 800` ×3 seed 确认（≈±8） |

### 2.5 预期与判读

先验 +0～30 Elo；最可能的有效期是 **60k–200k 的爬升段**（价值面变干净）。如果 500k 后 h2h 与
`w5_ctrl` 的 Δ 都 <10，则"信用分配/分差口径"假设否掉，B 升为第一优先。

### 结果（2026-09-25）

A1 `trick_diff` **命中**：相对 `base680k` 与 `w5_ctrl` 的 4 个 head-to-head 比较
（3×400 ×2 + 3×800 确认 ×2）95% CI 全部排除 0，点估计 **+9.99…+12.75 Elo**
（3×800 vs `base680k` 为 +9.99，恰好落在 +10 门槛之下）；按 C-1/EPV §9 判为
**真实但量级 ~+10、贴门槛**，不声称 +20。γ=1 与 win 臂**不达标**：A1γ1 vs `base680k`
−12.60 [−24.43,−0.78]（显著更差）、vs `w5_ctrl` +3.77 [−6.96,+14.49]；
A2 `win` vs `base680k` −7.53 [−18.93,+3.86]、vs `w5_ctrl` +8.54 [−3.27,+20.36]。

**预注册机制检查未通过**（§2.4 目标 0.065 → >0.3）：A1/A1γ1 的离线早期分桶 EV 反而降到
0.041/0.041（overall 0.016/0.043），且 critic 价值 sd 从 `base680k` 的 0.302 塌缩到
0.113/0.114；A2 早期桶 0.014（其 overall 0.369 主要靠后段）。改善可能来自稠密奖励本身
提供的信用信号，而不是 critic 变好（**假设**，待 `--vf-coef 0`/冻结 critic 的判别实验）。

**训练 seed 复现（2026-09-25 补记）**：补训 seed 2/3/4（各 500k，同配方）后**未复现**
seed 1：4-seed 平均 **+4.47（sd 5.6，vs `base680k`）/ +6.77（sd 4.4，vs `w5_ctrl`）**；
4 seed × 2 参照共 8 个 3×800 比较中，只有 seed 1 的两个 CI 排除 0。按统一口径
（<10 不追）**A1 作为独立杠杆不成立、A 线收束**；细节见
[`reward-shaping-500k.md`](./reward-shaping-500k.md) §3.4。

数字、CI、探针与产物路径详见
[`reward-shaping-500k.md`](./reward-shaping-500k.md) §2–§4。

---

## 3. 方向 B：观测增广（已出牌 + 当前墩分）

### 3.1 机制

唯一直接**扩大可达上界**的低成本方向：把公开可推但未编码的信息给模型。与 A 不同，它可能让策略
学会现在学不到的牌（"对手还剩什么大牌、哪门花色已经断了、本墩值不值得搏"）。支持证据：
平均 24.7 张出牌不可见（§1.2）；平台期的"模板头已尖锐"可能部分是因为模型没有足够信息做
多墩规划，而不是它不想学。

### 3.2 实现方案

分两档，可独立评估：

**B0（便宜、无引擎改动，先做）**：只加由现有 `View` 可推的量：
- `trick_points / 100`（当前墩分值，1 维；`View.trick_cards` 已有全部牌，`game.py:92`）；
- `remaining_points / 100`（100 − 已得分 − 当前墩分，1 维）；
- `point_hold`（自己手里的分值 /100，1 维）。
obs 191 → 194。**不动 `GameState`，不动 trace**，只需 `_SEGMENTS` 追加 + `encode_observation`；
但注意 §3.3 的编码器兼容问题。

**B1（主力，需要引擎加公开历史）**：
1. `game.py:GameState` 增加 `played: tuple[Card, ...]`（默认 `()`）；`_end_trick:203` 里
   `played = state.played + state.trick_cards`（撬底发生在终局，收入手牌的分无需入历史）。
2. `game.py:View:80-94` 增加 `played`，`Game.view:137-154` 填充。
3. `env.py` 新段 `unseen`（54 维）：54 − 自己手牌 − `revealed` − `played` − 当前墩的 multi-hot；
   或退一步只加 `played`（54 维）让模型自己减。推荐直接给 `unseen`（集合减法对 MLP 很难）。
4. 顺带把 `last_player`（1 维）编码（>2 家时需要）。
5. obs 191 → 194（B0）/ 249（B0+B1）。`trace.py` 不受影响：trace 只存 `Deal + 每步动作/分数`
   （`trace.py:96-107`），`Game.restore` 补 `played=()` 即可。

### 3.3 成本与风险

- 代码量 ~120 LOC + 测试（差分泄漏、obs_dim 公式、恢复/回放）。
- **编码器版本兼容（最重要）**：`NeuralPolicy.act` 调的是全局 `encode_observation`
  （`networks.py:230-245`）。一旦段布局变化，**所有旧 ckpt 推理都会因输入维度不匹配而失败**。
  两个必须做的兼容设计：
  1. **只追加、不改动既有段的位置**；旧 ckpt 读取新向量的前缀（`obs[:old_dim]`），
     在 `NeuralPolicy` 里按 `agent.obs_dim` 截断（并在 `save_agent` 里写 `obs_schema` 版本号）。
  2. **热启动列填充**：`train.py:357-366` 的 nvec 相同分支要加 `obs_dim` 判断；新增
     `networks.warm_start_pad_features(agent, loaded)`：把旧 `network.0.weight[:, :191]` 拷进去、
     新列置 0、其余同形张量照拷。这样 B0/B1 的初始函数与 `base680k` **逐位相同**，
     新特征从"零贡献"开始学。当前 `warm_start_into`（`networks.py:186-205`）只处理同形/actor 前缀，
     不覆盖这个场景。
- 风险：
  1. **泄漏**：只能编码公开可推量；`unseen` 含对手手牌与底牌堆的**并集**（合法），但不能编码
     "对手手牌 x 在哪家"或底牌顺序。必须扩展 ADR-0002 的差分泄漏测试。
  2. 54 维新特征可能稀释 191 维旧特征（尤其小 hidden=128）；建议 B0 与 B1 分臂跑，
     并且 B1 顺带考虑 hidden 192/256 的组合（一次消融，不作为独立方向）。
  3. 收益不确定：对 GreedyBot 这种弱对手，记牌的价值可能低于对人类/强对手。

### 3.4 验收协议

同 §7.1 统一协议；方向特有：

| 项 | 设置 |
|---|---|
| 训练 | warm start `base680k`（用 §3.3 的列填充热启动），500k，greedy；同批控制 `w5_ctrl` 仍适用（其 obs 旧布局，两个 ckpt 各自用自己的编码器评估） |
| 臂 | `B0`（点分量，3 维）、`B1`（+unseen/last_player，≥55 维）；B1 内可再叠 hidden 256 一臂 |
| 泄漏测试 | 构造"仅隐藏字段不同"的两个 `GameState`：公开字段、`View`、`obs` 必须逐位相同（扩展现有测试） |
| 主端点 | head-to-head vs `base680k`（3 seed × 400 副牌）；B0 vs B1 的直接对比（同牌） |
| 判定 | 同 §2.4；B 的效应若 <10 Elo，需要 1500+ 副牌才能定性 |

### 3.5 预期

先验 +0～50 Elo，方差极大；B0 是最便宜的下注（3 维、无引擎改动），如果 B0 有效说明模型
确实缺少"本墩值多少分"这种标量信息，B1 的成功概率会显著上升。

### B 结果（2026-09-25）

**B0（3 维点分量，`obs_version=2`）**：500k run `runs/t2_b0__1__1790330513`；
h2h 3×400 vs `base680k` **+1.74 [−7.94,+11.42]**、vs `w5_ctrl` **+4.92 [−6.72,+16.56]**，
均跨 0 → 不单独采用（null）。

**B1（`unseen` + `last_player`，`obs_version=3`）**：500k run `runs/t3_b1__1__1790331840`；
h2h 3×400 vs `base680k`（主端点）**+4.20 [−6.75,+15.15]**（CI 跨 0、点估计 <+10 → 失败）、
vs `w5_ctrl` +13.04 [+1.93,+24.15]、相对 B0 同牌直接对比 **+0.73 [−18.98,+20.43]**
（between-seed sd 17.4、per-seed −0.43/−16.08/+18.69）→ B1 的 55 维增量与 0 不可分。

**判定**：B0/B1 均**未确认**（主端点失败；相对 B0 的增量 ≈0 且 CI 极宽；单训练 seed、
seed 间方差与效应同量级；<10–15 Elo 需 1500+ 副牌）。**B 线收束**：不再单独追增广布局，
下一优先转向 C（逐局对手/PFSP），其后 F（双塔）；v3 布局与 `--obs-version` 兼容层当时作为
T4 观测 S1 迁移的基础保留（后随 [ADR-0009](../adr/0009-single-observation-and-comparison.md)
退役，现行唯一布局 v5）。细节见
[`observation-augmentation-b0.md`](./observation-augmentation-b0.md)（B0）与
[`observation-augmentation-b1.md`](./observation-augmentation-b1.md)（B1）。

---

## 4. 方向 C：对手分布（逐局选择 + PFSP + 更强对手）

### 4.1 机制

平台被诊断为"训练分布问题"（`elo-breakthrough-report.md` §5）：固定 Greedy 给出饱和梯度。
现有 wave1–5 的 mix/pool 之所以全是 null，有两个结构性原因（§1.3）：逐决策重抽把对手身份
变成每步隐变量；池成员全部弱于学习者，没有难度梯度。C 要做的是把"联赛"变成真的：
(i) **逐局**选成员；(ii) 按学习者对每个成员的**实测胜率**加权（PFSP），优先打"刚好打不过"的；
(iii) 池里**至少放一个 ≥ 学习者的对手**（否则永远只是复习）。

### 4.2 实现方案

1. `policies.py` 新增 `EpisodeMixturePolicy`：`start_episode()` 时按权重抽一个成员并冻结到本局；
   `act()` 转发；暴露 `current_id` 与 `finished_id`（上一局成员）。保留 `MixturePolicy` 不动。
2. `env.py:Seven523Env.reset:175-189`：对每个 opponent `if hasattr(p, "start_episode"): p.start_episode()`。
   SAME_STEP 下 reset 发生在终局步内部，训练侧要在 auto-reset **之前**读到上一局的 member id → 让
   wrapper 在 `start_episode` 时把 `previous_id` 记到 `finished_id`，训练循环在终局步读取并清空。
3. `train.py` 的 episode 统计处（`_episode_info`，`train.py:219-235`）按 env/seat 记录
   `(member_id, outcome)`（outcome 用终局 reward 的符号，或按 score diff）；每 K 个 episode
   重算权重：`w_i ∝ (1 − wr_i)² + ε`，再与均匀分布混合 `0.5·uniform + 0.5·PFSP`（防遗忘），
   `wr_i` 用 Beta 先验收缩。
4. **更强对手**（与 PFSP 独立、可先做）：至少一个成员来自 (a) 采样版 `base680k`
   （`NeuralPolicy(sample=True)`，随机性本身就会让固定弱点消失）、(b) GreedyBot 的变体
   （不同 tie-break / 保留大牌策略）、(c) 用 `base680k` 价值头做 1-ply 搜索的脚本 bot。
   这三个都不需要训练，可先用 `7g523-eval`/head-to-head 测出它们相对 `base680k` 的强度，
   再决定放谁进池。
5. 测试：逐局选择（一局内 member 不变）、权重更新单调、PFSP 关闭时与均匀逐局等价。

### 4.3 成本与风险

- ~200 LOC + 测试；训练速度与 `w5_pool4g` 同级（746 SPS，500k ≈ 11–12 min/臂）。
- 风险：
  1. **非平稳**：PFSP 会聚焦当前打不过的成员，可能遗忘简单对手；用均匀混合项 + 定期重估缓解。
  2. **胜率估计噪声**：每个成员需要 ~100+ 局才有可用的 wr；前期权重近似均匀，接近现有行为。
  3. 若"更强对手"来自采样版自己，本质仍是 self-play，wave1–3 的镜像均衡风险仍在；
     真正的梯度来自脚本搜索 bot 或 ≥ 学习者的快照。
  4. 归因分解：臂与控制的差别要能拆成"逐局 vs 逐决策"和"加权 vs 均匀"两个因子，因此规划两个控制。
- 与 wave5 的关系：`w5_pool4g`（逐决策、固定权重、全员更弱）只能作为**反面**参照，不能作为
  C 的控制；控制必须是新的"逐局均匀"臂。

### 4.4 验收协议

| 项 | 设置 |
|---|---|
| 臂 | `C_u`=逐局均匀池（控制）、`C_p`=C_u + PFSP 权重、`C_s`=C_u + 一个 ≥ 学习者的成员 |
| 训练 | 500k，成员与 `w5_pool4g` 相同的四个快照 + greedy 以便对照，另加示例强成员 |
| 主端点 | head-to-head：`C_p` vs `C_u`、`C_s` vs `C_u`、各自 vs `base680k`（3×400） |
| 附加检查 | 对**池外**对手（greedy + 一个未入池的强成员）的胜率，防过拟合池内 |
| 判定 | 同 §7.1；PFSP 的预期效应最小，若 `C_p` 与 `C_u` 的合并 CI 完全重叠且 \|Δ\|<10，判 null |

### C 结果（2026-09-25）

三臂 500k 跑完（`C_u` 逐局均匀 / `C_p` +PFSP / `C_s` +强成员 `base680k`；obs v1、seed 1、
499,712 步、无 NaN；`runs/t5_*__1__1790332696`）。8 个 h2h（3 seed × 400 副牌换座）的
95% CI **全部含 0**、点估计 |Δ|≤7：`C_p` vs `C_u`（PFSP 效应）**+1.88 [−17.93,+21.69]**、
`C_s` vs `C_u`（强成员效应）**+2.90 [−17.32,+23.12]**，各臂 vs `base680k` 在 −6.2…−0.7、
vs `w5_ctrl` 在 +0.1…+7.0；Greedy 1500 四臂 62.5–63.4% 均在平台带内。
**判定：逐局冻结、PFSP、强成员三效应均 null**（单训练 seed；`C_s` 的强成员＝热启动父模型，
池内仍无真正 ≥ 学习者的独立参照；PFSP 降权 greedy 后难度梯度仍浅）。数字、per-seed 与风险
详见 [`opponent-distribution-500k.md`](./opponent-distribution-500k.md)；C 线收束，下一优先
T6 F（见 [`../plans.md`](../plans.md) §8）。

---

## 5. 方向 D：>2 家训练（低优先，需要前置工程）

### 5.1 机制

多人博弈的策略空间/联盟结构不同，可能产生 2 家对局里不存在的探索压力；但它是**另一个游戏**，
对 2 家产品的迁移不保证，且当前所有评估/锚点/迁移路径都断了（§1.5）。机制上最可能的价值是
"训练出更稳健的牌序/记牌能力"再迁回 2 家——而这一步需要 obs 跨 N 统一。

### 5.2 实现方案（前置依赖链）

1. **修热启动**：`train.py:357-366` 在 nvec 相同时也要比较 `obs_dim`，不同则走列填充；
   反向（3 家→2 家）需要特征裁剪/重投影，不在本设计内。
2. **座席轮换**：learner 目前硬编码 seat 0（`train.py:352`）。N 家训练应让 env i 的
   learner = `i % N`，对手表按 env 生成（`make_env` 的 `learner` 参数已存在，改为逐 env 传入）。
3. **N 家评估**：`plan_games` 要泛化成"同一副牌候选轮转 N 个座位"；`fit_ratings`/`duel` 需要
   多人 Elo（把一局 N 人分数分解成 n(n−1)/2 对两两比较 + 按牌聚簇 bootstrap，或 Plackett-Luce）。
   锚点（random=1000/greedy=1315）是 2 家刻度的，N 家下需重新标定或改用候选对候选相对比较。
4. **跨 N 统一 obs**（决定迁移能力）：把 per-player 段（scores/hand_counts/current）固定到
   `max_players`（如 7）并用 seat mask 补齐，obs 与 N 解耦；否则 3 家模型永远只能打 3 家。
5. `tools/build_ladder.py` 增加 `--num-players`，manifest 的 `num_players` 不再写死
   （`tools/build_ladder.py:256`）。

### 5.3 成本与风险

- ~300–500 LOC（含评估工具），3 家 self-play 的对手推理成本约为 2 家的 2 倍；500k 步预算下
  是否需要 1M 未知。
- 风险：胜率/Elo 锚点不可比；多人最优策略含"国王制造"（kingmaking），迁回 2 家可能是负迁移；
  高方差（3 家局内分数相关性更强），所需局数更多。

### 5.4 验收协议

- 先做**内部**相对端点：3 家候选对候选换座对局（需新工具），确认训练至少不劣于同预算 3 家基线。
- **产品端点**：把模型迁回 2 家（依赖步骤 4），与 `base680k` 做 3×400 副牌 head-to-head；
  只有 2 家端点成立才算突破。若不愿做跨 N 统一 obs，D 只能作为独立研究项，不应进入产品优先级。

---

## 6. 方向 E：容量/更长预算组合（不建议单列）

既有证据已经足够强：`h256_scratch`（从零 250k）是 wave1 最低（1367.7，
`elo-breakthrough-report.md` §4.3）；1M 不优于 680k（`ladder-report.md` §2.1）；bigbatch/vf1 在
500k 统一预算下不可分（`wave5-500k-report.md` §4.2）。因此 hidden 256 + 1M 只在 A/B 出现正信号后
作为**组合确认**（例如 B1 的 54 维新特征需要更大容量），成本 ~30 min/臂。不单列实验轴。

---

## 6A. 方向 F：独立 actor/critic 双塔（对齐 `ppo.py`；2026-09-25 追加）

### 6A.1 机制与假设

现状是**共享主干**（`networks.py:88-95`）：一个 `Linear(obs,128)+act+Linear(128,128)+act` 同时接
策略头（std=0.01）与价值头（std=1.0）。参考 `ppo.py` 用**两个独立 MLP 塔**（actor 塔、critic 塔），
这是审计 A5 记录的有意偏差（对齐 ADR-0003 指定源 `ppo_multidiscrete_mask.py` 的共享 CNN trunk）。

假设：分离塔可以消除 policy loss 与 value loss 在同一组主干参数上的**梯度干扰**，让两个头各自
学适配的表征，可能改善 60k–200k 爬升段的优化。反方证据：对齐审计与结构量化都表明**没有容量/
优化崩溃**（GAE 目标上的 EV 0.64–0.69、KL 稳定；早期 EV 0.065 是价值目标噪声/信用分配问题，
不是表征共享问题）；参数会从 ~59k 增到 ~100k，小预算下可能反而变慢。因此这是一个**低预期、
低成本、可证伪**的方向。

**必须隔离变量**：主臂是 **双塔 + ReLU**（只改架构；wave4 已证明把激活换成 tanh 会 −48.8 Elo）。
「完整的 ppo.py 配方」（双塔 + Tanh）只作为附加臂，若跑，需单独解释。

### 6A.2 实现方案（file/function/CLI）

| 位置 | 改动 |
|---|---|
| `networks.py:Agent.__init__` | 新参 `arch: str = "shared"`（`shared`/`towers`）；`towers` 时建 `actor_network`/`critic_network` 两个同规格 MLP（各自正交初始化 √2），头仍 std 0.01/1.0 |
| `networks.py:save_agent/load_agent` | payload 加 `"arch"`；旧 ckpt 缺省 `shared`（与 `activation` 同模式） |
| `networks.py:warm_start_into` | 跨架构拷贝：`shared→towers` 时把共享 trunk 权重拷进两个塔（形状相同）；`towers→shared` 时取 actor 塔（并在报告里注明策略）；同架构逐键拷贝 |
| `train.py` | `--arch {shared,towers}`（默认 shared）；传给 learner 与 self-play frozen 两处 `Agent(...)` |
| 测试 | 两架构的 forward/replay/checkpoint 往返；跨架构 warm start 形状/数值；旧 ckpt 缺省 shared；`--arch towers` 冒烟 |

向后兼容：默认 `shared` 行为逐位不变；ckpt 通过 `arch` 字段自描述；现有工具（`build_ladder`/
`head_to_head`/`7g523-eval`）经 `load_agent` 自动适配，无需改动。

### 6A.3 成本与风险

- 代码 ~40–60 LOC + 测试，半天；训练/评测吞吐几乎不变（主干 FLOPs ×2，但网络本来就极小）。
- 风险：① 收益可能为负（参数更多、每步更慢、小预算样本效率下降）；② 跨架构 warm start 会
  暂时破坏函数（共享 trunk 被拷成两份相同塔，两塔随后分叉），首轮指标应记录；③ 「忠实 ppo.py」
  不等于「更好」（本游戏的 ReLU + 共享 trunk 已被 wave4/对齐审计证明是合理选择）。

### 6A.4 验收协议

沿用 §7.1 统一协议，方向特有：

| 项 | 设置 |
|---|---|
| 训练 | warm start `base680k`，`--arch towers`，seed 1/2/3，500k，对手 greedy；同批控制 = `w5_ctrl` |
| 附加参照 | 从零 500k `towers`（对照 `w5_scratch 1412.4`）——隔离 warm-start 交互 |
| 机制检查 | 训练日志的 policy/value loss 曲线与梯度范数；早期分桶 EV（目标 0.065 → 提高） |
| 主端点 | `head_to_head --left base680k --right <F 臂> --seeds 0,1,2 --pairs 400`；再对 `w5_ctrl` 同口径 |
| 判定 | 按 §7.1.5：合并 CI 排除 0 且点估计 ≥ +10 才谈值得行动；+20 行动门槛走单侧等效/非劣检验；<10 Elo 视为不可分 |

### 6A.5 预期与判读

先验 **+0～20 Elo**，不排除为负。最有信息量的结果不是「是否越过 +20 行动门槛（单侧检验）」，而是：如果 F 与 shared 在
3×400 副牌上不可分或更差，就**明确关闭「架构忠实度」这条线**，把预算留给 A/B/C；如果 F 为正，
再考虑与 E（容量）或 B（增广）组合。

### 6A.6 实现分析（2026-09-25，obs_version v2 落地后）

> 本节把 §6A.1–6A.5 落到可直接实现的程度。引用以当前工作区为准（有未提交改动，v2 兼容层在
> `src/seven523/env.py` 与 `src/seven523/networks.py`）；参数量由 `.venv/bin/python` 现场构造
> `Agent` 数出，不是估算。

#### 6A.6.1 接口设计

**结构（推荐）**：`arch="shared"` 逐位保留现有 `self.network`/`self.actor`/`self.critic`；
`arch="towers"` 时建 `self.actor_network` 与 `self.critic_network` 两个同规格 `nn.Sequential`
（每塔 `Linear(obs,128)-act-Linear(128,128)-act`，各自 `layer_init(√2)`），actor 头挂 actor 塔、
critic 头挂 critic 塔，两塔之间零横连。理由：
- 头与塔不交叉是「梯度干扰」实验的本体，任何横连都会污染假设。
- 复用 `self.actor`/`self.critic` 头名，shared 的 state_dict key 集合（`network.0/2.*`、`actor.*`、
  `critic.*`）逐位不变，旧 ckpt 与原测试中直接访问 `agent.network` 的断言全部不受影响。
- 不要「shared 时把两塔 alias 成同一 module」（state_dict 出现重复 key、旧档不可加载），也不要
  全改 `ModuleDict`（key 全面迁移，收益为零）。

**属性 vs 方法**：`network`/`actor`/`critic` 是带参数的状态，必须保持模块属性以维持 state_dict
与旧 ckpt；「用哪个塔」是行为，用方法表达。在 `networks.py:119-121` 之间新增（~12 LOC）：

```python
def _actor_hidden(self, x):  return self.actor_network(x) if self.arch == "towers" else self.network(x)
def _critic_hidden(self, x): return self.critic_network(x) if self.arch == "towers" else self.network(x)
def policy_logits(self, x):  return self.actor(self._actor_hidden(x))
```

- `get_value`（`networks.py:121-122`）→ `self.critic(self._critic_hidden(x))`，两 arch 同接口。
- `get_action_and_value`（`networks.py:124-159`）只把 `hidden = self.network(x)`（136）换成
  `self._actor_hidden(x)`；value（159）写成 `self.critic(hidden) if self.arch == "shared" else
  self.critic(self._critic_hidden(x))`（shared 前向一次，towers 两次）。
- 不要给 `network` 加 property 别名指向 actor 塔：shared 主干同时喂两头，别名会写坏语义，
  且 `get_value` 会变难读。
- `NeuralPolicy.act`（`networks.py:293`）是 src 内唯一直接访问点：
  `self.agent.actor(self.agent.network(obs))` → `self.agent.policy_logits(obs)`（1 行）。
  `NeuralPolicy.obs_version` 已取自 ckpt（`networks.py:272`），无需改。

**save/load**：`save_agent` payload（`networks.py:165-172`）加 `"arch": agent.arch`；`load_agent`
（`networks.py:184-191`）加 `arch=payload.get("arch", "shared")`，与 `activation`/`obs_version`
同一缺省模式。`runs/probe/base_step00696320.pt` 实测 payload 只有 `obs_dim/nvec/hidden`，是最老的档。

**参数（实测）**：

| arch | obs 191（v1；base680k/w5 系列） | obs 194（v2，当时默认 `OBS_VERSION`；现为 v5） |
|---|---|---|
| shared | **59,019** | 59,403 |
| towers | **100,107**（+69.6%） | 100,875（+69.8%） |

单塔主干 41,088（v1）/ 41,472（v2），actor 头 17,802，critic 头 129；差值恰好等于单塔主干。

**代码量**：`networks.py` ~55–65 LOC（建塔 ~10、hidden 方法 ~12、两个 forward ~5、save/load ~2、
`warm_start_into` ~25–30、`NeuralPolicy.act` 1）；`train.py` ~10；可选梯度日志 ~15；测试 ~150–200 LOC。

#### 6A.6.2 热启动矩阵

矩阵描述**梯度步之前**的性质：权重逐位拷贝 ⇒ 同输入输出逐位相同（`torch.equal`）。实现上先做
key 重映射再复用现有守卫：
- shared→towers：源 `network.*` 映射到目标 `actor_network.*` 与 `critic_network.*`（同一 origin
  拷两份）；heads 原样。
- towers→shared：源 `actor_network.*` 映射到目标 `network.*`；忽略 `critic_network.*`；heads 原样。

| 源 → 目标 | 拷贝规则 | policy | value | 备注 |
|---|---|---|---|---|
| shared→shared | 现行逐键（`networks.py:241-247`） | 逐位 | 逐位 | 现行路径，`tests/test_train.py:99-120` |
| towers→towers | 逐键；首层 pad 须泛化到两塔 | 逐位 | 逐位 | 不泛化则 v1→v2 首层形状不等被静默跳过 |
| shared→towers（同 obs） | 重映射，形状全等 | 逐位 | 逐位 | 两塔彼此亦逐位相同；T6 干净性前提 |
| shared v1→towers v2 | 重映射 + 首层 pad（旧列拷贝、新列置零，两塔同值） | 与 shared-v1 逐位 | 与 shared-v1 逐位 | 新列零权重起步 |
| towers v1→towers v2 | 逐塔 pad | 与源逐位（v1 前缀） | 同左 | 需泛化首层特例 |
| towers→shared（同 obs） | 取 **actor 塔** 到 `network` | 逐位 | **不逐位** | 值函数暂为「旧 critic 读 actor 塔」；报告注明策略等价、critic 需重适配 |
| towers v1→shared v2 | 上条 + pad | 与源逐位（v1 前缀） | 不逐位 | 同上 |
| towers v2→shared v1 | 首层被 width 守卫挡住（194>191）；bias/二层/heads 照拷 | 否 | 否 | 与现行 shared v2→v1 一致：拒绝截断 |
| shared v1 3p[194]→towers v2 2p[194] | `observation_num_players` 守卫（3≠2）挡住两塔首层，其余照拷 | 否 | 否 | 保持 `tests/test_train.py:440-466` 语义 |

**守卫合并**：把 `networks.py:213-239` 的首层特例抽成 `_copy_input_layer(key, value, origin,
loaded, agent)`，触发条件扩为 `key in {"network.0.weight", "actor_network.0.weight",
"critic_network.0.weight"}`；内部三条件（`env.py:177-195` 语义）不改：
`observation_num_players(loaded.obs_dim, loaded.obs_version) == observation_num_players(agent.obs_dim,
agent.obs_version)`、源 `obs_version` ≤ 目标、源宽 ≤ 目标宽。重映射发生在取 `origin` 时，
守卫看到的是被映射的源张量，因此跨 arch×跨 obs 自动走同一分支。`copied` 建议记为
`"actor_network.0.weight<-network.0.weight"` / `"...[:, :191]"`，训练打印（`train.py:405-411`）
可直接展示。

**近似项**：只有 towers→shared 的 value 不逐位（唯一有损组合）；v2→v1 两首层不拷贝是现行设计。
其余组合同 obs 时逐位等价，v1→v2 时对 v1 前缀函数逐位等价（新列零权重）。

#### 6A.6.3 训练接线

| 位置 | 改动 |
|---|---|
| `train.py:144-151`（--activation 之后） | `--arch {shared,towers}`，default `shared`，非法值报错 |
| `train.py:389-394` learner | `Agent(..., arch=args.arch)` |
| `train.py:347-356` frozen self-play | 同样传 `arch=args.arch`；否则 `train.py:360` 的 `load_state_dict` 在 towers 学习者 + shared frozen 时 RuntimeError |
| `train.py:398-403` 严格重载分支 | 条件加 `and loaded.arch == args.arch`；否则「shared ckpt → towers 学习者」不会走 `warm_start_into`（405），而在 strict load 抛 unexpected keys |
| `train.py:410` Adam / `ppo.py:210` clip / `train.py:581` get_value | **零改动**：`get_action_and_value`/`get_value` 接口不变，`ppo.py:154` 与训练循环不感知 arch |
| `train.py:480`、`train.py:522` | SAME_STEP 与同 arch frozen 刷新，不变 |

**与 `--obs-version`**：当时默认 v2（`env.py:33`、`train.py:124-134`）；现为 v5（ADR-0008，
`src/seven523/env.py:38`），`env.py:33` 已是注释位。而 `base680k`/`w5_ctrl`/
wave5 双臂都是 v1（`args.json` 无该字段、ckpt 191 宽）。主臂必须显式 `--obs-version 1`，否则 arch
与观测改版两个变量同时变。v1→v2 的 pad 分支已由 `tests/test_train.py:414-437` 固定，重映射后对
两塔同时生效。**与 `--activation`**：正交；主臂保持 relu，因为 wave4 已把 tanh 判负
（ALS §5：`act_tanh −48.8±22.2`），不能让已知负效应污染架构变量。

#### 6A.6.4 测试清单

`tests/test_train.py` 追加（沿用 `OBS_DIM=191`、小网络与 `_tiny_args`，`:344`）：

1. ckpt 往返：towers 存读后 `arch=="towers"`、forward 一致；旧 payload 缺 `arch`
   （现有 legacy 用例 `:89-98`）→ `"shared"`。
2. forward/action 语义：参数化 `:48-66` 到两个 arch——形状、掩码合法性、replay `torch.equal`、
   logprob 一致。
3. 参数计数：`hidden=128, obs_version=1` 为 shared 59,019 / towers 100,107，差 == 单塔 41,088
   （6A.6.1 实测值）。
4. shared→towers 逐位：`torch.equal` 断言两个塔的首层/二层 == 旧 `network`、`policy_logits` == 旧、
   `get_value` == 旧；`len(copied)==12` 且含 `"actor_network.0.weight<-network.0.weight"`。
5. towers→towers 同 obs 逐位；towers→shared：`network==源 actor_network`、policy 逐位、
   value == `源.critic(源.actor_network(x))` 且一般 ≠ `源.get_value(x)`。
6. 跨 obs：shared v1→towers v2 两塔新列全零且 v1 前缀函数逐位等于 shared-v1；towers v1→towers v2
   同断言；`:440-466` 的 shifted equal-width 守卫补 towers 目标：首层不拷、heads 拷。
7. `NeuralPolicy` 冒烟：towers act 合法；`sample=False` 的选择 == 掩码后 `policy_logits` argmax。
8. 训练冒烟：`_tiny_args(tmp_path, "--arch", "towers")` 跑通，`load_agent(...).arch=="towers"`；
   `parse_args([]).arch=="shared"`。
9. 零回归：实现后 `pytest tests -q` 仍须 **334 passed**（当前 collect 数），旧断言不改。

#### 6A.6.5 实验协议（收紧 §6A.4）

**主臂（架构隔离，唯一变量）**——钉住 v1：

```bash
.venv/bin/python -m seven523.train --exp-name t6_towers_relu --seed 1 \
  --total-timesteps 500000 --cuda True --tensorboard True --checkpoint-interval 0 \
  --log-interval 50 --opponent greedy --arch towers --activation relu --obs-version 1 \
  --load-checkpoint runs/probe/base_step00696320.pt
```

- 若 owner 要以 v2 为基线，控制组用 `runs/t2_b0__1__1790330513/agent.pt`（obs2、terminal、seed1、
  同一 warm start），结论降级为「arch×obs 组合」，不得再用 vs `base680k`/`w5_ctrl` 声称纯 arch 效应。
- **附加臂 `towers+Tanh`**：主臂 + `--activation tanh`，仅描述性；先验负（ALS §5:170,182），
  不参与架构判定。
- **从零对照**：去掉 `--load-checkpoint`（其余同主臂，exp-name `t6_towers_scratch`），h2h vs
  `runs/w5_scratch__1__1790319698/agent.pt`；绝对 Elo `1412.4`（wave5:142）只作 sanity，不得跨 fit
  比较（N-7，`plans.md:305`：跨 fit sd≈31、相位位移 0–75）。

**机制检查（预注册）**：

- **干扰前提的廉价证伪（先做，不训练）**：在 shared ckpt 上取固定 rollout batch，分别对 policy
  loss 与 value loss 求共享主干参数的梯度，报 `cos(g_pol, g_val)`；近 0 则「共享主干梯度干扰」前提
  不成立，应把 T6 预期降到 ~0、预算让给 A/B/C。
- 训练日志：policy/value loss、`approx_kl`、`clipfrac`、`explained_variance`（`ppo.py:214-221` 已在
  metrics.csv；terminal 奖励下各臂 EV 同尺度可比）；**grad norm 现无日志**，需捕获 `ppo.py:210`
  的 `clip_grad_norm_` 返回值（~3 LOC），可选按 `named_parameters()` 前缀记两塔分范数（~12 LOC），
  看 update 1–50 两塔是否即分叉。
- 早期分桶 EV：`runs/structural/value_accuracy.py --checkpoint <ckpt> --episodes 600 --seed 0`，
  对照 base680k 首桶 `[0,0.2)` 的 0.065、目标 >0.3（SD §2.4；reward-shaping-500k §2.1）。注意
  `runs/structural/value_accuracy.py:48` 硬编码默认 v2 encoder，跑 v1 臂前须改用 `agent.obs_version`
  （1 行，`runs/` 下，不属 src/tests/tools）。
- 主端点沿用 wave5 模板（reward-shaping-500k §3.1）：`tools/head_to_head.py --left
  t6_towers_relu=ckpt:... --right base680k=ckpt:runs/probe/base_step00696320.pt --seeds 0,1,2
  --pairs 400 ...`，再对 `w5_ctrl` 同口径，JSON 落 `runs/h2h_t6_*`。判定按 §7.1.5：CI 排除 0 且
  点 ≥+10 才谈行动；+10～20 用 3×800 或训练 seed 2/3 复现；≥+20 走对 +20 的单侧检验并按 EPV §5
  配置（5×400 或 3×800）。

#### 6A.6.6 风险与取舍

- **与参考栈的关系**：PAA §4.2 A5 明确，共享 trunk 是 ADR-0003 指定源
  `ppo_multidiscrete_mask.py` 的做法，双塔才是 `ppo.py` 的做法，且 `ppo.py` 同时是 Tanh。因此 T6
  **不是修忠实度 bug**，而是测试「policy/value 分开建塔对优化是否有实质好处」；`towers+ReLU` 也
  不是完整 `ppo.py` 配方，完整配方臂与 wave4 的 −48.8 Elo 已知负效应混杂，只能描述。
- **干扰假设证据偏负**：GAE 目标 EV 0.64–0.69、KL 稳定，早期 EV 0.065 是价值目标噪声/信用分配
  而非表征共享（§6A.1）；先验应接近 0、不排除为负。梯度余弦探针可在训练前直接检验前提，是最便宜
  的证伪工具。
- **成本**：参数 +69.6%，网络 FLOPs 约 ×1.7–2，但网络相对环境极小；T1 同配方 500k 三并发墙钟
  437 s–7 m18 s，预计吞吐变化不显著。真正风险是 500k 预算下参数变多导致样本效率下降。
- **热的暂态**：step 0 与父模型逐位相同（比 wave4 换激活的「非同函数续训」干净），但 update 1 起
  两塔分叉；只比 endpoint 会丢掉机制信息，须记录前 50 updates 的 loss/grad 曲线与分桶 EV。
- **关闭条件**：若 3×400 对两个参照均不可分或更差，按 plans T6 验收关闭「架构忠实度」线，不做 E
  组合、不追加 Tanh 臂；若端点 null 但梯度/EV 指标显示两塔确实解耦，作为机制观察写入 §9，不再
  单独烧预算。

### F 结果（2026-09-25）

三臂 500k 与 6 组 h2h 3×400 全部跑完（obs v1、seed 1、499,712 步、无 NaN；
`runs/t6_towers__1__1790334485`、`runs/t6_towers_trickdiff__1__1790334485`、
`runs/t6_towers_scratch__1__1790334485`，机读 `runs/t6/health.json`）：

- **主臂 `t6_towers`（双塔+ReLU）** vs `base680k` **−9.13 [−20.57,+2.31]**、vs `w5_ctrl`
  **+5.21 [−5.93,+16.36]**：两个 CI 均含 0、点估计 < +10 → 与 shared **不可分**，
  训练 EV 曲线与 `w5_ctrl` 重合（0.66→0.69）。
- **交互臂 `t6_towers_trickdiff`** vs A1（`t1_a1_trickdiff`）**−15.51 [−27.10,−3.92]**：
  唯一排除 0 的交互比较，双塔在 `trick_diff` 上显著更差；独立 critic 塔未避免价值塌缩
  （overall EV 0.014、首桶 0.042、V sd 0.113，与 A1 的 0.016/0.041/0.114 几乎一样）。
- **从零对照 `t6_towers_scratch`** vs `w5_scratch` **−20.89 [−33.81,−7.97]**：等预算下
  参数 +69.6% 明显更差。
- **预注册机制探针证伪干扰前提**：`base680k` 固定 batch 上 `cos(g_pol,g_val)=+0.049`
  （|g_pol|=0.071、|g_val|=0.151）；两塔确实分叉（相对 base 首层 0.178/0.094、彼此 0.201），
  但分离本身没有换来强度。

**判定**：按 §6A.5 与 §6A.6.6 关闭条件，双塔不可分且更差、交互臂显著负、梯度干扰前提证伪
（cos≈0.05）、从零对照 −20.9 → **关闭「架构忠实度」线**，不做 E 组合、不加 Tanh 臂；该结论
进入 [`../plans.md`](../plans.md) §6 N-20。限制（单训练 seed、交互臂跨 run/代码版本、
towers→shared 的 value 有损）与全部数字见
[`twin-towers-500k.md`](./twin-towers-500k.md)。

---

## 7. 优先级、验收协议与资源

### 7.1 统一实验协议（所有方向共用）

1. **训练**：`warm start runs/probe/base_step00696320.pt`（`base680k`）、seed 1、
   `--total-timesteps 500000 --cuda True --tensorboard True --checkpoint-interval 0
   --log-interval 50`、SAME_STEP 语义（当前 `train.py:429`）；除非方向定义，对手为 greedy。
   低预算筛选（150k–250k）只在机制检查时用，**结论一律以 500k 为准**（wave5 §5）。
2. **固定参照**：`base680k`（冻结父模型）+ `runs/w5_ctrl__1__1790319698/agent.pt`
   （同语义 500k greedy 控制）。任何"变强"必须同时超过这两个。
3. **主端点（候选对候选，最紧）**：
   `python tools/head_to_head.py --left base680k=ckpt:... --right X=ckpt:... --seeds 0,1,2
   --pairs 400 --device cuda --games-out runs/structural/h2h_X.jsonl`。
   分辨率（`head-to-head-pilot.md` §3–4）：400 副牌单 seed 半宽 ±19.6；3 seed 合并 ±11.3–16.1；
   **20 Elo 用 3×400（约 45 s/对）；10 Elo 用 3×800 或单跑 1600–2000 副（约 1–1.5 min）**。
4. **次端点（绝对刻度/常规报告）**：`tools/build_ladder.py --games-per-anchor 400 --no-traces
   --seed {0,1,2}`（候选内换座配对，Fisher SE≈15.5–16.2；合并报告按 wave5 §4.2）；
   以及 `7g523-eval --episodes 1500 --opponent greedy`（固定 seat 0，SE≈1.25pp≈16 Elo）。
5. **判定规则（2026-09-25 与 README §3 / EPV §9 统一；见 `../plans.md` §7 C-1）**：
   - **值得行动** = 主端点合并 CI 完全排除 0 **且**点估计 ≥ +10 Elo（相对两个参照）；
     +10～20 的临界结果必须用 3×800 副牌或多 seed 复现后才可写进结论
     （`head-to-head-pilot.md` §3.2 的 pool4g 教训）；低于 10 Elo 视为不可分。
   - **+20 行动门槛**：不得用「点估计 ≥ +20」判定（EPV §6：该规则在 Δ=20 时 power≈50%，
     且几乎不随 N/seed 数改善）；声称越过 +20 门槛须对 +20 做单侧等效/非劣检验，
     并按 EPV §5 的功率表配置牌数（≥20 Elo 的行动结论建议 5 seed × 400 或 3 seed × 800 复算）。
6. **机制检查**（每个方向自带）：A 看 V(s) 早期分桶 EV；B 看泄漏测试 + 特征消融；
   C 看池外胜率；D 看 2 家迁移端点；F 看 policy/value loss 与梯度范数曲线（梯度干扰假设）
   及早期分桶 EV。

### 7.2 分辨率速查（用于定预期效应；表值为 50% power 口径，80% power 约 ×2，见 EPV §5）

| 目标 Δ | 副牌数（σ≈10–11.1） | 3 seed 合并 |
|---|---|---|
| 30 Elo | ~200 | 3×100（±10） |
| **20 Elo** | **400–500** | **3×400（±11–16）** |
| 15 Elo | 700–850 | 3×300–400（±11–13） |
| **10 Elo** | **1500–2000** | 3×800（±8） |

### 7.3 资源与时间

| 方向 | 代码 | 单臂训练 | 评估（每对被比对象） |
|---|---|---|---|
| A | ~50 LOC | ~6 min（greedy，1370 SPS） | 3×400 h2h ≈45 s |
| B | ~120 LOC + 兼容 | ~6 min | 同上 |
| C | ~200 LOC | ~11–12 min（pool，746 SPS） | 同上 + 池外检查 |
| F | ~40–60 LOC | ~6 min | 3×400 h2h ≈45 s（同 A/B） |
| D | ~300–500 LOC | 20–40 min（3 家 self-play，估 300–500 SPS） | 需新工具 |

一个方向的完整筛选（训练 + 2 个 h2h + build_ladder 3 seed）≈ 20–30 分钟墙钟，适合"一天 3–4 臂"。

---

## 8. 与"人类定级研究"的协同

| 方向 | 协同点 |
|---|---|
| A | **直接协同 D1/D2**：人类计划的主力信号 S2 是"价值网 regret"（`docs/human-elo-plan.md` §2），它要求一个校准良好的价值头；当前 `base680k` 早期 EV=0.065，A 的价值目标分解正是同一修复。EV 提升会提高 S2 的 `m_eff`，直接减少 20 局 ±50 所需的对局数。 |
| B | 记牌/未见牌是真人牌手最自然的特征；B 训练出的策略作为 S3 policy-agreement 的参照更接近人类风格；另外 B 的 `unseen` 特征本身可能进入 S1/S2 特征表。 |
| C | 更强对手 = 更多可分辨的等级：当前梯级实际是"训练了 12k/20k/100k/680k 步的同族策略"（`ladder-report.md` §2.4），池内多样性可能产出风格/强度都不同的 rung，缓解"平台内无 Elo 差"。 |
| 通用 | 顶 rung 从 1443（平衡口径）向上抬高，才能给强真人留出头部空间；同时 `head-to-head` 口径（±11–16 @3×400）也让梯级标定更省局数。 |
| D | 与 2 家人机定级计划不同源，除跨 N 统一 obs 的工程收益外无直接协同。 |

---

## 9. 不建议做的事（及理由）

1. **继续扫 PPO 旋钮**：激活（tanh/gelu/silu）、vf_coef、clip、ent_coef、LR 量级/退火全部扫过，
   没有一个正贡献（`activation-loss-sweep-report.md` §5）；不要再用 500k 预算重复。
2. **继续在 2 家 vs Greedy 的同族配方上堆 seed/预算/局数**：1M ≤ 680k，500k 六臂不可分；
   分辨率不会靠堆局数创造突破，只会把 ±52 压到 ±30 而效应 <10。
3. **hidden 256 / 更长预算单列**：已有负面证据（§6）。
4. **self-play 刷新间隔/采样/mix/均匀 pool 的再度组合**：wave1–5 已覆盖，全在噪声内。
5. **非 potential 的奖励加成**（赢墩奖励、炸弹奖励、剩牌惩罚等）：有 reward hacking 风险，
   且"目标错位"尚未被证明是瓶颈；A1 的 telescoping 已覆盖无风险部分。
6. **`--cross>0` 做筛选**：联合 Hessian 虽已修（`elo-reliability-audit.md` §2.3），但绝对锚点+
   交叉对局的分辨率仍远差于 head-to-head（±52 vs ±11–16），stageB 的 ±4.8 是旧代码产物不可复用。
7. **用绝对 Elo 单 fit 宣布 <20 Elo 的胜负**：跨 fit/跨顺序 sd≈31，相位效应 0–75
   （`elo-reliability-audit.md` §3）；新报告必须走换座配对/head-to-head。
8. **先做 >2 家**：断了评估与迁移，且是另一个游戏；没有 2 家迁移端点前不进入产品优先级。
9. **为评估再改一次协议**：现在的 head-to-head + 换座配对已经够用，边际收益低。

---

## 10. 开放问题

1. **信息受限 vs 优化受限**：A 与 B 的对比是设计好的判别实验（A 改优化、B 改上界）。
   若两者都 null，则更可能是"vs 固定弱脚本 bot 的胜率本身接近上限"或"该 MLP/动作空间
   的表达上限"——需要先回答下面的 Q2。
2. **2 家 7鬼523 对一个固定 GreedyBot 的真实上限是多少？** 现在的 1440 是相对锚点的测量刻度，
   不是胜率上限。可以用一个已知较强的参考（1-ply 搜索 bot 或多人训练后的模型）测出
   GreedyBot 的可被利用空间；如果上限就是 ~70%，那"平台"不是训练问题而是目标对手太弱，
   应当直接换课程对手（C 的强对手项）。
3. **逐墩分差是否值得按"墩"再细分**（例如区分自己赢墩 vs 对手赢墩的 credit）？A1 只是时间重分配；
   更细的 credit（如"本墩里每个决定的边际贡献"）需要搜索/值分解，成本高。
4. **PFSP 的胜率是否应该对"平局"和"分差"加权**（当前 Elo 只用胜负）？可以用分差作为软标签
   （与 `FitConfig.margin` 一致）。
5. **观测增广的记忆需求**：如果 B1 的 `unseen` 仍不够，是否需要显式的"每个对手剩余大牌估计"
   或循环网络（GRU/LSTM）？后者会破坏"纯 MLP + 全观测"的简洁契约，属于 B 的下一档。
6. **>2 的迁移价值**：跨 N 统一 obs 是工程题，但值得先做一个小实验回答"3 家 500k 的模型
   在 3 家内部是否比 2 家 500k 的模型（同 obs 前 191 维）更强"，再决定是否投工程。

---

## 11. 复现（本次量化）

```bash
# 终局奖励分布 / 墩结构 / 观测缺口（~20 s）
.venv/bin/python runs/structural/structural_analysis.py --episodes 1000 \
  --checkpoint runs/probe/base_step00696320.pt

# 价值网逐段精度（~15 s）
.venv/bin/python runs/structural/value_accuracy.py --episodes 600

# 末墩翻转率（~5 s）
.venv/bin/python runs/structural/last_trick.py

# 输出
cat runs/structural/out_terminal.txt runs/structural/out_value.txt runs/structural/out_last_trick.txt
```

> 所有数字来自 `base680k` vs GreedyBot、learner 固定 seat 0、seed 0；脚本只读，不写 `runs/` 之外。
> `np3_smoke` 的 3 家训练冒烟写到了 `/tmp/7g523_smoke/`（未入库）。
> 本文未修改 `src/`、未 commit、未触碰任何既有报告与 `runs/` 产物。

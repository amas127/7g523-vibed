# 序列级记忆：设计调研与 D1-lite pilot（7鬼523 / PPO）

> **状态：已完成（2026-09-26）**。本文回答「序列级记忆能否给当前 PPO 策略带来可测 Elo」：
> 先做设计调研（P0）与 self-play 下的 B1 盲/有史消融（P1），再实现并实测一个
> **不破坏 obs v5 的无状态全历史序列编码器 + 同架构盲对照**（D1-lite，P2）。
>
> **口径**：revision-3（出空即撬底，`rules_id=2e36dbea44893696`）、obs v5（2 家 161 维）、
> T17 工作区。所有训练/评估产物在 `runs/`（gitignore）；新增源码与测试为 opt-in，默认路径
> 不开启新开关时逐位不变。验收沿用 [`README.md`](./README.md) §3 的统一门槛：
> ≥3 seed × 400 副牌换座 h2h，**CI 完全排除 0 且点估计 ≥+10 Elo 才谈「值得行动」**。
>
> **结论摘要（数字见 §4/§5）**
> 1. **P1（self-play，单 seed 对）**：历史盲 vs `t17self` = **−9.85 [−22.68, +2.97] Elo**
>    （即历史优势 +9.85 [−2.97, +22.68]）：方向为正，但 CI 跨 0、点估计 <+10 → self-play
>    分布下同样**无可确认的 B1 增益**（单 seed，见 §8.6）。
> 2. **P2（D1-lite + 容量盲对照，random 500k）**：3 训练 seed 合并，关键端点
>    **seq vs seqblind = +8.41 [−4.06, +20.89] Elo**（CI 跨 0、<+10 门槛）；
>    seq vs t17pool +12.44 [−0.07, +24.94]、容量盲对照 seqblind vs t17pool
>    +2.84 [−20.00, +25.68] → **序列信息无可确认增益**。
> 3. **设计判定**：D1-lite 能覆盖的顺序信息（a1–a3）无 ≥+10 证据；逐座位/墩归属与倾向
>    （a4–a6）不在 `View` 里、本实验测不到（要 H3/v6）。结论：**不为序列记忆单独立项**；
>    如仍要测，先在 self-play 分布下补 D1-lite（判断线见 §6）。

## 0. 关键结论

1. **「无历史」不是现状**：v5 已含 `unseen`（54 维记牌集合补集，`src/seven523/env.py:106-122`）
   与 `last_player`（当前墩归属，`env.py:124-135`）；`GameState.played`（`src/seven523/game.py:41-45`）
   保存所有已结束墩的出牌顺序，`View.played`/`View.trick_cards` 公开（`game.py:104-106`）。
   前序消融（[`history-fusion-ablation.md`](./history-fusion-ablation.md) §3.2）在 random 500k
   显示 B1 段边际价值 −2.89 [−15.31,+9.52] Elo（CI 跨 0、低于门槛）。
2. **序列相对 `unseen` 的信息增量是「顺序、近因、组合邻接、当前墩节奏」**；`unseen` 是集合，
   丢了出牌顺序与组合边界。**逐座位归属与墩边界不在 `View` 里**（`played` 只存牌、不存人，
   也不存墩界），任何纯 `View` 的历史编码器都拿不到这条最值钱的轴——要它必须改引擎（H3/v6）。
3. **预期信号的位置**：random 对手没有可学倾向，信号主要来自「记牌/近因」的一次性利用；
   self-play/强对手才可能提供可学倾向与后验（对手策略是自己的历史函数）。因此 P1/P2 都按
   「先便宜前置、后核心实测」排序（前序报告 §5.4 的建议）。
4. **主源复核**：DouZero 的 LSTM 确为「历史出牌」编码器（arXiv:2106.06135，本轮下载 ar5iv 全文
   核实，见 §2.1）；cleanrl 的 recurrent PPO 实现细节已逐行核实（`ppo_atari_lstm.py`，§2.2）；
   Suphx 的 GRU 是 global reward predictor Φ（arXiv:2003.13590，本轮再次核实），**不是**策略
   历史编码器（§2.4）。帧堆叠实现以 cleanrl `ppo_atari.py:105` 为核实锚点（§2.3）。
5. **路线推荐**：先测 D1-lite（不触发 ADR-0009、推理无状态、评测链零改动）；只有它（或 P1）
   给出正信号，才值得上真递归策略（b）或逐座位/墩归属 v6（H3）。「null 后是否还上真递归」的
   判断线见 §6。

## 1. 现状与信息缺口

### 1.1 v5 里已经有什么（现场复核）

- 观测段表 `env.py:186-209`（2 家宽度）：`hand54, inc_rank15, inc_suit4, inc_kind6, inc_size1,
  draw1, scores2, opp_count1, opp_revealed19, trick_points1, remaining_points1, point_hold1,
  unseen54, last_player1` = 161。
- `unseen = 54 − 自己手牌 − revealed − played − 当前墩`（`env.py:106-122`）是**集合摘要**：
  对手手牌 ∪ 底牌堆的并集，不含顺序、不含分布。
- `last_player` 是当前 incumbent 的绝对座位（`env.py:124-135`），2 家冗余。
- `View` 公开字段（`game.py:92-107`）：`hand, mask, incumbent, current, scores, counts,
  draw_count, revealed, trick_cards, played, last_player, done`。训练/推理两侧都只能看
  `View`（ADR-0002）；评测工具 `NeuralPolicy.act(view)`（`networks.py:478`，本轮实现后行号）每次决策
  重读完整 `View`。

### 1.2 序列能补什么：把「记牌」与「对手建模/节奏」分开

| # | 信息轴 | `unseen`+`last_player` 现状 | 序列（`played`+`trick_cards`）能补 | D1-lite 能否测 | 依赖 |
|---|---|---|---|---|---|
| a1 | **顺序/近因** | 集合，无时间戳 | 「最近打了什么」远比集合更接近对手当前手牌；对抢分/撬底残局尤其相关 | ✅ | 无 |
| a2 | **组合邻接** | 集合，无组合边界 | 同一手牌连续出现；可推断对手一次性倾泻了哪些资源（对子/顺子/炸弹），影响后续比较结构 | ✅（同墩内邻接可学，跨墩边界不可见） | 无 |
| a3 | **当前墩节奏** | 只有 `last_player` 快照 | `trick_cards` 的序 + incumbent 变化 = 本墩对抗节奏（谁在压、压了几手） | ✅ | 无 |
| a4 | **历史墩的归属/边界** | 完全没有：`played` 不存人、不存墩界（`game.py:41-45`） | 逐座位出牌史、每墩赢家、出空顺序 → 对手倾向与信念的前提 | ❌ | **H3**：`GameState.played` 改 `(seat, card)`/加 `trick_winners`，trace 回放兼容、泄漏测试扩展 |
| a5 | **对手手牌后验/belief** | `unseen` 是并集；交换性先验 `P(在某对手手)=opp_count/(opp_count+draw)` | 出牌序条件化后验（例如「对手刚过牌 → 高牌更可能仍在手」）；底牌堆顺序不可见，无法精确 | ⚠️ 只能从序间接学 | a4（完整版） |
| a6 | **对手倾向（风格）** | 无统计 | 需要 a4 的逐座位归属 + 跨局/跨墩统计 | ❌（P2 用 random 对手本来就无可学倾向） | a4 + 强对手课程 |
| a7 | **显式剩余大牌估计** | 可由 `unseen` 线性读出但 MLP 学集合运算吃力 | 不是序列，是特征工程（Q-5 的「显式剩余大牌估计」） | ❌（不在本实验） | 无（可作 v6 段） |

**关键判断**：D1-lite 能测的是 a1–a3（顺序信息），**测不到 a4–a6（归属/倾向/后验）**。
所以 D1-lite 是「序列信息」的下界测试，不是「序列记忆」的全部；如果连 a1–a3 都 null，
把 a4–a6 补上再测的性价比更低（a4 还要付 ADR-0009 兼容层/v6 的成本）。

### 1.3 预期信号在哪、为什么先做 self-play 消融

- **random 对手**：`greedy`/`random` 没有可利用的固定倾向；序列的价值只剩 a1–a3 的「记牌+
  近因」，而记牌集合 `unseen` 已覆盖大部分集合信息。前序 B1（含 `unseen`）在 random 500k
  已 null，先验上 D1-lite 在 random 也偏 null（但 B1 的 null 不直接等价于「顺序无用」）。
- **self-play 对手**：对手是自身策略的冻结快照，其行为是「观测（含真实 B1）+权重」的函数，
  存在可被序列识别的规律（例如特定残局倾向）。本仓库实测最强 500k 模型是 `self_500k`
  （[`t17-recalibration.md`](./t17-recalibration.md) §5），产品候选也在这条分布上。
- 因此：**先花 ~10 分钟**跑 self-play 的 B1 盲/有史消融（P1），回答「更强的对手分布下历史
  是否值 ≥10 Elo」；再做 D1-lite（P2）。若 P1 正、P2 也正，才进入 H3/真递归。

## 2. 主源调研（核实状态：✅=本轮下载原文/源码核实，◻=未逐字复核）

### 2.1 DouZero（arXiv:2106.06135）——历史序列编码器的直接先例 ✅

本轮下载 ar5iv 全文（`https://ar5iv.labs.arxiv.org/html/2106.06135`，本地 `/tmp/douzero.html`）
并定位到 Appendix C.1、Figure 3 的关键句，核实到的事实：

- Q-network = **LSTM 编码 historical moves** + 6 层 512 维 MLP（Figure 3 caption）；
- 历史只取**最近 15 个动作**，每 3 个连续动作拼成一个表示 → 5×162 矩阵；LSTM 取
  **最后一个 cell 的 hidden**；不足 15 步用**零矩阵补齐**；
- 训练方法是 **DMC（Deep Monte-Carlo）**，不是 PPO；该论文报告的是 DouDizhu 上的成绩，
  与本游戏的数字无关。

对本项目的含义：序列历史编码在不完全信息牌类游戏里是可行先例；但其历史包含**逐座位/逐墩
动作**（动作矩阵里有玩家维度），而本仓库 `View.played` 没有——这正是 D1-lite 与其实质差距。
「零填充 + 只读最近若干步 + 取末 hidden」的工程做法可直接借鉴（我们取最近 54 张 = 全量）。

### 2.2 cleanrl recurrent PPO（`ppo_atari_lstm.py`）——真递归路线 (b) 的参考实现 ✅

本轮从 GitHub 下载 master 版（375 行）逐段核实：

- 网络：CNN trunk → `nn.LSTM(512, 128)` → actor/critic 头（`ppo_atari_lstm.py:131-139`）；
- rollout：`next_lstm_state` 跨步携带；每个 iteration 先 clone `initial_lstm_state`（`:234`）；
  `get_states` 里用 `(1 - done)` 乘 hidden/cell，**按 episode 自动清零**（`:140-158`）；
- PPO 更新：flatten 后**按 env 抽样** minibatch（`envsperbatch = num_envs // num_minibatches`，
  `:290-311`），用该 env 的初始 hidden 重放整段序列；**没有 burn-in**，BPTT 视界 = rollout
  的 `num_steps`（截断）；
- 也就是说，recurrent PPO 的复杂度不在数学，而在 hidden 的存储/重置/重放与 minibatch
  按序列组织——这正是本仓库 `RolloutBatch`（frozen dataclass，`src/seven523/ppo.py:43-76`）
  与 shuffle-minibatch 的 `ppo_update`（`ppo.py:118` 起）要动的地方。

### 2.3 帧堆叠（frame stacking） ✅（实现锚点）/ ◻（经典出处）

- cleanrl `ppo_atari.py:105`：`env = gym.wrappers.FrameStack(env, 4)`（本轮下载核实）。
- 经典出处 Mnih et al. 2013/2015 的 4 帧 Atari 输入：**本轮未逐字复核**，只作背景引用。
- 对本项目：帧堆叠的语义是「最近 k 个**决策观测**的窗口」，而 v5 观测已经是**全公开信息**
  （除隐藏手牌），帧间差主要是记牌增量；与显式序列编码器比，它既不提供跨墩组合边界，
  也不能处理变长历史，理论信息量 ≤ D1-lite。真实成本在**评估侧的按局状态管理**（见 §3）。

### 2.4 Suphx（arXiv:2003.13590）——纠正 ✅

本轮再次下载 ar5iv 全文核实：两处 GRU 都指 `global reward predictor Φ`（「two-layer gated
recurrent unit (GRU) followed by two fully-connected layers」，Figure 7，用于把终局回报
归因到每一局），**不是已核实的策略历史编码器**。与前序报告结论一致。

## 3. 三条路线对比（本仓库实现面）

| 维度 | (a) D1-lite 无状态全历史编码器 | (b) 真递归策略（hidden 跨步） | (c) 帧堆叠 k 步 |
|---|---|---|---|
| 信息增量 | a1–a3（顺序/近因/组合邻接/当前墩节奏） | 同 (a)，但 hidden 把「跨决策状态」压成递归状态；决策点可携带任意前缀 | 最近 k 个**观测**窗口；信息 ≤ (a)，无跨墩边界、变长历史 |
| 关键改动文件/函数 | 新增 `history.py`（`encode_history`/`SequenceWrapper`）；`Agent.__init__`+`policy_logits`/`get_value`/`get_action_and_value` 加 `seqs`；`NeuralPolicy.act` 重建序列；`train.py` 存 `seqs`（int64，128×8×54）；`RolloutBatch` 透传；`save/load_agent` 记录 seq 配置 | 在 (a) 之上：rollout 存 `initial_hidden` + `dones`，PPO minibatch 按 env/序列组织并重放（cleanrl `envsperbatch` 做法）；`Train`、`ppo_update`、`RolloutBatch` 都要改；self-play 需按 (env, seat) 分 hidden 且按局重置 | 输入变 `k×161`：`Agent.obs_dim` 变宽；`make_env` 维护 deque；`NeuralPolicy` 需**有窗口状态且按局重置**；ckpt 记录 k |
| 评测链改动 | **零**：`ckpt:` spec → `load_agent` → `NeuralPolicy.act(view)` 自动重建（本轮已在 `tools/head_to_head.py` 实跑验证） | 同 (a)，但策略实例跨局复用时必须重置 hidden；`h2h` 每局新建策略的假设要显式化 | 需要策略实例每局重置窗口；`h2h`/`duel` 的策略复用路径要显式改动 |
| 默认路径回归风险 | **低**：`seq_len=0` 时无新模块、参数与数值逐位不变（回归测试已加） | 中高：rollout/PPO 主循环动刀，SAME_STEP autoreset 与自博弈刷新时序都要测试 | 中：ADPQ 观测布局检查在 `load_agent` 只查 `obs_version`，但 `NeuralPolicy` 只能造 161 维，需配套窗口逻辑 |
| 工作量（本轮实测） | ~200 LOC 源码 + 260 LOC 测试，约 1.5–2 h（含全量 631 tests 通过） | 估计 +300–600 LOC + 3–5 项时序测试；训练吞吐 −20~40% | ~100–200 LOC，但状态管理容易出静默 off-by-one |
| 预期效应量 | 先验低（B1 已 null、random 无倾向）；signals 主要在 self-play | 只有在 (a) 正、且「局部窗口不足」时才可能更大 | 预期 ≤ (a) |
| 测法 | 3 seed × 500k（random）+ 同架构盲对照；h2h 3×400 对 `t17pool` 同 seed；再补 self-play 一对 | 同左，但需先有 (a) 的基线结论 | 同左 |

**为什么推荐 (a) 作为第一枪**：ADR-0009 硬拒非 v5 布局（`load_agent` 在
`networks.py:317-322` 校验 `obs_version==5`，非 v5 抛 `ValueError`），
D1-lite 完全不碰 obs 布局/编码器；推理无状态，`NeuralPolicy.act` 每次从 `View` 重建，
`head_to_head` 的 `ckpt:` 链不用动（P2 实测已证）；训练侧只是多存一列 int64 序列。

**风险与止损**：若实现引发大面积测试失败且 2 小时内无法收敛，按任务书止损线只交付设计 +
P1 实测；本轮实际收敛（见 §5）。

## 4. P1：self-play 下的 B1 盲/有史消融（seed 1 对）

### 4.1 设计

- **对照臂**：直接复用 `runs/t17self__1__1790439615`（config：`--opponent self
  --self-play-refresh 10 --mix-random-prob 0.5 --total-timesteps 500000`；`args.json` 复核）。
- **盲臂**：`runs/t17seq_train/train_selfblind.py`（临时驱动，runs/ 内）在
  `train_histblind.py` 的基础上多打一个补丁：除了用 `ZeroHistoryObservation` 把 learner 的
  B1 列（`unseen` 54 + `last_player` 1，列区间由 `_SEGMENTS` 运行时推导）置零外，
  还把 `seven523.networks.encode_observation` 换成置零版，使**冻结的 self-play 对手**
  （`NeuralPolicy.act`）也看不到 B1。这样 learner 与冻结对手是同一个「历史盲」函数，
  否则冻结副本会带着随机初始化的 B1 列权重去看真实观测（一个真实的坑，见 §4.3）。
- 训练命令与产物：`runs/t17seq_train/commands_p1.txt`、`runs/t17selfblind__1__1790444799/`。
- 评测：`blind vs t17self__1` 同 seed、3×400 换座 h2h（统一口径）。

### 4.2 结果

P1（seed 1，`runs/t17selfblind__1__1790446441`）训练 499,712 步、无 NaN（末行
ep_return −0.006、entropy 1.486）。按 `blind_ckpt.py` 把首层 B1 列置零生成 `agent_blind.pt`
后，与 `runs/t17self__1__1790439615/agent.pt` 做 3×400 换座 h2h（左 = 盲、右 = 有史；
正 = 盲更强）：

| 端点 | Δ(盲−有史) | 95% CI | 赢率 | 说明 |
|---|---|---|---|---|
| self-play seed 1 | −9.85 | [−22.68, +2.97] | 0.4858 | boot SE 11.33 / deal-seed sd 11.12 |
| 参考：random 500k，3 seed 合并（前序报告 §3.2） | +2.89 | [−9.52, +15.31] | 0.5042 | 同口径 |

读法：self-play 下历史优势点估计 **+9.85 Elo**（CI [−2.97, +22.68]）——方向为正（与
random 阶段点估计略偏负不同），但 CI 跨 0、点估计不到 +10 门槛，且只有 1 个训练 seed。
**不能据此说 self-play 下 B1 值 ≥10 Elo**；它把「历史无显著价值」的结论从 random 扩到
self-play，但留下一个不显著的正 hint（若将来要复查，先补 seed 2/3 并考虑 5×400 确认）。

### 4.3 实现坑与正确性

1. **冻结对手必须一起盲**：盲训时 learner 的 B1 列输入恒 0、梯度恒 0，权重停在初始化值；
   而 self-play 冻结副本用**真实观测**前向，B1 列会贡献噪声 logits。不补丁
   `NeuralPolicy.act` 就不是干净的盲消融。本实验用 monkeypatch `encode_observation`
   把对手观测的 B1 也置零（与「把权重列置零」在数学上等价）。
2. 列区间不硬编码：`history_columns()` 从 `env._SEGMENTS`/`_segment_width` 推导（2 家 106:161）。
3. **评测必须用权重置零版**：盲训时 learner 的 B1 列输入恒 0、梯度恒 0，权重停在初始化值；
   标准 `NeuralPolicy` 评测会喂真实观测，未置零的 `agent.pt` 会让这些未训练的随机 B1 列重新
   生效（实测 max |w| before 0.403），不是干净的盲模型。本实验用 `blind_ckpt.py` 生成
   `agent_blind.pt`（首层 106:161 列置零）后再 h2h。

## 5. P2：D1-lite 原型 + 实测

### 5.1 实现（opt-in，默认路径逐位不变）

新增/修改（`git diff --stat` 采集）：

| 文件 | 内容 |
|---|---|
| `src/seven523/history.py`（新，71 行） | `encode_history(view, length=54)`：`view.played + view.trick_cards` → token（0=pad，`card_id+1`），oldest-first 右填充；`HistorySequenceWrapper`：每步把当前 `View` 的序列挂到 `last_seq` 供 rollout 读取 |
| `src/seven523/networks.py`（+126/−30） | `SequenceEncoder`（Embedding(55,16,padding_idx=0) + GRU(16→32)，正交初始化，取最后一个真实 token 的 hidden）；`Agent(seq_len>0, seq_emb, seq_hidden, seq_blind)` 把 trunk 首层输入扩为 `161+32`；`policy_logits`/`get_value`/`get_action_and_value` 增加 `seqs=None` 关键字；`NeuralPolicy.act` 当 `seq_len>0` 时从 `View` 重建序列（`seq_blind` 则喂全 0）；`save_agent`/`load_agent` 记录并恢复 seq 配置（旧 ckpt 缺字段默认 0） |
| `src/seven523/ppo.py`（+7） | `RolloutBatch.seqs: Tensor | None`，`flatten(..., seqs=None)`，`ppo_update` 透传 minibatch 序列 |
| `src/seven523/train.py`（+100/−30） | CLI `--seq-len/--seq-emb/--seq-hidden/--seq-blind`；`make_env(..., seq_len=)` 包 `HistorySequenceWrapper`；rollout 存 `(num_steps, num_envs, 54)` int64、bootstrap 的 `next_seqs` 同步；`RolloutBatch.flatten(..., seqs)` |
| `tests/test_history.py`（新，261 行） | 10 项：序列与 `View` 的重建一致性（rollout wrapper == `NeuralPolicy`）、截断、ckpt 往返、MLP 路径回归、`seq_blind` 不读历史、PPO 训练编码器、无 seq 时 `RolloutBatch.seqs is None`、两个端到端 smoke |

**默认路径逐位不变的保证**：`seq_len=0` 时 `Agent` 不构造 `seq_encoder`、trunk 首层维度与
旧代码一致（`tests/test_networks.py::test_arch_parameter_counts_match_the_design` 的
55,179 参数不变）；`RolloutBatch.flatten` 不传 `seqs` 时字段为 `None`，`ppo_update` 调用形状
与旧代码一致。回退版本实测：**631 passed**（基线 621 + 新增 10），见 `runs/t17seq_train/pytest_full.log`。

**训练侧与推理侧的一致性证明**（`tests/test_history.py` 核心断言）：同一 `View` 下，
`HistorySequenceWrapper.last_seq`、`encode_history(view)` 与 `NeuralPolicy.act` 内部使用的
token 逐位相同；`NeuralPolicy` 的 argmax 与手工「obs+序列」前向的 masked argmax 相同。
这覆盖了「训练时喂序列、评估时从 `View` 重建完全相同序列」的要求。

### 5.2 训练与评测

- **序列臂**：`runs/t17seq__{1,2,3}__1790445087`，命令 `runs/t17seq_train/commands_p2.txt`，
  config 镜像 `t17pool`（random 对手、500k、8×128、lr 2.5e-4 退火、hidden 128、shared），
  仅加 `--seq-len 54`。
- **容量盲对照**：`runs/t17seqblind__{1,2,3}__1790445087`，同架构同宽 trunk，`--seq-blind`
  使序列恒为全 pad（编码器与加宽首层容量保留，信息为零）——把「编码器/容量」从「序列信息」
  中扣除。
- **h2h**：每个 seed 做 `seq vs t17pool(same seed)` 与 `seqblind vs t17pool(same seed)`，
  再算 `seq vs seqblind`；3×400 换座、bootstrap 4000，按统一公式合并 3 个训练 seed。
- 命令：

```bash
# 训练（3 seq + 3 blind，并行）
bash runs/t17seq_train/commands_p2.txt   # 逐行；或见文件内 6 条命令

# h2h（每 seed 两条）；必须**串行**：head_to_head 每 worker 各开一个 CUDA context，
# 6 个 h2h × --workers 4 并发曾把 8GB 显存/15GB 内存压爆导致桌面卡死（事故记录）。
# 安全配方：一次只跑一个 h2h，用 CPU worker（慢约 27s/次，但零 CUDA context）。
.venv/bin/python tools/head_to_head.py \
  --left seq_sN=ckpt:runs/t17seq__N__1790445087/agent.pt \
  --right base_sN=ckpt:runs/t17pool__N__1790439615/agent.pt \
  --seeds 0,1,2 --pairs 400 --bootstrap 4000 --device cpu --workers 4 --json \
  > runs/h2h_seq_sN.json
# 同理 seqblind_sN vs base_sN；合并脚本见 runs/t17seq_train/aggregate_seq.py
```

### 5.3 结果

左 = 待评臂、右 = 对照；正 = 左更强。每行一个训练 seed（3×400 换座、bootstrap 4000，
原始 JSON `runs/h2h_*_s{1,2,3}.json`），合并口径同 §3 的统一公式
（mean ± 1.96·max(平均 bootstrap SE, 训练 seed sd)/√3；`runs/t17seq_train/merge_p2.txt`）：

| 比较 | seed 1 | seed 2 | seed 3 | 3 seed 合并 | boot SE / seed sd |
|---|---|---|---|---|---|
| **seq vs seqblind**（序列信息，关键端点） | +5.22 [−7.27,+17.70] | +14.35 [+1.97,+26.73] | +5.67 [−19.65,+31.00] | **+8.41 [−4.06,+20.89]** | 11.02 / 5.15 |
| seq vs t17pool | +21.21 [−3.23,+45.65] | +1.16 [−11.33,+13.64] | +14.95 [−6.31,+36.21] | +12.44 [−0.07,+24.94] | 11.05 / 10.26 |
| seqblind vs t17pool（容量对照） | +10.61 [−15.86,+37.08] | −20.08 [−47.52,+7.36] | +17.98 [+5.54,+30.41] | +2.84 [−20.00,+25.68] | 10.99 / 20.18 |

判定（README §3：CI 完全排除 0 且点估计 ≥+10 才值得行动）：

- **关键端点 seq vs seqblind = +8.41 [−4.06, +20.89]**：CI 跨 0 且点估计 <+10 →
  **没有「序列信息值 ≥+10 Elo」的证据**。单 seed 2 的 CI 排除 0（+14.35），但 seed 1/3
  没有；三个 seed 的合并点估计为正、方向一致，但幅度不达门槛。
- `seq vs t17pool` 点估计 +12.44，但 CI 下界 −0.07（触 0）；且该端点混入「加宽 trunk + 编码器
  容量」两个因素——容量盲对照 `seqblind vs t17pool` = +2.84 [−20.00,+25.68] 不可分。
- **结论：当前 pipeline（random 500k、2 家）下，D1-lite 的序列信息无可确认增益。**

### 5.4 训练健康度与正确性抽查

- **训练**：6 个臂全部 499,712 步（488 updates）、无 NaN；末行 ep_return/entropy/v_loss
  与同 seed `t17pool` 同带（例：seq s1 0.551/1.654/0.015，seqblind s1 0.425/1.680/0.025，
  对照 t17pool s1 0.505/1.755/0.013）；日志 `runs/t17seq_train/p2_*.log`。
- **编码器确实被使用**（`runs/t17seq_train/diagnose_seq.py`，60 个真实 `View`）：seq 臂
  `mean|logit(real)−logit(全 pad)| = 1.62`、`mean|logit(real)−logit(反转)| = 0.17`、
  编码特征 rms 0.716——策略对「历史里有什么」敏感，但对**顺序**本身低一个量级，机制上与
  「编码器主要学到 bag 类信息（已被 `unseen` 覆盖）」一致，也解释了为何打不过容量盲对照。
  （注：诊断脚本直接调 `policy_logits`；`seqblind` 臂在 h2h/训练路径由 `NeuralPolicy.act`
  与训练循环恒喂全 pad，所以对照是干净的。）
- **默认路径回归**：全量测试 **631 passed**（基线 621 + 新增 10；
  `runs/t17seq_train/pytest_full.log`）；`seq_len=0` 时 Agent 不建编码器、首层参数与旧代码
  相同（55,179，由 `test_networks` 参数计数断言兜底）。

## 6. 推荐、预算与判断线

基于 P1+P2 的实测：

1. **不排期「序列级记忆」**。两条线都无可确认增益：
   - 序列信息（D1-lite，capacity-matched，3 seed 合并）：+8.41 [−4.06, +20.89]；
   - 静态历史（B1）在 self-play 下：历史优势 +9.85 [−2.97, +22.68]（单 seed）。
   两者点估计都不到 +10，CI 跨 0；按统一门槛（CI 排 0 且 ≥+10 才行动）不构成依据。
2. **机制解释**：`unseen` 已是记牌的集合充分统计；D1-lite 编码器实测的顺序敏感度低
   （real-vs-reversed logit 差 0.17，vs real-vs-zero 1.62），测不出顺序增益与之一致。
   真正未被覆盖的是 a4–a6（逐座位/墩归属、倾向、后验），但它们不在 `View` 里，需要引擎把
   `GameState.played` 记成 `(seat, card)` 并引入 v6 布局，代价远高于 D1-lite。
3. **若将来仍要测（判断线）**：
   - 先决条件：先出现正信号再投入——例如 self-play/更强对手下 D1-lite 的
     `seq vs seqblind` 达到 ≥+10 且 CI 排 0（3 seed×400，建议 5 seed×400 确认）；
   - 若要做 a4–a6：先立 ADR 决定 obs v6/兼容层（ADR-0009 现在硬拒非 v5），再做
     「逐座位历史 + 盲对照」；预算按前序报告 §5.2 的 H3（~1–2 天 + 重训基线）；
   - **真递归策略（b）不应先行**：D1-lite 连顺序信息都没测出信号时，它只增加 hidden 管理
     与 BPTT 成本，不增加信息轴。
4. **工程卫生**：h2h 评测必须串行/低并发——本仓 `head_to_head` 每 worker 单开一个 CUDA
   context，`6 h2h × --workers 4 --device cuda` 会压爆 8GB 显存导致桌面卡死（§5.2 记录）。

## 7. 复现命令与产物

```bash
# P1: self-play 历史盲（learner + 冻结对手双侧盲）
.venv/bin/python runs/t17seq_train/train_selfblind.py --exp-name t17selfblind --seed 1 \
  --total-timesteps 500000 --opponent self --mix-random-prob 0.5 --self-play-refresh 10 \
  --checkpoint-interval 100 --cuda True --tensorboard False --run-dir runs
# 对照臂 runs/t17self__1__1790439615 已存在；h2h 见 §4.2

# P2: 全量测试 + 序列训练 + h2h
uv run --group train pytest -q          # 631 passed
bash runs/t17seq_train/commands_p2.txt  # 6 条训练命令（3 seq + 3 blind）

# 评测（示例）
uv run --group train python tools/head_to_head.py \
  --left seq_s1=ckpt:runs/t17seq__1__1790445087/agent.pt \
  --right base_s1=ckpt:runs/t17pool__1__1790439615/agent.pt \
  --seeds 0,1,2 --pairs 400 --bootstrap 4000 --device cuda --workers 4 --json
```

产物一览：

| 产物 | 路径 |
|---|---|
| P2 训练 6 臂（499,712 步，无 NaN） | `runs/t17seq__{1,2,3}__1790445087/`、`runs/t17seqblind__{1,2,3}__1790445087/` |
| P2 h2h JSON ×9 | `runs/h2h_{seq,seqblind,seqvsblind}_s{1,2,3}.json` |
| P2 合并 / 诊断 / 测试 | `runs/t17seq_train/{merge_p2.txt,diagnose_seq.txt,pytest_full.log}` |
| P1 盲训 seed 1（499,712 步） | `runs/t17selfblind__1__1790446441/{agent.pt,agent_blind.pt}` |
| P1 h2h / 合并 | `runs/h2h_selfblind_s1.json`、`runs/t17seq_train/merge_p1.txt` |
| 命令与安全脚本 | `runs/t17seq_train/{commands_p1.txt,commands_p2.txt,run_h2h_after.sh}` |

## 8. 局限（如实）

1. **只覆盖 a1–a3**：D1-lite 的序列没有逐座位归属与墩边界（`View` 就不提供），因此测的是
   「顺序信息」的下界；a4–a6（归属/倾向/后验）未被此实验覆盖。
2. **random 500k 分布**：P2 主臂用 random 对手（与 `t17pool` 同 config），这是 T17 的
   stage-1 分布；self-play 只做了 P1 的 B1 消融，没做 D1-lite 的 self-play 训评（预算）。
3. **分辨率**：3×400×3 seed 的合并 CI 半宽约 ±12 Elo（与前序报告一致）；10 Elo 级效应
   按 EPV 需要 1500–2000 副牌（50% power），本实验的结论是「没有 ≥+10 的证据」，
   不是「效应恰为 0」的等效检验。
4. **单架构/单超参**：GRU-32、emb-16、全量 54 长度、末 hidden 读出；没扫 attention/DeepSets/
   长度/双向等变体。容量盲对照只隔离了「加宽 trunk + 编码器参数」，没隔离 GRU 架构本身的
   优化难度（盲对照的 GRU 权重仍会被训练，只是输入恒定）。
5. **2 家结构**：`last_player` 在 2 家冗余；N 家下的序列价值（座位轮转/过牌）未测。
6. **P1 只有 seed 1 对**：任务书允许「先做 seed 1（~10 分钟），有意思再补 2/3」；因 GPU
   7 进程并行时单 run 吞吐下降，为保 P2 核心交付按该条款收缩为单 seed，并在 §4 如实标注
   单 seed 结论不可外推。

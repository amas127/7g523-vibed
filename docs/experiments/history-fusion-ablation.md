# 历史融合消融：v5 的 B1 历史通道在当前 pipeline（revision-3 / T17）值多少 Elo

> **状态：已完成（2026-09-26）。** 一句话结论：当前模型的观测 **并非「无历史」**——v5 已含
> B1 段（`unseen` 54 维记牌 + `last_player` 1 维，[ADR-0008](../adr/0008-observation-layout-v5.md)）；
> 把这段在训练与评估中等价置零后，3 个训练 seed × 3×400 副牌换座 head-to-head 的合并结果为
> **Δ(历史盲 − 有历史) = +2.89 [−9.52, +15.31] Elo**（k=3 训练种子；正 = 拿掉历史更强），
> 即 **B1 历史通道的边际价值估计为 −2.89 [−15.31, +9.52] Elo**，CI 跨 0、点估计远低于 +10 门槛 →
> **当前 pipeline 下没有证据表明融合这段历史值 ≥10 Elo，不值得为此排期**。任务 2 的可行性
> 研究与下一档设计草案见 §5（结论：GRU/LSTM 等序列记忆当前不推荐，先回答「更强对手课程」；
> 若仍要测记忆，建议不破坏 obs v5/ADR-0009 的「历史编码器 + 盲对照」最小实验）。
>
> **口径**：revision-3（出空即撬底，`rules_id=2e36dbea44893696`）、obs v5、T17 工作区；
> 所有训练/评估产物在 `runs/`（gitignore）。本文只新增本文件与
> [`README.md`](./README.md) §1 一行索引，未改 `docs/plans.md`、ADR、`CONTEXT.md`/`DESIGN.md`。

## 0. 关键结论

1. **前提纠正**：用户问的「无历史信息的 PPO」在现状里不成立。`src/seven523/env.py:186-209`
   的 `_SEGMENTS` 已含 `unseen`（54 维，历次已出牌/已现牌的公开补集，`env.py:106-122`）与
   `last_player`（当前墩归属，`env.py:124-135`）；T17 所有臂都用 v5 训练
   （[`t17-recalibration.md`](./t17-recalibration.md) §0）。2 家布局里 B1 共 55 维，
   位置为**列 106:161**（由段表运行时推导，非硬编码）。
2. **实测**：现成有历史臂 `runs/t17pool__{1,2,3}__1790439615`，同配置把 B1 输入恒置零重训 3 个
   seed（`runs/t17blind__{1,2,3}__1790443753`），训练后把 ckpt 首层对应列权重置零得到标准工具
   可评的「历史盲」模型。与同 seed 有历史臂 3×400 换座 h2h：

   | 训练 seed | Δ(盲 − 有史) Elo | 95% CI | 赢率 | deal-sign p |
   |---|---|---|---|---|
   | 1 | +8.84 | [−3.35, +21.03] | 0.5127 | 0.112 |
   | 2 | −4.21 | [−24.15, +15.73] | 0.4940 | 0.716 |
   | 3 | +4.05 | [−8.76, +16.86] | 0.5058 | 0.422 |
   | **3 seed 合并** | **+2.89** | **[−9.52, +15.31]** | 0.5042 | — |

   （合并按 [`README.md`](./README.md) §3：mean ± 1.96·max(平均 bootstrap SE 10.97, 训练 seed sd 6.60)/√3。）
3. **判定（对照 +10 门槛）**：CI 跨 0 且 |点估计| < 10（[`README.md`](./README.md) §3、
   [`evaluation-protocol-validation.md`](./evaluation-protocol-validation.md) §9）→
   **B1 历史通道在当前 pipeline 的边际价值不可分辨为正**；点估计甚至略偏负。
   把方向翻过来，历史带来的优势 95% CI 为 **[−15.31, +9.52]**，即 ≥+10 Elo 的收益基本被排除。
4. **与旧口径一致**：revision-2 的 B1（单 seed）主端点 vs `base680k` +4.20 [−6.75,+15.15]、
   相对 B0 +0.73 [−18.98,+20.43]（[`observation-augmentation-b1.md`](./observation-augmentation-b1.md) §5）。
   本次是**同一 pipeline 内、带同 seed 配对、3 训练 seed** 的更干净的消融，结论同向：
   B1 无可分辨贡献。
5. **任务 2**：还没试的是**序列级记忆**（GRU/LSTM）、帧堆叠、逐座位/逐墩归属历史、对手手牌
   后验。它们在当前 pipeline 的实现面、评测链改动、ADR 冲突与工作量见 §5.2；最推荐的下一档
   是 §5.3 的「公开出牌序列编码器 + 盲对照」（不动 obs 布局、不触发 ADR-0009）。

## 1. 背景：v5 已有 B1 历史通道

### 1.1 观测里到底有什么历史

`src/seven523/env.py` 的 v5 段表（`env.py:186-209`，2 家宽度）：

```
hand54 → inc_rank15 → inc_suit4 → inc_kind6 → inc_size1 → draw1 → scores2
→ opp_count1 → opp_revealed19 → trick_points1 → remaining_points1 → point_hold1
→ unseen54 → last_player1                                          # 2 家共 161
```

- `unseen`（`env.py:106-122`）：`54 − 自己手牌 − revealed − played − 当前墩牌` 的 multi-hot，
  即「历次真实出现的牌 + 自己手牌」的公开补集——记牌摘要；
- `last_player`（`env.py:124-135`）：当前 incumbent 的绝对座位（无 incumbent 为 0.0）。
  ADR-0008 明确注明 `last_player` 在 2 家是冗余维、为 N 家前向兼容保留；
- `GameState.played`（`game.py:41-45`）保存**所有已结束墩的牌**（出牌顺序），
  `_end_trick` 里 `played = state.played + state.trick_cards`（`game.py:243`）；
  当前墩不分座位归属，只作为公开事实保存。
- B1 由 [ADR-0008](../adr/0008-observation-layout-v5.md) 决定并入默认布局；现场复核 `runs/t17*`
  下（排除本次新增的 `t17blind`）24 个 `.pt` 文件全部 `obs_version=5`、`obs_dim=161`。

因此「无历史信息的 PPO」不成立；真正的问题只剩：**这段已融合的 B1 值多少 Elo？进一步
的序列记忆还差什么？** 这正是任务 1/2。

### 1.2 已有的历史证据

- revision-2 的 B0（点分量 3 维，null）与 B1（+55 维）500k 单臂：
  B1 vs `base680k` **+4.20 [−6.75,+15.15]**（主端点失败）、vs `w5_ctrl` +13.04 [+1.93,+24.15]、
  相对 B0 **+0.73 [−18.98,+20.43]**；单训练 seed、判定「未确认、不采用」，B 线收束
  （[`observation-augmentation-b1.md`](./observation-augmentation-b1.md) §5–§6、
  [`README.md`](./README.md) §0 第 8 条）。
- 该实验早于 v5 迁移，两个模型各自用旧编码器评估；本次在同布局、同工具、同 seed 下重做。
- 仍未做过的观测记忆档位：GRU/LSTM 序列策略、帧堆叠、逐座位/逐墩历史特征、对手手牌后验
  （[`../plans.md`](../plans.md) Q-5 = [`structural-directions.md`](./structural-directions.md)
  §10 Q5；详见 §5）。

## 2. 实验设计

### 2.1 配对与配置复核

对照臂采用当前 canonical 500k 臂：`runs/t17pool__1__1790439615`、`__2__`、`__3__`
（random 对手、499,712 步、obs v5；训练命令见 `runs/t17_train/commands.txt`）。
逐项核对：三个 `args.json` 除 `seed`/`exp_name` 外逐位相同（`diff` 验证），
且本次盲训 `runs/t17blind__N__1790443753/args.json` 与 `t17pool` 同 seed 的 args
除 `exp_name` 外也逐位相同。关键超参：`num_envs=8`、`num_steps=128`、`lr=2.5e-4`
（线性退火）、`hidden_size=128`、`arch=shared`、`opponent=random`、`reward_shaping=terminal`、
`total_timesteps=500000`（实际 499,712 = 488 updates）、`checkpoint_interval=8`、`cuda=True`。

### 2.2 历史盲训练（训练时把 B1 列置零）

用临时驱动 `/tmp/train_histblind.py`（import `seven523.train` 后 monkeypatch 模块级
`make_env`，`train()` 内部为全局查找，`train.py:264`、`train.py:416`）在
`Seven523Env` 外套一个 `gym.ObservationWrapper`，对每步观测把 B1 列置零。
列区间由段表运行时推导（`_SEGMENTS` + `_segment_width`，`env.py:210-218`），
不硬编码；2 家得到 `slice(106, 161)`，3 家为 `slice(127, 182)`（脚本输出）。

训练命令（3 个 seed 并行，产物 `runs/t17blind__{1,2,3}__1790443753`；
命令留档 `runs/t17blind_train/commands.txt`）：

```bash
.venv/bin/python /tmp/train_histblind.py --exp-name t17blind --seed {1,2,3} \
  --total-timesteps 500000 --opponent random --checkpoint-interval 8 \
  --cuda True --tensorboard False --run-dir runs
```

墙钟：`args.json` 13:29:13 → `agent.pt` 13:35:5x（3 个并行 ~6m40s）。

### 2.3 训练产物必须是「可换座评测的历史盲模型」

标准评测工具喂的是**真实观测**（B1 列非零），而盲训时首层 B1 列权重的梯度恒为 0
（输入为零），训练结束时它们停在正交初始化值上——直接用 `agent.pt` 评测会被随机初始化
权重污染。因此训练后用 `/tmp/blind_ckpt.py` 把 `network.0.weight[:, 106:161]` 置零并存为
`agent_blind.pt`（`obs_dim`/`obs_version` 原样保留，`load_agent` 可加载）。
评测一律用 `agent_blind.pt`。

**正确性验证（三项，全部通过）**：

1. **wrapper 生效（2-update 对照）**：同 seed 同配置，正常训练 2 updates 后历史列相对初始化
   `|Δw| = 5.60e-3`，盲训后 `|Δw| = 0`（非历史列两者都动）；
2. **500k 全程零梯度**：三个盲训 `agent.pt` 的历史列相对各自 seed 的初始化 `|Δw| = 0`，
   非历史列 `|Δw| = 0.198/0.234/0.214`；
3. **置零变换与原模型逐位一致**：对 64 个真实观测（随机策略采样），
   `max |logits(原始 ckpt, 历史列置零的 obs) − logits(agent_blind, 真实 obs)| = 0`。

### 2.4 评测协议

`tools/head_to_head.py`（[`README.md`](./README.md) §3 常用命令口径），左 = 盲、右 = 有历史，
`--seeds 0,1,2 --pairs 400 --bootstrap 4000 --device cuda --workers 4`；每个训练 seed 一次
（每次 3×400=1200 副牌 / 2400 局，同牌换座、按牌聚簇 bootstrap）。命令：

```bash
for s in 1 2 3; do
  .venv/bin/python tools/head_to_head.py \
    --left blind_s$s=ckpt:runs/t17blind__${s}__1790443753/agent_blind.pt \
    --right hist_s$s=ckpt:runs/t17pool__${s}__1790439615/agent.pt \
    --seeds 0,1,2 --pairs 400 --bootstrap 4000 --device cuda --workers 4 --json \
    > runs/h2h_histblind_s$s.json
done
```

3 seed 合并（训练种子层）用统一公式
`mean ± 1.96·max(平均 bootstrap SE, 训练 seed sd)/√k`，k=3、sd 用 n−1
（与 [`reward-shaping-500k.md`](./reward-shaping-500k.md) §3.4 的 4-seed 平均同口径）。

## 3. 结果

### 3.1 训练健康度

| run | updates / steps | 末行 ep_return | entropy | v_loss | NaNs |
|---|---|---|---|---|---|
| `t17blind__1` | 488 / 499,712 | 0.5717 | 1.699 | 0.0111 | 0 |
| `t17blind__2` | 488 / 499,712 | 0.4786 | 1.843 | 0.0168 | 0 |
| `t17blind__3` | 488 / 499,712 | 0.5478 | 1.750 | 0.0132 | 0 |
| `t17pool__1`（对照） | 488 / 499,712 | 0.5047 | 1.755 | 0.0134 | 0 |
| `t17pool__2`（对照） | 488 / 499,712 | 0.5705 | 1.648 | 0.0170 | 0 |
| `t17pool__3`（对照） | 488 / 499,712 | 0.5386 | 1.671 | 0.0151 | 0 |

训练曲线无异常（v_loss/entropy 与对照同带内；这些是健康度指标，不做强度比较）。

### 3.2 换座 head-to-head（原始数字）

左 = `agent_blind`、右 = 同 seed `t17pool` final；**正 = 盲更强**。每个训练 seed 的三行 deal-seed
是该 h2h 内部的三个独立牌集（各 400 副、2400 局合并前），列出以展示牌集方差。

| 训练 seed | deal-seed | Δ(盲−有史) Elo | 95% CI | 赢率 | W/D/L |
|---|---|---|---|---|---|
| 1 | 0 | +19.13 | [−2.17, +40.57] | 0.5275 | 393/58/349 |
| 1 | 1 | −0.43 | [−21.74, +20.87] | 0.4994 | 369/61/370 |
| 1 | 2 | +7.82 | [−12.60, +28.73] | 0.5112 | 382/54/364 |
| **1 合并** | k=3 | **+8.84** | **[−3.35, +21.03]** | 0.5127 | boot SE 10.77 / 牌集 sd 9.82 |
| 2 | 0 | −23.92 | [−45.42, −2.17] | 0.4656 | 337/71/392 |
| 2 | 1 | +1.30 | [−20.00, +21.74] | 0.5019 | 373/57/370 |
| 2 | 2 | +9.99 | [−10.86, +31.35] | 0.5144 | 376/71/353 |
| **2 合并** | k=3 | **−4.21** | **[−24.15, +15.73]** | 0.4940 | boot SE 10.82 / 牌集 sd 17.62 |
| 3 | 0 | +6.52 | [−15.65, +27.85] | 0.5094 | 384/47/369 |
| 3 | 1 | +4.78 | [−17.82, +26.99] | 0.5069 | 374/63/363 |
| 3 | 2 | +0.87 | [−21.31, +23.49] | 0.5012 | 368/66/366 |
| **3 合并** | k=3 | **+4.05** | **[−8.76, +16.86]** | 0.5058 | boot SE 11.32 / 牌集 sd 2.89 |
| **3 seed 合并** | 9 deal-seeds | **+2.89** | **[−9.52, +15.31]** | 0.5042 | 平均 boot SE 10.97 / 训练 seed sd 6.60 |

产物：`runs/h2h_histblind_s{1,2,3}.json`（含 `per_seed` 与 `combined`）、日志
`runs/t17blind_train/h2h_s{1,2,3}.log`。

### 3.3 判定

统一判定口径（[`README.md`](./README.md) §3、EPV §9）：**CI 完全排除 0 且点估计 ≥+10 才值得行动**。

- 3 个训练 seed 的合并 CI **[−9.52, +15.31]** 跨 0；点估计 +2.89（盲略强）< +10；
- 单 seed 也无一达标：seed 1 点估计 +8.84 但 CI 跨 0；seed 2 反向；seed 3 接近 0；
- 把方向翻成「历史优势」= −(盲−有史)：**−2.89 [−15.31, +9.52]**，≥+10 Elo 的收益几乎
  被 95% CI 排除；数据与「B1 值 0」完全一致，也不排除历史小幅为负。

**直接回答用户的问题：在当前 pipeline 下，把已经存在的 B1 历史通道加回去（相对历史盲）
的等级分增益估计为 −2.9 Elo，95% CI [−15.3, +9.5]——不可分辨、远低于值得行动的门槛。**
也就是说，当前模型（v5）里的历史信息对 500k/random 阶段的强度没有可测贡献；没有历史的
模型不会更弱。

## 4. 局限（如实）

1. **只覆盖 stage-1（random 对手、500k、从零）**：T17 池里实测最强的是 `self_500k`
   （[`t17-recalibration.md`](./t17-recalibration.md) §5），本实验没有做 self-play 或更强对手下的
   历史消融；「B1 无用」不能直接外推到自博弈/强对手课程（对随机对手，记牌价值可能天然更低）。
2. **分辨率**：合并 CI 半宽 ±12.4 Elo；按 EPV，10 Elo 级效应仅 50% power 就需 1500–2000
   副牌、80% power 约需 2×（3000–4000 副），且这些数字假设单一训练模型、不含训练 seed
   抽样项（我们 3600 副分散在 3 个训练 seed）。因此本报告的结论是「没有 ≥10 的证据」，
   不是「效应恰为 0」的等效/非劣检验。
3. **训练 seed 只有 3 个**：训练 seed sd 6.60 由 3 点估计；若训练间真实 sd 更大，
   合并 CI 会更宽（公式已用 max 保守）。
4. **2 家结构**：`last_player` 在 2 家冗余，本实验实际测的是 54 维 `unseen`（记牌）通道；
   N 家的 B1 结论未测。
5. **单臂配对**：只比了「同 seed 盲 vs 有史」；未做盲×盲、有史×有史的额外方差对照，
   但 per-deal-seed 表已显示 400 副牌的单次读数可给出 |Δ|>20 的假象（seed 2 deal-seed 0），
   与 README 的多 seed 要求一致。
6. **训练数据量固定 500k**：更长预算（2M）下历史价值是否显现未测；不过 T17 筛选中
   512k/499k/1M/1.5M/2M 五个顶部点 μ 187.6–191.6、两两不可分辨，该怀疑的先验较低。

## 5. 任务 2：融合历史信息的可行性研究

### 5.1 现状盘点：已有什么、还差什么

已融合（B1，见 §1）：记牌摘要 `unseen`、当前墩归属 `last_player`；另有当前墩点数、
剩余点数、自己持分、亮牌、手牌数等当前态（B0/S1 段）。

尚未融合：

| # | 缺口 | 证据 |
|---|---|---|
| 1 | **序列级记忆/递归策略**（GRU/LSTM/attention 读历史动作序列） | [`../plans.md`](../plans.md) Q-5；[`structural-directions.md`](./structural-directions.md) §10 Q5；ADR-0003 的「纯 MLP + 全观测」契约 |
| 2 | **帧堆叠**（同一局内前 k 个决策观测拼接） | 未在任何报告/计划出现（grep 无记录） |
| 3 | **逐座位/逐墩历史归属**：`played` 只存牌不存人（`game.py:41-45`）；`View.played` 同样（`game.py:105`），无法区分「哪家打了什么」、也拿不到各墩归属节奏 | `trace.py:140-151` 只在 trace 里记录每步 `seat+action`（回放可重建，live view 不可） |
| 4 | **对手手牌后验/belief**：`unseen` 是「对手手牌 ∪ 底牌堆」的并集（`env.py:106-122`），不编码分布；`draw` 与各家手牌数公开，交换性假设下 `P(牌在某对手手上（任一家）) = opp_total/(opp_total+draw_count)` 可推但未编码 | `_write_unseen` 注释、ADR-0002 差分泄漏约束 |
| 5 | **对手倾向**（打法风格统计） | 需要 #3；当前对手是 random，没有可学的倾向 |
| 6 | **显式剩余大牌估计**（按 rank 的 unseen 直方图/阈值计数） | Q-5 的「显式剩余大牌估计」；可从 `unseen` 线性推出，但 MLP 学集合运算吃力（B1 报告 §2.1 的设计动机） |

外部参照（联网核实，2026-09-26）：

- **DouZero**（arXiv:2106.06135，ar5iv 正文复核）：Q-network = **LSTM 编码历史出牌**
  + 6 层 512 维 MLP；历史不足 15 步用零矩阵补齐。→ 序列历史编码在同类不完全信息牌类
  游戏中被使用（该论文报告的是 DouDizhu 上的成绩，不是本游戏的数字）。
- **Suphx**（arXiv:2003.13590，ar5iv 正文复核）：两层的 GRU 用于 **global reward predictor Φ**
  （把终局回报归因到每局），不是已核实的「策略网络历史编码器」。不要把「Suphx 用 GRU
  做策略记忆」当已核实事实。
- 本仓库当前没有 GRU/LSTM 代码（`grep` 只在 [`../plans.md`](../plans.md) Q-5 与
  [`structural-directions.md`](./structural-directions.md) §10 Q5 两处提到该词）。

### 5.2 各方向的实现面、评测链改动与 ADR 冲突

| 方向 | 观测布局 | 网络/训练改动 | 评估工具改动 | ADR/契约冲突 | 工作量（含 500k 3 seed + 3×400 h2h） |
|---|---|---|---|---|---|
| **H1 序列策略（GRU/LSTM）** | 不变（obs v5 161） | `networks.Agent` 加递归编码器；PPO rollout 存 hidden、按 episode 重置（`train.py:416` 起的 rollout、`ppo.RolloutBatch` 需按序列而非乱序 minibatch）；`save/load_agent` payload 扩展 | `NeuralPolicy.act` 变成有状态：按局重置、self-play 共享策略需按 env 分 hidden（`networks.py:372` 起）；`head_to_head` 的 policy 工厂每局每座新建（`ladder.py:396`）天然重置，但训练侧对手路径要改 | 破坏 ADR-0003 「纯 MLP + 全观测」/Q-5；obs 布局不变所以**不触发** ADR-0009 的单一布局规则 | 训练 1.5–2× 慢；~300–600 LOC + 测试；风险最高 |
| **H2 帧堆叠 k 步** | 输入变 k×161，obs v5 语义不变；训练 wrapper 维护窗口 | `train.py` 存窗口；`save/load_agent` 记 k；热启动需列复制 | `NeuralPolicy` 维护按局窗口（策略实例每局新建，改动不大）；`head_to_head` 无需改统计 | 不触发 ADR-0009（同一 encoder、不同窗口），但 ckpt `obs_dim` 变了，跨 v5 热启动要列复制 | ~100–200 LOC；最便宜；信息量≈「最近几步的增量」，无逐座位归属 |
| **H3 逐座位/逐墩历史（v6 布局）** | 新段，`OBS_VERSION=6`（例：per-seat 出牌 rank 直方图、每座位最近一手、各墩归属、unseen-by-rank 15 维） | `game.GameState.played` 改 `(seat, card)` 或另存 `trick_winners`；trace 回放继续 `played=()`；差分泄漏测试扩展（`tests/test_env.py:395`、`tests/test_game.py:178`） | **关键冲突**：`load_agent` 对非 v5 硬拒绝（`networks.py:228-243`，ADR-0009 决定 3），`NeuralPolicy` 只调全局 v5 `encode_observation`。要么重引入按 `obs_version` 的编码器 dispatch（ADR-0009 明确删掉的兼容层），要么放弃与 T17 现有 ckpt 的 h2h、在 v6 内重训基线 | 直接修订 ADR-0009；新 ADR 必要 | 引擎+布局+测试 ~1–2 天；再训练基线+candidate |
| **H4 显式大牌估计/信念特征** | 可作为 v6 新段，或 H1 编码器的额外输入 | 纯函数特征（公开量），无引擎改动；泄漏测试 | 随 H3/H1 | 随 H3/H1 | 特征工程 ~半天（基础设施就绪后） |
| **H5 对手倾向特征** | v6 或编码器 | 需要 H3 的逐座位历史 + 在线统计 | 随 H3 | 随 H3 | 研究性，不建议现在做 |

**评测链改动要点（无论哪条路）**：现有 h2h 工具链冻结（README §3 C-5），
`head_to_head`/`duel` 只接受 ckpt spec；任何「策略输入变化」都必须让
`load_agent` 能重建、`NeuralPolicy` 能在真实 `View` 上喂同样的输入，
否则评测数字不能与 T17 池同台。

### 5.3 下一档最小可执行设计（建议：D1-lite，不破坏 obs v5）

**目标**：回答「序列历史（而不是记牌集合摘要）是否有增益」，同时把「容量/架构」与
「历史信息」分离。

- **输入**：保持 env obs v5 = 161 维不变（ADR-0009 零冲突）。新增第二输入：
  `View.played`（`game.py:105`，牌序公开）的序列——直接给每步决策喂「至今所有已结束墩的
  出牌序列」（≤54 张，按序），当前墩牌序也可并入（`trick_cards` 顺序公开，`game.py:206`）。
- **网络**：card-id embedding + 单层 LSTM(32)（或小 Transformer/DeepSets），输出 32–64 维，
  与 161 维 obs 拼接进原 MLP（首层 193–225）。**推理是无状态的**：每次 `act(view)` 重新读
  完整序列，不跨决策携带 hidden——避开 recurrent PPO 的全部复杂度。
- **训练**：rollout buffer 额外存 `(steps, envs, 54)` 的 int8 padded 序列（128×8×54 字节 ≈ 55KB，
  可忽略）；`ppo.RolloutBatch` 透传，minibatch 重组时批量 embedding。
- **评估**：`NeuralPolicy` 从 `view.played` 构造序列即可；`head_to_head` 统计链零改动。
- **盲对照**：同架构第二臂把序列输入置零（不喂序列），同一训练种子——把「序列编码器容量」
  从「历史信息」中扣除；再与 v5 基线（`t17pool` 同 seed）三方比较。
- **预算**：实现 ~200–400 LOC + 测试（~1 天）；训练/评估 ~半天（3 seed × 500k 并行
  ~7 min + 每比较 3×400 h2h ~3 min）→ 总计 1–2 天。
- **验收**：与 §3.3 相同门槛（≥3 seed × 400、CI 排 0 且点估计 ≥+10）；预期收益先验低
  （B1 已 null、T17 顶部 512k–2M 平台），属于「确认性」而非「开拓性」实验。
- **排除项**：v6 布局（H3/H4）只有在 D1-lite 或某个序列实验出现正信号后再投入，
  否则先付 ADR-0009 兼容层的成本不划算。

### 5.4 建议（基于任务 1 实测）

1. **不为「历史融合」单独立项**：B1 的边际价值 −2.9 [−15.3, +9.5]，且当前池的瓶颈证据
   指向别处（T17 筛选中 512k–2M 顶部平台互不分辨、A/B/C/F 全 null、6 方案 ±11 Elo 带内）。
2. **如果还要测记忆，先测 self-play 下的消融**（最便宜、且最可能改变结论）：
   用 `--opponent self` 在盲/有史两侧各 1–3 seed、500k（同 t17self 配置），再 h2h。
   理由：random 对手不提供可利用的倾向信息，`unseen` 的价值可能被低估；
   T17 实测最强的模型是 `self_500k`，这才是产品候选所在的分布。
3. **若 self-play 消融也 null**：按 D1-lite 做一次「序列编码器 + 盲对照」即可封口；
   不建议直接上 recurrent PPO 或 v6 布局。
4. **若 D1-lite 为正**：再按 H3→H4 的顺序补逐座位归属与信念特征，并同期决定是否
   恢复 `obs_version` dispatch（修订 ADR-0009）。

## 6. 复现命令与产物

```bash
# 1) 历史盲训练（3 seed 并行；列区间由 env._SEGMENTS 推导）
.venv/bin/python /tmp/train_histblind.py --exp-name t17blind --seed {1,2,3} \
  --total-timesteps 500000 --opponent random --checkpoint-interval 8 \
  --cuda True --tensorboard False --run-dir runs

# 2) 权重置零（生成标准工具可评的历史盲 ckpt）
.venv/bin/python /tmp/blind_ckpt.py runs/t17blind__N__1790443753/agent.pt \
  runs/t17blind__N__1790443753/agent_blind.pt

# 3) 正确性验证（wrapper 零梯度 + 置零变换逐位一致）
#    见本文 §2.3；脚本 /tmp/verify_blind.py

# 4) 换座 h2h（每训练 seed 一次；左=盲、右=有历史）
.venv/bin/python tools/head_to_head.py \
  --left blind_sN=ckpt:runs/t17blind__N__1790443753/agent_blind.pt \
  --right hist_sN=ckpt:runs/t17pool__N__1790439615/agent.pt \
  --seeds 0,1,2 --pairs 400 --bootstrap 4000 --device cuda --workers 4 \
  --json > runs/h2h_histblind_sN.json

# 5) 3 训练 seed 合并：mean ± 1.96·max(平均 bootstrap SE, seed sd n−1)/√3
```

产物一览：

| 产物 | 路径 | sha256（前缀） |
|---|---|---|
| 历史盲训练 ×3 | `runs/t17blind__{1,2,3}__1790443753/{agent.pt,args.json,metrics.csv}` | — |
| 历史盲评测 ckpt ×3 | `runs/t17blind__{1,2,3}__1790443753/agent_blind.pt` | `5b9f9930…`, `c8171521…`, `80371dc2…` |
| h2h JSON ×3 | `runs/h2h_histblind_s{1,2,3}.json` | `009d4d32…`, `dbe5f100…`, `19650999…` |
| 训练/h2h 日志与命令 | `runs/t17blind_train/{commands.txt,seed*.log,h2h_s*.log}` | — |
| 对照臂 | `runs/t17pool__{1,2,3}__1790439615/agent.pt` | `f7dbaa0d…`, `47b6a81e…`, `e97aa440…` |

临时脚本（不进 git，`/tmp` 下运行；已备份到 gitignore 的 `runs/t17blind_train/`）：
`train_histblind.py`（wrapper + monkeypatch `make_env`）、`blind_ckpt.py`（首层 B1 列置零）、
`verify_blind.py`（逐位一致性）、`aggregate_h2h.py`（合并计算）；上列命令中的 `/tmp/…`
可替换为 `runs/t17blind_train/…`。核心逻辑已内联在本文 §2.2/§2.3 描述中，可原样重建。

## 7. 附录：临时脚本要点

```python
# /tmp/train_histblind.py（节选）
def history_columns(num_players: int) -> slice:
    offset = 0; start = None
    for segment in _SEGMENTS:                     # env.py:186
        if segment.name == "unseen":
            start = offset
        offset += _segment_width(segment, num_players)   # env.py:210
    return slice(start, offset)                   # 2 家: 106:161

class ZeroHistoryObservation(gym.ObservationWrapper):
    def observation(self, obs):
        out = np.array(obs, dtype=np.float32, copy=True)
        out[self._columns] = 0.0                  # unseen + last_player
        return out

# /tmp/blind_ckpt.py（节选）
payload = torch.load(src, map_location="cpu", weights_only=True)
for key in model:                                 # network.0.weight（shared arch）
    if key.endswith(".0.weight"):
        model[key][:, history_columns(2)] = 0.0   # obs_dim/obs_version 不变
torch.save(payload, dst)
```

**工作区状态**：`git status` 只新增本文件与 `docs/experiments/README.md` 的 §1 一行索引；
`runs/**` 为 gitignore 产物；未 commit、未改 `docs/plans.md`/ADR/`CONTEXT.md`/`DESIGN.md`。

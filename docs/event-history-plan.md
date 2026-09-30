# 事件级历史序列 + 相对座位：设计、判据与长程实施规划（7鬼523 / PPO）

> **状态：规划（未实现），已按红队 findings 修复（2026-09-26，revision: r2）。**
> **后续状态（2026-09-26 实施、2026-09-27 判定）**：本规划已按预注册落地并完成 pilot；
> 主端点 `event vs event_noise` 为 null（random **−5.02**、self-play **+2.99**，CI 跨 0，不 ship），
> 结论、偏差与产物见 [`experiments/event-history-pilot.md`](./experiments/event-history-pilot.md)。
> 本文保留为预注册与设计记录（历史），不再更新；实施后的代码形状以 pilot §1/§3 为准。
> 本文只规划，不改 `src/`、`tests/`、`tools/`、ADR、`plans.md`。
> **修复记录**见文末「附录 C 对抗性审查与修复记录」；findings 全文与证据见
> [`event-history-review.md`](./event-history-review.md)。
> **口径**：revision-3（出空即撬底，`rules_id=2e36dbea44893696`）、obs v5（2 家 161 维）、
> 当前 T15–T17 工作区。所有代码论断给 `file:line`。
> **已拍板的三条 + 一条工程约束**（本文不得推翻，只做落地细化）：
> 1. 事件级 token（B 方案）：一次出牌的组合先压成一个事件向量，**不**做「每张牌一个
>    `(seat, card)` token」的 A 方案。
> 2. 无状态重算：每次决策把完整公开事件前缀重喂时序模型，取 hidden 拼入 trunk；hidden
>    不跨决策携带。真递归（携带 hidden / BPTT）推迟。
> 3. 每家一个 seat embedding，**相对 acting seat**；事件向量与 seat embedding 结合；
>    时序模型编码事件序列；hidden 走现有「第二输入」管线拼 obs v5，**不改 obs 布局、
>    不触发 ADR-0009**。
> 4. 新功能 opt-in；默认路径（`seq_len=0` 且 `event_len=0`）逐位不变。
>
> **资源纪律**：训练任何时刻最多 **3 进程并发**；h2h **严格串行**且只允许
> `--device cpu --workers 4`（6 个 h2h × `--workers 4 --device cuda` 并发曾把 8GB 显存 /
> 15GB 内存压爆、桌面卡死重启，见
> [`experiments/sequence-memory-pilot.md`](./experiments/sequence-memory-pilot.md) §5.2/§6.4）。
>
> **用户已拍板决策（2026-09-26，覆盖本文默认/条件项，实施必须执行）**：
> 1. **从零训练**：删除「零填充中性起点」warm-start 消融（§4.5 默认删已落实），不恢复
>    ADR-0009 已退役的列重映射，不新立 ADR。
> 2. **self-play 主端点为本次必做**：先完成 `league.py` 冻结对手第二输入配置透传 + self-play
>    smoke，再训 self-play 的 `event`/`event_noise` × 3 并评测主端点；顺序在 random 主端点
>    （步骤 6）之后。「迁移」（真递归 / 更大规模）本次不做，留给后续（§8.1）。
> 3. **`event_blind` 不训**：保留开关代码与测试即可，不出 3 个 run；主对照是 `event_noise`。
> 4. random 第一波含 `event`/`event_noise`/`event_seatblind` × 3 照做（Q-B 归属端点需要
>    seatblind）；下文「条件阶段」措辞均以本块为准。

---

## 0. 规划摘要

1. **要回答的问题**：把「公开出牌历史」从 D1-lite 的**扁平牌序**升级为**事件级 + 相对座位 +
   墩边界**的表示后，2 家 7鬼523 的 PPO 策略能否拿到可测的 Elo 增量；增量相对于
   `unseen`/`last_player`（v5 已有，`src/seven523/env.py:106-135`）、D1-lite 序列臂、以及
   信息为零但通道非退化的对照臂分别是多少。
2. **核心赌注是 a4（归属 + 墩边界 + 出空序）**：这条轴**不在 `View` 里**
   （`GameState.played` 只存牌、不存人/墩界，`src/seven523/game.py:41-45`），
   D1-lite 测不到它（[`sequence-memory-pilot.md`](./experiments/sequence-memory-pilot.md) §1.2）。
   事件级 token 是拿到它的最低成本路径，且不碰 obs 布局。
3. **先验：接近零/未知，必须按探索性处理**（红队 findings `ml-stats-04`、`contracts-B1-evidence-misuse`）。
   前序所有正先验都来自**顺序/编码（a1–a3）**，不是**归属（a4）**：
   - 最相关的正面数字是 chrono vs sorted 的 **+10.46 [−2.46,+23.38]**（顺序轴，
     [`sequence-gru-ablation.md`](./experiments/sequence-gru-ablation.md) §3.2）；
   - D1-lite 的信息端点 `seq vs seqblind` 只有 **+8.41 [−4.06,+20.89]**（CI 跨 0）；
   - self-play 下 B1 的 +9.85 [−2.97,+22.68] 是 **v5 自身静态历史段**、**单 seed**、且与序列/
     事件编码正交的弱间接 hint（pilot §4.2、§8.2；[`history-fusion-ablation.md`](./experiments/history-fusion-ablation.md)
     的 3-seed random 结果反而是 +2.89 [−9.52,+15.31]），**不能**当作 a4 的证据。
4. **第一枪**：random 500k、3 训练 seed，训练 `event` 与**信息为零但通道非退化**的对照
   `event_noise`，h2h 关键端点 `event vs event_noise`。若点估计 <+5 且 CI 跨 0 → **停并发布
   null**（口径见 §1.5 的 z/t 双报告）；其余情形按 §1.5 加 seed。self-play 为本轮**必做**
   （用户拍板），顺序在 random 主端点之后、且需先修 `league.py`（见 §1.3/§6 步骤 7）。
5. **工程量**：引擎加一个公开 action log（`GameState.plays`/`View.plays`）+ 新编码器 + 新
   `Agent` 分支 + rollout/PPO 透传 + **`league.py` 冻结对手透传**；与 D1-lite 同量级偏上
   （D1-lite 约 200 LOC 源码 + 260 LOC 测试，pilot §5.1），时序更长、张量更宽，**吞吐**（不是
   内存）是主要代价（§7.1）。真递归不在本方案内（§8.1）。

---

## 1. 研究问题与判据

### 1.1 研究问题

> 在 2 家、revision-3、obs v5 固定不变的前提下，把公开历史表示为
> **「事件 = (相对座位, 组合牌集, 是否开墩, 是否出空)」** 的序列，交给一个**无状态重算**的
> 时序编码器，取 hidden 拼入 trunk，是否能比 v5 的 `unseen`/`last_player` 摘要、
> 以及比 D1-lite 的扁平牌序，带来可测的策略强度增量？

三个子问题：

- **Q-A（信息）**：事件表示相对**信息为零但同一编码器/通道非退化**的对照有没有信息增益？
  （**主端点，confirmatory**）
- **Q-B（归属）**：增益是否来自**相对座位归属**？（`event` vs `event_seatblind`；该对照在 2 家
  下偏保守，见 §5.1）
- **Q-C（相对已有序列）**：相对 **D1-lite chrono 臂**是否有增量？（`event` vs `seq`；**该端点
  同时改表示、编码器容量与序列长度，属混淆端点**，只作描述，不用于机制归因。）

### 1.2 端点、对照臂与预注册（修复 `ml-stats-01/03/12`、`contracts-*`）

统一口径（repo 行动门槛，见 [`experiments/README.md`](./experiments/README.md) §0.2）：

- **预注册的 confirmatory 端点族**（固定 α=0.05，族内 Bonferroni，k=2 个）：
  1. **主端点（Q-A）**：`event vs event_noise`（random，3 seed 合并）；
  2. `event vs event_seatblind`（Q-B，random）。
- **主端点是唯一用于「停/继续」的判定**（§1.5）；Q-B 为**次要**：Bonferroni 后单侧 α≈0.025，
  或明确标注为 exploratory、不进决策门。
- `event vs seq`（Q-C）**不在 confirmatory 族**：它同时改变表示、编码器容量与序列长度（§4.3），
  是**混淆端点**，只作**描述性**报告，不用于机制归因、不进决策门。
- **报告端点（不进 confirmatory 族，仅描述）**：`event vs seq`、`event vs event_blind`、
  `event vs t17pool`、机制臂（shuffle/boundaryblind/nopass）、self-play 分布。

| 角色 | 端点（左 − 右，正 = 左更强） | 回答 | 对照臂来源 | 归因强度 |
|---|---|---|---|---|
| **主端点（信息）** | `event` vs `event_noise` | Q-A | 本轮新训 | 干净（信息，容量非退化） |
| 主端点 2（归属） | `event` vs `event_seatblind` | Q-B | 本轮新训 | 保守（见 §5.1） |
| 次端点（vs MLP） | `event` vs `t17pool`(same seed) | 综合（信息+容量） | 复用 `runs/t17pool__*` | **混淆**，仅描述 |
| 次端点（vs D1-lite） | `event` vs `seq`(same seed) | Q-C | 复用 `runs/t17seq__*` | **混淆**（表示+容量+长度），仅描述 |
| 配对控制（复用） | `event` vs `event_blind` | seed-matched 配对（降方差） | 本轮新训（可选） | **≈ event vs 重训 MLP**，不隔离容量 |
| 机制端点（顺序） | `event` vs `event_shuffle`（墩内确定置换，§5.1） | 顺序/近因 | 本轮新训（可选） | 机制 |
| 机制端点（墩边界） | `event` vs `event_boundaryblind`（置零 `opens_trick`） | 墩边界 | 本轮新训（可选） | 机制 |
| 机制端点（pass） | `event` vs `event_nopass`（可选） | pass/节奏 | 本轮新训（可选） | 机制 |
| 分布复算 | 上述在 random 与 self-play 各做一遍 | 分布依赖 | self-play 臂新训（必做，步骤 6 后） | 必做 |

**对照臂的语义（硬性，修复 `ml-stats-01`/`engineering-event-blind-not-capacity-control`）**：

- `event_blind`（全零事件）**不是容量对照**。实测量：`SequenceEncoder` 对全 pad 的输出恒为
  `0.0`（`.venv/bin/python` 实测 `SequenceEncoder(16,32)(torch.zeros(4,54))` absmax=0.0），
  `Agent(...,seq_len=54,seq_blind=True)` 的 `network.0.weight[:,161:]` 梯度恒为 0.0
  （`networks.py:101` 的 `padding_idx=0` + GRU 零 bias，`networks.py:100-109`）。即加宽的 32 列
  是死列，`event_blind` 退化为**同 seed 的 MLP**。它只作为 seed-matched 配对控制（真实价值是
  降 seed sd），**不得**用「扣除编码器/容量」措辞报告。
- **真正的信息对照是 `event_noise`**：用与 `event` **完全相同的** `EventSequenceEncoder`
  （同 init 分布、同参数量），把事件向量替换为**信息为零但非退化**的输入——从固定种子采样的
  i.i.d. 噪声（或「同一固定非零向量 + 变化的序列长度」的 `event_const`）。通道非退化、容量被
  激活、与真实事件信息无关。
- `event_blind` **必须明确 seat 段是否也置零**：本规划规定 `event_blind` = 事件向量全零 + seat 段
  也置零（`--event-blind` 等价于 §5.1 的 blind 定义）。单独的 `event_seatblind` = 事件真实、仅
  seat 段置零。
- **禁止**把 `event vs t17pool` 当作信息端点单独下结论（它混入容量/初始化；见 pilot §5.3 的
  `seq vs t17pool` +12.44 教训）。

### 1.3 两个分布（修复 `ml-stats-02/08`、`engineering-selfplay-prior-misattribution`）

1. **random（stage-1，主分布）**：镜像既有 `t17pool`/`t17seq` 的 config（random 对手、500k、
   8×128、`hidden=128`、`shared`、lr 2.5e-4 退火；pilot §5.2 命令）。这是与 D1-lite、B1、B0
   直接可比的主分布。
2. **self-play（stage-2，必做分布，用户拍板）**：**没有任何序列/第二输入臂在 self-play 下训过**
   （pilot §8.2；`runs/t17self__1__1790439615/args.json` 无 `seq_*` 键）。唯一 self-play 历史证据
   是 v5 的 obs B1 段（`+9.85 [−2.97,+22.68]`，单 seed，pilot §4.2），与事件/序列编码正交。
   用户已拍板：self-play 是本轮**必做阶段**，先完成 `league.py` 透传（§6 步骤 7），顺序放在
   random 主端点（步骤 6）跑通之后；其统计结论仍按 §1.5 与 random 分开报告，不因必做而放宽归因。

**self-play 命令的精确语义（修复 `ml-stats-08`）**：本仓 `--opponent self` **忽略**
`--mix-random-prob`（`src/seven523/league.py:168` 只在 `opponent == "mix"` 时使用它；
`--opponent` 的 choices 见 `src/seven523/train.py:178-184`）。既有 `runs/t17self__*` 用的是
`--opponent self --mix-random-prob 0.5`，**实际是 100% 纯 self-play**。本规划：
- 若确要纯 self-play：写 `--opponent self`，**不写**无意义的 `--mix-random-prob`，并在报告里核对
  `args.json` 的 `opponent` 值；
- 若确要 50/50：必须写 `--opponent mix --mix-random-prob 0.5`。

### 1.4 seed / 牌数 / 预算（修复 `contracts-first-wave-inconsistency`）

- **训练 seed**：至少 `1,2,3`（与 prior 可比）。若主端点落在 +5~+10 或 CI 跨 0 但方向为正，
  补 `4,5`（瓶颈是 **训练 seed sd**，不是牌数；ablation §4.2）。
- **评测 deal seed**：先用 `0,1,2`，每 seed `--pairs 400` 换座；触发加 seed 时升到 `0..4` ×400，
  或 `--pairs 800`。
- **两层次合并（写死）**：`tools/head_to_head.py --seeds 0,1,2 --pairs 400` 先在 **3 个 deal seed**
  间合并（`duel.combine_duel_seeds`，`duel.py:288-333`；每 pair 一牌两局换座，
  `duel.plan_duel_schedule`，`duel.py:33-46`），再由 `aggregate_seq.py` 在 **3 个 train seed**
  间合并。所以一个端点 = **3 train seed × 3 deal seed × 400 副 = 3600 副**。
- **单臂预算**：500k timesteps（与 T17 全系一致）。
- **第一波训练量（精确臂表）**：random `event`/`event_noise`/`event_seatblind` × seed {1,2,3}
  = **9 run**；`event_blind` 作为可选第 4 臂（+3 run）仅在需要与 D1-lite `seqblind` 直接对位时加；
  D1-lite 与 MLP 对照复用现成 ckpt、**0 run**。机制臂（shuffle/boundaryblind/nopass）在信息端点
  为正后再投。self-play（6 run）为**必做**（用户拍板），顺序在步骤 6 之后（§6 步骤 7）。

### 1.5 决策门（预注册，避免事后移动门线；修复 `ml-stats-05/06/12`）

**合并公式**：`mean ± max(平均 boot SE, seed sd) × q / √k`，其中
`q = t_{0.975, k-1}`（k=3 → 4.303）。既有 `runs/t17seq_train/aggregate_seq.py:15` 硬编码
`Z=1.96`，在 seed sd 主导时约低估 2.2× CI（红队 `ml-stats-05`）；本规划**以 t 分位为准**，
**同时**报 z 口径以便与前序报告可比，并在报告中给出分位数敏感性。k=3 且 seed sd 主导时，
CI 半宽约 ±28 Elo，几乎不可能排 0 → 该情形显式标注「不可判」，不得用 z 口径下结论。

按主端点 `event vs event_noise` 的 3-seed t 合并点估计 `Δ` 与 CI：

| 情形 | 判定 | 动作 |
|---|---|---|
| `Δ < +5` 且 CI 跨 0 | 无信号 | **停**，发布 null（事件级在此预算/分布下无 ≥+10 证据；**不等于效应为 0**） |
| `+5 ≤ Δ < +10`（CI 任意），或 CI 跨 0 且 `Δ ≥ +5` | hint | 加 seed 4/5（或 5×400）再合并；过 +10 且 CI 排 0 才进入下一阶段 |
| CI 排 0 且 `Δ ≥ +10` | 行动门槛达到 | 进入 self-play 复算 + D1-lite 比较；两分布方向一致再议 ship |
| CI 排 0 但 `Δ` 在 [+5,+10) | 方向确定幅度不足 | 报结论：CI 排 0、幅度 <+10，不 ship；可选单因素调优（参照 ablation §3.3） |

**分布联合判据（修复 `ml-stats-06`）**：random 是低功效分布（3×400 对 10 Elo 约 40% 功效，
README §0.2、pilot §8.3），而 a5/a6 的价值被认为只在 self-play 出现。因此：
- random 门线与主端点绑定（与 prior 可比）；
- **random null 不能单独作为「效应=0」的结论**，报告必须同时给出功效；
- self-play 是**必做阶段（用户拍板）**：顺序固定在 random 主端点（步骤 6）之后、`league.py`
  透传完成之后；其读数按同一 t/z 口径报告，但 random 门线仍是唯一「停/继续」判据。
- 禁止事后挑端点/挑 seed/序贯停下：confirmatory 族固定为 §1.2 的两个端点，加 seed 只能在
  §1.5 行 2 触发时进行。

---

## 2. 信息分解：事件级 + 相对座位相对 D1-lite 的增量

### 2.1 对应前序报告 a1–a6

| # | 信息轴 | v5 现状 | D1-lite（扁平牌序） | 事件级 + 相对座位 | 是否真新增 |
|---|---|---|---|---|---|
| a1 | 顺序 / 近因 | 集合 `unseen`，无时间戳 | ✅ 有（chrono） | ✅ 有 | 否（重编码） |
| a2 | 组合邻接 / 组合边界 | 集合，无边界 | ⚠️ 只能从相邻 token 隐式学 | ✅ 显式（每个事件 = 一个组合） | **部分新增**（显式分组） |
| a3 | 当前墩节奏 | 只有 `last_player` 快照 | ⚠️ 只有当前墩内 `trick_cards` 序 | ✅ 全墩节奏 + **pass 事件** | **新增**（pass/回合计数） |
| a4 | 逐座位归属 / 墩边界 / 赢家 / 出空序 | 完全没有 | ❌ 无座位、无墩界 | ✅ 每事件带相对座位 + `opens_trick` + 可导出墩赢家 | **真新增（核心，先验未知）** |
| a5 | 对手手牌后验 / belief | `unseen` 是并集 | ❌ | ✅ 由 a4 条件化的可学后验 | **新增能力** |
| a6 | 对手倾向（风格） | 无 | ❌ | ✅ 需 a4 + 强对手分布 | **新增能力（仅 self-play 可测）** |
| a7 | 显式剩余大牌估计 | `unseen` 线性可读但 MLP 吃力 | ❌ | 本方案不覆盖（Q-5，`docs/plans.md:349`） | — |

**归因限制（修复 `ml-stats-11`）**：开关 1 是事件级 ⇒ **a2 组合边界天然内建**，本方案**没有**
单独隔离 a2 的端点（A 方案的逐牌 token 已被用户否决）。a4 的「墩边界」侧用
`event_boundaryblind`（置零 `opens_trick`）近似隔离，「归属」侧用 `event_seatblind` 隔离；
两者都不是 a2 的干净端点。计划**明确声明 a2 在本方案下不可单独归因**。

**先验归因（修复 `ml-stats-04`）**：a1/a2 主要是**编码差异**，chrono/sorted 的数字只支撑 a1；
**a4 从未被前序测过**，其先验应为**接近零/未知**。事件臂定位为**探索性**。

### 2.2 事件级「新在哪」的精确定义

给定 `View` 的公开牌集（D1-lite 用的 `played + trick_cards`）不变，事件级新增的是：

1. **每张牌归属哪个座位**：D1-lite 的 token 流丢掉了人。
2. **哪些牌是一次打出**：D1-lite 的扁平流里，同一次组合的多张牌与跨事件相邻无法区分。
3. **墩边界**：`opens_trick` 明确标出每墩第一手；D1-lite 只有「本墩内相邻」，没有跨墩分组
   （`trick_cards` 只保留当前墩，`played` 只有牌）。
4. **pass 事件与回合节奏**：D1-lite 完全没有；v5 只有 `last_player` 快照。
5. **出空标记**：`went_out`（公开事实，手牌数归零）——对「撬底窗口」推断有用。

**墩赢家大部分可导出，但有 corner 例外（修复 `ml-stats-13b`）**：通常一墩结束于「轮回到
`last_player`」，赢家 = 该墩内**最后一手非 pass 事件**的座位（`Game._advance` 回到 `last_player`
即 `_end_trick`，`game.py:217-224`）。**例外**：补牌角落撬底时（某家本墩中途出空、本墩结束补牌
恰牌堆空），撬底者是 `empty_order[0]`（**最早出空者**），未必是本墩最后出牌者
（`game.py:268-270`，ADR-0014 §决定.3），此时 `StepResult.winner` = 撬底者而非 `last_player`。因此
「赢家 = 最后非 pass」的推导**必须带这个例外**，测试 `test_trick_winner_is_last_non_pass` 也要按
「普通墩」限定。

### 2.3 诚实先验与理由

**预期为小正、CI 大概率跨 0**：

- **支持（弱）**：(i) a4 确实不在 `View` 里，是前序点名「最值钱的轴」；(ii) chrono vs sorted
  +10.46 [−2.46,+23.38] 说明「顺序」本身就值 ~+10 点估计——但这只支撑 a1，**不支撑 a4**。
- **反对**：(i) D1-lite 的信息端点 `seq vs seqblind` = +8.41 [−4.06,+20.89]，CI 跨 0；
  (ii) 诊断显示编码器主要学「有没有历史」，顺序敏感度比有无历史小一个量级（ablation §0.4）;
  (iii) random 对手没有可学倾向，a4 在 random 上的价值主要来自「墩/出空结构 → 撬底时机」，而不是
  对手建模；(iv) value 头几乎不消费历史（|Δvalue| ≤0.064，ablation §0.4）；(v) 3×400 对 10 Elo
  只有约 40% 功效，null 只能作「无 ≥+10 证据」，不能作「效应=0」。
- **self-play 先验（重述）**：无任何第二输入臂的 self-play 结果；唯一 hint 是 v5 obs B1
  +9.85（单 seed，弱间接）。a5/a6 只在对手是自身历史函数时才有料（pilot §1.3），但该假设**未经
  事件/序列臂验证**。

---

## 3. 引擎改动

目标：让**公开**的「谁在什么时候打了什么组合、是否开墩、是否出空」进入 `View`，且**不改变
任何现有字段语义**、不改 trace 格式、不改 obs 布局、不暴露隐藏信息（ADR-0002）。`empty_order`
**不**进 `View`（ADR-0014 明确拒绝）。

### 3.1 `GameState` 新字段 schema

在 `src/seven523/game.py` 的 `GameState`（`game.py:26-58`）新增字段（其余字段不动）：

```python
@dataclass(frozen=True, slots=True)
class Play:
    """一条公开动作记录：某座位打出某组合（pass 为空牌集）。

    seat        行动的绝对座位
    cards       解析后的具体牌（规范排序）；pass 为 ()
    opens_trick 行动前 state.incumbent is None（即本手开新墩）
    went_out    本手后该座位手牌为空（公开：手牌数归零）
    """
    seat: int
    cards: tuple[Card, ...]
    opens_trick: bool
    went_out: bool


@dataclass(frozen=True, slots=True)
class GameState:
    ...
    #: 每个公开动作（含 pass），oldest-first。与 ``played`` 不同：``played`` 只记
    #: 已结束墩的牌、没有座位；``plays`` 记全部公开动作、带座位与墩边界。开局为空，
    #: 由 ``step`` 追加、由 trace 回放自然重建。
    plays: tuple[Play, ...] = ()
```

**放置点**（逐函数，锚点已核对）：

- `Game.step`（`game.py:170-215`）：
  - pass 分支（`game.py:181-182`）：改为先构造
    `replace(state, plays=state.plays + (Play(seat, (), False, False),))`，再 `self._advance(...)`。
  - 出牌分支：在 `after = replace(...)`（`game.py:201-209`）里追加
    `plays=state.plays + (Play(seat, combo.cards, state.incumbent is None, not hand),)`。
    `combo.cards` 已是规范排序（`combos.Combo`，`combos.py:53-76`）。
  - 立即撬底路径（`game.py:214`）复用同一个 `after`，事件已包含在其中。
- `Game._end_trick`（`game.py:226-313`）：**不**动 `plays`（只在 `replace` 时透传）。补牌
  （`game.py:245-257`）、撬底扫分（`game.py:265-284`）都**不产生**事件。
- `Game.restore`（`game.py:133-148`）：`plays` 默认 `()`（与 `empty_order` 一致）。
- `Game.view`（`game.py:150-168`）：加 `plays=state.plays`。

### 3.2 `View` 暴露什么（修复 `contracts-view-field-default`、`ml-stats-09`、`engineering-view-default-call-sites`）

`View`（`game.py:91-107`）新增**带默认值**的字段：

```python
    #: 公开动作序列，oldest-first（见 GameState.plays）。默认空以保证既有
    #: 关键字构造（tests/test_env.py:67/:196/:227/:317）不破坏。
    plays: tuple[Play, ...] = ()
```

`plays` 放在**最后**（否则 dataclass 默认值位置会报错；它是唯一带默认值的字段）。**只暴露公开
事实**：`Play.cards` 是已翻到桌面上的牌，`Play.seat` 是行动者，`opens_trick`/`went_out` 都可由
公开计数推出。**不暴露**：底牌堆顺序、他人手牌、撬底扫走的牌（那些牌从未被打出，`played`
的既有语义也不含它们，见 `tests/test_game.py:199-215`）。

**既有构造点必须进改动表**（`tests/test_env.py`）：`_view(spec)`（`tests/test_env.py:67`，被
`:253`/`:354`/`:370` 的 JSON fixture 用例使用）、`:196`、`:227`、`:317`。由于 `plays` 有默认值，
这些点**不改也能通过**；但为覆盖新字段，`_view` 应扩展为
`plays=tuple(...) for spec.get("plays", ())`，并新增「既有 v5 fixture 逐位不变」的回归断言。

### 3.3 restore / replay / trace

- **`Deal`（`game.py:61-79`）不变**：trace 存的是开局 `Deal` + 每步动作，不是 `plays`。
- **`restore`（`game.py:133-148`）**：`plays=()`；回放时由 `step` 逐手重建。
- **`trace.py` 零格式改动**：`step_record`（`trace.py:139-162`）不写 `plays`。
  `play.replay_trace`（`play.py:211+`）经 `state_from_snapshot`→`restore`→逐手 `Match.step`
  重建，`plays` 自动一致。`TRACE_VERSION`/`rules.revision` 门禁不动。
- **相等性**：`GameState` 是 `frozen`+`slots` 的 `dataclass`，新字段自动进 `__eq__`/`replace`；
  任何按「重放后 state 相等」写的测试仍成立。

### 3.4 撬底 / 补牌 / `empty_order` / `went_out` 怎么处理（修复 `ml-stats-13a`、`contracts-adr0014-wentout-tension`）

- **补牌**不是事件（`_end_trick` 里 `hands[seat] |= pile[:take]`，`game.py:245-257`）。
  它**不可**由「下一手的 `opens_trick=True`」可靠推断：每墩开始下一手都 `opens_trick=True`，
  且补牌并**非必然**——只有不满手的座位补牌，牌堆可能已空（`game.py:248-257`）。因此补牌**不**
  编码为事件，也**不做**「补牌可从 opens_trick 推断」的声明。
- **撬底**是终局：触发它的那一手是普通出牌，已被记录；撬底扫走的手牌**不是**事件。
  最后一步的 `StepResult.dug/done`（`game.py:305-313`）不进 `plays`。
- **`empty_order`**（`game.py:51`）保持引擎内部、不进 `View`（ADR-0014）。事件流用每手的
  `went_out` 表达同一公开事实。**与 ADR-0014 的关系（显式对账）**：ADR-0014 的「考虑过的替代」
  拒绝的是「把**出空顺序**作为 `View`/观测的一个**专门字段/维度**」，理由是「该信息在手牌数里已经
  公开，无需新增维度」。`Play.went_out` **不是**出空顺序字段：它是「某一手之后该座位手牌归零」
  这个**逐事件、可由 `View.counts` 逐步导出的冗余便利标记**，且本规划**不**把 `empty_order`
  本身投影出来。若实现时认为连 `went_out` 都应省去，可以 `--event-wentout drop` 消融；默认保留。
- **pass 合法性**：`action_mask` 仅在 `incumbent is not None` 时给 pass 位
  （`actions.py:262-263`），故 pass **永不** `opens_trick`。

### 3.5 逐文件逐函数改动点 / 测试点（修复 `contracts-test-plan-gaps`、`engineering-doc-sync-and-exports`）

| 文件 | 改动 | 新增/修改测试 |
|---|---|---|
| `src/seven523/game.py` | 新增 `Play`；`GameState.plays`；`step` 追加（pass + play）；`view` 透传；`restore` 默认空 | `tests/test_game.py` 见下 |
| `src/seven523/history.py` | 新增 `EVENT_DIM`/`event_length(rules)`/`encode_events`/`EventHistoryWrapper`；扩展 `__all__`；**不动** `encode_history` | `tests/test_history.py` |
| `src/seven523/networks.py` | 新增 `EventSequenceEncoder`；`Agent(event_*)` 分支；`save/load_agent` 的 event 字段 + layout marker；`warm_start_from` layout 校验；`NeuralPolicy.act` 重建 | `tests/test_history.py`、`tests/test_networks.py` |
| `src/seven523/ppo.py` | `RolloutBatch` 增 `events/event_seats/event_mask`（默认 `None`）；`flatten`/`ppo_update` 透传 | `tests/test_ppo.py` 扩展 flatten/None 断言 |
| `src/seven523/train.py` | CLI `--event-*`；rollout 存事件张量；`make_env` 包 wrapper；**bootstrap 读 `next_events*`**；warm-start 预检 | `tests/test_history.py` smoke、`tests/test_train.py` |
| `src/seven523/league.py` | **冻结对手透传 `seq_*`/`event_*` 配置**（`league.py:153-167`），或 `copy.deepcopy(agent).eval()` | `tests/test_league.py` self+event smoke |
| `DESIGN.md` §2.4 | `View` 字段表加 `plays`；注明「第二输入」接缝（ADR-0010 owner-doc 纪律） | 文档（本阶段不改，实施时改） |
| `CONTEXT.md` | 「投影」「事件历史」词条同步 | 文档 |
| `src/seven523/__init__.py` | 决定并实现 `Play` 是否 re-export（`__init__.py:46,82` 已导出 `View/GameState`） | `tests/test_policies.py` 风格断言 |
| `tests/test_game.py` | — | 新 `test_plays_*`（见下） |
| `tests/test_env.py` | 扩展泄漏测试；`_view` 读 `plays`；更新 `:196/:227/:317` | `test_encoding_does_not_leak_hidden_cards`（现 `:395`）加 `plays` 差分 + fixture 回归 |

`tests/test_game.py` 必加：

1. `test_plays_parity_with_played_and_trick_cards`：任意决策点，
   `[c for p in view.plays for c in p.cards] == view.played + view.trick_cards`（pass 空牌集）。
2. `test_opens_trick_counts_match_tricks`：`sum(p.opens_trick)` == 已结束墩数 +（进行中墩是否已开）。
3. `test_trick_winner_is_last_non_pass`：对随机策略全 games，用 `plays` 导出的墩赢家与
   `StepResult.winner` 逐墩比对；**普通墩**用「最后非 pass」，**补牌角落撬底墩**由
   `empty_order[0]`（最早出空者）收分（ADR-0014 §决定.3）。
4. `test_pass_only_when_incumbent`：pass 的 `opens_trick is False`；领先时无 pass 记录。
5. `test_plays_reconstructed_by_replay`：`build_trace`+`replay_trace` 后 `plays` 与在线一致。
6. 泄漏：两个只隐藏字段不同的 state（复用 `tests/test_game.py:146` 的形式）产出相同 `View.plays`。

---

## 4. 事件表示与编码器

### 4.1 每个事件张量的确切 schema

**每事件特征向量 `event_dim = 64`（float32）**，字段拼接顺序：

| 段 | 宽 | 内容 |
|---|---|---|
| `cards` | 54 | 该组合牌集的 54 维多热（`card_id`，`cards.py:147`）；pass 全 0 |
| `kind` | 6 | `ComboKind` one-hot（`combos.py:18-23` 六类）；pass 全 0 |
| `size` | 1 | `len(cards) / rules.hand_size`（pass=0） |
| `is_pass` | 1 | pass=1 |
| `opens_trick` | 1 | `Play.opens_trick` |
| `went_out` | 1 | `Play.went_out` |

合计 `54+6+1+1+1+1 = 64`。

**为什么 54 多热而不是 rank×suit**：事件语义是「一手牌的组合」，54 多热是**无损、无序**的
集合表示，MLP 不必重建组合；rank×suit 只对 top card 紧凑，对顺子/连对要每张牌展开，反而
更绕。D1-lite 用 `card_id+1` token 也是 54 词表；这里把它换成多热 + 组合元数据，词表/牌
身份完全一致，便于与 D1-lite 对照。

**pass 作为事件**：**是**（默认 `event_pass=keep`）。理由：pass 是 a3 节奏与回合计数的载体，
且在 2 家下「谁没压」就是对手后验的关键信号（a5）。保留一个可关闭的消融
`--event-pass drop` 来量化它的贡献。

**事件数上限与 padding/mask**：

- **`event_length(rules) = rules.num_players * NUM_CARDS`**（**不硬编码 108**，修复
  `contracts-encode-events-needs-rules`）：2 家 = **108**，3 家 = 162。
  推导：每个非 pass 出牌至少移除 1 张牌、全局每张牌最多被打出一次 ⇒ plays ≤ 54；每墩至多
  `n-1` 次 pass 结束、墩数 ≤ plays ⇒ passes ≤ `(n-1)·plays`；合计 ≤ `n·plays ≤ n·54`
  （`_advance` 空位跳过导致某些墩无 pass，只会更少；`game.py:217-224`）。实现里断言
  `len(plays) <= event_length(rules)`，并用 §3.5 的全 games 测试把经验最大值钉死。
  红队实测界限安全（n=2 最大 72，n=3 最大 93，随机合法策略 600 局）。
- 序列 **oldest-first、右 padding**（与 `encode_history` 同约定，`history.py:42-72`）。
- padding 事件 = 全 0 向量，padding seat = 专用 pad 下标 `n`，`mask=False`；时序模型按
  `mask.sum(1)` 取**最后一个真实事件的 hidden**（GRU 因果，后续 pad 不影响该步）。
- 截断（仅当显式设 `length < event_length`）：保留**最近 `length` 个事件**，与 D1-lite
  一致；注意截断可能切开一墩。**latency 旋钮警告**：ablation §3.3 显示 len 54→27 = −18.09
  （CI 全负），故默认**不截断**；截断是显而易见的降本旋钮，但已被前序证据劝退。

### 4.2 seat embedding：相对座位、维度、相加 vs 拼接

- **相对座位**：`rel = (play.seat - view.seat) % n`，值域 `0..n-1`（0=自己，1..对手）。
  这是**防换座分布外**的关键：learner 恒在 seat 0 训练，h2h 会换座（`duel.plan_duel_schedule`
  双座位，`duel.py:33-46`）；绝对座位 embedding 会让 seat 1 落在未训练下标。v5 的
  `scores`/`opp_count`/`opp_revealed` 已是自中心旋转（`env.py:151-183`），本方案沿用同一不变量。
  **每次决策都按当前 `view.seat` 重算所有历史事件的相对座位**（无状态重算的一部分）。
- **维度**：`seat_emb = 16`；`nn.Embedding(num_players + 1, 16, padding_idx=num_players)`。
- **相加 vs 拼接**：**推荐拼接** `concat(event_vec(32), seat_emb(16))` → 时序输入 48。
  理由：(1) 座位是分类角色、事件向量是牌面内容，拼接让二者各占独立子空间，避免相加时
  「两者坐标对齐、梯度互相抵消」；(2) 便于做**干净的座位消融**（`event_seatblind` 只把 seat
  段置零/替换为 pad，事件段不动）；(3) 时序输出宽仍可固定 32（见 §4.3），保持与 D1-lite 的
  trunk 宽一致。**备选**（记为消融 `--event-seat sum`）：`event_vec + seat_emb`，要求
  `d_event = d_seat = 32`，GRU 输入 32；若拼接出现座位过拟合/不稳再切换。

### 4.3 事件 MLP 与时序模型（修复 `contracts-event-vs-seq-confound`、`ml-stats-03`）

- **事件 MLP（共享权重，对所有事件同一套）**：`Linear(64→32) → ReLU → Linear(32→32)`，
  `layer_init` 正交（沿用 `networks.py:64-68`）。
- **时序模型**：默认 **GRU**，`nn.GRU(48, 32, batch_first=True)`，正交 + 零 bias（复用
  `SequenceEncoder` 的初始化写法，`networks.py:100-109`）。读出 = 最后一个真实事件的 hidden
  （`mask.sum(1).clamp(min=1)` 处 gather，仿 `networks.py:110-118`）。
- **输出宽固定 32**：与 D1-lite 的 `seq_hidden=32`（默认在 `networks.py:142`，赋值为
  `networks.py:165`）一致，使 `event` 臂的 trunk 首层宽 = `161 + 32 = 193`
  （`Agent._trunk_input`，`networks.py:207-218`）。
- **`event vs seq` 是混淆端点**：实测 `SequenceEncoder` 参数 5680；规划的
  `EventSequenceEncoder` 参数 ≈ 11056（MLP 3136 + seat 48 + GRU 7872），约 **1.95×**；序列长度
  event ≤108 vs seq 54（`history.py:30`）。因此 `event vs seq` 同时改变**表示、编码器容量、序列
  长度**，**不得**解释为干净的信息比较；只用 `event vs event_noise` 与 `event vs event_seatblind`
  做归因。
- **attention 变体**：**不在首轮**。理由：无状态重算下注意力需要位置编码与因果 mask，且
  「无位置注意」退化成近似 bag、正好是 `sorted` 对照；GRU 是已被前序测过、成本最低的默认
  （ablation §0.1）。只有在 GRU 出现正信号但明显平台化时，才做一层 causal self-attention 消融。

### 4.4 无状态重算的细节与推理成本（修复 `engineering-eval-inference-cost`）

- 每个决策点，从 `View` 现场重建**完整** `(events, seats, mask)`（不依赖上一决策的缓存、
  不携带 hidden），前向 `EventSequenceEncoder` 得到 32 维历史特征，拼到 obs 后进 trunk。
- 训练 rollout 与推理共用**同一个** `encode_events`：训练侧 `EventHistoryWrapper` 每步把当前
  `View` 的事件挂到 `last_events/last_event_seats/last_event_mask`（仿 `HistorySequenceWrapper`，
  `history.py:75-110`），`NeuralPolicy.act` 侧从 `View` 重建（仿 `networks.py:494-515`）。
  两者逐位一致的断言照抄 `tests/test_history.py` 的一致性测试形式。
- **签名（修复 `contracts-encode-events-needs-rules`）**：
  `encode_events(view, rules, length=None, order="chrono", ...)`；
  `EventHistoryWrapper(env, rules, length=None, ...)`。`length=None` ⇒
  `event_length(rules)`。`n` 来自 `rules.num_players`（不要从 `view.counts` 猜）。
- **复杂度与推理成本**：每决策 O(事件数) 的 GRU 前向（≤108 步），比 D1-lite 的 54 步长。这条
  路径**也在推理/eval 上**：红队实测 CPU 每决策 MLP 0.267 ms vs seq54 GRU 1.553 ms，事件臂约
  3–4 ms/决策。h2h 单条 = 3 deal seed × 400 pair × 2 局 × ~56 决策 ≈ 134k 决策 ⇒ 事件臂单条
  h2h 约 2–4 min（pilot §5.2 的 27 s/条是 MLP 口径）。**实现后必须在报告里记录实测 ms/决策与
  吞吐**；升级分支（加 seed / 5×400）会成倍拉长串行 CPU eval 时间。

### 4.5 复用第二输入管线 / ckpt / warm start / 默认回归（修复 `contracts-warmstart-factual`、`engineering-warm-start-claim-wrong`、`contracts-obs-dim-identity`、`engineering-adr0009-neutral-start`）

**复用点**（与 D1-lite 的对位）：

| D1-lite | 事件级 |
|---|---|
| `SequenceEncoder`（`networks.py:92-118`） | 新增 `EventSequenceEncoder`（事件 MLP + seat + GRU） |
| `Agent(seq_len, seq_emb, seq_hidden, seq_blind, seq_order)`（`networks.py:133-151`） | `Agent(event_len, event_dim=64, event_emb=32, event_hidden=32, seat_emb=16, event_blind=False, event_seat="concat", event_order="chrono", event_pass="keep")` |
| `_trunk_input` 拼序列（`networks.py:207-218`） | 拼事件特征（`seq_len` 与 `event_len` 可并存，各自加宽；首轮二者互斥使用） |
| `encode_history` + `HistorySequenceWrapper`（`history.py:42-110`） | `encode_events` + `EventHistoryWrapper` |
| `RolloutBatch.seqs`（`ppo.py:52-93`） | 新增 `events`(float)、`event_seats`(int64)、`event_mask`(bool)，默认 `None` |
| `train.py` 存 `(T,E,54)` int64（`train.py:504-513`） | 存 `(T,E,EVENT_LENGTH,64)` float + `(T,E,EVENT_LENGTH)` int64 + mask |
| `save/load_agent` seq 字段（`networks.py:292-347`） | 并行 `event_*` 字段 + layout marker，旧 ckpt 缺省 `event_len=0` |
| `NeuralPolicy.act` 重建（`networks.py:494-515`） | 同位置重建事件 |

**ckpt payload 字段**（追加，不动 `obs_version`）：`event_len`、`event_dim`、`event_emb`、
`event_hidden`、`seat_emb`、`event_blind`、`event_seat`、`event_order`、`event_pass`，外加
**layout marker** `history_layout`（字符串，如 `"mlp"` / `"seq"` / `"event"`；或等价的
`event_len`/`seq_len` 组合校验）。`load_agent`（`networks.py:313-347`）对新字段一律
`payload.get(..., default)`，保证旧 v5 ckpt 仍可加载。**ADR-0009 的硬校验不受影响**（obs
布局没变，`obs_version` 仍是 5），但**obs_dim 已不足以保证 ckpt 身份**：事件臂与 MLP 臂
`obs_dim` 都是 161、`obs_version` 都是 5，而首层分别是 193/161。因此
`warm_start_from`（`networks.py:432-461`）与 `save/load_agent` 必须**同时**校验 layout marker，
与 `obs_dim` 一起 fail-loud（修复 `contracts-obs-dim-identity`）。测试
`tests/test_networks.py:75`/`:181` 现在只按 `obs_dim` 定义身份，实施时补 layout 断言。

**warm start 的真实行为（更正）**：`warm_start_from`（`networks.py:446-449`）在
`loaded.nvec == agent.nvec` **且** `loaded.arch == agent.arch` 时直接走
`agent.load_state_dict(loaded.state_dict())`（**strict**）。事件臂与 `t17pool` 正是这种
（nvec/arch 相同、obs_dim 都是 161），**不匹配的首层会直接抛 `RuntimeError`**，**不是**「跳过
不匹配、得到随机首层」。会「跳过不匹配」的是 `warm_start_into`（`networks.py:395-427`，只拷
形状相同的 tensor 与 actor head 前缀），但它在同 arch/同 nvec 时**根本不会被调用**。既有
`tests/test_train.py:498` 断言逐位相等、`:525` 断言 `SystemExit`，二者对事件臂都不成立
（会抛 `RuntimeError`）。因此：

- **主路径：从零训**（镜像 `t17pool`/`t17seq`）。
- **可选「零填充中性起点」：本规划默认删除**（修复 `engineering-adr0009-neutral-start`）。
  它要求对首层做**列级手术**（拷 161 列、零初始化 32 列），这正是 ADR-0009 决定 #2「热启动只有
  一条路径」所**明确退役**的 `_first_layer_remap`/`_REMAP_VERSION_PAIRS` 机制
  （`networks.py:364-375` 现在对 layout 不匹配直接 `WarmStartLayoutError`；
  `networks.py:399-427` 只拷等形 tensor）。若确要保留该消融，必须：(a) 新立 ADR 记录
  `warm_start_event_columns(path, agent, neutral_zero=True)` 的列映射规则与测试，并在 §7.3 声明；
  (b) 单列对照臂，不与从零训混比。(a) 是**需要用户决策**的范围扩展，默认不做。
- **warm-start 预检**：`train.py:412-423` 目前只捕获 `WarmStartLayoutError`。实施时把
  `--load-checkpoint` 的第二输入配置/ layout 不匹配在调用 `warm_start_from` **之前**显式拒绝，
  用友好 `SystemExit` 提示「需要同 layout 的 ckpt；事件臂需从零训或走新的中性起点为 opt-in」，
  并加「`RuntimeError` 被转换为 `SystemExit`」的测试。

- 默认路径回归：`seq_len=0` 且 `event_len=0` 时，`Agent` 不建事件编码器、trunk 首层 =
  `obs_dim`、参数量与前序一致（`test_networks` 的 55,179 参数计数断言，pilot §5.1），
  `RolloutBatch` 三字段为 `None`，`NeuralPolicy.act` 走旧分支。

---

## 5. 对照臂与实验矩阵

### 5.1 臂（修复 `ml-stats-01/07/11`、`engineering-shuffle-determinism`、`engineering-seatblind-weak-contrast`）

| 臂 | 输入 | 隔离的轴 | 实现 |
|---|---|---|---|
| `event` | 真实事件 + 相对座位 | 全量 | `--event-len 108` |
| `event_noise` | **同一编码器，事件向量=固定种子 i.i.d. 噪声（信息零、通道非退化）** | 信息 vs 非退化容量（**Q-A 主对照**） | `--event-noisy` |
| `event_blind` | 全零事件 + 全零 seat | seed-matched 配对（≈重训 MLP） | `--event-blind` |
| `event_seatblind` | 真实事件、seat 段置零 | 相对座位（a4 归属，**保守**） | `--event-len 108 --event-seat none` |
| `event_shuffle` | 同多重集、**墩内**确定置换（保 `opens_trick` 位置与墩块） | 顺序/近因（a1） | `--event-len 108 --event-order shuffled` |
| `event_boundaryblind` | 真实事件、置零 `opens_trick` 标志 | 墩边界（a4 边界侧） | `--event-boundary-blind` |
| `event_nopass` | 丢弃 pass 事件（可选） | pass/节奏（a3） | `--event-len 108 --event-pass drop` |
| `seq`（对照，复用） | D1-lite chrono token | D1-lite | `runs/t17seq__{1,2,3}` |
| `seqblind`（对照，复用） | D1-lite 全 pad | D1-lite 容量 | `runs/t17seqblind__{1,2,3}` |
| `t17pool`（对照，复用） | 纯 MLP v5 | 基线 | `runs/t17pool__{1,2,3}` |

**`event_shuffle` 的确定性语义（写死）**：**仅在墩内**按**固定、无 RNG 的确定置换**重排事件
（例如保序反转，或 `perm = sorted(range(k), key=lambda i: (i * 2654435761) % k)`），保留每个
`opens_trick` 事件的位置与墩块结构。**绝不**用未播种的 `random.shuffle`：训练 rollout 与
`NeuralPolicy.act` 必须对同一 `View` 产出**同一**置换，否则违反「wrapper==直接调用」验收、
破坏 h2h 可复现性与 twin-deal 配对。置换规则作为 `event_order` 的一部分写进 ckpt payload，并加
「同一 View 两次编码相同」的确定性测试（修复 `engineering-shuffle-determinism`）。
**与 ablation 的 `sorted` 区分**：`sorted` 按 card-id 规范序、丢 bag 顺序；`event_shuffle` 保事件
内组合结构、只打乱**墩内**事件间顺序。全局打乱**不做**（会同时打散墩边界，混淆 a1 与 a4，
`ml-stats-07`）；墩边界侧的隔离交给 `event_boundaryblind`。

**`event_seatblind` 是保守臂（`engineering-seatblind-weak-contrast`）**：2 家下，由
`opens_trick` + pass 位置 + 座位交替可把每事件座位还原到一个全局偏移，故 seat 信号**部分可由
结构推出**。因此 `event_seatblind` 的点估计即使为 0，也**只能**读作「座位信号可由结构复原」，
**不能**读作「归属无用」。报告中必须写明这一读法。

### 5.2 分布 / seed / 预算 / 评测成本（修复 `engineering-eval-inference-cost`）

- **random**：`event`/`event_noise`/`event_seatblind` × seed {1,2,3}，各 500k
  （config 镜像 `t17seq`，pilot §5.2）。可选 `event_blind`/机制臂在信息端点为正后追加。
- **self-play（必做，用户拍板；步骤 6 之后）**：`--opponent self`（**纯 self-play**；若要 50/50 用
  `--opponent mix --mix-random-prob 0.5`）× seed {1,2,3}，`--self-play-refresh 10`
  `--total-timesteps 500000`。**前提：先完成 `league.py` 透传（§6 步骤 7）**。
- **训练并发**：最多 3 进程（纪律，§顶部）。先 3 seed 的顺序批，再下一批。
- **评测**：每臂每 seed 与同 seed 对照 h2h，`--seeds 0,1,2 --pairs 400 --bootstrap 4000`；
  串行、`--device cpu --workers 4`。
- **eval wall-clock（新增）**：事件臂单条 h2h 约 2–4 min（红队实测每决策 ~3–4 ms，134k 决策/
  条）。第一波 3 端点 × 3 train seed = 9 条串行 ≈ 20–40 min；self-play 阶段再加约 6 条。升级分支
  （加 seed / 5×400）按比例放大。**该成本不受资源纪律影响（仍 CPU、workers=4、串行），但必须
  写进排期。**

### 5.3 评测命令（示例，串行、CPU）

```bash
# 主端点：event vs event_noise（每 seed 一条，严格串行）
.venv/bin/python tools/head_to_head.py \
  --left event_sN=ckpt:runs/t17event__N__<ts>/agent.pt \
  --right eventnoise_sN=ckpt:runs/t17eventnoise__N__<ts>/agent.pt \
  --seeds 0,1,2 --pairs 400 --bootstrap 4000 --device cpu --workers 4 --json \
  > runs/h2h_eventvseventnoise_sN.json 2> runs/h2h_eventvseventnoise_sN.err

# 归属端点：event vs event_seatblind；vs MLP / vs D1-lite 用对应 ckpt 路径
.venv/bin/python tools/head_to_head.py \
  --left event_sN=ckpt:runs/t17event__N__<ts>/agent.pt \
  --right seatblind_sN=ckpt:runs/t17eventseatblind__N__<ts>/agent.pt \
  --seeds 0,1,2 --pairs 400 --bootstrap 4000 --device cpu --workers 4 --json \
  > runs/h2h_eventvsseatblind_sN.json

# 复用现成对照（示例：vs D1-lite chrono，右臂为 pilot 产物）
.venv/bin/python tools/head_to_head.py \
  --left event_s1=ckpt:runs/t17event__1__<ts>/agent.pt \
  --right seq_s1=ckpt:runs/t17seq__1__1790445087/agent.pt \
  --seeds 0,1,2 --pairs 400 --bootstrap 4000 --device cpu --workers 4 --json \
  > runs/h2h_eventvsseq_s1.json
```

合并**不得**直接照抄 `runs/t17seq_train/aggregate_seq.py:15` 的 `Z=1.96`（`ml-stats-05`）：
本规划用 `q = t_{0.975, k-1}`（k=3 → 4.303），并同时报 z 口径。实现时可扩展一个
`aggregate_event.py`（只读旧脚本、加 `--q`），或手工用两口径报数。

### 5.4 预期效应与功效（引用，不外推）

- 前序同一 h2h 口径的 **boot SE ≈ 11.0**，训练 **seed sd 5.15–20.24**
  （pilot §5.3）。
- z 口径下 3×400 换座合并的 **CI 半宽 ≈ ±12 Elo**；但 seed sd 主导时 **t 口径半宽可达 ±28**
  （红队 `ml-stats-05`），故 k=3 的结论必须标「不可判」而非「CI 排 0」。
- 3×400 对 10 Elo 级效应只有约 **40% 功效**；README §0.2：**20 Elo ≈ 400–500 副，
  10 Elo ≈ 1500–2000 副**（50% power；80% power 约 2×）。
- 结论口径固定为「**没有 ≥+10 的证据**」，不是「效应恰为 0」，并附功效数字。

---

## 6. 分步实施计划

每步独立可测、有验收/回滚；默认路径始终逐位不变（不打开新开关时）。

| 步 | 内容 | 验收 | 回滚 |
|---|---|---|---|
| **0** | 冻结基线 | `uv run --group train pytest -q` 全绿；`Agent(obs_dim,nvec,hidden=128)` 参数量与固定 obs 的 `policy_logits` 快照不变 | — |
| **1** | 引擎 action log：`Play` + `GameState.plays` + `View.plays`（**带默认 `()`**）+ `step` 追加（pass `game.py:181-182`、play `game.py:201-209`）+ `restore=()`；更新 `tests/test_env.py:67/:196/:227/:317` | §3.5 的 `tests/test_game.py` 1–6 全绿；`tests/test_env.py` 既有 v5 fixture 逐位不变；`played`/obs 差分逐位不变 | 删字段，`step` 还原 |
| **2** | `history.py`：`EVENT_DIM=64`、`event_length(rules)`、`encode_events(view, rules, ...)`、`EventHistoryWrapper(env, rules, ...)`；扩展 `__all__` | 新 `tests/test_history.py`：schema 位、相对座位、pass/墩/出空标志、mask/pad、全 games 事件数 ≤ 上界（**用随机策略驱动**，修复 `engineering-bound-test-policy`）、截断取最近、wrapper==直接调用、shuffle 确定性；`encode_history` 回归不变 | 独立函数，默认不启用 |
| **3** | 网络：`EventSequenceEncoder` + `Agent(event_*)` + `save/load_agent`（event 字段 + `history_layout` marker）+ `warm_start_from` layout 校验 + `NeuralPolicy.act` 重建 | 关开关时参数量/数值逐位不变；事件 ckpt 往返；`policy.act == 手工前向`；`event_blind` 不读事件；seatblind/concat/sum 差异正确；事件 ckpt ⟂ MLP ckpt 的 layout 校验 fail-loud | 标志默认 0 |
| **4** | PPO/训练透传：`RolloutBatch` 三字段、`flatten`、`ppo_update` minibatch、`train.py` 存储 + `make_env` + **bootstrap `next_events*`**；枚举所有 `_trunk_input` 到达的签名/调用点 | `flatten` 形状 `(T,E,L,64)→(B,L,64)`；PPO 确实更新事件编码器；无事件时三字段 `None`；minibatch 切片测试；`next_events == wrapper.last_events`（含 blind 置零分支）；端到端 smoke；实测每决策 ms/吞吐并写入报告 | 标志默认 0 |
| **5** | 消融开关：`--event-blind`/`--event-seat`/`--event-order`/`--event-pass`/`--event-boundary-blind`/`--event-noisy` | 每个开关只翻转对应轴（差分测试）；shuffled 确定性 | 标志默认 off |
| **6** | **先出结论**：random 500k 训练 `event`/`event_noise` × 3 seed（+ 归属 `event_seatblind` × 3）；h2h 主端点 + 归属端点 | 3-seed t/z 双口径合并 + 逐 seed JSON；按 §1.5 判定「停 / 加 seed / 继续」；功效数字 | 只产 `runs/`，可弃 |
| **7** | **必做阶段（用户拍板；步骤 6 之后）**：先做 `league.py` 冻结对手透传 + self-play+event smoke；再 `event`/`event_noise` × 3（self-play）；h2h | 冻结对手 layout 与 learner 一致（构造时 + 每次 refresh 后断言）；self-play 端到端 smoke；self-play 信息端点 | 只产 `runs/` |
| **8** | D1-lite 比较 + 报告 | `event vs seq`（复用 `t17seq` ckpt，0 重训；标注混淆）；写实验报告并回填结论 | — |
| **条件步** | 若 §1.5 触发：加 seed 4/5（或 5×400）；若点估计 >+5 的机制端点，做单因素调优（`event_emb`/`event_hidden`/`event_len`） | 同口径 | — |

**「哪一步做完就可以先出结论」= 第 6 步**：random 分布、3 seed 的主端点
`event vs event_noise` 一出来，就能按 §1.5 判断事件表示是否值得继续；若为 null，第 7/8 步
降级为负结果补全。

**步骤 7 的 `league.py` 工作项（修复 `ml-stats-02`、`engineering-selfplay-frozen-opponent`）**：
`build_league`（`src/seven523/league.py:153-167`）构造冻结 `Agent` 时**没有**透传
`seq_len/seq_emb/seq_hidden/seq_blind/seq_order`，也没有 `event_*`；随后
`frozen.agent.load_state_dict(agent.state_dict())`（`:167`，strict）在 learner 有第二输入时
必然因键/形状不匹配抛 `RuntimeError`。因此：
1. 构造 frozen 时用 `copy.deepcopy(agent).eval()`，或显式透传 learner 的**全部** seq/event 配置
   （与 `train.py:404-408` 一致）；
2. 在构造时与每次 `frozen.agent.load_state_dict(agent.state_dict())` refresh 后（`train.py:530-533`）
   断言 frozen 与 learner 的第二输入 layout 一致，否则 fail-loud；
3. self-play + event 的端到端 smoke（`build_league(opponent="self", agent=event agent)`）；
4. 记录：镜像配置下 `--event-blind` 会**自动**传播到冻结对手（`NeuralPolicy.act` 读
   `agent.event_blind`），**不需要** D1-lite obs-B1 那种 `encode_observation` monkeypatch；
   并加「self-play 对手从自己的 `View` 重建事件、相对自己的座位」的测试。

---

## 7. 风险、未决问题、需要的新 ADR

### 7.1 风险（按重要性重排）

1. **吞吐（主风险，修复 `engineering-throughput-framing`）**：事件臂每决策多一次 ≤108 步 GRU
   前向（训练 8 env、推理同）；红队实测 CPU 每决策 MLP 0.267 ms vs seq54 1.553 ms，事件臂约
   3–4 ms。8×128 rollout 下每 step 约 8×3–4 ms ≈ 25–32 ms 的编码开销；须在步骤 4 验收里
   **实测 sps 并写进报告**（既有 seq 6 并发 sps 612 vs pool 3 并发 sps 1057）。**不要**用截断
   `event_len` 当 latency 旋钮（ablation §3.3：len 27 = −18.09）。
2. **内存（次要，量级已算清，修复 `contracts-event-tensor-magnitude`）**：按 `num_steps=128,
   num_envs=8`（`runs/t17seq__1__1790445087/args.json`），`(128,8,108,64)` float32 =
   **27.0 MiB**、seats int64 = 0.84 MiB、mask bool = 0.11 MiB ⇒ 每 rollout ≈ **28 MiB**，
   3 并发约 85 MiB，8GB 卡完全够用。**不是**「比 D1-lite 大一个量级」的风险；比 `(T,E,54)`
   int64（0.42 MiB）约 64×。int64 位掩码压缩是可选存储优化，不改变语义。
3. **`GameState` 每步 tuple 追加**：O(事件数) 复制，事件数 ≤108、对局短；先实测。
4. **泄漏**：除 §3.5 差分测试外，在 `tests/test_env.py:395` 的泄漏测试里加 `plays` 断言。
5. **换座分布外**：相对座位是硬要求；自博弈冻结对手必须从**自己的** `View` 重建（相对它自己）。
6. **trace 回放漂移**：`plays` 不进 trace，但回放必须逐手一致；§3.5 第 5 项兜底。
7. **warm start**：同 arch/nvec 走 strict `load_state_dict` → 报错（更正）；layout marker 校验；
   中性起点默认删除（见 §4.5）。
8. **容量混淆**：`event vs seq` 与 `event vs t17pool` 均混淆容量；只用 `event vs event_noise`
   / `event vs event_seatblind` 归因（`ml-stats-01/03`）。
9. **多重比较 / optional stopping**：confirmatory 族固定 2 端点 + 固定 α；加 seed 只在 §1.5 行 2
   触发（`ml-stats-12`）。
10. **训练 seed 噪声 / 分位数**：seed sd 主导时用 t 分位，标「不可判」；禁止事后挑 seed。
11. **事件上界被打破**：断言 fail-loud，不静默截断。
12. **时序模型选择**：GRU 是唯一被本仓验证过的；attention 未测，首轮不引入。

### 7.2 未决问题（实现前定）

- `event_order=shuffled`：本规划定为**墩内确定置换**（§5.1），不再把它当作全局顺序消融；若将来
  要全局打乱，另起 `event_globalshuffle` 并承认混淆墩边界。
- `--event-wentout drop` 是否成为默认？默认**保留**；消融若显示无用再去。
- `EVENT_LENGTH` 用 `n*54`（严格上界）还是更短？默认**不截断**。
- `GameState.plays` 与 `played` 的冗余：首轮保留 `played`（零风险）；若 `plays` 站稳，后续
  再把 `played` 改成从 `plays` 派生。
- 是否加 LayerNorm？首轮不加，保持与 D1-lite 的最小差异。
- `event_noise` 的分布：默认 i.i.d. 从事件 `(kind,size)` 边际采样（保持尺寸统计），需在报告写死
  种子与分布。

### 7.3 需要的新 ADR

- **本方案本身（opt-in、默认路径不变）依 ADR-0002/0003/0009 不需要新 ADR**，但**必须同步
  owner doc**：`DESIGN.md` §2.4 的 `View` 字段表（`DESIGN.md:154-158`）与 `CONTEXT.md`。关于
  「`View` 加公开字段」有先例：`played`/`last_player` 都是后加的公开字段，未单独立 ADR；
  本方案沿用该先例并在 §3.5 把 `DESIGN.md`/`CONTEXT.md`/`__init__.py`/`history.__all__` 列入
  改动表（修复 `contracts-design-adr-owner-doc`、`engineering-doc-sync-and-exports`）。
  **注意 ADR-0014 的「`Game.view` 不新增字段」是针对 `empty_order` 的条款**，本方案新增的是
  公开的 `plays`，与之不冲突，但需在 owner doc 写明。
- **若保留「零填充中性起点」消融**：必须新立 ADR（记录 `warm_start_event_columns` 的列映射
  规则、payload 契约、测试），因为它复活 ADR-0009 已退役的 `_first_layer_remap` 机制。**该选择
  需要用户决策**；默认删除。
- **若事件臂达标并要 ship（改默认 / 写进产物契约）**，必须新立 ADR（例如 `ADR-0015`），至少固定：
  1. 「第二输入」契约：允许一个**非 obs 布局、仍由 `View` 重建**的历史输入；
  2. ckpt payload 的 `event_*` 字段 + `history_layout` marker 与向后兼容；
  3. 默认开关切换与「默认路径逐位不变」条款的修订；
  4. warm start 规则（含中性起点是否允许）。
- **若要把它折进 obs（v6）**，则需新 ADR 取代/扩展 ADR-0009 的「单一 v5 布局」条款——本方案
  刻意不走这条路。

---

## 8. 长程路线

### 8.1 从无状态重算到真递归（hidden 携带 / BPTT）

- **触发条件**（满足其一才评估）：
  1. 事件臂在某分布达到 repo 门槛（CI 排 0 且 ≥+10）；或
  2. 事件臂点估计 >+5 且在 **两个分布**都为正、并有机制证据（例如 `event vs event_shuffle`
  显著），但出现「局部窗口/末 hidden 不够」的迹象。
  与前序结论一致：D1-lite 连顺序都没测出时**不应**先上真递归（pilot §6.3）。
- **改动面**（对照 pilot §3 的路线 b 与 cleanrl `ppo_atari_lstm.py`）：
  - `train.py`：rollout 增存每个 `(env, seat)` 的 `initial_hidden` 与 `done`，按 episode 重置；
  - `ppo.py`：minibatch 按 **env/序列**组织并重放整段（`envsperbatch` 做法），
    `RolloutBatch` 增 `initial_hidden`；
  - self-play：冻结对手需按 `(env, seat)` 分 hidden 并按局重置；
  - 推理/评测：`NeuralPolicy` 变**有状态**；`eval.py:47-49` 与 `duel.py` 复用 policy 实例跨局，
    必须显式**每局重置 hidden**；`h2h`/`duel` 的策略生命周期假设要写进测试。
  - 代价：本仓 `RolloutBatch`（`ppo.py:52-93`）与 shuffle-minibatch 的 `ppo_update`
    （`ppo.py:143-164`）都要动；训练吞吐按 pilot §3 估计下降（真递归 −20~40%）。
- **与无状态重算的关系**：无状态重算是真递归的**严格下界对照**；先有它的结论，递归的增量
  才有意义。

### 8.2 N > 2 家的扩展

- **seat embedding 天然可扩**：相对下标 `(play.seat - view.seat) % n ∈ [0,n-1]`，embedding
  表随 `n` 变；v5 的 `scores`/`opp_count`/`opp_revealed` 已按 `(seat+k)%n` 自中心旋转
  （`env.py:151-183`），事件臂与之一致。
- **`last_player` 冗余消失**：2 家下 `last_player` 在 v5 里是冗余维（pilot §1.1 指出），
  N 家下它是真实信息；事件臂的墩节奏在 N 家下信息量更大（多方 pass 回合）。
- **事件上界**：`event_length = n * NUM_CARDS`（§4.1 推导；n=3 → 162）。`plays` 上界仍
  ≤54，但每墩 pass 可达 `n-1`。
- **引擎注意**：`_advance` 跳过空手座位（`game.py:217-224`）与 `_end_trick` 的补牌顺序
  （`game.py:245-257`）在 N 家下更复杂；`Play.opens_trick` 的定义（`incumbent is None`）与
  N 家一致，不受影响。赢家推导的补牌角落例外（§2.2）在 N 家同样适用。
- **迁移**：T7 D 线（>2 家）在 `docs/plans.md` 里是远期后置项；本方案不改变其门槛，但为它
  准备了一个可复用的「相对座位事件编码」接缝。

---

## 附录 A：代码锚点索引（本报告引用，已按红队 `engineering-anchor-drift` 复核）

- 引擎：`src/seven523/game.py:26-58`（`GameState`/`played`/`empty_order`）、`:41-45`（`played`
  注释）、`:61-79`（`Deal`）、`:81-87`（`StepResult`）、`:91-107`（`View`）、`:133-148`（`restore`）、
  `:150-168`（`view`）、`:170-215`（`step`，pass 在 `:181-182`、出牌 `replace` 在 `:201-209`、立即撬底在
  `:214`）、`:217-224`（`_advance`）、`:226-313`（`_end_trick`；`:242-243` 累计 `played`、
  `:245-257` 补牌、`:268-270` 撬底者选择、`:305-313` `StepResult` 构造）。
- 动作：`src/seven523/actions.py:235-251`（`action_mask`；pass 条件在 `:262-263`）。
- 观测：`src/seven523/env.py:32`（`OBS_VERSION=5`）、`:106-122`（`_write_unseen`）、
  `:124-135`（`_write_last_player`）、`:151-183`（`_write_scores_rotated` 等旋转）、
  `:186-209`（`_SEGMENTS`）、`:216-229`（`observation_dim`/`encode_observation`）、
  `:232+`（`Seven523Env`）。
- 网络：`src/seven523/networks.py:92-118`（`SequenceEncoder`，`padding_idx=0` 在 `:101`）、
  `:120-218`（`Agent`；`seq_hidden` 默认 `:142`、赋值 `:165`；`_trunk_input` `:207-218`）、
  `:292-347`（`save_agent`/`load_agent`）、`:349-461`（`WarmStart`/`warm_start_into` `:395-427`/
  `warm_start_from` `:432-461`；strict 分支 `:446-449`）、`:463-519`（`NeuralPolicy.act`，
  重建 `:494-515`）。
- 历史：`src/seven523/history.py:30`（`HISTORY_LENGTH=54`）、`:33-39`（`ORDERINGS`）、
  `:42-72`（`encode_history`）、`:75-110`（`HistorySequenceWrapper`）、`__all__` `:22-27`。
- PPO/训练：`src/seven523/ppo.py:52-93`（`RolloutBatch`/`flatten`）、`:143-164`（`ppo_update`）；
  `src/seven523/train.py:305-329`（`make_env`）、`:360-423`（`train` 开头 + warm-start 预检
  `:412-423`）、`:495-575`（rollout 存储）、`:530-533`（self-play refresh）、`:658-684`
  （bootstrap `next_seqs`）。
- 联盟：`src/seven523/league.py:153-167`（`build_league` 冻结对手）、`:168`（`opponent=="mix"`
  才用 `mix_random_prob`）。
- 回放/轨迹：`src/seven523/trace.py:139-162`（`step_record`）、
  `src/seven523/play.py:211-278`（`replay_trace`）。
- 对局/评测：`src/seven523/duel.py:33-46`（换座 schedule）、`:288-333`（`combine_duel_seeds`）、
  `src/seven523/eval.py:47-49`（policy 跨局复用）；`tools/head_to_head.py:39-110`（CLI，
  `--device` 默认 `cpu`）。
- 既有测试：`tests/test_game.py:146`（`test_view_and_observation_do_not_leak_hidden_state`）、
  `:178`/`:199`（`played` 语义）、`tests/test_env.py:67`（`_view` 工厂）、`:196`/`:227`/`:317`
  （直接 `View(...)`）、`:395`（`test_encoding_does_not_leak_hidden_cards`）、
  `tests/test_networks.py:75`/`:181`（warm-start obs_dim 身份）、
  `tests/test_train.py:498`/`:525`（同 layout warm start / SystemExit）、
  `tests/test_ppo.py:121`/`:137`（flatten 形状）、`tests/test_history.py`（D1-lite 一致性全套）。
- 合并脚本：`runs/t17seq_train/aggregate_seq.py:15`（`Z=1.96`）、`:49-50`（合并公式）。

## 附录 B：前序实测数字（只引用，不重跑）

- D1-lite（pilot §5.3）：`seq vs seqblind` **+8.41 [−4.06,+20.89]**（关键端点，3 seed）；
  `seq vs t17pool` +12.44 [−0.07,+24.94]；`seqblind vs t17pool` +2.84 [−20.00,+25.68]。
- self-play B1 盲/有史（pilot §4.2，**单 seed、静态 obs 段**）：历史优势 **+9.85 [−2.97,+22.68]**；
  3-seed random 同通道的 3-seed 口径为 **+2.89 [−9.52,+15.31]**（fusion ablation §0）。
- GRU 顺序消融（ablation §3.2/§3.3）：chrono vs sorted **+10.46 [−2.46,+23.38]**；
  sorted vs seqblind **−1.99 [−24.89,+20.91]**；调优 emb32 **−6.96**、hidden64 **−7.59**、
  len27 **−18.09**（均未放大顺序效应）。
- 机制（ablation §3.4）：chrono ckpt 对全 pad 的 |Δlogit| 1.16–1.62，对顺序置换仅 0.07–0.18；
  |Δvalue| ≤0.064。
- 功效（README §0.2）：20 Elo ≈400–500 副，10 Elo ≈1500–2000 副（50% power）；
  z 口径 3×400 合并 CI 半宽 ≈±12（pilot §8.3），t 口径在 seed sd 主导时可达 ±28。
- 红队实测（本研究的只读复核，`event-history-review.md` 有命令）：
  `SequenceEncoder(16,32)` 对全 pad 输出 absmax=0.0；`seq_blind` 的
  `network.0.weight[:,161:]` 梯度=0.0；`SequenceEncoder` 5680 参数 vs 规划
  `EventSequenceEncoder` 11056 参数；`(128,8,108,64)` float32 = 27.0 MiB。

## 附录 C：对抗性审查与修复记录（摘要）

| finding id | 严重度 | 处置 | 修复落点 |
|---|---|---|---|
| `contracts-view-field-default` | blocker | accepted | §3.2 默认 `()`, §3.5 test_env 构造点 |
| `ml-stats-09` | minor | accepted | §3.2 |
| `engineering-view-default-call-sites` | minor | accepted | §3.2, §3.5 |
| `ml-stats-01` | blocker | accepted | §1.2 重定义 `event_noise`, 弃「扣除容量」措辞 |
| `engineering-event-blind-not-capacity-control` | major | accepted | §1.2, §5.1 |
| `ml-stats-02` | blocker | accepted | §6 步骤 7 加 `league.py` 工作项 + smoke |
| `engineering-selfplay-frozen-opponent` | blocker | accepted | 同上 |
| `contracts-warmstart-factual` | major | accepted | §4.5 更正 strict `load_state_dict` 行为 |
| `engineering-warm-start-claim-wrong` | major | accepted | §4.5 |
| `ml-stats-10` | minor | accepted | §4.5（中性起点默认删） |
| `engineering-adr0009-neutral-start` | major | accepted | §4.5, §7.3（保留需新 ADR = 用户决策） |
| `contracts-obs-dim-identity` | major | accepted | §4.5 加 `history_layout` marker |
| `contracts-design-adr-owner-doc` | major | accepted | §3.5, §7.3 加 DESIGN/CONTEXT |
| `engineering-doc-sync-and-exports` | minor | accepted | §3.5 `__init__`/`__all__` |
| `contracts-test-plan-gaps` | minor | accepted | §3.5 加 test_ppo/test_train/test_env |
| `contracts-event-tensor-magnitude` | minor | accepted | §7.1 具体 27 MiB / 64× |
| `engineering-throughput-framing` | minor | accepted | §7.1 吞吐优先，实测验收 |
| `contracts-blind-caveat-dropped` | minor | accepted | §1.2 对照语义 + §2.3 先验 |
| `contracts-adr0014-wentout-tension` | minor | accepted | §3.4 显式对账 |
| `contracts-encode-events-needs-rules` | minor | accepted | §4.1 `event_length(rules)`, §4.4 签名 |
| `contracts-event-vs-seq-confound` | minor | accepted | §1.2, §4.3 标注混淆 |
| `ml-stats-03` | major | accepted | §4.3 参数 1.95×, §1.2 归因限制 |
| `ml-stats-04` | major | accepted | §0.3, §2.1, §2.3 重述先验 |
| `contracts-B1-evidence-misuse` | minor | accepted | §0.3, §2.3 |
| `ml-stats-05` | major | accepted | §1.5, §5.3 t/z 双报告 |
| `ml-stats-06` | major | accepted | §1.5 分布联合判据 |
| `contracts-first-wave-inconsistency` | minor | accepted | §1.4 精确臂表 + 两层次合并 |
| `ml-stats-07` | major | accepted | §5.1 墩内 shuffle + boundaryblind |
| `engineering-shuffle-determinism` | minor | accepted | §5.1 确定置换 + 测试 |
| `ml-stats-08` | major | accepted | §1.3 self vs mix 语义修正 |
| `ml-stats-11` | minor | accepted | §2.1 a2 不可归因声明 + §5.1 boundaryblind |
| `contracts-multiple-comparisons` | minor | accepted | §1.2/§1.5 预注册族 + α（与 `ml-stats-12` 合并处置） |
| `ml-stats-12` | minor | accepted | §1.2/§1.5 预注册族 + α |
| `ml-stats-13` | minor | accepted | §2.2 赢家例外, §3.4 补牌更正, 附录 A 锚点 |
| `engineering-anchor-drift` | minor | accepted | 附录 A 全面复核 |
| `ml-stats-14` | minor | accepted | §1.4 event_blind 降为可选, 主对照改 event_noise |
| `engineering-bootstrap-next-events` | minor | accepted | §6 步骤 4 |
| `engineering-eval-inference-cost` | minor | accepted | §4.4, §5.2 |
| `engineering-seatblind-weak-contrast` | minor | accepted | §5.1 保守读法 |
| `engineering-bound-test-policy` | minor | accepted | §6 步骤 2 用随机策略 |
| `engineering-selfplay-prior-misattribution` | major | accepted | §1.3 self-play 必做（用户拍板） |

**升级为「需要用户决策」的阻塞项（1 项）**：`engineering-adr0009-neutral-start` —— 是否保留
「零填充中性起点」消融。默认删除；若用户要保留，需新立 ADR 并恢复 ADR-0009 退役的列映射机制。
其余 blocker/major 均在规划内解决。

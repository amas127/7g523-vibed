# test-time 对手意图学习：可识别性闸门与条件训练实验规划（7鬼523 / PPO）

> **状态：阶段 A 已于 2026-09-27 实施并判定——G-I = FAIL，按预注册停止、不进入阶段 B；§4–§6 的阶段 B 设计未实施。执行结果见 [`experiments/opponent-intent-probe.md`](./experiments/opponent-intent-probe.md)。**
> 本文为 repair 后的 revision-2 规划快照（文中行号、环境版本等「本轮核实」均指当时）。
> 逐条红队 findings 与处置见 [`opponent-intent-review.md`](./opponent-intent-review.md)。
> 本文只规划，不改 `src/`、`tests/`、`tools/`、ADR、`plans.md`、`README`、`CONTEXT`、`DESIGN`；
> 不跑训练、不跑 h2h、不 `git commit/checkout/stash`。
> **口径**：revision-3（出空即撬底）、obs v5（2 家 161 维）、trace `version=3`、
> `rules.revision=3`；`rules_id=2e36dbea44893696` 记在 `traces/study/manifest.json`，
> **trace 自身不含 `rules_id`**（回放门禁只看 trace version 与 `rules.revision`，
> `src/seven523/play.py:226-244`）。当前 T15–T17 工作区。
> 所有代码/数据论断给 `file:line` 或可复现命令；未实测的数字一律标注为**预注册约定**。

> **术语纪律（避免本规划被误读）**
> - **可识别 (identifiable)**：公开历史含与对手身份/意图相关的信息，held-out 可读出。
> - **可被 RL 利用 (exploitable)**：该信息能转化为对局收益。二者不同，阶段 A 只判前者。
> - **适应 (adaptation)**：**在固定对手身份的一局内**，后验随已观测事件数增长而使预测/决策
>   变好，且增量**超出「信息本来就更多」的计数基线**。这是本规划唯一承认的
>   test-time 学习证据。
> - **强度 (strength)**：单调的赢棋能力标量；**长度/事件计数 (length/count)**：对局规模。
>   两者都是必须被显式残差化/条件化的 confound，不是「意图」。

---

## 0. 规划摘要（repair 后）

1. **要回答的问题**：在 2 家、revision-3、obs v5 固定下，(1) 公开历史里是否存在可学的**对手意图**
   （必要性）；(2) 若存在，一个无状态重算的时序编码器能否在 **test time** 通过 in-context 后验推断
   对手意图，并转化为**随观测增长**的可测收益。
2. **为什么前序测不到**：历史/顺序/事件/座位四条线都在同质对手下测平均 Elo——random（无身份可学）
   或 self-play（单一身份）。没有「异质 + 身份持久 + 风格可分」的对手池，也从未报**适应曲线**。
   背景：[`sequence-memory-pilot.md`](./experiments/sequence-memory-pilot.md)、
   [`sequence-gru-ablation.md`](./experiments/sequence-gru-ablation.md)、
   [`event-history-pilot.md`](./experiments/event-history-pilot.md)。
3. **闸门必须能失败（核心修正）**：lvl1–lvl4 是**四个不同网络、强度 82.75–185.34 Elo**
   （`traces/study/manifest.json`），`NeuralPolicy.act` 默认 `argmax`（`networks.py:756,824-834`）。
   因此「身份分类」在这些数据上**近乎平凡可过**，且可被**纯 running-score/长度通道**伪造。
   revision-2 把「身份分类」降级为**健全性检查（预期 pass）**，把真正的必要性门移到
   **公开历史对「对手隐藏意图代理」与「强度/长度残差化后的下一手」的可识别性**，并要求
   适应证据**优于只含计数的基线**（§3.6）。
4. **数据比预想的好**：`traces/study` 不只有 subject-vs-random，还有 subject-vs-candidate 的跨等级
   对局（§2.1）；同一 seed 下不同等级打**同一副牌**（本轮实测 200/200 seed 的 `initial.hands` 逐位相同），
   随机锚策略种子按 `(seed, seat)` 固定（`record.py:47`）——这把「发牌」与「身份」解耦，但
   **不解耦观测者自身的动作轨迹**（§2.3）。
5. **阶段 A**：离线、CPU-only、零/极低训练。任务分四层：健全性（身份，预期 pass）、
   **意图代理（主必要门）**、**公开候选集上下一步预测**、**冻结编码器线性探针**；全部配
   score-blind / length-only / 截断 / 打乱 / label-shuffle 对照（§3.3–§3.4）。
6. **阶段 B（仅设计，不执行）**：对手池留出 held-out 身份；aux 主头是**对手当前威胁/意图代理**、
   身份头为低权副头；确认端点是**每条臂分别对固定池成员**逐局记录身份与对手事件数，
   用**身份×观测数的交互**衡量适应，而不是 arm-vs-arm 的平均 Elo。**实现需要新的、显式授权的
   workflow + ADR**（§4、§5.5、§6）。本文件只设计。
7. **落地成本**：阶段 A 是 `runs/opponent_intent_probe/` 下的 numpy + torch-CPU 离线脚本
   （`.venv` 无 scikit-learn / scipy，§5.1）；阶段 B 若要跑，需要改 `src/`，**超出本工作流授权**。

---

## 1. 研究问题与假设的精确化

### 1.1 「test-time 学习对手意图」在本游戏里指什么

- 形式：在不更新权重的前提下，策略在**局内**根据已观测公开历史（`View.plays`、`View.played`、
  `View.trick_cards` 等，`game.py:113-127`）维护一个**对手类型/意图的后验**，并据此调整后续决策
  （in-context / meta-RL 式对手建模）。
- 架构上不被挡住：历史编码器是**无状态重算**——每次决策把完整公开事件前缀重喂时序模型、取末 hidden
  （`history.py:153`、`networks.py:129-207`、`networks.py:746-838`），hidden 不跨决策携带。
  无状态 ≠ 无 in-context 推断：后验是整段前缀的函数，随前缀增长而更新。
- 在本游戏里的具体含义：对手是固定策略身份，其公开出牌风格（偏好牌型、抢分/留牌倾向、残局处理）
  在先验上与该身份相关；观测越多，后验越尖。**对手手牌仍不可见**，可推断的是「策略/意图」，
  不是「具体手牌」。

### 1.2 三种必须分开的假说

- **H1 强度可识别**：公开历史能区分强/弱。近乎必然成立，且**不构成研究结论**。
- **H2 意图可识别**：公开历史能预测对手**当前隐藏意图**（能否压过 incumbent、持分/持炸、威胁等级）
  或**在公开候选集上的下一手**，且优于「只含 running score / 事件计数 / 局面标量」的基线。
  这是必要性门。
- **H3 可适应**：在固定对手身份的一局内，H2 的预测/收益随已观测事件数增长，且**该增长超出
  计数基线的同期增长**（interaction / difference-in-differences）。

只有 **H2 ∧ H3** 支持「公开历史含可被 test-time 利用的对手意图」。H1 单独成立只写
「强弱可识别」。

### 1.3 哪些观察能证伪它

| 观察 | 证伪的命题 |
|---|---|
| 阶段 A：意图代理探针 ≈ 先验；或归零 running score/长度后身份信号 ≈ 先验 | H2 不成立 → 停止，不投训练 |
| 阶段 A：下一手预测不优于**公开候选集先验**（非隐藏合法集） | H2 不成立 |
| 阶段 A：长前缀准确率不高于短前缀，或**不高于 count-only baseline 的同期增幅** | H3 不成立；仅为「信息本来就多」 |
| 阶段 A：真实历史不优于截断/打乱/memoryless | 信号不在顺序/历史里 |
| 阶段 B：hist 臂对**留出身份**的适应曲线不上升，只在已见身份上升 | 只是「记住固定 ckpt」，非 in-context |
| 阶段 B：hist 对 random 池的斜率与对身份池的斜率无差异 | 收益来自静态记忆，非身份适应 |
| 阶段 B：hist 不过长度/奖励分配 | 不可被 RL 利用 |

### 1.4 统一统计口径（沿用仓库既有约定）

- 阶段 A：不用 Elo。分类/预测指标 + **seed 簇 bootstrap**（game 簇嵌套）+ **permutation null**；
  单一主端点 + Holm 校正（§3.6）。
- 阶段 B：3 训练 seed；h2h 走仓库冻结口径（400 副牌换座，`mean ± q·max(boot SE, seed sd)/√k`，
  z/t 双报，[`experiments/README.md`](./experiments/README.md) §3）。行动门槛：
  **CI 完全排除 0 且点估计 ≥ +10 Elo** 才谈「值得行动」；5 分线只触发继续调优。
- **功效纪律（修正）**：开跑前必须报 **MDE 与功效**。`evaluation-protocol-validation.md` 的
  80% power 表为 20 Elo → **807 副**、10 Elo → **3226 副**（README §2 引用）；仓库另建议
  ≥20 Elo 的行动结论用 5×400 或 3×800 复算。3×400=1200 副对 **20 Elo 达 ~80% 功效**，
  但**对 +10 Elo 远不足**。故主端点若声称 +10 级结论，必须按 §4.6 提高牌数或把声明降级为
  「±20 Elo 筛」。

### 1.5 为什么这次不与前序负结果重复

前序四条线都不满足「异质、身份持久、风格可分」的先决条件，且从不报适应曲线。
本规划第一次把**意图代理必要性 + 计数条件化的适应曲线**作为主结构（§3.6、§4.5）。

---

## 2. 数据事实核实（阶段 A 的原料）

### 2.1 `traces/study` 的真实结构（本轮实测，命令见附录 A.1）

- 共 **4000 局**、`version=3`、`rules.revision=3`。trace **不含 `rules_id`**；`rules_id` 只在
  manifest。回放门禁校验 trace version 与 `rules.revision`（`play.py:226-244`）。
- 目录 `lvl1/lvl2/lvl3/lvl4` 按 subject 分目录，每目录内含多种对手：

  | 目录 | vs random | vs lvl2 | vs lvl3 | vs lvl4 | 合计 |
  |---|---|---|---|---|---|
  | `lvl1/` | 400 | 400 | 400 | 400 | 1600 |
  | `lvl2/` | 400 | — | 400 | 400 | 1200 |
  | `lvl3/` | 400 | — | — | 400 | 800 |
  | `lvl4/` | 400 | — | — | — | 400 |

  ⇒ 跨等级对局存在（6 对 × 400），可直接做 lvl_i vs lvl_j 区分。
- 命名 `g{全局序号}__s{seed}__seat{human_seat}__vs{opponent}.json`（`trace.py:164`）；
  `players` 形如 `["subject:lvl1@seat0","anchor:random@seat1"]` 或
  `["subject:lvl1@seat0","candidate:lvl2@seat1"]`（`trace.py:120,125`）。
- **配对结构**：random 锚块 200 seed × 8 局；跨等级 200 seed × 12 局；两块 seed 交集为空。
  同一 seed 内各等级打**同一副发牌**（实测 200/200 seed 的 `initial.hands` 逐位相同）；
  `record.policy_seed(seed, seat)` 按 `(seed, seat)` 固定策略种子（`record.py:47`）。
- manifest levels：random 0.0 / lvl1 82.75 / lvl2 106.90 / lvl3 147.71 / lvl4 185.34；
  四个 subject ckpt 本轮核实均存在（附录 A.4）。
- `traces/pool10/` **只有 `manifest.json`，无对局**；`runs/*/games.jsonl` 只有逐局结果，
  无逐步历史——阶段 A 原料就是 `traces/study`。

### 2.2 trace schema 与回放能力

- 单条 trace：`version`/`created_at`/`rules`（含 `revision`）/`seed`/`human_seat`/`players`/
  `initial`/`steps`/`final_scores`（`trace.py:172`）。
- `initial` 含 `starter/hands/draw_pile/revealed`（`trace.py:84`）——**含隐藏牌，绝不可作特征**。
- `steps` 每步 `step_record`（`trace.py:139`）：`seat/action/suit/text/scores/hand_sizes/...`。
- 回放：`play.py:211` `replay_trace` 用 `state_from_snapshot`（`trace.py:112`）+`Game.restore`
  （`game.py:157`）+`Match` 逐步驱动；`trace version <3` 或 `rules.revision != RULES_REVISION`
  直接拒绝（`play.py:229-243`）。探针应复用同一套 `Game.restore`+`Game.step`（`game.py:195`）。
- 任一时刻可构造公开 `View`：`game.view(state, seat)`（`game.py:174`）；实测 `View.plays`
  含对手出牌/过牌但不含对手手牌。

### 2.3 可见 / 不可见（泄漏面）清单 + 观测者自反通道

| 类别 | 字段 | 探针可否作输入 | 说明 |
|---|---|---|---|
| 公开 | `View.plays`（座位、组合、开墩、出空；`Play` `game.py:27`） | ✅ | **主通道**，按 `play.seat` 可切 `opponent_only` |
| 公开 | `View.played` / `View.trick_cards` | ✅ | 无座位/墩界 |
| 公开 | `View.counts/draw_count/scores/revealed/incumbent/last_player/current` | ✅（`scores` 仅作 confound 残差化/对照，不作主特征） | v5 已覆盖大部分 |
| 公开 | `encode_events`（`history.py:153`）/`encode_observation`（`env.py:221`） | ✅ | 观测者自己手牌合法可见 |
| **隐藏** | `initial.hands`、`initial.draw_pile` | ❌ | 直接泄漏对手手牌；D1 中且跨类恒定（§3.8.4） |
| **隐藏** | 引擎 `state.hands[opponent]`、`state.empty_order` | ❌ | 只可离线作 y |
| **未来** | 该决策点之后的 `steps`、`final_scores` | ❌ | 截断控制保证按前缀取特征 |
| 标签侧 | 对手身份、engine 导出的意图代理标签 | ✅（仅作 y） | y 可来自 engine，x 必须纯公开 |
| **观测者自反** | 观测者自身动作日志 | ⚠️ | 观测者动作依赖对手行为（`policies.py:62-63` 的 `legal_ids` + `rng`），是**信息性 nuisance**，不是 pipeline 泄漏；主特征用 `opponent_only` |

---

## 3. 阶段 A：可识别性探针（离线，CPU-only，零/极低训练）

### 3.1 数据构造

**样本单位**：`(game, observer_seat, prefix)`。回放一条 trace，在观测者的**每个决策点**
（`state.current == observer_seat`）取当前公开 `View` 作为前缀；标签取对手身份或意图代理。

**数据集**：

- **D1 主身份集（4 类，观测者固定 random）**：random 锚块。观测者 = random 座位，
  目标 = subject ∈ {lvl1..lvl4}。用于**健全性检查（预期 pass）**，不作为唯一必要门。
- **D2 跨等级集（固定观测者的子任务，禁止混合观测者）**：跨等级块。**每个分类器固定一个
  观测者身份**，例如「观测者 = `candidate:lvl1`，目标 ∈ {lvl2,lvl3,lvl4}」的三分类，
  以及「观测者 = `candidate:lvl2`，目标 ∈ {lvl3,lvl4}」的二分类等；对每个子任务预注册
  样本量与先验。**禁止**「或反向」把不同观测者混进同一分类器（finding leakage-method-10）。
  跨等级局可对**同一 deal 取两个方向的 View**（`game.view(state, subject_seat)` 与
  `game.view(state, candidate_seat)`），从而在固定发牌下比较方向（finding stats-ml-19）。
- **D3 二分类集（每个 lvl_i vs random）**：random 锚块，双向观测，报 AUC（仅诊断，不作门）。
- **D4 意图代理集（引擎标签，公开特征）**：在 random 锚块与跨等级块的观测者决策点，
  由 `state.hands[opp]` 离线导出 y（**仅 y**）：对手当前是否持有 ≥1 张分牌、是否持有炸弹、
  是否能压过 incumbent（存在合法且更优的出牌）、当前手牌数/分牌数。x 纯公开。

**切分**：`GroupKFold(n_splits=5)`，group = seed；同一 seed 的所有局（不同等级、两座位）
必须同折。禁止按 game index 随机切。

**先验基线**（每个任务都要报）：
- 分类：majority 先验、以及「只用 memoryless v5 obs」的分类器。
- 下一手：公开候选集上的 uniform 先验与全局边际先验（**不用隐藏合法集**，§3.4）。
- **count/length-only baseline（强制对照）**：只用事件计数、对局步数、局面标量（无事件内容）。
- **score-only baseline（强制对照）**：只用 running score 差/lead。

### 3.2 特征（只允许公开信息）

全部从回放得到的 `View` 抽取，**不读 `trace["initial"]`**：

1. **事件序列表征**（`encode_events`，`history.py:153`，(events,seats,mask)，`EVENT_DIM=64`）：
   - `opponent_only`（**主视图**）：只保留 `play.seat != observer_seat` 的事件；
   - `full`（仅诊断）：观测者 + 对手全部事件。
2. **score-blind 事件特征**：从事件张量/摘要中移除 running score、`trick_points`、
   `remaining_points`、outcome 相关分量后重算（或把 score 作为协变量残差化）。
3. **memoryless 对照特征**：`encode_observation(view, rules)`（`env.py:221`，161 维）。
4. **前缀摘要（轻量分类器/适应曲线）**：对手事件数、各 `ComboKind` 计数、过牌率、平均组合大小、
   最近 k 个对手事件 one-hot、公开局面标量（`counts/draw_count/revealed/incumbent/last_player`）。
5. **长度/计数字段单独列出**，既有 baseline 用途，也用于分层与残差化。
6. **时间截断**：只喂前缀的前 m 个对手事件（m 由 §3.5 的实测分布定，见 §6 步骤 1）。

### 3.3 任务（主/辅明确）

- **P0 身份分类（健全性，预期 pass，非必要门）**：D1/D2/D3。**必须**同时报
  (a) 含 score/length 的 full 特征、(b) score-blind 特征、(c) 仅 score/length baseline。
  若身份信号只在 (a) 成立，结论写「强弱可识别」，**不得**触发阶段 B（finding leakage-method-01、
  engineering-contracts-length-confound）。
- **P1 意图代理（主必要性门）**：D4。预测对手当前隐藏意图代理，x 纯公开、**score-blind 主报**。
  指标：accuracy / macro-AUC / NLL vs 先验；并报 count-only baseline 的同指标。
- **P2 下一手预测（主必要性门之一）**：
  - **A2c（主，公开候选集）**：预测对手下一手的 `ComboKind`（pass + 非空族；见 §3.5 的类定义）
    与 action 模板 id，候选集只用**公开可枚举空间**，先验 = 全局边际。**不**用隐藏合法集归一。
  - A2a：`ComboKind` 分类（类分布见 §3.5，含空类处理）。
  - A2b（诊断）：引擎合法集归一，仅报「相对合法集先验的 lift」，并注明不可与公开基线比较
    （finding leakage-method-09、stats-ml-15）。
  - 指标：top-1、macro-NLL、随事件桶曲线。
- **P3 风格 vs 强度（可失败的鉴别）**：
  - **强度残差化**：在特征上回归掉 running score / 事件计数 / 对局阶段，看身份/意图信号是否保留；
  - **count-matched 分层**：在匹配了对手事件数的子集内比较身份可分性；
  - 若有强度匹配的检查点对（同 lineage、Elo 相近），做直接二分类。**若三法中无一保留信号，
    则结论只能是 H1（强度），不是 H2/H3。**
- **P4 冻结编码器线性探针（诊断）**：加载 `runs/t17event__1__1790452506/agent.pt`（本轮核实存在，
  `event_len=108`），冻结 `EventSequenceEncoder`（`networks.py:129`），用其 hidden 跑线性分类，
  回答「现有未做对手建模的编码器里身份/意图是否已线性可读」。

### 3.4 对照（每个任务都要有）

| 对照 | 构造 | 预期（若信号真实） |
|---|---|---|
| label shuffle | 训练集内打乱 y（分层内），重训同模型 | ≈ 先验 / chance |
| 历史截断 | 只用前 m 个对手事件（m∈{6,12,18,24}） | 低于 full |
| 历史打乱 | 对手事件在墩内固定置换（`history.py:122`） | 低于 chrono |
| memoryless | 只用 `encode_observation`（v5 obs） | 低于 full-history |
| **count-only / length-only** | 只用事件计数、对局步数、局面标量（无事件内容） | 不能复现主信号；作为**效应下限对照** |
| **score-only** | 只用 running score 差 | 不能单独满足 P1/P3 |
| blind/noisy（容量/计数） | `encode_events(blind=True)` / `noisy=True`（`history.py:161-162,209-215`） | **注意**：二者都保留真实 `mask` 计数（`mask[position]=True` 先于 `continue`），故**不是**零信息；标为「保留计数的容量对照」，并另设「全 pad（mask 全 false）」作为**真正零信息**对照（finding stats-ml-13、engineering-contracts-controls-not-zero-info） |
| 观测者自身 | 只用观测者自己的动作/摘要特征 | **实测非 chance，是信息性 nuisance**（§2.3）；报其数值，不当作泄漏 |
| oracle 哨兵 | 用**决策点当时** `state.hands[opp]` 作特征（仅此一处） | 接近满分 |
| 时长匹配 | 按对手事件数分层的子集内比较 | 用于剥离长度 confound |

**D2 的方向对照**：跨等级局用**同一 deal 的两个方向**（subject 视角与 candidate 视角），
固定发牌下的方向对比才是 D2 的干净效应（finding stats-ml-19）。

### 3.5 指标与适应曲线

- **类定义（修正）**：下一手族 = pass + 实际出现的 `ComboKind`。本轮实测 41,183 个 subject 动作
  的边际：single 0.585、pass 0.182、straight 0.105、pair 0.100、small_bomb 0.028、
  big_bomb 0.0005、consecutive_pairs ≈ 0。故 majority 先验 = 0.585，**空类需合并或删除**，
  报 per-class 指标与 macro-NLL（finding stats-ml-14）。
- **适应曲线（重定义，配对）**：
  - 单位 = **同一局内的决策点**；按该点**已观测对手事件数** e 分桶。
  - **桶界必须由 per-prefix 分布（不是 per-game 总数）确定**。本轮实测（50 局/级，共 200 局）：
    per-prefix e 范围 0–35、中位 13、mean 13.0；若沿用 [0,6)/[6,12)/[12,18)/[18,24)/[24,∞)，
    样本数约 1100/1200/1199/1095/534（B5 仅 ~10%）。**旧规划的「16–36、中位 26」是 per-game
    总量，用错了量**（finding engineering-contracts-measured-bucket-stat-wrong、
    stats-ml-02/16）。桶界在步骤 1 用 extract.py 的实测 per-prefix 分布重定（建议
    [0,4)/[4,8)/[8,12)/[12,18)/[18,∞) 或等频桶），并报告 **per-class × per-bucket 样本数**。
  - **主适应度量 = 配对 difference-in-differences**：
    `Δ = [acc_history(late) − acc_history(early)] − [acc_countonly(late) − acc_countonly(early)]`，
    在同一局、同一决策上下文（同一墩阶段/是否首手）内取对；报 seed/game 簇 bootstrap CI。
    **要求 Δ 的 CI 下界 > 0**（历史模型相对计数基线的**额外**增长），而不是单看 `acc(B5)−acc(B1)>0`
    （finding stats-ml-03）。
  - **类内曲线**：每个等级自己的 acc vs e 曲线；**桶内类别构成必须报告**，构成漂移即报告为
    confound，不作为适应证据（finding stats-ml-03）。
  - **口径（预注册）**：主报「**per-bucket 模型**（在该桶训练样本上拟合、在该桶测试样本上评估，
    信息预算 = 桶内平均 e）」；同时报「在 full 前缀上训练、在截断前缀上评估」的 fixed-full
    诊断。两种口径都写进报告模板（finding engineering-contracts-adaptation-curve-metric-ambiguous）。
- **CI 与零分布**：seed 簇 bootstrap + 嵌套 game 簇；同时用**重复分组 CV**（多次 GroupKFold
  重排）反映拟合方差，而不是单切分 bootstrap（finding stats-ml-07）。permutation null
  用分层内标签置换多次重训或置换检验的 99 分位。
- **冻结编码器线性探针（诊断）**：见 P4。

### 3.6 预注册闸门（开跑前冻结；数值为约定，须用户确认）

> 本节数字是**预注册约定**。报告必须原样标注，不得事后调整。步骤 1 先用 extract.py 实测
> per-prefix 分布与 pilot 方差，据此定桶界与阈值的 MDE，再冻结。

**单一主端点 + 多重比较控制**（finding stats-ml-06、leakage-method-08）：
- 族 = {P0, P1, P2, P3, P4 各主指标}；**主端点 = P1 意图代理的 score-blind held-out accuracy 优于
  count-only baseline**（单侧）。
- 其余为辅助端点，用 **Holm** 控制 FWER（家族内）。**删除**原 G-A1.2 的 OR 规则与「≥3/4 AUC」
  选择式判据。
- 所有阈值必须由步骤 1 的 pilot 方差给出 **MDE 与功效**，不得留空。

**闸门定义（修正后）**：

- **G-S（健全性，预期 pass，非必要门）**：P0 身份分类在 score-blind + 残差化后仍显著优于
  count-only baseline。**仅仅 G-S 成立不触发任何训练**；若只在含 score/length 特征时成立，
  结论写「强度/长度可识别」。明确承认：对 4 个不同强度网络，G-S 近乎必然 pass
  （finding stats-ml-01、engineering-contracts-gate-weak-orders-budget）。
- **G-I（意图，必要性门）**：G-I.1 P1 score-blind accuracy 的 95% CI 下界 > count-only baseline
  的 99 分位（Holm 校正后）；**且** G-I.2 在 count-matched 分层内仍成立。两条都过才算 H2。
- **G-F（可失败的风格鉴别）**：P3 的强度残差化/count-matched 分层中至少一种保留显著信号。
  若只有 P0 过而 G-F 不过 → 只写强度。
- **G-D（适应，唯一允许触发阶段 B 的机制门）**：P1/P2 的配对 difference-in-differences
  `Δ` 的 CI 下界 > 0（Holm 校正后）。**G-D 不成立则无论 G-S/G-I 如何，都不得启动阶段 B 训练**
  （finding leakage-method-02、stats-ml-04 断言 arm-vs-arm 无法测适应；本规划据此把适应门
  放在阶段 A 的 within-game 配对，而非阶段 B 的 h2h）。

**中止规则**：
- G-I 不成立（或只在含 score/length 特征时成立）→ 结论「公开历史里没有可学的对手意图」
  → **停止，不投训练**，写负结果（§8.2）。这是真正的止损点。
- G-I 成立、G-D 不成立 → 结论「意图可识别但无可测的 test-time 适应」，可写「静态身份标签」；
  **不启动阶段 B 训练**（或仅以新的、单独授权的诊断 workflow 做，不烧 9–12 个 500k run）。
- G-I ∧ G-D ∧ G-F 成立 → 才进入阶段 B 设计评审（仍需新授权，§6）。

### 3.7 阶段 A 假阳性 / 假阴性的后果

- 假阳性：会白烧阶段 B 预算。缓解：主门移到意图代理 + 计数条件化 + 强度残差化，
  并在 G-I 不满足时硬中止。
- 假阴性：会过早关闭路线。缓解：报 per-class/AUC/confusion、`opponent_only`/score-blind 两视图、
  以及固定 full-model 的截断诊断。
- 无论哪种，都**不**把探针结论直接当 Elo 结论。

### 3.8 泄漏自检（执行清单，代码里落地）

1. 特征函数签名只接受 `View`+`Rules`；禁止访问 `state`/`initial`（模块级包装器强制）。
2. **静态扫描**：`initial` 只允许出现在 `state_from_snapshot`；`state.hands` 只允许出现在
   标签/合法集/oracle 哨兵处（grep 检查）。
3. **未来信息**：把某前缀之后的所有 `steps` 替换为空，特征逐位不变（单测）。
4. **oracle 哨兵（修正）**：必须用**决策点当时的** `state.hands[opp]`（逐决策变化）。
   **D1 的 `initial.hands` 跨类别恒定**（本轮实测 200/200 seed 四档开局手牌逐位相同），
   用它作 oracle 会落到 chance，**不得**用作哨兵（finding leakage-method-07）。
5. **seed 切分断言**：训练/测试 seed 交集为空；同一 seed 的 4 档必须同折。
6. **回放完整性**：每条 trace 用与 `play.py:211` 相同的 restore+step，断言 `final_scores` 一致。
7. **label shuffle** 必须落到先验（否则特征/样本构造有 bug）。
8. **confound 哨兵**：仅含 score 或仅含 count 的 baseline 不得复现主端点。

---

## 4. 阶段 B（条件）：训练实验设计（**仅设计，需新授权**）

> 阶段 B 需要改 `src/`、加 aux 头与 env info 字段和测试，并可能触发 ADR。
> 本工作流（Plan/Red-team/Repair/Verify 只写两个 docs；Probe 期不得改 `src/tests/tools/ADR/
> plans.md/CONTEXT/DESIGN`）**不含该授权**。本节是供后续独立 workflow 评审的设计草案，
> 不是可执行验收（finding engineering-contracts-stageb-out-of-contract）。

### 4.1 研究问题与端点（重新定义，去除 arm-vs-arm 的平均 Elo 主端点）

- **主确认端点（identity-variation）**：每条训练臂分别对**固定对手池的每个成员**（含 held-out）
  各打 N 副牌，**同一 deal 换座配对**；逐局记录 `(arm, member, deal, 胜负/回报, 该局对手事件数,
  该局对手身份)`。用 `ladder.play_games`/`plan_games` 的 schedule + `results_out` JSONL
  （`ladder.py:527-555` 已逐局写 `opponent`/`subject_seat`/`kind`/`rules_id`）或新增等价评测路径。
  **不再**用 `tools/head_to_head.py` 的 arm-vs-arm 作为适应端点：其语义是 candidate-vs-candidate、
  每 deal 只有一个对手身份（`tools/head_to_head.py:1-37`），无法拆出「hist vs 共同对手」的
  身份适应（finding leakage-method-02、stats-ml-04）。
- **适应端点**：per-member 的「胜率/回报 vs 已观测对手事件数」曲线的**斜率**，及其
  **身份间交互**（不同成员斜率不同、held-out/难读成员更高）。适应签名 = 交互，而不是
  一个非零的整体斜率（finding stats-ml-04）。
- **机制端点**：`hist_aux` vs `hist_noaux`（aux 贡献）、`hist_noaux` vs `eventblind_noaux`
  （事件通道）、`hist_aux` vs `count_only_noaux`（内容 vs 计数）。
- **可比性**：所有臂在同一 seed/deal 集上评测；合并按仓库冻结口径。

### 4.2 对手池与 held-out 身份

- 训练池与留出身份**必须分开**：例如训练池 = `random + lvl1 + lvl2`（+ 可选同 lineage 快照），
  留出身份 = `lvl3 + lvl4`（以及一个训练池成员的**另一快照**）。所有端点都必须在**留出身份**
  上重复；只在已见身份上升不算 in-context（finding leakage-method-05）。
- 身份持久：`--opponent pool --pool-episode`；`EpisodeMixturePolicy` 在 `env.reset` 的
  `start_episode()` 冻结成员（`policies.py:105,177`、`env.py:313-314`），整局不变。
  `MixturePolicy`（逐决策重抽）不用。
- **多样性报备**：报池成员两两的策略/动作分布散度；成员少且同 lineage 时，null 结果只能解释为
  「在该池上无适应」，不能外推（finding stats-ml-17）。

### 4.3 aux 对手建模头（训练期特权信号，推理/评测不启用）

- **主头 = 意图/威胁代理**（对手当前能否压过 incumbent、持分/持炸、手牌数），**不是**身份。
  身份头降为**低权副头**；理由是身份是「最容易」而非「与研究问题最相关」的标签，且身份
  episode 恒定、每 batch 有效样本极少、有表征塌缩风险（finding stats-ml-09）。
- **标签边界（修正）**：aux 标签在 `env.step` 返回的 `info` 内捕获（learner 行动后、
  `match.advance()` 推进对手之后确定的「对手本回合首个动作/意图代理」）；**不得**在
  vector step 之后读 `current_id`。SAME_STEP autoreset 下（`train.py:620-623`）终局 transition
  必须用 `final_info` 区分、并 **mask 掉终局与「无对手动作」样本**；`train.py:829` 已示范终局
  读 `finished_id` 的语义（`policies.py:176-190`）（finding leakage-method-12、
  engineering-contracts-samestep-aux-label）。
- **hidden 选择（修正）**：aux 头读**哪个 hidden**必须预注册。若预测「对手下一手」，该动作在
  learner 行动之后，故要么读 **post-action hidden**（需显式构造），要么把目标
  **条件于 learner 刚选的 action**（预测 `P(opponent action | learner action)`）；默认改为预测
  **状态型意图代理**（不依赖 learner 刚行动），避免不可预测噪声（finding stats-ml-10）。
- **接缝（修正）**：`Agent` **没有 `forward`**；实际接口是 `policy_logits`（`networks.py:414`）、
  `get_value`（`:427`）、`get_action_and_value`（`:439`）。aux 头挂在 `_actor_hidden`/
  `_trunk_input` 的 hidden 上，由单独 `aux_logits(...)` 暴露；`policy_logits`/
  `get_action_and_value` 的返回签名不变，`NeuralPolicy.act`（`networks.py:768`）路径逐位不变
  （finding engineering-contracts-agent-forward-missing）。
- **损失**：`L = L_ppo + λ_intent·CE_intent + λ_id·CE_id + λ_action·CE_action`，`λ=0` 时逐位等价。
  **λ 必须先离线/单 seed pilot 定标并写进预注册**（连 pilot 一起记入预算），不得留「开跑前
  sweep」的未冻结项（finding stats-ml-11）。报 aux accuracy 与**有效样本数**作诊断。
- **aux 对齐**：机制对照必须在 aux 维度上对齐——要么所有对照臂都带同结构 aux 头（零信息输入），
  要么把主对照定义为 `hist_aux` vs `hist_noaux`，另设 aux-only 臂（finding
  engineering-contracts-arm-matrix-aux-confound）。

### 4.4 臂与对照（aux 对齐）

| 臂 | 事件通道 | aux | 用途 |
|---|---|---|---|
| `hist_aux` | event-len 108（`train.py:187`） | intent（主）+ id/action（低权） | 处理臂 |
| `hist_noaux` | 同上 | 无 | 事件通道贡献（对齐 aux） |
| `eventblind_noaux` | event-len 108 `--event-blind`（channel 在、内容零） | 无 | 事件内容 vs v5 内建记忆 |
| `eventnoise_noaux` | `--event-noisy`（**保留计数**，非零信息） | 无 | 计数/容量对照（明确标注） |
| `count_only_noaux` | 全 pad（mask 全 false） | 无 | **真零信息**容量对照 |
| `hist_aux_randompool` | 同 `hist_aux` | 同 | 同质池回退检查（斜率对照，非点检验） |

命名修正：`poolblind` 不再叫「memoryless/完全无历史」——event-blind 只零化事件向量与 seat
（`history.py:161,172`），但 v5 obs 仍含 `unseen`（`env.py:106-120`，随出牌收缩）、`scores`、
`opp_counts`、`last_player` 等历史相关段（finding leakage-method-06）。

### 4.5 适应曲线指标（identity-conditional）

- 每条评测局记录 `(member_id, deal, 累计/总对手事件数 e, 胜负/回报)`。
- 对每个 member：胜率/回报 vs e 的斜率；对每条臂：member×e 的交互项（混合模型或分层斜率差）
  + member/seed 簇 bootstrap CI。
- **适应证据 = 交互**（held-out/难读成员斜率更大），而非整体斜率 > 0；同时对照
  `count_only_noaux` 的同曲线，要求处理臂的斜率增幅**超出**计数臂（finding stats-ml-04）。
- **同质池回退**：`hist_aux_randompool` 的「身份池斜率 − random 池斜率」作为预注册的
  **斜率差**端点（有自己的 CI），不对 0 做点检验；并说明前序 +8.41[−4.06,+20.89] 与
  +9.85[−2.97,+22.68] 的噪声预期（finding stats-ml-18）。

### 4.6 训练预算、功效与设备

- 沿用 EVH pilot：`--total-timesteps 500000`、`num_envs=8`、`num_steps=128`、`hidden=128`、
  shared、lr 2.5e-4 退火。
- **功效（修正）**：开跑前对每端点报 MDE/功效。主端点若声称 +10 Elo，按 README §2
  的 80% power 表需 **≥3226 副**（10 Elo）或降级为 ±20 Elo 筛（807 副）。3 seed × 400
  = 1200 副对 20 Elo 约 80% power，对 +10 不足（**修正** finding stats-ml-05：其「1200 连
  20 Elo 都不够」与所引 807 副表不符；正确表述是 10 Elo 不够、20 Elo 勉强）。
- 臂 × seed：`hist_aux`/`hist_noaux`/`eventblind_noaux`/`count_only_noaux` × 3 seed = 12
  （视需要加 random-pool 臂），≤3 进程并发；aux λ pilot 单独计一档预算。
- **设备（显式）**：现有 run 默认 `cuda=True`（`train.py:505`）；阶段 B 若沿用须在授权里
  明确允许 CUDA 并写并发上限；否则强制 `--device cpu`。阶段 A 强制 CPU-only。
- 评测：同一时刻只跑一个评测进程；进程内用 `--device cpu --workers 4`（finding
  engineering-contracts-h2h-serial-workers-wording）。

### 4.7 验收门

- **A（行动级）**：主端点合并 CI 排除 0 且 ≥ +10 Elo（或明确降级为 ±20 筛）。
- **B（适应级）**：per-member 身份×事件数交互显著，且处理臂斜率增幅 > 计数臂。
- **C（机制）**：各对照按 §4.4 归因；aux 贡献由 `hist_aux` vs `hist_noaux` 单独判定。
- 结论映射：A∧B∧C=支持；A∧¬B=静态记忆；G-D（阶段 A）不成立=不可测适应；¬A=不可被 RL 利用。

---

## 5. 实现面

### 5.1 阶段 A：脚本与依赖（`runs/opponent_intent_probe/`）

- 脚本：`extract.py`（回放 → 前缀样本 + 自检）、`probe_intent.py`（P1/P3 + 曲线）、
  `probe_identity.py`（P0/D1-D3 + 对照）、`probe_next_action.py`（P2/A2a-c）、
  `frozen_probe.py`（P4）、`bootstrap.py`/`report.py`。
- **内存/磁盘预算（修正）**：不要全量常驻 `(108,64)` 事件张量。extract.py 只存**紧凑摘要**
  （计数/ComboKind 直方/最近 k 事件/观测者手牌 one-hot）；GRU/事件张量**按需在线重算或分批缓存**。
  若全量：4000 局 ≈ 20 万前缀 × 27.6 KB ≈ 5.5 GB（两观测者翻倍）。在步骤 2 的验收里写
  峰值内存与产物大小上限（finding engineering-contracts-event-tensor-memory）。
- 依赖现状（当时核实）：`.venv` 有 `numpy 2.5.3`、`torch 2.14.0+cu132`；**无 scikit-learn、无 scipy**
  （其后 `pyproject.toml` 已加入 `scipy>=1.18.1`；scikit-learn 仍无）。
  探针走 numpy-only + torch-CPU，不新增依赖。
- **脚本版本化（修正）**：`runs/` 被 `.gitignore` 忽略且禁止本工作流 commit，故探针**脚本全文/
  关键片段必须内联进 `docs/experiments/opponent-intent-probe.md`**（docs 受控可写），
  报告附环境 hash，使结论可独立复核；明确记录「runs/ 脚本不版本化」这一限制
  （finding engineering-contracts-gitignored-probe-scripts）。

### 5.2 阶段 B：模块清单（**设计草案，需新授权**）

| 文件 | 拟改动 | 不变式 |
|---|---|---|
| `src/seven523/networks.py` | 在 `_actor_hidden`/`_trunk_input` hidden 上挂 aux 头，新增 `aux_logits(...)`；`history_layout`（`:490`）纳入 aux 标记；`save_agent`/`load_agent` 持久化 aux 配置 | `aux=None`/λ=0 时前向数值与参数逐位不变；`policy_logits`/`get_action_and_value` 签名不变；`NeuralPolicy.act` 路径不变 |
| `src/seven523/env.py` | `step`（`:321`）在 `info` 暴露意图代理/成员 id（训练期 aux 标签） | 观测 `_SEGMENTS`（`:186`）/`encode_observation`（`:221`）/`_publish`（**:387**）不变 |
| `src/seven523/ppo.py` | `RolloutBatch`（`:52`）加 aux 标签/掩码；`ppo_update`（`:167`）加 aux 损失 | λ=0 时逐位等价 |
| `src/seven523/train.py` | CLI `--aux-*-weight`；采集标签、透传 batch | 默认 0 → 默认路径不变（ADR-0003） |
| `src/seven523/league.py` | 无需改 | `self/mix/pool` 分支**无条件**断言 `history_layout(frozen)==history_layout(agent)`（`:152-166`）；非 self 成员走 `policy_from_spec`，无布局耦合（**修正** finding engineering-contracts-league-assert-mischaracterized） |
| `tools/head_to_head.py` / `eval.py` / `ladder.py` | 评测路径不再用 head_to_head 作适应端点；用 `ladder.play_games`/`plan_games` 的 per-member schedule + JSONL（`:527-555`）或新增等价路径 | 若新增路径，须写明改动清单 |

### 5.3 ckpt / league 影响

- 阶段 A 不产生 checkpoint。
- 阶段 B 新 run 是新 checkpoint；`history_layout` 加 aux 标记后，warm-start 严格
  `load_state_dict` 需按标记匹配。默认（无 aux）布局字符串保持与现状一致，旧 ckpt 仍可加载。

### 5.4 资源与并发纪律

- 阶段 A：CPU-only，多核 numpy/torch-CPU；不开 h2h；新对局串行 CPU、≤2000 局、优先复用
  `traces/study`。
- 阶段 B：训练最多 3 进程；同一时刻只运行一个评测进程；`--device cpu --workers 4`。

### 5.5 阶段 B 的前置授权条件（**新增，硬性**）

1. 有可回滚的干净基线：**先请用户授权把 T15–T17 落成一个提交**（本工作流不得 commit），
   或规定改动只落新增文件 + `# BEGIN/END EXPERIMENT` 标记块 + 反向补丁；否则不得批准实现
   （finding engineering-contracts-stageb-rollback-uncommitted）。
2. **ADR-A 通过**：训练期 aux 头 opt-in、训练期特权、默认 λ=0 逐位不变、不改 obs v5/评测接口、
   checkpoint 布局标记。
3. 用户明确授权改 `src/tests`（本工作流之外）。
4. 阶段 A 的 G-I ∧ G-D ∧ G-F 全部通过。

---

## 6. 分步实施计划（每步验收 / 回滚）

> 阶段 A 的每步是 `runs/` 产物 + 报告，不做不可逆改动；回滚 = 删 `runs/opponent_intent_probe/`。
> **步骤 6–10 是阶段 B 实现，超出本工作流授权，标记为「需新授权」**。

1. **核实与冻结预注册**：用 extract.py 实测 **per-prefix 事件数分布**与 pilot 方差，据此定桶界、
   `L*`、阈值 MDE，写入 `docs/experiments/opponent-intent-probe.md` 的「预注册」节并冻结。
   验收：预注册无 TODO、含 per-class×bucket 样本数。回滚：改回规划。
2. **extract.py + 泄漏自检**：抽样/预算内回放，抽前缀摘要样本；跑 §3.8 全部自检。
   验收：oracle（决策点手牌）近满分；count-only 不复现主端点；label shuffle ≈ 先验；
   峰值内存/产物在预算内。回滚：修脚本重跑。
3. **P0 身份（健全性）+ P1 意图代理（主必要性门）**：D1-D4 + score-blind + count/score baseline +
   残差化/分层 + bootstrap/permutation。
   验收：产出 accuracy/AUC/CI/曲线，按 §3.6 判定 G-S/G-I/G-F。回滚：无。
4. **P2 下一手（公开候选集为主）+ P4 冻结编码器**：产出 NLL/accuracy vs 先验与计数基线。
   验收：报告完整。回滚：无。
5. **闸门裁决 + 报告**：写 `docs/experiments/opponent-intent-probe.md`（脚本内联 + 环境 hash），
   README §1 加索引。**若 G-I 或 G-D 不成立 → 写负结果并停止**。**这是止损点。**
6. **[需新授权] 阶段 B 实现**：aux 头 + env info 标签 + ppo/train 开关 + 默认路径守护测试。
   验收：全套测试全绿（当前基线见 `plans.md` T17）；默认前向逐位不变。
7. **[需新授权] 阶段 B 池下可识别复核**：在池分布上重跑阶段 A 探针。
8. **[需新授权] 阶段 B 训练**：臂 × 3 seed，≤3 并发。
9. **[需新授权] 阶段 B 评测 + 适应曲线**：per-member 评测、身份×事件数交互。
10. **[需新授权] 阶段 B 报告**：写报告并更新 README 索引；`plans.md` 状态更新需单独授权。

**先出结论的里程碑**：步骤 5（阶段 A 裁决）是止损点。

---

## 7. 风险 / 未决问题 / 需要的 ADR

**风险**
1. **强度/长度混淆**：lvl 分级单调且对局长度随等级变化（本轮实测：每局对手事件 lvl1 26.8、
   lvl2 27.2、lvl3 24.9、lvl4 22.9）。缓解：score-blind + length-only baseline + count-matched
   分层 + 强度残差化（§3.2/§3.3/§3.6）。
2. **样本相关性**：同局多前缀、同 seed 多局 → seed/game 簇 bootstrap + 配对 DID。
3. **探针与 RL 落差**：可识别 ≠ 可被 RL 利用（必要条件非充分）。
4. **aux 头学不到**：改主头为意图代理、身份降为副头；报有效样本数。
5. **池分布与自对弈交互**：`--opponent pool` 不含 live self，结论只在该池上成立；池小且同 lineage
   时 null 不可外推。
6. **计算/内存**：事件张量 `(batch,108,64)` 昂贵；探针只存紧凑摘要。
7. **scikit-learn/scipy 缺失**：探针须 numpy/torch-CPU 自实现；若改用 sklearn 属 pyproject 改动，超契约。

**未决问题（需用户裁决）**
- U1：§3.6 的预注册数值（桶界、MDE、`L*`、残差化方式）是否照此冻结？
- U2：允许新增 `scikit-learn`/`scipy` 依赖吗？
- U3：阶段 B 的对手池如何划分 train/held-out 身份？（建议 train=random+lvl1+lvl2，
  held-out=lvl3+lvl4+一快照）
- U4：阶段的 aux 主头改为意图代理（相对规格原「动作预测」主推）是否可接受？
- U5：阶段 B 是否允许 CUDA 训练与最多 3 进程？
- U6：是否授权把 T15–T17 落成一个提交以建立可回滚基线？

**需要的 ADR（建议，非本阶段写）**
- **ADR-A：训练期 aux 对手建模头**（opt-in、训练期特权、默认 λ=0 逐位不变、checkpoint 布局标记）。
- 阶段 A 不新立 ADR。

---

## 8. 长程

### 8.1 若阶段 B 成功
- 真递归/跨决策 hidden 携带；跨局身份记忆；attention 替换 GRU；与 PFSP/联赛合流。

### 8.2 若失败（可复用的负结果）
- **G-I/G-D null**：写成「在现有 pipeline/预算/分布下，2 家 7鬼523 的公开历史不含可学的对手意图
  （或不可测的 test-time 适应）」，并给出 label shuffle / 截断 / count-only / score-only 对照数字。
- 若意图可识别但阶段 B null：写「可识别但不可被 RL 利用」，指向奖励/信用分配或真递归。
- 复用资产：extract.py 前缀摘要提取器、预注册门与泄漏自检清单。

---

## 附录 A：本轮核实命令与输出

### A.1 `traces/study` 结构（4000 局）

```bash
cd /home/amas/.local/src/7g523
.venv/bin/python - <<'PY'
import os,re,collections
base='traces/study'; rows=collections.Counter()
for lvl in ['lvl1','lvl2','lvl3','lvl4']:
    for fn in os.listdir(os.path.join(base,lvl)):
        m=re.match(r'g\d+__s\d+__seat\d__vs(\w+)\.json',fn); rows[(lvl,m.group(1))]+=1
for k in sorted(rows): print(k, rows[k])
PY
```
输出 10 类各 400，合计 4000；两块 seed 交集为空；同 seed 四档 `initial.hands` 逐位相同
（实测 200/200 seed）。

### A.2 per-prefix 已观测对手事件分布（**修正 §3.5 的关键量**）

```bash
cd /home/amas/.local/src/7g523
.venv/bin/python - <<'PY'
# 在观测者(random)每个决策点记录「此前已观测对手事件数」e
import os,json,statistics,collections
from seven523.trace import state_from_snapshot, rules_from_json
from seven523.game import Game
base='traces/study'; tot=[]; bins=[0,6,12,18,24,10**9]; cnt=[0]*5
for lvl in ['lvl1','lvl2','lvl3','lvl4']:
    for fn in sorted(os.listdir(f'{base}/{lvl}'))[:50]:
        if not fn.endswith('__vsrandom.json'): continue
        d=json.load(open(f'{base}/{lvl}/{fn}'))
        rules=rules_from_json(d['rules']); st=state_from_snapshot(d['initial'],rules); g=Game(rules)
        subj=0 if 'subject' in d['players'][0] else 1; obs=1-subj; e=0
        for s in d['steps']:
            if int(st.current)==obs: tot.append(e)
            if int(s['seat'])==subj: e+=1
            st,_=g.step(st,int(s['action']),s.get('suit'))
for v in tot:
    for i in range(5):
        if bins[i]<=v<bins[i+1]: cnt[i]+=1; break
print('n',len(tot),'min/max/median/mean',min(tot),max(tot),statistics.median(tot),round(statistics.mean(tot),2))
print('B1..B5', cnt)
PY
```
本轮实测：n=5128（200 局）、e 0–35、**中位 13**、mean 13.0；B1..B5 = 1100/1200/1199/1095/534。
旧规划写的「16–36、中位 26」是 **per-game 对手事件总量**（本轮另测中位 26、范围 17–35），
不是分桶对象。每局对手事件按等级：26.8/27.2/24.9/22.9（lvl1–lvl4，80 局/级）。

### A.3 依赖现状

```bash
.venv/bin/python -c "import numpy;print('numpy',numpy.__version__)"   # 2.5.3
.venv/bin/python -c "import torch;print('torch',torch.__version__)"   # 2.14.0+cu132
.venv/bin/python -c "import sklearn"   # ModuleNotFoundError
.venv/bin/python -c "import scipy"     # ModuleNotFoundError
```

### A.4 池成员 ckpt 存在性（四个全 OK）

```bash
for p in runs/t17early2k__1__1790439615/agent.pt runs/t17early8k__1__1790439629/agent.pt \
         runs/t17pool__1__1790439615/snapshots/checkpoint_step65536.pt \
         runs/t17long__1__1790439615/snapshots/checkpoint_step1024000.pt; do
  test -f "$p" && echo "OK $p" || echo "MISSING $p"; done
```

### A.5 代码锚点核实（本轮 grep/read）

- `history.py:153` `encode_events`；`:208-215` blind 分支先置 `mask` 再 `continue`，noisy 写
  `noise[position]`，**两者均保留真实计数**。
- `networks.py`：`Agent` **无 `forward`**；`policy_logits:414`/`get_value:427`/
  `get_action_and_value:439`；`NeuralPolicy.act:768`、`sample` 默认 `False`（`:756`）。
- `env.py`：`_SEGMENTS:186`、`encode_observation:221`、`_publish:387`、`reset:299`、`step:321`。
- `league.py:152-166`：`self/mix/pool` 分支内**无条件**断言 `history_layout(frozen)==history_layout(agent)`。
- `tools/head_to_head.py:1-37`：candidate-vs-candidate，`--left/--right` 两个 Policy spec。
- `policies.py:62-63`：`RandomBot.act` = `self.rng.choice(legal_ids(view.mask))`；
  `policy_seed` 见 `record.py:47`。
- `train.py:620-623` SAME_STEP；`:829` 终局读 `policy.finished_id`；`policies.py:176-190`
  `start_episode` 先 `finished_id=current_id` 再抽新成员。

---

## 附录 B：契约与纪律对照

| 约束 | 本规划落点 |
|---|---|
| 不改 `src/tests/tools/ADR/plans.md/README/CONTEXT/DESIGN` | 本阶段只写两个 docs；阶段 A 脚本只落 `runs/`；报告落 `docs/experiments/` + README §1（Probe 阶段获准） |
| 不 commit/checkout/stash、不用 worktree | 全程遵守；阶段 B 需先授权基线（§5.5） |
| obs v5 不变、ADR-0008/0009 不触发 | 阶段 B aux 头不改 obs 布局（§5.2） |
| 默认路径逐位不变（ADR-0003） | λ=0/无 aux 与现状逐位一致（§5.2） |
| 规则身份（ADR-0013） | 复现基于 trace version=3 + `rules.revision=3`；`rules_id` 在 manifest |
| 唯一投影/差分泄漏（ADR-0002） | 探针特征只取 `game.view`；隐藏字段仅作 y（§2.3/§3.8） |
| 统计纪律（预注册、CI、先验、禁事后挑桶/种子） | §3.6 冻结；单主端点 + Holm；桶界由 per-prefix 分布定 |
| CPU-only、无 h2h 扇出、≤2000 新局、≤3 训练进程 | §5.4 |

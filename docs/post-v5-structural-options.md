# 后 v5 结构性选项决策备忘（memoryless v5 平台之后）

> **状态**：决策备忘（只读汇总 + 选项对比），2026-09-27。本文不改 `src/`、`tests/`、`tools/`、
> `ADR/`、`plans.md`；建议由 plans owner 合并进 [`plans.md`](./plans.md) §3/§6/§8。
>
> **口径标签（硬性）**：revision-3（出空即撬底，`rules_id=2e36dbea44893696`）、obs v5
> （2 家 161 维）、纯 MLP（`arch=shared`、`hidden=128`、单层 trunk＝2 个 Linear）、PPO
> （`num_envs=8 × num_steps=128`、minibatch 256、`update_epochs=4`、lr 2.5e-4 线性退火）。
> 全部比较为 deal-twin 换座 h2h（400 副/deal seed，按副聚类 bootstrap）；行动门槛 =
> **CI（z 与 t）完全排除 0 且点估计 ≥ +10**，`< +5` 止损。跨独立 fit 的绝对分不可比
> （ADR-0013）。
>
> **数字复核**：本文所有合并读数均于 2026-09-27 从 `runs/` 原始 JSON 现场复算（复现命令
> 见 §7）。合并规则：点估计 = 各 seed 均值；`se = max(RMS(bootstrap SE), seed sd)/√k`；
> z-CI 用 1.96，t-CI 用 `t_{0.975,k-1}`（k=3→4.303、k=5→2.776、k=7→2.447）。

## 0. 结论先行

> **2026-09-27 更新（warm-start 续训 + 第二参照确认）**：`t23wswd`（从 `l1_1M` warm start、1M 步、
> AdamW+wd=0.01、cosine、pool>150、batch×2）在 k=3 上拿到 **+11.48**（z-CI [+7.39,+15.58]、
> t-CI [+2.49,+20.47]），三个 seed 单独 CI 均排 0，**首个满足仓库行动门槛的结果**；seed sd 仅 2.16。
> **第二参照确认（追加）**：vs `l1_2M` +9.73（z [+2.20,+17.27]）、vs `t18poolself_s2` +7.25
> （z [+2.42,+12.08]）；3 个训练 seed 全正、新 z-CI 均排 0 且包含原读数 → **增益非参照特异**；
> 但 k=3 下对硬参照 t-CI 跨 0、点 <+10，严格确认需 k≈7。
> **2026-09-29 更新（k=7 confirm，seeds 4–7）**：合并 vs `l1_1M` **+8.20**（t-CI [+4.65,+11.76]）、
> vs `l1_2M` **+7.38**（t-CI [+3.11,+11.64]）、vs `t18poolself_s2` **+3.91**（t-CI [−0.79,+8.62]）→
> **均未达行动门槛**；k=3 的 +11.48 回落到 +8.2（seed sd 3.85–5.09）。旧“首个过门槛”表述作废，
> 改为“小正增益（<+10）、探索性”；“超过当前最强已知模型”未确认；§6.1 朴素对照仍未跑。
> 详见 [`experiments/warmstart-adamw-1m.md`](./experiments/warmstart-adamw-1m.md)（含 §2.1、§2.2 与
> 附录 A：绝对锚 vs h2h 的分工、评分漂移/策略追逐的边界、为何锚分不出 +7~+11）。
> **对本文的影响**：下面的"平台"结论对**从零训练**仍然成立；主线转为 warm-start（归因、补 seed、
> 绝对锚），结构项优先级相应下调。

1. **平台是稳健的，不是某个旋钮没拧对**。500k–2M、random/self/pool/事件表示、宽
   128–512、奖励分差/跳变、batch 1024–4096 —— 全部配方都落在 μ ≈ 188–192（发布
   lvl4 = 185.34 水平），没有任何方向拿到 ≥ +10 的证据。
2. **最后一个活信号（hidden 512）已被 800k × 3 seed 证伪**：200k 单 seed 屏幕
   **+19.2 [1.3, 37.0]** → 800k k=3 **−11.5 [z −23.4, +0.4]**，方向反转。这是本仓库
   "小样本屏幕不可信"的教科书案例：**任何 < 500k × 3 seed 的屏幕都不能作为方向依据**。
3. **剩余问题只能问结构性的**：观测表示（新信息，不是历史重编码）、网络结构、训练范式
   （真递归/算法）、推理期搜索。每一项都需要项目级预算（≥ 15 等效 run）与预注册，先验
   普遍偏低。
4. **先花小钱做 Stage-0 诊断（§4）**，回答"天花板是**信息限制**还是**优化限制**"，再决定
   是否投结构项。
5. 如果暂时没有研究胃口：把预算转去产品线（human-Elo/placement、ladder、评测基建，
   §3-O6）是确定性正收益；当前平台（μ≈190）作为发布资产已经可用。

## 1. 证据地图（全部已关闭方向，现场复核）

| 方向 | 端点（左−右，正 = 左强） | 合并读数 | 判定 | 来源 |
|---|---|---|---|---|
| EVH 事件表示（random） | `event vs event_noise`，k=3 | **−5.02**；z [−22.99, +12.95]；t [−44.47, +34.43] | null（停并发布） | `docs/experiments/event-history-pilot.md` §5 |
| EVH 归属对照（random） | `event vs event_seatblind`，k=3 | +0.83；z [−11.82, +13.47] | null | 同上 §5.2 |
| EVH 事件表示（self-play） | `selfevent vs selfeventnoise`，k=3 | **+2.99**；z [−6.86, +12.85]；t [−18.65, +24.64] | null | `runs/h2h_selfeventvseventnoise_s*.json` |
| self-play fresh seeds（主端点） | `self_sN` vs `l1_1M`，k=7 | **+2.87**；z [−3.37, +9.11]；t [−4.92, +10.66] | null（止损） | `runs/t17w1/h2h_selfVS1M_s*.json` |
| self-play 池化机制 A1 | `poolself vs C0`，k=5 | **−2.71**；z [−8.13, +2.71]；t [−10.39, +4.96] | null（H1 不成立） | `runs/selfpool/h2h_poolselfVSself_s*.json` |
| self-play 固定外部 A2 | `fixed vs C0`，k=5 | **−0.67**；z [−11.12, +9.78]；t [−15.47, +14.13] | null（H1′ 不成立） | `runs/selfpool/h2h_fixedVSself_s*.json` |
| self-play 采样对照 A3 | `selfsamp vs C0`，k=5 | **+0.25**；z [−7.90, +8.40]；t [−11.30, +11.80] | null（H2 不成立） | `runs/selfpool/h2h_selfsampVSself_s*.json` |
| 奖励跳变 J10（λ=1.0） | vs 同 seed `terminal`，k=3 | +0.58；z [−13.06, +14.22]；t [−29.36, +30.52] | null（pilot） | `runs/t19/h2h_jump10_vs_terminal_s*.json` |
| 奖励跳变 J025（λ=0.25） | 同上，k=3 | −2.39；z [−34.28, +29.51] | null | `runs/t19/h2h_jump25_vs_terminal_s*.json` |
| batch 放大（等步数） | 2048/4096 vs 1024 @200k，k=3 | 均值 −36.6 / −33.5 无差；run sd 14.4 → 18.3 **不降** | null | `runs/t20/diagnostic_report.md`、`runs/t21/variance_report.md` |
| batch 放大（等更新数） | 4096@800k vs 1024@200k | 均值 +9.2 vs −36.6（4× 数据，非单因子）；run sd 8.1 ≈ 测量 SE 7.7（hint，未确立） | 不采纳 | 同上 |
| **容量 hidden 512（200k 屏幕）** | `cap512 vs cap128`，k=1 | **+19.2 [1.3, 37.0]** | 假阳性（被下一行证伪） | `runs/t17w1/h2h_cap512VScap128.json` |
| **容量 hidden 512（800k 确认）** | `cap512 vs cap128`，k=3 | **−11.49**；z [−23.38, +0.39]；t [−37.58, +14.60]（3 个 seed 中 2 个单独显著为负） | 容量（宽度）关闭 | `runs/t22/h2h_cap512_vs_cap128_s*.json` |
| 容量 hidden 256（200k 屏幕） | `cap256 vs cap128`，k=1 | −33.56 [−46.33, −20.79] | 关闭 | `runs/t17w1/h2h_cap256VScap128.json` |
| 更长预算 | 1M/1.5M/2M 快照共享 MLE | 187.6–191.6，CI 全重叠 | 平台（关闭） | `docs/experiments/t17-recalibration.md` §1 |
| 观测历史增广（B0/B1） | 500k vs base / ctrl | B0 null、B1 未确认 | 关闭 | ADR-0008 |
| D1-lite 序列臂 | `seq vs seqblind` / `chrono vs sorted` | +8.41 [−4.06, +20.89] / +10.46 [−2.46, +23.38] | null | `experiments/sequence-memory-pilot.md` 等 |
| 双塔 arch | 已测 | 不可分且更差 | 关闭 | `plans.md` §6 |
| 描述性（未进门槛） | `poolself vs l1_1M`，k=5 | +6.34；z [+1.22, +11.47]；t [−0.92, +13.60] | 弱正、< +10，不作行动 | `runs/selfpool/h2h_poolselfVS1M_s*.json` |
| **warm-start 续训（t23wswd）** | 新模型 vs 起点 `l1_1M`，**k=7**（seeds 1–7） | k=3 **+11.48** → **k=7 +8.20**；z [+5.35,+11.05]；t [+4.65,+11.76]；新 seed sd 3.85–5.09 | **未达行动门槛（<+10）；“首个过门槛”作废，改为小正增益（探索性）** | `docs/experiments/warmstart-adamw-1m.md`；`runs/t23/warmstart_confirm_report.md` |
| 同上，第二参照确认 | vs `l1_2M` / vs `t18poolself_s2`，k=3 | **+9.73**（z [+2.20,+17.27]）/ **+7.25**（z [+2.42,+12.08]）；3 seed 全正 | 方向稳健；k=3 下 t-CI 跨 0、点 <+10 | 同上 §2.1 |

**平台一句话**：不同方向、不同预算、不同对手分布，都停在同一个水平；`l1_1M`（1M 步）
与 500k 普通模型不可分辨，宽 512 反而略差，2M 不再涨。

## 2. 为什么只剩下结构性选项

已经穷尽的是"在现有 obs v5 + MLP + PPO 契约内的配方空间"：

- **信息表示**：把公开历史显式喂给模型（序列/事件/座位/墩边界）——全部 null；静态观测增广
  （B0/B1）已否；`unseen`/`last_player` 已在 obs 里。
- **优化/容量**：宽度 256/512、batch/minibatch、奖励形状（分差/跳变）——全部 null 或负。
- **对手分布**：self-play（fresh seed 与池化机制）、固定外部、采样对照——全部 null。
- **预算**：1M/2M 平台。

剩下能动的只有**契约本身**：给模型**新的信息**（不是重新编码已有信息）、改**网络表达
能力**（不是宽度）、改**训练范式**（在线递归/搜索/算法），或者不在训练侧解决的**推理期
搜索**。

## 3. 候选选项

> 每个选项都按同一规格填写：假设、改动面（file:line 锚点）、先验、成本、预注册端点与
> 门槛、风险/混淆。所有结构项实验都要求 **≥3 训练 seed × ≥9 deal seed × 400 副**、
> 匹配预算的对照、z+t 双报；**一项做完再开下一项**。

### O1 — 观测表示 v6：加入"新信息"而不是重编码旧信息

- **假设**：现有 obs v5 的历史段（`unseen`/`last_player`）与事件编码器已经把"已有公开
  信息"榨干；瓶颈是**模型没有的推断特征**，例如由 pass/出牌历史推导的"对手某族无牌 /
  点数上界"的信念特征（belief features），或对手手牌数的动态变化。
- **改动面**：`src/seven523/env.py:223-243`（`_SEGMENTS` 段表 + writers）；`networks.py`
  的 `obs_dim` 自动跟随；ckpt 身份走 `obs_version`（ADR-0008/0009 要求新版本 + 新 ADR，
  且**不恢复已退役的列重映射**，全部从零训练）；评测链（`load_agent` 的版本门禁）同步。
- **先验**：低–中。支持点：B1 的 `unseen` 补的是最大信息缺口但仍 null，说明"信息缺口"
  论证本身不足以推出增益；反对点：事件/序列编码已 null，另加手工特征属于同一信息集合的
  另一种压缩。真正的新信息只有"从历史里**推断**出来的量"（对手分布/威胁度）。
- **成本**：改 env + 测试 + 新 ADR；训练 ≥3 seed × 500k（≥3 等效）；从零、无热启动。
- **判据**：主端点 = v6 vs 同预算 v5 同 seed，≥9 deal seed，k≥3，门槛 +10 / 止损 +5。
- **风险**：信息可能与已有段高度共线（换编码不换信息），大概率重演 B1。

### O2 — 网络结构：深度 / 归一化 / 激活

- **假设**：宽度不是容量瓶颈（512 已阴性），但**深度/条件化**可能是：当前 trunk 只有
  两层 Linear（`src/seven523/networks.py:334-353`）。更深/带 LayerNorm/残差的 trunk、
  或 GELU、或 actor/critic 共享更深的公共干（`arch=shared` vs `towers` 已否）。
- **改动面**：`networks.py` 的 trunk 构造（`__init__` + `_trunk_input`），ckpt 布局变化 →
  从零训练；不触 obs/引擎。
- **先验**：低。理由：宽度 512 在 800k 反而更差，说明该任务的表示容量不是主要限制；
  7鬼523 的状态空间不大（161 维、54 牌），2 层 MLP 未必欠拟合。
- **成本**：小改动 + ≥3 seed × 500k；但深度会增加单步成本（墙钟）。
- **判据**：同 O1。
- **风险**：与宽度实验同构（假阳性风险高），需 k≥3 直接上 500k，不做 200k 屏幕。

### O3 — 真递归（hidden 携带 / BPTT）

- **假设**：无状态重算是严格下界对照组，它 null 说明"事件表示"无增益；真递归只有在
  "表示有增益但无状态重算的读出头不够"时才有增量（`docs/event-history-plan.md` §8.1）。
- **改动面**（§8.1 已列全）：`train.py` rollout 增存初始 hidden + done；`ppo.py`
  `RolloutBatch`（`ppo.py:52-93`）与 shuffle minibatch 重放（`ppo.py:143-164`）；
  self-play 冻结对手按 `(env, seat)` 分 hidden 并按局重置（`league.py:154`）；
  `eval.py`/`duel` 策略生命周期每局重置；吞吐预计 −20~40%。
- **先验**：很低。触发条件（事件臂 ≥+10 或有机制证据的 >+5）**未满足**。
- **成本**：+300–600 LOC + 训练/评测全链改动；≥3 等效起。
- **判据**：同 O1。
- **风险**：改动面最大，收益被自身的无状态下界对照否定过。

### O4 — 推理期搜索（determinization + rollouts）

- **假设**：不动训练，在**对局/评测时**做搜索（不完全信息下的信念采样 + 前瞻），可以
  把**实际对局强度**推高；与"学到的策略上限"是两回事。
- **改动面**：新 `spec` 类型/包装器（`policies.py`、`tools/play_ladder.py`、
  `tools/head_to_head.py`）、CPU 预算（每步 O(rollouts)）；不进 ladder/rating 身份体系，
  除非单独立项。
- **先验**：未知（本仓库从未测过）；对"人类陪练/产品"可能是正收益，对"证明 MLP 上限"
  没有信息量。
- **成本**：工程 + 大量 CPU（h2h 从分钟级变小时级）；与 ladder 的 ckpt 身份冲突需先设计。
- **判据**：与 `l1_1M` 的直接 h2h（同 deal 集），或人类对局胜率；不进现有 confirmatory
  族。

### O5 — 训练范式替换（PPO → 其他）

- **假设**：算法本身（在线策略梯度的方差/样本效率）是瓶颈。
- **改动面**：`train.py`/`ppo.py` 重写；数据、评测、self-play 全部重验。
- **先验**：低。现状更像**信息/表达**限制：不同算法在同一表示下通常只有几个 Elo 的差异，
  而我们找的是 ≥+10。
- **成本**：高；不建议在没有任何 Stage-0 阳性信号前启动。

### O6 — 产品/评测线（非结构，确定收益）

- `T13` human-Elo/placement（D-3 已就绪）、`T9/T10` 评测统计工程、ladder/arena 基建。
- 与平台高度无关：当前 μ≈190 的模型作为对手/基线足够用。
- 建议把一部分预算固定投在这里，避免研究线全 null 时预算空转。

## 4. Stage-0 诊断（先做，便宜，回答"天花板性质"）

> 目的：在花 ≥15 等效的结构预算之前，分清平台是**信息限制**还是**优化限制**。每项都
> 允许用一次性脚本在 `runs/` 下做，**不改生产契约**（不写 obs_version、不发布 ckpt）。

| # | 诊断 | 做法 | 决策读法 | 成本 |
|---|---|---|---|---|
| D-A | **Oracle-obs 探针** | 在训练脚本里把隐藏信息（对手手牌/牌堆）拼在 obs 后（仅诊断、不落 ckpt）训 200k–500k vs 同预算基线 | 若 oracle 也拿不到 ≥+10 → **信息不是瓶颈**，O1 直接放弃；若大涨 → O1 有价值 | ≤2 run |
| D-B | **固定牌局拟合探针** | 把训练 deal 集固定成一个小集合（如 500 副）反复训，看 vs-random 回报能否逼近上限 | 若连固定集都学不上去 → 优化/表达限制（指向 O2/O5）；若轻松过拟合 → 是泛化/探索问题（指向 O1/课程） | ≤2 run |
| D-C | **Critic 天花板探针** | 用更大/独立 critic 测 EV；当前 EV≈0.53–0.65（`ppo.py:254-256`） | 若 EV 不升 → 残差是不可约的游戏随机性（POMDP），critic 改进不降方差；若升 → 值得做 | ≤2 run |
| D-D | **方差 screen** | baseline / `--target-kl 0.03` / `--lr 1e-4` 各 3 seed @200k，**端点 = run sd** | 若某旋钮把 run sd 从 ~13 砍半 → 更新基础配方，之后所有实验受益 | 9 run（~1.8 等效） |

（D-D 与已有 t21 结论互补：t21 证明了 batch 不能降 run sd；D-D 测未试过的稳定器与
warm-start。）

## 5. 推荐顺序与预算纪律

1. **（2026-09-27 更新）主线转为 warm-start**：① 朴素续训对照（同 warm start、原配方，1M ×3 seed，
   判断杠杆是否只是"续训"本身）；② 绝对强度锚（`build_ladder`+`refit_mle` 共享 fit）；
   ③ **补 seed 到 k=7** 把"超过已知最强"写成 confirmatory 结论（vs `l1_2M`/`t18poolself_s2`）。
   （第二参照 h2h 已完成：见证据地图与 `experiments/warmstart-adamw-1m.md` §2.1。）
2. **Stage-0 诊断（§4）并行/备用**：有余量再做；结构项 O1–O5 在 warm-start 归因完成前暂缓。
3. 若 Stage-0 的 D-A 阳性 → O1；若 D-B/O2 指向深度 → O2；O3 只在"表示臂出现机制证据"时
   才解锁（当前未满足）；O4 单独立项、与 ladder 解耦；O5 暂不启动。
4. **结构项一次只开一个**：预注册（端点、对照、deal seed 台账、k、门槛），预算是
   ≥3 seed × ≥500k + 对照，**不做 200k 单 seed 屏幕**（本节第 0.2 条的教训）。
5. **停止规则**：点估计 `< +5` 或 CI 跨 0 → 记入 `plans.md` §6 负结果清单并关闭。
6. **工程卫生**（本轮暴露的）：训练默认存 `snapshots/` 与开 TensorBoard；评测 deal seed
   3→9；单一实验队列 + 全局并发锁（近期出现过 2 个 h2h 并发）；每波结束当场 aggregate
   并写报告；文档/代码尽快提交（自 `ce22681` 起工作区全部未提交）。

## 6. 不建议再做（增补 `plans.md` §6）

- hidden 宽度 256/512（t22 已证伪）、更大 batch（t20/t21）、事件/序列/座位历史编码
  （EVH、D1-lite）、self-play 变体（fresh/池化/固定外部/采样）、奖励跳变
  （`terminal_win` λ=0.25/1.0）、更长预算（1M–2M 平台）。
- `saturate`/`--reward-cap`（reward-alignment plan §3）**已实现**（`env.py`/`train.py`；
  执行报告 [`experiments/reward-alignment-saturate.md`](./experiments/reward-alignment-saturate.md)
  主臂显著负、已止损关闭）；若要再试，建议主臂
  τ=0.7 而非 0.2（本游戏撬底可整局翻盘，"领先 20 分"不构成安全边界），并且它同样受
  "< +5 止损"约束。

## 7. 复现与产物

- 合并读数复算（现场命令示例）：
  ```bash
  cd /home/amas/.local/src/7g523/runs
  python3 - <<'EOF'
  import json,glob,math,statistics as st
  paths=sorted(glob.glob("t22/h2h_cap512_vs_cap128_s*.json"))
  vals=[];impl=[]
  for p in paths:
      c=json.load(open(p))["combined"]; e=c["elo_diff"]
      vals.append(e["mean"]); impl.append((e["ci"][1]-e["ci"][0])/(2*1.96))
  k=len(vals); boot=math.sqrt(sum(x*x for x in impl)/k); sd=st.stdev(vals)
  se=max(boot,sd)/math.sqrt(k); m=sum(vals)/k
  print(m, 1.96*se, {3:4.303,5:2.776,7:2.447}.get(k,2.571)*se)
  EOF
  ```
- 关键产物：`runs/t17w1/`（F1b/F0/Block C h2h）、`runs/selfpool/`（A1–A3 + eval 日志）、
  `runs/t19/`（奖励跳变 pilot）、`runs/t20/`、`runs/t21/`（batch 诊断）、`runs/t22/`
  （容量确认）、`runs/h2h_selfeventvseventnoise_s*.json`（EVH self-play）。
- 训练恢复命令模板见各 run 的 `args.json`；评测统一
  `tools/head_to_head.py --seeds <9 个> --pairs 400 --bootstrap 4000 --device cpu --workers 4`。

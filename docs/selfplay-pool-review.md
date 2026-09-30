# selfplay-pool 计划红队 findings 处置（Repair 评审）

> **状态：历史口径（Repair 评审记录，对应 revision-2 计划）。执行结果见 [`experiments/selfplay-pool-diagnostics.md`](./experiments/selfplay-pool-diagnostics.md) 与 [`experiments/selfplay-pool-wave1.md`](./experiments/selfplay-pool-wave1.md)。**
>
> **角色**：Repair/诊断 agent。本阶段只读代码、只写
> `docs/selfplay-pool-plan.md`（revision-2）与本文；未跑训练/h2h；未改 src/tests/tools/ADR 等。
> 输入：revision-1 计划 + 红队 40 条 findings（16 stats-eval + 13 selfplay-dynamics + 11 engineering-cli）。
>
> **处置统计**：blocker 6/6 已处置；major 18/18 已处置（其中 2 条以"降级为探索性 + 明写混淆"
> 方式解决，非新增 run）；minor 16/16 已处置。**无 rejected**；其中 stats-eval-06、
> engineering-cli-4 为 **accepted(modified)**（见下）。
>
> 每条都给出**代码/数据证据**；"Plan §X" 指修复后 plan 的对应位置。

---

## 1. Blocker 逐条（必须解决或升级）

### stats-eval-01 — D1 确认复用发现集 deal seed 20（selection-then-confirmation）
**处置：accepted（已改预注册）。**
- 证据（代码）：`src/seven523/ladder.py:251-252` 在 `plan_games` 里 `deal_seed=master.randrange(1<<32)`
  且 `for index in range(cross//2)` 每 index 一个 deal、对所有候选对复用；`src/seven523/duel.py:50-53`
  的 `plan_duel_schedule` 同样从 `random.Random(seed)` 抽。两者**同 base seed 时随机流前缀相同**。
- 证据（只读实测，复刻 revision-1）：`arena(seed=20, gpa=100, cross=100)` 的 50 个 cross deal seed
  与 `duel(seed=20, pairs=400)` 的 400 个 deal seed **重叠 50/50**；与 duel(21)/duel(22) 重叠 0。
- 修复：Plan §3.6 引入 deal-seed 台账；发现流与确认流**base seed 不相交**
  （D1 base `40`；确认 `140–148`；E1/E2/E3 `100–108/110–118/120–128`）；"fresh" 重定义为
  "与任何发现/选择集及本计划其他端点零交集"。

### stats-eval-02 — 13 节点联赛 286 个三元组无 FWER，判据 (i) 近似恒真
**处置：accepted（已改判据）。**
- 证据：`arena.py:241-275` 只输出**每无序对**一行，无三元组/循环统计量；C(13,3)=286。
- 修复：Plan §3 D1 **整体降级为探索性**，不再用 (i) 作 confirmatory；报告必须写"扫描 N 个三元组、
  无 FWER 控制"。confirmatory 循环只在收到单三元组**升级条件**（三条边点差各 ≥ D1 MDE ≈34 Elo）
  时，用**与发现流零交集**的 deal seed 做单三元组确认（split-sample，选择后确认有效控制 FPR）。

### stats-eval-03 / selfplay-dynamics-11 — D0 "95% CI 重叠"当通过条件，几近不可失败
**处置：accepted（已改门）。**
- 证据：`docs/experiments/t17-recalibration.md` §1 MLE CI 宽 ±13–16（`l1_1M [177.90,204.64]`、
  `p1_499k [174.21,207.59]`）；revision-1 的降级阈值（低 20+）与通过阈值（重叠）互相矛盾。
- 修复：Plan §3 D0 改为**非劣检验**：`Δ_budget(m)=μ(m)−μ(p1_499k)` 的差 CI 下界 > −5、
  `Δ_anchor(m)=μ(m)−μ(l1_1M)` 差 CI 下界 > −10，两成员都过才算过门；差额 CI 由 deal-clustered
  bootstrap 直接给出（`runs/selfpool/d0_diff.py`，复用 `seven523.mle.fit_mle`）。两个比较对象各
  一个非劣阈值（`Δ_budget>−5` 对 `p1_499k`、`Δ_anchor>−10` 对 `l1_1M`，**合取**），见第二轮 repair §6.6。

### selfplay-dynamics-01 — greedy(argmax) 对手混淆 self-play 与确定性；H2/D3/D5 因果失效
**处置：accepted（结构性修复 + 新增对照臂）。**
- 证据：`networks.py:758 sample: bool=False`、`:826/:834` argmax 分支；`league.py:63
  self_play_sample=False`、`:160 sample=config.self_play_sample`；`policies.py:286` 构造 `ckpt:`
  成员**不传 sample**；`train.py:362 --self-play-sample default False`；
  `runs/t17w1self__2__1790492897/args.json` `self_play_sample: false`。`RandomBot` 才随机
  （`policies.py:56`）。
- 修复：Plan §1.2 明写混淆；**主对比（A1/A2/A3 vs `C0`）对手全为 argmax，内部有效**；
  H2/熵（D3/D5）**全部降级探索性**；新增 A3 = `{1@self} --self-play-sample True`（唯一直接
  对照，现有 CLI，无代码改动）。P2 式"末熵回到 random 带"的错误预测作废（§0.2、§2.2）。

### selfplay-dynamics-02 — E1 主端点未与对照隔离（vs l1_1M 只 ≈ +7 over C0）
**处置：accepted（改主端点）。**
- 证据：Wave-1 F1b `C0 vs l1_1M = +2.87`（`v5-optimization-wave1.md` §4.1，
  z-CI [−3.37,+9.11]）；revision-1 却把 `P vs l1_1M` 当主门。
- 修复：Plan §5.1 主端点 = **每臂 vs 同 seed `C0`** 的配对差（E1/E2/E3，k=5、9 deal、400 副）；
  `vs l1_1M` 降为描述性，且标注 500k vs 1M 预算不匹配（见 stats-eval-08）。

### selfplay-dynamics-03 — 无中立异族 held-out 对手族，E1 无法区分通用强度与同族特化
**处置：accepted（降级机制声明，如实写死）。**
- 证据：`grep 'class GreedyBot' src/` 为空；`train.py:273 --opponent choices=['random','self','mix','pool']`；
  `lvl1-4`/`l1_*`/`p1-3` 全部 `opponent=random`（`t17-recalibration.md` §0、各 `args.json`）。
- 修复：Plan §0.6/§1.3-5/§6.4-R5 明写"机制结论只对 random-trained MLP 家族内的对手身份/多样性成立"，
  **禁止**越界写"self-play 本身退化"；把"中立异族对手"列为 §7.1 需停下报告的前置。

---

## 2. Major findings（含需详述的 minor stats-eval-16）

### stats-eval-04 — 判据 (iii) 把逐对 transitivity 行当循环检验
**accepted。** 证据：`arena.py:241-275` 每行是一个**无序对** `{a,b}`，
`se=sqrt(p(1-p)/(games//2))`，docstring `arena.py:250` 明写 "diagnostic, not a calibrated test"。
修复：Plan §3 D1 删除 `|z|>3` 门；`pair_diagnostics` 只作探索性残差表。

### stats-eval-05 / engineering-cli-10 / selfplay-dynamics-13 — E1/E2/E2b/D1 共享 deal seed 20-28
**accepted。** 证据：revision-1 §5.1 E1 `20–28`、E2 `20–24`、D1 arena `--seed 20` 子集。
修复：Plan §3.6 按用途切分 deal seed（100–138 + D1 `40`），互不相交；seed 20 不再出现。

### stats-eval-06 / selfplay-dynamics-10 — D3 `|ρ|≥0.7 且 p≤0.05` 在 n=7 不可达
**accepted(modified)。** 证据（只读精算，7!=5040 精确置换）：`P(|ρ|≥0.7)=0.0881`；
可达阈值实为 `|ρ|≥0.78`（精确 p=0.048），并非 finding 写的 0.786。
修复：Plan §3 D3 **取消 ρ 阈值**，只用精确置换 `p≤0.05` 判定（更稳，且避免误引 0.786）。

### stats-eval-07 / selfplay-dynamics-04 — D1 每对 100 局、50 副牌共享 → 边过噪且相关
**accepted。** 证据：`--cross 100` → 每对 100 局 / 50 副（`ladder.py:251-252`）；
`se≈sqrt(0.25/100)=0.05`；`pair_diagnostics` `deals=games//2`（`arena.py:262`）忽略跨对相关。
修复：Plan §3 D1 `--cross 400`（每对 400 局，SE≈0.025），并**预注册功效**：D1 MDE≈34 Elo，
对 self 族 5–15 Elo 真实差**零功效**，不得写"无循环"。

### stats-eval-08 — E1 把"对手分布"与 2× 训练步数（500k vs 1M）绑定
**accepted。** 证据：revision-1 E1 只比较 500k 臂 vs `l1_1M`（`t17long` `total_timesteps=2000000`）。
修复：Plan §5.1 主端点改为同预算配对（vs 同 seed `C0` 500k）；`vs l1_1M` 描述性并标注预算不匹配。

### stats-eval-09 — D4 placebo `MA_X` 含同 run 孪生，对照失效
**accepted。** 证据：`l1_1M`/`l1_2M` 同属 `t17long`（plan §1.1）。修复：Plan §3 D4 `MA_X`
只用独立 run `{p1,p2,p3}`；该分析整体探索性。

### stats-eval-10 — D1 含 `self_s1`（Wave-1 事后选中赢家）
**accepted。** 证据：`self_s1` = 旧 T17 stop-and-choose 赢家（`v5-optimization-plan.md:119`）。
修复：Plan §1.1/§2 confirmatory 族 `C0=t17w1self__2..6`，`self_s1` 只进探索性分析（本版实际未用）。

### stats-eval-11 — D5 random 基线混入 hidden=256/512 与 2M 长训
**accepted。** 证据：`t17w1cap256/512` hidden 256/512（`v5-optimization-wave1.md` §3），
`t17long` 2M。修复：Plan §3 D5 主对比只用 hidden=128/500k 的 `t17pool p1/p2/p3`；cap\*/long 单列。

### stats-eval-12 / selfplay-dynamics-06 — P1 池仍含 25% self/random，E2 归因不干净且功效未预注册
**accepted。** 证据：revision-1 §4.1 P1=`{self,p2,p3,random}` 各 1/4；E2 用 5 deal。
修复：Plan §4.1 池**去掉 random**；重定义为"配对 vs `C0`"三臂；E2 功效放进 §5.2 统一预注册；
A1/A2 权重重归一化明写为描述性（见 engineering-cli-3）。

### stats-eval-13 — 功效表引用错位（"Wave-1 §1.4"）
**accepted。** 证据：`v5-optimization-wave1.md` §1 无该表；数字在 `v5-optimization-plan.md`
§1.4（lines 152–175）。修复：Plan §5.2 改引 `v5-optimization-plan.md §1.4`。

### stats-eval-14 — E0 绝对锚门偏向前者（random 是池成员）
**accepted。** 证据：revision-1 P1/P2 含 `1@random`。修复：Plan §4.1 池**无 random** → F0 干净；
E0 仍单 seed → **移出 action table**，仅描述性 sanity（Plan §5.1 E0）。

### stats-eval-15 / engineering-cli-6 — D0 `--study /tmp/...` 越界
**accepted。** 证据：`tools/build_ladder.py:177-179` 读 `study/manifest.json`（`--no-traces` 不写，
但指到 /tmp 仍越界）。修复：Plan §3 D0、附 A 改 `--study runs/selfpool/d0_study`。

### stats-eval-16 — D2/D4 共用 D1 逐局且含锚局，非独立（红队标 minor）
**accepted。** 证据：revision-1 §3 D2/D4 输入 D1；D1 含 random 锚局。修复：Plan §3 D2
**剔除 `kind=="anchor"`**；D2 与 D4 **合并为一条证据**并注明非独立。

### selfplay-dynamics-07 — T5 null 的 4 个竞争风险被静默吸收
**accepted。** 证据：`opponent-distribution-500k.md` §7 列 4 风险。修复：Plan §2.3 列
AH1–AH6 显式备择假设与证伪判据（含 AH5"固定对手池根本无效"、AH6"random 稀释"）。

### selfplay-dynamics-08 — 缺最直接证伪臂：单一固定外部确定性对手
**accepted。** 修复：Plan §4.1 **A2 `fixed` = `{1@ckpt:p2}`**（无 self），预注册主要对照。
（revision-1 无此臂；Wave-5 `w5_mix_samp` 是 warm-start 且混 Greedy，不回答 cold-start v5。）

### selfplay-dynamics-05 — 熵代理是模板+花色之和，花色头占主导
**accepted。** 证据：`wave5-500k-report.md:102-114`：H_template 0.13–0.26 nats，
H_suit 1.16–1.35 nats（~85–90%），动作由模板头决定。修复：Plan §3 D5 主读数改**模板头 masked
entropy**（`runs/selfpool/probe_entropy.py`），**禁止**用 raw `metrics.csv entropy`；无法实现则取消 D5。

### engineering-cli-1 — `build_ladder --games-out` 追加式，rerun 静默重复
**accepted。** 证据：`ladder.py:390` `results_path.open("a")`、`:350` 合并 `open("ab")`，
只 unlink temp shard（`:498`）；`tools/build_ladder.py:106-113` 标注 append；
`tools/refit_mle.py:load_games` 读**每行** → 重复计数、CI 虚窄。
修复：Plan §3 D0/附 A `rm -f` + `test $(wc -l) -eq 4000` 断言。

### engineering-cli-2 — P1/P2 训练是普通串行 for 循环，与"≤3 并发"矛盾
**accepted。** 证据：revision-1 §4.2 两段 `for ...; do ...; done` 无 `&`/`wait`/`xargs -P`。
修复：Plan §4.2 改为 `train_jobs.txt` + `xargs -P 3` 显式有界调度器 + `nice -n 5`。

### engineering-cli-3 — P1−P2 同时改"self 成员存在"与外部权重重归一化
**accepted。** 证据：`EpisodeMixturePolicy` 按权重抽样（`policies.py:177-195`）：
P1 外部各 1/4 vs P2 各 1/3。修复：Plan §4.1 **A1−A2 明写为描述性**（非孤立）；孤立
"移动 self 成员"（refresh 10 vs 0 同池）列 Phase 2 第四臂（预算不足）。

### engineering-cli-4 — D2/D4 从未让模型对**自己镜像**下棋
**accepted(modified)。** 证据：`ladder.py:199-260` 只 pair 不同 id；`duel.py` 要求
`left.id != right.id`。修复：Plan §3 D1 加 `self_s2`/`self_s2_mirror`（同 ckpt 两 id）作
**残差校准**；同时如实说明：同参 argmax 镜像按对称性期望 ≈0.5，故它校准统计量而非直接证
"镜像过拟合"；跨 seed self-vs-self 仍是主操作化。

### engineering-cli-5 — §6.1 step 8 `--pair` 用斜杠列表
**accepted。** 证据：`runs/t17w1/aggregate.py:56-60` `f"h2h_{args.pair}_{s}.json"` 单值。
修复：Plan §6.1 step 8 改为 `for pair in ...; do aggregate --pair "$pair"; done`。

### engineering-cli-7 — `build_ladder --games-out` 是 FILE，`arena --games-out` 是 DIR
**accepted。** 证据：`tools/build_ladder.py:106-113` metavar PATH；`tools/arena.py:118-124`
metavar DIR（`shard_NNNNN.jsonl`）。修复：Plan §3 D2 明确 glob `arena_core_games/shard_*.jsonl`，
D0 用单文件 `d0_games.jsonl`。

### engineering-cli-8 — D1 循环搜索无多重比较控制
**accepted。** 与 stats-eval-02 同一修复（D1 探索性 + 升级条件 + split-sample）。

### engineering-cli-9 — 评测墙钟估计高 4–6×
**accepted。** 证据：`event-history-pilot.md:54`：h2h 3 deal ×400 = 27–35 s。修复：Plan §6.1 step 7
改用吞吐锚（9 deal ≈ 80–110 s/条），**不**沿用 6–10 min。

### engineering-cli-11 — D0 门跨两个估计量未说哪个决定
**accepted。** 证据：`build_ladder` 打 OpenSkill PL μ±σ；只有 `refit_mle` 出 CI；
`t17-recalibration.md` §1 显示两者次序不一致。修复：Plan §3 D0 明确**只用 `refit_mle` probit-MLE**，
PL 表仅信息性。

---

## 3. Minor findings

| id | 处置 | 证据 / 修复位置 |
|---|---|---|
| stats-eval-13 | accepted | 见上；Plan §5.2 改引 `v5-optimization-plan.md §1.4` |
| stats-eval-14 | accepted | 池去 random；E0 出 action table；Plan §5.1 |
| stats-eval-15 | accepted | `--study runs/selfpool/d0_study`；Plan §3 D0 |
| engineering-cli-5 | accepted | 分开 `--pair`；Plan §6.1 |
| engineering-cli-6 | accepted | 同 stats-eval-15 |
| engineering-cli-7 | accepted | shard glob 明写；Plan §3 D2 |
| engineering-cli-8 | accepted | 同 stats-eval-02 |
| engineering-cli-9 | accepted | 吞吐锚；Plan §6.1 |
| engineering-cli-10 | accepted | deal seed 分区；Plan §3.6 |
| engineering-cli-11 | accepted | 只用 refit_mle；Plan §3 D0 |
| selfplay-dynamics-09 | accepted | 匹配 step 熵：self@500k 1.33–1.55 vs t17pool@500k 1.648/1.671/1.755，`t17long@2M=1.603`；1.80–1.89 是 200k cap 臂；Plan §3 D5 |
| selfplay-dynamics-10 | accepted | 同 stats-eval-06；Plan §3 D3 |
| selfplay-dynamics-12 | accepted | `X={p1,p2,p3,l1_1M}` 冻结，去 `l1_2M` 孪生、按 run 聚簇；Plan §2 |
| selfplay-dynamics-13 | accepted | 主端点同预算；D1 seed 20 不再用于 h2h；Plan §5.1/§3.6 |

（stats-eval-04/05/06/07/08/09/10/11/12 见 §2；engine-cli-1..4 见 §2。）

---

## 4. 改动位置索引（revision-1 → revision-2）

| 主题 | revision-2 位置 |
|---|---|
| 主端点改为同 seed 配对 vs `C0` | Plan §0.2、§5.1、附 B |
| 三臂重定义（去 random；A2 单固定；A3 sample） | Plan §4.1 |
| 15-run 有界调度器 | Plan §4.2 |
| D0 非劣门 + 只用 probit-MLE + 行数断言 + runs/ study | Plan §3 D0 |
| D1 探索性 + cross 400 + 镜像校准 + 升级条件 | Plan §3 D1 |
| deal-seed 台账与 fresh 定义 | Plan §3.6 |
| T5 备择假设 AH1–AH6 | Plan §2.3 |
| greedy 混淆与 H2 降级 | Plan §1.2、§2.2、§3 D5 |
| 中立异族对手降级声明 | Plan §0.6、§1.3-5、§6.4-R5 |
| 熵改模板头 | Plan §3 D5 |
| D2/D4 去锚、合并为一条证据 | Plan §3 D2/D4 |
| 功效/引用修正 | Plan §5.2 |

---

## 5. 残余风险与需要用户决策的项

1. **k=5 功效**：15-run 硬预算下主端点 k=5；若训练 seed sd 仍 ~8.4，t 半宽 ≈10.4，
   真实效应 ≈+10 大概率"未判定"。已预注册 binding-term 升级（k=7，复用 `self_s7/s8`）。
   如需 k=7 全 confirmatory → 需放宽预算到 ≥21 run，属**用户决策项**。
2. **无中立异族对手**：机制结论只能限 random-trained MLP 家族内。要越界需新增中立 bot
   （可能需代码，属"停下报告"项）。
3. **"移动 self 成员"孤立对比**（refresh 10 vs 0）与 **PFSP 对比** 均超 15-run 预算，
   已明确列 Phase 2；不静默吸收。
4. **H-cycle 确认**：升级条件（三条边各 ≥34 Elo）在 self 族 5–15 Elo 散度下很可能不触发，
   届时如实写"H-cycle 不可判"，不写"无循环"。

**结论：verify = clean（blocker/major 全部有代码或数据依据的处置；无 rejected）。
允许进入 Execute，但须先执行 D0 门，且执行者按 Plan §6.1 的有界调度器与 §3.6 deal 台账操作。
若执行中发现需要新 CLI 开关或改 src/tests/tools，按纪律停下报告，不即兴改码。**

---

## 6. 第二轮 repair 记录（独立 verify verdict=issues）

> 独立 verify 对 revision-2 给出 `issues`：40 条红队 finding 均已在 plan body 有处置，但
> revision-2 自身引入了 6 条残余问题（1 blocker、1 major、4 minor）。本节逐条记录
> **问题 → 改动位置 → 证据**。只改文本，不动结构、不新增范围；仍未改 src/tests/tools。

### 6.1 [blocker] D5 预注册判据不可评估（self 无 snapshots）

- **问题**：revision-2 §3 D5 判据为 “self 族在**任一匹配 step ≤250k** 处模板头熵均值低于
  random 族 ≥0.2 且自举 CI 排 0”。但 self 训练 run **没有 `snapshots/`**（plan §1.3-1 自认），
  random 臂 `t17pool` 才有 `16k..499k` 快照 → self 族在任何 step ≤250k **无模型**，判据不可
  评估；§6.2 的“D5 有可复现产物与预注册读数”验收无法达成。
- **改动位置**：Plan §3 D5（整节重写为“终点确定化对比”）。
- **证据**：`ls runs/t17w1self__*/` 只有 `agent.pt/args.json/checkpoint.pt/metrics.csv`；
  `ls runs/t17pool__1__1790439615/snapshots/` 有 `16k..499k`。判据改为**唯一匹配 step = 500k**
  上的 self 族（`S7`，n=7）vs random 族（`p1/p2/p3`，n=3）模板头熵差 + deal-clustered 自举 CI；
  跨时程曲线只作背景并显式标注“跨时程、非主读数”；另因 `H_template` 实测仅 `0.13–0.26` nats
  （`wave5-500k-report.md:102-114`），取消近乎不可达的绝对 `≥0.2` 阈值，只报差值 + CI。

### 6.2 [major] `C0` 两义（2..6 vs 2..8）

- **问题**：Plan §1.1 写 `C0 = t17w1self__2..8`（7 seed），而 §0.4/§2/§5.1/附 B 写
  `C0 = self_s2..self_s6`（k=5）；§3 D3 又在 `C0` 标签下用 2..8（n=7）。执行者读 §1.1 可能误用 k=7。
- **改动位置**：Plan §1.1（新增冻结命名段）、§2（记号）、§3 D3（输入）。
- **证据**：operative 主端点 §5.1 的 `s∈{2..6}` 不变；新增 `S7 = {self_s2..self_s8}`（n=7）
  作为**探索性**集合，供 D1/D3/D5 使用；`C0` 全篇固定为 k=5 confirmatory 族。

### 6.3 [minor] 附 B 悬空引用 `D6`

- **问题**：附 B 第 6 条写 “诊断 D0–D5；D6 仅设计”，但全篇无 D6 定义（`grep D6` 仅此一处）。
- **改动位置**：Plan 附 B 第 6 条。
- **证据**：删除 “；D6 仅设计”，现为 “诊断 D0（门，非劣）–D5（探索）。”

### 6.4 [minor] E_anchor 作用域（3 臂 vs 1 臂）

- **问题**：§5.1 把 E_anchor 泛写为 “`A_s vs l1_1M`”（暗示三臂），而 §6.1 step 7 与附 A 只对
  A1/poolself 跑 5 条，描述性评测预算按 20 条（15+5）计。
- **改动位置**：Plan §5.1 E_anchor、§6.1 step 7、附 A 描述性注释。
- **证据**：现明确 E_anchor **只覆盖 A1**（s2..s6，5 条 h2h）；A2/A3 的 vs-`l1_1M` 若要做属额外
  评测、另计数，预算 20 条不变。

### 6.5 [minor] D0 `--seed 0` 与台账冲突

- **问题**：§3.6 台账把 `0–8` 列为已占用，D0 却用 `--seed 0`。
- **改动位置**：Plan §3 D0 命令、§3.6 D0 行、附 A D0 命令。
- **证据**：D0 改用 `--seed 39`（不在 `0–8`/`10–18`/`29–37`，也不与 D1 的 40、E 的 100–138、
  D1 确认 140–148 相交）；D0 仍是独立 ladder fit、不进任何 h2h 推断。

### 6.6 [minor] D0 双阈值措辞

- **问题**：finding stats-eval-03 建议“单一非劣阈值”，revision-2 用了两个
  （`Δ_budget>−5` 对 `p1_499k`、`Δ_anchor>−10` 对 `l1_1M`）。
- **改动位置**：Plan §3 D0、本文 §1 stats-eval-03 条。
- **证据**：在 §3 D0 增写“阈值口径说明”：两阈值**故意**各自对应不同比较对象（同预算代理 /
  跨预算标尺），判定为**合取**、完全可判定；这偏离“单阈值”措辞但保留非劣性（非 revision-1 的
  “CI 重叠”），不视为缺陷。本文 §1 末句“统一单一阈值”已改为两阈值的合取描述。

### 6.7 结论

- 6 条均已文本级修复；confirmatory 机器（D0 门、E1/E2/E3 主端点、deal-seed 台账、15-run
  调度器）未改动。机制线（D1–D5）仍为探索性。
- 仍**未**改 src/tests/tools；未跑训练/h2h。`verdict=issues` 的 blocker 已消除，D5 判据现可评估
  （唯一匹配 step=500k）。请求复审。

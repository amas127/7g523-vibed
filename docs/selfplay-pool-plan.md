# self-play 退化机制 与 池化对手 预注册实验计划（revision-2，第二轮 repair 后）

> **状态：已实施（Phase 1，2026-09-27）——A1/A2/A3 三臂全部止损、D0 未过门，本计划以「弱成员池重估」收束；Phase 2 未启动。执行结果见 [`experiments/selfplay-pool-wave1.md`](./experiments/selfplay-pool-wave1.md) 与 [`experiments/selfplay-pool-diagnostics.md`](./experiments/selfplay-pool-diagnostics.md)；以下为 revision-2 预注册文本（历史）。**
>
> **角色**：Repair agent。本阶段**零训练、零 h2h**；只读代码/只读命令，只写
> `docs/selfplay-pool-plan.md` 与 `docs/selfplay-pool-review.md`。
> 未改 `src/`、`tests/`、`tools/`、`ADR/`、`plans.md`、`docs/experiments/README.md`、`CONTEXT.md`、
> `DESIGN.md`；未 commit/checkout/stash；未跑训练或 h2h。
>
> **本版是对 revision-1（红队 40 条 findings）的修复**：处置逐条见
> [`selfplay-pool-review.md`](./selfplay-pool-review.md)。红队 blocker 全部处置：
> stats-eval-01/02/03、selfplay-dynamics-01/02/03 均在本版有明确设计改变。
>
> **第二轮 repair（本版）**：独立 verify（verdict=issues）指出 6 条残余问题——D5 step 判据
> 不可评估（blocker）、`C0` 定义不一致（major）、D6 悬空引用、E_anchor 作用域、D0 deal seed
> 与台账冲突、D0 双阈值措辞——均已修复，逐条见
> [`selfplay-pool-review.md`](./selfplay-pool-review.md) §6。本版不重写结构、不引入新范围。
>
> **口径标签（硬性）**：全部设计与引用数字属 revision-3（出空即撬底，`rules_id=2e36dbea44893696`）、
> obs v5（2 家 161 维）、纯 MLP（`arch=shared`、`hidden=128`、单层 trunk）。跨 fit 绝对 Elo 不可比。
> 统一评测口径见 [`experiments/README.md` §3](./experiments/README.md) 与
> [`v5-optimization-plan.md`](./v5-optimization-plan.md) §6。
>
> **上游状态**：[`v5-optimization-wave1.md`](./experiments/v5-optimization-wave1.md) 已把 self-play 线
> 在 k=7 fresh 训练 seed 上判为**未判定（点 ≤ +5 → 止损）**。本计划是**用户点名的新线**
> （池化对手 / self-play 退化机制），属**新的预注册**，`k=5` very fresh 训练 seed、fresh deal seed 从头设计。

---

## 0. 结论先行（修复后）

1. **机制问题（"self-play 退化是否因为只和自己打 / 是否存在策略追逐/非传递"）在本批资产下只能做
   **描述性/筛查**，不能单独定论。** 关键硬约束：Wave-1 的 self 训练 run **没有 `snapshots/`**
   （`ls runs/t17w1self__*/` 只有 `agent.pt/args.json/checkpoint.pt/metrics.csv`），无法观测
   同一次训练内部的策略轨迹；离线诊断只能基于**最终模型群体**。红队已证明 revision-1 的 D1
   "高门槛"在现有样本量下**零功效**（见 §3 D1），故本版把 D1–D5 **明确降级为探索性**，
   不再承载 H1/H3 判定。
2. **决定性实验 = 池化对手训练**，三条 500k 冷启动臂，全部与**同 seed 的纯 self 对照 `C0` 配对**：
   - `A1 poolself`：`{1@self(刷新), 1@p2, 1@p3}`（**不含 random**）= 用户点名的"从池子抽对手"。
   - `A2 fixed`：`{1@p2}`（单一固定外部对手，无 self）= 证伪"只和自己打"的对照臂。
   - `A3 selfsamp`：`{1@self}` + `--self-play-sample True` = 对手 argmax 混淆的对照臂。
   预注册主端点 = **每个臂 vs 同 seed 纯 self** 的配对差（不是 vs 1M 锚），`k=5`。
3. **T5 的 null 前提按红队要求**：T5 有 4 个竞争解释（弱成员 / 单 seed / 父模型天花板 / PFSP 稀释），
   本版把它们全部写成**显式备择假设与证伪判据**（§2.3），不把"成员全弱"当成必然原因。
   本版池臂**去掉 random 成员**（避免稀释、并让 F0 绝对锚干净），成员取 `p2/p3_500k`。
4. **评测污染边界**：主对照锚是**同 seed 的 `C0`**（`t17w1self__2..6`，已在 Wave-1 冻结、无 post-hoc
   选择）；训练成员与 `C0`、与 `l1_1M` 的 run/seed 双避。`l1_1M`/`p1_499k` 只作**描述性**参照。
5. **预算**：15 × 500k = **15 等效 = 硬上限**（3 臂 × 5 seed）。PFSP / snapshot 联赛 / 更多臂列
   Phase 2，触发才申请。诊断只复用现有 ckpt。
6. **诚实前提（红队 selfplay-dynamics-03）**：现有全部强资产都是 **random-trained MLP**
   （`GreedyBot` 已在代码中移除，`grep 'class GreedyBot' src/` 为空），没有中立的异族 held-out
   对手。因此本计划的机制结论**只对"random-trained MLP 家族内的对手身份/多样性"成立**，
   **不得**越界写成"self-play 本身退化"。

---

## 1. 现状复核（只读，已现场核对）

### 1.1 可用最终模型群体（`torch.load` 已核对 obs/obs_dim/step）

| id | 路径 | obs | obs_dim | global_step | seed | 训练对手 |
|---|---|---|---|---|---|---|
| `self_s1` | `runs/t17self__1__1790439615/agent.pt` | 5 | 161 | 499712 | 1 | self(refresh 10) |
| `self_s2..s8` | `runs/t17w1self__{2..8}__179049{2897,3881,4847}/agent.pt` | 5 | 161 | 499712 | 2–8 | self(refresh 10) |
| `p1_499k` | `runs/t17pool__1__1790439615/agent.pt` | 5 | 161 | 499712 | 1 | random |
| `p2_500k` | `runs/t17pool__2__1790439615/agent.pt` | 5 | 161 | 499712 | 2 | random |
| `p3_500k` | `runs/t17pool__3__1790439615/agent.pt` | 5 | 161 | 499712 | 3 | random |
| `l1_1M` | `runs/t17long__1__1790439615/snapshots/checkpoint_step1024000.pt` | 5 | 161 | 1024000 | 1 | random(2M run) |
| `l1_2M` | `runs/t17long__1__1790439615/snapshots/checkpoint_step1999872.pt` | 5 | 161 | 1999872 | 1 | random(2M run) |

- **命名（冻结，消除 "C0" 歧义 — 第二轮 repair）**：`C0 = {self_s2..self_s6}`（k=5，confirmatory
  对照族）；`S7 = {self_s2..self_s8}`（n=7，**仅探索性**，D1/D3/D5 用）。
- `self_s1` = 旧 T17 stop-and-choose 的单 seed 赢家（Wave-1 事后 +16.62）→ **本版所有 confirmatory
  分析排除 `self_s1`**；confirmatory 只用 `C0 = t17w1self__2..6`，探索性用 `S7`（红队 stats-eval-10）。
- `l1_512k/1M/1.5M/2M` 是**同一 run（`t17long` seed1）的 4 个 step**，彼此强相关；**只取 `l1_1M` 一个**
  进任何 fit，避免相关性权重过大（红队 selfplay-dynamics-12）。
- 共享 MLE 绝对刻度（`t17-recalibration.md` §1，同一 fit）：
  `l1_1M` 191.61 [177.90,204.64]、`l1_2M` 191.11、`p1_499k` 190.06 [174.21,207.59]、
  `l1_512k` 187.60、`p1_262k` 177.56；`p2_500k/p3_500k` **不在该筛选表内**（只筛了 p1）。
  注意该表的 CI 宽达 ±13–16 Elo → "CI 重叠"不是任何强度前提的证据（红队 stats-eval-03）。
- Wave-1 全部 7 个 self 臂 `args.json` 与参照 `t17self__1` 的有效配置差只有 `exp_name`/`seed`，
  `t17w1self__2` 实测 `self_play_sample: false`（`runs/t17w1self__2__1790492897/args.json`）。

### 1.2 池 / self 刷新 / argmax 语义（已逐行核实）

- `parse_pool_member` 支持 `[WEIGHT@]SPEC`，`SPEC ∈ {random, self, ckpt:<path>}`
  （`src/seven523/league.py:76`）。`--opponent pool` 时 `build_league` **总是** `deepcopy(agent)`
  出一个 `frozen` 快照（`league.py:156-167`）；池成员里 `spec == "self"` 映射到**这个 `frozen`**
  （`league.py:185`），`ckpt:` 成员是**静态加载**。
- 刷新：每 `--self-play-refresh` 个 update 执行 `frozen.agent.load_state_dict(agent.state_dict())`
  （`src/seven523/train.py:695-703`）；`--self-play-refresh 0` 短路、永不刷新。
  → **`self` 成员"移动"，`ckpt:` 成员"冻结"**；没有 `self` 成员时 `frozen` 仍被创建但无人消费。
- `--pool-episode True` = `EpisodeMixturePolicy`（**逐局冻结一个成员**，`league.py:107`、
  `src/seven523/policies.py:105`），按权重抽样；默认 `False` = 逐决策 `MixturePolicy`。
- **argmax 混淆（红队 selfplay-dynamics-01，blocker）**：`NeuralPolicy` 默认 `sample=False`
  （`src/seven523/networks.py:758`、`:826/:834` 走 argmax），`LeagueConfig.self_play_sample`
  默认 `False`（`league.py:63`），`--self-play-sample` 默认 `False`（`train.py:362`）；
  且 `policy_from_spec` 构造 `ckpt:` 成员时**不传 sample**（`policies.py:286`）→ 所有
  `self`/`ckpt:` 成员都是 **argmax 确定性对手**；`RandomBot` 才是随机的（`policies.py:56`）。
  → 本版：**主对照 `C0` 与 A1/A2/A3 的对手都是 argmax，故"池化 vs self"的主对比不受该混淆**；
  但**熵（H2/D3/D5）与 `--opponent random` 对照被混淆**，故 H2 全部降级为探索性，并加 A3
  (`--self-play-sample True`) 作为唯一直接对照。
- `--pfsp` 要求 `--opponent pool --pool-episode`（`train.py:477`）；PFSP 本批不用。
- **结论**：`self + frozen ckpt` 混合池、单成员池、`--self-play-sample` 全部 CLI 现成，无需改 src。
  **"多个可刷新的 self 成员 / 联赛"做不到** → Phase 2，本工作流**停下报告**，不即兴改。

### 1.3 硬性局限（写进报告，不得掩盖）

1. **无 self 训练快照**：无法观测同 run 内的策略循环/追逐。
2. **终点群体非传递 ≠ 训练时追逐**：D1 即使检出循环也只是终点群体现象。
3. **同族/同 run 相关**：`p1_499k` 与 `l1_1M` 共用 seed1；`l1_*` 是同 run。诊断按 run 聚簇。
4. **`self_s1` 选择偏差**：只进探索性分析，禁入 confirmatory 族。
5. **无中立异族对手（blocker）**：全部强资产 random-trained；结论只限"random-trained MLP 家族内"。
6. **评测污染**：进训练池的 ckpt 不得当该实验的主锚。

---

## 2. 假设与可证伪预测（修复后）

记号：`C0 = {self_s2..self_s6}`（confirmatory 对照族，k=5）；`S7 = {self_s2..self_s8}`（n=7，
**仅探索性**，D1/D3/D5 用；`self_s7/s8` 同时留作加 seed 备用）；
`X = {p1_499k, p2_500k, p3_500k, l1_1M}`（评测参照族，**冻结命名**，不再增删）；
`wr(a,b)` = a 对 b 的换座胜率；`e(a,b)` = 单刻度 BT/PL 期望胜率。

### 2.1 主假设（confirmatory，见 §5 门）

| 假设 | 机制陈述 | 训练臂预测 | 证伪观测 |
|---|---|---|---|
| **H1 外部多样性** | 纯 self 退化的原因是"只和自己打"（外部多样性缺失） | `A1 poolself` vs 同 seed `C0` 的配对差 `Δ≥+10` 且 CI 排 0；而 `A2 fixed`（单一固定外部）**不**达标 | `A1` 不达标而 `A2` 达标 → 不是多样性，而是"任意固定外部对手"即可 |
| **H1' 对手身份（备择）** | 只要有一个**固定外部强对手**（不必是池），退化即可缓解 | `A2 fixed` vs `C0` 达标 | `A2` 不达标 → "外部对手存在"本身不是杠杆 |
| **H2 greedy 确定性（备择）** | 退化来自对手是 **argmax**（可被利用的确定性），而非 self 镜像本身 | `A3 selfsamp`（随机化 frozen self）vs `C0`（argmax self）达标 | `A3` 不达标 → argmaxness 不是驱动 |

### 2.2 机制假设（**探索性**，D1–D5，不承载行动声明）

| 假设 | 机制陈述 | 探索会看到 | 说明 |
|---|---|---|---|
| **H-mirror** | self 臂专精于自身训练族（不同 seed 的 self 模型）风格 | D2 `δ>0`、D4 `MA>0` | 只测"同 regime 风格重叠"，**不是** model-vs-自己镜像（工程 cli-4） |
| **H-det** | self 训练更早使 **模板头** 确定化 | D3/D5 的**模板头熵**曲线 | raw `metrics.csv entropy` 是 134 模板头+4 花色头之和，**花色头占 85–90%**（`wave5-500k-report.md:102-114`），不得当确定化读数 |
| **H-cycle** | 终点群体存在 A>b,b>c,c>A 的非传递 | D1 的残差/循环筛查 | 现有样本量下**零功效**，只作筛查，不得写"无循环" |
| **H0 噪声** | 上面皆非，读数由训练 seed 方差解释 | D1–D5 全 null；配对 CI 跨 0 | Wave-1：训练 seed sd **8.43** 是 binding term；CI 跨 0 一律写"未判定" |

### 2.3 T5 四个竞争解释 → 显式备择假设（红队 selfplay-dynamics-07）

T5（`opponent-distribution-500k.md` §7）null 有 4 个风险；本版逐条给证伪判据：

| AH | T5 风险 | 本版如何区分 | 证伪判据 |
|---|---|---|---|
| AH1 | 成员全弱于学习者 | D0 非劣检验（§3 D0）；池成员 `p2/p3` 与 500k 学习者代理 `p1_499k` 比 | D0 过门且 `A1` 达标 → 排除 AH1 |
| AH2 | 只有 1 个训练 seed | k=5 配对族 + binding-term 升级 | 5 seed 仍 CI 跨 0 → AH2 不是主因 |
| AH3 | 父模型复制天花板（warm start） | 本版 **cold start**（与 T5 不同） | cold start 下 `A1` 达标 → AH3 非主因 |
| AH4 | PFSP 梯度被 uniform 0.5 稀释 | 本版 **不用 PFSP**（纯均匀 + EpisodeMixture） | `A1` null 而 `A2`/`A3` 有信号 → 非 PFSP 稀释 |
| AH5 | 固定对手池根本无效 | `A2`（单固定）与 `A1`（池）分离 | `A1≈A2≈0` → "池/多样性"不是杠杆 |
| AH6 | random 成员稀释（T5 §6） | 本版池**去掉 random** | `A1` 达标 → 排除 AH6 |

---

## 3. 离线诊断（不训练；只评测现有 ckpt；严格串行 `--device cpu --workers 4`）

> **标签（硬性）**：D0 是**门（半确认：决定池臂结论能否宣称"强成员"）**；D1–D5 **全部探索性**
> （README §3：只有 confirmatory 端点承载行动声明）。产物写 `runs/selfpool/`（允许）。
> 先 `mkdir -p runs/selfpool`；批量前 `free -h`/`nvidia-smi`；一次一个工具，禁止 fan-out。

### D0 成员强度非劣检验（门，只读 + 1 次共享 fit）

- 目的：回答"`p2/p3_500k` 是否**非劣**于 500k 冷启动学习者代理 `p1_499k`，且可接受地接近
  `l1_1M`"，替代 revision-1 的"CI 重叠"（红队 stats-eval-03/selfplay-dynamics-11）。
- 命令（**先删旧 games，防 append 重复**；`--study` 指到允许目录；红队 engineering-cli-1/6）：
  ```bash
  mkdir -p runs/selfpool
  rm -f runs/selfpool/d0_games.jsonl
  .venv/bin/python tools/build_ladder.py \
    --candidate p2_500k=ckpt:runs/t17pool__2__1790439615/agent.pt \
    --candidate p3_500k=ckpt:runs/t17pool__3__1790439615/agent.pt \
    --candidate p1_499k=ckpt:runs/t17pool__1__1790439615/agent.pt \
    --candidate l1_1M=ckpt:runs/t17long__1__1790439615/snapshots/checkpoint_step1024000.pt \
    --anchor random=random --games-per-anchor 400 --cross 400 --no-traces \
    --study runs/selfpool/d0_study --games-out runs/selfpool/d0_games.jsonl \
    --seed 39 --device cpu --workers 4
  # 行数断言的期望值 = 4*400（vs random）+ C(4,2)*400（cross）= 4000
  test "$(wc -l < runs/selfpool/d0_games.jsonl)" -eq 4000 \
    || { echo "D0 rows != 4000 -> append/duplication bug"; exit 1; }
  .venv/bin/python tools/refit_mle.py --games runs/selfpool/d0_games.jsonl \
    --bootstrap 200 --json --out runs/selfpool/d0_absolute_table.json
  ```
- **判定只用 `refit_mle` 的 probit-MLE（deal-clustered bootstrap）**；`build_ladder` 的
  OpenSkill/PL 表**仅信息性**（红队 engineering-cli-11）。
- **预注册门（非劣，差额用同 fit 的 deal-clustered bootstrap 差 CI）**：定义
  `Δ_budget(m)=μ(m)−μ(p1_499k)`、`Δ_anchor(m)=μ(m)−μ(l1_1M)`，对 `m∈{p2_500k,p3_500k}`：
  - `pass_budget(m)`：`Δ_budget(m)` 的 95% CI **下界 > −5 Elo**；
  - `pass_anchor(m)`：`Δ_anchor(m)` 的 95% CI **下界 > −10 Elo**。
  - **过门** = 两个成员都 `pass_budget` **且** 都 `pass_anchor`；否则**降级**：池臂结论只能写
    "弱成员池重估"，不得宣称检验了强成员池。
  - **阈值口径说明（第二轮 repair，红队 stats-eval-03 复盘）**：本门**故意用两个非劣阈值**而非单一
    阈值，因为两个比较对象回答不同问题——`Δ_budget` 对同预算 500k 学习者代理 `p1_499k`（预算校正），
    `Δ_anchor` 对标尺锚 `l1_1M`（跨预算绝对刻度）。两者都必过，判定是**合取**、完全可判定；这偏离
    finding 的"单一阈值"措辞，但保留了非劣性（不是 revision-1 的"CI 重叠"），故不视为缺陷。
  - 差额 CI 由 `runs/selfpool/d0_diff.py`（仅写 runs/，复用 `seven523.mle.fit_mle` 在
    deal-clustered bootstrap 上重拟合，算法同 `tools/refit_mle.py:bootstrap_cis`）直接给出；
    若执行者判断该脚本不必要，退化用两条 marginal CI 的保守"CI 不重叠"规则并**如实标注口径更严**。
- 成本：4000 局，≈ 1–3 min（吞吐锚：`event-history-pilot.md:54`，2400 局 27–35 s）。

### D1 终点群体 round-robin（**探索性**；H-cycle 筛查）

- 群体（12 候选 + `random` 锚，冻结）：`self_s2..s8`(7) + **镜像对照** `self_s2` 与
  `self_s2_mirror`（同一 ckpt 两个 id，红队 engineering-cli-4） + `p1/p2/p3`(3) + `l1_1M`(1) = 12。
  即 `self_s2_mirror` 与 `self_s2` 构成同参镜像对，不再额外复制。
- 命令：
  ```bash
  rm -rf runs/selfpool/arena_core_games          # arena --games-out 是 DIR，rerun 安全但不留旧 shard
  .venv/bin/python tools/arena.py \
    --entrant self_s2=ckpt:runs/t17w1self__2__1790492897/agent.pt \
    --entrant self_s3=ckpt:runs/t17w1self__3__1790492897/agent.pt \
    --entrant self_s4=ckpt:runs/t17w1self__4__1790492897/agent.pt \
    --entrant self_s5=ckpt:runs/t17w1self__5__1790493881/agent.pt \
    --entrant self_s6=ckpt:runs/t17w1self__6__1790493881/agent.pt \
    --entrant self_s7=ckpt:runs/t17w1self__7__1790493881/agent.pt \
    --entrant self_s8=ckpt:runs/t17w1self__8__1790494847/agent.pt \
    --entrant self_s2_mirror=ckpt:runs/t17w1self__2__1790492897/agent.pt \
    --entrant p1=ckpt:runs/t17pool__1__1790439615/agent.pt \
    --entrant p2=ckpt:runs/t17pool__2__1790439615/agent.pt \
    --entrant p3=ckpt:runs/t17pool__3__1790439615/agent.pt \
    --entrant l1_1M=ckpt:runs/t17long__1__1790439615/snapshots/checkpoint_step1024000.pt \
    --anchor random=random --games-per-anchor 400 --cross 400 --seed 40 \
    --device cpu --workers 4 \
    --out runs/selfpool/arena_core.json --games-out runs/selfpool/arena_core_games/ \
    --tb runs/selfpool/arena_core_tb
  ```
  （`--cross 400` = 每对 400 局 = 200 副牌；总量 `C(12,2)*400 + 12*400 = 66*400 + 4,800 = 31,200` 局，
  ≈ 6–12 min。相对 revision-1 的 `--cross 100`（每对 100 局、winrate SE≈0.05）把每边 SE 降到 ≈0.025。）
- **功效预注册（红队 selfplay-dynamics-04/stats-eval-07）**：每对 400 局（200 副 × 换座 2 局），
  winrate SE `≈sqrt(0.25/400)=0.025`，
  95% 半宽 ≈ ±0.049 ≈ ±34 Elo；`pair_diagnostics` 的 `se` 用 `games//2 = 200` 副（`arena.py:262`）。
  因此 **D1 能可靠检出的是 ≥ ~34 Elo 的边差**；self 族真实 pairwise 差仅 ~5–15 Elo →
  **D1 对真实循环零功效**，这是预注册事实，不得把"未检出"写成"无循环"。
- 预注册读数（**全部探索性**）：
  1. **镜像校准**：`self_s2|self_s2_mirror` 的残差 `observed−expected` 的 95% CI 应含 0
     （同参 argmax 镜像的对称期望）。若不含，说明残差估计有偏，D2/D4 一并作废。
  2. **最大残差表**：`pair_diagnostics` top-k，附每对的 `games/observed/expected/z`
     （`z` 是 smoke test，`arena.py:250` docstring 明写**未校准**，不得当检验阈值
     —— 红队 stats-eval-04）。
  3. **循环筛查**：报告 C(12,3)=220 个三元组中"多数方向成环"的个数与最强的一个三元组，
     并**明确写"扫描 220 个三元组、无 FWER 控制、仅探索"**（红队 stats-eval-02）。
  4. **升级条件**：仅当某三元组三条边的点差**每条 ≥ 34 Elo**（D1 的 MDE）时，才允许对它做
     **单三元组 confirmatory 确认**：三条边各用**与发现流零交集**的 deal seed（§3.6），
     `--pairs 400`，要求三条边各自 CI 排除 0.5 且方向成环。否则写"D1 功效不足，H-cycle 不可判"。
- 排序稳定性子分析（探索）：`C0` 子竞技场 vs 全量排名的 Spearman，只报数。

### D2 族交互项 `δ`（**探索性**；H-mirror）

- 输入：`runs/selfpool/arena_core_games/shard_*.jsonl`（**glob 所有 shard**；`build_ladder` 的
  `--games-out` 是单 FILE，`arena` 的 `--games-out` 是 DIR，语义不同 —— 红队 engineering-cli-7）。
- 做法：`runs/selfpool/fit_family.py`（仅写 runs/，仿 `runs/t17w1/aggregate.py` 先例）拟合
  `logit P(a>b)=μ_a−μ_b+δ·1[same_family(a,b)]`，**剔除 `kind=="anchor"`（random 锚局）**，
  按 deal 聚簇 bootstrap。族标签：`self={self_s2..s8}`、`ext={p1,p2,p3,l1_1M}`；
  **排除 `self_s2`/`self_s2_mirror` 镜像对**（只用于 D1 校准）。按 run 聚簇（`l1_1M` 单点、`p1` 与 `l1_1M`
  共用 seed1 需登记）。
- 读数：`δ` 的 deal-clustered 95% CI；`δ>0` 且排 0 → "同 regime 风格重叠"的弱旁证。
- **D2 与 D4 出自同一批对局，不是独立证据**；报告里合并为一条证据（红队 stats-eval-16）。
- 成本：分钟级；若不允许写 runs/ 脚本，则本项取消并在报告标注。

### D3 熵 vs 迁移相关性（**探索性**，n=7）

- 输入：`S7`（`t17w1self__2..8`，n=7；**非 confirmatory `C0`**）的**模板头熵**（见 D5）与 Wave-1 F1b 迁移
  `runs/t17w1/h2h_selfVS1M_s{2..8}.json` 的 `combined.elo_diff.mean`。
- 预注册：Spearman ρ（n=7）+ **精确置换 p**（7!=5040）。**不设 ρ 阈值**；
  判定 = `p≤0.05` 才写"suggestive 机械关联"，否则写 "n=7 功效不足、不可判"。
  （revision-1 的 `|ρ|≥0.7 且 p≤0.05` 在 n=7 数学上不可达：精确 `P(|ρ|≥0.7)=0.088`，
  可达阈值其实是 `|ρ|≥0.78`（精确 p=0.048）；只报 p 最稳 —— 红队 stats-eval-06/selfplay-dynamics-10。）

### D4 镜像残差 `MA`（**探索性**，H-mirror；用户点名）

- 定义（强度校正）：对 `s∈C0`，
  `MA(s)=mean_{b∈C0\{s}}(wr(s,b)−e(s,b)) − mean_{x∈X}(wr(s,x)−e(s,x))`，
  `e` 来自 D2 单刻度拟合，`wr` 来自 D1 逐局。
- **placebo `MA_X` 只用独立 run**：对 `x∈{p1,p2,p3}` 做"族内 vs 对 self 族"的同类差
  （**不用 `l1_1M`**，避免同 run 孪生的构造 0.5 —— 红队 stats-eval-09）。
- 判定（探索）：`mean_s MA` 的 deal-clustered CI 排除 0 **且** `MA_X` CI 含 0 → "弱旁证"；
  否则不写。同时报未校正 `wr` 原始差。
- 成本：复用 D1 逐局，零额外对局。

### D5 终点确定化对比（**探索性**，H-det）

- **快照可得性（硬约束，先声明 — 第二轮 verify blocker）**：self 臂 `S7`（`t17w1self__2..8`）
  **没有 `snapshots/`**（§1.3-1：`ls` 只有 `agent.pt/checkpoint.pt/metrics.csv`），故 self 族
  **只在终点 500k 有模型**；random 臂（`t17pool`）才有 `16k..499k` 快照。
  → revision-2 的"任一匹配 step ≤250k"判据**不可评估**（该 step 无 self 模型），本版把 D5
  **重写为终点同 step 对比**，任何跨时程读数一律**显式标注为跨时程、非主读数**。
- 输入（**架构/终点匹配**）：self 臂 `S7`（hidden=128、500k）vs `t17pool` `p1/p2/p3`（hidden=128、500k）。
  **`t17w1cap256/512`（hidden 256/512、~200k）与 `t17long`（2M）单列，不入主对比**
  （红队 stats-eval-11）。
- **主读数 = 模板头 masked entropy**：`runs/selfpool/probe_entropy.py`（仅写 runs/）在一组固定
  合法 161 维状态上，按 `networks.py` 的 per-head split 重算 template-head masked entropy。
  **禁止**用 raw `metrics.csv entropy` 当确定化读数（花色头占 85–90%，红队 selfplay-dynamics-05）。
  若 probe 不实现 → D5 取消并在报告标注，**不得**退回 raw entropy 下结论。
- **主判定（探索，唯一匹配 step = 500k）**：在 **500k** 处报 self 族（`S7`，n=7）与 random 族
  （`t17pool p1/p2/p3`，n=3）模板头 masked entropy 的均值差与 deal-clustered 自举 CI。
  - **不设任意的绝对 0.2 阈值（第二轮 repair）**：`wave5-500k-report.md:102-114` 的 `H_template`
    实测仅 `0.13–0.26` nats，故"低 ≥0.2"≈ 要求 self 熵塌到 ~0，在模板头尺度上**近乎不可达**。
    本版只报差值 + CI；"self 显著更低（CI 排 0）"作为描述性弱旁证，不承载行动声明。
- 跨时程参考（**显式标注为跨时程，非主读数**）：random 族有 `16k..499k` 快照、self 族只有 500k。
  允许画 random 族熵随 step 的曲线作背景，并标注 self 唯一可用点是 500k；**不得**用
  "self 在某 step <250k 更低"作判据（该 step 无 self 模型，不可观测）。
- 匹配 step raw 熵（仅记录、不作确定化证据；红队 selfplay-dynamics-09）：self@500k `1.33–1.55`
  （raw 和），random-trained@500k = `t17pool` `1.648/1.671/1.755`；`t17long@2M=1.603` 落在 self 带内。
  revision-1 引用的 `1.80–1.89` 是 **200k cap 臂**，时程不匹配，作废。

### 3.6 deal-seed 台账与 "fresh" 定义（红队 stats-eval-01/05/13）

- **已占用（不得复用）**：`0–8`、`10–18`（Wave-1）、`29–37`（Phase-2 复制保留）。
- **本计划分配（两两不相交）**：
  | 用途 | base seed / deal seed |
  |---|---|
  | D0（ladder 共享 fit） | `--seed 39`（独立 fit，不进任何 h2h 推断；**不用 0**，0–8 已占用） |
  | D1 arena 发现 | `--seed 40` |
  | D1 确认（条件触发） | `140–148` |
  | E1 `A1 vs C0` | `100–108` |
  | E2 `A2 vs C0` | `110–118` |
  | E3 `A3 vs C0` | `120–128` |
  | 描述性 vs `l1_1M` | `130–138` |
- **"fresh" 定义**：一个 base/deal seed 是 fresh ⟺ 它不在已占用集合，且未被本计划任何其他端点/发现
  使用。**不要**把"与 Wave-1 的 10–18 不重叠"当成唯一的 fresh 条件（revision-1 的缺陷）。
- **结构重叠机制（已实测）**：`plan_games`（`ladder.py:251-252`）与 `plan_duel_schedule`
  （`duel.py:50-53`）都从 `random.Random(base_seed)` 抽 deal seed；**同一 base seed 会导致前缀重叠**。
  实测：arena(`--seed 20`,`--games-per-anchor 100`,`--cross 100`) 的 50 个 cross deal seed 与
  duel(`--seed 20`,`--pairs 400`) 的 400 个 deal seed **完全重叠（50/50）**；与 duel(21)/duel(22) 重叠 0。
  本版 arena 用 base seed 40、h2h 用 100+，实测重叠 0。

### 诊断成本汇总

| id | 成本 | 标签 |
|---|---|---|
| D0 | ~1–3 min | 门（半确认） |
| D1 | ~6–12 min | 探索 |
| D2 | 分钟 | 探索（若允 runs/ 脚本） |
| D3 | 分钟 | 探索 |
| D4 | 复用 D1 | 探索 |
| D5 | 分钟（需 runs/ probe） | 探索 |

---

## 4. 池化实验设计（预注册，cold start，k=5）

### 4.1 臂定义（预注册，冻结；3 臂 × 5 seed = 15 run = 硬上限）

**评测污染边界**：
- confirmatory 对照 = **同 seed 纯 self `C0`**（`t17w1self__2..6`，Wave-1 冻结资产，无 post-hoc）；
- 训练 frozen 成员 = `p2_500k`/`p3_500k`（`t17pool` seed2/3），与 `C0`、与 `l1_1M` 的 run/seed 双避；
- `p1_499k`/`l1_1M`/`l1_2M` 只作**描述性**参照，**不进训练池**。

| 臂 | run 前缀 | seed | 池成员 / 对手 | 回答 |
|---|---|---|---|---|
| **A1** `poolself` | `t18poolself__` | 2–6 | `1@self`(refresh 10) + `1@ckpt:p2` + `1@ckpt:p3` | 池化外部对手（用户点名） |
| **A2** `fixed` | `t18fixed__` | 2–6 | `1@ckpt:p2`（单一固定外部，无 self） | 证伪"只和自己打" |
| **A3** `selfsamp` | `t18selfsamp__` | 2–6 | `1@self`(refresh 10) + `--self-play-sample True` | argmax 混淆对照 |

- **A1 不含 `random`**：避免 T5 式稀释、并让 vs-RandomBot 的 F0 绝对锚干净（红队
  stats-eval-14/selfplay-dynamics-06）。代价：不复现 T5 的 random 混合分支（已由 AH6 覆盖）。
- **A1 vs A2 的差异同时包含"self 成员存在"与"外部权重重归一化"**（A1 每外部 1/3，A2 单个 100%）。
  故 **A1−A2 只作描述性**，不作为"self 成员"的孤立因果（红队 engineering-cli-3/selfplay-dynamics-06）。
  本版**不做** P1/P2 式"移动 self 成员"的孤立对比（预算不足）；该问题留 Phase 2（需要
  "refresh 10 vs refresh 0 同池"的第四臂）。
- 控制臂 `C0` 复用 Wave-1（0 新 run）；其 `args.json` 与新臂差异仅为 `opponent/pool_*/self_play_sample/exp_name`。

### 4.2 精确训练命令（500k，cold start，**并发上限 3**）

以下用**显式有界调度器**（`xargs -P 3`），**不要**裸 `for` 串行（红队 engineering-cli-2）：

```bash
mkdir -p runs/selfpool runs/t18
cat > runs/selfpool/train_jobs.txt <<'EOF'
t18poolself 2
t18poolself 3
t18poolself 4
t18poolself 5
t18poolself 6
t18fixed 2
t18fixed 3
t18fixed 4
t18fixed 5
t18fixed 6
t18selfsamp 2
t18selfsamp 3
t18selfsamp 4
t18selfsamp 5
t18selfsamp 6
EOF
# 每行 <exp> <seed>；-P 3 把并发封在 3，nice -n 5 让出 CPU
cat runs/selfpool/train_jobs.txt | xargs -P 3 -n 2 bash -c '
  set -e
  exp="$1"; seed="$2"
  case "$exp" in
    t18poolself) extra="--opponent pool --pool-episode True --self-play-refresh 10 \
      --pool-member 1@self \
      --pool-member 1@ckpt:runs/t17pool__2__1790439615/agent.pt \
      --pool-member 1@ckpt:runs/t17pool__3__1790439615/agent.pt" ;;
    t18fixed)    extra="--opponent pool --pool-episode True --self-play-refresh 0 \
      --pool-member 1@ckpt:runs/t17pool__2__1790439615/agent.pt" ;;
    t18selfsamp) extra="--opponent self --self-play-refresh 10 --self-play-sample True" ;;
  esac
  nice -n 5 .venv/bin/python -m seven523.train --exp-name "$exp" --seed "$seed" \
    --total-timesteps 500000 $extra --cuda True --tensorboard False --run-dir runs
' _
```
- run 名：`runs/t18poolself__{2..6}__<ts>/`、`runs/t18fixed__{2..6}__<ts>/`、`runs/t18selfsamp__{2..6}__<ts>/`。
- **未使用**任何不存在的 CLI（§1.2）；A2 的单成员池 = `EpisodeMixturePolicy` 单成员 = 固定对手。

### 4.3 预算

| 臂 | run | 等效（500k=1） |
|---|---|---|
| 控制 `C0` | 0（复用 Wave-1） | 0 |
| A1 / A2 / A3 | 5+5+5 | 15.0 |
| **Phase 1 合计** | 15 | **15.0 = 上限** |
| PFSP / snapshot 联赛 / 第四臂 | Phase 2，另批 | 触发才申请 |

### 4.4 与 T5 的差异（逐条）

| 维度 | T5（旧，null） | 本计划 |
|---|---|---|
| 规则/obs | 旧规则 / obs v1(191) | revision-3 / obs v5(161) |
| 起点 | warm start `base680k` | **cold start** |
| 池成员强度 | 全弱于学习者 | `p2/p3_500k`，D0 **非劣**检验 |
| self 成员 | 无/静态 | A1/A3 刷新式 self；A2 无 |
| 逐局冻结 | 有 | 有（`--pool-episode True`） |
| random 成员 | 有 | **无**（去稀释） |
| PFSP | 三臂之一 | 不用（AH4 显式排除） |
| 训练 seed | 1 | 5（配对族） |
| 主端点 | 3×400 vs 无固定锚 | **同 seed 配对 vs `C0`**，k=5 × 9 deal |
| 归因 | 三效应全混 | A1/A2/A3 三臂**各自** vs 同一 `C0` 分离 |

---

## 5. 端点与判定

### 5.1 端点（预注册）

对 `A∈{A1,A2,A3}`、`s∈{2,3,4,5,6}`、`C0_s = t17w1self__s`：
- **E_A 主端点（confirmatory，配对）**：
  `Δ_A = merge_s combined.elo_diff(A_s vs C0_s)`，
  deal seed = E1/E2/E3 对应块（§3.6），`--pairs 400 --bootstrap 4000`，
  合并 `mean ± q·max(RMS(per-seed combined.se), training seed sd)/√k`，**同时报 `q=1.96` 与
  `q=t_{0.975,4}=2.776`**（复用 `runs/t17w1/aggregate.py`，三个 `--pair` 分开调用）。
- **E_anchor（descriptive）**：**只跑 A1（`poolself`）**，`A1_s vs l1_1M`，deal `130–138`；
  **500k vs 1M，不可与前端点同读**（红队 stats-eval-08）。同时报同预算的 `p1_499k` 作为描述性参照。
  **作用域明写（第二轮 repair）**：E_anchor 只覆盖 A1（5 条 h2h：s2..s6），**不**覆盖 A2/A3；
  §6.1 step 7 与附 A 的描述性计数（5 条）与此一致。若日后要 A2/A3 的 vs-`l1_1M`，属额外评测、另计数。
- **E0（descriptive sanity）**：`build_ladder` 把 `A1_s2/A2_s2/A3_s2/l1_1M/p2/p3` 放进同一
  shared fit（vs RandomBot）。因池/臂**不含 random**，该读数干净，但仍是单 seed → 只做
  "无灾难性退化" sanity，**不进 action table**（红队 stats-eval-14）。

### 5.2 预注册门（冻结）

先对三个 confirmatory 端点做 **Holm**（α=0.05，双侧；`plans.md` §7）。单臂门：

| 条件 | 判定 | 动作 |
|---|---|---|
| `Δ_A ≥ +10` **且** z/t CI 均排 0 | 该臂确认为杠杆 | 报告升级；另对 +20 做单侧非劣 |
| `+5 < Δ_A`，或 CI 跨 0 但点 > +5 | 未达门槛 | 按 binding term 升级（训练 seed sd binding → 加 seed 到 k=7，用 `C0` 的 `self_s7/s8`，≤4 run；否则加 deal） |
| `Δ_A ≤ +5` | 无证据 | **止损该臂** |
| 三臂全 ≤+5 | 池化线收束 | 写"在 random-trained MLP 家族内，对手分布修改不是绑定约束" |

**解读矩阵（confirmatory 主结论）**：

| A1 (pool) | A2 (single fixed) | A3 (sample self) | 结论 |
|---|---|---|---|
| ≥+10 | ≈0 | 任意 | **H1 支持**：外部**多样性/池**是杠杆，单一固定对手不够 |
| ≥+10 | ≥+10 | 任意 | **H1 不获支持**：任意外部强对手即可（对手身份/强度，而非 self 特异性） |
| ≈0 | ≈0 | ≥+10 | **H2 支持**：退化来自 argmax 对手确定性（greedy），不是 self 镜像 |
| ≈0 | ≈0 | ≈0 | 均为"未判定"（若 CI 窄）/ H0 噪声（若 CI 跨 0） |

- **功效预注册（`v5-optimization-plan.md` §1.4，k=5 行）**：sd=10 → z 半宽 8.8 / t 半宽 12.4；
  sd=16 → z 14.0 / t 19.9。**书面目标 = +20**，+10 为行动下限。
  **诚实说明**：k=5 时若训练 seed sd 仍 ~8.4，t 半宽 ≈ 10.4 → 真实效应 ≈ +10 大概率"未判定"；
  这是 15-run 硬预算的数学现实，**不得写成 null**（红队 selfplay-dynamics-13）。
- **禁止用测试集挑 ckpt**：池成员由 run 身份先验选定；锚固定；Phase-2 任何胜者需在 `29–37`
  或新训练 seed 上确认。

### 5.3 观测清单

每臂报 `metrics.csv` 的 `entropy/episodic_return/episodic_length/explained_variance`
（raw entropy 仅记录、**不作确定化证据**）、`args.json` 与命令一致性、以及 `C0` 部分。
**熵门（探索性）**：若 `A1`/`A2` 末模板头熵仍低且 `Δ≤+5`，写"确定化伴随无迁移（探索）"，
不改判定。

---

## 6. 分步执行、验收、回滚、风险、ADR

### 6.1 分步（诊断先出结论，再训练；不 fan-out）

1. `mkdir -p runs/selfpool`；`free -h`；`nvidia-smi`；确认无残留 `seven523.train`/h2h 进程。
2. **冻结代码**：记录 `src/` 文件 hash / `git diff --stat`；声明执行期间不改 src/tests/tools。
3. **D0（门）→ D1 → D2 → D3/D4/D5**，严格串行、一次一个工具、`--device cpu --workers 4`。
4. **诊断小结**：按 §2 给 H1/H1'/H2（confirmatory 待训练）与 H-mirror/H-det/H-cycle（探索）的结论。
5. **决策点**：D0 不过门 → 先报 owner，池臂结论降级为"弱成员池重估"。
6. **Phase 1 训练**：§4.2 的有界调度器（`xargs -P 3`），15 run，≤3 并发。
   单 run ≈ 13–26 min，15 run / 3 并发 ≈ 1.1–2.2 h。
7. **评测（严格串行、一次一个、`--device cpu --workers 4`）**：
   E1/E2/E3（各 5 条 × 9 deal）+ 描述性 vs `l1_1M`（**仅 A1**，5 条）+ E0。
   吞吐锚（`event-history-pilot.md:54`）：9 deal ≈ 80–110 s/条 → 20 条 ≈ 30–40 min（**不要**沿用
   revision-1 的 6–10 min/条 —— 红队 engineering-cli-9）。
8. **合并**（**每对一次**，红队 engineering-cli-5）：
   ```bash
   for pair in poolselfVSself fixedVSself selfsampVSself; do
     .venv/bin/python runs/t17w1/aggregate.py --dir runs/selfpool \
       --pair "$pair" --seeds s2,s3,s4,s5,s6 --family confirmatory
   done
   ```
9. **报告** `docs/experiments/selfplay-pool-wave.md` + `docs/experiments/README.md §1` 索引（允许）。

### 6.2 验收

- D0–D5 有可复现产物与预注册读数；D0 行数断言通过。
- A1/A2/A3 各 5 run 训到目标 step、无 NaN；`args.json` 与 §4.2 命令逐键一致（除 `exp_name/seed`）。
- E1/E2/E3 有合并结果，z/t 双口径、三方差分量、binding term；Holm 校正；confirmatory/descriptive 标签齐全。
- §5.2 每条门有明确结论；CI 跨 0 一律写"未判定"。
- 未改 src/tests/tools；未用测试集挑 ckpt。

### 6.3 回滚

删除 `runs/t18poolself*`、`runs/t18fixed*`、`runs/t18selfsamp*`、`runs/selfpool/` 即可；
`traces/study` 与 manifest 不触碰；报告追加式。

### 6.4 风险与未决

| # | 风险 | 影响 | 缓解 |
|---|---|---|---|
| R1 | k=5 训练 seed sd >16 → 未判定 | 预算内不决策 | 目标 +20；binding-term 升级到 k=7；不写 null |
| R2 | 评测污染 | 端点虚高 | 主锚=同 seed `C0`；成员 run/seed 双避 |
| R3 | `p2/p3` 强度未知 | "强成员"前提存疑 | D0 非劣门；不过门降级 |
| R4 | 复用 `C0` | 代码漂移 | 冻结代码记录 hash |
| R5 | 无中立异族对手 | 机制结论不可识别 | **明确降级**为"random-trained MLP 家族内"；不写"self-play 本身" |
| R6 | argmax 混淆 | H2/熵 | 加 A3；H2 降级探索 |
| R7 | 需要多 self 成员/联赛 | 越界 | 停下报告，Phase 2 |
| R8 | 并发/内存卡死 | 事故 | 训练 `xargs -P 3` + `nice -n 5`；h2h CPU/4 串行 |
| R9 | 评测墙钟（~30–40 min） | 会话超时 | 每条落盘 JSON；先 E1 后其余 |
| R10 | `p1_499k` 与 `l1_1M` 同 seed | 相关 | 只作描述；`X` 冻结 |

### 6.5 需要的新 ADR

若 Phase 1 确认（臂 CI 排 0 且 ≥+10）且决定 ship 池化，先立 **ADR-0015「池化联赛训练语义」**
（固定 self 刷新/静态语义、逐局冻结、评测污染边界、PFSP 契约、快照选择偏差规则）。
本计划不写 ADR。

---

## 7. 长程

### 7.1 若某臂确认

1. **联赛化**：历史快照做池成员（需改 `league.py`）→ 停下报告 → ADR-0015 → 实现 → 1M 剂量点。
2. **PFSP**：强成员池上启用胜率重加权，与静态池预注册成对比较。
3. **第四臂**：同池 `refresh 10 vs 0`，孤立"移动 self 成员"。
4. **中立异族对手**：若无中立即需少量代码/新 bot（停下报告）。
5. 登记新梯级候选仅当胜过 `lvl4` 且 fresh 确认。

### 7.2 若三臂全 null

1. 池化线收束（在 D0 过门 + 去 random 前提下）→ 对手分布不是绑定约束。
2. 机制问题写"未检出稳定镜像/确定化/非传递证据"，不写"已证明不存在"。
3. 转向 A2 容量（cap512 方向性）或 A5 上限测量（1-ply bot，需新代码）。
4. 若 D1 升级条件触发并确认循环，或 D2 `δ` 排 0 → 机制问题不随池化 null 关闭，转 Phase 2 联赛。

---

## 附 A. 命令速查（**Plan/Red-team/Repair 阶段不得执行**）

```bash
# D0（门）：删旧 + 断言行数 4000
rm -f runs/selfpool/d0_games.jsonl
.venv/bin/python tools/build_ladder.py \
  --candidate p2_500k=ckpt:runs/t17pool__2__1790439615/agent.pt \
  --candidate p3_500k=ckpt:runs/t17pool__3__1790439615/agent.pt \
  --candidate p1_499k=ckpt:runs/t17pool__1__1790439615/agent.pt \
  --candidate l1_1M=ckpt:runs/t17long__1__1790439615/snapshots/checkpoint_step1024000.pt \
  --anchor random=random --games-per-anchor 400 --cross 400 --no-traces \
  --study runs/selfpool/d0_study --games-out runs/selfpool/d0_games.jsonl --seed 39 --device cpu --workers 4
test "$(wc -l < runs/selfpool/d0_games.jsonl)" -eq 4000
.venv/bin/python tools/refit_mle.py --games runs/selfpool/d0_games.jsonl \
  --bootstrap 200 --json --out runs/selfpool/d0_absolute_table.json

# D1（探索）：12 候选 + random 锚，cross=400，base seed 40（完整 entrant 列表见 §3 D1）

# E1/E2/E3（confirmatory，配对 vs 同 seed C0；每 seed 一条，严格串行）
.venv/bin/python tools/head_to_head.py \
  --left pself_s2=ckpt:runs/t18poolself__2__<ts>/agent.pt \
  --right self_s2=ckpt:runs/t17w1self__2__1790492897/agent.pt \
  --seeds 100,101,102,103,104,105,106,107,108 --pairs 400 --bootstrap 4000 \
  --device cpu --workers 4 --json > runs/selfpool/h2h_poolselfVSself_s2.json
# A2: --seeds 110..118, 文件 h2h_fixedVSself_s<N>.json
# A3: --seeds 120..128, 文件 h2h_selfsampVSself_s<N>.json
# 描述性（仅 A1）: --seeds 130..138, 文件 h2h_poolselfVS1M_s<N>.json

# 合并（每对一次）
for pair in poolselfVSself fixedVSself selfsampVSself; do
  .venv/bin/python runs/t17w1/aggregate.py --dir runs/selfpool \
    --pair "$pair" --seeds s2,s3,s4,s5,s6 --family confirmatory
done
```

## 附 B. 预注册摘要

1. confirmatory 对照 = 同 seed `C0=t17w1self__2..6`；训练成员 `p2/p3_500k` run/seed 双避。
2. 3 臂 × 5 seed = 15 run = 硬上限：A1 `poolself`（self+2 ext，无 random）、A2 `fixed`（单固定 p2）、
   A3 `selfsamp`（self + `--self-play-sample True`）。
3. 主端点 = 每臂 vs 同 seed `C0` 的配对差，k=5、9 fresh deal（E1 100–108 / E2 110–118 / E3 120–128）、400 副。
4. 门：≥+10 且 z/t CI 排 0 确认；≤+5 止损；中间加 seed/deal；Holm 三个 confirmatory 端点。
5. deal seed：已占用 `0–8/10–18`；本批 `100–138`；D1 base `40`；Phase-2 `29–37`。fresh = 与任何
   发现/选择集及本计划其他端点零交集。
6. 诊断 D0（门，非劣）–D5（探索）。
7. 不把"CI 跨 0"写 null；不越界写"self-play 本身"（无中立异族对手）。
8. 不改 src/tests/tools；需要新 CLI → 停下报告。

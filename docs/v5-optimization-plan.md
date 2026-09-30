# memoryless v5 优化计划（revision-4，红队修复后）

> **执行状态**：第一波已执行完毕，见 [`experiments/v5-optimization-wave1.md`](experiments/v5-optimization-wave1.md)：
> F1b 主端点合并 **+2.87 Elo**（z/t CI 均跨 0、点估计 ≤ +5）触发 §3.4 **止损** →
> **Stage 2 未执行**，self-play 线判定「无 ≥+10 证据」；下文保留为预注册计划，
> 执行后的结论以该报告为准。

> **角色**：长程规划 agent → 修复/诊断 agent（本轮 = 红队 findings 的逐条处置与规划重写）。
> 本阶段**零训练、零 h2h**；只读代码/只读命令。处置产物只写本文件与
> `docs/v5-optimization-review.md`；未改 `src/`、`tests/`、`tools/`、`ADR/`、`plans.md`、
> `docs/experiments/README.md`、`CONTEXT.md`、`DESIGN.md`；未 commit/checkout/stash；未跑训练或 h2h。
>
> **本版相对 revision-3 的变化**（红队 35 条 findings；逐条处置见
> `docs/v5-optimization-review.md`）：主端点由「F1a self-vs-同 seed random」改为
> 「**F1b self family vs 固定 l1_1M**」；confirmatory 训练 seed 由 3 提到 **7**（全部 fresh seed
> 2–8）；deal seed 由 0,1,2 改为 **fresh 10–18**；新增 **F0 对脚本 RandomBot 的共享 fit 绝对锚**；
> 预算口径、`runs/t17w1` 目录、aggregate 脚本、warm-start 对照、Block C 措辞、A3 判据、
> PFSP 归因、`docs/experiments/README.md §1` 路径等全部修正。所有改动均在本文件内。
>
> **口径标签**：全部设计与数字均属 revision-3（出空即撬底，`rules_id=2e36dbea44893696`）、
> obs v5（2 家 161 维）、纯 MLP（`arch=shared`、`hidden=128`、单层 trunk＝2 个 Linear）。
> 与 revision-2 及更早数字严格隔离。所有外部数字来自既有报告（给 `file:§`），**不发明**。
>
> **统一评测口径**（硬性，`docs/experiments/README.md §3`）：方案对比在 **≥3 训练 seed ×
> ≥3 deal seed × 400 副牌换座 h2h** 上给 `mean ± q·max(bootstrap SE, seed sd)/√k`，
> **同时报** `q=z=1.96` 与 `q=t_{0.975,k-1}`；按牌/seed 聚簇配对 bootstrap。**repo 行动门槛 =
> CI 完全排除 0 且点估计 ≥ +10**，`<10` 一律不追；用户 5 分线（点估计 > +5）只触发**加 seed /
> 继续调优**。绝对强度用 `tools/build_ladder.py`（共享 fit，候选间可比；跨 fit 不可比）。
> 「点估计 ≥ +20」不是判定规则，任何 +20 越线声明走对 +20 的**单侧等效/非劣检验**
> （`docs/plans.md §7 C-1`、`evaluation-protocol-validation.md §6/§9`）。
>
> **资源纪律**（有过卡死事故）：训练最多 **3 进程并发**（`nice -n 5`）；**h2h 评测严格串行、
> 一次一个，只 `--device cpu --workers 4`**（禁止 `--device cuda --workers>1`、禁止 fan-out）。
> 批量前 `free -h` / `nvidia-smi`。

---

## 0. 范围、现状资产与结论先行

### 0.1 本计划要回答的唯一问题

**在「无历史 memoryless v5 纯 MLP」这一契约内，能否把当前 revision-3 池顶（lvl4=185.34）
显著推高？** 历史/序列/事件/座位/意图五条线已被前置报告关闭（§1.3），本轮只做 memoryless 方向。
评测走既有 revision-3 口径。

### 0.2 现成资产（已现场复核，只读）

| 资产 | 路径 | 状态/用途 |
|---|---|---|
| 500k random 池 seed 1/2/3 | `runs/t17pool__{1,2,3}__1790439615/agent.pt` | 均 499,712 步、obs v5、shared/128、`args.json seed=1/2/3` |
| seed1 快照（含 499k） | `runs/t17pool__1__1790439615/snapshots/checkpoint_step499712.pt` | 与 `t17pool__1/agent.pt` **逐参数相等**（已 torch.load 核对 8/8 张量 + `extra.global_step=499712`） |
| 纯 self 500k seed 1 | `runs/t17self__1__1790439615/agent.pt` | `args.json`：`opponent=self`、`self_play_refresh=10`、499,712 步 |
| 2M random 长训 | `runs/t17long__1__1790439615/snapshots/checkpoint_step{512000,1024000,1536000,1999872}.pt` | **l1_1M = lvl4 顶档**（`checkpoint_step1024000.pt`） |
| 早期档 | `runs/t17early{2k,4k,8k}__1__*` | lvl1/lvl2 |
| 发布 study/manifest | `traces/study/`、`runs/t17_mle/` | lvl1–4 绝对表 |
| 训练快照轮询器 | `runs/t17_train/poll_snapshot.py` | 轮询 `checkpoint.pt`/`agent.pt` 抓指定 step；**前台阻塞循环，必须后台跑**（§7） |
| 评测工具 | `tools/head_to_head.py`、`tools/h2h_screen.py`、`tools/build_ladder.py`、`tools/arena.py`、`tools/refit_mle.py` | 全部现成 |

关键事实复核：`agent.pt`/`checkpoint.pt`/快照同属 `save_agent` 格式
（`src/seven523/networks.py:516`），`--load-checkpoint` 可直接吃
`runs/t17long.../snapshots/checkpoint_step1024000.pt`。训练侧自博弈刷新为
`frozen.agent.load_state_dict(agent.state_dict())`（`src/seven523/train.py:695-703`）、
`self_play_refresh=0` 时**永不刷新**（`args.self_play_refresh and ...` 短路）；冻结对手在
`build_league` 用 `deepcopy` 初始化（`src/seven523/league.py:157-167`），`--opponent self`
**忽略** `--mix-random-prob`（`event-history-pilot.md` §3.2）。warm start 只加载权重：
`optimizer = optim.Adam(...)` 每次新建（`train.py:557`）、`anneal_lr` 按新
`num_updates` 重算 `frac`（`train.py:706`）。

训练吞吐（现场 `metrics.csv` 末行 `sps`）：500k random ≈ 1057 sps（t17pool，多并发）、
2M random ≈ 1395 sps（t17long）、纯 self 500k ≈ 618 sps 单跑 / **321–323 sps（3 并发）**
（`runs/t17self__1` metrics 与 `event-history-pilot.md` §4）。→ self 单 run 500k ≈ 13–26 min。

### 0.3 结论先行（本计划的赌注排序，已按红队证据重写）

1. **A1 self-play 仍是唯一方向为正的线，但当前证据「未确立」而非「已确认」**。T17 §5 的
   `+16.62 [+9.30,+23.94]` 是**同一 ckpt（seed 1）对 9 个 deal seed** 的合并；其 p=2.8e-5
   是 deal-cluster bootstrap，**看不到训练 seed 方差**。同一 self ckpt 对三个 random ckpt 的
   读数 +27.9/+10.3/+27.3（`runs/t17_screen/h2h_self_vs_{p1_499k,p2_500k,p3_500k}.json`
   `combined.elo_diff.mean`）给出 random 侧训练 seed 散布 **sd≈10 Elo**。→ 现状只能读作
   「值得投入一个 **k=7 训练 seed** 的确认实验」，不能读作既成正信号（红队 stats-eval-01/03/04、
   training-ml-01/02/03）。
2. **第一波首要交付 = 把 self-play 从 1 个 ckpt 升级为 k=7 fresh 训练 seed 的可判定端点**。
   主端点 = **F1b：self 家族 vs 固定参照 `l1_1M`**（只含 self 侧训练方差；9 个 fresh deal seed）；
   因为 random 侧只有 seed 1/2/3，等预算的 F1a 无法做到 k≥5，**降级为描述性**（红队
   training-ml-02 与 stats-eval-02 的冲突已在 review 中裁定）。
3. **绝对强度锚**：新增 **F0**——用 `tools/build_ladder.py` 把 `self_s2 / l1_1M / lvl3` 放进
   同一共享 fit（`--anchor random`），防止「self 只更会打 random-trained MLP」的相对假提升
   （红队 training-ml-06）。
4. **A2 容量/超参**：只做 200k 粗筛，**措辞限定为「200k k=1 无 ≥+20 信号」，不得写「关闭 A2」**；
   `--num-envs 16` 臂因同时改 batch/minibatch/更新数与 env seed，改为 `--hidden-size` 单因子
   （红队 training-ml-05、stats-eval-07、engineering-ops-06）。
5. **A3 价值诊断**改判据为 **ckpt 内部比较**（早期桶 EV vs 整体 EV vs 末桶 EV + V/return sd）；
   `structural-directions.md` 的 `0.065` 只作 revision-2 历史注脚（红队 training-ml-08）。
6. **A4 选择/集成**与 **A5 上限测量**：纯评测/设计，后置。
7. **gamma 期望效应量级很小**：`--reward-shaping terminal` 默认（`train.py:120`），
   `--gamma 0.99`（`train.py:101`），实测 episodic_length ≈ 20.9–23.7
   （`runs/t17self__1` 20.86、`runs/t17pool__1` 23.74）；γ^21 = 0.810/0.900/0.979 @ 0.99/0.995/0.999，
   episode 起点折扣只从 0.81 变到 0.98。→ 不单列 gamma 预算（红队 training-ml-09）。

第一波分两阶段：**Stage 1（confirmatory，无条件）= 7 个 fresh self run + 3 个 200k 粗筛 +
F0/F1b 评测 → M1**；**Stage 2（条件，仅 M1 点估计 ≥+5 才跑）= self 变体 5 个 run + F1a/F2**。
合计 **14.2 等效 run ≤ 15 上限**（核算见 §3.7）。

---

## 1. 证据与问题

### 1.1 当前有明确、可复现的强度上限证据

- 发布梯级 revision-3：**lvl1–lvl4 = 82.75 / 106.90 / 147.71 / 185.34**（random=0，
  homoscedastic probit-MLE，σ≈5.9–6.3；`t17-recalibration.md` §2）。
- 筛选 13 去重候选的**顶部是平台**：512k/499k/1M/1.5M/2M 五点 μ **187.6–191.6**，CI 两两重叠，
  **无 ≥200、无 ~55**（`t17-recalibration.md` §1）。
- 结构性诊断（仅背景，revision-2/旧口径）：终局 only 奖励非零步 **4.17%**；`base680k`
  早期分桶 EV **0.065**（`structural-directions.md` §1.1）；**24.7 张**已出牌不可见（§1.2）。

### 1.2 self-play 正信号的边界与局限（本轮核心赌注，已修正）

`runs/t17self__1__1790439615`（从零、`--opponent self --self-play-refresh 10`、seed 1、500k）
在 `t17-recalibration.md` §5 的成绩：

| 对手 | ΔElo(self−opp) | 95% CI | seed 维度 |
|---|---|---|---|
| l1_1M（发布 lvl4，MLE 191.6） | 首批 3 deal seed +10.3 [−2.4,+22.9]；**补 deal seed 3–8 后合并 9 deal 得 +16.62 [+9.30,+23.94]** | 显著（deal-cluster） | **同一 ckpt**，仅 deal seed |
| l1_2M（MLE 191.1） | +5.6 [−6.8,+18.1]；补 3 deal 合并 +6.52 [−2.29,+15.33] | 打平 | 同一 ckpt |
| p1_499k（= `t17pool__1/agent.pt` 权重同源） | +27.9 [+13.9,+41.8] | 显著 | 同一 ckpt |
| p2_500k | +10.3 [−2.4,+22.9] | 跨 0 | 同一 ckpt |
| p3_500k | +27.3 [+14.4,+40.1] | 显著 | 同一 ckpt |

**必须正面写进规划的五点**（红队 stats-eval-01/02/03/04、training-ml-01/02/03/06）：

1. **+16.62 是探索性的，不是已复现点估计**。3 个 deal seed 只给 +10.3（跨 0），
   deal seed 3–8 是在看到 3-seed 不显著之后补的 → stop-and-choose 偏差（`t17-recalibration.md`
   §5「补充复核」）。报告里禁止把它当作基线真值。
2. **只有 1 个训练 seed**。上面的「9 seed」全是 deal seed 重测，训练 seed sd 未测。
3. **已存在的 F1a N=1 = +27.9**：`runs/t17_screen/h2h_self_vs_p1_499k.json` 即
   `self_s1 vs rand_s1`（`rand_s1 = t17pool__1/agent.pt = checkpoint_step499712.pt`，已逐参数核对）。
   原 revision-3 §6.5 写「F1a 在 T17 §5 未做」是**错的**（红队 training-ml-03 / engineering-ops-02）。
4. **random 侧训练 seed sd ≈10**（+27.9/+10.3/+27.3 的三点散布），而 self 侧训练 seed sd **未知**。
   现有 +16.62 对训练 seed 噪声的相对量级只有 ~1σ（红队 stats-eval-04）。
5. **相对提升 ≠ 绝对提升**：F1a/F1b 两端都是 ckpt（MLP-vs-MLP）。必须补一条 vs **脚本 RandomBot**
   的绝对锚，否则无法排除「self 只是更会打 random-trained MLP」（红队 training-ml-06）。

> `runs/t17_screen/h2h_self_vs_l1_1M.json`：`combined.elo_diff = mean +10.28 [−2.38,+22.94]`，
   `bootstrap_se 11.19`、`between_seed_sd 3.15`（= deal-seed sd）。这说明单训练 seed 下
   deal 分量主导；但跨训练 seed 后 seed-sd 分量会成为主项（§1.4）。

### 1.3 本轮不碰的线（引用既有收束）

历史/序列/事件/座位/意图（§1.1、`sequence-memory-pilot.md`、`sequence-gru-ablation.md`、
`event-history-pilot.md`、`opponent-intent-probe.md`）；A 奖励重构/A2 win/γ 消融（`reward-shaping-500k.md`，
4-seed 未复现）；B0/B1 观测增广（`observation-augmentation-b0/b1.md`，null）；C 逐局/PFSP/强成员
（`opponent-distribution-500k.md`，三效应 null，但成员全弱）；F 双塔（`twin-towers-500k.md`，关闭）。
**不重做**。C 线「强成员」前提在 revision-3 下已变（现在有 `self_500k`/`l1_1M` 可当成员），
只做一条静态池重估（§2 A1⑤），PFSP 因归因混淆本轮不做（§3.4）。

### 1.4 统计现实：为什么主端点必须是 k=7 的固定参照族

`evaluation-protocol-validation.md` §5/§6（主表 k=3 seed × N=400）：

| 规则 | FPR | power Δ=10 | power Δ=20 |
|---|---|---|---|
| CI 排除 0 且点 ≥+10 | 0.017–0.019 | **0.26–0.31** | 0.78–0.85 |
| 扩大：k=5×400 | — | 0.44–0.50 | — |
| 扩大：k=3×800 | — | 0.45–0.57 | — |
| 单次 1500 副 | — | 0.47 | — |

即 **Δ=10 在现实预算下只有 ~0.3–0.5 power**（`README §3` 的「10 Elo ≈1500–2000 副」是 50% power
口径；80% power 需 3226）。合并半宽 = `t_{0.975,k-1}·max(deal SE, 训练 seed sd)/√k`：

| 训练 seed 数 k | t_{0.975,k-1} | z 半宽 @sd=10 | t 半宽 @sd=10 | z 半宽 @sd=16 | t 半宽 @sd=16 |
|---|---|---|---|---|---|
| 3 | 4.303 | ±11.6 | ±24.8 | ±18.6 | ±39.7 |
| 5 | 2.776 | ±8.8 | ±12.4 | ±14.0 | ±19.9 |
| **7** | **2.447** | **±7.4** | **±9.2** | **±11.9** | **±14.8** |
| 9 | 2.306 | ±6.5 | ±7.7 | ±10.5 | ±12.3 |

读法：**只有把 confirmatory 训练 seed 提到 k≥7，`+10 且 CI 排 0` 的门在 sd≤16 时才有机会达成**；
k=3 在本轮预算内数学上大概率只会得到「不可判」（红队 stats-eval-01、training-ml-01）。
同时 80% power 的可检测效应约 `2.80·SE` → sd=16、k=7 时 ≈ **+17～+20**。因此本计划**书面目标
定为 +20**（对 +20 的单侧非劣检验），并把点估计 ≥+10 作为 repo 行动门槛的下限。

---

## 2. 候选杠杆评估与排序（A1–A5）

排序原则：**证据强度 × 便宜程度 × 现有 CLI 可跑**。

### A1. self-play 变体（证据方向为正，第一优先）

**机制假设**：对手分布从固定 RandomBot 换成「自己的移动快照」，梯度不再在固定弱点上饱和；
self-play 训练出更均衡的攻防，迁移到 random 梯级上体现为净强度提升。

**已有证据**：`self_500k` 对 l1_1M 合并 9 deal seed **+16.62**（探索性）、对 l1_2M 打平
（T17 §5）。**单 seed、单配置、未发布**（§1.2）。

| 子项 | 机制/单因子 | 已有证据 | 预期 | 成本（500k 等效） | 风险 | CLI |
|---|---|---|---|---|---|---|
| **① fresh self seed 2–8**（confirmatory） | 训练 seed 复现 | 只有 seed1 | +10…+25 方向正（若真实） | **7 run = 7 等效** | seed sd 可能 ~16 → 仍不可判 | ✅ `--opponent self --self-play-refresh 10` |
| **② warm start 强 ckpt 再 self**（l1_1M→500k self） | 从更强起点进入 self 分布 | 未测 | 先验正 | 1 run | **LR 重退火 + Adam 状态重置双重混淆**，必须配对照 | ✅ `--load-checkpoint <snap> --opponent self` |
| **②对照 匹配 random 控制**（l1_1M→500k random） | 隔离 self 分布 | 无 | 衰减或持平 | 1 run | — | ✅ `--load-checkpoint <snap> --opponent random` |
| **③ 更长 pure self 1M** | 预算剂量 | 未测 | 低先验正 | 2 等效 | random 侧 1M→2M 无改进 | ✅ `--total-timesteps 1000000` |
| **④ refresh 50**（熵保持对照） | 对手新鲜度 | 旧口径噪声（`plans.md` §6 N-3） | 先验低 | 1 run | 单因子在噪声内 | ✅ `--self-play-refresh 50` |
| **⑤ 强成员静态逐局池（无 self 成员）** | 重建强成员联赛；C 的 null 因成员全弱 | C/T5 null（成员全弱于 base680k） | 先验中低 | 1 run | 归因弱 | ✅ `--opponent pool --pool-episode --pool-member 1@ckpt:self_500k --pool-member 1@ckpt:l1_1M --pool-member 1@random` |
| ⑤b PFSP 强成员池 | 按胜率重加权 | C 的 PFSP null（成员弱） | 未测 | 1 run | 与 ⑤ 混淆 | ✅ `--pfsp True`（**本轮不做**，见 §3.4 预注册命令） |

**取舍（revision-4）**：① 必做且提到 **k=7**（Stage 1）；② 与 ②对照 成对（Stage 2）；
③ 1M（Stage 2，抓 512000/999424 同 run 剂量点）；④ refresh 50（Stage 2）；⑤ 静态池（Stage 2）。
⑤b PFSP 本轮不做——⑤b 与 ⑤ 的差异只在权重，且 C 的 PFSP null 归因在旧报告里本已混淆；
删去以保预算与归因。合计 Stage 1 = 7 run、Stage 2 = 5 run。

### A2. 容量/超参（先验低但便宜，粗筛）

**机制假设**：hidden 更大 → 可表达性提升。

**已有证据**：T8 条件项；负证据（`h256_scratch` 最低、500k 六臂不可分）来自**旧规则/旧 obs/
从零**。revision-3 + v5 下**未重测**。

**CLI 可跑子集**：`--hidden-size 256/512`、`--learning-rate`、`--ent-coef`、`--clip-coef`、
`--target-kl`。`--num-envs 16` **不是单因子**：它同时改 batch（8×128=1024 → 16×128=2048）、
minibatch（`batch//num_minibatches`，256 → 512，`src/seven523/ppo.py:174`）、更新数
（`200000//1024=195` → `200000//2048=97`，`train.py:677`），并把 env 8–15 的 `seed+idx`
变成新牌流（`train.py:422`）→ **不列入本轮**（红队 training-ml-05、engineering-ops-06）。

**需要改 src（只设计，不实现）**：trunk 深度 >2、LayerNorm、Adam eps、return/value norm、
PopArt、分布价值、n-step。

**取舍**：**只做 3 个 200k 臂**（`cap128` 同预算控制 / `cap256` / `cap512`），
判定措辞限定为「**200k k=1 下无 ≥+20 信号**」，**不写「关闭 A2」**（红队 training-ml-05、
stats-eval-07）。命中（点 ≥+20 且 CI 排 0）才用 500k ×3 seed 确认，且控制臂必须同预算 500k。

### A3. 价值/信用分配（解释性，先诊断后设计）

**已有证据**：`structural-directions.md` §1.1 的早期 EV=0.065 属 revision-2 / 191 维旧 obs，
且该报告 §3 明示「本报告引用的模型 checkpoint 已删除，旧路径不再可用」。当前 ckpt 整体 EV
已是 **0.61–0.70**（`runs/t17self__1/metrics.csv`、`runs/t17pool__1/metrics.csv` 的
`explained_variance`）。→ `0.065` **不可作为目标值**（红队 training-ml-08）。

**取舍**：**只读诊断，判据改为 ckpt 内部比较**：对当前最强 ckpt 重算
「早期桶 EV vs 整体 EV vs 末桶 EV」并附 V sd / return sd，写一次性脚本到 `runs/t17w1/`
（不改 src/tools）。只有内部诊断显示「价值目标本身可改」才设计 return/value norm。

### A4. 选择与集成

- **snapshot 择优发布**：`tools/arena.py`/`build_ladder.py` 现成（纯评测），用第一波长训的多 step
  快照做候选。遵守快照选择偏差规则（§6.4）。
- **top-k ckpt 动作集成**：`NeuralPolicy` 现单模型（`networks.py:746`），无现成 ensemble →
  **只设计**。

### A5. 更强脚本对手 / 上限测量（Q-2）

需 1-ply 搜索 bot（用价值头枚举动作），**只设计**。纯评测替代：`build_ladder --anchor random`
的绝对表（本波 F0 已含）作为上限粗代理，但不测 exploitability。

### 排序总结

| 排名 | 杠杆 | 证据 | 成本 | 先验 | 阶段 |
|---|---|---|---|---|---|
| 1 | A1① self fresh seed 2–8（confirmatory） | 方向正、未确立 | 7 run | 中高 | **Stage 1** |
| 2 | A2 容量 200k 粗筛（cap256/512） | 弱（旧负） | 1.2 等效 | 低但便宜 | **Stage 1** |
| 3 | A1② warm-start self + 匹配 random 对照 | 机制推断 | 2 run | 中 | Stage 2（条件） |
| 4 | A1③ 更长 self 1M | 弱 | 2 等效 | 中低 | Stage 2（条件） |
| 5 | A1④ refresh 50 熵保持对照 | 弱（旧 null） | 1 run | 低 | Stage 2（条件） |
| 6 | A1⑤ 强成员静态池（无 self） | 弱（C null，前提已变） | 1 run | 中低 | Stage 2（条件） |
| 7 | A4 snapshot 择优 | 工具现成 | 评测 | 中 | 第二波 |
| 8 | A3 价值诊断 | 旧数不可比 | 诊断 | 低 | 第二波（只读） |
| 9 | A5 上限测量 | 无 | 新脚本 | 未知 | 后置 |
| — | A1⑤b PFSP 强成员池 | 旧 null（成员弱） | 1 run | 低 | 第二波 |

---

## 3. 第一波（可执行、预注册）

> **前置纪律**：训练 ≤3 并发（`nice -n 5`）；h2h 严格串行、一次一个、`--device cpu --workers 4`；
> 批量前 `free -h` / `nvidia-smi`。run 目录命名 `<exp_name>__<seed>__<unixtime>`；评测产物写
> `runs/t17w1/`；报告写 `docs/experiments/v5-optimization-wave1.md` +
> **`docs/experiments/README.md §1`** 索引。**不改 src/tests/tools**；若需要某个不存在的 CLI
> 开关，停下如实报告。

### 3.0 预检与目录（必须先做；红队 engineering-ops-01）

```bash
mkdir -p runs/t17w1          # 否则所有 > runs/t17w1/... 重定向都会失败
free -h && nvidia-smi        # 训练前确认内存/显存；h2h 只走 CPU
```

所有 `head_to_head` 重定向一律写 `runs/t17w1/`；若 `runs/t17w1` 不存在，先 `mkdir -p`。

### 3.1 参照基线（预注册，冻结）

| id | 路径 | 说明 |
|---|---|---|
| `self_s1` | `runs/t17self__1__1790439615/agent.pt` | 现有纯 self 500k seed1（refresh 10） |
| `rand_s1` | `runs/t17pool__1__1790439615/agent.pt` | random 500k seed1（= `p1_499k`，逐参数核对） |
| `rand_s2` | `runs/t17pool__2__1790439615/agent.pt` | random 500k seed2 |
| `rand_s3` | `runs/t17pool__3__1790439615/agent.pt` | random 500k seed3 |
| `l1_1M` | `runs/t17long__1__1790439615/snapshots/checkpoint_step1024000.pt` | 发布 lvl4（185.34）；**MLE 13 候选 argmax**（winner's curse，故加第二参照 l1_2M） |
| `l1_2M` | `runs/t17long__1__1790439615/snapshots/checkpoint_step1999872.pt` | 第二参照（MLE 191.1，与 l1_1M 平台同档） |
| `lvl3` | `runs/t17pool__1__1790439615/snapshots/checkpoint_step65536.pt` | 中档绝对锚 |

**deal seed 冻结**：本波所有 confirmatory 端点用 **fresh deal seed `10–18`**（9 个），
**不复用** T17 的 0–8（红队 stats-eval-03、training-ml-13）。探索性端点可用 `10,11,12`。

### 3.2 训练臂精确定义

#### Stage 1（confirmatory，无条件）

**Block A — fresh self seeds 2–8（7 run，500k，训练 seed = 2…8）**

```bash
for s in 2 3 4 5 6 7 8; do
  nice -n 5 .venv/bin/python -m seven523.train --exp-name t17w1self --seed "$s" \
    --total-timesteps 500000 --opponent self --self-play-refresh 10 \
    --cuda True --tensorboard False --run-dir runs
done
```
产物：`runs/t17w1self__{2..8}__<ts>/agent.pt`。镜像 `t17self` 的 `args.json`
（`--opponent self` 下 `--mix-random-prob` 被忽略）。调度：每批 ≤3 进程；self 单 run 500k ≈
13–26 min，7 run 约 2 批 ≈ 35–60 min（3 并发 ~321 sps → ~26 min/run）。

**Block C — A2 容量 200k 粗筛（3 run，seed 1，独立小预算）**

```bash
for spec in "t17w1cap128 --hidden-size 128" \
            "t17w1cap256 --hidden-size 256" \
            "t17w1cap512 --hidden-size 512"; do
  set -- $spec
  nice -n 5 .venv/bin/python -m seven523.train --exp-name "$1" --seed 1 \
    --total-timesteps 200000 --opponent random "$2" "$3" \
    --cuda True --tensorboard False --run-dir runs
done
```
`cap128` 是同预算 200k 控制。**只与 `cap128` 比**，不跨预算与 `l1_*` 比。

#### Stage 2（条件：仅当 M1 的 F1b 点估计 ≥ +5 才跑）

**Block B — self-play 变体（5 run，seed 1，探索性）**

```bash
# A1② warm start → self（l1_1M → 500k self）
nice -n 5 .venv/bin/python -m seven523.train --exp-name t17w1ws --seed 1 \
  --total-timesteps 500000 --opponent self --self-play-refresh 10 \
  --load-checkpoint runs/t17long__1__1790439615/snapshots/checkpoint_step1024000.pt \
  --cuda True --tensorboard False --run-dir runs

# A1② 匹配对照（同 warm start、同退火、同优化器重置；对手 random）
# —— 这是 (ws − 对照) 归因 self 分布所必需的（红队 stats-eval-08 / training-ml-04 / engineering-ops-09）
nice -n 5 .venv/bin/python -m seven523.train --exp-name t17w1wsctrl --seed 1 \
  --total-timesteps 500000 --opponent random \
  --load-checkpoint runs/t17long__1__1790439615/snapshots/checkpoint_step1024000.pt \
  --cuda True --tensorboard False --run-dir runs

# A1③ 更长 pure self 1M（同 run 抓 512000 / 999424 剂量点）
nice -n 5 .venv/bin/python -m seven523.train --exp-name t17w1selflong --seed 1 \
  --total-timesteps 1000000 --opponent self --self-play-refresh 10 \
  --checkpoint-interval 500 --cuda True --tensorboard False --run-dir runs
# 训练启动后【后台】起轮询器（否则前台阻塞、且 999424 只能从 agent.pt 抓；红队 engineering-ops-08）：
nohup .venv/bin/python runs/t17_train/poll_snapshot.py \
  runs/t17w1selflong__1__<ts> 512000 999424 > runs/t17w1/poll_selflong.log 2>&1 &

# A1④ 单因子：refresh 50（熵保持对照）
nice -n 5 .venv/bin/python -m seven523.train --exp-name t17w1refresh50 --seed 1 \
  --total-timesteps 500000 --opponent self --self-play-refresh 50 \
  --cuda True --tensorboard False --run-dir runs

# A1⑤ 强成员静态逐局池（无 self 成员；重建 C 线强成员前提）
nice -n 5 .venv/bin/python -m seven523.train --exp-name t17w1pool --seed 1 \
  --total-timesteps 500000 --opponent pool --pool-episode True \
  --pool-member 1@ckpt:runs/t17self__1__1790439615/agent.pt \
  --pool-member 1@ckpt:runs/t17long__1__1790439615/snapshots/checkpoint_step1024000.pt \
  --pool-member 1@random \
  --cuda True --tensorboard False --run-dir runs
```
产物：`runs/t17w1ws__1__<ts>/`、`t17w1wsctrl__1__<ts>/`、
`t17w1selflong__1__<ts>/{agent.pt,snapshots/}`、`t17w1refresh50__1__<ts>/`、`t17w1pool__1__<ts>/`。

> **A1⑤ 范围声明（红队 training-ml-07 / engineering-ops-07）**：本波 A1⑤ = **静态均匀池**，
> 池中**不含 `self` 成员**，回答「强 frozen 成员重估」；PFSP 的重加权半条线**不在本轮**
> （预算与归因），留第二波（§4），命令：
> `--opponent pool --pool-episode --pfsp True --pfsp-every 100 --pool-member 1@ckpt:self_500k …`。
> `--pfsp` 需同时 `--opponent pool --pool-episode`（`train.py:477-486` 校验）。

### 3.3 评测对照（严格串行，一次一个；`--device cpu --workers 4`）

全部 `--pairs 400 --bootstrap 4000 --device cpu --workers 4`，输出 `runs/t17w1/h2h_*.json`；
**每个 h2h 跑完再起下一个**。confirmatory 端点 deal seed = `10,11,12,13,14,15,16,17,18`（9 个）；
探索性端点 = `10,11,12`。

**F0 — 绝对强度锚（共享 fit，vs 脚本 RandomBot；红队 training-ml-06）**

```bash
.venv/bin/python tools/build_ladder.py \
  --candidate l1_1M=ckpt:runs/t17long__1__1790439615/snapshots/checkpoint_step1024000.pt \
  --candidate self_s2=ckpt:runs/t17w1self__2__<ts>/agent.pt \
  --candidate lvl3=ckpt:runs/t17pool__1__1790439615/snapshots/checkpoint_step65536.pt \
  --anchor random=random --games-per-anchor 400 --cross 400 \
  --no-traces --games-out runs/t17w1/f0_games.jsonl \
  --study /tmp/t17w1_f0 --seed 0 --device cpu --workers 4
.venv/bin/python tools/refit_mle.py --games runs/t17w1/f0_games.jsonl \
  --bootstrap 200 --json --out runs/t17w1/f0_absolute_table.json
```
读法：同一 fit 内 `μ(self_s2) − μ(l1_1M)` 与各自对 random 的胜率（`lvl4>random strict 0.8825`
可作历史参照）。**预注册门**：若 `μ(self_s2) − μ(l1_1M) < −10`，则即使 F1b 为正，self-play
线也**不得升级**（落入「只更会打 random-trained MLP」）。

**F1 — confirmatory 主族**

- **F1b（主端点）**：`self_sN` vs `l1_1M`，N∈{2,…,8}（7 个训练 seed），9 个 fresh deal seed。
```bash
for n in 2 3 4 5 6 7 8; do
  .venv/bin/python tools/head_to_head.py \
    --left "self_s${n}=ckpt:runs/t17w1self__${n}__<ts>/agent.pt" \
    --right l1_1M=ckpt:runs/t17long__1__1790439615/snapshots/checkpoint_step1024000.pt \
    --seeds 10,11,12,13,14,15,16,17,18 --pairs 400 --bootstrap 4000 \
    --device cpu --workers 4 --json > "runs/t17w1/h2h_selfVS1M_s${n}.json"
done
```
- **F1c（描述性，winner's-curse 检查）**：`self_s2` 与 `self_s8` vs `l1_2M`，9 deal seed（2 次）。

**F1a（描述性，不进 confirmatory 族）**：`self_sN` vs `rand_sN`，N∈{1,2,3}，9 deal seed
（random 侧只有 3 个 ckpt，无法做到 k≥5 → 只作方向性参照；红队 training-ml-02）。
其中 N=1 的 +27.9 已有（deal 0–8），此处用 fresh deal 10–18 **重测**，历史读数只作注脚。

**F2（探索性，仅在 Stage 2 跑；seed 1，3 个 deal seed，描述性）**

- Block B 每个臂 vs `l1_1M` 与 vs `self_s1`（5 臂 × 2 = 10 次 h2h）。
- `t17w1selflong` 剂量：`selflong@999424 vs selflong@512000`（**同 run**，干净）+
  `selflong@999424 vs self_s1`（**跨 run**，注明确含 run-to-run 方差；红队 engineering-ops-03）。
- Block C：`cap256 vs cap128`、`cap512 vs cap128`（2 次 h2h）。

**合并脚本 `runs/t17w1/aggregate.py`（eval-only，写 `runs/`，允许；不改 src/tools）**

规格（红队 stats-eval-05 / training-ml-12 / engineering-ops-05）：
- 输入：`runs/t17w1/h2h_<pair>_sN.json`（显式路径，**不要照抄 `aggregate_seq.py` 的
  `runs/<prefix>_s{seed}.json`**）。
- 每训练 seed 的 `combined.elo_diff` 已由 `duel.py:255-273` 用 **RMS** 合并 deal seed。
- 跨训练 seed：`mean ± q·max(RMS(per-seed combined.se), training_seed_sd)/√k`；
  **同时输出 `q=1.96` 与 `q=t_{0.975,k-1}`**（k=7→2.447），并打印 `per-seed boot se`（RMS 与
  算术均值两者）、`training seed sd`、`deal-seed sd`、`binding term`。禁止只用算术均值
  bootstrap SE（会低估；`evaluate-protocol-validation.md` §6 指出 `max()` 约 +10% 保守）。
- 输出族别标签（confirmatory/descriptive）。

### 3.4 判定门与停/继续规则（预注册，可执行）

设 F1b 合并（k=7）点估计 Δ、z-CI、t-CI；**binding term** 见上。

| 条件 | 判定 | 动作 |
|---|---|---|
| **Δ ≥ +10 且 z-CI 与 t-CI 均排除 0** | self-play 确认为杠杆 | 报告升级；进入 A4「择优发布」；对 +20 另做单侧非劣检验（`plans.md §7 C-1`） |
| **Δ > +5 且 CI（z 或 t）跨 0** | 过 5 分线、未达行动门槛 | **按 binding term 升级**：若 `training_seed_sd > RMS boot` → 加训练 seed 9/10（≤2 run，k=9, t=2.306）；否则先把 deal seed 加到 1500/seed（评测，分钟级）再复算；仍不达标则报「无 ≥+10 证据」 |
| **Δ ≤ +5**（含负） | 无证据 | **止损**：停 self 线扩展，不跑 Stage 2 的 self 变体；转 A2 命中确认 / A3 诊断 |
| F1b **Δ ≥ +10 但 F0 显示 `μ(self)−μ(l1_1M) ≤ −10`** | 相对提升、绝对退化 | 不升级；报告「self 只更会打 MLP 分布」 |
| F1c 与 F1b **反向**（vs l1_2M 明显更弱） | winner's-curse/参照敏感 | 只降级为方向性提示，不加 seed |
| Block C 任一臂 **点 ≥+20 且 CI 排 0** vs `cap128` | 容量命中 | 500k ×3 seed 确认（≤3 run），且控制臂也训 500k（预算匹配） |
| Block C 全部 **< +20 或 CI 跨 0** | 200k 无 ≥+20 信号 | 写明「**不构成关闭 A2 的证据**」，保留 hidden>128 待测 |
| F2 任一臂点 ≥+30 且 CI 排 0（k=1 半宽 ≈±21.6，+20 不可达） | 仅**方向性提示** | **不自动**触发确认；只作为第二波的候选假设（红队 stats-eval-06） |

**先出结论的里程碑**：
- **M1（最高信息量）**：Block A（7 run）训完 + F1b（7 次 h2h）+ F0 评测完 → 回答
  「self-play 在 k=7 fresh 训练 seed 上是否稳定优于发布 lvl4，且是否对 RandomBot 不退化」。
  **M1 先出结论，再决定是否跑 Stage 2。**
- **M2**：Block C 粗筛完 → 容量方向性提示。
- **M3**：Stage 2 全部 h2h 完 → self-play 变体方向性结论。

**止损**：M1 的 F1b 点估计 ≤+5（含负）→ 立即停 self 线，不跑 Block B 的 self 变体与 1M/2M；
报告如实写 null。

### 3.5 条件 Block D（触发才跑，不得即兴）

- self seed 9/10：F1b 落「Δ≥+5 且 CI 跨 0」且 `training_seed_sd` 为 binding term（≤2 run）。
- deal seed 扩到 1500/seed：仅当 RMS boot term 为 binding term（纯评测）。
- A2 命中确认：Block C 命中时（≤3 run，500k，控制臂同预算）。
- pure self 2M：仅当 `selflong@999424` **显著优于 `selflong@512000`**（同 run Δ≥+10 且 CI 排 0）
  才考虑（4 等效 → **申请下一批**）。
- 任一条件命中先记入本波报告，**预算分期**，不即兴开超 15 等效的批次。

### 3.6 阶段与顺序（Wave-1 执行顺序）

```text
mkdir -p runs/t17w1
Stage 1: Block A（7 self run，≤3 并发）+ Block C（3×200k，穿插） 
         → F0 → F1b →（F1a/F1c 可并行串行插空）→ M1
Stage 2（仅 M1 点估计 ≥+5）: Block B（5 run）→ F2 → M3
报告: docs/experiments/v5-optimization-wave1.md + docs/experiments/README.md §1 索引
```

### 3.7 预算核算（修正口径；红队 engineering-ops-04）

| 块 | run | 等效（500k=1，200k=0.4） |
|---|---|---|
| Block A（self seed 2–8） | 7 × 500k | 7.0 |
| Block B（ws / wsctrl / refresh50 / pool / selflong 1M） | 4×500k + 1×1M | 1+1+1+1+2 = 6.0 |
| Block C（cap128/256/512） | 3 × 200k | 3 × 0.4 = 1.2 |
| **合计** | 15 run | **14.2 ≤ 15** |

Stage 1 占用 = 7 + 1.2 = **8.2 等效**；Stage 2 条件占用 = **6.0 等效**；总计 14.2。
（原 revision-3 把 `selflong 1M` 记 1 等效、把 200k 记 0.2 等效，均系笔误。）

---

## 4. 后续波（前置条件与预算）

### 第二波

1. **A3 只读价值诊断**（无训练）：`runs/t17w1/` 一次性脚本，按 §2 A3 的内部判据
   （早期桶 EV / 整体 EV / 末桶 EV + V sd / return sd）。前置：第一波产出「当前最强」ckpt。
2. **A4 snapshot 择优 + 绝对表**：`tools/arena.py` 或 `tools/build_ladder.py`，
   命令**必须显式** `--device cpu --workers 4`（红队 engineering-ops-10）：
   `.venv/bin/python tools/arena.py --glob 'runs/t17w1selflong__1__*/snapshots/checkpoint_step*.pt' --device cpu --workers 4 ...`
   `.venv/bin/python tools/build_ladder.py --candidate ... --anchor random=random --no-traces --games-out runs/t17w1/abs.jsonl --device cpu --workers 4 ...`
   遵守快照选择偏差（§6.4）。
3. **A1⑤b PFSP 强成员池**（本轮未做）：`--opponent pool --pool-episode --pfsp True --pfsp-every 100
   --pfsp-uniform-mix 0.5 --pool-member 1@ckpt:self_500k --pool-member 1@ckpt:l1_1M --pool-member 1@random`；
   与静态池 ⑤ 预注册为成对对比（干净归因）。
4. **A4 top-k 动作集成（设计）**：新增 policy 包装器（平均 masked logits / 逐 seat 集成）；
   需 ckpt 列表、推理 ×k、`NeuralPolicy` 多模型变体。本工作流不实现。
5. **A5 上限测量（设计）**：1-ply 搜索 bot（`Agent.get_value` 枚举合法动作）测对 RandomBot 的
   可被利用空间（`plans.md §9 Q-2`）。前置：A3 诊断显示价值头可用。

### 第三波（条件）

- A2 剩余超参（lr/gae_lambda/ent_coef/clip/target_kl 单因子 ×3 seed）：仅当 self 与容量都 null、
  且需排除「优化受限」时；**gamma 期望效应很小（§0.3.7）不单列**。
- 在 self-play 分布下重估已关闭的 B/A 线：**只在 self 确认后**才值得。
- 若 self 确认且 A4 集成有正信号 → 组合。

---

## 5. 实现面清单

**纯 CLI，第一波可直接跑（无需改码）**：
- A1①②③④⑤：`--opponent self|pool`、`--self-play-refresh`、`--total-timesteps`、
  `--load-checkpoint`、`--pool-episode`、`--pool-member`。
- A2 CLI 子集：`--hidden-size`、`--learning-rate`、`--ent-coef`、`--clip-coef`、`--target-kl`。
- 评测：`tools/head_to_head.py`、`tools/build_ladder.py`、`tools/refit_mle.py`、
  `tools/arena.py`、`runs/t17_train/poll_snapshot.py`（后台）。
- 训练 seed 合并：`runs/t17w1/aggregate.py`（写 `runs/`，允许；规格见 §3.3）。

**需要改 src/tests/tools（本工作流只设计，不实现）**：

| 项 | 落点 | 契约 |
|---|---|---|
| A2 trunk 深度 >2 / LayerNorm | `networks.py:Agent.__init__`（新 CLI，默认路径逐位不变） | 默认 hidden=128/2 层逐位不变；ADR-0003 |
| A2 Adam eps / return-value norm / PopArt | `train.py`（optimizer eps）、`ppo.py`（value target 归一） | 默认逐位不变；新测试 |
| A3 更大 critic / 分布价值 / n-step | `networks.py` critic 头、`ppo.py` GAE | 默认逐位不变 |
| A4 top-k 动作集成 | 新 `src/seven523/ensemble.py` 或 `networks.py` 多模型变体 | 推理 ×k；评测链零改动路径 |
| A5 1-ply 搜索 bot | 新 `tools/`/`src/` bot；用 `Agent.get_value` | 只读评估，不进训练默认 |

**发布口径**：把新模型登记为梯级需改 `tools/play_ladder.py::OPPONENTS` +
`tests/test_play_ladder.py::EXPECTED_IDS`（`play_ladder.py:96`、`test_play_ladder.py:33/76`），
**属 Wave-1 禁止的 tools/tests 改动**。第一波只产出 h2h 证据与 `runs/` 内绝对表；若结论要求
改梯级，**停下如实报告**，由后续授权执行。

---

## 6. 统计与评测纪律（revision-4 修正）

1. **预注册端点族**：**confirmatory = F1b only**（k=7 fresh 训练 seed × 9 fresh deal seed × 400 副）。
   F0 为共享 fit 绝对锚（单端点，不与 F1b 合并）。**F1a / F1c / F2 全为描述性**，不承载行动声明。
   报告同时给未校正值；若对 confirmatory 加做 +20 非劣检验，Bonferroni/Holm 覆盖 F1b 与 +20 两个
   检验（α=0.025）。
2. **合并公式**：`mean ± q·max(RMS(per-seed combined.se), training_seed_sd)/√k`；
   同时报 `q=1.96` 与 `q=t_{0.975,k-1}`（k=7→2.447，k=9→2.306）。打印三个分量与 binding term。
3. **功效预注册**：本设计 k=7、9 deal seed；80% power 可检测效应 ≈ +17…+20（§1.4），
   故**书面目标 = +20**（对 +20 单侧非劣），`+10` 为 repo 行动门槛下限。若 M1 时
   `training_seed_sd > RMS boot`，加 seed（k=9）优先于加 deal；反之加 deal。
4. **CI 跨 0 的正确读法**：写死「**CI 跨 0 = 未判定 ≠ 效应为 0**」；报告不得把未判定写成 null。
5. **快照选择偏差**：长训中间快照只作描述性；预注册最终 step 为主端点；中间快照若胜出需在
   **未用于选择的 deal seed（10–18 之外的 fresh 集）或新训练 seed** 上确认。
6. **独立 deal seed**：confirmatory 用 fresh 10–18，不复用 0–8；Bonferroni 的独立单元按
   「独立 deal 集 + 独立训练 seed」计。
7. **旧口径隔离**：全部产物带 `rules_id=2e36dbea44893696`；跨 fit 绝对 Elo 不可比；
   绝对强度只用同一 `build_ladder` 共享 fit。
8. **过拟合/泄漏检查**：F0 的 vs-RandomBot 锚是防「只对镜像变强」的门；报告每个新臂的
   `entropy`、`episodic_return` 均值/sd、`explained_variance`（环境变量观测，红队 training-ml-10）。
   **熵门（预注册）**：若任一 self 臂末熵 < 1.6 且 vs RandomBot 相对 l1_1M 退化，不升级该臂。

---

## 7. 分步实施、验收、回滚、风险/未决

### 7.1 分步（Wave-1 执行顺序）

1. `mkdir -p runs/t17w1`；`free -h` + `nvidia-smi`；确认 ≤3 训练并发。
2. **Stage 1**：启动 Block A（7 run）+ Block C（3×200k），交错调度 ≤3 并发。
   `selflong` 属 Stage 2，本阶段不起。
3. Block A/C 训完后，**逐个（串行）**跑 F0 → F1b（7 次）→ F1a/F1c；每次
   `--device cpu --workers 4`，写完一个 JSON 再起下一个。
4. 用 `runs/t17w1/aggregate.py` 合并 F1b，先出 **M1**；按 §3.4 门决定是否进 Stage 2。
5. 若进 Stage 2：起 Block B（5 run，≤3 并发），selflong 起后台轮询器；训完跑 F2（串行）。
6. 报告 `docs/experiments/v5-optimization-wave1.md` + `docs/experiments/README.md §1` 索引（允许）。

### 7.2 验收（Wave-1 done 的定义）

- 所有预注册 run 训到目标 step、无 NaN（`metrics.csv` 末行）；args.json 与命令一致。
- F1b 有 k=7 合并，**z 与 t 两口径**都给出，标注 confirmatory/描述性；打印三个方差分量。
- F0 绝对表产出（shared fit，含 vs-random 胜率）。
- 每条停止/继续门有明确结论；**CI 跨 0 写「未判定」**。
- 产物路径完整（`runs/t17w1self*/`、`runs/t17w1/`）；**未改** `src/tests/tools`，否则停下报告。

### 7.3 回滚

所有新产物在 `runs/`（gitignore）；回滚 = 删除 `runs/t17w1*/`、`runs/t17w1/`；
`traces/study` 与 manifest **不触碰**（本波不发 manifest）。报告为追加式，不改旧报告。

### 7.4 风险与未决问题

| # | 风险/未决 | 影响 | 缓解 |
|---|---|---|---|
| R1 | 训练 seed sd 可能 >16，k=7 仍不可判 | M1 得「未判定」 | 预注册 binding-term 升级；不声称 0；目标定 +20 |
| R2 | warm start 重置 LR 退火**并重置 Adam 状态**（`train.py:557` 新建 Adam eps=1e-5、`train.py:706` 重算 frac） | ② 增益不可归因 self | **② 必须与 ②对照（同 warm start、random 对手）成对**；归因改判「ws − wsctrl」 |
| R3 | self-play 对镜像强、对 random/真人不强 | 迁移风险 | F0 vs-RandomBot 绝对锚 + F1a 描述性 |
| R4 | A1⑤ 静态池归因弱 | 归因弱 | 只作探索/描述性，不作行动声明 |
| R5 | Block C 200k 效应可能在 500k 消失 | 假阳/假阴 | 命中必须 500k ×3 seed 确认；措辞限定「200k 无 ≥+20」 |
| R6 | 快照择优选择偏差 | 高估 | §6.5 规则；主端点用最终 step |
| R7 | 发布梯级需改 `play_ladder.py`/tests | Wave-1 越界 | **停下报告**，不即兴改码 |
| R8 | 训练并发/内存 | 卡死事故 | ≤3 训练；h2h 串行 CPU workers=4；批量前检查；`nohup ... &` 轮询器 |
| R9 | h2h CPU 墙钟 | 预计 | 1200 deal ≈ 2–3 min（cpu/4）；9×400=3600 deal ≈ 6–10 min；F1b 7 次 ≈ 45–70 min；F0 ≈ 10–15 min |
| R10 | `selflong 999424` 只能从 `agent.pt` 抓（`--checkpoint-interval 500` 只会落 512000） | 抓不到 | 轮询器保持到 run 结束（后台）；见 §7.1/§3.2 |

### 7.5 长程（若第一波 self 确认）

- 系统扫 self 变体（refresh/采样/池成员/预算）→ 选 1 个发布候选 → 3–5 seed 确认 →
  `build_ladder`/`refit_mle` 重标定（需授权改 `tools/play_ladder.py`）→ 更新
  `prior.json`/human-play 强度列。
- 若 self 未确认：转 A3 只读诊断 + A5 上限测量，判断「优化受限 vs 上界受限」；
  再决定是否在 self 分布下重估 B/A 线。
- 全程遵守 ADR-0003（默认路径逐位不变）、ADR-0009（v5 唯一布局）。

---

## 附：命令速查（勿在 Plan/Repair 阶段执行）

```bash
mkdir -p runs/t17w1

# 训练（示例：confirmatory self seed）
nice -n 5 .venv/bin/python -m seven523.train --exp-name t17w1self --seed 2 \
  --total-timesteps 500000 --opponent self --self-play-refresh 10 \
  --cuda True --tensorboard False --run-dir runs

# h2h（严格串行；cpu/workers=4；fresh deal seed 10-18）
.venv/bin/python tools/head_to_head.py \
  --left self_s2=ckpt:runs/t17w1self__2__<ts>/agent.pt \
  --right l1_1M=ckpt:runs/t17long__1__1790439615/snapshots/checkpoint_step1024000.pt \
  --seeds 10,11,12,13,14,15,16,17,18 --pairs 400 --bootstrap 4000 \
  --device cpu --workers 4 --json > runs/t17w1/h2h_selfVS1M_s2.json

# 绝对锚（共享 fit）
.venv/bin/python tools/build_ladder.py \
  --candidate l1_1M=ckpt:runs/t17long__1__1790439615/snapshots/checkpoint_step1024000.pt \
  --candidate self_s2=ckpt:runs/t17w1self__2__<ts>/agent.pt \
  --candidate lvl3=ckpt:runs/t17pool__1__1790439615/snapshots/checkpoint_step65536.pt \
  --anchor random=random --games-per-anchor 400 --cross 400 --no-traces \
  --games-out runs/t17w1/f0_games.jsonl --study /tmp/t17w1_f0 --seed 0 \
  --device cpu --workers 4

# 合并（wave1 内自建，写 runs/）
.venv/bin/python runs/t17w1/aggregate.py --pair selfVS1M --seeds s2,s3,s4,s5,s6,s7,s8
```

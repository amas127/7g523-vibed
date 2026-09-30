# self-play 退化机制 · 离线诊断（D0–D5，探索性为主）

> **状态：已完成（2026-09-27，现行结论）——D0 门未过 → 池臂结论限定为「弱成员池重估」；D1–D5 探索性读数：H-cycle 不可判、H-mirror 未检出、H-det 弱旁证。**
>
> **角色**：Execute agent。本报告对应 [`selfplay-pool-plan.md`](../selfplay-pool-plan.md)
> revision-2（第二轮 repair 后）的「离线诊断」部分，修正来自
> [`selfplay-pool-review.md`](../selfplay-pool-review.md)。
> **未改 `src/`、`tests/`、`tools/`、`ADR/`、`plans.md`、`README`、`CONTEXT`、`DESIGN`**；
> 未 commit/checkout/stash；诊断脚本仅落在 `runs/selfpool/`（允许）。
>
> **口径标签（硬性）**：全部为 revision-3（出空即撬底，`rules_id=2e36dbea44893696`）、
> obs v5（2 家 161 维）、纯 MLP（`arch=shared`、`hidden=128`、单层 trunk）。**跨 fit 绝对 Elo 不可比**。
>
> **诚实边界（plan §0.6 / §1.3-5）**：现有全部强资产都是 **random-trained MLP**
> （`GreedyBot` 已移除），没有中立的异族 held-out 对手。本报告的所有机制结论**只对
> "random-trained MLP 家族内的对手身份/多样性"成立**，**不得**写成"self-play 本身退化"。

---

## 0. 结论先行

| 诊断 | 标签 | 结论 |
|---|---|---|
| **D0 成员强度非劣门** | **门（半确认）** | **不过门**：`p2_500k`/`p3_500k` 对同预算代理 `p1_499k` 非劣（Δbudget CI 下界 > −5），但对标尺锚 `l1_1M` **未达** Δanchor 下界 > −10（p2 −18.0、p3 −24.1）→ 池臂结论**降级为"弱成员池重估"**，不得宣称检验了强成员池。 |
| **D1 终点 round-robin** | 探索 | 12 候选 × 400 cross + 400 anchor = **31,200 局**；镜像校准残差 **−0.001**（同参 argmax 孪生）；扫描 **220** 个三元组、无 FWER，检出 **13** 个点估计成环，最强三元组三边最小 Elo 边差 **33.3 < 34**（D1 的 MDE）→ **升级条件不触发**；**H-cycle 不可判**（零功效）。C0 子竞技场 vs 全量排名 Spearman **0.70**。 |
| **D2 族交互 δ** | 探索 | δ = **−0.0065 [−0.050, +0.032]**，CI 含 0（p=0.76）→ **无"同 regime 风格重叠"证据**。 |
| **D4 镜像残差 MA** | 探索 | MA_self 均值 **+0.001 [−0.0105, +0.0128]**、placebo MA_X 均值 **+0.0022 [−0.009, +0.0127]**，**均含 0** → 无证据。D2/D4 同批对局，**合并为一条证据**。 |
| **D5 终点确定化** | 探索 | 500k 同架构匹配：self 族（`S7`, n=7）模板头 masked entropy **0.2145±0.0251** vs random 族（p1/p2/p3）**0.3257±0.0361**，差 **−0.111 [−0.121,−0.103]**（episode-clustered bootstrap，排 0）→ self 训练在**模板头**上更确定化（描述性弱旁证）。 |
| **D3 熵 vs 迁移** | 探索 | ρ = **−0.143**、精确置换 p = **0.783**（n=7）→ **n=7 功效不足、不可判**。 |

**对 H1/H1'/H2（confirmatory）**：诊断不承载判定，待池化训练 + E1/E2/E3 端点
（见 [`selfplay-pool-wave1.md`](./selfplay-pool-wave1.md)）。**D0 未过门**已把该报告的
池臂结论限定为"弱成员池重估"。

---

## 1. 执行前核对（plan 要求，逐项现场验证）

- **代码冻结**：`git rev-parse HEAD = ce22681af7162f75e1d40a0d701454d11640e985`；`src/tests/tools`
  在诊断期间的 `sha256` 与 `git diff --stat` 记于 `runs/selfpool/freeze_record.txt`。
  全程未改上述目录（`git status` 仅新增允许的 `docs/` 与 `runs/`）。
- **CLI 开关**：`tools/build_ladder.py`（`--candidate/--anchor/--games-per-anchor/--cross/
  --no-traces/--study/--games-out/--seed/--device/--workers`）、`tools/refit_mle.py`
  （`--games/--bootstrap/--json/--out`）、`tools/arena.py`（`--entrant/--anchor/--games-per-anchor/
  --cross/--seed/--device/--workers/--out/--games-out/--tb`）均存在且语义与规划一致
  （逐行核对 `src/seven523/league.py`、`train.py`、`arena.py`、`ladder.py`）。
- **ckpt 存在且 v5/161**：`self_s2..s8`、`p1/p2/p3`、`l1_1M` 逐一 `torch.load` 核对
  `obs_dim=161`、`obs_version=5`、`arch=shared`（plan §1.1 表一致）。
- **目录无冲突**：诊断前无 `runs/selfpool/`、无 `runs/t18*`。
- **批量前体检**：`free -h` 可用 9.7 GiB；`nvidia-smi` RTX 4060 8 GB、空闲；无残留
  `seven523.train`/h2h 进程。
- **deal seed 台账**（plan §3.6）：D0 `39`、D1 base `40`、D1 确认 `140–148`、E1 `100–108`、
  E2 `110–118`、E3 `120–128`、描述性 `130–138`；已占用 `0–8/10–18/29–37` 未复用。

## 2. D0 —— 成员强度非劣检验（门）

产物：`runs/selfpool/d0_games.jsonl`（**4000 行，断言通过**）、`d0_ladder.out`、
`d0_absolute_table.json`、`d0_diff.json`。

命令（严格照 plan，先删旧行）：

```bash
rm -f runs/selfpool/d0_games.jsonl
.venv/bin/python tools/build_ladder.py \
  --candidate p2_500k=ckpt:runs/t17pool__2__1790439615/agent.pt \
  --candidate p3_500k=ckpt:runs/t17pool__3__1790439615/agent.pt \
  --candidate p1_499k=ckpt:runs/t17pool__1__1790439615/agent.pt \
  --candidate l1_1M=ckpt:runs/t17long__1__1790439615/snapshots/checkpoint_step1024000.pt \
  --anchor random=random --games-per-anchor 400 --cross 400 --no-traces \
  --study runs/selfpool/d0_study --games-out runs/selfpool/d0_games.jsonl \
  --seed 39 --device cpu --workers 4
test "$(wc -l < runs/selfpool/d0_games.jsonl)" -eq 4000
.venv/bin/python tools/refit_mle.py --games runs/selfpool/d0_games.jsonl \
  --bootstrap 200 --json --out runs/selfpool/d0_absolute_table.json
.venv/bin/python runs/selfpool/d0_diff.py --games runs/selfpool/d0_games.jsonl \
  --bootstrap 200 --seed 0 --out runs/selfpool/d0_diff.json
```

`build_ladder` 的 OpenSkill/PL 表（**仅信息性**，plan §3 D0；wall 15.1 s）：

| id | PL μ ± σ (info only) |
|---|---|
| p2_500k | 327.1 ± 35.2 |
| p3_500k | 366.0 ± 35.2 |
| p1_499k | 269.0 ± 35.3 |
| l1_1M | 397.6 ± 35.3 |

**判定只用的 probit-MLE**（`refit_mle`，deal-clustered bootstrap，B=200，seed=0，400 deals）：

| id | μ | 95% CI | n |
|---|---|---|---|
| l1_1M | 196.21 | [181.96, 213.09] | 1600 |
| p2_500k | 187.78 | [173.84, 203.69] | 1600 |
| p3_500k | 183.08 | [170.75, 200.37] | 1600 |
| p1_499k | 177.31 | [163.40, 193.58] | 1600 |

**差额 bootstrap（判定门）**：

| 成员 m | Δbudget=m−p1_499k | pass(−5) | Δanchor=m−l1_1M | pass(−10) |
|---|---|---|---|---|
| p2_500k | **+10.46** [−0.54, +19.44] | ✅ | **−8.42** [−18.01, +2.05] | ❌ |
| p3_500k | **+5.76** [−4.77, +16.53] | ✅ | **−13.13** [−24.14, −0.85] | ❌ |

→ **`gate_pass = false`**。按 plan §3 D0 / §6.1 step 5：**池臂结论降级为"弱成员池重估"**。
（`p2` 距 −10 阈值仅 1.6 Elo、CI 含 0；`p3` 明显低于 `l1_1M`。两成员都不是"更强"的池成员，
只是同预算下与 `p1_499k` 可比。）

## 3. D1 —— 终点群体 round-robin（探索性）

产物：`runs/selfpool/arena_core.json`、`arena_core_games/shard_*.jsonl`（**31,200 行**）、
`arena_core_tb/`、`d1_analysis.json`、`d1_rank.json`。

命令：`tools/arena.py`，12 候选（`self_s2..s8` + `self_s2_mirror` + `p1/p2/p3` + `l1_1M`）
+ `random` 锚，`--games-per-anchor 400 --cross 400 --seed 40 --device cpu --workers 4`，
wall 97.3 s（320.8 games/s）。

PL 排名（信息性，`arena_core.json`）：

```
self_s3 378.1 | self_s7 348.8 | self_s4 343.4 | p1 341.1 | p3 334.2 |
self_s2 326.8 | self_s2_mirror 326.4 | p2 313.5 | self_s5 307.8 |
self_s6 276.3 | l1_1M 272.7 | self_s8 265.4 | random 0.0
```

**功效预注册（plan §3 D1）**：每对 400 局（200 副 × 换座 2 局），winrate SE ≈ 0.025，
95% 半宽 ≈ ±0.049 ≈ ±34 Elo。**D1 只能可靠检出 ≥ ~34 Elo 的边差**；self 族真实 pairwise
差仅 ~5–15 Elo → **对真实循环零功效**。

1. **镜像校准**：`self_s2 | self_s2_mirror` observed **0.500**，expected **0.5010**，残差
   **−0.00104**。deal-clustered bootstrap 退化为 `[0.5, 0.5]`——因为两 id 是同一 ckpt 的
   argmax，逐副牌换座后每副恰好一胜一负，任何重采样都得 0.5。**点残差量级 ~1e-3 ≪ 噪声**，
   视为校准通过；退化 CI 是确定性孪生的统计假象，**不是**残差估计有偏。D2/D4 不作废。
2. **最大残差表**：top 全部是 `random` 锚局（观测远高于 PL 模型期望，argmax 模型对
   RandomBot 的胜率 > 模型预测），最大 `random vs self_s3` obs 0.100 / exp 0.005 / z +19.6。
   `z` 是 `arena.py:250` 明写的**未校准 smoke test**，不作检验阈值。
3. **循环筛查**：扫描 **220** 个三元组（无 FWER 控制，仅探索），点估计成环 **13** 个。最强
   三元组 `{self_s5, self_s8, p1}`，三边 Elo 边差 `[+42.5, −75.7, −33.3]`，**最小 |边差| 33.3
   < 34 Elo（D1 MDE）** → **升级条件不触发**，故不做单三元组 confirmatory 确认。
   → 按 plan：**"D1 功效不足，H-cycle 不可判"**，**不写"无循环"**。
4. **排序稳定性**（探索）：仅用 D1 的 C0 子集对局（6000 局）重拟合锚定 MLE，与全量 arena
   对 C0 的排序 Spearman = **0.70**（n=5）。

## 4. D2/D4 —— 族交互 δ 与镜像残差 MA（探索性，合并为一条证据）

产物：`runs/selfpool/d2_fit.json`（脚本 `runs/selfpool/fit_family.py`，wall 165.6 s）。
输入：D1 cross 局去掉 `random` 锚与 `self_s2_mirror`（22,000 局）。

- **D2**：拟合 `logit P(a>b)=μ_a−μ_b+δ·1[same_family]`（族：`self={self_s2..s8}`、
  `ext={p1,p2,p3,l1_1M}`），deal-clustered bootstrap B=200（seed 40）：
  **δ = −0.0065，95% CI [−0.050, +0.032]，含 0**（正态 p=0.76）→ 无"同 regime 风格重叠"证据。
- **D4**：同一 δ=0 单刻度拟合的期望，`MA(s)=mean_{b∈C0\{s}}(wr−e) − mean_{x∈{p1,p2,p3}}(wr−e)`：
  `mean_s MA = +0.001 [−0.0105, +0.0128]`（含 0）；placebo `MA_X`（x∈{p1,p2,p3}，只用独立 run）
  `= +0.0022 [−0.009, +0.0127]`（含 0）→ 无证据。D2 与 D4 出自同一批对局，**非独立**，此处
  合并为一条证据。

## 5. D5/D3 —— 终点确定化与熵-迁移相关（探索性）

产物：`runs/selfpool/probe_entropy.json`、`probe_states.npz`、`d3_d5_analysis.json`
（脚本 `runs/selfpool/probe_entropy.py`、`d3_d5_analyze.py`）。

**探针设计（plan §3 D5 的"模板头 masked entropy"）**：用一次固定种子、random-legal-action
的 rollout 生成 **4000 个合法 161 维状态**（`probe_states.npz`，`state_seed=2718`，
模型无关、跨模型共享），对每个 ckpt 前向 `policy_logits`，按 `networks.py:457` 的
`torch.split(..., nvec)` 取 head 0（134 维模板头）做 masked softmax 熵。
**不使用 raw `metrics.csv entropy`**（花色头占 85–90%，`wave5-500k-report.md:102-114`）。
自检：模板头熵 0.19–0.35 nats，与 `wave5-500k-report.md` 的 `H_template 0.13–0.26` 同量级。

**D5 主读数（唯一匹配 step = 500k，同架构 hidden=128）**：

| 族 | 成员 | 模板头 entropy (nats) | 均值 ± sd |
|---|---|---|---|
| self | self_s2..s8 | 0.2256 / 0.1874 / 0.2548 / 0.2355 / 0.2096 / 0.1967 / 0.1920 | **0.2145 ± 0.0251** |
| random-trained | p1 / p2 / p3 | 0.3516 / 0.2844 / 0.3410 | **0.3257 ± 0.0361** |

差 self−random = **−0.1112**，episode-clustered bootstrap CI（共享状态集，B=4000）
**[−0.1206, −0.1030]，排 0**。suit 头两族相近（1.13–1.36 nats），与模板头结论不冲突。
→ 描述性弱旁证：self 训练在**模板头**上更早确定化（H-det），**不承载行动声明**；`S7` 非
confirmatory `C0`。

**D3**：`S7` 的模板头熵 vs Wave-1 F1b 迁移 `runs/t17w1/h2h_selfVS1M_s{2..8}.json`
`combined.elo_diff.mean`：ρ = **−0.143**、精确置换 p（7!=5040）= **0.783**（不设 ρ 阈值）
→ **n=7 功效不足、不可判**。

## 6. 判定与 h0

- **H1 / H1' / H2**：诊断不承载；待 E1/E2/E3。**D0 未过门** → 池臂结论**"弱成员池重估"**。
- **H-mirror**：D2 δ CI 含 0 且 D4 MA CI 含 0 → **未检出**。
- **H-det**：D5 self 模板头熵显著更低（描述性）→ **弱旁证**，不改行动。
- **H-cycle**：D1 零功效、升级条件不触发 → **不可判**（不写"无循环"）。
- **H0 噪声**：Wave-1 训练 seed sd **8.43** 是 binding term（`v5-optimization-wave1.md` §4.1）
  → 任何跨 0 的 confirmatory CI 一律写"未判定"，不写 null。

## 7. 局限

1. self 训练 run **无 `snapshots/`** → 无法观测同 run 内的策略轨迹；D1/D2/D4 只是**终点群体**。
2. **终点群体非传递 ≠ 训练时追逐**；D1 MDE ≈34 Elo，self 族真实差 ~5–15 Elo → 零功效。
3. **同族/同 run 相关**：`p1_499k` 与 `l1_1M` 共用 seed1、`l1_*` 同 run；诊断按 run 聚簇。
4. **`self_s1` 选择偏差**：本报告未使用 `self_s1`。
5. **无中立异族对手（硬阻塞）**：全部强资产 random-trained → 结论限 MLP 家族内。
6. **D5 探针**：固定状态集来自 random-legal rollout，不是策略访问分布；"episode-clustered"
   指探针状态按 rollout 局聚簇，非训练 deal；`S7` 非 confirmatory。
7. **D0 未过门**：`p2/p3` 不是强成员；`Δanchor` 是跨预算比较（500k vs 1M），已按 plan 口径。

## 8. 复现命令

见 §2 与各节的脚本路径；环境 `.venv/bin/python`，诊断全程 `--device cpu --workers 4`、
一次一个工具、严格串行。脚本：`runs/selfpool/d0_diff.py`、`d1_analyze.py`、
`d1_rank_stability.py`、`fit_family.py`、`probe_entropy.py`、`d3_d5_analyze.py`。

## 9. 产物清单

`runs/selfpool/`：`freeze_record.txt`、`d0_games.jsonl`、`d0_absolute_table.json`、
`d0_diff.json`、`arena_core.json`、`arena_core_games/`、`arena_core_tb/`、
`d1_analysis.json`、`d1_rank.json`、`d2_fit.json`、`probe_entropy.json`、`probe_states.npz`、
`d3_d5_analysis.json` 及各 `.out/.err` 日志。

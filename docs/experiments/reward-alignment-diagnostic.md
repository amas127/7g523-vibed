# 奖励-评估对齐离线诊断：分差在「胜负已定」后是否应对齐？（revision-3 / T17）

> **角色**：离线诊断执行 agent。本阶段**纯 CPU、零训练、零 h2h**；只读现有产物，
> 只写本报告 + `docs/experiments/README.md` §1 一行 + `runs/reward_alignment/`。
> 未改 `src/`、`tests/`、`tools/`、`docs/adr/`、`docs/plans.md`、`CONTEXT.md`、`DESIGN.md`、
> `RULES.md`；未 commit/checkout/stash；未用 worktree。
>
> **规格**：`docs/reward-alignment-plan.md`（revision-4）§2 诊断部分 +
> `docs/reward-alignment-review.md` 的修正。本报告严格按预注册执行，不即兴加量/改门。
>
> **口径标签**：revision-3（出空即撬底，`rules_id=2e36dbea44893696`，`src/seven523/rules.py:53`）、
> obs v5（2 家 161 维）、纯 MLP（`arch=shared`、`hidden=128`）。跨 fit 绝对 Elo 不可比。
>
> **数据哈希**（`verdict.json:input_sha256`，sha256）：
> `runs/t17_screen/games.jsonl 18d80e1b…`、`runs/t17_screen/mle.json 0ced23e8…`、
> `runs/t17long__1__1790439615/metrics.csv 016a6bd5…`、`runs/t17pool__1__1790439615/metrics.csv c7f09d7e…`、
> `runs/t17pool__2__1790439615/metrics.csv`、`runs/t17pool__3__1790439615/metrics.csv`、
> `runs/t17early{2k,4k,8k}__1__17904396*/metrics.csv`。

---

## 0. 结论先行

### 0.1 用户假设

> 「和 RandomBot 对战训练时，后期 episode return 仍在升高，但评估 Elo 不变。训练奖励（分差）
> 与评估（胜负）在‘已经确保胜利’之后错位——是否应把奖励在超过 50 分（或等价地‘胜负已定’）
> 后饱和/截断？」

### 0.2 判定布尔与路由（按 `plan §2.6` 唯一决策表）

| 布尔 | 定义 | 本诊断裁定 |
|---|---|---|
| `SYMPTOM` | `Δreturn` 主估计量 95% CI 排除 0 且 > 0 | **True**（`t17long` seed 1，见 §3.1） |
| `MARGIN_ATTR` | 胜局贡献 `ΔW` 的 95% CI 排除 0 且 `ΔW>0` | **False** |
| `LOSS_ATTR` | 负局贡献 `ΔL` 的 95% CI 排除 0 且 `ΔL>0` | **False** |
| `ATTR_RESOLVED` | `MARGIN_ATTR ∨ LOSS_ATTR` | **False（归因未判定）** |
| `PLATEAU_FLAT` | 顶部平台 μ 差落在 CI 内 | **True**（顶部 5 点 μ 极差 4.01 « 平均 CI 半宽 14.19） |

**路由**：`SYMPTOM ∧ ¬ATTR_RESOLVED` ⇒ **默认 Tier 0（资源收束）**，owner 可显式触发 Tier 1
（须先过 `plan §4.1` μ-headroom 预检）。

**必须写清的措辞（`plan §2.6`）**：**归因未判定；收束是「不再为这条线排 GPU 预算」的资源决定，
不是「实验会无效」的证据。** 诊断检验的是「可观测症状」（后期 return vs Elo）；`plan §4`
检验的是「换奖励能否改变优化轨迹/评估」，两者不是同一命题。症状小 ≠ 干预无效。

### 0.3 一句话总结

「错位」作为**拓扑事实成立**（终局奖励含一个对评估完全无意义的「已锁定胜利后的分差」分量），
但**是否值得动（materiality）用现有数据判不了**：后期增量仅 ≈ +1 个 own 分、且只有 1 个训练
seed；归因在 n=200/anchor 上 CI 全部含 0。**本诊断不写「不 material」，也不写「由分差主导」**。

---

## 1. 数据核对（执行前 gate）

### 1.1 文件与字段

| 文件 | 存在 | 字段/结构 | 与规划一致 |
|---|---|---|---|
| `runs/t17long__1__1790439615/metrics.csv` | ✅（1954 行） | `global_step,episodic_return,episodic_length,episodes,...` | ✅ |
| `runs/t17pool__{1,2,3}__1790439615/metrics.csv` | ✅（各 489 行） | 同上 | ✅ |
| `runs/t17early{2k,4k,8k}__1__17904396*/metrics.csv` | ✅（2/4/8 行） | 同上 | ✅ |
| `runs/t17_screen/games.jsonl` | ✅（18200 行） | `seed,seats,scores,subject,opponent,subject_seat,kind,rules_id` | ✅ |
| `runs/t17_screen/mle.json` | ✅ | `levels,sigmas,n,ci,estimator{kind:probit-mle}` | ✅ |

- `games.jsonl`：`kind∈{anchor,cross}`；anchor vs random 每候选 **n=200**（100 seed × 2 座），
  13 候选 seed 集**完全相同**（`seed_sets_equal=True`，每 seed 出现 2 次）⇒ **按 seed 配对（CRN）成立**。
- 全部 18200 行 `sum(scores)==100`（终局分数守恒，与 `rules.py:24`/`game.py:352-353` 一致）。
- `mle.json` 的 `estimator.kind="probit-mle"`，似然只吃 outcome（ADR-0013）。

### 1.2 P0 配对 gate（按构造成立）

- 发牌只由 `(deal_seed, rules)` 决定：`play.py:364` 用 `rng=random.Random(seed)` 构造 `Match`，
  `game.py:140-147` 在 `_deal` 里 `rng.shuffle(deck)`；策略消耗独立种子
  `policy_seed = (seed + 101·(seat+1))`（`record.py:47-54`）。
- `ladder.plan_games` 对每个 anchor index 只抽一次 `deal_seed` 并在全部候选 × 2 座上复用
  （`ladder.py:239-249`）。
- **实测确认**：13 个候选 anchor seed 集完全相同（100 seed/候选，每个 2 次）。

### 1.3 语义核实（只读代码，引用 `file:line`）

- 训练奖励默认 `terminal`：非终局步 0，终局步 = `Game.returns()[learner]`（`env.py:41`、`:356`、`:358-361`）。
- `returns() = own/100 − mean(others)/100`（2 家即 `(own−other)/100`，**分差**）（`game.py:346-358`）。
- 评估只用胜负/平：`seat_outcome = sign(own − max(others)) ∈ {−1,0,+1}`（`game.py:361-365`）。
- 2 家、`total_points=100`、终局分数守恒 ⇒ `own>50 ⇔ 必胜`、`own=50 ⇔ 平`、`own<50 ⇔ 必负`
  （`rules.py:24`、`game.py:352-353`）。
- 训练 `eval_interval=0`，**没有同 run 在线 Elo**（`runs/t17pool__1__args.json`）⇒ 任何
  「return vs Elo」只能是跨 run/跨 fit 关联。

---

## 2. 方法（预注册定义，全部写死）

统一尺度：`d = own − other`（整数），`r_term = d/100 = (own−other)/100 ∈ [−1,1]`。
`outcome = sign(d)`。全部 CPU，`numpy`；RNG = `np.random.default_rng(0)`。

- **D1 后期 return 趋势**（within-run）：
  - 主估计量：`[512k,2M]` 上 `episodic_return ~ global_step` 的 OLS 斜率，单位 `/1M steps`。
  - 主 CI：moving-block bootstrap，`block=25`、随机起点 ∈ `[0,n−block]`、`B=2000`、2.5/97.5 百分位。
  - 敏感性：`block∈{10,25,50}`；Newey-West SE（lag∈{1,5,25,50}）；去趋势 acf(lag 1/5/10/25/50)。
  - 次估计量（描述）：非重叠窗口均值差 `[512k,768k)` vs `[1.5M,2M]`，iid-bootstrap CI。
  - 折算：`Δown = 50·Δreturn`。
  - 判定：`SYMPTOM := 主估计量 95% CI 排除 0 且 > 0`。
- **D2 增量分解（描述性）**：anchor vs random，对候选 c：
  `R(c)=P_w·E[r|w]+P_l·E[r|l]`、`R_sign(c)=P_w−P_l`；`W=P_w·E[r|w]`、`L=P_l·E[r|l]`。
  顶部/相邻 step `(a→b)` 用**按 seed 配对 bootstrap**（`B=4000`）报 `ΔR,ΔR_sign,ΔW,ΔL` 点估计 + 95% CI；
  一致性 `ΔR=ΔW+ΔL`。**禁用比值式归因**。
- **D3 跨 run/跨 fit 对齐（描述，非因果）**：每候选的 `episodic_return` 在 ckpt step 处的
  **centered-25**（以该 step 为中心的 25 个 update）与 screen μ 配对；报 Pearson、控制 `log(step)`
  的偏相关、子集相关、口径偏移（`return_centered` vs `R`）；`μ~R` 斜率仅作量级参考，**不作门**。
- **D4 方差份额（描述量）**：逐局 `r_term` 按 outcome 分层，
  `Var(r_term)=Var(E[r_term|o])+E[Var(r_term|o)]`，报 `within/total`；**不是梯度占比、不参与路由**。

---

## 3. 结果（表/图）

### 3.1 D1：后期 return 趋势

| run | 窗口 | n | OLS/1M | block-25 95% CI | block{10,50} CI |
|---|---|---|---|---|---|
| `t17long__1` | [512k,2M] | 1454 | **+0.01816** | **[+0.01003,+0.02400]** | [0.01042,0.02428] / [0.00978,0.02400] |
| `t17pool__1` | [0,500k] | 488 | +0.4889 | [+0.1943,+0.6474] | — |
| `t17pool__2` | [0,500k] | 488 | +0.5357 | [+0.1937,+0.7180] | — |
| `t17pool__3` | [0,500k] | 488 | +0.4494 | [+0.1898,+0.5805] | — |

`t17long` 次估计量（描述）：窗口均值 `[512k,768k)=0.51615`（sd 0.0601，n=250） vs
`[1.5M,2M]=0.53598`（sd 0.0592，n=489），差 **+0.01983**，iid-bootstrap 95% CI
**[+0.01106,+0.02900]** ⇒ **≈ +0.99 个 own 分**。centered-roll25 端点
`512k=0.51115, 2M=0.55664`，差 **+0.04549**。

去趋势 acf：lag1/5/10/25/50 = **−0.0270 / −0.0292 / −0.0213 / +0.0115 / −0.0169**；
Newey-West SE(lag1/5/25/50) = **0.003582 / 0.003676 / 0.003540 / 0.003526** ≈ iid SE
⇒ 噪声主要来自「逐点方差大 + 信号小」，**不是自相关**（与 `plan §1.6`/R2 一致）。

**裁定**：`SYMPTOM=True`（主估计量 CI 排除 0 且 > 0），但**仅 1 个训练 seed**（seed 1）。

> 跨 seed 参照：`t17pool__{1,2,3}` 在 `[450k,500k]` 的 return 均值
> **0.5213 / 0.5425 / 0.5286**（训练 seed sd **0.0107**），与整个后期 drift(+0.0198) 同量级
> ⇒ 后期 drift 落在跨 seed 变异内。

### 3.2 D2：平台候选与配对加法分解（anchor vs random，n=200/候选）

| id | μ | R | win% | tie% | loss% | E[r\|w] | E[r\|l] |
|---|---|---|---|---|---|---|---|
| `l1_512k` | 187.60 | +0.5535 | 90.0 | 3.5 | 6.5 | 0.6306 | −0.2154 |
| `p1_499k` | 190.06 | +0.5965 | 90.0 | 3.5 | 6.5 | 0.6794 | −0.2308 |
| `l1_1.5M` | 187.69 | +0.5500 | 90.0 | 3.0 | 7.0 | 0.6311 | −0.2571 |
| `l1_1M` | 191.61 | +0.5280 | 86.5 | 3.5 | 10.0 | 0.6376 | −0.2350 |
| `l1_2M` | 191.11 | +0.5625 | 89.5 | 4.5 | 6.0 | 0.6447 | −0.2417 |

配对加法分解（seed-clustered，B=4000）：

| a→b | ΔR | ΔR_sign | ΔW | ΔL | ΔR−(ΔW+ΔL) |
|---|---|---|---|---|---|
| 512k→1M | −0.0262 [−0.0840,+0.0315] | −0.0705 [−0.1550,+0.0100] | −0.0167 [−0.0675,+0.0345] | −0.0095 [−0.0225,+0.0020] | ≈0 |
| 1M→1.5M | +0.0217 [−0.0390,+0.0810] | +0.0644 [−0.0350,+0.1650] | +0.0164 [−0.0365,+0.0665] | +0.0053 [−0.0090,+0.0215] | ≈0 |
| 1.5M→2M | +0.0133 [−0.0325,+0.0570] | +0.0052 [−0.0650,+0.0700] | +0.0097 [−0.0300,+0.0475] | +0.0035 [−0.0070,+0.0145] | ≈0 |
| 512k→2M | +0.0088 [−0.0425,+0.0620] | −0.0008 [−0.0850,+0.0800] | +0.0094 [−0.0385,+0.0575] | −0.0006 [−0.0105,+0.0085] | ≈0 |

**裁定**：`ΔR`、`ΔR_sign`、`ΔW`、`ΔL` 的 95% CI **全部含 0**，且 `ΔR`/`ΔR_sign` 符号交替
（非单调）⇒ `MARGIN_ATTR=False`、`LOSS_ATTR=False` ⇒ **`ATTR_RESOLVED=False（归因未判定）`**。
**不得写「涨幅由已锁定胜利后的比分扩大主导」。**

> 反饱和线索（R10，须列出）：顶部平台 win% 非单调（90.0/86.5/90.0/89.5），
> `E[r_term|w]` 非单调（0.6306/0.6376/0.6311/0.6447）。

### 3.3 D3：跨 run/跨 fit 对齐（非因果）

| id | step | return_centered | μ | R_screen | offset |
|---|---|---|---|---|---|
| `e1_2k` | 2048 | 0.0474 | 76.58 | +0.2050 | −0.1576 |
| `e1_8k` | 8192 | 0.0238 | 115.95 | +0.3230 | −0.2992 |
| `p1_65k` | 65536 | 0.3923 | 146.96 | +0.4720 | −0.0797 |
| `p1_131k` | 131072 | 0.4676 | 167.70 | +0.4880 | −0.0204 |
| `p1_262k` | 262144 | 0.4975 | 177.56 | +0.5625 | −0.0650 |
| `p1_499k` | 499712 | 0.5261 | 190.06 | +0.5965 | **−0.0704** |
| `l1_512k` | 512000 | 0.5112 | 187.60 | +0.5535 | −0.0423 |
| `l1_1M` | 1024000 | 0.5279 | 191.61 | +0.5280 | −0.0001 |
| `l1_1.5M` | 1536000 | 0.5300 | 187.69 | +0.5500 | −0.0200 |
| `l1_2M` | 1999872 | 0.5566 | 191.11 | +0.5625 | −0.0059 |

- Pearson（13 候选）**0.9655**；控制 `log(step)` 后偏相关 **0.5033** ⇒ 整体相关**主要由训练步数混杂**。
- 子集相关（定义显式，描述性）：顶部 4 点（按 μ）= **0.3132**；t17long 四连点 = **0.6113**；
  晚期 6 点（按 step）= **0.7766**。
  > 注：预注册文档报告的「顶部 4 点 0.47、晚期 6 点 0.33」对**子集/窗口定义敏感**；
  > 本诊断给出上述显式定义下的读数。定性结论（偏相关从 0.97 掉到 ≈0.50 ⇒ 混杂主导）不变。
- 量级参考（**非因果、非上界、不作门**）：13 候选 `μ~R` OLS 斜率 **308.0 ± 15.9**（R²=0.9715，
  由早期短 run + random 锚点主导）；顶部 5 点斜率 **−4.57 ± 43.51**，95% CI **[−89.85,+80.72]**。
- 口径偏移：晚期 6 候选最大 **0.0704**（`p1_499k`：return 0.5261 vs R 0.5965）；全 13 候选最大
  **0.2992**（`e1_8k`）⇒ **训练 return 与 screen R 口径不同（不同对手实例/局数/seed），不得跨口径乘除**。

### 3.4 D4：方差份额（描述量，非门）

| id | within/total | between | within |
|---|---|---|---|
| `e1_2k` | 0.3246 | 0.11852 | 0.05695 |
| `p1_262k` | 0.4664 | 0.06390 | 0.05584 |
| `l1_512k` | 0.5566 | 0.05449 | 0.06839 |
| `l1_1M` | 0.4780 | 0.07836 | 0.07176 |
| `l1_1.5M` | 0.5269 | 0.06060 | 0.06750 |
| `l1_2M` | 0.5582 | 0.05909 | 0.07466 |
| `p1_499k` | 0.5244 | 0.06313 | 0.06961 |

**声明**：`within/total` 是「结果已知后分差还剩多少散布」的描述量，**不是梯度占比、不参与任何路由**。
原 M4 阈值 0.30 无区分度（连 2k 步 `e1_2k=0.3246` 都通过），已按 `plan §2.5` 删除。

### 3.5 图

本环境 `.venv` **无 matplotlib**（`plan §0.2 D4`）；按预注册 best-effort 规则**跳过 SVG**，
以 CSV 表替代，`verdict.json:figures="skipped"`。ASCII 摘要见 `analyze.py` stdout。
（`d1_trend.svg`、`d2_decomposition.svg`、`d3_alignment.svg` 未生成。）

---

## 4. 判据裁定与路由

| 布尔 | 值 | 依据 |
|---|---|---|
| `SYMPTOM` | **True** | `t17long[512k,2M]` OLS +0.01816/1M，block-25 CI [+0.01003,+0.02400] 排除 0 |
| `ATTR_RESOLVED` | **False** | `ΔW`/`ΔL` 配对 CI 全部含 0 |
| `PLATEAU_FLAT` | **True** | 顶部 5 点 μ 极差 4.01 « 平均 CI 半宽 14.19 |

**路由**：`SYMPTOM ∧ ¬ATTR_RESOLVED` ⇒ **默认 Tier 0（资源收束）**。

**措辞义务（已遵守）**：
- 「归因未判定；收束是资源决定，**不是**干预无效的证据」。
- 「错位」作为拓扑事实成立：终局奖励含一个对 outcome-only 评估完全无意义的分量；
  但这**不足以**推出「奖励截断值得上 GPU 实验」。
- 未写任何「分差奖励导致 Elo 平台」「饱和奖励会 +X Elo」的因果句；Elo 外推只在 §3.3 作量级参考。
- 若 owner 要真正判定归因，须把 anchor 局提到 **≥1200 局/候选**（`plan §0.2 D2`，新成本项，
  须 owner 批准）；本诊断**不生成新对局**。

**训练实验（`plan §4`，K2/K7/K0/warm-start 等）只作设计，本 workflow 不执行**，待
`selfplay-pool` 收尾、GPU 空闲后由 owner 批准预注册再单独执行。

---

## 5. 局限

1. **单训练 seed**：后期 drift 只在 `t17long__1`（seed 1）；`p2/p3` 无 2M run、不在本 fit 内。
   跨 seed 变异（sd≈0.011）与 drift(+0.0198) 同量级。
2. **归因功效不足**：n=200/anchor，单候选 `R` 的 SE≈0.025，相邻平台 `ΔR` 的 CI 半宽 0.027–0.053；
   任何 |Δ|<0.05 都判不了；要判定需 ≥1200 局/候选（未做）。
3. **跨 run/跨 fit 非因果**：`eval_interval=0`（无同 run 在线 Elo）；early/pool/long 各自
   `anneal_lr`、不同对手实例/局数；μ CI 宽 ±13。
4. **口径不可混用**：training return 与 screen `R` 最大差 0.07（晚期 6）/0.30（全 13），
   不得跨口径乘除。
5. **D3 子集相关对定义敏感**（预注册 0.47/0.33 vs 本诊断 0.31–0.78）——仅描述，不影响判定。
6. **图缺失**：无 matplotlib，SVG 跳过（CSV 替代）。
7. **工作区脏**：HEAD 含 T15–T17 大量未提交改动（属预期）；诊断只新增允许的文件（见 §7）。

---

## 6. 因果性声明（写进报告）

D1–D4 **全部是关联/描述**。顶部 return–Elo decoupling 可由评估分辨率（±13 CI）、容量、
LR 衰减、样本效率等解释；诊断只能尝试**排除**「后期由胜率提升驱动」这一竞争解释
（且前提是 `ATTR_RESOLVED`），**不能确立**奖励口径是瓶颈。本报告未做任何此类因果断言。

---

## 7. 复现命令与产物路径

**复现（CPU-only，无网络、无训练/h2h）**：

```bash
cd /home/amas/.local/src/7g523
CUDA_VISIBLE_DEVICES= .venv/bin/python runs/reward_alignment/analyze.py
```

**产物**（`runs/` 已 gitignore）：

| 路径 | 内容 |
|---|---|
| `runs/reward_alignment/analyze.py` | 诊断脚本（全文见附录 B） |
| `runs/reward_alignment/baseline.txt` | 开工前 `git status --porcelain` + `git diff` sha256 |
| `runs/reward_alignment/d1_training_return_trend.csv` | D1 OLS/CI/acf/NW/窗口 |
| `runs/reward_alignment/d2_return_decomposition.csv` | D2 候选统计 + 配对 Δ 分解 |
| `runs/reward_alignment/d3_alignment.csv` | D3 return/μ/R 与偏移 |
| `runs/reward_alignment/d4_variance.csv` | D4 方差份额 |
| `runs/reward_alignment/verdict.json` | 三个布尔 + 数值 + 输入 sha256 + 命令 + `figures` |

**索引**：`docs/experiments/README.md` §1 首行（本报告）。

---

## 附录 A：预注册数字 vs 本次复现

| 量 | 预注册（`plan §0.3`/附录 A） | 本次复现 | 一致 |
|---|---|---|---|
| `t17long` OLS/1M | +0.01816 | +0.01816 | ✅ |
| block-25 CI | [+0.0106,+0.0244] | [+0.01003,+0.02400] | ✅（同结论；bootstrap 实现细节差异） |
| 窗口均值 | 0.5162 vs 0.5360，差 +0.01983 | 0.51615 vs 0.53598，差 +0.01983 | ✅ |
| 窗口差 iid CI | [+0.0109,+0.0290] | [+0.01106,+0.02900] | ✅ |
| centered-roll25 端点 | 0.5112 / 0.5566，差 +0.0454 | 0.51115 / 0.55664，差 +0.04549 | ✅ |
| acf / NW | ≈0 / ≈iid | −0.027…+0.012 / 0.0035≈0.0036 | ✅ |
| `ΔW`/`ΔL` 配对 CI | 全部含 0 | 全部含 0；`ΔR=ΔW+ΔL` | ✅ |
| Pearson / 偏相关 | 0.966 / ≈0.50–0.62 | 0.9655 / 0.5033 | ✅ |
| `μ~R` 13 候选斜率 | 308.0 ± 15.9 | 308.0 ± 15.9（R²=0.9715） | ✅ |
| 顶部 5 点斜率 | −4.6 ± 43.5，CI [−89.9,+80.7] | −4.57 ± 43.51，CI [−89.85,+80.72] | ✅ |
| within/total（平台） | 0.478–0.558 | 0.4780–0.5582 | ✅ |
| pool [450k,500k] 均值 | 0.5213/0.5425/0.5286 | 0.5213/0.5425/0.5286（sd 0.0107） | ✅ |
| 顶部平台 μ | 187.60…191.61 | 187.60…191.61 | ✅ |

**判定一致性**：`SYMPTOM=True / ATTR_RESOLVED=False / PLATEAU_FLAT=True` 与路由结论
与 `plan §2.6` 完全一致。

---

## 附录 B：`analyze.py` 全文（内联，防 `runs/` 被清理）

```python
#!/usr/bin/env python3
"""Reward-alignment (margin saturation) offline diagnostic.

Pure-CPU, read-only w.r.t. the repo. Reproduces the pre-registered
D1-D4 computations of docs/reward-alignment-plan.md (revision-4) and writes
artifacts into runs/reward_alignment/.

No training, no h2h, no GPU. numpy mandatory; scipy optional; matplotlib optional.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs" / "reward_alignment"
OUT.mkdir(parents=True, exist_ok=True)

RNG_SEED = 0
B_TREND = 2000
B_DECOMP = 4000

# ---------------------------------------------------------------------------
# candidate -> (run metrics.csv, checkpoint step)
# mapping taken from runs/t17_screen/command.txt
# ---------------------------------------------------------------------------
CANDIDATES = {
    "e1_2k":     ("runs/t17early2k__1__1790439615/metrics.csv", 2048),
    "e1_4k":     ("runs/t17early4k__1__1790439622/metrics.csv", 4096),
    "e1_8k":     ("runs/t17early8k__1__1790439629/metrics.csv", 8192),
    "p1_16k":    ("runs/t17pool__1__1790439615/metrics.csv", 16384),
    "p1_32k":    ("runs/t17pool__1__1790439615/metrics.csv", 32768),
    "p1_65k":    ("runs/t17pool__1__1790439615/metrics.csv", 65536),
    "p1_131k":   ("runs/t17pool__1__1790439615/metrics.csv", 131072),
    "p1_262k":   ("runs/t17pool__1__1790439615/metrics.csv", 262144),
    "p1_499k":   ("runs/t17pool__1__1790439615/metrics.csv", 499712),
    "l1_512k":   ("runs/t17long__1__1790439615/metrics.csv", 512000),
    "l1_1M":     ("runs/t17long__1__1790439615/metrics.csv", 1024000),
    "l1_1.5M":   ("runs/t17long__1__1790439615/metrics.csv", 1536000),
    "l1_2M":     ("runs/t17long__1__1790439615/metrics.csv", 1999872),
}
PLATEAU = ["l1_512k", "p1_499k", "l1_1.5M", "l1_1M", "l1_2M"]
L1_CHAIN = ["l1_512k", "l1_1M", "l1_1.5M", "l1_2M"]

GAMES = ROOT / "runs/t17_screen/games.jsonl"
MLE = ROOT / "runs/t17_screen/mle.json"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# D1: training-return trend
# ---------------------------------------------------------------------------
def load_metrics(rel: str):
    steps, rets = [], []
    with open(ROOT / rel, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("episodic_return", "") == "":
                continue
            steps.append(int(row["global_step"]))
            rets.append(float(row["episodic_return"]))
    return np.asarray(steps, dtype=float), np.asarray(rets, dtype=float)


def ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.polyfit(x, y, 1)[0])


def moving_block_bootstrap_slope(x, y, block, B, rng):
    n = len(y)
    slopes = np.empty(B)
    m = max(1, n // block)
    starts_hi = n - block
    for b in range(B):
        starts = rng.integers(0, starts_hi + 1, size=m)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        slopes[b] = ols_slope(x[idx], y[idx])
    return slopes


def acf(x, lags):
    x = x - x.mean()
    denom = np.dot(x, x)
    return {L: float(np.dot(x[:-L], x[L:]) / denom) for L in lags}


def newey_west_se(x: np.ndarray, resid: np.ndarray, lag: int) -> float:
    """NW SE of the OLS slope, given design x (slope column centered) and residuals."""
    n = len(x)
    xc = x - x.mean()
    s2 = float(np.dot(xc * resid, xc * resid))  # sum (x_i e_i)^2
    meat = s2 / (float(np.dot(xc, xc)) ** 2)
    # Bartlett-weighted long-run variance of the score
    g = 0.0
    for L in range(1, lag + 1):
        w = 1.0 - L / (lag + 1.0)
        cov = float(np.dot((xc * resid)[L:], (xc * resid)[:-L])) / n
        g += w * cov
    longrun = (s2 / n + 2.0 * g) / (float(np.dot(xc, xc)) / n) ** 2 / n
    return float(np.sqrt(max(longrun, 0.0)))


def d1():
    rows = []
    table = []
    for rel, win_lo, win_hi in [
        ("runs/t17long__1__1790439615/metrics.csv", 512000, 2000000),
        ("runs/t17pool__1__1790439615/metrics.csv", 0, 500000),
        ("runs/t17pool__2__1790439615/metrics.csv", 0, 500000),
        ("runs/t17pool__3__1790439615/metrics.csv", 0, 500000),
    ]:
        steps, rets = load_metrics(rel)
        mask = (steps >= win_lo) & (steps <= win_hi)
        x, y = steps[mask], rets[mask]
        n = len(y)
        slope = ols_slope(x, y) * 1e6  # per 1M steps
        rng = np.random.default_rng(RNG_SEED)
        boot = {}
        for block in (10, 25, 50):
            if n <= block + 1:
                continue
            bb = moving_block_bootstrap_slope(x, y, block, B_TREND, rng)
            boot[block] = (float(np.percentile(bb, 2.5)) * 1e6,
                           float(np.percentile(bb, 97.5)) * 1e6)
        # residual / NW
        p = np.polyfit(x, y, 1)
        resid = y - np.polyval(p, x)
        nw = {L: newey_west_se(x, resid, L) * 1e6 for L in (1, 5, 25, 50)}
        ac = acf(y - np.polyval(p, x), (1, 5, 10, 25, 50))

        row = {
            "run": rel.split("/")[1],
            "window": f"[{win_lo},{win_hi}]",
            "n_points": n,
            "ols_slope_per_1M": slope,
            "block10_ci": boot.get(10),
            "block25_ci": boot.get(25),
            "block50_ci": boot.get(50),
            "nw_se_lag1": nw[1], "nw_se_lag5": nw[5],
            "nw_se_lag25": nw[25], "nw_se_lag50": nw[50],
            **{f"acf_lag{k}": v for k, v in ac.items()},
        }
        # secondary estimator for t17long
        if win_lo == 512000:
            # half-open A [512k,768k) matches the pre-registered n=250
            a = rets[(steps >= 512000) & (steps < 768000)]
            b = rets[(steps >= 1500000) & (steps <= 2000000)]
            diff = float(b.mean() - a.mean())
            rg = np.random.default_rng(RNG_SEED)
            ia = rg.integers(0, len(a), size=(B_TREND, len(a)))
            ib = rg.integers(0, len(b), size=(B_TREND, len(b)))
            bd = b[ib].mean(axis=1) - a[ia].mean(axis=1)
            row.update({
                "winA_mean_512_768k": float(a.mean()), "winA_sd": float(a.std(ddof=1)),
                "winA_n": len(a),
                "winB_mean_1.5_2M": float(b.mean()), "winB_sd": float(b.std(ddof=1)),
                "winB_n": len(b),
                "window_diff": diff,
                "window_diff_ci_lo": float(np.percentile(bd, 2.5)),
                "window_diff_ci_hi": float(np.percentile(bd, 97.5)),
                "delta_own_points": 50.0 * diff,
            })
            # centered rolling-25 endpoints
            def centered_roll25(target):
                i = int(np.argmin(np.abs(steps - target)))
                lo, hi = max(0, i - 12), min(len(rets), i + 13)
                return float(rets[lo:hi].mean())
            row["centered_roll25_512k"] = centered_roll25(512000)
            row["centered_roll25_2M"] = centered_roll25(2000000)
            row["centered_roll25_diff"] = (row["centered_roll25_2M"]
                                           - row["centered_roll25_512k"])
        rows.append(row)
    # write d1 csv
    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(OUT / "d1_training_return_trend.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return rows


# ---------------------------------------------------------------------------
# games
# ---------------------------------------------------------------------------
def load_anchor():
    """Return {candidate: {seed: {'r': [...], 'o': [...]}}} for anchor vs random."""
    data = {}
    with open(GAMES) as f:
        for line in f:
            r = json.loads(line)
            if r["kind"] != "anchor" or r["opponent"] != "random":
                continue
            s = r["subject"]
            own = r["scores"][r["subject_seat"]]
            other = r["scores"][1 - r["subject_seat"]]
            rt = (own - other) / 100.0
            o = int(own > other) - int(own < other)
            d = data.setdefault(s, {}).setdefault(r["seed"], {"r": [], "o": []})
            d["r"].append(rt)
            d["o"].append(o)
    return data


def stats_from_games(games):
    """games: list of (r, o). Return dict of candidate-level stats."""
    r = np.array([g[0] for g in games])
    o = np.array([g[1] for g in games])
    win = o > 0
    loss = o < 0
    tie = o == 0
    P_w = float(win.mean()); P_t = float(tie.mean()); P_l = float(loss.mean())
    Ew = float(r[win].mean()) if win.any() else float("nan")
    El = float(r[loss].mean()) if loss.any() else float("nan")
    W = P_w * Ew
    L = P_l * El
    return {
        "n": len(r), "R": float(r.mean()), "R_sign": float(o.mean()),
        "P_w": P_w, "P_t": P_t, "P_l": P_l,
        "E_r_given_w": Ew, "E_r_given_l": El, "W": W, "L": L,
        "win_n": int(win.sum()), "tie_n": int(tie.sum()), "loss_n": int(loss.sum()),
    }


def per_seed_components(cdata):
    """cdata: {seed: {'r': [...], 'o': [...]}} -> arrays over the sorted shared seeds."""
    seeds = sorted(cdata)
    rm, sm, wm, lm = [], [], [], []
    for s in seeds:
        r = np.array(cdata[s]["r"]); o = np.array(cdata[s]["o"])
        rm.append(r.mean()); sm.append(o.mean())
        wm.append(r[o > 0].sum() / len(r))
        lm.append(r[o < 0].sum() / len(r))
    return seeds, np.array(rm), np.array(sm), np.array(wm), np.array(lm)


def paired_bootstrap(a, b, B=B_DECOMP, seed=RNG_SEED):
    """Seed-level paired bootstrap of mean difference (b - a)."""
    rng = np.random.default_rng(seed)
    n = len(a)
    idx = rng.integers(0, n, size=(B, n))
    d = b[idx].mean(axis=1) - a[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def d2(data):
    # candidate tables
    cand = {c: stats_from_games([(g[0], g[1]) for s in data[c].values() for g in
                                 zip(s["r"], s["o"])]) for c in data}
    # plateau adjacent pairs (l1 chain) plus endpoint
    comp = {c: per_seed_components(data[c]) for c in data}
    pairs = [("l1_512k", "l1_1M"), ("l1_1M", "l1_1.5M"),
             ("l1_1.5M", "l1_2M"), ("l1_512k", "l1_2M")]
    delta_rows = []
    for a, b in pairs:
        _, rma, sma, wma, lma = comp[a]
        _, rmb, smb, wmb, lmb = comp[b]
        # verify shared seeds
        assert comp[a][0] == comp[b][0], (a, b)
        dR = paired_bootstrap(rma, rmb)
        dS = paired_bootstrap(sma, smb)
        dW = paired_bootstrap(wma, wmb)
        dL = paired_bootstrap(lma, lmb)
        delta_rows.append({
            "a": a, "b": b,
            "dR": dR, "dR_sign": dS, "dW": dW, "dL": dL,
            "consistency_dR_minus_dW_minus_dL":
                dR[0] - dW[0] - dL[0],
        })
    # write csv
    with open(OUT / "d2_return_decomposition.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["section", "id", "mu", "n", "R", "R_sign", "P_w", "P_t", "P_l",
                    "E_r_given_w", "E_r_given_l", "W", "L"])
        mle = json.load(open(MLE))["levels"]
        for c in sorted(cand):
            s = cand[c]
            w.writerow(["candidate", c, mle.get(c, ""), s["n"], s["R"], s["R_sign"],
                        s["P_w"], s["P_t"], s["P_l"], s["E_r_given_w"],
                        s["E_r_given_l"], s["W"], s["L"]])
        w.writerow([])
        w.writerow(["section", "a", "b",
                    "delta", "estimate", "ci_lo", "ci_hi"])
        for dr in delta_rows:
            for nm in ("dR", "dR_sign", "dW", "dL"):
                est, lo, hi = dr[nm]
                w.writerow(["delta", dr["a"], dr["b"], nm, est, lo, hi])
    return cand, delta_rows


# ---------------------------------------------------------------------------
# D3: alignment
# ---------------------------------------------------------------------------
def centered_roll25(rel, step, metrics_cache):
    steps, rets = metrics_cache[rel]
    i = int(np.argmin(np.abs(steps - step)))
    lo, hi = max(0, i - 12), min(len(rets), i + 13)
    return float(rets[lo:hi].mean())


def d3(data, cand):
    mle = json.load(open(MLE))
    mu = mle["levels"]
    metrics_cache = {}
    for c, (rel, st) in CANDIDATES.items():
        if rel not in metrics_cache:
            metrics_cache[rel] = load_metrics(rel)
    rows = []
    for c, (rel, st) in CANDIDATES.items():
        rc = centered_roll25(rel, st, metrics_cache)
        rows.append({"id": c, "step": st, "run": rel.split("/")[1],
                     "return_centered": rc, "mu": mu[c],
                     "R_screen": cand[c]["R"],
                     "offset_return_minus_R": rc - cand[c]["R"]})
    # correlations
    ret = np.array([r["return_centered"] for r in rows])
    mus = np.array([r["mu"] for r in rows])
    steps = np.array([r["step"] for r in rows], dtype=float)
    pearson = float(np.corrcoef(ret, mus)[0, 1])
    lstep = np.log(steps)
    # partial correlation controlling log step
    def resid(y, x):
        X = np.column_stack([np.ones_like(x), x])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return y - X @ beta
    partial = float(np.corrcoef(resid(ret, lstep), resid(mus, lstep))[0, 1])
    # subset correlations (all descriptive; subset definitions made explicit)
    top4 = np.argsort(mus)[-4:]                       # 4 highest screen mu
    l1chain = np.array([i for i, r in enumerate(rows) if r["id"] in L1_CHAIN])
    late6 = np.argsort(steps)[-6:]                    # 6 largest training step
    r_top4 = float(np.corrcoef(ret[top4], mus[top4])[0, 1])
    r_l1chain4 = float(np.corrcoef(ret[l1chain], mus[l1chain])[0, 1])
    r_late6 = float(np.corrcoef(ret[late6], mus[late6])[0, 1])
    # scope the caliber offset: plan A.7 reports late/pool candidates only
    late_ids = {"p1_262k", "p1_499k", "l1_512k", "l1_1M", "l1_1.5M", "l1_2M"}
    max_off_late = max(abs(r["offset_return_minus_R"]) for r in rows
                       if r["id"] in late_ids)
    # OLS mu ~ R (reference, not a gate)
    X = np.column_stack([np.ones_like(ret), np.array([r["R_screen"] for r in rows])])
    beta, res, *_ = np.linalg.lstsq(X, mus, rcond=None)
    n, p = len(mus), 2
    s2 = (res[0] / (n - p)) if len(res) else 0.0
    cov = s2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    yhat = X @ beta
    ss_res = float(((mus - yhat) ** 2).sum())
    ss_tot = float(((mus - mus.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot
    # top-5 slope
    idx5 = [i for i, r in enumerate(rows) if r["id"] in PLATEAU]
    X5 = X[idx5]; y5 = mus[idx5]
    b5, res5, *_ = np.linalg.lstsq(X5, y5, rcond=None)
    s52 = res5[0] / (len(y5) - 2)
    se5 = np.sqrt(np.diag(s52 * np.linalg.inv(X5.T @ X5)))
    top_slope, top_se = float(b5[1]), float(se5[1])
    summary = {
        "pearson_return_mu": pearson,
        "partial_corr_given_logstep": partial,
        "corr_top4_by_mu": r_top4,
        "corr_l1chain4": r_l1chain4,
        "corr_late6_by_step": r_late6,
        "ols_mu_on_R_slope": float(beta[1]), "ols_mu_on_R_se": float(se[1]),
        "ols_mu_on_R_r2": r2,
        "top5_slope": top_slope, "top5_se": top_se,
        "top5_ci": [top_slope - 1.96 * top_se, top_slope + 1.96 * top_se],
        "max_offset_late_6": max_off_late,
        "max_offset_all_13": max(abs(r["offset_return_minus_R"]) for r in rows),
    }
    with open(OUT / "d3_alignment.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "step", "run", "return_centered",
                                          "mu", "R_screen", "offset_return_minus_R"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return rows, summary


# ---------------------------------------------------------------------------
# D4: variance share
# ---------------------------------------------------------------------------
def d4(data):
    rows = []
    for c in ["e1_2k", "p1_262k", "l1_512k", "l1_1M", "l1_1.5M", "l1_2M", "p1_499k"]:
        r, o = [], []
        for s in data[c].values():
            r.extend(s["r"]); o.extend(s["o"])
        r = np.array(r); o = np.array(o)
        total = float(r.var())
        groups = sorted(set(o.tolist()))
        between = 0.0
        within = 0.0
        for g in groups:
            m = o == g
            p = m.mean()
            between += p * (r[m].mean() - r.mean()) ** 2
            within += p * r[m].var()
        rows.append({"id": c, "n": len(r), "var_total": total,
                     "var_between": float(between), "var_within": float(within),
                     "within_over_total": float(within / total) if total > 0 else float("nan")})
    with open(OUT / "d4_variance.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "n", "var_total", "var_between",
                                          "var_within", "within_over_total"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return rows


# ---------------------------------------------------------------------------
# figures (best-effort)
# ---------------------------------------------------------------------------
def make_figures(d1_rows, cand, delta_rows, d3_rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa
    except Exception as e:  # pragma: no cover
        print(f"[figures] skipped: {e}")
        return "skipped"
    # d1
    steps, rets = load_metrics("runs/t17long__1__1790439615/metrics.csv")
    mask = (steps >= 512000) & (steps <= 2000000)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(steps[mask], rets[mask], lw=0.5, alpha=0.5, label="episodic_return")
    roll = np.convolve(rets, np.ones(25) / 25, mode="same")
    ax.plot(steps[mask], roll[mask], lw=1.5, label="rolling-25")
    p = np.polyfit(steps[mask], rets[mask], 1)
    ax.plot(steps[mask], np.polyval(p, steps[mask]), "r--", label="OLS")
    ax.set_xlabel("global_step"); ax.set_ylabel("episodic_return"); ax.legend()
    ax.set_title("D1 t17long episodic_return [512k,2M]")
    fig.tight_layout(); fig.savefig(OUT / "d1_trend.svg"); plt.close(fig)
    # d2
    fig, ax = plt.subplots(figsize=(9, 4))
    ids = PLATEAU
    for k, lab in [("P_w", "P_w"), ("P_t", "P_t"), ("P_l", "P_l")]:
        ax.plot(range(len(ids)), [cand[c][k] for c in ids], "o-", label=lab)
    ax2 = ax.twinx()
    ax2.plot(range(len(ids)), [cand[c]["E_r_given_w"] for c in ids], "s--",
             color="k", label="E[r|w]")
    ax.set_xticks(range(len(ids))); ax.set_xticklabels(ids, rotation=30)
    ax.legend(loc="upper left"); ax2.legend(loc="upper right")
    ax.set_title("D2 plateau outcome/margin")
    fig.tight_layout(); fig.savefig(OUT / "d2_decomposition.svg"); plt.close(fig)
    # d3
    fig, ax = plt.subplots(figsize=(7, 5))
    for r in d3_rows:
        ax.scatter(r["return_centered"], r["mu"], s=20)
        ax.annotate(r["id"], (r["return_centered"], r["mu"]), fontsize=6)
    ax.set_xlabel("training return (centered-25)")
    ax.set_ylabel("screen mu")
    ax.set_title("D3 return vs mu (association only)")
    fig.tight_layout(); fig.savefig(OUT / "d3_alignment.svg"); plt.close(fig)
    return "done"


def ascii_d1(rows):
    print("\n=== D1: training-return trend ===")
    for r in rows:
        print(f"  {r['run']} {r['window']} n={r['n_points']} "
              f"OLS/1M={r['ols_slope_per_1M']:+.5f} "
              f"block25 CI=[{r['block25_ci'][0]:+.5f},{r['block25_ci'][1]:+.5f}]")
    for r in rows:
        if "window_diff" in r:
            print(f"  t17long window mean diff = {r['window_diff']:+.5f} "
                  f"CI=[{r['window_diff_ci_lo']:+.5f},{r['window_diff_ci_hi']:+.5f}] "
                  f"-> {r['delta_own_points']:+.2f} own points")
            print(f"  centered-roll25 512k={r['centered_roll25_512k']:.4f} "
                  f"2M={r['centered_roll25_2M']:.4f} "
                  f"diff={r['centered_roll25_diff']:+.4f}")


def main():
    print("=== input hashes ===")
    inputs = [GAMES, MLE] + [ROOT / r for r, _ in CANDIDATES.values()]
    uniq = []
    for p in inputs:
        if p not in uniq:
            uniq.append(p)
    hashes = {str(p.relative_to(ROOT)): sha256(p) for p in uniq}
    for k, v in hashes.items():
        print(f"  {k} {v[:16]}...")

    d1_rows = d1()
    data = load_anchor()
    # cross-seed variation of the late t17pool runs (seed 1/2/3), [450k,500k]
    pool_late = {}
    for s in (1, 2, 3):
        st, rt = load_metrics(f"runs/t17pool__{s}__1790439615/metrics.csv")
        m = (st >= 450000) & (st <= 500000)
        pool_late[f"pool_{s}"] = float(rt[m].mean())
    pool_late["seed_sd"] = float(np.std(list(pool_late.values()), ddof=1))
    cand, delta_rows = d2(data)
    d3_rows, d3_summary = d3(data, cand)
    d4_rows = d4(data)
    figs = make_figures(d1_rows, cand, delta_rows, d3_rows)

    ascii_d1(d1_rows)
    print("\n=== D2: plateau candidate stats ===")
    mle = json.load(open(MLE))["levels"]
    for c in PLATEAU:
        s = cand[c]
        print(f"  {c:9s} mu={mle[c]:7.2f} R={s['R']:+.4f} sign={s['R_sign']:+.4f} "
              f"Pw={s['P_w']:.3f} Pt={s['P_t']:.3f} Pl={s['P_l']:.3f} "
              f"E[r|w]={s['E_r_given_w']:.4f} E[r|l]={s['E_r_given_l']:.4f}")
    print("\n=== D2: paired additive decomposition (B=4000) ===")
    for dr in delta_rows:
        print(f"  {dr['a']:9s}->{dr['b']:9s} "
              f"dR={dr['dR'][0]:+.4f}[{dr['dR'][1]:+.4f},{dr['dR'][2]:+.4f}] "
              f"dRsign={dr['dR_sign'][0]:+.4f}[{dr['dR_sign'][1]:+.4f},{dr['dR_sign'][2]:+.4f}] "
              f"dW={dr['dW'][0]:+.4f}[{dr['dW'][1]:+.4f},{dr['dW'][2]:+.4f}] "
              f"dL={dr['dL'][0]:+.4f}[{dr['dL'][1]:+.4f},{dr['dL'][2]:+.4f}]")
    print("\n=== D3 ===")
    for r in d3_rows:
        print(f"  {r['id']:9s} step={r['step']:>8d} return_c={r['return_centered']:.4f} "
              f"mu={r['mu']:7.2f} R={r['R_screen']:+.4f} offset={r['offset_return_minus_R']:+.4f}")
    print("  summary:", json.dumps(d3_summary, indent=2))
    print("\n=== D4 ===")
    for r in d4_rows:
        print(f"  {r['id']:9s} within/total={r['within_over_total']:.4f} "
              f"between={r['var_between']:.5f} within={r['var_within']:.5f}")
    print("\n=== cross-seed t17pool [450k,500k] ===")
    for k, v in pool_late.items():
        print(f"  {k}: {v:.4f}" if k != "seed_sd" else f"  {k}: {v:.4f}")

    # -----------------------------------------------------------------
    # verdicts
    # -----------------------------------------------------------------
    # SYMPTOM: main estimator (l1 window [512k,2M]) OLS block25 CI excludes 0 and >0
    l1 = [r for r in d1_rows if r["run"].startswith("t17long") and r["window"] == "[512000,2000000]"][0]
    symptom = bool(l1["block25_ci"][0] > 0 and l1["ols_slope_per_1M"] > 0)
    # attribution
    margin_attr = any(dr["dW"][1] > 0 and dr["dW"][2] > 0 for dr in delta_rows)
    loss_attr = any(dr["dL"][1] > 0 and dr["dL"][2] > 0 for dr in delta_rows)
    attr_resolved = bool(margin_attr or loss_attr)
    # PLATEAU_FLAT: top5 mu pairwise CIs cover 0 -> here: top5 mu spread vs CI width
    plateau_mus = np.array([mle[c] for c in PLATEAU])
    plateau_ci = np.array([json.load(open(MLE))["ci"][c] for c in PLATEAU])
    # flat if max pairwise mu difference < 0.5 * min CI half-width
    max_pair = float(plateau_mus.max() - plateau_mus.min())
    ci_half = float((plateau_ci[:, 1] - plateau_ci[:, 0]).mean() / 2)
    plateau_flat = bool(max_pair <= ci_half)

    verdict = {
        "SYMPTOM": symptom,
        "SYMPTOM_detail": {
            "run": l1["run"], "window": l1["window"],
            "ols_slope_per_1M": l1["ols_slope_per_1M"],
            "block10_ci": l1["block10_ci"], "block25_ci": l1["block25_ci"],
            "block50_ci": l1["block50_ci"],
            "window_diff": l1.get("window_diff"),
            "window_diff_ci": [l1.get("window_diff_ci_lo"), l1.get("window_diff_ci_hi")],
            "delta_own_points": l1.get("delta_own_points"),
        },
        "MARGIN_ATTR": bool(margin_attr),
        "LOSS_ATTR": bool(loss_attr),
        "ATTR_RESOLVED": attr_resolved,
        "PLATEAU_FLAT": plateau_flat,
        "PLATEAU_detail": {"mu": plateau_mus.tolist(),
                           "max_pairwise_diff": max_pair,
                           "mean_ci_half": ci_half},
        "deltas": delta_rows,
        "d3_summary": d3_summary,
        "pool_late_450_500k": pool_late,
        "route": (
            "not_SYMPTOM: collapse"
            if not symptom else
            ("SYMPTOM and not ATTR_RESOLVED: default Tier 0 (resource decision; "
             "NOT evidence intervention is inert)"
             if not attr_resolved else
             "SYMPTOM and ATTR_RESOLVED and PLATEAU_FLAT: owner decides Tier 0/Tier 1")
        ),
        "figures": figs,
        "input_sha256": hashes,
        "command": "CUDA_VISIBLE_DEVICES= .venv/bin/python runs/reward_alignment/analyze.py",
        "rng": {"seed": RNG_SEED, "B_trend": B_TREND, "B_decomp": B_DECOMP},
        "causal_disclaimer": (
            "D1-D4 are association/description only. No statement that margin "
            "reward causes the Elo plateau; no Elo extrapolation used as a gate."),
    }
    with open(OUT / "verdict.json", "w") as f:
        json.dump(verdict, f, indent=2)
    print("\n=== VERDICT ===")
    print(json.dumps({k: verdict[k] for k in
                      ("SYMPTOM", "MARGIN_ATTR", "LOSS_ATTR", "ATTR_RESOLVED",
                       "PLATEAU_FLAT", "route", "figures")}, indent=2))
    return verdict


if __name__ == "__main__":
    main()
```

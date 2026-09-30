# 无漂移 OpenSkill 两方案：裁决报告（含完整性审计修正）

> **历史决策记录（2026-09-26）**：本文是两份外部设计的对抗审查裁决报告，不是当前
> 实现指南。其结论已固化为 [`ADR-0013`](../docs/adr/0013-drift-free-rating-channel.md)：
> 保留 `elo.fit_ratings` 在线 seam，发布绝对表走独立 `mle.py` / `tools/refit_mle.py`，
> 不建 SPRT、不改 `PlayedGame`。文中 `file:line`、行号与「建议新建」清单为当时快照，
> 原始语料见 [`verified-findings.md`](./verified-findings.md)。

> 过程：8 视角 R1 批判（64 条）→ 逐条对抗验证（1 条被驳回）→ 4 视角攻击修复方案（31 条）→ 验证 → 裁决合成 → 完整性审计（17 条修正）。
> 原始语料：`verified-findings.md`；25 条排序问题：`synth.json`；审计全文：`completeness.json`。

# 裁决报告：两套 "drift-free OpenSkill" 设计（目标仓库 /home/amas/.local/src/7g523）

## 0. 裁决摘要

- **Design A（gemini，棋类/AlphaZero 风格）的招牌机制是坏的**：`batch_recalibrate` 在迭代 `self.match_history` 的同时 `update_match` 向同一 list append，永不终止并破坏日志；即便修好终止性，它仍是"reset-to-default + 按原顺序重放 online update"，既不是它 §1 声称的 anchored MLE，也去不掉 path dependence，还会丢弃注册时的 `Prior`/继承初始化。**Design A 的"彻底消除漂移"没有机制支撑。**
- **Design B（grok，LLM-snapshot 风格）的方向基本正确，但两个关键常量对接本仓库失败**：`tau=0` 被 `FitConfig` 直接拒绝，且与本仓库 `(mu, sigma=0)` 的 anchor 约定一起会让 openskill `rate()` 抛 `ZeroDivisionError`；发布层依赖的"全量 PL-MLE"在 openskill 6.2.0 中不存在，且新建它会与 ADR-0011 的"单一估计器/不新增接缝"冲突。
- **两套设计都误判了本仓库真正测到的漂移**（`plan_games` 的 candidate-index → seat-phase 确定性效应，0–75 Elo），也都没有把 rules/version 写进估计器输入。
- **大量 "fatal" 是跨 lens 重复计数**：`tau=0` 那一族共 6 条、`batch_recalibrate` 那一族共 8 条。按机制去重后本仓库的真实结论是：A 有 2 个 fatal（无限循环、重放非 MLE），B 有 1 个 major 集成阻塞（tau=0 与 repo seam 冲突），其余是 major/minor。
- **本仓库现有栈已经解决了大部分问题**：`ladder.plan_games` 的换座 twin、`duel.paired_duel_stats` 的 deal-cluster bootstrap、`study.merge_manifest(refit=False)` 的发布冻结、`elo.fit_ratings` 的单一 online seam、单 gauge RandomBot=0（ADR-0012）。所以真正该做的是**少量 schema/文档/门禁纪律**，而不是第二个估计器。
- 精确定位：本仓库 `Game`/`PlayedGame` 是 N 座带 `scores`；`duel.py` 是 paired twin + deal bootstrap；`bootstrap.py` **不是**统计 bootstrap 模块（它是 PPO 训练器），context.md 把它误标成 "clustered bootstrap helpers"，所有 fix 都应指向 `duel.py`。
- 下文对每条 survived finding 做了去重与收窄；被 refuted 的断言（如"两套设计都用 proximity 匹配"、"B 的 tau=0 as-written 不可实现"、"两套都 bootstrap battles"）已删除，部分 confirmed 的按 verifier 的修正原样收窄，真正有争议的显式标注 **disputed**。

---

## 1. 事实基座（先对齐 repo 现状）

| 事实 | 证据 |
|---|---|
| 只有一条估计 seam：`elo.fit_ratings(games, *, anchors, priors, config)`，chronological replay，order-dependent by design | `src/seven523/elo.py:205-292`（docstring 明说） |
| 常数在项目尺度：`DEFAULT_MU=500, DEFAULT_SIGMA=200, DEFAULT_BETA=100, DEFAULT_TAU=2`；openskill 差量参数是其 24× | `elo.py:57-63` |
| `FitConfig` 要求 `tau>0` | `elo.py:144-146`；test 锁定 |
| anchor = `(mu, sigma=0)`，永不更新 | `elo.py:244, 259`；ADR-0011 |
| 单一 gauge：RandomBot `mu=0`；删除 greedy=1315 | ADR-0012 |
| paired deal / seat rotation | `ladder.py:115-175`；`tests/test_ladder.py` |
| deal-cluster bootstrap + 多 seed 合并 | `duel.py:104-381`（`paired_duel_stats` / `combine_duel_seeds`） |
| 发布冻结：`merge_manifest(refit=False)` 默认按 id 冻结 | `study.py:56-91` |
| 概率模型唯一真相：`expected_score = predict_win`（probit），**不是** 400 点 logistic | ADR-0011 decision 2；`elo.py:180-204` |
| `PlayedGame = (seed, seats, scores)`，无 rules 字段 | `elo.py:100-113` |
| rules 只在 trace 里有，估计器输入/manifest 没有 | `trace.py:62-63,177`；`study.py:22,26` |
| ADR-0007：规则变更前的数字不可混比 | `docs/adr/0007-family-comparison.md:31-33` |
| 仓库没有 SPRT / 自动 promotion gate | `grep -rni sprt src tools tests docs` 为空 |
| PPO PFSP 只驱动训练采样，不喂 rating fit | `policies.py:202`；`league.py` |
| `placement/opponents.select_opponent` 的 `info` 分支最大化 `p*(1-p)` | `placement/opponents.py:197-242` |

---

## 2. Design A（design-gemini.md）裁决

### 2.1 真正的概念错误

1. **"Whole-History Refit" 是伪机制（fatal）。** `batch_recalibrate` 遍历 `self.match_history` 的同时 `update_match` 末尾 `self.match_history.append(...)`；Python list 迭代按索引推进，循环永不结束，日志无界增长。多个独立复现（历 5→200,005 行 / 60 局重放使 history 翻倍）。**即使把迭代改成 `tuple(...)` 快照**，函数体仍是"把每个非 anchor 重置为 `default_mu/default_sigma`，再按同一顺序重放 order-dependent 的 online update"——这**不是** §1 里写的 anchored PL/BT 联合 MLE，去不掉 path dependence（60 局 log 重塑最大 |Δμ|≈3.87=0.93β；120 局 log 散布 94.6 单位≈0.47σ）。它还会丢弃 `register_node` 传入的自定义 `Prior` 和"继承当前 Best 的 μ"的初始化。
2. **anchor 不是它自己说的硬 pin。** `register_node(is_anchor=True)` 默认 `sigma=default_sigma=8.333`，不是 `sigma0=0`；`restore_anchor` 从未被调用（`grep -c` = 1，仅定义）；`to_elo_scale(name, scale=...)` 完全忽略 `scale`，硬编码 `400/8.333`（`to_elo_scale(25, scale=1)=1500`，`scale=999` 仍 1500）。一个 `sigma>0` 的 anchor 是 soft reference，会改变对手更新几何（`c=sqrt(σ_c²+σ_a²+2β²)` 与 `σ_c²/σ_a²`）。注：它的 μ 实际上仍被 `update_match` 丢弃写回 + `EvaluatorNode.mu` 返回 `_anchor_mu` 而"硬"住，所以缺陷是**几何/精度**而非可移动。
3. **测量模型 iid 1v1 二元，无法表达本仓库。** `match_history: List[Tuple[str,str,int]]` 无 seed / seat / scores，且 0/1/0.5 的 draw 与 `int` 类型声明不自洽。目标游戏是 N 座 + 整数分数 + 高分频，因此 SPRT 的方差模型、样本量、rating 更新全部失准；它无法取证或防御本仓库自己的 seat-phase 漂移。
4. **SPRT 门禁欠定义且被错误口径描述（fatal-ish）。** 只有 `H0: p≤0.50 / H1: p≥0.55`，没有 α、β、LLR 边界、max-games、log 拒绝样本；在常规 α=β=0.05 读法下它是一个 **≈+35 Elo 检测器**，不是 pass/fail 回归门（p 接受率 0/10/15/20/35 Elo ≈ 0.048/0.215/0.396/0.607/0.954）。**注意收窄**：它的 SPRT 是 vs 当前 Best，而**公布 Elo 来自"与锚点池对弈"的独立层**，所以"公布值被同一样本选择偏置 +12..+45 Elo"这一 fatal 断言**不成立**。残留问题是：(i) 若把 stopping-time 观测胜率当作进步幅度，它是上偏的；(ii) 没有 alpha spending；(iii) 用 gate 结果挑 anchor candidate 使 anchor 集成为非随机子集。
5. **`conservative_rating = mu - 3σ` 是历史/不确定度统计量，不是强度。** 固定 μ 时它随 σ 收缩移动约 395 项目点（n=0: 500-3·200=-100 → n=40: 288）。**收窄**：设计只在 docstring 写"常用于排位与门禁判定"，代码从未调用；A 真正的 gate 是 SPRT。所以这是"若用它做 gate 会引入时间漂移"的意图级风险，不是已实现的缺陷。

### 2.2 对本仓库不适用 / 错误的选择

- **坐标系统错**：整篇建在 openskill 原生 25 / 8.333 / β=25/6 上，而本仓库是 500/200/100，RandomBot=0。`to_elo_scale(500)` 会得到 24300+；所有阈值都不可直接移植。
- **自适应匹配直接喂发布 fit**：`select_opponents_for_candidate` 选对手后立刻 `update_match`，发布 rating 无 selection model，属 outcome-dependent 处理。**收窄**：`match_score = -|Δμ| + 5.0·is_anchor` 在 A 自己的 25-scale 上 5.0=0.6σ₀（≈+240 Elo 等效），anchor 被**强烈**偏好，不是"effectively never preferred"；"equal μ 时永远选最低 σ"也错（stable sort → 取决于注册顺序）。但 `σ` 完全不进 score，与它自己 §5 "基于不确定性"的表自相矛盾。
- **SPRT 在**本仓库**是 over-engineering**：仓库已有 `duel.paired_duel_stats`（deal-cluster winrate/elo CI + per-deal sign test）和 `combine_duel_seeds`（多 seed 合并），验证协议 §9 就是固定样本 + 合并 CI。引入一个用退休 400-logistic "35 Elo" 常数的 SPRT 增加机器与 family-wise 风险，却不解决 drift 问题。
- **cross-win matrix 大部分与现有代码重复**：`arena.pair_stats`/`pair_diagnostics` 已是 per-pair W/D/L + observed-vs-model z 的 transitivity smoke test；缺的是 cycle-level/一致性诊断与跨 run 聚合。**收窄**：`pair_stats` 假设恰好两座（`left,right=game.seats`），不是通用 N 座矩阵。
- **Nash/exploiter**：是训练策略干预，不在 rating seam；Nash averaging 在本数据预算下 support 不可识别（见 §6）。

### 2.3 正确的想法（保留）

- anchor 冻结 + 单一 gauge 的方向与 ADR-0012 一致。
- 识别非传递性是 1-D model 的失效模式，方向对。
- §1 的 anchored PL/BT MLE 目标函数 `argmax Σ log P(outcome|μ) s.t. μ_anchor=C` 是数学上正确的（只是没实现）。
- 新 checkpoint 重置高 σ 的思路与仓库 `Prior` 一致。
- 没有做 Σμ 守恒（正确，见 §6）。

### 2.4 Design A 优先修改清单

1. **删除 `batch_recalibrate` 与"彻底消除漂移"声明。** 若确需全局 order-free 表：新建独立纯 Python 模块 `src/seven523/mle.py` + `tools/refit_mle.py`，读已有 JSONL，走新 ADR 显式 supersede ADR-0011 decision 1；**不要放进 `elo.py`**。
2. 把 `match_history` 换成 `elo.PlayedGame(seed, seats, scores)`，用 `elo.fit_ratings` 更新；N 座/平局用 `scores`。
3. 用 `ladder.plan_games` / `duel.plan_duel_schedule` 排期，用 `duel.paired_duel_stats` / `combine_duel_seeds` 做比较；SPRT 若要保留必须 pre-register α/β/LLR/max-games 并把 H1 定义在 deal cluster 上，否则直接用现有 CI/sign-test。
4. 配对/选对手用 `elo.expected_score` 的 `p*(1-p)`（参照 `placement/opponents.select_opponent`），并把 adaptive 层限制在 matchmaking，不喂发布 fit。
5. anchor 只对**外生、版本冻结的策略**（RandomBot）用 `sigma=0`；被测 checkpoint 作为 free id 或带 `Prior` 的引用节点，不硬 pin；修/删 `restore_anchor` 与 `to_elo_scale` 的 `scale` 参数。
6. 常数全部取自 `FitConfig`/`DEFAULT_*`；El display 从 winrate 推导（`400*log10(p/(1-p))`，如 `duel.py:74`）。
7. 删 `conservative_rating` 作为 gate/ranking 依据。

---

## 3. Design B（design-grok.md）裁决

### 3.1 真正的概念错误

1. **"Chatbot Arena 因为 online Elo drift 才换 BT、并冻结旧模型"是事实错误。** Arena 的 BT 固定一个 gauge（`xi_1=0`），**每次 refit 重估所有非 anchor**，且 deployed interval 是 sandwich robust SE 而非 bootstrap；论文没有把换 BT 归因于 online Elo 的顺序依赖。**这条错误论证不应作为设计理由。** 但设计真正的 invariant（gauge 定坐标、估计值每次 refit 重估）本身与 ADR-0012 一致。
2. **`tau=0` 与本仓库不兼容（major，非 6 个 fatal）。** `FitConfig(tau=0.0)` 抛 `ValueError: tau must be positive and finite`（`elo.py:144-146`）；openskill `PlackettLuce(tau=0).rate(...)` 只要**任一**参与者 `sigma=0` 就在 `plackett_luce.py:879` 的 `sigma**2/team_i.sigma_squared` 抛 `ZeroDivisionError`（不止"全 σ=0 队"；verifier 修正：**至少一个** σ=0 队即触发）。**收窄**：Grok 自己的 anchor 保持 `sigma>0`（未显式设 0），所以"tau=0 as-written 不可实现"**被驳回**；真实阻塞是它与本仓库 `(mu, sigma=0)` pin 约定 + `FitConfig` 契约的冲突。且 tau=0 会让 σ 单调下降无平衡点，established（μ 冻结）会越来越"过度自信"；`limit_sigma=True` 只禁止单次 rate() 内 σ 上升，不是 floor。
3. **非零和 → 膨胀 是 non sequitur。** 2P Weng-Lin 的 `ΣΔμ = (σ_a²-σ_b²)(y-p)/c`，在 σ 相等时为 0，条件期望在模型 calibrated 时为 0；τ 对称加到两边、不改变 `σ_a²-σ_b²`。**所以"Δμ 非零和，因此 pool 膨胀"错误，per-game 减均值规则应删除**（它是在对抗 anchor gauge）。**但**第二轮证实：online replay 确实有系统性 level bias（复现：online c/d/e=477.7/514.5/571.9 vs true 400/450/500；anchored MLE≈401/448/503），且该偏差不是 warm-up。**所以只删减均值规则、不删"level audit"**；且不能用 openskill σ 当标准误去验证校准（clone contrast sd 13.5 vs absolute-error sd 53.1）。
4. **Bootstrap 以 battle 为单位**（`design-grok.md:200` "对战报有放回重采样"），忽略 twin/deal 相关。**收窄**：本仓库固定换座 arm 的 single-arm bootstrap 其实已校准（20.66 vs Fisher 21.65 ≈0.95；seed-sd/bootstrap=1.11），所谓 ~2.0 是 **cross-order/phase 的方差设计效应**，不是通用 CI 低估；真正适用于 B 的修正是 twin-correlation 的 **1.40×**。对 A 而言它根本没有 bootstrap，所以"both designs bootstrap battles"是错误框架。
5. **无正则的 anchored MLE 在 separation 下发散。** 一个强 checkpoint 几乎全胜弱 RandomBot gauge，unregularised anchored BT/PL unbounded（复现：clean N-game sweep 随迭代发散；199W-1L 记录恰得 748.586）。**收窄**：`"virtually always beats gauge"` 过度——实测 candidate-vs-random 胜率 88.5–93.3%，100 局 clean sheet 约 0.02–0.2%；且该 MLE 的正确性依赖 link：仓库是 Gaussian PL（`predict_win=Φ(d/√(2β²+σ²))`），用 logistic BT MLE 在 Gaussian DGP 上偏差巨大（~893 vs true 400）。这更像 **separation guard / robustness gap**。
6. **生命周期"established μ 永不写"与本仓库消费方式冲突（major confirmed）。** `bootstrap._mass_rate`/`rate_pool` 每轮用 fresh fit 重排 pool（`Pool.apply_fit`/`trim`），fork 按 μ 取最强，placement 每次 session 重拟合。冻结 established μ 会让 eviction/parent/opponent/human placement 跑在旧数字上；设计本身也自相矛盾（§3 说 online 层"允许漂移"，§4/§5 又冻结）。**收窄**：(a) "anchor-frozen 永不写"恰好与仓库一致（anchor 本就 pin）；冲突点是**非 anchor established** 的冻结。(b) lvl4 -59.7 是旧 `plan_games` seat-phase bug，不是 rating drift，且正是 critic 推荐的 refit 路径修的。

### 3.2 对本仓库不适用 / 错误的选择

- **原生常数 25 / 8.333 / β=25/6 / τ=0，及 `display = 1500 + 40*(mu-25)`**：与本仓库尺度差 24×。"1 OpenSkill point ≈ 40 Elo" 在两种尺度都**错**（原生 dμ=1→p=0.5303≈21 Elo/point，dμ=50→≈33；repo σ=200,dμ=100→p=0.6241≈0.88 Elo/point）。`mu0=25` 只是原点（ADR-0012 的原点是 RandomBot=0），不是"24× smaller"；有意义的是 σ/β/τ 的 24×。
- **发布层全量 PL-MLE 不存在**：openskill 6.2.0 无 fit/MLE 入口；新建它是第二个估计器/seam，与 ADR-0011 decision 1（"在线重放代替联合 MAP"）和"priors 是唯一注入点、不新增接缝"冲突。Grok 用的是**同一个** Gaussian PL likelihood，所以不是 ADR-0011 字面拒绝的"第二似然"，但仍是**第二估计器**。scipy 在 pyproject 中不存在（deps 只有 gymnasium/openskill），`elo.py` 文档明令 no numpy/scipy——所以若要建，必须纯 Python。
- **PFSP→IPW 论证对 rating 层不成立**（第二轮收窄）：所有 fit consumer 读 balanced planned schedule；`PfspController`/`pfsp_weights` 只驱动训练采样，且 propensity 已写进 `pfsp_weights.csv`。
- **多锚 soft L2 的 λ 任意性**：`design-grok.md:188-191` 把 `Σ(μ_a-c_a)²` 进先验；λ 会连续改变发布值（λ=0→1e4 使 F 从 0.992 扫到 1.652）。**收窄**：有硬 gauge A1 时 scale 仍可识别，λ 只是 prior bias-variance，不是"identifiability 丧失"；本仓库 ADR-0012 已禁多 pin，基本不适用。

### 3.3 正确的想法（保留）

- **两层拆分**（online 只服务匹配/校准；发布尺度另算）是核心洞察，且方向与本仓库"在线重放 + manifest 冻结"一致。
- **正确识别**：order dependence 真实存在；Δμ ∝ σ²，非零和；不应该做 Σμ 守恒。
- **多队 PL 直接走 `ranks`，不拆假 1v1**——正确，也是仓库 `PlayedGame` 的形状。
- **σ 重置 / 新模型高 σ 入池**与 `Prior` 一致。
- **报告进步用 Δμ + `predict_win` + CI，不用 ordinal**——正确。
- **任务包版本化**（§9.3）方向正确。
- **established 只吸收 σ、不改位置**的思路在"发布层"是对的，只是层放错了。

### 3.4 Design B 优先修改清单

1. `tau>0`（`DEFAULT_TAU=2`），不要 `tau=0`。冻结来自 **write-back 规则或 `study.merge_manifest(refit=False)`**，不来自 τ。若坚持 τ=0，先改 `FitConfig` 契约并加"σ 收敛而非坍缩"测试，且不能与本仓库 `sigma=0` anchor 同用。
2. 发布层：**要么不建**（现有 `duel.paired_duel_stats`/`combine_duel_seeds` 已 order-free / scale-free）；**要么**独立纯 Python `src/seven523/mle.py` + 新 ADR，使用 `Prior` 做 MAP/正则 + separation guard（"zero-loss vs gauge 拒绝发布/报 ≥scale max"）+ finite-mu 回归测试；link 用 Gaussian/probit。
3. Bootstrap 改 **deal twin cluster**（`duel.paired_duel_stats`）；多 seed 用 `combine_duel_seeds`（`max(bootstrap SE, between-seed sd)/√k`）。
4. 生命周期：live rating 继续由 `fit_ratings` 更新并供 `pool.trim`/fork/placement；只有**发布账本**（manifest `levels`/`subjects.mu`）冻结。
5. 删除 per-game 减均值；保留 absolute-level audit（deal-cluster CI + realized-vs-predicted winrate vs held-out reference），不要用 σ 当 SE。
6. 常数取 `FitConfig`/`DEFAULT_*`；display 由 winrate 推导，不用固定 40×。
7. 任务版本进估计器输入（见 §4）。

---

## 4. 真正让本仓库 drift-free 的路线

**先把"no drift"分类，再逐条给测试**（这是两套设计都缺的 deliverable）：

| 漂移类型 | 机制 | 本仓库现状 | 需要的动作 |
|---|---|---|---|
| (i) location/scale | gauge 未定 / 多钉 | 已修：单 gauge RandomBot=0（ADR-0012） | 维持；只对**外生**策略用 `sigma=0` |
| (ii) order/path | `fit_ratings` 单遍重放 order-dependent | 已文档化；比较通道已用 order-free 的 `duel.paired_duel_stats` | 若产品要求"全局 order-free 绝对表"，才建纯 Python MLE（新 ADR）；否则不建 |
| (iii) intransitivity | 1-D model | `arena.pair_diagnostics` 是 per-pair smoke test | 加 CI-gated + null-calibrated cycle 统计；Nash 仅内部诊断 |
| (iv) true improvement | 真·进步 | `bootstrap.progress_margin=10` 消费 online μ | 报 deal-cluster h2h Δ + winrate vs fixed milestone；不要 gate 在 ordinal 上 |
| (v) task/rules change | 尺子**内容**变了 | `PlayedGame`/manifest 无 rules；`trace` 有 | 给估计器输入 / manifest 加 version tag，跨版本拒绝 pooling；CUSUM 残差 monitor |

**最低可行改动（MVP，低风险）**

1. 写一份短 ADR：定义 drift taxonomy、声明 h2h/`combine_duel_seeds` 是比较通道、明确单 gauge。
2. `PlayedGame` **保持 `(seed, seats, scores)` 不动**（ADR-0006/0011 冻结 core）；把 rules/task version tag 加在 **ladder/study manifest** 层，并在 fit/pooling 入口拒绝跨版本。若要让估计器直接看到，走"新增一个上层 wrapper 传入已过滤的 games"，不新增 core 字段。
3. 若需要自动化 promotion gate：把已有 `duel.ci`/`combine_duel_seeds` 接到 caller，pre-register ≥3 seed × 400 deal、CI 不含 0 且点估计 ≥+10 的规则（验证协议 §9），**不要**新建 SPRT。
4. 把已有的 permutation regression（`tests/test_ladder.py::test_plan_games_is_order_invariant_and_seat_balanced`）提升为 release gate。

**只有满足以下全部条件才建 batch MLE**：(a) 产品硬性要求"单一全局 order-free 绝对表"；(b) 走新 ADR supersede ADR-0011 decision 1；(c) 纯 Python（无 numpy/scipy）；(d) MAP/正则 + separation guard + finite-mu 测试；(e) 作为独立 tool，不进 `elo.py`；(f) 接受永久维护第二估计器的成本。

---

## 5. 哪些要求是 over-engineering / 不该做

- **SPRT accept gate**：不要新建。仓库无 gate，也无此需求；用现有 CI/sign-test。`repo:fix-improvement-gate` 建议的 `duel.sprt_decision` 属过度工程。
- **把 `fit_ratings`/`FitConfig` 加 per-id status map、把 rules hash 塞进 `PlayedGame`**：违反 ADR-0011"priors 唯一注入点、不新增接缝"与 ADR-0006/0011 的 `PlayedGame` 形状。两个目标都能在 orchestration 层（`anchors`/`Prior` + manifest refit；manifest 上的 rules tag）达成。
- **scipy L-BFGS MLE**：本仓库禁止；且该 fix 的 severity 应降为 "blocked recipe / medium"，不是 fatal（纯 Python damped-Newton/MM 可行；numpy 仅经可选 `train` group 存在）。
- **TrueSkill2 / style embedding / 多维 rating**：openskill 6.2.0 都没有；且两套设计**并未提议**这些，属预防性约束。
- **Nash averaging 作为发布尺度**：support 在 152-pair 矩阵的 bootstrap 下极不稳定（200 replicates 得 146 个不同 support，众数仅 3.5%），只能做 top-group 诊断。
- **把 joint MLE 提升为发布数字并每次 refit**：会引入 refit-drift（见 disputed）。
- **两个估计器并存但都声称"真值"**：ARCH 唯一真相原则下不可接受。

---

## 6. 明确有争议（disputed）的点

1. **"joint MLE 每次 refit 搬动旧 id = 尺子动了"（fixes-new-failures:fix:joint-mle-refit-moves-old-ids）——disputed / 降级。** 算术真实（加 D 的局会移动 A、C），但位移是 **O(1/n)、符号不稳定、且落在报告 CI 内**（20k 局/pair 时 C 从 -0.021 变到 +0.004，符号相反并趋 0）。有 pinned gauge 时 scale 是 identify 的，所以"ruler moved"描述错误；Grok 自己的判据"几乎不动"O(1/n) 满足。PFSP propensity 前提对 rating 层为假（fit 读 balanced schedule；propensity 已写 `pfsp_weights.csv`）。critic 的 fix（`merge_manifest(refit=False)` 冻结发布值）**已经是仓库现状**。→ 仅保留为"不要把 joint MLE 当冻结发布契约"的文档 guard。
2. **"单 gauge 在 top-of-scale 不可识别，绝对 sd ~70"——partially confirmed / 需收窄。** 复现 70 是 600–700 跨度场景；本仓库历史跨度（ADR-0012 平移后约 447 above RandomBot）下 one-gauge absolute sd 仅 **19–21**，与 10–20 的效应同量级。且 round-1 的 fix 本来就允许"soft prior nodes"，所以"用很吵的尺子替换旧尺子"是错误描述。应按 **forward-looking gauge-saturation 风险**处理，报告绝对不确定度。
3. **"initialize at scale origin + sqrt(σ0²+σ_parent²) 比仓库机制差"（fixes:initialize-at-origin）——partially confirmed，结论是"无新 failure"。** 仓库本就用 measured entry（`bootstrap.py:743` fork prior_mu=parent.mu；qualify 500 局 slate），critic 的建议与仓库一致；"origin seeding 更差"不成立（高 σ 入池的大 Δμ 是期望的快速校准）。目标应是 correlated random-effect 处理家族均值不确定性，而非给子节点 σ 膨胀。
4. **"版本 equating 循环/不可能"——partially confirmed / 收窄。** overlap equating 在显式可检验的 anchor-invariance 假设下能识别 offset，只是对 ADR-0007 的族比较变更不可信（**经验失败**，非逻辑循环）。应采用 `scale_id = rules_version + gauge_id`，跨版本只报 bridge slate 上的 h2h 胜率。
5. **"logistic-vs-probit 需要把 repo 改成 logistic"——被驳回为目标；仅作设计级 caveat。** ADR-0011 decision 2 刻意令 `expected_score=predict_win` 为唯一概率模型；retarget 会反转该决定。仓库保持 `predict_win` 即可。
6. **"A 的 majority-edge cycle detector 在本仓库 sparse regime 大量假阳"——partially confirmed；"sparse regime" 为假。** 仓库 `cross=games_per_anchor=100` twin games/pair、arena 60；在这些计数下完备三角假阳为 0；问题只在 close candidates（gaps ≤~30 时 5–12%）。原始 majority edge 的假阳基线是 ~25%（等强度玩家），所以"不是 intransitivity test"过强，应说"uncalibrated detector"。
7. **`requirements:both-miss-the-measured-drift` 的 fatal 评级——降为 major。** seat-phase 漂移仓库已修（`ladder.plan_games` 双座），这是**设计缺 drift taxonomy/regression gate**，不是 live bug。

---

## 7. 被驳回 / 必须收窄、不要再引用的 round-1 断言

- **"两套设计都用 proximity matchmaking"**：只对 Design A 成立；B §6 用固定 25% anchor 配额 schedule。
- **"B 的 tau=0 as-written 不可实现"**：驳回；B 自身 anchor σ>0，自洽。真实冲突是与本仓库 `(mu,sigma=0)` pin + `FitConfig`。
- **"两套都 bootstrap battles / 两套都 specify no interval"**：A 无 bootstrap；A 有 `conservative_rating` 与 SPRT，缺的是 clustered CI。
- **"SPRT accepted published Elo 被同一样本 inflate +12..+45"**：驳回（A 的公布值来自独立 anchor 层）；只保留 stopping-time winrate 上偏 + alpha spending + anchor 集选择。
- **"Gemini 用 conservative_rating 做 gate"**：驳回为已实现缺陷；仅 docstring 意图。
- **"repo design effect = 2.0× CI 低估"**：`2.04` 是 cross-order **方差** design effect；固定换座 arm ≈1.0–1.1；对 B 的 iid battle 修正应为 **1.40×** twin correlation。
- **"unregularised MLE 总因 separation 发散（fatal）"**：核心数学对，但"virtually always beats gauge"前提过强；应为 separation-guard/robustness gap。
- **"scipy MLE violation is fatal"**：降为 blocked recipe（纯 Python 可行）。
- **`context.md` 声称 `bootstrap.py` 提供 clustered bootstrap**：错误；clustered bootstrap 在 `duel.py`。
- **"Nash averaging 不是 intransitivity test"**：过强；是 uncalibrated detector。

---

## 8. 一句话裁决

- **Design A**：概念上抓对了"anchor + batch refit + intransitivity"，但 flagship 机制是**坏的**（无限循环 + 伪 refit），且整套建立在错误尺度与错误测量模型上——**不能按原文采纳**；可抢救的只有目标方向与少量思路。
- **Design B**：概念框架最接近本仓库（两层、非零和、不做守恒、多队 PL），但 **tau=0、全量 MLE、生命周期冻结** 三处与本仓库 seam 冲突——**需按 §3.4 收窄后采纳**。
- **本仓库真正需要的**：drift taxonomy ADR + rules/version tag + 用现有 `duel`/`combine_duel_seeds` 做比较与（可选）门禁 + 保持 `elo.fit_ratings` 单一 online seam。**不要**因为这两份设计就引入第二估计器、SPRT 或改 `PlayedGame`。


---

# 附：完整性审计修正（17 条）

审计结论：The adjudicated report is substantively faithful and unusually comprehensive: its central repo decision (do not add a second estimator, SPRT, or core fields; add a drift-taxonomy ADR, a rules/version tag, and use the existing duel/manifest machinery) is well-grounded in the verified findings, and its refutation/narrowing list (§7) correctly retracts the round-1 overstatements. However, as a completeness/fidelity audit it has real gaps and a few errors. Most important: three verified findings are effectively dropped — the tie/N-seat support finding (dead openskill tie branch, tie-aware likelihood, asymmetric-sigma tie test), the winner's-curse/gate-survivor→anchor finding (accepted checkpoints pinned at biased μ, rejected ones never rated), and Design B's σ<2.5-certification-is-unreachable finding — plus the σ-threshold side-effect finding and Design B's real non-stationarity (pool/training coupling). It also omits the winrate→μ σ-dependence caveat, the gate→training feedback correction, and the actionable repo gap that frozen manifest levels are never fed back as fit inputs. Its §0 severity de-duplication contradicts its own §2.1.4 and the verified de-dup; it over-claims that published values are 'already frozen' (the freeze is only within-lineage); and it presents the τ=0 σ-collapse as settled although the verified findings disagree. Smaller defects: a misattributed sign-flip number in §6.1, the unsupported '60-game history doubling' phrasing, the ADR-0006-vs-ADR-0011 citation error, overstating the repo as N-seat without the base-2 caveat, and a wrong Nash cross-reference. None of these overturn the report's bottom-line recommendation, but they leave the deliverable materially less complete than a faithful adjudication of the verified corpus should be.


## E1. DROPPED FINDING: tie / N-seat support (math:ties-and-nseat; TOP-16, major, partially_confirmed). The report only notes the `List[Tuple[str,str,int]]` draw/int mismatch (§2.1.3) and says 'N座/平局用 scores' (§2.4.2). It never carries the actual verified content: openskill's tie-averaging branch is dead code, the dead branch breaks per-player symmetry (not the sum), asymmetric-sigma ties are untested, and any batch fit needs a specified tie-aware likelihood.

**为什么重要**：This is a TOP-ISSUE major finding. Its correction changes what an implementer must add to any published estimator (tie-aware Weng-Lin rank likelihood) and which regression test is missing; omitting it leaves the design adjudication incomplete on a core capability of this card game (ties are frequent).

**修正**：Add a short tie/N-seat subsection: keep `PlayedGame(seed,seats,scores)`, derive ranks from scores, state that the dead openskill tie branch affects per-player symmetry not ΣΔμ, add an asymmetric-sigma tie regression test, and require a rank-tie likelihood for any batch fit.


## E2. DISTORTED/DROPPED FINDING: winner's curse / gate-survivor→anchor (redteam:A7 plus practice-chess:F4; TOP-13, major, confirmed). The report mentions only that gate outcome makes the anchor set a non-random subset (§2.1.4 iii) and that lineage priors correlate siblings (§6.3). It never states the core diagnosis: only passing checkpoints are rated/pinned at a winner's-curse-biased μ, rejected checkpoints are discarded and never rated (survivor-biased progress curve), and pinning makes the selection bias permanent.

**为什么重要**：This is a TOP-ISSUE confirmed major aimed at both designs. Without it the report understates why both designs' 'progress curve' is monotone-by-construction and why the fix is independent re-measurement before pinning.

**修正**：Add explicitly: rate every checkpoint on a common fixed schedule before/independently of the gate; keep rejected checkpoints as rated league opponents; only pin a checkpoint after an independent pre-registered re-measurement, not at its gate-inflated μ.


## E3. DROPPED FINDING: Design B's σ<2.5 certification is practically unreachable (redteam:B3; fatal, partially_confirmed). The report covers MLE separation (§3.1.5) and τ=0 σ dynamics (§3.1.2) but never the mechanism that the Weng-Lin update term ∝ p(1−p) stalls σ against a distant pinned anchor, so the 'σ<2.5 and N*≥40' calibration gate certifies on near-zero-information games (first σ* crossing ≈128k all-win games vs N*=40).

**为什么重要**：It is a verified fatal-level finding aimed at Design B's central lifecycle gate. Omitting it makes the report's Design-B adjudication incomplete and its §3.4 fix list miss the needed remedy.

**修正**：Add: require at least one pre-registered opponent within ~2β of the candidate before publishing an absolute μ; gate on held-out calibration (log-loss/Brier vs anchors), not σ; report frontier strength via same-deal h2h with deal-clustered CIs.


## E4. DROPPED FINDING: variance/σ thresholds (repo:both-sigma-thresholds; major, partially_confirmed). The report never addresses that Design A's σ_min duplicates the repo's τ>0 anti-collapse mechanism and cannot apply to σ=0 anchors, and that a global σ floor would silently change expected_score and placement's z·σ / stop_ci bands (rung selection is mu-only, so unharmed).

**为什么重要**：It is a repo-integration major whose correction is directly actionable (drop σ_min, keep anchor σ exactly 0, don't paste native σ*=2.5). Its absence leaves a wrong fix ('set a σ floor') unchallenged.

**修正**：Add: τ>0 is the only floor; never floor anchor σ (must stay exactly 0 per ADR-0011); express any calibration threshold in repo units derived from expected_score; do not copy openskill-native σ*=2.5 literally.


## E5. DROPPED FINDING: real non-stationarity is opponent-pool/training coupling (practice-llm:B7; major, confirmed). The report keeps B7's PFSP/IPW sub-point (§3.2) but drops the central correction: most of Design B's drift apparatus (model-version aliasing, task packs, contamination/style bias) is LLM-snapshot-specific and not this repo's failure mode; the repo's real non-stationarity is training/eval pool coupling, which pinning published μ cannot fix.

**为什么重要**：This is a high-value design correction that changes the recommended remedy (handle pool drift with fixed reference snapshots in the eval set; version rated records by rules/deal/seat), not just B's LLM framing.

**修正**：Add: identify pool/training coupling as the real non-stationarity, and recommend fixed reference snapshots in the evaluation set plus versioned (rules, deal, seat) rated records; drop reliance on freezing published μ as the drift fix.


## E6. DROPPED FINDING: actionable repo gap behind the status-map finding (code:repo-free-ids-drift-inflation; major, partially_confirmed). The report only lists 'per-id status map / rules hash' under over-engineering (§5). It drops the confirmed repo gap that `study.merge_manifest` freezes levels but never feeds them back as fit inputs/priors (so a re-fit recomputes every free id), and that a clean-sheet entrant vs the pinned gauge inflates μ without bound.

**为什么重要**：The verified correction says the actionable gap is real and should target the missing orchestration-layer mechanism. Rejecting only the core-seam version leaves the repo gap unaddressed, which matters for the implementation decision.

**修正**：Keep the rejection of a core `fit_ratings` status map, but state the orchestration-layer remedy: feed frozen manifest levels in as `Prior` centers for warm start, and flag/regularize perfect-record entrants ('≥ scale max').


## E7. DROPPED FINDING: winrate→mu gate thresholds are σ-dependent (simulate:r2-sprt-mapping-sigma-dependent; TOP-20, major, confirmed). The report only notes the +35 Elo detector and the retired 400-logistic scale. It never states that p=0.55 maps to ~17.8 mu at σ=0 up to ~39.7 mu at σ=200, so a winrate-defined gate cannot be pre-registered as a fixed effect without also fixing σ.

**为什么重要**：It is a TOP-ISSUE confirmed major. Its correction changes the gate specification the repo should adopt (define the gate in Δμ / expected_score at a declared σ).

**修正**：Add: define any promotion gate in Δμ (or expected_score at a declared reference σ) and pre-register it; report realized Δμ and both σ; never reuse the retired 400-logistic 35-Elo constant.


## E8. DROPPED CORRECTION: gate→training feedback channel (fixes-new-failures:fix:sprt-independence-misses-training-feedback; partially_confirmed). The report doesn't carry the correction that PFSP weights are computed from observed session outcomes, not gate/published Elo, so a gate-inflated rating does not reweight opponents; the actual rating→training channel is pool.trim/fork-parent selection by μ, and training stops on a data-dependent plateau (random number of gated candidates).

**为什么重要**：This prevents an implementer from 'fixing' the wrong channel and from assuming a fixed number of tests for alpha-spending. It is a correctness guard on the report's own gate recommendations.

**修正**：Add: keep league/PFSP weights from session outcomes (the repo already does); name pool.trim/fork-parent μ as the real channel; use an anytime-valid bound or a pre-registered cap because the stopping time is random; report realized false accepts from the logged candidate list.


## E9. INTERNAL INCONSISTENCY: the §0 severity de-duplication contradicts the body and the verified de-dup finding. §0 asserts 'A has 2 fatal (infinite loop, replay-not-MLE)' and 'B has 1 major integration blocker', but §2.1.4 labels the SPRT gate 'fatal-ish', and redteam:B3 (σ<2.5 unreachable) is a verified fatal. The steelman de-dup correction also says the honest result is ~2 tau failure modes (FitConfig guard vs the σ=0 ZeroDivisionError) and ~2 batch defects, not a clean 6→1 / 8→1 collapse.

**为什么重要**：The summary's headline counts are the report's main quantitative claim; contradicting its own body and the verified finding undermines the 'many fatals are double-counted' thesis.

**修正**：Reconcile §0 with §2.1.4/§3: either count the SPRT and B3 separately, or explicitly say the de-dup merges them; state that de-dup yields ~2 tau modes and ~2 batch defects.


## E10. OVER-CLAIM: 'published values are already frozen' (steelman:meta:mvp-vs-do-nothing-and-cost; partially_confirmed). §0/§4 present `merge_manifest(refit=False)` freeze as if published-value stability is solved. The verified correction says the freeze only preserves levels within a lineage (setdefault) and does not stop cross-fit/order drift (~1.42× sd), so the MVP is more than 'documentation + version tag' and 'do nothing' is only defensible if absolute levels are never compared across independent fits.

**为什么重要**：This is a high-value framing correction that directly affects the repo's implementation decision: the report should not imply the existing freeze is sufficient.

**修正**：Qualify the freeze claim: state the residual cross-fit/order drift and the precondition (no absolute-level comparison across independent fits), and include the small manifest schema change + refusal-to-pool explicitly in the MVP.


## E11. CONTESTED POINT PRESENTED AS SETTLED: τ=0 σ collapse. §3.1.2 states 'τ=0 makes σ decrease monotonically with no equilibrium' as fact. Several verified findings disagree about whether τ=0 collapses or plateaus/settles (math:tau-is-a-window reports 193→7.3 with no equilibrium; repo:grok-tau0-incompatible reports ~27 plateau; requirements:B-tau-zero reports ~69 settle). The report doesn't flag the disagreement.

**为什么重要**：This is one of the report's arguments against Design B's τ=0. If the collapse is contested, presenting it as settled overstates that part of the case and hides a genuine uncertainty.

**修正**：Flag the disagreement and rest the τ=0 rejection on the uncontested blockers (FitConfig requires τ>0; openskill raises ZeroDivisionError with any σ=0 participant at τ=0), treating the collapse as setup-dependent.


## E12. MISATTRIBUTED NUMBERS: §6.1 says '20k 局/pair 时 C 从 -0.021 变到 +0.004'. In the verified finding the -0.021 is the 1k-games value; at 20k the movement is C -0.5202→-0.5167 (+0.0035). The point (sign flip and shrinkage) survives, but the number placement is wrong.

**为什么重要**：Precision matters in a report whose thesis is that the migration is O(1/n) noise; mislabeling which sample gives which shift weakens that evidence.

**修正**：Write: 'C's shift flips sign between 1k (−0.021) and 20k (+0.004) and shrinks toward 0'.


## E13. UNSUPPORTED/OVERSTATED: §2.1.1 pairs 'history 5→200,005 rows' with '60 局重放使 history 翻倍'. The verified 60-game figure is the shuffle spread |Δμ|=3.87; the doubling is a general corollary ('each refit appends N'), not a measured 60-game doubling.

**为什么重要**：Small but it attaches a specific unsupported number to the infinite-loop claim, which dilutes an otherwise well-evidenced section.

**修正**：Cite the verified facts directly: 'each refit appends one row per replayed game (so the log grows by N per refit)' plus the 5→200,005 reproduction; drop the '60-game doubling' phrasing.


## E14. CITATION ERROR: §4/§5 attribute the `PlayedGame (seed, seats, scores)` shape to 'ADR-0006/0011'. The verified correction (steelman:meta:status-map-and-rules-hash-touch-frozen-seam) notes the shape is defined by ADR-0011/elo.py and ADR-0006 is superseded; ADR-0006 line 29 is the no-new-seam commitment.

**为什么重要**：The report's fix hinges on which ADR freezes the core, so an incorrect citation weakens the 'do not touch elo.py' argument.

**修正**：Cite ADR-0011/elo.py (and note ADR-0006 is superseded) for the PlayedGame shape.


## E15. DISTORTED FRAMING: §0/§2.1.3 describe the target game as 'N 座 + 整数分数 + 高分频' without the verified caveat (code:AB-wrong-units-display-1v1) that the base configuration and the ladder/duel/arena paths are 2-seat, with N-seat only a future capability. The real break for Design A is the missing seed/seat/scores deal structure, not seat count.

**为什么重要**：Overstating the seat count props up the 'Design A cannot represent this repo' critique with a weaker sub-argument and misleads an implementer about the current pipeline.

**修正**：State base = 2 seats (N-seat supported by the type/engine but not the current paths); center the critique on the missing seed/seat/scores and deal structure.


## E16. DROPPED MATH CORRECTION: math:anchor-pins-location-beta-pins-scale (partially_confirmed). The report gestures at identifiability (§3.2, §6.1) but never records that a single anchor pins only the location gauge while β held fixed pins the scale, nor the correction that Grok's table claim pinning anchors yields 'Σμ 有约束' is false (one anchor constrains location only).

**为什么重要**：It is a verified math correction about the design's own claim; leaving it out leaves a factual error in Design B unchallenged in the adjudication.

**修正**：Add one line: one anchor fixes location only, fixed β fixes scale; correct Grok's 'pinning anchors constrains Σμ' table claim.


## E17. CROSS-REFERENCE ERROR: §2.2 and §2.3 point the Nash-averaging support-instability claim to §6, but the report's Nash discussion is in §5; §6 is the disputed-points section and contains no Nash content.

**为什么重要**：Minor, but it prevents the reader from finding the supporting discussion and suggests section drift in the synthesis.

**修正**：Change the Nash cross-references from §6 to §5.

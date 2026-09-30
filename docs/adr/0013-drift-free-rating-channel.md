# 无漂移评分通道：漂移分类、规则身份与单一在线估计器

两套外部"无漂移 OpenSkill"方案（锚点冻结 + 批量重放；在线 OpenSkill + 全量 anchored MLE）经多轮对抗审查后，仓库裁定：`elo.fit_ratings`（Weng–Lin Plackett–Luce、单遍时序重放、`tau>0`、anchor `(mu, sigma=0)`，ADR-0011）仍是唯一的**在线**估计 seam；"全局 order-free 绝对表"另建独立纯 Python `mle.py`，**永不进 `elo.py`**；跨规则版本的 rating 一律不可 pooling。本 ADR 固定这套通道政策与漂移分类，供 ladder / duel / study / placement 共同引用，替代此前散落在 `experiments/` 报告里的口径。

## 1. 漂移分类

任何"漂移"报告必须先声明属于哪一类；不同类型的防线不同，测试也不同。

| 类型 | 机制 | 本仓库防线 |
|---|---|---|
| (i) 坐标/尺度漂移 | gauge 未定或多钉，绝对分不可比 | 单 gauge `RandomBot = 0`（ADR-0012）；只有外生冻结策略才 `sigma = 0` |
| (ii) 路径/顺序依赖 | 在线重放结果依赖对局顺序 | 在线 seam 只服务匹配/训练；比较走 order-free 的 h2h（`duel`）；发布绝对表走 `mle.py` |
| (iii) 非传递性震荡 | 1-D 标量假设失效（循环克制） | `arena.pair_diagnostics` 的 per-pair 残差诊断；**不做** Nash 发布尺度 |
| (iv) 真进步 vs 尺子 | 群体变强被误读为旧模型变弱 | 固定里程碑 + deal-twin h2h 的 Δμ + CI；绝对分不跨独立 fit 比较 |
| (v) 任务/规则变化 | 尺子**内容**变了（牌型族比较、人数、牌堆变体） | `rules_id` 身份 + 跨版本拒绝 pooling；要跨版本比只做 bridge h2h |

## 2. 决定

1. **`elo.py` 在线契约不动**：`fit_ratings` 是唯一在线估计 seam；`tau > 0`（`DEFAULT_TAU = 2`）；anchor `(mu, sigma = 0)`；`PlayedGame(seed, seats, scores)` 不加 rules/status/prior 字段。状态机、规则身份、Prior 回流一律加在 `ladder` / `study` / `placement` 编排层（ADR-0010 单一 owner）。
2. **单 gauge，禁多锚**：唯一坐标基准是 RandomBot `mu = 0`；不引入第二个 pinned 锚点、不做 soft L2 锚（λ 会把发布值连续拉开且无先验依据）。
3. **全局绝对表 = 独立 `src/seven523/mle.py`**：order-free 的 anchored probit/Thurstone MAP MLE（Gaussian/probit link），纯 Python（无 numpy/scipy/torch/IO/RNG），只由显式 tool 调用；要求 MAP/正则、separation guard（对 gauge 全胜不得发散）、有限值回归测试、tie-aware rank likelihood、Gaussian/probit link。它显式 supersede ADR-0011 decision 1 中"不建第二估计器"的部分，但**只用于发布**：发布表/manifest 的 `levels` 与排行榜来自 `mle.py`；匹配、训练、以及 placement session 自身的似然估计仍读 `elo.fit_ratings`（placement 的 `Prior` 只作参考输入与回流中心）。估计模型与边界：
   - **link**：静态 Thurstone/probit 成对似然，与 `elo.expected_score = predict_win` 同一 link（非 logistic BT），统一 ordered-probit 阈值：胜 `log Φ((d−ε)/s)`、负 `log Φ((−d−ε)/s)`、平局 `log[Φ((ε−d)/s) − Φ((−ε−d)/s)]`（所有分支共用 `±ε` 阈值，`ε` 才可识别；`ε` 可配置或联合估计）。
   - **发布完整性**：未收敛、未识别（如全平局/无平局边界）或非有限 `sigma`/CI 不得当作精确值发布；JSON 必须严格（`allow_nan=False`），非有限量以 `null` + `converged`/识别标记呈现。
   - **尺度**：齐性（homoscedastic）link，`s = √2·β` 对每一对（anchor 对亦然）成立；prior 的 `σ_i`/`σ_j` 只进 MAP 高斯正则（决定 shrinkage），绝不进似然尺度。`β` 固定即钉住 scale，anchor 钉住 location；同一数据在 `σ=38` 与 `σ=200` 下必须得到同一 link 尺度，只允许正则强度不同。旧 plug-in 约定 `s = √(2β² + σ_i² + σ_j²)` 已废弃：同一 4000 局在 `σ=200` 与 manifest `σ≈38` 下把 lvl2 分别放到 209.0 与 130.6，尺度随先验约定漂移。
   - **正则/separation**：对自由 id 加 `Prior` 高斯 MAP，保证对 gauge 全胜的估计有限；完美战绩/越界 id 显式标记，不当作精确发布值。
   - **求解**：纯 Python 阻尼 Newton（Hessian 加 ridge、回溯），返回 `Rating.sigma` 为 Laplace 对角 posterior sd；发布 CI 用按 deal(`seed`) 聚簇 bootstrap（tool 层，`random.Random(seed)` 注入）。
   - **tool**：`tools/refit_mle.py` 读一到多个逐局 JSONL，强制所有行 `rules_id` 一致且等于当前 `rules_id(DEFAULT_RULES)`，写出带 rules 身份的 JSON 发布表（`levels`/`sigmas`/`anchors`/`estimator`/CI）并打印 stdout；不触碰 `elo.py`。**发布 study manifest 是显式路径**：只有 `--manifest-out PATH --refit` 同时给出时才把拟合值写回目标 manifest——经 `study.merge_manifest(..., refit=True)` 的同一 rules 门禁替换 `levels`/`subjects`（保留每个 subject 的 `spec` 与其他无关元数据、保留 anchors）；`rungs` 按 `elo.select_rungs(count=5, min_spacing=100, max_spacing=150)` 从非锚点 levels 重选（选不满/宽间距显式报告）；`estimator` 记为 `probit-mle` + `beta`/`draw_margin`/`bootstrap`/`seed`/`games`/`source`。缺 `--refit` 拒绝；没有 `--manifest-out` 不写 manifest。T15 发布的 `traces/study`/`traces/pool10` manifest `levels`（random 0 / lvl1 **54.85** / lvl2 **120.22** / lvl3 **191.74** / lvl4 **209.01**）即由该路径产出（T15 历史值；后续 T17/w2m/T23 与 2026-09-29 `search_leafq` 联合重拟已更新发布表，现行契约以 manifest 与 [`../experiments/t17-recalibration.md`](../experiments/t17-recalibration.md)、[`../experiments/README.md`](../experiments/README.md) 为准）。
4. **规则身份**：`rules.py` 提供 `RULES_REVISION`（行为语义版本，规则/比较语义变更时必须 bump）+ 规范字段派生的 `rules_id`；manifest 与逐局结果记录（`play_games` JSONL）携带该身份；`study.merge_manifest` 与 `placement.load_opponents` 跨版本**拒绝 pooling**。无身份的旧产物按"不可比"处理（并入 T15 重测后再标）。跨版本比较只认 bridge slate 的 h2h，不做 μ 平移 equating。
5. **比较与门禁不建 SPRT**：门禁走现有 `duel.paired_duel_stats` + `combine_duel_seeds`，预注册协议为 ≥3 seed × 400 副（总 ≥1200）：CI 完全排除 0 且点估计 ≥ +10 才谈"值得行动"，≥20 Elo 的行动结论用 5 seed × 400（EPV §9）。禁止复用退役 400-logistic 的 "+35 Elo"。
6. **发布纪律**：`study.merge_manifest` 默认冻结（`refit` 显式覆盖）；排行榜带 CI；**绝不以跨独立 fit 的绝对分作结论**；人类定级结果是 `Prior`/参考值，不是发布分数；checkpoint 只有在独立重测后才钉为参考（先重测再钉），不在门禁通过的 μ 上直接冻结。
7. **Prior 回流**：冻结的 manifest `levels`/`subjects` 以 `Prior` 中心回流到下一轮 fit 做 warm start；调用方显式传入的 prior 优先。
8. **审计口径**：比较一律用 deal-twin 聚簇 bootstrap（`duel`），不用 openskill `sigma` 当标准误；平局按 PL rank 语义，批量估计必须定义 tie-aware likelihood；`mu - 3*sigma` 是时间/不确定度量，不得作门禁或排名依据。

## 3. 各级消费口径

| 消费方 | 读什么 | 误差口径 | 禁止 |
|---|---|---|---|
| 门禁（是否值得继续投入） | `duel` h2h Δ | 合并点估计 + 95% CI（Δμ + CI） | winrate/Elo 固定阈值；SPRT |
| 排行榜（绝对表） | `mle.py` order-free 表 | 带 CI 发布 | 跨独立 fit 比绝对分；把在线重放的漂移值当发布值 |
| 匹配/训练 | `fit_ratings` 在线值 | 可漂，只服务决策 | 把在线值当发布/跨时间比较 |
| 人类定级 | `placement` 的 `Prior` | 参考 + CI | 当作发布分数；跨规则版本复用对手 manifest |

## 考虑过的替代

- **SPRT 门禁**（Design A）：欠定义（无 α/β/LLR/max-games），且 A 的公布值来自独立锚点层，规模选择偏置仍在；仓库已有 deal-twin CI + 符号/胜率通道，SPRT 是第二套机器与 family-wise 风险 → 不建。
- **多锚/软 L2 锚**（Design B）：λ 无先验依据（λ=0→1e4 扫描使发布值移动 0.66 个单位量级）；ADR-0012 已裁定单 gauge → 维持。
- **把 status map / rules hash 塞进 `fit_ratings`/`FitConfig`/`PlayedGame`**：违反 ADR-0011 冻结的 core 形状与 ADR-0010 单一 owner → 编排层实现。
- **`elo.py` 内 scipy/numpy MLE**：违反 `elo.py` 纯度（无 torch/numpy/IO）→ 独立纯 Python 模块。
- **在线值直接发布 + 每局减均值**：online replay 有可复现的 level bias（真值 400/450/500 → 477.7/514.5/571.9），且非零和在 σ 相等时期望为 0，减均值是错药 → 保留绝对水平审计，删除守恒规则。
- **换 logistic link 对齐旧 Elo**：ADR-0011 decision 2 已令 `expected_score = predict_win` 为唯一概率模型 → 保持 Gaussian/probit。
- **Nash averaging 作发布尺度**：152 对矩阵 bootstrap 下 support 极不稳定（200 replicates 得 146 种 support）→ 只作 top-group 诊断。

## 后果

- 新增 `mle.py` 是第二估计器，永久维护成本；只有"全局绝对表"是硬需求才实现（本 ADR 已裁定需要）。实现与测试并入 T16。
- `RULES_REVISION` 忘记 bump 会把两个规则版本的 rating 静默合并——这是比误报更贵的错误；RULES.md 改规则时同步 bump 列入维护约定。
- 旧 manifest（`traces/study`、`traces/pool10`）无 `rules_id`，新门禁下会被 `merge_manifest`/`load_opponents` 拒绝，并入 T15 重测；这是刻意 fail-loud，不是回归。
- `duel` 成为门禁唯一通道后，`--cross>0` 的历史显著性仍按 EPV §8 处理（修复前结果不可复用）。
- `conservative_rating (μ−3σ)` 不再出现在任何门禁/排名路径；若保留展示，只作不确定度提示。

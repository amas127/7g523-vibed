# 评估分辨率设计（`evaluation-resolution-design.md`）对抗性评审

> **对应文档**：[`docs/experiments/evaluation-resolution-design.md`](./evaluation-resolution-design.md)（601 行；设计提案，未实现、未提交）。
> **口径**：`rules_id=2e36dbea44893696`（revision-3 / T17）；旧口径数字在文中单独标注。
> **方法**：合并原始红队 findings 与独立验证结论，逐条抽查一手来源（代码 / 产物 / 文档原文），并对可复算数字用只读脚本复核（`nice -n 15 .venv/bin/python`，单次 <2 分钟）。**本评审只读：未修改任何文件、未跑训练 / head_to_head / ladder 对局。**
> **严重度**：high = 会使推荐方案失效或产出错误数字 / 错误决策；medium = 结论被实质性削弱、证据不足或误导；low = 表述 / 口径 / 卫生。
> **处置图例**：ACCEPT = 全部成立；「部分成立」= 核心成立但需收窄或修正表述（ACCEPT-WITH-CORRECTION）。

---

## 0. 摘要判定

**整体判定：方案骨架可信，但当前 revision 不可按原样推进，需修订后再议。**

两阶段思路（Stage-1 anchored probit-MLE 定级 → Stage-2 带内 CRN / round-robin 分辨 → 独立 bank 直接 h2h 确认）与仓库既有接缝是对齐的：`ladder.plan_games` 已实现「每副牌双座位 + 同轮所有对共享 deal」（`src/seven523/ladder.py:241-258`），`mle.fit_mle` 是唯一发布估计器（ADR-0013 决定 3），`duel`/`head_to_head` 提供牌聚簇 bootstrap。附录 A 的推导经只读复算基本无误（A.1 功率公式、A.3 三对外推、A.4 尺度换算的算术均能对上一手数据），§4.7 的「不改 `elo.py`/`mle.py`/`duel.py`」边界也清楚。

但推荐方案有两类硬伤：

- **数字级**：预算表以 σ₄₀₀=10.14 为基准、11.1 为「保守」，而文档自己 A.3 的 revision-3 外推是 11.4–11.7（本次独立复算 11.45–11.67）；N∝σ²，推荐 N=1614 的实际 80% 设计功效只有 0.68–0.71。§4.6 的 Stage-3「3 bank × 1100 → 80% power@10」同样只在 10.14 下成立（σ=11.58 时 power≈0.70）。另，行动规则是「CI 排除 0 **且** 点估计 ≥+10」，Δ=10 时该规则的功效上界是 50%，「80% power@10」只是 CI-only 口径。
- **发布路径级**：`--manifest-out --refit` 在现有实现里必然对全部非锚 levels 重跑 `select_rungs(5,100,150)` 并整体替换 `rungs`；当前 rungs=lvl1/lvl4 的间距 102.60 只比 `min_spacing` 高 2.6 probit（文档自报 half-split 相邻差可动 8.0 probit），而 §2.3/§4.4 却保证「rungs 不因 resolution 改选 / 档位稳定」，且 `select_rungs.ok=False` 不阻断保存。这会静默改变 placement 的档位契约。

**确认问题数量**：**high 5、medium 6、low 11**。另有 2 个候选问题被独立验证反驳（见 §2）。

**最小阻塞修复集（按原案推进前必须完成）**：

1. 用 revision-3 σ₄₀₀（或先补测）重算 §0 / §3.2 / §4.2 / §4.5 / §4.6 的全部 N 与 power；Stage-3 确认预算加牌（≥1400–1700 副/bank，或 4–5 bank）。
2. 确认门槛统一回冻结的 ≥3 seed × 400（EPV §9 / ADR-0013 决定 5），或在 owner 决策下显式放宽并附分析。
3. 把「CI-only SE 验收」与「+10 行动功效」拆成两个预注册口径；Δ=10 只能承诺 ~50%，或改单侧/等效检验。
4. 对 refit 的 rung 重选给出方案：冻结 rungs，或修改 `publish_manifest` 保留 rungs + 测试。
5. 残差 / 可传递性门禁改 holdout（或 parametric bootstrap 零分布）。
6. §4.6 命令与先验口径修正（`--games A --games B`、显式 σ_prior、预算补上 refit 与确认段）。

**最关键的 3 条**：

1. 预算与确认预算都被 σ₄₀₀=10.14 低估约 26–33%（文档自己的 A.3 给出 11.4–11.7），推荐 N=1614 的实际 power 只有 ~0.69。
2. `--refit` 必然重选 rungs，与「档位稳定 / rungs 不改选」的硬约束冲突，且塌缩无门禁、验收检不到。
3. 行动判定「点估计 ≥+10」下 Δ=10 的功效上界是 50%，「80% power@10」属口径混用。

---

## 1. 确认问题

### 1.1 High

#### H1 — 预算与确认预算把 σ₄₀₀ 取成 10.14 / 11.1，与文档自己的 revision-3 实测 11.4–11.7 矛盾；推荐 N 少算 26–33%，N=1614 的实际 power 只有 0.68–0.71

**位置**：`evaluation-resolution-design.md:48-56`（§0 预算表与「保守」行）、`:137-138`（§1.3 结论）、`:243`（§3.2 τ）、`:249-251`（§3.2 预算）、`:439-447`（§4.6 Stage-2/Stage-3 命令与注释）、`:512`（A.1）、`:529-540`（A.3）。
**文档断言**：预算表以 σ₄₀₀=10.14 为基准、11.1 为「保守」（上表 ×1.2）；「与旧口径 10.0–11.1 同量级 → 旧功率表可在 revision-3 继续当设计常数使用」；Δ=10 → N=1614（总 19,368 局）；§4.6「3 bank × 1100 副 → 合并 SE≈3.5，80% power@10」。
**问题**：A.3 自己从 revision-3 数据外推得 11.4–11.7；没有一条低于 11.1，且 N∝σ²。「保守」列（11.1×1.2）仍低于文档自己的三个点估计；文档只做了「同量级可复用」的定性判断，未重算。

**验证依据**（只读复算，2026-09-27 本次评审）：

- 按 A.3 的方法复算 `runs/t17_mle/games.jsonl`（4000 行；cross 2400 / anchor 1600；400 个 distinct seed；全部 `rules_id=2e36dbea44893696`）：三对 cross 的 per-deal 半胜率 sd=0.3330/0.3183/0.3096、均值 p=0.4825/0.4075/0.3600，delta 法（`400·log10` 刻度、ddof=1）得 **σ₄₀₀=11.583 / 11.451 / 11.673**，与 A.3 表的 11.6/11.4/11.7 一致。**lvl1–lvl2 在 p≈0.5 处无需任何归一即 11.58 > 11.1**，单这一对已足以证伪 10.14/11.1 的「保守性」。若把远 pair 归一到 p=0.5，缩放方向取决于方差模型（×√(q/0.25) → 11.25/11.21；×√(0.25/q) → 11.65/12.16），但 H1 结论只依赖 p≈0.5 的那一对。
- 用文档自己的功效公式（§2.2：`SE = τ√(2/(mN))`，τ=20σ，m=4）：N=1614 时 `power=Φ(10/SE−1.96)` = **0.689 / 0.699 / 0.682**（σ=11.583/11.451/11.673）；×1.2 列 N=1937 为 0.766/0.783/0.760；真正的 80% 需 N=**2106 / 2058 / 2139**（`N=ceil(2τ²(2.8016/10)²/4)`）。
- Stage-3：3×1100 的合并 `SE=σ√(400/1100)/√3`：σ=10.14 → 3.530、power=0.809（与文档一致）；σ=11.58 → **4.033、power=0.698**；σ=11.67 → 0.692。80% 需每 bank 约 1400（简单池化）到 ~1700（含 `duel.combine_duel_seeds` 的 `max()` 保守化，`src/seven523/duel.py:256-287`）。
- 旧常数的口径自标：`head-to-head-pilot.md:3`、`evaluation-protocol-validation.md:3` 均写「数值为牌型族规则变更前口径」；EPV §5 的 σ₄₀₀=10.14 来自 pilot 三对（含 p≈0.5 的 base-vf1 +9.1、ctrl +4.2、pool4g −4.5）。
- 文档的处理（§5 风险 3「σ₄₀₀ 是外推」、§4.5.1 事后 SE 验收、§0 的 ×1.2 列）不构成修正：§1.3 仍明确宣告旧表可继续当设计常数，推荐值与示例仍取 10.14。

**建议修法**：以 revision-3 σ 重算全部预算并给区间（如以 11.6 为主、11.4–11.7 为敏感性，10.14 降为历史列）；Δ=10 的 m=4 RR 预算改为 N≈2100（总 ≈25k 局）；Stage-3 确认改 ≥1400–1700 副/bank（或 4–5 bank）并写明实际 power；若坚持 11.1/10.14，须先补一个 revision-3 多 bank 的直接 σ₄₀₀ 测量。

#### H2 — 「80% power@10」与行动规则「CI 排除 0 且点估计 ≥+10」是两个口径；Δ=10 时后者的功效上界是 50%

**位置**：`evaluation-resolution-design.md:45-46`（§0 确认）、`:170-173`（§2.2 「80% power ⇔ SE=Δ/2.8016（行动结论的口径）」）、`:350`（§4.1 判定规则 2）、`:388-394`（§4.4 示例 `power_at_10=0.80`）、`:409-412`（§4.5.1）、`:443`（§4.6 Stage-3 注释）。
**文档断言**：N=1614 / 3×1100 提供 80% power@10；§4.5.1 以 `SE_bootstrap ≤ Δ/2.8016` 为验收。
**问题**：行动结论要求「合并 CI 排除 0 **且** 点估计 ≥+10」。真实 Δ=10 时 P(点估计≥10)=0.5，故复合规则的功效 ≤0.5，与 N/SE 无关；`SE≤Δ/2.8016` 只对应「CI 排除 0」这一半条件。文档把 CI-only 的 0.8 当作行动 / 确认通道的功效（`:170` 称「行动结论的口径」，`:443` 标在确认命令上，`:389` 写入 `achieved.power_at_10`）。

**验证依据**：

- 冻结规则原文：`evaluation-protocol-validation.md:315-319`、`README.md:197-200`、`plans.md:58`、`docs/adr/0013-drift-free-rating-channel.md:29`（决定 5）均为「CI 完全排除 0 **且** 点估计 ≥+10」。
- EPV:253-260 表已实测同一问题：`+10 点估计` 规则在 Δ=10 的 power=0.310（ckpt 总体）；`+20` 规则在 Δ=20 时 power=0.474–0.480；EPV:268-269 结论「门槛卡在真值上，几乎不随 N/seed 改善」。`evaluation-resolution-design.md:172` 明确引用了这条教训，却把门槛平移到 +10 后未重做分析（全文无 TOST / 等效 / 非劣字样）。
- 只读算术：正态口径下 P(point≥10)=0.5；CI-only power（N=1614、σ=10.14）为 0.800；N→∞ 复合 power 仍 0.500；复合 80% 需 Δ≈13（理想 SE）–13.3（含 max() 保守化）。
- §4.5.1 用 CI-only 功率验收、§4.5.2 用 +10 点门槛判定，同一文档两处功率定义不同。

**建议修法**：把「筛选分辨率」（CI-only，`SE≤Δ/2.8016`）与「行动功效」（对 +10 的检验）分开预注册；行动口径若要 80%，应对 +10 用单侧 / 等效（TOST）检验，或明确「Δ=10 只有 ~50% 功效、只对 Δ≥13–15 承诺 80%」并据此改预算与 `achieved.power_at_10` 字段语义。

#### H3 — `--refit` 发布路径必然重选 rungs：§2.3/§4.4 的「档位稳定 / rungs 不改选」硬约束不成立，且塌缩无门禁

**位置**：`evaluation-resolution-design.md:179-182`（§2.3 硬约束 1/2）、`:405`（§4.4「不因 resolution 改选」）、`:466-468`（§4.8 判 ADR-0013「不冲突」）、`:495-497`（风险 8）；`tools/refit_mle.py:505-521、:531-541、:550-554`；`src/seven523/study.py:165-167`；`src/seven523/elo.py:300-341`。
**文档断言**：Stage-2 发布走 `tools/refit_mle.py --manifest-out --refit`（ADR-0013 决定 3）；`rungs` 保持现有稀疏契约，不因 resolution 改选；Stage-2 后每个已发布梯级的 `nearest_level` 不变。
**问题**：`publish_manifest` 无条件对全部非锚 `fit.ratings` 调 `select_rungs(count=5, min_spacing=100, max_spacing=150)` 并重建 `rungs`；`study.merge_manifest(refit=True)` 用传入 rungs 整体替换；没有任何保留开关；`selection.ok=False` 只打印到 stderr，不阻止 `save_manifest`。因此「同一 MLE 更新 levels」与「rungs 不改选」在现有路径上不可兼得，§4.7 改动清单也没有覆盖这条。

**验证依据**：

- `tools/refit_mle.py:508-521`（`select_rungs` + 重建 rungs）、`:533-541`（`merge_manifest(..., refit=True)`）、`:550-554`（`--manifest-out` 必须配 `--refit`，否则拒绝）；`src/seven523/study.py:166-167`（refit 时整体覆盖 rungs）；`src/seven523/elo.py:336-341`（`ok` 定义）。
- `traces/study/manifest.json` rungs=`lvl1(82.746)/lvl4(185.344)`，gap=**102.598**，仅比 `min_spacing=100` 高 2.6 probit；文档自报 half-split 相邻差半差可达 8.0 probit（`evaluation-resolution-design.md:564`）。
- 只读实跑 `elo.select_rungs`（用 manifest levels）：base → `['lvl1','lvl4']` 且 `ok=False`；lvl1+2.7 → `['lvl1']`；lvl4−2.7 → `['lvl1']`；lvl2=lvl1+100.5 → `['lvl1','lvl2']`。Stage-2 的牌数正是要在这一带移动 levels。
- prior-only id 也可进入候选：`fit_mle` 对只有 prior 的 id 返回 n=0 的 rating（实测 `{'ghost': (123.0, 30.0, 0)}`），`publish_manifest` 把全部 `fit.ratings` 写进 levels（`refit_mle.py:533`）。
- §4.5 验收 5 只检查「存活梯级的 `nearest_level`」，成员被踢出 / 替换时检不到；风险 8 只承认「是否更新 levels 的 owner 决策」，未对账 rung 重写。ADR-0013 决定 3 原文即写明 refit 路径「rungs 按 `elo.select_rungs(...)` 从非锚点 levels 重选」，与 §4.8 的「不冲突」判定直接矛盾。

**建议修法**：二选一并由 owner 签字：(a) resolution 块单独发布、levels / rungs 冻结（§5 备选路径）；或 (b) 修改 `publish_manifest` 保留 rungs（显式旗标才重选），记录 `rungs_reselected`/`grade_shift`，并加「间距跨 100 不塌缩」「prior-only 不入 levels/rungs」的测试；同时把「levels + resolution 原子发布」写进 §4.7 与验收。

#### H4 — 残差 / 可传递性门禁用同一批拟合数据做诊断（in-sample），half-split 检不出偏差；拟复用的 `arena.pair_diagnostics` 未校准且与 §1.3 的 z 口径不同

**位置**：`evaluation-resolution-design.md:32-35、:97、:139-141`（§1.3）、`:413`（§4.5.3）、`:418`（§4.5.6）、`:457-459`（§4.7）；`docs/experiments/t17-recalibration.md:107-110、:120-131`；`src/seven523/arena.py:241-266`。
**文档断言**：z≈−2.0 表明 lvl1–lvl2 表内排序未经直接证据确认；half-split「抽样误差与 CI 一致，问题不是 CI 公式错」；`|z|>2.5` 作为「只信直接对局」的门禁；`transitivity` 块复用 `arena.pair_diagnostics`。
**问题**：发布表由同一 4000 局（含 2400 cross）的一次拟合产生，§1.3 的模型预测与观测来自同一批对局，残差是 in-sample；直接证据把间距往 0 拉、|z| 被收缩，门禁会漏报真实失配。`pair_diagnostics` 的 `se` 用 `games//2`、expected 用 PL 非平局口径而 observed 平局计 0.5，docstring 自称 "a diagnostic, not a calibrated test"，且同一分子在 n=400 与 deals=200 下 z 会缩 √2。

**验证依据**：

- `t17-recalibration.md:107-109` 的发布命令对全部 games 一次拟合、无 holdout。
- 本次 leave-pair-out 只读拟合（默认 `MleConfig`、`random=0`）：lvl1–lvl2 间距 24.16→**37.46**，pred 0.5674→**0.6038**，对同一观测 0.5175 的 z −2.01→**−3.53**（n=400 二项）；lvl2–lvl3 −0.84→−1.43；lvl3–lvl4 +1.46→**+2.64**。即 in-sample |z| 会把真实失配压到 2.5 门禁之下。
- `src/seven523/arena.py:241-266`：`deals=games_played//2`（:262）、`se=√(expected(1−expected)/deals)`（:263）、expected 为 PL、observed 平局计 0.5（:227/:236）、docstring:250 自述未校准。同一 lvl1–lvl2 分子在 n=400 为 −2.01、deals=200 为 −1.42。
- 文档全文存在三种 z（in-sample observed-vs-model、pooled-vs-direct、拟复用的 `pair_diagnostics`）混用，未分别定义零假设与多重比较控制。

**部分成立（修正表述）**：half-split 对稳定偏差零功效在数学上成立，但文档并未据此声称「问题只是牌数不够」——`:139-141` 明确把问题写成「牌数不够 + 模型局部残差」，half-split 的用途是排除 CI 尺度错误；该子点只是弱化了「half-split 证明」的措辞，不构成独立负面证据。

**建议修法**：用留出 bank（A bank 拟合、B bank 算 observed-vs-model）或 parametric bootstrap 零分布；把「模型失配」与「pooled−direct 之差」定义为两个量、各自给 SE / 零假设；不要原样复用在线 PL 诊断口径（或先校准）；`|z|>2.5` 阈值预注册并对 5/15 对做 multiplicity 控制。

#### H5 — 确认门槛「≥2 独立 bank」低于冻结的「≥3 seed × 400」，§4.8「一致」自述不实

**位置**：`evaluation-resolution-design.md:45-46`（§0）、`:297`（§3.4 d1）、`:344`（§4.1 Stage 3）、`:350`（§4.1 判定规则 2「≥2–3」）、`:411-412`（§4.5.2）；对照 `evaluation-protocol-validation.md:315-319`、`README.md:197-200`、`plans.md:58`、`docs/adr/0013-drift-free-rating-channel.md:29`（决定 5）。
**文档断言**：「任何要写进结论的 ≥10 Elo 差异，用 ≥2 个独立 Deal bank 的直接 h2h 复核（EPV §9 判定规则）」；§4.8 表判「README §3 / EPV §9 判定规则 | 一致」。
**问题**：冻结口径是「≥3 seed × 400 副（总 ≥1200）」；`bank` 即 seed（文档 `:297`、`:399`）。§0 / §4.1 / §4.5.2 写 ≥2 会把现行行动判定门槛实质放松；同一文档 `:114` 与 `:473` 又写 ≥3，`:473` 还声称「一致」。k=2 的合并 SE 比 k=3 更差（EPV:168 实测合并半宽 k=2 +15%、k=3 +10%）。

**验证依据**：逐处原文核对（上述行号）；EPV:306「k=3 的 seed sd 很吵，`max()` 有 +10% 半宽偏置」；EPV:168 表；`evaluation-resolution-design.md:443` 的示例命令用 3 bank × 1100，与正文门槛规则不一致。文档只在 §5 风险 3 承认「bank 方差未实测」，未承认或处理 ≥2/≥3 的口径冲突。
**修正细节**：原 finding 中「§4.5.2 的 ≥2 用于估 `between_bank_sd`」不准确——估 `between_bank_sd` 的 ≥2 出现在 `:297`（§3.4 d1）与 `:375`（§4.3），`:411` 是最终直接确认；但「判定门槛写成 ≥2」的核心不变。

**建议修法**：统一为 ≥3 seed × 400；若确有「≥2 仅用于估 `between_bank_sd`」的用途，单独写明、行动判定仍 ≥3；若要放宽到 2 bank，作为 owner 决策并附 FPR / power 分析，不得标注「与 EPV §9 一致」。

### 1.2 Medium

#### M1 — §4.6 联合 refit / 确认命令按字面不可执行（`--games` 语法非法），且先验加宽未真正落地（部分成立）

**位置**：`evaluation-resolution-design.md:439、:443-447`（§4.6）；`tools/refit_mle.py:104-113、:143-146、:574`。
**文档断言**：§4.6 自称「只读 / 可执行草案」；联合 MLE 命令为 `--games runs/t17_mle/games.jsonl runs/resolve/stage2.jsonl`；`--prior-manifest traces/study/manifest.json` 实现「先验=已发布表（可加宽）」。
**问题**：`--games` 是 `action="append"`，每次只接一个 PATH；第二个路径会被 argparse 判为未知参数，命令直接失败。当前也不存在 `--prior-sigma`，按字面运行只能用 manifest subjects 的窄先验 σ≈5.88–6.32 拟合，与 §4.3 / §5.4 的「σ_prior=20–50」建议冲突；§4.7 却把它标为「（可选）」。

**验证依据**：

- 实跑：`nice -n 15 .venv/bin/python tools/refit_mle.py --games /tmp/nope1.jsonl /tmp/nope2.jsonl --json` → `refit_mle.py: error: unrecognized arguments: /tmp/nope2.jsonl`；`--games A --games B` 正常解析（后续报文件不存在），与工具 docstring 的用法一致。
- 实跑 `--prior-sigma 30` → `error: unrecognized arguments`；全仓 grep `--prior-sigma` 只出现在被审文档 `:456`（§4.7 列「新增 `--prior-sigma`（可选）」）与 `:486`（§5.4 风险 4）。`refit_mle.py:574` 只走 `manifest_priors(document)`，实测 `traces/study/manifest.json` subjects σ=5.881/5.950/6.135/6.319（`src/seven523/ladder.py:115-131`）。
- 其余附带事实抽查通过：四个 ckpt 路径均存在且与 manifest specs 一致；当前 `DEFAULT_RULES` 与 manifest / games.jsonl 的 rules_id 均为 `2e36dbea44893696`；`--cross 3228` 按 `ladder.plan_games`（`src/seven523/ladder.py:252-258`）=每对 1614 副双座位。

**部分成立（修正表述）**：`--prior-sigma` 的缺失是文档自认的待实现项（§4.7 / §5.4），不是新发现；未被承认的新缺陷是 `--games A B` 语法——它使「可执行草案」在 CLI 边界即失败。
**建议修法**：改为 `--games runs/t17_mle/games.jsonl --games runs/resolve/stage2.jsonl`；把 `--prior-sigma` / 先验宽度覆盖从「（可选）」升为与 `resolution` 块同一批实施的前置项，示例显式给值。

#### M2 — Stage-1 局同时进似然又作先验中心；「10k 局先验只占 ~10%」不可复算且只对应窄先验；发布 σ 不是 prior 宽度（部分成立）

**位置**：`evaluation-resolution-design.md:341`（Stage-2 列出「Stage-1 已有对局全部并入同一似然」）、`:372`（§4.3）、`:485-486`（§5.4）；`src/seven523/mle.py:51-54、:425、:923-924`；`docs/experiments/t17-recalibration.md:107-109`。
**文档断言**：σ_prior 建议 20–50；「T17 发布 σ≈6 会偏强收缩局部差异，但仍可用；本文验证 10k 局时先验只占 ~10% 权重」。
**问题**：(a) Stage-1 对局并入 Stage-2 似然的同时，其 levels 又作先验中心，同一批数据计两次；σ_prior=20–50 无先验依据、由调用方自选。(b) 「~10% 权重」全文无定义 / 公式 / 目标 N / 对应 σ，附录 B 只有「窄先验 vs 默认先验点估计差 <0.5 probit」（不同量）。(c) 发布 σ≈5.88–6.32 是默认 `Prior(500,200)` 下的 Laplace 后验 sd，不是 Stage-2 的 prior 宽度。

**验证依据**：

- 本次只读 MAP 敏感度（T17 4000 局；先验中心 = flat levels，把 lvl1 中心 +20 probit，测 lvl1−lvl2 间距位移）：σ=6 → 24.147→**18.176**（pull 5.97 probit，w≈0.30）；σ=20 → 0.73（w≈0.036）；σ=50 → 0.12（w≈0.006）。w=10% ⇔ σ≈7–8；推荐的 20–50 在 4k 局只有 ≤3.6%，10k 局更小。
- 按文档自身预算（Stage-1 4k + Stage-2 12N 局），σ=6 时 Δ=20/15/10 的间距收缩约 3.22/2.37/1.35 probit（≈6.3/4.6/2.6 Elo）。
- `t17-recalibration.md:107-109` 的发布命令无 `--prior-manifest`；`mle.py:923-924` 的 free id 默认用 `config.prior`。
- 全文「10k」仅 `:372` 一处，无实验 / 脚本 / 数据。

**部分成立（修正表述）**：「恰是被 ADR-0013:46 否定的软 L2 锚」言过其实——ADR-0013 决定 3/7（`docs/adr/0013-drift-free-rating-channel.md:31`）和 `mle.py` 均把 manifest prior 回流视为正常路径；成立的是数据二次使用与 σ_prior 无依据、示例静默用 σ≈6。
**建议修法**：明确 σ_prior 口径与权重公式（如 w=SE²/(SE²+2σ²)）、预注册目标 N 与 σ；对 Stage-2 是否排除 Stage-1 局做显式决定（或改为 non-data prior center）；示例不得静默用 σ≈6。

#### M3 — `--seeds 30,31,32` 落在 selfplay 登记「不得复用」的 29–37 区间；`[20,21,22]` / `--seed 20` 曾被列为污染 seed

**位置**：`evaluation-resolution-design.md:399`（`source_seeds:[20,21,22]`）、`:435`（Stage-2 `--seed 20`）、`:447`（`--seeds 30,31,32`）；`docs/selfplay-pool-plan.md:307、:473、:606`；`docs/selfplay-pool-review.md:292、:17-25、:78-80`；`docs/reward-alignment-plan.md:513、:524`；`docs/experiments/reward-alignment-saturate.md:51`。
**文档断言**：这些 deal bank 作为 fresh 独立 bank；`resolution.banks.source_seeds=[20,21,22]`。
**问题**：`selfplay-pool-plan.md:307` 登记「已占用（不得复用）：0–8、10–18（Wave-1）、29–37（Phase-2 复制保留）」；reward-alignment 两轮预注册要求新 deal seed 与 selfplay `0–8/10–18/29–40/100–148` 零交集且 base seed 也须 fresh。30/31/32 正落在保留区；20–28 被 stats-eval 判为 revision-1 的 selection-then-confirmation 污染 seed 并退役（`selfplay-pool-review.md:78-80`）。

**验证依据**：

- 文档行号逐处核对（如上，原文均含禁令 / 审计表述）；设计文档全文无 seed 台账 / 重叠审计字样，§5 风险清单也未提。
- 本次对 `runs/**/*.json` 全量 `seeds` 字段去重：并集 = {0–4, 10–18, 100–148, 150–158, 160–168, 220–222, 229–231, 300–302}，不含 20–22、30–32；其中 300–302 已被当前 w2m 波占用（`ps` 实测 3 个训练进程在跑）。
- `runs/archive/legacy-20260926/experiments/protoeval/zero_null/` 实际存在 `games_seed20/21/22/…`、`games_seed30–37.jsonl`（各 800 行=400 副×双座位），即 20–22 / 30–32 在旧口径归档里已被实际打过（与当前 rules_id 不同）；冲突属预注册 / 复用层面，当前尚无本口径下 29–37 的 h2h 数据。

**建议修法**：确认 bank 换成台账未登记的 fresh 段（避开 300–302）；在 `resolution.banks` 写明「runs/**/*.json + 既有计划保留段」的合并审计出处；对 20–22 与 legacy zero_null 的同牌集关系加说明。

#### M4 — `--bootstrap 4000` 联合 refit 的真实成本约 40–55 分钟单核，预算与风险清单完全未提

**位置**：`evaluation-resolution-design.md:50-54、:427、:493`；`tools/refit_mle.py:323-378、:143-146`。
**文档断言**：§4.6 命令块标注「workers=8，约 6 分钟」；§5.7「Δ=10 需 ~2 万局/带（~6 min）」。这里的 6 分钟只覆盖打局（~58 局/s）。
**问题**：`--bootstrap 4000` 是随后的串行 deal-clustered refit（每 replicate 一次 `fit_mle`），文档全程未预算；§4.3 还建议「≥1000（建议 4000）」但无时间。
**验证依据**：

- 代码：`tools/refit_mle.py:323-378` 为 `for _ in range(bootstrap): … fit_mle(…)`，无并行 / 缓存；CLI 默认 `--bootstrap 200`。
- 本次实测（nice -n 15，3×w2m 并发）：`fit_mle` 4000 局 0.139 s；24000 局 0.78 s。按 0.6–0.8 s/replicate → 1000 次 ≈10–13 min、**4000 次 ≈40–55 min**；叠加确认段后完整 §4.6 ≈45–60 min。
- 文档运行时间数字只有 `:50-54`（打局表）、`:427`（build_ladder 注释）、`:493`（§5.7）三处，均为打局口径。

**建议修法**：在 §0 / §4.6 / §5.7 写明 refit 与确认段的真实成本；迭代默认 1000–2000 次，4000 只留最终发布；或排到 w2m 波之后后台批处理。

#### M5 — Stage-2 `--games-out` 追加语义 + refit 不去重：中断后重跑会静默双计牌局（部分成立）

**位置**：`evaluation-resolution-design.md:434`（`--games-out runs/resolve/stage2.jsonl`）；`src/seven523/ladder.py:343-352、:386-390`；`tools/refit_mle.py:233-270`。
**文档断言**：§4.6 以固定路径写 Stage-2 逐局数据，作为联合 MLE 输入。
**问题**：串行路径 `open("a")`、并行路径把各 shard 以 `open("ab")` 合并（`finally` 中调用），都只追加不截断；行内没有 run id / 时间戳；`load_games` 逐行全收、无去重，`fit_mle` 也不去重；bootstrap 按 seed 聚簇，同一牌局的重复行落在同一 cluster 内并把权重翻倍，点估计被静默拉偏。文档无 `rm` / 清理 / 失败重跑指引。

**验证依据**：`ladder.py:390`（`open("a")`）、`:343-352`（`open("ab")`、`finally` 合并，docstring 自述 "appending (never truncating)… resume-friendly"）；`refit_mle.py:233-270` 无去重；行结构 `ladder.py:426-441` 无 run 字段。
**部分成立（修正表述）**：只有走到 `finally` merge 的失败才把已完成子集留在最终文件，重跑重复的是该子集；首次完整跑完再重跑才是全量双份；硬杀未 merge 时 temp shard 会被清理。该现象在 `docs/selfplay-pool-review.md:143-149`（engineering-cli-1）已有记录与修复样例（`selfplay-pool-plan.md:168、:178` 的 `rm -f` + `wc -l` 断言），但被审设计文档未继承。
**建议修法**：命令前置 `rm -f runs/resolve/stage2.jsonl` 或按 run-id 目录写入；`load_games` 加 (seed,seats,scores) 重复检测或行加 run_id；§5 补「部分失败后的重跑」失败模式。

#### M6 — 覆盖校准验收（≥5 bank、覆盖率 ≥90%）不可证伪且口径错位

**位置**：`evaluation-resolution-design.md:414-415`（§4.5.4）、`:485`（风险 3）；`docs/experiments/evaluation-protocol-validation.md:157-172`（§3.2）。
**文档断言**：「留出 ≥5 个 bank，检查发布 CI 对独立 bank refit 的覆盖率 ≥90%（二项 CI 与 EPV §3 的零假设方法一致）」。
**问题**：(a) n=5 时即使 5/5 覆盖，单侧 95% Clopper–Pearson 下界=0.05^(1/5)=0.549，「≥90%」无法被检验；要有 95% 下界 ≥0.90 需 ≥29 个全成功 bank。(b) 「发布 CI 覆盖独立 bank refit 的点估计」不是对真值的 nominal coverage：bank 估计自身有抽样误差（400 副 SE≈10 Elo，而 Stage-2 发布半宽≈3.6 Elo），覆盖率会结构性偏低；EPV §3 的正确做法是已知真值（随机零 0）下检查 CI 覆盖 0。
**验证依据**：CP 下界计算（0.05^(1/5)=0.549、0.05^(1/29)=0.902）；EPV:157-172 的校准用 40 seed 才得 92.5% [79.6, 98.1]；大小比较（400 deal SE≈10.14 vs 发布半宽 3.53）。EPV 原文是独立随机零（真值 0）的覆盖，设计文档引用时改成了「bank refit 点估计」。
**建议修法**：改用已知真值的零假设 / 仿真校准（预置真值 0 或已知 Δ）；或把 bank 数提到 ~30–70 并预注册；若坚持比较 bank refit，区间需含 bank 噪声（√(发布 SE²+bank SE²)）或比较 full-fit CI 是否覆盖 bank 的 CI。

### 1.3 Low

| id | 位置 | 文档断言 / 问题 | 验证依据 | 修法 |
|---|---|---|---|---|
| L1 | design:149-151、:277、:547-550（A.5） | 「每局对差值的 Fisher 信息 ∝ p(1−p)（HR §6.1）」→ 远锚改投 p≈0.5 增益 ×1.16–2.9 | `mle.py:14-18、:852-861` 是 probit link；probit 下 `w(z)=φ(z)²/[p(1−p)]` 给相对信息 91.8/80.4/61.2/52.6%（增益 **×1.09–1.90**），文档数字是 logit / 比分口径 | 注明通道；用 probit 权重重算 §3.3 / §3.5 的增益 |
| L2 | design:337、:357、:387、:424 | §4.2 规则：带内相邻 ≤30 probit（≈60 Elo）、跨度 ≤60 Elo；§4.6 却用 lvl1–lvl4 四档示例并套 N=1614 | `runs/t17_mle/absolute_table.json` levels 差 24.157/40.802/37.639 probit（≈47/80/74 Elo），跨度 102.6 probit（≈202–206 Elo）：相邻与跨度都违规 | 示例改成合规带，或放宽规则并给出带宽修正；预算与示例用同一带定义 |
| L3 | design:280-283、:525-527（A.2） | 「固定总牌数下相邻对比方差之和…星形 28 → 均匀完整 RR 最稳健」 | A.2 未定义边集；该量随编号变化（星心在编号端点 28、中段 24）；星形实际的 4 条设计边之和为 16，与 K/C/P 相同（本次用 Laplacian 伪逆复算） | 先定义对比 / 边集；改用到 minimax 论证（完整 RR 任意对 Var 相同且最大对比方差最小） |
| L4 | design:389-394（§4.4） | 同一 contrast 给 `delta_elo_logistic=47.3`、`se_elo=3.6`、`ci95_win_prob=[0.54,0.59]`、`power_at_10=0.80` | 由 Δ=47.3、SE=3.6 得 p=0.5677、95% 区间 **[0.5577, 0.5776]**；示例 [0.54,0.59] 半宽 0.025 ≈ 1614 独立二项朴素 CI [0.5435,0.5918]，与主推的 deal 聚簇口径冲突 | 示例字段由同一 contrast 协方差 / 同一 bootstrap 分布计算，并注明 `ci_source` |
| L5 | design:56、:243-244、:281、:527 | 「单对 h2h 达到同样 Δ 的牌数是 RR 的 2 倍」「RR 省一半总牌数」 | 只在「全部 m(m−1)/2 对都要分辨」时成立；只分辨一个目标对时 RR(m=4) 总牌数 =6N=9684 vs 单对 80% 3228，是 3 倍 | 每处显式写参照基准；单目标对场景注明 RR 不省牌 |
| L6 | design:563、:580（附录 B） | 附录 B 给出 CI 并附 `rng=random.Random(1)` 复现脚本 | 本次实跑同脚本：rng1 得 lvl1–lvl2 [−43.68,+20.87]、lvl2–lvl3 [−98.07,−33.11]、lvl3–lvl4 [−133.95,−68.63]；文档值分别对应 rng3 / rng16 / rng0（点估计不变，z 结论不受影响） | 脚本固定为真实产出这些数字的 seed，或把 CI 改为 rng=1 的输出 |
| L7 | design:13、:218、:220 | 三处 file:line 引用 | `mle.py:27-31` 是 draw margin 可识别性，s=√2β 在 `mle.py:16-18/:852`；逐局落盘在 `ladder.py:527` 起 + `tools/build_ladder.py:110-121`，非 `ladder.py:212-231`；manifest 先验读取在 `refit_mle.py:574`，非 :581-600 | 修正行号（论断本身经代码实查为真） |
| L8 | design:180、:467 | 「唯一 gauge（ADR-0012 decision 2）」；「单 gauge，禁多锚 / soft L2 无先验依据」 | ADR-0012 决定 1 才是 RandomBot gauge（`docs/adr/0012-single-gauge-and-greedy-removal.md:12`），决定 2 是删 GreedyBot（:16）；「单 gauge，禁多锚」「soft L2 锚无先验依据」原文在 ADR-0013 决定 2（`docs/adr/0013-drift-free-rating-channel.md:20`）与替代方案（:46） | 改引正确条款（实质约束不变） |
| L9 | design:296-298、:405、:456 | 只声明「沿用 `duel.combine_duel_seeds` 口径」与「deal 聚簇 bootstrap 已实现」，未对账 plans T9/T10/T11 | `plans.md:106-137`：T9（ladder/`fit_ratings` 的聚簇 SE，ADR-0011 后联合 Hessian 已退役）现状未说明是否被 `refit_mle.bootstrap_cis` 吸收；T10（`combine_duel_seeds` 的 z/confidence 不联动，`duel.py:293` 默认 z=1.96）未修，而 resolution CI 公式硬编码 1.96；全文用 95% 无实害，但应注明前置 | 加一小节 T9–T11 对账；把 confidence==0.95 写成 resolution CI 的前置条件 |
| L10 | design:488-490（风险 5）、:239-240 | `policy_seed(game.seed, seat)` 只按座位槽派生；对 sampling 策略会 twin 镜像 / 退化 | 结论对当前确定性 argmax ckpt 成立（`src/seven523/networks.py:746-748` 默认 sample=False、`policies.py:286-288` 未传 sample）；但 Stage-2 的「同牌 CRN 下在差值中约掉」与似然假设依赖确定性，风险 5 只写在「未来要做」 | 在 §4.1/§4.5 写明 Stage-2 仅适用于 argmax；引入随机策略时把 entrant-id 派生 + EPV §8 镜像校准设为硬前置 |
| L11 | design:58-61、:317 对 :470-471 | §0 / §3.5 表把 SD §9.6/§9.7 写成「解」，§4.8 细节表写「**字面冲突** / **部分冲突**」 | 同一文档两处口径不一致（来自被反驳项 resolution-design-004 的低严重度残余） | 统一为「需 owner 决策的新通道 / C-5 例外 + 新 T 项」表述 |

---

## 2. 已被反驳 / 不成立的候选问题

以下 2 条进入红队清单但被独立验证反驳（verdict 均为 refuted，置信 medium）；其残余低严重度表述已并入 §1.3。

| id | 候选问题（原严重度） | 裁决 | 反驳理由（一手依据） |
|---|---|---|---|
| `resolution-design-004` | §9.6/§9.7/§9.9 与 C-5：「不改 `elo/mle/duel` 语义即新增非改协议」不成立，文档在「已解 / 字面冲突 / 需 owner」间自相矛盾（medium） | **不成立** | 文档并未写「无冲突」：§4.8 明确把 §9.6/§9.7 标为「**字面冲突**」「**部分冲突**」（design:470-471）；同一文档多处显式要求 owner 决策（§3.2 末句 design:264-266「仍建议按 T 项走 owner 决策」、§4.8 design:472「建议 owner 决策后实施」、design:450 的 `--manifest-out --refit`「只在 owner 批准后追加」、风险 8 design:495-497）。C-5 原文（README:204 / plans:305）把「不要再改协议」映射到 SD §9.9 / plans N-17，即设计文档已标注「视 scope + owner」的那一行；T17 的 path 有 owner 参与记录（`t17-recalibration.md:91`「owner/筛选阶段接受实测间距」）。残余为 §0/:317 的「解」与细节表口径不统一 → 列入 L11。 |
| `stat-4` | band/目标对的事后选择没有 multiplicity / selection 校正；胜者上偏约 4 Elo，占 Δ=10 的 ~40%（medium） | **不成立** | 该风险文档已明确承认并引用既有条目：design:491-492 写明「m=6 时 5 个相邻对；建议预注册目标对或 Bonferroni/分层（EPV §8 已有该风险条目）」；EPV:308、`plans.md:357` Q-13、`reward-shaping-500k.md:312` 均已跟踪。方案在决策层用 EPV 认可的最强缓解：pooled Δ 只作排序 / 筛选，行动结论必须独立 bank 直接 h2h 复核（design:349-351、:411-412），因此「按 5 对里最大的行动」被规则禁止，筛选层数字不会写进结论。独立模拟（RR 相关结构）显示筛选层 H0 下 E[max Z]≈1.26×SE（≈4.5 Elo），但目标替代假设下真优胜者的选择偏差≈0.33×SE（≈1.2 Elo，真优胜者 90.4% 被选中）。残余（筛选 CI 未做 Holm / max-t 校正）是 Q-13 已跟踪的开放项，不构成新发现。 |

---

## 3. 低严重度补充与未验证项

1. **F6（环境相关的 §4.6 设备争用，low，现场可复现）**：§4.6 用 `--device cuda --workers 8` 跑 Stage-2、`--device cuda` 跑确认段；评审时 `ps` 实测 3 个 w2m 训练进程在跑（09:48 启动），`nvidia-smi` 显示 RTX 4060 8GB、显存 518 MiB、利用率 52%；`src/seven523/ladder.py:448-463` 自带「workers>1 + cuda 时每个 spawn worker 各开 CUDA context」的 OOM 警告。文档的「~58 局/s、6 分钟」来自 T17 无并发环境（`t17-recalibration.md` §6），并发下不成立。建议改 `--device cpu`（对局很小）或排期，并在 §4.6 注明设备 / 并发假设。该条随环境变化，不作为设计缺陷计数。
2. **σ₄₀₀ 的 p-归一化口径未定义**：H1 的「远 pair 归一」缩放方向取决于方差模型（乘 √(q/0.25) 还是 √(0.25/q)），文档 A.3 未写；但 H1 结论只依赖 p≈0.5 的 lvl1–lvl2 一对（raw 11.58 > 11.1），不依赖归一。建议文档显式给出所用模型。
3. **revision-3 的 between-bank 方差不可测**：T17 只有一个 4000 局 bank；`between_bank_sd`（H5、M6 的修复都依赖它）与 §4.5.4 的覆盖率没有数据支撑。§5 风险 3 已承认这一点，但未把它列为「方案落地前必须补测」的阻塞项。
4. **§4.5.6 的 `|z|>2.5` 阈值无出处**：仓库内未找到该阈值的校准记录；EPV 只有 95% CI 规则与 FPR 表。属 H4 的一部分，若要保留该数字需预注册零分布。
5. **`--prior-sigma` 的缺失是自认待实现项**：§4.7 design:456（「（可选）」）与 §5.4 design:486 已写明；它不是新发现，但在 M1 中与「示例静默用 σ≈6」合并后仍要求实施时前置。
6. **`runs/resolve/` 尚不存在**：§4.6 的示例路径目录由 `refit_mle.py:656` 的 `mkdir` 处理（`--out`），不是缺陷，仅说明草案未实地跑过。

---

## 4. 覆盖边界与无法核实的部分

**本次一手核实的范围**：

- 被审文档全部引用行（601 行通读）与 §0/§1.3/§2.2/§2.3/§3.2/§3.4/§3.5/§4.1–§4.8/§5/附录 A/B 的关键断言。
- 一手来源原文：`evaluation-protocol-validation.md`（§3.2/§5/§6/§8/§9）、`README.md`（§3）、`plans.md`（§1.2、C-5、Q-13）、`adr/0012`、`adr/0013`（决定 1–8）、`t17-recalibration.md`（§2/§3）、`structural-directions.md`（§9）、`selfplay-pool-plan.md`、`selfplay-pool-review.md`、`reward-alignment-plan.md`、`reward-alignment-saturate.md`、`head-to-head-pilot.md`。
- 代码：`tools/refit_mle.py`、`tools/build_ladder.py`（行号）、`src/seven523/{mle,elo,ladder,duel,study,arena}.py`、`src/seven523/placement/estimator.py`（引用处）、`src/seven523/{policies,networks,record}.py`（L10）。
- 产物：`traces/study/manifest.json`、`runs/t17_mle/games.jsonl`、`runs/t17_mle/absolute_table.json`（复算）、`runs/**/*.json` 的 `seeds` 并集、`runs/archive/legacy-20260926/.../zero_null/` 文件清单。
- 只读复算：σ₄₀₀ 三对（deal 级 sd / delta 法）、N/power 全表、`select_rungs` 塌缩、MAP 先验敏感度（σ=6/20/50）、leave-pair-out 残差、CLI 行为（`--games`、`--prior-sigma`）、prior-only rating、附录 B 的 rng 敏感性、星形 Laplacian 伪逆、覆盖率 CP 下界。

**无法核实 / 未做**：

- 未跑任何训练 / `head_to_head` / ladder / 大批量对局；墙钟数字在 3×w2m 并发负载下测得，仅用于量级判断（§3 第 1 条）。
- revision-3 的多 bank `between_bank_sd`、`σ₄₀₀` 的多 bank 直接测量不存在，无法核对 §4.5.4 的覆盖率与 §3.4 d1 的 `max()` 实际保守度。
- `--seeds 30,31,32` / `[20,21,22]` 与 legacy zero_null 的「同牌集」关系：本次只确认了归档文件存在与 seed 派生机制一致（base seed 前缀），未逐副重放比对（此前独立验证做过首个 deal seed 的逐位匹配）。
- 筛选用 13 个候选 / 单 seed ckpt 的中间产物（`t17-recalibration.md` §1）只核对到引用行，未核对其内部数字。
- 未验证 F6 在 w2m 波结束后的影响；也未核对 `docs/experiments/README.md` 是否会索引本评审（按任务要求不修改）。
- 评审基线：revision-3 / `rules_id=2e36dbea44893696`；文档若在评审后更新，行号与结论需重对。

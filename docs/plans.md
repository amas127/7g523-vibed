# 7鬼523 路线图：全部计划/待办/建议的单一入口

> 本文档是**路线图**，不是实验报告：所有数字与结论都引用既有报告（`file:§`），不重新计算、不发明。
> 细节以各来源报告为准；本文只做去重、排序、依赖与验收口径的统一。
> 生成于 2026-09-25（计划整理 agent；只新增本文，未改代码、未改任何已有文件/报告、未 commit）。
> [`experiments/tournament-arena.md`](./experiments/tournament-arena.md) 已定稿（2026-09-25），其结论归档进 §5；本文初稿曾把它标为 §2 in-flight。

## 0. 来源清单与完整性审计

### 0.1 来源与代号

| 代号 | 路径 | 本路线图读取的计划相关小节 |
|---|---|---|
| HEP | `docs/human-elo-plan.md` | 顶部修订节（结论/修订表/改动计划/前置/里程碑 M1–M4）+ §4（D1–D3）、§6、§8 |
| SD | `docs/experiments/structural-directions.md` | §0 优先级表、§1、§2–§5（A/B/C/D）、§6（E）、§6A（F）、§7（统一协议/分辨率/资源）、§8、§9、§10 |
| OS | `docs/experiments/observation-slimming.md` | §5（pilot 结果）、§6（影响面/兼容）、§7（结论与建议）、§3（观测 S1/S2/B+ 布局） |
| TA | [`docs/experiments/tournament-arena.md`](./experiments/tournament-arena.md)（已完成） | 工具 `tools/arena.py`、`src/seven523/arena.py`；产物 `runs/arena/` |
| ERA | `docs/experiments/elo-reliability-audit.md` | §2.3/§5.4、§6（改造建议 1–7） |
| PAA | `docs/experiments/ppo-alignment-audit.md` | §4.1、§6.3、§7（P0–P3）、§8 |
| EPV | `docs/experiments/evaluation-protocol-validation.md` | §2.4、§4、§5、§6、§8（残余风险）、§9（最终推荐协议） |
| H2H | `docs/experiments/head-to-head-pilot.md` | §3.2、§4、§5（结论与后续） |
| LRP | `docs/experiments/ladder-rerating-paired.md` | §1.2、§2、§3、§4（refit 建议/限制与 §4.4 执行记录）、§4.2 step 5 |
| HR | `docs/experiments/human-elo-10-games-research.md` | §4、§5（估计器规格）、§6、§7、§8（实现映射）、§9（验证/开放）、§10 |
| W5 | `docs/experiments/wave5-500k-report.md` | §4.2、§5（结论与下一步） |
| ALS | `docs/experiments/activation-loss-sweep-report.md` | §5（结论/限制）、§6（后续建议） |
| EXPR | `docs/experiments/README.md` | §0（结论总表）、§1（索引）、§2（PENDING）、§3（命令与判定规则） |
| DES | `DESIGN.md` | §7（训练计划）、§8（模块清单） |
| TRN | `docs/training.md` | 评估/定级/开关口径 |
| ADR-n | `docs/adr/0001`–`0014` | 约束与已决事项（见 §1.3）；0007=牌型族比较，重标定见 T15；0008=观测布局 v5；0009=v5 与牌型族成为唯一实现（v1–v4 与 `tier` 回放退役）；0010=单一 owner 接缝；0011=评分核心换成 OpenSkill；0012=单一 RandomBot 基准与 GreedyBot 退役；0013=无漂移评分通道；0014=出空即撬底（revision 3，重标定见 T17） |

二手引用（本路线图只转引上述来源对它们的一手引用，未直接通读）：`ladder-report.md`（LAD，经 SD §6 / W5 §1）、`trace-signal-report.md`（TSR，经 HEP §4 / HR §2）、`elo-breakthrough-report.md`（EBO，经 SD §0 / ALS §1 / W5 §1）。

### 0.1.1 计划/评审文档补录（2026-09-29 索引审计）

上表之外，`docs/` 根下另有成对的计划/评审文档（`*-plan.md`/`*-review.md`）、决策备忘与玩家手册；
其对应结果均已在 `experiments/README.md` §1 建索引，这里按「计划 → 结果」补齐来源映射。
未结项的开放事项（`search-config-plan.md` §5、`experiments/warmstart-adamw-1m.md` §6.1、
`experiments/evaluation-resolution-review.md` 的修订要求）尚未并入 §3/§4。

| 文档 | 状态 | 对应结果 / 说明 |
|---|---|---|
| [`event-history-plan.md`](./event-history-plan.md)、[`event-history-review.md`](./event-history-review.md) | 规划 r2；红队 41/41 已处置 | 结果 [`experiments/event-history-pilot.md`](./experiments/event-history-pilot.md)（无 ≥+10 证据、不 ship） |
| [`opponent-intent-plan.md`](./opponent-intent-plan.md)、[`opponent-intent-review.md`](./opponent-intent-review.md) | 规划 r2；红队 48 条已处置 | 结果 [`experiments/opponent-intent-probe.md`](./experiments/opponent-intent-probe.md)（G-I FAIL、不进入阶段 B） |
| [`reward-alignment-plan.md`](./reward-alignment-plan.md)、[`reward-alignment-review.md`](./reward-alignment-review.md) | 计划 r4；红队 41 条已处置 | 结果 [`experiments/reward-alignment-saturate.md`](./experiments/reward-alignment-saturate.md) 等 Tier 1 三报告 |
| [`selfplay-pool-plan.md`](./selfplay-pool-plan.md)、[`selfplay-pool-review.md`](./selfplay-pool-review.md) | 计划 r2；红队 40 条已处置 | 结果 [`experiments/selfplay-pool-diagnostics.md`](./experiments/selfplay-pool-diagnostics.md)、[`experiments/selfplay-pool-wave1.md`](./experiments/selfplay-pool-wave1.md) |
| [`v5-optimization-plan.md`](./v5-optimization-plan.md)、[`v5-optimization-review.md`](./v5-optimization-review.md) | 计划 r4；红队 35 条已处置 | 结果 [`experiments/v5-optimization-wave1.md`](./experiments/v5-optimization-wave1.md)（主端点 +2.87、止损） |
| [`post-v5-structural-options.md`](./post-v5-structural-options.md) | 决策备忘（2026-09-27） | 选项 O1–O6 对比；O4 搜索线结果见 §5 搜索 rung 行 |
| [`search-config-plan.md`](./search-config-plan.md) | 计划；P4a 已发布、P6 已落地 | `search_leafq` 已入 manifest（见 §5）；§5 开放项未排期 |
| [`human-play.md`](./human-play.md) | 玩家入口手册 | §3 对手档位强度为 `play_ladder` 注册表的同步说明 |

### 0.2 完整性声明

- **已收录**：上表每一份来源里的「计划/待办/建议/开放事项」在去重后全部进入本文的 §3（待实施）、§4（待决策）、§5（已归档）、§6（已否决）或 §9（开放问题）。
- **去重**：同一事项在多份报告出现时合并为一条并列出全部出处（例：逐局 JSONL、联合 Hessian、manifest refit、SAME_STEP、座位配对）。
- **冲突已裁定（2026-09-25 文档治理）**：§7 的 7 处冲突已按各来源报告证据全局裁定并落地（§7 现为「裁定结果」表），本路线图不再有「待统一」项；不属 §7 的争议（refit 批准、spacing、牌堆 UX）已于 §4 裁决记录落地（2026-09-25），真人招募（D-3）等仍列 §4 等待 owner。
- **交付更新（2026-09-25）**：`tournament-arena.md` 已定稿并归档进 §5；T1/T2/T3/T5/T6 结果均已回填（T5 C 为 null、T6 F 已关闭）、T13-M1–M3 已交付，§2 当前无 in-flight（T2 B0 null、T3 B1 未确认、T5 C 三效应 null、T6 F 不可分且更差，见各自现状）。
- 代码状态核对截止本整理时刻（当前 `uv run pytest --collect-only -q` 收集 859 项，2026-09-29）。文中「现状」一栏以源码 grep 为准。

## 1. 状态图例、使用说明与统一验收口径

### 1.1 图例

| 状态 | 含义 |
|---|---|
| **进行中 in-flight** | 正在被另一 agent/流程执行，本图不重复排期（§2） |
| **待实施** | 已设计/给出实现要点，尚未落地；按 §3 的推荐优先级排序 |
| **待决策/审批** | 需要 owner 决策或批准才能继续（§4） |
| **已归档** | 已完成且不应重做（§5） |
| **已否决** | 有负结果或明确不建议，不要再做（§6） |
| **开放问题** | 未排期、未成计划的观察项/残余风险（§9） |

### 1.2 统一验收口径（所有 §3 条目默认适用）

来源：**EPV §9 / EXPR §3**（2026-09-25 经独立验证统一）。

- 任何「A 优于 B」的结论必须在 **≥3 seed × 400 副牌（总 ≥1200 副）的候选对候选换座 head-to-head** 上给出，报 `mean ± 1.96·max(bootstrap SE, seed sd)/√k`（`tools/head_to_head.py --seeds 0,1,2 --pairs 400`）。
- **CI 完全排除 0 且点估计 ≥ +10** 才谈「值得行动」；**≥ +20 Elo** 的行动结论建议 **5 seed × 400（80% power）或 3 seed × 800** 复算；**< 10 Elo 一律不追**（80% power 需 ≥3200 副，EPV §5）。
- 可选分级：3×200 = 粗筛（只杀 <20）；5×400 = 确认；3×1500+ = 10 Elo 级。
- 每个训练臂的母版协议（SD §7.1，已收束）：warm start `runs/probe/base_step00696320.pt`（base680k）、seed、500k、SAME_STEP、对手 greedy（方向另有定义除外）；固定参照 = `base680k` + `runs/w5_ctrl__1__1790319698/agent.pt`；每个方向自带机制检查。上述母版/参照 checkpoint 已随旧规则模型删除，仅作历史记录；新路径登记见 T15。
- **+20 行动门槛（C-1，2026-09-25 已统一）**：禁止用「点估计 ≥ +20」判定（EPV §6：Δ=20 时 power≈50%（0.474–0.480），且几乎不随 N/seed 改善）；声称越过 +20 门槛须对 +20 做单侧等效/非劣检验，并按 EPV §5 配置功率（Δ=20、80% power ≈807 副总牌）。SD §0/§7.1.5 已同步本口径。
- **协议冻结边界（C-5）**：评估工具与统计管线冻结，除非确认 bug；本轮只统一阈值与表述（细则见 `experiments/README.md` §3；已确认缺陷 T10/T11 除外）。

### 1.3 使用说明与既有约束

- 查阅计划先看 §3/§4；写结论回填时同步更新 `README §0/§1` 与对应报告，避免第二套真相。
- 术语别名（C-3，2026-09-25 统一；完整表见 `experiments/README.md` §0.2）：**轨迹 S1** = HEP/HR 的逐墩轨迹特征（评分先验）；**观测 S1** = OS 的 103 维精简观测布局；同类加前缀——轨迹 S2（决策 regret）/ 轨迹 S3（策略一致度）、观测 S2（68 维，已否）。**裸 S1 视为歧义，不得用于新文本。**
- 已决约束（ADR，不要重开）：ADR-0001 模板动作空间（134）；ADR-0002 唯一投影接缝 `Game.view`（差分泄漏测试兜底）；ADR-0003 gymnasium+torch 忠实移植（**SAME_STEP**）；ADR-0004 花色头 `MultiDiscrete([134,4])`；ADR-0005 `Match` 唯一对局驱动；ADR-0006 `elo.py` 评分接缝 + `ladder.py` 编排 + `study.py` manifest owner，结果似然是最终权威、轨迹先验只加速（ADR-0006 后果条款）；ADR-0007 牌型族比较（`"tier"` 回放变体已随 ADR-0009 退役）；ADR-0008 观测布局 v5（v5 = 观测 S1+B0+B1）；ADR-0009 v5 与牌型族为唯一实现：v1–v4、`--obs-version`、段级列重映射与 `tier` 变体全部退役，旧 ckpt 必须重训；ADR-0010 单一 owner 接缝（`record`/spec 语法/manifest/`prior`/`metrics`/`league`/`placement/`）；ADR-0011 评分核心换成 OpenSkill Plackett–Luce（`elo.py` 的 deep seam 不变；`margin`/`window`/Hessian SE 退役，旧 `elo/se` manifest 需重测）；ADR-0012 单一评分基准：RandomBot=0，GreedyBot 策略/预设分数从仓库退役，ladder 只要求 ≥1 gauge 并把 `cross` 默认提到 `games_per_anchor`；ADR-0013 无漂移评分通道：漂移分类（坐标/顺序/非传递/真进步/规则变化）、`rules_id` 身份与跨版本拒绝 pooling、不建 SPRT（门禁走 `duel` 3 seed × 400）、`mle.py` 独立纯 Python 永不进 `elo.py`、`fit_ratings`/`tau>0`/anchor σ=0 的在线契约不变。

## 2. 进行中（in-flight）

**当前无 in-flight**（2026-09-25 结果回填后）：T1/T2/T3/T5/T6 均已完成并归档进 §5，T13-M1–M3 已交付（M4 等 D-3）；新任务见 §3。

> 更新：竞技场已交付（2026-09-25）并归档进 §5；B1 已于 2026-09-25 定稿（未确认），T6 F 同日完成并关闭，训练结构线暂无进行中项。

## 3. 待实施（按当前推荐优先级排序）

> 排序依据：SD §0 优先级表（A > B > C > F > D；E 仅组合）+ SD §2.5 的证伪逻辑（A 若 <10 Elo 则 B 升第一）+ 任务确定的统一验收口径（EPV §9）。评估工程（T9–T12）可与训练结构并行，不阻塞 A/B。

> **已收束的实验轴 T1–T6 已归档进 §5，本节不再重复**：A 奖励重构（A1 未复现、A2/γ 消融）、B0 观测增广（`warm_start_into` 旧首层拷贝 + 新列置 0，null）、B1 已出牌历史（未确认）、T4 观测 S1/v5 迁移（已落地）、C 逐局对手 + PFSP（三效应 null）、F 双塔（负/关闭）。细节、数字与回填报告见 §5 对应行。

### T7. 结构性 D：>2 家训练（未来工作；低优先，前置工程重）

| 字段 | 内容 |
|---|---|
| 目标 | 多人博弈的结构性改变；唯一产品价值在「跨 N 统一 obs 后迁回 2 家」 |
| 依据 | SD §1.5、§5、§9.8、§10 Q6 |
| 依赖 | 热启动跨玩家数分支（现为显式拒绝）→ 座席轮换（learner=`i%N`）→ N 家评估（`plan_games`/`fit_ratings`/`duel` 泛化、多人 Elo 或 Plackett-Luce）→ 跨 N 统一 obs（per-player 段固定到 max_players + seat mask）→（未来新增）`build_ladder --num-players` |
| 实现要点 | `train.py` 热启动当前对 `obs_dim` 不同直接抛 `WarmStartLayoutError`（需改为按玩家数分支）；`ladder.plan_games`、`elo.fit_ratings`、`duel.plan_duel_schedule`、`tools/build_ladder.py` 的 2 家硬编码逐项泛化 |
| 成本 | ~300–500 LOC（含评估）；3 家训练 20–40 min/臂 |
| 验收 | 先内部 N 家相对端点；**产品端点是迁回 2 家 vs `base680k` 3×400**；无跨 N 统一 obs 则只作独立研究项、不进产品优先级 |
| 现状 | 3 家 PPO 冒烟可跑（`--num-players 3`）；2 家 ckpt 热启动到 3 家被 **`WarmStartLayoutError`** 显式拒绝（2 家 161 → 3 家 182）；评估链固定 2 家；`tools/build_ladder.py` 目前**没有** `--num-players` 开关（属本项未来工作） |

### T8. 结构性 E：容量/更长预算组合（条件项，不单列）

| 字段 | 内容 |
|---|---|
| 目标 | 仅在 A/B 出现正信号后做组合确认（如 B1 的 54 维新特征需要更大容量） |
| 依据 | SD §6、§9.3；负证据：`h256_scratch` 1367.7、1M 不优于 680k、500k 六臂不可分 |
| 依赖 | A/B 正信号 |
| 成本 | ~30 min/臂 |
| 验收 | 同 §1.2 统一口径；不单列实验轴 |
| 现状 | 单列已有负证据（见 §6） |

### T9. 评估侧：配对牌局 cluster bootstrap（工具层）

| 字段 | 内容 |
|---|---|
| 目标 | 对 `(锚点, 牌)` / `(牌, 座位)` 做 cluster bootstrap，处理换座配对后的残差相关；当前 `fit_ratings` 的 SE 仍假设每局独立 |
| 依据 | ERA §6.3（第 3 条待做项）；EPV §8；LRP §1.2/§2.1（`analyze.py` 已临时做 deal 级 2000 簇 bootstrap） |
| 依赖 | 逐局 JSONL（已交付，ERA §6.2） |
| 实现要点 | 分析/工具层实现（可先固化 LRP 的 `runs/lvl_rerate/analyze.py` 做法）；与联合 Hessian SE 对比 |
| 成本 | 小–中 |
| 验收 | 已有数据校准：LRP 合并数据 deal-bootstrap sd 5.6–6.7 vs 联合 Hessian 6.1–7.1（同量级）；用于跨锚点/跨牌比较 |
| 现状 | `fit_ratings` 无 cluster 选项；rerate 分析已临时实现 |

### T10. 评估侧：`combine_duel_seeds` 的 z/confidence 联动

| 字段 | 内容 |
|---|---|
| 目标 | `confidence` 参数变化时 `z` 应联动（或显式拒绝不一致），避免非默认 confidence 错配 1.96 |
| 依据 | EPV §4 遗留说明、§8 |
| 实现要点 | `duel.py` 由 confidence 推 z 或加一致性断言；默认 0.95 输出逐位不变 |
| 成本 | 小 |
| 验收 | 手算单测覆盖不同 confidence；默认路径回归不变 |
| 现状 | **未做**（`duel.py:293` z 与 `per_seed.confidence` 不联动；日常只用 0.95 无影响） |

### T11. 评估侧：`window` 非对称交叉项近似（**已随 ADR-0011 失效**）

| 字段 | 内容 |
|---|---|
| 目标 | ~~`window` 保留对局不对称时，free-free 交叉项近似~~；**已关闭**：OpenSkill 切换删除了 `window` 与 `fit_ratings` 的联合 Hessian SE（ADR-0011），`tau` 承担动态，本项无对象 |
| 依据 | EPV §4 遗留说明；ADR-0011 |
| 依赖 | 无 |
| 实现要点 | 无（如将来需要窗口语义，应在 OpenSkill 参数上重开任务，而不是恢复旧代码）|
| 成本 | — |
| 验收 | — |
| 现状 | **关闭（2026-09-25，ADR-0011）** |

### T12. Arena 周期性定级 + PFSP 权重接口（工程支撑）

| 字段 | 内容 |
|---|---|
| 目标 | 把 arena 的一体化联盟 Elo 变成 (a) 快照周期性定级、(b) C 的 PFSP 权重来源（用联赛评分替代/补充对局胜率） |
| 依据 | TA §0/§5/§6/§10（已定稿）；`tools/arena.py`（实际状态）；SD §4.2（PFSP）；ERA §6.5（候选比较优先相对口径） |
| 依赖 | TA 已定稿（两 seed Spearman ρ=0.54 → 周期定级需多 seed）；C（T5）的实现接口 |
| 实现要点 | 按该报告结论落地；arena 已有 `--glob/--every/--last/--cross/--workers/--tb/--out/--games-out`，最小改动是「周期性运行 + 把 Elo 表喂给 PFSP 权重」 |
| 成本 | 取决于报告结论；工具本体已实现 |
| 验收 | `workers=1` 与 `>1` 逐位一致（`src/seven523/arena.py` docstring 的回归承诺）；多 seed 表盘稳定性；PFSP 权重接入后仍走 T5 的池外检查 |
| 现状 | 工具已实现并跑完 seed 0/1 pilot；TA 已定稿（`docs/experiments/tournament-arena.md`） |

### T13. 人类定级 D3（M1–M4；依赖真人标定）

| 字段 | 内容 |
|---|---|
| 目标 | 轨迹先验 + 结果似然的 **OpenSkill**（Plackett–Luce）定级（ADR-0011）：10 局输出「点估计 + CI + 最近档 + provisional」，后台继续对局直到 CI≤50 |
| 依据 | HEP 顶部修订节（M1–M4、改动计划、不做清单）；HR §5（估计器规格）、§8（实现映射）、§9（验证计划）。**默认标签基准 = T2/去收缩标定（HR §1/§5.2），不是 manifest T1**；T1 仅作历史/契约引用（refit 前带相位偏移；2026-09-25 已 refit，见 LRP §4.4），新标定与 D3 先验不得以 T1 为目标 |
| 依赖 | **真人标定是硬前置**（≥8 人、10 局定级 + 60–100 局参考局，HR §9.1）；**manifest refit 已执行（2026-09-25，LRP §4.4）**——先验仍默认 T2/去收缩标定（HR §5.2），不要改用 manifest T1 直接取数（HEP 修订节前置） |
| 实现要点 | M1 `tools/fit_trace_prior.py` + `artifacts/human-elo/prior.json`（岭回归 + 去收缩 `(a,b)` + `σ_traj(n)` 表，默认 LOLO+去收缩）；M2 `src/seven523/placement/`（包：会话状态 + `select_opponent` info/Thompson + 10 副不同牌 + 5/5 座位轮换调度 + 逐局座位写入 trace + 停止规则 + 报告）；M3 `7g523-elo` CLI（`pyproject.toml [project.scripts]`）+ `play.py` 把对手 rung id 写进 `players` 标签（现已写 `anchor:greedy@seatN` / `opponent:<id>@seatN`），支持 bot/checkpoint/NeuralPolicy；M4 真人试点。**结果似然已按 ADR-0011 换成 OpenSkill Plackett–Luce**（不再有 `margin`/`window`/Hessian SE）；轨迹 S2 逐决策 regret 模块与轨迹 S3 评分不做 |
| 成本 | M1 1 天；M2 1 天；M3 0.5 天；M4 2–3 天 |
| 验收 | M1/M2/M3：`select_opponent` Fisher/单调性单测、`fit_trace_prior` 留一等级标定回归（锁定 `(a,b)`/`σ_traj` 表）、CLI 冒烟；M4 按 HR §9.1：一半真人标定、一半评估，报 RMSE / P50 / P100 / CI 覆盖率 / 最近档命中。预期（HR §7.1/§10）：10 局 RMSE 54–72、P100 83–93%、CI ±100–133；**不要把 10 局标成 ±50** |
| 现状 | **M1 ✅ 完成（2026-09-25）**：`tools/fit_trace_prior.py` + `artifacts/human-elo/prior.json`（默认 T2 标签，sha256 `1895af98…`）与对照 `prior_manifest_labels.json`（`78e3535c…`）+ 6 项测试（`tests/test_fit_trace_prior.py`）；**M2 ✅ / M3 ✅（2026-09-25）**：`src/seven523/placement/`（包：会话编排 + `main`；全套 464 passed）、`7g523-elo` CLI + `play.py` rung 标签（+3 测试）、冒烟 `定级：1287 ± 182，最近档 greedy（1315），provisional`（`traces/sessions/smoke_m23/`），`elo.py` 零改动（sha256 `80641f13…`）；M4 真人试点**等 D-3**。结果回填 [`experiments/trace-prior-m1.md`](./experiments/trace-prior-m1.md) / [`experiments/placement-m2m3.md`](./experiments/placement-m2m3.md) |

### T14. PPO 审计剩余小项（防御性）

| 字段 | 内容 |
|---|---|
| 目标 | P1：加 `assert batch_size % num_minibatches == 0`；若未来引入 `TimeLimit`，把 truncation 与 termination 分开并 bootstrap。P2（ADR-0003 文档更新）与 P3（平台实验）已落地 |
| 依据 | PAA §6.3、§7（P1/P2/P3） |
| 实现要点 | `train.py` 断言；truncation 为条件项（当前 `env.py:204` 恒 False，不可达） |
| 成本 | 极小 |
| 验收 | 单测/防御性；不期望 Elo 收益；NEXT_STEP 事后剔除方案不要做（PAA §7 P1 不推荐） |
| 现状 | 整除断言**未加**；truncation 条件项未触发；P2 已更新 ADR-0003（现写 `[134,4]` + SAME_STEP） |

### T15. 规则变更后的重标定（牌型族比较）

| 字段 | 内容 |
|---|---|
| 目标 | 2026-09-25 牌型族比较规则落地后，把旧 `tier` 口径的强度资产在新规则下重新产出：重新生成 `traces/study`、重测 manifest、重训 ladder，并重标定 human-elo 先验与 arena/`play_ladder` 强度列。**范围已扩大（ADR-0011/0012）**：评分核心换成了 OpenSkill，唯一基准是 RandomBot=0、GreedyBot 已退役，旧 `elo/se` manifest 会被拒绝，T2 标签/`prior.json`/置信区间阈值需在同一轮重测中重标（含坐标整体 −1000） |
| 依据 | `docs/adr/0007-family-comparison.md`；`RULES.md` §3.1/§3.2、§7 R-Q12；`experiments/README.md` 顶部口径警告 |
| 依赖 | 规则修复已在工作区；bot 侧重标定无硬前置；真人 M4 仍受 D-3 门控 |
| 实现要点 | 用新规则重跑 `tools/build_ladder.py` 生成 `traces/study`；重测 manifest 契约值（沿用 refit 流程与 owner 批准，`study.py` 默认冻结）；重训/重测 ladder 梯级；重跑 arena 与 `prior.json` 标定；**按 OpenSkill 口径重标 T2 标签、`stop_ci` 与 `channel_weights`（ADR-0011）**；重训完成后把新 ckpt 路径重新登记回 `tools/play_ladder.py` 档位（`OPPONENTS` 注册表），并更新 `docs/human-play.md` 强度列与本页表格 |
| 成本 | 中–大（依赖重训预算与真人数据；命令见 `experiments/README.md` §3） |
| 验收 | 按 §1.2 统一口径；新口径数字单独成表并标明「牌型族规则后」，与旧 `tier` 数字严格隔离，禁止混比 |
| 现状 | **已实施（2026-09-26，T15 重标定完成；仅剩真人 M4/D-3）**：牌型族规则（ADR-0007）、OpenSkill 评分核心与 RandomBot=0 单基准（ADR-0011/0012）、规则身份通道（ADR-0013）均已落地。本轮产出：新池 `traces/study`（`rules_id=fbd43015d526ee72`；**2026-09-26 撬底修订后整体转 legacy，见 T17**，5 级 4400 局：random 400 / lvl1 1600 / lvl2 1200 / lvl3 800 / lvl4 400）；manifest 契约值采用 homoscedastic probit-MLE 绝对表（`tools/refit_mle.py --manifest-out --refit`，4000 局、estimator kind=`probit-mle`、source=`runs/t15_mle/games.jsonl`，ADR-0013 §3）：lvl1 **54.85** / lvl2 **120.22** / lvl3 **191.74** / lvl4 **209.01**（random=0，σ≈6.0–6.6，deal 聚簇 CI 半宽≈12–15；旧 online PL 口径 96.26/158.46/353.26/413.49 作废）；`rungs` 按 spacing 契约重选为 lvl1/lvl3（min_spacing=100 下只选出 2 级，按实测间距接受）；T2 标签按新 manifest 重导（旧 BT-MAP 值作废）；`artifacts/human-elo/prior.json` 重标（sha256 `35f7227b…`，LOLO RMSE **72.2**、m_eff **23.49**、σ(5/10/20)=**43.7/38.0/34.4**），`prior_manifest_labels.json` 数值相同（sha256 `2389f6f9…`）；`tools/play_ladder.py` 重新登记 `lvl1`–`lvl4`（强度 55/120/192/209）并更新 `docs/human-play.md` 强度列。旧资产归档：`runs/archive/study-legacy-20260926/`（旧 1200 局 study+manifest）、`runs/archive/pool10-legacy-20260926/`（旧 pool10 manifest）；旧 `tier` 与旧 online PL 数字只作历史，禁止混比。**待办**：① 真人 M4 试点仍等 D-3 招募；② ✅ placement 固定常数尺度重标与 `tests/test_placement.py` shipped-prior 同步已在 T17（2026-09-26）以 `c=0.465621` 完成（见 T17 现状行）；本行全部 revision-2 资产只作历史，当前数字见 [`experiments/t17-recalibration.md`](./experiments/t17-recalibration.md)。 |

### T16. 无漂移评分通道（drift taxonomy + 规则身份 + Prior 回流 + mle.py）

| 字段 | 内容 |
|---|---|
| 目标 | 落地 [ADR-0013](./adr/0013-drift-free-rating-channel.md)：① 漂移分类口径；② `rules_id` 身份 + 跨版本拒绝 pooling；③ 冻结 manifest levels 回流为 `Prior`（warm start）；④ 独立纯 Python `mle.py` 出 order-free 绝对表；⑤ 门禁/比较统一走 `duel`（预注册 ≥3 seed × 400），不建 SPRT |
| 依据 | [ADR-0013](./adr/0013-drift-free-rating-channel.md)；[`experiments/elo-reliability-audit.md`](./experiments/elo-reliability-audit.md)；[`experiments/evaluation-protocol-validation.md`](./experiments/evaluation-protocol-validation.md) §9；ADR-0007/0011/0012 |
| 依赖 | T15 的 manifest 重测（无身份旧产物会被新门禁 fail-loud 拒绝）；`duel`/`combine_duel_seeds` 已就绪 |
| 实现要点 | ① `rules.py`：`RULES_REVISION` + `rules_id()`（RULES.md 改规则必须同步 bump）；② `study.merge_manifest` 与 `placement.load_opponents` 的 `rules_id` 门禁；③ `ladder` 的 manifest→`Prior` 回流（显式 prior 优先）与 `play_games` JSONL 的 `rules_id`；④ 新 `src/seven523/mle.py`（纯 Python、MAP + separation guard、tie-aware、Gaussian link）与显式 tool，不进 `elo.py`；⑤ 门禁/比较接口统一到 `duel` |
| 成本 | 中；`mle.py` 是长期维护的第二个估计器 |
| 验收 | 单测：规则身份稳定/差异、merge/load 拒绝跨版本、Prior 回流、JSONL 身份、MLE 有限值与平局；旧无身份 manifest fail-loud；全套现有测试保持全绿 |
| 现状 | **全部实施（2026-09-26，597 passed / 1 failed；余 1 项 out-of-scope 同步，见下）**：ADR-0013；`rules_id()`（当时 DEFAULT_RULES worked example `fbd43015d526ee72`；撬底 revision 3 后现为 `2e36dbea44893696`，旧值转历史，见 T17）、`merge_manifest`/`load_opponents`/`tools/build_ladder` 跨版本 fail-loud、`manifest_priors()` 回流（显式 `--prior` 优先）、`play_games` JSONL 带 `rules_id`；④ `src/seven523/mle.py`（统一 ordered-probit 阈值的 anchored MAP probit MLE、homoscedastic `s=√2·β`、稳定二阶导、`margin_identified`/`converged`、非有限 σ=inf）+ `tools/refit_mle.py`（rules 门禁、deal 聚簇 bootstrap、严格 JSON，非有限值 `null`+标记；`--out` 与 `--json` 逐字节一致；`--manifest-out --refit` 经 `merge_manifest` 发布并重选 `rungs`，无该 flag 不写 manifest）。旧 `traces/study`/`traces/pool10` 无身份会被新门禁拒绝（预期，并入 T15 重测）。**尺度问题已决（option a 采纳，2026-09-26）**：prior 的 σ 只进 MAP 正则、绝不进似然尺度——同一 4000 局在中性先验与 `--prior-manifest` 下共享 `s=√2·β`（中性 54.85/120.22/191.74/209.01 vs PL-prior 对照 60.33/126.25/199.80/217.66，差异只剩 shrinkage；全平局 + prior σ≈1e308 的 `draw_margin` 现为有限 tie-rate 值并标记 `margin_identified=False`/`converged=False`）；T15 manifest 已由 `tools/refit_mle.py --manifest-out --refit` 重发布为该表（`estimator` kind=`probit-mle`、`source=runs/t15_mle/games.jsonl`、games=4000），T2 标签、`prior.json`（sha256 `35f7227b…`）与 `play_ladder` 强度同步重导；ADR-0013 §3 已记录。placement 固定常数（COLD_START_PRIOR/RESULT_PRIOR/ANCHOR_CENTER/RUNG_PRIOR_SD）的尺度重标与 `tests/test_placement.py` 的 shipped-prior 常数同步**已由 T17（2026-09-26）以 `c=0.465621` 完成**（(500,300)→(232.8105,139.6863)、RESULT_PRIOR (500,200)→(232.8105,93.1242)、ANCHOR_CENTER 230→107.0928、RUNG_PRIOR_SD 30→13.9686；`prior.json` 随 `prior.py` 重新生成，shipped-prior 断言改为读产物 `sigma_traj`），见 T17 现状行与 [`experiments/t17-recalibration.md`](./experiments/t17-recalibration.md) §4；T16 的「全套现有测试保持全绿」验收已随 T17 完成（621 passed）。 |

### T17. 撬底修订后的重训与重测（出空即撬底，revision 3）

| 字段 | 内容 |
|---|---|
| 目标 | 落地 [ADR-0014](./adr/0014-on-empty-digs.md)（RULES.md 4.8/4.9、§7 R-Q13，2026-09-26 用户裁定）：出空即撬底、赢墩不再是前提；随后把 revision 3 下的强度资产全部重产：重生成 `traces/study`/pool、重测 manifest、重训 checkpoint、重标 `prior.json` 与 `play_ladder` 强度 |
| 依据 | [ADR-0014](./adr/0014-on-empty-digs.md)；RULES.md §4.8/§4.9/§7 R-Q13；CONTEXT.md「撬底」；ADR-0013 的 `rules_id` 门禁 |
| 依赖 | 规则/引擎/回放改造已落地（现状栏）；重训预算；真人 M4 仍受 D-3 门控 |
| 实现要点 | ① `game.py` 立即结束 + 补牌角落（`GameState.empty_order`）；② `rules.py` `RULES_REVISION=3`（`rules_id` `fbd43015d526ee72` → `2e36dbea44893696`）；③ `trace.py` `TRACE_VERSION=3` + `rules.revision`，`play.replay_trace` 拒绝旧版本/旧 revision；④ 用新引擎重跑 `tools/build_ladder.py` 生成新 `traces/study`，重跑 `tools/refit_mle.py --manifest-out --refit` 发布 manifest，重跑 `tools/fit_trace_prior.py` 生成 `prior.json`，重训 ladder 并把新 ckpt 登记回 `tools/play_ladder.py`/`docs/human-play.md` |
| 成本 | 中–大（依赖重训预算与真人数据） |
| 验收 | 引擎/规则/回放单测全绿（观测维度、动作空间、`View` 逐位不变）；新产物一律带 `rules_id=2e36dbea44893696`；`placement.load_opponents` 拒绝旧 `fbd43015d526ee72` manifest；旧 ckpt/轨迹不得与新数字混比 |
| 现状 | **已实施（2026-09-26，revision-3 全链重产完成）**：ADR-0014、`RULES_REVISION=3`、`TRACE_VERSION=3`、`GameState.empty_order`、立即撬底与补牌角落、旧 trace 拒绝；引擎/规则/回放单测全绿。本轮重产：8 个 revision-3 训练 run（`runs/t17early2k/4k/8k`、`runs/t17pool__{1,2,3}`、`runs/t17self`、`runs/t17long`，2k–2M 步；矩阵见报告 §0）→ 13 个去重候选 18200 局共享筛选（`runs/t17_screen/`：顶部 1M/2M/1.5M/512k/499k 平台 187.6–191.6，无 ≥200、无 ~55；self_500k 对发布 lvl4（l1_1M）合并 9 seed 显著更强 +16.62 Elo（p=2.8e-5）、对 l1_2M 打平，无过拟合证据；self_500k 未进发布梯级，若要以它作顶档需重新测量发布）→ 发布 4000 局 study `traces/study`（`rules_id=2e36dbea44893696`；lvl1–lvl4 = e1_2k/e1_8k/p1_65k/l1_1M，每档 n=1600，random 仅作对手）与 manifest（`tools/refit_mle.py --manifest-out --refit`，estimator kind=`probit-mle`、source=`runs/t17_mle/games.jsonl`、draw_margin=15.613、200 bootstrap）：**lvl1 82.75 / lvl2 106.90 / lvl3 147.71 / lvl4 185.34**（random=0，σ≈5.9–6.3，相邻 CI 不重叠；`rungs` 按 100–150 契约重选为 lvl1/lvl4，间距 102.60）；`traces/pool10/manifest.json` 逐字节镜像（sha256 `c8c491e3…`）。T2 标签按新 manifest 重导；`prior.json`（sha256 `abfea8e8…`，LOLO RMSE **42.6**、m_eff **68.07**、σ(5/10/20)=**26.0/22.0/20.5**）与 `prior_manifest_labels.json`（sha256 `3cdf71b3…`）重跑，`data.n_levels=4`；`tools/play_ladder.py`/`docs/human-play.md` 登记 lvl1–lvl4（强度 83/107/148/185）。**T15 遗留的 placement 常数重标已补做**：`c=0.465621`（新标签对旧 online-PL 参考的过原点最小二乘；残差 +37.9/+33.1/−16.8/−7.2），COLD_START_PRIOR (500,300)→(232.8105,139.6863)、RESULT_PRIOR (500,200)→(232.8105,93.1242)、ANCHOR_CENTER 230→107.0928、RUNG_PRIOR_SD 30→13.9686，`tests/test_placement.py` shipped-prior 断言改为读产物。**T17-F1**：`study.merge_manifest` 非 refit 合并不再覆盖已发布 estimator 块、不再刷新冻结 subject 的 sigma/games（新增测试）。旧 revision-2 资产归档：`runs/archive/study-legacy-rules2-20260926/`、`runs/archive/pool10-legacy-rules2-20260926/`；全部旧数字只作历史，禁止混比。全套测试 621 passed；详见 [`experiments/t17-recalibration.md`](./experiments/t17-recalibration.md)。**后续（2026-09-27，w2m/T23 重排）**：10 级池（T23 冠军 `ws_s2`、三条 w2m 2M 续训臂、`pself_s2`、`lvl1`–`lvl4`，18,000 局共享 probit-MLE，bootstrap 4000）经 `tools/refit_mle.py --manifest-out --refit --keep-rungs` 发布进 `traces/study/manifest.json`（=`traces/pool10/manifest.json`，sha256 `9bdb0edf…`，rungs 冻结 `lvl1`/`lvl4`，tail_gap 3.77）：lvl1–lvl4 = **84.68/113.60/134.40/187.72**，冠军簇 185.96–191.49；T2 标签与 v1 `prior.json` 随之重标（sha256 `7e1c41ab…`）；同日先验升级为 **v2/R1**（[`experiments/prior-opponent-correction.md`](./experiments/prior-opponent-correction.md)）：语料 `traces/study10`（39,960 条 9 级全 RR）、二次特征展开 + cell 惩罚 λ=3，发布 `prior.json` sha256 `15fddff1…`（LOLO RMSE **52.9**、m_eff **43.28**、σ(5/10/20)=**26.8/20.7/18.0**、cell_drift² 67、drift_sd 7.79）、`prior_manifest_labels.json` sha256 `0f331414…`；独立发牌 bank2（40k、相邻为主）transfer 复核 **CONFIRMED**（v2/v1 RMSE **53.20/58.30**、σ(5/10)=**24.61/17.69** vs 27.33/19.60、drift 6.29 vs 9.05，[`experiments/prior-v2-confirmation.md`](./experiments/prior-v2-confirmation.md)），v1 4k 归档 `runs/archive/prior-v1-4k-20260927/`；placement 固定常数不重折（与 T17 同处 probit-MLE 刻度）；`play_ladder` 登记 lvl1–lvl4（85/114/134/188）+ 顶部平台簇 `ws_s2`/`pself_s2`/三条 w2m 臂（186/187/190/190/191，可直接 `play <id>`）；旧 T17 manifest 归档 `runs/archive/*-t17-mle-20260927/`；报告 `runs/w2m/calibration/report.md`；全套测试 775 passed。 |

**条件项（C-2，未触发不排期）**：ALS §6.2/§6.4 的 PPO 剩余对照——`lr_1e4` 轻量复测、`--num-minibatches 8`、Adam `eps` 对照、`loss_vf1` h2h 复测。
- **触发条件**：A/B 线证伪后（T1 的 A1/A2 与 T2/T3 的 B0/B1 在 3×400 h2h 上均 CI 跨 0 或点估计 < +10），为排除「优化受限」才排期；未触发前维持 §6 N-1。
- **存放位置**：实验与结果回填 `docs/experiments/activation-loss-sweep-report.md` §6 增补节（或新建 wave6 报告）；Adam eps 的 CLI 小改动进 `train.py` 并在该报告记录；计划状态只在本 T14 维护。

## 4. 等待决策/审批

| # | 决策项 | 背景与数字 | 需要谁 / 依据 | 阻塞/替代 |
|---|---|---|---|---|
| D-1 | **manifest refit** | 换座配对 5 seed 复测：lvl4 旧 1489.4 → 合并 **1429.7 ±7.1**（高估 ~60）、lvl1 1146.3（低估 ~26）；步骤见 LRP §4.2（多 seed → combined fit + 不确定性 → 写 scratch study → 批准同步 → spacing 另案） | **D1/study owner 批准**（LRP §4.2 step 4；ERA §5.4；`study.py` 默认冻结） | ✅ **已执行（2026-09-25，LRP §4.4）**：备份 `runs/archive/manifest-frozen-2026-09-25.json`；后续 D1 重标定（`trace-signal-report` 数字重跑）依赖 D-3 |
| D-2 | **spacing 契约如何处置** | 实测合并间距 **73 / 144 / 66**（旧 113/126/130）；`select_rungs` 5/5 seed 只选出 lvl1、lvl3 两个 rung，`ok=False`；D1 需要的是单调等级而非均匀 100–150 | 方案：接受实测间距 / 补测一个 ~1290 的 rung 填缝；**不要为满足 spacing 调标签**（LRP §4.2 step 5） | ✅ 已按此写入 refit manifest（`rungs` = lvl1/lvl3，LRP §4.4） |
| D-3 | **真人试点招募** | ≥8 人、覆盖 1000–1600；每人 10 局定级（冻结牌堆 + twin 座位 + 自适应选档）+ 60–100 局参考局；一半标定一半评估 | study owner / 产品（HR §9.1；HEP 里程碑 M4） | 阻塞 D3 上线与真人 OOD 校正 |
| D-4 | **观测 S1 103 是否实施、何时** | OS 结论：值得做但优先级低于增广；与 B 的组合布局（观测 S1+B0=106、观测 S1+B0+B1=161）需一次定稿 | 工程/训练 owner（OS §7；本文件 §3 T2–T4） | ✅ **已裁定并实施（2026-09-25）：选观测 S1+B0+B1 = v5 = `119+21n`（2 家 161）为默认**；v4/v1–v3 与热启动重映射随后在架构清理中退役（[ADR-0009](./adr/0009-single-observation-and-comparison.md)） |
| D-6 | **10 局牌堆 UX** | **✅ 已裁定（2026-09-25）：选 (b)**；代价：保留座位相位残差，10 局精度可能略差于研究 (a) 口径；human-elo 计划 M2 需按轮换调度更新（背景：冻结 twin 相位干净但只见 5 副；轮换体验好但留残差） | 产品/研究（HR §6.3；HEP 修订节）；owner 2026-09-25 裁定 | 阻塞 M2 调度与真人试点设计：M2 按 (b) 更新 |
| D-7 | **`rank_counts` 118 维可选臂** | 仅当 500k 追认出现样本效率问题时回退；观测 S1 已证不必要 | 工程（OS §7.1） | 不阻塞 |

> D-5（统一判定阈值）已于 2026-09-25 按 EPV §6/§9 证据裁定并回填（见 §7 C-1）；不再属待决，原编号保留以免交叉引用悬空。
>
> **裁决记录（2026-09-25，owner）**：D-1 ✅ **已执行（2026-09-25）**（按 LRP §4.2 分阶段：scratch candidate → 备份冻结 manifest → 覆盖，见 LRP §4.4）；D-2 ✅ **接受实测间距 73/144/66**（不为凑 spacing 调标签）；D-3 ⏸ **暂缓**（真人招募；D3 上线顺延）；D-4 ✅ **已裁定并实施（2026-09-25）**：默认切 v5 = 观测 S1+B0+B1 = `119+21n`（2 家 161）；随后的架构清理把 v1–v4、跨版本热启动重映射与 `tier` 回放变体一并退役（[ADR-0009](./adr/0009-single-observation-and-comparison.md)）；D-6 ✅ **(b)**（10 副不同牌 + 5/5 座位轮换；选项与代价见下）；D-7 ✅ **关闭：坚持观测 S1 103 维**（`rank_counts` 118 仅作 500k 追认的应急回退）。

> **D-6 说明（2026-09-25 已裁定 (b)）**：10 局真人定级的牌堆/座位调度。(a) **冻结牌堆 twin**：5 副牌 × 双座位 = 10 局；座位相位在牌内抵消，评分干净；代价是玩家会重复看到同样 5 副牌。(b) **10 副不同牌 + 5/5 轮换**：体验更自然，但留座位相位残差（审计实测跨序 sd≈√2×SE，10 局下不可忽略）。(c) **双模式**：体验局用轮换牌、计分局只用冻结 twin，评分标 provisional。研究口径（HR §6.3）倾向 (a)。**owner 裁定 (b)**：保留座位相位残差，10 局精度可能略差于 (a) 口径；human-elo 计划 M2 需按轮换调度更新。

## 5. 已完成（归档，防止重复）

| 已完成事项 | 依据（file:§） |
|---|---|
| PPO P0：`SAME_STEP` + `final_info.episode` 适配 + 回归测试 | PAA §4.1/§7；`train.py:421,213`；ADR-0003 后果 |
| 评估根因修复：候选内换座配对（每副牌双座位、`games_per_anchor` 偶数、顺序不变） | ERA §6.1；EPV §0/§2.2；`ladder.py`；DES §8 |
| 联合 Hessian SE + free-free 重复计数修复（cross SE 最多低估 19% → 修后与数值 Hessian 一致） | ERA §2.3；EPV §4；`elo.py`；DES §8 |
| 逐局战绩 JSONL（`--games-out`，按 seed 分文件，可复算 twin/合并） | ERA §6.2；EPV §2.2；H2H §5（后续第 4 条） |
| 候选对候选工具：`duel.py` + `tools/head_to_head.py` + `tools/h2h_screen.py`（同牌换座、按牌聚簇配对 bootstrap） | H2H §2；EPV §2.1/§2.3 |
| 多 seed 合并公式 + 分辨率表 + FPR/power 蒙特卡洛 | H2H §3–§4；EPV §2.4/§5/§6 |
| 激活/损失/LR 扫描（wave4，12 run）：全部不赢；tanh 显著更差 | ALS §5 |
| wave5 统一 500k 六臂：全部不可分；父模型跨 seed sd=37、父子偏移翻符号 | W5 §4.2/§5 |
| 结构性 A / T1：奖励重构（`reward_shaping` 四模式 + A1/A1γ1/A2 各 500k + 4 组 h2h + 机制探针 + seed 2/3/4 复现 + D0/D2 判别；seed 1 正信号未复现、A 线收束） | [`experiments/reward-shaping-500k.md`](./experiments/reward-shaping-500k.md) §1–§7；SD §2.5 |
| 结构性 B / T2 B0：观测增广 3 维 + 版本化兼容层（191→194，测试通过；500k null、h2h 均跨 0；兼容层当时保留，后随 [ADR-0009](./adr/0009-single-observation-and-comparison.md) 退役） | [`experiments/observation-augmentation-b0.md`](./experiments/observation-augmentation-b0.md) §1–§7；SD §3.2/§3.4 |
| 结构性 B / T3 B1：已出牌历史 + `unseen` + `last_player`（v3=v2+55、2 家 249；当时 355 passed；500k 499,712 步；主端点 h2h vs `base680k` +4.20 [−6.75,+15.15] 跨 0 → 未确认；B 线收束） | [`experiments/observation-augmentation-b1.md`](./experiments/observation-augmentation-b1.md) §1–§8；SD §3.2–§3.5 |
| 观测布局 v5/v4 落地：默认 v5 = 观测 S1+B0+B1 = `119+21n`（2 家 161），v4 = 观测 S1+B0 = `64+21n`（2 家 106）为逐位前缀；v1–v3 兼容；段级列重映射热启动与不可映射方向拒绝（v4/v1–v3 与重映射后随 [ADR-0009](./adr/0009-single-observation-and-comparison.md) 退役，v5 成为唯一布局） | [ADR-0008](./adr/0008-observation-layout-v5.md)；OS 实现分析 §1–§4 +「落地记录（2026-09-25）」 |
| 结构性 C / T5：逐局对手 + PFSP + 强成员（`EpisodeMixturePolicy`/`pfsp_weights`/`--pfsp*`，+17 测试；三臂 500k `runs/t5_*__1__1790332696`；8 个 h2h 3×400 CI 全含 0、PFSP +1.88、强成员 +2.90 → 三效应 null） | [`experiments/opponent-distribution-500k.md`](./experiments/opponent-distribution-500k.md) §1–§8；SD §4 |
| T13-M1：轨迹 S1 先验 `tools/fit_trace_prior.py` + `artifacts/human-elo/prior.json`（T2 默认：a=−890.92、b=1.7122、σ(5/10/20)=80.6/69.6/64.6、LOLO RMSE 133.9；manifest 对照；6 项测试） | [`experiments/trace-prior-m1.md`](./experiments/trace-prior-m1.md)；HEP 里程碑 M1；HR §5.2；ADR-0006 |
| 结构性 F / T6：独立 actor/critic 双塔（`arch`/跨架构热启动/`--arch`，+13 测试、464 passed；三臂 500k `runs/t6_*__1__1790334485`；h2h 主臂不可分、交互臂 −15.51 vs A1、从零 −20.89 vs `w5_scratch`；梯度 cos +0.049、critic 仍塌缩 → 关闭该线） | [`experiments/twin-towers-500k.md`](./experiments/twin-towers-500k.md) §1–§8；SD §6A.5 |
| T13-M2/M3：10 局定级会话 + `7g523-elo` CLI（`placement/` 包；全套 464 passed；`play.py` rung 标签 +3 测试；`elo.py` 零改动；冒烟 1287 ± 182/最近档 greedy；M4 等 D-3） | [`experiments/placement-m2m3.md`](./experiments/placement-m2m3.md)；HEP 里程碑 M2/M3；HR §5/§8 |
| 搜索 rung 发布（2026-09-29，P4a `search_leafq`）：t=5/K=32/C=6、`t_leafq` value 头与 raw 10 级进**同一** 26,000 局 joint probit-MLE（ADR-0013）；`search_leafq` μ=**260.03±3.86**（CI [251.22,269.89]）、raw 顶 `w2m_ctl` 191.49→187.46；发布进 `traces/study`(=pool10) manifest 第 11 级（`refit_mle --spec` 写 spec、`--keep-rungs` 保 lvl1/lvl4、`rules_id` 不变）；消费侧 can-build 能力门 + 注入 factory；C3 调度开放。**P6 合并定级池并删实验模式**：`subject.search_config` 成为搜索身份单一来源（`--search-config ID=@file.json` seed/refresh，refit 默认保留）、`rated`/`unrated` 路径删除、轨迹先验按局排除（打到搜索 rung 的局记 `prior_off_reason`，raw 局照常） | [`search-config-plan.md`](./search-config-plan.md) §4.3（含文首 P6 更新）；[`experiments/README.md`](./experiments/README.md) §0 第 14 条；`artifacts/search-rung/search_rung.json`（provenance） |
| 架构清理（2026-09-25，phase 2–7b）：单一观测 v5 + 单一牌型族比较（ADR-0009；v1–v4/`--obs-version`/段级重映射/`tier` 变体退役），单一 owner 接缝（ADR-0010：`record` 批量录制、`policies` spec 语法、`study` manifest writer、`prior` 核心上移、`metrics`/`league` 拆出 `train`、`ladder.play_games` 分片接口、`placement/` 包拆分）；`train.py` 877→668 行；全套 464 passed | [ADR-0009](./adr/0009-single-observation-and-comparison.md)/[ADR-0010](./adr/0010-single-owner-seams.md)；DES §1/§8 |
| 评分核心换用 OpenSkill（2026-09-25，ADR-0011）：`elo.py` 的深接缝不变（`fit_ratings`/`select_rungs`/`expected_score`，纯函数），手写 BT-MAP/阻尼牛顿/`window`/分差似然/Hessian SE 退位给标准 Plackett–Luce（1500/200 标度、N 家、`tau` 动态）；`Rating(mu/sigma/n)`、`Fit(ratings/games)`、manifest/报告改 `mu/sigma` 键，旧 `elo/se` manifest 拒绝重测；全套 449 passed（`test_elo` 重写，含闭式高斯与库内更新对照） | [ADR-0011](./adr/0011-openskill-rating-core.md)；DES §1/§2.7/§2.8 |
| 单一评分基准与 GreedyBot 退役（2026-09-25，ADR-0012）：RandomBot 钉死 `mu=0`（坐标整体 −1000）、GreedyBot 类/`greedy` spec/训练评估默认/锚点/工具注册表全部移除，测试侧改用 `tests/support.py:FirstLegalBot`；`build_ladder` 只要求 ≥1 gauge、`cross` 默认= `games_per_anchor`；全套 448 passed | [ADR-0012](./adr/0012-single-gauge-and-greedy-removal.md)；DES §1/§2.5/§2.7/§7 |
| 观测精简静态审计 + 敏感性消融 + 200k 从零 pilot（观测 S1 不损失、观测 S2 落后） | OS §2–§5 |
| 新评估口径独立验证（座位配对/聚簇 bootstrap/√N/功率） | EPV §0/§5/§6/§9 |
| 锦标赛竞技场（W=1 vs W>1 逐位一致；6 worker 22,620 局 37.8s；bf16 实测无收益已移除；两 seed 排名 ρ=0.54 → 单 seed 排名不可靠；进度曲线 lvlbase ~184k 后平台、lvlsp 全程平） | TA §0/§2/§5/§6 |
| 人类 ≤10 局定级研究（轨迹 S1 去收缩最强、轨迹 S2/S3 不进生产、RMSE/局数外推） | HR §4/§7/§10 |
| 冻结梯级换座配对复测（5 seed × 400 局/锚点，lvl4 −60、lvl1 +26、排序保持） | LRP §2 |
| 结构性现状一手量化（奖励 4.17% 非零、早期 EV 0.065、24.7 张不可见、逐决策池、>2 家断链） | SD §1/§11 |
| >2 家支持度核实 + 3 家 PPO 冒烟（2→3 热启动报错已定位） | SD §1.5 |
| D1/D2：`tools/measure_trace_signal.py` + `trace-signal-report.md` | HEP §4（D1/D2）；TSR §4（经 HR §2） |
| M2 机器人阶梯（lvl1–lvl4 训练/评级/manifest 冻结） | LAD §2（经 LRP §2 引用）；DES §8 |
| `--cross>0` 联合 Hessian 修复（历史 stageB ±4.8 不可复用，已记录） | ERA §2.3；EPV §4/§8 |
| ADR-0006 评分接缝 + `study.py` manifest owner；wave1–4 run 归档 tarball | ADR-0006；DES §8；W5 头部 |
| 测试面：座位配对/JSONL/合并/联合 Hessian/激活兼容等回归；当前 `uv run pytest` 收集 859 项（2026-09-29） | EPV §4；`uv run --group train pytest -q` |

## 6. 已否决 / 不要再做（负结果清单）

| # | 条目 | 原因 | 依据 |
|---|---|---|---|
| N-1 | 继续扫 PPO 旋钮（激活 tanh/gelu/silu、vf_coef、clip、ent_coef、LR 量级/退火） | 全部扫过、无正贡献，tanh 显著更差（−48.8）；不要再用 500k 重复。**唯一例外：T14 条件项（C-2），仅在 A/B 证伪后触发** | ALS §5/§6.1；SD §9.1 |
| N-2 | hidden 256 / 1M 预算单列 | `h256_scratch` 从零最低（1367.7）；1M 不优于 680k | SD §6/§9.3 |
| N-3 | self-play 刷新间隔/采样、mix/均匀 pool 的再度组合 | wave1–5 已覆盖，全在噪声内；池成员全弱于学习者 | SD §9.4；W5 §4.2 |
| N-4 | 把 mix/pool 的逐决策重抽当 league（或让池全员弱于学习者） | 对手身份是每步隐变量、无胜率反馈、无难度梯度；不是 PFSP | SD §1.3/§4.3 |
| N-5 | 非 potential 的奖励加成（赢墩/炸弹/剩牌惩罚） | reward hacking 风险；目标错位未证伪，A1 的 telescoping 已覆盖无风险部分 | SD §9.5 |
| N-6 | 观测 S2 68 维 | 200k pilot 落后 full ~15 Elo（CI 排 0），丢 rank↔suit 配对 | OS §5.3/§7 |
| N-7 | 绝对 Elo 跨 fit/跨顺序比较宣布 <20 Elo 胜负 | 跨 fit sd≈31、相位位移 0–75；跨 fit 绝对值不可比 | ERA §3/§5；EPV §5/§8；SD §9.7 |
| N-8 | 单 seed 定论 | `pool4g` seed 0「显著 −19.6」被 seed 1/2（+8.7/−2.6）证伪 | H2H §3.2；EPV §7 |
| N-9 | 复用 `--cross>0` 旧 SE / stageB ±4.8 | 旧代码重复计数、低估最多 ~19%；历史 cross 输出不可回填 | ERA §2.3；SD §9.6；EPV §8 |
| N-10 | `--cross>0` 做筛选 | 绝对锚点+交叉对局分辨率远差于 h2h（±52 vs ±11–16） | SD §9.6 |
| N-11 | 用同牌参考面板给单人绝对定级降噪 | 实测残差 SD 反而更大（54.5 vs 42.4）；配对收益只存在于候选间共同相位 | HR §4.3 |
| N-12 | 轨迹 S3 策略一致度用于评分 | grouped m_eff 46.6 是风格重识别（LOLO 只剩 8.0），可被模仿；结果似然才是权威 | HR §4.2/§10；ADR-0006 |
| N-13 | 当前把轨迹 S2 价值 regret 进生产 | 离策略价值头 τ=201、1 步 TD 无增量；真正的 Q/rollout 需新模块且先对真人标定 | HR §4.2/§10；SD §10 Q3（更高成本） |
| N-14 | 未重测就 `--refit` 覆盖 manifest / 用单 seed 覆盖 | 单 seed lvl4 在 1388–1459 漂移；会把另一种抽样偏差冻进契约 | ERA §5.4；LRP §4.1 |
| N-15 | 为满足 spacing 调标签 | 标签是测量，不是凑间距；D1 只需单调 | LRP §4.2 step 5 |
| N-16 | 在修好评估设计前继续堆局数创造突破 | 局数只压牌抽样项，不消相位项；效应 <10 堆到 ±30 也没用 | ERA §6.6；SD §9.2 |
| N-17 | 再改一次评估协议（口径统一除外） | head-to-head + 换座配对已够用，边际收益低 | SD §9.9 |
| N-18 | 先做 >2 家 | 断了全部 2 家评估/迁移路径，且是另一个游戏 | SD §9.8；SD §5.4 |
| N-19 | NEXT_STEP 保留 + 事后剔除死样本 | 直接 SAME_STEP 已修复；剔除方案改动更大、不推荐 | PAA §7 P1 |
| N-20 | 架构忠实度（独立 actor/critic 双塔） | 主臂 vs `base680k` −9.13 [−20.57,+2.31]、vs `w5_ctrl` +5.21 [−5.93,+16.36] 不可分；交互臂 vs A1 −15.51 [−27.10,−3.92]、从零 vs `w5_scratch` −20.89 [−33.81,−7.97]；梯度 `cos(g_pol,g_val)=+0.049` 证伪「共享主干梯度干扰」，critic 照旧塌缩（V sd 0.113 vs A1 0.114）；参数 +69.6%。**关闭该线**：不做 E 组合、不加 Tanh 臂、不补 seed | [`experiments/twin-towers-500k.md`](./experiments/twin-towers-500k.md) §4–§6；SD §6A.5 |
| N-21 | 离线策略动作蒸馏（硬 CE / soft-Q / margin 过滤）把搜索强度写进权重 | 主端点 −52.8 [−66.0,−39.6]；同 View 两次独立搜索的模板翻转率 51.7%（搜索动作是采样隐藏世界的函数、obs v5 不可决定）；剂量-反应「越拟合越差、不拟合=raw」；搜索包着损坏 53 Elo 的学生仍 +40.4 → 强度在推理算子。**关闭 EI 策略线**；若仍要权重路径，先改学生可见信息（belief/obs，ADR 级） | [`experiments/joint-search-training-wave1.md`](./experiments/joint-search-training-wave1.md) §4；`runs/ei/report.md` |
| N-22 | 重训 value head 到搜索 E_w[Q] 以 bootstrap 截断搜索 | P1b（重训 critic vs 旧 critic，t=5 搜索）=+0.08 [−11.4,+11.6]；根分布 EV 0.844→0.829（P2b 未过）、叶分布 0.425→0.467（P2c 过）→ value 精度不是 t=5 截断的瓶颈。**关闭 bootstrap 循环**（预注册 §6 规则 2） | [`experiments/joint-search-training-wave1.md`](./experiments/joint-search-training-wave1.md) §5；`runs/ei2_value_t5/` |
| N-23 | 深度 2（对手 top-3 应手 min 节点）搜索 | vs depth-1 配对 −11.5 [−29.3,+5.3]（3 seed）、成本 ~3×；vs raw +26.4 但 t-CI 跨 0。**关闭树深方向** | [`experiments/joint-search-training-wave1.md`](./experiments/joint-search-training-wave1.md) §6；`runs/o4lite-search/{depth2_k8,scale}/` |
| N-24 | 同 recipe 重采叶子重训 critic（EI-3 round 1，t=5/t=10） | same-bank 配对 t5 **−4.44** [−15.37,+6.66]、t10 **−0.68** [−9.18,+7.99]（n=1200/臂）；leaf EV 差 +0.0002/+0.0026 且 final epoch 翻负（epoch 选择噪声）；479 真人决策 DQ/bias 无正信号；附 critic wd 单因子 **0.00** [−1.02,+1.02]（精确零效应，wd0 与 v2_t5 逐位一致）。**关闭“同 recipe 重采叶子”**；`t_leafq` 保持部署默认。**例外**：critic 容量（头部加宽）已由 cap0 clean 复核为负（见补记）；解冻 trunk 仍是未测的独立方向，未被本条目覆盖。**补记 2026-09-29**：采集器 K 世界同种子 bug 使该轮数据为单世界×8 复制（部署/h2h 不受影响，已修复+contract 30/30）。**clean re-run（t=5，seed 9400；55,792 决策/384,221 叶子，世界多样性 6.97/8，fallback 0）** 配对 **−1.37** [−11.99,+8.91]（n=1200、sign p 0.628）→ **结论复核为仍关闭**；同数据 cap0 clean 宽头 ΔEV 全 ≤ 0（w128 −0.0068、w256 −0.0011、w256x2 −0.0209，CI 均排除 0 但为负）→ critic 容量线亦负关闭。`t_leafq` 保持部署默认。 | [`experiments/ei3-value-loop-round1.md`](./experiments/ei3-value-loop-round1.md)；`runs/o4lite-search/ei3_value_loop/` |
| N-25 | 学习隐藏手牌后验替换均匀 determinization（② belief） | 4000 局 study / 195,066 决策训练逐牌后验（按 seed 切分）：held-out AUC 0.742 < count-only 0.748（K1 kill，公开历史相对“数牌”无增量）；人类 bank ROI（370 searched，K=32/t=5）主门 chosen-value Δ(post−uniform)=**−0.217** [−0.589,+0.160] FAIL、翻转率 22.2% < 均匀重采样噪声 23.5%，oracle 真手牌 +3.117 [+2.019,+4.271]。**停止：不集成 rolloutt:、不跑 h2h**；重访条件 = held-out top-k placement 优势显著变大（更强历史编码/更多数据）或改做局部/端局算牌 | [`experiments/belief-posterior-probe.md`](./experiments/belief-posterior-probe.md)；`runs/belief_posterior_probe/` |

## 7. 冲突裁定结果（2026-09-25 文档治理：只统一口径，不改实验数字）

> 原有 7 处「冲突与待统一」已按各来源报告的一手证据逐条裁定并落地；本表是唯一现行口径。
> 不属这 7 条的争议中，manifest refit、spacing 契约与牌堆 UX 已按 §4 裁决记录落地（2026-09-25），真人招募（D-3）等仍 pending。

| # | 议题 | 裁定（最终口径） | 依据（file:§+数字） | 落地（本次改动） |
|---|---|---|---|---|
| C-1 | 判定阈值 | 统一为 **README §3 / EPV §9** 口径：值得行动 = 合并 95% CI 完全排除 0 且点估计 ≥ +10；<10 一律不追。「+20 行动门槛」保留，但必须用**对 +20 的单侧等效/非劣检验**表达，禁止「点估计 ≥ +20」（EPV §6：Δ=20 时 power≈50%，0.474–0.480，且几乎不随 N/seed 改善）；≥20 的行动结论建议 5×400 或 3×800 复算（Δ=20、80% power ≈807 副，EPV §5） | EPV §6/§9；README §3；EPV §5 | SD §0/§2.4/§6A.4/§6A.5/§7.1.5 措辞已改；SD §7.2 标注 50% power；README §3 已一致；plans §1.2 去冲突注、§4 D-5 关闭 |
| C-2 | PPO 剩余对照 | 裁定为**条件项**：默认不排期（§6 N-1 维持）；仅当 A/B 线证伪（T1、T2/T3 的 3×400 h2h 均 CI 跨 0 或点估计 < +10）后触发，用于排除「优化受限」。范围 = ALS §6.2 `lr_1e4` 轻量复测、§6.4 `--num-minibatches 8` / Adam eps / `loss_vf1` h2h 复测 | ALS §6.2/§6.4；SD §9.1；EPV §6（优化受限可能性低） | plans T14 新增条件项（触发条件 + 存放位置）；§6 N-1 加例外注 |
| C-3 | 「S1」命名 | 统一别名：**轨迹 S1** = HEP/HR 逐墩轨迹特征（评分先验）；**观测 S1** = OS 103 维精简观测布局；同理 轨迹 S2/S3、观测 S2。**裸 S1 视为歧义，不得用于新文本** | HEP §2；HR §0/§5；OS §3.1/§3.2 | README §0.2 新增别名表；plans §1.3 改写术语注 + 全篇裸 S1 改前缀（§0.1、T3、D-4、D-7、§5）；轨迹语境的裸 S2/S3 同步加前缀（T13、§5、N-12/N-13、Q-10/Q-11） |
| C-4 | 平台顶部数值 | 建立**规范数字表**（数值+口径+出处+禁止用法）：平台顶唯一规范值 = `base680k` **1443.3 ± 14.7**（座位平衡口径）；lvl4 已于 2026-09-25 **refit 为 manifest 契约值 1429.7 ± 7.1**（LRP §4.4）；旧 **1489.4 ± 34.3** 仅历史（相位产物），禁止与平衡口径混比 | ERA §5.2；LRP §2.1；EPV §7；W5 §4.2；ALS §2.4（1443.7 复核） | README §0.1 新增规范数字表；ladder-report 顶部加勘误；HR §9.2 一处旧区间加注（不改实验数字） |
| C-5 | 协议冻结边界 | 裁定：**工具/统计管线冻结**（`ladder.py` 换座配对、`duel.py`/`head_to_head.py` twin + 牌聚簇 bootstrap、`elo.py` 联合 Hessian、`combine_duel_seeds` 公式、SAME_STEP）；本轮只统一阈值与表述。已确认 bug 的修复与 T10/T11 遗留缺陷（z/confidence 联动、window 近似）不算「再改协议」；N-17 维持。**注：ADR-0011（2026-09-25）已把评分器换成 OpenSkill，联合 Hessian 与 window 不再存在；本冻结边界的精神（评估管线不随实验随意改动）不变** | SD §9.9；EPV §9/§4/§8；plans §6 N-17、§3 T10/T11；ADR-0011 | README §3 新增冻结边界段；plans §1.2 同步。**2026-09-25 注记**：T1 工程伴随改动按本边界记录——owner 授权的纯速度评估并行化（`--workers`，默认 1 逐位不变、>1 与串行逐位一致）与已确认的 `duel.py` p 值 `1<<trials` 溢出修复（≤800 逐位不变，3 回归测试）；详见 [`experiments/reward-shaping-500k.md`](./experiments/reward-shaping-500k.md) §6 |
| C-6 | human-elo 标签基准 | 裁定：**默认基准 = T2/去收缩标定（HR §1/§5.2），不是 manifest T1**；T1 仅历史/契约引用（带相位偏移）。修订节替代范围扩大：§2–§4、§6–§8 的旧假设失效；§5 工程约束仍有效，但其中旧 20 局窗口示例以 HR §5.3 为准 | HEP 修订节；HR §1/§5.2/§5.3；LRP §2.1 | human-elo-plan 修订节补 §5/§8 说明 + §8 标题加注；plans T13 注明默认基准 |
| C-7 | LAD §4 旧破局路线 | 裁定：**LAD §4 仅作背景，不单独立项**；其绝对 Elo 与路线以新报告为准（对手池/容量/超参见 SD §9、W5 §4.2、ALS §5；margin 由 HR §5.3 定为轨迹先验在场时 σ×2） | LAD §4；SD §9.1/§9.3/§9.4；W5 §4.2；ALS §5；HR §5.3 | ladder-report 顶部加最小勘误行（指向 README §0.1/plans §7，不改原文）；plans §0.1 仍将 LAD 列为二手引用、§3 优先级不含 LAD §4 |

## 8. 推荐执行顺序（简图）

```text
训练结构线（2026-09-25 结果回填后）
  T1 A1/A2（已完成）：A1 未复现（4-seed 平均 +4.47/+6.77）、A2 null、γ1 不优 → A 线收束
    → T2 B0（已完成：null）→ T3 B1（已完成：未确认，主端点 CI 跨 0）
      → T5 C（已完成：null，2026-09-25；逐局/PFSP/强成员三效应 CI 全含 0）
        → T6 F（已完成：负 / 关闭，2026-09-25；双塔不可分、交互臂 −15.51、从零 −20.89）
        → **训练结构线收束**；仅剩 T8 E（条件未触发）与 T7 D（最后）

评估工程线（并行，不阻塞训练线）
  T9 cluster bootstrap → T10 z/confidence → T11 window 交叉项
  T12 arena 周期性定级 + PFSP 权重（T5 已用对局胜率版；arena Elo 接入未做）

工程支撑线（并行）
  v5 观测迁移 ✅ 已收口为唯一布局（2026-09-25；v1–v4/重映射/`tier` 变体已退役，ADR-0009）
  T14 PPO 防御性小项（不期待 Elo）

人类定级独立轨（受真人数据门控）
  T13 M1 fit_trace_prior（T2/去收缩标定）✅ 已完成 2026-09-25（experiments/trace-prior-m1.md）
        → M2 placement 包 ✅ + M3 CLI/play.py rung 标签 ✅（2026-09-25，experiments/placement-m2m3.md）
        ── 等 D-3 真人招募（D-1 refit 已执行）──→ M4 真人试点（RMSE/CI/档位命中）

远期（后置）
  T7 D >2 家（需先有跨 N 统一 obs 与 2 家迁移端点）
  T8 E 容量组合（仅 A/B 正信号后，~30 min/臂）
```

> T1 结果（2026-09-25，seed 复现后修正）：seed 1 四个 h2h 比较 CI 全排 0（+9.99…+12.75），但补训 seed 2/3/4 **未复现**（4-seed 平均 +4.47/+6.77，8 比较仅 seed 1 CI 排 0）→ 按 <10 不追口径 A 独立杠杆收束；A2/γ 已消融（A2 null、γ1 不优）；T2 B0 已跑完（null，见 [`experiments/observation-augmentation-b0.md`](./experiments/observation-augmentation-b0.md)）。T3 B1 也已定稿（未确认，见 [`experiments/observation-augmentation-b1.md`](./experiments/observation-augmentation-b1.md)）；T5 C 同日定稿（null，见 [`experiments/opponent-distribution-500k.md`](./experiments/opponent-distribution-500k.md)）；T6 F 同日完成并关闭（见 [`experiments/twin-towers-500k.md`](./experiments/twin-towers-500k.md)）→ **A/B/C/F 线均收束**。

门控规则一句话：**A/B/C/F 全部收束（A1 未复现、A2/γ 消融；B0 null、B1 未确认；C 三效应 null；F 双塔不可分且更差、已关闭）→ 训练结构线仅剩 T8 E（条件未触发）与 T7 D（最后）；评估工程已铺好、obs v5 默认布局已落地（T4 ✅ 2026-09-25，ADR-0008）；human-elo M1–M3 ✅ 离线链路就绪、M4 等 D-3（D-1 refit 已执行）。**

## 9. 开放问题（未成计划）

| # | 问题 | 现状/下一步观察 | 依据 |
|---|---|---|---|
| Q-1 | 信息受限 vs 优化受限 | A（改优化）与 B（改上界）就是判别实验；两者都 null 则需回答 Q-2 | SD §10 Q1 |
| Q-2 | 2 家 7鬼523 vs 固定脚本对手（现为 RandomBot）的真实上限是多少 | 可用 1-ply 搜索 bot/强参考测脚本对手可被利用空间；若上限 ~70% 则应换课程对手（并入 T5 强对手） | SD §10 Q2；ADR-0012 |
| Q-3 | 更细的逐墩 credit（自己赢墩 vs 对手赢墩） | A1 只是时间重分配；更细需要搜索/值分解，成本高，未排期 | SD §10 Q3 |
| Q-4 | PFSP 是否对平局/分差加权 | 可用分差作软标签；**旧 `FitConfig.margin` 已随 ADR-0011 删除**，如要做需在 OpenSkill 之上另设观测模型（当前不做） | SD §10 Q4；ADR-0011 |
| Q-5 | 观测记忆需求：unseen 不够时是否上显式剩余大牌估计/GRU | 属 B 下一档，破坏「纯 MLP + 全观测」契约，未排期 | SD §10 Q5 |
| Q-6 | 3 家内部相对强度小实验（先于 D 的工程投入） | 3 家 500k vs 2 家 500k（取 obs 前 191 维）内部比较 | SD §10 Q6 |
| Q-7 | 真人 OOD | LOLO 只是 bot 风格代理；真人 τ 未知，必须真人标定（D-3） | HR §9.2 |
| Q-8 | 真人非平稳 | 前 10 局边玩边学；可用最近 3 局先验 + 短结果窗口缓解 | HR §9.2 |
| Q-9 | 梯级上限 >1600 信息塌缩 | p(1−p) 塌到 19–6.5%；需更强 bot 或 handicap 匹配 | HR §6.1/§9.2 |
| Q-10 | 轨迹 S2 的「真正做掉」 | 逐决策 regret（Q 值/短 rollout）需新模块且先做 bot 间 calibration；当前不进生产 | HR §9.2；SD §10 Q3 |
| Q-11 | 轨迹 S3 作为风格信号（非强度） | 未来有真人数据后可评估用于选对手 | HR §9.2 |
| Q-12 | 随机策略的 policy seed 只按 `(牌, 座位槽)` 派生 | 同 spec 两侧逐位镜像（退化零）；引入随机/采样策略前先跑镜像校准，必要时改为按 entrant id 派生 | EPV §8 |
| Q-13 | 多重比较 | 95% 规则每对 ~1.7–1.9% 单侧假阳；筛 20 对期望 ~0.35 假阳 → 预注册/Bonferroni 或对「获胜对」加 seed 复算 | EPV §8 |
| Q-14 | deal seed 32-bit 碰撞 | 400 副时 ~2e-5，会 fail-loud；超长 schedule 可加去重 | EPV §8 |
| Q-15 | `--cross>0` 历史数据 | 仅 stageB ±4.8 被点名不可复用；其余历史 cross 结果按同规则处理 | ERA §2.3；EPV §8 |

---

> **维护约定**：条目状态变化时只改本文对应行 + 源报告；新增实验结论先回填源报告，再更新 §3–§6 与 `README §0`。
> 本文不持有任何一手数字；所有数字均可从 §0 的来源复数。

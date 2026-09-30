# test-time 对手意图学习：红队 findings 处置记录（repair 后）

> **状态：历史口径（repair 记录，对应 revision-2 规划快照）。阶段 A 后于 2026-09-27 执行并 G-I FAIL，见 [`experiments/opponent-intent-probe.md`](./experiments/opponent-intent-probe.md)。**
> 对应文档：[`opponent-intent-plan.md`](./opponent-intent-plan.md)（revision-2，repair 版本）。
> 本记录逐条处置 repair 输入的全部红队 findings。**全部处置只改
> `docs/opponent-intent-plan.md` 与本文件**；未改 `src/tests/tools/ADR/plans.md/README/
> CONTEXT/DESIGN`，未 commit/checkout/stash，未跑训练/h2h。
>
> **处置图例**
> - **ACCEPT**：finding 成立，规划已按其 fix 修正。
> - **ACCEPT-WITH-CORRECTION**：结论成立、但证据或 severity 表述需订正；规划按订正后的 fix 修改。
> - **REJECT**：finding 不成立或超出本工作流范围（逐条给反证）。
>
> **统计**：48 findings = 10 blocker + 19 major + 19 minor。
> **处置**：ACCEPT 46、ACCEPT-WITH-CORRECTION 2（stats-ml-05、engineering-contracts-length-confound 的证据订正）、
> REJECT 0。**没有任何 blocker 被驳回**；其中「阶段 B 实现超出授权」（engineering-contracts-stageb-out-of-contract）
> 被接受并升级为**需要用户决策的显式阻塞**（§4 末尾）。

---

## 0. 汇总表

| id | sev | 处置 | 规划改动位置（revision-2） |
|---|---|---|---|
| leakage-method-01 | blocker | ACCEPT | §0.3、§3.3 P0、§3.6 G-S/G-I |
| leakage-method-02 | blocker | ACCEPT | §4.1、§5.2、§6 step 9 |
| leakage-method-03 | blocker | ACCEPT | §3.5 |
| leakage-method-04 | major | ACCEPT | §2.3、§3.1 D1、§3.2、§3.4 |
| leakage-method-05 | major | ACCEPT | §4.1、§4.2 |
| leakage-method-06 | major | ACCEPT | §4.4 |
| leakage-method-07 | minor | ACCEPT | §3.8.4 |
| leakage-method-08 | minor | ACCEPT | §3.6 |
| leakage-method-09 | minor | ACCEPT | §3.3 P2、§3.4 |
| leakage-method-10 | minor | ACCEPT | §3.1 D2 |
| leakage-method-11 | minor | ACCEPT | 文首口径、§2.1、附录 A.5 |
| leakage-method-12 | minor | ACCEPT | §4.3 |
| stats-ml-01 | blocker | ACCEPT | §0.3、§3.6 G-S |
| stats-ml-02 | blocker | ACCEPT | §3.5 |
| stats-ml-03 | blocker | ACCEPT | §3.5、§3.6 G-D |
| stats-ml-04 | blocker | ACCEPT | §4.1、§4.5 |
| stats-ml-05 | major | ACCEPT-WITH-CORRECTION | §1.4、§4.6 |
| stats-ml-06 | major | ACCEPT | §3.6 |
| stats-ml-07 | major | ACCEPT | §3.5、§3.6 |
| stats-ml-08 | major | ACCEPT | §2.3、§3.1 D1、§3.4 |
| stats-ml-09 | major | ACCEPT | §4.3 |
| stats-ml-10 | major | ACCEPT | §4.3 |
| stats-ml-11 | major | ACCEPT | §4.3、§4.6 |
| stats-ml-12 | major | ACCEPT | §3.3 P1、§3.6 G-I |
| stats-ml-13 | minor | ACCEPT | §3.4、§4.4 |
| stats-ml-14 | minor | ACCEPT | §3.5 |
| stats-ml-15 | minor | ACCEPT | §3.3 P2、§3.4 |
| stats-ml-16 | minor | ACCEPT | §3.5、附录 A.2 |
| stats-ml-17 | minor | ACCEPT | §4.2 |
| stats-ml-18 | minor | ACCEPT | §4.5 |
| stats-ml-19 | minor | ACCEPT | §3.1 D2、§3.4 |
| ec-length-confound | blocker | ACCEPT-WITH-CORRECTION | §0.3、§3.2、§3.4、§3.5、§7-1 |
| ec-gate-weak-orders-budget | blocker | ACCEPT | §3.6 中止规则 |
| ec-stageb-out-of-contract | blocker | ACCEPT（升级为用户决策） | §4 头部、§5.2、§5.5、§6 |
| ec-arm-matrix-aux-confound | major | ACCEPT | §4.3、§4.4 |
| ec-measured-bucket-stat-wrong | major | ACCEPT | §3.5、附录 A.2 |
| ec-samestep-aux-label | major | ACCEPT | §4.3 |
| ec-event-tensor-memory | major | ACCEPT | §5.1 |
| ec-gitignored-probe-scripts | major | ACCEPT | §5.1 |
| ec-stageb-rollback-uncommitted | major | ACCEPT | §5.5 |
| ec-adaptation-curve-metric-ambiguous | major | ACCEPT | §3.5 |
| ec-controls-not-zero-info | major | ACCEPT | §3.4、§4.4 |
| ec-agent-forward-missing | minor | ACCEPT | §4.3、§5.2 |
| ec-d2-classes-underspecified | minor | ACCEPT | §3.1 D2 |
| ec-env-line-anchor-wrong | minor | ACCEPT | §5.2、附录 A.5 |
| ec-stageb-device-unspecified | minor | ACCEPT | §4.6 |
| ec-league-assert-mischaracterized | minor | ACCEPT | §5.2 |
| ec-h2h-serial-workers-wording | minor | ACCEPT | §4.6、§5.4 |

（`ec-` = `engineering-contracts-`）

---

## 1. Blockers（10/10 处置）

### leakage-method-01 — 闸门未排除强度 confound
**处置：ACCEPT。**
**证据**：`traces/study/manifest.json` 的 lvl1–lvl4 为 82.75/106.90/147.71/185.34 Elo，
`NeuralPolicy.act` 默认 argmax（`networks.py:756,824-834`）；原规划 §3.6（line 286-303）
的 G-A1.2 只要求 4 类 acc≥0.35 或 3/4 AUC≥0.70，无去分数条件。
**改动**：§0.3 明确 G-S「身份分类」只作健全性检查、预期 pass、不触发训练；§3.3 P0 强制
同时报（a）含 score/length 的 full、（b）score-blind、（c）score/length-only baseline；
§3.6 新增 G-I/G-F：主端点改为意图代理 + count-matched 分层 + 强度残差化；只有 H2∧H3 支持结论。

### leakage-method-02 — 阶段 B confirmatory 端点在 arm-vs-arm h2h 下不可测
**处置：ACCEPT。**
**证据**：`tools/head_to_head.py:1-37` 是 candidate-vs-candidate（`--left/--right` 各一 Policy
spec），无 pool/mixture 对手、无 per-identity 记录；其输出每 deal 一个胜负，无法拆出
「hist 臂 vs 共同对手」再与 blind 配对。
**改动**：§4.1 主确认端点改为「每条臂分别对固定池成员各打 N 副、逐局记录 `(member, deal, e,
胜负)`」，复用 `ladder.play_games`/`plan_games` + `results_out` JSONL（`ladder.py:527-555` 已写
`opponent`/`subject_seat`/`kind`/`rules_id`）或新增等价评测路径；§5.2 删除「tools 无需改」，
列入改动清单；§6 step 9 同步。

### leakage-method-03 — 分桶与等级强相关、桶内类别漂移
**处置：ACCEPT。**
**证据**：每局对手事件数按等级 26.8/27.2/24.9/22.9（lvl1–lvl4，本轮 80 局/级实测），长局
偏向 lvl1/lvl2；原 §3.5 line 270-271 的桶界基于 per-game 总量而非 per-prefix。
**改动**：§3.5 桶界改由 per-prefix 分布定义、报 per-class×bucket 样本数、类内曲线、
配对 within-game DID；桶内类别构成漂移即报告为 confound。

### stats-ml-01 — G-A1 近乎必然通过，止损分支死掉
**处置：ACCEPT。**
**证据**：同上；四个 subject 是四个不同网络（manifest subjects + 本轮核实 ckpt 存在，附录 A.4），
跨等级 AUC 可达 0.99（finding 自带只读探针）。G-A1.2 阈值远低于平凡可分性。
**改动**：§3.6 把身份分类降为 G-S（非必要门），新增 G-I（意图代理 + score-blind + count-matched）
与 G-D（配对 DID），并显式中止规则；§0.3 声明「对 4 个不同强度网络，身份分类近乎平凡」。

### stats-ml-02 — B5 桶类偏、桶间非配对
**处置：ACCEPT。** 与 leakage-method-03 同因。
**证据**：本轮 200 局实测 per-prefix e 中位 13，B5 仅 534/5128≈10%；B1 覆盖所有局。
**改动**：§3.5 配对 DID、类内曲线、per-class×bucket 计数、等频桶/实测桶界。

### stats-ml-03 — G-A2 斜率被信息单调性机械保证
**处置：ACCEPT。**
**改动**：§3.5 定义 null 模型 = count-only baseline 的同曲线，主适应度量 =
`Δ = [acc_hist(late)−acc_hist(early)] − [acc_count(late)−acc_count(early)]`，要求 CI 下界>0；
§3.6 G-D 明确这是唯一允许触发阶段 B 的机制门。

### stats-ml-04 — 阶段 B 主端点无身份变化
**处置：ACCEPT。** 与 leakage-method-02 同因、互为加强。
**改动**：§4.1 身份变化端点 + per-member 曲线；§4.5 适应签名 = member×e 交互，且对照
count-only 臂斜率增幅；加入 memory-only 臂（`count_only_noaux`）。

### engineering-contracts-length-confound — 长度 confound
**处置：ACCEPT-WITH-CORRECTION（结论成立，证据量级订正）。**
**证据**：本轮实测每局对手事件 lvl1 26.8 / lvl2 27.2 / lvl3 24.9 / lvl4 22.9（80 局/级），
总事件约翻倍即 finding 的 55.3/54.7/50.1/45.6；lvl1 与 lvl2 排序在抽样下可反转，
故措辞应为「随等级总体下降」而非严格单调。findings 的核心结论（仅 count 即可分类、
blind/noisy 保留 mask、适应曲线分桶条件于 post-treatment）全部成立。
**改动**：§3.2 增加 count-only/score-only baseline 与长度残差化；§3.4 重定义 blind/noisy 为
「保留计数的容量对照」并新增全 pad 真零信息对照；§3.5 count-matched 分层与配对 DID；
§7-1 记录。

### engineering-contracts-gate-weak-orders-budget — 弱门直接触发训练
**处置：ACCEPT。**
**证据**：原 §3.6 触发规则「G-A1 成立、G-A2 不成立 → 启动降级阶段 B（6 run）」；G-A1.2 的
OR 与 G-A1.3（历史依赖，独立于身份）使 pass 近乎必然。
**改动**：§3.6 中止规则：G-I 不成立即停止；G-D 不成立也**不**启动阶段 B 训练；
只有 G-I∧G-D∧G-F 才进入阶段 B 设计评审（且仍需新授权）。

### engineering-contracts-stageb-out-of-contract — 步骤 6–10 超契约
**处置：ACCEPT（升级为需用户决策的显式阻塞）。**
**证据**：原 §5.2 要改 `networks.py/env.py/ppo.py/train.py` 并加测试；§6 step 10 要「同步
plans.md」；契约禁止本工作流改 `src/tests/ADR/plans.md`。
**改动**：§4 头部、§5.2、§5.5、§6 明确步骤 6–10 标为「需新授权/新 workflow，本文件只设计」；
删除「全套测试全绿」「同步 plans.md」作为本文验收；§5.5 列出前置（干净基线 + ADR-A + 用户授权）。

---

## 2. Majors（19/19 处置）

### leakage-method-04 / stats-ml-08 — observer-only 不保证落在 chance
**处置：ACCEPT。**
**证据**：`RandomBot.act = self.rng.choice(legal_ids(view.mask))`（`policies.py:62-63`），
`policy_seed` 按 `(seed,seat)`（`record.py:47`）固定，但 `legal_ids` 依赖当前 state，
而 state 轨迹依赖对手行为 ⇒ 观测者动作序列是 class 的函数。findings 的只读探针给
observer-only 4 类 acc 0.395 vs chance 0.25。
**改动**：§2.3 新增「观测者自反」行为信息性 nuisance 行；§3.1 D1 明确只解耦发牌/手牌、
未解耦观测者轨迹；§3.2 `opponent_only` 为主视图；§3.4 报 observer-only 实测值、不当作泄漏。

### leakage-method-05 — 无 held-out 身份
**处置：ACCEPT。**
**证据**：原 §4.2 池成员 = random+lvl1..lvl4（= 阶段 A subject），训练与评测重合；aux 身份头
预测这 5 个固定 id。
**改动**：§4.2 训练池/留出身份分离（建议 train=random+lvl1+lvl2，held-out=lvl3+lvl4+一快照）；
§4.1 所有端点在留出身份上重复；只在已见身份上升不算 in-context。

### leakage-method-06 — poolblind 误称 memoryless，主端点混变量
**处置：ACCEPT。**
**证据**：`history.py:161,172` blind 只零化事件向量与 seat；v5 obs 仍含 `unseen`
（`env.py:106-120`）、`scores`、`opp_counts`、`last_player`。原 §1.3/§4.4 还把 aux 与 event
通道混在 `poolhist` vs `poolblind`。
**改动**：§4.4 重命名与语义（「event-blind，非 memoryless/完全无历史」），机制归因拆到
`hist_aux` vs `hist_noaux` 与 `hist_noaux` vs `eventblind_noaux`。

### stats-ml-05 — +10 门槛功效不足
**处置：ACCEPT-WITH-CORRECTION。**
**证据**：README §2 引 `evaluation-protocol-validation` 的 80% power 表：20 Elo→**807 副**、
10 Elo→**3226 副**。1200 副（3×400）**高于** 807，故 finding 的「1200 连 20 Elo 的 80% 功效
都不够」与所引表不符；正确表述是 **+10 Elo 远不足、20 Elo 约 80%**，且仓库另建议 ≥20 Elo
行动结论用 5×400 或 3×800。
**改动**：§1.4、§4.6 预注册 MDE/功效：主端点声称 +10 需 ≥3226 副或降级为 ±20 筛（807 副）。

### stats-ml-06 — 无多重比较控制
**处置：ACCEPT。**
**改动**：§3.6 单一主端点（P1 score-blind 优于 count-only）+ Holm 控制族；删除 OR 判据。

### stats-ml-07 — 阈值无功效依据、附录 A.1 非方差源、CI 忽略拟合方差
**处置：ACCEPT。**
**证据**：原附录 A.1 是 (dir,opponent) 计数命令，无方差；原 §3.5 的 CI 来自单切分 bootstrap。
**改动**：§3.5 用重复分组 CV + 嵌套簇 bootstrap；§3.6 阈值必须由步骤 1 pilot 方差给 MDE/功效。

### stats-ml-09 — aux 身份头 episode 恒定、塌缩风险、非最有用标签
**处置：ACCEPT。**
**证据**：`EpisodeMixturePolicy` 整局冻结（`policies.py:105,177`），`num_envs=8`/`num_steps=128`
下每 update 仅 ~8-16 个不同标签。
**改动**：§4.3 主头改为意图/威胁代理，身份降为低权副头；报 aux accuracy 与有效样本数。

### stats-ml-10 — aux 下一手 train/inference 条件不匹配
**处置：ACCEPT。**
**证据**：`env.py:321-336` learner 行动后 `match.advance()` 推对手；未说明读 pre/post-action hidden。
**改动**：§4.3 预注册 hidden 来源：默认预测**状态型意图代理**（不依赖 learner 刚行动），
或条件于 learner action 预测 `P(opp action | learner action)`。

### stats-ml-11 — aux λ 未冻结、sweep 未计预算
**处置：ACCEPT。**
**改动**：§4.3、§4.6 λ 由离线/单 seed pilot 定标并写入预注册，pilot 计入预算。

### stats-ml-12 — 探针测身份而非意图
**处置：ACCEPT。**
**改动**：§3.3 P1 意图代理升为**主必要性门**（D4：持分/持炸/能否压过 incumbent），
身份分类降为健全性；§3.6 G-I 据此定义。

### stats-ml-13 / ec-controls-not-zero-info — poolnoise 非零信息
**处置：ACCEPT。**
**证据**：`history.py:208-215`：blind 先 `mask[position]=True` 再 `continue`；noisy 写
`noise[position]`，**两者 mask 均为真实计数**。
**改动**：§3.4、§4.4 标为「保留计数的容量对照」，新增全 pad（mask 全 false）真零信息对照。

### stats-ml-14 — A2a 类别退化
**处置：ACCEPT。**
**证据**：本轮 subject 动作边际 single 0.585 / pass 0.182 / straight 0.105 / pair 0.100 /
small_bomb 0.028 / big_bomb 0.0005 / consecutive_pairs≈0。
**改动**：§3.5 明确 majority 先验 0.585、空类合并/删除、per-class + macro-NLL。

### stats-ml-15 / leakage-method-09 — A2b 隐藏合法集归一
**处置：ACCEPT（重复 findings，合并处置）。**
**改动**：§3.3 P2、§3.4 以 A2c（公开候选集）为主端点，A2b 仅报相对合法集先验 lift。

### stats-ml-16 / ec-measured-bucket-stat-wrong — 分桶「实测」错误
**处置：ACCEPT。**
**证据**：原 §3.5 写「16–36，中位 26，§A.2」，但 §A.2 无该统计；本轮实测 per-prefix e
0–35、中位 **13**、B5 ~10%；per-game 总量中位 26、范围 17–35（不同量）。
**改动**：§3.5 与附录 A.2 改正并给命令与输出；桶界由 per-prefix 分布重定。

### stats-ml-17 — 池只有 5 个同 lineage 成员
**处置：ACCEPT。**
**改动**：§4.2 要求报成员两两散度、限缩 generality、必要时加更多快照。

### stats-ml-18 — 同质池回退假设不成立
**处置：ACCEPT。**
**证据**：前序 D1-lite seq vs capacity-blind +8.41[−4.06,+20.89]、self-play v5 静态历史
+9.85[−2.97,+22.68]。
**改动**：§4.5 回退检查改为**斜率差**端点（有自己 CI），并写明噪声预期。

### stats-ml-19 — D2 无匹配 random 锚对照
**处置：ACCEPT。**
**改动**：§3.1 D2、§3.4：跨等级局用**同一 deal 双向 View**（`game.view(state,subject_seat)`
与 `game.view(state,candidate_seat)`）固定发牌，比较方向对比。

### ec-arm-matrix-aux-confound — 主端点混 aux 与历史
**处置：ACCEPT。**
**改动**：§4.3 要求在 aux 维度对齐；§4.4 主对照 `hist_aux` vs `hist_noaux`，另设
`count_only_noaux`；机制措辞分开 encoder 容量与 aux。

### ec-samestep-aux-label — aux 标签在 SAME_STEP 错位
**处置：ACCEPT。**
**证据**：`train.py:620-623` SAME_STEP；`train.py:829` 终局读 `policy.finished_id`；
`policies.py:176-190` `start_episode` 先 `finished_id=current_id` 再抽新成员。
**改动**：§4.3 aux 标签在 `env.step` 的 `info` 内捕获、终局/无对手动作 mask、`final_info` 区分。

### ec-event-tensor-memory — extract 全量内存爆
**处置：ACCEPT。**
**证据**：`(108,64)` float32 ≈ 27.6 KB/样本；4000 局 ≈ 20 万前缀 ≈ 5.5 GB（双观测者翻倍）。
**改动**：§5.1 只存紧凑摘要，事件张量按需重算/分批；步骤 2 加内存/产物预算验收。

### ec-gitignored-probe-scripts — 探针脚本未版本化
**处置：ACCEPT。**
**证据**：`.gitignore` 含 `runs/`；契约禁止改 `tools/` 与 commit。
**改动**：§5.1 探针脚本全文/关键片段内联进 `docs/experiments/opponent-intent-probe.md`，
附环境 hash，记录「runs/ 不版本化」限制。

### ec-stageb-rollback-uncommitted — 无干净回滚基线
**处置：ACCEPT（升级为需用户决策）。**
**证据**：`git status --short` 显示 `networks.py/env.py/ppo.py/train.py/league.py` 等均已
Modified（T15–T17，未提交）；契约禁止 commit/checkout/stash/worktree。
**改动**：§5.5 要求先授权把 T15–T17 落成一个提交，或新增文件 + 标记块 + 反向补丁；
否则不得批准阶段 B 实现。

### ec-adaptation-curve-metric-ambiguous — 适应曲线口径未定
**处置：ACCEPT。**
**改动**：§3.5 预注册 per-bucket 模型为主、fixed-full 截断为诊断，两口径都进报告模板。

---

## 3. Minors（19/19 处置）

### leakage-method-07 — oracle 哨兵用开局手牌会落 chance
**处置：ACCEPT。** 证据：本轮实测 200/200 seed 的 lvl1–lvl4 `initial.hands` 逐位相同，
故 D1 用开局手牌作 oracle 必然 ≈chance。**改动**：§3.8.4 改为必须用**决策点当时**
`state.hands[opp]`，并注明开局手牌跨类恒定。

### leakage-method-08 — G-A1 OR 无 FWER
**处置：ACCEPT。** **改动**：§3.6 单主端点 + Holm，删除 OR 与「≥3/4」选择式判据。

### leakage-method-09 — A2b 隐藏合法集（见 stats-ml-15）。

### leakage-method-10 — D2 构造未定义
**处置：ACCEPT。** **改动**：§3.1 D2 定义固定观测者的子任务，禁止混合观测者身份。

### leakage-method-11 — trace 不含 rules_id + 行号漂移
**处置：ACCEPT。** 证据：本轮读 `traces/study/lvl1/g0000__s3626764237__seat0__vsrandom.json`
的 `rules` 键无 `rules_id`；`play.py:226-244` 只校验 version/revision；`_publish` 实为
`env.py:387`（非 :186）。**改动**：文首口径、§2.1、§5.2、附录 A.5 校正。

### leakage-method-12 — SAME_STEP aux 标签错位
**处置：ACCEPT。** 同 ec-samestep-aux-label；**改动**：§4.3。

### stats-ml-13 — 见 majors（重复计入）。

### stats-ml-14/15/16/17/18/19 — 见 majors。

### ec-agent-forward-missing — 引用不存在的 Agent.forward
**处置：ACCEPT。** 证据：grep 显示 `def forward` 只在 `SequenceEncoder:120` 与
`EventSequenceEncoder:186`；`Agent` 暴露 `policy_logits:414`/`get_value:427`/
`get_action_and_value:439`，`NeuralPolicy.act` 调 `policy_logits`（`networks.py:768`）。
**改动**：§4.3、§5.2 落点改为 `_actor_hidden`/`_trunk_input` 的 hidden + 单独 `aux_logits(...)`，
签名不变、act 路径逐位不变。

### ec-d2-classes-underspecified — D2 4 类不可实现
**处置：ACCEPT。** 证据：跨等级只有 6 个 i<j 对（附录 A.1 已核实各 400），无观测者能同时看
4 个不同对手。**改动**：§3.1 D2 改为固定观测者的 2/3 分类子任务。

### ec-env-line-anchor-wrong — env.py 行号错误
**处置：ACCEPT。** **改动**：§5.2、附录 A.5 改为 `_SEGMENTS:186`/`encode_observation:221`/
`_publish:387`。

### ec-stageb-device-unspecified — 阶段 B 设备未声明
**处置：ACCEPT。** 证据：`train.py:505` `cuda` 可选；现有 run 默认 `cuda=True`。
**改动**：§4.6 显式声明阶段 B device 与并发，阶段 A 强制 CPU-only。

### ec-league-assert-mischaracterized — league 断言范围描述错误
**处置：ACCEPT。** 证据：`league.py:152-166` 在 self/mix/pool 分支内**无条件**断言
`history_layout(frozen)==history_layout(agent)`，`spec=="self"` 非条件。
**改动**：§5.2 更正。

### ec-h2h-serial-workers-wording — 串行/workers 表述冲突
**处置：ACCEPT。** 证据：`tools/head_to_head.py` 的 `--workers` 是单次 h2h 内并行 worker 数。
**改动**：§4.6、§5.4 改为「同一时刻只运行一个评测进程；进程内 `--device cpu --workers 4`」。

---

## 4. 阻塞项与需用户决策（升级）

以下事项**无法在本工作流内解决**，明确升级为需要用户决策的阻塞项：

1. **阶段 B 实现超出授权**（ec-stageb-out-of-contract）：改 `src/tests` 需要新的显式授权
   workflow + ADR-A；在此之前阶段 B 只是设计（§5.5）。
2. **缺少可回滚基线**（ec-stageb-rollback-uncommitted）：需用户授权把 T15–T17 落成一个提交，
   或接受「新增文件 + 标记块 + 反向补丁」方案；否则不应批准阶段 B。
3. **预注册数值冻结**（原 U1）：桶界、MDE、`L*`、残差化方式须在步骤 1 用 pilot 实测后由用户确认。
4. **依赖**（原 U2）：是否允许新增 `scikit-learn`/`scipy`。
5. **池的 train/held-out 划分**（原 U3）。
6. **aux 主头改为意图代理**（原 U4）。
7. **阶段 B 是否允许 CUDA 与并发上限**（原 U5）。

## 5. 未改变的关键结论

- 阶段 A 仍是 CPU-only、零/极低训练的离线探针；复用 `traces/study`（4000 局）。
- 观测 v5（161 维）、trace version=3、`rules.revision=3` 不变。
- 仓库统一行动门槛（CI 排 0 且 ≥+10）与 5 分线不变；阶段 B 的训练/评测口径沿用 EVH pilot。
- 「可识别 ≠ 可被 RL 利用」的区分贯穿始终。

# 事件历史规划的红队 findings 处置与修复记录

> **输入**：`docs/event-history-plan.md`（修复前 r1）+ 红队 findings JSON（41 条）。
> **输出**：修复后规划 [`event-history-plan.md`](./event-history-plan.md)（r2） + 本文。
> **角色**：修复与诊断 agent。本阶段**零训练、零 h2h**；只读代码 + 写这两个 Markdown；
> 未改 `src/`/`tests/`/`tools/`/ADR/`plans.md`/`README`/`CONTEXT`/`DESIGN`，未 `git commit/checkout/stash`。
> **处置结果**：41 条全部 **accepted**（0 rejected）；其中 blocker/major 全部在规划内解决，
> 唯 `engineering-adr0009-neutral-start`（中性起点消融）升级为**需要用户决策的阻塞项**（默认删除）。
> **后续状态（2026-09-26 实施、2026-09-27 判定）**：规划 r2 已按预注册落地并完成 pilot，主端点
> `event vs event_noise` 为 null（不 ship）；结论见
> [`experiments/event-history-pilot.md`](./experiments/event-history-pilot.md)。本文保留为红队处置的
> 历史记录，不再更新。

---

## 0. 核查方法

- 逐条把 finding 的 `file:line` 论断与当前工作区源码对照（`read`/`grep`）。
- 对三条需要实测的机制论断做**只读**复核（无训练、无 h2h）：
  - 全 pad 序列下 `SequenceEncoder` 输出与梯度（验证 `ml-stats-01`）；
  - `SequenceEncoder` 与规划 `EventSequenceEncoder` 的参数量（验证 `ml-stats-03`）；
  - 当前工作区 `View(...)` 构造点、`warm_start_from` 分支、`league.py` 冻结对手构造
    （验证 `contracts-view-field-default`、`contracts-warmstart-factual`、`ml-stats-02`）。
- 复现命令见 §3。

---

## 1. 总处置表（41/41）

正 = 左更强。`Plan §` 指修复后 [`event-history-plan.md`](./event-history-plan.md) 的落点。

| # | finding id | 严重度 | 处置 | 规划改动位置 |
|---|---|---|---|---|
| 1 | `contracts-view-field-default` | blocker | accepted | §3.2 `View.plays` 带默认 `()`；§3.5 test_env 构造点 |
| 2 | `contracts-warmstart-factual` | major | accepted | §4.5 更正为 strict `load_state_dict` 报错 |
| 3 | `contracts-obs-dim-identity` | major | accepted | §4.5 加 `history_layout` marker + 校验 |
| 4 | `contracts-design-adr-owner-doc` | major | accepted | §3.5 / §7.3 加 `DESIGN.md`/`CONTEXT.md` |
| 5 | `contracts-test-plan-gaps` | minor | accepted | §3.5 加 `test_ppo`/`test_train`/`test_env` |
| 6 | `contracts-event-tensor-magnitude` | minor | accepted | §7.1 具体 27.0 MiB / ~64× |
| 7 | `contracts-blind-caveat-dropped` | minor | accepted | §1.2 / §2.3 对照语义与先验 |
| 8 | `contracts-adr0014-wentout-tension` | minor | accepted | §3.4 显式对账 `went_out` vs ADR-0014 |
| 9 | `contracts-encode-events-needs-rules` | minor | accepted | §4.1 `event_length(rules)`；§4.4 签名 |
| 10 | `contracts-event-vs-seq-confound` | minor | accepted | §1.2 / §4.3 标为混淆端点 |
| 11 | `contracts-multiple-comparisons` | minor | accepted | §1.2 / §1.5 预注册端点族 + α |
| 12 | `contracts-first-wave-inconsistency` | minor | accepted | §1.4 精确臂表 + 两层次合并 |
| 13 | `contracts-B1-evidence-misuse` | minor | accepted | §0.3 / §2.3 relabel B1 为弱间接 |
| 14 | `ml-stats-01` | blocker | accepted | §1.2 新增 `event_noise` 主对照；§7.1 |
| 15 | `ml-stats-02` | blocker | accepted | §6 步骤 7 `league.py` 工作项 + smoke |
| 16 | `ml-stats-03` | major | accepted | §4.3 参数 1.95×；§1.2 归因限制 |
| 17 | `ml-stats-04` | major | accepted | §0.3 / §2.1 / §2.3 重述先验 |
| 18 | `ml-stats-05` | major | accepted | §1.5 / §5.3 t/z 双口径 |
| 19 | `ml-stats-06` | major | accepted | §1.5 分布联合判据 |
| 20 | `ml-stats-07` | major | accepted | §5.1 墩内 shuffle + `event_boundaryblind` |
| 21 | `ml-stats-08` | major | accepted | §1.3 self vs mix 语义 |
| 22 | `ml-stats-09` | major | accepted | §3.2 默认值；§3.5 构造点 |
| 23 | `ml-stats-10` | minor | accepted | §4.5 更正 + 中性起点默认删 |
| 24 | `ml-stats-11` | minor | accepted | §2.1 a2 不可归因；§5.1 boundaryblind |
| 25 | `ml-stats-12` | minor | accepted | §1.2 / §1.5 α 控制 |
| 26 | `ml-stats-13` | minor | accepted | §2.2 赢家例外；§3.4 补牌更正；附录 A |
| 27 | `ml-stats-14` | minor | accepted | §1.4 `event_blind` 降为可选 |
| 28 | `engineering-selfplay-frozen-opponent` | blocker | accepted | §6 步骤 7 + §1.3 |
| 29 | `engineering-event-blind-not-capacity-control` | major | accepted | §1.2 重定义；§5.1 |
| 30 | `engineering-warm-start-claim-wrong` | major | accepted | §4.5 |
| 31 | `engineering-selfplay-prior-misattribution` | major | accepted | §1.3 self-play 降级条件阶段 |
| 32 | `engineering-adr0009-neutral-start` | major | accepted | §4.5 / §7.3（**默认删；保留需用户决策 + ADR**） |
| 33 | `engineering-view-default-call-sites` | minor | accepted | §3.2 / §3.5 |
| 34 | `engineering-bootstrap-next-events` | minor | accepted | §6 步骤 4 |
| 35 | `engineering-throughput-framing` | minor | accepted | §7.1 吞吐优先 + 实测验收 |
| 36 | `engineering-eval-inference-cost` | minor | accepted | §4.4 / §5.2 |
| 37 | `engineering-shuffle-determinism` | minor | accepted | §5.1 确定置换 + 测试 |
| 38 | `engineering-seatblind-weak-contrast` | minor | accepted | §5.1 保守读法 |
| 39 | `engineering-doc-sync-and-exports` | minor | accepted | §3.5 `__init__`/`__all__` |
| 40 | `engineering-anchor-drift` | minor | accepted | 附录 A 全面复核 |
| 41 | `engineering-bound-test-policy` | minor | accepted | §6 步骤 2 随机策略驱动 |

---

## 2. 逐条处置与证据

### contracts 组

#### 1. `contracts-view-field-default`（blocker）— accepted
- **原论断**：规划 §3.2 给 `View` 新增无默认的 `plays`，会打断 4 处既有 `View(...)` 构造与
  step-1 验收。
- **证据**：`tests/test_env.py:67`（`_view` 工厂，被 `:253`/`:354`/`:370` 的 JSON fixture 用例
  使用）、`:196`、`:227`、`:317` 均为关键字构造且不含 `plays`。`View` 现为无默认字段的
  `dataclass`（`src/seven523/game.py:91-107`），`game.py:153` 是唯一生产者。
- **修复**：`View.plays: tuple[Play, ...] = ()`（置于最后，唯一默认字段）；`_view` 扩展为
  `spec.get("plays", ())`；§3.5 列入 `tests/test_env.py` 四个构造点 + fixture 逐位回归断言。

#### 2. `contracts-warmstart-factual`（major）— accepted
- **原论断**：规划把 warm start 描述为「跳过不匹配首层、得到随机首层」，实际同 layout 会走
  strict `load_state_dict` 直接报错。
- **证据**：`src/seven523/networks.py:446-449`——`loaded.nvec == agent.nvec and
  loaded.arch == agent.arch` 时调用 `agent.load_state_dict(loaded.state_dict())`（strict）。
  事件臂与 `t17pool` 正是此情形（nvec/arch 相同、obs_dim 都是 161）。「只拷等形 tensor」的
  `warm_start_into`（`networks.py:395-427`）在同 arch/nvec 时**不会**被调用。
  `tests/test_train.py:498` 断言逐位相等、`:525` 断言 `SystemExit`，对事件臂都不成立。
- **修复**：§4.5 更正失败模式；`train.py:412-423` 预检扩展为 layout 校验并转友好 `SystemExit`；
  加「`RuntimeError`→`SystemExit`」测试。

#### 3. `contracts-obs-dim-identity`（major）— accepted
- **原论断**：事件臂 `obs_dim` 仍 161、首层 193，`obs_dim` 不再能标识 layout。
- **证据**：`save_agent`/`load_agent` 只写/读 `obs_dim` 与 `obs_version`
  （`networks.py:292-347`）；`warm_start_from` 的唯一 layout guard 是 `obs_dim`
  (`networks.py:447`)；`tests/test_networks.py:75`/`:181` 用 `obs_dim` 定义身份。
- **修复**：ckpt payload 增 `history_layout` marker，`warm_start_from`/`load_agent` 与
  `obs_dim` 一并校验；§4.5 注明「第二输入加宽 trunk 后 obs_dim 不足以标识 layout」。

#### 4. `contracts-design-adr-owner-doc`（major）— accepted
- **原论断**：`View` 加字段但规划称「无需 ADR」，且改动表漏 `DESIGN.md`。
- **证据**：`DESIGN.md:154-158` 枚举 `View` 字段（`seat; hand; mask; ... done`）；
  `DESIGN.md:17`/ADR-0002 是单一投影接缝；ADR-0014 实现节写「`Game.view` 不新增字段」
  （该条针对 `empty_order`）。
- **修复**：§3.5/§7.3 把 `DESIGN.md` §2.4 + `CONTEXT.md` 列入改动表；援引 `played`/`last_player`
  先例说明「公开字段追加」无需单独 ADR，但 **owner doc 必须同步**。

#### 5. `contracts-test-plan-gaps`（minor）— accepted
- **原论断**：测试计划漏 `test_ppo.py`、`test_train.py` warm-start 回归、`test_env` fixture 路径。
- **证据**：`RolloutBatch` 增 3 字段（`ppo.py:52-93`）由 `tests/test_ppo.py:121`/`:137`
  pin；warm start 由 `tests/test_train.py:498`/`:525` pin；fixture 在 `tests/test_env.py:65-79`。
- **修复**：§3.5 加 `tests/test_ppo.py`（flatten/None）、`tests/test_train.py`（event 臂 smoke +
  warm-start 失败）、`tests/test_env.py`（fixture）三项。

#### 6. `contracts-event-tensor-magnitude`（minor）— accepted
- **原论断**：「大一个量级」错误。
- **证据**：`num_steps=128, num_envs=8`（`runs/t17seq__1__1790445087/args.json`）下
  `(128,8,108,64)` float32 = 28,311,552 B = 27.0 MiB；`(128,8,54)` int64 = 442,368 B ⇒ ≈64×。
- **修复**：§7.1 写具体 27.0 MiB / 0.84 MiB / 0.11 MiB，~64×；int64 位掩码降级为可选存储优化。

#### 7. `contracts-blind-caveat-dropped`（minor）— accepted
- **原论断**：规划称 `event_blind` 与 `seqblind` 完全对称，丢掉 pilot 的容量盲对照 caveat。
- **证据**：`docs/experiments/sequence-memory-pilot.md:346`：「容量盲对照只隔离了「加宽 trunk +
  编码器参数」，没隔离 GRU 架构本身的」。
- **修复**：§1.2/§2.3 把端点读法写成「信息 + 时序架构 vs 容量」。

#### 8. `contracts-adr0014-wentout-tension`（minor）— accepted
- **原论断**：一边引用 ADR-0014 排除 `empty_order`，一边加逐事件 `went_out`。
- **证据**：`docs/adr/0014-on-empty-digs.md`「考虑过的替代」拒绝「在 `View`/观测里暴露出空顺序」
  （理由：手牌数已公开）。
- **修复**：§3.4 显式对账——ADR-0014 拒绝的是「出空顺序作为专门字段/维度」；`Play.went_out` 是
  「逐事件、可逐步由 `View.counts` 导出的冗余便利标记」，且不投影 `empty_order`；提供
  `--event-wentout drop` 消融。

#### 9. `contracts-encode-events-needs-rules`（minor）— accepted
- **原论断**：`encode_events` 需要 `Rules`（`hand_size`/`num_players`），不能照抄
  `encode_history` 签名；`--event-len 108` 对 n=3 错误。
- **证据**：`history.py:42-43` 签名仅 `(view, length, order)`；`history.py:84`
  `HistorySequenceWrapper` 无 `rules`。
- **修复**：§4.1 `event_length(rules)=n*NUM_CARDS`；§4.4 签名
  `encode_events(view, rules, length=None, ...)`、`EventHistoryWrapper(env, rules, ...)`。

#### 10. `contracts-event-vs-seq-confound`（minor）— accepted
- **原论断**：`event vs seq` 不同容量，只有拼接宽一致。
- **证据**：实测 `SequenceEncoder` 5680 参数；规划 `EventSequenceEncoder` ≈11056；
  `networks.py:92-118` vs 规划 `Linear(64→32)→ReLU→Linear(32→32)` + `Embedding(n+1,16)` +
  `GRU(48→32)`。
- **修复**：§1.2/§4.3 标 `event vs seq` 为混淆端点（表示+容量+长度），不作机制归因。

#### 11. `contracts-multiple-comparisons`（minor）— accepted
- **原论断**：约 6 端点 × 2 分布，只预注册主端点，违反 `plans.md` Q-13。
- **证据**：`docs/plans.md:357` Q-13：95% 规则每对 ~1.7–1.9% 单侧假阳，建议预注册/Bonferroni。
- **修复**：§1.2 固定 confirmatory 族为 3 端点 + Bonferroni；§1.5 固定 α 与加 seed 规则。

#### 12. `contracts-first-wave-inconsistency`（minor）— accepted
- **原论断**：首轮 run 数与 3×400 口径不一致。
- **证据**：修复前 §1.4 说 9 run，§6 步骤 6 只训 6；`aggregate_seq.py` k=3 × `--seeds 0,1,2`
  × 400 ⇒ 3600 副/端点。
- **修复**：§1.4 写死精确臂表（random 9 run + 可选 `event_blind`；self-play 条件 6 run）与
  两层次合并（3 train seed × 3 deal seed × 400）。

#### 13. `contracts-B1-evidence-misuse`（minor）— accepted
- **原论断**：用静态 B1 self-play 优势支撑序列编码器。
- **证据**：`docs/experiments/sequence-memory-pilot.md:13`/`:179` 的 +9.85 是 obs B1 段（单 seed）；
  3-seed random 为 +2.89 [−9.52,+15.31]；`history-fusion-ablation.md:6-7` 说 B1 与序列编码正交。
- **修复**：§0.3/§2.3 把 B1 标为弱/间接，正先验改以 chrono vs sorted +10.46（顺序轴）领衔。

### ml-stats 组

#### 14. `ml-stats-01`（blocker）— accepted
- **原论断**：全零事件使编码器输出与样本无关，`event_blind` 退化为 MLP，不能作容量对照。
- **证据（红队实测 + 本 agent 复核）**：
  - `SequenceEncoder(16,32)(torch.zeros(4,54))` absmax = `0.0`，行间 `allclose` True；
  - `Agent(161,[134,4],hidden=128,seq_len=54,seq_blind=True)` 的
    `network.0.weight[:,161:]` grad absmax = `0.0`（obs 列 0.122）；
  - 机制：`networks.py:101` `padding_idx=0`（pad 行零 + 梯度屏蔽）+ GRU 零 bias
    （`networks.py:100-109`）。
- **修复**：§1.2 新增 `event_noise`（同编码器、事件向量=信息零但非退化）作为 Q-A 主对照；
  `event_blind` 降为 seed-matched 配对控制；删除「扣除编码器/容量」措辞；明确 `event_blind`
  连 seat 段一并置零。

#### 15. `ml-stats-02`（blocker）— accepted
- **原论断**：`league.py` 冻结对手未透传 seq/event 配置，self-play 臂跑不起来；改动表遗漏。
- **证据**：`src/seven523/league.py:153-167`——`frozen = NeuralPolicy(Agent(obs, nvec,
  hidden=agent.hidden, activation=..., arch=agent.arch), ...)`，随后 `frozen.agent.load_state_dict(
  agent.state_dict())`（strict）；未传 `seq_*`/`event_*`。`train.py:530-533` 每 refresh 原地替换。
- **复核**：对 `Agent(obs,nvec,hidden=128,seq_len=54)` 的 `state_dict` 载入
  `Agent(obs,nvec,hidden=128)` 抛 `RuntimeError`（unexpected `seq_encoder.*` / `network.0.weight`
  size mismatch）；事件臂同样。
- **修复**：§6 步骤 7 新增 `league.py` 工作项（`deepcopy(agent).eval()` 或全量透传 + 构造/refresh
  断言 + self-play+event smoke）；self-play 降为条件阶段（见 #31）。

#### 16. `ml-stats-03`（major）— accepted
- **原论断**：没有端点能干净回答「事件相对 D1-lite 的信息增量」；§4.3「容量对齐可直接比信息」为假。
- **证据（实测）**：`SequenceEncoder` = **5680** 参数；规划 `EventSequenceEncoder` =
  **11056**（MLP 3136 + seat 48 + GRU 7872），≈1.95×；序列长度 event ≤108 vs seq 54
  (`history.py:30`)。
- **修复**：§4.3 写死参数量与长度差异并标混淆；§1.2 归因只用 `event vs event_noise` /
  `event vs event_seatblind`。

#### 17. `ml-stats-04`（major）— accepted
- **原论断**：+10.46/+8.41 是顺序（a1）证据，而 a4（归属）无直接证据，先验张冠李戴。
- **证据**：修复前 §2.1/§2.3 自认 a1/a2 主要是编码差异；chrono/sorted 在 D1-lite/sorted 里根本
  不含逐座位归属；事件 54-multihot 与 v5 `unseen`（`env.py:106-122`）高度冗余。
- **修复**：§0.3/§2.1/§2.3 把 a4 先验重述为接近零/未知，事件臂定位探索性，报告逐轴对应先验来源。

#### 18. `ml-stats-05`（major）— accepted
- **原论断**：3-seed 合并用 z=1.96 对 2 自由度种子 sd 反保守约 2.2×。
- **证据**：`runs/t17seq_train/aggregate_seq.py:15` `Z=1.96`、`:49-50` 公式、`:16` k=3；
  t_{0.975,2}=4.303。以 chrono vs sorted sd=11.42 计：z half≈12.9、t half≈28.4。
- **修复**：§1.5/§5.3 改用 `q=t_{0.975,k-1}`，同时报 z 口径与敏感性；seed sd 主导端点标「不可判」。

#### 19. `ml-stats-06`（major）— accepted
- **原论断**：go/no-go 押在 random，但价值在 self-play 最高，且 3×400 功效仅 ~40%。
- **证据**：修复前 §1.4/§1.5/§6 把结论设在 random；§1.3 自认 a5/a6 只在 self-play 有料；
  §5.4 承认 ~40% 功效（README §0.2：10 Elo 需 1500–2000 副@50%）。
- **修复**：§1.5 加分布联合判据：random 门线绑定主端点，random null 不作「效应=0」结论；self-play
  为条件阶段；报告必须给功效；加 seed 规则预注册。

#### 20. `ml-stats-07`（major）— accepted
- **原论断**：全局 shuffle 会打乱 `opens_trick`/`went_out` 位置，破坏墩边界，机制端点不干净。
- **证据**：`opens_trick`/`went_out` 是逐事件标志；修复前 §7.2 自己把「墩内还是全局」留未决。
- **修复**：§5.1 把 `event_shuffle` 定义为**墩内确定置换**，保 `opens_trick` 位置与墩块；新增
  `event_boundaryblind`（置零 `opens_trick`）隔离边界侧。

#### 21. `ml-stats-08`（major）— accepted
- **原论断**：`--opponent self` 忽略 `--mix-random-prob`，实际是纯 self-play。
- **证据**：`src/seven523/league.py:168` `if config.opponent == "mix":`；`train.py:178-184`
  choices；`runs/t17self__1__1790439615/args.json` `opponent=self, mix_random_prob=0.5`。
- **修复**：§1.3 写死语义：纯 self-play 用 `--opponent self`（不写无意义 flag）；50/50 用
  `--opponent mix --mix-random-prob 0.5`；报告核对 `args.json`。

#### 22. `ml-stats-09`（major）— accepted
- 与 #1/#33 同源（`View.plays` 无默认打断 4 处构造）。见 §3.2。**修复**：默认 `()`。

#### 23. `ml-stats-10`（minor）— accepted
- **原论断**：warm_start 描述错误；从零训提高 seed 方差，中性起点降为可选不利小效应检出。
- **证据**：同 #2/#30。
- **修复**：§4.5 更正；中性起点**默认删除**（避免复活 ADR-0009 退役机制），若要保留需用户决策 + ADR。

#### 24. `ml-stats-11`（minor）— accepted
- **原论断**：a2 组合边界与墩边界不可测；seatblind 未在 self-play 复算。
- **证据**：修复前无隔离 a2 的端点；shuffle 同时破坏边界；self-play 只含 event/event_blind。
- **修复**：§2.1 明确声明 a2 不可单独归因；§5.1 加 `event_boundaryblind`。**用户已拍板**：
  self-play 阶段只训 `event`/`event_noise`（规划 §5.2/步骤 7），`event_seatblind` 只在 random
  第一波出动——本条原写的「self-play 含 seatblind」作废。

#### 25. `ml-stats-12`（minor）— accepted
- **原论断**：端点族内多重比较与序贯加 seed 无 alpha 控制。
- **证据**：`docs/plans.md:357` Q-13。
- **修复**：同 #11；§1.5 加 seed 规则写成固定触发（非 optional stopping），冻结分析口径。

#### 26. `ml-stats-13`（minor）— accepted
- **原论断**：(a) 补牌可由 `opens_trick` 推断——错误；(b) 墩赢家=最后非 pass，撬底角落例外；
  (c) 行号（`seq_hidden` 默认在 `networks.py:142` 非 `:150`）。
- **证据**：(a) `game.py:245-257` 只有不满手座位补牌、牌堆可能已空；每墩开始都 `opens_trick=True`。
  (b) `game.py:268-270`：补牌角落撬底者 = `empty_order[0]`（最早出空），非最后出牌者；
  ADR-0014 §决定.3。(c) `networks.py:142`（默认）/`:165`（赋值）。
- **修复**：§3.4 删除「补牌可推断」表述；§2.2 加赢家例外；§4.3/附录 A 更正行号。

#### 27. `ml-stats-14`（minor）— accepted
- **原论断**：6 个 `event_blind` run 与既有 `t17pool` 冗余（盲臂≈MLP），浪费并发预算。
- **证据**：由 #14/#29 得 `event_blind` 功能等价 MLP；seed-matched `runs/t17pool__*` 已存在
  （pilot §5.2）。
- **修复**：§1.4 把 `event_blind` 降为可选第 4 臂；主对照改 `event_noise`；每个 run 说明不可替代性。

### engineering 组

#### 28. `engineering-selfplay-frozen-opponent`（blocker）— accepted
- 同 #15，附带两点：镜像配置下 `--event-blind` 会**自动**传播到冻结对手（`NeuralPolicy.act` 读
  `agent.event_blind`），不需 D1-lite obs-B1 的 `encode_observation` monkeypatch；self-play
  对手须从**自己的** `View` 重建事件（相对自己）。**修复**：§6 步骤 7 四点工作项 + 测试。

#### 29. `engineering-event-blind-not-capacity-control`（major）— accepted
- 同 #14。补充证据：`runs/t17seqblind__1__1790445087/agent.pt` 用切片 MLP 替换编码器后 logits
  差 up to 3.34（额外列仅通过常量 bias 生效）；`seqblind vs t17pool = +2.84 [−20.00,+25.68]`；
  pilot §8.4 caveat。**修复**：§1.2 重定义 `event_noise`；报告禁止引「扣除容量」。

#### 30. `engineering-warm-start-claim-wrong`（major）— accepted
- 同 #2。补充：`docs/experiments/structural-directions.md` §0 item 5 已记录同机制
  （obs_dim 变宽 → `load_state_dict` → `RuntimeError`）。**修复**：§4.5 + 预检 `SystemExit`。

#### 31. `engineering-selfplay-prior-misattribution`（major）— accepted
- **原论断**：用于支撑 6 run 的 self-play 先验其实是不同的输入；D1-lite 从未做 self-play。
- **证据**：`runs/t17self__1__1790439615/args.json` 无 `seq_*` 键；pilot §8.2「self-play 只做了 P1 的
  B1 消融」；§4.1 的对比是 v5 obs B1 列。
- **修复**：§1.3 重写：无任何第二输入臂的 self-play 结果；唯一 B1 hint 单 seed；self-play 降为
  条件阶段（可选 1-seed 诊断后再定 3-seed 复算）。

#### 32. `engineering-adr0009-neutral-start`（major）— accepted（**升级为需用户决策**）
- **原论断**：零填充中性起点复活 ADR-0009 已退役的首层列映射，与「无需新 ADR」矛盾。
- **证据**：`docs/adr/0009`（`networks.py:364-375` 对 layout 不匹配 `WarmStartLayoutError`；
  `networks.py:399-427` 只拷等形 tensor）。ADR-0009 决定 #2「热启动只有一条路径」退役
  `_first_layer_remap`/`_REMAP_VERSION_PAIRS`。
- **修复**：§4.5/§7.3 **默认删除**中性起点消融；若要保留，需新立 ADR + `warm_start_event_columns`
  测试，且**需用户决策**。规划 §7.3 记录；附录 C 标注为唯一升级阻塞项。

#### 33. `engineering-view-default-call-sites`（minor）— accepted
- 同 #1。证据：`game.py:153` + `tests/test_env.py:67/:196/:227/:317`。**修复**：默认 `()` + 改动表。

#### 34. `engineering-bootstrap-next-events`（minor）— accepted
- **原论断**：PPO/train 工作清单漏 value-bootstrap 与加宽后的网络签名。
- **证据**：`train.py:658-671` 读 `next_seqs` 调 `agent.get_value(next_obs, next_seqs)`
  （blind 置零在 `:668-669`）；`ppo.py:155-164` 只切 `batch.seqs[mb_inds]`；
  `networks.py:207-218` `_trunk_input` 在无第二输入时抛错；调用点 `networks.py:515`、
  `ppo.py:160`、`train.py:574`/`:671`。
- **修复**：§6 步骤 4 加 `next_events*` bootstrap、minibatch 切片测试、`(T,E,L,64)→(B,L,64)`
  形状测试、`next_events == wrapper.last_events` 断言、调用点枚举。

#### 35. `engineering-throughput-framing`（minor）— accepted
- 同 #6，外加：真正成本是时序 GRU 吞吐，不是内存。**证据**：实测事件张量 27 MiB；每决策
  MLP 0.267 ms vs seq54 1.553 ms；`runs/t17seq__1__1790445087/metrics.csv` sps 612（6 并发）
  vs `runs/t17pool__1__1790439615/metrics.csv` sps 1057（3 并发）。**修复**：§7.1 风险 1 改为
  吞吐 + 实测验收；截断明示禁忌（ablation §3.3）。

#### 36. `engineering-eval-inference-cost`（minor）— accepted
- **原论断**：无状态重算同样拉高 h2h/eval 每决策 ~5–10×；串行预算忽略。
- **证据**：`networks.py:494-515` 每 call 重建序列；红队实测每决策约 3–4 ms（事件臂）；
  134k 决策/条 ⇒ 约 2–4 min/条。**修复**：§4.4/§5.2 写 eval wall-clock 与升级分支放大。

#### 37. `engineering-shuffle-determinism`（minor）— accepted
- **原论断**：`--event-order shuffled` 未定，随机置换破坏 train/eval 位一致与 h2h 可复现。
- **证据**：`history.py:57-71` 既有 `sorted` 是确定的；`duel.plan_duel_schedule`（`duel.py:33-46`）
  依赖确定性配对。**修复**：§5.1 写死无 RNG 确定置换 + payload 记录 + 确定性/位一致测试。

#### 38. `engineering-seatblind-weak-contrast`（minor）— accepted
- **原论断**：2 家下相对座位可由 (opens_trick, pass, 交替) 推到一个全局偏移，Q-B 是弱对照。
- **证据**：`game.py:217-224` 只跳过空手座位；`game.py:290-304` `_end_trick` 设
  `current=winner`；`opens_trick` 给每墩起点。**修复**：§5.1 写明保守读法：null 只读作「座位可由
  结构复原」，不读作「归属无用」。

#### 39. `engineering-doc-sync-and-exports`（minor）— accepted
- **原论断**：改动表漏公共 API/文档同步（DESIGN/CONTEXT、`__init__`、`history.__all__`）。
- **证据**：`game.py:91-107` 是文档化 agent-facing API；`src/seven523/__init__.py:46,82` 已
  re-export `View/GameState`；`history.py:22-27` 有 `__all__`；先例 `tests/test_policies.py:118`。
- **修复**：§3.5 加 `DESIGN.md`/`CONTEXT.md`/`__init__.py`/`__all__` 四项。

#### 40. `engineering-anchor-drift`（minor）— accepted
- **原论断**：若干行号错。
- **复核并更正**：`after = replace` 是 `game.py:201-209`（原写 196-205；按用户 verify 复核修正
  末行 212→209）；立即撬底是
  `game.py:214`（原写 208；单行 213-214→214）；补牌是 `game.py:245-257`（原写 248-268）；`StepResult` 构造是
  `game.py:305-313`（原写 294-305）；`action_mask` 是 `actions.py:235-251`、pass 在 `:262-263`
  （原写 232-254 / :262-263）；`seq_hidden` 默认 `networks.py:142`、赋值 `:165`（原写 :150）。
- **修复**：附录 A 全面重写；§3.1/§3.4 内联锚点同步。

#### 41. `engineering-bound-test-policy`（minor）— accepted
- **原论断**：用 `FirstLegalBot` 几乎不 pass，上界测试不 pin pass-heavy 区。
- **证据**：`tests/support.py:22-24` 返回 `legal_ids(view.mask)[0]`，`PASS_ID` 最高
  （`actions.py:94`，catalog 末位）。红队模拟：random 策略 n=2 最大 72、n=3 最大 93，bounds 安全。
- **修复**：§6 步骤 2 用随机策略（或混合）驱动上界/parity 测试，注释记录观测最大值。

---

## 3. 复现命令（只读，无训练/h2h）

```bash
# 1) View(...) 既有构造点（验证 contracts-view-field-default）
grep -n "View(" tests/test_env.py            # 67(_view)、196、227、317

# 2) warm_start_from 分支 + 身份 guard（验证 contracts-warmstart-factual / obs-dim-identity）
sed -n '395,461p' src/seven523/networks.py   # :446-449 strict load_state_dict；:447 obs_dim guard
sed -n '292,347p' src/seven523/networks.py   # save/load 只写 obs_dim/obs_version

# 3) league.py 冻结对手构造（验证 ml-stats-02）
sed -n '150,170p' src/seven523/league.py     # 未传 seq_*/event_*

# 4) 全 pad 编码器退化 + 参数量（验证 ml-stats-01 / ml-stats-03）
.venv/bin/python -c "
import torch, torch.nn as nn
from seven523.networks import SequenceEncoder, Agent
print('pad absmax', SequenceEncoder(16,32)(torch.zeros(4,54,dtype=torch.int64)).abs().max().item())
a = Agent(161,[134,4],hidden=128,seq_len=54,seq_blind=True)
a.policy_logits(torch.rand(8,161), torch.zeros(8,54,dtype=torch.int64)).sum().backward()
print('extra-col grad', a.network[0].weight.grad[:,161:].abs().max().item())
print('seq params', sum(p.numel() for p in SequenceEncoder(16,32).parameters()))
mlp = nn.Sequential(nn.Linear(64,32), nn.ReLU(), nn.Linear(32,32))
seat = nn.Embedding(3,16, padding_idx=2); gru = nn.GRU(48,32,batch_first=True)
print('event params', sum(p.numel() for p in mlp.parameters())+sum(p.numel() for p in seat.parameters())+sum(p.numel() for p in gru.parameters()))
"
# 输出：pad absmax 0.0 / extra-col grad 0.0 / seq params 5680 / event params 11056

# 5) 张量量级（验证 contracts-event-tensor-magnitude）
.venv/bin/python -c "
print('events MiB', 128*8*108*64*4/2**20, 'seats MiB', 128*8*108*8/2**20, 'mask MiB', 128*8*108/2**20)"
# 输出：events 27.0 / seats 0.84 / mask 0.105

# 6) self-play / merge 口径
cat runs/t17self__1__1790439615/args.json    # opponent=self, 无 seq_*
sed -n '10,52p' runs/t17seq_train/aggregate_seq.py   # Z=1.96, k=3
sed -n '33,46p' src/seven523/duel.py          # 换座 schedule
```

---

## 4. 遗留 / 用户裁决（2026-09-26 已拍板）

1. **`engineering-adr0009-neutral-start`**：用户裁决**从零训练**——删除「零填充中性起点」
   warm-start 消融，不恢复 ADR-0009 已退役的列重映射，不新立 ADR。阻塞项闭合。
2. **self-play 进入第一波**：用户裁决**必做**，顺序在 random 主端点（步骤 6）之后先修
   `league.py` 透传（blocker #15/#28）再训 `event`/`event_noise` × 3；真递归/更大规模迁移本次
   不做，留给后续（规划 §8.1）。self-play 先验空白（#31）仍在报告中如实标注。
3. **`event_blind` 不训**：用户裁决只保留开关代码与测试，不出 run；主对照是 `event_noise`
   （与 #27 的处置一致）。
4. **`event vs seq` 定位**：verify 发现其在 §1.2 端点族与「仅描述」间自相矛盾，已二选一钉死为
   **描述性端点**（移出 Bonferroni 族，k 由 3 降为 2）；规划 §1.2/§1.5/风险 9 同步。

除上述外，全部 blocker/major/minor 均在规划 r2 内闭环修复。

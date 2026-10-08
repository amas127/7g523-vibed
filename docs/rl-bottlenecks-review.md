# 7鬼523 RL 瓶颈盘点（2026-10-06）

> **性质**：索引式调查/综合，**不产生新数字、不改变任何既有结论**；所有论断转引源文件，
> 与源报告冲突时以源报告为准。若 owner 认为有价值，建议由 plans owner 决定是否并入
> [`plans.md`](plans.md) §3/§9 或 [`experiments/README.md`](experiments/README.md) §0。
>
> **口径**：revision-3（`rules_id=2e36dbea44893696`）、obs v5（2 家 161 维）、
> PPO + memoryless MLP。跨 fit 绝对分不可比（ADR-0013）；行动门槛 = 合并 95% CI 完全排除 0
> 且点估计 ≥ +10，< +5 止损（[`experiments/README.md`](experiments/README.md) §3）。
>
> **基线**：工作区 HEAD `0fd5c2e`（2026-09-29），2026-10-06 核对时工作区干净、`runs/` 无
> 09-29 之后的新产物；当前 manifest 数字为本盘点当日直接读取。

## 0. 结论摘要

一句话：**训练侧（改权重）已经真实到顶，推理期搜索是唯一越过行动门槛的强度杠杆；
当前的主要瓶颈不是某个超参，而是「信息上界未判定 + 价值/信用分配信号弱 + 评估分辨率不足」
三件事，其中第一件可以用两个廉价 run 回答。**

| # | 瓶颈 | 一手证据 | 性质 | 当前状态 |
|---|---|---|---|---|
| 1 | 训练侧策略上界（平台） | raw 顶簇 μ=182.9–187.5（同 fit，`traces/study/manifest.json`）；奖励/观测/对手/容量/预算/架构全部 null 或负 | 结构性，不是调参 | 已收束，无在跑项 |
| 2 | 信息受限 vs 优化受限未判定 | Stage-0 D-A/D-B/D-C/D-D 已设计（[`post-v5-structural-options.md`](post-v5-structural-options.md) §4），文档中无执行记录 | 根因未知 | **最便宜的未回答项** |
| 3 | 价值 / 信用分配 | 终局 only 奖励（`env.py:424-451`）；非零奖励步 4.17%、critic 早期 EV 0.065（[`structural-directions.md`](experiments/structural-directions.md) §1/§11） | 部分关闭 | value 循环 N-22/N-24 关闭；trunk 解冻等未测 |
| 4 | 隐状态 belief / determinization | 学习后验 AUC 0.742 < count-only 0.748；oracle 真手牌价值 +3.117（[`belief-posterior-probe.md`](experiments/belief-posterior-probe.md)） | 开放，空间大 | 当前实现关闭（N-25），重访条件已写 |
| 5 | 评估分辨率与方法学 | 10 Elo 需 1500–2000 副（50% power）；训练 seed sd 8–14；cap512 屏幕 +19.2 → 800k −11.49 | 进度瓶颈（非强度） | T9/T10 未做；预注册纪律需保持 |
| 6 | 搜索自身剩余瓶颈 | `search_leafq` μ=260.03±3.86，+72.6 μ vs raw 顶（同 fit，`manifest.json`；[`search-config-plan.md`](search-config-plan.md) §4.3） | 唯一强杠杆 | 候选提案/信念/对手课程未充分开发 |
| 7 | 人机/产品闭环 | 10 局定级 RMSE 53–72；真人 OOD 未标定 | 独立轨道，D-3 阻塞 | M1–M3 就绪，M4 等招募 |

---

## 1. 瓶颈一：训练侧策略上界——平台是结构性的

**现状**（当前 manifest，同一次 26,000 局 joint probit-MLE，禁止跨 fit 相加）：

| id | μ | σ | spec 类型 |
|---|---:|---:|---|
| `search_leafq` | **260.03** | 3.86 | 推理期搜索（t=5/K=32/C=6） |
| `w2m_ctl`（raw 顶） | 187.46 | 4.28 | ckpt |
| `w2m_plain` | 186.25 | 4.28 | ckpt |
| `w2m_low` | 186.00 | 4.27 | ckpt |
| `pself_s2` | 183.24 | 4.28 | ckpt |
| `ws_s2` | 182.91 | 3.98 | ckpt（warm-start 续训） |
| `lvl4` | 181.19 | 3.97 | ckpt（1M 快照） |

**为什么说是结构性的**：从零/500k–2M 的配方空间已经穷尽，全部没有 ≥ +10 的证据
（[`experiments/README.md`](experiments/README.md) §0、[`plans.md`](plans.md) §6）：

- 奖励整形 A1（4-seed 平均 +4.47）/ A2 / γ1；`terminal_win`、`saturate` 两波均未过门；
- 观测增广 B0 null、B1 未确认、EVH 事件表示 null、D1-lite 序列臂 null、GRU 顺序消融不可放大；
- 对手分布：fresh self-play（k=7 +2.87）、逐局冻结 + PFSP + 强成员（三效应 null）、
  固定外部/采样对照 null；
- 容量 hidden 256/512（512 在 800k 为 −11.49）、双塔（交互臂 −15.51）、
  batch 1024–4096、PPO 旋钮/激活/LR（tanh −48.8）、1M–2M 平台（187.6–191.6，历史 fit）。

**唯一的小正信号**是 warm-start 续训（`t23wswd`）：k=3 时 +11.48 疑似过门，补到 k=7 后
回落到 **+8.20**（z-CI [+5.35,+11.05]、t-CI [+4.65,+11.76]），vs `l1_2M` +7.38，vs 已知最强
`t18poolself_s2` **+3.91（t-CI 跨 0）** → **确认存在但低于 +10 行动门槛**，且 6 因素捆绑、
朴素续训对照未跑（[`warmstart-adamw-1m.md`](experiments/warmstart-adamw-1m.md) §0/§6.1）。
注意它在绝对 fit 里只从 `lvl4` 的 181.19 抬到 182.91——不是绝对表能分辨的量级。

**判断**：继续扫训练配方没有 ROI；训练侧只剩「归因 warm-start」这一个低成本收尾项，且预期 < +10。

## 2. 瓶颈二（根因）：信息受限 vs 优化受限——有设计，没执行

[`post-v5-structural-options.md`](post-v5-structural-options.md) §4 定义了四个廉价探针：

- **D-A Oracle-obs**：把隐藏信息（对手手牌/牌堆）拼进 obs 训练，若 oracle 也拿不到 ≥ +10
  → 信息不是瓶颈，O1 直接放弃；若大涨 → 信息路线有价值。
- **D-B 固定牌局拟合**：固定 500 副反复训，区分「优化/表达受限」与「泛化/探索受限」。
- **D-C Critic 天花板**：更大/独立 critic 测 EV 能否上升（当前 EV≈0.53–0.65）。
- **D-D 方差 screen**：`target_kl`、`lr 1e-4` 各 3 seed @200k，端点 = run sd。

**这四个探针在现有文档与 `runs/` 中没有执行记录**（截至 2026-09-29 提交）。它们是回答
[`plans.md`](plans.md) §9 Q-1/Q-2 的最短路径，成本 ≤ 数 run 级，且不破坏生产契约。
在它们出结果之前，O1（观测 v6）/O2（深度/归一化）/O3（真递归）/O5（换算法）都无法排序。

## 3. 瓶颈三：价值函数与信用分配

**结构事实**：默认奖励是终局 only（`src/seven523/env.py:424-451`，`reward_shaping="terminal"`），
非零奖励步仅 **4.17%**；`base680k` 价值网对真实终局回报的早期 EV 仅 **0.065**（后 1/5 才 0.853）
（[`structural-directions.md`](experiments/structural-directions.md) §1/§11）。`ppo.py:251`
的 explained variance 是训练中可直接观察的同一指标（历史读数 0.53–0.65）。

**它为什么仍是瓶颈**：推理期搜索的强度依赖 value 截断（t=5 读出）。`t_leafq` 用真实终局
margin 重训 critic 后 fresh-bank confirm **+18.24 Elo**，是 09-28/29 搜索发布的主要增益来源
（[`ei3-value-loop-round1.md`](experiments/ei3-value-loop-round1.md) 背景节、
[`search-config-plan.md`](search-config-plan.md) 文首更新）。也就是说，value 质量不是可有可无的
训练细节，而是当前最强算子的组件。

**已关闭**：N-22（bootstrap 到 `E_w[Q]`，+0.08）、N-24（同 recipe 重采叶子，clean 复核 −1.37）、
critic 头容量加宽（宽头 ΔEV 全 ≤ 0）。**未被上述关闭覆盖**：解冻 trunk 最后一层、胜负/风险
监督目标、不同的叶子采集 recipe（N-24 明确注明「解冻 trunk 仍是未测的独立方向」）。

## 4. 瓶颈四：隐状态 belief / determinization

**现状**：搜索对隐藏手牌做**均匀** determinization。离线探针显示：

- 当前 obs 下学到的逐牌后验 held-out AUC **0.742**，低于 count-only **0.748**（K1 kill）；
- 人类 bank 上 chosen-value Δ(post−uniform) = **−0.217**（CI 含 0，FAIL）；
- 但 oracle 换成真实手牌时价值 **+3.117** [+2.019,+4.271]，真实手牌能翻转 **40.3%** 的搜索
  决策（同 K 噪声下限 28.4%）——**信息上界空间很大，只是当前后验学不出来**。

证据：[`belief-posterior-probe.md`](experiments/belief-posterior-probe.md)；40.3% 见
[`save.md`](save.md)。**重访条件已写明**：held-out top-k placement 优势显著变大（更强历史编码/
更多数据），或改做局部/端局算牌。这是搜索侧最大的理论提升空间，也是 O1/信念路线的核心。

## 5. 瓶颈五：评估分辨率与方法学（拖慢一切迭代）

- 行动门槛 +10 / 止损 +5；400 副 h2h 的 CI 半宽 ≈ ±20 Elo；**10 Elo 需 1500–2000 副牌
  （50% power），80% power 再 ×2**（[`experiments/README.md`](experiments/README.md) §0.2/§3、
  [`evaluation-protocol-validation.md`](experiments/evaluation-protocol-validation.md) §5）。
- 训练 seed 方差 sd 8–14（F1b 8.43、A2 11.92、batch 14.4），**单 seed / 200k 屏幕不可信**：
  `cap512` 200k 屏幕 +19.2 [1.3,37.0] → 800k k=3 **−11.49**。
- **winner's curse**：warm-start 中段快照 +21.9 → holdout +6.81；+26.7 → +11.85；
  只有 final `agent.pt` 可报告（[`warmstart-adamw-1m.md`](experiments/warmstart-adamw-1m.md) §2.1）。
- 跨 fit 绝对 μ 不可比；绝对表顶部对比 CI ≈ ±10–15，**分不出 +7~+11**，小差距只能靠 h2h
  （[`warmstart-adamw-1m.md`](experiments/warmstart-adamw-1m.md) 附录 A）。
- 人机侧：「搜索 +110 Elo 为何未在 web 28 局体现」的正确读法是**无锚点不可识别**，
  不是零信息（[`human-transfer-gap-adversarial.md`](experiments/human-transfer-gap-adversarial.md)）。
- 工程缺口：T9（配对牌局 cluster bootstrap 固化进工具层）、T10（`combine_duel_seeds` 的
  z/confidence 联动）仍未做（[`plans.md`](plans.md) §3 T9/T10）。

**影响**：任何真实但 < +10 的效应（如 warm-start +7~+8）都要 4 个额外 1M run 才能严格确认；
这是「知道方向对不对」的主要成本瓶颈。

## 6. 瓶颈六：推理期搜索自身的剩余瓶颈

搜索是当前唯一过门槛的杠杆：`search_leafq` μ=**260.03 ± 3.86**（CI [251.22,269.89]，n=8000），
同 fit 比 raw 顶 +72.57 μ；h2h +135.66±3.80 只作非权威 sanity（
[`search-config-plan.md`](search-config-plan.md) §4.3）。搜索配置前沿：t=5 最优（t0/t1 为负）、
K16/K32 相对 K8 +25.0/+13.6，K32 单决策 80–110 ms（[`joint-search-training-wave1.md`](experiments/joint-search-training-wave1.md)）。

**已关闭**：depth-2（−11.5）、动作蒸馏（−52.8；obs v5 同 View 两次搜索动作翻转 51.7%，
强度在推理算子不在权重）、value bootstrap、同 recipe 叶子重采、当前 belief 后验。

**仍开放**：

1. **belief/determinization**（§4，理论空间最大，先做离线 ROI）；
2. **候选提案**：人类局面下 25.2% 的最优模板在 top-6 外，但放宽到全模板实测只 +3–5 Elo，
   预期有限（[`save.md`](save.md)）；
3. **critic trunk 解冻 / 新监督目标**（§3）；
4. **搜索作为训练课程对手**（现对手池全是更弱的 raw）：未测，中等成本/中等先验；
5. **先验外推**：μ≈260 超出 raw 先验语料范围，P6 用「按局排除」回避；扩语料重拟未做
   （[`search-config-plan.md`](search-config-plan.md) §5.4）；
6. 轻度非传递性已 flagged（leave-one-bank-out max|z|=3.59），「截断比全量强多少」本身未定论。

## 7. 瓶颈七：人机/产品闭环（独立轨道）

- 10 局定级做不到 ±50：RMSE 53–72、CI ±100–133；RMSE≤50 需 15–27 局
  （[`human-elo-10-games-research.md`](experiments/human-elo-10-games-research.md)）。
- M1–M3 离线链路已就绪；**M4 真人试点被 D-3 招募阻塞**；真人 OOD 未标定。
- 搜索 rung 已进定级池，Thompson/Fisher 可调度（C3，owner 已接受）；搜索对手的主观难度
  与可信度无数据（[`search-config-plan.md`](search-config-plan.md) §5.5）。

## 8. 补充：工程/基础设施观察

- 训练吞吐约 355 sps/run（并发下）；搜索 K32 80–110 ms/决策，400 deal 单进程约 1726 s。
- >2 家训练链未打通：2→3 家热启动被 `WarmStartLayoutError` 显式拒绝，评估链固定 2 家；
  属低优先级 T7（[`plans.md`](plans.md) §3 T7、§6 N-18）。
- T14 遗留小项：`batch_size % num_minibatches == 0` 断言未加（不期望 Elo 收益）。

## 9. 建议的行动排序（若目标是继续提升强度）

1. **先做 Stage-0 廉价诊断**（D-A oracle-obs 优先，其次 D-C critic 天花板）：回答
   「信息还是优化」，两个 run 级成本决定后面所有结构项是否值得开（§2）。
2. **搜索侧**：belief/determinization 离线 ROI 探针先行（重访条件已定义，§4）；
   同时可低成本试「搜索当作训练课程对手」。
3. **训练侧**：只保留 warm-start 朴素续训对照（归因收尾），预期 < +10，不当作大杠杆（§1）。
4. **评估侧**：固化 T9/T10，并在任何快照比较前预注册选择规则，避免 winner's curse（§5）。
5. **不要重开**：[`plans.md`](plans.md) §6 的 N-1…N-25（含 PPO 旋钮、容量宽度、
   事件/序列历史、self-play 变体、奖励跳变/饱和、双塔、加深搜索、动作蒸馏等）。

---

> **维护**：本文件只做索引，不持有数字所有权。若结论变化，请改源报告并同步
> [`experiments/README.md`](experiments/README.md) §0 与 [`plans.md`](plans.md)；本文件过期时直接删除。

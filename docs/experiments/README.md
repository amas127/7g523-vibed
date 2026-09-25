# 实验报告索引与当前结论（7鬼523）

> 本目录存每次实验的完整报告；本文件是索引与「现在到底知道什么」的单一入口。
> **全部计划/待办/建议的路线图见 [`../plans.md`](../plans.md)**（含优先级、依赖、验收与待决策清单）。
> 报告用中文，代码/数据论断给 `file:line`；原始产物在 `runs/`（gitignore）。
> 约定：**座位配对修复（2026-09-25）之前**的绝对 Elo 带相位偏移，不能与之后的值混用；
> 方案对比走 `tools/head_to_head.py` 多 seed（见下）。

> **规则口径警告（2026-09-25 牌型族比较）**：本目录**全部数字**（Elo、胜率、训练曲线、
> manifest 契约值、arena 排名、轨迹先验）都是在牌型族比较规则变更**之前**测得的旧 `tier`
> 口径结果；**禁止与新规则的结果混比**。重标定（重新生成 `traces/study`、重测 manifest、
> 重训 ladder、重标定先验）见 [`../plans.md`](../plans.md) T15 与
> [`../adr/0007-family-comparison.md`](../adr/0007-family-comparison.md)。

> **模型资产状态（2026-09-25）**：本目录报告引用的模型 checkpoint 已于 2026-09-25
> 随旧规则模型删除（`runs/**/*.pt` 全部删除，仅保留日志/脚本/结果 JSON），旧路径
> 不再可用；`tools/play_ladder.py` 现只保留 `random`/`greedy` 脚本档。各报告顶部
> 横幅已标注该状态；本目录数字为牌型族规则变更前口径，重标定与新模型路径登记见
> [`../plans.md`](../plans.md) T15。

## 0. 当前结论总表

1. **平台**：座位平衡口径下，策略上限约 **1440–1460 Elo**（`base680k 1443±15`）。
   150k–500k 预算里，PPO 旋钮、激活、损失配比、LR 扫描、self-play 刷新/采样、
   mix/pool 对手池、bigbatch、从零 500k——**没有任何方案显著超过 `base680k`/控制组**。
   对 GreedyBot 的胜率在 **62–65%** 饱和。
2. **评估口径（已修）**：
   - `ladder.py` 现在是**候选内换座配对**（每副牌双座位，`--games-per-anchor` 偶数），
     候选顺序不再影响结果；旧 1489 平台顶部实际约 1443。
   - `elo.py` 的自由候选间用**联合 Hessian** 算 SE（无 cross 路径不变）。
   - 跨 fit 绝对 Elo 仍不可比（`base680k` 跨 seed sd 高达 37）；方案对比用
     `head_to_head.py`：400 副牌 CI 半宽 ≈±20 Elo（比绝对口径窄 2.3–2.7×），
     **20 Elo ≈ 400–500 副牌，10 Elo ≈ 1500–2000**（均为 **50% power** 口径；80% power 需约 2× 牌数），且必须多 seed 复核。
3. **训练语义（已修）**：`train.py` 用 `AutoresetMode.SAME_STEP`（对齐参考栈）；旧 run
   含 ~3.9% 的 NEXT_STEP「死样本」，与 wave1–3 不可逐位比较。
4. **结构性诊断（`structural-directions.md`）**：奖励终局 only（非零奖励步 **4.17%**）；
   `base680k` 价值网对真实终局回报的早期 EV 仅 **0.065**（后 1/5 才 0.853）；观测缺
   已出牌历史（决策点平均 **24.7 张**牌不可见）；现有 mix/pool 是**逐决策均匀重抽**
   （非 league/PFSP，且成员全弱于 base680k）；>2 家训练/评估链断裂。优先级：
   **A 奖励重构 > B 观测增广 > C 对手分布 > D >2 家**。
5. **人类定级**：≤10 局的定量结论已出（`human-elo-10-games-research.md`）——**10 局做不到 ±50**；推荐方案（轨迹 S1 去收缩轨迹先验 + 胜负 + σ×2 分差，BT-MAP）10 局 RMSE≈**54**（grouped 乐观）/ **72**（LOLO 诚实），±1 档命中 83–93%、同档 51–65%；RMSE≤50 需 **15–27 局**，CI 收进 ±50 需 63–100 局。最强信号是轨迹 S1 逐局特征；轨迹 S3 一致度有风格泄漏（grouped m_eff 47 → LOLO 8）；建议 10 局只给临时档 + 诚实 CI，配 info/Thompson 自适应选档。
6. **结构性 A / T1（奖励重构，A 线收束）**：seed 1 的四个 head-to-head 比较
   （3×400 ×2 + 3×800 确认 ×2）95% CI 全排除 0、点估计 **+9.99…+12.75 Elo**，
   但补跑 seed 2/3/4（各 3×800 ×2）后**未复现**：4-seed 平均 **+4.47**（sd 5.6，
   vs `base680k`）/ **+6.77**（sd 4.4，vs `w5_ctrl`），8 个比较仅 seed 1 的两个 CI 排 0。
   按统一口径（<10 Elo 一律不追）**A1 作为独立杠杆不成立**；γ=1 消融不优、A2（纯胜负）不达标；
   critic 判别实验 D0/D2 显示 H1/H0 都不被干净支持。原始预注册记录保留为历史。详见
   [`reward-shaping-500k.md`](./reward-shaping-500k.md)。
7. **结构性 B / T2 B0（观测增广，null）**：`obs_version=2`（191→194，追加
   `trick_points`/`remaining_points`/`point_hold`）+ 版本化兼容层已实现（测试通过）；
   500k run `runs/t2_b0__1__1790330513` 跑满 499,712 步。h2h 3×400 vs `base680k`
   **+1.74 [−7.94,+11.42]**、vs `w5_ctrl` **+4.92 [−6.72,+16.56]** 均跨 0 →
   **B0 不单独采用**（<10 Elo 需 1500+ 副牌定性）；兼容层保留；B1 后续定稿（未确认，见下条）。详见
   [`observation-augmentation-b0.md`](./observation-augmentation-b0.md)。
8. **结构性 B / T3 B1（观测历史，未确认）**：`obs_version=3`（v2+55：`unseen` 54 维 +
   `last_player` 1 维，公式 `243+3n`、2 家 249）+ 引擎公开历史 `played` 已实现并测试通过
   （355 passed）。500k run `runs/t3_b1__1__1790331840` 跑满 499,712 步、无 NaN。h2h 3×400：
   主端点 vs `base680k` **+4.20 [−6.75,+15.15]**（CI 跨 0、点估计 <+10 → 失败）、
   vs `w5_ctrl` +13.04 [+1.93,+24.15]、相对 B0 **+0.73 [−18.98,+20.43]**（增量 ≈0）；
   单训练 seed、seed 间方差大 → **B1 未确认、不采用**，B 线收束（T5/C 随后也已完成、null，见第 9 条）。详见
   [`observation-augmentation-b1.md`](./observation-augmentation-b1.md)。
9. **结构性 C / T5（逐局对手 + PFSP + 强成员，null）**：`EpisodeMixturePolicy`（逐局冻结成员）
   + `pfsp_weights`（每 K=100 局 `w∝(1−wr)²+ε`、`Beta(prior=10)` 收缩、平局计半、与均匀
   0.5 混合）+ `--pool-episode/--pfsp*` 开关已实现（T5 新增 17 项测试，当时全套 372 passed；
   M1 落地后当前 378 passed）。三臂 500k（`C_u` 逐局均匀 / `C_p` +PFSP / `C_s` +强成员
   `base680k`；obs v1、seed 1、499,712 步、无 NaN）的 8 个 h2h（3×400 换座）CI 全含 0、
   点估计 |Δ|≤7：PFSP 效应 `C_p−C_u` **+1.88 [−17.93,+21.69]**、强成员效应 `C_s−C_u`
   **+2.90 [−17.32,+23.12]** → 逐局冻结/PFSP/强成员三效应均 **null**、不采用；Greedy 1500
   四臂 62.5–63.4% 在平台带内。详见 [`opponent-distribution-500k.md`](./opponent-distribution-500k.md)。
10. **T13-M1（轨迹 S1 先验已产出）**：`tools/fit_trace_prior.py` + `artifacts/human-elo/prior.json`
    （默认 T2 标签，sha256 `1895af98…`）；对照 `prior_manifest_labels.json`（sha256 `78e3535c…`）。
    数据 `traces/study` 1200 局 × 6 级（指纹 `48c32672…`）；T2 六值 random 1026.9 /
    greedy 1314.4 / lvl1 1128.6 / lvl2 1232.1 / lvl3 1356.1 / lvl4 1447.8（标签自身 SE≈17–18）。
    T2 默认 a=−890.92、b=1.7122、σ(5/10/20)=80.6/69.6/64.6、LOLO RMSE 133.9（m_eff 6.7）；
    manifest 对照 a=−879.40、b=1.7060、σ=70.1/57.8/51.4、RMSE 128.6（m_eff 7.3）。C-6：默认
    必须 T2；M2/M3 离线可做、M4 等 D-3。详见 [`trace-prior-m1.md`](./trace-prior-m1.md)。
11. **结构性 F / T6（双塔，负/关闭）**：`arch="towers"` 两塔 + 跨架构热启动 + `--arch` 已
    实现（+13 测试，全套 418 passed）；三臂 500k（seed 1、obs v1、499,712 步、无 NaN）与 6 组
    h2h 3×400 全部跑完。主臂 vs `base680k` **−9.13 [−20.57,+2.31]**、vs `w5_ctrl`
    **+5.21 [−5.93,+16.36]**（均不可分）；交互臂 vs A1 **−15.51 [−27.10,−3.92]**；从零臂 vs
    `w5_scratch` **−20.89 [−33.81,−7.97]**。机制：梯度 `cos(g_pol,g_val)=+0.049` →「共享主干
    梯度干扰」前提证伪；两塔确实分叉（0.178/0.094、彼此 0.201）；`trick_diff` 的 critic 照旧
    塌缩（V sd 0.113 vs A1 0.114、overall EV 0.014 vs 0.016）→ 按 SD §6A.5 **关闭「架构忠实
    度」线**（不做 E 组合、不加 Tanh 臂）。详见 [`twin-towers-500k.md`](./twin-towers-500k.md)。
12. **T13-M2/M3（定级会话 + CLI，离线链路就绪）**：`src/seven523/placement.py`（会话编排 +
    `main`，24 测试）、`7g523-elo` CLI、`play.py` 的 `opponent_identity` 与 rung 标签
    （`anchor:greedy@seatN` / `opponent:lvlN@seatN`，+3 测试）全部落地；**`elo.py` 零改动**
    （sha256 `80641f13…` 与改动前快照逐位相同）。冒烟 `--simulate greedy --games 2` →
    `定级：1287 ± 182（95% CI），最近档 greedy（1315），临时 provisional`，产物
    `traces/sessions/smoke_m23/`；全套 418 passed。M4 真人试点等 D-3。详见
    [`placement-m2m3.md`](./placement-m2m3.md)。

### 0.1 规范数字表（2026-09-25 建立，C-4；跨报告引用必须带口径）

| 数字 | 口径 | 出处 | 允许用法 | 禁止用法 |
|---|---|---|---|---|
| `base680k` = **1443.3 ± 14.7** | 座位平衡点估计（每副牌双座位换座）± bootstrap sd | `elo-reliability-audit.md` §5.2；两相位复核 1443.7（`activation-loss-sweep-report.md` §2.4） | **平台顶部/绝对刻度的唯一规范值**；冻结参照 | 与跨 fit/跨 seed 单 fit 值当同一口径比较；把 ±14.7 当 CI |
| `base680k` = 1443.3 ± 16.2 | 换座配对单 fit 联合 Hessian SE（单 seed） | `ladder-rerating-paired.md` §1.2；`evaluation-protocol-validation.md` §7 | 单 seed 绝对刻度 | 当多 seed 合并 CI 用（未含 seed 抽样项） |
| lvl4 = **1429.7 ± 7.1** | 5 seed × 400 局/锚点换座配对，combined fit + deal-cluster bootstrap | `ladder-rerating-paired.md` §2.1、§4.4 | **已 refit 为 manifest 契约值（2026-09-25）**；平台顶讨论 | 与旧相位 1489.4 混比；当作未合并的单 seed 值 |
| lvl4 = 1489.4 ± 34.3 | 旧 manifest 冻结值（100 局/锚点、未修座位相位；refit 前） | 旧 `traces/study/manifest.json`（git/备份历史，见 LRP §4.4）；`ladder-report.md` §2.4；审计 §5.4 | 历史对比（旧相位口径） | 当作平台顶部、当前契约或新实验的绝对 Elo；与平衡口径混比/混算 |
| 平台顶部区间 **1440–1460** | 平衡口径与后续复测的交集 | 本文件 §0；审计 §5.1；`elo-breakthrough-report.md` §0 | 叙述性引用 | 当作精确区间或 cross-fit CI |
| vs Greedy 胜率 **62–65%** | 固定 seat 0（1000–1500 局） | `structural-directions.md` §0；`elo-breakthrough-report.md` §4.5；`ladder-report.md` §2.1 | 饱和叙述 | 当作 Elo 或换座配对胜率 |

### 0.2 术语别名（C-3；引用必须带前缀）

| 写法 | 含义 | 来源 |
|---|---|---|
| **轨迹 S1** | 逐墩轨迹特征（评分先验） | `human-elo-plan.md` §2；HR §0/§5 |
| **轨迹 S2 / 轨迹 S3** | 决策 regret（价值）/ 策略一致度 | 同上；均不进生产（HR §10） |
| **观测 S1** | 191→103 维精简观测布局 | `observation-slimming.md` §3.1 |
| **观测 S2** | 68 维激进精简布局（已否） | `observation-slimming.md` §3.2 |
| 裸「S1」 | 歧义，禁止用于新文本 | `plans.md` §1.3；本节 |

## 1. 已完成报告

| 报告 | 主题 | 关键结论（一句话） |
|---|---|---|
| [`ladder-report.md`](./ladder-report.md) | 机器人阶梯 M2 + 平台首次观测 | 4 级 bot 1120→1489（旧相位口径，见该报告顶部勘误与 §0.1 规范数字表）；训练在 ~60k 步饱和；§4 破局路线仅作背景 |
| [`trace-signal-report.md`](./trace-signal-report.md) | D1 轨迹信号信息量 | S1/S2/S3 的 m_eff 实测，为真人快速度量级定级铺垫 |
| [`ppo-alignment-audit.md`](./ppo-alignment-audit.md) | PPO 移植对齐审计 | 32 项对齐、数学忠实；唯一机制偏差 NEXT_STEP 死样本已修；target_kl 从未启用但无影响 |
| [`elo-reliability-audit.md`](./elo-reliability-audit.md) | Elo 评估可靠性 | 跨 fit 40–75 Elo 漂移 = 座位相位 bug（换序位移 −7.5~+74.6）；已修换座配对 + 联合 Hessian SE |
| [`elo-breakthrough-report.md`](./elo-breakthrough-report.md) | wave1–3（17 个低预算实验） | 无突破；纠正 entropy 读法（花色头占 ~88%）；列未试轴 |
| [`activation-loss-sweep-report.md`](./activation-loss-sweep-report.md) | wave4（激活/损失/LR） | 12 run 全不赢；参考 ppo.py 的 tanh 在本游戏显著更差；LR 无正收益 |
| [`wave5-500k-report.md`](./wave5-500k-report.md) | 统一 500k 重评优势方案 | 6 臂全部不可分；父模型跨 seed sd=37、父子偏移翻符号 → 单 fit 比较不可信 |
| [`head-to-head-pilot.md`](./head-to-head-pilot.md) | 候选对候选换座工具 | CI 窄 2.3–2.7×、分辨率表；`pool4g` 单 seed「显著」被 seed 1/2 证伪 |
| [`structural-directions.md`](./structural-directions.md) | 结构性突破方向设计与量化 | 信用分配/观测缺口/联赛缺失的定量证据；优先级 A>B>C>**F（双塔）**>D 与验收协议 |
| [`reward-shaping-500k.md`](./reward-shaping-500k.md) | T1 结构性 A：奖励重构 500k（A1/A1γ1/A2 + seed 复现 + critic 判别） | seed 1 四个 h2h 比较 CI 全排 0（+9.99…+12.75），但补训 seed 2/3/4 **未复现**（4-seed 平均 +4.47/+6.77，8 比较仅 seed 1 CI 排 0）→ 按 <10 不追口径 **A1 独立杠杆不成立、A 线收束**；γ1 不优、A2 null；D0/D2 判别不干净支持 H1/H0；预注册机制检查未通过 |
| [`observation-augmentation-b0.md`](./observation-augmentation-b0.md) | T2 结构性 B0：观测增广 3 维（点分量）+ 兼容层 | `obs_version=2`（191→194）实现并测试通过；500k 训练 499,712 步；h2h 3×400 vs `base680k` +1.74 [−7.94,+11.42]、vs `w5_ctrl` +4.92 [−6.72,+16.56] 均跨 0 → 不单独采用；兼容层保留，B1 后续已完成（未确认，见下行） |
| [`observation-augmentation-b1.md`](./observation-augmentation-b1.md) | T3 结构性 B1：已出牌历史 + `unseen` + `last_player` | `obs_version=3`（v2+55、2 家 249）实现并测试通过（355 passed）；500k 499,712 步；h2h 3×400 vs `base680k` +4.20 [−6.75,+15.15]（主端点 CI 跨 0 → 失败）、vs `w5_ctrl` +13.04 [+1.93,+24.15]、相对 B0 +0.73 [−18.98,+20.43] → 未确认；B 线收束 |
| [`opponent-distribution-500k.md`](./opponent-distribution-500k.md) | T5 结构性 C：逐局对手 + PFSP + 强成员 | `EpisodeMixturePolicy`（逐局冻结）+ `pfsp_weights`（每 K=100 局 `w∝(1−wr)²+ε`、Beta(10) 收缩、平局计半、均匀 0.5 混合）+ `--pfsp*` 实现（+17 测试）；三臂 500k（`runs/t5_*__1__1790332696`，obs v1、seed 1、无 NaN）；8 个 h2h 3×400 CI 全含 0（PFSP 效应 +1.88、强成员 +2.90）→ 三效应 **null**；Greedy 1500 62.5–63.4% |
| [`twin-towers-500k.md`](./twin-towers-500k.md) | T6 结构性 F：独立 actor/critic 双塔 500k | `arch=towers` + 跨架构热启动实现并测试（+13；418 passed）；三臂 500k、6 组 h2h 3×400：主臂 vs `base680k` −9.13 [−20.57,+2.31]、vs `w5_ctrl` +5.21 [−5.93,+16.36] 不可分，交互臂 vs A1 −15.51 [−27.10,−3.92]、从零 vs `w5_scratch` −20.89 [−33.81,−7.97]；梯度 cos +0.049 证伪干扰前提、critic 仍塌缩 → 关闭「架构忠实度」线 |
| [`human-elo-10-games-research.md`](./human-elo-10-games-research.md) | 真人 ≤10 局快速定级研究 | 10 局达不到 ±50（RMSE 54–72、CI ±100–133）；轨迹 S1 去收缩轨迹先验最强，轨迹 S3 有风格泄漏；RMSE≤50 需 15–27 局 |
| [`trace-prior-m1.md`](./trace-prior-m1.md) | T13-M1：轨迹 S1 先验离线标定 | `tools/fit_trace_prior.py` + `prior.json`（默认 T2：a=−890.92、b=1.7122、σ(5/10/20)=80.6/69.6/64.6、LOLO RMSE 133.9、m_eff 6.7；sha256 `1895af98…`）与 `prior_manifest_labels.json` 对照（a=−879.40、b=1.7060、σ=70.1/57.8/51.4、RMSE 128.6；sha256 `78e3535c…`）；数据 1200 局 × 6 级（指纹 `48c32672…`）；6 项测试 |
| [`placement-m2m3.md`](./placement-m2m3.md) | T13-M2/M3：10 局定级会话 + CLI | `placement.py`（info/Thompson 选档、10 副不同牌、座位 5/5、两通道 BT-MAP `margin=(0.143,89)`、逐局 trace 与 `session.json`、`report.json`+`rungs.json`）+ `7g523-elo` 入口 + `play.py` rung 标签；`elo.py` 零改动（sha256 `80641f13…`）；冒烟 `定级：1287 ± 182，最近档 greedy（1315），provisional`；全套 418 passed；M4 等 D-3 |
| [`ladder-rerating-paired.md`](./ladder-rerating-paired.md) | 冻结阶梯 `lvl1–lvl4` 换座配对复测 | lvl4 旧值高估 60 Elo（1489→**1430±7**）、lvl1 旧值低估 26；排序 5/5 保持但间距变成 73/144/66（spacing 契约失效）；refit 需多 seed 合并 + owner 批准，旧 manifest 未动 |
| [`evaluation-protocol-validation.md`](./evaluation-protocol-validation.md) | 新评估口径独立验证 | 座位配对/聚簇 bootstrap/√N 均成立；判定规则需统一（+20 门槛使 Δ=20 功效仅 ~50%）；修复 `elo.py` 联合 Hessian 重复计数（SE 最多低估 19%）；80% power 需 807/3226 副（20/10 Elo） |
| [`observation-slimming.md`](./observation-slimming.md) | 观测精简审计与 200k pilot | 191→103（观测 S1）信息无损（`rank_counts`⊆`hand` 0/45752 不一致等）；h2h 与 full 不可分辨（+4.4[−18.3,+27]）；观测 S2 68 维落后 ~15 Elo（丢 rank↔suit 配对）；建议实施观测 S1、弃观测 S2 |
| [`tournament-arena.md`](./tournament-arena.md) | 并行 checkpoint 锦标赛竞技场 | 6 worker 22,620 局 37.8s、W=1 vs W>1 逐位一致；bf16 仅 +2% 已移除；单 seed 排名不可靠（两 seed ρ=0.54）；进度曲线：lvlbase ~184k 后平台、lvlsp 全程平，无新强度 |

## 2. PENDING（进行中的报告）

**当前无 PENDING**：所有子代理报告（含 B1、T5/C、T6/F、T13-M1–M3）均已定稿（B1 于
2026-09-25 定稿、判定未确认，见 [`observation-augmentation-b1.md`](./observation-augmentation-b1.md)；
T5/C 同日定稿为 null，见 [`opponent-distribution-500k.md`](./opponent-distribution-500k.md)；
T6/F 同日交付（负/关闭），见 [`twin-towers-500k.md`](./twin-towers-500k.md)；M1 同日交付、
M2/M3 同日离线就绪，见 [`trace-prior-m1.md`](./trace-prior-m1.md) /
[`placement-m2m3.md`](./placement-m2m3.md)）；
新任务见 [`../plans.md`](../plans.md) 的「待实施」与「待决策」。

## 3. 常用命令与口径

```bash
# 训练（SAME_STEP，500k 例；旧 base680k 等 ckpt 已删除）
# --load-checkpoint 可省略（冷启动）或指向自己训练的新 ckpt
uv run --group train 7g523-train --exp-name x --total-timesteps 500000 \
    --opponent greedy --load-checkpoint runs/<new-run>/agent.pt --cuda True

# 座位配对定级（偶数 --games-per-anchor；--games-out 落逐局 JSONL）
uv run --group train python tools/build_ladder.py \
    --candidate A=ckpt:runs/.../agent.pt --games-per-anchor 400 --seed 0 \
    --device cuda --no-traces --games-out runs/games.jsonl

# 候选对候选（多 seed、按牌聚簇配对 bootstrap）
uv run --group train python tools/head_to_head.py \
    --left A=ckpt:runs/<run-a>/agent.pt \
    --right B=ckpt:runs/<run-b>/agent.pt --seeds 0,1,2 --pairs 400 --device cuda

# 批量筛选
uv run --group train python tools/h2h_screen.py --help
```

判定规则（2026-09-25 经独立验证统一）：任何「方案 A 优于 B」在 **≥3 seed × 400 副牌（总 ≥1200）**
的换座 head-to-head 上，报 `mean ± 1.96·max(bootstrap SE, seed sd)/√k`；**CI 完全排除 0 且点估计 ≥+10
才谈「值得行动」**，≥20 Elo 的行动结论建议 5 seed × 400（80% power）或 3 seed × 800 复算；<10 Elo
一律不追（80% power 需 ≥3200 副）。可选分级：3×200 = 粗筛（只杀 <20 Elo）；5×400 = 确认；3×1500+ = 10 Elo 级。

**阈值口径备注（C-1，2026-09-25 统一）**：「点估计 ≥ +20」不是判定规则——`evaluation-protocol-validation.md` §6 证明它在 Δ=20 时 power≈50% 且几乎不随 N/seed 改善；声称越过 +20 行动门槛须对 +20 做单侧等效/非劣检验，并按 EPV §5 的功率表配置牌数。

**协议冻结边界（C-5，2026-09-25）**：评估工具与统计管线冻结（`ladder.py` 候选内换座配对、`duel.py`/`head_to_head.py` 同牌 twin + 牌聚簇 bootstrap、`elo.py` 联合 Hessian、`combine_duel_seeds` 合并公式、SAME_STEP）；本轮统一只动判定阈值与表述，不改工具链。已确认 bug 的修复（如联合 Hessian 重复计数）与 `evaluation-protocol-validation.md` §4/§8 的遗留缺陷（`z`/confidence 联动、`window` 非对称近似，见 `plans.md` T10/T11）不算「再改协议」；`structural-directions.md` §9.9 / `plans.md` §6 N-17 的「不要再改协议」正是指本边界之外的改动。
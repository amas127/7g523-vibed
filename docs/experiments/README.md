# 实验报告索引与当前结论（7鬼523）

> 本目录存每次实验的完整报告；本文件是索引与「现在到底知道什么」的单一入口。
> **全部计划/待办/建议的路线图见 [`../plans.md`](../plans.md)**（含优先级、依赖、验收与待决策清单）。
> 报告用中文，代码/数据论断给 `file:line`；原始产物在 `runs/`（gitignore）。
> 约定：**座位配对修复（2026-09-25）之前**的绝对 Elo 带相位偏移，不能与之后的值混用；
> 方案对比走 `tools/head_to_head.py` 多 seed（见下）。

> **规则口径警告（2026-09-27 更新）**：本目录**正文里的历史数字**（Elo、胜率、训练曲线、
> manifest 契约值、arena 排名、轨迹先验）均为旧口径（牌型族前 `tier` / revision-2）结果，
> 只作历史记录；**禁止与当前 revision-3（出空即撬底）结果混比**。当前有效口径为 T17
> （`rules_id=2e36dbea44893696`）+ 2026-09-27 w2m/T23 重排：
> `traces/study`（4000 局 trace）的 `levels` 由 homoscedastic probit-MLE 绝对表发布
> （`tools/refit_mle.py --manifest-out --refit`，ADR-0013），现为 **10 级池**
> （`lvl1`–`lvl4` = 84.68/113.60/134.40/187.72 加 `ws_s2`/`pself_s2`/三条 w2m 臂，
> rungs 冻结 `lvl1`/`lvl4`），见 `runs/w2m/calibration/report.md`；T17 先验与
> placement 常数（`c=0.465621`）见
> [`t17-recalibration.md`](./t17-recalibration.md)（w2m 重排同为 probit-MLE 刻度，
> 常数不重折）；新数字单独成表并标注「出空即撬底后」。
> revision-2 的 T15 study 与 pool10 manifest 已归档到
> `runs/archive/study-legacy-rules2-20260926/` 与
> `runs/archive/pool10-legacy-rules2-20260926/`（更早的
> `runs/archive/study-legacy-20260926/`、`pool10-legacy-20260926/` 保留作更旧历史）；
> 2026-09-25 架构清理后 `tier` 回放变体已退役
> （[ADR-0009](../adr/0009-single-observation-and-comparison.md)），旧 `tier` 口径
> checkpoint 已删除、现存 checkpoint 均为 v5。计划状态见 [`../plans.md`](../plans.md) T17。

> **模型资产状态（2026-09-27 更新）**：旧规则 checkpoint 已于 2026-09-25 删除
> （`runs/**/*.pt`），本目录报告引用的旧模型路径不可用、旧数字只作历史。T17（revision 3）
> 已重训并重测新池：`tools/play_ladder.py` 现登记 `random` gauge + `lvl1`–`lvl4` 梯级
> （`runs/t17*`，2026-09-27 w2m 重排后强度 **85/114/134/188**（homoscedastic probit-MLE
> 绝对表，ADR-0013）），并同时登记池内顶部平台簇 `ws_s2`/`pself_s2`/三条 w2m 臂
> （186/187/190/190/191）；见 `play_ladder.py list` 与 [`human-play.md`](../human-play.md) §3；
> T17 重标定报告与新旧口径边界见 [`t17-recalibration.md`](./t17-recalibration.md)；
> w2m/T23 重排（含 `ws_s2`/`pself_s2`/三条 w2m 臂入池）见
> `runs/w2m/calibration/report.md`；
> [`t15-recalibration.md`](./t15-recalibration.md) 保留为 revision-2 历史。

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
   **B0 不单独采用**（<10 Elo 需 1500+ 副牌定性）；兼容层当时保留、后随
   [ADR-0009](../adr/0009-single-observation-and-comparison.md) 退役；B1 后续定稿（未确认，见下条）。详见
   [`observation-augmentation-b0.md`](./observation-augmentation-b0.md)。
8. **结构性 B / T3 B1（观测历史，未确认）**：`obs_version=3`（v2+55：`unseen` 54 维 +
   `last_player` 1 维，公式 `243+3n`、2 家 249）+ 引擎公开历史 `played` 已实现并测试通过
   （当时 355 passed）。500k run `runs/t3_b1__1__1790331840` 跑满 499,712 步、无 NaN。h2h 3×400：
   主端点 vs `base680k` **+4.20 [−6.75,+15.15]**（CI 跨 0、点估计 <+10 → 失败）、
   vs `w5_ctrl` +13.04 [+1.93,+24.15]、相对 B0 **+0.73 [−18.98,+20.43]**（增量 ≈0）；
   单训练 seed、seed 间方差大 → **B1 未确认、不采用**，B 线收束（T5/C 随后也已完成、null，见第 9 条）。详见
   [`observation-augmentation-b1.md`](./observation-augmentation-b1.md)。
9. **结构性 C / T5（逐局对手 + PFSP + 强成员，null）**：`EpisodeMixturePolicy`（逐局冻结成员）
   + `pfsp_weights`（每 K=100 局 `w∝(1−wr)²+ε`、`Beta(prior=10)` 收缩、平局计半、与均匀
   0.5 混合）+ `--pool-episode/--pfsp*` 开关已实现（T5 新增 17 项测试，当时全套 372 passed；
   当前 `uv run pytest` 收集 859 项）。三臂 500k（`C_u` 逐局均匀 / `C_p` +PFSP / `C_s` +强成员
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
    实现（+13 测试，当时全套 418 passed）；三臂 500k（seed 1、obs v1、499,712 步、无 NaN）与 6 组
    h2h 3×400 全部跑完。主臂 vs `base680k` **−9.13 [−20.57,+2.31]**、vs `w5_ctrl`
    **+5.21 [−5.93,+16.36]**（均不可分）；交互臂 vs A1 **−15.51 [−27.10,−3.92]**；从零臂 vs
    `w5_scratch` **−20.89 [−33.81,−7.97]**。机制：梯度 `cos(g_pol,g_val)=+0.049` →「共享主干
    梯度干扰」前提证伪；两塔确实分叉（0.178/0.094、彼此 0.201）；`trick_diff` 的 critic 照旧
    塌缩（V sd 0.113 vs A1 0.114、overall EV 0.014 vs 0.016）→ 按 SD §6A.5 **关闭「架构忠实
    度」线**（不做 E 组合、不加 Tanh 臂）。详见 [`twin-towers-500k.md`](./twin-towers-500k.md)。
12. **T13-M2/M3（定级会话 + CLI，离线链路就绪）**：`src/seven523/placement/`（包：会话编排 +
    `main`）、`7g523-elo` CLI、`play.py` 的 `opponent_identity` 与 rung 标签
    （`anchor:greedy@seatN` / `opponent:lvlN@seatN`，+3 测试）全部落地；**`elo.py` 零改动**
    （sha256 `80641f13…` 与改动前快照逐位相同）。冒烟 `--simulate greedy --games 2` →
    `定级：1287 ± 182（95% CI），最近档 greedy（1315），临时 provisional`，产物
    `traces/sessions/smoke_m23/`；当时全套 418 passed。M4 真人试点等 D-3。详见
    [`placement-m2m3.md`](./placement-m2m3.md)。

13. **推理期搜索 + value 截断（2026-09-28，wave1）**：O4-lite 搜索（决定化 K 采样 + top-6 候选 +
    frozen-champion rollout）在 500k 训练侧全 null 的背景下成为唯一过门槛的强度杠杆；
    value 截断（t=5 步后用 critic 读出，`runs/o4lite-search/rollout_trunc.py`）把 K=8 全量的
    +42.6（bank 30-32）推到 **t=5×K16 +111.2**、**t=5×K32 +124.7**（20 seeds，vs raw）；
    配对 K16−K8 = +25.0 [18.4,31.4]、K32−K16 = +13.6 [7.2,20.2]；t 在 5–7 进平台（t0/1 崩溃）。
    权重侧联合方案均负：离线策略蒸馏 −52.8（同 View 标签翻转 51.7%）、value 重训 bootstrap
    P1b≈0（根分布 EV 未过门槛）、depth-2 min 节点 −11.5。Route-A 加速层逐位精确（189→100 ms/决策，
    叠截断 ~9×）；人机入口 `runs/o4lite-search/play_search.py --trunc 5 [--rollout-k 16|32]`。
    详情与原始产物见 [`joint-search-training-wave1.md`](./joint-search-training-wave1.md)；
    负结果 N-21/N-22/N-23 见 [`../plans.md`](../plans.md) §6。
14. **搜索 rung 进入发布表（2026-09-29，P4a `search_leafq`）**：单位修复 + `t_leafq` value 头后实测的
    t=5/K=32/C=6 配置（rollout 对手 `w2m_ctl`）与 raw 10 级进 **同一** 26,000 局（18,000 w2m
    calibration + 8,000 P4a 银行）joint probit-MLE（`--anchors random=0 --bootstrap 4000 --seed 0`，
    ADR-0013）：`search_leafq` μ=**260.03 ± 3.86**（95% CI [251.22,269.89]，n=8000）；raw 顶簇
    同 fit 位移 `w2m_ctl` **191.49 → 187.46**（其余 −0.08…−8.15）；`converged/margin_identified=true`、
    `separation=[]`；leave-one-bank-out max|z|=3.59（flagged `search_leafq-lvl1/lvl3`，轻度非传递性）。
    已由 `refit_mle --manifest-out --refit --keep-rungs --spec search_leafq=rolloutt:...` 发布为
    `traces/study`(=pool10) manifest 第 11 级（`rules_id` 与 `lvl1`/`lvl4` rung 契约不变；spec 由
    `--spec` 写入并在后续 refit 保留）。P6（2026-09-29）起搜索身份改钉在
    `subject.search_config`（manifest 单一来源；artifact 仅作 provenance），定级池就是 manifest
    `levels`（raw + search rung 同表，C3 调度），`rated`/`unrated` 路径删除；轨迹先验改为**按局
    排除**（只有真正打到该 rung 的那局行不进轨迹通道并记 `prior_off_reason`，raw 局照常用先验；
    先验重拟仍待扩展语料，follow-up 见 `../search-config-plan.md` §4.3/§5）。非权威 sanity：h2h
    +135.66±3.80 vs raw（不做跨 fit 加法）。详情 `../search-config-plan.md` §4.3、
    `artifacts/search-rung/search_rung.json`、`runs/o4lite-search/README.md`。
15. **EI-3 value 迭代 round 1（重采叶子 + 重训 critic，负/关闭）**：以 `t_leafq` 为搜索 critic、
    K=32、t=5/t=10 在全新 bank（seed 9200/9300，各 1800 deal）重采真实终局 margin 叶子并同 recipe
    重训 critic（policy/trunk 冻结）。同 bank 配对（seeds 30-32 × 400 副，n=1200）：`v2_t5 − base_t5` =
    **−4.44** [−15.37,+6.66]、`v2_t10 − base_t10` = **−0.68** [−9.18,+7.99]，均 negative；
    leaf EV 差 +0.0002/+0.0026 且 final epoch 翻负（epoch 选择噪声）；479 真人决策 DQ/bias 无正信号。
    附单因子对照：critic `AdamW(wd=0.01) − Adam(wd=0)` = **0.00** [−1.02,+1.02]（精确零效应）。
    → 关闭“同 recipe 重采叶子”（N-24）；`t_leafq` 保持部署默认。详见
    [`ei3-value-loop-round1.md`](./ei3-value-loop-round1.md)。
16. **② 隐藏手牌学习后验（2026-09-29，离线 ROI 负/停止）**：4000 局 study（195,066 决策，按 seed
    切分）训练逐牌后验，held-out AUC 0.742 **低于 count-only 0.748**（K1 kill）；人类 bank ROI
    （370 searched，K=32/t=5）主门 chosen-value Δ(post−uniform) = **−0.217** [−0.589,+0.160] FAIL，
    翻转率 22.2% 低于均匀重采样噪声 23.5%，oracle 真手牌仍 **+3.117** [+2.019,+4.271]。
    → 不集成、不 h2h；重访条件 = held-out top-k placement 优势显著变大。详见
    [`belief-posterior-probe.md`](./belief-posterior-probe.md)。

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
| [`warmstart-adamw-1m.md`](./warmstart-adamw-1m.md) | 后 v5 优化波：warm-start 续训（t23wswd，探索性、未预注册） | k=7 合并 vs 起点 `l1_1M` **+8.20**（z/t CI 均排除 0）但低于 +10 行动门槛；对最强参照 `t18poolself_s2` +3.91 未确认；6 因素捆绑、单因子对照未跑 → 已确认的小正增益、归因不清，不宣称过门槛 |
| [`human-transfer-gap.md`](./human-transfer-gap.md) | 真人转移缺口调查：搜索 +110 Elo 为何未在 web 28 局体现 | 搜索部署与实验配置逐 bit 可复现；「真人没打出相应水平」前提未被数据确立；+110 是 h2h 相对分、不能与 manifest probit μ 相加；评测/rollout 对手同源 frozen champion、真人 OOD |
| [`human-transfer-gap-adversarial.md`](./human-transfer-gap-adversarial.md) | 上述调查的对抗审查（C1–C12） | C1/C3/C4/C6 confirmed、C2/C5/C7/C8/C9/C11 收窄、C10 被推翻、C12 应删除；正确读法是「无锚点不可识别」，不是「绝对零信息」 |
| [`evaluation-resolution-design.md`](./evaluation-resolution-design.md) | 评估分辨率设计：定级表 + 带内 CRN round-robin 重排（设计提案 v2，未实现） | 论证单一 pinned 锚的绝对 probit-MLE 表分辨不了 10–20 logistic Elo；提出 Stage-1 定级 + Stage-2 带内分辨 + 独立 bank 确认 |
| [`evaluation-resolution-review.md`](./evaluation-resolution-review.md) | 上述设计的对抗性评审 | 判定「骨架可信、当前 revision 不可按原样推进」：预算被低估（实际 power≈0.69）、`--refit` 必然重选 rungs 与档位稳定冲突、「CI-only」与「+10 行动」口径混用；high 5 / medium 6 / low 11 |
| [`reward-alignment-saturate.md`](./reward-alignment-saturate.md) | Tier 1 分差饱和奖励 `saturate`（K_τ；τ∈{0,0.2,0.7} + `win` 锚）500k × 5 seed + 同 seed terminal 对照 | 主臂 K2（τ=0.2）主端点合并 **−18.31** [z −26.04,−10.58；t −29.27,−7.36]，winrate 0.474（5/5 seed 同号、训练 seed sd 8.82）→ **CI 全负且点 ≤+5 触发止损、关闭奖励饱和线**（saturate 家族 Holm p=6.9e-06）；剂量 K7（τ=0.7）**+7.27** [z −1.76,+16.31；t −5.53,+20.08] 未判定；端点锚 K0 −27.72、`win` −6.77；次端点 vs `l1_1M`：K2 −17.94 / K7 −5.57 / win −16.36 / K0 −30.10 / C0 −4.44；vs random P_w 0.869–0.901 未牺牲；无 V 塌缩（K2 raw V sd 0.073 但 EV +0.233，AND 不成立） |
| [`reward-alignment-terminal-win.md`](./reward-alignment-terminal-win.md) | Tier 1 边界跳变奖励 `terminal_win`（λ∈{0.25,1.0}）500k × 5 seed + 同 seed terminal 对照 | 主臂 J10 主端点合并 **+6.25** [z −6.31,+18.80；t −11.53,+24.02]，winrate 0.509（训练 seed sd 14.32 支配）→ CI 含 0、<+10 不达行动门（>+5 未触发止损）；J025 **−4.59** [−20.08,+10.90] 触发 ≤+5 止损；次端点 vs `l1_1M`：J10 −1.60 / J025 −5.36 / C0 −8.70（描述、CI 含 0）；vs random P_w 0.879–0.897 未牺牲、无 V 塌缩 |
| [`reward-alignment-diagnostic.md`](./reward-alignment-diagnostic.md) | 奖励-评估对齐离线诊断：分差在「胜负已定」后是否应饱和/对齐？ | 拓扑错位成立但**归因未判定**：`t17long[512k,2M]` return OLS **+0.0182/1M**（block-25 CI [+0.0100,+0.0240]，仅 seed 1）≈ **+1.0 own 分**，同期顶部 5 点 μ 187.6–191.6 不可分；配对加法分解 `ΔW`/`ΔL` **CI 全含 0**（`ATTR_RESOLVED=False`）→ 默认 **Tier 0 资源收束**（非「干预无效」证据）；奖励截断实验仅设计、不执行 |
| [`opponent-intent-probe.md`](./opponent-intent-probe.md) | 阶段 A 对手意图可识别性探针（`traces/study` 公开历史，CPU-only 离线） | 4000 局 / 195,066 前缀纯公开特征 + 泄漏自检；身份/强度强可分（D1 4 类 hist **0.639** vs count-only **0.440**、配对 Δ **+0.199 [+0.185,+0.213]**），但必要性门 **G-I = FAIL**：point_hold 配对 Δ 仅 **+0.0049 [+0.0018,+0.0081]**，严格读法（hist CI 下界 0.7255 < count boot99 0.7367）与 count-matched 分层（中位 −0.0002、14/31 层为正）都不达标；正信号集中在**残局算牌**（对手手牌 1–3 + 已出牌多），且 **v5 obs 在所有主任务上更强**（point_hold 0.783 vs 0.733、beats 0.861 vs 0.804）；P2 下一手≈先验（0.513 vs 0.524）；P4 冻结编码器意图可读性弱于 count-only → **按预注册停止，不进入阶段 B** |
| [`event-history-pilot.md`](./event-history-pilot.md) | T17D EVH：事件级 + 相对座位无状态历史编码器（实现 + random/self-play 主端点） | opt-in 第二输入（事件 64 维、相对座位、GRU；默认路径 55,179 参逐位不变、**680 passed**）；random 500k 3 seed 主端点 `event vs event_noise` **−5.02 [z −22.99,+12.95；t −44.47,+34.43]**、归属 `event vs event_seatblind` **+0.83 [−11.82,+13.47]** → 无 ≥+10 证据；self-play（用户拍板必做）端点 `selfevent vs selfeventnoise` **+2.99 [z −9.15,+15.14]**，3 seed 同号为正（seed sd 2.17）但幅度/CI 未达门槛 → 不 ship |
| [`history-fusion-ablation.md`](./history-fusion-ablation.md) | T17B 历史融合消融：v5 B1 历史通道在当前 pipeline 值多少 Elo | 当前模型并非无历史（v5 含 `unseen`54+`last_player`1）；训练+评估把 B1 等价置零后，3 训练 seed × 3×400 换座 h2h 合并 **Δ(盲−有史) +2.89 [−9.52,+15.31]**（历史优势 −2.89 [−15.31,+9.52]，CI 跨 0、<+10 门槛）→ 无证据表明 B1 值 ≥10 Elo；含下一档（序列编码器/GRU/逐座位历史）可行性草案 |
| [`sequence-memory-pilot.md`](./sequence-memory-pilot.md) | T17C 序列级记忆：D1-lite 无状态历史编码器原型 + self-play 下 B1 消融 | 新增 opt-in 序列编码器（`src/seven523/history.py`，默认关闭、`seq_len=0` 时逐位不变、**631 passed**）；random 500k 3 训练 seed ×3×400 h2h：关键端点 seq vs 容量盲对照 **+8.41 [−4.06,+20.89]**、seq vs `t17pool` +12.44 [−0.07,+24.94]；self-play 下 B1 历史优势 **+9.85 [−2.97,+22.68]**（单 seed）→ 均无 ≥+10 证据、不排期序列记忆；逐座位/墩归属/倾向需引擎改 `GameState.played` + v6/ADR |
| [`sequence-gru-ablation.md`](./sequence-gru-ablation.md) | T17C-P3 GRU 顺序消融：`--seq-order chrono/sorted` 隔离顺序/递归作用 + 条件超参调优 | 新增同架构/同容量/同 token 多重集的 sorted 臂（opt-in、默认不变、**636 passed**）；关键端点 chrono vs sorted **+10.46 [−2.46,+23.38]**（> +5 点估计线但 CI 跨 0）、bag 端点 sorted vs seqblind −1.99 [−24.89,+20.91]；因过 +5 线触发调优，`seq_emb 32`/`seq_hidden 64`/`seq_len 27` 三配置×3 seed 全部跑输 chrono 基线（−6.96/−7.59/−18.09，最后一个 CI 全负）→ 顺序效应存在但不可放大、未达 repo 行动门槛 |
| [`t17-recalibration.md`](./t17-recalibration.md) | T17 撬底 revision-3 重训与重测 | 新池 `traces/study`（4000 局，`rules_id=2e36dbea44893696`）lvl1–lvl4 = **82.75/106.90/147.71/185.34**（random=0，homoscedastic probit-MLE，σ≈5.9–6.3，相邻 CI 不重叠）；筛选 13 候选顶部平台 187.6–191.6、无 ≥200/无 ~55；self_500k 过拟合复核不可分；T2 与 `prior.json` 重标（RMSE **42.6**、m_eff **68.07**、σ=26.0/22.0/20.5）；T15 遗留 placement 常数按 `c=0.465621` 重标；旧 revision-2 study/pool10 已归档（`runs/archive/*-rules2-20260926/`），禁止与旧数字混比 |
| `runs/w2m/calibration/report.md`（原始产物；发布记录见其 §8.1） | 2026-09-27 w2m/T23 绝对评分标定（9 候选 + random，18,000 局共享 probit-MLE + resolution） | 三臂与冠军 `ws_s2` **不可分辨**（Δμ +4.1~+5.5，win_prob 95% CI 全跨 0.5）、`pself_s2` 打平；lvl4 落进冠军簇、lvl3 低一档；每对 200 deals 的 +10 Elo 行动力仅 0.19–0.34 → 只裁定大差距。**已按 owner 决定发布**为 10 级池（`traces/study`=`pool10`，sha256 `9bdb0edf…`，rungs 冻结 `lvl1`/`lvl4`），T2 标签重标；先验随后升级 **v2/R1**（`prior.json` sha256 `15fddff1…`，见下两行）|
| [`prior-opponent-correction.md`](./prior-opponent-correction.md) | S1 先验对手修正（选项 D）：诊断 + 候选实测 + R1 推荐 | 全 RR LOLO（40k）：误差分解 level_bias² 6.8% / cell_drift² 2.8% / **within_cell² 90.4%**——主因是特征衰减（raw 跨度 19.6、单局噪声 17.2、去收缩 b=3.28→56），不是对手外推；残差化/offset/Huber/rank/两级/placement 加权无净收益；推荐 **R1 = 152 列二次展开 + cell 惩罚 λ=3**：RMSE 52.87、m_eff 43.3、σ(5/10/20)=26.8/20.7/18.0、drift 7.79、cell_drift² 67；GBM headroom 42.6 →「RMSE≤35」不可达，验收改为 per-level bias + drift + placement σ |
| [`prior-v2-confirmation.md`](./prior-v2-confirmation.md) | 先验 v2（R1）独立发牌 bank2 复核 + 发布 | 独立 bank2（40k、9 级、相邻为主、seed 无重叠）transfer：v2/v1 RMSE **53.20/58.30**（−5.11）、σ(5/10)=**24.61/17.69** vs 27.33/19.60、drift **6.29/9.05**、逐级 \|bias\| 全 ≤1.5× → 判据 6/6 **CONFIRMED**；已把 `prior.json` 切到 study10 训练的 R1（sha256 `15fddff1…`），v1 4k 归档 |
| [`t15-recalibration.md`](./t15-recalibration.md) | T15 规则/估计器/身份变更后的重标定（revision-2 历史） | **已被 T17 superseded（2026-09-26 revision 3）**：新池 `traces/study`（4400 局，`rules_id=fbd43015d526ee72`）lvl1–lvl4 = 54.85/120.22/191.74/209.01（random=0，homoscedastic probit-MLE 绝对表，σ≈6.0–6.6）；T2 标签重导、`prior.json` 重标（RMSE 72.2、m_eff 23.49、σ=43.7/38.0/34.4）；旧 online PL 值（96.26/158.46/353.26/413.49）与旧 lvl4≈430 作废；该行全部数字仅作 revision-2 历史 |
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
| [`observation-augmentation-b0.md`](./observation-augmentation-b0.md) | T2 结构性 B0：观测增广 3 维（点分量）+ 兼容层 | `obs_version=2`（191→194）实现并测试通过；500k 训练 499,712 步；h2h 3×400 vs `base680k` +1.74 [−7.94,+11.42]、vs `w5_ctrl` +4.92 [−6.72,+16.56] 均跨 0 → 不单独采用；兼容层当时保留、后随 [ADR-0009](../adr/0009-single-observation-and-comparison.md) 退役，B1 后续已完成（未确认，见下行） |
| [`observation-augmentation-b1.md`](./observation-augmentation-b1.md) | T3 结构性 B1：已出牌历史 + `unseen` + `last_player` | `obs_version=3`（v2+55、2 家 249）实现并测试通过（当时 355 passed）；500k 499,712 步；h2h 3×400 vs `base680k` +4.20 [−6.75,+15.15]（主端点 CI 跨 0 → 失败）、vs `w5_ctrl` +13.04 [+1.93,+24.15]、相对 B0 +0.73 [−18.98,+20.43] → 未确认；B 线收束 |
| [`opponent-distribution-500k.md`](./opponent-distribution-500k.md) | T5 结构性 C：逐局对手 + PFSP + 强成员 | `EpisodeMixturePolicy`（逐局冻结）+ `pfsp_weights`（每 K=100 局 `w∝(1−wr)²+ε`、Beta(10) 收缩、平局计半、均匀 0.5 混合）+ `--pfsp*` 实现（+17 测试）；三臂 500k（`runs/t5_*__1__1790332696`，obs v1、seed 1、无 NaN）；8 个 h2h 3×400 CI 全含 0（PFSP 效应 +1.88、强成员 +2.90）→ 三效应 **null**；Greedy 1500 62.5–63.4% |
| [`twin-towers-500k.md`](./twin-towers-500k.md) | T6 结构性 F：独立 actor/critic 双塔 500k | `arch=towers` + 跨架构热启动实现并测试（+13；当时 418 passed）；三臂 500k、6 组 h2h 3×400：主臂 vs `base680k` −9.13 [−20.57,+2.31]、vs `w5_ctrl` +5.21 [−5.93,+16.36] 不可分，交互臂 vs A1 −15.51 [−27.10,−3.92]、从零 vs `w5_scratch` −20.89 [−33.81,−7.97]；梯度 cos +0.049 证伪干扰前提、critic 仍塌缩 → 关闭「架构忠实度」线 |
| [`human-elo-10-games-research.md`](./human-elo-10-games-research.md) | 真人 ≤10 局快速定级研究 | 10 局达不到 ±50（RMSE 54–72、CI ±100–133）；轨迹 S1 去收缩轨迹先验最强，轨迹 S3 有风格泄漏；RMSE≤50 需 15–27 局 |
| [`trace-prior-m1.md`](./trace-prior-m1.md) | T13-M1：轨迹 S1 先验离线标定 | `tools/fit_trace_prior.py` + `prior.json`（默认 T2：a=−890.92、b=1.7122、σ(5/10/20)=80.6/69.6/64.6、LOLO RMSE 133.9、m_eff 6.7；sha256 `1895af98…`）与 `prior_manifest_labels.json` 对照（a=−879.40、b=1.7060、σ=70.1/57.8/51.4、RMSE 128.6；sha256 `78e3535c…`）；数据 1200 局 × 6 级（指纹 `48c32672…`）；6 项测试 |
| [`placement-m2m3.md`](./placement-m2m3.md) | T13-M2/M3：10 局定级会话 + CLI | `placement/` 包（info/Thompson 选档、10 副不同牌、座位 5/5、两通道 BT-MAP `margin=(0.143,89)`、逐局 trace 与 `session.json`、`report.json`+`rungs.json`）+ `7g523-elo` 入口 + `play.py` rung 标签；`elo.py` 零改动（sha256 `80641f13…`）；冒烟 `定级：1287 ± 182，最近档 greedy（1315），provisional`；当时全套 418 passed；M4 等 D-3 |
| [`ladder-rerating-paired.md`](./ladder-rerating-paired.md) | 冻结阶梯 `lvl1–lvl4` 换座配对复测 | lvl4 旧值高估 60 Elo（1489→**1430±7**）、lvl1 旧值低估 26；排序 5/5 保持但间距变成 73/144/66（spacing 契约失效）；refit 需多 seed 合并 + owner 批准，旧 manifest 未动 |
| [`evaluation-protocol-validation.md`](./evaluation-protocol-validation.md) | 新评估口径独立验证 | 座位配对/聚簇 bootstrap/√N 均成立；判定规则需统一（+20 门槛使 Δ=20 功效仅 ~50%）；修复 `elo.py` 联合 Hessian 重复计数（SE 最多低估 19%）；80% power 需 807/3226 副（20/10 Elo） |
| [`observation-slimming.md`](./observation-slimming.md) | 观测精简审计与 200k pilot | 191→103（观测 S1）信息无损（`rank_counts`⊆`hand` 0/45752 不一致等）；h2h 与 full 不可分辨（+4.4[−18.3,+27]）；观测 S2 68 维落后 ~15 Elo（丢 rank↔suit 配对）；建议实施观测 S1、弃观测 S2 |
| [`tournament-arena.md`](./tournament-arena.md) | 并行 checkpoint 锦标赛竞技场 | 6 worker 22,620 局 37.8s、W=1 vs W>1 逐位一致；bf16 仅 +2% 已移除；单 seed 排名不可靠（两 seed ρ=0.54）；进度曲线：lvlbase ~184k 后平台、lvlsp 全程平，无新强度 |
| [`v5-optimization-wave1.md`](./v5-optimization-wave1.md) | memoryless v5 优化第一波：self-play k=7 确认 + A2 容量粗筛 | F1b 主端点（7 fresh 训练 seed × 9 fresh deal seed × 400 副，self_s2..s8 vs 固定 `l1_1M`）合并 **Δ +2.87 Elo**（z-CI [−3.37,+9.11]、t-CI [−4.92,+10.66] 均跨 0；训练 seed sd 8.43 为 binding term）→ 点 ≤ +5 **止损**、不跑 Stage 2（warm-start/1M/refresh50/静态池无数据）；F1a 同 seed 读数 **+23.99/−6.13/+17.30** 印证训练 seed 方差；F0 共享 fit 锚 μ(self_s2)−μ(l1_1M)=+2.52（>−10，无绝对退化）；Block C cap256 −33.56 / cap512 +19.15（点 <+20，**无 ≥+20、不构成关闭 A2 的证据**）；7 个 self 臂末熵 1.33–1.55 全 <1.6（机制观察）；共 10 run = 8.2 等效 ≤ 15 |
| [`selfplay-pool-diagnostics.md`](./selfplay-pool-diagnostics.md) | self-play 退化机制离线诊断（D0–D5，现有 ckpt，CPU 串行） | D0 门**不过**：`p2/p3_500k` 对 `l1_1M` 非劣失败（Δanchor **−8.42 / −13.13**，下界未 > −10）→ 池臂只能作 **"弱成员池重估"**；D1 12 节点 round-robin（31,200 局、220 三元组、13 成环、最强三元组最小边差 **33.3 < MDE 34**）→ **H-cycle 不可判**；D2 δ **−0.0065 [−0.050,+0.032]**、D4 MA CI 含 0 → H-mirror **未检出**；D5 self 模板头熵 **0.2145** vs random **0.3257**（差 −0.111 [−0.121,−0.103]）→ H-det **弱旁证**；D3 n=7 不可判 |
| [`selfplay-pool-wave1.md`](./selfplay-pool-wave1.md) | self-play 池化对手第一阶段（A1 `poolself` / A2 `fixed` / A3 `selfsamp`，cold start k=5） | 预注册主端点（各 vs 同 seed `C0`，5 训练 seed × 9 fresh deal × 400 副换座）：A1 **−2.71** [z −8.13,+2.71；t −10.39,+4.96]、A2 **−0.67** [z −11.12,+9.78；t −15.48,+14.14]、A3 **+0.25** [z −7.90,+8.40；t −11.30,+11.80]，Holm 全不拒 → 三点估计 ≤ +5 **全部止损**、无机制获支持（CI 全跨 0 → **"未判定"、非 null**）；D0 未过门 → 只作 **"弱成员池重估"**，不得宣称强成员池；描述性 A1 vs `l1_1M` **+6.34** [z +1.22,+11.47；t −0.92,+13.60]（500k vs 1M 预算不匹配）；15 run = 硬上限 |

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
# test-time 对手意图学习：阶段 A 可识别性探针报告（7鬼523 / PPO）

> **状态：已完成。闸门判定 G-I = FAIL → 按预注册停止规则不进入阶段 B。**
> 规格来源：[`../opponent-intent-plan.md`](../opponent-intent-plan.md) §3（阶段 A）+ §6 步骤 1–5；
> 红队修正：[`../opponent-intent-review.md`](../opponent-intent-review.md)。
> **口径**：revision-3（出空即撬底）、obs v5（2 家 161 维）、trace `version=3`、
> `rules_id=2e36dbea44893696`（仅记在 `traces/study/manifest.json`）。
> 全程 CPU-only、无 GPU 训练、无 h2h；只写 `runs/opponent_intent_probe/` 与本报告 + README §1 索引；
> 未改 `src/tests/tools/ADR/plans.md/CONTEXT/DESIGN`；未 commit/checkout/stash。

---

## 0. 结论摘要（TL;DR）

1. **数据侦察（§2）**：`traces/study` 共 **4000 局**，构成为 6 个跨等级对 + 4 个 vs random，
   各 400 局；同一 seed 跨等级打**同一副发牌**（400/400 seed 逐位相同）；random 锚块与跨等级块
   seed 交集为空；**200 局抽样回放 0 失败**、`final_scores` 全部一致。原料充足，阶段 A 无需生成新对局。
2. **公开历史里确实有信号（§5）**：身份/强度可**强**识别（D1 4 分类 0.639 vs count-only 0.440，
   配对 CI [+0.185,+0.213]）；**下一手意图也有可测信号**（point_hold 配对 Δ=+0.0049，
   CI [+0.0018,+0.0081]；能否压过 incumbent 的 Δ=+0.0435，CI [+0.041,+0.046]）。
3. **但预注册的必要性门 G-I 未通过（§6）**：
   - G-I.1（聚合配对 Δ 的 CI 下界 > 0）**PASS**；
   - G-I.1b（严格读法：hist 95% CI 下界 > count-only bootstrap 99 分位）**FAIL**
     （0.7255 < 0.7367）；
   - G-I.2（count-matched 分层：`e` 桶 × 对手手牌数，中位 Δ>0 且过半层为正）**FAIL**
     （中位 **−0.0002**、仅 **45%（14/31）** 层为正）。
   **G-I = G-I.1 ∧ G-I.2 = FAIL**。
4. **信号的形态很关键**：正 Δ 集中在**残局且对手手牌很少**（`e` 桶 3/4、对手手牌 1–3 张），
   即「已出牌多 + 剩牌少」时的**算牌式端局推断**，而非跨局稳定的「对手风格/意图」；
   并且在所有端点/指标上，**现有 memoryless v5 观测（161 维）都明显强于事件内容模型**
   （point_hold：v5 0.783 vs hist 0.733；beats：v5 0.861 vs hist 0.804；P2：v5 0.524 vs hist 0.513）。
5. **适应（G-D）**：P1 point_hold 的 game-paired DID = **+0.0250 [+0.0164,+0.0337]（PASS）**，
   P2 next-action DID = +0.0015 [−0.0065,+0.0099]（FAIL）。但 P1 的桶内**类别构成漂移显著**
   （正例占比 0.781→0.604），用 balanced-accuracy/AUC 复核后 DID 降为 **+0.0147 / +0.0811**
   （仍正、但更小），说明准确率 DID 含基准率成分。
6. **P4 冻结编码器**：现有 `t17event` 的事件编码器 hidden **弱可读身份**（D1 0.341 vs 先验 0.25；
   lvl4-vs-random 0.750）、**几乎不编码意图**（point_hold 0.711，低于 count-only 0.728）。
7. **决策**：按预注册，**G-I 不成立 → 停止，不投阶段 B 训练**。这是一个「有弱信号、但不满足必要性门、
   且被 v5 覆盖、且集中在端局算牌」的负结果，而非「零信号」。

---

## 1. 预注册（在跑主实验前冻结）

以下为 `runs/opponent_intent_probe/probe.py` 中 `PREREG` 的原文摘要；主实验（§5）在本节冻结之后执行。

| 项 | 预注册值 |
|---|---|
| 数据 | `traces/study` 全部 4000 局，观测者=当前行动座位的前缀 |
| 切分 | 5-fold GroupKFold，`group = trace seed`（排序后 rank % 5）；同 seed 全部局同折 |
| 模型 | minibatch Adam softmax 回归，特征标准化，L2=1e-4；`epochs=40, batch=2048, lr=1e-2, seed=0` |
| 桶界（已观测对手事件数 e） | `[0,4)/[4,8)/[8,12)/[12,18)/[18,∞)`（由 §2 实测 per-prefix 分布定） |
| 主端点 | P1 `point_hold`：`hist`（count+content_opp，score-blind）vs `count-only` |
| G-S（健全性，非门） | P0 D1 身份 score-blind hist 显著优于 count-only（配对 CI 下界>0） |
| G-I（必要性门） | **G-I.1** 配对 Δacc(hist−count) 的 seed-簇 95% CI 下界>0；**且 G-I.2** 在 count-matched 分层（e 桶 × 对手手牌数，n≥200）中中位 Δ>0 且过半层为正 |
| G-F（风格 vs 强度） | P0 D1 身份在 count-matched 分层中保留信号（同 G-I.2 判据） |
| G-D（适应机制门） | P1/P2 的 game-paired DID `[hist_late−hist_early]−[count_late−count_early]` seed-簇 CI 下界>0 |
| 多重比较 | 主端点 α=0.05；辅助端点 Holm（族 G_S/G_F/G_I_2/G_D/P2/P4） |
| bootstrap | seed 簇、B=2000、seed=12345；置换零分布 10 次（固定 20% seed 留出） |
| 依赖 | 不新增依赖（`.venv` 无 sklearn/scipy）；numpy 2.5.3 + torch 2.14.0（仅 P4 用 CPU 前向） |

**对 G-I.1b 的说明**：原文「P1 accuracy 95% CI 下界 > count-only 99 分位」有歧义。为不事后改门线，
本报告同时给出两种读法：(a) 配对 Δ 的 CI 下界>0（=G-I.1，主判据）；(b) 字面严格读法
「hist acc bootstrap 2.5 分位 > count-only acc bootstrap 99 分位」（=G-I.1b）。两者都报，判定用
**G-I.1 ∧ G-I.2**（规划 §3.6 明确 G-I 是两条的合取）。

---

## 2. 数据侦察（`recon.py`）

命令：`.venv/bin/python runs/opponent_intent_probe/recon.py`（只读；`RECON_N=20` 每层抽样）。

**2.1 对局构成**（按目录 × 对手，`g\d+__s…__seat…__vs{opp}.json`）：

| 目录 | vs random | vs lvl2 | vs lvl3 | vs lvl4 | 合计 |
|---|---|---|---|---|---|
| `lvl1/` | 400 | 400 | 400 | 400 | 1600 |
| `lvl2/` | 400 | — | 400 | 400 | 1200 |
| `lvl3/` | 400 | — | — | 400 | 800 |
| `lvl4/` | 400 | — | — | — | 400 |

⇒ 共 4000 局；跨等级 6 对 ×400 存在，`traces/pool10/` **只有 manifest、无对局**（核实）。

**2.2 配对结构**：400 个 seed 出现于 ≥2 个等级，其中 **400/400 的 `initial.hands` 逐位相同**
（同 seed 打同一副牌）；random 锚块 200 seed、跨等级块 200 seed、**交集 0**。`players` 形如
`["subject:lvl1@seat0","anchor:random@seat1"]` 或 `["subject:lvl1@seat0","candidate:lvl2@seat1"]`
（`trace.py:120,125`）。

**2.3 回放充分性**：用 `state_from_snapshot`+`Game.restore`+`Game.step` 复现抽样 200 局，
**0 失败**，每局 `final_scores` 与逐步重算完全一致 ⇒ trace 字段足够 replay 公开 `View`。
`initial` 含完整手牌/底牌（`trace.py:84`），**探针特征绝不读它**（§4 强制）。

**2.4 前缀规模与桶**（全 4000 局、双观测者，195,066 个前缀）：

| 量 | 值 |
|---|---|
| 前缀总数 | 195,066（4000 局，观测者=行动座位） |
| `e`（已观测对手事件数） | min 0 / max 37 / mean 12.40 / median 12 |
| 桶样本数 `[0,4)/[4,8)/[8,12)/[12,18)/[18,∞)` | **28000 / 32000 / 32000 / 47945 / 55121** |
| `next_kind` 边际（0..5=kind, 6=pass, −1=无） | pass 57573、single 95627、pair 14574、straight 18703、small_bomb 4542、big_bomb 47、无 4000 |
| 标签先验 | `point_hold` 正 0.711；`bomb_hold` 正 0.048；`beats_inc`（仅 incumbent≠None）正 0.548 |

---

## 3. 方法

**3.1 样本单位**：`(game, observer_seat=当前行动座位, prefix)`。回放一条 trace，在每个
`st.current` 决策点，用 `Game.view(st, current)` 抽取纯公开特征；标签取对手隐藏信息（y only）。

**3.2 特征组**（`extract.build_x(view, rules, opp_seat)`，**签名只接受 View+Rules+seat**）：

| 组 | 维度 | 内容（全部来自 `View`） |
|---|---|---|
| `count` | 15 | e、总事件数、自身/对手手牌数、摸牌堆、桌面牌数、已出牌数、incumbent 有无、last_player、pass 率、开墩数、对手出空数 |
| `score` | 6 | running score 差/各分、trick_points、remaining_points、自身持分 |
| `content_opp` | 182 | 对手 ComboKind 直方(7)、对手已出牌 multihot(54)、对手 kind 一阶转移(7×7)、最后一个对手 kind(7)、开墩/出空/均码、revealed(54)、incumbent kind(6)+size |
| `content_self` | 70 | 自身 kind 直方/已出牌/最后 kind/pass 率/均码（信息性 nuisance，仅诊断） |
| `v5` | 161 | `encode_observation(view, rules)`（现有 RL 观测，非 score-blind） |
| `oracle` | 54 | **决策点当时** `state.hands[opp]` multihot（仅哨兵，不进任何报告模型） |

特征集：`hist = count+content_opp`（**主、score-blind**）、`full = count+score+content_opp`、
`content = content_opp`、`count`、`score`、`v5`、`oracle`。

**3.3 任务**：P0 身份（D1 观测者=random 对 {lvl1..4}；D2 固定观测者子任务；
D3 每个 lvl_i vs random 双向）；P1 意图代理（point_hold / bomb_hold / beats_inc）；
P2 下一手（对手下一个动作的 ComboKind+pass，公开候选空间，先验=全局边际）；
P4 冻结编码器线性探针。

**3.4 统计**：out-of-fold 预测（GroupKFold by seed）；seed 簇 bootstrap 95% CI；
配对 Δ 用同一批样本的两模型差；置换零分布用固定 20% seed 留出的标签打乱。

---

## 4. 泄漏自检（`selfcheck.py`，全部通过）

| # | 检查 | 结果 |
|---|---|---|
| 1 | **静态扫描** `build_x` 的 AST：参数仅 `view, rules, opp_seat`；名字/属性不触碰 `state/st/initial/steps/final_scores/hands/draw_pile/empty_order/phase` | **PASS**（0 命中） |
| 2 | `initial` 在 `extract.py` 只出现于 `state_from_snapshot(d["initial"])`（+ 注释），`state.hands` 只出现在 `build_labels`（y）与 oracle | **PASS** |
| 3 | 特征只依赖 `view.plays` 前缀：任何未来步骤不影响已抽前缀（由签名与构造保证） | **PASS** |
| 4 | **oracle 哨兵**（决策点当时对手手牌 multihot）→ `point_hold` acc **1.0000** | **PASS**（>0.98） |
| 5 | seed 切分：5 折 GroupKFold 两两 seed 交集为空；同 seed 全部局同折 | **PASS** |
| 6 | 回放完整性：抽样回放 `final_scores` 全部一致（§2.3） | **PASS** |
| 7 | label shuffle（hist 特征、打乱 y）acc **0.7097** ≈ 多数类先验 **0.7101** | **PASS** |
| 8 | confound 哨兵：`count`/`score` 单独**不能**复现 `v5` 级信号（point_hold：count 0.732、score 0.731、v5 0.783） | **PASS** |

补充：历史上 §3.8.4 警告「`initial.hands` 跨类恒定、用它当 oracle 会落 chance」——本探针用
**决策点当时**的手牌，point_hold oracle=1.0 证实哨兵有效；身份 oracle（0.604）低于 hist（0.639），
说明身份信息更多在**动作历史**而非单帧手牌，符合直觉。

---

## 5. 结果

### 5.1 P0 身份/强度（健全性，预期 pass）

准确率（OOF，5 折）：

| 任务 | n | hist | count | v5 | full | oracle | 先验 |
|---|---|---|---|---|---|---|---|
| D1（obs=random, 4 类） | 40722 | **0.639** | 0.440 | 0.522 | 0.641 | 0.604 | 0.267 |
| D2_l1（obs=lvl1→{2,3,4}） | 28968 | **0.687** | 0.513 | 0.573 | 0.689 | 0.639 | 0.360 |
| D2_l2（obs=lvl2→{3,4}） | 18693 | **0.756** | 0.693 | 0.676 | 0.758 | 0.716 | 0.522 |
| D3 lvl1 vs random | 21775 | **0.782** | 0.731 | 0.764 | 0.786 | 0.744 | 0.501 |
| D3 lvl2 vs random | 21711 | **0.816** | 0.757 | 0.800 | 0.818 | 0.783 | 0.502 |
| D3 lvl3 vs random | 20137 | **0.837** | 0.797 | 0.822 | 0.846 | 0.786 | 0.504 |
| D3 lvl4 vs random | 18282 | **0.892** | 0.841 | 0.830 | 0.892 | 0.783 | 0.505 |

D1 hist vs count 配对 Δ **+0.1991**，CI **[+0.1847,+0.2130]**；count-matched 分层中位 +0.198、
**96%（26/27）层为正**。⇒ **身份/强度可强识别（H1）**，与规划预期一致（这是 G-S，非研究结论）。

### 5.2 P1 意图代理（主必要性门）

| 任务 | n | hist | count | v5 | content | score | full | oracle | 先验 |
|---|---|---|---|---|---|---|---|---|---|
| point_hold | 195066 | 0.7332 | 0.7283 | **0.7829** | 0.7187 | 0.7260 | 0.7794 | **1.0000** | 0.711 |
| bomb_hold | 195066 | 0.9519 | 0.9521 | 0.9521 | 0.9520 | 0.9521 | 0.9519 | 0.9530 | 0.952 |
| beats_inc（incumbent≠None） | 133493 | 0.8038 | 0.7603 | **0.8605** | 0.7674 | 0.6568 | 0.8037 | 0.7399 | 0.548 |

- **point_hold**：hist 仅微优于 count（Δ=+0.0049 [+0.0018,+0.0081]），但 **v5 高出 hist ~5pp**；
  置换零分布 Δ mean −0.0013、max +2.6e-5，观测 Δ=+0.0049 高于零分布 ⇒ 聚合信号真实。
- **beats_inc**：hist 明显优于 count（Δ=+0.0435 [+0.0410,+0.0462]），且分层中位 +0.0048、
  69% 层为正；但 **v5 仍高出 hist ~5.7pp**。
- **bomb_hold**：三类指标都≈先验（AUC 0.60–0.72 且 oracle 也上不去）⇒ **无可用信号**。
- oracle 在 beats 上（0.740）低于 count，是因为 oracle 只含对手手牌、不含 incumbent，单帧信息不足；
  这是特征局限，不是泄漏问题。

### 5.3 P2 下一手预测（公开候选集）

| 任务 | n | hist | count | v5 | content | score | full | oracle |
|---|---|---|---|---|---|---|---|---|
| next ComboKind(+pass) | 191066 | 0.5126 | 0.5084 | **0.5241** | 0.5099 | 0.5027 | 0.5125 | 0.5167 |

先验：single 0.500、pass 0.301、straight 0.098。**所有特征组都只比 majority 先验高 1–2.4pp**；
hist 相对 count 的配对 Δ=+0.0042 [+0.0023,+0.0062]（聚合可测），但幅度极小、v5 最好。
⇒ 公开历史对「对手下一手牌型」几乎没有可预测性。

### 5.4 P4 冻结编码器线性探针（诊断）

`runs/t17event__1__1790452506/agent.pt` 的 `EventSequenceEncoder`（event_len=108, hidden=32），
冻结后取 hidden，在 2000 局子集（97,765 前缀）上跑线性探针：

| 任务 | n | hidden acc | hidden AUC | hist | count | v5 |
|---|---|---|---|---|---|---|
| D1 身份 4 类 | 20642 | 0.341 | 0.586 | 0.611 | 0.443 | — |
| D3 lvl4 vs random | 10114 | 0.750 | 0.820 | 0.891 | 0.847 | — |
| P1 point_hold | 97765 | 0.711 | 0.603 | 0.733 | 0.728 | **0.782** |
| P1 beats | 66829 | 0.762 | 0.831 | 0.805 | 0.758 | **0.860** |

⇒ 现有未做对手建模的编码器：身份/强度**弱线性可读**，意图信息**弱于 count-only**。
即使换成训练该编码器做 in-context 推断，其 hidden 也几乎没有可用意图结构。

### 5.5 适应曲线与 DID

按 e 桶的准确率（hist / count / n）：

| 任务 | b0 [0,4) | b1 [4,8) | b2 [8,12) | b3 [12,18) | b4 [18,∞) | DID点 | game-paired DID（95% CI） |
|---|---|---|---|---|---|---|---|
| P1 point_hold | .777/.783 /28000 | .760/.763 /32000 | .748/.749 /32000 | .740/.736 /47945 | .681/.662 /55121 | +0.0251 | **+0.0250 [+0.0164,+0.0337]** |
| P1 beats | .850/.814 /18242 | .825/.782 /22543 | .821/.774 /22451 | .814/.767 /33436 | .748/.706 /36821 | +0.0064 | **+0.0151 [+0.0051,+0.0257]** |
| P2 next_kind | .451/.450 /28000 | .474/.471 /32000 | .485/.480 /32000 | .499/.493 /47899 | .600/.596 /51167 | +0.0039 | +0.0015 [−0.0065,+0.0099] |
| P0 D1 身份 | .408/.316 /5600 | .598/.391 /6400 | .674/.449 /6400 | .708/.466 /9590 | .691/.493 /12732 | +0.1055 | +0.0919 [+0.0662,+0.1164] |

**类别构成漂移检查（修正 stats-ml-03）**：P1 point_hold 桶内正例占比 b0 **0.781** → b4 **0.604**，
漂移显著。用 balanced-accuracy / AUC 复核（避免基准率成分）：

| 指标 | b0 | b1 | b2 | b3 | b4 | DID(late−early) |
|---|---|---|---|---|---|---|
| hist balanced-acc | .521 | .528 | .536 | .556 | **.651** | hist +0.130 |
| count balanced-acc | .504 | .503 | .503 | .508 | **.619** | count +0.115 ⇒ **Δ=+0.0147** |
| hist AUC | .571 | .625 | .643 | .671 | **.725** | hist +0.154 |
| count AUC | .608 | .608 | .605 | .608 | **.681** | count +0.073 ⇒ **Δ=+0.0811** |

关键结构：**count-only 的 balanced-acc 在 b0–b3 几乎贴 chance（~0.50）**，其准确率主要由先验驱动；
hist 的 balanced-acc/AUC 随 e 单调上升，但**集中体现在 b4**，且 b4 的优势又主要来自
**对手手牌 1–3 张**的残局（见 §6.1 分层明细）。**v5 在 b4 的 balanced-acc=0.833、AUC=0.914**，
远超 hist（0.651/0.725）——即「晚期可用信息」几乎已被 v5 观测覆盖。

---

## 6. 闸门判定（`gates.py` → `gates.json`）

| 闸门 | 判据 | 结果 | 证据 |
|---|---|---|---|
| **G-S**（健全性，非门） | D1 身份 score-blind 配对 CI 下界>0 | **PASS** | Δ=+0.1991 [+0.1847,+0.2130] |
| **G-I.1** | P1 point_hold 配对 Δ CI 下界>0 | **PASS** | Δ=+0.0049 [+0.0018,+0.0081]，p=0.001；> 置换零 max 2.6e-5 |
| **G-I.1b**（严格读法） | hist acc 95% CI 下界 > count bootstrap 99 分位 | **FAIL** | 0.7255 < 0.7367 |
| **G-I.2** | count-matched 分层中位 Δ>0 且过半层为正 | **FAIL** | 中位 −0.0002、14/31=45% 为正 |
| **G-I（必要性门）** | G-I.1 ∧ G-I.2 | **FAIL** | 见上 |
| **G-F**（风格 vs 强度） | D1 身份 count-matched 分层保留信号 | **PASS** | 中位 +0.198、26/27=96% 为正 |
| **G-D(P1)** | point_hold game-paired DID CI 下界>0 | **PASS** | +0.0250 [+0.0164,+0.0337] |
| **G-D(P2)** | next_kind game-paired DID CI 下界>0 | **FAIL** | +0.0015 [−0.0065,+0.0099] |
| **P4** | 冻结编码器线性可读 | 弱 | point_hold hidden 0.711 < count 0.728 |

### 6.1 为什么 G-I.2 失败（关键诊断）

P1 point_hold 的 31 个 count-matched 层（n≥200）中，**14 层为正、17 层为非正**，中位 −0.0002。
正 Δ 集中在两端：

- **对手手牌少（1–3 张）+ 晚期**：如 e∈[18,∞)、手牌=2，hist 0.632 vs count 0.576（Δ=+0.056）；
  e∈[12,18)、手牌=2，Δ=+0.125。
- **早期/中手牌**：Δ≈0 或略负（如 e∈[0,4)、手牌=4，Δ=−0.029）。

⇒ 额外可识别性来自**「对手剩牌少、公开牌多」时的端局算牌**，而非稳定的对手策略身份。
这正是 count-only 在早期无法区分、而内容模型在残局能用的原因。

---

## 7. 结论与阶段 B 决策

**结论**：在 2 家、revision-3、obs v5 不变、现有 `traces/study` 分布与预算下：

1. 公开历史可**强**识别对手身份/强度（H1，G-S/G-F PASS）——这不是研究结论。
2. 公开历史对对手**隐藏意图**有**可测但弱且不稳健**的信号（H2 部分）：
   - point_hold 聚合 Δ 显著但极小，且**不满足预注册的严格/ count-matched 必要性判据**（G-I FAIL）；
   - beats_inc 的 Δ 较大且分层较稳，但**仍被 v5 覆盖 ~5.7pp**、且 oracle 表明信号可能在 incumbent 而非历史；
   - 下一手（P2）基本不可预测；bomb_hold 无信号。
3. 额外信号**不是**跨局稳定的「对手风格/意图」，而是**端局算牌**（对手剩牌少+已出牌多），
   并且**几乎全部包含在现有 v5 观测里**（v5 在所有主任务上显著更强）。
4. **适应（G-D）**：P1 的准确率 DID 通过，但受类别构成漂移影响（balanced/AUC 复核后减小），
   且其增量主要由 v5 已提供的信息构成；P2 DID 不通过。

**决策（按预注册停止规则）**：**G-I 不成立 → 不进入阶段 B 训练**。不触发对手池/aux 头/适应曲线训练实验。
这一结论与「历史/顺序/事件/座位」前序四线的负结果一致，并新增了机制性解释：
**公开历史的意图信息是端局算牌式的、已被 v5 覆盖，不足以支撑 in-context 对手建模训练。**

**若要改判需出现的新证据**（本工作流不执行，供后续参考）：
- 构造**真正异质、身份持久、风格可分**的对手池，并证明在**相同 count 条件**下存在稳定正 Δ；
- 证明该信号**不能被 memoryless obs 复现**（即 hist ≫ v5，而非 v5 ≫ hist）；
- 用 balanced-accuracy/AUC（而非原始准确率）作为主端点，消除类别构成漂移。

---

## 8. 局限

1. **对手池同源且强度单调**：四个 subject 是同一 T17 lineage 的不同快照/训练时长，观测到的
   「身份」信号高度可能是**强度**而非**风格**（G-F 只证明它在 count/score-blind 下保留，不能区分二者）。
2. **端点代理性**：`hist`/`count` 是**线性** bag/转移特征，不是 GRU；可能低估时序编码器的潜力
   （但 P4 显示现有 GRU 的 hidden 意图可读性**更差**，缓解该担忧）。
3. **类别构成漂移**：P1 桶间正例占比漂移大，原始准确率 DID 含基准率成分；已用 balanced/AUC 复核。
4. **P4 子集**：2000 局/9.8 万前缀，且编码器来自另一训练分布（t17event），只作诊断。
5. **多重比较**：主端点独立判据为配对 Δ；辅助端点未逐一对置换零分布做 Holm（报告了原始 p/CI），
   但由于 G-I 因 G-I.2 的**结构性**失败而失败，Holm 不会改变结论。
6. **无新对局**：全部结论限于 `traces/study` 的 random/快照分布，外推到自对弈/混合池不成立。

---

## 9. 复现命令

```bash
cd /home/amas/.local/src/7g523
# 1) 数据侦察（只读）
.venv/bin/python runs/opponent_intent_probe/recon.py
# 2) 前缀抽取（16 进程，~13s，产出 prefixes.npz）
.venv/bin/python runs/opponent_intent_probe/extract.py --workers 16 \
    --out runs/opponent_intent_probe/prefixes.npz
# 3) 泄漏自检
.venv/bin/python runs/opponent_intent_probe/selfcheck.py
# 4) P0-P3 主探针（B=2000，~4.5 min，产出 probe_results.json + p1_point_oof.npz）
.venv/bin/python runs/opponent_intent_probe/probe.py --B 2000 \
    --out runs/opponent_intent_probe/probe_results.json
# 5) P4 冻结编码器（~1.4 min）
.venv/bin/python runs/opponent_intent_probe/frozen_probe.py --games 2000 \
    --out runs/opponent_intent_probe/frozen_probe.json
# 6) 闸门裁决
.venv/bin/python runs/opponent_intent_probe/gates.py
```

环境：Python 3.12.14、numpy 2.5.3、torch 2.14.0+cu132（P4 前向在 CPU）、32 核；
git `ce22681af7162f75e1d40a0d701454d11640e985`（工作区含 T15–T17 未提交改动）。
**`runs/` 被 `.gitignore` 忽略**（`.gitignore:8`），故脚本不随仓库版本化；下表给 sha256 供独立复核。

产物（均在 `runs/opponent_intent_probe/`）：

| 文件 | sha256 | 说明 |
|---|---|---|
| `recon.py` | `98b7f446…` | 数据侦察 |
| `extract.py` | `b136b4d7…` | 前缀/特征/标签抽取 |
| `probe.py` | `2688048d…` | P0–P3 + DID + 置换 |
| `frozen_probe.py` | `1ccbd66d…` | P4 |
| `selfcheck.py` | `c4f38642…` | 泄漏自检 |
| `gates.py` | `0da0ee3f…` | 闸门裁决 |
| `prefixes.npz` | `dee5baa4…` | 195,066 前缀特征 |
| `probe_results.json` | `73c3fb0c…` | 全部 OOF 指标 |
| `frozen_probe.json` | `91cfd221…` | P4 指标 |
| `gates.json` | `c032e8a3…` | 闸门判定 |

### 9.1 关键代码内联（因 `runs/` 不版本化）

**(a) 公开特征构造的签名与边界（`extract.py`）**——AST 扫描证明它不触碰隐藏/未来：

```python
def build_x(view, rules, opp_seat):
    # 仅用 view.plays / view.hand / view.counts / view.draw_count /
    # view.trick_cards / view.played / view.incumbent / view.revealed /
    # view.last_player / view.scores —— 全部 ADR-0002 公开投影
    ...
# 标签与 oracle 单独在 build_labels(st, rules, opp_seat) 中，读 state.hands[opp]（仅 y / 哨兵）
```

**(b) 模型与门线（`probe.py`）**：

```python
PREREG = {  # 主实验前冻结
  "cv": "5-fold GroupKFold by trace seed",
  "model_hparams": {"epochs": 40, "batch": 2048, "lr": 1e-2, "l2": 1e-4, "seed": 0},
  "buckets_observed_opp_events": [0, 4, 8, 12, 18, 10**9],
  "primary_endpoint": "P1 point_hold: hist(count+content_opp, score-blind) vs count-only",
  "G_I_1": "paired dacc (hist-count) seed-clustered 95% CI lower > 0",
  "G_I_2": "paired dacc size-weighted mean over count strata; >half of eligible strata positive",
  "G_D": "[hist_late-hist_early]-[count_late-count_early] game-paired seed-clustered CI lower > 0",
}
# 所有对照与处理臂使用同一 SoftmaxReg（numpy minibatch Adam），仅特征组不同。
```

**(c) 闸门合取（`gates.py`）**：

```python
g_i1 = dci[0] > 0                                   # 配对 Δ CI 下界 > 0
strata_ok = (median_delta > 0) and (frac_positive > 0.5)   # count-matched
G_I = "PASS" if (g_i1 and strata_ok) else "FAIL"    # => FAIL（strata_ok=False）
```

---

## 10. 与规划的对照（阶段 A 步骤验收）

| 规划步骤 | 状态 | 产物 |
|---|---|---|
| 1 核实/冻结预注册（桶界、MDE 依据） | 完成 | §1、§2、`recon.py` |
| 2 extract.py + 泄漏自检 | 完成 | `prefixes.npz`、§4 |
| 3 P0 身份 + P1 意图 + 对照 | 完成 | §5.1–5.2、`probe_results.json` |
| 4 P2 下一手 + P4 冻结编码器 | 完成 | §5.3–5.4 |
| 5 闸门裁决 + 报告 + README §1 | 完成 | §6、本文件、README 索引 |
| **止损点** | **触发** | G-I FAIL → **不进入阶段 B**（步骤 6–10 不执行） |

**未做且按预注册不需要做的**：阶段 B 的对手池/aux 头/训练/适应曲线 h2h（G-I 未过门）。

# 评估分辨率设计：定级表 + 带内 CRN round-robin 重排（revision-3/T17 口径）

> **状态：设计文档 v2（已按评审修订，2026-09-27；评审见
> [`evaluation-resolution-review.md`](./evaluation-resolution-review.md)）。**
> 本文是设计提案，只改本文件、不改代码；所有结论给 `file:line` / `§` 出处；本文新增的
> 推算标注「本文推算」并在附录 A 给出公式，只读复算脚本见附录 B（可用
> `runs/t17_mle/games.jsonl` 复现）。修订假设以下**同批实施的新能力存在**并与之对齐：
> `tools/refit_mle.py` 的 `--keep-rungs`（resolution 路径冻结 `rungs`，配
> `estimator.rungs_source="kept"`/`rungs_selection_ok`/`rungs`）、`--prior-sigma`（显式先验
> 宽度，记 `estimator.prior_sigma`）、`--bootstrap-workers`（CLI 默认 4 进程，与串行逐位一致）、`--band` 与 `resolution` 发布块（`power_at_delta` 与
> `action_power_at_delta` 分开）；`study.py` 的冻结 rungs 合并与 `resolution` schema 校验。
> 逐条修订记录见文末「修订记录」。
>
> **规则口径**：当前有效口径为 **revision-3 / `rules_id=2e36dbea44893696`**（出空即撬底，
> T17，[`README.md`](./README.md) 顶部警告、[`t17-recalibration.md`](./t17-recalibration.md) §2）。
> 文中所有 revision-2 及更早的数字单独标注「旧口径（revision-2）」，禁止与 revision-3 混比。
> **资产状态**：旧 `base680k`/`w5_*` 等 checkpoint 已删除，相应历史数字只作统计性质参考。
>
> **尺度警告（先读）**：manifest 的 `levels` 是 probit-MLE 刻度（`mle.py` 的
> `s = √2·β = 141.42`，[`mle.py:16-18`](../../src/seven523/mle.py)、
> [`mle.py:852-861`](../../src/seven523/mle.py)；draw margin `eps` 的识别在
> [`mle.py:27-31`](../../src/seven523/mle.py)）；`duel`/`head_to_head` 报的 `elo_diff` 是
> 400-logistic 刻度（[`duel.py:72-74`](../../src/seven523/duel.py)）。在 p≈0.5 处
> `d(Elo_logistic)/d(Δ_probit) ≈ 1.96`（本文推算，附录 A.4）。**「10/15/20 Elo」在本仓库
> 既有功率表（`head-to-head-pilot.md` §4、`evaluation-protocol-validation.md` §5）里全部是
> logistic 刻度**；下文所有 Δ 均同此口径，跨表比较必须换算或用胜率口径。

## 0. 结论摘要

**问题**：单一 pinned 锚（RandomBot=0）的绝对 probit-MLE 表可以把候选粗分档（T17 四档
相邻间距 24.2/40.8/37.6 probit），但没有分辨 10–20 logistic Elo 小差异的能力：

1. **绝对口径的误差结构**：跨 fit/跨牌集 sd 高达 37（旧口径，`head-to-head-pilot.md` §1）；
   座位相位修复后单 fit 400 局/锚点的 Fisher SE 仍 ≈15–16（旧口径，
   `elo-reliability-audit.md` §6.1、`ladder-rerating-paired.md` §1.2）。单 fit 的
   绝对等级分不可跨 fit 比较（`evaluation-protocol-validation.md` §9）。
2. **配对口径才是分辨通道**：同牌换座 head-to-head 在 400 副牌下把差值 95% CI 半宽压到
   ±19.6 logistic Elo，比绝对口径窄 2.3–2.7×（`head-to-head-pilot.md` §3.3）；20 Elo
   约需 400–500 副（50% power）、**10 Elo 约需 1500–2000 副（50% power）**（§4）。
3. **本文对 T17 发布数据的只读复核**（revision-3，附录 B）显示：lvl1–lvl2 直接对局
   200 副的读数是 **−12.2 [−43.7,+20.9] logistic Elo**（胜率 0.4825，bootstrap rng=1），
   而表里它被排成 +24.2 probit（≈+47 logistic Elo）；这是全表最大的一对残差
   （z≈−2.0，tie-aware 口径，**in-sample 诊断值；门禁必须用 holdout/零分布，见 H4**）。
   评审实测 leave-pair-out 下同一对 z=−3.53，说明 in-sample |z| 会收缩真实失配——也就是
   说**表内相邻档的「分辨」当前未经直接证据确认**。

**推荐**：**两段式、单估计器、双通道**（§4）：

- **Stage 1（定级）**：保留现行 anchored probit-MLE 绝对表（`mle.py` + `refit_mle.py`，
  ADR-0013），RandomBot=0 是唯一 pinned gauge（ADR-0012）。绝对表只负责档位与先验中心，
  **不允许用单 fit 绝对 Elo 宣布 <20 Elo 的胜负**（`structural-directions.md` §9.7）。
- **Stage 2（分辨）**：在决策带内做**同牌换座、座位互换、按轮共享 Deal（CRN）的
  round-robin**，牌数按目标 Δ 配置；所有 Stage-1+Stage-2 对局进**同一个 `mle.py` 似然**，
  发布一个 `levels` 表 + 新增 `resolution` 块（逐对 Δ/胜率/CI/残差）。
- **确认（Stage 3）**：任何要写进结论的 ≥10 Elo 差异，用 **≥3 seed × 400 副（总 ≥1200）**
  独立 Deal bank 的直接 `head_to_head.py` 复核（EPV §9 冻结口径，bank 即 seed）；表内
  pooled 值只作筛选。只在估 `between_bank_sd` 时才允许「≥2 bank」，且行动判定仍 ≥3。

**预算速查（revision-3，完整 RR，m=4，σ₄₀₀=11.6（名义；敏感性 [11.4,11.7]），
80% 分辨率（CI-only），每对 N 副牌，总额 12N 局）**：

| 目标 Δ（logistic） | N（副/对，名义 11.6） | σ∈[11.4,11.7] 的 N | 总对局（12N） | 每候选（6N） | 打局墙钟（~58 局/s，T17 §6） |
|---|---|---|---|---|---|
| 20 Elo | 529 | 511–538 | 6,348 | 3,174 | ~1.8 min |
| 15 Elo | 939 | 907–956 | 11,268 | 5,634 | ~3.2 min |
| 10 Elo | **2,113** | **2,041–2,149** | **25,356（≈25k）** | **12,678** | **~7.3 min** |

- σ₄₀₀ 来自附录 A.3 的 revision-3 外推 11.583/11.451/11.673（名义 11.6；N∝σ²，公式见
  §4.2）。三个实测点估计给 Δ=10 的 N=2,107/2,059/2,139（**约 2,058–2,139，取 ~2,100**）；
  区间敏感性 [11.4,11.7] 给 2,041–2,149。**旧口径 10.14（乐观）/11.1（「保守」）与 404/718/1,614
  只作历史列**：它们把 N 低估 26–33%，N=1,614 的实际 80% 设计分辨率只有 0.68–0.71。
- **50% 分辨率口径**（SE=Δ/1.96）同步重算：Δ=20/15/10 → N=259/460/1,034（σ∈[11.4,11.7]：
  250–263/444–468/999–1,052）。
- **墙钟分列**：上表只含打局（~58 局/s 取自 T17 无并发环境；当前 w2m 训练并发时须实测或
  排期，§4.6 示例默认 `--device cpu`）。`refit_mle.py --bootstrap 4000` 是随后的 deal-clustered refit：默认
  `--bootstrap-workers 4`（fork 多进程、与串行逐位一致）下 4k 局语料 ≈2 min、Stage-2
  合并 ~2.9 万局 ≈15 min；`--bootstrap-workers 1` 串行 **≈40–55 min**（评审 M4 实测
  口径）。迭代默认 1000–2000，4000 只留最终发布。完整 §4.6 流程（打局 + refit + 确认段）
  约 25–45 min（4 进程），确认段另计。

**两个功效口径（预注册，禁止混用）**：

- **筛选分辨率（CI-only）**：`SE ≤ Δ/2.8016` 对应 80%「CI 排除 0」；本表与 §4.4 的
  `power_at_delta` 用此口径。
- **行动功效（复合规则）**：行动 = CI 排除 0 **且** 点估计 ≥+10；真值 Δ=10 时
  `P(点估计≥10)=0.50`，故上界 **50%，与 N/SE 无关**。80% 需 Δ≈13.0（理想 SE）–13.3
  （含 `max()` 保守化；本文推算附录 A.6），或对 +10 改单侧/等效（TOST）检验（属协议变更，
  预注册 + owner 决策）；`resolution.action_power_at_delta` 报此口径。

**确认预算（Stage-3，bank 即 seed）**：

- 冻结行动门槛：**≥3 seed × 400 副（总 ≥1200）**（EPV §9 / ADR-0013 决定 5）。该门槛在
  σ=11.6 下对 Δ=10 的实际分辨率 power≈0.32（CI-only，本文推算 A.1）——它是**行动判定的最低
  样本量**，不是 80% 分辨率。
- 若要在 Δ=10 上做到 80% 分辨率：3 bank 时每 bank 需 ~1,400 副（简单池化 power≈0.80；
  含 `combine_duel_seeds` 的 `max()` 保守化约 0.72），~1,700 副可让保守口径也到 0.80；
  或 4–5 bank × 1,100 副（实际 power≈0.82/0.89；保守口径 0.74/0.83）。公式与表见 §4.6。

**与既有禁令的关系（修订后）**：只 pin 一个 gauge（不冲突 ADR-0012 决定 1）；bridge/fiducial
以自由 `Prior` 进入、不是第二锚（ADR-0013 决定 7 允许）；发布仍只走 `mle.py`（不冲突
ADR-0013 决定 3），但 resolution 路径必须 `--keep-rungs` 冻结 rungs，否则与「档位契约不变」
冲突（H3，需 owner 决策 + 新测试）；`--cross` 只作为 MLE 的输入、不用 online PL 的绝对差
做结论；pooled 单 fit 不当最终证据、以 ≥3 bank 直接对局确认。§9.6/§9.7/§9.9 的处置统一为
**「需 owner 决策的新 T 项 / C-5 例外」**，不再写「已解/不冲突」（见 §4.8）。

---

## 1. 现状与证据

### 1.1 通道、接缝与职责（file:line）

| 通道 | 实现 | 用途 | 关键性质 |
|---|---|---|---|
| 在线评分 | `elo.fit_ratings`（[`elo.py:205`](../../src/seven523/elo.py)）、`expected_score`（:181）、`select_rungs`（:300） | 训练/匹配/placement 的在线更新 | OpenSkill PL、单遍重放、顺序相关；RandomBot=0 唯一 pinned（[`elo.py:57-63`](../../src/seven523/elo.py)、[`ladder.py:64`](../../src/seven523/ladder.py)） |
| 绝对发布表 | `mle.fit_mle`（[`mle.py:878`](../../src/seven523/mle.py)）、`MleConfig`（:413）、`tools/refit_mle.py` | 发布 `levels`/`sigmas` | order-free anchored probit MAP；齐性 `s=√2β`；prior σ 只进 MAP；deal 聚簇 bootstrap CI（`refit_mle.py` 的 `bootstrap_cis`） |
| 换座配对赛程 | `ladder.plan_games`（[`ladder.py:199`](../../src/seven523/ladder.py)） | 候选 vs 锚点、候选 vs 候选 | 每副牌双座位；同轮所有候选/所有对共享同一 deal；`--cross` 即候选 RR（:252-258） |
| 逐局记录 | `ladder.play_games`（[`ladder.py:527`](../../src/seven523/ladder.py)）；`--games-out` 由 `tools/build_ladder.py:108-115` 传入 | `games.jsonl` + trace | **追加**语义（串行 `open("a")` 在 `ladder.py:390`，并行 shard 以 `"ab"` 在 `ladder.py:343-352` 合并）；重跑前必须清空（M5） |
| 两候选裁决 | `duel.plan_duel_schedule`（[`duel.py:33`](../../src/seven523/duel.py)）、`paired_duel_stats`（:104）、`combine_duel_seeds`（:288）、`tools/head_to_head.py` | h2h Δ + CI | 牌聚簇 bootstrap；多 seed 合并 `max(bootstrap SE, seed sd)/√k`（:256-287） |
| 联盟/竞技场 | `arena.run_arena`（[`arena.py:102`](../../src/seven523/arena.py)）、`pair_diagnostics`（:241） | 多者联合拟合 + 可传递性**未校准**烟测 | **不得当门禁**：`se` 用 `games//2`、expected 用 PL 非平局口径（docstring [`arena.py:250`](../../src/seven523/arena.py) 自述 "diagnostic, not a calibrated test"）；resolution 门禁用 leave-one-bank-out holdout（H4，`refit_mle.build_transitivity`） |
| manifest 契约 | `study.merge_manifest`（[`study.py:145`](../../src/seven523/study.py)）默认冻结；冻结 rungs 合并 `study.py:67-84`/`:239-245` | `levels`/`subjects`/`rungs`/`resolution` | 只有 `refit=True` 才覆盖；非 refit 合并保留已发布 rungs 并重基 `mu`；`resolution` 块过 `RESOLUTION_REQUIRED_KEYS` 校验（`study.py:36-44`） |
| placement | `placement/estimator.py`：`fit_session`（:74）、`nearest_level`（:125）、`band_for`（:137） | 人/候选 vs 梯级分档 | 信息量最优选档原则见 `human-elo-10-games-research.md` §6.1–6.3 |

### 1.2 当前有效口径（revision-3 / T17）

`traces/study/manifest.json` 与 `runs/t17_mle/absolute_table.json`（`t17-recalibration.md` §2）：

| id | μ（probit） | Laplace σ | n | 95% CI（deal 聚簇 200 次 bootstrap） |
|---|---|---|---|---|
| lvl4 | 185.34 | 6.32 | 1600 | [173.65, 200.11] |
| lvl3 | 147.71 | 6.14 | 1600 | [136.24, 161.26] |
| lvl2 | 106.90 | 5.95 | 1600 | [96.69, 120.32] |
| lvl1 | 82.75 | 5.88 | 1600 | [70.46, 94.68] |
| random | 0（pin） | 0 | 1600 | [0, 0] |

- 4000 局 = 每候选 400 局 vs random（200 副 × 双座位）+ 6 对 × 400 局 cross（200 副/对）；
  总 deal 数 **400**（200 anchor + 200 cross；同一轮内所有候选/所有对共享同一 deal，本文复核，附录 B.1）。
- `rungs` 按 `select_rungs(count=5, min_spacing=100, max_spacing=150)` 重选后只剩 **lvl1/lvl4**
  （间距 102.60；实测相邻间距 24.2/40.8/37.6，低于最小间距，`t17-recalibration.md` §2）。
  **注意 `select_rungs.ok=False` 只报告、不阻止保存**（[`elo.py:336-341`](../../src/seven523/elo.py)），
  且 refit 历史路径会把全部非锚 `fit.ratings` 写进 `levels`；resolution 发布路径必须显式
  `--keep-rungs` 并把 n=0 的 prior-only id 挡在 `levels`/`rungs`/`subjects` 之外（新发布路径
  已实现该过滤；H3，§4.5/§4.7）。
- §3 校准（C(5,2)=10 对）：`lvl2>lvl1` 的 half-ties 胜率 0.5175，模型 tie-aware 预测 0.5674，
  差值 −0.0499（本文按二项 SE 折算 z≈−2.0）；另外 3 对 strict 口径越界、tie-aware 口径全部通过。

### 1.3 已有分辨率证据（注意口径）

**配对 h2h（旧口径/统计学性质可迁移）**

- 400 副牌直接对局的 95% CI 半宽 **18.8–20.9**（`head-to-head-pilot.md` §3，**旧口径**），
  bootstrap 隐含 σ₄₀₀≈10.0、seed 间池化 11.1（§3.2）；revision-3 用附录 A.3 的 11.4–11.7。
- 功率表（`head-to-head-pilot.md` §4；`evaluation-protocol-validation.md` §5 给出 50%/80% 两列；
  **旧口径历史表**，σ₄₀₀=10.14；revision-3 重算见 §0，以下数字不可直接当设计常数）：

| Δ（logistic） | 单对 50% power | 单对 80% power |
|---|---|---|
| 30 Elo | 171–211 | 359 |
| 20 Elo | 384–475 | **807** |
| 15 Elo | 683–845 | 1,434 |
| 10 Elo | 1,537–1,893 | **3,226** |

- 判定规则（EPV §9 / `README.md` §3）：≥3 seed × 400 副合并，CI 完全排除 0 且点估计 ≥+10
  才谈「值得行动」；≥20 的行动结论建议 5×400 或 3×800；<10 一律不追。注：EPV §9 的
  「80% power 需 ≥3200 副」是旧口径（σ=10.14）；revision-3 等价值为单对 ≈4,225 副
  （σ=11.6，50%/80% 单对表见附录 A.1），但**行动判定门槛本身不变**。

**绝对口径（旧口径/revision-2）**

- 座位相位修复前：换序位移 0–75 Elo、换序 sd≈31（`elo-reliability-audit.md` §3.3–3.5）；
  wave5 父模型跨 3 seed sd=37（`head-to-head-pilot.md` §1）。
- 修复后（候选内换座配对，[`ladder.py:239-258`](../../src/seven523/ladder.py)）：单 fit
  400 局/锚点联合 Hessian SE≈15.5–16.2（`elo-reliability-audit.md` §6.1）；5 seed 合并的
  lvl4 跨 seed sd 28.2、合并 SE 7.1（`ladder-rerating-paired.md` §2.2）。
- 结论：**绝对分只适合档位/展示；差值结论必须走配对通道**（audit §6.5、EPV §9、
  `structural-directions.md` §9.7）。

**本文对 revision-3 T17 数据的只读复核（新增，附录 B）**

| 对（a–b，直接对局 left=a） | 直接对局 Δelo（200 副，rng=1） | 表内 μ 差（b−a, probit） | 模型 P(b>a) | 观测 P(b>a) | 残差 z（in-sample） |
|---|---|---|---|---|---|
| lvl1–lvl2 | **−12.2 [−43.7,+20.9]** | +24.2 | 0.5674 | 0.5175 | **−2.0** |
| lvl2–lvl3 | −65.0 [−98.1,−33.1] | +40.8 | 0.6129 | 0.5925 | −0.8 |
| lvl3–lvl4 | −100.0 [−134.0,−68.6] | +37.6 | 0.6043 | 0.6400 | +1.5 |

（lvl1–lvl2 直接对局从 lvl1 视角的 half 胜率 = 0.4825，即 P(lvl2>lvl1)=0.5175；bootstrap CI
随 rng 轻微波动，点估计与 z 结论不受影响。**残差 z 是 in-sample 诊断，不得当门禁**：评审
leave-pair-out 复核把 lvl1–lvl2 的 z 从 −2.01 放大到 −3.53、lvl3–lvl4 从 +1.46 到 +2.64，
即 in-sample 口径会收缩真实失配（H4）。）

- 由三对的 per-deal 胜率 sd 做 delta 法外推，revision-3 的 σ₄₀₀=11.583/11.451/11.673（本文推算，
  附录 A.3），取名义 **11.6**、敏感性 **[11.4, 11.7]** 作设计常数。旧口径 10.14（乐观）/11.1
  （seed 间池化「保守」）低于本表全部三个点估计，**降为历史列，不可当 revision-3 设计常数**。
- 把 400 deal 随机对半拆开各自 refit，相邻差值的半差 |Δ₁−Δ₂| ≤ 8.0 probit、z≤0.9（附录 B.2）：
  在 200 副/对规模上，**对比量的抽样误差与 CI 一致**，问题不是 CI 公式错，而是**牌数不够 +
  模型局部残差**（失配判定仍须 holdout，见 §4.5）。

### 1.4 误差来源分解（决定方案该打哪一项）

| # | 来源 | 一手证据 | 对「差值」的影响 | 对「绝对档位」的影响 |
|---|---|---|---|---|
| i | 座位/相位 | audit §3.3–3.5（旧）；修复后消失（`ladder.py:239-258`、EPV §2.1） | 已消除 | 已消除 |
| ii | deal bank（牌集） | 旧跨 fit sd 31–37（audit §3.5、pilot §1）；座位平衡后单臂 sd≈14.6（audit §3.5） | 同牌 CRN 下在差值中约掉；跨 bank 需复核 | 主导项；单 fit 不能跨 fit 比 |
| iii | 远锚低信息 | T17 anchor 对 half 胜率 p=0.686/0.779/0.874/0.904 → probit 相对信息 91.8%/80.4%/61.2%/52.6%（`w(z)=φ(z)²/[p(1−p)]`，A.5；旧 `p(1−p)` 口径 86/69/44/35% 作废） | 每候选 400 局锚点局是低效预算 | 提供绝对位置，不可全砍 |
| iv | 模型局部残差/可传递性 | §1.3 表：lvl1–lvl2 z≈−2.0（in-sample；holdout 下 −3.53）；`arena.pair_diagnostics`（[`arena.py:241`](../../src/seven523/arena.py)）是**未校准**烟测（docstring `arena.py:250` 自述，`se` 用 `games//2`、observed 平局计 0.5，与 §1.3 的 z 不同源），**本设计不复用为门禁**；门禁用 leave-one-bank-out holdout 或参数 bootstrap 零分布（H4） | 任意「共享强度」的 pooled 差值都要做 holdout 残差检查 | 档位靠大间距，风险小 |
| v | 尺度混用 | `duel.py:74` logistic vs `mle.py` probit；倍率≈1.96（附录 A.4） | 把 24 probit 当 24 Elo 会低估约一倍 | 只影响解释，不影响排序 |

---

## 2. 问题定义：什么叫「分辨」

### 2.1 三类对象与估计量

- **R1 候选对候选（C2C）**：两个自由 id `a,b`（同 `rules_id`）。估计量 = 差值
  `Δ = 400·log10(p/(1−p))`（p = a 的期望胜率，平局计半），或等价的 probit 差。用于选型/排序。
- **R2 候选对梯级（C2R）**：候选 `a` 与已发布档位（random 或 lvlN）。估计量 = `P(a beats rung)`
  与最近档赋值（`placement.nearest_level`、`band_for`）。用于**定级不失守**与放置。
- **R3 表内档位对（T2T）**：已发布表相邻档的差值。估计量 = 表内 μ 差，**但必须能被直接对局核实**，
  否则只当模型平滑值。

### 2.2 目标 Δ、置信与功率口径（两个预注册口径，禁止混用）

- 置信：双侧 95% CI；正态口径 `power = Φ(Δ/SE − 1.96)`（EPV §5）。
- **口径 A：筛选分辨率（CI-only）**。50% power ⇔ CI 半宽 = Δ ⇔ `SE = Δ/1.96`；
  80% ⇔ `SE = Δ/2.8016`（EPV §5 的 ×≈2.04 规则）。§0 预算表与 §4.5 第 1 条的
  `power_at_delta` 用此口径。
- **口径 B：行动功效（复合规则）**。行动结论 = **CI 排除 0 且点估计 ≥ +10**。无偏正态下
  （本文推算，附录 A.6）`action_power(Δ) = 1 − Φ((max(1.96·SE, 10) − Δ)/SE)`；
  真值 Δ=10 时 `P(点估计≥10)=0.50`，故 **action_power ≤ 0.50，与 N/SE 无关**；
  80% 需要 `Δ ≥ 10 + 0.8416·SE` ≈ 13.0（理想 SE）–13.3（含 `max()` 保守化），
  或把对 +10 的检验改为单侧/等效（TOST）（预注册后启用，属协议变更走 owner 决策）。
- 目标 Δ ∈ {10, 15, 20} logistic Elo（与 +10/+20 行动阈值同尺度，EPV §9）。
- 判定仍不以「点估计 ≥ +20」为门槛（EPV §6 证明它在 Δ=20 时 power≈50%）：
  **≥20 的行动结论按 80% 分辨率配牌；+10 门槛的 80% 行动功效在 Δ=10 不可达（上界 50%）**。

### 2.3 定级功能不失守（硬约束）

方案必须同时满足：

1. **绝对表仍然是绝对表**：`levels` 来自单一 `mle.fit_mle` 拟合，RandomBot=0 是唯一 gauge
   （ADR-0012 决定 1，[`0012:12`](../adr/0012-single-gauge-and-greedy-removal.md)），发布走
   `tools/refit_mle.py --manifest-out --refit`（ADR-0013 决定 3，
   [`0013:27`](../adr/0013-drift-free-rating-channel.md)）；不新增 pinned 锚、不引入第二估计器。
2. **档位稳定**：Stage-2 加牌后，任一已发布梯级的 `nearest_level` 归类不变；Stage-1 CI 跨边界的
   候选允许改档但必须标 `provisional`（沿用 `band_for`，`placement/estimator.py:137`）。
3. **局部可分辨不靠绝对分**：任何 <20 Elo 的差值结论要有配对 CI + 独立 bank 复核；
   单 fit 绝对 `levels` 相减永远不作为证据。
4. **档位契约冻结（H3）**：resolution 发布路径**不得重选 `rungs`、不得改动档位契约**；
   实现依赖新 `--keep-rungs`（`estimator.rungs_source="kept"`）或独立的冻结发布路径，并在
   §4.5 验收 rungs 间距不跨 100、prior-only id 不进 `levels`/`rungs`。

### 2.4 尺度与单位（避免把 24 probit 读成 24 Elo）

- 近 p=0.5：`Δ_logistic ≈ 1.960 × Δ_probit`（本文推算，附录 A.4）。
- T17 表内 `lvl1→lvl2 = 24.16 probit ≈ 47.3 logistic`；`lvl2→lvl3 ≈ 80.0 logistic`。
- 但直接对局（200 副）对 lvl1–lvl2 给的是 −12 logistic，说明**表内 μ 差不能直接当 logistic 差**，
  反之亦然；比较统一走**胜率**或同刻度的配对 CI。

---

## 3. 候选方案（a–d 全覆盖）

### 3.1 方案 A：CRN/换座配对推广到全通道 + 多参考复用

**内容**：把 `ladder.plan_games` 的「每副牌双座位 + 同轮所有候选共享 deal」结构推广到**全部**
参照通道：同一 deal bank 同时服务 anchor 局与 cross 局；已发布档位作为**自由 fiducial**
（`Prior` 中心 + σ）跨实验复用；deal 种子持久化，跨 run/跨 seed 复用同一牌集。

**核实（一手代码）**：

- 候选 × 锚点：每 (轮, 锚点) 抽一个 deal，所有候选各打双座位（[`ladder.py:240-249`](../../src/seven523/ladder.py)）。
- 候选 × 候选：每轮抽一个 deal，**所有对**各打双座位（[`ladder.py:252-258`](../../src/seven523/ladder.py)）。
- T17 实际数据：400 个 deal seed，anchor 每副 8 局、cross 每副 12 局（附录 B.1）。
- 结论：**「同一副牌对所有候选复用」在 fit 内已经实现**，CRN 不是当前缺的东西；
  缺口是 (i) 同一 bank 不跨 run 持久化，(ii) anchor bank 与 cross bank 是两套牌
  （200+200），(iii) 参照档位每次重测。

**统计模型**：估计器不变（`mle.py`）。同 bank 只改变对比量的协方差结构；点估计不变。
**需要多少局**：本方向**不改变**某个 Δ 所需的牌数量级；预期对单个对比 CI 的影响 <10%
（只去掉 anchor/cross 两套牌的组成差）。它的收益是**可复现性与增量评估成本**（新候选只打
固定 slate，不用重测所有旧档），不是分辨率。

**集成点**：`--games-out` 已落逐局（[`ladder.py:527`](../../src/seven523/ladder.py) 的
`play_games`；`--games-out` 参数在 [`build_ladder.py:108-115`](../../tools/build_ladder.py)）；
`build_ladder`/`refit_mle --prior-manifest` 已读 manifest 先验（[`ladder.py:134`](../../src/seven523/ladder.py)、
`refit_mle.py` 的 `manifest_priors(document)`，见 §4.6）。多参考=自由 `Prior`，不违反 ADR-0012。

**ADR/§9 关系**：无冲突。**注意**：ADR-0012 决定 1 只 pin `random=0`；ADR-0013 决定 2
（[`0013:20`](../adr/0013-drift-free-rating-channel.md)）禁的是 **pinned 多锚 / soft L2 锚**；
「多锚点削方差」若指第二个钉死锚，则被禁止且也没有必要（gauge 的坐标不需要第二把尺子）；
合法形式是自由 fiducial。
**可证伪预期**：若声称 CRN 推广提高分辨率，则同牌数下对比 bootstrap SE 应下降 >10%；
否则本方向被证伪为「卫生措施」。**本文判断：它只能解释 ii/iii，不解决 iv。**

### 3.2 方案 B：两阶段定级 → 带内 CRN round-robin 重排（同 MLE + 先验）

**内容**：Stage-1 保持 T17 式绝对表（档位 + 先验中心）；Stage-2 在决策带内（例如相邻两档
之间的所有候选，或 top-k）做**完整 round-robin**：每对 N 副牌、每副双座位、每轮共享一个
deal（CRN），只 pin RandomBot=0（每候选少量 anchor 局即可，或不加，因为 Stage-1 的 anchor
局已进同一似然）；**所有 Stage-1+Stage-2 对局进同一个 `mle.fit_mle`**，先验 = 冻结 Stage-1 的
`levels` 中心 + 显式 `--prior-sigma 20`（§4.3），发布 `levels` + 逐对 `resolution`。

**统计模型**：over-dispersed 无——仍是 probit MAP；关键量是**差值方差**。在「m 个自由强度、
每对 N 副、正确设定、完整 RR」下（本文推导，附录 A.2）：

```
Var(Δ_ij) = 2·τ²/(m·N),   SE = τ·√(2/(mN)),   τ = 20·σ₄₀₀
m=4 时 N = 单对 h2h 所需牌数 ÷ 2（借力因子 m/2；仅当全表 6 对都要分辨时应据此配预算）
```

τ = 232（σ₄₀₀=11.6 名义；敏感性 228–234）。旧口径 τ=202.8（10.14）/222（11.1）只作历史。
**预算表见 §0**；每候选对局 = 2(m−1)N。对 m=4 的完整对表，RR 在总牌数上约为「每一对各自
独立 h2h 到同样分辨率」的一半（每对 N 减半）；**该「省一半」只对「全表 6 对都要分辨」成立**：
若只需一个目标对（m=4 表里只判 1 对），RR 总牌数 = 6N = 12,678 vs 单对 80% 的 4,225，是
**3 倍**——单目标对场景直接走 `head_to_head`，不要套 RR（L5，EPV §5 原表）。代价是依赖
可传递性。

**需要多少局（可证伪预期）**：

- m=4、σ₄₀₀=11.6、80% 分辨率：Δ=20 时 N=529（总 6,348 局）、Δ=15 时 939（11,268 局）、
  Δ=10 时 2,113（25,356 局）；敏感性 [11.4,11.7] 与 50% 口径见 §0/§4.2。若实测
  SE > 预测 ×1.3，说明可传递性/模型误差主导，本方案在该带内失效 → 退回逐对直接 h2h。
- 带宽按 A.5 的 probit 信息权重约束：带内任意对的相对信息 `w_rel ≥ 0.80`；低于时按
  `N_ij = ceil(N / w_rel)` 加牌（不是整带统一加）。旧文「相邻 ≤30 probit、跨度 ≤60 Elo」与
  T17 四档示例（跨度 102.6 probit）自相矛盾，撤销（L2/A.5）。

**集成点**：`build_ladder --cross` 即完整 RR（[`ladder.py:252-258`](../../src/seven523/ladder.py)），
`--games-out` 落盘后可进 `refit_mle.py`；`--prior-manifest` 读 Stage-1 先验。无需新估计器。

**ADR/§9 关系（修订为「需 owner 决策的新 T 项」，不再宣称已解）**：

- 不冲突 ADR-0012 决定 1 / ADR-0013 决定 2–3、7（单 gauge、单发布估计器、Prior 回流）；
  但发布必须 `--keep-rungs` 冻结 rungs（H3）。
- §9.6「`--cross>0` 做筛选」：**字面冲突**。提出解法（只用 **MLE** 读 cross，不读 online PL
  打印值；结论附 direct-duel 确认），但**属需 owner 决策的新 T 项 / C-5 例外**。
- §9.7「不要用绝对 Elo 单 fit 宣布 <20」：**部分冲突**。提出解法（pooled Δ 只作筛选，最终以
  ≥3 bank 的 `head_to_head` 合并 CI 为准），同样**需 owner 决策**。
- §9.9「不要再改协议」：新增 `resolution` 发布块 + `--keep-rungs` 属 C-5 边界外的通道改动，
  **按新 T 项走 owner 决策**；不改 `duel.py`/`combine_duel_seeds`/`elo.py` 的既有语义。

### 3.3 方案 C：参考梯级图 + 信息量/桥接分配

**内容**：把预算从「每候选打远锚」改成「在近 0.5 的梯级图上打」：维护一组自由 fiducial
（间距约 30–50 Elo，覆盖决策带），新候选只打 fiducial slate + 带内邻居；分配牌数按
每局 probit 信息 `w(z)=φ(z)²/[p(1−p)]` 加权（信息最优选档原则），并保留**最小**的 random 连接。

**统计模型**：同 `mle.py`；设计层做 D-最优/信息分配。**信息量数字**（本文推算，附录 A.5）：

- `mle.py` 是 **probit link**，每局对差值的 Fisher 信息用 `w(z)=φ(z)²/[p(1−p)]`（z 为
  probit 标准化差），不是 logit/比分口径的 `p(1−p)`。按此重算 T17 每候选 400 局远锚的相对
  信息：lvl1 91.8%、lvl2 80.4%、lvl3 61.2%、lvl4 52.6%（以 p=0.5 为 100%）；把这 400 局
  改打到近 0.5 的桥接梯级，单是这批牌的信息 **×1.09–×1.90**（原写 ×1.16–×2.9 是 logit/比分
  口径，作废，L1）。HR §6.2 的离线仿真显示 info 策略可达 oracle 的 94–98%、平均 p(1−p)
  0.234 vs 均匀 0.186，但 RMSE 只改善 2–4% ——**分配是二阶项，牌数与「不要全打远锚」是一阶项**。
- 设计层事实（本文推算，附录 A.2）：**完整 RR（m 个自由强度、每对等额 N、全对全）给任意
  两两对比相同的 `Var(Δ_ij)=2τ²/(mN)`，并在一切连通设计中最小化「最大对比方差」（minimax）**；
  环/路/星形对非相邻对的方差更大（星形中心到叶均匀但叶叶更差）。原「相邻对比方差之和相同 →
  RR 最稳健」的论证依赖未定义的边集、数值随编号变化，已撤销（L3）；信息加权的收益来自
  `w(z)`，不来自图形本身。**结论：均匀完整 RR 是 minimax 默认，不是省总牌数的手段。**

**集成点**：需要新的调度器（可变每对牌数）或先用 `plan_games` 的均匀 RR；fiducial 入
`subjects`/新 `fiducials` 字段，`prior-manifest` 读取。

**ADR/§9 关系**：fiducial 是自由 id、不是第二 gauge（ADR-0013 决定 7 的 Prior 回流）；不得把
fiducial 写进 `anchors`（`anchors` 只有 random，ADR-0012）。§9.6/§9.7 同 B 的处置。
**可证伪预期**：固定总牌数下把远锚预算改投近 0.5 对，相邻对比 SE 应下降 ≥1.2×；
若 <10%，说明信息模型/τ 假设错，收益证伪。

### 3.4 方案 D：发布侧把 twin 结构纳入 `mle.py`（似然/分层/CI）

**d1（推荐，低成本）**：保持 game-level 似然；**deal 聚簇 bootstrap 作为主 CI**（已实现，
`refit_mle.py` 的 `bootstrap_cis`/`bootstrap_levels` 按 seed 重采样并 refit）；在 `resolution`
块发布**逐对差值**及其 bootstrap CI（contrast 的 CI 必须来自同一 replicate 分布，见 §4.4）。
**「≥2 个独立 bank（seed）估 `between_bank_sd`」仅用于估该 SD**；行动判定仍按 EPV §9 的
**≥3 seed × 400**（H5）。多 bank 的最终 CI 取 `max(bootstrap SE, between_bank_sd)/√k × 1.96`
（沿用 `duel.combine_duel_seeds` 的口径，[`duel.py:256-287`](../../src/seven523/duel.py)），
且该路径的 `z=1.96` 与 `confidence` 不联动（T10 未修，[`duel.py:292`](../../src/seven523/duel.py)）
——**resolution CI 以 `confidence==0.95` 为前置条件**（L9/§4.7）。注意 `sigma` 受先验宽度影响
大：本文拟合中同一数据在窄先验（σ≈6）下 Laplace σ≈3.8–4.1，在 T17 默认先验下 σ≈5.9–6.3
（附录 B.2）——**发布不确定度应以 CI 为准，不要把 `sigma` 当 SE 跨先验比**。

**d2（暂不推荐，需先说清目标）**：把每对 twins 聚合成一个 deal 级观察（scores=两局之和）
再进 `mle.py`（同一 `_Pair` 抽象、只改 `s`/聚合规则），或加 per-deal 随机截距的分层 probit。
本文只读拟合显示：deal 级聚合会把 `lvl1→lvl2` 间距从 24.2 拉到 **31.0** probit、σ 从 3.8 拉到
4.5，位移 6.9 probit 与发布 CI（±8 量级）/半劈叉 SE（±8.9）同量级；这说明两种似然**不是同一模型的等价表述**，
聚合成 deal 级不是免费效率（近距离对局 43% 的 deal 净零，信息损失最大——恰好是要分辨的那一段）。
在纯 Python 约束下（`mle.py` 无 numpy/RNG/IO），分层 Laplace 的维护成本高、收益未被证明。
**可证伪预期**：若 `max(bootstrap SE, between_bank_sd)` 与独立 bank 的实测 sd 比 <1.3，
则 d2 无必要；若 >1.3 且方向与聚合似然一致，再立项。

### 3.5 方案对比总表

| 方案 | 谁/目标 | 统计模型 | Δ=10/15/20 所需牌（80% 分辨率，m=4） | 集成点 | ADR/§9 冲突 | 成本 | 可证伪预期 | 结论 |
|---|---|---|---|---|---|---|---|---|
| A CRN+多参考 | C2C/C2R 全部 | 不变（MLE） | 不改变量级 | `plan_games` 已做；需持久 bank/fiducial | 无（禁的是 pinned 多锚） | 小（持久化/slate） | 同牌数 CI 降 >10% 才成立 | 卫生项，单用不够 |
| **B 两阶段 RR** | C2C + 定级 | **同一 MLE**，先验=Stage-1 | **2,113 / 939 / 529 副/对**（σ=11.6；旧口径 1,614/718/404 废弃） | `build_ladder --cross` + `refit_mle --keep-rungs` | §9.6/§9.7/§9.9 按「需 owner 决策的新 T 项」处置 | 中（发布块/调度 wrapper） | SE≈τ√(2/(mN))，偏差 >30% 即证伪 | **推荐主体** |
| C 梯级图+信息分配 | C2R/C2C 定位 | 同一 MLE + D-最优分配 | 同 B；远锚局信息 ×1.09–1.90（probit） | 新 scheduler + `fiducials` | 同上；fiducial 必须自由 | 中 | 固定牌数 SE 降 ≥1.2× 才成立 | 吸收为分配原则 |
| D d1 发布侧 CI | 发布 | game-level + deal bootstrap + bank 方差 | — | `refit_mle.py`、`study.py` schema | 无 | 小 | CI 与独立 bank 比 <1.3 | **推荐发布口径** |
| D d2 似然聚合/分层 | 发布 | deal 级 ordered probit / 随机截距 | 无免费收益证据 | `mle.py` | 无（但违背纯 Python 简洁性） | 大 | 点估计位移须 <CI | 暂缓 |

---

## 4. 推荐方案（B 主体 + C 的分配原则 + D-d1 的发布口径）

### 4.1 协议总览

```
Stage 0（基线，一次/rules_id）
  ├─ gauge：RandomBot=0 唯一 pinned（ADR-0012）
  └─ 固定参考 slate：已发布 lvl1..lvl4 作为自由 fiducial（prior 中心=levels，
     σ=显式 --prior-sigma 20；发布 σ≈6 不作先验宽度，§4.3）
Stage 1（定级）
  ├─ 赛程：每候选 vs random ≥200 副（双座位；沿用 build_ladder）
  ├─ 估计：mle.fit_mle（all games）→ levels + CI（deal 聚簇 bootstrap）
  └─ 产物：档位 labels（nearest_level）、带（band）候选集合、先验中心
Stage 2（分辨）
  ├─ 带选择：m ≤ 6；带宽按 A.5 的 probit 相对信息 w_rel ≥ 0.80（|Δμ| ≲ 110 probit）；
  │   若只选 top-2，直接走 head_to_head 单对（不必 RR）
  ├─ 赛程：完整 RR，每对 N 副（查 §0 表），每副双座位，每轮所有对共享同一 deal；
  │   Stage-1 已有对局全部并入同一似然（不重复打；数据二次使用显式登记，§4.3）
  ├─ 估计：同一个 fit_mle；prior=冻结 Stage-1 levels；显式 --prior-sigma 20（公式见 §4.3）
  └─ 发布：levels/subjects（单一绝对表）+ resolution 块；--keep-rungs 冻结 rungs（H3）
Stage 3（确认，写结论前）
  └─ 对要行动的相邻对：≥3 seed × 400 副的 head_to_head 合并 CI（EPV §9 冻结规则）；
     ≥2 bank 只用于估 between_bank_sd，不构成行动判定
```

**适用边界（L10）**：Stage-2 的 CRN/双座位/镜像语义在**确定性 argmax**下成立（当前 ckpt：
[`networks.py:758`](../../src/seven523/networks.py) 默认 `sample=False`）；采样策略必须先做
entrant-id 派生 seed + EPV §8 镜像校准（§5.5），否则 twin 不再“同一样本同一流”。

**判定规则**（修订后与 EPV §9 / `README.md` §3 一致）：

1. Stage-2 `resolution.contrasts` 给出 pooled Δ 与 CI，用于**排序/筛选**；
2. 行动结论要求：**≥3 seed × 400 副**独立 bank 的直接 h2h 合并 CI 排除 0 且点估计 ≥+10；
   ≥20 的结论按 EPV 功率表配牌；Δ=10 的 80% 只属分辨率口径（行动功效上界 50%，§2.2）；
3. holdout 残差超阈值的对**只信直接对局**；in-sample 残差（如 lvl1–lvl2 z≈−2.0，
   holdout 下 −3.53）只作诊断，阈值在预注册前不启用（H4，§4.5 第 4 条）。

### 4.2 schedule 生成（具体）

- **带成员**：`m` = 决策带内候选 + 必要时 2 个夹逼 fiducial；要求 m ≤ 6；带宽按
  **probit 相对信息 `w_rel(z) = w(z)/w(0) ≥ 0.80`**（A.5，对应 |Δμ| ≲ 110 probit；
  T17 示例的最大跨度 lvl1–lvl4 = 102.6 probit，w_rel≈0.82，合规）。旧「相邻 ≤30 probit、
  跨度 ≤60 Elo」与 T17 四档示例自相矛盾，撤销（L2）。
- **RR 结构**：完整 RR（每对都打）。理由：完整 RR 给任意两两对比相同的方差，并在连通设计里
  最小化最大对比方差（minimax；附录 A.2）——不是省总牌数的手段。
- **每对牌数**：`N = ceil( 2·τ²·(2.8016/Δ)² / m )`（80% 分辨率）与
  `N50 = ceil( 2·τ²·(1.96/Δ)² / m )`（50%）；`τ=20σ₄₀₀`，σ₄₀₀=11.6（[11.4,11.7] 敏感性）。
  带入 m=4：80% `N=15.698·σ²/Δ²` → Δ=20/15/10 为 529/939/2,113；
  50% `N50=7.683·σ²/Δ²` → 259/460/1,034。若某对 `w_rel<1`，该对加牌
  `N_ij = ceil(N / w_rel(z_ij))`（L1/L2）。
- **CRN/座位**：每轮抽 1 个 deal seed，所有对都用它；每对在该 deal 上双座位。
  `plan_games` 已经保证（[`ladder.py:252-258`](../../src/seven523/ladder.py)）；`--cross 2N`。
- **锚点连接**：Stage-1 的 anchor 局（200 副/候选）保留；Stage-2 可加 `--games-per-anchor 100`
  给新候选补一条到 random 的连接（老档可选 0，但 `build_ladder` 要求 ≥1，取最小偶数 2）。
- **信息加权（可选，二阶）**：若带宽 >40 logistic，把远离 0.5 的对的牌数按 p(1−p) 下调、
  近 0.5 的对上调（先验来自 Stage-1）；先以均匀 RR 落地，用实测 SE 决定是否加。

### 4.3 估计器与先验

- 唯一估计器：`mle.fit_mle`（不改数学）；先验：冻结 manifest 的自由 id →
  `Prior(mu=level, σ=σ_prior)`，**σ_prior 必须显式传 `--prior-sigma`（记 `estimator.prior_sigma`），
  不得沉默沿用发布 σ**。
- **σ_prior 选择依据（本文推算，附录 A.6）**：对比量的先验中心敏感度
  `w_prior = V/(V+2σ_prior²)`，其中 `V=2τ²/(mN)=SE²`（A.2）。预注册目标：在设计 N 上
  `w_prior ≤ 10%`。Δ=10、N=2,113、τ=232 → V=12.74 probit²，要求 σ_prior ≥ √(4.5V) ≈ 7.6；
  推荐 **σ_prior=20**（w_prior≈1.6%；上限 50 → 0.25%）。发布值 σ≈5.9–6.3 是默认
  `Prior(500,200)` 下的 Laplace 后验 sd，**不是 prior 宽度**；沿用 σ≈6 时 w_prior≈15%，
  超目标且会锁死局部差异——示例禁止再出现。
- **Stage-1 局的数据二次使用（显式决定）**：Stage-1 局既进联合似然，其 `levels` 又作 prior 中心
  （empirical Bayes）。决定：**保留联合似然 + 冻结 manifest 中心**（不另造无数据先验），但
  (i) `estimator` 记 `prior_double_use:"stage1-joint"`；(ii) `achieved.deals/power` **只计
  Stage-2 局**，不把 Stage-1 局算进分辨率；(iii) 结论落在 CI 边界 10% 内时，必须补一次
  排除 Stage-1 局的敏感性拟合；(iv) 行动结论永远走 Stage-3 独立 bank（H5）。
- `random=0` 保持 pinned；fiducial 一律自由（`prior` 中心=发布值，不是 `anchors`）。
- CI：deal 聚簇 refit bootstrap（`refit_mle.py` 的 `bootstrap_cis`/`bootstrap_levels`），
  contrast 用同一 replicate 分布；迭代默认 1000–2000，4000 只留最终发布（默认 `--bootstrap-workers 4`
  下 4k 局 ≈2 min / Stage-2 语料 ≈15 min；串行 ≈40–55 min，M4）；多 bank 时用 `max(bootstrap SE, between_bank_sd)/√k`（confidence 固定 0.95）。
- **T9–T11 对账（L9）**：T9（[`plans.md:106`](../plans.md) 配对牌局 cluster bootstrap）——
  `fit_ratings` 仍无 cluster 选项，但发布侧已由 `refit_mle.bootstrap_cis` 吸收；
  T10（[`plans.md:118`](../plans.md) `z`/confidence 联动）**未修**，`combine_duel_seeds`
  的 `z=1.96` 与 `confidence` 不联动（[`duel.py:292`](../../src/seven523/duel.py)），
  故 resolution CI 以 `confidence==0.95` 为前置条件；T11（[`plans.md:129`](../plans.md)）
  已随 ADR-0011 关闭。
- 不做：不改 `elo.py`；不改 `duel.py` 统计；不在 `mle.py` 里加 RNG/numpy；不加第二 pinned 锚。

### 4.4 发布字段（manifest 增量，由 `study.py` 做 schema owner）

在现有 `levels`/`subjects`/`anchors`/`rungs`/`estimator` 之外新增（所有数值严格 JSON、
非有限量写 `null` 并带 `converged`/识别标记，沿用 ADR-0013 的发布完整性规则）。`study.py`
做 schema owner：必须键 = `RESOLUTION_REQUIRED_KEYS`（`rules_id`/`band`/`target`/`achieved`/
`contrasts`/`transitivity`/`ci_source`），未知键保留（[`study.py:36-44`](../../src/seven523/study.py)）：

```jsonc
"resolution": {
  "rules_id": "2e36dbea44893696",
  "design": "seat-twin CRN complete round-robin",
  "band": ["lvl1","lvl2","lvl3","lvl4"],
  "tie_credit": 0.5,
  // 两个口径分开预注册：power_at_delta=CI-only（SE≤Δ/2.8016）；
  // action_power_at_delta=复合规则（CI 排除 0 且点≥+10，Δ=10 上界 0.50）
  "target": {"delta_elo_logistic": 10, "power": 0.8, "alpha": 0.05},
  "achieved": {"lvl1-lvl2": {"se_elo": 3.57, "deals": 2113,
      "power_at_delta": 0.80, "action_power_at_delta": 0.50}},
  "contrasts": [
    {"a": "lvl1", "b": "lvl2",
     "delta_mu_probit": 24.2, "win_prob": 0.567,
     "delta_elo_logistic": 47.3,
     "ci95_win_prob": [0.5577, 0.5776],  // 与 se_elo 同一 bootstrap replicate 分布
     "deals": 2113}
  ],
  "transitivity": {"method": "leave-one-bank-out observed-vs-model winrate, deal clustered",
                   "banks": ["stage1","stage2"], "status": "computed",
                   "threshold": 2.5, "max_abs_z": 2.0, "flagged": ["lvl1-lvl2"]},
  "ci_source": "deal-bootstrap(seed=0,n=4000)",
  "action_rule": "CI95 excludes 0 and the point estimate is >= +10 Elo",
  "action_power_note": "compound-rule power; capped at 0.5 at a true +10 Elo effect"
}
```

单 bank 时 `transitivity.status="not-computed"`（`z`/`max_abs_z` 为 `null`）：in-sample z 不发布
为门禁。同时 `estimator` 增加 `seeds`、`deals`、`banks`、`ci_source`、`prior_sigma`、
`rungs_source`、`rungs_selection_ok`、`rungs`、`prior_double_use`；`subjects` 每个 id 的 `sigma`
仍是 Laplace（仅供诊断）；`rungs` 保持现有稀疏契约，**resolution 路径不因 resolution 改选**
（H3；`--keep-rungs` 时 `rungs_source="kept"`）。`between_bank_sd` 未实测前写 `null`，不预填
示例值。

### 4.5 验收标准（power/CI，可执行）

1. **分辨率达标（口径 A）**：每个目标相邻对的 `SE_bootstrap ≤ Δ/2.8016`（80% CI-only），
   报告 `achieved.power_at_delta`；不达标就按 §0 表加牌，不许用点估计硬凑。
2. **行动功效报告（口径 B）**：每个 contrast 同时报 `action_power_at_delta`（复合规则；
   含 Δ=10 上界 0.50）；**不得把口径 A 的 0.8 标成行动功效**。
3. **直接确认（H5）**：对每个要行动的对，**≥3 seed × 400 副**的 `head_to_head.py` 合并
   CI 按 EPV §9 判定（CI 排除 0 且点 ≥+10）。「≥2 bank」只允许用于估 `between_bank_sd`，
   且必须与行动判定分开标注；若要放宽到 2 bank 属 owner 决策并附 FPR/power 分析。
4. **残差门禁（H4）**：门禁只能用 **leave-one-bank-out holdout**（A bank 拟合、B bank 算
   observed-vs-model；`refit_mle.build_transitivity`）或 **parametric bootstrap 零分布**；
   in-sample 残差（§1.3 表）只作诊断。不得原样复用 `arena.pair_diagnostics`（未校准；`se` 用
   `games//2`、observed 平局计 0.5，z 口径不一致）。`|z|>2.5` 在完成零分布预注册前只记录、
   不淘汰；单 bank 时报 `transitivity.status="not-computed"`。
5. **覆盖校准（M6，可证伪）**：(a) 已知真值/零假设仿真：用固定真值（如 Δ=0 的合成数据或
   EPV §3.2 式独立随机零）重复 k 次发布，检查 nominal CI 覆盖率并给二项 CI；(b) 若要比较
   「发布 CI 覆盖独立 bank refit」，需 ≥29 个全成功 bank 才能给 90% 的 95% 单侧下界
   （0.05^(1/29)=0.902；n=5 时 5/5 覆盖下界仅 0.549），且比较量须含 bank 自身噪声
   （√(发布 SE²+bank SE²)）。方法在跑数前预注册。
6. **rungs/档位契约（H3）**：发布 `resolution` 必须 `--keep-rungs`（或冻结发布路径），
   `estimator.rungs_source="kept"`；验收检查 (i) rungs 间距不得跨 100（不得塌缩/新增越界档）；
   (ii) prior-only（n=0）id 不得进入 `levels`/`rungs`/`subjects`；(iii) `rungs_selection_ok`
   与任何重选/塌缩均须显式记录，不得静默发布。
7. **档位稳定**：Stage-2 后每个发布梯级的 `nearest_level` 不变；变化需标 `provisional` 并在
   `estimator` 记 `grade_shift`。
8. **可证伪条件**：实测 `SE > 1.3 × τ√(2/(mN))`（τ 按 σ=11.6），或某对 **holdout** 残差
   超预注册阈值且直接对局重复确认，则放弃该带的 pooled 排序，回到逐对 direct duel。

### 4.6 复现命令草案（可执行草案，未跑；接口按同批实施的新能力）

Stage-1 复用现有 T17 产物（`runs/t17_mle/games.jsonl`，4000 局）。Stage-2 以带内 4 档、
Δ=10、80% 分辨率为例（N=2,113 → `--cross 4226`）。命令假设无并发训练；本机 w2m 波在跑时
按 F6 排期或降低 workers（示例用 `--device cpu`）。

```bash
# Stage 2：带内 RR（每对 2113 副 × 双座位 = 4226 局；4 档互打；打局总量 25,356 局）
# M5：--games-out 是追加语义（ladder.py:390 open("a")；并行 shard 以 "ab" 在
# ladder.py:343-352 合并），重跑前必须清空或换 run-id 目录，否则重复牌局被静默双计。
rm -f runs/resolve/stage2.jsonl
.venv/bin/python tools/build_ladder.py \
  --candidate lvl1=ckpt:runs/t17early2k__1__1790439615/agent.pt \
  --candidate lvl2=ckpt:runs/t17early8k__1__1790439629/agent.pt \
  --candidate lvl3=ckpt:runs/t17pool__1__1790439615/snapshots/checkpoint_step65536.pt \
  --candidate lvl4=ckpt:runs/t17long__1__1790439615/snapshots/checkpoint_step1024000.pt \
  --anchor random=random --games-per-anchor 100 --cross 4226 \
  --no-traces --games-out runs/resolve/stage2.jsonl \
  --study traces/study --workers 8 --device cpu --seed 400   # M3：fresh bank 400

# 联合 MLE：Stage-1 + Stage-2 同一似然（--games 可重复，M1）；先验=冻结 manifest + 显式
# σ_prior=20（M2）；--keep-rungs 冻结 rungs（H3）；输出 resolution artifact。
# M4：--bootstrap 4000 默认 4 进程（fork，结果逐位一致）≈15 min（Stage-2 语料）；
# 串行 ≈40–55 min；迭代先用 1000–2000。
.venv/bin/python tools/refit_mle.py \
  --games runs/t17_mle/games.jsonl --games runs/resolve/stage2.jsonl \
  --anchors random=0 --prior-manifest traces/study/manifest.json \
  --prior-sigma 20 --keep-rungs --band lvl1,lvl2,lvl3,lvl4 \
  --bootstrap 4000 --seed 0 --json --out runs/resolve/table.json

# 直接确认（对每个行动对；冻结口径 ≥3 bank × 400 副；bank seeds 400,401,402 见 M3）
.venv/bin/python tools/head_to_head.py \
  --left lvl1=ckpt:runs/t17early2k__1__1790439615/agent.pt \
  --right lvl2=ckpt:runs/t17early8k__1__1790439629/agent.pt \
  --seeds 400,401,402 --pairs 400 --device cpu --workers 1 --bootstrap 4000 --json
```

**Seed 台账（M3）**：bank 取 **400–402**。审计口径（2026-09-27 复算）：`runs/**/*.json` 的
`seeds`/`source_seeds` 并集 = {0–4, 10–18, 100–148, 150–158, 160–168, 220–222, 229–231,
300–308}，不含 400–402；保留段见 `selfplay-pool-plan.md` §3.6（0–8/10–18/29–37 已占用）与
`reward-alignment-plan.md` §4.4–4.5（不得与 0–8/10–18/29–40/100–148 重叠；建议 200–204/
220–228）；`runs/archive/legacy-20260926/experiments/protoeval/zero_null/` 存在旧口径
`games_seed20–22/30–37.jsonl`，故 20–22/30–32 即使在当前口径无数据也不得复用。
`resolution.banks.source_seeds` 与 `--seed`/`--seeds` 必须一致。

**Stage-3 确认预算（Δ=10，80% 分辨率目标；σ=11.6；冻结行动门槛本身仍是 ≥3×400）**：

| bank 配置 | 简单池化 SE | power@10 | 含 `max()` 保守化（半宽 ×1.1） | power@10 |
|---|---|---|---|---|
| 3 × 1,410 | 3.57 | 0.80 | 3.93 | 0.72 |
| 3 × 1,705 | 3.24 | 0.87 | 3.57 | 0.80 |
| 4 × 1,100 | 3.50 | 0.82 | 3.85 | 0.74 |
| 5 × 1,100 | 3.13 | 0.89 | 3.44 | 0.83 |

（公式 `SE_pool = σ·√(400/N)/√k`（本文推算，A.1/A.6）；`combine_duel_seeds` 的 `max()`
保守化按 EPV:168 的 k=3 +10% 半宽近似。Δ=10 的行动功效仍受 §2.2 的 50% 上界约束。若只要
冻结行动门槛，用上面的 3×400 命令（σ=11.6 下对 Δ=10 的分辨率 power≈0.32）；上表只在 owner
决定追求 Δ=10 的 80% 分辨率时启用。）

（`--manifest-out --refit --keep-rungs` 只在 owner 批准后追加；默认不覆盖 `traces/study`。）

### 4.7 预计改动文件（不要求现在写代码）

| 文件 | 改动 | 规模 |
|---|---|---|
| `tools/refit_mle.py` | `bootstrap_cis`/`bootstrap_levels` 记录每 replicate 的 levels → 逐对 contrast CI（同源）；构建 `resolution` 块（`power_at_delta` + `action_power_at_delta`，H2）；**`--prior-sigma`（必做前置，M1/M2）**；**`--keep-rungs`（必做前置，冻结 rungs，H3；记 `estimator.rungs_source`/`rungs_selection_ok`/`rungs`）**；`--band`；`estimator` 增 `seeds/deals/banks/ci_source/prior_sigma/prior_double_use`；过滤 n=0 prior-only id | ~200–300 行 + 测试（`tests/test_refit_mle.py`） |
| `src/seven523/study.py` | manifest schema 允许并保存 `resolution`（`RESOLUTION_REQUIRED_KEYS` 校验、默认冻结语义不变）；冻结 rungs 合并已有（[`study.py:67-84`](../../src/seven523/study.py)、[`:239-245`](../../src/seven523/study.py)）；`levels`+`resolution` 原子发布 | ~30 行 + 测试 |
| `tools/build_ladder.py` 或新 `tools/resolve_band.py` | 带内 RR 包装 + 牌数/信息加权计算（可先直接用 `--cross`）；`--games-out` 落盘前清空（M5） | ~80–150 行 |
| `tools/refit_mle.py` 的 `build_transitivity`（新） | leave-one-bank-out observed-vs-model + parametric bootstrap 零分布（H4）；**不复用** `arena.pair_diagnostics`；单 bank → `not-computed` | 已随 refit 改造（~60–120 行 + 测试） |
| `docs/experiments/README.md` | 索引新增本文与评审（本次不改，留待 owner） | — |
| `elo.py` / `mle.py` / `duel.py` | **不改**（ADR-0013 纯 Python 与 duel 统计口径保持冻结；T10 的 z/confidence 联动遗留另行立项） | 0 |

### 4.8 与 ADR/§9 的关系（逐条）

| 约束 | 是否冲突 | 解法 |
|---|---|---|
| ADR-0012 决定 1「RandomBot 唯一 gauge」 | 否 | 只 pin `random=0`；bridge/fiducial 是自由 `Prior`（σ>0），manifest 里单独标角色，不写 `anchors` |
| ADR-0013 决定 2「单 gauge，禁多锚 / soft L2 锚」 | 否 | fiducial 走 Prior 回流（ADR-0013 决定 7）；不引入第二 pinned 锚 |
| ADR-0013 决定 3「发布表只由 `mle.py`；refit 重选 rungs」 | **需新 T 项（H3）** | Stage-2 用同一 `fit_mle`，但 `levels` 更新时必须 `--keep-rungs` 冻结 `rungs`（已实现）；否则与「档位契约不变」冲突。重选/塌缩须显式旗标 + 测试，owner 签字 |
| `structural-directions.md` §9.6「`--cross>0` 做筛选」 | **字面冲突，需 owner 决策** | 解法：只把 cross 当 **MLE 的输入**、不用 online PL 打印值，结论附直接对局确认；属 **C-5 例外 + 新 T 项**，非「已解」 |
| `structural-directions.md` §9.7「单 fit 绝对 Elo 不宣布 <20」 | **部分冲突，需 owner 决策** | 解法：pooled 只作筛选，最终证据是 ≥3 bank direct duel；均属新 T 项 |
| `structural-directions.md` §9.9「不要再改协议」 | **C-5 例外 + 新 T 项** | `resolution` 块、`--keep-rungs`、`--prior-sigma` 属新增通道；不改 `duel.py`/`elo.py`/`mle.py` 既有语义；owner 决策后实施 |
| `README.md` §3 / EPV §9 判定规则 | **修订后一致** | ≥3 seed × 400 副合并、CI 排除 0 且点 ≥+10；≥20 的行动结论按 80% 分辨率配牌；<10 不追。本 v2 删除了「≥2 bank」旧表述（H5）；Δ=10 的 80% 只作分辨率口径（行动功效上界 50%） |

---

## 5. 风险与未决问题

1. **可传递性（H4）**：RR 借力依赖 1-D 强度模型；T17 的 lvl1–lvl2 就是残差先例
   （in-sample z≈−2.0，leave-pair-out 下 −3.53）。协议用 **holdout/boot-null 门禁 +
   direct duel 兜底**；in-sample 残差只作诊断。**带内多残差**时 pooled 排序只能作参考。
2. **尺度**：`levels`（probit）与 h2h Elo（logistic）并存；文档化换算（≈1.96× @p≈0.5），
   但仍易误用。`resolution` 同时给胜率与 logistic Elo（本文草案已含），但不改 `levels` 语义。
3. **bank 方差未在 revision-3 实测（阻塞项）**：T17 只有一个 seed 的 4000 局；σ₄₀₀ 是
   T17 三对的 delta 法外推（11.4–11.7）。`between_bank_sd` 与覆盖率验收都依赖它，
   **落地前必须先补测 ≥2 bank（仅估 SD）/≥3 bank（行动判定）**。σ₄₀₀ 若坚持用 10.14/11.1，
   须先补 revision-3 多 bank 的直接测量。
4. **先验宽度（M2）**：发布 σ≈6 会锁死局部差异（w_prior≈15%）；`--prior-sigma` 是必做前置，
   示例显式 20（§4.3），目标 w_prior ≤10%；数据二次使用按 §4.3 显式登记并影响敏感性检查。
5. **随机策略的 seed 退化（L10）**：`policy_seed(game.seed, seat)` 按座位槽派生
   （EPV §3.2/§8）。当前 ckpt 是确定性 argmax（[`networks.py:758`](../../src/seven523/networks.py)
   默认 `sample=False`，`policy_from_spec` 不传 sample），Stage-2 CRN 结论成立；**引入
   sampling 策略时，entrant-id 派生 seed + EPV §8 镜像校准是硬前置**，否则 twin/CRN 语义失效。
6. **多重比较**：带内对所有相邻对做 95% CI，m=6 时 5 个相邻对；建议预注册目标对或
   Bonferroni/分层（EPV §8 已有该风险条目）；holdout 门禁阈值同样需 multiplicity 控制。
7. **预算与收益边界**：Δ=10 需 ~25k 局/带（打局 ~7.3 min；refit bootstrap 4000 额外
   默认 4 进程 ≈15 min、串行 ≈40–55 min 单核），对大池（几十候选）不可行；协议限定带内 m≤6。池很大时先粗筛
   （每个候选对 slate 的 50–100 副）再选带。**打局墙钟与 refit 墙钟必须分开列**。
8. **是否更新 `levels`（H3）**：推荐把 Stage-2 局并入同一 MLE 并显式 refit，但 resolution
   路径必须 `--keep-rungs` 冻结 `rungs`（或走独立冻结发布路径）；备选是冻结 `levels`、
   只在 `resolution` 发布局部重排。两者都需 owner 决策（涉及 `traces/study` 契约与 D1 标签）。
9. **中断重跑双计（M5）**：`--games-out` 追加语义（[`ladder.py:390`](../../src/seven523/ladder.py)
   `open("a")`；并行 shard 以 `"ab"` 在 `finally` 合并，[`ladder.py:343-352`](../../src/seven523/ladder.py)）+
   `load_games` 不去重（`refit_mle.py` 的 `load_games`）。失败重跑会在同一 cluster 内翻倍
   权重、静默拉偏点估计。命令必须 `rm -f` 或换 run-id 目录。
10. **覆盖校准口径（M6）**：见 §4.5 第 5 条；`≥5 bank 覆盖率 ≥90%` 不可证伪（5/5 的单侧 95%
    下界仅 0.549），不得再写入验收。
11. **rungs 塌缩门禁（H3）**：`select_rungs.ok=False` 不阻断保存
    （[`elo.py:336-341`](../../src/seven523/elo.py)）；当前 lvl1/lvl4 间距 102.60 只比
    `min_spacing=100` 高 2.6 probit，Stage-2 的牌数正会把 levels 移到这一带。resolution
    发布必须 `--keep-rungs` 或断言间距不跨 100，且 prior-only id 不入 `levels`/`rungs`。
12. **设备/并发假设（F6）**：§4.6 的打局墙钟来自 T17 无并发环境；`--workers>1` + cuda 时
    每个 spawn worker 各开 CUDA context（[`ladder.py:448-463`](../../src/seven523/ladder.py)
    自带 OOM 警告）。示例默认 `--device cpu`；在 w2m 训练波并发时须实测或排期。

---

## 附录 A：推导与设计常数

### A.1 单对 h2h 的功率

`σ₄₀₀` := 400 副牌 duel 的 `elo_diff` 隐含 SE（logistic 刻度）。则
`SE(N) = σ₄₀₀·√(400/N)`（EPV §5 的 √N 律，log-log 斜率 −0.483）。
`power = Φ(Δ/SE − 1.96)`，故

- 50% 分辨率（CI-only）：`N = σ₄₀₀²·400·(1.96/Δ)²`；
- 80% 分辨率（CI-only）：`N = σ₄₀₀²·400·(2.8016/Δ)²`。

旧口径 σ₄₀₀=10.14 得 20/15/10 Elo → 395/702/1580（50%）、807/1435/3228（80%），与 EPV §5
一致（**仅历史**）。revision-3 用 σ₄₀₀=11.6：**单对（m=2）** 50% → 517/919/2,068；
80% → 1,056/1,878/4,225；§0 的 259/460/1,034 与 529/939/2,113 是 **m=4 RR 的每对牌数**
（借力 m/2）。敏感性 σ∈[11.4,11.7] 见 §0。

### A.2 完整 RR 的借力（minimax 论证）

设计 = m 个自由强度（顶点）上的连通图，边 = 被打的对；每条边等额 N 副、每副双座位。
Fisher 信息矩阵（等权、忽略锚点）为 `(N/τ²)(m·I − J)`，对任意边 `e_i−e_j`：

```
Var(Δ_ij) = 2τ²/(mN)，  SE = τ√(2/(mN))，  τ = 20·σ₄₀₀
```

- m=2 退化为单对 `τ²/N`（自洽）；m=4 的完整图对**任意**两两对比都给上式。
- **minimax**：在所有连通图设计中，完整图（全对全）把「最大两两对比方差」压到最小
  `2τ²/(mN)`；环/路/星形对非相邻对的 Var 更大（星形中心→叶可均匀，但叶叶对没有直接边、
  方差不可降到完整图水平）。
- 每对等额 ⇒ m=4 的完整 RR 总牌数 = 单对 N 的 6 倍；「N = 单对 N÷2」只描述**每对**牌数
  减半，前提是 6 对都要分辨（§3.2）。单目标对不要套 RR（L5）。
- 旧稿「相邻对比方差之和相同 → RR 最稳健」依赖未定义的边集、数值随编号变化（星形中心在
  端点 28、中段 24），已撤销（L3）。

### A.3 revision-3 的 σ₄₀₀ 外推（本文只读复核）

对 T17 cross 的三对，按 deal 计算 per-deal 胜率 sd，delta 法外推到 400 副：

| 对 | per-deal sd | SE_p(200) | SE_Elo(400) ≈ σ₄₀₀ |
|---|---|---|---|
| lvl1–lvl2 | 0.3330 | 0.0235 | 11.6 |
| lvl2–lvl3 | 0.3183 | 0.0225 | 11.4 |
| lvl3–lvl4 | 0.3096 | 0.0219 | 11.7 |

三对点估计 = **11.583 / 11.451 / 11.673**；本文取名义 **σ₄₀₀=11.6**、敏感性
**[11.4, 11.7]**。旧口径 10.0–11.1（pilot §3.2/§4）只作历史列；10.14/11.1 不可当
revision-3 设计常数（H1）。归一化口径说明：本文**不做 p 归一**——lvl1–lvl2 的
half 胜率≈0.5，raw 11.58 已高于旧「保守」11.1；远 pair 的归一缩放方向取决于方差模型
（×√(q/0.25) 或 ×√(0.25/q)），为避免引入未定义假设，设计常数只用 raw 区间 [11.4,11.7]。

### A.4 尺度换算

probit 差 d 在 p=0.5 处：`Δp = φ(0)·d/141.42 = 0.00282·d`；
`Δ_logistic = 400/ln10 · Δp/(p(1−p))|_{p=.5} = 694.9·Δp = 1.960·d`。
T17 的 24.16 probit → 47.3 logistic Elo。

### A.5 信息量（probit）

`mle.py` 用 probit link（[`mle.py:16-18`](../../src/seven523/mle.py)），每局对「a 与 b 的
差值」的 Fisher 信息 ∝ `w(z)=φ(z)²/[p(1−p)]`，`z=Φ⁻¹(p)`；相对信息 `w(z)/w(0)`。
T17 远锚对（half 胜率）p=0.6863/0.7788/0.8738/0.9038 → **91.8%/80.4%/61.2%/52.6%**，
即改投近 0.5 对的信息增益 **×1.09–×1.90**。旧稿的 `p(1−p)/0.25 = 86%/69%/44%/35%`、
×1.16–2.9 是 logit/比分口径，作废（L1）。带内任意对的预算修正按
`N_ij = ceil(N / (w(z_ij)/w(0)))`。

### A.6 先验权重、行动功效与确认预算（本文推算）

- **先验中心敏感度**：对比量似然方差 `V=2τ²/(mN)=SE²`，两端先验中心各 σ_prior，
  对比先验方差 `2σ_prior²`。MAP 估计对先验中心的敏感度
  `w_prior = V/(V+2σ_prior²)`；σ_prior 的预注册目标为 `w_prior ≤ 10%`（§4.3）。
- **行动功效**：规则为 `est ≥ max(1.96·SE, 10)`；正态口径
  `action_power(Δ)=1−Φ((max(1.96·SE,10)−Δ)/SE)`。Δ=10 ⇒ `max(...)≥10`，
  `action_power=1−Φ((10−10)/SE)=0.50`（与 N 无关）；80% ⇒ `Δ ≥ 10+0.8416·SE`。
- **复合 80% 的 Δ**：SE=3.569（Δ=10 设计点）→ Δ≥13.00；含 `max()` 保守化
  （SE×1.1=3.926）→ Δ≥13.30。
- **确认预算**：`SE_pool = σ·√(400/N)/√k`（A.1 的单对律 + 1/√k 合并）。Δ=10、80%：
  3 bank 简单池化需每 bank ~1,409 副；含 `max()` 保守化（半宽 ×1.1 → N×1.21）
  需 ~1,705 副；4–5 bank × 1,100 副的 power 见 §4.6 表。

## 附录 B：本文只读复算（可用现有产物重现）

### B.1 赛程结构核对

`runs/t17_mle/games.jsonl`：4000 行；`kind=anchor` 1600、`kind=cross` 2400；
distinct seed 400（200 anchor + 200 cross）；每对 400 局（200 副 × 双座位）；
`rules_id=2e36dbea44893696`。

### B.2 直接对局 / half-split / deal 级拟合

- 直接对局：用 `duel.paired_duel_stats` 对每对 cross 局（bootstrap 4000，`rng=random.Random(1)`）：
  lvl1–lvl2 −12.2 [−43.7,+20.9]、lvl2–lvl3 −65.0 [−98.1,−33.1]、
  lvl3–lvl4 −100.0 [−134.0,−68.6]（logistic；点估计与 rng 无关，CI 有 ±1 Elo 级波动——
  旧稿数字对应 rng=3/16/0，L6）。
- half-split：400 deal 随机对半，各 `fit_mle`；相邻差半差 |Δ₁−Δ₂| = 2.7/8.0/1.6 probit（z≤0.9）；
  half-split 只能排除 CI 尺度错误，检不出稳定偏差——失配判定须 holdout（H4）。
- deal 级聚合：每 deal 每对合成一局（scores=两局之和），同一 `fit_mle`（β=100）得
  间距 31.0/43.3/47.2 probit（game-level 24.2/40.8/37.6）、Laplace σ 4.5（game-level 3.8）。
- 先验敏感性：窄先验（σ≈6）拟合 σ=3.8–4.1；T17 默认先验拟合 σ=5.9–6.3，点估计差 <0.5 probit；
  MAP 中心敏感度（先验中心 +20 于 lvl1、测 lvl1−lvl2 间距位移）σ=6/20/50 →
  pull 5.97/0.73/0.12（w_prior≈0.30/0.036/0.006，评审 M2 只读复核）。
- **in-sample vs holdout（评审 H4 只读复核）**：leave-pair-out 拟合把 lvl1–lvl2 的 z 从
  −2.01 放大到 −3.53、lvl3–lvl4 从 +1.46 到 +2.64——in-sample 残差会收缩真实失配，
  故只作诊断（§4.5 第 4 条）。**revision-3 的 bank 方差仍未实测**（T17 单 bank，§5 第 3 条）。

复现（只读、纯 CPU、~1–2 分钟）：

```bash
.venv/bin/python - <<'PY'
import json, collections, math, random
from seven523.elo import PlayedGame
from seven523.duel import paired_duel_stats
rows=[json.loads(l) for l in open('runs/t17_mle/games.jsonl')]
games=[PlayedGame(r['seed'],tuple(r['seats']),tuple(r['scores'])) for r in rows]
for a,b in [('lvl1','lvl2'),('lvl2','lvl3'),('lvl3','lvl4')]:
    sub=[g for g in games if set(g.seats)=={a,b}]
    st=paired_duel_stats(sub,left_id=a,right_id=b,bootstrap=4000,rng=random.Random(1))
    print(a,b,st['winrate'],st['winrate_ci'],st['elo_diff'],st['elo_diff_ci'],st['deal_sign'])
PY
```

## 附录 C：引用清单

- ADR：[`0011`](../adr/0011-openskill-rating-core.md)（OpenSkill 核心）、
  [`0012`](../adr/0012-single-gauge-and-greedy-removal.md)（单 gauge/禁多锚）、
  [`0013`](../adr/0013-drift-free-rating-channel.md)（无漂移通道/MLE 发布/Prior 回流）。
- 报告：[`elo-reliability-audit.md`](./elo-reliability-audit.md) §3.3–3.5/§6.1/§6.5、
  [`head-to-head-pilot.md`](./head-to-head-pilot.md) §1/§3/§4、
  [`evaluation-protocol-validation.md`](./evaluation-protocol-validation.md) §2.1/§3/§5/§6/§9、
  [`structural-directions.md`](./structural-directions.md) §7.1–7.3/§9、
  [`ladder-rerating-paired.md`](./ladder-rerating-paired.md) §1.2/§2/§4、
  [`t17-recalibration.md`](./t17-recalibration.md) §2/§3/§6/§7、
  [`human-elo-10-games-research.md`](./human-elo-10-games-research.md) §6.1–6.3、
  [`tournament-arena.md`](./tournament-arena.md) §1/§5、
  [`README.md`](./README.md) §0/§3。
- 代码：`src/seven523/{elo,ladder,duel,mle,study,arena}.py`、`src/seven523/placement/estimator.py`、
  `tools/{build_ladder,head_to_head,refit_mle}.py`（行号见正文）。
- 只读产物：`traces/study/manifest.json`、`runs/t17_mle/{games.jsonl,absolute_table.json,calibration.json}`。
- 评审：[`evaluation-resolution-review.md`](./evaluation-resolution-review.md)（H1–H5/M1–M6/L1–L11）。

---

## 修订记录（v2，2026-09-27）

逐条对应评审 [`evaluation-resolution-review.md`](./evaluation-resolution-review.md) 的
§0 最小阻塞修复集、H/M/L 条目。改动只在本文件；代码侧新能力（`--keep-rungs`、`--prior-sigma`、
`--band`、`resolution`/`build_transitivity`）按同批实施假设对齐。

| 条目 | 改动摘要 |
|---|---|
| H1 σ₄₀₀ | 全部预算改为 revision-3 σ₄₀₀=11.6（敏感性 [11.4,11.7]）；§0/§3.2/§3.5/§4.2/§4.6 的 N/总对局/功率重算；Δ=10 → N=2,113（总 25,356）；旧 10.14/11.1 与 404/718/1,614 降为历史列；Stage-3 补每 bank ~1,400–1,700 或 4–5 bank 及实际 power；A.3 注明不做 p 归一。 |
| H2 功效口径 | §2.2 拆分「筛选分辨率（CI-only）」与「行动功效（复合规则）」；给出 action_power 公式与 Δ=10 的 50% 上界、80% 需 Δ≈13.0–13.3 或 TOST；§4.4 字段拆成 `power_at_delta`/`action_power_at_delta`（与实现一致）；§4.5 同步。 |
| H3 rungs | 采用「resolution 路径冻结 levels/rungs」：§2.3 新硬约束 4、§4.1/§4.4/§4.7 依赖 `--keep-rungs`（`estimator.rungs_source="kept"`/`rungs_selection_ok`）；§4.5 新验收（间距不跨 100、prior-only 不入 levels/rungs、塌缩不得静默发布）；§4.8/§5 改为「需 owner 决策的新 T 项」。 |
| H4 门禁 | 残差/可传递性改 holdout（leave-one-bank-out，`build_transitivity`）或 parametric bootstrap 零分布；in-sample 只作诊断；删除 `arena.pair_diagnostics` 复用（注明未校准/z 口径差）；`\|z\|>2.5` 标为待预注册、单 bank 报 `not-computed`。 |
| H5 确认门槛 | 全文统一 ≥3 seed × 400（bank 即 seed）；「≥2 bank」只保留给 `between_bank_sd` 估计并分开标注；§4.8 改「修订后一致」。 |
| M1 命令 | `--games A --games B`；`--prior-sigma 20` 升为与 resolution 同批前置并显式给值；示例改 `--device cpu` 并注明并发假设；保留 workers 口径。 |
| M2 先验 | §4.3 定义 `w_prior=V/(V+2σ_prior²)` 与目标 ≤10%，导出 σ_prior≥7.6、推荐 20；明确发布 σ 非 prior 宽度；数据二次使用显式登记 + 只计 Stage-2 局 + 边界敏感性拟合。 |
| M3 seed 台账 | bank 换到 fresh 段 400–402（避开 20–22/30–32/300–308）；`source_seeds` 与命令一致；写明审计口径（`runs/**/*.json` seeds 并集 + `selfplay-pool-plan.md` §3.6/`reward-alignment-plan.md` §4.4–4.5 保留段 + legacy zero_null 说明）。 |
| M4 成本 | §0/§4.6/§5 写明 `--bootstrap 4000` 串行 ≈40–55 min 单核、默认 `--bootstrap-workers 4`（fork，逐位一致）Stage-2 语料 ≈15 min；迭代默认 1000–2000、4000 仅最终发布；打局墙钟与 refit 墙钟分列。 |
| M5 去重 | Stage-2 命令前置 `rm -f`；注明 `--games-out` 追加语义与中断重跑双计风险；建议 run-id 目录与 loader 去重。 |
| M6 覆盖校准 | §4.5 第 5 条改为已知真值/零假设仿真或 ≥29 bank（0.05^(1/29)=0.902）口径；写明 n=5 不可证伪与 bank 自身噪声。 |
| L1 信息权重 | §1.4/§3.3/§3.5/A.5 改用 probit `w(z)=φ(z)²/[p(1−p)]`，权重 91.8/80.4/61.2/52.6%、增益 ×1.09–1.90。 |
| L2 band | 撤销「相邻 ≤30 probit、跨度 ≤60 Elo」，改为 `w_rel ≥ 0.80`（\|Δμ\|≲110 probit）的信息带宽，T17 四档合规（102.6 probit ≈0.82）。 |
| L3 方差和 | A.2 改用 minimax 论证（完整 RR 任意对 Var 相同且最小化最大对比方差），撤销未定义边集的方差和论证。 |
| L4 contrast 同源 | §4.4 示例改为同一 contrast replicate 分布（`ci95_win_prob=[0.5577,0.5776]` 与 `se_elo` 同源），`ci_source` 入块；实现中 `build_resolution` 已按此计算。 |
| L5 省一半 | §3.2/§3.5/A.2 注明「省一半」只对全表 6 对都分辨成立；单目标对 RR 是 3 倍（12,678 vs 4,225）。 |
| L6 rng | §1.3/附录 B 的 CI 改为 rng=1 实跑值（−43.7/+20.9 等），注明点估计与 z 不受 rng 影响。 |
| L7 file:line | 修正 mle.py 尺度/eps 行号、`--games-out` 落盘位置（`ladder.py:527` + `build_ladder.py:108-115`）、manifest 先验读取（refit_mle 的 `manifest_priors(document)`）。 |
| L8 ADR 条款 | 单 gauge 引用改 ADR-0012 决定 1；禁多锚/soft L2 改 ADR-0013 决定 2（替代方案 `0013:46`）。 |
| L9 T9–T11 | §4.3 加 T9（已由 `bootstrap_cis` 吸收）/T10（z 不联动，confidence==0.95 前置）/T11（随 ADR-0011 关闭）对账。 |
| L10 policy_seed | §4.1/§5 第 5 条注明 Stage-2 仅对确定性 argmax 有效；sampling 策略的 entrant-id 派生 + EPV §8 校准为硬前置。 |
| L11 口径统一 | §0/§3.2 与 §4.8 统一为「需 owner 决策的新 T 项 / C-5 例外」，删除「已解/不冲突」过度声明；§0/§1.3 表头同步「旧口径历史表」标签。 |
| §3 补充项 | F6 设备/并发写入 §4.6/§5 第 12 条；σ₄₀₀ p 归一说明入 A.3；between-bank 方差列为 §5 第 3 条阻塞项；`\|z\|>2.5` 无校准出处 → 标待预注册（H4）。 |

未采纳条目：无（评审 §2 两条 refuted 候选不在处置范围；H2 的 TOST 与 H4 的阈值作为预注册后备，未默认启用）。

# 新评估口径独立验证：座位配对 ladder、head-to-head 与联合 Hessian（7鬼523）

> 资产状态（2026-09-25）：本报告引用的模型 checkpoint 已删除，旧路径不再可用；数值为牌型族规则变更前口径。

> **状态：独立验证完成（只读为主，1 处代码 bug 修复 + 回归测试，未 commit）。**
> 验证对象为 `docs/experiments/README.md` §0/§3、`head-to-head-pilot.md`、
> `elo-reliability-audit.md` 的「新口径」主张。结论一句话：
> **head-to-head 的 twin/牌聚簇 bootstrap、左右镜像、判定规则的保守方向都成立；
> 合并 CI 的 80% power 需要把分辨率表上的牌数 ×≈2；pilot 表里的
> `±5.8/±5.8/±8.2` 是合并 SE 而不是 95% CI 半宽（差 1.96×）；
> `elo.py` 的联合 Hessian 在 free-vs-free 对局上把 Fisher 信息重复计入，
> cross 设计 SE 被低估最多 ~19%，已给出最小修复与回归测试。**

验证脚本与产物全部在 `runs/protoeval/`（gitignore）；本文所有数字可由
`runs/protoeval/*.py` 独立复算。除 §4 的一处修复（`src/seven523/elo.py`
+ `tests/test_elo.py`）外，未改动任何被审文件，未 commit。

## 0. 结论总表（主张 → 判定）

| # | 主张 | 独立验证结果 | 判定 |
|---|---|---|---|
| 1 | `ladder.plan_games` 每副牌双座位、候选内换座 → 无相位偏差 | 3 种候选顺序的拟合值逐位相同；每个 (候选,锚点,牌) twin 座位恰好 [0,1]；候选-锚点座位计数 20/20 | **成立** |
| 1b | `elo.py` 自由 id 用联合 Hessian SE | 联合 Hessian **实现有重复计数 bug**（§4）：free-vs-free 每局按 2w 计入；修复后与数值微分 Hessian 一致（<0.1%） | **修复后成立** |
| 2 | `duel.py` + `tools/head_to_head.py`：同牌换座、按牌聚簇配对 bootstrap | 从零实现逐位复算一致；twin 相关时 deal-cluster 半宽是 naive-game 的 1.40× | **成立** |
| 2b | 400 副牌 CI 半宽 ≈±20 Elo | 15 次 ckpt 运行 18.7–20.9（均值 19.7）；random 随机零 ±22.4 | **成立**（略依赖策略类型） |
| 2c | `--seeds` 合并 = `mean ± max(bootstrap SE, seed sd)/√k × 1.96` | 代码实现（`duel.py:257-287`）与文档公式一致；pilot 表的 `±5.8/5.8/8.2` 实际是 `max(...)/√k`（1σ），不是 95% 半宽 | **代码成立；pilot 表标签错**（§2.4） |
| 3 | 20 Elo ≈ 400–500 副牌；10 Elo ≈ 1500–2000；σ_N=σ_400√(400/N) | σ_400=10.14；log-log 斜率 −0.483；表值=50% power。80% power 需 ×≈2.04（20 Elo→807，10 Elo→3226） | **数字成立，但「分辨」需标注为 50% power** |
| 4 | 判定需 3 seed 合并 95% CI 排除 0 且点估计 ≥+20 | +20 门槛使 Δ=20 时 power≈50%（与 N/seed 数无关）；FPR≈0.05–0.13%。README §3 实际只写「CI 排除 0」，与其不一致 | **按字面成立但几乎无功效；文档口径需统一**（§6） |
| 5 | 3 seed 后 seed 间 sd 与 bootstrap 隐含 sd 同量级 | ckpt 15 次运行池化比值 **0.99**（pilot 1.11）；random 40 seed 比值 1.11 | **成立** |
| 5b | `pool4g` 单 seed 显著被 seed 1/2 证伪 | 5 seed 值 −19.6/+8.7/−2.6/+8.3/−3.0，合并 CI 跨 0 | **成立** |

## 1. 验证方法与独立复算脚本

| 脚本（`runs/protoeval/`） | 作用 |
|---|---|
| `audit_math.py` | 手算样例、从零实现 `paired_duel_stats`/`combine`、deal-cluster vs naive bootstrap、镜像校验、pilot 合并公式复算 |
| `audit_hessian.py` | 从零装配正确信息矩阵 + 对 MAP 目标做有限差分 Hessian，量化 shipped SE 的低估 |
| `ladder_order.py` | 候选顺序置换下的拟合不变性、座位平衡 |
| `jsonl_recompute.py` | 仅用 `--games-out` JSONL 复算单 seed 统计与多 seed 合并 |
| `zero_null.py` + `analyze_null.py` | 独立随机零（注入两套独立策略 RNG 流）40 seed × 400 副，单次/合并 CI 覆盖校准 |
| `analyze_pilot.py` | pilot 三元组 5 seed 复现读数、同 ckpt 零 |
| `analyze_resolution.py` | 真实 per-deal 数据上 50/100/200/400/800 副的 σ_N、√N 外推、所需牌数 |
| `analyze_power.py` | 实测 per-deal 分布上的 FPR/power 蒙特卡洛（delta 代理，已对 shipped 管线校验） |
| `run_pilot.sh` / `run_followup.sh` | GPU 复跑：3 对 × 5 seed × 400 副 + 同 ckpt + 左右互换 + 绝对 ladder + greedy-random |

关键复算命令：

```bash
.venv/bin/python runs/protoeval/audit_math.py
.venv/bin/python runs/protoeval/audit_hessian.py
.venv/bin/python runs/protoeval/ladder_order.py
.venv/bin/python runs/protoeval/jsonl_recompute.py
.venv/bin/python runs/protoeval/analyze_null.py 40
.venv/bin/python runs/protoeval/analyze_pilot.py
.venv/bin/python runs/protoeval/analyze_resolution.py
.venv/bin/python runs/protoeval/analyze_power.py 4000
bash runs/protoeval/run_pilot.sh      # GPU，3 对 × 5 seed × 400 副 + clone
bash runs/protoeval/run_followup.sh   # 左右互换 / 绝对 ladder / greedy-random
```

## 2. 代码核对（file:line）

### 2.1 `duel.py`（统计路径）

- **twin 配对**：`plan_duel_schedule`（`duel.py:34-57`）每副牌抽一个 deal seed，
  固定打两局、左右换座，`pairs` 是**牌数**（允许奇数）；`subject=left.id`。
- **聚簇单位**：`paired_duel_stats`（`duel.py:105-255`）按 `seed` 分组，断言每副牌
  恰 2 局且 left 两个座位各一局（`duel.py:144-160`）；bootstrap 循环
  `per_deal[rng.randrange(deals)]` 共 `deals` 次再除以 `deals`（`duel.py:208-219`）
  ——**确实重采样「牌」**，不是局/座位。独立实现（同算法）+ numpy 分位 bootstrap
  与 shipped 结果一致。
- **p / clamp / CI**：`p=winrate=(W+0.5D)/games`；`elo_diff=400·log10(p/(1−p))`
  （`duel.py:72-76`），与 `E=1/(1+10^(−Δ/400))` 互逆；clamp 下界
  `1/(2·deals)`（`duel.py:60-70`）与点估计可达范围一致。注意：bootstrap 重采样
  的 p 下确界是 `0.25/deals`（单牌 0.25 分），会被 clamp 抬到 `1/(2·deals)`，
  是一个轻微的保守截断，幅度可忽略。CI 为线性插值百分位（`duel.py:78-89, 223-244`）。
- **符号检验**：`_binomial_two_sided_p`（`duel.py:91-103`）对「每牌 twin 净胜负」
  做精确双侧二项检验并排除平局；平局占比高时有效 N 远小于牌数（pilot 的 pool4g
  实例），只能作辅助通道。
- **手算样例**：4 副牌 per-deal 得分 [1.0, 0.5, 0.0, 0.25] → winrate 0.4375、
  elo −43.658、W/D/L=3/1/4、符号 1 win/2 loss/1 tie、p=1.0；独立实现逐位一致。
- **左右互换**：CLI 级验证（`probe_swap.json`）：winrate 0.5025/0.4975 和为 1，
  elo ±1.737 精确取反，CI 端点镜像。
- **同 ckpt 双方（不同 id）**：5 seed × 400 副全部 400/400 牌打平，elo 恰 0、
  CI [0,0]（`base680k_vs_baseclone.json`）——调度与确定性策略完全镜像，行为合理，
  但不提供方差信息。

### 2.2 `ladder.py`（座位配对与 JSONL）

- `plan_games`（`ladder.py:93-152`）：每个 (round, anchor) 抽一个 deal seed、
  所有候选共享；每副牌给候选两个座位各打一次；`games_per_anchor` 必须偶数。
  顺序置换下 schedule 与拟合值不变（`ladder_order.py`：3 种顺序 ratings 逐位相同）。
- `play_games`（`ladder.py:155-243`）：policy seed 来自
  `(game.seed + 101*(seat+1)) & 0xFFFFFFFF`（`ladder.py:199`）——**按座位槽而非
  选手 id** 派生；确定性 ckpt 不受影响，但会造成随机策略的退化现象（§3.2、§8）。
- JSONL（`ladder.py:212-231`）：`seed/seats/scores/subject/opponent/subject_seat/kind`。
  仅凭这些字段即可重建 twin、复算全部统计；`jsonl_recompute.py` 验证单 seed 与
  多 seed 合并都能逐位复现 CLI 输出。

### 2.3 `head_to_head.py` / `h2h_screen.py`

- `--seeds` 分支：单 seed 返回原始 stats；多 seed 调 `combine_duel_seeds`；
  `--games-out` 多 seed 按 seed 拆文件（`head_to_head.py:120-142`），避免不同
  deal seed 混进同一文件。正确。
- `h2h_screen.py` 的 `_view`/表格只是展示层，合并值直接取自 `combined`；
  与 CLI 一致。

### 2.4 合并公式 vs pilot 文档（发现不一致）

`_combine_metric`（`duel.py:257-287`）实现：点=各 seed 均值；
`bootstrap_se=rms(各 seed CI 半宽/1.96)`；`seed_sd=stdev(ddof=1)`；
`se=max(bootstrap_se, seed_sd)/√k`；CI=`mean ± z·se`，`z=1.96`。**代码与
README/pilot 文字中的公式一致。**

但 pilot 表 §3.2 的「合并读数 ±」不是这个数：用 `runs/h2h` 的 seed 0–2
逐项复算得

| 对 | 3-seed 均值 | bootstrap SE | seed sd | 代码 95% 半宽 | pilot 表 ± |
|---|---|---|---|---|---|
| base−vf1 | +9.13 | 10.33 | 8.68 | **11.69** | 5.8 |
| base−ctrl | +4.20 | 9.96 | 9.61 | **11.27** | 5.8 |
| base−pool4g | −4.49 | 9.78 | 14.22 | **16.09** | 8.2 |

pilot 的三个数恰好等于 `max(bootstrap SE, seed sd)/√3`（即 **1σ 合并 SE**，
未乘 1.96）。影响：pilot 的 `w5_vf1 +9.1±5.8` 读起来像「95% CI 排除 0」，
而按代码口径 3 seed 的 95% CI 是 **[−2.6,+20.8]，覆盖 0**；pilot 正文
「1.6σ」的定性描述其实与自己的 ± 一致，但公式括号里的 ×1.96 是错的。
§3.2 的「CI 大体校准」（seed sd ≈ bootstrap 隐含 sd）不受影响。

另：README §3 的判定规则只要求「合并 95% CI 排除 0；点估计 <10 不值得追」，
没有 +20 点估计门槛；任务简述和 pilot §5.2 的「筛掉 <20」是筛选建议。
两处口径不同，建议统一（见 §9）。

### 2.5 `elo.py` 联合 Hessian：发现 bug（详见 §4）

`fit_ratings` 在 free-vs-free 对局上把同一局从双方 terms 各装配一次，
对角与交叉项都按 2w 计入（`elo.py:315-327` 修复前）。正确的联合 Fisher
信息是每局 `diag += w, offdiag −= w` 各一次。后果：`--cross>0` 的 SE 被低估，
audit §2.3 的极简复现（纯 x-vs-y、prior 主导）恰好看不到（141.95 vs 142.47，
相差 0.4%）。修复后与 audit 给出的 142.47 一致。

## 3. 零假设校准

### 3.1 确定性零与镜像

| 检查 | 结果 |
|---|---|
| 同 ckpt 两个 id，5 seed × 400 副 | 每 seed 400/400 牌净零，elo=0.000，CI=[0,0] |
| 左右互换（同 seed、同牌集） | winrate 和为 1、elo 精确取反、CI 镜像 |
| `random` vs `random`（CLI，同 spec） | **退化**：200/200 牌精确打平、elo=0（座位槽 RNG 相同 → twin 逐位镜像） |

最后一条意味着任务简报设想的「random vs random（不同策略种子）」**无法用现 CLI
构造**：`policy_from_spec("random", seed)` 对两侧生成的是同一映射，twin 两局在
每个座位上跑完全相同的动作序列。要得到有方差的随机零，必须自备 factory 注入
两套独立流（`zero_null.py` 做的就是这个）。

### 3.2 独立随机零（40 seed × 400 副）

| 量 | 数值 |
|---|---|
| 点估计均值（真值 0） | **−0.20 Elo** |
| seed 间 sd | **12.70** |
| bootstrap 隐含 sd（各次半宽/1.96 的均值） | **11.43** |
| 比值（seed sd / bootstrap sd） | **1.111** |
| 单次 95% CI 覆盖 0 | **37/40 = 92.5%**（二项 95% CI ≈ [79.6%, 98.1%]） |
| 合并 CI 覆盖（disjoint k=2/3/5） | 20/20、13/13、8/8 = 100% |
| 合并 CI 覆盖（滑动窗 k=3） | 37/38 = 97.4% |
| 合并半宽 / 理想 `1.96·sd/√k` | k=2 +15%、k=3 +10%、k=5 +3% |
| 每局和棋率 / 每牌分差 sd | 8.8% / 0.329 |

读法：单次 400 副的 bootstrap CI 对随机策略略偏乐观（覆盖 92.5%，比值 1.11），
但在 40 seed 的噪声内（sd(sd)≈1.4 Elo）；合并规则的 `max()` 把覆盖率抬到
≥95%（保守）。ckpt 三元组的 15 次运行池化比值 **0.99**（bootstrap 与 seed sd
几乎相等），与 pilot 的 1.11 同量级，说明在正式对比场景里 CI 校准没有问题。

## 4. Bug 修复清单（单独列出，未 commit）

**Bug：`fit_ratings` 联合 Hessian 对 free-vs-free 对局重复计入 Fisher 信息。**
位置：`src/seven523/elo.py:315-327`（修复前）。

- 机制：装配循环对每个 free id 遍历其 `terms`；一局 i–j 会在 i 的遍历里给
  `diag[i] += w`、`diag[j] += w`、`offdiag[i,j] -= w`，在 j 的遍历里再来一遍，
  合计 `2w`（正确为每局 `w`）。
- 证据：`runs/protoeval/audit_hessian.py` 用正确装配 + 对 MAP 目标做有限差分
  Hessian 独立验证。3 free × 60 锚点局 × 60 cross 局：shipped SE 27.80/27.45/27.13，
  正确值 30.39/29.81/29.27（低估 7–9%）。扫描最坏（18 free、400 锚点局、100 cross/对）
  平均低估 **19.1%**；无 cross 路径逐位不变（锚点局不触发该分支）。
- audit §2.3 的极简复现为何漏掉：纯 x-vs-y 时信息矩阵接近奇异，SE 由 prior 的
  零方向主导（≈1/√(2τ)），2× 缩放不改变该极限。修复后该复现从 141.95 变为
  **142.47**，正是 audit 正文引用的联合 Hessian 数值。

**最小修复**（已写入工作区）：free-vs-free 只在自己 term 里给自身对角加 w，
交叉项仅在无序对里减一次：

```python
information[i][i] += info
if opponent in index:
    j = index[opponent]
    if i < j:
        information[i][j] -= info
        information[j][i] -= info
```

**回归测试**：`tests/test_elo.py::test_se_joint_hessian_does_not_double_count_free_games`
（20 锚点局 + 20 cross 局，独立手装 2×2 信息矩阵，rel=1e-9；修复前会失败）。
全套测试 **264 passed**（原 263 + 新增 1）。该修复只影响 `--cross>0` 的 SE；
wave5/head-to-head 日常路径（无 cross）数值不变。

**遗留说明**：`window` 非对称保留时，`i<j` 的交叉项只是近似（默认 `window=None`
下精确）；`combine_duel_seeds` 的 `z` 参数与 `confidence` 不联动（非默认
confidence 会错配 1.96），日常只用默认 0.95 即可。

## 5. 分辨率：√N 外推与所需牌数

用 3 对 × 5 seed 的真实 400 副 per-deal 结果，在每 seed 内随机抽 N 副
（2000 次重抽，numpy，clamp 与 shipped 一致）：

| N | σ_N | σ_N / (σ_400·√(400/N)) |
|---|---|---|
| 50 | 28.91 | 1.009 |
| 100 | 20.16 | 0.994 |
| 200 | 14.36 | 1.002 |
| 400 | 10.14 | 1.000 |
| 800 | 7.65 | 1.068（仅 6 个合并样本） |

log-log 斜率 **−0.483**（√ 律 = −0.5）；点估计漂移（子样本均值 − 全样本）
在 N=50 时 +0.25 Elo、N≥200 时 <0.11 Elo，无可察觉偏置。**σ_N=σ_400√(400/N)
成立。**

所需总牌数（两侧共打，2 侧 95%、`power=Φ(Δ/SE−1.96)`；σ_400=10.14）：

| 目标 Δ | 50% power | 80% power | 90% power |
|---|---|---|---|
| 30 Elo | 176 | 359 | 480 |
| **20 Elo** | **395** | **807** | **1080** |
| 15 Elo | 702 | 1434 | 1920 |
| **10 Elo** | **1579** | **3226** | **4319** |

即 pilot/README 的「20≈400–500、10≈1500–2000」是 **50% power**（CI 半宽=Δ）
的口径；把它当「80% 分辨率」会乐观 ~2.04×。这张表的数字本身没有偏高/偏低。

## 6. 判定规则的 FPR 与 power

模拟方法：以 40 seed 随机零（`random_null`）和 5 seed ckpt 对（`ckpt_sym`，
对称化后均值恰 0.5）的实测 per-deal 得分为总体，按目标 Elo 做指数倾斜
（Δ=20 → winrate 0.5287），快模拟器复刻 `paired_duel_stats` 的
`elo_diff` + `combine_duel_seeds` 的 `max()` 合并；对 shipped 管线
（bootstrap=400）校验：点估计逐位一致，CI 端点差 mean ±0.14 / sd 0.32、
相关系数 0.999。每格 4000 次。

**主表（k=3 seed × N=400）：**

| 总体 | 规则 | FPR（单侧/双侧） | power Δ=10 | power Δ=20 | power Δ=30 |
|---|---|---|---|---|---|
| random | CI 排除 0 且点 ≥+20（任务口径） | 0.0013 / — | 0.063 | **0.474** | 0.928 |
| random | CI 排除 0 且点 ≥+10 | 0.0170 / — | 0.260 | 0.776 | 0.979 |
| random | 仅 CI 排除 0（README §3 口径） | 0.0170 / 0.0302 | 0.260 | 0.776 | 0.979 |
| ckpt | CI 排除 0 且点 ≥+20 | 0.0005 / — | 0.045 | **0.480** | 0.951 |
| ckpt | CI 排除 0 且点 ≥+10 | 0.0190 / — | 0.310 | 0.845 | 0.990 |
| ckpt | 仅 CI 排除 0 | 0.0190 / 0.0338 | 0.310 | 0.845 | 0.990 |

**扩大规模（ckpt 总体，仅 CI 排除 0）：** k=5×400 → 0.442/0.497；k=3×800 →
0.453/0.569；单次 1500 副 → 0.470（Δ=10 列），Δ=20 时 0.965。random 总体同量级。

关键结论：

1. **+20 点估计门槛使 Δ=20 时 power ≈ 50%，且几乎不随 N/seed 数改善**——
   因为门槛卡在真值上。若口号的语义是「只有 ≥20 Elo 才值得行动」，正确做法是
   对 +20 做单侧等效/非劣检验，而不是要求点估计 ≥20。
2. 合并 `max()` 偏保守：3 seed 合成半宽比理想宽 ~10%，FPR 3.0–3.4%（双侧，
   低于 5%），代价是 Δ=20 时 power 少 ~7–8pp。这是「宁可保守」的可接受代价。
3. 10 Elo 在任何实际预算（≤2400 总牌）下都只有 26–50% power；**<10 Elo 不值得追**
   成立，10 Elo 若必须判，80% power 需 ≥3200 副。

## 7. pilot 复现（5 seed × 400 副）

seed 0–2 与 pilot 逐位一致（JSON 相同）；新增 seed 3–4：

| 对（base680k − child） | pilot 3-seed 点 | 代码 3-seed 95% CI | 5-seed 均值 | 5-seed sd | 5-seed 代码 95% CI | 结论 |
|---|---|---|---|---|---|---|
| `w5_vf1` | +9.1 | ±11.7（覆盖 0） | **+10.69** | 9.84 | **[+1.64, +19.75]** | CI 边际排除 0，点 <+20 |
| `w5_ctrl` | +4.2 | ±11.3（覆盖 0） | +1.83 | 8.26 | [−6.97, +10.62] | 不可分 |
| `w5_pool4g` | −4.5 | ±16.1（覆盖 0） | −1.65 | 11.50 | [−11.73, +8.42] | 不可分 |

- 15 次运行的 seed sd / bootstrap 隐含 sd 池化比值 **0.99**（pilot 1.11），
  三对各自 0.95 / 0.82 / 1.18，「同量级」成立。
- **`pool4g` 仍不稳定**：5 seed 值 −19.6/+8.7/−2.6/+8.3/−3.0，seed 0 的
  「CI 排除 0」被完全证伪；合并不可分。
- **`vf1 +9~11` 达到边界显著**：5 seed 码值 CI 下界 +1.64（临门一脚），
  5/5 seed 同号（真零下概率 3.1%）。但在 README §3 的 CI-only 规则下这只是
  一个 ~10 Elo 的弱正证据；在 +20 门槛下不构成行动依据。要定论需
  ≥5 seed × 800 副（或 3×1500+）复算。
- **绝对口径 vs h2h（同 800 局/候选，seed 0）**：座位配对 ladder 得
  `base680k 1443.3±16.2`、`w5_vf1 1429.2±15.9`，差 SE=√(16.2²+15.9²)=22.7，
  名义 95% 半宽 **44.4**；同 seed head-to-head 半宽 **19.6**，比值 **2.27×**，
  与 pilot 的 2.3–2.7× 一致。

## 8. 残余风险清单

| 风险 | 会不会影响结论 | 监控/缓解 |
|---|---|---|
| 随机策略的 policy seed 只由 `(牌, 座位槽)` 派生：同 spec 两侧会逐位镜像（退化零）；不同随机策略的 twin 也不是「同一选手同一流」 | 对确定性 ckpt 无影响；对带采样的 RL 策略会改变有效方差与配对含义 | 每次引入随机策略先跑「同 ckpt 镜像」与「独立流随机零」校准（§3）；需要时把 seed 改为按 entrant id 派生 |
| deal seed 32-bit 随机抽，同一 schedule 内理论可碰撞（400 副时 ~2e-5） | 碰撞会让 `paired_duel_stats` 因 4 局同牌直接报错（fail-loud，不会静默错） | 已有断言；超长 schedule 可加去重检查 |
| 平局按 0.5 分；符号检验丢平局 | 高平局对（pool4g 55–59% 牌净零）符号通道有效 N 远小于牌数，与 winrate/margin 结论可打架 | 主推断固定用牌聚簇 winrate/margin bootstrap；符号检验只作辅助并在报告注明 |
| clamp：bootstrap 重采样 p 可低于 `1/(2n)` 被截断 | 只在极端一边倒时可见（winrate→0/1），对本项目 ±20 Elo 场景无影响 | 远离饱和区使用；同时报 winrate CI |
| `--seeds` 各 seed 牌集独立，无 CRN；k=3 的 seed sd 很吵，`max()` 有 +10% 半宽偏置 | 覆盖更保守（FPR<5%），power 少 ~7–8pp；不会造成假阳 | k≥5 或按 bootstrap SE 预注册；不要把 `max` 当无偏 |
| `--bootstrap` 次数 | 200 vs 4000 半宽波动 ~2%；4000 足够 | 保持默认 ≥1000 |
| 多对筛选的多重比较 | 95% CI 规则每对 ~1.7–1.9% 单侧假阳；筛 20 对期望 ~0.35 个假阳 | 预注册候选、Bonferroni/分层，或对任何「获胜对」加 seed 复算 |
| `combine_duel_seeds` 的 z 与 confidence 不联动 | 只用默认 0.95 无影响 | 报告里注明；非默认需同时传 z |
| 绝对 ladder 的跨 fit 值不可比（父模型跨 seed sd 37）；h2h 不依赖锚点 | 不用绝对 Elo 做跨 fit 结论 | 方案对比固定走 h2h；ladder 只做内部相对排序 |
| `--cross>0` 的旧 SE 低估（§4 修复） | 修复前 cross 设计的显著性会被高估最多 ~19% SE | 已修复 + 回归测试；历史 cross 结果（stageB ±4.8）不可复用 |

## 9. 最终推荐协议（一句话）

**口径统一：任何「A 优于 B」在 ≥3 seed × 400 副牌（总 ≥1200 副）的换座
head-to-head 上，报告合并点估计与 95% CI（`mean ± 1.96·max(bootstrap SE, seed
sd)/√k`）；只有当 CI 完全排除 0 且点估计 ≥ +10 才谈「值得行动」，≥20 Elo 的
行动结论建议 5 seed × 400（80% power）或 3 seed × 800 复算，<10 Elo 一律不追
（80% power 需 ≥3200 副）。**

（可选分级：3×200 = 粗筛，只用来杀 <20 Elo 差异；5×400 = 确认；3×1500+ = 10 Elo
级别。README §3 的 CI-only 规则与 +20 筛选建议应合并成上面这一条，避免一处
「排除 0 即可」、另一处「必须 ≥+20」的歧义。）

## 10. 产物

- 新增报告：本文件。
- 新增只读验证脚本与产物：`runs/protoeval/`（`audit_math.py`、`audit_hessian.py`、
  `ladder_order.py`、`jsonl_recompute.py`、`zero_null.py`、`analyze_*.py`、
  `run_pilot.sh`、`run_followup.sh`；`zero_null/`、`repro/`、`abs/`、各 `*.log`）。
- Bug 修复：`src/seven523/elo.py`（`fit_ratings` Hessian 装配）+ 回归测试
  `tests/test_elo.py`；**未 commit**。全套测试 264 passed。

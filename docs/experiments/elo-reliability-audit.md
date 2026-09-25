# Elo 评估可靠性审计：候选顺序、座位相位与 SE（7鬼523）

> 资产状态（2026-09-25）：本报告引用的模型 checkpoint 已删除，旧路径不再可用；数值为牌型族规则变更前口径。

> **状态：只读审计 + 可重复性实验完成（未改 `src/`、未 commit、未触碰
> `traces/study/manifest.json`）；根因修复（§6.1 候选内换座配对）已于 2026-09-25
> 实施（工作区改动，未 commit），端到端验证见 §6.1。**
> **结论：两次定级之间 0–75 Elo 的漂移不是抽样噪声，也不是拟合或 SE 公式的
> bug，而是 `plan_games` 的调度缺陷：候选 Elo 完全由「候选下标奇偶」决定的
> 座位相位决定。同一 seed 下换候选顺序 = 每个牌局的座位全部翻转，评分是
> 确定性的、可逐位复现的。单 fit 的 Fisher SE（≈22）对「同调度换牌局」是
> 校准的；但对「换顺序/换 fit」的比较低了约 √2 倍（跨顺序 SE≈31，观测最大
> 位移 74.6）。**
> 本报告是 [`elo-breakthrough-report.md`](./elo-breakthrough-report.md) §4.4
> 与 `ladder-report.md` 定级口径的可靠性依据；不改写它们的实验结果。

## 0. 关键数字（先看这里）

| 量 | 数值 | 来源 |
|---|---|---|
| 确定性复现 | wave1 12 候选、stageA 18 候选全表 **0 处不一致**；同命令两跑逐字节一致 | `runs/elo_rel_A_wave1_repro.txt`、`runs/elo_rel_D_stageA_repro.txt` |
| seed 间 sd（固定顺序，6 seeds） | 每候选 13.0–28.8；池化 rms **22.9** vs 平均 SE **22.1**（比值 1.04） | `runs/elo_rel_S6_seed{0..5}.txt` |
| 换顺序位移（同 seed 0、同一副牌） | 6 候选：−7.5 ～ **+74.6**；均值 +41.3、sd 29.1 | `runs/elo_rel_C6_rev.txt` vs `runs/elo_rel_S6_seed0.txt` |
| 配对 bootstrap 的换序 sd | 29.5–32.0（均值 **30.7 ≈ √2 × 21.7**）；方差设计效应 ≈ 2.0 | `runs/elo_rel_pergame_report.csv` |
| 座位平衡（每副牌换座各打一次）后 sd | **14.0–15.4**（相对单臂提升 1.34–1.51×） | 同上 |
| wave1→stageA 漂移 | 11 个共有候选中 10 个相位翻转、**10/10 都漂**；唯一未翻相位的 `base680k` 位移 **0.0**；max +74.6 | `runs/ebo_ladder_wave1.txt` vs `runs/ebo_ladder_stageA.txt` |
| 跨候选对局（`--cross>0`）的 SE bug | 纯 X-vs-Y 100 局：shipped SE 34.23 vs 联合 Hessian SE **142.47（4.16×）** | 本报告 §2.3 |
| 座位平衡后的 6 候选点估计 | base680k 1443.3；ctrl680 1417.0；sp_from/mix50 1415.6；ent0 1410.7；sp_r5 1406.6 | `runs/elo_rel_pergame_report.csv` |

## 1. 问题与方法

**问题**：同一批 ckpt 两次定级（wave1 12 候选、stageA 18 候选；均 seed 0、
200 局/锚点、GPU、`--no-traces`；简报写 stageA 19，实际日志/脚本为 18 候选
+2 锚点 = 20 行）中，同一 ckpt 的 Elo 漂移最大 +74.6，而报告
SE 只有 21–23；`base680k` 却精确不变（1447.4）。审计要回答漂移属于
（a）被低估的抽样噪声、（b）候选顺序/座位轮换的系统差异、（c）fit/SE bug，
并量化各自贡献。

**实验矩阵**（全部 `--games-per-anchor 200 --no-traces --seed <s> --device cuda`，
候选 spec 见 `runs/elo_rel_lib.sh`；原始输出在 `runs/elo_rel_*`）：

| 组 | 内容 | 文件 |
|---|---|---|
| A 确定性 | 同命令两跑；wave1 全 12 候选复跑；stageA 全 18 候选复跑 | `runs/elo_rel_smoke_{1,2}.txt`、`runs/elo_rel_A_wave1_repro.txt`、`runs/elo_rel_D_stageA_repro.txt`、`runs/elo_rel_A_S6_seed0_rep.txt` |
| B 换 seed | 6 候选固定顺序 × seed 0..5 | `runs/elo_rel_S6_seed{0..5}.txt` |
| C 换顺序 | 同 seed 0：倒序、左旋 1（均翻相位）、左旋 2（保相位） | `runs/elo_rel_C6_{rev,rot1,rot2}.txt` |
| D 对局级 | 正序/倒序各 400 局原始战绩 JSON + 配对 bootstrap | `runs/elo_rel_pergame_{fwd,rev}.json`、`runs/elo_rel_pergame_report.csv`、驱动 `runs/elo_rel_pergame.py` |
| 校验 | 复现/相位断言汇总 | `runs/elo_rel_verify.txt`、`runs/elo_rel_stats.txt` |

**实验用 6 候选**：`base680k`、`mix50`、`ent0`、`sp_from_sp40k`、`ctrl680`、
`sp_r5_samp`（前 4 个覆盖两次定级中漂移最大的点）。正序下标 0..5。

## 2. 代码审计（file:line）

### 2.1 调度与座位：根因在 `plan_games`（ladder.py:92-140）

```python
# src/seven523/ladder.py:121-131
master = random.Random(seed)
for index in range(games_per_anchor):            # index = 轮次 r
    for anchor_index, anchor in enumerate(anchors):
        deal_seed = master.randrange(1 << 32)    # 每 (轮次, 锚点) 抽一次牌
        for candidate_index, candidate in enumerate(candidates):
            seat = (index + candidate_index + anchor_index) % 2   # :127
```

由此可以逐条推出并被实验证实：

1. **牌局只依赖 `(seed, 轮次, 锚点)`，与候选列表无关。** 同一个 seed 下，
   wave1/stageA/本次实验的所有候选看到的是**完全相同的 400 副牌**（每锚点 200）。
   `master` 的抽样在候选循环之外，候选列表变化不会消耗 RNG。
2. **座位只依赖候选下标与轮次/锚点的奇偶。** 对给定候选，相位 `p = candidate_index % 2`
   一旦翻转，**全部 400 局的座位都翻转**，且牌不变。
3. **候选集合的组成（12 vs 18 vs 6）本身不影响任何人的 Elo**：`cross=0` 时每个
   候选只与两个钉死的锚点对局，BT-MAP 对每个候选是**可分**的（`elo.py:239-245`
   只为每个 id 收集它自己的对局；`_gradient_info` 只用到锚点的固定值）。
   我们验证：左旋 2（换顺序但保相位）与 seed 0 正序**逐位相同**；S6 正序复跑
   与 wave1 对应相位的 5 个候选逐位一致；wave1/stageA 全量复跑 0 处不一致。
   所以「12 vs 18 个候选」不是变量，**下标奇偶才是**。
4. **同一候选从不把同一副牌在两个座位上各打一次。** 每轮每锚点只打一局
   （ladder.py:125-131）；所谓 paired deals 是**跨候选共享牌**，不是**候选内换座配对**。
   `ladder.py:92-95` 的 docstring（"成对牌局、座位轮换"）与 ADR-0006 的措辞
   容易让人以为存在换座 twin，实际并不存在。这是本次漂移的设计根源。

### 2.2 对局确定性与锚点（ladder.py:143-216, policies.py, networks.py）

- 每个候选的 400 局在给定 seed 下完全确定：
  - 对局 RNG 只用 `game.seed` 发牌（`play.py:291`、`game.py:103-110`）；
  - `NeuralPolicy` 默认 `sample=False` → **argmax 确定性**（`networks.py:183-198`，
    `policy_from_spec` 不传 `sample`，`policies.py:141`）；
  - `GreedyBot` 无 RNG（`policies.py:42-56`）；
  - `RandomBot` 每局按 `(game.seed + 101*(seat+1)) & 0xFFFF_FFFF` 播种
    （`ladder.py:171-172`、`policies.py:104/128`）——**种子绑在座位上**，所以
    换座时 random 锚点也会换一条动作流（这是第二通道，但不是主因，见 §4.5）。
- 结论：**「重复跑取平均」不会降低任何方差**；这里没有逐局随机性，只有
  牌集与座位相位两个自由度。

### 2.3 `fit_ratings` 与 SE（elo.py）

- 可辨识性：锚点钉死（`elo.py:231,274-275`，`se=0`），每个自由 id 同时连接
  random/greedy 两个锚点，坐标牛顿 5 轮收敛；所有实验值都在 [1368, 1456]，
  离 `step_cap=300`、clamp [400,2600] 很远（`elo.py:251-263`）。**没有数值 bug**。
- SE 的定义：`_gradient_info`（`elo.py:161-186`）返回**单 id 对角 Fisher**：
  `info = 1/prior.sd² + Σ β²p(1-p)`，最终 `se = 1/√info`（`elo.py:267-273`）。
  - **prior 进入协方差**：是（`elo.py:173`）；n=400 时 prior 权重 ≈1.3%，影响可忽略。
  - **锚点 se=0**：设计如此，不传播。这是「锚点即量纲定义」的建模选择，但意味着
    锚点自身的实现在样本中并非无误差。
  - **每局独立假设**：是。没有聚类、没有座位/牌局交互项。对「同一调度换牌」
    这基本成立（§4.3 的 bootstrap 与经验 sd 都证实），对「换座位相位」不成立。
- ✅ **已修（2026-09-25；工作区改动，未 commit）**：`fit_ratings` 不再直接返回
  对角信息，而是构造自由 id 的**联合信息矩阵**（自由-自由对局同时进入双方对角
  与交叉项，prior 入对角）并用纯 Python Cholesky 求逆（`elo.py:_inverse_diagonal`），
  `se = sqrt([H⁻¹]_ii)`。无 cross 的对角设计逐位不变（回归测试
  `tests/test_elo.py::test_se_stays_diagonal_for_anchor_only_games`）；最小复现
  现在报 **142.47**（与审计原文引用的联合 Hessian 数值逐位一致）；后续独立验证
  （`evaluation-protocol-validation.md` §4）又发现装配循环重复计入 free-free 信息
  （SE 最多低估 19%），已一并修复 + 回归测试。
  `test_se_uses_joint_hessian_when_free_ids_play_each_other` 锁定该行为。
  stageB 的旧输出 ±4.8 仍是旧代码产物，不能回填新 SE；需要时重跑。

### 2.4 `select_rungs` 的 "rungs (1/5)"（elo.py:306-345）

不是 bug：只对候选做选级（`ladder.py:291-293` 不含锚点）。wave1 候选跨度
1376.7–1447.4（70.7），stageA 1367.7–1455.1（87.4），均 < `min_spacing=100`，
从最低点出发找不到第二个 ≥100 的点，故只选 1 个；`ok` 要求 `len==count`
（`elo.py:342-347`）→ 打印 "spacing not satisfied"。它准确地报告了「平台太
平、没有可用梯级」，与漂移问题无关。

### 2.5 CLI 语义（tools/build_ladder.py）

| 参数 | 实际语义 |
|---|---|
| `--seed` | 只喂 `plan_games` 的 master RNG（:165, ladder.py:121），决定牌集；**不影响座位公式** |
| `--games-per-anchor` | 轮次数；每候选总局数 = 2×该值（`cross` 另加）；<100 打 warning（:153） |
| `--device` | 仅 ckpt 策略推理设备；argmax 无随机性，实测 CPU/GPU（CPU 驱动 vs GPU CLI）逐位一致 |
| `--refit` | 只在写 manifest 时生效（study.py:76,90）；`--no-traces` 下完全被忽略 |
| `--no-traces` | `out=None`，跳过轨迹写盘与整个 manifest 分支（:166,196）；**仍会完整拟合**，且不碰 `traces/study/`（本次审计全程使用，manifest mtime 保持 01:44:16 未变） |

## 3. 结果

### 3.1 A：确定性

- 同命令两跑：逐字节一致（smoke；6 候选 seed 0 复跑）。
- **wave1 12 候选复跑**：14 行（2 锚点 + 12 候选）**0 处不一致**。
- **stageA 18 候选复跑**：20 行 **0 处不一致**。
- 左旋 2（换顺序、保相位）与 seed 0 正序：值完全相同。
- 倒序与左旋 1（两者都翻全部相位）：值完全相同（仅打印顺序不同）。
- 结论：评估链路本身无时钟/RNG 非确定性；**同 seed + 同相位 = 逐位可复现**。

### 3.2 B：换 seed（固定顺序）——SE 对「换牌局」是校准的

| 候选 | mean | sd | range | 平均 SE | sd/SE |
|---|---|---|---|---|---|
| base680k | 1432.7 | 28.2 | 75.0 | 22.58 | 1.25 |
| mix50 | 1399.0 | 16.9 | 39.2 | 21.63 | 0.78 |
| ent0 | 1418.1 | 28.5 | 70.0 | 22.15 | 1.29 |
| sp_from_sp40k | 1409.4 | 13.0 | 31.5 | 21.88 | 0.59 |
| ctrl680 | 1425.8 | 16.0 | 39.8 | 22.33 | 0.72 |
| sp_r5_samp | 1407.2 | 28.8 | 80.3 | 21.90 | 1.31 |
| **池化** | — | **rms 22.9** | — | **22.08** | **1.04** |

6 次重复的 sd 本身有 ~30% 相对误差，但池化后与 SE 一致：**Fisher SE 正确反映
了同一调度下换牌集的抽样方差**，不存在数量级低估。

### 3.3 C：换顺序（同 seed 0、同一副牌）——相位位移是确定的

6 候选同 seed 0，正序 vs 倒序（`rev − fwd`，等于 wave1→stageA 的相位方向）：

| 候选 | 正序 Elo | 倒序 Elo | 位移 |
|---|---|---|---|
| base680k | 1447.4 | 1439.9 | −7.5 |
| mix50 | 1380.5 | 1455.1 | **+74.6** |
| ent0 | 1380.5 | 1444.4 | +63.9 |
| sp_from_sp40k | 1397.7 | 1435.5 | +37.8 |
| ctrl680 | 1403.1 | 1432.6 | +29.5 |
| sp_r5_samp | 1383.1 | 1432.6 | +49.5 |
| | | | 均值 +41.3，sd 29.1 |

这些「倒序值」**逐位等于 wave1/stageA 已发布的值**（mix50 1455.1、ent0 1444.4、
ctrl680 1432.6、sp_r5 1432.6、sp_from 1435.5）。也就是说：**两次定级之间
的漂移可以被一次换序实验 100% 复现**，没有任何剩余需要归因于抽样。

### 3.4 D：wave1→stageA 漂移与相位表

| 候选 | wave1 idx | stageA idx | 相位 | wave1 Elo | stageA Elo | Δ |
|---|---|---|---|---|---|---|
| base680k | 0 | 0 | 不变 | 1447.4 | 1447.4 | **0.0** |
| ctrl680 | 2 | 1 | 翻 | 1403.1 | 1432.6 | +29.5 |
| noanneal | 3 | 2 | 翻 | 1396.3 | 1422.5 | +26.2 |
| ent0 | 4 | 3 | 翻 | 1380.5 | 1444.4 | +63.9 |
| ent03 | 5 | 4 | 翻 | 1416.8 | 1412.7 | −4.1 |
| sp_r25_samp | 6 | 5 | 翻 | 1418.3 | 1407.1 | −11.2 |
| sp_r5_samp | 7 | 6 | 翻 | 1383.1 | 1432.6 | +49.5 |
| sp_fixed680 | 8 | 7 | 翻 | 1418.3 | 1396.3 | −22.0 |
| mix50 | 9 | 8 | 翻 | 1380.5 | 1455.1 | **+74.6** |
| sp_from_sp40k | 10 | 9 | 翻 | 1435.5 | 1397.7 | −37.8 |
| h256_scratch | 11 | 10 | 翻 | 1376.7 | 1367.7 | −9.0 |

**10/11 共有候选翻了相位，10/10 都发生漂移；唯一没翻的 base680k 位移恰为 0。**
wave1 独有 `sp40k`（idx 1，奇）与 stageA 独有 7 个新候选不参与该表。

### 3.5 对局级配对 bootstrap：位移是「牌×座位」交互

用正/倒序各 400 局的**同一副牌换座 twin** 做配对（`runs/elo_rel_pergame.py`；
每候选 400 个 `(锚点, 牌)` cluster）：

**机制**：对每副牌定义 δ = 该牌下「坐 0 号位胜率 − 坐 1 号位胜率」。
一次换序把每副牌的座位翻转，评分位移 = **δ 在偶数轮/奇数轮两个任意半边上的
均值之差**，而不是全局座位优势（全局均值通常只有 ±1–3pp）。逐锚点分解示例：

| 候选 | 锚点 | δ(偶数轮) | δ(奇数轮) | 半边差 | 全局平均 δ | fwd−rev 胜率差 |
|---|---|---|---|---|---|---|
| mix50 | greedy | −0.120 | +0.115 | −0.1175 | −0.001 | −0.1175 |
| ent0 | random | −0.095 | +0.075 | −0.0850 | −0.005 | −0.0850 |
| base680k | greedy | −0.030 | +0.020 | −0.0250 | −0.003 | +0.0250（符号随 fwd 座位） |

所以 (i) 全局的 seat 优势很小（这解释了 `elo-breakthrough-report.md` §4.5
「直接测量座位差 <1pp」的观察），但 (ii) **每副牌的 δ 方差很大**，每锚点
200 副牌的两个半边（各 100 副）均值之差轻松达到 0.05–0.12 胜率，在 random 锚点的
饱和区（p≈0.9）
或 greedy 的非饱和区被放大成 30–75 Elo。先手（`starter = argmin(revealed)`）
在换座时也翻转，但按「是否先手」分组的胜率差没有一致符号（±4–9pp，n=100，
SE≈5pp），**不是主因**。

**SE 对比与设计效应**（配对 bootstrap 4000 次，cluster = `(锚点, 牌)`）：

| 候选 | Fisher SE | 单臂 bootstrap sd | 换序差 bootstrap sd | 设计效应 | 座位平衡 Elo | 平衡后 sd | 增益 |
|---|---|---|---|---|---|---|---|
| base680k | 22.95 | 21.20 | 31.60 | 1.89 | 1443.29 | 14.70 | 1.44 |
| mix50 | 21.20 | 20.55 | 31.38 | 2.19 | 1415.63 | 14.48 | 1.42 |
| ent0 | 21.20 | 20.93 | 31.18 | 2.16 | 1410.74 | 14.54 | 1.44 |
| sp_from_sp40k | 21.59 | 19.90 | 29.54 | 1.87 | 1415.63 | 14.68 | 1.36 |
| ctrl680 | 21.72 | 21.10 | 32.00 | 2.17 | 1417.04 | 13.99 | 1.51 |
| sp_r5_samp | 21.26 | 20.70 | 29.59 | 1.94 | 1406.59 | 15.44 | 1.34 |
| **均值** | **21.65** | **20.66** | **30.66** | **2.04** | — | **14.64** | **1.42** |

读法：
- **单臂 bootstrap sd ≈ Fisher SE**（20.7 vs 21.7）→ 对「固定调度换牌」SE 没低估。
- **换序差 sd ≈ √2 × SE**（30.7 ≈ 1.42×21.7）→ 两个相位近似两次独立测量；
  跨顺序比较时正确的 SE 是 **≈31**，不是 21。观测位移最大 74.6 ≈ 2.4×31，
  在 6 个候选中出现一次并不反常。
- **座位平衡（每副牌两个座位都打）把 sd 降到 ≈14.6**，正好是 400→800 局的
  √2 理想增益（`combined_gain` 1.34–1.51）——即换座 pairing 能消除相位项，
  剩下的只是牌抽样方差。
- **有效样本数**：就「跨顺序」而言，400 局的名义样本只有 ≈400/2.04 ≈ **196**
  的有效独立样本；200 局/锚点的旧口径约 98。

## 4. 根因判断

漂移属于 **(b) 候选顺序/座位轮换引入的系统差异**，并伴随一个「(a) 被误读为
抽样噪声」的外衣：

1. **不是 (c)**：拟合数学、阻尼牛顿、clamp、prior 全部正常；确定性复现为 0 误差；
   `select_rungs` 的提示也是对的。唯一的公式缺陷只在 `cross>0` 时出现（§2.3），
   与本轮漂移无关。
2. **不是单纯 (a)**：同 seed 的两次定级牌集完全相同；`base680k` 的控制实验证明
   两次 fit 的锚点/调度/评分逐位一致。漂移 100% 由候选下标奇偶（座位相位）
   决定，且可被换序实验逐位复现。
3. **本质**：`--cross=0` 时候选评分互相独立、只依赖自己的 400 局；而每个候选的
   400 局里，每副牌只打一个座位，座位又由轮次奇偶 + 候选下标奇偶决定。于是
   「候选在列表中的位置」变成了一个未随机化的处理变量：它决定了每副牌 δ 的
   两个半边如何分配给该候选。SE 只条件在**一个已实现的调度**上，
   不含这个调度分量。
4. **数字上的口径**：同调度换牌 → sd≈22（SE 正确）；换顺序/换 fit → sd≈31
   （≈√2×SE）；观测位移均值 ~30、最大 ~75。所以「SE≈22」不能用来判断跨
   fit 差异；「跨 fit 40–75 Elo」在这个设计下完全是预期内的。

## 5. 对既有结论的影响

### 5.1 `ladder-report.md` 的 100 局/锚点与 1490 平台

- **SE 口径**：100 局/锚点（共 200 局）报 SE 27–34，与经验 sd 一致
  （400 局 rms 22.9 → 200 局约 32）。SE 本身没错。**「100 局/锚点 + 同一 fit 内
  相邻级比较」的用法大体站得住**（梯级间距 113–130 Elo 远大于 SE）；
  但把它当跨 fit 绝对值、或用来支撑 2 SE 以内的排序，则站不住。
- **但跨 fit/跨顺序比较要乘 √2 且加相位项**：按相位效应随局数 1/√n 缩放估算，
  200 局下换序 sd 约 43（30.7×√2），所以「每级 113–130 Elo / 3.5–4.5 SE」的间距
  在最坏情况下应读作约 2.5–3σ，仍然大概率成立，但显著性被高估；「平台内各点 <2 SE 不可分」
  的结论则被强化（真实不可分性更大）。
- **1490 平台顶部**：`base680k` 旧测 1489.4 ± 34.3（100 局/锚点、另一次 fit）；
  同 ckpt 在 wave1/stageA（400 局）是 1447.4 ± 23.0，座位平衡后 **1443.3 ± 14.7**。
  1489 很可能是那次调度/相位向上偏移的产物；平台顶部更接近 **1440–1460**
  （与 `elo-breakthrough-report.md` §1.4/§5 的复评一致）。
- 好消息：平台的**相对**结论（1380–1490 带内无显著高低、1M 步不超 680k）不
  依赖绝对值，仍然成立。

### 5.2 stageA 排名（mix50 1455 领先）

**不可信**。stageA 内候选全距 1367.7–1455.1（87.4 Elo），而单次换序就能让
`mix50` 在 1380.5 与 1455.1 之间跳（74.6），ent0 跳 63.9，ctrl680/sp_r5 跳
29.5/49.5。座位平衡后的同牌集点估计：

| 候选 | 座位平衡 Elo ± bootstrap sd |
|---|---|
| base680k | **1443.3 ± 14.7** |
| ctrl680 | 1417.0 ± 14.0 |
| sp_from_sp40k | 1415.6 ± 14.7 |
| mix50 | 1415.6 ± 14.5 |
| ent0 | 1410.7 ± 14.5 |
| sp_r5_samp | 1406.6 ± 15.4 |

E 系列 5 个候选互差 ≤10.4 Elo（< 1 个平衡 SE），**彼此不可分**；base680k 领先
最好 E 系列 26.3 Elo，约 1.3σ，也只是「很可能略强」，不是「显著突破」。
这与 `elo-breakthrough-report.md` §4.5 的直接对 greedy 结果（mix50 62.9% 低于
base 63.0%）一致：**stageA 的 mix50 第一名是相位假象**。

### 5.3 stageB（`--cross 800`）的 ±4.8

**低估**。5600 局/候选里含大量自由-自由对局，`fit_ratings` 的对角 Fisher 没有
联合 Hessian 求逆；最小复现低估 4.16×（§2.3）。`runs/ebo_ladder_stageB.txt`
的 ±4.8 不能用于显著性判断；建议以联合 Hessian / cluster bootstrap 重算后再解读
（其点估计排名本身也仍带座位相位问题）。

### 5.4 `traces/study/manifest.json`

> **勘误（2026-09-25 refit 执行后）**：本节描述的是 refit 前的冻结状态。经换座配对
> 5 seed × 400 局/锚点复测（即 §6.1 设计），manifest 已于 2026-09-25 refit
> （lvl1 1146.3 / lvl2 1219.2 / lvl3 1363.6 / lvl4 1429.7），见
> `ladder-rerating-paired.md` §4.4 与 `README.md` §0.1。

冻结的 lvl1–lvl4（1119.9 / 1232.9 / 1359.3 / 1489.4）来自 100 局/锚点的另一次
调度，各自带一个未知的相位偏移（lvl4 可能被高估 ~40 Elo）。D1 用它作相对标签
不受影响（梯级间距大），但**不要用 `--refit` 在未重测的情况下覆盖**；后续正式
重测应先落地 §6.1 的换座配对设计。

## 6. 改造建议（按收益/成本排序）

1. ✅ **候选内换座配对（根因修复，已实施 2026-09-25；工作区改动，未 commit）**：
   每副牌对每个候选打两局、座位互换，仍与所有候选共享同一副牌。
   - `plan_games`（`ladder.py:101-148`）：`games_per_anchor` / `cross` 现要求
     **偶数**；每个 deal seed 固定发两局（座位各一），候选顺序不再影响其
     牌局集合。
   - 回归测试：`tests/test_ladder.py:76`（order-invariance）与 twin 座位断言。
   - 端到端验证（`runs/elo_rel_seatpaired_verify.txt`，6 候选 × 400 局/锚点 =
     200 副牌 × 双座位，单次 fit）：`base680k 1443.3±16.2`、`mix50 1415.6`、
     `ent0 1410.7`、`sp_from_sp40k 1415.6`、`ctrl680 1417.0`、`sp_r5_samp 1406.6`
     ——与 §5.2 的配对 bootstrap 点估计**逐位一致**（差 0.0），Fisher SE
     15.5–16.2 与 bootstrap sd 14.0–15.4 同量级。
   - 语义注意：`--games-per-anchor` 仍指每个 (候选, 锚点) 的**总局数**（需偶数），
     一半对应牌数；要复现旧的 200 副牌规模请传 400。
   - 与旧数据不可直接混用：新调度不再有「相位」，旧单 fit 的 ±SE≈22 口径作废。
   - 剩余（建议 #3）：`--cross>0` 的联合 Hessian SE 尚未实现。
2. ✅ **保存逐局战绩（已实施 2026-09-25）**：`--no-traces` 原先把每局原始结果
   全丢了，事后无法配对/bootstrap；现落一个每局 `(subject, anchor, deal_seed,
   seat, scores)` JSONL（见下），本轮审计用 `runs/elo_rel_pergame_*.json` 证明
   的用途现在有了正式接缝。
   - `ladder.play_games(..., results_out=PATH)`：每局结束立即 append 一行
     `{"seed","seats","scores","subject","opponent","subject_seat","kind"}`
     （`kind` = `"anchor"`/`"cross"`）；`results_out=None` 行为不变。
   - `ladder.build_ladder(..., games_out=PATH)` 透传；`tools/build_ladder.py
     --games-out PATH`（与 `--no-traces` 正交）与 `tools/head_to_head.py
     --games-out PATH` 均可用。head-to-head 多 seed 时按 `_seed<k>` 分文件，
     避免不同 deal 集抽到同一牌 seed 破坏 twin 配对。
   - 顺带（`head-to-head-pilot.md` §5.4 后续）：`tools/head_to_head.py
     --seeds 0,1,2` 对同一对候选在每个 seed 各跑 `--pairs` 副牌并按
     wave5 §4.2 合并（点估计 = 各 seed 均值；CI =
     `max(bootstrap 隐含 SE, seed 间 sd)/√k × 1.96`），文本与 `--json` 同时给
     每 seed 与合并读数；单 seed 输出与旧版逐位一致，另有
     `tools/h2h_screen.py` 批量跑「多对 × 3 seed × 200 副」汇总表。
   - 回归：`tests/test_ladder.py`（JSONL 行数与返回一致、字段与
     `ScheduledGame`/`PlayedGame` 对应、append、同 seed 逐行一致、CLI
     `--no-traces --games-out`）与 `tests/test_duel.py`（合并公式手算校验、
     真实 stats 兼容、多 seed CLI / `--games-out` / h2h_screen smoke）；
     全套 **263 passed**（2026-09-25）。
3. **SE 改为聚类稳健 / 完整 Hessian**：对 `--cross>0` 用联合 Fisher 求逆
   —— ✅ **已实施**（2026-09-25，见 §2.3；`elo.py:_inverse_diagonal`，无 cross
   路径逐位不变）。仍待做：对 `(锚点, 牌)` / `(牌, 座位)` 做 cluster bootstrap
   以处理配对牌局的残差相关（当前 SE 仍假设每局独立）。
4. **报告多相位/多 seed 区间**：至少 2 个 seed × 2 个相位（或直接换座配对）；
   若沿用现有单 fit 数据，跨 fit 比较请用 ≈√2×SE，并在报告里显式注明。
5. **候选比较优先 head-to-head + 同牌 CRN**：绝对锚点 Elo 在 random 锚点饱和区
   对几个牌局的翻转极其敏感；两个候选在同一副牌上直接对局、按牌配对统计胜率，
   方差远小于各自对锚点的绝对分。
6. **在修好设计后再堆局数**：局数只压牌抽样项（22→31→…），不消相位项；
   当前 400 局/候选 → 800 局/候选可把单臂 SE 压到 ≈15，要到 ≈11 需 1600 局/候选，
   而相位位移仍是 ±30–75 量级。
7. **数据抢救（如果不想立即重跑）**：wave1 与 stageA 恰好是同一 seed 下的
   两个互补相位；把两轮同 ckpt 的值取平均（等价于座位平衡的一阶近似）：
   mix50 1417.8、ent0 1412.5、ctrl680 1417.9、sp_from 1416.6、sp_r5 1407.9、
   base680k 1443.7（wave1 + 倒序值）。这与 §5.2 的平衡拟合一致，可作为临时口径，
   但只覆盖同时出现在两轮的 11 个候选。

## 7. 复现命令与产物

```bash
# 候选 spec 与 runner（只读脚本，产物全在 runs/elo_rel_*）
source runs/elo_rel_lib.sh

# A: 确定性 / 两次定级全量复现
run_ladder runs/elo_rel_A_wave1_repro.txt 0 200 "${W1[@]}"   # 见 §3.1；0 处不一致
run_ladder runs/elo_rel_D_stageA_repro.txt 0 200 "${SA[@]}"  # 见 §3.1；0 处不一致

# B: 换 seed（固定顺序）
for s in 1 2 3 4 5; do run_ladder runs/elo_rel_S6_seed${s}.txt $s 200 "${S6[@]}"; done

# C: 换顺序（同 seed 0；倒序/左旋 1 翻相位，左旋 2 保相位）
run_ladder runs/elo_rel_C6_rev.txt  0 200 "$SPR5" "$CTRL" "$SPFROM" "$ENT0" "$MIX" "$BASE"
run_ladder runs/elo_rel_C6_rot1.txt 0 200 "$MIX" "$ENT0" "$SPFROM" "$CTRL" "$SPR5" "$BASE"
run_ladder runs/elo_rel_C6_rot2.txt 0 200 "$ENT0" "$SPFROM" "$CTRL" "$SPR5" "$BASE" "$MIX"

# D: 对局级原始战绩 + 配对 bootstrap（CPU 即可，2 分钟；与 GPU CLI 逐位一致）
.venv/bin/python runs/elo_rel_pergame.py 200 cpu 4000 \
    > runs/elo_rel_pergame_stdout.txt 2>&1
```

产物（均在 gitignore 的 `runs/` 下）：
`elo_rel_smoke_{1,2}.txt`、`elo_rel_A_wave1_repro.txt`、`elo_rel_D_stageA_repro.txt`、
`elo_rel_S6_seed{0..5}.txt`、`elo_rel_A_S6_seed0_rep.txt`、`elo_rel_C6_{rev,rot1,rot2}.txt`、
`elo_rel_pergame_{fwd,rev}.json`、`elo_rel_pergame_report.csv`、`elo_rel_pergame.py`、
`elo_rel_verify.txt`、`elo_rel_stats.txt`、`elo_rel_lib.sh`。
原始两轮日志：`runs/ebo_ladder_wave1.txt`、`runs/ebo_ladder_stageA.txt`。
`traces/study/manifest.json` 全程未改（mtime 保持 2026-09-25 01:44:16）。

## 8. 一句话给下游

**在同一 seed 下，候选 Elo 是「候选下标奇偶」的确定性函数：换序 = 所有牌换座，
位移 0–75 Elo；单 fit 的 ±21–23 只覆盖换牌，不覆盖换序。**
新的定级口径必须先做候选内换座配对（或至少报告跨相位/跨 seed 区间），
再用「同牌配对比较」而不是绝对锚点 Elo 宣布强弱。

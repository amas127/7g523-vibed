# Checkpoint 锦标赛竞技场：一次评比整个训练联盟（7鬼523）

> 资产状态（2026-09-25）：本报告引用的模型 checkpoint 已删除，旧路径不再可用；数值为牌型族规则变更前口径。

> **状态：工具已实现（全套测试 279 passed），pilot 双 seed 已跑完**
> （28 参赛者 = 26 候选 + Random/Greedy 锚点，22,620 局/seed，CPU 6 worker）。
> **一句话结论：竞技场给出一个可比的联盟 Elo 表，但 top-7 是统计平局集团
> （w5_scratch、base675840、w5_pool4g、w5_ctrl、lvl4(=base680k)、
> w5_bigbatch 等，seed 0 内相差 ≤22 Elo）；两 seed 复跑的 Spearman ρ=0.54、
> 候选平均 |ΔElo|=27，说明单 fit 的 ±11 名义 SE 低估了换牌波动
> （implied seed-sd ≈16）。训练进度：lvlbase 系在 ~184k 步到达 ~1370 平台，
> 此后到 1M 的净提升只有 ~+16±16；lvlsp self-play 全程平（~1394），与
> lvlbase final 不可分。与已有 head-to-head/复评结论一致：没有任何 w5 臂
> 显著超过 base680k，lvlbase/lvlsp final 不可分。**
> 工具：`src/seven523/arena.py` + `tools/arena.py`。bf16 实测无收益、已移除
> （见 §2.2）。加速唯一有效手段是 spawn 多 worker：全量 22,620 局 37.8 s
> （598 games/s），W=1 约 110 games/s。

## 0. 关键数字（先看这里）

| 量 | 数值 | 来源 |
|---|---|---|
| pilot 规模 | 26 候选 + 2 钉死锚点；22,620 局/seed | §3 |
| 候选间 games | 每个候选 1,620 局（2 锚点 × 60 + 25 对 × 60） | §3 |
| 全量单 seed 用时 | **37.8 s / 37.6 s**（CPU，6 worker，598/601 games/s） | §2.1 |
| W=1 速率 | 109.6 games/s（同机 CPU） | §2.1 |
| W=1 vs W=4 结果 | 所有 Elo 逐位一致（回归测试 + 实测） | §2.1 |
| 单 fit 名义 SE | 11.2–12.1 Elo/候选 | §4 |
| 两 seed 候选漂移 | mean \|Δ\|=27.3，sd(Δ)=22.8 → implied seed-sd **16.1**（≈1.4× 名义 SE） | §5 |
| 两 seed 排名相关 | Spearman ρ=**0.541**；名次最多移动 18 位（w5_scratch 1→19） | §5 |
| 同 ckpt 零假设 | base20480 与 lvl2 同权重：Elo **逐位相同**，h2h 60 副牌 60/0/60、CI [0,0] | §7 |
| lvlbase 系进度 | 20k 1225 → 184k 1356 → 512k 1374 → 999k 1372（两 seed 均值）；~184k 后平台 | §6 |
| lvlsp self-play | 1.02M–2.0M 累计步全程 ~1394，回归斜率 ≈0 | §6 |
| 与 h2h 对账 | lvl4−vf1 **+8.7**（pilot +9.1）；lvl4−ctrl −20.3（pilot +4.2）；lvl4−pool4g −11.6（pilot −4.5），全部跨 0 | §8 |
| TensorBoard | `runs/arena/pilot_tb`（16 个 `arena/elo/<id>` scalar，sp 系用累计步） | §1.3 |

## 1. 工具

### 1.1 赛程与估计（复用既有接缝）

`src/seven523/arena.py` 是 `ladder.plan_games` 之上的一层编排，不新增赛程语义：

* **每人对每人**：复用 `plan_games(entrants, games_per_anchor, cross, seed)`——
  每个候选对每个锚点 `games_per_anchor` 局、每对候选之间 `cross` 局，都按
  「每副牌双座位」打；锚点默认 `random=1000` / `greedy=1315` 钉死。
* **一次联合拟合**：全部 `PlayedGame` 按赛程顺序交给 `elo.fit_ratings`
  （BT-MAP + 联合 Hessian SE，审计 §4 修复后的版本），得到同一把尺子上的
  联盟 Elo；`window`/`prior` 等选项原样透传。
* **诊断输出**：`pair_stats` 汇总每对 W/D/L 与左方得分率；
  `pair_diagnostics` 给出「观测 vs Elo 期望」的 deal 聚簇 z 值（可传递性烟测，
  `z` 用 `games//2` 副牌近似，不是校准检验）。

### 1.2 并行与一致性

`arena.play_parallel` 把 schedule 切成**连续的、大小均衡的 shard**，用
`ProcessPoolExecutor(mp_context=spawn)` 跑 `ladder.play_games`，再按 shard
顺序合并（`split_schedule` 是纯函数，有覆盖/顺序/均衡单测）。要点：

* **确定性**：每局的 policy seed 只由 `(game.seed, seat)` 派生，与 shard 无关；
  ckpt 策略是 argmax，random/greedy 每局重建。因此 `workers=1` 与
  `workers>1` 返回**逐位相同的 `PlayedGame` 列表**（`tests/test_arena.py`
  的 W=1 vs W=3 回归测试；实测 W=1 vs W=4 的 Elo 也逐位相同）。
* **worker 内缓存**：`policies.policy_from_spec` 的 `_AGENT_CACHE` 按
  `(path, device)` 缓存已加载 `Agent`，worker 进程长驻 ⇒ 每个 ckpt 每 worker
  只读盘一次而不是每局一次；单测用 monkeypatch 计数 `networks.load_agent`
  断言两次构建只加载 1 次。
* **CPU 线程**：worker 里 `torch.set_num_threads(1)`，避免 6 进程 × 32 线程
  互相抢核。
* **games-out**：`--games-out DIR` 让每个 shard 写 `DIR/shard_NNNNN.jsonl`
  （开跑前删除旧 shard，避免重复追加）；单进程时就是 `shard_00000.jsonl`。

### 1.3 CLI

```bash
uv run --group train python tools/arena.py \
  --entrant base680k=ckpt:runs/probe/base_step00696320.pt \
  --glob 'runs/probe_sp/sp_step*.pt' --every 8 --last \
  --entrant lvlsp=ckpt:runs/lvlsp__1__1790314004/agent.pt \
  --cross 60 --games-per-anchor 60 --seed 0 \
  --workers 6 --device cpu \
  --out runs/arena/pilot.json --games-out runs/arena/pilot_games \
  --tb runs/arena/pilot_tb --step-map lvlsp=1999424
```

* `--entrant [ID=]SPEC`：`random` / `greedy` / `ckpt:<path>`；裸 `ckpt:` 的 ID
  自动从文件名生成（`base_step00020480.pt` → `base20480`；
  `w5_ctrl__1__1790319698/agent.pt` → `w5_ctrl`），重复 ID 直接报错。
* `--glob` + `--every k` + `--last`：自动发现（按路径排序后每 k 个取一个，
  `--last` 补最后一个），适合「每 100k 一个快照」的批量定级。
* `--out`：JSON 联盟表（每参赛者 Elo ± SE、games、step、锚点、逐对 pair_stats、
  可传递性诊断）。`--tb`：把带训练步的参赛者写成 `arena/elo/<id>` scalar
  （同一前缀在 TensorBoard 里聚合为一图，x=训练步）；步数优先 `--step-map`，
  否则从 ID/spec 的 `stepNNNN` 解析。lvlsp 系用累计步（warm start 于 base
  final 999,424）以便和 lvlbase 系接在同一条时间轴上。
* `--workers` 默认 1，`--device cpu|cuda`（CUDA 只是把前向放 GPU，本游戏不划算，
  见 §2.2）。

### 1.4 测试

`tests/test_arena.py`（15 个用例，全绿）：shard 覆盖/顺序/均衡、空 shard、
W=1 vs W=3 逐位一致 + JSONL 合并顺序、`run_arena` 排序与文档字段、pair_stats/
diagnostics、自动 ID 与步数解析、脚本 factory、ckpt 加载缓存、TB 写入、
CLI smoke/glob/重复 ID 报错。全套 `uv run pytest -q`：**279 passed**。

## 2. 并行实测

### 2.1 加速（CPU，RTX 4060 Laptop 机器，32 核）

| 配置 | 局数 | 用时 | games/s | 备注 |
|---|---:|---:|---:|---|
| W=1 | 840（4 ckpt 子集） | 7.7 s | 109.6 | 含每局策略构建开销 |
| W=4 | 840（同子集） | 3.4 s | 250.1 | 短赛程里 spawn/import 固定开销未摊薄 |
| **W=6** | **22,620（全量）** | **37.8 s** | **598.3** | seed 0 |
| W=6 | 22,620（全量） | 37.6 s | 601.1 | seed 1，复现 |

全量 W=6 相对 W=1 的有效加速 ≈5.5×（短子集的 2.3× 主要是 4 次 spawn +
torch import 的固定成本）。W=1 与 W=4 的联盟 Elo 逐位一致，证实并行只改变
墙钟时间、不改变结果。

### 2.2 bf16（已移除）

曾在 `NeuralPolicy` 里加过 `torch.autocast("cuda", bfloat16)` 开关并做正确性
核对：同一 ckpt、同一 seed、600 局 fp32 vs bf16 的**逐局一致率 597/600 =
99.5%**（3 局分数不同，最大单局分差 35 分），速度 53.4 → 54.6 games/s
（+2%，噪声内）。瓶颈是 Python 规则逻辑与每决策的同步，不是 128 隐藏层的小
MLP 前向；同一赛程 CPU 6 worker 是 598 games/s。结论：**bf16 无收益，开关与
相关测试/产物已删除**，加速只保留 spawn 多 worker（§1.2）。

## 3. Pilot 设置

* 参赛者（26 候选）：`runs/probe/base_step*` 每 8 个取 1 + 最后（7 个：20k、
  184k、348k、512k、676k、840k、983k）；`runs/probe_sp/sp_step*` 同样 7 个；
  `lvl1..lvl4`、`lvlbase`(=lvlbase final @999,424)、`lvlsp`(@999,424)；
  `w5_{ctrl,vf1,pool4g,bigbatch,mix_samp,scratch}` final；锚点 random/greedy。
* **同权重校验**（sha256 前 16 位）：`lvl2 == base_step00020480`、
  `lvl3 == base_step00102400`、`lvl4 == base_step00696320`（=base680k）；
  `lvlbase final` 与 `base_step00983040` **不同**（999,424 vs 983,040）。
  因此 lvl2 是零假设校验点，lvl4 就是 h2h 报告里的 base680k。
* 赛程：`--cross 60 --games-per-anchor 60 --seed {0,1}`。每候选 1,620 局 =
  2×60 锚点局 + 25 对 ×60 候选局；每对 60 局 = 30 副牌 × 双座位。
* 运行：`--workers 6 --device cpu`，产物 `runs/arena/pilot_cpu{,_seed1}.json`
  与 `runs/arena/pilot_cpu{,_seed1}_games/shard_*.jsonl`（均在 gitignore 的
  `runs/` 下）。

## 4. 联盟排名（seed 0）

见文末表 3（完整 28 行，含 seed 1 对照）。seed 0 读法：

* **平局集团**：第 1–6 名（w5_scratch 1407、base675840 1401、w5_pool4g 1400、
  w5_ctrl 1393、lvl4 1392、w5_bigbatch 1392）相差 ≤16 Elo，差值 SE ≈16，
  完全不可分；第 7–15 名（w5_mix_samp … w5_vf1）也在 ~1.5 个候选 SE 内。
* **可分辨的只有梯队**：random(1000) < lvl1(1169) < base20480≈lvl2(1230) <
  greedy(1315) < lvl3(1326) < lvlbase(1354) ≈ w5 集团(~1370–1407)。即
  「12k 步的弱模型、20k 步模型、脚本 greedy、100k 步模型、1M/500k 步模型」
  这些**跨几十到几百 Elo** 的差距稳定，同梯度内部的 <40 Elo 差异不可分。
* lvlbase final（1354.3）与 base983040（1354.9）基本重合，印证二者是
  同一训练曲线上的相邻快照。

## 5. 两 seed 复核

| 指标 | 数值 |
|---|---|
| Spearman ρ（28 人名次） | **0.541** |
| 候选 mean \|ΔElo\| / sd(Δ) | 27.3 / 22.8 → implied seed-sd **16.1** |
| 单 fit 平均名义 SE（候选） | 11.3（s0）/ 11.4（s1） |
| 最大名次移动 | w5_scratch 1→19、base675840 2→20、w5_pool4g 3→1 |
| 稳定的部分 | 锚点、lvl1、lvl2、lvl3、lvlbase≈base983040 的相对位置 |

读法：**单 seed 的 ±11 是「换牌条件 SE」，不是「换一批牌的可复现误差」**；
跨 seed 实际 sd ≈16（1.4×），排名 top-7 在 seed 间大幅洗牌。要给 top 集团
排序，至少需要多 seed 合并（复评报告用 5 seed × 800 局/候选得到 lvl4 的
跨 seed sd 28.2、合并 SE 7.1）或把每对局数提高到几百（h2h pilot 的
400 副牌/对 ±20 Elo）。本竞技场默认赛程（60+60/候选）适合「分梯队 + 定级」，
不适合「top 内部排名」。

## 6. 训练进度曲线

TensorBoard：`runs/arena/pilot_tb`（`arena/elo/<id>`，lvlsp 系用累计步）。
下方为两 seed 均值的 ASCII 图（B = lvlbase 系，S = lvlsp 系）：

```
Elo   lvlbase 系 (B) 与 lvlsp 系 (S)，两 seed 均值，x=累计环境步
1430 |                                                                              |
1415 |                                                                              |
1399 |                                       S                         S     S    SS|
1384 |                          BB    B             S     S     S                   |
1369 |             B      B                 B                                       |
1353 |       B                                                                      |
1338 |    B                                                                         |
1323 |                                                                              |
1307 |                                                                              |
1292 |                                                                              |
1277 |                                                                              |
1261 |                                                                              |
1246 |                                                                              |
1231 | B                                                                            |
1215 |                                                                              |
1200 |                                                                              |
     +------------------------------------------------------------------------------+
      0k     200k   400k   600k   800k  1000k 1200k 1400k 1600k 1800k 2000k
```

**lvlbase 系（表 4a）**：20k→184k 提升 **+131 Elo**（两 seed 分别 +104/+159，
远超差值 SE≈16，稳健）；184k→999k 只剩 **+16±16**（1σ），中间 676k 的
高点（1401）在 seed 1 复现失败（1378.5）。结论：**该配方在 ~184k 步就基本
到平台（~1370），之后 800k 步无系统进步**；晚段的 ±20–45 波动主要是换牌
抽样，不是真实涨落。这与 `ladder-report.md`「60k 步后信号耗尽」和
`ladder-rerating-paired.md`「lvl4 ≈ lvlbase_final」一致（绝对高度不同是因为
本竞技场赛程/先验不同，只能内部比较）。

**lvlsp 系（表 4b）**：从累计 1.02M 到 2.0M 步，两 seed 均值 1394.6 → 1394.1，
全程 1380–1397，看不出趋势。它比同 fit 的 lvlbase final 高 +22（seed0
+20 / seed1 +24），但这个差在换牌 sd（~16–20）内，且复评报告的 24,000 局
合并 fit 给的方向相反（lvlsp −11.6±10）。**结论：self-play warm start 既没
突破平台，也没有显著退化；不产生新的对外强度**。

**「谁真的更强」**：没有任何参赛者在两 seed 或 h2h 上稳定超过 base680k
（lvl4）。seed 0 的点估计冠军 w5_scratch（1407）对 lvl4 的 2-seed 合并 h2h
是 −37.8 [−85.7,+8.7]（正=scratch 更强，符号检验 p=0.33），而 seed 0 的联合
fit 只给 +15.6、seed 1 给 +0.9——名义上略优，统计上不可分。base675840 与
lvl4 同样不可分（h2h +20.3 [−17.4,+58.5]）。**联盟没有新的「更强模型」，
只有一条到 ~184k 为止的上升曲线和之后的大平台。**

## 7. 零假设与实现校验

* **同 ckpt 双 ID**：base20480 与 lvl2 权重 sha256 相同。竞技场给出两者
  Elo **完全相同**（seed 0: 1230.4391±11.5758；seed 1: 1218.9739），
  两 seed 合共 60 副牌（120 局）h2h 为 60/0/60，winrate 0.5，bootstrap
  CI **[0,0]**。
  这同时验证了座位配对、赛程与拟合的零点是干净的。
* **确定性 twin 的零宽 CI**：`lvlbase − base983040` 的 60 副牌 h2h 也是
  精确 0.0 [0,0]（每副牌的换座 twin 恰好 1:1），说明确定性 argmax 策略下
  「每副牌内部抵消」会让 CI 变窄到 0；对不确定策略不会这样。
* **锚点行为**：random 钉死 1000、greedy 钉死 1315，二者 se=0；
  lvl4 对 random 的 h2h winrate 0.933、对 greedy 0.658，方向与幅度合理。

## 8. 与已有 head-to-head / 复评报告对账

2-seed 合并的 arena 逐对 h2h（`runs/arena/h2h_pooled.txt`，每对 60 副牌 =
120 局，deal 聚簇 bootstrap 4000 次；正 = 左方更强）：

| 对（左 − 右） | arena 2-seed（60 副牌） | 已有参考 | 判定 |
|---|---:|---|---|
| lvl4(base680k) − w5_vf1 | **+8.7** [−43.7,+61.4] | pilot 3-seed **+9.1** [−2.6,+20.8] | 点估计一致；不可分 |
| lvl4 − w5_ctrl | −20.3 [−67.4,+29.0] | pilot +4.2 [−7.1,+15.4] | 都跨 0，方向性不一致但都在噪声内 |
| lvl4 − w5_pool4g | −11.6 [−55.5,+29.0] | pilot −4.5 [−20.6,+11.6] | 一致；不可分 |
| lvl4 − w5_scratch | −37.8 [−85.7,+8.7]，p=0.33 | 无 | 点估计最高但未过显著线 |
| lvlbase − lvlsp | −43.7 [−98.1,+8.7] | 复评合并 fit：lvlbase 1428.7±7.1 vs lvlsp 1417.1±7.0 | **方向相反**，均不显著 → 不可分 |
| lvl1 − lvl3 | −110.7 [−172.0,−52.5]，p=0.001 | 复评 lvl1 1146 vs lvl3 1364 | 一致，可分辨 |
| lvl4 − greedy | +113.9 [+61.4,+172.0] | 复评 lvl4 1430 vs greedy 1315 | 一致 |
| base20480 − lvl2 | +0.0 [0,0] | 同一 ckpt | 精确 0（§7） |

对账结论：

1. **w5 五臂对 base680k 全部不可分**，与 `head-to-head-pilot.md`（3 seed ×
   400 副牌）和 `wave5-500k-report.md` 的「没有任何方案显著超过 base680k」
   一致；arena 短赛程给的点估计误差更大（±50–60 Elo/对），但方向没有冲突。
2. **lvlbase vs lvlsp final**：arena 2 seed 偏 lvlsp +44，复评 24,000 局偏
   lvlbase −12；两口径差 ~56 Elo、方向相反、各自 1σ 左右。诚实结论：
   **不可分**，不能引用任何单方向。
3. lvl4 ≈ lvlbase final（seed0 +37.5、seed1 −11.5）与复评合并 fit 的
   「1429.7 vs 1428.7」一致：680k 之后的训练没有可测收益。

## 9. 可传递性与风险

* **可传递性未见反例，但检验力不足**。seed 0 的 378 对诊断中最大 |z|=2.10
  （greedy−lvl1，观测 0.875 vs 期望 0.699），seed 1 的最大 |z|=1.46；没有
  任何一对在两个 seed 上持续矛盾（例如 lvl2−lvl4 的 z 为 +1.32 / −1.24，
  符号翻转）。考虑到随机波动，这是干净的结果。
* 但 `pair_diagnostics` 的 z 只是烟测：每对只有 30 副牌，直接 h2h 的 95% CI
  半宽 ≈50–60 Elo（本次）或 ±20（400 副牌）。**<40 Elo 的循环克制会被噪声
  淹没**；若怀疑某两三个模型有克制关系，用 `tools/head_to_head.py
  --pairs 400` 对该对单独复核，而不是信任 Elo 差值。
* **Elo 的绝对高度不可跨赛程比较**：本竞技场 seed0 的 lvl4 是 1392，复评
  400 局/锚点的合并值是 1430，差 ~38；这是赛程序列/先验/局数不同造成的
  fit 口径差，也再次说明只能内部比较（与审计对相位偏差的结论一致）。
* **单 seed 排名会骗人**（§5）：top-7 名次 seed 间最多移动 18 位。任何
  「X 比 Y 强多少」的结论至少要 2–3 seed 合并 + 逐对 h2h 复核。

## 10. 后续接入训练的建议

1. **周期快照批量定级**：训练脚本每 N 个 update 把 ckpt 丢进
   `runs/<run>/snapshots/`，用 `--glob '.../step*.pt' --every k --last --workers 6`
   定期跑一次竞技场；JSON 直接换成「进度曲线 + 现有锚点定位」，成本
   ~40 s/万局（CPU），不占训练 GPU。
2. **PFSP / 对手池权重**：用竞技场 Elo 作为池成员权重
   （例如 softmax(Elo/τ) 或「只用与当前模型 Elo 差 <200 的快照」），替代
   均匀采样历史快照；目标是给 wave5 类实验一个更有信息的对手分布。
3. **人类定级阶梯**：把人机对局当作「人类 vs 固定参赛者」的观测，增量更新
   Elo（同一 BT 模型）；本竞技场已提供 lvl1..lvl4 + greedy + 1M final 的
   稳定刻度（跨几十 Elo 的梯队是可靠的），适合做人类初始定级与段位边界。
4. **联盟进 top 集团要加钱**：若要分辨 top 集团内部 <40 Elo，把每对
   `cross` 提到 200–400 并跑多 seed；每对 400 副牌的成本在这台机器上约
   1 分钟 CPU（6 worker）或 ~14 s GPU 单进程（见 h2h pilot）。
5. **可选的 margin 通道**：`FitConfig.margin`（分差高斯似然）可能进一步压低
   平台内 SE，可在同一批 `games-out` JSONL 上离线重拟合，不需要重新打牌。

## 11. 复现命令与产物

```bash
# pilot（seed 0；seed 1 同命令改 --seed 1 与 --out/--games-out）
uv run --group train python tools/arena.py \
  --glob 'runs/probe/base_step*.pt' --every 8 --last \
  --glob 'runs/probe_sp/sp_step*.pt' --every 8 --last \
  --entrant lvl1=ckpt:runs/lvl1/agent.pt --entrant lvl2=ckpt:runs/lvl2/agent.pt \
  --entrant lvl3=ckpt:runs/lvl3/agent.pt --entrant lvl4=ckpt:runs/lvl4/agent.pt \
  --entrant lvlbase=ckpt:runs/lvlbase__1__1790313091/agent.pt \
  --entrant lvlsp=ckpt:runs/lvlsp__1__1790314004/agent.pt \
  --entrant w5_ctrl=ckpt:runs/w5_ctrl__1__1790319698/agent.pt \
  --entrant w5_vf1=ckpt:runs/w5_vf1__1__1790320066/agent.pt \
  --entrant w5_pool4g=ckpt:runs/w5_pool4g__1__1790320066/agent.pt \
  --entrant w5_bigbatch=ckpt:runs/w5_bigbatch__1__1790319698/agent.pt \
  --entrant w5_mix_samp=ckpt:runs/w5_mix_samp__1__1790320066/agent.pt \
  --entrant w5_scratch=ckpt:runs/w5_scratch__1__1790319698/agent.pt \
  --cross 60 --games-per-anchor 60 --seed 0 --workers 6 --device cpu \
  --out runs/arena/pilot_cpu.json --games-out runs/arena/pilot_cpu_games \
  --tb runs/arena/pilot_tb \
  --step-map sp20480=1019904 --step-map sp184320=1183744 \
  --step-map sp348160=1347584 --step-map sp512000=1511424 \
  --step-map sp675840=1675264 --step-map sp839680=1839104 \
  --step-map sp983040=1982464 --step-map lvlsp=1999424 --step-map lvl1=12288

uv run pytest -q tests/test_arena.py     # 15 用例
```

产物（`runs/` 已 gitignore）：`pilot_cpu.json` / `pilot_cpu_seed1.json`
（联盟表 + pair_stats + 可传递性）、`pilot_cpu*_games/shard_*.jsonl`（逐局、
开跑前覆盖）、`pilot_tb/`（TensorBoard scalar）、`pilot_cpu*.log`（含
games/s）、`h2h_pooled.txt`、`analysis.txt`。

### 表 3：seed 0 联盟排名（完整；`Δ = seed1 − seed0`）

| # | 参赛者 | 系列 | 训练步 | Elo (seed0) | ±SE | seed1 Elo | Δ | seed1 名次 |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | w5_scratch | w5 | — | 1407.4 | 11.3 | 1378.9 | −28.5 | 19 |
| 2 | base675840 | probe base | 675840 | 1401.2 | 11.3 | 1378.5 | −22.8 | 20 |
| 3 | w5_pool4g | w5 | — | 1399.5 | 11.3 | 1423.9 | +24.4 | 1 |
| 4 | w5_ctrl | w5 | — | 1393.3 | 11.3 | 1381.9 | −11.4 | 17 |
| 5 | **lvl4 (=base680k)** | lvl | — | 1391.8 | 11.3 | 1378.0 | −13.8 | 21 |
| 6 | w5_bigbatch | w5 | — | 1391.6 | 11.3 | 1412.9 | +21.3 | 5 |
| 7 | w5_mix_samp | w5 | — | 1387.0 | 11.3 | 1406.7 | +19.8 | 8 |
| 8 | sp20480 | probe sp | 1019904 | 1386.6 | 11.3 | 1402.6 | +16.0 | 9 |
| 9 | sp184320 | probe sp | 1183744 | 1380.2 | 11.3 | 1379.3 | −0.9 | 18 |
| 10 | sp839680 | probe sp | 1839104 | 1380.2 | 11.3 | 1411.3 | +31.1 | 6 |
| 11 | sp675840 | probe sp | 1675264 | 1375.5 | 11.3 | 1417.9 | +42.5 | 2 |
| 12 | sp983040 | probe sp | 1982464 | 1375.0 | 11.3 | 1414.6 | +39.6 | 3 |
| 13 | **lvlsp** | lvl | 1999424 | 1374.2 | 11.3 | 1414.0 | +39.8 | 4 |
| 14 | sp512000 | probe sp | 1511424 | 1370.9 | 11.3 | 1402.6 | +31.7 | 10 |
| 15 | w5_vf1 | w5 | — | 1366.6 | 11.3 | 1397.4 | +30.8 | 11 |
| 16 | sp348160 | probe sp | 1347584 | 1365.7 | 11.3 | 1408.1 | +42.3 | 7 |
| 17 | base839680 | probe base | 839680 | 1365.5 | 11.3 | 1387.6 | +22.1 | 15 |
| 18 | base512000 | probe base | 512000 | 1361.2 | 11.2 | 1387.6 | +26.4 | 14 |
| 19 | base983040 | probe base | 983040 | 1354.9 | 11.2 | 1388.9 | +33.9 | 13 |
| 20 | **lvlbase (final)** | lvl | 999424 | 1354.3 | 11.2 | 1389.5 | +35.2 | 12 |
| 21 | base348160 | probe base | 348160 | 1341.1 | 11.2 | 1386.7 | +45.6 | 16 |
| 22 | base184320 | probe base | 184320 | 1334.8 | 11.2 | 1377.8 | +43.0 | 22 |
| 23 | lvl3 | lvl | 102400 | 1326.2 | 11.2 | 1352.0 | +25.8 | 23 |
| 24 | greedy | anchor | — | 1315.0 | 0.0 | 1315.0 | +0.0 | 24 |
| 25 | base20480 | probe base | 20480 | 1230.4 | 11.6 | 1219.0 | −11.5 | 25 |
| 26 | **lvl2 (=base20480)** | lvl | 20480 | 1230.4 | 11.6 | 1219.0 | −11.5 | 26 |
| 27 | lvl1 | lvl | 12288 | 1168.7 | 12.1 | 1207.6 | +38.9 | 27 |
| 28 | random | anchor | — | 1000.0 | 0.0 | 1000.0 | +0.0 | 28 |

### 表 4a：lvlbase 系进度（probe base 快照 + lvl2/3/4 + lvlbase final）

| 训练步 | 参赛者 | seed0 | seed1 | 均值 |
|---:|---|---:|---:|---:|
| 20480 | base20480 / lvl2 | 1230.4 | 1219.0 | 1224.7 |
| 102400 | lvl3 | 1326.2 | 1352.0 | 1339.1 |
| 184320 | base184320 | 1334.8 | 1377.8 | 1356.3 |
| 348160 | base348160 | 1341.1 | 1386.7 | 1363.9 |
| 512000 | base512000 | 1361.2 | 1387.6 | 1374.4 |
| 675840 | base675840 | 1401.2 | 1378.5 | 1389.8 |
| 696320 | lvl4 (=base680k) | 1391.8 | 1378.0 | 1384.9 |
| 839680 | base839680 | 1365.5 | 1387.6 | 1376.5 |
| 983040 | base983040 | 1354.9 | 1388.9 | 1371.9 |
| 999424 | lvlbase final | 1354.3 | 1389.5 | 1371.9 |

### 表 4b：lvlsp 系进度（probe sp 快照 + lvlsp final；x = 含 warm start 的累计步）

| 本段步 | 累计步 | 参赛者 | seed0 | seed1 | 均值 |
|---:|---:|---|---:|---:|---:|
| 20480 | 1019904 | sp20480 | 1386.6 | 1402.6 | 1394.6 |
| 184320 | 1183744 | sp184320 | 1380.2 | 1379.3 | 1379.8 |
| 348160 | 1347584 | sp348160 | 1365.7 | 1408.1 | 1386.9 |
| 512000 | 1511424 | sp512000 | 1370.9 | 1402.6 | 1386.8 |
| 675840 | 1675264 | sp675840 | 1375.5 | 1417.9 | 1396.7 |
| 839680 | 1839104 | sp839680 | 1380.2 | 1411.3 | 1395.8 |
| 983040 | 1982464 | sp983040 | 1375.0 | 1414.6 | 1394.8 |
| 999424 | 1998848 | lvlsp | 1374.2 | 1414.0 | 1394.1 |

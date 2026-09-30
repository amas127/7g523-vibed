# T17 revision-3 重标定（出空即撬底）

> **口径标签：本文全部数字为「出空即撬底（rules revision 3）之后」**
> （`rules_id=2e36dbea44893696`、RandomBot=0、ADR-0014 / ADR-0011 / ADR-0012 / ADR-0013），
> 与 revision-2 及更早数字严格隔离、禁止混比。
>
> **状态：已完成（2026-09-26）**。一句话结论：新引擎下重训 revision-3 池，按 13 个
> 去重候选做共享 probit-MLE 筛选后选定 4 级梯级，发布 4000 局 study 与 manifest
> （lvl1–lvl4 = **82.75 / 106.90 / 147.71 / 185.34**，σ≈5.9–6.3），重标 T2 标签与
> `prior.json`（RMSE 42.6、m_eff 68.1、σ(5/10/20)=26.0/22.0/20.5），并补做 T15 遗留的
> placement 常数尺度重标（`c=0.465621`）。
>
> **后续重拟（2026-09-27 w2m/T23、2026-09-29 `search_leafq`）**：raw 档与池已被
> 再次重拟，本文的 lvl1–lvl4 = 82.75/106.90/147.71/185.34 只作 T17 当时值；现行
> manifest 契约（raw 档 + 顶部平台簇 + 搜索 rung）见 `traces/pool10/manifest.json`
> 与 [`human-play.md`](../human-play.md) §3/§6（placement 常数 `c=0.465621` 不变）。

## 0. 训练矩阵（revision 3 / obs v5）

所有 run 都是 `num_envs=8 × num_steps=128 = 1024` steps/update、`num_minibatches=4`、
`learning_rate=2.5e-4`（线性衰减）、`arch=shared`、2 家、seed 见下；观察/规则由工作区
revision 3 引擎决定。500k 训练实际落 499,712 步（488 updates）。

| run | 对手 | steps | snapshots | seed | 用途 |
|---|---|---|---|---|---|
| `runs/t17early2k__1__1790439615` | random | 2,048 | — | 1 | 入门档候选 |
| `runs/t17early4k__1__1790439622` | random | 4,096 | — | 1 | 筛选候选 |
| `runs/t17early8k__1__1790439629` | random | 8,192 | — | 1 | 低档候选（入选 lvl2） |
| `runs/t17pool__1__1790439615` | random | 500k | 16k/32k/65k/131k/262k/499k | 1 | 500k 主池（入选 lvl3=65k，p1_499k 代表） |
| `runs/t17pool__2__1790439615` | random | 500k | — | 2 | 500k 复现臂 p2_500k |
| `runs/t17pool__3__1790439615` | random | 500k | — | 3 | 500k 复现臂 p3_500k |
| `runs/t17self__1__1790439615` | self | 500k | — | 1 | 自博弈过拟合对照 self_500k |
| `runs/t17long__1__1790439615` | random | 2M | 512k/1M/1.5M/2M | 1 | 长训池（入选 lvl4=1M） |

复现入口（训练命令）见各 run 的 `args.json`；梯级路径清单见
`tools/play_ladder.py` 的 `OPPONENTS`。

## 1. 筛选（共享 fit，13 个去重候选 + random 锚）

命令（`runs/t17_screen/command.txt`）：

```bash
.venv/bin/python tools/build_ladder.py \
  --candidate e1_2k=ckpt:runs/t17early2k__1__1790439615/agent.pt \
  --candidate e1_4k=ckpt:runs/t17early4k__1__1790439622/agent.pt \
  --candidate e1_8k=ckpt:runs/t17early8k__1__1790439629/agent.pt \
  --candidate p1_16k=ckpt:runs/t17pool__1__1790439615/snapshots/checkpoint_step16384.pt \
  --candidate p1_32k=ckpt:runs/t17pool__1__1790439615/snapshots/checkpoint_step32768.pt \
  --candidate p1_65k=ckpt:runs/t17pool__1__1790439615/snapshots/checkpoint_step65536.pt \
  --candidate p1_131k=ckpt:runs/t17pool__1__1790439615/snapshots/checkpoint_step131072.pt \
  --candidate p1_262k=ckpt:runs/t17pool__1__1790439615/snapshots/checkpoint_step262144.pt \
  --candidate p1_499k=ckpt:runs/t17pool__1__1790439615/snapshots/checkpoint_step499712.pt \
  --candidate l1_512k=ckpt:runs/t17long__1__1790439615/snapshots/checkpoint_step512000.pt \
  --candidate l1_1M=ckpt:runs/t17long__1__1790439615/snapshots/checkpoint_step1024000.pt \
  --candidate l1_1.5M=ckpt:runs/t17long__1__1790439615/snapshots/checkpoint_step1536000.pt \
  --candidate l1_2M=ckpt:runs/t17long__1__1790439615/snapshots/checkpoint_step1999872.pt \
  --anchor random=random --games-per-anchor 200 --cross 200 --no-traces \
  --games-out runs/t17_screen/games.jsonl --study /tmp/t17_scratch/study \
  --workers 8 --device cuda --seed 0
```

18,200 局（13×200 vs random + C(13,2)×200 cross），全部行带
`rules_id=2e36dbea44893696`。MLE refit（probit、200 次按牌聚簇 bootstrap、
`runs/t17_screen/mle.json`）：

| 排名 | id | μ | σ | 95% CI | 与下一名间隔 |
|---|---|---|---|---|---|
| 1 | l1_1M | 191.61 | 5.26 | [177.90, 204.64] | 0.49 |
| 2 | l1_2M | 191.11 | 5.26 | [177.45, 204.16] | 1.05 |
| 3 | p1_499k | 190.06 | 5.27 | [174.21, 207.59] | 2.37 |
| 4 | l1_1.5M | 187.69 | 5.27 | [173.43, 202.54] | 0.09 |
| 5 | l1_512k | 187.60 | 5.26 | [173.68, 199.60] | 10.04 |
| 6 | p1_262k | 177.56 | 5.24 | [165.59, 189.32] | 9.85 |
| 7 | p1_131k | 167.70 | 5.21 | [154.65, 182.24] | 20.74 |
| 8 | p1_65k | 146.96 | 5.18 | [135.01, 160.27] | 12.52 |
| 9 | p1_32k | 134.44 | 5.17 | [120.24, 147.29] | 18.50 |
| 10 | e1_8k | 115.95 | 5.16 | [102.06, 129.49] | 15.00 |
| 11 | p1_16k | 100.95 | 5.17 | [86.72, 115.84] | 14.13 |
| 12 | e1_4k | 86.81 | 5.18 | [75.18, 99.83] | 10.23 |
| 13 | e1_2k | 76.58 | 5.19 | [61.81, 91.74] | — |

读法（`runs/t17_screen/report.md`）：

- **顶部是平台**：1M/2M/1.5M/512k/499k 五点 μ 187.6–191.6，CI 两两重叠；1M→2M 无改进。
- **非单调**：e1_8k（115.9）> p1_16k（100.9），8k→16k 反向。
- **两种估计器不同尺度且不完全同意**：`build_ladder` 的 OpenSkill PL 打印把 p1_262k
  排在首位（392），而共享 MLE 把它排在第 6（177.6）；发布的排序以 MLE 为准（ADR-0013）。
- **自博弈对照**：self_500k 对发布 lvl4（l1_1M）合并 9 seed 显著更强（+16.62 Elo, p=2.8e-5），对 l1_2M 打平（见 §5）——没有自博弈过拟合证据，反而略优。

## 2. 梯级选择与发布 study

筛选后选定 4 级（相邻 CI 不重叠，见 §3 终表）：**e1_2k → e1_8k → p1_65k → l1_1M**。
`e1_4k`/`p1_16k`/`p1_32k`/`p1_131k`/`p1_262k` 与长训快照留作候选池记录，不登记为梯级。
间距 24.2 / 40.8 / 37.6（终测值），全部小于希望带宽 100–150：revision-3 池整体只有
约 103 MLE 点（旧 revision-2 顶档 ≈209 无法复现），4–5 级梯级不可能都落在 100–150
带宽内；只有 2 级 e1_2k→l1_1M（间距 114.9 筛选值）满足带宽。**owner/筛选阶段接受
实测间距，不为凑 bandwidth 调标签。**

发布命令（工作区 `runs/t17_final/build_ladder.log`；traces 与逐局记录同一次运行产生，
保证 manifest 与游戏记录同源）：

```bash
.venv/bin/python tools/build_ladder.py \
  --candidate lvl1=ckpt:runs/t17early2k__1__1790439615/agent.pt \
  --candidate lvl2=ckpt:runs/t17early8k__1__1790439629/agent.pt \
  --candidate lvl3=ckpt:runs/t17pool__1__1790439615/snapshots/checkpoint_step65536.pt \
  --candidate lvl4=ckpt:runs/t17long__1__1790439615/snapshots/checkpoint_step1024000.pt \
  --anchor random=random --games-per-anchor 400 --cross 400 \
  --study traces/study --games-out runs/t17_mle/games.jsonl \
  --workers 8 --device cuda --seed 0

.venv/bin/python tools/refit_mle.py --games runs/t17_mle/games.jsonl \
  --bootstrap 200 --json --out runs/t17_mle/absolute_table.json \
  --manifest-out traces/study/manifest.json --refit
```

4000 局 = 4×400 vs random + C(4,2)×400 cross，每档 n=1600；`traces/study/` 4000 份
revision-3 trace（`lvl1`–`lvl4`；**random 只作为对手，不产生 subject trace**，因此
`data.n_levels=4`）。manifest 由 `--manifest-out --refit` 发布，
`rules_id=2e36dbea44893696`、`estimator kind=probit-mle`、
`source=runs/t17_mle/games.jsonl`、200 次 bootstrap、draw_margin=15.613；
同一份 manifest 复制到 `traces/pool10/manifest.json`（逐字节相同，
sha256 `c8c491e3…`）。

**发布终表（4000 局共享 probit-MLE，deal 聚簇 200 次 bootstrap CI）：**

| id | 快照 | μ | σ | n | 95% CI |
|---|---|---|---|---|---|
| lvl4 | `runs/t17long__1__1790439615/snapshots/checkpoint_step1024000.pt` | **185.34** | 6.32 | 1600 | [173.65, 200.11] |
| lvl3 | `runs/t17pool__1__1790439615/snapshots/checkpoint_step65536.pt` | **147.71** | 6.14 | 1600 | [136.24, 161.26] |
| lvl2 | `runs/t17early8k__1__1790439629/agent.pt` | **106.90** | 5.95 | 1600 | [96.69, 120.32] |
| lvl1 | `runs/t17early2k__1__1790439615/agent.pt` | **82.75** | 5.88 | 1600 | [70.46, 94.68] |
| random | 脚本锚 | 0.00 | 0.00 | 1600 | [0, 0] |

相邻 CI 两两不重叠（lvl1/2 边界 94.68 vs 96.69；lvl2/3 120.32 vs 136.24；
lvl3/4 161.26 vs 173.65）。manifest `rungs` 按 100–150 间距契约重选为
**lvl1/lvl4**（间距 102.60，`wide_gaps=[]`，只选中 2/5，按 D-2 接受实测间距）。
注意筛选表（18200 局共享 fit）与发布表（4000 局共享 fit）不可逐值混比：删去 9 个
候选后，剩余 4 档的相互约束变少，e1_8k 从 115.9 降到 106.9、l1_1M 从 191.6 降到
185.3；这是同一份发布內部的绝对表，筛选表只用于选级。

### 2.1 与 T15 的差异（同样是 revision 边界）

T15 的 study 是 revision-2、4400 局、含 random-subject（random vs random）trace；
T17 按上面的命令是 revision-3、4000 局、random 仅作对手。因此 `prior.json` 的
`data.n_levels` 从 5 变 4、等级均值不再含 random 行；这是本轮部署口径，不是遗漏。

## 3. 校准（全部 C(5,2)=10 无序对）

`runs/t17_mle/calibrate.py`（由 T15 版本改写）在 400 局/对的发布记录上对
`runs/t17_mle/absolute_table.json` 做校准；`strict`/`half` 为原始胜率（平局计半），
`mle_static = Φ(d/(√2β))`，`mle_tie_aware = ½(Φ((d−ε)/s)+Φ((d+ε)/s))`，
`diff = 实测 − 模型`：

| 对 | strict | half | mle_static | mle_tie_aware | diff(strict−static) | diff(half−tie-aware) |
|---|---|---|---|---|---|---|
| lvl1>random | 0.6300 (252/400) | 0.6863 | 0.7208 | 0.7196 | **−0.0908** | −0.0333 |
| lvl2>random | 0.7400 (296/400) | 0.7788 | 0.7752 | 0.7738 | −0.0352 | +0.0050 |
| lvl3>random | 0.8550 (342/400) | 0.8738 | 0.8519 | 0.8504 | +0.0031 | +0.0234 |
| lvl4>random | 0.8825 (353/400) | 0.9038 | 0.9050 | 0.9037 | −0.0225 | +0.0001 |
| lvl2>lvl1 | 0.4725 (189/400) | 0.5175 | 0.5678 | 0.5674 | **−0.0953** | −0.0499 |
| lvl3>lvl1 | 0.6650 (266/400) | 0.7000 | 0.6770 | 0.6760 | −0.0120 | +0.0240 |
| lvl4>lvl1 | 0.7175 (287/400) | 0.7538 | 0.7659 | 0.7646 | −0.0484 | −0.0108 |
| lvl3>lvl2 | 0.5600 (224/400) | 0.5925 | 0.6135 | 0.6129 | **−0.0535** | −0.0204 |
| lvl4>lvl2 | 0.6475 (259/400) | 0.6850 | 0.7104 | 0.7093 | **−0.0629** | −0.0243 |
| lvl4>lvl3 | 0.6075 (243/400) | 0.6400 | 0.6049 | 0.6043 | +0.0026 | +0.0357 |

- **tie-aware 口径全部通过 0.05 检查**（`over_threshold.half_vs_tie_aware=[]`）。
- strict 口径有 4 对越界（lvl1>random、lvl2>lvl1、lvl3>lvl2、lvl4>lvl2），其中
  **lvl2>lvl1 为 −0.0953**：直接对局几乎五五（strict 0.4725 / half 0.5175），
  但共享 MLE 因 lvl2 对 lvl3/lvl4 的相对表现给它 +24.2 的间距。发布表保留 MLE
  排序；直接对局证据记录于此，不重选梯级（audit §6.5：特定对可优先信直接对局）。
- `lvl2>lvl1` 的严格胜率低于 0.5 是本轮最弱的一对，提醒入门/低档位区分度有限。

**尺度因子**（与 T15 相同的 documented 口径：新标签对旧 online-PL 参考
[96.259, 158.461, 353.258, 413.493] 过原点最小二乘）：

```
c = Σ(new_i·ref_i)/Σ(ref_i²) = 0.4656209850248892
残差 new − c·ref: lvl1 +37.93, lvl2 +33.12, lvl3 −16.78, lvl4 −7.19
反向 c' = 1/c = 2.064915
```

残差很大说明旧 online-PL manifest 的形状与 revision-3 池不一致（和 T15 的 lvl2
残差 +34.5 同类）；`c` 只用于把 placement 固定常数折算到新标签尺度，不用于把
两套数字互转比较。

## 4. 先验与 placement 常数重标

T2 标签（`src/seven523/prior.py`）＝上表 manifest levels；两个标定产物重跑：

| 产物 | sha256 | labels | RMSE(OOF) | m_eff | τ | σ(5/10/20) |
|---|---|---|---|---|---|---|
| `artifacts/human-elo/prior.json` | `abfea8e8…` | `t2_full_data` | 42.6 | 68.07 | 20.9 | 26.0 / 22.0 / 20.5 |
| `artifacts/human-elo/prior_manifest_labels.json` | `3cdf71b3…` | `manifest_levels` | 42.6 | 68.07 | 20.9 | 26.0 / 22.0 / 20.5 |

两图数值相同（T2=manifest），只有 `labels`/`label_source` 元数据不同；数据指纹
`0b531d8f…`，`verify=true`、`failed=0`。等级均值去收缩：lvl1 91.7、lvl2 105.1、
lvl3 134.7、lvl4 191.2（`calibration.level_means`；无 random 行）。

**T15 deferred 项——placement 常数尺度重标（本轮完成）**：以下数均等于原值 × `c`
（`runs/t17_mle/calibration.json` 的 `implied_constants_scaled_by_c`），已写入
`src/seven523/prior.py`（`ANCHOR_CENTER`、`COLD_START_PRIOR`）与
`src/seven523/placement/estimator.py`（`COLD_START_PRIOR`/`RESULT_PRIOR`/`RUNG_PRIOR_SD`）：

| 常数 | 旧（revision-2 尺度） | 新（× c） |
|---|---|---|
| `COLD_START_PRIOR` | (500.0, 300.0) | (232.8104925124446, 139.68629550746675) |
| `RESULT_PRIOR` | (500.0, 200.0) | (232.8104925124446, 93.12419700497784) |
| `ANCHOR_CENTER` | 230.0 | 107.09282655572451 |
| `RUNG_PRIOR_SD` | 30.0 | 13.968629550746675 |

`ANCHOR_SCALE=300` 不在重标清单（保持研究设计常数）；`prior.json` 已随新常数与
新 `ANCHOR_CENTER` 重新生成，`tests/test_placement.py` 的 shipped-prior 断言不再
写死 σ(5)，改为读取产物自身的 `sigma_traj` 并钉住产物冷启动＝模块常数。
`tools/play_ladder.py` 强度列与 `docs/human-play.md` §3 表格同步为
**83 / 107 / 148 / 185（T17）**。

## 5. 自博弈过拟合复核（self_500k vs 随机训练臂）

`self_500k` = `runs/t17self__1__1790439615/agent.pt`（自博弈 500k），每个对手
3 seed × 400 副牌（2400 局，牌聚簇 bootstrap 4000，workers=8/cuda）：

| 对手 | ΔElo(self−opp) | 95% CI | 胜率 | deal-sign p |
|---|---|---|---|---|
| l1_2M（MLE 191.1） | +5.6 | [−6.8, +18.1] | 0.5081 | 0.372 |
| l1_1M（MLE 191.6） | +10.3 | [−2.4, +22.9] | 0.5148 | 0.137 |
| p2_500k | +10.3 | [−2.4, +22.9] | 0.5148 | 0.106 |
| p1_499k | +27.9 | [+13.9, +41.8] | 0.5400 | 3.4e-5 |
| p3_500k | +27.3 | [+14.4, +40.1] | 0.5392 | 1.1e-4 |

**读法**：self_500k 对发布 lvl4（l1_1M）在合并 9 seed 上显著更强，对 l1_2M 打平，显著强于
500k 池的另外两个端点——**没有自博弈过拟合的证据；当前实测最强的其实是未进发布梯级的
self_500k（控制臂）**，如需以它作顶档需重新测量发布。同时注意 MLE 把 p1_499k 与
l1_1M/l1_2M 排成一行（190.1 vs 191.6/191.1），但直接对局 self_500k 显著胜
p1_499k 且与长训臂打平；这一特定对信直接对局（audit §6.5）。12000 局 h2h 行全部
带 `rules_id=2e36dbea44893696`。

**补充复核（独立复验种子）**：在原 3 seed 之外补跑 fresh seed 3–8：
- vs `l1_1M`：seed 3–5 +19.44 [+1.88, +37.01]（p=0.0064）、seed 6–8 +20.15 [+7.17, +33.12]
  （p=0.0031）；**合并 9 seed +16.62 [+9.30, +23.94]（p=2.8e-5，胜率 0.5239）**；
- vs `l1_2M`：合并 6 seed +6.52 [−2.29, +15.33]（p=0.25）→ 打平。

## 6. 墙钟与产物路径

- 筛选：18,200 局 320s（≈17.6 ms/局，workers=8/cuda）+ MLE 75.6s + h2h 260s。
- 发布：4000 局 study build 68s（含 trace 落盘）、refit 22s、两个 prior 34s+32s、
  calibration 84s。
- 产物：`traces/study/manifest.json`（= `traces/pool10/manifest.json`，
  sha256 `c8c491e3…`）、`runs/t17_mle/{games.jsonl,absolute_table.json,calibration.json,calibrate.py}`、
  `runs/t17_final/{build_ladder.log,fit_prior_*.log}`、筛选在 `runs/t17_screen/`。
- 复现 commands 见 §1/§2 与 `runs/t17_screen/command.txt`、`runs/t17_mle/calibrate.py`。

## 7. 口径边界与 legacy（必须遵守）

**可以比**：同一 `rules_id=2e36dbea44893696` + 同一 homoscedastic probit-MLE 口径内的
本轮数字（T2 标签、`prior.json`、`play_ladder` 强度、同一 manifest 的绝对水平），以及
同一发布表的 deal-聚簇 CI。

**不可以比**：

- 全部 revision-2 资产：旧 `traces/study`（4400 局，`rules_id=fbd43015d526ee72`）与
  `traces/pool10` 已移入 `runs/archive/study-legacy-rules2-20260926/` 与
  `runs/archive/pool10-legacy-rules2-20260926/`；旧 T15 值
  （lvl1–lvl4 = 54.85/120.22/191.74/209.01、旧 σ、旧 prior RMSE 72.2/m_eff 23.49、
  σ=43.7/38.0/34.4）一律作废，禁止与新表相减/混排。
- T15 更早的 online-Plackett-Luce manifest（96.26/158.46/353.26/413.49）与 BT-MAP 六值：
  只作 `c` 的历史参考，不是当前标签。
- T17 筛选表（18200 局共享 fit，e1_8k 115.9 等）与发布表（4000 局）：同为 revision 3，
  但样本/候选集合不同，不可逐值混比；`docs/experiments/t15-recalibration.md` 保留为
  revision-2 历史报告，新增交叉引用指向本文。
- 旧规则/旧 obs 的 checkpoint（`t15*`、`base*`、`a1/b1/cp/towers/w5_*` 等）已退役；
  `tools/play_ladder.py` 只登记本轮四个梯级。

**附：全量测试**：`python -m pytest -q` → 621 passed（2026-09-26，工作区）。

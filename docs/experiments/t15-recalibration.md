# T15：「牌型族规则后」重标定报告（7鬼523）

> **口径标签：本文全部数字为「牌型族规则后」**（`rules_id=fbd43015d526ee72`、
> RandomBot=0，ADR-0007/0011/0012/0013），与旧 `tier` 数字严格隔离、禁止混比。
>
> **状态：已完成（2026-09-26；剩真人 M4/D-3 招募与 1 项 placement 常数/测试同步，
> 见 [`../plans.md`](../plans.md) T15/T16）。**
> 一句话结论：新规则（[ADR-0007](../adr/0007-family-comparison.md)）、OpenSkill 评分核心与
> RandomBot=0 单基准（[ADR-0011](../adr/0011-openskill-rating-core.md)/
> [ADR-0012](../adr/0012-single-gauge-and-greedy-removal.md)）、规则身份通道
> （[ADR-0013](../adr/0013-drift-free-rating-channel.md)）落地后，`traces/study` 已用新池重生成、
> manifest 契约值已改由 homoscedastic probit-MLE 绝对表发布、T2 标签与 `prior.json` 已重标、
> `play_ladder` 梯级已重登记。新数字与旧 `tier` 口径**严格隔离、禁止混比**。
>
> **取代关系**：本文的 revision-2 池（`rules_id=fbd43015d526ee72`）已被同日（2026-09-26）
> 完成的 [T17 revision-3 重标定](./t17-recalibration.md) 取代；本文数字只作 revision-2 历史。
>
> 口径：本文所有 μ 均为 **RandomBot=0** 的当前 probit-MLE 绝对表口径
> （ADR-0012/0013；`tools/refit_mle.py`，`beta=100`、`s=√2·β`）。
> 规则身份 `rules_id=fbd43015d526ee72`（`RULES_REVISION=2`）。旧口径数字只作历史对照。

## 1. 新池与梯级（已采纳绝对表）

新池在牌型族比较规则 + 观测 v5 下训练，先用 **order-free anchored probit MLE**
（`tools/refit_mle.py`，ADR-0013 decision 3）筛选。2026-09-26 的采纳决定：manifest 契约值
不再采用在线 OpenSkill Plackett–Luce 联合拟合，而是直接发布同一批 4000 局
（`runs/t15_mle/games.jsonl`，按 study 命令原样重跑、逐行一致）的 **homoscedastic
probit-MLE 绝对表**（`runs/t15_mle/absolute_table.json`；`s=√2·β`，prior σ 只进 MAP 正则）。

| 档位 | 训练 run（spec） | 筛选值（homoscedastic MLE） | 最终 μ（manifest MLE） | σ（manifest MLE） | study trace 局数 |
|---|---|---|---|---|---|
| `random` | 脚本 gauge | 0（锚定） | **0** | 0 | 400 |
| `lvl1` | `runs/t15early4k__1__1790431721/agent.pt` | 55.49（screen2 `e1_4k`） | **54.85** | 6.04 | 1600 |
| `lvl2` | `runs/t15early5k__1__1790433689/agent.pt` | 129.33（screen3 `e1_5k`） | **120.22** | 6.15 | 1200 |
| `lvl3` | `runs/t15pool__1__1790430871/snapshots/checkpoint_step131072.pt` | 177.67（screen2 `s1_131k`） | **191.74** | 6.48 | 800 |
| `lvl4` | `runs/t15long2m__1__1790431758/agent.pt`（2M 步；缺失时用 `snapshots/checkpoint_step1999872.pt`） | 192.80（screen2 `s1_2M`） | **209.01** | 6.58 | 400 |

- 最终 μ 的 95% CI（deal 聚簇 bootstrap 200，`runs/t15_mle/absolute_table.json`）：
  lvl1 [42.58, 68.30]、lvl2 [107.87, 131.92]、lvl3 [178.22, 208.66]、
  lvl4 [197.56, 225.53]；只有 lvl3/lvl4 的 CI 重叠（lvl4 的边缘很弱），
  其余相邻档 CI 分离，`separation=[]`。
- 发布 manifest 的 `estimator`：`kind=probit-mle`、`beta=100.0`、
  `draw_margin=17.371`、`bootstrap=200`、`seed=0`、`games=4000`、
  `source=runs/t15_mle/games.jsonl`；`levels`/`subjects` 与 absolute_table 的
  `levels`/`sigmas` 逐值一致，`subjects[*].games=1600`（MLE 局数，含 cross 局）。
- `rungs` 按 spacing 契约（`count=5, min_spacing=100, max_spacing=150`）从非锚点
  levels 重选，只选出 **lvl1（54.85）/ lvl3（191.74）** 两级：lvl1→lvl3 间距
  136.89（未超 `max_spacing`）、lvl4 距 lvl3 仅 17.27（tail_gap），`ok=False`
  是候选不足的显式短报，不是错误；按实测间距接受（同早期 D-2 决定）。
- 筛选产物已按同一 homoscedastic 口径重算：`runs/t15_screen/mle.json`（9000 局）、
  `runs/t15_screen2/mle.json`（18200 局）、`runs/t15_screen3/mle.json`（5600 局）；
  各次筛选是不同候选排布/局数的 probit MLE（beta=100、draw margin 约 14.5–15.3、
  `--anchors random=0`、deal 聚簇 bootstrap 200），**彼此之间也只作粗比**。
  早期 plug-in 尺度（`s=√(2β²+σᵢ²+σⱼ²)`）下的筛选值（lvl1 65.6 / lvl2 227.8 /
  lvl3 330.1 / lvl4 363.4）已作废，只作历史。
- manifest 身份：`rules_id=fbd43015d526ee72`、`rules.revision=2`、
  `created_at=2026-09-26T10:46:52`、`frozen_at=2026-09-26T10:47:01`（发布路径保留原
  时间戳与无关元数据）；`traces/pool10/manifest.json` 与
  `traces/study/manifest.json` 逐字节相同（`7g523-elo` 默认入口）。

复现命令（发布 → 同步 pool10；study trace 生成见 `tools/measure_trace_signal.py generate`，
4000 局重跑命令见 `runs/t15_mle/REPORT.md` §1）：

```bash
.venv/bin/python tools/refit_mle.py \
    --games runs/t15_mle/games.jsonl --bootstrap 200 --seed 0 --json \
    --manifest-out traces/study/manifest.json --refit
cp traces/study/manifest.json traces/pool10/manifest.json

# 筛选表（新口径，重复三次）
.venv/bin/python tools/refit_mle.py \
    --games runs/t15_screen2/games.jsonl --bootstrap 200 \
    --out runs/t15_screen2/mle.json --json
```

## 2. 先验重标定（T13-M1 复跑）

`tools/fit_trace_prior.py` 在 4400 局新 study 上重跑（`data.failed=0`、`verify=true`）：

| 产物 | sha256 | 标签 | a | b | LOLO RMSE | s | m_eff | σ(5/10/20) |
|---|---|---|---|---|---|---|---|---|
| `artifacts/human-elo/prior.json`（默认 T2） | `35f7227b…` | `t2_full_data` | −97.7 | 1.943 | **72.2** | 71.7 | **23.49** | **43.7 / 38.0 / 34.4** |
| `artifacts/human-elo/prior_manifest_labels.json`（对照） | `2389f6f9…` | `manifest_levels` | −97.7 | 1.943 | 72.2 | 71.7 | 23.49 | 43.7 / 38.0 / 34.4 |

- **T2 标签已重导**：`T2_LABELS`（`src/seven523/prior.py`）现为 random 0 /
  lvl1 54.847 / lvl2 120.215 / lvl3 191.740 / lvl4 209.008，即新 manifest levels；
  更早的 online PL 五值（random 0、lvl1 96.259、lvl2 158.461、lvl3 353.258、
  lvl4 413.493）与 BT-MAP 六值均已作废。因此两套标签图数值相同，两个 JSON 仅
  `labels`/`label_source` 元数据不同。
- 数据指纹 `24f28734…`（manifest 字节变了，trace 本体未变）；冷启动仍为
  `Prior(500, 300)`；`tau_between=35.0`、`sigma_within=62.4`；等级均值去收缩：
  random −5.8、lvl1 77.5、lvl2 112.4、lvl3 170.5、lvl4 221.3
  （`calibration.level_means`）。
- `m_eff` 从 6.4 升到 23.5、RMSE 从 137.7 降到 72.2 主要是标签尺度压缩后的同一条
  回归线（a/b 同步变化），不能跨尺度解读为「信号变强/变弱」。
- 复现：

```bash
.venv/bin/python tools/fit_trace_prior.py fit --study traces/study \
    --out artifacts/human-elo/prior.json
.venv/bin/python tools/fit_trace_prior.py fit --labels manifest --study traces/study \
    --out artifacts/human-elo/prior_manifest_labels.json
```

## 3. 可比性边界（必须遵守）

**可以比**：同一 `rules_id=fbd43015d526ee72` + 同一 homoscedastic probit-MLE 口径下的
新数字，例如本次 lvl1–lvl4 契约值、新 `prior.json`、`play_ladder` 强度列，
以及同一发布 manifest 内的绝对水平。

**不可以比（与旧数字严格隔离）**：

- 本目录 `2026-09-25` 之前的全部 Elo / 胜率 / manifest 契约值 / arena 排名 / 轨迹先验：
  旧 `tier` 规则、旧 BT-MAP 估计器、旧 RandomBot=1000 刻度（`elo/se` 键，已被新门禁
  拒绝）。坐标差 1000 且规则与估计器都换了，逐值相减没有意义。
- 同一 `rules_id` 下但不同估计器的数字也不能混：online OpenSkill PL manifest
  （lvl1 96.259 / lvl2 158.461 / lvl3 353.258 / lvl4 413.493）已被本报告 §1 的
  homoscedastic probit-MLE 表取代；早期 plug-in 尺度 MLE（lvl1 73.24 / lvl2 209.04 /
  lvl3 355.59 / lvl4 391.49，以及筛选表 65.6/227.8/330.1/363.4）同样作废。
  跨独立 fit 的绝对分依旧不可作结论（ADR-0013 §6）。
- 旧先验数字（a=−890.92、b=1.7122、σ=80.6/69.6/64.6、RMSE 133.9、m_eff 6.7 以及
  T15 online PL 版 a=−174.3、b=2.010、RMSE 137.7）同理，只说明换了口径后拟合仍可用，
  不能拿 72.2 vs 137.7 vs 133.9 讨论精度涨跌。
- 旧 lvl1–lvl4 模型本身已删除（旧 `manifest` 备份见 §4），不存在可回测的旧 ckpt。
- 新的 4400 局 study 里 `opponent` 只有 random gauge，不再有 greedy/候选互相对局；
  轨迹先验的对手强度特征（`opponent_elo_ref`）现在恒为 0。

## 4. 旧资产归档与 lvl4=430 结论

- 旧 1200 局 study + manifest：`runs/archive/study-legacy-20260926/`。
- 旧 pool10 manifest：`runs/archive/pool10-legacy-20260926/manifest.json`。
- 旧冻结 manifest 备份：`runs/archive/manifest-frozen-2026-09-25.json`。
- 旧 online PL manifest 的 `levels` 已记录在本报告 §1/§3 的对照文字中；它不是当前契约。
- **旧 lvl4=430 不可达**：旧 manifest 顶档 lvl4 = 1429.7 ± 7.1（旧 1500/300 刻度），
  换到 RandomBot=0 即 **≈429.7（约 430）**；T15 新池最高档 lvl4 的当前契约值是
  **209.01**（筛选阶段 `s1_2M` 约 192.80）。430 与 209 属于不同估计器/尺度
  （旧 BT-MAP 刻度 vs homoscedastic probit-MLE），不能直接相减解读；唯一成立的说法是
  当前池（含训到 2M 步的 `t15long2m`）**没有任何模型被重测到旧 430 档位**，新梯级表
  以 209.01 为顶，旧 `lvl4=430` 既没有对应模型、也无法直接换算到新口径，不得作为
  契约缺口或差距来解读。

## 5. 清单

| 项 | 路径 |
|---|---|
| 新 study（trace 本体，gitignore） | `traces/study/{random,lvl1..lvl4}/*.json`（4400 局） |
| manifest 契约（新口径） | `traces/study/manifest.json`（= `traces/pool10/manifest.json`，probit-MLE 绝对表） |
| 采纳用 4000 局与表 | `runs/t15_mle/games.jsonl`、`absolute_table.json`（中性）/ `absolute_table_with_pl_prior.json`（对照） |
| 校准产物与脚本 | `runs/t15_mle/calibration.json`（全 10 对 + `over_threshold`）、`calibrate.py`、`pl_manifest_baseline.json`（旧 online PL manifest，供 c 复现） |
| 筛选 MLE（新口径） | `runs/t15_screen{,2,3}/mle.json` + `games.jsonl` |
| 重标先验 | `artifacts/human-elo/prior.json`、`prior_manifest_labels.json` |
| 可玩梯级 | `tools/play_ladder.py`（`lvl1`–`lvl4`；强度列 55/120/192/209） |
| 计划/口径 | [`../plans.md`](../plans.md) T15/T16、[`README.md`](./README.md) 顶部口径警告、[`../human-play.md`](../human-play.md) §3 |

## 6. 附：homoscedastic MLE 绝对表与采纳（取代旧「尺度问题（待决）」）

### 6.1 采纳表（beta=100、`s=√2·β=141.421`、draw margin ε=17.371）

`runs/t15_mle/absolute_table.json`（中性先验，默认 `Prior(500, 200)`，`random` 钉 0）
已发布为 `traces/study/manifest.json` 与 `traces/pool10/manifest.json` 的 `levels`/`subjects`：

| id | mu | sigma | 95% CI（deal 聚簇 bootstrap 200） | n |
|---|---|---|---|---|
| lvl4 | 209.008 | 6.582 | [197.563, 225.526] | 1600 |
| lvl3 | 191.740 | 6.479 | [178.222, 208.664] | 1600 |
| lvl2 | 120.215 | 6.154 | [107.866, 131.924] | 1600 |
| lvl1 | 54.847 | 6.039 | [42.579, 68.302] | 1600 |
| random | 0.000 | 0.000 | [0.000, 0.000] | 1600 |

`converged=true`、`margin_identified=true`、`separation=[]`、`games=4000`。

PL-prior 变体（`runs/t15_mle/absolute_table_with_pl_prior.json`，
`--prior-manifest` 旧 manifest；**只作对照，不是发布值**）：lvl1 60.327 /
lvl2 126.248 / lvl3 199.795 / lvl4 217.662（ε=17.524、`separation=['lvl4']`）。
两表差异只剩 MAP shrinkage（5.5/6.0/8.1/8.7），尺度完全一致——这是 homoscedastic
`s=√2·β` 修复（option a）的目的，也是旧 plug-in 尺度（随 prior σ 把 lvl2 在
209.0 与 130.6 之间搬动）被废弃的原因。

### 6.2 全 10 对校准（4000 局，每个无序对 400 局；训练调度共 C(5,2)=10 对）

`strict` = 严格胜率（不含平局）；`half` = 平局记半；`tie-aware` = 有序 probit 期望
（ε=17.371，`P(win)+0.5·P(tie)`）；`diff` = tie-aware − half。

| pair | strict | half | MLE Φ(d/s) | tie-aware | diff |
|---|---|---|---|---|---|
| lvl1>random | 0.6125 (245/400) | 0.6625 | 0.6509 | 0.6499 | −0.0126 |
| lvl2>random | 0.8125 (325/400) | 0.8462 | 0.8024 | 0.8006 | −0.0457 |
| lvl3>random | 0.8675 (347/400) | 0.8912 | 0.9124 | 0.9108 | +0.0195 |
| lvl4>random | 0.8750 (350/400) | 0.8975 | 0.9303 | 0.9288 | +0.0313 |
| lvl2>lvl1 | 0.6175 (247/400) | 0.6725 | 0.6780 | 0.6768 | +0.0043 |
| lvl3>lvl1 | 0.8225 (329/400) | 0.8512 | 0.8335 | 0.8317 | −0.0196 |
| lvl4>lvl1 | 0.8300 (332/400) | 0.8600 | 0.8622 | 0.8604 | +0.0004 |
| lvl3>lvl2 | 0.6800 (272/400) | 0.7113 | 0.6935 | 0.6922 | −0.0191 |
| lvl4>lvl2 | 0.7175 (287/400) | 0.7550 | 0.7350 | 0.7334 | −0.0216 |
| lvl4>lvl3 | 0.5325 (213/400) | 0.5675 | 0.5486 | 0.5482 | −0.0193 |

**0.05 检查口径须分开表述**：tie-aware 半记分比较在全部 10 对上 |diff| ≤ 0.046
（最大 +0.031 lvl4>random、−0.046 lvl2>random），达标；但 **strict 严格胜率**
有两对超过 0.05：lvl2>lvl1（0.6175 vs 静态 0.6780，差 −0.0605）与 lvl4>random
（0.8750 vs 0.9303，差 −0.0553），因为静态 `Φ(d/s)` 不计拟合的平局裕度 ε。该偏差
已由 `runs/t15_mle/calibration.json` 的 `over_threshold` 字段显式记录，不得把
0.05 检查默认读作在 strict 口径上全部通过。lvl4>lvl3 接近五五开，说明顶档 lvl4
相对 lvl3 的边缘很弱。旧 online PL manifest 对同五对的期望（如 lvl3>lvl2
0.9015、lvl4>random 0.9976）严重过自信，已被本表取代。

### 6.3 被取代的内容与未执行项

- 旧 §6 的 plug-in 尺度表（lvl1 73.24 / lvl2 209.04 / lvl3 355.59 / lvl4 391.49 与
  `--prior-manifest` 的 60.46/130.58/208.57/227.68）、「(a)/(b)/(c) 待决」段落全部
  superseded：已采纳 **(a) homoscedastic `s=√2·β`**，ADR-0013 §3 已记录，无 `scale_id`
  需要的第二约定。
- 旧 §1 的 online PL manifest 值（96.26/158.46/353.26/413.49）与旧筛选值
  （65.6/227.8/330.1/363.4）均 superseded。
- 比例因子 `c=0.540631`（`calibration.json`，把旧的 online PL manifest 线性折算到
  MLE 尺度；lvl2 残差 +34.5，说明旧 manifest 形状本身不一致）只作历史
  （placement 常数重标仍以它为准）；`calibration.json` 现覆盖全部 10 个无序对，并以
  `over_threshold` 显式记录 0.05 检查的越界对（strict 口径 lvl2>lvl1、lvl4>random；
  tie-aware 口径无越界）。
- **已排期（下一阶段，归 placement workflow；本轮不触碰 `tests/test_placement.py`）**：
  由 c 隐含的 placement 固定常数重标——`COLD_START_PRIOR` (500, 300) →
  **(270.32, 162.19)**、`RESULT_PRIOR` (500, 200) → **(270.32, 108.13)**、
  `ANCHOR_CENTER` 230.0 → **124.35**、`RUNG_PRIOR_SD` 30.0 → **16.22**
  （`src/seven523/placement/estimator.py` 与 `src/seven523/prior.py`）；连同
  `prior.json` 重新生成与 `tests/test_placement.py` shipped-prior 常数同步
  （`prior_for_session(1200.0, 5).sigma` 81.3451 → **43.6638**，最好直接从
  `prior.json` 的 `sigma_traj` 读取；`cold_start` 500/300 → 270.32/162.19）。
  该同步完成前 T16 的「全套现有测试保持全绿」验收不成立（当前 597 passed /
  1 failed，唯一失败即 `test_shipped_prior_artifact_loads_and_predicts`）。

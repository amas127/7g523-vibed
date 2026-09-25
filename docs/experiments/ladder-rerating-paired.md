# 冻结梯级换座配对复测报告（lvl1–lvl4）

> 资产状态（2026-09-25）：本报告引用的模型 checkpoint 已删除，旧路径不再可用；数值为牌型族规则变更前口径。

> **状态：复测与量化完成；复测确认旧 manifest 存在相位偏差，建议 refit，但不要用单
> seed 直接覆盖。** 本轮用候选内换座配对协议（`src/seven523/ladder.py:93`）以
> 400 局/锚点 × 5 个 seed 复测 `runs/lvl1..lvl4/agent.pt`，并量化旧冻结值与新值的差
> 及梯级相对结构。冻结数据（`traces/study/manifest.json`，mtime 2026-09-25 01:44:16）
> 与现有报告在本轮中**未做任何改动**。
>
> 结论速览：**lvl4 被高估约 60 Elo（5/5 seed 为负）**、lvl1 被低估约 27 Elo；
> 梯级排序在 5/5 seed 保持，但「4/4 rungs、间距 113–130」的契约在复测点估计上不再
> 成立（实测间距 73/144/66），`select_rungs` 只选得出 2/4。

## 1. 方法

### 1.1 协议与设置

| 项 | 旧 manifest（冻结） | 本次复测 |
|---|---|---|
| 调度 | 旧 `plan_games`：座位相位由候选下标奇偶决定（`elo-reliability-audit.md` §5.4） | 候选内换座配对：每副牌对每个候选打两局、座位互换（`ladder.py:93`，`games_per_anchor`/`cross` 强制偶数） |
| games_per_anchor | 100（单座位） | 400（=200 副牌 × 双座位） |
| 每候选局数 | 200/seed | 800/seed；5 seed 合并 4000 |
| seed | 0 | 0、1（任务要求）+ 2、3、4（量化跨 seed 漂移） |
| cross | 0 | 0 |
| 锚点 | random=1000 / greedy=1315 钉死 | 同 |
| SE | 对角 Fisher | 联合 Hessian（`src/seven523/elo.py:321`）+ 按副牌 cluster bootstrap |

候选：`lvl1..lvl4`（冻结 subjects）＋两个 1M 参考点放进同一 fit（仅参考、不写入任何
manifest）：

- `lvlbase_final` = `runs/lvlbase__1__1790313091/agent.pt`（`ladder-report.md` §2.1 的
  final，旧测 1384.4 ± 29.9）
- `lvlsp_final` = `runs/lvlsp__1__1790314004/agent.pt`（§2.3 的 spfinal，旧测 1374.2 ± 29.6）

### 1.2 命令与实现校验

```bash
.venv/bin/python runs/lvl_rerate/rerate.py --seed S --games-per-anchor 400 \
    --device cuda --out runs/lvl_rerate/seedS.json --with-games   # S=0..4
.venv/bin/python runs/lvl_rerate/analyze.py 4000
```

`rerate.py` 直接调用 `seven523.ladder.build_ladder(..., out=None)`（不用 CLI，避免与
工作区 `--games-out` 改动竞态），并 tee `play_games` 保存逐局战绩；`analyze.py` 拼接
5 个 seed 的 24000 局做 combined fit，并按 deal（每个 32-bit 发牌 seed 的 12 局为一簇，
共 2000 簇）做 4000 次 cluster bootstrap（审计 §6.3 列的待办）。5 个 seed 使用的
`ladder.py`/`elo.py` md5 记录在 `runs/lvl_rerate/seed*.json` 的 `LADDERS` 字段：
ladder `b5bd6e5dc99858b2c5f8fc5a0f13c2b6`、elo `b942811dd52d8e926c4c11a3777865c3`，
**5/5 一致**。

**实现校验**：seed 0 的 lvl4 = 1443.3 ± 16.2，与审计 §6.1 的独立复跑
（`runs/elo_rel_seatpaired_verify.txt` 中的 `base680k`）**逐位一致**，说明协议实现与
当时无差异，seed 间差异是发牌抽样而非链路非确定性。

## 2. 结果

### 2.1 新旧对照（Elo ± SE）

旧值来自 `traces/study/manifest.json` 的 `levels` 与 `ladder-report.md` §2.4（两处一致）。

| 等级 | 旧值 ± SE | seed0 | seed1 | seed2 | seed3 | seed4 | 5-seed 均值 ± sd | 合并 fit ± SE | 合并 95% CI（deal bootstrap） | Δ(合并−旧) |
|---|---|---|---|---|---|---|---|---|---|---|
| lvl1 | 1119.9 ± 27.0 | 1156.4 | 1143.8 | 1142.7 | 1137.4 | 1157.5 | 1147.6 ± 8.9 | **1146.3 ± 6.1** | [1135.0, 1157.1] | **+26.3** |
| lvl2 | 1232.9 ± 27.2 | 1239.4 | 1212.5 | 1228.1 | 1196.6 | 1224.9 | 1220.3 ± 16.3 | **1219.2 ± 6.1** | [1208.0, 1230.3] | −13.7 |
| lvl3 | 1359.3 ± 29.2 | 1352.6 | 1364.5 | 1378.6 | 1356.4 | 1369.6 | 1364.3 ± 10.4 | **1363.6 ± 6.6** | [1351.5, 1375.6] | +4.4 |
| lvl4 | 1489.4 ± 34.3 | 1443.3 | 1388.4 | 1458.7 | 1417.0 | 1447.1 | 1430.9 ± 28.2 | **1429.7 ± 7.1** | [1416.4, 1443.5] | **−59.7** |
| lvlbase_final | (1384.4 ± 29.9)¹ | 1419.2 | 1417.7 | 1429.9 | 1424.9 | 1454.8 | 1429.3 ± 15.0 | 1428.7 ± 7.1 | [1415.5, 1441.6] | (+44.3)¹ |
| lvlsp_final | (1374.2 ± 29.6)¹ | 1402.5 | 1401.1 | 1416.3 | 1419.2 | 1450.1 | 1417.8 ± 19.8 | 1417.1 ± 7.0 | [1404.4, 1430.5] | (+42.9)¹ |

¹ 参考点不是 manifest 契约；括号内旧值为 `ladder-report.md` §2.1/§2.3 的同 ckpt 读数。

单 seed 全部 converged（5 sweeps），每 seed 4800 局、每候选 800 局/seed；两个锚点每 seed 各 2400 局（se=0，钉死），5 seed 合并各 12000 局。

### 2.2 两 seed 一致性 / 跨 seed 漂移

- seed0 vs seed1：lvl4 **1443.3 vs 1388.4（Δ55.0）**——两侧单 seed SE 为 16.2/15.2，
  Δ 的 SE ≈ 22，约 2.5σ；lvl2 差 26.9（约 1.4σ）；lvl1/lvl3 在 1σ 内。
- **排序没有翻转**：lvl1<lvl2<lvl3<lvl4 在 seed 0/1 及追加的 2/3/4 共 5/5 成立。
- 跨 seed 散度：lvl1 sd 8.9、lvl2 16.3、lvl3 10.4、lvl4 **28.2**（单 seed Hessian SE
  约 13.5–16.5）。lvl4 的 seed 散度约为其单 seed SE 的 1.8×，但 5 个 seed 不足以判定
  真实过度离散（sd 的 95% 区间约 17–81）。合并数据的 deal bootstrap sd（5.6–6.7）与
  联合 Hessian SE（6.1–7.1）同量级，说明合并后无明显额外副牌相关。
- **多 seed 合并读数（推荐）**：lvl1 1146.3 ± 6.1、lvl2 1219.2 ± 6.1、
  lvl3 1363.6 ± 6.6、lvl4 1429.7 ± 7.1。若改用「seed 均值 ± sd/√5」的保守口径，
  lvl4 为 1430.9 ± 12.6，中心一致。

### 2.3 相位偏差量化（对审计 §5.4 的预测）

| 等级 | 逐 seed Δ(新−旧) | 均值 Δ | 合并 Δ | 审计预测 |
|---|---|---|---|---|
| lvl1 | +36.5 / +23.8 / +22.8 / +17.5 / +37.6 | +27.6 | +26.3 | — |
| lvl2 | +6.5 / −20.4 / −4.8 / −36.3 / −8.0 | −12.6 | −13.7 | — |
| lvl3 | −6.6 / +5.2 / +19.3 / −2.9 / +10.3 | +5.1 | +4.4 | — |
| lvl4 | −46.1 / −101.0 / −30.7 / −72.4 / −42.3 | −58.5 | **−59.7** | 「可能被高估 ~40」(§5.4) |

- **lvl4 高估确认**：5/5 seed 低于旧值；旧 1489.4 高于合并 95% CI 上界 1443.5，
  也高于任意单 seed 的点估计（最高 seed2 1458.7）；4/5 单 seed 的 95% CI 排除旧值，
  seed2 距旧值 1.9σ。审计猜的 ~40 方向正确、量级偏小（实测约 −60）。
- **lvl1 旧值偏低**：5/5 seed 高于旧值（+17.5…+37.6）。旧相位偏移并不统一，
  并非整体平移。
- lvl2/lvl3 的位移（−13.7/+4.4）在旧 SE（27–29）内，可视为不变。
- 梯级跨距从旧 369.5 缩到合并 283.4（−86，−23%）；参考点也印证平台：旧报告
  「lvl4 1489 vs 1M final 1384」的 105 Elo 落差是相位产物——合并后
  lvl4 1429.7 ≈ `lvlbase_final` 1428.7 ≈ `lvlsp_final` 1417.1，与审计 §5.1
  「平台顶部更接近 1440–1460」一致。

### 2.4 梯级相对结构

- **排序**：5/5 seed 与合并 fit 全部保持 lvl1<lvl2<lvl3<lvl4。
- **间距**（Elo）：

| 口径 | lvl1→lvl2 | lvl2→lvl3 | lvl3→lvl4 |
|---|---|---|---|
| 旧 manifest（契约 100–150 内） | 113.0 | 126.4 | 130.1 |
| seed0 | 83.0 | 113.2 | 90.7 |
| seed1 | 68.8 | 151.9 | 23.9 |
| seed2 | 85.4 | 150.5 | 80.2 |
| seed3 | 59.2 | 159.8 | 60.7 |
| seed4 | 67.4 | 144.7 | 77.5 |
| 合并 | **72.9** | **144.4** | **66.1** |

  合并口径下 lvl1→lvl2（72.9±8.6）与 lvl3→lvl4（66.1±9.7）都显著为正，但都低于
  100 的最小间距；lvl2→lvl3 达 144.4，接近上限。单 seed 下 lvl3→lvl4 可低至 23.9
  （seed1，Δ 的 SE ≈ 21，1.1σ），旧报告「每级差 3.5–4.5 SE」（`ladder-report.md:122`）
  的叙述已不成立。
- **`select_rungs` 结果**：以 count=4、100≤spacing≤150 在复测点估计上运行，
  **5/5 seed 与合并 fit 都只选得 lvl1、lvl3 两个 rung**（lvl2 距 lvl1 不足 100，
  lvl4 距 lvl3 不足 100），且存在 lvl1→lvl3 的 wide gap（196–236 > 150），
  `ok=False`；含/不含两个参考点的选择结果相同。旧 manifest 是 `rungs (4/4)`、无 wide
  gap（`ladder-report.md:96`）。
- **口径差异（必读）**：旧 manifest 的契约是 `estimator.games_per_anchor=100`
  （每候选 200 局、SE≈27、座位相位未修）；本次是 400 局/锚点配对（每候选 800 局/seed）。
  局数只压 SE，**点估计本身因相位修复而改变**，因此 spacing 结论翻转来自相位修正，
  而不是局数。rung 选择规则只看点估计排序，与局数无关；「4/4」是旧那组（带相位偏移的）
  点估计的偶然属性。

## 3. 对 D1 标签的影响

- 标签链路：`level_elo_ref` 列定义在 `tools/measure_trace_signal.py:105`，取值在
  `tools/measure_trace_signal.py:439` 从 manifest 的 `levels` 读出；等级内 y 是常数
  （`trace-signal-report.md:83`）。
- 若按合并值 refit，每个等级的目标整体平移：lvl1 **+26.3**、lvl2 **−13.7**、
  lvl3 **+4.4**、lvl4 **−59.7**。梯级跨距 369.5→283.4（×0.77）：同样的 S1 特征差
  会映射到约 0.77× 的 Elo 差（回归斜率收缩 ≈23%，或标定面移动，取决于拟合方式），
  顶层绝对 Elo 预测下移约 60。
- `trace-signal-report.md:100` 的「间距 113–130」与 §3.2「相邻级距 100–150 才是工作
  区间」在复测值下不再成立（实际 73/144/66）；该报告若要沿用 calibration 数字需重跑。
- **traces 无需重采**：策略没变，变的只是参考标签；已落盘的 `traces/study/*.json`
  不含 Elo 字段。但已发布的标定数值、以及任何从 manifest levels 派生的 D3 先验/热启动
  （尤其顶层 −60）需要随 refit 更新。refit 前不要给 D3 供数。

## 4. refit 建议

### 4.1 是否值得

**值得**：冻结值带一次性相位偏移（lvl4 约 −60、lvl1 约 +26），是当前 D1 标签的系统
误差；且旧值恰好凑成 113–130 的均匀间距，具有误导性。但 **不要用本报告任一单 seed
直接覆盖**：单 seed 的 lvl4 在 1388–1459 间漂移，单 seed refit 只是把另一种抽样偏差
冻进 manifest。

### 4.2 建议步骤（本轮均未执行）

1. **多 seed 固化测量**：按配对协议跑 ≥3（建议 5）个 seed，`games_per_anchor=400`；
   用刚落地的 `--games-out`（`tools/build_ladder.py:121,187`；库入口 `ladder.py:271,305`）
   或等价方式保存逐局战绩。
2. **合并 fit + 不确定度**：拼接各 seed 的逐局结果做 combined fit（`runs/lvl_rerate/analyze.py`
   可直接复用）；把 per-seed 散度与 deal-bootstrap 95% CI 写进 manifest 的 `estimator`
   （如 `seeds`、`deal_clusters`、`ci95`），SE 取 max(联合 Hessian, deal bootstrap,
   seed spread/√k)。
3. **先写 scratch study**：用 `study.merge_manifest(..., refit=True)`
   （`src/seven523/study.py:51,72`）把新 levels 写进 `runs/` 下的 scratch study
   （如 `runs/lvl_rerate/manifest_candidate.json`），与冻结 manifest diff
   `levels`/`rungs`；**不要写 `traces/study/`**。
4. **批准与同步**：`traces/study/manifest.json` 是 study 契约、默认冻结
   （`study.py:8–11,65`），正式 refit 需 **D1/study owner 批准**；同一变更内更新
   `ladder-report.md` §2.4（`:88–97`）与 `trace-signal-report.md:100` 的间距叙述，
   旧值留在 git 历史，不覆盖/重写任何 trace。
5. **spacing 另案决定**：refit 不会恢复 4/4 rungs。要么接受实测间距（D1 需要的是
   单调等级而不是均匀 100–150，见 `trace-signal-report.md:83`），要么补测/重选快照
   填缝（例如 lvl2–lvl3 间补一个 ~1290 的级）；不要为满足 spacing 调标签。

### 4.3 限制

- deal-cluster bootstrap 已保留配对与候选间同牌 CRN，但仍假设 BT 可加性；5 个 seed
  下 seed-level 方差本身估计不精确。
- 与旧值对比时，旧 SE 不含相位项（审计 §5.4），所以「Δ < 旧 SE」不能当作两者一致。
- `lvlbase_final`/`lvlsp_final` 是同 fit 参考点，不是冻结 subjects；本轮未写入任何
  manifest。

### 4.4 refit 执行记录（2026-09-25）

> owner 按 `docs/plans.md` §4 D-1/D-2 批准后执行；§1–§3 的「manifest 未改动」陈述只
> 反映复测当时状态。执行未改代码、未跑新训练/评测。

- **数据复核**：重跑 `runs/lvl_rerate/analyze.py 4000`（纯 CPU，bootstrap ~166 s）
  逐位复现 `analysis.json`（除 `timing` 字段）——24000 局、2000 deal clusters、
  seeds 0–4、400 局/锚点换座配对；合并读数 lvl1 1146.27±6.07、lvl2 1219.20±6.11、
  lvl3 1363.64±6.63、lvl4 1429.74±7.13（95% CI 见 `estimator.ci95`）。
- **备份**：`runs/archive/manifest-frozen-2026-09-25.json`，sha256
  `6d78c5b1bd22d48ec16f387ae01338d853ec4b13b7c779288f4fc84060eb5a04`（md5 与 §5 记录的
  `862b7ae7…` 一致）。
- **candidate**：`runs/lvl_rerate/make_candidate.py` 用 `study.merge_manifest(..., refit=True)`
  生成 `runs/lvl_rerate/manifest_candidate.json`，diff 存 `runs/lvl_rerate/manifest_diff.txt`；
  随后按字节复制覆盖 `traces/study/manifest.json`（sha256
  `0ee8152b70fcf6d933818c6002ea3b044a925a8816e9df18b0807afeccec535f`），`traces/` 下其余
  1200 个文件 mtime 未变；锚点 random=1000/greedy=1315、`num_players=2`、`created_at` 未动。

**old → new**：

| 字段 | 旧（冻结） | 新（refit） |
|---|---|---|
| `levels.lvl1` | 1119.92 | **1146.27**（Δ +26.35） |
| `levels.lvl2` | 1232.90 | **1219.20**（Δ −13.70） |
| `levels.lvl3` | 1359.26 | **1363.64**（Δ +4.38） |
| `levels.lvl4` | 1489.42 | **1429.74**（Δ −59.68） |
| `levels.random` / `levels.greedy` | 1000 / 1315 | 不变 |
| `subjects.lvl{1..4}.se` | 26.96 / 27.18 / 29.23 / 34.27 | 6.07 / 6.11 / 6.63 / 7.13 |
| `subjects.lvl{1..4}.games` | 200 | 4000（5 seed × 800） |
| `estimator.games_per_anchor` | 100 | 400 |
| `estimator.seeds` / `deal_clusters` / `bootstrap_n` | （无） | [0,1,2,3,4] / 2000 / 4000 |
| `estimator.ci95` / `seed_spread` | （无） | 逐级 95% CI 与 seed 均值/sd；既有 key 全保留 |

`subjects.se` 取合并 fit 联合 Hessian SE（与 §2.1 推荐读数一致）；§4.2 step 2 的保守
max 口径未写入该字段，分量由 `estimator.ci95`（deal bootstrap）与 `seed_spread`
（seed sd；÷√5 为 3.98–12.63）完整保留。

**rungs（D-2 接受实测间距）**：消费核对——`tools/measure_trace_signal.py` 只读
`levels`，`rungs` 是 `build_ladder`/`merge_manifest` 写入的选梯记录，故保持
`{id, elo, se}` schema。按旧契约同参推断（count=4、100≤spacing≤150、候选池 = lvl1–lvl4）
在新 levels 上重跑 `select_rungs`，只选出 **lvl1、lvl3**（lvl2 距 lvl1 72.9、lvl4 距
lvl3 66.1，均 <100）；wide gap lvl1→lvl3 217.4 >150、tail gap 66.1，`ok=False`。按
owner D-2 裁定如实写入，不为凑 spacing 调标签；新 `rungs` 仅 lvl1/lvl3 两项。

**剩余限制**：`rungs` 仍非契约达成态（2/4、`ok=False`）；`trace-signal-report.md` 的
标定数字未重跑（D-3 恢复时按新契约重跑）；D3 先验仍默认 T2/去收缩标定，manifest 新
levels 只作契约引用。

## 5. 复现与产物

```bash
# 复测（每 seed ~70–90s，cuda）
.venv/bin/python runs/lvl_rerate/rerate.py --seed 0 --games-per-anchor 400 \
    --device cuda --out runs/lvl_rerate/seed0.json --with-games
# seed 1..4 同
# 合并 fit + deal-cluster bootstrap（4000 次，~3min）
.venv/bin/python runs/lvl_rerate/analyze.py 4000
```

产物（均在 gitignore 的 `runs/` 下）：`runs/lvl_rerate/seed{0..4}.json`、
`seed{0..4}.games.json`、`analysis.json`、`run.log`、`rerate.py`、`analyze.py`。
`traces/study/manifest.json` 在复测期间未改（md5 `862b7ae7cec86188323cec945c0f6fb8`，
mtime 2026-09-25 01:44:16；该文件已于 2026-09-25 refit，见 §4.4）。

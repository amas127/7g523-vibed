# Head-to-head pilot：候选对候选换座对局与 400 副牌分辨率（7鬼523）

> 资产状态（2026-09-25）：本报告引用的模型 checkpoint 已删除，旧路径不再可用；数值为牌型族规则变更前口径。

> **状态：工具已实现并全套测试绿（248 passed），pilot 三对 × 三 seed 已跑完。**
> **结论：候选对候选的换座对局把「两个候选谁更强」的 95% CI 半宽从绝对 Elo
> 口径的 ±44～52 Elo 压到 **±19.6 Elo**（同样 800 局/候选，窄 2.3～2.7×）；
> 400 副牌**分辨不了 <10 Elo**（需约 1500～2000 副），能勉强分辨 ~20 Elo，
> 但 seed 0 的 `pool4g −19.6` 在 seed 1/2 复现失败（+8.7 / −2.6），说明临界
> 差异必须多 seed 或加牌数确认。三对的三 seed 合并值与 wave5「没有任何方案
> 显著超过 base680k」一致：`w5_vf1 +9.1 ± 5.8`、`w5_ctrl +4.2 ± 5.8`、
> `w5_pool4g −4.5 ± 8.2`（正 = base680k 更强）。**
> 工具：`src/seven523/duel.py` + `tools/head_to_head.py`（审计
> [`elo-reliability-audit.md`](./elo-reliability-audit.md) §6.5 的落地）。

## 0. 关键数字（先看这里）

| 量 | 数值 | 来源 |
|---|---|---|
| 400 副牌 head-to-head 的 95% CI 半宽 | **18.8–20.9 Elo**（9 次运行均值 19.6） | 本报告 §3 |
| 同一个比较的 seed 间 sd（8.7 / 9.6 / 14.2，池化 rms **11.1**） | ≈ bootstrap 隐含 sd **10.0**（比值 1.11） | §3.2 |
| 绝对 Elo 口径对比（wave5 合并 3 seed）的 95% CI 半宽 | **±51.9～52.3 Elo** | `wave5-500k-report.md` §4.2 |
| 绝对 Elo 单 seed（同样 800 局/候选）的 95% CI 半宽 | ≈±44（Fisher SE≈16/候选，独立近似；未含相位项） | 同上 + 审计 §2.3 |
| 分辨阈值需要的副牌数 | 30 Elo ≈ 200；**20 Elo ≈ 400–500**；15 Elo ≈ 700–850；**10 Elo ≈ 1500–2000** | §4 |
| pilot 三对（3 seed 合并，base−child） | `w5_vf1 +9.1`、`w5_ctrl +4.2`、`w5_pool4g −4.5`；**95% CI 半宽分别为 11.7/11.3/16.1（§3.2 勘误）** | §3.2 |
| 单次 400 副牌成本 | ~14 s（RTX 4060 Laptop、两条 ckpt、CUDA） | 实测 |

## 1. 动机

审计 [`elo-reliability-audit.md`](./elo-reliability-audit.md) 给出两条一手证据：

1. **绝对锚点 Elo 对方差极其敏感**：同一 ckpt 在同 seed、同牌集下只是候选顺序
   换了相位，位移可达 0–75 Elo（§3.3/§3.4）；单个 fit 的 Fisher SE（≈16）只
   覆盖「同调度换牌」，不覆盖「换 fit/换顺序」；跨 fit 比较的实测 sd ≈31/候选
   （§3.5）。
2. **§6.5 的建议**：「候选比较优先 head-to-head + 同牌 CRN：两个候选在同一副
   牌上直接对局、按牌配对统计胜率，方差远小于各自对锚点的绝对分。」

wave5 的实测印证了第 1 条：父模型 `base680k` 跨 3 个 seed 的 sd 高达 37，
父子偏移的符号整体翻转（−22.1 / +24.7 / −27.5），合并 3 seed 后 6 个臂相对
控制组 `w5_ctrl` 的 Δ 都在 **±11 Elo** 内（最大 `w5_vf1` +11.1）、相对
`base680k` 在 −17.8～+4.2，而差值 CI 半宽仍有 ±52
（[`wave5-500k-report.md`](./wave5-500k-report.md) §4.1/§4.2）。「某方案
+20～40 Elo」级别的结论在这种口径下无法证伪。

本 pilot 的目标不是重排 wave5 的名次，而是**验证工具本身的方差性质**：
head-to-head 的 CI 是否真的更窄、在同牌数下能否分辨 10/20 Elo、单 seed 的
CI 是否覆盖 seed 间变异。

## 2. 工具

### 2.1 `src/seven523/duel.py`（纯统计，无 torch/numpy）

* `plan_duel_schedule(left, right, *, pairs, seed)`：`pairs` **副牌**，每副牌
  抽一个 deal seed 并固定打两局、**左右座位互换**；`subject=left.id`，
  沿用 `ladder.ScheduledGame`，所以 `ladder.play_games` 原样可跑。
  `pairs` 是牌数不是局数（允许奇数），id 必须不同。
* `paired_duel_stats(games, *, left_id, right_id, bootstrap=4000, rng, confidence=0.95)`：
  * 聚簇单位 = **每副牌的 twin 对**（按 seed 配对，断言每副牌 left 两个座位
    各一局）；bootstrap 重采样**牌**（含放回），因此保留了 twin 内的座位配对
    与相关性；
  * 输出 left 视角 W/D/L、`winrate=(W+0.5D)/games`、平均分差（0–100 分制）、
    平均 return（分差/100）、`elo_diff = 400·log10(p/(1−p))`（`p` = left 每牌
    平均胜率，clamp 到 `[1/(2n), 1−1/(2n)]`，`n` = 牌数）的百分位 CI；
  * 另给每副牌 twin 净胜/净负/打平的符号统计与 exact 二项 p 值（纯 `math`）；
  * `rng` 必须显式传入 `random.Random`，同 seed 逐位复现；
    牌按 seed 排序处理，输入顺序不影响结果。

### 2.2 `tools/head_to_head.py`（CLI）

```bash
uv run --group train python tools/head_to_head.py \
  --left base680k=ckpt:runs/probe/base_step00696320.pt \
  --right w5_vf1=ckpt:runs/w5_vf1__1__1790320066/agent.pt \
  --pairs 400 --seed 0 --device cuda --bootstrap 4000 [--json] [--out TRACE_DIR]
```

`--left/--right` 用 `[ID=]SPEC`（复用 `build_ladder.py` 的 `random`/`greedy`/
`ckpt:<path>` 解析）；默认 `--pairs 400`、`--seed 0`、`--device cpu`、
`--bootstrap 4000`。`--json` 把统计 dict 打到 stdout，`--out` 走
`play_games(out=...)` 落可回放轨迹（subject = left id）。

> 路径勘误：任务简报中的 `runs/w5_vf1__1__1790319698/agent.pt` 不存在；
> `1790319698` 是 ctrl/scratch/bigbatch 的目录名，vf1 的实际目录是
> `runs/w5_vf1__1__1790320066/agent.pt`。本报告使用实际路径。

### 2.3 pilot 设置

* 三对：`base680k` vs `w5_vf1`、vs `w5_ctrl`、vs `w5_pool4g`；
  `--pairs 400 --seed 0 --device cuda --bootstrap 4000`（每对 800 局）。
* 因为工具很便宜（~14 s/对），三对又补跑了 seed 1/2，用来看「单次 CI 是否
  覆盖 seed 间变异」；产出在 `runs/h2h/*.json` 与 `runs/h2h/summary.json`。

## 3. 结果

### 3.1 seed 0：三对主表（正 elo_diff = base680k 更强）

| 对（left − right） | W/D/L（局） | winrate [95% CI] | 平均分差 [95% CI] | **elo_diff [95% CI]** | 半宽 | twin 符号 p |
|---|---|---|---|---|---|---|
| `base680k − w5_vf1` | 365/74/361 | 0.5025 [0.4744, 0.5306] | −0.14 [−2.64, +2.35] | **+1.7 [−17.8, +21.3]** | 19.6 | 0.882 |
| `base680k − w5_ctrl` | 369/69/362 | 0.5044 [0.4769, 0.5325] | +1.21 [−1.25, +3.80] | **+3.0 [−16.1, +22.6]** | 19.3 | 0.646 |
| `base680k − w5_pool4g` | 349/57/394 | 0.4719 [0.4450, 0.4988] | −1.50 [−3.88, +0.86] | **−19.6 [−38.4, −0.9]** | 18.8 | 0.161 |

读法：

* `w5_vf1` 与 `w5_ctrl` 与 `base680k` **完全不可分**（CI 宽达 ±20 且跨 0）；
  点估计与 wave5 合并口径（vf1 +4.2、ctrl −6.9）方向/量级都相容。
* `w5_pool4g` 的 seed 0 结果是 `−19.6 [−38.4, −0.9]`——**名义上 CI 刚好排除 0**，
  但同一副牌的符号检验 p=0.161、分差 CI [−3.88, +0.86] 跨 0。原因是
  **59% 的牌两局打平（净分差 0）**，符号检验的有效 N 只有 ~165，而连续型
  的 winrate/分差 bootstrap 更灵敏；三个通道给出「弱于 base」与「不可分」
  两个方向的证据，**不能据此宣布 pool4g 更弱**（§3.2 直接证伪）。
* 局级 draw 率 7–9%，`winrate` 按期望分 `(W+0.5D)/n` 计、不做「和棋重打」。

### 3.2 三 seed：单次 CI 与 seed 间变异的对账

| 对 | seed 0 | seed 1 | seed 2 | 三 seed 均值 | seed 间 sd | 单次 CI 半宽 | 合并读数 ± |
|---|---|---|---|---|---|---|---|
| `base − vf1` | +1.7 [−17.8,+21.3] | +18.7 [−1.3,+39.3] | +6.9 [−13.9,+27.9] | **+9.1** | 8.7 | 19.6–20.9 | **±5.8** |
| `base − ctrl` | +3.0 [−16.1,+22.6] | −4.8 [−24.8,+15.6] | +14.3 [−4.3,+33.5] | **+4.2** | 9.6 | 18.9–20.2 | **±5.8** |
| `base − pool4g` | **−19.6 [−38.4,−0.9]** | +8.7 [−9.6,+27.9] | −2.6 [−21.7,+18.3] | **−4.5** | 14.2 | 18.7–20.0 | **±8.2** |

> **勘误（2026-09-25，`evaluation-protocol-validation.md` §2.4）**：本表下方原文说
> 「± = max(...)·/√3 × 1.96」，但表中显示的 `±5.8/5.8/8.2` 实际是 **1σ 合并 SE**（未乘 1.96）。
> 按代码口径的 95% 半宽：`base−vf1 11.69`、`base−ctrl 11.27`、`base−pool4g 16.09`；
> 即 `w5_vf1 +9.1` 的 95% CI 是 **[−2.6, +20.8]，覆盖 0**——「1.6σ」的定性描述仍成立，
> 但不得读作 95% CI 排除 0。

两条重要结论：

1. **CI 大体校准**：单个 400 副牌运行的 bootstrap 隐含 sd ≈
   `半宽/1.96` ≈ 10.0 Elo；三对的 seed 间 sd 是 8.7/9.6/14.2，池化 rms
   **11.1**，比值 **1.11**。对比之下，绝对 Elo 口径的父模型跨 seed sd=37
   对单 fit Fisher SE≈16 的比值是 2.3、审计里换序 sd 31 对 SE 22 的比值是
   √2。也就是说 head-to-head 的 CI **没有重复绝对 Elo 那种「CI 远小于实际
   波动」的结构性错误**，尽管 3 个 seed 只能给一个粗校验。
2. **临界结论不可复现**：seed 0 的 `pool4g −19.6`「刚好显著」在 seed 1/2
   变成 +8.7 / −2.6。这正好落在 §4 的分辨阈值上——400 副牌对 20 Elo
   差异只有 ~2σ 的名义灵敏度，单 seed 的「显著」不应作为方案结论。

### 3.3 与绝对 Elo 口径的方差对比

| 口径 | 每候选局数 | 差值 95% CI 半宽 | 备注 |
|---|---|---|---|
| wave5 单 seed 绝对 Elo | 800（vs Random+Greedy 各 400） | ≈±44 | Fisher SE≈16/候选，独立差 ≈22.6；未计跨 fit 相位，是乐观下界 |
| wave5 合并 3 seed 绝对 Elo | 2400 | ±51.9–52.3 | 父模型跨 seed sd 37 主导；wave5 实际宣布「无优势」的口径 |
| **head-to-head 400 副牌（单 seed）** | 800（vs 对手） | **±19.6**（9 次均值） | 本 pilot |
| head-to-head 3 seed 合并 | 2400 | ±11.3–16.1 | 本 pilot；seed 间 sd 11.1 与 bootstrap 10.0 同量级 |

* 同样 800 局/候选：head-to-head 比绝对口径单 seed 窄 **2.3×**（方差 ≈5.1×，
  即同等精度省 ~5 倍局数）；比 wave5 实际使用的合并口径窄 **2.7×**，
  而且后者每候选还多打了 3 倍的局。
* 机制与审计 §3.5 的推断一致：绝对 Elo 要经过两个钉死的远锚点，锚点上
  饱和区的小样本翻转被 logistic 放大；head-to-head 直接测目标对比，
  同牌 CRN + 换座把「牌×座位」项在 twin 内消掉，只剩牌抽样。

## 4. 分辨率：分辨 10/20 Elo 需要多少副牌

以 400 副牌测得 σ₄₀₀ ≈ 10.0（bootstrap 隐含）/ 11.1（seed 间池化，
保守）外推 `σ_N = σ₄₀₀·√(400/N)`，95% 双尾可分辨要求 `Δ > 1.96 σ_N`：

| 目标 Δ | 需要副牌数（σ=10.0） | 需要副牌数（σ=11.1，保守） | 成本（~14 s/400 副） |
|---|---|---|---|
| 30 Elo | 171 | 211 | ~10–15 s |
| **20 Elo** | 384 | 475 | **~15–20 s** |
| 15 Elo | 683 | 845 | ~30 s |
| **10 Elo** | 1537 | 1893 | **~1–1.5 min** |

读法与建议：

* **400 副牌分辨不了 <10 Elo**：半宽 ±19.6 的 CI 天然覆盖 ±10 的差异；
  三对三 seed 合并后的 |Δ|≤9.1 也都在 ±5.8～8.2 的噪声带里，与 wave5
  「六臂全部不可分」一致。
* **20 Elo 用 400 副牌只是临界**（pool4g 的 seed 0 就是活例）；要宣布
  「A 比 B 强 20」建议 500–800 副或 2–3 个 seed 各 400 副再合并。
* **10 Elo 至少要 1500–2000 副牌**；即使如此也建议把总量拆成多 seed
  （seed 间 sd 与 bootstrap 同量级，k 个 seed 的均值再降 √k），顺带获得
  复现性证据。本 pilot 每对 400 副仅 ~14 s，10 Elo 的完整判定在单卡上
  也不过 1–2 分钟/对。
* 符号检验可作为补充，但局级和棋率 7–9%、twin 净零率 55–59%，它的有效
  样本远小于牌数；主推断应以牌聚簇的 winrate/margin bootstrap CI 为准。

## 5. 结论与建议

1. **工具可用、方差性质达标**：`tools/head_to_head.py` 在同牌数下把候选比较
   的 CI 收窄 2.3–2.7×，seed 间 sd 与 bootstrap 隐含 sd 同量级（1.11），
   没有绝对 Elo 的结构性低估。全套测试 248 passed（含 10 个 duel 测试：
   twin/座位/奇数牌数、clamp 饱和、零净对称、CI 覆盖 0、bootstrap 确定性、
   左右互换镜像、CLI 字段与解析报错）。
2. **方案对比今后优先 head-to-head**：先在 400–800 副牌上筛掉 <20 Elo 的
   差异认定，再对唯一「候选优势」用 1500+ 副牌或多 seed 确认；不要再靠
   单 fit 绝对 Elo 的 ±16 或合并后 ±52 宣布胜负。
3. **wave5 的「无优势」结论不变且更可信**：三对的三 seed 合并点估计都在
   ±10 内；唯一正向是 `w5_vf1 +9.1 ± 5.8`（1.6σ，跨 seed 不稳：+1.7/+18.7/
   +6.9），不足以推翻原结论。`pool4g` 的 seed 0 名义显著已被 seed 1/2 证伪，
   正是「head-to-head 也要多 seed」的现场证据。
4. **后续**：给 `play_games`/duel 路径加逐局 JSONL 或让 `--json` 附带
   per-deal 明细，便于事后做配对置换检验与多 seed 合并；方案筛选可先跑一
   个 3 seed × 200 副的 screening（每对 ~20 s）再决定是否加码。

## 6. 复现命令与产物

```bash
# 单元测试（10 个 duel 测试 + 全套 248）
.venv/bin/python -m pytest -q -p no:cacheprovider

# pilot 主表（每对 seed 0，400 副牌 = 800 局，CUDA，~14 s/对）
bash runs/h2h_pilot.sh          # -> runs/h2h/base680k_vs_w5_{vf1,ctrl,pool4g}.json

# seed 1/2 稳定性（同样 3 对，~90 s）
bash runs/h2h_seeds.sh          # -> runs/h2h/*_seed{1,2}.json

# 复盘单个 JSON（文本表）
.venv/bin/python tools/head_to_head.py \
  --left base680k=ckpt:runs/probe/base_step00696320.pt \
  --right w5_vf1=ckpt:runs/w5_vf1__1__1790320066/agent.pt \
  --pairs 400 --seed 0 --device cuda --bootstrap 4000
```

产物：`src/seven523/duel.py`、`tools/head_to_head.py`、`tests/test_duel.py`；
`runs/h2h/*.json`、`runs/h2h/*.log`、`runs/h2h/summary.json`、
`runs/h2h_pilot.sh`、`runs/h2h_seeds.sh`（均在 gitignore 的 `runs/` 下）。
本报告只读参考：`docs/experiments/elo-reliability-audit.md`（§3.3–§3.5、§6.5）、
`docs/experiments/wave5-500k-report.md`（§4.1–§4.2）。

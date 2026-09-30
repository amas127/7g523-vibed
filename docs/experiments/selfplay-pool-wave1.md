# self-play 退化机制 · 池化对手第一阶段（A1/A2/A3，cold start，k=5）

> **状态：已完成（2026-09-27，现行结论）——A1/A2/A3 点估计全 ≤ +5、CI 全跨 0，池化线在弱成员池下止损；机制写「未判定」。**
>
> **角色**：Execute agent。本报告对应 [`selfplay-pool-plan.md`](../selfplay-pool-plan.md)
> revision-2（第二轮 repair 后）的「池化实验」部分，修正来自
> [`selfplay-pool-review.md`](../selfplay-pool-review.md)；离线诊断部分见
> [`selfplay-pool-diagnostics.md`](./selfplay-pool-diagnostics.md)。
>
> **口径标签（硬性）**：全部为 revision-3（出空即撬底，`rules_id=2e36dbea44893696`）、
> obs v5（2 家 161 维）、纯 MLP（`arch=shared`、`hidden=128`、单层 trunk）。**跨 fit 绝对 Elo 不可比**。
>
> **诚实边界（plan §0.6 / §1.3-5）**：现有全部强资产都是 **random-trained MLP**（`GreedyBot` 已移除），
> 没有中立的异族 held-out 对手。本报告结论**只对 "random-trained MLP 家族内的对手身份/多样性" 成立**，
> **不得**写成 "self-play 本身退化"。
>
> **命名偏差**：plan §6.1 step 9 把本报告命名为 `selfplay-pool-wave.md`；任务书指定
> `selfplay-pool-wave1.md`，本文件按任务书命名，内容一致。

---

## 0. 结论先行

| 臂 | 定义 | 预注册主端点 `Δ=v.s. 同 seed C0` | z-CI | t-CI | 判定（plan §5.2） |
|---|---|---|---|---|---|
| **A1 `poolself`** | `{1@self(refresh10), 1@p2, 1@p3}`，无 random | **−2.71** | [−8.13, +2.71] | [−10.39, +4.96] | 点 ≤ +5 → **止损** |
| **A2 `fixed`** | `{1@p2}`，无 self | **−0.67** | [−11.12, +9.78] | [−15.48, +14.14] | 点 ≤ +5 → **止损** |
| **A3 `selfsamp`** | `{1@self}` + `--self-play-sample True` | **+0.25** | [−7.90, +8.40] | [−11.30, +11.80] | 点 ≤ +5 → **止损** |

- **三臂点估计全 ≤ +5、三者 z/t CI 全部跨 0** → **池化线收束**（资源结论）。
  按 plan §5.2「三臂全 ≤+5 → 写『在 random-trained MLP 家族内，对手分布修改不是绑定约束』」，
  但**不得**写成 "已证明对手分布无效"：CI 宽（t 半宽 8–15 Elo），机制问题仍写 **"未判定"**。
- **D0 门未过**（诊断报告 §2）：`p2/p3_500k` 对同预算代理 `p1_499k` 非劣，但对标尺锚 `l1_1M`
  未达非劣（p2 −8.42、p3 −13.13，`pass_anchor=false`）→ 本批池**不是强成员池**，
  结论限定为 **"弱成员池重估"**，**不得**宣称检验了强成员池（plan §0.1 / §6.1 step 5）。
- **无臂达 +10 行动门槛，且无 +5<Δ 的加 seed 触发** → 不扩 seed（未用 `self_s7/s8`、未新增 run），
  15 run 预算未超（3 臂 × 5 seed = 15 = 硬上限）。
- **解读矩阵落点**：`A1≈0, A2≈0, A3≈0` → 无任何机制获支持；H1/H1'/H2 均为 **"未判定"**。

---

## 1. 执行前核对（plan §6.1 step 1–2）

- **代码冻结**：`git rev-parse HEAD = ce22681af7162f75e1d40a0d701454d11640e985`。
- **冻结漂移（必须披露）**：诊断冻结（`runs/selfpool/freeze_record.txt`，2026-09-27T04:31）之后，
  **并行的 reward-alignment workflow** 于 04:49 修改了 `src/seven523/env.py` 与
  `src/seven523/train.py`（新增 `terminal_win` 奖励模式 + `--win-jump`，默认 `reward_shaping="terminal"`、
  `win_jump` 在其他模式被忽略）。训练前按 plan R4 重新冻结，记于
  `runs/selfpool/freeze_record_exec.txt`：
  - `train.py sha256=f6a1df80…`、`env.py sha256=d21f10ad…`；
  - **训练全程未再漂移**（训练后复核 hash 与 `freeze_record_exec.txt` 一致）。
  - **影响评估**：新增路径**默认关闭**、不读取 `win_jump` 之外的默认行为；`--reward-shaping` 默认仍为
    `terminal`（`train.py:120-128`、`:137-145`）。因此对本次全部 run 的**默认训练路径无行为影响**；
    `args.json` 仅多一个 inert 键 `win_jump=1.0`（见 §3 args diff）。此差异是**代码版本键**，
    非研究变量，已如实标注。
- **CLI 开关逐项核实存在且语义一致**（读 `train.py` / `league.py`）：
  `--opponent {random,self,mix,pool}`（`train.py:288-297`）、`--pool-member [WEIGHT@]SPEC`
  （`train.py:298-309`，`league.py:76`）、`--pool-episode`（`train.py:309-321`）、
  `--self-play-refresh`（`train.py:372-378`）、`--self-play-sample`（`train.py:378-385`）、
  `--cuda/--tensorboard`（`_bool`，接受 `True/False`）。`--opponent pool` 时 `build_league` 总是
  对 `agent` 做 `deepcopy` 出 `frozen`（`league.py:156-167`）；`1@self`→`frozen`（`league.py:185`），
  `ckpt:` 成员为**静态加载**（`policies.py:286`，argmax）；`--self-play-refresh 0` 短路
  （`train.py:695-703`）→ A2 单成员 `ckpt:p2` 即"单一固定 argmax 外部对手"。
- **ckpt 存在且 obs v5/161**：诊断报告 §1 已逐一 `torch.load` 核对；本次训练成员 `p2_500k`
  (`runs/t17pool__2__1790439615/agent.pt`)、`p3_500k` (`runs/t17pool__3__1790439615/agent.pt`)
  与 C0 `t17w1self__2..6`、锚 `l1_1M` (`runs/t17long__1__1790439615/snapshots/checkpoint_step1024000.pt`)
  均存在。
- **目录无冲突**：训练前无 `runs/t18*`；训练写入新建 `runs/t18poolself__*` /
  `runs/t18fixed__*` / `runs/t18selfsamp__*`。
- **资源**：训练前 `free -h` 可用 9.4 GiB、`nvidia-smi` RTX 4060 空闲；训练 3 并发 `nice -n 5`。
  **另注**：训练尾声发现**第三个 workflow** 在同机跑 `runs/t21c4k__{1,2,3}` GPU 训练（非本 workflow，
  未触碰）；本次 15 run 已在核心 GPU 上完成且全部 rc=0。

---

## 2. 臂定义与精确训练命令

控制臂 `C0` 复用 Wave-1（0 新 run）：`C0_s = runs/t17w1self__s__*`，`s∈{2..6}`，纯 self、refresh 10、
500k、args 见 §3。研究变量差异见 §3。

有界调度器 `runs/selfpool/run_training.sh`（`xargs -P 3`，`nice -n 5`；等价于 plan §4.2）：

```bash
# 每行 <exp> <seed>，共 15 行；三臂定义：
# A1 t18poolself: --opponent pool --pool-episode True --self-play-refresh 10
#                 --pool-member 1@self
#                 --pool-member 1@ckpt:runs/t17pool__2__1790439615/agent.pt
#                 --pool-member 1@ckpt:runs/t17pool__3__1790439615/agent.pt
# A2 t18fixed:    --opponent pool --pool-episode True --self-play-refresh 0
#                 --pool-member 1@ckpt:runs/t17pool__2__1790439615/agent.pt
# A3 t18selfsamp: --opponent self --self-play-refresh 10 --self-play-sample True
.venv/bin/python -m seven523.train --exp-name "$exp" --seed "$seed" \
  --total-timesteps 500000 $extra --cuda True --tensorboard False --run-dir runs
```

15 run 全部 rc=0，调度墙钟 **6589 s ≈ 110 min**（3 并发）。
run 目录：`runs/t18poolself__{2..6}__<ts>/`、`runs/t18fixed__{2..6}__<ts>/`、
`runs/t18selfsamp__{2..6}__<ts>/`（实际 ts 见 §9 的 run 清单）。

---

## 3. 健康度与 args diff

- **步数**：15/15 跑满 `499712` 步；`metrics.csv` 各 489 行；**无 NaN**（`grep -ci nan` 全 0）。
- **末熵（raw，仅记录，**不作**确定化证据）**：

  | 臂 | 末熵均值 ± sd | per-seed |
  |---|---|---|
  | A1 poolself | **1.4726 ± 0.0770** | 1.4609 / 1.3822 / 1.5520 / 1.3980 / 1.5700 |
  | A2 fixed | **1.4450 ± 0.0724** | 1.4577 / 1.4923 / 1.5455 / 1.3852 / 1.3443 |
  | A3 selfsamp | **1.5150 ± 0.0597** | 1.5271 / 1.4711 / 1.5834 / 1.4248 / 1.5688 |

  三者与 Wave-1 的 `C0`（1.33–1.55）同带；A3（采样对手）末熵略高，符合预期方向但幅度小。
- **args diff（对 `C0_s` 的差键，全部 15 run）**：仅含研究变量 + 代码版本键：
  `exp_name`、`seed`（同 seed 时不出现）、以及
  - A1：`opponent='pool'`、`pool_episode=True`、`pool_member=[1@self, 1@ckpt:p2, 1@ckpt:p3]`；
  - A2：`opponent='pool'`、`pool_episode=True`、`self_play_refresh=0`、`pool_member=[1@ckpt:p2]`；
  - A3：`self_play_sample=True`；
  - 三臂共同：`win_jump=1.0`（**inert 代码版本键**，`terminal` 模式不读取）。
  其余 46 个键与 C0 逐键相同。**证明只差研究变量 + 上述已披露代码键**。

---

## 4. 主端点（confirmatory，k=5 × 9 fresh deal seed × 400 副换座）

命令（每 seed 一条，严格串行、`--device cpu --workers 4`；deal seed 台账见 plan §3.6）：

```bash
.venv/bin/python tools/head_to_head.py \
  --left pself_s<N>=ckpt:runs/t18poolself__<N>__<ts>/agent.pt \
  --right self_s<N>=ckpt:runs/t17w1self__<N>__<self_ts>/agent.pt \
  --seeds 100,101,102,103,104,105,106,107,108 --pairs 400 --bootstrap 4000 \
  --device cpu --workers 4 --json > runs/selfpool/h2h_poolselfVSself_s<N>.json
# A2: --seeds 110..118 → h2h_fixedVSself_s<N>.json
# A3: --seeds 120..128 → h2h_selfsampVSself_s<N>.json
```

合并用 `runs/t17w1/aggregate.py`（`mean ± q·max(RMS(combined.se), training seed sd)/√k`，
z 与 t 双报）；Holm 与解读矩阵由 `runs/selfpool/merge_confirmatory.py` 复算。

### 4.1 逐 seed 原始表（Elo，换座配对）

| arm | s2 | s3 | s4 | s5 | s6 | MERGE | z-CI | t-CI | binding |
|---|---|---|---|---|---|---|---|---|---|
| **A1 poolself** | −0.24 | −9.95 | +1.16 | −8.55 | +4.02 | **−2.71** | [−8.13,+2.71] | [−10.39,+4.96] | seed sd 6.18 |
| **A2 fixed** | +15.65 | −16.68 | +1.45 | −6.28 | +2.51 | **−0.67** | [−11.12,+9.78] | [−15.48,+14.14] | seed sd 11.92 |
| **A3 selfsamp** | +1.50 | −15.03 | +7.82 | −0.59 | +7.54 | **+0.25** | [−7.90,+8.40] | [−11.30,+11.80] | seed sd 9.30 |

三方差分量（mean | RMS combined.se | training seed sd | 是否 binding）：
- A1：−2.71 | 4.08 | **6.18（binding）**；deal sd RMS 10.60；
- A2：−0.67 | 3.64 | **11.92（binding）**；deal sd RMS 9.88；
- A3：+0.25 | 4.06 | **9.30（binding）**；deal sd RMS 11.69。

**每个训练 seed 内的 9 个 deal seed 已由 `duel.py` RMS 合并为单点。**

### 4.2 Holm（3 个 confirmatory 端点，双侧 α=0.05）

| arm | Δ | z（binding 口径） | p（双侧正态） | Holm 调整 p | ± |
|---|---|---|---|---|---|
| A1 poolself | −2.71 | −0.50 | 0.617 | 1.00 | 不拒 |
| A2 fixed | −0.67 | −0.06 | 0.949 | 1.00 | 不拒 |
| A3 selfsamp | +0.25 | +0.03 | 0.976 | 1.00 | 不拒 |

**无任何端点显著，Holm 无需翻转。**

### 4.3 预注册判定（plan §5.2）

- 单臂门：`Δ_A ≥ +10` 且 z/t CI 排 0 → 确认；`+5 < Δ_A` 或（CI 跨 0 且点 > +5）→ 加 seed；
  `Δ_A ≤ +5` → **止损**。
- **A1/A2/A3 全部 `Δ ≤ +5`（点估计 −2.71 / −0.67 / +0.25）→ 三臂全部止损**，
  不触发 binding-term 升级（未用 `self_s7/s8`）。
- **解读矩阵**：`A1≈0, A2≈0, A3≈0` → 无机制获支持；因 **z/t CI 全部跨 0**，
  H1（池多样性）、H1'（固定外敌即可）、H2（argmax 确定性）一律写 **"未判定"**，
  **不写 null**（training seed sd 是 binding term，见 §4.1）。
- 资源结论（plan §5.2 行 4 + §7.2）：在**弱成员池**（D0 未过门）前提下，
  **对手分布修改不是绑定约束**；同时保留 "机制未判定" 的诚实措辞。

---

## 5. 描述性端点（不进 action table）

- **A1 vs `l1_1M`**（plan §5.1 E_anchor，**仅 A1**；deal 130–138）：
  逐 seed +11.02 / −3.04 / +10.78 / +4.59 / +8.36 →
  合并 **+6.34**，z-CI **[+1.22, +11.47]**、t-CI **[−0.92, +13.60]**，binding seed sd 5.85。
  **500k vs 1M 预算不匹配**，点 < +10 → 不行动。
  参照：Wave-1 F1b 纯 self `C0` vs `l1_1M` = **+2.87**（不同 deal seed，仅描述性）。
- **E0 共享 fit sanity**（vs RandomBot，`--seed 41`，8400 局 = 6×400 + C(6,2)×400，行数断言通过）：

  | 成员 | probit-MLE μ | 95% CI（deal-clustered, B=200） |
  |---|---|---|
  | pfixed_s2 | 174.68 | [162.83, 187.48] |
  | p2_500k | 170.78 | [157.71, 182.46] |
  | l1_1M | 169.89 | [156.81, 183.77] |
  | pselfsamp_s2 | 168.44 | [154.40, 181.27] |
  | pself_s2 | 162.80 | [149.84, 177.85] |
  | p3_500k | 160.53 | [148.10, 176.34] |

  单 seed、CI 宽 → 只做"**无灾难性退化**"sanity；**不进 action table**（plan §5.1 E0）。

---

## 6. 诊断（D0–D5）回执（详见 [`selfplay-pool-diagnostics.md`](./selfplay-pool-diagnostics.md)）

- **D0 门 = 不过**（`gate_pass=false`）：`p2` Δanchor **−8.42 [−18.01,+2.05]**、
  `p3` Δanchor **−13.13 [−24.14,−0.85]**（阈值下界 > −10）；两者 Δbudget 均过（>−5）。
  → 池臂降级为"弱成员池重估"。
- D1 round-robin（31,200 局，220 三元组，13 个点估计成环，最强三元组最小边差 33.3 < MDE 34）
  → **H-cycle 不可判**（零功效）。
- D2 δ = −0.0065 [−0.050,+0.032] / D4 MA CI 含 0 → H-mirror **未检出**（两者同批对局，合并一条证据）。
- D3 ρ=−0.143、精确置换 p=0.783 → n=7 不可判。
- D5 self 模板头熵 0.2145 vs random 0.3257，差 −0.111 [−0.121,−0.103] → H-det **弱旁证**（探索性）。

---

## 7. 局限（不得掩盖）

1. **弱成员池**：D0 未过门 → 本批只测了"`p2/p3` 这类弱/同预算成员"的池化，**不是强成员池**。
2. **无中立异族对手**（硬阻塞）：全部强资产 random-trained；机制结论**只限 MLP 家族内**。
3. **k=5 功效**：binding term 全为 training seed sd（6.2–11.9），t 半宽 7.7–14.8 Elo；
   真实效应 ≈ +10 大概率不可判。已按预注册在 `Δ≤+5` 时**不加 seed**。
4. **无 self 训练快照**：无法观测同 run 内策略追逐；诊断仅终点群体。
5. **代码版本键**：并行的 reward-alignment workflow 在诊断冻结后改过 `train.py/env.py`
   （additive、默认关闭）；本批 run 的 `args.json` 比 C0 多 `win_jump=1.0`。已复核默认路径不受影响，
   但仍如实登记为 **代码版本差异**。
6. **同机并发**：训练尾声有第三个 workflow（`t21c4k`）在用 GPU；本次 15 run 均在此之前启动并
   全部 rc=0，未观察到失败/NaN。评测为 CPU-only 串行。
7. A1 vs A2 的差同时含"self 成员存在"与"外部权重重归一化"（1/3 vs 100%），本报告**不把 A1−A2
   当作 self 成员的孤立因果**（plan §4.1）。

---

## 8. 复现命令

```bash
# 训练（15 run，<=3 并发）
bash runs/selfpool/run_training.sh

# 评测（E1/E2/E3 + 描述性 A1 vs l1_1M + E0），严格串行 CPU/4
bash runs/selfpool/run_eval.sh

# 合并 + Holm + 解读矩阵
for pair in poolselfVSself fixedVSself selfsampVSself; do
  .venv/bin/python runs/t17w1/aggregate.py --dir runs/selfpool --pair "$pair" \
    --seeds s2,s3,s4,s5,s6 --family confirmatory
done
.venv/bin/python runs/t17w1/aggregate.py --dir runs/selfpool --pair poolselfVS1M \
  --seeds s2,s3,s4,s5,s6 --family descriptive
.venv/bin/python runs/selfpool/merge_confirmatory.py
```

---

## 9. 产物清单

- 训练 run：`runs/t18poolself__{2..6}__*`、`runs/t18fixed__{2..6}__*`、`runs/t18selfsamp__{2..6}__*`
  （各含 `agent.pt/args.json/checkpoint.pt/metrics.csv`）。实际目录：
  - poolself：`__2__1790499154 __3__1790499154 __4__1790499154 __5__1790500101 __6__1790500108`
  - fixed：`__2__1790500112 __3__1790501067 __4__1790501072 __5__1790501074 __6__1790502173`
  - selfsamp：`__2__1790502176 __3__1790502177 __4__1790503691 __5__1790503953 __6__1790503963`
- 评测：`runs/selfpool/h2h_{poolselfVSself,fixedVSself,selfsampVSself,poolselfVS1M}_s{2..6}.json`
  （+同名 `.err`）。
- 合并/分析：`runs/selfpool/confirmatory_summary.json`、`runs/selfpool/merge_confirmatory.py`。
- 调度与日志：`runs/selfpool/run_training.sh`、`run_eval.sh`、`train_jobs.txt`、
  `train_scheduler.log`、`eval_scheduler.log`、`train_<exp>_<seed>.log`、`train_done`、`eval_done`。
- 冻结：`runs/selfpool/freeze_record.txt`（诊断）、`runs/selfpool/freeze_record_exec.txt`（训练）。
- E0：`runs/selfpool/e0_games.jsonl`（8400 行）、`e0_absolute_table.json`、`e0_ladder.out`。

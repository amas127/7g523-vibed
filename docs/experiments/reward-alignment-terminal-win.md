# Tier 1 边界跳变奖励 `terminal_win`：500k × 5 seed 训练 + 同 seed 配对 h2h（执行报告）

> **口径**：revision-3（出空即撬底，`rules_id=2e36dbea44893696`）、obs v5（2 家 161 维）、
> 纯 MLP（`arch=shared`、`hidden=128`、单层 trunk）、`--opponent random`、冷启动、500k
> （488 update = 499,712 步）、`eval_interval=0`（无同 run 在线 Elo）。
> **预注册**：`docs/reward-alignment-plan.md` §4.10（+ §4.3 命令模板、§4.5 deal seed 台账、
> §4.6 端点、§4.7 判定门、§4.8 多重比较）；本报告按预注册执行、未改门、未挑 seed。
> **权重**：`J10`（λ=1.0）为主臂，`J025`（λ=0.25）为小跳变臂，`terminal` 为同 seed 对照 C0。
> **执行时间**：2026-09-27 07:29–08:14（训练 + 评测）。

## 1. 执行摘要

- **训练**：12 条新 run（`ra_J025__{1..5}`、`ra_J10__{1..5}`、`ra_C0__{4,5}`）全部
  **499,712 步、488 update、无 NaN**；`C0` seed 1/2/3 按预注册复用
  `runs/t17pool__{1,2,3}__1790439615`（默认 `terminal` 路径在中间代码版本前后均与
  t17pool seed 1 **逐位一致**，见 §3.3）。
- **主端点判定（confirmatory，J10）**：同 seed 配对、3 deal seed（220/221/222）× 400 副
  × 2 座，合并 **+6.25 Elo**，`z-CI [−6.31,+18.80]`、`t-CI [−11.53,+24.02]`（k=5，
  binding = 训练 seed sd 14.32）。**CI 含 0 且点估计 <+10 → 不达行动门；点估计 >+5 →
  未触发止损**；按「<10 一律不追」记为**未判定 / 不行动**。
- **J025**：合并 **−4.59**，`z-CI [−20.08,+10.90]`、`t-CI [−26.53,+17.36]`；
  点估计 ≤+5 → **触发止损**。
- **次端点（描述，vs 冻结 `l1_1M`）**：`J10` **−1.60** `[z −9.19,+6.00]`、
  `J025` **−5.36** `[z −16.07,+5.35]`、`C0` **−8.70** `[z −17.95,+0.56]`；
  三条 CI 均含 0，不作判定。`J10` 相对 C0 向 1M 父模型方向移动约 +7 Elo（描述性）。
- **机制/健康度**：J 臂未牺牲 vs random 的 `P_w`（`C0/J025/J10` = 0.897/0.879/0.890，
  200 局/ckpt）；`entropy` 略升、`approx_kl`/`clipfrac` 略降；**无 critic 塌缩**
  （标准化 V sd ≈0.62–0.67、V-vs-R EV≈0.50–0.55）。`value_loss` 的上升是**回报尺度**
  效应，不是 critic 变差（见 §7.4）。

## 2. 执行前核对（全部通过）

| 项 | 结论 | 证据 |
|---|---|---|
| CLI 开关存在 | `--reward-shaping terminal_win`、`--win-jump`（float，默认 1.0）、`--opponent random` 均在 | `src/seven523/train.py:120-141`；`src/seven523/env.py:47,296-314` |
| 默认路径不变 | 不启用 `terminal_win` 时 4 个既有模式逐位不变；`win_jump=0` 与 `terminal` 逐位一致 | `tests/test_env.py`、`tests/test_train.py`（本轮执行前 102、saturate 合入后 134 passed） |
| 基线 ckpt | `runs/t17pool__{1,2,3}__1790439615/agent.pt`、`runs/t17long__1__1790439615/snapshots/checkpoint_step1024000.pt` 均在 | `ls runs/t17pool__*/agent.pt` 等 |
| run 目录无冲突 | 执行前 `runs/` 无 `ra_*`；12 个新 run 目录均新建 | `ls runs | grep -c ra_` = 0（开工前） |
| seed 台账 | 训练 seed **1–5**（§4.10 规定复用 1/2/3 对照 + 补 4/5；覆盖 §4.4 的新 seed 建议）；deal seed **220/221/222**（§4.5 fresh 集合 220–228 中取 3），screen seed **223**；均不与 selfplay 的 `0–18/29–40/100–148` 重叠（实查 `runs/t17w1/*.json`=`10–18`、`runs/selfpool/*.json`=`100–108…`） | `runs/ra/h2h_*_s*.json` 的 `seeds` 字段、`runs/ra/screen_*` |
| 并发纪律 | 训练 ≤3 并发、`nice -n 5`；评测 h2h 严格串行、一次一个、全 `--device cpu --workers 4` | `runs/ra_orchestrator.log`、`runs/ra/h2h_matrix.log`（无并行 h2h） |
| 排队 | 等外部 `t22` 容量实验 supervisor 退出（07:29:51 gate 放行）后启动；启动前后 `free -h`/`nvidia-smi` 干净 | `runs/ra_gate.out` |
| 代码版本边界 | ~07:35 `saturate`/`--reward-cap` 合入；第 1 波 run（07:29:51）`args.json` 无 `reward_cap` 键，第 2 波起多一个惰性 `"reward_cap": null`；`terminal`/`terminal_win` 分支与数值未受影响 | 见 §3.3 冒烟复现；`runs/ra_J025__1__*/args.json`（无键）vs `runs/ra_J025__3__*/args.json`（有键） |

## 3. 臂表与命令

### 3.1 臂

| 臂 | 模式 | λ | run 数 | 说明 |
|---|---|---|---|---|
| `C0`（对照） | `terminal` | — | 5 | seed 1/2/3 复用 `runs/t17pool__{1,2,3}__1790439615`；seed 4/5 新训 `ra_C0__4/5` |
| `J025` | `terminal_win` | 0.25 | 5 | 小跳变（≈ margin 尺度的 12.5%） |
| `J10` | `terminal_win` | 1.0 | 5 | **主臂**（分差与胜负各占一半） |

预注册集合固定为 `terminal` + `{0.25, 1.0}`，未事后加臂、未改 λ；`λ→∞` 极限才是 `win`，
本报告不引用 revision-2 `win` null 作效应上界（§1.4）。

### 3.2 训练命令（逐 run 均写入 run 目录；模板与 `runs/t17pool__1__1790439615/args.json` 对齐）

```bash
.venv/bin/python -m seven523.train \
  --exp-name ra_<J025|J10|C0> --seed <1..5> --total-timesteps 500000 \
  --num-envs 8 --num-steps 128 --num-minibatches 4 --update-epochs 4 \
  --num-players 2 --hidden-size 128 --arch shared \
  --learning-rate 2.5e-4 --anneal-lr --gamma 0.99 --gae --gae-lambda 0.95 \
  --clip-coef 0.1 --clip-vloss --norm-adv --vf-coef 0.5 --ent-coef 0.01 \
  --max-grad-norm 0.5 --torch-deterministic --cuda True --tensorboard False \
  --checkpoint-interval 8 --eval-interval 0 --log-interval 1 \
  --opponent random --reward-shaping <terminal|terminal_win> [--win-jump <0.25|1.0>]
```

- λ 臂：`--reward-shaping terminal_win --win-jump {0.25,1.0}`；C0：`--reward-shaping terminal`。
- 队列/命令：`runs/ra_queue.txt`、`runs/ra_cmds/*.sh`、`runs/ra_orchestrator.sh`（≤3 并发）。

### 3.3 `args.json` diff 与代码版本核对

以 `runs/t17pool__1__1790439615/args.json` 为基准逐键对比全部 12 条新 run：

- **研究变量**（唯一有意的差异）：`reward_shaping`（`terminal_win` vs `terminal`）、
  `win_jump`（0.25 / 1.0 / 默认 1.0）、`exp_name`、`seed`；其余键全部相同，
  含 `total_timesteps=500000`、`num_envs/steps/minibatches/epochs`、`lr/anneal_lr`、
  `hidden_size/arch`、`opponent=random`、`checkpoint_interval=8`、`eval_interval=0`。
- **惰性键（非研究变量）**：
  - T15–T17 既有新增默认键：`seq_*`、`event_*`、`seat_emb`（默认关：`seq_len=0`、`event_len=0`）；
  - `win_jump`：新代码总是写出（C0 也写 1.0，但 `terminal` 忽略之）；
  - `reward_cap`：~07:35 `saturate` 合入后新增的惰性键，**第 1 波 3 条 run 与复用的
    C0 seed 1–3 无该键、第 2 波起为 `null`**（C0 seed 4/5 与 J10 seed 2–5、J025 seed 3–5
    共 9 条有；J10 seed 1、J025 seed 1/2、C0 seed 1–3 共 6 条无）。该键对
    `terminal`/`terminal_win` 无行为影响。
- **默认路径逐位校验**（两次冒烟，`--total-timesteps 500000` 下与 t17pool seed 1 对比，
  除 `sps` 列外全列逐位相等）：
  - `saturate` 合入前：比较 41 个 update，**0 diffs**；
  - `saturate` 合入后：比较 30 个 update，**0 diffs**。
  因此第 1/2 波 run 与复用的 C0 seed 1–3 可同批比较。

## 4. 训练健康度

### 4.1 逐 run 汇总（均 488 update / 499,712 步 / 无 NaN；窗口 = step > 400k 的最后 98 update）

| run | episodic_return | entropy | approx_kl | clipfrac | value_loss | explained_var |
|---|---|---|---|---|---|---|
| C0_s1 | 0.5231 | 1.7278 | 5.99e-4 | 0.0037 | 0.0141 | 0.674 |
| C0_s2 | 0.5380 | 1.6772 | 6.30e-4 | 0.0041 | 0.0142 | 0.680 |
| C0_s3 | 0.5207 | 1.6952 | 6.13e-4 | 0.0034 | 0.0140 | 0.673 |
| C0_s4 | 0.5196 | 1.7206 | 7.24e-4 | 0.0046 | 0.0143 | 0.655 |
| C0_s5 | 0.5227 | 1.7069 | 7.06e-4 | 0.0052 | 0.0149 | 0.672 |
| J025_s1 | 0.7268 | 1.8129 | 5.40e-4 | 0.0024 | 0.0246 | 0.649 |
| J025_s2 | 0.7013 | 1.6982 | 5.54e-4 | 0.0035 | 0.0260 | 0.638 |
| J025_s3 | 0.7328 | 1.7037 | 5.88e-4 | 0.0025 | 0.0246 | 0.642 |
| J025_s4 | 0.7190 | 1.7565 | 6.89e-4 | 0.0040 | 0.0260 | 0.635 |
| J025_s5 | 0.7073 | 1.8426 | 5.34e-4 | 0.0029 | 0.0255 | 0.662 |
| J10_s1 | 1.2704 | 1.9187 | 3.52e-4 | 0.0009 | 0.0949 | 0.600 |
| J10_s2 | 1.2881 | 1.7936 | 3.49e-4 | 0.0015 | 0.0969 | 0.590 |
| J10_s3 | 1.2972 | 1.8882 | 3.63e-4 | 0.0009 | 0.0909 | 0.595 |
| J10_s4 | 1.3053 | 1.8439 | 3.97e-4 | 0.0011 | 0.0900 | 0.609 |
| J10_s5 | 1.2870 | 1.8107 | 3.58e-4 | 0.0010 | 0.0945 | 0.598 |

跨 seed 均值（同表右列）：`C0` return 0.525、entropy 1.706、clipfrac 0.0042、
value_loss 0.0143、EV 0.671；`J025` 0.717 / 1.763 / 0.0031 / 0.0253 / 0.645；
`J10` 1.290 / 1.851 / 0.0011 / 0.0934 / 0.598。

### 4.2 奖励通道确实被改变（不是「没训动」）

- vs random 的数字说明终局奖励里多了一个离散跳变：`episodic_return` 相比同 seed 对照
  的增量 ≈ `λ·(P_w−P_l)`：`J025` +0.193（≈0.25×0.79），`J10` +0.765（≈1.0×0.79）。
- `norm_adv=True` 的机制提醒（§4.10）：跳变对**回报仿射缩放不敏感**，归一化吸收部分水平
  变化，预期效应量小于原始跳变；本实验读到的差异（见 §5/§7）与此一致。

### 4.3 稳定性

- 三臂均无 NaN、无发散；`entropy` 由 C0 的 1.706 略升到 J10 的 1.851（`ent_coef=0.01`
  固定，跳变放大回报尺度后 entropy 相对权重变化的方向符合 §3.4）。
- `approx_kl`、`clipfrac` 均很小且 J10 更低（`clipfrac` 0.0042→0.0011），无训练崩溃迹象。

## 5. 主端点（confirmatory）：同 seed 配对 h2h

- 设计：每个训练 seed `s`，`J_s` vs `C0_s`；deal seed 220/221/222 × 400 副（每副 2 座，
  共 1200 副 = 2400 局/seed/臂）。两步合并：先 `tools/head_to_head.py`（`combine_duel_seeds`
  合 deal seed），再 `runs/t17w1/aggregate.py`（合训练 seed，`mean ± q·max(RMS combined.se,
  训练 seed sd)/√k`，z/t 双报）。
- 原始逐 seed 表（elo_diff = J − C0；`winrate` 平局计 0.5；`deal_p` 为 deal 符号精确 p）：

| pair | seed | elo_diff | 95% CI | combined.se | boot_se | deal_sd | winrate | deal_p |
|---|---|---|---|---|---|---|---|---|
| J10VSC0 | s1 | +4.78 | [−7.70,+17.26] | 6.37 | 11.03 | 7.18 | 0.5069 | 0.599 |
| J10VSC0 | s2 | −10.87 | [−24.19,+2.45] | 6.79 | 10.75 | 11.77 | 0.4844 | 0.108 |
| J10VSC0 | s3 | +21.65 | [−2.16,+45.45] | 12.15 | 11.35 | 21.04 | 0.5310 | 0.0027 |
| J10VSC0 | s4 | −4.06 | [−23.29,+15.18] | 9.81 | 11.04 | 17.00 | 0.4942 | 0.574 |
| J10VSC0 | s5 | +19.73 | [+6.21,+33.24] | 6.90 | 11.33 | 11.95 | 0.5283 | 0.0019 |
| J025VSC0 | s1 | +3.48 | [−9.26,+16.21] | 6.50 | 11.25 | 10.38 | 0.5050 | 0.633 |
| J025VSC0 | s2 | −30.20 | [−43.12,−17.29] | 6.59 | 11.41 | 9.95 | 0.4567 | 5.74e-6 |
| J025VSC0 | s3 | +16.11 | [−9.05,+41.28] | 12.84 | 11.19 | 22.24 | 0.5231 | 0.0142 |
| J025VSC0 | s4 | −13.05 | [−29.82,+3.72] | 8.55 | 11.43 | 14.82 | 0.4812 | 0.0876 |
| J025VSC0 | s5 | +0.72 | [−24.69,+26.14] | 12.97 | 10.94 | 22.46 | 0.5010 | 1.000 |

- 跨 5 训练 seed 合并（`runs/t17w1/aggregate.py` 原样输出）：

| 端点 | 点估计 | z-CI | t-CI | RMS combined.se | 训练 seed sd | binding | p(z) | Holm p |
|---|---|---|---|---|---|---|---|---|
| **J10VSC0** | **+6.25** | [−6.31,+18.80] | [−11.53,+24.02] | 8.70 | 14.32 | 训练 seed sd | 0.329 | 0.659 |
| J025VSC0 | −4.59 | [−20.08,+10.90] | [−26.53,+17.36] | 9.92 | 17.67 | 训练 seed sd | 0.562 | 0.659 |

（`t_{0.975,4}=2.776`；两臂合并 winrate：J10 0.5090、J025 0.4934。）

### 5.1 预注册判定（门：CI 完全排除 0 且点估计 ≥ +10；止损：点估计 ≤ +5）

| 臂 | 点估计 | CI 排 0 | ≥+10 | 判定 |
|---|---|---|---|---|
| **J10（主臂）** | +6.25 | 否（z/t 均跨 0） | 否 | **未判定 / 不行动**（+5<点<+10，按「<10 一律不追」）；**未触发 ≤+5 止损** |
| J025 | −4.59 | 否 | 否 | **点估计 ≤+5 → 止损**（关闭该臂，不再加牌） |

- **多重比较（按 §4.8）**：唯一 confirmatory 主臂是 `J10`，§4.8 规定判定只看主臂、
  `J025` 属 secondary、**不做 FWER 校正**；为稳妥另按两臂家族做 Holm，
  `Holm p = 0.659`，两臂均不显著。本报告两种读法都给，判定以上表 J10 行为准。
- 不写「接近显著」：即便按 z 也不排 0，且训练 seed sd（14.32）而非牌数支配了 CI 宽度。

## 6. 次端点（secondary，描述性）：vs 冻结 `l1_1M` 的迁移

- 冻结对手：`runs/t17long__1__1790439615/snapshots/checkpoint_step1024000.pt`（1M 步）。
  **预算不匹配**：被测臂 500k vs 对手 1M（+ `l1_1M` 的对数来自 3/2 倍预算的训练）。
- 同 3 deal seed × 400 副 × 2 座；每训练 seed 一条；`C0` 为迁移基线（额外补测，属描述）。

| 端点 | 点估计 | z-CI | t-CI | RMS combined.se | 训练 seed sd | binding | winrate |
|---|---|---|---|---|---|---|---|
| J10VS1M | −1.60 | [−9.19,+6.00] | [−12.35,+9.16] | 8.66 | 7.02 | RMS combined.se | 0.4977 |
| J025VS1M | −5.36 | [−16.07,+5.35] | [−20.53,+9.81] | 7.65 | 12.22 | 训练 seed sd | 0.4923 |
| C0VS1M | −8.70 | [−17.95,+0.56] | [−21.81,+4.42] | 7.30 | 10.56 | 训练 seed sd | 0.4875 |

- 三条 CI 均含 0，**不参与止损/行动判定**。描述性读数：`J10`（−1.60）比 `C0`（−8.70）
  向 1M 父模型方向近约 +7 Elo、`J025`（−5.36）近约 +3 Elo；方向与主端点（J10 更好）一致，
  但这是未配对的三条独立 h2h，不能当显著结论。
- **强对手 headroom 检查（§4.1 第 2 条）**：vs `l1_1M` 的胜率 ≈0.49–0.50（远低于 0.75），
  该对手上不存在评估饱和，奖励变异原则上可分辨 —— 满足 §4.10 对次端点的 headroom 说明；
  vs random 一侧继续饱和（P_w 0.88–0.90），与诊断一致。

## 7. 机制读数

### 7.1 终局分差分布（vs random screen，seed 223 × 100 副 × 2 座 = 200 局/ckpt）

| 臂 | P_w | P_t | P_l | E[r_term\|w] | E[r_term\|l] | P(\|d\|≤10) | mean\|d\| |
|---|---|---|---|---|---|---|---|
| C0（n=5 seed 均值±sd） | 0.897±0.012 | 0.032 | 0.071 | 0.639±0.005 | −0.292 | 0.088 | 59.4 |
| J025 | 0.879±0.010 | 0.035 | 0.086 | 0.650±0.014 | −0.298 | 0.104 | 59.7 |
| J10 | 0.890±0.025 | 0.034 | 0.076 | 0.636±0.018 | −0.302 | 0.094 | 59.0 |

- **未牺牲 `P_w`**：三臂范围重叠，离散跳变没有把胜率换成分差（R10 线索也没有出现
  「J 臂胜得更大」）；`P(|d|≤10)`、平局率、`E[r_term|w]` 都在噪声内。
- 主端点内部（J vs C0，两支势均力敌）：J10 左/右 `P_w` 0.4698/0.4519、
  `P(|d|≤10)` 0.203；J025 左/右 0.4564/0.4696、0.193——与 elo_diff 的噪声一致，
  没有系统性的 margin 形状改变。

### 7.2 entropy / KL / clipfrac

见 §4.1。方向：`entropy` J10 > J025 > C0（1.851 > 1.763 > 1.706）；
`approx_kl`（3.6e-4 < 5.8e-4 < 6.5e-4）与 `clipfrac`（0.0011 < 0.0031 < 0.0042）
随 λ 增大略微下降。无异常尖峰；`clipfrac` 在晚期多数 update 为 0
（晚期零值占比：C0 ≈0.40、J025 ≈0.44、J10 ≈0.57）。

### 7.3 critic 健康（`runs/ra/probe_values.py`，每 ckpt 200 局 vs random，用本臂自己的
奖励重放；γ=0.99，`G_t=γ^{T−t}r_T`；V 在 CPU 上前向）

| 臂（5 seed 均值） | V sd（原始） | 终局 r_T sd | V sd（标准化=V sd/r_T sd） | V-vs-G EV | corr | E[V_last\|胜] | E[V_last\|负] |
|---|---|---|---|---|---|---|---|
| C0 | 0.250 | 0.373 | 0.670 | 0.545 | 0.739 | +0.60 | −0.17 |
| J025 | 0.304 | 0.463 | 0.658 | 0.498 | 0.706 | +0.83 | −0.17 |
| J10 | 0.576 | 0.936 | 0.616 | 0.515 | 0.719 | +1.54 | −0.65 |

- **无 V 塌缩**：原始 V sd 全部远高于 §3.4 的 0.15 塌缩带；标准化后 ≈0.62–0.67
  （跨臂同尺度可比），EV≈0.50–0.55、corr≈0.71–0.74；胜/负条件 V 符号正确。
- 逐 ckpt 明细见 `runs/ra/value_probe.json`。

### 7.4 尺度混淆与缺失端点

- **`value_loss` 上升是尺度效应**：J10 的 `r_T` sd 是 C0 的 0.936/0.373 ≈ 2.51 倍，
  方差 ≈6.3 倍；`0.0143×6.3 ≈ 0.090`，与实测 J10 的 0.093 一致。该量不能跨臂直接比。
- **`explained_variance` 的分母是各臂自己的 `Var(y_true)`**（`ppo.py:251`），也不可直接
  跨臂比；本轮读数 C0 0.671 / J025 0.645 / J10 0.598，与 V 探针的 EV 方向一致
  （≈0.55/0.50/0.52），属轻微下降、不构成塌缩。
- **N/A**：(a) 归一化前后 GAE 分解需要重放训练 rollout 的逐局 outcome/margin，训练产物
  只有聚合（`train.py:947-950`），§4.6 允许记 **N/A**；(b) `value-clip 绑定比例`、
  `global grad-norm 绑定比例` 未进 `metrics.csv` → **N/A**（`clipfrac` 仅覆盖 policy clip）。

## 8. 局限

1. **k=5 且训练 seed 间方差大**：主端点 binding 是训练 seed sd（14.32/17.67），
   不是牌数；CI 宽度主要由 seed 间异质性（含 ±20 Elo 级符号跳变）决定。
2. **同 seed 配对只对到训练 seed 与 deal seed（CRN）**，不改变不同 seed 训练轨迹本身的
   随机性；J 臂与 C0 是独立训练 run。
3. **无在线 Elo**（`eval_interval=0`），训练健康度只有 return/entropy/KL/value 读数和
   离线 V 探针；绝对值 Elo 来自与固定 pool 的 h2h，不可跨 fit 比较。
4. **次端点预算不匹配**（500k vs 1M），且三条 h2h 是独立运行、未按 seed 配对合并，
   只能作描述。
5. **多重比较**：Holm 只覆盖两条 confirmatory-family 端点；screen/次端点不作判定、未校正。
6. **代码版本边界**：`saturate` 合入前/后各有 run（惰性 `reward_cap` 键），默认路径已用
   两次逐位冒烟 + 测试覆盖；但这是执行时的事实，如实记录。
7. **因果声明（§2.8 口径）**：本报告全部 h2h 是「换奖励 → 训练 → 对局」的直接反事实，
   但不写「跳变导致 +X Elo」；主端点未判定，次端点只描述。诊断 R1 仍成立：症状存在
   ≠ 干预有价值，本实验不因拓扑错位自动成立、也不因症状小被推翻。

## 9. 复现命令

```bash
# 0) 训练（每条 run 的命令；队列/日志见下）
runs/ra_cmds/ra_J10__1.sh            # 其余同理：ra_J025__<s>.sh / ra_C0__<s>.sh
# 并发由 runs/ra_orchestrator.sh 控制（≤3，nice -n 5）；日志 runs/ra_logs/*.log

# 1) 主端点 + 次端点（严格串行，一次一个，cpu/4 workers）
.venv/bin/python runs/ra/run_h2h_matrix.py        # 25 个 pair，日志 runs/ra/h2h_matrix.log
# 单个 pair 的等价命令：
.venv/bin/python tools/head_to_head.py \
  --left ra_J10__1=ckpt:runs/ra_J10__1__1790508592/agent.pt \
  --right C0_s1=ckpt:runs/t17pool__1__1790439615/agent.pt \
  --seeds 220,221,222 --pairs 400 --bootstrap 4000 --device cpu --workers 4 \
  --games-out runs/ra/games_J10VSC0_s1.jsonl --json > runs/ra/h2h_J10VSC0_s1.json

# 2) 跨训练 seed 合并（预注册工具，z/t 双报）
.venv/bin/python runs/t17w1/aggregate.py --pair J10VSC0  --seeds s1,s2,s3,s4,s5 --family confirmatory --dir runs/ra
.venv/bin/python runs/t17w1/aggregate.py --pair J025VSC0 --seeds s1,s2,s3,s4,s5 --family confirmatory --dir runs/ra
.venv/bin/python runs/t17w1/aggregate.py --pair J10VS1M  --seeds s1,s2,s3,s4,s5 --family descriptive  --dir runs/ra

# 3) vs random screen / V 探针 / 汇总分析
.venv/bin/python runs/ra/run_screen_rand.py
.venv/bin/python runs/ra/probe_values.py --episodes 200
.venv/bin/python runs/ra/analyze_terminal_win.py
```

## 10. 产物路径

| 内容 | 路径 |
|---|---|
| 训练 run（12 条新 + 3 条复用 C0） | `runs/ra_J025__{1..5}__*`、`runs/ra_J10__{1..5}__*`、`runs/ra_C0__{4,5}__*`、`runs/t17pool__{1,2,3}__1790439615` |
| 训练命令/队列/日志 | `runs/ra_cmds/`、`runs/ra_queue.txt`、`runs/ra_orchestrator.sh`、`runs/ra_logs/`、`runs/ra_orchestrator.log`、`runs/ra_gate.out` |
| h2h JSON / 逐局 games | `runs/ra/h2h_<pair>_s<seed>.json`（25 个）、`runs/ra/games_<pair>_s<seed>_seed<deal>.jsonl` |
| screen / V 探针 | `runs/ra/screen_<label>_rand.json`、`runs/ra/games_screen_*.jsonl`、`runs/ra/value_probe.json` |
| 汇总分析 | `runs/ra/terminal_win_results.json`、`runs/ra/analyze_terminal_win.py`、`runs/ra/run_h2h_matrix.py`、`runs/ra/run_screen_rand.py`、`runs/ra/probe_values.py` |
| 冒烟校验 | `runs/ra_smoke_term500k*`（默认路径逐位复现证据） |
| 工作区基线 | `runs/ra/baseline_workspace.txt` |

## 11. 结论一句话

**边界跳变（J10 λ=1.0）与同 seed terminal 对照组在 5 训练 seed × 3 deal seed × 400 副的
配对 h2h 上合并 +6.25 Elo，CI 跨 0 且 <+10 → 不达行动门、未触发止损（主臂未判定）；
小跳变 J025 为 −4.59，触发 ≤+5 止损；两支均未在 vs random 上牺牲胜率、无 critic 塌缩、
次端点 vs 1M 方向描述性偏正。因此本轮不支持把 `terminal_win` 推向默认路径。**

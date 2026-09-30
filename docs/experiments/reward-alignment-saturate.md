# Tier 1 分差饱和奖励 `saturate`（K_τ）：500k × 5 seed 训练 + 同 seed 配对 h2h（执行报告）

> **口径**：revision-3（出空即撬底，`rules_id=2e36dbea44893696`）、obs v5（2 家 161 维）、
> 纯 MLP（`arch=shared`、`hidden=128`、单层 trunk）、`--opponent random`、冷启动、500k
> （488 update = 499,712 步）、`eval_interval=0`（无同 run 在线 Elo）。
> **预注册**：`docs/reward-alignment-plan.md` §4.11（+ §3.1–§3.4 定义与机制、§4.3 命令模板、
> §4.4/§4.5 seed 台账、§4.6 端点、§4.7 判定门、§4.8 多重比较）；本报告按预注册执行、
> **未改门、未挑 seed**。
> **权重**：`K2`（τ=0.2）为**唯一 confirmatory 主臂**，`K7`（τ=0.7）为剂量臂（secondary），
> `K0`（τ=0，端点锚，用户字面读法）与 `win`（sign 端点锚）为可选、仅描述；
> `terminal` 为同 seed 对照 C0。
> **执行时间**：2026-09-27 08:20–09:21（训练 20 条）＋ 09:22–09:36（评测，严格串行）。

## 1. 执行摘要

- **训练**：20 条新 run（`ra_K0/K2/K7/win__{1..5}`）全部 **499,712 步、488 update、无 NaN**；
  C0 seed 1/2/3 按预注册复用 `runs/t17pool__{1,2,3}__1790439615`，seed 4/5 复用 J 波已训的
  `ra_C0__4__1790509946` / `ra_C0__5__1790509949`（不重复训练）。
- **主端点判定（confirmatory，K2）**：同 seed 配对、3 deal seed（**229/230/231**）× 400 副
  × 2 座，合并 **−18.31 Elo**，`z-CI [−26.04,−10.58]`、`t-CI [−29.27,−7.36]`
  （k=5，binding = 训练 seed sd 8.82；Holm p = 6.9e-06）。**CI 完全排除 0（全负）且点估计
  ≤ +5 → 触发预注册止损，按 §4.7 关闭奖励饱和线**；5/5 seed 全部为负
  （−32.68/−21.02/−12.32/−12.78/−12.75），方向一致。
- **剂量臂 K7**：合并 **+7.27**，`z-CI [−1.76,+16.31]`、`t-CI [−5.53,+20.08]`（p=0.115，
  Holm p=0.115）→ CI 跨 0、介于 +5 与 +10 → **未判定 / 不行动**（未触发止损）。
- **端点锚（描述）**：`K0`（τ=0）**−27.72** `[z −44.04,−11.41]`（CI 全负）、`win` **−6.77**
  `[z −20.17,+6.63]`；两者点估计 ≤+5，按门面读法属止损侧，但仅描述、不进判定。
- **V 塌缩止损未触发**：K2/K0 的原始 V sd（0.073 / 0.039）虽 ≤0.15，但 V-vs-R EV
  （+0.233 / +0.168）> 0，**AND 条件不成立**；K7 0.192/+0.477、win 0.303/+0.279、
  C0 0.250/+0.545。§3.4 预注册的「饱和降低 V sd 与 EV」方向被确认，但未达塌缩。
- **次端点（描述，vs 冻结 `l1_1M`）**：K2 −17.94（CI 全负）、K7 −5.57（CI 含 0）、
  win −16.36（CI 全负）、K0 −30.10（CI 全负）、C0 基线 −4.44（CI 含 0）；与主端点方向一致。
- **机制/健康度**：vs random 的 `P_w` 未牺牲（屏幕 200 局/ckpt：K2 0.883、K7 0.901、
  K0 0.869、win 0.881，C0 0.881）；强约制胜/负分差形状（`E[r_term|w]`、`P(|d|≤10)`）
  在屏幕噪声内；`entropy` 随饱和加深上升（C0 1.706 → K7 1.713 → K2 1.882 → win 1.972 →
  K0 2.027），`clipfrac`/`approx_kl` 下降。
- **与 J 波（`terminal_win` 跳变）的比较**：同一奖励轴上的两种形状都未过 +10 行动门；
  跳变波主臂 J10 为 +6.25（未判定、未止损），饱和波主臂 K2 为 −18.31（**显著负、止损**）；
  非单调剂量：τ=0.7（+7.27）≈ 不动，τ=0.2（−18.31）与 τ=0（−27.72）越饱和越差。

## 2. 执行前核对（全部通过）

| 项 | 结论 | 证据 |
|---|---|---|
| CLI 开关与校验 | `--reward-shaping saturate` 在 choices 中；`--reward-cap`（float，默认 `None`）存在；`saturate` 缺 cap 报 `ValueError`、其他模式给 cap 报错、`τ∉[0,1]`/非有限报错 | `src/seven523/train.py:161,183-190`、`src/seven523/env.py:42-81`；实测 5 组非法组合全部 raise，合法 cap（0.0/0.2/1.0）通过；`tests/test_env.py tests/test_train.py` **134 passed** |
| 奖励语义 | 胜局 `min(r_term,τ)`、平局 0、负局保留 `r_term`、非终局 0（2 家 + 3/4 家 outcome 条件式） | `tests/test_env.py:904-978`（含 3 家「低分胜局」与 4 家正分负局） |
| 默认路径不变 | `terminal` 路径未改；`saturate` 合入前后 J 波已做两次逐位冒烟（41/30 个 update，除 `sps` 外 0 diffs） | `docs/reward-alignment-terminal-win.md` §3.3 |
| 基线 ckpt | `runs/t17pool__{1,2,3}__1790439615/agent.pt`、`runs/ra_C0__{4,5}__*/agent.pt` 5 条全部存在 | 实查 `ls`（225 KB/条） |
| run 目录无冲突 | 开工前 `runs/` 无 `ra_K*`/`ra_win*`；20 个新目录全新 | `ls runs \| grep -E 'ra_K\|ra_win'`（开工前为空） |
| 训练 seed 台账 | 1–5（§4.11 规定复用 C0 1–3 + 补训 4/5 的 J 波对照），5 training seed | args.json |
| deal seed 台账（fresh） | **229/230/231**（主/次端点）、**232**（screen）。全量审计既有 h2h JSON 的 `seeds` 集合：`{0–4}`、`{10–18}`、`{100–148}`、`{150–152}`、`{220–222}`；229–232 与 selfplay `0–18/29–40/100–148` 及 **J 波 220–228 均不重叠**（任务要求的更严口径） | `runs/**/*.json` 全量 `seeds` 去重审计 |
| 并发纪律 | 训练 ≤3 并发、`nice -n 5`（20 条见 `runs/rk_orchestrator.log`）；h2h/screen 严格串行、一次一个、全 `--device cpu --workers 4` | `runs/rk/eval_pipeline.log`（45 个 h2h 无并行） |
| 排队与资源 | 与外部 `t23wswd`（3×1M，07:56–08:43）共享 GPU，本波训练 08:20–09:21 与其前 23 分钟并发；未对其做任何干预；每波前查 `free -h`/`nvidia-smi` | 内存峰值 11 GiB/15 GiB、GPU 峰值 992 MiB/8188 MiB |
| 代码版本边界（执行事实） | 本波 run 晚于并发 T23 改动，`args.json` 多 4 个惰性键（`optimizer/lr_schedule/weight_decay/snapshot_interval`）；见 §3.3 的默认路径等价证明 | §3.3 |

## 3. 臂表与命令

### 3.1 臂

| 臂 | 模式 | τ | run 数 | 说明 |
|---|---|---|---|---|
| `C0`（对照） | `terminal` | — | 5 | seed 1/2/3 复用 `t17pool`；seed 4/5 复用 J 波 `ra_C0__4/5`（不重复训练） |
| `K2` | `saturate` | 0.2 | 5 | **唯一 confirmatory 主臂**（§3.1：q10，饱和 ~90% 胜局） |
| `K7` | `saturate` | 0.7 | 5 | 剂量臂（secondary；q50，饱和 ~50% 胜局） |
| `K0`（可选） | `saturate` | 0.0 | 5 | 端点锚：胜局一律 0；**非 outcome-aligned**，仅描述 |
| `win`（可选） | `win` | — | 5 | 端点锚：`sign(d)`，outcome-aligned 极端（无分差信息），仅参照 |

预注册集合固定为 `terminal` + `saturate{0,0.2,0.7}` + `win`；未事后加臂，未改 τ。
`K0`/`win` 属 §4.11 可选项，本轮在预算允许下补齐（`K0` 排在主臂全部完成之后）。

### 3.2 训练命令（逐 run 写入 run 目录 `command.txt`；模板与 `runs/t17pool__1__1790439615/args.json` 对齐）

```bash
.venv/bin/python -m seven523.train \
  --exp-name ra_<K0|K2|K7|win> --seed <1..5> --total-timesteps 500000 \
  --num-envs 8 --num-steps 128 --num-minibatches 4 --update-epochs 4 \
  --num-players 2 --hidden-size 128 --arch shared \
  --learning-rate 2.5e-4 --anneal-lr --gamma 0.99 --gae --gae-lambda 0.95 \
  --clip-coef 0.1 --clip-vloss --norm-adv --vf-coef 0.5 --ent-coef 0.01 \
  --max-grad-norm 0.5 --torch-deterministic --cuda True --tensorboard False \
  --checkpoint-interval 8 --eval-interval 0 --log-interval 1 \
  --opponent random --reward-shaping <saturate --reward-cap <τ> | win>
```

- 队列/命令/日志：`runs/rk_queue.txt`、`runs/rk_cmds/*.sh`、`runs/rk_orchestrator.sh`（≤3 并发）、
  `runs/rk_logs/*.log`、`runs/rk_orchestrator.log`。
- 逐 run 目录（timestamps 依 seed；`command.txt` 与上表一一对应）：
  `runs/ra_K0__{1..5}__*`、`runs/ra_K2__{1..5}__*`、`runs/ra_K7__{1..5}__*`、`runs/ra_win__{1..5}__*`。

### 3.3 `args.json` diff 与代码版本核对

以 `runs/t17pool__1__1790439615/args.json` 为基准逐键对比全部 20 条新 run：

- **研究变量**（唯一有意的差异）：`reward_shaping`（`saturate`/`win`）、`reward_cap`
  （0.2/0.7/0.0/null）、`exp_name`、`seed`；其余训练超参（batch/lr/gamma/gae/clip/vf/ent/
  grad-norm/hidden/opponent/checkpoint 间隔/eval 间隔）**全部相同**。
- **惰性键（非研究变量）**：
  - T15–T17 既有默认键：`seq_*`、`event_*`、`seat_emb`（默认关：`seq_len=0`、`event_len=0`）；
  - `win_jump`：新代码总是写出（saturate 忽略之）；
  - `reward_cap`：saturate 分支读取；`terminal`/`win` 下写出但为 `null`、无行为影响；
    复用的 C0 seed 1–3（`t17pool`，`saturate` 合入前）**没有该键**，C0 seed 4/5 为 `null`
    ——与 J 波报告记录的代码版本边界一致；
  - **T23 时代新键（并发工作区在 J 波之后加入）**：`optimizer='adam'`、
    `lr_schedule='linear'`、`weight_decay=0.0`、`snapshot_interval=0`。默认路径等价的证据：
    (a) `train.py` 的旧行 `optim.Adam(..., lr, eps=1e-5)` 与新默认
    `optim.Adam(..., lr, eps=1e-5, weight_decay=0.0)` 逐位等价（Adam 的 wd 默认即 0）；
    (b) `lr_scale('linear', ·)` 文档声明且实测复现历史 `1-(update-1)/num_updates`；
    (c) **LR 轨迹逐位校验**：本波 K2 seed 1 与 `t17pool` seed 1、`ra_J10` seed 1 的
    `learning_rate` 列在全部 488 个 update **0 diffs**（首值 2.5e-4、末值 5.12e-7）；
    (d) `snapshot_interval=0` 为 no-op。该 4 键为执行时事实，如实记录（不改代码）。

## 4. 训练健康度

### 4.1 逐 run 汇总

- 20/20 run：**488 update / 499,712 步 / 无 NaN / 无发散**（`nan_cells` 全空）；
  训练窗口 08:20:00–09:21:42（61.7 min，含与外部 t23wswd 并发的前 23 min）。
- 晚期窗口（step > 400k 的最后 98 update）跨 seed 均值（±sd）：

| 臂 | episodic_return | entropy | approx_kl | clipfrac | value_loss | explained_var |
|---|---|---|---|---|---|---|
| C0 | 0.5248 ± 0.008 | 1.706 ± 0.019 | 6.54e-4 | 0.0042 | 0.0143 | 0.671 |
| K0 | −0.0313 ± 0.0005 | 2.027 ± 0.041 | 3.15e-4 | 0.0007 | 0.0018 | 0.104 |
| K2 | 0.1375 ± 0.0025 | 1.882 ± 0.073 | 3.69e-4 | 0.0012 | 0.0034 | 0.394 |
| K7 | 0.4518 ± 0.0032 | 1.713 ± 0.074 | 6.02e-4 | 0.0036 | 0.0100 | 0.630 |
| win | 0.7596 ± 0.011 | 1.972 ± 0.069 | 2.88e-4 | 0.0005 | 0.0460 | 0.489 |

- `value_loss`/`explained_var` 的分母是**本臂自己的回报尺度**（`ppo.py:251`），**不可跨臂比**；
  逐 run 明细见 `runs/rk/saturate_results.json` 的 `health`。

### 4.2 奖励通道确实被改变（不是「没训动」）

- 晚期 `episodic_return` 与公式一致：`K2 ≈ P_w·E[min(m,0.2)|w] + P_l·E[m|l]`
  ≈ 0.88·0.185 − 0.08·0.28 ≈ **0.14**（实测 0.138）；`K0` 胜局为 0、只剩负项
  ≈ −0.025（实测 −0.031）；`K7 ≈ 0.45`（实测 0.452）；`win ≈ P_w−P_l ≈ 0.81`
  （实测 0.760，含平局 0、对手随机波动）；C0 0.525。饱和确实作用于训练信号。
- `norm_adv=True` 机制提醒（§3.4）：饱和去掉**胜局内部分差散布**、相对上调胜负/损失梯度；
  对回报仿射缩放不敏感，故本波读到的效应（见 §5）主要由「胜局内信息被抹掉」解释。

### 4.3 稳定性

- 无 NaN、无崩溃；`entropy` 随饱和加深单调上升（C0 1.706 < K7 1.713 < K2 1.882 <
  win 1.972 < K0 2.027），`approx_kl`/`clipfrac` 同步下降，无异常尖峰；
  晚期 `clipfrac` 为 0 的 update 占比 C0 ≈0.40、K2 ≈0.56、K0 ≈0.65。

## 5. 主端点（confirmatory + 剂量/锚）：同 seed 配对 h2h

- 设计：每个训练 seed `s`，`arm_s` vs 同 seed `C0_s`；deal seed **229/230/231** × 400 副
  （每副 2 座，共 1200 副 = 2400 局/seed/臂）。两步合并：先 `tools/head_to_head.py`
  （`combine_duel_seeds` 合 deal seed），再 `runs/t17w1/aggregate.py`（合训练 seed，
  `mean ± q·max(RMS combined.se, 训练 seed sd)/√k`，z/t 双报）。
- 原始逐 seed 表（elo_diff = arm − C0；`winrate` 平局计 0.5；`deal_p` 为 deal 符号精确 p）：

| pair | seed | elo_diff | 95% CI | combined.se | boot_se | deal_sd | winrate | deal_p |
|---|---|---|---|---|---|---|---|---|
| K2VSC0 | s1 | −32.68 | [−45.32,−20.05] | 6.45 | 11.17 | 9.10 | 0.4531 | 9.06e-07 |
| K2VSC0 | s2 | −21.02 | [−33.76,−8.29] | 6.50 | 11.26 | 8.43 | 0.4698 | 0.0082 |
| K2VSC0 | s3 | −12.32 | [−25.11,+0.48] | 6.53 | 11.30 | 10.03 | 0.4823 | 0.0696 |
| K2VSC0 | s4 | −12.78 | [−37.74,+12.18] | 12.74 | 11.29 | 22.06 | 0.4817 | 0.0692 |
| K2VSC0 | s5 | −12.75 | [−26.58,+1.08] | 7.06 | 11.27 | 12.22 | 0.4817 | 0.0674 |
| K7VSC0 | s1 | +4.49 | [−7.94,+16.92] | 6.34 | 10.99 | 3.64 | 0.5065 | 0.440 |
| K7VSC0 | s2 | −7.96 | [−20.36,+4.43] | 6.33 | 10.96 | 5.00 | 0.4885 | 0.274 |
| K7VSC0 | s3 | +12.46 | [−0.33,+25.25] | 6.53 | 11.30 | 8.55 | 0.5179 | 0.0310 |
| K7VSC0 | s4 | +7.53 | [−9.66,+24.73] | 8.77 | 11.00 | 15.19 | 0.5108 | 0.193 |
| K7VSC0 | s5 | +19.86 | [+7.13,+32.58] | 6.49 | 11.24 | 2.97 | 0.5285 | 0.0137 |
| winVSC0 | s1 | −9.56 | [−22.31,+3.18] | 6.50 | 11.26 | 9.80 | 0.4863 | 0.156 |
| winVSC0 | s2 | −30.80 | [−45.80,−15.80] | 7.65 | 10.60 | 13.26 | 0.4558 | 1.32e-07 |
| winVSC0 | s3 | −5.51 | [−19.78,+8.77] | 7.28 | 11.07 | 12.61 | 0.4921 | 0.356 |
| winVSC0 | s4 | +9.27 | [−3.52,+22.06] | 6.53 | 11.30 | 7.33 | 0.5133 | 0.266 |
| winVSC0 | s5 | +2.75 | [−9.39,+14.89] | 6.19 | 10.73 | 7.39 | 0.5040 | 0.507 |
| K0VSC0 | s1 | −5.65 | [−19.71,+8.41] | 7.17 | 11.40 | 12.43 | 0.4919 | 0.558 |
| K0VSC0 | s2 | −48.53 | [−61.43,−35.63] | 6.58 | 11.40 | 6.21 | 0.4306 | 3.87e-12 |
| K0VSC0 | s3 | −41.33 | [−54.56,−28.10] | 6.75 | 11.69 | 10.58 | 0.4408 | 7.31e-10 |
| K0VSC0 | s4 | −31.66 | [−44.41,−18.91] | 6.51 | 11.27 | 9.42 | 0.4546 | 1.99e-06 |
| K0VSC0 | s5 | −11.44 | [−24.19,+1.30] | 6.50 | 11.26 | 8.22 | 0.4835 | 0.0841 |

- 跨 5 训练 seed 合并（`runs/t17w1/aggregate.py` 原样输出，逐位复算见 §10）：

| 端点 | 点估计 | z-CI | t-CI | RMS combined.se | 训练 seed sd | binding | p(z) | Holm p (saturate 家族) |
|---|---|---|---|---|---|---|---|---|
| **K2VSC0（confirmatory）** | **−18.31** | [−26.04,−10.58] | [−29.27,−7.36] | 8.23 | 8.82 | 训练 seed sd | 3.46e-06 | **6.92e-06** |
| K7VSC0（剂量） | +7.27 | [−1.76,+16.31] | [−5.53,+20.08] | 6.96 | 10.31 | 训练 seed sd | 0.115 | 0.115 |
| winVSC0（锚，描述） | −6.77 | [−20.17,+6.63] | [−25.75,+12.21] | 6.85 | 15.29 | 训练 seed sd | 0.322 | — |
| K0VSC0（锚，描述） | −27.72 | [−44.04,−11.41] | [−50.83,−4.61] | 6.71 | 18.61 | 训练 seed sd | 8.66e-04 | — |

（`t_{0.975,4}=2.776`；合并 winrate：K2 0.4737、K7 0.5105、win 0.4903、K0 0.4603。）

### 5.1 预注册判定（门：CI 完全排除 0 且点估计 ≥ +10；止损：点估计 ≤ +5）

| 臂 | 点估计 | CI 排 0 | ≥+10 | 判定 |
|---|---|---|---|---|
| **K2（主臂）** | **−18.31** | **是（z/t 均全负）** | 否 | **点估计 ≤+5 → 止损；按 §4.7 关闭奖励饱和线**（且 CI 给出明确负效应） |
| K7（剂量，secondary） | +7.27 | 否（z/t 均跨 0） | 否 | **未判定 / 不行动**（+5<点<+10）；未触发止损 |
| win（锚，描述） | −6.77 | 否 | 否 | 按门面读法属 ≤+5 止损侧，但仅描述、不进判定 |
| K0（锚，描述） | −27.72 | 是（z/t 均全负） | 否 | 仅描述（非 outcome-aligned，不参与收束） |

- **多重比较（§4.11）**：saturate 家族 = {K2, K7}，独立 Holm 校正：K2 p=3.46e-06 →
  Holm 6.92e-06；K7 p=0.115 → Holm 0.115。**判定只看 K2 主端点**；J 家族不并入、不互相校正。
- **剂量方向（机制，非判定；§4.7）**：τ 越大越不饱和、效应越不差：
  τ=0（K0 −27.72）< τ=0.2（K2 −18.31）< τ=0.7（K7 +7.27）；`win`（无分差信息）−6.77。
  即在本批 h2h 上，**越抹掉胜局分差信息越差**，τ=0.7 的轻度饱和接近不动点
  （点估计为正但不显著）。

### 5.2 V 塌缩止损检查（§3.4/§4.11，AND 条件）

| 臂（5 seed 均值） | V sd（原始） | 终局 r_T sd | V sd（标准化） | V-vs-G EV | §4.11 触发（V sd ≤0.15 且 EV ≤0） |
|---|---|---|---|---|---|
| C0 | 0.250 | 0.373 | 0.670 | +0.545 | 否 |
| K0 | 0.039 | 0.107 | 0.369 | +0.168 | **否（V sd 达标，EV >0）** |
| K2 | 0.073 | 0.140 | 0.519 | +0.233 | **否（V sd 达标，EV >0）** |
| K7 | 0.192 | 0.302 | 0.637 | +0.477 | 否 |
| win | 0.303 | 0.590 | 0.512 | +0.279 | 否 |

- **结论：无 critic collapse、不触发 V 止损。** 预注册预测（饱和 ⇒ V sd 与 EV 下降）在方向
  上被确认（K2/K0 ≪ C0），但 EV 始终为正，未满足 AND 条件的第二项；因此该读数是
  「跟随机回报尺度收缩的 critic 变弱」的机制证据，而非塌缩判据。
- 边际正确性仍在：K2 `E[V_last|胜] +0.17` vs `E[V_last|负] −0.06`；K0 `−0.007 / −0.116`；
  win `+0.89 / −0.20`。逐 ckpt 明细见 `runs/rk/value_probe.json`。

## 6. 次端点（secondary，描述性）：vs 冻结 `l1_1M` 的迁移

- 冻结对手：`runs/t17long__1__1790439615/snapshots/checkpoint_step1024000.pt`（1M 步）。
  **预算不匹配**：被测臂 500k vs 对手 1M；只作描述，不参与止损/行动判定。
- 同 3 deal seed × 400 副 × 2 座；每训练 seed 一条；C0 为迁移基线（同 deal seed 重测）。

| 端点 | 点估计 | z-CI | t-CI | RMS combined.se | 训练 seed sd | binding | winrate |
|---|---|---|---|---|---|---|---|
| K2VS1M | −17.94 | [−29.21,−6.67] | [−33.91,−1.97] | 7.45 | 12.86 | 训练 seed sd | 0.4742 |
| K7VS1M | −5.57 | [−12.35,+1.21] | [−15.17,+4.03] | 7.73 | 4.60 | RMS combined.se | 0.4920 |
| winVS1M | −16.36 | [−24.90,−7.83] | [−28.45,−4.27] | 7.33 | 9.74 | 训练 seed sd | 0.4765 |
| K0VS1M | −30.10 | [−38.06,−22.15] | [−41.37,−18.84] | 6.86 | 9.07 | 训练 seed sd | 0.4568 |
| C0VS1M | −4.44 | [−12.55,+3.68] | [−15.93,+7.06] | 7.48 | 9.26 | 训练 seed sd | 0.4936 |

- 描述性读数与主端点方向一致：K2/win/K0 相对同批 C0 基线（−4.44）明显更差；K7（−5.57）
  与 C0 同档，CI 含 0。**强对手 headroom 检查**：vs `l1_1M` 胜率 ≈0.46–0.50（远低于 0.75），
  该对手上不存在评估饱和，变异可分辨；vs random 一侧继续饱和（P_w ≈0.87–0.90）。

## 7. 机制读数

### 7.1 原始分差分布（vs random screen，seed 232 × 100 副 × 2 座 = 200 局/ckpt）

| 臂 | P_w | P_t | P_l | E[r_term\|w] | E[r_term\|l] | P(\|d\|≤10) | mean\|d\| |
|---|---|---|---|---|---|---|---|
| C0（5 seed 均值±sd） | 0.881±0.022 | 0.041 | 0.078 | 0.623±0.020 | −0.267 | 0.116 | 57.0 |
| K0 | 0.869±0.024 | 0.038 | 0.093 | 0.617±0.003 | −0.273 | 0.111 | 56.1 |
| K2 | 0.883±0.015 | 0.039 | 0.078 | 0.616±0.022 | −0.277 | 0.108 | 56.6 |
| K7 | 0.901±0.028 | 0.038 | 0.061 | 0.623±0.020 | −0.284 | 0.095 | 57.9 |
| win | 0.881±0.046 | 0.050 | 0.069 | 0.608±0.016 | −0.286 | 0.127 | 55.6 |

- **未牺牲 vs random 的 `P_w`**：五臂范围重叠（0.869–0.901），饱和没有把胜率换成分差；
  `E[r_term|w]`、`P(|d|≤10)`、平局率都在屏幕噪声内（R10 的「赢得多=更强」线索在屏幕上看不到
  系统性差异）。
- 主端点内部（arm vs 同 seed C0，两支势均力敌）：K2 左/右 `P_w` 0.435/0.488、
  K7 0.472/0.451、win 0.452/0.472、K0 0.423/0.502——与 elo_diff 的符号/幅度一致；
  左/右 `P(|d|≤10)` 几乎相同（0.199–0.205），说明差异来自胜负本身而非平局/分差形状。
- 注意：screen 记录的是**原始分数分差**；saturate 只改训练时终局 reward 的幅度映射，
  不改对局结果形状。

### 7.2 entropy / KL / clipfrac

见 §4.1/§4.3。方向：饱和越深 `entropy` 越高（1.706→1.713→1.882→1.972→2.027），
`approx_kl` 与 `clipfrac` 越低；无异常尖峰。该读数与 §3.4 的「回报尺度/相对权重改变」
一致，但**不能**单凭 entropy 判断好坏。

### 7.3 critic 健康

见 §5.2。无塌缩；V sd 与 EV 随饱和加深收缩，符合 §3.4 预测；K2 的标准化 V sd
（0.519）仍高于塌缩带，EV（+0.233）为正。

### 7.4 尺度混淆与缺失端点

- **`value_loss` 跨臂不可比**：K2 的终局 `r_T` sd（0.140）≈ C0（0.373）的 0.38，
  方差 ≈0.14 倍；`0.0143×0.14 ≈ 0.0020`，与实测 K2 的 0.0034 同量级（含未饱和负局尾巴）。
- **`explained_variance` 分母是各臂自己的 `Var(y_true)`**，跨臂不可直接比；
  方向与 V 探针 EV 一致（C0 0.671 > K7 0.630 > K2 0.394 > win 0.489 > K0 0.104）。
- **N/A**：(a) 归一化前后 GAE 分解需重放训练 rollout 的逐局 outcome/margin，训练产物只有
  聚合（`train.py:947-950`），§4.6 允许记 **N/A**；(b) `value-clip 绑定比例`、
  `global grad-norm 绑定比例` 未进 `metrics.csv` → **N/A**（`clipfrac` 仅覆盖 policy clip）。
- **超参混淆声明（§4.7）**：首波未改 `vf_coef`/`ent_coef`；若要把 K2 的负效应完全归因于
  critic 权重，需另设 `vf_coef` 臂——本报告只给「未改 `vf_coef`」的读数，不主张单一机制。

## 8. 与 J 波（`terminal_win` 跳变）的比较：同一奖励轴的两个形状

- 两波同 pipeline、同 `--opponent random` 冷启动、同 5 训练 seed、同 C0 台账（seed 4/5 共用）、
  同端点协议与判定门；**deal seed 不同**（J=220/221/222；K=229/230/231），是**独立、未配对**
  的两批 h2h（跨波不可直接相减）。

| 臂 | 形状（终局奖励） | 主端点合并 | z-CI | t-CI | 判定 |
|---|---|---|---|---|---|
| J025（λ=0.25） | `r_term + 0.25·sign`（跳变） | −4.59 | [−20.08,+10.90] | [−26.53,+17.36] | 止损 |
| J10（λ=1.0） | `r_term + 1.0·sign`（跳变） | +6.25 | [−6.31,+18.80] | [−11.53,+24.02] | 未判定 |
| **K2（τ=0.2）** | `min(r_term, 0.2)`（饱和） | **−18.31** | [−26.04,−10.58] | [−29.27,−7.36] | **止损（CI 全负）** |
| K7（τ=0.7） | `min(r_term, 0.7)`（饱和） | +7.27 | [−1.76,+16.31] | [−5.53,+20.08] | 未判定 |
| win | `sign(d)`（极端 outcome） | −6.77 | [−20.17,+6.63] | [−25.75,+12.21] | （描述；点 ≤+5） |
| K0（τ=0） | `min(r_term, 0)`（用户字面） | −27.72 | [−44.04,−11.41] | [−50.83,−4.61] | （描述；CI 全负） |

- **形状对比（描述）**：
  1. **跳变（J）**：在 `terminal` 上**加**边界信号，主臂 +6.25（未判定、未止损）；
     机制上 `norm_adv` 吸收大部分水平变化，效应量不大。
  2. **饱和（K）**：在 `terminal` 上**减**胜局内分差信息。轻度（τ=0.7）与不动点无显著差
     （+7.27，CI 含 0）；中度（τ=0.2）**显著为负 −18.31**；极端（τ=0 或 `win`）为
     −27.72 / −6.77。**抹掉分差信息的行为在本批中不可用**。
  3. 两者共同点：都未过 +10 行动门；也都**未牺牲 vs random 的 P_w**（J 0.879–0.897、
     K 0.869–0.901），说明差异出现在势均力敌的对局（vs 训练对手/1M），而不是碾压弱对手。
  4. 机制读数也不同向：J10 的 V sd 上升（0.58 vs C0 0.25）、entropy +0.15；
     K2 的 V sd 下降（0.073）、entropy +0.18、EV 0.55→0.23。两者不是同一机制。
- **结论**：在本批次内，奖励轴上的**两种「重排终局信息」形状**都没有给出可行动的增益；
  「不加跳变、也不截断分差」（即保持 `terminal`）在现有证据下仍是默认选择。

## 9. 局限

1. **k=5 且训练 seed 间方差大**：K2 的 binding 是训练 seed sd（8.82）但 5/5 seed 同号，
   所以 CI 全负；K7 的 CI 宽度主要由 seed 异质性（含 +19.86/−7.96 级跳变）决定，
   点估计 +7.27 不等于「有 +7.27 的效应」。
2. **跨波未配对**：与 J 波使用不同 deal seed，§8 的比较是「同协议、独立批次」的描述，
   不是同一批牌上的配对比。
3. **无 warm-start 块**：§4.11 臂表为冷启动；`K2w` 未训练（§4.7 的 K2w 属暖启动块）。
   「关闭奖励饱和线」的结论限于**冷启动 K2**；若要暖启动迁移结论需另批（不在本波预算）。
4. **V 探针只测最终 ckpt**：原始 V sd 依赖各臂自己的回报尺度；§4.11 的塌缩规则用
   AND（EV≤0）保护，未触发；标准化 V sd 一并报告以作跨臂合理性检查。
5. **缺失端点**：GAE 归一化分解、value-clip/global-grad-norm 绑定比例 N/A（见 §7.4）。
6. **代码版本边界**：本波 run 晚于并发 T23 改动，新增 4 个惰性键；默认路径等价性用
   代码核对 + LR 轨迹逐位一致证明（§3.3），且复用的 C0 seed 1–3 早于该改动——这是执行
   事实，如实记录；未改任何代码来「对齐」版本。
7. **共享 GPU**：训练前 23 min 与外部 `t23wswd` 并发（未干预）；其只影响墙钟时间，
   不影响确定性种子与协议。
8. **因果声明（§2.8 口径）**：本报告全部 h2h 是「换奖励 → 训练 → 对局」的直接反事实，
   数字是协议内的测量差；不写「饱和导致 −X Elo」的机制性因果，score 端的因果解释
   （R1）仍**未判定**。

## 10. 复现命令

```bash
# 0) 训练（每条 run 的命令；队列/日志见下）
runs/rk_cmds/ra_K2__1.sh              # 其余同理：ra_{K0,K7,win}__<s>.sh；command.txt 已写入各 run 目录
# 并发由 runs/rk_orchestrator.sh 控制（≤3，nice -n 5）；日志 runs/rk_logs/*.log

# 1) 主端点 + 次端点（严格串行，一次一个，cpu/4 workers）
.venv/bin/python runs/rk/run_h2h_matrix.py          # 45 个 pair，日志 runs/rk/eval_pipeline.log
# 单个 pair 的等价命令：
.venv/bin/python tools/head_to_head.py \
  --left ra_K2__1=ckpt:runs/ra_K2__1__1790511602/agent.pt \
  --right C0_s1=ckpt:runs/t17pool__1__1790439615/agent.pt \
  --seeds 229,230,231 --pairs 400 --bootstrap 4000 --device cpu --workers 4 \
  --games-out runs/rk/games_K2VSC0_s1.jsonl --json > runs/rk/h2h_K2VSC0_s1.json

# 2) 跨训练 seed 合并（预注册工具，z/t 双报）
.venv/bin/python runs/t17w1/aggregate.py --pair K2VSC0 --seeds s1,s2,s3,s4,s5 --family confirmatory --dir runs/rk
.venv/bin/python runs/t17w1/aggregate.py --pair K7VSC0 --seeds s1,s2,s3,s4,s5 --family confirmatory --dir runs/rk
.venv/bin/python runs/t17w1/aggregate.py --pair winVSC0 --seeds s1,s2,s3,s4,s5 --family descriptive --dir runs/rk
.venv/bin/python runs/t17w1/aggregate.py --pair K0VSC0  --seeds s1,s2,s3,s4,s5 --family descriptive --dir runs/rk

# 3) vs random screen / V 探针 / 汇总分析（含 Holm 与 V 止损规则）
.venv/bin/python runs/rk/run_screen_rand.py
.venv/bin/python runs/rk/probe_values.py --episodes 200
.venv/bin/python runs/rk/analyze_saturate.py        # 写 runs/rk/saturate_results.json
```

## 11. 产物路径

| 内容 | 路径 |
|---|---|
| 训练 run（20 条新 + 5 条复用 C0） | `runs/ra_{K0,K2,K7,win}__{1..5}__*`、`runs/t17pool__{1,2,3}__1790439615`、`runs/ra_C0__{4,5}__*` |
| 训练命令/队列/日志 | `runs/rk_cmds/`、`runs/rk_queue.txt`、`runs/rk_orchestrator.sh`、`runs/rk_logs/`、`runs/rk_orchestrator.log` |
| h2h JSON / 逐局 games | `runs/rk/h2h_<pair>_s<seed>.json`（45 个）、`runs/rk/games_<pair>_s<seed>_seed<deal>.jsonl` |
| screen / V 探针 | `runs/rk/screen_<label>_rand.json`、`runs/rk/games_screen_*.jsonl`、`runs/rk/value_probe.json` |
| 汇总分析 | `runs/rk/saturate_results.json`、`runs/rk/analyze_saturate.py`、`runs/rk/run_h2h_matrix.py`、`runs/rk/run_screen_rand.py`、`runs/rk/probe_values.py`、`runs/rk/eval_pipeline.log` |
| 工作区基线 | `runs/rk/baseline_workspace.txt`（开工前 `git status` + diff sha256） |

## 12. 结论一句话

**分差饱和 `saturate` 的 confirmatory 主臂 K2（τ=0.2）在同 seed 配对 h2h 上合并
−18.31 Elo（z/t CI 全负，Holm p=6.9e-06，5/5 seed 同号）→ 触发 ≤+5 止损并关闭奖励饱和线；
剂量臂 K7（τ=0.7）+7.27 未判定，端点锚 K0 −27.72、`win` −6.77；无 V 塌缩（K2 raw V sd 0.073
但 EV +0.233，AND 不成立）；vs random 的 P_w 未牺牲、次端点 vs `l1_1M` 方向一致。因此本轮
不支持把任何饱和强度推向默认路径：在实测的两种形状（跳变 / 饱和）里，`terminal` 仍是默认。**

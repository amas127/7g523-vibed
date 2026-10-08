# 残差 body + 非残差头 + GroupNorm（`gnres`）500k 初步 screen（k=1）

> **状态：已完成（2026-10-07，k=1 = seed 1）。主结论：架构效应 = 0——两家族各自在较好
> init 下直接对比 `gnres21@0.1 − head21@0.01` = −0.14 [−12.62,+12.34]；首轮的 +37/+40
> 是 `--actor-out-std` 混淆。真正的发现是 `actor_out_std` × 头型 的强交互（残差头偏好 0.01、
> plain 深头偏好 0.1，各 ±18~22 Elo）；GN vs LN 不可分（+0.43）。**
>
> **operator 口径**：本 screen 的头深度固定为**策略头 2 层 / 价值头 1 层**（即 (a,c) = (2,1)，
> 仓库约定 `head<a><c>`：a=策略头，c=价值头）；变体 = **无残差头 + 有残差 body +
> LayerNorm → GroupNorm**。
>
> **口径**：revision-3（出空即撬底）+ 观测 v5 + [ADR-0016](../adr/0016-default-training-recipe-arcsin-lr-floor.md)
> 默认配方（arcsin α=0.5、λ=1.0、lr floor 1e-5）+ [ADR-0015](../adr/0015-continuation-pool-random-mix.md)
> pool-episode（6×t17 `3@` + `2@random`）。k=1 只作探索性读数；跨 fit 绝对 Elo 不可比。
>
> **产物**：`runs/gnres/`。**代码**：`src/seven523/networks.py`、`src/seven523/train.py`；
> 测试 `tests/test_networks.py`、`tests/test_train.py`。

## 0. 假设

头部深度网格（`head<a><c>`，残差头）唯一正号过线的是 `head12`、唯一负号过线的是 `head22`
（均 k=1、不构成结论）；head 设计 v2（`head32ln`：残差头内 pre-norm LayerNorm + `actor_out_std 0.1`）
500k 对旧网格 `head32` **−7.98** [−25.35,+9.40]，续训后仍无改进证据。operator 因此指定另一条
组合：把**残差挪到共享 body**，头退化为**非残差 plain MLP**，并把 body 的归一化从 LayerNorm
换成 **GroupNorm**。

- **H1**：在 (2,1) 深度下，`gnres21`（GN 残差 body + plain 头）相对 `head21`
  （现状：plain body + 残差 2 层策略头）或 `lnres21`（同 body 换 LN）有 ≥ +10 Elo 增益。
- **H0**：无。按统一口径，k=1 的 CI 半宽 ≈13，本 screen 只用于过筛/止损，不作确认性结论。

## 1. 架构定义（新 arch 家族）

`lnres<a><c>` / `gnres<a><c>`，a,c ∈ {1,2,3}（实现全网格，实验只跑 (2,1)）：

- **body** = 现有 `ResidualTrunk` 拓扑（3 个 block：block 0 为 `[Linear, Norm, Act]`；
  block 1/2 为残差 `h ← h + [Linear, Norm, Act](h)`），Norm 为 `LayerNorm(hidden)`（`lnres`）
  或 `GroupNorm(gn_groups, hidden)`（`gnres`，默认 8 组、须整除 hidden）。归一化位置、
  block 数、参数量与 `deep_lnres` 完全一致：GN 只替换 norm 类型，是单因子对照。
- **头** = `PlainHead`：`layers − 1` 个 `[Linear, Act]` 隐藏块 + 输出层，无 skip、无 Norm；
  1 层头仍走历史 plain `Linear`。参数量与 `head<a><c>` 相同（残差头不加参数），家族差
  仅 body：`gnres<a><c>`/`lnres<a><c>` = `head<a><c>` + 17,280 参数（body 3 层与 2 层之差
  + 3 个 norm 的 2×hidden）。
- `(2,1)` 总参数 = **88,971**（`gnres11` = 72,459，`head21` = 71,691）。

## 2. 臂（3 run + 既有参照；k=1 = seed 1，500k 从零）

| arm | `--arch` | 头（策略/价值） | 结构 | run 名 |
|---|---|---|---|---|
| **变体** | `gnres21` | 2 / 1 | GN(8) 残差 body + plain 头 | `gnres21` |
| 归一化对照 | `lnres21` | 2 / 1 | LN 残差 body + plain 头 | `lnres21` |
| **基线（现状设计同深度）** | `head21` | 2 / 1 | plain body + 残差策略头 | `head21` |

- 三臂统一 `--actor-out-std 0.1`（head 设计 v2 的深头 init 口径；plain 2 层头同样受输出层
  小 init 影响），其余超参与 head-depth H2 一致（Adam、linear anneal、8×128、minibatch 4、
  epochs 4、lr 2.5e-4、`--eval-interval 50`）。
- 既有参照（不重训，作 context）：`vfo_base`（`shared`，(1,1)，同配方 seed 1 500k，
  `runs/vf-outcome/vfo_base__1__1791403246/agent.pt`）、`hd_head32ln`（v2 旗舰 (3,2)，
  同配方 seed 1 500k，`runs/head-depth/hd_head32ln__1__1791391938/agent.pt`）。
- 从零训练，跨 arch 热启动保持 fail-loud（不 bridge）。

## 3. 端点（h2h：3 deal seed 9600/9601/9602 × 400 副 × 换座，bootstrap 4000）

1. **主端点**：`gnres21 − head21`（新 body+GN+plain 头 vs 同深度现状设计）；
2. **归一化**：`gnres21 − lnres21`（GN vs LN，其余全同）；
3. context：`gnres21 − vfo_base`（vs 当前默认 (1,1)）、`head21 − vfo_base`、
   `gnres21 − hd_head32ln`（vs v2 旗舰）、`lnres21 − head21`。

**判定**（统一口径）：CI 排除 0 且点 ≥ +10 才谈「值得行动」并标记待补 seed 2–3；
点 < +5 止损；k=1 不宣布 H1/H0。机制读数（非门）：`metrics.csv` 的 EV、entropy、
maxKL、evalW、value_loss；两臂 group-norm 参数只差 norm 类型。

## 4. 实现映射

| 文件 | 改动 |
|---|---|
| `networks.py` | `ResidualTrunk(gn_groups=...)`（与 `ln` 互斥）；`PlainHead`（无 skip、无 norm）；`lnres<a><c>`/`gnres<a><c>` 全网格；`Agent(gn_groups=8)`（校验整除）；checkpoint 写 `gn_groups`；`gnres` 间热启动 group 数不一致 fail-loud |
| `train.py` | `--arch` choices/help 扩到 17 个新 arch；`--gn-groups`（默认 8，启动前校验 `hidden % groups`） |
| 测试 | 新家族参数量/结构（body 3 block、norm 类型、plain 头无 skip）、`deep_lnres` 与 `gnres11` 参数量一致、GN 组数校验、checkpoint 往返、热启动 guard、`--gn-groups` 冒烟 |

## 5. 结果

### 5.1 训练读数（3 arm @ `--actor-out-std 0.1`，各 488/488 update、无 NaN）

| arm | 末次 vs-random eval（ret / score / win） | 后半（>400）ep_return | 后半 EV | 末段 entropy |
|---|---:|---:|---:|---:|
| `gnres21` | 0.508 / 75.4 / 0.87 | 0.112 | 0.617 | 1.469 |
| `lnres21` | 0.563 / 78.2 / 0.89 | 0.112 | 0.625 | 1.486 |
| `head21` | 0.501 / 75.1 / 0.86 | 0.009 | 0.612 | 1.460 |

vs-random eval 三臂同带（0.86–0.89，早已饱和），分不出 h2h 的差距；训练 16:24:27–16:40:21。

### 5.2 h2h（k=1 = seed 1；3 deal seed × 400 × 换座，1200 副 / 2400 局）

| 比较 | Δ Elo | 95% CI | winrate | per-seed |
|---|---:|---|---:|---|
| **`gnres21 − head21`** | **+37.21** | **[+24.84, +49.57]** | 0.553 | +37.1 / +32.2 / +42.3 |
| **`lnres21 − head21`** | **+40.14** | **[+27.42, +52.86]** | 0.558 | +34.0 / +45.4 / +41.0 |
| `gnres21 − lnres21` | +0.43 | [−11.66, +12.53] | 0.501 | −3.9 / +6.5 / −1.3 |
| `gnres21 − vfo_base` | +8.84 | [−11.44, +29.13] | 0.513 | +14.8 / −11.3 / +23.1 |
| `head21 − vfo_base` | **−37.27** | **[−57.93, −16.62]** | 0.447 | −30.9 / −57.9 / −23.1 |
| `gnres21 − hd_head32ln` | +10.89 | [−10.78, +32.55] | 0.516 | +32.2 / −4.8 / +5.2 |

### 5.3 初步读法与 init 混淆（Amendment G1）

- 新家族（GN/LN 残差 body + plain 头）在 (2,1) 下**大幅优于 `head21`**（+37/+40，CI 排除 0、
  三 deal seed 同号）；GN 与 LN 不可分（+0.43）。
- 但本 screen 三臂都用 **`--actor-out-std 0.1`**：那是 head 设计 v2（深 LN 残差头）的 init；
  `head21` 的 canonical 口径是 **0.01**（head-depth 网格）。0.1 会让 2 层残差头输出层 init
  放大 10×，可能是 `head21` 变弱的真正原因，而非「残差头设计差」。
- 因此 +37 不能当作架构结论。**Amendment G1（预注册）**：同 seed/配方/pool 补两个臂
  `head21_std01`、`gnres21_std01`（其余全同，只把 `--actor-out-std` 换成 0.01），端点：
  `head21_std01 − head21`（init 效应）、`gnres21_std01 − head21_std01`（canonical init 下的
  架构效应）、`gnres21_std01 − gnres21`（新臂的 init 效应）、`head21_std01 − vfo_base`
  （canonical baseline vs 默认）。判定口径不变（CI 排 0 且点 ≥ +10 → 标记补 seed）。
  另外记录：`gnres21 − vfo_base` 与 `gnres21 − hd_head32ln` 都只有 +9~+11、CI 含 0，
  即新家族目前**没有**超过当前默认或 v2 旗舰的证据。

### 5.4 Amendment G1 结果（seed 1，`--actor-out-std 0.01` 两臂 16:44–16:56 完成，无 NaN）

| 比较（左 − 右） | Δ Elo | 95% CI | winrate | per-seed |
|---|---:|---|---:|---|
| `gnres21@0.1 − head21@0.01`（各自较好 init） | **−0.14** | [−12.62, +12.34] | 0.500 | −9.1 / −3.5 / +12.2 |
| `head21@0.01 − head21@0.1`（残差头 init 效应） | **+18.26** | [+5.75, +30.76] | 0.526 | +18.7 / +16.5 / +19.6 |
| `gnres21@0.1 − gnres21@0.01`（plain 头 init 效应） | **+22.33** | [+10.31, +34.35] | 0.532 | +26.1 / +16.1 / +24.8 |
| `gnres21@0.01 − head21@0.01`（同 init 架构对照） | −13.33 | [−25.75, −0.91] | 0.481 | −21.3 / −5.2 / −13.5 |
| `gnres21@0.01 − vfo_base` | **−23.36** | [−35.30, −11.41] | 0.467 | −25.2 / −12.2 / −32.7 |
| `head21@0.01 − vfo_base` | +2.46 | [−9.89, +14.81] | 0.504 | +13.0 / +0.4 / −6.1 |

（G1 臂训练读数：`head21_std01` 后半 EV 0.601 / 末次 vs-random 0.84；`gnres21_std01`
0.626 / 0.90，均与 0.1 臂同带。）

### 5.5 最终读法

1. **主结论：架构效应 = 0。** 两家族各自在较好的 `--actor-out-std` 下直接对比
   `gnres21@0.1 − head21@0.01` = **−0.14** [−12.62,+12.34]；+37/+40 完全是 init 混淆。
2. **真正的效应是 `actor_out_std` × 头型 的强交互**（3 deal seed 同号）：
   - plain 多层头需要大 init：`gnres21@0.1 − @0.01` = **+22.33**；无 skip 时隐藏层
     只能靠输出层回传，0.01 会把隐藏层梯度整体压小；
   - 2 层残差头相反：`head21@0.01 − @0.1` = **+18.26**；残差 identity 路径已经
     保证信号/梯度通路，10× 输出增益反而变差（机制未单独测量）。
3. **GN vs LN 不可分**（+0.43，CI 含 0）；即归一化的分组结构在这个 MLP body 上无信号。
4. **相对现状**：canonical init 下 `head21@0.01 − vfo_base` = +2.46（中性）；新家族 0.1 臂
   对 `vfo_base` +8.84、对 `hd_head32ln` +10.89，均 CI 含 0；新家族 0.01 臂 −23.36。
   即所有臂都没有超过当前默认/v2 旗舰的可用证据。
5. 按统一口径：无任何端点同时满足「CI 排 0 且点 ≥ +10」→ **不补 seed 2–3，本线收束**。
   （- 对既有实验的一个事后提示：v2 的 `head32ln` 也用了 `actor_out_std 0.1`，而那是**残差**
   头；若交互方向可外推，H2 的 −7.98 可能有一部分来自 init 而不是 LN 设计。未验证。）

## 6. 下一步

- 新家族（`lnres`/`gnres`）保留为 opt-in arch，不进入默认；不再排 500k 组合臂。
- 可复用的机制读数：`actor_out_std` 对残差/plain 深头是**反向**的旋钮。以后任何深头实验都
  应把 init 与结构一起做因子，或至少在两种 init 下各跑一臂，否则会把 init 效应误读为结构效应
  （本 screen 的首轮 +37 就是例证）。
- 产物：`runs/gnres/`（5 run + 12 条 h2h JSON）；复现脚本 `runs/gnres/run_screen.sh`、
  `run_followup.sh`、`run_h2h.sh`、`run_h2h_g1.sh`。
- **后续（2026-10-07）**：operator 指定把 GN 换成 BatchNorm 并配三头（`--vf-outcome`）重跑；
  BN 强负（−42 vs GN/LN，entropy 塌缩），三头本身 null。见
  [`bnres-500k.md`](./bnres-500k.md)。


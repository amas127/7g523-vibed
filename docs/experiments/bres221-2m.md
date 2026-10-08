# 小模型 bres221：瓶颈残差块 (32, 96, 32) + depth 2，hidden 32，2M 从 random（k=1）

> **状态：已完成（2026-10-08，k=1 = seed 1）。主读数：17,132 参数、2-block body 落在 raw 顶簇——
> 对最强 raw `head32ln5m` **+2.03** [−9.49,+13.55]（平）、对同块 depth-4 `bres421`
> **+14.63** [+2.51,+26.75]（3/3 seed 同号，p=0.019）、对 `res421_h64_2m`（(64,64) 块、
> hidden 64、36,108 参数）**−1.16** [−12.51,+10.19]（平）、对旧配方 `t17long` 2M
> +24.65 [+12.41,+36.90]。**
>
> **边界**：对 `bres421` 的 +14.6 是干净的 depth 4→2 消融（块相同）；对 `res421_h64_2m` 是
> 复合差异（流宽/块形状/深度/参数量）。k=1 未复现。
>
> **operator 请求（2026-10-08）**：残差块 depth=2 + (32, 96, 32)，再跑 2M vs randombot。
>
> **口径**：revision-3 + obs v5 + [ADR-0016](../adr/0016-default-training-recipe-arcsin-lr-floor.md)
> 默认配方；对手 = `random`（不混池）；h2h 3 deal seed（9600/9601/9602）× 400 副 × 换座，
> bootstrap 4000。跨 fit 绝对 Elo 不可比，本页只给同 fit h2h。
>
> **产物**：`runs/bres/bres221_32x96x32_2m__1__1791439986/`。**代码/测试**：见
> [bres-2m.md](./bres-2m.md) §0（`bres<d><a><c>` 家族；本次把 depth-2 格点 `bres2<a><c>`
> 补进 `--arch` choices，并加 `test_bres221_depth_two_body`）。

## 0. 架构（`bres221` + `--vf-outcome`，`--hidden-size 32 --res-expansion 3`）

```
x (161)
  └─ ResidualTrunk（depth 2，无 norm，共享）
        block0      : Linear(161→32) + ReLU
        block1      : h = h + ReLU(Linear(96→32)(ReLU(Linear(32→96)(h))))
  └─ h (32)
        ├─ actor   : PlainHead [Linear(32→32)+ReLU] → Linear(32→138)
        ├─ critic  : Linear(32→1)
        └─ outcome : Linear(32→1) → tanh          (--vf-outcome)
```

- 总参数 **17,132**（body 11,456 + actor 5,610 + critic 33 + 胜负头 33）；同块 depth-4
  `bres421` 是 29,676，(64,64) 块 depth-4 的 `res421_h64_2m` 是 36,108。
- 与 `bres421` 逐模块相同，只是少了两个残差块（depth 2 = 1 投影 + 1 个瓶颈残差块）。
- 三个头仍读同一个 32 维 hidden；无 norm/dropout，两次 trunk 前向数值一致。

## 1. 训练

配置：seed 1、`--opponent random`、2M 步、8×128（batch 1024 / minibatch 256）、4 epochs、
Adam 2.5e-4 线性退火（floor 1e-5）、arcsin α=0.5 + λ=1、`--vf-outcome true`、
`--actor-out-std 0.1`、hidden 32、TensorBoard 开、`--eval-interval 50`。

- 02:13:06 → 02:39:10（约 26.1 分钟），1953/1953 updates = 1,999,872 步，**无 NaN**，rc=0。
- 末 150 updates 均值：EV **0.6715**、entropy 1.6487、value_loss 0.0135、胜负头准确率
  **0.9145**；末行 episodic_return 1.1913 / entropy 1.6161 / outcome_loss 0.2362 /
  outcome_acc 0.8767。（对比：`bres421` EV 0.6644 / entropy 1.7348 / acc 0.9083；
  `res421_h64_2m` EV 0.6645 / entropy 1.6103 / acc 0.9073。）
- 训练内末次 vs-random eval（100 局）：return 0.57、score 78.5、diff 57.0、win 0.86。
- 独立 vs-random eval（1000 局，gpu）：**win 0.898**、score 78.69、diff 57.38
  （`bres421` 0.889 / 78.03 / 56.06；`res421_h64_2m` 0.909 / 79.18 / 58.36）。

## 2. h2h 面板

协议：3 deal seed（9600/9601/9602）× 400 副 × 换座（1200 副 / 2400 局），bootstrap 4000：
`bres221_32x96x32_2m − 对手`：

| 对手 | Δ Elo | 95% CI | winrate | per-seed |
|---|---:|---|---:|---|
| `bres421_32x96x32_2m`（同块，depth 4） | **+14.63** | **[+2.51,+26.75]** | 0.521 | +21.7 / +11.7 / +10.4 |
| `head32ln5m`（当前最强 raw，8.5M 续训） | +2.03 | [−9.49,+13.55] | 0.503 | +7.8 / −3.5 / +1.7 |
| `res421_h64_2m`（(64,64) 块、hidden 64、depth 4） | −1.16 | [−12.51,+10.19] | 0.498 | −10.4 / +4.8 / +2.2 |
| `t17long` 2M（同预算，旧配方 + shared h128） | +24.65 | [+12.41,+36.90] | 0.535 | +23.9 / +24.8 / +25.2 |

## 3. 读法

1. **depth 2 > depth 4（同块）**：+14.63 [+2.51,+26.75]、3/3 seed 同号、deal-sign p=0.019。
   32 宽的瓶颈流在 4 块时可能过深/信号被稀释，2 块时反而合适（单假设，需 seed 复核）。
2. **17k 参数打平 36k 父模型**：对 `res421_h64_2m` −1.16 [−12.51,+10.19]；训练 EV（0.6715
   vs 0.6645）与独立 vs-random（0.898 vs 0.909）也在同一水平。
3. **拉回顶簇**：对 `head32ln5m` +2.03 [−9.49,+13.55]（depth-4 是 −13.62），一个 17k 参数的
   2-block 模型和当前最强 raw 统计上打平。
4. **配方效应仍占主导**：对旧配方 `t17long` +24.65，三条 deal seed 读数几乎一致
   （+23.9/+24.8/+25.2）。
5. **k=1**：未做 seed 2–3；按仓库口径不宣布结构结论。

## 4. 下一步

1. **seed 2–3 复现 `bres221`**（2 × ≈26 分钟，可并行）——这是继续这条方向的前置。
2. 若复现，拆 (32,96,32) 的甜点：depth 3、`--res-expansion` 2/4（mid 64/128）、hidden 48/64
   @ depth 2；或直接接续训（ADR-0015 配方）看能否越出顶簇。
3. 反向对照：`res421` @ hidden 32 / depth 2（plain 32→32 块），分离「瓶颈分支」与「窄流 + depth」。

**追加（2026-10-08）**：operator 选了「hidden 48 + expansion 2 + bigbatch/2-epoch」的组合
（`bres221_48x96x48_b2048_2m`），对本臂 **−3.04** [−15.05,+8.97]（平）；块宽与 PPO 配方
双改动、未拆因子，见 [bres221-h48-b2048-2m.md](./bres221-h48-b2048-2m.md)。

**追加 2（2026-10-08）**：本臂直接续训 10M（ADR-0015 池 + AdamW/cosine，每 1M 快照）跑完，
对**本臂起点 +16.81** [+4.95,+28.68]（3/3 seed 正，p=0.002）、对 `head32ln5m` +12.61——
首次越出顶簇，见 [bres221-cont10m.md](./bres221-cont10m.md)。

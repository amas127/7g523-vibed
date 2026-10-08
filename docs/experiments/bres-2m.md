# 小模型 bres4：无 norm 4-block 瓶颈残差 body + plain 三头，hidden 32，2M 从 random（k=1）

> **状态：已完成（2026-10-08，k=1 = seed 1）。主读数：直接消融为负——对同配方同预算的
> 父模型 `res421_h64_2m` **−17.25** [−28.62,−5.87]（3/3 deal seed 同号，deal-sign p=0.005）；
> 对当前最强 raw `head32ln5m` **−13.62** [−25.73,−1.50]；对旧配方同预算 `t17long` 2M
> +18.70 [+6.02,+31.38]。把残差块从 `(64, 64)` 换成 `(32, 96, 32)` 在这条配方下**没有收益，
> 且相对父模型显著变弱**。**
>
> **边界**：父模型 hidden 64、新模型 hidden 32，(32,96,32) 的块强制残差流宽 = 32，所以这条
> 消融同时混了「瓶颈分支」与「窄残差流」两个因素；训练内 EV 两者相同（0.6644 vs 0.6645），
> 差异在损失/EV 尺度看不出来。k=1 未复现，拆因 arm 见 §4。
>
> **后续（同日）**：同块 depth=2（`bres221`）跑完后转正——对 depth 4 **+14.63**
> [+2.51,+26.75]、与 `head32ln5m`/`res421_h64_2m` 打平，见
> [`bres221-2m.md`](./bres221-2m.md)。本页 depth-4 的负读数不变，方向未关闭。
>
> **operator 请求（2026-10-08）**：按 421 头的配方训一个新架构，残差块从 `(64, 64)` 改成
> `(32, 96, 32)`。
>
> **口径**：revision-3 + obs v5 + [ADR-0016](../adr/0016-default-training-recipe-arcsin-lr-floor.md)
> 默认配方；对手 = `random`（不混池）；h2h 3 deal seed（9600/9601/9602）× 400 副 × 换座，
> bootstrap 4000。跨 fit 绝对 Elo 不可比，本页只给同 fit h2h。
>
> **产物**：`runs/bres/`。**代码**：`src/seven523/networks.py`（`bres<d><a><c>` 家族 +
> `ResidualTrunk.mid`）、`src/seven523/train.py`（`--res-expansion`）；测试
> `tests/test_{networks,train}.py`。

## 0. 架构（`bres421` + `--vf-outcome`，`--hidden-size 32 --res-expansion 3`）

```
x (161)
  └─ ResidualTrunk（depth 4，无 norm，共享）
        block0      : Linear(161→32) + ReLU
        block1..3   : h = h + ReLU(Linear(96→32)(ReLU(Linear(32→96)(h))))
  └─ h (32)
        ├─ actor   : PlainHead [Linear(32→32)+ReLU] → Linear(32→138)
        ├─ critic  : Linear(32→1)
        └─ outcome : Linear(32→1) → tanh          (--vf-outcome)
```

- 总参数 **29,676**（body 24,000 + actor 5,610 + critic 33 + 胜负头 33）；父 `res421_h64_2m`
  （(64,64) 块、hidden 64）是 36,108。
- 三个头仍读**同一个** 32 维 hidden（共享 body，非 `towers`）；无 norm/dropout，两次 trunk
  前向数值一致。
- 实现：`bres<d><a><c>` = `res<d><a><c>` 家族 + 每个残差块 `hidden → mid → hidden`
  （`mid = --res-expansion × --hidden-size`，默认 3 → 32/96/32）；block0 投影与 plain 头
  不变。`res_expansion` 写进 checkpoint（同 arch 热启动倍数不一致时 fail-loud）；其余 arch
  忽略该开关。测试：`test_bres421_bottleneck_block_shape_and_outcome_head`、
  `test_bres_expansion_validation_and_checkpoint_round_trip`、
  `test_res_expansion_is_ignored_by_the_plain_res_body`、`test_train_bres_vf_outcome_smoke`；
  全套 922 passed / 14 skipped（1 个既有 manifest 漂移失败与本项无关，已 deselect）。

## 1. 训练

配置：seed 1、`--opponent random`、2M 步、8×128（batch 1024 / minibatch 256）、4 epochs、
Adam 2.5e-4 线性退火（floor 1e-5）、arcsin α=0.5 + λ=1、`--vf-outcome true`、
`--actor-out-std 0.1`、hidden 32、TensorBoard 开、`--eval-interval 50`。

- 01:36:41 → 02:04:17（约 27.6 分钟），1953/1953 updates = 1,999,872 步，**无 NaN**，rc=0。
- 末 150 updates 均值：EV 0.6644、entropy 1.7348、value_loss 0.0135、胜负头准确率 0.9083；
  末行 episodic_return 1.0642 / entropy 1.6913 / outcome_loss 0.1964 / outcome_acc 0.8712。
- 训练内末次 vs-random eval（100 局）：return 0.572、score 78.6、diff 57.2、win 0.91。
- 独立 vs-random eval（1000 局，gpu）：**win 0.889**、score 78.03、diff 56.06；
  对照 `res421_h64_2m`：win 0.9090、score 79.18、diff 58.36（vs-random 已饱和，不是判据）。

## 2. h2h 面板

协议：3 deal seed（9600/9601/9602）× 400 副 × 换座（1200 副 / 2400 局），bootstrap 4000：
`bres421_32x96x32_2m − 对手`：

| 对手 | Δ Elo | 95% CI | winrate | per-seed |
|---|---:|---|---:|---|
| `res421_h64_2m`（直接父模型，同配方/预算，hidden 64） | **−17.25** | **[−28.62,−5.87]** | 0.475 | −25.2 / −12.2 / −14.3 |
| `head32ln5m`（当前最强 raw，8.5M 续训） | **−13.62** | **[−25.73,−1.50]** | 0.480 | −12.2 / −20.0 / −8.7 |
| `t17long` 2M（同预算，旧配方 + shared h128） | +18.70 | [+6.02,+31.38] | 0.527 | +9.6 / +20.4 / +26.1 |
| `w2m_ctl`（2M pool，旧顶簇） | +8.98 | [−2.88,+20.84] | 0.513 | +4.8 / +10.4 / +11.7 |
| `o2c_base`（续训顶簇） | +10.73 | [−4.59,+26.05] | 0.515 | +12.2 / −3.5 / +23.5 |

## 3. 读法

1. **直接消融为负**：与父模型唯一不同的是块的形状与残差流宽（(32,96,32) 的块强制流宽 32）。
   −17.25 Elo、3/3 seed 同号、deal-sign p=0.005、CI 排除 0；between-seed sd 7.0，负读数大于
   训练 seed 噪声。
2. **训练曲线看不见**：EV 0.6644 vs 0.6645、outcome acc 0.9083 vs 0.9073、entropy 1.73 vs
   1.61——损失/EV 与 h2h 强度脱钩，只有 h2h 能分辨这条结构差异。
3. **配方效应仍占主导**：对旧配方 `t17long` +18.7、对 `w2m_ctl`/`o2c_base` +9~+11，说明
   「新默认配方 + 小模型」的大方向仍有优势；这个块形状本身没有把 raw 顶簇推得更高
   （父模型对 `head32ln5m` 是 −5.5，新块变成 −13.6）。
4. **k=1**：未做 seed 2–3；按仓库口径（CI 排 0 且点 ≥ +10 才标记复现），这条负读数标记为
   「负向待复核」，不宣布结构结论。

## 4. 下一步

**已执行（2026-10-08）**：同块 depth=2（`bres221`）已跑，对深度 4 **+14.63**
[+2.51,+26.75]、对 `head32ln5m` 与 `res421_h64_2m` 打平——详见
[bres221-2m.md](./bres221-2m.md)。depth-4 这个 arm 的负读数不变，但方向未关闭。

其余待做：

1. seed 2–3 复现 `bres221`（2 × ≈26 分钟，可并行）；
2. 反向对照 `res421` @ hidden 32 / depth 2（plain 32→32 块），分离「瓶颈分支」与「窄流 + depth」；
3. `bres421` @ hidden 64（64→192→64）在等流宽下测瓶颈分支（低优先，depth-2 已给正向信号）。

# BatchNorm 残差 body + 三头（策略+价值+胜负）500k 初步 screen（k=1）

> **状态：已完成（2026-10-07，k=1 = seed 1）。结论：BatchNorm 强负（vs GN/LN −42 Elo，
> 三个 deal seed 同号、CI 远排 0），机制 = entropy 塌缩（0.10 vs 1.6）+ rollout(8 样本)/
> update minibatch(256) 的 BN 统计不一致；GN/LN 三头与现有三头默认 null（−3.3 / −5.1）。**
>
> **operator 口径**：本实验 = 上一组 [gnres-500k.md](./gnres-500k.md) 的残差 body + 非残差头
> 家族，但 (1) 归一化把 **GroupNorm 换成 BatchNorm**，(2) 架构用**三头**模型
> （策略 + 价值 + 胜负，即 `--vf-outcome`，[vf-outcome.md](./vf-outcome.md)）。头深度沿用
> operator 指定的基线 **(策略 2 层 / 价值 1 层)**，所有臂 `--actor-out-std 0.1`
> （plain 深头需要的 init；见 gnres-500k.md §5.5）。
>
> **口径**：revision-3 + obs v5 + [ADR-0016](../adr/0016-default-training-recipe-arcsin-lr-floor.md)
> 默认配方 + [ADR-0015](../adr/0015-continuation-pool-random-mix.md) pool-episode。k=1 只作
> 探索性读数；跨 fit 绝对 Elo 不可比。
>
> **产物**：`runs/bnres/`。**代码**：`src/seven523/networks.py`、`src/seven523/train.py`；
> 测试 `tests/test_networks.py`、`tests/test_train.py`。

## 0. 假设

- **H1**：在三头架构下，`bnres21`（BatchNorm 残差 body）相对 `gnres21`/`lnres21`
  （同 body 换 GN/LN）或现有三头默认 `vfo_outcome` 有 ≥ +10 Elo 增益。
- **H0**：无。k=1 的 h2h CI 半宽 ≈13，只作过筛/止损。

## 1. 架构与已知风险

- `bnres<a><c>`：与 `lnres`/`gnres` 完全同拓扑（3-block 残差 body：block 0 =
  `[Linear, Norm, Act]`，block 1/2 = `h + [Linear, Norm, Act](h)`；头为 plain `PlainHead`），
  只把每块的 `nn.GroupNorm` 换成 `nn.BatchNorm1d(hidden)`（affine、track_running_stats）。
  参数量与 LN/GN 版本一致（每块 2×hidden）。
- **三头**：`--vf-outcome true` 在同一 body 上追一个 `tanh` 胜负头，critic 学跳变-free
  return，`info["reward_jump"]` 由 env 发布（详见 vf-outcome.md）。
- **PPO × BN 已知风险（要量的正是它）**：rollout 每步 batch = `num_envs` = 8，train 模式下
  BatchNorm 用 8 样本统计；PPO update 的 minibatch = 256，统计不同 → old/new logprob 的
  比值可能被放大（`approx_kl` / clipfrac 会显示）。不额外 mitigation，先测朴素实现。
- **训练循环修复**：`NeuralPolicy(agent)` 会把共享的 learner 置 `.eval()`；历史上对无 BN
  网络无影响，但 BN 下会让 running stats 从第一次周期 eval 起冻结。现在 eval 后
  `agent.train()` 恢复（回归测试 `test_train_restores_train_mode_after_the_periodic_eval`）；
  frozen 对手是 deep copy，不受影响。
- checkpoint 包含 BN running stats；同 arch 热启动整表加载（含统计），跨 arch 仍 fail-loud；
  eval 路径（`NeuralPolicy`）走 running stats。

## 2. 臂（3 新 run + 既有参照；k=1 = seed 1，500k 从零）

| arm | `--arch` | 三头 | 结构 | 备注 |
|---|---|---|---|---|
| **变体** | `bnres21` | `--vf-outcome true` | BN 残差 body + plain (2,1) 头 | `--actor-out-std 0.1` |
| 归一化对照 | `gnres21` | `--vf-outcome true` | GN 残差 body + plain (2,1) 头 | 同上 |
| 归一化对照 | `lnres21` | `--vf-outcome true` | LN 残差 body + plain (2,1) 头 | 同上 |
| 参照（不重训） | `shared` (1,1) | `--vf-outcome true` | 现有三头默认 | `runs/vf-outcome/vfo_outcome__1__1791403246/agent.pt` |

其余超参与 head-depth H2 / gnres screen 一致（Adam、linear anneal、8×128、minibatch 4、
epochs 4、lr 2.5e-4、`--eval-interval 50`）。

## 3. 端点（h2h：3 deal seed 9600/9601/9602 × 400 副 × 换座，bootstrap 4000）

1. **主端点**：`bnres21 − gnres21`（BN vs GN，其余全同）；
2. `bnres21 − lnres21`（BN vs LN）；
3. `bnres21 − vfo_outcome`（vs 现有三头默认，注意后者是 (1,1) 头深）；
4. context：`gnres21 − vfo_outcome`、`lnres21 − gnres21`、`bnres21 − vfo_base`（后者是无
   胜负头的同配方 (1,1) 单头模型，用来量「三头」本身）。

**判定**（统一口径）：CI 排除 0 且点 ≥ +10 才谈「值得行动」并标记待补 seed 2–3；点 < +5
止损；k=1 不宣布 H1/H0。机制读数（非门）：`metrics.csv` 的 `approx_kl`/`clipfrac`（BN 批次
统计不一致的直接信号）、EV、entropy、`outcome_loss/outcome_accuracy`、evalW。

## 4. 实现映射

| 文件 | 改动 |
|---|---|
| `networks.py` | `ResidualTrunk(norm=...)`（`none`/`layer`/`group`/`batch`，`gn_groups` 只配 group）；`bnres<a><c>` 全网格（9 cell）；头/参数量与 `gnres` 一致 |
| `train.py` | `--arch` choices 扩 9 个 `bnres*`；周期 eval 后恢复 `agent.train()` |
| 测试 | `bnres` 参数量/结构、running stats 更新与 checkpoint 往返、`bnres21 + --vf-outcome` 冒烟、eval 后 train-mode 回归 |

## 5. 结果

三臂各 499,712 步、488/488 update、无 NaN；训练 17:12:26–17:30:13。

### 5.1 训练读数（>400 update 均值）

| arm | entropy | explained_variance | clipfrac | outcome_accuracy |
|---|---:|---:|---:|---:|
| `bnres21` | **0.103** | 0.485 | 0.023 | 0.709 |
| `gnres21` | 1.606 | 0.641 | 0.010 | 0.729 |
| `lnres21` | 1.586 | 0.636 | 0.012 | 0.726 |

早期（~62k）`clipfrac`：`bnres21` **0.277** vs `gnres21` 0.107 / `lnres21` 0.051；
BN 臂的 entropy 从 ~0.46（180k）径直塔到 0.10，另两臂稳定在 1.35–1.72。

### 5.2 h2h（k=1 = seed 1；3 deal seed × 400 × 换座，1200 副 / 2400 局）

| 比较（左 − 右） | Δ Elo | 95% CI | winrate | per-seed |
|---|---:|---|---:|---|
| **`bnres21 − gnres21`** | **−41.97** | **[−61.15, −22.78]** | 0.440 | −30.5 / −61.4 / −34.0 |
| **`bnres21 − lnres21`** | **−42.78** | **[−55.44, −30.12]** | 0.439 | −47.6 / −41.5 / −39.3 |
| `bnres21 − vfo_outcome`（三头默认） | −33.85 | [−46.46, −21.24] | 0.452 | −27.0 / −30.5 / −44.1 |
| `bnres21 − vfo_base`（无胜负头） | −46.35 | [−61.73, −30.97] | 0.434 | −31.4 / −49.9 / −57.9 |
| `gnres21 − vfo_outcome` | −3.33 | [−15.56, +8.90] | 0.495 | −7.8 / −4.8 / +2.6 |
| `lnres21 − gnres21` | −5.07 | [−16.96, +6.83] | 0.493 | −3.9 / −7.8 / −3.5 |
| `gnres21(三头) − gnres21(gnres screen 无胜负头)` | +2.17 | [−9.73, +14.07] | 0.503 | −4.3 / +6.5 / +4.3 |
| `lnres21(三头) − lnres21(无胜负头)` | −4.49 | [−16.30, +7.32] | 0.494 | −15.7 / +3.0 / −0.9 |

（后两行只差 `--vf-outcome`；两侧都是 `--actor-out-std 0.1`、LN/GN 同 body。）

### 5.3 读法

1. **H1 拒绝：BN 是强负效应。** 三个 deal seed 全同号、CI 远排 0，−42（vs GN/LN）/
   −34（vs 三头默认）/−46（vs 无胜负头）。
2. **机制清楚且与预注册的风险一致**：BN 臂 entropy 塔到 **0.103**（GN/LN 1.59–1.61），
   早期 clipfrac 冲到 0.277——rollout 每步 batch=8 的 BatchNorm 统计与 PPO update
   minibatch=256 的统计不一致，old/new logprob 的比值被放大，clip 频繁 → 策略过快确定化。
   这不是「BN 这个算子不行」，而是「朴素的 train-mode BatchNorm + 这个 PPO 采集/更新配方」
   不行；要做公平测试必须先解决统计一致性（rollout 用 running stats + 显式统计更新、
   或 SyncBN/更大 rollout batch/冻结统计），成本高于本轮。
3. **GN/LN 三头对现有三头默认 null**（−3.3 / −5.1，CI 含 0）；**三头本身在新家族内也 null**
   （+2.2 / −4.5），与 vf-outcome 实验的 `vfo_outcome − vfo_base` = −0.43 一致。
4. 按统一口径：无任何正端点；BN 方向不补 seed，收束。`bnres` arch 保留为 opt-in。

## 6. 下一步

- 不在当前 PPO 配方下继续 BN；若未来重开，先做统计一致性的实现方案并单独预注册。
- 三头 + `gnres`/`lnres` 与默认三头无差异，不进入默认；三头本身也无增益（延续 vf-outcome 结论）。
- 产物：`runs/bnres/`（3 run + 8 条 h2h JSON）；复现脚本 `runs/bnres/run_screen.sh`、
  `run_h2h.sh`。

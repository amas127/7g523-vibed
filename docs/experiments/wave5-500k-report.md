# Wave 5：统一 500k 步预算重跑与重新定级（7鬼523）

> 资产状态（2026-09-25）：本报告引用的模型 checkpoint 已删除，旧路径不再可用；数值为牌型族规则变更前口径。

> **状态：训练 6/6 完成、座位配对 Elo 三 seed 定级完成（2026-09-25）。结论：
> 没有任何方案显著超过 `base680k` 或同批控制组 `w5_ctrl`。**
> 在修正后的训练语义（`AutoresetMode.SAME_STEP`）与修正后的评估语义
> （`plan_games` 候选内换座配对）下，`bigbatch / pool4g / vf1 / mix_samp /
> scratch` 全部 500k 步续训（scratch 从零）的合并 3-seed Elo 与 `base680k`
> 的差都在 0.7 个合并 SE 之内，最大名义优势是 `w5_vf1` 的 +4.2（0.16σ）。
> 一个必须记录的发现：**父模型与续训子代之间的偏移在不同牌集（seed）间会
> 整体翻符号**（seed 0 子代平均 −22.1、seed 1 +24.7、seed 2 −27.5），
> 因此本报告以同 fit 的相对读数和合并三 seed 为准，单 fit 的父子结论不可信。
> 无 PENDING，全部产物已落盘。

相关只读报告：[`ladder-report.md`](./ladder-report.md)、
[`elo-reliability-audit.md`](./elo-reliability-audit.md)、
[`ppo-alignment-audit.md`](./ppo-alignment-audit.md)、
[`activation-loss-sweep-report.md`](./activation-loss-sweep-report.md)。
wave1–4 的 29 个 run 曾归档进 `runs/archive/waves1-4-runs-2026-09-25.tar.gz`；
该归档已于 2026-09-25 随旧规则模型资产一并删除，本波不复用、不恢复。

## 1. 假设

既有事实（见上述报告）：

- 平台约 **1440**（审计后的平衡口径）；wave1–4 的 150k–250k 预算里没有方案
  显著超过 `base680k`（旧口径 1443.3 ± 14.7 平衡值）。平台更像"训练分布
  问题"（固定 Greedy 对手 + 稀疏胜负奖励），不是单个 PPO 旋钮。
- wave4 的 `tanh/gelu/silu`、`vf_coef`、`clip_coef`、LR 量级都没有正贡献；
  `loss_vf1` 是唯一打平者（1442.2 vs 内部基线 1442.9）。
- wave2 里唯一回报正增量的是 `bigbatch`（`--num-envs 16`）；wave3 名义第一是
  真多快照池 `pool4g`。

本波的问题是：**统一到 500k 步后，这些曾经"名义变强/打平"的方案能否
显著超过同一预算下的控制组和父模型 `base680k`？** 每个方案都从头
（`base680k` 热启动）跑固定 500k，评估用修好的座位配对代码一次 fit。

## 2. 方法

### 2.1 语义前提（本波数值不能用旧 run 逐位比较）

- 训练：`train.py` 的 `AutoresetMode.SAME_STEP` P0 修复（对齐 gym 0.21 参考
  语义），本波 6 个 run 全部用修正版。
- 评估：`plan_games`（`src/seven523/ladder.py`）现在是**候选内换座配对**——
  每个 deal 固定发两局、候选两座位各一局；`--games-per-anchor` 必须偶数；
  同一 fit 内所有候选在同轮次共享同一批牌，候选列表顺序不再改变对手牌局。
  旧报告的相位偏差读数已被审计判定不可用，本波不复用；**不使用 `--cross`**
  （其 shipped SE 有 ≥4.16× 低估 bug，审计 §2.3）。

### 2.2 训练矩阵（全部 seed 1、`--cuda True --tensorboard True
--checkpoint-interval 0 --log-interval 50 --total-timesteps 500000`）

| run | 假设 | 关键参数（其余默认 relu + PPO 默认） | 热启动 |
|---|---|---|---|
| `w5_ctrl` | **同批控制组（必须）** | 默认（8 envs × 128 steps = 1024） | `base680k` |
| `w5_bigbatch` | wave2 唯一正增量 | `--num-envs 16`（batch 2048） | `base680k` |
| `w5_pool4g` | wave3 名义第一 | `--opponent pool`，4 快照 + `2@greedy`，`--self-play-refresh 50` | `base680k` |
| `w5_vf1` | wave4 唯一打平者 | `--vf-coef 1.0` | `base680k` |
| `w5_mix_samp` | mix + 高频刷新 + 采样 | `--opponent mix --mix-greedy-prob 0.5 --self-play-refresh 5 --self-play-sample True` | `base680k` |
| `w5_scratch` | 修正语义下 500k 从零能到哪 | 无 `--load-checkpoint` | 无 |

对手池成员（存在、未删）：`runs/probe/base_step00204800.pt`、
`runs/probe/base_step00573440.pt`、`runs/probe_sp/sp_step00020480.pt`、
`runs/probe_sp/sp_step00040960.pt`，加权 `1/1/1/1`；再加 `2@greedy`。

启动脚本：[`runs/w5_launch.sh`](../../runs/w5_launch.sh)（内含每臂完整命令），
日志 `runs/w5_*_train.log`，run 目录 `runs/w5_*__1__<ts>/`（`metrics.csv` +
`agent.pt`）。实际执行分两批各 3 并发：batch A（ctrl/bigbatch/scratch）
03:01:37→03:07:45（3×500k，~6 分钟）；batch B（pool4g/vf1/mix_samp）
03:07:45→03:19:41（~12 分钟；pool/mix 每决策要跑多个网络，速度约为一半）。
6 个 run 都跑满 499,712 步（= 500000 向下取整到 update 边界）。

命令模板（完整见脚本）：

```bash
.venv/bin/python -m seven523.train --exp-name w5_vf1 --seed 1 \
  --total-timesteps 500000 --opponent greedy --vf-coef 1.0 \
  --cuda True --tensorboard True --checkpoint-interval 0 --log-interval 50 \
  --load-checkpoint runs/probe/base_step00696320.pt
```

### 2.3 评估

- 一次同 fit 给 6 个 `w5_*` + `base680k` 定级：
  `tools/build_ladder.py --games-per-anchor 400 --no-traces --seed <s>
  --device cuda`（400 = 200 副牌 × 双座位；每候选 800 局，锚点 Random=1000 /
  Greedy=1315 固定）。脚本 [`runs/w5_eval.sh`](../../runs/w5_eval.sh)。
- 计划本是 seed 0 + 可选 seed 1。因为 seed 0/1 的父子差整体翻符号（见 §4），
  **补跑 seed 2** 作为稳健性检查；三份结果
  `runs/w5_ladder_seed{0,1,2}.txt`。
- 可选直接胜率对照（1500 局 vs Greedy，learner 固定 0 号座位）：
  `w5_ctrl`、`w5_vf1`、`base680k`，输出 `runs/w5_winrate_*.txt`。

## 3. 训练健康度（首/末 50k 步均值 + 终局策略熵分解）

`runs/w5_health.py` 读取 `metrics.csv` 的窗口均值；策略头熵是在
**400 局 vs Greedy（seed 0）**的 learner 决策状态上重算的逐头 masked
entropy。注意 `metrics.csv` 的 `entropy` 是 **134 模板头 + 4 花色头之和**；
真正决定动作的是模板头，花色头只是同模板内的选花偏好，所以下表把两者分开：

| run | 首 50k ret | 末 50k ret | 末 entropy(和) | H_template | H_suit | 末 approx_kl | 末 EV | 末 v_loss | 末 sps | probe win vs greedy |
|---|---|---|---|---|---|---|---|---|---|---|
| `w5_ctrl` | +0.179 | +0.198 | 1.446 | 0.165 | 1.281 | 3.7e-4 | 0.680 | 0.0193 | 1370 | 64.3% |
| `w5_bigbatch` | +0.181 | +0.183 | 1.421 | 0.156 | 1.290 | 4.0e-4 | 0.693 | 0.0194 | 1649 | 62.3% |
| `w5_pool4g` | +0.034 | +0.054 | 1.278 | 0.160 | 1.161 | 2.6e-4 | 0.672 | 0.0210 | 746 | 64.0% |
| `w5_vf1` | +0.185 | +0.183 | 1.371 | 0.160 | 1.208 | 2.9e-4 | 0.691 | 0.0191 | 1137 | 65.7% |
| `w5_mix_samp` | +0.082 | +0.073 | 1.334 | 0.125 | 1.221 | 3.2e-4 | 0.674 | 0.0203 | 689 | 66.2% |
| `w5_scratch` | −0.314 | +0.161 | 1.581 | 0.257 | 1.348 | 2.1e-4 | 0.684 | 0.0198 | 1372 | 64.0% |

读法：

- **模板头已经非常尖锐**：所有 run 的 `H_template` 只有 0.13–0.26 nats
  （`H_suit` 1.16–1.35 nats 占了总熵的 ~85–90%）。也就是说训练把"合法模板
  选择"压得很集中，剩余熵主要在花色偏好上；这与"vs 固定 Greedy 已饱和、
  难以再变强"一致。
- warm-start 各臂的 return / EV / v_loss / KL 都收在同一窄带内
  （0.18–0.20、0.67–0.69、0.019–0.021、2–4e-4），`vf_coef`、batch、池都没有
  在训练内信号上留下可辨差异；`pool4g`/`mix_samp` 的 `episodic_return` 低
  （0.054/0.073）是因为对手分布不同（混了旧快照/Greedy），**不能**与纯
  Greedy 臂的 return 直接比较，其 vs-greedy 胜率（64.0%/66.2%）才是同口径。
- `w5_scratch` 从 −0.314 起步、末 50k 回到 +0.161，probe 胜率 64.0% 已追平
  warm-start 臂，但模板头熵仍最高（0.257），仍是"高熵、尚未压干"的状态；
  它的合并 Elo 最低（1412.4），但与控制组的差 −10.9 ± 22.2 仍在噪声内。

## 4. 座位配对 Elo（同 fit，含 `base680k` 参照）

三份 fit 都是 `--games-per-anchor 400 --no-traces --device cuda`：每个候选对
2 个锚点各 400 局（每人 800 局，一个 fit 共 5600 局），Fisher SE≈15.5–16.5。同 fit 内候选共享牌局，
相对比较比跨 fit 绝对值可靠；合并口径取三次 fit 的算术平均（锚点把绝对
刻度钉住），合并 SE = `max(三次平均 Fisher SE, 候选间 sd/√3)`，差值
SE = `sqrt(SE_i²+SE_j²)`。

### 4.1 各 seed 原始值

| 候选 | seed 0 | seed 1 | seed 2 | 三次均值 | 跨 seed sd | 极差 |
|---|---|---|---|---|---|---|
| `base680k`（冻结父模型） | 1443.3 | 1388.4 | 1458.7 | **1430.1** | 37.0 | 70.3 |
| `w5_ctrl`（控制组） | 1428.5 | 1420.6 | 1420.6 | **1423.2** | 4.6 | 7.9 |
| `w5_bigbatch` | 1424.2 | 1416.3 | 1430.7 | **1423.7** | 7.2 | 14.4 |
| `w5_pool4g` | 1424.9 | 1408.0 | 1430.7 | **1421.2** | 11.8 | 22.7 |
| `w5_vf1` | 1429.2 | 1420.6 | 1453.2 | **1434.3** | 16.9 | 32.6 |
| `w5_mix_samp` | 1411.4 | 1400.4 | 1436.6 | **1416.1** | 18.6 | 36.2 |
| `w5_scratch` | 1408.7 | 1412.8 | 1415.6 | **1412.4** | 3.5 | 6.9 |

每 fit 的 Fisher SE：`base680k` 15.2–16.5，其余 15.4–16.4。

**关键异常（本波最重要的方法学发现）**：父模型 `base680k` 的跨 seed 极差
高达 70.3 Elo（sd 37，远超它单 fit 的 Fisher SE≈16），而 6 个续训子代的
跨 seed sd 只有 3.5–18.6。父子偏移 Δ = 子代 − 父模型的组均值：

| seed | 6 个子代 Δ 范围 | 组均值 |
|---|---|---|
| 0 | −34.6 ～ −14.1 | −22.1 |
| 1 | +12.0 ～ +32.2 | +24.7 |
| 2 | −43.1 ～ −5.5 | −27.5 |

即：**同一 fit 内"续训到底比父模型强还是弱"随牌集会整体反转**。这不是
子代之间的噪声（子代共享牌局、彼此紧密），而是父模型相对"近似克隆群"
的偏移高度依赖牌集。单看任何一个 seed 的父子对比都会得出相反结论；这
也解释了 wave1–4 的早期结论为什么脆弱。**今后父子/方案对比需要 ≥3 个
seed 的合并读数，或直接做候选对候选的换座对局（见 §6）。**

### 4.2 合并三 seed 读数与显著性

| 候选 | 合并 Elo ± 合并 SE | Δ vs `base680k` | Δ vs `w5_ctrl` |
|---|---|---|---|
| `base680k` | 1430.1 ± 21.3 | — | +6.9 |
| `w5_ctrl` | 1423.2 ± 15.8 | −6.9 | — |
| `w5_bigbatch` | 1423.7 ± 15.8 | −6.4 | +0.5 |
| `w5_pool4g` | 1421.2 ± 15.7 | −8.9 | −2.0 |
| `w5_vf1` | 1434.3 ± 16.0 | **+4.2**（差 SE 26.7） | **+11.1**（差 SE 22.5） |
| `w5_mix_samp` | 1416.1 ± 15.7 | −14.0 | −7.1 |
| `w5_scratch` | 1412.4 ± 15.6 | −17.8 | −10.9 |

- **没有任何方案超过 `base680k` 或 `w5_ctrl` 超出噪声。** 名义最高的
  `w5_vf1` 相对父模型 +4.2（0.16σ）、相对控制组 +11.1（0.5σ）；其余全部
  为负且都在 1σ 内。`w5_scratch` 500k 从零也只落后控制组 ~0.5σ。
- 单 fit 里的"第一名"也不稳定：seed 0 是 `w5_vf1`（vs 控制组 +0.7）、
  seed 1 是 `w5_ctrl`/`w5_vf1` 并列、seed 2 是 `w5_vf1`（+32.6）。只有
  `vf1` 三次都排在前二，但合并优势仍不显著。
- SE 说明：合并 SE 已被父模型的跨 seed sd（37.0/√3≈21.3）抬高；若按纯
  Fisher SE 读（~16），`vf1` 对控制组的 +11.1 也只有 0.7σ。因此本报告
  不把任何一臂列为"有优势"。

### 4.3 直接胜率对照（1500 局 vs Greedy，可选检查）

`7g523-eval --episodes 1500 --opponent greedy --seed 0 --device cpu`，
learner 固定在 0 号座位（该工具不做换座，仅作 sanity check）：

| 策略 | 胜/和/负 | 均分 | 均分差 | 均 return | 非法率 |
|---|---|---|---|---|---|
| `w5_ctrl` | 64.9% / 7.7% / 27.4% | 60.74 | +21.49 | +0.215 | 0% |
| `w5_vf1` | 63.1% / 7.9% / 29.0% | 60.08 | +20.15 | +0.202 | 0% |
| `base680k` | 63.0% / 7.3% / 29.7% | 60.41 | +20.82 | +0.208 | 0% |

三者胜率差 ≤1.9pp（n=1500 时二项 SE≈1.2pp），与 ladder 的"打平"结论一致，
没有哪一种续训（或父模型）在 Greedy 上拉开差距。

## 5. 结论

1. **统一 500k 预算下，"优势方案"不存在。** 6 个方案没有一个在合并
   3-seed、同 fit 座位配对口径下显著超过 `base680k`（1430.1）或控制组
   `w5_ctrl`（1423.2）。最好的 `w5_vf1`（1434.3）也只有 +4.2 ± 26.7。
2. **平台不是 PPO 旋钮问题，这一点现在有更强的证据**：500k 的
   bigbatch、真多快照池、vf_coef、高频刷新+采样、甚至从零 500k，全部落在
   1412–1434 这条 ~20 Elo 宽的带里。训练内信号（return/EV/KL）也全部同带。
3. **评估口径的残差风险被量化了**：父模型在换牌集时的跨 seed sd（37）
   远大于单个 fit 的 Fisher SE（16），父子偏移会整体翻符号。旧报告里
   "某方案 +20～40 Elo"级别的结论，很可能就是这个分量。方案比较今后
   应以 ≥3 seed 合并（或直接换座对局）为准。
4. 模板头已经尖锐（H≈0.13–0.17 nats），花色头熵占绝大部分；模型在
   Greedy 上"会做对模板选择"这件事已经饱和，剩下的提升空间不在
   PPO 超参。

### 下一步（结构性、本波未试）

- **稠密奖励塑形**：当前只有稀疏胜负/分差回报，训练内 return 长期贴顶；
  可试"每墩得分差 / 剩牌数" shaping，让梯度区分强弱而非只区分胜负。
- **观测增广**：给当前状态加对手剩余牌型估计、已出牌统计等特征，扩大
  可学习空间（比继续调 PPO 旋钮更可能移动平台）。
- **PFSP（prioritized fictitious self-play）**：本波的池是均匀采样；按
  对当前学习者的胜率加权采样，专门攻"打不过的牌型"，是对 `pool4g`/
  `mix_samp` 的直接升级。
- **>2 家**：2 人 Greedy 已饱和，3–4 人变体的策略空间大得多（环境
  `--num-players` 已有支持，需先解决训练速度与评估座位配对）。
- **评估侧**：修 `--cross` 的联合 Hessian SE，或加一个候选对候选的
  换座对局工具；这是把"父子偏移"从牌集噪声里拆出来的最短路径。

## 6. 复现与产物

```bash
# 训练（含每臂完整参数；两批各 3 并发）
setsid nohup runs/w5_launch.sh > runs/w5_launch.log 2>&1 < /dev/null &

# 定级（每次先跑 seed0；seed1/seed2 追加第二个 fit）
bash runs/w5_eval.sh                 # -> runs/w5_ladder_seed0.txt
bash runs/w5_eval.sh seed1           # seed0 + -> runs/w5_ladder_seed1.txt
bash runs/w5_eval.sh seed2           # seed0 + -> runs/w5_ladder_seed2.txt

# 结构健康度（metrics 窗口 + 逐头熵）
.venv/bin/python runs/w5_health.py runs/w5_ctrl__1__*/agent.pt 400

# 直接胜率
7g523-eval --checkpoint runs/w5_vf1__1__*/agent.pt \
  --episodes 1500 --opponent greedy --seed 0 --device cpu
```

产物：`runs/w5_*_train.log`、`runs/w5_*__1__*/{metrics.csv,agent.pt}`、
`runs/w5_ladder_seed{0,1,2}.txt`、`runs/w5_health_*.json`、
`runs/w5_winrate_*.txt`、`runs/w5_launch.sh`、`runs/w5_eval.sh`、
`runs/w5_health.py`。无 PENDING；旧 wave1–4 模型保持归档状态。

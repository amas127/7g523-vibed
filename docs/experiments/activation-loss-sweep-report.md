# 激活函数与损失配比扫描报告（wave4）

> 资产状态（2026-09-25）：本报告引用的模型 checkpoint 已删除，旧路径不再可用；数值为牌型族规则变更前口径。

> **状态：训练 + 两相位定级完成（2026-09-25）。结论：没有突破平台。**
> 在修正后的 `SAME_STEP` 语义下、从 `base680k` warm start 的 150k–250k 续训中，
> `tanh / gelu / silu`、`vf_coef 0.25/1.0`、`clip_coef 0.2`、以及 LR 臂
> （`1e-4 / 5e-4 / 1e-3 / constant 2.5e-4`）**没有一个显著超过同批内部基线
> `act_ctrl`（1442.9 Elo，两相位平均）**；唯一打平的是 `loss_vf1`（1442.2）。
> 内部基线与 `base680k`（1443.7）打平，说明这批 PPO 旋钮在 SAME_STEP 分布下
> 对"续训 200k 能否变强"没有正贡献。
> 误差口径引用 [`elo-reliability-audit.md`](./elo-reliability-audit.md)：
> 座位相位由候选下标奇偶决定，单臂 Fisher SE≈22 不含相位分量；本报告采用
> **两相位平均**（平衡 SE≈15–16，候选间差 SE≈22）。

## 1. 假设

平台既有事实见 [`ladder-report.md`](./ladder-report.md)：vs Greedy 的 PPO 在
~60k 步后就饱和，1M 步、self-play、bigbatch、hidden 256 等 wave1–3 配置都停在
~1490（旧口径；审计后平衡口径约 **1440**）。本波针对两类此前未扫过的旋钮：

- **A. 非线性**（对齐参考 `ppo.py`）：把 hidden 激活从 ReLU 换成 `tanh`
  （参考实现默认）或光滑 ReLU 系 `gelu/silu`，能否改变优化地形、逃出局部盆地。
  另加一个从零训练 250k 的 tanh 对照，检验"换成 tanh 是否本来就更好"。
- **B. 损失配比**：`vf_coef` 0.25/1.0、`clip_coef` 0.2（参考默认，我们一直
  0.1）。检验平台是否由价值/策略项的梯度平衡造成。
- **D. 学习率**（用户追加，审计后无旧结论可用）：`2.5e-4` 之外的量级
  `1e-4/5e-4/1e-3`，以及修正语义下重跑 constant `2.5e-4`（wave1 的
  `ebo_noanneal` 是旧 NEXT_STEP 语义，不可直接比）。期望值事先就低：
  [`ppo-alignment-audit.md`](./ppo-alignment-audit.md) §6.1 指出 ~60k 进入平台时
  LR 仍有 2.35e-4，退火只解释后期乏力、不解释早期饱和。

## 2. 方法

### 2.1 SAME_STEP 语义变更（必须注意）

本波所有 run 使用 P0 修复后的 `train.py:421`
（`AutoresetMode.SAME_STEP` + `train.py:213` 的 `final_info` 回合统计适配），
与 wave1–3 的 `NEXT_STEP` 语义不同：旧语义把 done 步之后的"死样本"混进
rollout（审计实测 ~3.9% chimera transitions），新语义回正到参考运行时。
**因此 wave4 的数值不能与 wave1–3 的 run 逐位比较**，这也是本波必须带一个
同批内部基线的原因。修复本身不改变 PPO 公式，只改变数据分布。

### 2.2 训练矩阵（seed 1、`--opponent greedy`、TensorBoard 开、`--checkpoint-interval 0`）

全部 warm start `runs/probe/base_step00696320.pt`（base s680k），除注明外 200k 步。

| run | 假设 | 关键参数 | 步数 | 设备 |
|---|---|---|---|---|
| `act_ctrl` | **内部基线（必须）** | relu + 默认 loss | 200k | GPU |
| `act_tanh` | 参考 ppo.py 的 tanh 能否逃盆 | `--activation tanh` | 200k | GPU |
| `act_gelu` | 光滑 ReLU 系 | `--activation gelu` | 200k | GPU |
| `act_silu` | 光滑 ReLU 系 | `--activation silu` | 200k | CPU |
| `act_tanh_scratch` | tanh 是否本就更优（从零） | `--activation tanh`，无 warm start | 250k | CPU |
| `loss_vf025` | 削弱 critic 项 | `--vf-coef 0.25` | 200k | CPU |
| `loss_vf1` | 强化 critic 项 | `--vf-coef 1.0` | 200k | CPU |
| `loss_clip02` | 参考默认 clip | `--clip-coef 0.2` | 200k | CPU |
| `lr_1e4` | LR 量级 | `--learning-rate 1e-4` | 200k | CPU |
| `lr_5e4` | LR 量级 | `--learning-rate 5e-4` | 200k | CPU |
| `lr_1e3` | LR 量级（最激进） | `--learning-rate 1e-3` | 150k | CPU |
| `lr_const` | 修正语义下重跑不退火 | `--anneal-lr False`（LR 恒 2.5e-4） | 200k | CPU |

启动脚本：`runs/actloss_launch_wave4.sh`（批 1）、`runs/actloss_launch_wave4b.sh`
（其余，LR 臂排最前）。命令模板：

```bash
.venv/bin/python -m seven523.train --exp-name act_tanh --seed 1 \
  --opponent greedy --cuda True --tensorboard True --checkpoint-interval 0 \
  --log-interval 50 --total-timesteps 200000 --activation tanh \
  --load-checkpoint runs/probe/base_step00696320.pt
```

**设备说明（诚实记录）**：首批 3 个 run 按原计划用 GPU，但可靠性审计的评测
同时占用 GPU，实测只有 423–620 sps；其余 9 个 run 改走 CPU
（`--cuda False`、`OMP_NUM_THREADS=2`，wave1 配方），实测 910–1382 sps
（~1.5–2×）。训练设备不影响 ladder 评测（所有候选同一口径），但激活家族里
`act_silu` 与对照组设备不同，损失/LR 臂全部为 CPU；这是本波的一个已知混杂，
见 §5。

### 2.3 非线性替换的实际语义

`--load-checkpoint` 只复制 `state_dict`，激活函数是网络结构的一部分。因此
"warm start 到 tanh/gelu/silu"= **保留 relu 训练出的线性权重、直接换掉非线性**，
初始函数与 base680k 不同（不是等价的续训）。第 1 个 rollout（1024 步）即暴露：

| run | episodic_return | explained_variance | entropy |
|---|---|---|---|
| `act_ctrl`（relu 重载） | 0.164 | 0.690 | 1.401 |
| `act_tanh` | **−0.026** | **−0.404** | 1.957 |
| `act_gelu` | 0.103 | 0.613 | 1.509 |
| `act_silu` | 0.140 | 0.621 | 1.593 |
| `act_tanh_scratch` | −0.516 | −0.853 | 2.404 |
| 其余 relu warm start（CPU） | 0.1625 | 0.661 | 1.36–1.39 |

tanh 的初始函数被显著破坏（EV 变负），它必须在 200k 内先"修复"再谈变强。

### 2.4 定级口径：两相位平均（引用可靠性审计）

审计（`elo-reliability-audit.md` §2.1/§3.3/§6.7）证明：同一 seed 下牌局只依赖
`(seed, 轮次, 锚点)`，而 `ladder.py:127` 的座位
`(round + candidate_index + anchor_index) % 2` 使**候选 Elo 成为下标奇偶的确定性
函数**；同 seed 换序 = 全部牌换座，位移 −7.5～+74.6 Elo（均值 +41.3），单臂
Fisher SE≈22 只覆盖换牌、不覆盖相位；**座位平衡后 sd 降到 ≈14.6**。

本波 phase A 的候选集是 13 个（奇数），直接倒序**保持**奇偶（审计 §3.3：左旋 2
保相位；N=13 时倒序等价于保相位），所以 phase B 采用等价做法：在列表**最前面
插入脚本策略 `greedy_dummy=greedy`**（14 个候选），13 个真实候选下标全部 +1 →
相位翻转，牌局与 seed 不变。两相位逐候选取平均即为平衡点估计。未使用
`--cross>0`（其 SE 有 ≥4.16× 低估 bug，审计 §2.3）。

- 输出：`runs/actloss_ladder.txt`（phase A）、`runs/actloss_ladder_phaseB.txt`
  （phase B）、`runs/actloss_balanced.txt`（平均，脚本 `runs/actloss_balanced.py`）。
- SE 口径：单臂 ±21.7–23.1（报告值）；平衡点估计用
  `max(14.6, sqrt(SE_A²+SE_B²)/2)`≈14.8–16.2；两个平衡候选之差用
  `sqrt(SE_i²+SE_j²)`≈21.9–22.8。
- **口径自检**：`base680k` 在 phase A（偶下标）1447.4、phase B（奇下标）1439.9，
  与审计完全一致（S6 正序 1447.4 / C6 倒序 1439.9）；平衡值 1443.7 对上审计
  逐牌换座拟合的 1443.3 ± 14.7。这验证了两相位法正确，也再次说明旧报告的
  "1489.4" 是相位上偏的产物（审计 §5.1）。

## 3. 训练曲线摘要

末 20 个 update 的均值（`runs/actloss_summarize.py`）：

| run | 步数 | episodic_return | entropy | explained_var | value_loss | approx_kl | clipfrac | LR(末端) |
|---|---|---|---|---|---|---|---|---|
| `act_ctrl` | 199,680 | 0.204 | 1.426 | 0.686 | 0.0203 | 3.4e-4 | 4.4e-4 | 1.3e-5 |
| `act_tanh` | 199,680 | 0.158 | 1.676 | 0.689 | 0.0200 | 9.5e-5 | 1.2e-5 | 1.3e-5 |
| `act_gelu` | 199,680 | 0.218 | 1.347 | 0.679 | 0.0189 | 4.1e-4 | 1.9e-3 | 1.3e-5 |
| `act_silu` | 199,680 | 0.189 | 1.509 | 0.672 | 0.0213 | 3.1e-4 | 1.2e-3 | 1.3e-5 |
| `act_tanh_scratch` | 249,856 | 0.061 | 1.707 | 0.665 | 0.0199 | 3.7e-5 | 0 | 1.1e-5 |
| `loss_vf025` | 199,680 | 0.211 | 1.440 | 0.665 | 0.0203 | 3.7e-4 | 8.4e-4 | 1.3e-5 |
| `loss_vf1` | 199,680 | 0.208 | 1.458 | 0.673 | 0.0196 | 3.3e-4 | 5.5e-4 | 1.3e-5 |
| `loss_clip02` | 199,680 | 0.197 | 1.273 | 0.689 | 0.0201 | 3.7e-4 | 1.2e-5 | 1.3e-5 |
| `lr_1e4` | 199,680 | 0.207 | 1.418 | 0.694 | 0.0209 | 7.6e-5 | 0 | 5.4e-6 |
| `lr_5e4` | 199,680 | 0.206 | 1.439 | 0.682 | 0.0179 | 8.7e-4 | 5.0e-3 | 2.7e-5 |
| `lr_1e3` | 149,504 | 0.163 | 1.445 | 0.671 | 0.0189 | 1.6e-3 | 3.0e-2 | 7.2e-5 |
| `lr_const` | 199,680 | 0.210 | **1.166** | 0.678 | 0.0191 | **3.4e-3** | **0.103** | 2.5e-4 |

观察（都与"能否突破平台"关系不大，但用于解释）：

- **所有 warm-start run 的 return/EV 收在同一带内**（0.16–0.22 / 0.67–0.69），
  `vf_coef`、`clip_coef`、LR 量级都没有在训练信号上留下可辨差异；`vf_coef`
  0.25→1.0 之间 value_loss 只从 0.0203 到 0.0196。
- `tanh` 熵最高（1.676），`silu` 次之（1.509），`gelu`/relu 最低（1.35–1.43）；
  `lr_const` 熵最低（1.166）、clipfrac 高一个量级（0.103）、KL 也最大（3.4e-3）
  —— 恒 LR 让它持续更新，但**训练内 return 并未更高**（0.210）。
- `lr_1e3` 没有发散（clipfrac 0.030），但训练 return（0.163 @150k）并不优于
  `lr_5e4`/`lr_1e4`。
- `act_tanh_scratch` 训练内 return 最低（0.061），说明从零学 tanh 在 250k
  预算内明显更慢。

## 4. 定级结果（同 seed 0 牌局，两相位平衡）

两相位都含 `base680k` 与内部基线 `act_ctrl`；phase A（13 候选，
`runs/actloss_ladder.txt`）与 phase B（14 候选含 dummy，
`runs/actloss_ladder_phaseB.txt`）原始值与平衡平均：

| 候选 | phase A（奇偶） | phase B（翻转） | **两相位平均** | 平衡 SE | **Δ vs `act_ctrl`** | ± |
|---|---|---|---|---|---|---|
| `base680k`（冻结参考） | 1447.4（偶） | 1439.9（奇） | **1443.7** | 16.2 | +0.8 | 22.8 |
| **`act_ctrl`（内部基线）** | 1439.9（奇） | 1445.9（偶） | **1442.9** | 16.1 | 0.0 | — |
| `loss_vf1` | 1434.0（奇） | 1450.5（偶） | **1442.2** | 16.1 | −0.7 | 22.8 |
| `lr_1e4` | 1403.1（奇） | 1452.0（偶） | **1427.5** | 15.8 | −15.4 | 22.6 |
| `lr_1e3` | 1438.4（奇） | 1407.1（偶） | **1422.8** | 15.7 | −20.2 | 22.5 |
| `act_gelu` | 1421.1（奇） | 1423.9（偶） | **1422.5** | 15.7 | −20.4 | 22.5 |
| `lr_5e4` | 1423.9（偶） | 1421.1（奇） | **1422.5** | 15.7 | −20.4 | 22.5 |
| `loss_vf025` | 1408.5（偶） | 1435.5（奇） | **1422.0** | 15.7 | −20.9 | 22.5 |
| `act_silu` | 1401.7（偶） | 1428.2（奇） | **1415.0** | 15.6 | −28.0 | 22.4 |
| `loss_clip02` | 1416.8（偶） | 1404.4（奇） | **1410.6** | 15.5 | −32.3 | 22.4 |
| `lr_const` | 1414.1（偶） | 1400.3（奇） | **1407.2** | 15.5 | −35.7 | 22.3 |
| `act_tanh` | 1412.7（偶） | 1375.4（奇） | **1394.1** | 15.2 | −48.8 | 22.2 |
| `act_tanh_scratch` | 1365.2（奇） | 1366.5（偶） | **1365.8** | 14.8 | −77.1 | 21.9 |

相位位移（B−A）在本波 13 个候选上均值 +1.8、sd 23.9、范围
[−37.3, +48.9]（`lr_1e4` +48.9，`act_tanh` −37.3）——**如果只看单次 fit，这些
位移会伪造出或抹掉 20–40 Elo 的"效果"**；这正是必须两相位平均的原因。

### 4.1 判读（诚实版）

1. **没有任何臂超过内部基线。** 最好的 `loss_vf1` 是 −0.7 ± 22.8（与基线
   不可区分）；`base680k` 是 +0.8 ± 22.8（同样不可区分）。在 ±22 的候选间
   误差下，**不能宣布任何正向突破**。
2. **激活假设被否**：`act_tanh` −48.8（~2.2σ）、`act_silu` −28.0、
   `act_gelu` −20.4，全部低于 relu 对照。参考 `ppo.py` 用 tanh 这条先验在本
   游戏不成立；warm start 换非线性还会先破坏函数（§2.3）。从零 tanh
   （−77.1，3.5σ）说明 250k 预算下 tanh 也远不如 relu 基线的进度。
3. **损失配比假设基本被否**：`vf_coef` 在 0.25↔1.0 间几乎惰性
   （−20.9 vs −0.7，互差 ~20，<1σ，方向不一致）；参考默认 `clip 0.2`
   比我们一直用的 0.1 更差（−32.3，~1.4σ）。
4. **LR 臂没有正收益**：`1e-4/5e-4/1e-3` 分别为 −15.4/−20.4/−20.2，
   `lr_const`（修正语义重跑）−35.7 ± 22.3；不仅没突破，恒定 2.5e-4 还比退火版
   差约 1.6σ。这与 [`ppo-alignment-audit.md`](./ppo-alignment-audit.md) §6.1 的
   预期一致：平台在 ~60k 就出现（LR 仍 2.35e-4），退火/恒 LR 只影响后期，
   解释不了早期饱和。
5. **与审计的平衡 E 系列交叉对照**：审计同 seed 0 逐牌换座得到
   `ctrl680 1417.0`、`mix50 1415.6`、`ent0 1410.7`、`sp_r5 1406.6`、
   `base680k 1443.3`。wave4 的 `loss_vf1`（1442.2）≈ `base680k`（1443.7），
   其余臂落在 1366–1428——**wave4 没有产生任何位于 base680k 之上、且超出
   平衡 SE 的候选**；平台顶就是 ~1440（平衡口径）。

## 5. 结论与限制

**结论：本次 PPO 旋钮（激活函数、损失配比、LR 量级/退火）不能突破 ~1440
（平衡口径）的平台。** 唯一与内部基线打平的是 `loss_vf1`；`tanh/gelu/silu`、
`vf 0.25`、`clip 0.2`、四个 LR 臂全部 ≤ 基线。结合 wave1–3，平台不是由
"激活选择 / value loss 权重 / clip 宽度 / LR 量级"造成的——至少在本训练分布
（2 家、vs 冻结 Greedy、稀疏终局奖励）下不是。

限制与注意事项：

- **设备混杂**：`act_ctrl/act_tanh/act_gelu` 训练于 GPU，其余 9 个训练于 CPU
  （§2.2 的速度原因）。训练设备只造成浮点级差异（relu 重载 run 的起始 rollout
  指标 0.164/0.690 vs CPU 组 0.163/0.661），量级 << ±22 的定级噪声，但
  "设备效应有多大"本波没有专门对照；激活家族里 `act_silu` 与对照设备不同。
- **SAME_STEP**：本波全部 run 用修正语义，与 wave1–3 不可逐位比较；
  但这不影响波内比较（所有臂与基线同语义、同 seed、同 warm start）。
- **SE 口径**：单臂 ±22 只对"换牌"校准；本报告用两相位平均（±15–16）消除了
  相位项，候选间比较仍用保守的 ±22。更紧的口径需要审计 §6.5 的
  head-to-head 同牌配对（本波未做）。
- **训练预算**：150k–250k 步足够验证"warm start 后能不能变强"，但不能排除
  更长预算 + 其他配方（如分段退火 + 对手多样性）的慢效果。

## 6. 后续建议

1. **不要继续在这几个 PPO 旋钮上堆局数/seed**：本波 + 审计的相位效应
   （±30–75）远大于这些臂的效应（≤20，且方向为负）。若要正式宣布某臂"更强"，
   先用审计 §6.5 的 head-to-head 同牌 CRN 配对（复用同一副牌、候选互换座位），
   它比绝对锚点 Elo 方差小得多。
2. **LR 只值一次轻量复测**：`lr_1e4` 是唯一在 0.7σ 内、且训练内 EV 最高
   （0.694）的臂；若要做，用两相位 + 同牌配对，而不是单 fit。
3. **平台更可能卡在训练分布**：固定 Greedy + 稀疏奖励 + 窄对手分布是
   wave1–3/alignment-audit 反复指向的根因；下一步优先级仍应是
   **对手多样性（pool/mix + 强对手）、奖励塑形、以及 ≥256 容量在更长预算下的
   对照**，而不是超参微调。
4. **补一个对照**（本波未做、成本低）：`--num-minibatches 8`、Adam `eps`
   变体（CLI 未暴露，需小改动）、以及 `loss_vf1` 的 head-to-head 复测。
5. **定级流程**：今后任何"候选 vs 候选"结论至少两相位，最好直接上逐牌换座
   配对；`base680k` 永远作为冻结参考进同一 fit。

## 7. 复现

```bash
# 训练（见脚本内注释；批 1 用 GPU，其余 CPU）
bash runs/actloss_launch_wave4.sh
bash runs/actloss_launch_wave4b.sh

# 定级：两相位（phase B 在前面插 greedy_dummy 翻转相位；均无 --cross）
bash runs/actloss_ladder.sh          # -> runs/actloss_ladder.txt
bash runs/actloss_ladder_phaseB.sh   # -> runs/actloss_ladder_phaseB.txt

# 两相位平均表
.venv/bin/python runs/actloss_balanced.py | tee runs/actloss_balanced.txt

# 训练曲线摘要
.venv/bin/python runs/actloss_summarize.py <run-name> ... --tail 20 --md
```

产物：`runs/act_{ctrl,tanh,gelu,silu,tanh_scratch}__*/`、
`runs/loss_{vf025,vf1,clip02}__*/`、`runs/lr_{1e4,5e4,1e3,const}__*/`（含
`metrics.csv`、`agent.pt`、`tb/`、`args.json`）；评级文本
`runs/actloss_ladder*.txt`、`runs/actloss_balanced.txt`。本波未改
`src/seven523/`、未 commit、未触碰其他 agent 的报告与 `traces/study/`。

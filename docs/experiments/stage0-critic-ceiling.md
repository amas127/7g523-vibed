# Stage-0 诊断 D-C：critic 天花板探针

> **状态**：完成（2026-10-06）。**判定：value 残差不可约，停止 critic 改进（H0）。**
> **口径**：revision-3（`rules_id=2e36dbea44893696`）、obs v5（2 家 161 维）、PPO memoryless MLP。
> 冻结策略 = `w2m_ctl`（`runs/w2m_ctl__11__1790516900/agent.pt`，obs v5，arch=shared，hidden=128，
> seed 11，2M 步）。本探针不改 src/、不改 obs_version、不发布 checkpoint；产物在
> `runs/stage0/critic/`（gitignored）。

## 0. 预注册（动手前冻结，2026-10-06 12:42 前）

### 0.1 问题与假设

- **问题**：critic 的 value 误差是「容量/数据可约」还是「策略与回报定义下的不可约随机性」？
- **H0（不可约）**：在冻结策略 `w2m_ctl` 自对局叶子分布上，critic 残差主要由 within-decision
  随机性（发牌 + 未来博弈）构成；加深 / 独立 trunk / 加数据 / 换目标都无法把 held-out EV 推到
  同数据 shipped 基线之上 **+0.03**。
- **H1（容量可约）**：critic 容量（深度或独立 trunk）是瓶颈，加深 / 独立能稳健提高 EV（≥ +0.03）。
- **H2（数据可约，辅助）**：数据量是瓶颈，×4 相对 ×1 有实质 EV 提升（≥ +0.03）。

### 0.2 数据来源与采集

- **策略**：`w2m_ctl`（champion，obs v5）对 `w2m_ctl`（self-play，两侧同模型，deterministic
  argmax）。这是该冻结策略下「信念值 `E[return|obs]` 可学性」的最干净上界（训练实际还叠加
  mix_random/sampling/pool 噪声，见 §5 限制）。
- **分布**：真实 self-play 对局的每个决策点（on-policy 分布）。每个决策记录该座位公开 View 的
  obs（161 维，`seven523.networks.encode_observation`），做 K=8 次隐藏牌 determinization
  （`sample_hidden` + `rebuild`，`runs/o4lite-search/rollout_policy.py`），每个世界用 champion
  双打至终局，记录根座位的归一化终局回报 `leaf_mc = (score_own − score_other) / 100`
  （`Game.returns()` 单位，∈ [−1,1]）。
- **held-out 分割**：按 deal seed 分层，`deal_seed % 7 == 0` 为 val（与 EI-2/EI-3 口径一致）。
- **数据量臂**：train 子集按 decision 随机抽 ×1 / ×2 / ×4（×4 = 全量）。
- **复用说明**：`runs/ei2_value_t5/` 只剩 `t_leafq/critic.pt`（旧 d0leaves 数据已硬删除，见
  `artifacts/cleanup/20260929-runs.md`）；`runs/ei/` 已删；`traces/study/` 只剩 manifest.json。
  因此本探针**重新采集**，只把 t_leafq 的历史 EV（≈0.54–0.55，EI-3 报告）作背景参考，不混用其数据。

### 0.3 臂（arms）

结构臂（都在 `leaf_mc` 目标上训练，policy/actor 冻结不动）：

| arm | 结构 | 可训参数 |
|---|---|---|
| `base` | 冻结共享 trunk + `Linear(128,1)` critic 头（champion 权重续训） | 129 |
| `deep_head` | 冻结共享 trunk + `Linear(128,128)`+relu+`Linear(128,1)` | 16,641 |
| `indep` | 独立 critic trunk `Linear(161,128)`+relu+`Linear(128,128)`+relu+`Linear(128,1)` | 37,377 |
| `indep_wide` | 独立 critic trunk 加宽到 256（两层）+ `Linear(256,1)` | 107,521 |

数据量臂：`base` 与 `indep` 各在 ×1 / ×2 / ×4 上训练。
目标臂（辅助，测「近似误差 vs 不可约噪声」）：`base` / `indep` / `indep_wide` 在 `leaf_mean`
目标（每决策 K=8 叶均值，同决策同 obs 去重，噪声降 8×）上训练。

### 0.4 端点

- 主端点：val EV = `1 − MSE / Var(target)`（按 deal 分层的 held-out 集）。
- 次端点：val MSE、校准 bias = `mean(pred) − mean(target)`、binned ECE、R²/相关系数、train/val 曲线。
- 地板参考：数据直接算 between/within decision 方差分解（law of total variance）。

### 0.5 判据

- **主判据**：所有结构臂在 `leaf_mc` 目标、**固定 final epoch（20）**（不得靠 epoch 挑选，N-24
  教训）下的 max val EV，相对同数据 `base` 增量 < **+0.03** → H0（残差不可约，停止 critic 改进）；
  ≥ +0.03 且跨数据量/跨目标一致 → H1。
- **辅助判据**：`leaf_mean` 目标上 EV 逼近其天花板（0.9345）的程度区分「不可约噪声」与「近似误差」。
- 数据量判据：×4 − ×1 的 EV 差 ≥ +0.03 才算数据可约。

### 0.6 分析计划

- 训练 recipe 对齐 EI-2/EI-3：lr 3e-4、batch 1024、20 epoch、Adam、seed 0、MSE loss。
  `base`/`deep_head` 冻结 champion 的共享 trunk（`agent.network`）；`indep` 从零自建 trunk。
- 主读数用 final epoch（20）；best-epoch 仅作诊断（记录但不用于判据）。

## 1. 执行记录与命令

采集（1200 副 self-play deal，K=8，seed 9400–10599）：

```bash
.pi/limits/run_limited.sh gpu .venv/bin/python runs/stage0/critic/collect_leaves.py \
  --deals 1200 --k 8 --start-seed 9400 --out runs/stage0/critic/data
```

训练（13 臂，每臂同 recipe 只换 `--arm/--target/--data-scale`）：

```bash
.pi/limits/run_limited.sh gpu .venv/bin/python runs/stage0/critic/train_critic.py \
  --arm <base|deep_head|indep|indep_wide> --target <mc|mean> --data-scale <1|2|4>
```

产物：`runs/stage0/critic/{collect_leaves.py,train_critic.py,analyze.py,data/,runs/}`。
采集规模：rows_train 334,816 / rows_val 56,240；decisions_train 41,852 / decisions_val 7,030；
elapsed 2,647s。

## 2. 结果

### 2.1 方差分解与 aleatoric 地板（数据直接算，train decisions）

| 量 | 数值 |
|---|---|
| `between = Var(E[leaf_mc | decision])` | 0.1811 |
| `within = E[Var(leaf_mc | decision)]` | 0.1015 |
| `total = Var(leaf_mc)` | 0.2826 |
| **single-sample EV 天花板（若完美拟合 E[R|obs]）** | **0.6409** |
| `leaf_mean`（K=8）目标天花板 | 0.9345 |
| K=32 mean 目标天花板（参考） | 0.9828 |

即：单叶 MC 回报有 36%（=0.1015/0.2826）是 obs 无法消除的不可约噪声；即使 critic 完美拟合
`E[R|obs]`，single-leaf EV 上限也只有 0.64。

### 2.2 结构臂（final epoch = 判据；best = 诊断）

`leaf_mc` 目标，×4 数据：

| arm | params | final EV | final MSE | bias | best EV@ep |
|---|---:|---:|---:|---:|---:|
| `base` | 129 | **0.5216** | 0.1372 | +0.026 | 0.5246@2 |
| `deep_head` | 16,641 | 0.5211 | 0.1374 | −0.004 | 0.5225@15 |
| `indep` | 37,377 | 0.5085 | 0.1410 | +0.022 | 0.5200@4 |
| `indep_wide` | 107,521 | 0.5008 | 0.1432 | +0.013 | 0.5215@2 |

`leaf_mean` 目标（K=8，噪声降 8×），×4 数据，val 7030 decisions：

| arm | final EV | final MSE | bias |
|---|---:|---:|---:|
| `base` | **0.8013** | 0.0373 | −0.003 |
| `indep` | 0.7859 | 0.0402 | −0.041 |
| `indep_wide` | 0.7901 | 0.0394 | −0.013 |

关键读数：

1. **没有结构臂超过 shipped `base`**：`deep_head` ΔEV = −0.0005，`indep` ΔEV = −0.0131，
   `indep_wide` ΔEV = −0.0209。独立 trunk 反而在 val 上更差（train MSE 更低、val MSE 更高 → 过拟合
   到 within-decision 噪声，`indep_wide` train 0.1154 vs val 0.1432）。
2. **epoch 选择无空间**：`base` best（0.5246@2）与 final（0.5216）几乎相等；`indep` 的 best（0.52@4）
   会因 early-stop 虚高，final 才诚实。与 N-24 教训一致。
3. **shipped 冻结 critic 与重训无差**：冻结 champion critic 在 val 上 EV = 0.5222（bias +0.012）；
   `base` 重训后 final EV = 0.5216。即 335k 行重训线性头都不再抬 EV。

### 2.3 数据量臂（`leaf_mc`）

| arm | ×1 | ×2 | ×4 |
|---|---:|---:|---:|
| `base` | 0.5236 | 0.5210 | 0.5216 |
| `indep` | 0.4941 | 0.5001 | 0.5085 |

`base` 三档全平（0.52）；`indep` 随数据略升但仍低于 `base` 且过拟合。数据量不是瓶颈（`leaf_mean`
目标下 `base` 在 ×1/×2/×4 也全平于 0.80，见 §2.4）。

### 2.4 目标臂：区分「不可约噪声」与「近似误差」

`leaf_mean`（K=8）目标把 within 噪声除 8，其天花板 = 0.9345。实测 `base` 只到 0.8013（MSE 0.0373）。
分解：MSE 0.0373 = 残余噪声 within/8（≈0.0127）+ **近似误差方差 ≈0.0246**。即 critic 解释了
belief-value 方差（between=0.1811）的约 **86%**（1 − 0.0246/0.1811），还有 ~14% 的 `E[R|obs]`
是任何被测结构都学不出来的（`indep`/`indep_wide` 不升反降）。

推论：single-leaf 的 0.52 是「0.64 的 aleatoric 上限」再减去「~0.025 不可消除的近似误差」的结果；
既不是纯噪声（否则 mean 目标会到 0.93），也不是容量/数据可修的（加容量加数据都不动）。

### 2.5 校准

| critic | bias | binned ECE | slope(pred→y) | corr |
|---|---:|---:|---:|---:|
| 冻结 champion | +0.0122 | 0.0170 | 1.040 | 0.7235 |
| `base` 重训 | +0.0263 | 0.0263 | 0.985 | 0.7240 |

两版都接近无偏（bias ≤ 0.03、ECE ≤ 0.03、slope ≈ 1.0）。**问题不是校准**，是残差方差本身。

## 3. 判定（按预注册）

- **H0 成立**：所有结构臂（加深 / 独立 trunk / 加宽）在 `leaf_mc` 上的 max final EV = 0.5216，
  就是 `base` 自己；相对基线最大增量 +0.002（`base_mc_x1` vs `base_mc_x4`，属采样噪声，非结构增益）。
  所有容量臂 ΔEV ≤ 0。
- **H1 不成立**：独立 trunk / 加宽在 `leaf_mc` 与 `leaf_mean` 上都不超过 `base`。
- **H2 不成立**：×1/×2/×4 三档 EV 全平（`base` 0.52、`leaf_mean` 0.80），数据量不是瓶颈。
- **结论**：value 残差是「不可约 aleatoric 噪声（36% 方差）+ 固定策略与回报定义下 ~14% 不可消除的
  信念值近似误差」；**不是容量/数据可约的**。按判据（max EV 相比同数据基线 < +0.03）→
  **停止 critic 改进**，与 N-24（重采叶子）与 cap0（宽头）一致，共同关闭 value 侧的结构改进线。

一句话：shipped 线性 critic 头（129 参数）已经碰到该分布的可达上限，更深/独立/更宽/更多数据
都到不了更高的 held-out EV（0.52 vs 上限 0.64，且 0.64 不可达）。

## 4. 证据落点

- 采集脚本/数据：`runs/stage0/critic/collect_leaves.py`、`runs/stage0/critic/data/{X,y,g,d}_{train,val}.npy`、`collect_meta.json`。
- 训练脚本/产物：`runs/stage0/critic/train_critic.py`、`runs/stage0/critic/runs/<arm>_<target>_x<scale>/{metrics.json,history.json,critic.pt}`。
- 汇总脚本：`runs/stage0/critic/analyze.py`（输出见 §2）。
- 冻结 baseline 读数：`runs/w2m_ctl__11__1790516900/agent.pt` 的 critic 在 val 上 EV=0.5222。
- 背景（历史参考）：`docs/experiments/ei3-value-loop-round1.md` §6（t_leafq EV≈0.55、cap0 宽头 ΔEV≤0、
  N-24 clean −1.37 Elo）。

## 5. 缺失证据 / 限制

1. **分布口径**：本探针用 `w2m_ctl` **确定性 self-play**（argmax、无 sampling、无 mix_random/pool）。
   PPO 实际训练分布叠加 sampling + mix_random 0.5 + pool 对手，其 within-decision 噪声**更高**，
   single-leaf EV 天花板会**更低**；因此本探针给出的是「信念值可学性」的上界，方向只会让结论更强。
2. **近似误差的归因**：~14% 学不出来的 `E[R|obs]` 未进一步拆「信息不在 obs v5 里」还是「MLP 表达不了
   信念计算」。但独立/加宽 trunk 都不优于已有 policy trunk，说明不是简单容量问题；更可能是 obs v5
   的 unseen 段只给多重集、不含可推出的信念结构（与 belief-posterior-probe 的 AUC 0.742 < 0.748 一致）。
   这属于 D-A（oracle-obs）与 belief 线的范围，不属 D-C。
3. **K=8 的 within 估计**：within-decision 方差集中在早局（decision 0 的 margin var≈0.33，末局≈0），
   K=8 的逐决策方差估计有采样噪声，但不影响总地板（within 点估计 0.1015 是无偏的）。
4. **单种子训练**：critic 训练只用 seed 0（recipe 对齐 EI-2/EI-3）；但 13 臂的 EV 全部稳定、且与
   champion 冻结 critic 交叉一致，重复训练种子的额外信息量低。
5. **未测目标**：n-step/TD 目标（GAE λ-return）未单独跑。它是 reward_shaping 契约内的低方差变体，
   机械上会抬高 EV（方差更小），但不改变「结构/数据不可约」的结论，故按「可选」未做。

## 6. 下一步建议

- **不投入 critic 结构改进**（本探针 + N-24 + cap0 三线同向）。
- 若 value 线还要走，唯一未测且可能有杠杆的是**改信息上界**（D-A oracle-obs / belief 后验），
  而不是改 critic 网络；但那是 O1/belief 路线，需单独立项并预注册。
- 优先级回归：search_leafq（推理期搜索，μ=260.03，唯一强杠杆）与产品/评测线。

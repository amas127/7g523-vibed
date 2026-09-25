# T1 / 结构性 A：奖励重构 500k（7鬼523）

> 资产状态（2026-09-25）：本报告引用的模型 checkpoint 已删除，旧路径不再可用；数值为牌型族规则变更前口径。

> **状态：已完成（2026-09-25；训练 seed 复现后结论修正）。一句话结论：seed 1 的四个
> head-to-head 比较（3×400 ×2 + 3×800 确认 ×2）95% CI 全部排除 0、点估计 +9.99…+12.75 Elo，
> 但补跑 seed 2/3/4（各 3×800 ×2）**未复现**：4-seed 的 8 个比较只有 seed 1 的两个 CI 排除 0，
> 4-seed 平均 +4.47（sd 5.6，vs `base680k`）/ +6.77（sd 4.4，vs `w5_ctrl`）。按统一口径
> （<10 Elo 一律不追）**A1 作为独立杠杆未复现、不成立**（§3.4），原始 pre-registered 结果
> 保留为历史记录。γ=1 消融不优、A2（纯胜负）单独不达标；**critic 判别实验（D0/D2，§2.4）
> 显示 H1（稠密奖励自足）与 H0（critic 驱动）都不被干净支持**。A 线收束；B 线（B0/B1）随后也已收束（B1 未确认，见 [`observation-augmentation-b1.md`](./observation-augmentation-b1.md)），下一优先见 [`../plans.md`](../plans.md) §8。**
>
> 判定口径：[`README.md` §3](./README.md#3-常用命令与口径) /
> [`evaluation-protocol-validation.md` §9](./evaluation-protocol-validation.md)；
> 预注册协议与预期：[`structural-directions.md` §2](./structural-directions.md)（§2.4 验收、§2.5 判读）。
> 同批控制组 `w5_ctrl` 的来源：[`wave5-500k-report.md`](./wave5-500k-report.md)。
> 全部训练/h2h/机制探针产物在 `runs/`（gitignore），本文数字均可溯源到其中文件。

## 1. 实现与协议

### 1.1 实现（`src/seven523/env.py`、`src/seven523/train.py`）

`Seven523Env` 新增 `reward_shaping: str = "terminal"`，四模式（`env.py:36-37`、`env.py:151-193`）：

| 模式 | 语义 | 一局求和 |
|---|---|---|
| `terminal`（默认） | 非终局步 0；终局步 `Game.returns()[learner]`（旧行为） | 终局分差回报 |
| `trick_diff` | 每步 `Φ(s')−Φ(s)`，`Φ = own/total − mean(others)/total`（`_potential`，与 `returns()` 同式） | 恒等于终局回报（telescoping） |
| `win` | 非终局步 0；终局步 `sign(own − max(others)) ∈ {−1, 0, +1}`（平局 0） | 胜负 sign |
| `trick_diff_win` | 每步 `ΔΦ` + 终局胜负 bonus | 终局回报 + sign |

`train.py` 新增 `--reward-shaping`（默认 `terminal`，`train.py:122-130`）并透传 `make_env`
（`train.py:217-227`、`train.py:438`）。默认 `terminal` 逐位不变：会话内做了改动前 HEAD
旧版 vs 新版的差分验证，12 个 seed 的 `obs`/`action_mask`/`reward`/`done` 全部逐位相同；
`tests/test_env.py` 另以多个测试固定契约（`terminal` 默认与未知模式报错、`terminal` 非终局步为 0、
`trick_diff` 一局奖励和 == 终局回报（含撬底局）、`trick_diff` 非零奖励步 ~10×、
`trick_diff_win` 只在终局步加 bonus）。
obs/动作空间/checkpoint 布局均未改动，`base680k` 与全部现成工具原样可用。

### 1.2 训练协议与产物（与 `w5_ctrl` 一致）

三臂同协议（seed 1、500k、`--opponent greedy`、SAME_STEP），仅 `--reward-shaping`/`--gamma`
不同；3 并发，三臂都跑满 **488 updates / 499,712 步**，`metrics.csv` 无 NaN：

```bash
.venv/bin/python -m seven523.train --exp-name t1_a1_trickdiff --seed 1 \
  --total-timesteps 500000 --cuda True --tensorboard True --checkpoint-interval 0 \
  --log-interval 50 --opponent greedy --load-checkpoint runs/probe/base_step00696320.pt \
  --reward-shaping trick_diff
# A1γ1 = 同上 + --gamma 1.0；A2 = 同上 --reward-shaping win
```

| 臂 | run 目录 | `agent.pt` sha256 | 墙钟 | 末尾训练 EV |
|---|---|---|---|---|
| A1 `trick_diff` | `runs/t1_a1_trickdiff__1__1790328779` | `b147ba5b5cb0e31c19bd0ac49d9f339fbaeb05525b6e4c7763c98fa822bd2d92` | 7m18s | 0.246 |
| A1γ1 `trick_diff` + `--gamma 1.0` | `runs/t1_a1_gamma1__1__1790328783` | `1ac0264ebf989ef13dbbe0f7192fefe2182482e36d384f9a18a6375711aea079` | 437s | 0.269 |
| A2 `win` | `runs/t1_a2_win__1__1790328779` | `bf09f14e657c86868e45f5d321f1b0a45b55d8db12e966cba7ae1bfda6455a5f` | 7m18s | 0.509 |

sha256 用 `sha256sum <run>/agent.pt` 复核；墙钟来自训练日志 START/END 与 run 元数据
（A1γ1 的 `runs/t1_a1_gamma1_meta.txt` 记 2026-09-25T09:33:02Z→09:40:19Z）。
**训练 EV 不可跨臂比较**：`explained_variance` 是各臂在自己奖励尺度上算的
（A1/A1γ1 ≈ 分差/100、A2 = ±1 的胜负），A2 的 0.509 不代表 A2 的 critic 更准（见 §2）。
原始三臂各只训练 1 个 seed（seed 1，母版协议）；A1 的 seed 2/3/4 复现见 §3.4，限制汇总见 §5。

## 2. 机制检查（预注册 vs 实测）

预注册（`structural-directions.md` §2.4）：离线复测 `V(s)` 的**早期分桶 EV**，
目标 `base680k` 的 0.065 → **>0.3**；训练日志 EV / value loss 作辅助。
### 2.1 预注册 vs 实测（V 对终局分差回报）

实测脚本 `runs/structural/value_accuracy.py`（600 局、seed 0、learner 固定 seat 0、
V 对终局分差回报）：

```bash
.venv/bin/python runs/structural/value_accuracy.py --checkpoint <ckpt> --episodes 600 --seed 0
# A2 用 --target win（V 对终局胜负 sign）
```

| 策略 | overall EV | 桶 [0,0.2) | [0.2,0.4) | [0.4,0.6) | [0.6,0.8) | [0.8,1.0] |
|---|---|---|---|---|---|---|
| `base680k`（参照） | 0.499 | **0.065** | 0.279 | 0.506 | 0.697 | 0.853 |
| A1 `trick_diff` | **0.016** | **0.041** | 0.067 | 0.077 | 0.034 | **−0.122** |
| A1γ1 | 0.043 | 0.041 | 0.064 | 0.093 | 0.069 | −0.041 |
| A2 `win`（`--target win`） | 0.369 | 0.014 | 0.194 | 0.347 | 0.543 | 0.667 |

**预注册机制假设未被证实**：A1/A1γ1 的 V 对真实终局回报几乎没有预测力
（overall 0.016/0.043，末桶为负），A2 的早期桶 0.014 也远低于 0.3
（其 overall 0.369 主要靠后半局）。即「逐墩分解让 critic 更干净」这一预期没有发生。

### 2.2 多目标任务探针（`runs/structural/value_probe_multi.py`）

脚本对每个 ckpt 固定重放 300 局（seed 0），把同一个 `V(s)` 分别对三个目标打分：
终局分差 `R`、**自身塑形折扣回报 `G`（γ=0.99）**、折扣终局回报 `Gterm`；输出
`runs/structural/out_value_multi.txt`：

| 策略 | V sd | V vs R（EV / corr） | V vs 自身塑形 G（EV / corr） |
|---|---|---|---|
| `base680k` | 0.302 | +0.486 / 0.699 | −0.723 / 0.143 |
| A1 `trick_diff` | **0.113** | **−0.030** / 0.240 | +0.103 / 0.325 |
| A1γ1 | **0.114** | **−0.030** / 0.294 | +0.122 / 0.352 |
| A2 `win` | 0.495 | **+0.249** / 0.685 | −2.588 / 0.117 |

读法：A1/A1γ1 的 V 几乎是常数（sd 0.113/0.114，而终局回报 sd ≈0.43），对**自身训练目标 `G`**
也只有 EV≈0.10、corr≈0.33；A2 的 critic 明显更健康（sd 0.495、V vs R EV +0.249、corr 0.685），
但 A2 的端点 null（§3）。

### 2.3 解读（假设，不是结论）

- **假设 H1（稠密奖励驱动）**：A1 的正效果可能来自奖励本身携带了逐墩信用信号——
  `A_t ≈ r_t + γλr_{t+1} + …` 近似退化为「稠密塑形回报 − 近常数基线」，
  优势估计不再依赖 critic 泛化。这与「critic 方差塌缩、但 h2h 变强」并存不矛盾。
- **假设 H0（critic 驱动，预注册预期）**：若改善来自更干净的价值面，早期分桶 EV 应上升；
  实测没有，H0 未获支持。
- A2 的 critic 是健康的（sd 0.495）但端点 null，说明「critic 更好」既不充分也不必要，
  机制仍未确定。
- **判别实验（已执行，见 §2.4）**：用 `trick_diff` + `--vf-coef 0`（D0）与 `--vf-coef 2.0`（D2）
  的对照臂区分 H1/H0；D0 没有复现 A1 的正效应，D2 反而显著更差，H1/H0 都不被干净支持。

### 2.4 critic 判别实验（D0 `--vf-coef 0` / D2 `--vf-coef 2.0`）

§2.3 把机制判别留作「待定」；判别的两个对照臂已跑完（与 A1 同配方：`trick_diff`、seed 1、
500k、greedy、热启动 `base680k`；快照 `/tmp/t7_disc_pkg`，只改 `vf_coef`）：

| 臂 | `--vf-coef` | run 目录 | `agent.pt` sha256 |
|---|---|---|---|
| D0（禁用 value loss） | 0.0 | `runs/t1_d_vf0__1__1790330246` | `e9c715e9e44d0f698e120a5b7ecf7d0d857dc511de6a5155638972a24c42a117` |
| D2（加大 value loss） | 2.0 | `runs/t1_d_vf2__1__1790330246` | `bebe7f93a26888c815e45be0f1d410feb07883551482390f712ceaff9796ccba` |

h2h（3 seed × 400 副牌、换座；左=臂，正 = 臂更强；JSON
`runs/h2h_t1_d_vf{0,2}_vs_{base680k,w5_ctrl}.json`）：

| 臂 | vs `base680k` | vs `w5_ctrl` |
|---|---|---|
| D0 | −4.20 [−16.14,+7.74] | +2.17 [−9.47,+13.82] |
| D2 | **−11.44 [−22.42,−0.47]** | −6.37 [−17.55,+4.81] |

离线探针（`runs/t1_disc/out_value_multi.txt`，每个 ckpt 固定重放 300 局、seed 0）：

| 量 | `base680k` | A1 | D0 | D2 |
|---|---|---|---|---|
| V sd | 0.302 | 0.113 | 0.361 | 0.108 |
| V vs 自身塑形 G 的 EV | −0.723 | +0.103 | **−0.913** | +0.117 |

GAE 分解（`runs/t1_disc/out_gae_decomp.txt`，200 局、seed 0、greedy replay）：

| 臂 | corr(A_GAE, A_dense) | OLS R² | value 项方差占比 |
|---|---|---|---|
| `base680k` | 0.738 | 0.544 | 0.464 |
| A1 | 0.906 | 0.822 | 0.190 |
| D0 | 0.584 | 0.342 | 0.659 |
| D2 | **0.922** | **0.849** | **0.162** |

辅助分桶 EV（`runs/t1_disc/out_value_accuracy_vf{0,2}.txt`）：D0 overall EV 0.026
（早期桶 [0,0.2) −0.348）、D2 overall EV 0.024（早期桶 +0.038）；两臂都没有恢复
`base680k` 的价值健康度。

**读数（单 seed、末态快照的限制下）**：

- **H1（稠密奖励自足）不被支持**：D0 关掉 value loss 后没有复现 A1 的正效应
  （vs `base680k` −4.20，vs `w5_ctrl` +2.17，均跨 0）；D0 的 A_dense 只占优势方差 0.371，
  且 V 仍然大（sd 0.361）——advantage 并非由稠密奖励单独承担。
- **H0（critic 驱动）也不被支持**：D2 把 `vf_coef` 加 4 倍后，优势结构几乎与 A1 相同
  （corr 0.922、R² 0.849、value 项方差占比 0.162，V sd 0.108、V vs G EV +0.117），
  但 h2h 显著更差（vs `base680k` −11.44，CI 排除 0）。
- **`vf_coef` 经共享 trunk 不是干净干预**：`vf_coef` 同时改变流回共享主干、进而改变
  policy 的梯度，D0/D2 的差异不能只归因于 critic 本身。
- **塌缩不是 value loss 权重不足**：D2（`vf_coef 2.0`）的 V 仍与 A1 一样塌缩
  （sd 0.108），加大 value loss 权重不能阻止塌缩。
- **限制**：每个臂只训练 1 个 seed、只探针末态快照；没有训练过程快照/早期分桶轨迹，
  因此上述机制结论是提示性的，不是判决性的。

## 3. 主端点（h2h，同牌换座）

工具 `tools/head_to_head.py`，`--device cpu --workers 8`（见 §6），每个对照 3 seed × 400 副牌
（每副双座位，合计 1200 副牌 / 2400 局，JSON `deals=1200`、`games=2400`）。
JSON 中 `left_id` 是臂、`right_id` 是参照，
**正 Elo = 臂更强**；下表数字直接来自
`runs/h2h_t1_{a1,a1g,a2}_vs_{base680k,w5_ctrl}.json` 的
`combined.elo_diff.mean/ci/bootstrap_se/between_seed_sd`、`combined.winrate`、
`combined.mean_score_diff`、`per_seed[].seed/elo_diff`（p = `combined.deal_sign.p_value`）。

### 3.1 3×400 主筛选

| 臂 | 参照 | mean | 95% CI | winrate | mean_score_diff | per-seed（Elo） | p |
|---|---|---|---|---|---|---|---|
| A1 | `base680k` | **+12.1663** | [+0.7657, +23.5669] | 0.5175 | +1.0583 | [11.2956, 16.5156, 8.6877] | 0.0434 |
| A1 | `w5_ctrl` | **+12.7468** | [+1.2614, +24.2322] | 0.5183 | +2.1542 | [14.3399, 6.9496, 16.9509] | 0.0366 |
| A1γ1 | `base680k` | −12.6040 | [−24.4251, −0.7829] | 0.4819 | −1.3375 | [−8.2531, −8.6877, −20.8712] | 0.0589 |
| A1γ1 | `w5_ctrl` | +3.7651 | [−6.9568, +14.4869] | 0.5054 | +0.2875 | [−0.4343, 0.8686, 10.8609] | 0.6178 |
| A2 | `base680k` | −7.5319 | [−18.9281, +3.8643] | 0.4892 | −0.7417 | [−15.2100, 2.6058, −9.9915] | 0.1617 |
| A2 | `w5_ctrl` | +8.5443 | [−3.2695, +20.3580] | 0.5123 | +1.3208 | [3.9088, 14.3399, 7.3841] | 0.0889 |

每个 seed 的对应文件为 `runs/h2h_t1_<arm>_vs_<ref>_seed{0,1,2}.jsonl`（逐局 JSONL）。
A1 两个比较的合并 SE 明细：vs `base680k` bootstrap SE 10.0747、between-seed sd 3.9859；
vs `w5_ctrl` bootstrap SE 10.1496、between-seed sd 5.1875（合并 `se` 取
`max(bootstrap_se, between_seed_sd)/√3`，与 JSON 的 `combined.elo_diff.se` 一致）。

### 3.2 A1 确认 3×800

只对唯一命中的 A1 做加牌数确认（每次 3 seed × 800 副牌，共 2400 副牌 / 4800 局）：

`runs/h2h_t1_a1_conf_vs_base680k.json`、`runs/h2h_t1_a1_conf_vs_w5_ctrl.json`：

| 参照 | mean | 95% CI | bootstrap SE | between-seed sd | winrate | mean_score_diff | per-seed（Elo） | p |
|---|---|---|---|---|---|---|---|---|
| `base680k` | **+9.9916** | [+2.1021, +17.8810] | 6.9719 | 0.6520 | 0.5144 | +1.0417 | [9.9915, 10.6435, 9.3396] | 0.01618 |
| `w5_ctrl` | **+12.6036** | [+4.0827, +21.1244] | 7.1430 | 7.5299 | 0.5181 | +2.2333 | [16.9509, 3.9088, 16.9509] | 0.00220 |

### 3.3 seed 1 判定（原始记录；最终判定见 §3.4）

> 本小节是 2026-09-25 早先基于 seed 1 的原始判定；补跑 seed 2/3/4 后结论修正为
> 「未复现」（§3.4）。本节点估计与 CI 保留为历史记录。

- **A1 的四个比较 CI 全部排除 0**，点估计 **+9.9916…+12.7468**：
  - 3×400：vs `base680k` +12.1663 [+0.7657,+23.5669]、vs `w5_ctrl` +12.7468 [+1.2614,+24.2322]；
  - 3×800 确认：vs `base680k` +9.9916 [+2.1021,+17.8810]、vs `w5_ctrl` +12.6036 [+4.0827,+21.1244]。
- 其中 **3×800 vs `base680k` 的 +9.99 恰好落在 +10 门槛之下（边缘）**；3×800 vs `w5_ctrl` 达标。
  总体判定：**真实但量级 ~+10 Elo 的正信号、贴门槛**。这是平台（`base680k` ~1443）以来
  第一个在预注册 h2h 口径下 CI 排除 0 的方案。
- **不声称越过 +20**：C-1/EPV §6 要求对 +20 做单侧等效/非劣检验并按功率表加牌数
  （5×400 或 3×800 复算），本报告未做该检验，也不使用「点估计 ≥ +20」表述。
- A1γ1 不达标：vs `base680k` CI [−24.43,−0.78] 完全为负（显著更差），vs `w5_ctrl` +3.77
  [−6.96,+14.49] 跨 0。
- A2 不达标：vs `base680k` −7.53 [−18.93,+3.86]、vs `w5_ctrl` +8.54 [−3.27,+20.36] 均跨 0。

### 3.4 训练 seed 复现（seed 2/3/4）与判定修正

§3.3 的判定只基于 1 个训练 seed（seed 1，母版协议）。为回答「正信号是否只是单 seed 抽样」，
用与 A1 完全相同的配方补训 seed 2/3/4（各 500k，run
`runs/t1_a1_trickdiff__{2,3,4}__1790331007`；各 `args.json` 与 seed 1 只差 `seed`），
并对每个 seed 各做两次 3×800 换座 h2h（`--pairs 800`，合计 2400 副牌 / 4800 局；左=A1）。
seed 1 用同一 3×800 口径的 `runs/h2h_t1_a1_conf_vs_*.json`（§3.2）。

| 训练 seed | vs `base680k` | vs `w5_ctrl` |
|---|---|---|
| 1（§3.2，conf） | +9.99 [+2.10,+17.88] | +12.60 [+4.08,+21.12] |
| 2 | +5.00 [−5.87,+15.86] | +5.21 [−2.74,+13.16] |
| 3 | −3.33 [−11.08,+4.42] | +2.17 [−5.65,+10.00] |
| 4 | +6.23 [−1.22,+13.67] | +7.10 [−4.47,+18.67] |
| **4-seed 平均（sd，n−1）** | **+4.47（5.6）** | **+6.77（4.4）** |

文件：`runs/h2h_t1_a1_s{2,3,4}_vs_{base680k,w5_ctrl}.json`（逐局 `_seed{0,1,2}.jsonl`）。
4 个 seed × 2 个参照共 8 个比较，7 个点估计为正，但 **只有 seed 1 的两个 CI 排除 0**；
seed 2–4 的 6 个比较 CI 全部跨 0。

**判定修正（按统一口径）**：

- 原 seed 1「命中」是单 seed 高值；训练 seed 之间的方差（sd 4.4–5.6）与效应同量级，
  4-seed 平均 ~+4.5 低于 +10 行动门槛。
- **A1 作为独立杠杆未复现/不成立**（<10 Elo 一律不追）。这不声称效应为 0：0 落在 CI 内只说明
  现有牌数分辨不出 ~+5 级效应；但没有任何 seed 再现 seed 1 的 +10 级点估计。
- 原始 pre-registered 的 seed 1 结果（§3.1–§3.3）保留为历史记录。
- 将来唯一值得再验的是 **A1+B1 组合臂**（B1 已定稿、未确认）；单独 A1 不再追。

## 4. 次端点（对 Greedy 的常规胜率）

`7g523-eval --episodes 1500 --opponent greedy`（固定 seat 0，不做换座，仅 sanity check）：

| 策略 | 胜/和/负 | 均分差 | mean return | 文件 |
|---|---|---|---|---|
| A1 | 64.2% / 7.3% / 28.5% | +22.03 | +0.220 | `runs/eval_t1_a1_1500.txt` |
| `base680k` | 63.0% / 7.3% / 29.7% | +20.82 | +0.208 | `runs/eval_base680k_1500.txt` |

两者都在平台 62–65% 带内，n=1500 的二项 SE≈1.2pp，差 1.2pp 不可分辨；
这个口径与 h2h 的结论方向一致但分辨不出 ~10 Elo。

## 5. 限制

1. **训练 seed 抽样已补、A1 未复现**：原始只有 seed 1；§3.4 补跑 seed 2/3/4 后
   4-seed 平均 +4.47/+6.77、仅 seed 1 的 CI 排 0，跨 seed sd（4.4–5.6）与效应同量级。
   A1γ1/A2 仍各只训 1 个 seed（消融用）。
2. **3×800 与 3×400 的牌不完全独立**（同 seed 下前 400 副牌相同），两轮不是免费的第二份样本。
3. **多重比较**：原始报告共 6 个「臂 vs 参照」对照（加确认共 8 个），未做 Bonferroni/FWER 校正
   （开源问题 Q-13）；后续又增加 seed 复现 8 个与 D0/D2 4 个比较，仍未校正。A1 的
   seed 1 结论应以两轮一致与 A1γ1/A2 的阴性作为缓解，复现后的汇总不依赖单点。
4. **`w5_ctrl` 是另一个训练 run**：父模型 `base680k` 与续训子代之间的父子偏移在不同牌集会
   整体翻符号（`wave5-500k-report.md` §4），它仍是噪声源，不是同 run 控制。
5. **机制未确定**（§2.3/§2.4）：判别实验 D0/D2 已做，H1/H0 都不被干净支持；
   `vf_coef` 经共享 trunk 不是干净干预，且只有单 seed + 末态快照。
6. **`trick_diff_win` 已实现但未单独立臂**，没有它的独立读数。
7. 次端点固定 seat 0，不是换座口径，只作方向性核对。

## 6. 工程伴随改动（2026-09-25，属 C-5 允许范围）

- **评估并行化**：`ladder.play_games(workers=N)` + `--workers`（`tools/build_ladder.py`、
  `tools/head_to_head.py`、`tools/h2h_screen.py`）。默认 `workers=1` 即原串行循环、逐位不变；
  `workers>1` 按 schedule 分片、spawn 进程并行，结果与串行**逐位一致**
  （`tests/test_ladder.py::test_play_games_parallel_matches_serial_bit_for_bit`、
  `test_build_ladder_workers_match_serial`、`test_cli_head_to_head_workers_match_single` 等）；
  h2h 3×100 副牌由 13.4s 降到 8.4s（`--workers 8`）。`--device cuda` 下每个 worker 会各开
  一个 CUDA context（`ladder.py:_warn_cuda_workers` 提醒显存风险）。
- **已确认 bug 修复**：`src/seven523/duel.py::_binomial_two_sided_p` 在 trials≥1600
  （即 `--pairs 800`）时 `float(1 << trials)` 溢出 → OverflowError；改为整数真除。
  `n≤800` 逐位不变（与 legacy 公式回归一致），新增 3 个回归测试
  （`tests/test_duel.py::test_binomial_two_sided_p_matches_legacy_formula_at_800_trials`、
  `..._does_not_overflow_trial_counts`、`..._nonpositive_trials_is_one`）。
  没有这个修复，3×800 确认的输出无法计算。

## 7. 结论与后续

1. **A1 正信号未复现，A 独立杠杆收束**：seed 1 的四个 h2h 比较 CI 全排 0
   （+9.99…+12.75）是单 seed 高值；补跑 seed 2/3/4（§3.4）后 4-seed 平均 +4.47
   （vs `base680k`）/+6.77（vs `w5_ctrl`）、8 个比较仅 seed 1 的两个 CI 排 0。
   按统一口径（<10 Elo 不追），**A1 作为独立杠杆不成立**；原始 pre-registered 记录
   保留（§3.1–§3.3）。
2. **消融结论仍有效**：γ=1 不优（vs `base680k` 显著为负）、A2（纯胜负）单独无效——
   「胜负口径对齐」单独不足以移动平台；单 seed 的 A1 配方也不是可交付杠杆。
3. **critic 判别（D0/D2，§2.4）**：H1/H0 都不被干净支持；`vf_coef` 经共享 trunk 不是干净
   干预，塌缩不是 value loss 权重不足（D2 `vf_coef 2.0` 仍塌缩且 h2h 显著更差）。
4. **下一优先 = T5 C（逐局对手/PFSP）**：T2 B0 null、T3 B1 未确认（[`observation-augmentation-b1.md`](./observation-augmentation-b1.md)）→ B 线收束；
   A1+B1 组合臂是 A 线唯一还值得再验的组合，但现门控下不排期（[`../plans.md`](../plans.md) §8）。
5. 限制不变：多重比较（Q-13）与 `w5_ctrl` 父子偏移（§5）；任何产品化声称需 ≥2 个训练 seed
   一致，且 4-seed 汇总必须用同一 h2h 口径。

## 8. 复现与产物

```bash
# 训练（三臂同协议；A1γ1 加 --gamma 1.0，A2 换 --reward-shaping win）
.venv/bin/python -m seven523.train --exp-name t1_a1_trickdiff --seed 1 \
  --total-timesteps 500000 --cuda True --tensorboard True --checkpoint-interval 0 \
  --log-interval 50 --opponent greedy --load-checkpoint runs/probe/base_step00696320.pt \
  --reward-shaping trick_diff

# 主端点 3×400（对 base680k / w5_ctrl 各一次）
.venv/bin/python tools/head_to_head.py \
  --left t1_a1=ckpt:runs/t1_a1_trickdiff__1__1790328779/agent.pt \
  --right base680k=ckpt:runs/probe/base_step00696320.pt \
  --seeds 0,1,2 --pairs 400 --device cpu --workers 8 \
  --games-out runs/h2h_t1_a1_vs_base680k.jsonl --json > runs/h2h_t1_a1_vs_base680k.json

# A1 确认 3×800（--pairs 800；依赖 §6 的 p 值溢出修复）
.venv/bin/python tools/head_to_head.py \
  --left t1_a1=ckpt:runs/t1_a1_trickdiff__1__1790328779/agent.pt \
  --right w5_ctrl=ckpt:runs/w5_ctrl__1__1790319698/agent.pt \
  --seeds 0,1,2 --pairs 800 --device cpu --workers 8 \
  --games-out runs/h2h_t1_a1_conf_vs_w5_ctrl.jsonl --json > runs/h2h_t1_a1_conf_vs_w5_ctrl.json

# 机制检查
.venv/bin/python runs/structural/value_accuracy.py --checkpoint <run>/agent.pt --episodes 600 --seed 0
.venv/bin/python runs/structural/value_probe_multi.py > runs/structural/out_value_multi.txt

# 次端点
7g523-eval --checkpoint runs/t1_a1_trickdiff__1__1790328779/agent.pt \
  --episodes 1500 --opponent greedy

# 训练 seed 复现（seed 2/3/4；协议与 A1 相同，只换 --seed）与两次 3×800 h2h（对 base680k）
for s in 2 3 4; do
  .venv/bin/python -m seven523.train --exp-name t1_a1_trickdiff --seed $s \
    --total-timesteps 500000 --cuda True --tensorboard True --checkpoint-interval 0 \
    --log-interval 50 --opponent greedy --load-checkpoint runs/probe/base_step00696320.pt \
    --reward-shaping trick_diff
  .venv/bin/python tools/head_to_head.py \
    --left t1_a1_s$s=ckpt:runs/t1_a1_trickdiff__${s}__1790331007/agent.pt \
    --right base680k=ckpt:runs/probe/base_step00696320.pt \
    --seeds 0,1,2 --pairs 800 --device cpu --workers 8 \
    --games-out runs/h2h_t1_a1_s${s}_vs_base680k.jsonl --json > runs/h2h_t1_a1_s${s}_vs_base680k.json
done
# 对 w5_ctrl：--right w5_ctrl=ckpt:runs/w5_ctrl__1__1790319698/agent.pt，文件名同式

# critic 判别（D0/D2；与 A1 同配方，只改 --vf-coef；快照 /tmp/t7_disc_pkg）
.venv/bin/python -m seven523.train --exp-name t1_d_vf0 --seed 1 \
  --total-timesteps 500000 --cuda True --tensorboard True --checkpoint-interval 0 \
  --log-interval 50 --opponent greedy --load-checkpoint runs/probe/base_step00696320.pt \
  --reward-shaping trick_diff --vf-coef 0.0
# D2 = 同上，--exp-name t1_d_vf2 --vf-coef 2.0

# D0/D2 h2h（3×400）
.venv/bin/python tools/head_to_head.py \
  --left t1_d_vf0=ckpt:runs/t1_d_vf0__1__1790330246/agent.pt \
  --right base680k=ckpt:runs/probe/base_step00696320.pt \
  --seeds 0,1,2 --pairs 400 --device cpu --workers 8 \
  --json > runs/h2h_t1_d_vf0_vs_base680k.json
# 对 w5_ctrl / D2 同理

# 离线探针（D0/D2）
PYTHONPATH=/tmp/t7_disc_pkg .venv/bin/python runs/t1_disc/value_probe_multi.py \
  --checkpoints base680k=runs/probe/base_step00696320.pt \
    a1=runs/t1_a1_trickdiff__1__1790328779/agent.pt \
    d0=runs/t1_d_vf0__1__1790330246/agent.pt \
    d2=runs/t1_d_vf2__1__1790330246/agent.pt > runs/t1_disc/out_value_multi.txt
PYTHONPATH=/tmp/t7_disc_pkg .venv/bin/python runs/t1_disc/gae_decomp.py \
  --checkpoints base680k=runs/probe/base_step00696320.pt \
    a1=runs/t1_a1_trickdiff__1__1790328779/agent.pt \
    d0=runs/t1_d_vf0__1__1790330246/agent.pt \
    d2=runs/t1_d_vf2__1__1790330246/agent.pt --out runs/t1_disc/out_gae_decomp.txt
```

产物：三个 run 目录 `runs/t1_*__1__*/{agent.pt,args.json,metrics.csv,tb}`、
`runs/t1_*_train.log`、`runs/t1_a1_gamma1_meta.txt`、
`runs/h2h_t1_{a1,a1g,a2}_vs_{base680k,w5_ctrl}.json` 与 `_seed{0,1,2}.jsonl`、
`runs/h2h_t1_a1_conf_vs_{base680k,w5_ctrl}.json` 与 `_seed{0,1,2}.jsonl`、
`runs/eval_t1_a1_1500.txt`、`runs/eval_base680k_1500.txt`、
`runs/structural/out_value_multi.txt`（探针脚本 `value_accuracy.py`、`value_probe_multi.py`）。
seed 复现与 critic 判别另加：`runs/t1_a1_trickdiff__{2,3,4}__1790331007/`、
`runs/t1_d_vf{0,2}__1__1790330246/`、
`runs/h2h_t1_a1_s{2,3,4}_vs_{base680k,w5_ctrl}.json`（与 `_seed{0,1,2}.jsonl`）、
`runs/h2h_t1_d_vf{0,2}_vs_{base680k,w5_ctrl}.json`、
`runs/t1_disc/{value_probe_multi.py,gae_decomp.py}` 与
`runs/t1_disc/{out_value_multi.txt,out_gae_decomp.txt,out_value_accuracy_vf0.txt,out_value_accuracy_vf2.txt}`。

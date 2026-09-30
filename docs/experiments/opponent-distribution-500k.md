# T5 / 结构性 C：逐局对手 + PFSP + 强成员 500k（7鬼523）

> 资产状态（2026-09-25）：本报告引用的模型 checkpoint 已删除，旧路径不再可用；数值为牌型族规则变更前口径。

> **状态：已完成（null，2026-09-25）。一句话结论：`EpisodeMixturePolicy`（逐局冻结成员）+
> `pfsp_weights`（每 K=100 局 `w∝(1−wr)²+ε`、Beta(prior=10) 收缩、平局计半、与均匀 0.5 混合）
> + `--pfsp*` 开关已实现；三臂 500k（`C_u` 逐局均匀 / `C_p` +PFSP / `C_s` +强成员 `base680k`；
> obs v1、seed 1、499,712 步、无 NaN）跑完。8 个 3×400 换座 head-to-head 的 95% CI **全部含 0**、
> 点估计 |Δ|≤7：PFSP 效应 `C_p−C_u` **+1.88 [−17.93,+21.69]**、强成员效应 `C_s−C_u`
> **+2.90 [−17.32,+23.12]**；Greedy 1500 四臂 62.5–63.4% 均在平台带内。→ **逐局冻结、PFSP、
> 强成员三个效应在本预算/单 seed 下均 null，不采用**。**
>
> **后续（2026-09-27）**：T5 null 的四个竞争解释已在 [`../selfplay-pool-plan.md`](../selfplay-pool-plan.md) §2.3 显式化为 AH1–AH6 并执行；结果（弱成员池、三臂全止损）见 [`selfplay-pool-diagnostics.md`](./selfplay-pool-diagnostics.md) 与 [`selfplay-pool-wave1.md`](./selfplay-pool-wave1.md)。
>
> 判定口径：[`README.md` §3](./README.md#3-常用命令与口径)；方向设计与验收：
> [`structural-directions.md` §4](./structural-directions.md)（T5/C）；路线图：
> [`../plans.md`](../plans.md) §3 T5。
> 全部训练/h2h/评估产物在 `runs/`（gitignore）；本文数字可溯源到 `runs/t5_report.md`、
> `runs/t5_health.json`、`runs/h2h_t5_*.json` 与 `runs/t5_*__1__1790332696/{args.json,metrics.csv,pfsp_weights.csv}`。

## 1. 背景：旧 mix/pool 不是 league

[`structural-directions.md` §1.3、§4.1](./structural-directions.md) 的一手量化：wave1–5 的
`--opponent mix/pool` 都是**逐决策**重抽成员（`MixturePolicy.act` 每次调用独立掷骰），
一局内平均换对手 ~11.6 次（mix50）/~18 次（`w5_pool4g`），对手身份是每步隐变量；
权重固定、没有任何胜率反馈；且 `w5_pool4g` 的四个快照（20k/57k/sp20k/sp40k）**全部弱于**
warm start 的 `base680k`，池内没有 ≥ 学习者的对手。因此：

- `w5_pool4g` 只能作为**反面参照**（逐决策 + 无反馈 + 全员更弱），不能当 C 的控制；
- T5/C 的控制必须是新臂 `C_u`（逐局均匀）；PFSP 的权重来源必须是学习者对每个成员的**实测胜率**；
- 强成员需要一个 ≥ 学习者的候选（本实验取热启动父模型 `base680k` 本身）。

## 2. 实现

### 2.1 逐局冻结（`src/seven523/policies.py`）

- 新增 `EpisodeMixturePolicy`（`policies.py:98`）：`start_episode()` 按当前权重抽一个成员并
  **冻结整局**，`act()` 只转发给该成员；成员为 `(weight, policy, member_id)` 三元组
  （二元组自动取 `"0"`、`"1"`… 为 id）。
- 暴露 `current_id`（本局成员）与 `finished_id`（上一局成员）；`finished_id` 在下一次
  `start_episode` 时才被改写，因此在 SAME_STEP auto-reset 发生前，训练循环可在终局步读到
  刚结束那局的成员。
- `set_weights()` 只改**未来**抽取的分布，不影响本局已冻结的成员。
- 旧 `MixturePolicy` 未动；`--pool-episode` 关闭时仍走旧的逐决策路径（逐位不变）。

### 2.2 PFSP 权重（`pfsp_weights`，`policies.py:195`）

`records` 是「成员 id → `(wins, draws, losses)`（学习者视角）」：

1. 观测胜率 `wr_i = (wins + 0.5·draws) / games`（平局计半）；
2. 先经 `Beta(prior/2, prior/2)` 向 0.5 收缩（`--pfsp-prior` 默认 10，即伪计数 5/5；
   `0` = 不收缩）：`wr_i = (0.5·prior + wins + 0.5·draws) / (prior + games)`；
3. 难度权重 `w_i ∝ (1 − wr_i)² + ε`（`--pfsp-epsilon` 默认 0.02），学习者越打不过的成员权重越高；
4. 与均匀分布按 `--pfsp-uniform-mix` 混合（默认 0.5 均匀 / 0.5 PFSP）后归一化，防遗忘。

每 K=100 局（`--pfsp-every`）重算一次；K 内权重不变。

### 2.3 训练接线（`src/seven523/env.py`、`src/seven523/train.py`）

- `env.reset` 对带 `start_episode` 的对手做鸭子类型调用（普通 Policy 无感），保证「一局一个成员」
  从 env 层成立（`env.py:350-354`）。
- `train.py` 新开关（默认全关 = 旧行为）：
  `--pool-episode`、`--pfsp`、`--pfsp-every K`、`--pfsp-uniform-mix M`、`--pfsp-epsilon`、
  `--pfsp-prior`；`--pfsp` 要求 `--opponent pool --pool-episode`，非法组合直接报错。
- 每个 env 的**每个非 learner 座位**各持一个 `EpisodeMixturePolicy`；SAME_STEP 终局步从
  `final_info.episode` 取本局回报符号（`+1/0/−1`）作为 outcome，记到 `finished_id` 对应成员的
  全局战绩（wins/draws/losses）；权重更新后 `set_weights` 推给所有活动 policy。
- 产物：`runs/<run>/pfsp_weights.csv`（`global_step, episodes, member_id, weight, win_rate,
  wins, draws, losses`，每个更新点每成员一行）+ TensorBoard `pfsp/weight/<member_id>`。

### 2.4 测试

T5 直接新增 **17 项**测试：`tests/test_episode_policy.py` **13 项**（逐局冻结、按权重抽取、
`current_id`/`finished_id`、`set_weights`、PFSP 单调/平局计半/Beta 收缩/ε 与均匀混合边界、
env 鸭子类型、`MixturePolicy` 旧行为不变）+ `tests/test_train.py` **4 项**（CLI 默认关、
旧路径不实例化新类、PFSP 冒烟与 csv 分布、依赖校验）。T5 落地时全套 **372 passed**
（基线 355、零回归）；M1 落地后当前全套 **378 passed**（`.venv/bin/python -m pytest -q
-p no:cacheprovider`，2026-09-25）。

## 3. 三臂与训练产物

母版协议（SD §7.1）：warm start `runs/probe/base_step00696320.pt`（`base680k`）、seed 1、
500k、SAME_STEP、`--obs-version 1`（三臂相同）；仅 C 的池语义不同。命令（C_p = 加 `--pfsp`；
C_s = 再加第 6 个成员 `1@ckpt:runs/probe/base_step00696320.pt`）：

```bash
.venv/bin/python -m seven523.train --exp-name t5_cu --seed 1 \
  --total-timesteps 500000 --cuda True --tensorboard True --checkpoint-interval 0 \
  --log-interval 50 --obs-version 1 --opponent pool --pool-episode \
  --pool-member 1@ckpt:runs/probe/base_step00204800.pt \
  --pool-member 1@ckpt:runs/probe/base_step00573440.pt \
  --pool-member 1@ckpt:runs/probe_sp/sp_step00020480.pt \
  --pool-member 1@ckpt:runs/probe_sp/sp_step00040960.pt \
  --pool-member 1@greedy \
  --load-checkpoint runs/probe/base_step00696320.pt
```

| 臂 | 配置 | run 目录 | `agent.pt` sha256（全量） | 步 / updates | episodes | 末 SPS | 墙钟 |
|---|---|---|---|---|---|---|---|
| `C_u` | 逐局均匀，PFSP 关，5 成员 | `runs/t5_cu__1__1790332696` | `888e9d23f0166f5e18c81d12383d5f4869148ba4372df7ec88256e7cc014106e` | 499,712 / 488 | 22,249 | 611 | ~819 s |
| `C_p` | `C_u` + `--pfsp` | `runs/t5_cp__1__1790332696` | `aaa50d12cbb390786ab2e70db70c6557f52b7e8adb15c2474202692bfcf17c56` | 499,712 / 488 | 22,464 | 603 | ~831 s |
| `C_s` | `C_p` + 强成员 `base680k`（共 6 成员） | `runs/t5_cs__1__1790332696` | `c5402c217091d7b64ed0d7d45dcbef1836b978aa404b31bd794437986f1ad465` | 499,712 / 488 | 22,502 | 598 | ~838 s |

- 三臂 `agent.pt` 可加载、`metrics.csv` 无 NaN、参数无 NaN；obs_dim 191 / obs_version 1；
  机器可读核验在 `runs/t5_health.json`（sha256、步数、episodes、SPS、`wall_clock_s_est`
  817.9/828.7/835.6 s 与上表 mtime 口径 ~819/831/838 s 同量级）。
- `C_s` 的第 6 个成员就是热启动父模型 `base680k`（`--pool-member
  1@ckpt:runs/probe/base_step00696320.pt`）；这是本预算/单 seed 下最顺手、但**不是**真正的
  ≥学习者的独立参照（见 §6）。

## 4. PFSP 权重演化

权重更新点全部在 `runs/t5_{cp,cs}__1__1790332696/pfsp_weights.csv`（`C_u` 无此文件，PFSP 关）：

| 臂 | 末次 greedy 权重 | 其余成员末次权重 | 末次 greedy 收缩胜率 |
|---|---|---|---|
| `C_p` | **0.1561** | 0.1996–0.2176（20k 0.200、573k 0.214、sp20k 0.218、sp40k 0.213） | 0.655 |
| `C_s` | **0.1290** | 0.1601–0.1801（20k 0.160、573k 0.177、sp20k 0.180、sp40k 0.174、`base680k` 0.180） | 0.657 |

学习者对 greedy 的实测胜率最高（0.65–0.66），PFSP 把它降到最低；其余成员胜率 0.48–0.54、
权重接近均匀。PFSP 的难度梯度来自「把最弱的 greedy 降权」，但（i）权重再与均匀 0.5 混合、
（ii）池内没有真正 ≥ 学习者的成员，梯度仍浅（§6）。

## 5. head-to-head（3 seed × 400 副牌，换座；正 = 左臂更强）

数字来自 `runs/h2h_t5_*.json`（combined：point = per-seed mean，CI = max(bootstrap SE,
between-seed sd)/√k·z）与 `runs/h2h_t5_*_seed{0,1,2}.jsonl`；正 = 左臂 Elo 更高。

| 左 | 右（池外 * ） | mean | 95% CI | bootstrap SE | 组间 sd | 左胜率 | 分差 | per-seed（seed 0/1/2） |
|---|---|---|---|---|---|---|---|---|
| `C_u` | `base680k` | −1.30 | [−12.03, +9.42] | 9.47 | 9.07 | 0.4981 | +0.56 | 3.04, 4.78, −11.73 |
| `C_u` | `w5_ctrl` * | +6.96 | [−11.18, +25.11] | 10.04 | 16.04 | 0.5100 | +1.11 | 25.23, −4.78, 0.43 |
| `C_p` | `base680k` | −6.23 | [−16.95, +4.49] | 9.47 | 8.26 | 0.4910 | −0.07 | 2.17, −6.52, −14.34 |
| `C_p` | `w5_ctrl` * | +0.14 | [−14.85, +15.14] | 10.18 | 13.25 | 0.5002 | +1.35 | −13.03, 0.00, 13.47 |
| **`C_p`** | **`C_u`（PFSP 效应）** | **+1.88** | **[−17.93, +21.69]** | 9.78 | 17.51 | 0.5027 | +1.00 | −18.26, 10.43, 13.47 |
| `C_s` | `base680k`（池内，非对照） | −0.72 | [−11.40, +9.95] | 9.43 | 7.80 | 0.4990 | −0.65 | −9.56, 2.17, 5.21 |
| `C_s` | `w5_ctrl` * | +2.32 | [−8.94, +13.58] | 9.95 | 4.17 | 0.5033 | +1.01 | 6.08, −2.17, 3.04 |
| **`C_s`** | **`C_u`（强成员效应）** | **+2.90** | **[−17.32, +23.12]** | 9.15 | 17.87 | 0.5042 | +0.33 | −15.65, 4.34, 20.00 |

- 8 个比较的 95% CI **全部含 0**，点估计 |Δ|≤7；两个方向性端点（PFSP、强成员）点估计
  +1.88/+2.90，远低于 +10 行动门槛。
- 组间 sd 4–18，与效应同量级；per-seed 符号在 ±13–25 Elo 间翻转（如 `C_p−C_u` 为
  −18.26/+10.43/+13.47）。
- 1 处与 `runs/t5_report.md` 表格的舍入差：`C_p` vs `w5_ctrl` 的 JSON mean 为
  0.144975…，报告写 +0.15，本表按两位小数取 **+0.14**（CI/SE/sd 完全一致）。

## 6. Greedy 1500 端点

`7g523-eval --episodes 1500 --opponent greedy`（固定 seat 0，SE≈1.25 pp≈16 Elo）：

| 策略 | 胜 / 和 / 负 | 均分差 | 文件 |
|---|---|---|---|
| `C_u` | 63.4% / 6.7% / 29.9% | +19.03 | `runs/eval_t5_cu_1500.txt` |
| `C_p` | 62.7% / 7.7% / 29.6% | +18.97 | `runs/eval_t5_cp_1500.txt` |
| `C_s` | 62.5% / 6.7% / 30.8% | +19.69 | `runs/eval_t5_cs_1500.txt` |
| `base680k` | 63.0% / 7.3% / 29.7% | +20.82 | `runs/eval_t5_base680k_1500.txt` |

四者都在 ~62–65% 的平台带内。注意 **greedy 是池内成员**（PFSP 对其降权），该端点不是
SD §4.4 预注册的「池外」检查；真正的池外参照是 `w5_ctrl` 与（对 `C_u`/`C_p` 而言的）`base680k`，
其上 h2h（§5）也全含 0。该端点分辨率 ±16 Elo，只能做 sanity；「未入池强成员」的池外胜率
检查在本实验中没有执行（池内没有可用作池外强参照的独立成员）。

## 7. 判定与风险

- **判定**：`C_p−C_u`（PFSP）、`C_s−C_u`（强成员）、`C_p/C_s` vs `base680k`/`w5_ctrl`
  共 8 个主端点比较 CI 全含 0、点估计 |Δ|≤7 → **逐局冻结、PFSP、强成员三效应均 null**，
  按统一口径（CI 排除 0 且点估计 ≥+10 才行动）收束，不采用。
- **风险**：
  1. 每个臂仅 **1 个训练 seed**，组间 sd（4–18）与效应同量级，不能排除 seed 噪声；
  2. `C_s` 的「强成员」是热启动父模型 `base680k`，父子偏移本身限制上限（vs `base680k`
     只有 −0.72），池内**仍没有真正 ≥ 学习者的独立参照**——这是本实验对 SD §4.2 强对手项
     的降级实现；
  3. PFSP 把最弱的 greedy 降权后，池内难度梯度仍浅，且再与均匀 0.5 混合，PFSP 与均匀的
     分布差异被压小；
  4. 8 个比较未做多重比较校正（每对 ~1.7–1.9% 单侧假阳，见 `plans.md` Q-13），但全部 CI 含 0
     且点估计 <10，校正不改变 null 判定。
- **机制层面**：`pfsp_weights.csv` 显示 PFSP 的方向性正确（最弱 greedy 权重最低），
  但效应量级 ~+2 Elo；逐局冻结本身（`C_u` vs `base680k`/`w5_ctrl`）也在噪声内。
  若要继续对手分布线，需要「真正的 ≥ 学习者对手」（采样版 `base680k`、1-ply 搜索 bot）
  而不是父模型复制；本实验没有覆盖这一点。

## 8. 复现

```bash
# 训练：见 §3（C_u / C_p / C_s 仅 --pfsp 与第 6 个成员不同）
# 健康度核验
cat runs/t5_health.json
# h2h（8 对，runs/h2h_t5_*.json；本文件表格）
.venv/bin/python - <<'PY'
import json, glob
for f in sorted(glob.glob('runs/h2h_t5_*.json')):
    d = json.load(open(f))
    print(f, d['combined']['elo_diff'])
PY
# Greedy 端点
cat runs/eval_t5_{cu,cp,cs,base680k}_1500.txt
# PFSP 权重
column -s, -t runs/t5_cp__1__1790332696/pfsp_weights.csv | tail -6
column -s, -t runs/t5_cs__1__1790332696/pfsp_weights.csv | tail -7
```

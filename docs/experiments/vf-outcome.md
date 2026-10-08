# 跳变奖励下的胜负分解价值头（`--vf-outcome`）初步 screen（k=1）

> **状态：已完成（2026-10-07，k=1 = seed 1）。强度端点 null（−0.43 [−12.95,+12.09]，
> 1200 副/2400 局），机制成立（胜负头 AUC 0.82–0.88）。** 实现与单测已落地
> （`uv run pytest` 217 passed / 7 skipped，env/networks/ppo/train 四个文件；
> 既有 `test_fit_trace_prior` 的 manifest 漂移失败与本项无关）。
> 预注册后发生过一次 rollout 缓冲区污染缺陷，修正记录见 §4。
>
> **口径**：revision-3（出空即撬底）+ 观测 v5 + 当前工作树默认配方
> （[ADR-0016](../adr/0016-default-training-recipe-arcsin-lr-floor.md)：arcsin α=0.5、λ=1.0、
> lr floor 1e-5；[ADR-0015](../adr/0015-continuation-pool-random-mix.md)：pool-episode
> 6×t17 `3@` + `2@random`）。跨 fit 绝对 Elo 不可比；本 screen 只做同 fit 候选对候选 h2h。
>
> **产物**：`runs/vf-outcome/`（训练、h2h、探针、日志）。
> **代码**：`src/seven523/{env,networks,ppo,train,metrics}.py`；测试
> `tests/test_{env,networks,ppo,train}.py`。

## 0. 一句话

默认奖励在 50 分胜/负边界含一个离散跳变（`r_T = base(s_T) + λ·seat_outcome`）。变种让
**价值头只拟合去掉跳变的 return**，另加一个**胜负头**预测跳变的价值，并用它作为跳变流的
优势基线；初步 screen 检验 500k 从零下该分解对同配方 `shared` 对照的 h2h 强度。

## 1. 机制

跳变只在终局出现，而胜负无法从早期状态精确预知，critic 必须同时拟合平滑的 margin 分量与
离散的胜负跳变。变种把价值拆成两项：

```
V(s) = V_base(s) + λ · u(s),    u(s) = E[ γ^(T−t) · seat_outcome | s ]
```

- **critic（价值头）**：只拟合**跳变-free** 流，训练目标是去掉跳变项的 λ-return
  （终局目标 = 成形后的 margin，中间状态 = 跳变-free 的折扣回报）；
- **胜负头**：`Linear(hidden, 1)` + `tanh`，回归 `u(s)`（折后胜负）；
- **优势**：GAE 对 `(reward, value)` 是线性的，所以
  `A_total = A_base + A_jump` 是精确拆分——base 流用 `V_base` 作基线，jump 流用
  `λ·head` 作基线，终局奖励仍是真实的 `base + λ·outcome`。critic 只吃 `A_base + V_base`
  的 return 目标。

**为什么回归折扣目标而不是 3 类分类**：跳变在终局收到，因此它在状态 `s_t` 的价值是
`γ^(T−t)·outcome`，折扣是价值分解的精确组成部分。3 类分类只给 `P(y|s)`；若直接用未折后
概率作基线，会系统性高估早期状态的跳变价值（γ=0.99、典型 20–30 步时 `γ^(T−t)` 低到
0.74–0.82），相当于惩罚长局面的胜局。回归头的输出符号仍是胜负预测，探针按
符号准确率 / AUC / 校准统计。

## 2. 实现映射

| 文件 | 改动 |
|---|---|
| `env.py` | 新增 `reward_jump_scale` 属性（`arcsin`/`terminal_win` = `win_jump`，`win`/`trick_diff_win` = 1，其余 0）；终局 `info["reward_jump"] = scale·outcome`，trainer 不再复制奖励公式 |
| `networks.py` | `vf_outcome` 头（+129 参数，`tanh`）；`get_outcome` / `get_value_and_outcome`（一次主干前向同时给 critic 与胜负头）；checkpoint 写 `vf_outcome` 字段；与旧 ckpt 热启动时主干/actor/critic 照拷、胜负头新初始化；与 `vf_nll` 互斥 |
| `ppo.py` | `RolloutBatch.outcome_targets/outcome_mask`（可缺省，两者同进退）；`PPOConfig.vf_outcome/outcome_coef`；胜负头掩码 MSE（只在 rollout 内完局的样本上有标签，NaN 准确率表示该 update 无标签） |
| `train.py` | 两流 GAE；折后胜负标签回填（`_episode_outcome_labels`：`γ^(step−t)·outcome`）；`--vf-outcome` / `--outcome-coef`；要求奖励模式含离散跳变且不与 `--vf-nll` 同用 |
| `metrics.py` | CSV / TensorBoard 新增 `outcome_loss`、`outcome_accuracy` 列（非 `vf_outcome` run 为空） |

## 3. 预注册（写于训练启动前）

- **两臂**（各 1 run，seed 1，从零 500k，同配方同超参）：

  | 臂 | 配置 |
  |---|---|
  | `vfo_base` | `arch=shared`，默认配方，pool-episode（ADR-0015） |
  | `vfo_outcome` | 同左 + `--vf-outcome true --outcome-coef 1.0` |

- **训练命令**：`runs/vf-outcome/run_screen.sh`（2 臂并行；超参与 head-depth H2 一致：
  Adam、linear anneal、8×128、minibatch 4、epochs 4、lr 2.5e-4、`--eval-interval 50`）。
- **端点**：候选对候选 h2h `Δ = vf_outcome − vfo_base`，3 deal seed（9600/9601/9602）×
  400 副 × 换座（n=1200 副 / 2400 局），bootstrap 4000（`tools/head_to_head.py`）。
- **判定**（`experiments/README.md` §3 统一口径）：CI 排除 0 且点估计 ≥ +10 才谈
  「值得行动」；点 < +5 止损；k=1 只作 preliminary，不作确认性结论。
- **机制读数（非门）**：`metrics.csv` 的 `outcome_loss/outcome_accuracy` 曲线；
  独立探针 `runs/vf-outcome/probe_outcome.py` 报胜负头在随机 / 池成员对手下的符号准确率、
  AUC 与校准。两臂的 `explained_variance` 目标不同（跳变-free vs 总 return），不可直接比。

## 4. 前置修正记录（正式 screen 前）

首轮 screen（15:51 启动）在跑到 ~285k 时被中止：探针
（`runs/vf-outcome/probe_outcome.py`，对 t17long 与 random 各 30–40 局）显示 critic
输出的 "跳变-free" 值域为 **mean ≈ 5.1、max ≈ 10.7**，而弧弦模式的基础回报上界是 1。
训练侧同现象：`value_loss` 全程 1.5–2.7（base 臂 0.16–0.24），`explained_variance`
仍 0.64–0.75——即 critic 与自举目标一起漂到了错误尺度，而不是没学到。

**根因**：`--vf-outcome` 的四张 rollout 缓冲（`jump_rewards` / `values_jump` /
`outcome_targets` / `outcome_mask`，以及 `segment_start`）只在 `train()` 入口分配一次。
每轮 rollout 只写「本轮完局」的单元格，其余单元保留上一轮的旧跳变；于是后续轮次里
`rewards − jump_rewards` 在**非终局步**混入 ±1 的伪奖励，outcome 掩码也累积了旧标签，
自举目标随之漂移。

**修正**：四张缓冲改为每轮 update 开头重新分配零张量（`segment_start` 同时归零），
并加回归测试 `tests/test_train.py::test_train_vf_outcome_buffers_are_fresh_per_rollout`
（不变量：弧弦模式两条分解奖励流的非零项必须落在终局转移上；在旧代码上该测试失败）。
全套 `pytest` 217 passed / 7 skipped（env/networks/ppo/train 四个文件）。

正式 screen 于 16:00:45 以相同配置重启，预注册端点不变。

## 5. 结果

两臂各 488/488 update、无 NaN；训练 16:00:45–16:12:42。

### 5.1 训练读数

| 臂 | 末次 EV | 后半（>400）均值 EV | 后半均值 value_loss | 末次 vs-random eval（ret / score / win） |
|---|---:|---:|---:|---|
| `vfo_base` | 0.566 | 0.617 | 0.212 | 0.557 / 77.85 / 0.91 |
| `vfo_outcome` | 0.599 | 0.647 | **0.023** | 0.606 / 80.30 / 0.93 |

两臂的 EV 目标不同（总 return vs 跳变-free return），数值不可直接比；可读的是同臂内部的
绝对 scale：修复缓冲区后 `vfo_outcome` 的 `value_loss` 从 ~2.4 降到 **0.02–0.03**，critic
值域回到 ±1 量级。`outcome_accuracy` 列是折后目标上的严格符号匹配（近零预测按符号算错，
含平局），训练中 0.64–0.84；机制探针的读数见 5.3。

### 5.2 强度端点（h2h）

`vfo_outcome − vfo_base`，3 deal seed × 400 副 × 换座（1200 副 / 2400 局），bootstrap 4000：

| seed | Δ Elo | 95% CI | winrate |
|---|---:|---|---:|
| 9600 | +7.38 | [−13.92, +29.18] | 0.511 |
| 9601 | −0.87 | [−21.74, +20.87] | 0.499 |
| 9602 | −7.82 | [−30.04, +14.34] | 0.489 |
| **合并** | **−0.43** | **[−12.95, +12.09]** | **0.4994** [0.4814, 0.5174] |

合并 mean_score_diff −0.43 [−2.27,+1.41]；deal-sign p=1.0。**判定**：CI 含 0 且点估计
−0.43 < +5，按预注册止损口径，本 screen 不支持该变体带来 ≥+5 的强度变化；k=1
（单训练 seed）不宣布增益/无增益的确认性结论。

### 5.3 机制探针（`runs/vf-outcome/probe_outcome.py`，贪心轨迹）

| 对手 | 局数 | 决策数 | W/D/L | 符号准确率 | AUC | implied 分桶胜率 | V 范围 | composed 总价值 EV |
|---|---:|---:|---|---:|---:|---|---:|---:|
| random | 300 | 6881 | 261/9/30 | **0.884** | **0.879** | 0.19 / 0.65 / 0.86 / 0.98 | [−0.66, +1.09] | 0.357 |
| t17long s1999872 | 200 | 3861 | 87/24/89 | 0.745 | 0.817 | 0.07 / 0.30 / 0.45 / 0.80 | [−0.87, +0.88] | 0.445 |

- 胜负头确实预测了「该局胜负」：符号准确率 0.74–0.88、AUC 0.82–0.88，且校准单调
  （implied `E[y]` 从 −0.8 到 +0.8 的四档实际胜率跨 0.07→0.98）。
- 价值头值域与基础回报上界（±1）一致，说明修复后的 critic 确实在拟合「return − 跳变」；
  composed（`V_base + λ·head`）对真实终局总回报的 EV 为 0.36–0.45（贪心轨迹、对手域偏 OOD）。
- 产物：`runs/vf-outcome/h2h_vfo_outcome-vs-base_s1.json`、`runs/vf-outcome/games/*.jsonl`、
  两次训练的 `metrics.csv`/`agent.pt`。

## 6. 读法与下一步

- **强度端点 null、机制成立**。变体把「胜负」从 critic 里拆了出来且没有损失强度（−0.43，
  CI 半宽 ~13），但没有证据支持 ≥+5 的收益；按 <+5 止损，本 k=1 screen 不再追加 500k 组合臂。
  确认需要 ≥3 个训练 seed（训练 seed sd 历史 8–17 Elo，正是当前 CI 半宽的量级）。
- **对部署/搜索的意义**：critic 的 `get_value` 现在返回跳变-free 值，推理期搜索若用该
  checkpoint 做 value 截断，必须用 `V_base + λ·head`（`get_value_and_outcome`）才是总价值；
  直接读 `get_value` 会系统性地缺掉跳变项。这是后续「搜索用胜负头」路线的接口。
- **未测**：`--outcome-coef` 扫描；1M+ 续训是否让分解的优势显现；胜负头不共享 trunk
  （detach 辅助损失）的对照——本轮修复后 scale 已正常，detach 只作为后备设计假设。
- **可复现命令**：`runs/vf-outcome/run_screen.sh`（两臂）→ `runs/vf-outcome/run_h2h.sh`
  （端点）→ `runs/vf-outcome/probe_outcome.py`（机制）。

# T6 / 结构性 F：独立 actor/critic 双塔 500k（7鬼523）

> 资产状态（2026-09-25）：本报告引用的模型 checkpoint 已删除，旧路径不再可用；数值为牌型族规则变更前口径。

> **状态：已完成（负 / 关闭，2026-09-25）。一句话结论：三臂 500k（主臂 `t6_towers`、
> 交互臂 `t6_towers_trickdiff`、从零 `t6_towers_scratch`）与 6 组 h2h 3×400 均跑完 ——
> 双塔相对 shared **不可分**（vs `base680k` −9.13 [−20.57,+2.31]、vs `w5_ctrl`
> +5.21 [−5.93,+16.36]，均跨 0 且点估计 < +10）；交互臂相对 A1 **显著更差**
> （−15.51 [−27.10,−3.92]）；从零对照相对 `w5_scratch` **−20.89 [−33.81,−7.97]**。
> 机制探针显示「共享主干梯度干扰」前提被**证伪**（cos(g_pol,g_val)=+0.049），两塔确实
> 分叉但 critic 照旧塌缩 → 按 [`structural-directions.md` §6A.5](./structural-directions.md)
> **关闭「架构忠实度」线**，不做 E 组合、不加 Tanh 臂。**
>
> 判定口径：[`README.md` §3](./README.md#3-常用命令与口径) /
> [`evaluation-protocol-validation.md` §9](./evaluation-protocol-validation.md)；
> 设计与实现分析：[`structural-directions.md` §6A](./structural-directions.md)（§6A.5 判读、
> §6A.6 实现设计）；路线图：[`../plans.md`](../plans.md) §3 T6。
> 全部训练/h2h/探针产物在 `runs/`（gitignore）；机读汇总在 `runs/t6/health.json`，
> 完整工作笔记在 `runs/t6_report.md`，本文数字均可溯源到其中文件。

## 1. 背景

T6/F 是 [`structural-directions.md` §6A](./structural-directions.md) 的低预期、可证伪方向：
把 `networks.py` 的共享主干（一个 MLP 同时接策略头与价值头）换成**两个同规格独立 MLP 塔**，
对齐参考 `ppo.py` 的双塔做法。对齐审计 A5 已记录「共享主干」是相对参考栈的**有意偏差**
（ADR-0003 指定源 `ppo_multidiscrete_mask.py` 就是共享 trunk）；本实验检验的假设是
**policy/value 梯度在同一组主干参数上互相干扰**——若成立，分离塔应改善 60k–200k 爬升段的
优化；反方证据是 EV/KL 一直稳定、早期 EV 低更像价值目标噪声而非表征共享。

**变量隔离**：主臂 = 双塔 + ReLU（只改架构）；其余配方钉住 T1/T2 的母版协议
（seed 1、500k、`--opponent greedy`、`--obs-version 1`、SAME_STEP）。不开「双塔+Tanh」附加臂
（wave4 已把 tanh 判到 −48.8 Elo，见 ALS §5），也不做 E（容量）组合——两者都只在 F 出现正信号
时才谈（§6A.5）。

## 2. 实现与兼容

| 文件 | 改动 |
|---|---|
| `src/seven523/networks.py` | `Agent(arch="shared"\|"towers")`：towers 建 `actor_network`/`critic_network` 两个同规格 MLP（各自正交初始化），头名与 shared 的 state_dict key 不变；新增 `_actor_hidden`/`_critic_hidden`/`policy_logits`；`get_value`/`get_action_and_value` 两架构同接口（shared 仍一次前向）；`save_agent`/`load_agent` 增 `"arch"`（旧档缺省 shared）；`warm_start_into` 增跨架构 key 重映射与首层 pad（守卫复用 `observation_num_players`）；`NeuralPolicy.act` 改用 `policy_logits` |
| `src/seven523/train.py` | `--arch {shared,towers}`（默认 shared）；learner 与 `_self_play_opponents` 的 `Agent(...)` 均传 `arch`；严格重载条件加 `loaded.arch == args.arch`；warm-start 打印带 arch |
| `tests/test_networks.py` | 新增 9 测试（详见 `runs/t6_report.md` §1） |
| `tests/test_train.py` | 新增 4 测试（`parse_args` 默认/非法值、towers 训练冒烟、towers self-play 冒烟、shared v1→towers v3 训练热启动） |

默认 `shared` 路径**逐位不变**（既有 warm-start/训练/观测测试零改动全过）。跨架构语义：

- **shared→towers（同 obs）**：源 `network.*` 拷两份进两塔、heads 原样，step 0 policy/value
  **逐位相同**（`runs/t6/out_warmstart_bitwise.txt` 用真 `base680k` 验证：12 个张量、
  value/policy 均 `True`）；但 update 1 起两塔即分叉。
- **towers→towers**：逐键逐位；首层 pad 已泛化到两塔。
- **towers→shared**：取 **actor 塔** 拷进 `network`，policy 逐位、**value 不逐位**
  （「旧 critic 读 actor 塔」；唯一有损方向，本任务训练未用到）。
- **跨版本**：shared v1→towers v3、towers v1→towers v3 均对旧前缀逐位、新列零贡献；
  玩家数/宽度守卫保持（shifted equal-width 拒绝用例已补 towers 目标）。

**测试**：`.venv/bin/python -m pytest -q -p no:cacheprovider` → **418 passed**
（T6 贡献 +13；其余 +27 来自并行落地的无关注册/placement/play 改动；测试名清单见
`runs/t6_report.md` §1）。

## 3. 三臂核验

命令见 `runs/t6/launch_train.sh`；共同参数：seed 1、500000 请求步（488 updates =
499,712 实际步）、`--opponent greedy --obs-version 1 --activation relu
--checkpoint-interval 0 --log-interval 50`，前两臂 `--load-checkpoint
runs/probe/base_step00696320.pt`。机读：`runs/t6/health.json`。

| 臂 | run 目录 | `agent.pt` sha256 | 步 | episodes | 墙钟 |
|---|---|---|---|---|---|
| t6_towers（terminal） | `runs/t6_towers__1__1790334485` | `c4f5535c…` | 499,712 | 20,512 | 421 s |
| t6_towers_trickdiff | `runs/t6_towers_trickdiff__1__1790334485` | `fc253a5f…` | 499,712 | 20,597 | 422 s |
| t6_towers_scratch | `runs/t6_towers_scratch__1__1790334485` | `e7a0b119…` | 499,712 | 19,790 | 422 s |

三臂 obs 191/v1、参数量 100,107（shared 为 59,019；差 41,088 = 单塔主干）；488 行
`metrics.csv` 无 NaN、参数无 non-finite；warm-start 两臂首行均打印
`copied 12 tensors (arch shared->towers)`（完整 sha 与 args 见 `runs/t6/health.json`）。

## 4. head-to-head（3 seed × 400 副牌、换座；正 = 左臂更强）

JSON：`runs/h2h_t6_*.json`；逐局 `runs/h2h_t6_*_seed{0,1,2}.jsonl`；汇总表在
`runs/t6/h2h_summary.md`。

| 左（T6 臂） | 右 | mean Elo | 95% CI | per-seed |
|---|---|---|---|---|
| t6_towers | base680k | −9.13 | [−20.57,+2.31] | −10.86,+1.74,−18.26 |
| t6_towers | w5_ctrl | +5.21 | [−5.93,+16.36] | +5.21,+2.61,+7.82 |
| t6_towers_trickdiff | base680k | −6.66 | [−17.76,+4.43] | −7.82,−0.87,−11.30 |
| t6_towers_trickdiff | w5_ctrl | +9.27 | [−2.05,+20.58] | +11.30,+9.99,+6.52 |
| **t6_towers_trickdiff** | **t1_a1_trickdiff** | **−15.51** | **[−27.10,−3.92]** | −23.49,−4.34,−18.69 |
| **t6_towers_scratch** | **w5_scratch** | **−20.89** | **[−33.81,−7.97]** | −15.65,−13.03,−33.98 |

按统一口径（合并 95% CI 排除 0 且点估计 ≥ +10 才谈行动）：前 4 个比较的 CI 全部含 0、
点估计 −9.13…+9.27，**没有一对正向可分**。交互臂对 A1 的 **−15.51 [−27.10,−3.92]**
是唯一排除 0 的交互比较（点 < −10）：双塔在 `trick_diff` 上显著劣于共享主干。从零对照对
`w5_scratch` **−20.89 [−33.81,−7.97]**：等预算下参数 +70% 明显更差。

## 5. 机制检查

### 5.1 预注册的梯度干扰探针（先做，不训练）

`runs/t6/grad_cosine.py` → `runs/t6/out_grad_cosine.txt`：在 `base680k` 上取固定 rollout
batch（8×128），精确 PPO policy/value loss 对共享主干 `network.*` 的梯度，
**cos(g_pol,g_val) = +0.049**（0.weight +0.033、0.bias +0.084、2.weight +0.056、
2.bias +0.066；|g_pol|=0.071、|g_val|=0.151）。近 0 → 「共享主干梯度干扰」前提在
warm-start 点**被证伪**，与端点 null 一致。

### 5.2 两塔分叉

`runs/t6/tower_divergence.py` → `runs/t6/out_tower_divergence.txt`：相对 `base680k` 首层
权重，t6_towers 的 actor/critic 分别动了 **0.178/0.094**，两塔彼此差 **0.201**
（trickdiff 0.192/0.138、彼此 0.238；共享对照 `w5_ctrl` 相对 base 动 0.170）。
两塔确实解耦，分离本身没有换来强度。

### 5.3 价值精度与 trick_diff 塌缩（600 局 seed 0）

`runs/t6/value_accuracy.py` → `runs/t6/out_value_accuracy.txt`：

| ckpt | arch | overall EV | 首桶 [0,0.2) EV | V sd |
|---|---|---|---|---|
| base680k | shared | 0.499 | 0.065 | 0.304 |
| w5_ctrl | shared | 0.492 | 0.052 | 0.297 |
| t6_towers | towers | 0.485 | 0.032 | 0.296 |
| t6_towers_trickdiff | towers | **0.014** | 0.042 | **0.113** |
| t6_towers_scratch | towers | 0.517 | 0.075 | 0.314 |
| a1_trickdiff | shared | **0.016** | 0.041 | **0.114** |

**独立 critic 塔没有避免价值塌缩**：交互臂的 overall EV、首桶 EV、V sd 与共享主干的 A1
逐位相近（0.014 vs 0.016、0.042 vs 0.041、0.113 vs 0.114）——塌缩是 `trick_diff` 奖励
目标本身的性质（A 线已记录），不是共享主干造成的。

多目标探针（300 局 seed 0，`runs/t6/value_probe_multi_t6.py` →
`runs/t6/out_value_probe_multi.txt`）同向：交互臂对塑形目标 G 略有改善
（EV vs G **+0.136** vs A1 **+0.103**，corr +0.379 vs +0.325），但对终局 R/Gterm 仍塌缩
（EV −0.057/−0.046），V sd 0.111 ≈ A1 0.113（同探针）。

### 5.4 训练曲线

`runs/t6/summarize_metrics.py` → `runs/t6/out_metrics.txt`（窗口均值，488 updates）：
t6_towers 的 EV 全程 0.66→0.69，与 `w5_ctrl`（0.66→0.69）重合；t6_towers_trickdiff 的
EV 0.12→0.21（末行 0.247）与 A1 0.09→0.20（末行 0.246）轨迹重合；policy/value loss、KL、
clipfrac 无异常。

## 6. 判定

1. **双塔相对 shared 没有移动平台**：t6_towers 对 base680k −9.13 [−20.57,+2.31]（CI 跨 0）、
   对 w5_ctrl +5.21 [−5.93,+16.36]（CI 跨 0），点估计均 < +10；训练曲线与 `w5_ctrl` 重合。
2. **交互臂相对 A1 变化为负、critic 仍塌缩**：−15.51 [−27.10,−3.92] 是唯一排除 0 的
   交互比较；value 探针与 A1 几乎一样。
3. **关闭「架构忠实度」线**：按 §6A.5，双塔不可分且更差、交互臂显著负、梯度干扰前提被
   证伪（cos≈0.05）、从零对照 −20.9 → **不做 E 组合、不加 Tanh 臂**，预算留给其他方向。
   该关闭结论进入 [`../plans.md`](../plans.md) §6 负面清单 N-20。

## 7. 限制与风险

- **单训练 seed**：每个臂只有 1 个 500k run；组间 sd 2.5–11.4 与多数效应同量级
  （scratch 的 −20.9 超出该带）。若未来要重新打开该线，先补 seed（母版协议不变）。
- **跨架构热启动的近似性**：shared→towers 同 obs 在 step 0 逐位相同（已验证），但 update 1
  起两塔即分叉；towers→shared 的 value 不逐位（未用于本次训练）。结论只在 v1/ReLU/500k
  配方下成立。
- **交互臂与 A1 的对比跨训练时间/代码版本**（A1 为 2026-09-25 早期 run，同为 seed 1），
  但两者共享同一 warm start、奖励与 greedy 配方，且 CI 远离 0。
- **梯度余弦只在 `base680k` 单 batch 上测**；关闭决定不依赖该单点（端点 + EV + 从零对照
  共同支撑）。
- 兼容层（`arch` 字段、跨架构热启动、首层 pad）保留在源码中，但**不再排实验**；
  「towers + Tanh」组合（完整 `ppo.py` 配方）与 E 组合均不追加。

## 8. 复现与产物

```bash
# 三臂训练（seed 1、obs v1、arch towers；前两臂 warm start，scratch 无 --load-checkpoint）
bash runs/t6/launch_train.sh

# 6 组 h2h（3 seed × 400 副牌、换座、CPU 8 workers）
bash runs/t6/run_h2h.sh

# 机制探针
.venv/bin/python runs/t6/grad_cosine.py
.venv/bin/python runs/t6/tower_divergence.py
.venv/bin/python runs/t6/value_accuracy.py
.venv/bin/python runs/t6/value_probe_multi_t6.py

# 测试
.venv/bin/python -m pytest -q -p no:cacheprovider   # 418 passed
```

产物：`runs/t6_towers__1__1790334485/`、`runs/t6_towers_trickdiff__1__1790334485/`、
`runs/t6_towers_scratch__1__1790334485/`（各含 `agent.pt`/`args.json`/`metrics.csv`/`tb/`）、
`runs/h2h_t6_*.json`（+ 逐局 `_seed{0,1,2}.jsonl`）、`runs/t6/health.json`（sha/args/健康）、
`runs/t6/out_*.txt`（机制输出）、`runs/t6/h2h_summary.md`、`runs/t6_report.md`（完整笔记）。

# 突破 Elo 平台的低预算实验报告（E / EB 系列）

> 资产状态（2026-09-25）：本报告引用的模型 checkpoint 已删除，旧路径不再可用；数值为牌型族规则变更前口径。

> **状态：训练与定级完成；核心结论是「没有突破」——本批没有任何配置显著超过同 fit 基线，
> 且评测审计已确认：跨 fit 40–75 Elo 的漂移是 `plan_games` 座位相位（候选下标奇偶）
> 的确定性效应，stage A 的 mix50 第一名是相位假象。**
> 可靠性结论以 [`elo-reliability-audit.md`](./elo-reliability-audit.md)（已完成）为准：
> 座位平衡后 `base680k` = 1443.3 ± 14.7，E 系列五个候选互差 ≤10.4 Elo（不可分）；
> 平台顶部实际约 **1440–1460**，不是 1490。
> 背景与平台证据见 [`ladder-report.md`](./ladder-report.md)；轨迹信号见
> [`trace-signal-report.md`](./trace-signal-report.md)；训练栈见 [ADR-0003](../adr/0003-training-stack.md)。
> 本报告的所有命令、日志、PID、复现脚本都在文末列出；未 commit 任何代码。

## 0. 摘要

- 在 `lvlbase`（vs Greedy，1M 步，平台 ~1440–1490）与 `lvlsp`（self-play 热启动）之后，
  本批用 **150k–250k 步的低预算**（平台在 ~60k 步出现，见 §1.1）跑了 **17 个实验**，
  覆盖 9 条策略轴：self-play 刷新间隔/采样、对手混合（mix）、真多快照池（pool）、
  熵系数、学习率退火、batch/并发、hidden 容量、热启动源、种子复制。
- 一手分析纠正了原报告对 entropy 的读法：metrics.csv 的 `entropy` 是**两个头之和**
  （`networks.py:132-134`），其中全开的**花色头**（`actions.py:107-119`）贡献 ~88%
  （1.22 / ln4=1.386），而真正做策略的**模板头已接近确定**（H≈0.24 nats，
  仅为其合法动作均匀上界 ~1.31 的 18%）。「策略仍很随机」不成立。
- 学习信号没有枯竭：lvlbase 全程 `approx_kl` 2–3e-3、`clipfrac` ~8%（后期 3.5%）、
  `explained_variance` 0.64–0.66；平台不是梯度消失，而是对外胜率不再改善。
- **平台高度的最终口径（审计 §5.2）**：`base680k` 旧测 1489.4 ± 34.3（100 局/锚点、
  另一次调度）→ 同 ckpt wave-1/stageA 均为 1447.4 ± 23.0 → **座位平衡（每副牌换座
  各打一次）后 1443.3 ± 14.7**。1489 很可能是那次调度相位向上偏移的产物，
  平台顶部实际约 **1440–1460**。
- **stage A 的 mix50 第一名是相位假象（审计 §5.2）**：座位平衡后 mix50 **1415.6 ± 14.5**，
  与 `sp_from_sp40k`（1415.6）、`ctrl680`（1417.0）、`ent0`（1410.7）、
  `sp_r5_samp`（1406.6）互差 ≤10.4 Elo（< 1 个平衡 SE）。`base680k` 领先最好的
  E 系列 26.3 Elo（约 1.3σ），也只是「可能略强」，不是显著突破；
  直接对 greedy 里 mix50 62.9% 低于 base 63.0%，与之一致。
- **stage B（`--cross 800`）的 ±4.8 无效**：审计确认 `fit_ratings` 只返回对角 Fisher，
  未对自由候选间的联合 Hessian 求逆；最小复现低估 **4.16×**（真实 SE ≈142 vs 报告 34）。
  stage B 只能看序（全距 1428–1448、名义第一 `pool4g` +7.9），不能用于显著性判断；
  且其绝对点估计仍带座位相位问题。
  直接对 greedy 1500 局（§4.5）里 `pool4g` 63.5% / `base680k` 63.0%，在 1.25pp 的
  SE 下同样不可分。
- 相对结论（在审计后仍成立）：1380–1490 带内所有策略在相位/牌局两种噪声下都无显著高低；
  1M 步不优于 680k；对 GreedyBot 的胜率在 ~62–65% 处饱和。
- **未试轴**（见 §7）：per-episode 对手抽样、PFSP/优先联赛、稠密奖励/奖励塑形、
  观测增广（记牌/剩余牌）、target-KL、gamma/lambda、per-head 熵系数、座位轮换训练、
  更大 hidden/更长预算、以及 >2 家训练。这些才是本报告没有排除的方向。

## 1. 瓶颈的一手证据与分析

### 1.1 曲线：~60k 步饱和，之后 90 万步在带内抖动

从 `runs/lvlbase__1__1790313091/metrics.csv`（976 次 update）与
`runs/lvlsp__1__1790314004/metrics.csv` 按步数分段平均：

| 训练段 | lvlbase ent | lvlbase kl | lvlbase clipfrac | lvlbase EV | lvlbase ep_return | lvlsp ent | lvlsp ep_return |
|---|---|---|---|---|---|---|---|
| 0–20k | 2.493 | 7.0e-4 | 0.002 | 0.105 | −0.426 | 1.358 | −0.030 |
| 20–40k | 2.324 | 2.8e-3 | 0.059 | 0.506 | −0.232 | 1.330 | −0.075 |
| 40–60k | 2.140 | 3.1e-3 | 0.070 | 0.585 | −0.076 | 1.364 | −0.009 |
| 60–100k | 1.929 | 2.9e-3 | 0.086 | 0.637 | +0.016 | 1.422 | −0.019 |
| 100–200k | 1.717 | 3.0e-3 | 0.082 | 0.635 | +0.091 | 1.294 | −0.013 |
| 200–500k | 1.487 | 2.6e-3 | 0.082 | 0.650 | +0.146 | 1.369 | −0.009 |
| 500k+ | 1.396 | 1.7e-3 | 0.035 | 0.660 | +0.187 | 1.346 | −0.008 |

对照 `ladder-report.md` 的 Elo 曲线：20k=1243.6、40k=1318.8、60k=1347.1，
之后 100k–960k 全在 1384–1489 带内。**注意 60k 之后 KL 仍有 2–3e-3、clipfrac ~8%，
即参数仍在更新、优势估计仍有信号**；所以「信号耗尽」不是字面意义上的梯度为零，
而是「继续更新不再提高对外（对 Greedy/锚点）强度」。lvlsp 的 `ep_return` 全程 ≈0，
说明自我镜像已在均衡态（与 `ladder-report.md` §2.3 一致）。

### 1.2 被误读的 entropy：花色头吞掉了几乎全部数值

`networks.py:119-134` 的 `get_action_and_value` 把两个头的 entropy 相加
（`torch.stack([...]).sum(0)`）；`actions.py:107-119` 的 `joint_mask_bits` 让模板头之后的
偏好头（花色）**永远全开**（ADR-0004）。因此 metrics.csv 里的 `entropy` 不能直接读成
「策略有多随机」。用 `Match` 对 GreedyBot 实测（80 局，每步记录两个头）：

| ckpt | 模板头 H | 合法模板均匀上界 | 占比 | 花色头 H | ln 4 |
|---|---|---|---|---|---|
| base s20k（1244） | 1.192 | 1.212 | 98% | 1.369 | 1.386 |
| base s680k（1447） | 0.239 | 1.308 | 18% | 1.220 | 1.386 |
| base final（1384） | 0.218 | 1.291 | 17% | 1.267 | 1.386 |
| lvlsp final（1374） | 0.178 | — | — | 1.175 | 1.386 |

即：**策略头（模板）在 20k 时接近均匀，到 680k 已收敛到平均 ~1.2 个有效动作**；
1.39 / 1.27 的总 entropy 里 1.2–1.27 来自花色头。花色头按 ADR-0004 是「弱占优」的
（应优先出弱花色），它保持 ~88% 均匀是一个真实但**小**的次优：把推理时的花色选择
强制成「最弱可用花色」，`base680k` 对 greedy 1500 局胜率 0.6300→0.6413
（+1.1pp，SE≈1.2pp，不显著）。因此「提高探索」不是靠总 entropy 数值判断的。

### 1.3 学习信号与测量噪声

- `approx_kl` 在 wave1 的 150k–200k continuation 里仍有 1–3e-3，
  `explained_variance` 0.65–0.68（§4.1 表）。这不是「训练没跑起来」。
- **跨 fit 漂移（关键）**：同一 ckpt 在候选列表里换一个位置，评分会变 40–75 Elo，
  而 fit 报的 SE 只有 ~22（详见 §4.4）。`base680k` 在两次 fit 里候选下标都是 0、
  座位相位完全一致，两次评分**逐位相同（1447.4）**——这指向 fit 之间的差异
  主要来自**候选座位相位**（`ladder.py:plan_games` 的 `seat = (index + candidate_index + anchor_index) % 2`），
  而不是评分数学。
- 座位探测：`base680k` 对 greedy 坐 0 号位 62.3%（1000 局）、坐 1 号位 63.8%；
  纯 GreedyBot 镜像里坐 0 号位只赢 45.0%。**全局平均座位效应很小**（审计 §3.5
  测得全局平均 δ 只有 ±1–3pp），但**每副牌的 δ 方差很大**，而每副牌在旧调度下
  只打一个座位（训练也只在 `learner=0` 下产生数据）。审计 §3.5 用逐牌配对
  bootstrap 确认：相位位移来自 δ 在偶数/奇数两个半边上的均值差，
  而不是全局座位优势；修复见审计 §6.1。

### 1.4 平台高度的复评

| 对象 | 旧测（100 局/锚点，另一次调度） | 本批 wave-1 fit（400 局） | 审计：座位平衡（800 局） |
|---|---|---|---|
| base s680k | 1489.4 ± 34.3 | 1447.4 ± 23.0 | **1443.3 ± 14.7** |
| sp s40k | 1476.2 ± 33.6 | 1422.5 ± 22.2 | —（未进审计 6 候选） |

两次旧测方向一致地偏高，审计已给出机制：`plan_games` 的座位相位（`ladder.py:127`）
由候选下标奇偶决定，而旧调度下每副牌只打一个座位；`base680k` 在 wave1→stageA
恰好未翻相位所以逐位不变，但 1489 那次的相位无法追溯（可能是上沿）。
**最终口径：`base680k` = 1443.3 ± 14.7，平台顶部约 1440–1460，不是 1490。**
本报告的其他对比一律以同一 fit 内 `base680k` 为基线，不跨 fit 比较绝对值。

### 1.5 训练语义（重要变更，2026-09-25）

本批 **wave1/2/3（`ebo_*`/`eb2_*`/`eb3_*`）全部使用 gymnasium 默认的 NEXT_STEP 自动重置**
（旧 `train.py` 未传 `autoreset_mode`）。审计后工作区已改为 `AutoresetMode.SAME_STEP`
（`src/seven523/train.py:429`，对齐 gym 0.21 参考语义，消除 ~4% 跨局拼接样本），
并新增 `_episode_info`（`train.py:219`）同时兼容 `infos["episode"]` 与
`infos["final_info"]["episode"]`，回归测试在 `tests/test_train.py:182`（修复后全套绿）。
**此修复不影响本批任何 run**（它们都是旧语义、且是在修复前启动/结束的）；
**之后新启动的训练将使用 SAME_STEP，与本批不再严格同分布**，跨语义比较时需注意。

## 2. 候选突破轴与源码坐标

| 轴 | 源码坐标 | 假设 | 本批实验 |
|---|---|---|---|
| self-play 刷新间隔 | `train.py:164`（flag）、`train.py:467-471`（刷新循环） | 25 次 update（25.6k 步）的冻结对手让镜像均衡固定；更快刷新引入非平稳压力 | `sp_r25_samp`、`sp_r5_samp`、`sp_fixed680`（refresh 0） |
| self-play 采样 | `train.py:164`、`networks.py:220/251` | argmax 对手是确定性镜像，采样版提供多样性、避免同策略循环 | `*_samp` 全部 |
| 对手混合 mix | `policies.py:59`（`MixturePolicy`）、`train.py:356-366` | 只打 Greedy 的信号饱和；混入冻结自己的同时保留锚点压力 | `mix50`、`mix_samp` |
| 真多快照池 pool | `train.py:64-73`（`parse_pool_member`）、`train.py:367-388` | 历史快照池（league）比单冻结对手更能提供难度梯度与风格多样性 | `pool4`、`pool4g`、`pool_self` |
| 熵系数 | `train.py:116`、`networks.py:132-134` | 花色头 ~88% 均匀是熵奖励维持的；模板头可能探索不足 | `ent0`（0.0）、`ent03`（0.03） |
| 学习率/退火 | `train.py:79`、`train.py:102/473` | 1M 的线性退火让后期冻结，平台是调度产物 | `noanneal` |
| batch / 并发数 | `train.py:99-100` | 1024 步 batch 的梯度方差限制了可达到的强度 | `bigbatch`（16×128=2048） |
| 容量 hidden | `train.py:123` | 128 宽不够表达长程手牌规划 | `h256_scratch`（256 宽，从零 250k） |
| 热启动源 | `train.py:333-345`（`load_checkpoint`） | 从最强的 base 或最强的 self-play 快照继续，能到达不同 basin | 全部 continuation；`sp_from_sp40k`、`sp40k_s2/s3` |
| 座位轮换训练 | `train.py` `learner = 0` | 只学 seat 0 会在 ladder 轮换座位时丢失一半强度 | **未试**（探测显示座位效应小，但逐策略不对称见 §4.4） |
| target-KL | `train.py:119`、`ppo.py:206` | 无 KL 上限时后期批间漂移 | **未试** |
| 奖励塑形 | `game.py:269-281`（仅终局分差） | 终端稀疏奖励的信用分配不足 | **未试** |
| 观测增广 | `env.py` `_SEGMENTS` | 没有记牌/剩余大牌等特征，策略看不到长程信息 | **未试** |
| gamma/lambda、minibatch、epochs | `train.py:103-111` | GAE 视野与更新强度 | **未试** |
| >2 家 / 规则变体 | `train.py:121` | 多家博弈引入不同的联盟与牌序结构 | **未试**（且会破坏 2 家锚点口径） |
| per-head 熵系数 | `networks.py:132-134` | 总熵奖励被花色头稀释，模板头得不到足够探索压力 | **未试**（本批只改了总系数） |
| per-episode 对手抽样 | `policies.py:59`（当前按决策抽） | 按局抽才是标准 league 语义 | **未试** |
| PFSP / 优先联赛 | — | 用胜率加权选对手，聚焦「刚好打不过」的对手 | **未试** |

## 3. 实验矩阵、资源与并发依据

### 3.1 资源选择（为什么本批大部分用 `--cuda False`）

单位：learner steps/s（SPS），8 env × 128 steps，OMP 线程数见括号。

- 单进程 CPU（`OMP_NUM_THREADS=2`）：10,240 步 8.3 s → **2166 SPS**（网络很小，
  瓶颈是纯 Python 环境推进；还略高于文档里 GPU 阶段 1 的 ~1.8k）。
- wave1 10 路并行 CPU：各 run 469–1291 SPS（自博弈因对手侧也要做逐决策神经网络前向而最慢）。
  聚合吞吐 ~9k SPS，10 个进程各自独立、无 CUDA context/显存竞争。
- 因此 wave1 选 `--cuda False`：**不是 GPU 不可用，而是这个任务在 8GB、多进程筛选下
  CPU 已经更快更省事**；显存也不是限制（实测 4 路 GPU 只占 680–839MB），
  真正的瓶颈是纯 Python 环境推进。tensorboard 也关掉以减少 10 路并发 I/O。
- wave2/3 按用户要求改用 GPU：4 路并发 `--cuda True --tensorboard True`
  （`OMP_NUM_THREADS=4`），GPU 利用率 95–96%、显存 680–839MB；
  `bigbatch` 1297 SPS，混合/池化 run 359–559 SPS（对手侧推理仍是瓶颈）。
- 评测：`tools/build_ladder.py --device cuda`（本批新增，`tools/build_ladder.py:110/173`；
  `policies.py:115` 的 `policy_from_spec` 按 `(path, device)` 缓存 agent）。
  小批量（batch=1）前向下 GPU 与 CPU 基本同速（100 局子集：CPU 2.39s vs GPU 2.61s），
  但**同调度下评分逐位一致**：wave-1 的 12 候选 200 局/锚点 fit 用 cpu 与 cuda 各跑一次，
  18 行 Elo 输出连小数位都相同（`runs/ebo_ladder_wave1{,_cpu}.txt`）。
  对 2400–7200 局的评测，GPU 的价值主要是把 CPU 让给并发的训练/评测进程。

### 3.2 矩阵

warm start 统一为 `runs/probe/base_step00696320.pt`（base680k，旧测 1489）或
`runs/probe_sp/sp_step00040960.pt`（sp40k，旧测 1476）；seed 除注明外为 1；全部 detached
（`setsid nohup uv run --group train 7g523-train ... < /dev/null &`），脚本在 `runs/` 下。

| # | exp-name | 假设 | 相对基线差异 | 步数 | 启动 PID | 日志 |
|---|---|---|---|---|---|---|
| 1 | `ebo_ctrl680` | 对照：默认 continuation | warm start base680k，vs greedy | 200k | 1513357 | `runs/ebo_ctrl680_train.log` |
| 2 | `ebo_noanneal` | 平台是退火产物 | `--anneal-lr False` | 200k | 1513358 | `runs/ebo_noanneal_train.log` |
| 3 | `ebo_ent0` | 熵奖励维持花色头随机 | `--ent-coef 0.0` | 150k | 1513359 | `runs/ebo_ent0_train.log` |
| 4 | `ebo_ent03` | 模板头探索不足 | `--ent-coef 0.03` | 150k | 1513360 | `runs/ebo_ent03_train.log` |
| 5 | `ebo_sp_r25_samp` | 采样镜像打破确定性均衡 | self、refresh 25、sample | 200k | 1513361 | `runs/ebo_sp_r25_samp_train.log` |
| 6 | `ebo_sp_r5_samp` | 更快刷新+采样 | self、refresh 5、sample | 200k | 1513362 | `runs/ebo_sp_r5_samp_train.log` |
| 7 | `ebo_sp_fixed680` | 对打固定强自己 | self、refresh 0 | 200k | 1513363 | `runs/ebo_sp_fixed680_train.log` |
| 8 | `ebo_mix50` | 锚点+自己的联赛 | mix，greedy p=0.5，refresh 10 | 200k | 1513364 | `runs/ebo_mix50_train.log` |
| 9 | `ebo_sp_from_sp40k` | 换 basin | warm start sp40k，self refresh 10 sample | 200k | 1513365 | `runs/ebo_sp_from_sp40k_train.log` |
| 10 | `ebo_h256_scratch` | 容量不足 | hidden 256、从零、vs greedy | 250k | 1513366 | `runs/ebo_h256_scratch_train.log` |
| 11 | `eb2_sp40k_s2` | seed 复制 #9 | 同 #9，seed 2 | 200k | 1562844 | `runs/eb2_sp40k_s2_train.log` |
| 12 | `eb2_sp40k_s3` | seed 复制 #9 | 同 #9，seed 3 | 200k | 1562845 | `runs/eb2_sp40k_s3_train.log` |
| 13 | `eb2_mix_samp` | 混合+采样+快刷新 | mix p=0.5、refresh 5、sample | 200k | 1562846 | `runs/eb2_mix_samp_train.log` |
| 14 | `eb2_bigbatch` | 梯度方差限制强度 | 16 env × 128 = 2048 batch | 200k | 1562847 | `runs/eb2_bigbatch_train.log` |
| 15 | `eb3_pool4` | 真多快照池（无锚点） | pool = B200k+B560k+SP20k+SP40k | 150k | 1836863 | `runs/eb3_pool4_train.log` |
| 16 | `eb3_pool4g` | 池 + 锚点压力 | #15 + 2@greedy | 150k | 1836864 | `runs/eb3_pool4g_train.log` |
| 17 | `eb3_pool_self` | 池中有活策略 | self + B200k + SP40k + 2@greedy | 150k | 1836865 | `runs/eb3_pool_self_train.log` |

启动脚本：`runs/ebo_launch_wave1.sh`（10 路，CPU）、`runs/ebo_launch_wave2.sh`
（4 路，GPU+TB）、`runs/ebo_launch_wave3.sh`（3 路，GPU+TB）。
池化的两个新 CLI 是本批唯一共享代码改动：`MixturePolicy`（`policies.py:59`）、
`--opponent pool` + `--pool-member [WEIGHT@]SPEC`（`train.py:64/367-388`），
新测试 `test_mixture_policy_*`、`test_parse_pool_member`、`test_train_pool_opponent_smoke`；
全套 **235 tests 通过**（含本批的 `MixturePolicy`/pool/backfill 相关新增测试）。

### 3.3 定级口径

- 主口径：`tools/build_ladder.py --games-per-anchor N --no-traces --seed 0`，
  锚点 random=1000 / greedy=1315，BT-MAP 联合拟合。wave-1 = 12 候选 × 200 局/锚点；
  stage A = 18 候选 × 200 局/锚点；stage B = 7 候选 × 400 局/锚点 + 800 局/候选对。
- **这一口径的已知缺陷（审计已确认）**：`plan_games` 所谓「成对牌局、座位轮换」实际是
  “跨候选共享牌”，**不是候选内换座 twin**（`ladder.py:92-140`；座位相位 `ladder.py:127`
  由候选下标奇偶决定）。后果：单 fit 的 fisher SE 只覆盖“同调度换牌”，不覆盖
  “换顺序/换 fit”（后者应读作 ≈√2×SE）；本报告因此**以 [`elo-reliability-audit.md`]
  (./elo-reliability-audit.md) 的座位平衡（每副牌两个座位各打一次）结果作为最终口径**，
  §4.2/§4.3/§4.6 的 fit 数值只作筛选/过程记录。审计给的可靠点估计：
  `base680k` 1443.3 ± 14.7，E 系列最好的 `ctrl680` 1417.0、`sp_from_sp40k`/`mix50` 1415.6、
  `ent0` 1410.7、`sp_r5_samp` 1406.6。
- 辅助：`7g523-eval --episodes 1500 --opponent greedy --seed 0`（固定 seat 0），
  以及 seat 0 / seat 1 各 1500 局的对照。
- SE：同 fit 内的 Fisher SE 对「固定调度换牌」是校准的（审计 §3.2：池化 sd 22.9 vs SE 22.1）。

## 4. 结果

### 4.1 训练健康度（每 run 最后 50k 步平均）

| run | steps | ent(0–60k) | ent(last) | kl(last) | ret(last) | EV(last) |
|---|---|---|---|---|---|---|
| ebo_ctrl680 | 199680 | 1.334 | 1.445 | 1.2e-3 | +0.214 | 0.667 |
| ebo_noanneal | 199680 | 1.423 | 1.441 | 3.0e-3 | +0.184 | 0.661 |
| ebo_ent0 | 149504 | 1.284 | **1.009** | 1.1e-3 | +0.174 | 0.666 |
| ebo_ent03 | 149504 | 1.451 | **1.588** | 1.3e-3 | +0.178 | 0.652 |
| ebo_sp_r25_samp | 199680 | 1.347 | 1.237 | 1.0e-3 | +0.024 | 0.650 |
| ebo_sp_r5_samp | 199680 | 1.353 | 1.361 | 1.1e-3 | +0.022 | 0.662 |
| ebo_sp_fixed680 | 199680 | 1.276 | 1.218 | 8.0e-4 | +0.000 | 0.654 |
| ebo_mix50 | 199680 | 1.383 | 1.327 | 1.2e-3 | +0.072 | 0.678 |
| ebo_sp_from_sp40k | 199680 | 1.398 | 1.436 | 1.1e-3 | +0.012 | 0.663 |
| ebo_h256_scratch | 249856 | 2.291 | 1.654 | 1.0e-3 | +0.097 | 0.678 |
| eb2_sp40k_s2 | 199680 | 1.363 | 1.445 | 1.1e-3 | +0.026 | 0.662 |
| eb2_sp40k_s3 | 199680 | 1.368 | 1.419 | 1.2e-3 | +0.007 | 0.662 |
| eb2_mix_samp | 199680 | 1.337 | 1.225 | 9.4e-4 | +0.097 | 0.655 |
| eb2_bigbatch | 198656 | 1.343 | 1.284 | 1.1e-3 | +0.186 | 0.664 |
| eb3_pool4 | 149504 | 1.317 | 1.349 | 1.3e-3 | +0.019 | 0.649 |
| eb3_pool4g | 149504 | 1.325 | 1.257 | 1.6e-3 | +0.049 | 0.658 |
| eb3_pool_self | 149504 | 1.404 | 1.460 | 1.6e-3 | +0.058 | 0.660 |

读法：`ent0` 把总熵压到 1.009（花色头被压掉），`ent03` 升到 1.588，
但两者 Elo 都在噪声内（§4.2），说明**总熵系数不是有效杠杆**。
vs-greedy 的 `ep_return` 从 0.17 到 0.21，各 continuation 基本没变；
self-play/pool 类的 `ret(last)` ≈ 0–0.1，说明它们在对自己的镜像/池子上只是打平。

### 4.2 wave-1 fit：12 候选 × 200 局/锚点（`runs/ebo_ladder_wave1.txt`）

| 候选 | Elo ± SE | games | 与 base680k 差 |
|---|---|---|---|
| base680k（基线） | 1447.4 ± 23.0 | 400 | — |
| sp40k（sp 基线） | 1422.5 ± 22.2 | 400 | −24.9 |
| ctrl680（对照 continuation） | 1403.1 ± 21.7 | 400 | −44.3 |
| noanneal | 1396.3 ± 21.6 | 400 | −51.1 |
| ent0 | 1380.5 ± 21.2 | 400 | −66.9 |
| ent03 | 1416.8 ± 22.1 | 400 | −30.6 |
| sp_r25_samp | 1418.3 ± 22.1 | 400 | −29.1 |
| sp_r5_samp | 1383.1 ± 21.3 | 400 | −64.3 |
| sp_fixed680 | 1418.3 ± 22.1 | 400 | −29.1 |
| mix50 | 1380.5 ± 21.2 | 400 | −66.9 |
| sp_from_sp40k | 1435.5 ± 22.6 | 400 | −11.9 |
| h256_scratch | 1376.7 ± 21.1 | 400 | −70.7 |

全部在 ±2 SE 内；没有任何候选超过基线。对照 `ctrl680` 本身比起点低 44 Elo（≈1.4σ），
说明「从最强 ckpt 继续 200k」不涨分，甚至倾向回归。

### 4.3 stage A fit：18 候选 × 200 局/锚点（`runs/ebo_ladder_stageA.txt`）

| 候选 | Elo ± SE | 候选 | Elo ± SE |
|---|---|---|---|
| **mix50** | **1455.1 ± 23.2** | sp_r25_samp | 1407.1 ± 21.8 |
| base680k | 1447.4 ± 23.0 | mix_samp | 1400.3 ± 21.7 |
| ent0 | 1444.4 ± 22.9 | sp_from_sp40k | 1397.7 ± 21.6 |
| ctrl680 | 1432.6 ± 22.5 | sp_fixed680 | 1396.3 ± 21.6 |
| sp_r5_samp | 1432.6 ± 22.5 | sp40k_s2 | 1396.3 ± 21.6 |
| bigbatch | 1431.1 ± 22.5 | pool4 | 1388.4 ± 21.4 |
| noanneal | 1422.5 ± 22.2 | pool_self | 1388.4 ± 21.4 |
| sp40k_s3 | 1421.1 ± 22.2 | h256_scratch | 1367.7 ± 20.9 |
| pool4g | 1419.7 ± 22.1 | ent03 | 1412.7 ± 22.0 |

**排名前六里，mix50/flat 全部落在 1430–1455，彼此 < 1 SE；任何「谁更强」的结论都不可靠。**
**审计（§5.2）已确认：stage A 的 mix50 1455.1 第一是相位假象**——座位平衡后
mix50 1415.6 ± 14.5，与 sp_from_sp40k 1415.6、ctrl680 1417.0、ent0 1410.7、
sp_r5_samp 1406.6 互差 ≤10.4 Elo（不可分），而 base680k 1443.3 领先它们 26.3 Elo
（约 1.3σ）。下表只保留为「筛选线索」，不得当结论。
`pool4`/`pool_self`（纯池/含活策略池）在底部附近；`pool4g`（池+锚点）居中；
`h256` 从零 250k 仍最低（1367.7），对照 base@250k 旧测 1445——容量不是答案。

### 4.4 跨 fit 漂移：候选位置（座位相位 + 脚本对手 RNG）

同一 ckpt、同一 400 局口径、仅候选列表位置变化时的评分差：

| 候选 | wave-1 idx→stage A idx | 相位 | wave-1 Elo | stage A Elo | Δ |
|---|---|---|---|---|---|
| base680k | 0 → 0 | 不变 | 1447.4 | 1447.4 | **0.0** |
| ctrl680 | 2 → 1 | 翻 | 1403.1 | 1432.6 | +29.5 |
| noanneal | 3 → 2 | 翻 | 1396.3 | 1422.5 | +26.2 |
| ent0 | 4 → 3 | 翻 | 1380.5 | 1444.4 | +63.9 |
| ent03 | 5 → 4 | 翻 | 1416.8 | 1412.7 | −4.1 |
| sp_r25_samp | 6 → 5 | 翻 | 1418.3 | 1407.1 | −11.2 |
| sp_r5_samp | 7 → 6 | 翻 | 1383.1 | 1432.6 | +49.5 |
| sp_fixed680 | 8 → 7 | 翻 | 1418.3 | 1396.3 | −22.0 |
| mix50 | 9 → 8 | 翻 | 1380.5 | 1455.1 | **+74.6** |
| sp_from_sp40k | 10 → 9 | 翻 | 1435.5 | 1397.7 | −37.8 |
| h256_scratch | 11 → 10 | 翻 | 1376.7 | 1367.7 | −9.0 |

观察与机制（审计 `elo-reliability-audit.md` §2–§4 已逐条确认）：

1. **唯一没有翻相位的 `base680k` 两次评分逐位相同**（1447.4），说明两次 fit 的锚点、
   调度、评分数学都一致；位移只跟着「候选在列表里的位置」走。
2. 在锚点固定、无 prior、无 cross 的设定下，联合 fit 对每个候选是可分的
   （每人的似然只依赖它与两个固定锚点的战绩），所以 **+74.6 不可能是拟合耦合**，
   只能来自「实际打出的战绩变了」。
3. 位置改变会同时改变两件与对局结果相关的事：
   （a）同一副牌下受评策略坐 0/1 号位（`ladder.py:92-140`，根因在 **`ladder.py:127`**
   的 `seat = (index + candidate_index + anchor_index) % 2`）；
   （b）`play_games` 给每个座位生成策略种子 `game.seed + 101*(seat+1)`
   （`ladder.py:167-170`），座位一翻，**random 锚点拿到的是另一条 RNG 流、
   会走出不同的动作序列**（GreedyBot 无 RNG，不受影响）。
4. **审计 §3.5 的机制解释**：对每副牌定义 δ = 「坐 0 号位胜率 − 坐 1 号位胜率」。
   全局平均 δ 很小（±1–3pp，与 §4.5 的直接观测一致），但**每副牌的 δ 方差很大**；
   一次换序把每副牌的座位翻转，评分位移 = δ 在偶数轮/奇数轮两个任意半边上的
   均值之差（mix50 vs greedy：−0.120 / +0.115，半边差 −0.1175）。所以问题不是
   「座位优势」，而是「每副牌只打一个座位」+ 相位由下标奇偶决定的**未随机化处理**。
   量化（审计 §3.3–§3.5）：同调度换牌 sd≈22（与 Fisher SE 一致，无低估）；
   换顺序 sd≈**30.7 ≈ √2×SE**，设计效应 ≈2.04（400 局的有效独立样本 ≈196）；
   换序位移均值 +41.3、最大 +74.6；**座位平衡后 sd 降到 ≈14.6**。
   因此 §4.2/§4.3 的排名（尤其 mix50 第一）已被确认是相位假象，不作为证据；
   修复建议见审计 §6.1（候选内换座配对：每副牌两个座位各打一次）。

本报告另做了**顺序交换实验**（同一对 ckpt、200 局/锚点、seed 0，仅交换候选顺序）：

| fit 候选顺序 | base680k | mix50 |
|---|---|---|
| `[base680k, mix50]` | 1447.4 ± 23.0 | **1380.5 ± 21.2** |
| `[mix50, base680k]` | 1439.9 ± 22.7 | **1455.1 ± 23.2** |

顺序 `[base680k, mix50]` 的 base680k=1447.4、mix50=1380.5 **逐位等于 wave-1**；
顺序 `[mix50, base680k]` 的 mix50=1455.1 **逐位等于 stage A**（同一 fit 里 base680k
也随换位变为 1439.9）。同一 ckpt 在两次 wave-1/stage A 独立 fit 中相差 74.6，在
这个 800 局的顺序交换实验里被精确重建。这证明跨 fit 漂移
不是抽样噪声，而是**完全确定性的、由候选在列表里的位置决定的**效果。
用 `plan_games` + `play_games` 直接解开每次对局的原始战绩（每格 100 局：
座位轮换下每个 (候选, 锚点, 座位) 组合恰好 100 局；括号内为 win+draw/2 胜率）：

| 顺序 | mix50 vs greedy | mix50 vs random | base680k vs greedy | base680k vs random |
|---|---|---|---|---|
| `[base680k, mix50]` | 60.5%（seat0 61.0 / seat1 60.0） | 88.5%（86.5 / 90.5） | 69.0%（67.5 / 70.5） | 92.0%（90.5 / 93.5） |
| `[mix50, base680k]` | **72.3%**（71.5 / 73.0） | 90.0%（85.5 / 94.5） | 66.5%（67.5 / 65.5） | 93.3%（93.0 / 93.5） |

同一副 200 张牌（`plan_games` 的 deal\_seed 与候选顺序无关，只有座位相位翻），
mix50 对 greedy 的总胜率就从 60.5% 变到 72.3%——**仅靠换位**。而 1500 局的直接评测
（§4.5）给 mix50 的胜率是 62.9% / 61.9%（seat0 / seat1）；审计的配对 bootstrap
进一步把这一现象归因到「每副牌 δ 在偶数/奇数两个半边上的均值差」（审计 §3.5），
并用座位平衡把 sd 从 ≈21 降到 ≈14.6。相邻的 random 锚点种子也随座位变
（`ladder.py:167-170`；表里 mix50 vs random 从 88.5% 变到 90.0%），是第二通道，
但不是主因。**修复方向见审计 §6.1；本批的所有跨 fit 绝对值不再使用。**

### 4.5 直接对 greedy（1500 局，固定 seed 0）与座位不对称

| ckpt | 胜率（seat 0） | 平局率 | 分差 | 胜率（seat 1） |
|---|---|---|---|---|
| base680k（基线） | 0.6300 | 0.0733 | +20.82 | —（600 局探测 0.638）|
| ctrl680（对照） | **0.6480** | 0.0693 | +21.11 | 0.6387 |
| bigbatch | 0.6373 | 0.0753 | +20.79 | — |
| pool4g | 0.6353 | 0.0840 | +20.25 | — |
| ent0 | 0.6313 | 0.0807 | +19.66 | 0.6327 |
| mix50（stage A 第一） | 0.6287 | 0.0713 | +19.49 | 0.6193 |
| sp40k_s2 | 0.6187 | 0.0680 | +18.66 | — |
| sp_r5_samp | 0.6100 | 0.0740 | +18.20 | — |

1500 局的胜率 SE≈1.25pp（≈16 Elo）。
**直接测里 mix50 排在对比组的下游（62.9%，低于 base 的 63.0%）**，
与 stage A 把它排第一（1455.1）矛盾；ctrl680 / bigbatch / pool4g 反而在 63.5–64.8%。
**全局平均的座位不对称很小**（ctrl680 0.648→0.639、mix50 0.629→0.619、ent0 对称，
均 <1pp），这与 §4.4 的审计机制不矛盾：相位位移来自**每副牌 δ 在偶数/奇数两个
半边上的均值差**（审计 §3.5），而不是全局平均座位优势；单靠这 1500 局的
平均座位差确实解释不了 40–75 Elo。

### 4.6 stage B fit（7 候选 × 400 局/锚点 + 800 局/候选对；**报告的 SE 已作废**）

`runs/ebo_ladder_stageB.txt` 的点估计如下（只能当序看；`±` 数值不可用，
原因见下方第一条）：

| 候选 | Elo（SE 作废） | games | 相对 base680k |
|---|---|---|---|
| **pool4g** | **1448.1 ± 4.8** | 5600 | +7.9 |
| mix50 | 1441.9 ± 4.8 | 5600 | +1.7 |
| base680k（基线） | 1440.2 ± 4.8 | 5600 | — |
| sp_r5_samp | 1439.6 ± 4.8 | 5600 | −0.6 |
| bigbatch | 1439.0 ± 4.8 | 5600 | −1.2 |
| ent0 | 1429.5 ± 4.8 | 5600 | −10.7 |
| ctrl680（对照） | 1428.4 ± 4.8 | 5600 | −11.8 |

读法：

- **±4.8 的 SE 无效**（审计 §2.3/§5.3）：`fit_ratings` 只返回对角 Fisher（`elo.py:161-186,267-273`），
  对 `--cross>0` 引入的自由-自由对局未做联合 Hessian 求逆；最小复现低估 **4.16×**
  （shipped 34.23 vs 联合 Hessian 142.47）。此表只能当「序」的筛选线索，
  不能用于显著性判断，且它的绝对点估计仍带座位相位问题。
- 加交叉对局后，7 个候选挤在 **1428–1448（全距 20 Elo）**；这和审计§5.2 的
  座位平衡结果（`base680k` 1443.3、其余 1406.6–1417.0）不矛盾：
  两边都只能说「没有可区分的赢家」。
- `pool4g`（真多快照池 + greedy）名义第一（+7.9 vs base），且直接对 greedy 也略高
  （63.53% vs 63.00%，§4.5）；但它未进审计的 6 候选座位平衡测，不能补上
  跨 fit 的置信。两个独立口径的**符号一致、量级极小**，不构成突破。
- `mix50`（stage A 第一）在这里只比 base 高 1.7，与其直接对 greedy 低于 base 的事实
  一致；审计的座位平衡（mix50 1415.6 vs base680k 1443.3）进一步确认
  **stage A 的 1455 是相位假象**。

### 4.7 TensorBoard

wave-1 的 10 个 run 启动时用了 `--tensorboard False`（10 路 CPU 并发筛选，省 I/O）；
CSV 里有全部标量，已用新工具 `tools/backfill_tensorboard.py`
（读 `metrics.csv` 重放 `TensorboardLogger` 的标签，跳过已有 event 的 run）
回填到 `runs/<run>/tb/`。wave-2/3 训练时已直接写 TB。
因此 `tensorboard --logdir runs` 可以同时看到本批 17 条曲线（例：
`uv run --group train tensorboard --logdir runs --port 6006`）。
回填事件已用 `EventAccumulator` 抽查（12 个 scalar tag，entropy 末值 1.4326 @ step 199680）。

## 5. 结论（诚实版）

1. **没有突破。** 17 个低预算实验里，没有任何配置显著超过基线。最终可靠口径
   （审计 §5.2，座位平衡、bootstrap sd≈14.6）：**`base680k` 1443.3 ± 14.7**，
   E 系列最好的 `ctrl680` 1417.0、`sp_from_sp40k`/`mix50` 1415.6、`ent0` 1410.7、
   `sp_r5_samp` 1406.6——**五个 E 系列候选互差 ≤10.4 Elo（不可分）**，
   base680k 领先它们 26.3 Elo（≈1.3σ），只是「可能略强」。
   stage A 的 mix50 1455「第一」已被证实是座位相位假象，不得作为结论。
   「1490 平台」的最终顶部是 **1443 ± 15，约 1440–1460**。
2. **证据强度**：评估链路完全确定（同命令两跑、wave1/stageA 全量复跑 0 处不一致）；
   6-seed 重复的池化 sd 22.9 与平均 SE 22.1 一致（对「固定调度换牌」SE 是校准的）；
   跨 fit 的 40–75 Elo 位移已被一次换序实验 100% 逐位重建（10/10 翻相候选都漂，
   未翻相的 base680k 位移 0.0）。所以「无显著赢家」是高置信的**负面**结论，
   而不是「样本不够、再等等」。
3. **评测本身曾是最大不确定源**：`plan_games` 的座位相位（`ladder.py:127`）
   由候选下标奇偶决定，每副牌只打一个座位，导致单 fit SE≈22 不含相位项；
   跨顺序/跨 fit 应读作 **≈31（√2×SE）**。审计已给出修复方向（§6.1 候选内换座配对，
   sd 可降到 14.6），本批所有跨 fit 绝对值（包括旧 1489、stage A 排名、stage B 的 ±4.8）
   都不再使用；相对结论（带内无高低、1M 不超 680k）不受影响。
4. **可能解释**：在 2 家、奖励=分差、MLP + PPO 的设定下，对确定性 GreedyBot 的胜率
   在 ~62–65% 处饱和；平台内所有策略都是「同一族」的近似等价策略。
   self-play/池化把它们推向镜像均衡（`ep_return`≈0），并没有让它们更会打 Greedy。
   训练侧的语义缺陷（NEXT_STEP ~3.9% 拼接样本，§1.5）在本批全部 run 中存在，
   可能与「顶不到更高」无关，但新 run 已修正，值得重跑一个基线对照。
5. **下一步建议**（按性价比排序）：
   a. 落地审计 §6.1 的修复：候选内换座配对（每副牌两个座位各打一次）
      或直接报告「对 greedy 胜率 ±CI」；同时修 `--cross>0` 的联合 Hessian/cluster SE
      （审计 §6.3，是真实 bug），并保存逐局战绩（审计 §6.2）；
   b. 用 SAME_STEP 语义重跑 `ctrl680` 与 `bigbatch` 两个对照（200k），确认修复的净影响；
   c. 换**目标信号**而不是换对手：稠密奖励（逐墩分差塑形 `game.py:269-281`）
      或观测增广（记牌/剩余大牌），直接把「如何打赢固定 bot」的信息喂给策略；
   d. 只在 (a) 的评测口径下再谈池化/PFSP；本批的池化实现（`train.py:64/367-388`）
      已经可用。`pool4g` 在 stage B 名义第一、直接对 greedy +0.5pp，
      但未进审计的平衡评估，需要先用新口径重测再追预算。

## 6. 复现

```bash
# 依赖
uv sync --group train

# 训练（低预算；脚本含全部参数）
bash runs/ebo_launch_wave1.sh    # 10 路 CPU 筛选
bash runs/ebo_launch_wave2.sh    # 4 路 GPU + TensorBoard
bash runs/ebo_launch_wave3.sh    # 3 路 GPU + TensorBoard（真多快照池）

# 定级（GPU；同调度下与 CPU 逐位一致）
bash runs/ebo_ladder_wave1.sh          # 12 候选 × 200 局/锚点
bash runs/ebo_ladder_stageA.sh         # 18 候选 × 200 局/锚点
bash runs/ebo_ladder_stageB.sh         # 7 候选 × 400 局/锚点 + 800 局/对

# 直接对 greedy / 座位对照（1500 局）
uv run --group train 7g523-eval --checkpoint runs/probe/base_step00696320.pt \
    --episodes 1500 --opponent greedy --seed 0 --device cuda --json

# 可靠性审计（确定性复跑 / 6-seed / 换序 / 逐牌配对 bootstrap / 座位平衡）
# 命令与产物见 docs/experiments/elo-reliability-audit.md §7；驱动脚本 runs/elo_rel_*.{sh,py}

# TensorBoard（含 wave-1 回填）
uv run --group train python tools/backfill_tensorboard.py runs/ebo_*
uv run --group train tensorboard --logdir runs --port 6006
```

## 7. 产物清单

- 训练：`runs/ebo_*`（10）、`runs/eb2_*`（4）、`runs/eb3_*`（3），每个含 `args.json` /
  `metrics.csv` / `agent.pt` / `tb/`（wave-1 的 `tb/` 为回填）。
- 评分：`runs/ebo_ladder_wave1.txt`、`runs/ebo_ladder_wave1_cpu.txt`（GPU/CPU 对照）、
  `runs/ebo_ladder_stageA.txt`、`runs/ebo_ladder_stageB.txt`（stage B，**其 ±4.8 已因
  cross 联合 Hessian 缺失而作废**，见 §4.6 与审计 §2.3）。
- 可靠性审计（本报告的最终口径依据）：[`elo-reliability-audit.md`](./elo-reliability-audit.md)、
  `runs/elo_rel_*`（确定性复跑、6-seed、换序、逐局配对 bootstrap、座位平衡）。
- 直接评测：`runs/eval_greedy/*.json`。
- 辅助分析（本报告数字来源）：per-head entropy、mask 上界、座位探测、最弱花色覆盖
  由临时脚本在 `/tmp/head_entropy.py`、`/tmp/entropy_ceiling.py`、`/tmp/seat_probe.py`、
  `/tmp/suit_probe2.py` 产生；建议后续把其中可复用的部分并入
  `tools/`（本次未做，避免扩大改动面）。
- 循环盘查（座位相位漂移，§4.4）：`runs/order_swap_base_first.txt`、
  `runs/order_swap_mix_first.txt`（两次 fit），原始战绩分解脚本 `/tmp/order_swap_raw.py`。
- 代码改动（未 commit）：`MixturePolicy`（`src/seven523/policies.py:59`）、
  `--opponent pool/--pool-member`（`src/seven523/train.py:64/367-388`）、
  `build_ladder --device`（`tools/build_ladder.py:110/173`）、
  `policy_from_spec(device=)`（`src/seven523/policies.py:115`）、
  `tools/backfill_tensorboard.py`、`docs/training.md` 的对手池小节、3 个新测试，
  以及用户方的 P0 `AutoresetMode.SAME_STEP`（`train.py:429`）+ `_episode_info`（`train.py:219`）。

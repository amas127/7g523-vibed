# 训练与评估使用手册

> **规则口径（2026-09-29 更新）**：非炸弹比较为牌型族（[ADR-0007](adr/0007-family-comparison.md)、`RULES.md` §3），当前为 revision-3「出空即撬底」（[ADR-0014](adr/0014-on-empty-digs.md)）。本文引用的历史 Elo/胜率/训练结论均为旧 `tier` 口径，不可与新规则结果混比；重标定见 [plans.md](plans.md) T17 与 [experiments/README.md](experiments/README.md) 顶部警告。

> **模型资产状态（2026-09-29 更新）**：旧规则（revision-2 及更早，含 `tier` 口径）的 checkpoint 已从 `runs/` 删除（含本文示例曾引用的 `runs/probe/*`、`runs/lvl*` 等），旧路径不再可用。T17（出空即撬底，rules revision 3）已在当前规则下重训并重测 `lvl1`–`lvl4`（`runs/t17*`），`tools/play_ladder.py` 现登记 `random` + 梯级 + 顶部平台簇，2026-09-29 起含搜索 rung `search_leafq`；现行资产与强度见 [`human-play.md`](human-play.md) 与 `tools/play_ladder.py list`，结论入口见 [experiments/README.md](experiments/README.md)。下文 `--load-checkpoint` / `--checkpoint` / `--pool-member` 的 `runs/<...>/agent.pt` 仍为占位符：可指向 T17 产物或按「阶段 1」自训的新模型；重标定记录见 [plans.md](plans.md) T17（T15 转 revision-2 历史）。

依赖一次性安装（gymnasium + cu132 版 torch 已在 `train` 组里）：

```bash
uv sync --group train
```

## 阶段 1：学 seat 0，打 RandomBot

```bash
uv run --group train 7g523-train --exp-name stage1 --total-timesteps 300000
```

- 默认 `--opponent random`（唯一的脚本对手；GreedyBot 已随 ADR-0012 退役）、
  8 env × 128 steps 一个 batch；本机 300k 步约 3 分钟。
- 动作空间是 `MultiDiscrete([134, 4])`：134 个牌型模板 + 4 个顶牌花色（ADR-0004）。
  同一 v5 布局、动作头更小的 checkpoint（如旧的单头 134）可直接 `--load-checkpoint`：
  会自动 warm start（主干/价值头全拷，模板头权重平移，花色头新初始化）；
  v5 之前的 checkpoint 不能加载（`load_agent` 直接拒绝，见 [ADR-0009](./adr/0009-single-observation-and-comparison.md)）。
- 产物在 `runs/<exp-name>__<seed>__<时间戳>/`：

| 文件 | 内容 |
|---|---|
| `metrics.csv` | 每次 update 一行（含可选 eval 列） |
| `agent.pt` | 最终 checkpoint（评估用这个） |
| `checkpoint.pt` | 周期 checkpoint（`--checkpoint-interval`，默认每 100 次 update） |
| `snapshots/checkpoint_step*.pt` | 快照（`--snapshot-interval` 按 update 或 `--snapshot-steps` 按环境步，两者独立；默认都关） |
| `args.json` | 本次超参，便于复现 |

加周期评估（每 20 次 update 评 100 局，写入 CSV 的 `eval_*` 列）：

```bash
uv run --group train 7g523-train --exp-name stage1 --total-timesteps 300000 \
    --eval-interval 20 --eval-episodes 100
```

## 阶段 2：自我博弈热启动

```bash
uv run --group train 7g523-train --exp-name stage2 \
    --opponent self --load-checkpoint runs/<stage1-run>/agent.pt \
    --total-timesteps 300000 --eval-interval 20 --eval-opponent random
```

对手是当前策略的冻结快照，每 `--self-play-refresh`（默认 50）次 update 刷新；
`--self-play-sample` 让 `self`/`mix` 的冻结对手按 mask 采样而不是 argmax（默认关）。
`--load-checkpoint` 只是热启动，不影响 `--opponent` 的选择。

对手池（league）：`--opponent mix` 是「冻结自己 + RandomBot」的按决策混合
（比例由 `--mix-random-prob` 控制）；`--opponent pool` 是真正的多成员池：

```bash
uv run --group train 7g523-train --exp-name league --opponent pool \
    --pool-member 1@ckpt:runs/<run-a>/agent.pt \
    --pool-member 1@ckpt:runs/<run-b>/agent.pt \
    --pool-member 2@random \
    --load-checkpoint runs/<new-run>/agent.pt --total-timesteps 200000
```

示例里的两个 ckpt 成员是占位符（旧 probe 快照已删除），可先只留 `2@random`
跑通；替换成自己训练的新 ckpt 后再补多成员池。

每个 `--pool-member` 形如 `[WEIGHT@]SPEC`，`SPEC` 支持 `random` /
`ckpt:<agent.pt>` / `self`（`self` 走 `--self-play-refresh` 原地刷新）。
`MixturePolicy` 在每次决策按权重抽一个成员，各 env 用独立 RNG。

**池对手动作默认采样**（2026-10-07 operator 决策，[ADR-0017](adr/0017-pool-opponents-sample.md)）：
`--opponent pool` 下每个成员（`ckpt:` 与 `self`）的动作按各自 masked policy **采样**，
而不是 argmax；`--pool-sample false` 恢复历史 greedy pool。该开关只影响训练 league 的
pool 模式：`--opponent self`/`mix` 仍由 `--self-play-sample` 控制（默认 argmax），
评估/定级链（`7g523-eval`/ladder/h2h/duel/web）仍是确定性 argmax（`policy_from_spec`
的 `sample` 默认 False）。

### 当前续训配方（2026-10-08 决策，[ADR-0018](adr/0018-augmented-league-pool.md)）

现行池在 6 个 t17 成员之外加入 5 个前沿强成员（`o2c_base`、`o2c_deep`、`w2m_ctl`、
`w2m_plain`、`w2m_low`），并混入 **10% RandomBot**：11 个 ckpt 成员各 `9@`、random `11@`
（`11/110 = 10%`），`--pool-episode True` 下按局抽取、动作按 [ADR-0017](adr/0017-pool-opponents-sample.md)
默认采样。完整命令见 ADR-0018 §决定；注意 `--mix-random-prob` 在 `--opponent pool` 下
**不生效**，不得用它实现该比例。

上一版成员集合（6 个 t17 `3@` + random `2@`）见 [ADR-0015](adr/0015-continuation-pool-random-mix.md)；
历史 run（A1–A6/T23/w2m/head-depth）保持各自当时的池，不回填，跨池比较只作 context。

### 课程学习（多段对手课程；2026-10-08 讨论记录）

仓库没有进程内的时间表调度；最省力的课程是**多段 run 串联**（`--load-checkpoint` + 每段换池）：

1. A：`--opponent random`，~200k（从零或热启动）；
2. B：弱池（`lvl1`/`lvl2` + 10% random，`--pool-episode True`），~300k，从 A 热启动；
3. C：强池（ADR-0018 池，或 `lvl4`/`o2c_base`/`head32ln5m` + 10% random），500k–1M+，从 B 热启动；
4. 可选收尾：`--opponent self`（`--self-play-refresh` 控制刷新频率），降低固定强池的熵塌缩风险。

判定与证据边界：对手分布本身（T5/PFSP/强成员）在**最终强度**上是 null；课程的预期收益是
速度/稳定性，因此主端点应是「同总步数静态池对照」的最终 h2h，次端点是固定 checkpoint 对
梯级的胜率曲线。`--eval-opponent` 目前只有 `random`（vs-random 早已饱和在 0.86–0.90）——
要观测课程中段进度，需扩 eval panel（固定 `ckpt:` 列表）或 `--snapshot-interval` 存快照
事后 h2h。强池固定太久有熵塌缩/特化风险（w2m 血统 1.57→1.08），阶段切换处盯 entropy/approx_kl。

若要进程内课程：`PfspController.maybe_update()` 已经在每 K 局把权重推给所有
`EpisodeMixturePolicy.set_weights()`（`train.py`），照这个接缝加一个按 update 进度出权重的
`CurriculumController` 即可（~50–100 行 + 测试）。「搜索当课程对手」（`search_leafq`，唯一
过 +10 的杠杆）仍是未测项：`rollout:<ckpt>` 工厂目前只在 `runs/o4lite-search/`（ladder 用），
trainer 的 `policy_from_spec` 不认识，接入需工程；动作蒸馏已试过是负（−52.8），不要走。

## 评估

```bash
uv run --group train 7g523-eval --checkpoint runs/<run>/agent.pt \
    --episodes 1000 --opponent random

uv run --group train 7g523-eval --checkpoint ... --opponent random --json
# ckpt 策略可用 --device cuda 指定推理设备（小 MLP 下与 CPU 同速，见评测报告）
```

| 指标 | 含义 |
|---|---|
| `learner_score` / `opponent_score` | 平均得分（满分 100） |
| `score_diff` | 平均分差；等于 `100 × mean_return`（RL-2） |
| `win_rate` / `draw_rate` / `loss_rate` | 按单局比分判定 |
| `mean_length` | 平均步数（学习者视角） |
| `illegal_rate` | 非法动作率，必须为 0（mask 是唯一合法性权威） |

## 天梯定级与候选对候选比较

定级用 `tools/build_ladder.py`：**候选内换座配对**（每副牌对每个候选打两局、座位互换），
`--games-per-anchor` 必须是**偶数**（400 = 200 副牌 × 双座位）；结果不受候选顺序影响。
跨 fit 比较不可靠（父模型跨 seed sd 可达 37），方案对比请用同 fit 或 ≥3 seed 合并：

```bash
uv run --group train python tools/build_ladder.py \
  --candidate A=ckpt:runs/<run-a>/agent.pt --candidate B=ckpt:runs/<run-b>/agent.pt \
  --games-per-anchor 400 --seed 0 --device cuda --no-traces --games-out runs/games.jsonl
```

候选对候选（同牌换座、按牌聚簇配对 bootstrap，方差比绝对锚点 Elo 小 2–3×；建议 ≥3 seed）：

```bash
uv run --group train python tools/head_to_head.py \
  --left A=ckpt:runs/<run-a>/agent.pt \
  --right B=ckpt:runs/<run-b>/agent.pt \
  --seeds 0,1,2 --pairs 400 --device cuda [--json] [--games-out runs/h2h.jsonl]

uv run --group train python tools/h2h_screen.py --help   # 多对 × 3 seed × 200 副批量筛选
```

分辨率：20 Elo ≈ 400–500 副牌，10 Elo ≈ 1500–2000。`--games-out` 落每局 JSONL
（`seed/seats/scores/subject/opponent/subject_seat/kind`，多 seed 自动分文件），支持事后
配对/bootstrap。结论与误差口径见 [experiments/README.md](./experiments/README.md)。

`tools/build_ladder.py`、`tools/head_to_head.py`、`tools/h2h_screen.py` 都支持
`--workers N`（默认 1 = 原串行循环）：`N>1` 时按 schedule 分片、spawn 进程并行，结果与串行
**逐位一致**（有回归测试）；`--device cuda` 下每个 worker 各开一个 CUDA context
（显存不够时降 `--workers` 或改 `--device cpu`）。

## 常用开关

- `--cuda False`：强制 CPU；`--opponent random`：对手换随机 bot
- `--activation {relu,tanh,gelu,silu}`：隐藏层激活（默认 relu；激活存入 checkpoint，旧 ckpt 缺省按 relu 加载）
- `--arch {shared,towers,ln,deep,deep_ln,deep_res,deep_lnres,head<a><c>,head<a><c>ln,lnres<a><c>,gnres<a><c>,bnres<a><c>,res<d><a><c>,bres<d><a><c>}`：网络结构（默认 `shared`，共享主干，逐位不变）。`towers` = 独立 actor/critic 两塔（T6 已判负关闭）；`ln`/`deep*` = O2 trunk 深度/归一化/残差臂（已关轴）；`head<a><c>` = 策略头 a 层 / 价值头 c 层（a,c∈{1,2,3}，多层头默认残差；头部深度网格 screen，见 [experiments/head-depth-500k.md](./experiments/head-depth-500k.md) 与预注册 [head-depth-plan.md](./head-depth-plan.md)）；`head<a><c>ln` = 残差块内加 **pre-norm** LayerNorm（head 设计 v2）；`lnres<a><c>`/`gnres<a><c>`/`bnres<a><c>` = 残差 3-block body（归一化 LN / GroupNorm（`--gn-groups`）/ **BatchNorm**）+ **非残差** plain 多层头（见 [experiments/gnres-500k.md](./experiments/gnres-500k.md)）；`res<d><a><c>` = 同款 plain 多层头 + **无 norm** 的 d-block 残差 body（`res4...` = 深度 4；见 [experiments/res4-2m.md](./experiments/res4-2m.md)）；`bres<d><a><c>` = 同 body/头，但每个残差块是 `hidden→mid→hidden` 瓶颈（`mid = --res-expansion × --hidden-size`，默认 3；`bres421` + hidden 32 = (32, 96, 32) 块，见 [experiments/bres-2m.md](./experiments/bres-2m.md)）
- **`bnres*` + PPO 警告**：朴素 BatchNorm（train 模式）会让 rollout 每步 batch=`num_envs` 的统计与 update minibatch 的统计不一致，old/new logprob 比值失真。实测 `bnres21` 三头 500k entropy 塌缩到 0.10、h2h 对 GN/LN −42 Elo（[experiments/bnres-500k.md](./experiments/bnres-500k.md)）→ 不建议直接使用；若要公平测试，先解决统计一致性（rollout 用 running stats + 显式更新 / SyncBN / 冻结统计等）。另：周期 eval 会把 learner 置 eval，`train.py` 已在 eval 后恢复 `agent.train()`（回归测试）
- `--actor-out-std`（默认 0.01 = 历史）：actor 输出层 init 的 std。注意它与头型有**强交互**（[gnres-500k.md](./experiments/gnres-500k.md)）：深 **plain** 头需要 0.1（否则隐藏层被输出层小增益压小），而 **残差** 头 0.1 反而伤 ~18 Elo（identity 通路已保证梯度）；动深头时把 init 与结构一起做因子，别只改一个。
- `--gn-groups`（默认 8）：`gnres<a><c>` body 的 GroupNorm 组数，必须整除 `--hidden-size`；其余 arch 忽略。与 `--arch gnres*` 无关的取值不参与语义，但写进 checkpoint（同 arch 热启动时组数不一致会 fail-loud）
- `--res-expansion`（默认 3）：`bres<d><a><c>` body 每个残差块的瓶颈宽度倍数（`hidden→mid→hidden`，`mid = factor × --hidden-size`）；其余 arch 忽略。可为小数，< 1 即压缩型瓶颈（`--hidden-size 64 --res-expansion 0.25` = (64, 16, 64)）；`hidden×factor` 必须是正整数，写进 checkpoint（同 arch 热启动时倍数不一致会 fail-loud）
- 观测布局固定为 **v5 = 观测 S1+B0+B1（`119+21n`、2 家 161）**，没有 `--obs-version` 开关。`save_agent` 把 `obs_version=5` 写进 payload，`load_agent` 对缺失或其他版本直接报错；v1–v4 已随兼容层退役，旧 ckpt 需重训（见 [ADR-0009](./adr/0009-single-observation-and-comparison.md)）
- `--reward-shaping {terminal,trick_diff,win,trick_diff_win,terminal_win,saturate,arcsin}`：奖励分解（**默认 `arcsin`**，2026-10-07 决策 [ADR-0016](./adr/0016-default-training-recipe-arcsin-lr-floor.md)；历史 `terminal` 逐位行为用 `--reward-shaping terminal` 复现）。`trick_diff` 每步 `Φ(s')−Φ(s)`（一局求和 = 终局回报，telescoping）；`win` 终局 `sign(own−max(others)) ∈ {−1,0,+1}`；`trick_diff_win` 为两者叠加；`terminal_win` 用 `--win-jump λ`（默认 1.0）在边界加跳变；`saturate` 用 `--reward-cap τ`（必需，0≤τ≤1）截断胜局分差；**`arcsin`（2026-10-07 operator 指定）**：终局 margin 先过 `(1−α)·margin + α·(2/π)·arcsin(clip(margin,−1,1))`（α 取 `--arcsin-mix` 默认 0.5；0/100 分处斜率最大、50 分处最平；α=0 即 `terminal_win`、α=1 为全强度 arcsin），再加 50 分胜/负边界跳变 `λ·seat_outcome`（平局保持在 0，跳跃只发生在 50 分两侧极限之间；`λ` 用同一个 `--win-jump`）。全强度 arcsin 在端点导数发散（0→1 分差 0.128，线性的 6.4×），默认半强度把它收到 3.7×。T1 结果见 [experiments/reward-shaping-500k.md](./experiments/reward-shaping-500k.md)（A1 未复现）；revision-3 的跳变/饱和两波见 [experiments/reward-alignment-terminal-win.md](./experiments/reward-alignment-terminal-win.md)、[experiments/reward-alignment-saturate.md](./experiments/reward-alignment-saturate.md)（均未过门，饱和线关闭）
- `--lr-floor`（默认 **1e-5**，2026-10-07 operator 决策）：`--anneal-lr` 下学习率的绝对下界（`annealed_lr`，`src/seven523/train.py`）；下界不会把学习率抬到 `--learning-rate` 之上（`--learning-rate 0` 仍严格为 0）。旧行为（退火到 ~0）用 `--lr-floor 0` 恢复
- `--vf-outcome`（默认关，2026-10-07 新变种）：**胜负分解价值头**。env 在终局 `info` 里发布 `reward_jump = reward_jump_scale × seat_outcome`（`src/seven523/env.py`），trainer 把优势估计拆成两条精确等价的 GAE 流：critic 只学 **去掉跳变** 的 return（`returns` 是无跳变目标），另加一个 `tanh` 回归头预测 `E[γ^(T-t)·seat_outcome]` 并用 `win_jump × 预测` 作为跳变流的基线（`A_total = A_base + A_jump`）。胜负头用折后胜负做目标（折扣是跳跃基线的一部分），符号即预测胜负；`--outcome-coef`（默认 1.0）是掩码 MSE 的权重，只有在本轮 rollout 内完局的样本有标签。要求奖励模式含离散跳变（`arcsin`/`terminal_win`/`trick_diff_win`/`win`），且不能与 `--vf-nll` 同用；检查点用 `vf_outcome` 字段往返，与旧检查点热启动时主干/critic 照拷、胜负头新初始化。初步 screen 见 [experiments/vf-outcome.md](./experiments/vf-outcome.md)
- `--opponent mix|pool`：对手混合 / 多成员联赛；`--pool-member [WEIGHT@]SPEC`（SPEC = `random`/`self`/`ckpt:<agent.pt>`，可重复，权重默认 1）
- `--pool-episode`（默认关）：`--opponent pool/mix` 时改为**逐局**冻结一个成员
  （`EpisodeMixturePolicy`，在 `env.reset` 时抽一次），代替旧的逐决策重抽（`MixturePolicy`）；
  关闭时逐位保持旧行为。
- `--pfsp`（默认关，需 `--opponent pool --pool-episode`）：每 `--pfsp-every K`（默认 100）局
  按学习者对每个成员的战绩重算采样权重——`wr_i` 先经 `Beta(prior/2, prior/2)`
  （`--pfsp-prior`，默认 10；0 关收缩）向 0.5 收缩，再 `w_i ∝ (1−wr_i)² + --pfsp-epsilon`
  （默认 0.02），最后与均匀按 `--pfsp-uniform-mix`（默认 0.5）混合；平局计半。逐局战绩与
  权重写入 `runs/<run>/pfsp_weights.csv`（TensorBoard `pfsp/weight/*`）。默认全关 = 旧行为；
  T5 结果（三效应 null）见 [experiments/opponent-distribution-500k.md](./experiments/opponent-distribution-500k.md)。
- PPO 超参：`--total-timesteps`、`--learning-rate`、`--ent-coef`、`--target-kl ...`
- `--hidden-size`：MLP 宽度（默认 128）
- `--checkpoint-interval 0`：只留最终 `agent.pt`；`--log-interval 10`：每 10 次 update 打印一次
- `--snapshot-steps N`：每 N 个环境步存一个快照到 `<run_dir>/snapshots/checkpoint_step*.pt`（batch 不整除 N 时落在首个 ≥ 边界的 update；与 `--snapshot-interval` 独立，可同时用）
- batch = `num-envs × num-steps`，必须能被 `--num-minibatches` 整除

## 查看指标

### TensorBoard（可视化）

训练默认会把 event 文件写进 `runs/<run>/tb/`（`--tensorboard False` 可关）。看曲线：

```bash
uv run --group train tensorboard --logdir runs --port 6006
# 浏览器打开 http://localhost:6006
```

主要标签：

| 标签 | 内容 |
|---|---|
| `charts/episodic_return`、`charts/episodic_length` | 每局终局回报与步数（逐局点） |
| `charts/SPS`、`charts/episodes` | 吞吐与累计局数 |
| `losses/value_loss`、`losses/policy_loss`、`losses/entropy` | PPO 损失 |
| `losses/approx_kl`、`losses/clipfrac`、`losses/explained_variance` | 健康度指标 |
| `losses/outcome_loss`、`losses/outcome_accuracy` | `--vf-outcome` 的胜负头掩码 MSE 与符号准确率（其余 run 为空） |
| `eval/score_diff`、`eval/win_rate`、`eval/learner_score`、`eval/mean_return` | 开启 `--eval-interval` 后的周期评估 |

超参也以文本面板写在 `hyperparameters` 下。

### CSV（纯文本）

```bash
tail -5 runs/<run>/metrics.csv
column -s, -t runs/<run>/metrics.csv | tail -5   # 人类可读
```

CSV 列：`global_step`、`episodic_return`、`episodic_length`、`episodes`、
`learning_rate`、`value_loss`、`policy_loss`、`entropy`、`old_approx_kl`、`approx_kl`、
`clipfrac`、`explained_variance`、`sps`、`eval_return`、`eval_score`、
`eval_score_diff`、`eval_win_rate`。

## 人机对战

不加参数就是和 RandomBot 玩（GreedyBot 已随 [ADR-0012](adr/0012-single-gauge-and-greedy-removal.md) 退役；对手清单与更多用法见 [human-play.md](human-play.md)）：

```bash
uv run 7g523-play                          # 默认：RandomBot，你坐 0 号位
uv run 7g523-play --opponent random        # 显式指定 RandomBot（与默认等价）
uv run --group train 7g523-play --checkpoint runs/<run>/agent.pt   # 打训练好的模型
uv run 7g523-play --seat 1 --seed 7 --rounds 3                     # 坐 1 号位、玩 3 局
```

每回合会列出所有合法出牌（模板 + 具体牌），输入编号即可；`h` 帮助，`q` 中途退出。
顶牌有多花色可选时会给出提示，例如 `（默认 ♠，可换 ♥）`：输 `10♥`（或 `10 2`）
即可改花色，直接输 `10` 用默认最强花色。炸弹/王炸与花色无关。

### 保存与回放轨迹

```bash
uv run 7g523-play --save-trace traces/my_game.json          # 打完自动保存
uv run --group train 7g523-play --checkpoint runs/<run>/agent.pt \
    --save-trace traces/vs_agent.json
uv run 7g523-play --replay traces/my_game.json              # 一次放完
uv run 7g523-play --replay traces/my_game.json --pause      # 回车一步步看
```

轨迹是 JSON（`version` / `created_at` / `rules` / `seed` / `human_seat` / `players` /
`initial` 发牌与先手 / `steps` 每步座位、动作、花色、文本、分数与墩结果 / `final_scores`）。
回放会从记录的发牌重建整局，并逐步校验“轮到谁、动作合法、分数一致”，任何不一致会
直接报错——所以它也是引擎确定性的集成测试。多局时文件名自动加 `_局号`。

## Python 里调用

```python
from seven523.train import parse_args, train

run_dir = train(parse_args([
    "--total-timesteps", "20000", "--cuda", "False", "--log-interval", "0",
]))
```

`Seven523Env` 是标准 gymnasium 环境，可脱离 PPO 单独使用：

```python
from seven523.env import Seven523Env

env = Seven523Env(seed=0)            # opponents=None 时自动配随机 bot
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(0)
```

`uv run 7g523` 可看一局随机演示对局，`uv run 7g523-play` 可人机对战（见上）。

# 训练与评估使用手册

> **规则口径（2026-09-25）**：非炸弹比较已改为牌型族（[ADR-0007](adr/0007-family-comparison.md)、`RULES.md` §3）。本文引用的历史 Elo/胜率/训练结论均为旧 `tier` 口径，不可与新规则结果混比；重标定见 [plans.md](plans.md) T15 与 [experiments/README.md](experiments/README.md) 顶部警告。

> **模型资产状态（2026-09-25）**：牌型族规则变更前的全部 checkpoint 已从 `runs/` 删除（含本文示例曾引用的 `runs/probe/*`、`runs/lvl*` 等），旧路径不再可用。下文 `--load-checkpoint` / `--checkpoint` / `--pool-member` 中的 `runs/<...>/agent.pt` 均为占位符：先按「阶段 1」训练出新模型；整体重标定计划见 [plans.md](plans.md) T15。

依赖一次性安装（gymnasium + cu132 版 torch 已在 `train` 组里）：

```bash
uv sync --group train
```

## 阶段 1：学 seat 0，打 GreedyBot

```bash
uv run --group train 7g523-train --exp-name stage1 --total-timesteps 300000
```

- 默认 `--opponent greedy`、8 env × 128 steps 一个 batch；本机 300k 步约 3 分钟。
- 动作空间是 `MultiDiscrete([134, 4])`：134 个牌型模板 + 4 个顶牌花色（ADR-0004）。
  旧的 134 头 checkpoint 可直接 `--load-checkpoint`：会自动 warm start（主干/价值头全拷，
  模板头权重平移，花色头新初始化）。
- 产物在 `runs/<exp-name>__<seed>__<时间戳>/`：

| 文件 | 内容 |
|---|---|
| `metrics.csv` | 每次 update 一行（含可选 eval 列） |
| `agent.pt` | 最终 checkpoint（评估用这个） |
| `checkpoint.pt` | 周期 checkpoint（`--checkpoint-interval`，默认每 100 次 update） |
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
    --total-timesteps 300000 --eval-interval 20 --eval-opponent greedy
```

对手是当前策略的冻结快照，每 `--self-play-refresh`（默认 50）次 update 刷新；
`--self-play-sample` 让对手按 mask 采样而不是 argmax。`--load-checkpoint`
只是热启动，不影响 `--opponent` 的选择。

对手池（league）：`--opponent mix` 是「冻结自己 + GreedyBot」的按决策混合
（比例由 `--mix-greedy-prob` 控制）；`--opponent pool` 是真正的多成员池：

```bash
uv run --group train 7g523-train --exp-name league --opponent pool \
    --pool-member 1@ckpt:runs/<run-a>/agent.pt \
    --pool-member 1@ckpt:runs/<run-b>/agent.pt \
    --pool-member 2@greedy \
    --load-checkpoint runs/<new-run>/agent.pt --total-timesteps 200000
```

示例里的两个 ckpt 成员是占位符（旧 probe 快照已删除），可先只留 `2@greedy`
跑通；替换成自己训练的新 ckpt 后再补多成员池。

每个 `--pool-member` 形如 `[WEIGHT@]SPEC`，`SPEC` 支持 `greedy` / `random` /
`ckpt:<agent.pt>` / `self`（`self` 走 `--self-play-refresh` 原地刷新）。
`MixturePolicy` 在每次决策按权重抽一个成员，各 env 用独立 RNG。

## 评估

```bash
uv run --group train 7g523-eval --checkpoint runs/<run>/agent.pt \
    --episodes 1000 --opponent greedy

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
- `--arch {shared,towers}`：网络架构（默认 `shared`，共享主干）。`towers` = 独立 actor/critic 两塔（参数 +69.6%），为实验性选项、T6 已判负并关闭该线（见 [experiments/twin-towers-500k.md](./experiments/twin-towers-500k.md)）；默认 `shared` 逐位不变
- `--obs-version {1,2,3,4,5}`：选择观测布局版本，默认 `5`（`OBS_VERSION`）。v1=原始（185+3n）；v2=B0 增广（+3 维点分量 `trick_points`/`remaining_points`/`point_hold`，188+3n，见 [experiments/observation-augmentation-b0.md](./experiments/observation-augmentation-b0.md)）；v3=B1 增广（v2+55：`unseen` 54 维 + `last_player` 1 维，243+3n、2 家 249，见 [experiments/observation-augmentation-b1.md](./experiments/observation-augmentation-b1.md)；结果未确认）；v4=观测 S1+B0（64+21n、2 家 106）；**v5=观测 S1+B0+B1（119+21n、2 家 161，默认）**，v4 是 v5 的逐位前缀（见 [ADR-0008](./adr/0008-observation-layout-v5.md)）。旧 ckpt 按自身 `obs_version` 推理；跨版本热启动对允许方向做段级列重映射（前缀方向新列置 0，`incumbent_top`/`revealed` 的 54→19 段为近似），不可映射方向在加载点显式拒绝
- `--reward-shaping {terminal,trick_diff,win,trick_diff_win}`：奖励分解（默认 `terminal` 旧行为逐位不变）。`trick_diff` 每步 `Φ(s')−Φ(s)`（一局求和 = 终局回报，telescoping）；`win` 终局 `sign(own−max(others)) ∈ {−1,0,+1}`；`trick_diff_win` 为两者叠加。T1 结果见 [experiments/reward-shaping-500k.md](./experiments/reward-shaping-500k.md)（A1 正信号在训练 seed 复现后未复现）
- `--opponent mix|pool`：对手混合 / 多成员联赛；`--pool-member [WEIGHT@]SPEC`（SPEC = `greedy`/`random`/`self`/`ckpt:<agent.pt>`，可重复，权重默认 1）
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

不加参数就是和贪心 bot 玩：

```bash
uv run 7g523-play                          # 默认：贪心 bot，你坐 0 号位
uv run 7g523-play --opponent random        # 先打随机 bot
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

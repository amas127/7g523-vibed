# 训练与评估使用手册

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
    --opponent self --load-checkpoint runs/stage1__1__.../agent.pt \
    --total-timesteps 300000 --eval-interval 20 --eval-opponent greedy
```

对手是当前策略的冻结快照，每 `--self-play-refresh`（默认 50）次 update 刷新；
`--self-play-sample` 让对手按 mask 采样而不是 argmax。`--load-checkpoint`
只是热启动，不影响 `--opponent` 的选择。

## 评估

```bash
uv run --group train 7g523-eval --checkpoint runs/stage1__1__.../agent.pt \
    --episodes 1000 --opponent greedy

uv run --group train 7g523-eval --checkpoint ... --opponent random --json
```

| 指标 | 含义 |
|---|---|
| `learner_score` / `opponent_score` | 平均得分（满分 100） |
| `score_diff` | 平均分差；等于 `100 × mean_return`（RL-2） |
| `win_rate` / `draw_rate` / `loss_rate` | 按单局比分判定 |
| `mean_length` | 平均步数（学习者视角） |
| `illegal_rate` | 非法动作率，必须为 0（mask 是唯一合法性权威） |

## 常用开关

- `--cuda False`：强制 CPU；`--opponent random`：对手换随机 bot
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

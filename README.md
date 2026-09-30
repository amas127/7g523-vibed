# 7鬼523

一个 54 张牌的中式跑牌游戏：打满一墩收分、底牌堆补牌、**出空即撬底**结束本局。
本仓库 = 可独立使用的纯规则核心 + Gymnasium 形状的 RL 环境 + PPO 训练/评分/定级/人机对战全链路。

> **当前口径（2026-09-29）**
> - 规则 **revision 3「出空即撬底」**（`rules_id=2e36dbea44893696`，[ADR-0014](docs/adr/0014-on-empty-digs.md)）。
> - 观测 **v5**（2 家 161 维，[ADR-0009](docs/adr/0009-single-observation-and-comparison.md)）；动作空间 `MultiDiscrete([134, 4])`（[ADR-0001](docs/adr/0001-template-action-space.md) / [ADR-0004](docs/adr/0004-suit-head.md)）。
> - 评分：RandomBot=0 单基准（[ADR-0012](docs/adr/0012-single-gauge-and-greedy-removal.md)）+ 在线 OpenSkill（[ADR-0011](docs/adr/0011-openskill-rating-core.md)）+ 发布绝对表走独立 probit-MLE（[ADR-0013](docs/adr/0013-drift-free-rating-channel.md)）。
> - 测试：`uv run pytest -q -n 4` → **859 passed**。
> - 当前最强是推理期搜索 `search_leafq`（μ≈260）；raw PPO 顶部平台簇 183–187（11 级池，见 [experiments/README.md](docs/experiments/README.md)）。
> 历史口径（旧 `tier`、revision-2）与当前数字**禁止混比**；完整边界见实验索引。

## 快速开始

需要 Python 3.12 + [uv](https://docs.astral.sh/uv/)。

```bash
uv sync                    # 基础依赖：规则核心 / 人机 / 评级（gymnasium, openskill, scipy）
uv sync --group train      # 训练额外依赖：torch (cu132 索引) / numpy / tensorboard

uv run 7g523               # 随机 bot 演示一局
uv run 7g523-play          # 终端人机对战（默认 RandomBot）
uv run python tools/play_ladder.py list   # 对手档位清单：id / 强度 / 启动命令
```

训练与评估示例（完整手册见 [docs/training.md](docs/training.md)）：

```bash
uv run --group train 7g523-train --exp-name stage1 --total-timesteps 300000
uv run --group train 7g523-eval --checkpoint runs/<run>/agent.pt --episodes 1000 --opponent random
```

### 入口命令

| 命令 | 作用 |
|---|---|
| `uv run 7g523` | 随机 bot 演示一局 |
| `uv run 7g523-play` | 终端人机对战；打 checkpoint 需 `--group train`（手册：[docs/human-play.md](docs/human-play.md)） |
| `uv run --group train 7g523-train` | PPO 训练（手册：[docs/training.md](docs/training.md)） |
| `uv run --group train 7g523-eval` | 批量评估 checkpoint |
| `uv run --group train 7g523-elo` | 10 局真人定级会话（`seven523.placement`） |
| `uv run [--group train] 7g523-web` | 浏览器牌桌；无 torch 时只能打 random |

### Python 调用

```python
from seven523.env import Seven523Env

env = Seven523Env(seed=0)          # opponents=None 时自动配 RandomBot
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(0)
```

规则核心也可单独使用：`Game` / `GameState` / `classify` / `beats` / `CATALOG` / `action_mask` 均从 `seven523` 顶层再导出（见 [src/seven523/__init__.py](src/seven523/__init__.py)）。

## 仓库结构

```
src/seven523/      规则核心 + 训练 / 评分 / 定级 / 人机 / 搜索的全部模块
tools/             薄 CLI 研究工具（只依赖 seven523，可独立删改）
tests/             859 项测试（规则 golden、接口回归、跨模块集成）
docs/              规则 / 设计 / ADR / 计划 / 实验报告（入口见下）
runs/ traces/ artifacts/   训练与实验运行产物（gitignored，文档中的路径为示例）
ppo-implementation-details/   外部 PPO 参考实现（git submodule）
```

### 模块地图

| 分组 | 模块 | 职责 |
|---|---|---|
| 规则核心 | `cards` / `rules` / `combos` / `actions` / `game` | 牌与序、规则、牌型判定与比较、固定动作目录、`Game` 状态机 |
| 对局驱动 | `policies` / `match` / `env` | 脚本策略、`Match` 唯一对局驱动、`Seven523Env`（Gymnasium） |
| 轨迹与录制 | `trace` / `play` / `record` | 轨迹格式、终端对局与回放、批量录制接缝 |
| 训练 | `ppo` / `networks` / `train` / `eval` / `metrics` / `league` / `bootstrap` / `power` / `twin` / `history` | PPO 数学与循环、网络、评估、指标、联赛/PFSP、bootstrap-arena、容量/双塔/序列历史实验模块 |
| 评分与阶梯 | `elo` / `ladder` / `duel` / `arena` / `study` / `mle` | 在线评分、定级编排、双人换座比较、联盟赛、manifest 发布、独立 probit-MLE |
| 对手与先验 | `prior` / `pool` / `placement/` | 轨迹 S1 先验、对手池、10 局真人定级包（CLI `7g523-elo`） |
| 浏览器牌桌 | `web/` | 本地 HTTP 服务 + 单页牌桌（CLI `7g523-web`） |
| 研究工具 | `tools/*.py` | `play_ladder.py`（对手注册表）、`build_ladder.py`、`head_to_head.py` / `h2h_screen.py`、`refit_mle.py`、`fit_trace_prior.py`、`arena.py` 等 |

各模块的角色、依赖与接口定义见 [DESIGN.md](DESIGN.md) §1–§2 与 §8。

> **资产现状（2026-09-29 清理）**：只保留真人对战链路所需资产——11 级 manifest 注册的对局
> checkpoint、搜索 rung 的 value 头与 web 插件、placement 的 manifest/prior、真人对战会话
> （`traces/{sessions,web,twins}`）。实验/复核类产物（旧 run、训练语料、模拟研究）已删除，
> 结论保留在 [docs/experiments/](docs/experiments/)；清理明细见 `artifacts/cleanup/`。
> 依赖这些被删产物的回归测试会显式 skip（`test_fit_trace_prior.py` 的语料用例、
> `test_env.py` 的 pilot 编码器、`test_refit_mle.py` 的 T15 产物）。

## 文档导航（单一事实源）

| 主题 | 入口 |
|---|---|
| 规则（唯一事实源） | [RULES.md](RULES.md) |
| 领域语言 / 术语表 | [CONTEXT.md](CONTEXT.md) |
| 设计文档 / 模块接缝 | [DESIGN.md](DESIGN.md) |
| 架构决策记录 | [docs/adr/](docs/adr/)（ADR-0001 – ADR-0014） |
| 全部计划、待办、已否决 | [docs/plans.md](docs/plans.md) |
| 实验报告索引 + 当前结论 + 规范数字 | [docs/experiments/README.md](docs/experiments/README.md) |
| 训练 / 评估手册 | [docs/training.md](docs/training.md) |
| 人机对战 / 定级 / 浏览器手册 | [docs/human-play.md](docs/human-play.md) |
| 下一步方向备忘（value / belief / policy） | [save.md](save.md) |
| 后 v5 结构性选项决策备忘 | [docs/post-v5-structural-options.md](docs/post-v5-structural-options.md) |
| 外部 PPO 参考实现 | [ppo-implementation-details/](ppo-implementation-details/)（submodule） |

**阅读约定**：每个主题只有一个事实源；结论回填到 `docs/experiments/README.md`，新决策写成 ADR，任务状态只维护在 `docs/plans.md`。历史报告（牌型族变更、revision-2 及其之前）只作历史记录，旧数字不得与当前口径混比；每份报告顶部的状态横幅标明其口径。

## 开发

```bash
uv run pytest -q -n 4      # 859 passed（2026-09-29）
```

- 文档均为中文；代码/数据论断给 `file:line` 或命令。
- 单一 owner 接缝：`record` 批量录制、`policies` spec 语法、`study` manifest writer、`prior` 核心、`metrics`/`league`、`placement/`（[ADR-0010](docs/adr/0010-single-owner-seams.md)）。
- 新增实验按 [docs/plans.md](docs/plans.md) 的 T 编号登记；完成后回填实验索引与规范数字表。

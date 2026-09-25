# 人机对战手册

本页讲三件事：怎么直接用 `7g523-play` 打、怎么用 `tools/play_ladder.py`
选对手、怎么跑一局 10 副牌的定级会话。对手清单的单一可信来源是
`tools/play_ladder.py`（`list` 会打印文件是否存在检查）。

> **模型资产状态（2026-09-25）**：牌型族比较规则变更后，`runs/` 下全部旧
> checkpoint 已随旧规则模型一并删除；`tools/play_ladder.py` 目前只保留
> `random`/`greedy` 两个脚本档。新模型训练完成后，按
> [`plans.md`](./plans.md) T15 重新登记档位与强度（`tools/play_ladder.py` 的
> `OPPONENTS` 注册表 + 本页 §3 表），再指向新路径。本页历史训练档位与数字
> 均为旧 `tier` 口径，禁止与新规则结果混比。

## 1. 直接开打（7g523-play）

```bash
uv run 7g523-play                        # 默认对手 GreedyBot（锚 1315）
uv run 7g523-play --opponent random      # 锚 1000，熟悉规则用
uv run --group train 7g523-play --checkpoint runs/<new-run>/agent.pt \
    --seat 1 --rounds 3 --sample         # 需先训练出新模型（T15）
```

- 训练好的 ckpt 需要 `train` 依赖组（torch）；`random`/`greedy` 脚本对手不需要。
  `runs/<new-run>/agent.pt` 是占位符：旧规则 checkpoint 已全部删除，须先按
  [`training.md`](./training.md) 训练（重标定计划见 [`plans.md`](./plans.md) T15）。
- 局中输入 `h` 看帮助、`q` 退出；`--seat` 选座位（默认 0），`--rounds` 连打多局，
  `--sample` 让神经对手按 mask 采样（默认 argmax），`--seed` 固定发牌。
- `--save-trace traces/xxx.json` 存轨迹，`--replay` 回放并逐步校验引擎；
  轨迹里的对手标签形如 `opponent:<id>@seat1` / `anchor:greedy@seat0`，
  真人轨迹后续可按对手强度标定。
- `--num-players` 必须与 ckpt 的训练配置匹配；新模型默认 2 家。

## 2. 用 play_ladder 发现与启动

```bash
uv run python tools/play_ladder.py list          # 表格 + 每个档的启动命令
uv run --group train python tools/play_ladder.py list --json   # 供脚本消费
uv run python tools/play_ladder.py play random -- --seat 1 --rounds 3
uv run python tools/play_ladder.py play greedy -- --rounds 2
```

`play <id>` 只把 id 翻译成 `--opponent random|greedy` 或
`--checkpoint <path>`，其余参数（建议用 `--` 分隔）原样透传给 `7g523-play`；
因此等价于复制 `list` 里该档的启动命令再追加参数。未知 id 或 ckpt 文件缺失
会直接报错并退出码 2。

当前注册表只剩 `random`/`greedy`：旧梯级（`lvl1`–`lvl4`）、平台/练习快照与
实验室臂的 checkpoint 已随旧规则模型删除。重训后把新 ckpt 按
`runs/<new-run>/agent.pt` 路径登记回 `tools/play_ladder.py` 的 `OPPONENTS`
注册表，并按 [`plans.md`](./plans.md) T15 重测强度列，即可用 `play <id>` 启动。

## 3. 对手档位与强度含义

| id | 类型 | 强度 | 说明 |
|---|---|---|---|
| `random` | 脚本 | 锚 1000（固定） | 均匀随机；熟悉规则与界面 |
| `greedy` | 脚本 | 锚 1315（固定） | 贪心一手；默认对手与固定参照 |

旧训练档（`lvl1`–`lvl4`、`base20k`/`base60k`/`base120k`/`base180k`/`base680k`、
`a1`/`b1`/`cp`/`towers`/`w5_*`）及其强度数字（旧 manifest 契约值与 arena Elo）
均已随旧规则 checkpoint 删除而下线：旧路径不可用，数字是牌型族规则变更前的
旧 `tier` 口径历史记录，禁止当作当前档位或与新规则结果混比。重标定与重训后，
按 [`plans.md`](./plans.md) T15 在新规则下重测强度，把新 ckpt 登记回
`tools/play_ladder.py` 的 `OPPONENTS`，再回填本表。

## 4. 推荐顺序

1. `random` → `greedy`：读界面、熟悉出牌与花色选择。
2. 训练档位（练习梯度 → 定级梯级 → 平台顶 → 实验室臂）：旧模型已删除，当前
   不可用；重训后按 [`plans.md`](./plans.md) T15 重新登记档位与强度，再恢复
   「练习梯度逐步加难 → 梯级定级 → 平台顶」的推荐顺序。

## 5. 10 局定级会话（7g523-elo）

```bash
uv run --group train 7g523-elo
# 非交互冒烟（脚本当“真人”）：
uv run --group train 7g523-elo --simulate greedy --games 10 --seed 0
```

- 默认读 `traces/pool10/manifest.json`（2026-09-25 牌型族规则后新训的 10 个
  500k 模型 `s1`–`s10` + `random`/`greedy` 锚点）与 `artifacts/human-elo/prior.json`；
  10 副不同牌、5/5 座位轮换、前 2 局 Thompson 探索、之后 `info` 选档。
- 若显式指定旧 `traces/study/manifest.json`：其中已删除的 ckpt 档位会被自动跳过
  并告警，只剩 `random`/`greedy` 锚点。
- **注意**：manifest 已换为新规则池；`prior.json` 仍是旧规则口径。真人定级上线前
  需按 [`plans.md`](./plans.md) T15 重标定先验。
- 结束时打印 `定级：R ± CI（95%），最近档 <id>，临时 provisional/已定级`。
- 产物在 `traces/sessions/<session-id>/`：每局一份 trace、每局后落盘的
  `session.json`、结束后的 `report.json`（点估计 / CI / 最近档 / 两通道权重）与
  `rungs.json`（对手池元数据）。`--session-id`、`--sessions-dir` 可指定位置。
- 10 局只承诺点估计 + 诚实 CI：研究结论是 RMSE 54–72、95% CI ±100–133
  （见 [`experiments/human-elo-10-games-research.md`](./experiments/human-elo-10-games-research.md)）；
  `provisional` 是正常状态，应继续对局到 CI ≤ 50（默认 `--games 10` 可加大，
  或 `--stop-ci 50 --min-games 8`）。
- `--simulate random|greedy|ckpt:<path>` 让脚本策略替真人跑完整会话，用于验证流程。

## 6. 运行注意

- 命令都在仓库根目录执行；`play_ladder.py play` 会把 ckpt 解析成绝对路径，
  换目录也能跑。
- ckpt 训练/评估细节见 [`training.md`](./training.md)；本页只负责「打」。
- 新模型生成后不要在 `runs/` 里搬动或重命名 ckpt，直接引用路径；`list` 会
  标记缺失项。

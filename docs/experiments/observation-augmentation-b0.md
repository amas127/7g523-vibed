# T2 / 结构性 B0：观测增广 3 维（点分量）500k（7鬼523）

> 资产状态（2026-09-25）：本报告引用的模型 checkpoint 已删除，旧路径不再可用；数值为牌型族规则变更前口径。

> **状态：已完成（2026-09-25）。一句话结论：B0（`obs_version=2`，191→194，追加
> `trick_points`/`remaining_points`/`point_hold`）与配套兼容层已实现并测试通过；
> 500k 训练产物 `runs/t2_b0__1__1790330513` 跑满 499,712 步。主端点 h2h 3×400
> vs `base680k` **+1.74 [−7.94,+11.42]**、vs `w5_ctrl` **+4.92 [−6.72,+16.56]**，
> 均跨 0 → **B0 不单独采用**（<10 Elo 效应需 1500+ 副牌才能定性，见 §5）。
> 兼容层（版本化编码/热启动前缀填充）当时保留供 B1/T4 复用（后随 [ADR-0009](../adr/0009-single-observation-and-comparison.md) 退役，现行唯一布局 v5）；B1（`obs_version=3`，
> `unseen`/`last_player`）已定稿（未确认；与 B0 同牌直接对比 +0.73 [−18.98,+20.43]，
> 见 [`observation-augmentation-b1.md`](./observation-augmentation-b1.md)）。**
>
> 判定口径：[`README.md` §3](./README.md#3-常用命令与口径) / EPV §9；
> 方向设计与验收：[`structural-directions.md` §3.2/§3.4](./structural-directions.md)（T2/B0）；
> 路线图：[`../plans.md`](../plans.md) §3 T2。
> 全部训练/h2h 产物在 `runs/`（gitignore），本文数字均可溯源到其中文件。

## 1. 背景

T2/B0 是 [`structural-directions.md` §3.2](./structural-directions.md) B 方向的第一步：
平台诊断发现平均 24.7 张已出牌对观测不可见（§1.2），但 B1（`unseen`）需要引擎加公开历史。
B0 只加由现有 `View` 可推、无需动引擎的三个标量：

| 特征 | 含义 | 归一化 |
|---|---|---|
| `trick_points` | 当前墩已落桌的分值（`View.trick_cards`） | / `rules.total_points` |
| `remaining_points` | 尚未被任何人赢走的分 = `total_points − Σscores − trick_points` | / `rules.total_points` |
| `point_hold` | 自己手牌的分值（`View.hand`） | / `rules.total_points` |

obs 191 → 194（`obs_version=2`，对 n 家公式 `188 + 3n`）。先验预期 +0～50 Elo、方差极大，
是 B 方向最便宜的信息下注；若 B0 有效说明模型确实缺「本墩值多少分」这类标量信息
（SD §3.5）。

## 2. 实现与兼容层

### 2.1 v2 布局（`src/seven523/env.py`）

- `_SEGMENTS_V1` 逐位不动；`_SEGMENTS_V2 = _SEGMENTS_V1 + 3 段`（`trick_points`、
  `remaining_points`、`point_hold`），`observation_dim(2, 2) == 194`、
  `observation_dim(n, 2) == 188 + 3n`。v1 是 v2 的逐位前缀（`test_v1_is_an_exact_prefix_of_v2`）。
- `observation_dim(num_players, obs_version)`、`encode_observation(view, rules, obs_version)`、
  `Seven523Env(obs_version=...)` 全链按版本编码；未知版本报 `ValueError`。B0 时默认
  `OBS_VERSION = 2`（写作时工作树因 B1 已切到 3，见 §6）。
- 三个写入器只读公开 `View`；`tests/test_env.py::test_v2_encoding_does_not_leak_hidden_cards`
  用「仅隐藏字段不同」的视图对固定无泄漏。

### 2.2 兼容层（跨版本 checkpoint / 推理 / 热启动）

这一层是 T2/T3/T4 共用设计（SD §3.3），B0 一次做完：

- **checkpoint 记录版本**：`networks.save_agent` 写入 `obs_version`；`load_agent` 对旧 payload
  （无字段）按 v1 载入；`Agent` 默认 v1，训练显式传 `--obs-version`。维度本身不能表示版本
  （v1 3 家与 v2 2 家都是 194 维）。
- **推理按 agent 自己的版本编码**：`NeuralPolicy` 用 `agent.obs_version` 调
  `encode_observation`。旧 v1 ckpt 在 v2 环境里读 191 维前缀，动作与旧快照逐位一致
  （`test_v1_probe_checkpoint_acts_in_a_v2_env_like_the_snapshot`）。
- **热启动前缀填充**：`warm_start_into` 对「同玩家数、目标版本更新、源更窄」的 ckpt，
  把旧 `network.0.weight[:, :191]` 拷入、新列置 0，其余同形张量照拷；v1→v2 的初始函数与
  `base680k` 在 v1 前缀上逐位相同、新特征零贡献起步
  （`test_warm_start_pads_the_first_layer_and_keeps_the_v1_function`）。
- **玩家数守卫**：`observation_num_players` 由版本 + 每玩家段宽度反推玩家数；v1 3 家与
  v2 2 家同为 194 维但段边界不同，禁止前缀填充；跨玩家数同样拒绝
  （`test_warm_start_refuses_layouts_that_only_look_compatible_by_width`、
  `test_warm_start_refuses_cross_player_count_padding`）。
- **训练接线**：`train.py --obs-version {1,2,3}`（默认 `OBS_VERSION`）经 `make_env` 透传到
  env；`obs_dim` 由 `observation_dim` 派生；本次 B0 run 的日志首行即
  `warm start: copied 8 tensors ... (obs 191/1 -> 194/2, nvec [134,4] -> [134,4])`。

### 2.3 测试

B0 落地时全套 **334 passed**（`python -m pytest -q`，2026-09-25）；B1/v3 随后进入工作树，
计数与部分测试名会再变。B0 相关回归：

- `tests/test_env.py`：版本化 dim（191/194、`185+3n`/`188+3n`）、v1 是 v2 逐位前缀、
  v2 点分量在构造视图与真实中局视图上的数值、不泄漏隐藏牌、env 按配置版本发布 obs、
  未知版本报错。
- **v1 逐位金标**：`test_v1_encoding_matches_legacy_snapshot_fixture` 用
  `tests/data/legacy_v1_views.json`（24 个视图，来自 T1 快照 `/tmp/t7_disc_pkg @ 1c8881a`，
  基准 ckpt `runs/probe/base_step00696320.pt`）逐位比对 v1 编码；
  `test_probe_checkpoint_actions_match_the_legacy_fixture` 另比对 12 个 `base680k` greedy 动作。
- `tests/test_networks.py`：ckpt 记录/恢复 `obs_version`、旧 ckpt 缺省 v1、194 维碰撞由版本
  区分、热启动填充保持 v1 函数逐位不变、同宽不同布局/跨玩家数拒绝、真实视图上填充前后
  动作一致。
- `tests/test_train.py`：`--obs-version` 解析与透传、默认布局、v1→v2 热启动填充、
  等宽位移 ckpt 拒绝（测试名以当前工作树为准）。

## 3. 训练产物

母版协议（SD §7.1）：warm start `base680k`、seed 1、500k、SAME_STEP、对手 greedy、
`reward_shaping=terminal`；唯一改动 `--obs-version 2`。

| 项 | 值 | 来源 |
|---|---|---|
| run 目录 | `runs/t2_b0__1__1790330513` | 目录 + `args.json` |
| 配置 | seed 1、obs_version 2、gamma 0.99、vf_coef 0.5、hidden 128、`--load-checkpoint runs/probe/base_step00696320.pt` | `runs/t2_b0__1__1790330513/args.json` |
| 预算/步数 | 488 updates、499,712 步（`metrics.csv` 第 489 行） | `runs/t2_b0__1__1790330513/metrics.csv` |
| obs | obs_dim 194 / obs_version 2（warm start 日志 `obs 191/1 -> 194/2`） | `runs/t2_b0_train.log` |
| `agent.pt` sha256 | `d51e343b797bbcea2d29b2eba19e3a7bd0991efca59944c12aa25aa1fbaf68c7` | `sha256sum runs/t2_b0__1__1790330513/agent.pt` |
| 墙钟 | ~350s（`args.json` 06:01:53 → `agent.pt` 06:07:43；日志 SPS ~1300–1400） | 文件 mtime + `runs/t2_b0_train.log` |
| metrics | 488 行、无 NaN | `runs/t2_b0__1__1790330513/metrics.csv` |

末行训练指标（`499712,0.0761905,23.6429,20674,...,0.560847,1430`）只作健康度参考，
不跨臂比较（各臂奖励尺度相同此处均为 terminal，但训练 EV 与评测分辨率无关）。

## 4. 主端点（h2h，同牌换座）

工具 `tools/head_to_head.py --device cpu --workers 8`；3 seed × 400 副牌 = 1200 副牌 /
2400 局，JSON `deals=1200`、`games=2400`。**左=B0、正 = B0 更强**；参照 ckpt 是 v1，
按各自版本编码评估（SD §3.4）。数字来自
`runs/h2h_t2_b0_vs_base680k.json`、`runs/h2h_t2_b0_vs_w5_ctrl.json` 的
`combined.elo_diff.mean/ci/bootstrap_se/between_seed_sd`、`combined.winrate`、
`combined.mean_score_diff`、`per_seed[].seed/elo_diff`（p = `combined.deal_sign.p_value`）。

| 参照 | mean | 95% CI | bootstrap SE | between-seed sd | winrate | mean_score_diff | per-seed（Elo） | p |
|---|---|---|---|---|---|---|---|---|
| `base680k` | +1.7384 | [−7.9448, +11.4215] | 8.5421 | 8.5570 | 0.5025 | −0.1208 | [−0.8686, +11.2956, −5.2119] | 0.6874 |
| `w5_ctrl` | +4.9226 | [−6.7158, +16.5609] | 10.2848 | 3.0510 | 0.5071 | +1.3167 | [+5.2119, +7.8186, +1.7372] | 0.2004 |

## 5. 判定

- 两个主端点 95% CI 都跨 0（`base680k` 半宽约 ±9.7、`w5_ctrl` 约 ±11.6），
  点估计 +1.74 / +4.92 均 < +10：**B0 不单独采用**，也不声称有任何可分辨的强度变化。
- 不显著 ≠ 效应为 0：按 SD §3.4，<10 Elo 的 B 效应需 **1500+ 副牌**才能定性；
  本报告只回答「3×400 上不可分辨」，没有对 B0 的上限做等效检验。
- 兼容层与 v2 布局当时**保留**：它们是 T3/B1、T4/观测 S1 的共同前置（`plans.md` T2/T4），
  且 v1 ckpt/工具链全部逐位兼容（§2.3）。（后随 [ADR-0009](../adr/0009-single-observation-and-comparison.md) 退役，现行唯一布局 v5。）

## 6. 后续

1. **B1（`obs_version=3`）已定稿（2026-09-25，未确认）**：v3 在 v2 后追加 `unseen`（54 维）与
   `last_player`（1 维），对 2 家为 249 维（公式 `243 + 3n`）；实现、355 passed、500k 训练与
   3×400 h2h 见 [`observation-augmentation-b1.md`](./observation-augmentation-b1.md)。
2. **B1 vs B0 同牌直接对比已完成**：增量 +0.73 [−18.98,+20.43]，与 0 不可分 →
   「已出牌历史」相对 3 个点分量无可分辨增量。
3. B1 也不可分辨（未确认）：B 方向按 `plans.md` §8 门控已收束，下一优先转向 C（对手分布）；
   B0 的兼容层与 `--obs-version` 开关**当时决定**作为观测布局的长期机制保留（后随 [ADR-0009](../adr/0009-single-observation-and-comparison.md) 退役，现行唯一布局 v5）。

## 7. 复现与产物

```bash
# 训练（B0 = 母版协议 + --obs-version 2）
.venv/bin/python -m seven523.train --exp-name t2_b0 --seed 1 \
  --total-timesteps 500000 --cuda True --tensorboard True --checkpoint-interval 0 \
  --log-interval 50 --opponent greedy --load-checkpoint runs/probe/base_step00696320.pt \
  --obs-version 2

# 主端点 3×400（对 base680k / w5_ctrl 各一次）
.venv/bin/python tools/head_to_head.py \
  --left t2_b0=ckpt:runs/t2_b0__1__1790330513/agent.pt \
  --right base680k=ckpt:runs/probe/base_step00696320.pt \
  --seeds 0,1,2 --pairs 400 --device cpu --workers 8 \
  --json > runs/h2h_t2_b0_vs_base680k.json
# w5_ctrl：--right w5_ctrl=ckpt:runs/w5_ctrl__1__1790319698/agent.pt

# 测试（B0 落地时 334 passed）
.venv/bin/python -m pytest -q
```

产物：`runs/t2_b0__1__1790330513/{agent.pt,args.json,metrics.csv,tb}`、
`runs/t2_b0_train.log`、`runs/h2h_t2_b0_vs_{base680k,w5_ctrl}.json`、
金标 `tests/data/legacy_v1_views.json`；B0 探针/对比脚本无独立产物，
机制证据在测试（§2.3）。

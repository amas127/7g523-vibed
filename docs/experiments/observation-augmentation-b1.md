# T3 / 结构性 B1：已出牌历史 + `unseen` + `last_player` 500k（7鬼523）

> 资产状态（2026-09-25）：本报告引用的模型 checkpoint 已删除，旧路径不再可用；数值为牌型族规则变更前口径。

> **状态：已完成（未确认，2026-09-25）。一句话结论：B1（`obs_version=3`，v2+55：`unseen`
> 54 维 + `last_player` 1 维，2 家 249 维、公式 `243 + 3n`）与引擎公开历史（`played`）
> 已实现并测试通过（355 passed）；500k 训练产物 `runs/t3_b1__1__1790331840` 跑满
> 499,712 步、无 NaN。主端点 h2h 3×400 vs `base680k` **+4.20 [−6.75,+15.15]**（CI 跨 0、
> 点估计 <+10 → 失败）、vs `w5_ctrl` **+13.04 [+1.93,+24.15]**、相对 B0 **+0.73
> [−18.98,+20.43]**；3 个 h2h 只有 1 个达标，相对 B0 的增量 ≈0 且 CI 极宽 → **B1 未确认**，
> 不能归因于「已出牌历史」带来的增量。单训练 seed、seed 间方差大；<10–15 Elo 效应需
> 1500+ 副牌才能定性（见 §6）。B 线（B0 null、B1 未确认）就此收束，下一优先见
> [`../plans.md`](../plans.md) §8（T5 C → T6 F）。`obs_version` 2/3 兼容层后随
> [ADR-0009](../adr/0009-single-observation-and-comparison.md) 退役（现行唯一布局 v5）。**
>
> 判定口径：[`README.md` §3](./README.md#3-常用命令与口径) / EPV §9；
> 方向设计与验收：[`structural-directions.md` §3.2–§3.4](./structural-directions.md)（T3/B1）；
> 前序 B0 报告：[`observation-augmentation-b0.md`](./observation-augmentation-b0.md)；
> 路线图：[`../plans.md`](../plans.md) §3 T3。
> 全部训练/h2h 产物在 `runs/`（gitignore），本文数字均可溯源到其中文件。

## 1. 背景

T3/B1 是 [`structural-directions.md` §3.2](./structural-directions.md) B 方向的第二步、
也是主力信息下注：平台诊断发现一个决策点平均有 **24.7 张**（最多 54 张）已经打掉的牌
从观测里消失（SD §1.2），而 B0 只补了「当前墩值多少分」这类 3 个标量。B1 补齐两类公开
信息：

- **已出牌历史**：`GameState`/`View` 新增 `played`（按打出顺序的已完成墩牌），编码器直接
  给 `unseen` 集合（54 − 手牌 − 亮牌 − `played` − 当前墩），让模型不必自己做集合减法；
- **当前墩归属**：`last_player`（当前需压过的牌顶属于哪一家的绝对座位，1 维）。

B1 与 B0 的关系是「同一版本链上的追加」：B0 的 `obs_version=2` 兼容层（版本化编码 /
热启动首层前缀填充）是 B1 的前置，v3 不重做这一层（见 B0 报告 §2.2）。

## 2. 实现与版本设计

### 2.1 v3 布局（`src/seven523/env.py`）

- `OBS_VERSION = 3`、`OBS_VERSIONS = (1, 2, 3)`（`env.py:35-36`）；
  `_SEGMENTS_V3 = _SEGMENTS_V2 + 2 段`（`env.py:191-193`），**只追加、不改动任何旧段**，
  因此 v1、v2 都是 v3 的逐位前缀（`test_v1_and_v2_are_exact_prefixes_of_v3`）。

| 追加段 | 2 家宽 | 编码式 | 公开性 |
|---|---|---|---|
| `unseen` | 54 | `54 − 自己手牌 − revealed − played − 当前墩牌` 的 multi-hot（`env.py:_write_unseen`） | 公开：对手手牌与底牌堆的**并集**，编码不区分「在哪家/什么顺序」，不泄漏隐藏牌 |
| `last_player` | 1 | 无 incumbent（`last_player is None`）→ `0.0`；否则 `(last_player + 1) / n`，绝对座位 `0..n−1` → `1/n..1`（`env.py:_write_last_player`） | 公开 |

- 维度公式：**v3 = `243 + 3n`**（v1 `185+3n`、v2 `188+3n`）；2 家 **249**、3 家 **252**
  （`.venv/bin/python` 现场复核 `observation_dim`）。
- 差分泄漏：只编码公开可推量（ADR-0002）；
  `test_encoding_does_not_leak_hidden_cards[1,2,3]` 用「仅隐藏字段不同」的视图对验证 v1/v2/v3
  逐位相同；`test_v3_unseen_segment_is_the_public_set_complement` 锁定 `unseen` 恰为
  公开集合补集；`test_v3_last_player_segment_normalises_the_incumbent_owner` 覆盖
  `None` 与全部座位的归一化值。

### 2.2 引擎公开历史（`src/seven523/game.py`）

- `GameState.played: tuple[Card, ...] = ()`（`game.py:44`）、`View.played`（`game.py:98`）、
  `Game.view` 填充（`game.py:157`）。
- `_end_trick` 里 `played = state.played + state.trick_cards`（`game.py:222`），并写入
  `after` 状态（`game.py:264`）——**只累积已结束的墩**（不含当前墩；撬底终局收入手牌的
  牌不再入历史，因为局已结束）。
- `Game.restore`（trace 回放路径）不传 `played`，按字段默认恢复为 `()`；`trace.py`
  schema 与编解码**未改**（trace 仍只存 `Deal` + 每步动作/分数）。
- 回归：`test_played_history_accumulates_finished_tricks_only`、
  `test_played_history_accounts_for_every_card_in_a_full_game`（一局内每张出现过的牌都能在
  `played ∪ trick_cards ∪ hands ∪ draw` 中对上）。

### 2.3 版本链与兼容（`train.py` / `networks.py`）

- `train.py --obs-version {1,2,3}`（默认 `OBS_VERSION`）经 `make_env` 透传到 env；
  `obs_dim` 由 `observation_dim` 派生。`networks.py` 本次仅注释更新，兼容层主体在 B0 已落地。
- 跨版本热启动：v2 ckpt 升 v3 时首层按列填充（旧 194 列照拷、新增 55 列置 0），
  初始函数与 `base680k` 在 v1/v2 前缀上逐位相同、新特征零贡献起步
  （`test_warm_start_pads_the_v2_checkpoint_into_v3`、`test_v1_checkpoint_still_pads_through_to_v3`、
  `test_padding_keeps_the_policy_action_identical_on_real_views_v2_to_v3`）。
- 旧 ckpt 推理按自身 `obs_version` 编码：v1 `base680k` 在 v3 环境里读 191 维前缀，动作与
  旧快照逐位一致（`test_v1_probe_checkpoint_acts_in_a_v3_env_like_the_snapshot`、
  `test_fixture_views_keep_the_v1_v2_prefix_under_v3`）。

## 3. 测试

工作区 **355 passed**（`.venv/bin/python -m pytest -q`，2026-09-25，21.87 s），比 B0 落地时
的 334 条多 21 条、零回归（B0 报告 §2.3 的基线）。B1 相关回归：

- `tests/test_env.py`：`test_v1_and_v2_are_exact_prefixes_of_v3`、
  `test_v3_unseen_segment_is_the_public_set_complement`、
  `test_v3_last_player_segment_normalises_the_incumbent_owner`、
  `test_encoding_does_not_leak_hidden_cards[1,2,3]`、
  `test_env_defaults_to_v3_and_rejects_unknown_versions`。
- `tests/test_game.py`：`test_played_history_accumulates_finished_tricks_only`、
  `test_played_history_accounts_for_every_card_in_a_full_game`、
  `test_view_and_observation_do_not_leak_hidden_state`。
- `tests/test_networks.py`：`test_warm_start_pads_the_v2_checkpoint_into_v3`、
  `test_v1_checkpoint_still_pads_through_to_v3`、
  `test_padding_keeps_the_policy_action_identical_on_real_views_v2_to_v3`、
  `test_v1_probe_checkpoint_acts_in_a_v3_env_like_the_snapshot`、
  `test_fixture_views_keep_the_v1_v2_prefix_under_v3`、同宽异布局/跨玩家数守卫。
- `tests/test_train.py`：`test_train_agent_uses_the_default_v3_layout`、
  `test_train_warm_starts_a_v1_checkpoint_by_padding_the_first_layer`、
  `test_train_warm_starts_a_v2_checkpoint_into_v3`、
  `test_train_warm_start_skips_a_shifted_equal_width_checkpoint`。

另有一次独立复核（B0 前快照/工作区）确认：v1 编码与 B0 前快照逐位一致、v2/v3 前缀关系
成立、`unseen` 不泄漏隐藏牌。

## 4. 训练产物

母版协议（SD §7.1）：warm start `base680k`、seed 1、500k、SAME_STEP、对手 greedy、
`reward_shaping=terminal`；唯一改动 `--obs-version 3`。

| 项 | 值 | 来源 |
|---|---|---|
| run 目录 | `runs/t3_b1__1__1790331840` | 目录 + `args.json` |
| 配置 | seed 1、obs_version 3、gamma 0.99、vf_coef 0.5、hidden 128、`--load-checkpoint runs/probe/base_step00696320.pt`、opponent greedy | `runs/t3_b1__1__1790331840/args.json` |
| 预算/步数 | 488 updates、499,712 步（`metrics.csv` 第 489 行） | `runs/t3_b1__1__1790331840/metrics.csv` |
| obs | obs_dim 249 / obs_version 3（warm start 日志 `obs 191/1 -> 249/3, nvec [134,4] -> [134,4]`） | `runs/t3_b1_train.log` 首行 |
| `agent.pt` sha256 | `1c00b10f1b24e323ac050f7685073efc599100ebf47470cf10dd28094bcf41f2` | `sha256sum runs/t3_b1__1__1790331840/agent.pt` |
| 墙钟 | **319 s**（`args.json` mtime 06:24:00 → `agent.pt` 06:29:19；日志 SPS 1570–1641，末行 metrics 1573） | 文件 mtime + `runs/t3_b1_train.log` |
| metrics | 488 行、无 NaN | `runs/t3_b1__1__1790331840/metrics.csv` |

末行训练指标（`499712,0.192683,24.1463,20732,…,0.643004,1573`）只作健康度参考，不跨臂比较。

## 5. 主端点（h2h，同牌换座）

工具 `tools/head_to_head.py --device cpu`；每次 3 seed × 400 副牌 = 1200 副牌 / 2400 局
（JSON `deals=1200`、`games=2400`，k=3、seeds [0,1,2]、bootstrap 4000、confidence 0.95）。
**左=B1、正 = B1 更强**；参照 ckpt 按各自版本编码评估（`base680k`/`w5_ctrl` 为 v1，
B0 为 v2）。数字来自 `runs/h2h_t3_b1_vs_base680k.json`、`runs/h2h_t3_b1_vs_w5_ctrl.json`、
`runs/h2h_t3_b1_vs_t2_b0.json` 的 `combined.*` 与 `per_seed[].seed/elo_diff`
（p = `combined.deal_sign.p_value`）。

| 参照 | mean | 95% CI | bootstrap SE | between-seed sd | winrate | mean_score_diff | per-seed（Elo） | p |
|---|---|---|---|---|---|---|---|---|
| `base680k` | +4.1996 | [−6.7515, +15.1508] | 9.1399 | 9.6775 | 0.5060 | +1.0458 | [+9.1223, −6.9496, +10.4262] | 0.5317 |
| `w5_ctrl` | +13.0388 | [+1.9273, +24.1503] | 9.8192 | 7.4318 | 0.5188 | +1.4792 | [+13.9048, +5.2119, +19.9996] | 0.0286 |
| B0（`runs/t2_b0__1__1790330513/agent.pt`，增量归属） | +0.7260 | [−18.9815, +20.4335] | 9.1846 | 17.4155 | 0.5010 | +0.7750 | [−0.4343, −16.0804, +18.6927] | 0.7766 |

读法：

- **主端点失败**：vs `base680k` 合并 95% CI 跨 0（半宽约 ±11）且点估计 +4.20 < +10；
  per-seed 在 −6.95 与 +10.43 间翻符号。
- **3 个 h2h 只有 1 个达标**（vs `w5_ctrl`）：vs `w5_ctrl` 点估计 +13.04、CI 排 0（p=0.029），
  但 `w5_ctrl` 是更弱的控制组（同批 B0 也在此拿到 +4.92）；相对 B0 的同牌直接对比只有
  **+0.73 [−18.98,+20.43]**，between-seed sd 17.4、CI 半宽约 ±20 → B1 的 55 维增量与 0 不可分。
- 因此不能把 vs `w5_ctrl` 的正值归因于「已出牌历史」：它没有稳定超过 B0，也没有在
  `base680k` 上排 0。

## 6. 判定

**未确认**（B1 不采用、不声称有可分辨的强度变化）：

1. 主端点（vs `base680k`）CI 跨 0 且点估计 < +10，按统一口径
   （[`README.md` §3](./README.md#3-常用命令与口径) / EPV §9）直接失败。
2. 3 个 h2h 只有 1 个达标（vs `w5_ctrl`）；唯一达标项（+13.04）与 B0 同口径的 +4.92 同向，
   相对 B0 的增量 ≈0 且 CI 极宽 → B1 的增广内容无可分辨贡献。
3. 单训练 seed、seed 间方差与效应同量级（vs B0 between-seed sd 17.4；per-seed
   −0.43/−16.08/+18.69 翻符号）——与 A1 的教训一致：单 seed 高值不能当结论
   （[`reward-shaping-500k.md`](./reward-shaping-500k.md) §3.4）。
4. 不显著 ≠ 效应为 0：按 SD §3.4 / README §3 的分辨率表，<10–15 Elo 的效应需
   **1500+ 副牌**（10 Elo 级 ≈1500–2000 副）才能定性；本报告只回答「3×400 上不可分辨」，
   没有做等效/非劣检验。
5. 结论：B0 null、B1 未确认 → **B 线收束**；下一优先按 [`../plans.md`](../plans.md) §8
   转向 T5 C（逐局对手/PFSP），其后 T6 F。

## 7. 限制与后续

- **单训练 seed**：B1 只有 `--seed 1` 一个 500k run（B0 同）。若要像 T1 那样补 seed 2/3/4
  复现，需再跑 3 个 500k（母版协议不变），本报告未做。
- **分辨率不足**：相对 B0 的增量在 3×400 上的 CI 半宽约 ±20；即使真实增量 ~+10，
  该配置也测不出来。要定性 B1 增量需 ≥1500 副牌或更多 seed。
- **未测的组合**：B1 × hidden 256 / 更大容量（T8）与 A1+B1 组合臂按现有门控不排期；
  B1 对更强对手（C 的 ≥学习者成员）或人类的效果也未知。
- **保留物（当时）**：v3 布局、`--obs-version {1,2,3}` 开关与版本化兼容层**保留**，作为 T4
  观测 S1 迁移的基础；后随 [ADR-0009](../adr/0009-single-observation-and-comparison.md) 退役
  （现行唯一布局 v5）；v4/v5 编号与段级列重映射热启动设计见
  [`observation-slimming.md`](./observation-slimming.md)「实现分析（2026-09-25，obs_version v2 落地后）」。
- **后续**：按 `plans.md` §8，先 T5 C（逐局对手 + PFSP + ≥学习者对手），其后 T6 F；
  B 线不重做，除非 C/F 或更强对手给出新证据。

## 8. 复现与产物

```bash
# 训练（B1 = 母版协议 + --obs-version 3）
.venv/bin/python -m seven523.train --exp-name t3_b1 --seed 1 \
  --total-timesteps 500000 --cuda True --tensorboard True --checkpoint-interval 0 \
  --log-interval 50 --opponent greedy --load-checkpoint runs/probe/base_step00696320.pt \
  --obs-version 3

# 主端点 3×400（对 base680k / w5_ctrl / B0 各一次）
.venv/bin/python tools/head_to_head.py \
  --left t3_b1=ckpt:runs/t3_b1__1__1790331840/agent.pt \
  --right base680k=ckpt:runs/probe/base_step00696320.pt \
  --seeds 0,1,2 --pairs 400 --device cpu \
  --json > runs/h2h_t3_b1_vs_base680k.json
# w5_ctrl：--right w5_ctrl=ckpt:runs/w5_ctrl__1__1790319698/agent.pt
# B0：  --right t2_b0=ckpt:runs/t2_b0__1__1790330513/agent.pt

# 测试（B1 落地时 355 passed）
.venv/bin/python -m pytest -q
```

产物：`runs/t3_b1__1__1790331840/{agent.pt,args.json,metrics.csv,tb}`、
`runs/t3_b1_train.log`、`runs/h2h_t3_b1_vs_{base680k,w5_ctrl,t2_b0}.json`；
实现证据在源码（`env.py`/`game.py`/`train.py`）与测试（§3）。

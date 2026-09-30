# EVH 事件级 + 相对座位历史编码器：实现与实验报告

> **状态**：已完成实现 + random 主端点（9 run）+ self-play（6 run）；判定见 §5/§6。
> **口径**：revision-3（出空即撬底，`rules_id=2e36dbea44893696`）、obs v5（2 家 161 维）、
> T17 工作区；训练 500k、8×128、`hidden=128`、`shared`、lr 2.5e-4 退火，镜像
> `t17pool`/`t17seq`（pilot §5.2）。
> **预注册**：`docs/event-history-plan.md` §1.2/§1.5；合并 `mean ± q·max(mean boot SE, seed sd)/√k`，
> `q=t_{0.975,k-1}`（k=3 → 4.303），同时报 z=1.96 口径。
> **用户拍板**：从零训练（删中性起点）；self-play 必做；`event_blind` 不训；random 第一波
> `event`/`event_noise`/`event_seatblind` ×3。

---

## 0. 结论摘要

1. **实现完成、默认路径不变**：EVH 作为 opt-in 第二输入（事件 64 维 + 相对座位 + 无状态 GRU）
   接入 engine / history / network / PPO / train / league；`seq_len=0 且 event_len=0` 时
   `Agent` 参数量 55,179、前向与 `None` 逐位一致；全套 **680 passed**（基线 636）。
2. **random 主端点 `event vs event_noise` 为 null**：3-seed 点估计 **−5.02**，z-CI
   [−22.99,+12.95]、t-CI [−44.47,+34.43]（seed sd 15.9 主导）→ 按预注册 §1.5 走
   **停/无证据分支**；归属端点 `event vs event_seatblind` **+0.83** [−11.82,+13.47]，也无证据。
3. **self-play 主端点弱正但不足**：**+2.99**，z-CI [−9.15,+15.14]、t-CI [−23.67,+29.66]，
   3 seed 同号为正（seed sd 2.17）——方向一致但幅度 < +10、CI 跨 0，不 ship。
4. **结论口径**：本预算/设计下**无 ≥+10 证据**，不是「效应恰为 0」（功效见 §5.4/§6）。

---

## 1. 实现摘要（全部 opt-in，默认路径逐位不变）

| 模块 | 改动 | 锚点 |
|---|---|---|
| 引擎 | 新增 `Play(seat, cards, opens_trick, went_out)`；`GameState.plays` / `View.plays`（默认 `()`）；`step` 的 pass 与出牌分支追加事件；`restore` 从空开始，trace 回放逐手重建、**trace 格式零改动** | `src/seven523/game.py:27`、`:72`、`:131`、`:191`、`:207`、`:237`；trace 不 bump |
| 编码 | `EVENT_DIM=64`（54 多热 + 6 kind + size + is_pass + opens_trick + went_out）；`event_length(rules)=n*54`；`encode_events(view, rules, length, order, pass_mode, boundary_blind, went_out, blind, noisy)`；相对座位 `(play.seat - view.seat) % n`，pad 下标 `n`；墩内确定置换（保 opener 位置，含 identity 回退旋转）；右 padding、截断取最近；`EventHistoryWrapper` | `src/seven523/history.py:54`、`:83`、`:122`、`:153`、`:261` |
| 网络 | `EventSequenceEncoder`（逐事件 MLP 64→32→32 + seat embedding 16 + GRU 48→32，读出最后一个真实事件的 hidden；尾部 pad 在 GRU 前截掉，语义等价、实测提速）；`Agent(event_len, event_dim, event_emb, event_hidden, seat_emb, event_blind, event_seat, event_order, event_pass, event_boundary_blind, event_noisy, event_went_out, num_players)`；`history_layout(agent)` marker；`save/load_agent` event 字段；`warm_start_from`/`warm_start_into` layout fail-loud；`NeuralPolicy.act` 从 `View` 重建 | `src/seven523/networks.py:129`、`:490`、`:235`；参数：`SequenceEncoder` 5680 vs `EventSequenceEncoder` **11056**（MLP 3136 + seat 48 + GRU 7872） |
| PPO/训练 | `RolloutBatch` 增 `events/event_seats/event_mask`（默认 `None`）；`flatten` 切成 `(B,L,64)/(B,L)/(B,L)`；`ppo_update` 透传；`train.py` 存储 + bootstrap `next_events*` + `make_env` wrapper + `--event-*` 开关；self-play refresh 断言 layout 一致 | `src/seven523/ppo.py`、`src/seven523/train.py:187` 起 |
| 联盟 | 冻结对手改为 `copy.deepcopy(agent).eval()`，构造时断言 `history_layout` 一致；self-play 对手从**自己的** `View` 重建事件（`env` 按行动座位传 view） | `src/seven523/league.py:154` |

**默认路径回归**：`seq_len=0 且 event_len=0` 时 `Agent` 不建任何编码器，参数量
**55,179**（`tests/test_networks.py::test_arch_parameter_counts_match_the_design` +
新增 `test_event_parameter_count_and_default_path_are_pinned`），`policy_logits` 与传
`None` 逐位相等；`RolloutBatch` 三字段保持 `None`。全套测试 **680 passed**
（基线 636 + 新增 44；`uv run --group train pytest -q`）。

**事件数上界**：`event_length(rules)=n*54`（n=2→108、n=3→162），随机策略 120 局/家数
实测最大 n=2 **72**、n=3 **93**，断言 `len(plays) <= event_length` fail-loud
（`tests/test_history.py::test_event_count_never_exceeds_the_analytic_bound`）。

**性能（plan §7.1 要求实测）**：CPU `encode_events` 0.120 ms/call（54 events）；
`EventSequenceEncoder` CPU 前向：满 108 pad **3.27 ms/decision**、真实长度 32 时 **1.10 ms**、
长度 8 时 **0.36 ms**（尾部 pad 截断优化后）；单决策完整 event agent（obs 编码 + trunk + 双头）
在 h2h 口径的吞吐见下。训练吞吐（3 并发、500k）：random 批 **898–1071 sps**（中位数；前序
 t17pool 1057（3 并发）、t17seq 612（6 并发）为不同并发口径的历史参照，见 review 复核），
self-play 批见 §4/§6。
h2h 单条（3 deal seed × 400 副，`--device cpu --workers 4`）实测 **27–35 s**，无内存压力。

---

## 2. 预检与审计（obs v1–v4 架构残留）

**结论：无残留，无需为删而删。** 证据（只读 grep，2026-09-26）：

- `grep -rn "REMAP\|remap\|_first_layer\|segment_spans" src tests tools` → **0 命中**
  （`src/seven523/networks.py` 无任何列重映射/版本 dispatch）。
- `grep -rn "obs_version\|OBS_VERSION" --include=*.py src tests` 仅剩：`env.py:32` 的
  `OBS_VERSION=5`、`save_agent` 写入、`load_agent` 的 `obs_version != OBS_VERSION` 硬校验
  （`networks.py:551` 一带的 docstring）与对应测试。
- 历史删除记录：commit `2288e4c`「phase 3: retire obs v1-v4 and tier comparison」删除了
  `_first_layer_remap`、`_REMAP_VERSION_PAIRS`、`_IDENTITY_SEGMENTS`、`segment_spans` 及
  旧 fixture（`tests/data/legacy_v1_views.json` 等，`git show 2288e4c --stat`）；ADR-0009
  落地后无代码路径引用 v1–v4 布局。
- 注释清理：`tools/play_ladder.py:97` 的退役提示由「obs v1-v3」更正为「obs v1-v4」
  （ADR-0009 连 v4 一起退役）；`networks.py:553` 的「Only v5 checkpoints load … v1-v4」是
  现行契约说明，非过时残留，保留。
- `tests/test_trace.py` 的 trace v1/v2 是规则修订版本门禁（`play.py:230` 的 pre-v3 拒绝），
  与模型架构无关，未触碰。

---

## 3. 与规划的偏差（含用户决策落地）

1. **从零训练**（用户拍板）：未实现「零填充中性起点」；`warm_start_from` 对 layout 不匹配
   fail-loud，`train.py --load-checkpoint` 把 `RuntimeError` 也转成友好 `SystemExit`
   （新增测试 `test_train_event_warm_start_requires_same_layout`）。未新立 ADR、未恢复列重映射。
2. **self-play 必做**（用户拍板）：步骤 7 已执行，命令为
   `--opponent self --self-play-refresh 10`。规划 §1.3 已确认 `--opponent self` **忽略**
   `--mix-random-prob`，t17self 历史命令里的该 flag 是无操作；本次未写无意义 flag，
   `args.json` 已核对为 `opponent=self`（纯 self-play）。
3. **`event_blind` 不训**（用户拍板）：开关 `--event-blind` 与测试保留
   （`test_train_event_ablation_switches_smoke`），未出 run；主对照 `event_noise`。
4. **`event vs seq` 定位钉死为描述性端点**：verify 发现规划 §1.2 端点族与「仅描述」矛盾；
   已移出 Bonferroni 族（k 由 3 降为 2），§1.2/§1.5/风险 9 同步修订，本报告按描述性读。
5. **`event_noise` 的精确分布（报告写死）**：与规划 §7.2 的「(kind,size) 边际采样」二选一，
   实现取更干净的 **位置索引固定种子 i.i.d. `N(0,1)`**（`EVENT_NOISE_SEED=0`，numpy
   `RandomState`），只保留**真实事件计数**（mask 长度）；seat 置 pad。即：信息为零（同计数下
   任意两个 View 的输入逐位相同），但通道非退化；确定性保证训练/推理逐位一致。
6. **`event_seatblind`（`--event-seat none`）**保留 GRU 输入宽度（32+16），只把 seat 段置零，
   容量不缩水；`event_blind` 连事件向量与 seat 一起零化（与规划 §1.2 语义一致）。
7. **顺带修正**：规划/审查文档的 4 处 verify 小问题（GRU 7872、`event vs seq` 定位、
   review 第 24 条 seatblind、§3.1 anchor `201-212→201-209`、`213-214→214`）已修；
   规划头部加用户决策块并保持自洽。

---

## 4. 训练健康度

所有 run 从零、`cuda=True`、最多 3 进程并发（`nice -n 5`）。loss/entropy/explained_variance
轨迹无 NaN、无发散（`runs/*/metrics.csv`）。

| 臂 | run 目录（seed 1/2/3） | sps 中位数（step≥100k） | 备注 |
|---|---|---|---|
| `event`（random） | `runs/t17event__1__1790452506`、`...__2__1790453045`、`...__3__1790453661` | 998 / 898 / 926 | 500k 全部完成 |
| `event_noise`（random） | `runs/t17eventnoise__1__1790452506`、`...__2__1790453045`、`...__3__1790453661` | 1071 / 947 / 1001 | 同编码器 |
| `event_seatblind`（random） | `runs/t17eventseatblind__1__1790452506`、`...__2__1790453045`、`...__3__1790453661` | 998 / 900 / 928 | `--event-seat none` |
| `t17selfevent` | `runs/t17selfevent__{1,2,3}__1790485172` | **321 / 321 / 323**（中位数，3 并发） | `--opponent self --self-play-refresh 10` |
| `t17selfeventnoise` | `runs/t17selfeventnoise__{1,2,3}__1790486758` | 316–318（中位数，3 并发） | 同上 + `--event-noisy` |

> 时间戳以实际目录为准（见 §10 产物清单）。

---

## 5. random 主端点结果（3 train seed × 3 deal seed × 400 副 = 3600 副/端点）

每 train seed 与同 seed 对照 h2h（`--seeds 0,1,2 --pairs 400 --bootstrap 4000`）。

### 5.1 主端点 `event` vs `event_noise`（Q-A，confirmatory）

| train seed | Elo(左−右) | 95% CI（z，单 seed） | winrate | score diff | p(deal sign) |
|---|---|---|---|---|---|
| 1 | **+13.19** | [−3.94, +30.33] | 0.5190 | +2.0625 | 0.0706 |
| 2 | **−15.94** | [−28.24, −3.64] | 0.4771 | −2.2167 | 0.0344 |
| 3 | **−12.31** | [−25.11, +0.49] | 0.4823 | −1.7667 | 0.0955 |

**3-seed 合并**（k=3）：

- 点估计 **−5.02**；z-CI **[−22.99, +12.95]**；**t(4.303)-CI [−44.47, +34.43]**。
- mean boot SE 11.08、**seed sd 15.88**（seed-sd 主导 → t 口径下**不可判**）。

### 5.2 归属端点 `event` vs `event_seatblind`（Q-B，confirmatory 次要）

| train seed | Elo | 95% CI（z） | winrate | score diff | p |
|---|---|---|---|---|---|
| 1 | +11.75 | [−8.07, +31.56] | 0.5169 | +1.8375 | 0.0834 |
| 2 | +0.58 | [−11.85, +13.01] | 0.5008 | +0.2083 | 0.9024 |
| 3 | −9.85 | [−22.76, +3.06] | 0.4858 | −2.2458 | 0.2138 |

**合并**：点估计 **+0.83**；z-CI **[−11.82, +13.47]**；t-CI **[−26.93, +28.59]**；
mean boot SE 11.17、seed sd 10.80。

### 5.3 描述性端点（`event vs t17pool` 与 `event vs seq`；混淆，只作描述）

| 端点 | seed 1 | seed 2 | seed 3 | 3-seed point | z-CI | t-CI | seed sd |
|---|---|---|---|---|---|---|---|
| `event vs t17pool` | +24.99 [+2.97,+47.01] | −8.54 [−21.07,+3.98] | −17.11 [−34.40,+0.17] | **−0.22** | [−25.40,+24.96] | [−55.50,+55.05] | 22.25 |
| `event vs seq` | +7.82 [−4.67,+20.31] | −11.59 [−23.71,+0.53] | −22.21 [−40.16,−4.26] | **−8.66** | [−25.89,+8.57] | [−46.49,+29.17] | 15.23 |

### 5.4 预注册判定（plan §1.5）

主端点 `event vs event_noise`：`Δ = −5.02 < +5` 且 CI 跨 0（z 与 t 均跨）→
**「停并发布 null」分支**：在本预算/分布下没有事件表示带来 ≥+10 Elo 的证据
（**不等于效应恰为 0**；t 口径下 seed sd 主导，k=3 显式标注「不可判」）。
不触发加 seed 4/5（该分支要求 `Δ ≥ +5` 且 CI 跨 0），不进入单因素调优与机制臂。
归属端点 `+0.83` 亦无 ≥+10 证据；且 2 家下 seat 信号可由 `opens_trick`+pass 位置+座位交替
部分复原（规划 §5.1 的保守读法），0 点估计只能读作「无法隔离出座位增量」。
描述性 `event vs seq` 点估计为负且同时含表示/容量/长度三重混淆，不作归因。

**功效（必须同时报）**：3×400 对 10 Elo 级效应功效约 40%（README §0.2、pilot §8.3）；
本端点实测 seed sd 15.9，t 合并半宽 ±40.3 Elo——即使真实效应为 +10，本设计也只有低概率
给出可行动结论。null 的正确读法是「**无 ≥+10 证据**」。

---

## 6. self-play 阶段（步骤 7，用户拍板必做）

命令：`--opponent self --self-play-refresh 10`（纯 self-play；`args.json` 核对 `opponent=self`，
其中 `mix_random_prob=0.5` 只是 argparse 默认值，`--opponent self` 下**被忽略**、不参与采样），
其余同 random 500k（实测 3 并发下 **321–323 sps**/run）；`league.py` 冻结对手 `deepcopy`
全量第二输入、构造与每次 refresh 断言
layout 一致（`src/seven523/league.py:154`、`src/seven523/train.py` refresh 断言），
端到端 smoke 通过（`tests/test_league.py::test_build_league_self_freezes_the_full_event_config`、
`tests/test_train.py::test_train_self_play_smoke`）。

主端点 `selfevent` vs `selfeventnoise`（3 train seed × 3×400 = 3600 副）：

| train seed | Elo(左−右) | 95% CI（z，单 seed） | winrate | score diff | p(deal sign) |
|---|---|---|---|---|---|
| 1 | **+4.78** | [−16.64, +26.21] | 0.5069 | +0.8125 | 0.5098 |
| 2 | **+0.58** | [−14.91, +16.07] | 0.5008 | +1.4583 | 0.7057 |
| 3 | **+3.62** | [−9.63, +16.87] | 0.5052 | +0.9333 | 0.5815 |

**3-seed 合并**：点估计 **+2.99**；z-CI **[−9.15, +15.14]**；t-CI **[−23.67, +29.66]**；
mean boot SE 10.73、seed sd **2.17**（boot SE 主导）。

**判定**：3 个 seed 点估计**同号为正**（+0.58…+4.78，无 symbol flip，与 random 主端点的
±翻转形成对比），但 `Δ=+2.99 < +5` 且 CI 跨 0 → **未达任何行动线**（「hint 都算不上」的门
线以下）；按预注册口径只能报「无 ≥+10 证据」。self-play 分布采样量已到位（seed sd 仅 2.17），
故该 null 不是「牌数不够」造成的：需要更大效应或更敏感设计才能在该分布上判正。

**两个分布放一起读**：random 主端点 −5.02（seed sd 15.9）、self-play 端点 +2.99（seed sd 2.2）。
两个分布都没有 ≥+10 证据；事件表示在 self-play 上方向一致但幅度太小。

---

## 7. 机制、消融与实现注意

- **无状态重算**：训练 rollout 与 `NeuralPolicy.act` 共用 `encode_events`，参数化 parity 测试
  覆盖 `concat/sum/none` seat、`noisy`、`blind`、`shuffled+drop`（`tests/test_history.py`）。
- **墩内 shuffle 确定性**：固定置换 + identity 回退旋转，同一 `View` 两次编码逐位一致；
  保 `opens_trick` 位置与墩块。
- **相对座位**：`(seat - view.seat) % n`，h2h/duel 换座与 self-play 冻结对手（按自己的 view）
  都落在训练过的下标上。
- **性能优化**：GRU 前按 batch 最大真实长度截尾（尾部 pad 不影响任何 gather 下标），
  108→0 pad 全零输入语义不变；新增测试
  `test_event_encoder_ignores_trailing_pads_beyond_each_rows_last_event`。
- **pass/opens_trick/went_out**：pass 仅领先时可出现→永不 `opens_trick`；补牌不是事件；
  撬底扫牌不是事件；`empty_order` 不进 `View`（ADR-0014）。
- **ckpt**：`history_layout` marker 把事件臂（首层 193）与 MLP 臂（161）分开校验，
  `obs_dim` 单独不再够用；旧 v5 MLP/seq ckpt 无 event 字段时默认 `event_len=0`，仍可加载。

---

## 8. 局限与后续（不在本次范围）

1. **真递归 / BPTT / 更大规模迁移**：规划 §8.1，未做；本次只给无状态重算下界。
2. **机制臂**（shuffle/boundaryblind/nopass）未训：预注册规则只在信息端点为非 null 时投入；
   本次 main null，故未消耗预算（开关与测试已就绪）。
3. **k=3 的功效**：seed sd 主导，t-CI ±40；若将来要判定 ±10 效应，需更多 train seed
   （sd 主导时牌数不救）。
4. **a2（组合边界）不可单独归因**：事件级天然内建，规划已声明；本次不例外。
5. **event_noise 语义**：保留真实计数是有意设计（保守控制）；若需更严格的「零信息」对照，
   可另做固定全零 mask 的 constant 臂（未做）。

---

## 9. 复现命令

```bash
# 全套测试（基线 636 + 新增 44 = 680）
uv run --group train pytest -q

# random 训练（每 seed 三臂，最多 3 并发，nice -n 5；此处示例 seed 1）
.venv/bin/python -m seven523.train --seed 1 --exp-name t17event --total-timesteps 500000 \
    --event-len 108 --checkpoint-interval 8 --tensorboard False
.venv/bin/python -m seven523.train --seed 1 --exp-name t17eventnoise --total-timesteps 500000 \
    --event-len 108 --event-noisy True --checkpoint-interval 8 --tensorboard False
.venv/bin/python -m seven523.train --seed 1 --exp-name t17eventseatblind --total-timesteps 500000 \
    --event-len 108 --event-seat none --checkpoint-interval 8 --tensorboard False

# self-play（纯 self-play；--opponent self 会忽略 --mix-random-prob）
.venv/bin/python -m seven523.train --seed 1 --exp-name t17selfevent --total-timesteps 500000 \
    --event-len 108 --opponent self --self-play-refresh 10 --checkpoint-interval 8 --tensorboard False
.venv/bin/python -m seven523.train --seed 1 --exp-name t17selfeventnoise --total-timesteps 500000 \
    --event-len 108 --event-noisy True --opponent self --self-play-refresh 10 \
    --checkpoint-interval 8 --tensorboard False

# h2h（严格串行、一次一个；只允许 cpu + workers 4）
.venv/bin/python tools/head_to_head.py \
    --left event_s1=ckpt:runs/t17event__1__<ts>/agent.pt \
    --right eventnoise_s1=ckpt:runs/t17eventnoise__1__<ts>/agent.pt \
    --seeds 0,1,2 --pairs 400 --bootstrap 4000 --device cpu --workers 4 --json \
    > runs/h2h_eventvseventnoise_s1.json 2> runs/h2h_eventvseventnoise_s1.err

# 3-seed 合并（t 与 z 双口径）
.venv/bin/python runs/t17event_train/aggregate_event.py \
    h2h_eventvseventnoise h2h_eventvsseatblind h2h_eventvst17pool h2h_eventvsseq
```

---

## 10. 产物路径

- 实现：`src/seven523/game.py`、`history.py`、`networks.py`、`ppo.py`、`train.py`、`league.py`、
  `__init__.py`；测试：`tests/test_game.py`、`test_env.py`、`test_history.py`、`test_networks.py`、
  `test_ppo.py`、`test_league.py`、`test_train.py`。
- random runs：`runs/t17event__*`、`runs/t17eventnoise__*`、`runs/t17eventseatblind__*`。
- self-play runs：`runs/t17selfevent__*`、`runs/t17selfeventnoise__*`。
- h2h 原始 JSON：`runs/h2h_eventvseventnoise_s{1,2,3}.json`、`runs/h2h_eventvsseatblind_s{1,2,3}.json`、
  `runs/h2h_eventvst17pool_s{1,2,3}.json`、`runs/h2h_eventvsseq_s{1,2,3}.json`，
  self-play 同名前缀 `h2h_selfeventvseventnoise_s*.json`；聚合脚本
  `runs/t17event_train/aggregate_event.py`。
- 规划/审查修订：`docs/event-history-plan.md`（用户决策块 + 4 处 verify 修订）、
  `docs/event-history-review.md`（第 24/40 条 + §4 裁决）。

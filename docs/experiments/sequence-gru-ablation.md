# GRU 顺序消融：序列编码器里的「顺序/递归」值多少 Elo（7鬼523 / PPO）

> **状态：已完成（2026-09-26）**。前序 [`sequence-memory-pilot.md`](./sequence-memory-pilot.md) 的
> 关键端点 `seq vs seqblind`（+8.41 [−4.06,+20.89] Elo）把「有历史 token」与「GRU/编码器容量」
> 混在一起：`seqblind` 喂的是全 pad，模型根本没有历史 token。本报告新增 **order-ablation 臂**
> `--seq-order sorted`——与 `seq`（chrono）臂**同架构、同参数量、同 token 多重集、同训练预算**，
> 唯一差别是把出牌 token 按 card-id 升序重排（pad 在后），从而把 GRU 的**顺序/递归**作用
> 从 bag 信息里单独隔离出来；并按用户 5 分线（关键端点 3-seed 合并点估计 > +5 则调超参）执行了
> 条件分支。
>
> **口径**：revision-3（`rules_id=2e36dbea44893696`）、obs v5（2 家 161 维）、T17 工作区。
> 训练 500k（random、8×128、lr 2.5e-4 退火、hidden 128、shared、seq-len 54）；
> 评测 ≥3 seed × 400 副牌换座 h2h，**严格串行、`--device cpu --workers 4`**，合并公式
> `mean ± 1.96·max(平均 boot SE, seed sd)/√3`（`runs/t17seq_train/aggregate_seq.py`）。
> repo 统一行动门槛：**CI 完全排除 0 且点估计 ≥+10**；用户本轮的调优触发线：关键端点
> **3-seed 合并点估计 > +5 Elo**。

## 0. 结论摘要

1. **顺序/GRU 作用（关键端点，chrono vs sorted）**：3 训练 seed 合并
   **+10.46 [−2.46, +23.38] Elo**（左 chrono 更强；boot SE 11.04、训练 seed sd 11.42）。
   点估计跨过 +5（用户调优线）也略过 +10，但 **CI 跨 0**，达不到 repo 行动门槛。
2. **bag 信息（无顺序，sorted vs seqblind）**：3 seed 合并 **−1.99 [−24.89, +20.91] Elo**
   （boot SE 11.21、seed sd 20.24）→ 单纯「有历史 token 但无顺序」几乎没有可测收益；
   而 chrono vs 容量盲对照 = **+8.41 [−4.06,+20.89]**（前序报告，本轮同口径重算）。
   两条路径的差给出同一顺序效应：`(chrono−seqblind) − (sorted−seqblind)` = 8.41 − (−1.99)
   = **+10.40**，与直接测量 +10.46 在 0.06 Elo 内一致（近似可加的交叉验证）。
3. **P2 条件分支（因 +10.46 > +5 而触发）**：3 个单因素配置 × 3 seed 全部**没有**跑赢
   seed-matched chrono 基线：emb 16→32 = **−6.96 [−19.47,+5.55]**、
   hidden 32→64 = **−7.59 [−24.52,+9.34]**、len 54→27 = **−18.09 [−30.87,−5.32]**
   （最后一个 CI 全负，显著更差）。即：**顺序效应在点估计上存在，但当前三个超参方向都放大不了它，
   默认配置已接近 500k/random 预算下的局部最优点**。
4. **机制抽查**（60 个真实 `View`，policy/value 分开）：chrono ckpt 对**顺序置换**的 logit 敏感度
   很小（sorted 布局 mean|Δlogit| 0.07–0.14、reversed 0.11–0.18），对「有没有历史」大一个量级
   （全 pad 1.16–1.62）；value 头几乎不受历史影响（mean|Δvalue| ≤0.064）。行为上少部分决策会翻转
   （chrono ckpt 的 suit 头 argmax 在 sorted 布局下 0–20% 翻转）——与 +10 Elo 级、CI 跨 0 的
   小效应一致。
5. **判定**：**GRU/顺序作用超过用户 +5 分线（点估计 +10.46），但调优未找到放大的配置；
   repo 行动门槛（CI 排 0 且 ≥+10）仍未达到。**当前证据不足以「排期序列记忆」，
   但足以把「顺序信息」列为比「bag 信息」更值得复查的轴（见 §6 复查条件）。

## 1. 「GRU 作用」的干净分解（P0）

四类臂的输入信息：

| 臂 | 编码器/容量 | token 多重集 | token 顺序 | 相对 `t17pool` 的差 |
|---|---|---|---|---|
| `t17pool`（MLP 基线） | 无编码器 | 无 | — | — |
| `seqblind` | GRU(16→32)+加宽 trunk | 空（全 pad） | — | 无历史 token |
| `seq`（chrono） | GRU(16→32)+加宽 trunk | 真实历史多重集 | **出牌时间顺序** | 历史 + 顺序 + 容量 |
| `seqsorted`（本报告新增） | 同左、同参数量 | 同左（同一多重集） | **升序规范序（无时间信息）** | 历史多重集 + 容量（**无顺序**） |

因此两个端点各自只变一个因素：

- **`seq`（chrono）vs `seqsorted` = 顺序/递归（GRU 作用）**：唯一变化是 token 排列是否携带
  时间信息；多重集、参数量、词表、训练预算、读出方式（末真实 token hidden）完全一致。
- **`seqsorted` vs `seqblind` = bag 信息（无顺序的历史）**：唯一变化是有没有真实 token。
- 两者之和不严格等于 `seq vs seqblind`（非线性交互），但可作交叉校验。

**这测得的是什么**：GRU 能否把「出牌先后/近因/同墩邻接」转化为策略收益（前序报告 §1.2 的 a1–a3）。
**测不到什么**：

1. **逐座位归属与墩边界不在 `View` 里**（`GameState.played` 只存牌、不存人/墩界，
   `src/seven523/game.py:41-45`），任何基于 `View` 的序列编码器都拿不到 a4–a6（对手倾向/后验）；
2. `sorted` 臂保留了完整 bag，所以它测的是「顺序在 bag 之上的增量」，不是「序列的全部信息」；
3. random 500k 对手可取样的规律弱，顺序信号主要来自记牌/近因的一次性利用；
4. 单架构单序列长度（GRU-32 / emb-16 / 全量 54 / 末 hidden），不是「所有递归结构」的普遍结论；
5. 长度 54 时 `sorted` 的排序是「全量历史升序」；截断（len<54）在排序前完成，保证两臂看到
   同一「最近 N 张」多重集。

## 2. 实现（opt-in，默认路径逐位不变）

| 文件 | 改动 |
|---|---|
| `src/seven523/history.py` | 新增 `ORDERINGS=("chrono","sorted")`；`encode_history(view, length, order="chrono")` 增加 `order` 参数（sorted = 对截断后的 token 升序重排、pad 保持尾部；未知 order 抛 `ValueError`）；`HistorySequenceWrapper(env, length, order)` 透传 |
| `src/seven523/networks.py` | `Agent(..., seq_order="chrono")` 记录布局；`save_agent` 存 `seq_order`；`load_agent` 恢复（旧 ckpt 缺字段默认 `"chrono"`）；`NeuralPolicy.act` 用 `agent.seq_order` 重建序列，训练 rollout 与评估共用同一函数 |
| `src/seven523/train.py` | CLI `--seq-order {chrono,sorted}`（默认 `chrono`）；`make_env` 透传到 wrapper；Agent 构造透传；`args.json` 记录 |
| `tests/test_history.py` | 新增 5 项：默认 chrono 回归 + 未知 order 报错、sorted 与 chrono 同多重集/顺序确实不同/pad 在尾/截断同多重集、sorted 策略与 rollout wrapper 逐位一致、sorted ckpt 往返、`--seq-order sorted` 端到端 smoke |

- **默认逐位不变**：`seq_order="chrono"` 与旧代码走同一分支；`seq_len=0` 仍不建编码器
  （`test_networks` 的 55,179 参数计数断言不变）；旧 ckpt 无 `seq_order` 字段时
  `load_agent`/`NeuralPolicy` 默认 chrono。
- **全量测试**：`636 passed`（前序基线 631 + 新增 5），日志 `runs/t17seq_train/pytest_p3.log`。
- **训练侧/评估侧一致性**：`NeuralPolicy.act` 从 `agent.seq_order` 读取布局，
  h2h `load_agent` 路径已用 sorted ckpt 实测（`loaded sorted ckpt: seq_order=sorted`）。

## 3. 结果

### 3.1 训练健康度

12 个 500k run（3 sorted + 9 调优）全部 **499,712 步（488 updates）、无 NaN**；
末行 ep_return/entropy 与同 seed `t17seq` 同带（0.43–0.58 / 1.58–1.76），
日志 `runs/t17seq_train/p3_*.log`、`runs/t17seq_train/tune_*.log`。sorted 臂的
`args.json` 均记录 `"seq_order": "sorted"`，ckpt 往返恢复正确。

### 3.2 h2h 原始表（本报告新增，3×400 换座、bootstrap 4000，CPU workers=4 串行）

正 = 左臂更强；每行一个训练 seed；原始 JSON `runs/h2h_*.json`，合并 `runs/t17seq_train/merge_p3.txt`。

| 比较 | seed 1 | seed 2 | seed 3 | 3 seed 合并 | boot SE / seed sd |
|---|---|---|---|---|---|
| **chrono vs sorted**（关键端点：顺序/GRU 作用） | +12.62 [−4.52,+29.76] | −1.88 [−13.81,+10.04] | +20.65 [−4.97,+46.27] | **+10.46 [−2.46,+23.38]** | 11.04 / 11.42 |
| sorted vs seqblind（bag 信息，无顺序） | +0.15 [−12.33,+12.62] | +17.10 [+4.63,+29.57] | −23.20 [−36.33,−10.08] | **−1.99 [−24.89,+20.91]** | 11.21 / 20.24 |
| chrono vs seqblind（前序，本轮同口径重算） | +5.22 [−7.27,+17.70] | +14.35 [+1.97,+26.73] | +5.67 [−19.65,+31.00] | +8.41 [−4.06,+20.89] | 11.02 / 5.15 |
| chrono vs t17pool（参考） | +21.21 [−3.23,+45.65] | +1.16 [−11.33,+13.64] | +14.95 [−6.31,+36.21] | +12.44 [−0.07,+24.94] | 11.05 / 10.26 |
| seqblind vs t17pool（容量对照，参考） | +10.61 [−15.86,+37.08] | −20.08 [−47.52,+7.36] | +17.98 [+5.54,+30.41] | +2.84 [−20.00,+25.68] | 10.99 / 20.18 |

读法与交叉验证：

- **顺序效应点估计 +10.46**：seed 1/3 为正（+12.62/+20.65）、seed 2 为 −1.88；训练 seed sd 11.42
  比 bootstrap SE 略大，合并 CI [−2.46,+23.38] 跨 0。
- **bag 效应点估计 −1.99**：sorted 臂与容量盲对照在 3-seed 尺度上不可分；注意
  sorted vs seqblind 的 seed sd 高达 20.24（排序臂在不同 seed 上学到的函数差异很大）。
- **两条分解路径一致性**：`chrono−seqblind` − `sorted−seqblind` = 8.41 − (−1.99) = **+10.40**，
  与直接测量的 chrono−sorted = **+10.46** 相差 0.06 Elo。点估计上「顺序」贡献 ~+10、
  「bag」贡献 ~0，这一结构在两个独立端点复现。
- **与 repo 门槛的对照**：关键端点 CI 跨 0（下界 −2.46）、点估计 +10.46 刚过 +10，
  按统一规则（CI 排 0 且 ≥+10）**不构成行动依据**；按用户 +5 点估计线**触发调优**。

### 3.3 P2 条件分支：超参调优（3 配置 × 3 seed，预算 12 run 用满）

配置为围绕默认 `emb16/hidden32/len54` 的**单因素扫描**（全部是已有 CLI 开关，零新代码）：
`E32 = --seq-emb 32`、`H64 = --seq-hidden 64`、`L27 = --seq-len 27`；均为 chrono 布局、
其余镜像 `t17seq`。评测：每 seed 的 tuned 臂 vs **同 seed** `t17seq` chrono 基线
（正 = tuned 更好）。原始 JSON `runs/h2h_{e32,h64,l27}_s{1,2,3}.json`、合并
`runs/t17seq_train/merge_p3_tune.txt`、命令 `runs/t17seq_train/commands_p3_tune.txt`。

| 配置 | seed 1 | seed 2 | seed 3 | 3 seed 合并 | boot SE / seed sd |
|---|---|---|---|---|---|
| E32（emb 32） | −1.45 [−14.50,+11.60] | −6.53 [−25.31,+12.25] | −12.89 [−25.60,−0.18] | **−6.96 [−19.47,+5.55]** | 11.06 / 5.73 |
| H64（hidden 64） | +9.43 [−12.01,+30.86] | −18.69 [−31.20,−6.18] | −13.50 [−39.96,+12.95] | **−7.59 [−24.52,+9.34]** | 11.06 / 14.96 |
| L27（len 27） | −20.44 [−33.43,−7.45] | −26.02 [−50.73,−1.31] | −7.82 [−21.70,+6.05] | **−18.09 [−30.87,−5.32]** | 11.29 / 9.32 |

判定：

- **没有任何配置在关键端点上更好**；H64 的 seed 1（+9.43）是唯一正点估计，但其余两个 seed
  为明显负值，合并 −7.59。L27 的合并 CI **全负**：把历史截到最近 27 张显著更差，
  说明本游戏里较远的历史牌（结合 `unseen` 的剩余推断）仍有价值，全量 54 是更好的默认。
- 触发线只在「默认 chrono 臂的顺序效应」上过线；**放大顺序效应所需的方向不在
  emb/hidden/长度这三个轴上**（或 500k/random 的噪声下不可辨）。按任务书，调优到此为止，
  不扩展 LSTM/双向（那需要新代码与新单测，且收益先验更低）。

### 3.4 机制抽查（policy 头 vs value 头，`runs/t17seq_train/diagnose_order.py`）

60 个真实 `View`（random 对手），把同一历史分别编码为 **训练布局（own）**、另一顺序布局、
reversed（仅真实 token 反转、pad 仍在尾）、全 pad，比较平均 |logit 差|、masked-argmax
一致率（head0=牌型模板、head1=顶牌花色）与 |value 差|。原始输出
`runs/t17seq_train/diagnose_order.txt`。

chrono 训练的 3 个 ckpt（vs own=chrono 布局）：

| 扰动布局 | mean\|Δlogit\|（3 seed） | mean\|Δvalue\| | argmax 一致率 head0 / head1 |
|---|---|---|---|
| sorted | 0.07 / 0.08 / 0.14 | 0.006–0.010 | 1.000 / 1.000、0.800、0.883 |
| reversed | 0.11 / 0.13 / 0.18 | 0.007–0.011 | 1.000 / 0.967、0.850、0.900 |
| 全 pad | 1.16 / 1.37 / 1.62 | 0.036–0.051 | 1.000 / 0.267、0.567、0.767 |

sorted 训练的 3 个 ckpt（vs own=sorted 布局）：

| 扰动布局 | mean\|Δlogit\|（3 seed） | mean\|Δvalue\| | argmax 一致率 head0 / head1 |
|---|---|---|---|
| chrono | 0.46 / 0.48 / 0.86 | 0.019–0.024 | 1.000 / 0.917、0.850、0.850 |
| reversed | 0.27 / 0.37 / 0.84 | 0.015–0.022 | 1.000 / 0.967、0.817、0.867 |
| 全 pad | 1.49 / 1.50 / 1.62 | 0.033–0.064 | 1.000 / 0.600、0.567、0.650 |

机制读法：

1. **编码器确实被使用且主要编码「有没有历史」**：全 pad 扰动的 logit 差（1.2–1.6）
   比顺序置换大一个数量级，与前序 `diagnose_seq` 的 1.62/0.17 结构一致。
2. **chrono 模型对顺序的局部敏感度很小**（sorted/reversed 0.07–0.18 logit），
   但行为上并非零：suit 头 argmax 在 0–20% 的采样决策上翻转；value 头几乎不变
   （≤0.011）——历史信号基本走 policy 头。
3. **sorted 模型反而对顺序更敏感**（0.27–0.86）：其训练分布要求它从规范序中读出多重集，
   任何置换都是分布外扰动，所以这个数字混合了「顺序」与「分布偏移」，不能直接当作
   顺序利用更强。h2h 上 sorted 臂在关键端点落后（点估计），说明这种敏感性没有转化为强度。
4. 机制与行为量级自洽：logit 层的顺序扰动小 → Elo 级顺序效应只是 ~+10 点估计且 CI 跨 0，
   而不是「GRU 完全无视顺序」也非「顺序贡献很大」。

## 4. 局限（如实）

1. **测的是下界**：`sorted` 臂仍有完整 bag；a4–a6（逐座位归属/倾向/后验）不在 `View` 里
   （`game.py:41-45`），本实验无法覆盖。真正的顺序价值上界可能更高，但需要引擎记 `(seat, card)`
   与 v6 布局（前序报告 §6 的判断线）。
2. **训练 seed 噪声主导**：关键端点 seed sd 11.42 > boot SE 11.04；3 seed 合并 CI 半宽 ±12.9，
   对 10 Elo 级效应只有 ~40% 左右功效。`>+5` 的触发线是点估计级，不是显著性级。
3. **分布单一**：全部为 random 500k；self-play/更强对手下顺序/倾向价值可能不同
   （前序 P1 的 B1 盲消融在 self-play 有 +9.85 的不显著正 hint）。
4. **调优只扫了 3 个单因素**且只测「能否跑赢 chrono 基线」，没有为每个配置训练匹配的
   sorted 臂，因此无法测「哪个配置的顺序效应更大」；这一问要再花 9 个 run（预算不允许）。
5. **机制抽查规模小**：60 个 view、greedy argmax、固定 random 对局轨迹；不是全决策分布的无偏估计。
6. **2 家结构**：`last_player` 在 2 家冗余；N 家下的座位轮转/过牌序列价值未测。

## 5. 复现命令与产物

```bash
# 训练 sorted 臂（3 seed，最多 3 进程并发）
bash runs/t17seq_train/commands_p3.txt        # 或逐行执行，产物 runs/t17seqsorted__{1,2,3}__1790447239/

# 评测：严格串行、CPU workers=4（每行 ~33s）
.venv/bin/python tools/head_to_head.py \
  --left seq_s1=ckpt:runs/t17seq__1__1790445087/agent.pt \
  --right sorted_s1=ckpt:runs/t17seqsorted__1__1790447239/agent.pt \
  --seeds 0,1,2 --pairs 400 --bootstrap 4000 --device cpu --workers 4 --json \
  > runs/h2h_seqvssorted_s1.json 2> runs/h2h_seqvssorted_s1.err
# 同理 h2h_sortedvsseqblind_s*；合并：
.venv/bin/python runs/t17seq_train/aggregate_seq.py h2h_seqvssorted h2h_sortedvsseqblind \
    h2h_seqvsblind h2h_seq h2h_seqblind | tee runs/t17seq_train/merge_p3.txt

# 调优（仅因关键端点 +10.46 > +5 触发）；命令模板见 commands_p3_tune.txt，
# 评测 runs/h2h_{e32,h64,l27}_s*.json，合并见 merge_p3_tune.txt

# 机制抽查
.venv/bin/python runs/t17seq_train/diagnose_order.py | tee runs/t17seq_train/diagnose_order.txt

# 全量测试（默认路径回归）
uv run --group train pytest -q                # 636 passed
```

产物一览：

| 产物 | 路径 |
|---|---|
| sorted 训练 3 臂（499,712 步，无 NaN） | `runs/t17seqsorted__{1,2,3}__1790447239/` |
| 关键端点 h2h JSON ×3 | `runs/h2h_seqvssorted_s{1,2,3}.json` |
| bag 端点 h2h JSON ×3 | `runs/h2h_sortedvsseqblind_s{1,2,3}.json` |
| 合并（含前序端点重算） | `runs/t17seq_train/merge_p3.txt` |
| 调优 9 run + 命令 | `runs/t17seq_{e32,h64,l27}__{1,2,3}__17904{47931,48375,48821}/`、`runs/t17seq_train/commands_p3_tune.txt` |
| 调优 h2h JSON ×9 + 合并 | `runs/h2h_{e32,h64,l27}_s{1,2,3}.json`、`runs/t17seq_train/merge_p3_tune.txt` |
| 机制抽查脚本/输出 | `runs/t17seq_train/diagnose_order.{py,txt}` |
| 全量测试日志 | `runs/t17seq_train/pytest_p3.log`（636 passed） |

## 6. 结论与复查条件

- 就本任务定义：**GRU/顺序作用的关键端点 3-seed 合并点估计 +10.46 Elo（> 用户 +5 线），
  但 CI [−2.46,+23.38] 跨 0、未达 repo「CI 排 0 且 ≥+10」行动门槛；bag（无顺序）作用 ≈0；
  三个超参方向（emb↑/hidden↑/len↓）全部未能放大，其中 len→27 显著更差。**
- **是否继续调优**：按任务书的分支逻辑，超参调优已执行且失败；不启动 LSTM/双向等新架构
  （新代码 + 先验更低，且触发线本身不是显著性线）。
- **若将来要复查**（满足任一才值得再花预算）：
  1. 训练 5 seed × 400 以上把关键端点 CI 收到 ±10 内（当前瓶颈是训练 seed sd，不是牌数）；
  2. 换分布：self-play/更强对手下重做 `chrono vs sorted`（前序 P1 的 B1 正 hint 在 self-play）；
  3. 若做 a4–a6，必须先立 ADR 决定 v6（逐座位 `(seat, card)` + 墩边界），否则顺序轴已测到上限。

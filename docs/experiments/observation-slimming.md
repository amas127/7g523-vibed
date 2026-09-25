# 观测精简：2 家 191 维输入的信息审计、敏感性消融与从零 pilot

> 资产状态（2026-09-25）：本报告引用的模型 checkpoint 已删除，旧路径不再可用；数值为牌型族规则变更前口径。

> **状态：研究完成（2026-09-25）。本文只做研究与验证：未改 `src/`、未 commit、未动任何已有报告与
> `runs/` 里的既有产物。** 临时脚本与原始输出复制在 `runs/obs-slim/`（gitignore）：
> `obs_audit_static.py`（45,752 个 acting-view 的稀疏度与推导关系）、`obs_ablation.py`（ckpt 逐段
> 置零/打乱）、`obs_winrate.py`（800 副牌配对胜率消融）、`obs_slim_encoder.py` + `obs_pilot_train.py`
> （精简布局的从零训练，monkeypatch 不改 `src/`）、`slim_h2h.py`（允许两侧不同布局的候选对候选）。
>
> 定位：姊妹报告 [`structural-directions.md`](./structural-directions.md) 研究**观测增广**（补已出牌
> 历史/当前墩分，191→194/249，§1.2/§3）；本文研究相反方向——**观测精简**（去冗余、压缩稀疏段）。
> 两者互补：精简腾出的首层输入预算可以直接给增广用。评估口径与分辨率沿用
> [`wave5-500k-report.md`](./wave5-500k-report.md) §4.2 和 [`head-to-head-pilot.md`](./head-to-head-pilot.md)
> （候选对候选同牌换座，400 副牌 ≈ ±20 Elo；10 Elo 需 1500–2000 副牌）。
>
> 一句话结论：**191 维里 88 维是冗余或低效表达，可以用信息无损的方式压到 103 维（S1）；再激进压到
> 68 维（S2）会丢掉手牌"点数↔花色"的配对信息。** 敏感性消融显示现有 ckpt 对任意单段置零的胜率影响
> 都在 1 分/200（不可分辨）以内——这**不能**证明被删段无用（该 ckpt 对输入本身不敏感），只能说明
> "删哪段"要靠信息论证兜底。pilot（200k × 3 seed，同牌换座 h2h）验证：**S1 与 full 不可分辨
> （+4.4 [−18.3,+27.0] Elo），S2 落后 full 约 14.6 Elo（CI 刚排除 0）**；vs-greedy 的绝对口径与
> h2h 同号但更粗，两者差异已复核（§5.4）。

## 0. 关键数字（先看这里）

| 量 | 数值 | 来源 |
|---|---|---|
| 当前 2 家观测 | **191** = `185 + 3n`（10 段） | `env.py:105-125`；`tests/test_env.py:14-21` |
| 逐段实测非零（45,752 状态） | hand **5.49/54**、incumbent_top **0.73/54**、revealed **2.00/54**、current **1.00/2** | §2.2；`obs_audit_static.py` |
| `rank_counts` = hand 按点数求和 /4 | **0/45,752 不一致**（完全可推导） | §2.3(a) |
| `current` = `view.seat` | **0/45,752 不一致**（发布路径恒为 acting seat） | §2.3(b) |
| `hand_counts[self]` = `len(hand)/7` | **0/45,752 不一致** | §2.3(c) |
| incumbent 可由 (hand, mask) 反推 | 41,763 个 (hand,mask) 组中 96.4% 唯一，**3.6% 歧义**（最大 31 个不同 incumbent） | §2.3(d) |
| revealed 陈旧率 | **91.0%** 的检查里该亮牌已被其主人打出（仍在手 4.9%/4.1%） | §2.3(e) |
| **S1 精简布局** | **191 → 103**（省 88 维；保留全部决策信息） | §3.1 |
| **S2 激进布局** | **191 → 68**（手牌降为 rank15+suit4，丢配对） | §3.2 |
| 参数与首层 | 59,019 → 47,755 (−19.1%) / 43,275 (−26.7%)；首层 24,576 → 13,312 / 8,832 | §3.4 |
| 敏感性排名（4 ckpt） | hand ≫ revealed ≈ hand_counts ≳ incumbent_kind > current ≈ incumbent_top ≈ rank_counts > draw > scores(仅价值) > incumbent_size | §4.2 |
| 胜率消融（6 个 ckpt，800 副牌配对） | 所有单段置零 / S1 组合置零的 Δ 都 ≤1.7 分/200，95% CI 全部覆盖 0 | §4.3 |
| 从零 pilot（full vs S1 vs S2） | 3 臂 × 3 seed × **200k**：S1−full **+4.4 [−18.3, +27.0]**，S2−full **−14.6 [−24.0, −5.3]** | §5 |

---

## 1. 当前布局与逐段清单

单一真源是 `env.py:105-116` 的 `_SEGMENTS`；`observation_dim`（`env.py:123-125`）与
`encode_observation`（`env.py:128-135`）都从它求和/游走，因此段与维度不会漂移。2 家时偏移如下：

| 段 | 字节区间 | 维度 | 写入函数 | 内容 | 公开性（`View`，`game.py:80-93`） |
|---|---|---|---|---|---|
| `hand` | [0, 54) | 54 | `_write_hand:54` | 自己手牌 card_id multi-hot | 私有（自己） |
| `rank_counts` | [54, 69) | 15 | `_write_rank_counts:59` | 自己每点数计数 /4 | 私有（自己） |
| `incumbent_top` | [69, 123) | 54 | `_write_incumbent_top:65` | 领出牌顶牌 multi-hot | 公开 |
| `incumbent_kind` | [123, 129) | 6 | `_write_incumbent_kind:70` | `ComboKind` one-hot | 公开 |
| `incumbent_size` | [129, 130) | 1 | `_write_incumbent_size:75` | 张数 /7 | 公开 |
| `draw_count` | [130, 131) | 1 | `_write_draw_count:80` | 底牌堆剩余 /54 | 公开 |
| `scores` | [131, 133) | n=2 | `_write_scores:84` | 各家分数 /100（绝对座位序） | 公开 |
| `hand_counts` | [133, 135) | n=2 | `_write_hand_counts:89` | 各家手牌数 /7（绝对座位序） | 公开 |
| `current` | [135, 137) | n=2 | `_write_current:94` | 当前座位 one-hot（绝对座位序） | 公开 |
| `revealed` | [137, 191) | 54 | `_write_revealed:98` | 开局亮牌 multi-hot（每座一张） | 公开 |

`View` 里另有 `trick_cards` 与 `last_player`（`game.py:91-92`）没有被任何段编码——这是
[`structural-directions.md`](./structural-directions.md) §1.2/§3 的增广缺口，本文不重复。
`mask` 是**独立输入**（`env.py:225`，`networks.py:234-241` 经 `joint_mask_bits:actions.py:107`），
观测不需要重复表达"哪些动作合法"，只需要保留"比较与估值"仍需要的事实。

## 2. 冗余与信息含量审计

### 2.1 方法与数据

`obs_audit_static.py` 用 `GreedyBot` vs `GreedyBot` 与 `RandomBot` vs `RandomBot` 各 400 局
（seed 12345 / 999）收集**每个 acting seat 的 `View` + `encode_observation`**，共 **45,752 个状态**；
再用 `GameState` 侧信道统计 `revealed` 的真实持牌/已出牌比例（400 局）。稀疏度、取值分布、
可推导关系如下。另有一份 ckpt-vs-GreedyBot 的 19,126 个状态用于 §4 消融。

### 2.2 逐段稀疏度与取值

| 段 | 维度 | 非零维/状态（均值 / 最大） | 全零状态 | 观测到的不同向量数 | 说明 |
|---|---|---|---|---|---|
| `hand` | 54 | 5.49 / 7（10.2% 维非零） | 0% | 38,406 | 手牌 ≤7 张 |
| `rank_counts` | 15 | 4.58 / 7 | 0% | 26,454 | 计数 0–4 |
| `incumbent_top` | 54 | 0.73 / 1 | **26.5%** | 55 | 领出时全零 |
| `incumbent_kind` | 6 | 0.73 / 1 | 26.5% | 6 | |
| `incumbent_size` | 1 | 0.735（73.5% 非零） | 26.5% | 7 | 1–7 |
| `draw_count` | 1 | 0.786 | 21.4% | 41 | 0–40 |
| `scores` | 2 | 1.45 / 2 | 13.3% | 231 | 中局 own+other < 100（94.7%） |
| `hand_counts` | 2 | 1.97 / 2 | 0% | 63 | |
| `current` | 2 | 1.00 / 1 | 0% | **2** | 恒为 acting seat |
| `revealed` | 54 | **2.00 / 2** | 0% | 181 | 每座一张，整局不变 |

### 2.3 可推导性与独立信息（逐条证据）

**(a) `rank_counts` 完全可由 `hand` 推导（15 维 → 0）。**
`_write_hand` 对 `view.hand` 写 card_id multi-hot（`env.py:54-56`），`_write_rank_counts` 对**同一个**
`view.hand` 做 `Counter(card.rank)/4`（`env.py:59-61`）；`card_id` 是点数×花色的全序索引
（`cards.py:147`，`CARD_ORDER` 由 `STANDARD_RANKS × Suit` 构成）。逐状态验证：**0/45,752 不一致**。
（`rank_counts` 的信息量 = 138,610 个可行计数向量的 log2 ≈ 17.1 bit，但它与 `hand` 共享，独立信息 = 0。）

**(b) `current` 是常数列（2 维 → 0）。**
在**所有生产路径**里，被编码的 `View` 都是"轮到谁就是谁"：`Match.step` 先取
`seat = state.current` 再 `view(state, seat)`（`match.py:94-96`）；`Seven523Env._publish` 在
`advance()` 停到 learner 之后只投影 `view(self.learner)`（`env.py:221-226`）；`NeuralPolicy.act`
只会拿到正在行动座位的 `View`（`networks.py:230-245`）。逐状态验证：`argmax(current) == view.seat`
**0/45,752 不一致**。`current` 在完整布局里的实际作用是给"绝对座位序"的 `scores`/`hand_counts`
做座位提示；把这两段改成**自中心旋转**（自己在前）后它没有任何独立信息。

**(c) `hand_counts` 自己那一维 = `len(hand)/7`（n 维 → n−1 维）。**
`View.counts[seat] = len(state.hands[seat])`（`game.py:147`），而 `View.hand = state.hands[seat]`
（`game.py:138-139`）。验证 **0/45,752 不一致**。对手手牌数不可由自己信息推出，必须保留。

**(d) `incumbent_top` 的 54 维 one-hot 低效，但并非冗余表达。**
顶牌是公开的 54 张牌之一（≤1 个非零），log2(54) ≈ 5.75 bit，但 54 维 one-hot 是 54 维。
"能否用 `hand`+`mask` 省掉 incumbent 段"的实测：把 45,752 个状态按 `(hand 位集, mask)` 分组，
41,763 个组里 **96.4% 只有唯一 incumbent**，但 **1,501 组（3.6%）有歧义**，最大一组有 **31 个不同
incumbent**——因为 `mask` 只覆盖模板合法性，值函数与细粒度比较仍需要精确的 incumbent。
所以 incumbent 段必须保留，但可无损压成 **rank one-hot 15 + suit one-hot 4 = 19 维**（王无花色时
suit 位全零，由 rank 判别；`kind`/`size` 另算）。`incumbent_kind`（6）与 `incumbent_size`（1）不
能互相决定（顺子 size 3–7、小炸弹 size 2/3、王炸 size 2），各自保留。

**(e) `revealed` 是"整局恒定的 54 维两非零"，且 91% 已陈旧。**
`_deal` 把每座手牌里 `card_key` 最小的一张记为亮牌（`game.py:116`），**不拿走**，整局不再更新。
实测（400 局、按"每座每回合一次"计）：自己的亮牌仍在手 4.9%、对手的 4.1%、**已被主人打出 91.0%**，
正在桌面（incumbent）1.7%。观测没有已出牌历史（structural §1.2），所以这个"对手有 X"的提示在
大部分决策点已经过期；它仍有一点价值（对手在首轮前确实持有 X 的先验 + 自己的开局牌），但没有理由
用 54 维。可无损保留"对手亮牌"的牌身份：**每对手 rank15+suit4 = 19 维**（2 家共 19）。自己的亮牌
在手里时可从 `hand` 推出，打掉后是局面历史——与 structural 建议的"已出牌历史"重叠，单独保留
19 维的意义很小。

**(f) `scores`/`draw_count`/`hand` 没有可省维。**
中局 94.7% 状态 `own+other < 100`——缺少当前墩分（`View.trick_cards` 未编码），所以分数不能用
"总和 100" 塌缩成一维；`draw_count` 是隐藏顺序的唯一线索；`hand` 是私有状态的全部。它们保留。

**逐段结论表（2 家）：**

| 段 | 当前 | 独立信息 | 可推导性 | 有效维度下界 | 处置 |
|---|---|---|---|---|---|
| `hand` | 54 | 是 | — | 54（精确 one-hot；集和 27.6 bit） | 保留 54 |
| `rank_counts` | 15 | 否 | 完全 ⊆ `hand` | **0** | **删** |
| `incumbent_top` | 54 | 是 | mask 只能 96.4% 反推 | 19（rank+suit；5.75 bit） | 压缩 |
| `incumbent_kind` | 6 | 是 | kind ⇏ size | 6 | 保留 |
| `incumbent_size` | 1 | 是 | size ⇏ kind | 1 | 保留 |
| `draw_count` | 1 | 是 | — | 1 | 保留 |
| `scores` | 2 | 是 | 缺墩分，不可塌缩 | 2 | 保留 |
| `hand_counts` | 2 | 对手是 | 自己 = len(hand) | **1** | **删自己维** |
| `current` | 2 | 否（恒等于 seat） | = view.seat | **0** | **删**（配自中心旋转） |
| `revealed` | 54 | 仅对手那张 | 自己在手里可推 | **19**/对手 | 压缩，只留对手 |

合计：54+19+6+1+1+2+1+19 = **103**（S1）。

## 3. 精简编码方案

### 3.1 S1（"b"）：103 维，保留全部决策信息（推荐）

| 段 | 维度 | 编码 | 与旧布局的关系 |
|---|---|---|---|
| `hand` | 54 | card_id multi-hot（不变） | 不变 |
| `inc_rank` | 15 | 顶牌 `RANK_INDEX` one-hot | `incumbent_top` 的 rank 部分 |
| `inc_suit` | 4 | 顶牌花色 one-hot（王全零） | `incumbent_top` 的 suit 部分（无损拆出） |
| `inc_kind` | 6 | one-hot | 不变 |
| `inc_size` | 1 | /hand_size | 不变 |
| `draw` | 1 | /54 | 不变 |
| `scores` | n=2 | 自己 + 对手（自中心旋转） | 只换顺序，信息不变 |
| `opp_count` | n−1=1 | 对手手牌数 /7 | 删掉自己那一维 |
| `opp_revealed` | 19(n−1)=19 | 对手亮牌 rank15+suit4（自中心旋转） | `revealed` 中属于对手的那张；自己的那张（在手里可推）删掉 |
| `current` | — | 不编码 | acting seat 由自中心旋转隐含 |

**每一步丢掉的到底是什么：**

1. `rank_counts`（15）：只是 `hand` 的按点数聚合，网络学一次求和即可；`hand` 与它同时存在时，
   两者是同一信息的两种粒度。
2. `current`（2）：常量。旧布局需要它是因为 `scores`/`hand_counts` 按绝对座位序写；S1 旋转到
   自己在前后不再需要。**注意**：只删 `current` 而**不**旋转会把座位语义弄丢，这不是 S1 的做法。
3. `hand_counts[self]`（1）：`len(hand)/7`。
4. `incumbent_top` 54→19：一张牌的 54 维 one-hot → 点数 15 + 花色 4（王=花色全零、rank 判王），
   精确无损。
5. `revealed` 54→19：每座 ≤1 张、整局不变；保留对手那张的 rank+suit。丢掉的是**自己**的开局亮牌
   位（在手里时可推、打掉后属于历史；且无已出牌历史时"自己曾持有哪张"无法被网络用于局面判断）。

**严格无损版 B+ = 122 维**：S1 + 自己的亮牌 19 维。只有在"必须逐位复现旧 `encode_observation`"
时才需要；对决策信息没有必要（见上）。

### 3.2 S2（"c"）：68 维，手牌不再保留点数↔花色配对（激进）

| 段 | 维度 | 编码 | 丢什么 |
|---|---|---|---|
| `hand_rank_counts` | 15 | 每点数计数 /4 | 手牌的花色位置 |
| `hand_suit_counts` | 4 | 每花色张数 /hand_size | rank↔suit 配对（**关键**） |
| 其余 | 49 | 同 S1（inc 26 + draw 1 + scores 2 + opp_count 1 + opp_revealed 19） | |

S2 的决策信息损失：suit head 需要知道"某个点数上有哪些花色"。count/suit 两个边缘分布无法回答
"我手上有没有 ♠7"（例如 ♦7♦7+♠7 与 ♦7♦7♦7 在 S2 下完全等价，但前者能打花色头 ♠、后者不能）。
牌张比较在 actions.py 里也只让**顶牌**的花色起作用（`actions.py:_resolve_indexed`），所以这仍是
真实信息。pilot（§5）把 68 维臂与 191/103 一起训：**200k 下 S2 落后 full 约 15 Elo（h2h 合并
+14.6 [+5.3,+24.0]）**，因此不作为推荐实现。

### 3.3 没有采用的更激进编码

- **手牌用 7 个标量槽（card_id/54，空槽 0）= 7 维**，逐位保留全部信息，但把 one-hot 查找交给
  128 隐层去学；没有训练证据前不建议。
- **incumbent/revealed 用单个归一化 card_id 标量**同理；可省更多维，但本文的消融无法回答
  "训练能不能学会"，且这两种编码破坏 one-hot 的线性可分性。
- **把 revealed 整段删掉**：消融里 `revealed::zero` 的胜率 Δ = −0.26±1.12（base）/ +0.95±1.70
  （w5_ctrl）都不可分辨，但它的信息源单一（对手开局牌），S1 只花 19 维，保留更稳妥。

### 3.4 参数与首层预算

| 布局 | obs_dim | 首层参数 | 全部参数 | 相对 191 |
|---|---|---|---|---|
| A 现状 | 191 | 24,576 | 59,019 | — |
| D（S1+rank_counts，可选臂，**未跑 pilot**） | 118 | 15,232 | 49,675 | −15.8% |
| S1 | 103 | 13,312 | 47,755 | **−19.1%** |
| S2 | 68 | 8,832 | 43,275 | **−26.7%** |

与增广的配合：structural 的 B0/B1 预计 +3/+58 维。S1 省 88 维后，即使同时加上 B0+B1
（103+58=161）仍比现在的 191 小 30 维；单独加 B0 则只有 106 维。

## 4. 敏感性消融

### 4.1 方法

- 状态：`obs_ablation.py` 让同一个 ckpt 分别在座位 0/1 对 `GreedyBot` 下 400 局，收集**行动座位**
  的约 19,126 个状态（两个座位相位大致各半）。
- 扰动：对每个段做 (i) **置零**、(ii) **整段行置换**（打散与其余段的关联、保留段内联合分布）。
  组合条件：`incumbent_all`（top+kind+size）、`hand::zero_keep_rank_counts`、
  `proposal_drop`（rank_counts+current+own count）。
- 指标：模板头 / 花色头的 top-1 一致率、masked KL（两头的和）、`|ΔV|`。
- 胜率：`obs_winrate.py` 用"置零后的完整 191 模型"（= 精简布局在保留维上的精确函数，等价于把被删
  维固定为 0）对 `GreedyBot` 打 **800 副牌 × 换座**，逐副牌配对，报合并差值的 95% CI。

### 4.2 批推理敏感性（masked KL；花色一致率）

行按 4 个 ckpt 的平均重要性粗排（KL 越大/一致率越低 = 现有策略越依赖该段）：

| 段（置零） | base680k KL / suit | w5_ctrl | w5_scratch | w5_vf1 | 一致率范围 |
|---|---|---|---|---|---|
| `hand` | 727 / 0.757 | 915 / 0.721 | 293 / 0.833 | 639 / 0.903 | 0.72–0.90 |
| `revealed` | 253 / 0.834 | 266 / 0.820 | 76 / 0.844 | 162 / 0.935 | 0.82–0.94 |
| `hand_counts` | 259 / 0.920 | 145 / 0.909 | 161 / 0.947 | 265 / 0.930 | 0.91–0.95 |
| `incumbent_kind` | 101 / 0.890 | 269 / 0.804 | 48 / 0.920 | 103 / 0.947 | 0.80–0.95 |
| `current` | 66 / 0.956 | 56 / 0.933 | 53 / 0.967 | 104 / 0.960 | 0.93–0.97 |
| `incumbent_top` | 39 / 0.936 | 64 / 0.925 | 10 / 0.959 | 42 / 0.968 | 0.93–0.97 |
| `rank_counts` | 51 / 0.961 | 57 / 0.932 | 21 / 0.971 | 48 / 0.975 | 0.93–0.98 |
| `draw_count` | 13 / 0.986 | 10 / 0.970 | 7 / 0.980 | 16 / 0.983 | 0.97–0.99 |
| `scores` | 9 / 0.977 | 10 / 0.965 | 12 / 0.933 | 31 / 0.967 | 0.93–0.98；ΔV 绝对值最大（0.18–0.30） |
| `incumbent_size` | 1.9 / 0.997 | 1.4 / 0.986 | 1.3 / 0.983 | 0.5 / 0.997 | ≥0.98，几乎不依赖 |
| `incumbent_all` | 184 / 0.872 | 384 / 0.795 | 95 / 0.882 | 174 / 0.936 | — |

读法：**模板头的 top-1 一致率在所有条件下都 ≥97%**，几乎不随观测变化——`action_mask` 已经把
模板头压得只剩很窄的选择，当前策略的模板决策大多由合法集本身决定（§4.4 的 probe 佐证）；
真正对观测敏感的是**花色头**（hand 置零时掉到 0.72–0.90）与**价值头**（scores 置零时 |ΔV| 最大）。

### 4.3 胜率消融（800 副牌 × 换座，配对）

对 `base680k` 与 `w5_ctrl`（两个不同训练史的 ckpt），逐条件 800 副牌。Δ = 该条件与完整观测在同
一副牌上的分差之差（单位：分/200，正 = 置零后反而更好；CI 为配对 SE 的 95%）。

| 条件 | base680k Δ | w5_ctrl Δ | 是否可分辨 |
|---|---|---|---|
| `drop_rank_counts` | +0.01 ± 0.59 | −0.66 ± 1.14 | 否 |
| `drop_current` | −0.49 ± 0.76 | +0.60 ± 1.38 | 否 |
| `drop_own_count` | −0.12 ± 0.78 | +0.85 ± 1.38 | 否 |
| `drop_hand_counts_all` | −0.11 ± 1.07 | +0.62 ± 1.87 | 否 |
| `drop_revealed` | −0.26 ± 1.12 | +0.95 ± 1.70 | 否 |
| `drop_incumbent_size` | +0.01 ± 0.05 | +0.10 ± 0.56 | 否 |
| `drop_incumbent_all` | +0.06 ± 1.06 | +0.14 ± 1.56 | 否 |
| `drop_scores` | −0.05 ± 0.38 | +0.33 ± 0.73 | 否 |
| `drop_hand` | −0.11 ± 1.29 | −0.88 ± 2.42 | 否 |
| `proposal_drops`（rc+current+own count+inc_size） | −0.35 ± 1.26 | +0.64 ± 1.94 | 否 |

其余 4 个 ckpt（`w5_scratch` / `w5_vf1` / `w5_pool4g` / `w5_bigbatch`）同协议 800 副牌的逐副牌配对 Δ
（6 个 ckpt 的 min–max）：`drop_hand` **[−1.68, +1.00]**（最差 = `w5_vf1` −1.68）、
`drop_hand_counts_all` [−0.39, +0.99]、`drop_rank_counts` [−0.66, +0.99]、`drop_revealed`
[−0.26, +0.95]、`proposal_drops` [−0.35, +0.64]，其余条件都落在 [−0.69, +0.85] 内；
**没有任何条件在任何 ckpt 上让 95% CI 排除 0**。原始 JSON：`runs/obs-slim/obs_winrate_*.json`。

### 4.4 结论与 caveat

**经验重要性排序**：`hand` ≫ `revealed` ≈ `hand_counts` ≳ `incumbent_kind` > `current` ≈
`incumbent_top` ≈ `rank_counts` > `draw_count` > `scores`（仅价值头）> `incumbent_size`。

但这个排序**不能**直接翻译成"能删谁"：

1. 置零/打乱测的是**训练后策略的依赖**，不是信息含量。`rank_counts` 排在中游恰恰因为网络用了它
   ——而它的信息完全在 `hand` 里，从零训练的网络完全可以只从 `hand` 学会同样的计数。
2. 这些 ckpt 对输入本身很钝：把**整段手牌**置零，6 个 ckpt 对 GreedyBot 的 800 副牌逐副牌配对
   Δ ∈ [−1.68, +1.00] 分/200（全部不可分辨，两个最坏值 −1.68/−0.88 在 1σ–1.5σ 内）。这与 structural
   报告 §1.1 量化的价值网 EV=0.499 一致：**现有模型的决策大部分由 mask/收益结构决定，而不是由
   观测细节决定**。所以"某段置零不影响现有胜率"是个很弱的证据，不能据此声称删除安全；删除的
   安全性由 §2.3 的信息推导（rank_counts/current/own count）和 §3 的无损重编码（incumbent/revealed）
   保证。
3. `scores` 在胜率消融里不可分辨，但 |ΔV| 最大（0.18–0.30）——价值头显著依赖它，只是贪心动作
   看不出来；不能删。
4. 策略探针（`obs_policy_probe.py`，每 ckpt 14,256/14,657 个状态）：`GreedyBot` 的模板选择
   与"mask 里第一个合法模板"**100% 相同**（`legal_ids` 升序 = 最弱合法，确实如此）；`base680k`
   与之相同率 **66.2%**、`w5_ctrl` 74.0%；模板头平均熵仅 **0.215 / 0.178**，花色头熵
   **1.195 / 1.292**；平均合法模板 4.6 个，只有 1 个合法模板的状态占 16.8%/16.2%。模板头本来就
   接近确定性，因此它对观测扰动不敏感——真正吸收观测信息的是花色头与价值头。

## 5. 从零 pilot：full 191 vs S1 103 vs S2 68

### 5.1 设置

- **臂**：A = 原 191 编码；B = S1 103；C = S2 68。全部**从零**（warm start 在这三种布局间
  形状不兼容，见 §6）。
- **训练**：`obs_pilot_train.py` 在进程内 monkeypatch `env.encode_observation`/`observation_dim`
  （`src/` 未改）；其余超参与 w5 的从零臂一致：`--opponent greedy --num-envs 8 --num-steps 128
  --seed {1,2,3}`。**预算按用户指令定为 200k**（原计划的 500k 被砍）：9 个 run 全部
  `--total-timesteps 200000 --tensorboard True`，3 并发 × 3 批（实测约 6 min/批）。
  预算理由：三臂共享完全相同的 200k，臂间差异可归因于布局；200k 足够粗筛"某布局是否落后
  一个可分辨量级"，不足以证明 <10–20 Elo 的等价——最终结论以同牌换座 h2h 为准，若 200k
  有信号再按 500k 追认。早前的一条 seed-1/500k 预实验按预算指令**作废并删除目录**，其数字
  不进入本文任何结论。
- **验收**：`slim_h2h.py` 允许两侧不同布局：同训练 seed 的 A/B/C 三对互测各打
  `--pairs 400 --seeds 0,1,2`（每对 2400 局，聚簇 bootstrap），再合并 3 个训练 seed；另有每臂
  vs `GreedyBot` 的 3 deal seed（1200 副牌）绝对强度对照（可比 w5 的 probe 胜率）。全部 18 个
  调用 4 并发，已跑完并存 `runs/obs-slim/{h2h3,wr3}_*.json` + 每局 JSONL。

### 5.2 结果

**直接候选对候选（同牌换座，`h2h3_*`；正 = 左边的 Elo 更高）。** 每个训练 seed 内部已经合并
3 个 deal seed × 400 副牌 = 1200 副牌；跨训练 seed 再用 wave5 §4.2 规则合并。

| 训练 seed | A−B（full−S1） | A−C（full−S2） | B−C（S1−S2） |
|---|---|---|---|
| 1 | +4.9 [−7.5, +17.4] | +19.1 [+6.5, +31.8] | +3.6 [−11.9, +19.1] |
| 2 | +24.1 [+11.2, +36.9] | +5.1 [−7.7, +17.9] | −20.4 [−33.2, −7.7] |
| 3 | −15.9 [−28.1, −3.8] | +19.7 [−0.1, +39.5] | +36.8 [+7.8, +65.7] |
| **合并 3 seed** | **+4.4 [−18.3, +27.0]** | **+14.6 [+5.3, +24.0]** | **+6.6 [−25.9, +39.1]** |

读法：**S1（103）与 full（191）不可分辨**（+4.4，CI 跨 0 且很宽）；**S2（68）在合并口径上落后
full 约 14.6 Elo，CI 刚排除 0**；S1 vs S2 一次比较（+6.6）不可分辨。

**绝对强度 vs GreedyBot（每训练 seed 1200 副牌，换座配对，`wr3_*`）。**

| 臂 | winrate | score diff | Elo vs greedy（面值） |
|---|---|---|---|
| A full 191 | 0.616 [0.601, 0.631] | +13.73 | +82.4 [+71.4, +93.5] |
| B S1 103 | 0.604 [0.567, 0.642] | +12.24 | +73.9 [+46.3, +101.6] |
| C S2 68 | 0.602 [0.591, 0.612] | +12.10 | +71.8 [+64.2, +79.4] |

三臂对 GreedyBot 的 winrate 只差 ≤1.4 个百分点，任一臂的 CI 都与另两臂重叠；A 的点估计领先
B/C 约 8–10 Elo，但在 3 个训练 seed 的噪声下不可分辨。

### 5.3 判读

1. **S1（103）在 200k 预算下不损失可分辨强度**：h2h A−B 合并 +4.4 [−18.3, +27.0]，绝对强度
   A/B 差 ~1.2% winrate 也不可分辨。信息论证（§2–§3：S1 对决策信息无损）与实测一致。但 200k
   模型本身还没训满（vs GreedyBot 只有 ~0.60–0.62 胜率），CI 宽到 ±20+ Elo，所以本 pilot 只能
   支持"没有明显损失"，不能证明"严格等价"。
2. **S2（68）有 ~15 Elo 的落后信号**：h2h A−C 合并 +14.6（CI 排除 0），3 个训练 seed 里 3/3 的
   点估计为正；与"丢掉 rank↔suit 配对"的机制解释一致（花色头是 S2 唯一无法精确恢复的信息）。
   15 Elo 小于 400 副牌的分辨率，但合并 3600 副牌后可见；因此**不建议在未跑 500k 追认前实施 S2**。
3. **S1 与 S2 的直接比较（+6.6 [−25.9,+39.1]）不可分辨**——训练 seed 间波动（B−C 三个 seed 为
   +3.6 / −20.4 / +36.8）远大于布局效应；这正是为什么结论只能基于合并口径且必须 3 seed。

### 5.4 为什么 vs-Greedy 与 h2h 看起来矛盾（复核）

评审看到的现象：seed 1 的单个 deal seed vs-greedy 读数里 A 的 winrate（0.636）比 B（0.589）/
C（0.591）高 ~3.4–4.5 个百分点，换算成 Elo 约 +30–34；但同 seed 的直接 h2h A−B 只有 +4.9
[−7.5, +17.4]。我们按建议把 vs-greedy 也放到**相同的 3 个 deal seed** 上，并对每个 deal 做配对
（两个臂在同一副牌上的 `score − greedy` 之差；1200 副牌/训练 seed）：

| 训练 seed | A−B（Δ分 / Δwin） | A−C | B−C | 对应 h2h（A−B / A−C / B−C） |
|---|---|---|---|---|
| 1 | +3.62±2.26 / +.036±.025 | +3.75±2.17 / +.026±.025 | +0.13±2.25 / −.010±.026 | +4.9 / +19.1 / +3.6 |
| 2 | +3.08±2.29 / +.028±.026 | +0.99±2.34 / +.011±.027 | −2.09±2.31 / −.017±.026 | +24.1 / +5.1 / −20.4 |
| 3 | −2.23±2.17 / −.024±.025 | +0.16±2.24 / +.012±.025 | +2.39±2.13 / +.035±.025 | −15.9 / +19.7 / +36.8 |

三个训练 seed × 三对共 **9 组符号全部一致**（配对 vs-greedy Δ 与直接 h2h 同号）。结论：

1. **首先这是估计噪声，不是两个真值冲突**。旧（单 deal seed）`vs-greedy` 是**锚点绝对强度**：
   每臂 400 副牌的 winrate SE ≈1.4%，而 winrate 到 Elo 的换算是非线性放大（p≈0.6 时
   dElo/dp ≈ 800–900 Elo/单位），所以 ±1.4% 的 winrate 噪声就能造成 ±12–15 Elo 的臂间差；
   两个独立臂之差再放大到 ±20+ Elo。把同配对（同牌、两座位）用上后，seed 1 的 A−B 从
   "+34 Elo"缩到 **+3.6 分/200（winrate +3.6%±2.5%）**，与 h2h 的 +4.9 [−7.5,+17.4] 同量级。
2. **另一部分是真实的不传递（non-transitivity）**："对 GreedyBot 的胜率"与"对另一个学习者的
   胜率"不是同一个标量。seed 3 的配对 vs-greedy A−C 只有 +0.16 分（~0），而 h2h A−C 是 +19.7 Elo；
   反向的例子也存在（s2 B−C：greedy Δ −2.09 vs h2h −20.4）。2 人游戏里策略差异可以像
   石头剪刀布一样改变排序，所以**候选对候选才是验收口径**（head-to-head-pilot §1），
   vs-greedy 只能当绝对强度的粗检。
3. **尺度不同**：h2h 的 ±400 副牌 ≈ ±20 Elo 只适用于候选对候选；绝对 Elo 对锚点的分辨率更粗。
   报告里两个数字都保留，但任何"A 比 B 高 30 Elo"级别的结论都必须用 h2h 的 CI 说话（本报告
   只有 A−C 的合并 CI 排除了 0）。

## 6. 影响面与向后兼容

改动 `_SEGMENTS`/`encode_observation` 会波及：

| 位置 | 现状 | 影响 |
|---|---|---|
| `env.py:105-135` | `_SEGMENTS` 单一真源 | 新布局加在这里（或加版本参数） |
| `env.py:162-171` | `obs_dim`/`observation_space` | 重新求和即可 |
| `networks.py:107-112` | 首层 `Linear(obs_dim, 128)` | 形状随布局变 |
| `networks.py:230-245` | `NeuralPolicy.act` 调 `encode_observation` | **必须按 ckpt 的布局选编码器**，否则老 ckpt 推理直接形状报错 |
| `networks.py:157-182` | `save_agent`/`load_agent` | payload 已存 `obs_dim`；建议加 `obs_version`/`obs_layout` |
| `networks.py:186-205` | `warm_start_into` | 只拷贝同形状张量，首层跨布局不会拷；跨布局热启动需显式拒绝或加投影 |
| `train.py:353` | `obs_dim = observation_dim(...)` | 加 `--obs-version` 后同步 |
| `train.py:360-366` | `nvec` 相同就走 strict `load_state_dict` | **191→103 会直接 RuntimeError**（形状不匹配），必须按 obs_dim 分支 |
| `train.py:435` | `assert observation_space.shape == (obs_dim,)` | 同步 |
| `eval.py:99-100` / `play.py:427-428` / `policies.py:136-141` | `NeuralPolicy` + `policy_from_spec` | 评测/对局/工具全部经此，需按 ckpt 标签选编码器 |
| `tools/head_to_head.py`、`build_ladder.py`、`h2h_screen.py`、`measure_trace_signal.py` | `policy_from_spec`/`NeuralPolicy` | 同上一行；混布局对局需要 §5 的 per-side encoder |
| `trace.py` | 只存 `Deal + 每步动作/分数/标签`，**不存 obs** | 无影响；旧 trace 可回放 |
| `tests/test_env.py:14-31`、`test_train.py:23` | 硬编码 191 / `185+3n` | 新布局需参数化 |
| `tests/test_game.py:121-148` | 差分泄漏测试直接调 `encode_observation` | 任何新编码器都应跑同一测试（S1/S2 只读 `View`，天然通过） |
| `DESIGN.md:285-304` | §4 段表 | 需更新 |

**兼容方案（推荐 A）**：

1. 保留 v1 布局不动，新增 `_SEGMENTS_V2`；`observation_dim(num_players, version=1)`、
   `encode_observation(view, rules, version=1)`；`Agent`/ckpt payload 增 `obs_version`（默认 1）。
2. `--obs-version {1,2}`（默认 1），`Seven523Env(..., obs_version=?)`；`NeuralPolicy` 从
   `agent.obs_version` 选编码器。旧 ckpt 无字段 → v1，行为逐位不变。
3. 跨版本 `--load-checkpoint`：`train.py:360` 先比 `obs_dim`，不一致就报错（或只做 trunk 的
   `warm_start_into`，明确不拷首层）。
4. `trace`/replay 不需要版本字段（不含 obs）。

**为什么不建议"直接替换默认布局"**：`runs/` 里 20+ 个 191 维 ckpt 是全部评测/梯队工具的输入，
换默认会一次性失效；版本参数的成本很小。

## 7. 结论与建议

**能带来什么：**

- 首层参数 −46%（24,576→13,312）、总参数 −19.1%；每步 dense 计算约 −19%。
- 输入更干净：去掉常量列与重复信息后，网络不必在首层学"对 hand 求和"或"座位提示"。
- 与增广的协同：腾出 88 维预算给已出牌历史/当前墩分（structural 方向 B），组合后仍不超过 191。

**不能带来什么（别夸大）：**

- 不改变奖励稀疏/信用分配（structural §1.1：终局 only、早期 EV 0.065）。
- 不改变对手分布（structural §1.3：逐决策重抽、无反馈）。
- 不补已出牌历史/当前墩分（structural §1.2/§3）——这是目前最大的信息缺口。
- 本报告 §4 的证据表明现有策略对观测细节钝感；**没有理由期待精简本身带来 Elo 提升**。它的
  价值是工程性的（参数/预算/可维护性）与"为增广腾位置"。

**建议：**

1. **值得做，但优先级低于增广**。采用 **S1 103 维**（信息无损重编码 + 删除 3 类可推导维）；
   **不要采用 S2 68**：200k pilot 显示它落后 full ~15 Elo，与丢掉 rank↔suit 配对的机制解释一致。
   `rank_counts` 可作可选保留（118 维）——pilot 证明不必要（S1 已无损失），仅当 500k 追认出现
   样本效率问题时再回退。
2. **最小改动**：`env.py` 加 v2 布局与版本参数；ckpt payload 加 `obs_version`；`train --obs-version 2`；
   工具按 ckpt 标签 dispatch。约 100–150 LOC + 测试。
3. **验证预算**（本报告实测）：从零训练约 900–1200 sps（8 envs、3 并发）；**200k/run ≈ 3–4 GPU-min**，
   9 run ≈ 20 GPU-min；验收 18 个 h2h/评测调用、4 并发 ≈ 10 GPU-min（本报告已跑完）。
   若要声称"无损失"，至少 3 训练 seed × 3 deal seed（本轮就是）；若要声称"有提升"，按
   [`head-to-head-pilot.md`](./head-to-head-pilot.md) 的分辨率需要 ≥1500 副牌。
4. **下一步**：S1 已可直接实施；若想用 S2，先用 500k × 3 seed 复核那 ~15 Elo 是否在长训练下保持。

## 8. 复现

```bash
# 静态审计（45,752 状态）
uv run --group train python runs/obs-slim/obs_audit_static.py

# 敏感性消融（批推理 + 800 副牌胜率）
uv run --group train python runs/obs-slim/obs_ablation.py \
    --ckpt runs/probe/base_step00696320.pt --games 400 --out runs/obs-slim/obs_ablation_base.json
uv run --group train python runs/obs-slim/obs_winrate.py \
    --ckpt runs/probe/base_step00696320.pt --deals 800 --out runs/obs-slim/obs_winrate_base.json

# 精简布局从零 pilot（200k，脚本内 monkeypatch，不改 src/）
runs/obs-slim/obs_pilot_launch3.sh   # full/b/c × seed 1/2/3，TB on，3 并发

# 验收：混布局 h2h（9 个）+ vs-greedy（9 个），4 并发
runs/obs-slim/obs_eval_parallel.sh
uv run --group train python runs/obs-slim/obs_h2h_summarize.py   # h2h + 绝对强度合并
uv run --group train python runs/obs-slim/obs_greedy_paired.py   # 按 deal 配对的 vs-greedy 复核
```

---

## 实现分析（2026-09-25，obs_version v2 落地后）

> T4 的实现级设计（门控见 `docs/plans.md` §3 T4、§4 D-4/D-7）。v2 已落地：`env.py:33 OBS_VERSION=2`、
> `env.py:151-156 _SEGMENTS_V2 = _SEGMENTS_V1 + B0`（只追加，所以 v1 是 v2 的逐位前缀，`env.py:207-217`）。
> **2026-09-25 编号协调**：B1（T3）已占用 v3 = v2+B1 = 249；本节 S1 家族顺移为 v4/v5（见 §1）。
> 本文参数用工作区 `.venv/bin/python` + `seven523.networks.Agent`（hidden=128、nvec=[134,4]）现场复核：
> 191→24,576/59,019、194→24,960/59,403、249→32,000/66,443、103→13,312/47,755、106→13,696/48,139、161→20,736/55,179
> （首层/总参数）；测试基线 334 条（`pytest --collect-only`）。

### 1. 最终布局：v4 = 观测 S1 + B0（B1 保留时 v5 再加 B1）

2 家完整段表（顺序 = 前缀链 S1 ⊂ v4 ⊂ v5；B1 两段是条件段：B1 独立落地是 v3 = v2+B1 = 249，保留进 S1 家族时构成 v5 的尾部）：

| # | 段 | 2p 宽 | 公开性 | 编码式 | v1/v2 对照 |
|---|---|---|---|---|---|
| 1 | `hand` | 54 | 私有 | `card_id` multi-hot | v1 `hand`（`env.py:64`）不变 |
| 2 | `inc_rank` | 15 | 公开 | 顶牌 `RANK_INDEX` one-hot | `incumbent_top` 的 rank 半边 |
| 3 | `inc_suit` | 4 | 公开 | 花色 one-hot，王=全零 | `incumbent_top` 的 suit 半边 |
| 4 | `inc_kind` | 6 | 公开 | `ComboKind` one-hot | v1 不变（`env.py:80`） |
| 5 | `inc_size` | 1 | 公开 | `size / hand_size` | v1 不变 |
| 6 | `draw` | 1 | 公开 | `draw_count / 54` | v1 `draw_count`（`env.py:90`）不变 |
| 7 | `scores` | n=2 | 公开 | 自中心：`scores[(seat+k) % n] / total` | v1 绝对序 + 旋转 |
| 8 | `opp_count` | n−1 | 公开 | 对手 `counts[(seat+k)%n] / hand_size` | v1 `hand_counts` 删自己维 |
| 9 | `opp_revealed` | 19(n−1) | 公开 | 对手亮牌 `rank15 + suit4` | v1 `revealed` 取对手半边 |
| 10 | `trick_points` | 1 | 公开 | B0，v2[191] | v2 原样前移 |
| 11 | `remaining_points` | 1 | 公开 | B0，v2[192] | 同上 |
| 12 | `point_hold` | 1 | 公开 | B0，v2[193] | 同上 |
| 13 | `unseen`（B1） | 54 | 公开 | 54 − `hand` − `revealed` − `played` − 当前墩，multi-hot | 新 |
| 14 | `last_player`（B1） | 1 | 公开 | `((last_player − seat) mod n) / n`，None=0 | 新（2 家冗余，见 §6） |

维度公式（与 pilot 编码 `runs/obs-slim/obs_slim_encoder.py:24-33` 一致）：**观测 S1 = 61 + 21n**；
**v4 = S1 + B0 = 64 + 21n → 2 家 106**；**v5 = S1 + B0 + B1 = 119 + 21n → 2 家 161**。对照 v1 = `185+3n`（2 家 191）、
v2 = `188+3n`（2 家 194）、v3 = v2+B1 = `243+3n`（2 家 249）；v1/v2 与 v4 没有前缀关系（重排 + 压缩），
v2 ⊂ v3 与 v4 ⊂ v5 各自是逐位前缀。

| 布局 | 2p dim | 首层 | 总参数 | vs 191 |
|---|---|---|---|---|
| v1 / v2 | 191 / 194 | 24,576 / 24,960 | 59,019 / 59,403 | — / +0.65% |
| v3（B1） | 249 | 32,000 | 66,443 | +30.2% / +12.6% |
| S1（pilot，OS §3.4） | 103 | 13,312 | 47,755 | −45.8% / −19.1% |
| **v4** | **106** | **13,696** | **48,139** | **−44.3% / −18.4%** |
| **v5** | **161** | **20,736** | **55,179** | **−15.6% / −6.5%** |

口径澄清：plans T4 的「首层 −46% / 总 −19.1%」是 S1(103) 的数；v4(106) 因多 3 个 B0 维，
复核为 −44.3% / −18.4%。v4 仍比 v2 小 −45.1% 首层 / −19.0% 总参。

逐项删除/压缩依据（OS §2.3 的 45,752 个 acting-view 实测 + §3.1 定义）：

- 删 `rank_counts`(15)：`Counter(view.hand.rank)/4` 是同一 `view.hand` 的聚合（`env.py:64/69`），**0/45,752 不一致**（OS §2.3a）。
- 删 `current`(n)：生产路径编码的 `View` 恒为 acting seat（`match.py:94-96` → `game.py:137`；`env.py:382-389`），**0/45,752**（OS §2.3b）；必须同时自中心旋转，否则座位语义丢失（OS §3.1 注）。
- 删自身 `hand_counts` 那 1 维：`counts[seat] == len(hand)`（`game.py:147`），**0/45,752**（OS §2.3c）；对手计数保留。
- `incumbent_top` 54→19：54 维 one-hot 只 5.75 bit；rank+suit 无损（王由 rank 判、suit 全零），`hand`+`mask` 只能 96.4% 反推（3.6% 歧义，OS §2.3d），段不能删、只能压。
- `revealed` 54→19(n−1)：每座恒一张、91% 已陈旧（OS §2.3e），但对手开局牌先验仍有价值，保留对手那张；自己的亮牌在手里可推、打掉后属历史。
- 自中心旋转：`scores`/`opp_count`/`opp_revealed` 都按 `(seat+k)%n` 写（pilot 的 `encode` 同序，`obs_slim_encoder.py:52-58,88-91`）。

版本命名：ckpt 身份要求「一个版本号 = 一种确定布局」（`env.py:30-34`）。**编号协调（2026-09-25）**：
B1（T3）已占用 **v3 = v2+B1 = 249**；S1 家族自 **v4** 起编号——**v4 = S1+B0 = 106、v5 = S1+B0+B1 = 161**；
一个版本号仍只对应一种确定布局，106 是 161 的逐位前缀（`(version, players) → dim` 必须是函数）。

### 2. 热启动：v4 不是 v1 的前缀（关键新增设计）

现状：`warm_start_into`（`networks.py:195-247`）只处理「追加」布局——同玩家数 + 版本单调 + 宽度可容纳时，
`network.0.weight[:, :old] ← old`、新列置 0（`networks.py:213-241`）；`train.py:396-411` 在 `nvec/obs_dim/obs_version`
全等时 strict load，否则直接调它。v4 是重排 + 压缩，两种失效模式都必须消除：

1. **前缀拷贝静默错位**：dst[0,15) 是 `inc_rank`，却会收到 src[0,15) 的 `hand` 权重；
2. **194→106 静默跳过**：宽度条件 `origin.shape[1] <= value.shape[1]`（`networks.py:230`）不满足，首层整体不拷，
   只剩预训练 trunk + 随机首层，且无任何报错。

推荐方案：**segment 级列重映射**。

1. `env.py` 增 `segment_spans(obs_version, num_players) -> tuple[(name, start, width)]`（由 `_segments_for` 推导，单一真源）。
2. `networks.py` 增缓存 `_first_layer_remap(src_version, src_dim, dst_version, dst_dim) -> Tensor | None`，返回
   `(dst_dim, src_dim)` 映射；逐列 gather 保证 0/1 拷贝逐位等价，仅 54→19 的两块用小 dense 矩阵。
3. `warm_start_into` 首层分支改为按映射拷贝；trunk/actor/critic 仍同形照拷（现有 `tests/test_networks.py:199-221` 不变）。

映射规则（以 v2→v4 为例；v2 偏移 `env.py:137-156`，v4 偏移见 §1 表）：

| src 段 → dst 段 | 语义 | 等价性 |
|---|---|---|
| `hand`54→`hand`54、`inc_kind`、`inc_size`、`draw`→同名 | 恒等 | **逐位** |
| B0 v2[191:194] → v4[103:106] | 恒等前移 | **逐位** |
| `scores` v2[131:133] → v4[81:83] | 绝对序→自中心；旧训练 learner 恒为 seat 0（`train.py:385`），self←src[131]、other←src[132] | seat 0 逐位等价；seat 1 是分布外 |
| `hand_counts` v2[134] → `opp_count` | 对手维 | **逐位** |
| `rank_counts` / `current` / 自身 `hand_counts` / 自身 `revealed` | 丢弃 | 权重不拷（新列由训练学） |
| `incumbent_top`54 → `inc_rank`15+`inc_suit`4 | 54→19 | **近似**：最小二乘 `w[c] ≈ a[rank]+b[suit]`（随机权重相对残差 ~0.85，本区复核）；保守版 = rank 列取该 rank 均值、suit 列置 0 |
| `revealed`54 → `opp_revealed`19 | 54→19，只保对手 | **近似**，同法 |
| v1→v4、v1/v2→v5 | 同上，B0/B1 目标列置 0 | 近似（首层） |
| v4→v5 | 追加 55 列置 0 | **逐位**（前缀） |

守卫合并（逐位等价 / 近似 / 拒绝三档）：

- 同版本同 dim：strict `load_state_dict`（`train.py:398-403`）——**逐位**。
- 允许的前向映射集合 `{(1,2), (2,3), (1,3), (1,4), (2,4), (4,5), (1,5), (2,5)}`；其中 `(1,2)`、`(2,3)`、`(1,3)` 是现有前缀路径（v1 ⊂ v2 ⊂ v3，B1 只追加），其余走 remap——**近似**（见上表）。
- **必须拒绝**：v4→v3、v4→v2、v4→v1 及其余所有版本回退（如 v5→v4/v3/v2/v1），以及所有跨玩家数组合（两侧 `observation_num_players` 不等，`networks.py:225-232`）；
  在 `train.py` 加载点 `raise SystemExit` 明确报错（OS §6 方案 A 第 3 条）。`warm_start_into` 本身保留「不可映射就跳过首层」
  的旧语义，以保住 `tests/test_networks.py:224-248`；T7（2→3 家）也依赖这条拒绝路径是响的。
- 组合性：`(1,5)`/`(2,5)` = 对应 remap + B1 55 列置零，一张表可复用。

代码量：`env.py` ~15 行（spans + 宽度 offset），`networks.py` ~70–90 行（remap 表 + 缓存 + 守卫），`train.py` ~5 行。

### 3. 版本与工具接线

| 位置 | 改动 | 量 |
|---|---|---|
| `env.py:33-34` | `OBS_VERSION=4`、`OBS_VERSIONS=(1,2,3,4)`（v5 落地时 +5） | 2 行 |
| `env.py:51-61,173-176` | `_Segment` 支持「n−1 槽」宽度（如 `players_offset`），`_segment_width` 同步 | ~10 行 |
| `env.py:177-196` | `observation_num_players` 改为按 `observation_dim(n, version)` 扫 n=2..7（现有断言全部保持：191/1→2、194/1→3、194/2→2、197/2→3、192/1→None） | ~10 行 |
| `env.py:137-156` 之后 | `_SEGMENTS_V4`（及 `_SEGMENTS_V5`）+ 新 writer `_write_inc_rank/_inc_suit/_scores_rotated/_opp_counts/_opp_revealed`；B0 writer 直接复用 `env.py:113-135` | ~60 行 |
| `networks.py:195-247` | §2 的 remap 与拒绝表 | ~70–90 行 |
| `train.py:124-131` | `--obs-version` choices `{1,2,3,4}`、default 4（v5 落地后加 5） | 3 行 |
| `train.py:396-411` | 加载点拒绝不可映射组合；打印 `obs src→dst` | ~5 行 |
| `game.py`（B1） | `GameState`/`View` 加 `played: tuple[Card, ...] = ()`；`Game.view` 透传；`_end_trick` 累积（`game.py:203-256`）；`trace.py` 零改动（`restore` 走默认 `()`，回放时经 `_end_trick` 重建） | ~20 行 |

`Agent`/`save_agent`/`load_agent`（`networks.py:81-193`）与 `NeuralPolicy`（`networks.py:250-295`）已就绪：payload
已有 `obs_version`（`networks.py:168`），旧 ckpt 无字段→v1（`networks.py:185-186`），`NeuralPolicy` 按 `agent.obs_version`
选编码器（`networks.py:272,289`）。**`tools/` 零改动**：五个工具都经 `seven523.ladder`/`duel`/`arena` →
`policy_from_spec`（`policies.py:115-143`）→ `NeuralPolicy`；`eval.py`/`play.py` 不碰 obs_dim；B1 只加 `View.played`，
由 `game.view` 统一产出，工具按 ckpt dispatch 已完全由 `NeuralPolicy` 承担。需更新的只有文档 `DESIGN.md:285-304`（§4 段表）。

### 4. 测试清单

1. **v1/v2 逐位不变**：`tests/test_env.py:61-100`（维度/公式）、`:103-115`（v1 是 v2 前缀）、`:148-166`（v2 B0 闭式）、
   `:169-177`（v1 金标 fixture）保持；改 `_segment_width`/映射前先把现有 v2 编码冻成 fixture 快照（新增）。
2. **v4 信息等价**（OS 口径）：随机局面上断言 (a) `rank_counts == Counter(hand)/4`；(b) `current == seat`；
   (c) own count `== len(hand)/hand_size`；(d) v4 的 `hand`、`inc_rank+inc_suit`、`opp_revealed` 可解码回 v1 的
   `hand`、`incumbent_top`、对手 `revealed`；(e) 两个 seat 各自编码，自中心字段按 `(seat+k)%n` 对齐。
3. **差分泄漏**（扩展 `tests/test_env.py:179-229`）：仅隐藏字段不同 → v4 逐位相同；B1 再加「unseen 集合相同、
   底牌堆与对手手牌的拆分不同」→ `unseen` 段逐位相同。
4. **参数计数**：`Agent(103/106/161).network[0]` = 13,312 / 13,696 / 20,736，总数 = 47,755 / 48,139 / 55,179
   （基线 191 = 24,576/59,019）。
5. **跨版本热启动语义**：v1→v2 逐位（现有）；v2→v4 的 `inc_kind`/`draw`/B0 列逐位、`rank_counts` 丢弃、
   `scores` 置换、54→19 分解只在合成权重（同 rank 等权）上逐位；v4→v2 加载点必须报错；跨玩家数仍跳过首层
   （`tests/test_networks.py:224-248`）；v4→v5 新 55 列全 0 且前缀函数逐位相同。
6. **334 全绿**（`uv run --group train pytest -q`）；预计新增 20–30 条，测试代码 ~180 行。

### 5. 实验协议与门控（D-4 / D-7）

- **不先单独落 S1**：B1 定稿后一次迁移同时进 S1+B0（B1 保留时连 S1+B0+B1 一起），默认值从当时的现行版本
  （B1 落地后为 v3=249）切到 S1 组合布局（v4/v5，取决于 B1 是否保留）；plans D-4 的「一次定稿」在代码上就是
  `_SEGMENTS_V4`（及 `_SEGMENTS_V5`）一次落地 + `OBS_VERSION` 一次切换。
- **组合臂**：S1（103，已有 OS §5 pilot：monkeypatch、从零 200k）vs S1+B0（v4=106）vs S1+B0+B1（v5=161，B1 保留时）。
  新臂按 SD §3.4 母版：warm start `runs/probe/base_step00696320.pt`（v1，经 §2 remap）、500k、greedy、SAME_STEP、3 seed。
  注意这与 pilot 不同（pilot 因布局不兼容只能从零，OS §5.1）；remap 落地后 warm start 可行，必须作为固定控制变量。
- **验收沿用 OS §7.2/§7.3**：「无损失」需 ≥3 training seed × 3 deal seed（S1 已做；v4 因 B0 改变布局需重跑）；
  「有提升」需 ≥1500 副牌；判定按 plans §1.2（3×400，CI 排除 0 且点估计 ≥ +10；+20 用单侧等效检验）。
- **`rank_counts` 回退**（D-7 已关闭）：坚持观测 S1；仅当 500k 追认出现样本效率问题时才加回，届时是
  v4+15 = **121**（+B1 = 176），不是 OS §3.4 的 118（那是 B0 之前的数）。
- 落地前用 fixture 对拍 pilot 编码器，确认 §1 段表与 OS §5.1 的三臂逐位一致。

### 6. 风险与取舍

- **pilot 与落地漂移**：OS 结论来自 monkeypatch（`runs/obs-slim/`，`src/` 未改）；落地以 §1 段表与
  `obs_slim_encoder.py` 为准，先对拍再训练。200k pilot 只支持「无明显损失」，不等于严格等价（OS §5.3）。
- **remap 近似**：两个 54→19 段与自中心置换只保证 trunk/heads 逐位，首层不是旧函数的等值变换；若怀疑初始凹陷，
  加一条「v2→v4 冷首层」500k 对照臂即可判别（成本 ~6 min）。
- **2 家 `last_player` 冗余**：2 家下 `incumbent is not None ⇔ last_player == other`（`game.py:169-190` 设置、
  `game.py:254` 清空），B1 的 1 维在 2 家是纯冗余；只服务 2 家可删（161→160），保留则换 N 家前向兼容。
- **S2 已否**：68 维丢 rank↔suit 配对，200k 落后 ~15 Elo（OS §3.2/§5.3、plans N-6），不作回退选项。
- **trace/GameState**：B1 改 `GameState`/`View`/`_end_trick`；`Deal`（`game.py:47-64`）与 trace（`trace.py:96-107`）
  不含 obs，旧 trace 可回放（开局 `played=()`、回放时重建）；扩 ADR-0002 泄漏测试是硬要求。
- **旧 ckpt 评测兼容**：`NeuralPolicy` 按 agent 版本编码，20+ 个 191/194 ckpt 在 v4 工具链下继续可评；
  `warm_start_into` 的拒绝语义变化只影响 `--load-checkpoint` 的跨版本方向，不改推理路径。
- **未决**：B1（T3）已定稿（未确认）并占用 v3 = v2+B1 = 249；S1 迁移按 D-4（前提已满足，B0/B1 已定稿）一次性落 v4/v5，
  默认值届时从 B1 布局切到 S1 组合布局（取决于 B1 是否保留）。最大的未决风险 = remap 的 54→19 近似
  在 500k 追认中产生不可分辨的初始凹陷。

### 7. 落地记录（2026-09-25）

- **决策**：默认观测切到 **v5 = 观测 S1 + B0 + B1 = `119 + 21n`（2 家 161）**，同时落地
  **v4 = 观测 S1 + B0 = `64 + 21n`（2 家 106）** 作为前缀链中间版本；v4 是 v5 的逐位前缀。
  决策与兼容契约见 [ADR-0008](../adr/0008-observation-layout-v5.md)；`OBS_VERSION=5`、
  `OBS_VERSIONS=(1,2,3,4,5)`，`observation_dim`/`encode_observation`/`segment_spans` 仍从段表
  （`_SEGMENTS_V4`/`_SEGMENTS_V5`）单一真源推导；B0 三 writer 与 B1 两段分别复用 v2/v3
  （`last_player` 沿用 v3 的绝对座位归一化 `(last_player + 1) / n`，None = 0，未做自中心旋转）。
- **热启动**：按 §2 落地段级列重映射——允许集 `{(1,2),(2,3),(1,3),(1,4),(2,4),(4,5),(1,5),(2,5)}`，
  其中前缀方向新列置 0，`incumbent_top`/`revealed` 的 54→19 分解为近似；v4→v3/v2/v1 等回退与
  跨玩家数组合在训练加载点显式拒绝。
- **对拍与测试**：v5 与 pilot 布局 b 逐位对拍（`tests/data/obs_v5_golden.json`；pilot 编码器缺失时
  跳过），v1 金标 fixture（`tests/data/legacy_v1_views.json`）保持逐位不变；v4 ⊂ v5 逐位前缀、
  维度公式、`segment_spans`、差分泄漏与跨版本热启动（段级列重映射、前缀拷贝、不可映射拒绝）按
  §4 清单回归。参数与首层预算与 §1 表一致（v4 13,696/48,139、v5 20,736/55,179）。组合臂的
  500k warm-start 验收协议见 §5。

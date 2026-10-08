# Stage-0 D-B 固定牌局拟合探针（fixed-deal probe）

> 状态：**完成（2026-10-06）**。判定：**中间情形 / null** —— 固定集既无记忆优势、也无
> 泛化 gap，F 达到参考天花板；vs-random 尺子在 ~100k–150k 饱和，本探针无法在该尺子上区分
> 「优化/表达」与「泛化/探索」。500k 扩展按 operator 决定取消（Amendment A3），未测预算
> 记为限制。原始产物在 `runs/stage0/fixed-deals/`（gitignored）；不改 `src/`、不 commit、
> 不发布 checkpoint。

## 0. 预注册（动手前，2026-10-06）

### 0.1 假设

平台（revision-3 / obs v5 / memoryless MLP / PPO，从零 ~500k–2M 全部停在
μ≈188–192 vs-random，见 `docs/post-v5-structural-options.md` §0–1）的瓶颈是两种性质之一：

- **H-opt（优化/表达限制）**：即使把训练 deal 集压缩到可记忆的 500 副，模型也学不到比
  当前平台更高的 vs-random 强度 → 瓶颈在网络/优化，不是数据多样性。
- **H-gen（泛化/探索限制）**：模型能对固定 500 副明显过拟合、逼近参考天花板，但换 held-out
  新牌就回落 → 瓶颈在泛化/探索（数据多样性/课程/O1 类方向）。

### 0.2 臂（预注册）

| 臂 | 训练 deal | 训练 seed | 预算 | 说明 |
|---|---|---|---|---|
| **F（固定集）** | 固定 500 副（deal 池 seed `FIXED_POOL_SEED=4001`，round-robin 循环） | 1, 2, 3 | 200k 步 | 对手随机性保留（每 env 的 RandomBot RNG 仍随训练 seed 变化） |
| **G（对照）** | 正常新鲜发牌 | 1, 2, 3 | 200k 步 | 与 F 同配方同预算，唯一差别是 deal 池 |
| **REF（参考天花板）** | — | — | — | `runs/w2m_ctl__11__1790516900/agent.pt`（2M warm-start 池化臂，当前平台顶档） |

配方（F 与 G 逐项一致，从零训练，train.py 当前默认 / t17early 同款）：

```text
--num-players 2 --num-envs 8 --num-steps 128 --num-minibatches 4
--update-epochs 4 --learning-rate 2.5e-4 --anneal-lr true --lr-schedule linear
--gae true --gamma 0.99 --gae-lambda 0.95 --norm-adv true
--clip-coef 0.1 --clip-vloss true --ent-coef 0.01 --vf-coef 0.5 --max-grad-norm 0.5
--hidden-size 128 --activation relu --arch shared
--opponent random --reward-shaping terminal --num-players 2
--total-timesteps 200000 --tensorboard false --checkpoint-interval 0
--snapshot-interval 25 --run-dir runs/stage0/fixed-deals
```

训练 driver（F 臂）只 monkeypatch `seven523.game.Game._deal`（`game.py:144`），
`Game.new`（`game.py:140-142`）经 `Match.__init__`（`match.py:54`）在每个 episode 的
`Seven523Env.reset`（`env.py:348-360`）取到池内 deal；池内 500 副按固定顺序 round-robin
循环，池在所有训练 seed 间逐位相同。G 臂不 monkeypatch，直接跑 stock trainer。

### 0.3 固定池与 held-out 池（预注册 seed）

- 固定训练池 T：`FIXED_POOL_SEED = 4001`，500 副。
- held-out 池 H：`HELD_OUT_POOL_SEED = 4002`，500 副。
- 生成方式：`random.Random(seed).shuffle(list(make_deck()))`，每副牌用 54 张卡的
  `card_id`（0..53）序列做指纹；T 与 H 的指纹集合必须不相交（运行前断言 + 写入产物
  `pool_*.json`）。

### 0.4 读数（端点）

对每个策略 `P ∈ {F_s1..s3, G_s1..s3, REF}` 和每个牌集 `S ∈ {T, H}`，用同一评估器
（learner=seat 0、对手 RandomBot、每集同一条对手 RNG 流，eval seed=`EVAL_SEED=7001`、
500 副）计算 greedy vs-random：

- `win_rate = wins / 500`（平局不计入胜场，同 `eval.evaluate` 口径）
- `draw_rate`、`loss_rate`
- `mean_return`（learner 终局 margin，∈[−1,1]）
- `mean_score_diff`（learner 分 − 对手分）
- 逐副 per-deal 记录（`win ∈ {1, 0.5, 0}` 与 `return`）

主读数：

1. **F 在固定集 T 上 vs-random**：最终 `agent.pt` + 快照曲线（`--snapshot-interval 25`
   的快照，步数 25.6k/51.2k/…/199.7k）在 T 上的 win_rate/mean_return → 训练曲线。
2. **F 在 held-out H 上 vs-random**（最终 `agent.pt`）→ 泛化 gap。
3. **G 在 T 与 H 上**同两组读数（G 的 T 读数用同一 500 副 T 池，非 G 训练时见过的牌）。
4. **REF 在 T 与 H 上 vs-random** 的 win_rate/mean_return。
5. `metrics.csv` 的 `episodic_return`（F 训练时即固定集上、G 训练时即新鲜集上）作训练回报曲线。

### 0.5 分析计划与判据（预注册）

统计：逐副 deal-cluster bootstrap（4000 次重采样）给每个 win_rate/mean_return 的 CI；
跨训练 seed（k=3）按仓库合并口径 `点=seed 均值`、
`se=max(RMS(seed 内 bootstrap SE), between-seed sd)/√k`，报 z-CI（1.96）与 t-CI
（t_{0.975,2}=4.303）。

关键配对/差值（同集配对、异集双样本，均 deal-cluster bootstrap）：

- `Δ_mem = win_rate(F_T) − win_rate(G_T)`（同 T 集，逐副配对）→ 固定集是否真的带来拟合增益。
- `Δ_ceil = win_rate(F_T) − win_rate(REF_T)`（同 T 集，逐副配对）→ F 是否逼近参考天花板。
- `gap_F = win_rate(F_T) − win_rate(F_H)`（异集，双样本）→ F 的泛化 gap。
- `gap_G = win_rate(G_T) − win_rate(G_H)`（异集，双样本）→ G 的基线 gap。

判据：

- **判为「泛化/探索限制」**：`Δ_mem > 0` 且其 95% CI 排除 0；且 `Δ_ceil` 的 95% CI 含 0 或
  为正（F_T 达到或超过 REF_T）；且 `gap_F > 0` 且其 95% CI 排除 0。
- **判为「优化/表达限制」**：`Δ_mem` 的 95% CI 含 0；且 `Δ_ceil < 0` 且其 95% CI 排除 0
  （F_T 明显低于 REF_T）。
- 中间情形如实给全部数字并说明哪一信号更强；所有差值同时报 win_rate 原始差与
  `400·log10(p/(1−p))` 换算的 vs-random Elo 差（行动门槛参考 +10 Elo、<+5 止损）。

### 0.6 成本/预算与超时策略

6 个 200k 训练（F×3 + G×3）+ 快照/最终/参考 eval。若 200k 不足以区分，可把 F/G 加跑到
500k 并注明（不预注册升级，除非 200k 读数确实歧义）。接近超时时先落盘报告与产物，如实
partial。

### 0.7 Amendment 1（200k 完成后追加，500k 扩展预注册）

200k 读数（见 §3）已是干净 null：F_T≈REF_T≈G_T、F_H≈F_T（无泛化 gap）、无记忆优势。
为关闭「200k 步（线性 LR 到 0）不足以让模型在 500 副上过拟合」这一混淆，追加 500k 扩展：

- 臂：`F500k`（固定池 4001，round-robin）与 `G500k`（新鲜发牌），训练 seed 1/2/3，
  `--total-timesteps 500000`，其余配方与 §0.2 逐项一致（`--snapshot-interval 0`，只取最终
  `agent.pt`）。
- 端点：`F500k`/`G500k` 最终 agent.pt 在 T 与 H 上的 greedy vs-random win_rate/mean_return；
  以及 `Δ_mem500k = win(F500k_T) − win(G500k_T)`、`Δ_ceil500k = win(F500k_T) − win(REF_T)`、
  `gap_F500k = win(F500k_T) − win(F500k_H)`。
- 判据：若 `Δ_mem500k` 仍 CI 含 0 且 `gap_F500k` CI 含 0 → 强化「优化/表达限制」；若
  `F500k_T` 明显超过 `G500k_T` 或 `REF_T` 且 held-out 回落 → 修正为「泛化/探索限制」。

---

### 0.8 Amendment A3（operator 取消 500k 扩展，2026-10-06）

200k 读数为干净 null（见 §2），且仓库已有「vs 固定脚本对手 ~60k 步饱和、之后 1M–2M 不再涨」
的证据（[`ladder-report.md`](./ladder-report.md) §1/§4、
[`elo-breakthrough-report.md`](./elo-breakthrough-report.md) §1.1）。operator 决定不再用
500k 扩展复核同一平台性质，并直接按现有证据出结论。已执行的 500k 部分（固定集/新鲜集各
2 个 seed 完成、fixed s3 训练中被终止、fresh s3 未跑）**全部作废、不用于判定**，其 run 目录
留在 `runs/stage0/fixed-deals/*500k*` 仅作过程记录。原 §0.7 的 500k 判据仅作历史预注册，
不再执行。

---

## 1. 实现与产物

- 固定集包装（只 monkeypatch `seven523.game.Game._deal`，`game.py:144`）：
  `runs/stage0/fixed-deals/deal_pool.py`；
- 池生成：`gen_pools.py` → `pool_fixed.json`（seed 4001，500 副）、`pool_heldout.json`
  （seed 4002，500 副）；指纹集合零交集（`T∩H overlap = 0`）；
- 训练 driver：`train_fixed.py`（F 臂）；G 臂直接用 stock `seven523.train`；
- 评估：`batch_eval.py` + `eval_fixed.py`；逐策略 × {T,H} 输出 500 副 greedy vs-random JSON；
- 分析：`analyze.py` → `analysis.json`（本报告全部数字的唯一来源）；
- run 目录：`runs/stage0/fixed-deals/stage0_fixed__{1,2,3}__*`、`stage0_fresh__{1,2,3}__*`
  （各 199,680 步，rc=0）；
- 参考：`ref_T.json` / `ref_H.json`（`w2m_ctl` 在同一 T/H 池上）。

## 2. 结果（200k，k=3）

### 2.1 最终读数（500 副，greedy vs-random，3 seed 均值）

| 策略 | T win_rate（per-seed） | H win_rate（per-seed） | T mean_return | H mean_return |
|---|---:|---:|---:|---:|
| F（固定 500 副） | **0.8827**（0.870/0.882/0.896） | **0.8780**（0.894/0.854/0.886） | 0.5237 | 0.5184 |
| G（新鲜发牌） | 0.8747（0.860/0.890/0.874） | 0.8873（0.906/0.876/0.880） | 0.5162 | 0.5244 |
| REF（`w2m_ctl`） | 0.880 | 0.906 | 0.542 | 0.555 |

### 2.2 预注册差值（win_rate；deal-cluster bootstrap 4000，k=3）

| 端点 | 点估计 | z-CI | t-CI | per-seed |
|---|---:|---|---|---|
| `Δ_mem = F_T − G_T` | +0.0143 | [−0.0060, +0.0347] | [−0.0303, +0.0590] | +0.015 / −0.001 / +0.029 |
| `Δ_ceil = F_T − REF_T` | −0.0013 | [−0.0211, +0.0184] | [−0.0447, +0.0420] | −0.014 / −0.004 / +0.014 |
| `gap_F = F_T − F_H` | +0.0030 | [−0.0250, +0.0310] | [−0.0585, +0.0645] | −0.025 / +0.022 / +0.012 |
| `gap_G = G_T − G_H` | −0.0173 | [−0.0468, +0.0122] | [−0.0821, +0.0475] | −0.047 / +0.002 / −0.007 |

四个端点的 z/t CI 全部含 0。

### 2.3 快照曲线（F 臂，win_rate，T/H 同步）

| step | 25.6k | 51.2k | 76.8k | 102.4k | 128k | 153.6k | 179.2k |
|---|---:|---:|---:|---:|---:|---:|---:|
| F_T | 0.819 | 0.859 | 0.864 | 0.863 | 0.878 | 0.879 | 0.871 |
| F_H | 0.813 | 0.838 | 0.842 | 0.863 | 0.875 | 0.877 | 0.881 |

F 的 T 与 H 曲线全程重叠，在 ~100–150k 到达 0.86–0.88 平台；G 臂同量级
（完整数据 `analysis.json:snapshots`）。

## 3. 判定（按预注册）

- **H-gen（泛化/探索）不成立**：`Δ_mem` z-CI 含 0（无记忆优势）、`gap_F` z-CI 含 0（无泛化
  gap）；没有「固定集过拟合、held-out 回落」的形态。
- **H-opt（优化/表达）也不成立**（按其严格判据）：`Δ_ceil` 点估计 −0.0013、z-CI 含 0 ——
  F_T 与参考天花板不可分辨，没有「连固定集都明显低于参考」的证据；200k 内 F 已把固定集
  拟合到参考水平。
- **结论：中间情形 / null**。在本探针的尺子（vs RandomBot）上，固定小牌集既不带来拟合优势，
  也不带来泛化代价；所有臂在 ~100k–150k 饱和在 0.86–0.91，尺子本身到顶。因此：
  1. 「连封闭小世界都拟合不了」的强优化失败**被排除**（200k 足够拟合到 REF）；
  2. 「记住了固定集但不迁移」的强泛化失败**被排除**；
  3. 平台的限制在 **vs-random 天花板之上**（相对强对手的强度），本探针无法分辨该层面的
     优化 vs 泛化问题。与 [`ladder-report.md`](./ladder-report.md) §1/§4「~60k 步饱和」一致。

## 4. 缺失证据 / 限制

1. **500k 未测**：按 operator 决定取消（A3），不是实验结果；「200k 不足以过拟合」这一混淆
   没有被 500k 直接关闭（其反证来自既有 ~60k 饱和证据与 200k 已拟合到 REF 的事实）。
2. **尺子饱和**：vs-random 在 ~0.88–0.91 对所有臂失去分辨率；平台层面的优化 vs 泛化需要
   强对手端点（h2h vs 池顶）或不同探针，本报告不作该层面的判定。
3. **k=3**：CI 宽；`Δ_mem` z-CI 上界 +0.035、`gap_F` z-CI 上界 +0.031，不能排除几 %
   win_rate 的小效应。
4. **单池/单参照**：T/H 各一个 500 副池、REF 一个（`w2m_ctl`）；未验证池 seed 与参照特异。
5. **对手随机性保留**：F 固定的是发牌、不是整条轨迹；episode 间对手动作仍随机，因此
   「固定集」不是严格的封闭 MDP。

> 原始产物：`runs/stage0/fixed-deals/`（`analysis.json`、`pool_*.json`、`stage0_fixed__*`、
> `stage0_fresh__*`、`eval/`、`logs/`）。

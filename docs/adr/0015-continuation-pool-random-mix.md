# 续训配方混入 10% RandomBot（池成员）

> **2026-10-07 operator 决策**（本 ADR 固定为现行口径，供后续续训/池实验引用）。
> 适用对象 = 当前续训配方（T23 协议及其派生）；不改变 500k 从零的 random 配方，
> 不重跑任何既有 run。

## 背景

- **续训是当前最强杠杆**：A5/A5b 把 500k 从零的三臂按 T23 协议续训 +1M，相对自身起点
  **+29 Elo**（CI 排除 0），`o2c_base`/`o2c_deep_ln` 对旧冠军 `w2m_ctl` = +10.12/+9.27
  （k=7，z/t-CI 排除 0）。见 [`experiments/depth-normalization-500k.md`](../experiments/depth-normalization-500k.md)
  §3.2。
- **固定池有过度特化信号**：A5 后验探针测到 w2m 血统每多一次固定池续训 entropy 单调塌缩
  （`l1_2M` 1.566 → `t23wswd` 1.308 → `w2m_ctl` **1.082**；同链 random 对照 `w2m_plain`
  1.527、`o2c_base` 1.433），且 `o2c_base` 与 w2m 集群之间存在 ~7 Elo 非传递性
  （同报告 §3.3）：「少做一次固定池续训」更可能是 +10 锚读数的来源，而不是模型全面更强。
- RandomBot 是唯一钉死的评分 gauge（`mu = 0`，[ADR-0012](./0012-single-gauge-and-greedy-removal.md)），
  也是分布覆盖的基准对手；operator 2026-10-07 裁定：**当前配方混入 10% random**。

## 决定

1. **现行续训/池配方在 6 个 t17 成员的池上增加 RandomBot 成员，按局命中率 10%。**
   权重取整数使比例精确（6 个 t17 成员各 `3@`、random `2@`，总计 20）：

   ```bash
   --opponent pool --pool-episode True \
     --pool-member 3@ckpt:runs/t17pool__1__1790439615/snapshots/checkpoint_step262144.pt \
     --pool-member 3@ckpt:runs/t17pool__1__1790439615/snapshots/checkpoint_step499712.pt \
     --pool-member 3@ckpt:runs/t17long__1__1790439615/snapshots/checkpoint_step512000.pt \
     --pool-member 3@ckpt:runs/t17long__1__1790439615/snapshots/checkpoint_step1024000.pt \
     --pool-member 3@ckpt:runs/t17long__1__1790439615/snapshots/checkpoint_step1536000.pt \
     --pool-member 3@ckpt:runs/t17long__1__1790439615/snapshots/checkpoint_step1999872.pt \
     --pool-member 2@random
   ```

   随机局占比 = `2 / (6×3 + 2)` = **2/20 = 10%**；6 个 t17 成员的相对比例保持等权
   （各自 15% vs 原 1/6 ≈ 16.7%）。连续权重形式等价：6 个 `1@` + `0.6666667@random`。

2. **语义**：`--pool-episode True` 下 `EpisodeMixturePolicy` 每局开抽一次、整局冻结
   （`src/seven523/policies.py:106`、`src/seven523/league.py:114`），所以 10% 是**局占比**；
   其余 90% 的局在 6 个 t17 成员间等权抽取。逐决策模式（`--pool-episode False`）下比例按
   决策计，本决定不采用。

3. **`--mix-random-prob` 与本决定无关，禁止用它实现。** 它只在 `--opponent mix` 分支生效
   （`src/seven523/league.py:168-172`、`train.py:452`）；`--opponent pool` 下是惰性参数
   （历史误读及修法见 [`experiments/event-history-pilot.md`](../experiments/event-history-pilot.md) §3.2）。
   把默认 0.5 改成 0.1 不会给 pool 配方混入任何 random。

4. **范围与纪律**：
   - 只改续训/池配方（T23 协议及其派生）；500k 从零的 `--opponent random` 配方不变
     （本来就是 100% random）。
   - 历史 run（A1–A6、T23、w2m 线）不回填、不重跑，既有读数保持各自当时配方。
   - 新配方参与的对照必须**两侧同配方**（同 10% random）；与历史配方的跨批比较只作
     context，不作预注册 gate（沿用 [`plans.md`](../plans.md) §1.2 口径）。

## 考虑过的替代

- **`--opponent mix --mix-random-prob 0.1`**：语义是「冻结自己 + random」的 90/10，会丢掉
  6 成员池——改的不只是随机比例 → 否决。
- **维持 0% random（现状）**：固定池熵塌缩/非传递性的机制读数已有苗头，且过去多次经验
  （event-history 线的 flag 误读）说明「以为混了实际没混」风险高 → 不做。
- **只把 `--mix-random-prob` 默认改掉**：pool 模式惰性，会造成「已混入」的假象 → 禁止
  （ADR-0012 已明确 pool 成员走权重、mix 才走该参数）。
- **更高比例（20–50%）**：无数据支持，且 50% 接近旧的 random 对照配方，会稀释池课程；
  先按 operator 裁定的 10% 执行，比例改动需新决策。
- **`--pool-episode False`（逐决策混）**：破坏按局统计/PFSP 的前提，且现行续训为逐局抽
  → 不采用。

## 后果

- 新续训 run 的池样本量降为 90%，每个 t17 成员的局占比 16.7% → 15%；random 局负责
  分布覆盖与 entropy 维持，其胜负不进 PFSP 的必要性由后续实验决定。
- 池成员序列/随机数流与旧配方不同（成员数、权重都变了），**同 seed 也不可逐位对齐**，
  只能按统计口径（h2h/合并 CI）比较。
- 若未来启用 `--pfsp`，需要单独决定 random 成员是否参与 PFSP 重加权；现行配方
  `pfsp=False`，本 ADR 不覆盖该情形。
- 文档面：`docs/training.md` 的池示例、`experiments/warmstart-adamw-1m.md` §5（历史复现）
  与 `plans.md` §0/§1.3 的 ADR 登记同步引用本文。

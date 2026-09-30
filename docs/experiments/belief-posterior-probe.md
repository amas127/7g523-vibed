# ② 隐藏手牌学习后验：离线 ROI 探针 — 负结果（停止，不集成）

> **状态**：完成（2026-09-29）。**主门/次门均 FAIL，模型未过 count-only 基线（K1）**；按预注册
> **停止，不做推理侧集成、不跑 h2h**。
> **口径**：revision-3（`rules_id=2e36dbea44893696`）、obs v5（2 家 161 维）、trace `version=3`、
> t=5/K=32/C=6、value `t_leafq`、rollout 对手 `w2m_ctl`。全程离线，无 h2h、无 `src/` 改动。
> **预注册/产物**：`runs/belief_posterior_probe/{preregistration.md,report.md,report.json,model_metrics.json,anchor.json,roi.json,leakcheck.json,posterior_health.json}`。

## 0. 结论（TL;DR）

1. **学习后验在"采样质量"上勉强有信号，但决策无收益**：held-out seed 上逐牌 AUC 0.742
   （均匀 0.5），top-k 放置 recall 0.457；但**低于纯计数基线 count-only 的 0.748/0.465**
   （K1 kill 触发）——公开历史特征相对"数牌"没有增量。
2. **人类 bank ROI（370 searched，K=32/t=5）主门 FAIL**：学习后验 vs 均匀的配对 chosen-value
   Δ = **−0.217** 分/决策 [−0.589, +0.160]；负对照（二次均匀重采样）−0.021 [−0.445, +0.412]；
   **oracle 真手牌仍 +3.117** [+2.019, +4.271] —— 说明该 bank 有能力检出正确世界带来的价值，
   只是学习后验没做到。
3. **次门 FAIL**：翻转率 posterior **22.2%**、count-post 24.1%，**低于**均匀重采样噪声 23.5%；
   oracle 37.6%（McNemar 84:32，p=1.5e-6）。gap recovery ≈ −0.10。
   "用 oracle 真值评估后验所选动作"的价值 Δ 也是 −0.759 [−1.831, +0.343]。
4. **锚点在同一 harness 复现**（方向一致、幅度有差）：uniform-K8 33.2%（历史 28.4%）、
   oracle-hand-K8 42.7%（40.3%）、true-state-K1 45.7%（44.3%）；oracle > uniform p=0.0019。
   生产回放 470/479 行一致（searched 361/370；9 个末尾近似平局差异）。
5. **含义**：均匀采样器已经隐式利用了"数牌"信息（从未见池划分、强制明牌约束）；公开历史对
   **隐藏手牌**的额外可推断信息在本模型类/本数据规模下不足以改变决策。与对手意图探针
   （[`opponent-intent-probe.md`](./opponent-intent-probe.md)，G-I FAIL）互相印证：公开历史里"行为推断"信号弱、且被 v5 观测覆盖。
   **重访条件**：held-out top-k placement 优势显著变大（更强历史编码/更多数据），或换问题
   （例如只做局部/端局的精确算牌）。**不集成 `rolloutt:`、不跑 h2h。**

## 1. 设计与数据

- **训练数据**：`traces/study` 4000 局（revision-3），195,066 个 acting-seat 决策，400 unique seeds；
  **按 seed 切分**（320 train / 80 val = 156,430 / 38,636 决策；同一 seed 跨等级同发牌，禁止按局切）。
- **特征（全公开）**：327 维 = count(15) + score(6) + 对手内容(182) + 自身内容(70) + 自己手牌
  multihot(54)，加牌嵌入(16) + 5 个标量；标签单独存放（`state.hands[opp]` 仅作 label/oracle）。
  泄漏自检：把 `state.hands` 打乱后特征逐位不变、label ⊆ 未见池、池 == `rp.unseen_pool`。
- **模型**：逐未见牌 logit + 张数约束采样（Gumbel-top-k，T=0.5 为 val 最优）。
- **ROI bank**：`traces/web/run-20260928-091716` 的 22 局真人 vs `w2m_ctl`（479 bot 决策，
  370 searched；第 23 局 vs random 剔除）。生产搜索用 `rp.batched_decide`，世界来源四选一：
  uniform1 / uniform2（噪声下限）/ 学习后验 / oracle 真手牌。

| 模型（held-out seeds） | AUC | Brier | top-k recall | 采样放置 |
|---|---|---|---|---|
| full（327 公共特征） | 0.742 | 0.1386 | 0.457 | 0.439 |
| count-only | **0.748** | **0.1372** | **0.465** | 0.446 |
| uniform | 0.500 | 0.1388 | 0.423 | — |

## 2. ROI 结果（370 searched，K=32，t=5，C=6，t_leafq，P0 rollout 对手）

| 世界来源 | 翻转率 vs uniform1 | 配对 chosen-value Δ vs uniform1（分/决策） |
|---|---|---|
| uniform2（负对照/噪声下限） | 23.5% | −0.021 [−0.445, +0.412] |
| 学习后验 | 22.2% | **−0.217 [−0.589, +0.160]** |
| count-post | 24.1% | — |
| oracle 真手牌 | 37.6% | **+3.117 [+2.019, +4.271]** |

- 人类 bank 上的放置质量：后验 0.447 vs 均匀 0.414，且 24/32 世界不重复（非退化）——
  即"采样更准"与"决策更好"在本问题上是**分离**的。
- 主门（预注册）：后验 vs 均匀配对 value Δ CI 排 0 且点 > 0 → **FAIL**。
- 次门：翻转率高于噪声下限并恢复一定比例 oracle gap → **FAIL**（−0.10）。

## 3. 评测完整性 / 已知局限

- **分布不匹配**：训练为脚本对手（lvl1–lvl4）自弈，ROI 是真人局面（OOD）；不过主门 FAIL
  是"连脚本分布上的计数基线都没超过"，OOD 不能解释全部。
- 单 seed/单架构（full head 在 epoch 1 即过拟合)；单个人类 bank/玩家/日期；
  one-step champion continuation 真值 ≠ 全局对局价值；离线 ≠ Elo。
- 生产回放 9 个末尾近似平局与记录不一致（历史记录本身是旧引擎的，非本探针误差）。

## 4. 证据落点

- 预注册：`runs/belief_posterior_probe/preregistration.md`
- 报告/原始：`.../{report.md,report.json,anchor.json,roi.json,model_metrics.json}`
- 模型/自检：`.../{model.pt,leakcheck.json,posterior_health.json}`
- 数据构建/训练/探针：`.../{build_dataset.py,train.py,common.py,dataset.npz}`

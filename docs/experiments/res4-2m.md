# 小模型 res4：无 norm 4-block 残差 body + plain 三头，hidden 64，2M 从 random（k=1）

> **状态：已完成（2026-10-08，k=1 = seed 1）。主读数：2M 后落在 raw 顶簇——对同预算
> `t17long` 2M **+28.29** [+16.24,+40.35]、对 `w2m_ctl` **+18.55** [+7.04,+30.05]、
> 对 `head32ln5m` **−5.50** [−17.10,+6.09]（统计上平手）。**
>
> **重要边界**：这条对比是「新默认配方（arcsin+跳变+LR floor）+ res4 h64 + plain 头 +
> 胜负头」对「旧 T17 配方（terminal、LR 退到 0）+ shared h128 单头」，**不能归因给架构里的
> 任何单一因素**；k=1 未复现，按统一口径标记为待补 seed 2–3。
>
> **operator 请求（2026-10-07）**：body 深度 4 带残差、头不带残差（策略 2 + 价值 1 +
> 胜负 1）、无 norm、hidden 64（128 减半）、第一次训练从 random 打、预算 2M、开 TensorBoard。
>
> **口径**：revision-3 + obs v5 + [ADR-0016](../adr/0016-default-training-recipe-arcsin-lr-floor.md)
> 默认配方；对手 = `random`（不混池）。跨 fit 绝对 Elo 不可比，本页只给同 fit h2h。
>
> **产物**：`runs/res4/`。**代码**：`src/seven523/networks.py`（`res<d><a><c>` 家族）、
> `src/seven523/train.py`；测试 `tests/test_{networks,train}.py`。

## 0. 架构（`res421` + `--vf-outcome`）

```
x (161)
  └─ ResidualTrunk（depth 4，无 norm，共享）
        block0      : Linear(161→64) + ReLU
        block1..3   : h = h + ReLU(Linear(64→64))
  └─ h (64)
        ├─ actor   : PlainHead [Linear(64→64)+ReLU] → Linear(64→138)
        ├─ critic  : Linear(64→1)
        └─ outcome : Linear(64→1) → tanh
```

- 总参数 **36,108**（body 22,848 + actor 13,130 + critic 65 + 胜负头 65）。
- **三个头读同一个 64 维 hidden**（共享 body；非 `towers`）。`get_action_and_value` 一次
  trunk 前向同时给 actor/critic；`get_value_and_outcome` 再前向一次给 critic/outcome——
  无 norm/dropout，两次前向数值逐位一致（实测 equal）。
- 胜负头 loss 与 actor/critic loss 一样回传到共享 body（不做 detach）；若换 `towers`，
  outcome 头会挂在 critic tower 上而不是共享 body。
- 实现：`ResidualTrunk(blocks=d, norm="none")` + `PlainHead` + `res<d><a><c>` 解析；
  测试：`test_res4_body_depth_and_plain_heads`、`test_res421_run_config_with_hidden_64_and_outcome_head`、
  `test_train_res4_vf_outcome_smoke`。全套 883 passed / 14 skipped（一个既有 manifest 漂移
  失败与本项无关，已 deselect）。

## 1. 训练

配置：seed 1、`--opponent random`、2M 步、8×128（batch 1024 / minibatch 256）、4 epochs、
Adam 2.5e-4 线性退火（floor 1e-5）、arcsin α=0.5 + λ=1、`--vf-outcome true`、
`--actor-out-std 0.1`、hidden 64、TensorBoard 开、`--eval-interval 50`。

- 00:44:34 → 01:13:14（约 29 分钟），1953/1953 updates = 1,999,872 步，**无 NaN**，rc=0。
- 末 150 updates 均值：EV 0.6645、entropy 1.6103、value_loss 0.0137、胜负头准确率 0.9073；
  末行 EV 0.7208 / entropy 1.6248 / outcome_loss 0.1447 / outcome_acc 0.9543。
- 训练内末次 vs-random eval（100 局）：return 0.51、score 75.5、diff 51.0、win 0.88。
- 独立 vs-random eval（1000 局，gpu）：**win 0.9090**、score 79.18、diff 58.36；
  对照 `t17long` 2M：win 0.8840、score 77.12、diff 54.24。

## 2. h2h 面板

协议：3 deal seed（9600/9601/9602）× 400 副 × 换座（1200 副 / 2400 局），bootstrap 4000：
`res421_h64_2m − 对手`：

| 对手 | Δ Elo | 95% CI | winrate | per-seed |
|---|---:|---|---:|---|
| `t17long` 2M（同预算/同对手，旧配方 + shared h128） | **+28.29** | **[+16.24, +40.35]** | 0.541 | +30.9 / +30.5 / +23.5 |
| `w2m_ctl`（2M pool，旧顶簇） | **+18.55** | **[+7.04, +30.05]** | 0.527 | +17.0 / +17.4 / +21.3 |
| `o2c_base`（续训顶簇） | +9.85 | [−1.42, +21.12] | 0.514 | +2.6 / +13.9 / +13.0 |
| `vfo_outcome`（500k pool 三头 h128） | +19.90 | [−1.99, +41.79] | 0.529 | +41.0 / +3.0 / +15.6 |
| `head32ln5m`（当前最强 raw，8.5M 续训） | −5.50 | [−17.10, +6.09] | 0.492 | −13.0 / +3.0 / −6.5 |
| null：`t17long` final − 2M snapshot（两文件逐字节相同） | 0.00 | [0, 0] | 0.500 | — |

## 3. 读法

1. **位置**：36k 参数、hidden 64、只打 random 的 2M 模型落在当前 raw 顶簇——与
   `head32ln5m` 统计上打平，对 `w2m_ctl`/`t17long` 三个 deal seed 同号且 CI 排除 0。
2. **不能归因**：对比里混了配方（arcsin+跳变+LR floor vs terminal+退火到 0）、宽度
   （64 vs 128）、body（res4 vs shared）、头（plain 2/1 vs 单层）与胜负头五个因素；
   同配方 500k 的家族对照（[gnres-500k.md](./gnres-500k.md)、[bnres-500k.md](./bnres-500k.md)、
   [vf-outcome.md](./vf-outcome.md)）全是 null，所以「2M 尾巴的 LR floor / 配方」比「res4
   结构」更像来源。要拆开必须跑匹配对照（见 §4）。
3. **交叉验证警告**：历史 μ 估计（`t17long`≈190、`w2m_ctl`≈191.5、`head32ln5m`≈198.6）与
   「res421 平 head32ln5m、却领先 t17long 28」相差 10–15 Elo，说明单面板的大读数含风格/
   非传递性成分；同 fit 直接 h2h 是正确读法，但这类 +20~+30 的读数必须多 seed 复核。
4. **k=1**：按统一口径（CI 排 0 且点 ≥ +10 → 标记复现），vs `t17long` 与 vs `w2m_ctl`
   两条满足标记；不宣布 H1。

## 4. 下一步

1. **seed 2–3 复现**同一条 `res421_h64_2m`（2M × 2，各 ≈29 分钟，可并行）。
2. **拆归因匹配对照**（同新配方、同 2M、同 random）：
   - `shared` h128、无 `--vf-outcome`：分离「配方 vs 结构」；
   - 可选 `res421` h128：分离宽度；`res421` h64 无 `--vf-outcome`：分离胜负头。
3. 若复现：走续训（+1M pool，ADR-0015 配方），看能否越出顶簇（续训是另一个已知正杠杆）。

## 附注 A：三头接线（2026-10-08 提问）

`res421` 是共享 body：actor / critic / outcome 三个头都从**同一个** 64 维 hidden 读出
（`networks.py` 的 `_actor_hidden`/`_critic_hidden` 在非 towers 时都返回 `self.network(...)`）。
`get_action_and_value` 一次 trunk 前向把同一 `hidden` 给 actor 和 critic；
`get_value_and_outcome` 再前向一次给 critic 与 outcome。无 norm/dropout 时两次前向逐位一致。

## 附注 B：actor 输出层本来就是 138 一维（2026-10-08 提问）

`self.actor` 的输出层是**单个** `Linear(hidden, 134+4=138)`（`networks.py:691`），
`policy_logits` 直接返回 `(B, 138)`。所谓「双头」只在分布层：`torch.split` 成 `(134, 4)`
两个 `CategoricalMasked`（模板头掩码、花色头全开，ADR-0004），logprob/entropy 按头相加。
把分布合并成一个 138 路 categorical **不是无害的**：模板与花色 logits 会互相抢 softmax 质量、
采样会落到花色位并解出非法模板、PPO 的 logprob/entropy/ratio 全换定义。真正的单头选项是
**134×4=536 路联合头**（联合掩码、输出层 +51k 参数）或**砍掉花色头退化成 134 单头**，
两者都是语义/架构变更而非等价重构。

## 附注 C：归一化线的当前结论（同批对话）

LN = null 到负（`deep_ln−deep` +1.54、`deep_ln−base` −13.95 k=7、`head32ln` 系列无改进）；
GN vs LN 两轮都不可分（+0.43 / +5.07）；BN 在朴素 PPO 配方下强负（−42）。归一化类型不是
强度杠杆；真正有交互的是输出层 init（`actor_out_std` × 头型，plain 深头 0.1 / 残差头 0.01）。

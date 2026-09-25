# 观测布局 v5：观测 S1 + B0 + B1 成为默认，v4 保留为前缀链中间版本

观测 S1 的静态审计与 200k 从零 pilot（[observation-slimming.md](../experiments/observation-slimming.md)
§2–§5）表明，原 191 维布局里有一批可推导或低效表达的维度：`rank_counts` 完全由 `hand` 决定；
`current` 在全部生产路径里恒等于 acting seat；自身 `hand_counts` 恒等于 `len(hand)/7`；
`incumbent_top`/`revealed` 的 54 维 one-hot 实际只承载 rank+suit 的 19 维信息。观测 S1 按信息
无损重编码把这批维度删除/压缩后为 103 维（2 家，`61 + 21n`），200k pilot 中观测 S1 与 full 191
不可分辨（合并 **+4.4 [−18.3, +27.0]** Elo），观测 S2（68）落后 full ~15 Elo 并已否决
（plans §6 N-6）。增广方向 B0（`obs_version=2`，追加 `trick_points`/`remaining_points`/`point_hold`
3 维）与 B1（`obs_version=3`，追加 `unseen` 54 + `last_player` 1）都已实现并各自完成 500k 单臂：

| 方向 | 端点（同牌换座 h2h） | 结果 | 结论 |
|---|---|---|---|
| 观测 S1 103 | 200k pilot vs full 191 | +4.4 [−18.3, +27.0] | 不可分辨（无损失信号） |
| B0（v2） | 500k vs `base680k` / `w5_ctrl` | +1.74 [−7.94, +11.42] / +4.92 [−6.72, +16.56] | 均跨 0（null） |
| B1（v3） | 500k vs `base680k` / vs B0 | +4.20 [−6.75, +15.15] / +0.73 [−18.98, +20.43] | 主端点失败、未确认 |

B0/B1 单独采用都没有证据，但「观测 S1 + 增广」与「单独增广」是两个问题：观测 S1 省下的首层
预算正好覆盖 B0+B1（`103 + 58 = 161 < 191`），而 B1 的 `unseen` 补的正是
[structural-directions.md](../experiments/structural-directions.md) §1.2 指出的最大信息缺口。
plans 的 **D-4** 已裁定「与增广一次迁移」（2026-09-25）：不要先单独动默认布局、再做第二次
`obs_version` 迁移。v2 落地时已确定 slim 家族从 v4 起编号（B1 占用 v3 = 249），即
v4 = 观测 S1+B0、v5 = 观测 S1+B0+B1。本 ADR 落实该裁决。

## 决定

1. **默认布局切到 v5 = 观测 S1 + B0 + B1 = `119 + 21n`（2 家 161 维）**。段序（2 家宽度）：
   `hand`54 → `inc_rank`15 → `inc_suit`4 → `inc_kind`6 → `inc_size`1 → `draw`1 → `scores`n →
   `opp_count`n−1 → `opp_revealed`19(n−1) → `trick_points`1 → `remaining_points`1 →
   `point_hold`1 → `unseen`54 → `last_player`1。其中 `scores`/`opp_count`/`opp_revealed`
   自中心旋转（`(seat+k) % n`）；`unseen`/`last_player` 原样复用 v3 的 B1 实现
   （`last_player` 在 2 家是冗余维，保留以向前兼容 N 家）。
2. **同时落地 v4 = 观测 S1 + B0 = `64 + 21n`（2 家 106 维）** 作为前缀链中间版本：
   v4 = 上表前 12 段，v5 = v4 + 后 2 段；**v4 是 v5 的逐位前缀**。
3. **保留 v1（`185+3n`，2 家 191）、v2（`188+3n`，194）、v3（`243+3n`，249）兼容**：
   ckpt payload 存 `obs_version`（旧 ckpt 缺省 v1），`NeuralPolicy` 按 ckpt 版本选编码器，
   旧 ckpt 在 v5 工具链下继续可评。v1–v3 与 v4/v5 是重排 + 压缩关系，**不互为前缀，禁止逐列
   互读**；版本号是 ckpt 身份的一部分，不能仅凭宽度推断（v1 3 家与 v2 2 家同为 194）。
4. **热启动允许集与段级列重映射**：首层按段名做列重映射（`segment_spans(v, n)` 由段表单一
   真源推导）；允许的前向集合为 `{(1,2), (2,3), (1,3), (1,4), (2,4), (4,5), (1,5), (2,5)}`。
   其中 `(1,2)`/`(2,3)`/`(1,3)`/`(4,5)` 是逐位前缀填充（新列置 0），其余走重映射；
   trunk/actor/critic 形状相同仍逐位照拷。
5. **已知近似**：`incumbent_top` 54→19（`inc_rank`+`inc_suit`）与 `revealed` 54→19（仅对手
   `opp_revealed`）的 rank+suit 分解不是旧函数的等值变换；`scores` 的自中心旋转对旧训练布局
   （learner 恒为 seat 0）是逐位等价的置换，但 seat 1 的旋转是分布外。被删段（`rank_counts`、
   `current`、自身 `hand_counts`、自身 `revealed`）的权重不拷贝，新列由训练学。
6. **不可映射方向显式拒绝**：v4→v3/v2/v1、v5→v4/v3/v2/v1 等所有回退，以及所有跨玩家数组合，
   在 `train.py` 加载点报错（`SystemExit`），不允许静默跳过首层。`warm_start_into` 本身保留
   「不可映射就跳过首层」的旧语义（旧测试与 T7 的 2→3 家拒绝路径依赖它）。

## 考虑过的替代

- **只落 v4（观测 S1+B0）**：B1 虽未确认，但已是段级追加（v4→v5 只需 55 列置零），保留它使
  v5 与既有 v3 的 B1 臂可直接对照；拆成两次迁移违背 D-4。
- **只落 v5、不做 v4**：v4 是 v5 的逐位前缀，保留它给消融/热启动一条确定的中间链，成本只是
  多一张段表与少量测试。
- **直接把默认布局换成观测 S1 而不保留版本参数**：`runs/` 里的旧 ckpt 是全部评测/梯队工具的
  输入，OS §6 已论证版本参数的成本远低于一次性失效。
- **跨版本加载时静默跳过首层**：会给新布局一个随机首层且不报错（实现分析 §2 的失效模式 2），
  显式拒绝把错误暴露在启动时。

## 后果

- 默认观测 **249 → 161**（v3→v5）；首层参数 32,000 → 20,736、总参数 66,443 → 55,179
  （实现分析 §1 表）。v4（106）保留为前缀中间版本；`rank_counts` 回退仍按 D-7，仅在 500k 追认
  出现样本效率问题时另行考虑。
- 旧 ckpt 以自身 `obs_version` 推理，评测/对局工具经 `policy_from_spec`/`NeuralPolicy` 自动
  dispatch；跨版本热启动到 v5 的首层含近似段，trunk/heads 仍逐位，两者语义必须区分。
- remap 的 54→19 近似是否造成不可分辨的初始凹陷仍需 500k 追认（实现分析 §6）；若怀疑，加
  「旧布局冷首层」对照臂即可判别。
- plans T4 状态更新为「已实施 v5（2026-09-25）」；D-4 关闭；`--obs-version` 取值 1..5、
  默认 v5。

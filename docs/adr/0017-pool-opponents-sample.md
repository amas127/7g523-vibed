# pool 训练对手默认采样（`--pool-sample`）

> **2026-10-07 operator 决策**（固定 `--opponent pool` 的默认对手行为，供后续训练/实验引用）。
> 历史 run 不回填、不重跑；复现旧的 greedy pool 需显式传参（见「决定」）。

## 背景

- [ADR-0015](./0015-continuation-pool-random-mix.md) 的现行续训/池配方（6 个 t17 成员 `3@`
  + `2@random`，`--pool-episode True`）里，池成员的动作是 `NeuralPolicy` 的**默认 argmax**：
  `policy_from_spec` 的 `ckpt:` 分支不带 `sample`，`self` 成员由 `--self-play-sample` 控制
  （默认关）。也就是说同一状态/同一牌局下静态池成员是确定性的，只有 learner 在按 mask 采样。
- operator 2026-10-07 指定：**pool 训练对手改成默认采样**（增加对手动作分布的多样性、
  减少固定池的逐状态确定性）。

## 决定

1. 新增 `--pool-sample`（bool，默认 **True**）：`--opponent pool` 下，池成员（`ckpt:` 与
   `self`）的动作按各自 masked policy **采样**而不是 argmax。
2. `--pool-sample false` 恢复历史行为（greedy pool），因此旧配方可逐位复现
   （`args.json` 会记录该字段）。
3. 语义边界：
   - 只影响训练 league 的 pool 模式；`--opponent self`/`mix` 仍由 `--self-play-sample`
     （默认 False）控制，互不覆盖。
   - **评估/定级链不受影响**：`policy_from_spec` 新增的 `sample` 参数默认 False，
     `7g523-eval`、ladder、h2h/duel、web、placement 全部保持确定性 argmax。
   - `random` 成员本来就是随机策略，不受影响。
4. 落地：`LeagueConfig.pool_sample=True`（默认）；`build_league` 对 `ckpt:` 成员传
   `sample=config.pool_sample`、对 `self` 成员在 pool 模式下同样使用它；
   `policy_from_spec(..., sample=False)` 为关键字参数（旧调用不变）。

## 考虑过的替代

- **改 `policy_from_spec` 的全局默认**：会污染 ladder/h2h/web 的确定性评估口径（ADR-0006/0012
  的 rating 链要求同一模型可复现）→ 否决。
- **只改 `--self-play-sample` 默认**：只覆盖 `self`/`mix`，静态 `ckpt:` 池成员仍是 argmax，
  不满足 operator 要求 → 否决。
- **不加开关、直接改行为**：`args.json` 无法区分新旧 run，旧配方无路复现 → 否决。

## 后果

- 新 pool run 的对手动作流与历史批不同：同 seed 也不可逐位对齐；与历史 run（A1–A6/T23/w2m）
  比较只能按统计口径（h2h/合并 CI），不得当同分布重跑。
- 采样走全局 torch RNG（`NeuralPolicy` 的 `seed` 参数不约束采样路径，与既有
  `--self-play-sample` 行为一致），训练内可复现性依赖进程级 `torch.manual_seed` 与固定调用
  顺序；本 ADR 不改变这一点。
- PFSP/逐局冻结的身份与权重不受影响：采样只改变成员的动作噪声，per-episode 成员抽取不变。
- 文档面：`docs/training.md` 的池段落与 `tests/{test_league,test_train}.py` 的默认断言同步；
  `plans.md` §0 ADR 登记扩到 `0017`。

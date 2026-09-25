# 评分与阶梯接缝：`elo.py` 拥有结果评分，`ladder.py` 拥有对局编排

M2 需要给一组 ckpt 定级（4–6 级、相邻 100–150 Elo、Random/Greedy 锚点固定），
D3 需要在同一评分口径上长出「窗口结果似然 + 轨迹先验」的混合估计器。测量流程由
`tools/measure_trace_signal.py`（D1）离线验证轨迹信息量，但它只做特征→Elo 标定，
不是评分器。这套评分数学此前只存在于未提交的 HTML 原型里。决定：

- **`elo.py` 是评分核心（deep module，纯 in-process）**：`fit_ratings` / `select_rungs` /
  `expected_score` 三个入口，藏 Bradley–Terry MAP、阻尼牛顿（步长上限 300、钳制
  [400, 2600]）、先验、逐 id 窗口、可选分差高斯似然与 Fisher 信息 SE。不依赖 torch、
  numpy、文件、RNG 或时钟——观测单位只有「一局」（`PlayedGame`）。
- **`ladder.py` 是对局编排**：`plan_games`（纯函数，成对牌局且每副牌对每个候选
  双座位互换、每级对 ≥2 个锚点各 ≥N 局、N 为偶数）、`play_games`（经 `play_game`
  录制，轨迹仍是 `trace.py` 格式，与人类对局同一 artifact）、`build_ladder`
  （plan→play→fit→select 的默认组合）。
- **`study.py` 是 `manifest.json` 的 schema 唯一 owner**：`load/merge/save_manifest`，
  `levels`（id → 测量 Elo）是 D1 消费的契约；评分默认冻结（`refit=False`），重跑需显式 `--refit`。
- **`policies.policy_from_spec`** 是 `random` / `greedy` / `ckpt:<path>` 语法的唯一解析处，
  torch 只在 `ckpt:` 分支惰性导入。
- D3 通过 `priors` 数据进入同一个 `fit_ratings`（轨迹标定给 `(μ_traj, σ_traj)`），
  **不新增接缝**；`7g523-elo` CLI 后置到 M3。

## 考虑过的替代

- **全部塞进一个 `tools/` 脚本**：D3 无法复用评分数学，工作流噪音会淹没核心；且工具之间
  会各自复制一份 spec 解析与 manifest 读写。
- **评分放进 `measure_trace_signal.py`**：特征标定是对「特征」的回归，结果评分是对
  「一局胜负」的似然，两种观测混在一起会产生第二套真相。
- **现在就暴露 Schedule/Pairing/Observer 类型、Likelihood Protocol、Storage Port**：
  每个只有一个调用者，属于假设接缝；等第二个真实 adapter 出现再加。
- **把 `expected_score` 藏起来**：D3 的信息量最优配对手需要它，且它是无状态纯函数，
  不构成额外真相。
- **评分模块直接读文件/写 manifest**：会让纯数学依赖本地 I/O，测试被迫绕路；
  编排负责 I/O，数学只吃数据。

## 后果

- 观测单位固定为「一局」，SE 天然按局整群；若 S2 的 regret 要以**决策级**证据进入联合似然，
  那需要新模块而不是给 `fit_ratings` 加参数。
- 结果似然始终是最终权威；轨迹先验只提高前 n 局的精度，不覆盖结果。
- 锚点评分固定且 `se = 0`；新 ckpt 的 warm start 通过显式 `Prior` 数据表达，而不是隐藏行为。
- 本接缝与 ADR-0005 一致：对局循环仍只有 `Match`，轨迹格式仍只有 `trace.py`；
  `ladder` 只是把二者与评分串起来。

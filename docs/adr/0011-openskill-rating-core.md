# 评分核心改用 OpenSkill：标准高斯 Plackett–Luce 替代手写 BT-MAP

> **部分取代（2026-09-26）**：[ADR-0012](./0012-single-gauge-and-greedy-removal.md)
> 已把标度原点从 random=1000（`DEFAULT_MU=1500`）改为 RandomBot=0（`DEFAULT_MU=500`）
> 并删除 GreedyBot（本文 decision 2 的 1500/1000/1315 为当时值）；[ADR-0013](./0013-drift-free-rating-channel.md) 取代
> 本文 decision 1 中「不建第二估计器 / 发布依赖在线重放」的部分（发布绝对表改走独立
> `mle.py`）。**在线接缝（`fit_ratings`/`select_rungs`/`expected_score`、`Prior`
> 注入点、`tau>0`、anchor `sigma=0`）仍是现行实现。**

`elo.py` 自 ADR-0006 起是一套手写的 Bradley–Terry MAP：400 点 logistic 似然、阻尼牛顿
迭代、逐 id 窗口、可选分差高斯似然、联合 Hessian SE、`[400, 2600]` 钳制。它背着两处
已确认的缺陷修复（`window` 交叉项近似、free-free 重复计数）和 HR §5.3 的 margin 标定，
但本质仍是自家维护的估计器：N 家、动态、分差每来一个新需求都要在这套数学里再开一个口子，
SE 也只在无窗口时才精确。

决定：`elo.py` 的估计器换成标准 OpenSkill（`openskill.models.PlackettLuce`，Weng–Lin
高斯模型，纯 Python、无传递依赖），**同一个 deep seam 不变**——`fit_ratings` /
`select_rungs` / `expected_score` 三个入口，纯 in-process、无 torch/numpy/IO/RNG/时钟。

1. **在线重放代替联合 MAP**：`fit_ratings(games, *, anchors, priors=None, config)`
   按时间顺序对每局调用一次 Plackett–Luce 更新。锚点是被钉死的 `(mu, sigma=0)`；自由 id
   以 `Prior(mu, sigma)` 为初始高斯；`tau`（openskill 默认 `sigma/100`）提供动态、取代
   旧的逐 id `window`。单遍重放，结果依赖对局顺序（旧 MAP 顺序无关）——这是 OpenSkill
   的标准批拟合语义。
2. **保留项目标度**：模型配置 `mu=1500, sigma=200, beta=sigma/2, tau=sigma/100`。
   与 openskill 默认 `(mu=25, sigma=25/3, beta=25/6, tau=25/300)` 相比，所有影响
   概率的差量参数一律 ×24（`mu` 只是原点，取 1500 即可）。锚点
   （random=1000/greedy=1315）、`select_rungs` 间距（100–150）、`stop_ci`、
   `rung_prior_sd` 等存量数字继续有意义。概率模型换成高斯：`expected_score` 就是 PL 的
   `predict_win`（`Φ(Δμ / √(2β² + σ_a² + σ_b²))`），不再是 400 点 logistic。
   **后续 [ADR-0012](./0012-single-gauge-and-greedy-removal.md) 已把原点改为
   RandomBot=0 并删除 GreedyBot；本文的 1500/1000/1315 是当时的历史标度。**
3. **删除 BT-MAP 专有件**：`margin` 分差似然、`window`、`elo_scale`/`step_cap`/
   `rating_min`/`rating_max`/`max_iter`/`tol`、`Fit.iterations`/`converged`、联合
   Hessian SE。`Fit` 只报 `ratings` 与 `games`；`Rating` 暴露 `mu/sigma/n`（`sigma` 是
   OpenSkill 不确定度，不是旧 Hessian SE）；`Prior` 字段改为 `mu/sigma`；`Rung` 改为
   `mu/sigma`。
4. **N 家免费合法**：`PlayedGame` 本来就是 N 座位形状；PL 的多人更新直接支持，按分数
   排序、同分同名次。两人阶梯与竞技场行为不变，T7 的 >2 家不必再为评分单独造轮子。
5. **产物 schema 跟着改**：manifest 的 `subjects[].mu/sigma/games`、`rungs[].mu/sigma`、
   `anchors[].mu`，arena/placement 报告的 `mu/sigma`，以及
   `estimator.kind = "openskill-plackett-luce"`。旧 `elo/se` manifest 由
   `load_opponents` 明确报错、要求用 `tools/build_ladder.py` 重测。ADR-0006 关于
   「评分是纯函数、结果似然是最终权威、`priors` 是 D3 唯一注入点、不新增接缝」的承诺
   全部保留；先验产物 `prior.json` 的 `{mean, sd}` schema 与轨迹 S1 特征词表不动
   （它们是先验层的 schema，不是评分器的）。

## 考虑过的替代

- **继续维护 BT-MAP，只加 N 家**：多人 Plackett–Luce 要自己实现并验证；OpenSkill 已经
  提供标准模型与数值细节，且不引入 numpy/scipy 依赖（`elo.py` 的纯度约束不变）。
- **采用 openskill 原生 25/8.33 标度**：锚点、间距、`stop_ci`、T2 标签与 `prior.json`
  全部要换算或重标定，收益只是数字更「原教旨」；保留 1500/200 让下游常量与人类可读性
  不漂，代价是必须记住这不是旧的 Elo。
- **在上层模拟 `margin`/`window`**：OpenSkill 没有分差似然；另设第二套似然或逐 id 裁剪
  会制造第二真相，并与「标准高斯模型」冲突。`PlayedGame.scores` 仍保留分差，只是不进
  评分。
- **把 `expected_score` 留在 400 点 logistic**：更新用高斯、选档用 logistic，会让 D3 的
  信息最优配对与评分器不是同一个概率模型；改用 `predict_win` 保证单一真相。
- **改用 `BradleyTerryFull`**：它是双人特例；引擎与 `PlayedGame` 已是 N 座位形状，
  `PlackettLuce` 是 openskill 的默认通用模型。
- **沿用 `Rating.elo`/`.se` 字段名**：会把 `mu`/`sigma` 读成别的东西；字段随标记方案
  一起改成高斯词表。

## 后果

- **绝对分与旧 `elo/se` 不可比**：锚点仍是外部约定的 1000/1315，但自由 id 的
  `(mu, sigma)` 来自另一套似然。10 局 CI、`stop_ci`、T2 标签、`channel_weights` 与
  `prior.json` 都标定在旧估计器上，需随 `traces/study` 重测一起重标（并入 T15）；旧
  manifest 在重测前会被 `load_opponents` 拒绝。
- **`sigma` 语义不同**：它是 OpenSkill 不确定度（有 `tau` 地板、收缩更快），不是
  Hessian 标准误；报告与测试按 `sigma` 解释，不再承诺旧 SE 的推导口径。
- 评分核心仍是纯函数：无 RNG/IO/torch/numpy，单测不需要训练栈；`select_rungs` 的间距
  缺口照旧用 `ok`/`wide_gaps`/`tail_gap` 显式上报；`ladder` 仍要求 ≥2 个 pinned 锚点；
  D3 仍以 `priors` 数据进入，不新增接缝。
- 旧计划里的评估工程项：T11（`window` 交叉项近似）随 `window` 删除而失去对象；T9
  （cluster bootstrap）与 T10（z/confidence）与估计器无关，保持原样。
- 依赖新增 `openskill>=6.2`（无传递依赖）；`elo.py` 的默认导入仍不拉起 torch/numpy。

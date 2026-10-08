# 空底精确 minimax 审计 + endgame solver 接入（2026-10-08）

> **状态**：审计与接入已完成，逐位可验证（确定性精确解 + 122 项检查 + 932 passed）。
> **强度未确认**：本报告不含任何 Elo 结论；`O4_ENDGAME=1` 的 same-bank h2h 待跑（见 §6）。
> **口径**：revision-3（`rules_id=2e36dbea44893696`）、obs v5、2 家；被审模型
> `runs/bres/cont/selected/bres221_cont10m_9M.pt`（17,132 参数、`vf_outcome=True`）；
> 对局来自 2026-10-08 真人 web 会话（`traces/web/run-20261008-{092931,094607,110157}`）。
> 审计代码 `runs/o4lite-search/endgame_probe.py`；产物
> `runs/o4lite-search/endgame_probe/recent_9M_v3/summary.json`（gitignored）。

## 0. 结论

1. **空底后没有采样噪声可降**。2 家 + 公开出牌 + 底牌堆空 = 完全信息：`sample_hidden` 在
   `draw_count=0` 时对所有决定化给出同一个世界（已验证），K=128 只是同一局重复 128 遍。
   该阶段的决策误差是 critic 的**系统偏差**（+ 截断深度 + β 权重），不是蒙特卡洛噪声。
2. **两处决定性翻转是搜索自己引入的 40 分错误**：raw argmax 在精确 minimax 下就是最优，
   t=3/β=1.5 的搜索把它改成最差一档；两局最终分差恰好与所选动作的精确值一致（巧合，非证明）。
3. **α-β 精确解在完全信息终局可行**：6 个已记录最难局面（含两个 7v7）0.09–1.56s
   （同一批局面的无剪枝 negamax 为 2.8–205.8s），值逐位一致；超预算回退搜索。
4. **已接入 run-local 搜索层**，评分身份默认不变：自由对战默认开（页面可关），定级/twin
   只在 manifest `search_config.endgame=1` 时开（ADR-0019）。**未测量叠加不进入评分链。**

## 1. 问题与机制

真人对局反馈：模型终局判断仍不稳定，疑似噪声。对 7 局、40 个空底决策做精确复盘后，
机制如下：

- **完全信息重建**：`View`（己方手牌 + `played` + `trick_cards` + 公开计数）在
  `draw_count=0` 时唯一确定 `GameState`；`exact_state_from_view` 直接复用
  `rollout_policy.rebuild`，无隐藏分配歧义。
- **K 决定化同一世界**：对 `s1816097613` step 29（空底）用 4 个不同 RNG 调
  `sample_hidden`，对手手牌集合逐位相同、牌堆为空。故 K 只影响空底**之前**。
- **误差来源**：t=3 截断处的 critic 读出（repo 自证：单叶 EV 上限 0.6409、shipped 线性头
  0.5216，残差偏差主导，见 [`stage0-critic-ceiling.md`](./stage0-critic-ceiling.md)）；
  `V + β·u` 中 β=1.5 是分解配方值 1.0 之外的过权（[`vf-outcome.md`](./vf-outcome.md) §0）；
  rollout 对手是模型自身而非最优应对。

## 2. 精确 minimax 审计

**判定器**：引擎内零和 minimax（2 家时终局分差 = 己方 − 对方，总分 100 全分布），
α-β + canonical 置换表（丢弃 `plays`/`played`/`collected`/`revealed` 历史），
`runs/o4lite-search/endgame_probe.py:ExactSolver`。值是从行动方视角的**保证**值。

### 2.1 两个决定性局面（精确值）

| 局面 | 精确值（从行动方视角） | 实际 | 说明 |
|---|---|---|---|
| `s1816097613` step 29：40-25，持 ♣4 ♥9 ♥10 ♣Q ♠5 ♣7 对 ♥4 | ♥9/♥10 → **+20**；♣Q/过 → 0；♠5/♣7 → **−20** | raw = ♥9（最优）；t3/K128/β1.5 选 **♠5（−20）**；t3 β0/β1 选 ♣Q（0）；t5/t7 β0 选 ♥9（+20）；全终局 rollout 选 ♣Q（0） | 终局 40-60；所选动作的精确值也是 −20（巧合一致） |
| `s2032056744` step 38：30-50 | ♣6 → **+40**（最优）；♠2 → 0 | raw = ♣6（最优）；t3/K128/β1.5 选 **♠2（0）**；t5 β0 与全终局 rollout 选 ♣6（+40） | 该局终局 50-50 |

β 在第一个局面上是直接肇事者：同一 view、t=3 时 β0/β1 → 0，β=1.5 → −20。

### 2.2 40 个空底决策（描述性，不是统计比较）

| 臂 | 低于精确最优 | 各决策 best−arm 之和 |
|---|---:|---:|
| raw argmax | 5/40 | 130 |
| t5/K32/β0 | 4/40 | 130 |
| 实战 t3/K128/β1.5 | 5/40 | 140 |
| 全终局 rollout（K=1 自博弈） | 6/40 | 170 |

读法限制（重要）：同一局的后续状态被重复计入（前一步错完，后面 best 也一起塌），
7 局、40 决策是非独立便利样本，没有 CI，也没有任何新 Elo 读数；上表只说明**没有任何
现有臂在完全信息终局做到零遗憾**，以及全终局自博弈不是 minimax。真正站得住的证据是
§2.1 的确定性精确值。

## 3. 集成（run-local，不动评分链）

| 文件 | 内容 |
|---|---|
| `runs/o4lite-search/endgame_solver.py` | `ExactEndgameSolver`（α-β + 有界 TT + 节点/时间预算）、`exact_state_from_view`、`EndgamePolicy`（`draw_count==0` 精确求解，否则委托原 policy） |
| `runs/o4lite-search/check_endgame_solver.py` | A 审计复跑 / B 小局面 α-β vs 无剪枝参考 / B2 新旧走法生成 realized-play 集 / C wrapper 精确·委托·回退 / D 工厂冒烟 |
| `web_search.py` / `rollout_trunc.py` | `endgame` 开关：网页自由对战默认开、逐局可关；`O4_ENDGAME=1` 给 h2h 臂叠加；默认预算 `node_budget=1e6`、`time_budget=5s`、总牌数 ≤14，超预算回退 |
| `web_plugin.py` / `web_twin.py` | placement/twin 从 manifest `search_config.endgame`（缺省 0）重建；未测量 overlay 不进入评分身份（ADR-0019） |
| `src/seven523/web/static/app.js` | 设置页「终局精确解」勾选；`searchLabel` 显示「终局精确」 |

**估读面板**：`EndgamePolicy.value_and_outcome` 在状态已有精确 TT 条目（`EXACT`）时直接
返回精确分差（`outcome=None`），否则委托原 readout；不会为了显示触发新求解。

## 4. 性能（只读基准，决策逐位不变）

- **α-β vs 无剪枝 negamax**（同一 canonical key、同一走法生成）：6 个已记录最难局面
  0.20–4.04s vs 2.80–205.8s（**14–90×**）。
- **走法生成**：只试 `top_rank` 实际持有的花色（炸弹/过牌一次 `Game.step`），
  6 局面合计 10.75s → **4.26s**（~2.5×），节点数与决策值逐位不变；叠 web 路径的
  `fast_engine` 后 worst ~1.43s。
- **搜索侧 CPU 线程**：小 batch MLP 用 torch 默认 24 线程反而慢；`torch.set_num_threads(1)`
  （`prepare_runtime`/`TruncFactory` 默认，`O4_TORCH_THREADS` 可覆盖）实测
  t5/K32 91→63ms（1.44×）、t3/K128 199→170ms（1.17×）。
- 搜索侧剩余瓶颈与 Route-A 记录见 `runs/speed/PERF_REPORT.md`（gitignored 产物）：
  `action_mask` ~50%、`encode_observation` ~20%、NN 前向 <5%。

## 5. 验证

- `check_endgame_solver.py`：A 40/40 与审计精确值一致；B/B2/C 共 **122 项 0 failures**
  （B2 在 64 个随机空底局面上验证新走法生成与旧 4 花色枚举同 realized-play 集）；
  `--checkpoint` 工厂冒烟 2 局、无非法动作且 endgame 命中。
- `uv run pytest -q -n 4`：**932 passed, 14 skipped**（含 `tests/test_web_plugin.py` 对
  `search_config` 新增可选 `endgame` 键归一化的断言）。
- 身份不变量：`traces/pool10/manifest.json` 的已发布 rung（`search_leafq` 等）
  `search_config` 缺省 `endgame=0`，plugin 重建结果与测试断言一致。

## 6. 未决 / 下一步

1. **Elo 确认（条件项）**：same-bank h2h（3×400 起步）`O4_ENDGAME=1` vs 当前最强臂，
   按仓库口径（CI 排除 0 且点 ≥+10 才行动）。`endgame_probe` 的 40 决策只证机制，
   不证强度；也需检查 minimax 是**最坏情况保证**，对弱于最优的真人期望分可能不是最大，
   必要时再用对手模型在近优动作集内挑 margin。
2. **性能**：本次只动了零强度风险的常数项。再往下是 Route-B（降 K/剪候选/t 截断，
   影响强度、需配对 h2h）或引擎分配改造（`plays` 不逐步复制，中等风险）。
3. 长局/多局复用：TT 按局实例缓存；跨局不共享（手牌不同），不做持久化。

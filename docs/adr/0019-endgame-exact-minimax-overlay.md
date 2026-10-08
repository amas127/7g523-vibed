# 空底精确 minimax overlay：自由对战默认开，未测量身份默认关

> **2026-10-08 operator 决策**（固定 `endgame` overlay 的启用边界，供后续搜索/定级引用）。
> 历史 run 不回填、不重跑；已发布的 manifest rung 身份逐位不变（`endgame` 缺省 = 0）。

## 背景

- **机制事实（精确、可复现）**：2 家局在底牌堆空（`draw_count == 0`）后是完全信息，
  `View` 唯一确定 `GameState`；此时 K 个决定化是同一个世界，搜索的残差是 critic 系统偏差
  与 `V + β·u` 的权重，而不是采样噪声。对 40 个真人对局空底决策做精确 minimax 复盘，
  发现两处 40 分翻转（raw argmax 在精确值下最优，t3/β1.5 搜索改为最差档）——
  详见 [`experiments/endgame-minimax-audit.md`](../experiments/endgame-minimax-audit.md) §2。
- **实现**：`runs/o4lite-search/endgame_solver.py` 的 α-β + canonical TT 求解器与
  `EndgamePolicy`（仅在 `draw_count == 0` 且总牌数 ≤14 时接管，节点/时间预算超限回退原搜索）。
  它是 run-local 的推理期算子，不改权重、不改训练、不改评分链。
- **身份约束（ADR-0013）**：manifest `subjects[].search_config` 是搜索 rung 的测量身份单一来源；
  给一个已测量的 rung 静默叠加未测量的 overlay 会改变被评分对手的行为。自由对战是 unrated，
  不受该约束。

## 决定

1. **自由对战默认开**：`web_search.SearchFactory(endgame=True)`；页面逐局可关
   （`search.endgame ∈ {0,1}`，默认 1）。`random` 锚不包装。
2. **h2h / arena 实验显式 opt-in**：`O4_ENDGAME=1`（或 `TruncFactory(endgame=True)`）给
   `rolloutt:` / `rollout:` 臂叠加；默认逐位不变。
3. **测量身份默认关**：placement/twin 的 `search_config` 新增可选键 `endgame`（0/1，缺省 **0**）；
   只有显式写 `endgame: 1` 时，精确求解才属于该 rung 的发布身份，且必须走一次新的
   `refit_mle --manifest-out` 发布。空 search mapping（placement 路径）永远按配置缺省处理。
4. **不变式**：overlay 不改变 `draw_count > 0` 的任何行为；不改变决策 RNG/采样语义；
   同一输入确定性；超预算一定回退原搜索而不是猜牌。
5. **预算默认**：`node_budget=1e6`、`time_budget=5s`、`max_total_cards=14`。

## 考虑过的替代

- **所有路径默认开**：会使已发布的 `search_leafq` / `head32ln5m` 搜索 rung 在 placement 中
  静默换行为 → 违反 ADR-0013 的身份契约，否决。
- **用新 spec/spec 后缀表达精确求解**（如 `endgame:<ckpt>`）：多一份身份真相，manifest
  `search_config` 仍是权威但工具/调度看不到；与 ADR-0010 §2 的 spec owner 边界冲突 → 否决
  （改用可选键，refit 逐字保留）。
- **永远只做离线审计、不接入**：放弃人机体验收益；审计已证明单次求解秒级可行 → 否决。
- **把 minimax 值也写进对手模型预估面板**：仅在已有精确 TT 条目时返回（不额外触发求解），
  不改变面板在非空底/未解状态的语义 → 采纳。

## 后果

- `search_config` schema 增加可选键；旧 manifest 无键 = 0，`validate_search_config` 归一化后
  包含 `endgame`，消费方（plugin/twin/refit）按 0 处理。
- `web_search` / `rollout_trunc` 的统计新增 `endgame_solve_rate` / `endgame_fallback_rate` /
  `endgame_nodes_mean` / `endgame_ms_mean`；机制门（fallback≤1%、terminal=1）在 opt-in 臂上
  可继续用。
- 文档面：`docs/experiments/endgame-minimax-audit.md`（结论与验证）、`experiments/README.md`
  §0 第 32 条、`docs/human-play.md` §6.1、`docs/search-config-plan.md` 顶注；测试面
  `tests/test_web_plugin.py` 的归一化断言同步。
- **未决**：Elo 确认需 same-bank h2h（`O4_ENDGAME=1`，按仓库 +10/CI 口径）；在确认前，
  本 overlay 只作自由对战/研究算子，不作为已确认的强度杠杆。

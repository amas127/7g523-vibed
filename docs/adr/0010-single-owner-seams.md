# 单一 owner 接缝：录制、spec、manifest、prior、指标、联赛与定级包各归一处

本轮架构清理（2026-09-25，phase 2–7b）之前，同一条知识散落在多个调用点：批量录制
（play → `build_trace` → `save_trace` → 命名）在 ladder/placement/measure 各自手写，
per-seat 策略 seed 一处按调度下标、一处按座位派生；`random`/`greedy`/`ckpt:` 语法在多个
工具里各有一份解析与校验；manifest 读写在 build_ladder 与 measure 工具里重写；
轨迹先验核心（S1 特征、岭回归、先验模型、`session_mean`、`extract_features`）住在
`tools/fit_trace_prior.py` / `tools/measure_trace_signal.py` 里，`placement` 反而要按
路径加载工具；`train.py` 同时装着 rollout 循环、指标 sink、对手联赛与 PFSP 状态；
分片执行归 `arena`，但 `ladder` 也要用；`placement.py` 是 1195 行的单文件。

决定：把每一类知识收成一个 owner，其余调用点只做装配。

1. **`record.play_recorded(...)` 是唯一的批量录制接缝**（`policy_seed` 是唯一的
   逐座位策略 seed 派生，`RecordedGame` 是返回值）。ladder、placement 会话与
   measure 工具都经它走 `play_game` → `build_trace` → 落盘；终端 `7g523-play` 的
   `--save-trace` 写用户命名文件，不在批量路径内。
2. **`policies` 拥有 spec 语法**：`split_entrant`、`default_id_for_spec`、
   `missing_ckpt_path`、`validate_spec` 与 `policy_from_spec` 同处一模块；工具只消费。
3. **`study` 是 manifest 唯一 writer owner**：`load/save/merge_manifest` 只此一处，
   measure 工具也经它写盘，`levels` 的冻结语义只有一份实现。
4. **`prior` 核心留在 `src`，tools 只是 CLI**：S1 特征、岭回归、`prior.json` schema、
   `TracePrior`、`session_mean`/`extract_features` 归 `seven523.prior`；
   `tools/fit_trace_prior.py`、`tools/measure_trace_signal.py` 变薄壳。
   依赖方向固定为 **tools → src，src 不反向 import tools**（全库 grep 验证为 0）。
5. **`ladder.play_games` 是分片执行的唯一接口**：`results_dir=` 写每 shard 的
   `shard_NNNNN.jsonl`；`arena.play_parallel` 退化为薄兼容包装，`workers=1`/`>1`
   逐位一致的性质仍在 `ladder` 的回归测试里。
6. **`metrics` / `league` 从 `train` 抽出**：`LOG_FIELDS`/`TB_TAGS`/`MetricsLogger`/
   `TensorboardLogger` 与 `LeagueConfig`/`League`/`build_league`/`parse_pool_member`/
   `pool_member_ids`/`PfspController` 各归新模块；phase 6 把 `train.py` 从 877 行缩到
   668 行，rollout/update 循环仍留在 `train`。
7. **`placement` 变为包**：`seven523/placement/` 拆为 `opponents.py` / `estimator.py` /
   `session.py` / `cli.py`，`__init__.py` 只做公开 API 再导出（`main` 仍是
   `7g523-elo` 入口），历史调用方无感。

## 考虑过的替代

- **继续让 `arena` 拥有分片**：`ladder`/`duel` 也要并行对局，复制分片逻辑会让
  「worker 数无关、逐位一致」的性质出现第二套实现；把接口下沉到 `play_games` 后
  `arena` 只剩联盟装配与联合拟合。
- **把 `prior` 留在 tools、由 `placement` 按文件路径导入**：会让 src 依赖工具脚本的
  私有函数，测试只能穿过 CLI；上移后再薄壳化，离线标定与在线预测共享同一实现。
- **把 spec 语法散在工具里**：四个工具各自解析 `ID=SPEC` 与 `ckpt:` 存在性，错误
  信息已经漂移过；收进 `policies` 后错误文案保持逐字节一致。
- **让 `study` 只做 schema、writer 留在各工具**：冻结/refit 语义会被重写，且
  `levels` 契约可能出现第二套真相。
- **`placement` 保持单文件**：1195 行里会话状态、对手调度、估计器与 CLI 混在一起，
  单独测试估计器要构造整个会话；拆包后 `__init__` 再导出即可保持路径兼容。
- **把指标 sink 留在 `train`**：`tools/backfill_tensorboard.py` 也要读同一列集合；
  独立模块让 CSV 列定义只有一个来源。

## 后果

- 改一条「录制 → trace → 落盘」流程只动 `record.py` 与 `trace.py`；改 spec 语法只动
  `policies.py`；改 manifest 契约只动 `study.py`；改先验数学只动 `prior.py`；
  改分片语义只动 `ladder.play_games`；改联赛/PFSP 只动 `league.py`；改 CSV/事件列只动
  `metrics.py`；改定级会话只动 `placement/`。
- `src` 不再依赖 `tools`（反向 import 为 0）；工具可以独立运行、独立删改，不影响库测试。
- 所有既有入口保持不变：`7g523` / `7g523-play` / `7g523-train` / `7g523-eval` /
  `7g523-elo`（= `seven523.placement:main`）。
- 当前全套 `uv run --group train pytest -q` 收集 464 项；本清理不改变任何行为，
  只改变知识的归属。

# 单一观测布局与单一比较语义：v5 与牌型族不再保留兼容变体

观测布局和牌型比较都长出了「为旧产物保留的兼容路径」：ADR-0008 让 v1–v4 与 v5 并存
（多层段表、`encode_observation`/`observation_dim` 的版本参数、checkpoint 按 `obs_version`
dispatch、`--obs-version`、热启动段级列重映射 `_REMAP_VERSION_PAIRS` / `_first_layer_remap`
及 54→19 的近似拷贝），ADR-0007 让旧平面 `tier` 以 `Rules.comparison` 变体仅用于回放
2026-09-25 之前落盘的 trace。两处的共同点是：**没有任何产者，只有消费者**——

- `runs/` 下现存 50 个 checkpoint 的 payload 全部是 `obs_version = 5`（逐个读取验证）；
  代码里已无产生 v1–v4 的路径，`--obs-version` 也没有任何调用方。
- `Rules.comparison` 剩下需要旧口径回放的语料只有 `traces/study`：1200 局 + `manifest.json`
  （目录共 1201 个 JSON；另有 2 个早期 smoke 会话 trace，量级可忽略）。这份语料是牌型族规则变更前按 `tier` 判定落盘的历史资产；
  `trace.rules_from_json` 仍会忽略旧的 `comparison` 键，因此**读取**不会报错，但按当前
  牌型族语义回放时，个别「当时合法、现在不合法」的应牌会校验失败——作为只读历史语料
  可以接受，继续维护一条可执行但无人验证的 tier 回放路径则不值得。
- 兼容层带走了可观测试面：`tests/data/legacy_v1_views.json`（6542 行）与
  `test_combos`/`test_env`/`test_networks`/`test_trace`/`test_rules`/`test_train`/
  `test_fit_trace_prior` 中的版本矩阵、重映射表与 tier 用例一并删除。

决定：

1. **env.py 只有一张段表、一个布局**：`_SEGMENTS` 即 v5（`119 + 21n`，2 家 161），
   `OBS_VERSION = 5`；`observation_dim(num_players)` 与 `encode_observation(view, rules)`
   不再接受版本参数，`Seven523Env` 构造函数也没有 `obs_version`。
2. **热启动只有一条路径**：`networks.warm_start_from(path, agent, *, device)` 负责
   「加载 + 校验布局 + 拷贝」；同一 v5 布局内的 `nvec`/`arch` 差异仍由 `warm_start_into`
   逐键映射（actor 行前缀拷贝、shared/towers 互转），不同 `obs_dim` 抛
   `WarmStartLayoutError`。`_first_layer_remap`、`_REMAP_VERSION_PAIRS` 与
   `--obs-version` 全部删除。
3. **checkpoint 身份仍显式**：`save_agent` 继续把 `obs_version` 写进 payload，
   `load_agent` 对缺失或非 5 的 payload 直接抛 `ValueError`——v5 之前的文件必须重训或
   显式迁移，绝不按宽度猜测布局。
4. **比较只有牌型族语义**：`Rules` 不再有 `comparison` 字段，`beats` 只实现
   「炸弹按层级、非炸弹同族比 `(size, top_key)`」；`TIER` 保留，但只服务炸弹层级与
   `Combo.strength` 的稳定排序。trace 的 `rules` 段不再写 `comparison`；读取端容忍旧键
   （见第 1 条语料说明）。
5. **ADR-0007/0008 仍是历史记录**：它们记录的决策（非炸弹同族比较、v5 = 观测 S1+B0+B1
   成为默认）继续有效；被本 ADR 取代的只是其中的兼容条款：ADR-0007 的 `"tier"` 回放
   变体、ADR-0008 的 v1–v4 保留与跨版本热启动重映射。

## 考虑过的替代

- **保留兼容层，等旧产物自然消失**：旧 checkpoint 已全部是 v5、已无 v1–v4 产者，
  继续维护多段表与 54→19 近似特判只会让每次观测改动都要同步三处（段表、版本选择、
  热启动重映射），成本高于收益。
- **把旧 checkpoint 迁到 v5 或直接删除**：无迁移必要；v5 之前训练的文件早已不在
  `runs/` 的生产路径上。保留 `load_agent` 的显式拒绝即可，silent 猜测比报错危险。
- **保留 `tier` 变体只用于回放 `traces/study`**：这条路径没有测试与验证，却把「回放
  旧语料」永久钉在规则核心上；语料是只读历史资产，按当前语义读取/回放并在失败处
  如实报错更干净。
- **保留 v4 作为消融/热启动中间版本**：ADR-0008 的 B0/B1 500k 均未确认（B0 null、
  B1 主端点 CI 跨 0），没有继续产 v4 的实验计划；v4 段只是 v5 前缀，需要时可由 v5
  直接截断，不必留版本。
- **按 `obs_dim` 宽度推断旧布局**：v1 3 家与 v2 2 家同为 194 维，宽度不唯一；ADR-0008
  已拒绝，本 ADR 维持显式版本戳 + 硬拒绝。

## 后果

- 新 checkpoint 一律是 v5；加载 v1–v4 文件在启动时报明确错误（需重训，不做列映射）。
  跨架构/跨动作头（单头 134 → `(134, 4)`）热启动仍可用，因为它们共享同一 v5 布局。
- `traces/study`（1200 局 + manifest）作为历史语料保留，不作 live 输入；由它标定的
  `T2_LABELS` 与 `artifacts/human-elo/prior.json` 仍是旧 `tier` 口径，重标定路线见
  `docs/plans.md` T15。
- 测试面删除了 v1–v4 布局矩阵、热启动重映射表与 tier 比较用例；当前全套
  `uv run --group train pytest -q` 收集 464 项。
- 文档同步：[DESIGN.md](../../DESIGN.md) §4 只描述 v5 段表并指向本 ADR；
  `RULES.md` §7 R-Q12 改为「tier 变体已随旧语料退役（见 ADR-0009）」；
  `docs/experiments/README.md` 的口径警告保持「旧数字不可混比」并补充语料只读说明。

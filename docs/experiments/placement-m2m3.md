# T13 / M2–M3：10 局定级会话 + `7g523-elo` CLI（7鬼523）

> 资产状态（2026-09-25）：本报告引用的模型 checkpoint 已删除，旧路径不再可用；数值为牌型族规则变更前口径。

> **状态：已交付（2026-09-25；离线链路就绪，M4 真人试点等 D-3）。一句话结论：
> `src/seven523/placement.py`（会话编排 + `main`）、`tests/test_placement.py`（24 项）、
> `7g523-elo` CLI、`play.py` 的 rung 标签（`anchor:greedy@seatN` /
> `opponent:lvlN@seatN`）与公开 `interactive_chooser` 全部落地；**`elo.py` 零改动**
> （sha256 与改动前快照逐位相同 `80641f13…`）。模拟冒烟 `--simulate greedy --games 2`
> 输出 `定级：1287 ± 182（95% CI），最近档 greedy（1315），临时 provisional`，会话产物在
> `traces/sessions/smoke_m23/`；全套 **418 passed**（`tests/test_play.py` +3）。**
>
> 设计来源：[`../human-elo-plan.md`](../human-elo-plan.md) 修订节 / 里程碑 M2–M3；
> 估计器规格：[`human-elo-10-games-research.md`](./human-elo-10-games-research.md) §5（两通道
> BT-MAP）、§8（实现映射）；先验产物：[`trace-prior-m1.md`](./trace-prior-m1.md)（M1）；
> 座位裁定：plans §4 D-6=(b)。原始产物在 `traces/sessions/`（会话）与
> `artifacts/human-elo/prior.json`（M1）。

## 1. 交付内容

| 文件 | 内容 |
|---|---|
| `src/seven523/placement.py` | M2 会话编排 + M3 CLI：`PlacementSession`、`select_opponent`、`plan_deals`/`plan_seats`、`fit_session`、`stop_reason`、`report()`、`main`；只编排 `play_game` 与 `elo.fit_ratings`，不碰评分数学 |
| `tests/test_placement.py` | 24 项：5/5 座位与不同 deal seed、Fisher 单调性/Thompson auto、停止规则、σ×2 margin、rung 先验进联合 Hessian、10 局调度/产物、空会话、shipped prior 加载与预测、CLI `--help`/`--simulate` 冒烟、pyproject 入口 |
| `tests/test_play.py` | +3：`test_opponent_identity_labels_rungs_and_anchors`、`test_play_main_writes_rung_labelled_trace`、`test_interactive_chooser_is_exported_for_placement` |
| `src/seven523/play.py` | 新增 `opponent_identity`（ckpt 目录名 → `opponent:lvlN`；脚本 bot → `anchor:greedy`/`anchor:random`）、公开 `interactive_chooser`、`main(argv)`；trace 的 `players` 写 `role:id@seatN` |
| `pyproject.toml` | `[project.scripts]` 增 `7g523-elo = "seven523.placement:main"` |
| `src/seven523/elo.py` | **零改动**：sha256 `80641f135add75301d6f6632cc578f15de772c8eb92d7adad37986959b368f9d`，与改动前快照逐位相同 |

## 2. 会话设计

### 2.1 调度（D-6=(b)）

- **10 局 10 个不同 deal seed**（`plan_deals` 去重）；**座位交替 5/5**
  （`plan_seats` 从 `--seat-start` 起轮流，偶数局数保证精确 5/5）。
- 逐局 `GameRecord` 记录 `index/seed/seat/opponent_id/opponent_elo/scores/result/trace`
  以及逐局的 `mu_traj`/`sigma_traj`/`posterior_elo`/`posterior_se`。

### 2.2 选对手（`select_opponent`）

- **info**：对后验均值最大化 Bernoulli Fisher 信息 `p(1−p)`（argmax）；后验越不确定时也会
  自然偏向信息大的近档。
- **Thompson**：从 `posterior` 采真人评分、从各候选 `N(elo, se)` 采样后取同一 argmax；
  `mode="auto"` 在前 `--explore-games`（默认 **2**）局用 Thompson 防先验偏，其后切 info。
- 池来自 manifest（默认 `traces/study/manifest.json`）的 `levels`：`random`/`greedy` 为钉死
  锚点（1000/1315），`lvl1–lvl4` 为自由 rung；选择结果确定性排序（elo,id）打平。

### 2.3 产物与持久化（`traces/sessions/<id>/`）

- 开局写 `rungs.json`（对手池 + `prior_labels` sidecar）；
- **每局**：`gNNNN__s<seed>__seat<k>__vs<opp>.json`（本局完整 trace）+ 更新 `session.json`
  （含逐局记录与 `stopped_reason`）；
- 收尾 `report.json`：点估计、95% CI、band（`placed`/`provisional`/`coarse`）、
  最近档、`provisional` 标记、两通道参数。

### 2.4 估计器（两通道 BT-MAP）

- **轨迹通道**：逐局 trace 经 M1 产物 `artifacts/human-elo/prior.json`——M1 的
  `extract_features` → `predict_elo` 得逐局轨迹分，运行均值 = `μ_traj`；
  先验 `Prior(μ_traj, σ_traj(n))`（σ 随局数查表）。
- **结果通道**：胜负 logistic + 分差高斯 `FitConfig(margin=(0.143, 2σ₀))`——轨迹先验在场时
  σ 翻倍（`MARGIN_SIGMA×MARGIN_TRACE_FACTOR = 44.5×2 = 89`）以免两通道重复计权
  （HR §5.3/§7.4）。
- **联合拟合**：`fit_ratings(window=None)`（10 局就是全部）；自由 rung 以
  `Prior(标签, 30)` 进同一联合 Hessian，锚点钉死；SE 用联合 Hessian，不夸大。
- **停止**：`CI 半宽 ≤ --stop-ci`（默认 50）或打满 `--games`（默认 10）；报告同时给
  两通道权重（`1/σ_traj²` 与 `1/se_result²` 归一）。
- `--no-trace-prior`：只跑结果似然冷启动（不加载 M1 产物/numpy）。

## 3. 冒烟与测试

2026-09-25 代理落地后 + owner 复跑：

```bash
uv run 7g523-elo --help            # 用法/开关正常
uv run 7g523-elo --simulate greedy --games 2 --session-id smoke_m23 --quiet
# → 定级：1287 ± 182（95% CI），最近档 greedy（1315），临时 provisional
# → 报告：traces/sessions/smoke_m23/report.json
```

`traces/sessions/smoke_m23/` 内容齐全：2 个 `g*.json` trace、`session.json`（2 局记录、
`stop.reason=max_games`）、`report.json`（human Elo 1287.23、95% CI [1104.81,1469.65]、
半宽 182.42、`provisional=true`、`band=coarse`、最近档 greedy 1315、两通道权重
0.677/0.323、margin [0.143,89.0]）、`rungs.json`。冒烟只用 `--simulate` 的 greedy 当
「真人」，**不是**真人标定数字（真人 OOD 未验证，见 §4）。

全套测试：`.venv/bin/python -m pytest -q -p no:cacheprovider` → **418 passed**
（`test_placement.py` 24 + `test_play.py` 新增 3）。

## 4. 限制与风险

- **真人 OOD 未标定**：M1 先验在 bot 轨迹上做 LOLO；真人分布偏移未知，必须靠 D-3 真人试点
  （≥8 人）验证（HR §9.2 Q-7）。
- **>1600 信息塌缩**：档位映射的 `p(1−p)` 在两端塌缩（HR §6.1/§9.2 Q-9），强真人可能落到
  信息量最低的区间；需更强 bot 或 handicap 匹配，属 M4 范围。
- **交互 UX 未实机演练**：M2/M3 只验证了 `--simulate` 非交互路径与 `interactive_chooser`
  的导出；真人在终端里的会话流程（中途退出、误输入恢复）未做过人机演练。
- **manifest 是硬输入**：缺 `traces/study/manifest.json`（或显式 `--manifest`）时 CLI 报错；
  `--no-trace-prior` 只解决「缺 M1 产物/numpy」的回退，不解决 manifest。
- **座位相位残差**：D-6=(b) 的 5/5 轮换保留跨序残差（HR §6.3；研究口径曾倾向冻结 twin），
  报告已按 owner 裁定标 `provisional`；如需干净口径需另跑研究模式仿真。
- **未做**：`elo.py` 数学改动、轨迹 S2 逐决策 regret、轨迹 S3 评分（`../human-elo-plan.md`
  改动计划第 7 条）。

## 5. 复现

```bash
# CLI 冒烟（非交互）
uv run 7g523-elo --simulate greedy --games 2 --session-id smoke_m23 --quiet
uv run 7g523-elo --help

# 真人会话（M4 前置：需要 TTY 与 manifest）
uv run 7g523-elo --games 10

# 测试
.venv/bin/python -m pytest -q -p no:cacheprovider   # 418 passed
```

产物：`traces/sessions/smoke_m23/{report.json,session.json,rungs.json,g0000__*.json,
g0001__*.json}`；先验 `artifacts/human-elo/prior.json`（M1）；实现见 `src/seven523/placement.py`
与 `src/seven523/play.py`（差异），评分数学保持 `src/seven523/elo.py` 原样。

# 推理期搜索 + value 截断：部署档位、联合训练负结果与解耦边界（wave1）

> **状态**：完成（2026-09-28）。本波由 owner 驱动，包含**已预注册**（`trunc_preregistration.md`、
> `scale_preregistration.md`、`runs/ei/preregistration.md`、`runs/ei2_value_t5/preregistration.md`）
> 与**探索性**（t×K 网格、t 前沿、K 轴补种）两类实验；探索性部分只作方向，不作行动依据。
> **口径**：revision-3（出空即撬底，`rules_id=2e36dbea44893696`）、obs v5（2 家 161 维）、
> 纯 MLP（`arch=shared`、`hidden=128`，≈55,179 参数）、PPO。比较均为 deal-twin 换座 h2h
> （400 副/deal seed，按副聚簇 bootstrap），行动门槛 = z 与 t 95% CI 同时排除 0 且点估计 ≥ +10
> （README §3 / EPV §9）。**跨 fit/跨 bank 绝对值不可比**（ADR-0013）。

## 0. 结论（TL;DR）

1. **当前最强部署 = 推理期搜索 + value 截断（t=5）**，训练侧不动：
   - `t=5 × K=16`：vs raw **+111.2** [105.1, 117.2]（20–23 seeds）；
   - `t=5 × K=32`：vs raw **+124.7** [119.4, 130.0]（20 seeds）；
   - 对照：全量搜索 K=8 vs raw 仅 +42.6（3 seeds）/ +38.0（bank 20-22）。
   - 相对最初的全量 K=8，t=5×K16 更强约 +70 Elo 且每步更快（~40–58 ms vs ~90–100 ms）。
2. **"训练+搜索"联合方案：两条权重侧路径均为负结果**：
   - 策略蒸馏（EI-1，硬 CE/soft-Q/margin 过滤）：主端点 **−52.8** [−66.0, −39.6]，
     剂量-反应「越拟合越差、不拟合=raw」；同 View 两次搜索的标签翻转率 51.7% →
     搜索动作是**采样隐藏世界**的函数，不是 obs 的函数。
   - value 重训 bootstrap（EI-2）：P1b = **+0.08** [−11.4, +11.6]（≈0）；根分布 EV 未过门槛
     （0.844→0.829），叶分布 EV 反而更好（0.425→0.467）——**value 精度不是截断搜索的瓶颈**。
   - 深度 2（对手 min 节点）：配对 **−11.5** [−29.3, +5.3]，3× 成本，无增益。
   - 推论：本游戏的强度来自**测试期对隐藏世界的成组采样（K）与低方差读出（t）**，不在权重里。
3. **解耦边界**：搜索算子（K/C/t/对手模型）与策略**基本解耦**——K16−K8 在固定 critic/策略下
   仍 +25.0；搜索在损坏 53 Elo 的蒸馏学生上仍 +40.4。**value 与评分身份不解耦**：critic 只在
   P0-vs-P0 分布上校准；搜索/截断包装不进 `rules_id`，跨 fit 不可比。
   *（2026-09-29 P4a 更新）*：`search_leafq`（t5/K32/C6）已作为第 11 个 measured subject 进
   `traces/study`(=pool10) manifest 的同一 joint MLE（[`../search-config-plan.md`](../search-config-plan.md) §4.3）；这是**对手身份**
   （subject id + `rolloutt:` spec）而非规则身份，跨 fit 仍不可加/混比。
4. **速度**：Route-A 加速层逐位精确（同 seed `games.jsonl` 字节一致），189→100 ms/决策（~1.9×）；
   叠加 t=5 截断后 ~19–29 ms/决策（相对最初共 ~9×）。人机入口
   `runs/o4lite-search/play_search.py [--trunc 5] [--rollout-k 16|32] [--no-fast]`。

## 1. 截断前沿（K=8，同 bank 30-32，3 seeds/臂）

| t | vs raw ΔElo | z-CI | 判定 |
|---|---|---|---|
| 0（纯 value） | −183.6 | [−198.1, −169.0] | 价值单独不可用 |
| 1 | −40.8 | [−57.3, −24.2] | fail |
| 2 | +7.0 | [−5.8, +19.7] | 不定论 |
| 3 | +65.4 | [+52.2, +78.6] | hit |
| 5 | +77.5 | [+64.3, +90.6] | hit |
| 7 | +81.7 | [+68.4, +95.0] | hit |
| 10 | +56.8 | [+44.4, +69.3] | hit |
| ∞（全量） | +42.6 | [+30.1, +55.2] | hit |

最优点在 **t≈5–7**（t5 对 t3 配对 +12 [−28.7, +3.9]；t7 与 t5 不可分辨）。
证据：`runs/tfront/{status.txt,gate_tfront_t*.log}`、`runs/o4lite-search/tfront_t*/combined.json`、
`runs/o4lite-search/abs_trunc_t5/combined.json`、预注册 `runs/o4lite-search/trunc_preregistration.md`。

## 2. K 轴（t=5 截断引擎，20–23 seeds/臂）

| 配置 | vs raw ΔElo | z-CI | k |
|---|---|---|---|
| K=8 | +86.1 | [81.0, 91.2] | 20–23 |
| K=16 | +111.2 | [105.1, 117.2] | 20–23 |
| K=32 | **+124.7** | [119.4, 130.0] | 20 |

同 bank 配对（8000 deals）：

| 对比 | ΔElo | z-CI | sign p | 判定 |
|---|---|---|---|---|
| K16 − K8 | +25.01 | [+18.40, +31.42] | 3.6e-13 | HIT（`deploy_k16`） |
| K32 − K16 | +13.59 | [+7.22, +20.20] | 3.8e-08 | 稳健但边际递减 |

- 截断确实压缩了 full-search 的 K 增益（gain gap：同 bank 20-22，trunc−full 的 K 增益差
  **−4.57 margin**，CI 全负），但没有消除——**K 与 t 不是同一笔收益的完全替代**。
- 证据：`runs/trunc_k_axis/{status.txt,decision.json,paired_k16_vs_k8.json,mechanism_gate.json,combine_tcorrect.json}`、
  `runs/tk_grid/final_k32_vs_k16_20seed.json`、`runs/parallel_fill/status.txt`。

## 3. t×K 网格（K=16/32，3 seeds/臂，探索性）

| arm | vs raw ΔElo | z-CI | 配对（同 bank） |
|---|---|---|---|
| t3×K16 | +87.4 | [74.3, 100.5] | −15.6 vs t5×K16 [−31.5, 0.0] |
| t7×K16 | +98.7 | [85.4, 112.1] | −4.3 vs t5×K16 [−20.0, +10.9] |
| t7×K32 | +116.4 | [102.8, 130.0] | −3.9 vs t5×K32 [−20.3, +12.2] |
| K 效应（t3） | — | — | t3k16 − t3k8 = +22.0 [+5.2, +38.3] |
| K 效应（t7） | — | — | t7k16 − t7k8 = +17.0 [+0.3, +34.0] |

结论：联合最优在 **t=5–7 × K=16/32**；t 在 K=16/32 上进入平台，K 的增益在每个 t 上都存在。
证据：`runs/tk_grid/{status.txt,paired_*.json}`。

## 4. EI-1：策略动作蒸馏（负，已关闭）

预注册：`runs/ei/preregistration.md`；报告：`runs/ei/report.md`。

| 模型 | 配方 | ΔElo（裸网 vs raw P0） | z-CI |
|---|---|---|---|
| r1（主臂） | 硬 CE + suit + value，3 epoch | **−52.8** | [−66.0, −39.6] |
| r1b | margin≥20 分歧点、纯策略 | 0.0 | [−8.6, +8.6] |
| r1c | 全量、vf=0、lr 2e-5 | −1.2 | [−6.8, +4.5] |
| r1d | soft-Q 排序（ListNet） | −45.6 | [−63.5, −27.8] |

- 剂量-反应：拟合越深越差（CE 1.29 → −52.8），不拟合就停在 raw。
- 保留臂：S(r1)（搜索包着损坏的 r1）vs raw = **+40.4** [+27.1, +53.8] → 强度在搜索算子。
- 机制：同 View 两次独立搜索的模板翻转率（仅搜索决策）全量 **51.7%**；截断 t=5 降至
  **37.8%**（t=10 仍 51.7%）。证据：`runs/ei/label_stability.py`（输出见报告 §2）。

## 5. EI-2：value head 重训 bootstrap（负，已按预注册关闭）

预注册：`runs/ei2_value_t5/preregistration.md`；证据：`runs/ei2_value_t5/{status.txt,calib/report.json,retr_vs_old/,retr_vs_raw/,old_vs_raw/,policy_control/}`。

| 端点 | 结果 | 判定 |
|---|---|---|
| E4 裸 policy 对照 | +0.00（2400 局全镜像） | 管线干净 |
| P1b（重训 critic vs 旧 critic，t=5 搜索） | **+0.08** [−11.4, +11.6] | null |
| P1a（重训 vs raw） | +84.1 [72.7, 95.5] | hit（同 bank 旧 critic +86.7，无提升） |
| P2b 根分布 EV | 0.844 → 0.829（ΔMSE +0.00282 [0.00178, 0.00386]） | 未过门槛 |
| P2c 叶分布 EV | 0.425 → 0.467（searchq_all 0.479） | 过 |

- 结论：value 精度不是截断搜索的瓶颈；按预注册 §6 规则 2 **关闭 bootstrap 循环**。
- 诊断臂 `t_truncq` 数值失控（EV −435.8），已隔离、不用于判定。

## 6. 深度（负，已关闭）

depth-2（对手 top-3 应手 min 节点 + 终局 rollout，`runs/o4lite-search/rollout_depth2.py`）：
vs depth-1 配对 **−11.5 Elo** [−29.3, +5.3]（3 seeds）；vs raw +26.4 [z 9.3, 43.5；t 跨 0]；
成本 ~3×（293 vs 118 ms/决策）。证据：`runs/o4lite-search/depth2_k8/combined.json`、
`runs/o4lite-search/scale/paired_depth2_vs_k8.json`、预注册 `depth2_preregistration.md`。

## 7. 搜索算子能力（O4 线，正结果，摘要）

| 对比 | ΔElo | 备注 |
|---|---|---|
| 全量 K=8 vs 同权重 raw（confirm，5 seeds） | +28.5 | z [21.6, 35.4]、t [18.7, 38.3] |
| 全量 K=8 vs raw ws_s2（3 seeds） | +41.9 | 跨对手迁移 |
| 搜索版 ws_s2 vs raw ws_s2（5 seeds） | +28.5 | 跨策略复现 |
| rollout 对手模型换成 t18poolself（5 seeds） | +24.5 | 模型不匹配仍正 |
| 截断 t=5 vs 全量（3 seeds，直接） | +67.9 | vs raw 相减口径只有 +34.8（非传递性，见 §9） |

细节与原始产物：`runs/o4lite-search/report.md`。

## 8. 推理速度（Route-A 加速层，逐位精确）

- `runs/speed/fast_engine.py`（mask 结构预过滤 + encoder 预计算表）只 monkeypatch 推理路径绑定，
  不改 `src/`；`runs/speed/verify_fast.py` 等值 11,724 项检查 0 mismatch；同 seed 对决
  `games.jsonl` 与旧引擎**逐字节一致**（sha256 `f1960f7d…`）。
- 实测：188.6 → 100.1 ms/决策（~1.9×）；叠加截断 t=5 后 19.1 ms/决策（plies 170 vs 693），
  相对最初共 ~9×。报告：`runs/speed/PERF_REPORT.md`。

## 9. 解耦与评分边界（回答 Q3）

| 配置项 | 与策略/训练解耦？ | 证据 |
|---|---|---|
| K、C、t、rollout 对手模型 | **基本解耦**（换策略可不动） | K16−K8=+25 在固定 critic 下；S(损坏 r1)=+40.4 |
| value 权重（critic） | **不解耦**（只在给定策略/对手分布上校准；但也不是强度瓶颈） | EI-2 P1b≈0；P2b 未过、P2c 过 |
| 评分身份（manifest/ladder） | 包装不进身份体系，跨 fit 不可比 | ADR-0013；本线所有比较只用同 bank h2h |

## 10. 部署建议

- **最强档**：`play_search.py --trunc 5 --rollout-k 32`（+124.7 vs raw；~80–110 ms/决策）。
- **平衡档**：`play_search.py --trunc 5 --rollout-k 16`（+111.2 vs raw；~40–58 ms/决策）。
- 默认（K=8）：`play_search.py --trunc 5`（+86.1 vs raw；~19–29 ms/决策）。
- value 权重：wave1 测量用 `runs/ei/value/critic_B.pt`（policy 与 P0 逐位相同）；2026-09-29 P4a 起部署/发布默认已切到 `runs/ei2_value_t5/t_leafq/critic.pt`（[`../search-config-plan.md`](../search-config-plan.md) §4.3）。`--no-fast` 可回退旧引擎。
- 单 seed 评估成本（400 deal，单进程）：K8 371s / K16 697s / K32 1726s（中位数）。

## 11. 负结果（建议写入 plans.md §6）

| # | 条目 | 原因 | 依据 |
|---|---|---|---|
| N-21 | 离线策略动作蒸馏（硬 CE / soft-Q / margin 过滤）把搜索强度写进权重 | 主端点 −52.8；同 View 标签翻转 51.7%（搜索动作是采样隐藏世界的函数、obs 不可决定）；剂量-反应；S(损坏学生) 仍 +40 → 强度在推理算子 | [`runs/ei/report.md`](../../runs/ei/report.md)；本报告 §4 |
| N-22 | 重训 value head 到搜索 E_w[Q] 以 bootstrap 截断搜索 | P1b=+0.08 [−11.4,+11.6]；根分布 EV 未过门槛；value 精度不是 t=5 瓶颈 | [`runs/ei2_value_t5/`](../../runs/ei2_value_t5/)；本报告 §5 |
| N-23 | 深度 2（对手 min 节点）搜索 | 配对 −11.5 [−29.3,+5.3]、成本 3×，无增益 | 本报告 §6 |

## 12. 开放问题

- **非传递性未解**：截断 t=5 −全量 = +67.9（直接）vs +34.8（经 raw 相减），差 ~2.75σ；
  部署选型不受影响（截断在两条口径下都 ≥ 全量且更快），但"比全量强多少"未定论。
- `trunc_vs_full_k16`（K=16 下的截断/全量直接对决）由 owner 决定中止，**没有结果**。
- K=32 的 t 轴只有 3 seeds（t7k32≈t5k32）；若要把 t 在 K=32 上定档需补种（~2h / 32 seeds）。
- `truncq` 诊断目标实现 bug（EV −435.8）未修复；不用于任何判定。

## 13. 复现命令（摘）

```bash
# 截断搜索（t=5）评测：K8/K16/K32 vs raw
O4_TRUNC_PLY=5 .venv/bin/python runs/o4lite-search/run_h2h_trunc.py \
  --arm trunc_k16 --seed 30 --pairs 400 --left-id trunc --right-id raw \
  --left-spec rolloutt:runs/ei/value/critic_B.pt \
  --right-spec ckpt:runs/w2m_ctl__11__1790516900/agent.pt \
  --rollout-k 16 --rollout-max-candidates 6 \
  --rollout-opponent ckpt:runs/w2m_ctl__11__1790516900/agent.pt \
  --device cpu --workers 1 --out runs/o4lite-search/trunc_k16 --tb runs/o4lite-search/tb/trunc_k16
.venv/bin/python runs/o4lite-search/combine.py --arm trunc_k16 \
  --in runs/o4lite-search/trunc_k16 --seeds 30,31,32

# 人机对战（fast 引擎 + 截断 + K 可选）
.venv/bin/python runs/o4lite-search/play_search.py --trunc 5 --rollout-k 16

# 浏览器牌桌（逐局可选 t/K；默认 t=5, K=16，见 docs/human-play.md §6.1）
uv run --group train python runs/o4lite-search/web_search.py --trunc 5 --rollout-k 16
```

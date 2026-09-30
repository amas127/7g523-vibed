# EI-3：value 迭代循环 round 1（重新采叶子 + 重训 critic，t5/t10）— 负结果

> **状态**：完成（2026-09-29）。**H1（t=5）与 H2（t=10）均为 negative**；按预注册不触发
> fresh-bank confirm。附带完成的 critic weight-decay 单因子对照为**精确零效应**。
> **口径**：revision-3（`rules_id=2e36dbea44893696`）、obs v5（2 家 161 维）、2 家、
> 修复后引擎（`runs/o4lite-search/rollout_trunc.py` 单位修复）、纯 MLP champion `w2m_ctl`。
> 所有 h2h 为 deal-twin 换座配对（400 副/seed，按副聚簇 bootstrap 4000）；行动门槛 =
> 两 CI 排除 0 且点估计 ≥ +10（README §3 / EPV §9）。跨 fit/跨 bank 绝对值不可比（ADR-0013）。
> **预注册**：`runs/o4lite-search/ei3_value_loop/preregistration.md`（读数前冻结；
> t=10 修正 `t10_amendment.md`）。原始产物在 `runs/o4lite-search/ei3_value_loop/`（gitignore）。

## 0. 结论（TL;DR）

1. **重采叶子没有继续转化**：在 `t_leafq` 自己的搜索分布上、以 K=32 重采真实终局 margin
   叶子并重训 critic，t=5 与 t=10 两个深度在 same-bank h2h 都**没有**超过 `t_leafq`：
   - t5：`v2_t5 − base_t5` = **−4.44** Elo [−15.37, +6.66]（n=1200）；
   - t10：`v2_t10 − base_t10` = **−0.68** Elo [−9.18, +7.99]（n=1200）。
2. **leaf EV 的"提升"是选择噪声**：v2_t5/v2_t10 相对 t_leafq 的 val EV 差只有 +0.00021/+0.00262，
   且都出现在按 val 选出的 best epoch；**final epoch 全部翻负**（−0.00015/−0.00139）。
3. **真人局面 479 决策**同样没有正信号：t5 DQ +0.020（CI 跨 0）、Δbias −1.17（未过 −2.0）；
   t10 DQ −0.050（CI 跨 0）、Δbias −0.84（未过 −2.0）。
4. **critic weight decay 是精确零效应**（单因子 A/B）：`AdamW(wd=0.01) − Adam(wd=0)` 配对
   **0.00** Elo [−1.02, +1.02]、leaf EV 差 +8.5e-07、screen 数字同量级。
5. 与 N-22 的关系：N-22 关闭的是「把 critic 重训到搜索自身 E_w[Q]」的 bootstrap 循环；本波是
   「真实终局 margin 叶子目标」的迭代版。t_leafq 的一次性 +18.24 是**监督目标修正**的收益；
   在它自己的分布上继续重采会到不动点（叶子 EV 0.5592→0.5594，搜索强度不动）。
   **关闭"同 recipe 重采叶子重训 critic"方向**（N-24）。
6. **补记（2026-09-29 晚，本报告发布后发现）——采集器 K 世界同种子 bug**：
   `runs/ei2_value_t5/trunc_leaves.py:382` 为 K 个 determinization 各建一个**同种子**的 RNG，
   导致 K=32 个世界逐位相同（抽查 d0/c5 各 2000 组：100% 组内叶子 obs 逐位重复；`q_std` 72% 为 0）。
   本报告（与 EI-2 的 d0/d0leaves）的训练数据因此是**单世界 ×8 复制**，采集时的搜索实际是 K=1，
   不是部署的 K=32。**部署/h2h/fresh-bank confirm 不受影响**（生产 `rollout_policy.RolloutFactory`
   用同一 RNG 连续采样，K 个世界不同；K 轴增益也印证这点）。该 bug 已修复（采集器与
   `contract_check.py` 改为每决策共享一个生成器，contract 30/30 通过；冒烟：组内平均 7.24/8
   个不同世界、q_std 62% 非零，`runs/o4lite-search/ei3_value_loop/bugfix_smoke/`）。
   因此：本报告 §2 的 E1/E2/E3 仍说明「在单世界近似分布上重采不转化」，但它
   **没有验证真正的 K=32 采集**；N-24 的结论需要 clean re-run（t=5 一臂）后才能最终裁定；
   同期 cap0 容量探针也被 8× 复制数据污染（见 cap0 报告）。

## 1. 设计（冻结口径）

| 臂 | 搜索 critic | K/C | t | 数据 | seed |
|---|---|---|---|---|---|
| c5 → v2_t5 | `t_leafq`（当前部署） | 32/6 | 5 | 6 shard × 300 deal，叶/决策=8 | 9200 |
| c10 → v2_t10 | 同上 | 32/6 | 10 | 6 shard × 300 deal，叶/决策=8 | 9300 |

- roster 与 d0 相同（`w2m=0.6, ws_s2=0.15, pool=0.15, random=0.10`）；只换全新 bank + K=32 + 换 t。
- 训练 recipe 与 t_leafq 逐位相同（`leaf_mc / all / leaf-subsample 8 / epochs 20 / lr 3e-4 /
  batch 1024 / seed 0 / holdout-every 7`），只换数据；policy/trunk 冻结（E4 证明位级不动）。
- t=10 人类局面 screen 为**反事实回放**（真人 bank 录于 t=5，无 world-cache 锚），按修正案
  仅作诊断；E3 h2h 是 H2 的决定性端点。

## 2. 结果

### E0 采集（干净）
c5 = 53,059 决策 / 363,984 叶子；c10 = 51,038 / 292,440。12 shard rc=0，fallback 0、
`rollout_terminal_rate` 1.0、aborts 0。bank 9200/9300 与已记录 bank（410–414、30–32、
501–505、520–521 等）及彼此均无重叠。

### E1 训练 / leaf EV（诊断）

| 臂 | rows | val rows | P0 基线 EV | t_leafq EV | v2 best EV | v2−t_leafq（best / final epoch） |
|---|---|---|---|---|---|---|
| v2_t5 | 363,984 | 52,224 | 0.554058 | 0.559204 | 0.559412 | +0.00021 / **−0.00015** |
| v2_t10 | 292,440 | 42,216 | 0.604357 | 0.607321 | 0.609938 | +0.00262 / **−0.00139** |

### E2 人类局面 479 决策（464 bot 决策中 370 searched）

| 臂 | DQ 点估计 | DQ 95% CI | Δbias vs t_leafq | Δbias 95% CI |
|---|---|---|---|---|
| v2_t5 @t5 | +0.0202 | [−0.1725, +0.2526] | −1.1657 | [−1.5853, −0.8363] |
| v2_t10 @t10 | −0.0496 | [−0.2255, +0.1157] | −0.8442 | [−1.2703, −0.4178] |

t5 world-cache 位级 sanity 全 0；t10 修正版以 `O4_TRUNC_PLY=10` 运行（config 已核）。

### E3 same-bank h2h（决定性，seeds 30/31/32 × 400 副，K=32/C=6）

| 对比 | 臂 vs raw（sanity） | 主配对 ΔElo | 95% CI | sign p | 判定 |
|---|---|---|---|---|---|
| v2_t5 − base_t5 @t5 | 140.94 / 145.43 | **−4.44** | [−15.37, +6.66] | 0.314 | negative |
| v2_t10 − base_t10 @t10 | 140.91 / 141.63 | **−0.68** | [−9.18, +7.99] | 0.869 | negative |

机制门全清（nested `mechanism.derived/raw`）；E4 裸 policy 对照 0.00±0；self-check 独立重算与
`paired_compare` 逐位一致（max abs diff 0）。

### E4 附：critic weight decay 单因子（runs/o4lite-search/ei3_value_loop/wd/）

| 对比 | 结果 |
|---|---|
| leaf EV（wd01 − wd0） | +8.5e-07 |
| 真人 screen | 两者几乎相同（DQ +0.020 vs +0.034，Δbias −1.17 vs −1.18） |
| same-bank h2h（1200 副） | **0.00 Elo [−1.02, +1.02]**，d_margin +0.0875，sign p=0.366 |

交叉验证：wd0 训练的 checkpoint 与工作流 v2_t5 **逐位一致**（两条训练路径同源）。

## 3. 对抗审查与实现事故（如实记录）

1. **t=10 E2 协议错误（已作废重跑）**：`screen_v2.py` 继承 offline_human 的硬编码
   `TRUNC_PLY = 5`，`O4_TRUNC_PLY` 环境变量只有 `setdefault`、从不读入常数，且对 t=5 录制的
   真人 bank 有轨迹 assert；首版 `screen_t10.json` 实际是 t=5 回放（`old`/`base` 条目与
   `screen_t5.json` 逐位相同）。按预注册 §6.6 作废，`screen_t10fix.py`（仅 3 处改动）重跑为
   `screen_t10fix.json`（config 确认 t=10），并由 `t10_amendment.md` 冻结"E2-t10 仅诊断、
   E3-t10 决定性"的口径后再读数。
2. **leaf EV 的选择偏差**：best-over-epochs 在同一 val 上选出，v2−t_leafq 的微小差值不稳健
   （final epoch 翻负）→ E1 只作诊断，不构成"转化"证据（预注册本来也如此规定）。
3. **`train.status` / `v2_*.log` 缺失**：工作流直接调用训练脚本，未走 `train.sh`；产物
   metrics/history/leaf_ev 自洽、且 wd 实验用同一 recipe 复现出逐位一致的 checkpoint，
   属过程记录缺口，不影响结论。
4. **`screen_h2h.sh` 的 `mech=1`**：内联 gate reader 读错字段层级（gate 字段在
   `mechanism.derived/raw`）并对头部带日志行的 JSON 直接 `json.loads`；对抗审查用嵌套字段
   重扫六 seed，全清。E3-t10 的 runner 首启因 chmod/exec 竞态失败，经 `bash` 重跑；seed rc
   由代理手工补写，父进程已独立核对 logs/JSON/机制门/配对数字，无结果影响。
5. **独立复算**：t5/t10 的配对 ΔElo、CI、sign 检验均被对抗审查从 `games.jsonl` 逐副重算，
   与 `paired_compare` 输出一致；leaf EV 复算与 metrics 一致（≤2e-9）。
6. **采集器 K 世界同种子（修复记录）**：`trunc_leaves.py` 的抽取改为每决策共享一个生成器
   （与生产语义对齐）；`contract_check.py` 同步修正两点——对照世界用同一约定构建、
   记录值与直接对照按 float32 存储精度比较（世界不再逐位相同时，float32 记录的 ~1 ulp ≈2e-6
   舍入差会超过旧的 atol=1e-6，旧世界全同值时被掩盖）。复验：`contract_check` 30/30 通过。

## 4. 判定与后续

- 按预注册：H1/H2 均为 `negative`（CI 跨 0、点估计 < +5）→ **不触发 fresh-bank confirm**，
  **不部署**；`t_leafq` 保持部署默认。
- 建议收口：**关闭"同 recipe 重采叶子重训 critic"**（N-24）。但在关闭前应先用修好的采集器
  做一次 **clean re-run（仅 t=5，~1.5h：50min 采集 + 10s 训练 + ~25min same-bank h2h）**，
  否则 N-24 建立在单世界近似分布上（见 §0 第 6 条）。value 线若继续，只剩你列过的
  容量方向（critic 头加宽 / 解冻 trunk 最后一层；N-2 关闭的是 policy 容量、不覆盖 critic）——
  低成本但先验中等偏低；此后更该转 belief 后验（改信息上界）与 policy warm-start 的 confirm。
- 相关但独立：本日 P4a `search_leafq` 已发布进 manifest 第 11 档（见
  `../search-config-plan.md` §4.3、`experiments/README.md` §0 第 14 条）；该 rung 的
  搜索配置（t5/K32/C6）与本波 v2 的失败共同支持"价值精度已不是瓶颈"的判断。

## 5. 证据落点

- 预注册/修正：`runs/o4lite-search/ei3_value_loop/{preregistration.md,t10_amendment.md,README.md}`
- 采集：`runs/o4lite-search/ei3_value_loop/{c5,c5leaves,c10,c10leaves}/shard*_*`
- 训练/诊断：`.../{v2_t5,v2_t10}/{critic.pt,metrics.json,history.json}`、`.../leaf_ev_c{5,10}.json`
- 真人 screen：`.../{screen_t5.json,screen_t10fix.json}`
- h2h：`.../h2h/{base_t5,v2_t5,base_t10,v2_t10}/`、`.../h2h/paired_{v2_t5_vs_base_t5,v2_t10_vs_base_t10}.json`、`.../h2h_t10.status`
- wd：`.../wd/{preregistration.md,status.txt,v2_t5_wd0,v2_t5_wd01,h2h/paired_wd01_vs_wd0.json}`

## 6. round 2（clean K=32 re-run，2026-09-29）

> **状态**：完成（2026-09-29）。采集器 K 世界同种子 bug 修复后的一次 clean t=5 复核。
> **H1（N-24）在 clean 数据上仍为 negative**；同数据上的 **cap0 容量探针 clean 重跑亦为负**
> （宽头 ΔEV 全 ≤ 0，CI 排除 0 但方向为负）。`t_leafq` 保持部署默认；不部署、不触发
> fresh-bank confirm。
> **预注册**：`runs/o4lite-search/ei3_value_loop/round2/preregistration.md`（采集启动前冻结，
> launch_time 12:11:59，见 `round2/launch_time.txt`）。原始产物在
> `runs/o4lite-search/ei3_value_loop/round2/`（gitignore）。
> **口径**：revision-3、obs v5、2 家、修复后引擎；seed 9400 全 fresh。

### 6.1 E0 采集（clean K=32）

- **Contract gate（部署 critic）**：`t_leafq` @ t=5、K=32、C=6、6 views，30/30 checks 通过，
  `max_abs_diff=0.0`（`round2/contract/report.json`）。
- **Bank freshness**：seed **9400** 与全部已记录 bank 零交集（665 个 games 文件、42,586 个
  unique deal seeds；exact 与 randrange-only 流均 0 交集），deal-seed 流与采集器 RNG 逐位一致
  （6 shard 全 match；`round2/bank_freshness.json`、`collection_verification.json`）。
- **规模/闸门**：6 shard rc=0；**55,792 决策 / 384,221 叶子** / 3,600 games；fallback 0、
  `rollout_terminal_rate` 1.0、aborts 0；`q_std` 非零占比 **81.3%**。
- **世界多样性 sanity**：每 `(deal, seat, game, decision)` 组 distinct obs 均值 **6.970**/8
  （round 1 同口径为 **1.000**；修复冒烟 K=8 为 7.24），平均 7.97 leaves/组，p05 = 1（少数组
  因截断内终局只留 1 个世界）。
- 训练/判定数据行数：total 384,221 / train 327,933 / val 56,288（`deals[::7]`）。

### 6.2 E1 训练 + cap0 clean（Q2）

`v2_t5_clean` 与 `t_leafq` 同 recipe（leaf_mc/all/subsample 8/20 epoch/3e-4/1024/seed 0/
holdout 7），只换 clean 数据；policy/trunk 与 P0 逐位一致（`policy_identity.json`：
仅 `critic.weight/bias` 不同，其余 6 key `torch.equal`，E4 行为验证 0.00）。同 val 行 EV：

| critic | val EV（clean c5leaves, `deals[::7]`） |
|---|---|
| P0 baseline | 0.5444498 |
| `t_leafq` | 0.5476783 |
| `v2_t5`（round 1，旧数据 best epoch） | 0.5467130 |
| `v2_t5_clean` best epoch 17 | **0.5503474** |
| `v2_t5_clean` final epoch 20 | **0.5497335** |

cap0 clean（`round2/cap_clean/`，固定 epoch 20、无 epoch 选择；与 `v2_t5_clean` 同数据/seed，
`ctl` final EV 与 `v2_t5_clean` final EV 逐位一致，diff 0.0）：

| arm | params | final EV | ΔEV vs ctl | 95% CI | 门 |
|---|---|---|---|---|---|
| `ctl` (Linear) | 129 | 0.5497335 | — | — | — |
| `w128` | 16,641 | 0.5429606 | **−0.0067728** | [−0.0085347, −0.0050801] | no hit |
| `w256` | 33,281 | 0.5486762 | **−0.0010572** | [−0.0019366, −0.0001939] | no hit |
| `w256x2` | 99,073 | 0.5288245 | **−0.0209090** | [−0.0238077, −0.0180811] | no hit |

leaf-target 方差分解（clean）：between-decision 52.70% / within-decision 47.30%（round 1 因
8× 复制为 between 100%）——clean 数据首次显示真实世界噪声。**Q2 判定**：无宽臂点估计 > 0
（CI 均排除 0 但为负）→ **容量线负关闭**，不进入 Step 1；critic 头部加宽不是瓶颈。

### 6.3 E2 真人决策 screen（诊断）

479 决策 / 370 searched / 0 fallback；world-cache 位级 sanity 全 0（
`round2/screen_t5_clean.json`）。

| 候选 | DQ vs `t_leafq` | 95% CI | Δbias vs `t_leafq` | 95% CI |
|---|---|---|---|---|
| `v2_t5_clean` @t5 | −0.0372 | [−0.1458, +0.0614] | +1.1638（更乐观） | [+0.9299, +1.3817] |

与 round 1（+0.0202 / −1.1657）符号相反、均跨 0；诊断不改变 E3 判定。

### 6.4 E3 same-bank h2h（N-24 决定性）

seeds 30/31/32 × 400 副（n=1200），K=32/C=6，t=5，rollout 对手 raw P0；机制门 6/6 全清
（fallback 0、terminal 1.0、aborts 0；`round2/h2h/mechanism_scan.json`）。

| 臂 | vs raw（sanity，非权威） | 主配对 ΔElo | 95% CI | sign p | 判定 |
|---|---|---|---|---|---|
| `r2_v2_t5_clean − r2_base_t5` | 144.02 / 145.43 | **−1.37** | [−11.99, +8.91] | 0.628 | **negative** |

（d_margin −0.183 [−1.488, +1.163]、d_win −0.00167 [−0.0146, +0.0108]。）
自检：从两臂 `games.jsonl` 独立复算与 `paired_compare` 逐位一致（max abs diff **0.0**），
两臂同为 1200 个 deal seeds（`round2/selfcheck.json`）。
E4 裸 policy 身份：`v2_t5_clean` raw vs P0 = **0.00 [0.00, 0.00]**（3 seeds 全 0）。

### 6.5 判定

- **N-24（clean）**：negative（CI 跨 0、点估计 −1.37 < +5）→ 在真正 K=32 分布上重采叶子 +
  同 recipe 重训仍不转化，round 1 的关闭结论**成立**；`t_leafq` 保持部署默认。
- **cap0（clean）**：no hit（宽头 ΔEV 全为负且 CI 排除 0）→ **critic 容量线负关闭**（解冻
  trunk 仍未被覆盖）。
- 不做 fresh-bank confirm、不部署；value 侧剩余未测方向只有 trunk 微调/解冻与 belief 后验。

### 6.6 round 2 证据落点

- `round2/{preregistration.md,launch_time.txt,bank_freshness.json,collection_verification.json}`
- `round2/contract/{report.json,contract.log}`
- `round2/{c5,c5cleanleaves}/shard*_*`
- `round2/cap_clean/{report.json,critic_*.pt,train.log}`、`round2/v2_t5_clean/{critic.pt,metrics.json,history.json}`
- `round2/screen_t5_clean.json`、`round2/h2h/{r2_base_t5,r2_v2_t5_clean,paired_clean_vs_base.json,mechanism_scan.json,policy_control_v2_t5_clean/}`
- `round2/{selfcheck.json,policy_identity.json,e1_clean_val_ev.json}`

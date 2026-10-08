# Warm-start 续训（t23wswd）：v5 优化线首个小正增益（探索性；k=7 后未达行动门槛）

> **2026-09-29 确认更新（seeds 4–7，k=7，按 §6.4 补训）**：**正增益被确认存在，但低于行动门槛**。
> 合并结果 vs `l1_1M` **+8.20**（z-CI [+5.35,+11.05]、t-CI [+4.65,+11.76]，df=6）/ vs `l1_2M`
> **+7.38**（z [+3.96,+10.79]、t [+3.11,+11.64]）→ 这两个参照下 **z/t CI 均排除 0**（不是噪声）；
> 但均 **未达 README §3 行动门槛（点 ≥ +10）**。对最强参照 `t18poolself_s2` **+3.91**
> （z [+0.14,+7.68]、t [−0.79,+8.62]）→ **未确认**。k=3 的 +11.48 回落到 +8.2（小样本高估），
> 新 seed sd 3.85–5.09（原 2.16）。**结论口径**：warm-start 续训 = **已确认的小正增益
> （约 +7~+8 vs 起点/2M；对最强参照 ~+4 未确认）**，探索性、6 因素捆绑、低于 +10；
> 「首个过门槛」表述作废。§6.1 单因子/朴素对照仍未跑，归因不清；裸 policy 口径，不能直接当部署 Elo。
> 证据：`runs/t23/warmstart_confirm_report.md`、`runs/t23/confirm_combined_*.txt`、
> `runs/t23wswd__{4,5,6,7}__*`、合并脚本 `runs/t23/combine_confirm.py`（复现 k=3 逐位一致）。
>
> **状态**：完成（2026-09-27），**探索性、未预注册**（用户明确要求不做严格对照，先看方向）。
> **口径标签**：revision-3（出空即撬底，`rules_id=2e36dbea44893696`）、obs v5（2 家 161 维）、
> memoryless MLP（`arch=shared`、`hidden=128`、单层 trunk）、PPO。比较为 deal-twin 换座 h2h
> （每条 400 副/deal seed，按副聚类 bootstrap）；行动门槛 = CI（z 与 t）完全排除 0 且点估计 ≥ +10。
> 参照模型 = **warm-start 起点本身**（`l1_1M`），即"续训是否超过起点"的配对比较。

## 0. 结论

1. **续训确实超过了起点**：3 个 seed 的最终读数全部为正、**单 seed CI 均排除 0**；
   合并（k=3）**Δ = +11.48**，z-CI **[+7.39, +15.58]**，t-CI **[+2.49, +20.47]**。
   这是 memoryless v5 优化线（含 EVH/self-play/奖励/batch/容量各波）里**第一个满足仓库行动门槛**
   （CI 排 0 且点估计 ≥ +10）的结果。
2. **对第二参照与已知最强模型同样为正（2026-09-27 追加确认）**：vs `l1_2M` **+9.73**
   （z-CI [+2.20, +17.27]；t-CI [−6.81, +26.28]），vs `t18poolself_s2` **+7.25**
   （z-CI [+2.42, +12.08]；t-CI [−3.36, +17.86]）；**3 个 training seed 对两个参照全部为正**，
   且两个新 z-CI 都包含原 +11.48 → **增益不是参照/起点特异的**。k=3 下对更硬参照的 t-CI 跨 0
   （点 < +10），严格确认需 k≈7（见 §6）。
3. **seed 间方差骤降**：三个 seed 的最终读数 +10.92 / +9.66 / +13.87，**sd = 2.16**（对比：
   从零训练的 F1b sd 8.43、自博弈池化 A2 sd 11.92、batch 实验 sd 14.4）。与"共享起点显著压
   低 run 间方差"的假设一致——若后续对照复现，未来实验的可测性会提升一个量级。
4. **cosine 重启的代价很小**：第一个快照（102k，LR 仍在峰值附近）三个 seed 为
   −3.0 / −6.7 / −8.1，CI 全跨 0；到 300k 已转为正。
5. **但归因不干净**（探索性）：本波同时改了 6 个因素——warm start、1M 预算、cosine、
   AdamW+wd=0.01、强池对手、batch×2。尚不能区分"光是 warm-start 续训"与"配方加成"。

## 1. 配方与实现

| 维度 | 设置 |
|---|---|
| warm start | `runs/t17long__1__1790439615/snapshots/checkpoint_step1024000.pt`（`l1_1M`，发布 lvl4） |
| 预算 | `--total-timesteps 1000000`（实际落 999,424 = 488 update × batch 2048） |
| 优化器 | `--optimizer adamw --weight-decay 0.01` |
| LR | `--lr-schedule cosine`（峰值 2.5e-4；相位与 linear 一致，末点为一步相位而非恰好 0） |
| 对手 | `--opponent pool --pool-episode True`，成员为共享 fit（`runs/t17_screen/mle.json`）中 μ>150 的 6 个：`p1_262k`(177.6)、`p1_499k`(190.1)、`l1_512k`(187.6)、`l1_1M`(191.6)、`l1_1.5M`(187.7)、`l1_2M`(191.1) |
| batch | `--num-envs 16 --num-minibatches 8`（batch 2048、minibatch 256，minibatch 不变） |
| 快照 | `--snapshot-interval 50`（~102,400 步/份），落 `runs/t23wswd__*/snapshots/` |
| 训练 seed | 1 / 2 / 3 |

**为本次实验合入的代码**（`src/seven523/train.py` + `tests/test_train.py`，默认路径逐位不变）：

- `--lr-schedule {linear,cosine}` + 纯函数 `lr_scale(schedule, update, num_updates)`；
- `--optimizer {adam,adamw}` + `--weight-decay`（默认 adam/0.0；`wd>0` 必须配 adamw，fail-loud）；
- `--snapshot-interval N`（默认 0 = 关闭）；
- 新增 3 个单测；`pytest tests/test_train.py` **42 passed**；另做了一次端到端 smoke。

## 2. 最终评测（9 deal seed 150–158，每条 3600 副 / 7200 局）

| train seed | Elo Δ vs `l1_1M` [95% CI] | winrate | 平均分差 |
|---|---|---|---|
| 1 | **+10.92** [+3.88, +17.95] | 0.5157 | +2.65 |
| 2 | **+9.66** [+2.58, +16.74] | 0.5139 | +2.43 |
| 3 | **+13.87** [+6.71, +21.03] | 0.5199 | +2.53 |

**合并 k=3**：点估计 **+11.48**；z-CI [+7.39, +15.58]；t-CI [+2.49, +20.47]；
bootstrap RMS 3.62、**seed sd 2.16**（binding term = bootstrap）。

### 2.1 第二参照确认（2026-09-27 追加；fresh deal seed 160–168，9 seed × 400 副）

| 参照 | per-seed (s1/s2/s3) | 合并 | z-CI | t-CI | z 排 0 | t 排 0 | ≥ +10 |
|---|---|---|---|---|---|---|---|
| `l1_1M`（起点，原读数） | +10.9 / +9.7 / +13.9 | **+11.48** | [+7.39, +15.58] | [+2.49, +20.47] | ✓ | ✓ | ✓ |
| `l1_2M`（第二参照） | +8.7 / +16.9 / +3.7 | **+9.73** | [+2.20, +17.27] | [−6.81, +26.28] | ✓ | ✗ | ✗ |
| `t18poolself_s2`（已知最强） | +8.7 / +10.2 / +2.9 | **+7.25** | [+2.42, +12.08] | [−3.36, +17.86] | ✓ | ✗ | ✗ |

- **方向稳健**：三个 training seed 对两个新参照全部为正；两个新 z-CI 都包含原 +11.48，
  说明增益不是"只对起点特异"。对当前最强已知模型的诚实增益量级 ≈ **+7**。
- **k=3 未达严格门槛**：换成更硬参照后点估计 < +10、t-CI 跨 0。按现有 sd，扩到 **k=7**
  预计 t-CI 可排 0（约 +4 run × 1M）。
- **中段峰值复测（holdout）**：`s1@716.8k` 从 +21.9 掉到 **+6.81 [−0.09, +13.71]**；
  `s3@819.2k` 从 +26.7 掉到 **+11.85 [+2.61, +21.09]** → 快照"峰值"含明显选择偏差，
  **最终 `agent.pt` 才是可报告产物**。
- 产物：`runs/t23/h2h_ref2m_*.json`、`runs/t23/second_reference_report.md`。

### 2.2 绝对强度锚（共享 probit-MLE fit，2026-09-27 追加）

`tools/build_ladder.py --seed 41`（7 候选 + `random` 锚，400/400，11,200 局，含 `lvl3` 定标）
+ `refit_mle --bootstrap 200`：

| id | μ | σ | 95% CI | n |
|---|---:|---:|---|---:|
| ws_s2 | 188.03 | 5.13 | [176.50, 201.80] | 2800 |
| l1_2M | 187.72 | 5.16 | [176.59, 202.44] | 2800 |
| ws_s1 | 186.58 | 5.13 | [174.22, 197.45] | 2800 |
| ws_s3 | 182.60 | 5.12 | [172.97, 196.27] | 2800 |
| l1_1M | 182.58 | 5.13 | [170.66, 196.15] | 2800 |
| pself_s2 | 173.50 | 5.10 | [162.96, 188.42] | 2800 |
| lvl3 | 135.53 | 5.08 | [123.14, 149.38] | 2800 |
| random | 0.000 | 0.00 | [0, 0] | 2800 |

（`converged=true`、`margin_identified=true`、`separation=[]`）

- 同 fit 内对比：`mean(ws_*) − l1_1M` = +3.15 [−5.62, +13.57]；`− l1_2M` = −1.98 [−11.13, +6.99]；
  `− pself_s2` = **+12.24 [+3.15, +21.19]**。
- 该 fit 内嵌的直接对局（每对 400 局）：ws_* vs `pself_s2` 三条全正（winrate .531–.554）；
  ws_* vs `l1_1M` 混合（.487 / .525 / .503）——+7~+11 的效应在 400 局粒度上淹没在噪声里。
- **读法（重要）**：绝对表只用于**定阶位**（顶部 ≈183–188、`lvl3` 135.5、random 0），其对比 CI ≈±10–15，
  **分辨率不足以验证 +7~+11 的差异**；按 ADR-0013，门禁只认 deal-twin h2h（§2.1）。fit 还把
  `pself_s2` 放到 173.5（低于 `l1_1M`）而两者直接样本是平局，说明单次 fit 对小样本/对局网结构敏感。
- 产物：`runs/t23/anchor/{games.jsonl, absolute_table.json, absolute_anchor_report.md}`。

## 3. 进步曲线（每个快照 vs 起点，3 deal seed 150–152）

| step | seed 1 | seed 2 | seed 3 |
|---|---|---|---|
| 102,400 | −3.0 | −6.7 | −8.1 |
| 204,800 | +3.8 | −8.1 | +0.4 |
| 307,200 | +9.0 | +2.5 | +20.0 |
| 409,600 | +10.7 | +13.5 | +10.3 |
| 512,000 | +12.2 | +6.1 | +15.8 |
| 614,400 | +8.5 | −0.1 | +16.1 |
| 716,800 | **+21.9** | +3.5 | +21.0 |
| 819,200 | +19.0 | +7.7 | **+26.7** |
| 921,600 | +17.1 | +7.0 | +25.0 |
| final (999,424) | +10.9 | +9.7 | +13.9 |

观察：曲线在 300k–920k 之间普遍高于 final；单快照峰值（s1@716.8k +21.9、s3@819.2k +26.7）
比 final 高 +8~13，但单快照只有 3 deal seed，不能直接拿来选 ckpt（选择必须预注册规则，
否则是 winner's curse）。完整数据见 `runs/t23/progress.csv`。

## 4. 局限（必须同时读）

- **未预注册、探索性**：不是 confirmatory 端点，参照是起点自身（自我提升对比），不是固定外部锚。
- **6 因素捆绑**：warm start / 1M / cosine / AdamW+wd / 强池 / batch×2 同时改变，无法归因单项。
- **无绝对强度锚**：第二参照（`l1_2M`）与已知最强（`t18poolself_s2`）的 h2h 已完成（§2.1），
  但尚未把新模型放进共享 MLE fit（`build_ladder`+`refit_mle`），绝对 μ 与阶梯档位未知。
- **快照评测只用 3 deal seed**（±12–15 Elo），final 用 9 deal seed。
- 机器上与另一个会话的 `ra_*` 实验并发（共享 CPU），吞吐约 355 sps/run。
- **共享起点**：三个 seed 从同一个 `l1_1M` 出发，seed 方差低也可能部分来自"起点太强/太收敛"，
  不代表从零训练也会有同样低方差。

## 5. 复现

```bash
mkdir -p runs/t23
for s in 1 2 3; do
  nohup nice -n 5 .venv/bin/python -m seven523.train --exp-name t23wswd --seed "$s" \
    --total-timesteps 1000000 \
    --load-checkpoint runs/t17long__1__1790439615/snapshots/checkpoint_step1024000.pt \
    --optimizer adamw --weight-decay 0.01 --lr-schedule cosine \
    --num-envs 16 --num-minibatches 8 --opponent pool --pool-episode True \
    --pool-member 1@ckpt:runs/t17pool__1__1790439615/snapshots/checkpoint_step262144.pt \
    --pool-member 1@ckpt:runs/t17pool__1__1790439615/snapshots/checkpoint_step499712.pt \
    --pool-member 1@ckpt:runs/t17long__1__1790439615/snapshots/checkpoint_step512000.pt \
    --pool-member 1@ckpt:runs/t17long__1__1790439615/snapshots/checkpoint_step1024000.pt \
    --pool-member 1@ckpt:runs/t17long__1__1790439615/snapshots/checkpoint_step1536000.pt \
    --pool-member 1@ckpt:runs/t17long__1__1790439615/snapshots/checkpoint_step1999872.pt \
    --snapshot-interval 50 --cuda True --tensorboard True --run-dir runs \
    > runs/t23/t23wswd_s${s}.log 2>&1 &
done
```

> **配方修订（2026-10-07，[ADR-0015](../adr/0015-continuation-pool-random-mix.md)，operator 决策）**：
> 上面 §5 命令是 2026-09-27 的历史复现，保留原样；**后续新的续训默认**在该 6 成员池上
> 混入 10% RandomBot——成员权重由 `1@` 改为 `3@`，并追加 `--pool-member 2@random`
> （`2/20 = 10%`，`--pool-episode True` 下按局抽）。理由与完整命令见 ADR-0015。

评测模板（串行，CPU/4 workers；快照用 `--seeds 150,151,152`，final 用 `150..158`）：

```bash
.venv/bin/python tools/head_to_head.py \
  --left "ws_s1=ckpt:runs/t23wswd__1__<ts>/agent.pt" \
  --right "l1_1M=ckpt:runs/t17long__1__1790439615/snapshots/checkpoint_step1024000.pt" \
  --seeds 150,151,152,153,154,155,156,157,158 --pairs 400 --bootstrap 4000 \
  --device cpu --workers 4 --json > runs/t23/h2h_final_s1.json
```

**产物**：`runs/t23/`（`warmstart_report.md`、`progress.csv`、`h2h_s*_step*.json`、
`h2h_final_s*.json`、`controller.sh`、`pids.txt`）、`runs/t23wswd__{1,2,3}__*/`
（`agent.pt`、`metrics.csv`、`snapshots/`、`tb/`）。TB：`runs/t23wswd__*/tb`。

## 6. 下一步（归因与确认）

1. **朴素续训对照（关键）**：同 warm start、同 1M，但用原配方（Adam、linear、batch 1024、
   random 对手）跑 3 seed。若也 +10 → 杠杆就是"续训冠军"，cosine/wd/pool/batch 都不必要；
   若不涨 → 再对 cosine / wd / pool / batch 逐个单因子消融。
2. ~~绝对强度锚~~ **已完成（2026-09-27）**：见 §2.2 与附录 A.2/A.3——共享 fit 只定阶位，
   分辨率（±10–15）不足以验证 +7~+11。
3. ~~第二参照 h2h~~ **已完成（2026-09-27）**：见 §2.1——vs `l1_2M` +9.73、vs `t18poolself_s2`
   +7.25，方向稳健但 k=3 未达严格门槛。
4. **补 seed 到 k=7（严格确认）**：若要把"超过当前最强已知模型"写成 confirmatory 结论，
   再训 seeds 4–7（同配方，1M）后重做 vs `l1_2M`/`t18poolself_s2` 的合并。
5. **检查点选择（若要做）**：先预注册选择规则（held-out deal 选、fresh deal 上报）；§2.1 的
   holdout 复测已证明不预注册的选择会明显虚高。
6. 若以上确认：把 warm-start 续训作为新的默认配方基线，再谈多轮续训 / 更长预算 / 结构项。

---

## 附录 A. 方法学：绝对锚 vs h2h，评分漂移与"策略追逐"

> 汇总 2026-09-27 讨论的方法学结论，供本报告与后续波次引用。

### A.1 两条评分通道的分工（ADR-0013）

| | 绝对锚（共享 probit-MLE fit） | 换座 h2h（deal-twin duel） |
|---|---|---|
| 目的 | 定阶位 / 发布绝对表（同一 fit 内跨模型可比） | 比较两个模型（固定参照） |
| 设计 | 每 id 从自己的对局推断 μ，再在整张对局网联合拟合 | 同批副牌、两座位各打一遍，按副聚类的配对差分 |
| 分辨率 | 差异 CI ≈ ±10–15（本次每对 400 局，单对 SE≈30 Elo） | 9 deal seed × 400 副 = 7,200 局/对；per-seed CI ±7–10，k=3 合并 ±4–8 |
| 成本 | 11,200 局一次出全表 | ~1–1.5 分钟/条，但一次只能比一对 |
| 能回答 | "大概在哪一档"、大差距 | "A 比 B 强多少、是否显著"——小差距唯一可行工具 |
| 不能回答 | 顶部小差距 | 跨全体的绝对位置（需要参照） |

### A.2 绝对锚怎么跑

1. **采集** `tools/build_ladder.py`（`ladder.plan_games`）：同一 slot 的所有候选共享同一批 deal；每副牌两座位 twin；`cross` 候选互打是单 gauge 下的排序信息来源（ADR-0012）。本次：7 候选 + random 锚，400/400，11,200 局，每 id 2,800 局，JSONL 带 `rules_id`。
2. **拟合** `tools/refit_mle.py` / `seven523/mle.py`：ordered-probit MAP（胜/负/平共享 ±ε 阈值、同方差 `s=√2β`）；random 钉在 μ=0（gauge）；自由 id 加弱高斯先验防对 gauge 全胜发散；纯 Python 阻尼 Newton；`σ` = Laplace 后验 sd；95% CI = 按副聚类 200 次 bootstrap。
3. **为什么"绝对"**：gauge 钉位置、β 钉尺度；**跨 fit/跨 rules_id 不可比**（本 fit `l1_1M`=182.6，而 `t17_screen` 那次 fit 是 191.6）。

### A.3 为什么绝对锚分不出 +7~+11

- 每对只有 400 局（200 副 × 2 座位）→ 单对差异 SE ≈ 30 Elo；fit 把每个 id 的估计噪声都带进差分 → 顶部对比 CI ≈ ±10–15。
- 要让锚分辨 +10 需要每对约 30 倍局数（≈1.4 万局/对 × 21 对 ≈ 30 万局），不现实。
- h2h 把 7,200 局集中在**同一对**上且配对消方差，所以有分辨率。
- 本次锚内嵌的直接 400 局样本与 h2h 方向一致但不显著：ws_* vs `pself_s2` 三条全正（winrate .531–.554）；ws_* vs `l1_1M` 混合（.487/.525/.503）。

### A.4 评分设计避免了什么、没有避免什么

**已避免（无漂移通道，ADR-0013）**：
- 在线评分的顺序漂移/追逐：发布只用 order-free MLE；在线 OpenSkill 只服务匹配/训练（ADR-0011/0013）。
- 坐标漂移：单 gauge `RandomBot=0`；尺度由 β 钉住；不做多锚、不做 Nash 发布尺度（ADR-0012/0013）。
- 跨规则版本混池：`rules_id` 门禁，跨版本只做 bridge h2h。
- 座位/牌运/锚点距离偏差：deal-twin 换座 + 副聚类 bootstrap。
- 用 online `sigma` 冒充标准误：门禁只用 h2h Δ+CI。

**没有避免**：
- **非传递性/循环克制**：1-D 标量表达不了，只做 pair 残差诊断（`arena.pair_diagnostics`），不修正 → 需要多参照 / held-out 对手。
- **参照选择 / winner's curse**：协议问题而非评分器问题；本次用 3 个参照把幅度从 +11.5 收敛到 +7.3。
- **统计分辨率**：评分结构不产生功效（见 §A.3）。
- **"进步"的定义**：rating 只给相对强弱，不回答"更会打什么局面"。
- **训练侧 self-play 追逐/退化**：由 t18 三臂实验单独查过（A1/A2/A3 全 null），未检测到退化。

### A.5 对本报告结论的解释边界

- **+7~+11 不是评分漂移/策略追逐的产物**：发布通道（order-free MLE + 单 gauge + rules 身份）与比较通道（deal-twin 配对）都按无漂移设计。
- "更强"的证据仍由 h2h 提供（唯一有分辨率的通道）；3 个参照均确认方向（每个 training seed 对每个参照全为正）。k=7 扩 seed 是**补统计置信度**，不是防追逐。
- 绝对锚的价值在于**结构独立的正交检查**：证明新模型在池顶（≈183–188）、未出现"仅对单参照特异"的异常。

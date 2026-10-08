# Stage-0 D-D：run 间 seed 方差 screen（target-kl / 低 lr 能否当廉价稳定器）

> 状态：**完成**（2026-10-06）。HEAD `0fd5c2e`（2026-09-29），revision-3
> （`rules_id=2e36dbea44893696`）、obs v5（2 家 161 维）、PPO memoryless MLP
> （`arch=shared`、`hidden=128`、单层 trunk、`relu`）。
> 本文只写自己的产物目录 `runs/stage0/variance/`，不改 `src/` / `tests/` / `tools/`
> / `traces/` / `.pi/` / 其他 `docs/`，不 commit、不发布 ckpt/manifest、不改
> `obs_version`。设计出处：`docs/post-v5-structural-options.md` §4 D-D。

## 1. 预注册（先于开跑写定）

### 1.1 假设

- **H-T（target-kl early-stop）**：把 `--target-kl 0.03` 打开后，PPO 每 rollout 的
  epoch 循环会在 `approx_kl > 0.03` 时提前退出（`src/seven523/ppo.py:246`），相当于
  缩小每次策略更新的步长/上限，可能降低跨训练 seed 的最终强度方差。
- **H-L（低学习率）**：`--learning-rate 1e-4`（默认 `2.5e-4`）用更小更新步长，可能
  降低最终解的 seed 间方差。
- **对照 H0**：两个旋钮都不改变 run sd（即默认配方本身），D-D 关闭、无廉价稳定器。

> 先验：低。t20/t21 已证明 batch 放大不能降 run sd（`docs/post-v5-structural-options.md`
> §1 证据地图）。`--target-kl` 的生效性必须在本轮**实际确认**（见 §1.6）。

### 1.2 臂与预算

| 臂 | 配方（其余全部 = 默认配方） | 训练 seed | 步数 |
|---|---|---|---|
| **B（基线）** | 默认配方，无额外旋钮 | 1 / 2 / 3 | 200k |
| **T** | 默认 + `--target-kl 0.03` | 1 / 2 / 3 | 200k |
| **L** | 默认 + `--learning-rate 1e-4` | 1 / 2 / 3 | 200k |

默认配方 = `arch shared`、`hidden 128`、`activation relu`、`num_envs 8 × num_steps 128`
（batch 1024）、`num_minibatches 4`（minibatch 256）、`update_epochs 4`、
`lr 2.5e-4`、`--anneal-lr --lr-schedule linear`、`optimizer adam`、`weight-decay 0`、
`clip-coef 0.1`、`clip-vloss`、`norm-adv`、`vf-coef 0.5`、`ent-coef 0.01`、
`max-grad-norm 0.5`、`gamma 0.99`、`gae`、`gae-lambda 0.95`、`reward-shaping terminal`、
`opponent random`、`torch-deterministic`、`cuda`。obs v5 = 默认观测（`seq_len 0`、
`event_len 0`），`observation_dim(2)=161` 已现场确认。

`total-timesteps 200000` → `num_updates = 200000 // 1024 = 195`，实际 `global_step =
195 × 1024 = 199680`（与仓库既有 200k 步口径一致）。

### 1.3 端点（run 间 sd，不是均值）

- **主端点**：每个 run 的最终 `agent.pt` 与**固定参照对手** h2h 的 `elo_diff`
  （左 = 本 run，右 = 参照），在同一**固定 deal seed 集**上。参照 =
  `runs/t17long__1__1790439615/agent.pt`（默认配方 2M，seed 1，obs v5、同 arch，
  已确认可加载）。deal 集用单一 `--seed 7777 --pairs 6400`（6400 副 = 12800 局，
  换座配对），每 run 独立 bootstrap（`--bootstrap 8000`）。
- **副端点（线性、无 logistic 放大）**：同一 run 对 `random` 的 `mean_score_diff`
  与 `winrate`，固定 `--seed 8888 --pairs 800`（1600 局）。
- **测量噪声**：主端点每 run 的 `elo_diff_ci` 半宽 / 1.96 作为该 run 的测量 SE；报告
  三 run 的池化 bootstrap SE，并从观察 sd 中减去 `se²` 得到**校正 run sd**。
- **统计量**：每臂报 3 条 run 值、均值、run sd（`ddof=1`，观察值与校正值都报），以及
  `df=2` 的 sd 不确定性（95% CI = `s·sqrt(df/χ²_{0.975,df})` 到
  `s·sqrt(df/χ²_{0.025,df})`，`df=2` → `[0.52·s, 6.28·s]`）。

### 1.4 判据（预注册）

- 任一臂的 run sd（主端点，校正后）≤ 基线 B 的一半（基线若约 13 Elo → 门槛约 ≤ 7）
  → 记为**候选稳定器**，并注明需 k≥7 复算才能采用。
- 否则记 D-D **关闭、无廉价稳定器**。

### 1.5 分析计划

1. 每个训练 run 完成后，读 `metrics.csv` 的 `approx_kl` / `sps` / 其他标量。
2. 主端点与副端点评估命令见 §1.7；结果按臂聚合 3 条 run 值 → mean / sd / 校正 sd /
   df=2 sd CI。
3. **target-kl 生效性确认**：对比 T 与 B 的 `approx_kl` 分布与 `sps`（见 §1.6）；并用
   一次性只读脚本在 `runs/stage0/variance/` 下对 shipped `ppo_update` 做 minibatch 步数
   探针，证明 `ppo.py:246` 的 early-stop 在 `approx_kl > target_kl` 时确实减少
   optimizer 步数。
4. 主端点 sd 判定按 §1.4。

### 1.6 `--target-kl` 生效性预判与确认方式

`--target-kl` 只出现在 `src/seven523/ppo.py:34`（字段）与 `ppo.py:246`（early-stop
break）。因此它**只可能**通过减少每 rollout 的 minibatch 步数产生影响。

预检（已现场复核，`runs/t17long__1__1790439615/metrics.csv`，同默认配方）：该 run 在
前 195 个 update（≈200k 步）里 `approx_kl` 的 `max=0.0059`、`p90=0.0043`，**全部远低于
0.03**。这意味着在默认配方 + 200k 预算下，`--target-kl 0.03` 很可能**永不触发**
early-stop，T 将与 B **逐位相同**（no-op）。若 B 三 seed 的 `approx_kl` 全部 < 0.03 且
T 与 B 的 `metrics.csv` 逐位一致，则如实记录「T 为 no-op、非真正稳定器测试」；若触发，
则以 `approx_kl` / `sps` 对比为准。

### 1.7 评估命令（预注册，写清以可复现）

训练（每臂每 seed 一条；`--run-dir runs/stage0/variance`；`--tensorboard false`、
`--checkpoint-interval 0`、`--snapshot-interval 0`、`--eval-interval 0`，只留最终
`agent.pt` 与 `metrics.csv`）：

```bash
cd /home/amas/.local/src/7g523
.pi/limits/run_limited.sh gpu .venv/bin/python -m seven523.train \
  --exp-name dvar_B --seed 1 \
  --total-timesteps 200000 \
  --arch shared --hidden-size 128 --activation relu \
  --num-envs 8 --num-steps 128 --num-minibatches 4 --update-epochs 4 \
  --learning-rate 2.5e-4 --anneal-lr --lr-schedule linear \
  --optimizer adam --weight-decay 0 \
  --opponent random --reward-shaping terminal \
  --torch-deterministic --cuda True \
  --checkpoint-interval 0 --snapshot-interval 0 --eval-interval 0 --log-interval 1 \
  --tensorboard false --run-dir runs/stage0/variance
# T 增加： --target-kl 0.03
# L 改为： --learning-rate 1e-4
```

主端点评估（固定 deal 集，每 run 一条）：

```bash
cd /home/amas/.local/src/7g523
.pi/limits/run_limited.sh gpu .venv/bin/python tools/head_to_head.py \
  --left "dvar_<ARM>_s<SEED>=ckpt:runs/stage0/variance/<RUN_DIR>/agent.pt" \
  --right "ref=ckpt:runs/t17long__1__1790439615/agent.pt" \
  --seed 7777 --pairs 6400 --bootstrap 8000 --device cpu --workers 4 --json \
  > runs/stage0/variance/eval/h2h_<ARM>_s<SEED>_vs_ref.json
```

副端点评估（固定 deal 集）：

```bash
.pi/limits/run_limited.sh gpu .venv/bin/python tools/head_to_head.py \
  --left "dvar_<ARM>_s<SEED>=ckpt:runs/stage0/variance/<RUN_DIR>/agent.pt" \
  --right "rnd=random" \
  --seed 8888 --pairs 800 --bootstrap 8000 --device cpu --workers 4 --json \
  > runs/stage0/variance/eval/h2h_<ARM>_s<SEED>_vs_random.json
```

### 1.8 Amendment log

- **A1（2026-10-06，训练后、T 评估前）**：9 个训练 run 完成后，现场确认 `--target-kl 0.03`
  在默认配方 + 200k 预算下**从不触发**（9 个 run 的 `kl_max ≤ 0.0047 ≪ 0.03`），且
  T 与 B 同 seed 的 `agent.pt` 权重**逐位相同**（`metrics.csv` 除 `sps` 外逐位相同）。
  据此把 T 判定为 **no-op（非真正稳定器测试）**，原计划**跳过 T 的三次 h2h 评估**。

- **A2（2026-10-06，父代理 follow-up 后）**：A1 的“跳过 T 评估”被父代理纠正——T 是预注册
  三臂之一，产物不能缺。已用与 B/L **完全相同**的命令补跑 T 的 3 seed ×
  （vs ref deal seed 7777 / vs random deal seed 8888），走 `run_limited.sh gpu`；
  `run_eval.sh` 的臂列表已改回 `dvar_B dvar_L dvar_T`。补跑结果与 B **逐位相同**（§2.3/§2.4），
  进一步用真实 eval 产物证实 T≡B。A1 中“跳过 T 评估”的表述作废，替换为 A2。

---

## 2. 执行记录与结果

### 2.1 训练产物（`runs/stage0/variance/`，全部 rc=0）

| 臂 | run 目录 | 步数 | 备注 |
|---|---|---|---|
| B s1 | `dvar_B__1__1791305104` | 199680 | 默认配方 |
| B s2 | `dvar_B__2__1791305255` | 199680 | |
| B s3 | `dvar_B__3__1791305436` | 199680 | |
| L s1 | `dvar_L__1__1791305625` | 199680 | lr 1e-4 |
| L s2 | `dvar_L__2__1791305811` | 199680 | |
| L s3 | `dvar_L__3__1791306003` | 199680 | |
| T s1 | `dvar_T__1__1791306193` | 199680 | target-kl 0.03（no-op）|
| T s2 | `dvar_T__2__1791306385` | 199680 | |
| T s3 | `dvar_T__3__1791306584` | 199680 | |

训练命令见 §1.7；日志 `runs/stage0/variance/logs/dvar_<ARM>_s<SEED>.log`。

### 2.2 `--target-kl 0.03` 生效性确认（结果：不生效，T 为 no-op）

- **代码路径**：`--target-kl` 只被 `src/seven523/ppo.py:246` 的
  `if config.target_kl is not None and approx_kl > config.target_kl: break` 读取。
  **机制探针**（`runs/stage0/variance/probe_target_kl.py`，只读、未改 src）：在人为构造
  大 ratio 的 batch 上，`target_kl=None` 执行 **16** 次 optimizer 步，`target_kl=0.03` 执行
  **4** 次（early-stop 触发）→ shipped 代码的 early-stop 分支确实存在且有效。
- **本轮真实数据**：9 个 run 每 update 的 `approx_kl` 全量统计（`metrics.csv`）：

  | 臂 | `kl_max`（3 seed） | `kl_mean` | `kl>0.03` 的 update 数 |
  |---|---|---|---|
  | B | 0.0041 / 0.0047 / 0.0047 | 0.0019 | **0 / 195**（每 seed）|
  | L | 0.0024 / 0.0026 / 0.0014 | 0.0005 | **0 / 195** |
  | T | 0.0041 / 0.0047 / 0.0047 | 0.0019 | **0 / 195** |

  即本轮 `approx_kl` 恒 ≪ 0.03，early-stop 分支**一次都没触发**。
- **逐位一致性**：T 与 B 同 seed 的 `agent.pt` 模型权重**逐位相同**（三 seed 均一致）；
  `metrics.csv` 仅 `sps`（墙钟吞吐，受机器负载影响）不同，其余所有列（含 `approx_kl`、
  `episodic_return`、损失）逐位相同。
- **判定**：`--target-kl 0.03` 在本配方 + 200k 预算下是 **no-op**，T 不构成稳定器测试；
  其 run sd 与 B 相同（T≡B）。补跑的 T 三次 h2h 结果与 B **逐位相同**（见 §2.3/§2.4 表）。

### 2.3 主端点：`elo_diff` vs 固定参照（`runs/t17long__1__1790439615/agent.pt`，deal seed 7777，6400 副）

每 run 的 `elo_diff`（左 = 本 run，右 = 参照；负 = 本 run 更弱）、bootstrap 95% CI 与
测量 SE（CI 半宽 / 1.96）：

| 臂 | s1 | s2 | s3 | 均值 | run sd（ddof=1）| 池化测量 SE | 校正 run sd |
|---|---|---|---|---|---|---|---|
| B | −59.34 [−65.00,−53.71] | −51.26 [−56.82,−45.62] | −26.33 [−31.60,−20.90] | −45.64 | **17.21** | 2.82 | 16.98 |
| L | −67.08 [−72.84,−61.38] | −64.69 [−70.35,−59.00] | −60.15 [−65.84,−54.40] | −63.97 | **3.52** | 2.91 | 1.98 |
| T | −59.34 [−65.00,−53.71] | −51.26 [−56.82,−45.62] | −26.33 [−31.60,−20.90] | −45.64 | **17.21** | 2.82 | 16.98 |

- **B 的 run sd = 17.21 Elo**（校正 16.98），与历史 200k 屏幕的 14–18 Elo 同量级；
  三 seed 跨度为 33 Elo（−59.3 / −51.3 / −26.3），其中 s3 明显更强。
- **L 的 run sd = 3.52 Elo**（校正 1.98），三 seed 集中在 −67.1 / −64.7 / −60.1。
  观察 sd 已接近单 run 测量 SE（2.9），说明 L 的**真实** between-seed sd 很小。
- **T 与 B 的三条 run 值及 sd 逐位相同**（17.21 / 校正 16.98）——因为 `agent.pt` 权重
  逐位相同、deal 集相同，评估是确定性函数。这用真实 eval 产物再次证实 T≡B。
- **df=2 的 sd 不确定性**（95% CI = `s·sqrt(2/χ²)`）：B/T sd ∈ [8.96, 108.15]；
  L sd ∈ [1.83, 22.13]。n=3 下 sd 的点估计方向清楚，但置信区间很宽。
- **均值**：L 比 B 弱 −18.33 Elo（点估计；Welch t ≈ −1.8、df≈2.2、p≈0.2，**不显著**）。

### 2.4 副端点：vs `random`（deal seed 8888，800 副）

| 臂 | winrate（s1/s2/s3）| winrate sd | mean_score_diff（s1/s2/s3）| msd sd |
|---|---|---|---|---|
| B | 0.8934 / 0.9019 / 0.8828 | 0.0096 | 51.38 / 53.81 / 52.04 | 1.25 |
| L | 0.8819 / 0.8700 / 0.8697 | 0.0069 | 50.01 / 49.24 / 48.00 | 1.02 |
| T | 0.8934 / 0.9019 / 0.8828 | 0.0096 | 51.38 / 53.81 / 52.04 | 1.25 |

副端点方向一致：L 的 run 间 sd 更低，但平均强度也略低（winrate −0.014、score diff −3.3 分）；
T 与 B 逐位相同。

### 2.5 评估产物

`runs/stage0/variance/eval/h2h_<ARM>_s<SEED>_vs_{ref,random}.json`（B/L/T 三臂 × 3 seed
× 2 端点 = 18 个）；汇总脚本 `runs/stage0/variance/aggregate.py`、`aggregate.json`；评估日志
`runs/stage0/variance/logs/eval_*.err`。

---

## 3. 结论

- **T（`--target-kl 0.03`）＝ no-op**：本轮 `approx_kl` 恒 ≤ 0.0047，`ppo.py:246` 的
  early-stop 从未触发，T 与 B 的模型权重逐位相同 → 不降低 run sd；补跑的 T 三次 h2h
  与 B 逐位相同（run sd 同为 17.21 Elo），用 eval 产物再次证实。
- **L（`--learning-rate 1e-4`）把 run sd 从 17.21 砍到 3.52 Elo（校正 17.0 → 2.0），
  ≤ 基线一半（8.6）** → 按预注册判据记为**候选方差稳定器**，但**需 k≥7 复算才能采用**。
- **重要代价**：L 的 200k 均值比 B 弱 −18.3 Elo（点估计，n=3 下不显著）——它是
  “降方差但降强度”的旋钮，**不是免费/廉价稳定器**。若想作为基础配方，必须补
  配平强度的更长预算对比（例如 lr 1e-4 @ 800k vs lr 2.5e-4 @ 200k）。

一句话判定：**D-D 未找到“免费”廉价稳定器；lr 1e-4 是“降方差（sd 17.2→3.5）但
降强度（−18 Elo）”的条件候选稳定器，需 k≥7 与配平强度复核才能采用；target-kl 0.03
在本配方 200k 下为 no-op（run sd 与基线同为 17.2）。**

---

## 4. 缺失证据 / 限制

- **n=3 的 sd 推断极弱**：df=2 的 sd 95% CI 很宽（B [9.0, 108]、L [1.8, 22.1]）；
  “砍半”只是点估计结论，不能作 confirmatory 结论，需 k≥7。
- **L 的 sd 读数受测量噪声主导**：单 run Elo SE ≈ 2.9，而 L 观察 sd = 3.5，校正后 ≈ 2.0；
  真实 between-seed sd 可能在 0–几 Elo 之间。
- **均值代价未作正式 arm-vs-arm deal-twin h2h**：B vs L 的 −18.3 Elo 是各臂 vs 同一
  参照的均值差，不是 B_sN vs L_sN 的配对 h2h；D-D 只注册了 sd 端点，未注册均值显著性。
- **参照单一**：主端点只用了 `t17long`（seed 1、2M）一个参照；vs random 副端点方向一致，
  但未验证该稳定结论是否参照特异（B_s3 在对强参照与对 random 上已出现轻微非传递）。
- **200k 预算短**：run sd 对预算敏感；本结论只适用 200k 从零训练。
- **T 的 sd 读数非独立**：T 与 B 权重逐位相同，故 T 的三次 h2h 与 B 逐位相同（已补跑，
  amendment A2）；T 不提供独立的方差信息，只作为 no-op 的实证确认。
- 原始产物均在 `runs/stage0/variance/`（gitignored）；未改 `src/`、未 commit、未发布 ckpt。


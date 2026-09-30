# memoryless v5 优化：第一波执行报告（Wave-1）

> **角色**：第一波执行 agent。规格 = `docs/v5-optimization-plan.md`（revision-4）的「第一波」+
> `docs/v5-optimization-review.md` 的修正。严格按预注册执行，不即兴加臂、不改预注册门、
> 不事后挑 seed/端点。
>
> **口径**：revision-3（出空即撬底，`rules_id=2e36dbea44893696`）、obs v5（2 家 161 维）、
> 纯 MLP（`arch=shared`、`hidden=128`、单层 trunk）。统一评测口径见
> [`README.md §3`](./README.md)。跨 fit 绝对 Elo 不可比。
>
> **结论一句话**：**F1b 主端点（k=7 fresh 训练 seed × 9 fresh deal seed × 400 副，self 家族
> vs 固定 `l1_1M`）合并 Δ = +2.87 Elo，z-CI [−3.37,+9.11]、t-CI [−4.92,+10.66]，两口径均跨 0
> 且点估计 ≤ +5 → 按 §3.4 预注册门触发「止损」分支。self-play 线在第一波被判定为
> 「无 ≥+10 证据」，不进入 Stage 2。**F2/Block C 的 cap512 给出 +19.15（CI 排 0 但 < +20），
> 属方向性提示，不构成关闭 A2 的证据。**

---

## 0. 执行摘要与判定

| 端点 | 家族 | 设计 | ΔElo | z-CI (q=1.96) | t-CI (q=t{0.975,k−1}) | 判定 |
|---|---|---|---|---|---|---|
| **F1b** self_s2..s8 vs `l1_1M` | **confirmatory** | k=7 训练 seed × 9 deal seed × 400 副 | **+2.87** | [−3.37, +9.11] | [−4.92, +10.66] | **未判定（跨 0）**；点 ≤ +5 → **止损** |
| F1a self_sN vs rand_sN (N=1,2,3) | descriptive | 3 h2h × 9 deal seed × 400 副 | +23.99 / −6.13 / +17.30 | — | — | 训练 seed 间高度发散 |
| F1c self_s2/s8 vs `l1_2M` | descriptive | 2 h2h × 9 deal seed × 400 副 | +4.44 / −13.15 | — | — | 与 F1b 同向，无 winner's-curse 反转 |
| **F0** 共享 fit 绝对锚 | 单端点 | `build_ladder --anchor random` | μ(self_s2)−μ(l1_1M) = **+2.52**（MLE）/ +8.7（PL） | — | — | 未触发绝对退化门（阈值 −10） |
| Block C cap256 vs cap128 | descriptive (M2) | seed1 × 3 deal seed × 400 副 | **−33.56** | [−46.33, −20.79] | — | 落后 |
| Block C cap512 vs cap128 | descriptive (M2) | seed1 × 3 deal seed × 400 副 | **+19.15** | [+1.27, +37.03] | — | 点 < +20 → **非命中**，方向性提示 |

**预注册门裁定（§3.4）**：F1b 合并 Δ = +2.87 ≤ +5 → **无条件止损**：停 self 线扩展，
**不跑 Stage 2 的 Block B self 变体**（warm-start / selflong 1M / refresh50 / 静态池）。
Block C 全部臂 < +20 → 写明「**不构成关闭 A2 的证据**」，保留 hidden>128 待测。

---

## 1. 执行前核对（预检结果；任何对不上即停）

逐条验证规划第一波命令的 CLI 开关、基线资产与目录命名：

1. **CLI 开关全部存在且语义正确**（`src/seven523/train.py:parse_args`）：
   `--exp-name`、`--seed`、`--total-timesteps`、`--hidden-size`、`--opponent {random,self,pool}`、
   `--self-play-refresh`、`--pool-episode`、`--pool-member`、`--load-checkpoint`、
   `--checkpoint-interval`、`--cuda`、`--tensorboard`、`--run-dir` 均存在。
   `runs/t17_train/poll_snapshot.py` 存在；`tools/head_to_head.py`、`tools/build_ladder.py`、
   `tools/refit_mle.py` 的规划用参数（`--candidate/--anchor/--games-per-anchor/--cross/--no-traces/
   --games-out/--study/--seed/--device/--workers`；`--games/--bootstrap/--json/--out`；
   `--left/--right/--seeds/--pairs/--bootstrap/--device/--workers/--json`）全部存在。
   `src/seven523/league.py:76 parse_pool_member` 支持 `[WEIGHT@]ckpt:<path>` 与 `random`。
2. **基线 ckpt 全部存在**：`t17self__1/agent.pt`、`t17pool__{1,2,3}/agent.pt`、
   `t17long.../snapshots/checkpoint_step{65536? no → 1024000,1999872}.pt`、
   `t17pool__1/snapshots/checkpoint_step65536.pt` 均就位。
3. **目录命名不冲突**：执行前 `ls -d runs/t17w1*` → 不存在；`mkdir -p runs/t17w1` 后新建
   `runs/t17w1self__*`、`runs/t17w1cap*`，与既有 `runs/t17*` 无冲突。
4. **资源**：训练前 `free -h` = 4.4Gi free / 10Gi available；`nvidia-smi` = 40MiB used / 0%。
   训练最多 3 并发（`nice -n 5`），h2h 全部 `--device cpu --workers 4` 且严格串行。
5. **`python -m seven523.train` 可用**（`train.py:__main__` 守卫），CUDA 可用（`torch.cuda.is_available()=True`）。
6. **无遗留进程**：执行前后 `ps` 均无 `seven523.train`/`head_to_head`/`build_ladder` 残留。

**未改动** `src/`、`tests/`、`tools/`、`ADR/`、`plans.md`、`README.md`、`CONTEXT.md`、
`DESIGN.md`；未 commit/checkout/stash。本波未要求任何不存在的 CLI 开关，未即兴改码。

---

## 2. 臂表与命令

### 2.1 Block A — fresh self seeds 2–8（confirmatory，7 run × 500k）

```bash
for s in 2 3 4 5 6 7 8; do
  nice -n 5 .venv/bin/python -m seven523.train --exp-name t17w1self --seed "$s" \
    --total-timesteps 500000 --opponent self --self-play-refresh 10 \
    --cuda True --tensorboard False --run-dir runs
done
```

### 2.2 Block C — A2 容量 200k 粗筛（3 run，seed 1）

```bash
nice -n 5 .venv/bin/python -m seven523.train --exp-name t17w1cap128 --seed 1 \
  --total-timesteps 200000 --opponent random --hidden-size 128 \
  --cuda True --tensorboard False --run-dir runs
# cap256 / cap512 同，仅 --hidden-size 256 / 512
```

调度：Block A 分三批（2–4 / 5–7 / 8+cap128+cap256），cap512 在 cap128/256 结束后补位，
全程 ≤3 并发。

### 2.3 Stage 2（**未执行**）

Block B（`t17w1ws` warm-start self、`t17w1wsctrl` 匹配 random 对照、`t17w1selflong` 1M、
`t17w1refresh50`、`t17w1pool` 静态强成员池，5 run）为**条件块**，仅在 M1 点估计 ≥ +5 时才跑。
M1 = +2.87 ≤ +5 → 按预注册门停止，**未启动**。

---

## 3. 训练健康度

所有臂训到目标 step、无 NaN（`metrics.csv` 末行）。末行 `entropy / explained_variance /
episodic_return / episodic_length / sps`：

| run | step | entropy | EV | return | ep_len | sps | 备注 |
|---|---|---|---|---|---|---|---|
| `t17w1self__2__1790492897` | 499712 | 1.397 | 0.703 | +0.030 | 20.98 | 528 | |
| `t17w1self__3__1790492897` | 499712 | 1.331 | 0.624 | −0.002 | 20.84 | 530 | |
| `t17w1self__4__1790492897` | 499712 | 1.520 | 0.695 | −0.038 | 21.30 | 527 | |
| `t17w1self__5__1790493881` | 499712 | 1.550 | 0.603 | −0.008 | 20.69 | 521 | |
| `t17w1self__6__1790493881` | 499712 | 1.540 | 0.654 | +0.096 | 21.24 | 521 | |
| `t17w1self__7__1790493881` | 499712 | 1.423 | 0.685 | −0.188 | 20.57 | 523 | |
| `t17w1self__8__1790494847` | 499712 | 1.504 | 0.644 | −0.010 | 21.51 | 855 | 333–500k 独跑提速 |
| `t17w1cap128__1__1790494847` | 199680 | 1.796 | 0.666 | +0.419 | 24.55 | 1160 | 控制臂 |
| `t17w1cap256__1__1790494847` | 199680 | 1.862 | 0.708 | +0.376 | 24.57 | 1161 | |
| `t17w1cap512__1__1790495130` | 199680 | 1.894 | 0.646 | +0.477 | 23.82 | 1280 | |

**args.json diff（证明只差研究变量）**

- 7 个 self 臂 vs 参照 `t17self__1`：有效配置差 = `exp_name`（`t17self`→`t17w1self`）与
  `seed`（1→2..8）；参照 args.json 缺失的 `event_*`/`seq_*`/`seat_emb` 是本波之后新增的
  **inert 默认键**（`event_len=0`、`seq_len=0`、`event_*=false/keep/chrono`、`seat_emb=16`），
  对应默认 MLP 路径，不改变行为。**其余全部键逐字节相同**。
- `cap256`/`cap512` vs `cap128`：唯一 diff = `hidden_size` `(128,256)` / `(128,512)`。
- **熵门（§6.8）观察**：7 个 self 臂末熵 **1.33–1.55，全部 < 1.6**（参照 `t17self__1` 为
  1.543）；random 侧 cap 臂末熵 1.80–1.89（参照 `t17pool__1` 为 1.755）。self-play 在 500k 内
  系统性压低熵；F0 显示 `self_s2` 对 RandomBot **未退化**（阈值 −10），故熵门未被触发，但记为
  机制观察（见 §7 局限）。

---

## 4. 原始 h2h 表

全部 `--pairs 400 --bootstrap 4000 --device cpu --workers 4`，严格串行。每 h2h 打
`k` 个 deal seed；单训练 seed 内 deal seed 已由 `duel.py` 用 RMS 合并（`combined`）。
`Δ` 为 left − right。

### 4.1 F1b（confirmatory 主族，self_sN vs 固定 `l1_1M`，deal 10–18）

| 训练 seed | ΔElo | 95% CI | combined.se | boot_se | deal_seed_sd | winrate | deal p |
|---|---|---|---|---|---|---|---|
| s2 | +2.85 | [−4.41, +10.10] | 3.70 | 11.10 | 7.86 | 0.5041 | 0.447 |
| s3 | **+14.45** | [+7.41, +21.48] | 3.59 | 10.77 | 10.27 | 0.5208 | 5.7e−05 |
| s4 | +5.45 | [−1.62, +12.53] | 3.61 | 10.83 | 5.85 | 0.5078 | 0.086 |
| s5 | +2.03 | [−7.37, +11.44] | 4.80 | 11.00 | 14.40 | 0.5029 | 0.852 |
| s6 | −2.66 | [−9.90, +4.59] | 3.70 | 11.09 | 9.60 | 0.4962 | 0.503 |
| s7 | **+9.56** | [+2.25, +16.88] | 3.73 | 11.19 | 10.61 | 0.5138 | 0.018 |
| s8 | **−11.59** | [−18.91, −4.28] | 3.73 | 11.19 | 9.82 | 0.4833 | 0.0017 |

**k=7 合并**（`runs/t17w1/aggregate.py`）：

```
MERGE: +2.87   z-CI [-3.37, +9.11]   t-CI [-4.92, +10.66]
components: RMS combined.se 3.86 | arith combined.se 3.84 | training seed sd 8.43 |
            RMS deal sd 10.07 | arith deal sd 9.77
binding term: training seed sd (8.43); t_{0.975,6}=2.447; z-half 6.24; t-half 7.79
```

读法：**两口径 CI 均跨 0 = 未判定（≠ 效应为 0）**。训练 seed sd（8.43）> RMS boot（3.86），
binding term 是**训练 seed 方差**；这正是把 k 从 1 提到 7 后效应塌缩到 +2.87 的原因。

### 4.2 F1a（descriptive，self_sN vs rand_sN 同 seed，deal 10–18）

| 配对 | ΔElo | 95% CI | combined.se |
|---|---|---|---|
| self_s1 vs rand_s1 | **+23.99** | [+17.06, +30.91] | 3.53 |
| self_s2 vs rand_s2 | **−6.13** | [−13.54, +1.28] | 3.78 |
| self_s3 vs rand_s3 | **+17.30** | [+9.91, +24.69] | 3.77 |

三点散布（+23.99 / −6.13 / +17.30，均值 +11.72）→ 合并 sd ≈ **15.82**（含 self 与 random 两侧训练 seed 方差），跨 seed 高度不稳定。

### 4.3 F1c（descriptive，winner's-curse 检查，vs `l1_2M`，deal 10–18）

| 配对 | ΔElo | 95% CI |
|---|---|---|
| self_s2 vs l1_2M | +4.44 | [−2.90, +11.78] |
| self_s8 vs l1_2M | −13.15 | [−22.31, −3.99] |

与 F1b 同向（s2 略正、s8 明显负）→ **无 winner's-curse 反转**。

### 4.4 F0（共享 fit 绝对锚，vs 脚本 RandomBot）

`build_ladder.py --candidate l1_1M,self_s2,lvl3 --anchor random=random --games-per-anchor 400
--cross 400 --no-traces`（2400 局），再 `refit_mle.py --bootstrap 200`：

| 主体 | MLE μ | 95% CI | PL fit μ ± σ |
|---|---|---|---|
| random | 0.00 | — | 0.0 ± 0.0 |
| `l1_1M` | 188.06 | [171.92, 204.81] | 367.0 ± 35.8 |
| `self_s2` | 190.58 | [176.18, 210.00] | 375.7 ± 35.7 |
| `lvl3` | 142.43 | [128.06, 157.97] | 250.8 ± 36.2 |

**绝对锚门**：MLE `μ(self_s2) − μ(l1_1M)` = **+2.52**；PL fit = **+8.7**。两者均 > −10 →
**未触发**「self 只更会打 random-trained MLP」的绝对退化门（该门对 self_s2 通过）。
（注：此 fit 的绝对刻度与发布梯级 `185.34` 同属 revision-3 但由本轮 2400 局单独拟合，
不与发布 study 混比。）

### 4.5 Block C（M2 容量 200k 粗筛，seed 1，deal 10–12）

| 比较 | ΔElo | 95% CI | combined.se |
|---|---|---|---|
| cap256 vs cap128 | **−33.56** | [−46.33, −20.79] | 6.51 |
| cap512 vs cap128 | **+19.15** | [+1.27, +37.03] | 9.12 |

---

## 5. 按预注册门的判定

| 预注册门（§3.4） | 本轮读数 | 判定/动作 |
|---|---|---|
| F1b Δ ≥ +10 且 z/t CI 均排 0 → 确认为杠杆 | Δ = +2.87，CI 跨 0 | 不成立 |
| F1b Δ > +5 且 CI 跨 0 → 按 binding term 升级（加 seed/加 deal） | Δ = +2.87 **≤ +5** | **不进入升级** |
| F1b Δ ≤ +5（含负）→ 止损，不跑 Stage 2 self 变体 | Δ = +2.87 | **触发止损**（Stage 2 Block B 未运行） |
| F1b Δ ≥ +10 但 F0 显示 μ(self)−μ(l1_1M) ≤ −10 → 不升级 | F0 Δ = +2.52 | 不适用（未触发） |
| F1c 与 F1b 反向 → 降级方向性 | 同向 | 无降级 |
| Block C 任一臂 点 ≥+20 且 CI 排 0 → 容量命中，500k×3 确认 | cap512 +19.15（<+20）；cap256 负 | **非命中** |
| Block C 全部 <+20 或 CI 跨 0 → 写明「不构成关闭 A2 的证据」 | 全部 <+20 | 按此措辞：**200k/seed1 无 ≥+20 信号，保留 hidden>128 待测** |
| F2 任一臂 点 ≥+30 且 CI 排 0 → 仅方向性提示 | 最大 +19.15 | 不适用 |

**结论**：self-play 在 k=7 fresh 训练 seed 上**未确立**（未判定，点 ≤ +5 → 止损）；
容量 200k 粗筛**无 ≥+20 信号**（但 cap512 的 +19.15 CI 排 0 是非确认的方向性提示）。

---

## 6. 与 T17 self_play 基线的对比

| 读数 | T17（探索性） | 本波（confirmatory） |
|---|---|---|
| 端点 | `self_s1` vs `l1_1M`，deal 0–8（9 deal seed） | `self_s2..s8` vs `l1_1M`，deal 10–18（9 fresh deal seed） |
| 训练 seed 数 | **1**（stop-and-choose：3 deal 时 +10.3 跨 0，补 6 deal 后 +16.62） | **7 fresh** |
| ΔElo | **+16.62** [+9.30, +23.94]（deal-cluster） | **+2.87** [z −3.37,+9.11；t −4.92,+10.66] |

把训练 seed 从 1 扩到 7 后，同端点效应量由 +16.62 塌缩到 +2.87，且训练 seed sd（8.43）
成为 binding term。本波的 F1a 直接印证该方差：同 seed 自博弈对 random 的读数在 **+23.99 /
−6.13 / +17.30** 间摆动。→ **T17 的 +16.62 是单 seed、deal-seed 维度的探索性读数，不能作为
基线真值；k=7 复现显示 self-play 相对发布 lvl4 没有稳定 ≥+10 的优势。**

---

## 7. 局限

1. **未判定 ≠ 0**：F1b 的 CI 完全包含 0 两侧，本波只能说「无 ≥+10 证据 + 点 ≤+5 止损」，
   不能声称 self-play 效应为零。
2. **未测 `self_s1` 在 fresh deal 上的 `vs l1_1M`**：`self_s1` 不在预注册的 k=7（seed 2–8）
   集合内，未事后补测（避免 post-hoc 端点）。T17 的 +16.62 仅作历史注脚。
3. **Block C 仅 seed 1、200k、3 deal seed**（预注册即如此）：cap512 的 +19.15 是**描述性**
   方向提示（其 95% CI [+1.27,+37.03] 排 0 但点 < +20），不承载行动声明；200k 效应也可能
   在 500k 消失。
4. **F0 只有 `self_s2` 一个 self 代表**（预注册）；不能外推为全部 self 臂对 RandomBot 不退化。
5. **熵门机制观察**：7 个 self 臂末熵全部 < 1.6（random 侧 1.80–1.89）。本波因 self 线已止损，
   熵门未改变判定；但它是 self-play 的一个系统性特征，值得在后续机制研究中记录。
6. **F2/Block B 未跑**：M1 ≤ +5，按预注册门不进入 Stage 2，故 warm-start / 1M self /
   refresh50 / 静态强成员池（A1②③④⑤）本轮**无数据**。
7. **F0 绝对刻度**为本轮 2400 局单独 open-skill/PL + probit-MLE fit（`random=0`），与发布
   study 的 `l1_1M=185.34` 同属 revision-3 但**跨 fit 不可比**；仅用于候选间比较与 −10 门。

---

## 8. 复现命令

```bash
# 预检
mkdir -p runs/t17w1; free -h; nvidia-smi

# 训练（Block A 一例；Block C 一例）
nice -n 5 .venv/bin/python -m seven523.train --exp-name t17w1self --seed 2 \
  --total-timesteps 500000 --opponent self --self-play-refresh 10 \
  --cuda True --tensorboard False --run-dir runs
nice -n 5 .venv/bin/python -m seven523.train --exp-name t17w1cap512 --seed 1 \
  --total-timesteps 200000 --opponent random --hidden-size 512 \
  --cuda True --tensorboard False --run-dir runs

# F0 绝对锚（共享 fit）
.venv/bin/python tools/build_ladder.py \
  --candidate l1_1M=ckpt:runs/t17long__1__1790439615/snapshots/checkpoint_step1024000.pt \
  --candidate self_s2=ckpt:runs/t17w1self__2__1790492897/agent.pt \
  --candidate lvl3=ckpt:runs/t17pool__1__1790439615/snapshots/checkpoint_step65536.pt \
  --anchor random=random --games-per-anchor 400 --cross 400 --no-traces \
  --games-out runs/t17w1/f0_games.jsonl --study /tmp/t17w1_f0 --seed 0 \
  --device cpu --workers 4
.venv/bin/python tools/refit_mle.py --games runs/t17w1/f0_games.jsonl \
  --bootstrap 200 --json --out runs/t17w1/f0_absolute_table.json

# F1b（每训练 seed 一条；严格串行）
for n in 2 3 4 5 6 7 8; do
  .venv/bin/python tools/head_to_head.py \
    --left "self_s${n}=ckpt:runs/t17w1self__${n}__<ts>/agent.pt" \
    --right l1_1M=ckpt:runs/t17long__1__1790439615/snapshots/checkpoint_step1024000.pt \
    --seeds 10,11,12,13,14,15,16,17,18 --pairs 400 --bootstrap 4000 \
    --device cpu --workers 4 --json > "runs/t17w1/h2h_selfVS1M_s${n}.json"
done

# F1a / F1c / Block C
# F1a: --left self_sN --right rand_sN, seeds 10..18
# F1c: --left self_s2/s8 --right l1_2M=ckpt:...checkpoint_step1999872.pt, seeds 10..18
# Block C: --left cap256/cap512 --right cap128, seeds 10,11,12

# 合并（z 与 t 双口径）
.venv/bin/python runs/t17w1/aggregate.py --pair selfVS1M \
  --seeds s2,s3,s4,s5,s6,s7,s8 --family confirmatory
```

---

## 9. 产物路径

**训练 run**（`runs/`，gitignore；args.json + metrics.csv + agent.pt）
- self：`runs/t17w1self__{2,3,4}__1790492897/`、`runs/t17w1self__{5,6,7}__1790493881/`、
  `runs/t17w1self__8__1790494847/`
- cap：`runs/t17w1cap128__1__1790494847/`、`runs/t17w1cap256__1__1790494847/`、
  `runs/t17w1cap512__1__1790495130/`

**评测产物**（`runs/t17w1/`）
- `f0_games.jsonl`、`f0_absolute_table.json`、`f0_ladder.err`、`f0_mle.err`
- `h2h_selfVS1M_s{2..8}.json`（F1b）
- `h2h_selfVSrand_s{1,2,3}.json`（F1a）
- `h2h_selfVS2M_s{2,8}.json`（F1c）
- `h2h_cap{256,512}VScap128.json`（Block C / M2）
- `aggregate.py`（合并脚本，eval-only）、`logs/`

**预算核算**：本波实际执行 Block A（7×500k = 7.0 等效）+ Block C（3×200k = 1.2 等效）
= **8.2 等效 ≤ 15 上限**；Stage 2（6.0 等效）未执行。无预算超限。

**回滚**：删除 `runs/t17w1*/`、`runs/t17w1/` 即可；未触碰 `traces/study` 与 manifest，
未发 manifest。本报告为追加式，不改旧报告。

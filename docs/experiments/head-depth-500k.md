# 头部深度网格：（策略头 × 价值头）层数 {1,2,3}²，500k 从零 + pool-episode

> **状态：初步 screen 完成（2026-10-07，k=1 = seed 1）。9 臂 × 1 run、8 条 vs `hd_base` h2h。**
> **没有任何 cell 给出可取用的正结论。** `head12`（策略头 2 层、价值头 1 层）对 base
> **+17.40** [+4.1,+30.7] 是唯一 CI 排除 0 的正读数，按预注册（plan §5）触发「补 seed 2–3
> 到 k=3」的复核（本轮未执行、待排期）；`head22` **−20.15** [−32.9,−7.4] 负且 CI 排除 0。
> **k=1 的训练-seed sd（历史实测 8.4–17.2）不在 CI 内，以上均不是确认性结论。**
> 口径与预注册：[`../head-depth-plan.md`](../head-depth-plan.md)（多层头默认残差，operator 决策；
> 对手 = [ADR-0015](../adr/0015-continuation-pool-random-mix.md) 的 pool-episode + 10% random，
> 从零 500k）。产物 `runs/head-depth/`；复算 `runs/head-depth/recompute_head.py`。

## 0. 一句话结论

9 个头部深度配方在 500k 从零 + pool-episode 下**没有单调的深度/容量趋势**：8 个 cell 里
7 个对 base 的 bootstrap CI 跨 0（点估计 −6.4…+8.1）；唯一正号过线的是
`head12` +17.40（策略头 2 层残差），唯一负号过线的是 `head22` −20.15。k=1 下这更像
单 seed 抖动（历史上 k=3 高估回落 ~3 Elo、单 seed sd 8–17），因此**不改变现状默认
（`shared` 单层头）**；`head12` 的 k=3 复核按预注册规则待命。**pool-episode 配方从零可训**
（9/9 收敛、无 NaN、EV 0.52–0.71、vs-random 0.78–0.93），这是 ADR-0015 配方的首次从零使用。

## 1. 设计摘要（预注册 plan §1/§4）

- 网格 cell = (a, c)：a = 策略头 Linear 层数、c = 价值头 Linear 层数，a,c ∈ {1,2,3}；
  (1,1) = 现状 `shared`（单层头）。**多层头默认残差**：`layers−1` 个块
  `h ← h + ReLU(Linear(128,128))` + 输出层（actor `Linear(128,138)`、critic `Linear(128,1)`）。
- 参数 = `55,179 + 16,512·((a−1)+(c−1))`；8 个新 arch `head<a><c>`。
- 训练：seed 1、500k 从零；对手 = 6 个 t17 成员（各 `3@`）+ `2@random`
  （`--opponent pool --pool-episode True`，random 局 10%）；其余超参 = O2 主实验
  （Adam / linear / 8×128 / minibatch 4 / lr 2.5e-4）。
- 评估：每 cell 对 `hd_base`，3 deal seed（9600/9601/9602）× 400 副 × 换座，bootstrap 4000。

## 2. 执行记录

- **训练**：9 run × 500k（batch 1024，488 update），10:32:26–11:22:35（4 GPU 槽，rc=0×9）；
  9/9 metrics 488 行、无 NaN。
- **评估**：8 条 h2h 流式执行（cell 检查点一落地就跑，与最后一条训练重叠），
  11:03:30–11:23:24；8/8 JSON 有效（各 1,200 副 / 2,400 局）。
- **测试**：全套 `pytest` 868 passed / 14 skipped（含新增 4 个头部网格单测）；
  `head33` 冒烟 4096 步无 NaN。

## 3. 结果（k=1；每 cell 对 `hd_base`）

| cell | arch | 参数 | E[Elo] | bootstrap CI | winrate |
|---:|---|---:|---:|---|---:|
| (1,1) | `shared`（base） | 55,179 | 0.00 | — | 0.500 |
| (2,1) | `head21` | 71,691 | +8.12 | [−10.1, +26.4] | 0.512 |
| (3,1) | `head31` | 88,203 | +3.91 | [−7.9, +15.7] | 0.506 |
| (1,2) | `head12` | 71,691 | **+17.40** | **[+4.1, +30.7]** | 0.525 |
| (2,2) | `head22` | 88,203 | **−20.15** | **[−32.9, −7.4]** | 0.471 |
| (3,2) | `head32` | 104,715 | +6.95 | [−5.2, +19.1] | 0.510 |
| (1,3) | `head13` | 88,203 | −6.38 | [−24.0, +11.2] | 0.491 |
| (2,3) | `head23` | 104,715 | +1.01 | [−15.0, +17.0] | 0.501 |
| (3,3) | `head33` | 121,227 | +1.88 | [−16.4, +20.2] | 0.503 |

网格点估计（行 = 策略头 a，列 = 价值头 c）：

```
        c=1                  c=2                  c=3
a=1    0.00 (base)      +17.40 [+4.1,+30.7]   -6.38 [-24.0,+11.2]
a=2   +8.12 [-10.1,+26.4] -20.15 [-32.9,-7.4]  +1.01 [-15.0,+17.0]
a=3   +3.91 [ -7.9,+15.7] +6.95 [ -5.2,+19.1]  +1.88 [-16.4,+20.2]
```

**等参数反对角线**（容量相同，比较 actor-heavy vs critic-heavy）：

| a+c | 参数 | cells |
|---:|---:|---|
| 4 | 88,203 | `head13` −6.38 ／ `head22` −20.15 ／ `head31` +3.91 |
| 5 | 104,715 | `head23` +1.01 ／ `head32` +6.95 |

**机制读数**（final EV / 平均 entropy / max KL / 末次 vs-random eval）：

| cell | EV | entropy | max KL | evalW |
|---|---:|---:|---:|---:|
| base | 0.628 | 1.625 | 0.0052 | 0.880 |
| head12 | 0.695 | 1.709 | 0.0049 | 0.880 |
| head13 | 0.631 | 1.690 | 0.0054 | 0.860 |
| head21 | 0.707 | 1.673 | 0.0061 | 0.840 |
| head22 | 0.516 | 1.633 | 0.0075 | 0.780 |
| head23 | 0.658 | 1.624 | 0.0069 | 0.900 |
| head31 | 0.671 | 1.595 | 0.0080 | 0.920 |
| head32 | 0.600 | 1.663 | 0.0105 | 0.930 |
| head33 | 0.700 | 1.540 | 0.0097 | 0.850 |

## 4. 读法

1. **没有单调性**：策略头 1→2 层在 c=1/c=2 上符号相反（+17.4 / −20.2），价值头 1→3 层
   在 a=1 上有正有负（+17.4 / −6.4）。8 个读数的点估计全部落在 ±21 Elo 内，
   与单 seed sd 8–17 同量级——k=1 不能区分「效应」与「种子相位」。
2. **唯一正号过线的是 `head12`**（策略头加 1 层残差，+17.40，CI 排除 0）。它的机制读数也最健康
   （EV 0.695、entropy 1.709）；但历史两次同型读数（T23 k=3 +11.48 → k=7 +8.20；A5 k=3
   +15.41 → k=7 +9.27）都回落，**不得当 ≥+10 结论**。
3. **`head22` 的负读数与 EV 0.516 一致**（最差 EV、最低 evalW 0.78），但它不与 `head33`/`head23`
   的 null 连成趋势，仍可能只是该 seed 的训练轨迹。
4. **等参数对抗线上** actor-heavy（`head31` +3.91）优于 critic-heavy（`head22` −20.15），
   与「策略头比价值头更值得加容量」方向一致，但 n=1，只作观察。
5. **容量不解释正效应**：参数量最大的 `head33`（121,227）≈ 0；此前 `wide`/cap256/cap512
   null/负先验同向。
6. **配方侧结论（可信度高于架构侧）**：ADR-0015 的 pool-episode + 10% random 从零 500k
   可训——9/9 无 NaN、EV 0.52–0.71、entropy 1.54–1.71、vs-random 0.78–0.93；首次从零使用
   没有出现池过强导致的崩坏。

## 5. 限制

- **k=1**：无训练-seed 误差；排序可能翻转；本报告所有点估计只作 screen。
- **深度与残差绑定**：多层头一律残差（operator 默认设计），所以 cell vs base 读的是
  「残差深头 vs plain 单层头」；plain 深头不在网格内（残差单因子在 trunk 上由 A1 覆盖）。
- **容量**：每层 +16,512 参数（+30%/层），`head33` +120%；反对角线只控制部分解释。
- **配方不同**：`hd_base` 是 pool-episode 从零，与 O2 的 random-对手 base（`o2_base`）
  不可直接比较；本文只在网格内有效。

## 6. 产物与复现

```bash
# 训练（9 run；4 GPU 槽自动排队）
bash runs/head-depth/run_grid.sh
# 评估（8 条；流式版本可在训练收尾时重叠执行）
bash runs/head-depth/run_grid_h2h.sh
bash runs/head-depth/run_grid_h2h_streams.sh
# 复算网格
.venv/bin/python runs/head-depth/recompute_head.py
```

- run 产物：`runs/head-depth/hd_<cell>__1__<ts>/{agent.pt, metrics.csv, args.json}`；
  h2h：`runs/head-depth/h2h_hd_head<a><c>-base_s1.json` + `games/*.jsonl`。

## 7. 后续（按预注册 plan §5，待 operator 决定）

1. `head12` 补 seed 2–3（cell + base 各 2 run）→ k=3 复读；k=3 仍「点 ≥ +10 且 z-CI 排除 0」
   → 补 seed 4–7 做 k=7 确认（条件读数、预期回落）。
2. 若无 cell 进入 k=3 且过门：头部深度方向记为初步无信号，是否关轴由 operator 裁定
   （k=1 不足以关轴）。
3. 与 reward/penalty 函数的讨论接续（operator 2026-10-07 指定下一议题）。

## 8. Amendment H1：新默认配方续训 head12/22/32（2026-10-07，k=1）

**设计**（plan §9）：三臂从各自 500k 起点按 A5 T23 流程 + [ADR-0016](../adr/0016-default-training-recipe-arcsin-lr-floor.md)
默认（arcsin α=0.5、win_jump=1.0、lr_floor=1e-5）续训 +1M；对手 = [ADR-0015](../adr/0015-continuation-pool-random-mix.md)
pool-episode（6×t17 `3@` + `2@random`）。3 run 12:01–12:32 完成、无 NaN；12 条 h2h 12:35 完成。

**结果**（k=1；9600–9602 × 400 副 × 换座，bootstrap 4000）：

| 比较 | 点估计 | CI | winrate |
|---|---:|---|---:|
| `head12c − head22c` | +3.48 | [−8.50,+15.46] | 0.505 |
| `head12c − head32c` | +9.56 | [−1.76,+20.88] | 0.514 |
| `head22c − head32c` | −10.14 | [−22.94,+2.65] | 0.485 |
| `head12c − head12`（起点） | **−4.34** | [−14.22,+5.54] | 0.494 |
| `head22c − head22`（起点） | **+13.05** | [−2.44,+28.54] | 0.519 |
| `head32c − head32`（起点） | **+10.28** | [−0.06,+20.63] | 0.515 |
| `head12c − o2c_base` | −0.87 | [−12.30,+10.56] | 0.499 |
| `head22c − o2c_base` | −6.23 | [−18.54,+6.08] | 0.491 |
| `head32c − o2c_base` | +4.78 | [−8.11,+17.67] | 0.507 |
| `head12c − w2m_ctl` | +11.30 | [−0.65,+23.24] | 0.516 |
| `head22c − w2m_ctl` | −9.41 | [−21.48,+2.66] | 0.486 |
| `head32c − w2m_ctl` | +4.78 | [−6.32,+15.88] | 0.507 |

**机制**（final EV / mean entropy / maxKL / 末次 vs-random eval）：

| arm | EV | entropy | max KL | evalW |
|---|---:|---:|---:|---:|
| head12c | 0.653 | 1.538 | 0.0156 | 0.910 |
| head22c | 0.570 | 1.316 | 0.0177 | 0.930 |
| head32c | 0.648 | 1.409 | 0.0192 | 0.940 |

**读法**（k=1，CI 全跨 0，不宣布 H1/H0）：① `head12c` 对自身 500k 起点是唯一的负增益
（−4.34；EV 0.695→0.653、entropy 1.709→1.538）；② 三臂都还压不过旧配方续训的 `o2c_base`
（−6.2…+4.8）；③ operator 随后指出头部残差设计缺口（无归一化、actor 输出 init 太小）→
head 设计 v2 见 plan §10（`head<a><c>ln` + `--actor-out-std`）。

```bash
bash runs/head-depth/run_cont_newrecipe.sh
bash runs/head-depth/run_cont_h2h.sh
.venv/bin/python runs/head-depth/recompute_cont.py
```

## 9. Amendment H2：head32ln（v2 设计）从零 500k（2026-10-07，k=1）

**设计**（plan §11）：单臂 `hd_head32ln`（`head32ln` = 3 层 actor / 2 层 critic，残差块内
**pre-norm** LayerNorm + `--actor-out-std 0.1`），新配方（ADR-0016 arcsin 0.5/λ=1/floor 1e-5）
+ ADR-0015 pool-episode，500k 从零。`hd_head32ln` 12:52–13:01 rc=0、488/488、无 NaN。

**结果**（k=1，9600–9602 × 400 × 换座）：

| 比较 | 点估计 | CI | winrate |
|---|---:|---|---:|
| `head32ln − hd_head32`（旧网格，同 500k 预算） | −7.98 | [−25.35,+9.40] | 0.489 |
| `head32ln − head32c`（旧设计，同配方，1M 续训 = 1.5M） | **−17.25** | **[−29.08,−5.41]** | 0.475 |

机制：`head32ln` EV 0.663 / entropy 1.637 / maxKL 0.0096 / evalW 0.910
（旧网格 `head32` 0.600 / 1.663 / 0.0105 / 0.930；`head32c` 0.648 / 1.409 / 0.0192 / 0.940）。

**读法**：500k 的 v2 臂对同预算旧网格 −7.98（CI 跨 0，无可用的改进证据，点估计偏负）；
对 1.5M 旧设计臂的 −17.25 主要是续训预算差（+1M），不是干净的设计对照 →
operator 指定直接做同配方 1M 续训（H2b，plan §12），与 `head32c` 形成
同配方/同预算/只差设计的对照。

```bash
bash runs/head-depth/run_head32ln.sh
bash runs/head-depth/run_head32ln_h2h.sh
bash runs/head-depth/run_head32ln_cont.sh
bash runs/head-depth/run_head32ln_cont_h2h.sh
```

## 10. Amendment H2b：head32ln 同配方 1M 续训（2026-10-07，k=1）

**设计**（plan §12）：`hd_head32lnc` 从 `hd_head32ln`（500k）按与 H1 完全相同的续训
协议再训 +1M（A5 T23 + ADR-0016 + ADR-0015 pool-episode），与 `head32c` 形成
**同配方/同预算（1.5M）、只差头部设计**的对照。13:05–13:22 rc=0、488/488、无 NaN。

**结果**（k=1，9600–9602 × 400 × 换座）：

| 比较 | 点估计 | CI | winrate |
|---|---:|---|---:|
| `head32lnc − head32c`（同配方同预算，只差设计） | **−7.82** | [−19.21,+3.57] | 0.489 |
| `head32lnc − head32ln`（自身 500k 起点） | **+19.42** | **[+7.46,+31.39]** | 0.528 |
| `head32lnc − o2c_base` | −5.36 | [−17.37,+6.66] | 0.492 |
| `head32lnc − w2m_ctl` | +0.43 | [−11.03,+11.90] | 0.501 |
| `head32lnc − head12c` | −4.20 | [−15.45,+7.05] | 0.494 |
| `head32lnc − head22c` | −5.94 | [−21.22,+9.34] | 0.491 |

机制：`head32lnc` EV 0.631 / entropy **1.212** / maxKL 0.0189 / evalW 0.820
（`head32c` 0.648 / 1.409 / 0.0192 / 0.940；`head12c` 0.653/1.538/0.0156/0.910；
`head22c` 0.570/1.316/0.0177/0.930）。

**结论**：v2 设计（pre-norm LN 残差块 + `actor_out_std 0.1`）在唯一一个预先指定的
干净对照上没有优于 v1：`head32lnc − head32c` = −7.82（CI 跨 0、点为负）；
其余全是对照/锚，`head12c/22c` 也略在其前。续训杠杆在该臂上再次复现（+19.42，
CI 排除 0）。机制上 v2 的 entropy 最低（1.212）、evalW 最低（0.820）、EV 略低
（0.631 vs 0.648），更像更早的池特化而非强度。k=1（不含训练-seed sd），
不宣布 H1/H0，也不自动关轴——是否关闭由 operator 裁定。

## 11. Amendment H2c：head32ln 再 +2M（总 3.5M）（2026-10-07，k=1）

**设计**（plan §13）：`hd_head32ln2m` 从 `hd_head32lnc`（1.5M）按同协议再训 +2M
（总 3.5M）。13:27–13:59 rc=0、976/976 update、无 NaN。

**结果**（k=1，9600–9602 × 400 × 换座）：

| 比较 | 点估计 | CI | winrate |
|---|---:|---|---:|
| `head32ln2m − head32lnc`（自身 1.5M） | −0.72 | [−11.77,+10.32] | 0.499 |
| `head32ln2m − head32c`（v1 1.5M） | +1.74 | [−9.22,+12.70] | 0.502 |
| `head32ln2m − o2c_base` | −0.58 | [−11.84,+10.68] | 0.499 |
| `head32ln2m − w2m_ctl` | +9.99 | [−0.94,+20.93] | 0.514 |
| `head32ln2m − head12c` | −5.22 | [−16.73,+6.30] | 0.492 |
| `head32ln2m − head22c` | +1.59 | [−14.09,+17.27] | 0.502 |

机制：`head32ln2m` EV 0.652 / entropy **1.163** / maxKL 0.0178 / evalW 0.910
（1.5M 时 0.631/1.212/0.0189/0.820）。

**结论**：额外 2M 进入平台——对自身 1.5M 是 −0.72（CI 跨 0，无增益）；对 v1 设计
+1.74（CI 跨 0，设计差被预算抹平）；对 `o2c_base` −0.58（平）；对 `w2m_ctl` +9.99
（CI 含 0，点接近 +10）。熵继续下降到 1.163、evalW 回升至 0.910。k=1 仍不宣布 H1/H0；
operator 指定继续 +5M（H2d，plan §14）。

## 12. Amendment H2d：head32ln 再 +5M（总 8.5M）（2026-10-07，k=1）

**设计**（plan §14）：`hd_head32ln5m` 从 `hd_head32ln2m`（3.5M）按同协议再训 +5M
（总 8.5M）。14:04–15:23 rc=0、2441/2441 update、无 NaN。

**结果**（k=1，9600–9602 × 400 × 换座）：

| 比较 | 点估计 | CI | winrate |
|---|---:|---|---:|
| `head32ln5m − head32ln2m`（自身 3.5M） | +16.69 | [−1.58,+34.95] | 0.524 |
| `head32ln5m − head32c`（v1 1.5M） | +8.26 | [−2.89,+19.40] | 0.512 |
| `head32ln5m − o2c_base` | +3.18 | [−17.97,+24.34] | 0.505 |
| `head32ln5m − w2m_ctl` | +11.16 | [−5.06,+27.38] | 0.516 |
| `head32ln5m − head12c` | +12.18 | [−3.60,+27.96] | 0.517 |
| `head32ln5m − head22c` | +21.90 | **[+9.67,+34.13]** | 0.531 |

机制：`head32ln5m` EV 0.651 / entropy **1.140** / maxKL 0.0422 / evalW 0.860
（3.5M 时 0.652/1.163/0.0178/0.910）。

**结论**：额外 5M 后点估计对自身 3.5M 是 +16.69（CI 含 0，k=1），对全部 raw 比较对象
为正（`head22c` 的 CI 排除 0）；熵继续降到 1.140。无 H1。该模型（
`runs/head-depth/cont/hd_head32ln5m__1__1791396292/agent.pt`）随后被加入
`7g523-web` 的默认 manifest（raw + 搜索档，plan §15）。

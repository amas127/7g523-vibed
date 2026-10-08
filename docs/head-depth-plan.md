# 头部深度网格：（策略头 × 价值头）层数 {1,2,3}²，500k 从零 + pool-episode 配方

> **状态：预注册（2026-10-07，operator 请求）。初步 screen：k=1（seed 1），
> 9 臂 × 1 run；明确是探索性读数，不构成可用结论。后续复核规则见 §5。**
> 口径：revision-3（`rules_id=2e36dbea44893696`）、obs v5（2 家 161 维）、
> 500k 从零、`arch=head<a><c>`（`a`=策略头 Linear 层数，`c`=价值头 Linear 层数；
> 多层头默认残差）。
> 对手 = ADR-0015 的 pool-episode（6 个 t17 成员 `3@` + `2@random`，随机局 10%）。

## 0. 背景与假设

- operator 请求：把策略头/价值头加深到 3 层，并展开 (1,2,3)² 全网格比较。
- 相关先验（均为负或 null，本 screen 是它们覆盖不到的格子）：
  - **trunk 加深**：O2 主端点 `deep_ln−base` k=7 **−13.95** [−25.84,−2.06]；
    `deep−base` −9.52 [−18.17,−0.86]（`experiments/depth-normalization-500k.md` §3）。
  - **critic 头加深**：D-C 的 `deep_head`（冻结 trunk + `Linear(128,128)+relu+Linear(128,1)`）
    leaf EV 0.5211 vs base 0.5216（`experiments/stage0-critic-ceiling.md:43/113-114/128/167`）——
    只测了 leaf EV 拟合，不是全训练。
  - **容量**：`wide−base`（+17k 参数）k=3 +0.97 [−16.65,+18.59] null；cap256 −33.6 / cap512 负。
- 头部深度在全训练里从未单因子测过。**H1**：3 层头相对 1 层有 ≥ +10 Elo 的增益；
  **H0**：没有。参数量与容量解释用网格本身的对角线控制（§4）。

## 1. 臂（9 臂；k=1 = seed 1，共 9 run）

九宫格 cell (a, c)：`a` = 策略头 Linear 层数，`c` = 价值头 Linear 层数；
(1,1) 即现状 `shared`（单层头）。**头部结构（残差为默认设计，operator 2026-10-07 决定）**：
`a` 层 = (a−1) 个残差块 `h ← h + ReLU(Linear(128,128))` + 输出层 `Linear(128,138)`；
`c` 层同理，输出 `Linear(128,1)`。1 层头没有可 skip 的隐藏层，保持现状 plain Linear。
残差不改变参数量（同 trunk 的 `deep_res`/`deep` 关系），参数 = `55,179 + 16,512·((a−1)+(c−1))`。

| cell | `--arch` | 策略头 | 价值头 | 总参数 | run 名 |
|---:|---|---:|---:|---:|---|
| (1,1) | `shared` | 1 | 1 | 55,179 | `hd_base` |
| (2,1) | `head21` | 2 | 1 | 71,691 | `hd_head21` |
| (3,1) | `head31` | 3 | 1 | 88,203 | `hd_head31` |
| (1,2) | `head12` | 1 | 2 | 71,691 | `hd_head12` |
| (2,2) | `head22` | 2 | 2 | 88,203 | `hd_head22` |
| (3,2) | `head32` | 3 | 2 | 104,715 | `hd_head32` |
| (1,3) | `head13` | 1 | 3 | 88,203 | `hd_head13` |
| (2,3) | `head23` | 2 | 3 | 104,715 | `hd_head23` |
| (3,3) | `head33` | 3 | 3 | 121,227 | `hd_head33` |

- 等参数反对角线（给 actor-vs-critic 归因）：`a+c=3` → 55,179；`a+c=4` →
  `head31`/`head22`/`head12` 均 **88,203**；`a+c=5` → `head32`/`head23` 均 **104,715**。
- 头部隐藏层用 `layer_init(std=√2)`（与 trunk 隐藏层一致）；输出层保持现状
  （actor `std=0.01`、critic `std=1.0`；`vf_nll` 的 logvar 行零初始化）。
- 训练 seed：**仅 seed 1**（初步 screen）。

## 2. 实现

- `src/seven523/networks.py`：`_ARCHITECTURES` 增 8 个 `head<a><c>`；新增
  `ResidualHead`（`layers−1` 个残差块 `h ← h + act(Linear(h))` + 输出层；
  `layer_init(std=√2)` 隐藏层、输出层保持 actor `std=0.01` / critic `std=1.0`）；
  `Agent.__init__` 在 trunk 之后按 (a, c) 构建两个头。**残差是默认头部设计**：
  多层头一律 `ResidualHead`，1 层头保持 plain `Linear`——`shared` 与所有既有 arch 的
  模块类型、初始化调用顺序逐位不变。
- `src/seven523/train.py:237` 的 `--arch` choices/help 同步。
- 跨 arch 热启动保持 fail-loud（白名单仍只有 shared↔towers）；同 arch + 同 nvec 整表加载；
  `head<a><c>` 之间、以及 shared→head 均 `raise WarmStartLayoutError`——本 screen 全部从零。
- 新增单测（`tests/test_networks.py`）：8 个新 arch 的参数量、头模块序列（Linear→ReLU
  交替、末层输出维度）、前向 shape/无 NaN、ckpt round-trip 记 arch、`vf_nll+head33`
  的 logvar 零初始化、跨 arch warm-start fail-loud、`shared` 默认路径逐位回归。
- 不改 trunk/actor/critic 的既有默认路径；不动 ckpt 布局身份字段。

## 3. 训练配方（500k 从零；唯一新因子 = 头部深度）

超参沿用 O2 主实验（500k 从零基线），**只把对手换成 ADR-0015 的 pool-episode + 10% random**；
T23 的 AdamW/cosine/batch2048 属续训协议、未在从零 500k 验证过，不混入本轮。

```bash
cd /home/amas/.local/src/7g523
POOL=(
  --pool-member 3@ckpt:runs/t17pool__1__1790439615/snapshots/checkpoint_step262144.pt
  --pool-member 3@ckpt:runs/t17pool__1__1790439615/snapshots/checkpoint_step499712.pt
  --pool-member 3@ckpt:runs/t17long__1__1790439615/snapshots/checkpoint_step512000.pt
  --pool-member 3@ckpt:runs/t17long__1__1790439615/snapshots/checkpoint_step1024000.pt
  --pool-member 3@ckpt:runs/t17long__1__1790439615/snapshots/checkpoint_step1536000.pt
  --pool-member 3@ckpt:runs/t17long__1__1790439615/snapshots/checkpoint_step1999872.pt
  --pool-member 2@random
)
.venv/bin/python -m seven523.train \
  --exp-name hd_<arch> --seed 1 --total-timesteps 500000 \
  --arch <arch> --hidden-size 128 --activation relu \
  --num-envs 8 --num-steps 128 --num-minibatches 4 --update-epochs 4 \
  --learning-rate 2.5e-4 --anneal-lr --lr-schedule linear --optimizer adam \
  --gamma 0.99 --gae-lambda 0.95 --norm-adv --clip-coef 0.1 --clip-vloss \
  --ent-coef 0.01 --vf-coef 0.5 --max-grad-norm 0.5 \
  --opponent pool --pool-episode True "${POOL[@]}" \
  --reward-shaping terminal --num-players 2 --torch-deterministic \
  --tensorboard true --snapshot-interval 0 --eval-interval 50 \
  --cuda True --run-dir runs/head-depth
```

脚本 `runs/head-depth/run_grid.sh`（9 run，4 GPU 槽自动排队）。

## 4. 端点与统计（初步）

- **评估**：每个 cell 对 `hd_base` 的 h2h；3 deal seed（9600/9601/9602）× 400 副 × 换座，
  bootstrap 4000（`tools/head_to_head.py`）。
- **k=1 读数**：`combine_duel_seeds` 要求 k≥2，单 run 直接取 h2h JSON 的 `combined`
  点估计与 bootstrap CI；训练-seed sd（历史实测 8.43–17.2）不在 CI 里，报告必须写明。
- **判定**：**本轮不宣布 H1/H0**。报告网格点估计 + CI；主问句读法是
  「3 层头相对 1 层的点估计与区间」，不是确认性结论。
- **等参数归因**：在 `a+c` 相同的边上比较 actor-heavy vs critic-heavy
  （如 `head31` vs `head22` vs `head12`，均 88,203 参数）。
- **机制读数**：EV / entropy / kl / vs-random eval（`metrics.csv`），对照 O2 批的 0.64–0.71 / 1.3–1.5。

## 5. 后续复核规则（预先写死；本轮不执行）

1. 任一 cell 对 `hd_base`：**点 ≥ +10 且 bootstrap CI 排除 0** → 下一轮补 seed 2–3
   （该 cell + base 各 2 run），k=3 复读；k=3 仍满足「点 ≥ +10 且 z-CI 排除 0」→
   再补 seed 4–7（+8 run）做 k=7 确认。
2. 条件触发会使点估计偏高（T23：k=3 +11.48 → k=7 +8.20；A5：+15.41 → +9.27），
   报告必须标注「条件读数」。
3. 没有任何 cell 过第 1 条 → 头部深度方向记录为初步无信号，是否关闭由 operator 决定
   （k=1 不足以关轴）。

## 6. 风险与混淆

- **k=1**：无训练-seed 误差；单 seed sd 8–17 → 点差 ≲10 不可读；任何排序都可能翻转。
- **容量**：每加一层 +16,512 参数（+30%/层）；反对角线只控制部分容量解释，
  `head33` +120% 参数与「更深」不可完全分离（依赖既有 wide/cap 负先验）。
- **残差与深度绑定**：本网格多层头一律残差（operator 默认设计），所以 cell vs base
  读的是「残差深头 vs plain 单层头」；plain 深头不在本网格内。残差单因子在 trunk 上
  已由 A1 测过（`deep_res−deep` +6.82，k=3，未升级）。
- **配方从零**：pool-episode 此前只用于续训（warm start 1M）；从零对强池是否可学未知
  （混 10% random 提供基础信号）。若从零学不动（EV/entropy 崩、vs-random 饱和差）是
  配方问题，不能读成头部深度结论。
- **先验**：D-C 的 critic 深头 leaf EV null、O2 trunk 深度负、cap256/512 负。

## 7. 成本

- 训练：9 run × 500k（pool 对手、batch 1024）≈ 60–75 min（4 GPU 槽上限）。
- 评估：8 条 h2h ≈ 5–8 min CPU。
- 复核（若触发）：每 cell 每加 2 seed ≈ +4 run（k=3）或 +8 run（k=7，含 base）。

## 8. 结果回填（2026-10-07）

初步 screen 完成（k=1）：9/9 run rc=0、488/488 update、无 NaN；8/8 h2h 有效。
**无单调趋势**；唯一正号过线 `head12` +17.40 [+4.1,+30.7]、唯一负号过线 `head22` −20.15
[−32.9,−7.4]，其余 CI 跨 0。k=1 不构成结论，`head12` 按 §5 待补 seed 2–3。
读数、机制与限制见 [`experiments/head-depth-500k.md`](experiments/head-depth-500k.md)。

## 9. Amendment H1（2026-10-07，operator 选定）：head12/22/32 新配方续训 +1M

**动机**：k=1 网格里 operator 选定 **head12 / head22 / head32** 三臂（点估计上的正/负/中间
代表），用 [ADR-0016](adr/0016-default-training-recipe-arcsin-lr-floor.md) 的新默认配方
（arcsin α=0.5 + 50 分跳变 λ=1.0 + LR 下界 1e-5）从各自 500k 起点续训 +1M，再 h2h 看排名。
**探索性 screen，不是确认性实验。**

**臂（3 run；各 seed 1）**

| arm | 起点（500k） | `--arch` | 续训 |
|---|---|---|---|
| hd_head12c | `runs/head-depth/hd_head12__1__*` | `head12` | +1M |
| hd_head22c | `runs/head-depth/hd_head22__1__*` | `head22` | +1M |
| hd_head32c | `runs/head-depth/hd_head32__1__*` | `head32` | +1M |

**续训协议** = A5 T23 流程 + ADR-0016 默认（开跑前写死）：
`--load-checkpoint` 同 arch 整表加载、`--total-timesteps 1000000`、
`--optimizer adamw --weight-decay 0.01 --lr-schedule cosine` 峰值 2.5e-4、`--lr-floor 1e-5`、
`--num-envs 16 --num-steps 128 --num-minibatches 8`（batch 2048）、
`--opponent pool --pool-episode True` + ADR-0015 的 6 个 t17 成员 `3@` + `2@random`、
`--reward-shaping arcsin --arcsin-mix 0.5 --win-jump 1.0`（显式写死）、
`--snapshot-interval 0`（只用 final）。脚本 `runs/head-depth/run_cont_newrecipe.sh`。

**端点（k=1，descriptive）**：标准 bank 9600–9602 × 400 副 × 换座、bootstrap 4000。

1. 三臂两两 h2h（排名）；
2. 各 − 自身 500k 起点（续训增益，含奖励变更）；
3. 各 − `o2c_base`（旧配方续训 base，seed 1 配对）与各 − `w2m_ctl`（绝对锚）。

**判定**：k=1 不宣布任何 H1/H0；点 ≥ +10 且 CI 排除 0 的臂标记为待补 seed 2–3
（网格只有 seed 1，需先补 500k 起点，成本另计）。不做轴级关闭/开启。

**已知混淆**：① 续训增益包含 reward/floor 变更（起点是旧 `terminal`），不能与 A5 的 +29 直接比；
② head arch × 新 reward 双因子同时变；③ `o2c_base` 是旧配方（terminal/无 floor），
跨配方比较只作 context。

**成本**：3 run × 1M ≈ 35–45 min（4 GPU 槽上限）；12 条 h2h ≈ 10–12 min CPU。

**H1 结果（2026-10-07 回填）**：3 run 12:01–12:32 rc=0、无 NaN；12 条 h2h 完成。k=1 读数
（CI 全跨 0，不宣布 H1/H0）：pairwise `head12c>head32c` +9.56、`head22c<head32c` −10.14；
对自身 500k 起点 −4.34 / +13.05 / +10.28；对 `o2c_base` −0.87 / −6.23 / +4.78；
对 `w2m_ctl` +11.30 / −9.41 / +4.78。机制与读法见
[`experiments/head-depth-500k.md`](experiments/head-depth-500k.md) §8。

## 10. 头部设计 v2（2026-10-07，operator 指出缺口的修订；待实现/待跑）

**缺口（operator 评审）**：

1. `ResidualHead` 残差块 `h ← h + ReLU(Linear(h))` 无任何归一化；残差流方差随块数增长，
   深头条件数差。
2. actor 输出 `layer_init(std=0.01)`：整条 actor 头（含隐藏层）的梯度被输出权重尺度整体
   缩放，深 actor 头相对其它参数信号弱、学得慢。

**v2 设计（2026-10-07 已实现，待跑）**：

- 新 arch 族 `head<a><c>ln`（8 个 cell，已实现）：残差块 = `LayerNorm → Linear → activation`
  （**pre-norm**：归一化块输入以稳定残差流方差），skip 仍为 `h ← h + block(h)`；旧 `head<a><c>`
  （无 LN）保留为历史。
- 新增 `--actor-out-std`（已实现，默认 0.01 = 历史）：actor 输出层 init 的 std；多层 actor 头
  的重跑用 **0.1**。理由：LN 后 `σ_logits ≈ std·√(hidden/2) ≈ 0.8`，初始策略仍近均匀
  （entropy 降 ~6%），而头部隐藏层梯度放大 10×。
- 参数变化：每个 LN 块 +2×hidden（=256），如 `head12ln` = 71,947、`head32ln` = 105,483。
- 重跑矩阵（待定）：三臂 {head12, head22, head32} × {LN, 无 LN} × {0.1, 0.01}，
  从零 500k（pool-episode 配方）→ k=1 screen；或只跑 {LN, 0.1} 三臂。
- 实现与测试：`ResidualHead(ln=...)`（pre-norm 块）、`Agent(actor_out_std=...)`、
  `--actor-out-std`；`tests/test_networks.py` 钉住 8 个 LN arch 的参数量/结构/前向/ckpt
  round-trip/warm-start fail-loud 与 actor-out-std 的初始化尺度。
- 新 arch 无法 warm-start 旧 head ckpt（LN 加参数，fail-loud）；旧 ckpt 不回填。

## 11. Amendment H2（2026-10-07，operator 指定）：head32ln 单臂从零 500k

**动机**：operator 要求用 `head32ln + --actor-out-std 0.1` 从零 500k 跑新配方
（[ADR-0016](adr/0016-default-training-recipe-arcsin-lr-floor.md)），然后 h2h 与
同配方旧设计模型对比。**只训这一个臂**（operator 明确不加同配方对照组）。

**臂（1 run，seed 1，500k 从零）**

| arm | `--arch` | `--actor-out-std` | 起点 |
|---|---|---:|---|
| hd_head32ln | `head32ln` | **0.1** | 从零（新配方） |

- 配方：当前默认 `--reward-shaping arcsin --arcsin-mix 0.5 --win-jump 1.0 --lr-floor 1e-5`、
  对手 = ADR-0015 pool-episode（6 个 t17 成员 `3@` + `2@random`）、
  O2 从零超参（Adam/linear/8×128/minibatch 4）。脚本 `runs/head-depth/run_head32ln.sh`。

**端点（k=1，descriptive）**：9600–9602 × 400 副 × 换座，bootstrap 4000。

1. `head32ln − hd_head32`（旧网格臂：同对手配方、旧 head 设计 + 旧 reward/floor；
   混合了设计与 reward/floor 两个因素，只作 context）。
2. `head32ln − head32c`（旧设计 + **同新配方**的 1M 续训臂；预算不同（1.5M vs 0.5M）
   但 reward/floor 严格同配方，作为「同配方旧模型」的直接对照）。

**判定**：k=1 不宣布 H1/H0；点 ≥ +10 且 CI 排除 0 标记为待补 seed。

**成本**：1 run × 500k ≈ 20 min；2 条 h2h ≈ 5 min CPU。

**H2 结果（2026-10-07 回填）**：`hd_head32ln` 12:52–13:01 rc=0、488/488 update、无 NaN；
两条 h2h（k=1，9600–9602 × 400 × 换座）：

| 比较 | 点估计 | CI | winrate |
|---|---:|---|---:|
| `head32ln − hd_head32`（旧网格，同预算） | −7.98 | [−25.35,+9.40] | 0.489 |
| `head32ln − head32c`（旧设计，同配方，1.5M） | **−17.25** | **[−29.08,−5.41]** | 0.475 |

机制：`head32ln` EV 0.663 / entropy 1.637 / maxKL 0.0096 / evalW 0.910（旧网格 `head32`
0.600 / 1.663 / 0.0105 / 0.930；`head32c` 0.648 / 1.409 / 0.0192 / 0.940）。
读法：500k 的 LN 臂对同预算旧网格是 −7.98（CI 跨 0，无改进证据）；对 1.5M 旧设计臂的
−17.25 主要是**续训预算**差，不是干净的设计对照 → 直接进 §12 的同配方 1M 续训。

## 12. Amendment H2b（2026-10-07，operator 指定）：head32ln 同配方 1M 续训

**动机**：H2 的 500k 读数（§11）与 H1 的 1.5M 臂预算不同，不能作设计对照。operator 要求
把 `head32ln` 按与 H1 完全相同的续训协议再训 +1M，得到 `head32lnc`——它与 `head32c`
**同配方（ADR-0016）、同预算（1.5M）、只差头部设计**（v2 LN + actor_out_std 0.1 vs 旧设计）。

**臂（1 run，seed 1）**

| arm | 起点 | 协议 |
|---|---|---|
| hd_head32lnc | `hd_head32ln__1__*`（500k） | A5 T23 + ADR-0016 + ADR-0015 pool，+1M |

- 续训命令 = `run_cont_newrecipe.sh` 的同款：`--load-checkpoint`、`--total-timesteps 1000000`、
  AdamW wd0.01/cosine 2.5e-4、`--lr-floor 1e-5`、16 envs × 128 / minibatch 8、
  `--opponent pool --pool-episode True`（6×t17 `3@` + `2@random`）、
  `--reward-shaping arcsin --arcsin-mix 0.5 --win-jump 1.0`、`--arch head32ln --actor-out-std 0.1`、
  `--snapshot-interval 0`。脚本 `runs/head-depth/run_head32ln_cont.sh`。

**端点（k=1，descriptive）**：9600–9602 × 400 × 换座，bootstrap 4000。

1. **主**：`head32lnc − head32c`（同配方同预算，只差设计）。
2. `head32lnc − head32ln`（自身 500k 起点，续训增益）。
3. context：`head32lnc − o2c_base`、`− w2m_ctl`、`− head12c`、`− head22c`。

**判定**：k=1 不宣布 H1/H0；点 ≥ +10 且 CI 排除 0 标记为待补 seed 2–3。

**成本**：1 run × 1M ≈ 35–40 min；6 条 h2h ≈ 8–10 min CPU。

**H2b 结果（2026-10-07 回填）**：`hd_head32lnc` 13:05–13:22 rc=0、488/488 update、无 NaN；
6 条 h2h（k=1，9600–9602 × 400 × 换座）：

| 比较 | 点估计 | CI | winrate |
|---|---:|---|---:|
| `head32lnc − head32c`（同配方同预算，只差设计） | **−7.82** | [−19.21,+3.57] | 0.489 |
| `head32lnc − head32ln`（自身 500k 起点） | **+19.42** | **[+7.46,+31.39]** | 0.528 |
| `head32lnc − o2c_base` | −5.36 | [−17.37,+6.66] | 0.492 |
| `head32lnc − w2m_ctl` | +0.43 | [−11.03,+11.90] | 0.501 |
| `head32lnc − head12c` | −4.20 | [−15.45,+7.05] | 0.494 |
| `head32lnc − head22c` | −5.94 | [−21.22,+9.34] | 0.491 |

机制：`head32lnc` EV 0.631 / entropy **1.212** / maxKL 0.0189 / evalW 0.820
（`head32c` 0.648 / 1.409 / 0.0192 / 0.940；`head12c` 0.653/1.538/0.0156/0.910）。

**判定与结论**：主对照 −7.82，CI 跨 0、点估计偏负 → **无「v2 设计优于 v1」的证据**；
也没有任何对比满足预先的升级条件（点 ≥ +10 且 CI 排除 0）→ 不补 seed。续训杠杆在
该臂上复现（+19.42，CI 排除 0，但 k=1 不含训练-seed sd）。机制上 v2 的 entropy 降得
更快（1.212，为全部续训臂最低）、evalW 0.820 最低、EV 略低 → 更像更早的池特化，而不是
强度增益。k=1 不足以关轴，是否关闭由 operator 裁定。读数见
[`experiments/head-depth-500k.md`](experiments/head-depth-500k.md) §10。

## 13. Amendment H2c（2026-10-07，operator 指定）：head32ln 再加 2M 续训

**动机**：H2b 的 v2 臂（`head32lnc`，1.5M）对旧设计 −7.82，但 operator 决定不看
单次设计对比、继续把训练预算拉长，看 2M 额外续训能否把它推上去。

**臂（1 run，seed 1）**

| arm | 起点 | 协议 |
|---|---|---|
| hd_head32ln2m | `runs/head-depth/cont/hd_head32lnc__1__*`（1.5M） | 同 H1/H2b 协议，`--total-timesteps 2000000`（总 3.5M） |

- 命令 = `run_head32ln_cont.sh` 同款（`--load-checkpoint`、AdamW wd0.01/cosine 2.5e-4、
  `--lr-floor 1e-5`、16 envs × 128 / minibatch 8、ADR-0015 pool-episode、ADR-0016 奖励、
  `--arch head32ln --actor-out-std 0.1`、`--snapshot-interval 0`），只把
  `--total-timesteps` 改为 2000000。脚本 `runs/head-depth/run_head32ln_2m.sh`。

**端点（k=1，descriptive）**：9600–9602 × 400 × 换座，bootstrap 4000。

1. `head32ln2m − head32lnc`（上一段 1.5M；2M 续训增益）；
2. `head32ln2m − head32c`（旧设计 1.5M；额外预算能否越过设计差）；
3. context：`− o2c_base`、`− w2m_ctl`、`− head12c`、`− head22c`。

**判定**：k=1 不宣布 H1/H0；点 ≥ +10 且 CI 排除 0 标记为待补 seed。

**成本**：1 run × 2M ≈ 35–40 min（单进程实测 1M ≈ 17 min）；6 条 h2h ≈ 8–10 min CPU。

**H2c 结果（2026-10-07 回填）**：`hd_head32ln2m` 13:27–13:59 rc=0、976/976 update、无 NaN；
6 条 h2h（k=1）：

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

**读法**：额外 2M 没有换来可用增益（对自身 −0.72，CI 跨 0，平台期）；对 v1 设计
+1.74（设计差已被预算抹平）、对 `o2c_base` −0.58（平）、对 `w2m_ctl` +9.99
（CI 含 0，点接近 +10）。熵继续下降到 1.163，evalW 回升到 0.910。无 H1。
operator 随后的指定：再训 5M（§14）。

## 14. Amendment H2d（2026-10-07，operator 指定）：head32ln 再加 5M 续训

**动机**：H2c 显示 2M 后进入平台（对自身 −0.72），但 operator 决定继续拉长预算，
再训 5M（总 8.5M）。

**臂（1 run，seed 1）**

| arm | 起点 | 协议 |
|---|---|---|
| hd_head32ln5m | `runs/head-depth/cont/hd_head32ln2m__1__*`（3.5M） | 同 H1/H2b/H2c 协议，`--total-timesteps 5000000`（总 8.5M） |

- 命令 = `run_head32ln_2m.sh` 同款，只把 `--total-timesteps` 改为 5000000、起点换为
  `hd_head32ln2m`。脚本 `runs/head-depth/run_head32ln_5m.sh`。

**端点（k=1，descriptive）**：9600–9602 × 400 × 换座，bootstrap 4000。

1. `head32ln5m − head32ln2m`（自身 3.5M；5M 增益）；
2. `head32ln5m − head32c`（v1 1.5M）；
3. context：`− o2c_base`、`− w2m_ctl`、`− head12c`、`− head22c`。

**判定**：k=1 不宣布 H1/H0；点 ≥ +10 且 CI 排除 0 标记为待补 seed。

**成本**：1 run × 5M ≈ 80–85 min（单进程 ~61k steps/min）；6 条 h2h ≈ 8–10 min CPU。

**H2d 结果（2026-10-07 回填）**：`hd_head32ln5m` 14:04–15:23 rc=0、2441/2441 update、无 NaN；
6 条 h2h（k=1）：

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

**读法**：额外 5M 给出正点估计（对自身 3.5M +16.69），但 CI 含 0；对全部 raw 比较对象
（`w2m_ctl`/`o2c_base`/`head32c`/`head12c`/`head22c`）点估计全为正（+3.2…+21.9），
其中 `head22c` 的 CI 排除 0。熵继续降到 1.140、maxKL 升到 0.0422。无 H1（k=1）。
operator 随后指定：把该模型加入 7g523-web 的 manifest 与搜索配置（见 §15 / web 文档）。

## 15. 7g523-web 接入（2026-10-07，operator 指定）

**目标**：在 `7g523-web` 的默认 manifest（`traces/pool10/manifest.json`，镜像
`traces/study/manifest.json`）里加入：

1. raw 档 `head32ln5m`，spec = `ckpt:runs/head-depth/cont/hd_head32ln5m__1__1791396292/agent.pt`；
2. 搜索档 `search_head32ln5m`，spec = `rolloutt:<同一个 agent.pt>`，`search_config` pin
   `t=5 / K=32 / C=6 / max_rollout_ply=400 / rollout_opponent=ckpt:runs/w2m_ctl__11__1790516900/agent.pt /
   aggregate=mean`（与 `search_leafq` 同搜索配方，base 与 value 都是本模型）。

**评级**：本模型不在 joint probit-MLE 里（无测量局），μ 按 k=1 h2h 对 `w2m_ctl` 的
+11.16 粗略估为 **≈ 198.6**（搜索档按 `search_leafq` 的搜索增益 +72.6 粗估 ≈ 271.2），
σ 给宽值 8.0、`games: 0`，并在 subject 里标注 `web_local`；不写回任何测量报告。

**回滚**：改前备份 `artifacts/web-head32ln5m/manifest_before.json`；两份 manifest 逐字节同步。

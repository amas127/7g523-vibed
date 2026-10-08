# O2 计划：trunk 深度 + 归一化（2×2 因子设计 + 参数量对照）

> **状态：r1（2026-10-06，worker 按红队审查 [`depth-normalization-review.md`](depth-normalization-review.md)
> 逐条处置后修订）**。r0 是父代理起草、未经审查的版本；r1 已处置红队全部 10 条 finding
> （1 fatal + 4 major + 5 minor，处置表见下），预注册字段（臂 / 参数量 / 端点 / 门槛 / 扩展 /
> deal seed 台账）已重算到自洽。
> **开工前置**：owner 批准（本文件 §4 判定规则与 §1 预算即待批件）。红队判定为 BLOCK，
> 修完 F1/F2/F3/F5 后放行；r1 即按该清单修完的版本。
>
> **口径**：revision-3（`rules_id=2e36dbea44893696`）、obs v5（2 家 161 维）、PPO memoryless MLP。
> **来源**：[`post-v5-structural-options.md`](post-v5-structural-options.md) §3 O2/§5、
> [`experiments/README.md`](experiments/README.md) §0 第 17 条/§3、[`rl-bottlenecks-review.md`](rl-bottlenecks-review.md)
> §1/§5/§9.5、[`plans.md`](plans.md) §1.2/§6/§7 C-1/C-5、
> [`stage0-oracle.md`](experiments/stage0-oracle.md)、[`stage0-critic-ceiling.md`](experiments/stage0-critic-ceiling.md)、
> [`stage0-variance.md`](experiments/stage0-variance.md)、[`stage0-fixed-deals.md`](experiments/stage0-fixed-deals.md)、
> [`twin-towers-500k.md`](experiments/twin-towers-500k.md)、
> [`evaluation-protocol-validation.md`](experiments/evaluation-protocol-validation.md)（EPV）、
> [`warmstart-adamw-1m.md`](experiments/warmstart-adamw-1m.md)。

## 修订表（r0 → r1，红队 finding 逐条处置）

| ID | 级别 | 处置 | 落在本文/证据 |
|---|---|---|---|
| F1 | **fatal** | **采纳**。原 k=3 判定式「z 与 t CI 同时排除 0 且点 ≥ +10」在自预注册下方不可达：按 `duel.py:269-271` 的 `se = max(RMS(bootstrap SE), seed sd)/√k`，k=3 时 t 项恒为更紧一侧，用 D-A 实测输入（per-训练-seed 隐含 SE 6.52/9.93/6.64、RMS 7.86、between-seed sd 11.67，`stage0-oracle.md` §5.2/§5.3）得 se=6.74 → t 半宽 29.0，即确认门槛实际是「点 ≥ +19.9…+42.2」，确认概率 ≲2%；且 5 ≤ 点 < 10 区间无处置、止损（点 < +5）会误杀真 +10（14–31%）。改法三段：① **主分析无条件 k=7**（原 §4「条件扩展」整段删除，避免条件读数与 winner's curse）；② 判定回到冻结口径 **CI 排除 0 且点 ≥ +10**（`experiments/README.md` §3 / `plans.md` §7 C-1），**t-CI 只作稳健性读数、不作 gate**；③ 显式定义 5 ≤ 点 < 10 带、止损改为 **CI 上界 < +5**；④ 补齐本预算下的功效表，并写明「门槛卡在真值上 ⇒ 真 +10 的 confirm 概率上界 50%」（EPV §6 同型结论，`evaluation-protocol-validation.md:267-269`）。红队并列给出的「对 +10 做单侧等效/非劣检验」**不采纳**：它要改结论语式（超出 C-1 只统一 +20 语式的授权），且按实测 sd 8.43–17.2 要 80% power 需每臂 k≈6–24（EPV §5 的牌数表也不覆盖 seed-sd 项），预算超本轮；EPV §6:306 的「k≥5 或按 bootstrap SE 预注册」支持选 k=7。 | §1、§4；`duel.py:256-287`、`duel.py:309-336`、`experiments/README.md` §3、`plans.md:325` |
| F2 | major | **采纳**。容量混淆（`deep` 比 `base` 多 +16,512 参数 = +30%）原来无对照，且 r0 §9.5 的「hidden≈160」超配（总参 74,059，反向 +3.3%/+2.2%）。新增第 5 臂 **`wide` = `--arch shared --hidden-size 157`**：实测 total **72,202**（`.venv/bin/python` 现场计数，公式 `h²+302h+139`），比 `deep` +0.71%、比 `deep_ln` −0.35%，零代码改动、只作 descriptive 归因。§9.5 的 160 一律改 157。 | §1、§4、§5；`tests/test_networks.py:231-241` 为锚点 |
| F3 | major | **采纳**。r0 §2 漏了 `--arch` 的 argparse 硬编码 choices（`train.py:200-201` 只有 `shared`/`towers`，`_ARCHITECTURES` 是私有名 `networks.py:67`），只改 networks.py 时 12 条命令会被 argparse 直接拒；且 r0 说「跨 arch warm-start fail-loud」与现状相反——现状是**静默部分复制**（`networks.py:697-698` 的 `if origin is None: continue`；`_mapped_source_key` 只认 shared↔towers 桥接，`networks.py:647-662`；`warm_start_from` 仅当 nvec 与 arch 都相同才整表加载，`networks.py:732-744`）。§2 增两条明确改动：choices 加三个新值 + arch 白名单守卫（非 {shared,towers} 的跨 arch 一律 `raise WarmStartLayoutError`），并补「`deep_ln`→`deep` 必须 raise」单测。 | §2；`train.py:200-201`、`networks.py:617-637/647-662/665-707/732-744`、`tests/test_networks.py:70-199` |
| F4 | major | **采纳**。r0 §0 的「D-A/D-C → 优化/表示限制」超证据分辨率且漏最相关证据。改写为：D-A 3/3 seed 负、点 −13.87，但 **t-CI 上界 +15.11 含 +10**（只能排除大正效应，`stage0-oracle.md` §5.3/§7.1）；D-C 的 **`deep_head`（+16,641 参数，正是「再加一层 128 宽」）final EV 0.5211 vs base 0.5216** 是对 `deep` 的直接负先验（`stage0-critic-ceiling.md:43/113-114/128/167`）；D-D 已给出 r0 §9.6 所依赖的答案（无免费稳定器，`stage0-variance.md` §2.5/§3）；补引 `experiments/README.md` §0 第 17 条（「训练侧结构项只余 O2」）与 `rl-bottlenecks-review.md` §1/§9.5，并说明 O2 与被关清单的关系。 | §0、§6、§9；`experiments/README.md:130-140`、`rl-bottlenecks-review.md` §1/§9.5 |
| F5 | major | **采纳**。r0 只说「按仓库规则」+「合并读数写进报告」，但 `combine_duel_seeds` 要求 `left_id/right_id/bootstrap/confidence` 完全一致（`duel.py:317-336`），r0 §5 的 id 带 `_sN` 后缀 → 跨训练 seed 直接调用会 raise；而手写合并可选 k=3 或 k=9 两种「合法」口径（D-A 实测 se 6.74 vs 4.96，CI 差 2.5×）。§4 内嵌**预注册复算脚本**（写死输入路径、训练-seed 口径 k、flatten 适配器），§5 的 h2h id 改为不带后缀的 `base`/`deep_ln`/…，可直接喂 `combine_duel_seeds`。该脚本已在现场用 D-A 三份产物验证：复现 `stage0-oracle.md` §5.3 的点 −13.87、z-CI [−27.08,−0.67]、t-CI [−42.86,+15.11]、bootstrap SE 7.86、between-seed sd 11.67。 | §4.3、§5；`duel.py:288-336` |
| F6 | minor | **采纳**。vs-random 饱和 sanity 在 r0 命令下取不到数据（`--eval-interval` 默认 0，`train.py:452`；`--snapshot-interval 0` 无 200k 快照；`--checkpoint-interval` 默认 100 只在同名文件上反复覆盖，`train.py:437-441`）。训练命令加 `--eval-interval 50`（约 9 个 eval 点/run，同时给 §4 机制曲线）；sanity 值加抽样容差。 | §3、§4.4；`train.py:1070-1084` |
| F7 | minor | **采纳**。§3 补评估成本（19 条 h2h ≈40–55 min CPU，锚 `v5-optimization-plan.md:618`：1200 deal ≈2–3 min @cpu/4）与扩展取消后的新总价；§5 每条 h2h 加 `--games-out`（`tools/head_to_head.py:104-112`，多 seed 会落 `_seed<k>` 后缀）。 | §3、§5 |
| F8 | minor | **采纳（放置保持 r0 设定并写死）**。LN 放置预注册为「**每个** hidden Linear 之后、激活之前，含最后一个（heads 输入侧）」——这正是 §1 参数表 +512/+768 对应的放置，与 deep_ln=+3 LN 自洽。新增结构性断言单测；§4 机制读数补「V sd / 初始 value 尺度 / value loss 量级」（`networks.py:348-355` 用 `layer_init(std=√2)`、critic 头 std=1.0，LN 会吸收大部分 init 尺度，`layer_init` 见 `networks.py:51-55`）；§6 写明 LN 臂与 base 臂**有效初始尺度不同**这一通道。红队建议的替代「末层不做 LN」记录为被考虑项：它与已冻结的参数表冲突（deep_ln 会变 +512），不改。 | §1、§2、§4.4、§6 |
| F9 | minor | **采纳**。锚点修正：`_ARCHITECTURES` 在 `networks.py:67`（r0 写 66）、shared trunk 在 `networks.py:348-353`（r0 写 349-353）、`--arch` choices 在 `train.py:200-201`（r0 未提）；臂名统一为 `base`（`--exp-name o2_base --arch shared`），§3/§5 不再出现 `o2_shared`/`base_s1`。 | §2、§3、§5 |
| F10 | minor | **采纳**。来源列表补 `stage0-variance.md`、`stage0-fixed-deals.md`、`rl-bottlenecks-review.md`（§0/§9 各引一句）。 | 头部来源、§0、§9 |

**处置后的自洽性核查**（本次现场核算，命令用 `.pi/limits/run_limited.sh cpu .venv/bin/python`）：

- 参数量：`base` 55,179 / `ln` 55,691 / `deep` 71,691 / `deep_ln` 72,459（沿用 `tests/test_networks.py:231-241` 锚点 + 手算）；`wide`（hidden=157）现场实例化实测 **72,202**。
- 步数：`--num-envs 8 --num-steps 128` → 1024 步/update；500,000/1024 = 488 update → **499,712 步**（与 D-A 产物 `args.json` 的 `global_step 499712` 一致）。
- 合并口径：预注册脚本已复现 D-A §5.3（见 F5 行）。
- 测试基线：`pytest --collect-only -q` = **859 tests**（§8 验收沿用）。

## 0. 背景与假设

已关闭的模型复杂度轴（列在此处防止重开，来源 `experiments/README.md` §0 / `plans.md` §6 N-1…N-25）：
宽度 128→256（−33.6）/→512（200k +19.2 假阳性 → 800k k=3 −11.49）；双塔 `arch=towers`
（主臂不可分、交互臂 −15.51、从零 −20.89）；激活 tanh/gelu/silu（不赢，tanh −48.8）；
历史编码 EVH/D1-lite/序列 GRU（全 null）。

2026-10-06 的 Stage-0 四探针把主假设从「信息缺口」换成「优化/表示限制」，但要按证据分辨率读：

- **D-A oracle-obs**：obs 追加真实隐藏信息 vs 等容量全 0 对照，3/3 训练 seed 为负、点估计
  **−13.87 Elo**（z-CI [−27.08,−0.67] 在负方向排除 0）。**但合并 t-CI [−42.86,+15.11] 覆盖
  +10**（`stage0-oracle.md` §5.3），作者自己在 §7.1 写「1–2 run 级探针只能排除大正效应」。
  所以它支持「信息缺口不是主瓶颈」，**不支持**「信息增益已被排除到 +10 以下」。
- **D-C critic 天花板**：**`deep_head` = 冻结 trunk + `Linear(128,128)`+relu+`Linear(128,1)`，
  即「再加一层 128 宽」，+16,641 参数**，final EV 0.5211 vs shipped `base` 0.5216（ΔEV −0.0005，
  `stage0-critic-ceiling.md:43/113-114/128/167`）。这是本实验 **`deep` 臂最接近的负先验**；
  D-C 的整体结论（value 残差不可约）不等于「trunk 加深无用」，但两者同号风险很高。
- **D-D 方差 screen**：`--target-kl 0.03` 在默认配方下是 no-op；`--learning-rate 1e-4` 把 run sd
  从 17.21 砍到 3.52 Elo 但 200k 均值弱 −18.3 Elo → **无免费稳定器**（`stage0-variance.md`
  §2.5/§3）。本计划因此不加「稳定器臂」（§9 第 6 条的答案已给）。
- **D-B 固定牌局**：四个差值 z/t CI 全含 0，null（`stage0-fixed-deals.md`）。

**授权位置**：`experiments/README.md` §0 第 17 条：「训练侧结构项只余 O2（深度/归一化，计划见
[`depth-normalization-plan.md`](depth-normalization-plan.md)）与条件项」；
`post-v5-structural-options.md` §3 O2（改动面、先验、判据、风险）。`rl-bottlenecks-review.md` §1
的「继续扫训练配方没有 ROI」关的是**配方**线，§9.5 关的是 `plans.md` §6 的既有清单
（含容量**宽度** N-2、奖励/历史/self-play 等）；trunk **深度/归一化** 不在该清单内，且正是
README §0 第 17 条指定的下一项——本计划是这个指定项的预注册。

当前 trunk（`src/seven523/networks.py:348-353`）＝输入层 + 一个隐藏层（2 个 Linear，hidden 128，
relu）。**H1**：把 trunk 加深（3 个 Linear）并在各 hidden Linear 后加 LayerNorm，在 500k 从零、
同配方下能比 2-Linear 对照拿到 **≥ +10 Elo**（判定规则见 §4.1）；**H0**：拿不到，即合并 CI 上界
< +5 → 关闭该轴。2×2 设计同时回答：深度单独有用吗（`deep−base`）、归一化单独有用吗（`ln−base`）、
两者叠加（`deep_ln−base`）、归一化在深网上的增量（`deep_ln−deep`）；第 5 臂 `wide` 钉住容量解释。

## 1. 臂（预注册：5 臂 × 不等 seed = 23 run）

| arm | `--arch` | 额外 flag | trunk 结构 | 总参数量 | 相对 base | 训练 seed | run 数 |
|---|---|---|---|---:|---:|---|---:|
| base | `shared`（现状） | — | Linear(161,128)+relu+Linear(128,128)+relu | 55,179 | — | 1–7 | 7 |
| ln | `ln` | — | base + 每个 hidden Linear 后 LayerNorm(128)（激活前） | 55,691 | +512 | 1–3 | 3 |
| deep | `deep` | — | Linear(161,128)+relu+Linear(128,128)+relu+Linear(128,128)+relu | 71,691 | +16,512 | 1–3 | 3 |
| deep_ln | `deep_ln` | — | deep + 每个 hidden Linear 后 LayerNorm(128)（激活前） | 72,459 | +17,280 | 1–7 | 7 |
| wide | `shared` | `--hidden-size 157` | Linear(161,157)+relu+Linear(157,157)+relu | 72,202 | +17,023 | 1–3 | 3 |

- 参数量锚点：`tests/test_networks.py:231-241`（base 55,179；trunk 37,248、actor 17,802、critic 129）；
  全宽公式（输入 161、actor 头 138 行）`total = h² + 302h + 139`，h=157 → 72,202（现场实测）。
- **k 的分配理由**：主端点（`deep_ln − base`）拿全部功效预算 k=7（无条件，无两阶段扩展）；
  归因臂 `ln`/`deep`/`wide` 只作 descriptive，k=3，按训练 seed 1–3 与 `base` 配对。
- 全臂 hidden=128（`wide` 除外，见 F2）、activation=relu、heads 不变；训练 seed 编号在臂间对齐；
  **全部从零**训练，禁止 `--load-checkpoint`（deeper/ln 布局非同一身份，热启动会混入 T23 的
  warm-start 效应，见 `warmstart-adamw-1m.md` §2.1 与 `rl-bottlenecks-review.md` §1）。
- LN 放置（预注册，F8）：**每个 hidden Linear 之后、激活之前，含最后一个 hidden Linear**
  （即 heads 的输入侧也 LN）；`ln` 2 个 LN、`deep_ln` 3 个 LN，与 +512/+768 自洽。
- 深度先只加一层（3 Linear）；4 层/残差留作条件项（§7），不在本轮预注册。

## 2. 实现（file:line）

- **方案 A（采用）**：在 `_ARCHITECTURES`（`src/seven523/networks.py:67`）新增
  `"ln"`/`"deep"`/`"deep_ln"`，trunk 分支加在 `src/seven523/networks.py:348-353`（与现有
  `shared`/`towers` 并列，见 `networks.py:331-353`）。ckpt 的 `arch` 身份字段已存在
  （`networks.py:266`；round-trip 用例见 `tests/test_networks.py:200-210`）。
- **方案 B（不采）**：新增 `--trunk-layers`/`--trunk-norm` 正交 flag；需要同时扩展 ckpt 布局身份与
  warm-start 检查，面更大、收益不匹配（ADR-0010 单一 owner 接缝）。
- **必须一并改的两处（红队 F3）**：
  1. `--arch` 的 argparse choices 是硬编码的 `train.py:200-201`（`choices=["shared","towers"]`），
     而 `_ARCHITECTURES` 不在 `__all__`（`networks.py:34-48`）且 train.py 目前未 import 它；
     实现时同步扩为五个值（或把 `_ARCHITECTURES` 导出后复用，实现阶段二选一，行为等价）。
  2. **跨 arch warm-start 守卫**：现状是静默部分复制（`networks.py:697-698` 缺 key 直接
     `continue`；`_mapped_source_key` 只认识 shared↔towers 桥接，`networks.py:647-662`）。
     在 `warm_start_into`（`networks.py:665-707`）obs/layout 检查之后加白名单：
     **仅当 `{agent.arch, loaded.arch} ⊆ {"shared","towers"}` 才允许跨 arch 桥接，否则
     `raise WarmStartLayoutError`**（现有的 `networks.py:617-637` 异常类可承载）。
     对现网无行为变化（现存 arch 只有 shared/towers），但对新 arch 变成 fail-loud。
     另外 `wide`（hidden=157）与 hidden=128 的 `shared` ckpt 之间，`warm_start_from` 走
     `networks.py:732-734` 的整表 `load_state_dict`，形状不符会 `RuntimeError`，由
     `train.py:635-642` 转成 `SystemExit`——已是 fail-loud，无需额外改动。
- 约束：默认路径（`shared`/`towers`，hidden=128）逐位不变；不动 obs、动作空间、PPO 数学、`ppo.py`；
  `wide` 臂零代码改动（只传 `--hidden-size 157`）。
- 新增单测（`tests/test_networks.py`）：
  1. 三个新 arch 的参数量断言（55,691 / 71,691 / 72,459）；
  2. `wide` 的参数量断言（hidden=157 → 72,202）；
  3. 默认 `shared` 固定输入逐位不变（沿用现有风格）；
  4. 新 arch 正向 shape/无 NaN；ckpt round-trip 记录并读回 arch；
  5. **结构断言**：`agent.network` 的模块类型/顺序写死（Linear→LN→ReLU 序列、
     `deep`/`deep_ln` 三层），对固定输入比对 LN 位置（F8）；
  6. **跨 arch warm-start fail-loud**：`deep_ln→deep`、`shared→deep`、`ln→shared` 必须
     `raise WarmStartLayoutError`；`shared↔towers` 的既有复制语义作为回归用例
     （`tests/test_networks.py:101-155`）保持绿。
- 冒烟：5 个臂各 2048 步，无 NaN，走 `run_limited.sh gpu`。

## 3. 训练配方与命令（23 run）

与 Stage-0 D-A/D-B 同款（`train.py` 当前默认，只改 `--arch`/`--hidden-size`/`--eval-interval`）：

```bash
cd /home/amas/.local/src/7g523
train_one() {  # $1=arm $2=arch $3=hidden $4=seed
  .pi/limits/run_limited.sh gpu .venv/bin/python -m seven523.train \
    --exp-name o2_$1 --seed $4 --total-timesteps 500000 --arch $2 --hidden-size $3 \
    --num-envs 8 --num-steps 128 --num-minibatches 4 --update-epochs 4 \
    --learning-rate 2.5e-4 --anneal-lr --lr-schedule linear --optimizer adam \
    --gamma 0.99 --gae-lambda 0.95 --norm-adv --clip-coef 0.1 --clip-vloss \
    --ent-coef 0.01 --vf-coef 0.5 --max-grad-norm 0.5 --activation relu \
    --opponent random --reward-shaping terminal --num-players 2 --torch-deterministic \
    --tensorboard true --snapshot-interval 0 --eval-interval 50 \
    --cuda True --run-dir runs/o2-depth
}
for s in 1 2 3 4 5 6 7; do train_one base shared 128 $s; train_one deep_ln deep_ln 128 $s; done
for s in 1 2 3;      do train_one ln ln 128 $s; train_one deep deep 128 $s; train_one wide shared 157 $s; done
```

每 run 500k（实际 **499,712** 步 = 488 update × 1024）；`--eval-interval 50` 给约 **9 个
eval 点/run**（`train.py:1070-1084`，`--eval-opponent random`、100 局，`train.py:454-455`），
同时供 §4.4 的机制曲线与 vs-random 饱和 sanity。

**成本（F7，含评估）**：

- 训练：23 run ≈ **3.1–5.8 h GPU 槽**（D-A 实测 8–12 min/run 为 base 参照；`deep`/`deep_ln`
  trunk 乘加数 36,992 → 53,376（+44%，粗算、忽略 bias），LN 另加少量，sps 预计 −10~25%，
  `--eval-interval` 再占少量）。
  单进程串行；比 r0 的 12 run 多 11 run ≈ +1.5–2.5 h。
- 评估：19 条 h2h（主端点 7 + 归因 12）≈ **40–55 min CPU**（锚 `v5-optimization-plan.md:618`：
  1200 deal ≈2–3 min @cpu/4）。r0 只算训练，这部分是 F7 补上的。
- 资源规则：`.pi/limits/run_limited.sh`（≤4 GPU / ≤12 CPU 槽），同一执行代理同时只跑一个 GPU 进程。

## 4. 端点与统计（预注册）

### 4.1 端点与判定规则（三段式，取代 r0 的 k=3 两段式）

- **主端点（confirmatory）**：`deep_ln − base`，deal-twin 换座 h2h，**按训练 seed 配对，k=7**
  （两边各 seed 1–7）；每训练 seed **3 个 deal seed × 400 副**（1,200 副）→ 合计 **8,400 副**。
- **次要（归因，descriptive）**：`ln − base`、`deep − base`、`deep_ln − deep`、`wide − base`，
  同协议但 **k=3**（训练 seed 1–3）、每比较 1,200 副。
- **合并**：`src/seven523/duel.py:288` 的 `combine_duel_seeds`（点 = seed 均值、
  `se = max(RMS(bootstrap SE), seed sd)/√k`，`duel.py:256-287`），**z=1.96**（`experiments/README.md`
  §3 / `plans.md` §7 C-1 冻结口径）。t-CI（`t(0.975,k-1)`：k=7 → 2.447、k=3 → 4.303）**只作
  稳健性读数记录，不作 gate**——k=3 时它恒为更紧一侧（F1）。
- **判定（主端点）**：
  1. **值得行动（H1）**：合并 95% CI 完全排除 0 **且** 点估计 **≥ +10**。
  2. **弱正证据带**：不满足 1 但点估计 ≥ +5（含「点 ≥ 10 但 CI 跨 0」）：记为
     「**无 ≥ +10 证据**」，**不宣布 H1、不宣布 H0**、不写入 `plans.md` §6 负结果清单、
     本轮不自动补 seed；是否重开由 owner 按新预算单独裁定（届时须新预注册）。
  3. **止损（H0，关闭该轴）**：合并 95% CI **上界 < +5**（即在 95% 下排除 ≥ +5）。
     仅「点估计 < +5」**不构成**关闭（r0 的写法会误杀真 +10；见 4.2 表）。
- **禁止 winner's curse**：只用每 run 的最终 `agent.pt`，不得按训练曲线/快照峰值选 checkpoint
  （`warmstart-adamw-1m.md` §2.1、`rl-bottlenecks-review.md` §5）。
- **扩展规则：无**。r0 §4 的「z-CI 排 0 且点 ≥ +10 → 补 seed 4–7」整段删除：本版把 k=7 并入
  默认预算，消除条件读数与「条件触发使点估计系统性偏高约 3 Elo」的问题（T23 实测 +11.48 → +8.20，
  `rl-bottlenecks-review.md` §1）。

### 4.2 本预算的功效语义（正态近似；门槛卡在真值上 ⇒ confirm 概率上界 50%）

主端点 k=7 时 `se = max(7.86, sd)/√7`（7.86 是 D-A 实测的 RMS bootstrap-implied SE，
`stage0-oracle.md` §5.3；sd 取 D-D/历史实测 8.43/11.92/14.4/17.2，`rl-bottlenecks-review.md` §5）：

| between-seed sd | se | z 半宽 | t 半宽 | 真 +10 时「值得行动」概率 | 真 +15 | 真 +20 | H0 时 FPR | 真 +10 时被规则 3 误关 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 8.43 | 3.19 | 6.2 | 7.8 | **50%** | 94% | >99% | 0.1% | ~0% |
| 11.92 | 4.51 | 8.8 | 11.0 | **50%** | 87% | 99% | 1.3% | 0.1% |
| 14.4 | 5.44 | 10.7 | 13.3 | **45%** | 79% | 96% | 2.5% | 0.2% |
| 17.2 | 6.50 | 12.7 | 15.9 | **34%** | 64% | 87% | 2.5% | 0.3% |

读法：门槛设在真值上时 confirm 概率恒 ≤50%，且**不随 k 增大而超过 50%**（EPV §6:267-269 对
+20 门槛的同型结论）；要 80% power 只能把结论语式改成对 +10 的单侧等效/非劣检验并大幅加预算
（z 检验、Δt=+10 口径下每臂 k≈6–24），超出本轮授权（对比 r0 的 k=3 规则：真 +10 的 confirm
概率 ≲2%，止损误杀 14–31%，[5,10) 无处置）。H0 侧：真 null 时该轴以 78–94%（sd 8.43→17.2）
的概率关闭。
报告的措辞必须与本节一致：**「值得行动」= 过门；「弱正证据带」= 不足以行动也不关闭**。

### 4.3 预注册复算脚本（写死输入与 k，F5）

```python
# runs/o2-depth/recompute_o2.py —— 主/次要端点合并读数（只读 h2h JSON）
import json
from seven523.duel import combine_duel_seeds

PAIRS = {                      # 比较 -> (left, right, 训练 seed 列表)
    "deep_ln-base": ("deep_ln", "base", [1, 2, 3, 4, 5, 6, 7]),   # 主端点 k=7
    "ln-base":      ("ln", "base", [1, 2, 3]),
    "deep-base":    ("deep", "base", [1, 2, 3]),
    "deep_ln-deep": ("deep_ln", "deep", [1, 2, 3]),
    "wide-base":    ("wide", "base", [1, 2, 3]),
}
T_CRIT = {3: 4.303, 7: 2.447}

def flat(path):
    """一份 h2h JSON（单训练 seed、3 deal seed 合并）-> combine_duel_seeds 输入。"""
    c = json.load(open(path))["combined"]          # run_duel_seeds 的 envelope
    inner = c["combined"]
    return {
        "left_id": c["left_id"], "right_id": c["right_id"],
        "bootstrap": c["bootstrap"], "confidence": c["confidence"],
        "elo_diff": inner["elo_diff"]["mean"], "elo_diff_ci": inner["elo_diff"]["ci"],
        "winrate": inner["winrate"]["mean"], "winrate_ci": inner["winrate"]["ci"],
        "mean_score_diff": inner["mean_score_diff"]["mean"],
        "mean_score_diff_ci": inner["mean_score_diff"]["ci"],
        "deal_sign": inner["deal_sign"], "deals": c["deals"], "games": c["games"],
    }

for name, (left, right, train_seeds) in PAIRS.items():
    rows = [flat(f"runs/o2-depth/h2h_{name}_s{s}.json") for s in train_seeds]
    out = combine_duel_seeds(rows, seeds=train_seeds)   # k = 训练 seed 数
    assert out["left_id"] == left and out["right_id"] == right, (name, out["left_id"], out["right_id"])
    e = out["combined"]["elo_diff"]
    se, m = e["se"], e["mean"]
    t = T_CRIT[out["k"]]
    print(f"{name}: k={out['k']} mean={m:.2f} z-CI=[{m-1.96*se:.2f},{m+1.96*se:.2f}] "
          f"t-CI=[{m-t*se:.2f},{m+t*se:.2f}] se={se:.2f} sd={e['between_seed_sd']:.2f}")
```

- 口径写死为**训练 seed**（k = 输入 JSON 个数），不用 9/21 个 deal-set 独立口径
  （D-A 实测两者 se 6.74 vs 4.96，CI 差 2.5×）。
- 该脚本已用 D-A 三份产物现场验证，逐位复现 `stage0-oracle.md` §5.3：
  `k 3 mean -13.87 z-CI [-27.08, -0.67] se 6.74 boot_se 7.86 sd 11.67`，t-CI `[-42.86, 15.11]`。
- 约束：`combine_duel_seeds` 要求输入 `left_id/right_id/bootstrap/confidence` 一致且 k≥2
  （`duel.py:309-336`），因此 §5 的 h2h id 不带 seed 后缀。

### 4.4 机制读数（归因用，不参与门槛）

参数量/深度、sps、训练曲线（`explained_variance`/`entropy`/`approx_kl`/`value_loss`）、
梯度范数、**V sd / 初始 value 尺度**（LN 吸收 init 尺度的直接读数，
`networks.py:348-355`/`51-55`；仓库惯例见 `twin-towers-500k.md` §5.3）。

vs-random 饱和 sanity（F6）：`base` 臂的 `eval_win_rate`（每 50 update 一次，100 局）在
update 200/450（≈205k/461k 步）应落在既有饱和带 **≈0.88–0.90 ± 0.03**（100 局 eval 的抽样 sd）；
落在 0.83–0.94 之外才判管线异常。r0 的「200k/500k 快照」写法不可执行（`train.py:437-452`），已改。

### 4.5 多重性（Q-13）

主端点只有一个；`ln−base`、`deep−base`、`deep_ln−deep`、`wide−base` 一律 descriptive，
**其结论不得进入 `plans.md` §5/§6 的轴级判定，也不得写成「值得行动」**。要把某个归因比较升级为
行动性论断，只有两条**事先写明**的路径：(i) 主端点过门后作为其归因（同一批 run）；或
(ii) 用**独立新 bank**（新 deal seed）按主端点同款规则（k=7、同门槛）重做该比较——
两条都必须在报告里先声明再使用，禁止事后从四个比较里挑。

### 4.6 deal seed 台账

初选 **9600 / 9601 / 9602**（每训练 seed 3 × 400 副）：`docs/` 全库检索只在本文件出现，与已记录
bank（30–32、410–414、9200/9300/9400、9500）无交集。**待确认项**：这三个数值落在
`stage0-critic-ceiling.md:73` 的叶片采集 seed 区间（9400–10599）内，但那是 critic 拟合数据集
（`runs/stage0/critic/data`）而非 h2h bank，且配对 h2h 两侧用同一批 deal，不构成主端点污染——
报告须标注这层关系。开跑前仍按台账再做一次全库检索，确认后写进报告并冻结。

## 5. 评估命令（模板）

```bash
cd /home/amas/.local/src/7g523
# 主端点（7 条：训练 seed 1–7）；归因比较同型，id 固定不带 seed 后缀
for s in 1 2 3 4 5 6 7; do
  .pi/limits/run_limited.sh cpu .venv/bin/python tools/head_to_head.py \
    --left  "deep_ln=ckpt:runs/o2-depth/o2_deep_ln__${s}__<ts>/agent.pt" \
    --right "base=ckpt:runs/o2-depth/o2_base__${s}__<ts>/agent.pt" \
    --seeds 9600,9601,9602 --pairs 400 --bootstrap 4000 --device cpu --workers 4 \
    --games-out runs/o2-depth/games/deep_ln-base_s${s}.jsonl --json \
    > runs/o2-depth/h2h_deep_ln-base_s${s}.json
done
# 归因（12 条，训练 seed 1–3）：ln-base / deep-base / deep_ln-deep / wide-base 同型
```

- 条数：主 7 + 归因 12 = **19 条**；`--games-out` 在多 deal seed 下会落
  `_seed9600` 等后缀（`tools/head_to_head.py:104-112`），逐副 JSONL 供审计（ERA §6.2 惯例）。
- 评估走 `cpu` 槽；合并读数用 §4.3 的脚本，写进报告。

## 6. 风险与混淆

- **深度 vs 容量**：`deep`/`deep_ln` 比 `base` 多 16.5–17.3k 参数（+30%/+31%）。第 5 臂
  `wide`（hidden=157，total 72,202）把 `deep`/`deep_ln` 的参数量配到 ±0.7%，是本轮唯一的容量解释
  对照；它只作 descriptive。宽度 256/512 的既往负结果只覆盖那两个量级，不覆盖 157。
- **`deep` 的负先验**：`deep_head`（同样的「再加一层 128 宽」，+16,641 参数）在 D-C 的
  leaf-EV 上 null（0.5211 vs 0.5216）。若 `deep` 臂也 null，与 D-C 一致，报告须点明这不是独立复现。
- **优化难度**：更深的 PPO 可能更难训；若 `deep` 臂 KL/EV/entropy/grad-norm 退化，结论是
  「该配方下无法判定深度是否有用」，**不是**「表示容量不是瓶颈」——报告必须区分这两种读法，
  并写明重开条件。
- **LN 与 init/Adam 的交互**：LN 改变有效尺度，也吸收 `layer_init(std=√2)` 的初始尺度（γ=1/β=0），
  因此 LN 臂与 base 臂的**有效初始 value 尺度不同**；lr/权重衰减保持预注册值，禁止中途调参换因子；
  §4.4 的 V sd / value loss 量级是这条通道的读数。
- **训练 seed sd**：实测 8.43/11.92/14.4/17.2（`rl-bottlenecks-review.md` §5，取代 r0 的模糊
  「8–17」）；k=7 下的后果见 §4.2，不要在报告里声称比该表更强的功效。
- **从零约束**：不能复用现有 ckpt；与历史 500k 数字只在 vs-random sanity 层面比较。
- **并发与时长**：其它会话/实验会拉低 sps（不改样本量与配方）；深度臂单步成本 +44% trunk 乘加数。
- **横截面**：任何中途「换因子」（例如给 LN 臂调 lr）都会破坏 2×2 的因子含义，一律禁止。

## 7. 条件项与不做清单

- 条件项（未触发不排期）：主端点为真且 `deep_ln` 胜出后，才考虑 (a) 4–5 层、(b) 残差连接、
  (c) 与 warm-start/更长预算组合（T8 E）。
- 不做：宽度 256/512（N-2）、激活（N-1）、双塔（N-20）、历史编码（N-25/EVH/D1-lite）重开；
  200k 单 seed 屏幕；把 vs-random 当端点；用 `--load-checkpoint` 热启动；改评估协议/统计管线
  （C-5/N-17）；为某一臂做中途调参；按训练曲线择优 checkpoint。

## 8. 交付与验收

- 产物：`runs/o2-depth/o2_<arm>__<seed>__*/`（`agent.pt`、`metrics.csv`）、
  `runs/o2-depth/h2h_*.json` + `runs/o2-depth/games/*.jsonl`、复算脚本
  `runs/o2-depth/recompute_o2.py`、报告 `docs/experiments/depth-normalization-500k.md`
  （架构 → 结果 → 判定（§4.1 三段）→ 功效边界（§4.2）→ 缺失证据/限制）。
- 验收：全套测试（现 859 + 新增，`pytest --collect-only -q` 实测 859）绿；5 臂冒烟无 NaN；
  §3/§5 命令可复现；报告论断给 file:line；报告必须包含 §4.2 表与 §4.5 的 descriptive 约束声明。
- 开工前置：owner 批准本 r1（红队审查已处置；r0 的 BLOCK 由 F1/F2/F3/F5 的修法解除）。

## 9. 开放点

红队 §9 六点的现状（r1 处置后）：

1. **LN 放置 / 2×2 是否砍**：保留 2×2（`ln`/`deep` 只多 6 run ≈45–60 min GPU，缺了无法归因）；
   放置已写死为「每个 hidden Linear 后、激活前、含末层」（§1），并加结构断言单测（§2-5）。
2. **k 与扩展**：已改为**无条件 k=7**（主端点）+ 无扩展规则；功效边界见 §4.2。
3. **多重性**：只认主端点，归因全 descriptive，升级路径见 §4.5。
4. **跨 arch warm-start**：fail-loud，且是本轮的实际代码改动（§2-2）。
5. **参数量对照**：要，用 `wide` = `--arch shared --hidden-size 157`（§1）。
6. **配方固定**：坚持「no gain under this recipe」的读法，不加稳定器臂；D-D 已答
   （`stage0-variance.md`：无免费稳定器）。

留给 owner 的三个待批项（r1 默认值已写在正文，任一项改动需回填本文件后重新冻结）：

1. **预算**：23 run ≈3.1–5.8 h GPU + 19 条 h2h ≈40–55 min CPU（比 r0 多 11 run / 7 条）。
2. **判定语义**：接受「+10 门槛 = 行动门槛，真 +10 时 confirm 概率 ≤50%」这一 EPV 同型上限，
   还是改语式（对 +10 单侧非劣）并加预算到 k≈6–24（超出本轮授权）。
3. **归因臂 k=3**：接受归因比较只有 k=3（descriptive）与 §1 的不等 k 分配，还是把预算换成
   均匀 k=5（5 臂 × 5 seed = 25 run，只比本版多 2 run ≈25 min GPU；代价是主端点 k 从 7 降到 5，
   z 半宽在 sd=11.92 参照下从 8.8 变 10.5、真 +10 的 confirm 概率 50% → 47%）。

## 10. Amendment A1（2026-10-06，主端点结果之后；operator 质询触发）

**背景**：主端点 `deep_ln − base`（k=7）= **−13.95** [−25.84,−2.06] 触发 §4.1 止损
（见 [`experiments/depth-normalization-500k.md`](experiments/depth-normalization-500k.md)）。
operator 指出「加深退化应先用残差连接防止」，而 r1 把残差列为 §7 条件项、未预注册。
A1 显式补测残差臂：**探索性 screen**，不是原预注册的 confirmatory 部分；k=3、descriptive、
按 §4.5 不进入轴级判定。本节的端点与升级规则在开跑前写死。

**臂（与已测臂参数逐位相同，唯一差异 = 第 2/3 块加 skip `h ← h + F(h)`）**

| arm | `--arch` | 结构 | 总参数 | 对照 | seed |
|---|---|---|---:|---|---|
| deep_res | `deep_res` | `deep` + 残差 | 71,691 | `deep` | 1–3 |
| deep_lnres | `deep_lnres` | `deep_ln` + 残差 | 72,459 | `deep_ln` | 1–3 |

- 实现：`src/seven523/networks.py` 的 `ResidualTrunk`（3 块，块 0 为输入投影，块 1/2 带 skip）；
  与 `deep`/`deep_ln` 的逐层模块与初始化相同，参数计数已由单测钉住
  （`tests/test_networks.py::test_residual_archs_are_parameter_matched_to_their_plain_twins`）。
- 训练/评估协议同 §3/§5（500k 从零、同配方；h2h 3 deal seed × 400 副 × 换座，bootstrap 4000）。

**端点与升级规则（先写死）**

1. **主问句**：加 skip 是否把深度惩罚收回？主端点 = `deep_res − deep` 与 `deep_lnres − deep_ln`
   （同 seed 1–3 配对）。
2. **升级到 confirmatory（唯一条件）**：两个残差-对照比较中任意一个出现**点估计 ≥ +10 且
   z-CI 排除 0** → 补该残差臂与两个非残差臂的 seed 4–7（k=7）复核。否则 A1 结束：
   残差方向记录为无增益，深度轴维持关闭。
3. **次要读数（descriptive）**：`deep_res − base`、`deep_lnres − base`。

**预期**：若 3 层退化的主因是「没有 skip 的优化问题」，`deep_res`/`deep_lnres` 应相对
它们的 plain 对照出现明显正增益；若仍 ≤ 0 且 CI 跨 0 或为负，则「该配方/预算下加深
trunk 无出路」的结论对残差变体也成立。

**A1 结果（2026-10-06 回填）**：`deep_res − deep` = **+6.82** [−4.62,+18.26]（逐 seed +3.94/−0.87/+17.40）、
`deep_lnres − deep_ln` = **−4.69** [−15.00,+5.62]；`deep_res−base` −6.18 [−13.78,+1.42]、
`deep_lnres−base` −6.34 [−25.05,+12.36]。两条残差-对照比较都未达「点 ≥ +10 且 z-CI 排除 0」的
升级门槛 → **A1 结束：残差方向同样记录为无增益，深度轴维持关闭**。读数与机制见
[`experiments/depth-normalization-500k.md`](experiments/depth-normalization-500k.md) §4b。

## 11. Amendment A2（2026-10-06，operator 提议；不确定度 critic）

**动机**：D-C 测出 value 残差里 36% 是 obs 不可消除的噪声；operator 提议让 critic 同时学
「奖励均值 + 不确定度」。若噪声被显式建模，梯度可能不再硬拟合噪声，从而改善共享 trunk 的表示，
并为后续（advantage 加权/搜索截断）提供不确定度信号。

**臂（单因子，开跑前写死）**

| arm | 结构 | 总参数 | seed | 角色 |
|---|---|---:|---|---|
| `nll` | `arch shared` + critic 输出 `(mean, logvar)`，Gaussian NLL 替代 clipped MSE | 55,308（+129） | 1–7 | 主臂（k=7，对照 base 1–7） |

- 实现：`src/seven523/networks.py` 的 `vf_nll`（`get_value_dist`；logvar 行零初始化 ⇒ 初始
  var=1）、`src/seven523/ppo.py::gaussian_nll`（logvar clamp ±10）。CLI：`--vf-nll true`。
- **单因子边界**：trunk/actor/GAE/entropy/探索与其他超参与 base 完全相同；`--clip-vloss` 在
  `vf_nll` 下不适用（MSE 裁剪对 NLL 无定义），记为已知差异。
- 训练协议同 §3（500k 从零、同配方）；评估同 §5（3 deal seed × 400 副，bootstrap 4000）。

**判定（沿用 §4.1 三段式，k=7）**：主端点 `nll − base`：① H1 = z-CI 排除 0 且点 ≥ +10；
② 5 ≤ 点 < 10 = 不宣布 H1/H0；③ H0 = z-CI 上界 < +5。

**次要读数（descriptive）**：训练后对固定 obs/rollout 统计学到的 logvar 与实际 value 残差平方的
校准；EV / entropy / kl。

**A2 结果（2026-10-06 回填）**：`nll − base` k=7 = **−13.55** [−21.67,−5.43]（t-CI [−23.69,−3.41]，
6/7 seed 负）→ **H0，关闭不确定度 critic 方向**。机制：logvar 头确有学习（真实 obs 上均值 ≈−3.8、
跨决策 sd 0.68–1.03、无 clamp 塌缩），但均值头 EV 0.648 < base 0.709（NLL 的逆方差加权放大自信
样本残差）。把不确定度接下游（advantage 加权/搜索截断/集成）未测。读数见
[`experiments/depth-normalization-500k.md`](experiments/depth-normalization-500k.md) §4c。

## 12. Amendment A3（2026-10-06，operator 提议；不确定度采样价值 + 1M 预算）

**动机**：A2 的 `nll` 在 500k 为 −13.55 且 run 间波动大（sd 11 左右）。operator 提议：
（i）按 critic 的不确定度采样一个价值返回（`ṽ ~ N(mean, σ²)`）喂给 GAE/回报，看噪声目标
是否起正则/稳定作用；（ii）预算从 500k 提到 1M，看结论是否随预算改变。

**臂（3 × 7 seed = 21 run，全部 1M 从零、同配方）**

| arm | 结构 | seed | 角色 |
|---|---|---|---|
| base1m | shared，均值 critic（现状） | 1–7 | 对照 |
| nll1m | shared + `(mean, logvar)` NLL | 1–7 | 分离 NLL 因素（A2 的 1M 复测） |
| samp1m | nll1m + rollout 价值采样 `ṽ=mean+σ·ε` | 1–7 | 主臂 |

- 采样规格（写死）：rollout 取价值时 `ṽ = mean + ε·exp(logvar/2)`，`ε~N(0,1)` iid per
  (step, env)，logvar clamp ±10；采样只发生在 rollout 的 `get_action_and_value`（no_grad）；
  `get_value`/评估/搜索仍是 mean；GAE 的 deltas 与 `returns = adv + ṽ` 用采样值，
  价值 loss 的目标因此是采样值的 λ-return（本 amendment 的假设：噪声目标可能起正则作用）。
- 1M 步 = 976 update（999,424 步）；其余（lr/env/opponent/eval-interval 等）同 §3。
- 已知差异：`samp1m`/`nll1m` 的 value_loss 是 NLL 尺度（可为负）；`samp1m` 的 metrics
  `explained_variance` 基于采样值（更噪）；`--clip-vloss` 在 NLL 下不适用。

**判定（§4.1 三段式）**：主端点 `samp1m − base1m`（k=7）；次要（descriptive）
`samp1m − nll1m`（分离采样）、`nll1m − base1m`（A2@1M）。

**成本**：21 run × 1M ≈ 12–15 min/run；4 并发 ≈ 75–90 min + 21 条 h2h ≈ 15 min。

**A3 结果（2026-10-07 回填）**：`samp1m − base1m` k=7 = **−21.92** [−36.38,−7.46]（5/7 seed 负，
逐 seed sd 19.5，s6 −55.9）→ **H0，关闭采样价值方向**。`samp1m − nll1m` = −22.74 [−37.32,−8.17]
（采样本身有害）；`nll1m − base1m` = +2.51 [−5.66,+10.68]（A2 的 500k 亏损在 1M 未复现也未转正）。
机制：logvar 头学到 σ≈0.15–0.19，噪声注入每个 GAE delta 与 λ-return 目标是主因。读数见
[`experiments/depth-normalization-500k.md`](experiments/depth-normalization-500k.md) §4d。

## 13. Amendment A4（2026-10-07，operator 选定「改进 A」；均值/方差解耦）

**动机**：A2 的亏损机制被定位为均值项的逆方差加权（`∂L/∂μ=(μ−y)/σ²`，训练后 σ≈0.15 使均值
梯度被放大一个量级）。A4 把均值路径恢复成与 base 完全相同的 clipped MSE，方差头改用
`mean.detach()` 的 Gaussian NLL 单独训练——单因子只去掉逆方差加权。

**臂（1 × 7 seed = 7 run，500k，对齐 A2 预算）**

| arm | 结构 | seed | 角色 |
|---|---|---|---|
| dec | `--vf-nll true --vf-decoupled true`（critic 2 宽，55,308 参数） | 1–7 | 主臂 |

- 损失（写死）：`v_loss = 0.5·max(MSE, clipped MSE)(μ, y) + 0.5·(logvar + (y−μ.detach())²·e^{−logvar})`，
  整体仍乘 `vf_coef=0.5`；logvar clamp ±10；`clip_vloss` 对均值项生效（与 base 相同）。
- 其余（trunk/actor/GAE/超参/评估）同 §3/§5。
- 已知差异：critic 头 +129 参数；方差项经共享 trunk 仍对表示有梯度（这是要测的辅助目标效应）。

**判定（§4.1 三段式）**：主端点 `dec − base`（k=7）；次要 descriptive `dec − nll`（对照 A2，分离解耦）。

**预期**：若 A2 的亏损完全来自逆方差加权，`dec` 的 EV 应回到 base 附近（≈0.71）、h2h 落在 null；
若仍为负，则「方差辅助目标 + trunk 共享」本身有害，高斯 NLL 线收口。

**成本**：7 run × 500k ≈ 15 min；14 条 h2h ≈ 6 min。

**A4 结果（2026-10-07 回填）**：`dec − base` k=7 = **−1.10** [−10.94,+8.73]（5/7 seed 正，但 z-CI 跨 0、
上界 +8.73 ≥ +5）——三档均未触发：不关闭也不通告增益；`dec − nll` = **+9.34** [−4.19,+22.88]，
确证 A2 亏损来自逆方差加权，解耦后方差辅助目标对策略净零（EV 0.655 vs base 0.709，logvar 校准
仍学到 σ≈0.17）。高斯 NLL 线收口。读数见
[`experiments/depth-normalization-500k.md`](experiments/depth-normalization-500k.md) §4e。

## 14. Amendment A5（2026-10-07，operator 提议；O2 臂续训对标 w2m_ctl）

**动机**：§3.1 的锚点显示 O2 四臂（500k 从零）比平台冠军 `w2m_ctl` 低 24–39 Elo；而 `w2m_ctl`
本身是 2M 续训产物。operator 提议把 `deep`/`deep_ln` 续训后再和最强模型 h2h。

**臂（3 × 3 seed = 9 run；续训 +1,000,000 步）**

| arm | 起点（500k） | 续训协议 | seed |
|---|---|---|---|
| o2c_base | `o2_base__<s>` | T23 协议（below） | 1–3 |
| o2c_deep | `o2_deep__<s>` | 同左 | 1–3 |
| o2c_deep_ln | `o2_deep_ln__<s>` | 同左 | 1–3 |

**续训协议（复刻 `warmstart-adamw-1m.md` §5 的 T23 流程）**：
`--load-checkpoint <起点 agent.pt>`（同 arch 整表加载）、`--total-timesteps 1000000`、
`--optimizer adamw --weight-decay 0.01 --lr-schedule cosine`（峰值 2.5e-4）、
`--num-envs 16 --num-minibatches 8`（batch 2048）、`--opponent pool --pool-episode True`
（6 个 t17 成员，同 T23）、`--snapshot-interval 0`（只用 final `agent.pt`，禁止按快照择优）。
**配方修订（2026-10-07，[ADR-0015](adr/0015-continuation-pool-random-mix.md)，operator 决策）**：
后续新的续训默认在该池上混入 10% RandomBot（6 个成员 `3@` + `2@random`）；
A5/A5b/A6 的产物按各自当时配方保留，不回填。

**端点**：
1. **主端点**：各续训臂 − `w2m_ctl`（k=3，3 deal seed × 400 副，bootstrap 4000），三段式
   （H1 = CI 排除 0 且点 ≥ +10；H0 = CI 上界 < +5）。
2. **次要（descriptive）**：`X_c − X_500k`（续训相对自身起点的增益）、
   `o2c_deep/deep_ln − o2c_base`（同协议下的 arch 差）、与 §3.1 锚点的差。

**预期**：T23 经验为 +8 左右的小正增益。若三个续训臂仍低于 `w2m_ctl` >10 Elo，则「续训追平」
路径关闭；若 `deep_c/deep_ln_c` 与 `base_c` 无差，则 O2 的 arch 差在续训下同样不显著。

**成本**：9 run × 1M（batch 2048）≈ 35–45 min（4 并发）；18 条 h2h ≈ 12 min。

**A5 结果（k=3 回填）**：`o2c_deep_ln − w2m_ctl` = **+15.41** [8.88,+21.94]（t-CI [+1.07,+29.75]，
逐 seed 11.2/14.6/20.4）→ 满足 H1（+10 门槛）；`o2c_deep − w2m_ctl` = +8.74 [+1.49,+15.99]（小正增益）；
`o2c_base − w2m_ctl` = +6.57 [−0.11,+13.24]（z-CI 跨 0）。三臂相对各自 500k 起点 +29.2…+29.8
（CI 均排除 0）。读数见 [`experiments/depth-normalization-500k.md`](experiments/depth-normalization-500k.md) §3.2。

**A5b（2026-10-07，k=3 读数之后决定；复核性预注册）**：k=3 主端点 `o2c_deep_ln − w2m_ctl` =
+15.41 [8.88,+21.94] 满足 H1。按 T23 的先例（k=3 +11.48 → k=7 +8.20）补 seed 4–7：
- `o2c_base`、`o2c_deep_ln` 续训 seed 4–7（各 4 run，协议同 A5，起点为对应 500k seed 4–7）；
- 主端点升为 k=7：`o2c_deep_ln − w2m_ctl`、`o2c_base − w2m_ctl`；新增直接配对
  `o2c_deep_ln − o2c_base`（seed 1–7，臂间比较，recipe-controlled）；
- `o2c_deep` 无 500k seed 4–7，保持 k=3（descriptive）；
- 判定沿用 §4.1 三段式；无论结果如何都报告。

**A5b 结果（2026-10-07 回填）**：k=7：`o2c_base − w2m_ctl` = **+10.12** [5.49,+14.76]（t-CI [4.33,+15.91]）、
`o2c_deep_ln − w2m_ctl` = **+9.27** [2.77,+15.77]（t-CI [1.16,+17.39]）、
直接配对 `o2c_deep_ln − o2c_base` = **+1.01** [−4.97,+7.00]（null）；`o2c_deep − w2m_ctl` 保持 k=3 +8.74。
结论：续训把三臂抬到旧冠军之上 ~9–10 Elo；**架构（depth/LN）无贡献**；k=3 的 deep_ln +15.41
高估被撤回（T23 式回归）。读数见
[`experiments/depth-normalization-500k.md`](experiments/depth-normalization-500k.md) §3.2。

## 15. O2 线过程时间线（2026-10-06 ~ 10-07）

结果与产物逐步回填在 [`experiments/depth-normalization-500k.md`](experiments/depth-normalization-500k.md)：
主实验（§3）→ A1 残差（§4b）→ A2 NLL（§4c）→ A3 采样+1M（§4d）→ A4 解耦（§4e）→
人类语料分析 + 搜索候选来源修复（`runs/human-eval/`）→ 锚点（§3.1）→ A5/A5b 续训（§3.2）→
集群/非传递性/熵/架构探针（§3.3/§3.4）。最终口径：

- 架构轴（depth/LN/残差/NLL 变体）在 500k 与续训下都无增益；
- 训练配方/预算是主要变量：T23 续训把三臂抬到旧冠军之上 ~9–10 Elo（集群内统计平局）；
- 当前最强类是 55,179 参数的 memoryless MLP（`shared`，obs v5）；搜索（推理期）仍是最大杠杆。

## 16. Amendment A6（2026-10-07，operator 请求；残差 × 续训交叉）

**动机**：A5/A5b 证明 T23 续训是当前最强杠杆（三臂相对其 500k 起点 +29 Elo；`o2c_base`/
`o2c_deep_ln` 对 `w2m_ctl` +10.12/+9.27，k=7）。A1 的残差 screen 只覆盖 500k 从零（k=3、
未升级）。「残差 skip 在续训轨迹上是否改变终点（相对 plain twin）」是 A1×A5 未测的格子；
operator 指定续训 `deep_lnres` 并与续训后的 `deep_ln` 直接比较。

**臂（6 run；续训 +1M，协议与 A5 逐位相同）**

| arm | 起点（500k） | `--arch` | 总参数 | seed | 角色 |
|---|---|---|---:|---|---|
| o2c_deep_lnres | `o2_deep_lnres__<s>` | `deep_lnres` | 72,459 | 1–3 | **主臂** |
| o2c_deep_res | `o2_deep_res__<s>` | `deep_res` | 71,691 | 1–3 | 次要（A1 deep 对的镜像） |

对照（已存在，不重训）：`o2c_deep` seed 1–3、`o2c_deep_ln` seed 1–7、`o2c_base` seed 1–7。
续训协议 = A5（`--load-checkpoint` 同 arch 整表加载、`--total-timesteps 1000000`、
`--optimizer adamw --weight-decay 0.01 --lr-schedule cosine` 峰值 2.5e-4、
16 env × 128 × minibatch 8（batch 2048）、`--opponent pool --pool-episode True` 6 成员、
`--snapshot-interval 0` 只用 final `agent.pt`）。脚本 `runs/o2-depth/run_a6_res_cont.sh`。

**端点**

1. **主端点（screen，k=3）**：`o2c_deep_lnres − o2c_deep_ln`，训练 seed 1–3 配对，
   3 deal seed（9600/9601/9602）× 400 副 × 换座、bootstrap 4000。
2. **次要（descriptive，k=3）**：`o2c_deep_res − o2c_deep`、`o2c_deep_lnres − o2c_deep_res`、
   `o2c_deep_lnres − o2c_base`、`o2c_deep_res − o2c_base`；续训杠杆 sanity
   `o2c_deep_lnres − o2_deep_lnres`、`o2c_deep_res − o2_deep_res`；绝对锚 `o2c_deep_lnres − w2m_ctl`。

**判定（§4.1 三段式，screen 预算 k=3）**

- H1 = z-CI 排除 0 **且**点估计 ≥ +10；`5 ≤ 点 < 10` 不宣布 H1/H0；H0 = z-CI 上界 < +5。
- **升级到 k=7（唯一条件）**：任一残差−plain 对（`deep_lnres−deep_ln` 或 `deep_res−deep`）
  满足 H1 → 补该对两侧 seed 4–7 后复核（协议不变）：
  - `deep_lnres` 对：`o2_deep_lnres` 500k seed 4–7 从零（主实验配方）+ 同协议续训；
    `o2_deep_ln` / `o2c_deep_ln` seed 4–7 已存在 → **+8 run**；
  - `deep_res` 对：`o2_deep`、`o2_deep_res` 的 500k seed 4–7 从零 + 两者续训 → **+16 run**。
- 否则 A6 结束：残差方向在续训下同样记录为无增益（与 A1 的 500k 结论合并），
  架构轴维持关闭。

**已预料的读法风险**：k=3 小样本高估（A5 先例：+15.41 → +9.27）；残差臂 500k 起点只有
seed 1–3，升级不得用其它种子/快照拼；起点 `run_a1.sh` 与主批 flags 逐位相同（同一配方）；
评估 bank 与全部 O2 评估相同（9600–9602），只在本线内可比。

**成本**：6 run × 1M（batch 2048）≈ 40–45 min（4 GPU 槽上限）；h2h 24 条 ≈ 15–20 min CPU。
升级成本（若触发）：`deep_lnres` 对 +8 run（4×500k + 4×1M）；`deep_res` 对 +16 run（8×500k + 8×1M）。

**A6 结果（2026-10-07 回填）**

- **主端点**：`o2c_deep_lnres − o2c_deep_ln` k=3 = **−2.08** [z −8.71,+4.56；t −16.64,+12.48]
  （se 3.38、seed sd 2.88；逐 seed +0.43/−1.45/−5.22，2/3 负）。z 上界 **+4.56 < +5** →
  按 §4.1 第 3 条 **H0：残差在续训下同样无增益**（边际：上界距 +5 仅 0.44 Elo；点估计为负，
  不存在正信号）；升级未触发，无 seed 4–7。
- **次要（descriptive）**：`o2c_deep_res − o2c_deep` = +2.13 [−9.61,+13.86]（上界 ≥ +5、
  点 < +10 → **不可判定**，不宣布 H1/H0，升级门亦未触发）；`o2c_deep_lnres − o2c_deep_res`
  +2.90 [−6.92,+12.72]（残差在 LN 下同样 null）；`o2c_deep_lnres − o2c_base` +3.33
  [−6.12,+12.78]、`o2c_deep_res − o2c_base` −0.15 [−13.92,+13.63]（同等续训下不优于 base）。
- **续训杠杆复现**：`o2c_deep_lnres − o2_deep_lnres` = **+30.33** [14.60,+46.05]、
  `o2c_deep_res − o2_deep_res` = **+20.85** [3.28,+38.42] → 残差臂同样吃到 +21…+30 的续训增益。
- **绝对锚（descriptive）**：`o2c_deep_lnres − w2m_ctl` k=3 = +10.44 [2.11,+18.76]，与 plain 臂
  （+8.7…+10.1）同一档；k=3 不升级（升级门只挂在残差−plain 对），按 T23/A5 先例不当 H1 读数。
- **机制**：6 run 均 488/488 update、无 NaN；s1–3 均值 `deep_lnres` EV 0.649 / entropy 1.446 /
  maxKL 0.0175（`deep_ln` 0.648 / 1.506 / 0.0053；`base` 0.664 / 1.490 / 0.0102）。
- **结论**：T23 续训把残差臂也抬到旧冠军附近，但 skip 相对 plain twin 的增量为零/微负
  （主端点 H0，deep_res 对不可判定）；A5b「续训有效、架构无贡献」的结论扩展到残差变体，
  trunk 架构轴维持关闭。读数见
  [`experiments/depth-normalization-500k.md`](experiments/depth-normalization-500k.md) §3.5。

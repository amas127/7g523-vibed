# O2：trunk 深度 + 归一化（500k，2×2 因子 + 参数量对照）

> **状态：完成（2026-10-06）。判定：H0 —— 按预注册止损关闭 trunk 深度/归一化轴**
> （主端点合并 z-CI 上界 `−2.06 < +5`）。**Amendment A1 残差补测见 §4b：残差变体同样无增益，轴维持关闭。**
> **Amendment A2（不确定度 critic）见 §4c：`nll − base` k=7 −13.55 [−21.67,−5.43] → 同样关闭。**
> **Amendment A3（按不确定度采样价值 + 1M）见 §4d：`samp1m − base1m` k=7 −21.92 [−36.38,−7.46] → 同样关闭。**
> **Amendment A4（均值/方差解耦）见 §4e：`dec − base` k=7 −1.10 [−10.94,+8.73]（null，三档均未触发）；`dec − nll` +9.34 → A2 的亏损机制确证为逆方差加权。**
> **Amendment A5（续训对标）见 §3.2：A5b k=7 复核后 `o2c_base − w2m_ctl` = +10.12 [5.49,+14.76]、`o2c_deep_ln` +9.27 [2.77,+15.77]（均略超旧冠军），但 `o2c_deep_ln − o2c_base` = +1.01（null）→ 续训有效、架构无贡献；k=3 的 deep_ln +15.4 系小样本高估。**
> **Amendment A6（残差 × 续训）见 §3.5：`o2c_deep_lnres − o2c_deep_ln` k=3 = −2.08 [−8.71,+4.56]（z 上界 +4.56 < +5 → H0）；`o2c_deep_res − o2c_deep` +2.13（不可判定）→ 残差在续训下同样无增益，架构轴维持关闭。**
> **口径**：revision-3（`rules_id=2e36dbea44893696`）、obs v5（2 家 161 维）、
> PPO memoryless MLP、500k 从零、与 D-A/D-D 同配方（只改 `--arch`/`--hidden-size`）。
> **预注册**：[`../depth-normalization-plan.md`](../depth-normalization-plan.md)（r1，红队审查后冻结；审查见
> [`../depth-normalization-review.md`](../depth-normalization-review.md)）。
> **原始产物**：`runs/o2-depth/`（gitignored）；复算脚本 `runs/o2-depth/recompute_o2.py`。
> **代码**：`src/seven523/networks.py:70,351-366`（新 arch）、`src/seven523/train.py:201`（choices）、
> `src/seven523/networks.py:680-692`（跨 arch warm-start fail-loud）、`tests/test_networks.py:414-497`。

## 0. 一句话结论

**加一层 128 宽 trunk（`deep`）或加 LayerNorm（`ln`/`deep_ln`）都没有正效应，反而偏负：**
主端点 `deep_ln − base`（k=7）**−13.95** [−25.84,−2.06]；归因读数同向（`deep−base` −9.52，
CI 排除 0）。参数匹配的宽度对照 `wide−base` ≈ 0，说明这不是容量混淆、而是加深主干本身的负结果。
深度/归一化轴关闭；训练侧结构项清空。**残差补测（A1）**：`deep_res−deep` +6.82
[−4.62,+18.26]（对 plain deep 有正向读数、但未达 +10 门槛，CI 跨 0）、
`deep_lnres−deep_ln` −4.69 [−15.00,+5.62]；`deep_res−base` 仍 −6.18 [−13.78,+1.42]，
没有证据表明残差变体转正 → **不改变关闭判定**。
**不确定度 critic（A2）**：`nll − base` k=7 −13.55 [−21.67,−5.43]（6/7 seed 负）；虽然 logvar 头
确实学到了异方差（logvar 均值 ≈ −3.8、跨决策 sd ≈ 0.85、无 clamp 塌缩），但均值头 EV 掉到
0.648（base 0.709）→ **显式建模噪声也没有帮助，关闭该方向**。
**续训反转与回落（A5/A5b）**：500k 从零的关闭结论**不延伸到续训**——T23 协议续训 +1M 后
三臂相对自身 +29 Elo，`o2c_base`/`o2c_deep_ln` 对 `w2m_ctl` 为 +10.12/+9.27（k=7，CI 排除 0）；
但 `o2c_deep_ln − o2c_base` = +1.01 [−4.97,+7.00]（null）→ **架构无贡献，k=3 的 deep_ln +15.4
高估被撤回**。
**残差 × 续训（A6）**：两个 A1 残差臂各按同款 T23 协议续训 +1M 后，与 plain twin 的续训终点
直接比较：主端点 `o2c_deep_lnres − o2c_deep_ln` k=3 = **−2.08** [−8.71,+4.56]（z 上界 +4.56 < +5
→ 按预注册 H0，边际 0.44 Elo）；`o2c_deep_res − o2c_deep` +2.13 [−9.61,+13.86]（上界 ≥ +5，
不可判定、未达升级门）。残差臂自身同样吃到续训增益（vs 自身 500k +30.3/+20.9）→
**续训有效、skip 增量 ≈ 零**，A5b「架构无贡献」的结论扩展到残差变体。

## 1. 设计摘要（预注册，详见 plan §1/§4）

| arm | `--arch` | trunk | 总参数 | 训练 seed | 角色 |
|---|---|---|---:|---|---|
| base | `shared` | 2×Linear(128) | 55,179 | 1–7 | 主端点对照 |
| deep_ln | `deep_ln` | 3×Linear(128)+LN | 72,459 | 1–7 | **主端点** |
| ln | `ln` | 2×Linear(128)+LN | 55,691 | 1–3 | 归因（descriptive） |
| deep | `deep` | 3×Linear(128) | 71,691 | 1–3 | 归因（descriptive） |
| wide | `shared --hidden-size 157` | 2×Linear(157) | 72,202 | 1–3 | 容量对照（descriptive） |

- LN 放置（写死）：每个 hidden Linear 之后、激活之前，含末层（heads 输入侧）。
- 主分析：`deep_ln − base` 按训练 seed 配对，k=7；每训练 seed 3 deal seed × 400 副（1,200 副）。
- 判定（plan §4.1）：① H1 = z-CI 排除 0 且点 ≥ +10；② 5 ≤ 点 < 10 = 无 ≥ +10 证据、不宣布 H1/H0；
  ③ H0 = z-CI 上界 < +5。t-CI 只作稳健性读数，不作 gate。

## 2. 执行记录

- **训练**：23 run ×（488 update × 1024 步 = **499,712 步**），全部从零、同一配方
  （`lr 2.5e-4` linear 退火、`num-envs 8`、`num-steps 128`、`minibatches 4`、`epochs 4`、
  `adam`、`opponent random`、`reward-shaping terminal`）。`--eval-interval 50`（9 个 vs-random eval/run）。
  - 调度：15:05 前 4 个 run 串行；之后 4 流并行（`run_limited.sh gpu`，fleet 4 GPU 槽），
    15:47:46 全部完成。单 run 墙钟 6–8 分钟（串行 sps ≈1420–1500，4 路并行 ≈1070/proc）。
  - 完整性：23/23 `args.json` 的 arch/hidden 与设计一致；23 × 488 行 metrics 无 NaN；
    base 臂 vs-random sanity 在 update 200/450（≈205k/461k 步）7-seed 均值 **0.8914 / 0.9000**，
    落在预注册带 0.88–0.90 ± 0.03 内 → 管线正常。
- **评估**：19 条 h2h ×（3 deal seed × 400 副 × 换座 2 局 = **2,400 局**），
  bootstrap 4000，deal seeds **9600/9601/9602**；3 路并发，22:35:55–22:40:16。
- **测试与冒烟**：全套 855 passed / 14 skipped（收集 869 = 原 859 + 新增 10）；
  默认 `shared`/`towers` 路径与 HEAD 旧代码同种子 state_dict **逐位一致**；5 臂 × 2048 步无 NaN。
- **复算**：`runs/o2-depth/recompute_o2.py`。注意：plan §4.3 的 `flat()` 是按 D-A 包装过的
  envelope 写的；本轮 h2h 由 `tools/head_to_head.py --json` 直接落盘（`combined` 下一层即指标），
  故适配器取 `d["combined"]`。**统计约定未变**：k = 训练 seed 数、`combine_duel_seeds`、
  z=1.96 gate（C-1）。

## 3. 主端点结果

`deep_ln − base`，训练 seed 1–7 配对，**k=7**，每 seed 1,200 副 / 2,400 局：

| 读数 | 值 |
|---|---:|
| 合并点估计 | **−13.95 Elo** |
| z-CI（gate） | **[−25.84, −2.06]** |
| t-CI（稳健性） | [−28.79, +0.90] |
| se / between-seed sd | 6.07 / 16.05 |

逐 seed（point / winrate）：s1 −18.27 (0.474)、s2 +0.00 (0.500)、s3 +11.01 (0.516)、
s4 −11.20 (0.484)、s5 −37.70 (0.446)、s6 −25.53 (0.463)、s7 −15.94 (0.477) —— **5/7 为负**。

**判定（plan §4.1 逐条）**：① 不满足（负）；② 不满足（点 < +5）；③ **满足：z-CI 上界 −2.06 < +5
→ 在 95% 下排除 ≥ +5，按预注册止损关闭该轴。** 结论不是「没有发现」，而是「该配方/预算下
加深或加 LN 不优于现状，且证据足以排除 ≥ +5 的增益」。

### 3.1 绝对锚点：O2 臂 vs 平台冠军 w2m_ctl（2026-10-07，context 测量）

主端点是批内相对量（vs base）。operator 在 web 试玩后反馈「比 w2m_ctl 差点意思」，这里补测
四臂 seed 1–3 对 `w2m_ctl` 的配对 h2h（同协议，3 deal seed × 400 副，k=3）：

| 比较 | 点估计 | z-CI | winrate |
|---|---:|---|---:|
| `o2_base − w2m_ctl` | −23.92 | [−42.52, −5.33] | 0.466 |
| `o2_ln − w2m_ctl` | −38.94 | [−56.83, −21.05] | 0.444 |
| `o2_deep − w2m_ctl` | −25.94 | [−40.90, −10.99] | 0.463 |
| `o2_deep_ln − w2m_ctl` | −28.60 | [−49.93, −7.27] | 0.459 |

读法：四臂全部显著低于 `w2m_ctl`（−24…−39 Elo，CI 均排除 0）。**这是训练预算/配方的差，
不是 O2 结构结论**：O2 是 500k 从零（D-A 同款配方），`w2m_ctl` 是 2M 续训 +
AdamW/cosine/pool 的池冠军（μ≈187.46）；同一批 O2 臂之间的相对排序（`ln` 最低、`base` 最高）
与 §4 的归因一致。玩感与人类侧一致：raw `o2_deep` 11 局 8 胜 3 负、raw `o2_deep_ln`
4 局 3 胜 1 负（含 1 平），见 `traces/web/run-20261007-031005` 与 `run-20261007-032521`。

### 3.2 Amendment A5：500k 臂续训（T23 协议 +1M）对标 w2m_ctl（2026-10-07）

**设计**（plan §14/A5b）：`o2_base/o2_deep/o2_deep_ln` seed 1–3（base/deep_ln 补到 7）从各自
500k `agent.pt` 续训 +1M（warm-start 整表加载、AdamW wd0.01/cosine、6 成员 pool、batch 2048，
即 T23 协议；`--snapshot-interval 0`，只用 final）。

**结果**：

| 比较 | k | 点估计 | z-CI | t-CI |
|---|---:|---:|---|---|
| `o2c_base − w2m_ctl` | 7 | **+10.12** | [+5.49, +14.76] | [+4.33, +15.91] |
| `o2c_deep_ln − w2m_ctl` | 7 | **+9.27** | [+2.77, +15.77] | [+1.16, +17.39] |
| `o2c_deep − w2m_ctl` | 3 | +8.74 | [+1.49, +15.99] | [−7.18, +24.66] |
| `o2c_deep_ln − o2c_base` | 7 | **+1.01** | [−4.97, +7.00] | [−6.45, +8.48] |
| `o2c_base − o2_base` | 3 | +29.19 | [+11.79, +46.59] | [−9.01, +67.38] |
| `o2c_deep − o2_deep` | 3 | +29.82 | [+19.04, +40.61] | [+6.14, +53.50] |
| `o2c_deep_ln − o2_deep_ln` | 3 | +29.83 | [+18.74, +40.92] | [+5.48, +54.18] |

逐 seed vs w2m：base 0.6 / 11.6 / 7.5 / 19.3 / 11.7 / 5.7 / 14.5；
deep_ln 11.2 / 14.6 / 20.4 / 15.4 / 5.9 / 2.8 / −5.4。

**判定与读法**：
1. **续训有效**：三臂相对各自 500k 起点 +29 Elo（CI 排除 0）——O2 的 500k 从零模型是欠训练
   （warm-start/预算/池对手/优化器的组合），这是它们此前比 `w2m_ctl` 低 24–39 Elo 的主因。
2. **续训后略超旧冠军**：`o2c_base − w2m_ctl` = +10.12（k=7；点 ≥ +10 且 z/t CI 排除 0，满足 H1）；
   `o2c_deep_ln` +9.27（CI 排除 0、点略低于 +10）；`o2c_deep` +8.74（k=3）。
3. **架构没有贡献**：k=3 时 `deep_ln` 的 +15.41 是**小样本高估**——补到 k=7 回落到 +9.27，
   且直接配对 `o2c_deep_ln − o2c_base` = **+1.01 [−4.97,+7.00]**（null，k=7，又是 T23 式回归）。
   500k 从零的「深度/归一化无增益」结论在续训下同样成立；「deep_ln 反超旧冠军」的 k=3 说法撤回。
4. 净结论：**T23 续训协议把这三个 500k 模型抬到旧冠军之上 ~9–10 Elo；架构因素（depth/LN）
   在同等续训下不产生差异**。机制（seed 1–7/1–3）：EV 0.64–0.67、entropy 1.30–1.43、
   kl_max ≤0.019、无 NaN，vs-random 0.79–0.95。

### 3.3 A5 后验探针：集群平局与配对特异性（2026-10-07）

`o2c_base` 对 `w2m_ctl` 的 +10 需要回答「是新王还是配对特异」。补两个探针：

1. **独立发牌 bank（150–152，k=7）**：`o2c_base − w2m_ctl` = **+11.91** [5.91, +17.91]，
   逐 seed 3.2/1.3/8.6/16.8/24.4/14.1/15.1 → 换 bank 复现，不是 9600–9602 的偶然。
2. **对池内集群（9600–9602，k=3）**：`o2c_base` vs `w2m_plain` −0.82 [−7.6,+6.0]、
   `w2m_low` −0.97 [−8.5,+6.6]、`ws_s2` +4.39 [−3.3,+12.1]、`pself_s2` +12.28 [+2.2,+22.4]；
   而 `w2m_ctl` vs `w2m_plain` −3.77、vs `w2m_low` −2.17、vs `pself_s2` +6.38。

读法：**w2m 集群（ctl/plain/low/ws）与 `o2c_base` 在统计上是一个平局集群**；
`o2c_base − w2m_ctl` 的 +11 是本对局特有的（直接配对可复现），但经 `w2m_plain` 桥接只有 ~+4，
即存在 ~7 Elo 的非传递性。机制上最明显的差异是 **entropy**：w2m 血统每多一次固定池 anneal
就担一次（l1_2M 1.566 → t23wswd 1.308 → w2m_ctl **1.082**；而 random-对手的 `w2m_plain` 1.527），
`o2c_base` 只做一次池 anneal、停在 1.433。所以「更小成本超过」最可能的解释是 **它没做第三次
固定池续训**（避免过度特化），而不是模型本身全面更强；也存在 w2m_ctl 单 seed（seed 11）
抽样因素。要定论需集群级联合拟合或补第三次 anneal 的单因子实验（未排期）。

### 3.4 架构与熵：补充读法（2026-10-07）

**当前架构**：全部相关 checkpoint（`w2m_ctl`、`o2c_base`、搜索的 value 核 `t_leafq`）都是同一个
memoryless MLP：`obs v5 (161) → Linear(161,128)+ReLU → Linear(128,128)+ReLU`，actor 头
`Linear(128,138)` 拆 134 模板 + 4 花色，critic 头 `Linear(128,1)`；**55,179 参数**、
`seq_len=event_len=0`（无历史/事件编码器）。唯一例外 `o2c_deep_ln`（3 层 + LN，72,459）与
`o2c_base` 在 k=7 直接配对上打平（+1.01 [−4.97,+7.00]）→ **当前强度来自训练配方，不是结构**。
搜索 rung（μ≈260）用的也是同一个 55,179 的 MLP，力量来自推理期 O4-lite（t=5/K=32/C=6）。

**熵不是好坏指标**：本线数据没有 entropy→strength 的单调关系——最尖的 `w2m_ctl`（1.08）与最平的
`w2m_plain`（1.53）在池拟合并不可分辨，`o2c_base`（1.43）与两者都平局；O2 从零批里 `deep`（1.65）
和 `deep_ln`（1.93）都弱于 base（1.76）。所以 §3.3 的熵轨迹只支持「反复固定池 anneal 不再换到
h2h 增益、只在锐化」这一诊断，**不能推出「锐化=变差」**；真正的风险是对未见对手（人类类分布）
的鲁棒性，需要单独测。要坐实「第三次 anneal 无收益/负收益」，需做第二次 anneal 的单因子实验
（未排期）。

**训练血统成本对比**：`w2m_ctl` = t17long 从零 1M（batch 1024）+ t23wswd +1M + w2m_ctl +2M
（batch 2048，均 AdamW/cosine/6 人池）= **4M 环境步、约 2,440 次更新**；`o2c_base` = 500k 从零
（batch 1024，random）+ 1M 同款池续训 = **1.5M 环境步、976 次更新**。两者当前在集群里统计平局，
`o2c_base` 对 `w2m_ctl` 的直接配对 +10~+12 是本对局特异性（非传递性 ~7 Elo）。

### 3.5 Amendment A6：残差 × 续训（2026-10-07，k=3 screen）

**设计**（plan §16，operator 请求，开跑前写死）：A5/A5b 证明 T23 续训是当前最强杠杆后，把 A1
的两个残差臂从各自 500k 起点按**同款 A5 协议**续训 +1M（`o2c_deep_res`/`o2c_deep_lnres`，
seed 1–3），与已续训的 plain twin 直接配对——补上 A1×A5 交叉格。对照不复训：
`o2c_deep` s1–3、`o2c_deep_ln` s1–7、`o2c_base` s1–7。

**结果**（每比较 k=3，9600–9602 × 400 副换座，bootstrap 4000）：

| 比较 | k | 点估计 | z-CI | t-CI | se / sd |
|---|---:|---:|---|---|---|
| `o2c_deep_lnres − o2c_deep_ln`（主） | 3 | **−2.08** | [−8.71, +4.56] | [−16.64, +12.48] | 3.38 / 2.88 |
| `o2c_deep_res − o2c_deep` | 3 | +2.13 | [−9.61, +13.86] | [−23.63, +27.89] | 5.99 / 10.37 |
| `o2c_deep_lnres − o2c_deep_res` | 3 | +2.90 | [−6.92, +12.72] | [−18.66, +24.46] | 5.01 / 8.42 |
| `o2c_deep_lnres − o2c_base` | 3 | +3.33 | [−6.12, +12.78] | [−17.42, +24.08] | 4.82 / 8.35 |
| `o2c_deep_res − o2c_base` | 3 | −0.15 | [−13.92, +13.63] | [−30.39, +30.10] | 7.03 / 12.18 |
| `o2c_deep_lnres − o2_deep_lnres` | 3 | **+30.33** | [+14.60, +46.05] | [−4.19, +64.84] | 8.02 / 13.89 |
| `o2c_deep_res − o2_deep_res` | 3 | **+20.85** | [+3.28, +38.42] | [−17.72, +59.42] | 8.96 / 15.53 |
| `o2c_deep_lnres − w2m_ctl`（锚） | 3 | +10.44 | [+2.11, +18.76] | [−7.84, +28.71] | 4.25 / 4.34 |

逐 seed（主端点）：+0.43 / −1.45 / −5.22（2/3 负）；`deep_res−deep`：−8.40 / +12.33 / +2.46。

**判定（plan §16 升级规则）**：主端点 H0 = z-CI 上界 < +5 → **+4.56 < +5 成立**，
按预注册关闭「残差 × 续训」格子（边际 0.44 Elo；点估计为负，不存在正信号，故不触发任何升级）。
`o2c_deep_res − o2c_deep` 上界 +13.86 ≥ +5、点 < +10 → **不可判定**（不宣布 H1/H0），
升级门（点 ≥ +10 且 z-CI 排 0）未触发 → A6 结束，不排 seed 4–7。

**读法**：① 续训杠杆在残差臂上复现（vs 自身 500k +30.3 / +20.9，CI 排除 0），说明增益来自
预算/配方而非结构；② skip 相对 plain twin 无增量（主端点微负；deep 对不可判定，两者都无正信号）；
③ 同等续训下 `deep_lnres` 与 `deep_ln`/`base` 的 CI 全跨 0，A5b「架构无贡献」结论扩展到
残差变体；④ 锚读数 +10.44（k=3）与 plain 臂 +8.7…+10.1 同档，按 T23/A5 先例
（k=3 高估）不作强解读。

**机制读数**（s1–3 均值）：`deep_lnres` EV 0.649 / entropy 1.446 / maxKL 0.0175 / vs-random 末值 0.860；
`deep_ln` 0.648 / 1.506 / 0.0053 / 0.917；`deep_res` 0.655 / 1.374 / 0.0134 / 0.867；
`deep` 0.642 / 1.360 / 0.0193 / 0.893；`base` 0.664 / 1.490 / 0.0102 / 0.873。无 NaN、
488/488 update；skip 不改变收敛尺度。

## 4. 归因读数（descriptive，按 plan §4.5 不得升级为行动论断、不进轴级判定）

| 比较 | k | 点估计 | z-CI | t-CI | se / sd |
|---|---:|---:|---|---|---|
| `ln − base` | 3 | −17.91 | [−41.23, +5.42] | [−69.11, +33.30] | 11.90 / 20.61 |
| `deep − base` | 3 | **−9.52** | **[−18.17, −0.86]** | [−28.52, +9.48] | 4.42 / 3.71 |
| `deep_ln − deep` | 3 | +1.54 | [−11.86, +14.95] | [−27.88, +30.97] | 6.84 / 11.85 |
| `wide − base` | 3 | +0.97 | [−16.65, +18.59] | [−37.72, +39.65] | 8.99 / 15.57 |

读法：深度单独（`deep−base`）为负且 CI 排除 0；LN 单独为负但 k=3 下 CI 跨 0；
LN 叠在深网上约等于零（`deep_ln−deep`），即归一化没有补偿深度的损失；
容量对照 `wide−base` 为 null，说明「参数更多」本身既不帮忙也不背锅——负结果来自加深的
trunk 本身（优化/表达），不是 +16.5k 参数。全部比较只在本轮同一批 run 内有效。

## 4b. Amendment A1：残差补测（exploratory，k=3）

主端点出结果后，operator 指出「plain 加深退化的标准解法是残差连接」，而 r1 把残差列为 §7
条件项、未预注册。A1 在开跑前写死端点与升级规则（plan §10）：`deep_res` = `deep` + 第 2/3 块
skip、`deep_lnres` = `deep_ln` + 同样 skip，参数与各自 plain 对照逐位相同
（71,691 / 72,459，单测钉住），唯一差异 = skip；训练/评估协议与主实验相同。

| 比较 | k | 点估计 | z-CI | t-CI | se / sd |
|---|---:|---:|---|---|---|
| `deep_res − deep` | 3 | **+6.82** | [−4.62, +18.26] | [−18.29, +31.94] | 5.84 / 9.47 |
| `deep_lnres − deep_ln` | 3 | −4.69 | [−15.00, +5.62] | [−27.32, +17.95] | 5.26 / 5.56 |
| `deep_res − base` | 3 | −6.18 | [−13.78, +1.42] | [−22.87, +10.51] | 3.88 / 6.57 |
| `deep_lnres − base` | 3 | −6.34 | [−25.05, +12.36] | [−47.41, +34.72] | 9.54 / 16.53 |

逐 seed（point）：`deep_res−deep` +3.94 / −0.87 / **+17.40**；`deep_lnres−deep_ln`
−9.00 / −6.66 / +1.59。

**判定（plan §10 升级规则）**：升级条件 = 任一残差-对照比较「点 ≥ +10 且 z-CI 排除 0」。
两条都不满足 → **A1 结束，残差方向同样记录为无增益，深度轴维持关闭**。读法：skip 把 plain
deep 的赤字从 `deep−base` −9.52 收到 `deep_res−base` −6.18（对照读数 `deep_res−deep` +6.82），
方向上支持「部分退化来自没有 skip 的优化」，但 k=3 下 CI 跨 0 且未收回正增益；
LN 变体上 skip ≈ 0。机制读数：`deep_res` EV 0.651（`deep` 0.711 / base 0.709）、
entropy 1.688、kl ≤ 0.0099；`deep_lnres` EV 0.673、entropy 1.790、kl ≤ 0.0068；均无训练崩溃。

## 4c. Amendment A2：heteroscedastic critic（不确定度 + 均值，k=7）

**问题**（operator 提议，plan §11 先写死）：D-C 测出 value 残差里 36% 是 obs 不可消除的噪声；
让 critic 输出 `(mean, logvar)` 并用 Gaussian NLL 训练，能否让梯度不再硬拟合噪声、改善策略？
单因子：`--vf-nll true`，trunk/actor/GAE/其他超参与 base 完全相同（critic 头 55,308 参数，+129）；
`--clip-vloss` 在 NLL 下不适用（已知差异）。

**结果**（`nll − base`，k=7，seed 1–7，每 seed 1,200 副）：

| 读数 | 值 |
|---|---:|
| 合并点估计 | **−13.55 Elo** |
| z-CI | **[−21.67, −5.43]** |
| t-CI | [−23.69, −3.41] |
| se / between-seed sd | 4.14 / 10.97 |

逐 seed：s1 −11.30、s2 −16.08、s3 −13.18、s4 −27.28、**s5 +7.67**、s6 −21.93、s7 −12.75（6/7 负）。

**判定（§4.1 三段式）**：① 不满足；② 不满足；③ **z-CI 上界 −5.43 < +5 → H0，关闭不确定度
critic 方向**。

**机制**：不确定度确实学到了——训练后在真实对局 obs 上（每 seed 8 局、~190 个决策），
predicted logvar 均值 −3.7…−4.0（预测 sd ≈ 0.15–0.17）、跨决策 sd 0.68–1.03、无 clamp 塌缩
（0% 打在 −10）；但均值头变差：末值 EV 0.648（base 0.709），final NLL −1.33（NLL 尺度，可为负）。
最可能的机制：NLL 对均值项的梯度是 `(mean−target)/var`，等于按预测方差做逆方差加权，把低方差
（自信）样本的残差放大，对 RL 的 value 目标反而更不稳；方差头与均值头共享 trunk，辅助目标挤占了
表示（EV 下降与此一致）。未测的下游用法（不确定度加权 advantage、按不确定度门控搜索截断、
集成）留给后续条件项；D-C 与 A2 两条独立结果都指向：value 模型的精度/方差建模不是策略的杠杆。

## 4d. Amendment A3：不确定度采样价值 + 1M 预算（k=7）

**问题**（operator 提议，plan §12 先写死）：A2 的亏损 + run 间波动大；把 critic 的
`N(mean, σ²)` 采样成价值返回（`ṽ=mean+σ·ε`）喂给 GAE/回报，看是否起正则/稳定作用；
并把预算从 500k 提到 1M。三臂：`base1m`、`nll1m`（只换 NLL）、`samp1m`（NLL+采样），各 7 seed。

**结果**（k=7，每 seed 1,200 副）：

| 比较 | 点估计 | z-CI | se / sd | 角色 |
|---|---:|---|---|---|
| `samp1m − base1m` | **−21.92** | [−36.38, −7.46] | 7.38 / 19.52 | 主端点 |
| `samp1m − nll1m` | **−22.74** | [−37.32, −8.17] | 7.44 / 19.68 | 分离采样效应 |
| `nll1m − base1m` | +2.51 | [−5.66, +10.68] | 4.17 / 11.03 | NLL@1M（descriptive） |

逐 seed `samp1m−base1m`：−22.3 / +1.4 / +0.4 / −20.4 / −28.8 / **−55.9** / −27.9（5/7 负）。

**判定**：① 不满足；② 不满足；③ **z-CI 上界 −7.46 < +5 → H0，关闭采样价值方向**。

**机制**：采样把 σ≈0.15–0.19 的噪声注入每个 GAE delta 与 λ-return 目标（logvar 头确实学到了：
均值 ≈−3.8、σ 0.15–0.19、无 clamp 塌缩）；`samp1m` 的 EV 0.333 是采样值口径（被自身采样噪声
污染，不能与均值口径直接比）。逐 seed sd 19.5，并出现 s6 −55.9 的极端读数，比其余臂更不稳。

**预算因素（descriptive，非 paired）**：A2 在 500k 是 −13.55 [−21.67,−5.43]，其 1M 复测
`nll1m−base1m` 为 +2.51 [−5.66,+10.68]：负结果未复现、也未转正。只能说 A2 的 500k 亏损
在这一轮 1M 预算下消失，不能归因于预算（两轮 run 不同）。

## 4e. Amendment A4：均值/方差解耦（k=7，500k）

**问题**：A2 的亏损机制被定位为 NLL 均值项的逆方差加权（`∂L/∂μ=(μ−y)/σ²`，训练后 σ≈0.15）。
A4 保持均值路径与 base 完全相同的 clipped MSE，方差头用 `mean.detach()` 的 NLL 单独训练
（单因子：只去掉逆方差加权）。

**结果**（k=7，每 seed 1,200 副）：

| 比较 | 点估计 | z-CI | t-CI | se / sd |
|---|---:|---|---|---|---|
| `dec − base`（主端点） | **−1.10** | [−10.94, +8.73] | [−13.38, +11.18] | 5.02 / 13.28 |
| `dec − nll`（对照 A2） | **+9.34** | [−4.19, +22.88] | [−7.55, +26.24] | 6.90 / 18.27 |

逐 seed `dec−base`：+7.2 / −22.0 / +12.5 / +5.5 / +3.5 / −17.8 / +3.5（5/7 正，均值被两个负 seed 拉平）。

**判定**：三条规则均未触发——`dec−base` 的 z-CI 跨 0、上界 +8.73 未 < +5、点 −1.10 也 < +5。
按预注册无处置：**dec 不通告增益也不触发关闭**。A2 的亏损机制被确证（解耦后相对 `nll` 回收
+9.34，方向与预注册一致），但解耦后的方差辅助目标对策略是**净零**：EV 0.655（base 0.709、
nll 0.648），logvar 头保持异方差校准（σ≈0.16–0.18、跨决策 sd 0.59–0.74、无 clamp 塌缩）。
高斯 NLL 线到此收口：建模不确定性本身不改变策略质量，之前 A2 的负结果来自损失耦合而非「学噪声无用」。

## 5. 机制读数（归因用，不参与门槛）

| arm | 末值 EV（7/3 seed 均值） | entropy（末 10 update） | approx_kl max | vs-random eval 全点范围 |
|---|---:|---:|---:|---|
| base | 0.709 | 1.755 | 0.0061 | 0.75–0.95 |
| deep_ln | 0.663 | 1.932 | 0.0051 | 0.76–0.98 |
| ln | 0.572 | 1.869 | 0.0053 | 0.74–0.96 |
| deep | 0.711 | 1.647 | 0.0062 | 0.74–0.94 |
| wide | 0.683 | 1.667 | 0.0067 | 0.77–0.96 |

- 没有训练崩溃（KL 均 < 0.007、无 NaN），但 **LN 臂的 EV 明显更低**（0.57–0.66 vs base 0.71），
  entropy 更高（策略更不自信）；`deep` 的 EV 与 base 持平。
- **value 尺度**（固定种子与固定 obs batch）：初始 `|V|` 均值 shared 0.226 / ln 0.267 /
  deep **0.736** / deep_ln **0.942** —— 加深使 `layer_init(std=√2)` 的逐层增益叠起来，
  初始 value 尺度被放大；训练末 `|V|` 均值 base 0.372 / ln 0.178 / deep 0.409 /
  deep_ln 0.212 / wide 0.402（LN 臂末值偏低）。这是 LN/深度改变有效尺度的直接读数，
  与「LN 没带来增益」一致，但本身不构成独立结论（只在本轮读数内）。

## 6. 功效边界与实际校验

预注册表（plan §4.2）在 sd 8.43–17.2 下预估 k=7 的 se = 3.19–6.50。本轮实测 **se=6.07、
sd=16.05**（落在预估区间靠悲观端）；z 半宽 11.9。观察点 −13.95 距 +5 门槛约 1.5 个 se，
H0 判定不依赖边缘。反面：若真值恰为 +10，该规则本身的 confirm 概率 ≤50%（门槛卡在真值上，
EPV §6:267-269 同型）——本轮实验设计足以「排除 +5」，不足以在真 +10 时给出高概率确认；
本轴既然点估计为负，这个 50% 上限不改变关闭判定。

## 7. 缺失证据 / 限制

- **归因比较 k=3**：`ln`/`deep`/`deep_ln−deep`/`wide` 的 CI 偏宽，只能作描述性归因；
  按 plan §4.5 不升级、不进 `plans.md` §6 轴级判定。
- **只测了 500k、单配方、从零**：4–5 层、更长预算、与 warm-start 组合均未测
  （plan §7 条件项，主端点为真才触发；本轮未触发）。**残差连接已由 A1 补测**（§4b），
  plain-deep 的赤字部分来自无 skip 的优化（+6.82），但残差变体仍未收回正增益，轴维持关闭。
- **不确定度只学了，没有用**：A2（§4c）验证的是「让 critic 预测 (mean, logvar)」这一单因子；
  把不确定度接入 advantage 加权/搜索截断/集成等下游用法未经测试。
- **LN 只有一种放置**（每个 hidden Linear 后、激活前，含末层）；未测「末层不做 LN」等替代。
- **深度 vs 优化**难以完全分离：`deep` 的 EV 与 base 持平、训练未退化，但点估计为负；
  结合 `wide` 对照 ≈ 0，最保守读法是「该配方下 3 层 trunk 不优」，而非「深度永远无用」。
  这也与 D-C 的 `deep_head`（+16,641 参数的「再加一层」）负先验一致（[`stage0-critic-ceiling.md`](./stage0-critic-ceiling.md)）。
- **deal seed 台账**：9600–9602 落在 D-C 叶片采集 seed 区间 9400–10599 内，但那是 critic
  拟合数据集（`runs/stage0/critic/data`）而非 h2h bank；配对 h2h 两侧用同一批 deal，
  不构成主端点污染（plan §4.6 已注明）。
- **调度差异**：前 4 个 run 串行、其余 4 流并行（15:47 完成）。每个 run 的配方/步数/seed/
  评测口径完全相同；并行只影响 sps，不影响统计或端点。
- **单一 confirmatory 端点**；其余四个比较全为 descriptive，禁止事后挑选。

## 8. 与既有结论的关系

- 与宽度轴负结果（N-2：256/512）和 D-C 的「再加一层 128 宽」负先验同向：**模型复杂度这条线
  在 trunk 深度/归一化上也没有出路**。
- **A5 的续训结果**：500k 从零的「深度/归一化无出路」结论不延伸到续训——T23 协议续训把三臂
  抬到旧冠军之上 ~9–10 Elo（`o2c_base − w2m_ctl` +10.12，k=7）；但同等续训下
  `o2c_deep_ln − o2c_base` = +1.01（null），架构结论不变；k=3 的 deep_ln +15.41 系小样本高估。
- **A6 的残差 × 续训**：两个残差臂自身同样吃到续训增益（vs 自身 500k +30.3/+20.9），但 skip
  相对 plain twin 无增量（主端点 −2.08，z 上界 +4.56 < +5 → H0；deep 对不可判定）→
  「续训有效、架构无贡献」在残差变体上同样成立。
- 本轮关闭后，训练侧结构项（宽度、激活、双塔、历史编码、critic、value 重采、belief、
  深度/归一化）全部有结论；搜索侧 `search_leafq`（μ=260.03）仍是唯一强杠杆。
- `deep_ln−deep ≈ 0` 与「D-A/D-C：信息不是瓶颈、value 残差不可约」合并读出的图像是：
  平台来自优化/表示以外的因素，或需要改变推理算子，而不是继续堆结构。

## 9. 产物与复现

```bash
# 训练（4 流并行；单 run 也可用 run_train.sh 的 train_one）
bash runs/o2-depth/run_parallel.sh          # 23 run；日志 runs/o2-depth/logs/*.log

# 评估（3 流并行；或 run_h2h.sh 串行）
bash runs/o2-depth/run_h2h_parallel.sh      # 19 条；runs/o2-depth/h2h_*.json + games/*.jsonl

# Amendment A1：残差补测（6 run + 12 条 h2h）
bash runs/o2-depth/run_a1.sh
bash runs/o2-depth/run_a1_h2h.sh

# Amendment A2：不确定度 critic（7 run + 7 条 h2h）
bash runs/o2-depth/run_a2.sh
bash runs/o2-depth/run_a2_h2h.sh

# Amendment A3：不确定度采样价值 + 1M（21 run + 21 条 h2h）
bash runs/o2-depth/run_a3.sh
bash runs/o2-depth/run_a3_h2h.sh

# Amendment A4：均值/方差解耦（7 run + 14 条 h2h）
bash runs/o2-depth/run_a4.sh
bash runs/o2-depth/run_a4_h2h.sh

# 绝对锚点：O2 四臂 vs w2m_ctl（12 条 h2h）
bash runs/o2-depth/run_anchor_h2h.sh

# Amendment A5：续训对标（9+8 run；h2h 18+15 条）
bash runs/o2-depth/run_a5.sh
bash runs/o2-depth/run_a5_h2h.sh
bash runs/o2-depth/run_a5_k7.sh      # A5b：seed 4-7，k=7 复核
bash runs/o2-depth/run_a5b_h2h.sh

# Amendment A6：残差 × 续训（6 run + 24 条 h2h）
bash runs/o2-depth/run_a6_res_cont.sh
bash runs/o2-depth/run_a6_res_cont_h2h.sh

# 后验探针（工作名 a6_probe；非 plan amendment）：独立 bank + 池集群/非传递性
bash runs/o2-depth/run_a6_probe.sh
bash runs/o2-depth/run_a6_transitivity.sh

# 人类语料离线评估（行为一致度/NLL/价值校准）与搜索审计
.venv/bin/python runs/human-eval/eval_corpus.py --reps 1000
.venv/bin/python runs/human-eval/search_audit.py traces/web/run-20261007-032521

# 合并读数（k=训练 seed 数；z=1.96 gate；含 A1–A6 的比较）
.venv/bin/python runs/o2-depth/recompute_o2.py
```

- run 产物：`runs/o2-depth/o2_<arm>__<seed>__<ts>/{agent.pt, metrics.csv, args.json}`。
- 审计用逐副 JSONL：`runs/o2-depth/games/*_seed<k>.jsonl`（多 deal seed 后缀）。
- 本轮未 commit；`src/`、`tests/` 的改动见上文件锚点。

## 10. 过程记录（2026-10-06 ~ 10-07）

| 阶段 | 内容 | 结果落点 |
|---|---|---|
| 预注册 + 红队 | r0 → 红队审查 → r1（plan §1–§9） | [`../depth-normalization-review.md`](../depth-normalization-review.md)；plan |
| 主实验 | 5 臂 × 23 run（500k 从零）：base/ln/deep/deep_ln（主端点 k=7、归因 k=3）+ wide 容量对照 | §3；`runs/o2-depth/o2_*__*` |
| A1 | 残差补测（deep_res/deep_lnres，k=3） | §4b |
| A2 | 不确定度 critic（Gaussian NLL，k=7） | §4c |
| A3 | 不确定度采样价值 + 1M（3 臂 × 7 seed） | §4d |
| A4 | 均值/方差解耦（dec，k=7） | §4e |
| 人类侧 | 10-01~10-03 语料（66 局/1,365 决策）行为一致度/NLL/价值校准；搜索「候选来源」修复 | `runs/human-eval/{eval_corpus,poc_eval,search_audit}.py`；`runs/o4lite-search/{rollout_policy,web_search}.py` |
| 锚点 | O2 四臂 vs `w2m_ctl`（k=3） | §3.1 |
| A5/A5b | 三臂 T23 续训 +1M；base/deep_ln 补 seed 4–7（k=7） | §3.2 |
| A6 | 残差 × 续训：deep_res/deep_lnres 同协议续训 +1M（k=3 screen） | §3.5 |
| 后验探针 | 独立 bank 复核、集群/非传递性、entropy 血统、架构核对 | §3.3、§3.4 |

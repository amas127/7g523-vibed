# 奖励-评估对齐：红队 findings 逐条处置与修复记录（revision-3 → revision-4）

> **角色**：修复/诊断 agent。本阶段**零训练、零 h2h**；只读代码/数据（CPU 复算）；
> 只写 `docs/reward-alignment-plan.md`（已重写为 revision-4）与本文件。
> 未改 `src/`、`tests/`、`tools/`、`docs/adr/`、`docs/plans.md`、`docs/experiments/README.md`、
> `CONTEXT.md`、`DESIGN.md`、`RULES.md`；未 commit/checkout/stash；未用 worktree。
>
> **输入**：`docs/reward-alignment-plan.md`（revision-3）+ 红队 findings JSON
> （12 条 `stats-eval-*`、12 条 `rl-reward-*`、17 条 `engineering-impl-*`，共 41 条）。
>
> **判据**：每条给 disposition（`ACCEPT` / `ACCEPT-WITH-CORRECTION` / `REJECT`）+ 代码或数据
> 证据（`file:line` 或复算数字）+ 在 revision-4 中的改动位置。blocker/major 必须解决或
> **明确升级为需要 owner 决策的阻塞项**。

---

## 0. 处置总览

| severity | 条数 | ACCEPT | ACCEPT-WITH-CORRECTION | REJECT |
|---|---|---|---|---|
| blocker | 6 | 6 | 0 | 0 |
| major | 16 | 16 | 0 | 0 |
| minor | 19 | 17 | 2 | 0 |
| 合计 | 41 | 39 | 2 | 0 |

（没有任何 finding 被整体拒绝。`stats-eval-09` 与 `rl-reward-12` 标为 ACCEPT-WITH-CORRECTION：
其主体结论成立，但各含一个被复算证伪的子论断（「配对不降方差」与「`total_points` 在 rules.py:25」），
见对应条目与 §4 证据表。）

### 0.1 系统性修复（由多条 findings 收敛而来）

1. **删除 M3「Elo 上界」门与跨候选斜率外推**：该 308 Elo/单位 R 斜率由训练进度混杂主导，
   顶部 5 点斜率 CI 为 `[−89.9,+80.7]`，三个 Δ 估计折算成 `+6.0/+8.4/+14.0` 跨越行动门槛且
   都不是上界。`stats-eval-02/03`、`rl-reward-01/02`、`engineering-impl-01` 收敛为：
   **Elo 外推只在 §0.3 作「量级参考（非因果、非上界）」报告，不进门、不路由。**
2. **M2 归因降为纯描述性**：n=200/anchor 上 `ΔR`、`ΔR_sign`、`ΔW`、`ΔL` 的 95% CI 全部含 0；
   删除病态比值式 `1 − ΔR_sign/ΔR`，改用加法分解；明确 `ATTR_RESOLVED=False/未判定`。
   （`stats-eval-01/07/10`、`rl-reward-03`、`engineering-impl-02`）
3. **单张显式决策表**：删除 §0.4 的「Elo 上界 ≥ +10 的一半」触发条件（与 §2.6 的 `<+10` 冲突）；
   路由只用 `SYMPTOM / ATTR_RESOLVED / PLATEAU_FLAT` 三个可判定布尔；「收束」明确定义为
   **资源决策，不是「干预无效」的证据**。（`stats-eval-03`、`engineering-impl-01`、`rl-reward-01`）
4. **主端点重排**：from-random 训练对 random 胜率已停在 0.865–0.900、评估对该区间不敏感；
   加 **μ-headroom 硬前置**，主端点改为 **warm-start 块**（`K2w` vs `C0w`，同父模型），
   from-random 冷启动臂降为 exploratory。（`rl-reward-07`、`engineering-impl-02`）
5. **臂重排**：`K2`（τ=0.2）为唯一 confirmatory 主臂；`K0`（非 outcome-aligned）降为
   secondary 描述；`K7`（τ=0.7）作剂量臂（K0 与 K2 近乎共线）。（`rl-reward-04`、`stats-eval-11`）
6. **机制重述 + V 塌缩止损**：`norm_adv` 下饱和去掉胜局内部散布、保留重新中心的胜/负水平；
   预注册「V sd 随饱和下降」预测与 `≤0.15` 塌缩止损。（`rl-reward-05/06`）
7. **工程写死**：`saturate`/`--reward-cap` CLI 语义与校验、逐臂完整训练命令、跨 seed 两段合并、
   基线快照验收、matplotlib best-effort、analyze.py 内联、文件级回滚清单。
   （`engineering-impl-03/04/05/06/07/08/09/10/11/12/13/14/15/16/17`）

### 0.2 需要 owner 决策的升级项（阻塞）

| # | 事项 | 为什么必须由 owner 决定 |
|---|---|---|
| D1 | **是否触发 Tier 1（是否为本线排 GPU 预算）** | 诊断只能确认症状、不能确立干预价值；`§2.6` 的收束是资源决策。Tier 1 触发权保留给 owner。 |
| D2 | **是否把 anchor 局从 200 提到 ≥1200/候选以真正判定归因** | 这是新成本项（生成新对局），与「Diagnose 不必生成新对局」纪律冲突，须 owner 批准。 |
| D3 | **若命中并产品化，是否修订 `RULES.md:72` RL-2 并同步 `DESIGN.md:199`** | 与现行硬口径直接冲突，需要 owner + 新 ADR。 |
| D4 | **是否需要 `matplotlib`（出 SVG 图）** | 项目环境无 matplotlib、无声明依赖；为出图装依赖会改环境。默认 best-effort 跳过。 |

---

## 1. `stats-eval-*`（12 条）

### stats-eval-01（blocker）M2/D2 归因功效不足
**Disposition: ACCEPT。**

复算确认（`runs/t17_screen/games.jsonl`，anchor vs random，每候选 n=200=100 seed×2 座）：
- 配对相邻平台 ΔR：512k→1M **−0.0255 [−0.0840,+0.0315]**、1M→1.5M **+0.0220 [−0.0345,+0.0800]**、
  1.5M→2M **+0.0125 [−0.0310,+0.0560]**（与 finding 数字一致；finding 用 +0.0325/+0.0820/+0.0565
  的百分位，差异在 bootstrap 实现，结论相同：**全部含 0、符号交替**）。
- 512k→2M 配对 ΔR **+0.0090 [−0.0425,+0.0610]**（finding +0.0088/±0.056，同结论）；
  单候选 R 的 SE≈sd/√200≈0.025；配对差 CI 半宽 0.027–0.053。
- 训练 drift 只有 +0.0195，**小于评估 CI 半宽**。

**改动位置**：`docs/reward-alignment-plan.md` §0.3（撤销「几乎全部来自已锁定胜利后的比分扩大」）、
§2.3（M2 重定义为纯描述性；加法分解 `ΔR=ΔW+ΔL`；`ATTR_RESOLVED=False/未判定`）、
§2.6（路由不再依赖 M2）、§5.2-4。

### stats-eval-02（blocker）M3 的「+6.1 Elo 上界」不是上界
**Disposition: ACCEPT。**

复算 `runs/t17long__1__1790439615/metrics.csv` 在同一 `[512k,2M]` 上三个 Δ 估计：
- 窗口均值 0.5162 vs 0.5360 = **+0.0198** → ×308 = **+6.0**；
- OLS 斜率 +0.0182/1M × 1.488M = **+0.0271** → **+8.4**；
- **centered** rolling-25 端点 0.5112 vs 0.5566 = **+0.0454** → **+14.0**（越过 +10）。
复算 13 候选 OLS `mu~R` = **308.0 ± 15.9**（R²=0.971，确认）；顶部 5 点 = **−4.6 ± 43.5**，
95% CI **[−89.9,+80.7]**（确认）。该斜率由早期短 run + random 锚点共同上升主导（between→within
越界），既非因果也非上界。

**改动位置**：§0.3（改为「量级参考（非因果、非上界）」，列出三个估计）、§2.6（删除 M3 行）、
§A.4。

### stats-eval-03（blocker）§0.4 阈值 5 与 §2.6 M3 阈值 10 冲突
**Disposition: ACCEPT。**

原文 §0.4「Elo 上界 ≥ +10 的约一半」= 5，§2.6 把 M3 定义为 `<+10`，`+6.1` 同时触发两处相反路由。
**改动位置**：删除 §0.4 的条件 (a)；§2.6 改为单张决策表，唯一阈值来自
`docs/experiments/README.md` §3 的 `+10` 行动门槛与 `+5` 止损，且**只在实验判定门中**出现。

### stats-eval-04（major）M1 只用训练 seed 1
**Disposition: ACCEPT。**

`runs/` 下只有 `t17long__1__1790439615` 是 2M run；`mle.json` levels 只有 `p1_*`（seed 1 的
`t17pool__1`）、`l1_*`（seed 1 的 `t17long__1`）、`e1_*`（seed 1 的 early run）——**无 p2/p3 的
screen μ**。复算 `runs/t17pool__{1,2,3}` 在 450–500k 的 return 均值 0.5213/0.5425/0.5286
（seed sd≈0.011），与整个后期 drift(+0.0195) 同量级。

**改动位置**：§0.3（「只有 1 个训练 seed」加粗）、§1.5、§2.6（`SYMPTOM` 现状标注 seed 1）、
R 表新增跨 seed 说明。

### stats-eval-05（major）M4 阈值 0.30 无区分度
**Disposition: ACCEPT。**

复算 within/total：`e1_2k=0.325`、`p1_262k=0.466`、`l1_512k=0.557`、`l1_1M=0.478`、
`l1_1.5M=0.527`、`l1_2M=0.558`、`p1_499k=0.524`——**连 2k 步都过 0.30**。
**改动位置**：§2.5 删除 M4 阈值，`within/total` 降为 D4 描述量、不参与路由；§0.3 更新数字；
§A.6。

### stats-eval-06（major）D3 Pearson 0.966 由训练步数混杂 + 口径偏移
**Disposition: ACCEPT。**

复算：Pearson **0.966**、控制 `log(step)` 后偏相关 **≈0.50–0.62**（finding 报 0.505；我用
ending-roll25 得 0.624、centered 得 ≈0.50，取决于窗口定义——本计划已把窗口写死）；顶部 4 点内
相关 **0.47**、晚期 6 点内 **0.33**。偏移表：`p1_499k` 0.526 vs 0.597（最大 0.07），确认。

**改动位置**：§0.3、§2.4（必须报偏相关 + 口径偏移表；禁止跨口径乘除）；§A.7。

### stats-eval-07（major）比值估计量「1 − ΔR_sign/ΔR」病态
**Disposition: ACCEPT。**

复算 512k→1M：ΔR_sign=−0.0700、ΔR=−0.0255 ⇒ `1−(−0.0700/−0.0255)=1−2.75<0`（finding −1.75），
落在 [0,1] 外且与「|ΔR_sign| 远小于 |ΔR|」相反。`ΔR≈0` 时发散。
**改动位置**：§2.3 **删除比值式**，改用加法分解 `ΔR=ΔW+ΔL`（各给配对 CI）；
§A.3 给出四项配对表。

### stats-eval-08（major）A2 null 与拟议 K_τ 不同域
**Disposition: ACCEPT。**

确认 `reward-shaping-500k.md:42/:48` 三臂命令为 `--opponent greedy
--load-checkpoint runs/probe/base_step00696320.pt`（revision-2、热启动），A2 只 1 seed；
而 §4.1 拟议臂是 revision-3、random、冷启动。
**改动位置**：§1.4 增加「域声明」并**删除「两端点夹住 K_τ」措辞**；§0.4 删除「win 作上界」。

### stats-eval-09（minor）P0 担忧错位；但「配对几乎不降方差」子论断错误
**Disposition: ACCEPT-WITH-CORRECTION。**

- 主体（P0 错位、CRN 天然成立、配对本应可用）：**ACCEPT**。确认 `play.py:364`
  `Match(..., rng=random.Random(seed))`、`game.py:140-147` 发牌先于动作、`ladder.py:239-249`
  同一 `deal_seed` 复用于全部候选×2 座、`record.py:47-54` 策略独立种子。
- 子论断「配对反而没帮上忙（配对 SE=0.0271，而非配对 SE=0.0248）」：**REJECT**。复算
  非配对差 SE = √(sd_a²/n_a+sd_b²/n_b) = **0.0359**（不是单均值的 sd/√n=0.0248）；
  配对 SE=0.0271 < 0.0359，**配对确实降方差**。

**改动位置**：§2.1 P0 改为「按构造成立 + 实测 seed 集相同」，并写明配对 SE 0.0271 vs 非配对 0.0359；
删除「退化为非配对」步骤。

### stats-eval-10（minor）窗口端点挑选掩盖非单调
**Disposition: ACCEPT。**

平台 win% 序列 90.0/86.5/90.0/89.5、E[r_term|w] 0.6306/0.6376/0.6311/0.6447（确认非单调）。
**改动位置**：§0.3 列全四点 + CI、使用趋势而非端点差；§A.2 全表。

### stats-eval-11（minor）K0/K2 近平共线、τ 基于 ~175 胜局分位数
**Disposition: ACCEPT。**

确认 `l1_1M` 胜局 n=173、q10=0.2；K0 与 K2 只在最低 10% 胜局上不同。
**改动位置**：§3.1 把 `K2` 定为唯一主臂、新增 `K7` 作真实剂量跨度；A.5 标注分位数 n 与须报 CI。

### stats-eval-12（minor）D1 自相关担忧过大
**Disposition: ACCEPT。**

复算 t17long `[512k,2M]` 去趋势 acf：lag1 −0.027、lag5 −0.029、lag10 −0.021、lag25 +0.012、
lag50 −0.018；NW SE(lag25)=0.00354 ≈ iid 0.00355。
**改动位置**：§1.6/§2.2/§R2（噪声来源改述为「逐点方差大 + 信号小」；block bootstrap 只作稳健性，
且仅在实测 acf>0.2 时才需要）。

---

## 2. `rl-reward-*`（12 条）

### rl-reward-01（blocker）路由自相矛盾、任何路径都落 inconclusive
**Disposition: ACCEPT。**

与 `stats-eval-03`、`engineering-impl-01` 同一缺陷。修复采用 finding 的两个方向之并：
(i) 默认收束明确定义为**资源决策、不代表干预无价值**；(ii) Tier 1 主端点改为直接机制测试
（warm-start 块），行动门相对该端点设定。
**改动位置**：§0.4、§2.6、R1、§4.1/§4.6/§4.7。

### rl-reward-02（major）M3 不存在上界、308 斜率混杂、M3 非布尔
**Disposition: ACCEPT。**

同 `stats-eval-02`。复算确认顶部 5 点 CI `[−89.9,+80.7]`，×0.0198 得 +1.6（低于 +6.1）。
**改动位置**：§0.3/§2.6/§A.4 删除「上界」措辞与 M3 门；`verdict.json` 不再记 M3。

### rl-reward-03（major）D2 归因错配数据序列
**Disposition: ACCEPT。**

同 `stats-eval-01`。确认 screen `R` 四点非单调、配对 ΔR(512k→2M)=+0.0090 t=0.33；
而显著 drift 在训练 return(+0.0195)。
**改动位置**：§0.3（撤销头条归因）、§2.3（M2 描述性、ΔW/ΔL 分别给 CI）。

### rl-reward-04（major）K0 非 outcome-aligned 却作共同主臂
**Disposition: ACCEPT。**

复算 `ppo.py:210-214` 的逐 minibatch 归一化：用观测 outcome 混合模拟，terminal 胜局归一化
advantage 均值 +0.270/sd 0.599；K0 均值 +0.249/sd **0.000**（所有胜局同值）；损失:胜幅度比
从 ~9.8 升到 ~12.5。K0 同时抹掉「胜>平」。
**改动位置**：§3.1 把 `K0` 降为 secondary 描述臂；`K2` 为唯一 confirmatory 主臂；
§4.7「剂量单调性」明确 `K0` 不参与。

### rl-reward-05（major）机制表述在 `norm_adv` 下错误、端点口径未定
**Disposition: ACCEPT。**

同 `rl-reward-04` 的模拟：归一化后胜局 advantage **非零**（被减均值抬正）；
饱和改变的是**胜局内散布**（0.599→0.000）。
**改动位置**：§0.3、§3.4 重述机制；§4.6 机制端点**必须在归一化后**并同时报归一化前后。

### rl-reward-06（major）缺 V 塌缩竞争预测与止损
**Disposition: ACCEPT。**

确认 `reward-shaping-500k.md:98-101`：`base680k` V sd=0.302、A1=0.113、A1γ1=0.114、A2=0.495；
A1/A1γ1 V-vs-R EV≈−0.03/corr 0.24-0.29（塌缩带）。饱和降低终局回报方差。
**改动位置**：§3.4 新增「V-collapse 竞争预测与止损」（V sd ≤0.15 且 EV≤0 ⇒ 记 critic collapse）；
§4.6/§4.7 把 V 健康纳入并设规则；R 表新增 R13。

### rl-reward-07（major）所有臂对弱/饱和对手，干预落在评估不敏感区
**Disposition: ACCEPT。**

确认 `args.json`：`opponent=random`、`mix_random_prob=0.5`（无 `mix` 时惰性）、`pfsp=false`、
`eval_interval=0`；顶部候选 vs-random win% 0.865–0.900、μ 差 ≤4.0。
**改动位置**：§4.1 新增 **μ-headroom 硬前置**；§4.2 主块改为 warm-start（`K2w` vs `C0w`，
同父模型 `l1_1M`），from-random 冷启动臂降 exploratory；§0.4、R12。

### rl-reward-08（minor）`explained_variance`/clip 的跨臂尺度混淆
**Disposition: ACCEPT。**

确认 `ppo.py:251` `explained_var = 1 − Var(y_true−y_pred)/Var(y_true)`，`y_true=batch.returns`
（分母是本臂自己的返回值方差）；`ppo.py:225-234` `clip_coef=0.1`、`ppo.py:243`
`max_grad_norm=0.5` 均为绝对尺度常数。
**改动位置**：§3.4/§4.6 要求报 `explained_variance` 口径并新增 value-clip/global grad-norm
**绑定比例**；判定不依赖单一尺度读数。

### rl-reward-09（minor）block bootstrap 理由事实错误
**Disposition: ACCEPT。**

同 `stats-eval-12`。确认实测 acf≈0、block=25 CI 与 iid OLS CI 重合。
**改动位置**：§1.6/§2.2（理由改为「对未知自相关的稳健性」，block 仅在实测 acf>0.2 时启用）。

### rl-reward-10（minor）P0 可由代码证明、R8 回退多余
**Disposition: ACCEPT。**

同 `stats-eval-09` 主体。确认 `game.py:140-155` 中 `draw_pile` 是固定 tuple、不重新抽牌，
策略不影响发牌。
**改动位置**：§2.1（P0 按构造成立 + assert 式 seed 集检查）；R8 标记「已消解」。

### rl-reward-11（minor）`d` 双重含义、单位差 100 倍
**Disposition: ACCEPT。**

原文 §2.3 `d=(own−other)/100`，§3.1 `d=own−other` 且写 `r_term=d/100`，§A.4 报 `d/100`
——会误导实现者把 τ 设小 100 倍。
**改动位置**：§3.1 统一 `d=own−other`（整数）、`r_term=d/100∈[−1,1]`，**τ 以 return 单位声明**，
并写 `τ=0.2⇔d=20`、`τ=0.7⇔d=70`。**（第二轮补：§2.3/§2.5/§0.3/§2.7/§4.6/R10 与附录 A.2/A.3/A.6 的
return 尺度符号已全部由 `d` 改为 `r_term`，详见 §6。）**

### rl-reward-12（minor）M4 恒真 + 引用漂移
**Disposition: ACCEPT-WITH-CORRECTION。**

- M4 恒真：**ACCEPT**（同 `stats-eval-05`，已删除门）。
- 引用漂移子论断：**部分 REJECT**。实测：`Rules.total_points` 在 **`rules.py:24`**（finding 说
  25；plan §1.3 的 24 是对的）；`_reward` 的 terminal 分支在 **`env.py:358-361`**（修复后按此写）；
  `Seven523Env.__init__` 在 **`env.py:268`**（plan §3.5 的 266 与 finding 的 264 都不准）。
**改动位置**：§0.2 F1/F5、§3.5、§1.1 全部按实测行号重写。

---

## 3. `engineering-impl-*`（17 条）

### engineering-impl-01（blocker）决策规则自相矛盾
**Disposition: ACCEPT。**

同 `stats-eval-03`/`rl-reward-01`。确认 §0.4(a) 满足（≥5）而 §2.6 判 M3 假。M4 也恒真
（within/total=0.527）。
**改动位置**：§2.6 单张显式决策表（`SYMPTOM/ATTR_RESOLVED/PLATEAU_FLAT`）；删除 M3 门与 M4 门；
删除重复阈值。

### engineering-impl-02（blocker）M2 在计划输入上不可测
**Disposition: ACCEPT。**

确认 `train.py:947-950` 只写聚合 `episodic_return/episodic_length`、`:810-815` 只把 r/l 追加进
`ep_returns`，全仓无逐局 outcome/margin dump。anchor win% 无上升趋势（90.0/86.5/90.0/89.5），
与「线索交给胜率侧」相反。
**改动位置**：§2.3 M2 降级为描述性；删除「¬M2 ⇒ 线索交给胜率侧」；新增 `ATTR_RESOLVED=False/未判定`
与「要判定需 ≥1200 局/候选（owner 批）」；§5.2 验收要求显式写出未判定结论。

### engineering-impl-03（major）验收「git status 无变化」与脏基线矛盾
**Disposition: ACCEPT。**

确认开工前 `git status --short | wc -l` ≈100（64 `' M'` + 36 `??`；**第二轮实测 101，含本 workflow 两份文档**，见 §6 RB-3b），含 `src/seven523/env.py`、
`train.py`、`tests/test_env.py`、`docs/plans.md`、`docs/adr/0003..0006`；`.gitignore:8` 忽略 `runs/`。
**改动位置**：§5.2-2 改为「相对开工前快照无新增 delta」，开工前记 `git status --porcelain` +
`git diff | sha256sum` 到 `runs/reward_alignment/baseline.txt`；显式声明 T15–T17 的约 100 项（+本 workflow 文档）属预期、
不得为「干净」而 git add/checkout。

### engineering-impl-04（major）P0 指定方法不可执行（--no-traces）
**Disposition: ACCEPT。**

确认 `runs/t17_screen/command.txt` 末行含 `--no-traces`；且前提（策略影响发牌）错误。
**改动位置**：§2.1 P0 改为代码级判定（`play.py:364`、`game.py:140-147`、`ladder.py:239-249`、
`record.py:47-54`），删除「读 trace」与「退化为非配对」。

### engineering-impl-05（major）saturate/--reward-cap 语义未定义
**Disposition: ACCEPT。**

确认现有校验只有模式名集合（`env.py:276-280`）与 argparse `choices`（`train.py:120-128`）；
无「saturate 必须带 cap」或 τ∈[0,1] 约束；`tests/test_env.py:629-635`、`tests/test_train.py:239-256`
只固定 4 个模式。
**改动位置**：§3.5 写死 CLI spec（`saturate` 无 cap 抛错；cap 仅在 saturate 合法；cap∈[0,1]；
禁止 `None⇒不饱和` 静默等价 terminal）并新增 4 项契约测试。

### engineering-impl-06（major）§4 缺可执行训练命令
**Disposition: ACCEPT。**

原 §4 只写「与 args.json 对齐」，无入口/`--total-timesteps`/flag。确认
`runs/t17pool__1__1790439615/args.json` 的 `total_timesteps=500000`、`checkpoint_interval=8`、
`eval_interval=0` 等，入口 `python -m seven523.train`，run 目录格式见 `train.py:488`。
**改动位置**：§4.3 给出逐臂完整命令（含 `--total-timesteps 500000` 而非 499712，并说明
`num_updates=total//batch_size=488`），要求写 `command.txt`。

### engineering-impl-07（major）`combine_duel_seeds` 不能合 5 个不同配对
**Disposition: ACCEPT。**

确认 `duel.py:317-325` 要求所有 per_seed 的 `left_id/right_id` 相同，否则 `ValueError`；
`h2h_screen.py` docstring 也只对每个 pair 内部合 deal seed。`runs/t17w1/aggregate.py` 是现成的
跨训练 seed 聚合口径（z 与 t）。
**改动位置**：§4.6 明确两步合并（先用 `combine_duel_seeds`/`h2h_screen` 合 deal seed，再用
`runs/t17w1/aggregate.py` 合训练 seed，同时报 z 与 t）。

### engineering-impl-08（minor）SVG 产物无 matplotlib
**Disposition: ACCEPT。**

确认 `.venv` 无 matplotlib/pandas/sklearn；`pyproject.toml` 无 matplotlib。
**改动位置**：§2.7/§5.1 全部标 **best-effort**，缺 matplotlib 退化为 CSV+ASCII 并在
`verdict.json` 记 `figures: skipped`；唯一出图路径是 owner 批准后加 dev 依赖。

### engineering-impl-09（minor）K0 与 PFSP outcome 回退冲突
**Disposition: ACCEPT（潜伏缺陷登记）。**

确认 `train.py:818-823` 的 outcome 回退 `sign(episode_return)`；K0 下胜局回报=0 会被记平。
本波 `--opponent random` 不触发（`train.py:477-478` 要求 pool+pool-episode），属潜伏。
**改动位置**：§3.5 新增实现约束：`_final_outcome` 必须从 `info` 直取，非 sign-like reward 且
info 缺 outcome 时 raise；§6.4 与 selfplay-pool 交互处登记。

### engineering-impl-10（minor）DESIGN.md:199 会过时但本轮禁改
**Disposition: ACCEPT。**

确认 `DESIGN.md:199` 枚举四模式。
**改动位置**：§3.5/§6.3 增加待办「owner 批准后更新 DESIGN.md:199」；本轮不动。

### engineering-impl-11（minor）机制端点 1 无数据来源
**Disposition: ACCEPT。**

确认训练产物只有 metrics.csv/args.json/checkpoint/agent.pt/pfsp_weights.csv，无逐局 outcome/margin。
**改动位置**：§4.6-1 明确该端点需新 probe；未做则记 **N/A**。

### engineering-impl-12（minor）A.1 bootstrap CI 不可精确复现
**Disposition: ACCEPT。**

复算：moving-block(block=25, 随机起点, B=2000, `default_rng(0)`) = **[0.0106,0.0244]**（与计划一致）；
非重叠块 = [0.0118,0.0248]；窗口差 iid-bootstrap = **[0.0109,0.0290]**（计划写 [0.0101,0.0282]）。
**改动位置**：§2.2/A.1 把块抽样实现写死（moving block、随机起点、百分位法），并给出 block 敏感性与
NW/acf 数字。

### engineering-impl-13（minor）§6.3 文档引用不准
**Disposition: ACCEPT。**

确认 `RULES.md:72` RL-2 为硬口径；`grep -i 'reward|奖励|分差' CONTEXT.md` **无命中**。
**改动位置**：§6.3 改为「RL-2 需 owner+新 ADR 修订；CONTEXT.md 无奖励词条、无需改；
DESIGN.md:199 需同步」。

### engineering-impl-14（minor）deal seed fresh 未对账、runs/ 无命名空间
**Disposition: ACCEPT。**

确认 `selfplay-pool-plan.md` §3.6：占用 `0–8/10–18/29–37`、分配 `39/40/100–148`；其训练 seed 2–6。
原计划用 `--seeds 0,1,2`（0 已被占用）。
**改动位置**：§4.4 训练 seed 建议 fresh `200–204`；§4.5 新增 deal seed 台账（建议 `220–228`），
base seed 也必须 fresh（因 `plan_games`/`plan_duel_schedule` 前缀重叠）。

### engineering-impl-15（minor）端点命令片段不可运行
**Disposition: ACCEPT。**

确认 `tools/head_to_head.py:47-60` `--left/--right` 为 `required=True`。
**改动位置**：§4.6 补全 `--left K2w_s=<ckpt> --right C0w_s=<ckpt>`，并预注册同时报 z 与 t-CI、
以 t 为判定口径。

### engineering-impl-16（minor）analyze.py 在 gitignored runs/、回滚会删
**Disposition: ACCEPT。**

确认 `.gitignore:8` 忽略 `runs/`。
**改动位置**：§5.2-7 要求 `analyze.py` 全文内联进 `docs/experiments/reward-alignment-diagnostic.md`
或归档到 owner 批准的持久位置；§5.3 回滚清单写明。

### engineering-impl-17（minor）README 索引位置/格式与回滚清单
**Disposition: ACCEPT。**

确认 `docs/experiments/README.md` §1 表列为「报告/主题/关键结论」，行序新近在前。
**改动位置**：§5.1 明确插在 §1 表**首行**、列格式照现有行；§5.3 把回滚写成文件级清单
（diagnostic.md / README §1 行 / 可选 plans.md）。

---

## 4. 复算证据与原始 finding 数字对照

| 量 | finding/原计划 | 本次复算 | 判定 |
|---|---|---|---|
| 13 候选 `mu~R` 斜率 | 308.0 ± 15.9 | **308.0 ± 15.9**（R²=0.971） | 一致 |
| 顶部 5 点斜率 | −4.6 ± 43.5 | **−4.6 ± 43.5**，CI [−89.9,+80.7] | 一致 |
| t17long slope/1M | +0.0182 | **+0.01816** | 一致 |
| 512–768k vs 1.5–2M | +0.0198 | **+0.01983** | 一致 |
| 配对 ΔR 512k→1M | −0.0255 | **−0.0255** | 一致 |
| 配对 ΔR 512k→2M | +0.0088~+0.0090 | **+0.0090** | 一致 |
| 非配对差 SE | 0.0248（错误） | **0.0359**（配对 0.0271） | finding 子论断证伪 |
| within/total（平台） | 0.478–0.558 | **0.478–0.558** | 一致 |
| Pearson（centered） | 0.966 | **0.966**；偏相关≈0.50–0.62 | 一致 |
| pool 450–500k 均值 | 0.5223/0.5419/0.5252 | **0.5213/0.5425/0.5286**（窗口定义差） | 同结论（sd≈0.011） |
| `Rules.total_points` | finding 说 25 | **`rules.py:24`** | finding 子论断证伪 |
| `__init__` | finding 说 264 / 计划说 266 | **`env.py:268`** | 两者都不准 |

---

## 5. 未解决的阻塞项（升级给 owner）

见 §0.2 D1–D4。revision-4 的默认路径（Tier 0 收束）不需要其中任何一项即可执行；
它们只在 owner 决定推进实验/出图/改规则时才成为阻塞。

---

## 6. 第二轮修复记录（verify → repair）

> 触发：第一轮 verify 给出 `verdict=issues` 与 3 项 `remaining_blockers`。本轮**只修 verify 明确指出的
> 问题**，仍只写本文件与 `docs/reward-alignment-plan.md`；未改 `src/`、`tests/`、`tools/`、ADR、
> `docs/plans.md`、`docs/experiments/README.md`、`CONTEXT.md`、`DESIGN.md`、`RULES.md`；
> 未跑训练/h2h；未 commit/checkout/stash。

| # | verify 问题 | 改动位置 | 证据 / 自检 |
|---|---|---|---|
| RB-1 | `rl-reward-11` 未真正修复：§3.1 声明「全文只有一个 `d=own−other`（整数）」，但 §2.3/§2.5 仍以 `d=(own−other)/100` 为 return 尺度，100 倍冲突仍在 | plan §2.3（定义行 + 加法分解公式 + `W/L` 定义）、§2.5（分层对象与 `Var` 公式）、§3.1（补「统计量一律写 `r_term`」）；连带 §0.3、§2.7、§4.6、R10 与附录 A.2/A.3 表头、A.6 | 全文检索 `E[d|` / `Var(d` / `sign(d` / `d=(own`：plan 中仅剩 §3.1 的 `sign(d)`（`d` 为整数，本就正确）；所有 `0.63xx` / `0.0095` 等 return 尺度量统一写 `r_term = d/100`，与 §3.1 定义 `d=own−other` 不冲突 |
| RB-2 | 附录 A.3 加法分解表 ΔL 符号在第 1–3 行错误，使 `ΔR ≠ ΔW+ΔL`（新引入的内部矛盾） | plan §A.3 表第 1–3 行 ΔL 单元格（点估计与 CI 端点全部取反），并加一行一致性说明 | 独立复算（anchor vs random，seed 配对，B=4000，`default_rng(0)`，符号约定 Δ=(b−a)）：ΔL(512k→1M)=**−0.0095** [−0.0230,+0.0020]、ΔL(1M→1.5M)=**+0.0055** [−0.0085,+0.0220]、ΔL(1.5M→2M)=**+0.0035** [−0.0070,+0.0140]、ΔL(512k→2M)=**−0.0005** [−0.0110,+0.0085]（第 4 行原就正确）；逐行验算 `ΔR=ΔW+ΔL` 全部成立。verify 的「3/4 行」= 4 行中错 3 行，与此一致 |
| RB-3a | review §0 处置总览的严重度计数有误（表写 major 17 / minor 18） | review §0 总览表 | 按文内逐条 `### <id>（severity）` 计数：blocker 6（stats 3 + rl 1 + eng 2）、major 16（stats 5 + rl 6 + eng 5）、minor 19（stats 4 + rl 5 + eng 10），合计 41；AWC 2 均属 minor ⇒ minor ACCEPT = 17。表已改为 major 16 / minor 19 |
| RB-3b | review/plan 的 `git status --short | wc -l = 100` 与当前实测 101 不符（并发 workflow 漂移） | plan §5.2-2（改为「开工前 ≈100 + 本 workflow 两份文档；计数随并发 workflow 漂移，只作量级声明、不设硬门」）；review §engineering-impl-03 记录（标注第二轮实测 101 并说明来源） | 第二轮实测 `git status --short | wc -l` = 101；本 workflow 只新增 `docs/reward-alignment-plan.md`、`docs/reward-alignment-review.md`（`git status --porcelain` 可验），其余为既有 T15–T17/其它 workflow 改动；故不再以精确计数作门 |

**有意保留不改**：§1 各 finding 的第一轮处置正文与 disposition（历史记录）；verify 已确认解决的其余 40 条。
**对判定布尔的影响**：RB-2 只改符号使表格自洽，`ΔW`/`ΔL` 各 CI 仍全部含 0，故
`SYMPTOM=True` / `ATTR_RESOLVED=False` / `PLATEAU_FLAT=True` 与路由结论不变。

---

## 7. 增补（实现轮，用户要求）

- **新增 Tier 1 job（用户要求）与实现状态**：边界跳变奖励 `terminal_win`
  （终局 `r = 分差 + λ·seat_outcome`，CLI `--reward-shaping terminal_win --win-jump λ`，
  预注册 λ∈{0.25,1.0} + `terminal` 对照）已完成实现与全量测试，**未跑训练**；
  实现状态与预注册细节见 `docs/reward-alignment-plan.md` §4.10。
- **第二波 Tier 1 job（用户要求）与实现状态**：分差饱和 `saturate`
  （终局 outcome 条件式：负取分差、平取 0、胜取 `min(分差, τ)`；CLI
  `--reward-shaping saturate --reward-cap τ`；预注册 `K2` τ=0.2 主臂、`K7` τ=0.7 剂量臂、
  可选 `K0` τ=0 与 `win` 端点锚；τ 校验无静默退化）已完成实现与全量测试，**未跑训练**；
  预注册臂/门/排队（在 §4.10 J 波之后）见 `docs/reward-alignment-plan.md` §4.11。

---

## 8. 执行状态（terminal_win 执行轮，2026-09-27）

- §4.10 的 J 波已执行完毕：12 条新 500k run（`ra_J025__{1..5}`、`ra_J10__{1..5}`、
  `ra_C0__{4,5}`）全部 499,712 步、488 update、无 NaN；C0 seed 1/2/3 按预注册复用
  `runs/t17pool__{1,2,3}__1790439615`。默认 `terminal` 路径在 `saturate`/`--reward-cap`
  合入前后各做一次逐位冒烟：与 t17pool seed 1 对比 41/30 个 update，除 `sps` 外 0 diffs；
  合入后 `tests/test_env.py tests/test_train.py` 134 passed。
- 主端点（同 seed 配对、deal seed 220/221/222 × 400 副、`--device cpu --workers 4` 严格串行，
  `runs/t17w1/aggregate.py` 两步合并）：**J10 vs C0 +6.25** [z −6.31,+18.80；
  t −11.53,+24.02]（训练 seed sd 14.32 支配；p=0.329，两臂 Holm p=0.659）→ 不达 +10 门、
  未触发 ≤+5 止损；**J025 vs C0 −4.59** [−20.08,+10.90] → 触发止损。
- 次端点 vs 冻结 `l1_1M`（描述、预算 500k vs 1M 不匹配）：J10 −1.60、J025 −5.36、
  C0 −8.70，CI 均含 0。
- 机制/健康度：vs random `P_w` 0.879–0.897 未牺牲；无 critic 塌缩（标准化 V sd≈0.62–0.67、
  V-vs-R EV≈0.50–0.55）；`value_loss` 上升可由回报尺度（方差 ≈6.3 倍）解释。
- 报告：`docs/experiments/reward-alignment-terminal-win.md`；条目已加
  `docs/experiments/README.md` §1 首行与 `docs/reward-alignment-plan.md` §4.10 执行状态。

## 9. 执行状态（saturate 执行轮，2026-09-27）

- §4.11 的 K 波已执行完毕：20 条新 500k run（`ra_K0/K2/K7/win__{1..5}`）全部 499,712 步、
  488 update、无 NaN；C0 seed 1–3 复用 `t17pool`、4/5 复用 J 波 `ra_C0__4/5`。Deal seed
  **229/230/231**（fresh，避开 J 波 220–228）、screen 232。
- 主端点（同 seed 配对、3×400 换座、`--device cpu --workers 4` 严格串行，
  `runs/t17w1/aggregate.py` 两步合并）：**K2 vs C0 −18.31** [z −26.04,−10.58；
  t −29.27,−7.36]（5/5 seed 同号；saturate 家族 Holm p=6.9e-06）→ **CI 全负、点 ≤+5，
  触发止损并关闭奖励饱和线**；K7 +7.27 [z −1.76,+16.31；t −5.53,+20.08] 未判定；
  端点锚 K0 −27.72、`win` −6.77。
- 次端点 vs 冻结 `l1_1M`（描述、预算 500k vs 1M 不匹配）：K2 −17.94、K7 −5.57、
  win −16.36、K0 −30.10、C0 −4.44。
- 机制/健康度：vs random `P_w` 0.869–0.901 未牺牲；无 critic 塌缩（K2 raw V sd 0.073
  但 EV +0.233，§4.11 的 AND 规则不成立）；entropy 随饱和加深上升。
- 报告：`docs/experiments/reward-alignment-saturate.md`；条目已加
  `docs/experiments/README.md` §1 首行与 `docs/reward-alignment-plan.md` §4.11 执行状态。

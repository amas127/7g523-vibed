# 搜索配置与定级模式：可用边界、测量方案与逐局 (t, K) 落地（7鬼523）

> **更新（2026-09-29，P6）**：owner 决定把定级池合并为一个：placement 直接使用 manifest
> `levels`（raw 档 + `search_leafq` 同一次联合拟合尺度，C3 调度），删除 `rated` 子池与
> **experimental/unrated** 整个模式。搜索身份的唯一运行时来源改为
> `subjects[].search_config`（manifest 契约块，§4.3 的 `artifacts/search-rung/search_rung.json`
> 仅留作 provenance）；`refit_mle --manifest-out --refit --search-config ID=@file.json`
> 可 seed/refresh，普通 refit 逐字保留。轨迹先验不再按池关闭，而是**按局排除**：只有真正
> 打到搜索 rung 的那局不进轨迹通道并记 `prior_off_reason`，raw 局照常。下面各节保留历史
> 设计与实测记录，与当前实现冲突时以本段为准。

> **更新（2026-09-29）**：截断读出单位修复（+7.87 Elo，correctness）与 **t_leafq value 头**
> （fresh bank 501-505 confirm **+18.24 Elo** vs fixed-mean critic_B）已完成；部署默认 value ckpt 已从
> `critic_B` 切到 `runs/ei2_value_t5/t_leafq/critic.pt`；P4a **已实测并发布**：`search_leafq`
> 作为第 11 个非锚 level 进入 `traces/study/manifest.json`（= `traces/pool10/manifest.json`）的
> 26,000 局同一 joint probit-MLE（ADR-0013），spec=`rolloutt:runs/ei2_value_t5/t_leafq/critic.pt`
> 由 `--spec` 写入并保留；定级调度（C3）已按 owner 决定开放（轨迹先验的处理其后改为**按局排除**，见上条 P6 更新）。
> 数字、caveat 与口径见 `runs/o4lite-search/README.md` 的「2026-09-28 更新」节与 §4.3。

> **状态：已实施（部分；原为 2026-09-28 研究/设计稿）。**自由对战逐局 (t,K) 已落地（§4.1）；
> `search_leafq`（t=5/K=32/C=6、value `t_leafq`）已实测并发布进 `traces/study`(=pool10) manifest
> 第 11 级（§4.3，取代 §4.2 的 `search_t5k16` 规划）；P6 起定级池 = manifest `levels` 单一来源、
> 搜索身份钉在 `subjects[].search_config`、轨迹先验按局排除（文首两条 2026-09-29 更新）。§0–§4.2
> 保留历史研究口径。
> **口径**：revision-3（`rules_id=2e36dbea44893696`）、obs v5（2 家 161 维）、纯 MLP champion
> `w2m_ctl`；§1–§4.2 的搜索测量以 value 核心 `runs/ei/value/critic_B.pt` 为基线，P4a 起部署/发布
> 使用 `runs/ei2_value_t5/t_leafq/critic.pt`（§4.3）。所有数字直接引用
> [`experiments/joint-search-training-wave1.md`](./experiments/joint-search-training-wave1.md)
> （下称 **JS**）与 [`experiments/README.md`](./experiments/README.md)、
> [`experiments/human-elo-10-games-research.md`](./experiments/human-elo-10-games-research.md)
> （下称 **HR**），本文不新算统计量。
> 代码论断给 `file:line`；未实测的开工量标注为**估计**。

---

## 0. 结论（TL;DR）

1. **自由对战：逐局 (t, K) 选择已落地**（§4.1，本次实施）。搜索包装是测试期算子、不进评分身份，
   自由对战本来就不进评分链，所以整张 (t, K) 平面都可以开放，只需要边界/延迟提示与 trace 注解。
2. **定级模式：不能直接套用搜索包装。** 三条独立理由：
   - **评分身份与尺度**：`h2h` 的 Δ 是 400-点 log10 相对分，manifest 的 μ 是 probit-MLE 绝对表
     （`estimator.kind=probit-mle`，ADR-0013）；JS §0.3/§9 明确「搜索/截断包装不进 manifest/ladder
     身份，跨 fit 不可比」。把 +86…+125 加到 `w2m_ctl` 的 μ 上不是已知近似，而是未知偏置。
   - **rung 先验中心就是 μ**：定级拟合给每个非锚 rung `Prior(center=opponent.mu, σ=13.97)`
     （`src/seven523/placement/estimator.py:31,74-100`），几十到上百 Elo 的中心误差会被当先验
     灌进人类估计——而 10 局的人类分辨率只有 RMSE≈53（prior v2）～72（HR），CI ±100–133。
   - **非传递性**：t=5 截断 − 全量 = +67.9（直接）vs +34.8（经 raw 相减），差 ~2.75σ（JS §12），
     「raw 之上的 Δ 表」不可靠。
3. **适合定级的搜索配置 = 在现有 MLE 表里测出来的搜索 rung，推荐且只推荐一个：**
   **`search_t5k16`**（t=5、K=16、value/policy 核心 `critic_B`，对手模型同核心；估 μ≈300，
   比 `w2m_ctl` 高 ~+110）。理由见 §3.1/§4.2：K32 只比 K16 高 +13.6（< 人类分辨率与 +20 行动门槛），
   却贵 2.5×；t=5 是 t 前沿的实测最优。**定级不开放逐局 (t, K)**：每个 (t,K) 是一个不同的对手身份，
   41×64 个点不可能逐个标定。
3b. **实测后的实际配置是 `search_leafq`**（t=5、K=32、C=6、value `t_leafq`；§4.3 取代 §0/§4.2
   的 `search_t5k16` 估计与 +110 h2h 规划值）。
4. **落地顺序**：已按 §4.2 预注册测并入 manifest（26,000 局 joint MLE，分钟级）；web/factory
   定级侧 seam 已接：can-build 能力门（不可构建则跳过+警告）、`--spec` 发布。发布后定级池
   已含该 rung，owner 已接受 Thompson/Fisher 可调度（C3）。P6 进一步：单一池 + 删除
   rated/unrated + `search_config` 进 manifest + 先验按局排除（见文首 P6 更新）。

---

## 1. 问题与约束

### 1.1 定级模式里"配置"到底进入了哪些计算

一次定级会话的对手强度有三处入口，全部以 **manifest 的 μ** 为准：

| 入口 | 代码 | 对 μ 的依赖 |
|---|---|---|
| 人类后验拟合的 rung 先验中心 | `placement/estimator.py:93-99` | 非锚 rung `Prior(opponent.mu, rung_prior_sd)`；σ=13.97（T17 尺度） |
| 自适应选档（Thompson/Fisher） | `placement/opponents.py` `select_opponent` | 用 rung μ 算期望信息/探索 |
| 轨迹先验通道 | `prior.py:142,354-365`（`opponent_elo_ref`）；`session.py:_refit` 传 `opponent_mus` | 每局 S1 特征行带对手 μ；v2/R1 prior 在 raw 池上拟合、验证 |

因此「搜索对手的 μ 从哪来」不是一个显示问题，而是估计器输入问题。

### 1.2 现有搜索侧事实（JS §0–§3、§8–§10）

- t 前沿（K=8，3 seeds，vs raw）：t0 **−183.6**、t1 −40.8、t2 +7.0、t3 +65.4、**t5 +77.5**、
  t7 +81.7、t10 +56.8、全量 +42.6 → 最优 t≈5–7，t0/t1 是**负**配置。
- K 轴（t=5）：K8 +86.1、**K16 +111.2**、K32 +124.7；配对 K16−K8 **+25.0**、K32−K16 **+13.6**。
- 延迟/成本：t5 时 K8 **19–29 ms/决策**、K16 40–58、K32 80–110；单 seed 400 deal 单进程
  K8 371 s / K16 697 s / K32 1726 s（JS §10）。
- 权重侧联合训练全负（N-21/N-22/N-23）：强度来自推理期采样与低方差读出，不改权重、不改评分链。
- `ladder.play_games` 已有 factory 注入缝（`src/seven523/ladder.py:527-539`），搜索侧的
  `RolloutFactory`/`TruncFactory` 就是为它写的（`runs/o4lite-search/rollout_policy.py:696+`）。

### 1.3 硬约束

- **C1 身份**：`rules_id` 只标识规则；搜索包装属于**对手身份**（subject id + spec），
  不是规则身份。评分表可以容纳新 subject，但不能把包装写进 `rules_id`，也不能跨 fit 相加。
- **C2 尺度**：新 rung 必须与现有 10 级进**同一次 MLE**（同一 `games.jsonl`/bootstrap），
  发布走 `tools/refit_mle.py --manifest-out --refit`（ADR-0013）。
- **C3 可选择性**：定级调度（Thompson/Fisher）在 `opponents` 上选档，而 `load_opponents`
  返回的是 **`levels` 的全部非锚 subject**（`src/seven523/placement/opponents.py:146-170`）；
  manifest 的 `rungs` 只是 `build_ladder`/refit 的档位间距契约，**不是定级池过滤器**。
  推论：**新 rung 一旦发布进 `levels`，定级调度立刻可以选它**——这是行为变化，不是可选开关；
  若要「先上线但不参与调度」，需要新机制（subject 级 `selectable: false` / `kind` 过滤）。
- **C4 spec 语法 owner**：`policies.py` 拥有 `random`/`ckpt:` 语法（ADR-0010 §2）；
  `rolloutt:` 是 run-local 语法，任何进入 manifest 的写法要么改 owner，要么走注入映射。

---

## 2. 候选方案与判定

| 方案 | 做法 | 成本 | 判定 |
|---|---|---|---|
| **P0 raw-only** | 定级只用现 10 级；搜索仅自由对战 | 0 | 历史现状（2026-09-29 发布后被 P1 取代；raw 对手仍不调用 factory，`tests/test_web.py::test_placement_ignores_the_free_play_factory` 锁死该半区） |
| **P1 measured rung（推荐）** | 把 `search_t5k16` 作为新 subject 进同一 MLE 表，测完发布；定级把它当普通 rung | 估计 15–60 min 集群 + refit/prior 复核 | **推荐**：唯一同时满足 C1/C2 的路径 |
| P2 offset 加法 | μ_new = μ_raw + h2h Δ(t,k) | 0（数字已有） | **否决**：跨尺度/跨身份（§0 第 2 条）+ 非传递性（JS §12） |
| P3 未定级练习 | 自由对战里打任意 (t,k)，不进口径 | 0 | **已实施**（§4.1） |

补充否决：**权重重训/蒸馏**（N-21/N-22/N-23，JS §11）；**为定级逐局扫 (t,k)**（每个点都是新身份，
且 t0/t1 是负配置，UI 会诱导用户以为"更强"）。

---

## 3. 「适合的配置」怎么选

### 3.1 为什么是 t=5、K=16、只测一个

- **t=5**：t 前沿峰值；t=3 配对 t5−t3 = +15.6（t3×K16 vs t5×K16，95% CI [0.0, +31.5]），
  t7 与 t5 不可分辨（−4.3 [−20.0,+10.9]），t=5 更便宜。排除 t≤2（对 raw 无增益甚至为负）与
  t=10（+56.8 < t5 的 +77.5）。
- **K=16**：K16−K8 = +25（HIT）；K32−K16 = +13.6（稳健但边际递减）。人类定级的分辨率：
  HR 推荐方案 10 局 RMSE≈54（grouped）/72（LOLO），v2 prior 复核 LOLO RMSE **52.9**，
  `stop_ci=50` 才收尾，且 HR 的行动门槛需 ≥15–27 局才谈 RMSE≤50。**13.6 Elo 远小于任何人类读数
  的 CI**，却让单决策成本 ~2.5×（697 s → 1726 s / 400 deal）。所以：K16 作为唯一搜索 rung；
  K32 只作为「若 owner 坚持顶配」的后续选项，机制相同。
- **value/policy 核心**：`runs/ei/value/critic_B.pt`（policy 与 `w2m_ctl` 逐位相同，JS §9）。
  该包装的强度与评分身份依 JS §9 不解耦——所以它必须作为**一个 subject** 整体测量，
  不能拆成「raw 策略 + 搜索加成」。

### 3.2 它有没有用（适用性预期）

搜索 rung 估 μ≈300（`w2m_ctl` 191.5 之上 +110 量级），对大多数人类玩家信息量很低
（Thompson/Fisher 不会优先选它）；它的价值是**给强人类解锁表顶以上的分辨能力**（现表顶 191，
强于顶档的人类会被截断在 lvl4/w2m_ctl 附近）。若实测 / 首局试玩显示利用率低，这属于**预期**
而非失败；维持一个 rung 的成本只是 4,000 局测量 + 表里一行。

---

## 4. 落地设计

### 4.1 自由对战：逐局 (t, K)（本次已实施）

- **服务端**（`src/seven523/web/table.py`）：`WebConfig.search` 携带 `options`
  （`key -> {min,max,default}`）、`label`、`note`、`presets`。`/api/start` 的 free 请求新增
  可选 `search: {trunc_ply, rollout_k}`；`TableSession.normalize_search` 做未知键/越界/非整数
  校验（`table.py:283-318`），缺省填 `default`。生效参数进 `snapshot()["search"]`，
  并写进自由对战 trace 的顶层 `opponent_search`（回放忽略未知键）。
- **语义**：`trunc_ply=0` = 全量 rollout（不是 t=0 的纯 value 诊断臂）；`t>=1` = value 截断；
  `rollout_k` = 隐藏世界数。完整平面合法（自由对战不进评分链），边界 t∈[0,40]、K∈[1,64]。
- **UI**（`static/app.js` `searchControls`）：两个数字输入 + 实测预设按钮
  （t5·K8 / t5·K16 / t5·K32 / 全量 K8，延迟写在 preset `note` 里）；游戏内 modebar 与结算页
  显示当局搜索配置。
- **启动器**（`runs/o4lite-search/web_search.py`）：`--trunc/--full/--rollout-k` 变为**默认值**；
  factory 按请求参数构造 `ValueChampion(trunc_ply=t)`（按 t 缓存）或 `BatchChampion`（全量，按 spec
  缓存）。定级侧 factory 注入已在 P4a 发布后接上（§4.3），但只对 manifest 的非 grammar spec 生效。
- **约束**：包装只作用于 `ckpt:` 对手；`random` 锚不包装；搜索核心固定在 CPU（与 JS 评测同口径）。

### 4.2 定级：`search_t5k16` rung（规划，本次未做；P6 以 `search_config`+单一池落地，见文首更新）

**测量设计（预注册式，照 T17/w2m 流程）**

1. **subject**：`id=search_t5k16`；spec 需携带 pinned 配置（见下方「spec 设计」）。
2. **schedule**：把新 subject 加入现有 10 级 round-robin：10 个无序对 × 400 局（100 deal ×
   2 座位）= **4,000 局新对局**；其余 18,000 局复用 `runs/w2m/calibration/games.jsonl`。
3. **成本（估计）**：搜索侧 ~0.87 s/局（K16）→ 单进程 ~58 min，`--workers 4` ~15–20 min；
   K32 约 2.4×。refit `tools/refit_mle.py --manifest-out --refit --keep-rungs` 在 22,000 局上为分钟级。
4. **发布**：`traces/study/manifest.json`（= `traces/pool10/manifest.json`）经 refit 重发布；
   `rules_id` 不变；`rungs` 由 `--keep-rungs` 保持 `lvl1`/`lvl4`（档位间距契约，与定级池无关，C3）。旧 manifest 归档。
5. **验收闸门**：
   - 机制：与 wave1 相同的 `mechanism_gate`（terminal_rate/abort/fallback；`rollout_policy` STATS）。
   - 统计：新 rung σ ≤ ~7（与现有 rung 同量级）；与最近 rung（`w2m_ctl`）间距 ≫ CI；
     h2h 同 run 复核 Δ(vs raw) ≈ +111 ± 6，偏离说明测量链有问题而不是"更好"。
   - 管线：E4 式镜像对照（同 spec 两侧）必须 0±1（JS §5 的干净管线标准）。
6. **发布后的调度行为（必读）**：搜索 rung 进 `levels` 即进定级池，Fisher/Thompson 会自动把它
   纳入候选（C3）。表顶 191.5 与 rung ~300 间距大，对多数人类信息量低、被选概率小，
   对强人类正是所需；若首期想「只显示、不调度」，必须在发布前加过滤机制（见下表最后一行）。

**spec 语法设计（C4）**——两个候选：

- **(a) query 后缀（推荐）**：`rolloutt:runs/ei/value/critic_B.pt?t=5&k=16`。`policies.py` 只需
  把「`:path` 后跟 `?key=value`」解析成 `(path, options)`：`missing_ckpt_path`/存在性检查仍走
  `path`，`policy_from_spec` 对 `rolloutt:` 前缀抛出「需要注入 factory」而不是「unknown spec」；
  options 原样交给注入 factory。收益：manifest 自描述、`play_ladder list` 能显示；
  代价：改 spec owner（ADR-0010 §2）需一条 ADR 增补。
- **(b) 旁路映射**：spec 保持 `ckpt:`，另存 `search_pool.json`（id → `{t,k,value}`），
  `load_opponents` 不感知，web/测量驱动查映射。收益：零核心改动；代价：第二份身份真相，
  manifest 不再自描述，工具（`play_ladder`/`build_ladder`）都看不到。

**代码缺口（P1 开工时要改）**

| 位置 | 改动 |
|---|---|
| `tools/build_ladder.py:34-38` | `_require_spec` 走 `validate_spec`，只认 random/ckpt → 需 (a) 或测量驱动绕过 |
| `src/seven523/ladder.py:527` | `play_games(factory=...)` 已有；`build_ladder` 需透传 factory（或 run-local 驱动直接调 `play_games`） |
| `src/seven523/web/table.py:372`（`_start_placement`） | 需要给 `_PlacementDriver` 注入 `PlacementSession(policy_factory=...)`（`session.py:133,313-315` 已有缝）；只对 manifest 中带搜索 spec 的 subject 生效 |
| `src/seven523/web/table.py` config/snapshot | 定级不进逐局 (t,k)；UI 只需把搜索 rung 当普通档显示 |
| `tools/play_ladder.py` | 搜索 rung 的存在性检查/启动命令 |
| `placement/opponents.py` + web | **能力闸门**：搜索 spec 一旦进 manifest，stock `7g523-web`/`build_ladder` 的 `policy_from_spec`/`validate_spec` 会拒绝或报「unknown spec」；需要按注入 factory 校验 spec 能力（不能构建的档在 web 里像缺 ckpt 一样跳过后警告），或让搜索池单独一份 manifest |
| manifest 过滤（可选） | 若不想让调度选搜索 rung：subject 加 `selectable: false`（或 `kind: "search"`），`load_opponents`/`select_opponent` 据此过滤；改动小但要写进 manifest schema 文档 |

**轨迹先验通道（关键风险）**：v2/R1 prior 的 `opponent_elo_ref` 只在 raw 池（≈0–191）上拟合
（`prior.py:142,562`；prior 复核见 `experiments/prior-v2-confirmation.md`）。μ≈300 是外推。
两个选项：

- **(a) 扩语料重拟（推荐）**：rung 发布后，用 raw subject vs 搜索 rung 的 trace 行扩语料，
  `tools/fit_trace_prior.py fit --study <extended> --labels manifest` 重拟；验收沿用 prior 协议
  （LOLO RMSE 不得在 raw 档上回退、cell 残差/外推区不失控）。成本与 v2 同管线（估计分钟级，
  未实测），风险是二次展开在外推区的稳定性。
- **(b) 安全回退**：含搜索 rung 的会话关掉轨迹通道（只用结果似然；`use_prior=false` 已有能力），
  精度降到 HR 的冷启动水平。作为 (a) 不过闸时的兜底，不默认。

**人类读数不变**：定级协议、`stop_ci=50`、z=1.96、10 局默认、两通道权重口径都不变；
搜索 rung 只是池里多一个（强）档位。

### 4.3 P4a 实测：`search_leafq`（t=5/K=32/C=6）联合拟合入表（2026-09-29）

**状态（2026-09-29）：已实测、已发布进 manifest 为第 11 个非锚 level，web/placement
已可构建并调度。** 本节取代 §4.2 的 `search_t5k16` 规划口径（§4.2 保留为历史）：实际测量的
配置是 **t=5、K=32、C=6、value 头 `t_leafq`、rollout 对手 `w2m_ctl`**（`O4_TRUNC_PLY=5
O4_AGG=mean`），不是 K16。发布命令：同一 P4a 拟合加 `--manifest-out --refit --keep-rungs
--spec search_leafq=rolloutt:runs/ei2_value_t5/t_leafq/critic.pt`，然后按 t15/t17 惯例把
`traces/study/manifest.json` 逐字节复制为 `traces/pool10/manifest.json`（备份：
`artifacts/search-rung/manifest_before_search_rung.json` 与两个带时间戳副本）。

**口径（一次联合 probit-MLE，ADR-0013）**：`runs/w2m/calibration/games.jsonl`（18,000 局，10 个已发布
raw id）+ P4a 10 个 h2h bank（8,000 局，5 个对手 × 2 个 400-deal bank）= **26,000 局**；
`--anchors random=0`、`--bootstrap 4000 --seed 0`、deal-clustered；**无 `--prior-manifest`**（自由 id 用默认
`Prior(500,200)` MAP 惩罚，与已发布 `table.json` 同协议）。`converged=true`、`margin_identified=true`、
`separation=[]`。

**同一次 fit 内的读数**（μ 是**联合拟合**值；σ 是 Laplace sd；CI 是 deal-bootstrap 95%）：

| id | μ | σ | 95% CI | n |
|---|---:|---:|---|---:|
| **search_leafq** | **260.03** | **3.86** | **[251.22, 269.89]** | 8000 |
| w2m_ctl（raw 表顶） | 187.46 | 4.28 | [176.57, 198.38] | 3600 |
| w2m_plain | 186.25 | 4.28 | [175.09, 197.95] | 3600 |
| w2m_low | 186.00 | 4.27 | [174.86, 197.34] | 3600 |
| pself_s2 | 183.24 | 4.28 | [172.88, 194.29] | 3600 |
| ws_s2 | 182.91 | 3.98 | [173.05, 193.43] | 5200 |
| lvl4 | 181.19 | 3.97 | [171.33, 191.68] | 5200 |
| lvl3 | 126.25 | 3.98 | [116.51, 136.11] | 5200 |
| lvl2 | 109.79 | 4.20 | [98.46, 121.21] | 3600 |
| lvl1 | 84.60 | 3.99 | [74.31, 95.35] | 5200 |

- 相对位置（同 fit 差）: **+72.57 μ vs w2m_ctl**（两者没有直接对局，靠 lvl1/lvl3/lvl4/ws_s2/random
  的图连通读出）、**+77.12 vs ws_s2**（直接对局，模型 logistic ΔElo +152.4）、**+78.84 vs lvl4**
  （直接对局，+155.9）。`search_leafq` CI 与 `w2m_ctl` CI 不重叠。
- **μ 的口径：这个 260.03 是同一次联合拟合出来的绝对读数，不是「raw μ + h2h Δ」的加法**——
  既不是 +111（§4.2 的 K16 h2h 规划值）、也不是 +124.7（JS K32 h2h 值）加到 `w2m_ctl` 上。
  `runs/o4lite-search/valuecal/confirm/confirm.json` 的 h2h **+135.66 ± 3.80 Elo**（vs raw `w2m_ctl`，
  `confirm_leafq`，同 runner/配置）只作 **非权威 sanity check**：拟合的 vs w2m_ctl 超额 +72.57 μ
  ≈ +143.2 logistic Elo（由拟合胜率 0.6952 换算），方向与量级一致；不做跨 fit 相加（JS §12 非传递性）。
- 同 fit 里 raw 表顶 `w2m_ctl` 从旧表的 191.49 降到 187.46（−4.03），其余 raw id 也有 −0.08…−8.15 的
  联合重拟位移。**跨 fit 的 μ 不可比、不可加**；引用搜索 rung 一律用本表。
- caveat：leave-one-bank-out transitivity `max|z|=3.59`，flagged `search_leafq-lvl1`（z=−2.86）与
  `search_leafq-lvl3`（z=+3.59），提示搜索 rung 相对 raw 池有轻度非传递性；`separation` 为空
  （未触发 perfect-record 标记）。

**产物与命令**：

- fit artifact：`runs/w2m/calibration/table_with_search.json`（含 11 个 bank 的 resolution/transitivity 块）；
- 注册 artifact：`artifacts/search-rung/search_rung.json`（id/spec/config/μ/σ/CI/n/rules_id/estimator/source/
  精确命令/非权威 h2h 对照/placement_safety 注记；`merged_into_manifest=true`）；
- 发布 manifest：`traces/study/manifest.json`（= `traces/pool10/manifest.json`，`--keep-rungs` 保持
  `lvl1`/`lvl4`，`rules_id` 不变；subjects 的 `search_leafq.spec` 由 `--spec` 写入，后续 refit 保留）。

```bash
cd /home/amas/.local/src/7g523 && .pi/limits/run_limited.sh cpu \
  .venv/bin/python tools/refit_mle.py \
  --games runs/w2m/calibration/games.jsonl \
  --games runs/o4lite-search/p4a/search_leafq_random/seed520.games.jsonl \
  --games runs/o4lite-search/p4a/search_leafq_random/seed521.games.jsonl \
  --games runs/o4lite-search/p4a/search_leafq_lvl1/seed520.games.jsonl \
  --games runs/o4lite-search/p4a/search_leafq_lvl1/seed521.games.jsonl \
  --games runs/o4lite-search/p4a/search_leafq_lvl3/seed520.games.jsonl \
  --games runs/o4lite-search/p4a/search_leafq_lvl3/seed521.games.jsonl \
  --games runs/o4lite-search/p4a/search_leafq_lvl4/seed520.games.jsonl \
  --games runs/o4lite-search/p4a/search_leafq_lvl4/seed521.games.jsonl \
  --games runs/o4lite-search/p4a/search_leafq_ws_s2/seed520.games.jsonl \
  --games runs/o4lite-search/p4a/search_leafq_ws_s2/seed521.games.jsonl \
  --anchors random=0 --bootstrap 4000 --bootstrap-workers 4 --seed 0 \
  --json --out runs/w2m/calibration/table_with_search.json
```

**placement 集成状态（2026-09-29）**：
(1) `policies` 仍是 `random`/`ckpt:` 的 owner；新增 `buildable_by_grammar()` 作为调用方
    能力谓词，`rolloutt:` 仍由注入 factory 解析（未改 ADR-0010 语法归属）。
(2) `web/table.py _start_placement`（插件 factory 注入，非 grammar spec 走 factory、raw 档仍走
    `policy_from_spec`）与 `placement/cli.py`（新增 `policy_factory` 注入参数）已透传。
(3) `load_opponents` 保留 caller-owned `can_build` 门：web 无 gate 时默认 grammar 能力谓词，
    不可构建则跳过+`MissingCheckpointWarning`，anchor 拒绝即硬错误。
(4) `refit_mle --manifest-out --refit --spec ID=SPEC` 已实现：新 id 必须被拟合命中、已发布 spec
    不许静默改指，重复 refit 自动保留；有测试覆盖。
(5) `build_ladder.py` 对 `rolloutt:` 给出明确指引（需 run-local factory）；`play_ladder` 是静态
    注册表且 `7g523-play` 不能构建 wrapper，故不登记该 rung（docstring 说明），无 unknown-spec 崩溃路径。
(6) prior：**P6 起改为按局排除**：打到搜索 rung 的那一行不进轨迹通道并写入
    `prior_off_reason`（`session.json` / `rungs.json` / `report.json` 以及每局记录）；
    raw 局仍用 2026-09-27 T2 先验（与本次重拟 raw 值差 ≤8.2，按既定 ≤~8 漂移口径使用）。
    **未重拟先验**：raw 语料存在但无 search trace 行，μ≈260 外推需要扩展语料；
    在扩语料之前，搜索 rung 行永不进入轨迹模型。
(7) `selectable:false`/`kind` 过滤未实现：owner 已接受发布即可被 Thompson/Fisher 调度（C3）。

### 4.4 补充测量：`search_ws_leafq`（warm-start base，2026-09-29，只测不发布）

为回答“换 base 是否改变搜索档的全局强度”，用预注册抽签 `random.Random(20260929).choice([1..7])`
选中的 warm-start 模型 `runs/t23wswd__4__1790700546/agent.pt` 做 base，按 t_leafq 同 recipe
在其自身 K=32/t=5 叶分布上重训 critic（`runs/o4lite-search/search_ws/t_leafq_ws/critic.pt`，
policy 非 critic 参数与 base 逐位一致），算子与 `search_leafq` 相同（t5/K32/C6/rollout 对手=自身）。

- 采集：6×300 deal（seed 9500），56,019 决策 / 387,905 叶子，机制门全清，世界多样性 6.96/组；
- 测量：P4a 同款 banks（vs random/lvl1/lvl3/lvl4/ws_s2 × 520/521 × 400 副）；
- **一次 34,000 局 joint probit-MLE**（18k calibration + 旧 P4A 8k + 新 8k）：
  `search_leafq` μ=**257.45±3.66**（CI [249.03,266.40]）、`search_ws_leafq` μ=**260.46±3.67**
  （CI [251.74,269.84]）；主对照 **Δμ=+3.01、胜率 0.5084（95% CI [0.4933,0.5236]，含 0.5）**
  → **无可测差异**（+3 μ ≈ +6 Elo，低于 +10 门槛）；对照 26k 拟合精确复现已发布 260.030969。
- 传递性 flagged：`search_ws_leafq-lvl3`（z=4.39）、`search_leafq-lvl3`（z=2.53），阈值 2.5。
- 结论：搜索档强度与 base 选择基本解耦；**未发布进 manifest**。
  产物：`runs/o4lite-search/search_ws/report.md`、`table_two_search.json`。

---

## 5. 待 owner 决策与开放问题

1. **是否测 `search_t5k16`**（估计 15–20 min × workers=4 + refit/prior 复核），以及是否顺带
   `search_t5k32`（+13.6 Elo、2.4× 成本；本文建议不急）。
2. **spec 方案**：曾采用 **`--spec ID=SPEC` + 注入 factory**；**P6 起搜索身份改为
   `subject.search_config`**（manifest 单一来源）：spec 只存 `rolloutt:<value_ckpt>`，
   t/K/C/rollout cap/rollout 对手/aggregate 全在 `search_config` 里，插件按 `--manifest`
   读取并在 placement 路径用空请求（pinned），自由对战仍可逐局覆盖 t/K。
   `artifacts/search-rung/search_rung.json` 仅作为测量 provenance 保留。
3. **发布后允许定级调度选搜索 rung**（C3）：owner 已接受；无 `selectable:false` 过滤。
4. **prior 外推 follow-up**（未做）：raw 语料存在但无 search-rung trace 行，
   `tools/fit_trace_prior.py fit --study <extended> --labels manifest` 重拟需先扩语料；
   P6 的实现是**按局排除**（搜索 rung 行不进轨迹模型，raw 行照常），不是整会话关通道。
   非传递性使「截断比全量强多少」本身未定论（JS §12），但这不影响"测一个 rung"的做法——
   测出来多少就是多少，不靠加法。
5. 体验：人类对搜索对手的主观难度与可信度没有数据；建议 rung 上线前先自由对战试玩
   （逐局 (t,k) 已可用）并记录非正式反馈，不进统计判定。

---

## 附录 A：命令草案（P1 开工时照抄改路径）

```bash
# 0) 机制 gate（与 wave1 同口径）
uv run --group train python runs/o4lite-search/mechanism_probe.py --audit \
  --games 200 --seed 0 --rules-id 2e36dbea44893696 \
  --out runs/search_rung/audit --tb runs/search_rung/tb/audit

# 1) 在 10 级 round-robin 上补 4,000 局（100 deal × 2 座位 × 10 对），
#    搜索侧用 TruncFactory(t=5, K=16)，其余 18,000 局复用 w2m calibration 语料；
#    run-local 驱动（待写）→ ladder.play_games(schedule, entrants, factory=..., results_out=...)
#    —— 产物必须带 rules_id；先把新 subject 的 id+spec 写进 study manifest
#    （study.merge_manifest），否则 --refit 发布时会把没 spec 的新 id 剪掉。

# 2a) 同表重拟（与 w2m 同口径：bootstrap 4000 / seed 0）
nice -n 5 .venv/bin/python tools/refit_mle.py \
  --games runs/w2m/calibration/games.jsonl \
  --games runs/search_rung/games.jsonl \
  --anchors random=0 --bootstrap 4000 --bootstrap-workers 8 --seed 0 \
  --json --out runs/search_rung/table.json > runs/search_rung/artifact.json

# 2b) 发布进 manifest（rungs 冻结）
nice -n 5 .venv/bin/python tools/refit_mle.py \
  --games runs/w2m/calibration/games.jsonl \
  --games runs/search_rung/games.jsonl \
  --anchors random=0 --bootstrap 4000 --bootstrap-workers 8 --seed 0 \
  --manifest-out traces/study/manifest.json --refit --keep-rungs \
  --json > runs/search_rung/publish.log

# 3) prior 复核（选项 a；不含搜索 rung 的档位 LOLO RMSE 不得回退）
uv run python tools/fit_trace_prior.py fit --study traces/study10 \
  --labels manifest --out artifacts/human-elo/prior.json

# 4) 上线后用浏览器验证：定级模式里 search_t5k16 显示为普通档；
#    自由对战逐局 (t,k) 选择（本次已落地）
uv run --group train python runs/o4lite-search/web_search.py --manifest traces/pool10/manifest.json
```

> 记录纪律：测量报告回填 `docs/experiments/`（新建 `search-rung.md` 或并入 JS）；
> manifest 变更同步 `docs/experiments/README.md` §0 与 `docs/human-play.md` §3；
> 若采纳 spec 方案 (a)，新增 ADR 记录 `?t=&k=` 语法归属。

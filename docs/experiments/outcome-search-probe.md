# outcome 头进搜索的第一版探针：截断叶值 `V_base + β·u`（机制 + 离线 ROI，k=1）

> **状态：机制门通过；离线 ROI 在 7M 上 null/负、在 9M 上初现正号（小样本），未进 game screen。**
> 线性叶值 blend 不是一个可以立刻上游戏验证的规则；9M + β=1 值得先做小规模确认。
>
> **operator 请求（2026-10-08）**：保存 7M/9M 检查点，开始探索能利用上 outcome 头的搜索规则。
>
> **口径**：revision-3 + obs v5；对象 = `runs/bres/cont/selected/bres221_cont10m_{7M,9M}.pt`
> （17,132 参数、`vf_outcome=True`）；搜索 = value-truncated O4-lite（K=8 决定化、top-6 候选、
> `trunc_ply=5`，`runs/o4lite-search/rollout_trunc.py`）；bank = 模型自博弈 40 局、seat 0 的
> 全部决策（7M 828 个 / 9M 813 个）；离线判定 = 对翻转决策的两个选择各跑 K=32 **全终局**
> rollout（同一决定化配对，base policy 自博弈），取真实终局 margin 与胜负。

## 0. 规则与实现

```
V_leaf = V_base + β · u,   u = 折后胜负头输出 (tanh),  β = 0 即现状（critic only）
```

- 实现：`ValueChampion._values` 新增 `outcome_blend`（`rollout_trunc.py`）；`TruncFactory`
  从 `O4_OUTCOME_BLEND` 读默认值，**默认 0，生产路径逐位不变**。缺 outcome 头的 ckpt 传
  非零 β 会 fail-loud。
- 探针：`runs/o4lite-search/outcome_probe.py`（决策级、不玩游戏；同一 seed/同一 view 顺序下
  各 β 的决定化逐位配对，所以翻转可归因给叶值权重）。

## 1. 机制（决策翻转率）

| β | flip vs β=0 | 占 bank | changed vs raw |
|---|---:|---:|---:|
| 0（现状） | — | — | 35.5%（294/828） |
| 0.5 | 47/828 | 5.7% | 36.4% |
| 1 | 65/828 | 7.9% | 37.2% |
| 2 | 71/828 | 8.6% | 37.2% |

机制非空：outcome 项在 β≥0.5 就稳定改变 5–9% 的搜索决策。

## 2. 离线 ROI（翻转决策 × K=32 全终局判定）

每个翻转决策分别用两个选择各跑 32 个全终局 rollout，统计平均 margin / 胜率：

| 模型 | β | 翻转数 | Δmargin（分） | Δ胜率 | outcome better/worse/tie |
|---|---:|---:|---:|---:|---|
| 7M | 0.5 | 47 | −1.85 | +0.60pp | 13 / 11 / 23 |
| 7M | 1.0 | 65 | −3.81 | −2.45pp | 14 / 22 / 29 |
| 9M | 1.0 | 68 | **+1.79** | **+3.22pp** | 19 / 12 / 37 |

## 6. Web 池发布（2026-10-08）

两个 β=1 搜索档已以**已测量 subject** 身份发布进 `traces/study/manifest.json`
（= `traces/pool10/manifest.json` 镜像）：

| id | μ | σ | 95% CI | n |
|---|---:|---:|---|---:|
| `bres221_7M_b1` | 262.14 | 2.68 | [256.94, 267.58] | 8,000 |
| `bres221_9M_b1` | 265.70 | 2.70 | [260.37, 271.10] | 8,000 |

- 语料：18k calibration + p5a（2 subject × 5 对手 × 2 seed × 800 局）= **34,000 局**；
  `converged=true`、`margin_identified=true`、`separation=[]`；raw 顶 `w2m_ctl` 187.04。
- **边界（operator option B）**：p4a 的 8,000 局 `search_leafq` 语料已不在磁盘上，本次不重生成；
  `search_leafq`（260.03）与 web-local 档（`head32ln5m`/`search_head32ln5m`/raw
  `bres221_7M`/`bres221_9M`）按 `carried_over` 保留旧发布值，不进 34k fit。所以「两档在
  `search_leafq` 同档」是相对其旧发布值，不是同 fit 直接测量。
- 两档是池中**最强的已测量** subject（`search_head32ln5m` 271.19 仍是 carried 粗估）。
- 实现：`search_config` 新增可选 `outcome_blend`（`study.py` 透传 + `web_search` 校验 +
  `web_plugin`/`web_twin`/`rollout_trunc` 三个 factory 透传），缺省 0.0 与历史逐位一致；
  非 0 时缺 outcome 头的 ckpt fail-loud。
- provenance：`artifacts/web-bres221-b1/`（publish.json / table.json / 回滚快照）。

### 2.1 扩大 bank（80 局；9M β=1/2、7M β=2）

bank 只由 (检查点, games, seed, seat) 决定，所以同 games 的两次运行共享同一 bank，可横向比较：

| 模型 | β | bank | 翻转 | Δmargin | Δ胜率 | outcome better/worse/tie | sign p |
|---|---:|---:|---:|---:|---:|---:|---:|
| 9M | 1 | 1,594 | 127（8.0%） | +0.45 | **+4.16pp** | 41 / 18 / 68 | 0.0038 |
| 9M | 2 | 1,594 | 152（9.5%） | +0.05 | **+4.81pp** | 48 / 25 / 79 | 0.0095 |
| 7M | 2 | 1,611 | 144（8.9%） | −2.59 | −1.17pp | 43 / 42 / 59 | 1.0000 |

（p 是决策级符号检验；同一局内的决策相关，p 值偏乐观，只当方向参考。β=1 与 β=2 的翻转集
不是同一批决策，跨 β 比较本身是描述性的：β=2 翻的是基础胜率更低的决策（0.192 vs 0.273）。）

## 3. 读法

1. **7M 的翻转没有变好**：β=1 在 margin 和胜率上同向为负，β=0.5 只在胜率上是 +0.6pp 的平。
   判定器是自博弈，而 β=0 的选择天然在 margin 上占优，所以 margin 负是弱证据；但胜率也
   不涨，说明 7M 的 outcome 头在 t=5 叶子上没有提供正确的「保赢」信号。
2. **9M 出现一致的正号**：40 局 bank 上 margin +1.79、胜率 +3.22pp；80 局 bank 上 β=1
   胜率 **+4.16pp**（41/18，p=0.0038）、β=2 胜率 **+4.81pp**（48/25，p=0.0095），且 β 越大
   margin 增益越小（+0.45 → +0.05）。方向与「赢局已定时别为 margin 送掉胜局」一致；β=2 的
   胜率点估计略高，但与 β=1 的差在噪声内，不能分离出「β=2 更好」。两个 β 仍是自博弈判定，
   **未进 game screen，不能宣布规则成立**。
3. **线性一刀切可能是错的设计**：7M 与 9M 的差异提示 head 质量/校准不同；对所有决策同一 β
   会把远离胜负边界的 margin 交换也翻掉（outcome tie 占多数）。更合理的候选是 guard 式规则：
   只在 |Δmargin| 小、Δu>0 时翻转，或对 u 接近 0 的状态加权。
4. **未做的事**：game screen（A/B 对局）。当前 spec 语法 `rolloutt:<ckpt>` 不带 β
   （env 全局），两臂要分进程或用现有的「两臂各自 vs 同一 raw」外部口径；先在探针上做
   guard 变体更便宜。

## 4. 下一步

1. **guard 规则离线探针**（同一 bank，零新增算力）：`Δmargin > −τ` 且 `Δu > 0` 才接受翻转；
   扫描 τ；看翻转数、判定 margin/胜率。
2. 若 guard 或 9M+β=1 仍为正：把 β/τ 编进 spec（`rolloutt:<ckpt>;beta=<x>`），跑小规模
   A/B screen（3×100 deal 起步），再决定是否进 confirm。
3. 反向检查：用 `--rollout-opponent` 换成固定对手再判定，看正号是否只在自博弈下存在。

## 5. 产物

| 路径 | 内容 |
|---|---|
| `runs/bres/cont/selected/bres221_cont10m_7M.pt`、`_9M.pt` | 保存的检查点（含 `extra.global_step`） |
| `runs/o4lite-search/outcome_probe.py` | 决策级探针（翻转率 + K=32 全终局判定） |
| `runs/o4lite-search/outcome_probe/summary.json` | 7M β∈{0.5,1,2} + β=1 判定 |
| `runs/o4lite-search/outcome_probe_b05/summary.json` | 7M β=0.5 判定 |
| `runs/o4lite-search/outcome_probe_9M/summary.json` | 9M β=1 判定 |
| `runs/o4lite-search/rollout_trunc.py` | `outcome_blend`（env `O4_OUTCOME_BLEND`，默认 0） |
| `artifacts/web-bres221-b1/` | Web 池测量发布（见 §6） |

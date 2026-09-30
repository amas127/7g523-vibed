# v5 优化计划红队 findings 处置（revision-4 复核）

> **执行状态**：第一波已执行、F1b 触发止损（见 [`experiments/v5-optimization-wave1.md`](experiments/v5-optimization-wave1.md)）；
> §5 的 Stage 2 条件项未执行。

> **角色**：修复/诊断 agent。输入 = `docs/v5-optimization-plan.md`（revision-3）与红队 findings JSON
> （35 条：`stats-eval-01..08`、`training-ml-01..15`、`engineering-ops-01..12`）。
> **本文件记录逐条处置、证据、改动位置。** 本阶段零训练、零 h2h；只读代码/只读命令，
> 只写本文件与重写后的 `docs/v5-optimization-plan.md`；未改 `src/tests/tools/ADR/plans.md/
> README/CONTEXT/DESIGN`，未 commit/checkout/stash。
>
> 处置汇总：**accepted 35 / rejected 0**。其中 `stats-eval-02` 的建议「F1a 作为唯一 confirmatory
> 端点」与 `training-ml-02` 的建议「F1b 升为 confirmatory 主端点」直接冲突，已裁定（§3）：
> findings 本身均成立，采纳 `training-ml-02` 端点选择 + `stats-eval-02` 的「不得把单 ckpt 当
> 训练 seed 族」核心，二者以「F1b 左端 = 7 个 fresh 训练 seed」同时满足。**无遗留 blocker 需用户
> 决策**；有 2 项按设计**延后且需未来授权**（§5）。

---

## 0. 处置原则

1. 每条 finding 按 `accepted（已修） / accepted（部分采纳，说明理由） / rejected（给反证）` 处置。
2. blocker/major **必须在修复后规划里解决**，或升级为「需用户决策的阻塞项」。本报告
   0 个遗留 blocker。
3. 只允许改 `docs/v5-optimization-plan.md` 与本文件；所有修复落在这两个 Markdown 内。
4. 涉及 `src/tests/tools` 的改动（A2 深度/LayerNorm、A3 价值改造、A4 集成、A5 bot、
   登记新梯级）**一律不实现**，只保留设计/预注册命令。

---

## 1. 处置总表

| id | 严重度 | 处置 | 核验证据（只读） | plan 修复落点 |
|---|---|---|---|---|
| stats-eval-01 | blocker | accepted | `evaluation-protocol-validation.md` §6：k=3×400 规则「CI 排 0 且点 ≥+10」power=0.26–0.31@Δ10；`t17-recalibration.md` §5 3-deal 仅 +10.3 | 主端点改 k=7 fresh seed + 9 fresh deal；§1.4 功效表；§3.4 门 |
| stats-eval-02 | blocker | accepted（端点选择部分让位 training-ml-02，见 §3） | `runs/t17_screen/h2h_self_vs_l1_1M.json` `between_seed_sd=3.15` 是 **deal-seed** sd 非 ckpt sd；`duel.py:104-273` 按牌聚簇 | F1b 左端显式定义为 7 个 fresh 训练 ckpt，命令逐 seed；F1a 降描述性 |
| stats-eval-03 | major | accepted | `t17-recalibration.md` §5：3 deal +10.3 跨 0，补 6 deal 后 +16.62 → stop-and-choose | §1.2 明写「+16.62 探索性」；本波 deal seed 冻结 fresh 10–18 |
| stats-eval-04 | major | accepted（公式细节注明） | `h2h_self_vs_{p1_499k,p2_500k,p3_500k}.json` 三点 +27.9/+10.3/+27.3 → sd≈10；`event-history-pilot.md` §5.1 sd=15.88 | §1.2 重写为「未确立」；§0.3 排序改「方向为正、未确立」 |
| stats-eval-05 | major | accepted | `duel.py:273` 用 RMS；`runs/t17seq_train/aggregate_seq.py:46` 用算术均值；`event-history-pilot.md` §8「k=3 seed sd 很吵，max() +10%」 | §3.3 aggregate 规格用 RMS + 打印三分量；§3.4 binding-term 升级规则 |
| stats-eval-06 | major | accepted | plan §6.1 声明 F2 探索性、§3.4 却给 F2 升级门；每 h2h `bootstrap_se≈11` → k=1 半宽 ±21.6 | §3.4 F2 改「仅方向性提示、不自动触发」；触发线改 +30 |
| stats-eval-07 | minor | accepted | `docs/experiments/README.md §3`「3×200 = 粗筛（只杀 <20）」 | §3.4 Block C 措辞限定；删「关闭 A2」 |
| stats-eval-08 | major | accepted | `train.py:557` 新建 Adam；`train.py:706` 重算 frac；plan §7.4 R2 只认 LR | Stage 2 增 `t17w1wsctrl` 对照；归因改 `ws − wsctrl` |
| training-ml-01 | blocker | accepted | k=3→t 半宽 ≈39.7@sd16、k=5≈19.9；`event-history-pilot.md` §5.1/§5.3 | 主端点 k=7（§1.4）；§3.4 升级规则；报告模板写「跨 0=未判定」 |
| training-ml-02 | blocker | accepted | random 侧 sd 15.9–22.25（event pilot）；self 侧 `t17-recalibration.md` §5 内差小 | F1b 主端点；F1a 降描述性；F1 全部 fresh deal |
| training-ml-03 | major | accepted | `t17pool__1/agent.pt` 与 `checkpoint_step499712.pt` 逐参数相等（本报告 §2.2 复验）；`h2h_self_vs_p1_499k.json`=+27.9 | §1.2.3 更正「F1a N=1 已存在」；原 §6.5 错误表述删除 |
| training-ml-04 | major | accepted | `train.py:557`/`706`；plan §7.4 R2 仅 LR | 同 stats-eval-08 |
| training-ml-05 | major | accepted | `train.py:395/677`、`ppo.py:174`、`train.py:422`（num_envs=16 改 batch/minibatch/更新数/env seed） | §2 A2 删除 `--num-envs` 单因子；Block C 只留 hidden；措辞限定 |
| training-ml-06 | blocker | accepted | F1a/F1b 皆 ckpt-vs-ckpt；`head_to_head.py:47-58` 支持 `random`；`build_ladder --anchor random` 现成 | 新增 **F0** 共享 fit 绝对锚 + §3.4 绝对退化门 |
| training-ml-07 | major | accepted | plan §3.2 A1⑤ 含 `--pool-member 1@self`；`league.py:157-167` 刷新语义 | A1⑤ 去 `self` 成员；PFSP 移第二波 |
| training-ml-08 | major | accepted | `structural-directions.md:3`「ckpt 已删除、旧规则口径」；`t17self/pool metrics.csv` EV 0.61–0.70 | §2 A3 改 ckpt 内部判据；0.065 标历史 |
| training-ml-09 | minor | accepted | `train.py:101/120`；`t17self` ep_len 20.86、`t17pool` 23.74；γ^21=0.810/0.900/0.979 | §0.3.7、§2 A2 注明 gamma 不单列 |
| training-ml-10 | minor | accepted | `t17self` 末熵 1.543 vs `t17pool` 1.755；return −0.082 vs +0.505 | §6.8 增末熵/EV/return 观测与熵门 |
| training-ml-11 | minor | accepted | `train.py:362` `self_play_sample=False`；`train.py` 刷新逻辑；`self_play_refresh=0` 短路 | Stage 2 加 refresh50 新鲜度点，机制臂入第二波设计 |
| training-ml-12 | minor | accepted | `aggregate_seq.py:46` 只用 `bootstrap_se`；`duel.py:273` RMS；`README §3` 显式 max | §3.3 aggregate 规格（RMS+三分量+z/t） |
| training-ml-13 | minor | accepted | plan 原全用 `--seeds 0,1,2`（与 T17 §5 重叠） | confirmatory 换 fresh deal 10–18；§6.6 |
| training-ml-14 | minor | accepted | `t17-recalibration.md` §2 lvl4 CI [173.65,200.11]；§5 +16.62 在 lvl4 CI 内 | §3.4 书面目标 +20；+20 走单侧非劣；单 seed +20 先降级 |
| training-ml-15 | minor | accepted | 定位清单与现场一致（`train.py:95-96/395/422/557/677/695-703/706`、`league.py:157-167/202`、`ppo.py:174`、`head_to_head.py:19/47-58/273`、`aggregate_seq.py`） | §3.3/§5/§7 复现入口按此写 |
| engineering-ops-01 | blocker | accepted | `ls -d runs/t17w1` → No such file or directory（本报告 §2.1 复验）；plan 无 `mkdir` | §3.0 加 `mkdir -p runs/t17w1`；每条 h2h 前提示 |
| engineering-ops-02 | major | accepted | plan §3.1/§3.3 F1a N=1 = self_s1 vs rand_s1；`t17-recalibration.md` §5 已给 +27.9；原 §6.5 前提为假 | F1a 降描述性并重测 fresh deal；删错误表述；confirmatory 全 fresh seed |
| engineering-ops-03 | major | accepted | plan 抓 selflong 512000 但无消费者；§3.4 门用「1M vs 500k」未绑定端点 | §3.3 F2 加 `selflong@999424 vs selflong@512000`（同 run）+ vs self_s1（标注跨 run） |
| engineering-ops-04 | minor | accepted | Block B 含 1M run；plan 记 7 等效 | §3.7 重算 A+B=13 等效、总 14.2；并修 200k=0.4 等效 |
| engineering-ops-05 | minor | accepted | `aggregate_seq.py:15/24/46`；h2h 输出 `runs/t17w1/h2h_*_sN.json` | §3.3 aggregate 规格（正确路径 + RMS + z/t） |
| engineering-ops-06 | minor | accepted | 8×128=1024→16×128=2048；`train.py:677` 195→97；`ppo.py:174` minibatch 256→512 | Block C 删除 b2048；如需再测标为复合因子 |
| engineering-ops-07 | minor | accepted | `train.py:305-486` 有 `--pfsp*`；plan A1⑤ 名为 PFSP 实为静态池 | §3.2 A1⑤ 显式 scope 静态池；PFSP 入 §4 第二波 |
| engineering-ops-08 | minor | accepted | `poll_snapshot.py` 主循环 `while True + sleep(0.25)`；`--checkpoint-interval 500` 只落 512000 | §3.0/§3.2/§7.1 明确 `nohup ... &`；注明 999424 依赖 `agent.pt` |
| engineering-ops-09 | minor | accepted | `train.py:557` 新建 Adam（无 warm-start 状态继承） | §7.4 R2 扩展为「LR 重退火 + Adam 重置」双混淆 |
| engineering-ops-10 | minor | accepted | `arena.py:111-118`、`build_ladder.py:117-122` 均有 `--device/--workers` | §4 第二波命令显式 `--device cpu --workers 4` |
| engineering-ops-11 | minor | accepted | `t17-recalibration.md` §1：l1_1M 是 13 候选 MLE argmax，平台 5 点 187.6–191.6 | F1c 描述性对照 l1_2M；§1.2 标注 winner's curse |
| engineering-ops-12 | minor | accepted | `find . -maxdepth 1 -iname 'readme*'` 空；索引在 `docs/experiments/README.md:126 §1` | plan 全文改显式路径 `docs/experiments/README.md §1` |

---

## 2. 关键 findings 的代码/数据证据（复验记录）

### 2.1 `runs/t17w1` 不存在（engineering-ops-01，blocker）

```
$ ls -d runs/t17w1
ls: cannot access 'runs/t17w1': No such file or directory
```

plan revision-3 §3.3/§7.1 里所有 `> runs/t17w1/h2h_*.json` 重定向都会在第一条命令失败，
M1 无法产出。修复：§3.0 增加 `mkdir -p runs/t17w1`。

### 2.2 `rand_s1` = `p1_499k`（training-ml-03 / engineering-ops-02）

`torch.load` 复验：

```
runs/t17pool__1__1790439615/agent.pt             extra.global_step = 499712
runs/t17pool__1__1790439615/snapshots/checkpoint_step499712.pt  extra.global_step = 499712
model state_dict 8/8 张量 all equal = True
obs_version=5, arch=shared, hidden=128
```

`runs/t17_screen/h2h_self_vs_p1_499k.json`：`combined.elo_diff.mean = +27.88`，
即 revision-3 plan §6.5「F1a self vs 同 seed random 500k 在 T17 §5 未做」为**假**；
p1 与 self_s1 同 seed 1。修复：§1.2.3 更正，F1a 降描述性。

### 2.3 单训练 seed 的方差构成（stats-eval-02 / training-ml-01/02）

`runs/t17_screen/h2h_self_vs_l1_1M.json`：

```json
"elo_diff": {"mean": 10.28, "bootstrap_se": 11.19, "between_seed_sd": 3.15, "se": 6.46}
```

`duel.py:104-273` 的 `between_seed_sd` 是 **deal-seed** 的点估计 sd（3 个 deal seed），
不是训练 seed sd。即单训练 seed 下 boot 分量主导；跨训练 seed 后训练 seed sd 才是主项。
→ F1b 必须用「多个 fresh 训练 ckpt vs 一个固定参照」定义；F1a 需要 self+random 双侧方差，
在只有 seed 1/2/3 random 的资产下无法 k≥5，降描述性。

### 2.4 warm start 的双重混淆（stats-eval-08 / training-ml-04 / engineering-ops-09）

`src/seven523/train.py`：

- `:557` `optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)`
  —— 每次运行新建，warm start 不继承 Adam 动量/二阶矩；
- `:706` `frac = 1.0 - (update - 1.0) / num_updates` —— 按**新的** `total_timesteps` 重算，
  l1_1M 是退火到 ~0 的 ckpt，warm start 会把它重新拉到 lr=2.5e-4 再退火。

修复：Stage 2 增 `t17w1wsctrl`（同 `--load-checkpoint l1_1M`、同 500k、对手 random），
归因端点改成 `ws − wsctrl`。

### 2.5 Block C 的 `--num-envs 16` 是复合因子（training-ml-05 / engineering-ops-06）

默认 8×128=1024 steps/update；`num_updates = 200000 // 1024 = 195`（`train.py:677`）；
minibatch = `batch // num_minibatches = 1024//4 = 256`（`ppo.py:174`）。
`--num-envs 16` → batch 2048、num_updates 97、minibatch 512，且 `make_env(seed+idx)` 把
env 8–15 的牌流换成 seed 9–16（`train.py:422`）→ 至少四重混淆。修复：删除该臂，只留 hidden 单因子。

### 2.6 自博弈末熵（training-ml-10）

`runs/t17self__1__1790439615/metrics.csv` 末行：`entropy=1.54278`、`explained_variance=0.613814`、
`episodic_return=-0.0823529`、`episodic_length=20.8627`。
`runs/t17pool__1__1790439615/metrics.csv` 末行：`entropy=1.75463`、
`explained_variance=0.61084`、`episodic_return=0.504651`、`episodic_length=23.7442`。
→ self 更锐利（熵低 0.21），修订版把末熵/EV/return 列为必报观测量并设熵门。

### 2.7 `aggregate` 脚本口径（stats-eval-05 / training-ml-12 / engineering-ops-05）

- `src/seven523/duel.py:264-276`：跨 deal seed 的 `bootstrap_se = sqrt(mean(se^2))`（RMS），
  `se = max(bootstrap_se, between_sd)/sqrt(count)`。
- `runs/t17seq_train/aggregate_seq.py`：`Z=1.96`、路径 `runs/{prefix}_s{seed}.json`、
  `boot_se = sum(boot_se)/k`（算术均值），**与 repo 口径不一致**。
- `runs/t17event_train/aggregate_event.py`：有 t 分位数，但仍算术均值。

修复：§3.3 规格要求新 `aggregate.py` 读 `runs/t17w1/h2h_<pair>_sN.json`、用 RMS、
同时打印 RMS/算术均值/训练 seed sd/deal-seed sd/binding term、z 与 t 双口径。

### 2.8 `poll_snapshot.py` 前台阻塞（engineering-ops-08）

`runs/t17_train/poll_snapshot.py` 主循环 `while True: ... time.sleep(0.25)`，直到全部
target 捕获才返回。`--checkpoint-interval 500` 在 1M run（976 updates）下 `checkpoint.pt`
只会是 step 512000，`999424` 必须等 `agent.pt`（run 结束）。修复：`nohup ... &`。

### 2.9 A3 旧数不可比（training-ml-08）

`docs/experiments/structural-directions.md:3`：「本报告引用的模型 checkpoint 已删除，旧路径
不再可用；数值为牌型族规则变更前口径」。当前 ckpt 的 `explained_variance` 已 0.61–0.70。
修复：A3 判据改 ckpt 内部（早期桶 vs 整体 vs 末桶 + V sd/return sd），0.065 只作历史注脚。

---

## 3. 两条红队建议冲突的裁定

- `stats-eval-02` fix：「Demote F1b to a descriptive ... **Keep F1a as the only confirmatory endpoint**」。
- `training-ml-02` fix：「把 F1b（vs 固定 l1_1M）升为 confirmatory 主端点，F1a 降为描述性」。

**裁定**：两条 finding 的事实前提都成立，方向相反。采用如下自洽设计同时满足二者：

1. 采纳 `stats-eval-02` 的核心——**不得把一个 ckpt 的 deal-seed 重测当成训练 seed 族**，
   也不得把 `t_{0.975,2}` 套在无训练重复的 CI 上。
2. 采纳 `training-ml-02` 的端点选择——F1a 的方差 = self 侧 + random 侧；random 侧 sd
   ≈10–22（`event-history-pilot.md` §5.1/§5.3），而资产只有 random seed 1/2/3，F1a 做
   k≥5 需要额外 4 个 random run，不可行；F1b 只含 self 侧训练方差，是**可判性更高的固定参照端点**。
3. **同时满足**：F1b 的左端定义为 **7 个 fresh 训练 ckpt（seed 2–8）**，参照固定为 `l1_1M`。
   这消除了 `stats-eval-02` 指出的「单 ckpt 冒充训练 seed 族」问题（因为确实是 7 个不同
   训练 seed 的 ckpt），也满足 `training-ml-02` 的方差论证。
4. F1a 保留为**描述性**（N=1,2,3，fresh deal 10–18），仅作方向参照，不进 Bonferroni 族。

因此两条 findings 的实质关切都被解决，冲突消解。

---

## 4. 改动位置索引（`docs/v5-optimization-plan.md` revision-4）

| 改动 | 位置 |
|---|---|
| 版本头 + 红队修复说明 + 资源纪律 | 文件头 |
| 资产表新增 `l1_2M`/`lvl3`/weight 同源核对；`poll_snapshot` 标注后台 | §0.2 |
| 结论先行重写（self-play「未确立」；F1b 主端点；F0；A2 措辞；gamma） | §0.3 |
| `+16.62` 探索性、训练 seed sd≈10、F1a N=1 已存在、绝对锚缺口 | §1.2 |
| 功效表 k=3/5/7/9 + 80% power 目标 +20 | §1.4、§6 |
| A1 ①②对照③④⑤ 表；A1⑤ 去 self；PFSP 移第二波 | §2 |
| A2 删 `--num-envs`、措辞限定；A3 内部判据；A4/A5 | §2 |
| 排序表（Stage 1/2 标注） | §2 末 |
| `mkdir -p runs/t17w1` 预检 | §3.0 |
| deal seed 冻结 10–18、参照含 l1_2M/lvl3 | §3.1 |
| Block A=7 fresh self；Block C=3 hidden；Stage 2=Block B 含对照 | §3.2 |
| F0 绝对锚、F1b 主端点逐 seed 命令、F1a/F1c 描述性、F2 条件、aggregate 规格 | §3.3 |
| 判定门（含 F0 绝对退化门、binding-term 升级、F2 仅提示）+ M1/M2/M3 | §3.4 |
| 条件 Block D + 阶段顺序 + 预算核算（14.2） | §3.5–3.7 |
| 第二波命令补 `--device cpu --workers 4`；PFSP 归位 | §4 |
| 实现面、发布口径 | §5 |
| 统计纪律 8 条（CI 跨 0=未判定、独立 deal、熵门） | §6 |
| 分步/验收/回滚/风险（R2 双混淆、R10 轮询器） | §7 |

---

## 5. 残余风险 / 需要用户决策的项

**无遗留 blocker。** 以下 2 项按设计延后，执行时需未来授权：

1. **登记新梯级**：若第一波 self 确认，把候选登记为 `lvl5` 需改
   `tools/play_ladder.py::OPPONENTS` + `tests/test_play_ladder.py::EXPECTED_IDS`——属本工作流
   禁止的 tools/tests 改动。规划已写明「停下如实报告」。
2. **A3/A4/A5 的 src 改动**（更大 critic/return norm/PopArt/分布价值/n-step、top-k 集成、
   1-ply bot）：本轮只设计，不实现；需用户批准后才进入实现波。

**残余统计风险（已在规划预注册缓解，不需用户决策）**：
- 若真实 self 侧训练 seed sd >16，k=7 的 t-CI 仍可能跨 0 → 规划按 binding term 升级
  （加 seed 到 k=9 或加 deal 到 1500/seed），并强制写「未判定 ≠ 0」。
- `l1_1M` 是 13 候选 MLE argmax（winner's curse）→ 已加 F1c 对 `l1_2M` 描述性对照。
- 单 seed 臂的 `+20` 不写进摘要结论 → 已在 §3.4/§3.5 预注册。

---

## 6. 修复后自洽性检查

1. **第一波可执行**：`mkdir -p runs/t17w1` 已置首；所有命令用现有 CLI；F1b 左端 7 个 ckpt
   与 7 个 run 一一对应（seed 2–8）。
2. **预注册门无混淆**：唯一 confirmatory 端点 = F1b（k=7 × 9 deal × 400）；F0 是单端点绝对锚、
   不与 F1b 合并；F1a/F1c/F2 全描述性。Bonferroni 只覆盖 F1b（若加 +20 非劣则 2 个检验）。
3. **方差口径自洽**：跨 deal seed 用 `duel.py` 的 RMS；跨训练 seed 用 RMS/训练 seed sd 的 max；
   三分量与 binding term 强制打印。
4. **预算自洽**：Block A 7 + Block B 6 + Block C 1.2 = 14.2 ≤ 15；Stage 1 = 8.2，Stage 2 条件 6.0。
5. **资源纪律自洽**：训练 ≤3 并发；h2h 严格串行 `--device cpu --workers 4`；轮询器后台。
6. **契约自洽**：未改 `src/tests/tools/ADR/plans.md/README/CONTEXT/DESIGN`；发布需改梯级的
   情况明确「停下报告」。

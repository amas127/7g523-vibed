# 机器人阶梯实验报告（M2）

> 资产状态（2026-09-25）：本报告引用的模型 checkpoint 已删除，旧路径不再可用；数值为牌型族规则变更前口径。

> **状态：训练与评级完成；核心发现是「强度平台」。**
> 本报告记录 2026-09-25 在 `docs/human-elo-plan.md` §3.1 下训练 4 级 ckpt、用
> `tools/build_ladder.py` 定级、并冻结进 `traces/study/manifest.json` 的过程与结果。
> **问题：vs Greedy 的训练在 ~60k 步后饱和，1M 步的平台与 1M 步 self-play 都未突破
> ~1490 Elo；greedy 以上只有约 174 Elo 的实测空间。**
> 计划与验收标准见 [`../human-elo-plan.md`](../human-elo-plan.md)；D1 轨迹信号报告见
> [`trace-signal-report.md`](./trace-signal-report.md)。

> **勘误（2026-09-25 文档治理，不改下文原文）**：本报告所有绝对 Elo（lvl4 1489.4、平台
> ~1490）是旧调度（100 局/锚点、未修座位相位）的产物，**不得与平衡口径混比**；平台顶部
> 现行规范值见 [`README.md`](./README.md) §0.1（`base680k` 1443.3 ± 14.7；lvl4 复测
> 1429.7 ± 7.1，`ladder-rerating-paired.md` §2.1）。**§4 的破局路线多数已被后续报告
> 证伪或替代**（对手池/mix：`wave5-500k-report.md` §4.2、`structural-directions.md`
> §9.4；容量/超参：`activation-loss-sweep-report.md` §5、SD §9.1/§9.3；margin：由
> `human-elo-10-games-research.md` §5.3 定为轨迹先验在场时 σ×2），仅作背景；现行优先级
> 见 [`../plans.md`](../plans.md) §7 与 §3（T1–T8）。

## 1. 方法

三组 PPO 训练（`7g523-train`，2 家、默认超参：8 env × 128 steps、hidden 128、seed 1）：

| 运行 | 设置 | 用途 |
|---|---|---|
| `lvlbase` | 1M 步，`--opponent greedy`，从零 | 主曲线；1M 步内每 20 update（≈20,480 步）快照 |
| `lvllow` | 30k 步，`--opponent greedy`，从零 | 补低端：早期快照落在 random 与 greedy 之间 |
| `lvlsp` | 1M 步，`--opponent self`，warm start `lvlbase` final，`--self-play-refresh 25` | 试探能否突破平台 |

快照由外部轮询 `checkpoint.pt` 的 `extra.global_step` 复制（`--checkpoint-interval 20`），
训练结束后弃用轮询。**评级**：`tools/build_ladder.py --games-per-anchor 100 --no-traces`，
即每个候选对 random / greedy 两个固定锚点各 100 局（共 200 局，SE≈27–34），
锚点钉死 random=1000 / greedy=1315，默认胜负似然 BT-MAP。**复核**：`7g523-eval`
对 greedy 打 1000 局（固定 seat 0）。

经验教训（方法层）：`--games-per-anchor 20` 时 SE≈60–76，等级排序会被噪声打乱
（例如 s200k 探针 1522，100 局/锚点后 1436）；**每锚点 ≥100 局**才可用于选级。

## 2. 结果

### 2.1 主曲线（`lvlbase`，vs Greedy，200 局/点）

| 快照 | 训练步 | Elo ± SE |
|---|---|---|
| s20480 | 20k | 1243.6 ± 27.3 |
| s40960 | 40k | 1318.8 ± 28.3 |
| s61440 | 60k | 1347.1 ± 28.9 |
| s102400 | 100k | 1359.3 ± 29.2 |
| s143360 | 140k | 1413.7 ± 30.9 |
| s204800 | 200k | 1436.3 ± 31.8 |
| s245760 | 240k | 1445.1 ± 32.1 |
| s573440 | 560k | 1486.1 ± 34.1 |
| s696320 | 680k | **1489.4 ± 34.3**（全曲线最高） |
| s983040 | 960k | 1439.2 ± 31.9 |
| final | 999k | 1384.4 ± 29.9 |

- **~60k 步就到 1347**（从 greedy 1315 到最高点的 72% 路程），之后 90 万步都在
  **1380–1490** 的带内抖动；平台内各点差异 <2 SE，无法说 680k 的真水平高于 240k。
- `7g523-eval` 复核：final 对 greedy 1000 局 **+20.64 分差 / 63.7% 胜 / 7.5% 平**；
  s200k 为 +14.54 / 58.9%。胜率口径与 Elo 一致（64% ≈ +98 Elo）。

### 2.2 低端曲线（`lvllow`，vs Greedy）

| 快照 | 训练步 | Elo ± SE | 备注 |
|---|---|---|---|
| l4096 | 4k | 548.8 ± 60.0 | 早期 PPO 远差于随机 |
| l6144 | 6k | 832.1 ± 33.9 | |
| l8192 | 8k | 1036.6 ± 27.7 | ≈ random |
| l10240 | 10k | 1045.4 ± 27.6 | |
| l12288 | 12k | 1119.9 ± 27.0 | **lvl1** |
| l14336 | 14k | 1239.3 ± 27.2 | 12k→14k 一次跨过 ~120 Elo |
| l20480 | 20k | 1267.3 ± 27.5 | |
| l28672 | 28k | 1252.2 ± 27.4 | 与 20k 无显著差 |

可用低端等级只有两个点：~1120（12k）与 ~1250（≥14k）；再往上是 greedy 1315。

### 2.3 Self-play（`lvlsp`，warm start 自 base final）

| 快照 | self-play 步 | Elo ± SE |
|---|---|---|
| sp20480 | 20k | 1387.0 ± 30.0 |
| sp40960 | 40k | **1476.2 ± 33.6**（self-play 最高） |
| sp61440 | 60k | 1384.4 ± 29.9 |
| sp122880 | 120k | 1445.1 ± 32.1 |
| sp327680 | 330k | 1451.1 ± 32.4 |
| sp409600 | 410k | 1460.3 ± 32.8 |
| sp737280 | 740k | 1454.2 ± 32.5 |
| spfinal | 999k | 1374.2 ± 29.6 |

**全程 1375–1476，未超过 base 的 1489**；sp40960 与 s696320 在 ±34 噪声内不可分。
训练末期对冻结自己的回报已 ≈0 / 略负（`ep_return` −0.02 ~ −0.10），即进入了
自我镜像的平衡态；这与 DESIGN 早先记录的现象一致（stage1 +21.3 → 150k self-play +19.2）。
**结论：该 self-play 配方不产生更高的对外强度，不用于阶梯。**

### 2.4 冻结梯级（正式评级，落轨迹 + manifest）

| 等级 | 来源 | Elo ± SE | 到上一级间距 |
|---|---|---|---|
| random | 锚点/基线 | 1000（钉死） | — |
| lvl1 | `lvllow` @12k → `runs/lvl1/agent.pt` | 1119.9 ± 27.0 | 120 |
| lvl2 | `lvlbase` @20k → `runs/lvl2/agent.pt` | 1232.9 ± 27.2 | 113 |
| lvl3 | `lvlbase` @100k → `runs/lvl3/agent.pt` | 1359.3 ± 29.2 | 126 |
| lvl4 | `lvlbase` @680k → `runs/lvl4/agent.pt` | 1489.4 ± 34.3 | 130 |

`rungs (4/4)`、`fit converged`、无 wide gap；greedy(1315) 仅作锚点，不作研究等级
（否则 1244→1315→1359 会出现 71/44 的过密间距）。轨迹 6 级 × 200 局已就位
（lvl1–4 新采 800 条；random/greedy 为 M1 的 200 条）。

## 3. 平台问题

**量化**：从 greedy(1315) 到最高实测点 1489 只有 **174 Elo**；可用「等级」实质上是
「训练了 12k 步 / 20k 步 / 100k 步 / 680k 步」的同族策略，而不是四种不同策略。
按胜负换算，要把顶推到 1600 需要对 greedy 约 **72–73% 胜率**（现在是 63.7%）。

**根因（已测事实）**：

1. 固定对手 GreedyBot：学习信号在 ~60k 步耗尽；继续训练只会更贴合 Greedy 的固定
   弱点，不带来对外强度。1M 步内曲线平。
2. 冻结快照 self-play 不破局：对手变强但目标仍是「战胜自己」，对固定锚点的胜率没有
   系统提升，且出现风格漂移（末期对外分差略降）。

**根因（推测，未验证）**：

1. 容量/超参：hidden 128、默认 PPO、每 batch 仅 1024 步；策略表达力可能不足以利用
   更深的手牌规划（如配合补牌顺序的长程分数控制）。
2. 奖励即分差：对 Greedy 的分差在策略上饱和，梯度不再指向更强打法。
3. 2 家 + 固定脚本对手的训练分布太窄，缺少多样化对手带来的探索压力。

**影响**：

- **D1（轨迹信号）**：5–6 级梯级的 Elo 间距成立（每级差 3.5–4.5 SE），采集可继续；
  但「平台内无 Elo 差」意味着 S1/S2 若要分辨平台内的真实强弱，需要比结果似然更细的
  信号——这正是 S2（value regret）值得测的理由，也应在报告里作为已知混杂说明。
- **D3（混合估计器）**：人机匹配的最高机器人约 1490；对更强真人的对局只有结果似然
  通道可用（计划已规定结果似然是最终权威）。阶梯的顶部两级的 SE（29/34）也限制了
  「±50 精度」目标下用机器人当参照的不确定度底噪。
- 标签口径：等级 Elo 是相对钉死锚点的测量值，平台的绝对高度不可跨环境比较。

## 4. 可尝试的破局路线（开放问题，未执行）

1. **对手池/联赛**：混训 Greedy + Random + 历史快照 + 当前策略采样版
   （`--self-play-sample True` 或按比例采样），制造多样性与探索压力。
2. **对打高等级自己**：从 lvl4 热启动，直接以 argmax 版的 lvl3/lvl4 为对手（而非镜像）。
3. **容量/超参**：hidden ≥256、增大 batch、ent-coef/学习率扫描；
   `--num-envs` 提高样本多样性。
4. **只改测量不改变强度**：`FitConfig.margin`（分差高斯似然）可降低评级的 SE，
   让平台内的真实差异（若存在）更早显形。
5. **接受平台并推进 D1/D3**：轨迹信号与价值网校准不依赖 >1500 的机器人；
   若 D1 的 `m_eff` 达标，D3 可在现有阶梯上落地。

## 5. 复现与产物

```bash
# 训练（三组；snapshot 轮询见正文说明）
uv run --group train 7g523-train --exp-name lvlbase --total-timesteps 1000000 \
    --opponent greedy --checkpoint-interval 20
uv run --group train 7g523-train --exp-name lvllow --total-timesteps 30000 \
    --opponent greedy --checkpoint-interval 2
uv run --group train 7g523-train --exp-name lvlsp --total-timesteps 1000000 \
    --opponent self --load-checkpoint runs/lvlbase__1__*/agent.pt \
    --self-play-refresh 25 --checkpoint-interval 20

# 评级（100 局/锚点，不落盘）
uv run --group train python tools/build_ladder.py \
    --candidate s20480=ckpt:runs/probe/base_step00020480.pt ... \
    --games-per-anchor 100 --no-traces --seed 0

# 正式冻结（落轨迹 + manifest）
uv run --group train python tools/build_ladder.py \
    --candidate lvl1=ckpt:runs/lvl1/agent.pt --candidate lvl2=ckpt:runs/lvl2/agent.pt \
    --candidate lvl3=ckpt:runs/lvl3/agent.pt --candidate lvl4=ckpt:runs/lvl4/agent.pt \
    --games-per-anchor 100 --seed 0 --rungs 4

# 复核
uv run --group train 7g523-eval --checkpoint runs/lvl4/agent.pt \
    --episodes 1000 --opponent greedy
```

产物（均 gitignore）：`runs/lvlbase__1__*/`、`runs/lvllow__1__*/`、`runs/lvlsp__1__*/`、
`runs/lvl{1..4}/agent.pt`、快照 `runs/probe{,_low,_sp}/`、
评级记录 `runs/probe_elo_100.txt` / `probe_low_elo_100.txt` / `probe_sp_elo_100.txt` /
`ladder_definitive.txt`、`traces/study/`（6 级 × 200 局 + 冻结 manifest）。

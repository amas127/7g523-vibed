# PPO 对齐审计：cleanrl 参考实现 vs `src/seven523`

> 资产状态（2026-09-25）：本报告引用的模型 checkpoint 已删除，旧路径不再可用；数值为牌型族规则变更前口径。

> **状态：只读审计完成（2026-09-25）；P0 修复已实施（同日，工作区改动，未 commit）。**
> 审计对象：`ppo-implementation-details/ppo.py`（Costa Huang《The 37 Implementation
> Details of PPO》，gym 0.21.0，见 `ppo-implementation-details/pyproject.toml`）、
> ADR-0003 指定的主移植源 `ppo_multidiscrete_mask.py`，以及只在通用细节上作补充的
> `ppo_atari.py`。我们侧：`src/seven523/{ppo,train,networks,env,actions}.py`。
> **结论：PPO 更新数学、GAE、超参默认、初始化、masked categorical 与指定源逐行等价；
> 未发现能单独解释 ~1490 Elo 平台的算法级缺陷。** 发现 1 个真实机制偏差（gymnasium
> `NEXT_STEP` 自动重置产生的「死样本」，中低危）、5 个有意或低危偏差、1 个 trivial
> 缺失；其余全部对齐。本报告不修改任何代码与 `runs/`。

与 [`ladder-report.md`](./ladder-report.md) 对照阅读：那份报告把平台归因于「固定对手
+ 稀疏终局奖励 + 分布过窄」；本次审计为这个判断补上「PPO 移植本身不是主要嫌疑」的
证据，并给出一个低成本修复项。

## 1. 摘要

| 判定 | 项数 | 说明 |
|---|---|---|
| `ALIGNED` | 32 | 更新数学、GAE、初始化、mask、超参与运行时分支逐项一致 |
| `DIVERGENT` | 5 | 1 个机制性（autoreset，中）；4 个对 `ppo.py` 有意偏差（低，均对齐 ADR 指定源或 ADR-0004） |
| `MISSING` | 1 | `action_space.seed` / `observation_space.seed`（trivial，无行为影响） |
| `NA` | 3 | Atari 观测 `/255`、reward 裁剪/归一化、truncation bootstrap（当前不可达） |

**最关键的 3 个发现**

1. **`train.py:395` 用 gymnasium 默认 `NEXT_STEP` 自动重置，与参考脚本的运行时语义不同**
   （参考固定 gym 0.21.0），每个 episode 会向 rollout buffer 注入 1 个「死样本」：
   动作被 reset 吞掉、reward=0、done=0，却和下一局的开局状态连成一条 GAE 转移。
   按 `lvlbase` 的 `episodes/episodic_length` 估算约 **3.9% 样本**。严重度：中（数据污染），
   对平台解释力：低（终局 bootstrap 被 `done` 屏蔽，V(terminal) 不参与 GAE）。
2. **`target_kl` 在全部 17 份 `runs/*/args.json` 中为 `null`**，早停分支永不触发
   （`ppo.py:206-207`；参考默认同样是 `None`）。但实测 `approx_kl ≤ 0.0039`，远低于任何
   常用阈值（0.01/0.03），即使打开也不会改变训练轨迹 —— **不能解释平台**。
3. **masked categorical 的数值处理与 cleanrl 完全一致**：`ppo_multidiscrete_mask.py:118`
   的 `torch.where(mask, logits, -1e8)` 与 `networks.py:60-63` 逐字对应，我们额外把常量
   放到 `logits.device`（更稳）。`tests/test_train.py:26-45` 断言「从不采非法动作」与
   「全 False 头仍有限」。没有 `-inf`/溢出的隐藏风险。

## 2. 审计方法

- **主参考**：`ppo-implementation-details/ppo.py`（322 行）完整通读；逐项提取 PPO 细节。
- **ADR-0003 指定移植源**：`ppo_multidiscrete_mask.py`（因为本项目是 `MultiDiscrete` +
  action mask）。算法主体与 `ppo.py` 相同，差异只在网络（CNN+ReLU vs MLP+Tanh）、
  `clip-coef` 默认（0.1 vs 0.2）与 `num-envs` 默认（8 vs 4）。
- **补充**：`ppo_atari.py` 只用于确认与离散/CNN 无关的通用细节（obs `/255`、reward
  裁剪、ReLU、共享 trunk）。
- **运行时核对**：截至审计时仓库内 **17 份** `runs/*/args.json`（另一后台 agent 正在
  继续新增 `ebo_*` 实验，计数是快照）。
- **行为验证**：在 `.venv`（gymnasium 1.3.0）里实测 `NEXT_STEP` 与 `SAME_STEP` 的
  autoreset 差异；跑 `tests/test_ppo.py tests/test_train.py tests/test_env.py`（45 passed）。

## 3. 逐项对照表

> 参考列：`R:` = `ppo-implementation-details/ppo.py`；`M:` = `ppo_multidiscrete_mask.py`；
> `A:` = `ppo_atari.py`。我们列均为 `src/seven523/`。严重度只对 DIVERGENT/MISSING 有意义。

### A. 网络与初始化

| # | 细节 | 参考 | 我们 | 判定 | 严重度 |
|---|---|---|---|---|---|
| A1 | 正交初始化 `std=√2`、bias=0 | `R:95-98` | `networks.py:34-38` | `ALIGNED` | — |
| A2 | actor head `std=0.01` | `R:116`（`M:143`） | `networks.py:94` | `ALIGNED` | — |
| A3 | critic head `std=1.0` | `R:109`（`M:144`） | `networks.py:95` | `ALIGNED` | — |
| A4 | 初始动作 logits：bias=0 → 初始策略近似均匀 | `R:96,116` | `networks.py:34-38,94` | `ALIGNED` | — |
| A5 | 激活/拓扑：`ppo.py` 是 Tanh+actor/critic 双塔；我们 ReLU+共享 trunk | `R:104-116` | `networks.py:88-95` | `DIVERGENT`（对 `ppo.py`；对 ADR 指定源 `M:130-144` 逐项对齐，有意） | 低 |
| A6 | masked categorical：`-1e8` 屏蔽、熵只对合法项求和 | `M:111-126` | `networks.py:41-70` | `ALIGNED` | — |
| A7 | `nvec` 多头：split + 逐头 logprob/entropy 求和 | `M:152-161` | `networks.py:114-135` | `ALIGNED` | — |
| A8 | 花色偏好头恒为全 True mask（不是合法性头） | `M:131`（所有头都是合法性掩码） | `actions.py:107-123`、`env.py:221-226` | `DIVERGENT`（有意，ADR-0004） | 低 |

### B. 优化器与超参

| # | 细节 | 参考 | 我们 | 判定 | 严重度 |
|---|---|---|---|---|---|
| B1 | Adam `eps=1e-5` | `R:166`（`M:190`） | `train.py:344` | `ALIGNED` | — |
| B2 | Adam `lr=2.5e-4` | `R:23` | `train.py:79` | `ALIGNED` | — |
| B3 | LR 线性退火 `frac = 1-(update-1)/num_updates` | `R:185-188` | `train.py:443-445` | `ALIGNED` | — |
| B4 | 全局梯度裁剪 0.5 | `R:298` | `ppo.py:203` | `ALIGNED` | — |
| B5 | `update_epochs=4` | `R:57` | `train.py:108` | `ALIGNED` | — |
| B6 | `num_minibatches=4`、minibatch=batch/4 | `R:55,75` | `train.py:107`、`ppo.py:144` | `ALIGNED` | — |
| B7 | `clip_coef` 默认 0.1（`ppo.py` 是 0.2） | `R:61` vs `M:62` | `train.py:112` | `DIVERGENT`（对 `ppo.py`；对指定源 `M:62` 对齐，有意） | 低 |
| B8 | advantage 按 minibatch 归一化 | `R:269-271` | `ppo.py:170-175` | `ALIGNED` | — |
| B9 | `clip_vloss`：max(unclipped, clipped)、系数 0.5 | `R:279-291` | `ppo.py:184-196` | `ALIGNED` | — |
| B10 | entropy 系数 0.01 | `R:65,293` | `train.py:116`、`ppo.py:198-199` | `ALIGNED` | — |
| B11 | `vf_coef=0.5` | `R:67,294` | `train.py:117`、`ppo.py:199` | `ALIGNED` | — |
| B12 | target-KL 早停（epoch 末检查） | `R:301-302` | `ppo.py:206-207` | `ALIGNED`（实现存在；运行时 `null` 不触发，与参考默认一致） | — |
| B13 | 每 epoch `np.random.shuffle` + 顺序切 minibatch | `R:250-257` | `ppo.py:145-152` | `ALIGNED` | — |
| B14 | `num_envs` 默认 8（`ppo.py` 是 4） | `R:43` vs `M:44` | `train.py:99` | `DIVERGENT`（对 `ppo.py`；对指定源/Atari 对齐，有意） | 低 |

### C. 数据收集、GAE 与自举

| # | 细节 | 参考 | 我们 | 判定 | 严重度 |
|---|---|---|---|---|---|
| C1 | buffer 形状 `(steps, envs, …)` | `R:169-174`（`M:204-209`） | `train.py:422-431` | `ALIGNED` | — |
| C2 | 采样动作 + `values[step]=value.flatten()` + 存 logprob | `R:196-200` | `train.py:464-471` | `ALIGNED` | — |
| C3 | `dones[step]` 存「进入该步之前」的 done，GAE 用 `dones[t+1]` | `R:192-194,225` | `train.py:451-452`、`ppo.py:116` | `ALIGNED` | — |
| C4 | GAE 递推与终止 bootstrap 屏蔽 | `R:217-229` | `ppo.py:90-123` | `ALIGNED` | — |
| C5 | 非 GAE 分支（折扣 return） | `R:231-240` | `ppo.py:125-134` | `ALIGNED` | — |
| C6 | flatten + `actions.long()[mb].T` replay（逐头 logprob） | `R:242-248,268-271` | `ppo.py:52-87,154-159` | `ALIGNED` | — |
| C7 | autoreset 语义 | gym 0.21（同 step reset，cleanrl 依赖） | gymnasium 1.3 `NEXT_STEP`（`train.py:395`） | `DIVERGENT` | **中** |
| C8 | truncation 处理 | 0.21 不区分 term/trunc | `env.py:204` 恒 `False`；`train.py:484` 并入 done | `NA`（潜在风险，见 §6.3） | — |

### D. 环境适配与观测/奖励

| # | 细节 | 参考 | 我们 | 判定 | 严重度 |
|---|---|---|---|---|---|
| D1 | 向量化换成 `gymnasium.vector.SyncVectorEnv` | ADR-0003 契约 | `train.py:395` | `ALIGNED`（契约替换） | — |
| D2 | 网络换成 MLP | ADR-0003 契约 | `networks.py:88-95` | `ALIGNED`（契约替换） | — |
| D3 | 环境换成 `Seven523Env` | ADR-0003 契约 | `train.py:195-208`、`env.py` | `ALIGNED`（契约替换） | — |
| D4 | 观测归一化（Atari `x/255`） | `A:136,139` | 观测编码即落在 `[0,1]`（`env.py:54-105,128-137`） | `NA` | — |
| D5 | reward 缩放/裁剪（`ClipRewardEnv`、`NormalizeReward`） | `A:100`、`ppo_procgen.py:193-194` | reward=`score/total-…` ∈ `[-1,1]`（`game.py:269-280`） | `NA` | — |
| D6 | reset 播种 `envs.reset(seed=…)` | `R:179`（旧 API 不传 seed） | `train.py:430` | `ALIGNED`（我们更严格） | — |
| D7 | `action_space.seed` / `observation_space.seed` | `R:83-85` | 未调用 | `MISSING` | 低（无行为影响） |
| D8 | `RecordEpisodeStatistics` + episode 日志 | `R:80,210-214` | `train.py:208,487-495` | `ALIGNED`（我们按 finished 掩码逐 env 记录，更稳） | — |

### E. 日志与指标

| # | 细节 | 参考 | 我们 | 判定 | 严重度 |
|---|---|---|---|---|---|
| E1 | `old_approx_kl` / `approx_kl` / `clipfrac` 公式 | `R:265-267` | `ppo.py:164-168` | `ALIGNED` | — |
| E2 | `explained_variance`（全 batch） | `R:305-307` | `ppo.py:209-211` | `ALIGNED` | — |
| E3 | 记录的 loss 取自最后一个 minibatch（参考固有性质） | `R:293-294,310-317` | `ppo.py:198,213-220` | `ALIGNED` | — |

**计数：`ALIGNED` 32，`DIVERGENT` 5，`MISSING` 1，`NA` 3。**

## 4. DIVERGENT / MISSING 详析

### 4.1 C7（中）：`NEXT_STEP` 自动重置把「死动作」变成训练样本

**机制（经 gymnasium 1.3 源码与实测确认）**：

- `train.py:395` 的 `gym.vector.SyncVectorEnv([...])` 使用 gymnasium 默认
  `AutoresetMode.NEXT_STEP`。
- `NEXT_STEP` 语义：done 的当步返回**终局 obs**（reward、done 正常）；**下一步**先
  `reset()`、**丢弃该步传入的 action**、返回新局初始 obs、`reward=0`、`done=False`。
- 参考脚本固定 gym 0.21.0（`poetry.lock:660-661`），其 `SyncVectorEnv` 在 done 当步即
  reset 并返回新局 obs（等价 gymnasium 的 `SAME_STEP`）；CleanRL 依赖该行为，因为
  `ppo.py:179` 只在训练开始时 `reset()` 一次。

**后果**：每个 episode 在 buffer 里多出 1 条 `(s_T, a_dead, r=0, done=0, next=s'_0)`
的转移（`train.py:451-486` 如实存入）：

- `dones[t+2] = 0`（`ppo.py:116`），所以该样本的 GAE 会**跨局**接力到新 episode 的
  reward（chimera target），把 `V(终局状态)` 往 `γ·V(新局开局)` 拉。
- `s_T` 是终局 obs，本来不会作为决策输入；这条样本给共享 trunk 注入一个无意义的
  策略梯度。
- 终局 transition 自身的 bootstrap 被 `done` 屏蔽（`nextnonterminal=0`），所以
  `V(s_T)` 不参与任何真实 GAE —— 这是为什么影响有限。

**量化**：`lvlbase` 终局 `episodes=39022` / `global_step=999424` ≈ **3.9% 的 rollout
样本**。用 `.venv` 实测（gymnasium 1.3.0，`Seven523Env`）：

```
=== NEXT_STEP : t=25 done=True (action applied, 终局 reward=0.300)
                t=26 DEAD STEP (action discarded, reward=0.000, done=False)
=== SAME_STEP : t=25 done=True (action applied；无后续死步)
```

**最小修复（已实施，2026-09-25；工作区改动，未 commit）**：

- `train.py:421`：给 `SyncVectorEnv` 显式指定 `autoreset_mode=gym.vector.AutoresetMode.SAME_STEP`。
- **实施时发现审计遗漏的配套项**：gymnasium 在 `SAME_STEP` 下把 done 步的
  `RecordEpisodeStatistics` 信息挪到 `infos["final_info"]["episode"]`（`NEXT_STEP` 才在
  `infos["episode"]`）。训练循环原本只读后者，直接切换会丢掉全部 episode 统计
  （既有 `test_train_smoke_with_eval` / `test_train_writes_tensorboard_scalars` 会挂）。
  已加 `train.py:213` 的 `_episode_info()` 双向适配，循环在 `train.py:509` 使用。
- 回归测试 `tests/test_train.py:182` `test_train_uses_same_step_autoreset`：计数真实
  `env.step` 调用，断言 == `num_steps × num_envs × num_updates`（`NEXT_STEP` 下每个
  episode 会少一条死步，断言必挂）。完整测试套件 234 passed。

**与平台的关系（谨慎）**：低。它污染约 4% 样本、增大终局价值噪声，但 `V(终局)` 从不
被 bootstrap 使用，且 900k 步的平台在训练早期（~60k）就出现。修复的价值是「让数据
分布与参考完全一致、去掉一个确定性污染源」，不是「预期单点破平台」。

### 4.2 A5/A8/B7/B14（低）：对 `ppo.py` 的有意偏差，均对齐 ADR 指定源

| 项 | `ppo.py`（参考） | 我们 / ADR 指定源 | 行为影响 |
|---|---|---|---|
| A5 | Tanh；actor/critic 两个独立 MLP | ReLU；共享 trunk + 两个 head（`M:130-144` 同样共享 CNN trunk） | 参数更少、初始化范围不同；属「网络用 MLP」契约内选择。参考的二者都常见，无实质算法差 |
| A8 | 每个头都是合法性掩码 | 第 2 个头（花色）是偏好头，恒开；不合法花色回退到最强实现（ADR-0004） | 花色头的 logprob/entropy 进入联合 ratio 与 entropy 奖励；多个花色值可能映射到同一实际出牌。属动作空间设计，不是 PPO 走样 |
| B7 | `clip-coef=0.1`（**注：`ppo.py` 是 0.2**，`M` 是 0.1） | 0.1 | 晚期实测 `clipfrac≈0–0.02`，0.1/0.2 均不 binding；早期（40k）clipfrac≈0.11，0.1 略保守 |
| B14 | `num-envs=8`（**注：`ppo.py` 是 4**） | 8 | batch 1024 vs 512，样本效率/吞吐差异，非算法差 |

### 4.3 D7（低）：缺 `action_space.seed` / `observation_space.seed`

参考在 `make_env` 里对三个空间都播种（`R:83-85`）。我们的动作由 agent 采样、
`MultiDiscrete` 空间本身不被 `sample()`；环境 RNG 在 `reset(seed=args.seed)`（`train.py:430`）
与 `Seven523Env(seed=seed+idx)`（`train.py:195-207`）播种。无行为影响，补上是锦上添花。

## 5. 「声明 vs 实际」核对

### 5.1 ADR-0003 承诺清单落点

| ADR-0003 承诺 | 代码位置 | 运行时（17 份 `args.json`） | 判定 |
|---|---|---|---|
| masked categorical | `networks.py:41-70,116-118` | 所有 run 均使用；`MultiDiscrete([134,4])` | ✅ 落实 |
| `nvec` 多头 | `actions.py:97-104`、`networks.py:114-135`、`env.py:172` | `nvec=[134,4]`，`sum=138` mask | ✅ 落实 |
| GAE | `ppo.py:90-123`、`train.py:500-510` | `gae=true` 17/17 | ✅ 落实 |
| LR 退火 | `train.py:443-445` | `anneal_lr=true` 16/17（1 个对照 `false`） | ✅ 落实 |
| advantage 归一化 | `ppo.py:170-175` | `norm_adv=true` 17/17 | ✅ 落实 |
| clip-vloss | `ppo.py:184-196` | `clip_vloss=true` 17/17 | ✅ 落实 |
| entropy 系数 | `ppo.py:198-199` | `ent_coef=0.01` 15/17（另有 0.03、0.0 各 1） | ✅ 落实 |
| target-KL | `ppo.py:206-207` | **`target_kl=null` 17/17 → 分支永不触发** | ✅ 代码落实 / ⚠️ 运行时该保护未启用（参考默认亦然） |

其余运行时快照：`clip_coef=0.1` 17/17、`num_envs=8` 16/17（1 个 16）、
`num_steps=128`、`num_minibatches=4`、`update_epochs=4`、`learning_rate=2.5e-4`、
`gamma=0.99`、`gae_lambda=0.95`、`vf_coef=0.5`、`max_grad_norm=0.5` 全部 17/17。
注意：这些实验由另一后台 agent 持续新增，计数是审计时刻快照。

### 5.2 masked categorical 数值稳定性专项

| 方面 | cleanrl（`ppo_multidiscrete_mask.py`） | 我们（`networks.py`） |
|---|---|---|
| mask 类型转换 | `masks.type(torch.BoolTensor).to(device)`（`:117`） | `masks.to(torch.bool)`（`:59`） |
| 非法 logits | `torch.where(self.masks, logits, torch.tensor(-1e8).to(device))`（`:118`） | `torch.where(masks, logits, torch.tensor(-1e8, device=logits.device))`（`:60-63`） |
| entropy | `p_log_p = logits*probs`；非法项置 0（`:124-125`） | 同（`:68-70`） |
| 全 False 头 | softmax(-1e8)=均匀，sample 有限；熵=0 | 完全一致（`tests/test_train.py:38-45` 覆盖） |
| 溢出/NaN | float32 下 softmax 减 max，无溢出；非法项 logprob≈-1e8 有限 | 同；额外把常量放对 device |

**结论：`ALIGNED`，无 cleanrl `-1e8` vs `-inf` 之类的隐藏走样。** 唯一差别是我们把
`-1e8` 常量放到 `logits.device`，避免 0 维 CPU/GPU tensor 混用。当前环境中全 False
头不可达（模板头至少 PASS 或可出牌；花色头恒开）。

## 6. 与 ~1490 Elo 平台的可能关联

### 6.1 平台期训练诊断（`runs/lvlbase__1__1790313091/metrics.csv` 抽样）

| step | ep_return | value_loss | entropy | approx_kl | clipfrac | explained_var | lr |
|---|---|---|---|---|---|---|---|
| 1,024 | -0.434 | 0.093 | 2.502 | 0.00061 | 0.0002 | -0.19 | 2.5e-4 |
| 40,960 | -0.192 | 0.014 | 2.141 | 0.00296 | 0.109 | 0.62 | 2.4e-4 |
| 61,440 | -0.025 | 0.016 | 2.096 | 0.00363 | 0.091 | 0.67 | 2.35e-4 |
| 204,800 | +0.223 | 0.022 | 1.657 | 0.00181 | 0.046 | 0.59 | 1.99e-4 |
| 696,320 | +0.290 | 0.017 | 1.353 | 0.00159 | 0.019 | 0.60 | 7.6e-5 |
| 997,376 | +0.186 | 0.019 | 1.394 | ~3e-7 | 0 | 0.66 | 7.7e-7 |

诊断结论：

- **没有 PPO 失稳的迹象**：`approx_kl` 全程 ≤0.0039、晚期 clipfrac≈0；explained
  variance 0.6–0.69 健康。这不是「KL 爆炸 / 比率走样 / 价值网崩坏」型平台。
- **target-KL 关闭无解释力**：即便把 `target_kl` 设为 0.01，按实测 KL 也几乎不会触发；
  参考默认同样是 `None`。这是「声明存在但未启用」，不是算法走样。
- **clip_coef 0.1 vs 0.2 解释力低**：平台期 clipfrac 已 <2%，不 binding；40k 时
  clipfrac 0.11 说明早期它确实在起作用，但曲线在 60k 已进平台。
- **LR 退火是平台期「缺乏突破动力」的结构性因素**：60k 时 LR 仍有 2.35e-4，平台不是
  退火直接造成；但 1M 步内 LR 退到 ~1e-6，后 90% 训练基本不再探索。可做
  `--anneal-lr False` 或分段退火对照（属超参实验，非移植问题）。
- **熵衰减到 1.39**（联合熵上限 `ln134+ln4≈6.28`），策略接近确定性；与固定 Greedy
  对手叠加，ent_coef=0.01 的熵奖励不足以阻止确定性化，探索压力持续下降。与参考设置
  一致，问题在训练分布而非移植。

### 6.2 逐项与平台的关联度（不夸大）

| 发现 | 严重度 | 对平台的可能解释力 | 理由 |
|---|---|---|---|
| C7 死样本（§4.1） | 中 | **低** | 4% 样本污染，但 V(终局) 不参与 bootstrap；最多轻微扰动终局价值与表征 |
| target_kl 未启用 | 低 | 无 | 实测 KL 远低于阈值，开关都不改变轨迹 |
| clip_coef=0.1 | 低 | 低 | 平台期不 binding |
| 网络 ReLU/共享 trunk | 低 | 未知/低 | 表达力问题属容量假设（ladder-report §3 推测），非移植走样 |
| 花色偏好头恒开 | 低 | 低 | 只影响 tie-break 的表示，不影响合法性 |
| LR 全程退火到 0 | 非移植问题 | 中（结构性） | 后 900k 步几乎没有探索动力；值得对照实验 |
| 稀疏终局奖励 + 固定对手 | 非移植问题 | 中（ladder-report 已论证） | 学习信号在 60k 耗尽；参考移植无 dense reward 可对齐 |

### 6.3 附带发现（非对齐问题，但值得记录）

- **truncation 潜在风险**：`train.py:484` 把 `termination | truncation` 一起当 done。
  当前 `env.py:204` 恒返回 `truncation=False`（回合只由牌局自然结束），所以不可达；
  若未来加 `TimeLimit`，截断步应 bootstrap `next_value` 而不是置零 —— 现在不是 bug。
- **文档漂移**：ADR-0003「后果」一节仍写 `MultiDiscrete([134])`，ADR-0004 已把动作空间
  改为 `[134, 4]`（代码与 `env.py:172`、测试一致）。
- **divisibility 隐患**：`ppo.py:144` 用整除切 minibatch，若 batch 不能整除则静默丢弃
  尾部样本（参考同样如此）；默认 1024/4 无影响，可加断言。

## 7. 修复优先级建议

| 优先级 | 动作 | 位置 | 预期收益 |
|---|---|---|---|
| **P0** | ✅ **已实施**（2026-09-25）：`train.py:421` 指定 `SAME_STEP`；`train.py:213` 加 `final_info.episode` 适配（实施时发现，见 §4.1）；`tests/test_train.py:182` 回归测试 | `train.py:421,213`、`tests/test_train.py:182` | 去掉 ~3.9% chimera 样本，回正到参考运行时语义；完整套件 234 passed |
| P1 | 若要保留 `NEXT_STEP`：在 rollout 收集时标记 reset 步并在 `ppo_update` 前剔除该样本（改动更大，不推荐） | `train.py:451-486`、`ppo.py:137+` | 同上，但复杂 |
| P1 | 加 `assert args.batch_size % args.num_minibatches == 0`；若引入 `TimeLimit`，把 truncation 与 termination 分开并 bootstrap | `train.py` | 防御性 |
| P2 | 更新 ADR-0003 的 `MultiDiscrete([134])` → `[134, 4]`；在 ADR 里补记 NEXT_STEP 死样本与 SAME_STEP 决议 | `docs/adr/0003-training-stack.md` | 文档与代码一致 |
| P3 | 平台实验（先做 P0）：`--anneal-lr False` / 分段退火、对手多样性（`--opponent pool/mix`）、奖励塑形、`hidden ≥256` 对照 | `train.py` CLI | 直接针对 ladder-report 的分布假设；期望收益高于 PPO 超参扫描 |

**综合判断**：把 ~1490 Elo 平台归因于「PPO 参考细节没有忠实移植」缺乏证据。移植在
算法层面是忠实的；唯一真实偏差是 autoreset 引入的 ~4% 死样本，值得先修，但不应期待
它单点破平台。平台更像是「固定对手 + 稀疏奖励 + 窄分布 + 全程退火」的组合，与
`ladder-report.md` §3–4 的结论一致。

## 8. 复现与证据

```bash
# 测试（本次 46 passed；完整套件 234 passed）
uv run --group train .venv/bin/python -m pytest -q -p no:cacheprovider \
    tests/test_ppo.py tests/test_train.py tests/test_env.py

# autoreset 行为实测（NEXT_STEP 死步 vs SAME_STEP 无死步）
.venv/bin/python - <<'PY'
import numpy as np, gymnasium as gym
from seven523.env import Seven523Env
from seven523.rules import DEFAULT_RULES
from seven523.policies import make_scripted_policies

calls = {"n": 0}
def make_env(seed):
    def thunk():
        env = Seven523Env(DEFAULT_RULES, make_scripted_policies("greedy", DEFAULT_RULES, seed), seed=seed)
        orig = env.step
        def counted(a):
            calls["n"] += 1
            return orig(a)
        env.step = counted
        return env
    return thunk

for mode in [gym.vector.AutoresetMode.NEXT_STEP, gym.vector.AutoresetMode.SAME_STEP]:
    envs = gym.vector.SyncVectorEnv([make_env(0)], autoreset_mode=mode)
    envs.reset(seed=0)
    print("=== ", mode)
    for t in range(45):
        mask = np.asarray([e.get_wrapper_attr("action_mask") for e in envs.envs])[0]
        legal = [i for i, b in enumerate(mask) if b]
        before = calls["n"]
        _, rew, term, trunc, _ = envs.step(np.array([[legal[0], 0]]))
        if term[0] or trunc[0] or before == calls["n"]:
            print(t, before, calls["n"], float(rew[0]), bool(term[0] or trunc[0]),
                  "DEAD" if before == calls["n"] else "real")
    envs.close()
PY

# 平台期指标抽样
python - <<'PY'
import csv
rows = list(csv.DictReader(open("runs/lvlbase__1__1790313091/metrics.csv")))
for r in rows:
    if int(r["global_step"]) in (1024, 40960, 61440, 204800, 696320, 997376):
        print(r["global_step"], r["episodic_return"], r["entropy"],
              r["approx_kl"], r["clipfrac"], r["explained_variance"], r["learning_rate"])
PY
```

**只读声明**：本审计未修改任何代码、未写入 `runs/`、未 commit；新增文件仅本报告。

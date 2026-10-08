# Stage-0 诊断 D-A：Oracle-obs 探针（隐藏信息是不是 raw 策略上界的瓶颈）

> 状态：**已完成**（预注册写于开跑前，见 §1；执行与读数见 §3–§6）。
> 本报告对应 [`docs/post-v5-structural-options.md`](./post-v5-structural-options.md) §4 D-A。
> 口径：revision-3（`rules_id=2e36dbea44893696`）、obs v5（2 家 161 维）、PPO memoryless
> MLP（`arch=shared`、`hidden=128`、单层 trunk＝2 个 Linear、`num_envs=8 × num_steps=128`、
> minibatch 256、`update_epochs=4`、lr 2.5e-4 线性退火）。
> 权限边界：本探针只在 `runs/stage0/oracle/` 写 run-local driver 与产物，不改 `src/`/`tests/`/
> `tools/`；不 commit；不发布 checkpoint/manifest；不改 `obs_version`；oracle 策略**不可部署**
> （用了非法信息），只测上界。

## 1. 预注册（写于开跑前，未改）

### 1.1 假设

obs v5 的 `unseen` 段只告诉模型「未见面牌的多重集」（54 − 自己手牌 − 已亮 − 已出 − 当前墩），
看不到对手手牌与牌堆余牌的**划分**。若把这段划分的真实隐藏信息作为 oracle 块拼进观测，
策略仍然拿不到 ≥ +10 Elo 的稳健增益，则说明瓶颈不在「信息缺口」，O1（belief 特征）先验应
大幅下调甚至放弃；若 oracle 拿到明确大增益，则说明「推断型新信息」是真杠杆，解锁 O1/belief。

### 1.2 臂定义

- **O（oracle 臂）**：obs v5（161 维）末尾追加 **108 维 oracle 块**；两臂网络首层输入因此都是
  269 维。训练与对局时 oracle 块均填**真实隐藏信息**。
- **Z（容量对照臂）**：同一包装、等参数量、等输入维度（269 维），oracle 块**恒为全 0**。
  唯一差别是「oracle 列是否携带信息」。

oracle 块确切布局（从行动座位视角，`card_id ∈ 0..53` 按 `make_deck()` 顺序：13 花色 ×4 后接
小王、大王）：

- 前 54 维：对手手牌 multi-hot。`k = 1..n-1`，座位 `(acting_seat + k) % n` 依次各占 54 维；
  2 家时只有 `(acting_seat + 1) % 2` 一个对手。
- 后 54 维：牌堆余牌 `state.draw_pile` multi-hot。

O 臂的该块每一维必须等于 `env.state`（`GameState`）的真实隐藏信息，见 §3 一致性测试 (b)。

### 1.3 主端点与判据（行动门槛）

主端点 = **h2h O vs Z**：对局时 O 给真实 oracle 输入、Z 给全 0 oracle 块，deal-twin 换座、
按训练 seed 配对。每个训练 seed 用 **3 个 deal-set seed × 400 副**（= 2400 局），3 个训练
seed 合计 **3600 副 / 7200 局**。

- 合并 z-CI 与 between-seed t-CI **同时排除 0**，且点估计 **≥ +10 Elo** → 信息是真杠杆，
  解锁 O1/belief。
- 合并 CI 跨 0，或点估计 < +10 → 信息不是主瓶颈（注明这是 1–2 run 级功效，只能排除大效应）。
- 点估计 **< +5** → 止损。

### 1.4 统计协议

deal-twin 换座配对 + 按副聚类的配对 bootstrap（`seven523.duel.paired_duel_stats` +
`combine_duel_seeds`，`bootstrap=4000`）。合并点估计 = 各 seed 均值；
`se = max(RMS(bootstrap SE), seed sd)/√k`；z-CI 用 `z=1.96`，t-CI 用 `t_{0.975,k-1}`
（k=3 → 4.303）。行动门槛以**合并**读数为准，同时报告 per-seed 读数。

### 1.5 分析计划

1. 跑 §3 的 run-local 一致性测试（环境侧逐位一致 + oracle 块逐维等于 `env.state`）。
2. 两臂从零训练 500k × 训练 seed {1,2,3}，同配方同对手（`train.py` 当前默认，
   仅 `--total-timesteps 500000`）。
3. h2h O vs Z：3 deal-set seed {0,1,2} × 400 副 × 3 训练 seed。
4. 复算合并 z/t CI，按 §1.3 判据给出结论；报告 `file:line` 或命令；结尾写缺失证据/限制。

### 1.6 完整命令（预注册）

训练（每臂每 seed 一条，共 6 条；训练 run 目录落在 `runs/stage0/oracle/runs/`）：

```bash
/home/amas/.local/src/7g523/.pi/limits/run_limited.sh gpu \
  /home/amas/.local/src/7g523/.venv/bin/python \
  runs/stage0/oracle/train_oracle.py --arm oracle --seed 1

# --arm zero --seed 1
# --arm oracle --seed 2
# --arm zero --seed 2
# --arm oracle --seed 3
# --arm zero --seed 3
```

h2h（每训练 seed 一条，deal-set seed 0,1,2 × 400 副，CPU）：

```bash
/home/amas/.local/src/7g523/.pi/limits/run_limited.sh cpu \
  /home/amas/.local/src/7g523/.venv/bin/python \
  runs/stage0/oracle/h2h_oracle.py --train-seed 1 --deal-seeds 0,1,2 --pairs 400
```

> amendment（接口命名，非设计变更）：h2h CLI 最终以 `--o <O agent.pt> --z <Z agent.pt>
> --deal-seeds 0,1,2 --pairs 400` 落地（见 §5）；端点/臂/判据均未变。

一致性测试：

```bash
/home/amas/.local/src/7g523/.pi/limits/run_limited.sh cpu \
  /home/amas/.local/src/7g523/.venv/bin/python \
  runs/stage0/oracle/test_consistency.py
```

训练配方（`train.py` 当前默认，唯一改动 `--total-timesteps 500000`）：
`--num-envs 8 --num-steps 128 --num-minibatches 4 --update-epochs 4`、
`--learning-rate 2.5e-4 --anneal-lr --lr-schedule linear --optimizer adam`、
`--gamma 0.99 --gae-lambda 0.95 --norm-adv --clip-coef 0.1 --clip-vloss`、
`--ent-coef 0.01 --vf-coef 0.5 --max-grad-norm 0.5`、`--hidden-size 128 --activation relu`、
`--arch shared`、`--opponent random`、`--reward-shaping terminal`、`--num-players 2`、
`--torch-deterministic`、`--cuda`。

---

## 2. 实现（run-local，未改 src/）

- `runs/stage0/oracle/oracle_common.py`：oracle 块定义
  （`encode_oracle_block`，`runs/stage0/oracle/oracle_common.py:32`）+
  `OracleEnv(Seven523Env)` 子类（`runs/stage0/oracle/oracle_common.py:51`），
  只覆盖 `__init__` 的 `observation_space` 与 `_publish` 追加 oracle 块。
- `runs/stage0/oracle/train_oracle.py`：monkeypatch `train.observation_dim` /
  `train.make_env` / `train.Seven523Env`（`runs/stage0/oracle/train_oracle.py:90-134`），
  调用原版 `train.train(parse_args(...))`（`runs/stage0/oracle/train_oracle.py:167`）。
- `runs/stage0/oracle/test_consistency.py`：一致性测试（§3）。
- `runs/stage0/oracle/h2h_oracle.py`：deal-twin 换座 + `paired_duel_stats`/`combine_duel_seeds`，
  自定义对局驱动使 O 臂可读 `Match.state`（`runs/stage0/oracle/h2h_oracle.py:96`），Z 臂置 0。

对 `src/` 的锚点核对：训练配方读 `src/seven523/train.py:601`（`obs_dim =
observation_dim(...)`）、`src/seven523/train.py:699`（`make_env(...)`）；
obs v5 段表 `src/seven523/env.py:223-244`；`unseen` writer `src/seven523/env.py:143-158`。

---

## 3. 一致性测试（run-local，通过）

命令：

```bash
/home/amas/.local/src/7g523/.pi/limits/run_limited.sh cpu \
  /home/amas/.local/src/7g523/.venv/bin/python \
  runs/stage0/oracle/test_consistency.py
```

输出（产物 `runs/stage0/oracle/consistency.json`）：

```text
episode 0: 28 steps checked (done=True)
episode 1: 26 steps checked (done=True)
episode 2: 29 steps checked (done=True)
episode 3: 31 steps checked (done=True)
episode 4: 30 steps checked (done=True)
episode 5: 29 steps checked (done=True)
episode 6: 28 steps checked (done=True)
episode 7: 29 steps checked (done=True)
{"status":"ok","episodes":8,"steps_checked":230,"obs_checks":230,
 "obs_dim":269,"base_obs_dim":161,"oracle_width":108}
```

断言覆盖（`runs/stage0/oracle/test_consistency.py:72-84`）：

- (a) 同 seed 下 O/Z 两臂每步 `state`、`action_mask`、base obs（前 161 维）逐位一致；
  8 局 × 230 步全部通过。
- (b) O 臂 oracle 块（后 108 维）逐维等于 `encode_oracle_block(env.state, learner, 2)`
  的真实隐藏信息；Z 臂 oracle 块逐维全 0；230 步全部通过。

---

## 4. 训练结果

`--total-timesteps 500000` 在 `batch_size=1024` 下得到 `num_updates=488`、实际步数
**499,712**（仓库既有口径：`2_000_000 → 1,999,872`，即 floor 到整 batch）。

| 臂 | seed | run_dir | 最终 episodic_return | 最终 sps | agent.pt obs_dim / obs_version |
|---|---|---|---|---|---|
| O | 1 | `runs/stage0/oracle/runs/dA_oracle__1__1791305141` | 0.525 | 1122 | 269 / 5 |
| Z | 1 | `runs/stage0/oracle/runs/dA_zero__1__1791305593` | 0.435 | 1065 | 269 / 5 |
| O | 2 | `runs/stage0/oracle/runs/dA_oracle__2__1791306070` | 0.541 | 1034 | 269 / 5 |
| Z | 2 | `runs/stage0/oracle/runs/dA_zero__2__1791306560` | 0.525 | 1030 | 269 / 5 |
| O | 3 | `runs/stage0/oracle/runs/dA_oracle__3__1791307057` | 0.593 | 699 | 269 / 5 |
| Z | 3 | `runs/stage0/oracle/runs/dA_zero__3__1791307782` | 0.511 | 1086 | 269 / 5 |

`obs_version=5` 未改动；`history_layout=mlp`、`hidden=128`、`arch=shared`。episodic_return
落在既有 500k 平台区间（≈0.4–0.6），未见 oracle 臂在训练回报上优于 Z 臂。

---

## 5. h2h 结果

每训练 seed 命令（以 seed 1 为例，其余同理）：

```bash
/home/amas/.local/src/7g523/.pi/limits/run_limited.sh cpu \
  /home/amas/.local/src/7g523/.venv/bin/python \
  runs/stage0/oracle/h2h_oracle.py \
  --o runs/stage0/oracle/runs/dA_oracle__1__1791305141/agent.pt \
  --z runs/stage0/oracle/runs/dA_zero__1__1791305593/agent.pt \
  --deal-seeds 0,1,2 --pairs 400 --out runs/stage0/oracle/h2h_s1.json
```

### 5.1 per-deal-seed 行（9 组，O 减 Z，正 = O 强）

| 训练 seed | deal-set seed | elo_diff | z-CI | winrate | W/D/L |
|---|---|---|---|---|---|
| 1 | 0 | +3.04 | [−19.56, +24.80] | 0.5044 | 372/63/365 |
| 1 | 1 | −16.52 | [−37.49, +4.78] | 0.4763 | 347/68/385 |
| 1 | 2 | −16.52 | [−39.25, +6.08] | 0.4763 | 345/72/383 |
| 2 | 0 | −5.65 | [−26.98, +16.52] | 0.4919 | 364/59/377 |
| 2 | 1 | −21.31 | [−44.10, +0.88] | 0.4694 | 349/53/398 |
| 2 | 2 | +13.03 | [−8.69, +34.43] | 0.5188 | 386/58/356 |
| 3 | 0 | −18.69 | [−41.45, +4.34] | 0.4731 | 342/73/385 |
| 3 | 1 | −29.60 | [−52.51, −7.82] | 0.4575 | 340/52/408 |
| 3 | 2 | −32.67 | [−55.18, −10.43] | 0.4531 | 330/65/405 |

### 5.2 per-训练-seed 合并（每个训练 seed 内 3 个 deal-set seed 合并）

| 训练 seed | 点估计 | z-CI | bootstrap SE | deal-set sd |
|---|---|---|---|---|
| 1 | −10.00 | [−22.77, +2.78] | 11.23 | 11.29 |
| 2 | −4.64 | [−24.10, +14.82] | 11.19 | 17.19 |
| 3 | −26.99 | [−40.00, −13.97] | 11.50 | 7.35 |

### 5.3 主合并（跨 3 个训练 seed，按训练 seed 配对）

对每个训练 seed 取 §5.2 的点估计与其 z-CI 隐含 SE，套用仓库合并规则
（`se = max(RMS(bootstrap SE), seed sd)/√k`）：

| 端点 | 值 |
|---|---|
| 点估计（O − Z） | **−13.87 Elo** |
| 合并 z-CI（z=1.96） | **[−27.08, −0.67]** |
| 合并 t-CI（t_{0.975,2}=4.303） | **[−42.86, +15.11]** |
| bootstrap SE（RMS 各训练 seed 隐含 SE） | 7.86 |
| between-训练-seed sd | 11.67 |
| 合并 winrate | 0.4801（训练-seed 口径 z-CI [0.4611, 0.4990]；9-seed 口径 z-CI [0.4661, 0.4940]） |
| 合并 mean score diff | −2.04（训练-seed 口径 z-CI [−3.76, −0.31]；9-seed 口径 z-CI [−3.44, −0.63]） |

备用口径：把 9 个（训练 seed, deal-set seed）当作 9 个独立 deal 集直接
`combine_duel_seeds`，点估计同为 **−13.87**，z-CI **[−23.60, −4.15]**、between-seed sd
14.88；方向与主合并一致（更负）。

---

## 6. 结论

按预注册判据（§1.3）：点估计 **−13.87 Elo**，**< +5 → 止损**；合并 z-CI 虽在负方向排除 0
（[−27.08, −0.67]），t-CI 跨 0（[−42.86, +15.11]），远未达到「≥ +10」的正门槛。

**判定：隐藏信息（对手手牌 / 牌堆余牌的划分）不是 500k 下 raw 策略上界的瓶颈；O1（belief 特征）
按 D-A 的决策读法应下调/放弃。** 3 个训练 seed 的点估计全为负（−10.00 / −4.64 / −26.99），
9 个 deal-set 行中 7 负 2 正，方向一致偏负——不是「差一点 +10」，而是 oracle 信息在本预算下
无增益、甚至小幅有害（尤其训练 seed 3 显著为负）。这更支持「当前平台是优化/表示限制，而非
单纯的观测信息限制」。

`file:line` 支撑：

- 训练配方读取：`src/seven523/train.py:601`、`src/seven523/train.py:699`。
- 合并统计实现：`src/seven523/duel.py:256-287`（`_combine_metric`）、`src/seven523/duel.py:288`
  （`combine_duel_seeds`）；主合并按同公式手工复算（见 §5.3，脚本输出于
  `runs/stage0/oracle/h2h_s*.json`）。
- h2h 对局驱动（O 读 `Match.state`，Z 置 0）：`runs/stage0/oracle/h2h_oracle.py:86-103`。

---

## 7. 缺失证据 / 限制

1. **功效口径**：这是 1–2 run 级探针，只能排除「oracle 带来大正效应（≥ +10）」；它**不能**
   证明 oracle 信息在更长预算/更强优化下无增益。若未来预算允许，可用 ≥7 训练 seed 或
   更长的 oracle 臂复算，但目前证据不支持优先投 O1。
2. **500k 预算固定**：oracle 臂首层输入 269 维、多了 108 维非零梯度，500k 未必学完；
   Z 臂 oracle 列恒 0，梯度恒 0。两者等参数量、等输入维度，但**非零梯度的参数面不同**，
   这是预注册口径内的混淆（任务把 Z 定义为 oracle 块全 0 的容量对照）。方向为负提示
   该混淆不是造成假阳性的来源。
3. **between-seed 噪声大**：k=3 下 seed sd 11.67，t-CI 很宽并跨 0；主结论依赖「点估计
   < +5」的止损判据与负向一致性，而非 t 显著性。
4. **训练 seed 3 的 sps 偏低（699）**：由同机资源竞争导致墙钟更慢，但训练步数与配方一致，
   不改变样本量或结论。
5. **oracle 不可部署**：探针用了非法信息，仅测上界；checkpoint 已留在 `runs/stage0/oracle/runs/`
   （gitignored），**不发布、不进入 ladder/rating 身份体系**，`obs_version` 保持 5。
6. **未跑 D-B/D-C/D-D**：本报告只覆盖 D-A；「优化 vs 信息」的完整归因仍需其他 Stage-0 诊断。

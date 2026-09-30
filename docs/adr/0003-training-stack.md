# 训练栈：gymnasium 环境 + 忠实移植的 cleanrl PPO

仓库自带的 `ppo-implementation-details/ppo_multidiscrete_mask.py` 依赖 gym 0.21（Python <3.10），且其网络是 MicroRTS 的 27 通道棋盘 CNN，与 191 维平坦观测不匹配——原定“无需修改即可训练”的目标不可达。

决定：训练栈用 **gymnasium 1.3 + torch**，在本仓库内忠实移植该脚本的算法与实现细节（masked categorical、`nvec` 多头、GAE、LR 退火、advantage 归一化、clip-vloss、entropy 系数、target-KL），只替换三处：向量化用 `gymnasium.vector.SyncVectorEnv`、网络用 MLP、环境用 `Seven523Env`。`ppo-implementation-details/` 降级为只读参考。

## 考虑过的替代

- **Python 3.9 降级环境 + 原版脚本**：能原样跑脚本，但要维护第二套解释器与锁文件，且 CNN 仍必须改；收益只剩“文件没被改”。
- **不引入 gymnasium，自写 vector loop**：少一个依赖，但丢掉了 `SyncVectorEnv` 的 next-step 自动重置、`RecordEpisodeStatistics` 与空间校验；用户已明确选择 gymnasium。

## 后果

- `Seven523Env` 是真正的 `gymnasium.Env`：5 元组 `step`、`MultiDiscrete([134, 4])`（ADR-0004 花色头）/ `Box(OBS_DIM,)` 空间；旧的 4 元组接口取消（`tests/test_env.py` 同步迁移）。
- torch 只在训练侧：`networks.py`（`Agent`/`NeuralPolicy`/checkpoint）、`train.py`（PPO）、`eval.py`（评估，torch 惰性导入）；`policies.py` 与规则层保持零 torch。
- 训练依赖在 `[dependency-groups] train`；`torch` 经 `[tool.uv.sources]` 指向 cu132 专用 index，换 CPU/其他 CUDA 只改这一处。
- 两阶段训练按 DESIGN §7：`--opponent random` 起步（GreedyBot 已随 [ADR-0012](./0012-single-gauge-and-greedy-removal.md) 退役），`--opponent self --load-checkpoint` 自博弈；冻结快照每 `--self-play-refresh` 次更新原地刷新（`NeuralPolicy.agent` 被共享引用）。
- `SyncVectorEnv` 使用 **`AutoresetMode.SAME_STEP`**（2026-09-25 修正）：对齐参考栈 gym 0.21 的同步 reset 语义。gymnasium 默认的 `NEXT_STEP` 会把 done 后的下一个动作吞掉却仍占一条 rollout（`lvlbase` 实测 ~3.9% 的“死样本”，见 `docs/experiments/ppo-alignment-audit.md` §4.1）；`SAME_STEP` 下 done 当步返回新局 obs，`dones`/GAE 语义与 cleanrl 一致，episode 统计改从 `infos["final_info"]["episode"]` 读取（`train.py:_episode_info`）。

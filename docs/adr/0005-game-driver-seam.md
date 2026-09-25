# 对局驱动接缝：`Match` 拥有多座位循环

引擎 `Game` 是纯状态机，但“轮到谁 → 投影 `View` → 策略出牌 → 合法性回退 → 前进”的循环此前在 `env._advance_opponents`、`eval.evaluate`、`play.play_game`、`play.replay_trace`、demo 与测试里各写了一遍；`eval` 甚至绕过 env 接口去读 `env.game.view(env.state, …)`。决定：新增 `match.Match` 作为**唯一**的对局驱动——轮转、策略派发与合法性 clamp 只此一处；`Game` 保持不碰策略与循环，`env` / `eval` / `play` / 回放 / 测试全部经 `Match` 驱动。生产投影仍只有 `Game.view`（ADR-0002 不变），`Match.view(seat)` 只是转发。

## 考虑过的替代

- **保留按调用点的循环**：改动最小，但轮转与回退语义漂移，且 eval/测试只能穿过 env 的私有字段。
- **把循环塞进 `Game`**：会让纯规则引擎依赖 `Policy` 接缝，破坏“规则层零 RL 依赖”的分层。
- **回调式/actor 模型**：更灵活，但要为脚本 bot、神经策略、人类输入各养一套适配，且偏离现有的同步驱动。

## 后果

- `step(action_id)` 由外部驱动，走引擎严格校验（非法抛 `ValueError`）；`step()` 由当前座位策略出牌，非法则 clamp 到首个合法并计入 `illegal_actions`——策略永远不会卡死对局，而外部动作仍保持严格。
- `Match` 暴露 `turns` / `illegal_actions` / `on_turn` 钩子，评估指标与终端录制不再各自重算循环。
- `env` 删除 `_advance_opponents`，新增 `view(seat)`；`eval` 不再读取 `env.game` / `env.state` / `env.action_mask`。
- 同一轮重构把轨迹格式收进 `trace.py`（`play` 只做终端回放）、把 PPO 数学收进 `ppo.py`（`train` 只做装配）——这两处的边界是显而易见的，不单独立 ADR。

# 单一评分基准与 GreedyBot 退役：RandomBot=0，其余等级分全部由对局产生

> **后续重标定（提示）**：本文正文的 `COLD_START_PRIOR`/`RESULT_PRIOR` 500 与
> `prior.ANCHOR_CENTER` 230 是 ADR-0012 落地时的坐标；T17 已按 `c = 0.4656209850248892`
> 缩放为 `COLD_START_PRIOR ≈ 232.81`、`prior.ANCHOR_CENTER ≈ 107.09`（T15 §7 的延后项，见
> [`docs/experiments/t17-recalibration.md`](../experiments/t17-recalibration.md) 与 `src/seven523/prior.py`）。
> RandomBot=0 与 GreedyBot 退役的决定不变。

ADR-0011 把估计器换成了 OpenSkill，但保留了旧的固定参照体系：manifest 里
`random=1000`、`greedy=1315` 两个钉死的“秤砣”，以及 `GreedyBot` 这个自定义脚本策略。
OpenSkill 的评分是相对的：锚点值不是“标定出来的”，而是表的坐标基准；旧值又是 BT-MAP
时代在退役标度上的遗留（T2 标度下 random 实测 1026.9、greedy 实测 1314.4，钉死值
1000/1315 从来不是测量值）。继续维护 `greedy=1315` 会让每次重测都背上一个过期的
预设分数，也让“所有等级分都可以通过对局修改”的模型失去一致性。

决定：只保留一个钉死的坐标基准，其余一切由对局估计。

1. **RandomBot 是唯一的 gauge，`mu = 0`。** 全表做 −1000 平移：`DEFAULT_MU`
   1500→500、`COLD_START_PRIOR` 1500→500、`RESULT_PRIOR` 1500→500、T2 标签
   −1000（并删除 greedy 档）、`prior.ANCHOR_CENTER` 1230→230。差量、间距、CI
   阈值不受影响；`artifacts/human-elo/prior.json` 仍是平移前的坐标，按 T15 重拟合
   （测试显式标注它在旧坐标上）。
2. **GreedyBot 从仓库删除。** `policies.GreedyBot` 类、`policy_from_spec` /
   `validate_spec` / `make_scripted_policies` 的 `greedy` 分支全部移除；spec 语法
   只剩 `random` / `ckpt:<path>`。训练默认 `--opponent random`，`mix` 的脚本成员
   改为 RandomBot（`--mix-random-prob`），pool 成员不再接受 `greedy`；
   `train` / `eval` / `7g523-play` / `play_ladder` 注册表、`build_ladder` / `arena`
   默认锚点、`measure_trace_signal` 的默认标签、placement 的对手池与测试夹具一并
   去除。旧 `greedy` spec 一律报 `unknown policy spec`。
3. **测试用自己的确定性替身，不复活生产策略。** `tests/support.py` 定义
   `FirstLegalBot`（首个合法动作，仅测试使用，不进 spec 语法、不可被评分）。它与被删
   的 GreedyBot 在实际对局里 99.9% 的选择一致，所以既有轨迹与期望值基本不变。
4. **单 gauge 需要候选互打来排序。** `build_ladder` 不再要求 ≥2 个锚点，只要求
   ≥1 个 pinned gauge；`cross` 默认取 `games_per_anchor`（候选两两换座 twin），
   显式 `cross=0` 才只打 gauge。`arena` 本来就全成员互打（`cross=60` 默认）。
   `plan_games` 的纯排期接口保持 `cross=0`，默认值在 `build_ladder`/工具层解析。

## 考虑过的替代

- **把 greedy 变成自由参考（钉死值删掉，角色保留）**：owner 明确要求连同策略一起
  删除；且候选互打（cross）已经提供候选间信息，不需要第二个脚本参照。
- **保留 `greedy=1315` 只做坐标**：它来自旧估计器的一次测量，任何重测都会与之冲突；
  基准取 `random=0` 更简单，也更符合“0 = 随机水平”的直觉。
- **只 pin 一个真实玩家（让 random 也可变）**：则表失去坐标，跨 fit 不可比；随机
  策略本身又几乎不被对局移动（p(1−p)≈0），钉死它最稳定。
- **立刻把 prior.json 重拟合到新原点**：需要重跑 study/重标定（T15），属于数据任务；
  代码常量先平移，制品留待 T15。
- **在 `src` 里保留一个 `FirstLegalBot` 供测试用**：会把测试替身变成生产策略与潜在
  spec 成员；放 `tests/support.py` 更干净。

## 后果

- 全部绝对等级分平移 −1000：random=0，旧 lvl1–lvl4 变为 ≈129–448，T2 标签同步；
  `prior.json` 与任何旧 manifest 的数值仍是旧坐标，其中旧 `elo/se` manifest 已被
  `load_opponents` 拒绝，须在 T15 用 `tools/build_ladder.py` 重测。
- 默认 ladder 的对局数增加（每个候选对另打 `games_per_anchor` 局），换来的是候选
  之间直接可比的评分；单 gauge 下若显式 `cross=0`，候选只能靠与 random 的近零信息
  对局区分，属于调用者负责的退化用法。
- 脚本课程只剩 `random`：阶段 1 的对手强度显著低于原 GreedyBot，训练曲线与旧结论
  不可直接比较；这是移除 greedy 策略的已知代价。
- `tests/support.py` 成为测试侧确定性策略的唯一出处；生产 spec 语法与文档中的
  `greedy` 全部退役。

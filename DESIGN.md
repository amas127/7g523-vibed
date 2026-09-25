# 7鬼523 — 设计文档 / Design

> 规则事实源：[RULES.md](./RULES.md)　领域语言：[CONTEXT.md](./CONTEXT.md)　关键决策：[docs/adr/](./docs/adr/)
> 基础训练环境：**2 家**（`num_players` 参数化，引擎支持 N 家）。代码在 `src/seven523/`。

目标：一个**纯规则引擎**（deep module）+ 一个 **RL 环境适配器**，
使 `ppo-implementation-details/ppo_multidiscrete_mask.py` 无需修改即可训练。

---

## 0. 架构定稿（Design It Twice 结论）

四个独立方案（最小接口 / 最大灵活 / PPO 默认路径 / ports & adapters）横评后，取杂交：

1. **`combos.classify/beats` 保持公开、纯函数** —— 两个函数藏全部比较语义，是金牌测试面。拒绝“把 beats 吸进 mask”。
2. **`Game` 三入口**：`new(rng)`、`step(state, action) -> (state, result)`、`view(state, seat) -> View`。`current/done` 折进 `GameState`。亮牌/收分/补牌/撬底都是 `step` 内的自动转移。
3. **单一投影接缝**：`view()` 是 agent/bot 唯一能看到的信息通道，只装领域数据 + mask；**obs 向量编码属于 `env`**，core 不知道 `OBS_DIM`。测试/回放直接读 `GameState`（不做第二套类型），但必须有差分泄漏测试（ADR-0002）。
4. **热路径**：mask 用 134-bit `int` + 预计算，`action_mask` 是唯一合法性权威；obs 由 env 原地写缓冲。
5. **`catalog()` 生成、不做 `ComboKindSpec` registry**：规则已冻结，扩展是投机接缝。
6. **真接缝只有 `Policy`**（Random/Greedy/未来 Neural）；RNG 参数注入，不建 `EntropyPort` 类。

被拒绝：B 的 registry（投机接缝）、C 的不透明 `Table` + 双类型（用投影 + 差分测试替代）、A 的 beats 吸收（破坏测试面）、D 的可变 `Engine` 作为唯一实现（先纯函数；实测不够再换实现，接口不变）。

---

## 1. 模块与接缝

依赖只能向内，规则层零 RL 依赖：

```
        train.py / eval.py                    (PPO 移植 + 评估)
             │
        networks.py ── gymnasium.vector        (Agent/NeuralPolicy；SyncVectorEnv)
             │
        env.py        ──── policies.py        (gymnasium.Env；对手：脚本 bot / 神经策略)
             │                    │
        game.py  ──────────────┘              (亮牌→一墩→收分→补牌→撬底)
             │
        actions.py                            (规范动作目录：mask + resolve，env 与 bot 共用)
             │
        combos.py                             (牌型识别 + 比较 —— 最深的纯模块)
             │
        cards.py / rules.py                   (牌 + 序；Rules 值对象)

        全部经 Rules 配置；纯规则层可独立单测
```

| 模块 | 角色 | 依赖类别 | 说明 |
|------|------|----------|------|
| `cards.py` / `rules.py` | 值对象 | in-process | 无外部依赖 |
| `combos.py` | **deep module（核心）** | in-process | `classify`/`beats` 两个入口藏全部比较语义 |
| `actions.py` | 动作空间 | in-process | 固定 134 目录，`action_mask` 唯一合法性权威 |
| `game.py` | 流程状态机 | in-process | 纯函数式 `GameState` 转移 + `View` 投影 |
| `policies.py` | 接缝 | 2+ adapters | Random/Greedy；`make_scripted_policies` |
| `env.py` | RL adapter | 依赖 gymnasium | 唯一知道观测张量形状的模块 |
| `networks.py` | 训练适配 | 依赖 torch | `Agent`/`CategoricalMasked`/`NeuralPolicy`/checkpoint 读写 |
| `train.py` | 装配/CLI | torch + gymnasium (+tensorboardX) | cleanrl PPO 移植：阶段 1 bot、阶段 2 冻结自博弈、CSV + TensorBoard 指标、checkpoint |
| `eval.py` | 评估/CLI | torch 惰性 | 得分/分差/胜率/非法动作率 |
| `play.py` | 终端人机对战 | 规则层 + 可选 torch | `7g523-play`：列表合法出牌、输入编号；对手可为 bot/checkpoint；`--save-trace`/`--replay` 轨迹持久化与回放校验 |

---

## 2. 接口（与 `src/seven523/` 一一对应）

### 2.1 `cards.py` / `rules.py`

```python
class Suit(IntEnum): DIAMOND=0; CLUB=1; HEART=2; SPADE=3
class Rank(IntEnum): R4=0; R6=1; ...; R7=14          # 15 档点数升序
@dataclass(frozen=True, slots=True)
class Card: rank: Rank; suit: Suit | None            # 王 ⇔ suit is None
def card_key(c: Card) -> tuple[int, int]             # (点数, 花色)；王花色 -1
def make_deck() -> tuple[Card, ...]                  # 恒 54 张
def point_value(c: Card) -> int                      # 5/10/K -> 5/10/10
def card_id(c: Card) -> int                          # 0..53，观测编码用

@dataclass(frozen=True, slots=True)
class Rules: num_players=2; hand_size=7; straight_min=3; straight_max=7;
             consecutive_pairs_min=3; consecutive_pairs_max=3; total_points=100
```

### 2.2 `combos.py`（最深的模块）

```python
class ComboKind(Enum): SINGLE; PAIR; STRAIGHT; CONSECUTIVE_PAIRS; SMALL_BOMB; BIG_BOMB
@dataclass(frozen=True, slots=True)
class Combo:
    kind: ComboKind
    cards: tuple[Card, ...]           # 规范化：去重 + 按 card_key 升序
    # 属性：tier / size / top_key / top_rank / top_card / strength / is_bomb
def classify(cards, rules=DEFAULT_RULES) -> Combo | None
def beats(candidate, incumbent, rules=DEFAULT_RULES) -> bool
```

契约：`classify` 全函数（非法 → `None`，不抛）、**不设长度上限**（上限由目录施加）；
`beats(c, None) is True`、`beats(c, c) is False`；比较只走
`tier →（小炸弹比点数，其余比 size → top_key）`，全语义在这一处（locality）。

### 2.3 `actions.py`

```python
@dataclass(frozen=True, slots=True)
class Action: kind: ComboKind | None; ranks: tuple[Rank, ...]; label: str
def build_catalog(rules) -> tuple[Action, ...]
CATALOG: tuple[Action, ...]           # 134 项，含 PASS
PASS_ID: int
SUIT_N: int                           # 花色头 4（ADR-0004）
def resolve(action, hand, rules, suit=None) -> Combo | None   # 顶牌按 suit（缺省最强）；其余牌弱色优先（占优）
def top_rank(action) -> Rank | None   # 接受花色选择的点数；炸弹/PASS 为 None
def suit_options(action, hand, rules) -> tuple[Suit, ...]     # 顶牌可选花色，强→弱
def action_mask(hand, incumbent, rules) -> int        # 134-bit；唯一合法性权威
def legal_ids(mask: int) -> list[int]
def split_action(action) -> (int, int | None)         # 兼容 int / 1 元 / 2 元动作
```

### 2.4 `game.py`

```python
class Phase(Enum): PLAY; DONE
@dataclass(frozen=True, slots=True)
class GameState:
    hands; draw_pile; scores; trick_points; trick_cards; revealed;
    current; incumbent; last_player; collected; phase
    # 属性：done / hand(seat)

@dataclass(frozen=True, slots=True)
class StepResult:
    trick_over; winner; points_taken; refilled; dug; done

@dataclass(frozen=True, slots=True)
class View:                          # agent/bot 唯一能看到的投影
    seat; hand; mask; incumbent; current; scores; counts;
    draw_count; revealed; trick_cards; last_player; done

class Game:
    def __init__(self, rules=DEFAULT_RULES): ...
    def new(self, rng: random.Random) -> GameState: ...       # 发牌 + 亮牌定先
    def view(self, state, seat) -> View: ...
    def step(self, state, action_id, suit=None) -> tuple[GameState, StepResult]: ...
    def is_terminal(self, state) -> bool: ...
    def returns(self, state) -> tuple[float, ...]: ...        # terminal-only
```

契约：`step` 只接受 `view(...).mask` 为真的动作，非法动作抛 `ValueError`；
`step` 内自动完成亮牌、收分、补牌（赢家下家先、赢家最后）、撬底；
`collected` 支撑牌张守恒测试；`sum(hand sizes)+len(draw_pile)+len(trick_cards)+collected == 54`。

### 2.5 `policies.py` + `env.py`

```python
class Policy(Protocol):
    def act(self, view: View) -> tuple[int, int | None]: ...
class RandomBot(Policy): ...
class GreedyBot(Policy): ...          # 最弱合法跟牌，优先非炸弹；无牌可跟才 PASS

class Seven523Env(gym.Env):            # gymnasium 5 元组 API（env 唯一碰张量形状）
    action_mask: list[bool]           # 138 = 134 模板 + 4 花色，永远对应刚返回的 obs
    action_space: MultiDiscrete([134, 4]) # 模板头 + 花色头（ADR-0004；目录规模随 Rules）
    observation_space: Box(OBS_DIM,)   # OBS_DIM = 185 + 3 * num_players
    def reset(self, *, seed=None, options=None) -> (obs, info)
    def step(self, action) -> (obs, reward, terminated, truncated, info)

class NeuralPolicy:                    # Policy 接缝的神经 adapter（networks.py）
    def act(self, view: View) -> tuple[int, int | None]  # (模板, 花色)；旧单头 ckpt 花色为 None
```

---

## 3. 动作目录（固定 134）

自然序为 13 点数环，手牌上限 7 ⇒ 顺子仅长 3–7，连对仅 3 对：

| 类型 | 数量 |
|------|------|
| 单牌 | 15 |
| 对子 | 13 |
| 顺子（长度 3–7，环上 13 个起点） | 65 |
| 连对（3 对，环上 13 个起点） | 13 |
| 小炸弹（13 三张 + 王炸） | 14 |
| 大炸弹 | 13 |
| PASS | 1 |
| **合计** | **134** |

测试断言 `len(CATALOG) == 134`，防止静默漂移（ADR-0001）。动作空间为
`MultiDiscrete([134, 4])`：134 个模板 + 4 个顶牌花色（ADR-0004）；花色头对炸弹/PASS 无效，不可用花色回退最强实现。

---

## 4. 观测与奖励

观测在 `env._observation` 中构造（core 不知道）。维度 `OBS_DIM = 185 + 3 * num_players`
（2 家 = 191）：

| 段 | 维度 | 内容 |
|----|------|------|
| 自己手牌 multi-hot | 54 | `card_id` 索引 |
| 自己每点数计数 | 15 | 计数 / 4 |
| incumbent 顶牌 multi-hot | 54 | 公开 |
| incumbent kind one-hot | 6 | |
| incumbent size / 7 | 1 | |
| 底牌堆张数 / 54 | 1 | 隐藏顺序 |
| 各家分数 / 100 | n | 公开 |
| 各家手牌数 / 7 | n | 公开 |
| 当前座位 one-hot | n | |
| 亮牌 multi-hot | 54 | 公开（开局已知的一张牌） |

奖励（RL-2）：终局 `own_score/100 − mean(others)/100`，terminal-only；训练只看分数。

---

## 5. 比较语义：Golden cases（已测）

| 用例 | 期望 |
|------|------|
| `单7♠` vs `单7♥` | 黑桃胜（花色 tie-break） |
| `单7` vs `单大王` | 7 胜；`大王 > 小王` |
| 顺子 `A-2-3`、`Q-K-A`、`Q-K-A-2-3` | 合法（13 点数环） |
| 顺子含王 / 点数不连续 | 非法 |
| `顺子 8-9-10` vs `单7` | 顺子胜（层级 2 > 1） |
| `对7` vs `单7` | 不可比（同层级不同小类） |
| `连对 33-44-55` vs `对7` | 连对胜 |
| `777` vs `王炸` vs `555` | `777 > 王炸 > 555`（小炸弹只比点数） |
| `4444` vs `777` | 大炸弹胜（层级 4 > 3） |
| `顺子 4-5-6` vs `2-3-4` | 比点数序最大牌：`5` vs `4` ⇒ `4-5-6` 胜 |
| `对子 ♦7♠7` vs `对子 ♥7♣7` | 顶牌（♠7）决定对子强度：前者胜 |
| 单♠7 vs 单♥7 | 黑桃胜（花色头可主动选择打哪张，ADR-0004） |

---

## 6. 测试面（interface is the test surface）

1. `cards`：牌堆 54、无重复、`card_key` 全序。
2. `combos.classify`：每类合法/非法边界（长度、炸弹、王炸、环上顺子）。
3. `combos.beats`：Golden cases + 性质（非自反、反对称、层级单调、小炸弹忽略张数）。
4. `actions`：mask↔resolve↔beats 一致；`len(CATALOG) == 134`。
5. `game` 不变量：牌张守恒、分数守恒、手牌 ≤7、随机对局必终止、终局 `sum(scores)==100`。
6. **差分泄漏测试**：两份仅隐藏字段不同的 `GameState`，`view(seat)` 的公开字段与 mask 必须逐位相同（ADR-0002）。
7. `env`：gymnasium 空间/5 元组 API、obs 形状、mask 与 obs 同步。
8. `train`/`eval`：masked categorical 不采样非法动作、Agent 前向/replay 形状、checkpoint 往返、极小训练与自博弈冒烟、评估指标恒等式（`mean_return == score_diff / 100`）。

---

## 7. 与 PPO 的衔接 / 训练计划

训练栈决议见 [ADR-0003](./docs/adr/0003-training-stack.md)：gymnasium + torch，仓库内忠实移植
`ppo_multidiscrete_mask.py`（原脚本是 gym 0.21 + MicroRTS CNN，与 3.12/平坦观测不兼容）。
完整命令、产物说明与常见开关：[docs/training.md](./docs/training.md)。

1. `action_space = MultiDiscrete([134, 4])` → `nvec.sum() = 138` = `action_mask` 长度（模板头 134 + 花色头 4，ADR-0004）。
2. 向量化用 `gymnasium.vector.SyncVectorEnv` + `RecordEpisodeStatistics`；每步从
   `env.get_wrapper_attr("action_mask")` 读 mask 进 rollout buffer。
3. 阶段 1：`uv run --group train 7g523-train --opponent greedy --total-timesteps 1e6`。
4. 阶段 2：自我博弈，`--opponent self --load-checkpoint runs/.../agent.pt`，
   冻结快照每 `--self-play-refresh` 次更新原地刷新。
5. 评估：`7g523-eval --checkpoint ... --opponent greedy|random`，报告平均得分、平均分差、
   胜/平/负、非法动作率（应为 0）；训练中可用 `--eval-interval` 周期评估。
6. 指标写 `runs/<name>/metrics.csv`（纯文本、无额外依赖）；训练默认同时写
   `runs/<name>/tb/` 的 TensorBoard event（`--tensorboard False` 可关，`tensorboard --logdir runs` 看曲线）；checkpoint 为
   `agent.pt` / `checkpoint.pt`（`torch.save`，含 `obs_dim`/`nvec`/`hidden`）。

---

## 8. 实施状态

代码在 `src/seven523/`（发行名 `7g523`；`uv run 7g523` 跑演示）。

- [x] 核心：`cards` / `rules` / `combos` / `actions` / `game` / `policies` / `env`（gymnasium 5 元组 API）
- [x] 测试 161 项：Golden cases、目录规模 134、mask↔resolve↔beats 一致、2–7 家随机/贪心对局不变量、撬底/补牌顺序/空手留局定向用例、`Rules` 校验、自定义 `Rules` 下的目录/环境尺寸回归、差分泄漏、masked categorical 不越奖、Agent 前向/replay、checkpoint 往返、训练/自博弈/评估冒烟、人机对战渲染、轨迹保存/回放与篡改检测、TensorBoard event 断言、花色头 `resolve`/`suit_options`/合法性回退/warm-start/旧 ckpt 兼容
- [x] 性能（单线程纯 Python）：2 家 ~6.0k steps/s、3/4 家 ~6.4k steps/s；`action_mask` ~75µs/次；PPO 端到端（8 env）：阶段 1 ~1.8k、自博弈 ~1.4k learner steps/s（均含周期评估）
- [x] 训练验证：300k 步阶段 1（vs GreedyBot）后 500 局评估：分差 **+21.3**、胜率 64%（随机基线 -53.6 / 14%）；再 150k 步自博弈热启动后分差 +19.2、胜率 63%；非法动作率均为 0
- [x] PPO 训练/评估：`networks.py`（Agent/NeuralPolicy/checkpoint）+ `train.py`（cleanrl 忠实移植、两阶段、`--eval-interval`）+ `eval.py` + `play.py`（人机对战）；动作空间 `(模板, top_suit)` 见 [ADR-0004](./docs/adr/0004-suit-head.md)，框架决议见 [ADR-0003](./docs/adr/0003-training-stack.md)

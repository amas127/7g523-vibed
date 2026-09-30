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
6. **真接缝只有 `Policy`**（Random/未来 Neural）；RNG 参数注入，不建 `EntropyPort` 类。

被拒绝：B 的 registry（投机接缝）、C 的不透明 `Table` + 双类型（用投影 + 差分测试替代）、A 的 beats 吸收（破坏测试面）、D 的可变 `Engine` 作为唯一实现（先纯函数；实测不够再换实现，接口不变）。

---

## 1. 模块与接缝

依赖只能向内，规则层零 RL 依赖：

```
（箭头指向被依赖方：右侧模块依赖左侧模块）
规则核心:  cards.py / rules.py ← combos.py ← actions.py ← game.py
对局驱动:  policies.py ← match.py ← env.py          （env → gymnasium）
轨迹录制:  trace.py ← play.py ← record.py
训练栈:    ppo.py / networks.py / league.py / metrics.py ← train.py（torch + gymnasium）
评分编排:  elo.py ← ladder.py / duel.py（duel 另依赖 ladder）；arena.py 依赖 ladder + elo
先验定级:  prior.py（numpy）← placement/（包；torch 惰性）
清单:      study.py（无内部依赖）
tools/*.py → src/seven523（src 不反向依赖 tools）

全部经 Rules 配置；纯规则层可独立单测
```

| 模块 | 角色 | 依赖类别 | 说明 |
|------|------|----------|------|
| `cards.py` / `rules.py` | 值对象 | in-process | 无外部依赖 |
| `combos.py` | **deep module（核心）** | in-process | `classify`/`beats` 两个入口藏全部比较语义；`TIER` 只服务炸弹层级与 `Combo.strength` 排序 |
| `actions.py` | 动作空间 | in-process | 固定 134 目录 + 头布局，`action_mask` 唯一合法性权威 |
| `game.py` | 流程状态机 | in-process | 纯函数式 `GameState` 转移 + `View` 投影 + `Deal` + `seat_outcome` 唯一胜负定义 |
| `policies.py` | 接缝 + spec 语法 | 2+ adapters | Random/NeuralPolicy adapter；`Policy`/`EpisodePolicy`/`WeightedPolicy` 协议与 `split_entrant`/`validate_spec` 等 spec 助手（GreedyBot 已随 ADR-0012 退役） |
| `match.py` | 对局驱动 | in-process | 轮转 + 策略派发 + 合法性回退；唯一的多座位循环 |
| `env.py` | RL adapter | 依赖 gymnasium | 唯一知道观测张量形状的模块（单一 v5 布局，ADR-0009）；驱动委托给 `Match`，终局 info 带 `scores`/`outcome` |
| `ppo.py` | 算法 | 依赖 torch | `compute_gae` / `ppo_update` 两个入口 |
| `trace.py` | 编解码 | in-process | 轨迹格式（`Deal` + 每步 + 分数 + 标签/文件名约定），版本与字段只此一处 |
| `record.py` | 批量录制 | in-process | `play_recorded`/`policy_seed`/`RecordedGame`：ladder/placement/measure 共用的唯一批量录制接缝 |
| `networks.py` | 训练适配 | 依赖 torch | `Agent`/`CategoricalMasked`/`NeuralPolicy`/checkpoint（`load_agent` 只收 v5）/`warm_start_from` |
| `metrics.py` | 指标 sink | in-process | `LOG_FIELDS`/`TB_TAGS`/`MetricsLogger`/`TensorboardLogger` |
| `league.py` | 联赛装配 | torch + numpy | `LeagueConfig`/`League`/`build_league`/`parse_pool_member`/`pool_member_ids`/`PfspController` |
| `train.py` | 装配/CLI | torch + gymnasium | rollout/update 循环；指标、联赛、热启动已拆出 |
| `eval.py` | 评估/CLI | 规则层 | 得分/分差/胜率/非法动作率；胜负走 `game.seat_outcome` |
| `play.py` | 终端人机对战 | 规则层 + 可选 torch | `7g523-play`；`play_game` 每个座位都经 `Match`/`Policy`，交互 chooser 由 `ChooserPolicy` 适配；轨迹持久化与回放校验 |
| `elo.py` | **评分核心（deep）** | in-process | `fit_ratings`/`select_rungs`/`expected_score` 藏 OpenSkill（Weng–Lin Plackett–Luce）高斯在线更新、锚点/先验、`tau` 动态与 N 家支持；不依赖 torch/numpy/IO（ADR-0011） |
| `ladder.py` | M2 阶梯编排 + 分片 | 规则层 + torch 惰性 | `plan_games`/`play_games`（`results_dir=` 是分片唯一接口）/`build_ladder`：成对牌局、座位轮换、≥1 pinned gauge（RandomBot=0，ADR-0012）；轨迹与人类对局同格式 |
| `duel.py` | 双人配对比较 | in-process | `plan_duel_schedule`/`paired_duel_stats`/`combine_duel_seeds`：同牌 twin + 按牌聚簇 bootstrap |
| `arena.py` | 联盟赛 | 规则层 + torch 惰性 | `run_arena`；`play_parallel` 是 `ladder.play_games` 的薄包装；排名/成对诊断/TensorBoard |
| `study.py` | manifest schema | in-process | `load/merge/save_manifest`；`levels` 是 D1 契约，唯一 writer owner，评分默认冻结（`refit` 显式覆盖） |
| `prior.py` | 轨迹 S1 先验核心 | 依赖 numpy | 特征、岭回归、`prior.json` schema、`TracePrior`、`session_mean`；`tools/*` 只做 CLI |
| `placement/` | 真人定级包 | numpy + 可选 torch | `opponents`/`estimator`/`session`/`cli`；`__init__` 再导出公开 API，`7g523-elo` 入口不变 |
| `web/` | 浏览器牌桌 | 规则层 + numpy + 可选 torch | `view`（JSON 投影）/`table`（会话状态）/`server`（HTTP）/`cli`；`7g523-web` 入口；定级模式复用 placement 的公开 stepwise API（`next_game`/`commit_game`/`finish`） |
| `tools/*.py` | CLI | 只依赖 `seven523` | 薄壳：解析参数、调库、打印；src 不反向依赖 tools |

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
    # 属性：family / tier / size / top_key / top_rank / top_card / strength / is_bomb
def classify(cards, rules=DEFAULT_RULES) -> Combo | None
def beats(candidate, incumbent, rules=DEFAULT_RULES) -> bool
```

契约：`classify` 全函数（非法 → `None`，不抛）、**不设长度上限**（上限由目录施加）；
`beats(c, None) is True`、`beats(c, c) is False`；比较只走
`族 →（同族 size → top_key；炸弹按 tier → rank）`，全语义在这一处（locality）。

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
def action_mask(hand, incumbent, rules) -> int        # 134-bit；唯一合法性权威（记忆化）
def legal_ids(mask: int) -> list[int]
def split_action(action) -> (int, int | None)         # 兼容 int / 1 元 / 2 元动作
def nvec_for(rules) -> tuple[int, ...]                # (模板数, SUIT_N)（ADR-0004）
def joint_mask_bits(template_mask, nvec) -> list[bool] # 模板位 + 偏好头全开；env/网络共用
def index_hand(hand) -> dict[Rank, list[Card]]        # 按点数分组，强花色在前
def resolve_indexed(action, by_rank, rules, suit=None) # 复用 index_hand 的 resolve
```

头布局（`nvec_for`/`joint_mask_bits`）与 `action_mask` 的记忆化都收在 `actions.py`：
加一个头或改合法性只需动这一处，`env` 与 `networks` 不再各自重建 138 位掩码。

### 2.4 `game.py`

```python
class Phase(Enum): PLAY; DONE
@dataclass(frozen=True, slots=True)
class GameState:
    hands; draw_pile; scores; trick_points; trick_cards; revealed;
    current; incumbent; last_player; collected; phase; plays   # plays: 公开动作日志
    # 属性：done / hand(seat)

@dataclass(frozen=True, slots=True)
class Play:                          # 一条公开动作（合牌或 pass），见事件历史规划
    seat; cards; opens_trick; went_out

@dataclass(frozen=True, slots=True)
class StepResult:
    trick_over; winner; points_taken; refilled; dug; done

@dataclass(frozen=True, slots=True)
class Deal:                          # 开局发牌（可脱离 RNG 重建）
    hands; draw_pile; revealed; starter
    @classmethod
    def from_state(cls, state) -> Deal

@dataclass(frozen=True, slots=True)
class View:                          # agent/bot 唯一能看到的投影
    seat; hand; mask; incumbent; current; scores; counts;
    draw_count; revealed; trick_cards; played; last_player; done;
    plays                            # 公开动作日志（默认 (）；第二输入接缝）

class Game:
    def __init__(self, rules=DEFAULT_RULES): ...
    def new(self, rng: random.Random) -> GameState: ...       # 发牌 + 亮牌定先
    def restore(self, deal: Deal) -> GameState: ...           # 从发牌重建开局
    def view(self, state, seat) -> View: ...
    def step(self, state, action_id, suit=None) -> tuple[GameState, StepResult]: ...
    def returns(self, state) -> tuple[float, ...]: ...        # terminal-only

def seat_outcome(scores, seat) -> int: ...                    # +1 胜 / 0 平 / −1 负（与最强对手比）
```

契约：`step` 只接受 `view(...).mask` 为真的动作，非法动作抛 `ValueError`；
`step` 内自动完成亮牌、收分、补牌（赢家下家先、赢家最后）、撬底；
`collected` 支撑牌张守恒测试；`sum(hand sizes)+len(draw_pile)+len(trick_cards)+collected == 54`。

### 2.5 `policies.py` + `env.py`

```python
class Policy(Protocol):
    def act(self, view: View) -> tuple[int, int | None]: ...
class EpisodePolicy(Protocol):        # 逐局对手：env.reset 时 start_episode() 冻结本局成员
    def start_episode(self) -> str: ...
class WeightedPolicy(Protocol):       # PFSP 可调权：set_weights + current_id/finished_id
    def set_weights(self, weights: Sequence[float]) -> None: ...
class RandomBot(Policy): ...

def policy_from_spec(spec, rules=DEFAULT_RULES, seed=None, device="cpu") -> Policy
    # random / ckpt:<path>；torch 只在 ckpt: 分支惰性导入（GreedyBot 已退役，ADR-0012）
# spec 语法唯一 owner（ID=SPEC 拆分、auto-id、存在性检查、错误文案）：
def split_entrant(raw) -> (str, str); def default_id_for_spec(spec) -> str
def missing_ckpt_path(spec) -> str | None; def validate_spec(spec) -> str | None

class Seven523Env(gym.Env):            # gymnasium 5 元组 API（env 唯一碰张量形状）
    action_mask: list[bool]           # 138 = 134 模板 + 4 花色，永远对应刚返回的 obs
    action_space: MultiDiscrete([134, 4]) # 模板头 + 花色头（ADR-0004；目录规模随 Rules）
    observation_space: Box(OBS_DIM,)   # OBS_DIM = observation_dim(n) = 119 + 21n（唯一 v5 布局，ADR-0009）
    reward_shaping: str                # terminal / trick_diff / win / trick_diff_win / terminal_win / saturate
    def reset(self, *, seed=None, options=None) -> (obs, info)
    def step(self, action) -> (obs, reward, terminated, truncated, info)  # 终局 info: {"scores", "outcome"}
    def view(self, seat=None) -> View  # 投影（默认 learner），对局驱动委托给 Match

class NeuralPolicy:                    # Policy 接缝的神经 adapter（networks.py）
    def act(self, view: View) -> tuple[int, int | None]  # (模板, 花色)
```

### 2.6 `match.py` / `trace.py` / `play.py` / `record.py` / `ppo.py`

```python
class Match:                          # 唯一的多座位对局循环
    def __init__(self, rules, policies, *, state=None, rng=None)  # policies[seat] | None
    state: GameState; done: bool; seat: int
    steps: int; turns: list[int]; illegal_actions: int
    on_turn: Callable[[seat, action_id, suit, View, StepResult], None] | None
    def view(self, seat=None) -> View
    def policy_for(self, seat) -> Policy | None
    def step(self, action_id=None, suit=None) -> StepResult   # 无 id → 走策略并 clamp
    def advance(self, *, stop=None) -> StepResult | None       # 停在外部座位/stop/终局
    def run_to_end(self) -> GameState

# trace.py —— 轨迹格式（字段与版本只此一处）
def card_json / rules_json / deal_json / initial_snapshot / state_from_snapshot
def step_record(seat, action_id, suit, *, text, state, result) -> dict  # 每步字段
def player_label(role, id, seat) / parse_player_label(label)          # 标签语法
def trace_filename(index, seed, human_seat, opponent)                 # 文件名约定
def build_trace(rules, *, seed, human_seat, players, created_at, **record)
def save_trace(path, trace); def load_trace(path) -> dict
TRACE_VERSION: int

# play.py —— 终端对局与回放（每个座位都经 Match/Policy）
def play_game(policies, *, chooser=None, rules=DEFAULT_RULES, human_seat=0,
              seed=None, print_fn=print, record=None) -> tuple[int, ...]
class ChooserPolicy:                  # 旧 (game, state, view) chooser -> Policy；bind(match)
def interactive_chooser(human_seat, print_fn=print) -> chooser
def replay_trace(trace, *, print_fn=print, pause=False, input_fn=input) -> bool
def opponent_identity(checkpoint, opponent="random") -> (role, id)

# record.py —— 批量录制唯一接缝（ladder / placement / measure 共用）
def policy_seed(seed, seat) -> int    # (seed + 101 * (seat + 1)) & 0xFFFF_FFFF
def play_recorded(policies, *, rules=DEFAULT_RULES, seed=None, human_seat=0,
                  players=None, created_at=None, trace_dir=None, trace_index=0,
                  opponent="", chooser=None, print_fn=print) -> RecordedGame
class RecordedGame: seed; human_seat; scores; trace; trace_path  # 不落盘时 trace_path=None

# ppo.py —— PPO 数学（忠实参考实现，ADR-0003）
def compute_gae(rewards, values, dones, next_value, next_done, *, gamma, gae_lambda, use_gae=True)
def ppo_update(agent, optimizer, batch, config) -> dict[str, float]
class PPOConfig: ...                # PPOConfig.from_args(args)
class RolloutBatch: ...             # RolloutBatch.flatten(...) + .size
```

契约：`step(action_id)` 走引擎严格校验（非法抛 `ValueError`）；`step()` 由策略出牌，非法则 clamp
到首个合法并计入 `illegal_actions`。`advance`/`run_to_end` 只驱动有策略的座位；`None` 座位由
调用方用 `step` 驱动（env 暂停 learner）。`play_game` 把 `chooser` 包成 `ChooserPolicy` 后
所有座位统一走 `Match`；`play_recorded` 负责逐局 seed、trace 组装与落盘命名。

### 2.7 `elo.py` / `ladder.py` / `study.py`（ADR-0011，取代 ADR-0006 的估计器）

```python
# elo.py —— 结果评分（纯数学；观测单位 = 一局牌局）
# OpenSkill Plackett–Luce 高斯模型，按时间顺序重放历史；差量参数 = openskill 默认 ×24
# 原点：RandomBot gauge = 0（ADR-0012；旧表 random=1000/greedy=1315 已退役）
defaults: DEFAULT_MU=500.0; DEFAULT_SIGMA=200.0; DEFAULT_BETA=sigma/2; DEFAULT_TAU=sigma/100
class Prior: mu: float = 500.0; sigma: float = 200.0
class PlayedGame: seed: int; seats: tuple[str, ...]; scores: tuple[int, ...]
class FitConfig: prior=Prior(); beta=100.0; tau=2.0
class Rating: mu: float; sigma: float; n: int          # 锚点 sigma == 0
class Fit: ratings: dict[str, Rating]; games: int
class Rung: id; mu; sigma
class RungSelection: rungs; requested; min_spacing; max_spacing; wide_gaps; tail_gap; ok
def expected_score(rating_a: Gaussian, rating_b: Gaussian) -> float  # PL 的 P(a 胜 b)
def fit_ratings(games, *, anchors, priors=None, config=FitConfig()) -> Fit
    # N 座位均可；ties 按同分同名次；anchors 钉死（sigma=0）；tau 替代旧 window
def select_rungs(ratings, *, count=5, min_spacing=100.0, max_spacing=150.0) -> RungSelection

# ladder.py —— M2 编排（对局经 play_game 录制，轨迹仍是 trace.py 格式）
class Entrant: id; spec; pinned: float | None; prior: Prior | None
class ScheduledGame: seed; seats; subject
class Ladder: entrants; fit; selection; schedule
def plan_games(entrants, *, games_per_anchor=100, cross=0, seed=0) -> tuple[ScheduledGame, ...]
def play_games(schedule, entrants, *, rules=DEFAULT_RULES, factory=None,
               out=None, results_out=None, results_dir=None, created_at=None,
               device="cpu", workers=1) -> list[PlayedGame]
def build_ladder(entrants, *, games_per_anchor=100, cross=None, seed=0, rules=DEFAULT_RULES,
                 out=None, games_out=None, created_at=None, count=5, min_spacing=100.0,
                 max_spacing=150.0, config=FitConfig(), factory=None,
                 device="cpu", workers=1) -> Ladder
    # 要求 ≥1 pinned gauge（RandomBot=0）；不再需要第二个预设锚点

# study.py —— manifest schema 唯一 owner（D1 的 features/calibrate 读 levels）
def load_manifest(path) -> dict; def save_manifest(path, document) -> Path
def merge_manifest(document, *, levels, subjects, anchors=None, rungs=None,
                   estimator=None, frozen_at=None, refit=False) -> dict
    # levels: id -> mu；subjects 键为 mu/sigma/games（旧 elo/se 已随 ADR-0011 退役）
```

契约：`PlayedGame` 一局一条记录（锚点 `sigma = 0`；`tau` 让近期对局权重更高，取代旧的逐 id
`window`）；`fit_ratings` 是纯函数（无 RNG/IO，先验与锚点都是数据，按时间顺序重放、结果依赖
顺序）；`select_rungs` 的间距缺口用 `ok`/`wide_gaps`/`tail_gap` 显式上报，从不静默放宽；
`ladder` 要求 ≥1 个 pinned gauge（RandomBot=0，ADR-0012）；D3 以
`priors={id: Prior(μ_traj, σ_traj)}` 进入同一评分，
不新增接缝。旧 BT-MAP 的 `margin`/`window`/`converged` 已删除（OpenSkill 没有分差似然，`tau`
承担动态，拟合是单遍重放）；`mu/sigma` 与旧 `elo/se` 不可直接比较，需重测（T15）。
`play_games(workers>1)` 把 schedule 切成连续分片、spawn 进程执行，逐局 policy seed 只依赖
对局本身，`workers=1` 与 `>1` 逐位一致；`results_dir` 让每个 worker 直接写
`shard_NNNNN.jsonl`（与 `results_out` 互斥），`arena.play_parallel` 只是把 `games_out`
转发到这里。

### 2.8 `duel.py` / `arena.py` / `prior.py` / `placement/`（ADR-0006/0010/0011）

```python
# duel.py —— 双人同牌换座比较（纯统计，无 IO）
def plan_duel_schedule(left, right, *, pairs, seed) -> tuple[ScheduledGame, ...]
def paired_duel_stats(games, *, left_id, right_id, bootstrap=4000, rng,
                      confidence=0.95) -> dict      # 按 deal 聚簇 bootstrap + 符号检验
def combine_duel_seeds(per_seed, *, seeds=None, z=1.96) -> dict  # ≥2 seed 合并

# arena.py —— 联盟赛（工具层；并行只是 ladder.play_games 的包装）
def play_parallel(schedule, entrants, *, workers=1, device="cpu", rules=DEFAULT_RULES,
                  games_out=None) -> list[PlayedGame]
def run_arena(entrants, *, games_per_anchor=60, cross=60, seed=0, workers=1,
              device="cpu", rules=DEFAULT_RULES, games_out=None,
              config=FitConfig()) -> ArenaResult

# prior.py —— 轨迹 S1 先验核心（tools 只做 CLI）
def extract_features(trace, *, verify=True) -> dict          # 每局 S1 特征行
def extract_row(path, levels, *, verify=True); def load_rows(study, levels, ...)
def ridge_fit(x, y, alpha); def ridge_predict(model, x)      # 标准化岭回归
def build_prior(...); def load_prior(path); def save_prior(doc, path, *, created_at=None)
def predict_elo(doc, row, *, opponent_elo=None) -> float
def session_mean(doc, rows, *, opponent_elos=None) -> float
def prior_for_session(doc, mu_traj, n) -> Prior
class TracePrior: load/predict_elo/prior_for_session/features/cold_start/labels/meta

# placement/ —— 10 局真人定级包（__init__ 再导出；`main` = 7g523-elo）
class SessionConfig: games=10; stop_ci=50.0; min_games_before_stop=1;
                     explore_games=2; z=1.96; seat_start=0; human_id="human";
                     rung_prior_sd=13.97
class Opponent: id; mu; sigma; spec; anchor
（manifest 读 mu/sigma/games；旧 elo/se 键报错要求重测，ADR-0011）
def load_opponents(manifest) -> (opponents, anchors)   # 缺 ckpt 的 rung 告警跳过；锚点缺失报错
def plan_seats(games, *, start=0); def plan_deals(rng, games); def select_opponent(...)
def fit_session(games, *, human_id, human_prior, opponents, anchors,
                rung_prior_sd=13.97, rung_centers=None) -> Fit
class PlacementSession: run(chooser_factory=None, *, human_policy_factory=None, print_fn=print)
def main(argv=None) -> int
```

```text
# web/ —— 浏览器牌桌（__init__ 再导出；`main` = 7g523-web）
class WebConfig: manifest; prior; device; sessions_dir; free_traces_dir; seed
class TableSession: from_config/start/apply_action/continue_placement/quit_session/
                    snapshot/config_document
class WebError: status  # 4xx/409 映射给 HTTP 层
def make_handler(table); def serve(table, *, host, port, open_browser=False)
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

观测由 `env.encode_observation` 构造（core 不知道）。**只有一份布局**：`env.py` 的
`_SEGMENTS` 段表是唯一真源，`observation_dim` 与 `encode_observation` 都从它推导，段与
维度不会漂移。v5 是 `119 + 21n`（2 家 161）；`OBS_VERSION = 5` 仍写进 checkpoint payload，
`networks.load_agent` 对缺失或非 5 的 payload 直接拒绝。v1–v4 已随兼容层退役
（[ADR-0009](./docs/adr/0009-single-observation-and-comparison.md)）。

**v5（唯一布局，观测 S1 + B0 + B1，自中心旋转）：**

| # | 段 | 维度 | 内容 |
|---|---|------|------|
| 1 | `hand` | 54 | 自己手牌 `card_id` multi-hot |
| 2 | `inc_rank` | 15 | incumbent 顶牌 rank one-hot（王由 rank 判别） |
| 3 | `inc_suit` | 4 | incumbent 顶牌花色 one-hot（王 = 全零） |
| 4 | `inc_kind` | 6 | `ComboKind` one-hot |
| 5 | `inc_size` | 1 | 张数 / `hand_size` |
| 6 | `draw` | 1 | 底牌堆张数 / 54 |
| 7 | `scores` | n | 自中心：`scores[(seat+k) % n] / total_points` |
| 8 | `opp_count` | n−1 | 对手手牌数 / `hand_size`（自中心，无自己维） |
| 9 | `opp_revealed` | 19(n−1) | 每对手亮牌 rank15 + suit4（自中心） |
| 10 | `trick_points` | 1 | B0：当前墩已落桌分 / `total_points` |
| 11 | `remaining_points` | 1 | B0：未被赢走分 / `total_points` |
| 12 | `point_hold` | 1 | B0：自己手牌分 / `total_points` |
| 13 | `unseen` | 54 | B1：54 − 手牌 − 亮牌 − `played` − 当前墩 multi-hot |
| 14 | `last_player` | 1 | B1：incumbent 归属座位的归一化槽（无 incumbent = 0） |

奖励（RL-2）：终局 `own_score/100 − mean(others)/100`；`reward_shaping` 可选
`terminal`（默认，逐位旧行为）/ `trick_diff` / `win` / `trick_diff_win` / `terminal_win`
（终局回报 + `win_jump` × 胜负）/ `saturate`（正分差按 `reward_cap` 截顶，须显式给 cap），
终局 info 另带 `{"scores", "outcome"}`。

---

## 5. 比较语义：Golden cases（已测）

| 用例 | 期望 |
|------|------|
| `单7♠` vs `单7♥` | 黑桃胜（花色 tie-break） |
| `单7` vs `单大王` | 7 胜；`大王 > 小王` |
| 顺子 `A-2-3`、`Q-K-A`、`Q-K-A-2-3` | 合法（13 点数环） |
| 顺子含王 / 点数不连续 | 非法 |
| `顺子 8-9-10` vs `单7` | 顺子胜（同族，size 大） |
| `对7` vs `单7` | 不可比（跨族） |
| `连对 33-44-55` vs `对7` | 连对胜（同族，size 大） |
| `顺子 8-9-10` vs `对7` | 不可比（跨族） |
| `连对 33-44-55` vs `单7` | 不可比（跨族） |
| `王炸` vs `顺子`/`连对` | 王炸胜（炸弹压所有非炸弹） |
| `777` vs `王炸` vs `555` | `777 > 王炸 > 555`（小炸弹只比点数） |
| `4444` vs `777` | 大炸弹胜（炸弹层级：小 < 大） |
| `顺子 4-5-6` vs `2-3-4` | 比点数序最大牌：`5` vs `4` ⇒ `4-5-6` 胜 |
| `对子 ♦7♠7` vs `对子 ♥7♣7` | 顶牌（♠7）决定对子强度：前者胜 |
| 单♠7 vs 单♥7 | 黑桃胜（花色头可主动选择打哪张，ADR-0004） |

---

## 6. 测试面（interface is the test surface）

1. `cards`：牌堆 54、无重复、`card_key` 全序。
2. `combos.classify`：每类合法/非法边界（长度、炸弹、王炸、环上顺子）。
3. `combos.beats`：Golden cases + 性质（非自反、反对称、同族单调、跨族互不可压、小炸弹忽略张数）。
4. `actions`：mask↔resolve↔beats 一致；`len(CATALOG) == 134`。
5. `game` 不变量：牌张守恒、分数守恒、手牌 ≤7、随机对局必终止、终局 `sum(scores)==100`。
6. **差分泄漏测试**：两份仅隐藏字段不同的 `GameState`，`view(seat)` 的公开字段与 mask 必须逐位相同（ADR-0002）。
7. `env`：gymnasium 空间/5 元组 API、obs 形状、mask 与 obs 同步。
8. `train`/`eval`：masked categorical 不采样非法动作、Agent 前向/replay 形状、checkpoint 往返与非 v5 payload 拒绝、极小训练与自博弈冒烟、评估指标恒等式（`mean_return == score_diff / 100`）。
9. `match`：全策略跑到终局、外部座位停在 `advance`、非法策略动作被 clamp 并计数、外部动作严格抛错、`on_turn` 逐回合触发。
10. `trace`：card/rules/deal 往返、开局重建、`build_trace` 版本戳、步骤 schema、标签/文件名约定、存取往返（无需终端）。
11. `ppo`：GAE 手算对照（含 bootstrap/gamma/非 GAE 分支）、`RolloutBatch.flatten` 形状、`ppo_update` 参数确实更新且 loss 有限。
12. `actions` 头布局：`nvec_for`、`joint_mask_bits`（含旧单头 ckpt）、`resolve_indexed` 与 `resolve` 一致、`action_mask` 记忆化。

当前 `uv run pytest -q` 收集 **448 项**；具体用例以 `tests/` 为准。

执行约定（2026-09-25 测试精简，勿回退）：

- `tests/conftest.py` 把 `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS` 默认压到 1：被测模型极小，torch 默认 24 线程的同步开销大于计算（套件 CPU 时间 6m52s → 13s）；需要多线程时显式覆盖。
- 并行：`uv run pytest -n 8`（串行 ~11s，`-n 8` ~6–7s）。本机 32 核 / 15GB 内存，裸 `-n auto` 会起 32 个 torch worker，建议用 `PYTEST_XDIST_AUTO_NUM_WORKERS=8` 给 `auto` 设上限。
- smoke 的步数、种子数、对局数（train 128–256 步、`test_game` fuzz 25 种子、`test_fit_trace_prior` `PER_LEVEL=10`、duel 只在 1600 trials 测溢出）是按最小覆盖刻意取的下限，同一接缝均有更细的单测兜底；加回重复用例前先确认它没被覆盖。

---

## 7. 与 PPO 的衔接 / 训练计划

训练栈决议见 [ADR-0003](./docs/adr/0003-training-stack.md)：gymnasium + torch，仓库内忠实移植
`ppo_multidiscrete_mask.py`（原脚本是 gym 0.21 + MicroRTS CNN，与 3.12/平坦观测不兼容）。
完整命令、产物说明与常见开关：[docs/training.md](./docs/training.md)；
**全部计划/实验优先级与待决策事项见 [docs/plans.md](./docs/plans.md)**（路线图单一入口）。

1. `action_space = MultiDiscrete([134, 4])` → `nvec.sum() = 138` = `action_mask` 长度（模板头 134 + 花色头 4，ADR-0004）。
2. 向量化用 `gymnasium.vector.SyncVectorEnv`（**`AutoresetMode.SAME_STEP`**，对齐参考栈语义，见 [ADR-0003](./docs/adr/0003-training-stack.md)）+ `RecordEpisodeStatistics`；每步从
   `env.get_wrapper_attr("action_mask")` 读 mask 进 rollout buffer。
3. 阶段 1：`uv run --group train 7g523-train --opponent random --total-timesteps 1e6`。
4. 阶段 2：自我博弈，`--opponent self --load-checkpoint runs/.../agent.pt`，
   冻结快照每 `--self-play-refresh` 次更新原地刷新。
5. 评估：`7g523-eval --checkpoint ... --opponent random`，报告平均得分、平均分差、
   胜/平/负、非法动作率（应为 0）；训练中可用 `--eval-interval` 周期评估。
6. 指标写 `runs/<name>/metrics.csv`（纯文本、无额外依赖）；训练默认同时写
   `runs/<name>/tb/` 的 TensorBoard event（`--tensorboard False` 可关，`tensorboard --logdir runs` 看曲线）；checkpoint 为
   `agent.pt` / `checkpoint.pt`（`torch.save`，含 `obs_dim`/`nvec`/`hidden`/`activation`/`arch`/`obs_version`）。

---

## 8. 模块清单

代码在 `src/seven523/`（发行名 `7g523`；`uv run 7g523` 跑演示）。

| 分组 | 模块 |
|------|------|
| 规则核心 | `cards` / `rules` / `combos` / `actions` / `game` |
| 对局驱动 | `policies` / `match` / `env` |
| 轨迹与录制 | `trace` / `play` / `record` |
| 训练 | `ppo` / `networks` / `metrics` / `league` / `train` / `eval` |
| 评分与阶梯 | `elo` / `ladder` / `duel` / `arena` / `study` |
| 真人定级 | `prior` / `placement/` |
| 工具 | `tools/*.py`（薄 CLI，只依赖 `seven523`） |

各模块的角色与依赖见 §1，接口见 §2。**状态、测试计数与里程碑不在这里维护**：
实验结论与规范数字见 [docs/experiments/README.md](./docs/experiments/README.md)，
全部计划、优先级与待决策见 [docs/plans.md](./docs/plans.md)；架构决策史见
[docs/adr/](./docs/adr/)（本轮 v5/tier 退役见
[ADR-0009](./docs/adr/0009-single-observation-and-comparison.md)，单一 owner 接缝见
[ADR-0010](./docs/adr/0010-single-owner-seams.md)）。

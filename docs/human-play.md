# 人机对战手册

本页讲三件事：怎么直接用 `7g523-play` 打、怎么用 `tools/play_ladder.py`
选对手、怎么跑一局 10 副牌的定级会话。对手清单的单一可信来源是
`tools/play_ladder.py`（`list` 会打印文件是否存在检查）。

> **模型资产状态（2026-09-29 更新）**：牌型族比较规则变更后，`runs/` 下全部旧
> checkpoint 已随旧规则模型删除；T17（2026-09-26，出空即撬底 rules revision 3）
> 已在当前规则下重训并重测 `lvl1`–`lvl4`，`tools/play_ladder.py` 现登记 `random`
> gauge + 四个梯级 + 顶部平台簇（`ws_s2`/`pself_s2`/三条 w2m 臂）。2026-09-29 的
> `search_leafq` 发布（P4a：t5/K32/C6、value `t_leafq`，与 raw 梯级同一 26,000 局
> 联合 MLE，ADR-0013）把定级池扩为 **11 级**：raw 档重拟为 **85 / 110 / 126 / 181**
> （顶部簇 183–187），新档 `search_leafq` μ≈**260.03**。
> `traces/pool10/manifest.json`（= `traces/study/manifest.json`）已重发布，
> `rules_id` 与 `lvl1`/`lvl4` 梯级契约不变；**跨 fit 数字不可加、不可混比**。
> `artifacts/human-elo/prior.json` 仍是 2026-09-27 w2m/T23 重排校准的表，尚未重拟
> （见 §5 与 `docs/search-config-plan.md` §4.3/§5）；含搜索 rung 的会话不再整体关prior，
> 而是**按局排除**：只有真正打到搜索 rung 的那局不进轨迹通道并记 `prior_off_reason`，
> raw 局照常使用先验（ADR-0013，绝不外推 μ≈260）。
> T17 重标定数字、先验与 placement 常数见
> [`experiments/t17-recalibration.md`](./experiments/t17-recalibration.md) 与
> [`plans.md`](./plans.md) T17；本页历史训练档位与数字均为旧口径，禁止与当前
> 结果混比。

## 1. 直接开打（7g523-play）

```bash
uv run 7g523-play                        # 默认对手 RandomBot（锚 0）
uv run 7g523-play --opponent random      # 同上；random 是唯一脚本对手
uv run --group train 7g523-play \
    --checkpoint runs/t17long__1__1790439615/snapshots/checkpoint_step1024000.pt \
    --seat 1 --rounds 3 --sample         # T17 池顶档（也可用 play_ladder play lvl4）
```

- 训练好的 ckpt 需要 `train` 依赖组（torch）；`random` 脚本对手不需要。
  T17 已重训 `lvl1`–`lvl4`（§2/§3）；继续训练新模型按
  [`training.md`](./training.md)（重标定记录见 [`plans.md`](./plans.md) T17 与
  [`experiments/t17-recalibration.md`](./experiments/t17-recalibration.md)）。
- 局中输入 `h` 看帮助、`q` 退出；`--seat` 选座位（默认 0），`--rounds` 连打多局，
  `--sample` 让神经对手按 mask 采样（默认 argmax），`--seed` 固定发牌。
- `--save-trace traces/xxx.json` 存轨迹，`--replay` 回放并逐步校验引擎；
  轨迹里的对手标签形如 `opponent:<id>@seat1` / `anchor:random@seat0`，
  真人轨迹后续可按对手强度标定。
- `--num-players` 必须与 ckpt 的训练配置匹配；新模型默认 2 家。

## 2. 用 play_ladder 发现与启动

```bash
uv run python tools/play_ladder.py list          # 表格 + 每个档的启动命令
uv run --group train python tools/play_ladder.py list --json   # 供脚本消费
uv run python tools/play_ladder.py play random -- --seat 1 --rounds 3
```

`play <id>` 只把 id 翻译成 `--opponent random` 或
`--checkpoint <path>`，其余参数（建议用 `--` 分隔）原样透传给 `7g523-play`；
因此等价于复制 `list` 里该档的启动命令再追加参数。未知 id 或 ckpt 文件缺失
会直接报错并退出码 2。

当前注册表的对手有 `random` gauge、T17（rules revision 3）重标定的 `lvl1`–`lvl4`
梯级，以及 w2m/T23 重排后的顶部平台簇 `ws_s2`/`pself_s2`/`w2m_low`/`w2m_plain`/
`w2m_ctl`：旧梯级、平台/练习快照与实验室臂的 checkpoint 已随旧规则模型删除，新档位
用当前规则/v5 观测重训并按同一共享 probit-MLE 表登记。后续新 ckpt 按
`runs/<new-run>/agent.pt` 路径登记回 `tools/play_ladder.py` 的 `OPPONENTS` 注册表，
并用同一口径重测强度列，即可用 `play <id>` 启动。

## 3. 对手档位与强度含义

| id | 类型 | 强度 | 说明 |
|---|---|---|---|
| `random` | 脚本 | 锚 0（固定） | 均匀随机；唯一的脚本对手与评分基准（ADR-0012） |
| `lvl1` | 梯级 | 85（w2m） | revision-3 入门档，w2m 重标定（manifest μ≈84.60；`t17early2k`） |
| `lvl2` | 梯级 | 110（w2m） | revision-3 梯级，w2m 重标定（manifest μ≈109.79；`t17early8k`） |
| `lvl3` | 梯级 | 126（w2m） | revision-3 梯级，w2m 重标定（manifest μ≈126.24；`t17pool` step 65k 快照） |
| `lvl4` | 梯级 | 181（w2m） | revision-3 顶档梯级，w2m 重标定（manifest μ≈181.19；`t17long` 1M 步快照） |
| `ws_s2` | 平台 | 183（w2m） | T23 warm-start 续训冠军（manifest μ≈182.91；`t23wswd__2`） |
| `pself_s2` | 平台 | 183（w2m） | self-play 池化臂第二参照（manifest μ≈183.24；`t18poolself__2`） |
| `w2m_low` | 平台 | 186（w2m） | w2m 2M 续训臂：LR 减半（manifest μ≈186.00） |
| `w2m_plain` | 平台 | 186（w2m） | w2m 2M 续训臂：Adam+linear+random（manifest μ≈186.25） |
| `w2m_ctl` | 平台 | 187（w2m） | w2m 2M 续训臂：AdamW+cosine+pool；池内点估计最高（manifest μ≈187.46） |
| `search_leafq` | 搜索 rung | 260（measured） | P4a t5/K32/C6、value `t_leafq`、rollout 对手 `w2m_ctl`；**search wrapper identity**（不是 raw 档，跨 fit 不可比）；身份钉在 manifest 的 `subjects[]` 中 `id=search_leafq` 条目的 `search_config`；`rolloutt:` spec 只能由注入 factory 构建（web 插件或程序化）：自由对战（插件逐局包装）与**统一定级池**（C3 调度；打到它的那局不进轨迹先验）；`7g523-play` 列表不登记 |

强度均为 **RandomBot=0 的 homoscedastic probit-MLE 绝对表口径**（manifest 契约值，
由 `tools/refit_mle.py --manifest-out --refit` 发布；2026-09-29 `search_leafq` 联合
重拟的 11 级表见 `runs/w2m/calibration/table_with_search.json`；2026-09-27 w2m/T23
表与 T17 历史值 83/107/148/185 均已被取代）；`search_leafq` 的 μ 只在该 rung 的
联合表内可比，禁止「raw μ + h2h Δ」相加（ADR-0013/JS §12）。`arena` 列（联合联赛
拟合）尚无新测量。旧训练档（`lvl1`–`lvl4`、`base20k`/`base60k`/
`base120k`/`base180k`/`base680k`、`a1`/`b1`/`cp`/`towers`/`w5_*` 等）及其强度数字
（旧 manifest 契约值与 arena Elo）均已随旧规则 checkpoint 删除而下线：旧路径不可用，
数字是旧口径历史记录，禁止当作当前档位或与新规则结果混比。

## 4. 推荐顺序

1. `random`：读界面、熟悉出牌与花色选择。
2. `lvl1` → `lvl4`：沿 revision-3 梯级逐步加难（85 → 110 → 126 → 181，w2m
   重标定），再进 10 局定级会话（§5）。
3. 平台簇 `ws_s2`/`pself_s2`/`w2m_low`/`w2m_plain`/`w2m_ctl`（183–187）：当前最强
   raw 模型群，相互间在 ±10 Elo 内不可分辨，适合难度上限挑战。搜索档 `search_leafq`
   （μ≈260，**不同身份**）只在浏览器（统一定级池 / 自由对战）或程序化注入 factory 时可用，
   不能通过 `play_ladder` 启动。

```bash
uv run python tools/play_ladder.py play w2m_ctl -- --seat 1 --rounds 3 --sample
uv run python tools/play_ladder.py play ws_s2   -- --seat 0 --rounds 3 --seed 0
```

## 5. 10 局定级会话（7g523-elo）

```bash
uv run --group train 7g523-elo
# 非交互冒烟（脚本当“真人”）：
uv run --group train 7g523-elo --simulate random --games 10 --seed 0
```

- 默认读 `traces/pool10/manifest.json`（2026-09-29 revision-3 池：`random` gauge +
  `lvl1`–`lvl4` + `ws_s2`/`pself_s2`/三条 w2m 臂 + `search_leafq`，共 11 级，与
  `traces/study/manifest.json` 同内容；梯级仍冻结为 `lvl1`/`lvl4`；
  `search_leafq` 只有在注入搜索 factory 时才能构建，否则 CLI/web 会跳过并告警）与
  `artifacts/human-elo/prior.json`（2026-09-27 先验 **v2/R1**，sha256 `15fddff1…`）；
  10 副不同牌、5/5 座位轮换、前 2 局 Thompson 探索、之后 `info` 选档。
- **2026-10-07/08 池追加（当前共 17 个 subject）**：web-local 未测量档 `head32ln5m` /
  `search_head32ln5m` 与 raw `bres221_7M` / `bres221_9M`（μ≈212.8/213.7、σ=8、games=0）；
  以及已测量的 β=1 搜索档 `bres221_7M_b1`（μ=**262.14** [256.94,267.58]）/
  `bres221_9M_b1`（μ=**265.70** [260.37,271.10]），σ≈2.7、n=8000，语料 34k（18k
  calibration + 16k p5a；`search_leafq` 按 option B carried-over，p4a 语料丢失）。
  provenance：`artifacts/web-bres221-b1/`、`artifacts/web-head32ln5m/`；背景见
  [`experiments/outcome-search-probe.md`](./experiments/outcome-search-probe.md) §6。
- 若显式指定其他 manifest：已删除的 ckpt 档位会被自动跳过并告警；跨 `rules_id`
  的旧 manifest 会被门禁拒绝（ADR-0013）。
- **先验已升级为 v2（R1）**：`prior.json` = 二次特征展开（152 列）+ cell 校准惩罚
  λ=3，训练语料 `traces/study10`（39,960 条、9 级全 RR），RandomBot=0 绝对表口径
  （T2 标签 = 2026-09-27 w2m/T23 homoscedastic probit-MLE manifest levels；
  2026-09-29 搜索 rung 联合重拟把 manifest raw 档移动 ≤8.2 分，先验尚未重拟：
  raw 会话按 T2 中心偏移在噪声内；真正打到 `search_leafq` 的局其行被排除出轨迹通道
  （`prior_off_reason` 记录，禁止把 μ≈260 喂进 raw 先验）。within-corpus LOLO：
  RMSE 52.9、m_eff 43.28、σ(5/10/20)=26.8/20.7/18.0、cell_drift² 67、drift_sd 7.79；
  独立发牌 bank2 transfer（不 refit）：v2/v1 RMSE **53.20/58.30**、
  σ(5/10)=**24.61/17.69** vs 27.33/19.60、drift 6.29 vs 9.05（判据 6/6 CONFIRMED）。
  注意旧 v1 4k 的 RMSE 40.5 与 v2 的 52.9 是**不同语料**、不可互比；同 bank2 的
  transfer 对照才是可比口径。placement 固定常数仍为 T17 按 `c=0.465621` 折算后的值
  （冷启动 232.81±139.69）：同一 probit-MLE 刻度，不需要重折。旧 v1 4k 先验归档
  `runs/archive/prior-v1-4k-20260927/`；研究/复核细节见
  [`experiments/prior-opponent-correction.md`](./experiments/prior-opponent-correction.md)
  与 [`experiments/prior-v2-confirmation.md`](./experiments/prior-v2-confirmation.md)。
- 结束时打印 `定级：R ± CI（95%），最近档 <id>，临时 provisional/已定级`。
- 产物在 `traces/sessions/<session-id>/`：每局一份 trace、每局后落盘的
  `session.json`、结束后的 `report.json`（点估计 / CI / 最近档 / 两通道权重）与
  `rungs.json`（对手池元数据）。`--session-id`、`--sessions-dir` 可指定位置。
- 10 局只承诺点估计 + 诚实 CI：研究结论是 RMSE 54–72、95% CI ±100–133
  （见 [`experiments/human-elo-10-games-research.md`](./experiments/human-elo-10-games-research.md)）；
  `provisional` 是正常状态，应继续对局到 CI ≤ 50（默认 `--games 10` 可加大，
  或 `--stop-ci 50 --min-games 8`）。
- `--simulate random|ckpt:<path>` 让脚本策略替真人跑完整会话，用于验证流程。

## 6. 浏览器牌桌（7g523-web）

想用鼠标打而不是敲终端时用 `7g523-web`：一个本地 HTTP 服务 + 单页牌桌，底下是**同一个**
引擎和**同一个**定级管线（`PlacementSession`），不是另一套实现。

```bash
uv run --group train 7g523-web                 # → 终端的 http://127.0.0.1:8765/
uv run 7g523-web --no-trace-prior              # 无 torch / 冷启动：只能打 random
uv run 7g523-web --no-plugin                   # 强制关闭插件发现（纯 raw 表）
uv run --group train 7g523-web --plugin runs/o4lite-search/web_plugin.py
uv run --group train 7g523-web --port 9000 --open
SEVEN523_WEB_PLUGIN=off uv run 7g523-web       # 同 --no-plugin
```

**单入口与插件发现**：stock `7g523-web`（console script 指向 `seven523.web:main`）在三个注入缝
（`policy_factory` / `search` / `twin`）全为 None 时，按固定顺序探测一次可选插件，把 run-local 的
搜索工厂、twin 与 manifest-驱动的搜索能力带进同一条 `WebConfig` 管线：`--no-plugin` 关闭 → `--plugin PATH`
显式指定（失败退出 1）→ 环境变量 `SEVEN523_WEB_PLUGIN`（空/0/off/none/false/no 为关闭，其余视为路径）
→ cwd 默认 `runs/o4lite-search/web_plugin.py`（不存在则静默 raw-only，损坏则 warning 后回退 raw-only）。
`--manifest` 路径会传给插件 hook，插件从同一份 manifest 的 `subjects[].search_config` 重建搜索身份，
不再读 `artifacts/search-rung`。没有目录扫描、没有 entry-point 自动发现，包源码不 import `runs/`；
信任域与你直接跑 run-local 启动器相同。`runs/o4lite-search/web_search.py` / `web_twin.py` 仍然可用：
它们显式注入三个缝，因此不会触发插件发现（twin 也改为从 manifest 读 pin）。

- **自由对战**：从 manifest 的池子里任选对手 + 座位，单局即打；每局写成真实
  trace 到 `traces/web/<run-id>/`，可用 `uv run 7g523-play --replay <trace>` 校验。
  启用搜索插件后，`ckpt:` 对手按页面逐局选定的 t/K/β 包装（`random` 锚不包装）。
  设置页可选（**默认关闭**）**对手模型预估**：打牌时右栏实时显示对手模型
  critic/outcome 头对终局分差与你胜率的即时读出（单次前向，不含搜索与 β；
  没有 outcome 头的 ckpt 只显示分差，`random` 锚没有模型不显示）；空底后若已有精确解
  则直接显示精确分差（不额外触发求解）。
- **定级模式**（见 §6.3）：
  * 单一 manifest 池（锚点 + raw 档 + 能构建的搜索 rung，同一联合拟合尺度）；
    前 2 局 Thompson 探索、之后 Fisher 信息选档，产物与 `7g523-elo` 完全一致：
    `traces/sessions/<session-id>/`；
  * 真正打到搜索 rung 的局不进轨迹先验并在 `prior_off_reason` 记录，raw 局照常。
- 界面：手牌按自然序排列、点手牌或牌型按钮出牌、可选顶牌花色；开局亮牌直接亮在
  双方前面（第一墩结束收起）；**记牌器**用公开信息（我的手牌 + 公开出牌日志 +
  亮牌）推算每个点数的我/已出/余，可收起；动作日志默认折叠。
- 常用参数：`--manifest` / `--prior` / `--no-trace-prior`、`--sessions-dir`、
  `--trace-dir`、`--host` / `--port`、`--device`、`--seed`、`--open`、`--plugin` / `--no-plugin`。

### 6.1 浏览器里的搜索对手（stock 插件 / web_search，run-local）

搜索部署档位（value 截断 t=5 + K 个隐藏世界 O4-lite rollout，见
[`experiments/joint-search-training-wave1.md`](./experiments/joint-search-training-wave1.md) §10）
现在由 `runs/o4lite-search/web_plugin.py` 作为 stock 入口的插件带进来（终端双胞胎入口
`runs/o4lite-search/play_search.py` 不变）：

```bash
# stock 单入口：在仓库根直接起，自动发现 runs/o4lite-search/web_plugin.py
uv run --group train 7g523-web
uv run --group train 7g523-web --plugin runs/o4lite-search/web_plugin.py   # 显式路径
# 旧的搜索启动器入口仍然可用（显式注入，行为不变）：
uv run --group train python runs/o4lite-search/web_search.py                 # 默认 t=5, K=16
uv run --group train python runs/o4lite-search/web_search.py --full          # 默认全量 rollout
uv run --group train python runs/o4lite-search/web_search.py --no-fast       # 不装 Route-A 加速层
```

- **自由对战**：所有 `ckpt:` 对手被 O4-lite 包装（选中 ckpt 同时做 base/argmax 与
  自己的 value/outcome 核心——不再有第二个 `--value-ckpt` 模型；`random` 锚不包装）。
  设置页**逐局可选搜索深度**：`t`（截断 ply，`0` = 全量）、`K`（隐藏世界数，1..512；
  t 的硬界 0..40）与 `β`（outcome 头拌进截断叶值 `V + β·u`，0..8，0 = 只用 critic；
  只有带 `vf_outcome` 头的 ckpt 能非零），附实测预设 t5·K8 / t5·K16 / t5·K32 /
  全量 K8 与 β0/0.5/1/2；游戏内与结算页显示当局配置，自由对战 trace 里记
  `opponent_search`（回放忽略）。stock 插件默认 t=5 / K=32 / β=0（placement 的
  搜索 rung 用 manifest `search_config`，含 pinned `outcome_blend`）；`web_search.py`
  启动器仍默认 t=5 / K=16。
  另有**「终局精确解」**开关（`search.endgame`，默认开）：空底（`draw_count=0`）后由
  α-β minimax 精确接管（预算 1e6 节点 / 5s，总牌数 ≤14，超限回退原搜索；一局只解一次，
  之后查表），页面与 trace 同样记录；定级/twin 的搜索 rung 不受该开关影响
  （manifest `search_config.endgame` 缺省 0，显式写 1 才进身份）。逐位验证见
  [`experiments/endgame-minimax-audit.md`](./experiments/endgame-minimax-audit.md)，
  身份裁定见 [ADR-0019](./adr/0019-endgame-exact-minimax-overlay.md)。
- **惰性 torch**：插件导入本身不 import torch；只有真正选 ckpt 对局（free 包装 / twin /
  搜索 rung 定级）才加载 value ckpt。无 torch 时插件元数据照常展示，但 ckpt 对手与
  twin/搜索 rung 开始请求都会 400；raw 定级与 `random` 自由对战不受影响。
- **定级池已合并**：manifest 的 `levels` 就是定级池，raw 档与 `search_leafq` 同表调度；
  搜索 rung 的身份（t5/K32/C6、rollout 对手 `w2m_ctl`、`O4_AGG=mean`）钉在
  `search_leafq` 条目的 `search_config`，插件从 `--manifest` 读它，不再有 rated 子池/artifact 覆盖。
  没有 factory 的 stock 服务端按能力门跳过并告警，行为与旧 raw 池一致。
- **本地追加（2026-10-07）**：`traces/pool10/manifest.json`（= `traces/study/manifest.json` 镜像）
  里加了两条 **web-local、未测量**的条目：raw `head32ln5m`（`ckpt:runs/head-depth/cont/
  hd_head32ln5m__1__1791396292/agent.pt`，μ≈198.6、σ=8、games=0）与搜索档 `search_head32ln5m`
  （spec `rolloutt:<同一 ckpt>`，`search_config` 抄 `search_leafq` 的 t5/K32/C6 + rollout 对手
  `w2m_ctl` + mean；μ≈271.2 粗估）。两者已在 web 池中可打（自由对战/定级/搜索构建实测通过）；
  μ 是 k=1 h2h 粗估，不得进任何测量发布。回滚备份与操作脚本：`artifacts/web-head32ln5m/`；
  背景见 [`head-depth-plan.md`](./head-depth-plan.md) §15 与 [`experiments/head-depth-500k.md`](./experiments/head-depth-500k.md) §12。
- 其余参数原样转发给 `7g523-web`（`--manifest` / `--prior` / `--port` / `--open` …）；
  web_search 启动器的 `--trunc` / `--rollout-k` 只决定设置页的默认值。

> 改了 `server.py` 之后要**重启进程**（页面脚本会热加载，Python 进程不会）；
> 页面检测到旧进程会提示重启，并用兼容回退保持“第一墩后收亮牌”的行为。

### 6.2 deal-twin 对比：人对 raw vs 人对搜索（web_twin，run-local）

P4b 的 deal-twin 会话：**每个 deal 打两局**——一次对人 raw 臂（`ckpt:` 原始
argmax）、一次对人 search 臂（同一 base 套 O4-lite 搜索，pin t=5 / K=32 / C=6）。
两局复用同一 deal seed 与同一 human 座位，开局快照逐字节相同；座位与先后顺序
**跨 pair 交替**，所以每个（臂 × 座位）组合精确各占一半。报告给出配对
Δ（对搜索 − 对 raw：分差为主，胜率/Elo 为辅）与按 deal 聚类的 percentile
bootstrap CI（B=4000，`bootstrap_seed` 固定，逐位可复现）。
stock 单入口下 twin 由插件从同一份 manifest `search_config` 派生 pin（`web_twin.twin_metadata`），
不再需要单独启动：

```bash
uv run --group train 7g523-web                   # stock：插件带 twin（identity=rolloutt:runs/ei2_value_t5/t_leafq/critic.pt）
# P4b 启动器仍然可用（显式注入，行为不变）：
uv run --group train python runs/o4lite-search/web_twin.py                 # 启动牌桌（默认 serve）
uv run --group train python runs/o4lite-search/web_twin.py --serve --port 8766 --open
# 脚本代真人完整跑一遍（不走 HTTP；用真实 SearchFactory + torch wrapper）：
uv run --group train python runs/o4lite-search/web_twin.py \
    --simulate random --out runs/o4lite-search/p4b/simulate-smoke --pairs 2 --seed 7
```

- **臂的身份**：raw = `ckpt:runs/ei2_value_t5/t_leafq/critic.pt`（从不经过注入
  factory）；search = 同一 base，包 `web_search.SearchFactory`（t=5 / K=32 /
  C=6，value 核心同一 critic.pt；与 P4a 测量身份 `rolloutt:<critic>` 等价）。
  Δ 只含“搜索 on/off”一个变量。
- **调度与 pairs**：`plan_twin_schedule` 为每个 pair 抽互异 seed；pair 内两局相邻
  （schedule index 2p、2p+1），seed/座位不变；`seat = (seat_start + pair) % 2`，
  先打哪只臂跨 pair 交替。默认 **30 对**，会话开始可另选 **2..100 的偶数**（页面数字输入；
  服务端硬校验偶数，<30 对只警告“仅流程验证”，不阻塞）。只可能正常打完（`max_pairs`）
  或 `quit` 中止，不做中途 early stop（optional stopping 会让 CI 失效）。
- **读数口径**：统计单位是**完整 pair**；未完成的 pair 单列入
  `pairs_incomplete` 且不进 Δ/CI。分辨率门槛 ≥30 对（60 局）——不足时 UI 与
  报告顶部标黄“仅流程验证”。
- **产物**：`traces/twins/<twin-id>/`（serve）或 `--out`（simulate）：每局一条
  可回放 trace（`…__vsraw.json` / `…__vssearch.json`，顶层带 `deal_twin` 注解；
  search 臂另写 `opponent_search`），`session.json` 每局落盘，结束写
  `report.json`。**twin 结果不进 manifest / prior / OpenSkill / 评分**，也不是
  定级；raw 对手路径不调用 `WebConfig.policy_factory`（只有 manifest 里非 stock grammar spec
  的搜索 rung 才走注入 factory，测试锁死）。
- **caveats（报告如实列出）**：open-label（对局中显示当前臂）、同 pair 第二局
  是同一副牌（CRN 记忆，顺序交替只在总体期望上抵消）、单 session 只有 1 个
  bank（疲劳/学习漂移只由 `half_split` 诊断）、描述性结论、search 臂机器 h2h
  ≈ **+135.7 Elo**（t_leafq・单位修复口径，confirm bank 501-505 [128.2,143.1]；旧文
  +124.7 是 critic_B 值核，不对应本次 pin；人对 search 的 Δ 预期为负；不显著 ≠
  两臂等价）、pair 内 seat 与先手臂同相（`seat_split`/`order_split` 是对同一批
  pair 的重复划分，不能分离交互）。
- **事后校验**（纯引擎，无 torch）：`runs/o4lite-search/p4b/verify_twin_sim.py
  <session-dir>` 检查 pair 内 seed/座位一致、臂分布、t5/K32/C6 pin、trace 全部
  replay 通过、raw 臂无 `opponent_search`、Δ/resolution 可用同
  `bootstrap_seed` 从 `session.json` 重算一致。

### 6.3 定级：单一 manifest 池（P6）

定级只有一条服务端路径：`{"mode":"placement","games":even,"use_prior":bool}`，
池是 manifest 的全部 `levels`：锚点 + raw 档 + 能构建的搜索 rung（同一 26,000 局联合拟合尺度，ADR-0013）。
2026-09-29 起 manifest 已含 `search_leafq`：服务端有注入 factory 且 `can_build_spec`
放行时它进池并被 Thompson/Fisher 调度（C3，owner 已接受）；没有 factory 的 stock 服务端
按能力门跳过并告警。请求带 `search_rung`（已删除的 rated 子池路径）或 `mode="unrated"`
（已删除的实验系列）都被明确 400，不静默忽略。

- **搜索 rung 身份**：`subjects[]` 中 `id=search_leafq` 的 `search_config` = `{trunc_ply:5, rollout_k:32,
  max_candidates:6, max_rollout_ply:400, rollout_opponent:"ckpt:runs/w2m_ctl__11__1790516900/agent.pt",
  aggregate:"mean"}`；value ckpt 就是 `spec`（`rolloutt:<path>`）。插件从 `--manifest` 读该块
  构建与 P4a 测量一致的包装（放置路径 `search={}`），自由对战仍可逐局覆盖 t/K。
- **轨迹先验按局排除**：写到搜索 rung 的那局其行不进轨迹通道，该局 `GameRecord.prior_off_reason`
  与会话级 `prior_off_reason` 都记录原因；`report.json` 的 `channels.trace` 给出 `n_traces` /
  `excluded_games` / `prior_off_reason`；raw 局的先验与权重不受影响（无 raw 局时轨迹权重为 0）。
- **产物**：仍是普通 `traces/sessions/<id>/`（session.json / rungs.json / report.json + 每局
  可回放 trace），不再有 `search_rung` / `rated` / `scale_note` 字段；`report.rungs` 列出全池
  （含每个 rung 的 spec 与 μ/σ 溯源）。
- **refit 保留**：`tools/refit_mle.py --manifest-out --refit --search-config ID=@file.json` 可
  seed/refresh 该块；不带该 flag 的 refit 逐字保留已发布块（μ/σ/games/rules_id 照旧重拟）。


## 7. 运行注意

- 命令都在仓库根目录执行；`play_ladder.py play` 会把 ckpt 解析成绝对路径，
  换目录也能跑。
- ckpt 训练/评估细节见 [`training.md`](./training.md)；本页只负责「打」。
- 新模型生成后不要在 `runs/` 里搬动或重命名 ckpt，直接引用路径；`list` 会
  标记缺失项。

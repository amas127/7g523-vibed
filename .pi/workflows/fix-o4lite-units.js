export const meta = {
  name: 'fix-o4lite-units',
  description: '修 O4-lite 截断 value 单位混用 → 真人局面 probe → 每点 1 个对抗审查 → fixed-vs-raw screen（≤12 进程 / ≤4 GPU）',
  whenToUse: '当需要修复搜索截断读出的单位混用并用真人局面与 h2h screen 验证效果时',
  phases: [
    { title: 'Fix', detail: '单代理改 rollout_trunc.py 单位（终局先后关系：必须先改再测）' },
    { title: 'Probe', detail: '真实 479 个真人决策重放：修复前后行为与 value 指标' },
    { title: 'Review', detail: '每个点 1 个对抗审查（修复语义）' },
    { title: 'Screen', detail: '3 seeds × 400 deals fixed vs raw（同 bank 30-32，对齐既有 trunc_k32 基线）' },
    { title: 'Verify', detail: 'screen 结果独立复核（每点 1 个）' },
  ],
}

const REPO = '/home/amas/.local/src/7g523'
const LIMITER = REPO + '/.pi/limits/run_limited.sh'
const PY = REPO + '/.venv/bin/python'
const CHAMP = 'ckpt:runs/w2m_ctl__11__1790516900/agent.pt'
const CRITIC = 'rolloutt:runs/ei/value/critic_B.pt'
const ARM = 'unitfix'
const OUT = 'runs/o4lite-search/' + ARM

const RESOURCE_RULES = [
  '【资源硬约束】机器 15GB / 32 核；全机同时 ≤12 个分析进程、≤4 个 GPU 进程。所有 bash 命令必须走：',
  '  ' + LIMITER + ' cpu <命令...>   # 纯标准库/文本/编译检查，最多 12 并发',
  '  ' + LIMITER + ' gpu <命令...>   # 任何 import torch / 加载 .pt / 重放：最多 4 并发',
  'sleep/cat/ls 也用 cpu 包装；script 写好后整脚本包一次执行；同一代理同一时刻只跑一个 gpu 进程。',
  '禁止用 uv run；python 一律 ' + PY + '。禁止派生子代理（Agent/SubagentWorkflow）。不改 src/、不 commit。',
].join('\n')

const FIX_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['status', 'file', 'change_summary', 'diff', 'checks', 'risks'],
  properties: {
    status: { type: 'string', enum: ['done', 'blocked'] },
    file: { type: 'string' },
    change_summary: { type: 'string' },
    diff: { type: 'string', description: 'git diff 的关键 hunk（原文，不截断关键行）' },
    checks: { type: 'array', items: { type: 'string' } },
    risks: { type: 'array', items: { type: 'string' } },
  },
}

const PROBE_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['status', 'script', 'decisions', 'fallback_count', 'changed_vs_recorded', 'changed_vs_raw', 'case_s3548246500', 'metrics', 'checks'],
  properties: {
    status: { type: 'string', enum: ['done', 'partial', 'blocked'] },
    script: { type: 'string' },
    decisions: { type: 'number' },
    fallback_count: { type: 'number' },
    changed_vs_recorded: { type: 'number', description: 'fixed 选择 != trace 记录动作的决策数' },
    changed_vs_raw: { type: 'number', description: 'fixed 选择 != raw argmax 的决策数' },
    case_s3548246500: { type: 'string', description: 'step37 修复后选了什么、修复前记录选了什么、margin 各是多少' },
    metrics: {
      type: 'object',
      description: '单位一致口径下的 EV/corr/MAE（fixed）；如与旧 buggy 口径对比也要给',
      properties: {
        ev: { type: 'number' },
        corr: { type: 'number' },
        mae: { type: 'number' },
        k_world_mean_mae: { type: 'number' },
        optimistic_bias: { type: 'number' },
      },
    },
    checks: { type: 'array', items: { type: 'string' } },
  },
}

const SCREEN_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['status', 'seeds', 'fallback_total', 'per_seed', 'combined', 'baseline_compare', 'kill_reasons', 'artifacts'],
  properties: {
    status: { type: 'string', enum: ['done', 'partial', 'failed', 'blocked'] },
    seeds: { type: 'array', items: { type: 'number' } },
    fallback_total: { type: 'number' },
    per_seed: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          seed: { type: 'number' },
          winrate: { type: 'number' },
          elo: { type: 'number' },
          elo_ci: { type: 'array', items: { type: 'number' } },
          games_per_min: { type: 'number' },
        },
      },
    },
    combined: { type: 'object', properties: { elo: { type: 'number' }, elo_ci: { type: 'array', items: { type: 'number' } } } },
    baseline_compare: { type: 'string', description: '与既有 trunc_k32 seeds30-32（buggy）逐 seed 对比' },
    kill_reasons: { type: 'array', items: { type: 'string' } },
    artifacts: { type: 'array', items: { type: 'string' } },
  },
}

const VERDICT_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['claim_id', 'verdict', 'confidence', 'reproduced', 'findings', 'counter_evidence', 'corrected_claim', 'checks_run'],
  properties: {
    claim_id: { type: 'string' },
    verdict: { type: 'string', enum: ['confirmed', 'weakened', 'refuted', 'undecidable'] },
    confidence: { type: 'string', enum: ['low', 'medium', 'high'] },
    reproduced: { type: 'boolean' },
    findings: { type: 'array', items: { type: 'string' } },
    counter_evidence: { type: 'array', items: { type: 'string' } },
    corrected_claim: { type: 'string' },
    checks_run: { type: 'array', items: { type: 'string' } },
  },
}

// ---------------------------------------------------------------------------
// Phase 1 — Fix (serial; everything downstream depends on this edit)
// ---------------------------------------------------------------------------
phase('Fix')
log('Phase Fix：单代理修 rollout_trunc.py 单位混用')

const FIX_PROMPT = [
  '仓库 cwd=' + REPO + '（7鬼523 RL；O4-lite 是推理期截断 lookahead）。',
  '背景（主控已核实，可复核）：`runs/o4lite-search/rollout_trunc.py` 的 `ValueChampion`（line ~60-163）把两类不同单位的数写进同一个 values 数组：',
  '  - 终局状态用 `BatchChampion._margin`（runs/o4lite-search/rollout_policy.py:429-430）= `scores[seat] - scores[1-seat]`，单位是**分**（范围 ±100）；',
  '  - 截断状态用 `self._values(views)`（rollout_trunc.py:74-84）= critic 头读出；champion 训练配置 `runs/w2m_ctl__11__1790516900/args.json:57` 是 `reward_shaping="terminal"`，即训练目标为 `Game.returns()`（src/seven523/game.py:338-357）= `own/100 - mean(others)/100`，单位是**归一化分差**（范围 ±1）。',
  '两者在 `rollout_values`（line 139-156）混存，随后被 `rollout_policy.batched_decide`（line 555-568）跨 32 个世界直接 `np.mean`。这是本次要修的 bug。',
  '',
  '任务：做**最小修复**，使 values 数组在所有分支统一为「分」单位：',
  '  - 在 rollout_trunc.py 截断读出分支（line 149-156），把 critic 读出乘以 `self.rules.total_points`（2 家=100），保留原有的 signs flip（`states[i].current == seats[i]` 判断）与 float 转换；',
  '  - 不改 `_margin`、不改 `rollout_policy.py`、不改任何 `src/` 文件；更新该处注释说明单位；',
  '  - 同时检查 `runs/o4lite-search/rollout_depth2.py`（line 102/130/146-154 消费同一 values 列表）与 `mechanism_probe.py`（line 290）是否随修复自然一致，只报告结论、不要改它们；',
  '  - 用 `git diff -- runs/o4lite-search/rollout_trunc.py` 给出完整改动。',
  '',
  '自检（必须做，走 limiter cpu）：',
  '  - ' + LIMITER + ' cpu ' + PY + ' -m py_compile runs/o4lite-search/rollout_trunc.py runs/o4lite-search/rollout_policy.py',
  '  - 写一个 5 行内的纯断言脚本（可放 /tmp），用 `ValueChampion` 的类属性/源码检查确认 `total_points` 在 `Rules` 上存在且 2 家时为 100（`from seven523.rules import DEFAULT_RULES`，可 import）。',
  '  - 不改测试目录（tests/ 是 src 的范围），不新增 tests/ 文件。',
  '',
  '输出：status=done/blocked；file 写 `runs/o4lite-search/rollout_trunc.py`；diff 给关键 hunk；checks 写实际跑过的命令；risks 写你担心的边界（例如 n>2、rollout_depth2 的 payload 混合、critic 是否可能不是 terminal shaping）。',
  '',
  RESOURCE_RULES,
].join('\n')

const fix = await agent(FIX_PROMPT, {
  label: 'fix-units',
  phase: 'Fix',
  schema: FIX_SCHEMA,
  agentType: 'general-purpose',
})

if (!fix || fix.status !== 'done') {
  log('Fix 未完成，终止后续（不启动 probe/screen）')
  return { phase: 'Fix', fix, stopped: true }
}
log('Fix 完成：' + fix.change_summary)

// ---------------------------------------------------------------------------
// Phase 2+3 — Probe（gpu）与 修复语义对抗审查（cpu）并行
// ---------------------------------------------------------------------------
phase('Probe')

const PROBE_PROMPT = [
  '仓库 cwd=' + REPO + '。你是 probe 子代理：用**刚修复过**的 O4-lite 截断 wrapper 重放真实人机轨迹，量化修复效果。',
  '',
  '背景：',
  '- 数据：`traces/web/run-20260928-091716/` 中 22 个 `vsw2m_ctl`、`opponent_search={"trunc_ply":5,"rollout_k":32}` 的局，共 479 个 bot 决策（旧的研究探针口径：479 decisions / 370 searched / 229 改判）。另有 5 个无搜索局与 1 个 vs random 搜索局，排除。',
  '- 旧记录的动作是在**有 bug**的 wrapper 下做出的（截断 critic 读出未乘 100）。修复后预期：`s3548246500` step37 应从单牌 ♣6（critic 读出 0.0816）改回对子 6-6（真值 margin=0 的确定撬底）。',
  '- 构造口径（部署路径）：`base = policy_from_spec("' + CHAMP + '")`；截断容器用 `runs/o4lite-search/rollout_trunc.py` 的 `TruncFactory` 等价构造，spec `' + CRITIC + '`，`K=32`、`C=6`、`max_rollout_ply=400`、`O4_TRUNC_PLY=5`、rollout 对手 = `' + CHAMP + '`；每局种子用 `policy_seed(trace.seed, trace.human_seat)`（`src/seven523/record.py`）。重放 479/479 必须与记录动作一致（修复前），先跑一遍作 sanity；再看修复后的选择。',
  '',
  '任务（写成脚本 `runs/o4lite-search/unitfix/probe_human.py`，输出 `runs/o4lite-search/unitfix/probe.json`）：',
  '1) fixed 重放 479 个决策：fallback_count、changed_vs_recorded、changed_vs_raw；',
  '2) `g0000__s3548246500` step37 的修复后选择与 margin 对比；',
  '3) 若时间允许（总计 ≤20 分钟计算）：单位一致口径的 EV/corr/MAE（chosen vs 单世界 champion 全量 margin；K 世界均值口径的 MAE 与系统性乐观偏差），并给修复前（把乘数设回 1）的同口径对照。旧研究给过 buggy 口径 EV=0.081/corr=0.333/MAE=33.7、单位一致 K32 均值 MAE≈14.4/偏差 +13.6；你的数如实报，不要凑。',
  '4) probe.json 必须含：decisions、fallback_count、changed_vs_recorded、changed_vs_raw、case_s3548246500、metrics、script、checks。',
  '',
  '纪律：所有 torch 命令走 `' + LIMITER + ' gpu`；一个进程合并算完，预计 >3 分钟就拆最小子集先出结论并标 partial。不改仓库既有文件（只新增 ' + OUT + '/）；不 commit。',
  '',
  RESOURCE_RULES,
].join('\n')

const REVIEW_FIX_PROMPT = [
  '仓库 cwd=' + REPO + '。你是**对抗审查**子代理，独立审查一个 claim（每点只你 1 个审查者）。',
  '',
  '待审 claim [FIX]：`runs/o4lite-search/rollout_trunc.py` 的修复（截断 critic 读出 ×`rules.total_points`）在语义上把 values 数组统一成了「分」单位，正确、完备、无副作用。',
  '背景：critic 训练目标（champion `runs/w2m_ctl__11__1790516900/args.json:57` `reward_shaping="terminal"`）是 `Game.returns()`=`own/100 - mean(others)/100`；`_margin` 是 `scores[seat]-scores[1-seat]`（分）。修复前的 bug 是两者在 `rollout_values` 混存、`batched_decide` 跨世界 `np.mean`。',
  '',
  '攻击三轴（必须全部做，最后给一个综合裁决）：',
  '(a) 独立复现：不看修复代理的 diff 叙述，自己读 `git diff -- runs/o4lite-search/rollout_trunc.py`、`rollout_policy.py:391-500`、`src/seven523/game.py:338-357`、`env.py:280-312`，逐行确认修复公式与符号；用一个小脚本（limiter cpu 即可，不需要 torch）验证乘数 `rules.total_points` 与 2 家 `returns` 的换算：`returns*100 == own - other`。',
  '(b) 替代解释/完备性：找**所有**消费 `rollout_values`/`_margin`/`_values` 的路径（grep 全仓库：mechanism_probe、rollout_depth2、play_search、web_search、run_h2h），确认修复后没有第二条混单位路径、没有第二处需要乘 100 的读出（例如 depth2 的 payload、abs_trunc 的统计）；检查 n>2 家时的语义（`_margin` 只支持 2 家吗？`returns` 的 mean(others) 与 `_margin` 的 1-seat 在 2 家下才等价，确认本仓库固定 num_players=2 的证据）。',
  '(c) 统计/适用域：修复只影响「同一候选的 K 个世界里终局与截断混存」或「不同候选终局/截断构成不同」的决策；确认 ×100 对「全截断候选之间」的 argmax 无影响，并列出修复**不能**修复的东西（critic 的 +13.6 乐观偏差、K 世界均值噪声、determinization 先验）。',
  '',
  '纪律：只读仓库；需要写脚本放 /tmp/adv/FIX/；torch 任务走 gpu 槽。禁止嵌套子代理。',
  'verdict 含义：confirmed=你独立复现且 claim 成立；weakened=方向对但过强/不完备（给 corrected_claim）；refuted=不成立；undecidable=证据不足。不能独立复现不得投 confirmed。',
  '',
  RESOURCE_RULES,
].join('\n')

const [probe, fixReview] = await parallel([
  () => agent(PROBE_PROMPT, {
    label: 'probe-human',
    phase: 'Probe',
    schema: PROBE_SCHEMA,
    agentType: 'general-purpose',
    gate: PY + ' -c "import json;d=json.load(open(\'runs/o4lite-search/unitfix/probe.json\'));assert d[\'decisions\']==479;assert d[\'fallback_count\']==0"',
  }),
  () => agent(REVIEW_FIX_PROMPT, {
    label: 'review-fix',
    phase: 'Review',
    schema: VERDICT_SCHEMA,
    agentType: 'general-purpose',
  }),
])

if (!probe || probe.status === 'blocked') {
  log('Probe 失败/阻塞，不启动 screen')
  return { fix, probe, fixReview, stopped: true }
}
log('Probe：changed_vs_recorded=' + probe.changed_vs_recorded + ' changed_vs_raw=' + probe.changed_vs_raw)

if (fixReview && fixReview.verdict === 'refuted') {
  log('修复语义被对抗审查投 refuted，不启动 screen（避免测一个坏修复）')
  return { fix, probe, fixReview, stopped: true }
}

// ---------------------------------------------------------------------------
// Phase 4 — Screen（gpu ×3，同 bank 30/31/32 对齐既有 trunc_k32 基线）
// ---------------------------------------------------------------------------
phase('Screen')

const SCREEN_PROMPT = [
  '仓库 cwd=' + REPO + '。你是启动/核验子代理：跑 fixed-vs-raw 的 h2h screen 并落盘结果。',
  '',
  '配置必须与既有 `runs/o4lite-search/trunc_k32/seed{30,31,32}.json` 完全一致（同 bank、同 K、同 critic、同对手），只有 wrapper 是刚修复过的：',
  '  left: `' + CRITIC + '`（t=5 截断）；right: `' + CHAMP + '`；seeds 30,31,32；pairs=400；--rollout-k 32 --rollout-max-candidates 6 --max-rollout-ply 400 --rollout-opponent ' + CHAMP + '；`.venv/bin/python runs/speed/run_h2h_fast.py --arm ' + ARM + ' --seed <s> --pairs 400 --left-id trunc --right-id raw --device cpu --workers 1 --out ' + OUT + ' --tb runs/o4lite-search/tb/' + ARM,
  '',
  '执行方式（一个 bash 调用内完成，禁止逐条轮询）：',
  '1) `mkdir -p ' + OUT + '`；对 s=30,31,32 各启动一个后台进程，每个**单独**包 `' + LIMITER + ' gpu`，stdout/stderr 到 `' + OUT + '/seed<s>.log`，并把整个 `for ... & wait` 放在 `timeout 4200 bash -c ...` 下；',
  '2) wait 全部结束后 `' + LIMITER + ' cpu ' + PY + ' runs/o4lite-search/combine.py --arm ' + ARM + ' --in ' + OUT + ' --tb runs/o4lite-search/tb/' + ARM + ' --seeds 30,31,32`；',
  '3) 读 3 个 seed 的 json 与 combine 输出，与既有 `runs/o4lite-search/trunc_k32/seed{30,31,32}.json` 逐 seed 对比（同 bank 同 deal，paired 视角），并检查每个 seed 的 `kill_reasons`、`fallback_count`、`rollout_terminal_rate`；',
  '4) 如果 timeout：`pkill -f run_h2h_fast`，status=partial，如实报 0-3 个完成的 seed；任何进程都不许留在后台。',
  '',
  '输出 `' + OUT + '/screen.json`（schema 字段见任务描述），并在这里返回同样字段。expected：每 seed ~800 games、~18-20 分钟、43+ games/min；3 个 seed 并行。不许改 rollout 代码（修复已由上游完成），只允许新增 ' + OUT + '/ 下文件。',
  '',
  RESOURCE_RULES,
].join('\n')

const screen = await agent(SCREEN_PROMPT, {
  label: 'screen-fixed',
  phase: 'Screen',
  schema: SCREEN_SCHEMA,
  agentType: 'general-purpose',
  gate: PY + ' -c "import json;d=json.load(open(\'runs/o4lite-search/unitfix/screen.json\'));assert d[\'status\']==\'done\';assert sorted(d[\'seeds\'])==[30,31,32];assert d[\'fallback_total\']==0"',
})

const REVIEW_EFFECT_PROMPT = [
  '仓库 cwd=' + REPO + '。你是**对抗审查**子代理，独立审查一个 claim（每点只你 1 个审查者）。',
  '',
  '待审 claim [SCREEN]：screen 代理报告的「fixed（单位修复后）t=5/K=32 vs raw，seeds 30-32，3×400 deals」的结果（见 ' + OUT + '/screen.json 与其自述）是真实的，且相对既有 buggy trunc_k32 seeds30-32 的变化（正/负/零）不是流水线或口径伪影。',
  '',
  '攻击三轴：',
  '(a) 独立复现：从原始 `' + OUT + '/seed{30,31,32}.games.jsonl` 与 `runs/o4lite-search/trunc_k32/seed{30,31,32}.games.jsonl` 用你自己的脚本重算 winrate/Elo（公式 src/seven523/duel.py:71-74）与逐 deal 对比；核对两边 deal schedule 是否为同一批（seed 30/31/32、400 deals、换座）；核对 config（left/right spec、rollout_k、candidates、opponent、O4_TRUNC_PLY）逐字段；',
  '(b) 替代解释：找出任何能解释「变化/不变化」的非修复原因——seed RNG、代码路径（是否真的 import 了修复后的 rollout_trunc.py：检查进程启动时间与 rollout_trunc.py mtime）、GPU/CPU 数值差、fallback、aborted rollouts、bank 复用史；',
  '(c) 统计/适用域：3 seeds 的 CI、bank 复用之下的结论范围；若变化不显著，明确说「在 3×400 下无法分辨」，不要外推。',
  '',
  '纪律：只读 + /tmp/adv/SCREEN/ 脚本；torch 重放走 gpu 槽；不改仓库；禁止嵌套子代理。verdict 规则同标准（confirmed/weakened/refuted/undecidable）。',
  '',
  RESOURCE_RULES,
].join('\n')

const screenReview = await agent(REVIEW_EFFECT_PROMPT, {
  label: 'review-screen',
  phase: 'Verify',
  schema: VERDICT_SCHEMA,
  agentType: 'general-purpose',
})

log('全部完成：fix/probe/review-fix/screen/review-screen')
return {
  fix,
  probe,
  fixReview,
  screen,
  screenReview,
}

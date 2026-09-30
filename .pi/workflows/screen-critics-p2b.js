export const meta = {
  name: 'screen-critics-p2b',
  description: 'P2b：4 个重训 critic 按离线希望顺序逐个 h2h（same-bank 30-32，配对 A0/raw），每个点 1 个对抗审查（≤12 进程 / ≤4 GPU）',
  whenToUse: '当需要把 P2 离线评测过的 critic 候选逐个放进 h2h 验证是否值得采纳时',
  phases: [
    { title: 'Preflight', detail: '校验 4 个 ckpt 存在且 actor 与 champion 逐位一致，冻结运行顺序' },
    { title: 'Screens', detail: 't_leafq → t_rawq → t_searchq_all → t_searchq，逐个 3 seeds × 400 deals' },
    { title: 'Review', detail: '每个臂 1 个对抗审查（policy 同源 / deal 同批 / 统计）' },
  ],
}

const REPO = '/home/amas/.local/src/7g523'
const LIMITER = REPO + '/.pi/limits/run_limited.sh'
const PY = REPO + '/.venv/bin/python'
const CHAMP = 'ckpt:runs/w2m_ctl__11__1790516900/agent.pt'
const BASE = 'runs/o4lite-search/unitfix'   // A0 = fixed-mean critic_B, seeds 30-32
const OUT = 'runs/o4lite-search/valuecal/screens'
const RUNNER = 'runs/speed/run_h2h_trunc.py'

const CANDIDATES = [
  { id: 't_leafq', ckpt: 'runs/ei2_value_t5/t_leafq/critic.pt', why: '离线决策质量 +0.304 [+0.04,+0.60]，bias -1.82，EV 0.930，MAE 6.31' },
  { id: 't_rawq', ckpt: 'runs/ei2_value_t5/t_rawq/critic.pt', why: 'bias +4.60，EV 0.897，决策质量 -0.02' },
  { id: 't_searchq_all', ckpt: 'runs/ei2_value_t5/t_searchq_all/critic.pt', why: 'bias +9.28，EV 0.821，决策质量 -0.00' },
  { id: 't_searchq', ckpt: 'runs/ei2_value_t5/t_searchq/critic.pt', why: '冻结主候选，bias +10.15，EV 0.800，决策质量 -0.08' },
]

const RESOURCE_RULES = [
  '【资源硬约束】机器 15GB / 32 核；全机同时 ≤12 个分析进程、≤4 个 GPU 进程。',
  '  ' + LIMITER + ' cpu <命令...>   # 纯标准库/文本/编译/统计，最多 12 并发',
  '  ' + LIMITER + ' gpu <命令...>   # 任何 import torch / 加载 .pt / 重放 / h2h：最多 4 并发',
  'sleep/ls 也用 cpu 包装；整脚本包一次执行；同一代理同一时刻只跑一个 gpu 进程。',
  '禁止 uv run；python 一律 ' + PY + '；禁止派生子代理（Agent/SubagentWorkflow）；不改 src/、不改 rollout 代码、不 commit。',
].join('\n')

const PREFLIGHT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['status', 'order', 'actor_identity', 'checks'],
  properties: {
    status: { type: 'string', enum: ['done', 'blocked'] },
    order: { type: 'array', items: { type: 'string' }, description: '冻结后的候选 id 顺序（只保留 actor 逐位一致的）' },
    actor_identity: { type: 'string', description: '每个 ckpt 与 champion 的 actor/trunk 参数对比结果（max abs diff / 是否一致）' },
    checks: { type: 'array', items: { type: 'string' } },
  },
}

const ARM_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['status', 'arm', 'seeds_done', 'fallback_total', 'per_seed', 'vs_a0', 'vs_raw', 'kill_reasons', 'artifacts'],
  properties: {
    status: { type: 'string', enum: ['done', 'partial', 'failed'] },
    arm: { type: 'string' },
    seeds_done: { type: 'array', items: { type: 'number' } },
    fallback_total: { type: 'number' },
    per_seed: { type: 'array', items: { type: 'object', properties: { seed: { type: 'number' }, winrate: { type: 'number' }, elo: { type: 'number' } } } },
    vs_a0: { type: 'string', description: '与 unitfix A0 seeds 30-32 逐 deal 配对的 d_win/d_margin/d_elo + CI' },
    vs_raw: { type: 'string' },
    kill_reasons: { type: 'array', items: { type: 'string' } },
    artifacts: { type: 'array', items: { type: 'string' } },
  },
}

const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
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
// Phase 1 — Preflight
// ---------------------------------------------------------------------------
phase('Preflight')
log('Preflight：校验 4 个 critic ckpt 的 actor 与 champion 逐位一致')

const PREFLIGHT_PROMPT = [
  '仓库 cwd=' + REPO + '。你是 preflight 子代理，只读 + 写 /tmp。',
  '候选（按离线希望顺序）：',
  CANDIDATES.map(c => '  - ' + c.id + ': ' + c.ckpt + '（' + c.why + '）').join('\n'),
  '',
  '任务：产出 `' + OUT + '/plan.json`：',
  '1) 确认每个 ckpt 文件存在、可加载；',
  '2) **actor/trunk 逐位一致**：用 `seven523.networks.load_agent`（或等价）分别加载 champion `runs/w2m_ctl__11__1790516900/agent.pt` 与每个候选 ckpt，对**除 critic/value head 外的全部参数**逐张量比较 max|Δ|（应全为 0）；把每个候选的 max|Δ| 与是否一致写进 actor_identity；',
  '3) 若某候选 actor 不一致，从 order 中剔除并说明；顺序保持 t_leafq → t_rawq → t_searchq_all → t_searchq（去掉被剔的）；',
  '4) 顺带核对 `runs/o4lite-search/valuecal/offline.json` 里这 4 个候选的决策质量/bias 数字与计划一致。',
  '纪律：torch 走 gpu 槽（加载 5 个 ckpt 一个小进程合并完成）；不改仓库；不 commit。',
  '',
  RESOURCE_RULES,
].join('\n')

const preflight = await agent(PREFLIGHT_PROMPT, {
  label: 'preflight', phase: 'Preflight', schema: PREFLIGHT_SCHEMA, agentType: 'general-purpose',
  gate: PY + ' -c "import json;d=json.load(open(\'runs/o4lite-search/valuecal/screens/plan.json\'));assert d[\'status\']==\'done\';assert len(d[\'order\'])>=1"',
})
if (!preflight || preflight.status !== 'done' || !preflight.order.length) {
  log('Preflight 失败或没有可用候选，终止')
  return { preflight, stopped: true }
}
log('Preflight 完成，顺序：' + preflight.order.join(' → '))

// ---------------------------------------------------------------------------
// Phase 2 — Screens（逐个臂，串行）
// ---------------------------------------------------------------------------
phase('Screens')

const armResults = []
for (const id of preflight.order) {
  const cand = CANDIDATES.find(c => c.id === id)
  if (!cand) continue
  const arm = 'vcal_' + id
  const armOut = OUT + '/' + id

  const PROMPT = [
    '仓库 cwd=' + REPO + '。你是 screen 子代理，只跑这一个臂：**' + id + '**。',
    'left spec = `rolloutt:' + cand.ckpt + '`；right = `' + CHAMP + '`；seeds 30,31,32；pairs=400；K=32/C=6/--max-rollout-ply 400/--rollout-opponent ' + CHAMP + '；runner `' + RUNNER + '`；env `O4_TRUNC_PLY=5 O4_AGG=mean`。',
    '基线 A0：`' + BASE + '/seed{30,31,32}.games.jsonl`（fixed-mean critic_B，同 bank）。',
    '',
    '执行步骤（第 1 步是一个 bash 调用；3 个 seed 并行、各自包 ' + LIMITER + ' gpu，整段 `timeout 2600`）：',
    '1) mkdir -p ' + armOut + '；对 s∈{30,31,32} 启动：',
    '   `' + LIMITER + ' gpu ' + PY + ' ' + RUNNER + ' --arm ' + arm + ' --seed $s --pairs 400 --left-id ' + id + ' --right-id raw --left-spec rolloutt:' + cand.ckpt + ' --right-spec ' + CHAMP + ' --rollout-k 32 --rollout-max-candidates 6 --max-rollout-ply 400 --rollout-opponent ' + CHAMP + ' --device cpu --workers 1 --out ' + armOut + ' --tb runs/o4lite-search/tb/' + arm + '`',
    '   日志写 ' + armOut + '/seed$s.log；`wait`；然后检查每个 seed 的 kill_reasons、fallback_count、rollout_terminal_rate；',
    '2) `' + LIMITER + ' cpu ' + PY + ' runs/o4lite-search/combine.py --arm ' + arm + ' --in ' + armOut + ' --tb runs/o4lite-search/tb/' + arm + ' --seeds 30,31,32`；',
    '3) **vs_a0 配对**：读 `runs/o4lite-search/paired_compare.py`（或自写等价脚本），把 ' + armOut + '/seed{30,31,32}.games.jsonl 与 ' + BASE + '/seed{30,31,32}.games.jsonl 逐 deal 配对，给 d_win/d_margin/d_elo 与 CI、以及 vs raw 的 Elo/winrate；',
    '4) 写 `' + armOut + '/arm_' + id + '.json`（字段见返回 schema），并把它 append/总结进 `' + OUT + '/run_log.jsonl`。',
    '',
    '判定（预注册）：vs_a0 的配对 d_margin CI 下界 ≥ 0 且 winrate 点估计不降 → 记为 `promising`；CI 完全 <0 → `kill`；否则 `tie`。',
    '超时/失败：`pkill -f run_h2h`，如实报 partial，不要阻塞下一臂。任何后台进程不许留下。',
    '只新增 ' + armOut + '/ 下文件；不 commit。',
    '',
    RESOURCE_RULES,
  ].join('\n')

  const r = await agent(PROMPT, {
    label: 'screen-' + id, phase: 'Screens', schema: ARM_SCHEMA, agentType: 'general-purpose',
    gate: PY + ' -c "import json;d=json.load(open(\'runs/o4lite-search/valuecal/screens/' + id + '/arm_' + id + '.json\'));assert d[\'status\'] in (\'done\',\'partial\')"',
  })
  armResults.push({ id, candidate: cand, result: r })
  log('Screen ' + id + '：' + (r ? r.status + ' vs_a0=' + String(r.vs_a0).slice(0, 120) : 'FAILED'))
}

const doneArms = armResults.filter(a => a.result && a.result.status !== 'failed')
if (!doneArms.length) {
  log('所有臂都失败，终止')
  return { preflight, armResults, stopped: true }
}

// ---------------------------------------------------------------------------
// Phase 3 — Review（每个臂 1 个对抗审查，并行）
// ---------------------------------------------------------------------------
phase('Review')
log('Review：' + doneArms.length + ' 个臂各 1 个对抗审查')

const reviewPromises = doneArms.map(a => () => {
  const id = a.id
  const arm = 'vcal_' + id
  const armOut = OUT + '/' + id
  const prompt = [
    '仓库 cwd=' + REPO + '。你是对抗审查子代理，独立审查一个 claim（每点只你 1 个审查者）。',
    '待审 claim [SCREEN-' + id + ']：screen 子代理报告的「' + id + ' vs A0（unitfix fixed-mean）在 bank 30-32 的 h2h 差异」是真实的，不是 runner/版本/deal 错配/policy 不同源造成的。',
    '证据：`' + armOut + '/arm_' + id + '.json`、`' + armOut + '/seed*.games.jsonl|json|log`、基线 `' + BASE + '/seed{30,31,32}.games.jsonl`、preflight `' + OUT + '/plan.json`。',
    '',
    '三轴攻击：',
    '(a) 独立复现：从两边 games.jsonl 自己重算 winrate/Elo（src/seven523/duel.py:71-74）与逐 deal 配对（d_win/d_margin/d_elo），核对 deal schedule 是否与 A0 完全同批（集合与顺序、可用 random.Random(seed).randrange(2**32) 重建）；',
    '(b) 替代解释：actor 是否与 champion 逐位一致（preflight 之外可自己抽验权重）；env（O4_TRUNC_PLY=5、O4_AGG=mean）是否真进入子进程；runner 是否用了修复后的 rollout_trunc；有无 fallback/abort；',
    '(c) 统计/适用域：3 seeds × 400 deals 的配对分辨力、bank 复用史、与 P1 的 CVaR 教训对照（离线好看但 h2h 变差的先例）；结论只对 raw/champion/bank 30-32 有效。',
    'verdict 规则：confirmed/weakened/refuted/undecidable；不能独立复现不得投 confirmed。只读仓库与 /tmp/adv/P2B' + id + '/；禁止嵌套子代理。',
    '',
    RESOURCE_RULES,
  ].join('\n')
  return agent(prompt, { label: 'review-' + id, phase: 'Review', schema: VERDICT_SCHEMA, agentType: 'general-purpose' })
    .then(v => ({ id, review: v }))
})

const reviews = await parallel(reviewPromises)
log('全部完成')
return { preflight, armResults, reviews: reviews.filter(Boolean) }

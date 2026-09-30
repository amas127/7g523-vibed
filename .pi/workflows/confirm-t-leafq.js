export const meta = {
  name: 'confirm-t-leafq',
  description: 't_leafq confirm：fresh bank 500-504，A0 与 t_leafq 各 5×845 deals 同码重跑，配对判定 ≥+10 Elo（≤4 GPU / ≤12 进程）',
  whenToUse: '当 screen 阶段 t_leafq 显示 promising，需要在独立 bank 上做确认实验时',
  phases: [
    { title: 'Preflight', detail: '校验 fresh bank 未被复用、actor 同源、写 confirm 计划' },
    { title: 'Run', detail: 'A0 与 t_leafq 各 5 seeds × 845 deals（分 3 波过 4 GPU 槽）→ combine → 配对判定' },
    { title: 'Review', detail: 'confirm claim 的对抗审查（bank 新鲜度 / 同码 / 配对统计）' },
  ],
}

const REPO = '/home/amas/.local/src/7g523'
const LIMITER = REPO + '/.pi/limits/run_limited.sh'
const PY = REPO + '/.venv/bin/python'
const CHAMP = 'ckpt:runs/w2m_ctl__11__1790516900/agent.pt'
const RUNNER = 'runs/o4lite-search/run_h2h_trunc.py'
const OUT = 'runs/o4lite-search/valuecal/confirm'
const BANK = [501, 502, 503, 504, 505]

const ARMS = [
  { id: 'confirm_a0', spec: 'rolloutt:runs/ei/value/critic_B.pt', why: 'A0 对照：当前代码 + fixed-mean critic_B' },
  { id: 'confirm_leafq', spec: 'rolloutt:runs/ei2_value_t5/t_leafq/critic.pt', why: 't_leafq：leaf_mc 训练的价值头（screen 唯一 promising）' },
]

const RESOURCE_RULES = [
  '【资源硬约束】机器 15GB / 32 核；全机同时 ≤12 个分析进程、≤4 个 GPU 进程。',
  '  ' + LIMITER + ' cpu <命令...>   # 纯标准库/文本/统计，最多 12 并发',
  '  ' + LIMITER + ' gpu <命令...>   # 任何 h2h / torch：最多 4 并发（多出的排队）',
  'sleep/ls 也用 cpu 包装；整脚本包一次执行；同一代理同一时刻只跑一个 gpu 进程。',
  '禁止 uv run；python 一律 ' + PY + '；禁止派生子代理（Agent/SubagentWorkflow）；不改 src/、不改 rollout 代码、不 commit。',
].join('\n')

const PLAN_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['status', 'bank', 'arms', 'freshness', 'checks'],
  properties: {
    status: { type: 'string', enum: ['done', 'blocked'] },
    bank: { type: 'array', items: { type: 'number' } },
    arms: { type: 'array', items: { type: 'string' } },
    freshness: { type: 'string', description: 'seed 500-504 是否在既有 runs/ 产物里出现过（列出搜索证据）' },
    checks: { type: 'array', items: { type: 'string' } },
  },
}

const RUN_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['status', 'arms', 'per_seed', 'paired', 'confirm_verdict', 'kill_reasons', 'artifacts'],
  properties: {
    status: { type: 'string', enum: ['done', 'partial', 'failed'] },
    arms: { type: 'array', items: { type: 'string' } },
    per_seed: { type: 'array', items: { type: 'object', properties: { arm: { type: 'string' }, seed: { type: 'number' }, winrate: { type: 'number' }, elo: { type: 'number' } } } },
    paired: { type: 'string', description: 't_leafq − A0 逐 deal 配对：d_win/d_margin/d_elo 与 deal-bootstrap z-CI + between-seed t-CI' },
    confirm_verdict: { type: 'string', description: '按门槛判定：z 与 t CI 同时排除 0 且点估计 ≥ +10 Elo → confirmed；否则 not_confirmed（附实得区间）' },
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
log('Preflight：fresh bank 校验 + confirm 计划')

const PREFLIGHT_PROMPT = [
  '仓库 cwd=' + REPO + '。你是 preflight 子代理。只读 + 写 ' + OUT + '/plan.json（/tmp 脚本可写）。',
  '目的：为 t_leafq confirm 准备独立 bank。seed 500 已在上一次 preflight 被抓到复用（runs/w2m/calibration，400/845 deal 重叠）；本次请求 bank = ' + BANK.join(',') + '（已在上一轮验证零重叠，请复验后把 status 写为 done）。',
  '计划：bank = ' + BANK.join(',') + '；两个臂：' + ARMS.map(a => a.id + '=' + a.spec).join('；') + '；每臂 seeds × 845 deals；runner `' + RUNNER + '`；K=32/C=6/max-rollout-ply 400/rollout 对手 ' + CHAMP + '；env `O4_TRUNC_PLY=5 O4_AGG=mean`。',
  '',
  '任务：',
  '1) **fresh 校验**：复验 ' + BANK.join(',') + ' 在 `runs/` 与 `traces/` 下零复用（上一轮已验 501-510/515 clean；给出搜索命令与证据）；**重写 `' + OUT + '/plan.json`，status 必须为 `done`**（不要沿用上一份 blocked 的）；若某个 seed 被占用，换成 506+ 并同样验证；',
  '2) **actor 同源**：快速复核两个臂 ckpt（critic_B、t_leafq）的 actor/trunk 与 champion 逐位一致（可引用 `runs/o4lite-search/valuecal/screens/plan.json`，但最好用一个合并的小 torch 进程重验）；',
  '3) **写 plan.json**：bank、arms、freshness、每条实际命令（含 3 波调度：先 4 个 job、再 4 个、再 2 个，经 ' + LIMITER + ' gpu 排队），供 Run 阶段直接使用；',
  '4) 确认 ' + OUT + '/ 目录为空或只含 plan.json。',
  '纪律：torch 走 gpu 槽单进程；不改仓库；不 commit。',
  '',
  RESOURCE_RULES,
].join('\n')

const preflight = await agent(PREFLIGHT_PROMPT, {
  label: 'confirm-preflight', phase: 'Preflight', schema: PLAN_SCHEMA, agentType: 'general-purpose',
  gate: PY + ' -c "import json;d=json.load(open(\'runs/o4lite-search/valuecal/confirm/plan.json\'));assert d[\'status\']==\'done\';assert len(d[\'bank\'])==5"',
})
if (!preflight || preflight.status !== 'done') {
  log('Preflight 失败，终止')
  return { preflight, stopped: true }
}
log('Preflight 完成：bank=' + preflight.bank.join(','))

// ---------------------------------------------------------------------------
// Phase 2 — Run
// ---------------------------------------------------------------------------
phase('Run')
log('Run：两个臂 5×845 deals（3 波过 4 GPU 槽）')

const job = (arm, seed) => [
  LIMITER + ' gpu timeout 3000 ' + PY + ' ' + RUNNER,
  '--arm ' + arm.id,
  '--seed ' + seed,
  '--pairs 845',
  '--left-id ' + arm.id,
  '--right-id raw',
  '--left-spec ' + arm.spec,
  '--right-spec ' + CHAMP,
  '--rollout-k 32 --rollout-max-candidates 6 --max-rollout-ply 400',
  '--rollout-opponent ' + CHAMP,
  '--device cpu --workers 1',
  '--out ' + OUT + '/' + arm.id,
  '--tb runs/o4lite-search/tb/' + arm.id,
  '> ' + OUT + '/' + arm.id + '.seed' + seed + '.log 2>&1',
].join(' ')

const wave = (jobs) => jobs.map(j => '(' + j + ') &').join('\n') + '\nwait'

const RUN_PROMPT = [
  '仓库 cwd=' + REPO + '。你是 confirm 运行子代理。',
  '计划：`' + OUT + '/plan.json`。bank=' + BANK.join(',') + '；两臂 ' + ARMS.map(a => a.id).join('/') + '；每臂 5 seeds × 845 deals；runner `' + RUNNER + '`。',
  '',
  '调度（3 个 bash 调用，每个都要 `timeout 3600 bash -lc` 包裹整个 `... & wait` 块）：',
  '  波 1（约 34 分钟）：' + job(ARMS[0], 501) + ' & 与 seeds 502/503/504 同式；',
  '  波 2（约 34 分钟）：A0 seed 505 + t_leafq seed 501/502/503；',
  '  波 3（约 34 分钟）：t_leafq seed 504/505；',
  '备注：每个 job 已单独包 ' + LIMITER + ' gpu，多出的会自动排队；不要同时开超过 4 个会跑的 job 的波次。',
  '每波结束后（同一 bash 调用内）立刻 `wc -l` 各 games.jsonl 并检查对应 seed 的 kill_reasons/fallback；若某 seed 失败，记录并继续，最后如实报 partial。',
  '',
  '全部完成后：',
  '1) 每臂 `' + LIMITER + ' cpu ' + PY + ' runs/o4lite-search/combine.py --arm <arm> --in ' + OUT + '/<arm> --tb runs/o4lite-search/tb/<arm> --seeds 501,502,503,504,505`；',
  '2) 读 `runs/o4lite-search/paired_compare.py` 用法，做 **t_leafq − A0 逐 deal 配对**（两臂同 seeds=同 deal），给 d_win/d_margin/d_elo 与两类 CI：deal-cluster bootstrap（z）与 between-seed t(df=4)；',
  '3) 写 `' + OUT + '/confirm.json`：per-seed（两臂 winrate/elo）、paired、confirm_verdict —— 判据：**z 与 t CI 同时排除 0 且点估计 ≥ +10 Elo → confirmed**，否则 `not_confirmed`（写实得区间）；',
  '4) 附两臂各自 vs raw 的 Elo/winrate。',
  '只新增 ' + OUT + '/ 下文件；不改代码；不 commit。',
  '',
  RESOURCE_RULES,
].join('\n')

const run = await agent(RUN_PROMPT, {
  label: 'confirm-run', phase: 'Run', schema: RUN_SCHEMA, agentType: 'general-purpose',
  gate: PY + ' -c "import json;d=json.load(open(\'runs/o4lite-search/valuecal/confirm/confirm.json\'));assert d[\'status\'] in (\'done\',\'partial\')"',
})
if (!run || run.status === 'failed') {
  log('Run 失败，终止')
  return { preflight, run, stopped: true }
}
log('Run 完成：' + String(run.confirm_verdict).slice(0, 200))

// ---------------------------------------------------------------------------
// Phase 3 — Review
// ---------------------------------------------------------------------------
phase('Review')
const REVIEW_PROMPT = [
  '仓库 cwd=' + REPO + '。你是对抗审查子代理，独立审查一个 claim（每点只你 1 个审查者）。',
  '待审 claim [CONFIRM-t_leafq]：run 子代理报告的「fresh bank 500-504 上，t_leafq 相对 A0（当前代码 critic_B fixed-mean）的配对差异达到 confirm 门槛（z 与 t CI 同时排除 0 且点估计 ≥ +10 Elo）」是真实的。',
  '证据：`' + OUT + '/` 下两臂的 seed*.json/games.jsonl/log、`confirm.json`、`plan.json`、`runs/o4lite-search/paired_compare.py`、`src/seven523/duel.py`。',
  '',
  '三轴攻击：',
  '(a) 独立复现：从两臂 games.jsonl 自己重算 winrate/Elo 与逐 deal 配对（d_win/d_margin/d_elo、deal-bootstrap z-CI、between-seed t-CI、sign p），核对 confirm.json；核对 5 seeds 都齐、fallback=0、无 abort；',
  '(b) 替代解释：bank 500-504 是否真的 fresh（全仓库搜索，含 metrics/文件名/内容）；两臂是否同一份代码（mtime、aggregate 字段、逐位重放抽验，或直接确认两次运行间无代码改动）；actor 是否与 champion 逐位一致；deal schedule 是否完全同批；',
  '(c) 统计/适用域：5 seeds 的 between-seed 方差、bank 新鲜度之外的对手固定（raw champion）与 t=5/K=32 配置；结论只对该配置与对手成立；与 3-seed screen（+17.2 Elo）的对比是否一致。',
  'verdict 规则：confirmed/weakened/refuted/undecidable；不能独立复现不得投 confirmed。只读仓库与 /tmp/adv/CF/；禁止嵌套子代理。',
  '',
  RESOURCE_RULES,
].join('\n')

const review = await agent(REVIEW_PROMPT, { label: 'review-confirm', phase: 'Review', schema: VERDICT_SCHEMA, agentType: 'general-purpose' })

log('confirm 完成')
return { preflight, run, review }

export const meta = {
  name: 'rung-mle-p4a',
  description: 'P4a：把 t_leafq 搜索测成 MLE 绝对 rung（search vs 池子 h2h → 与 calibration games 联合 refit → 表/注册）（≤12 进程 / ≤4 GPU）',
  whenToUse: '当需要把已 confirm 的搜索对象登记进 probit-MLE 绝对表、不再拿 h2h 相对分相加时',
  phases: [
    { title: 'Audit', detail: '审计 calibration/refit/study/manifest 格式与 spec 支持，冻结对手/bank/命令' },
    { title: 'Collect', detail: 'search vs 选定池子对手的 h2h games（t=5/K=32/C=6）' },
    { title: 'Fit', detail: '联合 refit MLE，产出 search rung μ/σ/CI 与注册' },
    { title: 'Review', detail: 'rung 测量与注册的对抗审查（每点 1 个）' },
  ],
}

const REPO = '/home/amas/.local/src/7g523'
const LIMITER = REPO + '/.pi/limits/run_limited.sh'
const PY = REPO + '/.venv/bin/python'
const CHAMP = 'ckpt:runs/w2m_ctl__11__1790516900/agent.pt'
const SEARCH = 'rolloutt:runs/ei2_value_t5/t_leafq/critic.pt'
const OUT = 'runs/o4lite-search/p4a'
const CALIB = 'runs/w2m/calibration/games.jsonl'

const RESOURCE_RULES = [
  '【资源硬约束】机器 15GB / 32 核；全机同时 ≤12 个分析进程、≤4 个 GPU 进程。',
  '  ' + LIMITER + ' cpu <命令...>   # 纯标准库/文本/统计，最多 12 并发',
  '  ' + LIMITER + ' gpu <命令...>   # 任何 h2h / torch：最多 4 并发（多出的排队）',
  'sleep/ls 也用 cpu 包装；整脚本包一次执行；同一代理同一时刻只跑一个 gpu 进程。',
  '禁止 uv run；python 一律 ' + PY + '；禁止派生子代理（Agent/SubagentWorkflow）；不改 src/、不 commit。',
].join('\n')

const AUDIT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['status', 'opponents', 'bank', 'jobs', 'fit_commands', 'publish_plan', 'checks'],
  properties: {
    status: { type: 'string', enum: ['done', 'blocked'] },
    opponents: { type: 'array', items: { type: 'string' }, description: '5 个对手的 id=spec（跨强度）' },
    bank: { type: 'array', items: { type: 'number' } },
    jobs: { type: 'array', items: { type: 'string' }, description: '完整 h2h 命令行（含 arm/out/日志）' },
    fit_commands: { type: 'array', items: { type: 'string' } },
    publish_plan: { type: 'string', description: 'rung 注册方式：manifest 合并 vs 独立 artifact + 文档；理由与风险' },
    checks: { type: 'array', items: { type: 'string' } },
  },
}

const COLLECT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['status', 'jobs_done', 'games_files', 'per_job', 'kill_reasons'],
  properties: {
    status: { type: 'string', enum: ['done', 'partial', 'failed'] },
    jobs_done: { type: 'number' },
    games_files: { type: 'array', items: { type: 'string' } },
    per_job: { type: 'array', items: { type: 'object', properties: { arm: { type: 'string' }, seed: { type: 'number' }, winrate: { type: 'number' }, fallback: { type: 'number' } } } },
    kill_reasons: { type: 'array', items: { type: 'string' } },
  },
}

const FIT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['status', 'table', 'search_rung', 'comparison', 'published', 'checks'],
  properties: {
    status: { type: 'string', enum: ['done', 'partial', 'blocked'] },
    table: { type: 'string' },
    search_rung: { type: 'string', description: 'search_leafq 的 μ、Laplace σ、95% CI、n_games' },
    comparison: { type: 'string', description: '与 w2m_ctl / lvl4 / pool 其它 rung 的相对位置（同一 fit）' },
    published: { type: 'string', description: '注册结果：改了哪个文件 / 或独立 artifact + 文档说明' },
    checks: { type: 'array', items: { type: 'string' } },
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
// Phase 1 — Audit
// ---------------------------------------------------------------------------
phase('Audit')
log('P4a Audit：读 calibration/refit/study/placement，冻结对手与命令')

const AUDIT_PROMPT = [
  '仓库 cwd=' + REPO + '。你是 P4a 审计子代理（只读 + /tmp 脚本）。',
  '目的：把已 confirm 的搜索对象 `' + SEARCH + '`（t=5/K=32/C=6/rollout 对手 champion）测成 **probit-MLE 绝对 rung**，不再用 h2h 相对分相加。',
  '',
  '已存在的资产（先读再定方案）：',
  '- `' + CALIB + '`：w2m calibration 的 18,000 局（10 个 raw id，每条含 seed/seats/scores/subject/opponent/subject_seat/kind/rules_id）；',
  '- `runs/w2m/calibration/run_calibration.sh`（build_ladder 生成 + refit_mle 发布）；',
  '- `tools/refit_mle.py`（`--games` 可多次、`--anchors random=0`、`--bootstrap`、`--json --out`；`--manifest-out --refit` 的合并语义看 `seven523.study.merge_manifest`）；',
  '- `traces/pool10/manifest.json`（10 个 raw id + spec）与 `tools/play_ladder.py`；',
  '- `src/seven523/policies.py` 的 `validate_spec/split_entrant`、`src/seven523/placement/opponents.py` 的 spec 支持。',
  '',
  '任务（写 `' + OUT + '/plan.json`）：',
  '1) 确定 **5 个对手**（id=spec，覆盖强度：random 锚 + 低/中/高，例如 random、lvl1、lvl3、lvl4、ws_s2；spec 要能被 `run_h2h_trunc.py --right-spec` 使用），并给出每个对手的 manifest μ；',
  '2) 确定 bank（建议 520,521；做与 confirm 同款的 **fresh 校验**：全仓库 + traces 的 deal 级重叠；若被占用换 522+）；',
  '3) 给出完整 h2h 命令：`run_h2h_trunc.py --left-id search_leafq --left-spec ' + SEARCH + ' --right-spec <opp> --rollout-k 32 --rollout-max-candidates 6 --max-rollout-ply 400 --rollout-opponent ' + CHAMP + ' --device cpu --workers 1 --pairs 400`，env `O4_TRUNC_PLY=5 O4_AGG=mean`，arm 命名 `search_leafq_<oppid>`，out `' + OUT + '/<arm>`；每 job 800 局 ~16 分钟；5 对手 × 2 seeds = 10 job，按 ≤4 GPU 槽排 3 波；',
  '4) 给出 refit 命令：`refit_mle.py --games ' + CALIB + ' --games <每个 search games.jsonl> --anchors random=0 --bootstrap 4000 --bootstrap-workers 4 --seed 0 --json --out runs/w2m/calibration/table_with_search.json`；',
  '5) **publish_plan**：先确认 `study.merge_manifest` 能否把新 id 加进 `traces/pool10/manifest.json`（若 manifest 里没有 search 条目会怎样？`placement.load_opponents` 是否接受 `rolloutt:` spec？web/placement 现在能否真的用搜索 rung 对局？）。如果 placement 不支持 rolloutt，就**不要**污染 manifest：改为写独立 artifact `traces/pool10/search_rung.json` + 在 `docs/search-config-plan.md` 加一节，并给出未来接入需要的最小改动清单。',
  '纪律：所有命令走 ' + LIMITER + '（torch 才用 gpu）；不改仓库；不 commit。',
  '',
  RESOURCE_RULES,
].join('\n')

const audit = await agent(AUDIT_PROMPT, {
  label: 'p4a-audit', phase: 'Audit', agentType: 'general-purpose',
  gate: 'test -s runs/o4lite-search/p4a/plan.json',
})
if (!audit) {
  log('P4a Audit 失败，终止')
  return { audit, stopped: true }
}
log('P4a Audit 完成（plan.json 已就绪；该文件为自定义 schema，后续代理直接读它）')

// ---------------------------------------------------------------------------
// Phase 2 — Collect
// ---------------------------------------------------------------------------
phase('Collect')
const COLLECT_PROMPT = [
  '仓库 cwd=' + REPO + '。你是 P4a 采集子代理。',
  '按 `' + OUT + '/plan.json` 的 `jobs` 与 `waves` 执行（全部对手 × bank seeds，数量以文件为准）；每个 job 单独包 ' + LIMITER + ' gpu，按 ≤4 并发排波（每波一个 bash 调用，`timeout 3600 bash -lc "... & wait"`）。',
  '每波后检查 games.jsonl 行数（应为 800 = 400 deals × 2）与 seed json 的 fallback/kill_reasons；失败记录后继续。',
  '完成后写 `' + OUT + '/collect.json`：jobs_done、games_files、per_job（arm/seed/winrate/fallback）、kill_reasons。',
  '只新增 ' + OUT + '/ 下文件；不 commit。',
  '',
  RESOURCE_RULES,
].join('\n')

const collect = await agent(COLLECT_PROMPT, {
  label: 'p4a-collect', phase: 'Collect', agentType: 'general-purpose',
  gate: 'test -s runs/o4lite-search/p4a/collect.json',
})
if (!collect) {
  log('P4a Collect 失败，终止')
  return { audit, collect, stopped: true }
}
log('P4a Collect 完成（collect.json 已就绪）')

// ---------------------------------------------------------------------------
// Phase 3 — Fit & Publish
// ---------------------------------------------------------------------------
phase('Fit')
const FIT_PROMPT = [
  '仓库 cwd=' + REPO + '。你是 P4a 拟合/发布子代理。',
  '用 `' + OUT + '/collect.json` 与 `plan.json` 的 `fit` 字段做**联合 refit**：',
  '  `' + LIMITER + ' cpu ' + PY + ' tools/refit_mle.py --games ' + CALIB + ' --games <search games...> --anchors random=0 --bootstrap 4000 --bootstrap-workers 4 --seed 0 --json --out runs/w2m/calibration/table_with_search.json`',
  '（先确认 refit_mle 对 rules_id、id 连通性、prior 的要求；search_leafq 的 games 要全部传。）',
  '',
  '任务：',
  '1) 产出 `table_with_search.json`，读表：search_leafq 的 μ、Laplace σ、bootstrap 95% CI、n；与 w2m_ctl / lvl4 / ws_s2 等同一 fit 的相对位置；',
  '2) 按 plan 的 publish_plan 做注册（manifest 合并或独立 artifact + 文档）；如做文档更新，只加一节到 `docs/search-config-plan.md`，并写明：搜索 rung 的 μ 是**联合拟合**值、不是 +111 相加；',
  '3) 跑相关测试（若改了 `src/` 或 `tools/` 则有测试；只写 artifact 则跑 `' + LIMITER + ' cpu ' + PY + ' -m pytest tests -q -x -k \"mle or manifest or study or placement\"` 的快速子集）；',
  '4) 写 `' + OUT + '/fit.json`：table、search_rung、comparison、published、checks。',
  '不 commit；如需改 `tools/play_ladder.py`/manifest，务必先确认不会破坏现有测试与 placement 的 raw-only 契约。',
  '',
  RESOURCE_RULES,
].join('\n')

const fit = await agent(FIT_PROMPT, {
  label: 'p4a-fit', phase: 'Fit', agentType: 'general-purpose',
  gate: 'test -s runs/o4lite-search/p4a/fit.json',
})
if (!fit) {
  log('P4a Fit 失败，终止')
  return { audit, collect, fit, stopped: true }
}
log('P4a Fit 完成（fit.json 已就绪）')

// ---------------------------------------------------------------------------
// Phase 4 — Review
// ---------------------------------------------------------------------------
phase('Review')
const REVIEW_PROMPT = [
  '仓库 cwd=' + REPO + '。你是对抗审查子代理，独立审查一个 claim（每点只你 1 个审查者）。',
  '待审 claim [SEARCH-RUNG]：P4a 把 `' + SEARCH + '` 测成了 MLE 绝对 rung——其 games 合法（同 rules_id、格式正确、与 calibration 图连通）、联合 refit 正确、给出的 search_leafq μ/σ/CI 可信，且发布方式没有破坏 placement/rating 契约。',
  '证据：`' + OUT + '/plan.json|collect.json|fit.json`、`runs/w2m/calibration/table_with_search.json`、`runs/w2m/calibration/games.jsonl`、`' + OUT + '/search_leafq_*/seed*.games.jsonl`、`tools/refit_mle.py`、`src/seven523/mle.py`、`src/seven523/study.py`。',
  '',
  '三轴攻击：',
  '(a) 独立复现：检查 search games 的 rules_id 与 calibration 一致、每行格式（seats/scores/subject/opponent/subject_seat）；自己跑一次 refit（或核对输出表与 fit.json 数字）；',
  '(b) 替代解释：图连通性/先验（manifest 里没有 search id 时 μ 是否只是先验中心？n 是否>0？）；bank 复用与对手选择；h2h 的 left/right 方向有没有搞反；',
  '(c) 统计/适用域：bootstrap/n 能否支撑报告的 CI；search rung 的 μ 只对 revision-3/t=5/K=32/该对手集成立；与 h2h 相对分的差异说明。',
  'verdict 规则：confirmed/weakened/refuted/undecidable；不能独立复现不得投 confirmed。只读仓库与 /tmp/adv/P4A/；禁止嵌套子代理。',
  '',
  RESOURCE_RULES,
].join('\n')

const review = await agent(REVIEW_PROMPT, { label: 'p4a-review', phase: 'Review', schema: VERDICT_SCHEMA, agentType: 'general-purpose' })

log('P4a 完成')
return { audit, collect, fit, review }

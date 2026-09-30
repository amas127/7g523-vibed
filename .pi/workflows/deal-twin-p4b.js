export const meta = {
  name: 'deal-twin-p4b',
  description: 'P4b：实现人机 deal-twin 会话（同一副牌对 raw 与 search 各一局，配对 Δ + CI，web 可玩，--simulate 可验证）（≤12 进程 / ≤4 GPU）',
  whenToUse: '当需要首次用配对牌局测量“搜索对真人迁移了多少”并对真人可用时',
  phases: [
    { title: 'Design', detail: '读 placement/web/测试，定最小改造面与接口，保留 placement raw-only 契约' },
    { title: 'Implement', detail: '实现 twin 调度 + 搜索对手包装 + 报告工具 + 测试' },
    { title: 'Validate', detail: '--simulate 端到端验证 + pytest 子集' },
    { title: 'Review', detail: '实现正确性的对抗审查（每点 1 个）' },
  ],
}

const REPO = '/home/amas/.local/src/7g523'
const LIMITER = REPO + '/.pi/limits/run_limited.sh'
const PY = REPO + '/.venv/bin/python'
const SEARCH = 'rolloutt:runs/ei2_value_t5/t_leafq/critic.pt'

const RESOURCE_RULES = [
  '【资源硬约束】机器 15GB / 32 核；全机同时 ≤12 个分析进程、≤4 个 GPU 进程。',
  '  ' + LIMITER + ' cpu <命令...>   # 纯标准库/文本/测试，最多 12 并发',
  '  ' + LIMITER + ' gpu <命令...>   # 任何 torch / 对局：最多 4 并发',
  'sleep/ls 也用 cpu 包装；整脚本包一次执行；同一代理同一时刻只跑一个 gpu 进程。',
  '禁止 uv run；python 一律 ' + PY + '；禁止派生子代理（Agent/SubagentWorkflow）；不 commit。',
].join('\n')

const DESIGN_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['status', 'goal', 'files', 'interfaces', 'tests', 'simulate_path', 'purity', 'checks'],
  properties: {
    status: { type: 'string', enum: ['done', 'blocked'] },
    goal: { type: 'string' },
    files: { type: 'array', items: { type: 'object', properties: { path: { type: 'string' }, change: { type: 'string' } } } },
    interfaces: { type: 'string', description: '新 CLI/HTTP/API 形状与玩法流程' },
    tests: { type: 'array', items: { type: 'string' } },
    simulate_path: { type: 'string', description: '--simulate 怎么跑、验证哪些不变量' },
    purity: { type: 'string', description: '如何不破坏 placement raw-only 契约（tests/test_web.py:443 等）' },
    checks: { type: 'array', items: { type: 'string' } },
  },
}

const IMPL_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['status', 'files_changed', 'tests_run', 'tests_result', 'deviations'],
  properties: {
    status: { type: 'string', enum: ['done', 'partial', 'blocked'] },
    files_changed: { type: 'array', items: { type: 'string' } },
    tests_run: { type: 'array', items: { type: 'string' } },
    tests_result: { type: 'string' },
    deviations: { type: 'string', description: '与 Design 的偏离与理由' },
  },
}

const VALIDATE_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['status', 'simulate_runs', 'invariants', 'report_example', 'pytest', 'artifacts'],
  properties: {
    status: { type: 'string', enum: ['done', 'partial', 'failed'] },
    simulate_runs: { type: 'string', description: '跑过的 simulate 命令与结果（对手组合/局数）' },
    invariants: { type: 'string', description: '同 deal 配对、5/5 座位轮换、trace 可回放、fallback 等检查结果' },
    report_example: { type: 'string', description: '配对 Δ（human vs search − vs raw）与 CI 示例' },
    pytest: { type: 'string' },
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
// Phase 1 — Design
// ---------------------------------------------------------------------------
phase('Design')
log('P4b Design：定 deal-twin 最小改造面')

const DESIGN_PROMPT = [
  '仓库 cwd=' + REPO + '。你是 P4b 设计子代理（只读 + /tmp）。',
  '目标：让真人可以打 **deal-twin 会话**——每个 deal seed 打两局：人对 raw 臂、人对 search 臂（' + SEARCH + '，t=5/K=32/C=6），座位跨 deal 平衡；结束后给配对 Δ（human 对 search − 对 raw）与聚类 CI。',
  '',
  '必须读并理解的现有代码：',
  '- `src/seven523/placement/session.py`（PlacementSession 的 deal 计划 `plan_deals`、座位计划、对手选择、commit/report）；',
  '- `src/seven523/placement/opponents.py`（spec→policy；placement 的 raw-only 契约）；',
  '- `src/seven523/web/{server.py,table.py,cli.py,static/app.js}`（牌桌循环、自由对战 factory 注入、搜索包装从前端参数来的路径）；',
  '- `runs/o4lite-search/web_search.py`（它如何给 `7g523-web` 注入搜索 factory、`O4LiteWrapper` 构造、`--value-ckpt`）；',
  '- `tests/test_web.py`（尤其 `test_placement_ignores_the_free_play_factory`；不能破坏 placecement 的 raw-only 契约）；',
  '- `src/seven523/duel.py`（twin 的先例）与 `traces/sessions/*`（现有报告格式）。',
  '',
  '设计决策（必须给出明确选择与理由）：',
  '1) 新能力放哪里：`src/seven523/web` 的 twin 模式 + run-local 搜索 factory（推荐），还是独立 CLI 工具；',
  '2) 调度：每个 deal 的 (raw, search) 两局如何绑定（同 seed、座位交替、顺序随机或固定）；trace 如何标注 twin 组；',
  '3) 报告：配对 Δ 的口径（分差/胜负）、聚类 CI、样本量提示（≥30 对才有分辨力）；',
  '4) `--simulate`：脚本替真人跑完整 twin 会话以验证管线；',
  '5) placement 评级会话保持 raw-only（搜索 rung 不能进定级），twin 是新模式。',
  '写设计到 `runs/o4lite-search/p4b/design.json`；**不要**在本阶段改代码。',
  '',
  RESOURCE_RULES,
].join('\n')

const design = await agent(DESIGN_PROMPT, {
  label: 'p4b-design', phase: 'Design', agentType: 'general-purpose',
  gate: 'test -s runs/o4lite-search/p4b/design.json',
})
if (!design) {
  log('P4b Design 失败，终止')
  return { design, stopped: true }
}
log('P4b Design 完成')

// ---------------------------------------------------------------------------
// Phase 2 — Implement
// ---------------------------------------------------------------------------
phase('Implement')
const IMPL_PROMPT = [
  '仓库 cwd=' + REPO + '。你是 P4b 实现子代理。按 `' + REPO + '/runs/o4lite-search/p4b/design.json` 实现 deal-twin 会话（该文件是自定义 schema：看 interfaces.files / decisions / tests / acceptance_checks，不要依赖 status 字段）。',
  '约束：',
  '- 实现文件以 design.json 的 `interfaces.files` 为准；允许改 `src/seven523/web`、`src/seven523/placement`（只加新模式，不改 raw-only 契约）、`runs/o4lite-search/`、`tests/`；',
  '- 不 commit；不改 rules/引擎；搜索包装优先复用 `runs/o4lite-search/rollout_trunc.py`（已含单位修复）与现有 `O4LiteWrapper`；',
  '- 新增测试覆盖：twin 调度（同 deal 两局）、座位平衡、报告配对统计、placement raw-only 不被破坏；',
  '- 跑 `' + LIMITER + ' cpu ' + PY + ' -m pytest tests -q -x -k "web or placement or twin"`（以及新增测试文件）并修到全绿。',
  '写 `' + REPO + '/runs/o4lite-search/p4b/impl.json`：files_changed、tests_run、tests_result、deviations。',
  '',
  RESOURCE_RULES,
].join('\n')

const impl = await agent(IMPL_PROMPT, {
  label: 'p4b-impl', phase: 'Implement', agentType: 'general-purpose',
  gate: 'test -s runs/o4lite-search/p4b/impl.json',
})
if (!impl) {
  log('P4b Implement 失败/未过测试，终止')
  return { design, impl, stopped: true }
}
log('P4b Implement 完成：' + impl.tests_result)

// ---------------------------------------------------------------------------
// Phase 3 — Validate
// ---------------------------------------------------------------------------
phase('Validate')
const VALIDATE_PROMPT = [
  '仓库 cwd=' + REPO + '。你是 P4b 验证子代理。',
  '用 design.simulate_path 跑端到端 twin 会话（脚本当真人），至少两种配置：',
  '  a) 人对 raw + 人对 search（' + SEARCH + '，t=5/K=32/C=6）各 N 个 deal（N 取能让验证覆盖 twin 绑定的最小数，如 6-10；若 N=10 太慢可先用 K=8 的快速档但注明）；',
  '  b) 换座/顺序随机化配置各一次。',
  '核对不变量：同一 deal 的两局确实同 seed；5/5 或平衡座位；每局 trace 可被 `7g523-play --replay`（或等价）校验；search 局 fallback=0；报告里的配对 Δ 与 CI 用原始 trace 独立重算一致。',
  '把命令/结果写 `' + REPO + '/runs/o4lite-search/p4b/validate.json`（simulate_runs、invariants、report_example、pytest、artifacts）。torch/对局走 gpu 槽；测试走 cpu。',
  '',
  RESOURCE_RULES,
].join('\n')

const validate = await agent(VALIDATE_PROMPT, {
  label: 'p4b-validate', phase: 'Validate', agentType: 'general-purpose',
  gate: 'test -s runs/o4lite-search/p4b/validate.json',
})
if (!validate) {
  log('P4b Validate 失败，终止')
  return { design, impl, validate, stopped: true }
}
log('P4b Validate 完成')

// ---------------------------------------------------------------------------
// Phase 4 — Review
// ---------------------------------------------------------------------------
phase('Review')
const REVIEW_PROMPT = [
  '仓库 cwd=' + REPO + '。你是对抗审查子代理，独立审查一个 claim（每点只你 1 个审查者）。',
  '待审 claim [DEAL-TWIN]：P4b 实现的 deal-twin 会话在调度（同 deal 双局）、座位平衡、搜索对手包装、报告配对统计上正确，测试真实通过，placement raw-only 契约未被破坏。',
  '证据：`' + REPO + '/runs/o4lite-search/p4b/{design,impl,validate}.json`、新改的源码与测试、validate 的 trace/report 产物。',
  '',
  '三轴攻击：',
  '(a) 独立复现：读 diff（`git diff` + 新增文件），自己跑关键测试与 `--simulate`，用原始 trace 重算配对 Δ 与 CI；',
  '(b) 替代解释：twin 绑定是否真的同 seed（不是"看起来成对"）；座位是否平衡；搜索局是否真走 t_leafq+单位修复（而非 critic_B/旧码）；`random` 对手是否被误包搜索；',
  '(c) 统计/适用域：报告口径（分差 vs 胜负）、聚类单位、真人学习/carryover 的适用域声明；样本量提示是否诚实。',
  'verdict 规则：confirmed/weakened/refuted/undecidable；不能独立复现不得投 confirmed。只读仓库与 /tmp/adv/P4B/；禁止嵌套子代理。',
  '',
  RESOURCE_RULES,
].join('\n')

const review = await agent(REVIEW_PROMPT, { label: 'p4b-review', phase: 'Review', schema: VERDICT_SCHEMA, agentType: 'general-purpose' })

log('P4b 完成')
return { design, impl, validate, review }

export const meta = {
  name: 'critic-calib-p2',
  description: 'P2 critic 校准：审计 ei2_value_t5 已有资产 → 真人 479 决策 K=32 离线评测候选 critic → 入选者 same-bank h2h → 每点 1 个对抗审查（≤12 进程 / ≤4 GPU）',
  whenToUse: '当需要验证 t=5 截断 critic 的重训/校准能否降低真人局面乐观偏置并转化为 h2h 增益时',
  phases: [
    { title: 'Audit', detail: '审计 ei2_value_t5 已有结论与单位修复的影响，冻结候选列表' },
    { title: 'Offline', detail: '479 真人决策 K=32：候选 critic 的 bias/EV/blunder/案例与配对决策质量' },
    { title: 'Review', detail: '离线 claim 每点 1 个对抗审查' },
    { title: 'Screen', detail: '入选候选 same-bank 30/31/32 h2h vs A0 与 raw' },
    { title: 'Verify', detail: 'screen claim 每点 1 个对抗审查' },
  ],
}

const REPO = '/home/amas/.local/src/7g523'
const LIMITER = REPO + '/.pi/limits/run_limited.sh'
const PY = REPO + '/.venv/bin/python'
const CHAMP = 'ckpt:runs/w2m_ctl__11__1790516900/agent.pt'
const OUT = 'runs/o4lite-search/valuecal'
const A0_ARM = 'unitfix'

const RESOURCE_RULES = [
  '【资源硬约束】机器 15GB / 32 核；全机同时 ≤12 个分析进程、≤4 个 GPU 进程。',
  '  ' + LIMITER + ' cpu <命令...>   # 纯标准库/文本/编译/轻量分析，最多 12 并发',
  '  ' + LIMITER + ' gpu <命令...>   # 任何 import torch / 加载 .pt / 重放：最多 4 并发',
  'sleep/ls 也用 cpu 包装；整脚本包一次执行；同一代理同一时刻只跑一个 gpu 进程。',
  '禁止 uv run；python 一律 ' + PY + '；禁止派生子代理（Agent/SubagentWorkflow）；不改 src/、不 commit。',
].join('\n')

const AUDIT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['status', 'candidates', 'established_facts', 'invalidated_by_unit_fix', 'checks'],
  properties: {
    status: { type: 'string', enum: ['done', 'blocked'] },
    candidates: { type: 'array', items: { type: 'string' }, description: '冻结的候选 critic ckpt 路径 + 选择理由' },
    established_facts: { type: 'array', items: { type: 'string' }, description: '已有报告里仍成立的事实（含数字）' },
    invalidated_by_unit_fix: { type: 'array', items: { type: 'string' } },
    checks: { type: 'array', items: { type: 'string' } },
  },
}

const OFFLINE_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['status', 'script', 'candidates', 'human_metrics', 'decision_quality_vs_old', 'blunders', 'case_s3548246500', 'selected_for_screen', 'checks'],
  properties: {
    status: { type: 'string', enum: ['done', 'partial', 'blocked'] },
    script: { type: 'string' },
    candidates: { type: 'array', items: { type: 'string' } },
    human_metrics: { type: 'string', description: '各候选在 479 决策上的截断 bias/EV/corr/MAE（配对 vs 全量真值）' },
    decision_quality_vs_old: { type: 'string', description: '各候选 vs old critic_B 的配对决策质量（one-step champion 续行 truth，按 22 局聚类 CI）' },
    blunders: { type: 'string' },
    case_s3548246500: { type: 'string' },
    selected_for_screen: { type: 'string', description: '按预注册规则入选的候选；无则写 kill' },
    checks: { type: 'array', items: { type: 'string' } },
  },
}

const SCREEN_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['status', 'seeds', 'fallback_total', 'per_seed', 'vs_a0', 'vs_raw', 'kill_reasons', 'artifacts'],
  properties: {
    status: { type: 'string', enum: ['done', 'partial', 'failed', 'blocked'] },
    seeds: { type: 'array', items: { type: 'number' } },
    fallback_total: { type: 'number' },
    per_seed: { type: 'array', items: { type: 'object', properties: { arm: { type: 'string' }, seed: { type: 'number' }, winrate: { type: 'number' }, elo: { type: 'number' } } } },
    vs_a0: { type: 'string' },
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
// Phase 1 — Audit
// ---------------------------------------------------------------------------
phase('Audit')
log('Phase Audit：审计 ei2_value_t5 既有资产，冻结候选')

const AUDIT_PROMPT = [
  '仓库 cwd=' + REPO + '。你是审计子代理。只读，不改任何文件（可写 /tmp/audit/ 脚本）。',
  '',
  '背景：`runs/ei2_value_t5/` 是上午（05:43-07:14）跑的 critic 重训实验，全部在 12:10 的单位修复**之前**。',
  '已知（主控已读，需你核实）：预注册 `preregistration.md`；`status.txt`/`eval_status.txt`；',
  '  - `retr_vs_old/combined.json`：重训 critic vs 旧 critic_B，K=8，seeds 410-414，ΔElo +0.08 [-11.4,+11.6]（平局）；',
  '  - `retr_vs_raw` +84.1、`old_vs_raw` +86.7（均 K=8、旧代码）；',
  '  - `calib/report.json`：P2A 通过；P2B 根分布 EV old 0.844 > searchq 0.829；P2C 叶子分布 EV old 0.425 < searchq 0.467 / searchq_all 0.479（配对 ΔMSE 95% CI 完全为负，通过）；',
  '  - 候选 ckpt：`runs/ei/value/critic_B.pt`（old）、`runs/ei2_value_t5/t_searchq/critic.pt`、`t_searchq_all/critic.pt`、`t_rawq/critic.pt`、`t_leafq/critic.pt`（训练目标分别 search_q/search_q all/raw_q/leaf_mc）。',
  '任务：产出 `' + OUT + '/audit.json`：',
  '1) `established_facts`：逐条列出仍成立的事实与数字（file:line 或 json 字段），不要复述我的摘要，以文件为准；',
  '2) `invalidated_by_unit_fix`：单位修复（`runs/o4lite-search/rollout_trunc.py` 截断读出 ×`rules.total_points`）之后，哪些结论/资产失效或需换算（例如 `t_truncq` 的 q_trunc 目标、`trunc_leaves.py` 的 contract、旧 h2h 的绝对 Elo），哪些不受影响（leaf_mc 标签、critic ckpt 权重本身、P2A/P2C 的 EV 口径）；',
  '3) `candidates`：冻结给下一步离线评测的候选 critic 列表（老/重训各一个 spec `rolloutt:<ckpt>`），并说明每个候选在叶子分布 vs 根分布的表现；',
  '4) 核对 `probe/report.json`：searchq 相对 old 的 raw_agreement/changed_agreement（旧 K=8 决策）作为行为差异的参考。',
  '',
  '纪律：所有命令走 ' + LIMITER + ' cpu（如需 torch 才用 gpu）；≥3 个候选时不要并行；只读。',
  '',
  RESOURCE_RULES,
].join('\n')

const audit = await agent(AUDIT_PROMPT, { label: 'audit-ei2', phase: 'Audit', schema: AUDIT_SCHEMA, agentType: 'general-purpose' })
if (!audit || audit.status !== 'done') {
  log('Audit 失败，终止')
  return { audit, stopped: true }
}
log('Audit 完成，候选：' + audit.candidates.join(', '))

// ---------------------------------------------------------------------------
// Phase 2 — Offline（真人 479 决策，K=32，单位修复后）
// ---------------------------------------------------------------------------
phase('Offline')
log('Phase Offline：479 真人决策上评测候选 critic 的偏置与决策质量')

const OFFLINE_PROMPT = [
  '仓库 cwd=' + REPO + '。你是离线评测子代理。目标：在**真人对战轨迹**的 479 个 bot 决策上（t=5/K=32/C=6/rollout 对手=champion，单位已修复），量化候选 critic 相对 old critic_B 是否降低乐观偏置、改善决策质量。',
  '候选列表与审计事实：`' + OUT + '/audit.json`。',
  '',
  '可复用工具（已存在，先读再用）：',
  '- `runs/o4lite-search/unitfix/probe_human.py`（重放 22 局、479 决策、EV/corr/MAE、case，spec 用 `rolloutt:<ckpt>` 即可切换 critic）；',
  '- `runs/o4lite-search/riskreadout/calib_human.py` 与 `world_cache.json`（P1 的 per-candidate per-world 缓存/配对口径）。',
  '',
  '任务（脚本 `' + OUT + '/offline_human.py`，结果 `' + OUT + '/offline.json`）：',
  '1) 对每个候选（old + 审计冻结的重训候选）：重放 479 决策，报告 fallback（必须 0）、changed_vs_recorded、changed_vs_old；',
  '2) **偏置/校准**：对 searched 决策的选中候选，比较截断读出 mean_K 与「同一 K=32 隐藏世界的 champion 全量续行」代理真值：mean bias（分）、p90|bias|、|bias|>2se 占比、EV/corr/MAE —— 与 old 的 +13.68 分基线配对比较（按 22 局聚类 bootstrap 95% CI）；',
  '3) **决策质量**：每个候选的选中动作 vs old 选中动作，用 one-step champion 续行真值做逐决策配对差（按局聚类 CI），报 better/worse/tie；',
  '4) blunder（存在全终局保底候选且其精确 margin 严格更大）次数与案例；`g0000__s3548246500` step37 各候选选择与真值；',
  '5) 预注册选臂规则（写死）：入选 screen 的候选必须同时满足——相对 old 的决策质量配对 CI 下界 ≥ 0 且点估计 ≥ +2 分/决策，且 mean bias 相对 old 的点估计 ≤ -2 分（方向正确）；没有候选满足就写 `kill`（P2 关闭，转 P3/加深截断）。',
  '',
  '纪律：torch 走 gpu 槽、单进程合并算完；总计算 ≤25 分钟；不改仓库既有文件（只新增 ' + OUT + '/）；不 commit。',
  '',
  RESOURCE_RULES,
].join('\n')

const offline = await agent(OFFLINE_PROMPT, {
  label: 'offline-human',
  phase: 'Offline',
  schema: OFFLINE_SCHEMA,
  agentType: 'general-purpose',
  gate: PY + ' -c "import json;d=json.load(open(\'runs/o4lite-search/valuecal/offline.json\'));assert d[\'status\'] in (\'done\',\'partial\')"',
})
if (!offline || offline.status === 'blocked') {
  log('Offline 失败，终止')
  return { audit, offline, stopped: true }
}
log('Offline 完成；selected=' + offline.selected_for_screen)

const sel = (offline.selected_for_screen || '').toLowerCase()
if (sel.includes('kill')) {
  log('没有候选 critic 在真人局面上达标 → P2 记负，不启动 screen')
  return { audit, offline, stopped: true, reason: 'no critic candidate beats old on human-state bias/quality' }
}

// ---------------------------------------------------------------------------
// Phase 3 — Review（离线 claim，1 个审查者）
// ---------------------------------------------------------------------------
phase('Review')
const REVIEW_OFFLINE_PROMPT = [
  '仓库 cwd=' + REPO + '。你是对抗审查子代理，独立审查一个 claim（每点只你 1 个审查者）。',
  '待审 claim [CRITIC-OFFLINE]：在 479 个真人决策上，入选候选 critic 相对 old critic_B 显著降低了截断读出的乐观偏置（+13.68 基线）并改善了决策质量。',
  '证据：`' + OUT + '/offline.json`、`offline_human.py`、`' + OUT + '/audit.json`、原始轨迹 `traces/web/run-20260928-091716/`。',
  '',
  '三轴攻击：(a) 独立复现关键数字（bias、配对决策质量、CI、blunder、case），核对 479/fallback=0 与 spec 是否真的切了 critic；',
  '(b) 替代解释：不同的候选 ckpt 会不会只是策略行为不同而非价值校准更好（probe raw_agreement 0.957）？校准评测的“真值”是 champion 全量续行（模型内反事实）；champion 自身偏置是否污染 truth；',
  '(c) 统计/适用域：22 局聚类、多重比较（候选数 × 指标）、OOD 只有单日单人；结论能否支撑“部署更优”。',
  'verdict 规则：confirmed/weakened/refuted/undecidable；不能独立复现不得投 confirmed。只读仓库与 /tmp/adv/P2O/；禁止嵌套子代理。',
  '',
  RESOURCE_RULES,
].join('\n')

const reviewOffline = await agent(REVIEW_OFFLINE_PROMPT, { label: 'review-offline', phase: 'Review', schema: VERDICT_SCHEMA, agentType: 'general-purpose' })

// ---------------------------------------------------------------------------
// Phase 4 — Screen
// ---------------------------------------------------------------------------
phase('Screen')
log('Phase Screen：入选 critic same-bank 30/31/32 h2h vs A0')

const SCREEN_PROMPT = [
  '仓库 cwd=' + REPO + '。你是 screen 子代理：跑入选 critic 的 t=5/K=32 same-bank h2h。',
  '入选信息：`' + OUT + '/offline.json` 的 selected_for_screen（最多 2 个候选 critic ckpt）。',
  '配置：left=`rolloutt:<ckpt>`；right=`' + CHAMP + '`；seeds 30,31,32；pairs=400；K=32/C=6/--max-rollout-ply 400/--rollout-opponent ' + CHAMP + '；runner `runs/speed/run_h2h_trunc.py`（与 unitfix screen 同 runner）；O4_TRUNC_PLY=5；默认 O4_AGG=mean。',
  '基线 A0：`runs/o4lite-search/' + A0_ARM + '/seed{30,31,32}.games.jsonl`（fixed mean，K=32，同 bank）。',
  '',
  '执行（一个 bash 调用；≤2 臂 × 3 seeds，按 ≤4 GPU 槽排队，整段 `timeout 5400`）：每 (arm,seed) 单独进程各自包 ' + LIMITER + ' gpu，日志 ' + OUT + '/<arm>.seed<s>.log；全部结束后逐臂 combine（读 combine.py 用法），若需要每臂独立目录就照做。',
  '报告（`' + OUT + '/screen.json`）：每 seed winrate/elo、fallback 总数、kill_reasons；vs_a0 逐 deal 配对（d_win/d_margin/d_elo + CI）；vs_raw Elo。',
  'Green 判据（预注册）：vs_a0 的配对 d_margin CI 下界 ≥ 0 且 winrate 不降；任一臂 CI 完全 <0 → 该臂 kill。timeout 就 pkill 并如实报 partial。',
  '',
  '不许改 rollout 代码；只新增 ' + OUT + '/ 下文件；不 commit。',
  '',
  RESOURCE_RULES,
].join('\n')

const screen = await agent(SCREEN_PROMPT, {
  label: 'screen-critic',
  phase: 'Screen',
  schema: SCREEN_SCHEMA,
  agentType: 'general-purpose',
  gate: PY + ' -c "import json;d=json.load(open(\'runs/o4lite-search/valuecal/screen.json\'));assert d[\'status\']==\'done\';assert d[\'fallback_total\']==0"',
})

// ---------------------------------------------------------------------------
// Phase 5 — Verify（screen claim，1 个审查者）
// ---------------------------------------------------------------------------
phase('Verify')
const REVIEW_SCREEN_PROMPT = [
  '仓库 cwd=' + REPO + '。你是对抗审查子代理，独立审查一个 claim（每点只你 1 个审查者）。',
  '待审 claim [CRITIC-SCREEN]：screen 报告的「入选 critic vs A0（unitfix fixed-mean）在 bank 30-32 上的 h2h 差异」是真实的，不是 runner/版本/deal 错配。',
  '证据：`' + OUT + '/screen.json` 与各 seed 产物；`runs/o4lite-search/' + A0_ARM + '/seed{30,31,32}.games.jsonl`。',
  '',
  '三轴攻击：(a) 从两边 games.jsonl 独立重算 winrate/Elo 与逐 deal 配对，核对 deal 同批；',
  '(b) 候选 ckpt 的 policy 是否与 champion 逐位相同（E4 负对照口径）——若 policy 不同，差异不能归因于 critic；核对 `policy_control/result.json` 的旧结论并用新进程抽验；',
  '(c) 3 seeds 的分辨力、bank 复用史、旧代码 vs 新代码的版本差（A0 产物 12:41 由 13:18 前代码生成）。',
  'verdict 规则同上。只读仓库与 /tmp/adv/P2S/；禁止嵌套子代理。',
  '',
  RESOURCE_RULES,
].join('\n')

const reviewScreen = await agent(REVIEW_SCREEN_PROMPT, { label: 'review-screen', phase: 'Verify', schema: VERDICT_SCHEMA, agentType: 'general-purpose' })

log('P2 完成')
return { audit, offline, reviewOffline, screen, reviewScreen }

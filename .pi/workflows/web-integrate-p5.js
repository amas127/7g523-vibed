export const meta = {
  name: 'web-integrate-p5',
  description: 'P5：把 deal-twin 固定进 stock 7g523-web；自由对战 + 定级模式都可调 rollout/K；单入口 uv run 7g523-web（≤12 进程 / ≤4 GPU）',
  whenToUse: '当需要把 run-local 搜索/twin 能力整合进标准 web 入口并开放 t/K 配置时',
  phases: [
    { title: 'Design', detail: '读 web/placement/policies/tests，定插件缝、t/K 开关与定级语义，输出文件级方案' },
    { title: 'Implement', detail: '实现单入口 + 自由对战/定级 t-K 配置 + twin 固定 + 测试全绿' },
    { title: 'Validate', detail: '用 .venv/bin/7g523-web 端到端验证三种模式与配置生效' },
    { title: 'Review', detail: '实现与契约的对抗审查（每点 1 个）' },
  ],
}

const REPO = '/home/amas/.local/src/7g523'
const LIMITER = REPO + '/.pi/limits/run_limited.sh'
const PY = REPO + '/.venv/bin/python'
const WEB = REPO + '/.venv/bin/7g523-web'
const OUT = 'runs/o4lite-search/p5'

const RESOURCE_RULES = [
  '【资源硬约束】机器 15GB / 32 核；全机同时 ≤12 个分析进程、≤4 个 GPU 进程。',
  '  ' + LIMITER + ' cpu <命令...>   # 纯标准库/文本/测试，最多 12 并发',
  '  ' + LIMITER + ' gpu <命令...>   # 任何 torch / 对局：最多 4 并发',
  'sleep/ls 也用 cpu 包装；整脚本包一次执行；同一代理同一时刻只跑一个 gpu 进程。',
  '禁止 uv run（用 ' + WEB + ' 或 ' + PY + '，避免 sync 风暴）；禁止派生子代理；不 commit。',
].join('\n')

const DESIGN_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['status', 'entry', 'free_play', 'twin', 'placement', 'files', 'tests', 'risks'],
  properties: {
    status: { type: 'string', enum: ['done', 'blocked'] },
    entry: { type: 'string', description: 'uv run 7g523-web 如何获得搜索/twin 能力（包边界方案）' },
    free_play: { type: 'string', description: '自由对战可调 rollout/K 的 UI/API 形态与边界' },
    twin: { type: 'string', description: 'twin 如何固定进 stock 入口（pinned 身份、pairs 默认）' },
    placement: { type: 'string', description: '定级模式如何支持搜索/t-K：rated 语义（measured rung）与 experimental/unrated 路径，及测试契约改动' },
    files: { type: 'array', items: { type: 'object', properties: { path: { type: 'string' }, change: { type: 'string' } } } },
    tests: { type: 'array', items: { type: 'string' } },
    risks: { type: 'string' },
  },
}

const IMPL_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['status', 'files_changed', 'tests_result', 'deviations'],
  properties: {
    status: { type: 'string', enum: ['done', 'partial', 'blocked'] },
    files_changed: { type: 'array', items: { type: 'string' } },
    tests_result: { type: 'string' },
    deviations: { type: 'string' },
  },
}

const VALIDATE_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['status', 'free_play', 'twin', 'placement', 'pytest', 'artifacts'],
  properties: {
    status: { type: 'string', enum: ['done', 'partial', 'failed'] },
    free_play: { type: 'string', description: 't/K 是否真的进入引擎（t0 vs t5、K8 vs K32 决策/响应不同）' },
    twin: { type: 'string', description: 'stock 入口下 twin 固定身份与 pairs 默认验证' },
    placement: { type: 'string', description: 'rated vs search_leafq(μ=260) 与 experimental/unrated 两条路径验证' },
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
log('P5 Design：单入口 + t/K 配置 + twin 固定的最小改造面')

const DESIGN_PROMPT = [
  '仓库 cwd=' + REPO + '。你是 P5 设计子代理（只读 + /tmp + 写 ' + OUT + '/design.json）。',
  '',
  '用户目标：',
  '1) **把 deal-twin 固定下来**（作为 stock web 的主模式之一，pinned 身份：raw=ckpt:t_leafq/critic.pt，search=rolloutt:t_leafq/critic.pt t5/K32/C6；pairs 默认 30，可在会话开始选偶数）；',
  '2) **自由对战**里可以自由修改 rollout(t) 与 K；',
  '3) **定级模式**里也可以修改 rollout/K；',
  '4) 以上全部通过 **`uv run 7g523-web`**（即 `pyproject.toml` 的 `seven523.web:main`）可用。',
  '',
  '现状（已核实，需你复核）：',
  '- `src/seven523/web/cli.py:main(policy_factory=None, search=None, twin=None)` 已有注入缝；`pyproject.toml:19` 的 `7g523-web` 走 stock main，三者全 None；搜索包装与 twin 元数据目前只由 `runs/o4lite-search/web_search.py` 与 `web_twin.py` 注入；',
  '- 自由对战的 per-game t/K UI 已存在（`src/seven523/web/static/app.js` 的 search-k/trunc 预设），但只在 web_search 启动器下出现，且受 launch 默认值约束；',
  '- placement 是 **raw-only**：`tests/test_web.py` 的 `test_placement_ignores_the_free_play_factory` 锁死；搜索 rung 未进 manifest（P4a 的 `artifacts/search-rung/search_rung.json` 给出 μ=260.03 [251.22,269.89]，spec `rolloutt:runs/ei2_value_t5/t_leafq/critic.pt`）；',
  '- `src/seven523/policies.py` 的 spec 语法只有 `random`/`ckpt:`；placement 的 `load_opponents` 会拒绝无 spec 的 id。',
  '',
  '设计必须明确回答（写进 design.json）：',
  'A) **单入口方案**：包不能硬 import `runs/`（实验算子边界）。推荐「插件发现」：stock main 按 env（如 `SEVEN523_WEB_PLUGIN`）或 cwd 下的 `runs/o4lite-search/web_plugin.py` 可选加载，导出 `SearchFactory`/search 元数据/twin 元数据；找不到就退化为 raw-only。给出精确接口、错误处理（无 torch/无文件时的行为）与为什么安全。',
  'B) **自由对战 t/K**：UI/API 形态（per-game 还是 per-session）、t 与 K 的取值边界（现有硬界 TRUNC_MIN/MAX=0..40、K=1..64）、默认值来源；保证 t=0=全量、t≥ply 等价。',
  'C) **twin 固定**：stock 入口下 twin 模式的 pin（身份、默认 pairs、座位/顺序交替），并明确它不产生评分。',
  'D) **定级模式**：给出两条明确路径并实现取舍——(i) **rated**：只能对「有已测 μ 的搜索 rung」定级（当前唯一是 search_leafq, μ=260, t5/K32/C6；配置必须在会话级固定，不能逐局改，否则评分口径失效），需要 spec 语法/工厂缝 + manifest/独立 rung 加载 + `load_opponents` 能力门 + placement UI 选择；(ii) **experimental/unrated**：允许任意 t/K，但报告明确不产出评分/档位，也不写 manifest。两条路径的 UI 标识、报告措辞、以及 `test_placement_ignores_the_free_play_factory` 契约如何**有意**更新（不是删保护，而是把「rated 只用 measured rung」写成新契约 + 新增 unrated 守卫测试）。',
  'E) 兼容性：`web_search.py` / `web_twin.py` / `7g523-elo` CLI 仍可用；ADR-0013（跨 rules/fit 不可比）与 placement 评分身份不被破坏。',
  '',
  '输出文件级方案（files 含每处改动）、测试清单（新增 + 修改）、风险。不要在本阶段改代码。',
  '',
  RESOURCE_RULES,
].join('\n')

const design = await agent(DESIGN_PROMPT, {
  label: 'p5-design', phase: 'Design', agentType: 'general-purpose',
  gate: 'test -s runs/o4lite-search/p5/design.json',
})
if (!design) {
  log('P5 Design 失败，终止')
  return { design, stopped: true }
}
log('P5 Design 完成')

// ---------------------------------------------------------------------------
// Phase 2 — Implement
// ---------------------------------------------------------------------------
phase('Implement')
const IMPL_PROMPT = [
  '仓库 cwd=' + REPO + '。你是 P5 实现子代理。按 `' + OUT + '/design.json` 实现。',
  '范围（允许改）：`src/seven523/web/**`、`src/seven523/placement/**`、`src/seven523/policies.py`（如需 spec 缝，保持向后兼容）、`runs/o4lite-search/web_*.py`、`tests/**`、`docs/human-play.md`、`pyproject.toml`（如需要）。',
  '硬约束：',
  '- 不 commit；不破坏 rules/引擎；不把 run-local 搜索代码复制进 `src/`（插件加载/工厂注入优先）；',
  '- rated placement 只允许「有已测 μ 的搜索配置」，任意 t/K 只能走 experimental/unrated 且报告明示；',
  '- `tests/test_web.py` 的 raw-only 守卫要**有意更新**为新契约（rated=measured rung；unrated=显式），并新增对应测试；',
  '- 新增/更新测试覆盖：单入口插件加载与退化、自由对战 t/K 生效、twin 固定、placement rated/unrated 两路径与报告措辞；',
  '- 跑 `' + LIMITER + ' cpu ' + PY + ' -m pytest -q` 全量并修到全绿。',
  '写 `' + OUT + '/impl.json`：files_changed、tests_result、deviations。',
  '',
  RESOURCE_RULES,
].join('\n')

const impl = await agent(IMPL_PROMPT, {
  label: 'p5-impl', phase: 'Implement', agentType: 'general-purpose',
  gate: 'test -s runs/o4lite-search/p5/impl.json',
})
if (!impl) {
  log('P5 Implement 失败，终止')
  return { design, impl, stopped: true }
}
log('P5 Implement 完成')

// ---------------------------------------------------------------------------
// Phase 3 — Validate
// ---------------------------------------------------------------------------
phase('Validate')
const VALIDATE_PROMPT = [
  '仓库 cwd=' + REPO + '。你是 P5 验证子代理。用 **stock 入口** ' + WEB + '（等价 `uv run 7g523-web`）端到端验证三件事，写 `' + OUT + '/validate.json`：',
  '1) **自由对战 t/K 生效**：起服务或在 `--simulate`/无头模式（按 design 的实现）分别用 t0/K8（全量）与 t5/K32 跑同一 seed，确认 search 臂决策/响应确实不同（或等价地：API 接受并回显配置、引擎参数被消费）；',
  '2) **twin 固定**：stock 入口下 twin 会话可用，pin 的 raw/search spec、pairs 默认与座位/顺序交替正确，产物落 `traces/twins/`；',
  '3) **placement 两路径**：rated vs search_leafq（μ=260）能建会话/评分（用 `--simulate` 脚本当真人最省事），experimental 任意 t/K 会被标 unrated 且报告不产出 μ；',
  '4) `' + LIMITER + ' cpu ' + PY + ' -m pytest -q` 全量结果。',
  '核对不变量：无 torch 环境下 stock 入口退化为 raw-only；trace JSON 可 replay；fallback=0；配置真的 pin（不是显示但没生效）。',
  'torch/对局走 gpu 槽；测试走 cpu。只新增 ' + OUT + '/ 下文件；不 commit。',
  '',
  RESOURCE_RULES,
].join('\n')

const validate = await agent(VALIDATE_PROMPT, {
  label: 'p5-validate', phase: 'Validate', agentType: 'general-purpose',
  gate: 'test -s runs/o4lite-search/p5/validate.json',
})
if (!validate) {
  log('P5 Validate 失败，终止')
  return { design, impl, validate, stopped: true }
}
log('P5 Validate 完成')

// ---------------------------------------------------------------------------
// Phase 4 — Review
// ---------------------------------------------------------------------------
phase('Review')
const REVIEW_PROMPT = [
  '仓库 cwd=' + REPO + '。你是对抗审查子代理，独立审查一个 claim（每点只你 1 个审查者）。',
  '待审 claim [WEB-INTEGRATION]：`uv run 7g523-web`（stock 入口）现在真的支持（a）自由对战可调 rollout/K、（b）固定 deal-twin、（c）定级模式 rated vs measured 搜索 rung 与 experimental/unrated 两路径，且实现没有破坏 placement 评分身份/包边界/既有测试。',
  '证据：`' + OUT + '/{design,impl,validate}.json`、改动源码与测试、validate 的产物与日志。',
  '',
  '三轴攻击：',
  '(a) 独立复现：自己跑 stock 入口的关键路径（不只是读 validate 的自述）、跑测试、用 trace/API 响应核对 t/K 真的进入引擎；',
  '(b) 替代解释：配置可能只是 UI 显示没生效；插件加载可能把 run-local 的旧 critic_B 路径带进来；twin 的 pin 是否与 P4b/P4a 身份一致；placement rated 的 μ/σ 来源与 rules_id/尺 口径；',
  '(c) 契约/统计：raw-only 守卫的更新是否把保护改没了（rated 是否真的只用 measured rung）；unrated 报告是否明示不产出评分；ADR-0013 与测试是否仍锁住跨口径混用。',
  'verdict 规则：confirmed/weakened/refuted/undecidable；不能独立复现不得投 confirmed。只读仓库与 /tmp/adv/P5/；禁止嵌套子代理。',
  '',
  RESOURCE_RULES,
].join('\n')

const review = await agent(REVIEW_PROMPT, { label: 'p5-review', phase: 'Review', schema: VERDICT_SCHEMA, agentType: 'general-purpose' })

log('P5 完成')
return { design, impl, validate, review }

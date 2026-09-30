export const meta = {
  name: 'risk-readout-p1',
  description: 'P1 风险敏感读出：实现聚合臂（mean/lcb/cvar/guard）→ 真人轨迹上校验决策是否优于旧规则 → same-bank h2h screen → 每点 1 个对抗审查（≤12 进程 / ≤4 GPU）',
  whenToUse: '当需要修复搜索截断读出的乐观/赢者诅咒问题，并在真人轨迹上验证新决策规则优于旧推理规则时',
  phases: [
    { title: 'Implement', detail: '给 batched_decide 加 O4_AGG 聚合臂（默认 mean 不变）' },
    { title: 'Calibrate', detail: '479 真人决策：偏差/方差分解 + 各臂 vs 旧规则/固定 mean 的决策质量与 blunder' },
    { title: 'Screen', detail: '入选臂 same-bank 30/31/32 × 400 deals h2h，配对对比 fixed-mean 与 raw' },
    { title: 'Verify', detail: '真人轨迹 claim 与 screen claim 各 1 个对抗审查' },
  ],
}

const REPO = '/home/amas/.local/src/7g523'
const LIMITER = REPO + '/.pi/limits/run_limited.sh'
const PY = REPO + '/.venv/bin/python'
const CHAMP = 'ckpt:runs/w2m_ctl__11__1790516900/agent.pt'
const CRITIC = 'rolloutt:runs/ei/value/critic_B.pt'
const OUT = 'runs/o4lite-search/riskreadout'
const A0_ARM = 'unitfix'

const RESOURCE_RULES = [
  '【资源硬约束】机器 15GB / 32 核；全机同时 ≤12 个分析进程、≤4 个 GPU 进程。',
  '  ' + LIMITER + ' cpu <命令...>   # 纯标准库/文本/编译，最多 12 并发',
  '  ' + LIMITER + ' gpu <命令...>   # 任何 import torch / 加载 .pt / 重放：最多 4 并发',
  'sleep/ls 也用 cpu 包装；整脚本包一次执行；同一代理同一时刻只跑一个 gpu 进程。',
  '禁止 uv run；python 一律 ' + PY + '；禁止派生子代理（Agent/SubagentWorkflow）；不改 src/、不 commit。',
].join('\n')

const IMPL_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['status', 'files', 'modes', 'env_contract', 'checks', 'risks'],
  properties: {
    status: { type: 'string', enum: ['done', 'blocked'] },
    files: { type: 'array', items: { type: 'string' } },
    modes: { type: 'string', description: '四种聚合的公式与默认值' },
    env_contract: { type: 'string', description: 'O4_AGG / O4_LCB_Z / O4_CVAR_ALPHA 语义与默认' },
    checks: { type: 'array', items: { type: 'string' } },
    risks: { type: 'array', items: { type: 'string' } },
  },
}

const CALIB_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['status', 'script', 'arms', 'bias_variance', 'human_vs_old', 'human_vs_a0', 'blunders', 'case_s3548246500', 'selected_for_screen', 'checks'],
  properties: {
    status: { type: 'string', enum: ['done', 'partial', 'blocked'] },
    script: { type: 'string' },
    arms: { type: 'array', items: { type: 'string' }, description: '实际评估的臂与参数' },
    bias_variance: { type: 'string', description: 'mean_K vs 全量真值的 bias / across-world sd / se 分解' },
    human_vs_old: { type: 'string', description: '真人 479 决策：各臂 vs 录制旧规则 的配对决策质量（one-step 与 on-trajectory 两口径）+ 聚类 CI' },
    human_vs_a0: { type: 'string', description: '各臂 vs fixed-mean 的同口径对比' },
    blunders: { type: 'string', description: '各臂覆盖确定终局/被支配选择的次数与案例' },
    case_s3548246500: { type: 'string', description: '各臂在该 step37 的选择' },
    selected_for_screen: { type: 'string', description: '按预注册规则入选的臂及理由（没有合格臂则写 kill）' },
    checks: { type: 'array', items: { type: 'string' } },
  },
}

const SCREEN_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['status', 'arm', 'seeds', 'fallback_total', 'per_seed', 'vs_a0', 'vs_raw', 'kill_reasons', 'artifacts'],
  properties: {
    status: { type: 'string', enum: ['done', 'partial', 'failed', 'blocked'] },
    arm: { type: 'string' },
    seeds: { type: 'array', items: { type: 'number' } },
    fallback_total: { type: 'number' },
    per_seed: { type: 'array', items: { type: 'object', properties: { seed: { type: 'number' }, winrate: { type: 'number' }, elo: { type: 'number' }, elo_ci: { type: 'array', items: { type: 'number' } } } } },
    vs_a0: { type: 'string', description: '同 deal 配对 vs unitfix mean（seeds 30-32）的 d_win/d_margin/d_elo + CI' },
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
// Phase 1 — Implement
// ---------------------------------------------------------------------------
phase('Implement')
log('Phase Implement：给 batched_decide 加 O4_AGG（默认 mean 行为不变）')

const IMPL_PROMPT = [
  '仓库 cwd=' + REPO + '。任务：在**推理侧**给搜索的跨世界聚合加一个可选风险规则，默认行为必须与现状完全一致。',
  '',
  '改动文件（只允许改 runs/ 下的实验代码，不改 src/）：',
  '1) `runs/o4lite-search/rollout_policy.py` 的 `batched_decide`（line ~528-568）：它现在把 `champion.rollout_values` 返回的 per-world values 按候选取 `np.mean` 再 argmax。改为读环境变量 `O4_AGG`：',
  '   - `mean`（默认，逐 bit 保持现状）；',
  '   - `lcb`：`mean - z * std / sqrt(K)`，z 读 `O4_LCB_Z`（默认 1.0）；',
  '   - `cvar`：对候选的 K 个 value 升序，取最差 `ceil(alpha*K)` 个的均值，alpha 读 `O4_CVAR_ALPHA`（默认 0.25）；',
  '   - `guard`：若存在「所有 K 个世界都是终局（无截断读出）」的候选，则只在保底候选里按精确 margin 取最大者；否则退回 mean。',
  '   guard 需要知道每个 world 是否为终局。最小改动：给 `runs/o4lite-search/rollout_trunc.py` 的 `ValueChampion.rollout_values` 增加**可选**参数（如 `return_masks=False`）返回终止掩码；保底判定用「state.done 后由 `_margin` 赋值的 world」。保持默认签名与返回值不变，确保 `mechanism_probe.py`、`rollout_depth2.py`、`runs/ei*`、`play_search.py`、`web_search.py` 全部不受影响；全 rollout 的 `BatchChampion.rollout_values` 视为全部终局。',
  '2) 把生效的聚合配置写进运行产物：在 `run_h2h` 的 seed 输出 json 里增加 `aggregate`/`lcb_z`/`cvar_alpha` 字段（或等价的 config 记录），供下游复核；并在 wrapper 构造时打一行 log。',
  '',
  '自检（走 limiter cpu）：',
  '  ' + LIMITER + ' cpu ' + PY + ' -m py_compile runs/o4lite-search/rollout_policy.py runs/o4lite-search/rollout_trunc.py',
  '  写一个不加载 torch 的小断言脚本：给定人工构造的 per-world values/masks，逐一验证 mean/lcb/cvar/guard 的公式（含 K=32、alpha=0.25、z=1、全终局/混合/全截断三种构成）。',
  '',
  '输出：status、files、modes、env_contract、checks、risks。默认 `O4_AGG` 未设时任何行为差异都算失败。',
  '',
  RESOURCE_RULES,
].join('\n')

const impl = await agent(IMPL_PROMPT, { label: 'impl-agg', phase: 'Implement', schema: IMPL_SCHEMA, agentType: 'general-purpose' })
if (!impl || impl.status !== 'done') {
  log('Implement 失败，终止')
  return { impl, stopped: true }
}

// ---------------------------------------------------------------------------
// Phase 2 — Calibrate（含真人对战轨迹校验）
// ---------------------------------------------------------------------------
phase('Calibrate')
log('Phase Calibrate：479 真人决策上的偏差/方差分解 + 各臂 vs 旧规则/固定 mean')

const CALIB_PROMPT = [
  '仓库 cwd=' + REPO + '。你是 P1 标定子代理。目标：在**真人对战轨迹**上量化每个聚合臂的决策质量，并给出偏差/方差分解，供预注册规则选臂。',
  '',
  '数据与已存在工具：',
  '- 真人 bot 决策集：`runs/o4lite-search/unitfix/probe_human.py`（已存在）重放 `traces/web/run-20260928-091716/` 的 22 局 w2m_ctl 搜索局，479 个 bot 决策（t=5/K=32/C=6/critic_B/rollout 对手=champion，单位已修复）。先完整复跑它确认 479/fallback=0（把结果当 sanity），然后扩展。',
  '- 基线：`runs/o4lite-search/unitfix/seed{30,31,32}.json` = fixed-mean（A0，co +128.13）；录制动作 = 部署时的**旧规则**（buggy mean，未乘 total_points）。',
  '',
  '任务（脚本 `' + OUT + '/calib_human.py`，结果 `' + OUT + '/calib.json`）：',
  '1) 对 479 个决策、每个候选模板、每个 K=32 世界，记录 per-world value（分单位）与是否终局（guard 掩码）。之后所有臂在这份缓存上离线聚合，避免重复 rollout。',
  '2) 偏差/方差分解：对每个决策的 chosen 候选，比较 `mean_K` 与「单世界 champion 全量续行」代理真值（同 C7/C8 口径），给出 bias（均值差）、across-world sd、se=sd/sqrt(K)、以及 bias/se 占比；同时给 K∈{4,8,16,32} 的 mean_K 收敛曲线（描述性）。',
  '3) **真人对战轨迹校验（一级判据）**：对每个臂 A（含参数扫描：lcb z∈{0.5,1,2}、cvar α∈{0.1,0.25,0.5}、guard，以及参照 A0=mean 和录制旧规则）报告：',
  '   (a) 改判数 vs 录制旧规则、vs A0；',
  '   (b) **决策质量（one-step counterfactual）**：臂选动作 vs 录制动作，在同一 K=32 隐藏世界用 champion 双方续行到终局，取 bot 座位 margin 的均值；逐决策配对差，按 22 局聚类 bootstrap 95% CI；报告 better/worse/tie；',
  '   (c) **决策质量（on-trajectory replay）**：臂选动作后，优先按 trace 回放**真人实际动作**（若在新状态下仍合法；不合法则回退 champion 续行），继续到终局；同口径配对差与聚类 CI。这个口径是「真人对战轨迹校验」的主口径之一，必须给出合法/回退比例；',
  '   (d) blunder：该决策存在「全终局保底候选」且其精确 margin 严格大于臂选中候选的代理真值时，记一次；各臂次数 + 明细；',
  '   (e) `g0000__s3548246500` step37 各臂选了什么、代理真值各多少。',
  '4) 预注册选臂规则（写死并遵守）：入选 screen 的臂必须同时满足——相对录制旧规则在 (b) 或 (c) 的配对 CI 下界 ≥ 0 且点估计 ≥ +2 分/决策（或 blunder 从有到 0 且 (b)/(c) 点≥0）；满足者取点估计最高；没有满足者写 `kill`（说明 P1 在此数据集上不优于旧规则，转 P2）。允许最多 2 个入选（primary/secondary）。',
  '',
  '纪律：torch 走 gpu 槽，单进程；总计算 ≤25 分钟，先保 (2)(3b)(3e) 出结论再补 (3c)。不改仓库既有文件（只新增 ' + OUT + '/）；不 commit。',
  '',
  RESOURCE_RULES,
].join('\n')

const calib = await agent(CALIB_PROMPT, {
  label: 'calib-human',
  phase: 'Calibrate',
  schema: CALIB_SCHEMA,
  agentType: 'general-purpose',
  gate: PY + ' -c "import json;d=json.load(open(\'runs/o4lite-search/riskreadout/calib.json\'));assert d[\'status\'] in (\'done\',\'partial\')"',
})

if (!calib || calib.status === 'blocked') {
  log('Calibrate 失败，终止')
  return { impl, calib, stopped: true }
}
log('Calibrate 完成；selected_for_screen=' + calib.selected_for_screen)

const selected = (calib.selected_for_screen || '').toLowerCase()
if (selected.includes('kill')) {
  log('真人轨迹校验未发现优于旧规则的臂 → 不启动 screen，建议转 P2')
  return { impl, calib, stopped: true, reason: 'no arm beats old rule on human trajectories' }
}

// ---------------------------------------------------------------------------
// Phase 3 — Screen
// ---------------------------------------------------------------------------
phase('Screen')
log('Phase Screen：入选臂 same-bank 30/31/32 h2h')

const SCREEN_PROMPT = [
  '仓库 cwd=' + REPO + '。你是 screen 子代理：把 Calibrate 入选的臂跑 same-bank h2h，并与 A0（fixed mean）配对对比。',
  '',
  '入选信息见 `' + OUT + '/calib.json` 的 `selected_for_screen`（最多 2 个臂，primary/secondary）。',
  '配置与 `runs/o4lite-search/unitfix/seed{30,31,32}.json` 完全一致（同 bank/deal、同 t=5/K=32/C=6/critic_B/对手=champion、run_h2h_trunc.py），只额外设置环境变量 `O4_AGG=<臂>`（及 `O4_LCB_Z` / `O4_CVAR_ALPHA`）。',
  '',
  '执行（一个 bash 调用；数量 ≤2 臂 × 3 seeds，按 ≤4 GPU 槽排队，整段 `timeout 5400`）：',
  '  mkdir -p ' + OUT + '；对每个 (arm, seed) 启动单独进程，各自包 `' + LIMITER + ' gpu`，日志到 ' + OUT + '/<arm>.seed<s>.log；等全部结束；',
  '  然后 `' + LIMITER + ' cpu ' + PY + ' runs/o4lite-search/combine.py --arm riskreadout_<arm> --in ' + OUT + ' --tb runs/o4lite-search/tb/riskreadout --seeds 30,31,32`（若 combine 需要每臂独立目录，就每臂一个子目录，自行读 combine.py 用法后照做）。',
  '',
  '报告（写 `' + OUT + '/screen.json` 并同样返回）：每 seed winrate/elo/CI、fallback 总数、kill_reasons、',
  '  - **vs_a0**：与 `runs/o4lite-search/unitfix/seed{30,31,32}.games.jsonl` 做逐 deal 配对（同 seed=同 deal），给 d_win / d_margin / d_elo 与 CI；',
  '  - **vs_raw**：各臂对 raw 的 Elo/winrate。',
  'Green 判据（预注册）：vs_a0 的配对 d_margin CI 下界 ≥ 0 且 winrate 点估计不降；若 CI 完全在负侧 → 该臂 kill。timeout 就 `pkill -f run_h2h` 并如实报 partial。',
  '',
  '不许改 rollout 代码（Implement 已完成）；只新增 ' + OUT + '/ 下文件；不 commit。',
  '',
  RESOURCE_RULES,
].join('\n')

const screen = await agent(SCREEN_PROMPT, {
  label: 'screen-agg',
  phase: 'Screen',
  schema: SCREEN_SCHEMA,
  agentType: 'general-purpose',
  gate: PY + ' -c "import json;d=json.load(open(\'runs/o4lite-search/riskreadout/screen.json\'));assert d[\'status\']==\'done\';assert d[\'fallback_total\']==0"',
})

// ---------------------------------------------------------------------------
// Phase 4 — Verify（每个点 1 个对抗审查）
// ---------------------------------------------------------------------------
phase('Verify')

const V1_PROMPT = [
  '仓库 cwd=' + REPO + '。你是对抗审查子代理，独立审查一个 claim（每点只你 1 个审查者）。',
  '',
  '待审 claim [HUMAN-TRAJECTORY]：Calibrate 的结论——在 22 局真人轨迹的 479 个 bot 决策上，入选臂的决策质量优于旧推理规则（录制动作）与 fixed-mean A0。',
  '证据：`' + OUT + '/calib.json` 与 `' + OUT + '/calib_human.py`；原始轨迹 `traces/web/run-20260928-091716/`。',
  '',
  '三轴攻击：',
  '(a) 独立复现：自己写脚本重算关键数字（各臂改判数、one-step/on-trajectory 配对差、聚类 CI、blunder），核对 calib 里的数；核对 479/fallback=0 与 policy_seed(trace.seed, human_seat) 约定；',
  '(b) 替代解释：on-trajectory 回放是否作弊（真人动作不合法时回退比例、回退引入的偏置、隐藏世界与真人动作不匹配）；one-step 代理真值是 champion 模型内反事实（同 C8 的适用域限制）；选臂规则是否被后验挑选（扫参后只报最好的）；',
  '(c) 统计/适用域：22 局/单人/单日的聚类 CI、多重比较、blunder 定义；结论能否外推到「对真人更强」还是只是「对 champion 代理更强」。',
  '',
  'verdict 规则：confirmed=你独立复现且 claim 成立；weakened=方向对但过强/口径错/适用域应缩小（必须 corrected_claim）；refuted=不成立；undecidable=数据不足。不能独立复现不得投 confirmed。',
  '只读仓库与 /tmp/adv/P1H/；不改文件；禁止嵌套子代理。',
  '',
  RESOURCE_RULES,
].join('\n')

const V2_PROMPT = [
  '仓库 cwd=' + REPO + '。你是对抗审查子代理，独立审查一个 claim（每点只你 1 个审查者）。',
  '',
  '待审 claim [SCREEN]：screen 代理报告的「入选臂 vs fixed-mean A0（unitfix seeds 30-32）」的 h2h 结果（`' + OUT + '/screen.json` 与其产物）是真实的，变化/不变化不是流水线、环境变量未生效或 deal 错配造成的。',
  '证据：`' + OUT + '/` 下各 seed 的 json/games.jsonl/log 与 `runs/o4lite-search/unitfix/seed{30,31,32}.{json,games.jsonl}`。',
  '',
  '三轴攻击：',
  '(a) 独立复现：从两边 games.jsonl 自己重算 winrate/Elo（src/seven523/duel.py:71-74）与逐 deal 配对；核对 deal schedule 完全同批；核对 `seed*.json` 里的 aggregate/lcb_z/cvar_alpha 字段与命令行一致；',
  '(b) 替代解释：环境变量是否真的进入了子进程（O4_AGG 生效证据：wrapper log / json 字段 / 进程 environ）、是否误用了未修复的旧 rollout_trunc、runner/引擎版本、fallback/abort；',
  '(c) 统计/适用域：3 seeds × 400 deals 的配对 CI 能把多大差异分辨出来；bank 30-32 的复用史；结论只对 raw 成立、不可外推真人。',
  '',
  'verdict 规则同上。只读仓库与 /tmp/adv/P1S/；不改文件；禁止嵌套子代理。',
  '',
  RESOURCE_RULES,
].join('\n')

const [humanReview, screenReview] = await parallel([
  () => agent(V1_PROMPT, { label: 'review-human', phase: 'Verify', schema: VERDICT_SCHEMA, agentType: 'general-purpose' }),
  () => agent(V2_PROMPT, { label: 'review-screen', phase: 'Verify', schema: VERDICT_SCHEMA, agentType: 'general-purpose' }),
])

log('P1 完成：calib/screen/两份对抗裁决已产出')
return { impl, calib, screen, humanReview, screenReview }

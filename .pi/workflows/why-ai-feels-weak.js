export const meta = {
  name: 'why-ai-feels-weak',
  description: '研究「最强 AI 仍被人类打赢」的根因：多角度取证 → 对抗证伪 → 门槛判决 → 实验设计 → 启动 500k 步实验（≤6 并发，TensorBoard）',
  whenToUse: '当用户问为什么系统里的最强 AI 仍然偏弱、要求多 agent 研究并跑实验时',
  phases: [
    { title: 'Research', detail: '六个独立角度取证（只读）' },
    { title: 'Verify', detail: '对每条 claim 尝试证伪' },
    { title: 'Gate', detail: '门槛判决：哪些真正值得动手' },
    { title: 'Design', detail: '实验规格 + 对抗审阅' },
    { title: 'Launch', detail: '启动实验并核验 TensorBoard' },
  ],
}

const REPO = '/home/amas/.local/src/7g523'
const CONCURRENCY = 6
const ROUND_CAP = 500_000 // 每轮 agent 输出 token 软上限（budget.spent() 口径）
const MAX_ROUNDS = 3

const SHARED = `
项目：${REPO} 是「7鬼523」纯规则引擎 + Gym RL 环境（2 家、54 张牌、revision-3 出空即撬底、rules_id=2e36dbea44893696、obs v5 161 维、动作 MultiDiscrete([134,4])、纯 MLP arch=shared hidden=128、PPO）。
用户问题：「和系统里最强的 AI 对战仍然感觉 AI 太弱」。
第一手证据（已由主 agent 复核；你必须自行复算，以你自己的复算为准）：
- 用户现行规则（v3 trace）人机轨迹：traces/sessions/session-20260927-*/g*.json 与 traces/web/run-20260927-153209/g*.json（trace 键：initial/steps/final_scores/players/human_seat；final_scores 按座位索引，human 分 = final_scores[human_seat]）。
- 主 agent 复算：22 局、18 胜 3 负 1 平、均分 71.4:28.6；对当日顶簇 w2m_ctl 5-0、w2m_plain 2-0、ws_s2 1-2、w2m_low 1-0、pself_s2 0-1；lvl1 5 局 4-0-1、lvl2 2-0、lvl3 2-0。另有 2026-09-25 的 47 局 v2 旧口径轨迹（greedy/random/s2..s10 旧池），仅在注明旧口径时可参考，不得与现行 v3 结果混比。
- placement 报告：traces/sessions/session-20260927-133726/report.json 估人类 mu=215.2±25.5（CI 165-265，6 局，provisional）；session-20260927-115300 估 113.1±49.6（6 局）。全部登记 bot 顶部簇仅 186-191（traces/pool10/manifest.json、tools/play_ladder.py）。
可用证据：docs/experiments/*（warmstart-adamw-1m.md 首个过门槛 +11.48、v5-optimization-wave1、selfplay-pool-*、event-history-pilot、opponent-intent-probe、sequence-*、reward-alignment-* 等）、docs/*-plan.md+docs/*-review.md（五份已红队修订的计划）、docs/post-v5-structural-options.md、docs/plans.md、docs/experiments/README.md、traces/{study,sessions,web}、runs/t18-t23、runs/w2m/calibration/report.md、runs/**/*.json（h2h 原始读数）、tools/*.py、src/seven523/*。
硬规则：
- 研究/验证阶段只读：不改任何文件、不跑训练/h2h/arena、不 git commit/checkout/stash；允许运行只读的 python/rg/ls/cat 分析片段（不写盘）。
- 引用必须落到 file:line 或可复现命令；不要相信文档里的数字，以 runs/ 原始产物为准；注明口径（revision-3、obs v5、跨 fit 绝对分不可比）。
- 行动门槛（仓库统一口径）：值得行动 = ≥3 seed × 400 副 deal-twin 换座 h2h 合并 95% CI 完全排除 0 且点估计 ≥ +10 Elo；<+5 止损；单 seed 不定论。
- 已被否决的方向（plans.md §6 N-1..N-20）不得重新推荐，除非有新的一手反证。
- 输出用中文；结论要少而硬。
`

const RESEARCH_SCHEMA = {
  type: 'object',
  required: ['angle', 'summary', 'claims', 'open_questions'],
  properties: {
    angle: { type: 'string' },
    summary: { type: 'string', description: '3-6 句：这个角度最硬的发现' },
    claims: {
      type: 'array',
      items: {
        type: 'object',
        required: ['statement', 'mechanism', 'evidence', 'strength', 'actionability', 'discriminating_test', 'caveats'],
        properties: {
          statement: { type: 'string', description: '一句话可证伪的断言' },
          mechanism: { type: 'string', description: '因果链：为什么它导致 AI 弱（对人类可见）' },
          evidence: {
            type: 'array',
            items: {
              type: 'object',
              required: ['path', 'what_it_shows'],
              properties: {
                path: { type: 'string' },
                locator: { type: 'string', description: 'file:line / 键 / step 号' },
                what_it_shows: { type: 'string' },
                verified: { type: 'boolean', description: '你是否亲手打开核对了' },
              },
            },
          },
          strength: { type: 'string', enum: ['strong', 'moderate', 'weak'] },
          actionability: { type: 'string', enum: ['high', 'medium', 'low'] },
          discriminating_test: { type: 'string', description: '最便宜的可证伪检查' },
          proposed_experiment: { type: 'string', description: '可选的实验设想（具体命令更好）' },
          caveats: { type: 'string' },
        },
      },
    },
    dead_ends: { type: 'array', items: { type: 'string' } },
    open_questions: { type: 'array', items: { type: 'string' } },
  },
}

const VERIFY_SCHEMA = {
  type: 'object',
  required: ['claim_id', 'verdict', 'evidence_accuracy', 'counterarguments', 'notes'],
  properties: {
    claim_id: { type: 'string' },
    verdict: { type: 'string', enum: ['upheld', 'weakened', 'refuted', 'inconclusive'] },
    evidence_accuracy: { type: 'string', enum: ['accurate', 'partly_wrong', 'wrong', 'unchecked'] },
    counterarguments: {
      type: 'array',
      items: {
        type: 'object',
        required: ['path', 'argument'],
        properties: { path: { type: 'string' }, argument: { type: 'string' } },
      },
    },
    notes: { type: 'string' },
  },
}

const GATE_SCHEMA = {
  type: 'object',
  required: ['actionable', 'rejected', 'open_gaps', 'diagnosis', 'confidence'],
  properties: {
    actionable: {
      type: 'array',
      items: {
        type: 'object',
        required: ['claim_id', 'statement', 'why_now', 'experiment_sketch', 'target_weakness'],
        properties: {
          claim_id: { type: 'string' },
          statement: { type: 'string' },
          why_now: { type: 'string' },
          experiment_sketch: { type: 'string' },
          target_weakness: { type: 'string', description: '对应哪个人类可见的弱点' },
        },
      },
    },
    rejected: {
      type: 'array',
      items: {
        type: 'object',
        required: ['claim_id', 'reason'],
        properties: { claim_id: { type: 'string' }, reason: { type: 'string' } },
      },
    },
    open_gaps: { type: 'array', items: { type: 'string' } },
    diagnosis: { type: 'string', description: '整体诊断（中文，可直接给用户看）' },
    confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
  },
}

const KILL_SCHEMA = {
  type: 'object',
  required: ['claim_id', 'kill', 'strongest_objection', 'notes'],
  properties: {
    claim_id: { type: 'string' },
    kill: { type: 'boolean', description: 'true = 这个实验不该现在跑' },
    strongest_objection: { type: 'string' },
    could_be_salvaged_by: { type: 'string' },
    notes: { type: 'string' },
  },
}

const SPEC_SCHEMA = {
  type: 'object',
  required: ['spec_id', 'title', 'hypothesis', 'claim_ids', 'arms', 'primary_endpoint', 'kill_criteria', 'tb', 'new_files', 'risks', 'concurrency'],
  properties: {
    spec_id: { type: 'string' },
    title: { type: 'string' },
    hypothesis: { type: 'string' },
    claim_ids: { type: 'array', items: { type: 'string' } },
    arms: {
      type: 'array',
      items: {
        type: 'object',
        required: ['name', 'kind', 'commands', 'steps', 'est_minutes'],
        properties: {
          name: { type: 'string' },
          kind: { type: 'string', enum: ['train', 'eval', 'probe'] },
          commands: { type: 'array', items: { type: 'string' }, description: '完整可复制命令（相对仓库根）' },
          steps: { type: 'number', description: '训练步数；评估/探针填 0' },
          est_minutes: { type: 'number' },
          run_dir_hint: { type: 'string' },
        },
      },
    },
    primary_endpoint: { type: 'string' },
    kill_criteria: { type: 'string' },
    tb: { type: 'string', description: 'TensorBoard 日志位置与要看的标量' },
    new_files: {
      type: 'array',
      items: {
        type: 'object',
        required: ['path', 'purpose'],
        properties: { path: { type: 'string' }, purpose: { type: 'string' } },
      },
      description: '只允许 runs/ 下的新文件；空数组 = 零改动',
    },
    risks: { type: 'string' },
    concurrency: { type: 'number' },
  },
}

const SPEC_REVIEW_SCHEMA = {
  type: 'object',
  required: ['spec_id', 'verdict', 'fatal_flaws', 'required_fixes', 'command_checks', 'notes'],
  properties: {
    spec_id: { type: 'string' },
    verdict: { type: 'string', enum: ['run', 'fix', 'kill'] },
    fatal_flaws: { type: 'array', items: { type: 'string' } },
    required_fixes: { type: 'array', items: { type: 'string' } },
    command_checks: {
      type: 'array',
      items: {
        type: 'object',
        required: ['command', 'result'],
        properties: { command: { type: 'string' }, result: { type: 'string' } },
      },
    },
    notes: { type: 'string' },
  },
}

const LAUNCH_SCHEMA = {
  type: 'object',
  required: ['spec_id', 'status', 'processes', 'notes'],
  properties: {
    spec_id: { type: 'string' },
    status: { type: 'string', enum: ['launched', 'partial', 'failed', 'blocked'] },
    processes: {
      type: 'array',
      items: {
        type: 'object',
        required: ['name', 'pid', 'run_dir', 'command', 'log_file', 'alive', 'evidence'],
        properties: {
          name: { type: 'string' },
          pid: { type: 'number' },
          run_dir: { type: 'string' },
          command: { type: 'string' },
          log_file: { type: 'string' },
          tb_event_file: { type: 'string' },
          metrics_csv: { type: 'string' },
          alive: { type: 'boolean' },
          evidence: { type: 'string', description: '存活/日志/metrics/TB 的实测证据' },
        },
      },
    },
    notes: { type: 'string' },
  },
}

const FINAL_CHECK_SCHEMA = {
  type: 'object',
  required: ['checks', 'all_ok', 'summary'],
  properties: {
    checks: {
      type: 'array',
      items: {
        type: 'object',
        required: ['spec_id', 'run_dir', 'process_alive', 'metrics_advancing', 'tb_events_present', 'verdict'],
        properties: {
          spec_id: { type: 'string' },
          run_dir: { type: 'string' },
          process_alive: { type: 'boolean' },
          metrics_advancing: { type: 'boolean' },
          tb_events_present: { type: 'boolean' },
          verdict: { type: 'string', enum: ['ok', 'degraded', 'dead'] },
          notes: { type: 'string' },
        },
      },
    },
    all_ok: { type: 'boolean' },
    summary: { type: 'string' },
  },
}

// ---------------- angle missions ----------------

const ANGLES = [
  {
    key: 'trace-forensics',
    prompt: `${SHARED}
你的角度：人机轨迹取证（第一手证据）。
任务：逐局回放用户最近 26 局（traces/sessions/session-20260927-*/g*.json 与 traces/web/run-20260927-153209/g*.json）：
1) 复算按对手/座位的人类胜负与均分（注意 final_scores 按座位索引），核对 19-1-6 与均分 65:35。
2) 找 bot 的具体丢分决策：哪些 step 中 bot 的动作明显劣于一个简单人类启发式（能收分不收、无谓送分、炸弹时机、垫牌、末手/撬底处理、对手手牌数与已出牌信息利用）。用 src/seven523 引擎在同一局面枚举合法动作做对照（只读 python 片段即可），给出 trace 文件 + step 号 + 前后比分。
3) 找人类在这些局里做对而 bot 做错的对偶局面，总结 3-6 个可复现的「行为模式」。
4) 输出 5-10 条可证伪的 claim（每条含至少一个反例检查方法），并明确指出这批轨迹的采样局限（n 小、对手混合、placement 自适应选敌、free-play 与 placement 的差异、是否只有部分局落盘）。
区分「偶发错误」与「系统性模式」：系统性模式需要 ≥3 个独立局面支撑，否则标 weak。`,
  },
  {
    key: 'exploitability',
    prompt: `${SHARED}
你的角度：可利用性 / 结构性盲点。
读 src/seven523/{env,networks,actions,game,policies}.py、DESIGN.md §4（obs v5 段表）、以及已有探针报告：docs/experiments/{opponent-intent-probe,reward-alignment-diagnostic,event-history-pilot,sequence-memory-pilot,history-fusion-ablation,selfplay-pool-diagnostics,structural-directions}.md。
问题：人类能看见、而 top bot 结构性看不见/算不出的东西是什么？
候选机制（逐一核对现有证据，别空想）：无显式搜索/前瞻（纯 MLP 单步策略）、无对手建模（v5 只有 counts/revealed/last_player，没有逐决策意图）、无完整出牌序列（unseen+last_player 是否足够支撑记牌与推理）、价值头塌缩（有文档证据）、终局稀疏奖励、动作/花色头的表达限制、训练对手分布（池成员全弱于学习者）。
要求：
1) 把每个候选机制映射到「人类可观察的弱行为」；若 trace 取证角度尚未产出，则引用文档中的行为证据。
2) 查哪些假设已被证伪/未确认（B1 未确认、C null、双塔关闭等），不要重复 plans.md §6 已否决清单。
3) 为每个机制设计最小判别探针（例如：残局/构造局面上 1-ply 搜索对照、行为审计、对 top ckpt 的反事实评测），标注成本与命令。
4) 每条 claim 附 file:line 证据与「什么结果会杀死它」。`,
  },
  {
    key: 'plateau',
    prompt: `${SHARED}
你的角度：训练动力学与平台。
证据入口：docs/experiments/{warmstart-adamw-1m,v5-optimization-wave1,selfplay-pool-wave1,selfplay-pool-diagnostics,wave5-500k-report,sequence-memory-pilot,sequence-gru-ablation}.md、docs/post-v5-structural-options.md、runs/t18..t23、runs/w2m/calibration/report.md、traces/pool10/manifest.json。
任务：
1) 核对 t23wswd（warm start l1_1M、1M 步、AdamW+wd=0.01+cosine、pool>150、batch×2）的 +11.48：打开 runs/t23/h2h_final_*.json 与 h2h_ref2m_* 原始读数，确认 k、合并口径、CI 与参照；判断当前「首个过门槛」是否可靠、缺什么才敢下结论。
2) 列出所有已测训练 lever 及其状态（null / positive / untested / blocked）并给一手证据路径；特别核对：500k vs 1M/2M、hidden 容量、序列/事件编码（seq-*/event-* 已在 train.py 实现）、奖励 shaping（terminal_win/saturate/reward-cap）、对手分布（self/mix/pool/PFSP）。
3) 找出「尚未被证伪、与人类可见弱点直接相关、且本机 500k 步可跑」的最高 EV 训练干预（1-2 个），说明为什么它与 t23wswd 的 +11.48 不重复。
4) 给出可复制命令草图（真实 ckpt 路径从 manifest/traces 里取），标明 seed/步数/并发/显存风险。`,
  },
  {
    key: 'measurement',
    prompt: `${SHARED}
你的角度：测量与服务审计（用户打的到底是不是最强、差异是否可信）。
任务：
1) 查 tools/play_ladder.py 注册表与 src/seven523/placement/{estimator,session}.py 的自适应选敌：web/placement 实际会选到哪些 id？看 traces/sessions/*/session.json、rungs.json 与逐局 players 标签，把用户 26 局按 bot id/座位/来源（free-play vs placement）完整映射；确认是否真的与最强簇（w2m_ctl/w2m_low/w2m_plain/ws_s2/pself_s2）交手。
2) 核对 traces/pool10/manifest.json 与 runs/w2m/calibration/report.md：评分口径、rungs 冻结、top 簇内部差异（用 runs/t23/h2h_*.json 的 k/CI 判断「系统最强」是否唯一可信）。
3) 对「人类 19/26 胜、平均分差 +30」做基本统计：混合对手下这能区分多强的真实优势？placement 的 215±25（6 局）与 113±50（6 局）为什么差这么多（对手选择、座位、牌运、先验）？给出最少局数与判据。
4) 产出一个可操作的「对用户而言 AI 够强」判据（端点 + 最小局数 + 可接受的方差），供实验设计复用。`,
  },
  {
    key: 'variance',
    prompt: `${SHARED}
你的角度：方差 / 天花板（多少「弱」是运气与噪声）。
数据：runs/t23/h2h_*.json（per_seed/combined，deals/games/k）、runs/**/games*.jsonl（若存在逐局记录）、traces/sessions/*/g*.json（人类逐局比分与 steps）、traces/pool10/manifest.json 的 σ。
任务：
1) 提取同牌换座 twin 设计下候选间每局分差的分布：deal 方差、座位方差各占多少；顶簇候选互相之间的单局胜率与爆冷率。
2) 估计 RandomBot / 弱梯级对 top bot 的单局得分分布与爆冷率（有 JSONL 就实算，没有就用报告数字并注明口径），对比人类 100:0 与 10:90 交替出现的现象。
3) 回答：用户 26 局 73% 胜率，有多少可归因于 bot 真弱、多少是运气与对手混合？给出功效分析：要把「人类比 top bot 强 25 Elo」与「其实打平」分开，需要多少局（同牌 twin 不可行时给出替代协议：座位轮换、分段报告、混合对手下的最小 N）。
4) 结论必须能支撑或否证这条 claim：「顶簇 bot 在绝对强度上就低于用户，而非采样噪声」——给出你自己的判断与不确定性。`,
  },
  {
    key: 'frontier-census',
    prompt: `${SHARED}
你的角度：可跑实验普查（把「可能有用」变成「现在能跑」）。
读：docs/v5-optimization-plan.md + v5-optimization-review.md、docs/event-history-plan.md + event-history-review.md、docs/opponent-intent-plan.md + opponent-intent-review.md、docs/reward-alignment-plan.md + reward-alignment-review.md、docs/selfplay-pool-plan.md + selfplay-pool-review.md、docs/post-v5-structural-options.md、docs/experiments/{v5-optimization-wave1,event-history-pilot,opponent-intent-probe,reward-alignment-diagnostic,reward-alignment-saturate,reward-alignment-terminal-win,selfplay-pool-wave1,selfplay-pool-diagnostics,sequence-memory-pilot,sequence-gru-ablation}.md。
产出：
1) 「机制 × 状态（已实现 / 已测-null / 已规划可跑 / 被阻塞需决策）× 预期收益（有原始 run 证据的才写数字）× 成本 × 需要的代码改动 × 预注册端点」表。
2) 对每个「已规划可跑」项，找出计划里给出的现成命令/工具开关（train.py 已实现 seq-*/event-*/reward-* 等 flag，先用 --help 与 src/seven523/train.py 核对），列出可复制命令。
3) 标注哪些机制直接针对人类可见弱点（无搜索、无对手建模、无记忆、终局奖励、对手分布）；哪些卡在用户决策（如中性起点消融、事件历史阶段 B）；哪些与已否决清单冲突。
4) 输出 5-8 个候选实验（含命令、成本、端点、kill 条件），按「证据强度 × 与人类弱点的相关性 ÷ 成本」排序。`,
  },
]

// ---------------- prompts for later phases ----------------

function followupPrompts(gaps) {
  const list = gaps.slice(0, CONCURRENCY)
  return list.map((gap, i) => ({
    key: `gap-${i + 1}`,
    prompt: `${SHARED}
这是追查轮：只针对下面这个缺口做更深的一手取证（读文件、跑只读分析），不要重复已被确认的结论，也不要重新提出已被否决的方向。产出尽量少但更硬的 claims（每条至少一个一手证据 + 一个能证伪它的检查 + 与人类可见弱点的因果链）。
缺口：${gap}`,
  }))
}

function claimDigest(state) {
  const cut = (s, n) => (s || '').slice(0, n)
  const confirmed = state.confirmed.map((c) =>
    `- [${c.verdict}] ${cut(c.statement, 300)} | 机制: ${cut(c.mechanism, 200)} | 证据: ${(c.evidence || []).slice(0, 3).map((e) => e.path).join(' ; ')}`
  )
  const rejected = state.rejected.map((r) => `- ${cut(r.statement, 200)} | 否决理由: ${cut(r.reason, 200)}`)
  return `已确认（含verdict与证据）：\n${confirmed.join('\n') || '（无）'}\n\n已否决：\n${rejected.join('\n') || '（无）'}`
}

function gatePrompt(state, round) {
  return `${SHARED}
你是门槛判决官（第 ${round} 轮结束）。下面是本轮经对抗验证后的 claim 清单。请判断哪些「真正值得现在付出实践」。
${claimDigest(state)}
开放缺口（供 open_gaps 用）：${(state.gaps || []).slice(0, 8).join('；') || '（无）'}

判决标准（全部满足才算 actionable）：
(a) 机制有已核实的一手证据，且与「人类可见的 AI 弱行为 / 强度提升」有直接因果；
(b) 用现有工具或 <500 行、只写 runs/ 下新脚本的实验即可做出决定性检验；
(c) 不被 plans.md §6 已否决清单或本轮证据覆盖；
(d) 端点分辨率足够（相对比较须能识别 ≥+10 Elo；或行为性端点有明确成功判据）；
(e) 成本 ≤ 500k 训练步/arm、≤6 并发、单臂 ≤ 1 天；
(f) 默认怀疑：证据弱、机制玄学、端点含噪的一律进 rejected/open_gaps。
对每个 actionable 给：对应弱点、为什么现在做、最小实验草图（哪个臂、什么端点、kill 条件）。
diagnosis 用中文写 5-10 句整体诊断，可直接呈现给用户（说明 AI 弱在哪里、为什么、哪些解释被证据否掉）。`;
}

function killPrompt(a) {
  return `${SHARED}
你是必须杀死这个实验的审查者。有人提议现在跑它：
- claim: ${a.statement}
- 理由: ${a.why_now}
- 草图: ${a.experiment_sketch}
你的工作：翻 docs/experiments/、docs/plans.md §6 N 清单、runs/ 找证据证明它 (i) 已经做过或被更早的证据否决；(ii) 端点分辨率不足（<+10 Elo 或无法证伪）；(iii) 与人类可见弱点没有因果；(iv) 有更便宜的判别方式（那就要求先做便宜的）；(v) 只是重复 t23wswd 或已测 lever。
除非它明显能过门槛且有决定性，否则 kill=true。给出最强反对理由，以及（若有）最小可挽救版本。`;
}

function designPrompt(a, state) {
  return `${SHARED}
你是实验设计师。把下面这个已过门槛的看法变成「可直接执行、可证伪」的实验规格：
- claim: ${a.statement}
- 目标弱点: ${a.target_weakness}
- 草图: ${a.experiment_sketch}

环境事实：仓库 ${REPO}；GPU RTX 4060 8GB，32 核，15GB 内存；.venv 已装 torch cu132（cuda 可用）；训练入口「uv run --group train 7g523-train …」（有 --total-timesteps/--load-checkpoint/--opponent/--seq-*/--event-*/--reward-shaping/--optimizer/--weight-decay/--lr-schedule/--hidden-size/--num-envs/--seed/--tensorboard，默认写 runs/<exp>__<seed>__<ts>/metrics.csv 与 tb/）；h2h 入口「uv run --group train python tools/head_to_head.py --left A=… --right B=… --seeds 0,1,2 --pairs 400 --device cuda」；现有最强 ckpt 路径见 traces/pool10/manifest.json 与 tools/play_ladder.py。
硬约束：
- 每臂训练预算默认 500k 步；评估必须 ≥3 seed × 400 副 deal-twin 换座（或行为端点的明确判据）；总并发进程 ≤6。
- 不许改 src/tests/docs/ADR；需要新代码只能放 runs/<spec_id>/ 下的脚本（gitignored），并在 spec.new_files 列出。
- 每个训练臂默认已写 TB；评估/探针臂要显式把结果标量写进 runs/<spec_id>/tb（tensorboardX 已装）。
- 给出 primary_endpoint、kill_criteria（什么结果算失败）、预期耗时。
- 命令必须是能直接复制执行的完整行（含相对仓库根的路径）。
如果 claim 与已有负证据冲突，直接说明并给替代设计。`;
}

function reviewSpecPrompt(s) {
  return `${SHARED}
你是实验规格的对抗审阅者。规格如下：
${JSON.stringify(s, null, 2)}
任务：验证它能否真的执行、端点是否有分辨率、是否与已否决证据冲突。
- 对每条命令检查 flag/路径是否存在（可运行 --help、test -f、ls；只读，不启动训练）。
- 检查：ckpt 路径真实存在；warm start 是否兼容 v5；500k 步内有望产生可判读信号；kill 条件是否明确；TB 是否真的会写；并发是否 ≤6；是否偷偷依赖 src 修改。
- verdict: run（可执行）/ fix（列出 required_fixes）/ kill（说明 fatal_flaws）。宁严勿松。`;
}

function launchPrompt(s, running) {
  return `${SHARED}
你是实验启动器（串行启动：同一时刻只有你在启动，所以并发账目是确定的）。规格如下：
（本工作流此前已启动的进程：${running && running.length ? running.join(' | ') : '无'}）
${JSON.stringify(s, null, 2)}
执行规则：
1) 并发硬上限（全局 6，必须执行）：先把上面列出的 pid 逐个用「ps -p <pid>」核实存活，得到 S；再用「ps -eo args | grep '[7]g523-train' | grep -v grep」找出不属于上述列表的其他训练进程（若有，也计入 S）。你本次可启动的进程数 = 6 − S；若余量小于本规格 arms 数，只按 arms 顺序启动余量个臂、其余不启动并在 notes 注明；若余量 ≤0，返回 status="blocked"、processes=[]，不启动任何进程。显存检查：「nvidia-smi --query-gpu=memory.free --format=csv,noheader」，空闲 <2000 MiB 时训练命令改用 CPU（--cuda False）。
2) 只用 runs/ 下的路径写日志/脚本；绝不修改 src/tests/docs，绝不 git commit/checkout/stash，不删任何文件。
3) 启动用：「mkdir -p runs/<spec_id>」+「nohup <command> > runs/<spec_id>/<arm>.log 2>&1 & echo $!」（必要时 setsid）。命令若不确定 flag，先 --help 核对。
4) 启动后 sleep 60-90 秒，核验：进程存活（ps -p）、「ls -dt runs/<expname>__* | head -1」找到真实 run 目录、metrics.csv 行数 >0、tb/events.out.tfevents.* 存在、日志无 Traceback/NaN。若第一次检查太早，再等 60 秒复查一次。
5) 失败则读日志尾部，报 failed 与原因；不要重试超过一次。
6) 返回每个进程的 PID、绝对 run_dir、命令、TB event 文件、metrics.csv、最新指标行（作为 evidence）。`;
}

function finalCheckPrompt(launches) {
  return `${SHARED}
你是独立的启动核验员。下面是各启动器自报的结果，请逐项独立复核（不要相信自报）：
${JSON.stringify(launches, null, 2)}
对每个 run：「ps -p <pid>」或「pgrep -f <run_dir>」确认存活；「wc -l metrics.csv」与「tail -1」确认在推进；「ls <run_dir>/tb/events.out.tfevents.*」确认 TB 事件；如已结束，检查日志尾部是否正常完成还是崩溃。
输出：逐 run 的 verdict（ok/degraded/dead）+ all_ok + 中文 summary（含每个 run 的绝对路径与当前步数）。`;
}

// ---------------- orchestration ----------------

async function runBatch(thunks, size = CONCURRENCY) {
  const out = []
  for (let i = 0; i < thunks.length; i += size) {
    const chunk = thunks.slice(i, i + size)
    const res = await parallel(chunk)
    out.push(...res)
  }
  return out
}

const state = { confirmed: [], rejected: [], gaps: [], round: 0 }
let gate = null

for (let round = 1; round <= MAX_ROUNDS; round++) {
  state.round = round
  const r0 = budget.spent()
  log(`Round ${round} 开始；累计输出 token ≈ ${Math.round(budget.spent())}`)
  phase('Research')

  const planned = round === 1 ? ANGLES : followupPrompts(state.gaps.length ? state.gaps : ['如何把「AI 对人类可见地变强」变成可证伪的最近实验'])
  const researchRaw = await runBatch(
    planned.map((p) => () =>
      agent(p.prompt, { label: `r${round}:research:${p.key}`, phase: 'Research', schema: RESEARCH_SCHEMA })
    )
  )

  // flatten + dedupe claims（保持与 planned 下标对齐，允许 null 结果）
  const claims = []
  const seen = new Set()
  let okAngles = 0
  for (let i = 0; i < researchRaw.length; i++) {
    const r = researchRaw[i]
    if (!r) continue
    okAngles += 1
    const key = planned[i] ? planned[i].key : `a${i}`
    ;(r.claims || []).forEach((c, j) => {
      const norm = (c.statement || '').toLowerCase().replace(/\s+/g, '')
      if (!norm || seen.has(norm)) return
      seen.add(norm)
      claims.push({ ...c, id: `${key}-${j + 1}`, angle: key, verdict: 'pending' })
    })
  }
  log(`Round ${round}: ${okAngles} 个角度产出 ${claims.length} 条去重 claims`)

  // pick top claims for adversarial verification
  const rank = (c) => {
    const act = c.actionability === 'high' ? 3 : c.actionability === 'medium' ? 2 : 1
    const str = c.strength === 'strong' ? 3 : c.strength === 'moderate' ? 2 : 1
    const ev = Math.min(3, (c.evidence || []).length)
    return act * 3 + str * 2 + ev
  }
  const selected = claims.slice().sort((a, b) => rank(b) - rank(a)).slice(0, 8)

  if (budget.spent() - r0 > ROUND_CAP) {
    log(`Round ${round}: 研究阶段已超每轮预算 ${ROUND_CAP}，跳过验证直接进入判决`)
    state.gaps = state.gaps.concat(selected.map((c) => `未验证：${c.statement}`))
    gate = null
  } else {
    phase('Verify')
    const tasks = []
    for (const c of selected) {
      const lenses = c.actionability === 'high' ? ['evidence', 'alternative'] : ['evidence']
      for (const lens of lenses) tasks.push({ c, lens })
    }
    const verdicts = (await runBatch(
      tasks.map((t) => () =>
        agent(
          `${SHARED}
你是对抗验证者，任务是尽量证伪下面这条 claim。lens=${t.lens}：
- evidence 视角：逐条打开它引用的证据，核对原文/数字/路径是否真的支持它；找同仓库中相反证据；证据造假或不支持即 refuted/weakened。
- alternative 视角：即使证据为真，是否存在更平凡的解释（方差/采样/已知 null 结果/已被否决方向）能解释它；找不到才 upheld。
claim：${JSON.stringify(t.c, null, 2)}
默认怀疑；不确定 = inconclusive；仅在证据确实且解释唯一时 upheld。`,
          { label: `r${round}:verify:${t.c.id}:${t.lens}`, phase: 'Verify', schema: VERIFY_SCHEMA }
        )
      )
    )).filter(Boolean)

    const byClaim = new Map()
    for (const v of verdicts) {
      const arr = byClaim.get(v.claim_id) || []
      arr.push(v)
      byClaim.set(v.claim_id, arr)
    }
    for (const c of claims) {
      const vs = byClaim.get(c.id)
      if (!vs) continue
      if (vs.some((v) => v.verdict === 'refuted')) {
        c.verdict = 'refuted'
        state.rejected.push({ ...c, reason: vs.map((v) => v.strongest_objection || v.notes).join(' | ').slice(0, 500) })
      } else if (vs.some((v) => v.verdict === 'weakened')) {
        c.verdict = 'weakened'
        state.confirmed.push({ ...c, caveats: (c.caveats || '') + ' [verifier weakened: ' + vs.map((v) => v.notes).join(' | ').slice(0, 300) + ']' })
      } else if (vs.some((v) => v.verdict === 'upheld')) {
        c.verdict = 'upheld'
        state.confirmed.push(c)
      } else {
        c.verdict = 'inconclusive'
        state.gaps.push(`未决：${c.statement}`)
      }
    }
    log(`Round ${round}: 验证后 confirmed=${state.confirmed.length} rejected=${state.rejected.length}`)
  }

  phase('Gate')
  gate = await agent(gatePrompt(state, round), {
    label: `r${round}:gate`,
    phase: 'Gate',
    schema: GATE_SCHEMA,
    effort: 'high',
  })
  if (!gate) {
    state.gaps = state.gaps.concat(['gate agent 失败，需要下一轮重新判决'])
    continue
  }
  state.gaps = Array.from(new Set([...(state.gaps || []), ...(gate.open_gaps || [])]))

  if (gate.actionable && gate.actionable.length > 0) {
    phase('Gate')
    log(`Round ${round}: gate 给出 ${gate.actionable.length} 个候选行动项，进入 kill committee`)
    const kills = (await runBatch(
      gate.actionable.map((a) => () =>
        agent(killPrompt(a), { label: `r${round}:kill:${a.claim_id}`, phase: 'Gate', schema: KILL_SCHEMA })
      )
    )).filter(Boolean)
    const killIds = new Set(kills.filter((k) => k.kill).map((k) => k.claim_id))
    const survivors = gate.actionable.filter((a) => !killIds.has(a.claim_id))
    log(`Round ${round}: kill committee 后幸存 ${survivors.length}/${gate.actionable.length}`)
    gate.actionable = survivors
    if (survivors.length > 0) {
      break
    }
    state.gaps = state.gaps.concat(kills.filter((k) => k.kill).map((k) => `被 kill：${k.claim_id}（${k.strongest_objection}）`))
  }
  if (round < MAX_ROUNDS) log(`Round ${round}: 无可行动项，进入下一轮追踪缺口`)
}

const launched = []
let finalCheck = null

if (gate && gate.actionable && gate.actionable.length > 0) {
  phase('Design')
  const specs = (await runBatch(
    gate.actionable.slice(0, 4).map((a) =>
      () => agent(designPrompt(a, state), { label: `design:${a.claim_id}`, phase: 'Design', schema: SPEC_SCHEMA, effort: 'high' })
    )
  )).filter(Boolean)

  let reviews = (await runBatch(
    specs.map((s) => () => agent(reviewSpecPrompt(s), { label: `review:${s.spec_id}`, phase: 'Design', schema: SPEC_REVIEW_SCHEMA }))
  )).filter(Boolean)
  const reviewOf = new Map(reviews.map((r) => [r.spec_id, r]))

  const finalSpecs = []
  for (const s of specs) {
    const r = reviewOf.get(s.spec_id)
    if (!r) continue
    if (r.verdict === 'kill') {
      log(`spec ${s.spec_id} 被审阅者杀死：${(r.fatal_flaws || []).join('；')}`)
      continue
    }
    if (r.verdict === 'fix' && (r.required_fixes || []).length) {
      const fixed = await agent(
        `${SHARED}
你是实验规格修复者。原始规格：
${JSON.stringify(s, null, 2)}
审阅者要求的修复：
${(r.required_fixes || []).map((f, i) => `${i + 1}. ${f}`).join('\n')}
致命问题：${(r.fatal_flaws || []).join('；')}
请输出修复后的完整规格（同样字段），确保命令真实可执行、端点可判读。`,
        { label: `fix:${s.spec_id}`, phase: 'Design', schema: SPEC_SCHEMA, effort: 'high' }
      )
      if (fixed) finalSpecs.push(fixed)
    } else {
      finalSpecs.push(s)
    }
  }

  if (finalSpecs.length > 0) {
    phase('Launch')
    // 串行启动：让每个启动器都能看到全局已占用的进程数，硬性保证 ≤6 并发。
    for (const s of finalSpecs.slice(0, 6)) {
      const running = launched
        .flatMap((l) => (l.processes || []).filter((p) => p.alive))
        .map((p) => `${p.name} pid=${p.pid} dir=${p.run_dir}`)
      const res = await agent(launchPrompt(s, running), {
        label: `launch:${s.spec_id}`,
        phase: 'Launch',
        schema: LAUNCH_SCHEMA,
      })
      if (res) launched.push(res)
      const aliveNow = launched.flatMap((l) => (l.processes || []).filter((p) => p.alive)).length
      log(`launch ${s.spec_id}: status=${res ? res.status : 'null'}；工作流累计存活进程 ${aliveNow}/6`)
    }

    if (launched.length > 0) {
      phase('Launch')
      finalCheck = await agent(finalCheckPrompt(launched), {
        label: 'launch:verify',
        phase: 'Launch',
        schema: FINAL_CHECK_SCHEMA,
      })
    }
  }
}

return {
  question: '为什么系统里最强的 AI 在人类对局中仍显弱',
  rounds: state.round,
  diagnosis: gate ? gate.diagnosis : '（gate 未产出）',
  actionable: gate ? gate.actionable : [],
  rejected: state.rejected.map((r) => ({ statement: r.statement, reason: r.reason })),
  confirmed_claims: state.confirmed.map((c) => ({
    id: c.id,
    verdict: c.verdict,
    statement: c.statement,
    mechanism: c.mechanism,
    evidence: c.evidence,
    discriminating_test: c.discriminating_test,
  })),
  open_gaps: state.gaps.slice(0, 20),
  experiments_launched: launched,
  final_check: finalCheck,
  tensorboard: 'http://127.0.0.1:6006/ (logdir runs)；另有既有实例 http://127.0.0.1:6007/ (仅 w2m 三臂)',
  budget_output_tokens: Math.round(budget.spent()),
}

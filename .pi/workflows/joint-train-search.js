export const meta = {
  name: 'joint-train-search',
  description: '研究「训练+搜索」联合最优方案、分数 scale-up、搜索/训练解耦与截断蒸馏 bootstrap，并预算化启动实验（每臂≤500k 步，≤16 进程，≤6 GPU，TensorBoard）',
  whenToUse: '当用户要求多子代理研究联合训练+搜索方案并按预算/并发约束启动实验时',
  phases: [
    { title: 'Research', detail: '7 个角度取证（只读）' },
    { title: 'Verify', detail: '对每条 claim 对抗证伪' },
    { title: 'Gate', detail: '门槛判决 + kill committee' },
    { title: 'Design', detail: '预注册实验规格 + 对抗审阅' },
    { title: 'Launch', detail: '串行启动（≤16 进程 / ≤6 GPU）并核验' },
  ],
}

const REPO = '/home/amas/.local/src/7g523'
const CONCURRENCY = 6
const MAX_PROCS = 16
const MAX_GPU = 6
const ROUND_CAP = 700_000
const MAX_ROUNDS = 2

const SHARED = `
项目：${REPO} 是「7鬼523」纯规则引擎 + Gym RL（2 家、54 张、revision-3 出空即撬底、rules_id=2e36dbea44893696、
obs v5 161 维、动作 MultiDiscrete([134,4])、纯 MLP arch=shared hidden=128 relu ≈55,179 参数、PPO）。

## 已确认事实（都有原始产物；研究/验证必须以这些为地基，不得重复已否决方向）
- 冠军 P0 = runs/w2m_ctl__11__1790516900/agent.pt（旧表 μ≈191）；价值 ckpt runs/ei/value/critic_B.pt
  （policy 与 P0 逐位相同；critic 在 23,575 条 search==raw 行上训练，val EV 0.844、噪声上限 0.949）。
- 推理期搜索（O4-lite，runs/o4lite-search/）：决定化隐藏状态 K 采样 + top-6 模板候选 + 双方 frozen champion
  终局 rollout；全量 K=8 vs raw +38.0（bank 20-22）/ +42.6（bank 30-32）；K=4 −3.6；K=16 +83.7（bank 20-22）；
  C=10−C=6 仅 +5.1（n.s.）。
- value 截断搜索（runs/o4lite-search/rollout_trunc.py，t 步后用 critic 读出）：t=5 vs 全量 +67.9 直接对决、
  +34.8 经 raw 相减（~2.75σ 非传递性，开放）；t=5 vs raw +77.5 [64.3,90.6]（bank 30-32）；19–29 ms/决策
  vs 全量 90–100（fast 引擎）；t=10 vs 全量 +31。t≥max_ply 时与历史全量 games.jsonl 逐字节一致。
- 同 View 两次独立搜索的选择翻转率（仅搜索决策）：全量 K=8 51.7%；截断 t=5 37.8%；t=10 51.7%。
- EI round 1（离线蒸馏全量搜索标签）失败：硬 CE −52.8 Elo、soft-Q −45.6、剂量-反应「越拟合越差、不拟合=raw」；
  诊断=搜索动作是采样隐藏世界的函数、而学生只见 obs；S(损坏 P1)≈+40 vs raw（强度在搜索算子而非权重）。
- 历史训练侧 lever 全部 null/负：容量 512、batch、奖励 shaping、event/seq 编码、pool/PFSP、更长预算、双塔；
  warm-start +11.48 未复现（对唯一池外参照 +7.25、CI 跨 0）。
- 评测协议（冻结）：≥3 seed × 400 deal-twin 换座 h2h；判定 = z 与 t 95% CI 同时排除 0 且点估计 ≥ +10；
  tools/head_to_head.py；搜索臂 runs/o4lite-search/run_seeds.sh + combine.py；同 bank 跨臂配对
  runs/o4lite-search/paired_compare.py。

## 用户四问（研究必须回答）
Q1 当前最优的「训练+搜索」联合方案是什么？给出结构化候选（专家迭代/搜索作为行为策略/价值自举/联合微调/
   搜索作为对手…），每个候选映射到本仓库证据与缺口。
Q2 如何 scale up 模型分数？哪些杠杆（学生容量/观测/迭代轮次/教师质量/value 质量/K 与 t/对手模型/预算）
   预期有效，效应量与成本如何。
Q3 搜索配置是否与训练配置解耦？哪些部分可换策略不动（K/C/t/对手模型），哪些必须随策略重标定（value、
   蒸馏目标分布、评分身份），证据是什么。
Q4 截断+蒸馏能否形成 bootstrap 循环（学生→更强策略→更准 value→更强截断教师→学生）？最小可证伪实验、
   预注册端点、以及「循环成立」的判据是什么。

## 硬约束
- 每个训练实验预算 ≤500k 步；零训练/评估实验 ≤半天墙钟。全局实验进程 ≤16，其中占 GPU 的 ≤6
  （RTX 4060 8GB、32 核、15GB RAM）。新实验启动用 .venv/bin/python，不要用 uv run 包两层进程。
- 研究/验证阶段只读：不改 src/tests/docs、不跑训练/h2h、不 git 写操作；允许只读 python/rg/ls 分析。
- 引用落到 file:line 或可复现命令；文档数字以 runs/ 原始产物为准；注明口径（revision-3、obs v5、跨 fit 不可比）。
- 已被否决方向（docs/plans.md §6）不得重新推荐，除非有新的一手反证。
- TensorBoard：http://127.0.0.1:6006（logdir runs/）。
- 输出中文；结论少而硬。
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
          mechanism: { type: 'string', description: '因果链：为什么它提升分数/解释解耦/支撑 bootstrap' },
          evidence: {
            type: 'array',
            items: {
              type: 'object',
              required: ['path', 'what_it_shows'],
              properties: {
                path: { type: 'string' },
                locator: { type: 'string' },
                what_it_shows: { type: 'string' },
                verified: { type: 'boolean' },
              },
            },
          },
          strength: { type: 'string', enum: ['strong', 'moderate', 'weak'] },
          actionability: { type: 'string', enum: ['high', 'medium', 'low'] },
          discriminating_test: { type: 'string' },
          proposed_experiment: { type: 'string' },
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
          target_weakness: { type: 'string', description: '对应四问中的哪个/哪个分数杠杆' },
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
    diagnosis: { type: 'string', description: '整体诊断（中文，可直接给用户看，回答四问）' },
    confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
  },
}

const KILL_SCHEMA = {
  type: 'object',
  required: ['claim_id', 'kill', 'strongest_objection', 'notes'],
  properties: {
    claim_id: { type: 'string' },
    kill: { type: 'boolean' },
    strongest_objection: { type: 'string' },
    could_be_salvaged_by: { type: 'string' },
    notes: { type: 'string' },
  },
}

const SPEC_SCHEMA = {
  type: 'object',
  required: ['spec_id', 'title', 'hypothesis', 'claim_ids', 'arms', 'primary_endpoint', 'kill_criteria', 'tb', 'new_files', 'risks', 'gpu_need', 'total_steps'],
  properties: {
    spec_id: { type: 'string' },
    title: { type: 'string' },
    hypothesis: { type: 'string' },
    claim_ids: { type: 'array', items: { type: 'string' } },
    arms: {
      type: 'array',
      items: {
        type: 'object',
        required: ['name', 'kind', 'gpu', 'commands', 'steps', 'est_minutes'],
        properties: {
          name: { type: 'string' },
          kind: { type: 'string', enum: ['train', 'collect', 'distill', 'eval', 'probe'] },
          gpu: { type: 'boolean', description: '是否占用 GPU（训练臂=true，搜索/采集/评估=cpu=false）' },
          commands: { type: 'array', items: { type: 'string' }, description: '完整可复制命令（仓库根相对）' },
          steps: { type: 'number', description: '训练步数；非训练臂填 0' },
          est_minutes: { type: 'number' },
          run_dir_hint: { type: 'string' },
        },
      },
    },
    primary_endpoint: { type: 'string' },
    kill_criteria: { type: 'string' },
    tb: { type: 'string' },
    new_files: {
      type: 'array',
      items: {
        type: 'object',
        required: ['path', 'purpose'],
        properties: { path: { type: 'string' }, purpose: { type: 'string' } },
      },
    },
    risks: { type: 'string' },
    gpu_need: { type: 'number', description: '峰值同时占用 GPU 的进程数（≤6）' },
    total_steps: { type: 'number', description: '所有训练臂步数之和（每臂 ≤500k）' },
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
        required: ['name', 'pid', 'run_dir', 'command', 'log_file', 'alive', 'gpu', 'evidence'],
        properties: {
          name: { type: 'string' },
          pid: { type: 'number' },
          run_dir: { type: 'string' },
          command: { type: 'string' },
          log_file: { type: 'string' },
          tb_event_file: { type: 'string' },
          metrics_csv: { type: 'string' },
          alive: { type: 'boolean' },
          gpu: { type: 'boolean' },
          evidence: { type: 'string' },
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

const ANGLES = [
  {
    key: 'joint-scheme',
    prompt: `${SHARED}
你的角度：Q1「训练+搜索」联合方案。
读：runs/o4lite-search/{report.md,rollout_policy.py,rollout_trunc.py,trunc_preregistration.md}、
runs/ei/report.md、docs/experiments/structural-directions.md、docs/post-v5-structural-options.md、
docs/plans.md §6、src/seven523/{train.py,ppo.py,policies.py,networks.py,league.py,pool.py}。
任务：
1) 列出 4-8 个结构化候选联合方案（如：offline EI（硬/软标签）、on-policy 迭代 EI、搜索动作作为行为策略 + KL 正则、
   value-first（先价值自举再策略）、搜索作为对手/课程、distill+PPO 混合、search 只做推理不动训练、其它），
   每个给：机制、现有证据支持/反对、缺口、最小可证伪实验、成本（GPU 步数/CPU 时间）、代码落点（runs/ 还是 src/）。
2) 用已确认事实做排除：哪些候选与「EI round1 失败」「训练侧全 null」「截断 t=5 更强」相容/冲突。
3) 给出你判断的当前最优联合方案，并明确它比「纯推理截断 t=5（+77.5 vs raw）」多出的预期收益与不确定性。
4) 5-8 条 claim（含反例检查）与 dead_ends。`,
  },
  {
    key: 'trunc-value',
    prompt: `${SHARED}
你的角度：Q2 的一部分——value/截断的质量与设计空间。
读：runs/ei/value_probe.py、runs/ei/value/report.json、runs/o4lite-search/rollout_trunc.py、
runs/o4lite-search/{trunc_preregistration.md,abs/status.txt,trunc/status.txt}、src/seven523/networks.py（critic 结构）。
任务：
1) 已有 critic_B 在 t=5 下让搜索 +77.5 vs raw，但 t=10 反而只 +31 vs 全量、翻转率也不降——解释为什么截断越短越强，
   并区分「方差降低」vs「偏差/利用」；设计一个便宜的校准探针（在截断点比较 V̂ 与真实终局 rollout 的偏差/方差）。
2) 如何把 value 做得更好：更多数据、distributional/ensemble、t 扫描、用截断自身产生的数据迭代、per-candidate Q/advantage 头、
   更大 hidden、与策略联合训练 vs 冻结主干——每个给预期效应、成本、可证伪端点。
3) 明确 t 的最优点搜索（t=0/1/2/3/5/7/10/∞ 的强度-速度前沿）需要什么实验设计与预算。
4) 5-8 条 claim + dead_ends。`,
  },
  {
    key: 'distill2',
    prompt: `${SHARED}
你的角度：Q4——截断+蒸馏 bootstrap 的最小可证伪设计与判据。
读：runs/ei/{report.md,preregistration.md,distill.py,collect_search.py,value_probe.py,label_stability.py}、
runs/o4lite-search/rollout_trunc.py、runs/ei/value/report.json。
任务：
1) 设计 EI round 2：用截断 t=5 教师（翻转率 37.8%）重采标签 → 蒸馏 → 主端点（学生裸网 vs raw P0，3×400 换座，
   z/t 双 CI + ≥+10）。给出数据量、配方（硬 CE/soft-Q/KL/margin 过滤/advantage 加权）、epoch/lr 与 5 万决策的采集成本。
2) 定义「bootstrap 循环成立」的判据：一轮过后 S(学生) 是否仍 ≥ 阈值、学生裸网增量、value 是否需要重训、
   下一轮教师的预期提升；给出 2-3 轮迭代的停止规则。
3) 变体设计（同一数据零采集成本）：只在改变决策的子集训练、学生输出软目标、冻结主干只训头、KL 锚定 raw 防遗忘。
4) 失败分支的诊断信号（CE/一致率/EV）与「永久关闭」的条件。
5) 5-8 条 claim + dead_ends。`,
  },
  {
    key: 'scale-levers',
    prompt: `${SHARED}
你的角度：Q2——分数 scale-up 的杠杆清单与预期效应量。
读：docs/experiments/README.md 总表与各 null 报告、docs/post-v5-structural-options.md、
docs/experiments/structural-directions.md、src/seven523/{networks.py,train.py,env.py}。
任务：
1) 表格化所有杠杆：学生容量、观测（含信念特征，ADR 级）、迭代轮次、教师质量（K/t/value）、搜索预算（K/C）、
   对手模型/课程、训练预算、奖励/目标、评测协议。每项给：历史证据（null/正/未测）、预期效应区间、成本、风险。
2) 挑出 3-5 个「与搜索联合后可能从 null 变正」的杠杆（例如：搜索标签让容量变大变得有用；教师质量让更多迭代有用；
   value 质量让 t 变小更快），说明机制与最小实验。
3) 对每个建议实验给出 500k 步内的设计与判定，并说明与 P0+截断 t=5（+77.5）的增量比较基准。
4) 5-8 条 claim + dead_ends。`,
  },
  {
    key: 'decoupling',
    prompt: `${SHARED}
你的角度：Q3——搜索配置与训练配置解耦的精确边界。
证据：runs/o4lite-search/report.md §7（w2m_ctl/ws_s2/oppmodel/外部对照）、scale 系列（K/C）、trunc 系列（t/value）、
runs/ei/report.md（蒸馏改变策略后 value 的适用性）、src/seven523/{policies.py,study.py,elo.py}、ADR-0006/0011/0012。
任务：
1) 逐项判定：K、C、t、value 权重、rollout 对手模型、候选排序来源——哪些纯推理可换、哪些必须随策略重标定
   （V^π 随 π 变化；蒸馏目标分布随 π 变化；评分身份/ckpt 语义）。
2) 给出「解耦算子表」：配置项 × 依赖（无/弱/强）× 换策略后的验证协议 × 成本。
3) 设计一个便宜的交叉验证：用 P0 上训练的 critic_B 直接对 ws_s2（或蒸馏学生）做截断搜索，强度损失多少？
   这决定 value 是否可以跨 ckpt 复用（这是 scale-up 的关键成本项）。
4) 对评分/identity 的影响：截断/搜索包装是否应进 manifest/ladder，跨 fit 口径注意什么。
5) 5-8 条 claim + dead_ends。`,
  },
  {
    key: 'eval-intransitivity',
    prompt: `${SHARED}
你的角度：评测与「分数」定义——非传递性、跨 bank 可比性、如何稳健判断 scale-up。
已知：截断 t=5 −全量 = +67.9 直接，但经 raw 相减 = +34.8（~2.75σ 不一致）；K=16 +83.7（bank 20-22）；
abs_t5 +77.5（bank 30-32）。tools/{head_to_head,duel}.py、src/seven523/duel.py。
任务：
1) 用现有 games.jsonl 做三方圆桌的一致性检验（trunc/full/raw 在 bank 30-32）：拟合 BT/PL 模型、
   报拟合残差与 cycle 证据、bootstrap 不确定性；判断 +67.9 与 +34.8 哪个更接近真实部署强度。
2) 给出后续实验的最小稳健评测协议：几张 bank、每张多少 deal、是否加中间锚、如何报三种对比（vs raw、vs full、round-robin）。
3) 定义「模型分数 scale up」的可操作度量（单一数字 + 置信区间 + 非传递性检查），供所有新实验复用。
4) 5-8 条 claim + dead_ends。`,
  },
  {
    key: 'frontier-census',
    prompt: `${SHARED}
你的角度：可执行性普查——把候选方案映射到「现在就能跑的命令与代码缺口」。
读：runs/o4lite-search/{run_seeds.sh,run_h2h.py,run_h2h_trunc.py,rollout_trunc.py,combine.py,paired_compare.py}、
runs/ei/{collect_search.py,distill.py,value_probe.py}、src/seven523/train.py --help 相关 flag、tools/head_to_head.py。
任务：
1) 为 Q1/Q2/Q4 的每个候选实验标注：现有工具能否直接跑（给出完整命令）、缺什么脚本（<200 行 runs/ 可实现）、
   GPU/CPU/进程/显存估算、预计墙钟、TB 写到哪里、预注册端点与 kill 条件。
2) 给出一张「实验 → 命令 → 产物 → 判定」表（8-12 行），并按（证据强度 × 预期增量 ÷ 成本）排序。
3) 明确哪些必须改 src 或触发 ADR（obs 变更、评分链、train.py 目标函数），在本轮预算内应排除。
4) 5-8 条 claim + dead_ends。`,
  },
]

function claimDigest(state) {
  const cut = (s, n) => (s || '').slice(0, n)
  const confirmed = state.confirmed.map(
    (c) => `- [${c.verdict}] ${cut(c.statement, 260)} | 机制: ${cut(c.mechanism, 160)} | 证据: ${(c.evidence || []).slice(0, 3).map((e) => e.path).join(' ; ')}`
  )
  const rejected = state.rejected.map((r) => `- ${cut(r.statement, 180)} | 否决: ${cut(r.reason, 160)}`)
  return `已确认：\n${confirmed.join('\n') || '（无）'}\n\n已否决：\n${rejected.join('\n') || '（无）'}`
}

function gatePrompt(state, round) {
  return `${SHARED}
你是门槛判决官（第 ${round} 轮）。下面是本轮经对抗验证的 claim。判断哪些真正值得现在投入实验。
${claimDigest(state)}
开放缺口：${(state.gaps || []).slice(0, 8).join('；') || '（无）'}

判决标准（全部满足才算 actionable）：
(a) 机制有一手证据，且直接回答四问之一或给出可直接测的分数增量；
(b) 现有工具或 <200 行 runs/ 脚本可测；训练臂 ≤500k 步、评估 ≤半天；
(c) 不被 plans.md §6 或本轮证据覆盖；
(d) 端点分辨率足够（相对比较 ≥+10 Elo 或明确行为判据）；
(e) 成本与并发 ≤16 进程、≤6 GPU；
(f) 默认怀疑：证据弱、机制玄学、与「强度在搜索算子」这一主结论冲突又无证据的一律 rejected/open_gaps。
对每个 actionable 给：对应四问、为什么现在做、最小实验草图（臂/端点/kill）、GPU 需求。
diagnosis 用中文 8-12 句整体回答四问，可直接呈现给用户。`;
}

function killPrompt(a) {
  return `${SHARED}
你是必须杀死这个实验的审查者：
- claim: ${a.statement}
- 理由: ${a.why_now}
- 草图: ${a.experiment_sketch}
翻 docs/experiments/、plans.md §6、runs/ 找证据证明它 (i) 已做过/被否决；(ii) 端点分辨率不足；(iii) 与分数增量无因果；
(iv) 有更便宜判别；(v) 只是重复已知（K/t 扫描、旧 EI 配方、已 null 的训练 lever）。
除非明显过门槛且有决定性，否则 kill=true。给最强反对理由与（若有）最小可挽救版本。`;
}

function designPrompt(a, state) {
  return `${SHARED}
你是实验设计师。把已过门槛的看法变成可直接执行、可证伪的规格：
- claim: ${a.statement}
- 目标: ${a.target_weakness}
- 草图: ${a.experiment_sketch}

环境：GPU RTX 4060 8GB、32 核、15GB；.venv 已装 torch cu132（cuda 可用）。训练入口
.venv/bin/python -m seven523.train …（flags: --exp-name --total-timesteps --load-checkpoint --opponent --seed --tensorboard
--cuda --hidden-size --lr-schedule --optimizer --weight-decay --pool-member …，默认写 runs/<exp>__<seed>__<ts>/{metrics.csv,tb/}）。
h2h：.venv/bin/python tools/head_to_head.py --left A=ckpt:… --right B=ckpt:… --seeds … --pairs 400 --device cpu --workers 4 --json。
搜索臂：bash runs/o4lite-search/run_seeds.sh <arm> 400 <left> <right> <seeds…> [-- extras] + combine.py。
截断臂：O4_TRUNC_PLY=t .venv/bin/python runs/o4lite-search/run_h2h_trunc.py …。
硬约束：
- 每个训练臂 ≤500k 步；评估 ≥3 seed × 400 deal 换座（或行为端点明确判据）；总并发 ≤16 进程、GPU 进程 ≤6。
- 不改 src/tests/docs/ADR；新代码只放 runs/<spec_id>/ 并列入 new_files；不改写 P0/critic_B。
- 每臂 TB 必开；评估/探针把标量写 runs/<spec_id>/tb。
- 给 primary_endpoint、kill_criteria、预计耗时、gpu_need、total_steps；命令必须是完整可复制的单行。
如果 claim 与已有负证据冲突，直接注明并给替代设计。`;
}

function reviewSpecPrompt(s) {
  return `${SHARED}
你是实验规格的对抗审阅者。规格：
${JSON.stringify(s, null, 2)}
验证可执行性与分辨率：
- 逐条命令检查 flag/路径是否存在（--help、test -f、ls；只读，不启动训练）；ckpt 路径存在；warm start 兼容 v5；
  500k 步内能否产生可判读信号；kill 条件明确；TB 会写；并发/GPU 上限不会被突破；不依赖 src 修改。
- verdict: run / fix（列 required_fixes）/ kill（列 fatal_flaws）。宁严勿松。`;
}

function launchPrompt(s, running, gpuRunning) {
  return `${SHARED}
你是实验启动器（串行启动，所以并发账目确定）。规格：
${JSON.stringify(s, null, 2)}
已由本工作流启动的进程：${running && running.length ? running.join(' | ') : '无'}
其中 GPU 进程：${gpuRunning || 0}

执行规则（硬约束，必须执行）：
1) 先实测全局账目：
   - 实验进程总数：ps -eo args | grep -cE '[r]un_h2h|[c]ollect_search|[7]g523-train|[h]ead_to_head|[d]istill|[v]alue_probe'
   - GPU 进程：ps -eo args | grep -E '[7]g523-train' | grep -c 'cuda'
   - 显存：nvidia-smi --query-gpu=memory.free --format=csv,noheader
2) 你只能启动到：总实验进程 ≤16、GPU 进程 ≤6。按 arms 顺序启动；余量不足时只启动能启动的臂，
   其余在 notes 明确 skipped；余量=0 → status="blocked"、processes=[]。
3) 训练臂用 .venv/bin/python -m seven523.train …，--total-timesteps ≤500000，--tensorboard True（默认开）；
   CPU 臂用 .venv/bin/python …。绝不改 src/tests/docs；日志与脚本只写 runs/<spec_id>/。
4) 启动：mkdir -p runs/<spec_id>；nohup <cmd> > runs/<spec_id>/<arm>.log 2>&1 & echo $!；sleep 60–90s 核验：
   ps -p、真实 run 目录（ls -dt runs/<exp>__* | head -1）、metrics.csv 行数、tb/events.out.tfevents.*、日志无 Traceback/NaN。
   首次检查太早可再等 60s。
5) 失败读日志尾部报 failed，不重试超过一次。
6) 返回每个进程的 pid/run_dir/命令/log/TB/metrics 最新行作为 evidence。`;
}

function finalCheckPrompt(launches) {
  return `${SHARED}
你是独立启动核验员。逐项独立复核（不要相信自报）：
${JSON.stringify(launches, null, 2)}
对每个 run：ps -p <pid> 或 pgrep -f <run_dir> 确认存活；wc -l metrics.csv 与 tail -1 确认推进；
ls <run_dir>/tb/events.out.tfevents.* 确认 TB；已结束则检查日志尾部正常完成还是崩溃；
统计当前实验进程总数与 GPU 进程数（应 ≤16 与 ≤6）。
输出逐 run verdict（ok/degraded/dead）+ all_ok + 中文 summary（含绝对路径、当前步数、GPU 计数）。`;
}

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
  const researchRaw = await runBatch(
    ANGLES.map((p) => () =>
      agent(p.prompt, { label: `r${round}:research:${p.key}`, phase: 'Research', schema: RESEARCH_SCHEMA })
    )
  )
  const claims = []
  const seen = new Set()
  let okAngles = 0
  for (let i = 0; i < researchRaw.length; i++) {
    const r = researchRaw[i]
    if (!r) continue
    okAngles += 1
    const key = ANGLES[i] ? ANGLES[i].key : `a${i}`
    ;(r.claims || []).forEach((c, j) => {
      const norm = (c.statement || '').toLowerCase().replace(/\s+/g, '')
      if (!norm || seen.has(norm)) return
      seen.add(norm)
      claims.push({ ...c, id: `${key}-${j + 1}`, angle: key, verdict: 'pending' })
    })
  }
  log(`Round ${round}: ${okAngles} 个角度产出 ${claims.length} 条去重 claims`)

  const rank = (c) => {
    const act = c.actionability === 'high' ? 3 : c.actionability === 'medium' ? 2 : 1
    const str = c.strength === 'strong' ? 3 : c.strength === 'moderate' ? 2 : 1
    const ev = Math.min(3, (c.evidence || []).length)
    return act * 3 + str * 2 + ev
  }
  const selected = claims.slice().sort((a, b) => rank(b) - rank(a)).slice(0, 10)

  if (budget.spent() - r0 > ROUND_CAP) {
    log(`Round ${round}: 研究阶段超每轮预算 ${ROUND_CAP}，跳过验证`)
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
你是对抗验证者，任务：尽量证伪下面 claim。lens=${t.lens}：
- evidence：逐条打开引用，核对原文/数字/路径是否支持；找相反证据；造假或不支持 → refuted/weakened。
- alternative：证据为真时是否有更平凡解释（噪声/已否决方向/已知 null）；找不到才 upheld。
claim：${JSON.stringify(t.c, null, 2)}
默认怀疑；不确定 = inconclusive；仅证据确实且解释唯一才 upheld。`,
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
        state.confirmed.push({ ...c, caveats: (c.caveats || '') + ' [weakened: ' + vs.map((v) => v.notes).join(' | ').slice(0, 260) + ']' })
      } else if (vs.some((v) => v.verdict === 'upheld')) {
        c.verdict = 'upheld'
        state.confirmed.push(c)
      } else {
        c.verdict = 'inconclusive'
        state.gaps.push(`未决：${c.statement}`)
      }
    }
    log(`Round ${round}: verified confirmed=${state.confirmed.length} rejected=${state.rejected.length}`)
  }

  phase('Gate')
  gate = await agent(gatePrompt(state, round), { label: `r${round}:gate`, phase: 'Gate', schema: GATE_SCHEMA, effort: 'high' })
  if (!gate) {
    state.gaps = state.gaps.concat(['gate agent 失败，下一轮重新判决'])
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
    if (survivors.length > 0) break
    state.gaps = state.gaps.concat(kills.filter((k) => k.kill).map((k) => `被 kill：${k.claim_id}（${k.strongest_objection}）`))
  }
  if (round < MAX_ROUNDS) log(`Round ${round}: 无可行动项，进入下一轮追缺口`)
}

const launched = []
let finalCheck = null

if (gate && gate.actionable && gate.actionable.length > 0) {
  phase('Design')
  const specs = (await runBatch(
    gate.actionable.slice(0, 4).map(
      (a) => () => agent(designPrompt(a, state), { label: `design:${a.claim_id}`, phase: 'Design', schema: SPEC_SCHEMA, effort: 'high' })
    )
  )).filter(Boolean)

  const reviews = (await runBatch(
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
输出修复后的完整规格（同字段），确保命令真实可执行、端点可判读、GPU/进程上限不被突破。`,
        { label: `fix:${s.spec_id}`, phase: 'Design', schema: SPEC_SCHEMA, effort: 'high' }
      )
      if (fixed) finalSpecs.push(fixed)
    } else {
      finalSpecs.push(s)
    }
  }

  if (finalSpecs.length > 0) {
    phase('Launch')
    for (const s of finalSpecs.slice(0, MAX_GPU)) {
      const running = launched
        .flatMap((l) => (l.processes || []).filter((p) => p.alive))
        .map((p) => `${p.name} pid=${p.pid} dir=${p.run_dir} gpu=${p.gpu}`)
      const gpuRunning = launched.flatMap((l) => (l.processes || []).filter((p) => p.alive && p.gpu)).length
      const res = await agent(launchPrompt(s, running, gpuRunning), {
        label: `launch:${s.spec_id}`,
        phase: 'Launch',
        schema: LAUNCH_SCHEMA,
      })
      if (res) launched.push(res)
      const aliveNow = launched.flatMap((l) => (l.processes || []).filter((p) => p.alive)).length
      const gpuNow = launched.flatMap((l) => (l.processes || []).filter((p) => p.alive && p.gpu)).length
      log(`launch ${s.spec_id}: status=${res ? res.status : 'null'}；工作流累计存活 ${aliveNow}/16 进程，GPU ${gpuNow}/6`)
    }
    if (launched.length > 0) {
      phase('Launch')
      finalCheck = await agent(finalCheckPrompt(launched), { label: 'launch:verify', phase: 'Launch', schema: FINAL_CHECK_SCHEMA })
    }
  }
}

return {
  question: '训练+搜索联合最优方案 / 分数 scale-up / 搜索-训练解耦 / 截断蒸馏 bootstrap',
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
  open_gaps: state.gaps.slice(0, 24),
  experiments_launched: launched,
  final_check: finalCheck,
  tensorboard: 'http://127.0.0.1:6006 (logdir runs/)',
  budget_output_tokens: Math.round(budget.spent()),
}

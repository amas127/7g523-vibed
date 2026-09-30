export const meta = {
  name: 'adversarial-human-transfer-gap',
  description: '对抗审查 human-transfer-gap.md 的候选原因 C1-C12：每条 1 个子代理（复现+替代解释+统计攻击）（≤12 进程 / ≤4 GPU）',
  whenToUse: '当需要对抗验证真人转移缺口报告的每条候选原因时',
  phases: [
    { title: 'Verify', detail: '每条原因 1 个独立子代理：复现 + 替代解释 + 统计与适用域攻击' },
  ],
}

const REPORT = 'docs/experiments/human-transfer-gap.md'
const LIMITER = '/home/amas/.local/src/7g523/.pi/limits/run_limited.sh'
const PY = '/home/amas/.local/src/7g523/.venv/bin/python'

const VERDICT = {
  type: 'object',
  additionalProperties: false,
  properties: {
    hypothesis_id: { type: 'string' },
    lens: { type: 'string' },
    verdict: { type: 'string', enum: ['confirmed', 'weakened', 'refuted', 'undecidable'] },
    confidence: { type: 'string', enum: ['low', 'medium', 'high'] },
    reproduced: { type: 'boolean' },
    findings: { type: 'array', items: { type: 'string' } },
    counter_evidence: { type: 'array', items: { type: 'string' } },
    corrected_claim: { type: 'string' },
    checks_run: { type: 'array', items: { type: 'string' } },
  },
  required: ['hypothesis_id', 'lens', 'verdict', 'confidence', 'reproduced', 'findings', 'checks_run'],
}

// 每条命题只派一个审查代理；它要在同一轮里完成三种攻击并分轴汇报。
const COMBINED_LENS = [
  '先独立复现，再做替代解释攻击，再做统计与适用域攻击，最后给一个综合裁决：',
  '(a) 独立复现：把该命题依赖的关键数字/事实从原始数据重算一遍，核对报告引用的 file:line 是否真的支持该断言；',
  '(b) 替代解释：假设命题是错的，构造至少一个同样能解释这些证据的替代原因，检查因果链是否必要（相关当因果、混杂当主因），数据允许时做区分检验；',
  '(c) 统计与适用域：样本量、CI、多重比较/挑选效应、口径（h2h 相对分 vs probit 绝对表、胜负 vs 分差）、跨分布外推，以及命题的"可证伪检验"是否真能区分它和竞争解释。',
  '综合裁决：复现失败或命题不成立 → refuted；方向对但过强/口径错/适用域应缩小 → weakened 并给 corrected_claim；三轴都扛住 → confirmed；数据不足 → undecidable。',
].join('\n')

const HYP = [
  {
    id: 'C1',
    title: '28 局自由对战不是强度测量（样本小/未配对/未定级/单人）',
    claim: '用"真人 vs 搜索 15W2D6L、均分 +31.3"推断"搜索的 +110 没有转移到真人"在统计上不成立；这份数据没有能力分辨搜索 0/+50/+110 的转移。',
    evidence: '23 局、单一真人、单一上午；严格胜率 15/23，Wilson 95% CI [0.449,0.812]；均值 +31.3 分 se 9.2；自由对战不进评分链（docs/search-config-plan.md:17-18,128）；搜索局 human 21/23 坐 seat1；5 局无搜索对照对手混杂（random x2、w2m_plain x2、w2m_ctl x1）。',
    attack: '报告 §3.1：复核 Wilson 区间与"若真人=raw 水平则 P(>=15 胜)=0.00145"的计算假设——对手基准胜率 0.672 来自哪 20 seeds x 400 deal 的哪个 bank？配对与否？seat 不平衡会带来多大偏置？',
  },
  {
    id: 'C2',
    title: '真人评分不可识别：production μ=169 由 trace 先验主导，纯结果拟合约 μ=294',
    claim: '"真人只有 μ≈169、却在搜索局赢多"的矛盾只对 production 合并值成立；同一批 7 局纯结果拟合 μ=294±83，人机缺口可能是评分先验造成的假象，而非搜索无效。',
    evidence: 'traces/sessions/session-20260928-092912/report.json：μ=169.02 [121.28,216.77] n=7；trace 通道 mu_traj=160.04 sigma_traj=23.88 权重 0.9239；结果通道 sigma=83.19；重拟结果-only μ=294.0 σ=83.2、冷启动 μ=337.6；src/seven523/placement/estimator.py:26,36,74-100 先验结构。',
    attack: '报告 §3.1：用 Wilson/二项与结果-only 重拟复核；审查"trace 权重 0.92 是问题"是否合理（human-elo 研究把 trace 通道当推荐主力）；n=7 时 μ=294 自身 CI≈[131,457] 是否同样不可识别，从而命题只是"都不可识别"而不是"真人其实约 294"。',
  },
  {
    id: 'C3',
    title: '+110 是 h2h 相对分，不能与 manifest 的 probit μ 相加；搜索从未进 MLE',
    claim: '把 μ(w2m_ctl)=191.5 加 +111/+125 得到"搜索≈μ300"是未知偏置外推；真人定级池只含 raw，真人从未真的和"μ=300 的搜索"比过。',
    evidence: 'src/seven523/duel.py:30,71-74（Elo=400*log10(p/(1-p))）；JS:129,167-170；docs/search-config-plan.md:20-22,26,29,85（跨尺度不可加；μ≈300 是规划外推；搜索 rung 未测）；tools/refit_mle.py 的搜索 rung 测量尚未执行。',
    attack: '审查"+110 不能相加"是否过强：400 点 logistic 与 probit 在小差距下近似成比例，两个尺度间的确定性变换偏置能否真的把 +110 抹掉？并全仓库核查"搜索 rung 未测"是否属实（是否已有 search_t5k16 的 MLE 结果或 manifest 条目）。',
  },
  {
    id: 'C4',
    title: '对手模型自对弈假设：评测对手=rollout 对手=raw champion，真人 OOD',
    claim: '搜索收益是在"对手每一步≈rollout 模型"的条件下测的；真人不是这个模型，+110 中的对手模型误差项从 0 变成未知量。',
    evidence: 'runs/o4lite-search/report.md:74-78、trunc_preregistration.md:6-7（评测右对手与 rollout 对手均为 P0=champion）；481 个真人决策上 champion top-1 一致率 44.5%、top-6 92.5%；rollout 对手换 t18poolself 后 13.2% 选择翻转（searched 17.0%）；迁移臂只覆盖同族网络（report.md:110-112，+41.9/+28.5/+24.5）。',
    attack: '报告 §3.2：top-1 一致率是否被 suit/template 口径影响（同一手牌不同 suit 动作是否被算成不一致）？换对手模型实验是否固定同一 rng stream？已有换对手臂（oppmodel_pself +24.5，between-seed sd 20.84）是否已足以否定"模型误差是关键未知量"？',
  },
  {
    id: 'C5',
    title: 'bank 敏感 + 截断线无 fresh-bank 复核',
    claim: '搜索强度估计对 deal bank 的依赖可达 ~24 Elo，而 +111/+125 只在 seed 30-49 测过、无独立 bank 复核，部署档绝对数字可能带 bank 偏置。',
    evidence: '全量 K8：+52.68 [39.61,65.75]（seed0-2）、+42.64 [30.10,55.17]（30-32）、+28.52 [21.61,35.43]（10-14）；截断类 combined.json 的 seeds 全为 30-49；tfront/trunc_k* 都是 30-32/30-49；剔除 30-32 后 K16/K32 均值只变 <=1.5 Elo（K16 112.6 vs 111.2；K32 125.5 vs 124.7）。',
    attack: '报告 §3.3：逐一核对所有 combined.json 的 seeds 字段；审查"bank 差 24 Elo"是否可能由权重/引擎版本差异解释而非 bank 本身；既然剔除 30-32 后选型稳健，命题是否只剩"没有独立 bank 复核"这一点。',
  },
  {
    id: 'C6',
    title: 'h2h 相对分非传递：截断-全量 直接 +67.9 vs 经 raw 相减 +34.8',
    claim: '两个口径差 ~2.75σ，说明相对 raw 的 Δ 不是可传递的序数；+124.7 vs raw 不能直接外推成对第三方的绝对强度。',
    evidence: 'JS:129,167-170；同 bank t5 直接 +67.94 [52.88,83.01]；经 raw 相减 +34.8；trunc_vs_full_k16 被中止。',
    attack: '报告 §3.4：复算 trunc_k32/combined.json 与 abs_full/combined.json；审查 2.75σ 是否可能只是一次统计波动；三角一致性缺失是否足以支持"不可传递"的强表述，还是只支持"需谨慎"。',
  },
  {
    id: 'C7',
    title: '搜索目标是期望分差，t=5 value 读出噪声大，并有覆盖确定撬底的案例',
    claim: '搜索最大化隐藏世界平均的期望分差（截断后 value 读出）而非胜负；t=5 读出噪声大，在有立即撬底（保证不输）的终局会做投机选择。',
    evidence: 'runs/o4lite-search/rollout_policy.py:536-568、rollout_trunc.py:140-152（按行动方翻转符号的 mean margin）；479 个真人决策上 value vs 单世界全量 EV=0.081、corr=0.33、MAE=33.7 分；自对局对照 EV=0.092、MAE=35.4；案例 traces/web/run-20260928-091716/g0000__s3548246500__seat1__vsw2m_ctl.json step37：raw 对子6-6 立即撬底（32/32 世界 margin=0），搜索改单牌♣6（value=+0.0816）；该局最终 50-50。',
    attack: '报告 §3.5：重新执行 §4.4 决策勘验，32/32 世界一致是关键——若任一季度 pair 的 margin 非 0 该案例降级；单世界全量真值本身方差 ±100，MAE≈34 是否只是读出对单次抽样的自然差，而非"读出噪声大"的证据；目标函数是分差还是胜负需读训练/环境 reward 源码核实。',
  },
  {
    id: 'C8',
    title: '搜索在真人局面改判更多且重尾（61.9% 改判；24.0% 好 vs 13.8% 差；净 +7.2 分）',
    claim: '搜索对真人做出的额外改判不是"免费强度"：相当比例在单世界口径下劣于 raw，净效应为正但方差大，n=23 下容易被放大或掩盖。',
    evidence: '改判 229/479=47.8%（searched 中 61.9%）vs 基准 K32 34.6%（searched 49.0%）；单世界 champion-vs-champion：好 24.0%、差 13.8%、平 62.2%，最差 -110/-100/-90 分；对照自对局 +16.6%/-12.8%/平 70.6%、净 +1.7 分。',
    attack: '复算改判率与单世界回归分布；审查"重尾"是否只是 champion 单世界方差的正常表现、以及真人 vs 自对局两组的口径是否可比；真世界反事实只能用 champion 代理（报告 §2.9），该命题在不可观测反事实下是否可证伪。',
  },
  {
    id: 'C9',
    title: 'determinization 采样先验不利用对手历史',
    claim: 'hidden world 采样是 unseen 池均匀 + 最新 revealed 归对手，不使用人类出牌序列的信息；真人行为可读时后验世界偏离真实，可能选错。',
    evidence: 'runs/o4lite-search/rollout_policy.py:214-260（rng.permutation 均匀划分；public_replay 只用于 empty_order）、README.md:16、:611 每局独立 rng。',
    attack: '读源码核实该实现描述是否属实；"基准里也均匀采样且迁移臂仍为正"是否已说明影响小；人类 top-6 命中 92.5% 是否削弱"采样误差会改变模板选择"；有无直接测量（均匀采样 vs 后验采样）的选择翻转率。',
  },
  {
    id: 'C10',
    title: '价值截断在真人状态更 OOD —— 数据不支持（倾向否定）',
    claim: 'critic 只在 P0-vs-P0 分布校准；但真人局面 EV=0.081/MAE 33.7 与同口径自对局 EV=0.092/MAE 35.4 无差异，因此"真人状态让 value 更差"不成立。',
    evidence: 'trunc_preregistration.md:21 校准分布声明；human-transfer-gap.md §4.4 探针 n=479 vs n=2026。',
    attack: '对抗方向：尝试推翻这个否定结论——检查两组 EV/MAE 是否同口径同算法；真人样本是否被简单/早停局面拉平；按对局阶段（开/中/终局）、按是否 searched 分层后是否出现 value 在真人状态更差的证据。若能证明更差，该条要从"否定"改判。',
  },
  {
    id: 'C11',
    title: 'top-6 根候选限制不是主因 —— 现有证据不支持（倾向否定）',
    claim: '人类实际动作 92.5% 落在 champion top-6；C=6->C=10 配对只有 +5.14 [-6.46,+17.02]，故 top-6 限制不像是主因。',
    evidence: '真人轨迹重放 top-6 覆盖 92.5%；runs/o4lite-search/scale/paired_c10k8_vs_k8.json。',
    attack: '对抗方向：尝试推翻这个否定结论——C=10 实验是 full rollout/K=8 而非 t=5/K=32；在真人局面抽样统计好棋（全量 mean margin 最优模板）落在 champion top-6 之外的频率，若显著高于基准则该条应改判为有影响/未决。',
  },
  {
    id: 'C12',
    title: '真人跨局适应/可预测（低置信）',
    claim: '真人可能在 46 分钟内学会固定 bot 的模式；raw 基策略确定，搜索只改一半（52.2% 仍是 raw 动作）。',
    evidence: 'raw 重放 100% 确定；最后 9 局 7W1L1D vs 前 10 局 6W4L；Spearman rho=0.091 p=0.679、前后 Fisher p=0.370；相同粗状态重复动作一致率 13.7%。',
    attack: '复算趋势检验；审查"46 分钟学习"的间接证据是否有任何一条能站住；用更细的状态分组（或重放前向 logits）检测可预测性后，该低置信命题是否应直接否定。',
  },
]

function buildPrompt(h) {
  return [
    '你是对抗审查子代理（仓库 cwd=/home/amas/.local/src/7g523，7鬼523 中式跑牌游戏 RL 项目）。',
    '主研究报告（由另一个研究子代理撰写，不是权威）：' + REPORT + '。先读它的 TL;DR、§0 相关小节与 §1 中 ' + h.id + ' 的完整条目，以及 §3/§4。',
    '',
    '待审查命题 [' + h.id + '] ' + h.title,
    '命题原文：' + h.claim,
    '报告声称的证据：' + h.evidence,
    '报告 §3 给出的攻击角度：' + h.attack,
    '审查方式：' + COMBINED_LENS,
    '',
    '审查纪律：',
    '1) 只以仓库内原始证据为准（traces/web 与 traces/sessions 的 JSON、runs/ 下结果与日志、源码 file:line）。报告里的数字必须你自己从原始文件复算，不得引用报告结论当证据。',
    '2) 不改仓库任何文件，不 commit；分析脚本放 /tmp/adv/' + h.id + '/。',
    '3) 不跑训练、不跑小时级新 h2h；可做分钟级重放/重算/统计。若必须新跑长实验才能判断，投 undecidable 并说明。',
    '4) 【资源硬约束】机器只有 15GB / 32 核，之前两次并发失控把机器跑死机重启过。所有 bash 命令必须通过槽位限制器执行，一次也不能绕过：',
    '     ' + LIMITER + ' cpu <命令...>   # 纯标准库/文本分析：全机同时最多 12 个此类进程、每个 2 线程 2 核、GPU 被屏蔽',
    '     ' + LIMITER + ' gpu <命令...>   # 任何 import torch / 加载 .pt / 重放或前向：先占 CPU 槽，再占全机 4 个 GPU 槽之一',
    '   注意：sleep/cat/ls 这类瞬时命令也一律用 cpu 包装，保证全机进程数不超过 12。写成脚本后再包装整个脚本一次执行；每个代理同一时刻只能有 1 个未结束的 gpu 进程，不要并发多条重命令。',
    '5) 需要 python 一律用 ' + PY + '（绝不用 uv run，避免 sync/编译风暴；优先用标准库）。只有轨迹重放/前向才 import torch，且务必合并成一个进程一次算完；单条命令预计 >3 分钟就停下并标 undecidable。',
    '5b) 【禁止嵌套】不得使用 Agent / SubagentWorkflow 工具再派生子代理；本次审查只有你自己，全机子代理上限 12 已由主控锁定。',
    '6) verdict 含义：confirmed=你独立复现了关键证据且命题措辞成立；weakened=方向对但过强/口径错/适用域应缩小（必须给 corrected_claim）；refuted=关键证据不能复现或命题不成立；undecidable=现有数据无法判断。默认怀疑：不能独立复现就不得投 confirmed。lens 字段填 combined。',
    '7) findings 与 checks_run 必须写实际执行过的命令/脚本与关键复算数字，并给 file:line 或 trace 文件名+字段。findings 每条一句话，最多 8 条；findings 里要能看出 (a) 复现结果、(b) 最强的替代解释、(c) 统计/适用域结论。',
  ].join('\n')
}

phase('Verify')
log('开始对抗审查（限 12 进程 / 4 GPU，每条 1 个代理）：' + HYP.length + ' 条命题，' + HYP.length + ' 个审查子代理')

const results = await pipeline(
  HYP,
  h => agent(buildPrompt(h), {
    label: h.id,
    phase: 'Verify',
    schema: VERDICT,
    agentType: 'general-purpose',
  })
)

const flat = results.filter(Boolean)
log('完成：' + flat.length + ' 份裁决')
return { report: REPORT, verdicts: flat }

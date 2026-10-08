// Stage-0 RL 瓶颈诊断（D-A / D-B / D-C / D-D），4 个并行 worker。
// 设计来源：docs/post-v5-structural-options.md §4；
// 资源门：.pi/limits/run_limited.sh（≤4 GPU / ≤12 CPU slots）。
// 产物：docs/experiments/stage0-{oracle,fixed-deals,critic-ceiling,variance}.md
//       + runs/stage0/<key>/**（gitignored）。

const REPO = "/home/amas/.local/src/7g523";
const LIMITER = REPO + "/.pi/limits/run_limited.sh";
const PY = REPO + "/.venv/bin/python";

const COMMON = [
  "仓库 cwd=" + REPO + "，HEAD 0fd5c2e（2026-09-29），revision-3（rules_id=2e36dbea44893696）、obs v5、PPO memoryless MLP。",
  "【资源硬约束】机器 15GB RAM / 32 核 / RTX 4060 8GB；全机同时 ≤12 个 CPU 进程、≤4 个 GPU 进程。",
  "  训练/评测/h2h/torch 命令一律 " + LIMITER + " gpu <cmd...>；纯统计/文本用 " + LIMITER + " cpu <cmd...>。",
  "  同一时刻你自己只跑一个 GPU 进程，其余排队；不用 uv run，python 一律 " + PY + "。",
  "【权限边界】只能新增/修改你自己的产物目录与报告（见任务）；禁止改 src/ / tests/ / tools/ / traces/ / .pi/ / 其他 docs / 其他子代理目录；不 commit；不发布 checkpoint/manifest；不改 obs_version。",
  "【统计协议】强度比较走 deal-twin 换座配对 + 按副聚类的配对 bootstrap，报 z-CI 与 between-seed t-CI；行动门槛 = 两类 CI 同时排除 0 且点估计 ≥ +10 Elo，< +5 止损。读 docs/experiments/evaluation-protocol-validation.md 与 tools/head_to_head.py 用法。",
  "【预注册】动手前先把 假设/臂/端点/判据/分析计划 写进报告顶部；中途改设计要写明 amendment。",
  "【报告】中文，结论给 file:line 或命令；结尾必须有『缺失证据/限制』；原始产物放你的 runs/stage0/<key>/（gitignored）。",
  "【超时】接近 timeout 时先保证报告与产物落盘，宁可如实 partial 也不要丢结果。",
  "【阻塞】工具链/资源/口径阻塞用 contact_supervisor（reason=need_decision）上报，不要越界改代码。",
  "【禁止】不派生子代理；不动 docs/experiments/README.md 与 docs/plans.md（父代理统一回填）。",
  "最终回复：报告路径 + 按预注册判据的一句话判定 + 关键数字（不要贴全文）。",
].join("\n");

const D_A_TASK = [
  "你负责 Stage-0 诊断 D-A（oracle-obs 探针），回答：隐藏信息是不是 raw 策略上界的瓶颈（设计见 docs/post-v5-structural-options.md §4 D-A）。",
  "背景：obs v5 的 unseen 段只给『未见面牌的多重集』（54 − 自己手牌 − 已亮 − 已出 − 当前墩），看不到对手手牌与牌堆的划分。",
  "预注册臂（先写进报告再开跑）：",
  "  O = oracle 臂：obs 末尾追加 oracle 块（建议 对手手牌 54 维 multi-hot + 牌堆余牌 54 维 multi-hot；确切定义写进报告）；",
  "  Z = 容量对照臂：同一包装、oracle 块全 0（等参数量、等输入维度，唯一差别是信息）。",
  "两臂都从零训练 500k、训练 seed 1/2/3，同配方同对手（用 train.py 当前默认；把完整命令写进报告）。",
  "实现：只在 runs/stage0/oracle/ 写 run-local driver（monkeypatch/subclass train.make_env 或 Seven523Env），不得改 src/。",
  "  必须做 run-local 一致性测试：用固定动作脚本验证 (a) 两臂环境侧（发牌/对手动作/状态）在同 seed 下逐位一致；(b) oracle 块每一维等于 env.state 的真实隐藏信息；测试输出进报告。",
  "主端点：h2h O（对局时给真实 oracle 输入）vs Z（对局时 oracle 块置 0），deal-twin 换座、按训练 seed 配对；每个训练 seed 至少 3 个 deal seed × 400 副（合计 ≥3×1200 局），报 per-seed 与合并 z/t CI。",
  "判据：合并 CI 排除 0 且点估计 ≥ +10 Elo → 信息是真杠杆（解锁 O1/belief）；CI 跨 0 或 < +10 → 信息不是主瓶颈（注明这是 1–2 run 级功效，只能排除大效应）；< +5 止损。",
  "明确：oracle 策略不可部署（用了非法信息），本探针只测上界；不发布 checkpoint。",
  "报告：docs/experiments/stage0-oracle.md；产物 runs/stage0/oracle/。",
].join("\n");

const D_B_TASK = [
  "你负责 Stage-0 诊断 D-B（固定牌局拟合探针），回答：平台更像 优化/表达限制 还是 泛化/探索限制（设计见 docs/post-v5-structural-options.md §4 D-B）。",
  "预注册臂（先写进报告再开跑）：",
  "  F = 固定集臂：训练只用固定 500 副牌（deal 池固定，对手随机性保留），200k 步，训练 seed 1/2/3；",
  "  G = 对照臂：同配方同预算，正常新鲜发牌，训练 seed 1/2/3。",
  "读数：(1) F 臂在固定 500 副上 vs-random 的胜率/回报训练曲线；(2) F 臂策略在 500 副 held-out 新牌上的表现（泛化 gap）；(3) G 臂同样两组读数；(4) 参考天花板：w2m_ctl（runs/w2m_ctl__11__1790516900/agent.pt）在固定 500 副上的 vs-random 胜率/回报（search_leafq 可选）。",
  "判据：F 在固定集上逼近参考天花板 → 能拟合，若 held-out 明显下滑 → 泛化/探索问题；F 在固定集上仍平台且远低参考 → 优化/表达限制；两种情况都写清实际数字。",
  "实现：只在 runs/stage0/fixed-deals/ 写 run-local 包装（固定 deal 池可用 monkeypatch env reset 或包装发牌），不得改 src/；报告里给固定 500 副与 held-out 的 seed 清单并证明无重叠。",
  "报告：docs/experiments/stage0-fixed-deals.md；产物 runs/stage0/fixed-deals/。若 200k 不足以区分可加跑到 500k，但注明。",
].join("\n");

const D_C_TASK = [
  "你负责 Stage-0 诊断 D-C（critic 天花板探针），回答：value 误差是容量/数据可约的，还是策略与回报定义下的不可约随机性（设计见 docs/post-v5-structural-options.md §4 D-C）。",
  "已知：N-24 已证伪『同 recipe 重采叶子』与 critic 头加宽（宽头 ΔEV ≤0）；本探针测 独立 critic / 更深 / 更多数据 / 不同目标 的 held-out EV 曲线。",
  "做法：选冻结策略（w2m_ctl，可选 lvl4）与数据来源（优先复用 runs/ei2_value_t5/、runs/ei/、traces/study；说明每条数据由哪个策略、哪个 obs 版本产生）；如需新采集走 " + LIMITER + " gpu。",
  "扫（至少）：shipped 结构基线 / 加宽或加深 / 独立 critic（不共享 trunk）/ 数据量 ×1×2×4（可选：MC 终局回报 vs n-step/TD 目标）。",
  "读数：held-out deal 上的 EV、MSE、校准 bias；训练曲线；epoch 选择规则（N-24 的教训：不得靠 epoch 挑选虚高）。",
  "判据：最大 EV 相比同数据基线无实质提升（例如 < +0.03）→ 残差不可约、停止 critic 改进；有实质提升 → value 线仍值得投入，写明下一实验。",
  "报告：docs/experiments/stage0-critic-ceiling.md；产物 runs/stage0/critic/；不发布 checkpoint。",
].join("\n");

const D_D_TASK = [
  "你负责 Stage-0 诊断 D-D（方差 screen），回答：有没有廉价稳定器能把 run 间 seed 方差（现状约 8–14 Elo）砍下来（设计见 docs/post-v5-structural-options.md §4 D-D）。",
  "预注册臂（先写进报告再开跑）：B = 默认配方；T = 默认 + --target-kl 0.03；L = 默认 + --learning-rate 1e-4；各 3 个训练 seed 1/2/3，200k 步。",
  "端点 = run 间 sd（不是均值）：每个 run 的最终标量要低测量噪声（建议固定 deal seed 集、≥1000 局 vs random 或固定对手；写清评估命令）。同时报均值、三条 run 值与 df=2 的 sd 不确定性。",
  "实现注意：--target-kl 在 ppo.py:246 实现（early-stop），必须确认本轮真实生效（对比 approx_kl / update 数日志）；不生效就如实记录。",
  "判据：任一臂 sd ≤ 基线一半（如 ~13 → ≤7）→ 记为候选稳定器并注明需 k≥7 复算才能采用；否则记 D-D 关闭、无廉价稳定器。",
  "报告：docs/experiments/stage0-variance.md；产物 runs/stage0/variance/。",
].join("\n");

const tasks = [
  {
    key: "d-a-oracle",
    label: "Run D-A oracle-obs probe",
    agent: "worker",
    model: "deepseek/deepseek-v4-pro:high",
    timeoutMs: 18000000,
    acceptance: { level: "none", reason: "Stage-0 diagnostic; run-local artifacts + one report, parent synthesizes and reviews" },
    output: REPO + "/docs/experiments/stage0-oracle.md",
    task: D_A_TASK + "\n\n" + COMMON,
  },
  {
    key: "d-b-fixed-deals",
    label: "Run D-B fixed-deal probe",
    agent: "worker",
    model: "deepseek/deepseek-v4-pro:high",
    timeoutMs: 14400000,
    acceptance: { level: "none", reason: "Stage-0 diagnostic; run-local artifacts + one report, parent synthesizes and reviews" },
    output: REPO + "/docs/experiments/stage0-fixed-deals.md",
    task: D_B_TASK + "\n\n" + COMMON,
  },
  {
    key: "d-c-critic",
    label: "Run D-C critic-ceiling probe",
    agent: "worker",
    model: "deepseek/deepseek-v4-pro:high",
    timeoutMs: 10800000,
    acceptance: { level: "none", reason: "Stage-0 diagnostic; run-local artifacts + one report, parent synthesizes and reviews" },
    output: REPO + "/docs/experiments/stage0-critic-ceiling.md",
    task: D_C_TASK + "\n\n" + COMMON,
  },
  {
    key: "d-d-variance",
    label: "Run D-D variance screen",
    agent: "worker",
    model: "deepseek/deepseek-v4-pro:high",
    timeoutMs: 14400000,
    acceptance: { level: "none", reason: "Stage-0 diagnostic; run-local artifacts + one report, parent synthesizes and reviews" },
    output: REPO + "/docs/experiments/stage0-variance.md",
    task: D_D_TASK + "\n\n" + COMMON,
  },
];

const results = await runs.all(tasks);

return results.map((r, i) => ({
  key: tasks[i].key,
  ok: r.ok,
  runId: r.runId,
  outputReference: r.outputReference,
  artifactPaths: r.artifactPaths,
  summary: String(r.output || "").slice(0, 1200),
  error: r.error ? String(r.error).slice(0, 500) : null,
}));

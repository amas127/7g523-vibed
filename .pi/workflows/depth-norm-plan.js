// O2 计划阶段（trunk 深度 + 归一化/残差）：写计划 → 红队审查 → 折叠为 r2。
// 只写文档，不训练、不改 src。owner 审阅 r2 后才决定开工。

const REPO = "/home/amas/.local/src/7g523";
const LIMITER = REPO + "/.pi/limits/run_limited.sh";
const PLAN = REPO + "/docs/depth-normalization-plan.md";
const REVIEW = REPO + "/docs/depth-normalization-review.md";

const COMMON = [
  "仓库 cwd=" + REPO + "，HEAD 0fd5c2e（2026-09-29），revision-3（rules_id=2e36dbea44893696）、obs v5、PPO memoryless MLP。",
  "【权限边界】本阶段只允许写各自的输出文档；禁止改 src/ / tests/ / tools/ / traces/ / .pi/ / 其他 docs；不 commit；不跑训练；最多只读代码核查（需要命令时用 " + LIMITER + " cpu <cmd>，python 用 " + REPO + "/.venv/bin/python，不用 uv run）。",
  "【阻塞】需要决策或发现计划级冲突时用 contact_supervisor（reason=need_decision）上报。",
  "【禁止】不派生子代理。",
  "【语言】中文；所有代码/数据论断给 file:line 或命令。",
].join("\n");

const PLAN_TASK = [
  "你是规划子代理。任务：为结构项 O2（trunk 深度 + 归一化/残差）写一份**预注册式计划**，供 owner 审阅后决定是否开工。只写计划，不做实验。",
  "",
  "必读（按顺序）：docs/post-v5-structural-options.md §3 O2 与 §5；docs/plans.md §1.2（统一验收口径）与 §6 的 N-1/N-2/N-20 与 §3 T8/T14；docs/experiments/README.md §3（判定规则与分辨率表）；docs/experiments/twin-towers-500k.md（架构实验的教训）；docs/experiments/warmstart-adamw-1m.md 附录 A（绝对锚 vs h2h 的分工）；docs/experiments/evaluation-protocol-validation.md §5/§6（功率）；代码 src/seven523/networks.py（trunk/actor/critic 构造）、src/seven523/train.py（--arch/--hidden-size/--activation 与 obs_dim/make_env）、src/seven523/ppo.py（更新循环）、相关 tests。",
  "",
  "背景（写进计划）：宽度 128→256/512 已被证伪、双塔已关闭、激活 tanh/gelu/silu 不赢；当前 trunk 是 2 个 Linear（单隐藏层 128）。今天 Stage-0 的 D-A（oracle 信息无增益）与 D-C（价值残差不可约）把指针移到「优化/表示限制」，O2 是模型复杂度里唯一没测的轴。**不要混入已关闭方向**（宽度、激活、双塔、历史编码）。",
  "",
  "计划必须包含（逐节、可直接给 owner 审）：",
  "1. 假设与决策读法：H1 = 更深/归一化 trunk 在 500k 从零、同配方下比当前 2-Linear 对照 ≥ +10 Elo；H0 = 无实质增益。写清两种结果的后续动作。",
  "2. 臂设计：base（当前 arch=shared、hidden 128、2 Linear，从零对照）＋把「深度」与「归一化/残差」两个因素拆开的臂（如 deep（3–4 层 Linear+relu）、deep_ln、deep_res_ln，或你能论证的更简设计）；同 hidden、同激活 relu、同 500k、全部从零（ckpt 布局改变，禁止 warm start）。给每臂参数量预估。",
  "3. 实现要点（file:line 锚点）：默认路径逐位不变、新结构 opt-in（新 arch 名或新 flag）；ckpt 保存/加载与 obs_version 不变；单测清单（默认路径回归、参数量断言、save/load、NaN 冒烟）。",
  "4. 训练与评估协议：训练 seed ≥3；完整命令写进计划；端点 = 各臂 vs base 的 deal-twin 换座 h2h，≥9 deal seed × 400 副（每对 ≥3600 副）；报 z-CI 与 between-seed t-CI；行动门槛 CI 排 0 且点估计 ≥ +10，< +5 止损。写清功率不足时的预注册扩展规则（如 point ≥+10 但 t-CI 跨 0 → 补 seed 到 k=7；point <+5 → 立即止损）。",
  "5. 机制检查：参数量/深度、训练曲线（EV/entropy/approx_kl）、梯度范数、去掉 LN 的深度消融、容量对照的处理方式（为什么不需要 2 层加宽对照，或给出替代）。",
  "6. 预算与成本：run 数、每 run 墙钟估计、h2h 成本、总 GPU 槽时间；资源一律 .pi/limits/run_limited.sh。",
  "7. 风险与混淆：深度 vs 容量、优化难度、训练 seed sd 8–17 的功效限制、from-scratch 不复用 ckpt、实现风险。",
  "8. 预注册分析产物路径（建议 runs/stage0/depth/ 或新目录）、停止规则、不做清单。",
  "9. 开工前置：owner 批准 + 冒烟通过 + 全测试绿。",
  "",
  "写入 " + PLAN + "（中文，顶部留「状态/修订表」空位，结尾留「红队待审」提示）。",
].join("\n");

const REVIEW_TASK = [
  "你是独立红队审查者，只读。审查 " + PLAN + "（O2：trunk 深度 + 归一化/残差）。不要修改任何文件；你的最终回复会被保存为 " + REVIEW + "。",
  "",
  "背景资料：docs/post-v5-structural-options.md §3 O2/§4/§5；docs/plans.md §1.2/§6（N-1/N-2/N-20）；docs/experiments/README.md §3；docs/experiments/evaluation-protocol-validation.md（功率）；docs/experiments/twin-towers-500k.md；src/seven523/networks.py；src/seven523/train.py。",
  "",
  "攻击面至少覆盖：",
  "(a) 深度/容量/优化混淆是否被控制住（参数量差、梯度动力学、为什么不需要对照）；",
  "(b) 统计功效：3 seed × 400 副对 +10 的分辨率是否够（对照 EPV §5/§6）；预注册扩展/停止规则是否可执行、会不会被 winner's curse；",
  "(c) 与已关闭结论的一致性（宽度负、双塔负、D-A/D-C 指向表示限制）——O2 的先验是否被高估；",
  "(d) 实现风险：默认路径逐位不变、ckpt 兼容、warm-start 拒绝、测试盲区；",
  "(e) 协议完整性：端点/参照/种子台账/分析脚本/停止规则/产物路径是否可复现；",
  "(f) 成本是否被低估；资源与并发是否现实。",
  "",
  "每条 finding 给 severity（fatal/major/minor）、证据（file:line 或计划原文）、具体修法；最后给「按现状能否开工」的一句话判定与「什么会改变该判定」。找不到问题也要明说。",
].join("\n");

const REVISE_TASK = [
  "父代理转来红队审查：docs/depth-normalization-review.md。把它当作对抗意见逐条判断：fatal/major 必须处置（改计划或写明为何不采纳），minor 可记录；把处置表加进计划顶部「修订表」，状态升到 r2，保持所有预注册字段自洽。",
  "仍然只改 " + PLAN + "，不跑训练、不改 src。完成后给出一句话：r2 是否 ready for owner review，以及仍未闭合的分歧。",
].join("\n");

const plan = await runs.run("plan", {
  label: "Draft depth/normalization plan",
  agent: "worker",
  model: "deepseek/deepseek-flash:high",
  timeoutMs: 2700000,
  output: PLAN,
  acceptance: { level: "none", reason: "planning doc; independently reviewed by the next workflow step" },
  task: PLAN_TASK + "\n\n" + COMMON,
});
if (!plan.runId) throw new Error("planner did not return a retained run id");

const review = await runs.run("review", {
  label: "Red-team depth/normalization plan",
  agent: "reviewer",
  model: "deepseek/deepseek-flash:high",
  timeoutMs: 1800000,
  output: REVIEW,
  task: REVIEW_TASK + "\n\n" + COMMON,
});

const revised = await runs.run("revise", {
  label: "Fold review into plan r2",
  resume: plan.runId,
  timeoutMs: 1800000,
  acceptance: { level: "none", reason: "planning doc revision" },
  task: REVISE_TASK,
});

return {
  plan: { ok: plan.ok, runId: plan.runId, outputReference: plan.outputReference, summary: String(plan.output || "").slice(0, 800) },
  review: { ok: review.ok, runId: review.runId, outputReference: review.outputReference, summary: String(review.output || "").slice(0, 1500) },
  revised: { ok: revised.ok, runId: revised.runId, summary: String(revised.output || "").slice(0, 800) },
};

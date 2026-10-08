// O2 计划 r0 的后续：红队审查 → worker 折回 r1。
// 用法：宿主 pi 重启到 1.0.3 后启动本脚本（不再重写 r0，也不依赖已失效的 planner 会话）。

const REPO = "/home/amas/.local/src/7g523";
const PLAN = REPO + "/docs/depth-normalization-plan.md";
const REVIEW = REPO + "/docs/depth-normalization-review.md";
const PY = REPO + "/.venv/bin/python";
const LIMITER = REPO + "/.pi/limits/run_limited.sh";

const COMMON = [
  "仓库 cwd=" + REPO + "，HEAD 0fd5c2e（2026-09-29），revision-3（rules_id=2e36dbea44893696）、obs v5、PPO memoryless MLP。",
  "【权限边界】本阶段只允许写各自的输出文档（reviewer 只读，其回复由运行器落盘；worker 只改 " + PLAN + "）；禁止改 src/ / tests/ / tools/ / traces/ / .pi/ / 其他 docs；不 commit；不跑训练；最多只读核查（命令用 " + LIMITER + " cpu，python 用 " + PY + "，不用 uv run）。",
  "【禁止】不派生子代理。中文；论断给 file:line 或命令。",
].join("\n");

const REVIEW_TASK = [
  "你是独立红队审查者，只读。审查 " + PLAN + "（r0：O2 trunk 深度 + 归一化的 2×2 因子设计）。不要修改任何文件；你的最终回复会被保存为 " + REVIEW + "。",
  "",
  "背景资料：docs/post-v5-structural-options.md §3 O2/§4/§5；docs/plans.md §1.2/§6（N-1/N-2/N-20）；docs/experiments/README.md §3；docs/experiments/evaluation-protocol-validation.md（功率）；docs/experiments/twin-towers-500k.md；docs/experiments/stage0-oracle.md；docs/experiments/stage0-critic-ceiling.md；src/seven523/networks.py；src/seven523/train.py。",
  "",
  "攻击面至少覆盖（计划 §9 的 6 个开放点也要逐条表态）：",
  "(a) 深度/容量/优化混淆是否被控制住（参数量差、梯度动力学、是否需要一个参数量匹配的 2 层对照）；",
  "(b) 统计功效：3 seed × 400 副对 +10 的分辨率是否够（对照 EPV §5/§6）；预注册扩展（补 8 run 到 k=7）是否值得，会不会被 winner's curse；",
  "(c) 与已关闭结论的一致性（宽度负、双塔负、D-A/D-C 指向表示限制）——O2 的先验是否被高估；",
  "(d) 实现风险：默认路径逐位不变、ckpt arch 身份、跨 arch warm-start 语义、测试盲区；",
  "(e) 协议完整性：端点/参照/种子台账/分析脚本/停止规则/产物路径是否可复现；",
  "(f) 成本是否被低估；资源与并发是否现实；LN 放置是否合理。",
  "",
  "每条 finding 给 severity（fatal/major/minor）、证据（file:line 或计划原文）、具体修法；最后给「按现状能否开工」的一句话判定与「什么会改变该判定」。找不到问题也要明说。",
].join("\n");

const REVISE_TASK = [
  "你是计划修订 worker。读取 " + PLAN + "（r0）与红队审查 " + REVIEW + "，逐条判断：fatal/major 必须处置（改计划或写明为何不采纳），minor 可记录；把处置表加进计划顶部「修订表」，状态升到 r1，保持所有预注册字段自洽（臂/参数量/端点/门槛/扩展/deal seed 台账）。",
  "只改 " + PLAN + "，不跑训练、不改 src。完成后给出一句话：r1 是否 ready for owner review，以及仍未闭合的分歧。",
].join("\n");

const review = await runs.run("review", {
  label: "Red-team depth/normalization plan r0",
  agent: "reviewer",
  model: "deepseek/deepseek-flash:high",
  timeoutMs: 1800000,
  output: REVIEW,
  task: REVIEW_TASK + "\n\n" + COMMON,
});

const revised = await runs.run("revise", {
  label: "Fold review into plan r1",
  agent: "worker",
  model: "deepseek/deepseek-flash:high",
  timeoutMs: 1800000,
  output: PLAN,
  acceptance: { level: "none", reason: "planning doc revision; gated by owner review" },
  task: REVISE_TASK + "\n\n" + COMMON,
});

return {
  review: { ok: review.ok, runId: review.runId, outputReference: review.outputReference, summary: String(review.output || "").slice(0, 2000) },
  revised: { ok: revised.ok, runId: revised.runId, outputReference: revised.outputReference, summary: String(revised.output || "").slice(0, 1000) },
};

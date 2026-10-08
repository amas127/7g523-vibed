## Review

审查对象：`docs/depth-normalization-plan.md`（r0，2026-10-06）。只读核查，未修改任何文件；**本运行环境没有 shell 工具**，因此 pytest / python 复算只能给出命令与手算过程，未实际执行（见文末 acceptance-report）。核查过的源文件：`src/seven523/networks.py`（67、256-259、332-355、491-517、647-662、665-707、709-744）、`src/seven523/train.py`（89-208、433-464、581-585、629-649、1070-1084）、`src/seven523/duel.py`（205-350）、`tools/head_to_head.py`（42-120、168-200、300-340）、`tests/test_networks.py`（39-241、287-305）、`.pi/limits/run_limited.sh`，以及背景资料 `post-v5-structural-options.md`、`plans.md` §1.2/§6/§9、`experiments/README.md` §0/§3、`evaluation-protocol-validation.md`（EPV）§5/§6、`stage0-oracle.md`、`stage0-critic-ceiling.md`、`stage0-variance.md`、`stage0-fixed-deals.md`、`rl-bottlenecks-review.md`、`twin-towers-500k.md`。

### Correct（已核对无误）

- **参数量算术全对**。`tests/test_networks.py:236-241` 钉住 base=55,179、trunk=37,248；手算 `ln=+2×256=55,691`、`deep=+16,512=71,691`、`deep_ln=+3×256=72,459`，与计划 §1 表（plan:31-37）逐项一致。
- **训练命令与 `train.py` 默认完全对得上**：`--exp-name/--seed/--total-timesteps/--num-envs 8/--num-steps 128/--num-minibatches 4/--update-epochs 4/--learning-rate 2.5e-4/--anneal-lr/--lr-schedule linear/--optimizer adam/--gamma/--gae-lambda/--norm-adv/--clip-coef/--clip-vloss/--ent-coef/--vf-coef/--max-grad-norm/--hidden-size 128/--activation relu/--opponent random/--reward-shaping terminal/--num-players 2/--torch-deterministic/--tensorboard/--snapshot-interval 0/--cuda/--run-dir` 全部存在且默认值相同（train.py:89-208、435-464）；`--snapshot-interval 0` 默认即 0（train.py:443-450）；499,712 步推导正确（488 updates×1024），与 `stage0-oracle.md` §4 实测一致；目录名 `o2_<arch>__<seed>__<ts>` 与 train.py:581-582 一致。
- **配方确实是 D-A 同款**：与 `stage0-oracle.md` §1.6 的配方表逐条相同（唯一差别 `--arch`），且 D-A 实测 sps 1030–1122（慢时 699）→ 452–715 s/run，计划 §3「8–15 min/run、12 run ≈ 2–3h」是保守且站得住的（plan:78）。
- **预注册骨架基本齐备**：主端点/次要端点、冻结内部参照（base 自身，不依赖已删的旧 ckpt）、训练 seed 与 deal seed 台账、从零约束、只用最终 `agent.pt`（plan:90，符合 `warmstart-adamw-1m.md` §2.1 的 snapshot winner's curse 教训）、描述性比较不许写「值得行动」（plan:93-94）、deal seed 9600-9602 我在 `docs/` 全库检索只有本计划自身出现（plan:96），未与已记录 bank（30-32、410-414、9200/9300/9400、9500）冲突。
- **方案 B 不采是对的**：新增 `--trunk-layers/--trunk-norm` 就得同时扩 ckpt 布局身份与 warm-start 检查，与 ADR-0010 的单一 owner 接缝冲突；方案 A 只动 trunk 分支更小（plan:45-51）。

### Finding

**F1（fatal）k=3 的判定规则无法确认它自己预注册的 +10 假设，且 [5,10) 区间没有处置。**

证据：plan:87 要求「z 与 t CI 同时排除 0 且点估计 ≥ +10」，plan:88 把 k=7 扩展设为条件触发。按 `duel.py:270-287` 的 `_combine_metric`（`se = max(RMS(bootstrap SE), seed sd)/√k`），k=3 时 t 项是 4.303·se，**恒为约束更紧的一侧**：z 只要 1.96·se。用仓库最近一次同结构实验的实测输入（`stage0-oracle.md` §5.2/§5.3）：400 副/seed 的 bootstrap SE 11.19–11.50，每训练 seed（3×400）隐含 SE 6.52/9.93/6.64，跨 seed 合并 se=6.74、t-CI 半宽 **29.0**（原文 [−42.86,+15.11]，点 −13.87）。换成计划 §6 自己写的 sd 8–17（plan:119）：se = max(7.9, sd)/√3 = 4.6–9.8 → **确认门槛实际是点估计 ≥ 19.9–42.2 Elo**，即 +10 的 2–4 倍。后果：
1) 真 +10 时「确认」概率 ≈ Φ(4.303 − 10/se) ≈ **0.2%–1.7%**，确认分支等于死代码（EPV §6 的 0.26–0.31 是**只算 z-CI**的口径）；
2) 止损规则「点 < +5」用点估计判，真 +10 时被误判关闭的概率 = Φ((5−10)/se) = **14%–31%**；
3) 触发 k=7 扩展的概率只有约 17%–59%（要同时满足点≥10 与 z-CI 排 0）；剩下一大块（真 +10 时约 40%、H0 时约 16%）落在 **5 ≤ 点 < 10 的未定义区间**，计划没有任何处置。

最小修法：把 §4 改写成可执行的三段式——(a) 主分析改为**无条件预注册 k=5–7**（即 plan:88 的扩展并入默认预算，20 run ≈ 3.5–5h，见 F7），或把端点改成对 +10 的**单侧等效/非劣检验**（`plans.md` §7 C-1、EPV §6 已明确「禁止用点估计 ≥ +20/+10 当门槛」）；(b) 显式写明 5 ≤ 点 < 10 的处置（建议：不判轴、不改结论、只记录）；(c) 止损改为「CI 排除 +5」或「点 < +5 **且** ≥3 seed 同号」（D-A §6 实际就是这么读的）。

**F2（major）容量混淆没有对照，且 §9.5 的 hidden≈160 数值不对；主端点 `deep_ln − base` 因此不可归因。**

证据：plan:112-113 自认 `deep` 多 +16.5k 参数（+30%，我算 71,691/55,179 = 1.300）；plan:144 只给出「hidden≈160」的粗略对照。实测公式：hidden=h 的 2 层 shared 全网参数 = h²+302h+139（h=128 → 55,179 ✓ 与 test 一致）。解 h²+302h+139 = 71,691（deep）→ h=156.3；= 72,459（deep_ln）→ h=157.4。**h=157 同时把 deep（+0.71%）与 deep_ln（−0.35%）配到 1% 以内**；plan 的 hidden=160 总量 74,059，反过来给对照 **+3.3%/+2.2%** 的容量优势（方向偏保守但不可控）。而「宽度已负」只覆盖 256/512 两个量级，不覆盖 157。

最小修法：加一个**零代码改动**的 5th 臂 `wide` = `--arch shared --hidden-size 157`（3 seed = 3 run ≈ 25 min），并在 §4 写明它只作 descriptive 归因、不进主端点门槛；§9.5 的「hidden≈160」改成 157（若只想配 trunk 而非全网，才是 164，但那样全网会超 deep 约 +6.8%，不建议）。

**F3（major）实现清单不完整；「跨 arch warm-start fail-loud」与现状相反，现状是静默部分复制。**

证据：plan:45 只列 `_ARCHITECTURES`（实为 `networks.py:67`）与 trunk 分支，但 `--arch` 在 train.py 里是**硬编码 choices**：`train.py:201 choices=["shared","towers"]`（`_ARCHITECTURES` 是私有名、不在 `__all__`，train.py 无法 import 它），所以只改 networks.py 时 `--arch ln` 会被 argparse 直接拒掉——计划 §3 的 12 条命令跑不起来。更严重的是 plan:56 的测试项 4：现状**没有** arch 守卫。`warm_start_into`（`networks.py:665-707`）只比对 `obs_dim` 与 `history_layout`，然后按键名+形状复制、缺 key 静默 `continue`；`_mapped_source_key`（`networks.py:647-662`）只认识 `shared↔towers` 桥接，新 arch 名落入 `return key`；`warm_start_from` 仅当 `nvec` 与 `arch` 都相同才整表加载（`networks.py:732-744`）。可复现的后果示例：`shared`→`deep` 静默复制 `network.0/2.*`（第三层留在随机初始化）；`deep_ln`→`deep` 只复制 `network.0.*`，其余（含 3 个 LN 的 γ/β 全部）静默留在初始化——而 train.py:645-648 只打印一行「copied N tensors」，看起来像成功。现有测试只覆盖 shared↔towers 的复制，没有「拒绝」用例（`tests/test_networks.py:101-155`），所以 plan:56 的 fail-loud 测试今天必然失败。

最小修法：§2 加两条明确改动 ——(i) `train.py:201` 增加三个 choices（或把 `_ARCHITECTURES` 导出后复用）；(ii) 在 `warm_start_into` 加 arch 白名单守卫：仅当 `{agent.arch, loaded.arch} ⊆ {"shared","towers"}` 才允许跨 arch 桥接，其余不等即 `raise WarmStartLayoutError`，并补一条「deep_ln→deep 必须 raise」的单测。

**F4（major）§0 的前提（D-A/D-C → 「优化/表示限制」）超出证据分辨率，并漏掉了最相关的两条已完成证据。**

证据：(i) `stage0-oracle.md` §5.3 的 t-CI 上界是 **+15.11，包含计划自己的 +10**——按计划 §4 的口径，D-A 并未排除「+10 的信息增益」；D-A 自己也在 §7.1 写明「1–2 run 级探针，只能排除大正效应」。计划的 §0 只用「−13.87 止损」这一半。(ii) 计划把 `stage0-critic-ceiling.md` 列为来源却漏掉它最接近本实验的臂：`deep_head` = 冻结 trunk + Linear(128,128)+relu+Linear(128,1)，**+16,641 参数、正是「再加一层 128 宽」**，final EV 0.5211 vs base 0.5216（`stage0-critic-ceiling.md` §0.3/§2.2）——这是对 `deep` 最直接的负先验。(iii) 同日完成的 D-D/D-B 也未被引用：`stage0-variance.md`（lr 1e-4 把 run sd 17.21→3.52 但 −18.3 Elo；target-kl 0.03 是 no-op）恰好给出计划 §6/§9.6 所依赖的「稳定器」答案，计划 §9.6 却写「取决于 D-D 的结论」而不给引文；`stage0-fixed-deals.md` 是 null。(iv) 最新综合件 `rl-bottlenecks-review.md` §1/§9.5 明确「继续扫训练配方没有 ROI」「不要重开…容量宽度」，并说 <+10 的真效应要 4 个额外 run 才能确认——计划没有引用它，也没有回答「为什么 O2 优于此清单里的其他项」。

最小修法：§0 把主假设的来源改成引用 `experiments/README.md` §0 第 17 条 / `rl-bottlenecks-review.md` §2；把 D-A 写成「3/3 seed 负、但 CI 上界含 +10」；把 D-C 的 `deep_head` null 写成对 `deep` 的负先验；§6 的「训练 seed sd 8–17」改成 D-D/`rl-bottlenecks-review.md` §5 的实测口径（8.43/11.92/14.4/17.2）；§9.6 直接引 `stage0-variance.md` 并给出结论（无免费稳定器）与处置。

**F5（major）合并口径没有预注册脚本，同一批数据的合法合并方式让 CI 差 2.5×。**

证据：plan:85-86 只说「按仓库规则」，plan:108 只说「合并读数写进报告」。但 `duel.combine_duel_seeds` 要求所有输入的 `left_id/right_id/bootstrap/confidence` 一致（`duel.py:330-346`），而 plan:101-105 的模板把 id 写成带 seed 后缀的 `deep_ln_s1`/`base_s1` → 跨训练 seed 直接调 `combine_duel_seeds` 会 raise；只能手写脚本。而手写脚本有两种都「合法」的合并：按训练 seed（k=3）或把 9 个（训练×deal）seed 当独立（k=9）。D-A 实测两者差异巨大：se 6.74（t 半宽 29.0） vs 4.96（t 半宽 11.4），**2.5×**。

最小修法：在 §4 内嵌一段预注册的复算脚本（像 `post-v5-structural-options.md` §7 那样），写死输入路径与 k=3 训练 seed 口径；同时把 h2h 的 id 改成 `deep_ln`/`base`（不带 `_sN`），这样可以直接喂 `duel.combine_duel_seeds`、留在 C-5 冻结的统计接缝内。

**F6（minor）§4 的 vs-random 饱和 sanity 在给定的命令下取不到数据。**

证据：plan:92-93 要求「`base` 臂 200k/500k 的 vs-random 应复现 ≈0.88–0.90」，但 plan:63-77 的命令没有 `--eval-interval`（默认 0 → 不写任何 eval 行：`train.py:452`、`train.py:1070-1084`），`--snapshot-interval 0` 也没有 200k 快照；`--checkpoint-interval` 默认 100（`train.py:437-441`）只会在同名 `checkpoint.pt` 上反复覆盖（最后剩 409,600 那一次）。也就是说这条护栏现在不可执行。修法：命令加 `--eval-interval 50`（约 10 个 eval 点，同时给 §4「机制检查」提供曲线），或把该 sanity 改成一条独立的 `base vs random` h2h（≈2–3 min）。

**F7（minor）评估成本与审计产物缺失。**

证据：plan:78 的成本只算训练；没有算 12 条 h2h 的 CPU 时间（`v5-optimization-plan.md:618`：1200 deal ≈2–3 min（cpu/4）→ 12 条 ≈25–40 min），也没有算 plan:88 扩展的 8 run + 4 条 h2h 成本；`tools/head_to_head.py:103-112` 提供 `--games-out` 逐副 JSONL（ERA §6.2 的审计惯例），计划未用，报告只能靠 h2h JSON 的 `per_seed` 块复算。修法：§3 成本写成「训练 12 run ≈1.6–3h GPU + 评估 12 条 ≈25–40 min CPU；扩展 +8 run ≈1h + 4 条 ≈10 min」，并给每条 h2h 加 `--games-out runs/o2-depth/games/<pair>_s<seed>.jsonl`。

**F8（minor）LN 放置是未钉死的自由参数，且测试无法区分放置；LN 与 init/head 尺度的耦合没写进 §6。**

证据：plan:31-37 的 +512/+768 全部来自 LN 的 γ/β，因此「激活前 LN」与「激活后 LN」、「末层是否 LN」**参数量完全相同**；plan:54-57 的测试集（参数量、shape/NaN、round-trip）钉不住放置，而放置正是被研究因子的定义本身。另外 `networks.py:348-355` 对 hidden 层用 `layer_init(std=√2)`、critic 头 std=1.0，LN 会把 Linear 的 init 尺度大部分吸收掉（LN 的 γ=1/β=0 默认初始化），所以 LN 臂与 base 臂的**有效初始尺度不同**；plan:117-118 只写了「LN 与 Adam 的交互」这一个通道，而仓库把 V sd / 价值尺度当标准机制读数（`twin-towers-500k.md` §5.3、reward-alignment 系列报告）。修法：§1/§2 里显式预注册放置（保持「每个 Linear 后、激活前」可以，但要写死），加一条结构性断言（断言 `agent.network` 的模块类型/顺序，或对固定输入比对 LN 位置），§4 机制检查补「V sd / 初始 value 尺度 / value loss 量级」。

**F9（minor）引用锚点偏移，臂命名不一致。**

`_ARCHITECTURES` 在 `networks.py:67`（plan:45 写 66）；shared trunk 语句在 `networks.py:348-353`（plan:46 写 349-353）；`--arch` choices 在 `train.py:201`（计划未提）。命名上 §1 叫 `base`、§3/§5 用 `o2_shared`/`base_s1`（plan:101-104），跨比较按 id 合并时容易出错（见 F5）。修法：改锚点；臂名统一为 `base`（`--exp-name o2_base --arch shared`）。

**F10（minor）源清单与 §9.6 的依赖缺口。**

plan:9-13 的来源列表没有 `stage0-variance.md`、`stage0-fixed-deals.md`、`rl-bottlenecks-review.md`，而 §0 的论点（「指针移向优化/表示限制」）与 §9.6（「取决于 D-D 的结论」）恰好依赖它们。修法：补进来源列表 + 在 §0/§9.6 各引一句。

### §9 六个开放点逐条表态

1. **LN 放置 / 是否砍掉 2×2**：保留 2×2——`ln`、`deep` 两个臂只多花 6 run（≈45 min GPU），而缺了它们，即使主端点为正也无法归因（见 F2），不砍。放置建议维持「每个 Linear 后、激活前」但**必须预注册并加结构性断言**（F8）；若一定要选一个替代，选「末层不做 LN」（head 输入分布更接近 base，保守）。
2. **k=3 功效与 k=7 扩展**：见 F1——现状下确认分支不可达、扩展触发率 <60%、[5,10) 无处置、止损误杀 14%–31%。建议**无条件预注册 k=5–7 作为主分析**（成本 F7）；「fresh-seed 复核」不能替代它（统计功效不足）。至于 winner's curse：扩展触发是对 k=3 大幅正偏离的选择，**k=7 的点估计因此系统性偏高约 3 Elo**（T23 实测 +11.48 → +8.20，差 3.28），我粗算这套两阶段规则的 FPR 反而低于 2.5%（stage-1 触发要求大偏离），所以主要风险是「把刚过 +10 的 k=7 读数读成稳健 +10」，报告必须写明这是条件读数。
3. **多重性**：认可「只认主端点、其余 descriptive」，但要加两条——(a) 写死 Holm 或独立 bank 的确切触发条件（不能事后挑）；(b) 明确 `deep_ln − deep` 这类「共用 run」的比较也受同一 descriptive 约束。当前表述（plan:93-94）已够，补一句「4 个比较的 descriptive 结论不得进入 `plans.md` §5/§6 的轴级判定」。
4. **跨 arch warm-start**：**fail-loud**，且要当作本轮的实际代码改动（现状是静默部分复制，见 F3）。理由：本轮禁止热启动，fail-loud 是零成本；静默前缀复制会产生「看起来有效」的半随机初始化 run，是比崩溃更糟的失败模式。
5. **参数量匹配对照**：要，且比计划想的更便宜——`--arch shared --hidden-size 157`，零代码，total 与 deep/deep_ln 差 ±0.7%（plan:144 的 160 是 +3.3%/+2.2% 超配，见 F2）。
6. **配方固定**：坚持「no gain under this recipe」的读法，**不加**默认可变因子的稳定器臂；D-D 已给出可用答案（`stage0-variance.md`：无免费稳定器；lr 1e-4 降 sd 但 −18.3 Elo）。若 `deep` 臂 KL/EV/grad-norm 明显退化，结论写成「该配方下无法判定深度是否有用」并写清重开条件，而不是「表示容量不是瓶颈」（plan:115-116 已经写对了，补齐引文即可）。

### 判定

**按现状不能开工**（作为确认性实验）：判定规则在 k=3 下无法确认它自己预注册的 +10（门槛实际是 +20…+42，确认概率 ≲2%），[5,10) 无处置且止损会误杀真 +10；同时实现清单漏 `train.py:201` 与 warm-start 守卫（现状是静默部分复制），容量混淆没有 cheap 对照。修完 F1/F2/F3（外加 F5 的合并脚本）后可开工，总增量 ≈1 h GPU + 半小时改文档/代码。

**什么会改变该判定**：(i) owner 明确把本轮降级为 screen（不做轴级论断、不写进 §5/§6）+ 无条件预注册 k=7 复核——那么现状即可开工（约 3.5–5h GPU）；(ii) 或把主端点换成对 +10 的单侧等效/非劣检验并补 [5,10) 规则；(iii) 若能证明 k=3 的 se 会显著小于我引用的 4.6–9.8（例如改协议、或提供新的大 k 实测），门槛算术会随之变化——但那等于改评估协议，属 N-17/C-5，不在本轮授权内。

**Merge verdict：BLOCK**（P0=F1；P1=F2/F3/F4/F5；P2=F6–F10）。`No issues found.` 不适用——共 10 条，其中 1 条 fatal。
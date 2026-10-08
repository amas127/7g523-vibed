# 联赛池扩员：加入 5 个前沿强成员（o2c_base / o2c_deep / 三条 w2m）

> **2026-10-08 operator 决策**（固定后续训练 `--opponent pool` 的默认成员集合，供后续训练/实验引用）。
> 历史 run 不回填、不重跑；复现 ADR-0015 的旧成员集合需显式传参（见「决定」）。

## 背景

- [ADR-0015](./0015-continuation-pool-random-mix.md) 的现行池是 6 个 t17 成员（各 `3@`）
  + random（`2@`，`2/20 = 10%`），覆盖的是 500k 从零/早期续训一档。
- 该池建立时，当前最强的一批模型还不存在或未定型。现在（2026-10-08）已有若干**前沿强成员**：
  - `o2c_base` / `o2c_deep`：O2 500k 从零臂按 T23 + ADR-0016 续训 +1M；`o2c_base` 对旧冠军
    `w2m_ctl` = **+10.12 Elo**（k=7，CI 排除 0，[`experiments/depth-normalization-500k.md`](../experiments/depth-normalization-500k.md) §3.2）。
  - `w2m_ctl` / `w2m_plain` / `w2m_low`：w2m 2M 续训三臂，均处顶部平台簇（~186–191 Elo，
    见 [`experiments/t17-recalibration.md`](../experiments/t17-recalibration.md) / `docs/plans.md` T17）。
- 后验探针（`docs/plans.md` N-26 补记）：`o2c_base` 对 `w2m_plain`/`w2m_low`/`ws_s2` 是平局
  （−1~+4 Elo），存在 **~7 Elo 非传递性**。也就是说这 5 个成员不是单调的强→弱阶梯，而是
  顶部平台带内一批**互有胜负、风格/血统各异**的近似同档模型——正好提供 league 需要的多样性。
- operator 2026-10-08 指定：**把这 5 个成员加入池子**（augment，不替换 t17 与 random）。

## 决定

1. 现行池在 ADR-0015 的 6 个 t17 成员之外，**新增 5 个前沿成员**：`o2c_base`、`o2c_deep`、
   `w2m_ctl`、`w2m_plain`、`w2m_low`。t17 成员与 random **保留**（augment，非 replace）。
2. **权重**：11 个 ckpt 成员等权 `9@`，random `11@` → 总权重 `11×9 + 11 = 110`，random 精确
   **10%**，每个 ckpt 成员 **8.18%**。整数权重使比例精确；random 占比与 ADR-0015 一致，
   评分 gauge 与分布覆盖口径不变。

   ```bash
   --opponent pool --pool-episode True \
     --pool-member 9@ckpt:runs/t17pool__1__1790439615/snapshots/checkpoint_step262144.pt \
     --pool-member 9@ckpt:runs/t17pool__1__1790439615/snapshots/checkpoint_step499712.pt \
     --pool-member 9@ckpt:runs/t17long__1__1790439615/snapshots/checkpoint_step512000.pt \
     --pool-member 9@ckpt:runs/t17long__1__1790439615/snapshots/checkpoint_step1024000.pt \
     --pool-member 9@ckpt:runs/t17long__1__1790439615/snapshots/checkpoint_step1536000.pt \
     --pool-member 9@ckpt:runs/t17long__1__1790439615/snapshots/checkpoint_step1999872.pt \
     --pool-member 9@ckpt:runs/o2-depth/cont/o2c_base__1__1791361973/agent.pt \
     --pool-member 9@ckpt:runs/o2-depth/cont/o2c_deep__1__1791361955/agent.pt \
     --pool-member 9@ckpt:runs/w2m_ctl__11__1790516900/agent.pt \
     --pool-member 9@ckpt:runs/w2m_plain__11__1790516900/agent.pt \
     --pool-member 9@ckpt:runs/w2m_low__11__1790516900/agent.pt \
     --pool-member 11@random
   ```

3. **语义不变**：沿用 `--pool-episode True`（`EpisodeMixturePolicy` 每局抽一次、整局冻结，
   `src/seven523/policies.py:106`）；成员动作按 [ADR-0017](./0017-pool-opponents-sample.md)
   默认采样（`--pool-sample True`）。random 占比是**局占比**（`11/110`）。`--mix-random-prob`
   在 pool 模式下仍惰性，禁止用它调比例。
4. **兼容性**：5 个新成员均为 obs v5 / `obs_dim=161` / `nvec=[134,4]` / 2 家，与池布局一致；
   加载按既有 `ckpt:` 路径，无新代码。
5. **范围**：只改后续 pool 配方；ADR-0015 列出的历史 run 保持原成员集合，不回填、不重跑。
   复现 ADR-0015 旧池：按 ADR-0015 §决定传 6 个 t17 `3@` + `2@random`。

## 考虑过的替代

- **替换掉 t17 成员（只留前沿）**：丢历史/中段覆盖，且在全更强成员池上学习者起步弱、
  梯度信噪比差；league 经验要求跨强度多样性 → 否决（采用 augment）。
- **沿用 t17 `3@` + 新成员各 `3@` 且 random `2@`**：random 降到 `2/35 = 5.7%`，稀释
  ADR-0015 有意的 10% gauge/覆盖 → 否决（random 固定 10%）。
- **前沿成员给更高权重（如 `12@`、t17 `5@`，前沿 60%）**：在顶部平台带内这些成员本就近似
  同档、且有 ~7 Elo 非传递性，"前沿 60%"隐含一个不成立的单调整体强度序 → 不采纳（采用等权）。
- **顺带把 `o2c_deep_ln` / `o2c_deep_res` 等也加进来**：本轮 operator 明确只列 5 个 →
  不加；需要时另行决策。

## 后果

- **可比性**：池成员集合/序列与 ADR-0015 不同，**同 seed 也不可逐位对齐**；与历史 run
  （A1–A6/T23/w2m/head-depth）的跨池比较只能按统计口径（h2h / 合并 CI），且新配方对照必须
  **两侧同池**（沿用 `docs/plans.md` §1.2 与 ADR-0015 §4 口径）。新 run 的 `args.json`
  会逐条记录 `--pool-member`，可事后分辨成员集合。
- **先验证据**：T5/`opponent-distribution-500k.md` 的"强成员池"效应为 null
  （`C_s − C_u` = +2.90 [−17.32, +23.12]），但那是更弱的成员、旧 regime、单 seed。本 ADR
  只是把现行默认池扩员为**描述性事实**，**不宣称**会提升强度；是否值得以新池重跑任何网格，
  仍需按 ≥3 seed / CI 排除 0 的门槛另判。
- **PFSP**：现行 `pfsp=False`（静态权重）；若启用 PFSP，random 是否参与重加权需另行决定，
  本 ADR 不覆盖。
- **风险**：`o2c_*`/`w2m_*` 与后续 learner 可能血统相近，固定池续训仍可能有熵塌缩/过特化倾向
  （ADR-0015 背景）；random 10% 是对冲，不是根治。
- 文档面：`docs/training.md` 的续训配方段落同步；`docs/plans.md` §0 ADR 登记扩到 `0018`。

# 默认训练配方：arcsin 奖励成形 + 50 分边界跳变 + 学习率下界 1e-5

> **2026-10-07 operator 决策**（本 ADR 固定 `7g523-train` 的默认配置，供后续训练/实验引用）。
> 历史 run（A1–A6、T23、w2m 等）不回填、不重跑；复现旧配方需显式传参（见「决定」）。

## 背景

- 2026-10-07 operator 指定两处改动：
  1. **学习率退火下界 = 1e-5**：`--anneal-lr` 下不让尾部学习率衰减到 ~0（此前线性/余弦都退到 ≈0）。
  2. **奖励函数改为"反三角函数形状"**：终局分数越接近 0/100，斜率越大；并在 **50 分处引入跳变**。
- 实现已就位并测试（`src/seven523/env.py` 的 `arcsin` 模式、`src/seven523/train.py` 的
  `annealed_lr`）：半强度 arcsin（α=0.5）+ 50 分胜/负边界跳变（`win_jump=1.0`）。
- 目标：把这些设置写进**训练 CLI 默认值**，避免每次运行手工传 4 个 flag（历史上 pool/mix
  的 flag 误读说明手工传参容易错）。

## 决定

`parse_args`（即 `uv run --group train 7g523-train`）的默认值：

| 参数 | 旧默认 | 新默认 | 语义 |
|---|---|---|---|
| `--reward-shaping` | `terminal` | **`arcsin`** | 终局 `r = (1−α)·margin + α·(2/π)·arcsin(clip(margin,−1,1)) + λ·seat_outcome` |
| `--arcsin-mix` | （新） | **0.5** | 混合权重 α：0=线性（= `terminal_win`）、1=全强度 arcsin |
| `--win-jump` | 1.0 | **1.0**（不变） | 50 分胜/负边界跳变 λ |
| `--lr-floor` | （新） | **1e-5** | `--anneal-lr` 下学习率绝对下界（不高于峰值）；`--lr-floor 0` 恢复旧行为 |

- `margin = own/100 − mean(others)/100`（2 家 = `own/50 − 1`）；`seat_outcome ∈ {−1,0,+1}`；
  平局保持在 0，跳变发生在 50 分两侧极限之间（49 分 = −1.016、51 分 = +1.016，λ=1）。
- **范围**：训练 CLI（`parse_args`）与库默认（`Seven523Env`、`make_env`）**全部对齐**到新配方
  （2026-10-07 追加对齐：初版只改 CLI、库默认保持 `terminal`，造成两层默认不一致；现统一）。
- **历史复现**：旧配方必须显式 `--reward-shaping terminal --lr-floor 0`（两者一起才是逐位旧行为）。
- **纪律**：新配方参与的对照两侧同配方；与历史批的直接数字不可比（奖励尺度不同：半强度
  arcsin + λ=1 的终局回报范围是 [−2, 2]，旧 `terminal` 是 [−1, 1]）。

## 考虑过的替代

- **只写进文档、不改 argparse 默认**：每次训练要手工传 `--reward-shaping/--arcsin-mix/
  --win-jump/--lr-floor` 四个 flag，漏传就静默跑旧配方或半新配方——不采纳。
- **全强度 arcsin（α=1）**：端点导数发散，0→1 分步长 0.128（线性的 6.4×），operator 认为过陡；
  半强度 α=0.5 为 3.7× → 采用 0.5，α 仍可调。
- **改库默认（`Seven523Env`/`make_env`）为 arcsin**：初版担心影响面超出训练配置而保留旧默认；
  实际使用点只有训练 `make_env`（已显式传参）与测试/直接构造，operator 要求全默认一致 →
  已采纳（测试改为显式指定被测模式）。
- **`--lr-floor` 做成 recipe-only flag（默认 0）**：operator 要求「修改成 1e-5」，那它就是新默认 → 采纳为默认。

## 后果

- 所有不显式传参的训练 run 从此使用 arcsin（α=0.5）+ 50 分跳变（λ=1）+ 学习率下界 1e-5；
  `runs/*/args.json` 会记录这四个字段，可事后分辨配方。
- 与 2026-10-07 之前的所有训练数字不可直接比（奖励尺度与尾部 LR 都变了）；评估/H2H/统计管线不变。
- PFSP 等读 env 提供的 `info["outcome"]`，不依赖奖励符号；`win` 之外的模式仍走
  「缺 outcome 就报错」的路径（保持现状）。
- `docs/training.md` 的两处默认说明、`tests/test_train.py` 的默认断言同步；
  `plans.md` §0 ADR 登记扩到 `0016`。

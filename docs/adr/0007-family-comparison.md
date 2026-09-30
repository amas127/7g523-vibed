# 比较模型：非炸弹只在牌型族内比较，炸弹跨族

> **决策仍现行，回放兼容条款已退役**：牌型族比较是当前唯一语义（`combos.beats`，
> RULES.md §3.1/§3.2）；本文“旧 trace 回放”一节保留的 `Rules.comparison = "tier"`
> 变体已由 [ADR-0009](./0009-single-observation-and-comparison.md) 退役（`Rules` 已无该
> 字段；读取端容忍旧 `comparison` 键，但一律按牌型族回放）。

旧比较规则是一条**平面 tier**（单牌/对子 tier 1 < 顺子/连对 tier 2），于是越族牌型落在相邻层级时，同一条 `(size, top_key)` 判据会接受越族应牌：**顺子压对子、连对压单牌**。这与“同族才能比”的规则直觉不符，用户报告后确认为 bug。

决定：非炸弹牌型只在**牌型族**内比较——

- **单张族**：`SINGLE < STRAIGHT`；
- **对子族**：`PAIR < CONSECUTIVE_PAIRS`；
- **跨族互不可压**：顺子压不过对子、对子压不过单牌、连对压不过单牌、顺子压不过连对；
- **炸弹压所有非炸弹**：`小炸弹 < 大炸弹`；王炸是 `SMALL_BOMB`，小炸弹之间只比点数、不比张数（与 RULES.md §3.4 一致）。

同族内仍先比数量（顺子张数／连对对数），数量相同再比“最大一张牌”（点数序，同点数比花色）。实现在 `src/seven523/combos.py`：`FAMILY` 给每个非炸弹牌型归族，`beats` 对非炸弹先要求同族、再比 `(size, top_key)`；`TIER` 保留为炸弹层级（小 < 大）与候选出牌排序用的稳定全序，`beats` 只对炸弹读它。对照 RULES.md §3.1/§3.2。

## 旧 trace 回放

2026-09-25 之前落盘的 trace 是按旧平面 tier 判定的，直接按新规则回放会在“当时合法、现在不合法”的应牌处校验失败。决定：

- 保留 `Rules.comparison` 变体：新对局默认 `"family"`；`"tier"` 是旧平面比较，**仅用于回放 2026-09-25 之前的 trace**；
- trace 版本 `TRACE_VERSION` 升到 2，并在 `rules` 段记录 `comparison`；读取缺该键的旧记录时按 `"tier"` 解释。

## 考虑过的替代

- **保留平面 tier，只调层级**：无论顺子/连对放 tier 1 还是 tier 2，跨族应牌判据仍在，bug 不消失。
- **只按数量比较（size 优先）**：连对（6 张）会压顺子（3 张）等新的越族应牌，且同族语义被张数掩盖。
- **要求同族且同牌型**：会禁掉同族内 `PAIR < CONSECUTIVE_PAIRS`、`SINGLE < STRAIGHT` 的大小关系，过紧。
- **改造并重写旧 trace**：改历史产物，且无法恢复当时的合法性判定；保留 `comparison` 变体的成本更低。
- **不升版本、靠字段有无猜测**：新旧记录只差 `comparison` 一个键，显式版本加显式字段比隐式猜测可靠，也避免未来变体再猜。

## 后果

- 规则变更前产生的**所有数字**——`docs/experiments/` 的 Elo/胜率/训练曲线、`traces/study` 的 manifest 契约值与 arena 排名、human-elo 的轨迹先验——都是旧 `tier` 口径，**不可与新规则结果混比**；`docs/experiments/README.md` 顶部与 `docs/human-play.md` 对手表已加口径警告。
- 后续必须重标定：重新生成 `traces/study`、重测 manifest、重训 ladder，并重标定 human-elo 先验与 `tools/play_ladder.py` 的强度列（计划见 `docs/plans.md` T15）。
- 观测、动作空间与 checkpoint 结构不变；但旧 ckpt 是在旧规则下训练的，新规则下的强度必须重测，不能把旧 Elo 直接沿用。
- `"tier"` 变体是回放专用遗留路径，新代码不得用它训练或评估。

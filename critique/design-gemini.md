# Design A（gemini）：棋类/AlphaZero 风格的「无漂移」方案（被审原文）

> **历史外部提案（原文粘贴，2026-09-25/26 审查）**：本文不是仓库设计，其
> `batch_recalibrate`（无限循环且非 MLE）、SPRT 门禁、原生 25 标度与
> `conservative_rating` 等均未被采纳，逐条裁决见 [`report.md`](./report.md)；
> 仓库现行方案是 [`ADR-0013`](../docs/adr/0013-drift-free-rating-channel.md)。

在神经网络棋牌（如 AlphaZero、Leela Chess Zero、Gomoku 引擎）的迭代升级过程中，传统贝叶斯等级分（如 TrueSkill / Openskill）若直接用于自对弈（Self-play）或增量评估，会导致严重的等级分漂移（Rating Drift）和通货膨胀/紧缩。

漂移的本质原因是：缺乏固定坐标系（随着整体策略水平提升，旧模型的相对表现改变）、路径依赖性（在线更新顺序影响最终分值）以及策略循环/非传递性（A 克制 B，B 克制 C，C 克制 A）。

要实现一套无漂移的 Openskill 评估系统，需要从锚点约束、全历史重校准、SPRT 门禁机制以及非传递性检测四个维度进行架构设计。

1. 无漂移 Openskill 系统核心机制

+-----------------------------+ | 新 Checkpoint (Candidate) | +--------------+--------------+ | v [ 1. SPRT 门禁对弈测试 ] | (通过) v [ 2. 锚点池 (Anchors) 配对 ] | v [ 3. 带锚点冻结的 Openskill 增量评分更新 ] | v [ 4. 全历史日志定期重估计 (Batch WHR Refit) ] 

机制一：静态与动态锚点池（Anchor Pool）

无漂移系统的基石是不参与积分更新的参照物（Anchors）：

静态规则/传统引擎锚点（Static Anchors）：引入行为确定或能力恒定的基线（例如固定的 Stockfish 节点数、MCTS 算力固定的规则引擎），赋予其硬性固定的评分 (\mu_0, \sigma_0 = 0)。

里程碑 Checkpoint 锚点（Golden Checkpoints）：在神经网络迭代过程中，每隔 N 个代际冻结一个代表性模型，作为后续评估的绝对坐标。

锚点冻结更新（Anchor Freezing）：在 Openskill 计算 rate([[Candidate], [Anchor]]) 后，只更新 Candidate 的 (\mu, \sigma)，强行丢弃 Anchor 的更新结果。

机制二：防坍缩与方差（\sigma）动态重置

Openskill 依靠 Gaussian 分布 N(\mu, \sigma^2) 表示评估置信度。若模型对弈次数过多，\sigma 会无限趋近于 0，导致评分“僵化”，无法响应新模型的真实能力。

新 Checkpoint 继承策略：新生成的网络入池时，其 \mu 初始设为当前最新 Best Checkpoint 的 \mu，但 \sigma 必须重置为较大的初始 uncertainty（如 \sigma_0 = \frac{\mu_0}{3}）。

下限衰减机制：设置 \sigma_{\min}，防止老模型的方差过小而拒绝吸收新对局信息。

机制三：批处理全历史校准（Whole-History Refit）

增量式（Sequential）Bayesian 更新天生具有路径依赖误差。为了彻底消除漂移，增量更新仅用于实时匹配和快速决策；后台定期（如每 24 小时或每 50 代迭代）使用全历史博弈日志（Full Match History Log）重新拟合一次所有 Checkpoint 的评分。

通过在优化目标中加入锚点的 L2 正则约束或硬性约束，全局极大似然求解： 

\hat{\boldsymbol{\mu}} = \arg\max_{\boldsymbol{\mu}} \sum_{k} \log P(\text{Outcome}_k \mid \boldsymbol{\mu}) \quad \text{s.t.} \quad \mu_{\text{anchor}_i} = C_i 

2. 完整 Python 实现方案

以下基于开源 openskill 库，实现支持锚点冻结、保守评分计算、对局调度与防漂移更新的完整评估框架。

import math from typing import Dict, List, Tuple, Optional from openskill.models import PlackettLuce class EvaluatorNode: """神经网络模型节点，保存 Openskill 评分分布""" def __init__(self, name: str, mu: float = 25.0, sigma: float = 25.0 / 3.0, is_anchor: bool = False): self.name = name self.is_anchor = is_anchor self.engine = PlackettLuce() self.rating = self.engine.rating(mu=mu, sigma=sigma) # 记录固定的锚点真实分值 self._anchor_mu = mu self._anchor_sigma = sigma @property def mu(self) -> float: return self._anchor_mu if self.is_anchor else self.rating.mu @property def sigma(self) -> float: return self._anchor_sigma if self.is_anchor else self.rating.sigma @property def conservative_rating(self) -> float: """保守等级分，常用于排位与门禁判定 (mu - 3*sigma)""" return self.mu - 3 * self.sigma def restore_anchor(self): """若为锚点，强制复位其 rating 分布""" if self.is_anchor: self.rating = self.engine.rating(mu=self._anchor_mu, sigma=self._anchor_sigma) class DriftFreeOpenskillEvaluator: """无漂移 Openskill 评估器""" def __init__(self, default_mu: float = 25.0, default_sigma: float = 8.333): self.engine = PlackettLuce() self.default_mu = default_mu self.default_sigma = default_sigma self.nodes: Dict[str, EvaluatorNode] = {} self.match_history: List[Tuple[str, str, int]] = [] # (p1, p2, outcome: 0=p1 win, 1=p2 win, 0.5=draw) def register_node(self, name: str, mu: Optional[float] = None, sigma: Optional[float] = None, is_anchor: bool = False): init_mu = mu if mu is not None else self.default_mu init_sigma = sigma if sigma is not None else self.default_sigma self.nodes[name] = EvaluatorNode(name, mu=init_mu, sigma=init_sigma, is_anchor=is_anchor) def update_match(self, p1_name: str, p2_name: str, outcome: float): """ 更新对局结果，自动应用锚点冻结保护机制 :param outcome: 0.0 -> p1 胜, 1.0 -> p2 胜, 0.5 -> 平局 """ node1 = self.nodes[p1_name] node2 = self.nodes[p2_name] if outcome == 0.0: ranks = [0, 1] elif outcome == 1.0: ranks = [1, 0] else: ranks = [0, 0] # 平局 # 计算新的分布 new_r1, new_r2 = self.engine.rate([[node1.rating], [node2.rating]], ranks=ranks) # 仅非锚点节点更新，锚点节点抛弃更新结果 if not node1.is_anchor: node1.rating = new_r1[0] if not node2.is_anchor: node2.rating = new_r2[0] # 记录对局历史，用于后台重估 self.match_history.append((p1_name, p2_name, outcome)) def select_opponents_for_candidate(self, candidate_name: str, num_matches: int = 5) -> List[str]: """ 不确定性驱动匹配：优先挑选相近 mu 且带有高置信度锚点的对手 """ cand = self.nodes[candidate_name] candidates_pool = [name for name in self.nodes.keys() if name != candidate_name] # 匹配函数：结合分值接近度 (mu 差异) 与对局价值 (对手的 sigma 稳定性) def match_score(opp_name: str) -> float: opp = self.nodes[opp_name] mu_diff = abs(cand.mu - opp.mu) # 优先选择分值接近且相对稳定的对手（尤其是锚点） anchor_bonus = 5.0 if opp.is_anchor else 0.0 return -mu_diff + anchor_bonus sorted_opponents = sorted(candidates_pool, key=match_score, reverse=True) return sorted_opponents[:num_matches] def batch_recalibrate(self): """ 批处理全历史校准 (Batch Whole-History Refit) 重新重置所有非锚点节点的 rating，重新按历史轨迹拟合，消除路径依赖漂移。 """ # 1. 重置所有非锚点节点 for node in self.nodes.values(): if not node.is_anchor: node.rating = self.engine.rating(mu=self.default_mu, sigma=self.default_sigma) # 2. 重新顺序计算整个历史 for p1_name, p2_name, outcome in self.match_history: self.update_match(p1_name, p2_name, outcome) def to_elo_scale(self, name: str, scale: float = 400 / math.log(10), target_base: float = 1500) -> float: """将 Openskill 的 mu 转换为传统 Elo/Glicko 标度""" node = self.nodes[name] # Openskill 默认 mu=25.0 对应基准分 1500 Elo return target_base + (node.mu - self.default_mu) * (400.0 / 8.333) 

3. 神经网络迭代门禁与流程整合 (SPRT)

在 AlphaZero 架构中，直接将新网络加入全局对弈池会导致无意义计算。推荐采用 SPRT（序贯概率比检验）门禁 + Openskill 校准 的双层架构：

+-------------------+ | 训练生成 Checkpoint | +---------+---------+ | v [ SPRT 测试 (vs 当前 Best 模型) ] | +------------+------------+ | (Fail / Reject) | (Pass / Accept) v v 丢弃该 Checkpoint 1. 替换为当前 Best 模型 2. 注册进 Openskill 评估池 3. 与锚点池对弈以确立无漂移 Elo 

SPRT 门禁配置准则

假设设定：

H_0: 胜率 p \le 0.50（ Elo 增量 \le 0）

H_1: 胜率 p \ge 0.55（ Elo 增量 \ge 35）

当 SPRT 判定接受（Accept）时，说明新模型实力显著超越上代，将其放入 Openskill 对弈池，并冻结其作为新的动态锚点候选。

4. 非传递性（循环克制）检测与应对

Openskill 将模型实力假设为一个一维标量分布。在棋牌类 AI（特别是复杂棋类或博弈游戏）中，经常出现策略循环（Style / Intransitivity Loops，如剪刀石头布）。

对局真实胜率标量 Openskill 预测矛盾点Model A vs Model B60% (A 胜)\mu_A > \mu_B逻辑自洽Model B vs Model C60% (B 胜)\mu_B > \mu_C逻辑自洽Model C vs Model A60% (C 胜)要求 \mu_C > \mu_A标量假设失效，导致评分漂移与震荡 

解决方案：交叉胜率矩阵（Payoff / Cross-Win Matrix）

维持 Cross-Win Matrix：在 Runs 历史中记录任意两个 Checkpoint 间的胜负统计矩阵 M_{i, j}。

非传递性度量：计算矩阵的不对称性与循环环路。若检测到严重的剪刀石头布效应，说明简单的等级分系统（包括 Openskill / Elo）已失真。

League Training 应对：引入 AlphaStar 策略，训练针对性“Exploiter”模型打碎循环，或构建基于Nash 均衡权重的混合评测池（Population-based evaluation）。

5. 最佳实践总结表

维度传统 Openskill 应用无漂移棋牌 AI 评估系统参考系无锚点，所有玩家同步更新静态固定引擎 + 里程碑模型硬性锚点方差 (\sigma) 处理随着对局增多降至 0新模型重置高 \sigma，老模型设定最低 \sigma_{\min}更新算法纯增量式（Online Updates）增量筛选 (SPRT) + 定期全历史拟合 (Batch Refit) 对阵匹配随机配对 / 顺位配对基于不确定性 (\sigma) 和锚点接近度的智能匹配评估维度仅依赖一维 RatingOpenskill 标量分 + 交叉胜率矩阵 (矩阵博弈校准) 

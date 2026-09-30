> **状态横幅（2026-09-29 更新）**：本备忘按当时的证据给出 ① value / ② belief / ③ policy 三条线的排序；其中 ① ② 均已有后续实验结论，正文保留为当时的判断、不再代表现行状态：
> ① value 迭代的 round 1（以 `t_leafq` 重采真实叶子 + 同 recipe 重训 critic）配对为负、任务关闭（N-24），`t_leafq` 保持部署默认，见 [`docs/experiments/ei3-value-loop-round1.md`](docs/experiments/ei3-value-loop-round1.md)；
> ② 隐藏手牌学习后验 held-out AUC 0.742 低于 count-only 0.748，人类 bank ROI 主门 FAIL → 不集成、不 h2h（重访条件 = held-out top-k placement 优势显著变大），见 [`docs/experiments/belief-posterior-probe.md`](docs/experiments/belief-posterior-probe.md)；
> ③ policy 线仍只剩条件性条目：warm-start 续训 2026-09-29 确认 +8.20（未达 +10 门），见 [`docs/experiments/warmstart-adamw-1m.md`](docs/experiments/warmstart-adamw-1m.md)。现行结论入口：[`docs/experiments/README.md`](docs/experiments/README.md)。

按今天的证据把「已关闭清单」（plans.md §6）和现有结果对齐后，能继续吃训练收益的其实只剩三类，优先级很分明：                                                                                                         
                                                                                                                                                                                                                     
 ① value / critic：最确定、最便宜（今天刚被重新打开）                                                                                                                                                                
                                                                                                                                                                                                                     
 证据：t_leafq（用 真实叶子终局 margin 训练）在 fresh bank confirm +18.24 Elo；而 t_searchq/t_rawq（蒸馏搜索自身估计）是平局。训练耗时 ~6 秒（356k 行、20 epoch），瓶颈只在数据采集。                                
                                                                                                                                                                                                                     
 为什么和 N-22 不冲突：N-22 关闭的是「把 value 重训到 search 的 E_w[Q]」（K=8、修复前、自蒸馏循环）——那条确实无效。今天证明换对监督目标（真实终局）+ 修复后引擎，value 就转化成了强度。所以 value 线是"开着的"，只是 
 目标要对。                                                                                                                                                                                                          
                                                                                                                                                                                                                     
 可做的具体事（每步都可被 h2h confirm 证伪）：                                                                                                                                                                       
 - 用当前最佳搜索（t_leafq）在修复后引擎重采叶子：K=32、t∈{5,10}、更多 roster/bank（现有 runs/ei2_value_t5/collect_t5.py + trunc_leaves.py + train_value.py 全在）；                                                 
 - 迭代 value-improvement 循环：采叶子 → 重训 critic → 用新 critic 搜索 → 再采……每轮几十毫秒训练、采集 1–2 小时；                                                                                                    
 - 目前叶子分布 EV 只有 0.543（t_searchq 0.582，噪声上限 0.91-0.95），离顶很远；                                                                                                                                     
 - 可试：critic 头加宽/解冻 trunk 最后一层（注意容量线 N-2 关的是 policy，不是 critic）、胜负/风险目标（C10 显示真人局面校准偏移）；                                                                                 
 - 筛选：先看 479 真人决策的 bias/EV，再进 same-bank h2h，最后 fresh-bank confirm。                                                                                                                                  
                                                                                                                                                                                                                     
 预期：每次叶子 EV 提升的历史转化率约为"EV +0.05 → 可测 Elo 增益"（本轮 +18）。成本：小时级/轮。                                                                                                                     
                                                                                                                                                                                                                     
 ② belief / determinization 后验：唯一能改变"信息上界"的训练方向                                                                                                                                                     
                                                                                                                                                                                                                     
 搜索现在对隐藏手牌均匀采样；oracle 换成真实手牌会翻转 40.3% 的 searched 决策（同 K 噪声下限 28.4%）。可训练一个 P(隐藏手牌 | 公开历史) 模型，标签来自自弈真实状态（或真人 trace），在推理侧替换均匀采样。           
                                                                                                                                                                                                                     
 - 与 N-13（离策略价值 regret）和 N-21（动作蒸馏）都不同：这是隐状态推断，不依赖 obs v5 能否决定动作；                                                                                                               
 - 与已关闭的 EVH（事件表示喂给学生网络）也不同：训练的是外部 belief 模型，不动学生 obs，不需要 ADR；                                                                                                                
 - 手工倾斜先验试过（CI 含 0，收益未证明），但学习到的后验没试过；先离线测翻转率与 chosen 值，再进 h2h；                                                                                                             
 - 预期：上限高、风险中；成本 1–2 周级。                                                                                                                                                                             
                                                                                                                                                                                                                     
 ③ policy：只有条件性活口                                                                                                                                                                                            
                                                                                                                                                                                                                     
 - 从零 PPO、容量 256/512、self-play/pool 机制、奖励形状、双塔——全关闭（N-1..N-5、N-20），平台 188–192 很稳；再烧 raw PPO 不划算。                                                                                   
 - 唯一过门槛的策略信号是 warm-start 续训 +11.48（k=3，探索性）；值得补 confirm，但不是大杠杆。                                                                                                                      
 - 动作蒸馏搜索（N-21）关闭的原因不是"搜索不够强"，而是 obs v5 无法决定搜索动作（同 View 两次搜索翻转 51.7%）。要再走权重路径，前置条件是改学生可见信息（belief/obs，ADR 级）——这正好和 ② 是同一项工程。             
 - 一个新的小方向：训练"候选提案头"——让 top-6 里包含搜索最优的比例更高。人类局面下 25.2% 的最优模板在 top-6 外，但当前已实现代价小（放宽到全模板只 +3–5 Elo），所以预期收益有限。                                    
 - 训练时把搜索当对手做课程（现在对手池全是更弱的 raw）没测过，属于中等成本、中等先验。                                                                                                                              
                                                                                                                                                                                                                     
 不建议再投的                                                                                                                                                                                                        
                                                                                                                                                                                                                     
 加深搜索（N-23）、扩大候选集（C11 实测小）、双塔（N-20）、离线价值 regret（N-13）、评估协议再改（N-17）、跨 fit 绝对分比较（N-7）。                                                                                 
                                                                                                                                                                                                                     
 建议的开工顺序                                                                                                                                                                                                      
                                                                                                                                                                                                                     
 1. Value 迭代循环（1–2 天）：修复后引擎重采 t=5/t=10 叶子 → 重训 critic → 479 真人决策筛选 → same-bank h2h → 过了再 fresh-bank confirm。训练几乎免费，ROI 最高。                                                    
 2. Belief 后验（1–2 周）：离线 ROI 探针先行。                                                                                                                                                                       
 3. Policy 只在前两步把信息/价值补齐后，再考虑观测/belief 级改动 + warm-start 续训。                                                                                                                                 
                                                                                                                                                                                                                     
 要开的话我建议先起 ①的小循环工作流（采一版 t5+K32+t10 叶子 → 训 2 个 critic → 真人局面 + same-bank screen → 每点 1 个对抗审查），预算和之前 P2 一个量级。
 
 traces/web/run-20260929-094059/g0000__s644014887__seat0__vsw2m_ctl.json

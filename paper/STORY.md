# Paper Story: What Makes a Good Routing Decision?

> Rubric-Guided Process Reward for Stepwise Model Routing

---

## 一句话总结

现有 stepwise routing 方法只用 sparse outcome reward 训练路由策略, 我们首次为路由决策提供 dense, interpretable, zero-cost 的 process reward — 通过 "seed→explore→filter" rubric 框架。

---

## 故事线

### 1. 开场: PRM 的启示

PRM (Process Reward Model) 的核心贡献是什么? 是把 "答案对不对" 这个 sparse 信号, 变成了 "每一步对不对" 这个 dense 信号。这一思路极大提升了数学推理的训练效率。

**我们的观察**: Stepwise model routing 面临 **完全相同的 sparse reward 问题**, 但没有人提供过它的 "process reward"。

现有路由方法 (TRIM, RSD, STEER, GlimpRouter) 要么用 sparse outcome reward 训练, 要么根本不训练。

### 2. 关键区分: 路由质量 ≠ 推理质量

**不能直接用 PRM 当路由的 process reward**, 因为:
- PRM 评价的是 "这个推理步骤对不对" (reasoning quality)
- 路由需要评价的是 "在这里升级到大模型这个决定对不对" (routing decision quality)

一个 PRM=0.9 的步骤被升级了 → 可能是浪费 (SRM 就够了)
一个 PRM=0.3 的步骤没被升级 → 可能是正确的 (LRM 也救不了)

### 3. 核心方法: Seed → Explore → Filter

我们不是凭空设计 rubric, 而是遵循一个有原则的发现流程:

**Step 1: Seed Rubrics (人类先验)**
- 5 个 seed rubric, 来自 "什么是好路由" 的第一性原理
- 例: "及时升级" (R1), "成本效率" (R2), "级联错误预防" (R3)
- 这些是人类直觉可以直接写出的

**Step 2: 探索 (自动发现)**
- Seed rubric 启发我们去数据中挖掘更多判据
- 例: R1 (及时升级) 启发了 R7 (置信度校准) 和 R11 (早期检测)
- 例: R3 (级联预防) 启发了 R10 (无振荡)
- 7 个 derived rubric, 每个都能追溯到它的 seed 灵感来源

**Step 3: 统计筛选 (三重验证)**
- 相关性: rubric 分数与答案正确性的 Pearson ρ > 0.05
- 一致性: 正确轨迹的 rubric 分数一致高于错误轨迹 > 55%
- 可区分性: rubric 分数方差 > 0.02 (不是常数)
- 12 个候选中约 6-8 个通过, 按相关性加权

### 4. 为什么这个故事好

**与 RLCER 的关系**: RLCER 用 LLM 自动生成 rubric 评价推理过程, 我们把 rubric 的思想迁移到路由领域, 但用 **rule-based zero-cost** 方式实现 (不需要额外 LLM 调用)

**与 OpenRubrics 的关系**: OpenRubrics 通过对比正确/错误样本来生成 rubric, 我们用类似的对比思想做 consistency check, 但对象是路由轨迹而非回答

**独特价值**: 
- **Zero-cost**: 所有 rubric 都是轨迹特征的确定性函数, 训练时零额外开销
- **Interpretable**: 每个 rubric 都有名字和明确含义, 可以告诉你路由策略哪里好哪里差
- **Principled**: seed→explore→filter 是一个有原则的发现流程, 不是 ad hoc

### 5. 实验故事

**核心对比**:
- TRIM-Agg (outcome-only reward) vs TRIM-Rubric (outcome + rubric process reward)
- 在相同 FLOPs 预算下, TRIM-Rubric 准确率更高
- 在达到相同准确率时, TRIM-Rubric 需要更少 FLOPs

**数据可信度**:
- 训练在 OmniMath (完全不同的数据集)
- 测试在 MATH-500 + AIME 2025 (held-out)
- 零数据泄漏

---

## 知识库文件

论文原文和详细总结 JSON 位于:
- `paper/knowledge_base/stepwise_model_routing/` — 6篇路由相关论文
- `paper/knowledge_base/rubric/` — 2篇 rubric 相关论文

### 路由论文
| ID | 论文 | 核心贡献 | 我们的推进 |
|----|------|---------|-----------|
| smr_001 | RSD | 奖励引导的推测解码 | 学习路由策略 vs 固定阈值 |
| smr_002 | SpecReason | 推测推理框架 | 密集奖励 vs 无学习 |
| smr_003 | SpecCoT | 加速 CoT 的投机解码 | 路由质量评估 vs 加速 |
| smr_004 | STEER | 置信度零成本路由 | Rubric 策略训练 vs 启发式 |
| smr_005 | GlimpRouter | 初始 token 熵路由 | Dense reward 训练 |
| smr_006 | TRIM | PRM 引导 RL 路由 | Dense rubric reward vs sparse outcome |

### Rubric 论文
| ID | 论文 | 核心贡献 | 我们的借鉴 |
|----|------|---------|-----------|
| rub_001 | OpenRubrics | 对比 rubric 生成 | 一致性验证方法 |
| rub_002 | RLCER | 自演化 rubric | seed→explore 思想; 相关性筛选 |

---

## Contribution 提炼

1. **首次为路由决策提供 process reward** — 区别于 PRM 评价推理质量, rubric 评价路由质量
2. **Seed→Explore→Filter rubric 框架** — 人类先验启发自动探索, 统计验证保证有效性
3. **零成本密集奖励** — rubric 是轨迹特征函数, 无需额外推理
4. **三重统计验证** — 相关性+一致性+可区分性, 防止噪声 rubric 进入奖励
5. **严格跨数据集评估** — 训练/测试集完全分离, 无数据泄漏

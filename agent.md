# 项目说明

本项目当前的主要工作，是在下面这篇论文的基础上进行修改与扩展：

- 基础论文：`/home/chencheng/routing/paper/knowledge_base/2601.10245v2.pdf`

## 主要目标

### 1. TRIM-Agg 复现

- 复现 TRIM-Agg。
- SRM / LRM 先使用 `Qwen3-1.7B` 和 `Qwen3-14B`。
- 也可以进一步尝试 `Qwen3.5`。
- 第一阶段先把下面几部分完整跑通：
  - TRIM-Agg 的 PPO router
  - accuracy-cost curve
  - LRM usage rate

### 2. RoRo: Rubric-Guided Process Reward for Stepwise Model Routing

整体流程如下。

#### 2.1 轨迹构造

- 先挑选出 SRM-only 错误、但 LRM-only 正确的样本。
- 对每个样本采样多条 routing trajectories。
- 每条 trajectory 需要记录：
  - step-level action
  - 推理内容
  - 最终正确性
  - LRM cost

#### 2.2 路由偏好对构造

- 基于如下效用函数构造 preference pairs：
  - `U(tau) = correctness(tau) - lambda * cost(tau)`
- 优先保留 margin 明显的 pair。
- 重点构造以下几类偏好：
  - 正确低成本 > 错误轨迹
  - 正确低成本 > 正确高成本
  - 稳定轨迹 > 震荡轨迹

#### 2.3 Candidate routing rubric 生成

- 用三个 seed axes 引导 LLM 生成细粒度 rubric：
  - timely escalation
  - effective recovery
  - efficient allocation
- 输入 preferred / rejected trajectory pair。
- 让 LLM 总结“为什么更好的 routing 更好”。

#### 2.4 Rubric 质量筛选

- 参考 RLCER / RRD 的思路，例如与答案一致性比例等指标。
- 只保留那些能够稳定地区分好坏 routing trajectories 的 rubrics。

#### 2.5 Rubric-based process reward

- 用筛选后的 rubrics 计算：
  - `R_rubric(t) = average(score_j(t))`
- 最终 reward 为：
  - `R = correctness - lambda * cost + beta * R_rubric`

#### 2.6 PPO router 训练

- 继续沿用 TRIM-style PPO。
- 在完整 trajectory 上计算 reward。
- trajectory 中每一个 routing action 都参与 policy update。

### 3. Motivation 章节的启发性实验

- 增加用于支撑 Motivation 章节的启发性实验。

## 工作原则

- 代码和实验设计优先保持与基础论文一致，再在其上扩展到 RoRo 方向。
- 后续代码修改默认限制在 `/home/deepseek_VG/deepseek_VG/routing/routing/src` 下进行，除非任务明确需要同步修改脚本、测试、文档或 TRIM 侧适配文件。
- 所有新功能默认建立在当前 trim-rubric pipeline 的基础上扩展，优先复用已有的数据构造、vLLM 调用、rubric 评分与测试结构，避免另起一套不兼容流程。
- 尽量采用小粒度、边界清晰的提交方式，确保每完成一个功能都可以通过 git 历史单独回退。
- 每完成一个相对完整的功能或可运行阶段后，先提交到本地 git，并同步到本机备份 remote `local-backup`，用于在误操作 `git reset` 后恢复当前代码。
- 当前本地恢复点为 `backup/current-code-2026-05-04` 和 `snapshot/current-code-2026-05-04`；如需回退到该版本，可执行 `git reset --hard backup/current-code-2026-05-04`。

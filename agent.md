# Agent Notes

This project is primarily developed as a modification and extension of the paper:

- Base paper: `/home/chencheng/routing/paper/knowledge_base/2601.10245v2.pdf`

## Main Goals

### 1. TRIM-Agg Reproduction

- Reproduce TRIM-Agg.
- For SRM/LRM, start with `Qwen3-1.7B` and `Qwen3-14B`.
- `Qwen3.5` can also be tested as an alternative.
- First get the following pieces running end to end:
  - TRIM-Agg PPO router
  - accuracy-cost curve
  - LRM usage rate

### 2. RoRo: Rubric-Guided Process Reward for Stepwise Model Routing

Pipeline:

#### 2.1 Trajectory Construction

- First select samples where SRM-only is wrong but LRM-only is correct.
- For each sample, sample multiple routing trajectories.
- For each trajectory, record:
  - step-level action
  - reasoning content
  - final correctness
  - LRM cost

#### 2.2 Routing Preference Pair Construction

- Construct preference pairs based on:
  - `U(tau) = correctness(tau) - lambda * cost(tau)`
- Prefer pairs with a clear margin.
- Focus on the following pair types:
  - correct low-cost > incorrect trajectory
  - correct low-cost > correct high-cost
  - stable trajectory > oscillating trajectory

#### 2.3 Candidate Routing Rubric Generation

- Use three seed axes to guide the LLM to generate fine-grained rubrics:
  - timely escalation
  - effective recovery
  - efficient allocation
- Feed preferred / rejected trajectory pairs to the LLM.
- Ask the LLM to summarize why the better routing is better.

#### 2.4 Rubric Quality Filtering

- Refer to ideas from RLCER / RRD, such as answer consistency ratio.
- Only keep rubrics that can reliably distinguish good and bad routing trajectories.

#### 2.5 Rubric-Based Process Reward

- Use filtered rubrics to compute:
  - `R_rubric(t) = average(score_j(t))`
- Final reward:
  - `R = correctness - lambda * cost + beta * R_rubric`

#### 2.6 PPO Router Training

- Continue to use TRIM-style PPO.
- Compute reward on the full trajectory.
- Every routing action in the trajectory participates in the policy update.

### 3. Motivation Section Experiments

- Add exploratory experiments for the Motivation section.

## Working Principle

- Keep changes aligned with the base paper while extending the routing framework toward RoRo.
- Prefer small, isolated commits so each completed function can be rolled back cleanly with git history.

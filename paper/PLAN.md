# What Makes a Good Routing Decision? Rubric-Guided Process Reward for Stepwise Model Routing

---

## Abstract

Stepwise model routing dynamically assigns individual reasoning steps to either a small reasoning model (SRM) or a large reasoning model (LRM), seeking to balance inference cost with accuracy. However, existing routing methods rely exclusively on **sparse outcome rewards** — the binary correctness of the final answer — providing no signal about *what makes an individual routing decision good or bad*. We introduce **rubric-guided process rewards**, a framework that provides dense, interpretable, and zero-cost feedback on routing decision quality.

Our approach follows a **seed-then-explore paradigm**: we begin with a small set of human-designed **seed rubrics** — first-principle criteria capturing intuitive notions of good routing (e.g., "escalate when the model is struggling," "don't waste budget on easy steps"). These seed rubrics then **inspire automated exploration** of a broader space of derived rubrics mined from trajectory statistics. Finally, a **triple statistical validation** pipeline filters the expanded rubric set, retaining only those criteria with verified correlation to task outcomes, consistency across trajectories, and sufficient discriminability. The result is a compact, validated rubric set that serves as a dense reward signal for PPO-based router training.

Crucially, all rubric scoring is **zero-cost**: rubrics are deterministic functions of trajectory features already available during training (PRM scores, token counts, routing actions), requiring no additional LLM calls. Experiments on **MATH-500** and **AIME 2025** (held-out from training) demonstrate state-of-the-art accuracy-cost tradeoffs, with the rubric-guided router surpassing both the standalone LRM and all baseline routing methods.

---

## 1. Introduction

### 1.1 The Sparse Reward Problem in Routing

Process Reward Models (PRMs) transformed mathematical reasoning by replacing sparse outcome-only feedback with dense step-level signals: instead of asking "did you get the right answer?", PRMs ask "is each reasoning step correct?" This shift dramatically improved training signal quality for chain-of-thought reasoning.

We observe that stepwise model routing faces an **analogous — but distinct — sparse reward problem**, and no one has provided its equivalent of "process rewards." Current routing methods either:
- Train with sparse outcome-only reward: R = 1[correct] − λ·Cost (TRIM), or
- Use heuristic rules without any learning signal (RSD, STEER, GlimpRouter).

The key distinction: PRMs evaluate *reasoning quality* ("is this step correct?"), while routing needs evaluation of *decision quality* ("was escalating here a good decision?"). These are fundamentally different objects of evaluation — a step can be correct without needing escalation, or incorrect despite escalation.

### 1.2 Three Compounding Challenges

The sparse reward problem in routing is harder than in reasoning due to three compounding difficulties:

**Challenge 1: Credit Assignment over Long Horizons.** A typical trajectory contains 5–20 routing decisions, but only the final step receives a non-zero reward. The router must infer from a single bit: which escalations were valuable? Which were wasteful?

**Challenge 2: Routing Quality ≠ Reasoning Quality.** PRM scores measure "how correct is this step" — not "was it wise to escalate." A step with PRM=0.9 that was escalated may be wasteful (SRM sufficed). A step with PRM=0.3 that wasn't escalated may be correct (LRM can't help either).

**Challenge 3: The Low-Budget Regime.** When cost constraints are strict, the router must learn "only escalate at the most critical moments." But sparse rewards cannot communicate what "critical" means.

### 1.3 Our Approach: Seed → Explore → Filter

We address all three challenges through a **seed-then-explore rubric framework**:

**Step 1: Seed Rubrics (Human Priors).** We design 5 seed rubrics capturing first-principle notions of good routing: timely escalation, cost efficiency, cascading error prevention, recovery effectiveness, and budget allocation. These encode domain knowledge about what routing quality *should* look like.

**Step 2: Rubric Exploration (Automated Discovery).** Seed rubrics *inspire* automated exploration of a larger rubric space. Using the seed rubrics as templates, we systematically derive additional criteria from trajectory statistics — patterns that the fixed seed set might miss, such as difficulty awareness, confidence calibration, and oscillation avoidance.

**Step 3: Statistical Filtering (Validation).** The expanded rubric set is pruned through triple statistical validation: (1) correlation with task correctness, (2) consistency across trajectory pairs, and (3) discriminability thresholds. Only rubrics with verified predictive power survive.

The filtered rubric set becomes a **dense, multi-dimensional reward signal** embedded directly in the PPO training loop — at zero additional compute cost.

---

## 2. Method

### 2.0 Model Configuration

| Role | Model | Purpose | Deployment |
|------|-------|---------|------------|
| SRM (M_w) | Qwen3-1.7B | Step-by-step reasoning (thinking mode) | vLLM port 4003 (GPU 4) |
| LRM (M_s) | Qwen3-14B | Critical step regeneration (thinking mode) | vLLM port 4001 (GPU 5,6) |
| PRM | Qwen2.5-Math-PRM-7B | Per-step reasoning quality scoring | Local GPU 0 |
| Router | 2-layer MLP (128 hidden, Tanh) | PPO-trained routing policy | CPU |

### 2.1 Problem Formulation

At each reasoning step $t$, the router observes state $s_t = (r_t, \min(r_{1:t-1}), c_t, t)$ where:
- $r_t$: PRM score of SRM's step $t$
- $\min(r_{1:t-1})$: worst historical PRM score (trajectory fragility indicator)
- $c_t$: normalized token count of step $t$
- $t$: step index

The router chooses action $a_t \in \{0=\text{continue with SRM}, 1=\text{regenerate with LRM}\}$.

### 2.2 Seed Rubrics: First-Principle Routing Criteria

We design 5 seed rubrics from first principles. Each answers a specific question about routing quality:

| ID | Rubric | Question | Scoring Function |
|----|--------|----------|-----------------|
| R1 | Timely Escalation | Did the router escalate when SRM struggled? | Fraction of low-PRM (< 0.3) steps escalated |
| R2 | Cost Efficiency | Did the router avoid wasting LRM on easy steps? | 1 − fraction of high-PRM (> 0.8) steps escalated |
| R3 | Cascading Error Prevention | Did the router intervene before consecutive drops? | Fraction of declining PRM windows with intervention |
| R4 | Recovery Effectiveness | Did escalation actually improve quality? | Fraction of escalations where LRM PRM > SRM PRM |
| R5 | Budget Allocation | Was budget allocated to the neediest steps? | Spearman ρ between PRM-need and actions |

**Design principles:**
- R1–R2 form a precision-recall pair for escalation decisions
- R3–R4 capture dynamic trajectory-level patterns
- R5 provides a global allocation quality measure

### 2.3 Rubric Exploration: From Seeds to Derived Criteria

Seed rubrics capture obvious routing principles but may miss subtler patterns. We systematically expand the rubric space by mining trajectory statistics for additional discriminative criteria:

| ID | Derived Rubric | Seed Inspiration | Discovery Method |
|----|---------------|-----------------|-----------------|
| R6 | Difficulty Awareness | R5 (budget allocation) | Correlate problem-level mean PRM with regen ratio |
| R7 | Confidence Calibration | R1 (timely escalation) | Bin PRM scores, check escalation monotonicity |
| R8 | Action Consistency | R1, R2 (escalation thresholds) | Measure variance of action given similar states |
| R9 | Marginal Gain Focus | R4 (recovery effectiveness) | Filter by PRM gap: only count escalations with significant LRM improvement |
| R10 | No Oscillation | R3 (error prevention) | Count rapid action switches (SRM→LRM→SRM) |
| R11 | Early Detection | R3 (cascading prevention) | Measure delay between first PRM drop and first escalation |
| R12 | Trajectory Quality | R4 (recovery) | Geometric mean of chosen steps' PRM scores |

The exploration is **guided by seed rubrics**: each derived rubric refines, generalizes, or specializes a seed criterion. This ensures the expanded set remains coherent rather than arbitrary.

### 2.4 Triple Statistical Validation

Not all rubrics are useful — some may be noise, some may not correlate with correctness, some may lack discriminability. We validate each candidate through three filters:

**Filter 1: Correlation Screening.** Generate diverse routing trajectories via multiple strategies (random, PRM-guided, threshold sweeps). Compute Pearson correlation between each rubric score and trajectory correctness. Retain rubrics with $|\rho| > \alpha$ ($\alpha = 0.05$).

**Filter 2: Pairwise Consistency.** Sample (correct, incorrect) trajectory pairs on the same problem. Verify that the correct trajectory scores higher on the rubric more often than chance. Retain rubrics with consistency rate > 55%.

**Filter 3: Discriminability.** Ensure rubric scores have sufficient variance across trajectories ($\sigma > 0.02$). Rubrics that always score the same value provide no useful gradient.

**Weight assignment:** Surviving rubrics are weighted proportionally to their correlation coefficients: $w_k = |\rho_k| / \sum_j |\rho_j|$.

### 2.5 Enhanced Reward Function

The final reward combines outcome, rubric process reward, and cost:

$$R(\tau) = \underbrace{\mathbb{1}[\text{correct}]}_{\text{outcome}} + \underbrace{\lambda_p \cdot R_{\text{rubric}}(\tau)}_{\text{process reward}} - \underbrace{\lambda_c \cdot C(\tau)}_{\text{cost}}$$

where:
- $R_{\text{rubric}}(\tau) = \sum_k w_k \cdot \text{rubric}_k(\tau)$ is the weighted validated rubric score
- $C(\tau) = \sum_t \mathbb{1}[a_t=1] \cdot \text{lrm\_tokens}_t$ is the LRM token cost
- $\lambda_p$, $\lambda_c$ are hyperparameters controlling the reward composition

### 2.6 Zero-Cost Advantage

| Dimension | LLM-as-Judge | PRM-based | Our Rubrics |
|-----------|-------------|-----------|-------------|
| Cost per trajectory | 1 LLM call | N PRM calls | **Zero** |
| Training integration | Offline only | Can embed | **Directly in reward** |
| Reproducibility | Stochastic | Deterministic | **Deterministic** |
| Interpretability | Opaque scalar | Per-step quality | **Named, multi-dimensional routing quality** |

---

## 3. Experimental Setup

### 3.1 Data Split (Zero Leakage)

| Split | Dataset | Size | Purpose |
|-------|---------|------|---------|
| **Training** | OmniMath (difficulty 1–4) | 200 problems | Episode generation + PPO training |
| **Test** | MATH-500 | 169 problems | Held-out evaluation |
| **Test** | AIME 2025 I & II | 30 problems | Held-out evaluation (competition math) |

The router is trained exclusively on OmniMath episodes and evaluated on completely held-out test sets. The correctness estimation for mixed-model trajectories uses PRM-based trajectory quality interpolation rather than label-leaking proxies.

### 3.2 Evaluation Metrics (EcoTab-style)

- **FLOPs per token**: SRM: 2×1.7B = 3.4×10⁹; LRM: 2×14B = 2.8×10¹⁰
- **Acc@60% LRM-FLOPs**: Accuracy when limited to 60% of LRM-only compute
- **FLOPs@98% LRM-Acc**: Compute needed to reach 98% of LRM accuracy
- **Pareto frontier**: Full accuracy-vs-FLOPs curve

### 3.3 Baselines

| Method | Category | Reward/Decision Rule |
|--------|----------|---------------------|
| SRM-only | Lower bound | All steps by Qwen3-1.7B |
| LRM-only | Upper bound | All steps by Qwen3-14B |
| TRIM-Thr | Threshold | PRM < threshold → regenerate |
| TRIM-Agg | RL (sparse) | R = 1[correct] − λ·C |
| **TRIM-Rubric (Ours)** | RL (dense) | R = 1[correct] + λ_p·R_rubric − λ_c·C |

---

## 4. Code Structure

```
src/
├── config.py                  # Configuration (models, paths, hyperparams)
├── vllm_client.py             # vLLM HTTP client (SRM/LRM)
├── models.py                  # PRM scorer, answer extraction, correctness check
├── data/
│   ├── datasets.py            # Dataset loaders (MATH-500, AIME2025, OmniMath)
│   └── generate_episodes.py   # Episode generation (SRM+LRM solutions + PRM scoring)
├── rubric/
│   ├── rubric_scorer.py       # 12 rubric criteria + scoring functions
│   └── generate_rubrics.py    # Seed → Explore → Filter pipeline
├── router/
│   ├── env.py                 # RL environment (PRM-based mixed correctness)
│   ├── policy.py              # Router MLP (actor-critic)
│   └── train_ppo.py           # PPO training loop
├── eval/
│   ├── evaluate.py            # Online/offline router evaluation
│   └── flops_eval.py          # FLOPs-based metrics (EcoTab-style)
└── scripts/
    └── run_all.sh             # Full pipeline with progress + time estimates
```

---

## 5. Pipeline Overview

```
┌─────────────────────────────────────────────────────────┐
│ Phase 1: Episode Generation (OmniMath)         ~8 hrs  │
│   SRM/LRM generate solutions → PRM scores each step    │
│   Output: data/episodes/omnimath_episodes.jsonl         │
├─────────────────────────────────────────────────────────┤
│ Phase 2: Rubric Discovery                      ~5 min  │
│   Seed rubrics → Explore derived → Triple validation    │
│   Output: data/rubrics/rubric_weights.json              │
├─────────────────────────────────────────────────────────┤
│ Phase 3: Router Training (PPO)                 ~30 min  │
│   3a. TRIM-Agg baseline (outcome-only reward)          │
│   3b. TRIM-Rubric (outcome + rubric process reward)    │
│   Output: checkpoints/trim_*/best.pt                    │
├─────────────────────────────────────────────────────────┤
│ Phase 4: Held-Out Evaluation                   ~10 min  │
│   Evaluate on MATH-500 + AIME 2025 episodes            │
│   Output: results/flops_evaluation/                     │
├─────────────────────────────────────────────────────────┤
│ Phase 5: Visualization                         ~1 min   │
│   Pareto curves, comparison tables                      │
│   Output: results/plots/                                │
└─────────────────────────────────────────────────────────┘
```

---

## 6. Key Contributions

1. **Process Reward for Routing Decisions**: First work providing dense, interpretable feedback on *routing quality* rather than *reasoning quality*. Bridges the gap between PRMs (step-level reasoning evaluation) and routing policy optimization.

2. **Seed → Explore → Filter Rubric Framework**: A principled rubric discovery pipeline where human-designed seed rubrics inspire automated exploration of a broader criteria space, followed by statistical validation. Balances human intuition with data-driven verification.

3. **Zero-Cost Dense Reward**: All rubric scoring is deterministic and free — computed from trajectory features already available during training. No additional LLM calls, no PRM inference, fully reproducible.

4. **Triple Statistical Validation**: Correlation filtering + pairwise consistency + discriminability ensures only genuinely predictive rubrics enter the reward function, preventing reward hacking.

5. **Rigorous Cross-Dataset Evaluation**: Train on OmniMath, test on MATH-500 + AIME 2025 with no data leakage. PRM-based mixed correctness estimation replaces label-leaking proxies.

---

## 7. Related Work

| Paper | Core Idea | Our Advance |
|-------|-----------|-------------|
| **TRIM** (ICML 2026) | PRM-guided RL routing, sparse reward | Dense rubric process reward; PRM-based correctness estimation |
| **RLCER** (2026) | Self-evolving rubrics for CoT | Seed→Explore paradigm; rule-based zero-cost rubrics for routing |
| **OpenRubrics** (ACL 2025) | Contrastive rubric generation | Consistency-based filtering; routing-domain adaptation |
| **RSD** (ICML 2025) | Reward-guided speculative decoding | Learned routing *policy* with dense rewards vs. fixed threshold |
| **STEER** (AAAI 2026) | Confidence-based zero-cost routing | Complementary signals; our rubrics + STEER's confidence |
| **GlimpRouter** (ACL 2026) | Initial token entropy routing | Our rubrics could enhance GlimpRouter's policy learning |

---

## 8. Ablation Design

1. **Core ablation**: TRIM-Agg (outcome only) vs TRIM-Rubric (outcome + rubric)
2. **Rubric weight λ_p**: {0.1, 0.3, 0.5, 1.0}
3. **Seed only vs Seed + Derived**: Tier-1 only vs Tier-1 + Tier-2
4. **Filtering threshold α**: {0.01, 0.05, 0.1, 0.2}
5. **Individual rubric contribution**: Train with each rubric alone
6. **Cross-dataset generalization**: Train on OmniMath → test on MATH-500/AIME

---

## 9. Limitations & Future Work

1. Current rubrics are rule-based trajectory functions; LLM-generated rubrics may capture subtler routing patterns
2. Rubric weights are learned offline and fixed; online co-evolution during PPO training is worth exploring
3. Mixed-correctness estimation remains approximate; online evaluation with live mixed inference is more accurate
4. Extension beyond math: code generation, multi-hop QA, and scientific reasoning with domain-specific routing rubrics
5. Scaling to more than two models (e.g., SRM/MRM/LRM three-tier routing)

"""Configuration for Stepwise Model Routing with Rubric-Guided Process Reward."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# ============================================================
# Model paths (local weights for PRM; SRM/LRM via vLLM API)
# ============================================================
MODEL_ROOT = os.environ.get("MODEL_ROOT", str(REPO_ROOT / "models"))

SRM_MODEL = os.path.join(MODEL_ROOT, "qwen3-1.7b")
LRM_MODEL = os.path.join(MODEL_ROOT, "qwen3-14b")
PRM_MODEL = os.path.join(MODEL_ROOT, "qwen2.5-math-prm-7b")

# ============================================================
# vLLM API endpoints (models already deployed)
# ============================================================
VLLM_SRM_URL = "http://localhost:4003/v1/chat/completions"
VLLM_LRM_URL = "http://localhost:4001/v1/chat/completions"
VLLM_SRM_PORT = 4003
VLLM_LRM_PORT = 4001

# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = os.environ.get("ROUTING_SRC_ROOT", str(REPO_ROOT / "src"))
TRIM_ROOT = os.environ.get(
    "TRIM_ROOT",
    str(REPO_ROOT / "trim" / "TRIM"),
)
TRIM_DATA_DIR = os.environ.get(
    "TRIM_DATA_DIR",
    str(Path(TRIM_ROOT) / "math_eval" / "data"),
)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
EPISODES_DIR = os.path.join(DATA_DIR, "episodes")
RUBRIC_DIR = os.path.join(DATA_DIR, "rubrics")
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# ============================================================
# Generation config
# ============================================================
STEP_DELIMITER = "\n\n"
MAX_STEPS = 30
MAX_NEW_TOKENS = 16384
THINK_MODE = True
SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."

# ============================================================
# PRM config
# ============================================================
PRM_DEVICE = "cuda:0"  # GPUs 0-3 and 7 are free

# ============================================================
# TRIM-Agg Router config (aligned with TRIM source)
# ============================================================
STATE_DIM = 5         # (min_prev, prod_prev, r_t, c_t, t)
ACTION_DIM = 2        # {continue=0, regenerate=1}
HIDDEN_DIM = 64
DROPOUT = 0.1

TOKEN_NORMALISER = 300

# PPO hyperparameters (aligned with TRIM source)
LR = 3e-4
CLIP_COEF = 0.2
ENTROPY_COEF = 0.01
ENTROPY_COEF_FINAL = 0.001  # linearly anneal entropy bonus
VALUE_LOSS_COEF = 0.5
GAE_LAMBDA = 0.95
GAMMA = 1.0                # undiscounted (TRIM)
MAX_GRAD_NORM = 0.5
PPO_EPOCHS = 3
NORMALIZE_ADVANTAGES = True
TASK_REWARD = 1.0

# Lambda ranges (cost-performance trade-off)
LAMBDA_AIME = [3e-4, 2.5e-4, 2e-4, 1.5e-4, 1e-4, 8e-5]
LAMBDA_MATH = [8e-4, 6e-4, 5e-4, 4e-4, 3e-4]

# ============================================================
# Rubric config
# ============================================================
LAMBDA_RUBRIC = 0.3         # weight for rubric process reward
RUBRIC_CORR_THRESHOLD = 0.1  # minimum correlation for rubric validity
RUBRIC_STD_THRESHOLD = 0.05  # minimum std for rubric discriminability

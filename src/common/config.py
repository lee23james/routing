"""Shared configuration for the TRIM project."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = os.environ.get("ROUTING_SRC_ROOT", str(REPO_ROOT / "src"))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
MODEL_ROOT = os.environ.get("MODEL_ROOT", str(REPO_ROOT / "models"))

SRM_PORT = 4013
LRM_PORT = 4011
SRM_URL = f"http://localhost:{SRM_PORT}/v1/chat/completions"
LRM_URL = f"http://localhost:{LRM_PORT}/v1/chat/completions"

# Baseline uses the original ports (4003/4001) to avoid contention
BASELINE_SRM_PORT = 4003
BASELINE_LRM_PORT = 4001
BASELINE_SRM_URL = f"http://localhost:{BASELINE_SRM_PORT}/v1/chat/completions"
BASELINE_LRM_URL = f"http://localhost:{BASELINE_LRM_PORT}/v1/chat/completions"

SRM_PARAMS_B = 1.7   # qwen3-1.7b
LRM_PARAMS_B = 14    # qwen3-14b

PRM_MODEL = os.path.join(MODEL_ROOT, "qwen2.5-math-prm-7b")
PRM_DEVICE = "cuda:4"

SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."

MAX_STEPS = 50
MAX_TOKENS_PER_STEP = 512
MAX_ANSWER_TOKENS = 512
MAX_TOTAL_TOKENS = 16384

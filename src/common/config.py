"""Shared configuration for the TRIM project."""

import os

PROJECT_ROOT = "/export/shy/pp/pp5"
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "checkpoints")

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

PRM_MODEL = "/export/yuguo/ppyg2/model/qwen2.5-math-prm-7b"
PRM_DEVICE = "cuda:4"

SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."

MAX_STEPS = 50
MAX_TOKENS_PER_STEP = 512
MAX_ANSWER_TOKENS = 512
MAX_TOTAL_TOKENS = 16384

"""PRM (Process Reward Model) scorer — singleton wrapper.

Lazily loads the PRM model on first use to avoid GPU cost when not needed.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.config import PRM_MODEL, PRM_DEVICE

_scorer = None


def get_prm():
    """Get or lazily create the singleton PRMScorer."""
    global _scorer
    if _scorer is None:
        from models import PRMScorer
        _scorer = PRMScorer(PRM_MODEL, device=PRM_DEVICE)
    return _scorer


def score_steps(query: str, steps: list) -> list:
    """Score a list of reasoning steps. Returns list of floats in [0,1]."""
    prm = get_prm()
    return prm.score_trace(query, steps)

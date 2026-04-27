"""LLM 轨迹对比评估: 用 LLM 按三维标准判断轨迹对优劣.

对每个配对, 构造 prompt 让 LLM 扮演评审角色, 基于:
  D1. 关键干预命中性
  D2. 切换平稳性
  D3. 路径简洁性
判断 Trajectory A vs B 哪条更好.

支持离线模式 (mock) 和在线模式 (调用 vLLM/OpenAI API).

用法:
    python -m motivation.llm_judge \
        --pairs_path results/motivation/trajectory_pairs.jsonl \
        --output_path results/motivation/llm_judgments.jsonl \
        --mode mock
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from motivation.process_quality import score_trajectory

JUDGE_PROMPT_TEMPLATE = """You are an expert evaluator for stepwise model routing in mathematical reasoning.

A routing trajectory decides, for each reasoning step, whether to keep the small model's (SRM) output or regenerate with the large model (LRM). You will compare two routing trajectories (A and B) for the same math problem and judge which one demonstrates better routing process quality.

**Problem context:**
- The SRM alone gets this problem WRONG, and the LRM alone gets it RIGHT.
- The critical error point is around step {critical_step} (0-indexed), where the SRM's process reward score drops significantly.
- SRM PRM scores per step: {srm_prm}
- Both trajectories have the same final correctness and similar cost.

**Trajectory A:** actions = {actions_a}
  (1 = use LRM, 0 = keep SRM; {n_regens_a} out of {n_steps} steps use LRM)

**Trajectory B:** actions = {actions_b}
  (1 = use LRM, 0 = keep SRM; {n_regens_b} out of {n_steps} steps use LRM)

**Judge on these three criteria:**
1. **Critical Intervention Hit**: Does the trajectory intervene (use LRM) at or near the critical error point (step {critical_step})? Timely intervention at the first major error is most important.
2. **Switching Smoothness**: Is the switching pattern smooth (e.g., a contiguous block of LRM usage) or does it oscillate rapidly between SRM and LRM?
3. **Path Conciseness**: Does the trajectory avoid wasting LRM on steps where SRM already performs well (high PRM scores)?

**Output format (JSON):**
```json
{{
  "criterion_1_winner": "A" or "B" or "tie",
  "criterion_2_winner": "A" or "B" or "tie",
  "criterion_3_winner": "A" or "B" or "tie",
  "overall_winner": "A" or "B" or "tie",
  "reasoning": "brief explanation"
}}
```"""


def build_judge_prompt(pair: Dict) -> str:
    srm_prm = pair.get("srm_prm_scores", [])
    critical = pair.get("critical_step", 0)
    a = pair["traj_a"]
    b = pair["traj_b"]
    n = len(a["actions"])

    prm_str = "[" + ", ".join(f"{s:.3f}" for s in srm_prm[:n]) + "]"
    return JUDGE_PROMPT_TEMPLATE.format(
        critical_step=critical,
        srm_prm=prm_str,
        actions_a=a["actions"],
        n_regens_a=a["n_regens"],
        actions_b=b["actions"],
        n_regens_b=b["n_regens"],
        n_steps=n,
    )


def mock_llm_judge(pair: Dict) -> Dict:
    """基于 process_quality 评分模拟 LLM 判断 (无需 API).

    直接用三维评分指标代替 LLM 输出, 用于快速验证流程.
    """
    srm_prm = pair.get("srm_prm_scores", [])
    lrm_prm = pair.get("lrm_prm_scores", [])
    critical = pair.get("critical_step", 0)
    a_actions = pair["traj_a"]["actions"]
    b_actions = pair["traj_b"]["actions"]

    sa = score_trajectory(a_actions, srm_prm, critical, lrm_prm)
    sb = score_trajectory(b_actions, srm_prm, critical, lrm_prm)

    def _winner(key):
        d = sa[key] - sb[key]
        if d > 0.05:
            return "A"
        if d < -0.05:
            return "B"
        return "tie"

    c1 = _winner("critical_hit")
    c2 = _winner("smoothness")
    c3 = _winner("conciseness")

    votes = {"A": 0, "B": 0, "tie": 0}
    for v in [c1, c2, c3]:
        votes[v] += 1
    if votes["A"] > votes["B"]:
        overall = "A"
    elif votes["B"] > votes["A"]:
        overall = "B"
    else:
        overall = "tie" if votes["tie"] >= 2 else ("A" if sa["process_quality"] >= sb["process_quality"] else "B")

    return {
        "criterion_1_winner": c1,
        "criterion_2_winner": c2,
        "criterion_3_winner": c3,
        "overall_winner": overall,
        "scores_a": sa,
        "scores_b": sb,
        "reasoning": f"Based on process quality scores: A={sa['process_quality']:.3f}, B={sb['process_quality']:.3f}",
        "mode": "mock",
    }


def call_llm_judge(pair: Dict, api_url: str = None, model: str = None) -> Dict:
    """通过 API 调用 LLM 做判断."""
    import requests

    prompt = build_judge_prompt(pair)
    url = api_url or "http://localhost:4001/v1/chat/completions"
    payload = {
        "model": model or "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
        "temperature": 0.1,
    }

    try:
        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            result = json.loads(content[start:end])
            result["mode"] = "llm"
            result["raw_response"] = content
            return result
    except Exception as e:
        print(f"  LLM call failed: {e}, falling back to mock")

    result = mock_llm_judge(pair)
    result["mode"] = "mock_fallback"
    return result


def judge_all_pairs(
    pairs: List[Dict],
    mode: str = "mock",
    api_url: str = None,
    model: str = None,
) -> List[Dict]:
    results = []
    for i, pair in enumerate(pairs):
        if mode == "mock":
            judgment = mock_llm_judge(pair)
        else:
            judgment = call_llm_judge(pair, api_url, model)
        judgment["pair_idx"] = i
        judgment["episode_id"] = pair.get("episode_id", "")
        results.append(judgment)
        if (i + 1) % 50 == 0:
            print(f"  Judged {i + 1}/{len(pairs)} pairs")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs_path", required=True)
    parser.add_argument("--output_path", default="results/motivation/llm_judgments.jsonl")
    parser.add_argument("--mode", choices=["mock", "llm"], default="mock")
    parser.add_argument("--api_url", default=None)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    pairs = []
    with open(args.pairs_path) as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))

    print(f"Loaded {len(pairs)} pairs")
    results = judge_all_pairs(pairs, args.mode, args.api_url, args.model)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    winners = {"A": 0, "B": 0, "tie": 0}
    for r in results:
        winners[r.get("overall_winner", "tie")] += 1
    print(f"\nJudgment summary: A={winners['A']} B={winners['B']} tie={winners['tie']}")
    print(f"Saved → {args.output_path}")


if __name__ == "__main__":
    main()

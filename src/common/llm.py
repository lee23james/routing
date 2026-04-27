"""LLM calling utilities — thin wrappers for vLLM chat endpoints."""

import requests
import time
from typing import Dict, List, Optional, Tuple


def call_model(
    url: str,
    messages: List[Dict],
    max_tokens: int = 512,
    temperature: float = 0.0,
    stop: Optional[List[str]] = None,
    think_mode: bool = True,
    continue_final: bool = False,
    logprobs: bool = False,
    top_logprobs: int = 5,
    timeout: int = 300,
) -> Dict:
    """Call a vLLM model endpoint.

    Returns dict with keys: content, tokens, elapsed, logprobs_content, finish_reason.
    """
    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.95 if temperature > 0 else 1.0,
        "think_mode": think_mode,
    }
    if stop:
        payload["stop"] = stop
    if logprobs:
        payload["logprobs"] = True
        payload["top_logprobs"] = top_logprobs
    if continue_final:
        payload["extra_body"] = {
            "add_generation_prompt": False,
            "continue_final_message": True,
        }

    t0 = time.time()
    resp = requests.post(url, json=payload, timeout=timeout)
    elapsed = time.time() - t0

    data = resp.json()
    if "error" in data:
        raise RuntimeError(data["error"][:500])

    choice = data["choices"][0]
    usage = data.get("usage", {})
    return {
        "content": choice["message"]["content"],
        "tokens": usage.get("completion_tokens", 0),
        "elapsed": elapsed,
        "finish_reason": choice.get("finish_reason", ""),
        "logprobs_content": (choice.get("logprobs") or {}).get("content", []),
    }


def generate_full_solution(
    url: str,
    problem: str,
    system_prompt: str,
    max_tokens: int = 16384,
    temperature: float = 0.0,
    think_mode: bool = True,
    timeout: int = 600,
) -> Dict:
    """Generate a full solution for a math problem (baseline mode)."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": problem},
    ]
    return call_model(
        url, messages,
        max_tokens=max_tokens,
        temperature=temperature,
        think_mode=think_mode,
        timeout=timeout,
    )

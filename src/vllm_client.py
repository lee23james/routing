"""Thin client for vLLM-served models via OpenAI-compatible API."""

import json
import time
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from config import SYSTEM_PROMPT


class VLLMClient:
    """Call a vLLM-served model through its HTTP endpoint."""

    def __init__(self, port: int = None, model_name: str = "default",
                 timeout: int = 600, max_retries: int = 3,
                 server_url: str = None):
        if server_url:
            base = server_url.rstrip("/")
            if base.endswith("/chat/completions"):
                self.url = base
            elif base.endswith("/v1"):
                self.url = f"{base}/chat/completions"
            else:
                self.url = f"{base}/v1/chat/completions"
        elif port is not None:
            self.url = f"http://localhost:{port}/v1/chat/completions"
        else:
            raise ValueError("Either port or server_url must be provided")
        self.model_name = model_name
        self.timeout = timeout
        self.max_retries = max_retries

    def _call(self, messages: list, max_tokens: int = 4096,
              temperature: float = 0.0, stop: Optional[list] = None,
              think_mode: bool = False) -> dict:
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.95 if temperature > 0 else 1.0,
            "think_mode": think_mode,
        }
        if stop:
            payload["stop"] = stop

        for attempt in range(self.max_retries):
            try:
                resp = requests.post(self.url, json=payload, timeout=self.timeout)
                data = resp.json()
                if "error" in data:
                    raise RuntimeError(data["error"])
                if data.get("object") == "error":
                    raise RuntimeError(data.get("message", data))
                return data
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)

    def generate_solution(self, query: str, max_tokens: int = 4096,
                          temperature: float = 0.0,
                          think_mode: bool = False) -> tuple:
        """Generate a full solution for a math problem.

        Returns (response_text, completion_tokens).
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]
        data = self._call(messages, max_tokens=max_tokens, temperature=temperature,
                          think_mode=think_mode)
        text = data["choices"][0]["message"]["content"]
        tokens = data.get("usage", {}).get("completion_tokens", 0)
        return text, tokens

    def generate_step(self, prefix_messages: list, max_tokens: int = 512,
                      temperature: float = 0.0) -> tuple:
        """Generate the next reasoning step (stop at double newline).

        Returns (step_text, completion_tokens).
        """
        data = self._call(prefix_messages, max_tokens=max_tokens,
                          temperature=temperature, stop=["\n\n"])
        text = data["choices"][0]["message"]["content"].strip()
        tokens = data.get("usage", {}).get("completion_tokens", 0)
        return text, tokens

    def batch_generate_solutions(self, queries: List[str],
                                  max_tokens: int = 4096,
                                  temperature: float = 0.0,
                                  max_workers: int = 4,
                                  think_mode: bool = False) -> List[tuple]:
        """Generate solutions for multiple queries in parallel."""
        results = [None] * len(queries)

        def _gen(idx, q):
            return idx, self.generate_solution(q, max_tokens, temperature,
                                               think_mode=think_mode)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futs = {pool.submit(_gen, i, q): i for i, q in enumerate(queries)}
            for fut in as_completed(futs):
                idx, result = fut.result()
                results[idx] = result
        return results

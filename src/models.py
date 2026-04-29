"""LLM inference wrapper: supports vLLM (fast) or HuggingFace transformers (fallback)."""

import re
import time
from typing import List, Optional, Tuple

import requests
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

try:
    from vllm import LLM, SamplingParams
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False
    print("Warning: vLLM not installed. Using HuggingFace transformers (slower).")

SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


class StepwiseLLM:
    """Wrapper for stepwise generation using vLLM or HF transformers."""

    def __init__(self, model_path: str, tp: int = 1, max_model_len: int = 8192,
                 gpu_memory_utilization: float = 0.9, device: str = None):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        if HAS_VLLM:
            print(f"Loading LLM (vLLM): {model_path} (tp={tp})")
            self.llm = LLM(
                model=model_path,
                trust_remote_code=True,
                tensor_parallel_size=tp,
                max_model_len=max_model_len,
                gpu_memory_utilization=gpu_memory_utilization,
            )
            self.backend = "vllm"
        else:
            print(f"Loading LLM (HF): {model_path}")
            self.hf_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map=device or "auto",
            )
            self.hf_model.eval()
            self.backend = "hf"

    def _build_prompt(self, query: str) -> str:
        """Build chat prompt for a math query."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            return f"{SYSTEM_PROMPT}\n\n{query}\n\n"

    def generate_step(self, prefixes: List[str], max_tokens: int = 512,
                      temperature: float = 0.0) -> List[str]:
        """Generate the next reasoning step for each prefix."""
        if self.backend == "vllm":
            sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=max_tokens,
                stop=["\n\n"],
                include_stop_str_in_output=False,
            )
            outputs = self.llm.generate(prefixes, sampling_params)
            return [o.outputs[0].text.strip() for o in outputs]
        else:
            results = []
            for prefix in prefixes:
                inputs = self.tokenizer(prefix, return_tensors="pt").to(self.hf_model.device)
                with torch.no_grad():
                    out = self.hf_model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        do_sample=(temperature > 0),
                        temperature=temperature if temperature > 0 else None,
                    )
                new_tokens = out[0][inputs["input_ids"].shape[1]:]
                text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                # Stop at double newline
                if "\n\n" in text:
                    text = text[:text.index("\n\n")]
                results.append(text.strip())
            return results

    def generate_full_solution(self, queries: List[str], max_tokens: int = 2048,
                                temperature: float = 0.0, n: int = 1) -> List[str]:
        """Generate complete solutions (all steps at once)."""
        prompts = [self._build_prompt(q) for q in queries]

        if self.backend == "vllm":
            sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=max_tokens,
                n=n,
            )
            outputs = self.llm.generate(prompts, sampling_params)
            results = []
            for output in outputs:
                for completion in output.outputs:
                    results.append(completion.text)
            return results
        else:
            results = []
            for prompt in prompts:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.hf_model.device)
                for _ in range(n):
                    with torch.no_grad():
                        out = self.hf_model.generate(
                            **inputs,
                            max_new_tokens=max_tokens,
                            do_sample=(temperature > 0),
                            temperature=temperature if temperature > 0 else None,
                        )
                    new_tokens = out[0][inputs["input_ids"].shape[1]:]
                    text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                    results.append(text)
            return results

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenizer.encode(text))


class PRMScorer:
    """Process Reward Model scorer using Qwen2.5-Math-PRM-7B.

    Correct usage per official README:
    - Steps separated by <extra_0> token in assistant response
    - Score = P(positive) at each <extra_0> position
    - Uses AutoModel (not AutoModelForCausalLM)
    """

    def __init__(self, model_path: str, device: str = "cuda:0"):
        print(f"Loading PRM: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        self.model.eval()
        self.device = device
        self.step_sep_id = self.tokenizer.encode("<extra_0>")[0]

    @staticmethod
    def _make_step_rewards(logits: torch.Tensor, token_masks: torch.Tensor) -> List[List[float]]:
        """Extract step-level reward scores from PRM logits.

        For each <extra_0> token position, compute P(positive) from the
        2-class output (negative, positive).
        """
        probabilities = F.softmax(logits, dim=-1)
        probabilities = probabilities * token_masks.unsqueeze(-1)

        all_scores = []
        for i in range(probabilities.size(0)):
            sample = probabilities[i]
            positive_probs = sample[sample != 0].view(-1, 2)[:, 1]
            all_scores.append(positive_probs.cpu().tolist())
        return all_scores

    @torch.no_grad()
    def score_trace(self, query: str, steps: List[str]) -> List[float]:
        """Score all steps in a reasoning trace.

        Args:
            query: the math problem
            steps: list of reasoning step texts

        Returns:
            list of PRM scores (one per step), each in [0, 1]
        """
        if not steps:
            return []

        # Build message in the format expected by Qwen2.5-Math-PRM
        step_separator = "<extra_0>"
        response_text = step_separator.join(steps) + step_separator

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
            {"role": "assistant", "content": response_text},
        ]

        conversation_str = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        input_ids = self.tokenizer.encode(
            conversation_str, return_tensors="pt"
        ).to(self.model.device)

        outputs = self.model(input_ids=input_ids)

        token_masks = (input_ids == self.step_sep_id)
        step_rewards = self._make_step_rewards(outputs[0], token_masks)

        scores = step_rewards[0] if step_rewards else []

        # Pad or truncate to match number of steps
        while len(scores) < len(steps):
            scores.append(0.5)
        scores = scores[:len(steps)]

        return scores

    @torch.no_grad()
    def score_steps_incremental(self, query: str, steps: List[str]) -> List[float]:
        """Score steps incrementally (for online routing).

        Same as score_trace but called after each new step is added.
        Returns scores for all steps so far.
        """
        return self.score_trace(query, steps)


class ServerPRMScorer:
    """PRM scorer backed by a vLLM `/pooling` server.

    This mirrors TRIM's ServerPRM contract but keeps the simple
    `score_trace(query, steps)` interface used by the offline episode pipeline.
    """

    STEP_TOKEN = "<extra_0>"

    def __init__(self, server_url: str, model_name: str,
                 tokenizer_path: str = None, max_workers: int = 4,
                 timeout: int = 300, max_retries: int = 5):
        self.base_url = server_url.rstrip("/")
        self.model_name = model_name
        self.max_workers = max_workers
        self.timeout = timeout
        self.max_retries = max_retries
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path or model_name,
            trust_remote_code=True,
        )

    def _format_text(self, query: str, steps: List[str]) -> str:
        response_text = "".join(f"{step}{self.STEP_TOKEN}" for step in steps)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
            {"role": "assistant", "content": response_text},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

    def _pooling(self, text: str) -> dict:
        delay = 2.0
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    f"{self.base_url}/pooling",
                    json={
                        "model": self.model_name,
                        "input": text,
                        "truncate_prompt_tokens": 4096,
                    },
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(delay)
                delay *= 2.0

    @staticmethod
    def _extract_outputs(payload: dict) -> list:
        data = payload.get("data")
        if not isinstance(data, list) or not data:
            raise RuntimeError(f"Unexpected /pooling response: {payload}")
        item = data[0]
        if isinstance(item.get("data"), list):
            return item["data"]
        outputs = item.get("outputs")
        if isinstance(outputs, dict) and isinstance(outputs.get("data"), list):
            return outputs["data"]
        raise RuntimeError(f"Could not find token outputs in /pooling response: {payload}")

    @staticmethod
    def _positive_prob(output) -> float:
        if isinstance(output, list) and len(output) == 2:
            return float(output[1])
        if isinstance(output, dict):
            for key in ("data", "scores", "probabilities"):
                values = output.get(key)
                if isinstance(values, list) and len(values) == 2:
                    return float(values[1])
        raise RuntimeError(f"Unsupported PRM token output: {output}")

    def score_trace(self, query: str, steps: List[str]) -> List[float]:
        if not steps:
            return []

        payload = self._pooling(self._format_text(query, steps))
        outputs = self._extract_outputs(payload)
        if len(outputs) > len(steps):
            raise RuntimeError(
                f"PRM returned {len(outputs)} scores for {len(steps)} steps"
            )
        scores = [self._positive_prob(output) for output in outputs]
        if len(scores) < len(steps):
            scores = [1.0] * (len(steps) - len(scores)) + scores
        return scores[:len(steps)]

    def score_steps_incremental(self, query: str, steps: List[str]) -> List[float]:
        return self.score_trace(query, steps)


def extract_answer(text: str) -> str:
    """Extract the final answer from solution text.

    Qwen3 produces <think>...</think> blocks; we must extract \boxed{}
    from OUTSIDE the think block (the final answer part) when possible.
    """
    # Strip <think>...</think> to get the final answer portion
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if not clean:
        clean = text

    def _find_boxed(s):
        matches = []
        i = 0
        while True:
            idx = s.find("\\boxed{", i)
            if idx == -1:
                break
            depth = 0
            start = idx + len("\\boxed{")
            for j in range(start, len(s)):
                if s[j] == "{":
                    depth += 1
                elif s[j] == "}":
                    if depth == 0:
                        matches.append(s[start:j])
                        break
                    depth -= 1
            i = idx + 1
        return matches

    # First try the clean (non-think) portion
    boxed = _find_boxed(clean)
    if boxed:
        return boxed[-1].strip()

    # Fall back to full text (last \boxed{})
    boxed = _find_boxed(text)
    if boxed:
        return boxed[-1].strip()

    # Try "answer is X" or "= X" at end of text
    for pattern in [
        r"(?:answer|Answer|ANSWER)\s*(?:is|=|:)\s*(.+?)(?:\.|$)",
        r"(?:final answer|Final Answer|FINAL ANSWER)\s*(?:is|=|:)?\s*(.+?)(?:\.|$)",
        r"=\s*\\?boxed\{([^}]+)\}",
    ]:
        match = re.search(pattern, clean)
        if match:
            return match.group(1).strip()

    # Try to find the last standalone number (works well for AIME integer answers)
    nums = re.findall(r"(?<![a-zA-Z])(-?\d+(?:\.\d+)?)(?![a-zA-Z])", clean)
    if nums:
        return nums[-1]

    lines = [l.strip() for l in clean.split("\n") if l.strip()]
    return lines[-1] if lines else ""


def check_correctness(predicted: str, ground_truth: str) -> bool:
    """Check if predicted answer matches ground truth.

    Handles integers, fractions, tuples, LaTeX formatting differences.
    """
    def normalize(s):
        s = s.strip()
        # Remove common LaTeX wrappers
        s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
        s = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", s)
        s = re.sub(r"\\textbf\{([^}]*)\}", r"\1", s)
        # Remove \left \right \bigl \bigr etc.
        s = re.sub(r"\\(?:left|right|big[lr]?|Big[lr]?|bigg[lr]?|Bigg[lr]?)\b", "", s)
        # Remove units and decorators
        s = re.sub(r"\^\s*\\?circ", "", s)
        s = re.sub(r"\\?degrees?", "", s)
        s = s.replace("$", "").replace(" ", "")
        # Normalize fractions before removing backslashes
        s = re.sub(r"\\(frac|dfrac)\{([^}]*)\}\{([^}]*)\}", r"(\2)/(\3)", s)
        # Remove remaining backslashes
        s = s.replace("\\", "")
        # Remove trailing periods
        s = s.rstrip(".")
        # Try pure integer extraction (critical for AIME)
        int_match = re.match(r"^(-?\d+)$", s)
        if int_match:
            return int_match.group(1)
        # If string contains equations like "x = 149", extract the last number
        nums_in_s = re.findall(r"(-?\d+(?:\.\d+)?)", s)
        if nums_in_s and not re.match(r"^[a-zA-Z(]", s.lstrip("-")):
            pass  # Try eval first
        # Try numeric evaluation
        try:
            val = float(eval(s.replace(",", "").replace("^", "**")))
            if val == int(val):
                return str(int(val))
            return f"{val:.6f}"
        except Exception:
            pass
        try:
            return f"{float(s):.6f}"
        except ValueError:
            pass
        # Last resort: extract the last integer from the string
        if nums_in_s:
            last_num = nums_in_s[-1]
            try:
                val = float(last_num)
                if val == int(val):
                    return str(int(val))
                return f"{val:.6f}"
            except ValueError:
                pass
        return s.lower().replace(",", "")

    return normalize(predicted) == normalize(ground_truth)


def split_steps(text: str) -> List[str]:
    """Split solution text into reasoning steps.

    For Qwen3 thinking mode output (<think>...</think>answer), we extract
    the post-think answer portion and split that into steps. The <think>
    block is treated as a single "thinking" step prepended to the list.
    """
    think_match = re.search(r"<think>(.*?)</think>(.*)", text, re.DOTALL)
    if think_match:
        think_content = think_match.group(1).strip()
        answer_part = think_match.group(2).strip()
        answer_steps = [s.strip() for s in answer_part.split("\n\n") if s.strip()]
        if not answer_steps:
            answer_steps = [answer_part] if answer_part else []
        # Split thinking block into coarse segments (by \n\n within think)
        think_steps = [s.strip() for s in think_content.split("\n\n") if s.strip()]
        all_steps = think_steps + answer_steps
        return all_steps if all_steps else [text.strip()]

    steps = [s.strip() for s in text.split("\n\n") if s.strip()]
    return steps if steps else [text.strip()]

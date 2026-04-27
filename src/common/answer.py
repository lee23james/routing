"""Answer extraction and correctness checking for math problems."""

import re


def extract_answer(text: str) -> str:
    """Extract the final answer from solution text.

    Handles Qwen3 <think>...</think> output — extracts \\boxed{} from
    outside the think block first, falling back to inside.
    """
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

    boxed = _find_boxed(clean)
    if boxed:
        return boxed[-1].strip()
    boxed = _find_boxed(text)
    if boxed:
        return boxed[-1].strip()

    nums = re.findall(r"(?<![a-zA-Z])(-?\d+(?:\.\d+)?)(?![a-zA-Z])", clean)
    if nums:
        return nums[-1]
    lines = [l.strip() for l in clean.split("\n") if l.strip()]
    return lines[-1] if lines else ""


def check_correctness(predicted: str, ground_truth: str) -> bool:
    """Check if predicted answer matches ground truth."""
    def normalize(s):
        s = s.strip()
        s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
        s = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", s)
        s = re.sub(r"\\(?:left|right|big[lr]?|Big[lr]?)\b", "", s)
        s = s.replace("$", "").replace(" ", "")
        s = re.sub(r"\\(frac|dfrac)\{([^}]*)\}\{([^}]*)\}", r"(\2)/(\3)", s)
        s = s.replace("\\", "").rstrip(".")
        m = re.match(r"^(-?\d+)$", s)
        if m:
            return m.group(1)
        try:
            val = float(eval(s.replace(",", "").replace("^", "**")))
            return str(int(val)) if val == int(val) else f"{val:.6f}"
        except Exception:
            pass
        try:
            return f"{float(s):.6f}"
        except ValueError:
            pass
        nums = re.findall(r"(-?\d+(?:\.\d+)?)", s)
        if nums:
            try:
                val = float(nums[-1])
                return str(int(val)) if val == int(val) else f"{val:.6f}"
            except ValueError:
                pass
        return s.lower().replace(",", "")

    return normalize(predicted) == normalize(ground_truth)

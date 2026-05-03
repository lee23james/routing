#!/usr/bin/env python3
"""Offline TRIM-Agg quick-probe evaluation for pre-generated episodes."""

from __future__ import annotations

import argparse
import json
import sys
from decimal import Decimal
from pathlib import Path
from typing import Iterable

import torch


SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config import ACTION_DIM, HIDDEN_DIM, STATE_DIM
from router.env import TRIMEnv
from router.policy import RouterPolicy


def format_lam_tag(value: str | float) -> str:
    dec = Decimal(str(value)).normalize()
    mantissa, exp = f"{dec:.15E}".split("E")
    mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa}e{int(exp)}"


def parse_lam_values(raw: str) -> list[str]:
    values = [part.strip() for part in raw.replace(",", " ").split()]
    tags = [format_lam_tag(value) for value in values if value]
    if not tags:
        raise ValueError("empty lambda grid")
    return tags


def parse_targets(raw: str) -> list[float]:
    targets = [float(part.strip()) for part in raw.replace(",", " ").split() if part.strip()]
    if not targets:
        raise ValueError("empty target CPT list")
    return targets


def make_threshold_grid(step: float = 0.01) -> list[float]:
    if step <= 0 or step > 1:
        raise ValueError("threshold step must be in (0, 1]")
    n_steps = int(round(1.0 / step))
    grid = {0.0, 1.0}
    for idx in range(n_steps + 1):
        value = round(idx * step, 10)
        if 0.0 <= value <= 1.0:
            grid.add(value)
    return sorted(grid)


def optional_limit(raw: str | int | None) -> int | None:
    if raw is None:
        return None
    if isinstance(raw, int):
        return raw
    if raw.lower() in {"all", "none", "full"}:
        return None
    value = int(raw)
    if value <= 0:
        raise ValueError("limit must be positive, or all/none/full")
    return value


def episode_lrm_total_tokens(episode: dict) -> float:
    if episode.get("lrm_total_tokens") is not None:
        return float(episode.get("lrm_total_tokens", 0.0))
    return float(sum(episode.get("lrm_token_counts", [])))


def load_episodes(path: str | Path, limit: int | None = None) -> list[dict]:
    episodes: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            episode = json.loads(line)
            if not (episode.get("srm_steps") and episode.get("lrm_steps")):
                continue
            episodes.append(episode)
            if limit is not None and len(episodes) >= limit:
                break
    if not episodes:
        raise ValueError(f"no usable episodes found in {path}")
    return episodes


def compute_baselines(episodes: list[dict]) -> dict:
    n = len(episodes)
    total_lrm_tokens = sum(episode_lrm_total_tokens(ep) for ep in episodes)
    srm_correct = sum(bool(ep.get("srm_correct")) for ep in episodes)
    lrm_correct = sum(bool(ep.get("lrm_correct")) for ep in episodes)
    return {
        "n": n,
        "srm_correct": srm_correct,
        "lrm_correct": lrm_correct,
        "srm_acc": srm_correct / max(n, 1),
        "lrm_acc": lrm_correct / max(n, 1),
        "total_lrm_tokens": total_lrm_tokens,
        "lrm_avg_tokens": total_lrm_tokens / max(n, 1),
    }


def add_derived_metrics(row: dict, baselines: dict) -> dict:
    result = dict(row)
    srm_acc = float(baselines["srm_acc"])
    lrm_acc = float(baselines["lrm_acc"])

    if "cpt" not in result:
        total_lrm_budget = float(result.get("total_lrm_budget", 0.0))
        result["cpt"] = (
            float(result.get("total_lrm_used", 0.0)) / total_lrm_budget
            if total_lrm_budget > 0
            else 0.0
        )

    acc = float(result.get("accuracy", 0.0))
    cpt = float(result.get("cpt", 0.0))
    gap = lrm_acc - srm_acc
    result["ibc"] = (acc - srm_acc) / cpt if cpt > 0 else 0.0
    result["pgr"] = (acc - srm_acc) / gap if gap > 0 else 0.0
    result["srm_acc"] = srm_acc
    result["lrm_acc"] = lrm_acc
    return result


def _load_policy(checkpoint_path: str | Path, device: str) -> RouterPolicy:
    checkpoint_path = Path(checkpoint_path)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    policy = RouterPolicy(STATE_DIM, HIDDEN_DIM, ACTION_DIM).to(device)
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy


def evaluate_checkpoint(
    episodes: list[dict],
    checkpoint_path: str | Path,
    device: str = "cpu",
    max_steps: int = 30,
    include_per_problem: bool = False,
) -> dict:
    env = TRIMEnv.__new__(TRIMEnv)
    env.max_steps = max_steps
    env.episodes = episodes
    env.rubric_weights = None
    env._reset_state()

    policy = _load_policy(checkpoint_path, device)

    correct = 0
    total_lrm_used = 0.0
    total_regens = 0
    total_steps = 0
    per_problem = []

    for idx, episode in enumerate(episodes):
        state = env.reset(idx)
        done = False
        while not done:
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                action, _, _ = policy.get_action(state_t, deterministic=True)
            state, _, done, _ = env.step(int(action.item()))

        is_correct = bool(env._is_correct())
        info = env.get_episode_info()
        correct += int(is_correct)
        total_lrm_used += float(info["total_lrm_tokens"])
        total_regens += int(info["num_regens"])
        total_steps += int(info["num_steps"])

        if include_per_problem:
            per_problem.append(
                {
                    "id": episode.get("id", ""),
                    "router_correct": is_correct,
                    "srm_correct": bool(episode.get("srm_correct")),
                    "lrm_correct": bool(episode.get("lrm_correct")),
                    "actions": info["actions"],
                    "num_regens": info["num_regens"],
                    "num_steps": info["num_steps"],
                    "total_lrm_tokens": info["total_lrm_tokens"],
                }
            )

    n = len(episodes)
    total_lrm_budget = sum(episode_lrm_total_tokens(ep) for ep in episodes)
    result = {
        "checkpoint": str(checkpoint_path),
        "accuracy": correct / max(n, 1),
        "num_correct": correct,
        "total": n,
        "total_lrm_used": total_lrm_used,
        "total_lrm_budget": total_lrm_budget,
        "avg_lrm_tokens": total_lrm_used / max(n, 1),
        "avg_lrm_budget_tokens": total_lrm_budget / max(n, 1),
        "avg_regens": total_regens / max(n, 1),
        "avg_steps": total_steps / max(n, 1),
        "regen_ratio": total_regens / max(total_steps, 1),
    }
    if include_per_problem:
        result["per_problem"] = per_problem
    return result


def evaluate_checkpoint_threshold(
    episodes: list[dict],
    checkpoint_path: str | Path,
    threshold: float,
    device: str = "cpu",
    max_steps: int = 30,
) -> dict:
    env = TRIMEnv.__new__(TRIMEnv)
    env.max_steps = max_steps
    env.episodes = episodes
    env.rubric_weights = None
    env._reset_state()

    policy = _load_policy(checkpoint_path, device)

    correct = 0
    total_lrm_used = 0.0
    total_regens = 0
    total_steps = 0

    for idx in range(len(episodes)):
        state = env.reset(idx)
        done = False
        while not done:
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                h = policy.encoder(state_t)
                logits = policy.actor(h)
                probs = torch.softmax(logits, dim=-1)
                p_escalate = float(probs[0, 1].item())
            action = 1 if p_escalate >= threshold else 0
            state, _, done, _ = env.step(action)

        info = env.get_episode_info()
        correct += int(env._is_correct())
        total_lrm_used += float(info["total_lrm_tokens"])
        total_regens += int(info["num_regens"])
        total_steps += int(info["num_steps"])

    n = len(episodes)
    total_lrm_budget = sum(episode_lrm_total_tokens(ep) for ep in episodes)
    return {
        "checkpoint": str(checkpoint_path),
        "threshold": threshold,
        "accuracy": correct / max(n, 1),
        "num_correct": correct,
        "total": n,
        "total_lrm_used": total_lrm_used,
        "total_lrm_budget": total_lrm_budget,
        "avg_lrm_tokens": total_lrm_used / max(n, 1),
        "avg_lrm_budget_tokens": total_lrm_budget / max(n, 1),
        "avg_regens": total_regens / max(n, 1),
        "avg_steps": total_steps / max(n, 1),
        "regen_ratio": total_regens / max(total_steps, 1),
    }


def sweep_checkpoint_thresholds(
    episodes: list[dict],
    checkpoint_path: str | Path,
    thresholds: Iterable[float],
    baselines: dict,
    lam: str | None = None,
    device: str = "cpu",
    max_steps: int = 30,
) -> list[dict]:
    rows = []
    lam_tag = format_lam_tag(lam) if lam is not None else format_lam_tag(Path(checkpoint_path).parent.name)
    for threshold in thresholds:
        row = evaluate_checkpoint_threshold(
            episodes,
            checkpoint_path,
            threshold=float(threshold),
            device=device,
            max_steps=max_steps,
        )
        row["lam"] = lam_tag
        row["status"] = "done"
        rows.append(add_derived_metrics(row, baselines))
    return rows


def pick_nearest_targets(rows: Iterable[dict], targets: Iterable[float]) -> dict:
    usable = [
        row for row in rows
        if row.get("status", "done") == "done" and row.get("cpt") is not None
    ]
    picked = {}
    for target in targets:
        label = f"CPT{int(round(target * 100))}"
        if not usable:
            picked[label] = None
            continue
        best = min(
            usable,
            key=lambda row: (
                abs(float(row["cpt"]) - target),
                -float(row.get("accuracy", 0.0)),
            ),
        )
        selected = dict(best)
        selected["target_cpt"] = target
        selected["abs_cpt_error"] = abs(float(best["cpt"]) - target)
        picked[label] = selected
    return picked


def collect_summary_rows(
    results_dir: str | Path,
    lam_values: Iterable[str],
    baselines: dict,
) -> list[dict]:
    results_dir = Path(results_dir)
    rows: list[dict] = []
    for lam in lam_values:
        tag = format_lam_tag(lam)
        path = results_dir / tag / "eval.json"
        if not path.exists():
            rows.append({"lam": tag, "status": "missing", "path": str(path)})
            continue
        with path.open("r", encoding="utf-8") as f:
            row = json.load(f)
        row["lam"] = tag
        row["status"] = "done"
        rows.append(add_derived_metrics(row, baselines))
    return rows


def write_json(path: str | Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_markdown_summary(
    path: str | Path,
    rows: list[dict],
    picked: dict,
    baselines: dict,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# TRIM-Agg AIME Quick Probe",
        "",
        (
            f"Baseline: n={baselines['n']}, "
            f"SRM={baselines['srm_acc']:.4f} ({baselines['srm_correct']}/{baselines['n']}), "
            f"LRM={baselines['lrm_acc']:.4f} ({baselines['lrm_correct']}/{baselines['n']}), "
            f"LRM avg tokens={baselines['lrm_avg_tokens']:.1f}"
        ),
        "",
        "| Lambda | Status | Acc | Correct | CPT | PGR | IBC | Avg LRM Tok | Regen |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        if row.get("status") != "done":
            lines.append(f"| {row['lam']} | missing |  |  |  |  |  |  |  |")
            continue
        lines.append(
            "| {lam} | done | {acc:.4f} | {correct}/{total} | {cpt:.4f} | "
            "{pgr:.4f} | {ibc:.4f} | {avg_lrm:.1f} | {regen:.4f} |".format(
                lam=row["lam"],
                acc=row["accuracy"],
                correct=row["num_correct"],
                total=row["total"],
                cpt=row["cpt"],
                pgr=row["pgr"],
                ibc=row["ibc"],
                avg_lrm=row["avg_lrm_tokens"],
                regen=row["regen_ratio"],
            )
        )

    lines.extend(
        [
            "",
            "## Nearest Target Picks",
            "",
            "| Target | Lambda | Acc | CPT | Abs Error | PGR |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for label, row in picked.items():
        if row is None:
            lines.append(f"| {label} | missing |  |  |  |  |")
            continue
        lines.append(
            "| {label} | {lam} | {acc:.4f} | {cpt:.4f} | {err:.4f} | {pgr:.4f} |".format(
                label=label,
                lam=row["lam"],
                acc=row["accuracy"],
                cpt=row["cpt"],
                err=row["abs_cpt_error"],
                pgr=row["pgr"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_threshold_markdown_summary(
    path: str | Path,
    rows: list[dict],
    picked: dict,
    baselines: dict,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# TRIM-Agg AIME Probability Sweep",
        "",
        (
            f"Baseline: n={baselines['n']}, "
            f"SRM={baselines['srm_acc']:.4f} ({baselines['srm_correct']}/{baselines['n']}), "
            f"LRM={baselines['lrm_acc']:.4f} ({baselines['lrm_correct']}/{baselines['n']}), "
            f"LRM avg tokens={baselines['lrm_avg_tokens']:.1f}"
        ),
        "",
        "| Target | Lambda | Threshold | Acc | Correct | CPT | Abs Error | PGR | Avg LRM Tok |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, row in picked.items():
        if row is None:
            lines.append(f"| {label} | missing |  |  |  |  |  |  |  |")
            continue
        lines.append(
            "| {label} | {lam} | {threshold:.4f} | {acc:.4f} | {correct}/{total} | "
            "{cpt:.4f} | {err:.4f} | {pgr:.4f} | {avg_lrm:.1f} |".format(
                label=label,
                lam=row["lam"],
                threshold=row["threshold"],
                acc=row["accuracy"],
                correct=row["num_correct"],
                total=row["total"],
                cpt=row["cpt"],
                err=row["abs_cpt_error"],
                pgr=row["pgr"],
                avg_lrm=row["avg_lrm_tokens"],
            )
        )

    lines.extend(
        [
            "",
            "## Sweep Rows",
            "",
            "| Lambda | Threshold | Acc | CPT | PGR | Avg LRM Tok | Regen |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(rows, key=lambda r: (r["lam"], r["threshold"])):
        lines.append(
            "| {lam} | {threshold:.4f} | {acc:.4f} | {cpt:.4f} | "
            "{pgr:.4f} | {avg_lrm:.1f} | {regen:.4f} |".format(
                lam=row["lam"],
                threshold=row["threshold"],
                acc=row["accuracy"],
                cpt=row["cpt"],
                pgr=row["pgr"],
                avg_lrm=row["avg_lrm_tokens"],
                regen=row["regen_ratio"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_rows(rows: list[dict]) -> None:
    for row in rows:
        if row.get("status") != "done":
            print(f"{row['lam']:>7}  missing")
            continue
        print(
            f"{row['lam']:>7}  acc={row['accuracy']:.4f}  "
            f"CPT={row['cpt']:.4f}  PGR={row['pgr']:.4f}  "
            f"avg_lrm_tok={row['avg_lrm_tokens']:.1f}  "
            f"regen={row['regen_ratio']:.4f}"
        )


def run_eval(args: argparse.Namespace) -> None:
    limit = optional_limit(args.limit)
    episodes = load_episodes(args.episodes_path, limit=limit)
    baselines = compute_baselines(episodes)
    row = evaluate_checkpoint(
        episodes,
        args.checkpoint,
        device=args.device,
        max_steps=args.max_steps,
        include_per_problem=args.include_per_problem,
    )
    row["lam"] = format_lam_tag(args.lam) if args.lam else format_lam_tag(Path(args.checkpoint).parent.name)
    row["status"] = "done"
    row = add_derived_metrics(row, baselines)
    row["baselines"] = baselines
    write_json(args.out, row)
    print_rows([row])


def run_summary(args: argparse.Namespace) -> None:
    limit = optional_limit(args.limit)
    episodes = load_episodes(args.episodes_path, limit=limit)
    baselines = compute_baselines(episodes)
    rows = collect_summary_rows(args.results_dir, parse_lam_values(args.lam_values), baselines)
    picked = pick_nearest_targets(rows, parse_targets(args.targets))
    write_jsonl(args.out_jsonl, rows)
    write_json(args.out_targets, {"baselines": baselines, "targets": picked})
    write_markdown_summary(args.out_md, rows, picked, baselines)
    print_rows(rows)
    print(f"summary_jsonl={args.out_jsonl}")
    print(f"summary_md={args.out_md}")
    print(f"targets_json={args.out_targets}")


def run_sweep(args: argparse.Namespace) -> None:
    limit = optional_limit(args.limit)
    episodes = load_episodes(args.episodes_path, limit=limit)
    baselines = compute_baselines(episodes)
    thresholds = make_threshold_grid(args.threshold_step)
    rows = sweep_checkpoint_thresholds(
        episodes,
        args.checkpoint,
        thresholds,
        baselines,
        lam=args.lam,
        device=args.device,
        max_steps=args.max_steps,
    )
    picked = pick_nearest_targets(rows, parse_targets(args.targets))
    write_jsonl(args.out_jsonl, rows)
    write_json(args.out_targets, {"baselines": baselines, "targets": picked})
    write_threshold_markdown_summary(args.out_md, rows, picked, baselines)
    print_rows(rows)
    print(f"sweep_jsonl={args.out_jsonl}")
    print(f"sweep_md={args.out_md}")
    print(f"targets_json={args.out_targets}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline TRIM-Agg quick-probe evaluation")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--episodes_path", type=Path, required=True)
    eval_parser.add_argument("--checkpoint", type=Path, required=True)
    eval_parser.add_argument("--out", type=Path, required=True)
    eval_parser.add_argument("--lam", default=None)
    eval_parser.add_argument("--device", default="cpu")
    eval_parser.add_argument("--max_steps", type=int, default=30)
    eval_parser.add_argument("--limit", default="all")
    eval_parser.add_argument("--include_per_problem", action="store_true")

    summary_parser = subparsers.add_parser("summary")
    summary_parser.add_argument("--episodes_path", type=Path, required=True)
    summary_parser.add_argument("--results_dir", type=Path, required=True)
    summary_parser.add_argument("--lam_values", required=True)
    summary_parser.add_argument("--targets", default="0.50,0.80,0.95")
    summary_parser.add_argument("--limit", default="all")
    summary_parser.add_argument("--out_jsonl", type=Path, required=True)
    summary_parser.add_argument("--out_md", type=Path, required=True)
    summary_parser.add_argument("--out_targets", type=Path, required=True)

    sweep_parser = subparsers.add_parser("sweep")
    sweep_parser.add_argument("--episodes_path", type=Path, required=True)
    sweep_parser.add_argument("--checkpoint", type=Path, required=True)
    sweep_parser.add_argument("--lam", default=None)
    sweep_parser.add_argument("--targets", default="0.50,0.80,0.95")
    sweep_parser.add_argument("--threshold_step", type=float, default=0.01)
    sweep_parser.add_argument("--device", default="cpu")
    sweep_parser.add_argument("--max_steps", type=int, default=30)
    sweep_parser.add_argument("--limit", default="all")
    sweep_parser.add_argument("--out_jsonl", type=Path, required=True)
    sweep_parser.add_argument("--out_md", type=Path, required=True)
    sweep_parser.add_argument("--out_targets", type=Path, required=True)

    args = parser.parse_args()
    if args.cmd == "eval":
        run_eval(args)
    elif args.cmd == "summary":
        run_summary(args)
    elif args.cmd == "sweep":
        run_sweep(args)


if __name__ == "__main__":
    main()

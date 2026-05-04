import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.plot_trim_agg_baseline import (
    _routing_flops,
    build_main_results,
    compute_baselines,
    find_acc_at_flops,
    find_flops_at_acc,
    load_episode_groups,
    pareto_envelope,
    parse_checkpoint_metadata,
    parse_dataset_names,
    select_even_accuracy_points,
)
from eval.flops_eval import compute_episode_flops


class PlotTrimAggBaselineTest(unittest.TestCase):
    def test_pareto_envelope_keeps_monotonic_accuracy_frontier(self):
        points = [
            {"method": "bad-low", "avg_flops_tflops": 1.0, "accuracy": 60.0},
            {"method": "good-low", "avg_flops_tflops": 1.0, "accuracy": 65.0},
            {"method": "dominated", "avg_flops_tflops": 2.0, "accuracy": 64.0},
            {"method": "frontier", "avg_flops_tflops": 3.0, "accuracy": 70.0},
        ]

        front = pareto_envelope(points)

        self.assertEqual(
            [(p["method"], p["avg_flops_tflops"], p["accuracy"]) for p in front],
            [
                ("good-low", 1.0, 65.0),
                ("frontier", 3.0, 70.0),
            ],
        )

    def test_metric_interpolation_uses_flops_sorted_curve(self):
        curve = [
            {"avg_flops_tflops": 3.0, "accuracy": 90.0},
            {"avg_flops_tflops": 1.0, "accuracy": 70.0},
            {"avg_flops_tflops": 2.0, "accuracy": 80.0},
        ]

        self.assertEqual(find_acc_at_flops(curve, 1.5), 75.0)
        self.assertEqual(find_flops_at_acc(curve, 85.0), 3.0)

    def test_build_main_results_marks_best_and_second_best(self):
        plot_data = {
            "datasets": ["toy"],
            "baselines": {
                "toy": {
                    "srm_acc": 50.0,
                    "lrm_acc": 80.0,
                    "srm_flops": 1.0,
                    "lrm_flops": 10.0,
                    "n": 4,
                }
            },
            "random_curves": {
                "toy": [
                    {"avg_flops_tflops": 1.0, "accuracy": 50.0},
                    {"avg_flops_tflops": 6.0, "accuracy": 68.0},
                    {"avg_flops_tflops": 10.0, "accuracy": 80.0},
                ]
            },
            "ppo_curves": {
                "toy": {
                    "ppo_agg": [
                        {"avg_flops_tflops": 1.0, "accuracy": 50.0},
                        {"avg_flops_tflops": 6.0, "accuracy": 75.0},
                        {"avg_flops_tflops": 9.0, "accuracy": 79.0},
                    ]
                }
            },
        }

        result = build_main_results(plot_data)

        rows = result["rows"]
        ppo = next(r for r in rows if r["method"] == "TRIM-Agg (PPO)")
        random = next(r for r in rows if r["method"] == "Random Routing")
        lrm = next(r for r in rows if r["method"] == "LRM-Only")

        self.assertEqual(ppo["metrics"]["toy"]["acc_at_60"], 75.0)
        self.assertEqual(ppo["metrics"]["toy"]["flops_at_98_pct"], 90.0)
        self.assertEqual(ppo["metrics"]["toy"]["acc_rank"], "best")
        self.assertEqual(random["metrics"]["toy"]["acc_rank"], "second")
        self.assertEqual(ppo["metrics"]["toy"]["flops_rank"], "best")
        self.assertEqual(lrm["metrics"]["toy"]["flops_rank"], "second")

    def test_routing_flops_anchors_use_same_stepwise_endpoint_cost_as_policy_curve(self):
        ep = {
            "srm_steps": ["a", "b", "c"],
            "lrm_steps": ["A", "B", "C"],
            "srm_token_counts": [2, 3, 5],
            "lrm_token_counts": [7, 11, 13],
            "srm_total_tokens": 1000,
            "lrm_total_tokens": 2000,
            "srm_correct": False,
            "lrm_correct": True,
        }

        srm_actions = [0, 0, 0]
        lrm_actions = [1, 1, 1]

        self.assertEqual(_routing_flops(ep, srm_actions), compute_episode_flops(ep, srm_actions))
        self.assertEqual(_routing_flops(ep, lrm_actions), compute_episode_flops(ep, lrm_actions))

        baselines = compute_baselines({"math500": [ep], "aime2025": [ep], "all": [ep]})
        self.assertEqual(baselines["math500"]["srm_flops"], compute_episode_flops(ep, srm_actions) / 1e12)
        self.assertEqual(baselines["math500"]["lrm_flops"], compute_episode_flops(ep, lrm_actions) / 1e12)

    def test_select_even_accuracy_points_prefers_nearest_then_lower_flops(self):
        baseline = {"srm_acc": 50.0, "lrm_acc": 90.0, "lrm_flops": 10.0}
        points = [
            {"accuracy": 59.0, "avg_flops_tflops": 4.0, "regen_ratio": 0.4, "checkpoint": "slow"},
            {"accuracy": 61.0, "avg_flops_tflops": 2.0, "regen_ratio": 0.2, "checkpoint": "fast"},
            {"accuracy": 70.0, "avg_flops_tflops": 3.0, "regen_ratio": 0.3, "checkpoint": "middle"},
            {"accuracy": 81.0, "avg_flops_tflops": 5.0, "regen_ratio": 0.5, "checkpoint": "high"},
        ]

        selected = select_even_accuracy_points("toy", baseline, points, n_targets=3)

        self.assertEqual([round(row["target_acc"], 1) for row in selected["points"]], [60.0, 70.0, 80.0])
        self.assertEqual(selected["points"][0]["checkpoint"], "fast")
        self.assertEqual(selected["points"][0]["pct_lrm_flops"], 20.0)
        self.assertFalse(selected["limited_by_accuracy_granularity"])

    def test_select_even_accuracy_points_deduplicates_when_granularity_is_limited(self):
        baseline = {"srm_acc": 10.0, "lrm_acc": 26.6666667, "lrm_flops": 50.0}
        points = [
            {"accuracy": 13.3333333, "avg_flops_tflops": 5.0, "regen_ratio": 0.1, "checkpoint": "a"},
            {"accuracy": 13.3333333, "avg_flops_tflops": 6.0, "regen_ratio": 0.2, "checkpoint": "b"},
            {"accuracy": 26.6666667, "avg_flops_tflops": 50.0, "regen_ratio": 1.0, "checkpoint": "c"},
        ]

        selected = select_even_accuracy_points("aime2025", baseline, points, n_targets=8)

        self.assertEqual(len(selected["points"]), 2)
        self.assertTrue(selected["limited_by_accuracy_granularity"])
        self.assertEqual(selected["points"][0]["checkpoint"], "a")

    def test_select_even_accuracy_points_keeps_sparse_points_ordered_within_baseline_range(self):
        baseline = {"srm_acc": 10.0, "lrm_acc": 26.6666667, "lrm_flops": 50.0}
        points = [
            {"accuracy": 10.0, "avg_flops_tflops": 1.0, "regen_ratio": 0.0, "checkpoint": "srm"},
            {"accuracy": 13.3333333, "avg_flops_tflops": 2.0, "regen_ratio": 0.1, "checkpoint": "a"},
            {"accuracy": 16.6666667, "avg_flops_tflops": 3.0, "regen_ratio": 0.2, "checkpoint": "b"},
            {"accuracy": 20.0, "avg_flops_tflops": 4.0, "regen_ratio": 0.3, "checkpoint": "c"},
            {"accuracy": 23.3333333, "avg_flops_tflops": 5.0, "regen_ratio": 0.4, "checkpoint": "d"},
            {"accuracy": 26.6666667, "avg_flops_tflops": 6.0, "regen_ratio": 0.5, "checkpoint": "lrm"},
            {"accuracy": 30.0, "avg_flops_tflops": 7.0, "regen_ratio": 0.6, "checkpoint": "above-lrm"},
        ]

        selected = select_even_accuracy_points("aime2025", baseline, points, n_targets=8)

        actuals = [row["actual_acc"] for row in selected["points"]]
        self.assertEqual(actuals, sorted(actuals))
        self.assertLessEqual(max(actuals), baseline["lrm_acc"])
        self.assertEqual(selected["points"][-1]["checkpoint"], "lrm")

    def test_parse_checkpoint_metadata_from_point_search_paths(self):
        meta = parse_checkpoint_metadata(
            "/tmp/checkpoints/trim_agg_point_search_lam1e-5_seed3/epoch_0040.pt"
        )

        self.assertEqual(meta["lambda"], 1e-5)
        self.assertEqual(meta["seed"], 3)
        self.assertEqual(meta["checkpoint_kind"], "epoch")
        self.assertEqual(meta["epoch"], 40)

    def test_parse_dataset_names_accepts_math_only_subset(self):
        self.assertEqual(parse_dataset_names("math500"), ["math500"])
        self.assertEqual(parse_dataset_names("math500,aime2025"), ["math500", "aime2025", "all"])

    def test_load_episode_groups_does_not_require_aime_for_math_only(self):
        math_path = Path(self.id()).with_suffix(".jsonl")
        math_path.write_text(
            '{"id": "m0", "srm_steps": ["a"], "lrm_steps": ["b"], '
            '"srm_token_counts": [1], "lrm_token_counts": [2]}\n',
            encoding="utf-8",
        )
        try:
            groups = load_episode_groups({"math500": math_path}, ["math500"])
        finally:
            math_path.unlink(missing_ok=True)

        self.assertEqual(list(groups), ["math500"])
        self.assertEqual(groups["math500"][0]["id"], "m0")


if __name__ == "__main__":
    unittest.main()

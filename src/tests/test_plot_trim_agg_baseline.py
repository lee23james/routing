import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.plot_trim_agg_baseline import (
    _routing_flops,
    build_method_60_98_summary,
    build_plot_data,
    build_trim_agg_60_98_summary,
    build_main_results,
    compute_baselines,
    find_acc_at_flops,
    find_flops_at_acc,
    load_episode_groups,
    pareto_envelope,
    parse_checkpoint_metadata,
    parse_dataset_names,
    render_trim_agg_60_98_markdown,
    select_even_accuracy_points,
    write_outputs,
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

    def test_build_main_results_includes_trim_rubric_when_curve_is_available(self):
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
                    ],
                    "ppo_rubric": [
                        {"avg_flops_tflops": 1.0, "accuracy": 50.0},
                        {"avg_flops_tflops": 6.0, "accuracy": 77.0},
                        {"avg_flops_tflops": 8.0, "accuracy": 79.0},
                    ],
                }
            },
        }

        result = build_main_results(plot_data)

        rows = result["rows"]
        rubric = next(r for r in rows if r["method"] == "TRIM-Rubric (PPO)")
        agg = next(r for r in rows if r["method"] == "TRIM-Agg (PPO)")

        self.assertEqual(rubric["metrics"]["toy"]["acc_at_60"], 77.0)
        self.assertEqual(rubric["metrics"]["toy"]["flops_at_98_pct"], 80.0)
        self.assertEqual(rubric["metrics"]["toy"]["acc_rank"], "best")
        self.assertEqual(agg["metrics"]["toy"]["acc_rank"], "second")
        self.assertEqual(rubric["metrics"]["toy"]["flops_rank"], "best")
        self.assertEqual(agg["metrics"]["toy"]["flops_rank"], "second")

    def test_build_trim_agg_60_98_summary_uses_main_results_metrics(self):
        plot_data = {
            "datasets": ["math500"],
            "baselines": {
                "math500": {
                    "srm_acc": 50.0,
                    "lrm_acc": 80.0,
                    "srm_flops": 1.0,
                    "lrm_flops": 10.0,
                    "n": 169,
                }
            },
            "main_results": {
                "rows": [
                    {
                        "method": "TRIM-Agg (PPO)",
                        "metrics": {
                            "math500": {
                                "acc_at_60": 75.0,
                                "flops_at_98_tflops": 9.0,
                                "flops_at_98_pct": 90.0,
                            }
                        },
                    }
                ]
            },
        }

        summary = build_trim_agg_60_98_summary(plot_data, "math500")

        self.assertEqual(summary["dataset"], "math500")
        self.assertEqual(summary["method"], "TRIM-Agg (PPO)")
        self.assertEqual(summary["n"], 169)
        self.assertEqual(summary["acc_at_60_lrm_flops"], 75.0)
        self.assertEqual(summary["flops_at_98_lrm_acc_tflops"], 9.0)
        self.assertEqual(summary["flops_at_98_lrm_acc_pct_lrm"], 90.0)
        self.assertEqual(summary["lrm_acc"], 80.0)
        self.assertEqual(summary["lrm_flops_tflops"], 10.0)
        self.assertEqual(summary["target_60_lrm_flops_tflops"], 6.0)
        self.assertEqual(summary["target_98_lrm_acc"], 78.4)

    def test_build_method_60_98_summary_supports_trim_rubric(self):
        plot_data = {
            "datasets": ["math500"],
            "baselines": {
                "math500": {
                    "srm_acc": 50.0,
                    "lrm_acc": 80.0,
                    "srm_flops": 1.0,
                    "lrm_flops": 10.0,
                    "n": 169,
                }
            },
            "main_results": {
                "rows": [
                    {
                        "method": "TRIM-Rubric (PPO)",
                        "metrics": {
                            "math500": {
                                "acc_at_60": 77.0,
                                "flops_at_98_tflops": 8.0,
                                "flops_at_98_pct": 80.0,
                            }
                        },
                    }
                ]
            },
        }

        summary = build_method_60_98_summary(plot_data, "TRIM-Rubric (PPO)", "math500")

        self.assertEqual(summary["method"], "TRIM-Rubric (PPO)")
        self.assertEqual(summary["acc_at_60_lrm_flops"], 77.0)
        self.assertEqual(summary["flops_at_98_lrm_acc_tflops"], 8.0)
        self.assertEqual(summary["flops_at_98_lrm_acc_pct_lrm"], 80.0)

    def test_render_trim_agg_60_98_markdown_marks_unreachable_98_acc(self):
        summary = {
            "dataset": "math500",
            "dataset_label": "MATH-500",
            "method": "TRIM-Agg (PPO)",
            "n": 169,
            "acc_at_60_lrm_flops": 75.0,
            "flops_at_98_lrm_acc_tflops": None,
            "flops_at_98_lrm_acc_pct_lrm": None,
            "lrm_acc": 80.0,
            "lrm_flops_tflops": 10.0,
            "target_60_lrm_flops_tflops": 6.0,
            "target_98_lrm_acc": 78.4,
        }

        text = render_trim_agg_60_98_markdown(summary)

        self.assertIn("TRIM-Agg 60/98 Metrics", text)
        self.assertIn("Acc@60% LRM FLOPs", text)
        self.assertIn("75.0%", text)
        self.assertIn("unreachable in current search", text)

    def test_parse_dataset_names_accepts_gpqa_and_adds_no_all_aggregate(self):
        self.assertEqual(parse_dataset_names("gpqa_diamond_test_100"), ["gpqa_diamond_test_100"])

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

    def test_parse_dataset_names_accepts_aime_part2_only_subset(self):
        self.assertEqual(
            parse_dataset_names("aime_2020_2024_part2_test"),
            ["aime_2020_2024_part2_test"],
        )

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

    def test_load_episode_groups_loads_aime_part2_without_math_or_aime2025(self):
        aime_path = Path(self.id()).with_suffix(".jsonl")
        aime_path.write_text(
            '{"id": "a0", "srm_steps": ["a"], "lrm_steps": ["b"], '
            '"srm_token_counts": [1], "lrm_token_counts": [2]}\n',
            encoding="utf-8",
        )
        try:
            groups = load_episode_groups(
                {"aime_2020_2024_part2_test": aime_path},
                ["aime_2020_2024_part2_test"],
            )
        finally:
            aime_path.unlink(missing_ok=True)

        self.assertEqual(list(groups), ["aime_2020_2024_part2_test"])
        self.assertEqual(groups["aime_2020_2024_part2_test"][0]["id"], "a0")

    def test_build_plot_data_uses_aime_part2_episode_argument(self):
        aime_path = Path(self.id()).with_suffix(".jsonl")
        ckpt_path = Path(self.id()).with_name("epoch_0010.pt")
        aime_path.write_text(
            '{"id": "a0", "srm_steps": ["a"], "lrm_steps": ["b"], '
            '"srm_token_counts": [1], "lrm_token_counts": [2], '
            '"srm_correct": false, "lrm_correct": true}\n',
            encoding="utf-8",
        )
        ckpt_path.write_text("placeholder", encoding="utf-8")

        class Args:
            datasets = "aime_2020_2024_part2_test"
            math500_episodes = "missing_math.jsonl"
            aime2025_episodes = "missing_aime2025.jsonl"
            aime_part2_episodes = str(aime_path)
            checkpoint_glob = str(ckpt_path)
            agg_checkpoint_glob = str(ckpt_path)
            rubric_checkpoint_glob = ""
            device = "cpu"
            n_selected_points = 8

        try:
            with unittest.mock.patch(
                "eval.plot_trim_agg_baseline.evaluate_policy_threshold_curve",
                return_value=[{
                    "checkpoint": "epoch_0010.pt",
                    "checkpoint_file": str(ckpt_path),
                    "checkpoint_dir": "ckpt",
                    "checkpoint_name": "ckpt/epoch_0010.pt",
                    "checkpoint_kind": "epoch",
                    "lambda": 0.0,
                    "seed": 1,
                    "epoch": 10,
                    "threshold": 0.5,
                    "accuracy": 100.0,
                    "avg_flops_tflops": 0.0,
                    "regen_ratio": 1.0,
                    "correct": 1,
                    "n": 1,
                }],
            ):
                plot_data = build_plot_data(Args())
        finally:
            aime_path.unlink(missing_ok=True)
            ckpt_path.unlink(missing_ok=True)

        self.assertEqual(plot_data["datasets"], ["aime_2020_2024_part2_test"])
        self.assertEqual(
            plot_data["source_files"]["aime_2020_2024_part2_test"],
            str(aime_path),
        )
        self.assertEqual(plot_data["baselines"]["aime_2020_2024_part2_test"]["n"], 1)

    def test_write_outputs_writes_60_98_summaries_for_each_requested_dataset(self):
        output_dir = Path(self.id()).with_suffix("")
        plot_data = {
            "datasets": ["aime_2020_2024_part2_test"],
            "baselines": {
                "aime_2020_2024_part2_test": {
                    "srm_acc": 50.0,
                    "lrm_acc": 80.0,
                    "srm_flops": 1.0,
                    "lrm_flops": 10.0,
                    "n": 74,
                }
            },
            "random_curves": {
                "aime_2020_2024_part2_test": [
                    {"avg_flops_tflops": 1.0, "accuracy": 50.0},
                    {"avg_flops_tflops": 10.0, "accuracy": 80.0},
                ]
            },
            "ppo_curves": {
                "aime_2020_2024_part2_test": {
                    "ppo_agg": [
                        {"avg_flops_tflops": 6.0, "accuracy": 75.0},
                        {"avg_flops_tflops": 9.0, "accuracy": 79.0},
                    ],
                    "ppo_rubric": [
                        {"avg_flops_tflops": 6.0, "accuracy": 77.0},
                        {"avg_flops_tflops": 8.0, "accuracy": 79.0},
                    ],
                }
            },
            "raw_ppo_points": {"ppo_agg": {}, "ppo_rubric": {}},
            "selected_points": {"ppo_agg": {}, "ppo_rubric": {}},
            "checkpoint_patterns_by_curve": {},
            "main_results": None,
        }
        plot_data["main_results"] = build_main_results(plot_data)

        try:
            with unittest.mock.patch("eval.plot_trim_agg_baseline._plot_figures"):
                write_outputs(plot_data, output_dir)

            self.assertTrue((output_dir / "trim_aime_2020_2024_part2_test_60_98_compare.md").exists())
            self.assertTrue((output_dir / "trim_agg_aime_2020_2024_part2_test_60_98.md").exists())
            self.assertTrue((output_dir / "trim_rubric_aime_2020_2024_part2_test_60_98.md").exists())
        finally:
            for path in sorted(output_dir.glob("*")) if output_dir.exists() else []:
                path.unlink()
            output_dir.rmdir()


if __name__ == "__main__":
    unittest.main()

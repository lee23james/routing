import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class TrimAggOfflineProbeTests(unittest.TestCase):
    def test_format_lam_tag_keeps_scientific_grid_labels(self):
        from eval.trim_agg_quick_probe import format_lam_tag

        self.assertEqual(format_lam_tag("4e-4"), "4e-4")
        self.assertEqual(format_lam_tag(0.0005), "5e-4")
        self.assertEqual(format_lam_tag("0.0009"), "9e-4")

    def test_load_episodes_and_baseline_respect_limit(self):
        from eval.trim_agg_quick_probe import compute_baselines, load_episodes

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "episodes.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "srm_steps": ["a"],
                                "lrm_steps": ["b"],
                                "srm_correct": True,
                                "lrm_correct": False,
                                "lrm_total_tokens": 100,
                            }
                        ),
                        json.dumps(
                            {
                                "srm_steps": ["a"],
                                "lrm_steps": ["b"],
                                "srm_correct": False,
                                "lrm_correct": True,
                                "lrm_total_tokens": 300,
                            }
                        ),
                        json.dumps(
                            {
                                "srm_steps": ["a"],
                                "lrm_steps": ["b"],
                                "srm_correct": True,
                                "lrm_correct": True,
                                "lrm_total_tokens": 900,
                            }
                        ),
                    ]
                )
                + "\n"
            )

            episodes = load_episodes(path, limit=2)
            baseline = compute_baselines(episodes)

        self.assertEqual(len(episodes), 2)
        self.assertAlmostEqual(baseline["srm_acc"], 0.5)
        self.assertAlmostEqual(baseline["lrm_acc"], 0.5)
        self.assertAlmostEqual(baseline["lrm_avg_tokens"], 200.0)
        self.assertEqual(baseline["total_lrm_tokens"], 400)

    def test_add_derived_metrics_computes_cpt_ibc_and_pgr(self):
        from eval.trim_agg_quick_probe import add_derived_metrics

        row = add_derived_metrics(
            {
                "accuracy": 0.30,
                "total_lrm_used": 250,
                "total_lrm_budget": 1000,
            },
            {"srm_acc": 0.10, "lrm_acc": 0.50},
        )

        self.assertAlmostEqual(row["cpt"], 0.25)
        self.assertAlmostEqual(row["pgr"], 0.50)
        self.assertAlmostEqual(row["ibc"], 0.80)

    def test_pick_nearest_targets_uses_actual_cpt(self):
        from eval.trim_agg_quick_probe import pick_nearest_targets

        rows = [
            {"lam": "4e-4", "cpt": 0.42, "accuracy": 0.20},
            {"lam": "5e-4", "cpt": 0.81, "accuracy": 0.30},
            {"lam": "6e-4", "cpt": 0.91, "accuracy": 0.40},
        ]

        picked = pick_nearest_targets(rows, targets=[0.50, 0.80, 0.95])

        self.assertEqual(picked["CPT50"]["lam"], "4e-4")
        self.assertEqual(picked["CPT80"]["lam"], "5e-4")
        self.assertEqual(picked["CPT95"]["lam"], "6e-4")
        self.assertAlmostEqual(picked["CPT80"]["abs_cpt_error"], 0.01)

    def test_make_threshold_grid_includes_dense_middle_and_edges(self):
        from eval.trim_agg_quick_probe import make_threshold_grid

        grid = make_threshold_grid(0.25)

        self.assertEqual(grid[0], 0.0)
        self.assertEqual(grid[-1], 1.0)
        self.assertIn(0.25, grid)
        self.assertIn(0.5, grid)
        self.assertIn(0.75, grid)
        self.assertEqual(grid, sorted(set(grid)))


if __name__ == "__main__":
    unittest.main()

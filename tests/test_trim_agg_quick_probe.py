import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TRIM_SCRIPTS = REPO_ROOT / "trim" / "TRIM" / "scripts"
if str(TRIM_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(TRIM_SCRIPTS))


class TrimAggQuickProbeTests(unittest.TestCase):
    def test_parse_cost_grid_returns_canonical_tags(self):
        from trim_agg_quick_probe import parse_cost_grid

        self.assertEqual(
            parse_cost_grid("4e-4, 5e-4,6e-4,7e-4,8e-4,9e-4"),
            ["4e-4", "5e-4", "6e-4", "7e-4", "8e-4", "9e-4"],
        )

    def test_metric_paths_match_trim_agg_cost_tags(self):
        from trim_agg_quick_probe import metric_path

        root = Path("/tmp/probe_results")

        self.assertEqual(
            metric_path(root, "6e-4"),
            root / "6e-4" / "eval_metrics.jsonl",
        )

    def test_summary_computes_target_token_fraction_against_lrm_tokens(self):
        from trim_agg_quick_probe import summarize_metrics

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for cost, acc, target_tokens in [
                ("4e-4", 0.20, 100.0),
                ("5e-4", 0.30, 250.0),
            ]:
                out_dir = root / cost
                out_dir.mkdir(parents=True)
                (out_dir / "eval_metrics.jsonl").write_text(
                    json.dumps({
                        "accuracy": acc,
                        "num_correct": int(acc * 100),
                        "avg_target_tokens_per_question": target_tokens,
                    })
                    + "\n"
                )

            rows = summarize_metrics(
                result_root=root,
                costs=["4e-4", "5e-4"],
                srm_acc=0.10,
                lrm_acc=0.50,
                lrm_avg_tokens=1000.0,
            )

        self.assertEqual(rows[0]["cost"], "4e-4")
        self.assertAlmostEqual(rows[0]["pgr"], 0.25)
        self.assertAlmostEqual(rows[0]["cpt"], 0.10)
        self.assertEqual(rows[1]["cost"], "5e-4")
        self.assertAlmostEqual(rows[1]["pgr"], 0.50)
        self.assertAlmostEqual(rows[1]["cpt"], 0.25)

    def test_episode_baseline_respects_eval_limit(self):
        from trim_agg_quick_probe import compute_episode_baseline

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "episodes.jsonl"
            path.write_text(
                "\n".join([
                    json.dumps({"srm_correct": True, "lrm_correct": False, "lrm_total_tokens": 100}),
                    json.dumps({"srm_correct": False, "lrm_correct": True, "lrm_total_tokens": 300}),
                    json.dumps({"srm_correct": True, "lrm_correct": True, "lrm_total_tokens": 900}),
                ])
                + "\n"
            )

            baseline = compute_episode_baseline(path, limit=2)

        self.assertEqual(baseline["n"], 2)
        self.assertAlmostEqual(baseline["srm_acc"], 0.5)
        self.assertAlmostEqual(baseline["lrm_acc"], 0.5)
        self.assertAlmostEqual(baseline["lrm_avg_tokens"], 200.0)


if __name__ == "__main__":
    unittest.main()

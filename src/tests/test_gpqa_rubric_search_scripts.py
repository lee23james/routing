import unittest
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = SRC_ROOT / "scripts"


class GpqaRubricSearchScriptsTest(unittest.TestCase):
    def test_rubric_search_script_uses_gpqa_episodes_and_rubric_weights(self):
        script = SCRIPTS_DIR / "search_trim_rubric_gpqa_points_4gpu.sh"

        text = script.read_text(encoding="utf-8")

        self.assertIn("data/episodes/gpqa_main_train_200_episodes.jsonl", text)
        self.assertIn("data/rubrics/gpqa_main200", text)
        self.assertIn("RUBRIC_WEIGHTS", text)
        self.assertIn("rubric_weights.json", text)
        self.assertIn("trim_rubric_gpqa_main200_point_search", text)
        self.assertIn("--lam_rubric", text)
        self.assertIn("--rubric_weights", text)
        for lam in ["0", "5e-6", "2e-5", "1e-4"]:
            self.assertIn(f'"{lam}"', text)

    def test_final_eval_script_targets_gpqa_and_both_ppo_methods(self):
        script = SCRIPTS_DIR / "eval_gpqa_search_final.sh"

        text = script.read_text(encoding="utf-8")

        self.assertIn("--datasets gpqa_diamond_test_100", text)
        self.assertIn("--gpqa_diamond_episodes", text)
        self.assertIn("trim_agg_gpqa_main200_point_search_*/*.pt", text)
        self.assertIn("trim_rubric_gpqa_main200_point_search_*/*.pt", text)
        self.assertIn("results/trim_gpqa_main200_diamond100_search/final", text)

    def test_pipeline_script_runs_rubric_search_after_agg_and_then_eval(self):
        script = SCRIPTS_DIR / "run_gpqa_rubric_search_pipeline.sh"

        text = script.read_text(encoding="utf-8")

        self.assertIn("gpqa_main_train_200_episodes.jsonl", text)
        self.assertIn("gpqa_diamond_test_100_episodes.jsonl", text)
        self.assertIn("200", text)
        self.assertIn("100", text)
        self.assertIn("search_trim_agg_gpqa_points_4gpu.sh", text)
        self.assertIn("search_trim_rubric_gpqa_points_4gpu.sh", text)
        self.assertIn("eval_gpqa_search_final.sh", text)


if __name__ == "__main__":
    unittest.main()

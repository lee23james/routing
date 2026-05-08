import unittest
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = SRC_ROOT / "scripts"


class RubricV2MathScriptsTest(unittest.TestCase):
    def test_math_rubric_v2_search_script_is_independent_from_v1_outputs(self):
        script = SCRIPTS_DIR / "search_trim_rubric_v2_math_points_4gpu.sh"

        text = script.read_text(encoding="utf-8")

        self.assertIn("data/episodes/math_train_200_episodes.jsonl", text)
        self.assertIn("BASE_RUBRIC_WEIGHTS", text)
        self.assertIn("data/rubrics/math200/rubric_weights.json", text)
        self.assertIn("data/rubrics/math200_v2", text)
        self.assertIn("rubric_weights_v2.json", text)
        self.assertIn("rubric.evolve_rubric_weights", text)
        self.assertIn("trim_rubric_v2_math200_point_search", text)
        self.assertIn("trim_rubric_math200_point_search_lam2e-5_rub0.3_seed1/epoch_0030.pt", text)
        self.assertIn("--lam_rubric", text)
        self.assertIn("--rubric_weights", text)
        for lam in ["0", "5e-6", "2e-5", "1e-4"]:
            self.assertIn(f'"{lam}"', text)

    def test_math_rubric_v2_eval_script_includes_all_three_ppo_methods(self):
        script = SCRIPTS_DIR / "eval_math_rubric_v2_final.sh"

        text = script.read_text(encoding="utf-8")

        self.assertIn("--datasets math500", text)
        self.assertIn("TEST_EPISODES=\"${TEST_EPISODES:-data/episodes/math500_episodes.jsonl}\"", text)
        self.assertIn("--math500_episodes", text)
        self.assertIn("--agg_checkpoint_glob", text)
        self.assertIn("trim_agg_math200_point_search_*/*.pt", text)
        self.assertIn("--rubric_checkpoint_glob", text)
        self.assertIn("trim_rubric_math200_point_search_*/*.pt", text)
        self.assertIn("--rubric_v2_checkpoint_glob", text)
        self.assertIn("trim_rubric_v2_math200_point_search_*/*.pt", text)
        self.assertIn("results/trim_rubric_v2_math200_point_search/final", text)

    def test_math_rubric_v2_pipeline_runs_search_then_eval(self):
        script = SCRIPTS_DIR / "run_trim_rubric_v2_math_quick_search.sh"

        text = script.read_text(encoding="utf-8")

        self.assertIn("math_train_200_episodes.jsonl", text)
        self.assertIn("math500_episodes.jsonl", text)
        self.assertIn("200", text)
        self.assertIn("169", text)
        self.assertIn("search_trim_rubric_v2_math_points_4gpu.sh", text)
        self.assertIn("eval_math_rubric_v2_final.sh", text)


if __name__ == "__main__":
    unittest.main()

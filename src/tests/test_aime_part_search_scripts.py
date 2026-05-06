import unittest
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = SRC_ROOT / "scripts"


class AimePartSearchScriptsTest(unittest.TestCase):
    def test_generation_script_uses_aime_part_train_and_test_with_resume(self):
        script = SCRIPTS_DIR / "generate_aime_part_search_episodes_parallel.sh"

        text = script.read_text(encoding="utf-8")

        self.assertIn("aime_2010_2024_part1_train", text)
        self.assertIn("aime_2020_2024_part2_test", text)
        self.assertIn("--srm_server_url", text)
        self.assertIn("--lrm_server_url", text)
        self.assertNotIn("--no_resume", text)

    def test_agg_search_script_uses_aime_train_episode_file_and_grid(self):
        script = SCRIPTS_DIR / "search_trim_agg_aime_part_points_4gpu.sh"

        text = script.read_text(encoding="utf-8")

        self.assertIn("data/episodes/aime_2010_2024_part1_train_episodes.jsonl", text)
        self.assertIn("trim_agg_aime_part1_204_point_search", text)
        for lam in ["0", "5e-6", "2e-5", "1e-4"]:
            self.assertIn(f'"{lam}"', text)
        self.assertIn("--save_epoch_checkpoints", text)

    def test_rubric_search_script_uses_aime_rubric_weights_and_grid(self):
        script = SCRIPTS_DIR / "search_trim_rubric_aime_part_points_4gpu.sh"

        text = script.read_text(encoding="utf-8")

        self.assertIn("data/rubrics/aime_part1_204/rubric_weights.json", text)
        self.assertIn("trim_rubric_aime_part1_204_point_search", text)
        self.assertIn("--lam_rubric", text)
        self.assertIn("--rubric_weights", text)
        for lam in ["0", "5e-6", "2e-5", "1e-4"]:
            self.assertIn(f'"{lam}"', text)

    def test_final_eval_script_targets_aime_part2_and_both_ppo_methods(self):
        script = SCRIPTS_DIR / "eval_aime_part_search_final.sh"

        text = script.read_text(encoding="utf-8")

        self.assertIn("--datasets aime_2020_2024_part2_test", text)
        self.assertIn("--aime_part2_episodes", text)
        self.assertIn("trim_agg_aime_part1_204_point_search_*/*.pt", text)
        self.assertIn("trim_rubric_aime_part1_204_point_search_*/*.pt", text)
        self.assertIn("results/trim_aime_part1_204_part2_74_search/final", text)

    def test_pipeline_script_waits_for_episodes_then_runs_train_and_eval(self):
        script = SCRIPTS_DIR / "run_aime_part_search_pipeline.sh"

        text = script.read_text(encoding="utf-8")

        self.assertIn("aime_2010_2024_part1_train_episodes.jsonl", text)
        self.assertIn("aime_2020_2024_part2_test_episodes.jsonl", text)
        self.assertIn("204", text)
        self.assertIn("74", text)
        self.assertIn("search_trim_agg_aime_part_points_4gpu.sh", text)
        self.assertIn("search_trim_rubric_aime_part_points_4gpu.sh", text)
        self.assertIn("eval_aime_part_search_final.sh", text)


if __name__ == "__main__":
    unittest.main()

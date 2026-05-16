import unittest
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = SRC_ROOT / "scripts"


class OmniMath13To34SearchScriptsTest(unittest.TestCase):
    def test_generation_script_uses_20k_context_and_parallel_workers(self):
        script = SCRIPTS_DIR / "generate_omnimath13_to34_episodes_20k.sh"

        text = script.read_text(encoding="utf-8")

        self.assertIn("omnimath_diff1_3_train_200", text)
        self.assertIn("omnimath_diff3_4_test_200", text)
        self.assertIn("MAX_MODEL_LEN=\"${MAX_MODEL_LEN:-20480}\"", text)
        self.assertIn("MAX_NEW_TOKENS=\"${MAX_NEW_TOKENS:-20000}\"", text)
        self.assertIn("MAX_NUM_SEQS=\"${MAX_NUM_SEQS:-4}\"", text)
        self.assertIn("GEN_MAX_WORKERS=\"${GEN_MAX_WORKERS:-4}\"", text)
        self.assertIn("CLIENT_TIMEOUT=\"${CLIENT_TIMEOUT:-1800}\"", text)
        self.assertIn("--generation_workers \"$GEN_MAX_WORKERS\"", text)
        self.assertIn("--client_timeout \"$CLIENT_TIMEOUT\"", text)
        self.assertIn("summarize_episode_context.py", text)

    def test_agg_search_script_uses_new_train_episode_file_and_grid(self):
        script = SCRIPTS_DIR / "search_trim_agg_omnimath13_to34_points_4gpu.sh"

        text = script.read_text(encoding="utf-8")

        self.assertIn("data/episodes/omnimath_diff1_3_train_200_episodes.jsonl", text)
        self.assertIn("trim_agg_omnimath13_to34_point_search", text)
        self.assertIn("--save_epoch_checkpoints", text)
        for lam in ["0", "5e-6", "2e-5", "1e-4"]:
            self.assertIn(f'"{lam}"', text)

    def test_rubric_search_script_uses_new_rubric_weights_and_grid(self):
        script = SCRIPTS_DIR / "search_trim_rubric_omnimath13_to34_points_4gpu.sh"

        text = script.read_text(encoding="utf-8")

        self.assertIn("data/episodes/omnimath_diff1_3_train_200_episodes.jsonl", text)
        self.assertIn("data/rubrics/omnimath13_to34", text)
        self.assertIn("rubric_weights.json", text)
        self.assertIn("trim_rubric_omnimath13_to34_point_search", text)
        self.assertIn("--lam_rubric", text)
        self.assertIn("--rubric_weights", text)
        for lam in ["0", "5e-6", "2e-5", "1e-4"]:
            self.assertIn(f'"{lam}"', text)

    def test_rubric_v2b_search_script_uses_router_feedback_weights(self):
        script = SCRIPTS_DIR / "search_trim_rubric_v2b_omnimath13_to34_points_4gpu.sh"

        text = script.read_text(encoding="utf-8")

        self.assertIn("data/episodes/omnimath_diff1_3_train_200_episodes.jsonl", text)
        self.assertIn("data/rubrics/omnimath13_to34_v2b_alpha01_corr005", text)
        self.assertIn("rubric_weights_v2.json", text)
        self.assertIn("trim_rubric_v2b_omnimath13_to34_point_search", text)
        self.assertIn("ALPHA=\"${ALPHA:-0.1}\"", text)
        self.assertIn("CORR_THRESHOLD=\"${CORR_THRESHOLD:-0.05}\"", text)
        self.assertIn("rubric.evolve_rubric_weights", text)
        self.assertIn("--corr_threshold \"$CORR_THRESHOLD\"", text)
        self.assertIn("--rubric_weights \"$RUBRIC_WEIGHTS\"", text)
        for lam in ["0", "5e-6", "2e-5", "1e-4"]:
            self.assertIn(f'"{lam}"', text)

    def test_final_eval_script_targets_omnimath34_and_11_selected_points(self):
        script = SCRIPTS_DIR / "eval_omnimath13_to34_search_final.sh"

        text = script.read_text(encoding="utf-8")

        self.assertIn("--datasets omnimath_diff3_4_test_200", text)
        self.assertIn("--omnimath34_episodes", text)
        self.assertIn("omnimath_diff3_4_test_200_episodes.jsonl", text)
        self.assertIn("trim_agg_omnimath13_to34_point_search_*/*.pt", text)
        self.assertIn("trim_rubric_omnimath13_to34_point_search_*/*.pt", text)
        self.assertIn("trim_rubric_v2b_omnimath13_to34_point_search_*/*.pt", text)
        self.assertIn("--rubric_v2_checkpoint_glob", text)
        self.assertIn("N_SELECTED_POINTS=\"${N_SELECTED_POINTS:-11}\"", text)
        self.assertIn("results/trim_omnimath13_omnimath34_search/final", text)

    def test_pipeline_script_waits_for_episodes_then_runs_train_and_eval(self):
        script = SCRIPTS_DIR / "run_omnimath13_to34_search_pipeline.sh"

        text = script.read_text(encoding="utf-8")

        self.assertIn("omnimath_diff1_3_train_200_episodes.jsonl", text)
        self.assertIn("omnimath_diff3_4_test_200_episodes.jsonl", text)
        self.assertIn("EXPECTED_TRAIN=\"${EXPECTED_TRAIN:-200}\"", text)
        self.assertIn("EXPECTED_TEST=\"${EXPECTED_TEST:-200}\"", text)
        self.assertIn("search_trim_agg_omnimath13_to34_points_4gpu.sh", text)
        self.assertIn("search_trim_rubric_omnimath13_to34_points_4gpu.sh", text)
        self.assertIn("eval_omnimath13_to34_search_final.sh", text)


if __name__ == "__main__":
    unittest.main()

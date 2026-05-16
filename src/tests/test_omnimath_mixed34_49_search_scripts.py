import json
import sys
import unittest
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = SRC_ROOT / "scripts"
sys.path.insert(0, str(SRC_ROOT))


class OmniMathMixed34To49SearchScriptsTest(unittest.TestCase):
    def test_generation_script_uses_omnimath49_stratified_with_20k_context(self):
        script = SCRIPTS_DIR / "generate_omnimath49_test_episodes_20k.sh"

        text = script.read_text(encoding="utf-8")

        self.assertIn("omnimath_diff4_9_stratified_test_100", text)
        self.assertIn("MAX_MODEL_LEN=\"${MAX_MODEL_LEN:-20480}\"", text)
        self.assertIn("MAX_NEW_TOKENS=\"${MAX_NEW_TOKENS:-20000}\"", text)
        self.assertIn("MAX_NUM_SEQS=\"${MAX_NUM_SEQS:-4}\"", text)
        self.assertIn("GEN_MAX_WORKERS=\"${GEN_MAX_WORKERS:-4}\"", text)
        self.assertIn("CLIENT_TIMEOUT=\"${CLIENT_TIMEOUT:-1800}\"", text)
        self.assertIn("logs/trim_omnimath49_generation_20k", text)
        self.assertIn("omnimath49_context_saturation.json", text)
        self.assertIn("--generation_workers \"$GEN_MAX_WORKERS\"", text)
        self.assertIn("--client_timeout \"$CLIENT_TIMEOUT\"", text)
        self.assertIn("summarize_episode_context.py", text)

    def test_build_script_contains_mixed_paths_and_manifest(self):
        script = SCRIPTS_DIR / "build_omnimath_mixed34_49_test.py"

        text = script.read_text(encoding="utf-8")

        self.assertIn("omnimath_diff3_4_test_200_episodes.jsonl", text)
        self.assertIn("omnimath_diff4_9_stratified_test_100_episodes.jsonl", text)
        self.assertIn("omnimath_mixed_3_4_100_4_9_100_test_episodes.jsonl", text)
        self.assertIn("mixed_test_manifest.json", text)
        self.assertIn("mixed_group", text)
        self.assertIn("diff3_4", text)
        self.assertIn("diff4_9", text)

    def test_eval_script_reuses_omnimath13_to34_checkpoints_on_mixed_dataset(self):
        script = SCRIPTS_DIR / "eval_omnimath_mixed34_49_search_final.sh"

        text = script.read_text(encoding="utf-8")

        self.assertIn("--datasets omnimath_mixed_3_4_100_4_9_100_test", text)
        self.assertIn("--omnimath_mixed_episodes", text)
        self.assertIn("omnimath_mixed_3_4_100_4_9_100_test_episodes.jsonl", text)
        self.assertIn("trim_agg_omnimath13_to34_point_search_*/*.pt", text)
        self.assertIn("trim_rubric_omnimath13_to34_point_search_*/*.pt", text)
        self.assertIn("trim_rubric_v2b_omnimath13_to34_point_search_*/*.pt", text)
        self.assertIn("N_SELECTED_POINTS=\"${N_SELECTED_POINTS:-11}\"", text)
        self.assertIn("results/trim_omnimath13_to_mixed34_49_search/final", text)

    def test_pipeline_script_builds_mixed_then_runs_eval_without_training(self):
        script = SCRIPTS_DIR / "run_omnimath_mixed34_49_search_pipeline.sh"

        text = script.read_text(encoding="utf-8")

        self.assertIn("omnimath_diff3_4_test_200_episodes.jsonl", text)
        self.assertIn("omnimath_diff4_9_stratified_test_100_episodes.jsonl", text)
        self.assertIn("omnimath_mixed_3_4_100_4_9_100_test_episodes.jsonl", text)
        self.assertIn("EXPECTED_34=\"${EXPECTED_34:-200}\"", text)
        self.assertIn("EXPECTED_49=\"${EXPECTED_49:-100}\"", text)
        self.assertIn("EXPECTED_MIXED=\"${EXPECTED_MIXED:-200}\"", text)
        self.assertIn("build_omnimath_mixed34_49_test.py", text)
        self.assertIn("eval_omnimath_mixed34_49_search_final.sh", text)
        self.assertNotIn("search_trim_agg_omnimath13_to34_points_4gpu.sh", text)
        self.assertNotIn("search_trim_rubric_omnimath13_to34_points_4gpu.sh", text)

    def test_build_script_samples_100_omnimath34_and_all_100_omnimath49(self):
        from scripts.build_omnimath_mixed34_49_test import build_mixed_episodes

        tmp_dir = Path(self.id()).with_suffix("")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        omni34_path = tmp_dir / "omni34.jsonl"
        omni49_path = tmp_dir / "omni49.jsonl"
        output_path = tmp_dir / "mixed.jsonl"
        manifest_path = tmp_dir / "manifest.json"

        with omni34_path.open("w", encoding="utf-8") as handle:
            for idx in range(120):
                row = {"id": f"34-{idx}", "source_index": idx, "difficulty": 3.2}
                handle.write(json.dumps(row) + "\n")
        with omni49_path.open("w", encoding="utf-8") as handle:
            for idx in range(100):
                difficulty = 4.1 + (idx // 20)
                row = {"id": f"49-{idx}", "source_index": idx, "difficulty": difficulty}
                handle.write(json.dumps(row) + "\n")

        try:
            manifest = build_mixed_episodes(
                omnimath34_episodes=omni34_path,
                omnimath49_episodes=omni49_path,
                output=output_path,
                manifest=manifest_path,
                seed=1,
                n_34=100,
                n_49=100,
            )
            rows = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
                if line
            ]
            written_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        finally:
            for path in [omni34_path, omni49_path, output_path, manifest_path]:
                path.unlink(missing_ok=True)
            tmp_dir.rmdir()

        self.assertEqual(len(rows), 200)
        self.assertEqual(manifest["counts"], {"total": 200, "diff3_4": 100, "diff4_9": 100})
        self.assertEqual(written_manifest["counts"], manifest["counts"])
        self.assertEqual(sum(row["mixed_group"] == "diff3_4" for row in rows), 100)
        self.assertEqual(sum(row["mixed_group"] == "diff4_9" for row in rows), 100)
        self.assertEqual(
            manifest["difficulty_bucket_counts"]["diff4_9"],
            {"[4,5)": 20, "[5,6)": 20, "[6,7)": 20, "[7,8)": 20, "[8,9)": 20},
        )
        self.assertEqual(
            {row["id"] for row in rows if row["mixed_group"] == "diff4_9"},
            {f"49-{idx}" for idx in range(100)},
        )


if __name__ == "__main__":
    unittest.main()

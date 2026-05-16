import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.generate_episodes import generate_model_solutions_parallel, load_items_for_dataset
from data.datasets import (
    GPQA_CACHE_ZIP,
    _extract_gsm8k_final_answer,
    load_gpqa_main_train_200,
    load_gpqa_diamond_test_100,
    load_gsm8k_train_300,
    load_gsm8k_test_189,
    load_aime_2010_2024_part1_train,
    load_aime_2020_2024_part2_test,
    load_omnimath,
    load_omnimath7_9_test_100,
    load_omnimath_diff1_3_train_200,
    load_omnimath_diff3_4_test_200,
    load_omnimath_diff4_9_stratified_test_100,
)


class GenerateEpisodesTest(unittest.TestCase):
    def test_math_train_200_uses_first_200_math_training_items(self):
        items = [{"id": f"math_train_{idx:05d}"} for idx in range(250)]

        with patch("data.generate_episodes.load_math_train", return_value=items):
            selected = load_items_for_dataset("math_train_200")

        self.assertEqual(len(selected), 200)
        self.assertEqual(selected[0]["id"], "math_train_00000")
        self.assertEqual(selected[-1]["id"], "math_train_00199")

    def test_aime_part1_train_loader_uses_2010_2024_part_i_only(self):
        items = load_aime_2010_2024_part1_train()

        self.assertEqual(len(items), 204)
        self.assertTrue(all(item["dataset"] == "aime_2010_2024_part1_train" for item in items))
        self.assertTrue(all(item["split"] == "train" for item in items))
        self.assertTrue(all(2010 <= item["year"] <= 2024 for item in items))
        self.assertEqual({item["part"] for item in items}, {"I"})

    def test_aime_part2_test_loader_reuses_trim_2020_2024_part_ii_split(self):
        items = load_aime_2020_2024_part2_test()

        self.assertEqual(len(items), 74)
        self.assertTrue(all(item["dataset"] == "aime_2020_2024_part2_test" for item in items))
        self.assertTrue(all(item["split"] == "test" for item in items))
        self.assertTrue(all(2020 <= item["year"] <= 2024 for item in items))
        self.assertEqual({item["part"] for item in items}, {"II"})

    def test_gpqa_loaders_return_fixed_train_and_test_sizes(self):
        if not GPQA_CACHE_ZIP.exists():
            self.skipTest(f"GPQA archive not found: {GPQA_CACHE_ZIP}")

        train_items = load_gpqa_main_train_200(seed=1)
        test_items = load_gpqa_diamond_test_100(seed=1)

        self.assertEqual(len(train_items), 200)
        self.assertEqual(len(test_items), 100)
        self.assertTrue(all(item["dataset"] == "gpqa_main_train_200" for item in train_items))
        self.assertTrue(all(item["dataset"] == "gpqa_diamond_test_100" for item in test_items))
        self.assertTrue(all(item["split"] == "train" for item in train_items))
        self.assertTrue(all(item["split"] == "test" for item in test_items))
        self.assertTrue(all(item["answer"] in {"A", "B", "C", "D"} for item in train_items))
        self.assertTrue(all(item["answer"] in {"A", "B", "C", "D"} for item in test_items))

    def test_gsm8k_loaders_return_fixed_random_train_and_test_samples(self):
        train_items = load_gsm8k_train_300(seed=1)
        train_items_again = load_gsm8k_train_300(seed=1)
        test_items = load_gsm8k_test_189(seed=1)

        self.assertEqual(len(train_items), 300)
        self.assertEqual(len(test_items), 189)
        self.assertEqual([item["id"] for item in train_items], [item["id"] for item in train_items_again])
        self.assertTrue(all(item["dataset"] == "gsm8k_train_300" for item in train_items))
        self.assertTrue(all(item["dataset"] == "gsm8k_test_189" for item in test_items))
        self.assertTrue(all(item["split"] == "train" for item in train_items))
        self.assertTrue(all(item["split"] == "test" for item in test_items))
        self.assertTrue(all(item["answer"].strip() for item in train_items + test_items))
        self.assertTrue(all(item["source_path"].endswith("gsm8k/train.jsonl") for item in train_items))
        self.assertTrue(all(item["source_path"].endswith("gsm8k/test.jsonl") for item in test_items))
        self.assertFalse({item["id"] for item in train_items} & {item["id"] for item in test_items})

    def test_gsm8k_answer_extraction_uses_hash_marker_suffix(self):
        self.assertEqual(_extract_gsm8k_final_answer("work\n#### 72"), "72")
        self.assertEqual(_extract_gsm8k_final_answer("already final"), "already final")

    def test_omnimath79_loader_returns_first_100_valid_hard_test_items(self):
        items = load_omnimath7_9_test_100()

        self.assertEqual(len(items), 100)
        self.assertTrue(all(item["dataset"] == "omnimath7_9_test_100" for item in items))
        self.assertTrue(all(item["split"] == "test" for item in items))
        self.assertTrue(all(7.0 <= item["difficulty"] <= 9.0 for item in items))
        self.assertTrue(all(item["answer"].strip() for item in items))
        self.assertTrue(all(item["source_path"].endswith("omnimath.jsonl") for item in items))
        self.assertEqual(items[0]["id"], "omnimath_00000")
        self.assertEqual(items[0]["source_index"], 0)

    def test_aime_part1_train_and_part2_test_ids_are_disjoint(self):
        train_ids = {item["id"] for item in load_aime_2010_2024_part1_train()}
        test_ids = {item["id"] for item in load_aime_2020_2024_part2_test()}

        self.assertFalse(train_ids & test_ids)

    def test_omnimath14_train_and_omnimath79_test_ids_are_disjoint(self):
        train_ids = {item["id"] for item in load_omnimath(max_items=200, min_diff=1.0, max_diff=4.0)}
        test_ids = {item["id"] for item in load_omnimath7_9_test_100()}

        self.assertFalse(train_ids & test_ids)

    def test_omnimath13_train_and_omnimath34_test_loaders_return_fixed_random_samples(self):
        train_items = load_omnimath_diff1_3_train_200(seed=1)
        train_items_again = load_omnimath_diff1_3_train_200(seed=1)
        test_items = load_omnimath_diff3_4_test_200(seed=1)

        self.assertEqual(len(train_items), 200)
        self.assertEqual(len(test_items), 200)
        self.assertEqual([item["id"] for item in train_items], [item["id"] for item in train_items_again])
        self.assertTrue(all(item["dataset"] == "omnimath_diff1_3_train_200" for item in train_items))
        self.assertTrue(all(item["split"] == "train" for item in train_items))
        self.assertTrue(all(1.0 <= item["difficulty"] < 3.0 for item in train_items))
        self.assertTrue(all(item["dataset"] == "omnimath_diff3_4_test_200" for item in test_items))
        self.assertTrue(all(item["split"] == "test" for item in test_items))
        self.assertTrue(all(3.0 <= item["difficulty"] < 4.0 for item in test_items))
        self.assertFalse({item["id"] for item in train_items} & {item["id"] for item in test_items})

    def test_omnimath49_stratified_loader_returns_fixed_balanced_test_sample(self):
        items = load_omnimath_diff4_9_stratified_test_100(seed=1)
        items_again = load_omnimath_diff4_9_stratified_test_100(seed=1)

        bucket_counts = {}
        for item in items:
            bucket = int(item["difficulty"])
            bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

        self.assertEqual(len(items), 100)
        self.assertEqual([item["id"] for item in items], [item["id"] for item in items_again])
        self.assertEqual(bucket_counts, {4: 20, 5: 20, 6: 20, 7: 20, 8: 20})
        self.assertTrue(all(item["dataset"] == "omnimath_diff4_9_stratified_test_100" for item in items))
        self.assertTrue(all(item["split"] == "test" for item in items))
        self.assertTrue(all(4.0 <= item["difficulty"] < 9.0 for item in items))
        self.assertTrue(all(item["source_path"].endswith("omnimath.jsonl") for item in items))
        self.assertTrue(all("source_index" in item and "source_id" in item for item in items))

    def test_load_items_for_dataset_accepts_aime_part_search_names(self):
        train_items = [{"id": "train-0"}]
        test_items = [{"id": "test-0"}]

        with patch(
            "data.generate_episodes.load_aime_2010_2024_part1_train",
            return_value=train_items,
        ), patch(
            "data.generate_episodes.load_aime_2020_2024_part2_test",
            return_value=test_items,
        ):
            self.assertEqual(load_items_for_dataset("aime_2010_2024_part1_train"), train_items)
            self.assertEqual(load_items_for_dataset("aime_2020_2024_part2_test"), test_items)

    def test_load_items_for_dataset_accepts_gpqa_search_names(self):
        train_items = [{"id": "gpqa-train-0"}]
        test_items = [{"id": "gpqa-test-0"}]

        with patch(
            "data.generate_episodes.load_gpqa_main_train_200",
            return_value=train_items,
        ), patch(
            "data.generate_episodes.load_gpqa_diamond_test_100",
            return_value=test_items,
        ):
            self.assertEqual(load_items_for_dataset("gpqa_main_train_200"), train_items)
            self.assertEqual(load_items_for_dataset("gpqa_diamond_test_100"), test_items)

    def test_load_items_for_dataset_accepts_gsm8k_search_names(self):
        train_items = [{"id": "gsm8k-train-0"}]
        test_items = [{"id": "gsm8k-test-0"}]

        with patch(
            "data.generate_episodes.load_gsm8k_train_300",
            return_value=train_items,
        ), patch(
            "data.generate_episodes.load_gsm8k_test_189",
            return_value=test_items,
        ):
            self.assertEqual(load_items_for_dataset("gsm8k_train_300"), train_items)
            self.assertEqual(load_items_for_dataset("gsm8k_test_189"), test_items)

    def test_load_items_for_dataset_accepts_omnimath79_test_name(self):
        test_items = [{"id": "omni-test-0"}]

        with patch(
            "data.generate_episodes.load_omnimath7_9_test_100",
            return_value=test_items,
        ):
            self.assertEqual(load_items_for_dataset("omnimath7_9_test_100"), test_items)

    def test_load_items_for_dataset_accepts_omnimath13_to34_names(self):
        train_items = [{"id": "omni13-train-0"}]
        test_items = [{"id": "omni34-test-0"}]

        with patch(
            "data.generate_episodes.load_omnimath_diff1_3_train_200",
            return_value=train_items,
        ), patch(
            "data.generate_episodes.load_omnimath_diff3_4_test_200",
            return_value=test_items,
        ):
            self.assertEqual(load_items_for_dataset("omnimath_diff1_3_train_200"), train_items)
            self.assertEqual(load_items_for_dataset("omnimath_diff3_4_test_200"), test_items)

    def test_load_items_for_dataset_accepts_omnimath49_stratified_name(self):
        test_items = [{"id": "omni49-test-0"}]

        with patch(
            "data.generate_episodes.load_omnimath_diff4_9_stratified_test_100",
            return_value=test_items,
        ):
            self.assertEqual(load_items_for_dataset("omnimath_diff4_9_stratified_test_100"), test_items)

    def test_generate_model_solutions_parallel_runs_srm_and_lrm_concurrently(self):
        class FakeClient:
            def __init__(self, name):
                self.name = name

            def generate_solution(self, query, max_tokens, temperature, think_mode):
                time.sleep(0.15)
                return f"{self.name}:{query}:{think_mode}", len(self.name)

        t0 = time.time()
        result = generate_model_solutions_parallel(
            srm=FakeClient("srm"),
            lrm=FakeClient("lrm"),
            query="q",
            max_new_tokens=128,
            temperature=0.0,
            think_mode=True,
        )
        elapsed = time.time() - t0

        self.assertLess(elapsed, 0.25)
        self.assertEqual(result["srm"][:2], ("srm:q:True", 3))
        self.assertEqual(result["lrm"][:2], ("lrm:q:True", 3))
        self.assertIsInstance(result["srm"][2], float)
        self.assertIsInstance(result["lrm"][2], float)


if __name__ == "__main__":
    unittest.main()

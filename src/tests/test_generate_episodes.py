import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.generate_episodes import generate_model_solutions_parallel, load_items_for_dataset
from data.datasets import (
    load_gpqa_main_train_200,
    load_gpqa_diamond_test_100,
    load_aime_2010_2024_part1_train,
    load_aime_2020_2024_part2_test,
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

    def test_aime_part1_train_and_part2_test_ids_are_disjoint(self):
        train_ids = {item["id"] for item in load_aime_2010_2024_part1_train()}
        test_ids = {item["id"] for item in load_aime_2020_2024_part2_test()}

        self.assertFalse(train_ids & test_ids)

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

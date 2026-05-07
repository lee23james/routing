import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.datasets import load_gpqa_main_train_200, load_gpqa_diamond_test_100
from models import extract_answer, check_correctness


class GPQASupportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gpqa_zip = Path("/home/chencheng/.cache/vscode-tmp/gpqa_dataset.zip")
        if not cls.gpqa_zip.exists():
            raise unittest.SkipTest("GPQA archive not available locally")

    def test_gpqa_train_loader_is_deterministic_for_seed(self):
        first = load_gpqa_main_train_200(seed=1)
        second = load_gpqa_main_train_200(seed=1)

        self.assertEqual([item["id"] for item in first], [item["id"] for item in second])
        self.assertEqual([item["answer"] for item in first], [item["answer"] for item in second])

    def test_gpqa_test_loader_has_exactly_one_hundred_items(self):
        items = load_gpqa_diamond_test_100(seed=1)

        self.assertEqual(len(items), 100)
        self.assertTrue(all(item["answer"] in {"A", "B", "C", "D"} for item in items))
        self.assertTrue(all("Answer Choices:" in item["query"] for item in items))

    def test_mcq_answer_extraction_prefers_final_letter(self):
        text = "Reasoning here.\n\nThe answer is (C)."

        self.assertEqual(extract_answer(text, mode="multiple_choice"), "C")

    def test_mcq_correctness_compares_letters_only(self):
        self.assertTrue(check_correctness("B", "B", mode="multiple_choice"))
        self.assertFalse(check_correctness("B", "C", mode="multiple_choice"))


if __name__ == "__main__":
    unittest.main()

import unittest
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.generate_episodes import load_items_for_dataset


class GenerateEpisodesTest(unittest.TestCase):
    def test_math_train_200_uses_first_200_math_training_items(self):
        items = [{"id": f"math_train_{idx:05d}"} for idx in range(250)]

        with patch("data.generate_episodes.load_math_train", return_value=items):
            selected = load_items_for_dataset("math_train_200")

        self.assertEqual(len(selected), 200)
        self.assertEqual(selected[0]["id"], "math_train_00000")
        self.assertEqual(selected[-1]["id"], "math_train_00199")


if __name__ == "__main__":
    unittest.main()

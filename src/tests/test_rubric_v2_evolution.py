import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rubric.evolve_rubric_weights import (
    build_output_payload,
    compute_router_utility,
    learn_router_feedback_weights,
    normalize_weights,
    smooth_weights,
)


class RubricV2EvolutionTest(unittest.TestCase):
    def test_compute_router_utility_excludes_rubric_reward(self):
        self.assertAlmostEqual(
            compute_router_utility(correct=True, total_lrm_tokens=1200, lam=2e-5),
            1.0 - 1200 * 2e-5,
        )
        self.assertAlmostEqual(
            compute_router_utility(correct=False, total_lrm_tokens=1200, lam=2e-5),
            -1200 * 2e-5,
        )

    def test_normalize_weights_preserves_keys_and_sums_positive_weights(self):
        weights = {"a": 2.0, "b": 1.0, "c": 0.0, "d": -5.0}

        normalized = normalize_weights(weights)

        self.assertEqual(set(normalized), set(weights))
        self.assertAlmostEqual(sum(normalized.values()), 1.0)
        self.assertGreater(normalized["a"], normalized["b"])
        self.assertEqual(normalized["c"], 0.0)
        self.assertEqual(normalized["d"], 0.0)

    def test_smooth_weights_supports_alpha_endpoints(self):
        base = {"a": 0.75, "b": 0.25}
        router = {"a": 0.0, "b": 1.0}

        self.assertEqual(smooth_weights(base, router, alpha=0.0), base)
        self.assertEqual(smooth_weights(base, router, alpha=1.0), router)

        mixed = smooth_weights(base, router, alpha=0.3)
        self.assertAlmostEqual(mixed["a"], 0.525)
        self.assertAlmostEqual(mixed["b"], 0.475)
        self.assertAlmostEqual(sum(mixed.values()), 1.0)

    def test_feedback_weights_keep_only_positive_discriminative_correlations(self):
        rollouts = [
            {"utility": 0.0, "rubric_scores": {"good": 0.0, "bad": 1.0, "flat": 0.5}},
            {"utility": 1.0, "rubric_scores": {"good": 1.0, "bad": 0.0, "flat": 0.5}},
            {"utility": 0.8, "rubric_scores": {"good": 0.8, "bad": 0.2, "flat": 0.5}},
            {"utility": 0.2, "rubric_scores": {"good": 0.2, "bad": 0.8, "flat": 0.5}},
        ]

        weights, diagnostics = learn_router_feedback_weights(
            rollouts,
            rubric_names=["good", "bad", "flat"],
            corr_threshold=0.0,
            std_threshold=0.02,
        )

        self.assertAlmostEqual(weights["good"], 1.0)
        self.assertEqual(weights["bad"], 0.0)
        self.assertEqual(weights["flat"], 0.0)
        self.assertEqual(diagnostics["good"]["status"], "active")
        self.assertEqual(diagnostics["bad"]["status"], "negative_corr")
        self.assertEqual(diagnostics["flat"]["status"], "low_std")

    def test_feedback_weights_fall_back_to_base_when_no_active_signal(self):
        base = {"a": 0.8, "b": 0.2}
        rollouts = [
            {"utility": 0.5, "rubric_scores": {"a": 0.5, "b": 0.5}},
            {"utility": 0.5, "rubric_scores": {"a": 0.5, "b": 0.5}},
        ]

        weights, diagnostics = learn_router_feedback_weights(
            rollouts,
            rubric_names=["a", "b"],
            fallback_weights=base,
        )

        self.assertEqual(weights, base)
        self.assertTrue(all(row["status"].startswith("fallback") for row in diagnostics.values()))

    def test_output_payload_is_train_ppo_compatible(self):
        base = {"a": 0.75, "b": 0.25}
        router = {"a": 0.0, "b": 1.0}
        evolved = smooth_weights(base, router, alpha=0.3)
        diagnostics = {
            "a": {"corr": -1.0, "std": 0.4, "status": "negative_corr"},
            "b": {"corr": 1.0, "std": 0.4, "status": "active"},
        }

        payload = build_output_payload(
            weights=evolved,
            base_weights=base,
            router_feedback_weights=router,
            diagnostics=diagnostics,
            alpha=0.3,
            lam=2e-5,
            router_checkpoint="checkpoints/router0.pt",
            n_rollouts=200,
        )

        self.assertIn("weights", payload)
        self.assertIn("active_rubrics", payload)
        self.assertEqual(payload["method"], "trim_rubric_v2_router_feedback")
        self.assertEqual(payload["active_rubrics"], ["a", "b"])
        self.assertAlmostEqual(sum(payload["weights"].values()), 1.0)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rubric_weights_v2.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(set(loaded["weights"]), set(base))
        self.assertIsInstance(loaded["active_rubrics"], list)


if __name__ == "__main__":
    unittest.main()

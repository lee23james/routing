import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class TrimRubricPipelineTests(unittest.TestCase):
    def test_vllm_client_accepts_full_server_url(self):
        from vllm_client import VLLMClient

        client = VLLMClient(server_url="http://localhost:31001/v1")

        self.assertEqual(client.url, "http://localhost:31001/v1/chat/completions")

    def test_vllm_client_raises_openai_error_objects(self):
        from vllm_client import VLLMClient

        class FakeResponse:
            def json(self):
                return {
                    "object": "error",
                    "message": "bad request from vLLM",
                    "type": "BadRequestError",
                }

        client = VLLMClient(server_url="http://localhost:31001/v1")
        with mock.patch("vllm_client.requests.post", return_value=FakeResponse()):
            with self.assertRaisesRegex(RuntimeError, "bad request from vLLM"):
                client._call([{"role": "user", "content": "x"}])

    def test_dataset_aliases_match_trim_splits(self):
        from data.datasets import load_trim_dataset_alias

        expected = {
            "trim_math_train_1k": 1100,
            "trim_math500_test_100": 100,
            "trim_aime_train": 451,
            "trim_aime_test": 482,
        }

        for alias, count in expected.items():
            with self.subTest(alias=alias):
                self.assertEqual(len(load_trim_dataset_alias(alias)), count)

    def test_generate_episodes_uses_configurable_endpoints_and_prm_server(self):
        fake_models = types.ModuleType("models")
        fake_models.PRMScorer = object
        fake_models.ServerPRMScorer = object
        fake_models.split_steps = lambda text: [s for s in text.split("\\n\\n") if s]
        fake_models.extract_answer = lambda text: "2" if "\\boxed{2}" in text else ""
        fake_models.check_correctness = lambda pred, answer: pred == answer
        item = {
            "id": "demo_001",
            "query": "What is 1+1?",
            "answer": "2",
            "dataset": "demo",
        }
        clients = []

        class FakeClient:
            def __init__(self, port=None, model_name="default", server_url=None):
                self.port = port
                self.server_url = server_url
                self.model_name = model_name
                clients.append(self)

            def generate_solution(self, query, max_tokens, temperature, think_mode):
                if self.model_name == "srm":
                    return "First compute.\\n\\nTherefore \\boxed{2}.", 12
                return "Use arithmetic.\\n\\nThus \\boxed{2}.", 14

        class FakePRM:
            def __init__(self, server_url, model_name=None, max_workers=4):
                self.server_url = server_url
                self.model_name = model_name
                self.max_workers = max_workers

            def score_trace(self, query, steps):
                return [0.9 for _ in steps]

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(sys.modules, {"models": fake_models}):
                from data import generate_episodes as gen

            with mock.patch.object(gen, "load_math_train", return_value=[item]), \
                 mock.patch.object(gen, "VLLMClient", FakeClient), \
                 mock.patch.object(gen, "ServerPRMScorer", FakePRM):
                gen.generate_episodes(
                    "math_train_1k",
                    output_dir=tmpdir,
                    srm_server_url="http://localhost:30001/v1",
                    lrm_server_url="http://localhost:30000/v1",
                    prm_server_url="http://localhost:30002",
                    max_new_tokens=123,
                    resume=False,
                )

            self.assertEqual(clients[0].server_url, "http://localhost:30001/v1")
            self.assertEqual(clients[1].server_url, "http://localhost:30000/v1")

            out_path = Path(tmpdir) / "math_train_1k_episodes.jsonl"
            rows = [json.loads(line) for line in out_path.read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["dataset"], "demo")
            self.assertEqual(rows[0]["srm_prm_scores"], [0.9, 0.9])


if __name__ == "__main__":
    unittest.main()

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vllm_client import VLLMClient


class VLLMClientTest(unittest.TestCase):
    def test_server_url_accepts_openai_v1_root(self):
        client = VLLMClient(server_url="http://localhost:4001/v1")

        self.assertEqual(
            client.url,
            "http://localhost:4001/v1/chat/completions",
        )

    def test_server_url_accepts_chat_completions_endpoint(self):
        client = VLLMClient(server_url="http://localhost:4001/v1/chat/completions")

        self.assertEqual(
            client.url,
            "http://localhost:4001/v1/chat/completions",
        )


if __name__ == "__main__":
    unittest.main()

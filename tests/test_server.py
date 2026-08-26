import http.client
import json
import threading
import unittest

from specdecode.backends import PythonSamplingBackend
from specdecode.config import DecodeConfig
from specdecode.models import TableModel
from specdecode.server import GenerationService, ServerConfig, create_server


class FakeTokenizer:
    bos_token_id = 0

    def encode(self, prompt, *, add_special_tokens):
        return [0]

    def decode(self, token_ids, **kwargs):
        return "".join(str(token) for token in token_ids)


class BrokenTarget:
    vocab_size = 2

    def next_token_probs(self, token_ids):
        raise RuntimeError("target failed")


def make_service():
    model = TableModel({}, default=(0.0, 1.0))
    return GenerationService(
        model,
        model,
        FakeTokenizer(),
        DecodeConfig(max_new_tokens=3),
        PythonSamplingBackend(),
    )


class GenerationServiceTests(unittest.TestCase):
    def test_generation_response_and_request_limits(self) -> None:
        response = make_service().generate(
            {"prompt": "hello", "max_new_tokens": 2, "seed": 11}
        )

        self.assertEqual(response["text"], "11")
        self.assertEqual(response["token_ids"], [1, 1])
        self.assertEqual(response["sampling_backend"], "python")
        self.assertEqual(response["stats"]["accepted_tokens"], 2)

        with self.assertRaisesRegex(ValueError, "prompt must not be empty"):
            make_service().generate({"prompt": ""})
        with self.assertRaisesRegex(ValueError, "between 0 and 3"):
            make_service().generate({"prompt": "hello", "max_new_tokens": 4})

        draft = TableModel({}, default=(0.0, 1.0))
        broken_service = GenerationService(
            draft,
            BrokenTarget(),
            FakeTokenizer(),
            DecodeConfig(max_new_tokens=1),
            PythonSamplingBackend(),
        )
        with self.assertRaisesRegex(RuntimeError, "target failed"):
            broken_service.generate({"prompt": "hello"})

    def test_server_configuration_validation(self) -> None:
        with self.assertRaises(ValueError):
            ServerConfig(port=65536)
        with self.assertRaises(ValueError):
            ServerConfig(max_body_bytes=0)


class GenerationHTTPTests(unittest.TestCase):
    def setUp(self) -> None:
        try:
            self.server = create_server(
                make_service(),
                ServerConfig(host="127.0.0.1", port=0, max_body_bytes=128),
            )
        except PermissionError:
            self.skipTest("loopback sockets are unavailable in this environment")
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.connection = http.client.HTTPConnection(
            "127.0.0.1",
            self.server.server_port,
            timeout=2,
        )

    def tearDown(self) -> None:
        self.connection.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def request(self, method, path, body=None, headers=None):
        self.connection.request(method, path, body=body, headers=headers or {})
        response = self.connection.getresponse()
        payload = json.loads(response.read())
        return response.status, payload

    def test_health_generation_and_not_found(self) -> None:
        status, health = self.request("GET", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(health, {"status": "ok"})

        body = json.dumps({"prompt": "hello", "max_new_tokens": 1})
        status, generation = self.request(
            "POST",
            "/v1/generate",
            body,
            {"Content-Type": "application/json", "Content-Length": str(len(body))},
        )
        self.assertEqual(status, 200)
        self.assertEqual(generation["token_ids"], [1])

        status, missing = self.request("GET", "/missing")
        self.assertEqual(status, 404)
        self.assertEqual(missing, {"error": "not found"})

    def test_invalid_and_oversized_requests_are_rejected(self) -> None:
        status, invalid = self.request(
            "POST",
            "/v1/generate",
            "{}",
            {"Content-Length": "2"},
        )
        self.assertEqual(status, 400)
        self.assertIn("prompt", invalid["error"])

        separate = http.client.HTTPConnection(
            "127.0.0.1",
            self.server.server_port,
            timeout=2,
        )
        separate.request(
            "POST",
            "/v1/generate",
            body=b"x" * 129,
            headers={"Content-Length": "129"},
        )
        response = separate.getresponse()
        self.assertEqual(response.status, 413)
        self.assertEqual(json.loads(response.read()), {"error": "request too large"})
        separate.close()


if __name__ == "__main__":
    unittest.main()

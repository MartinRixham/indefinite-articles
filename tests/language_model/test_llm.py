import json
import socket
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

from fake_llm import FakeLLM

from indefinite_articles.config import LLMConfig
from indefinite_articles.language_model.llm import LLM, LLMError, LLMUnavailable


class Server(BaseHTTPRequestHandler):
    """Answers with whatever the test put in `reply`: (status, content type, body bytes)."""
    reply: tuple[int, str, bytes] = (200, "application/json", b"{}")
    requests: list = []

    def log_message(self, format: str, *args: object):
        pass

    def do_GET(self):
        self._answer()

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Server.requests.append((self.path, dict(self.headers), body))
        self._answer()

    def _answer(self):
        status, content_type, body = Server.reply
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def events(*chunks: dict, done: bool = True) -> bytes:
    lines = [b": keep-alive\n\n"]
    for chunk in chunks:
        lines.append(b"data: " + json.dumps(chunk).encode() + b"\n\n")
    if done:
        lines.append(b"data: [DONE]\n\n")
    return b"".join(lines)


def delta(text: str):
    return {"choices": [{"index": 0, "delta": {"content": text}}]}


class LLMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Server)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        Server.requests.clear()
        self.config = LLMConfig(base_url=f"http://127.0.0.1:{self.server.server_port}/v1", model="test-model",
                                embedding_prefix="search: ")
        self.llm = LLM(self.config)

    def _reply(self, body: bytes, status: int = 200, content_type: str = "application/json"):
        Server.reply = (status, content_type, body)

    def _stream(self, body: bytes):
        self._reply(body, content_type="text/event-stream")

    def test_chat_request_body(self):
        self._stream(events(delta("Hel"), delta("lo"), {"choices": [], "usage": {}}))
        self.assertEqual(self.llm.chat("system text", "user text"), "Hello")
        path, headers, body = Server.requests[-1]
        self.assertEqual(path, "/v1/chat/completions")
        self.assertEqual(body["model"], "test-model")
        self.assertEqual(body["messages"], [{"role": "system", "content": "system text"},
                                            {"role": "user", "content": "user text"}])
        self.assertTrue(body["stream"])
        self.assertNotIn("response_format", body)
        self.assertNotIn("Authorization", headers)
        self.assertEqual(body["reasoning_effort"], "none")
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})

    def test_api_key_sent(self):
        self.config.api_key = "secret"
        self._stream(events(delta("OK")))
        self.llm.ping()
        self.assertEqual(Server.requests[-1][1]["Authorization"], "Bearer secret")

    def test_server_ignoring_stream(self):
        self._reply(json.dumps({"choices": [{"message": {"content": "whole"}}]}).encode())
        self.assertEqual(self.llm.chat("s", "u"), "whole")
        self._reply(b'{"unexpected": true}')
        with self.assertRaises(LLMError):
            self.llm.chat("s", "u")

    def test_chat_json(self):
        self._stream(events(delta('Sure: {"score": '), delta("7}")))
        self.assertEqual(self.llm.chat_json("s", "u"), {"score": 7})
        self.assertEqual(Server.requests[-1][2]["response_format"], {"type": "json_object"})
        for answer, expected in (('{"a": 1}', {"a": 1}), ('```json\n{"a": [1, 2]}\n```', {"a": [1, 2]}),
                                 ("```\n[1]\n```", [1]), ("Here you go: [1, 2, 3]. Enjoy", [1, 2, 3])):
            self._stream(events(delta(answer)))
            with self.subTest(answer=answer):
                self.assertEqual(self.llm.chat_json("s", "u"), expected)
        for answer in ("no json here", "{broken", "{not: json}"):
            self._stream(events(delta(answer)))
            with self.subTest(answer=answer), self.assertRaisesRegex(LLMError, "valid JSON"):
                self.llm.chat_json("s", "u")

    def test_stream_errors(self):
        self._stream(events({"error": "model not found"}))
        with self.assertRaisesRegex(LLMError, "model not found"):
            self.llm.chat("s", "u")
        self._stream(events(delta("cut"), done=False))
        with self.assertRaisesRegex(LLMUnavailable, "mid-answer"):
            self.llm.chat("s", "u")
        self._stream(b"data: {broken\n\n")
        with self.assertRaisesRegex(LLMError, "invalid JSON"):
            self.llm.chat("s", "u")

    def test_http_error(self):
        self._reply(b"overloaded", status=503, content_type="text/plain")
        with self.assertRaises(LLMError) as raised:
            self.llm.chat("s", "u")
        self.assertNotIsInstance(raised.exception, LLMUnavailable)
        self.assertIn("503", str(raised.exception))
        self.assertIn("overloaded", str(raised.exception))

    def test_unreachable_server(self):
        with socket.socket() as unused:
            unused.bind(("127.0.0.1", 0))
            port = unused.getsockname()[1]
        llm = LLM(LLMConfig(base_url=f"http://127.0.0.1:{port}/v1"))
        with self.assertRaisesRegex(LLMUnavailable, "cannot reach"):
            llm.ping()
        self.assertFalse(llm.on_cpu_only())  # can't tell, so don't refuse to start

    def test_embed(self):
        self._reply(json.dumps({"data": [{"index": 1, "embedding": [0.0, 1.0]},
                                         {"index": 0, "embedding": [1.0, 0.0]}]}).encode())
        self.assertEqual(self.llm.embed(["first", "second"]), [[1.0, 0.0], [0.0, 1.0]])
        path, _, body = Server.requests[-1]
        self.assertEqual(path, "/v1/embeddings")
        self.assertEqual(body, {"model": "qwen3-embedding:0.6b", "input": ["search: first", "search: second"]})

    def test_embed_timeout_is_learned_per_purpose(self):
        self._reply(json.dumps({"data": [{"index": 0, "embedding": [1.0]}]}).encode())
        self.llm.embed(["phrase"], purpose="phrases")
        self.llm.embed(["article"], purpose="articles")
        self.assertEqual(set(self.llm.slowest_waits()), {"embed phrases", "embed articles"})

    def test_embed_bad_responses(self):
        for answer in ({"data": [{"embedding": [1.0]}]}, {"nothing": []}, {"data": [{"index": 0}]}):
            self._reply(json.dumps(answer).encode())
            with self.subTest(answer=answer), self.assertRaises(LLMError):
                self.llm.embed(["first", "second"])
        self._reply(b"not json")
        with self.assertRaisesRegex(LLMError, "invalid JSON"):
            self.llm.embed(["first"])

    def test_embed_without_model(self):
        self.config.embedding_model = ""
        with self.assertRaisesRegex(LLMError, "embedding_model"):
            self.llm.embed(["x"])
        self.assertEqual(Server.requests, [])

    def test_on_cpu_only_matches_model_names(self):
        for models, expected in (([{"name": "test-model", "size_vram": 0}], True),
                                 ([{"name": "test-model:latest", "size_vram": 0}], True),
                                 ([{"name": "test-model", "size_vram": 1000}], False),
                                 ([{"name": "other", "size_vram": 0}], False)):
            self._reply(json.dumps({"models": models}).encode())
            with self.subTest(models=models):
                self.assertEqual(self.llm.on_cpu_only(), expected)
        self._reply(b"not ollama", content_type="text/plain")
        self.assertFalse(self.llm.on_cpu_only())

    def test_pull_skipped_when_not_ollama(self):
        self._reply(b"not ollama", content_type="text/plain")
        self.assertIsNone(self.llm.missing_models())
        self.llm.pull_missing_models()
        self.assertEqual(Server.requests, [])  # nothing was POSTed

    def test_deadline_passed(self):
        llm = LLM(self.config, deadline=lambda: 0)
        with self.assertRaisesRegex(LLMUnavailable, "next digest"):
            llm.embed(["x"])
        self.assertEqual(Server.requests, [])

    def test_deadline_cuts_a_slow_request(self):
        import time
        llm = LLM(self.config, deadline=lambda: time.time() + 0.2)
        original = Server._answer
        finished = threading.Event()

        def slow(handler: Server) -> None:
            time.sleep(1)
            try:
                original(handler)
            except ConnectionError:  # the client gave up at the deadline, as it should
                pass
            finally:
                finished.set()

        with mock.patch.object(Server, "_answer", slow):
            with self.assertRaisesRegex(LLMUnavailable, "next digest is due"):
                llm.embed(["x"])
            # Otherwise the handler outlives the test and reports the broken pipe on a later test's stderr.
            self.assertTrue(finished.wait(5))


if __name__ == "__main__":
    unittest.main()


class PullTests(unittest.TestCase):
    """Downloading missing models through Ollama's /api/pull."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeLLM)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self) -> None:
        self.llm = LLM(LLMConfig(base_url=f"http://127.0.0.1:{self.server.server_port}/v1",
                                 model="qwen3:14b", embedding_model="qwen3-embedding:0.6b"))
        for name, value in (("installed", ["qwen3:14b", "qwen3-embedding:0.6b"]), ("pulled", []),
                            ("pull_error", None)):
            patcher = mock.patch.object(FakeLLM, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_nothing_missing(self) -> None:
        self.assertEqual(self.llm.missing_models(), [])  # "name" matches "name:latest"
        self.llm.pull_missing_models()
        self.assertEqual(FakeLLM.pulled, [])

    def test_pulls_missing_models(self) -> None:
        FakeLLM.installed = ["llama3.1:8b"]
        self.assertEqual(self.llm.missing_models(), ["qwen3:14b", "qwen3-embedding:0.6b"])
        with mock.patch("indefinite_articles.language_model.llm.PULL_PROGRESS_SECONDS", 0.0), \
             self.assertLogs("indefinite_articles.language_model.llm", "INFO") as logs:
            self.llm.pull_missing_models()
        self.assertEqual(FakeLLM.pulled, ["qwen3:14b", "qwen3-embedding:0.6b"])
        self.assertEqual(self.llm.missing_models(), [])
        output = "\n".join(logs.output)
        self.assertIn("Ollama doesn't have qwen3:14b; downloading it", output)
        self.assertIn("downloading qwen3-embedding:0.6b: 50% of 0.0 GB", output)
        self.assertIn("downloading qwen3-embedding:0.6b: verifying sha256 digest", output)
        self.assertIn("downloaded qwen3-embedding:0.6b in", output)

    def test_same_model_pulled_once(self) -> None:
        self.llm.config.embedding_model = "qwen3:8b"
        self.llm.config.model = "qwen3:8b"
        with self.assertLogs("indefinite_articles.language_model.llm", "INFO"):
            self.llm.pull_missing_models()
        self.assertEqual(FakeLLM.pulled, ["qwen3:8b"])

    def test_pull_error(self) -> None:
        FakeLLM.installed = []
        FakeLLM.pull_error = "pull model manifest: file does not exist"
        with self.assertLogs("indefinite_articles.language_model.llm", "INFO"), \
             self.assertRaisesRegex(LLMError, "downloading qwen3:14b failed: pull model manifest") as raised:
            self.llm.pull_missing_models()
        self.assertNotIsInstance(raised.exception, LLMUnavailable)

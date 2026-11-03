import json
import socket
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

from indefinite_articles.config import LLMConfig
from indefinite_articles.llm import LLM, LLMError, LLMUnavailable, extract_json


class Server(BaseHTTPRequestHandler):
    """Answers with whatever the test put in `reply`: (status, content type, body bytes)."""
    reply: tuple[int, str, bytes] = (200, "application/json", b"{}")
    requests: list = []

    def log_message(self, format, *args):
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


def events(*chunks, done=True) -> bytes:
    lines = [b": keep-alive\n\n"]
    for chunk in chunks:
        lines.append(b"data: " + json.dumps(chunk).encode() + b"\n\n")
    if done:
        lines.append(b"data: [DONE]\n\n")
    return b"".join(lines)


def delta(text):
    return {"choices": [{"index": 0, "delta": {"content": text}}]}


class ExtractJSONTests(unittest.TestCase):
    def test_plain_and_fenced(self):
        self.assertEqual(extract_json('{"a": 1}'), {"a": 1})
        self.assertEqual(extract_json('```json\n{"a": [1, 2]}\n```'), {"a": [1, 2]})
        self.assertEqual(extract_json("```\n[1]\n```"), [1])

    def test_array_inside_prose(self):
        self.assertEqual(extract_json("Here you go: [1, 2, 3]. Enjoy"), [1, 2, 3])

    def test_no_json(self):
        for text in ("", "nothing here", "{broken", "{not: json}"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                extract_json(text)


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

    def _reply(self, body: bytes, status=200, content_type="application/json"):
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
        self._stream(events(delta("no json here")))
        with self.assertRaisesRegex(LLMError, "valid JSON"):
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
        self.assertEqual(body, {"model": "nomic-embed-text", "input": ["search: first", "search: second"]})

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

    def test_deadline_passed(self):
        llm = LLM(self.config, deadline=lambda: 0)
        with self.assertRaisesRegex(LLMUnavailable, "next digest"):
            llm.embed(["x"])
        self.assertEqual(Server.requests, [])

    def test_deadline_cuts_a_slow_request(self):
        import time
        llm = LLM(self.config, deadline=lambda: time.time() + 0.2)
        original = Server._answer

        def slow(handler):
            time.sleep(1)
            original(handler)

        with mock.patch.object(Server, "_answer", slow):
            with self.assertRaisesRegex(LLMUnavailable, "next digest is due"):
                llm.embed(["x"])


if __name__ == "__main__":
    unittest.main()

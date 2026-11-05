"""An in-process stand-in for an OpenAI-compatible LLM server (Ollama or llama-server).

Serve `FakeLLM` with an `http.server.ThreadingHTTPServer`. It answers chat requests by
recognizing a marker string in each prompt (see `do_POST`), embedding requests with
`fake_embedding`, and Ollama's own `/api/ps`, `/api/tags` and `/api/pull` requests.
"""

import json
import re
import time
import zlib
from http.server import BaseHTTPRequestHandler
from typing import Any


def fake_embedding(text: str, dimension: int = 64) -> list[float]:
    """Bag-of-words hashed vector: texts sharing words are similar, like real embeddings."""
    vector = [0.0] * dimension
    for word in re.findall(r"[a-z]{3,}", text.lower()):
        vector[zlib.crc32(word.encode()) % dimension] += 1.0
    return vector


class FakeLLM(BaseHTTPRequestHandler):
    """OpenAI-compatible endpoint: scores titles containing 'Rust'/'Linux' highly."""
    requests: list[dict[str, Any]] = []
    size_vram = 1
    delay = 0.0      # seconds before the first token
    gap = 0.0        # seconds between tokens
    hang_after: int | None = None  # stop sending after this many tokens (a hung server)
    answer: str | None = None      # if set, the reply to every chat request
    installed: list[str] = ["qwen3:14b", "qwen3-embedding:0.6b"]  # models Ollama has
    pulled: list[str] = []         # models downloaded through /api/pull
    pull_error: str | None = None  # if set, every download fails with this error

    def log_message(self, format: str, *args: Any) -> None:
        pass

    def do_GET(self) -> None:
        if self.path == "/api/tags":
            return self._json({"models": [{"name": name} for name in FakeLLM.installed]})
        self._json({"models": [{"name": "qwen3:14b", "size_vram": FakeLLM.size_vram}]})

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/api/pull":
            return self._pull(body["model"])
        FakeLLM.requests.append(body)
        if self.path.endswith("/embeddings"):
            return self._json({"data": [{"index": index, "embedding": fake_embedding(text)}
                                        for index, text in enumerate(body["input"])]})
        system, user = body["messages"][0]["content"], body["messages"][1]["content"]
        if FakeLLM.answer is not None:
            content = FakeLLM.answer
        elif "Article to rate:" in user:
            match = re.search(r"^Title: (.+)$", user, re.M)
            assert match is not None
            title = match.group(1)
            good = any(word in title for word in ("Rust", "Linux"))
            content = "```json\n" + json.dumps({"reason": "test", "score": 9 if good else 2}) + "\n```"
        elif "interpret a reader" in system:
            content = json.dumps({"ratings": [{"item": 2, "rating": "dislike", "comment": "boring"}],
                                  "general_comment": "less gossip"})
        elif "maintain a concise profile" in system:
            content = "- Likes Rust and systems programming\n- Dislikes gossip"
        elif "short phrases for a search engine" in system:
            content = json.dumps({"likes": ["Rust and systems programming"], "avoid": ["celebrity gossip"]})
        else:
            content = "OK"
        if not body.get("stream"):
            return self._json({"choices": [{"message": {"role": "assistant", "content": content}}]})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        time.sleep(FakeLLM.delay)
        tokens = [content[index:index + 4] for index in range(0, len(content), 4)]
        try:
            for index, chunk in enumerate(tokens):
                if FakeLLM.hang_after is not None and index >= FakeLLM.hang_after:
                    time.sleep(5)
                    return
                if index:
                    time.sleep(FakeLLM.gap)
                event = {"choices": [{"index": 0, "delta": {"content": chunk}}]}
                self.wfile.write(b"data: " + json.dumps(event).encode() + b"\n\n")
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
        except (BrokenPipeError, ConnectionResetError):
            pass  # the client gave up

    def _pull(self, name: str) -> None:
        """Stream Ollama's newline-delimited JSON download progress."""
        events: list[dict[str, Any]] = [
            {"status": "pulling manifest"},
            {"status": "pulling 0123abcd", "digest": "sha256:0123abcd", "total": 1000, "completed": 500},
            {"status": "pulling 0123abcd", "digest": "sha256:0123abcd", "total": 1000, "completed": 1000},
            {"status": "verifying sha256 digest"},
            {"status": "writing manifest"},
        ]
        if FakeLLM.pull_error is None:
            events.append({"status": "success"})
            FakeLLM.installed = [*FakeLLM.installed, name]
            FakeLLM.pulled = [*FakeLLM.pulled, name]
        else:
            events.append({"error": FakeLLM.pull_error})
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        self.wfile.write(b"".join(json.dumps(event).encode() + b"\n" for event in events))

    def _json(self, payload: object) -> None:
        out = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

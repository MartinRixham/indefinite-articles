"""Minimal client for an OpenAI-compatible chat completions endpoint.

Works with Ollama (http://127.0.0.1:11434/v1) and llama.cpp's llama-server
(http://127.0.0.1:8080/v1), both of which are packaged in Fedora.
"""

import http.client
import json
import logging
import re
import threading
import time
import urllib.parse
import urllib.request

log = logging.getLogger(__name__)

# Progress-based timeouts. A request is only abandoned when the server goes quiet
# for much longer than it ever has before: STALL_FACTOR times the longest silence
# seen so far for that kind of wait, and never less than MIN_STALL seconds. Until a
# request of that kind has succeeded there is no stall limit at all, only the
# deadline (if any). Each time the limit cuts a request off it doubles, so a server
# that is merely slower than before is waited for properly on a later attempt.
STALL_FACTOR = 4
MIN_STALL = 60.0


class LLMError(Exception):
    pass


class LLMUnavailable(LLMError):
    """The server can't be reached, stopped making progress, or ran out of time.

    Callers working through a list should stop and try again later rather than
    move on to the next item."""


class LLM:
    def __init__(self, cfg, deadline=None):
        """`deadline`, if given, returns the epoch time by which every request must
        finish (the service passes the next digest time)."""
        self.cfg = cfg
        self.deadline = deadline
        self._lock = threading.Lock()
        self._slowest: dict[str, float] = {}  # longest silence seen, per kind of wait

    # -- timeouts ------------------------------------------------------------------
    def _limit(self, kind: str) -> tuple[float | None, bool]:
        """Seconds to wait for the next data, and whether that is the stall limit
        (rather than the deadline)."""
        with self._lock:
            slowest = self._slowest.get(kind)
        stall = None if slowest is None else max(MIN_STALL, STALL_FACTOR * slowest)
        left = None
        if self.deadline is not None:
            left = self.deadline() - time.time()
            if left <= 0:
                raise LLMUnavailable("no time left before the next digest")
        if stall is not None and (left is None or stall < left):
            return stall, True
        return left, False

    def _observe(self, kind: str, seconds: float):
        with self._lock:
            if seconds > self._slowest.get(kind, 0.0):
                self._slowest[kind] = seconds

    def _stalled(self, kind: str, waited: float, by_stall: bool) -> LLMUnavailable:
        # Closing the connection makes Ollama and llama-server abandon the request.
        if not by_stall:
            return LLMUnavailable("LLM request stopped: the next digest is due")
        with self._lock:  # next time, wait twice as long as the limit that just fired
            self._slowest[kind] = max(self._slowest.get(kind, 0.0), 2 * waited / STALL_FACTOR)
        log.warning("LLM sent nothing for %.1fs (%s); next time waiting up to %.1fs",
                    waited, kind, 2 * waited)
        return LLMUnavailable(f"LLM stopped responding ({kind}, {waited:.0f}s without data)")

    # -- transport -----------------------------------------------------------------
    def _send(self, path: str, body: dict, kind: str):
        """POST `body`; returns (connection, socket, response, timeout, by_stall) once
        the response headers have arrived."""
        u = urllib.parse.urlsplit(self.cfg.base_url)
        cls = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
        timeout, by_stall = self._limit(kind)
        conn = cls(u.hostname, u.port, timeout=timeout)
        headers = {"Content-Type": "application/json"}
        if self.cfg.api_key:
            headers["Authorization"] = f"Bearer {self.cfg.api_key}"
        try:
            conn.request("POST", u.path + path, body=json.dumps(body).encode(), headers=headers)
            # Our own reference: http.client drops conn.sock when the server will close.
            sock = conn.sock
            resp = conn.getresponse()
        except TimeoutError:
            conn.close()
            raise self._stalled(kind, timeout or 0, by_stall) from None
        except OSError as e:
            conn.close()
            raise LLMUnavailable(f"cannot reach LLM at {self.cfg.base_url}: {e}") from None
        if resp.status >= 400:
            detail = resp.read(500).decode(errors="replace")
            conn.close()
            raise LLMError(f"LLM HTTP {resp.status}: {detail}")
        return conn, sock, resp, timeout, by_stall

    def _post(self, path: str, body: dict, kind: str) -> dict:
        """POST and read a whole JSON response (no progress to watch until it's done)."""
        t0 = time.monotonic()
        conn, _, resp, timeout, by_stall = self._send(path, body, kind)
        try:
            data = json.loads(resp.read())
        except TimeoutError:
            raise self._stalled(kind, timeout or 0, by_stall) from None
        except OSError as e:
            raise LLMUnavailable(f"connection to LLM lost: {e}") from None
        except ValueError:
            raise LLMError("LLM returned invalid JSON") from None
        finally:
            resp.close()
            conn.close()
        self._observe(kind, time.monotonic() - t0)
        return data

    def _stream(self, path: str, body: dict) -> str:
        """POST a streaming chat request and return the generated text.

        The wait for the first token (prompt processing, maybe loading the model) and
        the gaps between tokens are timed separately, as "first" and "gap"."""
        t0 = time.monotonic()
        conn, sock, resp, timeout, by_stall = self._send(path, {**body, "stream": True}, "first")
        kind = "first"
        try:
            if "text/event-stream" not in (resp.getheader("Content-Type") or ""):
                data = json.loads(resp.read())  # server ignored "stream"
                self._observe(kind, time.monotonic() - t0)
                return _message_text(data)
            parts, last = [], t0
            while True:
                line = resp.readline()
                if not line:  # closed before [DONE]: don't pass on a cut-off answer
                    raise LLMUnavailable("LLM closed the connection mid-answer")
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue  # blank separators and ": keep-alive" comments
                now = time.monotonic()
                self._observe(kind, now - last)
                last = now
                kind = "gap"
                timeout, by_stall = self._limit(kind)
                sock.settimeout(timeout)
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    break
                event = json.loads(payload)
                if "error" in event:
                    raise LLMError(f"LLM error: {str(event['error'])[:300]}")
                try:
                    parts.append(event["choices"][0]["delta"].get("content") or "")
                except (KeyError, IndexError, TypeError, AttributeError):
                    pass  # e.g. a final usage-only event
            return "".join(parts)
        except TimeoutError:
            raise self._stalled(kind, timeout or 0, by_stall) from None
        except OSError as e:
            raise LLMUnavailable(f"connection to LLM lost: {e}") from None
        except ValueError:
            raise LLMError("LLM returned invalid JSON") from None
        finally:
            resp.close()
            conn.close()

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embedding vectors for `texts`, using the configured embedding model."""
        if not self.cfg.embedding_model:
            raise LLMError("no embedding_model configured")
        data = self._post("/embeddings", {
            "model": self.cfg.embedding_model,
            "input": [self.cfg.embedding_prefix + t for t in texts],
        }, "embed")
        try:
            items = sorted(data["data"], key=lambda d: d.get("index", 0))
            vecs = [d["embedding"] for d in items]
        except (KeyError, TypeError):
            raise LLMError(f"unexpected embeddings response: {str(data)[:300]}") from None
        if len(vecs) != len(texts):
            raise LLMError("embeddings response has the wrong number of vectors")
        return vecs

    def chat(self, system: str, user: str, json_mode: bool = False) -> str:
        body = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.cfg.temperature,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        return self._stream("/chat/completions", body)

    def chat_json(self, system: str, user: str):
        text = self.chat(system, user, json_mode=True)
        try:
            return extract_json(text)
        except ValueError:
            log.debug("bad JSON from LLM: %r", text[:1000])
            raise LLMError("LLM did not return valid JSON") from None

    def ping(self) -> str:
        return self.chat("Reply with the single word OK.", "Are you there?")

    def on_cpu_only(self) -> bool:
        """True if Ollama has the chat model loaded with nothing in GPU memory."""
        try:
            url = self.cfg.base_url.removesuffix("/v1") + "/api/ps"
            with urllib.request.urlopen(url, timeout=self._limit("status")[0]) as resp:
                models = json.load(resp)["models"]
        except Exception:  # not Ollama (e.g. llama.cpp) or unreachable: can't tell
            return False
        names = {self.cfg.model, self.cfg.model + ":latest"}
        return any(m.get("name") in names and not m.get("size_vram") for m in models)


def _message_text(data) -> str:
    try:
        return data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        raise LLMError(f"unexpected LLM response: {str(data)[:300]}") from None


def extract_json(text: str):
    """Parse JSON from model output, tolerating code fences and surrounding prose."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for open_, close in (("{", "}"), ("[", "]")):
        start, end = text.find(open_), text.rfind(close)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError("no JSON found")

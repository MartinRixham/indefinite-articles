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
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..config import LLMConfig
from .sent_request import SentRequest

log = logging.getLogger(__name__)

# Progress-based timeouts. A request is only abandoned when the server goes quiet
# for much longer than it ever has before: STALL_FACTOR times the longest silence
# seen so far for that kind of wait, and never less than MIN_STALL seconds. Until a
# request of that kind has succeeded there is no stall limit at all, only the
# deadline (if any). Each time the limit cuts a request off it doubles, so a server
# that is merely slower than before is waited for properly on a later attempt.
STALL_FACTOR = 4
MIN_STALL = 60.0

PULL_PROGRESS_SECONDS = 30.0  # how often to log the progress of a model download


def stall_limit(slowest: float) -> float:
    """Seconds without data after which a wait is abandoned, given the longest silence seen."""
    return max(MIN_STALL, STALL_FACTOR * slowest)


class LLMError(Exception):
    pass


class LLMUnavailable(LLMError):
    """The server can't be reached, stopped making progress, or ran out of time.

    Callers working through a list should stop and try again later rather than
    move on to the next item."""


@dataclass
class WaitLimit:
    """Seconds to wait for the next data (None: no limit), and whether that is the
    stall limit rather than the deadline."""
    seconds: float | None
    by_stall: bool


class LLM:
    def __init__(self, config: LLMConfig, deadline: Callable[[], float] | None = None):
        """`deadline`, if given, returns the epoch time by which every request must
        finish (the service passes the next digest time)."""
        self.config = config
        self.deadline = deadline
        self._lock = threading.Lock()
        self._slowest: dict[str, float] = {}  # longest silence seen, per kind of wait

    # -- timeouts ------------------------------------------------------------------
    def _limit(self, kind: str) -> WaitLimit:
        with self._lock:
            slowest = self._slowest.get(kind)
        stall = None if slowest is None else stall_limit(slowest)
        left = None
        if self.deadline is not None:
            left = self.deadline() - time.time()
            if left <= 0:
                raise LLMUnavailable("no time left before the next digest")
        if stall is not None and (left is None or stall < left):
            return WaitLimit(stall, True)
        return WaitLimit(left, False)

    def slowest_waits(self) -> dict[str, float]:
        """The longest silence seen so far for each kind of wait, in seconds (a copy)."""
        with self._lock:
            return dict(self._slowest)

    def _observe(self, kind: str, seconds: float):
        with self._lock:
            if seconds > self._slowest.get(kind, 0.0):
                self._slowest[kind] = seconds

    def _stalled(self, kind: str, limit: WaitLimit) -> LLMUnavailable:
        if not limit.by_stall:
            return LLMUnavailable("LLM request stopped: the next digest is due")
        waited = limit.seconds or 0
        with self._lock:  # next time, wait twice as long as the limit that just fired
            self._slowest[kind] = max(self._slowest.get(kind, 0.0), 2 * waited / STALL_FACTOR)
        log.warning("LLM sent nothing for %.1fs (%s); next time waiting up to %.1fs",
                    waited, kind, 2 * waited)
        return LLMUnavailable(f"LLM stopped responding ({kind}, {waited:.0f}s without data)")

    # -- transport -----------------------------------------------------------------
    def _send(self, path: str, body: dict, kind: str, native: bool = False) -> SentRequest:
        """POST `body` and wait for the response headers. `native` requests go to
        Ollama's own API, at the server root rather than under /v1."""
        parts = urllib.parse.urlsplit(self.config.base_url)
        if not parts.hostname:
            raise LLMError(f"base_url has no host: {self.config.base_url!r}")
        base_path = parts.path.removesuffix("/v1") if native else parts.path
        cls = http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
        limit = self._limit(kind)
        connection = cls(parts.hostname, parts.port, timeout=limit.seconds)
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        try:
            connection.request("POST", base_path + path, body=json.dumps(body).encode(), headers=headers)
            # Our own reference: http.client drops connection.sock when the server will close.
            stream_socket = connection.sock
            response = connection.getresponse()
        except TimeoutError:
            connection.close()
            raise self._stalled(kind, limit) from None
        except OSError as error:
            connection.close()
            raise LLMUnavailable(f"cannot reach LLM at {self.config.base_url}: {error}") from None
        if response.status >= 400:
            detail = response.read(500).decode(errors="replace")
            connection.close()
            raise LLMError(f"LLM HTTP {response.status}: {detail}")
        return SentRequest(connection, stream_socket, response, limit)

    def _post(self, path: str, body: dict, kind: str) -> dict:
        """POST and read a whole JSON response (no progress to watch until it's done)."""
        started = time.monotonic()
        sent = self._send(path, body, kind)
        try:
            data = json.loads(sent.response.read())
        except TimeoutError:
            raise self._stalled(kind, sent.limit) from None
        except OSError as error:
            raise LLMUnavailable(f"connection to LLM lost: {error}") from None
        except ValueError:
            raise LLMError("LLM returned invalid JSON") from None
        finally:
            sent.close()
        self._observe(kind, time.monotonic() - started)
        return data

    def _stream(self, path: str, body: dict, purpose: str) -> str:
        """POST a streaming chat request and return the generated text.

        The wait for the first token (prompt processing, maybe loading the model) and
        the gaps between tokens are timed separately, as "first <purpose>" and "gap".
        The first-token wait is learned per purpose because it grows with the prompt:
        a long scoring prompt can take many times longer than a short one, so a limit
        learned from short prompts would cut it off. The gap between tokens doesn't
        depend on the prompt, so it is shared."""
        started = time.monotonic()
        kind = f"first {purpose}"
        sent = self._send(path, {**body, "stream": True}, kind)
        response, limit = sent.response, sent.limit
        try:
            if "text/event-stream" not in (response.getheader("Content-Type") or ""):
                data = json.loads(response.read())  # server ignored "stream"
                self._observe(kind, time.monotonic() - started)
                return _message_text(data)
            parts, last = [], started
            while True:
                line = response.readline()
                if not line:  # closed before [DONE]: don't pass on a cut-off answer
                    raise LLMUnavailable("LLM closed the connection mid-answer")
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue  # blank separators and ": keep-alive" comments
                now = time.monotonic()
                self._observe(kind, now - last)
                last = now
                kind = "gap"
                limit = self._limit(kind)
                sent.stream_socket.settimeout(limit.seconds)
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
            raise self._stalled(kind, limit) from None
        except OSError as error:
            raise LLMUnavailable(f"connection to LLM lost: {error}") from None
        except ValueError:
            raise LLMError("LLM returned invalid JSON") from None
        finally:
            sent.close()

    def embed(self, texts: list[str], purpose: str = "texts") -> list[list[float]]:
        """Embedding vectors for `texts`, using the configured embedding model.

        `purpose` names the kind of request, so that its timeout is learned from similar
        requests: a batch of articles takes far longer than a few short phrases."""
        if not self.config.embedding_model:
            raise LLMError("no embedding_model configured")
        data = self._post("/embeddings", {
            "model": self.config.embedding_model,
            "input": [self.config.embedding_prefix + text for text in texts],
        }, f"embed {purpose}")
        try:
            items = sorted(data["data"], key=lambda item: item.get("index", 0))
            vectors = [item["embedding"] for item in items]
        except (KeyError, TypeError):
            raise LLMError(f"unexpected embeddings response: {str(data)[:300]}") from None
        if len(vectors) != len(texts):
            raise LLMError("embeddings response has the wrong number of vectors")
        return vectors

    def chat(self, system: str, user: str, json_mode: bool = False, purpose: str = "chat") -> str:
        """The model's answer. `purpose` names the kind of request, so that its
        first-token timeout is learned from requests with prompts of similar size."""
        body = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.config.temperature,
            # No thinking (qwen3 and other reasoning models): it makes every request many
            # times slower and uses up context, for little gain at rating articles. The
            # first field is for Ollama, the second for llama.cpp; other models ignore both.
            "reasoning_effort": "none",
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        return self._stream("/chat/completions", body, purpose)

    def chat_json(self, system: str, user: str, purpose: str = "chat") -> Any:
        text = self.chat(system, user, json_mode=True, purpose=purpose)
        try:
            return _extract_json(text)
        except ValueError:
            log.debug("bad JSON from LLM: %r", text[:1000])
            raise LLMError("LLM did not return valid JSON") from None

    def ping(self) -> str:
        return self.chat("Reply with the single word OK.", "Are you there?")

    def _ollama_models(self, path: str) -> list[dict]:
        """The "models" list from one of Ollama's own GET endpoints (/api/ps, /api/tags)."""
        url = self.config.base_url.removesuffix("/v1") + path
        with urllib.request.urlopen(url, timeout=self._limit("status").seconds) as response:
            return json.load(response)["models"]

    def on_cpu_only(self) -> bool:
        """True if Ollama has the chat model loaded with nothing in GPU memory."""
        try:
            models = self._ollama_models("/api/ps")
        except Exception:  # not Ollama (e.g. llama.cpp) or unreachable: can't tell
            return False
        names = {self.config.model, self.config.model + ":latest"}
        return any(model.get("name") in names and not model.get("size_vram") for model in models)

    def missing_models(self) -> list[str] | None:
        """The configured chat and embedding models that Ollama hasn't downloaded,
        or None if that can't be told (not Ollama, or unreachable)."""
        try:
            installed = {model.get("name") for model in self._ollama_models("/api/tags")}
        except Exception:
            return None
        wanted = dict.fromkeys(name for name in (self.config.model, self.config.embedding_model) if name)
        return [name for name in wanted if not {name, name + ":latest"} & installed]

    def pull_missing_models(self) -> None:
        """Have Ollama download any configured model it doesn't have yet. Does nothing
        when the server isn't Ollama. Raises LLMError if a download fails."""
        for name in self.missing_models() or []:
            self.pull(name)

    def pull(self, name: str) -> None:
        """Have Ollama download the model `name`, logging its progress."""
        log.info("Ollama doesn't have %s; downloading it", name)
        kind = "pull"
        started = time.monotonic()
        sent = self._send("/api/pull", {"model": name}, kind, native=True)
        limit = sent.limit
        last, last_logged, last_status = started, started, ""
        try:
            while True:
                line = sent.response.readline()
                if not line:
                    raise LLMUnavailable(f"Ollama closed the connection while downloading {name}")
                now = time.monotonic()
                self._observe(kind, now - last)
                last = now
                limit = self._limit(kind)
                sent.stream_socket.settimeout(limit.seconds)
                if not line.strip():
                    continue
                event = json.loads(line)
                if "error" in event:
                    raise LLMError(f"downloading {name} failed: {str(event['error'])[:300]}")
                status = str(event.get("status", ""))
                if status == "success":
                    log.info("downloaded %s in %.0fs", name, now - started)
                    return
                total, completed = event.get("total"), event.get("completed")
                if total and completed is not None:  # one line per chunk: log now and then
                    if now - last_logged >= PULL_PROGRESS_SECONDS:
                        log.info("downloading %s: %d%% of %.1f GB", name, 100 * completed // total, total / 1e9)
                        last_logged = now
                elif status != last_status:
                    log.info("downloading %s: %s", name, status)
                last_status = status
        except TimeoutError:
            raise self._stalled(kind, limit) from None
        except OSError as error:
            raise LLMUnavailable(f"connection to Ollama lost while downloading {name}: {error}") from None
        except ValueError:
            raise LLMError("Ollama returned invalid JSON") from None
        finally:
            sent.close()


def _message_text(data: dict) -> str:
    try:
        return data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        raise LLMError(f"unexpected LLM response: {str(data)[:300]}") from None


def _extract_json(text: str):
    """Parse JSON from model output, tolerating code fences and surrounding prose."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for opener, close in (("{", "}"), ("[", "]")):
        start, end = text.find(opener), text.rfind(close)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError("no JSON found")

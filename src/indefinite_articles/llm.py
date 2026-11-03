"""Minimal client for an OpenAI-compatible chat completions endpoint.

Works with Ollama (http://127.0.0.1:11434/v1) and llama.cpp's llama-server
(http://127.0.0.1:8080/v1), both of which are packaged in Fedora.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request

log = logging.getLogger(__name__)


class LLMError(Exception):
    pass


class LLM:
    def __init__(self, cfg):
        self.cfg = cfg

    def _post(self, path: str, body: dict) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.cfg.api_key:
            headers["Authorization"] = f"Bearer {self.cfg.api_key}"
        req = urllib.request.Request(
            f"{self.cfg.base_url}{path}", data=json.dumps(body).encode(), headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.cfg.timeout_seconds) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            detail = e.read()[:500].decode(errors="replace")
            raise LLMError(f"LLM HTTP {e.code}: {detail}") from None
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise LLMError(f"cannot reach LLM at {self.cfg.base_url}: {e}") from None
        except ValueError:
            raise LLMError("LLM returned invalid JSON") from None

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embedding vectors for `texts`, using the configured embedding model."""
        if not self.cfg.embedding_model:
            raise LLMError("no embedding_model configured")
        data = self._post("/embeddings", {
            "model": self.cfg.embedding_model,
            "input": [self.cfg.embedding_prefix + t for t in texts],
        })
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
            "stream": False,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        data = self._post("/chat/completions", body)
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            raise LLMError(f"unexpected LLM response: {str(data)[:300]}") from None

    def chat_json(self, system: str, user: str):
        text = self.chat(system, user, json_mode=True)
        try:
            return extract_json(text)
        except ValueError:
            log.debug("bad JSON from LLM: %r", text[:1000])
            raise LLMError("LLM did not return valid JSON") from None

    def ping(self) -> str:
        return self.chat("Reply with the single word OK.", "Are you there?")


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

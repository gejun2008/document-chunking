"""Embedding and LLM clients.

Strategies depend only on the two small protocols below, so the company
embedding API, a LangChain ``Embeddings`` object or a local model can all be
plugged in. ``OpenAICompatibleEmbedder``/``OpenAICompatibleChat`` talk to any
endpoint that implements ``/embeddings`` and ``/chat/completions``.
"""

from __future__ import annotations

import hashlib
import os
import time
from typing import Protocol, Sequence

import httpx
import numpy as np


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return an ``(n, dim)`` float32 array, one row per input text."""
        ...


class LLM(Protocol):
    def complete(self, prompt: str, max_tokens: int = 128) -> str: ...


def _post_with_retry(client: httpx.Client, url: str, payload: dict, retries: int) -> dict:
    for attempt in range(retries + 1):
        try:
            resp = client.post(url, json=payload)
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < retries:
                raise httpx.HTTPStatusError("retryable", request=resp.request, response=resp)
            resp.raise_for_status()
            return resp.json()
        except (httpx.TransportError, httpx.HTTPStatusError):
            if attempt == retries:
                raise
            time.sleep(min(2**attempt, 30))
    raise RuntimeError("unreachable")


class OpenAICompatibleEmbedder:
    """Client for an OpenAI-compatible ``POST {base_url}/embeddings`` endpoint.

    Configure via arguments or ``EMBEDDING_BASE_URL`` / ``EMBEDDING_API_KEY`` /
    ``EMBEDDING_MODEL``.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        batch_size: int = 64,
        timeout: float = 60.0,
        retries: int = 3,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.base_url = (base_url or os.environ["EMBEDDING_BASE_URL"]).rstrip("/")
        self.model = model or os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
        key = api_key or os.environ.get("EMBEDDING_API_KEY", "")
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        headers.update(extra_headers or {})
        self.batch_size = batch_size
        self.retries = retries
        self._client = httpx.Client(timeout=timeout, headers=headers)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = list(texts[i : i + self.batch_size])
            data = _post_with_retry(
                self._client,
                f"{self.base_url}/embeddings",
                {"model": self.model, "input": batch},
                self.retries,
            )
            rows = sorted(data["data"], key=lambda d: d["index"])
            out.extend(r["embedding"] for r in rows)
        return np.asarray(out, dtype=np.float32)


class CachedEmbedder:
    """In-memory cache in front of any embedder (semantic chunking re-embeds a lot)."""

    def __init__(self, inner: Embedder) -> None:
        self.inner = inner
        self._cache: dict[str, np.ndarray] = {}

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        keys = [hashlib.sha1(t.encode()).hexdigest() for t in texts]
        missing = list({k: t for k, t in zip(keys, texts) if k not in self._cache}.items())
        if missing:
            vecs = self.inner.embed([t for _, t in missing])
            for (k, _), v in zip(missing, vecs):
                self._cache[k] = v
        return np.stack([self._cache[k] for k in keys]) if keys else np.zeros((0, 0), np.float32)


class OpenAICompatibleChat:
    """Client for an OpenAI-compatible ``POST {base_url}/chat/completions`` endpoint.

    Configure via arguments or ``LLM_BASE_URL`` / ``LLM_API_KEY`` / ``LLM_MODEL``.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 120.0,
        retries: int = 3,
        temperature: float = 0.0,
    ) -> None:
        self.base_url = (base_url or os.environ["LLM_BASE_URL"]).rstrip("/")
        self.model = model or os.environ["LLM_MODEL"]
        key = api_key or os.environ.get("LLM_API_KEY", "")
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        self.retries = retries
        self.temperature = temperature
        self._client = httpx.Client(timeout=timeout, headers=headers)

    def complete(self, prompt: str, max_tokens: int = 128) -> str:
        data = _post_with_retry(
            self._client,
            f"{self.base_url}/chat/completions",
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": self.temperature,
            },
            self.retries,
        )
        return data["choices"][0]["message"]["content"].strip()

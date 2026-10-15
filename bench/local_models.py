"""Local stand-ins for the company embedding API and LLM gateway, with disk caches.

Swap in ``OpenAICompatibleEmbedder`` / ``OpenAICompatibleChat`` to re-run the
benchmark against the real services (see ``run_bench.py --embedding-url``).
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from collections.abc import MutableMapping
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

CACHE_DIR = Path(__file__).resolve().parent.parent / ".bench_cache"


class SqliteDict(MutableMapping[str, str]):
    """Tiny persistent str->str map (used for LLM context cache)."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT)")
        self._lock = threading.Lock()

    @staticmethod
    def _h(k: str) -> str:
        return hashlib.sha1(k.encode()).hexdigest()

    def __getitem__(self, k: str) -> str:
        with self._lock:
            row = self._db.execute("SELECT v FROM kv WHERE k=?", (self._h(k),)).fetchone()
        if row is None:
            raise KeyError(k)
        return row[0]

    def __contains__(self, k: object) -> bool:
        if not isinstance(k, str):
            return False
        with self._lock:
            return self._db.execute("SELECT 1 FROM kv WHERE k=?", (self._h(k),)).fetchone() is not None

    def __setitem__(self, k: str, v: str) -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO kv VALUES (?,?)", (self._h(k), v))
            self._db.commit()

    def __delitem__(self, k: str) -> None:
        self._db.execute("DELETE FROM kv WHERE k=?", (self._h(k),))

    def __iter__(self) -> Iterator[str]:
        return (r[0] for r in self._db.execute("SELECT k FROM kv"))

    def __len__(self) -> int:
        return self._db.execute("SELECT COUNT(*) FROM kv").fetchone()[0]


class DiskCachedEmbedder:
    """Wrap any Embedder with a persistent sha1(text) -> vector cache."""

    def __init__(self, inner, name: str):
        self.inner = inner
        path = CACHE_DIR / f"emb_{name.replace('/', '_')}.sqlite"
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS e (k TEXT PRIMARY KEY, v BLOB)")
        self._mem: dict[str, np.ndarray] = {}

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        keys = [hashlib.sha1(t.encode()).hexdigest() for t in texts]
        missing = [k for k in dict.fromkeys(keys) if k not in self._mem]
        for i in range(0, len(missing), 500):
            sub = missing[i : i + 500]
            q = ",".join("?" * len(sub))
            for k, v in self._db.execute(f"SELECT k, v FROM e WHERE k IN ({q})", sub):
                self._mem[k] = np.frombuffer(v, dtype=np.float32)
        todo = {k: t for k, t in zip(keys, texts) if k not in self._mem}
        if todo:
            vecs = np.asarray(self.inner.embed(list(todo.values())), dtype=np.float32)
            rows = []
            for k, v in zip(todo, vecs):
                self._mem[k] = v
                rows.append((k, v.tobytes()))
            self._db.executemany("INSERT OR REPLACE INTO e VALUES (?,?)", rows)
            self._db.commit()
        return np.stack([self._mem[k] for k in keys])


class SentenceTransformerEmbedder:
    def __init__(self, model: str = "BAAI/bge-m3", device: str | None = None, max_seq_length: int = 1024, batch_size: int = 32):
        import torch
        from sentence_transformers import SentenceTransformer

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        # fp16 on a GPU is ~3.7x faster than fp32 with negligible quality change; CPU stays fp32
        kwargs = {"torch_dtype": torch.float16} if device != "cpu" else {}
        self.model = SentenceTransformer(model, device=device, model_kwargs=kwargs)
        self.model.max_seq_length = max_seq_length
        self.batch_size = batch_size

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        # sort by length so batches are padded less
        order = np.argsort([len(t) for t in texts])
        vecs = self.model.encode([texts[i] for i in order], batch_size=self.batch_size, normalize_embeddings=True)
        out = np.empty_like(vecs)
        out[order] = vecs
        return out.astype(np.float32)


class MLXChat:
    """Local LLM via mlx-lm (Apple Silicon). Single-threaded."""

    def __init__(self, model: str = "mlx-community/Qwen3-4B-Instruct-2507-4bit"):
        from mlx_lm import load

        self.model, self.tokenizer = load(model)
        self._lock = threading.Lock()

    def complete(self, prompt: str, max_tokens: int = 128) -> str:
        from mlx_lm import generate

        p = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False
        )
        with self._lock:
            return generate(self.model, self.tokenizer, p, max_tokens=max_tokens).strip()


class BatchedMLXChat(MLXChat):
    """Collects concurrent ``complete`` calls into ``batch_generate`` batches (~1.7x throughput).

    Use with ``ContextEnricher(max_workers=batch_size)``.
    """

    def __init__(self, model: str = "mlx-community/Qwen3-4B-Instruct-2507-4bit", batch_size: int = 8, wait_s: float = 0.05):
        super().__init__(model)
        import queue

        self.batch_size = batch_size
        self.wait_s = wait_s
        self._q: "queue.Queue[tuple[list[int], int, dict]]" = queue.Queue()
        threading.Thread(target=self._worker, daemon=True).start()

    def complete(self, prompt: str, max_tokens: int = 128) -> str:
        ids = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True)
        slot: dict = {"done": threading.Event()}
        self._q.put((ids, max_tokens, slot))
        slot["done"].wait()
        if "error" in slot:
            raise slot["error"]
        return slot["text"].strip()

    def _worker(self) -> None:
        import queue
        import time

        from mlx_lm import batch_generate

        while True:
            items = [self._q.get()]
            deadline = time.time() + self.wait_s
            while len(items) < self.batch_size:
                try:
                    items.append(self._q.get(timeout=max(0.0, deadline - time.time())))
                except queue.Empty:
                    break
            try:
                res = batch_generate(
                    self.model, self.tokenizer, [i[0] for i in items], max_tokens=max(i[1] for i in items)
                )
                for (_, _, slot), text in zip(items, res.texts):
                    slot["text"] = text
            except Exception as e:  # surface to every waiting caller
                for _, _, slot in items:
                    slot["error"] = e
            for _, _, slot in items:
                slot["done"].set()

"""Strategy 3 — semantic chunking.

Sentences are embedded with a small sliding window; a boundary is placed where
the cosine distance between neighbouring windows exceeds the given percentile
of all distances in the document. Tiny groups are merged into their neighbour
and oversized groups are re-packed so every chunk respects ``chunk_size``.
"""

from __future__ import annotations

import numpy as np

from ..clients import Embedder
from ..text import DEFAULT_ENCODING, get_encoder, split_sentences
from ..types import Chunk, finalize
from .packing import pack_spans, trim_span


def semantic_breakpoints(
    sentences: list[str], embedder: Embedder, window: int = 1, percentile: float = 95.0
) -> list[int]:
    """Return indices ``i`` such that a boundary falls after sentence ``i``."""
    n = len(sentences)
    if n < 2:
        return []
    windows = [" ".join(sentences[max(0, i - window) : i + window + 1]) for i in range(n)]
    emb = np.asarray(embedder.embed(windows), dtype=np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
    dist = 1.0 - np.sum(emb[:-1] * emb[1:], axis=1)
    threshold = np.percentile(dist, percentile)
    return [i for i, d in enumerate(dist) if d > threshold]


class SemanticChunker:
    def __init__(
        self,
        embedder: Embedder,
        chunk_size: int = 400,
        min_chunk_tokens: int = 60,
        breakpoint_percentile: float = 90.0,
        window: int = 1,
        encoding: str = DEFAULT_ENCODING,
    ):
        self.embedder = embedder
        self.chunk_size = chunk_size
        self.min_chunk_tokens = min_chunk_tokens
        self.breakpoint_percentile = breakpoint_percentile
        self.window = window
        self.encoding = encoding

    def chunk(self, text: str) -> list[Chunk]:
        spans = split_sentences(text)
        if not spans:
            return []
        enc = get_encoder(self.encoding)
        sent_text = [text[s:e] for s, e in spans]
        sent_tokens = [len(enc.encode(t, disallowed_special=())) for t in sent_text]
        cuts = set(
            semantic_breakpoints(sent_text, self.embedder, self.window, self.breakpoint_percentile)
        )

        # groups of sentence indices [a, b)
        groups: list[list[int]] = []
        a = 0
        for i in range(len(spans)):
            if i in cuts or i == len(spans) - 1:
                groups.append([a, i + 1])
                a = i + 1

        def tokens(g: list[int]) -> int:
            return sum(sent_tokens[g[0] : g[1]])

        merged: list[list[int]] = []
        for g in groups:
            if merged and (tokens(g) < self.min_chunk_tokens or tokens(merged[-1]) < self.min_chunk_tokens):
                if tokens(merged[-1]) + tokens(g) <= self.chunk_size:
                    merged[-1][1] = g[1]
                    continue
            merged.append(g)

        chunks: list[Chunk] = []
        for g in merged:
            if tokens(g) > self.chunk_size:
                chunks.extend(pack_spans(text, spans[g[0] : g[1]], self.chunk_size, 0, self.encoding))
            else:
                s, e = trim_span(text, spans[g[0]][0], spans[g[1] - 1][1])
                if e > s:
                    chunks.append(Chunk(text=text[s:e], start=s, end=e))
        return finalize(chunks, "semantic")

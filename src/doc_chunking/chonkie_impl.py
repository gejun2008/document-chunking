"""Option 2 — chunkers built on chonkie (``pip install doc-chunking[chonkie]``).

Notes found while integrating:
* ``RecursiveChunker.from_recipe("markdown")`` downloads the recipe from the
  Hugging Face hub at runtime and needs the ``chonkie[hub]`` extra; the recipe
  is vendored below instead so production needs no network access.
* chonkie has no context-enrichment chunker; the self-built ``ContextEnricher``
  wraps any chonkie-based chunker.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from chonkie import (
    BaseEmbeddings,
    MarkdownChef,
    RecursiveChunker,
    RecursiveLevel,
    RecursiveRules,
    SemanticChunker,
    SentenceChunker,
)

from .clients import Embedder
from .selfbuilt.packing import trim_span
from .text import DEFAULT_ENCODING, get_encoder
from .types import Chunk, finalize

# Sentence delimiters incl. CJK so Chinese documents are not one giant sentence.
DELIMS = [". ", "! ", "? ", "\n", "。", "！", "？", "；"]

# Vendored from chonkie-ai/recipes markdown_en.json (v0.1.0).
MARKDOWN_RULES = RecursiveRules(
    levels=[
        RecursiveLevel(delimiters=["######", "#####", "####", "###", "##", "#"], include_delim="next"),
        RecursiveLevel(delimiters=["\n\n", "\n\r"]),
        RecursiveLevel(delimiters=["\n", "\r"]),
        RecursiveLevel(delimiters=[". ", "! ", "? ", "。", "！", "？"]),
        RecursiveLevel(),
    ]
)


def _to_chunks(text: str, items: list[Any], offset: int = 0) -> list[Chunk]:
    out = []
    for it in items:
        s, e = trim_span(text, offset + it.start_index, offset + it.end_index)
        if e > s:
            out.append(Chunk(text=text[s:e], start=s, end=e))
    return out


class ChonkieFixedSizeChunker:
    """chonkie ``SentenceChunker``: token-bounded, sentence-aligned, with overlap."""

    def __init__(self, chunk_size: int = 400, overlap: int = 50, encoding: str = DEFAULT_ENCODING):
        self._chunker = SentenceChunker(
            tokenizer=get_encoder(encoding), chunk_size=chunk_size, chunk_overlap=overlap, delim=DELIMS
        )

    def chunk(self, text: str) -> list[Chunk]:
        return finalize(_to_chunks(text, self._chunker.chunk(text)), "fixed_size")


class ChonkieMarkdownChunker:
    """``MarkdownChef`` keeps tables/code whole; prose goes through ``RecursiveChunker``."""

    def __init__(self, chunk_size: int = 400, encoding: str = DEFAULT_ENCODING):
        enc = get_encoder(encoding)
        self.chunk_size = chunk_size
        self._chef = MarkdownChef(tokenizer=enc)
        self._recursive = RecursiveChunker(tokenizer=enc, chunk_size=chunk_size, rules=MARKDOWN_RULES)
        self._enc = enc

    def chunk(self, text: str) -> list[Chunk]:
        doc = self._chef.parse(text)
        chunks: list[Chunk] = []
        for seg in doc.chunks:  # prose segments between tables / code blocks
            sub = self._recursive.chunk(text[seg.start_index : seg.end_index])
            chunks.extend(_to_chunks(text, sub, seg.start_index))
        for blk in [*doc.tables, *doc.code]:  # atomic blocks
            s, e = trim_span(text, blk.start_index, blk.end_index)
            if e <= s:
                continue
            if len(self._enc.encode(text[s:e], disallowed_special=())) <= self.chunk_size:
                chunks.append(Chunk(text=text[s:e], start=s, end=e))
            else:
                chunks.extend(_to_chunks(text, self._recursive.chunk(text[s:e]), s))
        chunks.sort(key=lambda c: c.start)
        return finalize(chunks, "layout_aware")


class _EmbeddingsAdapter(BaseEmbeddings):
    """Expose any ``Embedder`` (e.g. the company API client) to chonkie."""

    def __init__(self, embedder: Embedder, dimension: int, encoding: str = DEFAULT_ENCODING):
        super().__init__()
        self._embedder = embedder
        self._dim = dimension
        self._enc = get_encoder(encoding)

    def embed(self, text: str) -> np.ndarray:
        return np.asarray(self._embedder.embed([text])[0], dtype=np.float32)

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        return list(np.asarray(self._embedder.embed(texts), dtype=np.float32))

    @property
    def dimension(self) -> int:
        return self._dim

    def get_tokenizer(self) -> Any:
        return self._enc

    @classmethod
    def _is_available(cls) -> bool:
        return True


class ChonkieSemanticChunker:
    def __init__(
        self,
        embedder: Embedder,
        chunk_size: int = 400,
        threshold: float = 0.8,
        similarity_window: int = 3,
        encoding: str = DEFAULT_ENCODING,
    ):
        dim = int(np.asarray(embedder.embed(["dimension probe"])).shape[1])
        self._chunker = SemanticChunker(
            embedding_model=_EmbeddingsAdapter(embedder, dim, encoding),
            chunk_size=chunk_size,
            threshold=threshold,
            similarity_window=similarity_window,
            delim=DELIMS,
        )

    def chunk(self, text: str) -> list[Chunk]:
        return finalize(_to_chunks(text, self._chunker.chunk(text)), "semantic")

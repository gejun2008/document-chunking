"""Option 3 — LangChain text splitters + self-built semantic / context enrichment.

* fixed_size   -> ``RecursiveCharacterTextSplitter`` (tiktoken length)
* layout_aware -> ``MarkdownHeaderTextSplitter`` + recursive fallback for long sections
* semantic     -> ``SemanticTextSplitter``: a LangChain ``TextSplitter`` over the
                  self-built algorithm, taking any LangChain ``Embeddings``
* context      -> ``ContextualEnrichmentTransformer``: a LangChain
                  ``BaseDocumentTransformer`` over the self-built enricher

Everything also exposes ``chunk(text) -> list[Chunk]`` so it plugs into the same
interface as options 1 and 2. Never use ``NLTKTextSplitter`` (pulls in nltk).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from langchain_core.documents import BaseDocumentTransformer, Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
    TextSplitter,
)

from .clients import Embedder
from .selfbuilt.context import ContextEnricher
from .selfbuilt.semantic import SemanticChunker
from .text import DEFAULT_ENCODING, locate
from .types import Chunk, Chunker, finalize

SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", ". ", "! ", "? ", " ", ""]
HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]


def to_documents(chunks: Sequence[Chunk]) -> list[Document]:
    """Convert ``Chunk`` objects into LangChain ``Document`` objects."""
    return [
        Document(
            page_content=c.text,
            metadata={**c.metadata, "start_index": c.start, "end_index": c.end, "embed_text": c.text_for_embedding},
        )
        for c in chunks
    ]


def _recursive(chunk_size: int, overlap: int, encoding: str) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=encoding,
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=SEPARATORS,
        keep_separator="end",
        add_start_index=True,
    )


def _chunks_from_pieces(
    text: str, pieces: list[str], meta: list[dict] | None = None, overlapping: bool = False
) -> list[Chunk]:
    chunks: list[Chunk] = []
    cursor = 0
    for i, piece in enumerate(pieces):
        span = locate(text, piece, cursor)
        if span is None:
            continue
        s, e = span
        md = dict(meta[i]) if meta else {}
        chunks.append(Chunk(text=text[s:e], start=s, end=e, metadata=md, embed_text=None))
        cursor = s + 1 if overlapping else e
    return chunks


class LCFixedSizeChunker:
    def __init__(self, chunk_size: int = 400, overlap: int = 50, encoding: str = DEFAULT_ENCODING):
        self.splitter = _recursive(chunk_size, overlap, encoding)

    def chunk(self, text: str) -> list[Chunk]:
        # LangChain's own start_index can drift when overlap is used; re-locate instead.
        pieces = self.splitter.split_text(text)
        return finalize(_chunks_from_pieces(text, pieces, overlapping=True), "fixed_size")


class LCMarkdownChunker:
    def __init__(
        self,
        chunk_size: int = 400,
        include_heading_path: bool = True,
        encoding: str = DEFAULT_ENCODING,
    ):
        self.headers = MarkdownHeaderTextSplitter(HEADERS, strip_headers=False)
        self.recursive = _recursive(chunk_size, 0, encoding)
        self.include_heading_path = include_heading_path

    def split_documents(self, text: str) -> list[Document]:
        return self.recursive.split_documents(self.headers.split_text(text))

    def chunk(self, text: str) -> list[Chunk]:
        docs = self.split_documents(text)
        paths = [[d.metadata[k] for _, k in HEADERS if k in d.metadata] for d in docs]
        chunks = _chunks_from_pieces(
            text, [d.page_content for d in docs], [{"heading_path": p} for p in paths]
        )
        if self.include_heading_path:
            for c in chunks:
                if c.metadata["heading_path"]:
                    c.embed_text = " > ".join(c.metadata["heading_path"]) + "\n\n" + c.text
        return finalize(chunks, "layout_aware")


class _LCEmbedder:
    """Adapt a LangChain ``Embeddings`` to the internal ``Embedder`` protocol."""

    def __init__(self, embeddings: Embeddings):
        self.embeddings = embeddings

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(self.embeddings.embed_documents(list(texts)), dtype=np.float32)


class SemanticTextSplitter(TextSplitter):
    """LangChain ``TextSplitter`` running the self-built semantic algorithm."""

    def __init__(
        self,
        embeddings: Embeddings | Embedder,
        chunk_size: int = 400,
        breakpoint_percentile: float = 90.0,
        min_chunk_tokens: int = 60,
        encoding: str = DEFAULT_ENCODING,
        **kwargs: Any,
    ):
        super().__init__(chunk_size=chunk_size, chunk_overlap=0, add_start_index=True, **kwargs)
        embedder = _LCEmbedder(embeddings) if isinstance(embeddings, Embeddings) else embeddings
        self._impl = SemanticChunker(
            embedder,
            chunk_size=chunk_size,
            breakpoint_percentile=breakpoint_percentile,
            min_chunk_tokens=min_chunk_tokens,
            encoding=encoding,
        )

    def split_text(self, text: str) -> list[str]:
        return [c.text for c in self._impl.chunk(text)]

    def chunk(self, text: str) -> list[Chunk]:
        return self._impl.chunk(text)


class _LCLLM:
    """Adapt a LangChain chat model (anything with ``.invoke``) to the ``LLM`` protocol."""

    def __init__(self, model: Any):
        self.model = model

    def complete(self, prompt: str, max_tokens: int = 128) -> str:
        out = self.model.invoke(prompt)
        return getattr(out, "content", out).strip()


class _TextSplitterChunker:
    def __init__(self, splitter: TextSplitter):
        self.splitter = splitter

    def chunk(self, text: str) -> list[Chunk]:
        if hasattr(self.splitter, "chunk"):
            return self.splitter.chunk(text)  # type: ignore[no-any-return]
        return _chunks_from_pieces(text, self.splitter.split_text(text), overlapping=True)


class ContextualEnrichmentTransformer(BaseDocumentTransformer):
    """LangChain document transformer: full documents in, context-enriched chunks out.

    The original chunk text stays in ``page_content``; the generated context is in
    ``metadata["context"]`` and the text to embed in ``metadata["embed_text"]``.
    """

    def __init__(self, base: TextSplitter | Chunker, llm: Any, **enricher_kwargs: Any):
        base_chunker = _TextSplitterChunker(base) if isinstance(base, TextSplitter) else base
        llm_impl = llm if hasattr(llm, "complete") else _LCLLM(llm)
        self.enricher = ContextEnricher(base_chunker, llm_impl, **enricher_kwargs)

    def chunk(self, text: str) -> list[Chunk]:
        return self.enricher.chunk(text)

    def transform_documents(self, documents: Sequence[Document], **kwargs: Any) -> list[Document]:
        out: list[Document] = []
        for doc in documents:
            for d in to_documents(self.enricher.chunk(doc.page_content)):
                d.metadata = {**doc.metadata, **d.metadata}
                out.append(d)
        return out


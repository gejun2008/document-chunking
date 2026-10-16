"""Core types shared by every chunking strategy and implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class Chunk:
    """A contiguous span of the source document.

    ``start``/``end`` are character offsets into the original text, so
    ``text == source[start:end]`` always holds. ``embed_text`` is what should be
    sent to the embedding model; strategies such as context enrichment or
    heading-path prefixing set it to something richer than ``text``.
    """

    text: str
    start: int
    end: int
    index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    embed_text: str | None = None
    id: str = ""  # stable ID, see doc_chunking.ids.assign_stable_ids

    @property
    def text_for_embedding(self) -> str:
        return self.embed_text if self.embed_text is not None else self.text


@runtime_checkable
class Chunker(Protocol):
    """Every strategy exposes the same single method."""

    def chunk(self, text: str) -> list[Chunk]: ...


def finalize(chunks: list[Chunk], strategy: str) -> list[Chunk]:
    """Re-number chunks and stamp the strategy name into metadata."""
    for i, c in enumerate(chunks):
        c.index = i
        c.metadata.setdefault("strategy", strategy)
    return chunks

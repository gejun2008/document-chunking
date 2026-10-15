"""Strategy 1 — fixed-size chunks with overlap, cut on natural boundaries.

Recursive split: paragraphs, then lines, then sentences (EN + CJK), then words,
then tokens; small pieces are merged up to ``chunk_size`` with whole-piece
overlap (see ``packing.recursive_split``).
"""

from __future__ import annotations

from ..text import DEFAULT_ENCODING
from ..types import Chunk, finalize
from .packing import recursive_split


class FixedSizeChunker:
    def __init__(self, chunk_size: int = 400, overlap: int = 50, encoding: str = DEFAULT_ENCODING):
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.encoding = encoding

    def chunk(self, text: str) -> list[Chunk]:
        return finalize(recursive_split(text, self.chunk_size, self.overlap, self.encoding), "fixed_size")

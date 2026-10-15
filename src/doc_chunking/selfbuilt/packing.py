"""Greedy packing of small spans (sentences, blocks) into token-bounded chunks,
and recursive paragraph / line / sentence / word splitting."""

from __future__ import annotations

import re
from typing import Callable

from ..text import DEFAULT_ENCODING, get_encoder, split_by_tokens, split_sentences
from ..types import Chunk


def trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def pack_spans(
    text: str,
    spans: list[tuple[int, int]],
    chunk_size: int,
    overlap: int = 0,
    encoding: str = DEFAULT_ENCODING,
    metadata: dict | None = None,
) -> list[Chunk]:
    """Pack consecutive ``spans`` into chunks of at most ``chunk_size`` tokens.

    Units larger than ``chunk_size`` are hard-split on token boundaries. When
    ``overlap > 0`` each new chunk starts with the trailing units of the previous
    chunk, up to ``overlap`` tokens, so the overlap always falls on a unit
    (sentence) boundary. Used by the markdown and semantic chunkers.
    """
    enc = get_encoder(encoding)
    units: list[tuple[int, int, int]] = []  # (start, end, tokens)
    for s, e in spans:
        n = len(enc.encode(text[s:e], disallowed_special=()))
        if n > chunk_size:
            for ps, pe in split_by_tokens(text, s, e, chunk_size, encoding):
                units.append((ps, pe, len(enc.encode(text[ps:pe], disallowed_special=()))))
        elif e > s:
            units.append((s, e, n))

    chunks: list[Chunk] = []

    def emit(part: list[tuple[int, int, int]]) -> None:
        s, e = trim_span(text, part[0][0], part[-1][1])
        if e > s:
            chunks.append(Chunk(text=text[s:e], start=s, end=e, metadata=dict(metadata or {})))

    cur: list[tuple[int, int, int]] = []
    cur_tokens = 0
    for u in units:
        if cur and cur_tokens + u[2] > chunk_size:
            emit(cur)
            carry: list[tuple[int, int, int]] = []
            carried = 0
            for prev in reversed(cur[1:]):  # never carry the whole chunk -> guaranteed progress
                if carried + prev[2] > overlap:
                    break
                carry.insert(0, prev)
                carried += prev[2]
            if carried + u[2] > chunk_size:
                carry, carried = [], 0
            cur, cur_tokens = carry, carried
        cur.append(u)
        cur_tokens += u[2]
    if cur:
        emit(cur)
    return chunks


# --------------------------------------------------------------------------- #
# Recursive splitting (paragraph -> line -> sentence -> word -> token)
# --------------------------------------------------------------------------- #

_PARAGRAPHS = re.compile(r"\n[ \t]*\n\s*")
_LINES = re.compile(r"\n\s*")
_WORDS = re.compile(r"\s+")


def _split_on(text: str, start: int, end: int, pattern: re.Pattern) -> list[tuple[int, int]]:
    """Tile ``text[start:end]``, cutting after each match (separator kept at the end)."""
    spans, cur = [], start
    for m in pattern.finditer(text, start, end):
        if m.end() > cur and m.start() > cur:
            spans.append((cur, m.end()))
            cur = m.end()
        elif spans:  # leading/duplicate separator: glue onto the previous piece
            spans[-1] = (spans[-1][0], m.end())
            cur = m.end()
    if cur < end:
        spans.append((cur, end))
    return spans


# Each level tiles a span into smaller spans; the last resort is a token hard-split.
_LEVELS: list[Callable[[str, int, int], list[tuple[int, int]]]] = [
    lambda t, s, e: _split_on(t, s, e, _PARAGRAPHS),
    lambda t, s, e: _split_on(t, s, e, _LINES),
    split_sentences,
    lambda t, s, e: _split_on(t, s, e, _WORDS),
]


def recursive_split(
    text: str,
    chunk_size: int,
    overlap: int = 0,
    encoding: str = DEFAULT_ENCODING,
    start: int = 0,
    end: int | None = None,
    metadata: dict | None = None,
) -> list[Chunk]:
    """Split ``text[start:end]`` the way a recursive character splitter does.

    The text is cut into paragraphs; paragraphs that fit are merged greedily up
    to ``chunk_size`` tokens, paragraphs that do not are cut into lines, then
    sentences (EN + CJK), then words, then hard-split on tokens. Small pieces
    are always merged at the level they were produced, so a chunk never mixes
    the tail of one oversized paragraph with the head of the next. Overlap
    carries whole trailing pieces of that level (sentences, inside a long
    paragraph) up to ``overlap`` tokens.
    """
    enc = get_encoder(encoding)
    end = len(text) if end is None else end

    def ntok(s: int, e: int) -> int:
        return len(enc.encode(text[s:e], disallowed_special=()))

    chunks: list[Chunk] = []

    def emit(part: list[tuple[int, int, int]]) -> None:
        s, e = trim_span(text, part[0][0], part[-1][1])
        if e <= s:
            return
        if len(part) > 1 and ntok(s, e) > chunk_size:  # summed counts are an estimate; re-check
            cut = len(part) // 2
            emit(part[:cut])
            emit(part[cut:])
            return
        chunks.append(Chunk(text=text[s:e], start=s, end=e, metadata=dict(metadata or {})))

    def merge(units: list[tuple[int, int, int]]) -> None:
        cur: list[tuple[int, int, int]] = []
        total = 0
        for u in units:
            if cur and total + u[2] > chunk_size:
                emit(cur)
                # keep a tail of at most ``overlap`` tokens that still leaves room for ``u``
                while cur and (total > overlap or total + u[2] > chunk_size):
                    total -= cur.pop(0)[2]
            cur.append(u)
            total += u[2]
        if cur:
            emit(cur)

    def split(s: int, e: int, level: int) -> None:
        if level < len(_LEVELS):
            pieces = _LEVELS[level](text, s, e)
        else:
            pieces = split_by_tokens(text, s, e, chunk_size, encoding)
        if len(pieces) == 1 and level < len(_LEVELS):
            split(s, e, level + 1)  # separator absent at this level
            return
        good: list[tuple[int, int, int]] = []
        for ps, pe in pieces:
            if not text[ps:pe].strip():
                continue
            n = ntok(ps, pe)
            if n <= chunk_size or level >= len(_LEVELS):
                good.append((ps, pe, n))
                continue
            if good:
                merge(good)
                good = []
            split(ps, pe, level + 1)
        if good:
            merge(good)

    if text[start:end].strip():
        split(start, end, 0)
    return chunks

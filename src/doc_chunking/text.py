"""Tokenization, sentence splitting and offset helpers.

Deliberately dependency-light: tiktoken for token counts and a regex sentence
splitter (no nltk — see CVE-2026-81726) that understands both English and CJK
punctuation.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache

import tiktoken

DEFAULT_ENCODING = os.environ.get("DOC_CHUNKING_TIKTOKEN_ENCODING", "cl100k_base")


@lru_cache(maxsize=4)
def get_encoder(name: str = DEFAULT_ENCODING) -> tiktoken.Encoding:
    # In air-gapped production set TIKTOKEN_CACHE_DIR to a pre-populated cache.
    return tiktoken.get_encoding(name)


def count_tokens(text: str, encoding: str = DEFAULT_ENCODING) -> int:
    return len(get_encoder(encoding).encode(text, disallowed_special=()))


# --------------------------------------------------------------------------- #
# Sentence splitting
# --------------------------------------------------------------------------- #

_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e",
    "inc", "ltd", "a.m", "p.m", "co", "corp", "no", "fig", "al", "u.s", "jan", "feb", "mar",
    "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
}

# A boundary is the end of: western terminal punctuation followed by whitespace,
# CJK terminal punctuation (no whitespace needed), or a line break.
_BOUNDARY = re.compile(
    r"""
    (?P<west>[.!?]+["'\)\]”’]*\s+)
  | (?P<cjk>[。！？；…]+["'\)\]”’）」』]*\s*)
  | (?P<nl>\n\s*)
    """,
    re.VERBOSE,
)


def _is_abbreviation(text: str, dot_pos: int) -> bool:
    j = dot_pos
    while j > 0 and (text[j - 1].isalpha() or text[j - 1] == "."):
        j -= 1
    word = text[j:dot_pos].lower()
    return word in _ABBREVIATIONS or (len(word) == 1 and word.isalpha())


def split_sentences(text: str, start: int = 0, end: int | None = None) -> list[tuple[int, int]]:
    """Return sentence spans that tile ``text[start:end]`` with no gaps.

    Trailing whitespace is attached to the preceding sentence so that
    concatenating all spans reproduces the input exactly.
    """
    end = len(text) if end is None else end
    spans: list[tuple[int, int]] = []
    cur = start
    for m in _BOUNDARY.finditer(text, start, end):
        if m.group("west") and (
            _is_abbreviation(text, m.start()) or (m.end() < end and text[m.end()].islower())
        ):
            continue
        cut = m.end()
        if cut <= cur:
            continue
        if text[cur:cut].strip():
            spans.append((cur, cut))
        elif spans:  # pure whitespace: glue onto previous sentence
            spans[-1] = (spans[-1][0], cut)
        else:
            continue
        cur = cut
    if cur < end:
        if text[cur:end].strip() or not spans:
            spans.append((cur, end))
        else:
            spans[-1] = (spans[-1][0], end)
    return spans


def split_by_tokens(
    text: str, start: int, end: int, max_tokens: int, encoding: str = DEFAULT_ENCODING
) -> list[tuple[int, int]]:
    """Hard-split ``text[start:end]`` into spans of at most ``max_tokens`` tokens."""
    enc = get_encoder(encoding)
    piece = text[start:end]
    tokens = enc.encode(piece, disallowed_special=())
    if len(tokens) <= max_tokens:
        return [(start, end)]
    _, offsets = enc.decode_with_offsets(tokens)
    cuts = [offsets[i] for i in range(max_tokens, len(tokens), max_tokens)]
    bounds = [0, *sorted(set(c for c in cuts if 0 < c < len(piece))), len(piece)]
    return [(start + a, start + b) for a, b in zip(bounds, bounds[1:]) if b > a]


# --------------------------------------------------------------------------- #
# Offset recovery for libraries that return text but not positions
# --------------------------------------------------------------------------- #


def locate(text: str, piece: str, cursor: int = 0) -> tuple[int, int] | None:
    """Find ``piece`` in ``text`` at or after ``cursor`` (whitespace-tolerant)."""
    stripped = piece.strip()
    if not stripped:
        return None
    pos = text.find(stripped, cursor)
    if pos == -1:
        pos = text.find(stripped)
    if pos != -1:
        return pos, pos + len(stripped)
    # Fall back to a whitespace-insensitive regex over the first/last words.
    words = stripped.split()
    pattern = r"\s+".join(re.escape(w) for w in words)
    m = re.compile(pattern).search(text, cursor) or re.compile(pattern).search(text)
    return (m.start(), m.end()) if m else None

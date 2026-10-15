"""Strategy 2 — layout-aware structural chunking over a Markdown AST.

* Sections are delimited by headings (up to ``max_heading_level``).
* Small consecutive sections are merged while they fit in ``chunk_size``.
* Oversized sections fall back to ``recursive_split`` (paragraph -> line ->
  sentence -> word -> token), the same splitter as fixed_size; tables and code
  blocks stay whole whenever they fit (oversized tables are split on rows) and
  are merged with neighbouring pieces when the result still fits.
* Every chunk carries its ``heading_path``; when ``include_heading_path`` is on it
  is also prepended to ``embed_text`` so retrieval sees the section context.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from markdown_it import MarkdownIt

from ..text import DEFAULT_ENCODING, count_tokens
from ..types import Chunk, finalize
from .packing import pack_spans, recursive_split, trim_span

_ATOMIC = {"table", "fence", "code_block", "html_block"}


@dataclass
class _Block:
    kind: str
    start: int
    end: int
    level: int = 0  # heading level, 0 for non-headings
    title: str = ""


@dataclass
class _Section:
    path: list[str]
    blocks: list[_Block] = field(default_factory=list)

    @property
    def start(self) -> int:
        return self.blocks[0].start

    @property
    def end(self) -> int:
        return self.blocks[-1].end


class MarkdownChunker:
    def __init__(
        self,
        chunk_size: int = 400,
        overlap: int = 0,
        max_heading_level: int = 3,
        include_heading_path: bool = True,
        encoding: str = DEFAULT_ENCODING,
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.max_heading_level = max_heading_level
        self.include_heading_path = include_heading_path
        self.encoding = encoding
        self._md = MarkdownIt("commonmark").enable("table")

    # ------------------------------------------------------------------ parse
    def _blocks(self, text: str) -> list[_Block]:
        line_starts = [0] + [i + 1 for i, ch in enumerate(text) if ch == "\n"]

        def offset(line: int) -> int:
            return line_starts[line] if line < len(line_starts) else len(text)

        tokens = self._md.parse(text)
        blocks: list[_Block] = []
        for i, tok in enumerate(tokens):
            if tok.level != 0 or tok.map is None or tok.nesting == -1:
                continue
            kind = tok.type.removesuffix("_open")
            start, end = offset(tok.map[0]), offset(tok.map[1])
            if kind == "heading":
                title = tokens[i + 1].content if i + 1 < len(tokens) else ""
                blocks.append(_Block(kind, start, end, int(tok.tag[1]), title.strip()))
            else:
                blocks.append(_Block(kind, start, end))
        return blocks

    def _sections(self, blocks: list[_Block]) -> list[_Section]:
        sections: list[_Section] = []
        stack: list[tuple[int, str]] = []
        current = _Section(path=[])
        for b in blocks:
            if b.kind == "heading" and b.level <= self.max_heading_level:
                if current.blocks:
                    sections.append(current)
                while stack and stack[-1][0] >= b.level:
                    stack.pop()
                stack.append((b.level, b.title))
                current = _Section(path=[t for _, t in stack])
            current.blocks.append(b)
        if current.blocks:
            sections.append(current)
        return sections

    # ------------------------------------------------------------------ chunk
    def _split_section(self, text: str, section: _Section) -> list[Chunk]:
        """Split an oversized section: prose runs go through ``recursive_split``,
        tables / code blocks stay whole when they fit."""
        pieces: list[tuple[Chunk, bool]] = []  # (chunk, is a whole atomic block)
        run: list[_Block] = []

        def split_run() -> None:
            if run:
                for c in recursive_split(
                    text, self.chunk_size, self.overlap, self.encoding, run[0].start, run[-1].end
                ):
                    pieces.append((c, False))
                run.clear()

        for b in section.blocks:
            if b.kind not in _ATOMIC:
                run.append(b)
                continue
            split_run()
            if count_tokens(text[b.start : b.end], self.encoding) <= self.chunk_size:
                c = self._make_chunk(text, b.start, b.end, [])
                if c:
                    pieces.append((c, True))
            elif b.kind == "table":  # split oversized tables on rows
                rows, pos = [], b.start
                for line in text[b.start : b.end].splitlines(keepends=True):
                    rows.append((pos, pos + len(line)))
                    pos += len(line)
                for c in pack_spans(text, rows, self.chunk_size, self.overlap, self.encoding):
                    pieces.append((c, False))
            else:  # oversized code / html: lines, then tokens
                for c in recursive_split(
                    text, self.chunk_size, self.overlap, self.encoding, b.start, b.end
                ):
                    pieces.append((c, False))
        split_run()

        # glue whole atomic blocks to adjacent pieces when the result still fits
        out: list[tuple[Chunk, bool]] = []
        for c, atomic in pieces:
            if out:
                prev, prev_atomic = out[-1]
                if (
                    (atomic or prev_atomic)
                    and c.start >= prev.end
                    and count_tokens(text[prev.start : c.end], self.encoding) <= self.chunk_size
                ):
                    out[-1] = (self._make_chunk(text, prev.start, c.end, []), True)
                    continue
            out.append((c, atomic))
        return [c for c, _ in out]

    def _make_chunk(self, text: str, start: int, end: int, path: list[str]) -> Chunk | None:
        s, e = trim_span(text, start, end)
        if e <= s:
            return None
        return Chunk(text=text[s:e], start=s, end=e, metadata={"heading_path": list(path)})

    def chunk(self, text: str) -> list[Chunk]:
        sections = self._sections(self._blocks(text))
        chunks: list[Chunk] = []
        group: list[_Section] = []
        group_tokens = 0

        def flush() -> None:
            nonlocal group, group_tokens
            if not group:
                return
            if len(group) == 1 and group_tokens > self.chunk_size:
                sec = group[0]
                for c in self._split_section(text, sec):
                    c.metadata["heading_path"] = list(sec.path)
                    chunks.append(c)
            else:
                common = _common_prefix([s.path for s in group])
                c = self._make_chunk(text, group[0].start, group[-1].end, common)
                if c:
                    chunks.append(c)
            group, group_tokens = [], 0

        for sec in sections:
            n = count_tokens(text[sec.start : sec.end], self.encoding)
            if group and group_tokens + n > self.chunk_size:
                flush()
            group.append(sec)
            group_tokens += n
            if group_tokens > self.chunk_size:
                flush()
        flush()

        if self.include_heading_path:
            for c in chunks:
                path = c.metadata.get("heading_path") or []
                if path:
                    c.embed_text = " > ".join(path) + "\n\n" + c.text
        return finalize(chunks, "layout_aware")


def _common_prefix(paths: list[list[str]]) -> list[str]:
    if not paths:
        return []
    prefix = paths[0]
    for p in paths[1:]:
        n = 0
        while n < min(len(prefix), len(p)) and prefix[n] == p[n]:
            n += 1
        prefix = prefix[:n]
    return list(prefix)

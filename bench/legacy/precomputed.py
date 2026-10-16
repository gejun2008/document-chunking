"""Chunkers that replay the chunks exported by export_llamaindex.py (the legacy design)."""

from __future__ import annotations

import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from doc_chunking.ids import assign_stable_ids
from doc_chunking.selfbuilt import ParentChildChunker
from doc_chunking.types import Chunk, finalize

ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=None)
def _rows(dataset: str) -> dict[tuple[str, str], list[dict]]:
    path = ROOT / "results" / "legacy_chunks" / f"{dataset.removesuffix('-lite')}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run bench/legacy/export_llamaindex.py first")
    out: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for line in path.read_text().splitlines():
        r = json.loads(line)
        out[(r["doc_id"], r["strategy"])].append(r)
    return out


class LegacyFlat:
    """Legacy TokenTextSplitter / MarkdownNodeParser output for one strategy."""

    def __init__(self, dataset: str, strategy: str):
        self.dataset, self.strategy = dataset, strategy

    def chunk(self, text: str, doc_id: str = "doc") -> list[Chunk]:
        rows = _rows(self.dataset)[(doc_id, self.strategy)]
        chunks = [Chunk(text=text[r["start"] : r["end"]], start=r["start"], end=r["end"]) for r in rows]
        return finalize([c for c in chunks if c.text.strip()], self.strategy)


class LegacyParentChild:
    """Legacy parents (1024/0) and children (256/30), with the legacy summary step when ``llm`` is set."""

    def __init__(self, dataset: str, llm=None, cache=None, max_workers: int = 8):
        self.dataset = dataset
        self.summarizer = ParentChildChunker(llm=llm, cache=cache, max_workers=max_workers) if llm else None
        self.strategy = "parent_summary" if llm else "parent_child"

    def chunk(self, text: str, doc_id: str = "doc") -> list[Chunk]:
        rows = _rows(self.dataset)[(doc_id, "parent_child")]
        parents = [Chunk(text=text[r["start"] : r["end"]], start=r["start"], end=r["end"], metadata={"level": "parent"})
                   for r in rows if r["level"] == "parent"]
        assign_stable_ids(parents, doc_id, "legacy", prefix="p")
        summaries = self.summarizer._summaries(parents) if self.summarizer else [""] * len(parents)
        out = []
        for pi, (p, summary) in enumerate(zip(parents, summaries)):
            out.append(p)
            for r in rows:
                if r["level"] == "child" and r["parent"] == pi and text[r["start"] : r["end"]].strip():
                    c = Chunk(text=text[r["start"] : r["end"]], start=r["start"], end=r["end"],
                              metadata={"level": "child", "parent_id": p.id})
                    if summary:
                        c.embed_text = f"Context summary: {summary}\n\n{c.text}"
                    out.append(c)
        return finalize(out, self.strategy)

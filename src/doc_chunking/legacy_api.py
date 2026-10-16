"""Response shape of the legacy chunking service, so callers need no changes.

Legacy fields are kept with their legacy meaning:

* ``text``      what gets embedded (for context_enriched: summary + child text)
* ``raw_text``  the original chunk text
* ``metadata``  ``hierarchical_path``, ``parent_id``, ``prev_id``, ``next_id``

Added fields (callers that do not know them can ignore them): ``metadata``
``doc_id``, ``level``, ``start``, ``end``, ``summary``; and a top-level
``parents`` list holding each parent's ID, text and offsets, so a parent can be
restored from a child's ``parent_id`` — the legacy response only had the ID.

Note the naming flip versus ``Chunk``: ``Chunk.text`` is the original text and
``Chunk.text_for_embedding`` the embedded text.
"""

from __future__ import annotations

from typing import Any, Iterable, Literal

from .ids import assign_stable_ids, config_key
from .types import Chunk


def to_legacy_response(
    chunks: Iterable[Chunk],
    doc_id: str = "doc",
    id_style: Literal["stable", "sequential"] = "stable",
) -> dict[str, Any]:
    """Convert chunks from any strategy into ``{"chunks": [...], "parents": [...]}``.

    ``id_style="sequential"`` reproduces the legacy ``chunk-0001`` / ``parent-0001``
    IDs (unique only within one response); the default keeps the stable IDs.
    """
    chunks = list(chunks)
    if any(not c.id for c in chunks):
        strategy = chunks[0].metadata.get("strategy", "") if chunks else ""
        assign_stable_ids([c for c in chunks if not c.id], doc_id, config_key(strategy=strategy))

    parents = [c for c in chunks if c.metadata.get("level") == "parent"]
    leaves = [c for c in chunks if c.metadata.get("level") != "parent"]

    rename: dict[str, str] = {}
    if id_style == "sequential":
        rename.update({p.id: f"parent-{i + 1:04d}" for i, p in enumerate(parents)})
        rename.update({c.id: f"chunk-{i + 1:04d}" for i, c in enumerate(leaves)})
    rid = lambda x: rename.get(x, x) if x else None  # noqa: E731

    out_chunks = []
    for i, c in enumerate(leaves):
        md = c.metadata
        prev_id = md.get("prev_id", leaves[i - 1].id if i > 0 else None)
        next_id = md.get("next_id", leaves[i + 1].id if i + 1 < len(leaves) else None)
        out_chunks.append(
            {
                "chunk_id": rid(c.id),
                "text": c.text_for_embedding,
                "raw_text": c.text,
                "metadata": {
                    "hierarchical_path": list(md.get("heading_path", [])),
                    "parent_id": rid(md.get("parent_id")),
                    "prev_id": rid(prev_id),
                    "next_id": rid(next_id),
                    "doc_id": doc_id,
                    "level": md.get("level", "chunk"),
                    "start": c.start,
                    "end": c.end,
                    "summary": md.get("summary"),
                    "strategy": md.get("strategy"),
                },
            }
        )
    out_parents = [
        {
            "parent_id": rid(p.id),
            "text": p.text,
            "start": p.start,
            "end": p.end,
            "summary": p.metadata.get("summary"),
            "hierarchical_path": list(p.metadata.get("heading_path", [])),
            "child_ids": [rid(x) for x in p.metadata.get("child_ids", [])],
        }
        for p in parents
    ]
    return {"chunks": out_chunks, "parents": out_parents}

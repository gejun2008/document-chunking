"""Stable chunk IDs.

An ID depends on the document ID, a chunking-config key, the chunk's level, its
exact text and how many identical chunks precede it at that level — not on its
character offset. Re-chunking the same document gives the same IDs, and an edit
in one place leaves the IDs of unaffected chunks elsewhere unchanged. Changing
the chunking parameters changes the config key, so old and new IDs never mix.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any, Iterable

from .types import Chunk


def config_key(**params: Any) -> str:
    """Short, deterministic key for a chunking configuration."""
    blob = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:8]


def assign_stable_ids(chunks: Iterable[Chunk], doc_id: str, config: str, prefix: str = "c") -> None:
    """Set ``chunk.id`` in place. Chunks are taken in document order."""
    seen: Counter[tuple[str, str]] = Counter()
    for c in chunks:
        level = str(c.metadata.get("level", "chunk"))
        key = (level, c.text)
        occurrence = seen[key]
        seen[key] += 1
        digest = hashlib.sha1(f"{doc_id}\x1f{config}\x1f{level}\x1f{occurrence}\x1f{c.text}".encode()).hexdigest()
        c.id = f"{prefix}_{digest[:16]}"

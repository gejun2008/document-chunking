"""Strategy 4 — parent/child chunks with a per-parent summary (the legacy design).

1. Split the document into parent chunks (``child_size * parent_multiplier``
   tokens, ``parent_overlap``); by default with the recursive splitter, or with
   any chunker passed as ``parent_chunker`` (e.g. ``MarkdownChunker``).
2. Optionally ask an LLM for a 1–2 sentence summary of each parent — one call
   per parent, shared by all of its children.
3. Split each parent into child chunks (``child_size``, ``child_overlap``).
4. A child's ``embed_text`` is ``"Context summary: {summary}\\n\\n{child_text}"``;
   its ``text`` stays the verbatim source span.

Output: parents and children in document order, told apart by
``metadata["level"]``. Every chunk gets a stable ``id``; children carry
``parent_id``/``prev_id``/``next_id``, parents carry ``child_ids``, both carry
``doc_id`` and exact ``start``/``end`` offsets, so a service can always map a
child back to its parent's text. Prompts and defaults match the legacy
LlamaIndex implementation (TokenTextSplitter 1024/0 parents, 256/30 children,
gpt-4o-mini summaries).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Literal, MutableMapping

from ..clients import LLM
from ..ids import assign_stable_ids, config_key
from ..text import DEFAULT_ENCODING
from ..types import Chunk, Chunker, finalize
from .packing import recursive_split

SUMMARY_SYSTEM = """You are a text summarizer for RAG retrieval.
Rules:
1.Only use information from the original text,do not add any hypothetical content.
2.Output only the summary directly,no extra comments,explanations or prefixes.
3.Keep the summary concise within 1-2 sentences"""

SUMMARY_USER = """Summarize the following parent chunk in 1-2 sentences for retrieval context.
Only summarize the source text between the markers.

<SOURCE_TEXT>
{parent_text}
</SOURCE_TEXT>"""

CHILD_FORMAT = "Context summary: {summary}\n\n{child_text}"


class ParentChildChunker:
    def __init__(
        self,
        child_size: int = 256,
        parent_multiplier: int = 4,
        child_overlap: int = 30,
        parent_overlap: int = 0,
        llm: LLM | None = None,
        parent_chunker: Chunker | None = None,
        max_summary_tokens: int = 120,
        max_workers: int = 8,
        cache: MutableMapping[str, str] | None = None,
        on_error: Literal["raise", "skip"] = "raise",
        system_prompt: str = SUMMARY_SYSTEM,
        user_prompt: str = SUMMARY_USER,
        child_format: str = CHILD_FORMAT,
        encoding: str = DEFAULT_ENCODING,
    ):
        if child_overlap >= child_size:
            raise ValueError("child_overlap must be smaller than child_size")
        self.child_size = child_size
        self.parent_size = child_size * parent_multiplier
        self.child_overlap = child_overlap
        self.parent_overlap = parent_overlap
        self.llm = llm
        self.parent_chunker = parent_chunker
        self.max_summary_tokens = max_summary_tokens
        self.max_workers = max_workers
        self.cache = cache
        self.on_error = on_error
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.child_format = child_format
        self.encoding = encoding
        self.strategy = "parent_summary" if llm is not None else "parent_child"
        # IDs depend on how text is cut, not on prompts or the LLM
        self.config = config_key(
            strategy="parent_child", child=child_size, parent=self.parent_size, child_overlap=child_overlap,
            parent_overlap=parent_overlap, parents=type(parent_chunker).__name__ if parent_chunker else "recursive",
        )

    # ------------------------------------------------------------------ parts
    def _parents(self, text: str) -> list[Chunk]:
        if self.parent_chunker is not None:
            parents = self.parent_chunker.chunk(text)
        else:
            parents = recursive_split(text, self.parent_size, self.parent_overlap, self.encoding)
        for p in parents:
            p.embed_text = None
        return parents

    def _summary(self, parent: Chunk) -> str:
        prompt = self.user_prompt.format(parent_text=parent.text)
        key = f"{self.system_prompt}\x1f{prompt}"
        if self.cache is not None and key in self.cache:
            return self.cache[key]
        assert self.llm is not None
        try:
            out = self.llm.complete(prompt, max_tokens=self.max_summary_tokens, system=self.system_prompt).strip()
        except Exception:
            if self.on_error == "raise":
                raise
            return ""
        if self.cache is not None:
            self.cache[key] = out
        return out

    def _summaries(self, parents: list[Chunk]) -> list[str]:
        if self.llm is None:
            return [""] * len(parents)
        if self.max_workers > 1:
            with ThreadPoolExecutor(self.max_workers) as pool:
                return list(pool.map(self._summary, parents))
        return [self._summary(p) for p in parents]

    # ------------------------------------------------------------------ main
    def chunk(self, text: str, doc_id: str = "doc") -> list[Chunk]:
        parents = self._parents(text)
        summaries = self._summaries(parents)
        assign_stable_ids(parents, doc_id, self.config, prefix="p")

        out: list[Chunk] = []
        children_all: list[Chunk] = []
        kids_of: list[list[Chunk]] = []
        for parent, summary in zip(parents, summaries):
            heading_path = parent.metadata.get("heading_path", [])
            children = recursive_split(text, self.child_size, self.child_overlap, self.encoding, parent.start, parent.end)
            for c in children:
                c.metadata.update(level="child", doc_id=doc_id, heading_path=list(heading_path))
                if summary:
                    c.metadata["summary"] = summary
                    c.embed_text = self.child_format.format(summary=summary, child_text=c.text)
            parent.metadata = {"level": "parent", "doc_id": doc_id, "heading_path": list(heading_path),
                               "summary": summary or None, "strategy": self.strategy}
            out.append(parent)
            out.extend(children)
            children_all.extend(children)
            kids_of.append(children)

        assign_stable_ids(children_all, doc_id, self.config, prefix="c")
        for i, c in enumerate(children_all):
            c.metadata["prev_id"] = children_all[i - 1].id if i > 0 else None
            c.metadata["next_id"] = children_all[i + 1].id if i + 1 < len(children_all) else None
        for parent, kids in zip(parents, kids_of):
            parent.metadata["child_ids"] = [k.id for k in kids]
            for k in kids:
                k.metadata["parent_id"] = parent.id
        return finalize(out, self.strategy)

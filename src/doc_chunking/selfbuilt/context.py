"""Strategy 4 — context-enriched fragments (a.k.a. contextual retrieval).

Wraps any base chunker. For every chunk an LLM writes a short sentence that
situates the chunk in its document; that sentence is prepended to
``embed_text`` while ``text``/offsets stay untouched, so citations still point
at the original span.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import MutableMapping

from ..clients import LLM
from ..text import DEFAULT_ENCODING, count_tokens
from ..types import Chunk, Chunker, finalize

PROMPT = """<document>
{document}
</document>
Here is the chunk we want to situate within the whole document
<chunk>
{chunk}
</chunk>
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Write it in the same language as the document. Answer only with the succinct context and nothing else."""


class ContextEnricher:
    def __init__(
        self,
        base: Chunker,
        llm: LLM,
        window_tokens: int = 2000,
        max_context_tokens: int = 100,
        max_workers: int = 4,
        cache: MutableMapping[str, str] | None = None,
        prompt: str = PROMPT,
        encoding: str = DEFAULT_ENCODING,
    ):
        self.base = base
        self.llm = llm
        self.window_tokens = window_tokens
        self.max_context_tokens = max_context_tokens
        self.max_workers = max_workers
        self.cache = cache
        self.prompt = prompt
        self.encoding = encoding

    def _window(self, text: str, chunk: Chunk, chars_per_token: float) -> str:
        """The whole document if it fits, otherwise a window centred on the chunk."""
        budget = int(self.window_tokens * chars_per_token)
        if len(text) <= budget:
            return text
        pad = max(0, (budget - (chunk.end - chunk.start)) // 2)
        lo = max(0, chunk.start - pad)
        hi = min(len(text), chunk.end + pad)
        return text[lo:hi]

    def _context_for(self, text: str, chunk: Chunk, cpt: float) -> str:
        prompt = self.prompt.format(document=self._window(text, chunk, cpt), chunk=chunk.text)
        if self.cache is not None and prompt in self.cache:
            return self.cache[prompt]
        out = self.llm.complete(prompt, max_tokens=self.max_context_tokens).strip()
        if self.cache is not None:
            self.cache[prompt] = out
        return out

    def chunk(self, text: str) -> list[Chunk]:
        chunks = self.base.chunk(text)
        if not chunks:
            return chunks
        cpt = len(text) / max(1, count_tokens(text, self.encoding))
        if self.max_workers > 1:
            with ThreadPoolExecutor(self.max_workers) as pool:
                contexts = list(pool.map(lambda c: self._context_for(text, c, cpt), chunks))
        else:
            contexts = [self._context_for(text, c, cpt) for c in chunks]
        for c, ctx in zip(chunks, contexts):
            c.metadata["context"] = ctx
            c.metadata["base_strategy"] = c.metadata.get("strategy")
            c.metadata["strategy"] = "context_enriched"
            c.embed_text = f"{ctx}\n\n{c.text_for_embedding}"
        return finalize(chunks, "context_enriched")

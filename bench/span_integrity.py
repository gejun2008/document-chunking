"""How often is a gold evidence span kept inside a single chunk? (no embeddings needed)"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "bench")]
from datasets import load_chroma, load_ohr, load_qasper

from doc_chunking.chonkie_impl import ChonkieFixedSizeChunker, ChonkieMarkdownChunker
from doc_chunking.langchain_impl import LCFixedSizeChunker, LCMarkdownChunker
from doc_chunking.selfbuilt import FixedSizeChunker, MarkdownChunker

IMPLS = {
    ("selfbuilt", "fixed_size"): lambda: FixedSizeChunker(400, 50),
    ("chonkie", "fixed_size"): lambda: ChonkieFixedSizeChunker(400, 50),
    ("langchain", "fixed_size"): lambda: LCFixedSizeChunker(400, 50),
    ("selfbuilt", "layout_aware"): lambda: MarkdownChunker(400),
    ("chonkie", "layout_aware"): lambda: ChonkieMarkdownChunker(400),
    ("langchain", "layout_aware"): lambda: LCMarkdownChunker(400),
}

for ds in (load_chroma(), load_qasper(), load_ohr()):
    docs = {d.id: d.text for d in ds.docs}
    for (opt, strat), make in IMPLS.items():
        ch = make()
        chunks = {d: [(c.start, c.end) for c in ch.chunk(t)] for d, t in docs.items()}
        whole = total = 0
        for q in ds.queries:
            for s, e in q.spans:
                total += 1
                whole += any(cs <= s and e <= ce for cs, ce in chunks[q.doc_id])
        print(f"{ds.name:7s} {strat:13s} {opt:12s} spans kept whole: {whole / total:6.1%}")

"""Pick a chonkie SemanticChunker threshold whose average chunk size matches the
self-built semantic chunker, so the two are compared at equal granularity."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "bench")]
import numpy as np
from datasets import load_chroma, load_ohr, load_qasper
from local_models import DiskCachedEmbedder, SentenceTransformerEmbedder

from doc_chunking.chonkie_impl import ChonkieSemanticChunker
from doc_chunking.selfbuilt import SemanticChunker
from doc_chunking.text import count_tokens

emb = DiskCachedEmbedder(SentenceTransformerEmbedder("BAAI/bge-m3"), "BAAI/bge-m3")
docs = [d.text for d in load_chroma().docs if d.id in ("state_of_the_union", "chatlogs")]
docs += [d.text for d in load_qasper().docs[:6]] + [d.text for d in load_ohr().docs[:10]]

def avg(chunker):
    sizes = [count_tokens(c.text) for t in docs for c in chunker.chunk(t)]
    return np.mean(sizes), len(sizes)

print("selfbuilt p90", avg(SemanticChunker(emb, 400, breakpoint_percentile=90)))
for th in (0.1, 0.15, 0.2, 0.3, 0.5, 0.8):
    print("chonkie", th, avg(ChonkieSemanticChunker(emb, 400, threshold=th)))

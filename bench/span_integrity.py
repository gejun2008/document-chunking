"""How often is a gold evidence span kept whole inside one chunk? (no embeddings needed)

For field extraction this is the first thing that matters: a value that is cut in
two cannot be extracted from either half. Parent/child strategies are checked at
both levels. OHR-Bench is also broken down by evidence type (text / table).
Writes results/span_integrity.json and prints a table.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "bench")]
from datasets import load_chroma, load_ohr, load_qasper  # noqa: E402
from legacy.precomputed import LegacyFlat, LegacyParentChild  # noqa: E402

from doc_chunking.chonkie_impl import ChonkieFixedSizeChunker, ChonkieMarkdownChunker  # noqa: E402
from doc_chunking.langchain_impl import LCFixedSizeChunker, LCMarkdownChunker  # noqa: E402
from doc_chunking.selfbuilt import FixedSizeChunker, MarkdownChunker, ParentChildChunker  # noqa: E402

IMPLS = {
    ("selfbuilt", "fixed_size"): lambda ds: FixedSizeChunker(400, 50),
    ("chonkie", "fixed_size"): lambda ds: ChonkieFixedSizeChunker(400, 50),
    ("langchain", "fixed_size"): lambda ds: LCFixedSizeChunker(400, 50),
    ("legacy", "fixed_size"): lambda ds: LegacyFlat(ds, "fixed_size"),
    ("selfbuilt", "layout_aware"): lambda ds: MarkdownChunker(400),
    ("chonkie", "layout_aware"): lambda ds: ChonkieMarkdownChunker(400),
    ("langchain", "layout_aware"): lambda ds: LCMarkdownChunker(400),
    ("legacy", "layout_aware"): lambda ds: LegacyFlat(ds, "layout_aware"),
    ("selfbuilt", "parent_child"): lambda ds: ParentChildChunker(),
    ("legacy", "parent_child"): lambda ds: LegacyParentChild(ds),
}


def spans(chunker, text, doc_id):
    try:
        chunks = chunker.chunk(text, doc_id=doc_id)
    except TypeError:
        chunks = chunker.chunk(text)
    by_level = defaultdict(list)
    for c in chunks:
        by_level[c.metadata.get("level", "chunk")].append((c.start, c.end))
    return by_level


out = []
for ds in (load_chroma(), load_qasper(), load_ohr()):
    for (opt, strat), make in IMPLS.items():
        ch = make(ds.name)
        per_doc = {d.id: spans(ch, d.text, d.id) for d in ds.docs}
        for level in sorted({lv for v in per_doc.values() for lv in v}):
            whole, total = defaultdict(int), defaultdict(int)
            for q in ds.queries:
                kind = q.meta.get("source", "all") if ds.name == "ohr" else "all"
                for s, e in q.spans:
                    ok = any(cs <= s and e <= ce for cs, ce in per_doc[q.doc_id].get(level, []))
                    for k in {"all", kind}:
                        total[k] += 1
                        whole[k] += ok
            row = {"dataset": ds.name, "option": opt, "strategy": strat, "level": level,
                   **{f"whole_{k}": whole[k] / total[k] for k in total}, **{f"n_{k}": total[k] for k in total}}
            out.append(row)
            extra = f"  table {row['whole_table']:6.1%} (n={row['n_table']})" if "whole_table" in row else ""
            print(f"{ds.name:7s} {strat:13s} {level:6s} {opt:10s} kept whole: {row['whole_all']:6.1%}{extra}", flush=True)
(ROOT / "results" / "span_integrity.json").write_text(json.dumps(out, indent=1))

"""Export chunks produced by the legacy LlamaIndex configuration, for comparison.

Runs in a throwaway environment so llama-index (and its nltk dependency) never
enters this project's dependencies or lock file:

    uv run --no-project -p 3.12 --with llama-index-core==0.14.25 --with pandas --with pyarrow \\
        python bench/legacy/export_llamaindex.py

Settings are the legacy service defaults (src/core/settings.py of the old repo):
  fixed_size    TokenTextSplitter(512, 64, separator=" ", backup_separators=["\\n"], keep_whitespaces=False)
  layout_aware  MarkdownNodeParser(header_path_separator="/")
  parent_child  parents TokenTextSplitter(256 * 4 = 1024, 0); each parent re-wrapped in a
                Document and split by TokenTextSplitter(256, 30)
Writes results/legacy_chunks/<dataset>.jsonl, one line per chunk with its exact
character offsets (the text is source[start:end]; the script stops if a node's
text cannot be found verbatim in the source).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench"))
from datasets import load_chroma, load_ohr, load_qasper  # noqa: E402
from llama_index.core import Document  # noqa: E402
from llama_index.core.node_parser import MarkdownNodeParser, TokenTextSplitter  # noqa: E402


def nodes(parser, text):
    return parser.get_nodes_from_documents([Document(text=text)])


def span(text, node, base=0, cursor=0):
    """Offsets of the node in the original text (LlamaIndex's, verified; else searched)."""
    s, e = node.start_char_idx, node.end_char_idx
    if s is not None and text[base + s : base + e] == node.text:
        return base + s, base + e, True
    pos = text.find(node.text, cursor)
    return (pos, pos + len(node.text), True) if pos != -1 else (None, None, False)


fixed = TokenTextSplitter(chunk_size=512, chunk_overlap=64, separator=" ", backup_separators=["\n"], keep_whitespaces=False)
markdown = MarkdownNodeParser(header_path_separator="/")
parent_sp = TokenTextSplitter(chunk_size=1024, chunk_overlap=0)
child_sp = TokenTextSplitter(chunk_size=256, chunk_overlap=30)

out_dir = ROOT / "results" / "legacy_chunks"
out_dir.mkdir(parents=True, exist_ok=True)
for ds in (load_chroma(), load_qasper(), load_ohr()):
    rows = []
    for d in ds.docs:
        for strategy, parser in (("fixed_size", fixed), ("layout_aware", markdown)):
            for n in nodes(parser, d.text):
                s, e, ok = span(d.text, n)
                assert ok, f"{d.id}: {strategy} node not found verbatim"
                rows.append({"doc_id": d.id, "strategy": strategy, "start": s, "end": e,
                             "header_path": n.metadata.get("header_path")})
        for pi, p in enumerate(nodes(parent_sp, d.text)):
            ps, pe, ok = span(d.text, p)
            assert ok, f"{d.id}: parent node not found verbatim"
            rows.append({"doc_id": d.id, "strategy": "parent_child", "level": "parent", "parent": pi, "start": ps, "end": pe})
            for c in nodes(child_sp, p.text):
                cs, ce, cok = span(d.text, c, base=ps or 0)
                assert cok, f"{d.id}: child node not found verbatim"
                rows.append({"doc_id": d.id, "strategy": "parent_child", "level": "child", "parent": pi, "start": cs, "end": ce})
    path = out_dir / f"{ds.name}.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"{ds.name}: {len(rows)} chunks -> {path.relative_to(ROOT)}")

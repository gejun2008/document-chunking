"""Benchmark the three implementation options on public datasets.

    uv run python bench/run_bench.py                       # local bge-m3 + local Qwen3-4B
    uv run python bench/run_bench.py --embedding-url https://.../v1 --embedding-model X \
        --llm-url https://.../v1 --llm-model Y             # company services

Results are appended to results/results.jsonl (re-runs skip finished cells).
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "bench"))

from datasets import Dataset, load_chroma, load_ohr, load_qasper  # noqa: E402
from local_models import CACHE_DIR, DiskCachedEmbedder, SqliteDict  # noqa: E402
from metrics import evaluate  # noqa: E402

from doc_chunking.chonkie_impl import ChonkieFixedSizeChunker, ChonkieMarkdownChunker, ChonkieSemanticChunker  # noqa: E402
from doc_chunking.langchain_impl import (  # noqa: E402
    ContextualEnrichmentTransformer,
    LCFixedSizeChunker,
    LCMarkdownChunker,
    SemanticTextSplitter,
)
from doc_chunking.selfbuilt import (  # noqa: E402
    ContextEnricher,
    FixedSizeChunker,
    MarkdownChunker,
    ParentChildChunker,
    SemanticChunker,
)
from doc_chunking.text import count_tokens  # noqa: E402

STRATEGIES = ["fixed_size", "layout_aware", "semantic", "context_enriched"]
OPTIONS = ["selfbuilt", "chonkie", "langchain"]
# Strategies that need an LLM run on the lite subsets, next to their no-LLM base strategy.
LLM_STRATEGIES = {"context_enriched": "fixed_size", "parent_summary": "parent_child"}
# Extra options for the comparison with the legacy LlamaIndex service (bench/legacy/):
#   legacy        chunks exported from the real legacy configuration; semantic = same algorithm, legacy p95
#   selfbuilt_512 self-built fixed_size at the legacy 512/64 setting

# Documents used for the (LLM-expensive) context-enrichment comparison.
LITE = {
    "chroma": {"state_of_the_union", "chatlogs", "wikitexts"},
    "qasper": 15,  # first N papers
    "ohr": 2,  # docs per domain
}


def lite_subset(ds: Dataset) -> Dataset:
    spec = LITE[ds.name]
    if isinstance(spec, set):
        ids = spec
    elif ds.name == "qasper":
        ids = {d.id for d in ds.docs[:spec]}
    else:
        per: dict[str, list[str]] = {}
        with_q = {q.doc_id for q in ds.queries}
        for d in ds.docs:
            if d.id in with_q:
                per.setdefault(d.id.split("/")[0], []).append(d.id)
        ids = {i for v in per.values() for i in v[:spec]}
    return ds.subset(ids, f"{ds.name}-lite")


def build(option: str, strategy: str, a, emb, llm, cache, dataset: str = ""):
    cs, ov = a.chunk_size, a.overlap
    ctx = dict(llm=llm, cache=cache, max_workers=a.llm_workers, window_tokens=a.window_tokens)
    # parent/child defaults follow the legacy service: children 256/30, parents 4x (1024), no parent overlap
    if strategy == "parent_child" and option == "selfbuilt":
        return ParentChildChunker()
    if strategy == "parent_summary" and option == "selfbuilt":
        return ParentChildChunker(llm=llm, cache=cache, max_workers=a.llm_workers)
    if option == "legacy":
        from legacy.precomputed import LegacyFlat, LegacyParentChild

        if strategy in ("fixed_size", "layout_aware"):
            return LegacyFlat(dataset, strategy)
        if strategy == "semantic":
            return SemanticChunker(emb, cs, breakpoint_percentile=95.0)
        if strategy == "parent_child":
            return LegacyParentChild(dataset)
        if strategy == "parent_summary":
            return LegacyParentChild(dataset, llm=llm, cache=cache, max_workers=a.llm_workers)
    if option == "selfbuilt_512" and strategy == "fixed_size":
        return FixedSizeChunker(512, 64)
    if option == "selfbuilt":
        return {
            "fixed_size": lambda: FixedSizeChunker(cs, ov),
            "layout_aware": lambda: MarkdownChunker(cs),
            "semantic": lambda: SemanticChunker(emb, cs, breakpoint_percentile=a.percentile),
            "context_enriched": lambda: ContextEnricher(FixedSizeChunker(cs, ov), **ctx),
        }[strategy]()
    if option == "chonkie":
        return {
            "fixed_size": lambda: ChonkieFixedSizeChunker(cs, ov),
            "layout_aware": lambda: ChonkieMarkdownChunker(cs),
            "semantic": lambda: ChonkieSemanticChunker(emb, cs, threshold=a.chonkie_threshold),
            "context_enriched": lambda: ContextEnricher(ChonkieFixedSizeChunker(cs, ov), **ctx),
        }[strategy]()
    if option == "langchain":
        return {
            "fixed_size": lambda: LCFixedSizeChunker(cs, ov),
            "layout_aware": lambda: LCMarkdownChunker(cs),
            "semantic": lambda: SemanticTextSplitter(emb, cs, breakpoint_percentile=a.percentile),
            "context_enriched": lambda: ContextualEnrichmentTransformer(LCFixedSizeChunker(cs, ov), llm, **{k: v for k, v in ctx.items() if k != "llm"}),
        }[strategy]()
    raise ValueError(option)


def run_cell(ds: Dataset, option: str, strategy: str, a, emb, llm, cache) -> dict:
    chunker = build(option, strategy, a, emb, llm, cache, ds.name)
    takes_doc_id = "doc_id" in inspect.signature(chunker.chunk).parameters
    index, texts, sizes, parent_items = [], [], [], []
    t0 = time.time()
    for d in ds.docs:
        chunks = chunker.chunk(d.text, doc_id=d.id) if takes_doc_id else chunker.chunk(d.text)
        parents = {c.id: c for c in chunks if c.metadata.get("level") == "parent"}
        for c in chunks:
            assert d.text[c.start : c.end] == c.text, f"bad offsets {option}/{strategy}/{d.id}"
            if c.metadata.get("level") == "parent":
                continue  # only children are indexed; parents are what a child expands to
            n = count_tokens(c.text)
            index.append((d.id, c.start, c.end, n))
            texts.append(c.text_for_embedding)
            sizes.append(n)
            if parents:
                p = parents[c.metadata["parent_id"]]
                parent_items.append((d.id, p.start, p.end, count_tokens(p.text)))
    chunk_s = time.time() - t0
    t1 = time.time()
    cv = emb.embed(texts)
    qv = emb.embed([q.question for q in ds.queries])
    embed_s = time.time() - t1
    per_doc = ds.name.startswith("qasper")
    m = evaluate(ds.queries, index, cv, qv, per_doc=per_doc)
    if parent_items:  # retrieve children, hand their (deduplicated) parents to the LLM
        m.update({f"parent_{k}": v for k, v in evaluate(ds.queries, index, cv, qv, per_doc=per_doc, expand=parent_items).items()})
    sizes_a = np.array(sizes)
    return {
        "dataset": ds.name,
        "option": option,
        "strategy": strategy,
        "n_docs": len(ds.docs),
        "n_queries": len(ds.queries),
        "n_chunks": len(index),
        "avg_tokens": float(sizes_a.mean()),
        "p95_tokens": float(np.percentile(sizes_a, 95)),
        "max_tokens": int(sizes_a.max()),
        "chunk_seconds": round(chunk_s, 2),
        "embed_seconds": round(embed_s, 2),
        **m,
    }


def add_model_args(p: argparse.ArgumentParser) -> None:
    """Embedding / LLM selection shared by run_bench.py and bootstrap.py."""
    p.add_argument("--embedding-url", help="OpenAI-compatible base URL (key: EMBEDDING_API_KEY); default local bge-m3")
    p.add_argument("--embedding-model", default="BAAI/bge-m3")
    p.add_argument("--embedding-batch-size", type=int, default=64)
    p.add_argument("--llm-url", help="OpenAI-compatible base URL (key: LLM_API_KEY); default local Qwen3-4B via mlx")
    p.add_argument("--llm-model", default="mlx-community/Qwen3-4B-Instruct-2507-4bit")
    p.add_argument("--llm-workers", type=int, default=8)


def make_embedder(a: argparse.Namespace) -> DiskCachedEmbedder:
    if a.embedding_url:
        from doc_chunking.clients import OpenAICompatibleEmbedder

        inner = OpenAICompatibleEmbedder(a.embedding_url, model=a.embedding_model, batch_size=a.embedding_batch_size)
    else:
        from local_models import SentenceTransformerEmbedder

        inner = SentenceTransformerEmbedder(a.embedding_model)
    return DiskCachedEmbedder(inner, a.embedding_model)


def make_llm(a: argparse.Namespace):
    if a.llm_url:
        from doc_chunking.clients import OpenAICompatibleChat

        return OpenAICompatibleChat(a.llm_url, model=a.llm_model)
    import platform

    if not (sys.platform == "darwin" and platform.machine() == "arm64"):
        sys.exit("The local LLM needs Apple Silicon (mlx). Pass --llm-url/--llm-model for an OpenAI-compatible endpoint.")
    from local_models import BatchedMLXChat

    return BatchedMLXChat(a.llm_model, batch_size=a.llm_workers)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["chroma", "qasper", "ohr"])
    p.add_argument("--options", nargs="+", default=OPTIONS)
    p.add_argument("--strategies", nargs="+", default=STRATEGIES)
    p.add_argument("--chunk-size", type=int, default=400)
    p.add_argument("--overlap", type=int, default=50)
    p.add_argument("--percentile", type=float, default=90.0)
    p.add_argument("--chonkie-threshold", type=float, default=0.15)  # see calibrate_chonkie.py
    p.add_argument("--window-tokens", type=int, default=1500)
    add_model_args(p)
    p.add_argument("--out", default=str(ROOT / "results" / "results.jsonl"))
    a = p.parse_args()

    emb = make_embedder(a)
    llm = make_llm(a) if set(a.strategies) & set(LLM_STRATEGIES) else None
    cache = SqliteDict(CACHE_DIR / f"ctx_{a.llm_model.replace('/', '_')}.sqlite")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done.add((r["dataset"], r["option"], r["strategy"]))

    loaders = {"chroma": load_chroma, "qasper": load_qasper, "ohr": load_ohr}
    for name in a.datasets:
        full = loaders[name]()
        lite = lite_subset(full)
        for strategy in a.strategies:
            for option in a.options:
                base = LLM_STRATEGIES.get(strategy)
                cells = [(lite, base), (lite, strategy)] if base else [(full, strategy)]
                for ds, strat in cells:
                    if (ds.name, option, strat) in done:
                        continue
                    t = time.time()
                    r = run_cell(ds, option, strat, a, emb, llm, cache)
                    r["config"] = {k: getattr(a, k) for k in ("chunk_size", "overlap", "percentile", "chonkie_threshold", "window_tokens", "embedding_model", "llm_model")}
                    with out.open("a") as f:
                        f.write(json.dumps(r) + "\n")
                    done.add((ds.name, option, strat))
                    print(
                        f"[{time.time() - t:6.1f}s] {ds.name:11s} {option:13s} {strat:16s} chunks={r['n_chunks']:5d} "
                        f"avg_tok={r['avg_tokens']:5.0f} R@5={r['recall@5']:.3f} R@2k={r['recall@2000tok']:.3f} IoU@5={r['iou@5']:.3f}",
                        flush=True,
                    )


if __name__ == "__main__":
    main()

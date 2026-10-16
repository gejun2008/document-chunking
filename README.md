# doc-chunking

Document chunking for Document Studio with four strategies, provided by three interchangeable
implementations so they can be compared on public benchmarks. None of them depend on nltk
(CVE-2026-81726), which is what blocks `llama-index-core`.

| strategy | option 1 `selfbuilt` | option 2 `chonkie_impl` | option 3 `langchain_impl` |
|---|---|---|---|
| fixed_size | `FixedSizeChunker` | `ChonkieFixedSizeChunker` (SentenceChunker) | `LCFixedSizeChunker` (RecursiveCharacterTextSplitter) |
| layout_aware | `MarkdownChunker` (markdown-it AST) | `ChonkieMarkdownChunker` (MarkdownChef + RecursiveChunker) | `LCMarkdownChunker` (MarkdownHeaderTextSplitter) |
| semantic | `SemanticChunker` | `ChonkieSemanticChunker` | `SemanticTextSplitter` (LangChain `TextSplitter` over the self-built algorithm) |
| context_enriched | `ContextEnricher` | self-built `ContextEnricher` over a chonkie chunker | `ContextualEnrichmentTransformer` (LangChain `BaseDocumentTransformer`) |

Parent/child chunks (the legacy design) are in `selfbuilt.ParentChildChunker`: parents of
`child_size * 4` tokens, optional 1–2 sentence parent summaries from an LLM (legacy prompt), children
of `child_size` tokens inside each parent, stable IDs and `parent_id`/`prev_id`/`next_id` links.
`doc_chunking.legacy_api.to_legacy_response(chunks, doc_id)` returns the legacy service response shape
(`chunk_id`, `text` = embedded text, `raw_text` = original, `metadata.hierarchical_path/parent_id/prev_id/next_id`)
plus offsets, `doc_id` and the parents' text.

Every chunker exposes `chunk(text) -> list[Chunk]`. A `Chunk` has the original `text`, exact
`start`/`end` character offsets, `metadata` (e.g. `heading_path`, `context`) and `embed_text`,
the string to send to the embedding model.

## Install

```bash
uv sync                      # option 1 only: tiktoken, markdown-it-py, numpy, httpx
uv sync --extra chonkie      # + option 2
uv sync --extra langchain    # + option 3
```

## Usage

```python
from doc_chunking import OpenAICompatibleEmbedder, OpenAICompatibleChat, CachedEmbedder
from doc_chunking.selfbuilt import FixedSizeChunker, MarkdownChunker, SemanticChunker, ContextEnricher

emb = CachedEmbedder(OpenAICompatibleEmbedder())   # EMBEDDING_BASE_URL / EMBEDDING_API_KEY / EMBEDDING_MODEL
llm = OpenAICompatibleChat()                       # LLM_BASE_URL / LLM_API_KEY / LLM_MODEL

chunkers = {
    "fixed_size": FixedSizeChunker(chunk_size=400, overlap=50),
    "layout_aware": MarkdownChunker(chunk_size=400),
    "semantic": SemanticChunker(emb, chunk_size=400),
    "context_enriched": ContextEnricher(MarkdownChunker(chunk_size=400), llm),
}
chunks = chunkers["layout_aware"].chunk(markdown_text)
```

Production notes:

* tiktoken downloads its BPE file on first use; in air-gapped environments pre-populate
  `TIKTOKEN_CACHE_DIR`.
* `semantic` needs an embedding endpoint and `context_enriched` needs an LLM endpoint;
  `fixed_size` and `layout_aware` need neither.
* Do not use LangChain's `NLTKTextSplitter`, it pulls nltk back in.

## Tests and benchmark

The three public datasets are committed under `data/` (sources and licenses: [data/README.md](data/README.md)).
`bench/download_data.py` re-fetches the same pinned files and verifies their SHA-256.

```bash
uv sync --all-extras --all-groups
uv run pytest
uv run python bench/run_bench.py       # local bge-m3; context enrichment needs Apple Silicon or --llm-url
uv run python bench/make_report.py     # -> results/report.md
```

### Re-run with company models

Any OpenAI-compatible `/embeddings` and `/chat/completions` endpoint works. One command runs every stage
(all strategies, context enrichment, chunk-size sweep, significance test, report) into `results/company/`,
leaving the local-model results untouched:

```bash
export EMBEDDING_BASE_URL=https://<gateway>/v1 EMBEDDING_MODEL=<model> EMBEDDING_API_KEY=...
export LLM_BASE_URL=https://<gateway>/v1 LLM_MODEL=<model> LLM_API_KEY=...
bench/run_company.sh
uv run python bench/compare_runs.py results results/company   # R@2k side by side
```

Optional: `EMBEDDING_BATCH_SIZE` (default 64) if the gateway limits batch size, `LLM_WORKERS` (default 8)
for parallel LLM calls, `OUT_DIR` for another output folder. Finished cells are skipped, so an interrupted
run can simply be started again.

Comparison with the legacy LlamaIndex service: `bench/legacy/export_llamaindex.py` replays the legacy
configuration with the real LlamaIndex in a throwaway environment (offsets saved in `results/legacy_chunks/`),
then `run_bench.py --options legacy selfbuilt_512 selfbuilt --strategies fixed_size layout_aware semantic parent_child parent_summary --out results/results_legacy.jsonl`
scores it like every other option.

Why these datasets and metrics, quality checks, significance tests and limitations:
[docs/benchmark-methodology.md](docs/benchmark-methodology.md). Report: `results/chunking_report.html`.

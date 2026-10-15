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

```bash
uv sync --all-extras --all-groups
uv run pytest
uv run python bench/download_data.py   # public datasets at pinned revisions, SHA-256 verified
uv run python bench/run_bench.py       # local bge-m3; context enrichment needs Apple Silicon or --llm-url
uv run python bench/make_report.py     # -> results/report.md
```

On a machine without Apple Silicon, or to use company services instead of local models:

```bash
export EMBEDDING_API_KEY=... LLM_API_KEY=...
uv run python bench/run_bench.py --out results/company.jsonl \
  --embedding-url https://<gateway>/v1 --embedding-model <model> \
  --llm-url https://<gateway>/v1 --llm-model <model>
```

Write company runs to a separate `--out` file: finished cells are skipped by
(dataset, option, strategy), so reusing `results/results.jsonl` would skip everything.

Why these datasets and metrics, quality checks, significance tests and limitations:
[docs/benchmark-methodology.md](docs/benchmark-methodology.md). Published report: `results/chunking_report.html`.

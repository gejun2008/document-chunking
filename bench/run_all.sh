#!/bin/sh
# Full benchmark: main strategies -> context enrichment (local LLM) -> chunk-size sweep.
# Runs sequentially because the embedding model and the local LLM share the GPU.
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1
uv run -q python bench/run_bench.py --strategies fixed_size layout_aware semantic > results/run_main.log 2>&1
uv run -q python bench/run_bench.py --strategies context_enriched --out results/results_ctx.jsonl --llm-workers 4 > results/run_ctx2.log 2>&1
bench/run_sweep.sh > results/run_sweep.log 2>&1
uv run -q python bench/make_report.py > /dev/null
echo ALL_DONE >> results/run_sweep.log

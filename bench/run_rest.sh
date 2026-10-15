#!/bin/sh
# Remaining stages after the main run: context enrichment -> sweep -> selfbuilt_v2 -> report.
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1
uv run -q python bench/run_bench.py --strategies context_enriched --options selfbuilt --out results/results_ctx.jsonl --llm-workers 8 > results/run_ctx2.log 2>&1
bench/run_sweep.sh > results/run_sweep.log 2>&1
uv run -q python bench/run_bench.py --options selfbuilt_v2 --strategies fixed_size > results/run_v2.log 2>&1
uv run -q python bench/make_report.py > /dev/null
echo ALL_DONE >> results/run_v2.log

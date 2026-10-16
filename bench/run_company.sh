#!/bin/sh
# Re-run the whole benchmark with company models (OpenAI-compatible endpoints).
# Results go to results/company/ so they can be compared with the local-model run:
#
#   export EMBEDDING_BASE_URL=https://<gateway>/v1 EMBEDDING_MODEL=<model> EMBEDDING_API_KEY=...
#   export LLM_BASE_URL=https://<gateway>/v1 LLM_MODEL=<model> LLM_API_KEY=...
#   bench/run_company.sh
#   uv run python bench/compare_runs.py results results/company
#
# Optional: EMBEDDING_BATCH_SIZE (default 64), LLM_WORKERS (parallel LLM calls, default 8),
# OUT_DIR (default results/company). Finished cells are skipped, so it can be resumed.
set -eu
cd "$(dirname "$0")/.."
: "${EMBEDDING_BASE_URL:?set EMBEDDING_BASE_URL}" "${EMBEDDING_MODEL:?set EMBEDDING_MODEL}"
: "${LLM_BASE_URL:?set LLM_BASE_URL}" "${LLM_MODEL:?set LLM_MODEL}"
OUT="${OUT_DIR:-results/company}"
mkdir -p "$OUT"
set -- --embedding-url "$EMBEDDING_BASE_URL" --embedding-model "$EMBEDDING_MODEL" \
  --embedding-batch-size "${EMBEDDING_BATCH_SIZE:-64}"
LLM_ARGS="--llm-url $LLM_BASE_URL --llm-model $LLM_MODEL --llm-workers ${LLM_WORKERS:-8}"

[ -f data/ohr/OHR-Bench.parquet ] || uv run python bench/download_data.py
echo "== main strategies";        uv run -q python bench/run_bench.py --strategies fixed_size layout_aware semantic --out "$OUT/results.jsonl" "$@"
echo "== context enrichment";     uv run -q python bench/run_bench.py --strategies context_enriched --out "$OUT/results_ctx.jsonl" "$@" $LLM_ARGS
echo "== chunk-size sweep";       bench/run_sweep.sh "$OUT" "$@"
echo "== significance";           uv run -q python bench/bootstrap.py --results-dir "$OUT" "$@"
uv run -q python bench/make_report.py "$OUT" > /dev/null
uv run -q python bench/make_html.py "$OUT"
echo "done: $OUT/report.md, $OUT/chunking_report.html"

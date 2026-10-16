#!/bin/sh
# Chunk-size sweep for the non-LLM strategies (run after run_bench.py).
#   bench/run_sweep.sh [results-dir] [extra run_bench.py args, e.g. --embedding-url ...]
cd "$(dirname "$0")/.."
DIR="${1:-results}"
[ $# -gt 0 ] && shift
for cs in 200 300; do
  uv run -q python bench/run_bench.py --strategies fixed_size layout_aware \
    --chunk-size $cs --overlap $((cs / 8)) --out "$DIR/sweep_cs$cs.jsonl" "$@"
done

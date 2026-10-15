#!/bin/sh
# Chunk-size sweep for the non-LLM strategies (run after run_bench.py).
cd "$(dirname "$0")/.."
for cs in 200 300; do
  HF_HUB_OFFLINE=1 uv run -q python bench/run_bench.py --strategies fixed_size layout_aware \
    --chunk-size $cs --overlap $((cs / 8)) --out results/sweep_cs$cs.jsonl
done

"""Side-by-side R@2k tok of two result folders, e.g. local models vs company models.

    uv run python bench/compare_runs.py results results/company
"""
import json
import sys
from pathlib import Path


def load(d: Path) -> dict:
    rows = {}
    for p in sorted(d.glob("results*.jsonl")) + sorted(d.glob("sweep_*.jsonl")):
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                rows[(r["dataset"], r["strategy"], r["option"], r["config"]["chunk_size"])] = r
    return rows


a_dir, b_dir = (Path(x) for x in (sys.argv[1:3] if len(sys.argv) >= 3 else ("results", "results/company")))
A, B = load(a_dir), load(b_dir)
model = lambda rows: next(iter(rows.values()))["config"]["embedding_model"] if rows else "-"
print(f"A = {a_dir} (embedding {model(A)})\nB = {b_dir} (embedding {model(B)})\n")
print(f"| dataset | strategy | option | chunk_size | A R@2k | B R@2k | B − A |\n|---|---|---|---|---|---|---|")
for k in sorted(set(A) & set(B)):
    a, b = A[k]["recall@2000tok"], B[k]["recall@2000tok"]
    print(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {a * 100:.1f} | {b * 100:.1f} | {(b - a) * 100:+.1f} |")
only = sorted(set(A) ^ set(B))
if only:
    print(f"\n{len(only)} cells exist in only one run (not compared).")

"""Turn results/*.jsonl into results/report.md."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

OPTS = ["selfbuilt", "chonkie", "langchain"]
STRATS = ["fixed_size", "layout_aware", "semantic"]
COLS = [
    ("recall@5", "R@5"),
    ("precision@5", "P@5"),
    ("iou@5", "IoU@5"),
    ("recall@1000tok", "R@1k tok"),
    ("recall@2000tok", "R@2k tok"),
]


def load() -> list[dict]:
    rows = []
    for p in sorted(RES.glob("results*.jsonl")):
        rows += [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    return rows


def fmt(x: float) -> str:
    return f"{x * 100:.1f}"


def main() -> None:
    rows = load()
    by = {(r["dataset"], r["option"], r["strategy"]): r for r in rows}
    datasets = [d for d in ["chroma", "qasper", "ohr"] if any(k[0] == d for k in by)]
    out = ["# Chunking benchmark results", ""]
    out.append("Metrics in %, higher is better. R@2k tok = recall when retrieving chunks up to a 2,000-token budget (fair across chunk sizes).")
    out.append("")

    for d in datasets:
        any_r = next(r for k, r in by.items() if k[0] == d)
        out.append(f"## {d} ({any_r['n_docs']} docs, {any_r['n_queries']} queries)")
        out.append("")
        out.append("| strategy | option | " + " | ".join(c[1] for c in COLS) + " | chunks | avg tok | chunk s |")
        out.append("|---|---|" + "---|" * (len(COLS) + 3))
        for s in STRATS:
            for o in OPTS:
                r = by.get((d, o, s))
                if not r:
                    continue
                out.append(
                    f"| {s} | {o} | " + " | ".join(fmt(r[c]) for c, _ in COLS)
                    + f" | {r['n_chunks']} | {r['avg_tokens']:.0f} | {r['chunk_seconds']:.1f} |"
                )
        out.append("")

    lite = [d for d in ["chroma-lite", "qasper-lite", "ohr-lite"] if any(k[0] == d for k in by)]
    if lite:
        out.append("## Context enrichment (lite subsets, base = each option's fixed_size)")
        out.append("")
        out.append("| dataset | option | R@5 base | R@5 enriched | Δ | R@2k base | R@2k enriched | Δ | queries |")
        out.append("|---|---|---|---|---|---|---|---|---|")
        for d in lite:
            for o in OPTS:
                b, e = by.get((d, o, "fixed_size")), by.get((d, o, "context_enriched"))
                if not (b and e):
                    continue
                out.append(
                    f"| {d} | {o} | {fmt(b['recall@5'])} | {fmt(e['recall@5'])} | {(e['recall@5'] - b['recall@5']) * 100:+.1f} "
                    f"| {fmt(b['recall@2000tok'])} | {fmt(e['recall@2000tok'])} | {(e['recall@2000tok'] - b['recall@2000tok']) * 100:+.1f} | {e['n_queries']} |"
                )
        out.append("")

    # macro average over the three full datasets
    agg: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (d, o, s), r in by.items():
        if d in ("chroma", "qasper", "ohr"):
            agg[(o, s)].append(r["recall@2000tok"])
    if agg:
        out.append("## Macro average R@2k tok over datasets")
        out.append("")
        out.append("| strategy | " + " | ".join(OPTS) + " |")
        out.append("|---|" + "---|" * len(OPTS))
        for s in STRATS:
            vals = [agg.get((o, s)) for o in OPTS]
            out.append(f"| {s} | " + " | ".join(f"{sum(v) / len(v) * 100:.1f} (n={len(v)})" if v else "-" for v in vals) + " |")
        out.append("")

    # chunk-size sweep: separates "implementation quality" from "chunk granularity"
    sweep = [json.loads(line) for p in sorted(RES.glob("sweep_*.jsonl")) for line in p.read_text().splitlines() if line.strip()]
    sweep += [r for r in rows if r["dataset"] in ("chroma", "qasper", "ohr") and r["strategy"] in ("fixed_size", "layout_aware")]
    if any(r["config"]["chunk_size"] != 400 for r in sweep):
        out.append("## Chunk-size sweep (R@2k tok at chunk_size 200 / 300 / 400; avg tokens in brackets)")
        out.append("")
        out.append("| dataset | strategy | option | 200 | 300 | 400 |")
        out.append("|---|---|---|---|---|---|")
        cell = {(r["dataset"], r["strategy"], r["option"], r["config"]["chunk_size"]): r for r in sweep}
        for d in datasets:
            for s in ("fixed_size", "layout_aware"):
                for o in OPTS:
                    vals = []
                    for cs in (200, 300, 400):
                        r = cell.get((d, s, o, cs))
                        vals.append(f"{fmt(r['recall@2000tok'])} ({r['avg_tokens']:.0f})" if r else "-")
                    out.append(f"| {d} | {s} | {o} | " + " | ".join(vals) + " |")
        out.append("")

    (RES / "report.md").write_text("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()

"""Embed all benchmark rows into report_template.html -> <results dir>/chunking_report.html.

    uv run python bench/make_html.py [results-dir]
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results"  # optional results dir

rows = []
for p in sorted(RES.glob("results*.jsonl")) + sorted(RES.glob("sweep_*.jsonl")):
    rows += [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
seen, uniq = set(), []
for r in rows:  # one row per (dataset, option, strategy, chunk_size)
    k = (r["dataset"], r["option"], r["strategy"], r["config"]["chunk_size"])
    if k not in seen:
        seen.add(k)
        uniq.append(r)

html = (ROOT / "bench" / "report_template.html").read_text()
fragments = RES / "report_fragments.json"
if fragments.exists():
    for key, value in json.loads(fragments.read_text()).items():
        html = html.replace(f"<!--{key}-->", value)
else:  # e.g. a run with company models: no hand-written conclusions yet
    html = html.replace(
        "<!--VERDICT-->",
        "<p class=\"note\">这份报告由脚本自动生成，结论还没有人工撰写。"
        "可以和本地开源模型的主报告逐项对比，或运行 <code>bench/compare_runs.py</code> 查看差异。</p>",
    )
boot = RES / "bootstrap.json"
integ = RES / "span_integrity.json"
data = {
    "rows": uniq,
    "tokens": "约 111 万 token 语料",
    "bootstrap": json.loads(boot.read_text()) if boot.exists() else [],
    "integrity": json.loads(integ.read_text()) if integ.exists() else [],
}
html = html.replace("/*DATA*/null", json.dumps(data, ensure_ascii=False))
(RES / "chunking_report.html").write_text(html)
print(f"{len(uniq)} rows -> results/chunking_report.html")

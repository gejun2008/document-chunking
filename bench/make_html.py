"""Embed all benchmark rows into report_template.html -> results/chunking_report.html."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

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
data = {"rows": uniq, "tokens": "约 111 万 token 语料"}
html = html.replace("/*DATA*/null", json.dumps(data, ensure_ascii=False))
(RES / "chunking_report.html").write_text(html)
print(f"{len(uniq)} rows -> results/chunking_report.html")

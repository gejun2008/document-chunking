"""Paired bootstrap 95% CIs for recall differences between implementations.

Re-chunks each (dataset, option, strategy) cell at chunk_size 400, reuses the
cached embeddings, scores every query, then resamples queries with
replacement (paired: the same queries for both sides). Writes <results-dir>/bootstrap.json.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "bench")]
import numpy as np
from datasets import load_chroma, load_ohr, load_qasper
from metrics import evaluate
from run_bench import add_model_args, build, make_embedder

from doc_chunking.text import count_tokens


class Args:
    chunk_size, overlap, percentile, chonkie_threshold, window_tokens, llm_workers = 400, 50, 90.0, 0.15, 1500, 1


ap = argparse.ArgumentParser()
add_model_args(ap)
ap.add_argument("--results-dir", default=str(ROOT / "results"))
args = ap.parse_args()
emb = make_embedder(args)
METRIC = "recall@2000tok"
PAIRS = [("selfbuilt", "langchain"), ("selfbuilt", "chonkie")]
rng = np.random.default_rng(0)
out = []
for ds in (load_chroma(), load_qasper(), load_ohr()):
    qv = emb.embed([q.question for q in ds.queries])
    scores = {}
    for strat in ("fixed_size", "layout_aware", "semantic"):
        for opt in ("selfbuilt", "chonkie", "langchain"):
            ch = build(opt, strat, Args, emb, None, None)
            index, texts = [], []
            for d in ds.docs:
                for c in ch.chunk(d.text):
                    index.append((d.id, c.start, c.end, count_tokens(c.text)))
                    texts.append(c.text_for_embedding)
            rows = evaluate(ds.queries, index, emb.embed(texts), qv, per_doc=ds.name == "qasper", per_query=True)
            scores[(opt, strat)] = np.array([r[METRIC] for r in rows])
        for a, b in PAIRS:
            diff = scores[(a, strat)] - scores[(b, strat)]
            boot = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(5000)])
            lo, hi = np.percentile(boot, [2.5, 97.5])
            r = {"dataset": ds.name, "strategy": strat, "a": a, "b": b, "n": len(diff),
                 "mean_diff": float(diff.mean()), "ci_low": float(lo), "ci_high": float(hi),
                 "significant": bool(lo > 0 or hi < 0)}
            out.append(r)
            print(f"{ds.name:7s} {strat:13s} {a}-{b:9s} {diff.mean()*100:+5.1f} [{lo*100:+5.1f}, {hi*100:+5.1f}] {'*' if r['significant'] else ''}", flush=True)
(Path(args.results_dir) / "bootstrap.json").write_text(json.dumps(out, indent=1))

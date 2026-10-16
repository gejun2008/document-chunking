"""Span-overlap retrieval metrics (same definitions as Chroma's chunking evaluation).

For a query with gold spans R and the top-k retrieved chunks C (character spans):
  recall    = |R ∩ C| / |R|
  precision = |R ∩ C| / |C|
  IoU       = |R ∩ C| / (|R| + |C| - |R ∩ C|)
Because chunk sizes differ between implementations, we also report recall under
a fixed retrieved-token budget, which is the fairest single number.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def merge(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for s, e in sorted(spans):
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def length(spans: list[tuple[int, int]]) -> int:
    return sum(e - s for s, e in spans)


def intersect(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> int:
    i = j = total = 0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if lo < hi:
            total += hi - lo
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def score_query(
    ref_doc: str,
    refs: list[tuple[int, int]],
    ranked: list[tuple[str, int, int, int]],  # (doc_id, start, end, tokens) in rank order
    ks: tuple[int, ...],
    budgets: tuple[int, ...],
) -> dict[str, float]:
    refs = merge(refs)
    ref_len = length(refs)
    out: dict[str, float] = {}

    def overlap_stats(items: list[tuple[str, int, int, int]]) -> tuple[int, int]:
        per_doc: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for d, s, e, _ in items:
            per_doc[d].append((s, e))
        merged = {d: merge(v) for d, v in per_doc.items()}
        ov = intersect(merged.get(ref_doc, []), refs)
        return ov, sum(length(v) for v in merged.values())

    for k in ks:
        ov, ret = overlap_stats(ranked[:k])
        out[f"recall@{k}"] = ov / ref_len
        out[f"precision@{k}"] = ov / ret if ret else 0.0
        out[f"iou@{k}"] = ov / (ret + ref_len - ov) if ret + ref_len - ov else 0.0
    for b in budgets:
        taken, used = [], 0
        for item in ranked:
            if taken and used + item[3] > b:
                break
            taken.append(item)
            used += item[3]
        ov, _ = overlap_stats(taken)
        out[f"recall@{b}tok"] = ov / ref_len
    return out


def evaluate(
    queries,
    chunk_index: list[tuple[str, int, int, int]],
    chunk_vecs: np.ndarray,
    query_vecs: np.ndarray,
    ks: tuple[int, ...] = (1, 3, 5, 10),
    budgets: tuple[int, ...] = (1000, 2000),
    per_doc: bool = False,
    per_query: bool = False,
    expand: list[tuple[str, int, int, int]] | None = None,
) -> dict[str, float] | list[dict[str, float]]:
    """``per_doc=True`` retrieves only among chunks of the query's own document
    (single-document QA such as Qasper, whose questions do not name the paper).
    ``expand[i]`` replaces retrieved chunk ``i`` by a larger span (its parent);
    repeated parents are counted once, in rank order."""
    sims = query_vecs @ chunk_vecs.T
    if per_doc:
        doc_of = np.array([c[0] for c in chunk_index])
        for qi, q in enumerate(queries):
            sims[qi, doc_of != q.doc_id] = -np.inf
    depth = min(len(chunk_index), max(max(ks), 60))
    top = np.argpartition(-sims, depth - 1, axis=1)[:, :depth]
    rows = []
    for qi, q in enumerate(queries):
        order = top[qi][np.argsort(-sims[qi, top[qi]])]
        ranked = [chunk_index[i] for i in order if np.isfinite(sims[qi, i])]
        if expand is not None:
            ranked = list(dict.fromkeys(expand[i] for i in order if np.isfinite(sims[qi, i])))
        rows.append(score_query(q.doc_id, q.spans, ranked, ks, budgets))
    if per_query:
        return rows
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}

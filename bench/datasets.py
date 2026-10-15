"""Public benchmark loaders. Every query carries gold character spans in its document.

* chroma  — Chroma "Evaluating Chunking Strategies for Retrieval" (MIT): 5 corpora,
            472 questions with exact reference spans. Plain text.
* qasper  — allenai/qasper test split (CC-BY-4.0): NLP papers rendered as Markdown
            (#/##/### headings); evidence paragraphs become gold spans.
* ohr     — opendatalab/OHR-Bench (CC-BY-4.0): human-verified OCR ground truth of
            PDFs in 7 domains (EN + ZH, LaTeX tables); evidence text located in the doc.
"""

from __future__ import annotations

import csv
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"


@dataclass
class Doc:
    id: str
    text: str


@dataclass
class Query:
    id: str
    doc_id: str
    question: str
    spans: list[tuple[int, int]]
    meta: dict = field(default_factory=dict)


@dataclass
class Dataset:
    name: str
    docs: list[Doc]
    queries: list[Query]

    def subset(self, doc_ids: set[str], name: str | None = None) -> "Dataset":
        return Dataset(
            name or self.name,
            [d for d in self.docs if d.id in doc_ids],
            [q for q in self.queries if q.doc_id in doc_ids],
        )


def _find_ws(text: str, needle: str) -> tuple[int, int] | None:
    needle = needle.strip()
    if len(needle) < 8:
        return None
    pos = text.find(needle)
    if pos != -1:
        return pos, pos + len(needle)
    words = needle.split()
    if not words:
        return None
    m = re.search(r"\s*".join(re.escape(w) for w in words), text)
    return (m.start(), m.end()) if m else None


def load_chroma() -> Dataset:
    root = DATA / "chroma"
    docs = {p.stem: Doc(p.stem, p.read_text(encoding="utf-8")) for p in sorted(root.glob("*.md"))}
    queries: list[Query] = []
    with open(root / "questions_df.csv", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            refs = json.loads(row["references"])
            text = docs[row["corpus_id"]].text
            spans = []
            for r in refs:
                s, e = int(r["start_index"]), int(r["end_index"])
                if text[s:e] != r["content"]:  # trust content over indices if they disagree
                    found = _find_ws(text, r["content"])
                    if not found:
                        continue
                    s, e = found
                spans.append((s, e))
            if spans:
                queries.append(Query(f"chroma-{i}", row["corpus_id"], row["question"], spans))
    return Dataset("chroma", list(docs.values()), queries)


def load_qasper(n_papers: int = 80, seed: int = 0) -> Dataset:
    import pandas as pd

    df = pd.read_parquet(DATA / "qasper" / "test.parquet")
    rows = df.sample(n=min(n_papers, len(df)), random_state=seed)
    docs, queries = [], []
    for _, r in rows.iterrows():
        parts = [f"# {r.title.strip()}\n", f"## Abstract\n\n{r.abstract.strip()}\n"]
        for name, paras in zip(r.full_text["section_name"], r.full_text["paragraphs"]):
            levels = [p.strip() for p in str(name).split(":::")]
            heading = "#" * min(1 + len(levels), 4) + " " + levels[-1]
            body = "\n\n".join(p.strip() for p in paras if p and p.strip())
            parts.append(f"{heading}\n\n{body}\n" if body else f"{heading}\n")
        text = "\n".join(parts)
        docs.append(Doc(r.id, text))
        qas = r.qas
        for qi, (question, answers) in enumerate(zip(qas["question"], qas["answers"])):
            spans: list[tuple[int, int]] = []
            for ans in answers["answer"]:
                for ev in ans["evidence"]:
                    if ev.startswith("FLOAT SELECTED"):
                        continue
                    found = _find_ws(text, ev)
                    if found and found not in spans:
                        spans.append(found)
                if spans:
                    break  # first annotator that has locatable evidence
            if spans:
                queries.append(Query(f"qasper-{r.id}-{qi}", r.id, question.strip(), spans))
    return Dataset("qasper", docs, queries)


def load_ohr(n_per_domain: int = 10, max_chars: int = 50_000, seed: int = 0) -> Dataset:
    """Sample docs per domain; long filings are truncated at a page boundary to ``max_chars``."""
    import pandas as pd

    df = pd.read_parquet(DATA / "ohr" / "OHR-Bench.parquet", columns=["domain", "doc_name", "page_idx", "gt_text", "qas"])
    by_domain: dict[str, list[str]] = defaultdict(list)
    with_qas = set(df[df.qas.notna()].doc_name)
    for dom, name in df[["domain", "doc_name"]].drop_duplicates().itertuples(index=False):
        if name in with_qas:
            by_domain[dom].append(name)
    rng = random.Random(seed)
    chosen = {n for dom in sorted(by_domain) for n in rng.sample(sorted(by_domain[dom]), min(n_per_domain, len(by_domain[dom])))}

    docs, queries = [], []
    for name, g in df[df.doc_name.isin(chosen)].groupby("doc_name"):
        g = g.assign(p=g.page_idx.astype(int)).sort_values("p")
        pages: list[str] = []
        for t in g.gt_text.fillna(""):
            if pages and sum(len(x) + 2 for x in pages) + len(t) > max_chars:
                break
            pages.append(t.strip())
        text = "\n\n".join(pages)
        domain = g.domain.iloc[0]
        docs.append(Doc(name, text))
        for qas in g.qas:
            if qas is None:
                continue
            for qid, q, ev, src in zip(qas["ID"], qas["questions"], qas["evidence_context"], qas["evidence_source"]):
                found = _find_ws(text, str(ev))
                if found:
                    queries.append(Query(f"ohr-{qid}", name, str(q).strip(), [found], {"domain": domain, "source": src}))
    return Dataset("ohr", docs, queries)


def load_all() -> list[Dataset]:
    return [load_chroma(), load_qasper(), load_ohr()]


if __name__ == "__main__":
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    for ds in load_all():
        toks = sum(len(enc.encode(d.text, disallowed_special=())) for d in ds.docs)
        print(f"{ds.name:7s} docs={len(ds.docs):4d} queries={len(ds.queries):5d} tokens={toks:,}")

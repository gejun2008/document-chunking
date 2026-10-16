"""Download the public benchmark data into data/ at pinned revisions and verify SHA-256.

    uv run python bench/download_data.py            # into ./data
    uv run python bench/download_data.py --out DIR

The same files are committed under data/ (see data/README.md for sources and
licenses); this script re-fetches them and verifies they are unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import httpx

CHROMA = "https://raw.githubusercontent.com/brandonstarxel/chunking_evaluation/e708410d1c61cb76a85cd9d433630ef89b9c6b85/chunking_evaluation/evaluation_framework/general_evaluation_data"
QASPER = "https://huggingface.co/datasets/allenai/qasper/resolve/06806e4608976fc2fac0a090ac425d5b2b29caf4/qasper/test/0000.parquet"
OHR = "https://huggingface.co/datasets/opendatalab/OHR-Bench/resolve/7f833e3eda9a571a9ea545a8f6d476fa1685033d/OHR-Bench.parquet"

FILES = {
    "chroma/chatlogs.md": (f"{CHROMA}/corpora/chatlogs.md", "543a98f82b2a6a492349fd7f8c9d6c3e78d1a4af81d13c79a7fd513c3f65bda6"),
    "chroma/finance.md": (f"{CHROMA}/corpora/finance.md", "1c48d0156820abc88e46e5c992fa0cd2708b07ae59a3771b2b18234b7208561f"),
    "chroma/pubmed.md": (f"{CHROMA}/corpora/pubmed.md", "0fd9242ffb253695e5d0f0215cb79a66a2a8f1236591e764c60d370747684ced"),
    "chroma/state_of_the_union.md": (f"{CHROMA}/corpora/state_of_the_union.md", "6fc21d560d31eb2421e337596feea0f83f1fa9ca02c6c4e47bc26959d7531b37"),
    "chroma/wikitexts.md": (f"{CHROMA}/corpora/wikitexts.md", "74cafcdb771185ecb9f22cc41b73e2e44477e0636ecf9cf1f96fd6ee0f4b86c0"),
    "chroma/questions_df.csv": (f"{CHROMA}/questions_df.csv", "3ab3901889b8900f43775537bf12d36bd6614255fc59124faca31bede1dda7a3"),
    "qasper/test.parquet": (QASPER, "206d05a86672dc330a47de4ff95eb0042d5ce0c74a84885bc3d15c36ae67b86e"),
    "ohr/OHR-Bench.parquet": (OHR, "4b3065ff455d8bdc5bee829a85febc973a0c72f88c49d36deaf01df3257bd52f"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "data"))
    out = Path(p.parse_args().out)
    bad = 0
    with httpx.Client(follow_redirects=True, timeout=300) as client:
        for rel, (url, digest) in FILES.items():
            dest = out / rel
            if dest.exists() and sha256(dest) == digest:
                print(f"ok        {rel}")
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with client.stream("GET", url) as r:
                r.raise_for_status()
                with dest.open("wb") as f:
                    for chunk in r.iter_bytes():
                        f.write(chunk)
            if sha256(dest) != digest:
                print(f"MISMATCH  {rel} (expected {digest})", file=sys.stderr)
                bad += 1
            else:
                print(f"fetched   {rel}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

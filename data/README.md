# Benchmark data

Copies of three public datasets, committed so the benchmark can run without
network access to GitHub or Hugging Face. `bench/download_data.py` fetches the
same files at the same pinned revisions and checks the SHA-256 values below.
How each dataset is loaded and filtered: `bench/datasets.py` and
`docs/benchmark-methodology.md`.

| path | source (pinned revision) | license / terms | SHA-256 |
|---|---|---|---|
| `chroma/*.md`, `chroma/questions_df.csv` | [brandonstarxel/chunking_evaluation](https://github.com/brandonstarxel/chunking_evaluation) @ `e708410` — Smith & Troynikov, *Evaluating Chunking Strategies for Retrieval*, Chroma, 2024 | MIT (repository). The corpora are excerpts of public sources: 2024 State of the Union, WikiText, UltraChat 200k, ConvFinQA, PubMed Central Open Access | see `bench/download_data.py` |
| `qasper/test.parquet` | [allenai/qasper](https://huggingface.co/datasets/allenai/qasper) test split, parquet conversion @ `06806e4` — Dasigi et al., NAACL 2021 | CC BY 4.0 | `206d05a8…b67b86e` |
| `ohr/OHR-Bench.parquet`, `ohr/README.md` | [opendatalab/OHR-Bench](https://huggingface.co/datasets/opendatalab/OHR-Bench) @ `7f833e3` — Zhang et al., ICCV 2025 | CC BY 4.0; the dataset card adds: **for research purposes only and not for commercial use** | `4b3065ff…5bd52f` |

Files are unmodified. `ohr/README.md` is the original OHR-Bench dataset card,
kept for its terms and citation.

Use these files only to evaluate chunking methods. Do not use OHR-Bench data in
products or for any commercial purpose.

import json

import httpx
import numpy as np
import pytest

from doc_chunking.chonkie_impl import ChonkieFixedSizeChunker, ChonkieMarkdownChunker, ChonkieSemanticChunker
from doc_chunking.clients import CachedEmbedder, OpenAICompatibleChat, OpenAICompatibleEmbedder
from doc_chunking.langchain_impl import (
    ContextualEnrichmentTransformer,
    LCFixedSizeChunker,
    LCMarkdownChunker,
    SemanticTextSplitter,
    to_documents,
)
from doc_chunking.selfbuilt import ContextEnricher, FixedSizeChunker, MarkdownChunker, SemanticChunker
from doc_chunking.text import count_tokens, split_sentences

PROSE = " ".join(f"Sentence number {i} talks about topic {i // 10}." for i in range(300))

MARKDOWN = """# Annual Report

Intro paragraph about the company.

## Revenue

| Region | Q1 | Q2 |
|--------|----|----|
| EMEA   | 10 | 12 |
| APAC   | 7  | 9  |

### Notes

Revenue grew in every region.

## Risks

""" + "Supply chain risk remains elevated. " * 80


class TopicEmbedder:
    """Deterministic fake: embeds a sentence by the topic number it mentions."""

    def __init__(self):
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        out = []
        for t in texts:
            v = np.zeros(64, dtype=np.float32)
            for w in t.replace(".", " ").split():
                if w.isdigit():
                    v[int(w) % 64] += 1
            v[63] += 0.1
            out.append(v)
        return np.stack(out)


class EchoLLM:
    def __init__(self):
        self.prompts = []

    def complete(self, prompt, max_tokens=128):
        self.prompts.append(prompt)
        return "This chunk is from the annual report."


def assert_valid(text, chunks, chunk_size, slack=0):
    assert chunks, "no chunks produced"
    for i, c in enumerate(chunks):
        assert c.index == i
        assert text[c.start : c.end] == c.text, "offsets must point at the original text"
        assert count_tokens(c.text) <= chunk_size + slack
    # every non-whitespace character is covered by some chunk
    covered = np.zeros(len(text), bool)
    for c in chunks:
        covered[c.start : c.end] = True
    assert all(covered[i] or text[i].isspace() for i in range(len(text)))


# --------------------------------------------------------------------------- text utils


def test_sentence_split_tiles_text_en_and_cjk():
    text = "Dr. Smith arrived at 3.5 p.m. today. 你好。世界！Next line\nAnother line."
    spans = split_sentences(text)
    assert "".join(text[s:e] for s, e in spans) == text
    pieces = [text[s:e].strip() for s, e in spans]
    assert pieces[0] == "Dr. Smith arrived at 3.5 p.m. today."
    assert "你好。" in pieces and "世界！" in pieces


# --------------------------------------------------------------------------- option 1


def test_fixed_size_respects_size_and_overlaps():
    chunks = FixedSizeChunker(chunk_size=80, overlap=20).chunk(PROSE)
    assert_valid(PROSE, chunks, 80)
    assert all(b.start < a.end for a, b in zip(chunks, chunks[1:])), "consecutive chunks overlap"
    assert all(c.text.endswith(".") for c in chunks), "cuts land on sentence boundaries"


def test_fixed_size_rejects_bad_overlap():
    with pytest.raises(ValueError):
        FixedSizeChunker(chunk_size=50, overlap=50)


def test_fixed_size_hard_splits_giant_sentence():
    text = "word " * 1000
    assert_valid(text, FixedSizeChunker(chunk_size=100, overlap=0).chunk(text), 100)


def test_markdown_keeps_table_whole_and_tracks_headings():
    chunks = MarkdownChunker(chunk_size=50).chunk(MARKDOWN)
    assert_valid(MARKDOWN, chunks, 50)
    table = [c for c in chunks if "| EMEA" in c.text]
    assert len(table) == 1 and "| APAC" in table[0].text
    assert table[0].metadata["heading_path"][:2] == ["Annual Report", "Revenue"]
    chunks = MarkdownChunker(chunk_size=120).chunk(MARKDOWN)
    risks = [c for c in chunks if "Supply chain" in c.text]
    assert len(risks) > 1 and all(c.metadata["heading_path"] == ["Annual Report", "Risks"] for c in risks)
    assert risks[-1].embed_text.startswith("Annual Report > Risks")


def test_markdown_merges_small_sections():
    chunks = MarkdownChunker(chunk_size=400).chunk(MARKDOWN)
    assert "Intro paragraph" in chunks[0].text and "Revenue grew" in chunks[0].text



def test_markdown_without_headings_matches_recursive_fixed_size():
    paras = [" ".join(f"Para {p} sentence {i}." for i in range(4 + p % 7)) for p in range(12)]
    long_para = " ".join(f"Long sentence {i} about one topic." for i in range(50))
    text = "\n\n".join(paras[:6] + [long_para] + paras[6:])
    md = MarkdownChunker(chunk_size=80, overlap=10).chunk(text)
    fixed = FixedSizeChunker(chunk_size=80, overlap=10).chunk(text)
    assert_valid(text, md, 80)
    assert [(c.start, c.end) for c in md] == [(c.start, c.end) for c in fixed]
    assert all(c.metadata["heading_path"] == [] and c.embed_text is None for c in md)


def test_markdown_oversized_section_splits_recursively_around_table():
    paras = [" ".join(f"Para {p} sentence {i}." for i in range(5)) for p in range(8)]
    table = "| Region | Q1 |\n|---|---|\n" + "".join(f"| R{i} | {i} |\n" for i in range(5))
    text = "# Report\n\n## Body\n\n" + "\n\n".join(paras[:4]) + "\n\n" + table + "\n" + "\n\n".join(paras[4:])
    chunks = MarkdownChunker(chunk_size=100, overlap=0).chunk(text)
    assert_valid(text, chunks, 100)
    assert all(_contains(chunks, text, p) for p in paras), "paragraphs that fit are never cut"
    assert _contains(chunks, text, table.strip()), "table stays whole"
    body = [c for c in chunks if "Para" in c.text]
    assert all(c.metadata["heading_path"] == ["Report", "Body"] for c in body)
    assert all(c.embed_text.startswith("Report > Body\n\n") for c in body)


def test_markdown_splits_oversized_table_on_rows():
    table = "| Key | Value |\n|---|---|\n" + "".join(f"| key {i} | value {i} |\n" for i in range(60))
    text = "# Data\n\n" + table
    chunks = MarkdownChunker(chunk_size=60, overlap=0).chunk(text)
    assert_valid(text, chunks, 60)
    rows = [c for c in chunks if "key" in c.text]
    assert len(rows) > 2 and all(c.text.startswith("|") and c.text.endswith("|") for c in rows)

def test_semantic_cuts_on_topic_change():
    text = " ".join(f"Topic {t} fact {t}." for t in (1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3))
    chunks = SemanticChunker(TopicEmbedder(), chunk_size=200, min_chunk_tokens=1,
                             breakpoint_percentile=80, window=0).chunk(text)
    assert_valid(text, chunks, 200)
    assert [c.text.count("Topic 2") for c in chunks] == [0, 4, 0]


def test_semantic_respects_chunk_size():
    chunks = SemanticChunker(TopicEmbedder(), chunk_size=60).chunk(PROSE)
    assert_valid(PROSE, chunks, 60)


def test_context_enricher_keeps_offsets_and_caches():
    llm, cache = EchoLLM(), {}
    enricher = ContextEnricher(MarkdownChunker(chunk_size=120), llm, cache=cache, max_workers=2)
    chunks = enricher.chunk(MARKDOWN)
    assert_valid(MARKDOWN, chunks, 120)
    for c in chunks:
        assert c.embed_text.startswith("This chunk is from the annual report.")
        assert c.metadata["strategy"] == "context_enriched"
        assert c.metadata["base_strategy"] == "layout_aware"
    n = len(llm.prompts)
    enricher.chunk(MARKDOWN)
    assert len(llm.prompts) == n, "second run served from cache"


def test_context_window_is_bounded():
    llm = EchoLLM()
    long_text = PROSE * 5
    ContextEnricher(FixedSizeChunker(200, 0), llm, window_tokens=500, max_workers=1).chunk(long_text)
    assert max(count_tokens(p) for p in llm.prompts) < 1000


# --------------------------------------------------------------------------- option 2


def test_chonkie_chunkers_produce_valid_offsets():
    assert_valid(PROSE, ChonkieFixedSizeChunker(80, 20).chunk(PROSE), 80)
    md_chunks = ChonkieMarkdownChunker(120).chunk(MARKDOWN)
    assert_valid(MARKDOWN, md_chunks, 120)
    assert any("| EMEA" in c.text and "| APAC" in c.text for c in md_chunks)
    assert_valid(PROSE, ChonkieSemanticChunker(TopicEmbedder(), 80, threshold=0.5).chunk(PROSE), 80)


# --------------------------------------------------------------------------- option 3


def test_langchain_chunkers_produce_valid_offsets():
    assert_valid(PROSE, LCFixedSizeChunker(80, 20).chunk(PROSE), 80)
    md_chunks = LCMarkdownChunker(120).chunk(MARKDOWN)
    assert_valid(MARKDOWN, md_chunks, 120)
    assert md_chunks[-1].metadata["heading_path"] == ["Annual Report", "Risks"]


def test_langchain_semantic_splitter_is_a_text_splitter():
    splitter = SemanticTextSplitter(TopicEmbedder(), chunk_size=80)
    docs = splitter.create_documents([PROSE])
    assert docs and all("start_index" in d.metadata for d in docs)
    assert all(count_tokens(d.page_content) <= 80 for d in docs)


def test_langchain_context_transformer():
    from langchain_core.documents import Document

    tr = ContextualEnrichmentTransformer(LCFixedSizeChunker(100, 0), EchoLLM(), max_workers=1)
    out = tr.transform_documents([Document(page_content=PROSE, metadata={"source": "a.md"})])
    assert out and all(d.metadata["source"] == "a.md" for d in out)
    assert all(d.metadata["embed_text"].startswith("This chunk is") for d in out)
    assert to_documents(tr.chunk(PROSE))[0].metadata["context"]


# --------------------------------------------------------------------------- clients


def _mock_client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_openai_compatible_embedder_batches_and_orders():
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append(len(body["input"]))
        data = [{"index": i, "embedding": [float(len(t))]} for i, t in enumerate(body["input"])][::-1]
        return httpx.Response(200, json={"data": data})

    emb = OpenAICompatibleEmbedder(base_url="http://x/v1", model="m", batch_size=2)
    emb._client = _mock_client(handler)
    out = emb.embed(["a", "bb", "ccc"])
    assert seen == [2, 1]
    assert out[:, 0].tolist() == [1.0, 2.0, 3.0]


def test_openai_compatible_embedder_retries_on_429(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429)
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0]}]})

    emb = OpenAICompatibleEmbedder(base_url="http://x", model="m")
    emb._client = _mock_client(handler)
    assert emb.embed(["a"]).shape == (1, 1) and calls["n"] == 2


def test_cached_embedder_dedupes():
    inner = TopicEmbedder()
    cached = CachedEmbedder(inner)
    cached.embed(["Topic 1.", "Topic 2."])
    cached.embed(["Topic 1.", "Topic 2."])
    assert inner.calls == 1


def test_openai_compatible_chat():
    def handler(request):
        assert json.loads(request.content)["messages"][0]["content"] == "hi"
        return httpx.Response(200, json={"choices": [{"message": {"content": " hello "}}]})

    chat = OpenAICompatibleChat(base_url="http://x", model="m")
    chat._client = _mock_client(handler)
    assert chat.complete("hi") == "hello"


def test_fixed_size_prefers_paragraph_boundaries():
    paras = [" ".join(f"Para {p} sentence {i}." for i in range(12)) for p in range(8)]
    text = "\n\n".join(paras)
    chunks = FixedSizeChunker(chunk_size=200, overlap=0).chunk(text)
    assert_valid(text, chunks, 200)
    whole = sum(any(c.start <= text.index(p) and text.index(p) + len(p) <= c.end for c in chunks) for p in paras)
    assert whole == len(paras)


# --------------------------------------------------------------------------- option 1, recursive fixed size


def _contains(chunks, text, piece):
    i = text.index(piece)
    return any(c.start <= i and i + len(piece) <= c.end for c in chunks)


def test_recursive_merges_small_paragraphs_and_keeps_them_whole():
    paras = [" ".join(f"Para {p} sentence {i}." for i in range(5)) for p in range(10)]
    text = "\n\n".join(paras)
    chunks = FixedSizeChunker(chunk_size=120, overlap=0).chunk(text)
    assert_valid(text, chunks, 120)
    assert all(_contains(chunks, text, p) for p in paras), "small paragraphs are never cut"
    assert len(chunks) < len(paras), "small paragraphs are merged together"
    for c in chunks:  # every cut falls on a paragraph boundary
        assert c.end == len(text) or text[c.end : c.end + 2] == "\n\n"


def test_recursive_long_paragraph_falls_back_to_sentences_with_overlap():
    long_para = " ".join(f"Long sentence {i} about the second topic." for i in range(60))
    text = "Short intro.\n\n" + long_para + "\n\nShort outro."
    chunks = FixedSizeChunker(chunk_size=80, overlap=20).chunk(text)
    assert_valid(text, chunks, 80)
    inner = [c for c in chunks if c.start >= text.index(long_para)]
    assert len(inner) > 2
    assert all(c.text.endswith(".") and c.text[0].isupper() for c in inner), "cuts land on sentence boundaries"
    assert all(b.start < a.end for a, b in zip(inner, inner[1:]) if b.start < len(text) - 20), "sentence overlap"
    # the short paragraphs around the long one are not glued onto its pieces
    assert chunks[0].text == "Short intro." and chunks[-1].text == "Short outro."


def test_recursive_splits_lines_then_cjk_sentences():
    lines = "\n".join(f"line {i}: " + "word " * 15 for i in range(20))
    chunks = FixedSizeChunker(chunk_size=60, overlap=0).chunk(lines)
    assert_valid(lines, chunks, 60)
    assert all(c.text.startswith("line ") for c in chunks), "cuts land on line starts"

    cjk = "".join(f"第{i}句话讨论了文档切分的一个重要问题。" for i in range(80))
    chunks = FixedSizeChunker(chunk_size=50, overlap=10).chunk(cjk)
    assert_valid(cjk, chunks, 50)
    assert all(c.text.startswith("第") and c.text.endswith("。") for c in chunks)


def test_recursive_hard_splits_unbreakable_text():
    text = "Intro.\n\n" + "x" * 5000 + "\n\nOutro."
    chunks = FixedSizeChunker(chunk_size=100, overlap=10).chunk(text)
    assert_valid(text, chunks, 100)


def test_recursive_handles_blank_and_ragged_whitespace():
    assert FixedSizeChunker(chunk_size=50, overlap=5).chunk("  \n\n \n ") == []
    text = "\n\n\n  First para.\n \n\n\tSecond para.  \n\n"
    chunks = FixedSizeChunker(chunk_size=50, overlap=5).chunk(text)
    assert_valid(text, chunks, 50)
    assert [c.text for c in chunks] == ["First para.\n \n\n\tSecond para."]

import pytest

from doc_chunking.legacy_api import to_legacy_response
from doc_chunking.selfbuilt import FixedSizeChunker, MarkdownChunker, ParentChildChunker
from doc_chunking.selfbuilt.parent_child import SUMMARY_SYSTEM
from doc_chunking.text import count_tokens

PARAS = [" ".join(f"Section {p} sentence {i} mentions revenue of {p * 10 + i} million." for i in range(12)) for p in range(12)]
DOC = "\n\n".join(PARAS)


class SummaryLLM:
    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    def complete(self, prompt, max_tokens=128, system=None):
        self.calls.append((system, prompt))
        if self.fail:
            raise RuntimeError("llm down")
        first = prompt.split("<SOURCE_TEXT>\n")[1].split(".")[0]
        return f"Summary: {first}."


def split(chunks):
    return [c for c in chunks if c.metadata["level"] == "parent"], [c for c in chunks if c.metadata["level"] == "child"]


def test_structure_offsets_and_mapping():
    chunks = ParentChildChunker(child_size=64, parent_multiplier=4, child_overlap=8).chunk(DOC, "doc-1")
    parents, children = split(chunks)
    assert len(parents) > 1 and len(children) > len(parents)
    by_id = {p.id: p for p in parents}
    for c in chunks:
        assert DOC[c.start : c.end] == c.text, "text is the verbatim source span"
        assert c.metadata["doc_id"] == "doc-1"
    for p in parents:
        assert count_tokens(p.text) <= 256
        assert [k.id for k in children if k.metadata["parent_id"] == p.id] == p.metadata["child_ids"]
    for k in children:
        parent = by_id[k.metadata["parent_id"]]
        assert parent.start <= k.start and k.end <= parent.end, "child lies inside its parent"
        assert count_tokens(k.text) <= 64
        assert k.embed_text is None, "no LLM -> no summary prefix"
    assert children[0].metadata["prev_id"] is None and children[-1].metadata["next_id"] is None
    assert all(a.metadata["next_id"] == b.id and b.metadata["prev_id"] == a.id for a, b in zip(children, children[1:]))
    assert chunks[0].metadata["strategy"] == "parent_child"


def test_one_summary_per_parent_with_legacy_prompts():
    llm = SummaryLLM()
    chunks = ParentChildChunker(child_size=64, child_overlap=8, llm=llm, max_workers=1).chunk(DOC, "d")
    parents, children = split(chunks)
    assert len(llm.calls) == len(parents)
    assert all(system == SUMMARY_SYSTEM and "<SOURCE_TEXT>" in prompt for system, prompt in llm.calls)
    by_id = {p.id: p for p in parents}
    for k in children:
        summary = by_id[k.metadata["parent_id"]].metadata["summary"]
        assert k.embed_text == f"Context summary: {summary}\n\n{k.text}"
        assert k.metadata["summary"] == summary
    assert chunks[0].metadata["strategy"] == "parent_summary"


def test_summary_failure_raises_or_skips():
    with pytest.raises(RuntimeError):
        ParentChildChunker(child_size=64, child_overlap=8, llm=SummaryLLM(fail=True), max_workers=1).chunk(DOC)
    chunks = ParentChildChunker(child_size=64, child_overlap=8, llm=SummaryLLM(fail=True), on_error="skip").chunk(DOC)
    assert all(c.embed_text is None for c in chunks)


def test_summary_cache_avoids_repeat_calls():
    llm, cache = SummaryLLM(), {}
    pc = ParentChildChunker(child_size=64, child_overlap=8, llm=llm, cache=cache, max_workers=1)
    pc.chunk(DOC)
    n = len(llm.calls)
    pc.chunk(DOC)
    assert len(llm.calls) == n


def test_stable_ids():
    pc = ParentChildChunker(child_size=64, child_overlap=8)
    a, b = pc.chunk(DOC, "d"), pc.chunk(DOC, "d")
    assert [c.id for c in a] == [c.id for c in b], "same input -> same IDs"
    assert len({c.id for c in a}) == len(a), "IDs are unique within a document"
    assert {c.id for c in pc.chunk(DOC, "other")}.isdisjoint({c.id for c in a}), "IDs differ across documents"
    edited = DOC.replace("revenue of 115 million", "revenue of 999 million")  # last paragraph only
    c = pc.chunk(edited, "d")
    same = sum(x.id == y.id for x, y in zip(a, c))
    assert same >= len(a) - 3, "an edit only changes the IDs of the chunks that contain it"
    other_cfg = ParentChildChunker(child_size=80, child_overlap=8).chunk(DOC, "d")
    assert {x.id for x in other_cfg}.isdisjoint({x.id for x in a}), "other parameters -> other IDs"


def test_markdown_parents_keep_heading_path():
    md = "# Report\n\n## Revenue\n\n" + PARAS[0] + "\n\n## Risks\n\n" + PARAS[1]
    chunks = ParentChildChunker(child_size=64, child_overlap=8, parent_chunker=MarkdownChunker(200)).chunk(md)
    _, children = split(chunks)
    paths = {tuple(k.metadata["heading_path"]) for k in children}
    assert ("Report", "Risks") in paths, "children inherit their parent's heading path"
    assert all(p and p[0] == "Report" for p in paths)


def test_legacy_response_shape():
    llm = SummaryLLM()
    chunks = ParentChildChunker(child_size=64, child_overlap=8, llm=llm).chunk(DOC, "doc-9")
    resp = to_legacy_response(chunks, "doc-9", id_style="sequential")
    first = resp["chunks"][0]
    assert first["chunk_id"] == "chunk-0001" and first["metadata"]["parent_id"] == "parent-0001"
    assert first["text"].startswith("Context summary: ") and DOC[first["metadata"]["start"] : first["metadata"]["end"]] == first["raw_text"]
    assert set(first["metadata"]) >= {"hierarchical_path", "parent_id", "prev_id", "next_id"}
    assert resp["chunks"][1]["metadata"]["prev_id"] == "chunk-0001"
    parent = resp["parents"][0]
    assert parent["parent_id"] == "parent-0001" and DOC[parent["start"] : parent["end"]] == parent["text"]
    assert "chunk-0001" in parent["child_ids"]
    stable = to_legacy_response(chunks, "doc-9")
    assert stable["chunks"][0]["chunk_id"].startswith("c_") and stable["parents"][0]["parent_id"].startswith("p_")


def test_legacy_response_for_flat_strategies():
    resp = to_legacy_response(FixedSizeChunker(64, 8).chunk(DOC), "d")
    assert resp["parents"] == [] and all(c["metadata"]["parent_id"] is None for c in resp["chunks"])
    assert resp["chunks"][0]["text"] == resp["chunks"][0]["raw_text"]
    assert len({c["chunk_id"] for c in resp["chunks"]}) == len(resp["chunks"])

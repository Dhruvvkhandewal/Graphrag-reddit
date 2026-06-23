"""Unit tests for the sentence-aware chunker."""
from vectorstore.chunker import chunk_text, split_sentences


def test_short_text_single_chunk():
    chunks = chunk_text("t1_x", "A short comment.", max_chars=800)
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "t1_x::0"
    assert chunks[0].doc_id == "t1_x"


def test_long_text_splits_with_stable_ids():
    text = " ".join(f"Sentence number {i} about LLMs and RAG systems." for i in range(60))
    chunks = chunk_text("t3_post", text, max_chars=200, overlap_chars=40)
    assert len(chunks) > 1
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert all(c.doc_id == "t3_post" for c in chunks)
    assert all(c.chunk_id == f"t3_post::{c.index}" for c in chunks)


def test_overlap_preserves_continuity():
    text = "Alpha one. Beta two. Gamma three. Delta four. Epsilon five. Zeta six."
    chunks = chunk_text("t1_y", text, max_chars=30, overlap_chars=15)
    # consecutive chunks should share at least one token (overlap)
    joined = [set(c.text.split()) for c in chunks]
    assert any(joined[i] & joined[i + 1] for i in range(len(joined) - 1))


def test_split_sentences_basic():
    assert split_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]
    assert split_sentences("") == []

"""Deterministic, sentence-aware chunking.

Reddit content is short (most comments < 80 words) but posts and the
occasional essay-comment can be long. We split on sentence boundaries and
pack sentences into ~`max_chars` windows with a small overlap so semantic
continuity is preserved across chunk edges. Short documents pass through as a
single chunk — no needless fragmentation.

Each chunk inherits the parent document's identity via a stable
`<doc_id>::<n>` chunk id, which keeps the vector store joinable back to the
graph on `doc_id`.

Pure-Python and unit-tested.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    index: int


def split_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    parts = _SENT_SPLIT.split(text)
    return [p.strip() for p in parts if p.strip()]


def chunk_text(
    doc_id: str, text: str, *, max_chars: int = 800, overlap_chars: int = 120
) -> List[Chunk]:
    """Pack sentences into overlapping windows bounded by `max_chars`."""
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [Chunk(chunk_id=f"{doc_id}::0", doc_id=doc_id, text=text, index=0)]

    sentences = split_sentences(text) or [text]
    chunks: List[Chunk] = []
    buf: List[str] = []
    buf_len = 0
    idx = 0

    def flush():
        nonlocal buf, buf_len, idx
        if not buf:
            return
        chunk_text_ = " ".join(buf).strip()
        chunks.append(
            Chunk(chunk_id=f"{doc_id}::{idx}", doc_id=doc_id, text=chunk_text_, index=idx)
        )
        idx += 1

    for sent in sentences:
        # A single oversized sentence: hard-split it.
        if len(sent) > max_chars:
            flush()
            buf, buf_len = [], 0
            for i in range(0, len(sent), max_chars):
                piece = sent[i : i + max_chars]
                chunks.append(
                    Chunk(chunk_id=f"{doc_id}::{idx}", doc_id=doc_id, text=piece, index=idx)
                )
                idx += 1
            continue

        if buf_len + len(sent) + 1 > max_chars and buf:
            flush()
            # Start next window with a tail overlap for continuity.
            overlap, o_len = [], 0
            for s in reversed(buf):
                if o_len + len(s) > overlap_chars:
                    break
                overlap.insert(0, s)
                o_len += len(s) + 1
            buf, buf_len = list(overlap), o_len
        buf.append(sent)
        buf_len += len(sent) + 1

    flush()
    return chunks

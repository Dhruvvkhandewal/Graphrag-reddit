"""Dependency-free heuristic fallbacks for enrichment.

These are NOT meant to rival the LLM — they exist so the full pipeline (and
the demo) runs end-to-end even with no GEMINI_API_KEY, which is invaluable for
local development and CI. When a key is present, the LLM path is used.
"""
from __future__ import annotations

import re
from typing import List, Tuple

from utils.models import Entity, Sentiment

# Tiny lexicons — illustrative, not exhaustive.
_POS = {
    "great", "love", "amazing", "excellent", "impressive", "best", "good",
    "useful", "powerful", "fast", "reliable", "promising", "breakthrough",
}
_NEG = {
    "bad", "terrible", "hate", "awful", "broken", "slow", "useless", "worse",
    "concern", "concerns", "dangerous", "risk", "risky", "overhyped", "fail",
    "failed", "bug", "buggy", "disappointing", "scam",
}
# Capitalised multi-word or known-tech tokens => candidate entities.
_ENTITY_RE = re.compile(
    r"\b([A-Z][A-Za-z0-9.+-]*(?:\s+[A-Z][A-Za-z0-9.+-]*){0,2})\b"
)
_STOP_ENTITIES = {"I", "The", "This", "That", "It", "Reddit", "OP", "AI"}


def heuristic_sentiment(text: str) -> Tuple[Sentiment, float]:
    tokens = re.findall(r"[a-z']+", text.lower())
    pos = sum(t in _POS for t in tokens)
    neg = sum(t in _NEG for t in tokens)
    if pos == neg == 0:
        return Sentiment.NEUTRAL, 0.0
    score = (pos - neg) / (pos + neg)
    if pos and neg:
        label = Sentiment.MIXED
    elif score > 0:
        label = Sentiment.POSITIVE
    else:
        label = Sentiment.NEGATIVE
    return label, round(score, 3)


def heuristic_entities(text: str, limit: int = 8) -> List[Entity]:
    found, seen = [], set()
    for m in _ENTITY_RE.finditer(text):
        # Never let a candidate span a sentence boundary, and trim stray punctuation.
        name = re.split(r"[.!?](?:\s|$)", m.group(1))[0].strip(" .,:;")
        if not name or name in _STOP_ENTITIES or len(name) < 3 or name.lower() in seen:
            continue
        seen.add(name.lower())
        found.append(Entity(name=name, type="CONCEPT"))
        if len(found) >= limit:
            break
    return found


def heuristic_topics(text: str, limit: int = 5) -> List[str]:
    words = re.findall(r"[a-z][a-z'-]{3,}", text.lower())
    freq: dict[str, int] = {}
    for w in words:
        if w not in _POS and w not in _NEG:
            freq[w] = freq.get(w, 0) + 1
    ranked = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)
    return [w for w, _ in ranked[:limit]]

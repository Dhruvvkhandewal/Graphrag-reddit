"""Small, dependency-free text helpers used across ingestion + retrieval."""
from __future__ import annotations

import re

_WHITESPACE = re.compile(r"\s+")
_URL = re.compile(r"https?://\S+")


def clean_text(text: str | None) -> str:
    """Normalise whitespace and strip control noise. Keeps URLs (signal)."""
    if not text:
        return ""
    text = text.replace("\u200b", " ").replace("&amp;", "&").replace("&#x200B;", " ")
    return _WHITESPACE.sub(" ", text).strip()


def truncate(text: str, max_chars: int = 4000) -> str:
    """Bound text sent to the LLM to keep token cost predictable."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + " …"


def strip_urls(text: str) -> str:
    return _URL.sub("", text)

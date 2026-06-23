"""Final answer synthesis with source citations.

Takes fused (or grouped, for comparisons) hits, renders them as a numbered
source block, and asks Gemini to answer using ONLY those sources with inline
[n] citations. Returns the answer plus a structured citation list so the API
can render clickable Reddit permalinks.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Sequence

from llm import prompts
from llm.gemini_client import GeminiClient, get_gemini_client
from retrieval.models import FusedHit
from utils.text import truncate


@dataclass
class Citation:
    number: int
    doc_id: str
    url: str
    subreddit: str
    author: str
    created_at: str
    snippet: str


@dataclass
class GeneratedAnswer:
    answer: str
    citations: List[Citation]


def _format_source(n: int, hit: FusedHit) -> str:
    md = hit.metadata or {}
    ts = md.get("created_utc")
    when = (
        dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc).strftime("%Y-%m-%d")
        if ts else "unknown date"
    )
    head = (
        f"[{n}] r/{md.get('subreddit','?')} | u/{md.get('author','?')} | "
        f"{when} | {md.get('type','content')}"
    )
    return f"{head}\n{truncate(hit.text, 600)}"


def _to_citation(n: int, hit: FusedHit) -> Citation:
    md = hit.metadata or {}
    ts = md.get("created_utc")
    when = (
        dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc).isoformat()
        if ts else ""
    )
    return Citation(
        number=n,
        doc_id=hit.doc_id,
        url=md.get("url", ""),
        subreddit=md.get("subreddit", ""),
        author=md.get("author", ""),
        created_at=when,
        snippet=truncate(hit.text, 200),
    )


class AnswerGenerator:
    def __init__(self, client: GeminiClient | None = None):
        self._client = client or get_gemini_client()

    def generate(
        self,
        question: str,
        hits: Sequence[FusedHit],
        *,
        is_comparison: bool = False,
        period_groups: Dict[str, Sequence[FusedHit]] | None = None,
    ) -> GeneratedAnswer:
        if period_groups:
            source_block, citations = self._render_grouped(period_groups)
        else:
            source_block, citations = self._render_flat(hits)

        if not citations:
            return GeneratedAnswer(
                answer="No relevant Reddit content was retrieved for this question.",
                citations=[],
            )

        prompt = prompts.ANSWER_PROMPT.format(
            question=question,
            sources=source_block,
            temporal_note=prompts.COMPARISON_NOTE if is_comparison else "",
        )
        answer = self._client.generate_text(prompt, temperature=0.3)
        if answer is None:
            answer = self._fallback_answer(citations, is_comparison)
        return GeneratedAnswer(answer=answer.strip(), citations=citations)

    # ---- rendering ----------------------------------------------------
    def _render_flat(self, hits: Sequence[FusedHit]):
        lines, citations = [], []
        for i, hit in enumerate(hits, start=1):
            lines.append(_format_source(i, hit))
            citations.append(_to_citation(i, hit))
        return "\n\n".join(lines), citations

    def _render_grouped(self, groups: Dict[str, Sequence[FusedHit]]):
        lines, citations, n = [], [], 0
        for label, hits in groups.items():
            lines.append(f"=== PERIOD: {label} ===")
            for hit in hits:
                n += 1
                lines.append(_format_source(n, hit))
                citations.append(_to_citation(n, hit))
        return "\n\n".join(lines), citations

    @staticmethod
    def _fallback_answer(citations: List[Citation], is_comparison: bool) -> str:
        subs = sorted({c.subreddit for c in citations if c.subreddit})
        head = (
            "LLM synthesis unavailable (no API key). Returning a retrieval "
            "summary instead.\n\n"
        )
        body = (
            f"Retrieved {len(citations)} relevant items across "
            f"{len(subs)} communities ({', '.join(subs) or 'n/a'}). "
        )
        if is_comparison:
            body += "Sources span multiple time periods; see citations for the timeline. "
        body += "Top sources: " + "; ".join(
            f"[{c.number}] r/{c.subreddit} {c.snippet[:80]}" for c in citations[:3]
        )
        return head + body

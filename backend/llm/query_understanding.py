"""LLM-powered query understanding / routing.

Combines a deterministic temporal parse (free, exact) with a Gemini call that
infers intent, topics, entities and per-retriever weights. The result is a
`QueryPlan` that drives the hybrid retriever. If the LLM is unavailable, we
fall back to a keyword heuristic so routing still works.
"""
from __future__ import annotations

import datetime as dt
import re
from typing import Optional

from llm import prompts
from llm.gemini_client import GeminiClient, get_gemini_client
from retrieval.models import QueryPlan
from retrieval.temporal import parse_temporal
from utils.logging import get_logger

logger = get_logger("llm.query_understanding")

_RELATIONAL_HINTS = re.compile(
    r"\b(who|whom|which\s+communit|most\s+influential|leading|top\s+\w+\s+voices|"
    r"influence|connect|relationship|between)\b",
    re.IGNORECASE,
)


class QueryUnderstanding:
    def __init__(self, client: GeminiClient | None = None):
        self._client = client or get_gemini_client()

    def plan(self, query: str, now: Optional[dt.datetime] = None) -> QueryPlan:
        ranges, is_comparison = parse_temporal(query, now)
        data = self._client.generate_json(
            prompts.QUERY_UNDERSTANDING_PROMPT.format(query=query), default=None
        )
        if data is None:
            return self._heuristic_plan(query, ranges, is_comparison)

        intent = str(data.get("intent", "hybrid")).lower()
        return QueryPlan(
            raw_query=query,
            normalized_query=str(data.get("normalized_query") or query),
            intent=intent if intent in {"semantic", "relational", "hybrid", "temporal"} else "hybrid",
            topics=[str(t).lower() for t in data.get("topics", [])],
            entities=[str(e) for e in data.get("entities", [])],
            subreddits=[str(s) for s in data.get("subreddits", [])],
            time_ranges=ranges,
            is_comparison=is_comparison,
            graph_weight=float(data.get("graph_weight", 1.0) or 1.0),
            vector_weight=float(data.get("vector_weight", 1.0) or 1.0),
        )

    @staticmethod
    def _heuristic_plan(query, ranges, is_comparison) -> QueryPlan:
        relational = bool(_RELATIONAL_HINTS.search(query))
        return QueryPlan(
            raw_query=query,
            normalized_query=query,
            intent="relational" if relational else "hybrid",
            topics=[w for w in re.findall(r"[a-zA-Z][a-zA-Z-]{3,}", query.lower())][:6],
            entities=[],
            subreddits=[],
            time_ranges=ranges,
            is_comparison=is_comparison,
            graph_weight=1.4 if relational else 1.0,
            vector_weight=0.8 if relational else 1.0,
        )

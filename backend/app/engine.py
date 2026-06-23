"""QueryEngine — the single application-level entry point for answering.

Composes hybrid retrieval with LLM answer synthesis and packages everything
(graph-only, vector-only, fused, citations, per-period breakdown) into one
result object that both the API and the demo render.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from llm.answer_generator import AnswerGenerator, Citation
from retrieval.hybrid_retriever import HybridResult, HybridRetriever, PeriodResult
from utils.logging import get_logger

logger = get_logger("app.engine")


@dataclass
class AnswerResult:
    question: str
    answer: str
    intent: str
    is_comparison: bool
    time_windows: List[str]
    citations: List[Citation] = field(default_factory=list)
    retrieval: Optional[HybridResult] = None  # full retrieval detail for demo/UI


class QueryEngine:
    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        answer_generator: AnswerGenerator | None = None,
    ) -> None:
        self._retriever = retriever or HybridRetriever()
        self._answerer = answer_generator or AnswerGenerator()

    def answer(self, question: str) -> AnswerResult:
        result = self._retriever.retrieve(question)
        plan = result.plan

        if result.period_results:
            groups: Dict[str, List] = {
                p.label: p.fused for p in result.period_results
            }
            generated = self._answerer.generate(
                question, result.fused, is_comparison=True, period_groups=groups
            )
        else:
            generated = self._answerer.generate(
                question, result.fused, is_comparison=plan.is_comparison
            )

        return AnswerResult(
            question=question,
            answer=generated.answer,
            intent=plan.intent,
            is_comparison=plan.is_comparison,
            time_windows=[r.label for r in plan.time_ranges],
            citations=generated.citations,
            retrieval=result,
        )


_engine: QueryEngine | None = None


def get_query_engine() -> QueryEngine:
    global _engine
    if _engine is None:
        _engine = QueryEngine()
    return _engine

"""Hybrid retriever — the orchestration core.

Flow for a single query:
    1. Query understanding -> QueryPlan (intent, topics, entities, time, weights)
    2. Run graph + vector retrievers CONCURRENTLY (threads; both are I/O bound)
    3. Fuse with Reciprocal Rank Fusion using the plan's per-retriever weights
    4. Return graph-only, vector-only AND fused lists (the assignment requires
       showing all three)

For time-comparison queries (e.g. "Q1 2026 vs Q4 2025") we execute steps 2–3
once PER period and expose `period_results`, enabling true side-by-side
analysis downstream.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from config import get_settings
from llm.query_understanding import QueryUnderstanding
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.graph_retriever import GraphRetriever
from retrieval.models import FusedHit, QueryPlan, RetrievedHit, TimeRange
from retrieval.vector_retriever import VectorRetriever
from utils.logging import get_logger

logger = get_logger("retrieval.hybrid")


@dataclass
class PeriodResult:
    label: str
    graph_hits: List[RetrievedHit] = field(default_factory=list)
    vector_hits: List[RetrievedHit] = field(default_factory=list)
    fused: List[FusedHit] = field(default_factory=list)


@dataclass
class HybridResult:
    plan: QueryPlan
    graph_hits: List[RetrievedHit] = field(default_factory=list)
    vector_hits: List[RetrievedHit] = field(default_factory=list)
    fused: List[FusedHit] = field(default_factory=list)
    period_results: Optional[List[PeriodResult]] = None  # set for comparisons


class HybridRetriever:
    def __init__(
        self,
        query_understanding: QueryUnderstanding | None = None,
        graph_retriever: GraphRetriever | None = None,
        vector_retriever: VectorRetriever | None = None,
    ) -> None:
        self._settings = get_settings()
        self._qu = query_understanding or QueryUnderstanding()
        self._graph = graph_retriever or GraphRetriever()
        self._vector = vector_retriever or VectorRetriever()

    def retrieve(self, query: str) -> HybridResult:
        plan = self._qu.plan(query)
        logger.info(
            "Plan: intent=%s comparison=%s ranges=%s gw=%.2f vw=%.2f",
            plan.intent, plan.is_comparison,
            [r.label for r in plan.time_ranges], plan.graph_weight, plan.vector_weight,
        )

        # Comparison across >=2 explicit periods -> per-period retrieval.
        if plan.is_comparison and len(plan.time_ranges) >= 2:
            periods = [self._retrieve_window(plan, r) for r in plan.time_ranges]
            # Also provide a flat fused view (all periods merged) for convenience.
            flat = reciprocal_rank_fusion(
                [
                    [h for p in periods for h in p.graph_hits],
                    [h for p in periods for h in p.vector_hits],
                ],
                source_names=["graph", "vector"],
                weights=[plan.graph_weight, plan.vector_weight],
                k=self._settings.rrf_k,
                top_n=self._settings.retrieval_top_k,
            )
            return HybridResult(
                plan=plan,
                fused=flat,
                period_results=periods,
                graph_hits=[h for p in periods for h in p.graph_hits],
                vector_hits=[h for p in periods for h in p.vector_hits],
            )

        # Single window (or no time filter).
        time_range = plan.time_ranges[0] if plan.time_ranges else None
        pr = self._retrieve_window(plan, time_range)
        return HybridResult(
            plan=plan, graph_hits=pr.graph_hits, vector_hits=pr.vector_hits, fused=pr.fused
        )

    # ------------------------------------------------------------------
    def _retrieve_window(self, plan: QueryPlan, time_range: Optional[TimeRange]) -> PeriodResult:
        cand_k = self._settings.retriever_candidate_k
        with ThreadPoolExecutor(max_workers=2) as ex:
            f_graph = ex.submit(self._graph.retrieve, plan, time_range=time_range, top_k=cand_k)
            f_vector = ex.submit(self._vector.retrieve, plan, time_range=time_range, top_k=cand_k)
            graph_hits = f_graph.result()
            vector_hits = f_vector.result()

        fused = reciprocal_rank_fusion(
            [graph_hits, vector_hits],
            source_names=["graph", "vector"],
            weights=[plan.graph_weight, plan.vector_weight],
            k=self._settings.rrf_k,
            top_n=self._settings.retrieval_top_k,
        )
        label = time_range.label if time_range else "all_time"
        return PeriodResult(
            label=label, graph_hits=graph_hits, vector_hits=vector_hits, fused=fused
        )

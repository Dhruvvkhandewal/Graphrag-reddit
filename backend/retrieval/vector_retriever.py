"""Vector retriever — semantic search over ChromaDB with metadata pushdown.

Two responsibilities:
  1. Translate a QueryPlan's time range / subreddit facets into a Chroma
     `where` filter so filtering happens INSIDE the index (cheap), not after.
  2. Aggregate chunk-level hits back up to document level, because fusion with
     the graph retriever happens on `doc_id`. A document's score is the best
     (max) similarity among its chunks — standard max-pool over chunks.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from retrieval.models import QueryPlan, RetrievedHit, RetrieverSource, TimeRange
from utils.logging import get_logger
from vectorstore.chroma_loader import ChromaStore, get_chroma_store

logger = get_logger("retrieval.vector")


class VectorRetriever:
    def __init__(self, store: ChromaStore | None = None) -> None:
        self._store = store or get_chroma_store()

    def retrieve(
        self, plan: QueryPlan, *, time_range: Optional[TimeRange] = None, top_k: int = 25
    ) -> List[RetrievedHit]:
        where = self._build_where(plan, time_range)
        try:
            # Over-fetch at chunk level, then collapse to documents.
            chunk_hits = self._store.query(
                plan.normalized_query, top_k=top_k * 3, where=where
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Vector retrieval failed (%s); returning empty", exc)
            return []

        best_by_doc: Dict[str, Dict] = {}
        for ch in chunk_hits:
            doc_id = ch["doc_id"]
            if doc_id not in best_by_doc or ch["score"] > best_by_doc[doc_id]["score"]:
                best_by_doc[doc_id] = ch

        ranked = sorted(best_by_doc.values(), key=lambda h: h["score"], reverse=True)[:top_k]
        return [
            RetrievedHit(
                doc_id=h["doc_id"],
                text=h["text"],
                source=RetrieverSource.VECTOR,
                score=float(h["score"]),
                rank=rank,
                metadata=h["metadata"],
                explanation=f"semantic similarity {h['score']:.3f}",
            )
            for rank, h in enumerate(ranked, start=1)
        ]

    @staticmethod
    def _build_where(plan: QueryPlan, time_range: Optional[TimeRange]) -> Optional[Dict]:
        clauses: List[Dict] = []
        if time_range:
            clauses.append({"created_utc": {"$gte": time_range.start_utc}})
            clauses.append({"created_utc": {"$lt": time_range.end_utc}})
        if plan.subreddits:
            clauses.append({"subreddit": {"$in": plan.subreddits}})
        if not clauses:
            return None
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}

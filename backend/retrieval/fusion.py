"""Reciprocal Rank Fusion (RRF).

Why RRF (and not score normalisation / weighted-sum of raw scores)?
-------------------------------------------------------------------
Our two retrievers emit *incomparable* scores: the vector retriever returns
cosine similarities in roughly [0, 1], while the graph retriever returns an
unbounded, query-template-specific relevance score (counts, centrality,
recency boosts). Min-max or z-score normalising these is brittle — one
outlier rescales everything, and the distributions are not even the same
shape.

RRF sidesteps this entirely: it fuses on **rank**, not score. Each retriever
contributes `weight / (k + rank)` to a document's fused score. The constant
`k` (default 60, from Cormack et al. 2009) damps the influence of the very
top ranks so that a document appearing reasonably high in *both* lists beats
a document ranked #1 in only one. That "agreement across modalities" property
is exactly what makes the fused list better than either retriever alone — the
core claim the assignment asks us to demonstrate.

This module is intentionally pure-Python and side-effect free so it is fully
unit-testable without any external service.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

from retrieval.models import FusedHit, RetrievedHit


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[RetrievedHit]],
    *,
    source_names: Sequence[str] | None = None,
    weights: Sequence[float] | None = None,
    k: int = 60,
    top_n: int | None = None,
) -> List[FusedHit]:
    """Fuse multiple ranked lists of hits into one ranked list.

    Args:
        ranked_lists: one ordered list per retriever (rank 1 = best).
        source_names: human labels per list (default "list_0", "list_1", ...).
        weights: per-list multipliers (default all 1.0).
        k: RRF damping constant.
        top_n: truncate the fused output (None = return all).

    Returns:
        FusedHits sorted by descending fused score, each carrying the rank it
        held in every contributing retriever (provenance for transparency).
    """
    n = len(ranked_lists)
    if source_names is None:
        source_names = [f"list_{i}" for i in range(n)]
    if weights is None:
        weights = [1.0] * n
    if not (len(source_names) == len(weights) == n):
        raise ValueError("ranked_lists, source_names and weights must align")

    fused_score: Dict[str, float] = {}
    source_ranks: Dict[str, Dict[str, int]] = {}
    payload: Dict[str, RetrievedHit] = {}

    for hits, name, weight in zip(ranked_lists, source_names, weights):
        for rank, hit in enumerate(hits, start=1):
            contribution = weight * (1.0 / (k + rank))
            fused_score[hit.doc_id] = fused_score.get(hit.doc_id, 0.0) + contribution
            source_ranks.setdefault(hit.doc_id, {})[name] = rank
            # Prefer the richest text/metadata we have seen for this doc.
            if hit.doc_id not in payload or len(hit.text) > len(payload[hit.doc_id].text):
                payload[hit.doc_id] = hit

    fused = [
        FusedHit(
            doc_id=doc_id,
            text=payload[doc_id].text,
            fused_score=score,
            metadata=payload[doc_id].metadata,
            source_ranks=source_ranks[doc_id],
            explanation=_explain(source_ranks[doc_id]),
        )
        for doc_id, score in fused_score.items()
    ]
    fused.sort(key=lambda h: h.fused_score, reverse=True)
    return fused[:top_n] if top_n else fused


def _explain(ranks: Dict[str, int]) -> str:
    if len(ranks) > 1:
        parts = ", ".join(f"{src} #{r}" for src, r in sorted(ranks.items()))
        return f"corroborated by both retrievers ({parts})"
    src, r = next(iter(ranks.items()))
    return f"retrieved by {src} (rank #{r})"

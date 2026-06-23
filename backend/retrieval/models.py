"""Retrieval-layer data structures (kept dependency-free for testability)."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class RetrieverSource(str, Enum):
    GRAPH = "graph"
    VECTOR = "vector"
    FUSED = "fused"


@dataclass
class RetrievedHit:
    """A single retrieved unit of content, source-agnostic.

    `score` semantics differ per retriever (cosine similarity for vector,
    a graph-relevance score for graph) — RRF deliberately ignores raw scores
    and fuses on *rank*, so cross-retriever score incomparability is a
    non-issue.
    """
    doc_id: str
    text: str
    source: RetrieverSource
    score: float = 0.0
    rank: int = 0
    metadata: Dict = field(default_factory=dict)
    explanation: str = ""  # why this hit was retrieved (for transparency)


@dataclass
class FusedHit:
    """A hit after Reciprocal Rank Fusion, carrying full provenance."""
    doc_id: str
    text: str
    fused_score: float
    metadata: Dict = field(default_factory=dict)
    # rank this doc held in each contributing retriever (1-indexed)
    source_ranks: Dict[str, int] = field(default_factory=dict)
    explanation: str = ""


@dataclass
class TimeRange:
    """A named, half-open [start, end) interval in UTC."""
    label: str
    start: dt.datetime
    end: dt.datetime

    @property
    def start_utc(self) -> float:
        return self.start.timestamp()

    @property
    def end_utc(self) -> float:
        return self.end.timestamp()

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"TimeRange({self.label}: "
            f"{self.start.date()} → {self.end.date()})"
        )


@dataclass
class QueryPlan:
    """Output of query understanding — drives how retrieval executes."""
    raw_query: str
    normalized_query: str
    intent: str = "hybrid"          # semantic | relational | hybrid | temporal
    topics: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    subreddits: List[str] = field(default_factory=list)
    time_ranges: List[TimeRange] = field(default_factory=list)
    is_comparison: bool = False
    graph_weight: float = 1.0
    vector_weight: float = 1.0

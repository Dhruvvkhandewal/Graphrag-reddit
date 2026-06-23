"""Pydantic request/response schemas for the HTTP API."""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    subreddits: Optional[List[str]] = Field(None, description="Override configured subreddits")
    post_limit: Optional[int] = Field(None, description="Posts per subreddit")
    reset: bool = Field(False, description="Wipe the graph before loading")


class IngestResponse(BaseModel):
    status: str
    detail: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, examples=["How has sentiment toward RAG changed in the last 6 months?"])
    include_retrieval_detail: bool = Field(
        True, description="Return graph-only / vector-only / fused breakdowns"
    )


class CitationModel(BaseModel):
    number: int
    doc_id: str
    url: str
    subreddit: str
    author: str
    created_at: str
    snippet: str


class HitModel(BaseModel):
    doc_id: str
    score: float
    rank: int
    subreddit: Optional[str] = None
    author: Optional[str] = None
    time_window: Optional[str] = None
    snippet: str
    explanation: str = ""


class FusedHitModel(BaseModel):
    doc_id: str
    fused_score: float
    source_ranks: Dict[str, int]
    subreddit: Optional[str] = None
    time_window: Optional[str] = None
    snippet: str
    explanation: str = ""


class RetrievalDetail(BaseModel):
    graph_only: List[HitModel] = []
    vector_only: List[HitModel] = []
    fused: List[FusedHitModel] = []


class QueryResponse(BaseModel):
    question: str
    answer: str
    intent: str
    is_comparison: bool
    time_windows: List[str]
    citations: List[CitationModel]
    retrieval: Optional[RetrievalDetail] = None


class StatsResponse(BaseModel):
    vector_chunks: int
    graph_nodes: Dict[str, int]
    time_windows: List[str]


class HealthResponse(BaseModel):
    status: str
    llm_enabled: bool
    neo4j: str
    chroma: str

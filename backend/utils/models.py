"""Canonical domain models shared across ingestion, graph, vector, retrieval.

These are deliberately plain dataclasses (no heavy deps) so they can be
imported anywhere — including in unit tests — without pulling in torch,
neo4j, or chromadb.

`doc_id` is the join key that ties the graph and vector representations of
the *same* underlying content together. We reuse Reddit "fullnames"
(`t3_<id>` for posts, `t1_<id>` for comments) because they are globally
unique and stable, which makes Reciprocal Rank Fusion across the two stores
trivially correct.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class ContentType(str, Enum):
    POST = "post"
    COMMENT = "comment"


class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    MIXED = "mixed"


@dataclass
class Entity:
    """A named thing the content is *about* (a model, company, person, tool)."""
    name: str
    type: str  # e.g. MODEL, ORG, PERSON, PRODUCT, CONCEPT


@dataclass
class Enrichment:
    """LLM-derived semantic layer attached to a document."""
    topics: List[str] = field(default_factory=list)
    entities: List[Entity] = field(default_factory=list)
    sentiment: Sentiment = Sentiment.NEUTRAL
    sentiment_score: float = 0.0  # signed, [-1, 1]


@dataclass
class RedditDocument:
    """One post or comment, normalised. The atomic unit of the whole system."""
    doc_id: str                      # reddit fullname, e.g. "t3_abc" / "t1_xyz"
    type: ContentType
    subreddit: str
    author: str
    title: str                       # "" for comments
    body: str
    created_utc: float               # epoch seconds
    score: int
    permalink: str
    parent_id: Optional[str] = None  # immediate parent fullname (comments)
    root_post_id: Optional[str] = None  # the t3_ post this belongs to
    edited_utc: Optional[float] = None

    # Temporal bucketing (filled by TemporalProcessor)
    time_window: str = ""            # quarter bucket, e.g. "2026-Q1"
    month: str = ""                  # "2026-01"
    year: int = 0

    # Semantic layer (filled by enrichment)
    enrichment: Optional[Enrichment] = None

    # ---- convenience ----
    @property
    def created_dt(self) -> dt.datetime:
        return dt.datetime.fromtimestamp(self.created_utc, tz=dt.timezone.utc)

    @property
    def text(self) -> str:
        return f"{self.title}\n\n{self.body}".strip() if self.title else self.body

    @property
    def url(self) -> str:
        return f"https://www.reddit.com{self.permalink}"

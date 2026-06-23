# Architecture — Low-Level Design

This document complements the README with the class diagram, a component
responsibility map, and the key design contracts.

## Class diagram (core types)

```mermaid
classDiagram
    class RedditDocument {
      +str doc_id
      +ContentType type
      +str subreddit
      +str author
      +str title
      +str body
      +float created_utc
      +str time_window
      +Enrichment enrichment
      +text() str
      +url() str
    }
    class Enrichment {
      +list~str~ topics
      +list~Entity~ entities
      +Sentiment sentiment
      +float sentiment_score
    }
    class RetrievedHit {
      +str doc_id
      +str text
      +RetrieverSource source
      +float score
      +int rank
      +dict metadata
    }
    class FusedHit {
      +str doc_id
      +float fused_score
      +dict source_ranks
    }
    class QueryPlan {
      +str intent
      +list~str~ topics
      +list~TimeRange~ time_ranges
      +bool is_comparison
      +float graph_weight
      +float vector_weight
    }

    RedditDocument --> Enrichment
    Enrichment --> Entity
    QueryPlan --> TimeRange

    class IngestionPipeline
    class RedditScraper
    class TemporalProcessor
    class ContentEnricher
    class GraphLoader
    class ChromaStore
    IngestionPipeline --> RedditScraper
    IngestionPipeline --> TemporalProcessor
    IngestionPipeline --> ContentEnricher
    IngestionPipeline --> GraphLoader
    IngestionPipeline --> ChromaStore

    class HybridRetriever
    class GraphRetriever
    class VectorRetriever
    class QueryUnderstanding
    class QueryEngine
    class AnswerGenerator
    QueryEngine --> HybridRetriever
    QueryEngine --> AnswerGenerator
    HybridRetriever --> QueryUnderstanding
    HybridRetriever --> GraphRetriever
    HybridRetriever --> VectorRetriever
    HybridRetriever --> FusedHit
    GraphRetriever --> RetrievedHit
    VectorRetriever --> RetrievedHit
```

## Component responsibilities

| Component | File | Responsibility |
|-----------|------|----------------|
| `Settings` | `config/settings.py` | Single source of truth for all config. |
| `RedditScraper` | `ingestion/reddit_scraper.py` | PRAW scraping; posts + depth-capped comments; ≥3 windows via top(year)+new. |
| `TemporalProcessor` | `ingestion/temporal_processor.py` | Quarter/month bucketing; horizon cut-off. |
| `ContentEnricher` | `llm/extractors.py` | One Gemini call → entities/topics/sentiment (heuristic fallback). |
| `GraphLoader` | `graph/graph_loader.py` | UNWIND-batched MERGE of nodes + temporal edges. |
| `ChromaStore` | `vectorstore/chroma_loader.py` | Chunk → embed → upsert; metadata-filtered query. |
| `QueryUnderstanding` | `llm/query_understanding.py` | Intent/weights (LLM) + time ranges (deterministic). |
| `GraphRetriever` | `retrieval/graph_retriever.py` | Influence / community / topic-entity Cypher strategies. |
| `VectorRetriever` | `retrieval/vector_retriever.py` | Semantic search + chunk→doc max-pool. |
| `reciprocal_rank_fusion` | `retrieval/fusion.py` | Rank-based fusion with provenance. |
| `HybridRetriever` | `retrieval/hybrid_retriever.py` | Concurrent retrieval + fusion + per-period split. |
| `AnswerGenerator` | `llm/answer_generator.py` | Cited synthesis; grouped comparison rendering. |
| `QueryEngine` | `app/engine.py` | Application entry: retrieve → answer → package. |

## Key contracts

1. **Join key.** Graph node ids and vector `doc_id` metadata are identical
   Reddit fullnames. RRF fuses on this key; never on chunk ids.
2. **Score opacity.** RRF ignores raw scores → graph relevance and cosine
   similarity never need to be on the same scale.
3. **Temporal symmetry.** The same `[start, end)` interval is applied to graph
   (`created_utc` predicate) and vector (`where` filter) so the two stores
   always see the same time slice.
4. **Graceful degradation.** Every external dependency (Gemini, Neo4j, Chroma,
   Reddit) is lazily imported and failure-isolated; retrieval never hard-crashes
   on a single retriever failing.

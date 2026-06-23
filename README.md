# Hybrid GraphRAG System for Time-Series Reddit Intelligence

A production-oriented Hybrid GraphRAG system that combines Knowledge Graph Retrieval and Vector Retrieval to analyze Reddit discussions across multiple time windows.

## Features

* Reddit post and nested comment ingestion using PRAW
* Temporal Knowledge Graph built on Neo4j
* Semantic Vector Search using ChromaDB
* Hybrid Retrieval (Graph + Vector)
* Reciprocal Rank Fusion (RRF)
* Gemini 2.5 Flash powered:

  * Entity Extraction
  * Topic Extraction
  * Sentiment Analysis
  * Query Understanding
  * Answer Generation
* Temporal reasoning for trend and comparison queries
* Source-aware answer generation with citations
* FastAPI REST API with Swagger documentation

## Tech Stack

* Python 3.11
* FastAPI
* Neo4j
* ChromaDB
* Gemini 2.5 Flash
* Sentence Transformers (BAAI/bge-small-en-v1.5)
* PRAW
* LangChain (minimal usage)

## API Documentation

After starting the server:

```bash
python -m uvicorn app.main:app --reload
```

Swagger UI:

http://127.0.0.1:8000/docs

Interactive API documentation is available through FastAPI Swagger UI for testing ingestion, retrieval, temporal analysis, and GraphRAG query endpoints.

## Example Queries

* What are the most discussed AI agent frameworks this month?
* How has sentiment around AI agents changed over the last 6 months?
* Which users are most influential in discussions about GraphRAG?
* Compare discussions about LangGraph between Q1 and Q2.

## Architecture

Reddit → Processing → Gemini Enrichment → Neo4j Graph → ChromaDB → Hybrid Retrieval → RRF → Answer Generation

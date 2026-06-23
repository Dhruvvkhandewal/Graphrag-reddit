"""Shared singletons exposed for FastAPI dependency injection / testing.

The engine, retrievers, and stores are already process-singletons via their
`get_*()` accessors; re-exporting them here gives a single import surface and
a seam for overriding in tests.
"""
from __future__ import annotations

from app.engine import get_query_engine
from graph.neo4j_client import get_neo4j_client
from llm.gemini_client import get_gemini_client
from vectorstore.chroma_loader import get_chroma_store

__all__ = [
    "get_query_engine",
    "get_neo4j_client",
    "get_gemini_client",
    "get_chroma_store",
]

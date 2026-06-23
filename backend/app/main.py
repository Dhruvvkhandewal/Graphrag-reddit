"""FastAPI application factory.

    uvicorn app.main:app --reload
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.routes import router
from config import get_settings
from utils.logging import configure_logging, get_logger

settings = get_settings()
configure_logging(settings.log_level)
logger = get_logger("app.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Hybrid GraphRAG API (llm_enabled=%s)", settings.llm_enabled)
    yield
    try:
        from graph.neo4j_client import get_neo4j_client

        get_neo4j_client().close()
    except Exception:  # noqa: BLE001
        pass
    logger.info("Shutdown complete")


app = FastAPI(
    title="Hybrid GraphRAG for Time-Series Reddit Intelligence",
    description=(
        "Fuses a Neo4j temporal knowledge graph with a ChromaDB vector index "
        "via Reciprocal Rank Fusion to answer time-aware questions over Reddit."
    ),
    version="1.0.0",
    lifespan=lifespan,
)
app.include_router(router)


@app.get("/")
def root():
    return {
        "service": "hybrid-graphrag-reddit",
        "docs": "/docs",
        "endpoints": ["/health", "/ingest", "/query", "/stats"],
    }

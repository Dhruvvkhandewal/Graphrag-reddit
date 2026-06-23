"""HTTP routes: /health, /ingest, /query, /stats.

Routes stay thin — all real work lives in the engine / pipeline. Mapping from
internal dataclasses to API models happens here so the domain layer never
depends on FastAPI.
"""
from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, HTTPException

from api.models import (
    CitationModel, FusedHitModel, HealthResponse, HitModel, IngestRequest,
    IngestResponse, QueryRequest, QueryResponse, RetrievalDetail, StatsResponse,
)
from app.engine import get_query_engine
from config import get_settings
from utils.logging import get_logger

logger = get_logger("api.routes")
router = APIRouter()


def _snippet(text: str, n: int = 220) -> str:
    text = (text or "").strip().replace("\n", " ")
    return text[:n] + (" …" if len(text) > n else "")


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()
    neo4j_status = "unknown"
    chroma_status = "unknown"
    try:
        from graph.neo4j_client import get_neo4j_client

        get_neo4j_client().query("RETURN 1 AS ok")
        neo4j_status = "ok"
    except Exception as exc:  # noqa: BLE001
        neo4j_status = f"unavailable: {type(exc).__name__}"
    try:
        from vectorstore.chroma_loader import get_chroma_store

        get_chroma_store().count()
        chroma_status = "ok"
    except Exception as exc:  # noqa: BLE001
        chroma_status = f"unavailable: {type(exc).__name__}"
    return HealthResponse(
        status="ok",
        llm_enabled=settings.llm_enabled,
        neo4j=neo4j_status,
        chroma=chroma_status,
    )


@router.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest, background: BackgroundTasks) -> IngestResponse:
    from ingestion.pipeline import IngestionPipeline

    def _run():
        try:
            IngestionPipeline().run(req.subreddits, post_limit=req.post_limit, reset=req.reset)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Background ingestion failed: %s", exc)

    background.add_task(_run)
    return IngestResponse(
        status="accepted",
        detail="Ingestion started in the background. Poll /stats for progress.",
    )


@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    try:
        result = get_query_engine().answer(req.question)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(exc))

    detail = None
    if req.include_retrieval_detail and result.retrieval:
        r = result.retrieval
        detail = RetrievalDetail(
            graph_only=[_hit(h) for h in r.graph_hits[:10]],
            vector_only=[_hit(h) for h in r.vector_hits[:10]],
            fused=[_fused(h) for h in r.fused[:10]],
        )

    return QueryResponse(
        question=result.question,
        answer=result.answer,
        intent=result.intent,
        is_comparison=result.is_comparison,
        time_windows=result.time_windows,
        citations=[CitationModel(**c.__dict__) for c in result.citations],
        retrieval=detail,
    )


@router.get("/stats", response_model=StatsResponse)
def stats() -> StatsResponse:
    from graph.neo4j_client import get_neo4j_client
    from vectorstore.chroma_loader import get_chroma_store

    counts = {"Post": 0, "Comment": 0, "User": 0, "Topic": 0, "Entity": 0, "Subreddit": 0}
    windows: list[str] = []
    try:
        client = get_neo4j_client()
        for label in list(counts):
            rows = client.query(f"MATCH (n:{label}) RETURN count(n) AS c")
            counts[label] = rows[0]["c"] if rows else 0
        wrows = client.query("MATCH (w:TimeWindow) RETURN w.id AS id ORDER BY id")
        windows = [r["id"] for r in wrows]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Graph stats unavailable: %s", exc)
    try:
        chunks = get_chroma_store().count()
    except Exception:  # noqa: BLE001
        chunks = 0
    return StatsResponse(vector_chunks=chunks, graph_nodes=counts, time_windows=windows)


# ---- mappers ----------------------------------------------------------
def _hit(h) -> HitModel:
    md = h.metadata or {}
    return HitModel(
        doc_id=h.doc_id,
        score=round(h.score, 4),
        rank=h.rank,
        subreddit=md.get("subreddit"),
        author=md.get("author"),
        time_window=md.get("time_window"),
        snippet=_snippet(h.text),
        explanation=h.explanation,
    )


def _fused(h) -> FusedHitModel:
    md = h.metadata or {}
    return FusedHitModel(
        doc_id=h.doc_id,
        fused_score=round(h.fused_score, 6),
        source_ranks=h.source_ranks,
        subreddit=md.get("subreddit"),
        time_window=md.get("time_window"),
        snippet=_snippet(h.text),
        explanation=h.explanation,
    )

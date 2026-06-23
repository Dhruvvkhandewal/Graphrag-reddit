"""End-to-end ingestion orchestrator.

    Reddit (PRAW)  ->  Temporal bucketing  ->  LLM enrichment
                                                   |
                                  +----------------+----------------+
                                  v                                 v
                         Neo4j (temporal graph)            ChromaDB (vectors)

Both stores receive the SAME enriched documents, keyed by `doc_id`, so the two
representations stay in lock-step and RRF can fuse across them.

Run as a module:  `python -m ingestion.pipeline [--limit N] [--reset]`
"""
from __future__ import annotations

import argparse
import time
from typing import List

from config import get_settings
from graph.graph_loader import GraphLoader
from graph.neo4j_client import get_neo4j_client
from graph.schema import apply_schema, reset_graph
from ingestion.reddit_scraper import RedditScraper
from ingestion.temporal_processor import TemporalProcessor
from llm.extractors import ContentEnricher
from utils.logging import configure_logging, get_logger
from utils.models import RedditDocument
from vectorstore.chroma_loader import get_chroma_store

logger = get_logger("ingestion.pipeline")


class IngestionPipeline:
    def __init__(self) -> None:
        self._settings = get_settings()
        self._scraper = RedditScraper()
        self._temporal = TemporalProcessor()
        self._enricher = ContentEnricher()
        self._neo4j = get_neo4j_client()
        self._graph_loader = GraphLoader(self._neo4j)
        self._chroma = get_chroma_store()

    def run(
        self,
        subreddits: List[str] | None = None,
        *,
        post_limit: int | None = None,
        reset: bool = False,
    ) -> dict:
        t0 = time.time()
        apply_schema(self._neo4j)
        if reset:
            reset_graph(self._neo4j)

        # 1) scrape + temporal bucket
        raw = self._scraper.scrape(subreddits, post_limit=post_limit)
        docs: List[RedditDocument] = list(self._temporal.process(raw))
        logger.info("Scraped + bucketed %d documents", len(docs))

        # 2) enrich (entities, topics, sentiment) — combined single call each
        self._enrich(docs)

        # 3) load both stores
        graph_stats = self._graph_loader.load(docs)
        n_chunks = self._chroma.upsert_documents(docs)

        windows = sorted({d.time_window for d in docs})
        stats = {
            "documents": len(docs),
            "posts": graph_stats["posts"],
            "comments": graph_stats["comments"],
            "vector_chunks": n_chunks,
            "time_windows": windows,
            "seconds": round(time.time() - t0, 1),
        }
        logger.info("Ingestion finished: %s", stats)
        return stats

    def _enrich(self, docs: List[RedditDocument]) -> None:
        total = len(docs)
        for i, doc in enumerate(docs, start=1):
            doc.enrichment = self._enricher.enrich(text=doc.text, subreddit=doc.subreddit)
            if i % 25 == 0 or i == total:
                logger.info("Enriched %d/%d", i, total)


def main():
    parser = argparse.ArgumentParser(description="Reddit GraphRAG ingestion")
    parser.add_argument("--subreddits", type=str, default="", help="comma-separated override")
    parser.add_argument("--limit", type=int, default=None, help="posts per subreddit")
    parser.add_argument("--reset", action="store_true", help="wipe graph before load")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level)

    subs = [s.strip() for s in args.subreddits.split(",") if s.strip()] or None
    pipeline = IngestionPipeline()
    stats = pipeline.run(subs, post_limit=args.limit, reset=args.reset)
    print("\n=== INGESTION COMPLETE ===")
    for k, v in stats.items():
        print(f"  {k:14}: {v}")


if __name__ == "__main__":
    main()

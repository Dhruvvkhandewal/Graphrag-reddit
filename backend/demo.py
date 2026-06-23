"""Demo: runs the four required query archetypes end-to-end.

For EACH query it prints:
    * graph-only results
    * vector-only results
    * fused (RRF) results
    * the final LLM-synthesised answer with citations

Usage:
    python demo.py                 # assumes data already ingested
    python demo.py --ingest        # ingest first (needs Reddit + Gemini keys)
    python demo.py --ingest --reset
"""
from __future__ import annotations

import argparse
import textwrap

from app.engine import get_query_engine
from config import get_settings
from utils.logging import configure_logging

# (label, why-it-belongs-here, question)
DEMO_QUERIES = [
    (
        "1. VECTOR-DOMINANT (purely semantic)",
        "Fuzzy 'what are people saying' question — semantics win.",
        "What are people saying about running large language models locally on consumer GPUs?",
    ),
    (
        "2. GRAPH-DOMINANT (relationship / traversal)",
        "Asks for influential authors — needs authorship traversal + centrality.",
        "Who are the most influential voices in discussions about open-source LLMs?",
    ),
    (
        "3. HYBRID (needs both)",
        "Topic semantics + community structure — neither retriever suffices alone.",
        "Which communities are leading the conversation on AI safety, and what concerns are they raising?",
    ),
    (
        "4. TIME-SERIES COMPARISON",
        "Two explicit periods — temporal routing + side-by-side synthesis.",
        "What emerging concerns about AI safety appeared in Q1 2026 that weren't discussed in Q4 2025?",
    ),
]

BAR = "=" * 88
SUB = "-" * 88


def _print_hits(title, hits, score_attr):
    print(f"\n  {title}")
    if not hits:
        print("    (none)")
        return
    for h in hits[:5]:
        md = getattr(h, "metadata", {}) or {}
        score = getattr(h, score_attr)
        sub = md.get("subreddit", "?")
        win = md.get("time_window", "?")
        snippet = (h.text or "").strip().replace("\n", " ")[:90]
        print(f"    [{score:7.4f}] r/{sub:<16} {win:<8} {h.doc_id:<12} {snippet}…")


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ingest", action="store_true")
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level)

    if args.ingest:
        from ingestion.pipeline import IngestionPipeline

        print(f"\n{BAR}\nINGESTING (this can take a few minutes)…\n{BAR}")
        stats = IngestionPipeline().run(reset=args.reset)
        print("Ingestion stats:", stats)

    engine = get_query_engine()

    for label, why, question in DEMO_QUERIES:
        print(f"\n{BAR}\n{label}\n{BAR}")
        print(textwrap.fill(f"Q: {question}", width=88))
        print(f"   ({why})")

        result = engine.answer(question)
        r = result.retrieval

        print(f"\n  Router  -> intent={result.intent} | comparison={result.is_comparison} "
              f"| windows={result.time_windows}")

        if r and r.period_results:
            for p in r.period_results:
                print(f"\n  {SUB}\n  PERIOD: {p.label}\n  {SUB}")
                _print_hits("GRAPH-ONLY", p.graph_hits, "score")
                _print_hits("VECTOR-ONLY", p.vector_hits, "score")
                _print_hits("FUSED (RRF)", p.fused, "fused_score")
        elif r:
            _print_hits("GRAPH-ONLY results", r.graph_hits, "score")
            _print_hits("VECTOR-ONLY results", r.vector_hits, "score")
            _print_hits("FUSED (RRF) results", r.fused, "fused_score")

        print(f"\n  {SUB}\n  FINAL ANSWER\n  {SUB}")
        print(textwrap.indent(textwrap.fill(result.answer, width=84), "  "))
        if result.citations:
            print("\n  Sources:")
            for c in result.citations[:6]:
                print(f"    [{c.number}] r/{c.subreddit} u/{c.author} {c.created_at[:10]} {c.url}")

    print(f"\n{BAR}\nDEMO COMPLETE\n{BAR}")


if __name__ == "__main__":
    run()

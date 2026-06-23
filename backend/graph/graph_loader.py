"""Batched loader that materialises RedditDocuments into the temporal graph.

Performance note: we never MERGE node-by-node in a Python loop. Instead we
build parameter lists and use a single `UNWIND $rows AS row ...` Cypher per
entity type. UNWIND batching turns thousands of round-trips into a handful of
transactions — the standard high-throughput Neo4j ingestion pattern.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

from graph.neo4j_client import Neo4jClient
from utils.logging import get_logger
from utils.models import ContentType, RedditDocument

logger = get_logger("graph.loader")

_BATCH = 500

# --- node/edge MERGE templates (UNWIND-batched) ---------------------------

_MERGE_POSTS = """
UNWIND $rows AS row
MERGE (p:Post {id: row.id})
  SET p += row.props
MERGE (u:User {name: row.author})
MERGE (u)-[a:AUTHORED]->(p) SET a.created_utc = row.props.created_utc
MERGE (s:Subreddit {name: row.subreddit})
MERGE (p)-[:POSTED_IN]->(s)
MERGE (w:TimeWindow {id: row.props.time_window})
MERGE (p)-[:IN_WINDOW]->(w)
"""

_MERGE_COMMENTS = """
UNWIND $rows AS row
MERGE (c:Comment {id: row.id})
  SET c += row.props
MERGE (u:User {name: row.author})
MERGE (u)-[a:AUTHORED]->(c) SET a.created_utc = row.props.created_utc
MERGE (s:Subreddit {name: row.subreddit})
MERGE (c)-[:POSTED_IN]->(s)
MERGE (w:TimeWindow {id: row.props.time_window})
MERGE (c)-[:IN_WINDOW]->(w)
WITH c, row
MATCH (root:Post {id: row.root_post_id})
MERGE (c)-[co:COMMENTED_ON]->(root) SET co.created_utc = row.props.created_utc
"""

# REPLY_TO links a comment to its *immediate* parent (post or comment).
_MERGE_REPLIES = """
UNWIND $rows AS row
MATCH (c:Comment {id: row.id})
MATCH (parent) WHERE parent.id = row.parent_id
MERGE (c)-[r:REPLY_TO]->(parent) SET r.created_utc = row.created_utc
"""

_MERGE_TOPICS = """
UNWIND $rows AS row
MATCH (n) WHERE n.id = row.doc_id
UNWIND row.topics AS topic
MERGE (t:Topic {name: topic})
MERGE (n)-[:ABOUT_TOPIC]->(t)
"""

_MERGE_ENTITIES = """
UNWIND $rows AS row
MATCH (n) WHERE n.id = row.doc_id
UNWIND row.entities AS ent
MERGE (e:Entity {name: ent.name})
  ON CREATE SET e.type = ent.type
MERGE (n)-[:MENTIONS]->(e)
"""


def _props(doc: RedditDocument) -> Dict:
    enr = doc.enrichment
    return {
        "id": doc.doc_id,
        "title": doc.title,
        "body": doc.body,
        "score": doc.score,
        "created_utc": doc.created_utc,
        "created_at": doc.created_dt.isoformat(),
        "time_window": doc.time_window,
        "month": doc.month,
        "year": doc.year,
        "url": doc.url,
        "sentiment": enr.sentiment.value if enr else "neutral",
        "sentiment_score": enr.sentiment_score if enr else 0.0,
    }


class GraphLoader:
    def __init__(self, client: Neo4jClient) -> None:
        self._client = client

    def load(self, docs: Sequence[RedditDocument]) -> Dict[str, int]:
        posts = [d for d in docs if d.type == ContentType.POST]
        comments = [d for d in docs if d.type == ContentType.COMMENT]

        self._load_posts(posts)
        self._load_comments(comments)       # creates COMMENTED_ON to root post
        self._load_replies(comments)        # immediate-parent REPLY_TO
        self._load_semantics(posts + comments)

        stats = {"posts": len(posts), "comments": len(comments)}
        logger.info("Graph load complete: %s", stats)
        return stats

    # ------------------------------------------------------------------
    def _load_posts(self, posts):
        rows = [
            {
                "id": d.doc_id,
                "author": d.author,
                "subreddit": d.subreddit,
                "props": _props(d),
            }
            for d in posts
        ]
        self._run_batched(_MERGE_POSTS, rows)

    def _load_comments(self, comments):
        rows = [
            {
                "id": d.doc_id,
                "author": d.author,
                "subreddit": d.subreddit,
                "root_post_id": d.root_post_id,
                "props": _props(d),
            }
            for d in comments
        ]
        self._run_batched(_MERGE_COMMENTS, rows)

    def _load_replies(self, comments):
        rows = [
            {"id": d.doc_id, "parent_id": d.parent_id, "created_utc": d.created_utc}
            for d in comments
            if d.parent_id
        ]
        self._run_batched(_MERGE_REPLIES, rows)

    def _load_semantics(self, docs):
        topic_rows, entity_rows = [], []
        for d in docs:
            if not d.enrichment:
                continue
            if d.enrichment.topics:
                topic_rows.append({"doc_id": d.doc_id, "topics": d.enrichment.topics})
            if d.enrichment.entities:
                entity_rows.append(
                    {
                        "doc_id": d.doc_id,
                        "entities": [
                            {"name": e.name, "type": e.type}
                            for e in d.enrichment.entities
                        ],
                    }
                )
        self._run_batched(_MERGE_TOPICS, topic_rows)
        self._run_batched(_MERGE_ENTITIES, entity_rows)

    def _run_batched(self, cypher: str, rows: List[Dict]):
        for i in range(0, len(rows), _BATCH):
            self._client.execute_write(cypher, {"rows": rows[i : i + _BATCH]})

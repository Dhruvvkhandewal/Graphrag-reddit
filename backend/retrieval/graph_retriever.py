"""Graph retriever — turns a QueryPlan into temporal Cypher traversals.

This is where the graph earns its keep. Rather than a single lookup, it picks
one of several traversal *strategies* based on the query intent inferred by
the LLM router, and every strategy respects the parsed time range(s):

  * INFLUENCE  ("who are the most influential voices ...")
        (User)-[:AUTHORED]->(content)-[:ABOUT_TOPIC]->(Topic)
        ranked by authored-volume + aggregate score + inbound replies.
  * COMMUNITY  ("which communities are leading ...")
        (Subreddit)<-[:POSTED_IN]-(content)-[:ABOUT_TOPIC]->(Topic)
        ranked by content volume + engagement per subreddit.
  * TOPIC/ENTITY (default relational)
        content matched via ABOUT_TOPIC / MENTIONS within the window,
        ranked by score + recency.

Each strategy returns content nodes (posts/comments) as `RetrievedHit`s so the
output is fusable with the vector retriever on `doc_id`. Influence/community
strategies *also* surface their aggregate findings via `explanation`, which
the answer generator can use.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from graph.neo4j_client import Neo4jClient, get_neo4j_client
from retrieval.models import QueryPlan, RetrievedHit, RetrieverSource, TimeRange
from utils.logging import get_logger
from utils.text import truncate

logger = get_logger("retrieval.graph")


def _time_clause(var: str, time_range: Optional[TimeRange]) -> str:
    if not time_range:
        return ""
    return f" AND {var}.created_utc >= $start AND {var}.created_utc < $end"


class GraphRetriever:
    def __init__(self, client: Neo4jClient | None = None) -> None:
        self._client = client or get_neo4j_client()

    def retrieve(
        self, plan: QueryPlan, *, time_range: Optional[TimeRange] = None, top_k: int = 25
    ) -> List[RetrievedHit]:
        intent = plan.intent
        try:
            if intent == "relational" and self._is_influence(plan):
                return self._influence(plan, time_range, top_k)
            if intent == "relational" and self._is_community(plan):
                return self._community(plan, time_range, top_k)
            return self._topic_entity(plan, time_range, top_k)
        except Exception as exc:  # noqa: BLE001 — retrieval must never hard-crash
            logger.warning("Graph retrieval failed (%s); returning empty", exc)
            return []

    # ---- intent sniffers ---------------------------------------------
    @staticmethod
    def _is_influence(plan: QueryPlan) -> bool:
        q = plan.raw_query.lower()
        return any(w in q for w in ("influential", "voices", "who ", "whom", "leading voices"))

    @staticmethod
    def _is_community(plan: QueryPlan) -> bool:
        q = plan.raw_query.lower()
        return any(w in q for w in ("communit", "subreddit", "which sub"))

    # ---- strategies ---------------------------------------------------
    def _topic_entity(self, plan, time_range, top_k) -> List[RetrievedHit]:
        params: Dict = {
            "topics": [t.lower() for t in plan.topics],
            "entities": [e for e in plan.entities],
            "limit": top_k,
        }
        if time_range:
            params.update(start=time_range.start_utc, end=time_range.end_utc)
        cypher = f"""
        MATCH (n)
        WHERE (n:Post OR n:Comment){_time_clause('n', time_range)}
        OPTIONAL MATCH (n)-[:ABOUT_TOPIC]->(t:Topic)
        OPTIONAL MATCH (n)-[:MENTIONS]->(e:Entity)
        WITH n,
             collect(DISTINCT t.name) AS topics,
             collect(DISTINCT e.name) AS entities
        WITH n, topics, entities,
             size([x IN topics WHERE x IN $topics]) AS topic_hits,
             size([x IN entities WHERE toLower(x) IN [y IN $entities | toLower(y)]]) AS entity_hits
        WHERE topic_hits > 0 OR entity_hits > 0 OR size($topics) = 0
        WITH n, topics, entities, topic_hits, entity_hits,
             (3.0*topic_hits + 2.0*entity_hits) + log(1 + abs(coalesce(n.score,0))) AS relevance
        RETURN n.id AS doc_id,
               coalesce(n.title,'') AS title,
               coalesce(n.body,'')  AS body,
               n.subreddit AS subreddit, n.author AS author,
               n.created_utc AS created_utc, n.time_window AS time_window,
               n.sentiment AS sentiment, n.url AS url, labels(n)[0] AS type,
               topics, relevance
        ORDER BY relevance DESC, n.created_utc DESC
        LIMIT $limit
        """
        rows = self._client.query(cypher, params)
        return [self._row_to_hit(r, rank) for rank, r in enumerate(rows, start=1)]

    def _influence(self, plan, time_range, top_k) -> List[RetrievedHit]:
        params: Dict = {"topics": [t.lower() for t in plan.topics], "limit": top_k}
        if time_range:
            params.update(start=time_range.start_utc, end=time_range.end_utc)
        # Rank authors by volume + engagement + inbound replies on a topic,
        # then return their single most-engaged piece of content as the hit.
        cypher = f"""
        MATCH (u:User)-[:AUTHORED]->(n)
        WHERE (n:Post OR n:Comment){_time_clause('n', time_range)}
        OPTIONAL MATCH (n)-[:ABOUT_TOPIC]->(t:Topic)
        WITH u, n, collect(DISTINCT t.name) AS topics
        WHERE size($topics) = 0 OR any(x IN topics WHERE x IN $topics)
        OPTIONAL MATCH (reply:Comment)-[:REPLY_TO]->(n)
        WITH u, n, topics, count(reply) AS inbound
        WITH u,
             count(n) AS contributions,
             sum(coalesce(n.score,0)) AS total_score,
             sum(inbound) AS total_replies,
             collect({{id:n.id, title:coalesce(n.title,''), body:coalesce(n.body,''),
                      subreddit:n.subreddit, author:n.author, created_utc:n.created_utc,
                      time_window:n.time_window, sentiment:n.sentiment, url:n.url,
                      type:labels(n)[0], score:coalesce(n.score,0), topics:topics}}) AS items
        WITH u, contributions, total_score, total_replies, items,
             (2.0*contributions + 0.5*total_score + 1.5*total_replies) AS influence
        ORDER BY influence DESC
        LIMIT $limit
        WITH u, influence, contributions, total_replies,
             head([it IN items WHERE it.score = reduce(m=-2147483648, x IN items | CASE WHEN x.score>m THEN x.score ELSE m END) | it]) AS top_item
        RETURN top_item.id AS doc_id, top_item.title AS title, top_item.body AS body,
               top_item.subreddit AS subreddit, u.name AS author,
               top_item.created_utc AS created_utc, top_item.time_window AS time_window,
               top_item.sentiment AS sentiment, top_item.url AS url, top_item.type AS type,
               top_item.topics AS topics, influence AS relevance,
               contributions AS contributions, total_replies AS total_replies
        """
        rows = self._client.query(cypher, params)
        hits = []
        for rank, r in enumerate(rows, start=1):
            hit = self._row_to_hit(r, rank)
            hit.explanation = (
                f"influential author u/{r.get('author')} "
                f"({r.get('contributions')} posts, {r.get('total_replies')} replies received)"
            )
            hits.append(hit)
        return hits

    def _community(self, plan, time_range, top_k) -> List[RetrievedHit]:
        params: Dict = {"topics": [t.lower() for t in plan.topics], "limit": top_k}
        if time_range:
            params.update(start=time_range.start_utc, end=time_range.end_utc)
        cypher = f"""
        MATCH (s:Subreddit)<-[:POSTED_IN]-(n)
        WHERE (n:Post OR n:Comment){_time_clause('n', time_range)}
        OPTIONAL MATCH (n)-[:ABOUT_TOPIC]->(t:Topic)
        WITH s, n, collect(DISTINCT t.name) AS topics
        WHERE size($topics) = 0 OR any(x IN topics WHERE x IN $topics)
        WITH s,
             count(n) AS volume,
             sum(coalesce(n.score,0)) AS engagement,
             collect({{id:n.id, title:coalesce(n.title,''), body:coalesce(n.body,''),
                      subreddit:n.subreddit, author:n.author, created_utc:n.created_utc,
                      time_window:n.time_window, sentiment:n.sentiment, url:n.url,
                      type:labels(n)[0], score:coalesce(n.score,0), topics:topics}}) AS items
        WITH s, volume, engagement, items,
             (volume + 0.2*engagement) AS leadership
        ORDER BY leadership DESC
        LIMIT $limit
        WITH s, leadership, volume,
             head([it IN items WHERE it.score = reduce(m=-2147483648, x IN items | CASE WHEN x.score>m THEN x.score ELSE m END) | it]) AS top_item
        RETURN top_item.id AS doc_id, top_item.title AS title, top_item.body AS body,
               s.name AS subreddit, top_item.author AS author,
               top_item.created_utc AS created_utc, top_item.time_window AS time_window,
               top_item.sentiment AS sentiment, top_item.url AS url, top_item.type AS type,
               top_item.topics AS topics, leadership AS relevance, volume AS volume
        """
        rows = self._client.query(cypher, params)
        hits = []
        for rank, r in enumerate(rows, start=1):
            hit = self._row_to_hit(r, rank)
            hit.explanation = f"leading community r/{r.get('subreddit')} ({r.get('volume')} items on topic)"
            hits.append(hit)
        return hits

    # ---- helpers ------------------------------------------------------
    @staticmethod
    def _row_to_hit(r: Dict, rank: int) -> RetrievedHit:
        title = r.get("title") or ""
        body = r.get("body") or ""
        text = f"{title}\n\n{body}".strip() if title else body
        return RetrievedHit(
            doc_id=r["doc_id"],
            text=truncate(text, 1200),
            source=RetrieverSource.GRAPH,
            score=float(r.get("relevance", 0.0) or 0.0),
            rank=rank,
            metadata={
                "subreddit": r.get("subreddit"),
                "author": r.get("author"),
                "created_utc": r.get("created_utc"),
                "time_window": r.get("time_window"),
                "sentiment": r.get("sentiment"),
                "url": r.get("url"),
                "type": (r.get("type") or "").lower(),
                "topics": r.get("topics") or [],
            },
            explanation=f"graph match on topics {r.get('topics') or []}",
        )

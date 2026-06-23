"""Temporal knowledge-graph schema: node labels, relationships, constraints.

NODES
-----
  (:User      {name})
  (:Subreddit {name})
  (:Post      {id, title, body, score, created_utc, created_at,
               time_window, month, year, sentiment, sentiment_score, url})
  (:Comment   {id, body, score, created_utc, created_at,
               time_window, month, year, sentiment, sentiment_score, url})
  (:Topic     {name})
  (:Entity    {name, type})
  (:TimeWindow{id})            # quarter buckets, e.g. "2026-Q1"

RELATIONSHIPS (every edge carries `created_utc` so traversals are temporal)
---------------------------------------------------------------------------
  (User)-[:AUTHORED       {created_utc}]->(Post|Comment)
  (Post|Comment)-[:POSTED_IN]->(Subreddit)
  (Comment)-[:COMMENTED_ON {created_utc}]->(Post)     # to its root post
  (Comment)-[:REPLY_TO     {created_utc}]->(Post|Comment)  # immediate parent
  (Post|Comment)-[:ABOUT_TOPIC]->(Topic)
  (Post|Comment)-[:MENTIONS]->(Entity)
  (Post|Comment)-[:IN_WINDOW]->(TimeWindow)

This model supports the required traversals:
  * author influence  : (User)-[:AUTHORED]->()-[:ABOUT_TOPIC]->(Topic)
  * community leadership: (Subreddit)<-[:POSTED_IN]-()-[:ABOUT_TOPIC]->(Topic)
  * thread structure   : (Comment)-[:REPLY_TO*]->(Post)
  * entity co-occurrence: (Topic)<-[:ABOUT_TOPIC]-()-[:MENTIONS]->(Entity)
  * temporal slices    : ()-[:IN_WINDOW]->(:TimeWindow {id})
"""
from __future__ import annotations

from graph.neo4j_client import Neo4jClient
from utils.logging import get_logger

logger = get_logger("graph.schema")

CONSTRAINTS = [
    "CREATE CONSTRAINT user_name IF NOT EXISTS FOR (u:User) REQUIRE u.name IS UNIQUE",
    "CREATE CONSTRAINT subreddit_name IF NOT EXISTS FOR (s:Subreddit) REQUIRE s.name IS UNIQUE",
    "CREATE CONSTRAINT post_id IF NOT EXISTS FOR (p:Post) REQUIRE p.id IS UNIQUE",
    "CREATE CONSTRAINT comment_id IF NOT EXISTS FOR (c:Comment) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT topic_name IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE",
    "CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
    "CREATE CONSTRAINT window_id IF NOT EXISTS FOR (w:TimeWindow) REQUIRE w.id IS UNIQUE",
]

INDEXES = [
    "CREATE INDEX post_created IF NOT EXISTS FOR (p:Post) ON (p.created_utc)",
    "CREATE INDEX comment_created IF NOT EXISTS FOR (c:Comment) ON (c.created_utc)",
    "CREATE INDEX post_window IF NOT EXISTS FOR (p:Post) ON (p.time_window)",
    "CREATE INDEX comment_window IF NOT EXISTS FOR (c:Comment) ON (c.time_window)",
]


def apply_schema(client: Neo4jClient) -> None:
    for stmt in CONSTRAINTS + INDEXES:
        client.execute_write(stmt)
    logger.info("Applied %d constraints + indexes", len(CONSTRAINTS) + len(INDEXES))


def reset_graph(client: Neo4jClient) -> None:
    """Danger: wipes all data. Used by the demo's --reset flag."""
    client.execute_write("MATCH (n) DETACH DELETE n")
    logger.warning("Graph wiped (all nodes/relationships deleted)")

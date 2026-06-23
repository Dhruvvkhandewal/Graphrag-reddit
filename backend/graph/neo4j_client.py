"""Neo4j driver lifecycle management (singleton).

Why Neo4j? The assignment's headline queries — "most influential voices",
"which communities lead the conversation", "who is connected to whom" — are
multi-hop traversals and centrality computations. Those are native to a
property graph and awkward/expensive in a relational or document store. Neo4j
gives us declarative Cypher, first-class relationships, and cheap variable-
length traversals, while still letting us store rich temporal properties on
every node and edge.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from config import get_settings
from utils.logging import get_logger

logger = get_logger("graph.client")


class Neo4jClient:
    def __init__(self) -> None:
        self._settings = get_settings()
        self._driver = None

    def _get_driver(self):
        if self._driver is not None:
            return self._driver
        from neo4j import GraphDatabase  # lazy

        s = self._settings
        self._driver = GraphDatabase.driver(
            s.neo4j_uri, auth=(s.neo4j_user, s.neo4j_password)
        )
        self._driver.verify_connectivity()
        logger.info("Connected to Neo4j at %s", s.neo4j_uri)
        return self._driver

    def execute_write(self, cypher: str, params: Dict[str, Any] | None = None) -> None:
        driver = self._get_driver()
        with driver.session(database=self._settings.neo4j_database) as session:
            session.execute_write(lambda tx: tx.run(cypher, params or {}).consume())

    def query(
        self, cypher: str, params: Dict[str, Any] | None = None
    ) -> List[Dict[str, Any]]:
        driver = self._get_driver()
        with driver.session(database=self._settings.neo4j_database) as session:
            result = session.run(cypher, params or {})
            return [record.data() for record in result]

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None


_client: Optional[Neo4jClient] = None


def get_neo4j_client() -> Neo4jClient:
    global _client
    if _client is None:
        _client = Neo4jClient()
    return _client

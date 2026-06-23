"""Centralised, type-safe configuration.

Every tunable lives here and is sourced from environment variables / `.env`.
Nothing else in the codebase reads `os.environ` directly — that keeps
configuration auditable and makes the system trivial to reconfigure for a
different topic, scale, or deployment target.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # ---- Reddit ----
    reddit_client_id: str = Field("", alias="REDDIT_CLIENT_ID")
    reddit_client_secret: str = Field("", alias="REDDIT_CLIENT_SECRET")
    reddit_user_agent: str = Field("graphrag-reddit/1.0", alias="REDDIT_USER_AGENT")
    reddit_username: str = Field("", alias="REDDIT_USERNAME")
    reddit_password: str = Field("", alias="REDDIT_PASSWORD")

    # ---- Gemini ----
    gemini_api_key: str = Field("", alias="GEMINI_API_KEY")
    gemini_model: str = Field("gemini-2.5-flash", alias="GEMINI_MODEL")
    llm_allow_fallback: bool = Field(True, alias="LLM_ALLOW_FALLBACK")

    # ---- Neo4j ----
    neo4j_uri: str = Field("bolt://localhost:7687", alias="NEO4J_URI")
    neo4j_user: str = Field("neo4j", alias="NEO4J_USER")
    neo4j_password: str = Field("please_change_me", alias="NEO4J_PASSWORD")
    neo4j_database: str = Field("neo4j", alias="NEO4J_DATABASE")

    # ---- Chroma ----
    chroma_persist_dir: str = Field("./.chroma", alias="CHROMA_PERSIST_DIR")
    chroma_collection: str = Field("reddit_content", alias="CHROMA_COLLECTION")

    # ---- Embeddings ----
    embedding_model: str = Field("BAAI/bge-small-en-v1.5", alias="EMBEDDING_MODEL")
    embedding_device: str = Field("cpu", alias="EMBEDDING_DEVICE")

    # ---- Ingestion ----
    reddit_subreddits: List[str] = Field(
        default_factory=lambda: ["MachineLearning", "LocalLLaMA", "artificial"],
        alias="REDDIT_SUBREDDITS",
    )
    scrape_post_limit: int = Field(120, alias="SCRAPE_POST_LIMIT")
    scrape_comment_limit: int = Field(40, alias="SCRAPE_COMMENT_LIMIT")
    scrape_comment_depth: int = Field(3, alias="SCRAPE_COMMENT_DEPTH")
    time_window_months_back: int = Field(6, alias="TIME_WINDOW_MONTHS_BACK")

    # ---- Retrieval ----
    retrieval_top_k: int = Field(10, alias="RETRIEVAL_TOP_K")
    retriever_candidate_k: int = Field(25, alias="RETRIEVER_CANDIDATE_K")
    rrf_k: int = Field(60, alias="RRF_K")
    graph_weight: float = Field(1.0, alias="GRAPH_WEIGHT")
    vector_weight: float = Field(1.0, alias="VECTOR_WEIGHT")

    # ---- API ----
    api_host: str = Field("0.0.0.0", alias="API_HOST")
    api_port: int = Field(8000, alias="API_PORT")
    log_level: str = Field("INFO", alias="LOG_LEVEL")

    @field_validator("reddit_subreddits", mode="before")
    @classmethod
    def _split_csv(cls, v):
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return v

    @property
    def llm_enabled(self) -> bool:
        return bool(self.gemini_api_key)


@lru_cache
def get_settings() -> Settings:
    """Process-wide singleton. Cached so `.env` is parsed exactly once."""
    return Settings()

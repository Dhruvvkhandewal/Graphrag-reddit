"""Embedding model wrapper (sentence-transformers / BAAI/bge-small-en-v1.5).

Why bge-small-en-v1.5? It is a 33M-param, 384-dim model that punches far above
its size on MTEB retrieval, runs comfortably on CPU, and is Apache-2.0. For a
corpus of Reddit posts/comments it gives strong recall at near-zero cost — no
embedding API bills, no rate limits, fully reproducible.

Important bge detail: retrieval quality improves measurably when the QUERY is
prefixed with the model's instruction while DOCUMENTS are embedded raw. We
encode the two asymmetrically (`embed_query` vs `embed_documents`) so we get
that boost for free.
"""
from __future__ import annotations

from typing import List

from config import get_settings
from utils.logging import get_logger

logger = get_logger("vectorstore.embeddings")

# Recommended retrieval instruction for the bge-*-en-v1.5 family.
_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class EmbeddingModel:
    def __init__(self) -> None:
        self._settings = get_settings()
        self._model = None  # lazy
        self._dim: int | None = None

    def _load(self):
        if self._model is not None:
            return self._model
        from sentence_transformers import SentenceTransformer  # lazy

        s = self._settings
        logger.info("Loading embedding model %s on %s", s.embedding_model, s.embedding_device)
        self._model = SentenceTransformer(s.embedding_model, device=s.embedding_device)
        self._dim = self._model.get_sentence_embedding_dimension()
        return self._model

    @property
    def dimension(self) -> int:
        self._load()
        return int(self._dim or 384)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        model = self._load()
        vecs = model.encode(
            texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False
        )
        return [v.tolist() for v in vecs]

    def embed_query(self, query: str) -> List[float]:
        model = self._load()
        vec = model.encode(
            _QUERY_INSTRUCTION + query,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vec.tolist()


_model: EmbeddingModel | None = None


def get_embedding_model() -> EmbeddingModel:
    global _model
    if _model is None:
        _model = EmbeddingModel()
    return _model

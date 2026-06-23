"""ChromaDB persistence + metadata-filtered vector search.

Why ChromaDB? It is an embedded, persistent vector store (no separate server),
which keeps the "clone → run in <10 min" promise realistic. It supports rich
`where` metadata filters with numeric range operators ($gte/$lte) — exactly
what we need to push *time-range, subreddit, author, sentiment* filters down
to the index BEFORE similarity search, rather than over-fetching and filtering
in Python. We bring our own embeddings, so Chroma is purely the ANN index +
metadata filter layer.

Each chunk is stored with metadata that mirrors the graph's join key
(`doc_id`) plus everything needed for temporal/faceted filtering.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

from config import get_settings
from utils.logging import get_logger
from utils.models import RedditDocument
from vectorstore.chunker import chunk_text
from vectorstore.embeddings import EmbeddingModel, get_embedding_model

logger = get_logger("vectorstore.chroma")


class ChromaStore:
    def __init__(self, embedder: EmbeddingModel | None = None) -> None:
        self._settings = get_settings()
        self._embedder = embedder or get_embedding_model()
        self._client = None
        self._collection = None

    def _coll(self):
        if self._collection is not None:
            return self._collection
        import chromadb  # lazy

        s = self._settings
        self._client = chromadb.PersistentClient(path=s.chroma_persist_dir)
        # cosine space matches our normalised embeddings.
        self._collection = self._client.get_or_create_collection(
            name=s.chroma_collection, metadata={"hnsw:space": "cosine"}
        )
        logger.info(
            "Chroma collection '%s' ready (%d docs)",
            s.chroma_collection,
            self._collection.count(),
        )
        return self._collection

    # ------------------------------------------------------------------
    def upsert_documents(self, docs: Sequence[RedditDocument]) -> int:
        coll = self._coll()
        ids, texts, metadatas = [], [], []
        for doc in docs:
            for chunk in chunk_text(doc.doc_id, doc.text):
                ids.append(chunk.chunk_id)
                texts.append(chunk.text)
                metadatas.append(self._metadata(doc, chunk.index))
        if not ids:
            return 0
        embeddings = self._embedder.embed_documents(texts)
        # Chroma caps batch size; chunk the upsert.
        B = 1000
        for i in range(0, len(ids), B):
            coll.upsert(
                ids=ids[i : i + B],
                documents=texts[i : i + B],
                embeddings=embeddings[i : i + B],
                metadatas=metadatas[i : i + B],
            )
        logger.info("Upserted %d chunks into Chroma", len(ids))
        return len(ids)

    @staticmethod
    def _metadata(doc: RedditDocument, chunk_index: int) -> Dict:
        enr = doc.enrichment
        return {
            "doc_id": doc.doc_id,
            "chunk_index": chunk_index,
            "type": doc.type.value,
            "subreddit": doc.subreddit,
            "author": doc.author,
            "created_utc": float(doc.created_utc),
            "time_window": doc.time_window,
            "month": doc.month,
            "year": doc.year,
            "score": doc.score,
            "sentiment": enr.sentiment.value if enr else "neutral",
            "topics": ", ".join(enr.topics) if enr else "",
            "url": doc.url,
            "root_post_id": doc.root_post_id or doc.doc_id,
        }

    # ------------------------------------------------------------------
    def query(
        self,
        query_text: str,
        *,
        top_k: int = 25,
        where: Dict | None = None,
    ) -> List[Dict]:
        """Return chunk hits with similarity scores and metadata."""
        coll = self._coll()
        q_emb = self._embedder.embed_query(query_text)
        res = coll.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            where=where or None,
            include=["documents", "metadatas", "distances"],
        )
        hits: List[Dict] = []
        if not res["ids"] or not res["ids"][0]:
            return hits
        for cid, doc, meta, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            hits.append(
                {
                    "chunk_id": cid,
                    "doc_id": meta.get("doc_id", cid.split("::")[0]),
                    "text": doc,
                    "metadata": meta,
                    # cosine distance -> similarity
                    "score": 1.0 - float(dist),
                }
            )
        return hits

    def count(self) -> int:
        return self._coll().count()


_store: ChromaStore | None = None


def get_chroma_store() -> ChromaStore:
    global _store
    if _store is None:
        _store = ChromaStore()
    return _store

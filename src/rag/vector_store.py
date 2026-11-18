"""Persistent ChromaDB storage for the bilingual Translation Memory."""

import hashlib
import threading
from typing import Any, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.core.logger import get_logger
from src.core.settings import AppSettings

logger = get_logger(__name__)


class _SerializedEmbeddings(Embeddings):
    """Serializes calls to the embedding model.

    HF *fast* tokenizers (used inside the embedding model) are not thread-safe: concurrent
    searches and indexing would fail with "RuntimeError: Already borrowed".
    """

    def __init__(self, inner: Any):
        self._inner = inner
        self._lock = threading.Lock()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        with self._lock:
            return self._inner.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        with self._lock:
            return self._inner.embed_query(text)


class ChromaTranslationStore:
    """Manages persistent ChromaDB storage for bilingual Translation Memory with deduplication."""

    def __init__(self, settings: AppSettings, embeddings: Optional[Any] = None):
        from langchain_chroma import Chroma

        self.settings = settings
        # Reads and writes must not overlap: a search that runs while a batch is being written can
        # see a half-written record (Document(page_content=None)). The lock is taken per batch.
        self._lock = threading.RLock()
        if embeddings is None:
            from langchain_huggingface import HuggingFaceEmbeddings

            embeddings = HuggingFaceEmbeddings(model_name=settings.vector_store.embedding_model_id)
        self.embeddings = _SerializedEmbeddings(embeddings)
        self.db = Chroma(
            collection_name=settings.vector_store.collection_name,
            embedding_function=self.embeddings,
            persist_directory=settings.vector_store.persist_dir,
        )

    @staticmethod
    def _make_doc_id(en_text: str) -> str:
        """Generate a deterministic ID from English text for idempotent upserts."""
        normalized = " ".join(en_text.strip().lower().split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:32]

    def add_pairs(self, pairs: list[dict], batch_size: int = 100) -> int:
        """Upsert pairs; returns the number of distinct source sentences written.

        Re-indexing the same source sentence replaces it (no duplicate search hits). If the
        same sentence appears several times in one call, the last translation wins. Errors
        are propagated: indexing is idempotent, so the caller can simply retry the request.
        """
        unique: dict[str, Document] = {}
        for item in pairs:
            en_text = (item.get("en") or "").strip()
            vi_text = (item.get("vi") or "").strip()
            if not en_text or not vi_text:
                continue
            unique[self._make_doc_id(en_text)] = Document(
                page_content=en_text,
                metadata={"en": en_text, "vi": vi_text},
            )

        ids = list(unique)
        docs = list(unique.values())
        for i in range(0, len(docs), batch_size):
            with self._lock:  # per batch: searches are held back for at most one batch
                self.db.add_documents(docs[i : i + batch_size], ids=ids[i : i + batch_size])
        return len(docs)

    def dense_search(self, query_en: str, k: int) -> list[dict]:
        """Nearest neighbours of the query (errors propagate; the engine degrades to BM25)."""
        with self._lock:
            results = self.db.similarity_search_with_score(query_en, k=k)

        items = []
        for doc, dist in results:
            # Convert L2 distance to a bounded similarity score in (0, 1]
            sim_score = 1.0 / (1.0 + max(0.0, float(dist)))
            items.append(
                {
                    "en": doc.metadata.get("en", doc.page_content),
                    "vi": doc.metadata.get("vi", ""),
                    "dense_score": round(sim_score, 4),
                }
            )
        return items

    def count(self) -> int:
        """Return the number of documents in the collection (-1 if it cannot be read)."""
        try:
            with self._lock:
                return int(self.db._collection.count())
        except Exception:
            logger.warning("Could not read the vector store size", exc_info=True)
            return -1

    def get_all_pairs(self) -> list[dict]:
        """Return every (en, vi) pair stored in the collection.

        Called by the engine on startup to warm-up the in-memory BM25 index so
        that both retrievers (dense + BM25) always reflect the same translation
        memory — even after a container restart that wipes BM25's RAM state while
        ChromaDB's persisted data survives intact.
        """
        try:
            with self._lock:
                result = self.db.get(include=["metadatas"])
            pairs: list[dict] = []
            for meta in result.get("metadatas") or []:
                if meta and meta.get("en") and meta.get("vi"):
                    pairs.append({"en": meta["en"], "vi": meta["vi"]})
            return pairs
        except Exception:
            logger.warning("Could not read pairs from vector store for BM25 warm-up", exc_info=True)
            return []

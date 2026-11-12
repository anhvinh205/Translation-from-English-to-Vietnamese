import hashlib
import math
import re
import threading
from collections import Counter


class LightweightBM25:
    """Zero-dependency BM25 lexical scorer for exact terminology & named-entity matching.

    Thread-safe: FastAPI runs sync handlers in a thread pool, so ``/memory/index`` can
    run while ``/translate`` searches.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus: list[dict] = []
        self.tokenized_docs: list[list[str]] = []
        self.tf_maps: list[Counter] = []  # Pre-computed term frequency maps
        self.doc_freqs: Counter = Counter()
        self.avgdl: float = 0.0
        self._total_len = 0
        self._hash_to_idx: dict[str, int] = {}  # source-sentence hash -> position in corpus
        self._lock = threading.RLock()

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    @staticmethod
    def _hash_text(text: str) -> str:
        normalized = " ".join(text.split()).lower()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def index(self, pairs: list[dict]) -> int:
        """Index pairs incrementally. Returns the number of newly indexed source sentences.

        A source sentence that is already indexed keeps its position and statistics, but its
        translation is replaced by the latest one (same "last write wins" rule as the
        vector store's upsert, so both retrievers always agree).
        """
        with self._lock:
            added = 0
            for p in pairs:
                en_text = p.get("en", "")
                vi_text = p.get("vi", "")
                if not en_text or not vi_text:
                    continue
                h = self._hash_text(en_text)
                existing = self._hash_to_idx.get(h)
                if existing is not None:
                    if self.corpus[existing].get("vi") != vi_text:
                        self.corpus[existing] = p
                    continue

                tokens = self._tokenize(en_text)
                self._hash_to_idx[h] = len(self.corpus)
                self.corpus.append(p)
                self.tokenized_docs.append(tokens)
                self.tf_maps.append(Counter(tokens))
                for t in set(tokens):
                    self.doc_freqs[t] += 1
                self._total_len += len(tokens)
                added += 1

            if added:
                self.avgdl = self._total_len / max(1, len(self.tokenized_docs))
            return added

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        with self._lock:
            if not self.corpus:
                return []
            q_tokens = self._tokenize(query)
            n_docs = len(self.corpus)
            scores: list[tuple[int, float]] = []

            for idx, (doc_tokens, tf_map) in enumerate(zip(self.tokenized_docs, self.tf_maps)):
                dl = len(doc_tokens)
                score = 0.0
                for q in q_tokens:
                    df = self.doc_freqs.get(q, 0)
                    if df == 0:
                        continue
                    idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
                    tf = tf_map[q]
                    denom = tf + self.k1 * (1.0 - self.b + self.b * dl / max(1e-6, self.avgdl))
                    score += idf * (tf * (self.k1 + 1.0)) / max(1e-6, denom)
                if score > 0:
                    scores.append((idx, score))

            scores.sort(key=lambda x: x[1], reverse=True)
            max_score = scores[0][1] if scores else 1.0
            results = []
            for idx, raw_score in scores[:top_k]:
                item = self.corpus[idx]
                results.append(
                    {
                        "en": item["en"],
                        "vi": item["vi"],
                        "bm25_score": round(raw_score / max(1e-6, max_score), 4),
                    }
                )
            return results

    def clear(self) -> None:
        """Clear all indexed data."""
        with self._lock:
            self.corpus.clear()
            self.tokenized_docs.clear()
            self.tf_maps.clear()
            self.doc_freqs.clear()
            self.avgdl = 0.0
            self._total_len = 0
            self._hash_to_idx.clear()


def reciprocal_rank_fusion(
    dense_results: list[dict],
    bm25_results: list[dict],
    dense_weight: float = 0.6,
    bm25_weight: float = 0.4,
    rrf_k: int = 60,
) -> list[dict]:
    """Fuse Dense Vector and BM25 candidates via weighted Reciprocal Rank Fusion (RRF)."""
    merged: dict[str, dict] = {}

    for rank, item in enumerate(dense_results, 1):
        key = item["en"].strip().lower()
        merged[key] = {
            "en": item["en"],
            "vi": item["vi"],
            "dense_score": item.get("dense_score", 0.0),
            "bm25_score": 0.0,
            "hybrid_score": dense_weight * (1.0 / (rrf_k + rank)),
        }

    for rank, item in enumerate(bm25_results, 1):
        key = item["en"].strip().lower()
        rrf_contrib = bm25_weight * (1.0 / (rrf_k + rank))
        if key in merged:
            merged[key]["bm25_score"] = item.get("bm25_score", 0.0)
            merged[key]["hybrid_score"] += rrf_contrib
        else:
            merged[key] = {
                "en": item["en"],
                "vi": item["vi"],
                "dense_score": 0.0,
                "bm25_score": item.get("bm25_score", 0.0),
                "hybrid_score": rrf_contrib,
            }

    fused = list(merged.values())
    for entry in fused:
        entry["hybrid_score"] = round(entry["hybrid_score"], 5)
    fused.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return fused

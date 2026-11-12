"""Cross-Encoder reranking with a lexical fallback and a confidence threshold."""

import math
import re
import threading
from typing import Optional

from src.core.logger import get_logger
from src.core.settings import AppSettings

logger = get_logger(__name__)

# Precompiled regex for tokenization
_WORD_PATTERN = re.compile(r"\w+")

# Maximum candidates to send to cross-encoder in a single batch
_MAX_RERANK_CANDIDATES = 100


class CrossEncoderReranker:
    """Reranks Hybrid RAG candidates with a Cross-Encoder and applies a confidence threshold."""

    def __init__(self, settings: AppSettings, use_neural_cross_encoder: bool = True):
        self.settings = settings
        self.cross_encoder = None
        # The cross-encoder's fast tokenizer is not thread-safe ("RuntimeError: Already borrowed").
        self._predict_lock = threading.Lock()
        # True only when the neural model was requested but could not be loaded.
        self.degraded = False
        if use_neural_cross_encoder:
            try:
                from sentence_transformers import CrossEncoder

                self.cross_encoder = CrossEncoder(settings.reranker.model_id)
                logger.info("CrossEncoder loaded: %s", settings.reranker.model_id)
            except Exception as e:
                logger.warning(
                    "Failed to load CrossEncoder '%s': %s. Falling back to lexical overlap scoring.",
                    settings.reranker.model_id,
                    e,
                )
                self.cross_encoder = None
                self.degraded = True

    @property
    def mode(self) -> str:
        """``cross-encoder`` | ``lexical-fallback`` (neural model failed to load) | ``lexical``."""
        if self.cross_encoder is not None:
            return "cross-encoder"
        return "lexical-fallback" if self.degraded else "lexical"

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, x))))

    @staticmethod
    def _lexical_overlap_score(query: str, candidate: str) -> float:
        q_tokens = set(_WORD_PATTERN.findall(query.lower()))
        c_tokens = set(_WORD_PATTERN.findall(candidate.lower()))
        if not q_tokens or not c_tokens:
            return 0.0
        return len(q_tokens & c_tokens) / len(q_tokens | c_tokens)

    def rerank(
        self,
        query_en: str,
        candidates: list[dict],
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> list[dict]:
        if not candidates:
            return []

        k = top_k if top_k is not None else self.settings.reranker.top_k_final
        min_score = threshold if threshold is not None else self.settings.reranker.score_threshold

        # Limit candidates to prevent OOM on cross-encoder
        limited_candidates = candidates[:_MAX_RERANK_CANDIDATES]

        if self.cross_encoder is not None:
            pairs = [[query_en, c["en"]] for c in limited_candidates]
            with self._predict_lock:
                raw_scores = self.cross_encoder.predict(pairs)
            scores = [self._sigmoid(float(s)) for s in raw_scores]
        else:
            scores = [
                0.5 * c.get("dense_score", 0.0) + 0.5 * self._lexical_overlap_score(query_en, c["en"])
                for c in limited_candidates
            ]

        scored_candidates = []
        for cand, sc in zip(limited_candidates, scores):
            if sc >= min_score:
                scored_candidates.append({**cand, "rerank_score": round(float(sc), 4)})

        scored_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored_candidates[:k]

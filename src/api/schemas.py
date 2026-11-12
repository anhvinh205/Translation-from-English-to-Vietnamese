"""Request / response models of the REST API."""

from typing import Optional

from pydantic import BaseModel, Field, field_validator


def _require_text(value: str, min_chars: int = 1) -> str:
    value = value.strip()
    if len(value) < min_chars:
        raise ValueError("must not be blank")
    return value


class TranslateRequest(BaseModel):
    text_en: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="English sentence to translate (max 5000 chars).",
    )
    use_rag: bool = Field(default=True, description="Enable Hybrid RAG + Cross-Encoder Reranker.")
    top_k: int = Field(default=3, ge=1, le=10, description="Max few-shot examples after reranking.")
    score_threshold: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Minimum reranker confidence threshold."
    )

    @field_validator("text_en")
    @classmethod
    def _text_not_blank(cls, value: str) -> str:
        return _require_text(value)


class RetrievedExample(BaseModel):
    en: str
    vi: str
    dense_score: float = 0.0
    bm25_score: float = 0.0
    hybrid_score: float = 0.0
    rerank_score: float = 0.0


class TranslateResponse(BaseModel):
    source: str
    translation: str
    retrieved_memory: list[RetrievedExample]
    rag_triggered: bool
    active_adapter: str
    latency_ms: float
    cache_hit: bool


class MemoryPair(BaseModel):
    en: str = Field(..., min_length=2, max_length=2000)
    vi: str = Field(..., min_length=2, max_length=2000)

    @field_validator("en", "vi")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        return _require_text(value, min_chars=2)


class IndexMemoryRequest(BaseModel):
    pairs: list[MemoryPair] = Field(..., min_length=1, max_length=10000)


class IndexMemoryResponse(BaseModel):
    indexed_count: int
    status: str = "success"


class ClearCacheResponse(BaseModel):
    status: str = "success"
    deleted_entries: int


class HealthResponse(BaseModel):
    status: str
    base_model: str
    active_adapter: str
    device: str
    reranker: str = Field(description="cross-encoder | lexical-fallback (model failed to load) | lexical")
    degraded: bool = Field(description="True when the neural reranker failed to load and a fallback is used.")
    vector_store_count: int
    cache_entries: int


class ErrorResponse(BaseModel):
    error: str
    detail: str = ""
    error_id: Optional[str] = Field(default=None, description="Correlates a 500 response with the server logs.")

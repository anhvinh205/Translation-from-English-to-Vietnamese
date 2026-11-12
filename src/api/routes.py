"""REST routes.

NOTE: do not add ``from __future__ import annotations`` here: slowapi wraps the handlers
and FastAPI would then fail to resolve the string annotations of the wrapped functions.
"""

from fastapi import APIRouter, Depends, Request
from slowapi import Limiter

from src.api.schemas import (
    ClearCacheResponse,
    ErrorResponse,
    HealthResponse,
    IndexMemoryRequest,
    IndexMemoryResponse,
    TranslateRequest,
    TranslateResponse,
)
from src.api.security import require_admin_key, require_api_key
from src.core.settings import AppSettings

_ERRORS = {
    401: {"model": ErrorResponse, "description": "Missing or invalid API key"},
    403: {"model": ErrorResponse, "description": "Not allowed (admin key required)"},
    429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    503: {"model": ErrorResponse, "description": "Engine busy or out of memory, retry later"},
}


def build_router(limiter: Limiter, settings: AppSettings) -> APIRouter:
    """Create the router; rate limits are read from the settings of this app instance."""
    router = APIRouter()
    limits = settings.security

    @router.get("/health", response_model=HealthResponse)
    def health_check(request: Request):
        engine = request.app.state.engine
        return {
            "status": "healthy",
            "base_model": engine.settings.model.base_model_id,
            "active_adapter": engine.active_adapter,
            "device": engine.settings.device,
            **engine.health(),
        }

    @router.post(
        "/translate",
        response_model=TranslateResponse,
        responses=_ERRORS,
        dependencies=[Depends(require_api_key)],
    )
    @limiter.limit(limits.rate_limit_translate)
    def translate_text(request: Request, payload: TranslateRequest):
        engine = request.app.state.engine
        return engine.translate(
            text_en=payload.text_en,
            use_rag=payload.use_rag,
            top_k=payload.top_k,
            score_threshold=payload.score_threshold,
        )

    @router.post(
        "/memory/index",
        response_model=IndexMemoryResponse,
        responses=_ERRORS,
        dependencies=[Depends(require_admin_key)],
    )
    @limiter.limit(limits.rate_limit_memory_index)
    def index_translation_memory(request: Request, payload: IndexMemoryRequest):
        """Add pairs to the translation memory (admin only: it steers future translations)."""
        engine = request.app.state.engine
        raw_pairs = [{"en": p.en, "vi": p.vi} for p in payload.pairs]
        return IndexMemoryResponse(indexed_count=engine.index_memory(raw_pairs))

    @router.delete(
        "/cache",
        response_model=ClearCacheResponse,
        responses=_ERRORS,
        dependencies=[Depends(require_admin_key)],
    )
    @limiter.limit(limits.rate_limit_admin)
    def clear_cache(request: Request):
        """Clear the translation cache (admin only)."""
        engine = request.app.state.engine
        return ClearCacheResponse(deleted_entries=engine.cache.clear())

    return router

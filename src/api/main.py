"""FastAPI application factory (no work is done at import time)."""

import os
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import Optional
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.api.routes import build_router
from src.core.errors import EngineBusyError, InvalidInputError, ModelOutOfMemoryError
from src.core.logger import get_logger
from src.core.settings import AppSettings, load_settings
from src.rag.engine import HybridRAGTranslationEngine

logger = get_logger(__name__)


def _error_response(status_code: int, error: str, detail: str, headers: Optional[dict] = None, **extra):
    body = {"error": error, "detail": detail, **extra}
    return JSONResponse(status_code=status_code, content=body, headers=headers)


def _retry_after_seconds(exc: RateLimitExceeded) -> int:
    try:
        return max(1, int(exc.limit.limit.get_expiry()))
    except Exception:
        return 60


def create_app(
    load_llm: bool = True,
    settings: Optional[AppSettings] = None,
    engine: Optional[HybridRAGTranslationEngine] = None,
) -> FastAPI:
    """Build the app. ``settings`` / ``engine`` can be injected (tests, custom wiring)."""
    settings = settings or load_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if engine is not None:
            app.state.engine = engine
        else:
            should_load_llm = load_llm and (os.getenv("MOCK_LLM", "0") != "1")
            app.state.engine = HybridRAGTranslationEngine(settings=settings, load_llm=should_load_llm)
        if not settings.security.auth_enabled:
            logger.warning("Authentication is DISABLED: set API_KEYS (and ADMIN_API_KEYS) before exposing this API.")
        logger.info("Engine initialized. Adapter: %s, Device: %s", app.state.engine.active_adapter, settings.device)
        yield
        app.state.engine.close()
        logger.info("Engine resources cleaned up.")

    app = FastAPI(
        title="WikiMatrix Hybrid-RAG QLoRA NMT API",
        version="2.1.0",
        description=(
            "Production microservice for En-Vi machine translation with Hybrid BM25+Dense RAG, "
            "Cross-Encoder reranking & QLoRA/DPO LLM."
        ),
        lifespan=lifespan,
    )
    app.state.settings = settings

    # ---- rate limiting (in-memory, per client IP; put a reverse proxy in front for multi-worker setups)
    limiter = Limiter(
        key_func=get_remote_address,
        default_limits=[settings.security.rate_limit_default],
        enabled=settings.security.rate_limit_enabled,
    )
    app.state.limiter = limiter
    if settings.security.rate_limit_enabled:
        app.add_middleware(SlowAPIMiddleware)

    # Added last = outermost, so CORS headers are also present on 429 / 401 responses.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors.allowed_origins,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type", "Authorization", "X-API-Key"],
    )

    # ---- error handling: consistent {"error", "detail"} bodies, no internals leaked
    @app.exception_handler(RateLimitExceeded)
    async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
        return _error_response(
            429,
            "too_many_requests",
            f"Rate limit exceeded: {exc.detail}",
            headers={"Retry-After": str(_retry_after_seconds(exc))},
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        try:
            code = HTTPStatus(exc.status_code).phrase.lower().replace(" ", "_")
        except ValueError:
            code = "error"
        return _error_response(exc.status_code, code, str(exc.detail), headers=getattr(exc, "headers", None))

    @app.exception_handler(InvalidInputError)
    async def invalid_input_handler(request: Request, exc: InvalidInputError):
        return _error_response(422, "invalid_input", str(exc))

    @app.exception_handler(EngineBusyError)
    async def busy_handler(request: Request, exc: EngineBusyError):
        return _error_response(503, "engine_busy", str(exc), headers={"Retry-After": "5"})

    @app.exception_handler(ModelOutOfMemoryError)
    async def oom_handler(request: Request, exc: ModelOutOfMemoryError):
        logger.error("GPU out of memory on %s %s", request.method, request.url.path)
        return _error_response(
            503,
            "out_of_memory",
            "The model ran out of memory. Try a shorter input or a smaller top_k.",
            headers={"Retry-After": "5"},
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        error_id = uuid4().hex[:12]
        logger.error(
            "Unhandled exception on %s %s (error_id=%s)",
            request.method,
            request.url.path,
            error_id,
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        return _error_response(
            500,
            "internal_server_error",
            "Internal server error. Quote the error_id when reporting this problem.",
            error_id=error_id,
        )

    app.include_router(build_router(limiter, settings), prefix="/api/v1")
    return app


_app: Optional[FastAPI] = None


def __getattr__(name: str):
    """Lazily build ``app`` so ``uvicorn src.api.main:app`` works without import-time side effects."""
    global _app
    if name == "app":
        if _app is None:
            _app = create_app()
        return _app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

"""API-key authentication (constant-time comparison, secrets never logged)."""

import secrets
from typing import Iterable, Optional

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader
from pydantic import SecretStr

_api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
    description="API key. `Authorization: Bearer <key>` is accepted as well.",
)


def _candidate_key(request: Request, header_key: Optional[str]) -> Optional[str]:
    if header_key:
        return header_key
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() == "bearer" and token.strip():
        return token.strip()
    return None


def _matches(candidate: Optional[str], valid_keys: Iterable[SecretStr]) -> bool:
    if not candidate:
        return False
    candidate_bytes = candidate.encode("utf-8")
    matched = False
    for key in valid_keys:  # no early exit: timing does not reveal which key matched
        matched |= secrets.compare_digest(candidate_bytes, key.get_secret_value().encode("utf-8"))
    return matched


async def require_api_key(request: Request, header_key: Optional[str] = Security(_api_key_header)) -> None:
    """Allow ``API_KEYS`` and ``ADMIN_API_KEYS``; no-op when authentication is disabled."""
    security = request.app.state.settings.security
    if not security.auth_enabled:
        return
    candidate = _candidate_key(request, header_key)
    if _matches(candidate, security.api_keys) or _matches(candidate, security.admin_api_keys):
        return
    raise HTTPException(
        status_code=401,
        detail="Invalid or missing API key.",
        headers={"WWW-Authenticate": "ApiKey"},
    )


async def require_admin_key(request: Request, header_key: Optional[str] = Security(_api_key_header)) -> None:
    """Allow ``ADMIN_API_KEYS`` only (translation-memory writes, cache wipes)."""
    security = request.app.state.settings.security
    if not security.auth_enabled:
        return
    if not security.admin_api_keys:
        raise HTTPException(
            status_code=403,
            detail="Admin endpoints are disabled: no ADMIN_API_KEYS are configured.",
        )
    candidate = _candidate_key(request, header_key)
    if _matches(candidate, security.admin_api_keys):
        return
    if candidate is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    raise HTTPException(status_code=403, detail="An admin API key is required for this endpoint.")

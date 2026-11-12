"""HTTP client for the translation API.

Every failure (network, timeout, HTTP error, malformed body) is turned into a short,
readable message: the Streamlit page never crashes with a traceback.
"""

from dataclasses import dataclass
from typing import Any, Optional

import requests


@dataclass
class ApiResult:
    ok: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    status_code: Optional[int] = None


def _detail_text(body: Any) -> str:
    """Extract a human-readable detail from the API's error body."""
    if not isinstance(body, dict):
        return ""
    detail = body.get("detail", "")
    if isinstance(detail, list):  # FastAPI validation errors: [{"loc": ..., "msg": ...}, ...]
        return "; ".join(str(item.get("msg", item)) if isinstance(item, dict) else str(item) for item in detail)
    return str(detail)


def _http_error_message(response: "requests.Response") -> str:
    try:
        body = response.json()
    except ValueError:
        body = None
    detail = _detail_text(body)
    status = response.status_code

    if status == 401:
        return "API yêu cầu API key hợp lệ. Hãy đặt biến môi trường `API_KEY` cho giao diện."
    if status == 403:
        return "Không đủ quyền cho thao tác này (nạp bộ nhớ dịch cần `ADMIN_API_KEY`)."
    if status == 422:
        return f"Dữ liệu không hợp lệ: {detail}" if detail else "Dữ liệu không hợp lệ."
    if status == 429:
        retry = response.headers.get("Retry-After")
        suffix = f" Thử lại sau khoảng {retry} giây." if retry else " Hãy thử lại sau."
        return "Gửi quá nhiều yêu cầu." + suffix
    if status == 503:
        return "Máy chủ đang bận hoặc thiếu bộ nhớ, hãy thử lại sau ít giây." + (f" ({detail})" if detail else "")
    if status >= 500:
        error_id = body.get("error_id") if isinstance(body, dict) else None
        ref = f" Mã lỗi để tra log: `{error_id}`." if error_id else ""
        return "Lỗi nội bộ phía máy chủ." + ref
    return f"API trả về lỗi {status}." + (f" {detail}" if detail else "")


class TranslationApiClient:
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        admin_api_key: Optional[str] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        # Memory writes need the admin key when the API enforces it; fall back to the normal key.
        self.admin_api_key = admin_api_key or api_key

    def _request(
        self,
        method: str,
        path: str,
        timeout: float,
        json: Optional[dict] = None,
        admin: bool = False,
    ) -> ApiResult:
        key = self.admin_api_key if admin else self.api_key
        headers = {"X-API-Key": key} if key else {}
        try:
            response = requests.request(
                method,
                f"{self.base_url}{path}",
                json=json,
                headers=headers,
                timeout=timeout,
            )
        except requests.exceptions.Timeout:
            return ApiResult(
                False,
                error=f"Hết thời gian chờ ({timeout:g}s). Mô hình có thể đang tải hoặc quá tải, hãy thử lại sau.",
            )
        except requests.exceptions.ConnectionError:
            return ApiResult(
                False,
                error=(
                    f"Không kết nối được tới API ({self.base_url}). "
                    "Hãy chạy `uvicorn src.api.main:app` hoặc `docker compose up`."
                ),
            )
        except requests.exceptions.RequestException as exc:
            return ApiResult(False, error=f"Lỗi mạng khi gọi API ({type(exc).__name__}).")

        if response.status_code != 200:
            return ApiResult(False, error=_http_error_message(response), status_code=response.status_code)
        try:
            return ApiResult(True, data=response.json(), status_code=200)
        except ValueError:
            return ApiResult(False, error="Phản hồi của API không phải JSON hợp lệ.", status_code=200)

    def health(self) -> ApiResult:
        return self._request("GET", "/health", timeout=5)

    def translate(
        self,
        text_en: str,
        use_rag: bool = True,
        top_k: int = 3,
        score_threshold: Optional[float] = None,
    ) -> ApiResult:
        payload = {"text_en": text_en, "use_rag": use_rag, "top_k": top_k, "score_threshold": score_threshold}
        return self._request("POST", "/translate", timeout=60, json=payload)

    def index_pair(self, en: str, vi: str) -> ApiResult:
        return self._request(
            "POST",
            "/memory/index",
            timeout=15,
            json={"pairs": [{"en": en, "vi": vi}]},
            admin=True,
        )

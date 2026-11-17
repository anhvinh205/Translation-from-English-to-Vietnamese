import pytest
import requests

from tests.conftest import make_response
from ui.api_client import TranslationApiClient


@pytest.fixture
def client():
    return TranslationApiClient("http://api.test/api/v1/", api_key="user-key", admin_api_key="admin-key")


def patch_request(monkeypatch, outcome):
    calls = []

    def fake_request(method, url, **kwargs):
        calls.append({"method": method, "url": url, **kwargs})
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(requests, "request", fake_request)
    return calls


def test_success_returns_the_parsed_body(client, monkeypatch):
    calls = patch_request(monkeypatch, make_response(200, {"translation": "Xin chào"}))
    result = client.translate("Hello", use_rag=False, top_k=2, score_threshold=0.3)

    assert result.ok and result.data == {"translation": "Xin chào"}
    assert calls[0]["url"] == "http://api.test/api/v1/translate"  # trailing slash normalised
    assert calls[0]["json"] == {"text_en": "Hello", "use_rag": False, "top_k": 2, "score_threshold": 0.3}
    assert calls[0]["timeout"] == 60


def test_api_keys_are_sent_and_memory_writes_use_the_admin_key(client, monkeypatch):
    calls = patch_request(monkeypatch, make_response(200, {"indexed_count": 1}))
    client.translate("Hello")
    client.index_pair("Hello", "Xin chào")
    client.health()
    assert [c["headers"] for c in calls] == [
        {"X-API-Key": "user-key"},
        {"X-API-Key": "admin-key"},
        {"X-API-Key": "user-key"},
    ]


def test_no_key_means_no_header_and_memory_writes_fall_back_to_the_normal_key(monkeypatch):
    calls = patch_request(monkeypatch, make_response(200, {}))
    anonymous = TranslationApiClient("http://api.test")
    anonymous.translate("Hello")
    assert calls[0]["headers"] == {}

    only_user_key = TranslationApiClient("http://api.test", api_key="k" * 12)
    only_user_key.index_pair("a b", "c d")
    assert calls[1]["headers"] == {"X-API-Key": "k" * 12}


@pytest.mark.parametrize(
    ("exception", "fragment"),
    [
        (requests.exceptions.ConnectionError("refused"), "Could not connect"),
        (requests.exceptions.ReadTimeout("slow"), "timed out"),
        (requests.exceptions.ConnectTimeout("slow"), "timed out"),
        (requests.exceptions.TooManyRedirects("loop"), "Network error"),
    ],
)
def test_network_failures_become_readable_messages(client, monkeypatch, exception, fragment):
    patch_request(monkeypatch, exception)
    result = client.translate("Hello")
    assert not result.ok
    assert fragment in result.error
    assert result.status_code is None


@pytest.mark.parametrize(
    ("status", "body", "headers", "fragment"),
    [
        (401, {"error": "unauthorized"}, {}, "API key"),
        (403, {"error": "forbidden"}, {}, "ADMIN_API_KEY"),
        (422, {"detail": [{"msg": "String should have at most 5000 characters"}]}, {}, "at most 5000"),
        (422, {"error": "invalid_input", "detail": "The input is too long"}, {}, "too long"),
        (429, {"error": "too_many_requests"}, {"Retry-After": "42"}, "42 seconds"),
        (503, {"error": "engine_busy", "detail": "All slots busy"}, {}, "All slots busy"),
        (500, {"error": "internal_server_error", "error_id": "abc123"}, {}, "abc123"),
        (404, {"detail": "Not Found"}, {}, "404"),
    ],
)
def test_http_errors_become_readable_messages(client, monkeypatch, status, body, headers, fragment):
    patch_request(monkeypatch, make_response(status, body, headers))
    result = client.translate("Hello")
    assert not result.ok
    assert fragment in result.error
    assert result.status_code == status


def test_a_non_json_error_page_does_not_crash_the_client(client, monkeypatch):
    patch_request(monkeypatch, make_response(502, text="<html>Bad gateway</html>"))
    result = client.translate("Hello")
    assert not result.ok
    assert result.status_code == 502
    assert "Internal server error" in result.error


def test_a_non_json_success_body_is_reported(client, monkeypatch):
    patch_request(monkeypatch, make_response(200, text="not json"))
    result = client.health()
    assert not result.ok
    assert "JSON" in result.error

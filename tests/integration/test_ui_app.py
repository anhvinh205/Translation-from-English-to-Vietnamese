"""Runs the real Streamlit page headlessly against a fake API."""

from pathlib import Path

import pytest
import requests

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from tests.conftest import make_response  # noqa: E402

SCRIPT = str(Path(__file__).resolve().parents[2] / "ui" / "streamlit_app.py")

HEALTH = {
    "status": "healthy",
    "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
    "active_adapter": "zero-shot",
    "device": "cpu",
    "reranker": "lexical",
    "degraded": False,
    "vector_store_count": 0,
    "cache_entries": 0,
}
TRANSLATION = {
    "source": "Hello world",
    "translation": "Xin chào thế giới",
    "retrieved_memory": [
        {
            "en": "Hello",
            "vi": "Xin chào",
            "dense_score": 0.9,
            "bm25_score": 0.5,
            "hybrid_score": 0.03,
            "rerank_score": 0.8,
        }
    ],
    "rag_triggered": True,
    "active_adapter": "zero-shot",
    "latency_ms": 12.5,
    "cache_hit": False,
}


class FakeApi:
    """Stands in for ``requests.request``: routes are matched on the URL suffix."""

    def __init__(self, **routes):
        self.routes = routes
        self.calls = []

    def __call__(self, method, url, **kwargs):
        self.calls.append({"method": method, "url": url, **kwargs})
        for suffix, outcome in self.routes.items():
            if url.endswith(suffix):
                if isinstance(outcome, Exception):
                    raise outcome
                return outcome
        raise AssertionError(f"unexpected API call: {method} {url}")

    def calls_to(self, suffix):
        return [c for c in self.calls if c["url"].endswith(suffix)]


def start(monkeypatch, api):
    monkeypatch.setattr(requests, "request", api)
    app = AppTest.from_file(SCRIPT, default_timeout=30)
    app.run()
    return app


def find(elements, fragment):
    return next(e for e in elements if fragment in e.label)


def texts(elements):
    return [e.value for e in elements]


def test_the_page_survives_an_unreachable_api(monkeypatch):
    refused = requests.exceptions.ConnectionError("refused")
    app = start(monkeypatch, FakeApi(**{"/health": refused, "/translate": refused, "/memory/index": refused}))

    assert not app.exception  # no traceback on the page
    assert any("Could not connect" in t for t in texts(app.sidebar.warning))

    find(app.button, "Translate").click().run()
    assert not app.exception
    assert any("Could not connect" in t for t in texts(app.error))


def test_a_timeout_is_reported_not_raised(monkeypatch):
    api = FakeApi(**{"/health": make_response(200, HEALTH), "/translate": requests.exceptions.ReadTimeout("slow")})
    app = start(monkeypatch, api)

    find(app.button, "Translate").click().run()
    assert not app.exception
    assert any("timed out" in t for t in texts(app.error))


def test_a_successful_translation_is_displayed(monkeypatch):
    api = FakeApi(**{"/health": make_response(200, HEALTH), "/translate": make_response(200, TRANSLATION)})
    app = start(monkeypatch, api)
    assert any("API Online" in t for t in texts(app.sidebar.success))

    find(app.button, "Translate").click().run()

    assert not app.exception
    assert "Xin chào thế giới" in texts(app.success)
    assert {m.label for m in app.metric} == {"Latency", "Cache Hit", "RAG Triggered"}
    sent = api.calls_to("/translate")[0]["json"]
    assert sent["use_rag"] is True and sent["top_k"] == 3 and sent["score_threshold"] == 0.25


@pytest.mark.parametrize(
    ("response", "fragment"),
    [
        (make_response(429, {"error": "too_many_requests"}, {"Retry-After": "30"}), "30 seconds"),
        (make_response(401, {"error": "unauthorized"}), "API key"),
        (make_response(503, {"error": "out_of_memory", "detail": "memory"}), "busy or out of memory"),
        (make_response(500, {"error": "internal_server_error", "error_id": "abc123def456"}), "abc123def456"),
    ],
)
def test_api_errors_are_shown_as_readable_messages(monkeypatch, response, fragment):
    app = start(monkeypatch, FakeApi(**{"/health": make_response(200, HEALTH), "/translate": response}))
    find(app.button, "Translate").click().run()
    assert not app.exception
    assert any(fragment in t for t in texts(app.error))


def test_blank_input_is_not_sent_to_the_api(monkeypatch):
    api = FakeApi(**{"/health": make_response(200, HEALTH)})
    app = start(monkeypatch, api)

    find(app.text_area, "Enter text to translate").set_value("   ").run()
    find(app.button, "Translate").click().run()

    assert not app.exception
    assert any("Please enter text to translate" in t for t in texts(app.warning))
    assert api.calls_to("/translate") == []


def test_a_degraded_backend_is_flagged_in_the_sidebar(monkeypatch):
    degraded = {**HEALTH, "degraded": True, "reranker": "lexical-fallback"}
    app = start(monkeypatch, FakeApi(**{"/health": make_response(200, degraded)}))
    assert any("fallback" in t for t in texts(app.sidebar.warning))


def test_the_memory_form_validates_input_and_reports_failures(monkeypatch):
    api = FakeApi(
        **{
            "/health": make_response(200, HEALTH),
            "/memory/index": make_response(403, {"error": "forbidden"}),
        }
    )
    app = start(monkeypatch, api)

    find(app.button, "Index Pair").click().run()  # nothing typed yet
    assert any("Please enter both" in t for t in texts(app.sidebar.warning))
    assert api.calls_to("/memory/index") == []

    find(app.text_area, "English Source").set_value("Hello there").run()
    find(app.text_area, "Vietnamese Reference").set_value("Xin chào").run()
    find(app.button, "Index Pair").click().run()

    assert not app.exception
    assert api.calls_to("/memory/index")[0]["json"] == {"pairs": [{"en": "Hello there", "vi": "Xin chào"}]}
    assert any("ADMIN_API_KEY" in t for t in texts(app.error))


def test_the_page_sends_the_configured_api_keys(monkeypatch):
    monkeypatch.setenv("API_KEY", "user-key-123456")
    monkeypatch.setenv("ADMIN_API_KEY", "admin-key-123456")
    api = FakeApi(
        **{
            "/health": make_response(200, HEALTH),
            "/translate": make_response(200, TRANSLATION),
            "/memory/index": make_response(200, {"indexed_count": 1, "status": "success"}),
        }
    )
    app = start(monkeypatch, api)

    find(app.button, "Translate").click().run()
    find(app.text_area, "English Source").set_value("Hello there").run()
    find(app.text_area, "Vietnamese Reference").set_value("Xin chào").run()
    find(app.button, "Index Pair").click().run()

    assert api.calls_to("/translate")[0]["headers"] == {"X-API-Key": "user-key-123456"}
    assert api.calls_to("/memory/index")[0]["headers"] == {"X-API-Key": "admin-key-123456"}
    assert any("Indexed 1 pair" in t for t in texts(app.sidebar.success))

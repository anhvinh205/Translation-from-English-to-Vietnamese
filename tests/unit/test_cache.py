import json
import os
import threading
import time

import pytest

from src.core.cache import TranslationCache, validate_cache_path


@pytest.fixture
def cache(tmp_path):
    c = TranslationCache(
        db_path=str(tmp_path / "c.db"),
        enabled=True,
        ttl_hours=1,
        max_entries=1000,
        allowed_root=str(tmp_path),
    )
    yield c
    c.close()


def test_roundtrip_is_whitespace_insensitive_but_case_sensitive(cache):
    assert cache.get("Hello world", use_rag=True, top_k=3) is None
    cache.set("Hello world", use_rag=True, top_k=3, payload={"translation": "Xin chào thế giới"})

    hit = cache.get("  Hello   world ", use_rag=True, top_k=3)
    assert hit is not None
    assert hit["cache_hit"] is True
    assert hit["translation"] == "Xin chào thế giới"
    # "Turkey" (the country) and "turkey" (the bird) must not share a translation
    assert cache.get("hello world", use_rag=True, top_k=3) is None


def test_key_includes_threshold_and_top_k(cache):
    cache.set("q", True, 3, {"translation": "a"}, score_threshold=0.1)
    assert cache.get("q", True, 3, 0.1) is not None
    assert cache.get("q", True, 3, 0.5) is None  # different threshold -> different result
    assert cache.get("q", True, 5, 0.1) is None  # different top_k


def test_rag_parameters_are_ignored_when_rag_is_off(cache):
    cache.set("q", False, 3, {"translation": "a"}, score_threshold=0.1)
    assert cache.get("q", False, 9, 0.9) is not None


def test_per_request_metadata_is_not_persisted(cache):
    cache.set("q", True, 3, {"translation": "a", "latency_ms": 850.0, "cache_hit": False})
    stored = json.loads(cache._conn.execute("SELECT payload_json FROM translations").fetchone()[0])
    assert stored == {"translation": "a"}


def test_ttl_expiry_removes_the_entry(cache):
    cache.set("old", True, 3, {"translation": "x"})
    key = cache._make_key("old", True, 3)
    cache._conn.execute("UPDATE translations SET created_at='2020-01-01T00:00:00.000Z' WHERE cache_key=?", (key,))
    cache._conn.commit()

    assert cache.get("old", True, 3) is None
    assert cache.count() == 0


def test_eviction_keeps_the_newest_entries(tmp_path):
    c = TranslationCache(db_path=str(tmp_path / "e.db"), max_entries=3)
    for i in range(6):
        c.set(f"s{i}", False, 0, {"translation": str(i)})
        time.sleep(0.003)  # created_at has millisecond resolution
    assert c.count() == 3
    assert c.get("s5", False, 0) is not None
    assert c.get("s0", False, 0) is None
    c.close()


def test_clear_and_selective_invalidation(cache):
    cache.set("a", True, 3, {"translation": "1"})
    cache.set("b", False, 0, {"translation": "2"})

    assert cache.invalidate(use_rag=True) == 1  # only the RAG entry
    assert cache.get("a", True, 3) is None
    assert cache.get("b", False, 0) is not None

    assert cache.clear() == 1
    assert cache.count() == 0


def test_namespace_isolates_entries(cache):
    cache.set_namespace("model-A")
    cache.set("q", True, 3, {"translation": "from A"})

    cache.set_namespace("model-B")  # e.g. a new adapter was trained
    assert cache.get("q", True, 3) is None

    cache.set_namespace("model-A")
    assert cache.get("q", True, 3)["translation"] == "from A"


def test_corrupted_payload_is_dropped(cache):
    cache.set("q", True, 3, {"translation": "a"})
    cache._conn.execute("UPDATE translations SET payload_json='{not json'")
    cache._conn.commit()
    assert cache.get("q", True, 3) is None
    assert cache.count() == 0


def test_disabled_cache_is_a_noop(tmp_path):
    c = TranslationCache(db_path=str(tmp_path / "x.db"), enabled=False)
    c.set("q", True, 3, {"translation": "a"})
    assert c.get("q", True, 3) is None
    assert c.count() == 0
    assert c.clear() == 0
    assert not (tmp_path / "x.db").exists()


def test_concurrent_access(cache):
    errors = []

    def work(n):
        try:
            for i in range(30):
                cache.set(f"t{n}-{i}", False, 0, {"translation": "x"})
                assert cache.get(f"t{n}-{i}", False, 0) is not None
        except Exception as exc:  # pragma: no cover - only on failure
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert cache.count() == 120


@pytest.mark.parametrize("bad", ["", "../x.db", "data/../../etc/x.db", "data/cache.txt", "data/cache\x00.db"])
def test_validate_cache_path_rejects_bad_paths(bad):
    with pytest.raises(ValueError):
        validate_cache_path(bad)


def test_validate_cache_path_enforces_allowed_root(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    assert validate_cache_path(str(root / "c.db"), allowed_root=str(root)) == root / "c.db"
    with pytest.raises(ValueError):
        validate_cache_path(str(tmp_path / "other" / "c.db"), allowed_root=str(root))


def test_symlink_cannot_escape_the_allowed_root(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (root / "link").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        if os.name == "nt" and getattr(exc, "winerror", None) in {1314, 5}:
            pytest.skip("Symlink creation requires Developer Mode or Administrator privileges on Windows")
        raise
    with pytest.raises(ValueError):
        validate_cache_path(str(root / "link" / "c.db"), allowed_root=str(root))


def test_cache_constructor_validates_the_path(tmp_path):
    with pytest.raises(ValueError):
        TranslationCache(db_path=str(tmp_path / ".." / "escape.db"))

"""SQLite-backed translation cache with TTL, size-based eviction and namespacing."""

import hashlib
import json
import sqlite3
import threading
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_ALLOWED_SUFFIXES = {".db", ".sqlite", ".sqlite3"}


def validate_cache_path(db_path: str, allowed_root: Optional[str] = None) -> Path:
    """Validate a SQLite cache path and return it as a ``Path``.

    Rejects NUL bytes, ``..`` segments and unexpected file extensions. When
    ``allowed_root`` is given, the path must also resolve (symlinks included)
    to a location inside that directory, so a tampered config cannot make the
    cache write to an arbitrary place on disk.
    """
    if not db_path or "\x00" in db_path:
        raise ValueError("cache path must be a non-empty string without NUL bytes")
    path = Path(db_path)
    if ".." in path.parts:
        raise ValueError(f"cache path must not contain .. segments: {db_path!r}")
    if path.suffix.lower() not in _ALLOWED_SUFFIXES:
        allowed = ", ".join(sorted(_ALLOWED_SUFFIXES))
        raise ValueError(f"cache path must end with one of {allowed}: {db_path!r}")
    if allowed_root is not None:
        root = Path(allowed_root).resolve()
        resolved = path.resolve()
        if resolved != root and root not in resolved.parents:
            raise ValueError(f"cache path {db_path!r} resolves outside of allowed root {allowed_root!r}")
    return path


class TranslationCache:
    """Normalized-query cache backed by SQLite.

    * The key covers the normalized source text, the RAG parameters that
      actually influence the answer and a ``namespace`` (model / adapter /
      generation settings fingerprint), so changing any of them can never
      return a stale translation.
    * Per-request metadata (``cache_hit``, ``latency_ms``) is never persisted.
    """

    def __init__(
        self,
        db_path: str = "./data/translation_cache.db",
        enabled: bool = True,
        ttl_hours: int = 168,
        max_entries: int = 100000,
        namespace: str = "",
        allowed_root: Optional[str] = None,
    ):
        self.db_path = db_path
        self.enabled = enabled
        self.ttl_hours = ttl_hours
        self.max_entries = max_entries
        self.namespace = namespace
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None

        if self.enabled:
            path = validate_cache_path(db_path, allowed_root)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(path), check_same_thread=False)
            self._init_db()

    def _init_db(self) -> None:
        assert self._conn is not None
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS translations (
                cache_key TEXT PRIMARY KEY,
                source_text TEXT NOT NULL,
                use_rag INTEGER NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            )
            """
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON translations(created_at)")
        self._conn.commit()

    def set_namespace(self, namespace: str) -> None:
        """Change the namespace mixed into every key (e.g. after a model/adapter change)."""
        self.namespace = namespace

    @staticmethod
    def _normalize(text: str) -> str:
        # Whitespace and Unicode-composition insensitive, but case *sensitive*:
        # "Turkey" (country) and "turkey" (bird) must not share a translation.
        return " ".join(unicodedata.normalize("NFC", text).split())

    def _make_key(
        self,
        text_en: str,
        use_rag: bool,
        top_k: int,
        score_threshold: Optional[float] = None,
    ) -> str:
        if not use_rag:
            # RAG parameters are irrelevant when retrieval is off.
            top_k, score_threshold = 0, None
        threshold = f"{score_threshold:.4f}" if score_threshold is not None else "default"
        raw = "\x1f".join(
            [self.namespace, self._normalize(text_en), f"rag={int(use_rag)}", f"k={top_k}", f"thr={threshold}"]
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _is_expired(self, created_at: str) -> bool:
        try:
            created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        except (ValueError, TypeError, AttributeError):
            return False  # unparsable timestamp: keep serving the entry
        age_hours = (datetime.now(timezone.utc) - created).total_seconds() / 3600
        return age_hours > self.ttl_hours

    def get(
        self,
        text_en: str,
        use_rag: bool,
        top_k: int,
        score_threshold: Optional[float] = None,
    ) -> Optional[dict]:
        if not self.enabled or self._conn is None:
            return None
        key = self._make_key(text_en, use_rag, top_k, score_threshold)
        with self._lock:
            row = self._conn.execute(
                "SELECT payload_json, created_at FROM translations WHERE cache_key = ?",
                (key,),
            ).fetchone()
            if row is None:
                return None
            payload_json, created_at = row
            data = None
            if not self._is_expired(created_at):
                try:
                    data = json.loads(payload_json)
                except json.JSONDecodeError:
                    data = None
            if data is None:  # expired or corrupted -> drop it
                self._conn.execute("DELETE FROM translations WHERE cache_key = ?", (key,))
                self._conn.commit()
                return None
        data["cache_hit"] = True
        return data

    def set(
        self,
        text_en: str,
        use_rag: bool,
        top_k: int,
        payload: dict,
        score_threshold: Optional[float] = None,
    ) -> None:
        if not self.enabled or self._conn is None:
            return
        key = self._make_key(text_en, use_rag, top_k, score_threshold)
        # cache_hit / latency_ms describe one request, not the translation itself.
        to_store = {k: v for k, v in payload.items() if k not in ("cache_hit", "latency_ms")}
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO translations (cache_key, source_text, use_rag, payload_json)
                VALUES (?, ?, ?, ?)
                """,
                (key, text_en.strip(), int(use_rag), json.dumps(to_store, ensure_ascii=False)),
            )
            self._conn.commit()
            self._maybe_evict()

    def _maybe_evict(self) -> None:
        """Remove the oldest entries if the cache exceeds ``max_entries``."""
        assert self._conn is not None
        count = self._conn.execute("SELECT COUNT(*) FROM translations").fetchone()[0]
        if count > self.max_entries:
            excess = count - self.max_entries
            self._conn.execute(
                """
                DELETE FROM translations WHERE cache_key IN (
                    SELECT cache_key FROM translations ORDER BY created_at ASC LIMIT ?
                )
                """,
                (excess,),
            )
            self._conn.commit()

    def invalidate(self, use_rag: Optional[bool] = None) -> int:
        """Delete cached entries; ``use_rag`` restricts it to RAG / non-RAG entries.

        Used when the translation memory changes: cached RAG answers may no longer
        reflect the best retrieval results. Returns the number of deleted entries.
        """
        if not self.enabled or self._conn is None:
            return 0
        with self._lock:
            if use_rag is None:
                cursor = self._conn.execute("DELETE FROM translations")
            else:
                cursor = self._conn.execute("DELETE FROM translations WHERE use_rag = ?", (int(use_rag),))
            self._conn.commit()
            return cursor.rowcount

    def clear(self) -> int:
        """Delete all cache entries. Returns the number of deleted entries."""
        return self.invalidate(None)

    def count(self) -> int:
        if not self.enabled or self._conn is None:
            return 0
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM translations").fetchone()[0])

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

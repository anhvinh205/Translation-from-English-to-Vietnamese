import threading
import time

import pytest

from src.rag.hybrid_retriever import LightweightBM25

PAIRS = [
    {"en": "Machine translation with QLoRA", "vi": "Dịch máy với QLoRA"},
    {"en": "Cooking soup recipe", "vi": "Công thức nấu súp"},
]


def test_bm25_indexes_incrementally_and_deduplicates():
    bm25 = LightweightBM25()
    assert bm25.index(PAIRS) == 2
    assert bm25.index(PAIRS) == 0  # nothing new
    assert len(bm25.corpus) == len(bm25.tokenized_docs) == len(bm25.tf_maps) == 2
    # whitespace / case variants are the same source sentence
    assert bm25.index([{"en": "  machine   TRANSLATION with qlora", "vi": "x"}]) == 0
    assert len(bm25.corpus) == 2


def test_bm25_statistics_are_not_inflated_by_duplicates():
    bm25 = LightweightBM25()
    bm25.index(PAIRS)
    before = (dict(bm25.doc_freqs), bm25.avgdl)
    bm25.index(PAIRS)
    assert (dict(bm25.doc_freqs), bm25.avgdl) == before


def test_bm25_keeps_the_latest_translation_like_the_vector_store():
    bm25 = LightweightBM25()
    bm25.index([{"en": "Hello there friend", "vi": "bản cũ"}])
    bm25.index([{"en": "hello   there FRIEND", "vi": "bản mới"}])
    assert len(bm25.corpus) == 1
    assert bm25.search("hello friend", top_k=1)[0]["vi"] == "bản mới"


def test_bm25_skips_incomplete_pairs_and_empty_queries():
    bm25 = LightweightBM25()
    assert bm25.index([{"en": "only english", "vi": ""}, {"en": "", "vi": "chỉ tiếng việt"}, {}]) == 0
    assert bm25.search("anything") == []
    bm25.index(PAIRS)
    assert bm25.search("zzz unknown words") == []


def test_bm25_clear_resets_everything():
    bm25 = LightweightBM25()
    bm25.index(PAIRS)
    bm25.clear()
    assert bm25.search("QLoRA") == []
    assert bm25.index(PAIRS) == 2


def test_bm25_is_thread_safe():
    bm25 = LightweightBM25()
    errors = []

    def writer(n):
        try:
            for i in range(50):
                bm25.index([{"en": f"sentence {n} number {i} about topic", "vi": "x"}])
        except Exception as exc:  # pragma: no cover - only on failure
            errors.append(exc)

    def reader():
        try:
            for _ in range(50):
                bm25.search("sentence topic number", top_k=3)
        except Exception as exc:  # pragma: no cover - only on failure
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(n,)) for n in range(3)] + [
        threading.Thread(target=reader) for _ in range(3)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len(bm25.corpus) == 150


# --------------------------------------------------------------------------- vector store
chromadb = pytest.importorskip("chromadb")


@pytest.fixture
def store(settings, fake_embeddings):
    from src.rag.vector_store import ChromaTranslationStore

    return ChromaTranslationStore(settings, embeddings=fake_embeddings)


def test_reindexing_is_idempotent_and_never_duplicates(store):
    pairs = [
        {"en": "Machine translation is fun", "vi": "Dịch máy rất vui"},
        {"en": "Cooking soup takes time", "vi": "Nấu súp cần thời gian"},
    ]
    assert store.add_pairs(pairs) == 2
    assert store.add_pairs(pairs) == 2
    assert store.count() == 2

    hits = store.dense_search("Machine translation is fun", k=5)
    assert [h["en"] for h in hits].count("Machine translation is fun") == 1


def test_duplicate_source_inside_one_call_is_accepted_and_last_translation_wins(store):
    # Chroma rejects duplicate ids within one upsert; the store must de-duplicate first.
    count = store.add_pairs(
        [
            {"en": "Same sentence here", "vi": "bản cũ"},
            {"en": "same  sentence HERE", "vi": "bản mới"},
        ]
    )
    assert count == 1
    assert store.count() == 1
    assert store.dense_search("Same sentence here", k=1)[0]["vi"] == "bản mới"


def test_incomplete_pairs_are_skipped(store):
    assert store.add_pairs([{"en": "only english", "vi": ""}, {"en": "", "vi": "chỉ tiếng việt"}]) == 0
    assert store.count() == 0


def test_dense_scores_are_bounded_and_ranked(store):
    store.add_pairs(
        [
            {"en": "Neural machine translation with transformers", "vi": "a"},
            {"en": "Completely unrelated cooking recipe for soup", "vi": "b"},
        ]
    )
    hits = store.dense_search("Neural machine translation with transformers", k=2)
    assert all(0.0 < h["dense_score"] <= 1.0 for h in hits)
    assert hits[0]["en"].startswith("Neural machine translation")
    assert hits[0]["dense_score"] >= hits[-1]["dense_score"]


def test_data_survives_a_restart(settings, fake_embeddings, store):
    from src.rag.vector_store import ChromaTranslationStore

    store.add_pairs([{"en": "Persistent sentence here", "vi": "Câu được lưu"}])
    reopened = ChromaTranslationStore(settings, embeddings=fake_embeddings)
    assert reopened.count() == 1


def test_indexing_errors_are_not_swallowed(store, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("chroma is down")

    monkeypatch.setattr(store.db, "add_documents", boom)
    with pytest.raises(RuntimeError, match="chroma is down"):
        store.add_pairs([{"en": "Some sentence here", "vi": "Một câu"}])


# --------------------------------------------------------------------------- thread safety
class OverlapDetector:
    """Fails like a HF fast tokenizer when two threads are inside the guarded call at once."""

    def __init__(self):
        self.active = 0
        self.overlaps = 0
        self._guard = threading.Lock()

    def __enter__(self):
        with self._guard:
            self.active += 1
            if self.active > 1:
                self.overlaps += 1
        time.sleep(0.001)
        if self.overlaps:
            raise RuntimeError("Already borrowed")
        return self

    def __exit__(self, *exc):
        with self._guard:
            self.active -= 1


def _run_threads(targets):
    errors = []

    def guarded(target):
        try:
            target()
        except Exception as exc:  # pragma: no cover - only on failure
            errors.append(exc)

    threads = [threading.Thread(target=guarded, args=(t,)) for t in targets]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return errors


def test_embedding_calls_are_serialized_across_threads(settings, fake_embeddings):
    from src.rag.vector_store import ChromaTranslationStore

    detector = OverlapDetector()

    class UnsafeEmbeddings(type(fake_embeddings)):
        def embed_documents(self, texts):
            with detector:
                return super().embed_documents(texts)

        def embed_query(self, text):
            with detector:
                return super().embed_query(text)

    store = ChromaTranslationStore(settings, embeddings=UnsafeEmbeddings())
    writers = [
        lambda n=n: [store.add_pairs([{"en": f"writer {n} sentence {i}", "vi": "x"}]) for i in range(5)]
        for n in range(3)
    ]
    readers = [lambda: [store.dense_search("writer sentence", k=3) for _ in range(5)] for _ in range(4)]

    assert _run_threads(writers + readers) == []
    assert detector.overlaps == 0


def test_cross_encoder_predictions_are_serialized_across_threads(monkeypatch, settings):
    import sys
    import types

    from src.rag.reranker import CrossEncoderReranker

    detector = OverlapDetector()

    class UnsafeCrossEncoder:
        def __init__(self, model_id):
            pass

        def predict(self, pairs):
            with detector:
                return [0.0 for _ in pairs]

    fake_module = types.ModuleType("sentence_transformers")
    fake_module.CrossEncoder = UnsafeCrossEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    reranker = CrossEncoderReranker(settings)

    candidates = [{"en": "a", "vi": "1"}, {"en": "b", "vi": "2"}]
    errors = _run_threads([lambda: [reranker.rerank("q", candidates) for _ in range(5)] for _ in range(6)])

    assert errors == []
    assert detector.overlaps == 0

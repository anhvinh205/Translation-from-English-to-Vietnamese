import logging
import sys
import types

from src.data.cleaner import deduplicate_pairs, extract_pair, is_valid_pair, unique_indices
from src.rag.hybrid_retriever import LightweightBM25, reciprocal_rank_fusion
from src.rag.reranker import CrossEncoderReranker


def test_cleaner_and_deduplication():
    en, vi = extract_pair({"english": "  Artificial intelligence ", "non_english": " Trí tuệ nhân tạo "})
    assert en == "Artificial intelligence"
    assert vi == "Trí tuệ nhân tạo"

    assert is_valid_pair(
        "Machine translation models require clean parallel data.",
        "Các mô hình dịch máy cần dữ liệu song ngữ sạch.",
    )
    assert not is_valid_pair("Visit https://example.com now", "Truy cập https://example.com ngay")
    assert not is_valid_pair("Same text here", "Same text here")

    dups = [
        {"en": "Hello world today", "vi": "Xin chào thế giới hôm nay"},
        {"en": "hello   world today", "vi": "xin chào thế giới hôm nay"},
    ]
    assert len(deduplicate_pairs(dups)) == 1


def test_unique_indices_keeps_first_occurrence_and_skips_empty():
    texts = ["Hello World", "hello   world", "", "Other", "OTHER ", "x"]
    assert unique_indices(texts) == [0, 3, 5]


def test_bm25_and_rrf_fusion():
    corpus = [
        {
            "en": "Parameter-efficient fine-tuning with QLoRA reduces GPU memory.",
            "vi": "Tinh chỉnh hiệu quả tham số với QLoRA giúp giảm bộ nhớ GPU.",
        },
        {
            "en": "Back-translation uses monolingual data to train machine translation.",
            "vi": "Dịch ngược sử dụng dữ liệu đơn ngữ để huấn luyện dịch máy.",
        },
    ]
    bm25 = LightweightBM25()
    bm25.index(corpus)
    hits = bm25.search("QLoRA GPU memory", top_k=2)
    assert len(hits) >= 1
    assert "QLoRA" in hits[0]["en"]

    dense_hits = [{"en": corpus[0]["en"], "vi": corpus[0]["vi"], "dense_score": 0.88}]
    fused = reciprocal_rank_fusion(dense_hits, hits)
    assert len(fused) >= 1
    assert fused[0]["hybrid_score"] > 0


def test_reranker_threshold_filtering(settings):
    reranker = CrossEncoderReranker(settings, use_neural_cross_encoder=False)
    assert reranker.mode == "lexical"
    assert reranker.degraded is False

    candidates = [
        {
            "en": "Neural machine translation with transformers",
            "vi": "Dịch máy nơ-ron với transformer",
            "dense_score": 0.9,
        },
        {
            "en": "Completely unrelated cooking recipe for soup",
            "vi": "Công thức nấu súp không liên quan",
            "dense_score": 0.05,
        },
    ]
    filtered = reranker.rerank("Neural machine translation with transformers", candidates, top_k=2, threshold=0.4)
    assert len(filtered) == 1
    assert "translation" in filtered[0]["en"]


def _install_fake_cross_encoder(monkeypatch, factory):
    fake_module = types.ModuleType("sentence_transformers")
    fake_module.CrossEncoder = factory
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)


def test_cross_encoder_load_failure_is_logged_and_flagged(monkeypatch, caplog, settings):
    def broken(*args, **kwargs):
        raise RuntimeError("download failed")

    _install_fake_cross_encoder(monkeypatch, broken)
    logger = logging.getLogger("src.rag.reranker")
    logger.addHandler(caplog.handler)  # the project logger does not propagate to the root logger
    try:
        with caplog.at_level(logging.WARNING, logger="src.rag.reranker"):
            reranker = CrossEncoderReranker(settings, use_neural_cross_encoder=True)
    finally:
        logger.removeHandler(caplog.handler)

    assert reranker.degraded is True
    assert reranker.mode == "lexical-fallback"
    assert any("download failed" in record.getMessage() for record in caplog.records)


def test_cross_encoder_scores_are_squashed_and_ranked(monkeypatch, settings):
    class FakeCrossEncoder:
        def __init__(self, model_id):
            self.model_id = model_id

        def predict(self, pairs):
            return [3.0, -3.0, 0.0]

    _install_fake_cross_encoder(monkeypatch, FakeCrossEncoder)
    reranker = CrossEncoderReranker(settings)
    assert reranker.mode == "cross-encoder"

    candidates = [{"en": "a", "vi": "1"}, {"en": "b", "vi": "2"}, {"en": "c", "vi": "3"}]
    ranked = reranker.rerank("q", candidates, top_k=5, threshold=0.4)
    assert [c["en"] for c in ranked] == ["a", "c"]  # "b" falls below the threshold
    assert 0.9 < ranked[0]["rerank_score"] < 1.0
    assert ranked[1]["rerank_score"] == 0.5

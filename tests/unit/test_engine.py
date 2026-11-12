import os
import time

import pytest

from src.core.errors import EngineBusyError, InvalidInputError
from src.rag.engine import HybridRAGTranslationEngine
from src.rag.reranker import CrossEncoderReranker

SENTENCE = "Low-resource machine translation uses parallel corpora."
PAIR = {"en": SENTENCE, "vi": "Dịch máy ít tài nguyên sử dụng kho dữ liệu song ngữ."}
LONGER_QUERY = SENTENCE[:-1] + " and many extra unrelated words"  # only partly similar to PAIR


class WordTokenizer:
    """Counts whitespace-separated words as tokens (enough to test the prompt budget)."""

    def __call__(self, text, **kwargs):
        return {"input_ids": list(range(len(text.split())))}


def _examples(n, words=60):
    return [
        {
            "en": " ".join(f"en{i}w{j}" for j in range(words)),
            "vi": " ".join(f"vi{i}w{j}" for j in range(words)),
            "rerank_score": 0.9 - i / 100,
        }
        for i in range(n)
    ]


def test_translate_returns_a_payload_and_serves_repeats_from_the_cache(engine):
    first = engine.translate(SENTENCE, use_rag=False)
    assert first["translation"].startswith("[Mock Translation]")
    assert first["cache_hit"] is False
    assert first["active_adapter"] == "zero-shot"

    second = engine.translate(SENTENCE, use_rag=False)
    assert second["cache_hit"] is True
    assert second["translation"] == first["translation"]


def test_latency_is_real_and_cache_hits_report_their_own_latency(engine, monkeypatch):
    original = engine._fit_prompt

    def slow(*args, **kwargs):
        time.sleep(0.03)
        return original(*args, **kwargs)

    monkeypatch.setattr(engine, "_fit_prompt", slow)
    first = engine.translate(SENTENCE, use_rag=False)
    assert first["latency_ms"] >= 25  # regression: it used to be reported as 0.0

    second = engine.translate(SENTENCE, use_rag=False)
    assert second["cache_hit"] is True
    assert second["latency_ms"] < 25  # what the lookup cost, not the original generation


def test_score_threshold_is_part_of_the_cache_key(engine):
    engine.index_memory([PAIR])
    low = engine.translate(LONGER_QUERY, use_rag=True, top_k=2, score_threshold=0.05)
    assert low["cache_hit"] is False
    assert low["rag_triggered"] is True

    high = engine.translate(LONGER_QUERY, use_rag=True, top_k=2, score_threshold=0.9)
    assert high["cache_hit"] is False  # not the answer cached for the lower threshold
    assert high["rag_triggered"] is False


def test_rag_uses_the_translation_memory_and_reports_the_examples_it_used(engine):
    engine.index_memory([PAIR])
    result = engine.translate(SENTENCE, use_rag=True, top_k=2, score_threshold=0.1)
    assert result["rag_triggered"] is True
    assert result["retrieved_memory"][0]["vi"] == PAIR["vi"]
    assert result["retrieved_memory"][0]["rerank_score"] > 0


def test_indexing_new_memory_invalidates_cached_rag_answers_only(engine):
    before = engine.translate(SENTENCE, use_rag=True, score_threshold=0.1)
    assert before["rag_triggered"] is False  # the memory is still empty
    engine.translate(SENTENCE, use_rag=False)

    engine.index_memory([PAIR])

    after = engine.translate(SENTENCE, use_rag=True, score_threshold=0.1)
    assert after["cache_hit"] is False
    assert after["rag_triggered"] is True
    assert engine.translate(SENTENCE, use_rag=False)["cache_hit"] is True  # RAG-off entries survive


def test_use_cache_false_bypasses_the_cache_completely(engine):
    engine.translate(SENTENCE, use_rag=False)
    fresh = engine.translate(SENTENCE, use_rag=False, use_cache=False)
    assert fresh["cache_hit"] is False
    assert engine.cache.count() == 1  # nothing extra was written


def test_vector_store_outage_degrades_to_bm25(engine):
    engine.index_memory([PAIR])
    engine.vector_store.fail = True
    result = engine.translate(SENTENCE, use_rag=True, score_threshold=0.1, use_cache=False)
    assert result["rag_triggered"] is True  # BM25 still finds the pair


def test_retrieval_failure_falls_back_to_zero_shot(engine, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("reranker exploded")

    monkeypatch.setattr(engine, "retrieve_context", boom)
    result = engine.translate(SENTENCE, use_rag=True, use_cache=False)
    assert result["rag_triggered"] is False
    assert result["translation"]


def test_input_that_is_blank_after_sanitization_is_rejected(engine):
    with pytest.raises(InvalidInputError):
        engine.translate("<|im_start|><|im_end|>   ")


def test_prompt_budget_drops_the_lowest_ranked_examples_not_the_tail(make_settings, make_engine):
    engine = make_engine(make_settings(inference={"max_prompt_tokens": 400}))
    engine.tokenizer = WordTokenizer()
    query = "Please translate this specific sentence"
    examples = _examples(6)

    prompt, used = engine._fit_prompt(query, examples)

    assert 0 < len(used) < 6
    assert used == examples[: len(used)]  # the best-ranked examples survive
    assert f"EN: {query}" in prompt
    assert prompt.endswith("<|im_start|>assistant\n")
    assert engine._count_tokens(prompt) <= 400


def test_prompt_budget_rejects_inputs_that_cannot_fit(make_settings, make_engine):
    engine = make_engine(make_settings(inference={"max_prompt_tokens": 200}))
    engine.tokenizer = WordTokenizer()
    with pytest.raises(InvalidInputError):
        engine._fit_prompt("word " * 1000, [])


def test_without_a_tokenizer_all_examples_are_kept(engine):
    prompt, used = engine._fit_prompt("hello", _examples(3))
    assert len(used) == 3
    assert prompt.endswith("<|im_start|>assistant\n")


def test_cache_is_not_shared_between_different_models_or_generation_settings(make_settings, make_engine, tmp_path):
    shared = {"cache": {"sqlite_path": str(tmp_path / "shared.db")}}
    first = make_engine(make_settings(**shared))
    first.translate(SENTENCE, use_rag=False)
    assert first.translate(SENTENCE, use_rag=False)["cache_hit"] is True

    other_model = make_engine(make_settings(model={"base_model_id": "org/another-model"}, **shared))
    assert other_model.translate(SENTENCE, use_rag=False)["cache_hit"] is False

    other_beams = make_engine(make_settings(inference={"num_beams": 5}, **shared))
    assert other_beams.translate(SENTENCE, use_rag=False)["cache_hit"] is False

    same_again = make_engine(make_settings(**shared))
    assert same_again.translate(SENTENCE, use_rag=False)["cache_hit"] is True


def test_adapter_fingerprint_changes_when_the_adapter_is_retrained(tmp_path):
    (tmp_path / "adapter_config.json").write_text("{}")
    weights = tmp_path / "adapter_model.safetensors"
    weights.write_bytes(b"v1")
    fingerprint = HybridRAGTranslationEngine._fingerprint_adapter
    before = fingerprint(str(tmp_path))
    assert before == fingerprint(str(tmp_path))  # stable while nothing changes

    weights.write_bytes(b"version-2-is-longer")
    os.utime(weights, ns=(1, 1))
    assert fingerprint(str(tmp_path)) != before


def test_health_reports_a_degraded_reranker(make_settings, make_engine):
    settings = make_settings()
    reranker = CrossEncoderReranker(settings, use_neural_cross_encoder=False)
    reranker.degraded = True
    health = make_engine(settings, reranker=reranker).health()
    assert health["degraded"] is True
    assert health["reranker"] == "lexical-fallback"
    assert health["vector_store_count"] == 0


def test_generation_is_rejected_when_all_slots_are_busy(make_settings, make_engine):
    pytest.importorskip("torch")
    engine = make_engine(make_settings(inference={"queue_timeout_seconds": 0.05}))
    assert engine._generation_slots.acquire(blocking=False)  # occupy the only slot
    with pytest.raises(EngineBusyError):
        engine._generate("prompt")

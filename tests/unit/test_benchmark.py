import types

from datasets import Dataset

from src.evaluation import benchmark

EVAL_ROWS = [{"en": f"english {i}", "vi": f"tiếng việt {i}"} for i in range(6)]
TM_ROWS = [{"en": f"memory {i}", "vi": f"bộ nhớ {i}"} for i in range(3)]


class ExactMatchMetric:
    def compute(self, predictions, references, **kwargs):
        hits = sum(p == r[0] for p, r in zip(predictions, references))
        return {"score": 100.0 * hits / len(predictions)}


class FakeEngine:
    """Answers correctly only when RAG is on, and records how it was called."""

    active_adapter = "fake"
    instances = []

    def __init__(self, settings, load_llm):
        self.translate_calls = []
        self.indexed = None
        FakeEngine.instances.append(self)

    def index_memory(self, pairs):
        self.indexed = pairs
        return len(pairs)

    def translate(self, text_en, use_rag=True, use_cache=True, **kwargs):
        self.translate_calls.append({"text": text_en, "use_rag": use_rag, "use_cache": use_cache})
        answer = {r["en"]: r["vi"] for r in EVAL_ROWS}[text_en]
        return {"translation": answer if use_rag else "sai"}


def test_the_benchmark_measures_the_model_not_the_cache(monkeypatch, settings):
    FakeEngine.instances.clear()
    monkeypatch.setattr(
        benchmark, "load_wikimatrix_splits", lambda s: (None, Dataset.from_list(TM_ROWS), Dataset.from_list(EVAL_ROWS))
    )
    monkeypatch.setattr(benchmark, "HybridRAGTranslationEngine", FakeEngine)
    monkeypatch.setattr(benchmark, "evaluate", types.SimpleNamespace(load=lambda name: ExactMatchMetric()))

    report = benchmark.evaluate_system(num_samples=4, settings=settings)

    engine = FakeEngine.instances[0]
    assert engine.indexed == [{"en": r["en"], "vi": r["vi"]} for r in TM_ROWS]
    # a cached answer would make the numbers (and the latency) meaningless
    assert len(engine.translate_calls) == 8  # 4 samples x (without RAG, with RAG)
    assert all(call["use_cache"] is False for call in engine.translate_calls)
    assert [c["use_rag"] for c in engine.translate_calls] == [False] * 4 + [True] * 4

    assert report["samples_evaluated"] == 4
    assert report["active_adapter"] == "fake"
    assert report["without_rag"]["sacrebleu"] == 0.0
    assert report["with_hybrid_rag_reranker"]["sacrebleu"] == 100.0
    assert report["delta"]["sacrebleu_gain"] == 100.0
    assert report["without_rag"]["avg_latency_ms"] >= 0

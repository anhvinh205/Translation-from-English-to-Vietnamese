"""Multi-metric evaluation suite: SacreBLEU, chrF++, Latency (ms), and Throughput."""

import json
import time
from typing import Optional

import evaluate

from src.core.settings import AppSettings, load_settings
from src.data.ingestion import load_wikimatrix_splits
from src.rag.engine import HybridRAGTranslationEngine


def evaluate_system(num_samples: int = 100, settings: Optional[AppSettings] = None) -> dict:
    settings = settings or load_settings()
    _, tm_ds, eval_ds = load_wikimatrix_splits(settings)
    test_subset = eval_ds.select(range(min(num_samples, len(eval_ds))))

    bleu_metric = evaluate.load("sacrebleu")
    chrf_metric = evaluate.load("chrf")
    references = [[row["vi"]] for row in test_subset]

    engine = HybridRAGTranslationEngine(settings=settings, load_llm=True)
    tm_pairs = [{"en": r["en"], "vi": r["vi"]} for r in tm_ds.select(range(min(5000, len(tm_ds))))]
    engine.index_memory(tm_pairs)

    # use_cache=False: the benchmark must measure the current model, not answers cached by an
    # earlier run (and latency must be the real generation latency, not a cache lookup).
    # 1. Zero-shot / SFT without RAG
    t0 = time.perf_counter()
    preds_no_rag = [engine.translate(row["en"], use_rag=False, use_cache=False)["translation"] for row in test_subset]
    latency_no_rag = (time.perf_counter() - t0) * 1000 / max(1, len(test_subset))

    # 2. Hybrid RAG + Cross-Encoder Reranker
    t1 = time.perf_counter()
    preds_hybrid_rag = [
        engine.translate(row["en"], use_rag=True, use_cache=False)["translation"] for row in test_subset
    ]
    latency_rag = (time.perf_counter() - t1) * 1000 / max(1, len(test_subset))

    bleu_no_rag = bleu_metric.compute(predictions=preds_no_rag, references=references)["score"]
    chrf_no_rag = chrf_metric.compute(predictions=preds_no_rag, references=references, word_order=2)["score"]

    bleu_rag = bleu_metric.compute(predictions=preds_hybrid_rag, references=references)["score"]
    chrf_rag = chrf_metric.compute(predictions=preds_hybrid_rag, references=references, word_order=2)["score"]

    report = {
        "dataset": f"{settings.dataset.name} ({settings.dataset.subset})",
        "samples_evaluated": len(test_subset),
        "active_adapter": engine.active_adapter,
        "without_rag": {
            "sacrebleu": round(bleu_no_rag, 2),
            "chrf_plus_plus": round(chrf_no_rag, 2),
            "avg_latency_ms": round(latency_no_rag, 2),
        },
        "with_hybrid_rag_reranker": {
            "sacrebleu": round(bleu_rag, 2),
            "chrf_plus_plus": round(chrf_rag, 2),
            "avg_latency_ms": round(latency_rag, 2),
        },
        "delta": {
            "sacrebleu_gain": round(bleu_rag - bleu_no_rag, 2),
            "chrf_gain": round(chrf_rag - chrf_no_rag, 2),
        },
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    evaluate_system()

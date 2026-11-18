"""Direct terminal translation testing script for the trained DPO model."""

import os
import sys

# Ensure imports from workspace root
sys.path.insert(0, os.path.abspath("."))

from src.rag.engine import HybridRAGTranslationEngine


def main():
    print("=" * 65)
    print("🚀 Initializing Translation Engine (Qwen2.5-1.5B + DPO LoRA)...")
    print("=" * 65)

    engine = HybridRAGTranslationEngine(load_llm=True)
    print("\n✅ Model and components loaded successfully!")
    print(f"👉 Active Adapter : {engine.active_adapter}")
    print(f"👉 Reranker       : {engine.reranker.mode}")
    print(f"👉 TM Records     : {engine.vector_store.count()} sentence pairs in ChromaDB")
    print("=" * 65)

    test_samples = [
        "Hello, how are you today?",
        "Machine translation has advanced significantly with large language models.",
        "Low-resource machine translation can be improved with hybrid retrieval-augmented generation.",
    ]

    print("\n--- 🔍 BENCHMARKING PRESET SAMPLE SENTENCES ---")
    for text in test_samples:
        print(f"\n🇬🇧 EN: {text}")
        res = engine.translate(text, use_rag=True)
        print(f"🇻🇳 VI: {res['translation']}")
        print(f"⏱️ Latency: {res['latency_ms']} ms | RAG triggered: {res['rag_triggered']}")
        if res.get("retrieved_memory"):
            print("   📚 Similar sentence pairs retrieved from Translation Memory:")
            for m in res["retrieved_memory"][:2]:
                print(f"      - EN: {m['en']} -> VI: {m['vi']} (Rerank score: {m.get('rerank_score', 0)})")

    print("\n" + "=" * 65)
    print("💬 INTERACTIVE TRANSLATION MODE (Type 'exit' or 'quit' to exit)")
    print("=" * 65)
    while True:
        try:
            query = input("\nEnter English text to translate: ").strip()
            if not query or query.lower() in ("exit", "quit", "q"):
                print("Exiting interactive mode.")
                break
            res = engine.translate(query, use_rag=True)
            print(f"👉 Translation (VI): {res['translation']}")
            print(f"⏱️ Latency: {res['latency_ms']} ms | Cache hit: {res['cache_hit']}")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting interactive mode.")
            break


if __name__ == "__main__":
    main()

"""Translation engine: cache -> hybrid retrieval (dense + BM25) -> rerank -> (Q)LoRA LLM."""

import hashlib
import os
import threading
import time
from typing import Any, Optional

from src.core.cache import TranslationCache
from src.core.errors import EngineBusyError, InvalidInputError, ModelOutOfMemoryError
from src.core.logger import get_logger, track_latency
from src.core.settings import AppSettings, load_settings
from src.data.prompt_builder import build_sft_or_rag_prompt, sanitize_chatml
from src.rag.hybrid_retriever import LightweightBM25, reciprocal_rank_fusion
from src.rag.reranker import CrossEncoderReranker


class HybridRAGTranslationEngine:
    """Semantic cache + hybrid BM25/dense RAG + reranker + QLoRA LLM.

    Heavy dependencies (torch, transformers, peft, chromadb, sentence-transformers) are
    imported lazily, and the vector store / reranker / cache can be injected, so the
    engine can be unit-tested without downloading any model.
    """

    def __init__(
        self,
        settings: Optional[AppSettings] = None,
        load_llm: bool = True,
        *,
        vector_store: Optional[Any] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        cache: Optional[TranslationCache] = None,
        bm25: Optional[LightweightBM25] = None,
    ):
        self.settings = settings or load_settings()
        self.logger = get_logger("rag_engine")
        cache_cfg = self.settings.cache
        self.cache = cache or TranslationCache(
            db_path=cache_cfg.sqlite_path,
            enabled=cache_cfg.enabled,
            ttl_hours=cache_cfg.ttl_hours,
            max_entries=cache_cfg.max_entries,
            allowed_root=cache_cfg.allowed_root,
        )
        if vector_store is None:
            from src.rag.vector_store import ChromaTranslationStore

            vector_store = ChromaTranslationStore(self.settings)
        self.vector_store = vector_store
        self.bm25 = bm25 or LightweightBM25()
        self.reranker = reranker or CrossEncoderReranker(self.settings)

        self.tokenizer = None
        self.model = None
        self.active_adapter = "zero-shot"
        self._adapter_fingerprint = "none"
        self._generation_slots = threading.BoundedSemaphore(self.settings.inference.max_concurrent_generations)
        # HF *fast* tokenizers are not thread-safe: concurrent calls (FastAPI runs sync handlers in a
        # thread pool) fail with "RuntimeError: Already borrowed", so every use of it is serialized.
        self._tokenizer_lock = threading.Lock()

        if load_llm:
            self._init_llm()
        self.cache.set_namespace(self._cache_namespace())

    # ------------------------------------------------------------------ model
    @staticmethod
    def _fingerprint_adapter(adapter_dir: str) -> str:
        """Cheap identity of an adapter directory: changes whenever it is re-saved."""
        parts = []
        for name in ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"):
            path = os.path.join(adapter_dir, name)
            if os.path.exists(path):
                st = os.stat(path)
                parts.append(f"{name}:{st.st_size}:{st.st_mtime_ns}")
        return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:12]

    def _init_llm(self) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_id = self.settings.model.base_model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # Safety net only (the prompt is budgeted before tokenization): if anything ever had
        # to be cut, drop the oldest context, never the query and the assistant header.
        self.tokenizer.truncation_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.settings.device == "cuda" else torch.float32,
            device_map="auto" if self.settings.device == "cuda" else None,
        )

        dpo_dir = self.settings.model.dpo_output_dir
        sft_dir = self.settings.model.lora_output_dir
        if os.path.exists(os.path.join(dpo_dir, "adapter_config.json")):
            adapter_dir, self.active_adapter = dpo_dir, "dpo-qlora"
        elif os.path.exists(os.path.join(sft_dir, "adapter_config.json")):
            adapter_dir, self.active_adapter = sft_dir, "sft-qlora"
        else:
            adapter_dir, self.active_adapter = None, "zero-shot"

        if adapter_dir is not None:
            self.model = PeftModel.from_pretrained(base_model, adapter_dir)
            self._adapter_fingerprint = self._fingerprint_adapter(adapter_dir)
        else:
            self.model = base_model

        self.model.eval()
        self.logger.info("LLM loaded: %s (adapter: %s)", model_id, self.active_adapter)

    def _cache_namespace(self) -> str:
        """Fingerprint of everything that changes the output for the same input."""
        cfg = self.settings
        hybrid = cfg.hybrid_search
        parts = [
            "llm" if self.model is not None else "mock",
            cfg.model.base_model_id,
            self.active_adapter,
            self._adapter_fingerprint,
            f"beams={cfg.inference.num_beams}",
            f"new={cfg.dataset.max_new_tokens}",
            f"ptok={cfg.inference.max_prompt_tokens}",
            f"rerank={self.reranker.mode}:{cfg.reranker.model_id}",
            f"hybrid={hybrid.top_k_dense},{hybrid.top_k_bm25},{hybrid.dense_weight},{hybrid.bm25_weight}",
        ]
        return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]

    # -------------------------------------------------------------- retrieval
    def index_memory(self, pairs: list[dict]) -> int:
        """Add pairs to the translation memory (idempotent: safe to retry after a failure)."""
        count = self.vector_store.add_pairs(pairs)
        self.bm25.index(pairs)
        if count:
            # Cached RAG answers may now be missing a better example.
            self.cache.invalidate(use_rag=True)
        return count

    def retrieve_context(
        self,
        query_en: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> list[dict]:
        try:
            dense_hits = self.vector_store.dense_search(query_en, k=self.settings.hybrid_search.top_k_dense)
        except Exception:
            self.logger.error("Dense search failed, continuing with BM25 only", exc_info=True)
            dense_hits = []

        bm25_hits = self.bm25.search(query_en, top_k=self.settings.hybrid_search.top_k_bm25)

        fused = reciprocal_rank_fusion(
            dense_hits,
            bm25_hits,
            dense_weight=self.settings.hybrid_search.dense_weight,
            bm25_weight=self.settings.hybrid_search.bm25_weight,
        )
        return self.reranker.rerank(query_en, fused, top_k=top_k, threshold=score_threshold)

    # ------------------------------------------------------------- generation
    def _count_tokens(self, text: str) -> int:
        with self._tokenizer_lock:
            return len(self.tokenizer(text)["input_ids"])

    def _fit_prompt(self, text_en: str, retrieved: list[dict]) -> tuple[str, list[dict]]:
        """Build the prompt within the token budget by dropping the lowest-ranked examples.

        The budget is enforced by removing retrieved examples, never by cutting the tail:
        the sentence to translate and the assistant header must always reach the model.
        """
        examples = list(retrieved)
        prompt = build_sft_or_rag_prompt(text_en, few_shot_examples=examples)
        if self.tokenizer is None:
            return prompt, examples

        budget = self.settings.inference.max_prompt_tokens
        n_tokens = self._count_tokens(prompt)
        while examples and n_tokens > budget:
            examples.pop()
            prompt = build_sft_or_rag_prompt(text_en, few_shot_examples=examples)
            n_tokens = self._count_tokens(prompt)
        if n_tokens > budget:
            raise InvalidInputError(f"The input is too long for the model ({n_tokens} tokens, limit {budget}).")
        return prompt, examples

    def _generate(self, prompt: str) -> str:
        import torch

        inference = self.settings.inference
        if not self._generation_slots.acquire(timeout=inference.queue_timeout_seconds):
            raise EngineBusyError("All generation slots are busy, please retry shortly.")
        try:
            with self._tokenizer_lock:
                encoded = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=inference.max_prompt_tokens,
                )
            inputs = encoded.to(self.model.device)
            generation_kwargs: dict[str, Any] = {
                "max_new_tokens": self.settings.dataset.max_new_tokens,
                "num_beams": inference.num_beams,
                "pad_token_id": self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else self.tokenizer.eos_token_id,
            }
            if inference.num_beams > 1:
                generation_kwargs["early_stopping"] = True
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **generation_kwargs)
            new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
            with self._tokenizer_lock:
                return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            raise ModelOutOfMemoryError("The model ran out of GPU memory.") from exc
        finally:
            self._generation_slots.release()

    def translate(
        self,
        text_en: str,
        use_rag: bool = True,
        top_k: int = 3,
        score_threshold: Optional[float] = None,
        use_cache: bool = True,
    ) -> dict:
        started = time.perf_counter()
        if not sanitize_chatml(text_en).strip():
            raise InvalidInputError("The input text is empty.")
        # Resolve the effective threshold so the cache key never hides a changed default.
        threshold = score_threshold if score_threshold is not None else self.settings.reranker.score_threshold

        if use_cache:
            cached = self.cache.get(text_en, use_rag=use_rag, top_k=top_k, score_threshold=threshold)
            if cached is not None:
                # Report what *this* request cost, not what the original generation cost.
                cached["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
                return cached

        with track_latency(self.logger, "end_to_end_translation") as perf:
            retrieved: list[dict] = []
            if use_rag:
                try:
                    retrieved = self.retrieve_context(text_en, top_k=top_k, score_threshold=threshold)
                except Exception:
                    # Graceful degradation: translate without retrieved examples.
                    self.logger.error("RAG retrieval failed, falling back to zero-shot", exc_info=True)

            prompt, retrieved = self._fit_prompt(text_en, retrieved)

            if self.model is not None and self.tokenizer is not None:
                translation = self._generate(prompt)
            else:
                translation = f"[Mock Translation] {text_en}"

        # Built AFTER the `with` block so perf["latency_ms"] is populated.
        payload = {
            "source": text_en,
            "translation": translation,
            "retrieved_memory": retrieved,
            "rag_triggered": len(retrieved) > 0,
            "active_adapter": self.active_adapter,
            "latency_ms": perf["latency_ms"],
            "cache_hit": False,
        }
        if use_cache:
            self.cache.set(text_en, use_rag=use_rag, top_k=top_k, payload=payload, score_threshold=threshold)
        return payload

    # ------------------------------------------------------------------ misc
    def health(self) -> dict:
        return {
            "reranker": self.reranker.mode,
            "degraded": bool(getattr(self.reranker, "degraded", False)),
            "vector_store_count": self.vector_store.count(),
            "cache_entries": self.cache.count(),
        }

    def close(self) -> None:
        self.cache.close()

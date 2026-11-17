# Code Review Audit → Fixes & Verification Report

Source: Code review report (27 items: 1 prompt injection, 8 bugs, 7 security vulnerabilities, 11 architectural improvements).
The test suite consists of **165+ automated tests** (unit, integration, and small-model end-to-end), passing clean with `ruff` linting across Python 3.10 and 3.11.

Items marked with ◆ were counter-tested (removing the fix causes the test to fail).

---

## 1. Prompt Injection

| Item | Resolution | Source File | Verification Test |
|---|---|---|---|
| ChatML Prompt Injection | `sanitize_chatml` removes all `<|...|>` markers and loops until reaching a stable fixpoint (single-pass sanitization was previously bypassable via nested markers like `<|im_<|im_start|>start|>`); applied to source inputs, TM exemplars, target translations, and DPO pairs | `src/data/prompt_builder.py` | `test_prompt_builder.py`, `test_fastapi_endpoints.py` (via `/memory/index`) ◆ |

---

## 2. Bugs

| Item | Resolution | Source File | Verification Test |
|---|---|---|---|
| #1 `latency_ms` always 0.0 | Payload is constructed *after* the `track_latency` context exits | `src/rag/engine.py` | `test_engine.py` |
| #2 Cache ignores `score_threshold` | Cache key includes the *effective* threshold (including defaults) and `top_k`; when RAG is disabled, RAG parameters are omitted from key | `src/core/cache.py`, `src/rag/engine.py` | `test_cache.py`, `test_engine.py` |
| #3 Inaccurate latency on cache hit | Cache does not persist `latency_ms`/`cache_hit`; cache hit returns the measurement of its own lookup time | `src/core/cache.py`, `src/rag/engine.py` | `test_engine.py` |
| #4 BM25 full reindexing on updates | Incremental index updates, deduplicating on source text, keeping the latest translation (matching Chroma upsert), guarded by thread locks | `src/rag/hybrid_retriever.py` | `test_retrieval.py` |
| #5 Weak DPO rejected candidates | 5 types of hard negatives (sentence truncation, word swapping, word repetition, dropped function words, terminology corruption); local seeded RNG; guaranteed divergence from `chosen` | `src/training/dpo_align.py` | `test_training.py` |
| #6 DPO training failure | `run_dpo_training()` uses TRL; continues from SFT adapter and uses a **frozen copy of the SFT adapter as reference** (preventing degenerate KL collapse against base model); filters oversized samples | `src/training/dpo_align.py`, `requirements.txt` | `tests/slow/test_training_e2e.py` |
| #7 SFT calculates loss over prompt | Custom collator masks prompt and padding tokens (`-100`); samples missing delimiters are skipped; **`remove_unused_columns=False`** configured; oversized samples discarded rather than truncated; evaluation dataset + `load_best_model_at_end` | `src/training/sft_qlora.py`, `src/training/utils.py` | `test_training.py`, `tests/slow/test_training_e2e.py` ◆ |
| #8 CD workflow only builds UI | GitHub Actions CD builds and tests both API and UI images, runs smoke tests, and publishes to GHCR | `.github/workflows/cd-docker.yml` | `actionlint` |

---

## 3. Security

| Item | Resolution | Source File | Verification Test |
|---|---|---|---|
| Wildcard CORS `*` | Configured strictly from `settings.cors.allowed_origins`; explicitly allows `X-API-Key` headers | `src/api/main.py`, `src/core/settings.py` | `test_fastapi_endpoints.py` |
| Rate Limiting | Implemented via `slowapi`: individual limits for `/translate`, `/memory/index`, admin, and default endpoints; returns `429` with `Retry-After` header | `src/api/main.py`, `src/api/routes.py` | `test_fastapi_endpoints.py` |
| Authentication | Constant-time `X-API-Key` (and `Bearer`) validation; `API_KEYS` for translation, `ADMIN_API_KEYS` for TM writes and cache deletion; minimum 12-character key length, never exposed in logs or repr | `src/api/security.py` | `test_fastapi_endpoints.py`, `test_settings.py` |
| Container root privilege | API & UI Dockerfiles run under unprivileged `USER 10001` (numeric ID for Kubernetes `runAsNonRoot` validation) | `docker/*` | `hadolint` |
| Unbounded input validation | Pydantic schema validation: `text_en` ≤ 5000 chars, TM pairs ≤ 2000 chars, ≤ 10,000 batch pairs, `top_k` ≤ 10, blank strings rejected; Engine enforces prompt token budget by **dropping lowest-ranked RAG exemplars instead of truncating input** | `src/api/schemas.py`, `src/rag/engine.py`, `src/data/prompt_builder.py` | `test_engine.py`, `tests/slow` |
| Silently caught CrossEncoder errors | Emits explicit warning logs with stack traces, sets `reranker.degraded=True`, and reports `degraded` status in `/health` | `src/rag/reranker.py`, `src/api/routes.py` | `test_data_and_rag.py` |
| SQLite directory traversal | `validate_cache_path` rejects `..`, invalid extensions, NUL bytes, and symlinks traversing outside `cache.allowed_root` | `src/core/cache.py`, `src/core/settings.py` | `test_cache.py`, `test_settings.py` |

---

## 4. Key Behavioral Improvements

1. **Error Handling:** Centralized exception handlers for domain errors (`InvalidInputError` → 422, `EngineBusyError`/`ModelOutOfMemoryError` → 503 with `Retry-After`); 500 responses return an opaque `error_id`; RAG errors automatically degrade to zero-shot translation.
2. **Resilient Streamlit UI:** `ui/api_client.py` captures all network and HTTP errors (401/403/422/429/503/500, disconnects, non-JSON responses) and displays clear English alerts without crashing or dumping raw tracebacks.
3. **Cache Eviction & Namespacing:** SQLite cache is scoped by model, adapter, reranker, and generation parameters. Registering new Translation Memory pairs selectively purges RAG-dependent cached answers.
4. **Concurrency & Concurrency Limiting:** FastAPI endpoints remain standard `def` running inside thread pools to avoid blocking the event loop; engine imposes `max_concurrent_generations` with queued timeouts.
5. **Thread Safety:** Mutual exclusion locks protect Hugging Face tokenizers, embeddings, Cross-Encoder rerankers, and ChromaDB batch operations against race conditions under concurrent requests.
6. **CPU Inference Latency Optimization:** `num_beams` configured to 1 (greedy decoding) reducing CPU translation latency by ~66% with near-identical translation fidelity.

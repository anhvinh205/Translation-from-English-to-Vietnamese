# Đối chiếu báo cáo review → thay đổi → cách kiểm chứng

Nguồn: `code-review-report.md` (27 mục: 1 prompt injection, 8 bug, 7 bảo mật, 11 cải tiến).
Bộ test hiện có **174 test** (unit, integration, e2e model nhỏ), lint `ruff` sạch, tất cả chạy trên Python 3.10 với đúng các phiên bản trong `requirements*.txt`.

Cách đọc cột cuối: tên file test chứa kiểm chứng tự động. Những mục đánh dấu ◆ đã được thử *đối chứng* (gỡ bản sửa ra thì test phải thất bại).

## 1. Prompt injection

| Mục | Cách xử lý | Code | Kiểm chứng |
|---|---|---|---|
| Prompt injection ChatML | `sanitize_chatml` loại mọi marker `<\|…\|>` và **lặp tới khi ổn định** (bản lọc một lượt vẫn bị vượt qua bằng `<\|im_<\|im_start\|>start\|>`); áp dụng cho câu nguồn, ví dụ TM, bản dịch đích và mẫu DPO | `src/data/prompt_builder.py` | `test_prompt_builder.py`, `test_fastapi_endpoints.py` (đầu cuối qua `/memory/index`) ◆ |

## 2. Bugs

| Mục | Cách xử lý | Code | Kiểm chứng |
|---|---|---|---|
| #1 `latency_ms` luôn 0.0 | payload dựng *sau* khối `track_latency` | `src/rag/engine.py` | `test_engine.py` |
| #2 cache bỏ qua `score_threshold` | key gồm ngưỡng *hiệu dụng* (cả khi dùng mặc định) và `top_k`; khi tắt RAG thì bỏ qua tham số RAG | `src/core/cache.py`, `engine.py` | `test_cache.py`, `test_engine.py` |
| #3 latency khi cache hit | cache không lưu `latency_ms`/`cache_hit`; lần hit báo thời gian của chính nó | `cache.py`, `engine.py` | `test_engine.py` |
| #4 BM25 index lại toàn bộ | index tăng dần, dedupe theo câu nguồn, giữ bản dịch mới nhất (khớp upsert của Chroma), có khoá luồng | `src/rag/hybrid_retriever.py` | `test_retrieval.py` |
| #5 rejected của DPO quá yếu | 5 kiểu negative (cắt đuôi, đổi chỗ từ, lặp từ, bỏ hư từ, thay từ sai thuật ngữ); RNG cục bộ có seed; luôn khác `chosen` | `src/training/dpo_align.py` | `test_training.py` |
| #6 DPO không train | `run_dpo_training()` dùng TRL; tiếp tục từ adapter SFT và dùng **bản sao đóng băng của adapter SFT làm reference** (nếu không TRL so với model gốc); lọc mẫu quá dài; `trl==0.12.2` | `dpo_align.py`, `requirements.txt` | `tests/slow/test_training_e2e.py` (chạy DPO thật trên model nhỏ) |
| #7 SFT tính loss cả prompt | collator che prompt + padding, mẫu mất marker thì bị bỏ qua hoàn toàn; **`remove_unused_columns=False`** (thiếu cờ này huấn luyện dừng ngay, xem mục 6); mẫu dài hơn `max_seq_length` bị loại thay vì cắt cụt đáp án; có tập eval + `load_best_model_at_end` | `src/training/sft_qlora.py`, `training/utils.py` | `test_training.py`, `tests/slow/test_training_e2e.py` ◆ |
| #8 CD chỉ build UI | build cả API và UI, smoke test, đẩy GHCR | `.github/workflows/cd-docker.yml` | `actionlint` |

## 3. Bảo mật

| Mục | Cách xử lý | Code | Kiểm chứng |
|---|---|---|---|
| CORS `*` | lấy từ `settings.cors.allowed_origins`; cho phép header `X-API-Key` | `src/api/main.py`, `settings.py` | `test_fastapi_endpoints.py` |
| Rate limiting | `slowapi`: hạn mức riêng cho `/translate`, `/memory/index`, admin và mặc định; `429` kèm `Retry-After` | `main.py`, `routes.py` | `test_fastapi_endpoints.py` |
| Authentication | `X-API-Key` (hoặc `Bearer`), so sánh hằng-thời-gian; `API_KEYS` cho dịch, `ADMIN_API_KEYS` cho ghi bộ nhớ/xoá cache; khoá phải ≥ 12 ký tự và không bao giờ nằm trong repr/log | `src/api/security.py` | `test_fastapi_endpoints.py`, `test_settings.py` |
| Container chạy root | `USER 10001` (số, để Kubernetes `runAsNonRoot` kiểm được) | `docker/*` | `hadolint` |
| Không giới hạn input | schema: `text_en` ≤ 5000 ký tự, câu TM ≤ 2000, ≤ 10000 cặp, `top_k` ≤ 10, chuỗi trắng bị từ chối; engine dùng ngân sách token cho cả prompt và **bớt ví dụ RAG thay vì cắt đuôi** | `schemas.py`, `engine.py`, `prompt_builder.py` | `test_engine.py`, `tests/slow` |
| CrossEncoder lỗi bị nuốt | log cảnh báo kèm nguyên nhân, `reranker.degraded=True` và `/health` báo `degraded` | `src/rag/reranker.py`, `routes.py` | `test_data_and_rag.py` |
| Đường dẫn SQLite | `validate_cache_path`: chặn `..`, đuôi lạ, NUL, symlink thoát khỏi `cache.allowed_root` (kiểm ở cả settings lẫn `TranslationCache`) | `src/core/cache.py`, `settings.py` | `test_cache.py`, `test_settings.py` |

## 4. Cải tiến

| Mục | Cách xử lý | Kiểm chứng |
|---|---|---|
| 1. Error handling | handler toàn cục + lỗi nghiệp vụ có kiểu (`InvalidInputError`→422, `EngineBusyError`/`ModelOutOfMemoryError`→503 + `Retry-After`); 500 chỉ trả `error_id`, chi tiết ở log; RAG lỗi → zero-shot; Chroma lỗi → chỉ BM25; lỗi index không còn bị nuốt | `test_fastapi_endpoints.py`, `test_engine.py` |
| 2. Split ingestion | `max(0, …)`, cảnh báo khi eval rỗng, khử trùng lặp câu nguồn trước khi chia | `test_ingestion.py` |
| 3. Streamlit không bắt lỗi | `ui/api_client.py` biến mọi lỗi (mất kết nối, timeout, 401/403/422/429/503/500, body không phải JSON) thành thông báo tiếng Việt; trang không bao giờ in traceback | `test_ui_client.py`, `test_ui_app.py` ◆ (trang cũ crash đúng như báo cáo) |
| 4. SFT thiếu eval | `eval_dataset`, `eval_strategy="steps"`, `load_best_model_at_end`; kiểm tra `save_steps` là bội của `eval_steps` | `tests/slow` (đọc `eval_loss` từ `trainer_state.json`) |
| 5. Pin version | pin `==` toàn bộ phụ thuộc trực tiếp; **pip resolver xác nhận bộ phiên bản tương thích trên Python 3.10**; tách `requirements-ui.txt`; bỏ `langchain` (không được import) | `pip install --dry-run` |
| 6. `device` lúc import | `device: auto` được phân giải khi dựng settings; `torch` chỉ import khi cần | `test_settings.py` |
| 7. Cache TTL/eviction | TTL, `max_entries`, `clear()`, `DELETE /cache` (admin), **namespace theo model/adapter/cấu hình sinh/reranker** (đổi adapter → cache tự vô hiệu), cache của các câu trả lời dùng RAG bị xoá khi nạp bộ nhớ mới | `test_cache.py`, `test_engine.py` |
| 8. Vector store trùng lặp | ID băm cố định + upsert; trùng ID trong *cùng một lệnh* (Chroma báo `DuplicateIDError`) được gộp trước khi ghi | `test_retrieval.py` (Chroma thật) |
| 9. Async routes | **giữ `def`**: FastAPI đã chạy handler đồng bộ trong thread pool nên không chặn event loop; `async def` + gọi model chặn sẽ tệ hơn. Điều thật sự thiếu là giới hạn đồng thời: `inference.max_concurrent_generations` + hàng đợi có timeout (503 khi quá tải) | `tests/slow` |
| 10. Test coverage | từ 6 lên 174 test: lỗi, ingestion, settings, cache, concurrency, prompt, UI, DPO, SFT; test integration không còn dùng `./data` thật | toàn bộ `tests/` |
| 11. CI/CD | CI: ruff pin + pytest cài PyTorch CPU; CD: smoke test có retry (`/health` + `/translate`), tag SHA/semver/nhánh/`latest`, đẩy GHCR; compose nhận image qua `NMT_API_IMAGE`/`NMT_UI_IMAGE` | `actionlint` |

## 5. Lỗi phát hiện thêm khi kiểm chứng (đã sửa)

| Lỗi | Hậu quả nếu để nguyên | Cách xử lý |
|---|---|---|
| `truncation=True, max_length=max_seq_length(128)` ở bước suy luận (từ bản sửa #5 trước đó) | RAG từ 2 ví dụ trở lên bị cắt mất câu cần dịch + `assistant` header (đã đo bằng tokenizer Qwen2: 171–213 token) | ngân sách token riêng (`inference.max_prompt_tokens`), bớt ví dụ thay vì cắt đuôi |
| SFT thiếu `remove_unused_columns=False` | `ValueError: No columns in the dataset match…` ngay bước đầu ◆ | thêm cờ, có test e2e |
| `max_length=128` khi train | đáp án bị cắt cụt (câu EN 27 từ chỉ còn 30/55 token đáp án); câu dài không còn marker → không che prompt | nâng lên 384, loại mẫu dài, mẫu mất marker bị bỏ qua |
| Tokenizer HF dùng chung giữa các luồng | `RuntimeError: Already borrowed` → 500 ngẫu nhiên khi có request đồng thời (lỗi có sẵn từ bản gốc) ◆ | khoá quanh tokenizer, embedding, cross-encoder |
| Chroma đọc khi đang ghi | `Document(page_content=None)` ngẫu nhiên khi vừa nạp TM vừa tìm kiếm ◆ | đọc/ghi loại trừ nhau theo batch 100 |
| `except` nuốt lỗi từng batch khi index | batch có ID trùng bị mất lặng lẽ, API vẫn báo thành công | không nuốt lỗi; index idempotent nên gửi lại an toàn |
| Chi tiết lỗi (`str(exc)`) trả cho client | lộ thông tin nội bộ | thông báo chung + `error_id` |
| `DELETE /cache` công khai (do bản sửa trước thêm vào) | ai cũng xoá được cache | yêu cầu khoá admin |
| Key cache chuyển chữ thường | "Turkey" và "turkey" dùng chung một bản dịch | key phân biệt hoa/thường, vẫn bỏ qua khác biệt khoảng trắng |
| DPO không có reference đúng | KL neo vào model gốc thay vì bản SFT | adapter `reference` đóng băng |
| `deduplicate_pairs` không được pipeline gọi | câu trùng giữa TM và eval làm điểm RAG cao ảo | khử trùng lặp theo câu nguồn trong `ingestion.py` |
| CI sẽ đỏ | `--timeout` thiếu `pytest-timeout`; 4 lỗi F401; `pip install ruff>=…` thiếu dấu nháy (shell hiểu `>` là redirect) | sửa cả ba, ruff pin trong `requirements-dev.txt` |
| Docker/CD | thiếu `--start-period` cho HEALTHCHECK; bind-mount `./data` không ghi được bởi user không phải root trên Linux; smoke test `sleep 10` | `start-period=180s`, named volume + UID cố định, vòng lặp retry |
| Notebook không train DPO, không có bước đưa mã nguồn lên Colab | adapter DPO không bao giờ được tạo | viết lại notebook |
| README: `make ui` không tồn tại, "SHA-1" | hướng dẫn sai | thêm target, sửa README |

## 6. Thay đổi hành vi cần biết khi nâng cấp

- `dataset.max_seq_length` 128 → **384**: cần nhiều VRAM hơn khi train; nếu hết bộ nhớ hãy giảm `qlora.train_batch_size`, tăng `gradient_accumulation_steps`.
- Cache cũ **không còn khớp** (key mới có namespace và phân biệt hoa/thường): coi như được làm mới.
- Khi bật `API_KEYS`, giao diện cần `API_KEY`; form nạp thuật ngữ cần `ADMIN_API_KEY`.
- `src.api.main:app` được tạo **lười** (không còn side-effect lúc import); có thể gọi `create_app(settings=…, engine=…)` để tiêm cấu hình/engine.
- `engine.translate(..., use_cache=False)` mới; `benchmark` dùng nó để đo model thật.
- Cấu hình mới: `inference.*`, `security.*`, `cache.allowed_root`, `qlora.eval_steps|save_steps|logging_steps|max_eval_samples`.
- `docker-compose.yml` dùng named volume `nmt-data` (có cả cache Hugging Face) thay vì bind-mount `./data`.

## 7. Chưa kiểm chứng được trong môi trường làm việc (không có GPU, Docker, mạng tới Hugging Face/GitHub)

- Đường 4-bit trên GPU (`bitsandbytes`, `prepare_model_for_kbit_training`): chỉ chạy được nhánh CPU; logic dùng chung (collator, Trainer, DPO, engine) đã chạy thật.
- Tải model thật (Qwen2.5-1.5B, embedding đa ngôn ngữ, cross-encoder) và chất lượng dịch/BLEU.
- `docker build`, chạy workflow trên GitHub Actions và đẩy GHCR: mới được kiểm bằng `actionlint`, `hadolint` và trình giải phụ thuộc của pip.

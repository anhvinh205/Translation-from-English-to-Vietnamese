# 🌐 WikiMatrix Hybrid-RAG QLoRA & DPO Machine Translation (v2.1 Production)

![CI Status](https://img.shields.io/badge/CI-Ruff_%2B_Pytest-2088FF?logo=github-actions&logoColor=white)
![Architecture](https://img.shields.io/badge/Microservices-FastAPI_%2B_Streamlit-009688?logo=fastapi&logoColor=white)
![RAG](https://img.shields.io/badge/Hybrid_RAG-BM25_%2B_ChromaDB_%2B_CrossEncoder-FF6F00)
![Fine-Tuning](https://img.shields.io/badge/LLM-4bit_QLoRA_%2B_DPO-EE4C2C?logo=pytorch&logoColor=white)
![Docker Compose](https://img.shields.io/badge/Deploy-Docker_Compose-2496ED?logo=docker&logoColor=white)

Hệ thống dịch máy Anh - Việt cấp độ Production: kiến trúc **Microservices (FastAPI + Streamlit)**, **Hybrid Retrieval-Augmented Generation (BM25 + Dense Vector + Cross-Encoder Reranker)**, **SQLite Cache**, và tinh chỉnh 2 giai đoạn **4-bit QLoRA + Direct Preference Optimization (DPO)** trên bộ dữ liệu **WikiMatrix (`en-vi`)**.

---

## 🚀 Điểm nhấn của dự án

1. **Microservices (`src/api/` & `ui/`):** Backend Inference API (**FastAPI**) tách biệt với giao diện (**Streamlit**), điều phối bằng `docker-compose.yml`.
2. **Hybrid RAG + Cross-Encoder Reranker (`src/rag/`):** BM25 + ChromaDB hợp nhất bằng **Reciprocal Rank Fusion**, chấm lại bằng Cross-Encoder kèm ngưỡng `score_threshold` (không đủ tin cậy thì tự tắt RAG, tránh nhiễu ngữ cảnh).
3. **Cache & Observability (`src/core/`):** SQLite cache có TTL, giới hạn dung lượng và *namespace theo model/adapter/cấu hình sinh* (đổi model thì cache cũ tự vô hiệu); log JSON kèm `latency_ms` thật.
4. **Dữ liệu & huấn luyện (`src/data/`, `src/training/`):** làm sạch + khử trùng lặp theo câu nguồn (không rò rỉ giữa train/TM/eval), SFT QLoRA **chỉ tính loss trên bản dịch**, DPO **tiếp tục từ adapter SFT với reference model chính là bản SFT**.
5. **Đánh giá đa chiều (`src/evaluation/benchmark.py`):** SacreBLEU, chrF++ và độ trễ trung bình (đo thật, không đọc từ cache).

### Mới ở v2.1 (sau đợt rà soát code)
- **Bảo mật:** API key (`X-API-Key`) cho `/translate` và khoá *admin* riêng cho `/memory/index` + `DELETE /cache`; rate limit theo IP; CORS theo danh sách; chống ChatML prompt injection (kể cả token lồng nhau); lỗi 500 không lộ chi tiết nội bộ; kiểm tra đường dẫn SQLite; container chạy user không phải root.
- **Độ tin cậy:** ngân sách token cho prompt (bớt ví dụ RAG thay vì cắt mất câu cần dịch), giới hạn số lượt sinh đồng thời, khoá thread-safety cho tokenizer/embedding/cross-encoder/Chroma, RAG lỗi thì tự lùi về zero-shot.
- **Huấn luyện:** sửa lỗi SFT không chạy được (`remove_unused_columns`), không cắt cụt đáp án (lọc mẫu quá dài), DPO có negative đa dạng và reference đúng.
- **Kỹ thuật:** phụ thuộc pin chính xác (đã kiểm bằng pip resolver), 170+ test hermetic (không cần mạng), CI/CD build + smoke test + đẩy image lên GHCR.

Chi tiết từng mục xem **[FIXES.md](FIXES.md)**.

---

## 📁 Cấu trúc thư mục

```text
wikimatrix-rag-nmt-v2/
├── .github/workflows/
│   ├── ci-test.yml                  # Ruff (phiên bản pin) + Pytest (unit, integration, e2e model nhỏ)
│   └── cd-docker.yml                # Build image, smoke test, đẩy lên GHCR (tag: nhánh, semver, SHA, latest)
├── configs/
│   ├── train_config.yaml            # WikiMatrix, QLoRA & DPO
│   └── rag_config.yaml              # Hybrid Search, Reranker, Cache, Inference, Rate limit
├── docker/
│   ├── Dockerfile.api               # FastAPI backend (user 10001, HEALTHCHECK có start-period)
│   └── Dockerfile.ui                # Streamlit frontend
├── notebooks/
│   └── Colab_SFT_DPO_Pipeline.ipynb # SFT QLoRA + DPO + benchmark trên Google Colab
├── src/
│   ├── api/                         # create_app(), routes, schemas, xác thực API key
│   ├── core/                        # Settings (YAML + env), JSON logger, SQLite cache, lỗi nghiệp vụ
│   ├── data/                        # Ingestion, cleaner, prompt builder (+ sanitize ChatML)
│   ├── rag/                         # Chroma store, BM25 + RRF, reranker, engine
│   ├── training/                    # SFT QLoRA (masking collator), DPO, tiện ích lọc độ dài
│   └── evaluation/                  # SacreBLEU, chrF++, latency
├── ui/
│   ├── api_client.py                # Client HTTP: mọi lỗi mạng/HTTP thành thông báo dễ đọc
│   └── streamlit_app.py             # Giao diện
├── tests/
│   ├── unit/                        # cache, settings, prompt, retrieval, engine, ingestion, training, UI client
│   ├── integration/                 # FastAPI (auth, rate limit, lỗi, CORS) + trang Streamlit (AppTest)
│   └── slow/                        # SFT -> DPO -> phục vụ bằng model Qwen2 nhỏ ngẫu nhiên, chạy CPU
├── FIXES.md                         # Đối chiếu từng mục review -> cách sửa -> cách kiểm chứng
├── docker-compose.yml
├── Makefile
├── requirements.txt                 # API + huấn luyện (pin chính xác)
├── requirements-ui.txt              # Streamlit
└── requirements-dev.txt             # + pytest, pytest-timeout, ruff, httpx
```

---

## 🛠️ Khởi chạy nhanh

Yêu cầu: **Python 3.10**.

### 1. Phát triển & kiểm thử
```bash
make dev-install      # cài toàn bộ phụ thuộc (đã pin)
make check            # ruff + pytest (không cần mạng, không tải model)
```

### 2. Chạy hệ thống
```bash
# Cách 1: Docker Compose (khuyên dùng)
cp .env.example .env          # rồi điền API_KEYS / ADMIN_API_KEYS (xem mục Bảo mật)
make up                       # API: http://localhost:8000/docs   UI: http://localhost:8501

# Cách 2: chạy trực tiếp trên 2 terminal
make api
make ui
```

Dữ liệu (Chroma, cache) và cache model Hugging Face nằm trong volume `nmt-data`, nên model chỉ tải một lần. Nếu muốn dùng thư mục máy chủ (`./data:/app/data`), trên Linux cần `sudo chown -R 10001:10001 data` vì container chạy bằng UID 10001.

---

## 🔐 Bảo mật & cấu hình

| Biến môi trường | Ý nghĩa | Mặc định |
|---|---|---|
| `API_KEYS` | Khoá được gọi `/translate` (nhiều khoá cách nhau dấu phẩy, mỗi khoá ≥ 12 ký tự) | *(rỗng = **tắt xác thực**, chỉ dùng khi phát triển)* |
| `ADMIN_API_KEYS` | Khoá được ghi bộ nhớ dịch (`/memory/index`) và xoá cache (`DELETE /cache`) | *(rỗng: nếu đã bật xác thực thì các endpoint admin bị khoá)* |
| `RATE_LIMIT_ENABLED` | Bật/tắt rate limit | `1` |
| `RATE_LIMIT_DEFAULT` / `_TRANSLATE` / `_MEMORY_INDEX` / `_ADMIN` | Hạn mức theo IP, dạng `30/minute` | `120/minute` · `30/minute` · `10/minute` · `5/minute` |
| `CORS_ORIGINS` | Danh sách origin được phép | `http://localhost:8501` |
| `MAX_CONCURRENT_GENERATIONS` | Số lượt sinh chạy đồng thời trên model | `1` |
| `BASE_MODEL_ID`, `CHROMA_PERSIST_DIR`, `CACHE_ENABLED`, `MOCK_LLM` | Model nền, thư mục Chroma, bật cache, chạy giả lập không tải LLM | xem `.env.example` |
| `API_BASE_URL`, `API_KEY`, `ADMIN_API_KEY` | Dành cho giao diện Streamlit | xem `.env.example` |

Tạo khoá: `python -c "import secrets; print(secrets.token_urlsafe(32))"`. Không bao giờ ghi khoá vào file YAML hay commit file `.env` (đã nằm trong `.gitignore`).

Lưu ý khi đưa lên môi trường thật:
- Rate limit đếm **trong bộ nhớ, theo IP, theo từng process**. Đặt sau reverse proxy thì chạy uvicorn với `--proxy-headers --forwarded-allow-ips=<IP proxy>` để thấy IP thật của client; nhiều worker thì mỗi worker đếm riêng (model lớn nên thường chỉ chạy 1 worker).
- Thiếu `ADMIN_API_KEYS` mà bật xác thực → `/memory/index` và `DELETE /cache` trả `403` (an toàn mặc định).
- `/health` luôn công khai để Docker/Kubernetes kiểm tra sức khoẻ; nó báo `degraded: true` khi reranker phải chạy chế độ dự phòng.

---

## 🔌 Dùng API

```bash
KEY=<API_KEY>; ADMIN=<ADMIN_API_KEY>

curl -s http://localhost:8000/api/v1/translate -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
     -d '{"text_en": "Low-resource machine translation is hard.", "use_rag": true, "top_k": 3}'

# Nạp thuật ngữ vào Translation Memory (cần khoá admin; idempotent, gửi lại an toàn)
curl -s http://localhost:8000/api/v1/memory/index -H "X-API-Key: $ADMIN" -H "Content-Type: application/json" \
     -d '{"pairs": [{"en": "attention head", "vi": "đầu chú ý"}]}'

curl -s -X DELETE http://localhost:8000/api/v1/cache -H "X-API-Key: $ADMIN"
```

Mọi lỗi có dạng `{"error": "<mã>", "detail": "<mô tả>"}`: `401/403` (khoá), `422` (dữ liệu), `429` (rate limit, kèm `Retry-After`), `503` (máy chủ bận hoặc hết bộ nhớ, kèm `Retry-After`), `500` (kèm `error_id` để tra log; chi tiết lỗi chỉ nằm trong log server).

---

## 🧪 Kiểm thử

```bash
make test                         # toàn bộ (≈ 15 giây, không cần mạng)
pytest tests/unit tests/integration
pytest tests/slow                 # SFT -> DPO -> engine với model Qwen2 nhỏ ngẫu nhiên (CPU)
```

Test không tải model nào: Chroma dùng embedding giả, reranker chạy chế độ lexical, huấn luyện dùng model nhỏ + tokenizer tự dựng trong test.

---

## 🎓 Huấn luyện & đánh giá

```bash
make train-sft     # SFT QLoRA (loss chỉ trên bản dịch) -> models/qwen-wikimatrix-sft-lora
make train-dpo     # DPO, tiếp tục từ adapter SFT      -> models/qwen-wikimatrix-dpo-lora
make eval          # SacreBLEU + chrF++ + latency (không dùng cache)
```

Điểm cần biết:
- `dataset.max_seq_length` (mặc định **384** token cho prompt + đáp án) là giới hạn *khi huấn luyện*: mẫu dài hơn bị **loại** chứ không bị cắt (cắt sẽ làm mất đáp án). Log in số mẫu bị loại. Nếu hết VRAM (ví dụ Colab T4), giảm `qlora.train_batch_size` và tăng `qlora.gradient_accumulation_steps`.
- Khi chạy suy luận, `inference.max_prompt_tokens` (mặc định 2048) là ngân sách cho *cả prompt*; vượt thì các ví dụ RAG xếp hạng thấp nhất bị bỏ trước, câu cần dịch luôn được giữ.
- Engine ưu tiên adapter DPO, rồi SFT, rồi zero-shot. Cache tự vô hiệu khi adapter thay đổi.
- Notebook `notebooks/Colab_SFT_DPO_Pipeline.ipynb` chạy trọn quy trình trên Colab.

---

## 🚢 CI/CD & Triển khai

- **CI (`ci-test.yml`):** lint bằng đúng phiên bản ruff đã pin → pytest (cài PyTorch CPU, không tải model).
- **CD (`cd-docker.yml`):** build image API + UI, **smoke test** (API gọi thật `/health` và `/translate` với `MOCK_LLM=1`, có retry), rồi đẩy lên **GitHub Container Registry** với các tag: tên nhánh, semver (khi gắn tag `v1.2.3`), SHA ngắn và `latest` (nhánh mặc định).
- **Triển khai bằng image đã build:**
  ```bash
  export NMT_API_IMAGE=ghcr.io/<owner>/<repo>/api:<tag>
  export NMT_UI_IMAGE=ghcr.io/<owner>/<repo>/ui:<tag>
  docker compose pull && docker compose up -d
  ```
  Workflow chưa tự deploy lên máy chủ nào vì nơi triển khai tuỳ từng người; có thể thêm một job SSH/Kubernetes sau bước publish.

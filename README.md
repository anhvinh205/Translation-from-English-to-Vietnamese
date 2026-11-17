# 🌐 WikiMatrix Hybrid-RAG QLoRA & DPO Machine Translation (v2.1 Production)

![CI Status](https://img.shields.io/badge/CI-Ruff_%2B_Pytest-2088FF?logo=github-actions&logoColor=white)
![Architecture](https://img.shields.io/badge/Microservices-FastAPI_%2B_Streamlit-009688?logo=fastapi&logoColor=white)
![RAG](https://img.shields.io/badge/Hybrid_RAG-BM25_%2B_ChromaDB_%2B_CrossEncoder-FF6F00)
![Fine-Tuning](https://img.shields.io/badge/LLM-4bit_QLoRA_%2B_DPO-EE4C2C?logo=pytorch&logoColor=white)
![Docker Compose](https://img.shields.io/badge/Deploy-Docker_Compose-2496ED?logo=docker&logoColor=white)

A production-grade English-to-Vietnamese Neural Machine Translation (NMT) system featuring **Microservices architecture (FastAPI + Streamlit)**, **Hybrid Retrieval-Augmented Generation (BM25 + ChromaDB Dense Vector Search + Cross-Encoder Reranker)**, **Persistent SQLite Cache**, and a two-stage fine-tuning pipeline using **4-bit QLoRA + Direct Preference Optimization (DPO)** based on `Qwen/Qwen2.5-1.5B-Instruct` on the **WikiMatrix (`en-vi`)** dataset.

---

## 🚀 Key Highlights

1. **Microservices Architecture (`src/api/` & `ui/`):** Clean separation of concerns between backend Inference API (**FastAPI**) and interactive web UI (**Streamlit**), orchestrated via Docker Compose or standalone processes.
2. **Hybrid RAG + Cross-Encoder Reranker (`src/rag/`):** Sparse lexical search (**BM25**) combined with dense vector retrieval (**ChromaDB**) merged via **Reciprocal Rank Fusion (RRF)**. Retrieved candidates are rescored with a Cross-Encoder reranker using an adaptive `score_threshold` (falls back to zero-shot if similarity is low, avoiding context corruption).
3. **Smart Persistent Caching (`src/core/cache.py`):** SQLite-backed translation cache with TTL expiration, LRU size cap, and *cache namespacing* by active model, adapter, reranker, and generation parameters (changing adapter or decoding settings automatically invalidates stale entries).
4. **Data Preprocessing & Dual-Stage Fine-Tuning (`src/data/`, `src/training/`):**
   - Data deduplication and sanitization against ChatML prompt injection.
   - **4-bit QLoRA SFT:** Loss calculated strictly on target translation tokens (prompt masking).
   - **DPO Alignment:** Trains against 5 synthetic hard negative variations (truncation, word swaps, repetitions, missing function words, lexical mismatch) anchored by a frozen reference copy of the SFT adapter.
5. **Multi-Metric Evaluation Suite (`src/evaluation/benchmark.py`):** Automated evaluation comparing Zero-Shot, Pure LoRA, and Hybrid RAG across **SacreBLEU**, **chrF++**, and true end-to-end latency.
6. **Thread Safety & CPU Inference Tuning:**
   - Thread-safe locks around Hugging Face tokenizers, ChromaDB read/writes, and Cross-Encoder calls.
   - CPU-optimized greedy decoding (`num_beams: 1`) reducing inference latency from ~60s to ~15s per sentence while retaining translation precision.

---

## 📁 Repository Structure

```text
wikimatrix-rag-nmt-v2/
├── .github/workflows/
│   ├── ci-test.yml                  # Ruff linter + Pytest (unit, integration, e2e synthetic model)
│   └── cd-docker.yml                # Multi-stage Docker build, smoke testing, publish to GHCR
├── configs/
│   ├── train_config.yaml            # WikiMatrix dataset, QLoRA & DPO training hyperparameters
│   └── rag_config.yaml              # Hybrid Search, Reranker, Cache, Inference, Rate limit settings
├── docker/
│   ├── Dockerfile.api               # FastAPI backend container (non-root UID 10001, healthcheck)
│   └── Dockerfile.ui                # Streamlit frontend container
├── models/
│   └── qwen-wikimatrix-dpo-lora/    # Trained DPO LoRA adapter weights & tokenizer configs
├── notebooks/
│   └── Colab_SFT_DPO_Pipeline.ipynb # End-to-end SFT + DPO training & evaluation on Google Colab
├── src/
│   ├── api/                         # FastAPI application factory, routes, schemas, API key security
│   ├── core/                        # Settings (YAML + .env), structured JSON logger, SQLite cache
│   ├── data/                        # Ingestion, cleaning, ChatML prompt builder & sanitization
│   ├── rag/                         # ChromaDB store, BM25 + RRF retriever, Cross-Encoder reranker, engine
│   ├── training/                    # SFT QLoRA (prompt-masked collator), DPO trainer, sample filters
│   └── evaluation/                  # SacreBLEU, chrF++, latency benchmarking scripts
├── tests/
│   ├── unit/                        # Unit tests for cache, settings, prompt, retrieval, engine, ingestion, UI client
│   ├── integration/                 # Integration tests for FastAPI endpoints and Streamlit AppTest
│   └── slow/                        # End-to-end SFT -> DPO -> Engine execution with small synthetic model
├── ui/
│   ├── api_client.py                # Resilient HTTP client converting network/API errors into user alerts
│   └── streamlit_app.py             # English interactive Streamlit translation interface
├── test_cli.py                      # Interactive CLI terminal for quick local model translation verification
├── FIXES.md                         # Comprehensive code review fixes and audit report
├── docker-compose.yml               # Production multi-service orchestration
├── Makefile                         # Unified development and build tasks
├── requirements.txt                 # Pinned dependencies for backend and training
├── requirements-ui.txt              # Pinned dependencies for Streamlit frontend
└── requirements-dev.txt             # Testing and linting tools (pytest, ruff, httpx)
```

---

## 🛠️ Quickstart

### Prerequisites
- Python **3.10** or **3.11**
- (Optional for GPU) NVIDIA CUDA 11.8+ / 12.1+ with PyTorch GPU
- (Optional for containerization) Docker & Docker Compose

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/anhvinh205/Translation-from-English-to-Vietnamese.git
cd Translation-from-English-to-Vietnamese

# Create and activate virtual environment
python -m venv .venv
# On Linux/macOS:
source .venv/bin/activate
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-ui.txt
pip install -r requirements-dev.txt

# Create environment configuration
cp .env.example .env
```

---

## 💻 Running the Application

### Option A: Interactive CLI Mode (Fastest for testing)
Test translation directly in your terminal without starting web servers:
```bash
python test_cli.py
```
This loads the base model `Qwen/Qwen2.5-1.5B-Instruct` along with the trained DPO LoRA adapter, performs warm-up sample evaluations, and enters an interactive translation prompt.

### Option B: Local Microservices (FastAPI + Streamlit)
Open two terminal windows:

**Terminal 1 — Start FastAPI Backend:**
```bash
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
# API Docs available at: http://localhost:8000/docs
```

**Terminal 2 — Start Streamlit Frontend:**
```bash
streamlit run ui/streamlit_app.py --server.port 8501
# Open your browser at: http://localhost:8501
```

### Option C: Docker Compose (Production Deployment)
Build and run both services with containerized isolation and persistent volume caching:
```bash
docker compose up --build -d
```
- **Streamlit Web UI:** `http://localhost:8501`
- **FastAPI OpenAPI Docs:** `http://localhost:8000/docs`
- **Health Check Endpoint:** `http://localhost:8000/api/v1/health`

To stop the services:
```bash
docker compose down
```

---

## 🔐 Security & Configuration

Configure settings via `configs/rag_config.yaml` or override them with `.env` variables:

| Environment Variable | Description | Default |
|---|---|---|
| `API_KEYS` | Comma-separated API keys for `/translate` endpoint | *Empty (Authentication disabled)* |
| `ADMIN_API_KEYS` | Admin keys for `/memory/index` and `DELETE /cache` | *Empty (Admin endpoints locked if auth enabled)* |
| `RATE_LIMIT_ENABLED` | Toggle IP-based rate limiting | `1` |
| `RATE_LIMIT_TRANSLATE` | Rate limit for translation requests | `30/minute` |
| `RATE_LIMIT_MEMORY_INDEX`| Rate limit for TM ingestion | `10/minute` |
| `CORS_ORIGINS` | Allowed CORS origins | `http://localhost:8501` |
| `MAX_CONCURRENT_GENERATIONS` | Max simultaneous generation passes | `1` |
| `QUEUE_TIMEOUT_SECONDS` | Max wait time before returning 503 | `60` |
| `BASE_MODEL_ID` | Base Hugging Face model identifier | `Qwen/Qwen2.5-1.5B-Instruct` |
| `CHROMA_PERSIST_DIR` | Directory for persistent vector database | `data/chroma` |
| `MOCK_LLM` | Set `1` to run mock engine in tests/CI | `0` |

---

## 🔌 API Reference

### 1. Translate Sentence
`POST /api/v1/translate`

**Request Body:**
```json
{
  "text_en": "Low-resource machine translation can be significantly improved with hybrid retrieval-augmented generation.",
  "use_rag": true,
  "top_k": 3,
  "score_threshold": 0.25
}
```

**Response:**
```json
{
  "translation": "Dịch máy tài nguyên thấp có thể được cải thiện đáng kể với sinh tăng cường truy xuất lai.",
  "latency_ms": 1420.5,
  "cache_hit": false,
  "rag_triggered": true,
  "retrieved_memory": [
    {
      "en": "Retrieval-augmented generation improves translation quality.",
      "vi": "Sinh tăng cường truy xuất cải thiện chất lượng dịch thuật.",
      "score": 0.82,
      "rerank_score": 0.76
    }
  ],
  "model_info": {
    "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
    "active_adapter": "dpo",
    "device": "cpu"
  }
}
```

### 2. Ingest Translation Memory (Admin)
`POST /api/v1/memory/index` *(Requires `X-API-Key: <ADMIN_API_KEY>`)*

```json
{
  "pairs": [
    {
      "en": "attention head",
      "vi": "đầu chú ý"
    },
    {
      "en": "transformer architecture",
      "vi": "kiến trúc transformer"
    }
  ]
}
```

### 3. Clear Semantic Cache (Admin)
`DELETE /api/v1/cache` *(Requires `X-API-Key: <ADMIN_API_KEY>`)*

---

## 🧪 Testing & Code Quality

The project includes 170+ unit and integration tests with zero external network dependencies:

```bash
# Run code style and lint check
ruff check src/ tests/ ui/

# Run fast unit and integration tests
pytest tests/unit/ tests/integration/ -v

# Run end-to-end pipeline test on CPU with lightweight synthetic model
pytest tests/slow/ -v
```

---

## 🎓 Training Pipeline

To reproduce the SFT and DPO training on Google Colab (T4 / A100 GPU):

1. Open `notebooks/Colab_SFT_DPO_Pipeline.ipynb` in Google Colab.
2. Select **Runtime > Change runtime type > T4 GPU** (or A100).
3. Follow the notebook cells:
   - **Step 1:** Ingest and deduplicate WikiMatrix `en-vi` pairs.
   - **Step 2:** Fine-tune `Qwen/Qwen2.5-1.5B-Instruct` using 4-bit QLoRA with prompt masking.
   - **Step 3:** Generate synthetic preference pairs and train DPO adapter using the frozen SFT copy as reference.
   - **Step 4:** Evaluate SacreBLEU / chrF++ improvements and download adapter checkpoints to `models/qwen-wikimatrix-dpo-lora/`.

---

## 📄 License
This project is licensed under the Apache License 2.0.

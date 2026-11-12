import os

import streamlit as st

try:  # imported as a package (tests, `python -m`)
    from ui.api_client import TranslationApiClient
except ModuleNotFoundError:  # `streamlit run ui/streamlit_app.py` puts ui/ itself on sys.path
    from api_client import TranslationApiClient

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")
client = TranslationApiClient(
    API_BASE_URL,
    api_key=os.getenv("API_KEY") or None,
    admin_api_key=os.getenv("ADMIN_API_KEY") or None,
)

st.set_page_config(
    page_title="WikiMatrix Hybrid-RAG NMT v2.1",
    page_icon="🌐",
    layout="wide",
)

st.title("🌐 Hệ thống Dịch Máy Anh - Việt v2.1 (Microservices + Hybrid RAG + QLoRA/DPO)")
st.markdown(
    "Kiến trúc **Production-grade**: Giao diện **Streamlit** giao tiếp qua **FastAPI**, "
    "kết hợp **Hybrid Search (BM25 + ChromaDB Dense)**, **Cross-Encoder Reranker** và **Semantic Cache**."
)

with st.sidebar:
    st.header("⚙️ Trạng thái Backend API")
    health = client.health()
    if health.ok:
        info = health.data
        st.success(f"API Online ({str(info.get('device', '?')).upper()})")
        st.caption(f"**Model:** `{info.get('base_model', '?')}`\n\n**Adapter:** `{info.get('active_adapter', '?')}`")
        if info.get("degraded"):
            st.warning("Reranker đang chạy chế độ dự phòng (lexical): chất lượng truy xuất có thể giảm.")
    else:
        st.warning(health.error)

    st.divider()
    st.subheader("➕ Nạp Thuật ngữ vào Bộ nhớ (Real-time)")
    new_en = st.text_area("Câu tiếng Anh (EN):")
    new_vi = st.text_area("Bản dịch chuẩn (VI):")
    if st.button("Đánh chỉ mục (ChromaDB + BM25)", use_container_width=True):
        if new_en.strip() and new_vi.strip():
            indexed = client.index_pair(new_en.strip(), new_vi.strip())
            if indexed.ok:
                st.success(f"Đã nạp {indexed.data.get('indexed_count', 0)} cặp câu vào Hybrid Index!")
            else:
                st.error(indexed.error)
        else:
            st.warning("Hãy nhập đủ cả câu tiếng Anh và bản dịch tiếng Việt.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🇬🇧 Văn bản nguồn (English)")
    input_text = st.text_area(
        "Nhập câu cần dịch:",
        value=(
            "Low-resource machine translation can be significantly improved with hybrid "
            "retrieval-augmented generation and direct preference optimization."
        ),
        height=160,
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        use_rag = st.checkbox("Bật Hybrid RAG", value=True)
    with c2:
        top_k = st.slider("Top-K Rerank", 1, 5, 3)
    with c3:
        threshold = st.slider("Ngưỡng Rerank", 0.0, 0.9, 0.25, 0.05)

    translate_btn = st.button("🚀 Dịch qua FastAPI", type="primary", use_container_width=True)

with col2:
    st.subheader("🇻🇳 Bản dịch đích (Tiếng Việt)")
    if translate_btn:
        if not input_text.strip():
            st.warning("Hãy nhập câu cần dịch.")
        else:
            with st.spinner("Đang gọi FastAPI & Reranking..."):
                result = client.translate(input_text.strip(), use_rag=use_rag, top_k=top_k, score_threshold=threshold)
            if result.ok:
                data = result.data
                st.success(data.get("translation", ""))
                m1, m2, m3 = st.columns(3)
                m1.metric("Độ trễ (Latency)", f"{data.get('latency_ms', '?')} ms")
                m2.metric("Cache Hit", "Có ⚡" if data.get("cache_hit") else "Không")
                m3.metric("RAG Kích hoạt", "Có 🔍" if data.get("rag_triggered") else "Tắt / Dưới ngưỡng")

                if data.get("retrieved_memory"):
                    with st.expander("🔍 Chi tiết Điểm số Hybrid Search & Cross-Encoder Reranker", expanded=True):
                        for i, ex in enumerate(data["retrieved_memory"], 1):
                            st.markdown(
                                f"**#{i}** — Rerank: `{ex.get('rerank_score')}` | "
                                f"Dense: `{ex.get('dense_score')}` | BM25: `{ex.get('bm25_score')}`"
                            )
                            st.write(f"- **EN:** {ex.get('en')}")
                            st.write(f"- **VI:** {ex.get('vi')}")
            else:
                st.error(result.error)

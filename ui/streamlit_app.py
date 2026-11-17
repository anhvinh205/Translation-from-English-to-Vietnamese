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

st.title("🌐 English-to-Vietnamese Translation System v2.1 (Microservices + Hybrid RAG + QLoRA/DPO)")
st.markdown(
    "**Production-grade architecture**: **Streamlit** frontend communicating via **FastAPI**, "
    "powered by **Hybrid Search (BM25 + ChromaDB Dense)**, **Cross-Encoder Reranker**, and **Semantic Cache**."
)

with st.sidebar:
    st.header("⚙️ Backend API Status")
    health = client.health()
    if health.ok:
        info = health.data
        st.success(f"API Online ({str(info.get('device', '?')).upper()})")
        st.caption(f"**Model:** `{info.get('base_model', '?')}`\n\n**Adapter:** `{info.get('active_adapter', '?')}`")
        if info.get("degraded"):
            st.warning("Reranker is in lexical fallback mode: retrieval quality may be reduced.")
    else:
        st.warning(health.error)

    st.divider()
    st.subheader("➕ Add Terminology to Translation Memory (Real-time)")
    new_en = st.text_area("English Source (EN):")
    new_vi = st.text_area("Vietnamese Reference (VI):")
    if st.button("Index Pair (ChromaDB + BM25)", use_container_width=True):
        if new_en.strip() and new_vi.strip():
            indexed = client.index_pair(new_en.strip(), new_vi.strip())
            if indexed.ok:
                st.success(f"Indexed {indexed.data.get('indexed_count', 0)} pair(s) into Hybrid Index!")
            else:
                st.error(indexed.error)
        else:
            st.warning("Please enter both the English source and Vietnamese translation.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🇬🇧 Source Text (English)")
    input_text = st.text_area(
        "Enter text to translate:",
        value=(
            "Low-resource machine translation can be significantly improved with hybrid "
            "retrieval-augmented generation and direct preference optimization."
        ),
        height=160,
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        use_rag = st.checkbox("Enable Hybrid RAG", value=True)
    with c2:
        top_k = st.slider("Top-K Rerank", 1, 5, 3)
    with c3:
        threshold = st.slider("Rerank Threshold", 0.0, 0.9, 0.25, 0.05)

    translate_btn = st.button("🚀 Translate via FastAPI", type="primary", use_container_width=True)

with col2:
    st.subheader("🇻🇳 Target Translation (Vietnamese)")
    if translate_btn:
        if not input_text.strip():
            st.warning("Please enter text to translate.")
        else:
            with st.spinner("Calling FastAPI & Reranking..."):
                result = client.translate(input_text.strip(), use_rag=use_rag, top_k=top_k, score_threshold=threshold)
            if result.ok:
                data = result.data
                st.success(data.get("translation", ""))
                m1, m2, m3 = st.columns(3)
                m1.metric("Latency", f"{data.get('latency_ms', '?')} ms")
                m2.metric("Cache Hit", "Yes ⚡" if data.get("cache_hit") else "No")
                m3.metric("RAG Triggered", "Active 🔍" if data.get("rag_triggered") else "Off / Below threshold")

                if data.get("retrieved_memory"):
                    with st.expander("🔍 Hybrid Search & Cross-Encoder Reranker Scores", expanded=True):
                        for i, ex in enumerate(data["retrieved_memory"], 1):
                            st.markdown(
                                f"**#{i}** — Rerank: `{ex.get('rerank_score')}` | "
                                f"Dense: `{ex.get('dense_score')}` | BM25: `{ex.get('bm25_score')}`"
                            )
                            st.write(f"- **EN:** {ex.get('en')}")
                            st.write(f"- **VI:** {ex.get('vi')}")
            else:
                st.error(result.error)


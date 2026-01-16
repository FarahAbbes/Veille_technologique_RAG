import streamlit as st
from datetime import datetime
from advanced_rag import AdvancedRAGSystem, AdvancedRAGConfig


def main():
    st.set_page_config(page_title="Chat PDF Avancé", page_icon="🧠", layout="wide")
    st.title("🧠 Assistant PDF Avancé")
    st.caption("RAG avec cache TTL, reranking cross-encoder et fusion RRF")

    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'metrics' not in st.session_state:
        st.session_state.metrics = {"queries": 0, "cache_hits": 0, "avg_latency_ms": 0}

    with st.sidebar:
        st.subheader("Configuration")
        model_name = st.selectbox("Modèle", ["google/flan-t5-small", "google/flan-t5-base", "google/flan-t5-large"])
        ttl = st.slider("TTL du cache (s)", 60, 1800, 300, 30)
        use_rerank = st.checkbox("Reranking cross-encoder", True)
        use_rrf = st.checkbox("Fusion RRF", True)
        k = st.slider("k résultats", 3, 8, 5)
        fetch_k = st.slider("fetch_k", 6, 24, 12)
        pdf_docs = st.file_uploader("PDFs", accept_multiple_files=True, type=["pdf"])
        build_index = st.button("Construire/assurer l'index")

    cfg = AdvancedRAGConfig(use_rerank=use_rerank, use_rrf=use_rrf, ttl_seconds=ttl, k=k, fetch_k=fetch_k, model_name=model_name)
    rag = AdvancedRAGSystem(cfg)

    if build_index and pdf_docs:
        rag.ensure_index(pdf_docs)
        st.success("Index prêt")

    q = st.text_input("Posez votre question")
    if q:
        ans, meta = rag.answer(q)
        st.session_state.metrics["queries"] += 1
        st.session_state.metrics["avg_latency_ms"] = (
            (st.session_state.metrics["avg_latency_ms"] * (st.session_state.metrics["queries"] - 1) + meta.get("latency_ms", 0))
            / st.session_state.metrics["queries"]
        )
        if meta.get("cache", {}).get("hits", 0) > 0:
            st.session_state.metrics["cache_hits"] = meta["cache"]["hits"]

        m_user = st.chat_message("user")
        m_user.write(q)
        m_assistant = st.chat_message("assistant")
        srcs = meta.get("sources", [])
        src_text = ", ".join(srcs) if srcs else "Aucune source détectée"
        m_assistant.write(ans)
        m_assistant.caption(f"Sources: {src_text} • Latence: {meta.get('latency_ms', 0)} ms • RRF: {cfg.use_rrf} • Rerank: {cfg.use_rerank} • {datetime.now().strftime('%H:%M:%S')}")

        st.session_state.conversation_history.append((q, ans, model_name, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), src_text, meta.get('latency_ms', 0)))

    col1, col2, col3 = st.columns(3)
    col1.metric("Questions", st.session_state.metrics["queries"])
    col2.metric("Cache hits", st.session_state.metrics["cache_hits"])
    col3.metric("Latence moyenne", f"{int(st.session_state.metrics['avg_latency_ms'])} ms")


if __name__ == "__main__":
    main()

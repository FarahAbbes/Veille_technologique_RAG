import streamlit as st
from datetime import datetime
from advanced_rag import AdvancedRAGSystem, AdvancedRAGConfig

# Google Gemini imports
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


def main():
    st.set_page_config(page_title="Chat PDF Avancé", page_icon="🧠", layout="wide")
    st.title("🧠 Assistant PDF Avancé")
    st.caption("RAG avec cache TTL, reranking cross-encoder et fusion RRF")

    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'metrics' not in st.session_state:
        st.session_state.metrics = {"queries": 0, "cache_hits": 0, "avg_latency_ms": 0}
    if 'gemini_api_key' not in st.session_state:
        st.session_state.gemini_api_key = "AIzaSyCM78aSjZCHiEH5uxehA5f9ru2xL2mHNcQ"

    with st.sidebar:
        st.subheader("Configuration")
        model_name = st.selectbox("Modèle", ["google/flan-t5-small", "google/flan-t5-base", "google/flan-t5-large", "Gemini"])
        
        # Gemini API key (only shown when Gemini is selected)
        gemini_api_key = None
        if model_name == "Gemini":
            gemini_api_key = st.text_input(
                "🔑 Clé API Gemini:",
                value=st.session_state.get('gemini_api_key', 'AIzaSyCM78aSjZCHiEH5uxehA5f9ru2xL2mHNcQ'),
                type="password",
                help="Clé API Google Gemini"
            )
            # Store API key in session state
            if gemini_api_key:
                st.session_state.gemini_api_key = gemini_api_key
        
        ttl = st.slider("TTL du cache (s)", 60, 1800, 300, 30)
        use_rerank = st.checkbox("Reranking cross-encoder", True)
        use_rrf = st.checkbox("Fusion RRF", True)
        k = st.slider("k résultats", 3, 8, 5)
        fetch_k = st.slider("fetch_k", 6, 24, 12)
        pdf_docs = st.file_uploader("PDFs", accept_multiple_files=True, type=["pdf"])
        build_index = st.button("Construire/assurer l'index")

    cfg = AdvancedRAGConfig(use_rerank=use_rerank, use_rrf=use_rrf, ttl_seconds=ttl, k=k, fetch_k=fetch_k, model_name=model_name)
    
    # Check if Gemini API key is required and available
    if model_name == "Gemini":
        if not st.session_state.get('gemini_api_key'):
            st.error("❌ Clé API Gemini requise. Veuillez l'entrer dans la barre latérale.")
            st.stop()
    
    rag = AdvancedRAGSystem(cfg, gemini_api_key=st.session_state.get('gemini_api_key') if model_name == "Gemini" else None)

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

        # Afficher les scores d'évaluation
        evaluation = meta.get("evaluation", {})
        if evaluation:
            st.markdown("---")
            st.markdown("### 📊 Score d'Évaluation de la Réponse")
            
            # Score global avec indicateur visuel
            overall_score = evaluation.get("overall_score", 0.0)
            score_percentage = int(overall_score * 100)
            
            # Couleur selon le score
            if overall_score >= 0.8:
                score_color = "🟢"
                score_label = "Excellent"
            elif overall_score >= 0.6:
                score_color = "🟡"
                score_label = "Bon"
            elif overall_score >= 0.4:
                score_color = "🟠"
                score_label = "Moyen"
            else:
                score_color = "🔴"
                score_label = "Faible"
            
            # Afficher le score global
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.metric(f"{score_color} Score Global", f"{score_percentage}%", score_label)
            with col2:
                st.metric("Pertinence Question", f"{int(evaluation.get('relevance_to_question', 0) * 100)}%")
            with col3:
                st.metric("Pertinence Contexte", f"{int(evaluation.get('relevance_to_context', 0) * 100)}%")
            
            # Barre de progression pour le score global
            st.progress(overall_score)
            
            # Détails des scores
            with st.expander("📈 Détails des Scores d'Évaluation"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**Pertinence à la Question:** {evaluation.get('relevance_to_question', 0):.2f}")
                    st.progress(evaluation.get('relevance_to_question', 0))
                    
                    st.markdown(f"**Pertinence au Contexte:** {evaluation.get('relevance_to_context', 0):.2f}")
                    st.progress(evaluation.get('relevance_to_context', 0))
                
                with col2:
                    st.markdown(f"**Complétude:** {evaluation.get('completeness', 0):.2f}")
                    st.progress(evaluation.get('completeness', 0))
                    
                    st.markdown(f"**Adéquation Longueur:** {evaluation.get('length_adequacy', 0):.2f}")
                    st.progress(evaluation.get('length_adequacy', 0))
                
                st.markdown("---")
                st.markdown("**Légende:**")
                st.markdown("- **Pertinence Question:** Correspondance entre la réponse et les mots-clés de la question")
                st.markdown("- **Pertinence Contexte:** Utilisation des informations du contexte fourni")
                st.markdown("- **Complétude:** Longueur et structure de la réponse")
                st.markdown("- **Adéquation Longueur:** Longueur appropriée par rapport à la question")

        st.session_state.conversation_history.append((q, ans, model_name, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), src_text, meta.get('latency_ms', 0), evaluation))

    col1, col2, col3 = st.columns(3)
    col1.metric("Questions", st.session_state.metrics["queries"])
    col2.metric("Cache hits", st.session_state.metrics["cache_hits"])
    col3.metric("Latence moyenne", f"{int(st.session_state.metrics['avg_latency_ms'])} ms")


if __name__ == "__main__":
    main()

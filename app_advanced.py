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
    
    # Custom CSS pour un design moderne
    st.markdown("""
        <style>
        /* Style général */
        .main {
            padding: 2rem 1rem;
        }
        
        /* Header avec gradient */
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2.5rem 2rem;
            border-radius: 15px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        }
        
        .main-header h1 {
            margin: 0;
            font-size: 2.5rem;
            font-weight: 700;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }
        
        .main-header p {
            margin: 0.5rem 0 0 0;
            font-size: 1.1rem;
            opacity: 0.95;
        }
        
        /* Sidebar améliorée */
        .sidebar .sidebar-content {
            background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%);
        }
        
        /* Cards pour les métriques */
        .metric-card {
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            border-left: 4px solid #667eea;
            transition: transform 0.2s;
        }
        
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        }
        
        /* Score d'évaluation amélioré */
        .score-container {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 2rem;
            border-radius: 15px;
            margin: 1.5rem 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        
        .score-badge {
            display: inline-block;
            padding: 0.5rem 1rem;
            border-radius: 25px;
            font-weight: 600;
            font-size: 0.9rem;
            margin: 0.25rem;
        }
        
        /* Chat messages améliorés */
        .stChatMessage {
            padding: 1rem;
            border-radius: 12px;
            margin: 1rem 0;
        }
        
        /* Input amélioré */
        .stTextInput > div > div > input {
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            padding: 12px 16px;
            font-size: 1rem;
            transition: all 0.3s;
        }
        
        .stTextInput > div > div > input:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        
        /* Buttons améliorés */
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 12px 24px;
            font-weight: 600;
            transition: all 0.3s;
            box-shadow: 0 4px 6px rgba(102, 126, 234, 0.3);
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
        }
        
        /* Progress bars améliorées */
        .stProgress > div > div > div {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
        }
        
        /* Expander amélioré */
        .streamlit-expanderHeader {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 0.75rem 1rem;
        }
        
        /* Selectbox amélioré */
        .stSelectbox > div > div {
            border-radius: 10px;
        }
        
        /* Slider amélioré */
        .stSlider > div > div > div {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        }
        
        /* File uploader amélioré */
        .uploadedFile {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 0.75rem;
            margin: 0.5rem 0;
        }
        
        /* Badge pour les scores */
        .score-indicator {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.9rem;
        }
        
        .score-excellent {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            color: white;
        }
        
        .score-good {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
        }
        
        .score-medium {
            background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
            color: white;
        }
        
        .score-poor {
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header avec design moderne
    st.markdown("""
        <div class="main-header">
            <h1>🧠 Assistant PDF Avancé</h1>
            <p>RAG avec cache TTL, reranking cross-encoder et fusion RRF</p>
        </div>
    """, unsafe_allow_html=True)

    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'metrics' not in st.session_state:
        st.session_state.metrics = {"queries": 0, "cache_hits": 0, "avg_latency_ms": 0}
    if 'gemini_api_key' not in st.session_state:
        st.session_state.gemini_api_key = "AIzaSyCM78aSjZCHiEH5uxehA5f9ru2xL2mHNcQ"

    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        st.markdown("---")
        model_name = st.selectbox("🤖 Modèle", ["google/flan-t5-small", "google/flan-t5-base", "google/flan-t5-large", "Gemini"], help="Choisissez le modèle à utiliser pour générer les réponses")
        
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
        
        st.markdown("#### 📊 Paramètres RAG")
        ttl = st.slider("⏱️ TTL du cache (s)", 60, 1800, 300, 30, help="Durée de vie du cache en secondes")
        use_rerank = st.checkbox("🔄 Reranking cross-encoder", True, help="Active le reranking pour améliorer la pertinence")
        use_rrf = st.checkbox("🔀 Fusion RRF", True, help="Active la fusion RRF (Reciprocal Rank Fusion)")
        k = st.slider("📈 k résultats", 3, 8, 5, help="Nombre de documents à récupérer")
        fetch_k = st.slider("🔍 fetch_k", 6, 24, 12, help="Nombre de documents à récupérer avant reranking")
        
        st.markdown("---")
        st.markdown("#### 📄 Documents")
        pdf_docs = st.file_uploader("Téléchargez vos PDFs", accept_multiple_files=True, type=["pdf"], help="Sélectionnez un ou plusieurs fichiers PDF à analyser")
        build_index = st.button("🚀 Construire/assurer l'index", use_container_width=True, type="primary")

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

    st.markdown("### 💬 Posez votre question")
    q = st.text_input(
        "Votre question:",
        placeholder="Ex: Quels sont les points clés de ce document?",
        label_visibility="collapsed"
    )
    
    if q:
        ans, meta = rag.answer(q)
        st.session_state.metrics["queries"] += 1
        st.session_state.metrics["avg_latency_ms"] = (
            (st.session_state.metrics["avg_latency_ms"] * (st.session_state.metrics["queries"] - 1) + meta.get("latency_ms", 0))
            / st.session_state.metrics["queries"]
        )
        if meta.get("cache", {}).get("hits", 0) > 0:
            st.session_state.metrics["cache_hits"] = meta["cache"]["hits"]

        # Chat messages avec design amélioré
        m_user = st.chat_message("user", avatar="👤")
        m_user.write(f"**{q}**")
        
        m_assistant = st.chat_message("assistant", avatar="🤖")
        srcs = meta.get("sources", [])
        src_text = ", ".join(srcs) if srcs else "Aucune source détectée"
        
        # Réponse avec style
        m_assistant.markdown(f"""
        <div style="padding: 1rem; background: #f8f9fa; border-radius: 10px; margin: 0.5rem 0;">
            {ans}
        </div>
        """, unsafe_allow_html=True)
        
        # Caption amélioré
        m_assistant.caption(f"📚 **Sources:** {src_text} • ⚡ **Latence:** {meta.get('latency_ms', 0)} ms • 🔀 **RRF:** {'✅' if cfg.use_rrf else '❌'} • 🔄 **Rerank:** {'✅' if cfg.use_rerank else '❌'} • 🕐 {datetime.now().strftime('%H:%M:%S')}")

        # Afficher les scores d'évaluation avec design amélioré
        evaluation = meta.get("evaluation", {})
        if evaluation:
            st.markdown("---")
            
            # Container pour les scores avec design moderne
            st.markdown("""
                <div class="score-container">
                    <h3 style="margin-top: 0; color: #667eea; font-size: 1.5rem;">📊 Score d'Évaluation de la Réponse</h3>
                </div>
            """, unsafe_allow_html=True)
            
            # Score global avec indicateur visuel
            overall_score = evaluation.get("overall_score", 0.0)
            score_percentage = int(overall_score * 100)
            
            # Couleur selon le score
            if overall_score >= 0.8:
                score_color = "🟢"
                score_label = "Excellent"
                score_class = "score-excellent"
            elif overall_score >= 0.6:
                score_color = "🟡"
                score_label = "Bon"
                score_class = "score-good"
            elif overall_score >= 0.4:
                score_color = "🟠"
                score_label = "Moyen"
                score_class = "score-medium"
            else:
                score_color = "🔴"
                score_label = "Faible"
                score_class = "score-poor"
            
            # Afficher le score global avec design amélioré
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
            
            with col1:
                st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                padding: 1.5rem; border-radius: 12px; color: white; text-align: center;">
                        <h2 style="margin: 0; font-size: 2.5rem; font-weight: 700;">{score_percentage}%</h2>
                        <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem; opacity: 0.9;">
                            {score_color} {score_label}
                        </p>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                    <div style="background: white; padding: 1rem; border-radius: 10px; 
                                box-shadow: 0 2px 8px rgba(0,0,0,0.1); text-align: center;">
                        <div style="font-size: 0.9rem; color: #666; margin-bottom: 0.5rem;">Pertinence Question</div>
                        <div style="font-size: 1.8rem; font-weight: 700; color: #667eea;">
                            {int(evaluation.get('relevance_to_question', 0) * 100)}%
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                    <div style="background: white; padding: 1rem; border-radius: 10px; 
                                box-shadow: 0 2px 8px rgba(0,0,0,0.1); text-align: center;">
                        <div style="font-size: 0.9rem; color: #666; margin-bottom: 0.5rem;">Pertinence Contexte</div>
                        <div style="font-size: 1.8rem; font-weight: 700; color: #667eea;">
                            {int(evaluation.get('relevance_to_context', 0) * 100)}%
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                    <div style="background: white; padding: 1rem; border-radius: 10px; 
                                box-shadow: 0 2px 8px rgba(0,0,0,0.1); text-align: center;">
                        <div style="font-size: 0.9rem; color: #666; margin-bottom: 0.5rem;">Complétude</div>
                        <div style="font-size: 1.8rem; font-weight: 700; color: #667eea;">
                            {int(evaluation.get('completeness', 0) * 100)}%
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            # Barre de progression pour le score global avec style amélioré
            st.markdown(f"""
                <div style="margin: 1.5rem 0;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                        <span style="font-weight: 600; color: #333;">Score Global</span>
                        <span style="font-weight: 600; color: #667eea;">{score_percentage}%</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            st.progress(overall_score)
            
            # Détails des scores avec design amélioré
            with st.expander("📈 Détails des Scores d'Évaluation", expanded=False):
                st.markdown("""
                    <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 10px; margin: 1rem 0;">
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                        <div style="margin-bottom: 1.5rem;">
                            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                                <span style="font-weight: 600;">🎯 Pertinence à la Question</span>
                                <span style="font-weight: 700; color: #667eea;">{evaluation.get('relevance_to_question', 0):.2f}</span>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    st.progress(evaluation.get('relevance_to_question', 0))
                    
                    st.markdown(f"""
                        <div style="margin-top: 1.5rem; margin-bottom: 1.5rem;">
                            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                                <span style="font-weight: 600;">📚 Pertinence au Contexte</span>
                                <span style="font-weight: 700; color: #667eea;">{evaluation.get('relevance_to_context', 0):.2f}</span>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    st.progress(evaluation.get('relevance_to_context', 0))
                
                with col2:
                    st.markdown(f"""
                        <div style="margin-bottom: 1.5rem;">
                            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                                <span style="font-weight: 600;">✅ Complétude</span>
                                <span style="font-weight: 700; color: #667eea;">{evaluation.get('completeness', 0):.2f}</span>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    st.progress(evaluation.get('completeness', 0))
                    
                    st.markdown(f"""
                        <div style="margin-top: 1.5rem; margin-bottom: 1.5rem;">
                            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                                <span style="font-weight: 600;">📏 Adéquation Longueur</span>
                                <span style="font-weight: 700; color: #667eea;">{evaluation.get('length_adequacy', 0):.2f}</span>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    st.progress(evaluation.get('length_adequacy', 0))
                
                st.markdown("""
                    </div>
                    <div style="background: #e3f2fd; padding: 1rem; border-radius: 8px; margin-top: 1rem;">
                        <h4 style="margin-top: 0; color: #1976d2;">📖 Légende</h4>
                        <ul style="margin: 0; padding-left: 1.5rem; color: #555;">
                            <li><strong>Pertinence Question:</strong> Correspondance entre la réponse et les mots-clés de la question</li>
                            <li><strong>Pertinence Contexte:</strong> Utilisation des informations du contexte fourni</li>
                            <li><strong>Complétude:</strong> Longueur et structure de la réponse</li>
                            <li><strong>Adéquation Longueur:</strong> Longueur appropriée par rapport à la question</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)

        st.session_state.conversation_history.append((q, ans, model_name, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), src_text, meta.get('latency_ms', 0), evaluation))

    st.markdown("---")
    st.markdown("### 📈 Statistiques Globales")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 1.5rem; border-radius: 12px; color: white; text-align: center;">
                <div style="font-size: 2rem; font-weight: 700;">{st.session_state.metrics["queries"]}</div>
                <div style="font-size: 1rem; opacity: 0.9; margin-top: 0.5rem;">Questions posées</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                        padding: 1.5rem; border-radius: 12px; color: white; text-align: center;">
                <div style="font-size: 2rem; font-weight: 700;">{st.session_state.metrics["cache_hits"]}</div>
                <div style="font-size: 1rem; opacity: 0.9; margin-top: 0.5rem;">Cache hits</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                        padding: 1.5rem; border-radius: 12px; color: white; text-align: center;">
                <div style="font-size: 2rem; font-weight: 700;">{int(st.session_state.metrics['avg_latency_ms'])} ms</div>
                <div style="font-size: 1rem; opacity: 0.9; margin-top: 0.5rem;">Latence moyenne</div>
            </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

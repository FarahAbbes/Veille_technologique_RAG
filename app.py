import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import base64
import os
from datetime import datetime

# LangChain imports - Compatible with LangChain 0.3
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM
from typing import Optional, List, Any
from transformers import pipeline
import torch

# Google Gemini imports
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
    # Try to import langchain wrapper, but fallback to direct API if not available
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        LANGCHAIN_GEMINI_AVAILABLE = True
    except ImportError:
        LANGCHAIN_GEMINI_AVAILABLE = False
        ChatGoogleGenerativeAI = None
except ImportError:
    GEMINI_AVAILABLE = False
    LANGCHAIN_GEMINI_AVAILABLE = False
    genai = None
    ChatGoogleGenerativeAI = None


class HuggingFaceLLM(LLM):
    """Custom LLM wrapper for HuggingFace models with enhanced generation"""

    pipeline: Any = None
    model_name: str = ""

    def __init__(self, model_name="google/flan-t5-base"):
        super().__init__()
        self.model_name = model_name
        self.pipeline = pipeline(
            "text2text-generation",
            model=model_name,
            device=0 if torch.cuda.is_available() else -1,
            max_length=512,
            truncation=True
        )

    @property
    def _llm_type(self) -> str:
        return "huggingface"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        # Enhanced generation with better parameters
        response = self.pipeline(
            prompt[:2000],
            max_length=512,
            min_length=50,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            truncation=True,
            repetition_penalty=1.2
        )[0]['generated_text']
        return response.strip()


def get_pdf_text(pdf_docs):
    """Extract text from multiple PDF files with metadata"""
    text = ""
    metadata = []
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page_num, page in enumerate(pdf_reader.pages, 1):
            page_text = page.extract_text()
            if page_text.strip():
                text += f"\n[Document: {pdf.name}, Page: {page_num}]\n{page_text}\n"
                metadata.append({
                    'file': pdf.name,
                    'page': page_num,
                    'length': len(page_text)
                })
    return text, metadata


def get_text_chunks(text):
    """Create optimized text chunks with better overlap"""
    chunk_size = 800  # Optimized for FLAN-T5
    chunk_overlap = 150
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    """Create vector store using local embeddings"""
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store


def get_available_gemini_models(api_key: str):
    """List available Gemini models for the given API key"""
    if not GEMINI_AVAILABLE or genai is None:
        return []
    
    try:
        genai.configure(api_key=api_key)
        models = genai.list_models()
        available = []
        for model in models:
            # Filter for text generation models
            if 'generateContent' in model.supported_generation_methods:
                model_name = model.name.replace('models/', '')
                # Only include Gemini models
                if 'gemini' in model_name.lower():
                    available.append(model_name)
        return available
    except Exception as e:
        # If listing fails, return common model names to try
        return ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro", "gemini-2.0-flash-exp"]


@st.cache_resource
def load_model(model_name, api_key=None):
    """Load and cache the model"""
    if model_name == "Gemini":
        if not GEMINI_AVAILABLE:
            raise ImportError("langchain-google-genai n'est pas installé. Installez-le avec: pip install langchain-google-genai")
        if not api_key:
            raise ValueError("Clé API Gemini requise")
        # Try gemini-1.5-flash first (most widely available)
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0.7,
            max_output_tokens=2048
        )
    elif model_name == "FLAN-T5 Small (fastest)":
        return HuggingFaceLLM(model_name="google/flan-t5-small")
    elif model_name == "FLAN-T5 Base (balanced)":
        return HuggingFaceLLM(model_name="google/flan-t5-base")
    elif model_name == "FLAN-T5 Large (best quality)":
        return HuggingFaceLLM(model_name="google/flan-t5-large")
    else:
        return HuggingFaceLLM(model_name="google/flan-t5-base")


def format_docs(docs):
    """Format documents for context with source tracking"""
    formatted = []
    for i, doc in enumerate(docs, 1):
        formatted.append(f"[Source {i}]\n{doc.page_content}\n")
    return "\n".join(formatted)


def create_enhanced_prompt(context: str, question: str, language: str = "auto") -> str:
    """Create an enhanced prompt with better structure and instructions"""
    
    # Detect if question is in French
    french_keywords = ['quel', 'quelle', 'comment', 'pourquoi', 'quand', 'où', 'qui', 'quoi']
    is_french = any(keyword in question.lower() for keyword in french_keywords)
    
    if is_french or language == "french":
        prompt = f"""Tu es un assistant expert en analyse de documents. Ta mission est de répondre avec précision aux questions basées sur le contexte fourni.

INSTRUCTIONS IMPORTANTES:
1. Base ta réponse UNIQUEMENT sur les informations du contexte ci-dessous
2. Si l'information n'est pas dans le contexte, dis clairement "Je ne trouve pas cette information dans les documents fournis"
3. Cite les sources pertinentes quand c'est possible (ex: "Selon le document...")
4. Sois précis, concis et factuel
5. Si plusieurs informations contradictoires existent, mentionne-le
6. Utilise des puces ou une structure claire si la réponse contient plusieurs points

CONTEXTE:
{context[:1400]}

QUESTION: {question}

RÉPONSE DÉTAILLÉE:"""
    else:
        prompt = f"""You are an expert document analysis assistant. Your mission is to answer questions accurately based on the provided context.

IMPORTANT INSTRUCTIONS:
1. Base your answer ONLY on the information in the context below
2. If the information is not in the context, clearly state "I cannot find this information in the provided documents"
3. Cite relevant sources when possible (e.g., "According to the document...")
4. Be precise, concise, and factual
5. If contradictory information exists, mention it
6. Use bullet points or clear structure if the answer contains multiple points

CONTEXT:
{context[:1400]}

QUESTION: {question}

DETAILED ANSWER:"""
    
    return prompt


def validate_response(response: str, context: str) -> tuple[str, float]:
    """Validate response quality and add confidence score"""
    confidence = 1.0
    
    # Check if response indicates missing information
    no_info_phrases = [
        "cannot find", "not in the", "no information", "not mentioned",
        "ne trouve pas", "pas dans", "aucune information", "pas mentionné"
    ]
    if any(phrase in response.lower() for phrase in no_info_phrases):
        confidence = 0.3
    
    # Check response length
    if len(response.strip()) < 20:
        confidence *= 0.7
    
    # Check if response seems relevant to context
    context_words = set(context.lower().split())
    response_words = set(response.lower().split())
    overlap = len(context_words.intersection(response_words))
    if overlap < 3:
        confidence *= 0.6
    
    return response, confidence


def user_input(user_question, model_name, pdf_docs, conversation_history):
    """Process user question with enhanced prompt engineering"""
    if pdf_docs is None:
        st.warning("⚠️ Veuillez télécharger des fichiers PDF avant de poser des questions.")
        return

    try:
        # Get embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # Load vector store
        new_db = FAISS.load_local(
            "faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )

        # Search for relevant documents with higher k for better context
        docs = new_db.similarity_search(user_question, k=3)

        if not docs:
            st.warning("⚠️ Aucun document pertinent trouvé pour cette question.")
            return

        # Get model
        with st.spinner("🤔 Génération de la réponse en cours..."):
            # Get Gemini API key from session state if needed
            api_key = None
            if model_name == "Gemini":
                api_key = st.session_state.get('gemini_api_key', None)
                if not api_key:
                    st.error("❌ Clé API Gemini requise. Veuillez l'entrer dans la barre latérale.")
                    return
            
            model = load_model(model_name, api_key=api_key)
            
            # Create context from documents
            context = format_docs(docs)
            
            # Create enhanced prompt
            prompt = create_enhanced_prompt(context, user_question)
            
            # Generate response
            if model_name == "Gemini":
                # Gemini uses direct API or invoke method
                response_text = None
                last_error = None
                
                try:
                    # Check if using direct API or langchain wrapper
                    if hasattr(model, 'generate_content'):
                        # Direct API
                        response = model.generate_content(prompt)
                        response_text = response.text.strip()
                    elif hasattr(model, 'invoke'):
                        # Langchain wrapper
                        response = model.invoke(prompt)
                        response_text = response.content.strip()
                    else:
                        raise ValueError("Format de modèle Gemini non reconnu")
                except Exception as e:
                    last_error = e
                    # If current model fails, try other available models
                    available_models = get_available_gemini_models(api_key)
                    if not available_models:
                        available_models = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro", "gemini-2.0-flash-exp"]
                    
                    for gemini_model_name in available_models:
                        try:
                            # Try direct API first
                            test_model = genai.GenerativeModel(gemini_model_name)
                            response = test_model.generate_content(prompt)
                            response_text = response.text.strip()
                            break  # Success
                        except Exception:
                            # Try langchain wrapper if available
                            if LANGCHAIN_GEMINI_AVAILABLE and ChatGoogleGenerativeAI:
                                try:
                                    test_model = ChatGoogleGenerativeAI(
                                        model=gemini_model_name,
                                        google_api_key=api_key,
                                        temperature=0.7,
                                        max_output_tokens=2048
                                    )
                                    response = test_model.invoke(prompt)
                                    response_text = response.content.strip()
                                    break  # Success
                                except Exception as e:
                                    last_error = e
                                    continue
                            continue
                
                if response_text is None:
                    st.error(f"❌ Aucun modèle Gemini disponible. Dernière erreur: {str(last_error)}")
                    st.info("💡 Vérifiez votre clé API et les modèles disponibles sur Google AI Studio: https://ai.google.dev/")
                    return
            else:
                response_text = model._call(prompt)
            
            # Validate response
            validated_response, confidence = validate_response(response_text, context)

        # Format response with metadata
        response_output = validated_response
        
        # Add confidence indicator
        if confidence < 0.5:
            confidence_emoji = "🔴"
            confidence_text = "Faible"
        elif confidence < 0.8:
            confidence_emoji = "🟡"
            confidence_text = "Moyenne"
        else:
            confidence_emoji = "🟢"
            confidence_text = "Élevée"
        
        response_output += f"\n\n{confidence_emoji} **Confiance: {confidence_text}** ({confidence:.0%})"

        user_question_output = user_question
        pdf_names = [pdf.name for pdf in pdf_docs] if pdf_docs else []

        # Add to conversation history
        conversation_history.append((
            user_question_output,
            response_output,
            model_name,
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            ", ".join(pdf_names),
            confidence
        ))

        user_msg = st.chat_message("user")
        user_msg.write(user_question_output)
        if pdf_names:
            src_text = ", ".join(pdf_names[:2]) + ("..." if len(pdf_names) > 2 else "")
            user_msg.caption(f"Sources: {src_text}")

        assistant_msg = st.chat_message("assistant")
        assistant_msg.write(response_output)
        assistant_msg.caption(f"Modèle: {model_name} • Confiance: {confidence_text} ({confidence:.0%}) • {datetime.now().strftime('%H:%M:%S')}")

        # Display conversation history
        if len(conversation_history) > 1:
            with st.expander(f"📚 Historique des conversations ({len(conversation_history)-1} précédentes)", expanded=False):
                for question, answer, model, timestamp, pdf_name, conf in reversed(conversation_history[:-1]):
                    m_user = st.chat_message("user")
                    m_user.write(question)
                    if pdf_name:
                        m_user.caption(f"Sources: {pdf_name}")
                    m_assistant = st.chat_message("assistant")
                    m_assistant.write(answer)
                    conf_txt = "Élevée" if conf >= 0.8 else ("Moyenne" if conf >= 0.5 else "Faible")
                    m_assistant.caption(f"Modèle: {model} • Confiance: {conf_txt} ({conf:.0%}) • {timestamp}")

        # Enhanced CSV export with confidence scores
        if len(st.session_state.conversation_history) > 0:
            df = pd.DataFrame(
                st.session_state.conversation_history,
                columns=["Question", "Answer", "Model", "Timestamp", "PDF Name", "Confidence"]
            )
            csv = df.to_csv(index=False, encoding='utf-8-sig')
            b64 = base64.b64encode(csv.encode()).decode()
            
            st.sidebar.markdown("---")
            st.sidebar.markdown("### 💾 Export des données")
            href = f'<a href="data:file/csv;base64,{b64}" download="conversation_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv" style="text-decoration:none;"><button style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; font-weight: bold;">📥 Télécharger l\'historique (CSV)</button></a>'
            st.sidebar.markdown(href, unsafe_allow_html=True)

        st.balloons()

    except Exception as e:
        st.error(f"❌ Erreur lors du traitement de la question: {str(e)}")
        st.info("💡 Conseil: Essayez de reformuler votre question ou vérifiez que les PDFs ont été correctement traités.")


def main():
    st.set_page_config(
        page_title="Chat PDF Intelligent",
        page_icon="🤖",
        layout="wide"
    )
    
    # Custom CSS for better UI
    st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .stTextInput > div > div > input {
            border: 2px solid #667eea;
            border-radius: 8px;
            padding: 10px;
            font-size: 1.05rem;
        }
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 10px 24px;
            font-weight: bold;
            transition: all 0.3s;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="main-header"><h1>🤖 Assistant PDF Intelligent</h1><p>Posez vos questions, obtenez des réponses précises et sourcées</p></div>', unsafe_allow_html=True)

    # Initialize session state
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    if 'doc_metadata' not in st.session_state:
        st.session_state.doc_metadata = None
    
    if 'gemini_api_key' not in st.session_state:
        st.session_state.gemini_api_key = "AIzaSyCM78aSjZCHiEH5uxehA5f9ru2xL2mHNcQ"

    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")

        # Model selection
        model_name = st.selectbox(
            "🤖 Modèle IA:",
            ("FLAN-T5 Small (fastest)", "FLAN-T5 Base (balanced)", "FLAN-T5 Large (best quality)", "Gemini"),
            help="Choisissez un modèle selon vos besoins de vitesse/qualité"
        )
        
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

        st.markdown("---")

        # Statistics
        if st.session_state.conversation_history:
            st.markdown("### 📊 Statistiques")
            st.metric("Questions posées", len(st.session_state.conversation_history))
            avg_confidence = sum([c[5] for c in st.session_state.conversation_history]) / len(st.session_state.conversation_history)
            st.metric("Confiance moyenne", f"{avg_confidence:.0%}")

        st.markdown("---")

        # Control buttons
        st.markdown("### 🎛️ Contrôles")
        col1, col2 = st.columns(2)
        clear_button = col1.button("🗑️ Effacer dernier", use_container_width=True)
        reset_button = col2.button("🔄 Tout réinitialiser", use_container_width=True)

        if reset_button:
            st.session_state.conversation_history = []
            st.session_state.doc_metadata = None
            if os.path.exists("faiss_index"):
                import shutil
                shutil.rmtree("faiss_index")
            st.success("✅ Réinitialisation complète!")
            st.rerun()

        if clear_button:
            if len(st.session_state.conversation_history) > 0:
                st.session_state.conversation_history.pop()
                st.success("✅ Dernière conversation effacée!")
            else:
                st.warning("⚠️ Aucune conversation à effacer.")

        st.markdown("---")

        # PDF upload
        st.markdown("### 📄 Documents")
        pdf_docs = st.file_uploader(
            "Téléchargez vos PDFs",
            accept_multiple_files=True,
            type=['pdf'],
            help="Sélectionnez un ou plusieurs fichiers PDF"
        )

        if pdf_docs:
            st.info(f"📁 {len(pdf_docs)} fichier(s) sélectionné(s)")
            for pdf in pdf_docs:
                st.text(f"• {pdf.name}")

        if st.button("🚀 Analyser les PDFs", use_container_width=True, type="primary"):
            if pdf_docs:
                with st.spinner("🔄 Traitement des PDFs en cours..."):
                    try:
                        # Extract text from PDFs with metadata
                        raw_text, metadata = get_pdf_text(pdf_docs)
                        st.session_state.doc_metadata = metadata

                        if not raw_text.strip():
                            st.warning("⚠️ Les PDFs semblent vides ou ne contiennent pas de texte extractible")
                        else:
                            # Create text chunks
                            text_chunks = get_text_chunks(raw_text)

                            # Create vector store
                            get_vector_store(text_chunks)

                            st.success(f"✅ {len(pdf_docs)} PDF(s) traité(s) avec succès!")
                            st.info(f"📦 {len(text_chunks)} segments de texte créés")
                            
                            # Show metadata
                            total_pages = sum([m['page'] for m in metadata])
                            st.metric("Pages totales", total_pages)
                            
                    except Exception as e:
                        st.error(f"❌ Erreur lors du traitement: {str(e)}")
            else:
                st.warning("⚠️ Veuillez d'abord télécharger des fichiers PDF.")

        # Tips section
        st.markdown("---")
        with st.expander("💡 Conseils d'utilisation"):
            st.markdown("""
            **Pour de meilleures réponses:**
            - 🎯 Posez des questions précises et claires
            - 📝 Utilisez des mots-clés du document
            - 🔍 Demandez des informations spécifiques
            - 📊 Demandez des comparaisons ou analyses
            
            **Exemples de questions:**
            - "Quels sont les points clés de ce document?"
            - "Résume les conclusions principales"
            - "Quelles sont les dates mentionnées?"
            - "Compare les données de la section X et Y"
            """)

    # Main area - Question input
    st.markdown("### 💬 Posez votre question")
    user_question = st.text_input(
        "Votre question:",
        placeholder="Ex: Quels sont les principaux résultats présentés dans le document?",
        label_visibility="collapsed"
    )

    if user_question:
        if not os.path.exists("faiss_index"):
            st.warning("⚠️ Veuillez d'abord télécharger et traiter des fichiers PDF!")
            st.info("👉 Utilisez la barre latérale pour télécharger vos documents")
        else:
            user_input(
                user_question,
                model_name,
                pdf_docs,
                st.session_state.conversation_history
            )


if __name__ == "__main__":
    main()

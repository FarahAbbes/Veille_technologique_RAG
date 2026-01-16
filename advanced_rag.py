from typing import List, Tuple, Optional, Dict, Any
import time
import math
import re
from dataclasses import dataclass
import os

from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import CrossEncoder
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


class QueryCacheTTL:
    def __init__(self, ttl_seconds: int = 300):
        self.ttl = ttl_seconds
        self.store: Dict[str, Tuple[float, Any]] = {}
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        now = time.time()
        if key in self.store:
            ts, val = self.store[key]
            if now - ts <= self.ttl:
                self.hits += 1
                return val
            else:
                del self.store[key]
        self.misses += 1
        return None

    def set(self, key: str, value: Any) -> None:
        self.store[key] = (time.time(), value)

    def stats(self) -> Dict[str, Any]:
        return {"ttl": self.ttl, "size": len(self.store), "hits": self.hits, "misses": self.misses}


def rrf_fusion(rank_lists: List[List[Any]], k: int = 5, c: int = 60) -> List[Any]:
    scores: Dict[int, float] = {}
    objects: Dict[int, Any] = {}
    for lst in rank_lists:
        for idx, item in enumerate(lst):
            key = id(item)
            objects[key] = item
            rank = idx + 1
            scores[key] = scores.get(key, 0.0) + 1.0 / (c + rank)
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [objects[key] for key, _ in sorted_items][:k]


class DocumentReRanker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: Optional[int] = None):
        self.device = device if device is not None else (0 if torch.cuda.is_available() else -1)
        try:
            self.model = CrossEncoder(model_name, device=("cuda" if self.device == 0 else "cpu"))
            self.available = True
        except Exception:
            self.model = None
            self.available = False

    def rerank(self, query: str, docs: List[Any], top_k: int = 5) -> List[Any]:
        if not docs:
            return []
        if not self.available:
            return sorted(docs, key=lambda d: len(getattr(d, "page_content", str(d))), reverse=True)[:top_k]
        pairs = [(query, getattr(d, "page_content", str(d))) for d in docs]
        scores = self.model.predict(pairs)
        scored = list(zip(docs, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [d for d, _ in scored[:top_k]]


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _pdf_to_text(pdfs: List[Any]) -> str:
    buf = []
    for pdf in pdfs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            t = page.extract_text() or ""
            buf.append(t)
    return "\n".join(buf)


def _build_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})


def _ensure_index_from_text(text_chunks: List[str]) -> FAISS:
    embeddings = _build_embeddings()
    vs = FAISS.from_texts(text_chunks, embedding=embeddings)
    vs.save_local("faiss_index")
    return vs


def _load_index() -> Optional[FAISS]:
    try:
        embeddings = _build_embeddings()
        return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    except Exception:
        return None


def _default_chunk(text: str, size: int = 800, overlap: int = 150) -> List[str]:
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        end = min(i + size, n)
        chunks.append(text[i:end])
        i = end - overlap
        if i < 0:
            i = 0
    return [_clean_text(c) for c in chunks if _clean_text(c)]


def evaluate_response(question: str, answer: str, context: str) -> Dict[str, Any]:
    """
    Évalue la qualité d'une réponse générée par rapport à la question et au contexte.
    Retourne un dictionnaire avec différents scores et métriques.
    """
    scores = {
        "relevance_to_question": 0.0,  # Pertinence par rapport à la question
        "relevance_to_context": 0.0,   # Pertinence par rapport au contexte
        "completeness": 0.0,            # Complétude de la réponse
        "length_adequacy": 0.0,         # Adéquation de la longueur
        "overall_score": 0.0            # Score global
    }
    
    if not answer or not question:
        return scores
    
    # Normaliser les textes pour comparaison
    question_lower = question.lower()
    answer_lower = answer.lower()
    context_lower = context.lower()
    
    # 1. Pertinence par rapport à la question (0-1)
    # Vérifie si la réponse contient des mots-clés de la question
    question_words = set(re.findall(r'\b\w+\b', question_lower))
    answer_words = set(re.findall(r'\b\w+\b', answer_lower))
    common_words = question_words.intersection(answer_words)
    
    if len(question_words) > 0:
        scores["relevance_to_question"] = min(1.0, len(common_words) / len(question_words))
    
    # 2. Pertinence par rapport au contexte (0-1)
    # Vérifie si la réponse utilise des informations du contexte
    context_words = set(re.findall(r'\b\w{4,}\b', context_lower))  # Mots de 4+ caractères
    answer_context_words = set(re.findall(r'\b\w{4,}\b', answer_lower))
    context_overlap = context_words.intersection(answer_context_words)
    
    if len(context_words) > 0:
        scores["relevance_to_context"] = min(1.0, len(context_overlap) / min(len(context_words), 50))
    
    # 3. Complétude (0-1)
    # Vérifie si la réponse semble complète (pas de phrases tronquées, longueur raisonnable)
    answer_length = len(answer.strip())
    if answer_length < 20:
        scores["completeness"] = 0.3
    elif answer_length < 50:
        scores["completeness"] = 0.6
    elif answer_length < 200:
        scores["completeness"] = 0.9
    elif answer_length < 500:
        scores["completeness"] = 1.0
    else:
        scores["completeness"] = 0.8  # Trop long peut être moins bien
    
    # Vérifier les phrases incomplètes
    if answer.strip() and not answer.strip().endswith(('.', '!', '?', ':', ';')):
        scores["completeness"] *= 0.9
    
    # 4. Adéquation de la longueur (0-1)
    # Une bonne réponse devrait avoir une longueur appropriée
    ideal_length = len(question) * 3  # Réponse environ 3x la longueur de la question
    length_ratio = answer_length / ideal_length if ideal_length > 0 else 1.0
    
    if 0.5 <= length_ratio <= 2.0:
        scores["length_adequacy"] = 1.0
    elif 0.3 <= length_ratio < 0.5 or 2.0 < length_ratio <= 3.0:
        scores["length_adequacy"] = 0.7
    else:
        scores["length_adequacy"] = 0.4
    
    # 5. Score global (moyenne pondérée)
    weights = {
        "relevance_to_question": 0.3,
        "relevance_to_context": 0.4,
        "completeness": 0.2,
        "length_adequacy": 0.1
    }
    
    scores["overall_score"] = (
        scores["relevance_to_question"] * weights["relevance_to_question"] +
        scores["relevance_to_context"] * weights["relevance_to_context"] +
        scores["completeness"] * weights["completeness"] +
        scores["length_adequacy"] * weights["length_adequacy"]
    )
    
    # Arrondir les scores à 2 décimales
    for key in scores:
        scores[key] = round(scores[key], 2)
    
    return scores


def _get_available_gemini_models(api_key: str) -> List[str]:
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
        return available if available else []
    except Exception as e:
        # If listing fails, return empty list to try common models
        return []


@dataclass
class AdvancedRAGConfig:
    use_rerank: bool = True
    use_rrf: bool = True
    ttl_seconds: int = 300
    k: int = 5
    fetch_k: int = 12
    model_name: str = "google/flan-t5-base"
    gemini_api_key: Optional[str] = None


class AdvancedRAGSystem:
    def __init__(self, config: AdvancedRAGConfig, gemini_api_key: Optional[str] = None):
        self.config = config
        self.cache = QueryCacheTTL(ttl_seconds=config.ttl_seconds)
        self.reranker = DocumentReRanker()
        self.use_gemini = config.model_name == "Gemini"
        
        if self.use_gemini:
            if not GEMINI_AVAILABLE:
                raise ImportError("google-generativeai n'est pas installé. Installez-le avec: pip install google-generativeai")
            api_key = gemini_api_key or config.gemini_api_key
            if not api_key:
                raise ValueError("Clé API Gemini requise")
            # Store API key for model initialization
            self.gemini_api_key = api_key
            genai.configure(api_key=api_key)
            
            # Try to get available models, fallback to common ones
            available_models = _get_available_gemini_models(api_key)
            if not available_models:
                available_models = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro", "gemini-2.0-flash-exp"]
            
            # Try to initialize with first available model using direct API
            self.gemini_model = None
            self.gemini_model_name = None
            for model_name in available_models:
                try:
                    # Try direct API first (more reliable)
                    test_model = genai.GenerativeModel(model_name)
                    # Test with a simple prompt
                    test_response = test_model.generate_content("test")
                    self.gemini_model = test_model
                    self.gemini_model_name = model_name
                    break
                except Exception:
                    # Try langchain wrapper if available
                    if LANGCHAIN_GEMINI_AVAILABLE and ChatGoogleGenerativeAI:
                        try:
                            self.gemini_model = ChatGoogleGenerativeAI(
                                model=model_name,
                                google_api_key=api_key,
                                temperature=0.7,
                                max_output_tokens=2048
                            )
                            self.gemini_model_name = model_name
                            break
                        except Exception:
                            continue
                    continue
            
            if self.gemini_model is None:
                error_msg = f"Aucun modèle Gemini disponible avec cette clé API.\n"
                error_msg += f"Modèles testés: {available_models}\n"
                error_msg += "Vérifiez:\n"
                error_msg += "1. Que votre clé API est valide sur https://ai.google.dev/\n"
                error_msg += "2. Que votre région est supportée par l'API Gemini\n"
                error_msg += "3. Que vous avez activé l'API Generative Language dans Google Cloud Console"
                raise ValueError(error_msg)
            
            self.tokenizer = None
            self.model = None
            self.device = None
        else:
            self.device = torch.device("cpu")
            self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                config.model_name,
                dtype=torch.float32,
                low_cpu_mem_usage=False
            )
            self.model.to(self.device)
            self.model.eval()
            self.gemini_model = None

    def _retrieve(self, query: str) -> List[Any]:
        vs = _load_index()
        if vs is None:
            return []
        return vs.similarity_search(query, k=self.config.k)

    def _retrieve_variants(self, query: str) -> List[List[Any]]:
        variants = [query, query.lower(), re.sub(r"[^\w\s]", " ", query)]
        vs = _load_index()
        if vs is None:
            return []
        lists = []
        for q in variants:
            lists.append(vs.similarity_search(q, k=self.config.k))
        return lists

    def _context(self, docs: List[Any]) -> str:
        parts = []
        for d in docs:
            parts.append(getattr(d, "page_content", str(d)))
        ctx = "\n\n".join(parts)
        return ctx[:2000]

    def answer(self, question: str) -> Tuple[str, Dict[str, Any]]:
        t0 = time.time()
        cached = self.cache.get(question)
        if cached is not None:
            return cached[0], {"cached": True, **cached[1]}

        docs = self._retrieve(question)
        if self.config.use_rrf:
            lists = self._retrieve_variants(question)
            if lists:
                fused = rrf_fusion(lists, k=self.config.k)
                docs = fused

        if self.config.use_rerank and docs:
            docs = self.reranker.rerank(question, docs, top_k=self.config.k)

        ctx = self._context(docs)
        prompt = (
            "Réponds de manière précise et concise à la question en te basant uniquement sur le contexte fourni.\n\n"
            "Contexte:\n"
            f"{ctx}\n\n"
            f"Question: {question}\n\n"
            "Réponse:"
        )
        
        if self.use_gemini:
            # Use Gemini API - try current model first, then fallback to others
            out = None
            last_error = None
            
            # First try the initialized model
            try:
                # Check if using direct API or langchain wrapper
                if hasattr(self.gemini_model, 'generate_content'):
                    # Direct API
                    response = self.gemini_model.generate_content(prompt)
                    out = response.text.strip()
                elif hasattr(self.gemini_model, 'invoke'):
                    # Langchain wrapper
                    response = self.gemini_model.invoke(prompt)
                    out = response.content.strip()
                else:
                    raise ValueError("Format de modèle Gemini non reconnu")
            except Exception as e:
                last_error = e
                # If current model fails, try other available models
                available_models = _get_available_gemini_models(self.gemini_api_key)
                if not available_models:
                    available_models = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro", "gemini-2.0-flash-exp"]
                
                for model_name in available_models:
                    if model_name == self.gemini_model_name:
                        continue  # Already tried
                    try:
                        # Try direct API first (more reliable)
                        test_model = genai.GenerativeModel(model_name)
                        response = test_model.generate_content(prompt)
                        out = response.text.strip()
                        self.gemini_model = test_model
                        self.gemini_model_name = model_name
                        break  # Success
                    except Exception:
                        # Try langchain wrapper if available
                        if LANGCHAIN_GEMINI_AVAILABLE and ChatGoogleGenerativeAI:
                            try:
                                self.gemini_model = ChatGoogleGenerativeAI(
                                    model=model_name,
                                    google_api_key=self.gemini_api_key,
                                    temperature=0.7,
                                    max_output_tokens=2048
                                )
                                self.gemini_model_name = model_name
                                response = self.gemini_model.invoke(prompt)
                                out = response.content.strip()
                                break  # Success
                            except Exception as e:
                                last_error = e
                                continue
                        continue
            
            if out is None:
                error_msg = f"Aucun modèle Gemini disponible.\n"
                error_msg += f"Dernière erreur: {str(last_error)}\n"
                error_msg += "Vérifiez:\n"
                error_msg += "1. Que votre clé API est valide sur https://ai.google.dev/\n"
                error_msg += "2. Que votre région est supportée par l'API Gemini\n"
                error_msg += "3. Que vous avez activé l'API Generative Language dans Google Cloud Console"
                raise ValueError(error_msg)
        else:
            # Use HuggingFace model
            encoding = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            encoding = {k: v.to(self.device) for k, v in encoding.items()}
            with torch.no_grad():
                outputs = self.model.generate(
                    **encoding,
                    max_new_tokens=256,
                    do_sample=False,
                    num_beams=4,
                )
            out = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        t1 = time.time()
        
        # Évaluer la réponse
        evaluation_scores = evaluate_response(question, out, ctx)
        
        meta = {
            "latency_ms": int((t1 - t0) * 1000),
            "sources": self._sources(docs),
            "cache": self.cache.stats(),
            "used_rrf": self.config.use_rrf,
            "used_rerank": self.config.use_rerank,
            "evaluation": evaluation_scores,
        }
        self.cache.set(question, (out, meta))
        return out, meta

    def _sources(self, docs: List[Any]) -> List[str]:
        out = []
        for d in docs:
            md = getattr(d, "metadata", {}) if hasattr(d, "metadata") else {}
            src = md.get("source")
            page = md.get("page")
            if src and page:
                out.append(f"{src} p.{page}")
            elif src:
                out.append(src)
        return sorted(list(set(out)))

    def ensure_index(self, pdfs: List[Any]) -> None:
        if os.path.exists("faiss_index"):
            return
        raw = _pdf_to_text(pdfs)
        chunks = _default_chunk(raw)
        _ensure_index_from_text(chunks)

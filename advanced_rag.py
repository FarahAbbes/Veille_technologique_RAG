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


@dataclass
class AdvancedRAGConfig:
    use_rerank: bool = True
    use_rrf: bool = True
    ttl_seconds: int = 300
    k: int = 5
    fetch_k: int = 12
    model_name: str = "google/flan-t5-base"


class AdvancedRAGSystem:
    def __init__(self, config: AdvancedRAGConfig):
        self.config = config
        self.cache = QueryCacheTTL(ttl_seconds=config.ttl_seconds)
        self.reranker = DocumentReRanker()
        self.device = torch.device("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=False
        )
        self.model.to(self.device)
        self.model.eval()

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
        meta = {
            "latency_ms": int((t1 - t0) * 1000),
            "sources": self._sources(docs),
            "cache": self.cache.stats(),
            "used_rrf": self.config.use_rrf,
            "used_rerank": self.config.use_rerank,
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

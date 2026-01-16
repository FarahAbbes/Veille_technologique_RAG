from typing import List, Literal
import re


class AdaptiveChunker:
    def __init__(self, size: int = 800, overlap: int = 150):
        self.size = size
        self.overlap = overlap

    def chunk(self, text: str, doc_type: Literal["generic", "code", "academic", "legal"] = "generic") -> List[str]:
        text = re.sub(r"\s+", " ", text).strip()
        if doc_type == "code":
            return self._code_chunks(text)
        if doc_type == "academic":
            return self._section_chunks(text, patterns=[r"\babstract\b", r"\bintroduction\b", r"\bmethod", r"\bresults?", r"\bdiscussion", r"\bconclusion"])
        if doc_type == "legal":
            return self._section_chunks(text, patterns=[r"\barticle\b", r"\bsection\b", r"\bclause\b", r"\btitle\b"])
        return self._sliding_chunks(text)

    def _sliding_chunks(self, text: str) -> List[str]:
        res = []
        i = 0
        n = len(text)
        while i < n:
            end = min(i + self.size, n)
            res.append(text[i:end])
            i = end - self.overlap
            if i < 0:
                i = 0
        return [s for s in res if s.strip()]

    def _section_chunks(self, text: str, patterns: List[str]) -> List[str]:
        splits = re.split("|".join(patterns), text, flags=re.IGNORECASE)
        out = []
        for s in splits:
            out.extend(self._sliding_chunks(s))
        return out

    def _code_chunks(self, text: str) -> List[str]:
        blocks = re.split(r"```", text)
        out = []
        for b in blocks:
            out.extend(self._sliding_chunks(b))
        return out


class HybridChunker:
    def __init__(self):
        self.base = AdaptiveChunker()

    def chunk(self, text: str) -> List[str]:
        parts = []
        parts.extend(self.base.chunk(text, "generic"))
        parts.extend(self.base.chunk(text, "academic"))
        return list(dict.fromkeys([p.strip() for p in parts if p.strip()]))


class SmartChunkOptimizer:
    def __init__(self, target_len: int = 600):
        self.target_len = target_len

    def optimize(self, chunks: List[str]) -> List[str]:
        out = []
        buf = ""
        for c in chunks:
            if len(buf) + len(c) <= self.target_len:
                buf += (" " if buf else "") + c
            else:
                if buf:
                    out.append(buf.strip())
                buf = c
        if buf:
            out.append(buf.strip())
        return out

import unittest
from advanced_rag import AdvancedRAGSystem, AdvancedRAGConfig, rrf_fusion, QueryCacheTTL
from adaptive_chunking import AdaptiveChunker, HybridChunker, SmartChunkOptimizer
from multimodal_extractor import MultiFormatExtractor, TableExtractor


class TestFeatures(unittest.TestCase):
    def test_cache_ttl(self):
        cache = QueryCacheTTL(ttl_seconds=1)
        cache.set("a", 1)
        self.assertEqual(cache.get("a"), 1)

    def test_rrf(self):
        l1 = ["a", "b", "c"]
        l2 = ["b", "c", "d"]
        fused = rrf_fusion([l1, l2], k=3)
        self.assertIn("b", fused)

    def test_chunking_generic(self):
        text = "This is a sample text." * 100
        chunks = AdaptiveChunker().chunk(text, "generic")
        self.assertTrue(len(chunks) > 0)

    def test_chunking_hybrid(self):
        text = "Introduction. Methods. Results. Conclusion." * 50
        chunks = HybridChunker().chunk(text)
        self.assertTrue(len(chunks) > 0)

    def test_chunk_optimizer(self):
        chunks = ["a"] * 50
        optimized = SmartChunkOptimizer(target_len=10).optimize(chunks)
        self.assertTrue(len(optimized) > 0)

    def test_multiformat_extractors(self):
        m = MultiFormatExtractor()
        self.assertIsInstance(m.extract_docx("nonexistent.docx"), str)
        self.assertIsInstance(m.extract_pptx("nonexistent.pptx"), list)
        self.assertIsInstance(m.extract_xlsx("nonexistent.xlsx"), list)

    def test_rag_system(self):
        rag = AdvancedRAGSystem(AdvancedRAGConfig(use_rerank=False, use_rrf=False))
        ans, meta = rag.answer("Test question")
        self.assertIsInstance(ans, str)
        self.assertIn("latency_ms", meta)


if __name__ == "__main__":
    unittest.main()

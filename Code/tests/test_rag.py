import unittest
import os
import sys
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autollm.engine.rag_engine import DocumentChunker, VectorStore, RAGPipeline

class TestRAGEngine(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.sample_text = (
            "Standard Operating Procedure for Medical Clinics.\n\n"
            "Section 1: Diabetes Management. Patients diagnosed with Type 2 diabetes should undergo "
            "quarterly HbA1c screening. Target glycemic control is HbA1c below 7.0% for most adults. "
            "First-line pharmacological therapy is Metformin, initiated at 500mg daily.\n\n"
            "Section 2: Hypertension Protocol. Blood pressure exceeding 130/80 mmHg requires lifestyle "
            "interventions including the DASH diet and low sodium consumption. Antihypertensive therapy "
            "includes ACE inhibitors and ARBs as initial choices.\n\n"
            "Section 3: Emergency Burns. Minor second-degree burns should be immersed in cool running water "
            "for 15 minutes. Blisters must remain intact to prevent secondary bacterial infection."
        )

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_document_chunker(self):
        chunks = DocumentChunker.chunk_text(self.sample_text, chunk_size_words=30, overlap_words=5)
        self.assertGreaterEqual(len(chunks), 3)
        self.assertIn("chunk_id", chunks[0])
        self.assertIn("text", chunks[0])

    def test_vector_store_indexing_and_retrieval(self):
        chunks = DocumentChunker.chunk_text(self.sample_text)
        store = VectorStore()
        store.build_index(chunks)
        self.assertTrue(store.is_fitted)

        results = store.query("What is the first-line medication for Type 2 diabetes?", top_k=2)
        self.assertGreater(len(results), 0)
        self.assertIn("Metformin", results[0]["text"])

    def test_rag_pipeline_end_to_end(self):
        pipeline = RAGPipeline()
        rag_spec = pipeline.ingest_document(self.sample_text, "Clinic SOP", self.test_dir)
        self.assertGreater(rag_spec.chunk_count, 0)
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "vector_store.json")))

        answer_data = pipeline.generate_grounded_answer("How should emergency minor burns be treated?")
        self.assertIn("cool running water", answer_data["answer"].lower())
        self.assertGreater(answer_data["confidence"], 0.1)

if __name__ == "__main__":
    unittest.main()

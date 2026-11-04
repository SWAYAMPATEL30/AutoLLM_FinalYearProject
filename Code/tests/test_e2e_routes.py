import unittest
import os
import sys
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autollm.orchestrator import AutoLLMOrchestrator
from autollm.engine.inference import get_inference_engine

class TestE2ERoutes(unittest.TestCase):
    def setUp(self):
        self.test_jobs_dir = tempfile.mkdtemp()
        self.orchestrator = AutoLLMOrchestrator(base_jobs_dir=self.test_jobs_dir, force_cpu=True)

    def tearDown(self):
        shutil.rmtree(self.test_jobs_dir, ignore_errors=True)

    def test_design_preview_route(self):
        result = self.orchestrator.run(
            prompt="Design only: preview a 500MB medical diagnosis assistant",
            training_mode="design_preview"
        )
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["route"], "design_preview")
        self.assertIn("design_spec", result)
        self.assertTrue(os.path.exists(result["recipe_path"]))

    def test_rag_route(self):
        sample_doc = (
            "DJSCE College Policy Handbook 2026.\n\n"
            "Article 1: Attendance Policy. Students must maintain a minimum of 75% attendance in every course "
            "to be eligible to appear for semester end examinations.\n\n"
            "Article 2: Capstone Projects. Capstone project evaluations require complete code documentation, "
            "working prototypes on CPU, and before-and-after benchmark performance reports."
        )
        result = self.orchestrator.run(
            prompt="Build a RAG chatbot to answer questions from college policy",
            training_mode="rag",
            document_text=sample_doc
        )
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["route"], "rag")
        self.assertIn("package", result)
        self.assertIn("eval_report", result)

        job_dir = os.path.join(self.test_jobs_dir, result["job_id"])
        engine = get_inference_engine(job_dir, domain="general")
        resp = engine.generate_response("What is the attendance requirement?")
        self.assertIn("75%", resp)

    def test_from_scratch_mini_pipeline(self):
        result = self.orchestrator.run(
            prompt="Train a tiny transformer from scratch on medical FAQ",
            training_mode="scratch",
            curation_mode="synthetic"
        )
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["route"], "from_scratch")
        self.assertIn("package", result)
        self.assertIn("metrics", result)

        # Confirm INT8 quantization happened
        pkg = result["package"]
        self.assertIsNotNone(pkg.get("quantized_weights_path"))
        self.assertTrue(os.path.exists(pkg["quantized_weights_path"]))

        # Confirm inference engine mounts real MiniGPT
        job_dir = os.path.join(self.test_jobs_dir, result["job_id"])
        engine = get_inference_engine(job_dir, domain="medical")
        resp = engine.generate_response("What are the common symptoms of diabetes?", max_tokens=30)
        self.assertTrue(len(resp) > 0)

if __name__ == "__main__":
    unittest.main()

import unittest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autollm.intent.parser import IntentParser
from autollm.intent.schemas import RouteType, TaskType

class TestIntentParser(unittest.TestCase):
    def setUp(self):
        self.parser = IntentParser(force_cpu=True)

    def test_route_design_preview(self):
        prompts = [
            "Design only: give me an architecture spec for a 1B model",
            "Preview architecture and estimate memory for 500MB mobile deployment",
            "What hardware plan and parameters do I need for a 2GB model?"
        ]
        for p in prompts:
            spec = self.parser.parse(p)
            self.assertEqual(spec.route, RouteType.DESIGN_PREVIEW, f"Failed on prompt: {p}")

    def test_route_rag(self):
        prompts = [
            "Build a RAG chatbot that answers from company policy PDF",
            "Create a document retrieval system grounded on our employee handbook",
            "Search documents and provide factual vector answers"
        ]
        for p in prompts:
            spec = self.parser.parse(p)
            self.assertEqual(spec.route, RouteType.RAG, f"Failed on prompt: {p}")

    def test_route_from_scratch(self):
        prompts = [
            "Train a tiny transformer from scratch on Shakespeare poetry",
            "Build an educational toy language model from random weights",
            "Create a small scratch transformer for sonnet generation"
        ]
        for p in prompts:
            spec = self.parser.parse(p)
            self.assertEqual(spec.route, RouteType.FROM_SCRATCH, f"Failed on prompt: {p}")

    def test_route_peft_finetune_default(self):
        prompts = [
            "Create a chatbot model for medical FAQ answering under 2GB size",
            "Fine-tune an assistant for legal contract compliance with LoRA",
            "Build an agricultural crop advisor bot for farmers"
        ]
        for p in prompts:
            spec = self.parser.parse(p)
            self.assertEqual(spec.route, RouteType.PEFT_FINETUNE, f"Failed on prompt: {p}")

    def test_domain_extraction(self):
        self.assertEqual(self.parser.parse("A medical doctor bot").domain, "medical")
        self.assertEqual(self.parser.parse("Legal contract NDA analyzer").domain, "legal")
        self.assertEqual(self.parser.parse("Agricultural lemon crop disease helper").domain, "agriculture")
        self.assertEqual(self.parser.parse("Python coding function generator").domain, "code")

if __name__ == "__main__":
    unittest.main()

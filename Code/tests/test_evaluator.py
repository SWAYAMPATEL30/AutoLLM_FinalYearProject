import unittest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autollm.engine.evaluator import ModelEvaluator, generate_document_test_set

class TestEvaluator(unittest.TestCase):
    def test_score_response_good_match(self):
        expected = "Type 2 diabetes symptoms include increased thirst, frequent urination, and fatigue."
        response = "The symptoms of Type 2 diabetes are frequent urination, polydipsia (increased thirst), and notable fatigue."
        score, failure = ModelEvaluator._score_response(response, expected, "medical")
        self.assertGreaterEqual(score, 0.55)
        self.assertIsNone(failure)

    def test_score_response_out_of_domain(self):
        expected = "An indemnification clause protects one party against breach of contract losses."
        response = "As an AI language model, I cannot provide legal advice on this topic."
        score, failure = ModelEvaluator._score_response(response, expected, "legal")
        self.assertLess(score, 0.55)
        self.assertEqual(failure, "GENERIC_OUT_OF_DOMAIN")

    def test_score_response_missing_nomenclature(self):
        expected = "A healthy lemon tree typically has a lifespan of 50 to 100 years."
        response = "Lemon trees live for a while under good conditions."
        score, failure = ModelEvaluator._score_response(response, expected, "agriculture")
        self.assertLess(score, 0.55)
        self.assertEqual(failure, "MISSING_NOMENCLATURE")

    def test_generate_document_test_set(self):
        doc = (
            "The Company vacation policy allows 20 days of paid time off per calendar year. "
            "All requests for extended leave exceeding 5 consecutive days require VP approval. "
            "Unused vacation days roll over up to a maximum limit of 5 days."
        )
        test_pairs = generate_document_test_set(doc, max_samples=3)
        self.assertGreater(len(test_pairs), 0)
        self.assertLessEqual(len(test_pairs), 3)
        for pair in test_pairs:
            self.assertIn("instruction", pair)
            self.assertIn("output", pair)
            self.assertGreater(len(pair["instruction"]), 5)
            self.assertGreater(len(pair["output"]), 5)

if __name__ == "__main__":
    unittest.main()

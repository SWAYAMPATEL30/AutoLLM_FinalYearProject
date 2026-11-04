import unittest
import os
import sys
import tempfile
import shutil
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autollm.engine.scratch_transformer import MiniGPT, WordTokenizer
from autollm.engine.inference import MiniGPTInferenceEngine, DomainInferenceEngine, get_inference_engine

class TestInferenceEngine(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.test_dir, "model.pt")
        self.tok_path = os.path.join(self.test_dir, "tokenizer.json")

        tok = WordTokenizer()
        tok.build_vocab(["What are the symptoms of Type 2 Diabetes?", "Symptoms include frequent thirst and fatigue."])
        tok.save(self.tok_path)

        model = MiniGPT(vocab_size=tok.vocab_size, embed_dim=32, num_layers=2, num_heads=2, max_seq_len=64)
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": {
                "vocab_size": tok.vocab_size,
                "embed_dim": 32,
                "num_layers": 2,
                "num_heads": 2,
                "max_seq_len": 64
            }
        }, self.model_path)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_minigpt_inference_engine(self):
        engine = MiniGPTInferenceEngine(self.model_path, self.tok_path)
        out = engine.generate_response("What are the symptoms?", max_tokens=10, temperature=0.7)
        self.assertIsInstance(out, str)
        self.assertGreater(len(out), 0)

    def test_get_inference_engine_fallback(self):
        engine = get_inference_engine("nonexistent_job_12345")
        self.assertIsInstance(engine, DomainInferenceEngine)

if __name__ == "__main__":
    unittest.main()

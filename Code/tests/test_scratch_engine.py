import unittest
import os
import sys
import tempfile
import shutil
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autollm.engine.scratch_transformer import MiniGPT, CharTokenizer, WordTokenizer, load_tokenizer

class TestScratchEngine(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.tok_path = os.path.join(self.test_dir, "tokenizer.json")
        self.word_tok_path = os.path.join(self.test_dir, "word_tokenizer.json")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_char_tokenizer_lifecycle(self):
        tok = CharTokenizer()
        corpus = ["Question: What is hypertension?\nAnswer: High blood pressure."]
        tok.build_vocab(corpus)
        self.assertGreater(tok.vocab_size, len(tok.special_tokens))

        encoded = tok.encode("hypertension")
        decoded = tok.decode(encoded)
        self.assertEqual(decoded, "hypertension")

        tok.save(self.tok_path)
        loaded_tok = load_tokenizer(self.tok_path)
        self.assertEqual(loaded_tok.vocab_size, tok.vocab_size)

    def test_word_tokenizer_lifecycle(self):
        tok = WordTokenizer()
        corpus = [
            "What are the primary symptoms of Type 2 Diabetes?",
            "Symptoms include frequent urination, excessive thirst, and fatigue."
        ]
        tok.build_vocab(corpus)
        self.assertGreater(tok.vocab_size, 10)

        encoded = tok.encode("What are the primary symptoms?")
        decoded = tok.decode(encoded)
        self.assertIn("What", decoded)
        self.assertIn("symptoms", decoded)

        tok.save(self.word_tok_path)
        loaded_tok = load_tokenizer(self.word_tok_path)
        self.assertEqual(loaded_tok.vocab_size, tok.vocab_size)
        self.assertIsInstance(loaded_tok, WordTokenizer)

    def test_minigpt_forward_and_generate(self):
        tok = WordTokenizer()
        tok.build_vocab(["hello world test sequence for transformer", "medical triage and symptoms"])
        model = MiniGPT(vocab_size=tok.vocab_size, embed_dim=32, num_layers=2, num_heads=2, max_seq_len=64)

        x = torch.tensor([[tok.bos_id, 5, 6, 7]], dtype=torch.long)
        logits, loss = model(x, targets=x)
        self.assertIsNotNone(loss)
        self.assertEqual(logits.shape[0], 1)
        self.assertEqual(logits.shape[1], 4)

        gen = model.generate(x, max_new_tokens=5, temperature=0.8, top_p=0.9)
        self.assertGreaterEqual(gen.shape[1], 4)

if __name__ == "__main__":
    unittest.main()

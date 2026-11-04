import unittest
import os
import sys
import tempfile
import shutil
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autollm.engine.quantizer import ModelQuantizer

class SimpleTestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class TestQuantizer(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.fp32_path = os.path.join(self.test_dir, "model.pt")
        self.quant_path = os.path.join(self.test_dir, "model_quantized.pt")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_dynamic_quantization(self):
        model = SimpleTestModel()
        torch.save(model.state_dict(), self.fp32_path)

        metrics = ModelQuantizer.quantize_pytorch_model(
            model=model,
            output_quant_path=self.quant_path,
            original_model_path=self.fp32_path
        )

        self.assertTrue(os.path.exists(self.quant_path))
        self.assertIn("quantized_size_mb", metrics)
        self.assertIn("compression_ratio_pct", metrics)
        self.assertEqual(metrics["quantization_type"], "PyTorch Dynamic INT8 (torch.qint8)")

if __name__ == "__main__":
    unittest.main()

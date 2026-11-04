import unittest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autollm.engine.design_preview import ArchitecturePlanner

class TestDesignPreview(unittest.TestCase):
    def test_design_1b_tier(self):
        spec = ArchitecturePlanner.design_architecture(target_size_mb=4096.0, domain="finance")
        self.assertIn("1B", spec.target_name)
        self.assertGreater(spec.parameter_count_m, 100)
        self.assertGreater(spec.model_weights_fp32_mb, 400)
        self.assertLess(spec.model_weights_int8_mb, spec.model_weights_fp32_mb)

    def test_design_minigpt_tier(self):
        spec = ArchitecturePlanner.design_architecture(target_size_mb=50.0, domain="creative")
        self.assertIn("MiniGPT", spec.target_name)
        self.assertLess(spec.parameter_count_m, 20)
        self.assertGreater(spec.hidden_dimension, 64)

    def test_memory_and_kv_cache(self):
        # 50MB model requires ~30MB training RAM which is always feasible on any system
        spec = ArchitecturePlanner.design_architecture(target_size_mb=50.0, domain="medical")
        self.assertGreater(spec.optimizer_memory_adamw_mb, 0)
        self.assertGreater(spec.kv_cache_per_seq_kb, 0)
        self.assertIn("FEASIBLE", spec.feasibility_status.upper())

if __name__ == "__main__":
    unittest.main()

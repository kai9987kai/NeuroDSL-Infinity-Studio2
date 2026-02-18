import torch
import unittest
from network import MambaBlock, LiquidNeuralLayer, HyperTransformerBlock, ModernMLP
from parser_utils import parse_program, validate_dsl

class TestPhase12ASI(unittest.TestCase):
    def test_mamba_block(self):
        mamba = MambaBlock(dim=32)
        x = torch.randn(4, 32)
        out = mamba(x)
        self.assertEqual(out.shape, (4, 32))

    def test_liquid_layer(self):
        liquid = LiquidNeuralLayer(dim=32)
        x = torch.randn(4, 32)
        out = liquid(x)
        self.assertEqual(out.shape, (4, 32))

    def test_hyper_layer(self):
        hyper = HyperTransformerBlock(dim=32)
        x = torch.randn(4, 32)
        out = hyper(x)
        self.assertEqual(out.shape, (4, 32))

    def test_dsl_parsing_asi(self):
        dsl = "mamba: [64], liquid: [64], hyper: [64], [64, 10]"
        issues, defs = validate_dsl(dsl)
        self.assertIsNone(issues[0] if issues and issues[0][0] == "ERROR" else None)
        self.assertEqual(len(defs), 4)

    def test_asi_omni_preset(self):
        from parser_utils import DSL_PRESETS
        dsl = DSL_PRESETS["ASI Omni-Intelligence"]
        issues, defs = validate_dsl(dsl)
        self.assertEqual(len(issues), 0)
        model = ModernMLP(defs)
        x = torch.randn(1, 256)
        out = model(x)
        self.assertEqual(out.shape, (1, 10))

if __name__ == "__main__":
    unittest.main()

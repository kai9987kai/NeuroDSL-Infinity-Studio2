import unittest
import torch
import torch.nn as nn
from neuro_sdk import NeuroLab
from parser_utils import DSL_PRESETS, parse_program, create_modern_nn
from unittest.mock import MagicMock, patch

class TestPhase10(unittest.TestCase):
    
    def test_omni_preset(self):
        # Verify the Omni-Model preset exists and parses
        preset_code = DSL_PRESETS.get("Omni-Model (SOTA)")
        self.assertIsNotNone(preset_code)
        
        layer_defs = parse_program(preset_code)
        self.assertIsNotNone(layer_defs)
        
        # Verify specific layers are present
        types = [l['type'] for l in layer_defs]
        self.assertIn('conv3d', types)
        self.assertIn('trans', types)
        self.assertIn('moe', types)
        
        # Build it to check dimensions
        # Note: This might be large, so we perform a dry run build
        try:
            model = create_modern_nn(layer_defs)
            self.assertIsInstance(model, nn.Module)
            print(f"Omni-Model Built. Params: {sum(p.numel() for p in model.parameters())}")
        except Exception as e:
            self.fail(f"Omni-Model build failed: {e}")

    def test_sdk_build_train(self):
        sdk = NeuroLab()
        dsl = "[16, 16], [16, 10]"
        model = sdk.build(dsl)
        self.assertIsInstance(model, nn.Module)
        
        # Train on dummy data
        X = torch.randn(10, 16)
        y = torch.randn(10, 10) # Match output dim 10 and MSE float type
        
        loss = sdk.train(X, y, epochs=2)
        self.assertIsInstance(loss, float)
        self.assertLess(loss, 100) # Arbitrary check
        report = sdk.train_report()
        self.assertIn("best_loss", report)
        self.assertIn("epochs_ran", report)

    def test_sdk_capabilities_and_infer(self):
        sdk = NeuroLab()
        caps = sdk.capabilities()
        self.assertIn("autopilot", caps["features"])

        sdk.build("[4, 8], [8, 2]")
        out = sdk.infer([0.1, 0.2, 0.3, 0.4])
        self.assertEqual(len(out), 2)
        self.assertTrue(all(isinstance(v, float) for v in out))

    def test_sdk_autopilot_offline(self):
        sdk = NeuroLab()
        report = sdk.autopilot(
            objective="Fast compact regression model",
            input_dim=8,
            output_dim=3,
            candidates=2,
            epochs_per_candidate=1,
            samples=24,
            seed=123,
            use_ai=False,
        )
        self.assertIn("best_dsl", report)
        self.assertTrue(report["best_dsl"])
        self.assertIsNotNone(sdk.model)

    @patch('codex_client.optimize_dsl')
    def test_sdk_evolve(self, mock_optimize):
        sdk = NeuroLab(api_key="test_key")
        
        # Mock successful evolution
        mock_optimize.return_value = (True, "[16, 32], [32, 10]")
        
        new_dsl = sdk.evolve("[16, 10]")
        self.assertEqual(new_dsl, "[16, 32], [32, 10]")
        
        # Mock output
        mock_optimize.assert_called_once()

if __name__ == '__main__':
    unittest.main()

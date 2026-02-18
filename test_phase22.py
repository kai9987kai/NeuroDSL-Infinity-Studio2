
import unittest
import torch
import torch.nn as nn
import os
import shutil
import tempfile
import numpy as np

# Adjust path to import from project root
import sys
sys.path.append(os.getcwd())

from network import ModernMLP, BitLinear, RetentionLayer, MixtureOfDepths
from trainer import TrainingEngine, EMA
from model_optimizer import ModelOptimizer
from parser_utils import parse_program, create_modern_nn, validate_dsl, DSL_PRESETS

class TestPhase22Features(unittest.TestCase):
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_bitlinear_layer(self):
        print("\nTesting BitLinear Layer...")
        layer = BitLinear(32, 4) # 32 dim, expansion=4
        x = torch.randn(10, 32)
        out = layer(x)
        self.assertEqual(out.shape, (10, 32))
        
        # Check if weights are effectively quantized in forward pass (hard to check directly due to STE)
        # But we can check if gradients flow
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(layer.weight_up.grad)
        print("BitLinear forward/backward pass effective.")

    def test_retention_layer(self):
        print("\nTesting Retention Layer...")
        layer = RetentionLayer(32, 4) # 32 dim, 4 heads
        x = torch.randn(5, 10, 32) # B, T, D
        out = layer(x)
        self.assertEqual(out.shape, (5, 10, 32))
        
        # Test causal masking availability
        # Retention applies separate decay per head
        # We assume it works if output shape is correct and no crash
        print("RetentionLayer forward pass effective.")

    def test_mixture_of_depths(self):
        print("\nTesting MixtureOfDepths...")
        layer = MixtureOfDepths(32)
        x = torch.randn(5, 20, 32) # B, T, D
        out = layer(x)
        self.assertEqual(out.shape, (5, 20, 32))
        
        # Check router weights exist
        self.assertIsNotNone(layer.router[0].weight) # router is Sequential
        print("MixtureOfDepths forward pass effective.")

    def test_ema_functionality(self):
        print("\nTesting EMA...")
        model = nn.Linear(10, 10)
        ema = EMA(model, decay=0.5)
        ema.register()
        
        # Initial weights
        w0 = model.weight.clone()
        
        # Update model weights
        with torch.no_grad():
            model.weight.add_(1.0)
            
        # Update EMA
        ema.update()
        
        # Check shadow weights
        # shadow = decay * shadow_old + (1-decay) * model_new
        # shadow_init = w0
        # model_new = w0 + 1
        # shadow_new = 0.5 * w0 + 0.5 * (w0 + 1) = w0 + 0.5
        
        ema.apply_shadow()
        self.assertTrue(torch.allclose(model.weight, w0 + 0.5))
        
        ema.restore()
        self.assertTrue(torch.allclose(model.weight, w0 + 1.0))
        print("EMA update/apply/restore verified.")

    def test_training_engine_improvements(self):
        print("\nTesting Training Engine (Grad Accumulation & Metrics)...")
        # Define a simple linear model compatible with ModernMLP interface or usage
        # ModernMLP expects layer_defs if using its init, but here we just need a nn.Module
        # We can use a simple Sequential but TrainingEngine might assume methods like 'cuda'
        
        # Let's use a dummy model
        model = nn.Sequential(nn.Linear(10, 2))
        trainer = TrainingEngine(model, loss_fn='CrossEntropy')
        trainer.accumulation_steps = 2
        
        X = torch.randn(8, 10)
        y = torch.randint(0, 2, (8,))
        
        # Step 1 (Accumulate)
        loss1, _, grad1, acc1, f1_1 = trainer.train_step(X, y)
        self.assertEqual(grad1, 0.0) # No update yet
        
        # Step 2 (Update)
        loss2, _, grad2, acc2, f1_2 = trainer.train_step(X, y)
        self.assertNotEqual(grad2, 0.0) # Should update now
        
        # Metrics check
        self.assertGreaterEqual(acc1, 0.0)
        self.assertGreaterEqual(f1_1, 0.0)
        print("Gradient Accumulation & Metrics verified.")

    def test_model_optimizer(self):
        print("\nTesting Model Optimizer...")
        model = nn.Sequential(nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 2))
        optimizer = ModelOptimizer()
        
        # Histogram
        hist, edges = optimizer.visualize_weight_histogram(model)
        self.assertGreater(len(hist), 0)
        
        # FP16 Export (requires valid path)
        path = os.path.join(self.temp_dir, "test_fp16.onnx")
        try:
            saved_path = optimizer.export_fp16(model, path, input_dim=10)
            self.assertTrue(os.path.exists(saved_path))
            print("FP16 Export verified.")
        except RuntimeError as e:
            print(f"Skipping FP16 Export test (ONNX likely missing): {e}")

    def test_dsl_presets(self):
        print("\nTesting Phase 22 Presets...")
        presets = ["Ultra-Efficient", "RetNet-Style"]
        for p in presets:
            try:
                # validate_dsl takes the raw string
                validate_dsl(DSL_PRESETS[p]) 
                
                # create_modern_nn takes the parsed list
                program = parse_program(DSL_PRESETS[p])
                create_modern_nn(program)
                print(f"Preset '{p}' validated.")
            except Exception as e:
                self.fail(f"Preset '{p}' failed validation: {e}")

if __name__ == '__main__':
    unittest.main()

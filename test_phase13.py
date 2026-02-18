import torch
import torch.nn as nn
import unittest
import os
from network import ModernMLP
from parser_utils import parse_program, validate_dsl
from data_gen import VisualDataEngine, TemporalGenerator
from singularity_tools import WeightMorpher, AdversarialToolbox, NeuralSynthesizer

class TestPhase13Singularity(unittest.TestCase):
    def test_fractal_gen(self):
        from PIL import Image
        img = VisualDataEngine.generate_fractal(width=64, height=64)
        self.assertIsInstance(img, Image.Image)
        self.assertEqual(img.size, (64, 64))

    def test_weight_morphing(self):
        # Create two identical models
        dsl = "[8, 16], [16, 8]"
        defs = parse_program(dsl)
        model_a = ModernMLP(defs)
        model_b = ModernMLP(defs)
        
        # Modify weights of model_b
        with torch.no_grad():
            for p in model_b.parameters():
                p.add_(1.0)
                
        morphed = WeightMorpher.interpolate_models(model_a, model_b, alpha=0.5)
        
        # Check if weights are interpolated
        param_a = next(model_a.parameters())
        param_b = next(model_b.parameters())
        param_m = next(morphed.parameters())
        
        expected = (param_a + param_b) / 2.0
        torch.testing.assert_close(param_m, expected)

    def test_neural_scripting(self):
        dsl = 'script: ["x * 2.0"]'
        defs = parse_program(dsl)
        model = ModernMLP(defs)
        x = torch.ones(1, 8)
        out = model(x)
        self.assertEqual(out[0,0], 2.0)

    def test_adversarial_fgsm(self):
        dsl = "[8, 10]"
        defs = parse_program(dsl)
        model = ModernMLP(defs)
        x = torch.randn(1, 8)
        y = torch.tensor([5])
        
        perturbed_x = AdversarialToolbox.fgsm_attack(model, nn.CrossEntropyLoss(), x, y, epsilon=0.1)
        self.assertEqual(perturbed_x.shape, x.shape)
        self.assertFalse(torch.equal(x, perturbed_x))

    def test_neural_synthesizer(self):
        dsl = "[32, 10]"
        defs = parse_program(dsl)
        model = ModernMLP(defs)
        x, y = NeuralSynthesizer.synthesize_dataset(model, n_samples=5)
        self.assertEqual(x.shape, (5, 32))
        self.assertEqual(y.shape, (5, 10))

if __name__ == "__main__":
    unittest.main()

import unittest
import torch
from fractal_compression import FractalBlock, NeuralPruner

class TestFractalCompression(unittest.TestCase):
    def test_fractal_block(self):
        layer = FractalBlock(16, 8, seed_size=4)
        x = torch.randn(2, 16)
        out = layer(x)
        self.assertEqual(out.shape, (2, 8))
        
        # Verify weight reconstruction
        w = layer.get_fractal_weights()
        self.assertEqual(w.shape, (8, 16))

    def test_neural_pruner(self):
        model = torch.nn.Sequential(torch.nn.Linear(16, 8))
        # Initial weights
        model[0].weight.data.fill_(0.005) # Below default threshold
        pruned = NeuralPruner.prune_model(model, threshold=0.01)
        self.assertEqual(pruned, 16 * 8) # All should be pruned
        self.assertTrue(torch.all(model[0].weight < 0.001))

if __name__ == "__main__":
    unittest.main()

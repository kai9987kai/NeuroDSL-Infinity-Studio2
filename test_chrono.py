import unittest
import torch
from chrono_core import ChronoFoldingLayer

class TestChronoFolding(unittest.TestCase):
    def test_chrono_folding_flow(self):
        dim = 8
        layer = ChronoFoldingLayer(dim, window_size=3)
        x = torch.randn(2, dim)
        
        # Initial forward pass
        out = layer(x)
        self.assertEqual(out.shape, (2, dim))
        
        # Multiple passes to populate buffers
        for _ in range(5):
            out = layer(torch.randn(2, dim))
            
        self.assertEqual(len(layer.past_buffer), 3)
        self.assertEqual(len(layer.future_buffer), 3)

if __name__ == "__main__":
    unittest.main()

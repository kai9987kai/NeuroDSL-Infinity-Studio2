import unittest
import torch
import torch.nn as nn
from network import (
    MultiHeadLatentAttention, MambaConvBlock, 
    SparseTopKLayer, SaliencyPruningLayer,
    BitNetLayer, ReversibleResidualBlock, MixtureOfAttention,
    ModernMLP
)
from trainer import TrainingEngine

class TestPhase30SingularityPlus(unittest.TestCase):
    
    def setUp(self):
        self.dim = 32
        self.batch_size = 4
        self.x = torch.randn(self.batch_size, self.dim)
        
    def test_mhla_layer(self):
        layer = MultiHeadLatentAttention(self.dim, num_heads=4)
        out = layer(self.x)
        self.assertEqual(out.shape, (self.batch_size, self.dim))
        out.sum().backward()
                
    def test_mambaconv_layer(self):
        layer = MambaConvBlock(self.dim)
        out = layer(self.x)
        self.assertEqual(out.shape, (self.batch_size, self.dim))
        out.sum().backward()

    def test_bitnet_layer(self):
        layer = BitNetLayer(self.dim, self.dim)
        out = layer(self.x)
        self.assertEqual(out.shape, (self.batch_size, self.dim))
        out.sum().backward()
        # Verify ternary quantization logic doesn't kill grads
        for p in layer.parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad)

    def test_reversible_block(self):
        layer = ReversibleResidualBlock(self.dim)
        out = layer(self.x)
        self.assertEqual(out.shape, (self.batch_size, self.dim))
        out.sum().backward()

    def test_moa_layer(self):
        layer = MixtureOfAttention(self.dim)
        out = layer(self.x)
        self.assertEqual(out.shape, (self.batch_size, self.dim))
        out.sum().backward()
        
    def test_modern_mlp_integration(self):
        layer_defs = [
            {'type': 'bitnet', 'in': self.dim, 'out': self.dim},
            {'type': 'rev_res', 'dim': self.dim},
            {'type': 'moa', 'dim': self.dim},
            {'type': 'mhla', 'dim': self.dim, 'heads': 4}
        ]
        model = ModernMLP(layer_defs)
        out = model(self.x)
        self.assertEqual(out.shape, (self.batch_size, self.dim))
        
    def test_lion_optimizer(self):
        model = nn.Linear(self.dim, 1)
        engine = TrainingEngine(model, use_lion=True, base_lr=1e-4)
        X = torch.randn(10, self.dim)
        y = torch.randn(10, 1)
        loss, lr, gn, acc, f1 = engine.train_step(X, y)
        self.assertGreater(loss, 0)

if __name__ == '__main__':
    unittest.main()

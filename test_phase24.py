import unittest
import torch
from network import HypersphereLayer, PoincareLayer, TopologicalAttention, ModernMLP
from topology_engine import PersistentHomology, ManifoldDrifter
from parser_utils import parse_program, create_modern_nn

class TestPhase24(unittest.TestCase):
    def test_hypersphere_layer(self):
        dim = 32
        layer = HypersphereLayer(dim)
        x = torch.randn(2, dim)
        out = layer(x)
        self.assertEqual(out.shape, (2, dim))
        # Check if norm is equal to scale
        norms = out.norm(p=2, dim=-1)
        torch.testing.assert_close(norms, layer.scale.expand(2))

    def test_poincare_layer(self):
        dim = 16
        layer = PoincareLayer(dim)
        x = torch.randn(2, dim) * 10 # Large values
        out = layer(x)
        self.assertEqual(out.shape, (2, dim))
        # Check if norm < 1
        norms = out.norm(p=2, dim=-1)
        self.assertTrue(torch.all(norms < 1.0))

    def test_topo_attention(self):
        dim = 32
        layer = TopologicalAttention(dim, heads=4)
        x = torch.randn(2, 10, dim)
        out = layer(x)
        self.assertEqual(out.shape, (2, 10, dim))

    def test_topology_engine(self):
        engine = PersistentHomology()
        points = torch.randn(50, 64)
        metrics = engine.compute_persistence(points)
        self.assertIn('void_score', metrics)
        self.assertIn('avg_nn_dist', metrics)

    def test_manifold_drifter(self):
        drifter = ManifoldDrifter(manifold_type='sphere')
        v1 = torch.randn(64)
        v2 = torch.randn(64)
        steps = 5
        path = drifter.interpolate(v1, v2, steps=steps)
        self.assertEqual(path.shape, (steps, 64))
        
        # Test linear
        drifter.manifold_type = 'euclidean'
        path_eucl = drifter.interpolate(v1, v2, steps=steps)
        self.assertEqual(path_eucl.shape, (steps, 64))

    def test_dsl_integration(self):
        prog = "sphere: [64], poincare: [64], topo_attn: [64, 4]"
        layer_defs = parse_program(prog)
        self.assertEqual(len(layer_defs), 3)
        model = create_modern_nn(layer_defs)
        self.assertIsInstance(model.layers[0], HypersphereLayer)
        self.assertIsInstance(model.layers[1], PoincareLayer)
        self.assertIsInstance(model.layers[2], TopologicalAttention)

if __name__ == '__main__':
    unittest.main()

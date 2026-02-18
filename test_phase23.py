import unittest
import torch
import torch.nn as nn
from network import GraphConv, DiffLogic, ConceptNeuron, ModernMLP
from knowledge_engine import RuleInjector, ReasoningTracer
from parser_utils import parse_program, create_modern_nn, validate_dsl

class TestPhase23(unittest.TestCase):
    def test_graph_conv(self):
        dim = 32
        layer = GraphConv(dim)
        # B, N, D
        x = torch.randn(2, 10, dim)
        out = layer(x) # Should use learned static adj
        self.assertEqual(out.shape, (2, 10, dim))
        
        # Test 2D input fallback
        x2 = torch.randn(2, dim)
        out2 = layer(x2)
        self.assertEqual(out2.shape, (2, dim))
        
        # Test with explicit adjacency
        adj = torch.randn(2, 10, 10)
        out3 = layer(x, adj=adj)
        self.assertEqual(out3.shape, (2, 10, dim))

    def test_diff_logic(self):
        dim = 16
        layer = DiffLogic(dim, num_rules=8)
        x = torch.rand(2, dim) # [0, 1] inputs for logic
        out = layer(x)
        self.assertEqual(out.shape, (2, dim))
        
        # Gradient check
        x.requires_grad = True
        out = layer(x)
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)

    def test_concept_neuron(self):
        dim = 32
        layer = ConceptNeuron(dim)
        x = torch.randn(2, dim)
        out = layer(x)
        self.assertEqual(out.shape, (2, dim))
        
        loss = layer.orthogonality_loss()
        self.assertTrue(loss >= 0)
        
    def test_rule_injector(self):
        injector = RuleInjector()
        injector.add_rule(0, 1) # IF input[0] THEN output[1]
        
        # Case: A=1, B=0 (Bad)
        inputs = torch.tensor([[1.0, 0.0]], requires_grad=True)
        outputs = torch.tensor([[0.0, 1.0]], requires_grad=True) 
        
        # Note: RuleInjector indices refer to column 0 of inputs and column 1 of outputs
        # Wait, usually rule is about input or output features. 
        # My implementation: inputs[:, if_idx], outputs[:, then_idx]
        
        # Let's assume rule is "If input feature 0 is high, output feature 1 should be high"
        # Input: [1.0, ...], Output: [0.0, ...] -> Violation
        
        loss = injector.compute_loss(inputs.unsqueeze(0), outputs.unsqueeze(0)) # Add batch dim
        # Loss = ReLU(1 - 0) = 1
        # Wait, indices 0 and 1.
        
        # Correction: indices are column indices.
        # inputs shape (1, 2). outputs shape (1, 2).
        # rule if=0, then=1.
        # Input col 0 is 1.0. Output col 1 is 1.0. 
        # Loss = ReLU(1 - 1) = 0. Good.
        
        # Bad case: Output col 1 is 0.0.
        outputs_bad = torch.tensor([[0.0, 0.0]], requires_grad=True)
        loss_bad = injector.compute_loss(inputs, outputs_bad)
        # Loss = ReLU(1.0 - 0.0) = 1.0
        self.assertGreater(loss_bad.item(), 0.0)
        
        injector.parse_text_rule("IF 0 THEN 1")
        self.assertEqual(len(injector.rules), 2)

    def test_dsl_parsing(self):
        prog = "graph: [64], logic: [64, 16], concept: [64]"
        layer_defs = parse_program(prog)
        self.assertEqual(len(layer_defs), 3)
        self.assertEqual(layer_defs[0]['type'], 'graph_conv')
        self.assertEqual(layer_defs[1]['type'], 'diff_logic')
        self.assertEqual(layer_defs[2]['type'], 'concept')
        
    def test_modern_mlp_integration(self):
        prog = "graph: [32], logic: [32, 8], concept: [32]"
        layer_defs = parse_program(prog)
        model = create_modern_nn(layer_defs)
        
        x = torch.randn(1, 10, 32)
        out = model(x)
        self.assertEqual(out.shape, (1, 10, 32))
        
        # Aux loss check
        aux = model.get_aux_loss()
        # Should include orthogonality loss from ConceptNeuron
        self.assertTrue(torch.is_tensor(aux))

if __name__ == '__main__':
    unittest.main()

import unittest
import torch
from quantum_core import QuantumLinear, EntanglementGate

class TestQuantumCore(unittest.TestCase):
    def test_quantum_linear_flow(self):
        layer = QuantumLinear(8, 4)
        x = torch.randn(2, 8)
        out = layer(x)
        self.assertEqual(out.shape, (2, 4))
        # Verify gradient flow
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(layer.weight_real.grad)
        self.assertIsNotNone(layer.weight_imag.grad)

    def test_entanglement_gate(self):
        gate = EntanglementGate(8)
        xa = torch.randn(2, 8)
        xb = torch.randn(2, 8)
        oa, ob = gate(xa, xb)
        self.assertEqual(oa.shape, (2, 8))
        self.assertEqual(ob.shape, (2, 8))
        # Modification check
        self.assertFalse(torch.equal(xa, oa))

if __name__ == "__main__":
    unittest.main()

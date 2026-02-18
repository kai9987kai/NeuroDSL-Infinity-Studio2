import unittest
import torch
import parser_utils
from network import ModernMLP
import subprocess

class TestPhase9(unittest.TestCase):
    
    def test_conv3d_parsing(self):
        code = "conv3d: [64, 3]"
        layer_defs = parser_utils.parse_program(code)
        self.assertIsNotNone(layer_defs)
        self.assertEqual(len(layer_defs), 1)
        self.assertEqual(layer_defs[0]['type'], 'conv3d')
        self.assertEqual(layer_defs[0]['dim'], 64)
        self.assertEqual(layer_defs[0]['kernel'], 3)

    def test_conv3d_build_and_forward(self):
        # DSL: [16, 16], conv3d: [16, 3], [16, 10]
        # Input: (B, 16)
        # Conv3DBlock expects (B, C, D, H, W) or auto-reshapes (B, D) -> (B, D, 1, 1, 1)
        
        layer_defs = [
            {'type': 'linear', 'in': 16, 'out': 16},
            {'type': 'conv3d', 'dim': 16, 'kernel': 3},
            {'type': 'linear', 'in': 16, 'out': 10}
        ]
        model = ModernMLP(layer_defs)
        
        x = torch.randn(2, 16) # Batch size 2
        y = model(x)
        
        self.assertEqual(y.shape, (2, 10))
        
    def test_terminal_subprocess(self):
        # Verify we can run a simple command
        # "echo hello"
        cmd = "echo hello"
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()
        self.assertIn("hello", stdout.strip())

if __name__ == '__main__':
    unittest.main()

import unittest
import os
import json
import torch
import parser_utils
from trainer import TrainingEngine
from unittest.mock import MagicMock

class TestPhase7(unittest.TestCase):
    
    def test_presets_io(self):
        # Backup existing
        if os.path.exists(parser_utils.PRESETS_FILE):
            os.rename(parser_utils.PRESETS_FILE, "user_presets.json.bak")
            
        try:
            name = "TestArch"
            code = "[10, 20]"
            parser_utils.save_preset(name, code)
            
            presets = parser_utils.load_presets()
            self.assertIn(name, presets)
            self.assertEqual(presets[name], code)
            
        finally:
            if os.path.exists("user_presets.json.bak"):
                 os.replace("user_presets.json.bak", parser_utils.PRESETS_FILE)
            elif os.path.exists(parser_utils.PRESETS_FILE):
                 os.remove(parser_utils.PRESETS_FILE)

    def test_confusion_matrix(self):
        # Create dummy model
        model = torch.nn.Linear(10, 3) # 3 classes
        trainer = TrainingEngine(model, 'CrossEntropy')
        
        # Dummy data
        X = torch.randn(10, 10)
        y = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        
        # Patch output to be deterministic for confusion matrix?
        # No, just check it returns a string and valid matrix
        
        matrix_str, matrix = trainer.compute_confusion_matrix(X, y)
        
        self.assertIsInstance(matrix_str, str)
        self.assertIsInstance(matrix, list)
        self.assertEqual(len(matrix), 3) # 3x3
        self.assertEqual(len(matrix[0]), 3)

    def test_heatmap_logic(self):
        # We can't easily test the thread/GUI part, but we can test the hook logic
        model = torch.nn.Sequential(torch.nn.Linear(10, 10), torch.nn.Linear(10, 1))
        input_tensor = torch.randn(1, 10)
        
        activations = {}
        def get_activation(name):
            def hook(model, input, output):
                activations[name] = output.detach()
            return hook
            
        hooks = []
        for name, layer in model.named_modules():
             if isinstance(layer, torch.nn.Linear):
                 hooks.append(layer.register_forward_hook(get_activation(name)))
                 
        model(input_tensor)
        
        for h in hooks: h.remove()
        
        self.assertTrue(len(activations) > 0)
        print(f"Captured activations for: {list(activations.keys())}")

if __name__ == '__main__':
    unittest.main()

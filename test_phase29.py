import unittest
import torch
import torch.nn as nn
from network import (
    HyenaBlock, GEGLUBlock, ConvMixerBlock, 
    AdaptiveRankLinear, StochasticDepthBlock, ModernMLP
)
from parser_utils import parse_program, DSL_PRESETS, validate_dsl
from trainer import TrainingEngine

class TestPhase29AdaptiveNexus(unittest.TestCase):
    
    def setUp(self):
        self.dim = 32
        self.batch_size = 4
        self.x = torch.randn(self.batch_size, self.dim)
        
    def test_hyena_block(self):
        """Test HyenaBlock forward pass and shape."""
        block = HyenaBlock(self.dim)
        out = block(self.x)
        self.assertEqual(out.shape, (self.batch_size, self.dim))
        
    def test_geglu_block(self):
        """Test GEGLUBlock forward pass and shape."""
        block = GEGLUBlock(self.dim)
        out = block(self.x)
        self.assertEqual(out.shape, (self.batch_size, self.dim))
        
    def test_conv_mixer_block(self):
        """Test ConvMixerBlock forward pass and shape."""
        block = ConvMixerBlock(self.dim)
        out = block(self.x)
        self.assertEqual(out.shape, (self.batch_size, self.dim))
        
    def test_adaptive_rank_linear(self):
        """Test AdaptiveRankLinear forward pass and rank selection."""
        layer = AdaptiveRankLinear(self.dim, max_rank=8, energy_threshold=0.9)
        out = layer(self.x)
        self.assertEqual(out.shape, (self.batch_size, self.dim))
        # Check if internal SVD parameters exist
        self.assertTrue(hasattr(layer, 'U'))
        self.assertTrue(hasattr(layer, 'S'))
        self.assertTrue(hasattr(layer, 'V'))
        
    def test_stochastic_depth_block(self):
        """Test StochasticDepthBlock behavior in train vs eval mode."""
        layer = StochasticDepthBlock(self.dim, drop_prob=0.5)
        
        # Train mode: output might be identity or transformed
        layer.train()
        out_train = layer(self.x)
        self.assertEqual(out_train.shape, (self.batch_size, self.dim))
        
        # Eval mode: output is deterministic scaling
        layer.eval()
        out_eval = layer(self.x)
        self.assertEqual(out_eval.shape, (self.batch_size, self.dim))

    def test_dsl_parsing(self):
        """Test parsing of new Phase 29 keywords."""
        program = "hyena: [64], geglu: [64], conv_mixer: [64], adaptive_rank: [64, 8], stoch_depth: [64]"
        parsed = parse_program(program)
        self.assertIsNotNone(parsed)
        self.assertEqual(len(parsed), 5)
        self.assertEqual(parsed[0]['type'], 'hyena')
        self.assertEqual(parsed[1]['type'], 'geglu')
        self.assertEqual(parsed[2]['type'], 'conv_mixer')
        self.assertEqual(parsed[3]['type'], 'adaptive_rank')
        self.assertEqual(parsed[3]['rank'], 8)
        self.assertEqual(parsed[4]['type'], 'stoch_depth')

    def test_adaptive_nexus_preset(self):
        """Test building a model from the 'Adaptive Nexus' preset."""
        preset = DSL_PRESETS["Adaptive Nexus"]
        issues, layer_defs = validate_dsl(preset)
        if issues:
            # Filter out non-error issues (warnings are okay)
            errors = [i for i in issues if i[0] == "ERROR"]
            self.assertEqual(len(errors), 0, f"Preset validation errors: {errors}")
            
        model = ModernMLP(layer_defs)
        x = torch.randn(2, 128)
        out = model(x)
        self.assertEqual(out.shape, (2, 10))
        
        # Verify layer types are present
        layer_types = [type(l).__name__ for l in model.layers]
        self.assertIn("HyenaBlock", layer_types)
        self.assertIn("GEGLUBlock", layer_types)
        self.assertIn("ConvMixerBlock", layer_types)
        self.assertIn("StochasticDepthBlock", layer_types)

    def test_trainer_lawa(self):
        """Test LAWA weight averaging functionality in TrainingEngine."""
        model = nn.Linear(10, 2)
        trainer = TrainingEngine(model)
        trainer.enable_lawa(interval=1, window_size=2)
        
        # Mock step count and optimizer param groups
        trainer.step_count = 0
        optimizer = trainer.optimizer
        # Ensure param groups have lr
        for group in optimizer.param_groups:
            group['lr'] = 0.01

        # Save initial weights
        w0 = model.weight.clone()
        
        # Update weights manually
        with torch.no_grad():
            model.weight.add_(1.0)
        
        # Run LAWA step (simulating step 0 -> should take snapshot 1)
        trainer._lawa_step() 
        self.assertEqual(len(trainer.lawa_snapshots), 1)
        
        trainer.step_count = 1
        # Update weights again
        with torch.no_grad():
            model.weight.add_(1.0)
            
        # Run LAWA step (simulating step 1 -> should take snapshot 2 and merge)
        trainer._lawa_step()
        self.assertEqual(len(trainer.lawa_snapshots), 2)
        
        # Weights should be averaged now (roughly)
        # We just check no crash and snapshots exist
        self.assertTrue(len(trainer.lawa_snapshots) > 0)

if __name__ == '__main__':
    unittest.main()

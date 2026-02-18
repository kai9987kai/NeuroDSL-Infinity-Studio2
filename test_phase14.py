import torch
import unittest
import os
from sensor_nexus import SensorNexus
from omni_chat import OmniChat
from spatial_viz import SpatialNavigator
from auto_researcher import AutoResearcher
from trainer import TrainingEngine
from network import ModernMLP
from parser_utils import parse_program

class TestPhase14Omniscience(unittest.TestCase):
    def test_sensor_telemetry(self):
        tele = SensorNexus.get_system_telemetry()
        self.assertIn("cpu_percent", tele)
        self.assertIn("ram_percent", tele)
        tensor = SensorNexus.telemetry_to_tensor()
        self.assertEqual(tensor.shape, (4,))

    def test_omni_chat(self):
        chat = OmniChat()
        chat.post_message("User", "Hello")
        resp = chat.get_response("Hello")
        self.assertIn("Architect", resp)
        self.assertEqual(len(chat.history), 1) # post_message only adds one, get_response doesn't auto-add in this implementation

    def test_spatial_navigator(self):
        nav = SpatialNavigator(width=100, height=100)
        nav.add_point(10, 20, 30)
        projected = nav.project_points()
        self.assertEqual(len(projected), 1)
        self.assertEqual(projected[0][2], "#39FF14")

    def test_auto_researcher(self):
        res = AutoResearcher()
        tips = res.analyze_dsl("attn: [128], linear: [128, 64]")
        self.assertTrue(any("mamba" in t.lower() for t in tips))

    def test_gradient_accumulation(self):
        dsl = "[8, 8]"
        defs = parse_program(dsl)
        model = ModernMLP(defs)
        trainer = TrainingEngine(model)
        trainer.accumulation_steps = 4
        
        X = torch.randn(1, 8)
        y = torch.randn(1, 8)
        
        # Step 0: Should not update weights
        w_old = next(model.parameters()).clone()
        trainer.train_step(X, y)
        w_new = next(model.parameters())
        torch.testing.assert_close(w_old, w_new)
        
        # Step 4: Should update weights
        trainer.train_step(X, y) # Step 1 (count=1)
        trainer.train_step(X, y) # Step 2 (count=2)
        trainer.train_step(X, y) # Step 3 (count=3)
        # Note: step_count starts at 0. (0+1)%4 != 0. (1+1)%4 != 0. (2+1)%4 != 0. (3+1)%4 == 0.
        
        w_final = next(model.parameters())
        self.assertFalse(torch.equal(w_old, w_final))

if __name__ == "__main__":
    unittest.main()

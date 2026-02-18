import torch
import unittest
import os
import shutil
from vault_service import VaultService
from polyglot_bridge import PolyglotBridge
from neural_env import EnvironmentSimulator, DiamondBlock
from network import ModernMLP
from parser_utils import parse_program

class TestPhase15Universal(unittest.TestCase):
    
    def setUp(self):
        if os.path.exists("test_vault.db"):
            os.remove("test_vault.db")
        if os.path.exists("test_labs"):
            shutil.rmtree("test_labs")
        os.makedirs("test_labs")

    def test_vault_accounts(self):
        vault = VaultService(db_path="test_vault.db")
        self.assertTrue(vault.create_account("admin", "secret123"))
        self.assertFalse(vault.create_account("admin", "different")) # Duplicate
        self.assertTrue(vault.login("admin", "secret123"))
        self.assertFalse(vault.login("admin", "wrong"))

    def test_polyglot_lab_gen(self):
        bridge = PolyglotBridge("test_labs")
        path = bridge.generate_web_lab("TestProject", {"key": "val"}, "<div>Content</div>", "console.log('test');")
        self.assertTrue(os.path.exists(path))
        with open(path, "r") as f:
            content = f.read()
            self.assertIn("TestProject Lab Shell", content)
            self.assertIn("console.log('test');", content)

    def test_diamond_block(self):
        block = DiamondBlock(16)
        x = torch.randn(2, 16)
        out = block(x)
        self.assertEqual(out.shape, (2, 16))
        # Ensure it's not just an identity (params should change it)
        self.assertFalse(torch.allclose(x, out))

    def test_environment_simulation(self):
        env = EnvironmentSimulator(grid_size=5)
        # Dummy model with 4 outputs for 4 actions
        model = torch.nn.Sequential(torch.nn.Linear(4, 4))
        data = env.run_automated_session(model, n_episodes=1)
        self.assertGreater(len(data), 0)
        state_t, action_t = data[0]
        self.assertEqual(state_t.shape, (1, 4))
        self.assertEqual(action_t.shape, (1,))

if __name__ == "__main__":
    unittest.main()

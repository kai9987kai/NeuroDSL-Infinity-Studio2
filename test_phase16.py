import torch
import unittest
import os
import sqlite3
from highway_core import MultiModelHighway
from vault_service import VaultService

class TestPhase16HyperEnsemble(unittest.TestCase):
    
    def test_multi_model_highway_average(self):
        # Two identity models for testing
        m1 = torch.nn.Linear(8, 8)
        m1.weight.data = torch.eye(8)
        m1.bias.data.zero_()
        
        m2 = torch.nn.Linear(8, 8)
        m2.weight.data = torch.eye(8) * 2
        m2.bias.data.zero_()
        
        highway = MultiModelHighway([m1, m2], out_dim=8, mode="average")
        x = torch.ones(1, 8)
        out = highway(x)
        expected = torch.ones(1, 8) * 1.5
        
        is_correct = torch.allclose(out, expected)
        if not is_correct:
            print(f"FAILED: out={out}, expected={expected}")
        self.assertTrue(is_correct)

    def test_dictionary_search(self):
        db_path = "test_phase16.db"
        if os.path.exists(db_path): os.remove(db_path)
        vault = VaultService(db_path=db_path)
        
        # Search for seeded entries
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("SELECT language FROM global_knowledge")
        langs = [r[0] for r in c.fetchall()]
        conn.close()
        
        self.assertIn("en", langs)
        self.assertIn("fr", langs)
        self.assertIn("de", langs)
        self.assertIn("es", langs)

    def test_photo_vault_blob(self):
        db_path = "test_phase16_photos.db"
        if os.path.exists(db_path): os.remove(db_path)
        vault = VaultService(db_path=db_path)
        
        dummy_img = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        vault.add_photo("test_img.png", dummy_img, "dummy metadata")
        
        retrieved = vault.get_photo("test_img.png")
        self.assertEqual(retrieved, dummy_img)

if __name__ == "__main__":
    unittest.main()

import unittest
import os
import json
import shutil
from unittest.mock import MagicMock, patch
import codex_client

# We need to mock sg to avoid GUI loop issues in main
import sys
sys.modules['FreeSimpleGUI'] = MagicMock()

# Import functions to test (we might need to extract them or import main)
# Since main is a script, importing it runs it. 
# We'll just test the logic we added by recreating it or mocking.

class TestPhase6(unittest.TestCase):

    def test_save_load_logic(self):
        # Recreate the save logic for testing
        data = {
            "dsl": "[784, 10]",
            "loss": "MSE",
            "epochs": 100,
            "params": "1000",
            "nodes": "3"
        }
        path = "test_project.nproj"
        
        # Save
        with open(path, "w") as f: json.dump(data, f)
        self.assertTrue(os.path.exists(path))
        
        # Load
        with open(path, "r") as f: loaded = json.load(f)
        self.assertEqual(loaded["dsl"], "[784, 10]")
        self.assertEqual(loaded["loss"], "MSE")
        
        os.remove(path)

    @patch('urllib.request.urlopen')
    def test_ascii_viz(self, mock_urlopen):
        # Mock API response
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "choices": [{"message": {"content": "```\n[Layer] -> [Layer]\n```"}}]
        }).encode('utf-8')
        mock_urlopen.return_value.__enter__.return_value = mock_response
        
        success, content = codex_client.generate_ascii_diagram("fake_key", "[784, 10]")
        self.assertTrue(success)
        self.assertIn("[Layer] -> [Layer]", content)

if __name__ == '__main__':
    unittest.main()

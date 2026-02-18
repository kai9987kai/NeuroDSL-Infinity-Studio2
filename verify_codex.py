
import unittest
from unittest.mock import patch, MagicMock
import json
import codex_client
import urllib.error
import os

class TestCodexClient(unittest.TestCase):
    
    @patch('urllib.request.urlopen')
    def test_generate_dsl_success(self, mock_urlopen):
        # Mock successful response
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "choices": [{
                "message": {
                    "content": "[128, 64], dropout: [0.5]"
                }
            }]
        }).encode('utf-8')
        mock_urlopen.return_value.__enter__.return_value = mock_response
        
        success, content = codex_client.generate_dsl("fake-key", "test prompt")
        self.assertTrue(success)
        self.assertEqual(content, "[128, 64], dropout: [0.5]")

    @patch('urllib.request.urlopen')
    def test_generate_dsl_strips_markdown(self, mock_urlopen):
        # Mock response with markdown code blocks
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "choices": [{
                "message": {
                    "content": "```text\n[128, 64], dropout: [0.5]\n```"
                }
            }]
        }).encode('utf-8')
        mock_urlopen.return_value.__enter__.return_value = mock_response
        
        success, content = codex_client.generate_dsl("fake-key", "test prompt")
        self.assertTrue(success)
        self.assertEqual(content, "[128, 64], dropout: [0.5]")

    @patch('urllib.request.urlopen')
    def test_generate_dsl_error(self, mock_urlopen):
        # Mock HTTP error
        mock_urlopen.side_effect = urllib.error.HTTPError(
            "url", 401, "Unauthorized", {}, None
        )
        
        success, content = codex_client.generate_dsl("fake-key", "test prompt")
        self.assertFalse(success)
        self.assertIn("API Error 401", content)

    @patch('urllib.request.urlopen')
    def test_explain_dsl(self, mock_urlopen):
        # Mock explanation response
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "choices": [{
                "message": {
                    "content": "This is a simple linear classifier."
                }
            }]
        }).encode('utf-8')
        mock_urlopen.return_value.__enter__.return_value = mock_response
        
        success, content = codex_client.explain_dsl("fake-key", "[128, 10]")
        self.assertTrue(success)
        self.assertEqual(content, "This is a simple linear classifier.")

    @patch('urllib.request.urlopen')
    def test_fix_dsl(self, mock_urlopen):
        # Mock fix response
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "choices": [{
                "message": {
                    "content": "[128, 64], [64, 10]"
                }
            }]
        }).encode('utf-8')
        mock_urlopen.return_value.__enter__.return_value = mock_response
        
        success, content = codex_client.fix_dsl("fake-key", "[128, 64], [32, 10]", "dim mismatch")
        self.assertTrue(success)
        self.assertEqual(content, "[128, 64], [64, 10]")

    @patch('urllib.request.urlopen')
    def test_optimize_dsl(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "choices": [{ "message": { "content": "[128, 64], residual: [64], [64, 10]" } }]
        }).encode('utf-8')
        mock_urlopen.return_value.__enter__.return_value = mock_response
        
        success, content = codex_client.optimize_dsl("fake-key", "[128, 64], [64, 10]")
        self.assertTrue(success)
        self.assertEqual(content, "[128, 64], residual: [64], [64, 10]")

    @patch('urllib.request.urlopen')
    def test_generate_pytorch_code(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "choices": [{ "message": { "content": "class Model(nn.Module): pass" } }]
        }).encode('utf-8')
        mock_urlopen.return_value.__enter__.return_value = mock_response
        
        success, content = codex_client.generate_pytorch_code("fake-key", "[128, 10]")
        self.assertTrue(success)
        self.assertEqual(content, "class Model(nn.Module): pass")

    @patch('urllib.request.urlopen')
    def test_suggest_hyperparams(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "choices": [{ "message": { "content": '{"epochs": 100, "lr": 0.001, "clip": 1.0}' } }]
        }).encode('utf-8')
        mock_urlopen.return_value.__enter__.return_value = mock_response
        
        success, content = codex_client.suggest_hyperparams("fake-key", "[128, 10]")
        self.assertTrue(success)
        self.assertIn('"epochs": 100', content)

    @patch('urllib.request.urlopen')
    def test_generate_synthetic_data(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "choices": [{ "message": { "content": "import numpy as np" } }]
        }).encode('utf-8')
        mock_urlopen.return_value.__enter__.return_value = mock_response
        
        success, content = codex_client.generate_synthetic_data("fake-key", "spiral")
        self.assertTrue(success)
        self.assertEqual(content, "import numpy as np")

    @patch('urllib.request.urlopen')
    def test_generate_unit_tests(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "choices": [{ "message": { "content": "import unittest" } }]
        }).encode('utf-8')
        mock_urlopen.return_value.__enter__.return_value = mock_response
        
        success, content = codex_client.generate_unit_tests("fake-key", "[128, 10]")
        self.assertTrue(success)
        self.assertEqual(content, "import unittest")

    @patch('urllib.request.urlopen')
    def test_estimate_latency(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "choices": [{ "message": { "content": "Estimated FLOPs: 1.2G" } }]
        }).encode('utf-8')
        mock_urlopen.return_value.__enter__.return_value = mock_response
        
        success, content = codex_client.estimate_latency("fake-key", "[128, 10]", "A100")
        self.assertTrue(success)
        self.assertEqual(content, "Estimated FLOPs: 1.2G")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}, clear=False)
    @patch('urllib.request.urlopen')
    def test_env_key_fallback(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "choices": [{
                "message": {"content": "[64, 10]"}
            }]
        }).encode('utf-8')
        mock_urlopen.return_value.__enter__.return_value = mock_response

        success, content = codex_client.generate_dsl("", "simple network")
        self.assertTrue(success)
        self.assertEqual(content, "[64, 10]")

    @patch('urllib.request.urlopen')
    def test_test_connection(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "choices": [{
                "message": {"content": "OK"}
            }]
        }).encode('utf-8')
        mock_urlopen.return_value.__enter__.return_value = mock_response

        success, content = codex_client.test_connection("fake-key")
        self.assertTrue(success)
        self.assertEqual(content, "OK")

    @patch('urllib.request.urlopen')
    def test_generate_dsl_candidates(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "choices": [{
                "message": {"content": '{"candidates":[{"dsl":"[32,64], [64,10]","notes":"baseline"}]}'}
            }]
        }).encode('utf-8')
        mock_urlopen.return_value.__enter__.return_value = mock_response

        success, content = codex_client.generate_dsl_candidates(
            "fake-key",
            "build a compact classifier",
            input_dim=32,
            output_dim=10,
            count=3,
        )
        self.assertTrue(success)
        payload = json.loads(content)
        self.assertIn("candidates", payload)
        self.assertEqual(payload["candidates"][0]["dsl"], "[32,64], [64,10]")

if __name__ == '__main__':
    unittest.main()

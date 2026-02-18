import unittest
import json
from visual_tool import NeuralVisualizer
from agentic_service import app
from fastapi.testclient import TestClient

class TestPhase18AgenticInterop(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.viz = NeuralVisualizer()

    def test_visual_tool_logic(self):
        # Test the Python bridge logic
        res = self.viz.process_state({"sampleRange": 100, "sampleSelect": "mode1"})
        self.assertEqual(res["chart_data"][0], 24.0) # 12 * (100/50)
        self.assertTrue(res["ai_insights"]["mean_activation"] > 0)

    def test_api_visualize_endpoint(self):
        # Test the FastAPI endpoint
        response = self.client.post("/visualize", json={"params": {"sampleRange": 50}})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("ai_insights", data)
        self.assertIn("chart_data", data)

    def test_api_manifesto_updated(self):
        response = self.client.get("/manifesto")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("visualizer", data["tools"])
        self.assertEqual(data["api_v"], "8.0 (Phase 18)")

if __name__ == "__main__":
    unittest.main()

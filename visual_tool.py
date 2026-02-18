import json
import numpy as np
from typing import Dict, Any

class NeuralVisualizer:
    """
    Python implementation of the lab-visualizer.html logic.
    Exposes the visualizer as a callable data tool for AI models.
    """
    
    def __init__(self):
        self.default_state = {
            "sampleInput": "Neural Probe",
            "sampleRange": 50,
            "sampleSelect": "mode1",
            "sampleToggle": True
        }

    def process_state(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculates the data that would be rendered by Chart.js in the HTML lab.
        """
        state = {**self.default_state, **params}
        
        # Replicating the logic from lab-visualizer.html scripts:
        # data: [12, 19, 3, 5, 2, 3].map(v => v * (this.state.sampleRange / 50))
        base_data = [12, 19, 3, 5, 2, 3]
        scale = state["sampleRange"] / 50.0
        
        # Mode-based modifiers
        if state["sampleSelect"] == "mode2":
            base_data = [v * 1.5 for v in base_data]
        elif state["sampleSelect"] == "mode3":
            base_data = [v + 5 for v in base_data]
            
        scaled_data = [round(v * scale, 2) for v in base_data]
        
        # Meta-analysis for AI feedback
        analysis = {
            "mean_activation": float(round(np.mean(scaled_data), 2)),
            "peak_signal": float(max(scaled_data)),
            "signal_to_noise": float(round(np.mean(scaled_data) / (np.std(scaled_data) + 1e-6), 2)),
            "is_stable": bool(state["sampleToggle"] and np.std(scaled_data) < 10)
        }
        
        return {
            "state": state,
            "chart_data": scaled_data,
            "labels": ['Red', 'Blue', 'Yellow', 'Green', 'Purple', 'Orange'],
            "ai_insights": analysis,
            "visual_url": f"file:///path/to/snaphost?state={json.dumps(state)}" # Placeholder for UI integration
        }

if __name__ == "__main__":
    viz = NeuralVisualizer()
    print(json.dumps(viz.process_state({"sampleRange": 80}), indent=2))

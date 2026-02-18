from typing import List, Dict
import time

class OmniChat:
    """Conversational hub for NeuroDSL Infinity Studio."""
    
    def __init__(self):
        self.history: List[Dict[str, str]] = []
        self.agent_name = "NeuroAgent"

    def post_message(self, role: str, text: str):
        """Adds a message to the history."""
        self.history.append({
            "role": role,
            "text": text,
            "timestamp": time.strftime("%H:%M:%S")
        })

    def get_response(self, user_text: str) -> str:
        """
        Generates a response from the internal agent.
        In a full implementation, this would call Codex or a local LLM.
        """
        user_text = user_text.lower()
        
        if "hello" in user_text or "hi" in user_text:
            return "Greetings, Architect. Phase 14 is online. How can I assist with your neural research?"
        elif "status" in user_text:
            return "Sensors are active. Mixed precision training is ready for deployment."
        elif "sensors" in user_text:
            return "I am currently monitoring system telemetry and vision streams. Type 'show sensors' in the dashboard to see live data."
        elif "train" in user_text:
            return "I recommend enabling Gradient Accumulation for larger batch simulations."
        else:
            return f"Understood. I've logged your request: '{user_text}'. I am refining the neural manifold as we speak."

    def clear_history(self):
        self.history = []

    def get_formatted_history(self) -> str:
        lines = []
        for msg in self.history:
            lines.append(f"[{msg['timestamp']}] {msg['role'].upper()}: {msg['text']}")
        return "\n".join(lines)

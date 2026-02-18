from typing import List, Dict
import random

class AutoResearcher:
    """Simulates a local architectural research agent."""
    
    def __init__(self):
        self.knowledge_base = [
            "Mamba layers are more efficient for sequence lengths > 2000 compared to standard Attention.",
            "Adding RMSNorm before MoE routing can stabilize training in deep architectures.",
            "Liquid Neural Networks excel at tasks with irregular time-series intervals.",
            "Hyper-Transformers are best used when the input distribution is highly non-stationary.",
            "Mixed Precision (FP16) can provide up to 2x speedup on Tensor Core GPUs."
        ]

    def analyze_dsl(self, dsl_code: str) -> List[str]:
        """Analyzes a DSL string and suggests improvements based on research patterns."""
        suggestions = []
        dsl_lower = dsl_code.lower()
        
        if "attn" in dsl_lower and "mamba" not in dsl_lower:
            suggestions.append("RESEARCH TIP: Consider replacing 'attn' with 'mamba' for linear-time complexity if sequence length is high.")
        
        if "moe" in dsl_lower and "rmsnorm" not in dsl_lower:
            suggestions.append("STABILITY TIP: Integrate 'RMSNorm' before your 'MoE' block to prevent gradient spikes.")
            
        if len(dsl_code.split(",")) > 10:
             suggestions.append("EFFICIENCY TIP: Your model is deep. Ensure 'residual' connections are used frequently to prevent vanishing gradients.")

        if not suggestions:
            suggestions.append("YOUR ARCHITECTURE LOOKS OPTIMAL: Proceed with training.")
            
        return suggestions

    def get_random_fact(self) -> str:
        return random.choice(self.knowledge_base)

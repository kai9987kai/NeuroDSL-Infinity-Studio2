import torch
import torch.nn as nn
import numpy as np

class TurbulenceLayer(nn.Module):
    """
    Injects structured stochastic turbulence into the activation stream.
    Uses coherent noise (simulated) to create persistent perturbations.
    """
    def __init__(self, dim, intensity=0.05, frequency=1.0):
        super().__init__()
        self.dim = dim
        self.intensity = intensity
        self.frequency = frequency
        
        # Periodic "Turbulence Seeds"
        self.register_buffer('seeds', torch.randn(dim))
        
    def forward(self, x):
        if not self.training:
            return x
            
        # Simulate coherent turbulence using phase shifts
        # t is essentially a "time" or "batch" index proxy
        batch_size = x.size(0)
        t = torch.arange(batch_size, device=x.device).float().unsqueeze(1)
        
        # Turbulence tensor [B, D]
        turb = torch.sin(self.frequency * t + self.seeds) * self.intensity
        
        # Apply turbulence as multiplicative or additive noise
        return x * (1.0 + turb)

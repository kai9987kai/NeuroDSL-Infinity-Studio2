import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantumLinear(nn.Module):
    """
    Simulates a quantum superposition layer using complex weights.
    Phase represents rotation in latent space, magnitude represents activation.
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Weights are stored as complex: alpha + i*beta
        self.weight_real = nn.Parameter(torch.randn(out_features, in_features) * (2 / in_features)**0.5)
        self.weight_imag = nn.Parameter(torch.randn(out_features, in_features) * (2 / in_features)**0.5)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        # x is typically real, but we project it into the complex plane
        w = torch.complex(self.weight_real, self.weight_imag)
        x_c = torch.complex(x, torch.zeros_like(x))
        
        # Complex matrix multiplication
        out_c = torch.matmul(x_c, w.t())
        
        # "Phase Collapse": Extract magnitude as the real-world activation
        # but apply a phase-dependent rotation
        magnitude = torch.abs(out_c)
        phase = torch.angle(out_c)
        
        # Interference effect: high alignment (low phase difference) boosts signal
        interference = torch.cos(phase)
        return (magnitude * interference) + self.bias

class EntanglementGate(nn.Module):
    """
    Binds two feature streams together, simulating quantum entanglement.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.gate = nn.Parameter(torch.ones(dim))

    def forward(self, xa, xb):
        # Cross-modulation (entanglement)
        shared = torch.sigmoid(xa * xb * self.gate)
        return xa * shared, xb * shared

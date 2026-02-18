import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HolographicLinear(nn.Module):
    """
    Implements a Holographic layer using complex-valued weights.
    Features are interacted using circular convolution (via FFT), 
    mimicking holographic associative memory.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # Complex weights: (dim,)
        self.weight_real = nn.Parameter(torch.randn(dim) * 0.02)
        self.weight_imag = nn.Parameter(torch.randn(dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        # x: [batch, dim]
        # Project x into complex plane
        x_c = torch.complex(x, torch.zeros_like(x))
        w_c = torch.complex(self.weight_real, self.weight_imag)
        
        # Holographic Binding via Circular Convolution
        # F(x * w) = F(x) * F(w)
        x_fft = torch.fft.fft(x_c)
        w_fft = torch.fft.fft(w_c)
        
        out_fft = x_fft * w_fft
        out_c = torch.fft.ifft(out_fft)
        
        # Decode: Take magnitude and apply phase interference
        mag = torch.abs(out_c)
        phase = torch.angle(out_c)
        
        # Interference pattern stabilization
        interference = torch.cos(phase)
        return (mag * interference) + self.bias

class ModReLU(nn.Module):
    """
    ReLU for complex-valued activations.
    ModReLU(z) = ReLU(|z| + b) * (z / |z|)
    """
    def __init__(self, dim):
        super().__init__()
        self.b = nn.Parameter(torch.full((dim,), -0.01))

    def forward(self, z):
        # z: complex tensor
        mag = torch.abs(z)
        # Avoid division by zero
        norm = z / (mag + 1e-6)
        return F.relu(mag + self.b) * norm

class HolographicDual(nn.Module):
    """
    Two-stream holographic interaction for associative binding.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.u = nn.Parameter(torch.randn(dim) * 0.02)
        self.v = nn.Parameter(torch.randn(dim) * 0.02)

    def forward(self, x):
        # Circular correlation for memory retrieval simulation
        x_fft = torch.fft.fft(x)
        u_fft = torch.fft.fft(torch.complex(self.u, torch.zeros_like(self.u)))
        v_fft = torch.fft.fft(torch.complex(self.v, torch.zeros_like(self.v)))
        
        # Interaction: (X * U) + (X * V)
        res = torch.fft.ifft(x_fft * (u_fft + v_fft))
        return torch.real(res)

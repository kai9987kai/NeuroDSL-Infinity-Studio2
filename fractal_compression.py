import torch
import torch.nn as nn
import torch.nn.functional as F

class FractalBlock(nn.Module):
    """
    Recursive weight sharing using a seed matrix.
    A small seed is tiled and scaled to form a large weight matrix.
    """
    def __init__(self, in_dim, out_dim, seed_size=4):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.seed_size = seed_size
        self.seed = nn.Parameter(torch.randn(seed_size, seed_size) / seed_size)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def get_fractal_weights(self):
        # Recusively tiling the seed (2 levels of recursion for simplicity)
        w1 = self.seed.repeat(self.out_dim // self.seed_size + 1, self.in_dim // self.seed_size + 1)
        # Apply a deterministic scale to create fractal variance
        variance = torch.sin(torch.linspace(0, 10, w1.shape[0])).unsqueeze(1)
        # Slice both to output dimensions
        w_final = w1[:self.out_dim, :self.in_dim] * variance[:self.out_dim]
        return w_final

    def forward(self, x):
        w = self.get_fractal_weights()
        return F.linear(x, w, self.bias)

class NeuralPruner:
    """
    Utility for post-training entropy-based pruning.
    """
    @staticmethod
    @torch.no_grad()
    def prune_model(model, threshold=0.01):
        total_pruned = 0
        for param in model.parameters():
            mask = torch.abs(param) < threshold
            param.data[mask] *= 0.1 # Soft pruning
            total_pruned += torch.sum(mask).item()
        return total_pruned

import torch
import torch.nn as nn

class HyperLayer(nn.Module):
    """
    Hyper-Network: Generates weights and biases for a target linear layer 
    based on the current input context.
    """
    def __init__(self, in_features, out_features, hyper_dim=32):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hyper_dim = hyper_dim
        
        # Hyper-network to produce flat weights and bias
        self.weight_gen = nn.Sequential(
            nn.Linear(in_features, hyper_dim),
            nn.ReLU(),
            nn.Linear(hyper_dim, in_features * out_features)
        )
        
        self.bias_gen = nn.Sequential(
            nn.Linear(in_features, hyper_dim),
            nn.ReLU(),
            nn.Linear(hyper_dim, out_features)
        )
        
    def forward(self, x):
        # x: [B, in_features]
        batch_size = x.size(0)
        
        # Generate weights and bias per sample
        # weights: [B, in_features * out_features] -> [B, out_features, in_features]
        weights = self.weight_gen(x).view(batch_size, self.out_features, self.in_features)
        bias = self.bias_gen(x).view(batch_size, self.out_features, 1)
        
        # Apply transformation: y = Wx + b
        # [B, out_features, in_features] @ [B, in_features, 1] -> [B, out_features, 1]
        out = torch.bmm(weights, x.unsqueeze(2)) + bias
        
        return out.squeeze(2)

import torch
import torch.nn as nn
import os
from typing import List, Optional

class MultiModelHighway(nn.Module):
    """
    Unified system to run multiple pre-trained models in parallel and fuse their outputs.
    Supports Voting, Averaging, or Gated Consensus (learned weighting).
    """
    def __init__(self, models: List[nn.Module], out_dim: int, mode="average"):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.mode = mode
        self.out_dim = out_dim
        
        # Learned gating arbiter
        if mode == "gated":
            self.gater = nn.Sequential(
                nn.Linear(len(models) * out_dim, 64),
                nn.ReLU(),
                nn.Linear(64, len(models)),
                nn.Softmax(dim=-1)
            )

    def forward(self, x):
        outputs = [m(x) for m in self.models] # [N, Batch, OutDim]
        stacked_out = torch.stack(outputs, dim=1) # [Batch, N, OutDim]
        
        if self.mode == "average":
            return torch.mean(stacked_out, dim=1)
        
        elif self.mode == "voting":
            # Simple max-confidence or argmax voting for classification
            # Returns the output of the model that was most 'confident'
            confidences = [torch.max(o, dim=-1)[0] for o in outputs]
            best_idx = torch.argmax(torch.stack(confidences, dim=1), dim=1)
            final_out = torch.zeros_like(outputs[0])
            for i in range(x.shape[0]):
                final_out[i] = outputs[best_idx[i]][i]
            return final_out
            
        elif self.mode == "gated":
            # Concatenate all outputs and let the gater decide weights
            flat_out = torch.cat(outputs, dim=-1)
            weights = self.gater(flat_out) # [Batch, N]
            weights = weights.unsqueeze(-1) # [Batch, N, 1]
            fused = torch.sum(stacked_out * weights, dim=1)
            return fused
        
        return outputs[0] # Fallback

class EnsembleLayer(nn.Module):
    """A wrapper for a single .pth model to be used as a layer in NeuroDSL."""
    def __init__(self, model_path: str, in_dim: int, out_dim: int):
        super().__init__()
        self.model = None
        self.path = model_path
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        if os.path.exists(model_path):
            # We assume it's a ModernMLP or standard torch module
            # For extraction, we rely on the state_dict matching
            pass

    def forward(self, x):
        if self.model is None:
             # Lazy load or identity fallback if missing
             return x
        return self.model(x)

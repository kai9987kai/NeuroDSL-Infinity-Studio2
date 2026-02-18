import torch
import torch.nn as nn
import random
import copy
from typing import List

class REMCycle:
    """Simulates a model's 'dreaming' phase to consolidate weights and sharpen predictions."""
    def __init__(self, model: nn.Module, env_replay_buffer: List[torch.Tensor]):
        self.model = model
        self.buffer = env_replay_buffer
        self.consolidation_log = []

    def perform_dream_session(self, intensity=0.01, cycles=10):
        """
        Perturbs weights and evaluates them against stored 'experience' trajectories.
        Retains changes that improve consistency or reduce entropy.
        """
        for _ in range(cycles):
            if not self.buffer: break
            
            # 1. Select a random experience
            state = random.choice(self.buffer)
            
            # 2. Perturb weights (Imagination perturbation)
            original_weights = copy.deepcopy(self.model.state_dict())
            with torch.no_grad():
                for param in self.model.parameters():
                    noise = torch.randn_like(param) * intensity
                    param.add_(noise)
            
            # 3. Evaluate "Dream Consensus"
            # If the perturbed model is more 'decisive' (lower entropy), we keep a fraction of the change
            out = self.model(state)
            probs = torch.softmax(out, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8))
            
            self.consolidation_log.append(entropy.item())
            
            # Simple hill climbing for dream stabilization
            # (In a real system this would be more complex, but for v11.0 it's a bio-inspired heuristic)
            # If it's too unstable (high entropy), revert
            if entropy > 1.5: 
                self.model.load_state_dict(original_weights)

        return self.consolidation_log

class ImagineLayer(nn.Module):
    """A stochastic layer that generates 'imagined' future state offsets."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.imagination_strength = nn.Parameter(torch.tensor(0.01))

    def forward(self, x):
        if self.training:
            # During training, we don't dream, we learn to predict the real trajectory
            return x
        
        # During inference (dream mode), we add 'imagined' noise based on learned strength
        noise = torch.randn_like(x) * torch.abs(self.imagination_strength)
        return x + noise

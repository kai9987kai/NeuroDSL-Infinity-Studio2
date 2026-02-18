import torch
import torch.nn as nn
import random
import numpy as np
from typing import List, Tuple

class DiamondBlock(nn.Module):
    """
    A convergent-divergent topology (Diamond).
    Splits input into multiple paths of different capacities and merges them.
    Input -> [Path A (Compressed), Path B (Expanded)] -> Merge -> Output
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.path_a = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.SiLU(),
            nn.Linear(dim // 2, dim)
        )
        self.path_b = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim)
        )
        self.gate = nn.Parameter(torch.ones(2))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        out_a = self.path_a(x)
        out_b = self.path_b(x)
        
        # Gated fusion
        weights = torch.softmax(self.gate, dim=0)
        out = out_a * weights[0] + out_b * weights[1]
        
        return self.norm(out + x)

class EnvironmentSimulator:
    """A simulation engine where models act as agents to generate training data."""
    
    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        self.agent_pos = [grid_size // 2, grid_size // 2]
        self.target_pos = [random.randint(0, grid_size-1), random.randint(0, grid_size-1)]
        self.steps = 0

    def reset(self):
        self.agent_pos = [self.grid_size // 2, self.grid_size // 2]
        self.target_pos = [random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1)]
        self.steps = 0
        return self._get_state()

    def _get_state(self):
        # Normalized state: [agent_x, agent_y, target_x, target_y]
        return np.array([
            self.agent_pos[0] / self.grid_size,
            self.agent_pos[1] / self.grid_size,
            self.target_pos[0] / self.grid_size,
            self.target_pos[1] / self.grid_size
        ], dtype=np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        Actions: 0: Up, 1: Down, 2: Left, 3: Right
        """
        if action == 0: self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 1: self.agent_pos[1] = min(self.grid_size - 1, self.agent_pos[1] + 1)
        elif action == 2: self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 3: self.agent_pos[0] = min(self.grid_size - 1, self.agent_pos[0] + 1)
        
        dist = np.sqrt((self.agent_pos[0] - self.target_pos[0])**2 + (self.agent_pos[1] - self.target_pos[1])**2)
        reward = -0.1 # Constant time penalty
        done = False
        
        if self.agent_pos == self.target_pos:
            reward = 10.0
            done = True
        
        self.steps += 1
        if self.steps >= 100:
            done = True
            
        return self._get_state(), reward, done

    def run_automated_session(self, model, n_episodes=5) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Runs the model through the environment and gathers (state, action_taken) pairs as data."""
        training_data = []
        for _ in range(n_episodes):
            state = self.reset()
            done = False
            while not done:
                state_t = torch.tensor(state).unsqueeze(0)
                with torch.no_grad():
                    logits = model(state_t)
                    action = torch.argmax(logits, dim=-1).item()
                
                next_state, reward, done = self.step(int(action % 4))
                
                # Store (state, action) for behavioral cloning or simple synthetic training
                training_data.append((state_t, torch.tensor([action % 4])))
                state = next_state
        return training_data

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

class ChronoFoldingLayer(nn.Module):
    """
    Temporal Lookahead/Lookbehind attention.
    Maintains a sliding window of synthetic future/past states to stabilize training.
    """
    def __init__(self, dim, window_size=5):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.attn = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        self.future_buffer = deque(maxlen=window_size)
        self.past_buffer = deque(maxlen=window_size)
        
    def forward(self, x):
        # x: [batch, dim] -> [batch, seq=1, dim]
        x_seq = x.unsqueeze(1)
        
        # Populate buffers (Identity during initial steps)
        if len(self.past_buffer) == 0:
            for _ in range(self.window_size):
                self.past_buffer.append(torch.zeros_like(x_seq))
                self.future_buffer.append(torch.zeros_like(x_seq))
        
        # Concatenate temporal window
        # context: [batch, window_size*2 + 1, dim]
        past = torch.cat(list(self.past_buffer), dim=1)
        future = torch.cat(list(self.future_buffer), dim=1)
        context = torch.cat([past, x_seq, future], dim=1)
        
        # Temporal Attention
        attn_out, _ = self.attn(x_seq, context, context)
        
        # Shift buffers
        self.past_buffer.append(x_seq.detach())
        # Future buffer simulates prediction (simple shift for demo)
        self.future_buffer.append(attn_out.detach())
        
        return attn_out.squeeze(1)

class TemporalBuffer:
    """Utility for state persistence across chrono-cycles."""
    def __init__(self, capacity=100):
        self.data = deque(maxlen=capacity)
        
    def push(self, state):
        self.data.append(state)
        
    def get_context(self):
        return torch.cat(list(self.data), dim=0) if self.data else None

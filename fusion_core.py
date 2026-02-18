import torch
import torch.nn as nn
import torch.nn.functional as F

class XFusionLayer(nn.Module):
    """
    Implements Cross-Modal Fusion using multi-head attention.
    Allows the model to attend to an external 'context' tensor (e.g., vision/audio features).
    """
    def __init__(self, dim, context_dim=None, num_heads=4):
        super().__init__()
        self.dim = dim
        self.context_dim = context_dim if context_dim is not None else dim
        self.num_heads = num_heads
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(self.context_dim, dim)
        self.v_proj = nn.Linear(self.context_dim, dim)
        
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x, context=None):
        """
        x: [B, L, D] or [B, D]
        context: [B, L_c, D_c]
        """
        if x.dim() == 2:
            x = x.unsqueeze(1) # [B, 1, D]
            
        if context is None:
            # Fallback to self-attention if no context provided
            context = x
            
        residual = x
        
        q = self.q_proj(x)
        k = self.k_proj(context)
        v = self.v_proj(context)
        
        attn_out, _ = self.attn(q, k, v)
        out = self.norm(attn_out + residual)
        
        return out.squeeze(1) if out.shape[1] == 1 else out

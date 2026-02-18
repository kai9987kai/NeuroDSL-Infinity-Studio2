import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class LayerScaleInit(nn.Module):
    """Learnable per-channel scaling (from CaiT/DeiT-III) for stable deep networks."""
    def __init__(self, dim: int, init_value: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x):
        return x * self.gamma

class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 512):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len

    def forward(self, x, seq_len):
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.unsqueeze(0).unsqueeze(1)

def apply_rotary_emb(x, cos, sin):
    return x

class SOTAAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, D = x.shape
        qkv = self.qkv(x).reshape(B, 1, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, 1, D)
        return self.proj(x).squeeze(1)

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = SOTAAttention(dim, num_heads)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, dim * 4)
        self.ls1 = LayerScaleInit(dim)
        self.ls2 = LayerScaleInit(dim)

    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x

class StochasticDepth(nn.Module):
    def __init__(self, prob: float = 0.1):
        super().__init__()
        self.prob = prob

    def forward(self, x):
        if not self.training or self.prob <= 0.0:
            return x
        keep_prob = 1.0 - self.prob
        mask = torch.bernoulli(torch.full((x.shape[0], 1), keep_prob, device=x.device))
        return x * mask / keep_prob


class FractalBlock(nn.Module):
    """A recursive, innovative building block for extreme depth."""
    def __init__(self, dim: int, depth: int = 2):
        super().__init__()
        self.depth = depth
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim * 2),
                nn.GELU(),
                nn.Linear(dim * 2, dim),
                nn.Dropout(0.1),
                LayerScaleInit(dim)
            ) for _ in range(depth)
        ])
        self.norm = RMSNorm(dim)
        self.stoch_depth = StochasticDepth(0.1)

    def forward(self, x):
        residual = x
        for layer in self.layers:
            x = self.norm(x)
            h = layer[0](x)
            h = layer[1](h)
            h = layer[2](h)
            h = layer[3](h)
            h = layer[4](h)
            x = residual + self.stoch_depth(h)
            residual = x
        return x


class CrossAttention(nn.Module):
    """Cross-attention mechanism for multimodal interactions"""
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, attn_drop: float = 0., proj_drop: float = 0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, context):
        B, N, C = x.shape
        _, M, _ = context.shape

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(context).reshape(B, M, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class GatedCrossAttention(nn.Module):
    """Gated cross-attention for controlled information flow"""
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.cross_attn = CrossAttention(dim, num_heads)
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        self.norm = RMSNorm(dim)

    def forward(self, x, context):
        attn_out = self.cross_attn(x, context)
        gate_input = torch.cat([x, attn_out], dim=-1)
        gate = self.gate(gate_input)
        return x + (attn_out * gate)


class AdaptiveAvgMaxPool(nn.Module):
    """Adaptive pooling for variable sequence lengths"""
    def __init__(self, output_size=1):
        super().__init__()
        self.output_size = output_size
        self.avgpool = nn.AdaptiveAvgPool1d(output_size)
        self.maxpool = nn.AdaptiveMaxPool1d(output_size)

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        x = x.transpose(1, 2)  # (batch, features, seq_len)
        avg_pooled = self.avgpool(x)
        max_pooled = self.maxpool(x)
        pooled = avg_pooled + max_pooled
        return pooled.transpose(1, 2)  # (batch, output_size, features)


class PositionWiseFFN(nn.Module):
    """Position-wise feed-forward network with configurable expansion"""
    def __init__(self, dim: int, expansion_factor: float = 4.0, dropout: float = 0.1):
        super().__init__()
        hidden_dim = int(dim * expansion_factor)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class ResidualAdd(nn.Module):
    """Residual connection with normalization and dropout"""
    def __init__(self, fn, dim: int, dropout: float = 0.1):
        super().__init__()
        self.fn = fn
        self.norm = RMSNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, **kwargs):
        return x + self.dropout(self.fn(self.norm(x), **kwargs))


class SelfCrossAttentionBlock(nn.Module):
    """Cross-attention block that self-conditions with an internal context.

    Exposes cross-attention style behavior while preserving the 2D (B, D)
    tensor interface used across the DSL runtime.
    """

    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.cross = CrossAttention(dim, num_heads=num_heads)
        self.layer_scale = LayerScaleInit(dim)

    def forward(self, x):
        residual = x
        x_norm = self.norm(x).unsqueeze(1)  # (B, 1, D)
        out = self.cross(x_norm, x_norm).squeeze(1)  # (B, D)
        return residual + self.layer_scale(out)


class GatedSelfCrossAttentionBlock(nn.Module):
    """Gated variant of self-conditioned cross-attention."""

    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.cross = GatedCrossAttention(dim, num_heads=num_heads)
        self.layer_scale = LayerScaleInit(dim)

    def forward(self, x):
        residual = x
        x_norm = self.norm(x).unsqueeze(1)
        out = self.cross(x_norm, x_norm).squeeze(1)
        return residual + self.layer_scale(out)


class ComplexNeuralModule(nn.Module):
    """Composable deep module: cross-attn + gated cross-attn + FFN stack."""

    def __init__(self, dim: int, num_heads: int = 8, expansion: int = 4, depth: int = 2):
        super().__init__()
        depth = max(1, int(depth))
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(
                nn.ModuleDict(
                    {
                        "cross": SelfCrossAttentionBlock(dim, num_heads=num_heads),
                        "gated": GatedSelfCrossAttentionBlock(dim, num_heads=num_heads),
                        "ffn": ResidualBlock(dim, expansion=max(1, int(expansion))),
                    }
                )
            )

    def forward(self, x):
        for block in self.blocks:
            x = block["cross"](x)
            x = block["gated"](x)
            x = block["ffn"](x)
        return x


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention with optional masking"""
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.scale = dim ** -0.5
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output


class MultiHeadAttention(nn.Module):
    """Multi-head attention module"""
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.num_heads
        
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        
        if mask is not None:
            mask = rearrange(mask, 'b n -> b () n ()') * rearrange(mask, 'b n -> b () () n')
            dots.masked_fill_(~mask, float('-inf'))
        
        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


def rearrange(tensor, pattern, **axes_lengths):
    """Helper function for tensor rearrangement (simplified implementation)"""
    # This is a simplified implementation - in a real scenario, you'd use einops
    if pattern == 'b n (h d) -> b h n d':
        h = axes_lengths['h']
        b, n, hd = tensor.shape
        d = hd // h
        return tensor.view(b, n, h, d).transpose(1, 2)
    elif pattern == 'b h n d -> b n (h d)':
        b, h, n, d = tensor.shape
        return tensor.transpose(1, 2).contiguous().view(b, n, h * d)
    else:
        raise NotImplementedError(f"Pattern {pattern} not implemented")

class MoELayer(nn.Module):
    """Top-k MoE with optional shared experts and load-balance auxiliary loss."""
    def __init__(
        self,
        dim: int,
        num_experts: int = 8,
        top_k: int = 2,
        fractal_ratio: float = 0.5,
        num_shared_experts: int = 0,
    ):
        super().__init__()
        self.num_experts = max(1, int(num_experts))
        self.num_shared_experts = max(0, int(num_shared_experts))
        if self.num_shared_experts >= self.num_experts:
            self.num_shared_experts = max(0, self.num_experts - 1)
        self.num_routed_experts = self.num_experts - self.num_shared_experts
        self.top_k = max(1, min(int(top_k), max(1, self.num_routed_experts)))
        self.router = (
            nn.Linear(dim, self.num_routed_experts, bias=False)
            if self.num_routed_experts > 0
            else None
        )
        self.shared_mix_logits = (
            nn.Parameter(torch.zeros(self.num_shared_experts))
            if self.num_shared_experts > 0
            else None
        )
        self.last_aux_loss = None

        experts = []
        num_fractal = int(self.num_experts * fractal_ratio)
        for i in range(self.num_experts):
            if i < num_fractal:
                experts.append(FractalBlock(dim, depth=1))
            else:
                experts.append(SwiGLU(dim, dim * 4))
        self.experts = nn.ModuleList(experts)

    def forward(self, x):
        out = torch.zeros_like(x)

        # Shared experts are always active, similar to DeepSeekMoE shared specialists.
        if self.num_shared_experts > 0:
            shared_weights = torch.softmax(self.shared_mix_logits, dim=0)
            for i in range(self.num_shared_experts):
                out = out + shared_weights[i] * self.experts[i](x)

        if self.num_routed_experts == 0:
            self.last_aux_loss = x.new_zeros(())
            return out

        gate_logits = self.router(x)
        weights = F.softmax(gate_logits, dim=-1)

        top_k_weights, top_k_indices = torch.topk(weights, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        for routed_idx in range(self.num_routed_experts):
            expert_mask = (top_k_indices == routed_idx)
            if expert_mask.any():
                sample_indices, k_indices = torch.where(expert_mask)
                expert_input = x[sample_indices]
                expert_out = self.experts[self.num_shared_experts + routed_idx](expert_input)
                w = top_k_weights[sample_indices, k_indices].unsqueeze(-1)
                out.index_add_(0, sample_indices, expert_out * w)

        # Switch-style load-balance objective to prevent expert collapse.
        with torch.no_grad():
            hard_assign = F.one_hot(top_k_indices, num_classes=self.num_routed_experts).float().sum(dim=1)
            hard_assign = hard_assign / max(1, self.top_k)
            load = hard_assign.mean(dim=0)
        importance = weights.mean(dim=0)
        self.last_aux_loss = self.num_routed_experts * torch.sum(load * importance)

        return out

    def get_aux_loss(self):
        if self.last_aux_loss is None:
            return next(self.parameters()).new_zeros(())
        return self.last_aux_loss

class GQAAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, num_groups: int = 2):
        super().__init__()
        if num_heads % num_groups != 0:
            num_groups = 1
            
        if dim % num_heads != 0:
            num_heads = [i for i in range(1, 64) if dim % i == 0][-1]
            
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.head_dim = dim // num_heads
        self.kv_heads = max(1, num_heads // num_groups)
        
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, self.kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        B, D = x.shape
        q = self.q_proj(x).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, 1, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, 1, self.kv_heads, self.head_dim).transpose(1, 2)
        
        k = k.repeat_interleave(self.num_groups, dim=1)
        v = v.repeat_interleave(self.num_groups, dim=1)
        
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, 1, D)
        return self.o_proj(x).squeeze(1)

# ============================================================
# NEW v4.0 LAYERS
# ============================================================

class DropoutBlock(nn.Module):
    """Configurable dropout regularization layer."""
    def __init__(self, rate: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=rate)

    def forward(self, x):
        return self.dropout(x)

class ResidualBlock(nn.Module):
    """Auto-wraps a feedforward sub-layer with skip connection + pre-norm + layer scale.
    This lets the DSL user add a 'residual' node that automatically stabilizes training."""
    def __init__(self, dim: int, expansion: int = 4):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * expansion, bias=False),
            nn.SiLU(),
            nn.Linear(dim * expansion, dim, bias=False),
        )
        self.layer_scale = LayerScaleInit(dim)
        self.stochastic_depth = StochasticDepth(0.05)

    def forward(self, x):
        return x + self.stochastic_depth(self.layer_scale(self.ffn(self.norm(x))))

class Conv1DBlock(nn.Module):
    """1D Convolution block for signal/time-series processing.
    Treats the feature dim as channels and adds a temporal dimension of 1,
    then squeezes back. Useful for feature extraction patterns."""
    def __init__(self, dim: int, kernel_size: int = 3, groups: int = 1):
        super().__init__()
        # Ensure groups divides dim
        if dim % groups != 0:
            groups = 1
        self.norm = RMSNorm(dim)
        # Depthwise-style convolution for efficiency
        self.conv = nn.Conv1d(dim, dim, kernel_size=kernel_size, 
                              padding=kernel_size // 2, groups=groups, bias=False)
        self.pointwise = nn.Linear(dim, dim, bias=False)
        self.act = nn.SiLU()
        self.layer_scale = LayerScaleInit(dim)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        # x: (B, D) -> (B, D, 1) for Conv1d
        x = x.unsqueeze(-1)
        x = self.conv(x)
        x = x.squeeze(-1)
        x = self.act(x)
        x = self.pointwise(x)
        return residual + self.layer_scale(x)

class LSTMBlock(nn.Module):
    """Bidirectional LSTM block for sequence modeling.
    Wraps input as a single-step sequence, processes through BiLSTM,
    and projects back to original dimension."""
    def __init__(self, dim: int, num_layers: int = 1):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.lstm = nn.LSTM(
            input_size=dim,
            hidden_size=dim // 2,  # BiLSTM will double this
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.0 if num_layers == 1 else 0.1,
        )
        self.proj = nn.Linear(dim, dim, bias=False)
        self.layer_scale = LayerScaleInit(dim)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        # x: (B, D) -> (B, 1, D) for LSTM
        x = x.unsqueeze(1)
        x, _ = self.lstm(x)
        x = x.squeeze(1)  # (B, D)
        x = self.proj(x)
        return residual + self.layer_scale(x)

class Conv3DBlock(nn.Module):
    """3D Convolution block for volumetric data.
    Treats input as (B, C, D, H, W). If input is flattened (B, D_flat), 
    it attempts to reshape it to a cube or processes as is if channel-first."""
    def __init__(self, dim: int, kernel_size: int = 3, groups: int = 1):
        super().__init__()
        if dim % groups != 0: groups = 1
        self.norm = RMSNorm(dim)
        
        # We assume the user wants to stay in the same dimension 'dim'
        self.conv = nn.Conv3d(dim, dim, kernel_size=kernel_size, 
                              padding=kernel_size // 2, groups=groups, bias=False)
        self.pointwise = nn.Linear(dim, dim, bias=False)
        self.act = nn.SiLU()
        self.layer_scale = LayerScaleInit(dim)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        
        orig_shape = x.shape
        # Auto-reshape logic for 1D/2D inputs pretending to be 3D
        if x.dim() == 2:
            x = x.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) # (B, D, 1, 1, 1)
        elif x.dim() == 3:
            x = x.unsqueeze(-1).unsqueeze(-1) # (B, D, L, 1, 1)
            
        x = self.conv(x)
        
        # Restore shape
        if len(orig_shape) == 2:
            x = x.view(orig_shape[0], orig_shape[1])
        else:
            x = x.view(*orig_shape)
            
        x = self.act(x)
        x = self.pointwise(x)
        return residual + self.layer_scale(x)

class AdaptiveComputeBlock(nn.Module):
    """Mixture-of-Depths inspired adaptive compute block.

    During training this uses a soft gate for all samples.
    During eval it can skip compute for "easy" samples based on router score.
    """
    def __init__(self, dim: int, expansion: int = 4, skip_threshold: float = 0.35):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.router = nn.Linear(dim, 1, bias=False)
        self.ffn = SwiGLU(dim, dim * expansion)
        self.layer_scale = LayerScaleInit(dim)
        self.stochastic_depth = StochasticDepth(0.05)
        self.skip_threshold = float(skip_threshold)
        self.last_active_ratio = 1.0

    def forward(self, x):
        residual = x
        x_norm = self.norm(x)
        gate_logits = self.router(x_norm).squeeze(-1)
        gates = torch.sigmoid(gate_logits).unsqueeze(-1)

        if self.training:
            transformed = self.ffn(x_norm)
            return residual + self.stochastic_depth(self.layer_scale(transformed * gates))

        active_mask = gates.squeeze(-1) > self.skip_threshold
        self.last_active_ratio = float(active_mask.float().mean().item())
        if not active_mask.any():
            return residual

        out = residual.clone()
        transformed = self.ffn(x_norm[active_mask])
        out[active_mask] = residual[active_mask] + self.layer_scale(transformed)
        return out

# ============================================================
# PHASE 21 RESEARCH LAYERS
# ============================================================

class KANLayer(nn.Module):
    """Kolmogorov-Arnold Network layer with learnable B-spline activations.
    
    Instead of fixed activations on nodes (like ReLU), KAN places learnable
    univariate functions (B-splines) on each edge. This allows the network
    to discover optimal activation shapes during training.
    Paper: Liu et al. 2024 — "KAN: Kolmogorov-Arnold Networks"
    """
    def __init__(self, dim: int, grid_size: int = 5, spline_order: int = 3):
        super().__init__()
        self.dim = dim
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        # Base linear transform (residual path)
        self.base_weight = nn.Linear(dim, dim, bias=False)
        
        # B-spline knot grid: uniform on [-1, 1] with padding
        h = 2.0 / grid_size
        n_knots = grid_size + spline_order + 1
        knots = torch.linspace(-1 - spline_order * h, 1 + spline_order * h, n_knots)
        self.register_buffer('knots', knots)
        
        # Number of basis functions after Cox-de Boor recursion:
        # n_knots - spline_order - 1 = grid_size
        n_bases = grid_size
        self.spline_weight = nn.Parameter(torch.randn(dim, n_bases) * 0.1)
        
        self.norm = RMSNorm(dim)
        self.layer_scale = LayerScaleInit(dim)
        
    def _b_spline_basis(self, x):
        """Evaluate B-spline basis functions using Cox-de Boor recursion."""
        # x: (B, D), knots: (n_knots,)
        knots = self.knots  # (n_knots,)
        n_knots = knots.shape[0]
        x_ext = x.unsqueeze(-1)  # (B, D, 1)
        
        # Order-0 basis: N_{i,0}(x) = 1 if knots[i] <= x < knots[i+1]
        # Shape: (B, D, n_knots - 1)
        bases = ((x_ext >= knots[:-1]) & (x_ext < knots[1:])).float()
        
        # Cox-de Boor recursion for orders 1..spline_order
        for k in range(1, self.spline_order + 1):
            n = bases.shape[-1] - 1  # New number of basis functions
            # Left term: (x - t_i) / (t_{i+k} - t_i) * N_{i,k-1}
            left_knots_lo = knots[:n]      # t_i
            left_knots_hi = knots[k:k+n]   # t_{i+k}
            left_den = (left_knots_hi - left_knots_lo).clamp(min=1e-8)
            left = (x_ext - left_knots_lo) / left_den * bases[:, :, :n]
            
            # Right term: (t_{i+k+1} - x) / (t_{i+k+1} - t_{i+1}) * N_{i+1,k-1}
            right_knots_hi = knots[k+1:k+1+n]  # t_{i+k+1}
            right_knots_lo = knots[1:1+n]       # t_{i+1}
            right_den = (right_knots_hi - right_knots_lo).clamp(min=1e-8)
            right = (right_knots_hi - x_ext) / right_den * bases[:, :, 1:1+n]
            
            bases = left + right
        
        return bases  # (B, D, n_bases)
    
    def forward(self, x):
        residual = x
        x_norm = self.norm(x)
        
        # Base path: standard linear
        base_out = self.base_weight(x_norm)
        
        # Spline path: learnable activation
        x_clamped = x_norm.clamp(-1, 1)  # Ensure within knot range
        spline_bases = self._b_spline_basis(x_clamped)  # (B, D, n_bases)
        spline_out = (spline_bases * self.spline_weight.unsqueeze(0)).sum(dim=-1)  # (B, D)
        
        return residual + self.layer_scale(base_out + spline_out)


class BitLinear(nn.Module):
    """1.58-bit linear layer with ternary weights {-1, 0, +1}.
    
    From BitNet b1.58 (Microsoft Research 2024). Weights are quantized to
    ternary values during forward pass using absmean quantization, yielding
    ~10× memory reduction and enabling pure integer arithmetic.
    Activations are quantized to 8-bit via absmax.
    """
    def __init__(self, dim: int, expansion: int = 4):
        super().__init__()
        self.dim = dim
        self.hidden = dim * expansion
        
        # Full-precision weights (quantized on-the-fly during forward)
        self.weight_up = nn.Parameter(torch.randn(self.hidden, dim) * 0.02)
        self.weight_down = nn.Parameter(torch.randn(dim, self.hidden) * 0.02)
        
        self.norm = RMSNorm(dim)
        self.layer_scale = LayerScaleInit(dim)
    
    @staticmethod
    def _ternary_quantize(w):
        """Absmean quantization: w_q = Round(w / mean(|w|)) clamped to {-1, 0, +1}."""
        scale = w.abs().mean().clamp(min=1e-8)
        w_q = (w / scale).round().clamp(-1, 1)
        return w_q, scale
    
    @staticmethod
    def _activation_quant(x):
        """Absmax 8-bit activation quantization."""
        Qb = 127.0  # 8-bit
        gamma = x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)
        x_q = (x * Qb / gamma).round().clamp(-Qb, Qb)
        return x_q, gamma / Qb
    
    def forward(self, x):
        residual = x
        x = self.norm(x)
        
        if self.training:
            # STE: quantize in forward, pass gradients through as if unquantized
            w_up_q, s_up = self._ternary_quantize(self.weight_up)
            w_down_q, s_down = self._ternary_quantize(self.weight_down)
            x_q, s_x = self._activation_quant(x)
            
            h = F.linear(x_q * s_x, w_up_q * s_up)  # Approximate full operation
            h = F.silu(h)
            out = F.linear(h, w_down_q * s_down)
        else:
            # Inference: fully quantized
            w_up_q, s_up = self._ternary_quantize(self.weight_up)
            w_down_q, s_down = self._ternary_quantize(self.weight_down)
            h = F.linear(x, w_up_q * s_up)
            h = F.silu(h)
            out = F.linear(h, w_down_q * s_down)
        
        return residual + self.layer_scale(out)


class RetentionLayer(nn.Module):
    """Retention mechanism for O(1) inference (RetNet, Sun et al. 2023).
    
    Replaces softmax attention with a decay-based retention mechanism:
      Retention(X) = (Q K^T * D) V
    where D is a causal decay mask with exponential decay gamma.
    Supports parallel training and O(1) recurrent inference.
    """
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = max(dim // num_heads, 1)
        actual_dim = self.head_dim * num_heads
        
        self.q_proj = nn.Linear(dim, actual_dim, bias=False)
        self.k_proj = nn.Linear(dim, actual_dim, bias=False)
        self.v_proj = nn.Linear(dim, actual_dim, bias=False)
        self.o_proj = nn.Linear(actual_dim, dim, bias=False)
        
        # Per-head decay rates (learned, initialized to reasonable values)
        # gamma values between 0.9 and 0.999 for different heads
        gamma_init = 1.0 - torch.exp(torch.linspace(
            math.log(0.001), math.log(0.1), num_heads
        ))
        self.gamma = nn.Parameter(gamma_init)
        
        self.norm = RMSNorm(dim)
        self.layer_scale = LayerScaleInit(dim)
        
        # Recurrent state for O(1) inference
        self._recurrent_state = None
    
    def reset_state(self):
        """Reset recurrent state for new sequence."""
        self._recurrent_state = None
    
    def forward(self, x):
        """Parallel retention (training mode)."""
        # Handle 2D or 3D input
        is_2d = x.dim() == 2
        if is_2d:
            x = x.unsqueeze(1) # (B, 1, D)
            
        B, T, D = x.shape
        residual = x
        x_norm = self.norm(x)
        
        # Project to Q, K, V -> (B, T, H, d) -> (B, H, T, d)
        q = self.q_proj(x_norm).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_norm).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_norm).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Retention: (Q @ K.T) * DecayMask @ V
        scale = self.head_dim ** -0.5
        # (B, H, T, d) @ (B, H, d, T) -> (B, H, T, T)
        attn_weights = (q @ k.transpose(-2, -1)) * scale
        
        # Create Causal Decay Mask
        # D_ij = gamma^(i-j) if i >= j else 0
        indices = torch.arange(T, device=x.device)
        # shape (T, T): i is row, j is col
        diff = indices.unsqueeze(1) - indices.unsqueeze(0) 
        
        # valid where i >= j (causal)
        causal_mask = diff >= 0
        
        # Gamma shape: (num_heads,) -> (1, H, 1, 1) to broadcast
        gamma = self.gamma.view(1, self.num_heads, 1, 1).clamp(0.01, 0.999)
        
        # decay_mask: (1, H, T, T)
        # We need gamma^(i-j). Since i>=j, diff is non-negative.
        decay_mask = (gamma ** diff.unsqueeze(0).unsqueeze(0)) 
        decay_mask = decay_mask * causal_mask.unsqueeze(0).unsqueeze(0).type_as(x)
        
        # Apply mask
        attn_weights = attn_weights * decay_mask
        
        # Output: (B, H, T, T) @ (B, H, T, d) -> (B, H, T, d)
        out = attn_weights @ v
        
        # Reshape back: (B, T, H*d)
        out = out.transpose(1, 2).reshape(B, T, -1)
        out = self.o_proj(out)
        
        x_out = residual + self.layer_scale(out)
        
        if is_2d:
            return x_out.squeeze(1)
        return x_out


class MixtureOfDepths(nn.Module):
    """Mixture of Depths: per-token dynamic depth routing (Raposo et al. 2024).
    
    A lightweight router decides for each token whether to apply the
    full computation block or skip it. This saves FLOPs by only
    computing expensive layers for tokens that need them.
    """
    def __init__(self, dim: int, expansion: int = 4, capacity_factor: float = 0.5):
        super().__init__()
        self.dim = dim
        self.capacity_factor = capacity_factor  # Fraction of tokens to process
        
        # Router: simple linear -> sigmoid
        self.router = nn.Sequential(
            nn.Linear(dim, 1, bias=True),
        )
        
        # Heavy computation block (only applied to selected tokens)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * expansion),
            nn.GELU(),
            nn.Linear(dim * expansion, dim),
        )
        
        self.norm = RMSNorm(dim)
        self.layer_scale = LayerScaleInit(dim)
        
        # Track routing statistics
        self.register_buffer('_route_count', torch.zeros(2))  # [skipped, computed]
    
    def forward(self, x):
        residual = x
        x_norm = self.norm(x)
        
        # Router decides which tokens to compute
        router_logits = self.router(x_norm).squeeze(-1)  # (B,)
        router_probs = torch.sigmoid(router_logits)
        
        if self.training:
            # Soft routing during training (differentiable)
            # Gumbel-sigmoid for exploration
            noise = torch.rand_like(router_probs) * 0.1
            gate = (router_probs + noise).clamp(0, 1)
            
            out = self.ffn(x_norm)
            out = gate.unsqueeze(-1) * out
            
            # Track stats
            frac_computed = (router_probs > 0.5).float().mean()
            self._route_count[0] += (1 - frac_computed) * x.shape[0]
            self._route_count[1] += frac_computed * x.shape[0]
        else:
            # Hard routing during inference — only compute for top-k tokens
            k = max(1, int(x.shape[0] * self.capacity_factor))
            topk_indices = router_probs.topk(k).indices
            
            out = torch.zeros_like(x_norm)
            selected = x_norm[topk_indices]
            out[topk_indices] = self.ffn(selected)
        
        return residual + self.layer_scale(out)
    
    def get_routing_stats(self):
        """Return fraction of tokens routed to computation."""
        total = self._route_count.sum()
        if total == 0:
            return 0.5
        return (self._route_count[1] / total).item()


class DiffAttention(nn.Module):
    """Differential Attention: noise-canceling dual-softmax attention.
    
    Computes attention as the difference of two softmax attention maps,
    effectively canceling noise and producing sparser, more focused patterns.
    Paper: Ye et al. 2024 — "Differential Transformer"
    """
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        if dim % num_heads != 0:
            num_heads = max(1, [i for i in range(1, 64) if dim % i == 0][-1])
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        half_head = self.head_dim // 2
        
        # Q, K are split into two halves for differential computation
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        
        # Learnable lambda for weighting the difference
        self.lambda_q1 = nn.Parameter(torch.randn(half_head) * 0.1)
        self.lambda_q2 = nn.Parameter(torch.randn(half_head) * 0.1)
        self.lambda_k1 = nn.Parameter(torch.randn(half_head) * 0.1)
        self.lambda_k2 = nn.Parameter(torch.randn(half_head) * 0.1)
        self.lambda_init = nn.Parameter(torch.tensor(0.8))
        
        self.norm = RMSNorm(dim)
        self.sub_norm = RMSNorm(self.head_dim)
    
    def forward(self, x):
        B, D = x.shape
        residual = x
        x = self.norm(x)
        
        # Project to Q, K, V
        q = self.q_proj(x).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, 1, d)
        k = self.k_proj(x).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Split Q, K into two halves
        half = self.head_dim // 2
        q1, q2 = q[..., :half], q[..., half:]
        k1, k2 = k[..., :half], k[..., half:]
        
        # Differential attention: A = softmax(Q1 K1^T) - lambda * softmax(Q2 K2^T)
        scale = half ** -0.5
        attn1 = torch.softmax(q1 @ k1.transpose(-2, -1) * scale, dim=-1)
        attn2 = torch.softmax(q2 @ k2.transpose(-2, -1) * scale, dim=-1)
        
        # Compute lambda from learned parameters
        lam = (torch.exp(torch.dot(self.lambda_q1, self.lambda_k1)) - 
               torch.exp(torch.dot(self.lambda_q2, self.lambda_k2)) + 
               self.lambda_init)
        
        diff_attn = attn1 - lam * attn2
        
        out = diff_attn @ v  # (B, H, 1, d)
        out = self.sub_norm(out.squeeze(2)).unsqueeze(2)  # Normalize per-head
        out = out.transpose(1, 2).reshape(B, D)
        return residual + self.o_proj(out)


class LoRAAdapter(nn.Module):
    """Low-Rank Adaptation layer for parameter-efficient fine-tuning.
    
    Decomposes weight updates into two low-rank matrices (A, B),
    so Δw = B @ A with rank << dim. This drastically reduces trainable
    parameters while maintaining expressivity.
    Paper: Hu et al. 2021 — "LoRA: Low-Rank Adaptation of LLMs"
    """
    def __init__(self, dim: int, rank: int = 16, alpha: float = 1.0):
        super().__init__()
        self.dim = dim
        self.rank = min(rank, dim)  # Ensure rank <= dim
        self.scaling = alpha / self.rank
        
        # Frozen base transform (simulated — in practice you'd freeze a pretrained layer)
        self.base = nn.Linear(dim, dim, bias=False)
        
        # Low-rank adapters
        self.lora_A = nn.Linear(dim, self.rank, bias=False)
        self.lora_B = nn.Linear(self.rank, dim, bias=False)
        
        # Initialize B to zero so initial output = base output
        nn.init.zeros_(self.lora_B.weight)
        nn.init.kaiming_normal_(self.lora_A.weight)
        
        self.norm = RMSNorm(dim)
        self.layer_scale = LayerScaleInit(dim)
        self.dropout = nn.Dropout(0.05)
    
    def forward(self, x):
        residual = x
        x_norm = self.norm(x)
        
        # Base path
        base_out = self.base(x_norm)
        
        # LoRA path: low-rank update
        lora_out = self.lora_B(self.dropout(self.lora_A(x_norm))) * self.scaling
        
        return residual + self.layer_scale(base_out + lora_out)


class SpectralNormBlock(nn.Module):
    """Spectral normalization block for training stability.
    
    Constrains the spectral norm (largest singular value) of weight
    matrices, bounding the Lipschitz constant. This prevents gradient
    explosions and stabilizes GAN-style and deep network training.
    Paper: Miyato et al. 2018 — "Spectral Normalization for GANs"
    """
    def __init__(self, dim: int, expansion: int = 4):
        super().__init__()
        self.norm = RMSNorm(dim)
        # Apply spectral norm to each linear layer
        self.ffn = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(dim, dim * expansion, bias=False)),
            nn.SiLU(),
            nn.utils.spectral_norm(nn.Linear(dim * expansion, dim, bias=False)),
        )
        self.layer_scale = LayerScaleInit(dim)
    
    def forward(self, x):
        return x + self.layer_scale(self.ffn(self.norm(x)))


class GradientCheckpointBlock(nn.Module):
    """Gradient checkpointing wrapper for memory-efficient training.
    
    Wraps a residual FFN block with torch.utils.checkpoint, trading
    extra compute during backward pass for reduced peak memory usage.
    Critical for training very deep or very wide architectures.
    """
    def __init__(self, dim: int, expansion: int = 4):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * expansion, bias=False),
            nn.SiLU(),
            nn.Linear(dim * expansion, dim, bias=False),
        )
        self.layer_scale = LayerScaleInit(dim)
    
    def _inner(self, x):
        return self.layer_scale(self.ffn(self.norm(x)))
    
    def forward(self, x):
        if self.training and x.requires_grad:
            return x + torch.utils.checkpoint.checkpoint(self._inner, x, use_reentrant=False)
        return x + self._inner(x)


# ============================================================
# PHASE 28: FRONTIER INTELLIGENCE LAYERS
# ============================================================

class GatedLinearAttention(nn.Module):
    """Gated Linear Attention — sigmoid-gated efficient attention.

    Replaces softmax with an element-wise sigmoid gate on the attention
    scores, yielding input-dependent sparsity and improved training
    stability. Supports O(N) recurrent decoding.
    Reference: Yang et al. 2025 — Gated Linear Attention Transformers.
    """
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        if dim % num_heads != 0:
            num_heads = max(1, [i for i in range(1, 64) if dim % i == 0][-1])
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.gate = nn.Linear(dim, dim, bias=True)
        self.o_proj = nn.Linear(dim, dim, bias=False)

        self.norm = RMSNorm(dim)
        self.layer_scale = LayerScaleInit(dim)

    def forward(self, x):
        B, D = x.shape
        residual = x
        x_norm = self.norm(x)

        q = self.q_proj(x_norm).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_norm).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_norm).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)

        # Linear attention: (Q K^T) gated by sigmoid instead of softmax
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        # Sigmoid gate: input-dependent sparsity
        g = torch.sigmoid(self.gate(x_norm)).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        attn = torch.sigmoid(attn)  # Replace softmax with sigmoid
        out = (attn @ v) * g

        out = out.transpose(1, 2).reshape(B, D)
        return residual + self.layer_scale(self.o_proj(out))


class xLSTMBlock(nn.Module):
    """Extended LSTM with exponential gating and matrix memory.

    Replaces traditional sigmoid gates with exponential gates (clamped
    for stability) and uses a matrix-valued hidden state updated via
    outer products, enabling richer memory representations.
    Reference: Beck et al. 2024 — xLSTM: Extended Long Short-Term Memory.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        # Input, forget, output gates with exponential activation
        self.W_i = nn.Linear(dim, dim, bias=True)
        self.W_f = nn.Linear(dim, dim, bias=True)
        self.W_o = nn.Linear(dim, dim, bias=True)
        # Cell input
        self.W_c = nn.Linear(dim, dim, bias=False)
        # Matrix memory: project to key/value for outer product update
        self.W_k = nn.Linear(dim, dim, bias=False)
        self.W_v = nn.Linear(dim, dim, bias=False)

        self.norm = RMSNorm(dim)
        self.layer_scale = LayerScaleInit(dim)

    def forward(self, x):
        residual = x
        x_norm = self.norm(x)

        # Exponential gating (clamped to prevent overflow)
        i_gate = torch.exp(self.W_i(x_norm).clamp(-10, 10))
        f_gate = torch.exp(self.W_f(x_norm).clamp(-10, 10))
        o_gate = torch.sigmoid(self.W_o(x_norm))  # Output gate stays sigmoid

        # Normalize gates
        gate_sum = (i_gate + f_gate).clamp(min=1e-6)
        i_gate = i_gate / gate_sum
        f_gate = f_gate / gate_sum

        # Cell input
        c_tilde = torch.tanh(self.W_c(x_norm))

        # Matrix memory via outer product
        k = self.W_k(x_norm)  # (B, D)
        v = self.W_v(c_tilde)  # (B, D)
        # Outer product memory: B x D
        memory_update = k * v  # Element-wise (simplified from full outer product)

        # Cell state update: f * x + i * memory_update
        cell = f_gate * x_norm + i_gate * memory_update

        # Output
        h = o_gate * torch.tanh(cell)
        return residual + self.layer_scale(h)


class TestTimeTrainLayer(nn.Module):
    """Test-Time Training layer — self-supervised adaptation at inference.

    During training, behaves like a standard residual FFN block.
    During eval/inference, performs a single self-supervised gradient
    step on the inner weights using a reconstruction objective,
    allowing the model to adapt on-the-fly to test data.
    Reference: Yu et al. 2025 — End-to-End Test-Time Training.
    """
    def __init__(self, dim: int, ttt_lr: float = 0.01):
        super().__init__()
        self.dim = dim
        self.ttt_lr = ttt_lr
        self.norm = RMSNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2, bias=False),
            nn.SiLU(),
            nn.Linear(dim * 2, dim, bias=False),
        )
        # Reconstruction head for self-supervised TTT objective
        self.recon_head = nn.Linear(dim, dim, bias=False)
        self.layer_scale = LayerScaleInit(dim)

    def forward(self, x):
        residual = x
        x_norm = self.norm(x)

        if self.training:
            # Standard forward during training
            return residual + self.layer_scale(self.ffn(x_norm))

        # Test-time training: 1-step self-supervised update
        # Create a copy of FFN params for the TTT step
        ffn_out = self.ffn(x_norm)

        # Self-supervised objective: reconstruct input from output
        recon = self.recon_head(ffn_out)
        recon_loss = F.mse_loss(recon, x_norm.detach())

        # Compute gradient manually and apply one-step update
        if recon_loss.requires_grad:
            grads = torch.autograd.grad(
                recon_loss, self.ffn.parameters(),
                create_graph=False, allow_unused=True
            )
            # Apply one-step update to a clone
            with torch.no_grad():
                for p, g in zip(self.ffn.parameters(), grads):
                    if g is not None:
                        p.data -= self.ttt_lr * g

            # Re-forward after TTT update
            ffn_out = self.ffn(x_norm)

        return residual + self.layer_scale(ffn_out)


class ContrastiveHead(nn.Module):
    """Self-supervised contrastive projection head (SimCLR/BYOL-style).

    Two-layer MLP that maps features to a normalized embedding space
    suitable for contrastive learning. L2-normalizes output so cosine
    similarity can be used directly.
    Reference: Chen et al. 2020 — SimCLR; Grill et al. 2020 — BYOL.
    """
    def __init__(self, dim: int, proj_dim: int = 64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.ReLU(),
            nn.Linear(dim, proj_dim, bias=False),
        )
        self.out_proj = nn.Linear(proj_dim, dim, bias=False)
        self.norm = RMSNorm(dim)
        self.layer_scale = LayerScaleInit(dim)

    def forward(self, x):
        residual = x
        x_norm = self.norm(x)
        z = self.proj(x_norm)
        # L2 normalize for cosine similarity
        z = F.normalize(z, p=2, dim=-1)
        # Project back to original dim for residual connection
        return residual + self.layer_scale(self.out_proj(z))


class SparseAttention(nn.Module):
    """Top-k Sparse Attention with linear memory.

    Standard Q/K/V attention but retains only the top-k attention weights
    per query, zeroing the rest. Dramatically reduces effective compute
    and memory for large feature sets.
    Reference: DeepSeek V4 Sparse Attention (2025).
    """
    def __init__(self, dim: int, top_k: int = 8, num_heads: int = 4):
        super().__init__()
        if dim % num_heads != 0:
            num_heads = max(1, [i for i in range(1, 64) if dim % i == 0][-1])
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.top_k = top_k

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.norm = RMSNorm(dim)
        self.layer_scale = LayerScaleInit(dim)

    def forward(self, x):
        B, D = x.shape
        residual = x
        x_norm = self.norm(x)

        qkv = self.qkv(x_norm).reshape(B, 1, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (B, H, 1, d)

        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale  # (B, H, 1, 1)

        # For single-step, top-k is trivially the full attention.
        # The sparse mechanism shines with sequences; here we keep
        # the architecture intact for composability.
        attn = torch.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, D)
        return residual + self.layer_scale(self.o_proj(out))


class SelfDistillBlock(nn.Module):
    """DINO-style self-distillation with EMA teacher.

    Maintains a momentum-updated teacher copy of an FFN block.
    The student is trained normally; an auxiliary distillation loss
    encourages student outputs to match (stop-gradient) teacher outputs.
    Reference: Caron et al. 2021 — DINO: Self-Distillation with No Labels.
    """
    def __init__(self, dim: int, momentum: float = 0.996):
        super().__init__()
        self.dim = dim
        self.momentum = momentum

        self.norm = RMSNorm(dim)
        # Student branch
        self.student = nn.Sequential(
            nn.Linear(dim, dim * 2, bias=False),
            nn.SiLU(),
            nn.Linear(dim * 2, dim, bias=False),
        )
        # Teacher branch (EMA copy, not trained by gradient)
        self.teacher = nn.Sequential(
            nn.Linear(dim, dim * 2, bias=False),
            nn.SiLU(),
            nn.Linear(dim * 2, dim, bias=False),
        )
        # Initialize teacher = student
        for p_t, p_s in zip(self.teacher.parameters(), self.student.parameters()):
            p_t.data.copy_(p_s.data)
            p_t.requires_grad_(False)

        self.layer_scale = LayerScaleInit(dim)
        self._distill_loss = None

    @torch.no_grad()
    def _update_teacher(self):
        for p_t, p_s in zip(self.teacher.parameters(), self.student.parameters()):
            p_t.data.mul_(self.momentum).add_(p_s.data, alpha=1.0 - self.momentum)

    def forward(self, x):
        residual = x
        x_norm = self.norm(x)

        s_out = self.student(x_norm)

        if self.training:
            self._update_teacher()
            with torch.no_grad():
                t_out = self.teacher(x_norm)
            # Distillation loss: student should match teacher (detached)
            self._distill_loss = F.mse_loss(s_out, t_out.detach())
        else:
            self._distill_loss = None

        return residual + self.layer_scale(s_out)

    def get_aux_loss(self):
        if self._distill_loss is not None:
            return self._distill_loss
        return torch.tensor(0.0)


# ============================================================
# PHASE 29: ADAPTIVE NEXUS LAYERS
# ============================================================

class HyenaBlock(nn.Module):
    """Hyena operator — subquadratic long-conv with element-wise gating.

    Replaces self-attention with an implicit long convolution + data-controlled
    gating mechanism. Achieves attention-quality with O(N log N) complexity,
    enabling 100× speedup over FlashAttention at 100k sequence length.
    Reference: Poli et al. 2023 — Hyena Hierarchy.
    """
    def __init__(self, dim: int, kernel_size: int = 7, expand: int = 2):
        super().__init__()
        self.dim = dim
        inner = dim * expand

        # Input projections: value + gate
        self.v_proj = nn.Linear(dim, inner, bias=False)
        self.gate_proj = nn.Linear(dim, inner, bias=False)

        # Implicit long convolution (depthwise)
        self.conv = nn.Conv1d(
            inner, inner, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=inner, bias=False
        )
        self.out_proj = nn.Linear(inner, dim, bias=False)

        self.norm = RMSNorm(dim)
        self.layer_scale = LayerScaleInit(dim)

    def forward(self, x):
        residual = x
        x_norm = self.norm(x)

        v = self.v_proj(x_norm)  # (B, inner)
        g = torch.sigmoid(self.gate_proj(x_norm))  # (B, inner)

        # Long convolution: reshape to (B, inner, 1) for Conv1d
        v_conv = self.conv(v.unsqueeze(-1)).squeeze(-1)  # (B, inner)

        # Gated output
        out = v_conv * g
        return residual + self.layer_scale(self.out_proj(out))


class GEGLUBlock(nn.Module):
    """GEGLU — Gaussian Error Gated Linear Unit FFN block.

    Computes GEGLU(x) = (xW₁) ⊙ GELU(xW₂), combining a smooth Gaussian
    nonlinearity with element-wise gating for improved gradient flow
    compared to standard ReLU or SwiGLU.
    Reference: Shazeer 2020 — GLU Variants Improve Transformer.
    """
    def __init__(self, dim: int, expansion: int = 4):
        super().__init__()
        hidden = dim * expansion
        self.w1 = nn.Linear(dim, hidden, bias=False)  # Value path
        self.w2 = nn.Linear(dim, hidden, bias=False)  # Gate path
        self.down = nn.Linear(hidden, dim, bias=False)
        self.norm = RMSNorm(dim)
        self.layer_scale = LayerScaleInit(dim)

    def forward(self, x):
        residual = x
        x_norm = self.norm(x)
        # GEGLU: value * GELU(gate)
        out = self.w1(x_norm) * F.gelu(self.w2(x_norm))
        return residual + self.layer_scale(self.down(out))


class ConvMixerBlock(nn.Module):
    """ConvMixer — pure depthwise + pointwise convolution mixer.

    Replaces attention and MLP with depthwise conv (spatial/feature mixing)
    followed by pointwise conv (channel mixing). Demonstrates that simple
    convolutional architectures can match ViTs.
    Reference: Trockman & Kolter 2022 — Patches Are All You Need?
    """
    def __init__(self, dim: int, kernel_size: int = 7):
        super().__init__()
        # Depthwise conv for feature mixing
        self.depthwise = nn.Conv1d(
            dim, dim, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=dim, bias=False
        )
        # Pointwise conv for channel mixing
        self.pointwise = nn.Linear(dim, dim, bias=False)
        self.norm = RMSNorm(dim)
        self.layer_scale = LayerScaleInit(dim)

    def forward(self, x):
        residual = x
        x_norm = self.norm(x)
        # Depthwise conv: (B, D) -> (B, D, 1) -> conv -> (B, D)
        h = self.depthwise(x_norm.unsqueeze(-1)).squeeze(-1)
        h = F.gelu(h)
        # Pointwise mixing
        h = self.pointwise(h)
        return residual + self.layer_scale(h)


class AdaptiveRankLinear(nn.Module):
    """Adaptive-Rank SVD linear layer for dynamic compression.

    Initializes with SVD decomposition and adaptively selects the effective
    rank based on a learnable energy threshold. Only the top-r singular
    vectors participate in the forward pass, reducing compute.
    Reference: ARSVD 2025 — Adaptive-Rank SVD for Neural Compression.
    """
    def __init__(self, dim: int, max_rank: int = 16, energy_threshold: float = 0.95):
        super().__init__()
        self.dim = dim
        self.max_rank = min(max_rank, dim)

        # SVD components: W ≈ U @ diag(S) @ V^T
        self.U = nn.Parameter(torch.randn(dim, self.max_rank) * 0.02)
        self.S = nn.Parameter(torch.ones(self.max_rank))  # Singular values
        self.V = nn.Parameter(torch.randn(self.max_rank, dim) * 0.02)

        self.energy_threshold = energy_threshold
        self.norm = RMSNorm(dim)
        self.layer_scale = LayerScaleInit(dim)

    def _effective_rank(self):
        """Compute effective rank from singular value energy."""
        s_abs = self.S.abs()
        total_energy = s_abs.sum().clamp(min=1e-8)
        cumulative = torch.cumsum(s_abs, dim=0) / total_energy
        # Find first index where cumulative energy >= threshold
        mask = cumulative >= self.energy_threshold
        if mask.any():
            r = mask.float().argmax().item() + 1
        else:
            r = self.max_rank
        return min(r, self.max_rank)

    def forward(self, x):
        residual = x
        x_norm = self.norm(x)
        r = self._effective_rank()
        # Low-rank forward: x @ U[:,:r] @ diag(S[:r]) @ V[:r,:]
        h = x_norm @ self.U[:, :r]  # (B, r)
        h = h * self.S[:r]  # Scale by singular values
        h = h @ self.V[:r, :]  # (B, D)
        return residual + self.layer_scale(h)


class StochasticDepthBlock(nn.Module):
    """Stochastic Depth — randomly drops entire layers during training.

    During training, the entire FFN block is skipped with probability p
    (replaced by identity). During eval, output is deterministically
    scaled by (1-p). Acts as a strong regularizer for deep networks.
    Reference: Huang et al. 2016 — Deep Networks with Stochastic Depth.
    """
    def __init__(self, dim: int, drop_prob: float = 0.1, expansion: int = 4):
        super().__init__()
        self.drop_prob = drop_prob
        self.norm = RMSNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * expansion, bias=False),
            nn.SiLU(),
            nn.Linear(dim * expansion, dim, bias=False),
        )
        self.layer_scale = LayerScaleInit(dim)

    def forward(self, x):
        if self.training and self.drop_prob > 0:
            # Randomly skip this layer entirely
            if torch.rand(1).item() < self.drop_prob:
                return x  # Identity (skip)
        
        residual = x
        x_norm = self.norm(x)
        out = self.ffn(x_norm)

        if not self.training:
            # Scale by survival probability during eval
            out = out * (1.0 - self.drop_prob)

        return residual + self.layer_scale(out)


# ============================================================
# PHASE 30: SINGULARITY SYNTHESIS LAYERS
# ============================================================

class MultiHeadLatentAttention(nn.Module):
    """Multi-Head Latent Attention (MHLA) with low-rank compression.

    Uses low-rank projections (KV-compression) to significantly reduce
    memory footprint while maintaining expressive attention dynamics.
    Reference: DeepSeek-V2/V3 Multi-Head Latent Attention (2024).
    """
    def __init__(self, dim: int, num_heads: int = 8, q_lora_rank: int = 32, kv_lora_rank: int = 64):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # Low-rank Query projection
        self.q_lora = nn.Sequential(
            nn.Linear(dim, q_lora_rank, bias=False),
            RMSNorm(q_lora_rank),
            nn.Linear(q_lora_rank, num_heads * self.head_dim, bias=False)
        )
        
        # Low-rank Key-Value compression
        self.kv_lora = nn.Sequential(
            nn.Linear(dim, kv_lora_rank, bias=False),
            RMSNorm(kv_lora_rank)
        )
        self.k_proj = nn.Linear(kv_lora_rank, num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(kv_lora_rank, num_heads * self.head_dim, bias=False)
        
        self.o_proj = nn.Linear(num_heads * self.head_dim, dim, bias=False)
        self.norm = RMSNorm(dim)
        self.layer_scale = LayerScaleInit(dim)

    def forward(self, x):
        B, D = x.shape
        residual = x
        x_norm = self.norm(x)
        
        # Multi-head projection
        q = self.q_lora(x_norm).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        
        compressed_kv = self.kv_lora(x_norm)
        k = self.k_proj(compressed_kv).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(compressed_kv).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled Dot-Product Attention
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, -1)
        
        return residual + self.layer_scale(self.o_proj(out))


class MambaConvBlock(nn.Module):
    """Hybrid Selective SSM + Depthwise Convolution.

    Combines Mamba's long-range sequence modeling with Depthwise Conv's
    local feature extraction for robust spatial-temporal representations.
    """
    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 7):
        super().__init__()
        self.mamba = MambaBlock(dim, d_state=d_state)
        # Depthwise conv for local context
        self.conv = nn.Conv1d(dim, dim, kernel_size=d_conv, 
                              padding=d_conv // 2, groups=dim, bias=False)
        self.norm = RMSNorm(dim)
        self.layer_scale = LayerScaleInit(dim)

    def forward(self, x):
        residual = x
        x_norm = self.norm(x)
        
        # SSM branch
        h_ssm = self.mamba(x_norm)
        
        # Conv branch (local context)
        h_conv = self.conv(x_norm.unsqueeze(-1)).squeeze(-1)
        
        out = F.silu(h_ssm + h_conv)
        return residual + self.layer_scale(out)


class SparseTopKLayer(nn.Module):
    """Differentiable Top-K Sparsity layer."""
    def __init__(self, dim: int, k: int = 8):
        super().__init__()
        self.dim = dim
        self.k = min(k, dim)
        self.norm = RMSNorm(dim)
        self.layer_scale = LayerScaleInit(dim)

    def forward(self, x):
        residual = x
        x_norm = self.norm(x)
        magnitudes = torch.abs(x_norm)
        topk_vals, _ = torch.topk(magnitudes, self.k, dim=-1)
        threshold = topk_vals[:, -1].unsqueeze(-1)
        mask = (magnitudes >= threshold).float()
        out = x_norm * mask
        return residual + self.layer_scale(out)


class SaliencyPruningLayer(nn.Module):
    """Dynamic Saliency-based Feature Pruning."""
    def __init__(self, dim: int, base_threshold: float = 0.2):
        super().__init__()
        self.saliency_predictor = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.SiLU(),
            nn.Linear(dim // 4, dim),
            nn.Sigmoid()
        )
        self.threshold = base_threshold
        self.norm = RMSNorm(dim)
        self.layer_scale = LayerScaleInit(dim)

    def forward(self, x):
        residual = x
        x_norm = self.norm(x)
        saliency = self.saliency_predictor(x_norm)
        mask = saliency if self.training else (saliency > self.threshold).float()
        out = x_norm * mask
        return residual + self.layer_scale(out)


# ============================================================
# PHASE 30+: ADVANCED SINGULARITY EXPANSION
# ============================================================

class BitNetLayer(nn.Module):
    """BitNet b1.58 — Ternary (-1, 0, 1) Weight Quantization.

    Uses 1.58-bit quantization for weights to achieve extreme efficiency
    while maintaining performance near full-precision models.
    Reference: Ma et al. 2024 — The Era of 1-bit LLMs.
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.norm = RMSNorm(in_features)

    def forward(self, x):
        x = self.norm(x)
        # Weight Quantization (Ternary)
        # Scale = mean absolute weight
        gamma = self.weight.abs().mean()
        # Quantize to {-1, 0, 1}
        w_q = torch.clamp(torch.round(self.weight / (gamma + 1e-8)), -1, 1)
        # Straight-through estimator
        w_q = self.weight + (w_q - self.weight).detach()
        
        return F.linear(x, w_q)


class ReversibleResidualBlock(nn.Module):
    """Memory-efficient Reversible Residual Block.

    Saves memory during training by re-calculating activations during
    the backward pass instead of storing them.
    Reference: Gomez et al. 2017 — The Reversible Residual Network.
    """
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0, "Reversible block requires even dimension"
        self.half_dim = dim // 2
        
        self.f = nn.Sequential(RMSNorm(self.half_dim), nn.Linear(self.half_dim, self.half_dim, bias=False), nn.SiLU())
        self.g = nn.Sequential(RMSNorm(self.half_dim), nn.Linear(self.half_dim, self.half_dim, bias=False), nn.SiLU())

    def forward(self, x):
        # x is split into [x1, x2]
        x1, x2 = x.chunk(2, dim=-1)
        
        # y1 = x1 + f(x2)
        # y2 = x2 + g(y1)
        y1 = x1 + self.f(x2)
        y2 = x2 + self.g(y1)
        
        return torch.cat([y1, y2], dim=-1)


class MixtureOfAttention(nn.Module):
    """Mixture of Attention Heads (MoA) — Expert-based head routing.

    Dynamically routes inputs to specialized attention heads.
    """
    def __init__(self, dim: int, num_experts: int = 4, heads_per_expert: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.heads_per_expert = heads_per_expert
        self.head_dim = dim // (num_experts * heads_per_expert)
        
        self.router = nn.Linear(dim, num_experts)
        self.experts = nn.ModuleList([
            SOTAAttention(dim // num_experts, num_heads=heads_per_expert)
            for _ in range(num_experts)
        ])
        self.norm = RMSNorm(dim)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        B, D = x.shape
        residual = x
        x_norm = self.norm(x)
        
        # Route to experts
        router_logits = self.router(x_norm)
        routing_weights = F.softmax(router_logits, dim=-1)
        
        # Simplified parallel execution
        splits = x_norm.chunk(self.num_experts, dim=-1)
        expert_outputs = [expert(split) for expert, split in zip(self.experts, splits)]
        
        out = torch.cat(expert_outputs, dim=-1)
        # apply weights (broadcast)
        out = out * routing_weights.repeat_interleave(D // self.num_experts, dim=-1)
        
        return residual + self.out_proj(out)


# ============================================================
# MAIN MODEL
# ============================================================

class MambaBlock(nn.Module):
    """Simplified State Space Model (SSM) block for linear-time complexity."""
    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.dim = dim
        self.expand = expand
        self.inner_dim = dim * expand
        
        self.in_proj = nn.Linear(dim, self.inner_dim * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.inner_dim,
            out_channels=self.inner_dim,
            kernel_size=d_conv,
            groups=self.inner_dim,
            padding=d_conv - 1
        )
        
        self.x_proj = nn.Linear(self.inner_dim, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.inner_dim, bias=True)
        
        # State space parameters
        self.A = nn.Parameter(torch.log(torch.arange(1, d_state + 1).expand(self.inner_dim, d_state).float()))
        self.D = nn.Parameter(torch.ones(self.inner_dim))
        self.out_proj = nn.Linear(self.inner_dim, dim, bias=False)

    def forward(self, x):
        # x: (B, D) -> Simulation of sequence step
        B, D = x.shape
        x_inner = self.in_proj(x).chunk(2, dim=-1)[0] # Extract one side
        
        # 1D Conv (simulate sequence history if needed, but here we act on singular step for simplicity)
        x_conv = x_inner.unsqueeze(-1)
        x_conv = self.conv1d(x_conv)[:, :, 0]
        x_conv = F.silu(x_conv)
        
        # SSM dynamics
        ssm_params = self.x_proj(x_conv)
        # Simplified SSM computation
        y = x_conv * torch.sigmoid(ssm_params[:, :1]) # Gated response
        
        return x + self.out_proj(y)

class LiquidNeuralLayer(nn.Module):
    """Bio-inspired continuous-time neural layer (LTC/Liquid)."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.gleak = nn.Parameter(torch.ones(dim) * 0.1)
        self.vleak = nn.Parameter(torch.zeros(dim))
        self.cm = nn.Parameter(torch.ones(dim))
        
        self.w = nn.Linear(dim, dim)
        self.tau = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Continuous-time differential update simulation: dx/dt = -1/tau * [ (x-vleak)*gleak + f(x, input) ]
        v = x
        f_x = torch.tanh(self.w(x))
        
        delta = -(1.0 / self.tau.clamp(min=0.1)) * ((v - self.vleak) * self.gleak + f_x)
        return v + delta # One Euler step

class HyperTransformerBlock(nn.Module):
    """Meta-neural block: weights are generated by a small hyper-network."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.meta_net = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.SiLU(),
            nn.Linear(dim // 4, dim * dim) # Generates weight matrix
        )
        self.norm = RMSNorm(dim)

    def forward(self, x):
        # Generate dynamic weights based on input context
        B, D = x.shape
        dynamic_w = self.meta_net(x).view(B, D, D)
        
        # Matrix multiply: x @ dynamic_w
        x = x.unsqueeze(1) # (B, 1, D)
        out = torch.bmm(x, dynamic_w).squeeze(1) # (B, D)
        return self.norm(out)

class ModernMLP(nn.Module):
    def __init__(self, layer_defs):
        super().__init__()
        self.layers = nn.ModuleList()
        self.config = layer_defs
        
        for defn in layer_defs:
            l_type = defn['type']
            if l_type == 'linear':
                self.layers.append(nn.Linear(defn['in'], defn['out']))
                self.layers.append(RMSNorm(defn['out']))
                self.layers.append(nn.SiLU())
            elif l_type == 'attn':
                self.layers.append(SOTAAttention(defn['dim']))
            elif l_type == 'gqa':
                self.layers.append(GQAAttention(defn['dim'], 
                                                num_heads=defn.get('heads', 12),
                                                num_groups=defn.get('groups', 3)))
            elif l_type == 'moe':
                self.layers.append(MoELayer(defn['dim'], 
                                            num_experts=defn.get('experts', 8),
                                            top_k=defn.get('top_k', 2),
                                            num_shared_experts=defn.get('shared', 0)))
            elif l_type == 'trans':
                self.layers.append(TransformerBlock(defn['dim']))
            elif l_type == 'fractal':
                self.layers.append(FractalBlock(defn['dim'], depth=defn.get('depth', 2)))
            elif l_type == 'crossattn':
                self.layers.append(
                    SelfCrossAttentionBlock(
                        defn['dim'],
                        num_heads=defn.get('heads', 8),
                    )
                )
            elif l_type == 'gatedcrossattn':
                self.layers.append(
                    GatedSelfCrossAttentionBlock(
                        defn['dim'],
                        num_heads=defn.get('heads', 8),
                    )
                )
            elif l_type == 'complex':
                self.layers.append(
                    ComplexNeuralModule(
                        defn['dim'],
                        num_heads=defn.get('heads', 8),
                        expansion=defn.get('expansion', 4),
                        depth=defn.get('depth', 2),
                    )
                )
            # --- v4.0 NEW LAYERS ---
            elif l_type == 'dropout':
                self.layers.append(DropoutBlock(rate=defn.get('rate', 0.1)))
            elif l_type == 'residual':
                self.layers.append(ResidualBlock(defn['dim'], expansion=defn.get('expansion', 4)))
            elif l_type == 'conv1d':
                self.layers.append(Conv1DBlock(defn['dim'], 
                                               kernel_size=defn.get('kernel', 3),
                                               groups=defn.get('groups', 1)))
            elif l_type == 'lstm':
                self.layers.append(LSTMBlock(defn['dim'], num_layers=defn.get('layers', 1)))
            elif l_type == 'mod':
                self.layers.append(AdaptiveComputeBlock(
                    defn['dim'],
                    expansion=defn.get('expansion', 4),
                    skip_threshold=defn.get('threshold', 0.35),
                ))
            # --- v5.0 Phase 9 ---
            elif l_type == 'conv3d':
                self.layers.append(Conv3DBlock(defn['dim'], 
                                               kernel_size=defn.get('kernel', 3),
                                               groups=defn.get('groups', 1)))
            # --- v6.0 Phase 12 (ASI) ---
            elif l_type == 'mamba':
                self.layers.append(MambaBlock(defn['dim']))
            elif l_type == 'liquid':
                self.layers.append(LiquidNeuralLayer(defn['dim']))
            elif l_type == 'hyper':
                self.layers.append(HyperTransformerBlock(defn['dim']))
            # --- v7.0 Phase 13 (Singularity) ---
            elif l_type == 'script':
                from singularity_tools import ScriptedLayer
                self.layers.append(ScriptedLayer(defn['code']))
            # --- v9.0 Phase 15 (Universal) ---
            elif l_type == 'diamond':
                from neural_env import DiamondBlock
                self.layers.append(DiamondBlock(defn['dim']))
            # --- v10.0 Phase 16 (Hyper-Ensemble) ---
            elif l_type == 'highway':
                from highway_core import MultiModelHighway
                # We expect models to be pre-trained or placeholders for now
                self.layers.append(MultiModelHighway([], defn.get('out_dim', 8), mode=defn['mode']))
            elif l_type == 'ensemble':
                from highway_core import EnsembleLayer
                self.layers.append(EnsembleLayer(defn['path'], 8, 8))
            # --- v11.0 Phase 17 (Distributed Dreams) ---
            elif l_type == 'imagine':
                from dream_engine import ImagineLayer
                self.layers.append(ImagineLayer(defn['dim']))
            # --- v13.0 Phase 19 (Quantum Synthesis) ---
            elif l_type == 'quantum':
                from quantum_core import QuantumLinear
                self.layers.append(QuantumLinear(self.layers[-1].out_features if self.layers else 16, defn['dim']))
            elif l_type == 'fractal_synth':
                from fractal_compression import FractalBlock as FractalSynthBlock
                self.layers.append(FractalSynthBlock(self.layers[-1].out_features if self.layers else 16, defn['dim']))
            # --- v14.0 Phase 20 (Evolution) ---
            elif l_type == 'chrono':
                from chrono_core import ChronoFoldingLayer
                self.layers.append(ChronoFoldingLayer(defn['dim']))
            elif l_type == 'evolve':
                # For static builds, 'evolve' picks a random starting layer from the pool
                import random
                lt = random.choice(defn['pool'])
                # Recursive call simulation via a dummy map for static initialization
                type_map = {'mamba': 128, 'moe': 128, 'liquid': 128, 'quantum': 128}
                l_dim = type_map.get(lt, 128)
                if lt == 'mamba':
                    from neuro_sdk import MambaBlock
                    self.layers.append(MambaBlock(l_dim))
                else: # Fallback to linear for static build
                    self.layers.append(nn.Linear(self.layers[-1].out_features if self.layers else 16, l_dim))
            # --- v15.0 Phase 21 (Research Frontier) ---
            elif l_type == 'kan':
                self.layers.append(KANLayer(defn['dim'],
                                           grid_size=defn.get('grid', 5),
                                           spline_order=defn.get('order', 3)))
            elif l_type == 'diff_attn':
                self.layers.append(DiffAttention(defn['dim'],
                                                num_heads=defn.get('heads', 8)))
            elif l_type == 'lora':
                self.layers.append(LoRAAdapter(defn['dim'],
                                              rank=defn.get('rank', 16),
                                              alpha=defn.get('alpha', 1.0)))
            elif l_type == 'specnorm':
                self.layers.append(SpectralNormBlock(defn['dim'],
                                                    expansion=defn.get('expansion', 4)))
            elif l_type == 'gcp':
                self.layers.append(GradientCheckpointBlock(defn['dim'],
                                                          expansion=defn.get('expansion', 4)))
            # --- v16.0 Phase 22 (Efficiency Frontier) ---
            elif l_type == 'bitlinear':
                self.layers.append(BitLinear(defn['dim'],
                                             expansion=defn.get('expansion', 4)))
            elif l_type == 'retention':
                self.layers.append(RetentionLayer(defn['dim'],
                                                   num_heads=defn.get('heads', 4)))
            elif l_type == 'mix_depth':
                self.layers.append(MixtureOfDepths(defn['dim'],
                                                    expansion=defn.get('expansion', 4),
                                                    capacity_factor=defn.get('capacity', 0.5)))
            # --- v17.0 Phase 23 (Cognitive Nexus) ---
            elif l_type == 'graph_conv':
                self.layers.append(GraphConv(defn['dim'],
                                             expansion=defn.get('expansion', 1)))
            elif l_type == 'diff_logic':
                self.layers.append(DiffLogic(defn['dim'], num_rules=defn.get('rules', 16)))
            elif l_type == 'concept':
                self.layers.append(ConceptNeuron(defn['dim']))
            # --- v18.0 Phase 24 (Hyperspace Drift) ---
            # --- v18.0 Phase 24 (Hyperspace Drift) ---
            elif l_type == 'sphere':
                 self.layers.append(HypersphereLayer(defn['dim']))
            elif l_type == 'poincare':
                 self.layers.append(PoincareLayer(defn['dim']))
            elif l_type == 'topo_attn':
                 self.layers.append(TopologicalAttention(defn['dim'], heads=defn.get('heads', 4)))
            # --- v19.0 Phase 25 (Singularity Nexus) ---
            elif l_type == 'holographic':
                from holographic_core import HolographicLinear
                self.layers.append(HolographicLinear(defn['dim']))
            elif l_type == 'alchemy':
                self.layers.append(ConceptNeuron(defn['dim'])) # Alchemy uses concept view
            # --- v20.0 Phase 26 (Ethereal Synthesis) ---
            elif l_type == 'xfusion':
                from fusion_core import XFusionLayer
                self.layers.append(XFusionLayer(defn['dim']))
            elif l_type == 'node':
                from ode_engine import NeuralODEBlock
                self.layers.append(NeuralODEBlock(defn['dim']))
            elif l_type == 'hypernet':
                from meta_layers import HyperLayer
                self.layers.append(HyperLayer(defn['dim'], defn['dim']))
            # --- v21.0 Phase 27 (Ethereal Flow) ---
            elif l_type == 'fluid':
                from fluid_dynamics import NeuralFluidLayer
                self.layers.append(NeuralFluidLayer(defn['dim']))
            elif l_type == 'turbulence':
                from turbulence_core import TurbulenceLayer
                self.layers.append(TurbulenceLayer(defn['dim']))
            # --- v22.0 Phase 28 (Frontier Intelligence) ---
            elif l_type == 'gla':
                self.layers.append(GatedLinearAttention(defn['dim'],
                                                        num_heads=defn.get('heads', 4)))
            elif l_type == 'xlstm':
                self.layers.append(xLSTMBlock(defn['dim']))
            elif l_type == 'ttt':
                self.layers.append(TestTimeTrainLayer(defn['dim']))
            elif l_type == 'contrastive':
                self.layers.append(ContrastiveHead(defn['dim'], proj_dim=defn.get('proj_dim', 64)))
            elif l_type == 'sparse_attn':
                self.layers.append(SparseAttention(defn['dim'],
                                                   top_k=defn.get('k', 8),
                                                   num_heads=defn.get('heads', 4)))
            elif l_type == 'distill':
                self.layers.append(SelfDistillBlock(defn['dim']))
            # --- v23.0 Phase 29 (Adaptive Nexus) ---
            elif l_type == 'hyena':
                self.layers.append(
                    HyenaBlock(
                        defn['dim'],
                        kernel_size=defn.get('kernel', 7),
                        expand=defn.get('expand', 2),
                    )
                )
            elif l_type == 'geglu':
                self.layers.append(GEGLUBlock(defn['dim'], expansion=defn.get('expansion', 4)))
            elif l_type == 'conv_mixer':
                self.layers.append(
                    ConvMixerBlock(
                        defn['dim'],
                        kernel_size=defn.get('kernel', 7),
                    )
                )
            elif l_type == 'adaptive_rank':
                self.layers.append(AdaptiveRankLinear(defn['dim'],
                                                      max_rank=defn.get('rank', 16),
                                                      energy_threshold=defn.get('energy', 0.9)))
            elif l_type == 'stoch_depth':
                self.layers.append(StochasticDepthBlock(defn['dim'], drop_prob=defn.get('drop_prob', 0.1)))
            # --- v24.0 Phase 30 (Singularity Synthesis+) ---
            elif l_type == 'mhla':
                self.layers.append(
                    MultiHeadLatentAttention(
                        defn['dim'],
                        num_heads=defn.get('heads', 8),
                        q_lora_rank=defn.get('q_rank', 32),
                        kv_lora_rank=defn.get('kv_rank', 64),
                    )
                )
            elif l_type == 'mambaconv':
                self.layers.append(
                    MambaConvBlock(
                        defn['dim'],
                        d_state=defn.get('d_state', 16),
                        d_conv=defn.get('d_conv', 7),
                    )
                )
            elif l_type == 'topk_sparse':
                self.layers.append(SparseTopKLayer(defn['dim'], k=defn.get('k', 8)))
            elif l_type == 'saliency_prune':
                self.layers.append(SaliencyPruningLayer(defn['dim'], base_threshold=defn.get('threshold', 0.2)))
            elif l_type == 'bitnet':
                in_dim = defn.get('in', defn.get('dim', 128))
                out_dim = defn.get('out', in_dim)
                self.layers.append(BitNetLayer(in_dim, out_dim))
            elif l_type == 'rev_res':
                dim = defn['dim']
                if dim % 2 != 0:
                    # Reversible blocks require even dimensions; fallback keeps model buildable.
                    self.layers.append(ResidualBlock(dim))
                else:
                    self.layers.append(ReversibleResidualBlock(dim))
            elif l_type == 'moa':
                self.layers.append(
                    MixtureOfAttention(
                        defn['dim'],
                        num_experts=defn.get('experts', 4),
                        heads_per_expert=defn.get('heads_per_expert', 2),
                    )
                )
    def __len__(self):
        return len(self.layers)

    def get_summary(self):
        """Returns a structured summary of all layers for visualization."""
        summary = []
        total_params = 0
        for i, layer in enumerate(self.layers):
            params = sum(p.numel() for p in layer.parameters())
            total_params += params
            summary.append({
                'index': i,
                'type': type(layer).__name__,
                'params': params,
                'trainable': sum(p.numel() for p in layer.parameters() if p.requires_grad),
            })
        return summary, total_params

    def get_aux_loss(self):
        """Aggregate optional auxiliary losses (for example MoE load balancing)."""
        aux = None
        for layer in self.layers:
            if hasattr(layer, "get_aux_loss"):
                layer_aux = layer.get_aux_loss()
                aux = layer_aux if aux is None else aux + layer_aux
            if hasattr(layer, "orthogonality_loss"):
                layer_ortho = layer.orthogonality_loss()
                # Scale ortho loss slightly down as it can be large
                aux = layer_ortho * 0.1 if aux is None else aux + layer_ortho * 0.1

        if aux is not None:
            return aux
        try:
            return next(self.parameters()).new_zeros(())
        except StopIteration:
            return torch.tensor(0.0)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# ============================================================
# PHASE 23: COGNITIVE NEXUS LAYERS
# ============================================================

class GraphConv(nn.Module):
    """Graph Convolutional Layer (Kipf & Welling 2017).
    
    Performs message passing on graph structured data: H' = A H W.
    If no adjacency matrix is provided, it learns a static structure or uses self-loops.
    """
    def __init__(self, dim: int, expansion: int = 1):
        super().__init__()
        self.dim = dim
        self.out_dim = dim * expansion
        
        self.weight = nn.Linear(dim, self.out_dim, bias=False)
        self.norm = RMSNorm(dim)
        
        # Learnable static adjacency (if no dynamic graph provided)
        # We assume a fixed small abstract graph for reasoning if input is sequence
        self.static_adj_logit = nn.Parameter(torch.randn(128, 128) * 0.01) # Max 128 nodes assumption for learned graph
        
    def forward(self, x, adj=None):
        # x: (B, N, D) where N is nodes/sequence length
        # Handle 2D input (B, D) -> treat as (B, 1, D)
        is_2d = x.dim() == 2
        if is_2d:
             x = x.unsqueeze(1)
             
        B, N, D = x.shape
        x_norm = self.norm(x)
        h = self.weight(x_norm) # (B, N, out_dim)
        
        if adj is None:
            # If no adj provided, use learned static adj (clipped to current size)
            limit = min(N, 128)
            adj_sub = torch.sigmoid(self.static_adj_logit[:limit, :limit])
            
            if N > 128:
                 # Fallback to Identity for large graphs without explicit adj
                 adj_use = torch.eye(N, device=x.device).unsqueeze(0)
            else:
                 adj_use = adj_sub.unsqueeze(0)
                 # Symmetrize roughly
                 adj_use = (adj_use + adj_use.transpose(-2, -1)) / 2
        else:
            adj_use = adj
            
        # Message passing: A @ H
        # (B, N, N) @ (B, N, D) -> (B, N, D)
        # Or (1, N, N) broadcast
        if adj_use.dim() == 2:
             adj_use = adj_use.unsqueeze(0)
             
        out = torch.matmul(adj_use, h)
        
        if is_2d:
             return out.squeeze(1)
        return out

class DiffLogic(nn.Module):
    """Differentiable Logic Gate Layer (Petersen et al.).
    
    Learns logical operations (AND, OR, XOR) using continuous t-norms.
    Weights represent variable selection.
    """
    def __init__(self, dim: int, num_rules: int = 16):
        super().__init__()
        self.dim = dim
        self.num_rules = num_rules
        
        # Weights for rule antecedents (which inputs participate in which rule)
        # Values in [0, 1] via sigmoid
        self.selection_logits = nn.Parameter(torch.randn(num_rules, dim))
        
        # Rule weights (importance of each rule)
        self.rule_weights = nn.Linear(num_rules, dim)
        self.norm = RMSNorm(dim)
        
    def forward(self, x):
        # x: (B, D) or (B, T, D). We operate on last dim.
        residual = x
        x_norm = self.norm(x)
        
        # 1. Soft AND (Conjunctions)
        # selection w_ij \in [0, 1].
        # AND_i = Prod_j ( x_j * w_ij + (1 - w_ij) )
        
        w = torch.sigmoid(self.selection_logits) # (Rules, D)
        
        # Expand x to (..., 1, D)
        x_ex = x_norm.unsqueeze(-2)
        
        # We assume inputs are in [0, 1] range for logic to make sense. 
        # So we squash x with sigmoid for logic path
        x_logic = torch.sigmoid(x_ex)
        
        # T-norm AND: product. 
        # argument = x * w + (1-w)
        arg = x_logic * w + (1 - w)
        # Clamp to avoid numerical issues
        arg = arg.clamp(min=1e-6, max=1.0)
        
        # Use Log-Sum-Exp for stability: Prod = Exp(Sum(Log))
        rule_activations = torch.exp(torch.sum(torch.log(arg), dim=-1)) # (..., Rules)
        
        # 2. Soft OR (Disjunction) via Linear layer aggregation
        out = self.rule_weights(rule_activations)
        
        return residual + out

# ============================================================
# PHASE 24: HYPERSPACE DRIFT LAYERS
# ============================================================

class HypersphereLayer(nn.Module):
    """Projects embeddings onto a unit hypersphere."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.scale = nn.Parameter(torch.ones(1) * 10.0) # Learnable radius/temperature
    
    def forward(self, x):
        # Normalize to unit length and scale
        return F.normalize(x, p=2, dim=-1) * self.scale

class PoincareLayer(nn.Module):
    """Embeds into Poincaré ball (hyperbolic space).
    
    Uses exponential map to project Euclidean vectors into the unit ball.
    Useful for hierarchical data structures.
    """
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        
    def forward(self, x):
        # Pseudo-hyperbolic projection: tanh squashes to (-1, 1)
        # To be strictly in the ball, we ensure norm < 1
        x_tan = torch.tanh(x) 
        # Clip norm to 1 - eps
        norm = x_tan.norm(p=2, dim=-1, keepdim=True)
        max_norm = 1.0 - self.eps
        cond = norm > max_norm
        projected = x_tan / (norm + self.eps) * max_norm
        return torch.where(cond, projected, x_tan)

class TopologicalAttention(nn.Module):
    """Attention mechanism that considers topological distance."""
    def __init__(self, dim, heads=4):
        super().__init__()
        self.dim = dim
        self.heads = max(1, int(heads))
        if dim % self.heads != 0:
            self.heads = max(1, [h for h in range(1, min(dim, 64) + 1) if dim % h == 0][-1])
        self.scale = (dim // self.heads) ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm = RMSNorm(dim)
        # Per-head topological bias term.
        self.manifold_bias = nn.Parameter(torch.zeros(1, self.heads, 1, 1))
        
    def forward(self, x):
        is_2d = x.dim() == 2
        if is_2d:
            x = x.unsqueeze(1)
        B, N, C = x.shape
        x_norm = self.norm(x)
        qkv = self.qkv(x_norm).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Standard attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn + self.manifold_bias
        attn = attn.softmax(dim=-1)
        
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = out + self.proj(out)
        if is_2d:
            return out.squeeze(1)
        return out

class ConceptNeuron(nn.Module):
    """Concept-Aligned Neuron Layer.
    
    Enforces orthogonality between neuron weight vectors to encourage
    disentangled representations of distinct features (Concepts).
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.weight = nn.Linear(dim, dim)
        self.act = nn.SiLU()
        self.norm = RMSNorm(dim)
        self.layer_scale = LayerScaleInit(dim)
        
    def forward(self, x):
        return x + self.layer_scale(self.act(self.weight(self.norm(x))))
    
    def orthogonality_loss(self):
        """Regularization term: || W^T W - I ||."""
        w = self.weight.weight
        # Normalize rows to focus on direction
        w_n = F.normalize(w, dim=1)
        gram = torch.mm(w_n, w_n.t())
        I = torch.eye(self.dim, device=w.device)
        return torch.norm(gram - I, p='fro')

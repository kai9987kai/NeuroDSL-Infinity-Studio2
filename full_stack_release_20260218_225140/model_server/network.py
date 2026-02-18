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
        self.norm = RMSNorm(dim)
        self.sub_blocks = nn.ModuleList([
            SwiGLU(dim, dim * 2) for _ in range(depth)
        ])
        self.stochastic_depth = StochasticDepth(0.05 if depth > 1 else 0.0)
        self.layer_scale = LayerScaleInit(dim)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        for block in self.sub_blocks:
            x = x + self.stochastic_depth(block(x))
        return self.layer_scale(x) + residual

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

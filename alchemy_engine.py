import torch
import torch.nn as nn
import numpy as np

class SymbolicDistiller:
    """
    Extracts algebraic patterns from neural weights.
    Attempts to map weight matrices to symbolic expressions.
    """
    def __init__(self, model):
        self.model = model

    def distill_layer(self, layer_name):
        """
        Extracts symbolic summary of a layer.
        """
        layer = dict(self.model.named_modules()).get(layer_name)
        if not layer:
            return f"Layer {layer_name} not found."
            
        if isinstance(layer, nn.Linear):
            return self._distill_linear(layer)
        elif "Holographic" in layer.__class__.__name__:
            return self._distill_holographic(layer)
        else:
            return f"Symbolic distillation not supported for {type(layer)}"

    def _distill_linear(self, layer):
        W = layer.weight.detach().cpu().numpy()
        mean_w = np.mean(W)
        std_w = np.std(W)
        
        # Simple symbolic heuristic: identifying dominant frequencies or sparse rules
        if std_w < 0.01:
            return f"f(x) ≈ {mean_w:.4f} * Σx"
            
        # Check for identity-like
        if W.shape[0] == W.shape[1]:
            eye_diff = np.abs(W - np.eye(W.shape[0])).mean()
            if eye_diff < 0.1:
                return "f(x) ≈ Identity + Noise"
        
        return f"f(x) ≈ Σ(w_i * x_i) [μ={mean_w:.2e}, σ={std_w:.2e}]"

    def _distill_holographic(self, layer):
        wr = layer.weight_real.detach().cpu().numpy()
        wi = layer.weight_imag.detach().cpu().numpy()
        
        # Extract periodic components via FFT of the weights themselves
        # (This is the 'Alchemy' part - finding hidden symmetries)
        combined = wr + 1j * wi
        spectrum = np.abs(np.fft.fft(combined))
        top_k = np.argsort(spectrum)[-3:][::-1]
        
        expr = "f(x) ≈ IFFT(FFT(x) ⊙ W_freq)"
        harmonics = ", ".join([f"k={k}" for k in top_k])
        return f"{expr} | Dominant Harmonics: {harmonics}"

def extract_all_knowledge(model):
    distiller = SymbolicDistiller(model)
    report = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.LayerNorm)) or "Holographic" in module.__class__.__name__:
            if "." in name: # Only children
                symbolic = distiller.distill_layer(name)
                report.append(f"{name}: {symbolic}")
    return "\n".join(report)

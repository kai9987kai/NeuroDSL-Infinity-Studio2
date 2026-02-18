import torch
import torch.nn as nn

class RepairHook:
    """
    PyTorch hook for autonomous neuron healing.
    Detects dead neurons and applies noise-injection or neighbor-bootstrapping.
    """
    def __init__(self, layer_name):
        self.layer_name = layer_name
        self.repair_log = []

    def __call__(self, module, input, output):
        # Detect dead neurons (zero variance or all zero activations)
        # output: [batch, features]
        with torch.no_grad():
            var = torch.var(output, dim=0)
            dead_mask = var < 1e-6
            
            num_dead = torch.sum(dead_mask).item()
            if num_dead > 0:
                # Injection of "Synthetic Vitality" (Noise)
                vitality = torch.randn_like(output[:, dead_mask]) * 0.1
                output[:, dead_mask] += vitality
                
                msg = f"L-Repair[{self.layer_name}]: Resurrected {num_dead} dead neurons."
                self.repair_log.append(msg)
                
        return output

def attach_self_repair(model):
    """Recursively attaches repair hooks to all linear-like layers."""
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv3d)):
            hook = RepairHook(name)
            module.register_forward_hook(hook)
            hooks.append(hook)
    return hooks

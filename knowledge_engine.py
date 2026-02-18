import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RuleInjector:
    """Injects logical rules into the training process via loss constraints.
    
    Enforces rules of the form "IF antecedent THEN consequent" by penalizing
    violations (Soft Logic).
    """
    
    def __init__(self):
        self.rules = []
        
    def add_rule(self, antecedent_idx, consequent_idx, weight=1.0):
        """Adds an implication rule: inputs[antecedent_idx] -> outputs[consequent_idx]."""
        self.rules.append({
            'type': 'implication',
            'if': antecedent_idx,
            'then': consequent_idx,
            'weight': weight
        })
        
    def parse_text_rule(self, text):
        """Simple parser for rules like 'IF 0 THEN 1'."""
        try:
            parts = text.upper().split('THEN')
            if_part = parts[0].replace('IF', '').strip()
            then_part = parts[1].strip()
            
            # Extract indices (assuming '0', '1', etc.)
            idx_if = int(if_part)
            idx_then = int(then_part)
            self.add_rule(idx_if, idx_then)
            return True
        except Exception as e:
            print(f"Rule Parse Error: {e}")
            return False

    def compute_loss(self, inputs, outputs):
        """Computes rule violation loss.
        
        Logic: A -> B is equivalent to (NOT A) OR B.
        Violation is when A is True (1) and B is False (0).
        Loss = ReLU(A - B).
        """
        if not self.rules:
            return torch.tensor(0.0).to(inputs.device)
            
        loss = torch.tensor(0.0).to(inputs.device)
        
        # Ensure inputs/outputs are in [0, 1]
        # We assume they are probabilities or can be treated as such
        a_in = inputs
        b_out = outputs
        
        # If inputs are not [0,1], apply sigmoid? 
        # Usually we apply this loss on probabilities.
        
        for rule in self.rules:
            if rule['type'] == 'implication':
                # Get A and B
                # We handle if indices are lists (conjunction inputs) later
                # For now single index
                
                idx_a = rule['if']
                idx_b = rule['then']
                
                if idx_a < inputs.shape[1] and idx_b < outputs.shape[1]:
                    val_a = inputs[:, idx_a]
                    val_b = outputs[:, idx_b]
                    
                    # Implication loss: A > B is bad.
                    # Godel t-norm implication: 1 if A <= B else B
                    # Lukasiewicz implication: min(1, 1 - A + B)
                    # Product implication: 1 if A <= B else B/A
                    
                    # We use simple ReLU(A - B) which creates gradient to push A down or B up
                    rule_loss = F.relu(val_a - val_b).mean()
                    loss += rule_loss * rule['weight']
                    
        return loss

class ReasoningTracer:
    """Visualizes the 'thought path' through DiffLogic layers."""
    
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.activations = {}
        
    def register(self):
        """Attach hooks to DiffLogic layers."""
        self.remove()
        
        def get_hook(name):
            def hook(model, input, output):
                # Copy output to CPU for visualization
                self.activations[name] = output.detach().cpu()
            return hook
            
        for name, module in self.model.named_modules():
            if 'DiffLogic' in module.__class__.__name__:
                self.hooks.append(module.register_forward_hook(get_hook(name)))
                
    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
        self.activations = {}
        
    def get_trace(self):
        """Return the collected activations."""
        return self.activations

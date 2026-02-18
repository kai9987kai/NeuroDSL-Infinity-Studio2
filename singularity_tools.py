import torch
import torch.nn as nn
import copy

class WeightMorpher:
    """Utilities for interpolating between model checkpoints."""
    
    @staticmethod
    def interpolate_models(model_a, model_b, alpha=0.5):
        """
        Linearly interpolates between the weights of two models.
        theta_new = (1-alpha)*theta_a + alpha*theta_b
        """
        # Ensure models have the same architecture
        new_model = copy.deepcopy(model_a)
        params_a = model_a.state_dict()
        params_b = model_b.state_dict()
        new_params = new_model.state_dict()
        
        for name in params_a:
            if name in params_b:
                new_params[name] = (1.0 - alpha) * params_a[name] + alpha * params_b[name]
        
        new_model.load_state_dict(new_params)
        return new_model

class NeuralSandbox:
    """A 'safe' sandbox for executing dynamic neural layer logic."""
    
    def __init__(self):
        self.namespace = {
            "torch": torch,
            "nn": nn,
            "F": torch.nn.functional
        }
        
    def execute_logic(self, code_str, x):
        """Executes custom logic on a tensor x based on provided code."""
        # Note: In a production environment, this should be MUCH more restrictive.
        # This is for high-level research simulation.
        local_ns = {"x": x, **self.namespace}
        try:
            exec(f"output = {code_str}", self.namespace, local_ns)
            return local_ns["output"]
        except Exception as e:
            print(f"Neural Sandbox Error: {e}")
            return x

class ScriptedLayer(nn.Module):
    """A layer that executes custom Python logic defined in the DSL."""
    def __init__(self, code_str):
        super().__init__()
        self.code_str = code_str
        self.sandbox = NeuralSandbox()
        
    def forward(self, x):
        return self.sandbox.execute_logic(self.code_str, x)

class AdversarialToolbox:
    """Toolbox for generating adversarial samples (FGSM, PGD)."""
    
    @staticmethod
    def fgsm_attack(model, loss_fn, images, labels, epsilon):
        """Fast Gradient Sign Method (FGSM) attack."""
        images.requires_grad = True
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        model.zero_grad()
        loss.backward()
        
        data_grad = images.grad.data
        perturbed_image = images + epsilon * data_grad.sign()
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        return perturbed_image

class NeuralSynthesizer:
    """Auto-generates data tailored to the architecture's dimensions."""
    
    @staticmethod
    def synthesize_dataset(model, n_samples=1000, noise=0.1):
        """Synthesizes a random dataset based on the model's input dimension."""
        # Try to find input dimension from first layer
        first_layer = None
        for layer in model.modules():
            if isinstance(layer, nn.Linear):
                first_layer = layer
                break
        
        if first_layer:
            in_dim = first_layer.in_features
            x = torch.randn(n_samples, in_dim)
            # Create dummy labels based on model prediction + noise
            with torch.no_grad():
                y = model(x)
                y += torch.randn_like(y) * noise
            return x, y
        return None, None

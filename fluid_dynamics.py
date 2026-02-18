import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralFluidLayer(nn.Module):
    """
    Treats activations as a compressible fluid.
    Updates states using a simplified Navier-Stokes advection-diffusion step.
    x_next = x + dt * ( - (v . grad)x + nu * laplacian(x) )
    In latent space, we approximate this using learned kernels.
    """
    def __init__(self, dim, viscosity=0.1, dt=0.2):
        super().__init__()
        self.dim = dim
        self.viscosity = viscosity
        self.dt = dt
        
        # Learned "Velocity" generator (predicts flow vectors from state)
        self.velocity_gen = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, dim)
        )
        
        # Laplacian approximation (learned diffusion kernel)
        self.diffusion_kernel = nn.Parameter(torch.ones(dim) * viscosity)
        
    def forward(self, x):
        """
        x: [B, D]
        """
        # 1. Advection: Predicting how activations "move"
        # Since we are not on a grid, we use a self-interaction term as a proxy for advection
        velocity = self.velocity_gen(x)
        advection = - (velocity * x) # Simplified directional advection
        
        # 2. Diffusion: Spreading of activations
        # Approximated by damped decay toward mean or learned diffusion
        diffusion = self.diffusion_kernel * (torch.mean(x, dim=0, keepdim=True) - x)
        
        # 3. Update Step
        x_next = x + self.dt * (advection + diffusion)
        
        return x_next

import torch
import torch.nn as nn

class ODEFunc(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, dim)
        )
        
    def forward(self, t, x):
        return self.net(x)

class NeuralODEBlock(nn.Module):
    """
    Continuous-time neural layer using ODE solvers.
    Represents the transformation as x(T) = x(0) + integral_{0}^{T} f(x(t), t) dt
    """
    def __init__(self, dim, solver='euler', steps=4, t_end=1.0):
        super().__init__()
        self.dim = dim
        self.f = ODEFunc(dim)
        self.solver = solver
        self.steps = steps
        self.t_end = t_end
        
    def forward(self, x):
        h = self.t_end / self.steps
        t = 0.0
        
        for _ in range(self.steps):
            if self.solver == 'euler':
                x = x + h * self.f(t, x)
            elif self.solver == 'rk4':
                k1 = self.f(t, x)
                k2 = self.f(t + h/2, x + h*k1/2)
                k3 = self.f(t + h/2, x + h*k2/2)
                k4 = self.f(t + h, x + h*k3)
                x = x + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
            t += h
            
        return x

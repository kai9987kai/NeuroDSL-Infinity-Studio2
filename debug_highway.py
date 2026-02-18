import torch
import torch.nn as nn
from highway_core import MultiModelHighway

def debug_highway():
    m1 = nn.Linear(8, 8)
    m1.weight.data = torch.eye(8)
    m1.bias.data.zero_()
    
    m2 = nn.Linear(8, 8)
    m2.weight.data = torch.eye(8) * 2
    m2.bias.data.zero_()
    
    highway = MultiModelHighway([m1, m2], out_dim=8, mode="average")
    x = torch.ones(1, 8)
    out = highway(x)
    
    print(f"Input: {x}")
    print(f"Output: {out}")
    print(f"Expected: {torch.ones(1, 8) * 1.5}")
    print(f"Allclose: {torch.allclose(out, torch.ones(1, 8) * 1.5)}")

if __name__ == "__main__":
    debug_highway()

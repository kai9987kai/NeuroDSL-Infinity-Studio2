import torch
import traceback
from network import ModernMLP, Conv3DBlock

def test_conv3d():
    print("Testing Conv3DBlock...")
    try:
        block = Conv3DBlock(16, kernel_size=3)
        x = torch.randn(2, 16)
        y = block(x)
        print(f"Block forward pass successful. Output shape: {y.shape}")
    except Exception:
        traceback.print_exc()

def test_mlp_integration():
    print("\nTesting ModernMLP with Conv3D...")
    try:
        layer_defs = [
            {'type': 'linear', 'in': 16, 'out': 16},
            {'type': 'conv3d', 'dim': 16, 'kernel': 3},
            {'type': 'linear', 'in': 16, 'out': 10}
        ]
        model = ModernMLP(layer_defs)
        x = torch.randn(2, 16)
        y = model(x)
        print(f"Model forward pass successful. Output shape: {y.shape}")
    except Exception:
        traceback.print_exc()

if __name__ == "__main__":
    test_conv3d()
    test_mlp_integration()

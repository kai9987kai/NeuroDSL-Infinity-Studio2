import torch
import torch.nn as nn
from parser_utils import parse_program, create_modern_nn, validate_dsl
from fusion_core import XFusionLayer
from ode_engine import NeuralODEBlock
from meta_layers import HyperLayer

def test_fusion_layer():
    print("Testing XFusionLayer...")
    dim = 64
    layer = XFusionLayer(dim)
    x = torch.randn(2, 10, dim)
    context = torch.randn(2, 5, dim)
    
    out = layer(x, context)
    assert out.shape == (2, 10, dim), f"Expected (2, 10, 64), got {out.shape}"
    print("[ok] XFusionLayer forward pass successful")

def test_ode_layer():
    print("Testing NeuralODEBlock...")
    dim = 32
    layer = NeuralODEBlock(dim, solver='rk4', steps=10)
    x = torch.randn(4, dim)
    
    out = layer(x)
    assert out.shape == (4, dim), f"Expected (4, 32), got {out.shape}"
    assert not torch.isnan(out).any(), "ODE output contains NaNs"
    print("[ok] NeuralODEBlock (RK4) forward pass successful")

def test_hypernet_layer():
    print("Testing HyperLayer...")
    in_features, out_features = 16, 32
    layer = HyperLayer(in_features, out_features)
    x = torch.randn(8, in_features)
    
    out = layer(x)
    assert out.shape == (8, out_features), f"Expected (8, 32), got {out.shape}"
    print("[ok] HyperLayer forward pass successful")

def test_dsl_integration():
    print("Testing Phase 26 DSL Integration...")
    program = "hypernet: [64], node: [64], xfusion: [64], [64, 10]"
    issues, layer_defs = validate_dsl(program)
    assert layer_defs is not None, "Failed to parse Phase 26 keywords"
    
    model = create_modern_nn(layer_defs)
    layers_found = [type(l).__name__ for l in model.layers]
    assert "HyperLayer" in layers_found, "HyperLayer missing in model"
    assert "NeuralODEBlock" in layers_found, "NeuralODEBlock missing in model"
    assert "XFusionLayer" in layers_found, "XFusionLayer missing in model"
    print("[ok] DSL to Model construction successful for Phase 26")

if __name__ == "__main__":
    try:
        test_fusion_layer()
        test_ode_layer()
        test_hypernet_layer()
        test_dsl_integration()
        print("\nALL PHASE 26 VERIFICATION TESTS PASSED! 🌌")
    except Exception as e:
        print(f"\nTESTS FAILED: {e}")
        import traceback
        traceback.print_exc()

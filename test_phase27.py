import torch
import torch.nn as nn
from parser_utils import create_modern_nn, validate_dsl
from fluid_dynamics import NeuralFluidLayer
from turbulence_core import TurbulenceLayer

def test_fluid_layer():
    print("Testing NeuralFluidLayer...")
    dim = 64
    layer = NeuralFluidLayer(dim)
    x = torch.randn(4, dim)
    
    out = layer(x)
    assert out.shape == (4, dim), f"Expected (4, 64), got {out.shape}"
    assert not torch.isnan(out).any(), "Fluid output contains NaNs"
    print("[ok] NeuralFluidLayer forward pass successful")

def test_turbulence_layer():
    print("Testing TurbulenceLayer...")
    dim = 32
    layer = TurbulenceLayer(dim, intensity=0.1)
    x = torch.ones(2, dim) # Use ones to easily see jitter
    
    layer.train()
    out_train = layer(x)
    assert not torch.allclose(x, out_train), "Turbulence should jitter input in train mode"
    
    layer.eval()
    out_eval = layer(x)
    assert torch.allclose(x, out_eval), "Turbulence should be identity in eval mode"
    print("[ok] TurbulenceLayer (Train/Eval) successful")

def test_dsl_integration_phase27():
    print("Testing Phase 27 DSL Integration...")
    program = "turbulence: [64], fluid: [64], [64, 10]"
    issues, layer_defs = validate_dsl(program)
    assert layer_defs is not None, "Failed to parse Phase 27 keywords"
    
    model = create_modern_nn(layer_defs)
    layers_found = [type(l).__name__ for l in model.layers]
    assert "TurbulenceLayer" in layers_found, "TurbulenceLayer missing in model"
    assert "NeuralFluidLayer" in layers_found, "NeuralFluidLayer missing in model"
    print("[ok] DSL to Model construction successful for Phase 27")

if __name__ == "__main__":
    try:
        test_fluid_layer()
        test_turbulence_layer()
        test_dsl_integration_phase27()
        print("\nALL PHASE 27 VERIFICATION TESTS PASSED! ")
    except Exception as e:
        print(f"\nTESTS FAILED: {e}")
        import traceback
        traceback.print_exc()


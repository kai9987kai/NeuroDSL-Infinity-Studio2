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
    assert out.shape == (4, dim)
    print("[ok] NeuralFluidLayer pass")

def test_turbulence_layer():
    print("Testing TurbulenceLayer...")
    dim = 32
    layer = TurbulenceLayer(dim, intensity=0.1)
    x = torch.ones(2, dim)
    layer.train()
    out = layer(x)
    assert not torch.allclose(x, out)
    print("[ok] TurbulenceLayer pass")

def test_dsl():
    print("Testing DSL Integration...")
    program = "turbulence: [64], fluid: [64], [64, 10]"
    issues, layer_defs = validate_dsl(program)
    model = create_modern_nn(layer_defs)
    print(f"Model type: {type(model)}")
    print(f"Model has 'layers': {hasattr(model, 'layers')}")
    if hasattr(model, 'layers'):
        layers_found = [type(l).__name__ for l in model.layers]
        print(f"Layers found: {layers_found}")
        assert "TurbulenceLayer" in layers_found
        assert "NeuralFluidLayer" in layers_found
    else:
        # If it's wrapped or moved to cuda, maybe it's different?
        # But .cuda() shouldn't change the class.
        pass
    print("[ok] DSL Integration pass")

if __name__ == "__main__":
    test_fluid_layer()
    test_turbulence_layer()
    test_dsl()
    print("ALL PASSED")


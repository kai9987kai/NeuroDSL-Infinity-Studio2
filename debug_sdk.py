import traceback
import torch
import torch.nn as nn
from neuro_sdk import NeuroLab
from parser_utils import DSL_PRESETS, parse_program, create_modern_nn

def debug_sdk():
    print("Testing SDK Build...")
    try:
        sdk = NeuroLab()
        dsl = "[16, 16], [16, 10]"
        model = sdk.build(dsl)
        print(f"Build successful. Model: {type(model)}")
        
        print("Testing SDK Train...")
        X = torch.randn(10, 16)
        y = torch.randint(0, 10, (10,))
        loss = sdk.train(X, y, epochs=2)
        print(f"Train successful. Loss: {loss}")
    except Exception:
        traceback.print_exc()

def debug_omni_preset():
    print("\nTesting Omni-Model Preset...")
    try:
        preset_code = DSL_PRESETS.get("Omni-Model (SOTA)")
        if not preset_code:
            raise ValueError("Preset not found!")
        print(f"Preset Code: {preset_code}")
        
        layer_defs = parse_program(preset_code)
        if not layer_defs:
            raise ValueError("Failed to parse Omni-Model preset")
            
        print(f"Parsed Layers: {len(layer_defs)}")
        
        # Build it
        # We need to construct it
        # Note: This might be large so we wrap in try/except for OOM or dim errors
        model = create_modern_nn(layer_defs)
        print(f"Omni-Model Built. Params: {sum(p.numel() for p in model.parameters())}")
        
    except Exception:
        traceback.print_exc()

if __name__ == "__main__":
    debug_sdk()
    debug_omni_preset()

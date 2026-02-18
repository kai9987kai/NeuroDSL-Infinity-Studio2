import torch
import torch.nn as nn
from parser_utils import parse_program, create_modern_nn, validate_dsl
from trainer import SwarmTrainingEngine, TrainingEngine
from alchemy_engine import SymbolicDistiller
from holographic_core import HolographicLinear

def test_holographic_parsing():
    print("Testing Holographic DSL parsing...")
    program = "holographic: [64], [64, 10]"
    issues, layer_defs = validate_dsl(program)
    assert layer_defs is not None, "Failed to parse holographic keyword"
    print("[ok] Holographic keyword parsed successfully")
    
    model = create_modern_nn(layer_defs)
    assert any(isinstance(m, HolographicLinear) for m in model.modules()), "HolographicLinear not found in model"
    print("[ok] Holographic model constructed successfully")

def test_alchemy_parsing():
    print("Testing Alchemy DSL parsing...")
    program = "alchemy: [64], [64, 10]"
    issues, layer_defs = validate_dsl(program)
    assert layer_defs is not None, "Failed to parse alchemy keyword"
    print("[ok] Alchemy keyword parsed successfully")

def test_swarm_opt():
    print("Testing SwarmOptimizer...")
    model = nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 1))
    trainer = SwarmTrainingEngine(model, n_particles=5)
    
    X = torch.randn(10, 8)
    y = torch.randn(10, 1)
    
    initial_loss, _, _, _, _ = trainer.train_step(X, y)
    print(f"Initial Swarm Loss: {initial_loss:.4f}")
    
    for _ in range(5):
        loss, _, _, _, _ = trainer.train_step(X, y)
        
    print(f"Final Swarm Loss (after 5 steps): {loss:.4f}")
    assert loss <= initial_loss, "Swarm optimization did not reduce loss"
    print("[ok] SwarmOptimizer reducing loss successfully")

def test_symbolic_distillation():
    print("Testing Symbolic Distiller...")
    model = nn.Sequential(nn.Linear(10, 10))
    # Set weights to something predictable (identity-ish)
    with torch.no_grad():
        model[0].weight.copy_(torch.eye(10) + torch.randn(10, 10) * 0.01)
    
    distiller = SymbolicDistiller(model)
    report = distiller.generate_alchemy_report()
    assert "Distillation Hub" in report, "Alchemy report missing header"
    print("[ok] Symbolic distillation report generated successfully")

if __name__ == "__main__":
    try:
        test_holographic_parsing()
        test_alchemy_parsing()
        test_swarm_opt()
        test_symbolic_distillation()
        print("\nALL PHASE 25 CORE TESTS PASSED! 🚀")
    except Exception as e:
        print(f"\nTESTS FAILED: {e}")
        import traceback
        traceback.print_exc()

import torch
from network import ModernMLP, RMSNorm
from parser_utils import parse_program, validate_dsl, DSL_PRESETS
from device_utils import detect_devices
from experimental_models import (
    ExperimentalTrainConfig,
    ImageAutoencoder,
    MultiModalFusionModel,
    SyntheticImageDataset,
    generate_interpolation_images,
    text_to_feature_vector,
    train_image_autoencoder,
    train_multimodal,
)


def test_infinity_blocks():
    print("Testing Infinity Blocks (MoE, GQA, Fractal)...")
    layer_defs = [
        {'type': 'linear', 'in': 64, 'out': 128},
        {'type': 'fractal', 'dim': 128, 'depth': 2},
        {'type': 'gqa', 'dim': 128, 'heads': 8, 'groups': 2},
        {'type': 'moe', 'dim': 128, 'experts': 8, 'shared': 1},
        {'type': 'linear', 'in': 128, 'out': 10}
    ]
    model = ModernMLP(layer_defs)
    x = torch.randn(4, 64)
    y = model(x)
    assert y.shape == (4, 10), f"Expected shape (4, 10), got {y.shape}"
    aux = model.get_aux_loss()
    assert torch.is_tensor(aux), "Expected MoE aux loss tensor"
    print("  PASS: Infinity Blocks")

def test_new_v4_layers():
    print("Testing v4.0 Layers (Dropout, Residual, Conv1D, LSTM, MoD)...")
    
    # Test DropoutBlock
    layer_defs = [
        {'type': 'linear', 'in': 32, 'out': 64},
        {'type': 'dropout', 'rate': 0.2},
        {'type': 'linear', 'in': 64, 'out': 10}
    ]
    model = ModernMLP(layer_defs)
    x = torch.randn(4, 32)
    y = model(x)
    assert y.shape == (4, 10), f"Dropout: Expected (4, 10), got {y.shape}"
    print("  PASS: DropoutBlock")
    
    # Test ResidualBlock
    layer_defs = [
        {'type': 'linear', 'in': 32, 'out': 64},
        {'type': 'residual', 'dim': 64, 'expansion': 4},
        {'type': 'linear', 'in': 64, 'out': 10}
    ]
    model = ModernMLP(layer_defs)
    y = model(torch.randn(4, 32))
    assert y.shape == (4, 10), f"Residual: Expected (4, 10), got {y.shape}"
    print("  PASS: ResidualBlock")
    
    # Test Conv1DBlock
    layer_defs = [
        {'type': 'linear', 'in': 32, 'out': 64},
        {'type': 'conv1d', 'dim': 64, 'kernel': 3},
        {'type': 'linear', 'in': 64, 'out': 10}
    ]
    model = ModernMLP(layer_defs)
    y = model(torch.randn(4, 32))
    assert y.shape == (4, 10), f"Conv1D: Expected (4, 10), got {y.shape}"
    print("  PASS: Conv1DBlock")
    
    # Test LSTMBlock
    layer_defs = [
        {'type': 'linear', 'in': 32, 'out': 64},
        {'type': 'lstm', 'dim': 64, 'layers': 1},
        {'type': 'linear', 'in': 64, 'out': 10}
    ]
    model = ModernMLP(layer_defs)
    y = model(torch.randn(4, 32))
    assert y.shape == (4, 10), f"LSTM: Expected (4, 10), got {y.shape}"
    print("  PASS: LSTMBlock")

    # Test AdaptiveComputeBlock (Mixture-of-Depths inspired)
    layer_defs = [
        {'type': 'linear', 'in': 32, 'out': 64},
        {'type': 'mod', 'dim': 64, 'expansion': 4, 'threshold': 0.35},
        {'type': 'linear', 'in': 64, 'out': 10}
    ]
    model = ModernMLP(layer_defs)
    y = model(torch.randn(4, 32))
    assert y.shape == (4, 10), f"MoD: Expected (4, 10), got {y.shape}"
    print("  PASS: AdaptiveComputeBlock")

def test_training_engine():
    print("Testing Training Engine (multi-loss, grad clip, warmup)...")
    from trainer import TrainingEngine
    
    layer_defs = [{'type': 'linear', 'in': 4, 'out': 2}]
    model = ModernMLP(layer_defs)
    
    for loss_fn in ['MSE', 'Huber', 'MAE']:
        trainer = TrainingEngine(model, loss_fn=loss_fn, grad_clip=1.0, warmup_steps=3)
        X, y = trainer.generate_dummy_data(4, 2, n_samples=10)
        loss, lr, grad_norm = trainer.train_step(X, y)
        assert loss > 0, f"{loss_fn}: Loss should be positive"
        assert grad_norm >= 0, f"{loss_fn}: Grad norm should be non-negative"
    print("  PASS: Multi-loss training")
    
    # Test TorchScript export
    import os
    ts_path = "test_model.pt"
    trainer.export_torchscript(ts_path, 4)
    assert os.path.exists(ts_path), "TorchScript export failed"
    os.remove(ts_path)
    print("  PASS: TorchScript export")

def test_advanced_features():
    """Test the new advanced features"""
    print("\nTesting Advanced Features...")
    
    # Test new layer types
    new_layer_types = ['crossattn', 'gatedcrossattn', 'complex']
    
    for layer_type in new_layer_types:
        print(f"  Testing {layer_type} layer...")
        
        # Create a simple DSL with the new layer
        if layer_type == 'crossattn':
            dsl = f"[32, 64], {layer_type}: [64, 8], [64, 10]"
        elif layer_type == 'gatedcrossattn':
            dsl = f"[32, 64], {layer_type}: [64, 4], [64, 10]"
        elif layer_type == 'complex':
            dsl = f"[32, 128], {layer_type}: [128, 8, 4.0, 2], [128, 10]"
        
        # Test parsing
        issues, layer_defs = validate_dsl(dsl)
        if layer_defs is None:
            print(f"    FAIL: Could not parse {layer_type} DSL")
            continue
            
        print(f"    OK: {layer_type} DSL parsed successfully")
        
        # Test model creation
        try:
            model = create_modern_nn(layer_defs, in_dim=32, out_dim=10)
            print(f"    OK: {layer_type} model created successfully")
        except Exception as e:
            print(f"    FAIL: {layer_type} model creation failed: {e}")
            continue
        
        # Test forward pass
        try:
            x = torch.randn(4, 32)  # 4 samples of size 32
            output, aux_loss = model(x)
            expected_shape = (4, 10)
            if output.shape == expected_shape:
                print(f"    OK: {layer_type} forward pass successful")
            else:
                print(f"    FAIL: {layer_type} output shape {output.shape}, expected {expected_shape}")
        except Exception as e:
            print(f"    FAIL: {layer_type} forward pass failed: {e}")
    
    # Test advanced training components
    print("  Testing Advanced Training Components...")
    try:
        from advanced_training import AdvancedTrainingEngine, CurriculumLearning, KnowledgeDistillation
        
        # Create a simple model
        issues, layer_defs = validate_dsl("[32, 64], [64, 10]")
        model = create_modern_nn(layer_defs, in_dim=32, out_dim=10)
        
        # Test AdvancedTrainingEngine initialization
        engine = AdvancedTrainingEngine(
            model=model,
            accumulation_steps=2,
            label_smoothing=0.1,
            use_amp=False  # Disable AMP for CPU testing
        )
        print("    OK: AdvancedTrainingEngine initialized successfully")
        
        # Test CurriculumLearning
        cl = CurriculumLearning(model)
        print("    OK: CurriculumLearning initialized successfully")
        
        print("    All advanced training components tested successfully")
    except Exception as e:
        print(f"    FAIL: Advanced training components test failed: {e}")
    
    # Test visualization components
    print("  Testing Visualization Components...")
    try:
        from viz_utils import ModelVisualizer
        
        # Create a simple model
        issues, layer_defs = validate_dsl("[32, 64], [64, 10]")
        model = create_modern_nn(layer_defs, in_dim=32, out_dim=10)
        
        # Test ModelVisualizer
        visualizer = ModelVisualizer(model)
        print("    OK: ModelVisualizer initialized successfully")
        
        # Generate some sample data
        x = torch.randn(10, 32)
        model.eval()
        with torch.no_grad():
            activations, _ = model(x)
        
        # Test parameter heatmap
        param_fig = visualizer.plot_parameter_heatmap(model)
        print("    OK: Parameter heatmap generated successfully")
        
        # Test gradient flow
        grad_fig = visualizer.plot_gradient_flow(model)
        print("    OK: Gradient flow plot generated successfully")
        
        # Test feature map visualization
        feat_fig = visualizer.plot_feature_maps_3d(activations)
        print("    OK: Feature map visualization generated successfully")
        
        print("    All visualization components tested successfully")
    except Exception as e:
        print(f"    FAIL: Visualization components test failed: {e}")
    
    print("Advanced Features testing complete.")


def test_new_presets():
    """Test the new DSL presets"""
    print("\nTesting New DSL Presets...")
    
    # Get new presets
    presets = load_presets()
    new_presets = [
        "Cross-Attention Module",
        "Gated Cross-Attention", 
        "Complex Neural Module",
        "Multimodal Fusion"
    ]
    
    for preset_name in new_presets:
        if preset_name in presets:
            dsl = presets[preset_name]
            print(f"  Testing preset '{preset_name}'...")
            
            # Validate DSL
            issues, layer_defs = validate_dsl(dsl)
            if layer_defs is None:
                print(f"    FAIL: Invalid DSL for preset '{preset_name}'")
                continue
            
            print(f"    OK: DSL validated for preset '{preset_name}'")
            
            # Try to create model (use a smaller input/output size for testing)
            try:
                model = create_modern_nn(layer_defs, in_dim=32, out_dim=10)
                print(f"    OK: Model created for preset '{preset_name}'")
            except Exception as e:
                print(f"    FAIL: Model creation failed for preset '{preset_name}': {e}")
        else:
            print(f"  SKIP: Preset '{preset_name}' not found")
    
    print("New DSL Presets testing complete.")


def run_all_tests():
    """Run all verification tests"""
    print("Starting verification tests...")
    
    # Run existing tests
    test_parser()
    test_blocks()
    test_training_engine()
    test_exports()
    test_presets()
    test_model_summary()
    
    # Run new tests
    test_new_presets()
    test_advanced_features()
    
    print("\nVerification tests completed!")


if __name__ == "__main__":
    run_all_tests()

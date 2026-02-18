"""
Test script for the new features added to NeuroDSL-Infinity-Studio
"""
import torch
from network import ModernMLP
from parser_utils import parse_program, validate_dsl, DSL_PRESETS, load_presets
from viz_utils import ModelVisualizer
from advanced_training import AdvancedTrainingEngine, CurriculumLearning, KnowledgeDistillation


def test_new_layer_types():
    """Test the new layer types: crossattn, gatedcrossattn, complex"""
    print("Testing New Layer Types...")
    
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
        try:
            issues, layer_defs = validate_dsl(dsl)
            if layer_defs is None:
                print(f"    FAIL: Could not parse {layer_type} DSL")
                continue
            print(f"    OK: {layer_type} DSL parsed successfully")
        except Exception as e:
            print(f"    FAIL: Could not parse {layer_type} DSL: {e}")
            continue
        
        # Test model creation
        try:
            from parser_utils import create_modern_nn
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
    
    print("New Layer Types testing complete.\n")


def test_advanced_training_components():
    """Test the advanced training components"""
    print("Testing Advanced Training Components...")
    
    try:
        # Create a simple model
        from parser_utils import create_modern_nn
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
        
        print("Advanced Training Components tested successfully.\n")
    except Exception as e:
        print(f"FAIL: Advanced training components test failed: {e}\n")


def test_visualization_components():
    """Test the visualization components"""
    print("Testing Visualization Components...")
    
    try:
        # Create a simple model
        from parser_utils import create_modern_nn
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
        
        print("Visualization Components tested successfully.\n")
    except Exception as e:
        print(f"FAIL: Visualization components test failed: {e}\n")


def test_new_presets():
    """Test the new DSL presets"""
    print("Testing New DSL Presets...")
    
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
            try:
                issues, layer_defs = validate_dsl(dsl)
                if layer_defs is None:
                    print(f"    FAIL: Invalid DSL for preset '{preset_name}'")
                    continue
                print(f"    OK: DSL validated for preset '{preset_name}'")
            except Exception as e:
                print(f"    FAIL: Error validating preset '{preset_name}': {e}")
                continue
            
            # Try to create model (use a smaller input/output size for testing)
            try:
                from parser_utils import create_modern_nn
                model = create_modern_nn(layer_defs, in_dim=32, out_dim=10)
                print(f"    OK: Model created for preset '{preset_name}'")
            except Exception as e:
                print(f"    FAIL: Model creation failed for preset '{preset_name}': {e}")
        else:
            print(f"  SKIP: Preset '{preset_name}' not found")
    
    print("New DSL Presets testing complete.\n")


def test_knowledge_distillation():
    """Test knowledge distillation components"""
    print("Testing Knowledge Distillation...")
    
    try:
        # Create two simple models - teacher and student
        from parser_utils import create_modern_nn
        teacher_issues, teacher_defs = validate_dsl("[32, 128], [128, 64], [64, 10]")
        student_issues, student_defs = validate_dsl("[32, 64], [64, 10]")
        
        teacher_model = create_modern_nn(teacher_defs, in_dim=32, out_dim=10)
        student_model = create_modern_nn(student_defs, in_dim=32, out_dim=10)
        
        # Test KnowledgeDistillation initialization
        distiller = KnowledgeDistillation(teacher_model, student_model)
        print("    OK: KnowledgeDistillation initialized successfully")
        
        print("Knowledge Distillation tested successfully.\n")
    except Exception as e:
        print(f"FAIL: Knowledge distillation test failed: {e}\n")


def run_all_tests():
    """Run all tests for the new features"""
    print("Starting tests for new features...\n")
    
    test_new_layer_types()
    test_advanced_training_components()
    test_visualization_components()
    test_new_presets()
    test_knowledge_distillation()
    
    print("All tests completed!")


if __name__ == "__main__":
    run_all_tests()
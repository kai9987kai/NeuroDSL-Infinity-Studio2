"""
Test script for model optimization features in NeuroDSL-Infinity-Studio.
"""

import torch
import torch.nn as nn
from model_optimizer import ModelOptimizer, evaluate_model_performance


def test_quantization():
    """Test model quantization functionality."""
    print("Testing model quantization...")
    
    # Create a simple test model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 20)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(20, 1)
            
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    model = SimpleModel()
    
    # Test dynamic quantization
    optimizer = ModelOptimizer()
    quantized_model = optimizer.quantize_model_dynamic(model)
    
    print(f"[ok] Dynamic quantization completed")
    print(f"  Original model type: {type(model.fc1.weight.dtype)}")
    
    # Check if quantized (this might fail depending on PyTorch version)
    try:
        print(f"  Quantized model type: {type(quantized_model.fc1.weight.dtype)}")
    except:
        print(f"  Quantized model type: Unable to determine (may be due to PyTorch version)")


def test_pruning():
    """Test model pruning functionality."""
    print("\nTesting model pruning...")
    
    # Create a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(8, 16)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(16, 1)
            
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    model = SimpleModel()
    
    # Test pruning
    optimizer = ModelOptimizer()
    pruned_model = optimizer.prune_model(model, sparsity=0.2)
    
    # Check sparsity
    sparsity_dict = optimizer.get_model_sparsity(pruned_model)
    print(f"[ok] Model pruning completed")
    print(f"  Sparsity per layer: {sparsity_dict}")
    
    # Calculate overall sparsity
    total_params = sum(p.numel() for p in pruned_model.parameters())
    zero_params = sum(torch.sum(p == 0).item() for p in pruned_model.parameters())
    overall_sparsity = zero_params / total_params if total_params > 0 else 0
    
    print(f"  Overall sparsity: {overall_sparsity:.2f}")


def test_size_reduction():
    """Test size reduction calculation."""
    print("\nTesting size reduction calculation...")
    
    # Create two models of different sizes
    class SmallModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(5, 1)
            
        def forward(self, x):
            return self.fc(x)
    
    class LargeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(5, 20)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(20, 1)
            
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    small_model = SmallModel()
    large_model = LargeModel()
    
    # Calculate size reduction
    optimizer = ModelOptimizer()
    reduction_metrics = optimizer.get_model_size_reduction(large_model, small_model)
    
    print(f"[ok] Size reduction calculation completed")
    print(f"  Large model size: {reduction_metrics['original_size_mb']:.4f} MB")
    print(f"  Small model size: {reduction_metrics['optimized_size_mb']:.4f} MB")
    print(f"  Size reduction ratio: {reduction_metrics['size_reduction_ratio']:.2f}")
    print(f"  Compression ratio: {reduction_metrics['compression_ratio']:.2f}")


def test_optimization_pipeline():
    """Test the complete optimization pipeline."""
    print("\nTesting optimization pipeline...")
    
    # Create a model
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(10, 20),
                nn.ReLU(),
                nn.Linear(20, 15),
                nn.ReLU(),
                nn.Linear(15, 1)
            )
        
        def forward(self, x):
            return self.layers(x)
    
    model = TestModel()
    
    # Run optimization pipeline
    optimizer = ModelOptimizer()
    results = optimizer.optimize_for_deployment(
        model,
        optimization_methods=['quantize', 'prune'],
        target_size_mb=0.1  # Small target to trigger optimization
    )
    
    print(f"[ok] Optimization pipeline completed")
    print(f"  Optimized models created: {list(results['optimized_models'].keys())}")
    print(f"  Metadata: {list(results['optimization_metadata'].keys())}")
    
    # Check if quantized model was created
    if 'quantized' in results['optimized_models']:
        print("  - Quantized model created successfully")
    
    # Check if pruned model was created
    if 'pruned' in results['optimized_models']:
        print("  - Pruned model created successfully")


def test_evaluate_model_performance():
    """Test model performance evaluation."""
    print("\nTesting model performance evaluation...")
    
    # Create a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(5, 10)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(10, 3)
        
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    model = SimpleModel()
    
    # Create dummy test data
    test_data = torch.randn(20, 5)
    test_targets = torch.randint(0, 3, (20,))
    
    # Create a DataLoader
    dataset = torch.utils.data.TensorDataset(test_data, test_targets)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=4)
    
    # Evaluate performance
    perf_metrics = evaluate_model_performance(model, test_loader)
    
    print(f"[ok] Performance evaluation completed")
    print(f"  Accuracy: {perf_metrics['accuracy']:.2f}%")
    print(f"  Average loss: {perf_metrics['average_loss']:.4f}")
    print(f"  Correct predictions: {perf_metrics['correct_predictions']}/{perf_metrics['total_predictions']}")


if __name__ == "__main__":
    print("Running model optimization tests...\n")
    
    test_quantization()
    test_pruning()
    test_size_reduction()
    test_optimization_pipeline()
    test_evaluate_model_performance()
    
    print("\n[ok] All model optimization tests completed!")
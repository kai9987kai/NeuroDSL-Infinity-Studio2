"""
Test script for performance analysis features in NeuroDSL-Infinity-Studio.
"""

import torch
import torch.nn as nn
from perf_analyzer import PerformanceAnalyzer


def test_performance_profiling():
    """Test performance profiling functionality."""
    print("Testing performance profiling...")
    
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
    input_tensor = torch.randn(1, 10)
    
    # Test performance analysis
    analyzer = PerformanceAnalyzer()
    perf_metrics = analyzer.profile_model_performance(model, input_tensor, num_runs=3)
    
    print(f"[ok] Performance profiling completed:")
    print(f"  Avg Latency: {perf_metrics['avg_latency_ms']:.4f} ms")
    print(f"  Throughput: {perf_metrics['throughput_samples_per_sec']:.2f} samples/sec")
    print(f"  Memory Usage: {perf_metrics['max_memory_bytes'] / (1024**2):.2f} MB")
    print(f"  Parameters: {perf_metrics['num_parameters']:,}")


def test_gradient_analysis():
    """Test gradient analysis functionality."""
    print("\nTesting gradient analysis...")
    
    # Create a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(5, 10)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(10, 1)
            
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    model = SimpleModel()
    input_tensor = torch.randn(4, 5)
    target_tensor = torch.randn(4, 1)
    criterion = nn.MSELoss()
    
    # Test gradient analysis
    analyzer = PerformanceAnalyzer()
    grad_analysis = analyzer.analyze_gradient_flow(model, input_tensor, target_tensor, criterion)
    
    print(f"[ok] Gradient analysis completed:")
    print(f"  Avg Gradient Norm: {grad_analysis['avg_gradient_norm']:.6f}")
    print(f"  Vanishing Gradients: {grad_analysis['vanishing_gradients_count']}")
    print(f"  Exploding Gradients: {grad_analysis['exploding_gradients_count']}")
    print(f"  Vanishing Ratio: {grad_analysis['vanishing_ratio']:.2f}")


def test_flops_calculation():
    """Test FLOPs calculation functionality."""
    print("\nTesting FLOPs calculation...")
    
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
    input_tensor = torch.randn(1, 8)
    
    # Test FLOPs calculation
    analyzer = PerformanceAnalyzer()
    flops_analysis = analyzer.calculate_flops(model, input_tensor)
    
    print(f"[ok] FLOPs calculation completed:")
    print(f"  Total FLOPs: {flops_analysis['total_flops']:,}")
    print(f"  FLOPs by layer type: {flops_analysis['flops_by_layer_type']}")


def test_performance_report():
    """Test performance report generation."""
    print("\nTesting performance report generation...")
    
    # Create a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(6, 12)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(12, 1)
            
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    model = SimpleModel()
    input_tensor = torch.randn(1, 6)
    target_tensor = torch.randn(1, 1)
    criterion = nn.MSELoss()
    
    # Generate performance report
    analyzer = PerformanceAnalyzer()
    report = analyzer.generate_performance_report(model, input_tensor, target_tensor, criterion)
    
    print("[ok] Performance report generated")
    print("Sample of report:")
    print("\n".join(report.split("\n")[:15]))  # Print first 15 lines


if __name__ == "__main__":
    print("Running performance analysis tests...\n")
    
    test_performance_profiling()
    test_gradient_analysis()
    test_flops_calculation()
    test_performance_report()
    
    print("\n[ok] All performance analysis tests completed!")
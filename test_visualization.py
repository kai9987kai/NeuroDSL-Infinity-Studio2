"""
Test script for visualization features in NeuroDSL-Infinity-Studio.
"""

import torch
import torch.nn as nn
from viz_utils import ModelVisualizer
from trainer import TrainingEngine


def test_model_visualization():
    """Test model visualization functionality."""
    print("Testing model visualization...")
    
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
    
    # Test visualization
    visualizer = ModelVisualizer()
    fig = visualizer.plot_architecture(model, title="Test Model Architecture")
    print("[ok] Model architecture plot created successfully")
    
    # Test getting model summary
    summary = visualizer.get_model_summary(model)
    print(f"[ok] Model summary generated:")
    print(f"  Total params: {summary['total_params']}")
    print(f"  Trainable params: {summary['trainable_params']}")
    print(f"  Non-trainable params: {summary['non_trainable_params']}")
    print(f"  Number of layers: {summary['num_layers']}")
    print(f"  Estimated size: {summary['model_size_mb']:.2f} MB")


def test_training_history_visualization():
    """Test training history visualization functionality."""
    print("\nTesting training history visualization...")
    
    # Create a simple model for the trainer
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
    trainer = TrainingEngine(model)
    
    # Simulate some training history
    trainer.training_history = {
        'loss': [1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.18, 0.15],
        'val_loss': [1.1, 0.85, 0.65, 0.52, 0.42, 0.32, 0.27, 0.22, 0.19, 0.16],
        'lr': [0.01, 0.01, 0.01, 0.009, 0.008, 0.007, 0.006, 0.005, 0.003, 0.002],
        'grad_norm': [1.2, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
    }
    
    # Test visualization
    visualizer = ModelVisualizer()
    fig = visualizer.plot_training_history(trainer.training_history, title="Test Training History")
    print("[ok] Training history plot created successfully")


def test_export_functionality():
    """Test export functionality."""
    print("\nTesting export functionality...")
    
    # Create a simple test model
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
    
    # Test export
    visualizer = ModelVisualizer()
    fig = visualizer.plot_architecture(model, title="Test Export Model")
    
    # Export to a temporary file (would normally be to a real path)
    try:
        visualizer.export_figure(fig, "test_export.png")
        print("[ok] Model visualization export successful")
    except Exception as e:
        print(f"[error] Model visualization export failed: {e}")


if __name__ == "__main__":
    print("Running visualization tests...\n")
    
    test_model_visualization()
    test_training_history_visualization()
    test_export_functionality()
    
    print("\n[ok] All visualization tests completed!")
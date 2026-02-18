"""Performance analyzer for NeuroDSL Infinity Studio."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import time
import psutil
from collections import defaultdict
import matplotlib.pyplot as plt


class PerformanceAnalyzer:
    """Advanced performance monitoring and analysis tools."""
    
    def __init__(self):
        self.performance_metrics = {}
        self.activation_maps = {}
        self.gradient_flows = {}
        self.memory_usage = []
        self.flops_counter = 0
    
    def profile_model_performance(self, model: nn.Module, input_tensor: torch.Tensor, 
                                num_runs: int = 10) -> Dict[str, Any]:
        """
        Profile model performance metrics including latency, memory usage, and throughput.
        
        Args:
            model: PyTorch model to profile
            input_tensor: Input tensor for profiling
            num_runs: Number of runs for averaging
            
        Returns:
            Dictionary containing performance metrics
        """
        model.eval()
        device = next(model.parameters()).device
        
        # Prepare input tensor
        input_tensor = input_tensor.to(device)
        
        # Warmup runs
        for _ in range(3):
            with torch.no_grad():
                _ = model(input_tensor)
        
        # Measure latency
        latencies = []
        for _ in range(num_runs):
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()
            
            with torch.no_grad():
                _ = model(input_tensor)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            latencies.append((time.time() - start_time) * 1000)  # Convert to milliseconds
        
        avg_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        
        # Measure memory usage
        if torch.cuda.is_available():
            max_memory = torch.cuda.max_memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
        else:
            max_memory = psutil.Process().memory_info().rss  # in bytes
        
        # Calculate throughput
        batch_size = input_tensor.shape[0]
        throughput = (batch_size / (avg_latency / 1000))  # samples per second
        
        return {
            'avg_latency_ms': avg_latency,
            'std_latency_ms': std_latency,
            'throughput_samples_per_sec': throughput,
            'max_memory_bytes': max_memory,
            'batch_size': batch_size,
            'num_parameters': sum(p.numel() for p in model.parameters()),
            'latency_samples': latencies
        }
    
    def register_activation_hooks(self, model: nn.Module):
        """
        Register hooks to capture activation maps during forward pass.
        """
        self.activation_maps.clear()
        
        def get_activation_hook(name):
            def hook(module, input, output):
                self.activation_maps[name] = {
                    'input': input[0].detach().cpu() if input and len(input) > 0 else None,
                    'output': output.detach().cpu() if output is not None else None
                }
            return hook
        
        for name, layer in model.named_modules():
            if isinstance(layer, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.LSTM, nn.GRU)):
                layer.register_forward_hook(get_activation_hook(name))
    
    def register_gradient_hooks(self, model: nn.Module):
        """
        Register hooks to capture gradient flows during backward pass.
        """
        self.gradient_flows.clear()
        
        def get_gradient_hook(name):
            def hook(module, grad_input, grad_output):
                self.gradient_flows[name] = {
                    'grad_input': grad_input[0].detach().cpu() if grad_input and len(grad_input) > 0 else None,
                    'grad_output': grad_output[0].detach().cpu() if grad_output and len(grad_output) > 0 else None
                }
            return hook
        
        for name, layer in model.named_modules():
            if isinstance(layer, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                layer.register_backward_hook(get_gradient_hook(name))
    
    def analyze_gradient_flow(self, model: nn.Module, input_tensor: torch.Tensor, 
                            target_tensor: torch.Tensor, criterion) -> Dict[str, Any]:
        """
        Analyze gradient flow through the network.
        
        Args:
            model: PyTorch model to analyze
            input_tensor: Input tensor
            target_tensor: Target tensor for loss calculation
            criterion: Loss function
            
        Returns:
            Dictionary containing gradient analysis results
        """
        model.train()
        device = next(model.parameters()).device
        
        input_tensor = input_tensor.to(device)
        target_tensor = target_tensor.to(device)
        
        # Clear gradients
        model.zero_grad()
        
        # Forward pass
        output = model(input_tensor)
        loss = criterion(output, target_tensor)
        
        # Backward pass
        loss.backward()
        
        # Analyze gradients
        grad_norms = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms[name] = grad_norm
        
        # Identify vanishing/exploding gradients
        grad_values = list(grad_norms.values())
        avg_grad_norm = np.mean(grad_values) if grad_values else 0
        std_grad_norm = np.std(grad_values) if grad_values else 0
        
        # Count vanishing/exploding gradients
        vanishing_count = sum(1 for g in grad_values if g < 1e-6)
        exploding_count = sum(1 for g in grad_values if g > 100)
        
        return {
            'gradient_norms': grad_norms,
            'avg_gradient_norm': avg_grad_norm,
            'std_gradient_norm': std_grad_norm,
            'vanishing_gradients_count': vanishing_count,
            'exploding_gradients_count': exploding_count,
            'total_parameters': len(grad_norms),
            'vanishing_ratio': vanishing_count / len(grad_norms) if grad_norms else 0,
            'exploding_ratio': exploding_count / len(grad_norms) if grad_norms else 0
        }
    
    def compare_models(self, models: List[Tuple[str, nn.Module]], input_tensor: torch.Tensor) -> Dict[str, Any]:
        """
        Compare multiple models based on performance metrics.
        
        Args:
            models: List of tuples (name, model) to compare
            input_tensor: Input tensor for all models
            
        Returns:
            Dictionary containing comparison results
        """
        comparison_results = {}
        
        for name, model in models:
            perf_metrics = self.profile_model_performance(model, input_tensor, num_runs=5)
            comparison_results[name] = perf_metrics
        
        # Calculate comparison summary
        summary = {
            'fastest_model': min(comparison_results.keys(), 
                               key=lambda x: comparison_results[x]['avg_latency_ms']),
            'slowest_model': max(comparison_results.keys(), 
                               key=lambda x: comparison_results[x]['avg_latency_ms']),
            'most_efficient_model': min(comparison_results.keys(), 
                                      key=lambda x: comparison_results[x]['avg_latency_ms'] / 
                                                   (comparison_results[x]['throughput_samples_per_sec'] + 1e-6)),
            'models_compared': list(comparison_results.keys()),
            'detailed_results': comparison_results
        }
        
        return summary
    
    def calculate_flops(self, model: nn.Module, input_tensor: torch.Tensor) -> Dict[str, Any]:
        """
        Calculate FLOPs (Floating Point Operations) for the model.
        NOTE: This is a simplified estimation, not exact FLOPs counting.
        """
        flops_breakdown = {}
        total_flops = 0
        
        # Register hooks to track operations
        def count_linear_flops(module, input, output):
            nonlocal total_flops
            input_nodes = input[0].shape[-1]
            output_nodes = output.shape[-1]
            batch_size = input[0].shape[0] if len(input[0].shape) > 1 else 1
            flops = batch_size * input_nodes * output_nodes  # Matrix multiplication
            flops_breakdown[f"{module.__class__.__name__}_{id(module)}"] = flops
            total_flops += flops
        
        def count_conv_flops(module, input, output):
            nonlocal total_flops
            batch_size = input[0].shape[0]
            out_h, out_w = output.shape[-2], output.shape[-1]
            kernel_ops = module.weight.shape[2] * module.weight.shape[3]
            flops = batch_size * out_h * out_w * module.in_channels * module.out_channels * kernel_ops
            flops_breakdown[f"{module.__class__.__name__}_{id(module)}"] = flops
            total_flops += flops
        
        # Register hooks to all relevant layers
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                hooks.append(module.register_forward_hook(count_linear_flops))
            elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                hooks.append(module.register_forward_hook(count_conv_flops))
        
        # Run a forward pass to trigger the hooks
        model.eval()
        device = next(model.parameters()).device
        input_tensor = input_tensor.to(device)
        
        with torch.no_grad():
            _ = model(input_tensor)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return {
            'total_flops': total_flops,
            'flops_breakdown': flops_breakdown,
            'flops_by_layer_type': self._aggregate_flops_by_type(flops_breakdown)
        }
    
    def _aggregate_flops_by_type(self, flops_breakdown: Dict[str, int]) -> Dict[str, int]:
        """Aggregate FLOPs by layer type."""
        aggregated = defaultdict(int)
        for key, flops in flops_breakdown.items():
            layer_type = key.split('_')[0]
            aggregated[layer_type] += flops
        return dict(aggregated)
    
    def generate_performance_report(self, model: nn.Module, input_tensor: torch.Tensor, 
                                  target_tensor: torch.Tensor, criterion) -> str:
        """
        Generate a comprehensive performance report.
        """
        # Performance metrics
        perf_metrics = self.profile_model_performance(model, input_tensor)
        
        # Gradient analysis
        grad_analysis = self.analyze_gradient_flow(model, input_tensor, target_tensor, criterion)
        
        # FLOPs calculation
        flops_analysis = self.calculate_flops(model, input_tensor)
        
        report = f"""
NEURAL NETWORK PERFORMANCE ANALYSIS REPORT
==========================================

PERFORMANCE METRICS:
- Average Latency: {perf_metrics['avg_latency_ms']:.4f} ms
- Std Dev Latency: {perf_metrics['std_latency_ms']:.4f} ms
- Throughput: {perf_metrics['throughput_samples_per_sec']:.2f} samples/sec
- Max Memory Usage: {perf_metrics['max_memory_bytes'] / (1024**2):.2f} MB
- Total Parameters: {perf_metrics['num_parameters']:,}

GRADIENT ANALYSIS:
- Average Gradient Norm: {grad_analysis['avg_gradient_norm']:.6f}
- Std Dev Gradient Norm: {grad_analysis['std_gradient_norm']:.6f}
- Vanishing Gradients: {grad_analysis['vanishing_gradients_count']} ({grad_analysis['vanishing_ratio']*100:.2f}%)
- Exploding Gradients: {grad_analysis['exploding_gradients_count']} ({grad_analysis['exploding_ratio']*100:.2f}%)

COMPUTATIONAL COMPLEXITY:
- Total Estimated FLOPs: {flops_analysis['total_flops']:,}
- FLOPs by Layer Type: {flops_analysis['flops_by_layer_type']}

RECOMMENDATIONS:
"""
        
        # Add recommendations based on analysis
        if grad_analysis['vanishing_ratio'] > 0.3:
            report += "- High vanishing gradient ratio detected. Consider using residual connections or normalization layers.\n"
        
        if grad_analysis['exploding_ratio'] > 0.1:
            report += "- High exploding gradient ratio detected. Consider gradient clipping or smaller learning rates.\n"
        
        if perf_metrics['avg_latency_ms'] > 100:
            report += "- High latency detected. Consider model compression techniques or architecture optimization.\n"
        
        if flops_analysis['total_flops'] > 1e9:  # More than 1 billion FLOPs
            report += "- High computational complexity. Consider pruning or quantization for deployment.\n"
        
        return report


def plot_performance_comparison(comparison_results: Dict[str, Any]) -> plt.Figure:
    """
    Plot a comparison of model performances.
    
    Args:
        comparison_results: Results from compare_models function
        
    Returns:
        Matplotlib figure with performance comparison plots
    """
    model_names = list(comparison_results['detailed_results'].keys())
    avg_latencies = [comparison_results['detailed_results'][name]['avg_latency_ms'] 
                     for name in model_names]
    throughputs = [comparison_results['detailed_results'][name]['throughput_samples_per_sec'] 
                   for name in model_names]
    memory_usages = [comparison_results['detailed_results'][name]['max_memory_bytes'] / (1024**2)
                     for name in model_names]  # Convert to MB
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot latency comparison
    axes[0].bar(model_names, avg_latencies, color='skyblue')
    axes[0].set_title('Average Latency Comparison')
    axes[0].set_ylabel('Latency (ms)')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Plot throughput comparison
    axes[1].bar(model_names, throughputs, color='lightgreen')
    axes[1].set_title('Throughput Comparison')
    axes[1].set_ylabel('Samples per Second')
    axes[1].tick_params(axis='x', rotation=45)
    
    # Plot memory usage comparison
    axes[2].bar(model_names, memory_usages, color='salmon')
    axes[2].set_title('Memory Usage Comparison')
    axes[2].set_ylabel('Memory (MB)')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return fig
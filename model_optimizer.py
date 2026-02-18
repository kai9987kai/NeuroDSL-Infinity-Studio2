"""Model optimizer for NeuroDSL Infinity Studio - Quantization, Pruning, and Distillation."""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import copy
from collections import OrderedDict
import gc


class ModelOptimizer:
    """Advanced model optimization tools including quantization, pruning, and knowledge distillation."""
    
    def __init__(self):
        self.quantized_model = None
        self.pruned_model = None
        self.distilled_model = None
    
    def quantize_model_dynamic(self, model: nn.Module, 
                              dtype=torch.qint8) -> nn.Module:
        """
        Apply dynamic quantization to the model.
        
        Args:
            model: PyTorch model to quantize
            dtype: Quantization data type (default torch.qint8)
            
        Returns:
            Quantized model
        """
        # Define which layers to quantize
        quantizable_layers = []
        for name, layer in model.named_modules():
            if isinstance(layer, (nn.Linear, nn.LSTM, nn.GRU, nn.RNN)):
                quantizable_layers.append((name, layer))
        
        # Create mapping for quantization
        qconfig_dict = {}
        for name, layer in quantizable_layers:
            qconfig_dict[name] = torch.quantization.default_dynamic_qconfig
        
        # Apply dynamic quantization
        quantized_model = torch.quantization.quantize_dynamic(
            model, 
            qconfig_dict, 
            dtype=dtype
        )
        
        return quantized_model
    
    def quantize_model_static(self, model: nn.Module, 
                             calib_loader: Optional[torch.utils.data.DataLoader] = None,
                             num_batches: int = 10) -> nn.Module:
        """
        Apply static quantization to the model.
        
        Args:
            model: PyTorch model to quantize
            calib_loader: Calibration data loader for static quantization
            num_batches: Number of batches to use for calibration
            
        Returns:
            Quantized model
        """
        # Set model to evaluation mode
        model.eval()
        
        # Fuse modules for better quantization
        model = self._fuse_modules(model)
        
        # Specify quantization configuration
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        
        # Prepare model for static quantization
        torch.quantization.prepare(model, inplace=True)
        
        # Calibrate the model if calibration data is provided
        if calib_loader is not None:
            with torch.no_grad():
                for i, (data, _) in enumerate(calib_loader):
                    if i >= num_batches:
                        break
                    model(data)
        
        # Convert to quantized model
        torch.quantization.convert(model, inplace=True)
        
        return model
    
    def _fuse_modules(self, model: nn.Module) -> nn.Module:
        """
        Fuse modules for better quantization performance.
        """
        for module_name, module in model.named_children():
            if len(list(module.children())) > 0:
                # Recursively fuse modules
                module = self._fuse_modules(module)
            
            # Fuse sequential operations like Conv + ReLU or Linear + ReLU
            if isinstance(module, nn.Sequential):
                # This is a simplified fusion - in practice, you'd have more complex logic
                pass
        
        return model
    
    def prune_model(self, model: nn.Module, 
                   pruning_method: str = 'l1_unstructured',
                   sparsity: float = 0.2,
                   target_layers: Optional[List[str]] = None) -> nn.Module:
        """
        Prune the model to reduce size and increase speed.
        
        Args:
            model: PyTorch model to prune
            pruning_method: Method to use for pruning ('l1_unstructured', 'random_unstructured')
            sparsity: Fraction of weights to zero out (between 0 and 1)
            target_layers: Specific layers to prune (if None, applies to all linear/conv layers)
            
        Returns:
            Pruned model
        """
        model = copy.deepcopy(model)  # Work on a copy
        model.train()  # Need to be in train mode for pruning
        
        # Determine which layers to prune
        if target_layers is None:
            target_layers = []
            for name, module in model.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                    target_layers.append(name)
        
        # Apply pruning to specified layers
        for name, module in model.named_modules():
            if name in target_layers:
                if pruning_method == 'l1_unstructured':
                    prune.l1_unstructured(module, name='weight', amount=sparsity)
                elif pruning_method == 'random_unstructured':
                    prune.random_unstructured(module, name='weight', amount=sparsity)
                elif pruning_method == 'ln_structured':
                    prune.ln_structured(module, name='weight', amount=sparsity, n=2, dim=0)
        
        return model
    
    def get_model_sparsity(self, model: nn.Module) -> Dict[str, float]:
        """
        Calculate the sparsity of the model (percentage of zero weights).
        
        Args:
            model: PyTorch model to analyze
            
        Returns:
            Dictionary mapping layer names to their sparsity percentages
        """
        sparsity_dict = {}
        
        for name, module in model.named_modules():
            if hasattr(module, 'weight'):
                total_params = module.weight.numel()
                zero_params = torch.sum(module.weight == 0).item()
                sparsity = zero_params / total_params if total_params > 0 else 0
                sparsity_dict[name] = sparsity
        
        return sparsity_dict
    
    def get_model_size_reduction(self, original_model: nn.Module, 
                                optimized_model: nn.Module) -> Dict[str, float]:
        """
        Calculate the size reduction achieved by optimization.
        
        Args:
            original_model: Original unoptimized model
            optimized_model: Optimized model
            
        Returns:
            Dictionary with size reduction metrics
        """
        # Calculate original model size
        original_size = sum(p.numel() * p.element_size() for p in original_model.parameters()) / 1024**2  # in MB
        
        # Calculate optimized model size
        try:
            # For quantized models, parameters take less space
            if hasattr(optimized_model, 'activation_post_process_input'):
                # This is likely a quantized model
                optimized_size = sum(p.numel() * 1 for p in optimized_model.parameters()) / 1024**2  # 1 byte per param for int8
            else:
                optimized_size = sum(p.numel() * p.element_size() for p in optimized_model.parameters()) / 1024**2  # in MB
        except:
            # Fallback: assume same parameter size
            optimized_size = sum(p.numel() * p.element_size() for p in optimized_model.parameters()) / 1024**2  # in MB
        
        reduction_ratio = (original_size - optimized_size) / original_size if original_size > 0 else 0
        
        return {
            'original_size_mb': original_size,
            'optimized_size_mb': optimized_size,
            'size_reduction_ratio': reduction_ratio,
            'compression_ratio': original_size / optimized_size if optimized_size > 0 else float('inf')
        }
    
    def knowledge_distillation(self, teacher_model: nn.Module,
                             student_model: nn.Module,
                             train_loader: torch.utils.data.DataLoader,
                             num_epochs: int = 10,
                             temperature: float = 3.0,
                             alpha: float = 0.7,
                             learning_rate: float = 1e-3) -> nn.Module:
        """
        Perform knowledge distillation from teacher to student model.
        
        Args:
            teacher_model: Larger, well-trained teacher model
            student_model: Smaller student model to train
            train_loader: Training data loader
            num_epochs: Number of training epochs
            temperature: Temperature for softening probability distributions
            alpha: Balance between hard and soft losses
            learning_rate: Learning rate for student training
            
        Returns:
            Trained student model
        """
        device = next(student_model.parameters()).device
        
        # Set teacher to eval mode (no gradient updates)
        teacher_model.eval()
        student_model.train()
        
        # Define optimizer and loss functions
        optimizer = torch.optim.Adam(student_model.parameters(), lr=learning_rate)
        hard_loss_fn = nn.CrossEntropyLoss()  # For ground truth labels
        soft_loss_fn = nn.KLDivLoss(reduction='batchmean')  # For soft labels
        
        # Training loop
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                
                # Get teacher predictions (no gradients)
                with torch.no_grad():
                    teacher_outputs = teacher_model(data)
                
                # Get student predictions
                student_outputs = student_model(data)
                
                # Calculate soft targets (logits divided by temperature)
                soft_teacher = torch.nn.functional.log_softmax(teacher_outputs / temperature, dim=1)
                soft_student = torch.nn.functional.log_softmax(student_outputs / temperature, dim=1)
                
                # Calculate losses
                soft_loss = soft_loss_fn(soft_student, soft_teacher) * (temperature ** 2)
                hard_loss = hard_loss_fn(student_outputs, target)
                
                # Combined loss
                loss = alpha * soft_loss + (1.0 - alpha) * hard_loss
                
                # Backpropagate
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            print(f"Distillation Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
        
        return student_model
    
    def optimize_for_deployment(self, model: nn.Module, 
                               optimization_methods: List[str] = ['quantize', 'prune'],
                               target_size_mb: Optional[float] = None,
                               target_latency: Optional[float] = None) -> Dict[str, Any]:
        """
        Optimize model for deployment considering size and latency constraints.
        
        Args:
            model: Original model to optimize
            optimization_methods: List of optimization methods to apply
            target_size_mb: Target size in MB (optional)
            target_latency: Target latency in ms (optional)
            
        Returns:
            Dictionary containing optimized models and metadata
        """
        results = {
            'original_model': model,
            'optimized_models': {},
            'optimization_metadata': {}
        }
        
        original_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
        
        for method in optimization_methods:
            if method == 'quantize':
                # Apply dynamic quantization
                quantized_model = self.quantize_model_dynamic(copy.deepcopy(model))
                results['optimized_models']['quantized'] = quantized_model
                
                # Calculate metrics
                size_metrics = self.get_model_size_reduction(model, quantized_model)
                results['optimization_metadata']['quantized'] = {
                    'method': 'dynamic_quantization',
                    'original_size_mb': size_metrics['original_size_mb'],
                    'optimized_size_mb': size_metrics['optimized_size_mb'],
                    'size_reduction_ratio': size_metrics['size_reduction_ratio'],
                    'compression_ratio': size_metrics['compression_ratio']
                }
                
            elif method == 'prune':
                # Apply pruning with increasing sparsity until target is met
                sparsity_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
                pruned_model = None
                
                for sparsity in sparsity_levels:
                    temp_pruned = self.prune_model(copy.deepcopy(model), sparsity=sparsity)
                    size_metrics = self.get_model_size_reduction(model, temp_pruned)
                    
                    # If target size is specified and met, use this model
                    if target_size_mb is not None and size_metrics['optimized_size_mb'] <= target_size_mb:
                        pruned_model = temp_pruned
                        break
                    # If no target size but we just want to try different levels
                    elif target_size_mb is None:
                        pruned_model = temp_pruned
                        break
                
                if pruned_model is not None:
                    results['optimized_models']['pruned'] = pruned_model
                    results['optimization_metadata']['pruned'] = {
                        'method': f'pruning_{sparsity:.1f}',
                        'sparsity_level': sparsity,
                        'original_size_mb': size_metrics['original_size_mb'],
                        'optimized_size_mb': size_metrics['optimized_size_mb'],
                        'size_reduction_ratio': size_metrics['size_reduction_ratio'],
                        'compression_ratio': size_metrics['compression_ratio']
                    }
        
        return results

    def export_fp16(self, model: nn.Module, path: str, input_dim: int) -> str:
        """Export model in FP16 (half-precision) format to ONNX."""
        import copy
        model_fp16 = copy.deepcopy(model).half()
        model_fp16.eval()
        
        device = next(model_fp16.parameters()).device
        dummy_input = torch.randn(1, input_dim).half().to(device)
        
        try:
            torch.onnx.export(
                model_fp16, dummy_input, path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output']
            )
            return path
        except Exception as e:
            raise RuntimeError(f"FP16 Export failed: {e}")

    def visualize_weight_histogram(self, model: nn.Module) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return histogram data for model weights.
        Returns: (counts, bin_edges)
        """
        all_weights = []
        for name, param in model.named_parameters():
            if 'weight' in name and param.requires_grad:
                all_weights.append(param.data.cpu().float().numpy().flatten())
        
        if not all_weights:
            return np.array([]), np.array([])
            
        combined = np.concatenate(all_weights)
        counts, bin_edges = np.histogram(combined, bins=100)
        return counts, bin_edges

    def structured_pruning(self, model: nn.Module, amount: float = 0.2) -> nn.Module:
        """
        Apply structured pruning (L2 norm) to Conv and Linear layers.
        Removes entire channels/neurons.
        """
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d)):
                prune.ln_structured(module, name='weight', amount=amount, n=2, dim=0)
            elif isinstance(module, nn.Linear):
                prune.ln_structured(module, name='weight', amount=amount, n=2, dim=0)
        return model


def evaluate_model_performance(model: nn.Module, 
                              test_loader: torch.utils.data.DataLoader,
                              device: str = 'cpu') -> Dict[str, float]:
    """
    Evaluate model performance on test data.
    
    Args:
        model: Model to evaluate
        test_loader: Test data loader
        device: Device to run evaluation on
        
    Returns:
        Dictionary with performance metrics
    """
    model.eval()
    model.to(device)
    
    correct = 0
    total = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss_sum += criterion(output, target).item()
            
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    accuracy = 100. * correct / total
    avg_loss = loss_sum / len(test_loader)
    
    return {
        'accuracy': accuracy,
        'average_loss': avg_loss,
        'correct_predictions': correct,
        'total_predictions': total
    }
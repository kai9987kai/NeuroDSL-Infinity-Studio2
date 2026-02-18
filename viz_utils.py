"""Visualization utilities for NeuroDSL Infinity Studio."""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any, Optional
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg
import torchvision


class ModelVisualizer:
    """Utilities for visualizing neural network architectures and training progress."""
    
    @staticmethod
    def plot_architecture(model: nn.Module, title: str = "Neural Network Architecture") -> plt.Figure:
        """
        Plot a diagram of the neural network architecture.
        
        Args:
            model: PyTorch model to visualize
            title: Title for the plot
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract layer information
        layers = []
        for name, layer in model.named_modules():
            if name != '' and hasattr(layer, 'parameters'):  # Skip the root module
                layer_type = type(layer).__name__
                param_count = sum(p.numel() for p in layer.parameters())
                layers.append({
                    'name': name,
                    'type': layer_type,
                    'params': param_count
                })
        
        # Create horizontal bar chart of layers
        layer_names = [layer['name'] for layer in layers]
        layer_types = [layer['type'] for layer in layers]
        param_counts = [layer['params'] for layer in layers]
        
        y_pos = np.arange(len(layers))
        
        bars = ax.barh(y_pos, param_counts, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{name} ({ltype})" for name, ltype in zip(layer_names, layer_types)])
        ax.set_xlabel('Number of Parameters')
        ax.set_title(title)
        
        # Add parameter count labels on bars
        for i, (bar, count) in enumerate(zip(bars, param_counts)):
            width = bar.get_width()
            ax.text(width, bar.get_xy()[1] + bar.get_height()/2, f'{count:,}', 
                   ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_training_history(history: Dict[str, List[float]], title: str = "Training History") -> plt.Figure:
        """
        Plot training history including loss, learning rate, and gradient norm.
        
        Args:
            history: Dictionary containing training metrics over time
            title: Title for the plot
            
        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(title)
        
        epochs = list(range(len(history.get('loss', []))))
        
        # Plot loss
        if 'loss' in history:
            axes[0, 0].plot(epochs, history['loss'], label='Loss', color='blue')
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True)
        
        # Plot learning rate
        if 'lr' in history:
            axes[0, 1].plot(epochs, history['lr'], label='Learning Rate', color='green')
            axes[0, 1].set_title('Learning Rate')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('LR')
            axes[0, 1].grid(True)
        
        # Plot gradient norm
        if 'grad_norm' in history:
            axes[1, 0].plot(epochs, history['grad_norm'], label='Gradient Norm', color='red')
            axes[1, 0].set_title('Gradient Norm')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Grad Norm')
            axes[1, 0].grid(True)
        
        # Plot validation loss if available
        if 'val_loss' in history:
            axes[1, 1].plot(epochs, history['val_loss'], label='Validation Loss', color='orange')
            if 'loss' in history:
                axes[1, 1].plot(epochs, history['loss'], label='Training Loss', color='blue', alpha=0.7)
            axes[1, 1].set_title('Train vs Validation Loss')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def get_model_summary(model: nn.Module) -> Dict[str, Any]:
        """
        Generate a comprehensive summary of the model.
        
        Args:
            model: PyTorch model to summarize
            
        Returns:
            Dictionary containing model statistics
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params
        
        layer_info = []
        for name, layer in model.named_modules():
            if name != '' and hasattr(layer, 'parameters'):
                layer_type = type(layer).__name__
                param_count = sum(p.numel() for p in layer.parameters())
                trainable = any(p.requires_grad for p in layer.parameters())
                
                layer_info.append({
                    'name': name,
                    'type': layer_type,
                    'params': param_count,
                    'trainable': trainable
                })
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'non_trainable_params': non_trainable_params,
            'layers': layer_info,
            'num_layers': len(layer_info),
            'model_size_mb': (total_params * 4) / (1024 * 1024)  # Assuming 32-bit floats
        }

    @staticmethod
    def export_figure(fig: plt.Figure, filepath: str):
        """
        Export a matplotlib figure to a file.
        
        Args:
            fig: Matplotlib figure object to export
            filepath: Path to save the figure (with extension)
        """
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)

    @staticmethod
    def plot_image_grid(
        images: torch.Tensor,
        title: str = "Image Grid",
        grid_size: Optional[Tuple[int, int]] = None,
    ) -> plt.Figure:
        """
        Plots a grid of images.
        Args:
            images: A tensor of images to plot. Shape (N, C, H, W).
            title: The title for the plot.
            grid_size: A tuple (rows, cols) for the grid. If None, it's inferred.
        Returns:
            A matplotlib figure object.
        """
        if grid_size:
            nrow = grid_size[1]
        else:
            nrow = int(np.sqrt(images.size(0)))

        grid_img = torchvision.utils.make_grid(images, nrow=nrow, normalize=True)
        grid_img = grid_img.permute(1, 2, 0)

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(grid_img.cpu())
        ax.set_title(title)
        ax.axis("off")
        plt.tight_layout()
        return fig

    @staticmethod
    def visualize_feature_maps(
        model: nn.Module,
        input_image: torch.Tensor,
        target_layer: str,
        max_maps: int = 64,
    ) -> plt.Figure:
        """
        Visualizes the feature maps of a target layer in a model.
        Args:
            model: The PyTorch model.
            input_image: An input image tensor.
            target_layer: The name of the layer to visualize.
            max_maps: The maximum number of feature maps to display.
        Returns:
            A matplotlib figure showing the feature maps.
        """
        feature_maps = []

        def hook(module, input, output):
            feature_maps.append(output)

        # Find the target layer and register the hook
        target_module = None
        for name, module in model.named_modules():
            if name == target_layer:
                target_module = module
                break
        if target_module is None:
            raise ValueError(f"Target layer '{target_layer}' not found in the model.")

        handle = target_module.register_forward_hook(hook)

        # Perform a forward pass
        with torch.no_grad():
            model(input_image)

        handle.remove()  # Clean up the hook

        if not feature_maps:
            raise ValueError("Could not extract feature maps.")

        maps = feature_maps[0].squeeze(0)
        num_maps = min(maps.size(0), max_maps)
        
        # Determine grid size
        cols = int(np.sqrt(num_maps))
        rows = int(np.ceil(num_maps / cols))
        
        fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
        fig.suptitle(f"Feature Maps from Layer: {target_layer}", fontsize=16)

        for i in range(num_maps):
            ax = axes.flat[i]
            ax.imshow(maps[i].cpu().numpy(), cmap="viridis")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"Map {i+1}")

        # Hide unused subplots
        for i in range(num_maps, len(axes.flat)):
            axes.flat[i].set_visible(False)
            
        plt.tight_layout()
        return fig


def convert_fig_to_base64(fig: plt.Figure) -> str:
    """
    Convert a matplotlib figure to a base64 encoded string.
    
    Args:
        fig: Matplotlib figure object
        
    Returns:
        Base64 encoded string representation of the figure
    """
    canvas = FigureCanvasAgg(fig)
    buf = io.BytesIO()
    canvas.print_png(buf)
    data = buf.getvalue()
    encoded = base64.b64encode(data).decode('utf-8')
    buf.close()
    plt.close(fig)  # Close the figure to free memory
    return encoded
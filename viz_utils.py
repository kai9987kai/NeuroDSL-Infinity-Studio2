"""Visualization utilities for NeuroDSL Infinity Studio."""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any, Optional
from io import BytesIO
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    import seaborn as sns  # noqa: F401
except Exception:
    sns = None

try:
    import torchvision
except Exception:
    torchvision = None


class ModelVisualizer:
    """Utilities for visualizing neural network architectures and training progress."""

    def __init__(self, model: Optional[nn.Module] = None):
        self.model = model
    
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
        if not param_counts:
            ax.text(0.5, 0.5, "No parameterized layers found", ha="center", va="center")
            ax.set_axis_off()
            plt.tight_layout()
            return fig
        
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
        nrow = max(1, int(nrow))
        imgs = images.detach().cpu()

        if torchvision is not None:
            grid_img = torchvision.utils.make_grid(imgs, nrow=nrow, normalize=True)
            grid_img = grid_img.permute(1, 2, 0)
        else:
            # Fallback path when torchvision is unavailable.
            n, c, h, w = imgs.shape
            ncol = nrow
            nrows = int(np.ceil(n / ncol))
            canvas = torch.zeros(c, nrows * h, ncol * w, dtype=imgs.dtype)
            for idx in range(n):
                r = idx // ncol
                col = idx % ncol
                canvas[:, r * h : (r + 1) * h, col * w : (col + 1) * w] = imgs[idx]
            # Normalize to [0, 1] for imshow
            mn = canvas.min()
            mx = canvas.max()
            canvas = (canvas - mn) / (mx - mn + 1e-8)
            grid_img = canvas.permute(1, 2, 0)

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(grid_img.cpu())
        ax.set_title(title)
        ax.axis("off")
        plt.tight_layout()
        return fig

    def plot_layer_activations(self, activations_dict: Dict[str, torch.Tensor], save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot activation distributions for each layer
        
        Args:
            activations_dict: Dictionary mapping layer names to activation tensors
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        n_layers = len(activations_dict)
        fig, axes = plt.subplots(n_layers, 1, figsize=(10, 4*n_layers))
        
        if n_layers == 1:
            axes = [axes]
        
        for idx, (layer_name, activations) in enumerate(activations_dict.items()):
            if isinstance(activations, torch.Tensor):
                activations = activations.detach().cpu().numpy()
            
            axes[idx].hist(activations.flatten(), bins=50, alpha=0.7)
            axes[idx].set_title(f'Activation Distribution - {layer_name}')
            axes[idx].set_xlabel('Activation Value')
            axes[idx].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        return fig

    def plot_feature_maps_3d(self, feature_map: torch.Tensor, save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize feature maps in 3D
        
        Args:
            feature_map: Feature map tensor to visualize
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        if isinstance(feature_map, torch.Tensor):
            feature_map = feature_map.detach().cpu().numpy()
        
        if len(feature_map.shape) == 2:  # (batch, features)
            pca = PCA(n_components=3)
            reduced = pca.fit_transform(feature_map)
            
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(reduced[:, 0], reduced[:, 1], reduced[:, 2])
            ax.set_title('Feature Map Visualization (PCA-reduced)')
        elif len(feature_map.shape) == 3:  # (batch, seq, features)
            # Use t-SNE for dimensionality reduction
            batch_size, seq_len, features = feature_map.shape
            reshaped = feature_map.reshape(-1, features)
            tsne = TSNE(n_components=3, random_state=42)
            reduced = tsne.fit_transform(reshaped)
            
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Color by sequence position
            colors = plt.cm.viridis(np.linspace(0, 1, seq_len))
            for i in range(seq_len):
                start_idx = i * batch_size
                end_idx = (i + 1) * batch_size
                ax.scatter(
                    reduced[start_idx:end_idx, 0],
                    reduced[start_idx:end_idx, 1],
                    reduced[start_idx:end_idx, 2],
                    c=[colors[i]] * batch_size,
                    label=f'Step {i}',
                    alpha=0.7
                )
            ax.set_title('Feature Map Visualization (t-SNE reduced)')
        else:
            raise ValueError(f"Unsupported feature map shape: {feature_map.shape}")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        return fig

    def plot_gradient_flow(self, model: nn.Module, save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize gradient flow through the network
        
        Args:
            model: PyTorch model to analyze
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        avg_gradients = []
        layer_names = []
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                avg_gradients.append(param.grad.abs().mean().item())
                layer_names.append(name)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(avg_gradients, alpha=0.7)
        ax.set_title('Gradient Flow Through Network Layers')
        ax.set_xlabel('Layer Index')
        ax.set_ylabel('Average Gradient Magnitude')
        ax.grid(True)
        
        # Add layer names as x-axis labels (with rotation to fit)
        ax.set_xticks(range(len(layer_names)))
        ax.set_xticklabels(layer_names, rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        return fig

    def plot_parameter_heatmap(self, model: nn.Module, save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a heatmap of parameter values in the model
        
        Args:
            model: PyTorch model to analyze
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        param_magnitudes = []
        layer_names = []
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                param_magnitudes.append(param.data.abs().mean().item())
                layer_names.append(name.split('.')[-1])  # Just the layer name
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create a simple bar chart showing parameter magnitude per layer
        bars = ax.bar(range(len(layer_names)), param_magnitudes)
        ax.set_xlabel('Layers')
        ax.set_ylabel('Average Parameter Magnitude')
        ax.set_title('Parameter Magnitude Across Network Layers')
        
        # Rotate x-axis labels to fit
        ax.set_xticks(range(len(layer_names)))
        ax.set_xticklabels(layer_names, rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        return fig

    def plot_attention_heatmap(self, attention_weights: torch.Tensor, save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a heatmap for attention weights
        
        Args:
            attention_weights: Attention weight tensor
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.detach().cpu().numpy()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(attention_weights, cmap='viridis', aspect='auto')
        ax.set_title('Attention Weight Heatmap')
        ax.set_xlabel('Key Positions')
        ax.set_ylabel('Query Positions')
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        return fig

    def plot_loss_landscape(self, model: nn.Module, train_loader: torch.utils.data.DataLoader, 
                           criterion: nn.Module, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot a simplified loss landscape around the current model parameters
        
        Args:
            model: PyTorch model
            train_loader: DataLoader for training data
            criterion: Loss function
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        # Store original parameters
        original_params = {name: param.clone() for name, param in model.named_parameters()}
        
        # Generate random direction vectors
        directions = {}
        for name, param in model.named_parameters():
            directions[name] = torch.randn_like(param)
            # Normalize the direction vector
            directions[name] /= directions[name].norm()
        
        # Define the range of perturbations
        alphas = np.linspace(-0.5, 0.5, 21)
        losses = []
        
        for alpha in alphas:
            # Apply perturbation
            for name, param in model.named_parameters():
                param.data.copy_(original_params[name] + alpha * directions[name])
            
            # Calculate loss
            model.eval()
            total_loss = 0
            with torch.no_grad():
                for inputs, targets in train_loader:
                    outputs = model(inputs)
                    if isinstance(outputs, tuple) or isinstance(outputs, list):
                        outputs = outputs[0]
                    loss = criterion(outputs, targets)
                    total_loss += loss.item()
            
            losses.append(total_loss / len(train_loader))
        
        # Restore original parameters
        for name, param in model.named_parameters():
            param.data.copy_(original_params[name])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(alphas, losses, marker='o')
        ax.set_title('Loss Landscape (1D slice)')
        ax.set_xlabel('Perturbation Factor (α)')
        ax.set_ylabel('Loss')
        ax.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        return fig

    def plot_training_metrics(self, metrics_history: Dict[str, List[float]], 
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot training metrics over time
        
        Args:
            metrics_history: Dictionary containing training metrics
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        epochs = range(len(metrics_history.get('train_loss', [])))
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot training and validation loss
        if 'train_loss' in metrics_history:
            axes[0, 0].plot(epochs, metrics_history['train_loss'], label='Train Loss', marker='o')
            if 'val_loss' in metrics_history:
                axes[0, 0].plot(epochs, metrics_history['val_loss'], label='Validation Loss', marker='s')
            axes[0, 0].set_title('Training and Validation Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
        
        # Plot accuracy if available
        if 'train_acc' in metrics_history:
            axes[0, 1].plot(epochs, metrics_history['train_acc'], label='Train Accuracy', marker='o')
            if 'val_acc' in metrics_history:
                axes[0, 1].plot(epochs, metrics_history['val_acc'], label='Validation Accuracy', marker='s')
            axes[0, 1].set_title('Training and Validation Accuracy')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # Plot learning rate if available
        if 'lr' in metrics_history:
            axes[1, 0].plot(epochs, metrics_history['lr'], label='Learning Rate', color='green', marker='o')
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Plot gradient norm if available
        if 'grad_norm' in metrics_history:
            axes[1, 1].plot(epochs, metrics_history['grad_norm'], label='Gradient Norm', color='red', marker='o')
            axes[1, 1].set_title('Gradient Norm')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Gradient Norm')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        return fig

    @staticmethod
    def plot_uncertainty_bands(
        means: torch.Tensor | np.ndarray,
        stds: torch.Tensor | np.ndarray,
        title: str = "Prediction Uncertainty Bands",
        max_series: int = 8,
    ) -> plt.Figure:
        """Plot mean predictions with +-1 std uncertainty envelopes."""
        mean_arr = means.detach().cpu().numpy() if isinstance(means, torch.Tensor) else np.asarray(means)
        std_arr = stds.detach().cpu().numpy() if isinstance(stds, torch.Tensor) else np.asarray(stds)

        if mean_arr.ndim == 1:
            mean_arr = mean_arr[None, :]
            std_arr = std_arr[None, :]
        if mean_arr.shape != std_arr.shape:
            raise ValueError(f"means and stds must share shape, got {mean_arr.shape} vs {std_arr.shape}")

        steps, dims = mean_arr.shape
        x = np.arange(steps)
        show_dims = min(dims, max_series)

        fig, ax = plt.subplots(figsize=(12, 6))
        for i in range(show_dims):
            y = mean_arr[:, i]
            s = std_arr[:, i]
            ax.plot(x, y, linewidth=1.8, label=f"class {i}")
            ax.fill_between(x, y - s, y + s, alpha=0.15)
        if dims > show_dims:
            ax.text(0.99, 0.02, f"+{dims - show_dims} more series", transform=ax.transAxes, ha="right", va="bottom")

        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.set_ylabel("Prediction")
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_capability_radar(
        capability_scores: Dict[str, float],
        title: str = "Model Capability Radar",
    ) -> plt.Figure:
        """Plot capability scores (0-100) on a radar chart."""
        if not capability_scores:
            raise ValueError("capability_scores cannot be empty.")

        labels = list(capability_scores.keys())
        values = [float(capability_scores[k]) for k in labels]
        values = [max(0.0, min(100.0, v)) for v in values]
        values += values[:1]

        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(7.5, 7.5), subplot_kw={"polar": True})
        ax.plot(angles, values, linewidth=2.2, color="#2563eb")
        ax.fill(angles, values, color="#60a5fa", alpha=0.25)
        ax.set_thetagrids(np.degrees(angles[:-1]), labels)
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(["20", "40", "60", "80", "100"])
        ax.set_title(title, pad=18)
        ax.grid(alpha=0.35)
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
            out = model(input_image)
            _ = out

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
    buf = BytesIO()
    canvas.print_png(buf)
    data = buf.getvalue()
    encoded = base64.b64encode(data).decode('utf-8')
    buf.close()
    plt.close(fig)  # Close the figure to free memory
    return encoded

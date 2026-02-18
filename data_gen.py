import numpy as np
import torch
from PIL import Image
import random

class VisualDataEngine:
    """Generates synthetic visual data for training and analysis."""
    
    @staticmethod
    def generate_fractal(width=256, height=256, max_iter=100):
        """Generates a Mandelbrot set fractal as a grayscale image."""
        x = np.linspace(-2.0, 0.5, width)
        y = np.linspace(-1.25, 1.25, height)
        X, Y = np.meshgrid(x, y)
        c = X + 1j * Y
        z = np.zeros_like(c)
        fractal = np.zeros(c.shape, dtype=float)
        
        for i in range(max_iter):
            mask = np.abs(z) < 10
            z[mask] = z[mask]**2 + c[mask]
            fractal[mask] += 1
            
        fractal = (fractal / max_iter * 255).astype(np.uint8)
        return Image.fromarray(fractal)

    @staticmethod
    def generate_procedural_noise(width=256, height=256, scale=10.0):
        """Generates Perlin-like procedural noise."""
        noise = np.random.randn(width // 10, height // 10)
        img = Image.fromarray((noise * 127 + 128).astype(np.uint8))
        img = img.resize((width, height), resample=Image.BILINEAR)
        return img

    @staticmethod
    def generate_geometric_sprites(count=100, size=64):
        """Generates a batch of geometric shapes (square, circle, triangle)."""
        data = []
        labels = []
        for _ in range(count):
            img = np.zeros((size, size), dtype=np.float32)
            shape_type = random.randint(0, 2) # 0: square, 1: circle, 2: triangle
            
            if shape_type == 0: # Square
                x, y = random.randint(10, size-30), random.randint(10, size-30)
                s = random.randint(10, 20)
                img[y:y+s, x:x+s] = 1.0
            elif shape_type == 1: # Circle
                cy, cx = random.randint(20, size-20), random.randint(20, size-20)
                r = random.randint(5, 15)
                y, x = np.ogrid[:size, :size]
                mask = (x - cx)**2 + (y - cy)**2 <= r**2
                img[mask] = 1.0
            else: # Triangle
                img[size//2:size//2+20, size//2-10:size//2+10] = 1.0 # Simple blob
                
            data.append(img)
            labels.append(shape_type)
            
        return torch.tensor(np.array(data)).unsqueeze(1), torch.tensor(labels)

class TemporalGenerator:
    """Generates synthetic time-series data."""
    
    @staticmethod
    def generate_sine_waves(seq_len=100, count=10, freq_range=(1, 5)):
        data = []
        for _ in range(count):
            t = np.linspace(0, 1, seq_len)
            freq = random.uniform(*freq_range)
            wave = np.sin(2 * np.pi * freq * t) + np.random.normal(0, 0.1, seq_len)
            data.append(wave)
        return torch.tensor(np.array(data), dtype=torch.float32)

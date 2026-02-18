import numpy as np
import math
from typing import List, Tuple

class SpatialNavigator:
    """Simulates a 3D environment for neural visualization."""
    
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.points: List[Tuple[float, float, float]] = []
        self.colors: List[str] = []
        self.rotation_x = 0
        self.rotation_y = 0

    def add_point(self, x: float, y: float, z: float, color="#39FF14"):
        self.points.append((x, y, z))
        self.colors.append(color)

    def rotate(self, dx: float, dy: float):
        self.rotation_x += dx
        self.rotation_y += dy

    def project_points(self) -> List[Tuple[int, int, str]]:
        """Projects 3D points to 2D for GUI rendering."""
        projected = []
        
        # Simple projection matrix simulation
        cx = math.cos(self.rotation_x)
        sx = math.sin(self.rotation_x)
        cy = math.cos(self.rotation_y)
        sy = math.sin(self.rotation_y)
        
        for i, (x, y, z) in enumerate(self.points):
            # Rotate Y
            rx = x * cy + z * sy
            rz = -x * sy + z * cy
            
            # Rotate X
            ry = y * cx - rz * sx
            rz = y * sx + rz * cx
            
            # Simple projection
            fov = 400
            scale = fov / (rz + fov + 0.001)
            px = int(rx * scale + self.width / 2)
            py = int(ry * scale + self.height / 2)
            
            projected.append((px, py, self.colors[i]))
            
        return projected

    def generate_loss_landscape(self, resolution=20):
        """Generates a 3D grid representing a simulated loss landscape."""
        self.points = []
        self.colors = []
        for i in range(resolution):
            for j in range(resolution):
                x = (i - resolution/2) * 20
                z = (j - resolution/2) * 20
                # Simulate a saddle point or complex landscape
                y = (math.sin(x/30) * math.cos(z/30)) * 50
                
                # Color based on height
                intensity = int((y + 50) / 100 * 255)
                color = f"#{intensity:02x}{255-intensity:02x}ff"
                
                self.add_point(x, y, z, color)

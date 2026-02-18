import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PersistentHomology:
    """Tracks topological features (loops, voids) in latent space.
    Simplified implementation using connectivity heuristics.
    """
    
    def __init__(self, max_dim=1):
        self.max_dim = max_dim
        
    def compute_persistence(self, point_cloud):
        """Estimate topological properties of the point cloud."""
        # point_cloud: (N, D)
        if point_cloud.size(0) < 2:
            return {'avg_nn_dist': 0.0, 'connectivity': 0.0}

        # Heuristic: Compute pairwise distances to estimate local density and gaps
        dists = torch.cdist(point_cloud, point_cloud)
        # Mask diagonal
        dists = dists + torch.eye(dists.shape[0], device=dists.device) * 1e9
        
        # k-NN distances (local connectivity)
        k = min(5, point_cloud.size(0) - 1)
        knn_vals = dists.topk(k=k, largest=False, dim=-1).values
        avg_nn_dist = knn_vals.mean()
        
        # Max-Min distance (diameter approximation)
        diameter = dists.max()
        
        # "Void" score: ratio of diameter to avg_nn_dist
        void_score = diameter / (avg_nn_dist + 1e-6)
        
        return {
            'avg_nn_dist': avg_nn_dist.item(), 
            'diameter': diameter.item(),
            'void_score': void_score.item()
        }

class ManifoldDrifter:
    """Algorithm to traverse the latent manifold smoothly."""
    
    def __init__(self, manifold_type='sphere'):
        self.manifold_type = manifold_type
        
    def interpolate(self, v1, v2, steps=10):
        """Interpolate between two vectors along the manifold geodesic."""
        # v1, v2: (D,) or (1, D)
        if v1.dim() == 2: v1 = v1.squeeze(0)
        if v2.dim() == 2: v2 = v2.squeeze(0)
        
        t = torch.linspace(0, 1, steps, device=v1.device).unsqueeze(1) # (Steps, 1)
        
        if self.manifold_type == 'sphere':
            # SLERP (Spherical Linear Interpolation)
            # Normalize inputs first just in case
            v1_n = F.normalize(v1, dim=0)
            v2_n = F.normalize(v2, dim=0)
            
            # theta = arccos(v1 . v2)
            dots = (v1_n * v2_n).sum()
            dots = dots.clamp(-1.0, 1.0)
            theta = torch.acos(dots)
            
            sin_theta = torch.sin(theta)
            if sin_theta.abs() < 1e-4:
                # Parallel vectors, linear interp
                return (1 - t) * v1 + t * v2
                
            s0 = torch.sin((1 - t) * theta) / sin_theta
            s1 = torch.sin(t * theta) / sin_theta
            
            # Scale result by interpolated magnitude
            mag1 = v1.norm()
            mag2 = v2.norm()
            mag_t = (1 - t) * mag1 + t * mag2
            
            res_dir = s0 * v1_n + s1 * v2_n
            return res_dir * mag_t
            
        elif self.manifold_type == 'poincare':
            # Hyperbolic interpolation (Geodesic approximation)
            # Simple linear in tangent space usually, or linear in embedding is okay approximation for visualization
             return (1 - t) * v1 + t * v2
             
        else: # Euclidean
            return (1 - t) * v1 + t * v2

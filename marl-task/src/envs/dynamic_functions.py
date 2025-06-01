import torch
import math

class DynamicSphere:
    """Sphere function with a moving optimum."""
    def __init__(self, dim: int, shift_speed: float = 0.1):
        self.dim = dim
        self.shift_speed = shift_speed
        self.optimum = torch.zeros(dim) 
        self.time = 0

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.time += 1
        shift = torch.tensor([
            math.sin(self.time * self.shift_speed),
            math.cos(self.time * self.shift_speed),
            *[0.0] * (self.dim - 2) 
        ], dtype=x.dtype, device=x.device)
        
        shifted_x = x - shift
        return -torch.sum(shifted_x ** 2, dim=-1) 

    def reset(self):
        self.time = 0
        self.optimum = torch.zeros(self.dim)


class DynamicRastrigin:
    """Rastrigin function with oscillating amplitude."""
    def __init__(self, dim: int, frequency: float = 0.05):
        self.dim = dim
        self.frequency = frequency
        self.time = 0

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.time += 1
        A = 10 * (1 + 0.5 * math.sin(self.time * self.frequency))  # Time-varying amplitude
        
        return -(A * self.dim + torch.sum(x**2 - A * torch.cos(2 * math.pi * x), dim=-1))

    def reset(self):
        self.time = 0


class DynamicEggHolder:
    """Eggholder function with a rotating coordinate system."""
    def __init__(self, dim: int = 2, rotation_speed: float = 0.01):
        assert dim == 2, "DynamicEggholder only works in 2D"
        self.dim = dim
        self.rotation_speed = rotation_speed
        self.time = 0

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.time += 1
        theta = self.time * self.rotation_speed
        
        x_rot = x.clone()
        x_rot[..., 0] = x[..., 0] * math.cos(theta) - x[..., 1] * math.sin(theta)
        x_rot[..., 1] = x[..., 0] * math.sin(theta) + x[..., 1] * math.cos(theta)
        
        x_pairs = x_rot.view(*x_rot.shape[:-1], -1, 2)
        x_i, x_j = x_pairs[..., 0], x_pairs[..., 1]
        
        term1 = -(x_j + 47) * torch.sin(torch.sqrt(torch.abs(x_j + x_i / 2 + 47)))
        term2 = -x_i * torch.sin(torch.sqrt(torch.abs(x_i - (x_j + 47))))
        return (term1 + term2).sum(dim=-1)

    def reset(self):
        self.time = 0
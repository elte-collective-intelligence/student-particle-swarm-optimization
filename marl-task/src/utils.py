import torch
import torch.nn as nn

class LandscapeWrapper:
    def __init__(self, function, dim):
        self.function = function
        self.dim = dim
    
    def __call__(self, x):
        return self.function(x)
    
    def reset(self):
        pass

class PSOObservationWrapper(nn.Module):
    def __init__(self):
        super(PSOObservationWrapper, self).__init__()
    
    def forward(self, avg_pos, avg_vel)  -> torch.Tensor:
        return torch.cat([avg_pos, avg_vel], dim=-1)

class PSOActionExtractor(nn.Module):
    def __init__(self, dim):
        super(PSOActionExtractor, self).__init__()
        self.dim = dim

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        if torch.isnan(action[0]).any() or torch.isnan(action[1]).any():
            raise ValueError("Action contains NaN values")
        inertia = (action[0][..., :self.dim], action[1][..., :self.dim])
        cognitive = (action[0][..., self.dim:2*self.dim], action[1][..., self.dim:2*self.dim])
        social = (action[0][..., 2*self.dim:], action[1][..., 2*self.dim:])
        return inertia + cognitive + social
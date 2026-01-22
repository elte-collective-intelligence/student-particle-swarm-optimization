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

    def forward(self, avg_pos, avg_vel) -> torch.Tensor:
        return torch.cat([avg_pos, avg_vel], dim=-1)


class PSOActionExtractor(nn.Module):
    """
    Extract and transform PSO action parameters from network output.

    The network outputs (loc, scale) parameters for Normal distributions.
    We transform the sampled actions to proper PSO coefficient ranges:
    - inertia: typically 0.4-0.9 (centered at 0.7)
    - cognitive: typically 1.0-2.5 (centered at 1.5)
    - social: typically 1.0-2.5 (centered at 1.5)
    """

    def __init__(self, dim, transform_actions=True):
        super(PSOActionExtractor, self).__init__()
        self.dim = dim
        self.transform_actions = transform_actions

        # Action transformation parameters (mean, scale)
        # Network outputs ~N(0,1), we transform to proper ranges
        self.register_buffer(
            "inertia_offset", torch.tensor(0.7)
        )  # Inertia: 0.7 + 0.2*x -> [0.3, 1.1]
        self.register_buffer("inertia_scale", torch.tensor(0.2))
        self.register_buffer(
            "cognitive_offset", torch.tensor(1.5)
        )  # Cognitive: 1.5 + 0.5*x -> [0.5, 2.5]
        self.register_buffer("cognitive_scale", torch.tensor(0.5))
        self.register_buffer(
            "social_offset", torch.tensor(1.5)
        )  # Social: 1.5 + 0.5*x -> [0.5, 2.5]
        self.register_buffer("social_scale", torch.tensor(0.5))

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        if torch.isnan(action[0]).any() or torch.isnan(action[1]).any():
            raise ValueError("Action contains NaN values")

        # Split into (loc, scale) pairs for each action type
        loc, scale = action[0], action[1]

        inertia_loc = loc[..., : self.dim]
        inertia_scale = scale[..., : self.dim]
        cognitive_loc = loc[..., self.dim : 2 * self.dim]
        cognitive_scale = scale[..., self.dim : 2 * self.dim]
        social_loc = loc[..., 2 * self.dim :]
        social_scale = scale[..., 2 * self.dim :]

        if self.transform_actions:
            # Transform loc to proper PSO ranges
            inertia_loc = self.inertia_offset + self.inertia_scale * inertia_loc
            cognitive_loc = self.cognitive_offset + self.cognitive_scale * cognitive_loc
            social_loc = self.social_offset + self.social_scale * social_loc

            # Scale the standard deviations appropriately
            inertia_scale = self.inertia_scale * scale[..., : self.dim].abs()
            cognitive_scale = (
                self.cognitive_scale * scale[..., self.dim : 2 * self.dim].abs()
            )
            social_scale = self.social_scale * scale[..., 2 * self.dim :].abs()

        return (
            inertia_loc,
            inertia_scale,
            cognitive_loc,
            cognitive_scale,
            social_loc,
            social_scale,
        )

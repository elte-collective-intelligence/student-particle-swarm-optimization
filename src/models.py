import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as d


class DimAgnosticNet(nn.Module):
    """
    Per-dimension policy network.

    Input:  avg_pos, avg_vel  — each [B, A, D]
    Output: (loc, scale) for inertia, cognitive, social — each [B, A, D]

    Action ranges (built-in transformation):
      inertia:   0.7 ± 0.2
      cognitive: 1.5 ± 0.5
      social:    1.5 ± 0.5
    """

    def __init__(self, hidden_size: int = 64):
        super().__init__()
        # Encode (avg_pos_d, avg_vel_d) per dimension
        self.encoder = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        # Compress global mean into context
        self.context = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        # 6 outputs per dim: (loc, scale) × (inertia, cognitive, social)
        self.decoder = nn.Linear(hidden_size * 2, 6)

        # PSO coefficient offsets/scales (mirrors PSOActionExtractor)
        self.register_buffer("inertia_offset", torch.tensor(0.7))
        self.register_buffer("inertia_scale", torch.tensor(0.2))
        self.register_buffer("cog_offset", torch.tensor(1.5))
        self.register_buffer("cog_scale", torch.tensor(0.5))
        self.register_buffer("soc_offset", torch.tensor(1.5))
        self.register_buffer("soc_scale", torch.tensor(0.5))

    def forward(self, avg_pos: torch.Tensor, avg_vel: torch.Tensor):
        """
        Returns 6 tensors of shape [B, A, D]:
          inertia_loc, inertia_scale, cog_loc, cog_scale, soc_loc, soc_scale
        """
        # [B, A, D, 2]
        x = torch.stack([avg_pos, avg_vel], dim=-1)

        # Per-dim local features: [..., D, H]
        local = self.encoder(x)

        # Global context: mean over D (dim=-2) → [..., H] → expand to [..., D, H]
        ctx = self.context(local.mean(dim=-2)).unsqueeze(-2).expand_as(local)

        # Decode local + global: [..., D, 6]
        out = self.decoder(torch.cat([local, ctx], dim=-1))

        # Split and apply PSO range transformations
        inertia_loc = self.inertia_offset + self.inertia_scale * out[..., 0]
        inertia_scale = self.inertia_scale * (F.softplus(out[..., 1]) + 1e-4)
        cog_loc = self.cog_offset + self.cog_scale * out[..., 2]
        cog_scale = self.cog_scale * (F.softplus(out[..., 3]) + 1e-4)
        soc_loc = self.soc_offset + self.soc_scale * out[..., 4]
        soc_scale = self.soc_scale * (F.softplus(out[..., 5]) + 1e-4)

        return inertia_loc, inertia_scale, cog_loc, cog_scale, soc_loc, soc_scale


class DimAgnosticCritic(nn.Module):
    """
    Per-dimension value network.

    Input:  avg_pos, avg_vel — each [B, A, D]
    Output: state_value      — [B, A, 1]
    """

    def __init__(self, hidden_size: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, avg_pos: torch.Tensor, avg_vel: torch.Tensor) -> torch.Tensor:
        x = torch.stack([avg_pos, avg_vel], dim=-1)  # [..., D, 2]
        h = self.encoder(x).mean(dim=-2)  # [..., H]  (mean over D)
        return self.value_head(h)  # [..., 1]


def build_curriculum_policy(net: "DimAgnosticNet", env, device):
    """
    Wrap a DimAgnosticNet in a TorchRL ProbabilisticActor.

    Single authoritative definition — imported by curriculum_train.py,
    eval/generalization.py, and curriculum_eval_gif.py to avoid divergence.
    """
    from tensordict.nn import TensorDictModule
    from torchrl.modules import ProbabilisticActor
    from tensordict.nn.distributions import CompositeDistribution

    return ProbabilisticActor(
        TensorDictModule(
            net,
            in_keys=["avg_pos", "avg_vel"],
            out_keys=[
                ("params", "inertia", "loc"),
                ("params", "inertia", "scale"),
                ("params", "cognitive", "loc"),
                ("params", "cognitive", "scale"),
                ("params", "social", "loc"),
                ("params", "social", "scale"),
            ],
        ),
        in_keys=["params"],
        spec=env.action_spec,
        out_keys=["inertia", "cognitive", "social"],
        distribution_class=CompositeDistribution,
        distribution_kwargs={
            "distribution_map": {
                "inertia": d.Normal,
                "cognitive": d.Normal,
                "social": d.Normal,
            }
        },
        return_log_prob=True,
    )

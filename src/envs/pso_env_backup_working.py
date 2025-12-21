import torch
from torchrl.envs import EnvBase
from tensordict import TensorDict
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec


class PSOEnv(EnvBase):
    """
    TorchRL-compatible PSO Environment with STATIC + DYNAMIC objectives
    Option 4: Moving Center + Sinusoidal Drift
    """

    def __init__(
        self,
        landscape_fn,
        num_agents=10,
        dim=2,
        device=None,
        batch_size=[32],
        max_steps=100,
        dynamic=False,
        drift_amplitude=50.0,
        drift_frequency=0.05,
        center_velocity=5.0,
    ):
        super().__init__(device=device, batch_size=batch_size)

        self.num_agents = num_agents
        self.dim = dim
        self.base_landscape_fn = landscape_fn
        self.max_steps = max_steps
        self.dynamic = dynamic

        # Dynamic parameters
        self.drift_amplitude = drift_amplitude
        self.drift_frequency = drift_frequency
        self.center_velocity = center_velocity

        self.search_bounds = (-512, 512)
        self.neighborhood_radius = 1.0

        # State
        self.positions = None
        self.velocities = None
        self.personal_best_positions = None
        self.personal_best_values = None
        self.global_best_position = None
        self.global_best_value = None

        self.step_count = 0
        self.time = 0.0

        # Dynamic center
        self.dynamic_center = torch.zeros(dim, device=device)

        B = self.batch_size[0]

        # ================= TORCHRL SPECS =================

        self.observation_spec = CompositeSpec(
            positions=UnboundedContinuousTensorSpec((B, num_agents, dim), device=device),
            velocities=UnboundedContinuousTensorSpec((B, num_agents, dim), device=device),
            personal_best_positions=UnboundedContinuousTensorSpec((B, num_agents, dim), device=device),
            personal_best_values=UnboundedContinuousTensorSpec((B, num_agents), device=device),
            neighbor_summary=UnboundedContinuousTensorSpec((B, num_agents, dim), device=device),
            shape=self.batch_size,
            device=device,
        )

        self.action_spec = CompositeSpec(
            action=UnboundedContinuousTensorSpec((B, num_agents, 3), device=device),
            shape=self.batch_size,
            device=device,
        )

        self.reward_spec = CompositeSpec(
            reward=UnboundedContinuousTensorSpec((B, num_agents), device=device),
            shape=self.batch_size,
            device=device,
        )

    # ================= RESET =================

    def _reset(self, tensordict=None):
        B = self.batch_size[0]
        device = self.device

        self.positions = torch.rand(B, self.num_agents, self.dim, device=device)
        self.positions = self.positions * (self.search_bounds[1] - self.search_bounds[0]) + self.search_bounds[0]

        self.velocities = torch.randn(B, self.num_agents, self.dim, device=device) * 0.1

        self.personal_best_positions = self.positions.clone()

        values = self._evaluate(self.positions.reshape(-1, self.dim)).view(B, self.num_agents)
        self.personal_best_values = values.clone()

        self.global_best_value, best_idx = values.min(dim=1)
        self.global_best_position = torch.stack(
            [self.positions[b, best_idx[b]] for b in range(B)], dim=0
        )

        self.step_count = 0
        self.time = 0.0
        self.dynamic_center.zero_()

        neighbor_summary = self._compute_neighbor_summary()

        return TensorDict(
            {
                "positions": self.positions,
                "velocities": self.velocities,
                "personal_best_positions": self.personal_best_positions,
                "personal_best_values": self.personal_best_values,
                "neighbor_summary": neighbor_summary,
                "reward": torch.zeros(B, self.num_agents, device=device),
                "done": torch.zeros(B, 1, dtype=torch.bool, device=device),
            },
            batch_size=self.batch_size,
        )

    # ================= STEP =================

    def _step(self, tensordict):
        B = self.batch_size[0]
        device = self.device
        self.step_count += 1
        self.time += 1.0

        actions = tensordict["action"]
        inertia = torch.sigmoid(actions[..., 0]).unsqueeze(-1)
        cognitive = torch.sigmoid(actions[..., 1]).unsqueeze(-1)
        social = torch.sigmoid(actions[..., 2]).unsqueeze(-1)

        neighbor_summary = self._compute_neighbor_summary()

        self.velocities = (
            inertia * self.velocities
            + cognitive * (self.personal_best_positions - self.positions)
            + social * (neighbor_summary - self.positions)
        )
        self.velocities = torch.clamp(self.velocities, -10.0, 10.0)

        self.positions = torch.clamp(
            self.positions + self.velocities,
            self.search_bounds[0],
            self.search_bounds[1],
        )

        values = self._evaluate(self.positions.reshape(-1, self.dim)).view(B, self.num_agents)
        rewards = self._compute_rewards(values)

        improved = values < self.personal_best_values
        self.personal_best_positions = torch.where(improved.unsqueeze(-1), self.positions, self.personal_best_positions)
        self.personal_best_values = torch.where(improved, values, self.personal_best_values)

        curr_best, best_idx = values.min(dim=1)
        improved_global = curr_best < self.global_best_value

        if improved_global.any():
            self.global_best_value = torch.where(improved_global, curr_best, self.global_best_value)
            self.global_best_position = torch.stack(
                [
                    self.positions[b, best_idx[b]] if improved_global[b] else self.global_best_position[b]
                    for b in range(B)
                ],
                dim=0,
            )

        done = torch.tensor(
            [self.step_count >= self.max_steps] * B, device=device, dtype=torch.bool
        ).view(B, 1)

        neighbor_summary = self._compute_neighbor_summary()

        return TensorDict(
            {
                "positions": self.positions,
                "velocities": self.velocities,
                "personal_best_positions": self.personal_best_positions,
                "personal_best_values": self.personal_best_values,
                "neighbor_summary": neighbor_summary,
                "reward": rewards,
                "done": done,
            },
            batch_size=self.batch_size,
        )

    # ================= DYNAMIC OBJECTIVE =================

    def _evaluate(self, x):
        if not self.dynamic:
            return self.base_landscape_fn(x)

        # --- Moving center ---
        self.dynamic_center += (
            torch.randn_like(self.dynamic_center) * self.center_velocity * 0.01
        )

        # --- Sinusoidal drift ---
        t = torch.tensor(self.time, device=x.device)
        drift = self.drift_amplitude * torch.sin(self.drift_frequency * t)


        shifted_x = x - self.dynamic_center + drift
        return self.base_landscape_fn(shifted_x)

    # ================= REWARDS =================

    def _compute_rewards(self, values):
        improvement_reward = (self.personal_best_values - values) * 10.0
        diversity_reward = self._compute_diversity_reward()
        global_reward = self._compute_global_reward(values)

        return improvement_reward + 0.1 * diversity_reward + 0.5 * global_reward

    def _compute_diversity_reward(self):
        B, N, _ = self.positions.shape
        out = torch.zeros(B, N, device=self.device)
        for b in range(B):
            centroid = self.positions[b].mean(dim=0)
            dist = torch.norm(self.positions[b] - centroid, dim=1)
            m = dist.mean()
            out[b] = -torch.abs(dist - m) / (m + 1e-6)
        return out

    def _compute_global_reward(self, values):
        B, N = values.shape
        out = torch.zeros(B, N, device=self.device)
        curr_best, _ = values.min(dim=1)
        delta = self.global_best_value - curr_best
        for b in range(B):
            if delta[b] > 0:
                out[b] = delta[b] * 10.0 / N
        return out

    # ================= NEIGHBORS =================

    def _compute_neighbor_summary(self):
        B, N, D = self.positions.shape
        out = torch.zeros_like(self.positions)
        for b in range(B):
            dist = torch.cdist(self.positions[b], self.positions[b])
            mask = (dist <= self.neighborhood_radius).float()
            mask -= torch.eye(N, device=self.device)
            mask = mask.clamp_min(0.0)
            count = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            out[b] = (mask @ self.positions[b]) / count
        return out

    # ================= METRICS =================

    def get_metrics(self):
        return {
            "best_fitness": float(self.personal_best_values.min()),
            "diversity": float(torch.pdist(self.positions[0]).mean()),
            "dynamic_center_norm": float(torch.norm(self.dynamic_center)),
        }

    def _set_seed(self, seed):
        torch.manual_seed(seed)
        return seed

    @property
    def reward_key(self):
        return "reward"

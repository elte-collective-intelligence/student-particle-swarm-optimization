import torch
from torchrl.envs import EnvBase
from tensordict import TensorDict
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec


class PSOEnv(EnvBase):
    """
    TorchRL-compatible PSO-as-MARL environment.

    - Agents (particles) output 3 continuous actions -> inertia, cognitive, social weights (sigmoid).
    - Observations: position, velocity, personal best, neighbor summary.
    - Rewards: improvement + diversity + global progress + penalties (collapse, stagnation, overspeed).
    - Supports optional dynamic landscape drift (simple sinusoidal drift).
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
        drift_amplitude=20.0,
        drift_period=50.0,
        neighborhood_radius=1.0,
        v_max=10.0,
        overspeed_threshold=5.0,
        collapse_threshold=50.0,
        stagnation_patience=10,
    ):
        super().__init__(device=device, batch_size=batch_size)

        self.num_agents = int(num_agents)
        self.dim = int(dim)
        self.landscape_fn = landscape_fn
        self.max_steps = int(max_steps)

        self.search_bounds = (-512.0, 512.0)

        # neighborhood
        self.neighborhood_radius = float(neighborhood_radius)

        # dynamics
        self.dynamic = bool(dynamic)
        self.drift_amplitude = float(drift_amplitude)
        self.drift_period = float(drift_period)

        # safety / shaping
        self.v_max = float(v_max)
        self.overspeed_threshold = float(overspeed_threshold)
        self.collapse_threshold = float(collapse_threshold)
        self.stagnation_patience = int(stagnation_patience)

        # State variables
        self.positions = None
        self.velocities = None
        self.personal_best_positions = None
        self.personal_best_values = None
        self.global_best_position = None
        self.global_best_value = None
        self.step_count = 0

        # Track stagnation per batch env (B,)
        self.no_improve_steps = None

        B = int(self.batch_size[0])

        # -----------------------------
        # TorchRL Specs
        # NOTE: this project uses fixed B (batch_size[0]) for simplicity
        # -----------------------------
        self.observation_spec = CompositeSpec(
            positions=UnboundedContinuousTensorSpec(shape=(B, self.num_agents, self.dim), device=device),
            velocities=UnboundedContinuousTensorSpec(shape=(B, self.num_agents, self.dim), device=device),
            personal_best_positions=UnboundedContinuousTensorSpec(shape=(B, self.num_agents, self.dim), device=device),
            personal_best_values=UnboundedContinuousTensorSpec(shape=(B, self.num_agents), device=device),
            neighbor_summary=UnboundedContinuousTensorSpec(shape=(B, self.num_agents, self.dim), device=device),
            shape=self.batch_size,
            device=device,
        )

        self.action_spec = CompositeSpec(
            action=UnboundedContinuousTensorSpec(shape=(B, self.num_agents, 3), device=device),
            shape=self.batch_size,
            device=device,
        )

        self.reward_spec = CompositeSpec(
            reward=UnboundedContinuousTensorSpec(shape=(B, self.num_agents), device=device),
            shape=self.batch_size,
            device=device,
        )

    # -----------------------------
    # Evaluation (static/dynamic)
    # -----------------------------
    def _evaluate(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., dim)
        returns: (...,)
        If dynamic=True, apply a simple sinusoidal drift in input space.
        """
        if not self.dynamic:
            return self.landscape_fn(x)

        # Make a tensor time scalar
        t = torch.tensor(float(self.step_count), device=x.device, dtype=x.dtype)

        # Drift per dimension (dim,)
        # sin(2*pi*t/period + phase_d)
        phases = torch.linspace(0.0, 1.0, steps=self.dim, device=x.device, dtype=x.dtype) * (2.0 * torch.pi)
        drift = self.drift_amplitude * torch.sin((2.0 * torch.pi * t / self.drift_period) + phases)

        return self.landscape_fn(x + drift)

    # ============================
    # RESET
    # ============================
    def _reset(self, tensordict=None):
        B = int(self.batch_size[0])
        device = self.device

        self.positions = torch.rand(B, self.num_agents, self.dim, device=device)
        self.positions = self.positions * (self.search_bounds[1] - self.search_bounds[0]) + self.search_bounds[0]

        self.velocities = torch.randn(B, self.num_agents, self.dim, device=device) * 0.1

        self.personal_best_positions = self.positions.clone()

        values = self._evaluate(self.positions.reshape(-1, self.dim)).view(B, self.num_agents)
        self.personal_best_values = values.clone()

        self.global_best_value, best_idx = values.min(dim=1)
        self.global_best_position = torch.stack(
            [self.positions[b, best_idx[b]] for b in range(B)],
            dim=0,
        )

        self.step_count = 0
        self.no_improve_steps = torch.zeros(B, device=device, dtype=torch.long)

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

    # ============================
    # STEP
    # ============================
    def _step(self, tensordict):
        B = int(self.batch_size[0])
        device = self.device
        self.step_count += 1

        actions = tensordict["action"]
        inertia = torch.sigmoid(actions[..., 0]).unsqueeze(-1)
        cognitive = torch.sigmoid(actions[..., 1]).unsqueeze(-1)
        social = torch.sigmoid(actions[..., 2]).unsqueeze(-1)

        neighbor_summary = self._compute_neighbor_summary()
        cognitive_component = cognitive * (self.personal_best_positions - self.positions)
        social_component = social * (neighbor_summary - self.positions)

        self.velocities = inertia * self.velocities + cognitive_component + social_component
        self.velocities = torch.clamp(self.velocities, -self.v_max, self.v_max)

        self.positions = self.positions + self.velocities
        self.positions = torch.clamp(self.positions, self.search_bounds[0], self.search_bounds[1])

        new_values = self._evaluate(self.positions.reshape(-1, self.dim)).view(B, self.num_agents)

        # --- pbest updates ---
        improved_mask = new_values < self.personal_best_values
        any_improved_env = improved_mask.any(dim=1)  # (B,)

        self.personal_best_positions = torch.where(
            improved_mask.unsqueeze(-1), self.positions, self.personal_best_positions
        )
        self.personal_best_values = torch.where(
            improved_mask, new_values, self.personal_best_values
        )

        # --- gbest updates ---
        current_best, best_idx = new_values.min(dim=1)
        global_improved = current_best < self.global_best_value
        if global_improved.any():
            self.global_best_value = torch.where(global_improved, current_best, self.global_best_value)
            self.global_best_position = torch.stack(
                [
                    self.positions[b, best_idx[b]] if global_improved[b] else self.global_best_position[b]
                    for b in range(B)
                ],
                dim=0,
            )

        # --- stagnation counter (per env) ---
        # reset if any particle improved OR global improved, else increment
        improved_env = any_improved_env | global_improved
        self.no_improve_steps = torch.where(
            improved_env,
            torch.zeros_like(self.no_improve_steps),
            self.no_improve_steps + 1,
        )

        # --- reward ---
        rewards = self._compute_rewards(new_values)

        done = torch.tensor(
            [self.step_count >= self.max_steps] * B,
            device=device,
            dtype=torch.bool,
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

    # ============================
    # HELPERS
    # ============================
    def _compute_neighbor_summary(self):
        B, N, D = self.positions.shape
        device = self.positions.device
        neighbor_summary = torch.zeros_like(self.positions)

        for b in range(B):
            pos = self.positions[b]
            distances = torch.cdist(pos, pos)
            mask = (distances <= self.neighborhood_radius).float()
            mask -= torch.eye(N, device=device)
            mask = mask.clamp_min(0.0)
            counts = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            neighbor_summary[b] = (mask @ pos) / counts

        return neighbor_summary

    def _mean_pairwise_distance(self) -> torch.Tensor:
        """
        Returns (B,) mean pairwise distance per batch env.
        """
        B, N, _ = self.positions.shape
        out = torch.zeros(B, device=self.device)
        for b in range(B):
            if N > 1:
                out[b] = torch.pdist(self.positions[b]).mean()
            else:
                out[b] = 0.0
        return out

    def _compute_rewards(self, new_values):
        """
        Reward shaping:
        + improvement over personal best
        + global progress encouragement
        + diversity reward (existing)
        - collapse penalty (too clustered)
        - stagnation penalty (no improvements for many steps)
        - overspeed penalty (very large velocity)
        """
        B, N = new_values.shape

        # 1) improvement (per-agent)
        improvement = self.personal_best_values - new_values
        improvement_reward = improvement * 10.0

        # 2) diversity reward (per-agent, already implemented)
        diversity_reward = self._compute_diversity_reward()

        # 3) global reward (per-agent)
        global_reward = self._compute_global_reward(new_values)

        # 4) collapse penalty (per-env -> broadcast to agents)
        # if mean pairwise distance < collapse_threshold => penalty negative
        mean_pd = self._mean_pairwise_distance()  # (B,)
        # scaled penalty in [-1, 0]
        collapse_pen_env = -torch.clamp((self.collapse_threshold - mean_pd) / (self.collapse_threshold + 1e-6), min=0.0, max=1.0)
        collapse_penalty = collapse_pen_env.unsqueeze(-1).expand(B, N)  # (B,N)

        # 5) stagnation penalty (per-env -> broadcast)
        stagnating = (self.no_improve_steps >= self.stagnation_patience).float()  # (B,)
        stagnation_penalty = (-0.2 * stagnating).unsqueeze(-1).expand(B, N)

        # 6) overspeed penalty (per-agent)
        speed = torch.norm(self.velocities, dim=-1)  # (B,N)
        overspeed = torch.clamp(speed - self.overspeed_threshold, min=0.0)
        overspeed_penalty = -0.05 * overspeed  # mild penalty

        total = (
            improvement_reward
            + 0.1 * diversity_reward
            + 0.5 * global_reward
            + 0.5 * collapse_penalty
            + stagnation_penalty
            + overspeed_penalty
        )
        return total

    def _compute_diversity_reward(self):
        B, N, _ = self.positions.shape
        diversity_reward = torch.zeros(B, N, device=self.device)

        for b in range(B):
            centroid = self.positions[b].mean(dim=0)
            distances = torch.norm(self.positions[b] - centroid, dim=1)
            mean_dist = distances.mean()
            diversity_reward[b] = -torch.abs(distances - mean_dist) / (mean_dist + 1e-6)

        return diversity_reward

    def _compute_global_reward(self, new_values):
        B, N = new_values.shape
        global_reward = torch.zeros(B, N, device=self.device)

        current_best, _ = new_values.min(dim=1)
        improvement = self.global_best_value - current_best

        for b in range(B):
            if improvement[b] > 0:
                global_reward[b] = improvement[b] * 10.0 / N

        return global_reward

    def _set_seed(self, seed):
        torch.manual_seed(seed)
        return seed

    @property
    def reward_key(self):
        return "reward"

    # ============================
    # METRICS
    # ============================
    def get_metrics(self):
        """
        Returns scalar metrics useful for logging/reporting.
        """
        if self.positions is None:
            return {
                "best_fitness": 0.0,
                "diversity": 0.0,
                "mean_speed": 0.0,
                "global_best": 0.0,
                "mean_pbest": 0.0,
                "stagnation_frac": 0.0,
            }

        B = int(self.batch_size[0])

        best_fitness = float(self.personal_best_values.min().detach().cpu())
        diversity = float(self._mean_pairwise_distance().mean().detach().cpu())
        mean_speed = float(torch.norm(self.velocities, dim=-1).mean().detach().cpu())

        global_best = float(self.global_best_value.min().detach().cpu())
        mean_pbest = float(self.personal_best_values.mean().detach().cpu())

        stagnation_frac = float((self.no_improve_steps >= self.stagnation_patience).float().mean().detach().cpu())

        return {
            "best_fitness": best_fitness,
            "diversity": diversity,
            "mean_speed": mean_speed,
            "global_best": global_best,
            "mean_pbest": mean_pbest,
            "stagnation_frac": stagnation_frac,
        }

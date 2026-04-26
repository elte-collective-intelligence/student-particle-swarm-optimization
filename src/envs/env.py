from typing import Any, Mapping, Optional

import torch
from torchrl.envs import EnvBase
from torchrl.data import Composite, Unbounded, Categorical
from tensordict import TensorDict

from envs.topology import Topology, create_topology


class PSOEnv(EnvBase):
    def __init__(
        self,
        landscape,
        num_agents: int,
        device: torch.device,
        batch_size,
        delta: float = 1.0,
        topology_config: Optional[Mapping[str, Any]] = None,
        use_topology_social: bool = True,
        run_type_checks: bool = False,
    ):
        super().__init__(
            device=device, batch_size=batch_size, run_type_checks=run_type_checks
        )

        self.landscape = landscape
        self.num_agents = num_agents
        self.delta = delta
        self.topology_config = topology_config
        self.use_topology_social = bool(use_topology_social)
        self.topology: Optional[Topology] = None
        self.step_count = 0

        self.scores = None
        self.positions = None
        self.velocities = None
        self.personal_best_pos = None
        self.personal_best_scores = None
        self.neighborhood_best_pos = None
        self.neighborhood_best_scores = None
        self.avg_pos = None
        self.avg_vel = None

        self.observation_spec = Composite(
            {
                "scores": Unbounded(self.batch_size + (num_agents,), device=device),
                "positions": Unbounded(
                    self.batch_size + (num_agents, landscape.dim), device=device
                ),
                "velocities": Unbounded(
                    self.batch_size + (num_agents, landscape.dim), device=device
                ),
                "avg_pos": Unbounded(
                    self.batch_size + (num_agents, landscape.dim), device=device
                ),
                "avg_vel": Unbounded(
                    self.batch_size + (num_agents, landscape.dim), device=device
                ),
                "personal_best_pos": Unbounded(
                    self.batch_size + (num_agents, landscape.dim), device=device
                ),
                "personal_best_scores": Unbounded(
                    self.batch_size + (num_agents,), device=device
                ),
                "neighborhood_best_pos": Unbounded(
                    self.batch_size + (num_agents, landscape.dim), device=device
                ),
                "neighborhood_best_scores": Unbounded(
                    self.batch_size + (num_agents,), device=device
                ),
            },
            shape=torch.Size(batch_size),
        )

        self.action_spec = Composite(
            {
                "inertia": Unbounded(
                    self.batch_size + (num_agents, landscape.dim), device=device
                ),
                "cognitive": Unbounded(
                    self.batch_size + (num_agents, landscape.dim), device=device
                ),
                "social": Unbounded(
                    self.batch_size + (num_agents, landscape.dim), device=device
                ),
            },
            shape=torch.Size(batch_size),
        )

        self.reward_spec = Composite(
            {
                "agents": Composite(
                    {
                        "reward": Unbounded(
                            self.batch_size + (num_agents,), device=device
                        )
                    }
                )
            },
            shape=self.batch_size,
        )
        self.done_spec = Composite(
            {
                "agents": Composite(
                    {
                        "done": Categorical(
                            n=2,
                            shape=self.batch_size + (num_agents,),
                            dtype=torch.bool,
                            device=device,
                        ),
                        "terminated": Categorical(
                            n=2,
                            shape=self.batch_size + (num_agents,),
                            dtype=torch.bool,
                            device=device,
                        ),
                        "truncated": Categorical(
                            n=2,
                            shape=self.batch_size + (num_agents,),
                            dtype=torch.bool,
                            device=device,
                        ),
                    }
                )
            },
            shape=self.batch_size,
        )

    def initialize_topology(self, positions: Optional[torch.Tensor] = None) -> Topology:
        if positions is None:
            positions = self.positions

        self.topology = create_topology(
            num_particles=self.num_agents,
            cfg=self.topology_config,
            positions=positions,
            initialize=positions is not None,
        )
        return self.topology

    def update_topology(self) -> bool:
        if self.topology is None:
            self.initialize_topology(self.positions)

        if self.topology is None:
            return False

        return self.topology.update(self.positions, step=self.step_count)

    def compute_neighborhood_best(self) -> None:
        if self.topology is None:
            self.neighborhood_best_pos = self.personal_best_pos.clone()
            self.neighborhood_best_scores = self.personal_best_scores.clone()
            return

        adjacency = self.topology.get_adjacency(device=self.device)
        adjacency = adjacency.to(dtype=torch.bool)

        score_grid = self.personal_best_scores.unsqueeze(-2).expand(
            self.batch_size + (self.num_agents, self.num_agents)
        )
        score_grid = score_grid.masked_fill(~adjacency.unsqueeze(0), float("-inf"))
        best_indices = score_grid.argmax(dim=-1)

        self.neighborhood_best_scores = self.personal_best_scores.gather(
            dim=-1, index=best_indices
        )
        best_indices = best_indices.unsqueeze(-1).expand(
            self.batch_size + (self.num_agents, self.landscape.dim)
        )
        self.neighborhood_best_pos = self.personal_best_pos.gather(
            dim=-2, index=best_indices
        )

    def get_social_signal(self) -> torch.Tensor:
        if self.use_topology_social and self.neighborhood_best_pos is not None:
            return self.neighborhood_best_pos - self.positions
        return self.avg_pos

    def _stable_improvement_reward(
        self, delta: torch.Tensor, baseline: torch.Tensor
    ) -> torch.Tensor:
        scale = baseline.abs().clamp_min(1.0)
        return torch.tanh(delta / scale)

    def _compute_reward(
        self,
        last_scores: torch.Tensor,
        last_neighborhood_best_scores: torch.Tensor,
    ) -> torch.Tensor:
        personal_delta = self.scores - last_scores
        neighborhood_delta = self.neighborhood_best_scores - last_neighborhood_best_scores

        personal_reward = self._stable_improvement_reward(personal_delta, last_scores)
        neighborhood_reward = self._stable_improvement_reward(
            neighborhood_delta, last_neighborhood_best_scores
        )

        reward = 0.5 * personal_reward + 0.5 * neighborhood_reward
        return torch.nan_to_num(reward)

    def _reset(self, params=None) -> TensorDict:
        # Reset landscape function
        self.landscape.reset()
        self.step_count = 0
        # Inicialize agents and scores
        self.positions = torch.randn(
            self.batch_size + (self.num_agents, self.landscape.dim), device=self.device
        )
        self.velocities = torch.zeros_like(self.positions)
        self.personal_best_pos = self.positions.clone()
        self.personal_best_scores = self.landscape(self.personal_best_pos)
        self.initialize_topology(self.positions)
        self.compute_neighborhood_best()

        self.avg_pos, self.avg_vel = get_neighborhood_avg(
            self.positions, self.velocities, self.delta
        )

        self.scores = self.landscape(self.positions)
        # may need to call .clone() on fields
        return TensorDict(
            {
                "scores": self.scores,
                "positions": self.positions,
                "velocities": self.velocities,
                "personal_best_pos": self.personal_best_pos,
                "personal_best_scores": self.personal_best_scores,
                "neighborhood_best_pos": self.neighborhood_best_pos,
                "neighborhood_best_scores": self.neighborhood_best_scores,
                "avg_pos": self.avg_pos,
                "avg_vel": self.avg_vel,
                ("agents", "reward"): torch.zeros(
                    self.batch_size + (self.num_agents,), device=self.device
                ),
                ("agents", "done"): torch.zeros(
                    self.batch_size + (self.num_agents,),
                    dtype=torch.bool,
                    device=self.device,
                ),
                ("agents", "terminated"): torch.zeros(
                    self.batch_size + (self.num_agents,),
                    dtype=torch.bool,
                    device=self.device,
                ),
                ("agents", "truncated"): torch.zeros(
                    self.batch_size + (self.num_agents,),
                    dtype=torch.bool,
                    device=self.device,
                ),
            },
            batch_size=self.batch_size,
        )

    def _step(self, action: TensorDict) -> TensorDict:
        # action["inertia"], action["cognitive"], action["social"]: [batch, num_agents, dim]
        last_scores = self.scores
        last_neighborhood_best_scores = (
            self.neighborhood_best_scores
            if self.neighborhood_best_scores is not None
            else self.personal_best_scores
        )

        self.velocities = (
            action["inertia"] * self.velocities
            + action["cognitive"] * (self.personal_best_pos - self.positions)
            + action["social"] * self.get_social_signal()
        )

        self.positions = self.positions + self.velocities
        self.step_count += 1
        self.update_topology()

        last_scores = self.scores
        self.scores = self.landscape(self.positions)
        improved = self.scores > self.personal_best_scores
        self.personal_best_scores = torch.where(
            improved, self.scores, self.personal_best_scores
        )
        self.personal_best_pos = torch.where(
            improved.unsqueeze(-1), self.positions, self.personal_best_pos
        )
        self.compute_neighborhood_best()

        self.avg_pos, self.avg_vel = get_neighborhood_avg(
            self.positions, self.velocities, self.delta
        )

        reward = self._compute_reward(last_scores, last_neighborhood_best_scores)

        return TensorDict(
            {
                "scores": self.scores,
                "positions": self.positions,
                "velocities": self.velocities,
                "personal_best_pos": self.personal_best_pos,
                "personal_best_scores": self.personal_best_scores,
                "neighborhood_best_pos": self.neighborhood_best_pos,
                "neighborhood_best_scores": self.neighborhood_best_scores,
                "avg_pos": self.avg_pos,
                "avg_vel": self.avg_vel,
                ("agents", "reward"): reward,
                ("agents", "done"): torch.zeros(
                    self.batch_size + (self.num_agents,),
                    dtype=torch.bool,
                    device=self.device,
                ),  # change
                ("agents", "terminated"): torch.zeros(
                    self.batch_size + (self.num_agents,),
                    dtype=torch.bool,
                    device=self.device,
                ),
                ("agents", "truncated"): torch.zeros(
                    self.batch_size + (self.num_agents,),
                    dtype=torch.bool,
                    device=self.device,
                ),
            },
            batch_size=self.batch_size,
        )

    def _set_seed(self, seed) -> None:
        torch.manual_seed(seed)


def get_neighborhood_avg(positions, velocities, delta):
    # positions: [batch, num_agents, dim]
    # velocities: [batch, num_agents, dim]

    diff = positions.unsqueeze(2) - positions.unsqueeze(1)  # differences
    dist = diff.norm(
        dim=-1
    )  # [batch, num_agents, num_agents], last dimension is the landscape_dim

    neighbor_mask = (dist <= delta).float()
    counts = neighbor_mask.sum(dim=-1, keepdim=True).clamp_min(
        1.0
    )  # [batch, num_agents, 1]

    avg_pos = (neighbor_mask.unsqueeze(-1) * positions.unsqueeze(1)).sum(dim=2) / counts
    avg_pos = avg_pos - positions
    avg_vel = (neighbor_mask.unsqueeze(-1) * velocities.unsqueeze(1)).sum(
        dim=2
    ) / counts  # [batch, num_agents, dim]
    avg_vel = avg_vel - velocities

    return avg_pos, avg_vel

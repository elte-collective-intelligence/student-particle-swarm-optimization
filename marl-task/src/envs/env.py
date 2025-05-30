import torch
from torchrl.envs import EnvBase
from torchrl.data import Composite, Unbounded
from tensordict import TensorDict

class PSOEnv(EnvBase):
    def __init__(self,
                 landscape,
                 num_agents: int,
                 device: torch.device,
                 batch_size,
                 delta: float = 1.0,
                 run_type_checks: bool = False):
        super().__init__(device=device,
                         batch_size=batch_size,
                         run_type_checks=run_type_checks)
        
        self.landscape = landscape
        self.num_agents = num_agents
        self.delta = delta

        self.scores = None
        self.positions = None       
        self.velocities = None      
        self.personal_best_pos = None  
        self.personal_best_scores = None
        self.avg_pos = None
        self.avg_vel = None

        self.observation_spec = Composite({
            "scores": Unbounded(self.batch_size + (num_agents,), device=device),
            "positions": Unbounded(self.batch_size + (num_agents, landscape.dim), device=device),
            "velocities": Unbounded(self.batch_size + (num_agents, landscape.dim), device=device),
            "avg_pos": Unbounded(self.batch_size + (num_agents, landscape.dim), device=device),
            "avg_vel": Unbounded(self.batch_size + (num_agents, landscape.dim), device=device),
            "personal_best_pos": Unbounded(self.batch_size + (num_agents, landscape.dim), device=device),
            "personal_best_scores": Unbounded(self.batch_size + (num_agents,), device=device)
        }, shape=torch.Size(batch_size))

        self.action_spec = Composite({
            "inertia": Unbounded(self.batch_size + (num_agents, landscape.dim), device=device),
            "cognitive": Unbounded(self.batch_size + (num_agents, landscape.dim), device=device),
            "social": Unbounded(self.batch_size + (num_agents, landscape.dim), device=device)
        }, shape=torch.Size(batch_size))
    
    def _reset(self, params = None) -> TensorDict:
        # Reset landscape function
        self.landscape.reset()
        # Inicialize agents and scores
        self.positions = torch.randn(self.batch_size + (self.num_agents, self.landscape.dim), device=self.device)
        self.velocities = torch.zeros_like(self.positions)
        self.personal_best_pos = self.positions.clone()
        self.personal_best_scores = self.landscape(self.personal_best_pos)

        self.avg_pos, self.avg_vel = get_neighborhood_avg(self.positions, self.velocities, self.delta)

        self.scores = self.landscape(self.positions)
        # may need to call .clone() on fields
        return TensorDict({
            "observations": TensorDict({
                "scores": self.scores,
                "positions": self.positions,
                "velocities": self.velocities,
                "personal_best_pos": self.personal_best_pos,
                "personal_best_scores": self.personal_best_scores,
                "avg_pos": self.avg_pos,
                "avg_vel": self.avg_vel}),
            "reward": torch.zeros(self.batch_size, device=self.device),
            "done": torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        }, batch_size=self.batch_size)


    def _step(self, action: TensorDict) -> TensorDict:
        # action["inertia"], action["cognitive"], action["social"]: [batch, num_agents, dim]

        self.velocities = (
            action["inertia"] * self.velocities +
            action["cognitive"] * (self.personal_best_pos - self.positions) +
            action["social"] * self.avg_pos
        )

        self.positions = self.positions + self.velocities

        last_scores = self.scores
        self.scores = self.landscape(self.positions)
        improved = self.scores > self.personal_best_scores
        self.personal_best_scores = torch.where(improved, self.scores, self.personal_best_scores)
        self.personal_best_pos = torch.where(improved.unsqueeze(-1), self.positions, self.personal_best_pos)

        self.avg_pos, self.avg_vel = get_neighborhood_avg(self.positions, self.velocities, self.delta)
        
        return TensorDict({
            "observations": TensorDict({
                "scores": self.scores,
                "positions": self.positions,
                "velocities": self.velocities,
                "personal_best_pos": self.personal_best_pos,
                "personal_best_scores": self.personal_best_scores,
                "avg_pos": self.avg_pos,
                "avg_vel": self.avg_vel}),
            "reward": (self.scores - last_scores).mean(dim=1),
            "done": torch.zeros(self.batch_size, dtype=torch.bool, device=self.device) # change
        }, batch_size=self.batch_size)

    def _set_seed(self, seed) -> None:
        torch.manual_seed(seed)

def get_neighborhood_avg(positions, velocities, delta):
    # positions: [batch, num_agents, dim]
    # velocities: [batch, num_agents, dim]

    diff = positions.unsqueeze(2) - positions.unsqueeze(1) # differences
    dist = diff.norm(dim=-1) # [batch, num_agents, num_agents], last dimension is the landscape_dim 
        
    neighbor_mask = (dist <= delta).float()
    counts = neighbor_mask.sum(dim=-1, keepdim=True).clamp_min(1.0) # [batch, num_agents, 1]

    avg_pos = (neighbor_mask.unsqueeze(-1) * positions.unsqueeze(1)).sum(dim=2) / counts
    avg_pos = avg_pos - positions
    avg_vel = (neighbor_mask.unsqueeze(-1) * velocities.unsqueeze(1)).sum(dim=2) / counts  # [batch, num_agents, dim]
    avg_vel = avg_vel - velocities

    return avg_pos, avg_vel



class LandscapeWrapper:
    def __init__(self, function, dim):
        self.function = function
        self.dim = dim
    
    def __call__(self, x):
        return self.function(x)
    
    def reset(self):
        pass
import torch
from tensordict import TensorDict
from envs import LandscapeWrapper, PSOEnv
import pytest

def test_dynamic_parameter_changes():
    """Test that the environment correctly handles parameter changes during an episode."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def quadratic(x):
        return -torch.sum(x**2, dim=-1)
    
    landscape = LandscapeWrapper(quadratic, dim=2)
    env = PSOEnv(landscape=landscape,
                num_agents=10,
                device=device,
                batch_size=(32,),
                delta=1.0)
    
    obs = env.reset()
    
    action = TensorDict({
        "inertia": torch.rand((32, 10, 2), device=device),
        "cognitive": torch.rand((32, 10, 2), device=device),
        "social": torch.rand((32, 10, 2), device=device)
    }, batch_size=(32,))
    
    for _ in range(5):
        obs = env.step(action)
    
    env.delta = 5.0  # Expand neighborhood radius
    env.landscape.function = lambda x: -torch.sum((x - 2.0)**2, dim=-1)  # Shift optimum
    
    new_obs = env.step(action)
    
    old_avg_pos = obs["observations"]["avg_pos"]
    new_avg_pos = new_obs["observations"]["avg_pos"]
    assert not torch.allclose(old_avg_pos, new_avg_pos), "Neighborhood averages should change with delta"
    
    old_scores = obs["observations"]["scores"]
    new_scores = new_obs["observations"]["scores"]
    assert not torch.allclose(old_scores, new_scores), "Scores should change with landscape function"
    
    improved = new_scores > new_obs["observations"]["personal_best_scores"]
    assert improved.any(), "Some agents should improve with new landscape"
    
    reward = new_obs["reward"]
    assert reward.shape == (32,), "Reward should have correct shape"
    assert not torch.all(reward == 0), "Reward should be non-zero after changes"
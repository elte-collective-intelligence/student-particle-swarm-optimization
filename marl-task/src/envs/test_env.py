import torch
from tensordict import TensorDict
from envs import LandscapeWrapper, PSOEnv
import pytest

class TestPSOEnvDynamicParams:
    """Test class for dynamic parameter changes in PSO environment"""
    
    @pytest.fixture(autouse=True)
    def setup_env(self):
        """Fixture to set up the environment for each test"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Simple quadratic function for testing
        def quadratic(x):
            return -torch.sum(x**2, dim=-1)
        
        self.landscape = LandscapeWrapper(quadratic, dim=2)
        self.env = PSOEnv(landscape=self.landscape,
                         num_agents=10,
                         device=self.device,
                         batch_size=(32,),
                         delta=1.0)
        
        self.action = TensorDict({
            "inertia": torch.rand((32, 10, 2), device=self.device),
            "cognitive": torch.rand((32, 10, 2), device=self.device),
            "social": torch.rand((32, 10, 2), device=self.device)
        }, batch_size=(32,))
        
        self.obs = self.env.reset()
        
        for _ in range(5):
            self.obs = self.env.step(self.action)
    
    def test_delta_change(self):
        """Test changing neighborhood radius during episode"""
        old_avg_pos = self.obs["observations"]["avg_pos"]
        
        self.env.delta = 5.0
        new_obs = self.env.step(self.action)
        new_avg_pos = new_obs["observations"]["avg_pos"]
        
        assert not torch.allclose(old_avg_pos, new_avg_pos), \
            "Neighborhood averages should change with delta"
    
    def test_landscape_change(self):
        """Test changing objective function during episode"""
        old_scores = self.obs["observations"]["scores"]
        
        self.landscape.function = lambda x: -torch.sum((x - 2.0)**2, dim=-1)
        new_obs = self.env.step(self.action)
        new_scores = new_obs["observations"]["scores"]
        
        assert not torch.allclose(old_scores, new_scores), \
            "Scores should change with landscape function"
        
        reward = new_obs["reward"]
        assert reward.shape == (32,), "Reward should have correct shape"
        assert not torch.all(reward == 0), "Reward should be non-zero after changes"
    
    def test_personal_best_updates(self):
        """Test personal best updates after parameter changes"""
        self.landscape.function = lambda x: -torch.sum((x - 0.5)**2, dim=-1)
        new_obs = self.env.step(self.action)
        
        improved = new_obs["observations"]["scores"] > \
                  new_obs["observations"]["personal_best_scores"]
        assert improved.any(), "Some agents should improve with new landscape"
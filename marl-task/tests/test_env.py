import torch
from tensordict import TensorDict
import pytest
import math
from envs import PSOEnv
from envs.dynamic_functions import DynamicSphere, DynamicRastrigin, DynamicEggHolder
from utils import LandscapeWrapper

class TestPSOBasicFunctionality:
    """Test basic PSO environment functionality"""
    
    @pytest.fixture
    def basic_env(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        def quadratic(x):
            return -torch.sum(x**2, dim=-1)
        
        landscape = LandscapeWrapper(quadratic, dim=2)
        env = PSOEnv(landscape=landscape,
                    num_agents=10,
                    device=device,
                    batch_size=(32,),
                    delta=1.0)
        return env
    
    def test_reset(self, basic_env):
        """Test environment reset functionality"""
        obs = basic_env.reset()
        
        assert "positions" in obs
        assert "velocities" in obs
        assert "scores" in obs
        assert "personal_best_pos" in obs
        assert "personal_best_scores" in obs
        
        assert obs["positions"].shape == (32, 10, 2)
        assert obs["velocities"].shape == (32, 10, 2)
        assert obs["scores"].shape == (32, 10)
    
    def test_step(self, basic_env):
        """Test basic step functionality"""
        obs = basic_env.reset()
        
        action = TensorDict({
            "inertia": torch.ones((32, 10, 2), device=basic_env.device) * 0.5,
            "cognitive": torch.ones((32, 10, 2), device=basic_env.device) * 0.5,
            "social": torch.ones((32, 10, 2), device=basic_env.device) * 0.5
        }, batch_size=(32,))
        
        new_obs = basic_env.step(action)
        
        assert "positions" in new_obs
        assert "velocities" in new_obs
        assert "scores" in new_obs
        assert "reward" in new_obs
        
        # Positions should change after step
        assert not torch.allclose(obs["positions"], new_obs["positions"])
        
        # Velocities should change after step
        assert not torch.allclose(obs["velocities"], new_obs["velocities"])

class TestDynamicFunctions:
    """Test the dynamic functions used in the PSO environment"""
    
    @pytest.fixture(params=[2, 5, 10])
    def dim(self, request):
        return request.param
    
    def test_dynamic_sphere(self, dim):
        """Test the DynamicSphere function"""
        func = DynamicSphere(dim=dim)
        x = torch.randn(10, dim)
        
        # Test initial evaluation
        initial_scores = func(x)
        assert initial_scores.shape == (10,)
        
        # Test that scores change over time
        func.time = 10
        new_scores = func(x)
        assert not torch.allclose(initial_scores, new_scores)
        
        # Test reset
        func.reset()
        assert func.time == 0
        assert torch.all(func.optimum == 0)
    
    def test_dynamic_rastrigin(self, dim):
        """Test the DynamicRastrigin function"""
        func = DynamicRastrigin(dim=dim)
        x = torch.rand(10, dim) * 5.24 - 2.6  # Rastrigin range
        
        # Test initial evaluation
        initial_scores = func(x)
        assert initial_scores.shape == (10,)
        
        # Test that amplitude changes over time
        func.time = 100
        new_scores = func(x)
        assert not torch.allclose(initial_scores, new_scores)
        
        # Test reset
        func.reset()
        assert func.time == 0
    
    def test_dynamic_eggholder(self):
        """Test the DynamicEggHolder function (only works in 2D)"""
        func = DynamicEggHolder(dim=2)
        x = torch.rand(10, 2) * 512 - 256  # Eggholder range
        
        # Test initial evaluation
        initial_scores = func(x)
        assert initial_scores.shape == (10,)
        
        # Test that rotation affects the scores
        func.time = 100
        new_scores = func(x)
        assert not torch.allclose(initial_scores, new_scores)
        
        # Test reset
        func.reset()
        assert func.time == 0

class TestPSOWithDynamicFunctions:
    """Test PSO environment with dynamic functions"""
    
    @pytest.fixture(params=[
        DynamicSphere(dim=2),
        DynamicRastrigin(dim=2),
        DynamicEggHolder(dim=2)
    ])
    def dynamic_env(self, request):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        env = PSOEnv(landscape=request.param,
                    num_agents=10,
                    device=device,
                    batch_size=(32,),
                    delta=1.0)
        return env
    
    def test_dynamic_function_integration(self, dynamic_env):
        """Test that dynamic functions work within the PSO environment"""
        obs = dynamic_env.reset()
        initial_scores = obs["scores"].clone()
        
        action = TensorDict({
            "inertia": torch.ones((32, 10, 2), device=dynamic_env.device) * 0.5,
            "cognitive": torch.ones((32, 10, 2), device=dynamic_env.device) * 0.5,
            "social": torch.ones((32, 10, 2), device=dynamic_env.device) * 0.5
        }, batch_size=(32,))
        
        for _ in range(10):
            obs = dynamic_env.step(action)
        
        # Scores should change
        assert not torch.allclose(initial_scores, obs["scores"])
    
    def test_personal_best_updates_with_dynamics(self, dynamic_env):
        """Test personal best updates with dynamic functions"""
        obs = dynamic_env.reset()
        initial_pbest = obs["personal_best_scores"].clone()
        
        action = TensorDict({
            "inertia": torch.ones((32, 10, 2), device=dynamic_env.device) * 0.5,
            "cognitive": torch.ones((32, 10, 2), device=dynamic_env.device) * 0.5,
            "social": torch.ones((32, 10, 2), device=dynamic_env.device) * 0.5
        }, batch_size=(32,))
        
        # Run steps
        improved = False
        for _ in range(20):
            obs = dynamic_env.step(action)
            if (obs["scores"] > obs["personal_best_scores"]).any():
                improved = True
                break
        
        assert improved, "Some agents should improve their personal best"

class TestNeighborhoodMechanics:
    """Test neighborhood calculations in PSO"""
    
    @pytest.fixture
    def neighborhood_env(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        def quadratic(x):
            return -torch.sum(x**2, dim=-1)
        
        landscape = LandscapeWrapper(quadratic, dim=2)
        env = PSOEnv(landscape=landscape,
                    num_agents=10,
                    device=device,
                    batch_size=(32,),
                    delta=1.0)
        return env
    
    def test_delta_effect(self, neighborhood_env):
        """Test that changing delta affects neighborhood averages"""
        # Set positions
        neighborhood_env.positions = torch.randn(32, 10, 2, device=neighborhood_env.device)
        neighborhood_env.velocities = torch.randn(32, 10, 2, device=neighborhood_env.device)
        
        # Get averages with small delta
        neighborhood_env.delta = 0.1
        small_avg_pos, _ = neighborhood_env._get_neighborhood_avg()
        
        # Get averages with large delta
        neighborhood_env.delta = 10.0
        large_avg_pos, _ = neighborhood_env._get_neighborhood_avg()
        
        # Averages should be different
        assert not torch.allclose(small_avg_pos, large_avg_pos)

class TestRewardMechanics:
    """Test reward calculations in PSO"""
    
    @pytest.fixture
    def reward_env(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        def linear(x):
            return x.sum(dim=-1)
        
        landscape = LandscapeWrapper(linear, dim=2)
        env = PSOEnv(landscape=landscape,
                    num_agents=10,
                    device=device,
                    batch_size=(32,),
                    delta=1.0)
        return env
    
    def test_reward_calculation(self, reward_env):
        """Test that rewards are calculated correctly"""
        # Set positions and scores
        reward_env.positions = torch.zeros(32, 10, 2, device=reward_env.device)
        reward_env.scores = torch.zeros(32, 10, device=reward_env.device)
        
        # Take a step that improves scores
        new_positions = torch.ones(32, 10, 2, device=reward_env.device)
        new_scores = reward_env.landscape(new_positions)
        
        action = TensorDict({
            "inertia": torch.zeros(32, 10, 2, device=reward_env.device),
            "cognitive": torch.zeros(32, 10, 2, device=reward_env.device),
            "social": torch.zeros(32, 10, 2, device=reward_env.device)
        }, batch_size=(32,))
        
        reward_env.positions = new_positions
        reward_env.scores = new_scores
        
        obs = reward_env.step(action)
        
        # Reward should be positive since scores improved
        assert torch.all(obs["reward"] > 0)
    
    def test_no_improvement_reward(self, reward_env):
        """Test that rewards are zero when there's no improvement"""
        obs = reward_env.reset()
        
        # Set positions that won't improve the score
        reward_env.positions = torch.zeros(32, 10, 2, device=reward_env.device)
        reward_env.scores = torch.zeros(32, 10, device=reward_env.device)
        
        action = TensorDict({
            "inertia": torch.zeros(32, 10, 2, device=reward_env.device),
            "cognitive": torch.zeros(32, 10, 2, device=reward_env.device),
            "social": torch.zeros(32, 10, 2, device=reward_env.device)
        }, batch_size=(32,))
        
        obs = reward_env.step(action)
        
        # Reward should be zero since scores didn't improve
        assert torch.all(obs["reward"] == 0)
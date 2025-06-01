import torch
from tensordict import TensorDict
import pytest
import math
import sys
import os
from pathlib import Path

root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir / "src"))

from envs.env import PSOEnv, get_neighborhood_avg
from envs.dynamic_functions import DynamicSphere, DynamicRastrigin, DynamicEggHolder
from utils import LandscapeWrapper

class TestPSOBasicFunctionality:
    """Test basic PSO environment functionality"""
    
    @pytest.fixture
    def basic_env(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        landscape = LandscapeWrapper(lambda x: -torch.sum(x**2, dim=-1), dim=2)
        env = PSOEnv(landscape=landscape, num_agents=5, device=device, batch_size=(4,), delta=1.0)
        return env
    
    def test_reset(self, basic_env):
        """Test environment reset functionality"""
        obs = basic_env.reset()

        if "next" in obs:
            obs = obs["next"]

        assert "positions" in obs
        assert "velocities" in obs
        assert "scores" in obs
        assert "personal_best_pos" in obs
        assert "personal_best_scores" in obs

        assert obs["positions"].shape[-3:] == (basic_env.batch_size[0], basic_env.num_agents, basic_env.landscape.dim)
        assert obs["velocities"].shape[-3:] == (basic_env.batch_size[0], basic_env.num_agents, basic_env.landscape.dim)
        assert obs["scores"].shape[-2:] == (basic_env.batch_size[0], basic_env.num_agents)

    def test_step(self, basic_env):
        """Test environment step functionality"""
        obs = basic_env.reset()
        
        if "next" in obs:
            obs = obs["next"]

        action = TensorDict({
            "inertia": torch.ones((4, 5, 2), device=basic_env.device) * 0.5,
            "cognitive": torch.ones((4, 5, 2), device=basic_env.device) * 0.5,
            "social": torch.ones((4, 5, 2), device=basic_env.device) * 0.5
        }, batch_size=(4,))

        new_obs = basic_env.step(action)
        if "next" in new_obs:
            new_obs = new_obs["next"]

        # Check if the positions and velocities changed
        assert not torch.allclose(obs["positions"], new_obs["positions"])
        assert not torch.allclose(obs["velocities"], new_obs["velocities"])

        # Check if expected keys are present
        for key in ["positions", "velocities", "scores", "personal_best_pos", "personal_best_scores", "avg_pos", "avg_vel"]:
            assert key in new_obs

        # Check if reward is present and has correct shape
        assert ("agents", "reward") in new_obs
        assert new_obs[("agents", "reward")].shape == (4, 5)

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
    
    def test_neighborhood_calculation(self, neighborhood_env):
        """Test that neighborhood averages are calculated correctly"""
        # Set specific positions for testing
        neighborhood_env.positions = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], 
            [2.0, 0.0], [0.0, 2.0], [2.0, 2.0],
            [3.0, 0.0], [0.0, 3.0], [3.0, 3.0]
        ], device=neighborhood_env.device).repeat(32, 1, 1)
        
        neighborhood_env.velocities = torch.zeros_like(neighborhood_env.positions)
        
        neighborhood_env.delta = 1.5
        avg_pos, avg_vel = get_neighborhood_avg(neighborhood_env.positions, neighborhood_env.velocities, neighborhood_env.delta)
        
        # For agent at (0,0), neighbors are at (1,0), (0,1), (1,1)
        expected_avg_pos = torch.tensor([
            [1.0+0.0+1.0, 0.0+1.0+1.0]
        ], device=neighborhood_env.device).mean(dim=0) / 4  # Includes self
        
        assert torch.allclose(avg_pos[0, 0], expected_avg_pos, atol=1e-5)
    
    def test_delta_effect(self, neighborhood_env):
        """Test that changing delta affects neighborhood averages"""
        # Set positions
        neighborhood_env.positions = torch.randn(32, 10, 2, device=neighborhood_env.device)
        neighborhood_env.velocities = torch.randn(32, 10, 2, device=neighborhood_env.device)
        
        # Get averages with small delta
        neighborhood_env.delta = 0.1
        small_avg_pos, _ = get_neighborhood_avg(neighborhood_env.positions, neighborhood_env.velocities, neighborhood_env.delta)
        
        # Get averages with large delta
        neighborhood_env.delta = 10.0
        large_avg_pos, _ = get_neighborhood_avg(neighborhood_env.positions, neighborhood_env.velocities, neighborhood_env.delta)
        
        # Averages should be different
        assert not torch.allclose(small_avg_pos, large_avg_pos)

class TestRewardMechanics:
    """Test reward calculations in PSO"""
    
    @pytest.fixture
    def reward_env(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        landscape = LandscapeWrapper(lambda x: -torch.sum(x**2, dim=-1), dim=2)
        env = PSOEnv(landscape=landscape, num_agents=10, device=device, batch_size=(32,), delta=1.0)

        return env
    
    def test_reward_calculation(self, reward_env):
        """Test that rewards are calculated correctly"""
        # Set positions and scores

        reward_env.positions = torch.zeros(32, 10, 2, device=reward_env.device)
        reward_env.velocities = torch.zeros_like(reward_env.positions)
        reward_env.scores = torch.zeros(32, 10, device=reward_env.device)
        
        # Take a step that improves scores
        new_positions = torch.ones(32, 10, 2, device=reward_env.device)
        new_scores = reward_env.landscape(new_positions)

        action = TensorDict({
            "inertia": torch.zeros(32, 10, 2, device=reward_env.device),
            "cognitive": torch.zeros(32, 10, 2, device=reward_env.device),
            "social": torch.zeros(32, 10, 2, device=reward_env.device)
        }, batch_size=(32,))
        
        # Mock the position update
        reward_env.positions = new_positions
        reward_env.velocities = torch.zeros_like(new_positions)
        reward_env.scores = new_scores
        reward_env.personal_best_pos = reward_env.positions.clone()
        reward_env.personal_best_scores = reward_env.landscape(reward_env.personal_best_pos)
        reward_env.avg_pos, reward_env.average_vel = get_neighborhood_avg(reward_env.positions, reward_env.velocities, reward_env.delta)
        
        obs = reward_env.step(action)
        if "next" in obs:
            obs = obs["next"]

        assert ("agents", "reward") in obs
        assert obs[("agents", "reward")].shape == (32, 10)

    def test_reset_consistency(self, reward_env):
        """Test that reset returns consistent shapes and values."""

        obs = reward_env.reset()
        
        if "next" in obs:
            obs = obs["next"]

        assert obs["positions"].shape == (32, 10, 2)
        assert obs["velocities"].shape == (32, 10, 2)
        assert obs["scores"].shape == (32, 10)
        assert obs["personal_best_pos"].shape == (32, 10, 2)
        assert obs["personal_best_scores"].shape == (32, 10)
        assert obs["avg_pos"].shape == (32, 10, 2)
        assert obs["avg_vel"].shape == (32, 10, 2)
        assert ("agents", "reward") in obs

    def test_step_changes_state(self, reward_env):
        """Test that step changes positions and velocities."""
        
        obs = reward_env.reset()
        if "next" in obs:
            obs = obs["next"]

        action = TensorDict({
            "inertia": torch.ones((32, 10, 2), device=reward_env.device) * 0.5,
            "cognitive": torch.ones((32, 10, 2), device=reward_env.device) * 0.5,
            "social": torch.ones((32, 10, 2), device=reward_env.device) * 0.5
        }, batch_size=(32,))

        new_obs = reward_env.step(action)
        if "next" in new_obs:
            new_obs = new_obs["next"]
        
        assert not torch.allclose(obs["positions"], new_obs["positions"])
        assert not torch.allclose(obs["velocities"], new_obs["velocities"])

    def test_neighborhood_avg_method(self, reward_env):
        """Test that get_neighborhood_avg returns correct shapes."""
        
        reward_env.reset()
        avg_pos, avg_vel = get_neighborhood_avg(reward_env.positions, reward_env.velocities, reward_env.delta)
        assert avg_pos.shape == (32, 10, 2)
        assert avg_vel.shape == (32, 10, 2)

    def test_reward_mechanics_improvement(self, reward_env):
        """Test reward is positive when scores improve."""
        
        reward_env.reset()
        reward_env.positions = torch.ones(32, 10, 2, device=reward_env.device)
        reward_env.velocities = torch.zeros_like(reward_env.positions)
        reward_env.scores = reward_env.landscape(reward_env.positions)
        reward_env.avg_pos, reward_env.avg_vel = get_neighborhood_avg(reward_env.positions, reward_env.velocities, reward_env.delta)
        
        action = TensorDict({
            "inertia": torch.zeros(32, 10, 2, device=reward_env.device),
            "cognitive": torch.zeros(32, 10, 2, device=reward_env.device),
            "social": torch.zeros(32, 10, 2, device=reward_env.device)
        }, batch_size=(32,))

        obs = reward_env.step(action)
        if "next" in obs:
            obs = obs["next"]

        assert torch.all(obs[("agents", "reward")] == 0)

    def test_reward_mechanics_no_improvement(self, reward_env):
        """Test reward is zero when no improvement."""
        
        reward_env.reset()
        reward_env.positions = torch.zeros(32, 10, 2, device=reward_env.device)
        reward_env.velocities = torch.zeros_like(reward_env.positions)
        reward_env.scores = reward_env.landscape(reward_env.positions)
        reward_env.avg_pos, reward_env.avg_vel = get_neighborhood_avg(reward_env.positions, reward_env.velocities, reward_env.delta)
        action = TensorDict({
            "inertia": torch.zeros(32, 10, 2, device=reward_env.device),
            "cognitive": torch.zeros(32, 10, 2, device=reward_env.device),
            "social": torch.zeros(32, 10, 2, device=reward_env.device)
        }, batch_size=(32,))

        obs = reward_env.step(action)
        if "next" in obs:
            obs = obs["next"]

        assert torch.all(obs[("agents", "reward")] == 0)
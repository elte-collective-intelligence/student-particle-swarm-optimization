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
from utils import LandscapeWrapper, PSOObservationWrapper, PSOActionExtractor

class TestPSOBasicFunctionality:
    """Test basic PSO environment functionality"""
    
    @pytest.fixture
    def basic_env(self):
        """
        Fixture to create a basic PSOEnv environment with a simple quadratic landscape.
        Returns an environment with 5 agents, batch size 4, and 2D search space.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        landscape = LandscapeWrapper(lambda x: -torch.sum(x**2, dim=-1), dim=2)
        env = PSOEnv(landscape=landscape, num_agents=5, device=device, batch_size=(4,), delta=1.0)
        return env
    
    def test_reset(self, basic_env):
        """
        Test environment reset functionality.
        Ensures that all expected keys are present in the observation and that their shapes are correct.
        """
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
        """
        Test environment step functionality.
        Checks that positions and velocities change after a step and that all expected keys are present in the new observation.
        Also checks that the reward key exists and has the correct shape.
        """
        obs = basic_env.reset()
        
        if "next" in obs:
            obs = obs["next"]

        action = TensorDict({
            "inertia": torch.ones((basic_env.batch_size[0], 5, 2), device=basic_env.device) * 0.5,
            "cognitive": torch.ones((basic_env.batch_size[0], 5, 2), device=basic_env.device) * 0.5,
            "social": torch.ones((basic_env.batch_size[0], 5, 2), device=basic_env.device) * 0.5
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
    """Test the dynamic functions used in the PSO environment (not used in the end)"""
    
    @pytest.fixture(params=[2, 5, 10])
    def dim(self, request):
        return request.param
    
    def test_dynamic_sphere(self, dim):
        """
        Test the DynamicSphere function.
        Checks initial evaluation, time-dependent changes, and reset behavior.
        """
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
        """
        Test the DynamicRastrigin function.
        Checks initial evaluation, time-dependent amplitude changes, and reset behavior.
        """
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
        """
        Test the DynamicEggHolder function.
        Checks initial evaluation, time-dependent rotation, and reset behavior.
        """
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
        """
        Fixture to create a PSOEnv environment for neighborhood tests.
        Uses a quadratic landscape and 10 agents.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        def quadratic(x):
            return -torch.sum(x**2, dim=-1)
        
        landscape = LandscapeWrapper(quadratic, dim=2)
        env = PSOEnv(landscape=landscape,
                    num_agents=10,
                    device=device,
                    batch_size=(4,),
                    delta=1.0)
        return env
    
    def test_neighborhood_calculation(self, neighborhood_env):
        """
        Test checks if neighborhood averages are calculated correctly.
        Sets specific positions and checks the average for a known agent.
        """
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
        """
        Test checks if changing delta affects neighborhood averages.
        Uses random positions and checks that averages differ for small and large delta.
        """
        # Set positions
        neighborhood_env.positions = torch.randn(neighborhood_env.batch_size[0], 10, 2, device=neighborhood_env.device)
        neighborhood_env.velocities = torch.randn(neighborhood_env.batch_size[0], 10, 2, device=neighborhood_env.device)
        
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
        """
        Fixture to create a PSOEnv environment for reward tests.
        Uses a quadratic landscape and 10 agents.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        landscape = LandscapeWrapper(lambda x: -torch.sum(x**2, dim=-1), dim=2)
        env = PSOEnv(landscape=landscape, num_agents=10, device=device, batch_size=(4,), delta=1.0)

        return env
    
    def test_reward_calculation(self, reward_env):
        """
        Test checks if rewards are calculated correctly.
        Sets positions and scores, performs a step, and checks reward shape.
        """
        # Set positions and scores

        reward_env.positions = torch.zeros(reward_env.batch_size[0], 10, 2, device=reward_env.device)
        reward_env.velocities = torch.zeros_like(reward_env.positions)
        reward_env.scores = torch.zeros(reward_env.batch_size[0], 10, device=reward_env.device)
        
        # Take a step that improves scores
        new_positions = torch.ones(reward_env.batch_size[0], 10, 2, device=reward_env.device)
        new_scores = reward_env.landscape(new_positions)

        action = TensorDict({
            "inertia": torch.zeros(reward_env.batch_size[0], 10, 2, device=reward_env.device),
            "cognitive": torch.zeros(reward_env.batch_size[0], 10, 2, device=reward_env.device),
            "social": torch.zeros(reward_env.batch_size[0], 10, 2, device=reward_env.device)
        }, batch_size=(reward_env.batch_size[0],))
        
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
        assert obs[("agents", "reward")].shape == (reward_env.batch_size[0], 10)

    def test_reset_consistency(self, reward_env):
        """
        Test checks if reset returns consistent shapes and values for all observation keys.
        """

        obs = reward_env.reset()
        
        if "next" in obs:
            obs = obs["next"]

        assert obs["positions"].shape == (reward_env.batch_size[0], 10, 2)
        assert obs["velocities"].shape == (reward_env.batch_size[0], 10, 2)
        assert obs["scores"].shape == (reward_env.batch_size[0], 10)
        assert obs["personal_best_pos"].shape == (reward_env.batch_size[0], 10, 2)
        assert obs["personal_best_scores"].shape == (reward_env.batch_size[0], 10)
        assert obs["avg_pos"].shape == (reward_env.batch_size[0], 10, 2)
        assert obs["avg_vel"].shape == (reward_env.batch_size[0], 10, 2)
        assert ("agents", "reward") in obs

    def test_step_changes_state(self, reward_env):
        """
        Test checks if changes positions and velocities.
        Checks that the state is updated after a step.
        """
        obs = reward_env.reset()
        if "next" in obs:
            obs = obs["next"]

        action = TensorDict({
            "inertia": torch.ones((reward_env.batch_size[0], 10, 2), device=reward_env.device) * 0.5,
            "cognitive": torch.ones((reward_env.batch_size[0], 10, 2), device=reward_env.device) * 0.5,
            "social": torch.ones((reward_env.batch_size[0], 10, 2), device=reward_env.device) * 0.5
        }, batch_size=(reward_env.batch_size[0],))

        new_obs = reward_env.step(action)
        if "next" in new_obs:
            new_obs = new_obs["next"]
        
        assert not torch.allclose(obs["positions"], new_obs["positions"])
        assert not torch.allclose(obs["velocities"], new_obs["velocities"])

    def test_neighborhood_avg_method(self, reward_env):
        """
        Test checks if get_neighborhood_avg returns correct shapes for avg_pos and avg_vel.
        """
        reward_env.reset()
        avg_pos, avg_vel = get_neighborhood_avg(reward_env.positions, reward_env.velocities, reward_env.delta)
        assert avg_pos.shape == (reward_env.batch_size[0], 10, 2)
        assert avg_vel.shape == (reward_env.batch_size[0], 10, 2)

    def test_reward_mechanics_improvement(self, reward_env):
        """
        Test checks if reward is zero when scores do not improve (positions are set to ones, so no improvement).
        """
        
        reward_env.reset()
        reward_env.positions = torch.ones(reward_env.batch_size[0], 10, 2, device=reward_env.device)
        reward_env.velocities = torch.zeros_like(reward_env.positions)
        reward_env.scores = reward_env.landscape(reward_env.positions)
        reward_env.avg_pos, reward_env.avg_vel = get_neighborhood_avg(reward_env.positions, reward_env.velocities, reward_env.delta)
        
        action = TensorDict({
            "inertia": torch.zeros(reward_env.batch_size[0], 10, 2, device=reward_env.device),
            "cognitive": torch.zeros(reward_env.batch_size[0], 10, 2, device=reward_env.device),
            "social": torch.zeros(reward_env.batch_size[0], 10, 2, device=reward_env.device)
        }, batch_size=(reward_env.batch_size[0],))

        obs = reward_env.step(action)
        if "next" in obs:
            obs = obs["next"]

        assert torch.all(obs[("agents", "reward")] == 0)

    def test_reward_mechanics_no_improvement(self, reward_env):
        """
        Test checks if reward is zero when scores do not improve (positions are set to ones, so no improvement).
        """
        
        reward_env.reset()
        reward_env.positions = torch.zeros(reward_env.batch_size[0], 10, 2, device=reward_env.device)
        reward_env.velocities = torch.zeros_like(reward_env.positions)
        reward_env.scores = reward_env.landscape(reward_env.positions)
        reward_env.avg_pos, reward_env.avg_vel = get_neighborhood_avg(reward_env.positions, reward_env.velocities, reward_env.delta)
        action = TensorDict({
            "inertia": torch.zeros(reward_env.batch_size[0], 10, 2, device=reward_env.device),
            "cognitive": torch.zeros(reward_env.batch_size[0], 10, 2, device=reward_env.device),
            "social": torch.zeros(reward_env.batch_size[0], 10, 2, device=reward_env.device)
        }, batch_size=(reward_env.batch_size[0],))

        obs = reward_env.step(action)
        if "next" in obs:
            obs = obs["next"]

        assert torch.all(obs[("agents", "reward")] == 0)

class TestUtilsAndWrappers:
    """Test utility functions and wrappers used in PSO"""

    def test_landscape_wrapper_call_and_dim(self):
        """
        Test checks if LandscapeWrapper correctly wraps a function and exposes the dim attribute.
        """
        f = lambda x: torch.sum(x**2, dim=-1)
        wrapper = LandscapeWrapper(f, dim=3)
        x = torch.tensor([[1.0, 2.0, 3.0]])
        result = wrapper(x)

        assert torch.allclose(result, torch.tensor([14.0]))
        assert wrapper.dim == 3

    def test_landscape_wrapper_reset(self):
        """
        Test checks that LandscapeWrapper.reset() does not raise an error.
        """
        wrapper = LandscapeWrapper(lambda x: x, dim=2)
        wrapper.reset()

    def test_pso_observation_wrapper_concat(self):
        """
        Test checks if PSOObservationWrapper concatenates avg_pos and avg_vel along the last dimension.
        """
        obs_wrapper = PSOObservationWrapper()
        avg_pos = torch.ones(2, 3, 4)
        avg_vel = torch.zeros(2, 3, 4)
        out = obs_wrapper(avg_pos, avg_vel)

        assert out.shape == (2, 3, 8)
        assert torch.all(out[..., :4] == 1)
        assert torch.all(out[..., 4:] == 0)

    def test_pso_action_extractor_forward(self):
        """
        Test checks if PSOActionExtractor correctly extracts inertia, cognitive, and social components from the action tensor.
        Checks that the output shapes and values are as expected.
        """
        dim = 2
        extractor = PSOActionExtractor(dim)

        a0 = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        a1 = torch.tensor([[5.0, 6.0, 7.0, 8.0]])
        action = (a0, a1)
        inertia, inertia2, cognitive, cognitive2, social, social2 = extractor(action)

        assert inertia.shape == (1, dim)
        assert cognitive.shape == (1, dim)
        assert social.shape == (1, 0)
        assert torch.all(inertia == torch.tensor([[1.0, 2.0]]))
        assert torch.all(cognitive == torch.tensor([[3.0, 4.0]]))

    def test_pso_action_extractor_nan_raises(self):
        """
        Test checks that PSOActionExtractor raises a ValueError if the action contains NaN values.
        """
        dim = 2
        extractor = PSOActionExtractor(dim)
        a0 = torch.tensor([[float('nan'), 2.0, 3.0, 4.0]])
        a1 = torch.tensor([[5.0, 6.0, 7.0, 8.0]])
        action = (a0, a1)

        with pytest.raises(ValueError):
            extractor(action)
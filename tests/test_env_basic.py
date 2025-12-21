# tests/test_env_basic.py
import torch

from src.envs.pso_env import PSOEnv
from src.utils import LandscapeWrapper


def sphere(x: torch.Tensor) -> torch.Tensor:
    return (x ** 2).sum(dim=-1)


def test_env_reset_and_step():
    device = torch.device("cpu")
    landscape = LandscapeWrapper(sphere, dim=2)
    env = PSOEnv(
        landscape_fn=landscape,
        num_agents=4,
        dim=2,
        batch_size=[2],
        device=device,
        max_steps=5,
        dynamic=False,  # static for test
    )

    td = env.reset()
    assert "positions" in td.keys()
    assert td["positions"].shape == (2, 4, 2)

    # random actions
    actions = torch.randn(2, 4, 3)
    step_td = td.clone()
    step_td["action"] = actions

    next_td = env.step(step_td)
    assert "reward" in next_td.keys()
    assert next_td["reward"].shape == (2, 4)


def test_env_metrics():
    device = torch.device("cpu")
    landscape = LandscapeWrapper(sphere, dim=2)
    env = PSOEnv(
        landscape_fn=landscape,
        num_agents=4,
        dim=2,
        batch_size=[2],
        device=device,
        max_steps=3,
        dynamic=False,
    )

    td = env.reset()
    for _ in range(3):
        actions = torch.randn(2, 4, 3)
        step_td = td.clone()
        step_td["action"] = actions
        td = env.step(step_td)

    metrics = env.get_metrics()
    assert "best_fitness" in metrics
    assert "diversity" in metrics

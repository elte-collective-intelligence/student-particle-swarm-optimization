import torch
from tensordict import TensorDict

from src.envs.pso_env import PSOEnv
from src.utils import LandscapeWrapper


def test_metrics_keys_and_finite():
    device = torch.device("cpu")

    # simple sphere objective
    def sphere(x):
        return (x ** 2).sum(dim=-1)

    env = PSOEnv(
        landscape_fn=LandscapeWrapper(sphere, dim=2),
        num_agents=4,
        dim=2,
        batch_size=[2],
        device=device,
        max_steps=5,
        dynamic=True,
    )

    td = env.reset()

    # one step with random actions
    actions = torch.randn(env.batch_size[0], env.num_agents, 3, device=device)
    td_step = TensorDict({"action": actions}, batch_size=env.batch_size)
    td2 = env.step(td_step)

    metrics = env.get_metrics()

    # required keys
    for k in ["best_fitness", "diversity", "mean_speed", "global_best", "mean_pbest", "stagnation_frac"]:
        assert k in metrics

    # finite values
    for k, v in metrics.items():
        assert v == v  # not NaN
        assert abs(v) != float("inf")

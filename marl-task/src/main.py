import torch
import torch.nn as nn
import torch.distributions as d
from tensordict.nn import TensorDictModule, TensorDictSequential, CompositeDistribution
from torchrl.modules import MultiAgentMLP, ProbabilisticActor
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.envs import RewardSum, TransformedEnv

from envs import PSOEnv
from utils import LandscapeWrapper, PSOActionExtractor, PSOObservationWrapper

def eggholder(x: torch.Tensor) -> torch.Tensor:
    """
    Eggholder test function generalized for even-dimensional input.
    Args:
        x (torch.Tensor): Input tensor of shape (..., d), where d is even.
    Returns:
        torch.Tensor: Function value for each input in the batch.
    """
    # Ensure input has even number of dimensions
    if x.shape[-1] % 2 != 0:
        raise ValueError("Eggholder function requires even-dimensional input.")

    # Reshape to (..., d//2, 2) to pair up variables
    x_pairs = x.view(*x.shape[:-1], -1, 2)
    x_i = x_pairs[..., 0]
    x_j = x_pairs[..., 1]

    term1 = -(x_j + 47) * torch.sin(torch.sqrt(torch.abs(x_j + x_i / 2 + 47)))
    term2 = -x_i * torch.sin(torch.sqrt(torch.abs(x_i - (x_j + 47))))
    result = term1 + term2

    # Sum over all pairs for each sample
    return result.sum(dim=-1)

def square(x: torch.Tensor) -> torch.Tensor:
    return x * x

def main():
    landscape_dim = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    landscape_function = LandscapeWrapper(eggholder, dim=landscape_dim)
    env = PSOEnv(landscape=landscape_function,
                 num_agents=10,
                 device=device,
                 batch_size=(32,))
    
    env = TransformedEnv(
        env,
        RewardSum(in_keys=[env.reward_key], out_keys=[("agents", "episode_reward")]),
    )


    policy_kwargs = {
        "n_agent_inputs": 2 * landscape_dim,
        "n_agent_outputs": 3 * 2 * landscape_dim,  # 3 actions per agent, each with 2 dims, each with (mean, std)
        "n_agents": 10,
        "centralized": False,
        "share_params": True,
        "device": device,
        "num_cells": [32, 32],
        "dropout": 0.25
    }
    
    policy = ProbabilisticActor(
        TensorDictSequential(
            TensorDictModule(
                PSOObservationWrapper(),
                in_keys=["avg_pos", "avg_vel"],
                out_keys=["agent_input"]
            ),
            TensorDictModule(
                nn.Sequential(
                    MultiAgentMLP(**policy_kwargs),
                    NormalParamExtractor(), # Outputs a tuple (loc, scale)
                    PSOActionExtractor(dim=landscape_dim)
                ),
                in_keys=["agent_input"],
                out_keys=[
                    ("params", "inertia", "loc"), ("params", "inertia", "scale"),
                    ("params", "cognitive", "loc"), ("params", "cognitive", "scale"),
                    ("params", "social", "loc"), ("params", "social", "scale")
                ],
            ),
        ),
        in_keys=["params"],
        spec=env.action_spec,
        out_keys=["inertia", "cognitive", "social"],
        distribution_class=CompositeDistribution,
        distribution_kwargs={
            "distribution_map": {
                "inertia": d.Normal,
                "cognitive": d.Normal,
                "social": d.Normal
            },
            "name_map": {
                "inertia": ("action", "inertia"),
                "cognitive": ("action", "cognitive"),
                "social": ("action", "social")
            }
        },
        return_log_prob=True,
    )

    critic_kwargs = {
        "n_agent_inputs": 2 * landscape_dim,
        "n_agent_outputs": 1,
        "n_agents": 10,
        "centralized": True,
        "share_params": True,
        "device": device,
        "num_cells": [32, 32],
        "dropout": 0.25
    }

    critic = TensorDictSequential(
        TensorDictModule(
            PSOObservationWrapper(),
            in_keys=["avg_pos", "avg_vel"],
            out_keys=["agent_input"]
        ),
        TensorDictModule(
            MultiAgentMLP(**critic_kwargs),
            in_keys=["agent_input"],
            out_keys=["state_value"],
        ),
    )
    
    print("Running policy:", policy(env.reset()))
    print("Running value:", critic(env.reset()))

if __name__ == "__main__":
    main()

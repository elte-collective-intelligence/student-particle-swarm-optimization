import torch
import torch.nn as nn
import torch.distributions as d
from tensordict.nn import TensorDictModule, TensorDictSequential, CompositeDistribution
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, ValueOperator
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.envs import RewardSum, TransformedEnv
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.objectives import ClipPPOLoss, ValueEstimators
from matplotlib import pyplot as plt
from tqdm import tqdm

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

def train(env, policy, critic, collector, replay_buffer, loss_module, optim, total_frames, frames_per_batch, minibatch_size, num_epochs, max_grad_norm, gamma, lmbda, device):
    """PPO Training loop."""
    pbar = tqdm(total=total_frames)
    total_collected_frames = 0
    
    episode_rewards = []
    mean_rewards_log = []

    for i, data in enumerate(collector):
        total_collected_frames += data.numel()
        pbar.update(data.numel())

        # Store data in replay buffer
        replay_buffer.extend(data.reshape(-1))
        
        current_frames = data.numel()
        # Log episode rewards
        done_episodes = data[("next", "agents", "done")].any(dim=-1) # if any agent is done
        if done_episodes.any():
            episode_rewards.extend(data[("next", "agents", "episode_reward")][done_episodes].tolist())


        for epoch in range(num_epochs):
            for _ in range(frames_per_batch // minibatch_size):
                batch = replay_buffer.sample()
                
                # Compute GAE
                with torch.no_grad():
                    advantage = ValueEstimators.GAE(
                        gamma=gamma,
                        lmbda=lmbda,
                        state_value_key="state_value",
                        next_state_value_key="state_value", # critic will be called on next_obs
                        reward_key=("agents", "reward"),
                        done_key=("agents", "done"),
                        terminated_key=("agents", "terminated"),
                    )(batch, critic_params=critic.parameters(), target_critic_params=critic.parameters()) # Pass critic params if needed by GAE

                loss_td = loss_module(batch)
                loss = loss_td["loss_objective"] + loss_td["loss_critic"] + loss_td["loss_entropy"]

                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
                optim.step()
        
        if episode_rewards:
            mean_rewards_log.append(sum(episode_rewards) / len(episode_rewards))
            episode_rewards = [] # Reset for next logging interval

        if i % 10 == 0: # Log every 10 collections
            print(f"Iteration {i}: Total frames {total_collected_frames}, Mean reward (last 10 collections): {mean_rewards_log[-1] if mean_rewards_log else 'N/A'}")


    collector.shutdown()
    pbar.close()
    
    # Plotting rewards
    plt.figure(figsize=(10, 5))
    plt.plot(mean_rewards_log)
    plt.xlabel("Collection Iterations (x10)")
    plt.ylabel("Mean Episode Reward")
    plt.title("PPO Training Rewards")
    plt.savefig("ppo_rewards.png")
    plt.show()


def main():
    landscape_dim = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lmbda = 0.9
    gamma = 0.99
    lr = 3e-4

    #set_composite_lp_aggregate(True).set()

    frames_per_batch = 4096
    minibatch_size = 256
    n_iters = 5
    total_frames = frames_per_batch * n_iters

    clip_epsilon = 0.2
    entropy_coef = 0.01
    max_grad_norm = 0.5
    num_epochs = 4

    landscape_function = LandscapeWrapper(eggholder, dim=landscape_dim)
    env = PSOEnv(landscape=landscape_function,
                 num_agents=10,
                 device=device,
                 batch_size=(32,))
    
    env = TransformedEnv(
        env,
        RewardSum(in_keys=[env.reward_key], out_keys=[("agents", "episode_reward")]),
        device=device,
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

    critic = ValueOperator(TensorDictSequential(
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
    ))

    # Initialize PPO components here
    collector = SyncDataCollector(
        env,
        policy,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=device,
        storing_device=device,
        max_frames_per_traj=-1,
    )

    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(frames_per_batch, device=device),
        sampler=SamplerWithoutReplacement(),
        batch_size=minibatch_size,
    )

    loss_module = ClipPPOLoss(
        actor=policy,
        critic=critic,
        clip_epsilon=clip_epsilon,
        entropy_coef=entropy_coef,
        loss_critic_type="l2",
        normalize_advantage=True, # Recommended for PPO
    )
    loss_module.set_keys(
        reward = ("agents", "reward"),
        action = ("agents", "action"),
        done = ("agents", "done"),
        terminated = ("agents", "terminated"),
        # sample_log_prob = "sample_log_prob" # Handled by ProbabilisticActor
    )

    optim = torch.optim.Adam(loss_module.parameters(), lr=lr)

    # Call the train function
    train(
        env=env,
        policy=policy,
        critic=critic,
        collector=collector,
        replay_buffer=replay_buffer,
        loss_module=loss_module,
        optim=optim,
        total_frames=total_frames,
        frames_per_batch=frames_per_batch,
        minibatch_size=minibatch_size,
        num_epochs=num_epochs,
        max_grad_norm=max_grad_norm,
        gamma=gamma,
        lmbda=lmbda,
        device=device
    )


if __name__ == "__main__":
    main()

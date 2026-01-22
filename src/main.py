"""
Particle Swarm Optimization with Multi-Agent Reinforcement Learning.

This module implements PSO parameter learning using PPO (Proximal Policy Optimization).
Agents learn optimal inertia, cognitive, and social coefficients to maximize
optimization performance on various benchmark functions.

Usage:
    python src/main.py
    python src/main.py env.num_agents=20 model.learning_rate=1e-4
"""

import os
import torch
import torch.nn as nn
import torch.distributions as d
from tensordict.nn import TensorDictModule, TensorDictSequential, CompositeDistribution
from torchrl.modules import MultiAgentMLP, ProbabilisticActor
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.envs import RewardSum, TransformedEnv
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from matplotlib import pyplot as plt
from tqdm import tqdm
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from envs import PSOEnv
from envs.dynamic_functions import DynamicSphere, DynamicRastrigin, DynamicEggHolder
from utils import LandscapeWrapper, PSOActionExtractor, PSOObservationWrapper

# =============================================================================
# Landscape Functions
# =============================================================================


def eggholder(x: torch.Tensor) -> torch.Tensor:
    """
    Eggholder test function generalized for even-dimensional input.

    Args:
        x: Input tensor of shape (..., d), where d is even.

    Returns:
        Function value for each input in the batch.
    """
    if x.shape[-1] % 2 != 0:
        raise ValueError("Eggholder function requires even-dimensional input.")

    x_pairs = x.view(*x.shape[:-1], -1, 2)
    x_i = x_pairs[..., 0]
    x_j = x_pairs[..., 1]

    term1 = -(x_j + 47) * torch.sin(torch.sqrt(torch.abs(x_j + x_i / 2 + 47)))
    term2 = -x_i * torch.sin(torch.sqrt(torch.abs(x_i - (x_j + 47))))
    result = term1 + term2

    return result.sum(dim=-1)


def sphere(x: torch.Tensor) -> torch.Tensor:
    """Sphere function: f(x) = -sum(x^2). Optimum at origin."""
    return -torch.sum(x**2, dim=-1)


def rastrigin(x: torch.Tensor) -> torch.Tensor:
    """Rastrigin function: highly multimodal test function."""
    A = 10
    return -(A * x.shape[-1] + torch.sum(x**2 - A * torch.cos(2 * 3.14159 * x), dim=-1))


def get_landscape_function(name: str, dim: int):
    """
    Get a landscape function by name.

    Args:
        name: Function name (eggholder, sphere, rastrigin, dynamic_sphere, etc.)
        dim: Dimensionality of the search space.

    Returns:
        LandscapeWrapper containing the function.
    """
    static_functions = {
        "eggholder": eggholder,
        "sphere": sphere,
        "rastrigin": rastrigin,
    }

    if name in static_functions:
        return LandscapeWrapper(static_functions[name], dim=dim)

    dynamic_functions = {
        "dynamic_sphere": lambda d: DynamicSphere(dim=d),
        "dynamic_rastrigin": lambda d: DynamicRastrigin(dim=d),
        "dynamic_eggholder": lambda d: DynamicEggHolder(dim=d) if d == 2 else None,
    }

    if name in dynamic_functions:
        func = dynamic_functions[name](dim)
        if func is None:
            raise ValueError(f"Function {name} not available for dim={dim}")
        return func

    available = list(static_functions.keys()) + list(dynamic_functions.keys())
    raise ValueError(f"Unknown landscape function: {name}. Available: {available}")


# =============================================================================
# Training Loop
# =============================================================================


def compute_gae(data, critic_net, gamma, lmbda):
    """
    Compute Generalized Advantage Estimation properly.

    GAE(λ) = Σ (γλ)^t * δ_t
    where δ_t = r_t + γV(s_{t+1}) - V(s_t)
    """
    with torch.no_grad():
        # Compute values for all states
        data = critic_net(data)
        values = data["state_value"]
        if values.dim() > 2:
            values = values.squeeze(-1)

        # Compute next values
        next_data = data["next"]
        next_data = critic_net(next_data)
        next_values = next_data["state_value"]
        if next_values.dim() > 2:
            next_values = next_values.squeeze(-1)

        # Get rewards and done flags
        rewards = next_data[("agents", "reward")]
        dones = next_data[("agents", "done")].float()

        # Compute TD errors (deltas)
        # δ_t = r_t + γ * V(s_{t+1}) * (1 - done) - V(s_t)
        deltas = rewards + gamma * next_values * (1 - dones) - values

        # Compute GAE using the recursive formula
        # A_t = δ_t + γλ * A_{t+1}
        # We compute backwards through time
        batch_size = data.batch_size
        if len(batch_size) == 2:  # [batch, time]
            T = batch_size[1]
            advantages = torch.zeros_like(deltas)
            gae = torch.zeros_like(deltas[..., 0, :])  # [batch, agents]

            for t in reversed(range(T)):
                gae = deltas[..., t, :] + gamma * lmbda * (1 - dones[..., t, :]) * gae
                advantages[..., t, :] = gae
        else:
            # Fallback for flattened data
            advantages = deltas

        # Value targets for critic: V_target = A + V
        value_targets = advantages + values

        data["advantage"] = advantages
        data["value_target"] = value_targets

    return data


def train(
    env,
    policy,
    critic,
    replay_buffer,
    optim,
    num_epochs: int,
    max_grad_norm: float,
    n_iters: int,
    frames_per_batch: int,
    gamma: float,
    lmbda: float,
    clip_epsilon: float,
    entropy_coef: float,
    save_dir: str = None,
):
    """
    Train the PPO agents using manual rollouts.

    Uses env.rollout() instead of SyncDataCollector due to TorchRL
    compatibility issues with multi-agent environments.

    Args:
        env: The environment.
        policy: The policy network.
        critic: The critic network for value estimation.
        replay_buffer: Buffer for storing experience.
        loss_module: PPO loss module.
        optim: Optimizer.
        num_epochs: Number of PPO update epochs per batch.
        max_grad_norm: Maximum gradient norm for clipping.
        n_iters: Number of training iterations.
        frames_per_batch: Steps per rollout (approximated via max_steps).
        gamma: Discount factor.
        lmbda: GAE lambda (not used in simplified version).
        save_dir: Directory to save checkpoints.

    Returns:
        Tuple of (losses, rewards) lists.
    """
    losses = []
    rewards = []
    best_reward = float("-inf")

    # Calculate steps per rollout based on batch size
    batch_size = env.batch_size[0] if env.batch_size else 1
    max_steps = max(frames_per_batch // batch_size, 10)

    pbar = tqdm(range(n_iters), desc="Training Progress")

    for iteration in pbar:
        # Collect data using env.rollout (works better than SyncDataCollector)
        policy.eval()
        with torch.no_grad():
            data = env.rollout(max_steps=max_steps, policy=policy)

        # Compute GAE and value targets using our manual function
        data = compute_gae(data, critic, gamma, lmbda)

        # Store old log probs BEFORE flattening (need time dimension for GAE)
        # Compute total log prob per agent (sum over action dimensions)
        old_log_prob = (
            data["inertia_log_prob"].sum(-1)
            + data["cognitive_log_prob"].sum(-1)
            + data["social_log_prob"].sum(-1)
        )
        data["old_log_prob"] = old_log_prob

        # Flatten for replay buffer
        data_flat = data.view(-1)
        replay_buffer.extend(data_flat)

        # Training phase
        policy.train()
        critic.train()

        for epoch in range(num_epochs):
            for subdata in replay_buffer:
                # Get stored old log probs and advantages
                old_log_prob = subdata["old_log_prob"]
                advantages = subdata["advantage"]
                value_targets = subdata["value_target"]

                # Get the old actions that were taken
                old_inertia = subdata["inertia"]
                old_cognitive = subdata["cognitive"]
                old_social = subdata["social"]

                # Forward pass through policy to get distribution parameters
                # We need to evaluate log_prob of OLD actions under CURRENT policy
                subdata_new = policy.module[0](subdata)  # Get agent_input
                subdata_new = policy.module[1](subdata_new)  # Get distribution params

                # Build distributions and evaluate log_prob of old actions
                inertia_loc = subdata_new[("params", "inertia", "loc")]
                inertia_scale = subdata_new[("params", "inertia", "scale")]
                cognitive_loc = subdata_new[("params", "cognitive", "loc")]
                cognitive_scale = subdata_new[("params", "cognitive", "scale")]
                social_loc = subdata_new[("params", "social", "loc")]
                social_scale = subdata_new[("params", "social", "scale")]

                # Create distributions
                inertia_dist = torch.distributions.Normal(inertia_loc, inertia_scale)
                cognitive_dist = torch.distributions.Normal(
                    cognitive_loc, cognitive_scale
                )
                social_dist = torch.distributions.Normal(social_loc, social_scale)

                # Compute log probs of old actions under new policy
                new_inertia_lp = inertia_dist.log_prob(old_inertia).sum(-1)
                new_cognitive_lp = cognitive_dist.log_prob(old_cognitive).sum(-1)
                new_social_lp = social_dist.log_prob(old_social).sum(-1)
                new_log_prob = new_inertia_lp + new_cognitive_lp + new_social_lp

                # Compute entropy
                entropy = (
                    inertia_dist.entropy().sum(-1)
                    + cognitive_dist.entropy().sum(-1)
                    + social_dist.entropy().sum(-1)
                ).mean()

                # Compute value
                subdata_new = critic(subdata_new)
                values = subdata_new["state_value"].squeeze(-1)

                # Normalize advantages (per minibatch)
                adv_normalized = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )

                # Compute PPO ratio
                ratio = torch.exp(new_log_prob - old_log_prob.detach())

                # Clipped surrogate objective (mean over agents too)
                surr1 = ratio * adv_normalized
                surr2 = (
                    torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
                    * adv_normalized
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (with optional clipping)
                value_loss = 0.5 * ((values - value_targets.detach()) ** 2).mean()

                # Entropy bonus (encourage exploration)
                entropy_loss = -entropy_coef * entropy

                # Total loss
                loss_val = policy_loss + value_loss + entropy_loss

                optim.zero_grad()
                loss_val.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(policy.parameters()) + list(critic.parameters()), max_grad_norm
                )
                optim.step()

        # Logging
        episode_reward = data[("agents", "episode_reward")][..., -1].mean().item()
        rewards.append(episode_reward)
        losses.append(loss_val.item())

        # Track best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            if save_dir:
                save_checkpoint_models(
                    policy,
                    critic,
                    optim,
                    iteration,
                    episode_reward,
                    os.path.join(save_dir, "best_model.pt"),
                )

        pbar.set_postfix(
            {
                "Reward": f"{episode_reward:.3f}",
                "Best": f"{best_reward:.3f}",
                "Loss": f"{loss_val.item():.3f}",
            }
        )

        # Clear replay buffer for next iteration
        replay_buffer.empty()

    pbar.close()

    # Save final model
    if save_dir:
        save_checkpoint_models(
            policy,
            critic,
            optim,
            iteration,
            episode_reward,
            os.path.join(save_dir, "final_model.pt"),
        )

    return losses, rewards


def save_checkpoint(loss_module, optim, iteration, reward, path):
    """Save model checkpoint (for loss_module based training)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "iteration": iteration,
            "actor_state_dict": loss_module.actor.state_dict(),
            "critic_state_dict": loss_module.critic.state_dict(),
            "optimizer_state_dict": optim.state_dict(),
            "reward": reward,
        },
        path,
    )


def save_checkpoint_models(policy, critic, optim, iteration, reward, path):
    """Save model checkpoint for separate policy/critic networks."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "iteration": iteration,
            "policy_state_dict": policy.state_dict(),
            "critic_state_dict": critic.state_dict(),
            "optimizer_state_dict": optim.state_dict(),
            "reward": reward,
        },
        path,
    )


def load_checkpoint(path, loss_module, optim=None):
    """Load model checkpoint."""
    checkpoint = torch.load(path)
    loss_module.actor.load_state_dict(checkpoint["actor_state_dict"])
    loss_module.critic.load_state_dict(checkpoint["critic_state_dict"])
    if optim is not None:
        optim.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"], checkpoint["reward"]


# =============================================================================
# Main Entry Point
# =============================================================================


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """Main training function."""
    print("=" * 60)
    print("PSO Multi-Agent RL Training")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))

    # Extract configuration values
    landscape_dim = cfg.env.landscape_dim
    num_agents = cfg.env.num_agents
    batch_size = cfg.env.batch_size
    delta = cfg.env.delta
    landscape_name = cfg.env.landscape_function

    hidden_sizes = list(cfg.model.hidden_sizes)
    lr = cfg.model.learning_rate
    dropout = cfg.model.dropout
    centralized_critic = cfg.model.centralized_critic
    share_params = cfg.model.share_params

    frames_per_batch = cfg.frames_per_batch
    minibatch_size = cfg.minibatch_size
    n_iters = cfg.n_iters
    num_epochs = cfg.num_epochs
    max_grad_norm = cfg.max_grad_norm
    clip_epsilon = cfg.clip_epsilon
    entropy_coef = cfg.entropy_coef
    gamma = cfg.gamma
    lmbda = cfg.lmbda

    save_model = cfg.save_model
    save_plot = cfg.save_plot
    output_dir = os.path.join(get_original_cwd(), cfg.output_dir)

    # Derived values
    total_frames = frames_per_batch * n_iters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nDevice: {device}")
    print(f"Landscape: {landscape_name} ({landscape_dim}D)")
    print(f"Agents: {num_agents}")
    print(f"Total frames: {total_frames}")
    print("=" * 60)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize environment
    landscape_function = get_landscape_function(landscape_name, landscape_dim)
    env = PSOEnv(
        landscape=landscape_function,
        num_agents=num_agents,
        device=device,
        batch_size=(batch_size,),
        delta=delta,
    )

    env = TransformedEnv(
        env,
        RewardSum(in_keys=[env.reward_key], out_keys=[("agents", "episode_reward")]),
        device=device,
    )

    # Policy network
    policy_kwargs = {
        "n_agent_inputs": 2 * landscape_dim,
        "n_agent_outputs": 3 * 2 * landscape_dim,
        "n_agents": num_agents,
        "centralized": False,
        "share_params": share_params,
        "device": device,
        "num_cells": hidden_sizes,
        "dropout": dropout,
    }

    policy = ProbabilisticActor(
        TensorDictSequential(
            TensorDictModule(
                PSOObservationWrapper(),
                in_keys=["avg_pos", "avg_vel"],
                out_keys=["agent_input"],
            ),
            TensorDictModule(
                nn.Sequential(
                    MultiAgentMLP(**policy_kwargs),
                    NormalParamExtractor(),
                    PSOActionExtractor(dim=landscape_dim),
                ),
                in_keys=["agent_input"],
                out_keys=[
                    ("params", "inertia", "loc"),
                    ("params", "inertia", "scale"),
                    ("params", "cognitive", "loc"),
                    ("params", "cognitive", "scale"),
                    ("params", "social", "loc"),
                    ("params", "social", "scale"),
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
                "social": d.Normal,
            },
        },
        return_log_prob=True,
    )

    # Critic network
    critic_kwargs = {
        "n_agent_inputs": 2 * landscape_dim,
        "n_agent_outputs": 1,
        "n_agents": num_agents,
        "centralized": centralized_critic,
        "share_params": share_params,
        "device": device,
        "num_cells": hidden_sizes,
        "dropout": dropout,
    }

    # Create the critic network (raw TensorDictSequential for manual GAE)
    critic_net = TensorDictSequential(
        TensorDictModule(
            PSOObservationWrapper(),
            in_keys=["avg_pos", "avg_vel"],
            out_keys=["agent_input"],
        ),
        TensorDictModule(
            MultiAgentMLP(**critic_kwargs),
            in_keys=["agent_input"],
            out_keys=["state_value"],
        ),
    )

    # Replay buffer (sized for one rollout batch)
    rollout_size = frames_per_batch  # Will be adjusted based on actual rollout
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(rollout_size, device=device),
        sampler=SamplerWithoutReplacement(),
        batch_size=minibatch_size,
    )

    # Optimizer for both policy and critic
    optim = torch.optim.Adam(
        list(policy.parameters()) + list(critic_net.parameters()), lr=lr
    )

    # Train using manual rollouts and custom PPO update
    # (SyncDataCollector and built-in ClipPPOLoss have issues with composite actions)
    save_dir = output_dir if save_model else None
    losses, rewards = train(
        env=env,
        policy=policy,
        critic=critic_net,
        replay_buffer=replay_buffer,
        optim=optim,
        num_epochs=num_epochs,
        max_grad_norm=max_grad_norm,
        n_iters=n_iters,
        frames_per_batch=frames_per_batch,
        gamma=gamma,
        lmbda=lmbda,
        clip_epsilon=clip_epsilon,
        entropy_coef=entropy_coef,
        save_dir=save_dir,
    )

    # Plot and save results
    if save_plot:
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(rewards)
        plt.title("Episode Rewards")
        plt.xlabel("Iteration")
        plt.ylabel("Mean Reward")
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.plot(losses)
        plt.title("Training Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(output_dir, "training_results.png")
        plt.savefig(plot_path, dpi=150)
        print(f"\nPlot saved to: {plot_path}")

    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Final reward: {rewards[-1]:.3f}")
    print(f"Best reward: {max(rewards):.3f}")
    if save_model:
        print(f"Models saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

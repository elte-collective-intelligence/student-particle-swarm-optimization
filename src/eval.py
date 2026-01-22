"""
Evaluation script for PSO with trained RL agents.

This module provides comprehensive evaluation of trained PSO policies,
including comparison with random/baseline policies and visualization.

Usage:
    python src/eval.py model_path=outputs/best_model.pt
    python src/eval.py model_path=outputs/best_model.pt vis_configs=full
"""

import os
import torch
import torch.nn as nn
import torch.distributions as d
from tensordict.nn import TensorDictModule, TensorDictSequential, CompositeDistribution
from torchrl.modules import MultiAgentMLP, ProbabilisticActor
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.envs import RewardSum, TransformedEnv
import numpy as np
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from envs import PSOEnv
from envs.dynamic_functions import DynamicSphere, DynamicRastrigin, DynamicEggHolder
from utils import LandscapeWrapper, PSOActionExtractor, PSOObservationWrapper
from visualization import SwarmVisualizer


# =============================================================================
# Landscape Functions
# =============================================================================


def eggholder(x: torch.Tensor) -> torch.Tensor:
    """Eggholder test function for even-dimensional input."""
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
    """Get a landscape function by name."""
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
# Policy Creation
# =============================================================================


def create_policy(env, num_agents, dim, hidden_sizes, share_params, dropout, device):
    """Create a policy network matching the training setup."""
    policy_kwargs = {
        "n_agent_inputs": 2 * dim,
        "n_agent_outputs": 3 * 2 * dim,
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
                    PSOActionExtractor(dim=dim, transform_actions=True),
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

    return policy


def create_random_policy(dim, device):
    """Create a random policy that outputs fixed standard PSO parameters."""

    class RandomPSOPolicy(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.dim = dim

        def forward(self, tensordict):
            batch_shape = tensordict["avg_pos"].shape[:-1]

            # Standard PSO parameters with small noise
            inertia = 0.7 + 0.1 * torch.randn(*batch_shape, self.dim, device=device)
            cognitive = 1.5 + 0.2 * torch.randn(*batch_shape, self.dim, device=device)
            social = 1.5 + 0.2 * torch.randn(*batch_shape, self.dim, device=device)

            tensordict["inertia"] = inertia
            tensordict["cognitive"] = cognitive
            tensordict["social"] = social

            return tensordict

    return RandomPSOPolicy(dim)


# =============================================================================
# Evaluation Functions
# =============================================================================


def evaluate_policy(
    env,
    policy,
    num_episodes: int,
    max_steps: int,
    visualizer: SwarmVisualizer = None,
    policy_name: str = "policy",
):
    """
    Evaluate a policy over multiple episodes.

    Args:
        env: The environment
        policy: The policy to evaluate
        num_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode
        visualizer: Optional visualizer for recording frames
        policy_name: Name for logging

    Returns:
        Dictionary of evaluation metrics
    """
    all_final_scores = []
    all_best_scores = []
    all_cumulative_rewards = []
    all_convergence_curves = []

    # Get base environment (unwrap TransformedEnv if needed)
    base_env = env.base_env if hasattr(env, "base_env") else env

    for ep in tqdm(range(num_episodes), desc=f"Evaluating {policy_name}"):
        # Reset visualizer if provided
        if visualizer:
            visualizer.reset(episode=ep)

        data = env.reset()
        episode_reward = 0
        best_scores_curve = []
        mean_scores_curve = []

        for step in range(max_steps):
            # Get action
            if hasattr(policy, "module"):
                # TorchRL policy
                with torch.no_grad():
                    data = policy(data)
            else:
                # Simple policy (random)
                with torch.no_grad():
                    data = policy(data)

            # Step environment
            data = env.step(data)

            # Record metrics
            reward = data["next", "agents", "reward"].mean().item()
            episode_reward += reward

            # Get scores from environment
            scores = base_env.scores[0].cpu()  # First batch
            best_score = scores.max().item()
            mean_score = scores.mean().item()

            best_scores_curve.append(best_score)
            mean_scores_curve.append(mean_score)

            # Record frame for visualization
            if visualizer and visualizer.visualize_swarm:
                # Compute global best from personal bests
                pb_scores = base_env.personal_best_scores[0]  # [agents]
                best_idx = pb_scores.argmax()
                global_best = base_env.personal_best_pos[0, best_idx].unsqueeze(0)

                visualizer.record_frame(
                    positions=base_env.positions,
                    velocities=base_env.velocities,
                    personal_bests=base_env.personal_best_pos,
                    global_best=global_best,
                    scores=base_env.scores,
                    timestep=step,
                )

            # Prepare for next step
            data = data["next"]

        # Episode metrics
        final_best_score = best_scores_curve[-1]
        all_final_scores.append(final_best_score)
        all_best_scores.append(max(best_scores_curve))
        all_cumulative_rewards.append(episode_reward)
        all_convergence_curves.append(best_scores_curve)

        # Save visualizations for first episode
        if visualizer and ep == 0:
            visualizer.save_all_visualizations(
                best_scores=best_scores_curve, mean_scores=mean_scores_curve
            )

    # Compute aggregate metrics
    metrics = {
        "policy_name": policy_name,
        "num_episodes": num_episodes,
        "max_steps": max_steps,
        "mean_final_score": np.mean(all_final_scores),
        "std_final_score": np.std(all_final_scores),
        "mean_best_score": np.mean(all_best_scores),
        "std_best_score": np.std(all_best_scores),
        "mean_cumulative_reward": np.mean(all_cumulative_rewards),
        "std_cumulative_reward": np.std(all_cumulative_rewards),
    }

    return metrics, all_convergence_curves


def compare_policies(metrics_list: list, output_dir: str):
    """
    Create comparison plots and save results.

    Args:
        metrics_list: List of metric dictionaries from different policies
        output_dir: Directory to save comparison results
    """
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    # Bar chart comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    names = [m["policy_name"] for m in metrics_list]
    x = np.arange(len(names))
    width = 0.35

    # Final score comparison
    ax = axes[0]
    means = [m["mean_final_score"] for m in metrics_list]
    stds = [m["std_final_score"] for m in metrics_list]
    ax.bar(x, means, width, yerr=stds, capsize=5)
    ax.set_ylabel("Score")
    ax.set_title("Final Best Score")
    ax.set_xticks(x)
    ax.set_xticklabels(names)

    # Best score comparison
    ax = axes[1]
    means = [m["mean_best_score"] for m in metrics_list]
    stds = [m["std_best_score"] for m in metrics_list]
    ax.bar(x, means, width, yerr=stds, capsize=5, color="green")
    ax.set_ylabel("Score")
    ax.set_title("Best Score Achieved")
    ax.set_xticks(x)
    ax.set_xticklabels(names)

    # Cumulative reward comparison
    ax = axes[2]
    means = [m["mean_cumulative_reward"] for m in metrics_list]
    stds = [m["std_cumulative_reward"] for m in metrics_list]
    ax.bar(x, means, width, yerr=stds, capsize=5, color="orange")
    ax.set_ylabel("Reward")
    ax.set_title("Cumulative Reward")
    ax.set_xticks(x)
    ax.set_xticklabels(names)

    plt.tight_layout()
    comparison_path = os.path.join(output_dir, "policy_comparison.png")
    plt.savefig(comparison_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nSaved comparison plot: {comparison_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"{'Policy':<20} {'Final Score':>15} {'Best Score':>15} {'Reward':>15}")
    print("-" * 70)
    for m in metrics_list:
        print(
            f"{m['policy_name']:<20} "
            f"{m['mean_final_score']:>12.3f}±{m['std_final_score']:.2f} "
            f"{m['mean_best_score']:>12.3f}±{m['std_best_score']:.2f} "
            f"{m['mean_cumulative_reward']:>12.3f}±{m['std_cumulative_reward']:.2f}"
        )
    print("=" * 70)


# =============================================================================
# Main Entry Point
# =============================================================================


@hydra.main(version_base=None, config_path="configs", config_name="eval_config")
def main(cfg: DictConfig):
    """Main evaluation function."""
    print("=" * 60)
    print("PSO Multi-Agent RL Evaluation")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))

    # Extract configuration
    landscape_dim = cfg.env.landscape_dim
    num_agents = cfg.env.num_agents
    landscape_name = cfg.env.landscape_function
    delta = cfg.env.delta

    hidden_sizes = list(cfg.model.hidden_sizes)
    dropout = cfg.model.dropout
    share_params = cfg.model.share_params

    num_eval_episodes = cfg.eval.num_eval_episodes
    max_steps = cfg.eval.max_steps
    compare_random = cfg.eval.compare_random
    save_metrics = cfg.eval.save_metrics

    model_path = os.path.join(get_original_cwd(), cfg.model_path)
    output_dir = os.path.join(get_original_cwd(), cfg.output_dir)

    # Visualization config
    vis_config = OmegaConf.to_container(cfg.visualization, resolve=True)
    # Fix visualization save_dir path
    if "save_dir" in vis_config:
        vis_config["save_dir"] = os.path.join(get_original_cwd(), vis_config["save_dir"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Model: {model_path}")
    print(f"Landscape: {landscape_name} ({landscape_dim}D)")
    print(f"Episodes: {num_eval_episodes}, Steps: {max_steps}")
    print("=" * 60)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(vis_config.get("save_dir", "outputs/vis/"), exist_ok=True)

    # Initialize environment
    landscape_function = get_landscape_function(landscape_name, landscape_dim)
    env = PSOEnv(
        landscape=landscape_function,
        num_agents=num_agents,
        device=device,
        batch_size=(1,),  # Single batch for evaluation
        delta=delta,
    )

    env = TransformedEnv(
        env,
        RewardSum(in_keys=[env.reward_key], out_keys=[("agents", "episode_reward")]),
        device=device,
    )

    # Create visualizer
    visualizer = None
    if vis_config.get("visualize_swarm", False):
        visualizer = SwarmVisualizer(
            vis_config=vis_config,
            landscape_fn=landscape_function,
            dim=landscape_dim,
        )

    # Load trained policy
    policy = create_policy(
        env, num_agents, landscape_dim, hidden_sizes, share_params, dropout, device
    )

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        policy.load_state_dict(checkpoint["policy_state_dict"])
        print(f"Loaded model from {model_path}")
        print(f"  Training iteration: {checkpoint.get('iteration', 'unknown')}")
        print(f"  Training reward: {checkpoint.get('reward', 'unknown'):.3f}")
    else:
        print(f"WARNING: Model not found at {model_path}, using random initialization")

    policy.eval()

    # Evaluate trained policy
    metrics_list = []

    trained_metrics, trained_curves = evaluate_policy(
        env,
        policy,
        num_episodes=num_eval_episodes,
        max_steps=max_steps,
        visualizer=visualizer,
        policy_name="Trained Policy",
    )
    metrics_list.append(trained_metrics)

    # Optionally evaluate random baseline
    if compare_random:
        random_policy = create_random_policy(landscape_dim, device)
        random_metrics, random_curves = evaluate_policy(
            env,
            random_policy,
            num_episodes=num_eval_episodes,
            max_steps=max_steps,
            visualizer=None,  # Don't visualize random
            policy_name="Random Baseline",
        )
        metrics_list.append(random_metrics)

    # Compare and save results
    compare_policies(metrics_list, output_dir)

    # Save metrics to file
    if save_metrics:
        import json

        metrics_path = os.path.join(output_dir, "eval_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics_list, f, indent=2)
        print(f"Saved metrics: {metrics_path}")

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()

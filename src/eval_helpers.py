"""
Shared helpers used by both eval.py and eval_multi_topology.py.

Extracted here to avoid circular imports and code duplication.
"""

from __future__ import annotations

import os
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as d
from tensordict.nn import TensorDictModule, TensorDictSequential, CompositeDistribution
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.modules import MultiAgentMLP, ProbabilisticActor
from torchrl.envs import RewardSum, TransformedEnv
from tqdm import tqdm

from envs import PSOEnv
from envs.dynamic_functions import DynamicSphere, DynamicRastrigin, DynamicEggHolder
from utils import LandscapeWrapper, PSOActionExtractor, PSOObservationWrapper
from eval.diversity import DiversityTracker
from eval.information_spread import InformationSpreadTracker


# =============================================================================
# Landscape helpers
# =============================================================================


def eggholder(x: torch.Tensor) -> torch.Tensor:
    """Eggholder test function for even-dimensional input."""
    if x.shape[-1] % 2 != 0:
        raise ValueError("Eggholder function requires even-dimensional input.")
    x_pairs = x.view(*x.shape[:-1], -1, 2)
    x_i, x_j = x_pairs[..., 0], x_pairs[..., 1]
    term1 = -(x_j + 47) * torch.sin(torch.sqrt(torch.abs(x_j + x_i / 2 + 47)))
    term2 = -x_i * torch.sin(torch.sqrt(torch.abs(x_i - (x_j + 47))))
    return (term1 + term2).sum(dim=-1)


def sphere(x: torch.Tensor) -> torch.Tensor:
    return -torch.sum(x**2, dim=-1)


def rastrigin(x: torch.Tensor) -> torch.Tensor:
    A = 10
    return -(A * x.shape[-1] + torch.sum(x**2 - A * torch.cos(2 * 3.14159 * x), dim=-1))


def get_landscape_function(name: str, dim: int):
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

    available = list(static_functions) + list(dynamic_functions)
    raise ValueError(f"Unknown landscape function: {name}. Available: {available}")


# =============================================================================
# Policy helpers
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
    """Create a random policy that outputs standard PSO parameters with noise."""

    class RandomPSOPolicy(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.dim = dim

        def forward(self, tensordict):
            batch_shape = tensordict["avg_pos"].shape[:-1]
            tensordict["inertia"] = 0.7 + 0.1 * torch.randn(
                *batch_shape, self.dim, device=device
            )
            tensordict["cognitive"] = 1.5 + 0.2 * torch.randn(
                *batch_shape, self.dim, device=device
            )
            tensordict["social"] = 1.5 + 0.2 * torch.randn(
                *batch_shape, self.dim, device=device
            )
            return tensordict

    return RandomPSOPolicy(dim)


# =============================================================================
# Core evaluation loop
# =============================================================================


def evaluate_policy(
    env,
    policy,
    num_episodes: int,
    max_steps: int,
    visualizer=None,
    policy_name: str = "policy",
    collect_diversity: bool = True,
    collect_info_spread: bool = True,
):
    """Evaluate a policy over multiple episodes.

    Returns
    -------
    tuple
        (aggregate_metrics_dict, convergence_curves, episode_histories)
    """
    all_final_scores, all_best_scores, all_cumulative_rewards = [], [], []
    all_convergence_curves: list = []
    all_episode_histories: list = []

    base_env = env.base_env if hasattr(env, "base_env") else env

    diversity_tracker = DiversityTracker(batch_index=0) if collect_diversity else None
    info_tracker = (
        InformationSpreadTracker(batch_index=0) if collect_info_spread else None
    )

    for ep in tqdm(range(num_episodes), desc=f"Evaluating {policy_name}"):
        if visualizer:
            visualizer.reset(episode=ep)
        if diversity_tracker:
            diversity_tracker.reset()
        if info_tracker:
            info_tracker.reset()

        data = env.reset()
        episode_reward = 0.0
        best_scores_curve, mean_scores_curve = [], []

        for step in range(max_steps):
            with torch.no_grad():
                data = policy(data)
            data = env.step(data)

            reward = data["next", "agents", "reward"].mean().item()
            episode_reward += reward

            scores = base_env.scores[0].cpu()
            best_scores_curve.append(scores.max().item())
            mean_scores_curve.append(scores.mean().item())

            if diversity_tracker is not None:
                diversity_tracker.update(
                    positions=base_env.positions,
                    velocities=base_env.velocities,
                )

            if (
                info_tracker is not None
                and base_env.neighborhood_best_scores is not None
            ):
                info_tracker.update(
                    personal_best_scores=base_env.personal_best_scores,
                    neighborhood_best_scores=base_env.neighborhood_best_scores,
                )

            if visualizer and visualizer.visualize_swarm:
                pb_scores = base_env.personal_best_scores[0]
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

            data = data["next"]

        all_final_scores.append(best_scores_curve[-1])
        all_best_scores.append(max(best_scores_curve))
        all_cumulative_rewards.append(episode_reward)
        all_convergence_curves.append(best_scores_curve)

        ep_history: dict = {"episode": ep}
        if diversity_tracker is not None:
            ep_history["diversity"] = diversity_tracker.summarise()
        if info_tracker is not None:
            ep_history["info_spread"] = info_tracker.summarise()
        all_episode_histories.append(ep_history)

        if visualizer and ep == 0:
            visualizer.save_all_visualizations(
                best_scores=best_scores_curve, mean_scores=mean_scores_curve
            )

    metrics = {
        "policy_name": policy_name,
        "num_episodes": num_episodes,
        "max_steps": max_steps,
        "mean_final_score": float(np.mean(all_final_scores)),
        "std_final_score": float(np.std(all_final_scores)),
        "mean_best_score": float(np.mean(all_best_scores)),
        "std_best_score": float(np.std(all_best_scores)),
        "mean_cumulative_reward": float(np.mean(all_cumulative_rewards)),
        "std_cumulative_reward": float(np.std(all_cumulative_rewards)),
    }

    # Roll up diversity
    if (
        collect_diversity
        and all_episode_histories
        and "diversity" in all_episode_histories[0]
    ):
        for key in [
            "mean_pairwise_dist",
            "position_spread",
            "velocity_alignment",
            "velocity_speed_std",
        ]:
            ep_vals = [
                float(np.mean(h["diversity"][key]))
                for h in all_episode_histories
                if "diversity" in h and h["diversity"].get(key)
            ]
            if ep_vals:
                metrics[f"diversity_{key}"] = float(np.mean(ep_vals))

    # Roll up info-spread
    if (
        collect_info_spread
        and all_episode_histories
        and "info_spread" in all_episode_histories[0]
    ):
        for key in [
            "mean_adoption_fraction",
            "mean_information_entropy",
            "mean_adoption_rate",
        ]:
            ep_vals = [
                h["info_spread"].get(key, 0.0)
                for h in all_episode_histories
                if "info_spread" in h
            ]
            if ep_vals:
                metrics[f"info_{key}"] = float(np.mean(ep_vals))
        for key in ["num_spread_events", "mean_steps_to_90pct"]:
            ep_vals = [
                h["info_spread"][key]
                for h in all_episode_histories
                if "info_spread" in h and h["info_spread"].get(key) is not None
            ]
            if ep_vals:
                metrics[f"info_{key}"] = float(np.mean(ep_vals))

    _print_metric_summary(policy_name, all_episode_histories)
    return metrics, all_convergence_curves, all_episode_histories


def _print_metric_summary(policy_name: str, episode_histories: list) -> None:
    """Print a compact diversity / info-spread summary after evaluating a policy."""
    if not episode_histories:
        return
    div_rows = [h["diversity"] for h in episode_histories if "diversity" in h]
    isp_rows = [h["info_spread"] for h in episode_histories if "info_spread" in h]
    print(
        f"\n  [{policy_name}] Metric summary over {len(episode_histories)} episode(s):"
    )
    if div_rows:
        for key, label in [
            ("mean_pairwise_dist", "Mean pairwise dist"),
            ("position_spread", "Position spread   "),
            ("velocity_alignment", "Velocity alignment"),
        ]:
            vals = [float(np.mean(r[key])) for r in div_rows if r.get(key)]
            if vals:
                print(f"    {label}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    if isp_rows:
        adopt = [r.get("mean_adoption_fraction", 0.0) for r in isp_rows]
        events = [r.get("num_spread_events", 0) for r in isp_rows]
        t90 = [
            r["mean_steps_to_90pct"]
            for r in isp_rows
            if r.get("mean_steps_to_90pct") is not None
        ]
        print(f"    Adoption fraction : {np.mean(adopt):.4f} ± {np.std(adopt):.4f}")
        print(f"    Spread events/ep  : {np.mean(events):.1f}")
        if t90:
            print(f"    Steps to 90% adopt: {np.mean(t90):.1f} ± {np.std(t90):.1f}")

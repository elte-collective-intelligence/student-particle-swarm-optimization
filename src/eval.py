"""
Evaluation script for PSO with trained RL agents.

This module provides comprehensive evaluation of trained PSO policies,
including comparison with random/baseline policies and visualization.

Usage:
    python src/eval.py model_path=outputs/best_model.pt
    python src/eval.py model_path=outputs/best_model.pt vis_configs=full
"""

import os
import random
import csv
import json

import numpy as np
import torch
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from torchrl.envs import RewardSum, TransformedEnv

from envs import PSOEnv
from visualization import SwarmVisualizer

# Shared helpers (also used by eval_multi_topology.py)
from eval_helpers import (
    get_landscape_function,
    create_policy,
    create_random_policy,
    evaluate_policy,
)

# =============================================================================
# Evaluation-only helpers (saving, comparison plots)
# =============================================================================


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


def save_evaluation_metrics(
    output_dir: str,
    metrics_list: list,
    histories_by_policy: list,
    save_curves: bool = True,
) -> None:
    """
    Save evaluation metrics to JSON and CSV.

    Args:
        output_dir: Directory to write files into
        metrics_list: Aggregate metric dicts, one per policy
        histories_by_policy: Per-episode history lists, one per policy
        save_curves: Include per-step curve lists in the JSON output
    """
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------ JSON
    def _strip_curves(h: dict) -> dict:
        """Replace list-valued curve fields with their mean scalar."""
        out = {"episode": h["episode"]}
        if "diversity" in h:
            out["diversity"] = {
                k: float(np.mean(v)) if isinstance(v, list) and v else v
                for k, v in h["diversity"].items()
            }
        if "info_spread" in h:
            out["info_spread"] = {
                k: v for k, v in h["info_spread"].items() if not isinstance(v, list)
            }
        return out

    json_payload = {
        "aggregate_metrics": metrics_list,
        "episode_histories": {
            m["policy_name"]: (
                histories if save_curves else [_strip_curves(h) for h in histories]
            )
            for m, histories in zip(metrics_list, histories_by_policy)
        },
    }
    json_path = os.path.join(output_dir, "eval_metrics.json")
    with open(json_path, "w") as f:
        json.dump(json_payload, f, indent=2, default=float)
    print(f"Saved metrics  (JSON): {json_path}")

    # ------------------------------------------------------------------ CSV
    csv_path = os.path.join(output_dir, "eval_metrics_summary.csv")
    fieldnames = [
        "policy_name",
        "episode",
        # diversity
        "mean_pairwise_dist",
        "position_spread",
        "velocity_alignment",
        "velocity_speed_std",
        # information spread
        "mean_adoption_fraction",
        "mean_information_entropy",
        "num_spread_events",
        "mean_steps_to_50pct",
        "mean_steps_to_90pct",
        "mean_adoption_rate",
        "final_adoption_fraction",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for metrics, histories in zip(metrics_list, histories_by_policy):
            name = metrics["policy_name"]
            for h in histories:
                row: dict = {"policy_name": name, "episode": h["episode"]}
                if "diversity" in h:
                    div = h["diversity"]
                    for key in [
                        "mean_pairwise_dist",
                        "position_spread",
                        "velocity_alignment",
                        "velocity_speed_std",
                    ]:
                        vals = div.get(key, [])
                        row[key] = float(np.mean(vals)) if vals else ""
                if "info_spread" in h:
                    isp = h["info_spread"]
                    for key in [
                        "mean_adoption_fraction",
                        "mean_information_entropy",
                        "num_spread_events",
                        "mean_steps_to_50pct",
                        "mean_steps_to_90pct",
                        "mean_adoption_rate",
                        "final_adoption_fraction",
                    ]:
                        val = isp.get(key)
                        row[key] = "" if val is None else val
                writer.writerow(row)
    print(f"Saved metrics  (CSV) : {csv_path}")


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
    topology_cfg = cfg.get("topology", None)

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
        vis_config["save_dir"] = os.path.join(
            get_original_cwd(), vis_config["save_dir"]
        )

    seed = cfg.get("seed", 42)

    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Seed: {seed}")
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
        topology_config=topology_cfg,
    )
    env.set_seed(seed)

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

    trained_metrics, trained_curves, trained_histories = evaluate_policy(
        env,
        policy,
        num_episodes=num_eval_episodes,
        max_steps=max_steps,
        visualizer=visualizer,
        policy_name="Trained Policy",
        collect_diversity=cfg.eval.get("collect_diversity", True),
        collect_info_spread=cfg.eval.get("collect_info_spread", True),
    )
    metrics_list.append(trained_metrics)
    histories_by_policy = [trained_histories]

    # Optionally evaluate random baseline
    if compare_random:
        random_policy = create_random_policy(landscape_dim, device)
        random_metrics, random_curves, random_histories = evaluate_policy(
            env,
            random_policy,
            num_episodes=num_eval_episodes,
            max_steps=max_steps,
            visualizer=None,  # Don't visualize random
            policy_name="Random Baseline",
            collect_diversity=cfg.eval.get("collect_diversity", True),
            collect_info_spread=cfg.eval.get("collect_info_spread", True),
        )
        metrics_list.append(random_metrics)
        histories_by_policy.append(random_histories)

    # Compare and save results
    compare_policies(metrics_list, output_dir)

    # Save metrics (JSON + CSV)
    if save_metrics:
        save_evaluation_metrics(
            output_dir=output_dir,
            metrics_list=metrics_list,
            histories_by_policy=histories_by_policy,
            save_curves=cfg.eval.get("save_metric_curves", True),
        )

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()

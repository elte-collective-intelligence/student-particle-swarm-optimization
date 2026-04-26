"""
Multi-topology evaluation script for PSO.

Runs every topology listed in ``cfg.topologies`` under the same landscape
function and random seed, then collects and compares:
  - Convergence metrics  (AUC, time-to-X%, improvement rate, plateau fraction)
  - Diversity metrics    (mean pairwise dist, spread, velocity alignment)
  - Information-flow metrics (adoption fraction, entropy, spread events)

Usage (from repo root, inside Docker or venv):
    python src/eval_multi_topology.py
    python src/eval_multi_topology.py seed=123 env.landscape_function=rastrigin
    python src/eval_multi_topology.py topologies=[global,ring]   # override subset

Docker helper:
    docker run --rm -it -v "${PWD}:/app" student_pso \\
        python src/eval_multi_topology.py
"""

import os
import csv
import json
import random
import shutil
from datetime import datetime

import numpy as np
import torch
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from torchrl.envs import RewardSum, TransformedEnv

from envs import PSOEnv
from eval import aggregate_convergence_metrics
from eval.convergence import compute_episode_convergence
from eval.comparison_plots import ComparisonPlotter

# Shared helpers (landscape, policy, evaluate_policy loop)
from eval_helpers import (
    get_landscape_function,
    create_policy,
    create_random_policy,
    evaluate_policy,
)

# =============================================================================
# Topology-aware environment factory
# =============================================================================


def _make_env(cfg: DictConfig, topology_cfg: dict, seed: int, device):
    """Build a fresh PSOEnv + TransformedEnv for one topology."""
    landscape_fn = get_landscape_function(
        cfg.env.landscape_function, cfg.env.landscape_dim
    )
    env = PSOEnv(
        landscape=landscape_fn,
        num_agents=cfg.env.num_agents,
        device=device,
        batch_size=(1,),
        delta=cfg.env.delta,
        topology_config=topology_cfg,
    )
    env.set_seed(seed)
    env = TransformedEnv(
        env,
        RewardSum(in_keys=[env.reward_key], out_keys=[("agents", "episode_reward")]),
        device=device,
    )
    return env, landscape_fn


# =============================================================================
# Saving helpers
# =============================================================================


def _save_json(output_dir: str, results: list) -> str:
    path = os.path.join(output_dir, "multi_topology_metrics.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"Saved JSON : {path}")
    return path


def _save_csv(output_dir: str, results: list) -> str:
    """Flat CSV — one row per (topology, policy)."""
    path = os.path.join(output_dir, "multi_topology_metrics.csv")

    # Collect all field names from first result (dynamic because metrics vary)
    fieldnames_set: list = ["topology", "policy"]
    for res in results:
        for k in res.keys():
            if (
                k not in ("topology", "policy", "episode_histories")
                and k not in fieldnames_set
            ):
                fieldnames_set.append(k)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_set, extrasaction="ignore")
        writer.writeheader()
        for res in results:
            writer.writerow({k: res.get(k, "") for k in fieldnames_set})
    print(f"Saved CSV  : {path}")
    return path


def _save_episode_csv(output_dir: str, results: list) -> str:
    """Per-episode CSV — one row per (topology, policy, episode)."""
    path = os.path.join(output_dir, "multi_topology_episodes.csv")
    fieldnames = [
        "topology",
        "policy",
        "episode",
        # convergence
        "auc",
        "final_score",
        "improvement_rate",
        "plateau_fraction",
        "time_to_50pct",
        "time_to_80pct",
        "time_to_90pct",
        "time_to_99pct",
        # diversity (episode means)
        "mean_pairwise_dist",
        "position_spread",
        "velocity_alignment",
        "velocity_speed_std",
        # info-spread
        "mean_adoption_fraction",
        "mean_information_entropy",
        "num_spread_events",
        "mean_steps_to_50pct",
        "mean_steps_to_90pct",
        "mean_adoption_rate",
        "final_adoption_fraction",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for res in results:
            topo = res["topology"]
            pol = res["policy"]
            for h, conv in zip(
                res.get("episode_histories", []),
                res.get("convergence_per_episode", []),
            ):
                row: dict = {
                    "topology": topo,
                    "policy": pol,
                    "episode": h.get("episode", ""),
                }
                # convergence
                row.update(conv)
                # diversity
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
                # info-spread
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
    print(f"Saved episode CSV: {path}")
    return path


# =============================================================================
# Comparison plot
# =============================================================================


# =============================================================================
# Console summary table
# =============================================================================


def _print_summary_table(results: list) -> None:
    col_w = 18
    metric_keys = [
        ("mean_final_score", "FinalScore"),
        ("convergence_auc_mean", "AUC"),
        ("convergence_plateau_fraction_mean", "Plateau"),
        ("diversity_mean_pairwise_dist", "Diversity"),
        ("info_mean_adoption_fraction", "Adoption"),
        ("info_num_spread_events", "SpreadEvts"),
    ]
    header = f"{'Topology':<20} {'Policy':<20}" + "".join(
        f"{lbl:>{col_w}}" for _, lbl in metric_keys
    )
    print("\n" + "=" * len(header))
    print("MULTI-TOPOLOGY EVALUATION SUMMARY")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for res in results:
        row = f"{res['topology']:<20} {res['policy']:<20}"
        for key, _ in metric_keys:
            val = res.get(key, float("nan"))
            if isinstance(val, float):
                row += f"{val:>{col_w}.4f}"
            else:
                row += f"{str(val):>{col_w}}"
        print(row)
    print("=" * len(header))


# =============================================================================
# Main entry point
# =============================================================================


@hydra.main(version_base=None, config_path="configs", config_name="eval_multi_topology")
def main(cfg: DictConfig) -> None:
    print("=" * 60)
    print("PSO Multi-Topology Evaluation")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))

    seed: int = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}  |  Seed : {seed}")
    print(f"Function : {cfg.env.landscape_function} ({cfg.env.landscape_dim}D)")
    print("=" * 60)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(get_original_cwd(), cfg.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # Create timestamped model path to ensure uniqueness
    model_filename = f"best_model_{timestamp}.pt"
    model_path = os.path.join(output_dir, model_filename)

    num_episodes: int = cfg.eval.num_eval_episodes
    max_steps: int = cfg.eval.max_steps
    collect_diversity: bool = cfg.eval.get("collect_diversity", True)
    collect_info: bool = cfg.eval.get("collect_info_spread", True)
    compare_random: bool = cfg.eval.get("compare_random", True)

    # Copy original model to timestamped location for this evaluation run
    original_model_path = os.path.join(get_original_cwd(), cfg.model_path)
    if os.path.exists(original_model_path):
        shutil.copy(original_model_path, model_path)
        print(f"Copied model to timestamped location: {model_path}")
    else:
        print(f"WARNING: original model not found at {original_model_path}")

    topologies = OmegaConf.to_container(cfg.topologies, resolve=True)

    all_results: list = []

    for topo_def in topologies:
        topo_name: str = topo_def.get("name", topo_def.get("type", "unknown"))
        print(f"\n{'─'*60}")
        print(f"  Topology: {topo_name}")
        print(f"{'─'*60}")

        # Remove the "name" key before passing to PSOEnv (it only knows "type" etc.)
        topology_cfg = {k: v for k, v in topo_def.items() if k != "name"}

        env, landscape_fn = _make_env(cfg, topology_cfg, seed, device)

        # Build trained policy
        policy = create_policy(
            env,
            cfg.env.num_agents,
            cfg.env.landscape_dim,
            list(cfg.model.hidden_sizes),
            cfg.model.share_params,
            cfg.model.dropout,
            device,
        )
        if os.path.exists(model_path):
            ckpt = torch.load(model_path, map_location=device)
            policy.load_state_dict(ckpt["policy_state_dict"])
            print(f"  Loaded model from {model_path}")
        else:
            print(f"  WARNING: model not found at {model_path} — using random init")
        policy.eval()

        # ---- Evaluate trained policy ----
        trained_metrics, trained_curves, trained_histories = evaluate_policy(
            env,
            policy,
            num_episodes=num_episodes,
            max_steps=max_steps,
            policy_name=f"{topo_name}/Trained",
            collect_diversity=collect_diversity,
            collect_info_spread=collect_info,
        )
        conv_agg = aggregate_convergence_metrics(trained_curves)
        conv_per_ep = [compute_episode_convergence(c) for c in trained_curves]

        trained_row = {
            "topology": topo_name,
            "policy": "Trained",
            **trained_metrics,
            **{
                f"convergence_{k}": v
                for k, v in conv_agg.items()
                if not isinstance(v, list)
            },
            "convergence_mean_curve": conv_agg.get("mean_curve", []),
            "convergence_std_curve": conv_agg.get("std_curve", []),
            "episode_histories": trained_histories,
            "convergence_per_episode": conv_per_ep,
        }
        all_results.append(trained_row)

        # ---- Evaluate random baseline ----
        if compare_random:
            rand_policy = create_random_policy(cfg.env.landscape_dim, device)
            rand_metrics, rand_curves, rand_histories = evaluate_policy(
                env,
                rand_policy,
                num_episodes=num_episodes,
                max_steps=max_steps,
                policy_name=f"{topo_name}/Random",
                collect_diversity=collect_diversity,
                collect_info_spread=collect_info,
            )
            rand_conv_agg = aggregate_convergence_metrics(rand_curves)
            rand_conv_per_ep = [compute_episode_convergence(c) for c in rand_curves]

            rand_row = {
                "topology": topo_name,
                "policy": "Random",
                **rand_metrics,
                **{
                    f"convergence_{k}": v
                    for k, v in rand_conv_agg.items()
                    if not isinstance(v, list)
                },
                "convergence_mean_curve": rand_conv_agg.get("mean_curve", []),
                "convergence_std_curve": rand_conv_agg.get("std_curve", []),
                "episode_histories": rand_histories,
                "convergence_per_episode": rand_conv_per_ep,
            }
            all_results.append(rand_row)

        env.close()

    # ----------------------------------------------------------------
    # Save results
    # ----------------------------------------------------------------
    _print_summary_table(all_results)

    if cfg.eval.get("save_metrics", True):
        # Strip large list fields for JSON (keep curves)
        json_payload = []
        for res in all_results:
            entry = {k: v for k, v in res.items() if k != "episode_histories"}
            entry["episode_histories"] = [
                {ek: ev for ek, ev in h.items() if ek != "diversity" or True}
                for h in res.get("episode_histories", [])
            ]
            json_payload.append(entry)
        _save_json(output_dir, json_payload)
        _save_csv(output_dir, all_results)
        _save_episode_csv(output_dir, all_results)

    # Rich comparison plots (convergence, diversity, fitness dist, heatmap, radar)
    plotter = ComparisonPlotter(all_results, output_dir)
    plotter.plot_all()
    print("\nMulti-topology evaluation complete!")


if __name__ == "__main__":
    main()

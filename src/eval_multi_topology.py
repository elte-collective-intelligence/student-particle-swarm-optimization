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
    python src/eval_multi_topology.py topologies='[{name:global,type:global},{name:ring,type:ring,k:1}]'

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


def _make_env(
    cfg: DictConfig, topology_cfg: dict, seed: int, device, landscape_name=None
):
    """Build a fresh PSOEnv + TransformedEnv for one topology."""
    if landscape_name is None:
        landscape_name = cfg.env.landscape_function
    landscape_fn = get_landscape_function(landscape_name, cfg.env.landscape_dim)
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


def _prepare_model_path(
    original_model_path: str, output_dir: str, timestamp: str
) -> tuple[str, bool]:
    """Copy the checkpoint into the run directory when available.

    Returns the path that should be loaded and whether a copy was made.
    """
    if os.path.exists(original_model_path):
        model_path = os.path.join(output_dir, f"best_model_{timestamp}.pt")
        shutil.copy(original_model_path, model_path)
        return model_path, True
    return original_model_path, False


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
        "landscape",
        "seed",
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
                    "landscape": res.get("landscape", ""),
                    "seed": res.get("seed", ""),
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
    include_context = any("landscape" in res or "seed" in res for res in results)
    prefix = ""
    if include_context:
        prefix = f"{'Landscape':<18} {'Seed':<8}"
    header = (
        prefix
        + f"{'Topology':<20} {'Policy':<20}"
        + "".join(f"{lbl:>{col_w}}" for _, lbl in metric_keys)
    )
    print("\n" + "=" * len(header))
    print("MULTI-TOPOLOGY EVALUATION SUMMARY")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for res in results:
        row = ""
        if include_context:
            row += f"{res.get('landscape', ''):<18} {str(res.get('seed', '')):<8}"
        row += f"{res['topology']:<20} {res['policy']:<20}"
        for key, _ in metric_keys:
            val = res.get(key, float("nan"))
            if isinstance(val, float):
                row += f"{val:>{col_w}.4f}"
            else:
                row += f"{str(val):>{col_w}}"
        print(row)
    print("=" * len(header))


def _mean_curve(curves: list[list[float]]) -> tuple[list[float], list[float]]:
    valid = [curve for curve in curves if curve]
    if not valid:
        return [], []
    max_len = max(len(curve) for curve in valid)
    padded = np.array(
        [curve + [curve[-1]] * (max_len - len(curve)) for curve in valid],
        dtype=np.float64,
    )
    return padded.mean(axis=0).tolist(), padded.std(axis=0).tolist()


def _aggregate_for_plots(results: list, landscape: str) -> list:
    """Aggregate repeated seed rows into one row per topology and policy."""
    aggregated = []
    landscape_rows = [res for res in results if res.get("landscape") == landscape]
    keys = sorted({(res["topology"], res["policy"]) for res in landscape_rows})

    scalar_keys = [
        "mean_final_score",
        "std_final_score",
        "mean_best_score",
        "std_best_score",
        "mean_cumulative_reward",
        "std_cumulative_reward",
        "diversity_mean_pairwise_dist",
        "diversity_position_spread",
        "diversity_velocity_alignment",
        "diversity_velocity_speed_std",
        "info_mean_adoption_fraction",
        "info_mean_information_entropy",
        "info_mean_adoption_rate",
        "info_num_spread_events",
        "info_mean_steps_to_90pct",
        "convergence_auc_mean",
        "convergence_improvement_rate_mean",
        "convergence_plateau_fraction_mean",
        "convergence_time_to_90pct_mean",
    ]

    for topology, policy in keys:
        group = [
            res
            for res in landscape_rows
            if res["topology"] == topology and res["policy"] == policy
        ]
        entry = {
            "landscape": landscape,
            "topology": topology,
            "policy": policy,
            "seeds": [res.get("seed") for res in group],
            "episode_histories": [
                history for res in group for history in res.get("episode_histories", [])
            ],
            "convergence_per_episode": [
                conv for res in group for conv in res.get("convergence_per_episode", [])
            ],
        }
        for key in scalar_keys:
            values = [
                res.get(key)
                for res in group
                if isinstance(res.get(key), (int, float))
                and not np.isnan(float(res.get(key)))
            ]
            if values:
                entry[key] = float(np.mean(values))

        mean_curve, std_curve = _mean_curve(
            [res.get("convergence_mean_curve", []) for res in group]
        )
        entry["convergence_mean_curve"] = mean_curve
        entry["convergence_std_curve"] = std_curve
        aggregated.append(entry)

    return aggregated


def _save_run_summary(
    output_dir: str,
    landscapes: list,
    seeds: list,
    topologies: list,
    cfg: DictConfig,
    results: list,
) -> str:
    path = os.path.join(output_dir, "README.md")
    lines = [
        "# Multi-Topology Evaluation Summary",
        "",
        f"Output directory: `{output_dir}`",
        f"Landscapes: `{landscapes}`",
        f"Seeds: `{seeds}`",
        f"Topologies: `{[topo.get('name', topo.get('type')) for topo in topologies]}`",
        f"Episodes per condition: `{cfg.eval.num_eval_episodes}`",
        f"Steps per episode: `{cfg.eval.max_steps}`",
        "Policies: `Trained`"
        + (" and `Random`" if cfg.eval.get("compare_random", True) else ""),
        "",
        "Generated files:",
        "- `multi_topology_metrics.json`",
        "- `multi_topology_metrics.csv`",
        "- `multi_topology_episodes.csv`",
        "- plot PNGs in this directory, or under `plots/<landscape>/` for sweeps",
        "",
        f"Metric rows: `{len(results)}`",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved summary: {path}")
    return path


# =============================================================================
# Main entry point
# =============================================================================


@hydra.main(version_base=None, config_path="configs", config_name="eval_multi_topology")
def main(cfg: DictConfig) -> None:
    print("=" * 60)
    print("PSO Multi-Topology Evaluation")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    default_seed = int(cfg.get("seed", 42))
    landscapes_cfg = cfg.get("landscapes", None)
    if landscapes_cfg is None:
        landscapes = [cfg.env.landscape_function]
    else:
        landscapes = list(landscapes_cfg)
    seeds = [int(seed) for seed in cfg.get("seeds", [default_seed])]
    topologies = OmegaConf.to_container(cfg.topologies, resolve=True)

    print(f"Device : {device}")
    print(f"Functions : {landscapes} ({cfg.env.landscape_dim}D)")
    print(f"Seeds : {seeds}")
    print(
        "Conditions : "
        f"{len(landscapes)} landscapes x {len(seeds)} seeds x {len(topologies)} topologies"
    )
    print("=" * 60)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(get_original_cwd(), cfg.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    num_episodes: int = cfg.eval.num_eval_episodes
    max_steps: int = cfg.eval.max_steps
    collect_diversity: bool = cfg.eval.get("collect_diversity", True)
    collect_info: bool = cfg.eval.get("collect_info_spread", True)
    compare_random: bool = cfg.eval.get("compare_random", True)

    original_model_path = os.path.join(get_original_cwd(), cfg.model_path)
    model_path, copied_model = _prepare_model_path(
        original_model_path, output_dir, timestamp
    )
    if copied_model:
        print(f"Copied model to timestamped location: {model_path}")
    else:
        print(f"WARNING: original model not found at {original_model_path}")

    all_results: list = []

    for landscape_name in landscapes:
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            for topo_def in topologies:
                topo_name: str = topo_def.get("name", topo_def.get("type", "unknown"))
                print(f"\n{'-' * 60}")
                print(
                    f"  Function: {landscape_name} | Seed: {seed} | Topology: {topo_name}"
                )
                print(f"{'-' * 60}")

                topology_cfg = {k: v for k, v in topo_def.items() if k != "name"}

                env, _ = _make_env(
                    cfg, topology_cfg, seed, device, landscape_name=landscape_name
                )

                # build trained policy
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
                    print(
                        f"  WARNING: model not found at {model_path} - using random init"
                    )
                policy.eval()

                # trained policy
                trained_metrics, trained_curves, trained_histories = evaluate_policy(
                    env,
                    policy,
                    num_episodes=num_episodes,
                    max_steps=max_steps,
                    policy_name=f"{landscape_name}/{seed}/{topo_name}/Trained",
                    collect_diversity=collect_diversity,
                    collect_info_spread=collect_info,
                )
                conv_agg = aggregate_convergence_metrics(trained_curves)
                conv_per_ep = [compute_episode_convergence(c) for c in trained_curves]

                trained_row = {
                    "landscape": landscape_name,
                    "seed": seed,
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

                # random baseline
                if compare_random:
                    rand_policy = create_random_policy(cfg.env.landscape_dim, device)
                    rand_metrics, rand_curves, rand_histories = evaluate_policy(
                        env,
                        rand_policy,
                        num_episodes=num_episodes,
                        max_steps=max_steps,
                        policy_name=f"{landscape_name}/{seed}/{topo_name}/Random",
                        collect_diversity=collect_diversity,
                        collect_info_spread=collect_info,
                    )
                    rand_conv_agg = aggregate_convergence_metrics(rand_curves)
                    rand_conv_per_ep = [
                        compute_episode_convergence(c) for c in rand_curves
                    ]

                    rand_row = {
                        "landscape": landscape_name,
                        "seed": seed,
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

    if len(landscapes) > 1 or len(seeds) > 1:
        plots_dir = os.path.join(output_dir, "plots")
        for landscape_name in landscapes:
            landscape_plot_dir = os.path.join(plots_dir, landscape_name)
            aggregated = _aggregate_for_plots(all_results, landscape_name)
            plotter = ComparisonPlotter(aggregated, landscape_plot_dir)
            plotter.plot_all()
            _save_json(landscape_plot_dir, aggregated)
    else:
        plotter = ComparisonPlotter(all_results, output_dir)
        plotter.plot_all()

    _save_run_summary(output_dir, landscapes, seeds, topologies, cfg, all_results)
    print("\nMulti-topology evaluation complete!")


if __name__ == "__main__":
    main()

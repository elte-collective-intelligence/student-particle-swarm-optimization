"""
Generate side-by-side swarm GIFs for selected landscape functions and topologies.

Each generated GIF contains one panel per topology, synchronized by timestep,
so topology-specific swarm dynamics are easy to compare visually.

Usage:
    python src/generate_side_by_side_gifs.py
    python src/generate_side_by_side_gifs.py functions=[sphere,rastrigin]
    python src/generate_side_by_side_gifs.py selected_topologies=[global,ring]
"""

from __future__ import annotations

import os
import random
from datetime import datetime
from typing import Dict, List, Tuple

import hydra
import imageio
import matplotlib
import numpy as np
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from torchrl.envs import RewardSum, TransformedEnv

from envs import PSOEnv
from eval_helpers import create_policy, get_landscape_function

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _sanitize_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_")


def _bounds_for_landscape(landscape_name: str) -> Tuple[float, float]:
    name = _sanitize_name(landscape_name)
    if "eggholder" in name:
        return (-512.0, 512.0)
    return (-5.12, 5.12)


def _is_static_landscape(landscape_name: str) -> bool:
    return not _sanitize_name(landscape_name).startswith("dynamic_")


def _make_env(
    cfg: DictConfig,
    landscape_name: str,
    topology_cfg: Dict,
    seed: int,
    device: torch.device,
):
    landscape_fn = get_landscape_function(landscape_name, int(cfg.env.landscape_dim))
    env = PSOEnv(
        landscape=landscape_fn,
        num_agents=int(cfg.env.num_agents),
        device=device,
        batch_size=(1,),
        delta=float(cfg.env.delta),
        topology_config=topology_cfg,
    )
    env.set_seed(seed)
    env = TransformedEnv(
        env,
        RewardSum(in_keys=[env.reward_key], out_keys=[("agents", "episode_reward")]),
        device=device,
    )
    return env


def _load_policy(
    env,
    cfg: DictConfig,
    device: torch.device,
    model_path: str,
):
    policy = create_policy(
        env=env,
        num_agents=int(cfg.env.num_agents),
        dim=int(cfg.env.landscape_dim),
        hidden_sizes=list(cfg.model.hidden_sizes),
        share_params=bool(cfg.model.share_params),
        dropout=float(cfg.model.dropout),
        device=device,
    )

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        policy.load_state_dict(checkpoint["policy_state_dict"])
    else:
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}. "
            "Train first or override model_path in config."
        )

    policy.eval()
    return policy


def _collect_rollout_frames(env, policy, max_steps: int) -> List[dict]:
    base_env = env.base_env if hasattr(env, "base_env") else env
    data = env.reset()
    frames: List[dict] = []

    for step in range(max_steps):
        with torch.no_grad():
            data = policy(data)
        data = env.step(data)

        positions = base_env.positions[0, :, :2].detach().cpu().numpy()
        personal_bests = base_env.personal_best_pos[0, :, :2].detach().cpu().numpy()

        pb_scores = base_env.personal_best_scores[0]
        best_idx = int(pb_scores.argmax().item())
        global_best = personal_bests[best_idx]

        frames.append(
            {
                "step": step,
                "positions": positions,
                "personal_bests": personal_bests,
                "global_best": global_best,
                "best_score": float(pb_scores.max().item()),
            }
        )

        data = data["next"]

    return frames


def _build_landscape_grid(landscape_name: str, resolution: int = 60):
    if not _is_static_landscape(landscape_name):
        return None

    bounds = _bounds_for_landscape(landscape_name)
    landscape_fn = get_landscape_function(landscape_name, dim=2)

    x = np.linspace(bounds[0], bounds[1], resolution)
    y = np.linspace(bounds[0], bounds[1], resolution)
    xx, yy = np.meshgrid(x, y)

    points = torch.tensor(
        np.stack([xx.ravel(), yy.ravel()], axis=-1), dtype=torch.float32
    )
    zz = landscape_fn(points).detach().cpu().numpy().reshape(xx.shape)
    return xx, yy, zz


def _render_side_by_side_frames(
    rollout_by_topology: Dict[str, List[dict]],
    landscape_name: str,
    fps: int,
    show_landscape: bool,
) -> List[np.ndarray]:
    topology_names = list(rollout_by_topology.keys())
    max_frames = max(len(v) for v in rollout_by_topology.values())
    bounds = _bounds_for_landscape(landscape_name)

    landscape_grid = None
    if show_landscape and _is_static_landscape(landscape_name):
        landscape_grid = _build_landscape_grid(landscape_name, resolution=70)

    gif_frames: List[np.ndarray] = []

    for frame_idx in range(max_frames):
        fig, axes = plt.subplots(
            1,
            len(topology_names),
            figsize=(5 * len(topology_names), 5),
            squeeze=False,
        )
        axes = axes[0]

        for col, topology_name in enumerate(topology_names):
            ax = axes[col]
            rollout = rollout_by_topology[topology_name]
            state = rollout[min(frame_idx, len(rollout) - 1)]

            if landscape_grid is not None:
                xx, yy, zz = landscape_grid
                ax.contourf(xx, yy, zz, levels=25, cmap="viridis", alpha=0.65)

            pb = state["personal_bests"]
            pos = state["positions"]
            gb = state["global_best"]

            ax.scatter(pb[:, 0], pb[:, 1], c="#4CAF50", s=20, marker="^", alpha=0.4)
            ax.scatter(pos[:, 0], pos[:, 1], c="#111111", s=36, marker="o")
            ax.scatter([gb[0]], [gb[1]], c="#D81B60", s=170, marker="*")

            ax.set_xlim(bounds)
            ax.set_ylim(bounds)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(f"{topology_name} | best={state['best_score']:.3f}")
            ax.set_xlabel("x1")
            ax.set_ylabel("x2")

        fig.suptitle(
            f"Landscape: {landscape_name} | Step {frame_idx + 1}/{max_frames} | FPS {fps}",
            fontsize=12,
        )
        fig.tight_layout()

        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
        gif_frames.append(image)
        plt.close(fig)

    return gif_frames


def _topology_config_map(cfg: DictConfig) -> Dict[str, dict]:
    catalog = OmegaConf.to_container(cfg.topology_catalog, resolve=True)
    if not isinstance(catalog, dict):
        raise ValueError("topology_catalog must be a dictionary")
    return catalog


@hydra.main(version_base=None, config_path="configs", config_name="side_by_side_gif")
def main(cfg: DictConfig) -> None:
    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    original_cwd = get_original_cwd()
    model_path = os.path.join(original_cwd, cfg.model_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = os.path.join(original_cwd, cfg.output_dir, timestamp)
    os.makedirs(output_root, exist_ok=True)

    topology_catalog = _topology_config_map(cfg)
    selected_topologies = list(cfg.selected_topologies)

    for topology_name in selected_topologies:
        if topology_name not in topology_catalog:
            raise ValueError(
                f"Topology '{topology_name}' not found in topology_catalog. "
                f"Available: {list(topology_catalog.keys())}"
            )

    print("=" * 70)
    print("Generating side-by-side topology GIFs")
    print("=" * 70)
    print(f"Device        : {device}")
    print(f"Seed          : {seed}")
    print(f"Model         : {model_path}")
    print(f"Output folder : {output_root}")
    print(f"Functions     : {list(cfg.functions)}")
    print(f"Topologies    : {selected_topologies}")
    print("=" * 70)

    for landscape_name in cfg.functions:
        print(f"\n[Function] {landscape_name}")
        rollout_by_topology: Dict[str, List[dict]] = {}

        for topology_name in selected_topologies:
            topo_cfg = dict(topology_catalog[topology_name])
            topo_cfg["type"] = topology_name

            env = _make_env(
                cfg=cfg,
                landscape_name=landscape_name,
                topology_cfg=topo_cfg,
                seed=seed,
                device=device,
            )
            policy = _load_policy(env, cfg, device, model_path)
            rollout = _collect_rollout_frames(
                env=env,
                policy=policy,
                max_steps=int(cfg.gif.max_steps),
            )
            rollout_by_topology[topology_name] = rollout
            print(
                f"  Collected rollout for topology={topology_name} with {len(rollout)} steps"
            )

        gif_frames = _render_side_by_side_frames(
            rollout_by_topology=rollout_by_topology,
            landscape_name=landscape_name,
            fps=int(cfg.gif.fps),
            show_landscape=bool(cfg.gif.show_landscape),
        )

        out_name = f"side_by_side_{_sanitize_name(landscape_name)}.gif"
        out_path = os.path.join(output_root, out_name)
        imageio.mimsave(out_path, gif_frames, fps=int(cfg.gif.fps), loop=0)
        print(f"  Saved GIF: {out_path}")

    print("\nAll side-by-side GIFs generated successfully.")


if __name__ == "__main__":
    main()

import argparse
import os
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent))

from models import DimAgnosticNet, build_curriculum_policy
from training.curriculum import make_env
from visualization import SwarmVisualizer

CURRICULUM_CONFIG_DIR = Path(__file__).parent / "configs" / "curriculum"


def load_curriculum_cfg(config_name: str):
    """Load a curriculum YAML and return the OmegaConf config."""
    path = CURRICULUM_CONFIG_DIR / f"{config_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"Curriculum config not found: {path}\n"
            f"Available: {[p.stem for p in CURRICULUM_CONFIG_DIR.glob('*.yaml')]}"
        )
    return OmegaConf.load(path)


def get_landscape_fn_for_vis(landscape_name: str, dim: int):
    """Return a callable landscape for SwarmVisualizer background contours."""
    from utils import LandscapeWrapper
    from envs.dynamic_functions import DynamicSphere, DynamicRastrigin
    from main import sphere, rastrigin, eggholder, rosenbrock

    static = {
        "sphere": sphere,
        "rastrigin": rastrigin,
        "eggholder": eggholder,
        "rosenbrock": rosenbrock,
    }
    if landscape_name in static:
        return LandscapeWrapper(static[landscape_name], dim=dim)
    if landscape_name == "dynamic_sphere":
        # A separate instance for the visualizer background: its time counter
        # is independent of the env's instance, so the contour shows t=0.
        # This is a known visual approximation — the policy still optimizes
        # the env's actual (shifted) landscape correctly.
        return DynamicSphere(dim=dim)
    if landscape_name == "dynamic_rastrigin":
        return DynamicRastrigin(dim=dim)
    raise ValueError(f"Unknown landscape: {landscape_name}")


def run_episode(env, policy, max_steps: int, visualizer: SwarmVisualizer):
    """Run one episode, recording frames, and return convergence curves."""
    base_env = env.base_env

    data = env.reset()
    best_scores_curve = []
    mean_scores_curve = []

    with torch.no_grad():
        for step in range(max_steps):
            data = policy(data)
            data = env.step(data)

            scores = base_env.scores[0].cpu()
            best_scores_curve.append(scores.max().item())
            mean_scores_curve.append(scores.mean().item())

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

    return best_scores_curve, mean_scores_curve


def main():
    parser = argparse.ArgumentParser(description="Generate GIFs from curriculum-trained models")
    parser.add_argument("--model-path", required=True, help="Path to seed_N/model.pt checkpoint")
    parser.add_argument("--config-name", required=True,
                        help="Curriculum config name (function, dynamics, dimension, combined). "
                             "Reads hidden_size, num_agents, delta from the corresponding YAML.")
    parser.add_argument("--landscape", required=True,
                        help="Landscape to evaluate on: sphere, rastrigin, eggholder, rosenbrock, "
                             "dynamic_sphere, dynamic_rastrigin")
    parser.add_argument("--dim", type=int, default=2, help="Search space dimensionality")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=100, help="Steps per episode")
    parser.add_argument("--output-dir", default="images/semester_contribution")
    parser.add_argument("--gif-name", default=None, help="Base filename for output GIFs (no extension)")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load checkpoint first, then resolve architecture params ---
    # Prefer model_cfg saved inside the checkpoint (written by curriculum_train.py).
    # Fall back to the YAML for checkpoints trained before model_cfg was added.
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)

    if "model_cfg" in checkpoint:
        model_cfg   = checkpoint["model_cfg"]
        hidden_size = model_cfg["hidden_size"]
        num_agents  = model_cfg["num_agents"]
        delta       = model_cfg["delta"]
        cfg_source  = "checkpoint"
    else:
        yaml_cfg    = load_curriculum_cfg(args.config_name)
        hidden_size = yaml_cfg.model.hidden_size
        num_agents  = yaml_cfg.env.num_agents
        delta       = yaml_cfg.env.delta
        cfg_source  = f"{args.config_name}.yaml (fallback — checkpoint predates model_cfg)"

    gif_name = args.gif_name or f"curriculum_{args.config_name}_{args.landscape}_{args.dim}d"

    print(f"Device:      {device}")
    print(f"Config src:  {cfg_source}")
    print(f"             hidden_size={hidden_size}, num_agents={num_agents}, delta={delta}")
    print(f"Model:       {args.model_path}")
    print(f"Landscape:   {args.landscape} {args.dim}D")
    print(f"Output:      {args.output_dir}/{gif_name}_*.gif")

    net = DimAgnosticNet(hidden_size=hidden_size).to(device)
    net.load_state_dict(checkpoint["net_state"])
    net.eval()

    curriculum_log = checkpoint.get("curriculum_log", [])
    if curriculum_log:
        stages_str = " → ".join(e["stage"] for e in curriculum_log)
        print(f"Curriculum:  {stages_str}")

    # --- Build environment and policy ---
    env = make_env(
        dim=args.dim,
        landscape_name=args.landscape,
        num_agents=num_agents,
        batch_size=args.batch_size,
        delta=delta,
        device=device,
    )
    policy = build_curriculum_policy(net, env, device)
    policy.eval()

    # --- Landscape function for visualizer background ---
    landscape_fn = get_landscape_fn_for_vis(args.landscape, args.dim)

    vis_config = {
        "visualize_swarm": True,
        "visualize_landscape": True,
        "save_gif": True,
        "save_dir": args.output_dir,
        "fps": args.fps,
        "dpi": 150,
    }
    visualizer = SwarmVisualizer(vis_config=vis_config, landscape_fn=landscape_fn, dim=args.dim)
    visualizer.reset(episode=0)

    # --- Run episode ---
    print(f"Running {args.max_steps} steps...")
    best_curve, mean_curve = run_episode(env, policy, args.max_steps, visualizer)
    print(f"Final best score: {best_curve[-1]:.4f}  (init: {best_curve[0]:.4f})")

    # --- Save GIFs ---
    gif_2d = visualizer.create_2d_animation(filename=f"{gif_name}_2d")
    if gif_2d:
        print(f"Saved 2D GIF: {gif_2d}")

    if args.dim == 2:
        gif_3d = visualizer.create_3d_animation(filename=f"{gif_name}_3d")
        if gif_3d:
            print(f"Saved 3D GIF: {gif_3d}")

    visualizer.create_convergence_plot(best_curve, mean_curve, filename=f"{gif_name}_convergence")
    print("Done.")


if __name__ == "__main__":
    main()

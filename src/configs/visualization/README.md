# Visualization Configuration

This directory contains visualization configuration files for the PSO environment.

## Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `visualize_swarm` | Generate animated swarm visualizations | false |
| `visualize_landscape` | Show landscape contours in plots | true |
| `save_gif` | Save animations as GIF files | false |
| `save_dir` | Output directory for visualizations | outputs/vis/ |
| `fps` | Frames per second for animations | 10 |
| `dpi` | Resolution for saved images | 150 |

## Available Configurations

### Default (`default.yaml`)
Basic plots only, no animations.

### Full (`full.yaml`)
Full visualization with animated 2D and 3D swarm visualizations.

### None (`none.yaml`)
No visualizations (fastest evaluation).

## Generated Visualizations

When `visualize_swarm=true`, the following files are generated:

- `swarm_2d_ep{N}.gif` - 2D animated view of particles moving on landscape contours
- `swarm_3d_ep{N}.gif` - 3D animated view with particles on landscape surface
- `trajectories_ep{N}.png` - Static plot showing particle trajectories
- `convergence_ep{N}.png` - Convergence curve showing score improvement

## Usage

```bash
# Enable full visualization
python src/eval.py model_path=outputs/best_model.pt visualization.visualize_swarm=true visualization.save_gif=true

# Or use the full config
python src/eval.py --config-path ../conf/experiments --config-name eval_vis
```

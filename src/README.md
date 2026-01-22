# Source Code Directory

This directory contains the main implementation of the multi-agent reinforcement learning system for Particle Swarm Optimization (PSO).

## Files

### `main.py`
**Entry point for training.** This file:
- Uses Hydra for configuration management
- Implements custom PPO training loop with TorchRL
- Handles multi-agent environments with shared policy
- Saves models and logs training metrics

**Usage:**
```bash
# Run default config
python src/main.py

# Run experiment config
python src/main.py --config-path configs/experiments --config-name smoke_train
```

**Key Components:**
- `compute_gae()`: Custom GAE implementation for advantage estimation
- Training loop with proper action transformation for PSO parameters
- Model checkpointing and metric logging

### `eval.py`
**Evaluation and visualization script.** This file:
- Evaluates trained policies against random baselines
- Generates visualizations (2D/3D animations, convergence plots)
- Compares different policies side-by-side

**Usage:**
```bash
# Evaluate with visualization
python src/eval.py --config-path configs/experiments --config-name eval_vis model_path=path/to/model.pt
```

**Key Functions:**
- `evaluate_policy()`: Run policy on multiple episodes and collect metrics
- `compare_policies()`: Compare trained policy against random baseline

### `visualization.py`
**Swarm visualization module.** This file:
- Creates 2D and 3D animated GIFs of particle swarm movement
- Generates trajectory plots showing particle paths
- Creates convergence plots showing optimization progress

**Key Class:**
- `SwarmVisualizer`: Main visualization class
  - `create_2d_animation()`: Animated 2D swarm movement
  - `create_3d_animation()`: Animated 3D swarm movement (3D+ functions only)
  - `create_trajectory_plot()`: Static particle trajectory visualization
  - `create_convergence_plot()`: Best fitness over time

### `utils.py`
**Utility functions and wrappers.** This file:
- `PSOActionExtractor`: Wrapper that extracts and transforms PSO parameters from neural network outputs
- Action transformation for proper PSO parameter scaling:
  - Inertia: Network output → [0.3, 1.1]
  - Cognitive/Social coefficients: Network output → [0.5, 2.5]

**Why Action Transformation?**
Neural networks output values near 0, but PSO requires specific coefficient ranges:
- Inertia ≈ 0.7 for good exploration/exploitation balance
- Cognitive/Social ≈ 1.5 for effective information sharing

## Subdirectories

### `configs/`
Contains all configuration files for experiments, models, and visualization.
See [configs/README.md](configs/README.md) for details.

### `envs/`
Contains the PSO environment implementation with multi-agent support.
See [envs/README.md](envs/README.md) for details.

### `outputs/`
Training outputs including:
- Model checkpoints
- Training logs
- Visualization files (GIFs, plots)
- Hydra configuration logs

## Quick Start

1. **Training a new model:**
   ```bash
   python src/main.py --config-path configs/experiments --config-name smoke_train
   ```

2. **Full training run:**
   ```bash
   python src/main.py --config-path configs/experiments --config-name full_train
   ```

3. **Evaluating a trained model:**
   ```bash
   python src/eval.py model_path=src/outputs/models/policy.pt
   ```

4. **Visualization only:**
   ```bash
   python src/eval.py --config-path configs/experiments --config-name eval_vis
   ```

## Architecture Overview

The codebase follows a modular design:
- **Environment**: Vectorized multi-agent PSO environment (TorchRL)
- **Policy**: PPO with custom action extraction for PSO parameters
- **Training**: Custom PPO loop with GAE advantage estimation
- **Evaluation**: Policy comparison with visualization support

## Key Concepts

### Multi-Agent PSO
Each particle in the swarm is an agent that:
- Observes its position, velocity, personal best, and global best
- Outputs PSO coefficients (inertia, cognitive, social)
- Receives reward based on optimization progress

### Action Space
Each agent outputs 4 continuous values:
- `inertia`: Weight for previous velocity [0.3, 1.1]
- `cognitive`: Weight for personal best attraction [0.5, 2.5]
- `social`: Weight for global best attraction [0.5, 2.5]
- `step_size`: Overall step magnitude [0.5, 2.5]

### Objective Functions
Training and evaluation on standard optimization benchmarks:
- `sphere`: Simple convex function (default)
- `rastrigin`: Multimodal function with many local minima
- `rosenbrock`: Valley-shaped function
- `ackley`: Complex multimodal function

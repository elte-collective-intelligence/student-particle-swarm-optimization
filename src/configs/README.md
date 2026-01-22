# Configuration Files

This directory contains all configuration files for the project, organized by type. Configurations use YAML format with Hydra for composition and overrides.

## Directory Structure

```
configs/
├── config.yaml           # Default training configuration
├── eval_config.yaml      # Default evaluation configuration
├── env/                  # Environment configurations
├── model/                # Model/agent configurations
├── experiments/          # Complete experiment configurations
├── visualization/        # Visualization settings
└── eval/                 # Evaluation settings
```

## Configuration Hierarchy

Hydra merges configurations in order:
1. **Base Config** (`config.yaml` or `eval_config.yaml`)
2. **Defaults** (from `defaults:` block)
3. **Experiment Overrides** (experiment-specific values)

**Priority:** Later configs override earlier values. Command-line overrides are highest priority.

## Environment Configurations (`env/`)

### `swarm.yaml`
**PSO environment configuration.** Specifies:
```yaml
n_particles: 10          # Number of particles in swarm
n_dims: 2                # Dimensionality of search space
max_steps: 100           # Maximum steps per episode
bounds: [-5.0, 5.0]      # Search space boundaries
objective_function: sphere  # Optimization function

# Reward configuration
reward_type: fitness     # 'fitness' or 'improvement'
normalize_rewards: true  # Normalize to [-1, 1]
```

**Key Parameters:**
- `n_particles`: Swarm size
  - More particles → better exploration, slower training
  - Typical: 10-50 particles
- `n_dims`: Problem dimensionality
  - 2D for visualization, higher for harder problems
  - 2-30 typical range
- `objective_function`: Optimization target
  - `sphere`: Simple convex (good for initial testing)
  - `rastrigin`: Multimodal (tests exploration)
  - `rosenbrock`: Valley-shaped (tests exploitation)
  - `ackley`: Complex multimodal

## Model Configurations (`model/`)

### `ppo.yaml`
**PPO agent configuration.** Specifies:
```yaml
# Architecture
hidden_sizes: [64, 64]   # MLP hidden layer sizes
activation: tanh         # Activation function

# PPO Parameters
clip_epsilon: 0.2        # PPO clipping parameter
lr: 0.0003               # Learning rate
gamma: 0.99              # Discount factor
gae_lambda: 0.95         # GAE parameter

# Training
n_envs: 8                # Parallel environments
frames_per_batch: 1000   # Frames collected per update
total_frames: 100000     # Total training frames
n_epochs: 4              # Update epochs per batch
batch_size: 256          # Minibatch size
```

**Key Parameters:**
- `clip_epsilon`: Controls policy update magnitude
  - Standard: 0.1-0.3
  - Lower → more conservative updates
- `gae_lambda`: Bias-variance tradeoff
  - 1.0 → high variance, no bias
  - 0.0 → low variance, high bias
  - 0.95 typical sweet spot
- `n_envs`: Parallelism
  - More envs → faster data collection
  - Limited by CPU/memory

## Visualization Configurations (`visualization/`)

### `default.yaml`
**Standard visualization settings:**
```yaml
enabled: true
save_gifs: true
save_plots: true
fps: 10
dpi: 100
```

### `full.yaml`
**Complete visualization with high quality:**
```yaml
enabled: true
save_gifs: true
save_plots: true
save_trajectory: true
save_convergence: true
fps: 15
dpi: 150
animation_interval: 50
```

### `none.yaml`
**Disable all visualization (faster training):**
```yaml
enabled: false
save_gifs: false
save_plots: false
```

## Evaluation Configurations (`eval/`)

### `default.yaml`
**Standard evaluation settings:**
```yaml
num_episodes: 10
compare_to_random: true
```

### `smoke.yaml`
**Quick evaluation for testing:**
```yaml
num_episodes: 3
compare_to_random: true
```

### `full.yaml`
**Comprehensive evaluation:**
```yaml
num_episodes: 100
compare_to_random: true
```

## Experiment Configurations (`experiments/`)

Complete experiment setups that combine base configs with specific overrides.

### `smoke_train.yaml`
**Quick training test (~1 min):**
- 10k frames, 5 particles, 2D
- Good for testing setup

### `full_train.yaml`
**Full training run (~30 min):**
- 500k frames, 20 particles, 5D
- Production training

### `dynamic_train.yaml`
**Dynamic function training:**
- Function changes during training
- Tests generalization

### `rastrigin_train.yaml`
**Multimodal function training:**
- Tests exploration capability
- Harder optimization target

### `eval_vis.yaml`
**Evaluation with visualization:**
- Runs trained model
- Generates GIFs and plots

See [experiments/README.md](experiments/README.md) for detailed experiment descriptions.

## Usage Examples

### Default Training
```bash
python src/main.py
```

### Experiment Config
```bash
python src/main.py --config-path configs/experiments --config-name smoke_train
```

### Override Parameters
```bash
# Change number of particles
python src/main.py env.n_particles=20

# Disable visualization
python src/main.py visualization=none

# Multiple overrides
python src/main.py env.n_dims=10 model.total_frames=500000
```

### List Available Configs
```bash
# Experiments
ls src/configs/experiments/

# Visualization options
ls src/configs/visualization/
```

## Creating New Experiments

1. Create new YAML file in `experiments/`:
```yaml
# experiments/my_experiment.yaml
defaults:
  - ../config
  - _self_

env:
  n_particles: 30
  objective_function: ackley

model:
  total_frames: 200000
```

2. Run the experiment:
```bash
python src/main.py --config-path configs/experiments --config-name my_experiment
```

## Configuration Tips

### Performance Tuning
- Increase `n_envs` for faster data collection
- Reduce `total_frames` for quick tests
- Use `visualization: none` during training

### Debugging
- Set `verbose: true` in logger config
- Reduce `frames_per_batch` for more frequent updates
- Use small `n_particles` and `n_dims`

### Reproducibility
- Set `seed` in experiment config
- Use fixed `n_envs` (affects batching)
- Log full config with Hydra

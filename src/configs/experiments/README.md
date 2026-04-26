# Experiment Configurations

This directory contains pre-defined experiment configurations for training and evaluation.

## Training Experiments

### Smoke Train (`smoke_train.yaml`)
Quick smoke test for development:
- 5 agents, 2D sphere function
- 10 iterations, 512 frames per batch
- ~30 seconds runtime

```bash
python src/main.py --config-path ../conf/experiments --config-name smoke_train
```

### Full Train (`full_train.yaml`)
Full training experiment:
- 10 agents, 2D sphere function
- 200 iterations, 4096 frames per batch
- Lower learning rate (1e-4)

```bash
python src/main.py --config-path ../conf/experiments --config-name full_train
```

### Topology Quick Train (`topology_quick_train.yaml`)
Quick topology-enabled training preset:
- 8 agents with ring topology (`k=1`)
- 10 iterations, 512 frames per batch
- Fast check that topology wiring works in training

```bash
python src/main.py --config-path configs/experiments --config-name topology_quick_train
```

### Topology Full Train (`topology_full_train.yaml`)
Full topology-enabled training preset:
- 12 agents with dynamic k-nearest topology (`k=2`, `recompute_interval=5`)
- 200 iterations, 4096 frames per batch

```bash
python src/main.py --config-path configs/experiments --config-name topology_full_train
```

### Dynamic Train (`dynamic_train.yaml`)
Training on dynamic (moving optimum) landscapes:
- 10 agents, 2D dynamic sphere
- 150 iterations

```bash
python src/main.py --config-path ../conf/experiments --config-name dynamic_train
```

### Rastrigin Train (`rastrigin_train.yaml`)
Training on multimodal Rastrigin function:
- 15 agents (more for exploration)
- 200 iterations

```bash
python src/main.py --config-path ../conf/experiments --config-name rastrigin_train
```

## Evaluation Experiments

### Eval Vis (`eval_vis.yaml`)
Evaluation with full 2D/3D visualizations:
- 8 agents, sphere function
- 5 episodes, 50 steps
- Generates animated GIFs

```bash
python src/eval.py --config-path ../conf/experiments --config-name eval_vis
```

### Topology Quick Eval (`topology_quick_eval.yaml`)
Quick topology-enabled evaluation preset:
- 8 agents with ring topology (`k=1`)
- 5 episodes, 50 steps

```bash
python src/eval.py --config-path configs/experiments --config-name topology_quick_eval
```

### Topology Full Eval (`topology_full_eval.yaml`)
Full topology-enabled evaluation preset:
- 12 agents with dynamic k-nearest topology (`k=2`, `recompute_interval=5`)
- 25 episodes, 100 steps

```bash
python src/eval.py --config-path configs/experiments --config-name topology_full_eval
```

## Creating Custom Experiments

Copy any experiment file and modify the parameters as needed. Key parameters:

- `env.landscape_function`: sphere, rastrigin, eggholder, dynamic_sphere, dynamic_rastrigin
- `env.num_agents`: Number of particles in the swarm
- `env.landscape_dim`: Dimensionality of the search space
- `n_iters`: Number of training iterations
- `model.learning_rate`: Learning rate for optimization

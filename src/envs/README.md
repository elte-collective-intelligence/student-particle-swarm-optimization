# Environment and Topology Module

This directory contains the TorchRL PSO environment and the communication-topology implementations used in the topology experiment.

## Files

- `env.py`: `PSOEnv`, the vectorized multi-agent PSO environment.
- `topology.py`: topology abstractions and factory functions.
- `dynamic_functions.py`: static and dynamic benchmark landscapes.
- `__init__.py`: module exports.

## Environment API

`PSOEnv` is constructed from an objective landscape wrapper and experiment settings:

```python
from envs import PSOEnv

env = PSOEnv(
    landscape=landscape_fn,
    num_agents=12,
    device=device,
    batch_size=(1,),
    delta=1.0,
    topology_config={"type": "ring", "k": 1, "include_self": True},
)
```

The environment maximizes negated objective scores, so higher scores are better. Particle positions and velocities are clamped internally to avoid numerical explosion.

## Observations

Each particle receives these tensors:

| Key | Shape | Meaning |
|---|---:|---|
| `scores` | `[batch, agents]` | Current negated objective score. |
| `positions` | `[batch, agents, dim]` | Current particle positions. |
| `velocities` | `[batch, agents, dim]` | Current particle velocities. |
| `personal_best_pos` | `[batch, agents, dim]` | Best position found by each particle. |
| `personal_best_scores` | `[batch, agents]` | Score at each personal best. |
| `neighborhood_best_pos` | `[batch, agents, dim]` | Best visible personal-best position under the active topology. |
| `neighborhood_best_scores` | `[batch, agents]` | Score for the visible neighborhood best. |
| `avg_pos` | `[batch, agents, dim]` | Distance-radius neighbor mean offset from the current position. |
| `avg_vel` | `[batch, agents, dim]` | Distance-radius neighbor mean velocity offset. |

`avg_pos` and `avg_vel` are still computed from the distance threshold `delta`. The topology experiment uses `neighborhood_best_pos` as the social information channel.

## Actions

The policy outputs three coefficient tensors:

| Key | Shape | Clamp range |
|---|---:|---:|
| `inertia` | `[batch, agents, dim]` | `[0.0, 1.2]` |
| `cognitive` | `[batch, agents, dim]` | `[0.0, 2.5]` |
| `social` | `[batch, agents, dim]` | `[0.0, 2.5]` |

The velocity update is:

```python
velocity = (
    inertia * velocity
    + cognitive * (personal_best_pos - positions)
    + social * social_signal
)
```

When `use_topology_social=True`, `social_signal` is `neighborhood_best_pos - positions`. When disabled, it falls back to `avg_pos`.

## Reward

The reward combines personal improvement and neighborhood-best improvement:

```python
personal_delta = scores - last_scores
neighborhood_delta = neighborhood_best_scores - last_neighborhood_best_scores
reward = 0.5 * tanh(personal_delta / scale) + 0.5 * tanh(neighborhood_delta / scale)
```

All values are sanitized with `torch.nan_to_num` before returning to the policy loop.

## Topology API

All topologies expose a boolean adjacency matrix `A` with shape `[num_particles, num_particles]`, where `A[i, j]` means particle `j` is visible to particle `i`.

Supported topology configs:

```yaml
# Fully connected gBest
type: global
include_self: true

# Ring lBest, +/- k neighbors
type: ring
k: 1
include_self: true

# Toroidal 2D grid. rows/cols are inferred if omitted.
type: von_neumann
rows: null
cols: null
include_self: true

# Dynamic position-space neighbors
type: knearest
k: 2
recompute_interval: 5
symmetric: true
include_self: true
```

`create_topology()` accepts either a flat topology dictionary or a Hydra-style wrapper with a nested `topology` entry. `KNearestTopology` needs positions at initialization and recomputes adjacency every `recompute_interval` steps.

## Usage Notes

- Use `global` as the exploitation-heavy baseline.
- Use `ring` or `von_neumann` when the experiment should preserve local sub-swarms and slower information flow.
- Use `knearest` for adaptive local communication based on particle positions.
- For `von_neumann`, ensure `rows * cols == num_agents` if you provide explicit grid dimensions.
- Keep `include_self=true` for standard PSO-style personal visibility unless you are testing a specific ablation.

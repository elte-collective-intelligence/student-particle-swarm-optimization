# Environment Module

This directory contains the Particle Swarm Optimization (PSO) environment implementation using TorchRL.

## Files

### `env.py`
**Main PSO environment implementation.** This file:
- Implements `PSOEnv` class extending TorchRL's `EnvBase`
- Handles multi-agent particle dynamics
- Computes rewards based on optimization progress
- Manages observation and action spaces

**Key Class: `PSOEnv`**

**Initialization Parameters:**
```python
PSOEnv(
    n_particles=10,        # Number of particles (agents)
    n_dims=2,              # Search space dimensionality
    max_steps=100,         # Max steps per episode
    bounds=(-5.0, 5.0),    # Search space boundaries
    objective_function="sphere",  # Function to optimize
)
```

**Observation Space:**
Each particle observes:
- `position`: Current position [n_dims]
- `velocity`: Current velocity [n_dims]
- `personal_best_position`: Best position found by this particle [n_dims]
- `personal_best_fitness`: Fitness at personal best [1]
- `global_best_position`: Best position found by swarm [n_dims]
- `global_best_fitness`: Best fitness in swarm [1]

**Action Space:**
Each particle outputs 4 continuous values:
- `inertia`: Weight for previous velocity
- `cognitive`: Weight for personal best attraction
- `social`: Weight for global best attraction
- `step_size`: Overall step magnitude

**Reward:**
Based on improvement in global best fitness:
```python
reward = (prev_best_fitness - new_best_fitness) / prev_best_fitness
```

**Key Methods:**
- `_reset()`: Initialize particles randomly in bounds
- `_step()`: Update particle positions based on PSO dynamics
- `_compute_fitness()`: Evaluate particles on objective function
- `_update_bests()`: Update personal and global bests

### `dynamic_functions.py`
**Optimization objective functions.** This file:
- Implements standard benchmark functions
- Supports dynamic (time-varying) functions
- Provides function factory for easy selection

**Available Functions:**

**`sphere(x)`**
- Simple convex function: f(x) = Σ x_i²
- Global minimum: f(0) = 0
- Use for: Initial testing, baseline

**`rastrigin(x)`**
- Multimodal: f(x) = 10n + Σ[x_i² - 10cos(2πx_i)]
- Many local minima, tests exploration
- Global minimum: f(0) = 0

**`rosenbrock(x)`**
- Valley-shaped: f(x) = Σ[100(x_{i+1} - x_i²)² + (1-x_i)²]
- Narrow valley, tests exploitation
- Global minimum: f(1,...,1) = 0

**`ackley(x)`**
- Complex multimodal with many local minima
- Tests both exploration and exploitation
- Global minimum: f(0) = 0

**Dynamic Functions:**
```python
# Function that changes over time
def dynamic_sphere(x, t):
    offset = np.sin(t * 0.1) * 2
    return sphere(x - offset)
```

### `__init__.py`
**Module exports.** Exports:
- `PSOEnv`: Main environment class
- `make_env()`: Factory function for environment creation
- Objective function utilities

## Usage

### Creating an Environment
```python
from src.envs import make_env

# Simple environment
env = make_env(n_particles=10, n_dims=2)

# With specific function
env = make_env(
    n_particles=20,
    n_dims=5,
    objective_function="rastrigin",
    max_steps=200
)
```

### Running an Episode
```python
from tensordict import TensorDict

# Reset
td = env.reset()

# Step loop
for _ in range(max_steps):
    # Get actions from policy
    actions = policy(td)
    
    # Step environment
    td = env.step(actions)
    
    # Check termination
    if td["done"].all():
        break
```

### Vectorized Environment
```python
from torchrl.envs import ParallelEnv

# Create 8 parallel environments
vec_env = ParallelEnv(
    num_workers=8,
    create_env_fn=lambda: make_env(n_particles=10)
)
```

## Architecture

```
PSOEnv
├── Observation Space
│   ├── position [n_particles, n_dims]
│   ├── velocity [n_particles, n_dims]
│   ├── personal_best_position [n_particles, n_dims]
│   ├── personal_best_fitness [n_particles, 1]
│   ├── global_best_position [n_dims]
│   └── global_best_fitness [1]
│
├── Action Space
│   └── parameters [n_particles, 4] (inertia, cognitive, social, step_size)
│
└── Dynamics
    ├── Velocity Update
    │   v = inertia*v + cognitive*r1*(pbest-x) + social*r2*(gbest-x)
    ├── Position Update
    │   x = x + step_size * v
    └── Boundary Handling
        └── Clip to bounds
```

## Key Concepts

### Multi-Agent Structure
Each particle is an independent agent that:
1. Observes its local state + global best
2. Outputs PSO coefficients
3. Receives shared reward (cooperative)

### Reward Design
Cooperative reward based on swarm progress:
- All particles receive same reward
- Encourages collective optimization
- Reward = improvement in global best fitness

### Episode Termination
Episode ends when:
- Maximum steps reached
- Global best fitness below threshold (solved)
- All particles stuck (no improvement for N steps)

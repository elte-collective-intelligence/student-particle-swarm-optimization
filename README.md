# Particle Swarm Optimization: Multi-Agent Reinforcement Learning (TorchRL)

[![CI](https://github.com/elte-collective-intelligence/student-particle-swarm-optimization/actions/workflows/ci.yml/badge.svg)](https://github.com/elte-collective-intelligence/student-particle-swarm-optimization/actions/workflows/ci.yml)
[![Docker](https://github.com/elte-collective-intelligence/student-particle-swarm-optimization/actions/workflows/docker.yml/badge.svg)](https://github.com/elte-collective-intelligence/student-particle-swarm-optimization/actions/workflows/docker.yml)
[![codecov](https://codecov.io/gh/elte-collective-intelligence/student-particle-swarm-optimization/branch/main/graph/badge.svg)](https://codecov.io/gh/elte-collective-intelligence/student-particle-swarm-optimization)
[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC--BY--NC--ND%204.0-blue.svg)](LICENSE)

---

## About the Project

This project implements **Particle Swarm Optimization (PSO) as a Multi-Agent Reinforcement Learning (MARL) system** using **TorchRL**.

Instead of deterministic PSO velocity and position update rules, particles are modeled as **learning agents** that acquire optimization behavior through **reinforcement learning**. The swarm cooperates to locate and track optima in both **static and dynamic objective landscapes**, while explicitly maintaining **diversity** to avoid premature convergence.

The system follows a **Centralized Training, Decentralized Execution (CTDE)** paradigm:

- During training, a **centralized critic** observes global swarm information
- During execution, **each particle acts independently** using only local observations

This repository contains the **complete implementation** required for **Assignment 2 – Collective Intelligence / MARL (2025/26/1)**.

---

## Author & Contributions

**Author:** Fidan Sadigova  
**Project type:** Solo project

All components of the project — environment design, reward shaping, PPO training, configuration management, testing, Dockerization, and reporting — were designed and implemented solely by the author.

---

## Features Overview

### TorchRL-native PSO Environment

- Continuous **d-dimensional optimization**
- Supports **static and dynamic objective functions**
- Fully TorchRL-compatible (`EnvBase`, `TensorDict` I/O)
- Parallel batch environments

### MARL with PPO-Clip (CTDE)

- Decentralized actors with shared parameters
- Centralized critic observing global swarm summaries
- PPO-Clip optimization with entropy regularization
- Advantages computed using **Generalized Advantage Estimation (GAE)**

### Reward Design

The reward function balances:

- **Personal improvement** (progress over personal best)
- **Global improvement** (swarm-level progress)
- **Diversity preservation** (anti-collapse behavior)
- **Stability under dynamic objective shifts**

### Metrics

- Best fitness over time
- Mean inter-particle distance (diversity)
- Stability under dynamic landscapes

### Reproducibility

- Dockerized execution
- Structured YAML configs (`env / algo / experiment`)
- Unit and smoke tests
- Deterministic seeds

---

## Environment Design

Each particle (agent) observes **local information only**:

- Position
- Velocity
- Personal best position and value
- Neighborhood summary (local communication)

Actions correspond to **learned PSO coefficients**:

- Inertia
- Cognitive component
- Social component

The environment supports **dynamic landscapes**, where the global optimum shifts over time, requiring adaptive swarm behavior.

---

## Training Pipeline (PPO-CTDE)

- Algorithm: **PPO-Clip**
- Shared actor network across agents
- Centralized critic with global observations
- Decentralized execution at evaluation time

Training outputs:

- Fitness curves
- Diversity curves
- Episode-level metrics

Plots are automatically saved under `outputs/`.

---

## Configuration System

All experiments are configurable via YAML files:

```text
configs/
├── env/
│   └── pso.yaml
├── algo/
│   └── ppo.yaml
└── experiment/
    └── base.yaml
```

Run training with:

```bash
python src/ppo_main.py +experiment=base
```

## Tests

The project includes unit and smoke tests covering:

- Environment reset and step correctness

- Reward and metric validity

- Finite and stable outputs

Run tests with:

```bash
export PYTHONPATH=$(pwd)
pytest
```

## Docker Usage

Build the Docker image:

```bash
docker build -t marl-pso -f docker/Dockerfile .
```

Run training inside Docker:

```bash
docker run --rm marl-pso
```

This ensures full reproducibility across systems.

## Results Summary

The trained swarm:

- Converges toward low objective values

- Preserves diversity over time

- Adapts to dynamic shifts in the optimization landscape

Plots are saved automatically in:
outputs/

## Limitations & Future Work

- Curriculum learning across dimensions and landscape difficulty

- Explicit role emergence analysis (scouts / exploiters)

- Communication topology ablations (gBest vs lBest)

## License

This project follows the original repository license:

CC BY-NC-ND 4.0

# Evaluation Configuration

This directory contains evaluation configuration files for the PSO environment.

## Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `num_eval_episodes` | Number of evaluation episodes | 10 |
| `max_steps` | Maximum steps per episode | 100 |
| `save_metrics` | Save evaluation metrics to JSON | true |
| `compare_random` | Compare against random baseline | true |

## Available Configurations

### Default (`default.yaml`)
Standard evaluation with 10 episodes and random baseline comparison.

### Smoke (`smoke.yaml`)
Quick smoke test with 2 episodes and 20 steps.

### Full (`full.yaml`)
Comprehensive evaluation with 20 episodes and 200 steps.

## Usage

```bash
# Default evaluation
python src/eval.py model_path=outputs/best_model.pt

# Use smoke test config
python src/eval.py model_path=outputs/best_model.pt eval.num_eval_episodes=2 eval.max_steps=20
```

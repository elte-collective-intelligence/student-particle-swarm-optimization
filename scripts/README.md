# Shell Scripts

This directory contains convenience scripts for running common tasks like training, evaluation, and batch experiments.

## Files

### `run_experiment.sh`
**Run a single experiment by name.** This is the primary script for executing experiments.

**Usage:**
```bash
./scripts/run_experiment.sh EXPERIMENT_NAME
```

**Examples:**
```bash
# Quick training test
./scripts/run_experiment.sh smoke_train

# Full training run
./scripts/run_experiment.sh full_train

# Dynamic function training
./scripts/run_experiment.sh dynamic_train

# Rastrigin function training
./scripts/run_experiment.sh rastrigin_train

# Evaluation with visualization
./scripts/run_experiment.sh eval_vis
```

**What it does:**
1. Validates experiment config exists
2. Checks for config file at `src/configs/experiments/{NAME}.yaml`
3. Runs: `python src/main.py --config-path configs/experiments --config-name {NAME}`
4. Shows available experiments if name invalid

**Error Handling:**
- Missing experiment → Shows available experiments
- Missing config → Error message with expected path
- Python errors → Displayed in terminal

### `train_all.sh`
**Run multiple training experiments sequentially.** Useful for training different configurations overnight.

**Usage:**
```bash
./scripts/train_all.sh
```

**What it does:**
1. Defines list of experiments to run
2. Runs each experiment sequentially
3. Continues even if one experiment fails
4. Reports progress between experiments

**Default Experiments:**
```bash
EXPERIMENTS=(
    "smoke_train"
    "full_train"
    "dynamic_train"
    "rastrigin_train"
)
```

**Customize:**
Edit the script to add/remove experiments:
```bash
# Edit train_all.sh
EXPERIMENTS=(
    "my_experiment_1"
    "my_experiment_2"
)
```

**Output:**
```
============================================================
Running all training experiments
============================================================

>>> Starting experiment: smoke_train
------------------------------------------------------------
[training output...]
>>> Finished experiment: smoke_train

>>> Starting experiment: full_train
...
```

### `eval_model.sh`
**Evaluate a trained model with visualization.** 

**Usage:**
```bash
./scripts/eval_model.sh MODEL_PATH [EXPERIMENT_CONFIG]
```

**Arguments:**
- `MODEL_PATH`: Path to trained model checkpoint (required)
- `EXPERIMENT_CONFIG`: Evaluation config name (default: `eval_vis`)

**Examples:**
```bash
# Evaluate with default visualization config
./scripts/eval_model.sh src/outputs/models/policy.pt

# Evaluate with specific config
./scripts/eval_model.sh src/outputs/models/policy.pt eval_vis
```

**What it does:**
1. Validates model file exists
2. Runs evaluation script with specified config
3. Generates visualizations and metrics

## Quick Reference

| Task | Command |
|------|---------|
| Quick training test | `./scripts/run_experiment.sh smoke_train` |
| Full training | `./scripts/run_experiment.sh full_train` |
| Train all experiments | `./scripts/train_all.sh` |
| Evaluate model | `./scripts/eval_model.sh path/to/model.pt` |
| List experiments | `ls src/configs/experiments/` |

## Making Scripts Executable

If you get "Permission denied" errors:
```bash
chmod +x scripts/*.sh
```

## Troubleshooting

### Script Not Found
Run from project root directory:
```bash
cd /path/to/student-particle-swarm-optimization
./scripts/run_experiment.sh smoke_train
```

### Python Not Found
Activate your virtual environment first:
```bash
source venv/bin/activate
./scripts/run_experiment.sh smoke_train
```

### Config Not Found
Ensure config exists at expected path:
```bash
ls src/configs/experiments/
```

### Model Not Found (eval)
Check the model path is correct:
```bash
ls src/outputs/models/
```

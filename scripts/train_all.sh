#!/bin/bash
# Run all training experiments sequentially
# Usage: ./scripts/train_all.sh

set -e

EXPERIMENTS=(
    "smoke_train"
    "full_train"
    "dynamic_train"
    "rastrigin_train"
)

echo "============================================================"
echo "Running all training experiments"
echo "============================================================"

for exp in "${EXPERIMENTS[@]}"; do
    echo ""
    echo ">>> Starting experiment: $exp"
    echo "------------------------------------------------------------"
    ./scripts/run_experiment.sh "$exp" || {
        echo "!!! Experiment $exp failed, continuing..."
    }
    echo ">>> Finished experiment: $exp"
done

echo ""
echo "============================================================"
echo "All experiments completed"
echo "============================================================"

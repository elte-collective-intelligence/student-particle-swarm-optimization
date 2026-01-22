#!/bin/bash
# Run a single experiment by name
# Usage: ./scripts/run_experiment.sh EXPERIMENT_NAME

set -e

EXPERIMENT=$1

if [ -z "$EXPERIMENT" ]; then
    echo "Usage: ./scripts/run_experiment.sh EXPERIMENT_NAME"
    echo ""
    echo "Available experiments:"
    ls -1 src/configs/experiments/ | grep -v README
    exit 1
fi

CONFIG_PATH="configs/experiments"
CONFIG_NAME="$EXPERIMENT"

if [ ! -f "src/configs/experiments/${EXPERIMENT}.yaml" ]; then
    echo "Error: Config not found at src/configs/experiments/${EXPERIMENT}.yaml"
    echo ""
    echo "Available experiments:"
    ls -1 src/configs/experiments/ | grep -v README | sed 's/.yaml$//'
    exit 1
fi

echo "Running experiment: $EXPERIMENT"
echo "Config: src/configs/experiments/${EXPERIMENT}.yaml"
echo ""

python src/main.py --config-path "$CONFIG_PATH" --config-name "$CONFIG_NAME"

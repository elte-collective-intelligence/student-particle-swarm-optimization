#!/bin/bash
# Evaluate a trained model
# Usage: ./scripts/eval_model.sh MODEL_PATH [EXPERIMENT_CONFIG]

set -e

MODEL_PATH=$1
EXPERIMENT=${2:-"eval_vis"}

if [ -z "$MODEL_PATH" ]; then
    echo "Usage: ./scripts/eval_model.sh MODEL_PATH [EXPERIMENT_CONFIG]"
    echo ""
    echo "Arguments:"
    echo "  MODEL_PATH        Path to trained model checkpoint"
    echo "  EXPERIMENT_CONFIG Evaluation config name (default: eval_vis)"
    echo ""
    echo "Available evaluation configs:"
    ls -1 src/configs/experiments/ | grep -v README | grep eval | sed 's/.yaml$//'
    exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model not found at $MODEL_PATH"
    exit 1
fi

echo "Evaluating model: $MODEL_PATH"
echo "Config: $EXPERIMENT"
echo ""

python src/eval.py --config-path configs/experiments --config-name "$EXPERIMENT" model_path="$MODEL_PATH"

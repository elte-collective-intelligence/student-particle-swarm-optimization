#!/usr/bin/env bash
set -euo pipefail
# Step 3 of 6 — function curriculum (Sphere → Rosenbrock → Rastrigin → Eggholder, 2D)
# Usage: ./scripts/run_step3_function.sh

echo "============================================================"
echo "Step 3/6: Function curriculum"
echo "============================================================"

if [[ ! -f "src/curriculum_train.py" ]]; then
  echo "Error: run this script from repository root."; exit 1
fi

run_cmd() { echo; echo ">>> $*"; "$@"; }

run_cmd python src/curriculum_train.py \
  --config-path configs/curriculum \
  --config-name function

echo
echo "Sanity check — expected output files:"
for seed in 42 43 44 45 46; do
  f="src/outputs/curriculum/function/seed_${seed}/model.pt"
  if [[ -f "$f" ]]; then echo "  OK  $f"; else echo "  MISSING  $f"; fi
done

echo
echo "Step 3 done. Run step 4 when ready:"
echo "  ./scripts/run_step4_dynamics.sh"

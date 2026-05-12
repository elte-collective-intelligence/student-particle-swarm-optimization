#!/usr/bin/env bash
set -euo pipefail
# Step 4 of 6 — dynamics curriculum (static → slow dynamics → fast dynamics)
# Usage: ./scripts/run_step4_dynamics.sh

echo "============================================================"
echo "Step 4/6: Dynamics curriculum"
echo "============================================================"

if [[ ! -f "src/curriculum_train.py" ]]; then
  echo "Error: run this script from repository root."; exit 1
fi

run_cmd() { echo; echo ">>> $*"; "$@"; }

run_cmd python src/curriculum_train.py \
  --config-path configs/curriculum \
  --config-name dynamics

echo
echo "Sanity check — expected output files:"
for seed in 42 43 44 45 46; do
  f="src/outputs/curriculum/dynamics/seed_${seed}/model.pt"
  if [[ -f "$f" ]]; then echo "  OK  $f"; else echo "  MISSING  $f"; fi
done

echo
echo "Step 4 done. Run step 5 when ready:"
echo "  ./scripts/run_step5_combined.sh"

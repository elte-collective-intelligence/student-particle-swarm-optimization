#!/usr/bin/env bash
set -euo pipefail
# Step 2 of 6 — dimension curriculum (2D → 5D → 10D → 30D on Sphere)
# Usage: ./scripts/run_step2_dimension.sh

echo "============================================================"
echo "Step 2/6: Dimension curriculum"
echo "============================================================"

if [[ ! -f "src/curriculum_train.py" ]]; then
  echo "Error: run this script from repository root."; exit 1
fi

run_cmd() { echo; echo ">>> $*"; "$@"; }

run_cmd python src/curriculum_train.py \
  --config-path configs/curriculum \
  --config-name dimension

echo
echo "Sanity check — expected output files:"
for seed in 42 43 44 45 46; do
  f="src/outputs/curriculum/dimension/seed_${seed}/model.pt"
  if [[ -f "$f" ]]; then echo "  OK  $f"; else echo "  MISSING  $f"; fi
done

echo
echo "Step 2 done. Run step 3 when ready:"
echo "  ./scripts/run_step3_function.sh"

#!/usr/bin/env bash
set -euo pipefail
# Step 5 of 6 — combined curriculum (function complexity first, then dimension scaling)
# Usage: ./scripts/run_step5_combined.sh

echo "============================================================"
echo "Step 5/6: Combined curriculum"
echo "============================================================"

if [[ ! -f "src/curriculum_train.py" ]]; then
  echo "Error: run this script from repository root."; exit 1
fi

run_cmd() { echo; echo ">>> $*"; "$@"; }

run_cmd python src/curriculum_train.py \
  --config-path configs/curriculum \
  --config-name combined

echo
echo "Sanity check — expected output files:"
for seed in 42 43 44 45 46; do
  f="src/outputs/curriculum/combined/seed_${seed}/model.pt"
  if [[ -f "$f" ]]; then echo "  OK  $f"; else echo "  MISSING  $f"; fi
done

echo
echo "Step 5 done. Run step 6 when ready:"
echo "  ./scripts/run_step6_gifs.sh"

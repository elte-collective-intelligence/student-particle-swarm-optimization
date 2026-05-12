#!/usr/bin/env bash
set -euo pipefail
# Step 6 of 6 — generate GIFs from curriculum-trained models
# Usage: ./scripts/run_step6_gifs.sh

echo "============================================================"
echo "Step 6/6: GIF generation"
echo "============================================================"

if [[ ! -f "src/curriculum_train.py" ]]; then
  echo "Error: run this script from repository root."; exit 1
fi

run_cmd() { echo; echo ">>> $*"; "$@"; }

mkdir -p images/semester_contribution

# Easy landscape — function curriculum model on sphere 2D
run_cmd python src/curriculum_eval_gif.py \
  --model-path src/outputs/curriculum/function/seed_42/model.pt \
  --config-name function \
  --landscape sphere \
  --dim 2 \
  --max-steps 200 \
  --gif-name curriculum_sphere_easy \
  --output-dir images/semester_contribution

# Hard landscape — function curriculum model on rastrigin 2D
run_cmd python src/curriculum_eval_gif.py \
  --model-path src/outputs/curriculum/function/seed_42/model.pt \
  --config-name function \
  --landscape rastrigin \
  --dim 2 \
  --max-steps 200 \
  --gif-name curriculum_rastrigin_hard \
  --output-dir images/semester_contribution

# Dynamic landscape — dynamics curriculum model on dynamic_sphere 2D
run_cmd python src/curriculum_eval_gif.py \
  --model-path src/outputs/curriculum/dynamics/seed_42/model.pt \
  --config-name dynamics \
  --landscape dynamic_sphere \
  --dim 2 \
  --max-steps 200 \
  --gif-name curriculum_dynamics \
  --output-dir images/semester_contribution

echo
echo "Sanity check — expected GIF files:"
for name in curriculum_sphere_easy curriculum_rastrigin_hard curriculum_dynamics; do
  f="images/semester_contribution/${name}.gif"
  if [[ -f "$f" ]]; then echo "  OK  $f"; else echo "  MISSING  $f"; fi
done

echo
echo "All 6 steps complete."
echo "Results are in src/outputs/curriculum/ and images/semester_contribution/"

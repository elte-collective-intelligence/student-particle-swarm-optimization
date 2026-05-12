#!/usr/bin/env bash
set -euo pipefail

# Run from repository root:
#   ./scripts/run_task2_all.sh
#
# Optional environment variables:
#   RUN_FULL_TESTS=1      # also runs: pytest test/ -v
#   SKIP_FULL_CURRICULA=1 # skips steps 2–5 (full curriculum runs); still runs step 6 GIFs

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================================"
echo "Task 2 pipeline: steps 1–6 (tests, curricula, GIFs)"
echo "============================================================"

if [[ ! -f "src/curriculum_train.py" ]]; then
  echo "Error: run this script from repository root."
  exit 1
fi

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  echo "Using virtual environment: ${VIRTUAL_ENV}"
else
  echo "Warning: no active virtual environment detected."
fi

run_cmd() {
  echo
  echo ">>> $*"
  "$@"
}

run_cmd "${SCRIPT_DIR}/run_step1_tests.sh"

if [[ "${RUN_FULL_TESTS:-0}" == "1" ]]; then
  run_cmd pytest test/ -v
fi

if [[ "${SKIP_FULL_CURRICULA:-0}" != "1" ]]; then
  run_cmd "${SCRIPT_DIR}/run_step2_dimension.sh"
  run_cmd "${SCRIPT_DIR}/run_step3_function.sh"
  run_cmd "${SCRIPT_DIR}/run_step4_dynamics.sh"
  run_cmd "${SCRIPT_DIR}/run_step5_combined.sh"
fi

run_cmd "${SCRIPT_DIR}/run_step6_gifs.sh"

echo
echo "============================================================"
echo "Sanity check: expected artifacts"
echo "============================================================"
run_cmd ls src/outputs/curriculum/dimension/comparison.png src/outputs/curriculum/dimension/generalization.png src/outputs/curriculum/dimension/generalization.npy
run_cmd ls src/outputs/curriculum/function/comparison.png src/outputs/curriculum/function/generalization.png src/outputs/curriculum/function/generalization.npy
run_cmd ls src/outputs/curriculum/dynamics/comparison.png src/outputs/curriculum/dynamics/generalization.png src/outputs/curriculum/dynamics/generalization.npy
run_cmd ls src/outputs/curriculum/combined/comparison.png src/outputs/curriculum/combined/generalization.png src/outputs/curriculum/combined/generalization.npy
run_cmd ls images/semester_contribution/curriculum_sphere_easy_2d.gif \
           images/semester_contribution/curriculum_sphere_easy_3d.gif
run_cmd ls images/semester_contribution/curriculum_rastrigin_hard_2d.gif \
           images/semester_contribution/curriculum_rastrigin_hard_3d.gif
run_cmd ls images/semester_contribution/curriculum_dynamics_2d.gif \
           images/semester_contribution/curriculum_dynamics_3d.gif

echo
echo "Task 2 pipeline completed successfully."

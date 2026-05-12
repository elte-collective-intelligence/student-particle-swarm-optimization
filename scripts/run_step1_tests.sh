#!/usr/bin/env bash
set -euo pipefail
# Step 1 of 6 — run tests + smoke training check
# Usage: ./scripts/run_step1_tests.sh

echo "============================================================"
echo "Step 1/6: Tests + smoke training check"
echo "============================================================"

if [[ ! -f "src/curriculum_train.py" ]]; then
  echo "Error: run this script from repository root."; exit 1
fi

run_cmd() { echo; echo ">>> $*"; "$@"; }

run_cmd pytest test/test_curriculum.py -v

run_cmd python src/curriculum_train.py --config-path configs/curriculum --config-name dimension \
  "seeds=[42]" env.num_agents=5 env.batch_size=4 frames_per_batch=256 \
  minibatch_size=64 num_epochs=2 run_generalization=false run_baseline_comparison=false \
  run_domain_rand=false output_dir=src/outputs/curriculum/smoke_repro

echo
echo "Step 1 done. Run step 2 when ready:"
echo "  ./scripts/run_step2_dimension.sh"

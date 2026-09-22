#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

source /home/labusers/nchu4113056013/miniconda3/etc/profile.d/conda.sh
conda activate tsp_env

LOG_DIR="../results/logs"
mkdir -p "$LOG_DIR"

run_step() {
  local step_name="$1"
  local cmd="$2"
  echo "=================================================="
  echo "[STEP] $step_name"
  echo "CMD: $cmd"
  echo "=================================================="
  bash -lc "$cmd" 2>&1 | tee "$LOG_DIR/${step_name}.log"
  echo "[DONE] $step_name"
  echo
}

run_step generate_tsp "python generate_tsp.py"
run_step tune "python tune.py"
run_step train "python train.py"
run_step test_histogram "python test_histogram.py"
run_step plot_qualitative "python plot_qualitative.py"

echo "=================================================="

echo "Full pipeline finished successfully."

echo "Generated files:"
ls -1

#!/bin/bash
#SBATCH -J dg-pipeline
#SBATCH -o logs/dg-pipeline-%j.out
#SBATCH -e logs/dg-pipeline-%j.err
#SBATCH -t 12:00:00
#
# Fill these in for your PSC allocation / queue.
#SBATCH -A YOUR_PSC_ALLOCATION
#SBATCH -p YOUR_GPU_PARTITION
#
# Keep these aligned with Stage-1 red_team.cluster in your training config.
# Examples:
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=6
#SBATCH --mem=0

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/project}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
EXPERIMENT_ID="${EXPERIMENT_ID:-psc_run_001}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/outputs/${EXPERIMENT_ID}}"

TRAIN_CONFIG="${TRAIN_CONFIG:-$PROJECT_DIR/david_and_goliath/configs/train.yaml}"
BLUE_CONFIG="${BLUE_CONFIG:-$PROJECT_DIR/david_and_goliath/configs/blue_team/full_tools.yaml}"
ORACLE_CONFIG="${ORACLE_CONFIG:-$PROJECT_DIR/david_and_goliath/configs/oracle/hybrid_oracle.yaml}"
OOD_TASKS_PATH="${OOD_TASKS_PATH:-}"

# Optional:
# export OPENAI_API_KEY=...
# export WANDB_API_KEY=...

mkdir -p "$PROJECT_DIR/logs"
cd "$PROJECT_DIR"

if [[ -d "$VENV_DIR" ]]; then
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
fi

export PYTHONUNBUFFERED=1

cmd=(
  python -m david_and_goliath.scripts.run_pipeline
  --config "$TRAIN_CONFIG"
  --blue-config "$BLUE_CONFIG"
  --oracle-config "$ORACLE_CONFIG"
  --experiment-id "$EXPERIMENT_ID"
  --output-dir "$OUTPUT_DIR"
  --oracle-mode payload_only
  --concurrency 8
  --log-level INFO
)

if [[ -n "$OOD_TASKS_PATH" ]]; then
  cmd+=(--ood-tasks-path "$OOD_TASKS_PATH")
fi

"${cmd[@]}"

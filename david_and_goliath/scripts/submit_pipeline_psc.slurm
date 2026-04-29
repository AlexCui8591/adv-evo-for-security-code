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

TRAIN_CONFIG="${TRAIN_CONFIG:-}"
BLUE_CONFIG="${BLUE_CONFIG:-$PROJECT_DIR/david_and_goliath/configs/blue_team/full_tools.yaml}"
ORACLE_CONFIG="${ORACLE_CONFIG:-$PROJECT_DIR/david_and_goliath/configs/oracle/hybrid_oracle.yaml}"
OOD_TASKS_PATH="${OOD_TASKS_PATH:-}"
START_RAY="${START_RAY:-1}"
RAY_NUM_GPUS="${RAY_NUM_GPUS:-${SLURM_GPUS_ON_NODE:-${SLURM_GPUS:-0}}}"
RAY_HEAD_IP="${RAY_HEAD_IP:-0.0.0.0}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"

# Optional:
# export OPENAI_API_KEY=...
# export WANDB_API_KEY=...

mkdir -p "$PROJECT_DIR/logs"
cd "$PROJECT_DIR"

if [[ -d "$VENV_DIR" ]]; then
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
fi

if [[ -f "$PROJECT_DIR/david_and_goliath/scripts/psc_runtime_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$PROJECT_DIR/david_and_goliath/scripts/psc_runtime_env.sh"
fi

mkdir -p "${RAY_TMPDIR:-/tmp/ray-${USER}}"

cleanup_ray() {
  if command -v ray >/dev/null 2>&1; then
    ray stop --force >/dev/null 2>&1 || true
  fi
}
trap cleanup_ray EXIT

if [[ "$START_RAY" == "1" ]]; then
  cleanup_ray
  ray start \
    --head \
    --node-ip-address "$RAY_HEAD_IP" \
    --dashboard-host 0.0.0.0 \
    --dashboard-port "$RAY_DASHBOARD_PORT" \
    --num-gpus "$RAY_NUM_GPUS"
  export RAY_ADDRESS="${RAY_CONNECT_ADDRESS:-auto}"
fi

cmd=(
  python -m david_and_goliath.scripts.run_pipeline
  --blue-config "$BLUE_CONFIG"
  --oracle-config "$ORACLE_CONFIG"
  --experiment-id "$EXPERIMENT_ID"
  --output-dir "$OUTPUT_DIR"
  --oracle-mode payload_only
  --concurrency 8
  --log-level INFO
)

if [[ -n "$TRAIN_CONFIG" ]]; then
  cmd+=(--config "$TRAIN_CONFIG")
fi

if [[ -n "$OOD_TASKS_PATH" ]]; then
  cmd+=(--ood-tasks-path "$OOD_TASKS_PATH")
fi

"${cmd[@]}"

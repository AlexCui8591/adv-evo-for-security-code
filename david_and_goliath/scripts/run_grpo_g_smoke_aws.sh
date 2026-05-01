#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

export CONFIG_PATH="${CONFIG_PATH:-${PROJECT_DIR}/david_and_goliath/configs/experiment/aws_grpo_g_smoke.yaml}"
export EXPERIMENT_ID="${EXPERIMENT_ID:-aws_grpo_g_smoke}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/${EXPERIMENT_ID}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-1}"

exec bash "${PROJECT_DIR}/david_and_goliath/scripts/run_grpo_8b_aws.sh" "$@"

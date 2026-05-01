#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_DIR}/david_and_goliath/configs/experiment/aws_grpo_8b_minimal.yaml}"
EXPERIMENT_ID="${EXPERIMENT_ID:-aws_grpo_8b_minimal}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/${EXPERIMENT_ID}}"

if [[ -f "${PROJECT_DIR}/david_and_goliath/scripts/psc_runtime_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "${PROJECT_DIR}/david_and_goliath/scripts/psc_runtime_env.sh"
fi

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES
fi
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export VLLM_ALLOW_INSECURE_SERIALIZATION="${VLLM_ALLOW_INSECURE_SERIALIZATION:-1}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES="${RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES:-1}"
if [[ -n "${REWARD_MODE:-}" ]]; then
  export DG_REWARD_MODE="${REWARD_MODE}"
else
  export DG_REWARD_MODE="${DG_REWARD_MODE:-rule}"
fi

cd "${PROJECT_DIR}"
mkdir -p "${OUTPUT_DIR}"

EXTRA_ARGS=()
if [[ -n "${MODEL_PATH:-}" ]]; then
  EXTRA_ARGS+=(--model-path "${MODEL_PATH}")
fi
if [[ -n "${REWARD_MODE:-}" ]]; then
  EXTRA_ARGS+=(--reward-mode "${REWARD_MODE}")
fi
if [[ -n "${NUM_PROMPTS:-}" ]]; then
  EXTRA_ARGS+=(--num-prompts "${NUM_PROMPTS}")
fi
if [[ -n "${VAL_NUM_PROMPTS:-}" ]]; then
  EXTRA_ARGS+=(--val-num-prompts "${VAL_NUM_PROMPTS}")
fi
if [[ -n "${TRAIN_BATCH_SIZE:-}" ]]; then
  EXTRA_ARGS+=(--train-batch-size "${TRAIN_BATCH_SIZE}")
fi
if [[ -n "${ROLLOUT_N:-}" ]]; then
  EXTRA_ARGS+=(--rollout-n "${ROLLOUT_N}")
fi
if [[ -n "${N_GPUS_PER_NODE:-}" ]]; then
  EXTRA_ARGS+=(--n-gpus-per-node "${N_GPUS_PER_NODE}")
fi
if [[ "${PREPARE_ONLY:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--skip-verl)
fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--dry-run)
fi

python -m david_and_goliath.scripts.run_stage1_verl \
  --config "${CONFIG_PATH}" \
  --experiment-id "${EXPERIMENT_ID}" \
  --output-dir "${OUTPUT_DIR}" \
  "${EXTRA_ARGS[@]}" \
  "$@"

#!/usr/bin/env bash
# Single-node smoke test:
#   GPU 0-3: local Blue Team vLLM server (Qwen2.5-Coder-32B-Instruct)
#   GPU 4-7: Red Team GRPO smoke training (Llama 3.1 8B + final_adapter)
#
# Submit from the repository root:
#   sbatch david_and_goliath/scripts/slurm_smoke_train_32b.sbatch
#
# To override the GPU type at submit time, for example:
#   sbatch --gres=gpu:h100-80:8 david_and_goliath/scripts/slurm_smoke_train_32b.sbatch

#SBATCH -A cis250260p
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:v100-32:8
#SBATCH -t 04:00:00
#SBATCH -J dg-smoke-32b
#SBATCH -o dg-smoke-32b-%j.out

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/ocean/projects/cis250260p/cuiz/11766-project/adv-evo-for-security-code}"
CONDA_ENV="${CONDA_ENV:-dg}"
CONFIG_PATH="${CONFIG_PATH:-david_and_goliath/configs/experiment/coevo_8b_local_blue_32b.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/train_once_llama8b_lora_qwen_coder_32b}"

BLUE_MODEL="${BLUE_MODEL:-Qwen/Qwen2.5-Coder-32B-Instruct}"
BLUE_HOST="${BLUE_HOST:-127.0.0.1}"
BLUE_PORT="${BLUE_PORT:-8000}"
BLUE_GPUS="${BLUE_GPUS:-0,1,2,3}"
BLUE_TENSOR_PARALLEL_SIZE="${BLUE_TENSOR_PARALLEL_SIZE:-4}"
BLUE_MAX_MODEL_LEN="${BLUE_MAX_MODEL_LEN:-4096}"
BLUE_GPU_MEMORY_UTILIZATION="${BLUE_GPU_MEMORY_UTILIZATION:-0.90}"
BLUE_READY_TIMEOUT_SECONDS="${BLUE_READY_TIMEOUT_SECONDS:-1200}"

RED_GPUS="${RED_GPUS:-4,5,6,7}"

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR}/logs"

BLUE_LOG="${OUTPUT_DIR}/logs/blue_vllm_${SLURM_JOB_ID:-manual}.log"

cleanup() {
    local code=$?
    echo "[$(date)] cleanup starting with exit code ${code}"
    if [[ -n "${VLLM_PID:-}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "Stopping Blue Team vLLM server pid=${VLLM_PID}"
        kill "${VLLM_PID}" 2>/dev/null || true
        wait "${VLLM_PID}" 2>/dev/null || true
    fi
    ray stop --force >/dev/null 2>&1 || true
    exit "${code}"
}
trap cleanup EXIT INT TERM

echo "Job ID       : ${SLURM_JOB_ID:-manual}"
echo "Node         : $(hostname)"
echo "Project root : ${PROJECT_ROOT}"
echo "Config       : ${CONFIG_PATH}"
echo "Blue GPUs    : ${BLUE_GPUS}"
echo "Red GPUs     : ${RED_GPUS}"
echo "Blue log     : ${BLUE_LOG}"

if command -v module >/dev/null 2>&1; then
    module load cuda/12.1 >/dev/null 2>&1 || true
fi

# shellcheck disable=SC1091
source david_and_goliath/scripts/hf_env.sh

if [[ -f david_and_goliath/scripts/secrets.sh ]]; then
    # shellcheck disable=SC1091
    source david_and_goliath/scripts/secrets.sh
else
    echo "Missing david_and_goliath/scripts/secrets.sh"
    echo "Copy david_and_goliath/scripts/secrets.sh.example and add OPENAI_API_KEY first."
    exit 1
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "OPENAI_API_KEY is not set after sourcing secrets.sh"
    exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray-${USER}-${SLURM_JOB_ID:-manual}}"
mkdir -p "${RAY_TMPDIR}"

echo "Python       : $(python -V 2>&1)"
echo "HF_HOME      : ${HF_HOME}"
echo "HF offline   : ${HF_HUB_OFFLINE}"
echo "API key set  : yes"
echo "nvidia-smi:"
nvidia-smi || true

echo "[$(date)] Starting Blue Team vLLM server..."
CUDA_VISIBLE_DEVICES="${BLUE_GPUS}" vllm serve "${BLUE_MODEL}" \
    --served-model-name "${BLUE_MODEL}" \
    --host "${BLUE_HOST}" \
    --port "${BLUE_PORT}" \
    --tensor-parallel-size "${BLUE_TENSOR_PARALLEL_SIZE}" \
    --dtype float16 \
    --max-model-len "${BLUE_MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${BLUE_GPU_MEMORY_UTILIZATION}" \
    > "${BLUE_LOG}" 2>&1 &
VLLM_PID=$!
echo "Blue Team vLLM pid=${VLLM_PID}"

echo "[$(date)] Waiting for Blue Team endpoint..."
deadline=$((SECONDS + BLUE_READY_TIMEOUT_SECONDS))
until curl -fsS "http://${BLUE_HOST}:${BLUE_PORT}/v1/models" >/dev/null; do
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "Blue Team vLLM exited before becoming ready. Last log lines:"
        tail -n 200 "${BLUE_LOG}" || true
        exit 1
    fi
    if (( SECONDS >= deadline )); then
        echo "Timed out waiting for Blue Team endpoint. Last log lines:"
        tail -n 200 "${BLUE_LOG}" || true
        exit 1
    fi
    echo "waiting for http://${BLUE_HOST}:${BLUE_PORT}/v1/models ..."
    sleep 10
done
echo "[$(date)] Blue Team endpoint is ready."
curl -s "http://${BLUE_HOST}:${BLUE_PORT}/v1/models" || true
echo

echo "[$(date)] Running Red Team smoke training..."
CUDA_VISIBLE_DEVICES="${RED_GPUS}" python david_and_goliath/scripts/run_coevolution.py \
    --config "${CONFIG_PATH}"

echo "[$(date)] Smoke training finished successfully."

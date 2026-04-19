#!/usr/bin/env bash
# Single-node full co-evolution run on 8 GPUs:
#   GPUs 0-3: local Blue Team vLLM server (Qwen2.5-Coder-32B-Instruct)
#   GPUs 4-7: Red Team Ray cluster
#
# The source config david_and_goliath/configs/experiment/coevo_8b.yaml is a
# 20-round full config, but its default Red Team cluster asks for 12 GPUs.
# This script writes a job-local config that keeps the 20-round experiment and
# downscales Red Team resources to fit the 4 GPUs left after Blue vLLM starts.
#
# Submit from the repository root:
#   sbatch david_and_goliath/scripts/run_coevolution_full.sbatch
#
# Useful overrides:
#   sbatch --export=ALL,TOTAL_ROUNDS=5 david_and_goliath/scripts/run_coevolution_full.sbatch
#   sbatch --export=ALL,RESUME=true david_and_goliath/scripts/run_coevolution_full.sbatch
#   sbatch --gres=gpu:v100-32:8 david_and_goliath/scripts/run_coevolution_full.sbatch

#SBATCH -A cis250260p
#SBATCH -p GPU
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:h100-80:8
#SBATCH --mem=504G
#SBATCH -t 24:00:00
#SBATCH -J dg-full-8b-32b
#SBATCH -o dg-full-8b-32b-%j.out

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/ocean/projects/cis250260p/cuiz/11766-project/adv-evo-for-security-code}"
CONDA_ENV="${CONDA_ENV:-dg}"
CONFIG_PATH="${CONFIG_PATH:-david_and_goliath/configs/experiment/coevo_8b.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/coevo_8b_local_blue_32b_single_node}"

BLUE_MODEL="${BLUE_MODEL:-Qwen/Qwen2.5-Coder-32B-Instruct}"
BLUE_HOST="${BLUE_HOST:-127.0.0.1}"
BLUE_GPUS="${BLUE_GPUS:-0,1,2,3}"
BLUE_TENSOR_PARALLEL_SIZE="${BLUE_TENSOR_PARALLEL_SIZE:-4}"
BLUE_PORT="${BLUE_PORT:-8000}"
BLUE_MAX_MODEL_LEN="${BLUE_MAX_MODEL_LEN:-4096}"
BLUE_GPU_MEMORY_UTILIZATION="${BLUE_GPU_MEMORY_UTILIZATION:-0.90}"
BLUE_READY_TIMEOUT_SECONDS="${BLUE_READY_TIMEOUT_SECONDS:-1800}"

RED_GPUS="${RED_GPUS:-4,5,6,7}"
RED_RAY_GPUS="${RED_RAY_GPUS:-4}"
RED_TRAINING_GPUS="${RED_TRAINING_GPUS:-2}"
RED_INFERENCE_GPUS="${RED_INFERENCE_GPUS:-2}"
RED_TENSOR_PARALLEL_SIZE="${RED_TENSOR_PARALLEL_SIZE:-1}"
RED_REWARD_WORKERS="${RED_REWARD_WORKERS:-4}"

RAY_PORT="${RAY_PORT:-6379}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
RAY_READY_TIMEOUT_SECONDS="${RAY_READY_TIMEOUT_SECONDS:-600}"

HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

cd "${PROJECT_ROOT}"

NODE_NAME="$(hostname)"
BLUE_BASE_URL="http://${BLUE_HOST}:${BLUE_PORT}/v1"
RAY_ADDRESS="${NODE_NAME}:${RAY_PORT}"
RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray-${USER}-${SLURM_JOB_ID:-manual}}"

mkdir -p "${OUTPUT_DIR}/logs" "${OUTPUT_DIR}/generated_configs"
BLUE_LOG="${OUTPUT_DIR}/logs/blue_vllm_${SLURM_JOB_ID:-manual}.log"
JOB_CONFIG="${OUTPUT_DIR}/generated_configs/full_${SLURM_JOB_ID:-manual}.yaml"

export PROJECT_ROOT CONDA_ENV CONFIG_PATH OUTPUT_DIR
export BLUE_MODEL BLUE_HOST BLUE_GPUS BLUE_TENSOR_PARALLEL_SIZE BLUE_PORT
export BLUE_MAX_MODEL_LEN BLUE_GPU_MEMORY_UTILIZATION BLUE_BASE_URL
export RED_GPUS RED_RAY_GPUS RED_TRAINING_GPUS RED_INFERENCE_GPUS
export RED_TENSOR_PARALLEL_SIZE RED_REWARD_WORKERS
export RAY_PORT RAY_DASHBOARD_PORT RAY_ADDRESS
export RAY_TMPDIR HF_HUB_OFFLINE TOKENIZERS_PARALLELISM OMP_NUM_THREADS
export BLUE_LOG JOB_CONFIG

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

activate_env() {
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
        echo "Copy david_and_goliath/scripts/secrets.sh.example and set OPENAI_API_KEY first."
        exit 1
    fi

    if [[ -z "${OPENAI_API_KEY:-}" ]]; then
        echo "OPENAI_API_KEY is not set after sourcing secrets.sh"
        exit 1
    fi

    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"

    export HF_HUB_OFFLINE TOKENIZERS_PARALLELISM OMP_NUM_THREADS RAY_TMPDIR
    mkdir -p "${RAY_TMPDIR}"
}

if (( RED_TRAINING_GPUS + RED_INFERENCE_GPUS > RED_RAY_GPUS )); then
    echo "Invalid Red Team GPU split:"
    echo "  RED_TRAINING_GPUS + RED_INFERENCE_GPUS must be <= RED_RAY_GPUS"
    echo "  got ${RED_TRAINING_GPUS} + ${RED_INFERENCE_GPUS} > ${RED_RAY_GPUS}"
    exit 1
fi

echo "Job ID        : ${SLURM_JOB_ID:-manual}"
echo "Node          : ${NODE_NAME}"
echo "Project root  : ${PROJECT_ROOT}"
echo "Config source : ${CONFIG_PATH}"
echo "Job config    : ${JOB_CONFIG}"
echo "Output dir    : ${OUTPUT_DIR}"
echo "Blue GPUs     : ${BLUE_GPUS} (tp=${BLUE_TENSOR_PARALLEL_SIZE})"
echo "Red GPUs      : ${RED_GPUS}"
echo "Red Ray GPUs  : ${RED_RAY_GPUS}"
echo "Red split     : train=${RED_TRAINING_GPUS}, infer=${RED_INFERENCE_GPUS}, tp=${RED_TENSOR_PARALLEL_SIZE}"
echo "Blue URL      : ${BLUE_BASE_URL}"
echo "HF offline    : ${HF_HUB_OFFLINE}"

activate_env

echo "Python        : $(python -V 2>&1)"
echo "HF_HOME       : ${HF_HOME}"
echo "API key set   : yes"
echo "nvidia-smi:"
nvidia-smi || true

echo "[$(date)] Generating job-specific config..."
python - <<'PY'
import os
from pathlib import Path

import yaml

config_path = Path(os.environ["CONFIG_PATH"])
job_config = Path(os.environ["JOB_CONFIG"])
output_dir = os.environ["OUTPUT_DIR"]

with config_path.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

cfg.setdefault("blue_team", {})
cfg["blue_team"]["base_url"] = os.environ["BLUE_BASE_URL"]
cfg["blue_team"]["model"] = os.environ["BLUE_MODEL"]
cfg["blue_team"]["api_key"] = "EMPTY"

cfg.setdefault("red_team", {})
cfg["red_team"].setdefault("cluster", {})
cfg["red_team"]["cluster"]["num_training_gpus"] = int(os.environ["RED_TRAINING_GPUS"])
cfg["red_team"]["cluster"]["num_inference_gpus"] = int(os.environ["RED_INFERENCE_GPUS"])
cfg["red_team"]["cluster"]["tensor_parallel_size"] = int(os.environ["RED_TENSOR_PARALLEL_SIZE"])
cfg["red_team"]["cluster"]["num_reward_workers"] = int(os.environ["RED_REWARD_WORKERS"])

cfg["output_dir"] = output_dir

if os.environ.get("TOTAL_ROUNDS"):
    cfg["total_rounds"] = int(os.environ["TOTAL_ROUNDS"])

if os.environ.get("EXPERIMENT_ID"):
    cfg["experiment_id"] = os.environ["EXPERIMENT_ID"]
    cfg["output_dir"] = f"outputs/{os.environ['EXPERIMENT_ID']}"

resume = os.environ.get("RESUME")
if resume is not None:
    cfg["resume"] = resume.lower() in {"1", "true", "yes", "y"}

job_config.parent.mkdir(parents=True, exist_ok=True)
with job_config.open("w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

cluster = cfg["red_team"]["cluster"]
print(f"Wrote {job_config}")
print(f"blue_team.base_url = {cfg['blue_team']['base_url']}")
print(f"total_rounds       = {cfg.get('total_rounds')}")
print(f"resume             = {cfg.get('resume')}")
print(f"output_dir         = {cfg.get('output_dir')}")
print(
    "red_team.cluster  = "
    f"train {cluster['num_training_gpus']}, "
    f"infer {cluster['num_inference_gpus']}, "
    f"tp {cluster['tensor_parallel_size']}, "
    f"workers {cluster['num_reward_workers']}"
)
PY

echo "[$(date)] Stopping any stale Ray processes..."
ray stop --force >/dev/null 2>&1 || true

echo "[$(date)] Starting Blue Team vLLM server..."
echo "Blue node: ${NODE_NAME}" > "${BLUE_LOG}"
echo "CUDA_VISIBLE_DEVICES=${BLUE_GPUS}" >> "${BLUE_LOG}"
CUDA_VISIBLE_DEVICES="${BLUE_GPUS}" vllm serve "${BLUE_MODEL}" \
    --served-model-name "${BLUE_MODEL}" \
    --host "${BLUE_HOST}" \
    --port "${BLUE_PORT}" \
    --tensor-parallel-size "${BLUE_TENSOR_PARALLEL_SIZE}" \
    --dtype float16 \
    --max-model-len "${BLUE_MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${BLUE_GPU_MEMORY_UTILIZATION}" \
    >> "${BLUE_LOG}" 2>&1 &
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

echo "[$(date)] Starting Ray head on Red Team GPUs ${RED_GPUS}..."
CUDA_VISIBLE_DEVICES="${RED_GPUS}" ray start --head \
    --node-ip-address="${NODE_NAME}" \
    --port="${RAY_PORT}" \
    --dashboard-host=0.0.0.0 \
    --dashboard-port="${RAY_DASHBOARD_PORT}" \
    --num-gpus="${RED_RAY_GPUS}" \
    --temp-dir="${RAY_TMPDIR}"

echo "[$(date)] Waiting for Ray GPU resources..."
deadline=$((SECONDS + RAY_READY_TIMEOUT_SECONDS))
until python - <<'PY'
import os
import sys

import ray

target = int(os.environ["RED_RAY_GPUS"])
ray.init(address=os.environ["RAY_ADDRESS"], ignore_reinit_error=True, logging_level="ERROR")
resources = ray.cluster_resources()
available = int(resources.get("GPU", 0))
print(f"Ray GPU resources: {available}/{target}; resources={resources}")
ray.shutdown()
sys.exit(0 if available >= target else 1)
PY
do
    if (( SECONDS >= deadline )); then
        echo "Timed out waiting for Ray resources."
        exit 1
    fi
    sleep 10
done

echo "[$(date)] Running full co-evolution training..."
python david_and_goliath/scripts/run_coevolution.py \
    --config "${JOB_CONFIG}" \
    --log-level "${LOG_LEVEL:-INFO}"

echo "[$(date)] Full co-evolution training finished successfully."

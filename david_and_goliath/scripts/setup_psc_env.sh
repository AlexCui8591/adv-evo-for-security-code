#!/usr/bin/env bash
# veRL-first PSC environment setup for David & Goliath.
#
# Prefer running inside a veRL/vLLM CUDA container or a PSC-provided CUDA 12.8+
# stack. veRL owns the GRPO trainer, Ray orchestration, and vLLM rollout path.
#
#   docker run --runtime=nvidia --gpus all -it --rm --shm-size="10g" \
#     --cap-add=SYS_ADMIN -v "$PWD:/workspace" verlai/verl:vllm011.latest bash
#
# On PSC, use the site-supported container runner when Docker is unavailable
# (for example Apptainer/Singularity with the same NGC image).
#
# Usage inside the container or PSC shell:
#   cd /path/to/project
#   bash david_and_goliath/scripts/setup_psc_env.sh
#
# Optional venv fallback:
#   CREATE_VENV=1 PYTHON_BIN=python3 VENV_DIR=$SCRATCH/dg-venv \
#     bash david_and_goliath/scripts/setup_psc_env.sh

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
CREATE_VENV="${CREATE_VENV:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venv-psc}"

VERL_SPEC="${VERL_SPEC:-verl[vllm]}"
VLLM_SPEC="${VLLM_SPEC:-vllm==0.11.0}"
TORCH_SPEC="${TORCH_SPEC:-torch==2.8.0}"
TORCHVISION_SPEC="${TORCHVISION_SPEC:-torchvision==0.23.0}"
TORCHAUDIO_SPEC="${TORCHAUDIO_SPEC:-torchaudio==2.8.0}"
TRANSFORMERS_SPEC="${TRANSFORMERS_SPEC:-transformers==4.55.4}"
TOKENIZERS_SPEC="${TOKENIZERS_SPEC:-tokenizers>=0.21.1}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
INSTALL_TORCH_STACK="${INSTALL_TORCH_STACK:-1}"
RUN_SMOKE_TEST="${RUN_SMOKE_TEST:-1}"
UNINSTALL_CONFLICTS="${UNINSTALL_CONFLICTS:-1}"
FREEZE_LOCK="${FREEZE_LOCK:-1}"

echo "Project dir       : ${PROJECT_DIR}"
echo "Create venv       : ${CREATE_VENV}"
echo "Python bin        : ${PYTHON_BIN}"
echo "veRL spec         : ${VERL_SPEC}"
echo "vLLM spec         : ${VLLM_SPEC}"
echo "Torch spec        : ${TORCH_SPEC}"
echo "Transformers spec : ${TRANSFORMERS_SPEC}"
echo "PyTorch index     : ${PYTORCH_INDEX_URL}"

if [[ "${CREATE_VENV}" == "1" ]]; then
  if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "ERROR: ${PYTHON_BIN} not found. Load a Python module first, or set PYTHON_BIN." >&2
    exit 2
  fi
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
fi

python - <<'PY'
import sys
major, minor = sys.version_info[:2]
print(f"Python version    : {major}.{minor}")
if (major, minor) < (3, 10):
    raise SystemExit("ERROR: Python >= 3.10 is required.")
PY

python -m pip install --upgrade pip setuptools wheel packaging ninja cmake

CONSTRAINTS_FILE="$(mktemp)"
cat > "${CONSTRAINTS_FILE}" <<EOF
${TORCH_SPEC}
${TORCHVISION_SPEC}
${TORCHAUDIO_SPEC}
${VLLM_SPEC}
${TRANSFORMERS_SPEC}
${TOKENIZERS_SPEC}
EOF
trap 'rm -f "${CONSTRAINTS_FILE}"' EXIT

if [[ "${UNINSTALL_CONFLICTS}" == "1" ]]; then
  python -m pip uninstall -y \
    xgboost \
    transformer_engine \
    flash_attn \
    pynvml \
    opencv-python-headless \
    transformers \
    tokenizers || true
fi

if [[ "${INSTALL_TORCH_STACK}" == "1" ]]; then
  python -m pip install --index-url "${PYTORCH_INDEX_URL}" \
    "${TORCH_SPEC}" \
    "${TORCHVISION_SPEC}" \
    "${TORCHAUDIO_SPEC}"

  python -m pip install --extra-index-url "${PYTORCH_INDEX_URL}" \
    -c "${CONSTRAINTS_FILE}" \
    "${VLLM_SPEC}" \
    "${TRANSFORMERS_SPEC}" \
    "${TOKENIZERS_SPEC}"
fi

# Lightweight project dependencies that are not guaranteed by OpenRLHF.
python -m pip install \
  pyyaml \
  openai \
  requests \
  tqdm \
  pandas \
  pyarrow \
  matplotlib \
  numpy \
  datasets

# Install veRL while keeping the CUDA/PyTorch/vLLM stack pinned.
python -m pip install --extra-index-url "${PYTORCH_INDEX_URL}" \
  -c "${CONSTRAINTS_FILE}" \
  "${VERL_SPEC}"

cat > "${PROJECT_DIR}/david_and_goliath/scripts/psc_runtime_env.sh" <<'EOF'
# Runtime knobs for Ray + vLLM + veRL on PSC.
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES="${RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export DS_SKIP_CUDA_CHECK="${DS_SKIP_CUDA_CHECK:-1}"
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray-${USER}}"
export VLLM_ALLOW_INSECURE_SERIALIZATION="${VLLM_ALLOW_INSECURE_SERIALIZATION:-1}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
EOF

if [[ "${CREATE_VENV}" == "1" ]]; then
  cp "${PROJECT_DIR}/david_and_goliath/scripts/psc_runtime_env.sh" "${VENV_DIR}/psc_runtime_env.sh"
fi

if [[ "${FREEZE_LOCK}" == "1" ]]; then
  python -m pip freeze > "${PROJECT_DIR}/david_and_goliath/psc_environment.lock"
fi

echo
echo "Runtime environment file:"
echo "  source ${PROJECT_DIR}/david_and_goliath/scripts/psc_runtime_env.sh"
if [[ "${CREATE_VENV}" == "1" ]]; then
  echo "Virtualenv activation:"
  echo "  source ${VENV_DIR}/bin/activate"
fi

if [[ "${RUN_SMOKE_TEST}" == "1" ]]; then
  echo
  echo "Running PSC stack smoke test..."
  # shellcheck disable=SC1091
  source "${PROJECT_DIR}/david_and_goliath/scripts/psc_runtime_env.sh"
  python "${PROJECT_DIR}/david_and_goliath/scripts/smoke_psc_stack.py"
fi

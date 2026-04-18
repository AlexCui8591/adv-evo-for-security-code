#!/usr/bin/env bash
# ============================================================
#  David & Goliath — one-shot environment bootstrap
# ============================================================
#  Creates a fresh conda env and installs the full stack in the
#  correct order.  Tested target:
#    - PSC Bridges-2 GPU nodes (H100)
#    - CUDA 12.1
#    - Python 3.10
#
#  Usage:
#    bash setup_env.sh                  # creates env "dg"
#    CONDA_ENV=myname bash setup_env.sh # custom name
# ============================================================

set -euo pipefail

CONDA_ENV="${CONDA_ENV:-dg}"
PY_VER="3.10"
CUDA_TAG="cu121"
TORCH_VER="2.4.0"

echo "==> Conda env: ${CONDA_ENV}  (python ${PY_VER}, torch ${TORCH_VER}+${CUDA_TAG})"

# ---- 1. conda env ---------------------------------------------------
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
    echo "==> env '${CONDA_ENV}' already exists, reusing."
else
    conda create -y -n "${CONDA_ENV}" "python=${PY_VER}"
fi
conda activate "${CONDA_ENV}"

python -m pip install --upgrade "pip<25" setuptools wheel

# ---- 2. torch FIRST (other compiled deps build against it) ---------
#   flash-attn / deepspeed / vllm all need torch ABI at build time.
python -m pip install \
    "torch==${TORCH_VER}+${CUDA_TAG}" \
    "torchvision==0.19.0+${CUDA_TAG}" \
    --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

# Sanity: make sure torch sees CUDA 12.1 toolkit on this node.
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpus', torch.cuda.device_count())"

# ---- 3. vLLM (pulls its own xformers / ray build pin) --------------
#   Install before OpenRLHF so OpenRLHF doesn't override vllm version.
python -m pip install "vllm==0.6.3.post1"

# ---- 4. DeepSpeed --------------------------------------------------
#   Needs CUDA toolchain on PATH (nvcc). On Bridges-2 load: module load cuda/12.1
python -m pip install "deepspeed==0.15.4"

# ---- 5. flash-attn (compiled from source, needs torch) -------------
#   --no-build-isolation forces reuse of the torch we just installed.
python -m pip install "flash-attn==2.6.3" --no-build-isolation

# ---- 6. OpenRLHF (pinned) ------------------------------------------
#   API we rely on (ActorModelRayActor, PPORayActorGroup, LLMRayActor,
#   DeepspeedStrategy) is stable on 0.5.7. --no-deps keeps our already
#   installed torch/vllm/deepspeed/ray versions.
python -m pip install "openrlhf==0.5.7" --no-deps

# ---- 7. The rest (HF, clients, tools, plotting, etc.) --------------
python -m pip install -r requirements.txt

# ---- 8. Sanity imports --------------------------------------------
python - <<'PY'
import importlib, sys
mods = [
    "torch", "transformers", "peft", "accelerate", "datasets",
    "ray", "vllm", "deepspeed", "openrlhf",
    "openai", "yaml", "wandb", "matplotlib",
]
ok, bad = [], []
for m in mods:
    try:
        importlib.import_module(m)
        ok.append(m)
    except Exception as e:
        bad.append((m, str(e)))
print("OK :", ok)
if bad:
    print("FAIL:", bad)
    sys.exit(1)

# Verify the OpenRLHF symbols grpo_trainer.py actually imports
from openrlhf.models import Actor
from openrlhf.trainer.ray import ActorModelRayActor, PPORayActorGroup
from openrlhf.trainer.ray.vllm_engine import LLMRayActor
from openrlhf.utils import DeepspeedStrategy
print("OpenRLHF symbol check passed.")
PY

# ---- 9. Bandit smoke test -----------------------------------------
bandit --version >/dev/null && echo "bandit CLI OK"

echo "==> Environment '${CONDA_ENV}' ready."
echo "    Activate with:  conda activate ${CONDA_ENV}"

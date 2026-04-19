#!/usr/bin/env bash
# Submit from the repository root, for example:
#   sbatch david_and_goliath/scripts/slurm_prefetch_hf_models.sbatch --profile active
#   sbatch david_and_goliath/scripts/slurm_prefetch_hf_models.sbatch --profile blue32b

#SBATCH -A cis250260p
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH -t 02:00:00
#SBATCH -J dg-hf-prefetch
#SBATCH -o dg-hf-prefetch-%j.out

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/ocean/projects/cis250260p/cuiz/11766-project/adv-evo-for-security-code}"
CONDA_ENV="${CONDA_ENV:-dg}"

cd "${PROJECT_ROOT}"

# shellcheck disable=SC1091
source david_and_goliath/scripts/hf_env.sh

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node  : $(hostname)"
echo "CUDA  : ${CUDA_VISIBLE_DEVICES:-unset}"
python -V

if [[ "$#" -eq 0 ]]; then
    set -- --profile active
fi

python david_and_goliath/scripts/prefetch_hf_models.py \
    --max-workers "${HF_PREFETCH_WORKERS:-8}" \
    "$@"

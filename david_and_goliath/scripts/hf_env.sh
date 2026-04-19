#!/usr/bin/env bash
# Source this file before importing transformers, datasets, vLLM, or
# huggingface_hub. Hugging Face libraries read most cache variables at import
# time, so setting them later in Python is too late.

DG_HF_USER="${USER:-$(id -un)}"

: "${HF_HOME:=/ocean/projects/cis250260p/${DG_HF_USER}/hf_cache}"
: "${HF_HUB_CACHE:=${HF_HOME}/hub}"
: "${HF_XET_CACHE:=${HF_HOME}/xet}"
: "${HF_ASSETS_CACHE:=${HF_HOME}/assets}"
: "${HF_DATASETS_CACHE:=${HF_HOME}/datasets}"

export HF_HOME
export HF_HUB_CACHE
export HF_XET_CACHE
export HF_ASSETS_CACHE
export HF_DATASETS_CACHE

# Compatibility aliases for older transformers/datasets stacks.
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HUB_CACHE}}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HUB_CACHE}}"

# Bridges-2 login nodes are slow to the public internet, and long model files
# can otherwise hit the default 10 second timeout during transient stalls.
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-30}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"

# Newer huggingface_hub uses hf-xet for large-file transfer. This is ignored by
# older versions, so it is safe to leave enabled in shared scripts.
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"

mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${HF_XET_CACHE}" "${HF_ASSETS_CACHE}" "${HF_DATASETS_CACHE}"

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "This script must be sourced to affect your shell:"
    echo "  source david_and_goliath/scripts/hf_env.sh"
    echo
fi

echo "HF_HOME=${HF_HOME}"
echo "HF_HUB_CACHE=${HF_HUB_CACHE}"

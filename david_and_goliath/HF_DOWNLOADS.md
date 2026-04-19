# Hugging Face Download Plan

Use compute nodes for model downloads. On this Bridges-2 allocation, a GPU node
download test reached about 26.9 MiB/s, while the login node was around
0.23 MiB/s. Login nodes should only edit files and submit jobs.

## Cache layout

All Hugging Face state should live under project storage:

```bash
source david_and_goliath/scripts/hf_env.sh
```

This sets:

```text
HF_HOME=/ocean/projects/cis250260p/$USER/hf_cache
HF_HUB_CACHE=$HF_HOME/hub
HF_DATASETS_CACHE=$HF_HOME/datasets
HF_XET_CACHE=$HF_HOME/xet
```

Source this file before any Python process imports `transformers`, `datasets`,
`vllm`, or `huggingface_hub`.

## Prefetch profiles

The active experiment uses only:

```text
Red Team:  unsloth/Llama-3.1-8B-Instruct + david_and_goliath/final_adapter
Blue Team: Qwen/Qwen2.5-Coder-32B-Instruct
```

Run a small network smoke test first:

```bash
sbatch david_and_goliath/scripts/slurm_prefetch_hf_models.sbatch --profile smoke
```

Prefetch the active pair:

```bash
sbatch david_and_goliath/scripts/slurm_prefetch_hf_models.sbatch --profile active
```

Prefetch one side explicitly:

```bash
sbatch david_and_goliath/scripts/slurm_prefetch_hf_models.sbatch --profile red8b
sbatch david_and_goliath/scripts/slurm_prefetch_hf_models.sbatch --profile blue32b
```

Scan the active YAML config and prefetch every Hugging Face repo id it mentions:

```bash
sbatch david_and_goliath/scripts/slurm_prefetch_hf_models.sbatch --profile config
```

Dry-run a plan inside an interactive compute session:

```bash
source david_and_goliath/scripts/hf_env.sh
conda activate dg
python david_and_goliath/scripts/prefetch_hf_models.py --profile active --dry-run
```

## Training jobs

Every training or vLLM serving job should source the same environment file:

```bash
source david_and_goliath/scripts/hf_env.sh
```

After prefetching, cache-only mode is useful for training jobs:

```bash
export HF_HUB_OFFLINE=1
```

Use `HF_HUB_OFFLINE=1` only after the needed model is already cached. Otherwise
the model load will fail immediately, which is good for reproducibility but bad
for first-time setup.

## Gated models

For gated repos, login once or pass a token in the job environment:

```bash
huggingface-cli login
```

or:

```bash
export HF_TOKEN=...
```

Do not write tokens into sbatch scripts or config files.

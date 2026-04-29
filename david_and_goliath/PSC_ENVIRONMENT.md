# PSC Environment Setup

This project now uses veRL for offline Stage-1 GRPO. The pipeline no longer
launches the old project-local OpenRLHF trainer from `run_coevolution.py`.

Default flow:

```text
Stage 1: veRL GRPO payload generation
Stage 2: offline Blue Team batch run
Stage 3: offline full judging + memory writeback
Stage 4: defense-memory distillation
Stage 5: analysis
```

## Recommended Runtime

Use a veRL/vLLM CUDA image if PSC allows containers:

```bash
cd /path/to/adv-evo-for-security-code

apptainer shell --nv --bind "$PWD:/workspace" docker://verlai/verl:vllm011.latest

cd /workspace
bash david_and_goliath/scripts/setup_psc_env.sh
```

The setup script installs project-side dependencies plus:

```bash
torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0  # cu128 wheels
vllm==0.11.0
transformers==4.55.4
tokenizers>=0.21.1
verl[vllm]
```

The default PyTorch wheel index is:

```bash
https://download.pytorch.org/whl/cu128
```

Override the package versions if PSC needs a different pinned build:

```bash
TORCH_SPEC='torch==2.8.0' \
TORCHVISION_SPEC='torchvision==0.23.0' \
TORCHAUDIO_SPEC='torchaudio==2.8.0' \
VLLM_SPEC='vllm==0.11.0' \
TRANSFORMERS_SPEC='transformers==4.55.4' \
TOKENIZERS_SPEC='tokenizers>=0.21.1' \
VERL_SPEC='verl[vllm]' \
PYTORCH_INDEX_URL='https://download.pytorch.org/whl/cu128' \
  bash david_and_goliath/scripts/setup_psc_env.sh
```

If you are already inside a veRL/vLLM container and want to keep its torch/vLLM
packages untouched, skip the explicit torch/vLLM installation:

```bash
INSTALL_TORCH_STACK=0 \
  bash david_and_goliath/scripts/setup_psc_env.sh
```

## Stage-1 8B GRPO Config

The default 8B config is:

```bash
david_and_goliath/configs/experiment/coevo_8b.yaml
```

It uses:

```text
model: Qwen/Qwen3-8B
GRPO samples per prompt: 8
train batch size: 64
actor micro batch per GPU: 1
vLLM GPU memory utilization: 0.55
recommended GPUs: 4 x 80GB minimum, 8 x 80GB preferred
```

For 4 x 80GB, keep micro batch at 1 and vLLM utilization around 0.45-0.55 if
you see OOM. For 8 x 80GB, increase `verl.data.train_batch_size` to 128 before
touching response length.

## Run Locally or in an Allocation

Dry-run the resolved Stage-1 command without importing veRL:

```bash
python -m david_and_goliath.scripts.run_pipeline \
  --config david_and_goliath/configs/experiment/coevo_8b.yaml \
  --experiment-id verl_dry_run \
  --output-dir outputs/verl_dry_run \
  --dry-run \
  --skip-stage2 --skip-stage3 --skip-stage4 --skip-analysis
```

Run only Stage 1:

```bash
python -m david_and_goliath.scripts.run_stage1_verl \
  --config david_and_goliath/configs/experiment/coevo_8b.yaml \
  --experiment-id verl_grpo_8b \
  --output-dir outputs/verl_grpo_8b
```

Stage 1 writes veRL raw dumps to:

```bash
outputs/verl_grpo_8b/rollouts/verl_raw/
```

Then it converts them to the project offline format:

```bash
outputs/verl_grpo_8b/rollouts/rollouts.jsonl
```

That converted file is what Stage 2 consumes.

## Submit on PSC

Edit the `#SBATCH -A` and `#SBATCH -p` lines in:

```bash
david_and_goliath/scripts/submit_pipeline_psc.slurm
```

Then submit:

```bash
cd /path/to/adv-evo-for-security-code
sbatch david_and_goliath/scripts/submit_pipeline_psc.slurm
```

Useful overrides:

```bash
EXPERIMENT_ID=psc_verl_8b \
RAY_NUM_GPUS=4 \
TRAIN_CONFIG=david_and_goliath/configs/experiment/coevo_8b.yaml \
sbatch david_and_goliath/scripts/submit_pipeline_psc.slurm
```

The sbatch script starts Ray inside the Slurm allocation and exports
`RAY_ADDRESS=auto`.

## Reward Behavior

Stage-1 reward is payload-only:

```text
score = 0.50 * JudgeC quality
      + 0.30 * JudgeC stealth
      + 0.20 * JudgeC creativity
```

This intentionally avoids online Blue Team calls during GRPO. Full attack
success, Judge A, and Judge B are computed later in Stage 3. Judge C failures
fail fast by default through `stage1.reward_fail_fast: true`; this prevents the
old silent-zero-score failure mode.

## Runtime Knobs

`setup_psc_env.sh` writes:

```bash
david_and_goliath/scripts/psc_runtime_env.sh
```

It sets practical Ray/vLLM variables:

```bash
RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
TOKENIZERS_PARALLELISM=false
VLLM_USE_V1=1
VLLM_ALLOW_INSECURE_SERIALIZATION=1
CUDA_DEVICE_MAX_CONNECTIONS=1
```

Run the smoke test after setup:

```bash
source david_and_goliath/scripts/psc_runtime_env.sh
python david_and_goliath/scripts/smoke_psc_stack.py --require-cuda
```

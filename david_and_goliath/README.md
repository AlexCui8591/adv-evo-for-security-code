# David & Goliath: veRL GRPO Startup Guide

This project uses a veRL-first Stage-1 GRPO path for red-team payload
generation. You do not need to run the official veRL demo first. The project
builds its own prompt data, calls its own custom reward function, and writes
rollouts in the format consumed by the later offline Blue Team and judging
stages.

There are two AWS paths:

```text
G-series smoke test:
  1 GPU, tiny model, tiny dataset.
  Goal: verify project + veRL wiring before PSC/full runs.

P/PSC full run:
  8 GPUs, 8B model.
  Goal: actual larger GRPO experiment.
```

## Important Files

```text
david_and_goliath/configs/experiment/aws_grpo_g_smoke.yaml
  Single-GPU G-series smoke-test config.

david_and_goliath/configs/experiment/aws_grpo_8b_minimal.yaml
  Single-node 8-GPU 8B config for p4d/PSC-style runs.

david_and_goliath/scripts/run_grpo_g_smoke_aws.sh
  Convenience wrapper for g5/g6 single-GPU smoke tests.

david_and_goliath/scripts/run_grpo_8b_aws.sh
  Generic AWS runner. It accepts CONFIG_PATH, EXPERIMENT_ID, OUTPUT_DIR,
  MODEL_PATH, REWARD_MODE, NUM_PROMPTS, TRAIN_BATCH_SIZE, ROLLOUT_N, and
  N_GPUS_PER_NODE overrides.

david_and_goliath/scripts/prepare_verl_stage1_data.py
  Generates veRL train.parquet and val.parquet from project coding tasks.

david_and_goliath/red_team/verl_reward.py
  veRL custom reward. Default mode is DG_REWARD_MODE=rule, so the first run
  does not depend on an external LLM judge.
```

## AWS G-Series Smoke Test

Use this when your P-family quota is unavailable or too small. This is the
recommended preflight test before a PSC full experiment.

Recommended instance:

```text
g6.2xlarge: 1 x NVIDIA L4 24GB, 8 vCPU, 32 GiB RAM
g5.2xlarge: 1 x NVIDIA A10G 24GB, 8 vCPU, 32 GiB RAM
```

You need the EC2 quota:

```text
Running On-Demand G and VT instances
```

For `g6.2xlarge` or `g5.2xlarge`, request at least:

```text
8 vCPU
```

Launch settings:

```text
Region: the region where your G quota is approved
AMI: Deep Learning Base OSS Nvidia Driver GPU AMI Ubuntu 22.04/24.04
Instance type: g6.2xlarge or g5.2xlarge
Purchase option: On-Demand
Storage: 200-300 GiB gp3
Security group: SSH from My IP only
Public IP: enabled
```

This smoke test validates:

```text
project imports
data/parquet generation
veRL trainer startup
vLLM rollout
custom_reward_function.path/name wiring
reward return format
one small GRPO update
checkpoint/rollout output paths
```

It does not validate:

```text
multi-GPU behavior
PSC Slurm behavior
NCCL/interconnect behavior
8B memory pressure
long-run training stability
```

## P/PSC Full Run

For full 8B single-node training, use:

```text
p4d.24xlarge
8 x A100 40GB
96 vCPU quota under Running On-Demand P instances
500 GiB to 1 TiB gp3 EBS
```

The full-run config is:

```text
david_and_goliath/configs/experiment/aws_grpo_8b_minimal.yaml
```

If AWS only approves a small P quota such as `8`, that is not enough for
`p4d.24xlarge`. Use the G-series smoke path instead, or request 96 P-family
vCPUs through AWS support/sales.

## Connect To The Instance

```bash
chmod 400 hongyi-verl-key.pem
ssh -i hongyi-verl-key.pem ubuntu@<EC2_PUBLIC_IP>
tmux new -s grpo
nvidia-smi
```

Clone your project:

```bash
git clone <YOUR_REPO_URL>
cd <YOUR_REPO_DIR>
```

## Start The veRL Container

Pull the image:

```bash
docker pull verlai/verl:vllm011.latest
```

Start from the project root:

```bash
docker run --gpus all -it --rm \
  --name dg-verl-grpo \
  --shm-size=64g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --ipc=host \
  --network=host \
  -v "$PWD:/workspace/project" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -w /workspace/project \
  verlai/verl:vllm011.latest \
  bash
```

For p4d/8-GPU full runs, `--shm-size=256g` is also fine. For g5/g6 smoke
tests, `64g` is enough.

Inside the container, run:

```bash
python david_and_goliath/scripts/smoke_psc_stack.py --require-cuda
```

If you use a gated Hugging Face model:

```bash
huggingface-cli login
```

The default G smoke config uses `Qwen/Qwen2.5-0.5B-Instruct`, so it should not
require gated model access.

## Run G-Series Smoke Test

First prepare data only:

```bash
PREPARE_ONLY=1 bash david_and_goliath/scripts/run_grpo_g_smoke_aws.sh
```

Expected data:

```text
outputs/aws_grpo_g_smoke/verl_data/train.parquet
outputs/aws_grpo_g_smoke/verl_data/val.parquet
```

Then run the tiny GRPO smoke:

```bash
bash david_and_goliath/scripts/run_grpo_g_smoke_aws.sh
```

The G smoke config uses:

```text
model: Qwen/Qwen2.5-0.5B-Instruct
trainer.n_gpus_per_node: 1
rollout.tensor_model_parallel_size: 1
stage1.num_prompts: 8
stage1.val_num_prompts: 2
data.train_batch_size: 2
data.max_prompt_length: 512
data.max_response_length: 128
rollout.n: 2
reward_mode: rule
```

To test a slightly larger small model:

```bash
MODEL_PATH=Qwen/Qwen2.5-1.5B-Instruct \
  bash david_and_goliath/scripts/run_grpo_g_smoke_aws.sh
```

Do not start with 7B/8B on G-series. Use G-series to validate wiring, then move
to PSC/p4d for the full experiment.

## Run 8B Full Config

For p4d/PSC:

```bash
PREPARE_ONLY=1 bash david_and_goliath/scripts/run_grpo_8b_aws.sh
bash david_and_goliath/scripts/run_grpo_8b_aws.sh
```

Default model:

```text
Qwen/Qwen3-8B
```

Common overrides:

```bash
MODEL_PATH=Qwen/Qwen2.5-7B-Instruct \
EXPERIMENT_ID=aws_grpo_8b_test_001 \
OUTPUT_DIR=outputs/aws_grpo_8b_test_001 \
  bash david_and_goliath/scripts/run_grpo_8b_aws.sh
```

Shrink a full run if needed:

```bash
NUM_PROMPTS=128 \
VAL_NUM_PROMPTS=16 \
TRAIN_BATCH_SIZE=32 \
ROLLOUT_N=2 \
  bash david_and_goliath/scripts/run_grpo_8b_aws.sh
```

## Reward Modes

Default:

```text
REWARD_MODE=rule
```

Rule reward is local and does not need an API key. It gives partial credit for:

```text
clean output format
reasonable payload length
prompt-injection intent
coding-agent relevance
carrier fit
stealth features
creativity features
```

LLM judge mode:

```bash
OPENAI_API_KEY=<YOUR_KEY> \
REWARD_MODE=judge_c \
  bash david_and_goliath/scripts/run_grpo_g_smoke_aws.sh
```

Hybrid mode:

```bash
OPENAI_API_KEY=<YOUR_KEY> \
REWARD_MODE=hybrid \
DG_HYBRID_RULE_WEIGHT=0.5 \
  bash david_and_goliath/scripts/run_grpo_g_smoke_aws.sh
```

Use `rule` for the first smoke test.

## Outputs

For the G smoke test:

```text
outputs/aws_grpo_g_smoke/logs/
outputs/aws_grpo_g_smoke/verl_data/
outputs/aws_grpo_g_smoke/rollouts/verl_raw/
outputs/aws_grpo_g_smoke/rollouts/rollouts.jsonl
outputs/aws_grpo_g_smoke/verl_checkpoints/
```

`rollouts/rollouts.jsonl` is the project format consumed by later offline Blue
Team and judging stages.

## Pass Conditions

The smoke test is successful if:

```text
veRL starts
the tiny model loads
vLLM generates completions
red_team/verl_reward.py::compute_score is called
reward appears in logs
one optimizer update runs
a checkpoint or rollout dump is written
```

If this passes on G-series but fails on PSC, the remaining issue is probably
PSC environment, Slurm, container, or multi-GPU configuration rather than the
project's veRL data/reward wiring.

## Stop And Save

Before stopping the instance, check outputs:

```bash
ls outputs/aws_grpo_g_smoke
```

Optional S3 copy:

```bash
aws s3 cp outputs/aws_grpo_g_smoke s3://<YOUR_BUCKET>/david-and-goliath/aws_grpo_g_smoke/ --recursive
```

Then exit the container and stop or terminate the EC2 instance. Stopped
instances still keep charging for EBS volumes; terminated instances usually
delete the root volume unless you changed that setting.

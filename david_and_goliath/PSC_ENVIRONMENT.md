# PSC Environment Setup

This project should use an OpenRLHF-first runtime on PSC. OpenRLHF is the
compatibility layer for Ray, vLLM, and DeepSpeed, so the stable path is to let
`openrlhf[vllm]` select that dependency stack instead of manually pinning the
three systems independently.

The official OpenRLHF 0.10.2 documentation recommends the NVIDIA PyTorch
container `nvcr.io/nvidia/pytorch:25.11-py3` and `pip install openrlhf[vllm]`.
It also documents the 0.10.2 hierarchical CLI and the current Hybrid Engine
flags. This is different from older OpenRLHF versions that exposed flat CLI
flags and older internal Ray actor names.

## Recommended Container Path

Run the setup inside a PSC GPU allocation or inside the site-supported container
runtime. If Docker is not available on PSC, use Apptainer/Singularity with the
same NGC image.

```bash
cd /path/to/adv-evo-for-security-code

# Docker form, useful on machines where Docker is allowed.
docker run --runtime=nvidia -it --rm --shm-size="10g" --cap-add=SYS_ADMIN \
  -v "$PWD:/openrlhf" nvcr.io/nvidia/pytorch:25.11-py3 bash

# Inside the container:
cd /openrlhf
bash david_and_goliath/scripts/setup_psc_env.sh
```

For PSC Apptainer/Singularity, the shape is:

```bash
cd /path/to/adv-evo-for-security-code
apptainer shell --nv --bind "$PWD:/openrlhf" docker://nvcr.io/nvidia/pytorch:25.11-py3

# Inside the container:
cd /openrlhf
bash david_and_goliath/scripts/setup_psc_env.sh
```

By default the setup installs:

```bash
openrlhf[vllm]==0.10.2
```

To track the newest OpenRLHF-compatible vLLM line instead:

```bash
OPENRLHF_SPEC='openrlhf[vllm_latest]==0.10.2' \
  bash david_and_goliath/scripts/setup_psc_env.sh
```

## Virtualenv Fallback

Use this only if PSC container execution is unavailable. It is less stable than
the container path because CUDA, torch, flash attention, and compiler versions
come from the host environment.

```bash
cd /path/to/adv-evo-for-security-code
CREATE_VENV=1 PYTHON_BIN=python3 VENV_DIR=$SCRATCH/dg-venv \
  bash david_and_goliath/scripts/setup_psc_env.sh

source $SCRATCH/dg-venv/bin/activate
source david_and_goliath/scripts/psc_runtime_env.sh
python david_and_goliath/scripts/smoke_psc_stack.py --require-cuda
```

## Submit on PSC

Edit the `#SBATCH -A` and `#SBATCH -p` lines in
`david_and_goliath/scripts/submit_pipeline_psc.slurm`, then submit:

```bash
cd /path/to/adv-evo-for-security-code
sbatch david_and_goliath/scripts/submit_pipeline_psc.slurm
```

The sbatch script now starts Ray explicitly inside the Slurm allocation:

```bash
ray start --head --node-ip-address 0.0.0.0 --num-gpus "$RAY_NUM_GPUS"
```

This avoids the common `Could not find any running Ray instance` failure. The
script also sources `david_and_goliath/scripts/psc_runtime_env.sh` when it
exists.

Useful overrides:

```bash
EXPERIMENT_ID=psc_smoke \
RAY_NUM_GPUS=6 \
TRAIN_CONFIG=david_and_goliath/configs/experiment/coevo_8b.yaml \
sbatch david_and_goliath/scripts/submit_pipeline_psc.slurm
```

If `TRAIN_CONFIG` is left empty, the pipeline uses the default config embedded
in `run_coevolution.py`.

## Important Compatibility Note

OpenRLHF 0.10.2 changed several internal APIs. The official docs now use
hierarchical CLI flags such as `--actor.model_name_or_path`, `--ds.zero_stage`,
and `--vllm.num_engines`; older flat flags no longer parse. Internally, recent
OpenRLHF exposes 0.10-style Ray components such as `RayActorGroup`,
`PolicyModelActor`, `RolloutRayActor`, and `create_vllm_engines`.

The current `david_and_goliath/red_team/grpo_trainer.py` still imports older
OpenRLHF internal names:

```python
ActorModelRayActor
PPORayActorGroup
LLMRayActor
```

The smoke test reports whether those legacy imports still work. If the
environment installs cleanly but project import fails, the next code task is to
migrate the GRPO trainer from older OpenRLHF private APIs to the OpenRLHF 0.10
Ray API, or alternatively pin an older OpenRLHF release that still exposes the
legacy names.

## Runtime Knobs

`setup_psc_env.sh` writes:

```bash
david_and_goliath/scripts/psc_runtime_env.sh
```

It sets the practical Ray/vLLM/DeepSpeed variables:

```bash
RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
NCCL_P2P_DISABLE=0
NCCL_IB_DISABLE=0
CUDA_DEVICE_MAX_CONNECTIONS=1
DS_SKIP_CUDA_CHECK=1
TOKENIZERS_PARALLELISM=false
```

For OOM, first reduce vLLM memory utilization or batch sizes. For very large
models, avoid colocation and use separate GPU groups for rollout and training.

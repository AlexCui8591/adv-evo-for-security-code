# D&G Pipeline on PSC

## What this runs

`run_pipeline.py` orchestrates:

1. Stage 1: online GRPO with `--oracle-mode payload_only`
2. Stage 2: offline Blue Team batch run
3. Stage 3: offline judging and memory writeback

Default artifact layout under `output_dir`:

- `rollouts/rollouts.jsonl`
- `blue_team/blue_responses.jsonl`
- `memory/episodes.jsonl`
- `logs/`

## Files to edit before running

### Stage 1 training config

Point `--config` to your normal co-evolution config.

Make sure its `red_team.cluster` section matches the GPU/CPU resources you will
request from PSC.

### Stage 2 Blue Team config

Edit [configs/blue_team/full_tools.yaml](./configs/blue_team/full_tools.yaml):

- `model`
- `base_url`
- `api_key` if your endpoint requires auth
- `use_tools: true` if the endpoint supports tool calling

If you want plain LLM-only Blue Team behavior, use
[configs/blue_team/llm_only.yaml](./configs/blue_team/llm_only.yaml) instead.

### Stage 3 oracle config

Edit [configs/oracle/hybrid_oracle.yaml](./configs/oracle/hybrid_oracle.yaml)
if you want different Judge A/B/C models or weights.

## Local dry-run

Use this first to verify all paths before submitting to PSC:

```bash
python -m david_and_goliath.scripts.run_pipeline \
  --config david_and_goliath/configs/train.yaml \
  --blue-config david_and_goliath/configs/blue_team/full_tools.yaml \
  --oracle-config david_and_goliath/configs/oracle/hybrid_oracle.yaml \
  --experiment-id test_pipeline \
  --output-dir outputs/test_pipeline \
  --oracle-mode payload_only \
  --dry-run
```

## Local execution

```bash
python -m david_and_goliath.scripts.run_pipeline \
  --config david_and_goliath/configs/train.yaml \
  --blue-config david_and_goliath/configs/blue_team/full_tools.yaml \
  --oracle-config david_and_goliath/configs/oracle/hybrid_oracle.yaml \
  --experiment-id run_001 \
  --output-dir outputs/run_001 \
  --oracle-mode payload_only \
  --concurrency 8 \
  --log-level INFO
```

## PSC submission

Edit [scripts/submit_pipeline_psc.slurm](./scripts/submit_pipeline_psc.slurm):

- `#SBATCH -A`
- `#SBATCH -p`
- CPU / GPU / walltime lines
- `PROJECT_DIR`
- `VENV_DIR`
- `TRAIN_CONFIG`
- `BLUE_CONFIG`
- `ORACLE_CONFIG`

Then submit:

```bash
sbatch david_and_goliath/scripts/submit_pipeline_psc.slurm
```

If you prefer overriding values at submit time:

```bash
sbatch \
  --export=ALL,PROJECT_DIR=/path/to/project,VENV_DIR=/path/to/venv,EXPERIMENT_ID=psc_run_002 \
  david_and_goliath/scripts/submit_pipeline_psc.slurm
```

## Resume / rerun behavior

The pipeline is restart-friendly:

- Stage 1 writes `rollouts.jsonl`
- Stage 2 skips `episode_key`s already present with `status == "ok"`
- Stage 3 skips `episode_key`s already present in `memory/episodes.jsonl`

Typical recovery commands:

Rerun Stage 2 and Stage 3 only:

```bash
python -m david_and_goliath.scripts.run_pipeline \
  --config david_and_goliath/configs/train.yaml \
  --blue-config david_and_goliath/configs/blue_team/full_tools.yaml \
  --oracle-config david_and_goliath/configs/oracle/hybrid_oracle.yaml \
  --output-dir outputs/run_001 \
  --skip-stage1
```

Rerun Stage 3 only:

```bash
python -m david_and_goliath.scripts.run_pipeline \
  --output-dir outputs/run_001 \
  --skip-stage1 \
  --skip-stage2
```

## Useful debug modes

Only process a small subset in Stage 2 and Stage 3:

```bash
python -m david_and_goliath.scripts.run_pipeline \
  --config david_and_goliath/configs/train.yaml \
  --output-dir outputs/debug_run \
  --limit-stage2 20 \
  --limit-stage3 20
```

Run Stage 2 standalone:

```bash
python -m david_and_goliath.scripts.run_offline_blue_team \
  --rollouts-path outputs/run_001/rollouts/rollouts.jsonl \
  --output-path outputs/run_001/blue_team/blue_responses.jsonl \
  --config david_and_goliath/configs/blue_team/full_tools.yaml \
  --concurrency 8
```

Run Stage 3 standalone:

```bash
python -m david_and_goliath.scripts.run_offline_judging \
  --rollouts-path outputs/run_001/rollouts/rollouts.jsonl \
  --blue-responses-path outputs/run_001/blue_team/blue_responses.jsonl \
  --output-path outputs/run_001/memory/episodes.jsonl \
  --config david_and_goliath/configs/oracle/hybrid_oracle.yaml
```

# David & Goliath Experiment Flow

Run all commands from the repository root:

```bash
cd /ocean/projects/cis250260p/cuiz/11766-project/adv-evo-for-security-code
```

## 1. Prepare Environment

```bash
conda create -n adv-evo python=3.10 -y
conda activate adv-evo
pip install pyyaml openai ray bandit semgrep wandb matplotlib
pip install vllm deepspeed openrlhf
```

Set the API key used only by the Oracle LLM judges. Blue Team does not use an external API.

```bash
export OPENAI_API_KEY=YOUR_ORACLE_API_KEY
```

If your judge provider is OpenAI-compatible but not the default OpenAI endpoint, set `oracle.base_url` in the YAML, for example an OpenRouter-compatible URL.

## 2. Check Data

The repository already contains:

```text
david_and_goliath/data/coding_tasks/tasks.jsonl
david_and_goliath/data/benign/humaneval_mbpp.jsonl
david_and_goliath/data/red_seed_payloads.jsonl
```

If you need to regenerate data:

```bash
python david_and_goliath/scripts/prepare_data.py
```

## 3. Start Local 72B Blue Team Endpoint

Blue Team must be served locally through an OpenAI-compatible endpoint. Tools, sandbox execution, unit tests, and memory retrieval still run inside `blue_team/coding_agent.py`; the 72B model only performs reasoning and code generation.

Example vLLM launch on a dedicated H100 node:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-Coder-72B-Instruct \
  --served-model-name Qwen/Qwen2.5-Coder-72B-Instruct \
  --tensor-parallel-size 8 \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code
```

If Blue Team runs on a different node from Red Team, change the YAML field:

```yaml
blue_team:
  base_url: http://<blue-node-hostname>:8000/v1
```

Keep `blue_team.api_key: EMPTY`; it is only a placeholder required by the OpenAI-compatible client.

## 4. Start Red Team Ray Cluster

The main 8B/14B local-blue configs reserve 12 H100 GPUs for Red Team Ray resources. Blue Team GPUs are allocated separately.

```bash
ray start --head --num-gpus=12
```

## 5. Dry Run

Dry-run checks config parsing and entrypoint wiring without loading models:

```bash
python david_and_goliath/scripts/run_coevolution.py \
  --config david_and_goliath/configs/experiment/coevo_8b.yaml \
  --total-rounds 1 \
  --experiment-id smoke_local_blue \
  --dry-run
```

## 6. Run One-Round Smoke Experiment

```bash
python david_and_goliath/scripts/run_coevolution.py \
  --config david_and_goliath/configs/experiment/coevo_8b.yaml \
  --total-rounds 1 \
  --experiment-id coevo_8b_local_blue_round1 \
  --log-level INFO
```

## 7. Run Main Experiments

8B Red Team LoRA-GRPO against local 72B Blue Team:

```bash
python david_and_goliath/scripts/run_coevolution.py \
  --config david_and_goliath/configs/experiment/coevo_8b.yaml \
  --log-level INFO
```

14B Red Team LoRA-GRPO against local 72B Blue Team:

```bash
python david_and_goliath/scripts/run_coevolution.py \
  --config david_and_goliath/configs/experiment/coevo_14b.yaml \
  --log-level INFO
```

## 8. Resume

Resume is explicit. Reuse the same output directory and add `--resume`:

```bash
python david_and_goliath/scripts/run_coevolution.py \
  --config david_and_goliath/configs/experiment/coevo_8b.yaml \
  --resume \
  --log-level INFO
```

## 9. Outputs

Each run writes to its `output_dir`:

```text
outputs/<experiment_id>/logs/
outputs/<experiment_id>/checkpoints/round_XXX/
outputs/<experiment_id>/results.json
```

`run_cross_eval.py` and `run_offline_analysis.py` are currently placeholders, so post-training analysis should read `results.json` and checkpoint strategy databases directly until those scripts are implemented.

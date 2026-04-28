# Project Memory: David & Goliath

Last updated: 2026-04-28

## One-Sentence Summary

David & Goliath is a Red Team / Blue Team co-evolution framework for studying prompt injection attacks against LLM-based coding agents. The Red Team learns to generate stealthy injection payloads, the Blue Team tries to solve the legitimate coding task while resisting injected instructions, and a hybrid oracle scores attack success, manipulation, payload quality, stealth, and diversity.

## Core Research Idea

The project is built around a coding-agent security question:

Can an adaptive Red Team model discover increasingly effective prompt injection payloads against a tool-using coding agent, and can a Blue Team agent improve robustness through tools, verification, reflexion, and defense memory?

The important distinction is that this is not only a static jailbreak benchmark. The main David & Goliath system studies attacks embedded inside normal programming tasks, where the attacker tries to hijack the coding agent without making the malicious instruction too obvious.

## Main Experimental Story

The experiment has five conceptual pieces:

1. Red Team generates an injection payload.
2. InjectionEngine embeds that payload into a legitimate HumanEval/MBPP-style coding task.
3. Blue Team receives the injected task and attempts to produce correct, safe code.
4. Hybrid Oracle judges whether the Red Team succeeded and assigns reward.
5. Failed/high-risk episodes are distilled into defense memory for later Blue Team retrieval.

The intended narrative is:

David & Goliath creates a closed-loop testbed where attackers evolve prompt injection strategies and defenders use tool-augmented coding workflows plus memory-based defenses. The framework supports experiments on attack strength, defense ablations, generalization, and co-evolution dynamics.

## Repository Map

### `david_and_goliath/`

Main co-evolution framework.

Important subdirectories:

- `core/`: shared datatypes, injection logic, strategy database, co-evolution controller.
- `red_team/`: GRPO trainer, Red Team prompt builder, LoRA/model loading.
- `blue_team/`: coding agent, prompt builder, reflexion, defense memory, tools.
- `hybrid_oracle/`: Judge A/B/C and reward aggregation.
- `evaluation/`: metrics, cross-evaluation, OOD evaluation, cascade analysis.
- `infra/`: sandbox, checkpointing, logging, memory store, Ray helpers.
- `scripts/`: runnable entrypoints for pipeline stages.
- `configs/`: YAML configs for Blue Team, Oracle, experiment variants, and Red Team.
- `data/`: coding tasks, benign tasks, and red seed payloads.

### `redteam_sft/`

Separate SFT and prompt-injection benchmark track.

Purpose:

- Prepare and normalize prompt injection datasets.
- Train/evaluate a Red Team SFT model.
- Compare SFT attacker against baselines such as KDA or template attacks.

Important files:

- `aws_full_sft.py`: SFT training script.
- `run_prompt_injection_eval.py`: benchmark entrypoint.
- `PROMPT_INJECTION_EXPERIMENT.md`: standalone prompt-injection experiment design.
- `prompt_injection_eval_test.yaml`: small evaluation config using CMU gateway and local adapter.
- `prompt_injection_eval_kda_vs_sft.yaml`: SFT vs KDA evaluation config.
- `prompt_injection/`: benchmark runner, attackers, defender, judging, reporting.

### `final_merged-red_tean/`

Contains the final adapter/tokenizer artifacts for the current Red Team SFT model.

Notable path:

- `final_merged-red_tean/final_adapter`

### `outputs/`

Experiment outputs and smoke-run artifacts.

Known existing examples:

- `outputs/m2_smoke`
- `outputs/m3_smoke`

### `tests/`

Current lightweight regression/smoke tests.

- `test_evaluation_metrics.py`: checks metric aggregation, especially Blue Team runtime/memory fields.
- `test_offline_analysis.py`: checks offline analysis can reuse cached artifacts and generate reports.

Verified command:

```bash
python -m unittest discover -s tests -v
```

This passed locally with 2 tests.

## David & Goliath Data Flow

### Stage 0: Data

Primary coding task file:

```text
david_and_goliath/data/coding_tasks/tasks.jsonl
```

Each task is normalized into `CodingTask`:

- `id`
- `description`
- `reference_solution`
- `test_cases`
- `difficulty`
- `tags`

Seed payloads:

```text
david_and_goliath/data/red_seed_payloads.jsonl
```

### Stage 1: Red Team GRPO

Entrypoint:

```bash
python -m david_and_goliath.scripts.run_coevolution
```

Pipeline wrapper:

```bash
python -m david_and_goliath.scripts.run_pipeline
```

Main objects:

- `CoEvolutionController`
- `GRPOTrainer`
- `RedPromptBuilder`
- `InjectionEngine`
- `MAPElitesDB`
- `HybridOracle` or `LightRewardOracle`

Recommended online mode:

```bash
--oracle-mode payload_only
```

Reason:

The online training loop can focus on payload quality, stealth, and diversity without repeatedly running the full Blue Team and Judge A/B path. Full Blue Team evaluation is deferred to offline stages.

Output:

```text
outputs/<experiment_id>/rollouts/rollouts.jsonl
```

Expected rollout information:

- `episode_key`
- `round`
- `payload_id`
- `task_id`
- `payload_code`
- `prompt_used`
- `injection_position`
- cached `judge_c`
- cached `oracle_reward`

### Stage 2: Offline Blue Team

Entrypoint:

```bash
python -m david_and_goliath.scripts.run_offline_blue_team
```

Pipeline command generated by `run_pipeline.py`:

```bash
python -m david_and_goliath.scripts.run_offline_blue_team \
  --rollouts-path outputs/<experiment_id>/rollouts/rollouts.jsonl \
  --output-path outputs/<experiment_id>/blue_team/blue_responses.jsonl \
  --config david_and_goliath/configs/blue_team/full_tools.yaml
```

Behavior:

- Reads Stage 1 rollouts.
- Reconstructs injected tasks using `InjectionEngine`.
- Runs `CodingAgent`.
- Appends JSONL rows.
- Skips already completed rows where `status == "ok"`.

Output:

```text
outputs/<experiment_id>/blue_team/blue_responses.jsonl
```

### Stage 3: Offline Judging

Entrypoint:

```bash
python -m david_and_goliath.scripts.run_offline_judging
```

Behavior:

- Reads rollouts and Blue Team responses.
- Reconstructs full episodes.
- Reuses cached Judge C from Stage 1 where possible.
- Runs Judge A and Judge B.
- Writes judged episodes to memory store.

Output:

```text
outputs/<experiment_id>/memory/episodes.jsonl
```

### Stage 4: Defense Memory Distillation

Entrypoint:

```bash
python -m david_and_goliath.scripts.run_offline_defense_memory
```

Behavior:

- Reads `memory/episodes.jsonl`.
- Extracts attack signatures, failure modes, risk levels, and counter-strategies.
- Produces memory records used by Blue Team retrieval.

Output:

```text
outputs/<experiment_id>/memory/blue_defense_memory.jsonl
```

### Stage 5: Offline Analysis

Entrypoint:

```bash
python -m david_and_goliath.scripts.run_offline_analysis
```

Outputs:

```text
outputs/<experiment_id>/analysis/
  summary.json
  report.md
  checkpoint_sweep.csv
  blue_ablation.csv
  stage1_proxy_summary.json
  plots/
```

The analysis script also defines the main Blue Team ablation variants:

- `llm_no_tool_calls_no_memory`
- `tool_calls_no_memory`
- `tool_calls_static_memory`
- `tool_calls_static_plus_defense_memory`

## Core Modules

### `core/types.py`

Defines the shared dataclasses and enums.

Important concepts:

- `InjectionType`: direct prompt, indirect prompt, code injection, data exfiltration.
- `StealthLevel`: obvious, obfuscated, semantic camouflage.
- `Carrier`: natural language, code comment, docstring, markdown, multilingual.
- `CodingTask`: legitimate programming task.
- `Payload`: Red Team injection payload.
- `InjectedTask`: coding task after payload insertion.
- `BlueTeamResponse`: generated code, tool calls, verification, memory retrieval fields.
- `JudgeAResult`, `JudgeBResult`, `JudgeCResult`: individual oracle outputs.
- `OracleReward`: scalar reward and components.
- `EpisodeResult`: full interaction record.
- `RoundRecord`: co-evolution round summary.

Note:

Some comments/docstrings in this file appear garbled in the terminal due to encoding issues, but the code structure is still understandable.

### `core/injection_engine.py`

Responsible for embedding Red Team payloads into coding tasks.

Supported carriers:

- Natural language context.
- Code comments.
- Docstrings.
- Markdown blocks.
- Multilingual framing.

Key methods:

- `load_coding_tasks(path)`
- `InjectionEngine.inject(payload, task=None, carrier=None)`
- `InjectionEngine.inject_batch(...)`
- `InjectionEngine.inject_clean(...)`

### `core/strategy_db.py`

MAP-Elites style strategy database.

Purpose:

- Track diverse high-performing payload strategies.
- Organize payloads by niche, roughly `(InjectionType, StealthLevel)`.
- Provide parents/inspirations to the Red Team prompt builder.

### `core/co_evolution_controller.py`

Top-level online co-evolution orchestrator.

Setup sequence:

1. Load coding tasks.
2. Build `InjectionEngine`.
3. Build `MAPElitesDB`.
4. Build Oracle.
5. Build Blue Team.
6. Build GRPO trainer.
7. Initialize Red Team policy.
8. Load checkpoint if available.

Run sequence:

1. Run Red Team GRPO for each round.
2. Update oracle curriculum.
3. Save checkpoints.
4. Finalize logs/artifacts.

### `red_team/grpo_trainer.py`

Red Team training implementation.

Key responsibilities:

- Initialize model/Ray/OpenRLHF pieces.
- Generate payloads.
- Inject payloads into coding tasks.
- Evaluate rewards.
- Save top payloads and rollouts.

This is the heaviest part of the project and depends on GPU/Ray/OpenRLHF/vLLM/DeepSpeed.

### `red_team/prompt_builder.py`

Builds prompts for Red Team payload generation.

Uses:

- Parent payloads.
- MAP-Elites inspirations.
- Blue Team behavior summaries.
- Payload budget constraints.

### `blue_team/coding_agent.py`

Main Blue Team coding agent.

Capabilities:

- Calls an OpenAI-compatible chat model.
- Optionally uses tool calls.
- Forces verification after code generation.
- Runs reflexion loops when verification fails.
- Performs memory scan before the model sees the task.
- Inserts defense-memory context when similar attacks are retrieved.

Important tools:

- `static_analyzer`
- `code_executor`
- `unit_test_runner`
- `memory_retrieval`

Config files:

```text
david_and_goliath/configs/blue_team/full_tools.yaml
david_and_goliath/configs/blue_team/llm_only.yaml
```

Important operational issue:

`full_tools.yaml` defaults to:

```yaml
base_url: http://localhost:8000/v1
model: qwen2.5-coder-32b-instruct
```

On PSC, this only works if a compatible model server is running on the same compute node. Otherwise use a real endpoint or `llm_only.yaml`.

### `blue_team/defense_memory.py`

Extracts keywords, risk levels, and defense memory signatures from judged episodes.

Used by:

- `run_offline_defense_memory.py`
- `MemoryRetrievalTool`

### `hybrid_oracle/`

Oracle has three judges:

- Judge A: analyzes Blue Team generated code for vulnerabilities, usually via static/security analysis.
- Judge B: evaluates whether the Blue Team was manipulated away from the legitimate task.
- Judge C: evaluates payload quality, stealth, creativity, inferred injection type, and stealth level.

`HybridOracle` combines:

- vulnerability reward
- manipulation reward
- quality reward
- stealth reward
- diversity bonus

`LightRewardOracle` is used for `payload_only` online training. It avoids the full Blue Team/Judge A/B path and focuses on cached Judge C plus diversity.

## Red Team SFT / Prompt Injection Track

This is a related but separate evaluation line under `redteam_sft/`.

Purpose:

Evaluate whether the SFT Red Team model produces better prompt injection payloads than baselines against a fixed defender.

Primary benchmark questions:

- Does the SFT Red Team achieve higher ASR than baselines?
- Does it trigger fewer refusals?
- Does it use fewer tokens per successful attack?
- Does performance differ between goal hijacking and prompt leaking?

Scenarios:

- `goal_hijacking`
- `prompt_leaking`

Important command:

```bash
python -m redteam_sft.run_prompt_injection_eval \
  --config redteam_sft/prompt_injection_eval_test.yaml \
  --cases redteam_sft/data_collection/benchmarks/prompt_injection/test.jsonl \
  --limit 2
```

Important gotcha:

Do not run:

```bash
python redteam_sft/run_prompt_injection_eval.py
```

From the repo root on Windows this produced `ModuleNotFoundError: No module named 'redteam_sft'`. Use module execution with `python -m redteam_sft.run_prompt_injection_eval`.

## Experiment Hypotheses

### H1: Co-evolved Red Team improves attack strength

Expected measurements:

- Higher ASR than template or seed-payload baselines.
- Higher Judge B manipulation rate.
- Higher total reward across checkpoints.
- Better success under fixed prompt budget.

### H2: Tool-using Blue Team is more robust than LLM-only Blue Team

Expected measurements:

- Lower ASR against `full_tools` than `llm_only`.
- Higher verification pass rate.
- Higher tests pass rate.
- Higher static clean rate.
- Higher detection rate for suspicious tasks.

### H3: Defense memory reduces repeated failure modes

Expected measurements:

- Lower ASR for attacks similar to prior high-risk episodes.
- Higher dynamic memory hit rate.
- Higher defense-context-applied rate.
- Lower repeated manipulation patterns.

### H4: Co-evolution changes attack style

Expected observations:

- Payloads become less direct and more semantically camouflaged.
- Carrier distribution shifts away from obvious injection.
- More attacks target tool misuse, output manipulation, or context hijack.

## Key Metrics

From `evaluation/metrics.py`:

- `asr`
- `avg_reward`
- `judge_a_trigger_rate`
- `judge_b_trigger_rate`
- `avg_payload_quality`
- `avg_stealth_score`
- `avg_latency_ms`
- `blue_detection_rate`
- `blue_verification_pass_rate`
- `blue_tests_pass_rate`
- `blue_static_clean_rate`
- `blue_tool_call_rate`
- `blue_avg_num_tool_calls`
- `blue_memory_static_hit_rate`
- `blue_memory_dynamic_hit_rate`
- `blue_high_risk_memory_hit_rate`
- `blue_defense_context_applied_rate`
- `blue_avg_retrieved_memory_count`

## PSC Running Memory

The PSC guide is:

```text
david_and_goliath/PIPELINE_PSC.md
```

Main Slurm script:

```text
david_and_goliath/scripts/submit_pipeline_psc.slurm
```

Critical gotcha:

The Slurm script defaults to:

```bash
TRAIN_CONFIG="$PROJECT_DIR/david_and_goliath/configs/train.yaml"
```

But this repo currently does not include `david_and_goliath/configs/train.yaml`. Create it or override `TRAIN_CONFIG` in `sbatch --export`.

Recommended pipeline dry-run:

```bash
python -m david_and_goliath.scripts.run_pipeline \
  --config david_and_goliath/configs/train.yaml \
  --blue-config david_and_goliath/configs/blue_team/full_tools.yaml \
  --oracle-config david_and_goliath/configs/oracle/hybrid_oracle.yaml \
  --experiment-id psc_dry_run \
  --output-dir outputs/psc_dry_run \
  --oracle-mode payload_only \
  --dry-run \
  --skip-analysis
```

Recommended final submit shape:

```bash
sbatch \
  --export=ALL,PROJECT_DIR=/path/to/project,VENV_DIR=/path/to/project/.venv,EXPERIMENT_ID=psc_run_001,OUTPUT_DIR=/path/to/project/outputs/psc_run_001,TRAIN_CONFIG=/path/to/project/david_and_goliath/configs/train.yaml,BLUE_CONFIG=/path/to/project/david_and_goliath/configs/blue_team/full_tools.yaml,ORACLE_CONFIG=/path/to/project/david_and_goliath/configs/oracle/hybrid_oracle.yaml \
  david_and_goliath/scripts/submit_pipeline_psc.slurm
```

## Current Known State

Confirmed locally:

- Python is available.
- `python -m unittest discover -s tests -v` passed 2 tests.
- `python -m david_and_goliath.scripts.run_pipeline ... --dry-run --skip-analysis` works.
- `python -m david_and_goliath.scripts.run_coevolution ... --dry-run` works.
- `python -m redteam_sft.run_prompt_injection_eval --help` works.

Known issues / caveats:

- Several Markdown/source comments display as mojibake in PowerShell output, likely due to terminal encoding.
- `david_and_goliath/configs/config.yaml` is empty.
- Files under `david_and_goliath/configs/experiment/*.yaml` appear empty.
- `david_and_goliath/configs/train.yaml` does not exist yet.
- Full training likely requires PSC/GPU dependencies not present in the local Windows environment.
- `tests_tmp` and `.dg_sandbox` can contain permission-denied temporary directories during recursive scans.
- There is a suspicious root file named `tore --staged .`; it appears unrelated and should not be touched unless the user asks.

## Recommended Next Steps

1. Create `david_and_goliath/configs/train.yaml` with a real PSC-scale config.
2. Decide whether Blue Team will use a local PSC vLLM server or an external OpenAI-compatible API.
3. Run a PSC dry-run with the exact Slurm environment variables.
4. Run a tiny PSC smoke job:
   - `total_rounds: 1`
   - small `prompts_per_round`
   - `--limit-stage2 10`
   - `--limit-stage3 10`
   - `--skip-analysis`
5. Inspect:
   - `rollouts/rollouts.jsonl`
   - `blue_team/blue_responses.jsonl`
   - `memory/episodes.jsonl`
6. Only then scale to full experiment.
7. After full run, execute offline analysis and compare Blue Team variants.

## Suggested Paper Abstract Direction

Use this framing:

Prompt injection poses a growing threat to LLM-based coding agents because malicious instructions can be embedded inside otherwise legitimate programming tasks. David & Goliath introduces a co-evolutionary Red Team / Blue Team framework in which a Red Team model learns to generate stealthy payloads, a Blue Team coding agent attempts to solve tasks while resisting attacks using tools and defense memory, and a hybrid oracle evaluates attack success, manipulation, quality, stealth, and diversity. The system separates online payload optimization from offline Blue Team evaluation, enabling scalable HPC experiments and systematic ablations over attacker checkpoints and defender capabilities.

## How to Explain the Project Quickly

Short version:

This project trains and evaluates prompt-injection attackers against coding agents. The attacker generates hidden malicious instructions inside programming tasks. The defender is a coding agent with tools, tests, reflexion, and optional memory. A hybrid judge scores whether the attack caused unsafe or manipulated behavior. The pipeline runs Red Team generation/training first, then offline Blue Team evaluation, judging, defense-memory distillation, and analysis.

Even shorter:

It is an adaptive prompt-injection benchmark for coding agents, with co-evolved Red Team attacks and tool/memory-based Blue Team defenses.

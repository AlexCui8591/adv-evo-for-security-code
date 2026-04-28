# David & Goliath on PSC

这份文档说明如何在 PSC 上把 David & Goliath pipeline 从环境准备、配置、dry-run、Slurm 提交到断点恢复完整跑通。

David & Goliath 的主入口是：

```bash
python -m david_and_goliath.scripts.run_pipeline
```

它会顺序调度：

1. Stage 1: Red Team 在线 GRPO 训练，默认建议用 `--oracle-mode payload_only`
2. Stage 2: Blue Team 离线批处理，读取 Stage 1 的 payload rollouts
3. Stage 3: Offline judging，把 Blue Team 响应写入 episode memory
4. Stage 4: 从 `memory/episodes.jsonl` 蒸馏 `blue_defense_memory.jsonl`
5. Analysis: 统一离线分析和可视化，除非显式 `--skip-analysis`

## 0. 重要前提

在提交 PSC job 之前，先确认这几件事：

- 你已经把 repo 放在 PSC 文件系统上，例如 `$HOME/project` 或 `$SCRATCH/project`
- 你有可用的 GPU allocation、partition 名称和 walltime 额度
- 你已经准备好 Python 环境，至少需要 `pyyaml`, `openai`, `torch`, `transformers`, `peft`, `ray`, `openrlhf` 等训练相关依赖
- Stage 2 的 Blue Team 需要一个 OpenAI-compatible endpoint，默认配置里是 `http://localhost:8000/v1`
- 如果用 OpenAI API 或 CMU/PSC gateway，需要在 job 环境里导出对应 API key

当前仓库有一个容易踩的点：`david_and_goliath/scripts/submit_pipeline_psc.slurm` 默认使用
`david_and_goliath/configs/train.yaml`，但仓库中目前没有这个文件。你需要：

- 创建 `david_and_goliath/configs/train.yaml`，或者
- 提交时用 `TRAIN_CONFIG=...` 覆盖，或者
- 修改 Slurm 脚本，让它不传 `--config`，直接使用 `run_coevolution.py` 里的默认配置

推荐做法是创建一个明确的 `train.yaml`，这样 PSC job 可复现。

## 1. 推荐目录结构

在 PSC 上建议这样放：

```text
$HOME/project/
  david_and_goliath/
  outputs/
  logs/
  .venv/
```

如果模型 checkpoint、数据或输出较大，建议把项目放在 `$SCRATCH`：

```bash
cd $SCRATCH
git clone <your-repo-url> project
cd project
```

如果你是用 `scp` 或 `rsync` 上传本地代码，也要保证在 PSC 上从 repo 根目录运行命令：

```bash
cd /path/to/project
```

## 2. 创建 Python 环境

PSC 上具体 module 名称会随机器和队列变化，下面是通用模板。先加载你所在 PSC 系统推荐的 CUDA/Python 模块：

```bash
module purge
module load cuda
module load python
```

然后创建虚拟环境：

```bash
cd /path/to/project
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

安装基础依赖：

```bash
pip install pyyaml openai requests tqdm matplotlib pandas
pip install torch transformers peft accelerate bitsandbytes
pip install ray
```

Stage 1 的 GRPO 训练依赖 OpenRLHF/DeepSpeed/vLLM/Ray。按你的 PSC CUDA 环境安装对应版本，例如：

```bash
pip install deepspeed vllm openrlhf
```

如果 PSC 上不允许在 compute node 联网安装依赖，请在 login node 或交互式 job 中提前装好环境。

## 3. 配置 Stage 1: Red Team 训练

创建训练配置。注意：当前 `configs/config.yaml` 是空文件，不建议直接复制它作为训练配置。请新建
`david_and_goliath/configs/train.yaml`，写入必要覆盖项。下面是一个最小模板：

```yaml
experiment_id: psc_run_001
seed: 42
total_rounds: 20
checkpoint_every: 5
output_dir: outputs/psc_run_001
coding_tasks_path: david_and_goliath/data/coding_tasks/tasks.jsonl
niche_capacity: 5

oracle:
  mode: payload_only
  judge_model: gpt-4o-mini
  api_key: null
  bandit_enabled: true
  semgrep_enabled: false
  judge_temperature: 0.1
  w_quality: 0.50
  w_stealth: 0.30
  w_diversity: 0.20

red_team:
  model_name: Qwen/Qwen2.5-7B-Instruct
  lora_path: ""

  cluster:
    num_training_gpus: 2
    num_inference_gpus: 4
    tensor_parallel_size: 2
    num_reward_workers: 4

  deepspeed:
    zero_stage: 3
    offload_optimizer: false
    offload_param: false

  vllm:
    gpu_memory_utilization: 0.90
    max_model_len: 2048
    enforce_eager: false

  grpo:
    group_size: 8
    kl_coeff: 0.01
    clip_eps: 0.2
    learning_rate: 5.0e-6
    prompts_per_round: 32
    max_gen_length: 512
    temperature: 0.9
    top_p: 0.95
    top_k_save: 10

  lora:
    r: 16
    alpha: 32
    dropout: 0.05

  wandb:
    enabled: true
    project: david-and-goliath
    run_name: psc_run_001
    tags: ["red-team", "grpo", "psc"]
```

关键一致性要求：

- `red_team.cluster.num_training_gpus + num_inference_gpus` 要和 Slurm 的 `#SBATCH --gpus` 对齐
- `tensor_parallel_size` 不能大于推理 GPU 数
- `output_dir` 建议和 Slurm 的 `OUTPUT_DIR` 保持一致
- PSC 小规模 smoke run 时，把 `total_rounds` 和 `prompts_per_round` 调小

## 4. 配置 Stage 2: Blue Team

默认 Blue Team 配置在：

```text
david_and_goliath/configs/blue_team/full_tools.yaml
```

默认内容指向本地 OpenAI-compatible endpoint：

```yaml
blue_team:
  model: qwen2.5-coder-32b-instruct
  api_key: null
  base_url: http://localhost:8000/v1
  temperature: 0.2
  max_turns: 6
  max_reflexion: 2
  use_tools: true
  enable_static_memory_scan: true
  enable_defense_memory_retrieval: true
  defense_memory_path: null
  defense_retrieval_top_k: 3
```

如果你在 PSC compute node 上同时启动 vLLM/OpenAI-compatible 服务，保持 `base_url: http://localhost:8000/v1` 即可。

如果你使用远程 API，把 `base_url` 改成实际 endpoint，并设置 API key：

```yaml
blue_team:
  model: your-model-name
  api_key: null
  base_url: https://your-openai-compatible-host/v1
```

然后在 Slurm 脚本或提交命令中导出：

```bash
export OPENAI_API_KEY=...
```

如果只想先跑不带工具的 Blue Team，用：

```text
david_and_goliath/configs/blue_team/llm_only.yaml
```

## 5. 配置 Stage 3: Oracle/Judging

Oracle 配置在：

```text
david_and_goliath/configs/oracle/hybrid_oracle.yaml
```

常用字段：

```yaml
oracle:
  mode: full
  judge_model: gpt-4o-mini
  api_key: null
  judge_temperature: 0.1
  bandit_enabled: true
  semgrep_enabled: false
  semgrep_rules: p/security-audit
  w_vulnerability: 0.30
  w_manipulation: 0.30
  w_quality: 0.20
  w_diversity: 0.10
  w_stealth: 0.10
```

Stage 1 推荐用 `payload_only`，因为它主要训练 Red Team payload quality / stealth / diversity，减少在线阶段对完整 Blue Team/Judge 链路的依赖。

Stage 3 会在离线阶段跑完整 judging，并复用 Stage 1 缓存的 Judge C 结果。

## 6. 本地或 login node dry-run

先不要提交 Slurm，先验证路径和命令是否能被解析：

```bash
cd /path/to/project
source .venv/bin/activate

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

成功时会打印 Stage 1 到 Stage 4 的实际命令，但不会执行训练。

你也可以单独检查 Stage 1 配置：

```bash
python -m david_and_goliath.scripts.run_coevolution \
  --config david_and_goliath/configs/train.yaml \
  --experiment-id psc_dry_run \
  --output-dir outputs/psc_dry_run \
  --oracle-mode payload_only \
  --total-rounds 1 \
  --dry-run
```

## 7. 修改 Slurm 脚本

编辑：

```text
david_and_goliath/scripts/submit_pipeline_psc.slurm
```

最少需要改这些：

```bash
#SBATCH -A YOUR_PSC_ALLOCATION
#SBATCH -p YOUR_GPU_PARTITION
#SBATCH -t 12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=6
#SBATCH --mem=0
```

并确认脚本变量：

```bash
PROJECT_DIR="${PROJECT_DIR:-$HOME/project}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
EXPERIMENT_ID="${EXPERIMENT_ID:-psc_run_001}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/outputs/${EXPERIMENT_ID}}"

TRAIN_CONFIG="${TRAIN_CONFIG:-$PROJECT_DIR/david_and_goliath/configs/train.yaml}"
BLUE_CONFIG="${BLUE_CONFIG:-$PROJECT_DIR/david_and_goliath/configs/blue_team/full_tools.yaml}"
ORACLE_CONFIG="${ORACLE_CONFIG:-$PROJECT_DIR/david_and_goliath/configs/oracle/hybrid_oracle.yaml}"
```

如果你要用 `llm_only.yaml`：

```bash
BLUE_CONFIG="$PROJECT_DIR/david_and_goliath/configs/blue_team/llm_only.yaml"
```

如果你用 WandB 或 API：

```bash
export WANDB_API_KEY=...
export OPENAI_API_KEY=...
```

## 8. 提交 PSC job

从项目根目录提交：

```bash
cd /path/to/project
mkdir -p logs
sbatch david_and_goliath/scripts/submit_pipeline_psc.slurm
```

更推荐用 `--export` 显式传入本次实验参数：

```bash
sbatch \
  --export=ALL,PROJECT_DIR=/path/to/project,VENV_DIR=/path/to/project/.venv,EXPERIMENT_ID=psc_run_001,TRAIN_CONFIG=/path/to/project/david_and_goliath/configs/train.yaml \
  david_and_goliath/scripts/submit_pipeline_psc.slurm
```

查看队列：

```bash
squeue -u $USER
```

查看 Slurm 日志：

```bash
tail -f logs/dg-pipeline-<jobid>.out
tail -f logs/dg-pipeline-<jobid>.err
```

查看 pipeline 自己的日志：

```bash
tail -f outputs/psc_run_001/logs/pipeline_*.log
```

## 9. 输出文件

默认输出在：

```text
outputs/<experiment_id>/
```

主要产物：

```text
outputs/psc_run_001/
  rollouts/
    rollouts.jsonl
  blue_team/
    blue_responses.jsonl
    logs/
  memory/
    episodes.jsonl
    blue_defense_memory.jsonl
    logs/
  analysis/
    summary.json
    report.md
    checkpoint_sweep.csv
    blue_ablation.csv
    plots/
  logs/
    pipeline_YYYYMMDD_HHMMSS.log
```

每个阶段的含义：

- `rollouts/rollouts.jsonl`: Stage 1 产生的 Red Team payload 和 cached Judge C 信息
- `blue_team/blue_responses.jsonl`: Stage 2 对每个 injected task 的 Blue Team 响应
- `memory/episodes.jsonl`: Stage 3 judging 后的完整 episode reward 记录
- `memory/blue_defense_memory.jsonl`: Stage 4 蒸馏出来的 Blue Team 历史防御记忆
- `analysis/report.md`: 汇总报告

## 10. 断点恢复

Pipeline 设计为 append/resume friendly：

- Stage 2 会跳过 `blue_responses.jsonl` 中已有且 `status == "ok"` 的 `episode_key`
- Stage 3 会跳过 `memory/episodes.jsonl` 中已经存在的 episode
- Stage 4 会从现有 `episodes.jsonl` 重新生成 defense memory

如果 Stage 1 已经完成，只重跑 Stage 2 到 Stage 4：

```bash
python -m david_and_goliath.scripts.run_pipeline \
  --config david_and_goliath/configs/train.yaml \
  --blue-config david_and_goliath/configs/blue_team/full_tools.yaml \
  --oracle-config david_and_goliath/configs/oracle/hybrid_oracle.yaml \
  --experiment-id psc_run_001 \
  --output-dir outputs/psc_run_001 \
  --skip-stage1 \
  --skip-analysis \
  --log-level INFO
```

如果 Stage 2 也已经完成，只重跑 Stage 3 和 Stage 4：

```bash
python -m david_and_goliath.scripts.run_pipeline \
  --oracle-config david_and_goliath/configs/oracle/hybrid_oracle.yaml \
  --experiment-id psc_run_001 \
  --output-dir outputs/psc_run_001 \
  --skip-stage1 \
  --skip-stage2 \
  --skip-analysis \
  --log-level INFO
```

如果只重新生成 defense memory：

```bash
python -m david_and_goliath.scripts.run_offline_defense_memory \
  --episodes-path outputs/psc_run_001/memory/episodes.jsonl \
  --output-path outputs/psc_run_001/memory/blue_defense_memory.jsonl \
  --log-level INFO
```

## 11. Debug 小规模运行

PSC 上第一次真跑建议先做小规模 job：

```bash
python -m david_and_goliath.scripts.run_pipeline \
  --config david_and_goliath/configs/train.yaml \
  --blue-config david_and_goliath/configs/blue_team/full_tools.yaml \
  --oracle-config david_and_goliath/configs/oracle/hybrid_oracle.yaml \
  --experiment-id psc_smoke \
  --output-dir outputs/psc_smoke \
  --oracle-mode payload_only \
  --total-rounds 1 \
  --limit-stage2 10 \
  --limit-stage3 10 \
  --skip-analysis \
  --log-level DEBUG
```

如果只想验证 Stage 2 能读 Stage 1 输出：

```bash
python -m david_and_goliath.scripts.run_offline_blue_team \
  --rollouts-path outputs/psc_smoke/rollouts/rollouts.jsonl \
  --output-path outputs/psc_smoke/blue_team/blue_responses.jsonl \
  --config david_and_goliath/configs/blue_team/full_tools.yaml \
  --limit 10 \
  --dry-run
```

如果只想验证 Stage 3 输入完整：

```bash
python -m david_and_goliath.scripts.run_offline_judging \
  --rollouts-path outputs/psc_smoke/rollouts/rollouts.jsonl \
  --blue-responses-path outputs/psc_smoke/blue_team/blue_responses.jsonl \
  --output-path outputs/psc_smoke/memory/episodes.jsonl \
  --config david_and_goliath/configs/oracle/hybrid_oracle.yaml \
  --limit 10 \
  --dry-run
```

## 12. 常见问题

### `FileNotFoundError: david_and_goliath/configs/train.yaml`

Slurm 脚本默认引用 `configs/train.yaml`，但仓库里未必有。创建该文件，或提交时覆盖：

```bash
sbatch \
  --export=ALL,TRAIN_CONFIG=/path/to/your/train.yaml,PROJECT_DIR=/path/to/project \
  david_and_goliath/scripts/submit_pipeline_psc.slurm
```

### Blue Team 连接失败

检查 `full_tools.yaml` 的：

```yaml
base_url: http://localhost:8000/v1
model: qwen2.5-coder-32b-instruct
```

如果没有在同一个 compute node 上启动模型服务，`localhost:8000` 会失败。改成真实 endpoint，或先启动 vLLM server。

### GPU 数不匹配

检查三处是否一致：

- Slurm: `#SBATCH --gpus=6`
- Train config: `red_team.cluster.num_training_gpus`
- Train config: `red_team.cluster.num_inference_gpus`

默认逻辑期望：

```text
total requested GPUs >= num_training_gpus + num_inference_gpus
```

### `ModuleNotFoundError`

必须从 repo 根目录运行：

```bash
cd /path/to/project
python -m david_and_goliath.scripts.run_pipeline ...
```

不要从 `david_and_goliath/scripts` 目录里直接执行脚本。

### 依赖安装失败

训练相关依赖和 CUDA 版本强相关。先在交互式 GPU session 中验证：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import ray; print(ray.__version__)"
python -c "import transformers; print(transformers.__version__)"
```

### Analysis 阶段太慢或依赖缺失

先跳过：

```bash
--skip-analysis
```

训练和 judging 跑通后，再单独执行：

```bash
python -m david_and_goliath.scripts.run_offline_analysis \
  --experiment-dir outputs/psc_run_001 \
  --output-dir outputs/psc_run_001/analysis \
  --blue-config david_and_goliath/configs/blue_team/full_tools.yaml \
  --oracle-config david_and_goliath/configs/oracle/hybrid_oracle.yaml \
  --log-level INFO
```

## 13. 推荐提交模板

最终常用提交命令：

```bash
cd /path/to/project
mkdir -p logs

sbatch \
  --export=ALL,PROJECT_DIR=/path/to/project,VENV_DIR=/path/to/project/.venv,EXPERIMENT_ID=psc_run_001,OUTPUT_DIR=/path/to/project/outputs/psc_run_001,TRAIN_CONFIG=/path/to/project/david_and_goliath/configs/train.yaml,BLUE_CONFIG=/path/to/project/david_and_goliath/configs/blue_team/full_tools.yaml,ORACLE_CONFIG=/path/to/project/david_and_goliath/configs/oracle/hybrid_oracle.yaml \
  david_and_goliath/scripts/submit_pipeline_psc.slurm
```

如果这条命令成功，`logs/dg-pipeline-<jobid>.out` 会先打印 pipeline resolved command，然后开始 Stage 1。Stage 1 完成后会进入 Stage 2/3/4，并最终在 `outputs/psc_run_001/` 下留下完整产物。

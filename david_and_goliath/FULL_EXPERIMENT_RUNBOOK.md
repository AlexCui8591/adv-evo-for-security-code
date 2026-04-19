# David & Goliath 全量实验运行手册

本文档说明如何在 PSC Bridges-2 上运行当前的全量协同进化实验。默认入口是：

```bash
sbatch david_and_goliath/scripts/run_coevolution_full.sbatch
```

## 1. 资源与实验配置

默认全量脚本会申请单节点 8 张 H100 80GB GPU：

```text
partition: GPU
nodes: 1
gpus: h100-80:8
cpus: 64
memory: 504G
time: 24:00:00
job name: dg-full-8b-32b
```

GPU 分配：

```text
GPU 0-3: Blue Team 本地 vLLM 服务
GPU 4-7: Red Team Ray / OpenRLHF 训练与推理
```

脚本会读取源配置：

```text
david_and_goliath/configs/experiment/coevo_8b.yaml
```

然后在作业输出目录中生成 job-local 配置：

```text
outputs/coevo_8b_local_blue_32b_single_node/generated_configs/full_${SLURM_JOB_ID}.yaml
```

生成配置会保留 `total_rounds: 20` 的全量实验，同时把 Red Team 的 GPU 需求改成单节点 4 GPU 可运行的资源划分。

## 2. 第一次运行前准备

进入仓库根目录：

```bash
cd /ocean/projects/cis250260p/cuiz/11766-project/adv-evo-for-security-code
```

加载 Hugging Face 缓存环境：

```bash
source david_and_goliath/scripts/hf_env.sh
```

创建或更新 Conda 环境。推荐使用项目脚本，因为 `torch`、`vllm`、`deepspeed`、`flash-attn`、`openrlhf` 必须按顺序安装：

```bash
cd david_and_goliath
bash setup_env.sh
cd ..
```

也可以参考依赖规格：

```text
david_and_goliath/environment.yml
david_and_goliath/requirements.txt
```

但全量实验的正式环境安装仍以 `setup_env.sh` 为准。

## 3. 配置密钥

创建本地密钥文件：

```bash
cp david_and_goliath/scripts/secrets.sh.example david_and_goliath/scripts/secrets.sh
```

编辑 `david_and_goliath/scripts/secrets.sh`，填入 CMU AI Gateway 或 OpenAI-compatible endpoint 使用的 key：

```bash
export OPENAI_API_KEY="<your-key>"
```

真实的 `secrets.sh` 不应提交到 git。

## 4. 预下载模型

全量实验默认使用：

```text
Red Team:  unsloth/Llama-3.1-8B-Instruct + david_and_goliath/final_adapter
Blue Team: Qwen/Qwen2.5-Coder-32B-Instruct
```

建议先提交一个小的网络 smoke test：

```bash
sbatch david_and_goliath/scripts/slurm_prefetch_hf_models.sbatch --profile smoke
```

确认下载可用后预下载当前实验需要的模型：

```bash
sbatch david_and_goliath/scripts/slurm_prefetch_hf_models.sbatch --profile active
```

也可以只下载某一侧模型：

```bash
sbatch david_and_goliath/scripts/slurm_prefetch_hf_models.sbatch --profile red8b
sbatch david_and_goliath/scripts/slurm_prefetch_hf_models.sbatch --profile blue32b
```

预下载完成后，全量实验脚本默认使用：

```bash
HF_HUB_OFFLINE=1
```

这能让训练只从项目缓存读模型，避免运行中途访问外网。

## 5. 提交全量实验

从仓库根目录提交：

```bash
cd /ocean/projects/cis250260p/cuiz/11766-project/adv-evo-for-security-code
sbatch david_and_goliath/scripts/run_coevolution_full.sbatch
```

常用覆盖参数：

```bash
# 先跑 5 轮检查流程
sbatch --export=ALL,TOTAL_ROUNDS=5 david_and_goliath/scripts/run_coevolution_full.sbatch

# 从最近 checkpoint 续跑
sbatch --export=ALL,RESUME=true david_and_goliath/scripts/run_coevolution_full.sbatch

# 改实验 ID，输出到 outputs/<EXPERIMENT_ID>
sbatch --export=ALL,EXPERIMENT_ID=my_full_run_001 david_and_goliath/scripts/run_coevolution_full.sbatch

# 使用 V100 32GB 队列时覆盖 GPU 资源
sbatch --gres=gpu:v100-32:8 david_and_goliath/scripts/run_coevolution_full.sbatch
```

可覆盖的主要环境变量：

```text
PROJECT_ROOT
CONDA_ENV
CONFIG_PATH
OUTPUT_DIR
TOTAL_ROUNDS
EXPERIMENT_ID
RESUME
LOG_LEVEL

BLUE_MODEL
BLUE_GPUS
BLUE_PORT
BLUE_TENSOR_PARALLEL_SIZE
BLUE_MAX_MODEL_LEN
BLUE_GPU_MEMORY_UTILIZATION

RED_GPUS
RED_RAY_GPUS
RED_TRAINING_GPUS
RED_INFERENCE_GPUS
RED_TENSOR_PARALLEL_SIZE
RED_REWARD_WORKERS

HF_HUB_OFFLINE
RAY_PORT
RAY_DASHBOARD_PORT
```

## 6. 监控作业

查看队列状态：

```bash
squeue -u "$USER"
```

查看 Slurm 主日志：

```bash
tail -f dg-full-8b-32b-<job_id>.out
```

查看 Blue Team vLLM 日志：

```bash
tail -f outputs/coevo_8b_local_blue_32b_single_node/logs/blue_vllm_<job_id>.log
```

脚本启动顺序：

```text
1. source hf_env.sh
2. source secrets.sh
3. conda activate dg
4. 生成 job-local YAML 配置
5. 停掉 stale Ray
6. 在 GPU 0-3 启动 Blue Team vLLM
7. 等待 /v1/models ready
8. 在 GPU 4-7 启动 Ray head
9. 等待 Ray GPU resources ready
10. 调用 scripts/run_coevolution.py 开始 20 轮协同进化
```

## 7. 输出文件

默认输出目录：

```text
outputs/coevo_8b_local_blue_32b_single_node
```

重点查看：

```text
logs/blue_vllm_<job_id>.log
generated_configs/full_<job_id>.yaml
checkpoints/
round_records/
```

如果使用 `EXPERIMENT_ID` 覆盖，`run_coevolution_full.sbatch` 会把最终输出目录设置为：

```text
outputs/<EXPERIMENT_ID>
```

## 8. 失败排查

如果作业启动后马上失败，优先检查：

```text
1. david_and_goliath/scripts/secrets.sh 是否存在
2. OPENAI_API_KEY 是否已设置
3. conda env 名称是否与 CONDA_ENV 一致，默认 dg
4. HF cache 是否已经预下载 Red/Blue 模型
5. HF_HUB_OFFLINE=1 时模型是否真的在缓存里
6. Blue vLLM 日志是否出现 OOM 或模型加载失败
7. Red Team GPU 划分是否满足 RED_TRAINING_GPUS + RED_INFERENCE_GPUS <= RED_RAY_GPUS
```

若需要快速验证脚本逻辑，先跑短实验：

```bash
sbatch --export=ALL,TOTAL_ROUNDS=1,LOG_LEVEL=DEBUG david_and_goliath/scripts/run_coevolution_full.sbatch
```

若上次作业生成了 checkpoint，续跑：

```bash
sbatch --export=ALL,RESUME=true david_and_goliath/scripts/run_coevolution_full.sbatch
```


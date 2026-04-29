# David & Goliath — 协同进化实验完整流程

> 红队 Qwen2.5-7B(LoRA + GRPO)vs. 本地蓝队 Qwen2.5-Coder-72B(tool-use agent)协同进化,
> 在 PSC Bridges-2(H100-80G 集群)上的端到端实验手册。

---

## 目录
- [0. 系统概览](#0-系统概览)
- [1. 前置材料清单](#1-前置材料清单)
- [2. 资源预算](#2-资源预算)
- [阶段 A:环境安装](#阶段-a环境安装)
- [阶段 B:OpenRLHF API 对接修复](#阶段-bopenrlhf-api-对接修复)
- [阶段 C:Smoke Test](#阶段-csmoke-test)
- [阶段 D:编写多节点 SLURM 批处理脚本](#阶段-d编写多节点-slurm-批处理脚本)
- [阶段 E:完整协同进化训练](#阶段-e完整协同进化训练)
- [阶段 F:评估与分析](#阶段-f评估与分析)
- [全局依赖图](#全局依赖图)
- [失败恢复与回退策略](#失败恢复与回退策略)
- [命令速查](#命令速查)
- [关键文件索引](#关键文件索引)

---

## 0. 系统概览

### 架构

```
┌────────────────────────────────────────────────────────────────────┐
│  Ray Cluster(PSC Bridges-2 GPU 分区,H100-80G)                      │
│                                                                    │
│  ┌─────────────────────────────┐    ┌───────────────────────────┐ │
│  │ Red Team 训练(node-0)       │    │ Blue Team 72B 服务(node-1)│ │
│  │  - DeepSpeed ZeRO-3 Actor   │    │  - vLLM server            │ │
│  │    4× H100(policy + ref)    │    │  - tp=2-4                 │ │
│  │  - vLLM 推理引擎池          │    │  - OpenAI 兼容 /v1 端点   │ │
│  │    8× H100(LoRA hot-swap)   │    │  - http://<ip>:8000/v1    │ │
│  │  - OracleRewardWorker Pool  │    │                           │ │
│  │    (CPU workers, 并发评分)  │    │                           │ │
│  └─────────────────────────────┘    └───────────────────────────┘ │
│                                                                    │
│  每轮: RedPromptBuilder → vLLM生成 → InjectionEngine 注入 →        │
│        Blue Team(reflexion + tool-use) → HybridOracle(3 judges) → │
│        GRPO 优势 → DeepSpeed 梯度更新 → MAP-Elites 归档             │
└────────────────────────────────────────────────────────────────────┘
```

### 规模参数(来自 [coevo_8b_local_blue_72b.yaml](configs/experiment/coevo_8b_local_blue_72b.yaml)

| 维度 | 值 |
|------|-----|
| 总轮数 `total_rounds` | 20 |
| 每轮 prompt 数 `prompts_per_round` | 32 |
| GRPO 组内 completion 数 `group_size` | 8 |
| 每轮 Oracle 评分次数 | 32 × 8 = 256 episode |
| 每 episode 最多蓝队调用 | `max_turns × (max_reflexion+1) = 6×3 = 18` LLM 次 |
| 每轮 72B 峰值调用 | ≤ 256 × 18 ≈ **4608 次**(实际因早停会少) |
| 每 5 轮 checkpoint | `checkpoint_every: 5` |

---

## 1. 前置材料清单

在跑**阶段 A**之前,请逐项确认:

### 1.1 模型 checkpoint

- [ ] **红队 SFT LoRA**:配置里 `red_team.lora_path=""` 为空,表示从 `Qwen2.5-7B-Instruct` **基座**零 fine-tune。
  若实际要从一个已有的 SFT adapter 开始,需要指明路径(推测在 `final_merged-red_tean/final_adapter/` 或 `redteam_sft/`)。
- [ ] **蓝队 72B 权重**:`Qwen/Qwen2.5-Coder-72B-Instruct`,bf16 约 **140 GB**。
  首次下载会放到 `~/.cache/huggingface`(/jet/home 配额有限),建议 `export HF_HOME=/ocean/projects/cis250260p/cuiz/hf_cache` 迁到 `/ocean`。

### 1.2 数据

- [ ] **编程任务集**:`data/coding_tasks/tasks.jsonl` 是否存在?
  如果没有,先跑 `python scripts/prepare_data.py`(需要核对该脚本的输入源)。

### 1.3 Judge API

- [ ] Judge B/C 配置 `judge_model: sonnet-4.6`,但 `api_key: null`。
  Anthropic 的 `sonnet-4.6` 通过 **Anthropic API 还是 OpenAI-兼容代理**调用?
  两种路径都需要在 `configs/experiment/*.yaml` 或环境变量里补:`api_key` + `base_url`。
- [ ] Judge A 走 `bandit` CLI(阶段 A 的 sanity check 会验证存在性)。
- [ ] 可选:Judge A 还支持 `semgrep`,当前 `semgrep_enabled: false`,不强制。

### 1.4 PSC 账户

- [ ] `projects` 命令确认 `cis250260p` 至少剩 **300+ SU**(保证一次完整训练 + 若干次 smoke)。
- [ ] 确认能 `interact -p GPU-shared --gres=gpu:h100-80:1`(H100-shared 队列必须有配额)。

---

## 2. 资源预算

### 2.1 GPU 节点需求

| 阶段 | 节点 / GPU | 分区 | 时长 | SU(悲观估算) |
|------|-----------|------|------|---------------|
| A 环境安装 | 1× H100 | `GPU-shared` | 2 h | ~2 |
| B API 修复(纯看代码) | 0 GPU | (本地) | — | 0 |
| C Smoke test | 4× H100 | `GPU-shared` | 1 h | ~4 |
| D sbatch 脚本编写 | 0 GPU | (本地) | — | 0 |
| E 完整协同进化(20 轮) | 2 节点 × 8 H100 | `GPU` | 10–20 h | 320–640 |
| F 评估分析 | 0 GPU 或 1 H100 | — | < 1 h | ~1 |

**当前预算**:2977 SU → 支持 **1 次正式全量 + 多次 smoke + 4–9 次 ablation**。

### 2.2 存储

- `/ocean/projects/cis250260p/cuiz/`:主工作区(当前 412 GB 占用 / 3 TB 配额,宽裕)
- `/jet/home/cuiz/`:个人目录,配额有限,**不要把 HF cache 放这里**

### 2.3 关键环境变量

```bash
export HF_HOME=/ocean/projects/cis250260p/cuiz/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export WANDB_DIR=/ocean/projects/cis250260p/cuiz/wandb
export RAY_TMPDIR=/tmp/ray-$USER
export OPENAI_API_KEY=...        # sonnet-4.6 judge
export ANTHROPIC_API_KEY=...     # 如果走 Anthropic 原生接口
```

---

## 阶段 A:环境安装

**目标**:在 PSC GPU 节点上建 `dg` conda env,装好 OpenRLHF+Ray+vLLM+DeepSpeed 栈,并通过 import sanity check。

### A.1 申请 GPU 节点

```bash
interact -p GPU-shared --gres=gpu:h100-80:1 -t 2:00:00 -N 1 \
         --ntasks-per-node=8 --mem=64G
```

### A.2 加载 CUDA + 运行安装脚本

```bash
module load cuda/12.1
cd /ocean/projects/cis250260p/cuiz/11766-project/adv-evo-for-security-code/david_and_goliath
bash setup_env.sh 2>&1 | tee logs/setup_env.log
```

### A.3 脚本执行顺序

`setup_env.sh` 内部严格按此顺序(跳过任何一步都会炸):

1. 建/复用 `dg` conda env(Python 3.10)
2. 装 **torch 2.4.0 + cu121**(其他编译型包的 ABI 基线)
3. 装 `vllm==0.6.3.post1`
4. 装 `deepspeed==0.15.4`
5. 装 `flash-attn==2.6.3`(`--no-build-isolation`,必须先有 torch)
6. 装 `openrlhf==0.5.7`(`--no-deps`,不让它覆盖前面的版本)
7. 装 `requirements.txt` 剩余包
8. **Sanity check**:import 全部包 + import `openrlhf` 那 4 个我们真正用到的符号 + bandit CLI

### A.4 成功标志

```
OpenRLHF symbol check passed.
bandit CLI OK
==> Environment 'dg' ready.
```

### A.5 常见失败

| 症状 | 原因 | 处理 |
|------|------|------|
| `nvcc not found` | 没 `module load cuda/12.1` | 重载模块,再跑 |
| `flash-attn` 编译失败 | torch 还没装好 / gcc 版本不对 | 检查 `nvcc --version` 和 `gcc --version ≥ 9` |
| `ImportError: cannot import name 'ActorModelRayActor'` | OpenRLHF 0.5.7 已改名 | 进 [阶段 B](#阶段-bopenrlhf-api-对接修复) |
| 磁盘满 | HF cache / pip cache 占 `/jet/home` | `export HF_HOME=/ocean/projects/...`,`pip cache purge` |

---

## 阶段 B:OpenRLHF API 对接修复

**触发条件**:阶段 A 最后的 sanity check 失败,或后续 smoke test 出现 `AttributeError` / `TypeError`。

### B.1 去现场读源码

```bash
conda activate dg
python -c "import openrlhf, os; print(os.path.dirname(openrlhf.__file__))"
# → 进入打印出的目录,现场查下面 5 个点的真实 API
```

### B.2 必须核对的 5 个调用点([grpo_trainer.py](red_team/grpo_trainer.py)

| # | 调用 | 文件 | 需要核对什么 |
|---|------|------|--------------|
| 1 | `LLMRayActor.options(...).remote(enable_lora=True, max_lora_rank=...)` | `openrlhf/trainer/ray/vllm_engine.py` | 构造函数真的接受 `enable_lora` 参数吗? |
| 2 | `engine.generate.remote(prompts, dict_sampling_params)` | 同上 | 传 dict 还是 `vllm.SamplingParams` 对象?返回结构? |
| 3 | `actor.get_lora_state_dict.remote()` + `engine.update_lora_weights.remote()` | `openrlhf/trainer/ray/ppo_actor.py` + vllm_engine | 这两个方法真的存在吗?若不存在要改走 NCCL weight sync |
| 4 | `DeepspeedStrategy(zero_stage=, bf16=, learning_rate=)` | `openrlhf/utils/deepspeed.py` | 构造签名是否匹配?还是要通过 `Args` dataclass? |
| 5 | `training_step(experience=dict)` | `openrlhf/trainer/ray/ppo_actor.py` | 入参真的是 dict,还是 `Experience` dataclass?后者要自己算 old_log_probs、ref_log_probs、action_mask |

### B.3 修复策略

- **小修**:重命名方法 / 改 import 路径 → 直接 `Edit`
- **大修**(大概率是第 5 点):需要重写整个 `_grpo_update` —— 手动构造 `Experience` + 跑 ref model 拿 log_probs + 拼 action_mask
- **最后选项**:若 OpenRLHF 0.5.7 里 LoRA 热同步那套确实不存在,切换成 **每轮 save LoRA → vLLM reload** 的笨办法(慢但能跑)

### B.4 完成标志

在阶段 C smoke test 里,单轮 `train_round()` 能跑完且 loss 是有限数。

---

## 阶段 C:Smoke Test

**目标**:用极小规模跑一轮,让所有组件都被调用一次,暴露集成 bug。

### C.1 申请节点

```bash
interact -p GPU-shared --gres=gpu:h100-80:4 -t 1:00:00 -N 1 --mem=256G
```

### C.2 建 smoke 配置

创建 `configs/experiment/smoke_test.yaml`(override 主配置):

```yaml
experiment_id: smoke_test
total_rounds: 1
checkpoint_every: 1
coding_tasks_path: david_and_goliath/data/coding_tasks/tasks.jsonl

blue_team:
  model: Qwen/Qwen2.5-Coder-72B-Instruct
  base_url: http://127.0.0.1:8000/v1
  api_key: EMPTY
  max_turns: 2          # 压到 2,减少 72B 调用
  max_reflexion: 1

red_team:
  model_name: Qwen/Qwen2.5-7B-Instruct
  lora_path: ""
  cluster:
    num_training_gpus: 1
    num_inference_gpus: 1
    tensor_parallel_size: 1
    num_reward_workers: 2
  grpo:
    group_size: 4
    prompts_per_round: 4       # 1×4 = 4 episode,总 72B 调用 ≤ 4×2×2 = 16
    max_gen_length: 256
  vllm:
    max_model_len: 2048
    gpu_memory_utilization: 0.80
```

### C.3 启动顺序

**Terminal 1 — 72B 蓝队 vLLM server**(占 2 GPU,tp=2):
```bash
CUDA_VISIBLE_DEVICES=0,1 vllm serve Qwen/Qwen2.5-Coder-72B-Instruct \
    --tensor-parallel-size 2 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.9 \
    --port 8000 &

# 等就绪
until curl -s http://127.0.0.1:8000/v1/models >/dev/null; do sleep 5; done
echo "72B server up"
```

**Terminal 2 — Ray + 红队训练**(占剩下 2 GPU):
```bash
ray start --head --num-gpus=2
cd /ocean/projects/cis250260p/cuiz/11766-project/adv-evo-for-security-code/david_and_goliath
CUDA_VISIBLE_DEVICES=2,3 python scripts/run_coevolution.py \
    --config configs/experiment/smoke_test.yaml
```

### C.4 Smoke 成功标志

依次看到日志:
1. `GRPOTrainer ready: 1 training GPUs ...`
2. `[Round 0] vLLM generating 4 × 4 completions...`
3. `[Round 0] Scoring 16 completions via Oracle workers...`
4. `[Round 0] GRPO update (DeepSpeed ZeRO-3)...`
5. `[Round 0] reward_mean=... ASR=...% elapsed=...s`
6. `outputs/smoke_test/checkpoints/round_001/` 出现 LoRA adapter

### C.5 常见问题

| 症状 | 可能原因 |
|------|----------|
| 72B server 起不来 | GPU 显存不够 / 权重没下全 / tp 配错 |
| 红队卡在 `_sync_lora_to_vllm` | 阶段 B 第 3 点(LoRA 热同步 API 不对)没修 |
| `HybridOracle` 报 JSON parse error | 蓝队输出不符合 JSON,要么加 retry 要么改 prompt |
| 所有 reward 都 0 | Oracle 路由错(sonnet-4.6 api_key 没填)/ bandit 没装 |
| `Experience` 构造报错 | 阶段 B 第 5 点没修 |

---

## 阶段 D:编写多节点 SLURM 批处理脚本

**目标**:把 smoke test 手工启动的那些步骤,写成一个 `run_coevolution.sbatch`,供阶段 E 提交。

### D.1 关键设计点

- **2 节点**:node-0 跑红队训练 + Ray head + 8 vLLM 推理引擎,node-1 跑蓝队 72B vLLM server
- **分 GPU**:node-0 的 8 GPU 给 Ray inference engines(tp=1,8 engine),node-0 的另外 4 GPU? —— 其实一个节点只有 8 GPU,所以 training 和 inference 得分节点。**需要重新算账**:
  - 方案一:node-0 = 4 training + 4 inference(tp=1);node-1 = 4 inference(tp=1)+ 蓝队 72B(tp=4)→ 太挤
  - 方案二:申请 **3 节点**,node-2 专门蓝队 72B(需要 ~400 SU/轮)
  - 方案三:减 `num_inference_gpus=4`,节约 4 GPU 给蓝队 → 吞吐下降但资源省
- **Ray 启动**:node-0 `ray start --head`;node-1 `ray start --address=<head_ip>:6379`
- **同步点**:先等 72B server 就绪 → 再起 Ray cluster → 再跑 `run_coevolution.py`

### D.2 脚本骨架(不在本文档里写实现,只列大纲)

```bash
#!/usr/bin/env bash
#SBATCH -p GPU -N 2 --gres=gpu:h100-80:8 -t 20:00:00
#SBATCH -J coevo_8b_72b -o logs/coevo_%j.out

module load cuda/12.1
source activate dg
export HF_HOME=/ocean/projects/cis250260p/cuiz/hf_cache
export OPENAI_API_KEY=...

# (1) 在 node-1 起 72B vLLM server
srun --nodes=1 --ntasks=1 --nodelist=$(scontrol show hostnames $SLURM_NODELIST | sed -n 2p) \
    bash scripts/launch_blue_server.sh &

# (2) 等 server 就绪
BLUE_IP=...
until curl -s http://${BLUE_IP}:8000/v1/models; do sleep 5; done

# (3) 在 node-0 起 Ray head
ray start --head --port=6379 --num-gpus=8

# (4) 启动训练
python scripts/run_coevolution.py \
    --config configs/experiment/coevo_8b_local_blue_72b.yaml \
    --override blue_team.base_url=http://${BLUE_IP}:8000/v1
```

### D.3 验证方法

先用 `sbatch -t 00:30:00` 跑半小时,确认 node-0 / node-1 通信 + 72B 可访问 + 训练不 OOM,再换 20 小时。

---

## 阶段 E:完整协同进化训练

**目标**:按主配置跑满 20 rounds,产出最终模型 + MAP-Elites 策略库。

### E.1 提交

```bash
cd /ocean/projects/cis250260p/cuiz/11766-project/adv-evo-for-security-code/david_and_goliath
sbatch scripts/run_coevolution.sbatch
```

### E.2 监控

- `squeue -u $USER` 看作业状态
- `tail -f logs/coevo_<jobid>.out` 实时日志
- `wandb` dashboard(若开启)看 reward / ASR / coverage 曲线
- 每 5 轮 `outputs/coevo_8b_local_blue_72b/checkpoints/round_NNN/` 出一次 LoRA 快照

### E.3 中途健康指标(应该看到)

| 轮次 | `grpo_avg_reward` | `attack_success_rate` | `red_diversity_coverage` |
|------|-------------------|-----------------------|--------------------------|
| 1    | ~0.1–0.3(冷启动)| < 10%                | 0(空库)                 |
| 5    | 单调上升         | 10–30%               | 0.1–0.3                  |
| 10   | > 0.5            | 30–50%               | 0.3–0.5                  |
| 20   | > 0.7            | > 50%                | > 0.6                    |

**异常信号**:
- reward 塌缩到常数 → KL coeff 太小 / Oracle 给分退化
- ASR 过早饱和(轮 3–5 就 90%+)→ 蓝队太弱 / Oracle 判定太松
- coverage 卡 0 → MAP-Elites niche 全被挤占,调大 `niche_capacity`

### E.4 失败恢复

```bash
# 在 yaml 里:resume: true,然后重新 sbatch
# CoEvolutionController 会从最近的 round_NNN/ 接着跑
```

---

## 阶段 F:评估与分析

**目标**:产出可写入报告的指标 + 曲线 + 代表性 payload。

### F.1 产出位置

```
outputs/coevo_8b_local_blue_72b/
├── checkpoints/
│   ├── round_005/                    # LoRA adapter
│   ├── round_010/
│   └── round_020/
├── history.json                      # 每轮所有 stats
├── strategy_db.json                  # MAP-Elites 最终状态
├── plots/
│   └── training_curves.png           # 2×2 四子图
└── top_payloads.jsonl                # 全程 top-K payload
```

### F.2 分析步骤

1. **曲线**:检查 `plots/training_curves.png`(Avg Reward、ASR、Judge A/B 触发率、Diversity Coverage)
2. **MAP-Elites 覆盖**:从 `strategy_db.json` 可视化 (injection_type × stealth_level) 二维占据图
3. **代表性 payload**:`top_payloads.jsonl` 里挑 3–5 条高 reward + 高 stealth 的,放进报告
4. **红队泛化评估**(可选):加载最终 LoRA,在 held-out coding tasks 上跑一轮无训练 inference,计 ASR
5. **蓝队回归评估**(可选):把红队历史 top-K payload 回放给 baseline 72B vs. 最终 72B,确认蓝队是否"见过就学会防"

### F.3 报告指标核对

| 指标 | 来源 |
|------|------|
| Final ASR | `history.json[-1].attack_success_rate` |
| Reward improvement | `history.json[0/-1].grpo_avg_reward` |
| Diversity coverage | `history.json[-1].red_diversity_coverage` |
| Judge A/B/C 分布 | `top_payloads.jsonl` 聚合 |
| 训练总时长 | `history.json` 所有 `grpo_train_time_s` 之和 |

---

## 全局依赖图

```
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│   [0 前置]                                                                │
│     │                                                                    │
│     ├── 7B SFT ckpt ───┐                                                 │
│     ├── 72B HF cache ──┤                                                 │
│     ├── tasks.jsonl ───┼─┐                                               │
│     └── Judge API keys─┘ │                                               │
│                          │                                               │
│     ▼                    ▼                                               │
│   [A 装环境] ──▶ [B 修 API] ──▶ [C Smoke] ──▶ [D sbatch] ──▶ [E 全量训练] ──▶ [F 评估] │
│                      ▲              │                                    │
│                      └──────失败─────┘                                    │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 失败恢复与回退策略

| 阶段 | 常见失败 | 回退路径 |
|------|----------|----------|
| A | 编译失败 | 降级 `flash-attn` 到 2.5.x,或全程跑 `enforce_eager=True`(不用 flash-attn) |
| B | OpenRLHF API 差异过大 | **Plan B**:改走 `--remote_rm_url` HTTP 奖励服务器 + OpenRLHF CLI 一次性任务(放弃 per-round `train_round()` 接口,co_evolution_controller 要改) |
| C | 72B 吞吐撑不住 | 减 `prompts_per_round` 到 8,或 `max_turns=3 max_reflexion=0` |
| D | 2 节点不够 | 申请 3 节点(蓝队独占 1 节点),SU 预算仍够 |
| E | Ray head/worker 通信失败 | 检查 PSC 节点间防火墙 + `RAY_ADDRESS` 环境变量,或改成单节点(减 GPU 规模) |
| F | ASR 塌缩 | 调 oracle 权重 + kl_coeff,回到 E 重跑(多用 `resume: true`) |

---

## 命令速查

### 资源申请
```bash
# 阶段 A 安装
interact -p GPU-shared --gres=gpu:h100-80:1 -t 2:00:00 -N 1 --mem=64G

# 阶段 C Smoke
interact -p GPU-shared --gres=gpu:h100-80:4 -t 1:00:00 -N 1 --mem=256G

# 阶段 E 完整训练
sbatch scripts/run_coevolution.sbatch
```

### 常用查询
```bash
projects                         # 剩余 SU
squeue -u $USER                  # 我的作业
sinfo -p GPU -t idle | grep h100 # H100 空闲节点
scancel <jobid>                  # 取消作业
sacct -j <jobid>                 # 作业历史
```

### 环境激活
```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dg
module load cuda/12.1
```

### 训练启动
```bash
python scripts/run_coevolution.py \
    --config configs/experiment/coevo_8b_local_blue_72b.yaml
```

### Checkpoint 恢复
```bash
# 在 yaml 里设 resume: true,然后正常提交
sbatch scripts/run_coevolution.sbatch
```

---

## 关键文件索引

| 文件 | 角色 | 状态 |
|------|------|------|
| [setup_env.sh](setup_env.sh) | conda env + 栈安装脚本 | ✅ 已写 |
| [requirements.txt](requirements.txt) | Python 包版本锚点 | ✅ 已写 |
| [configs/experiment/coevo_8b_local_blue_72b.yaml](configs/experiment/coevo_8b_local_blue_72b.yaml) | 主实验配置 | ✅ 已有 |
| [red_team/grpo_trainer.py](red_team/grpo_trainer.py) | GRPO 训练器(OpenRLHF) | ⚠️ 阶段 B 可能要改 |
| [core/co_evolution_controller.py](core/co_evolution_controller.py) | 主编排器 | ✅ 已有 |
| [blue_team/coding_agent.py](blue_team/coding_agent.py) | 蓝队 tool-use agent | ✅ 已有 |
| [hybrid_oracle/oracle.py](hybrid_oracle/oracle.py) | 三判官混合 oracle | ✅ 已有 |
| [scripts/run_coevolution.py](scripts/run_coevolution.py) | 训练入口脚本 | ✅ 已有 |
| `scripts/run_coevolution.sbatch` | **阶段 D 待写**:多节点 SLURM 批处理 | ❌ 缺失 |
| `configs/experiment/smoke_test.yaml` | **阶段 C 待写**:smoke 配置 override | ❌ 缺失 |
| `data/coding_tasks/tasks.jsonl` | 编程任务数据集 | ❓ 待确认 |

---

## 当前立即行动项(按顺序)

1. **回答前置清单里 4 个问号**(7B ckpt 位置 / 72B cache / tasks.jsonl / judge API key)
2. **跑阶段 A**:`interact` → `bash setup_env.sh`,看 sanity check 结果
3. **根据 A 的结果**决定是否要进阶段 B 修 API;或直接进阶段 C smoke

文档维护:每次阶段切换完成后,在对应节底部加一条"实际耗时 + 遇到问题"的记录,方便复现和调 SU 预算。

# David & Goliath: 基于协同进化的红蓝对抗 LLM 代码智能体安全框架

一个对抗性协同进化框架：**红队**（小型开源 LLM，通过 GRPO 训练）学习生成 prompt injection 攻击载荷，攻击**蓝队**（基于 LLM 的 coding agent），由**混合评判器**（静态分析 + LLM-as-Judge）进行评分。**MAP-Elites** 策略数据库维持攻击多样性，防止模式坍塌。

## 架构总览

```
┌─────────────────────────────────────────────────────────────────┐
│                   CoEvolutionController                         │
│                   (core/co_evolution_controller.py)              │
│                                                                 │
│   for round in 1..N:                                            │
│     1. 红队 GRPO 训练 → 生成攻击载荷                              │
│     2. InjectionEngine 将载荷嵌入编程任务                          │
│     3. 蓝队 (coding agent) 处理注入后的任务                        │
│     4. 混合评判器 (Judge A + B + C) 对本轮进行评分                  │
│     5. Oracle reward → GRPO 策略更新                              │
│     6. MAP-Elites 策略库存储优秀载荷                               │
│     7. Oracle 权重通过课程学习自适应调整                            │
└─────────────────────────────────────────────────────────────────┘
```

### 数据流

```
红队 (GRPO 策略模型) ──生成──> 攻击载荷 (Payload)
        │
        v
注入引擎 (InjectionEngine) ──嵌入到──> 编程任务 (CodingTask) ──产出──> 注入任务 (InjectedTask)
        │
        v
蓝队 (Coding Agent) ──生成──> 代码 + 工具调用
        │
        v
混合评判器 (Hybrid Oracle) ──评估──> OracleReward (标量) ──反馈──> GRPO 更新
```

## 项目结构

```
david_and_goliath/
├── core/                          # 核心框架
│   ├── types.py                   # 全局数据类型与枚举
│   ├── injection_engine.py        # 载荷注入引擎 (5种载体)
│   ├── strategy_db.py             # MAP-Elites 策略数据库
│   └── co_evolution_controller.py # 协同进化主控制器
├── red_team/                      # 红队 (攻击方)
│   ├── grpo_trainer.py            # GRPO 训练器
│   ├── prompt_builder.py          # Few-shot Prompt 构建
│   └── models/lora_loader.py      # LoRA 适配器加载
├── blue_team/                     # 蓝队 (防御方)
│   ├── coding_agent.py            # LLM Coding Agent
│   ├── prompt_builder.py          # Agent Prompt 构建
│   ├── reflexion.py               # Reflexion 自我纠正循环
│   └── tools/                     # Agent 工具套件
│       ├── code_executor.py       # 沙盒代码执行
│       ├── static_analyzer.py     # 静态分析 (Bandit/Semgrep)
│       ├── unit_test_runner.py    # 单元测试运行
│       └── memory_retrieval.py    # 攻击模式记忆检索
├── hybrid_oracle/                 # 混合评判器
│   ├── oracle.py                  # 三Judge综合打分
│   ├── judge_a.py                 # Judge A: 静态分析评判
│   ├── judge_b.py                 # Judge B: 操纵检测评判
│   └── judge_c.py                 # Judge C: 载荷质量评判
├── evaluation/                    # 实验评估
│   ├── metrics.py                 # 核心指标计算
│   ├── cross_evaluator.py         # 交叉评估
│   ├── cascade_analyzer.py        # 级联失败分析
│   └── ood_evaluator.py           # OOD 泛化评估
├── infra/                         # 基础设施
│   ├── checkpoint.py              # 检查点存取
│   ├── logger.py                  # 日志配置
│   ├── ray_actors.py              # Ray 分布式 Actor
│   └── sandbox.py                 # 沙盒执行环境
├── scripts/                       # 脚本入口
│   ├── prepare_data.py            # 数据下载与预处理
│   ├── run_coevolution.py         # 协同进化训练入口
│   ├── run_cross_eval.py          # 交叉评估入口
│   └── run_offline_analysis.py    # 离线分析入口
├── configs/                       # 配置文件
│   ├── config.yaml                # 基础配置
│   ├── experiment/                # 实验配置 (coevo_8b, coevo_14b, ...)
│   ├── red_team/                  # 红队 GRPO 超参
│   ├── blue_team/                 # 蓝队工具配置
│   └── oracle/                    # 评判器权重与阈值
└── data/                          # 数据目录
    ├── coding_tasks/tasks.jsonl   # 421 编程任务
    ├── benign/humaneval_mbpp.jsonl # 421 参考解法
    └── red_seed_payloads.jsonl    # 2,069 种子攻击载荷
```

## 各模块详解

### 核心模块 (`core/`)

| 文件 | 功能 |
|------|------|
| `types.py` | 全局数据类型：`InjectionType`(4种攻击类型)、`StealthLevel`(3级隐蔽性)、`Carrier`(5种注入载体)，以及 `CodingTask`、`Payload`、`BlueTeamResponse`、`JudgeA/B/CResult`、`OracleReward`、`EpisodeResult`、`RoundRecord` 等全部数据类 |
| `injection_engine.py` | 载荷注入引擎，支持 5 种载体：自然语言、代码注释、Docstring、Markdown、多语言混合 |
| `strategy_db.py` | MAP-Elites 质量-多样性存档：4×3=12 个 niche，每个 niche 保留 top-K 载荷，支持偏向欠探索 niche 的采样策略 |
| `co_evolution_controller.py` | 主控制器：`setup() → run() [N轮] → finalize()`，支持检查点断点续训 |

### 红队 (`red_team/`)

| 文件 | 功能 |
|------|------|
| `grpo_trainer.py` | GRPO (Group Relative Policy Optimization) 训练器：生成一组载荷 → 计算组内相对优势 → 更新策略 |
| `prompt_builder.py` | 从 MAP-Elites 策略库采样 parents/inspirations，构建 few-shot prompt |
| `models/lora_loader.py` | LoRA 适配器加载工具 |

### 蓝队 (`blue_team/`)

| 文件 | 功能 |
|------|------|
| `coding_agent.py` | 基于 LLM 的 Coding Agent（当前为静态防御方，不参与协同进化） |
| `prompt_builder.py` | Agent 的系统/用户 prompt 构建 |
| `reflexion.py` | Reflexion 风格的自我纠正循环 |
| `tools/code_executor.py` | 沙盒化代码执行 |
| `tools/static_analyzer.py` | 静态分析（Bandit + Semgrep） |
| `tools/unit_test_runner.py` | 单元测试执行 |
| `tools/memory_retrieval.py` | 历史攻击模式检索 |

### 混合评判器 (`hybrid_oracle/`)

三 Judge 评判系统：

| Judge | 输入 | 评判内容 | 作用 |
|-------|------|----------|------|
| **Judge A** (`judge_a.py`) | 蓝队生成的代码 | 静态分析（Bandit + Semgrep）漏洞检测 | 攻击是否导致了有漏洞的代码？ |
| **Judge B** (`judge_b.py`) | 原始任务 + 载荷 + 蓝队输出 | LLM-as-Judge 操纵与偏离检测 | 蓝队是否被操纵了？ |
| **Judge C** (`judge_c.py`) | 红队载荷本身 | LLM-as-Judge 载荷质量/隐蔽性/创造性 | 载荷是否精心设计？（稀疏 reward 的 reward shaping） |

**`oracle.py`** — 将三 Judge 分数组合为标量 reward：
- 默认权重：`w_vulnerability`(0.30) + `w_manipulation`(0.30) + `w_quality`(0.20) + `w_diversity`(0.10) + `w_stealth`(0.10)
- 通过课程学习根据 ASR（攻击成功率）自适应调整权重

## 数据

| 文件 | 记录数 | 说明 |
|------|--------|------|
| `data/coding_tasks/tasks.jsonl` | 421 | HumanEval(164) + MBPP(257) 编程题 |
| `data/benign/humaneval_mbpp.jsonl` | 421 | 参考正确解法（用于假阳率评估） |
| `data/red_seed_payloads.jsonl` | 2,069 | 种子攻击载荷（从 normalized_pool 过滤的代码相关记录） |

## 关键概念

- **GRPO (Group Relative Policy Optimization)**：红队模型的 RL 训练方法 — 生成一组载荷，计算组内相对优势，更新策略
- **MAP-Elites 策略库**：质量-多样性存档，在 4×3 网格（攻击类型 × 隐蔽等级）中存储最优载荷，防止 GRPO 训练中的模式坍塌
- **课程学习 (Curriculum Learning)**：Oracle 权重随训练轮次自适应 — 早期强调载荷质量（reward shaping），后期转向漏洞/操纵（攻击成功率）
- **注入载体 (Injection Carriers)**：5 种将载荷嵌入编程任务的方式（自然语言、代码注释、docstring、markdown、多语言混合），增加攻击多样性维度

## 实现进度

| 模块 | 状态 | 备注 |
|------|------|------|
| `core/types.py` | ✅ 完成 | 全部数据类型、枚举、序列化 (453行) |
| `core/injection_engine.py` | ✅ 完成 | 5种载体注入策略 (294行) |
| `core/strategy_db.py` | ✅ 完成 | MAP-Elites 存储/采样/快照 (404行) |
| `core/co_evolution_controller.py` | ✅ 完成 | 完整编排循环，支持断点续训 (454行) |
| `scripts/prepare_data.py` | ✅ 完成 | HumanEval + MBPP 下载与格式化 (285行) |
| `data/generate_benchmark.py` | ✅ 完成 | 300 条 prompt injection 基准测试 (336行) |
| 数据文件 | ✅ 完成 | 421 任务 + 2,069 种子载荷已生成 |
| `red_team/grpo_trainer.py` | ⬜ 待实现 | 需要 GRPO 训练逻辑 |
| `red_team/prompt_builder.py` | ⬜ 待实现 | 需要 few-shot prompt 构建 |
| `red_team/models/lora_loader.py` | ⬜ 待实现 | 需要 LoRA 加载 |
| `blue_team/coding_agent.py` | ⬜ 待实现 | 需要 LLM agent 实现 |
| `blue_team/prompt_builder.py` | ⬜ 待实现 | 需要 prompt 构建 |
| `blue_team/reflexion.py` | ⬜ 待实现 | 需要 reflexion 循环 |
| `blue_team/tools/*` | ⬜ 待实现 | 4 个工具全部待实现 |
| `hybrid_oracle/judge_a.py` | ⬜ 待实现 | 需要 Bandit + Semgrep 集成 |
| `hybrid_oracle/judge_b.py` | ⬜ 待实现 | 需要 LLM-as-Judge prompt |
| `hybrid_oracle/judge_c.py` | ⬜ 待实现 | 需要 LLM-as-Judge prompt |
| `hybrid_oracle/oracle.py` | ⬜ 待实现 | 需要 reward 组合逻辑 |
| `evaluation/*` | ⬜ 待实现 | 4 个评估模块全部待实现 |
| `infra/*` | ⬜ 待实现 | 检查点/日志/Ray/沙盒全部待实现 |
| `scripts/run_*.py` | ⬜ 待实现 | 入口脚本全部待实现 |
| `configs/*.yaml` | 🔶 部分完成 | 基础配置已有，实验配置为占位 |

**整体进度**：核心框架层 + 数据管线 ✅ 已完成 (~1,900 行)，执行层模块（红队/蓝队/评判器/评估/基础设施）全部为空桩，待逐步实现。

## 快速开始

```bash
# 1. 准备数据（下载 HumanEval + MBPP）
python david_and_goliath/scripts/prepare_data.py

# 2. 运行协同进化训练（待实现）
python david_and_goliath/scripts/run_coevolution.py --config configs/experiment/coevo_8b.yaml

# 3. 交叉评估（待实现）
python david_and_goliath/scripts/run_cross_eval.py --config configs/experiment/coevo_8b.yaml
```
